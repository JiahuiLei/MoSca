# the GS controlling model for static scene

# given a colored pcl, construct GS models.

import sys, os, os.path as osp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import colorsys
import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F

from gs_utils.gs_optim_helper import *
import logging

from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from pytorch3d.ops import knn_points


def sph_order2nfeat(order):
    return (order + 1) ** 2


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


class StaticGaussian(nn.Module):
    def __init__(
        self,
        init_mean,  # N,3
        init_rgb,  # N,3
        init_s,  # N,3
        init_q=None,  # N,4
        init_o=None,  # N,1
        init_id=None,
        max_scale=0.1,  # use sigmoid activation, can't be too large
        min_scale=0.0,
        max_sph_order=0,
    ) -> None:
        super().__init__()
        self.op_update_exclude = []

        self.register_buffer("max_scale", torch.tensor(max_scale).squeeze())
        self.register_buffer("min_scale", torch.tensor(min_scale).squeeze())
        self.register_buffer("max_sph_order", torch.tensor(max_sph_order).squeeze())
        self._init_act(self.max_scale, self.min_scale)

        # * init the parameters
        self._xyz = nn.Parameter(torch.as_tensor(init_mean).float())
        if init_q is None:
            logging.warning("init_q is None, using default")
            init_q = torch.Tensor([1, 0, 0, 0]).float()[None].repeat(len(init_mean), 1)
        if init_o is None:
            logging.warning("init_o is None, using default")
            init_o = torch.Tensor([0.99]).float()[None].repeat(len(init_mean), 1)
        self._rotation = nn.Parameter(init_q)
        assert len(init_s) == len(init_mean)
        assert init_s.ndim == 2
        assert init_s.shape[1] == 3
        self._scaling = nn.Parameter(self.s_inv_act(init_s))
        o = self.o_inv_act(init_o)
        self._opacity = nn.Parameter(o)
        sph_rest_dim = 3 * (sph_order2nfeat(self.max_sph_order) - 1)
        self._features_dc = nn.Parameter(RGB2SH(init_rgb))
        self._features_rest = nn.Parameter(torch.zeros(self.N, sph_rest_dim))
        if init_id is None:
            init_id = torch.zeros(self.N, dtype=torch.int32)
        self.register_buffer("group_id", init_id)

        # * init states
        # warning, our code use N, instead of (N,1) as in GS code
        self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
        self.register_buffer("max_radii2D", torch.zeros(self.N).float())

        # ! dangerous flags
        # * for viz the cate color
        self.return_cate_colors_flag = False

        self.summary()
        return

    @classmethod
    def load_from_ckpt(cls, ckpt, device=torch.device("cuda:0")):
        init_mean = ckpt["_xyz"]
        init_rgb = ckpt["_features_dc"]
        init_s = ckpt["_scaling"]
        max_sph_order = ckpt["max_sph_order"]
        model = cls(
            init_mean=init_mean,
            init_rgb=init_rgb,
            init_s=init_s,
            max_sph_order=max_sph_order,
        )
        model.load_state_dict(ckpt, strict=True)
        # ! important, must re-init the activation functions
        logging.info(
            f"Resume: Max scale: {model.max_scale}, Min scale: {model.min_scale}, Max sph order: {model.max_sph_order}"
        )
        model._init_act(model.max_scale, model.min_scale)
        return model

    def summary(self):
        logging.info(f"StaticGaussian: {self.N/1000.0:.1f}K points")
        # logging.info number of parameters per pytorch sub module
        for name, param in self.named_parameters():
            logging.info(f"{name}, {param.numel()/1e6:.3f}M")
        logging.info("-" * 30)
        return

    def _init_act(self, max_s_value, min_s_value):
        max_s_value = max_s_value.item()
        min_s_value = min_s_value.item()

        def s_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)

        def s_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            x = torch.clamp(
                x, min=min_s_value + 1e-6, max=max_s_value - 1e-6
            )  # ! clamp
            y = (x - min_s_value) / (max_s_value - min_s_value) + 1e-5
            y = torch.clamp(y, min=1e-5, max=1 - 1e-5)
            y = torch.logit(y)
            if torch.isnan(y).any():
                logging.error(f"{x.min()}, {x.max()}")
                logging.error(f"{y.min()}, {y.max()}")
            assert not torch.isnan(
                y
            ).any(), f"{x.min()}, {x.max()}, {y.min()}, {y.max()}"
            return y

        def o_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.sigmoid(x)

        def o_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.logit(x)

        self.s_act = s_act
        self.s_inv_act = s_inv_act
        self.o_act = o_act
        self.o_inv_act = o_inv_act

        return

    @property
    def device(self):
        return self._xyz.device

    @property
    def N(self):
        try:  # for loading from file dummy init
            return len(self._xyz)
        except:
            return 0

    @property
    def get_x(self):
        return self._xyz

    @property
    def get_R(self):
        return quaternion_to_matrix(self._rotation)

    @property
    def get_o(self):
        return self.o_act(self._opacity)

    @property
    def get_s(self):
        return self.s_act(self._scaling)

    @property
    def get_c(self):
        return torch.cat([self._features_dc, self._features_rest], dim=-1)

    @property
    def get_group(self):
        assert len(self.group_id) == self.N
        return self.group_id

    @torch.no_grad()
    def get_cate_color(self, color_plate=None, perm=None):
        gs_group_id = self.get_group
        unique_grouping = torch.unique(gs_group_id).sort()[0]
        if not hasattr(self, "group_colors"):
            if color_plate is None:
                n_cate = len(self.group_id.unique())
                hue = np.linspace(0, 1, n_cate + 1)[:-1]
                color_plate = torch.Tensor(
                    [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue]
                ).to(self.device)
            self.group_colors = color_plate
            self.group_sphs = RGB2SH(self.group_colors)
        if perm is None:
            perm = torch.arange(len(unique_grouping))

        cate_sph = torch.zeros(self.N, 3).to(self.device)
        index_color_map = {}
        for ind in perm:
            gid = unique_grouping[ind]
            cate_sph[gs_group_id == gid] = self.group_sphs[ind].unsqueeze(0)
            index_color_map[gid] = self.group_colors[ind]
        return cate_sph, index_color_map

    def forward(self, active_sph_order=None):
        if active_sph_order is None:
            active_sph_order = self.max_sph_order
        else:
            assert active_sph_order <= self.max_sph_order
        xyz = self.get_x
        frame = self.get_R
        s = self.get_s
        o = self.get_o

        sph_dim = 3 * sph_order2nfeat(active_sph_order)
        sph = self.get_c
        sph = sph[:, :sph_dim]

        if self.return_cate_colors_flag:
            # logging.warning(f"VIZ purpose, return the cate-color")
            cate_sph, _ = self.get_cate_color()
            sph = torch.zeros_like(sph)
            sph[..., :3] = cate_sph  # zero pad

        return xyz, frame, s, o, sph

    def get_optimizable_list(
        self,
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest=None,
    ):
        lr_sph_rest = lr_sph / 20 if lr_sph_rest is None else lr_sph_rest
        l = [
            {"params": [self._xyz], "lr": lr_p, "name": "xyz"},
            {"params": [self._opacity], "lr": lr_o, "name": "opacity"},
            {"params": [self._scaling], "lr": lr_s, "name": "scaling"},
            {"params": [self._rotation], "lr": lr_q, "name": "rotation"},
            {"params": [self._features_dc], "lr": lr_sph, "name": "f_dc"},
            {"params": [self._features_rest], "lr": lr_sph_rest, "name": "f_rest"},
        ]
        return l

    ######################################################################
    # * Gaussian Control
    ######################################################################

    def record_xyz_grad_radii(self, viewspace_point_tensor_grad, radii, update_filter):
        # Record the gradient norm, invariant across different poses
        assert len(viewspace_point_tensor_grad) == self.N
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor_grad[update_filter, :2], dim=-1, keepdim=False
        )
        self.xyz_gradient_denom[update_filter] += 1
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter], radii[update_filter]
        )
        return

    def _densification_postprocess(
        self,
        optimizer,
        new_xyz,
        new_r,
        new_s,
        new_o,
        new_sph_dc,
        new_sph_rest,
        new_group_id,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_sph_dc,
            "f_rest": new_sph_rest,
            "opacity": new_o,
            "scaling": new_s,
            "rotation": new_r,
        }
        d = {k: v for k, v in d.items() if v is not None}

        # First cat to optimizer and then return to self
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]

        self.xyz_gradient_accum = torch.zeros(self._xyz.shape[0], device=self.device)
        self.xyz_gradient_denom = torch.zeros(self._xyz.shape[0], device=self.device)
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros_like(new_xyz[:, 0])], dim=0
        )

        self.group_id = torch.cat([self.group_id, new_group_id], dim=0)
        return

    def clean_gs_control_record(self):
        self.xyz_gradient_accum = torch.zeros_like(self._xyz[:, 0])
        self.xyz_gradient_denom = torch.zeros_like(self._xyz[:, 0])
        self.max_radii2D = torch.zeros_like(self.max_radii2D)

    def append_gs(
        self,
        optimizer,
        new_mu,
        new_fr,
        new_scale,
        new_opacity,
        new_sph,
        new_group_id=None,
    ):
        # ! the inputs are all after activation
        logging.info(f"Append {len(new_mu)} points")
        new_xyz = new_mu.detach().clone()
        new_rotation = matrix_to_quaternion(new_fr.detach().clone())
        new_scale = torch.clamp(new_scale, max=self.max_scale, min=self.min_scale)
        new_scaling = self.s_inv_act(new_scale.detach().clone())
        new_opacity = torch.clamp(new_opacity, max=1.0, min=0.0)
        new_opacity = self.o_inv_act(new_opacity.detach().clone())
        new_features_dc = new_sph[:, :3].detach().clone()
        new_features_rest = new_sph[:, 3:].detach().clone()
        if new_group_id is None:
            new_group_id = torch.zeros_like(new_mu[:, 0]).to(self.group_id)

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacity,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_group_id=new_group_id,
        )
        return

    def _densify_and_clone(self, optimizer, grad_norm, grad_threshold, scale_th):
        # Extract points that satisfy the gradient condition
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device=self.device)
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_s, dim=1).values <= scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0

        new_xyz = self._xyz[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_group_id = self.group_id[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_group_id=new_group_id,
        )

        return len(new_xyz)

    def _densify_and_split(
        self,
        optimizer,
        grad_norm,
        grad_threshold,
        scale_th,
        N=2,
    ):
        # Extract points that satisfy the gradient condition
        _scaling = self.get_s
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device=self.device)
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(_scaling, dim=1).values > scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0

        stds = _scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = _scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scaling = torch.clamp(new_scaling, max=self.max_scale, min=self.min_scale)
        new_scaling = self.s_inv_act(new_scaling)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
        new_group_id = self.group_id[selected_pts_mask].repeat(N)

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_group_id=new_group_id,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device=self.device, dtype=bool
                ),
            )
        )
        self._prune_points(optimizer, prune_filter)
        return len(new_xyz)

    def densify(
        self,
        optimizer,
        max_grad,
        percent_dense,
        extent,
        max_grad_node=None,
        node_ctrl_flag=True,
        verbose=True,
    ):
        grads = self.xyz_gradient_accum / self.xyz_gradient_denom
        grads[grads.isnan()] = 0.0

        # n_clone = self._densify_and_clone(optimizer, grads, max_grad)
        n_clone = self._densify_and_clone(
            optimizer, grads, max_grad, percent_dense * extent
        )
        n_split = self._densify_and_split(
            optimizer, grads, max_grad, percent_dense * extent, N=2
        )

        if verbose:
            logging.info(f"Densify: Clone[+] {n_clone}, Split[+] {n_split}")
            # logging.info(f"Densify: Clone[+] {n_clone}")
        # torch.cuda.empty_cache()
        return

    def prune_points(
        self,
        optimizer,
        min_opacity,
        max_screen_size,
        verbose=True,
    ):
        opacity = self.o_act(self._opacity)
        prune_mask = (opacity < min_opacity).squeeze()
        logging.info(f"opacity_pruning {prune_mask.sum()}")
        if max_screen_size:  # if a point is too large
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            logging.info(f"radii2D_pruning {big_points_vs.sum()}")
            # * reset the maxRadii
            self.max_radii2D = torch.zeros_like(self.max_radii2D)
        self._prune_points(optimizer, prune_mask)
        if verbose:
            logging.info(f"Prune: {prune_mask.sum()}")

    def _prune_points(self, optimizer, mask):
        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer(
            optimizer,
            valid_points_mask,
            exclude_names=self.op_update_exclude,
        )

        self._xyz = optimizable_tensors["xyz"]
        if getattr(self, "color_memory", None) is None:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_denom = self.xyz_gradient_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # torch.cuda.empty_cache()
        self.group_id = self.group_id[valid_points_mask]
        return

    def reset_opacity(self, optimizer, value=0.01, verbose=True):
        opacities_new = self.o_inv_act(
            torch.min(self.o_act(self._opacity), torch.ones_like(self._opacity) * value)
        )
        optimizable_tensors = replace_tensor_to_optimizer(
            optimizer, opacities_new, "opacity"
        )
        if verbose:
            logging.info(f"Reset opacity to {value}")
        self._opacity = optimizable_tensors["opacity"]

    def load(self, ckpt):
        # because N changed, have to re-init the buffers
        self._xyz = nn.Parameter(torch.as_tensor(ckpt["_xyz"], dtype=torch.float32))

        self._features_dc = nn.Parameter(
            torch.as_tensor(ckpt["_features_dc"], dtype=torch.float32)
        )
        self._features_rest = nn.Parameter(
            torch.as_tensor(ckpt["_features_rest"], dtype=torch.float32)
        )
        self._opacity = nn.Parameter(
            torch.as_tensor(ckpt["_opacity"], dtype=torch.float32)
        )
        self._scaling = nn.Parameter(
            torch.as_tensor(ckpt["_scaling"], dtype=torch.float32)
        )
        self._rotation = nn.Parameter(
            torch.as_tensor(ckpt["_rotation"], dtype=torch.float32)
        )
        self.xyz_gradient_accum = torch.as_tensor(
            ckpt["xyz_gradient_accum"], dtype=torch.float32
        )
        self.xyz_gradient_denom = torch.as_tensor(
            ckpt["xyz_gradient_denom"], dtype=torch.int64
        )
        self.max_radii2D = torch.as_tensor(ckpt["max_radii2D"], dtype=torch.float32)
        # load others
        self.load_state_dict(ckpt, strict=True)
        # this is critical, reinit the funcs
        self._init_act(self.max_scale, self.min_scale)
        return
