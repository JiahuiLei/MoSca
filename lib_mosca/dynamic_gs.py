import sys, os, os.path as osp
import torch_geometric.nn.pool as pyg_pool
import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F
import logging
import time
from matplotlib import pyplot as plt
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from pytorch3d.ops import knn_points
import open3d as o3d
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gs_utils.gs_optim_helper import *
from mosca import MoSca, _compute_curve_topo_dist_, resample_curve, DQ_EPS
import colorsys


# TODO: flexiblly swtich between GS mu parameterization, for later efficiency, have to directly save the location.


class DynSCFGaussian(nn.Module):
    #################################################################
    # * Trace buffers and parameters
    #################################################################
    # * buffers
    # max_scale
    # min_sacle
    # max_sph_order

    # attach_ind
    # ref_time

    # xyz_gradient_accum
    # xyz_gradient_denom
    # max_radii2D

    # * parameters
    # _xyz
    # _rotation
    # _scaling
    # _opacity
    # _features_dc
    # _features_rest
    # _skinning_weight
    # _dynamic_logit
    #################################################################

    # ! note: use_dyn_o = False so permanently remove

    def __init__(
        self,
        scf: MoSca = None,
        max_scale=0.1,  # use sigmoid activation, can't be too large
        min_scale=0.0,
        max_sph_order=0,
        device=torch.device("cuda:0"),
        max_node_num=100000, #16384,
        leaf_local_flag=True,  # save nodes to local nodes
        init_N=None,
        node_ctrl=True,
        nn_fusion=-1,  # if -1 use all gs, otherwise, find the nearest n time frame to warp
        #
        dyn_o_flag=False,
        #
        min_num_gs=32,
    ) -> None:
        # * Init only init the pre-computed SCF, later will append leaves!
        super().__init__()
        self.op_update_exclude = [
            "node_xyz",
            "node_rotation",
            "node_sigma",
            "sk_rotation",
        ]

        self.fast_inference_flag = False
        self.min_num_gs = min_num_gs

        self.register_buffer("nn_fusion", torch.tensor(nn_fusion))
        logging.info(f"ED Model use default {nn_fusion} nearest frame fusion")
        self.register_buffer("dyn_o_flag", torch.tensor(dyn_o_flag).bool())
        if dyn_o_flag:
            logging.warning(
                f"Dynamic GS use Dyn_O, Far away points will have ID motion, the SE(3) field has an ambient motion I"
            )

        self.scf = scf

        self.node_ctrl = node_ctrl
        if self.node_ctrl:
            logging.info(
                f"Node control is enabled! (max number of nodes {max_node_num})"
            )
        # self.max_node_num = max_node_num
        self.register_buffer("max_node_num", torch.tensor(max_node_num))

        self.register_buffer("w_correction_flag", torch.tensor([0]).squeeze().bool())
        self.register_buffer(
            "leaf_local_flag", torch.tensor([leaf_local_flag]).squeeze().bool()
        )

        # * prepare activation
        self.register_buffer("max_scale", torch.tensor(max_scale).to(self.device))
        self.register_buffer("min_scale", torch.tensor(min_scale).to(self.device))
        self.register_buffer(
            "max_sph_order", torch.tensor(max_sph_order).to(self.device)
        )
        self._init_act(self.max_scale, self.min_scale)

        # * Init the empty leaf attr
        if init_N is None:
            init_N = self.N
        self._xyz = nn.Parameter(torch.zeros(init_N, 3))  # N,3
        self._rotation = nn.Parameter(torch.zeros(init_N, 4))  # N,4
        self._scaling = nn.Parameter(torch.zeros(init_N, 3))  # N,3
        self._opacity = nn.Parameter(torch.zeros(init_N, 1))  # N,1
        sph_rest_dim = 3 * (sph_order2nfeat(self.max_sph_order) - 1)
        self._features_dc = nn.Parameter(torch.zeros(init_N, 3))  # N,3
        self._features_rest = nn.Parameter(torch.zeros(init_N, sph_rest_dim))
        self._skinning_weight = nn.Parameter(torch.zeros(init_N, self.scf.skinning_k))
        self._dynamic_logit = nn.Parameter(self.o_inv_act(torch.ones(init_N, 1) * 0.99))
        # * leaf important status
        self.register_buffer("attach_ind", torch.zeros(init_N).long())  # N
        self.register_buffer("ref_time", torch.zeros(init_N).long())  # N
        # * init states
        self.register_buffer("xyz_gradient_accum", torch.zeros(init_N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(init_N).long())
        self.register_buffer("max_radii2D", torch.zeros(init_N).float())

        # * for tracing the correspondence gradient
        self.register_buffer("corr_gradient_accum", torch.zeros(init_N).float())
        self.register_buffer("corr_gradient_denom", torch.zeros(init_N).long())

        self.to(self.device)
        self.summary(lite=True)

        # ! dangerous flags
        # * for viz the cate color
        self.return_cate_colors_flag = False

        return

    @classmethod
    def load_from_ckpt(cls, ckpt, device=torch.device("cuda:0")):
        # first recover the
        scf_sub_ckpt = {k[4:]: v for k, v in ckpt.items() if k.startswith("scf.")}
        scf = MoSca.load_from_ckpt(scf_sub_ckpt, device=device)
        if "scf.mlevle_detach_nn_flag" not in ckpt.keys():
            # old ckpt
            ckpt["scf.mlevel_detach_nn_flag"] = torch.tensor(True)
        if "scf.mlevel_detach_self_flag" not in ckpt.keys():
            # old ckpt
            ckpt["scf.mlevel_detach_self_flag"] = torch.tensor(False)
        if "scf.w_corr_maintain_sum_flag" not in ckpt.keys():
            # old ckpt
            ckpt["scf.w_corr_maintain_sum_flag"] = torch.tensor(False)
        model = cls(
            scf=scf,
            device=device,
            init_N=ckpt["_xyz"].shape[0],
            max_sph_order=ckpt["max_sph_order"],
        )
        if "corr_gradient_accum" not in ckpt.keys():
            logging.info(f"Old ckpt, add corr_gradient_accum")
            ckpt["corr_gradient_accum"] = ckpt["xyz_gradient_accum"].clone() * 0.0
            ckpt["corr_gradient_denom"] = ckpt["xyz_gradient_denom"].clone() * 0
        if "max_node_num" not in ckpt.keys():
            logging.info(f"Old ckpt, add max_node_num")
            ckpt["max_node_num"] = torch.tensor([100000])
        model.load_state_dict(ckpt, strict=True)
        model.summary()
        return model

    def __load_from_file_init__(self, load_fn):
        self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
        self.register_buffer("max_radii2D", torch.zeros(self.N).float())
        self.register_buffer("attach_ind", torch.zeros(self.N))
        self.register_buffer("ref_time", torch.zeros(self.N))  # N
        # * init the parameters from file
        logging.info(f"Loading dynamic model from {load_fn}")
        self.load(torch.load(load_fn))
        self.forward(0)
        self.summary()
        return

    def summary(self, lite=False):
        logging.info(
            f"DenseDynGaussian: {self.N/1000.0:.1f}K points; {self.M} Nodes; K={self.scf.skinning_k}; and {self.T} time step"
        )
        if lite:
            return
        # logging.info number of parameters per pytorch sub module
        for name, param in self.named_parameters():
            logging.info(f"{name}, {param.numel()/1e6:.3f}M")
        logging.info("-" * 30)
        logging.info(
            f"ED MODEL DEBUG: w_corr: max={self._skinning_weight.max()}, min={self._skinning_weight.min()}, mean={self._skinning_weight.mean()}"
        )
        logging.info(f"ED MODEL DEBUG: use nn_fusion={self.nn_fusion}")
        return

    def _init_act(self, max_s_value, min_s_value):
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
        return self.scf.device

    @property
    def N(self):
        try:  # for loading from file dummy init
            return self._xyz.shape[0]
        except:
            return 0

    @property
    def M(self):
        return self.scf.M

    @property
    def T(self):
        return self.scf.T

    @property
    def get_t_range(self):
        # ! this is not useful in current version, but to be compatible with old, maintain this
        return 0, self.T - 1

    def get_tlist_ind(self, t):
        # ! this is not useful in current version, but to be compatible with old, maintain this
        assert t < self.T
        return t

    @property
    def get_o(self):
        return self.o_act(self._opacity)

    @property
    def get_group(self):
        # fetch the group id from the attached to nearest node
        group_id = self.scf._node_grouping[self.attach_ind]
        return group_id

    @torch.no_grad()
    def get_cate_color(self, color_plate=None, perm=None):
        if not hasattr(self, "group_colors"):
            if color_plate is None:
                n_cate = len(self.scf.unique_grouping)
                hue = np.linspace(0, 1, n_cate + 1)[:-1]
                color_plate = torch.Tensor(
                    [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue]
                ).to(self.device)
            self.group_colors = color_plate
            self.group_sphs = RGB2SH(self.group_colors)
        if perm is None:
            perm = torch.arange(len(self.group_colors))
        gs_group_id = self.get_group
        cate_sph = torch.zeros(self.N, 3).to(self.device)
        index_color_map = {}
        for ind in perm:
            gid = self.scf.unique_grouping[ind]
            cate_sph[gs_group_id == gid] = self.group_sphs[ind].unsqueeze(0)
            index_color_map[gid] = self.group_colors[ind]
        return cate_sph, index_color_map

    # @property
    # def get_d(self):
    #     return self.o_act(self._dynamic_logit)

    @property
    def get_s(self):
        return self.s_act(self._scaling)

    @property
    def get_c(self):
        return torch.cat([self._features_dc, self._features_rest], dim=-1)

    def get_xyz(self):
        if self.leaf_local_flag:
            nn_ref_node_xyz, sk_ref_node_quat = self.scf.get_async_knns(
                self.ref_time, self.attach_ind[:, None]
            )
            nn_ref_node_xyz = nn_ref_node_xyz.squeeze(1)
            nn_ref_node_R_wi = q2R(sk_ref_node_quat.squeeze(1))
            return (
                torch.einsum("nij,nj->ni", nn_ref_node_R_wi, self._xyz)
                + nn_ref_node_xyz
            )
        else:
            return self._xyz

    def get_R_mtx(self):
        if self.leaf_local_flag:
            _, sk_ref_node_quat = self.scf.get_async_knns(
                self.ref_time, self.attach_ind[:, None]
            )
            nn_ref_node_R_wi = q2R(sk_ref_node_quat.squeeze(1))
            return torch.einsum("nij,njk->nik", nn_ref_node_R_wi, q2R(self._rotation))
        else:
            return q2R(self._rotation)

    def set_surface_deform(self):
        # * different to using RBF field approximate the deformation field (more flexible when changing the scf topology), set the deformation to surface model, the skinning is saved on each GS
        logging.info(f"ED Model convert to surface mode")
        self.w_correction_flag = torch.tensor(True).to(self.device)
        self.scf.fixed_topology_flag = torch.tensor(True).to(self.device)
        return

    def set_inference_mode(self):
        # bake the buffer and enable fast inference FPS

        self.eval()

        baked_query_xyz = self.get_xyz()
        baked_query_dir = self.get_R_mtx()
        self.register_buffer("baked_query_xyz", baked_query_xyz)
        self.register_buffer("baked_query_dir", baked_query_dir)

        (
            baked_sk_ind,
            baked_sk_w,
            sk_w_sum,
            baked_sk_ref_node_xyz,
            baked_sk_ref_node_quat,
        ) = self.scf.get_skinning_weights(
            query_xyz=baked_query_xyz,
            query_t=self.ref_time,
            attach_ind=self.attach_ind,
            skinning_weight_correction=(
                self._skinning_weight if self.w_correction_flag else None
            ),
        )
        if self.dyn_o_flag:
            baked_dyn_o = torch.clamp(sk_w_sum, min=0.0, max=1.0)
        else:
            baked_dyn_o = torch.ones_like(sk_w_sum) * (1.0 - DQ_EPS)
        self.register_buffer("baked_sk_ind", baked_sk_ind)
        self.register_buffer("baked_sk_w", baked_sk_w)
        self.register_buffer("baked_sk_ref_node_xyz", baked_sk_ref_node_xyz)
        self.register_buffer("baked_sk_ref_node_quat", baked_sk_ref_node_quat)
        self.register_buffer("baked_dyn_o", baked_dyn_o)
        self.fast_inference_flag = True
        logging.warning((f"ED Model convert to inference mode"))
        return

    def forward(self, t: int, active_sph_order=None, nn_fusion=None):
        assert t < self.T, "t is out of range!"
        if active_sph_order is None:
            active_sph_order = int(self.max_sph_order)
        else:
            assert active_sph_order <= self.max_sph_order

        s = self.get_s
        o = self.get_o
        sph_dim = 3 * sph_order2nfeat(active_sph_order)
        sph = self.get_c[:, :sph_dim]

        if self.fast_inference_flag:
            mu_live, fr_live = self.scf.fast_warp(
                target_tid=t,
                # all below are baked
                sk_ind=self.baked_sk_ind,
                sk_w=self.baked_sk_w,
                sk_ref_node_xyz=self.baked_sk_ref_node_xyz,
                sk_ref_node_quat=self.baked_sk_ref_node_quat,
                dyn_o=self.baked_dyn_o,
                query_xyz=self.baked_query_xyz,
                query_dir=self.baked_query_dir,
            )
        else:
            mu_live, fr_live = self.scf.warp(
                attach_node_ind=self.attach_ind,
                query_xyz=self.get_xyz(),
                query_dir=self.get_R_mtx(),
                query_tid=self.ref_time,
                target_tid=t,
                skinning_w_corr=(
                    self._skinning_weight if self.w_correction_flag else None
                ),
                dyn_o_flag=self.dyn_o_flag,
            )

        if nn_fusion is None:
            nn_fusion = self.nn_fusion.item()
        if nn_fusion > 0:
            # filter the gaussian by nn mask
            with torch.no_grad():
                supporting_t = torch.unique(self.ref_time)
                # find nearest supporting t
                t_dist = (supporting_t - t).abs()
                supporting_t = supporting_t[torch.argsort(t_dist)[:nn_fusion]]
                # get the nn mask
                filter_mask = (self.ref_time[:, None] == supporting_t[None]).any(dim=1)
            if not filter_mask.any():
                logging.warning(f"nn_fusion has no gs supporting, dummy mask!")
                # ! set the first two gs to be visible
                filter_mask[:2] = True
            # mu_live = mu_live[filter_mask]
            # fr_live = fr_live[filter_mask]
            # s = s[filter_mask]
            # o = o[filter_mask]
            # sph = sph[filter_mask]
            # ! do this by mask the opacity to zero
            o = o * filter_mask[:, None].float()  # ! mask the opacity to zero
            assert (
                not self.return_cate_colors_flag
            ), " not support cate color in nn fusion"
        if self.return_cate_colors_flag:
            # logging.warning(f"VIZ purpose, return the cate-color")
            cate_sph, _ = self.get_cate_color()
            sph = torch.zeros_like(sph)
            sph[..., :3] = cate_sph  # zero pad
        return mu_live, fr_live, s, o, sph

    def local_gs_warp(
        self,
        query_xyz: torch.Tensor,
        src_t: int,
        dst_t: int,
        K_candidates=16,
        query_frame=None,
        query_group=None,
    ):
        assert query_xyz.ndim == 2 and query_xyz.shape[1] == 3
        # use nearest gs local frame to warp, not the coarse scaffolds
        src_gs5 = self.forward(src_t)
        dst_gs5 = self.forward(dst_t)

        # find nearest K gs in the src frame and compute the influence!
        _, gs_id, _ = knn_points(query_xyz[None], src_gs5[0][None], K=K_candidates)
        gs_id = gs_id.squeeze(0)  # Q,K
        gs_center_w = src_gs5[0][gs_id]  # Q,K,3
        gs_fr_T_wg = src_gs5[1][gs_id]  # Q,K,3,3
        local_coord = query_xyz[:, None] - gs_center_w  # Q,K,3
        local_coord = torch.einsum("qkji,qkj->qki", gs_fr_T_wg, local_coord)
        scaling = src_gs5[2][gs_id]  # Q,K,3
        local_coord_normalzied = local_coord / (scaling + 1e-6)
        # directly check the nearest
        dist = local_coord_normalzied.norm(dim=-1)
        nearest_knn_id = torch.argmin(dist, dim=1)
        nearest_gs_id = torch.gather(gs_id, 1, nearest_knn_id[:, None]).squeeze(1)

        local_coord = torch.gather(
            local_coord, 1, nearest_knn_id[:, None, None].expand(-1, -1, 3)
        )
        local_coord = local_coord.squeeze(1)  # Q,3

        dst_gs_fr_T_wg = dst_gs5[1][nearest_gs_id]  # Q,3,3
        dst_gs_center = dst_gs5[0][nearest_gs_id]  # Q,3
        new_coord = (
            torch.einsum("qij,qj->qi", dst_gs_fr_T_wg, local_coord) + dst_gs_center
        )

        if query_frame is not None:
            src_gs_fr_T_wg = src_gs5[1][nearest_gs_id]  # Q,3,3
            local_query_frame = torch.einsum(
                "qji,qjk->qik", src_gs_fr_T_wg, query_frame
            )
            new_frame = torch.einsum("qij,qjk->qik", dst_gs_fr_T_wg, local_query_frame)
            return new_coord, new_frame
        else:
            return new_coord, None

    def get_optimizable_list(
        self,
        # leaf
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest=None,
        lr_dyn=0.0,
        lr_w=0.0,
        # node
        lr_np=0.0001,
        lr_nq=0.0001,
        lr_nsig=0.00001,
    ):
        l = []
        if lr_p is not None:
            l.append({"params": [self._xyz], "lr": lr_p, "name": "xyz"})
        if lr_q is not None:
            l.append({"params": [self._rotation], "lr": lr_q, "name": "rotation"})
        if lr_s is not None:
            l.append({"params": [self._scaling], "lr": lr_s, "name": "scaling"})
        if lr_o is not None:
            l.append({"params": [self._opacity], "lr": lr_o, "name": "opacity"})
        if lr_sph is not None:
            if lr_sph_rest is None:
                lr_sph_rest = lr_sph / 20
            l.append({"params": [self._features_dc], "lr": lr_sph, "name": "f_dc"})
            l.append(
                {"params": [self._features_rest], "lr": lr_sph_rest, "name": "f_rest"}
            )

        if lr_dyn is not None:
            l.append(
                {"params": [self._dynamic_logit], "lr": lr_dyn, "name": "dyn_logit"}
            )
        if lr_w is not None:
            # logging.warning("lr_w is not supported yet!")
            l.append(
                {"params": [self._skinning_weight], "lr": lr_w, "name": "skinning_w"}
            )
        l = l + self.scf.get_optimizable_list(lr_np=lr_np, lr_nq=lr_nq, lr_nsig=lr_nsig)
        return l

    ######################################################################
    # * Model Grow
    ######################################################################

    def __check_uncovered_mu__(self, mu_w, tid):
        node_xyz_live = self.scf._node_xyz[self.get_tlist_ind(tid)]
        dist_sq, _, _ = knn_points(mu_w[None], node_xyz_live[None], K=1)
        covered_by_existing_node = dist_sq[0].squeeze(-1) < self.scf.spatial_unit**2
        uncovered_mu_w = mu_w[~covered_by_existing_node]
        return uncovered_mu_w

    @torch.no_grad()
    def append_new_gs(self, optimizer, tid, mu_w, quat_w, scales, opacity, rgb):
        # ! note, this function never grow new nodes for now
        # * Append leaves
        if scales.ndim == 1:
            scales = scales[:, None].expand(-1, 3)
        new_s_logit = self.s_inv_act(scales)
        assert opacity.ndim == 2 and opacity.shape[1] == 1
        new_o_logit = self.o_inv_act(opacity)
        new_feat_dc = RGB2SH(rgb)
        new_feat_rest = torch.zeros(len(scales), self._features_rest.shape[1]).to(
            self.device
        )

        # first update the knn ind
        _, attach_ind, _ = knn_points(mu_w[None], self.scf._node_xyz[tid][None], K=1)
        self.attach_ind = torch.cat([self.attach_ind, attach_ind[0, :, 0]], dim=0)
        self.ref_time = torch.cat(
            [self.ref_time, torch.ones_like(attach_ind[0, :, 0]) * tid], dim=0
        )

        # if use local leaf storage, have to convert to local!
        if self.leaf_local_flag:
            attach_node_xyz = self.scf._node_xyz[tid][attach_ind[0, :, 0]]
            attach_node_R_wi = q2R(self.scf._node_rotation[tid][attach_ind[0, :, 0]])
            mu_save = torch.einsum(
                "nji,nj->ni", attach_node_R_wi, mu_w - attach_node_xyz
            )
            R_new = torch.einsum("nji,njk->nik", attach_node_R_wi, q2R(quat_w))
            quat_save = matrix_to_quaternion(R_new)
        else:
            mu_save, quat_save = mu_w, quat_w

        # finally update the parameters
        self._densification_postprocess(
            optimizer,
            new_xyz=mu_save,  # ! store the position in live frame
            new_r=quat_save,
            new_s_logit=new_s_logit,
            new_o_logit=new_o_logit,
            new_sph_dc=new_feat_dc,
            new_sph_rest=new_feat_rest,
            new_skinning_w=torch.zeros(len(mu_w), self.scf.skinning_k).to(mu_w),
            new_dyn_logit=self.o_inv_act(torch.ones_like(scales[:, :1]) * 0.99),
        )
        # self.summary(lite=True)
        return

    ######################################################################
    # * Node Control
    ######################################################################

    @torch.no_grad()
    def append_new_node_and_gs(
        self,
        optimizer,
        tid,
        mu_w,
        quat_w,
        scales,
        opacity,
        rgb,
    ):
        # ! note, this function never grow new nodes for now

        # * Analysis and append the topology
        # check whether the new leaves covered by existing nodes

        # ! all uncovered
        new_node_xyz_l = subsample_vtx(mu_w, self.scf.spatial_unit)
        new_node_quat_l = (
            torch.tensor([1.0, 0.0, 0.0, 0.0])
            .to(self.device)[None]
            .expand(len(new_node_xyz_l), -1)
        )
        logging.info(f"Grow [{len(new_node_xyz_l)}] new nodes from {len(mu_w)} leaves")
        old_M = self.scf.M
        succ = self.scf.append_nodes_pnt(
            optimizer,
            new_node_xyz_l,
            new_node_quat_l,
            tid,
        )
        if not succ:
            logging.warning(f"ED model grow failed, skip the append!")
            return

        if scales.ndim == 1:
            scales = scales[:, None].expand(-1, 3)
        new_s_logit = self.s_inv_act(scales)
        assert opacity.ndim == 2 and opacity.shape[1] == 1
        new_o_logit = self.o_inv_act(opacity)
        new_feat_dc = RGB2SH(rgb)
        new_feat_rest = torch.zeros(len(scales), self._features_rest.shape[1]).to(
            self.device
        )

        # first update the knn ind
        # ! always attach to newly added node!
        _, attach_ind, _ = knn_points(
            mu_w[None], self.scf._node_xyz[tid][old_M:][None], K=1
        )
        attach_ind = attach_ind + old_M
        self.attach_ind = torch.cat([self.attach_ind, attach_ind[0, :, 0]], dim=0)
        self.ref_time = torch.cat(
            [self.ref_time, torch.ones_like(attach_ind[0, :, 0]) * tid], dim=0
        )

        # if use local leaf storage, have to convert to local!
        if self.leaf_local_flag:
            attach_node_xyz = self.scf._node_xyz[tid][attach_ind[0, :, 0]]
            attach_node_R_wi = q2R(self.scf._node_rotation[tid][attach_ind[0, :, 0]])
            mu_save = torch.einsum(
                "nji,nj->ni", attach_node_R_wi, mu_w - attach_node_xyz
            )
            R_new = torch.einsum("nji,njk->nik", attach_node_R_wi, q2R(quat_w))
            quat_save = matrix_to_quaternion(R_new)
        else:
            mu_save, quat_save = mu_w, quat_w

        # finally update the parameters
        old_N = self.N
        self._densification_postprocess(
            optimizer,
            new_xyz=mu_save,  # ! store the position in live frame
            new_r=quat_save,
            new_s_logit=new_s_logit,
            new_o_logit=new_o_logit,
            new_sph_dc=new_feat_dc,
            new_sph_rest=new_feat_rest,
            new_skinning_w=torch.zeros(len(mu_w), self.scf.skinning_k).to(mu_w),
            new_dyn_logit=self.o_inv_act(torch.ones_like(scales[:, :1]) * 0.99),
        )
        # self.summary(lite=True)
        # check_xyz = self.forward(tid)[0]
        # error = (check_xyz[old_N:] - mu_w).norm(dim=-1).max()
        # logging.info(f"error append check error {error:.3f}")
        # verify the append xyz is still there
        return

    @torch.no_grad()
    def prune_nodes(self, optimizer, prune_sk_th=0.02, viz_fn=None):
        # if a node is not carrying leaves, and the effect to all neighbors are small. then can prune it; and also update the knn skinning weight, during this update, also have to be careful about the inner scf-knn-ind for scf, only replace some where!!!

        acc_w = self.get_node_sinning_w_acc("max")
        # viz
        if viz_fn is not None:
            fig = plt.figure(figsize=(10, 5))
            plt.hist(acc_w.cpu().numpy(), bins=100)
            plt.plot([prune_sk_th, prune_sk_th], [0, 100], "r--")
            plt.title(f"Node Max supporting sk-w hist")
            plt.savefig(f"{viz_fn}prune_sk_hist.jpg")
            plt.close()

        prune_mask_sk = acc_w < prune_sk_th  # if prune, true

        # also check whether this node carries some leaves
        supporting_node_id = torch.unique(self.attach_ind)
        prune_mask_carry = torch.ones(self.M, device=self.device).bool()
        prune_mask_carry[supporting_node_id] = False

        node_prune_mask = prune_mask_sk & prune_mask_carry
        logging.info(
            f"Prune {node_prune_mask.sum()} nodes (max_sk<th={prune_sk_th}) with carrying check ({node_prune_mask.float().mean()*100.0:.3f}%)"
        )

        prune_M = node_prune_mask.sum().item()
        if prune_M == 0:
            return

        # first remove the leaves
        # ! actually this is not used in our case for now
        leaf_pruning_mask = (
            self.attach_ind[:, None]
            == torch.arange(self.M, device=self.device)[None, node_prune_mask]
        ).any(-1)
        if leaf_pruning_mask.any():
            self._prune_points(optimizer, leaf_pruning_mask)

        if self.w_correction_flag:
            # identify the sk corr that related to the old node
            sk_corr_affect_mask = node_prune_mask[
                self.scf.topo_knn_ind[self.attach_ind]
            ]
            logging.warning(
                f"Prune under surface mode, check {sk_corr_affect_mask.sum()}({sk_corr_affect_mask.float().mean()*100:.3f}%) sk_corr to be updated"
            )
            # ! later make these position sk to be zero

        # then update the attach ind
        new_M = self.M - prune_M
        ind_convert = torch.ones(self.M, device=self.device).long() * -1
        ind_convert[~node_prune_mask] = torch.arange(new_M, device=self.device)
        self.attach_ind = ind_convert[self.attach_ind]

        # finally remove the nodes
        self.scf.remove_nodes(optimizer, node_prune_mask)

        # now update the sk corr again, make sure the updated = 0.0
        if self.w_correction_flag:

            _, sk_w, sk_w_sum, _, _ = self.scf.get_skinning_weights(
                query_xyz=self.get_xyz(),
                query_t=self.ref_time,
                attach_ind=self.attach_ind,
                # skinning_weight_correction=self._skinning_weight,
            )
            sk_w_field = sk_w * sk_w_sum[:, None]
            new_sk_corr = self._skinning_weight.clone()
            new_sk_corr[sk_corr_affect_mask] = -sk_w_field[sk_corr_affect_mask]
            # replace sk_w again

            optimizable_tensors = replace_tensor_to_optimizer(
                optimizer,
                [new_sk_corr],
                ["skinning_w"],
            )
            self._skinning_weight = optimizable_tensors["skinning_w"]
        return

    ########
    # * Another densification
    ########
    @torch.no_grad()
    def gradient_based_node_densification(
        self, optimizer, gradient_th, resample_factor=1.0, max_gs_per_new_node=100000 #32
    ):

        grad = self.corr_gradient_accum / (self.corr_gradient_denom + 1e-6)
        candidate_mask = grad > gradient_th  # N

        if not candidate_mask.any():
            logging.info(f"No node to densify")
            return

        gs_mu_list, gs_fr_list = [], []
        for t in range(self.T):
            mu, fr, _, _, _ = self.forward(t)
            gs_mu_list.append(mu[candidate_mask])
            gs_fr_list.append(fr[candidate_mask])

        gs_mu_list = torch.stack(gs_mu_list, dim=0)  # T,N,3
        gs_fr_list = torch.stack(gs_fr_list, dim=0)  # T,N,3,3

        # subsample the gs_mu_list

        resample_ind = resample_curve(
            D=_compute_curve_topo_dist_(
                gs_mu_list,
                curve_mask=None,
                top_k=self.scf.topo_dist_top_k,
                max_subsample_T=self.scf.topo_sample_T,
            ),
            sample_margin=resample_factor * self.scf.spatial_unit,
            mask=None,
            verbose=True,
        )

        if self.scf.M + len(resample_ind) > self.max_node_num:
            logging.warning(f"Node num exceeds the maximum limit, do not increase")
            return

        new_node_xyz = gs_mu_list[:, resample_ind]
        new_node_quat = matrix_to_quaternion(gs_fr_list[:, resample_ind])

        # append these new node into scf
        old_M = self.scf.M
        self.scf.append_nodes_traj(
            optimizer,
            new_node_xyz,
            new_node_quat,
            torch.zeros(new_node_xyz.shape[1]).to(self.scf._node_grouping),
        )
        self.scf.incremental_topology()  # ! manually must set this

        # ! warning, for now, all appended nodes are set to have group-id=0

        # find these gs's original attached node
        original_attach_ind = self.attach_ind[candidate_mask][resample_ind]

        new_attach_ind_list, new_gs_ind_list = [], []
        for _i in range(new_node_xyz.shape[1]):
            _attach_ind = old_M + _i
            # the same carrying leaves duplicate them
            neighbors_mask = self.attach_ind == original_attach_ind[_i]
            if not neighbors_mask.any():
                continue
            # ! bound the number of leaves here
            # !  WARNING, THIS MODIFICATION IS AFTER MANY BASE VERSION, BE CAREFUL
            if neighbors_mask.long().sum() > float(max_gs_per_new_node):
                # random sample max_gs_per_new_node and mark the flag
                neighbors_ind = torch.arange(self.N, device=self.device)[neighbors_mask]
                neighbors_ind = neighbors_ind[
                    torch.randperm(neighbors_ind.shape[0])[:max_gs_per_new_node]
                ]
                neighbors_mask = torch.zeros_like(neighbors_mask)
                neighbors_mask[neighbors_ind] = True
            #
            gs_ind = torch.arange(self.N, device=self.device)[neighbors_mask]
            new_attach_ind_list.append(torch.ones_like(gs_ind) * _attach_ind)
            new_gs_ind_list.append(gs_ind)
        if len(new_attach_ind_list) == 0:
            logging.info(f"No new leaves to append")
            return
        new_attach_ind = torch.cat(new_attach_ind_list, dim=0)
        new_gs_ind = torch.cat(new_gs_ind_list, dim=0)

        logging.info(
            f"Append {new_node_xyz.shape[1]} new nodes and dup {new_gs_ind.shape[0]} leaves"
        )

        assert new_gs_ind.max() < self.N, f"{new_gs_ind.max()}, {self.N}"

        self._densification_postprocess(
            optimizer,
            new_xyz=self._xyz[new_gs_ind].detach().clone(),
            new_r=self._rotation[new_gs_ind].detach().clone(),
            new_s_logit=self._scaling[new_gs_ind].detach().clone(),
            new_o_logit=self._opacity[new_gs_ind].detach().clone(),
            new_sph_dc=self._features_dc[new_gs_ind].detach().clone(),
            new_sph_rest=self._features_rest[new_gs_ind].detach().clone(),
            new_skinning_w=self._skinning_weight[new_gs_ind].detach().clone(),
            new_dyn_logit=self.o_inv_act(
                torch.ones_like(self._xyz[new_gs_ind][:, :1]) * 0.99
            ),
        )

        self.attach_ind = torch.cat(
            [self.attach_ind, new_attach_ind.to(self.attach_ind)], dim=0
        )
        assert (
            self.attach_ind.max() < self.scf.M
        ), f"{self.attach_ind.max()}, {self.scf.M}"
        self.ref_time = torch.cat(
            [self.ref_time, self.ref_time.clone()[new_gs_ind]], dim=0
        )
        assert self.ref_time.max() < self.T, f"{self.ref_time.max()}, {self.T}"

        self.clean_corr_control_record()
        return

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

    def record_corr_grad(self, viewspace_point_tensor_grad, update_filter):
        # Record the gradient norm, invariant across different poses
        assert len(viewspace_point_tensor_grad) == self.N
        self.corr_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor_grad[update_filter, :2], dim=-1, keepdim=False
        )
        self.corr_gradient_denom[update_filter] += 1
        return

    def _densification_postprocess(
        self,
        optimizer,
        new_xyz,
        new_r,
        new_s_logit,
        new_o_logit,
        new_sph_dc,
        new_sph_rest,
        # additional parameters
        new_skinning_w,
        new_dyn_logit,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_sph_dc,
            "f_rest": new_sph_rest,
            "opacity": new_o_logit,
            "scaling": new_s_logit,
            "rotation": new_r,
            "skinning_w": new_skinning_w,
            "dyn_logit": new_dyn_logit,
        }
        d = {k: v for k, v in d.items() if v is not None}

        # * update parameters
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._skinning_weight = optimizable_tensors["skinning_w"]
        self._dynamic_logit = optimizable_tensors["dyn_logit"]

        # * update the recording buffer
        # ! Note, must update the other buffers outside this function
        self.xyz_gradient_accum = torch.zeros(self.N, device=self.device)
        self.xyz_gradient_denom = torch.zeros(self.N, device=self.device)
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros_like(new_xyz[:, 0])], dim=0
        )
        self.corr_gradient_accum = torch.cat(
            [
                self.corr_gradient_accum,
                torch.zeros_like(new_xyz[:, 0]).to(self.corr_gradient_accum),
            ],
            dim=0,
        )
        self.corr_gradient_denom = torch.cat(
            [
                self.corr_gradient_denom,
                torch.zeros_like(new_xyz[:, 0]).to(self.corr_gradient_denom),
            ],
            dim=0,
        )
        return

    def clean_gs_control_record(self):
        self.xyz_gradient_accum = torch.zeros_like(self.xyz_gradient_accum)
        self.xyz_gradient_denom = torch.zeros_like(self.xyz_gradient_denom)
        self.max_radii2D = torch.zeros_like(self.max_radii2D)
        return

    def clean_corr_control_record(self):
        self.corr_gradient_accum = torch.zeros_like(self.corr_gradient_accum)
        self.corr_gradient_denom = torch.zeros_like(self.corr_gradient_denom)
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
        new_skinning_w = self._skinning_weight[selected_pts_mask]
        new_dyn_logit = self._dynamic_logit[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s_logit=new_scaling,
            new_o_logit=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_skinning_w=new_skinning_w,
            new_dyn_logit=new_dyn_logit,
        )

        # update leaf buffer
        new_attach_ind = self.attach_ind[selected_pts_mask]
        new_ref_time = self.ref_time[selected_pts_mask]
        self.attach_ind = torch.cat(
            [self.attach_ind, new_attach_ind], dim=0
        )  # ! now copy the topology, but the best is to recompute the topology!
        self.ref_time = torch.cat([self.ref_time, new_ref_time], dim=0)

        return new_xyz.shape[0]

    def _densify_and_split(
        self,
        optimizer,
        grad_norm,
        grad_threshold,
        scale_th,
        N=2,
    ):
        # TODO: check whether this has a bug, seems very noisy
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
        # ! no matter local or global, such disturbance is correct
        rots = quaternion_to_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.einsum("nij,nj->ni", rots, samples) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = _scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scaling = torch.clamp(new_scaling, max=self.max_scale, min=self.min_scale)
        new_scaling = self.s_inv_act(new_scaling)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
        new_dyn_logit = self._dynamic_logit[selected_pts_mask].repeat(N, 1)
        new_skinning_w = self._skinning_weight[selected_pts_mask].repeat(N, 1)

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s_logit=new_scaling,
            new_o_logit=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_skinning_w=new_skinning_w,
            new_dyn_logit=new_dyn_logit,
        )

        # update leaf buffer
        new_attach_ind = self.attach_ind[selected_pts_mask].repeat(N)
        new_ref_time = self.ref_time[selected_pts_mask].repeat(N)
        self.attach_ind = torch.cat(
            [self.attach_ind, new_attach_ind], dim=0
        )  # ! now copy the topology, but the best is to recompute the topology!
        self.ref_time = torch.cat([self.ref_time, new_ref_time], dim=0)

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device=self.device, dtype=bool
                ),
            )
        )
        self._prune_points(optimizer, prune_filter)
        return new_xyz.shape[0]

    def densify(
        self,
        optimizer,
        max_grad,
        percent_dense,
        extent,
        verbose=True,
    ):
        grads = self.xyz_gradient_accum / self.xyz_gradient_denom
        grads[grads.isnan()] = 0.0
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
        max_3d_radius=1000,
        min_scale=0.0,
        verbose=True,
    ):
        opacity = self.o_act(self._opacity)
        prune_mask = (opacity < min_opacity).squeeze()
        logging.info(f"opacity_pruning {prune_mask.sum()}")
        if max_screen_size:  # if a point is too large
            big_points_vs = self.max_radii2D > max_screen_size
            logging.info(f"radii2D_pruning {big_points_vs.sum()}")
            # ! also consider the scale
            max_scale = self.get_s.max(dim=1).values
            big_points_scale = max_scale > max_3d_radius
            logging.info(f"radii3D_pruning {big_points_scale.sum()}")
            big_points_vs = torch.logical_or(big_points_vs, big_points_scale)
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            # * reset the maxRadii
            self.max_radii2D = torch.zeros_like(self.max_radii2D)
        # also check the scale
        too_small_mask = (self.get_s < min_scale).all(dim=-1)
        logging.info(f"small_pruning {big_points_scale.sum()}")
        prune_mask = torch.logical_or(prune_mask, too_small_mask)
        if (~prune_mask).sum() < 32:
            logging.warning("If prune, too few GS will left (less than 32), skip!")
            return
        rest_cnt = (~prune_mask).sum()
        if rest_cnt < self.min_num_gs:
            logging.warning(
                f"Prune will result in less than {self.min_num_gs} GS, skip!"
            )
            return
        self._prune_points(optimizer, prune_mask)
        if verbose:
            logging.info(f"Prune: {prune_mask.sum()}")
        return

    def _prune_points(self, optimizer, mask):
        valid_points_mask = ~mask
        # if valid_points_mask.all():
        #     return
        optimizable_tensors = prune_optimizer(
            optimizer,
            valid_points_mask,
            exclude_names=self.op_update_exclude,
        )

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._skinning_weight = optimizable_tensors["skinning_w"]
        self._dynamic_logit = optimizable_tensors["dyn_logit"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_denom = self.xyz_gradient_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.corr_gradient_accum = self.corr_gradient_accum[valid_points_mask]
        self.corr_gradient_denom = self.corr_gradient_denom[valid_points_mask]

        # update leaf buffer
        self.attach_ind = self.attach_ind[valid_points_mask]
        self.ref_time = self.ref_time[valid_points_mask]
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
        # * manually assign self attr with the keys
        for key in ckpt:
            if hasattr(self, key):
                # if is parameter, then assign
                if isinstance(getattr(self, key), nn.Parameter):
                    getattr(self, key).data = torch.as_tensor(ckpt[key])
                else:
                    setattr(self, key, ckpt[key])
            else:
                logging.warning(f"Key [{key}] is not in the model!")

        # load others
        self._init_act(self.max_scale, self.min_scale)
        self.load_state_dict(ckpt, strict=True)
        # this is critical, reinit the funcs
        self._init_act(self.max_scale, self.min_scale)

        return

    ######################################################################
    # * Regularization
    ######################################################################

    # ! note, all the vel and arap loss here are mean reduction
    def compute_vel_acc_loss(self, tids=None, detach_mask=None):
        return self.scf.compute_vel_acc_loss(tids, detach_mask)

    def compute_arap_loss(
        self,
        tids=None,
        temporal_diff_weight=[0.75, 0.25],
        temporal_diff_shift=[1, 4],
        detach_tids_mask=None,
    ):
        return self.scf.compute_arap_loss(
            tids, temporal_diff_weight, temporal_diff_shift, detach_tids_mask
        )

    #########
    # viz
    ########

    def get_node_sinning_w_acc(self, reduce="sum"):
        sk_ind, sk_w, _, _, _ = self.scf.get_skinning_weights(
            query_xyz=self.get_xyz(),
            query_t=self.ref_time,
            attach_ind=self.attach_ind,
            skinning_weight_correction=(
                self._skinning_weight if self.w_correction_flag else None
            ),
        )
        sk_ind, sk_w = sk_ind.reshape(-1), sk_w.reshape(-1)
        acc_w = torch.zeros_like(self.scf._node_xyz[0, :, 0])
        acc_w = acc_w.scatter_reduce(0, sk_ind, sk_w, reduce=reduce, include_self=False)
        return acc_w

    def get_node_attached_leaves_count(self):
        attach_ind = self.attach_ind
        leaf_count = torch.zeros(self.M, device=self.device)
        leaf_count = leaf_count.scatter_add(
            dim=0, index=attach_ind, src=torch.ones_like(attach_ind).float()
        )
        return leaf_count


##################################################################################


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


def sph_order2nfeat(order):
    return (order + 1) ** 2


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def subsample_vtx(vtx, voxel_size):
    # vtx: N,3
    # according to surfelwarp global_config.h line28 d_node_radius=0.025 meter; and warpfieldinitializer.cpp line39 the subsample voxel is 0.7 * 0.025 meter
    # reference: VoxelSubsamplerSorting.cu line  119
    # Here use use the mean of each voxel cell
    pooling_ind = pyg_pool.voxel_grid(pos=vtx, size=voxel_size)
    unique_ind, compact_ind = pooling_ind.unique(return_inverse=True)
    candidate = torch.scatter_reduce(
        input=torch.zeros(len(unique_ind), 3).to(vtx),
        src=vtx,
        index=compact_ind[:, None].expand_as(vtx),
        dim=0,
        reduce="mean",
        # dim_size=len(unique_ind),
        include_self=False,
    )
    assert not (candidate == 0).all(dim=-1).any(), "voxel resampling has an error!"
    return candidate
