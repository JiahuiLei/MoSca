# useful function for reconstruction solver

# Single File
from matplotlib import pyplot as plt
import torch, numpy as np
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
import os, sys, os.path as osp
import torch.nn.functional as F
from tqdm import tqdm
from pytorch3d.ops import knn_points
from torch import nn
import kornia
import logging
import imageio
import colorsys


sys.path.append(osp.dirname(osp.abspath(__file__)))

from camera import MonocularCameras

from dynamic_gs import DynSCFGaussian
from static_gs import StaticGaussian
from gs_utils.gs_optim_helper import update_learning_rate, get_expon_lr_func

import logging
import sys, os, os.path as osp


def apply_gs_control(
    render_list,
    model,
    gs_control_cfg,
    step,
    optimizer_gs,
    first_N=None,
    last_N=None,
    record_flag=True,
):
    for render_dict in render_list:
        if first_N is not None:
            assert last_N is None
            grad = render_dict["viewspace_points"].grad[:first_N]
            radii = render_dict["radii"][:first_N]
            visib = render_dict["visibility_filter"][:first_N]
        elif last_N is not None:
            assert first_N is None
            grad = render_dict["viewspace_points"].grad[-last_N:]
            radii = render_dict["radii"][-last_N:]
            visib = render_dict["visibility_filter"][-last_N:]
        else:
            grad = render_dict["viewspace_points"].grad
            radii = render_dict["radii"]
            visib = render_dict["visibility_filter"]
        if record_flag:
            model.record_xyz_grad_radii(grad, radii, visib)
    if (
        step in gs_control_cfg.densify_steps
        or step in gs_control_cfg.prune_steps
        or step in gs_control_cfg.reset_steps
    ):
        logging.info(f"GS Control at {step}")
    if step in gs_control_cfg.densify_steps:
        N_old = model.N
        model.densify(
            optimizer=optimizer_gs,
            max_grad=gs_control_cfg.densify_max_grad,
            percent_dense=gs_control_cfg.densify_percent_dense,
            extent=0.5,
            verbose=True,
        )
        logging.info(f"Densify: {N_old}->{model.N}")
    if step in gs_control_cfg.prune_steps:
        N_old = model.N
        model.prune_points(
            optimizer_gs,
            min_opacity=gs_control_cfg.prune_opacity_th,
            max_screen_size=1e10,  # disabled
        )
        logging.info(f"Prune: {N_old}->{model.N}")
    if step in gs_control_cfg.reset_steps:
        model.reset_opacity(optimizer_gs, gs_control_cfg.reset_opacity)
    return


class GSControlCFG:
    def __init__(
        self,
        densify_steps=300,
        reset_steps=900,
        prune_steps=300,
        densify_max_grad=0.0002,
        densify_percent_dense=0.01,
        prune_opacity_th=0.012,
        reset_opacity=0.01,
    ):
        if isinstance(densify_steps, int):
            densify_steps = [densify_steps * i for i in range(100000)]
        if isinstance(reset_steps, int):
            reset_steps = [reset_steps * i for i in range(100000)]
        if isinstance(prune_steps, int):
            prune_steps = [prune_steps * i for i in range(100000)]
        self.densify_steps = densify_steps
        self.reset_steps = reset_steps
        self.prune_steps = prune_steps
        self.densify_max_grad = densify_max_grad
        self.densify_percent_dense = densify_percent_dense
        self.prune_opacity_th = prune_opacity_th
        self.reset_opacity = reset_opacity
        self.summary()
        return

    def summary(self):
        logging.info("GSControlCFG: Summary")
        logging.info(
            f"GSControlCFG: densify_steps={self.densify_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(
            f"GSControlCFG: reset_steps={self.reset_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(
            f"GSControlCFG: prune_steps={self.prune_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(f"GSControlCFG: densify_max_grad={self.densify_max_grad}")
        logging.info(
            f"GSControlCFG: densify_percent_dense={self.densify_percent_dense}"
        )
        logging.info(f"GSControlCFG: prune_opacity_th={self.prune_opacity_th}")
        logging.info(f"GSControlCFG: reset_opacity={self.reset_opacity}")
        return


class OptimCFG:
    def __init__(
        self,
        # GS
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest_factor=20.0,
        lr_p_final=None,
        lr_cam_q=0.0001,
        lr_cam_t=0.0001,
        lr_cam_f=0.00,
        lr_cam_q_final=None,
        lr_cam_t_final=None,
        lr_cam_f_final=None,
        # # dyn
        lr_np=0.00016,
        lr_nq=0.001,
        lr_nsig=0.00001,
        lr_w=0.0,  # ! use 0.0
        lr_dyn=0.01,
        lr_np_final=None,
        lr_nq_final=None,
        lr_w_final=None,
    ) -> None:
        # gs
        self.lr_p = lr_p
        self.lr_q = lr_q
        self.lr_s = lr_s
        self.lr_o = lr_o
        self.lr_sph = lr_sph
        self.lr_sph_rest = lr_sph / lr_sph_rest_factor
        # cam
        self.lr_cam_q = lr_cam_q
        self.lr_cam_t = lr_cam_t
        self.lr_cam_f = lr_cam_f
        # # dyn
        self.lr_np = lr_np
        self.lr_nq = lr_nq
        self.lr_w = lr_w
        self.lr_dyn = lr_dyn
        self.lr_nsig = lr_nsig

        # gs scheduler
        self.lr_p_final = lr_p_final if lr_p_final is not None else lr_p / 100.0
        self.lr_cam_q_final = (
            lr_cam_q_final if lr_cam_q_final is not None else lr_cam_q / 10.0
        )
        self.lr_cam_t_final = (
            lr_cam_t_final if lr_cam_t_final is not None else lr_cam_t / 10.0
        )
        self.lr_cam_f_final = (
            lr_cam_f_final if lr_cam_f_final is not None else lr_cam_f / 10.0
        )
        self.lr_np_final = lr_np_final if lr_np_final is not None else lr_np / 100.0
        self.lr_nq_final = lr_nq_final if lr_nq_final is not None else lr_nq / 10.0
        if lr_w is not None:
            self.lr_w_final = lr_w_final if lr_w_final is not None else lr_w / 10.0
        else:
            self.lr_w_final = 0.0
        return

    def summary(self):
        logging.info("OptimCFG: Summary")
        logging.info(f"OptimCFG: lr_p={self.lr_p}")
        logging.info(f"OptimCFG: lr_q={self.lr_q}")
        logging.info(f"OptimCFG: lr_s={self.lr_s}")
        logging.info(f"OptimCFG: lr_o={self.lr_o}")
        logging.info(f"OptimCFG: lr_sph={self.lr_sph}")
        logging.info(f"OptimCFG: lr_sph_rest={self.lr_sph_rest}")
        logging.info(f"OptimCFG: lr_cam_q={self.lr_cam_q}")
        logging.info(f"OptimCFG: lr_cam_t={self.lr_cam_t}")
        logging.info(f"OptimCFG: lr_cam_f={self.lr_cam_f}")
        logging.info(f"OptimCFG: lr_p_final={self.lr_p_final}")
        logging.info(f"OptimCFG: lr_cam_q_final={self.lr_cam_q_final}")
        logging.info(f"OptimCFG: lr_cam_t_final={self.lr_cam_t_final}")
        logging.info(f"OptimCFG: lr_cam_f_final={self.lr_cam_f_final}")
        logging.info(f"OptimCFG: lr_np={self.lr_np}")
        logging.info(f"OptimCFG: lr_nq={self.lr_nq}")
        logging.info(f"OptimCFG: lr_w={self.lr_w}")
        logging.info(f"OptimCFG: lr_dyn={self.lr_dyn}")
        logging.info(f"OptimCFG: lr_nsig={self.lr_nsig}")
        logging.info(f"OptimCFG: lr_np_final={self.lr_np_final}")
        logging.info(f"OptimCFG: lr_nq_final={self.lr_nq_final}")
        logging.info(f"OptimCFG: lr_w_final={self.lr_w_final}")
        return

    @property
    def get_static_lr_dict(self):
        return {
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "lr_s": self.lr_s,
            "lr_o": self.lr_o,
            "lr_sph": self.lr_sph,
            "lr_sph_rest": self.lr_sph_rest,
        }

    @property
    def get_dynamic_lr_dict(self):
        return {
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "lr_s": self.lr_s,
            "lr_o": self.lr_o,
            "lr_sph": self.lr_sph,
            "lr_sph_rest": self.lr_sph_rest,
            "lr_np": self.lr_np,
            "lr_nq": self.lr_nq,
            "lr_w": self.lr_w,
            "lr_dyn": self.lr_dyn,
            "lr_nsig": self.lr_nsig,
        }

    @property
    def get_dynamic_node_lr_dict(self):
        return {
            "lr_p": 0.0,
            "lr_q": 0.0,
            "lr_s": 0.0,
            "lr_o": 0.0,
            "lr_sph": 0.0,
            "lr_sph_rest": 0.0,
            "lr_np": self.lr_np,
            "lr_nq": self.lr_nq,
            "lr_w": 0.0,
            "lr_dyn": 0.0,
            "lr_nsig": self.lr_nsig,
        }

    @property
    def get_cam_lr_dict(self):
        return {
            "lr_q": self.lr_cam_q,
            "lr_t": self.lr_cam_t,
            "lr_f": self.lr_cam_f,
        }

    def get_scheduler(self, total_steps):
        # todo: decide whether to decay skinning weights
        gs_scheduling_dict = {
            "xyz": get_expon_lr_func(
                lr_init=self.lr_p,
                lr_final=self.lr_p_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "node_xyz": get_expon_lr_func(
                lr_init=self.lr_np,
                lr_final=self.lr_np_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "node_rotation": get_expon_lr_func(
                lr_init=self.lr_nq,
                lr_final=self.lr_nq_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
        }
        cam_scheduling_dict = {
            "R": get_expon_lr_func(
                lr_init=self.lr_cam_q,
                lr_final=self.lr_cam_q_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "t": get_expon_lr_func(
                lr_init=self.lr_cam_t,
                lr_final=self.lr_cam_t_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "f": get_expon_lr_func(
                lr_init=self.lr_cam_f,
                lr_final=self.lr_cam_f_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
        }
        return gs_scheduling_dict, cam_scheduling_dict


@torch.no_grad()
def fetch_leaves_in_world_frame(
    cams: MonocularCameras,
    n_attach: int,  # if negative use all
    #
    input_mask_list,
    input_dep_list,
    input_rgb_list,
    input_normal_list=None,  # ! in new version, do not use this
    input_inst_list=None,  # ! in new version, do not use this
    #
    save_xyz_fn=None,
    start_t=0,
    end_t=-1,
    t_list=None,
    subsample=1,
    squeeze_normal_ratio=1.0,
):
    device = cams.rel_focal.device

    if end_t == -1:
        end_t = cams.T
    if t_list is None:
        t_list = list(range(start_t, end_t))
    if subsample > 1:
        logging.info(f"2D Subsample {subsample} for fetching ...")

    mu_list, quat_list, scale_list, rgb_list, time_index_list = [], [], [], [], []
    inst_list = []  # collect the leaf id as well

    for t in tqdm(t_list):
        mask2d = input_mask_list[t].bool()
        H, W = mask2d.shape
        if subsample > 1:
            mask2d[::subsample, ::subsample] = False

        dep_map = input_dep_list[t].clone()
        cam_pcl = cams.backproject(
            cams.get_homo_coordinate_map(H, W)[mask2d].clone(), dep_map[mask2d]
        )
        cam_R_wc, cam_t_wc = cams.Rt_wc(t)
        mu = cams.trans_pts_to_world(t, cam_pcl)
        rgb_map = input_rgb_list[t].clone()
        rgb = rgb_map[mask2d]
        K = cams.K(H, W)
        radius = cam_pcl[:, -1] / (0.5 * K[0, 0] + 0.5 * K[1, 1]) * float(subsample)
        scale = torch.stack([radius / squeeze_normal_ratio, radius, radius], dim=-1)
        time_index = torch.ones_like(mu[:, 0]).long() * t

        if input_normal_list is not None:
            nrm_map = input_normal_list[t].clone()
            cam_nrm = nrm_map[mask2d]
            nrm = F.normalize(torch.einsum("ij,nj->ni", cam_R_wc, cam_nrm), dim=-1)
            rx = nrm.clone()
            ry = F.normalize(torch.cross(rx, mu, dim=-1), dim=-1)
            rz = F.normalize(torch.cross(rx, ry, dim=-1), dim=-1)
            rot = torch.stack([rx, ry, rz], dim=-1)
        else:
            rot = torch.eye(3)[None].expand(len(radius), -1, -1)
        quat = matrix_to_quaternion(rot)

        mu_list.append(mu.cpu())
        quat_list.append(quat.cpu())
        scale_list.append(scale.cpu())
        rgb_list.append(rgb.cpu())
        time_index_list.append(time_index.cpu())

        if input_inst_list is not None:
            inst_map = inst_list[t].clone()
            inst = inst_map[mask2d]
            inst_list.append(inst.cpu())

    mu_all = torch.cat(mu_list, 0)
    quat_all = torch.cat(quat_list, 0)
    scale_all = torch.cat(scale_list, 0)
    rgb_all = torch.cat(rgb_list, 0)

    logging.info(f"Fetching {n_attach/1000.0:.3f}K out of {len(mu_all)/1e6:.3}M pts")
    if n_attach > len(mu_all) or n_attach <= 0:
        choice = torch.arange(len(mu_all))
    else:
        choice = torch.randperm(len(mu_all))[:n_attach]

    # make gs5 param (mu, fr, s, o, sph) no rescaling
    mu_init = mu_all[choice].clone()
    q_init = quat_all[choice].clone()
    s_init = scale_all[choice].clone()
    o_init = torch.ones(len(choice), 1).to(mu_init)
    rgb_init = rgb_all[choice].clone()
    time_init = torch.cat(time_index_list, 0)[choice]
    if len(inst_list) > 0:
        inst_all = torch.cat(inst_list, 0)
        inst_init = inst_all[choice].clone().to(device)
    else:
        inst_init = None
    if save_xyz_fn is not None:
        np.savetxt(
            save_xyz_fn,
            torch.cat([mu_init, rgb_init * 255], 1).detach().cpu().numpy(),
            fmt="%.6f",
        )
    torch.cuda.empty_cache()
    return (
        mu_init.to(device),
        q_init.to(device),
        s_init.to(device),
        o_init.to(device),
        rgb_init.to(device),
        inst_init,
        time_init.to(device),
    )


def estimate_normal_map(
    vtx_map,
    mask,
    normal_map_patchsize=7,
    normal_map_nn_dist_th=0.03,
    normal_map_min_nn=6,
):
    # * this func is borrowed from my pytorch4D repo in 2022 May.
    # the normal neighborhood is estimated
    # the normal computation refer to pytorch3d 0.6.1, but updated with linalg operations in newer pytroch version

    # note, here the path has zero padding on the border, but due to the computation, sum the zero zero covariance matrix will make no difference, safe!
    H, W = mask.shape
    v_map_patch = F.unfold(
        vtx_map.permute(2, 0, 1).unsqueeze(0),
        normal_map_patchsize,
        dilation=1,
        padding=(normal_map_patchsize - 1) // 2,
        stride=1,
    ).reshape(3, normal_map_patchsize**2, H, W)

    mask_patch = F.unfold(
        mask.unsqueeze(0).unsqueeze(0).float(),
        normal_map_patchsize,
        dilation=1,
        padding=(normal_map_patchsize - 1) // 2,
        stride=1,
    ).reshape(1, normal_map_patchsize**2, H, W)

    # Also need to consider the neighbors distance for occlusion boundary
    nn_dist = (vtx_map.permute(2, 0, 1).unsqueeze(1) - v_map_patch).norm(dim=0)
    valid_nn_mask = (nn_dist < normal_map_nn_dist_th)[None, ...]
    v_map_patch[~valid_nn_mask.expand(3, -1, -1, -1)] = 0.0
    mask_patch = mask_patch * valid_nn_mask

    # only meaningful when there are at least 3 valid pixels in the neighborhood, the pixels with less nn need to be exclude when computing the final output normal map, but the mask_patch shouldn't be updated because they still can be used to compute normals for other pixels
    neighbor_counts = mask_patch.sum(dim=1).squeeze(0)  # H,W
    valid_mask = torch.logical_and(mask, neighbor_counts >= normal_map_min_nn)

    v_map_patch = v_map_patch.permute(2, 3, 1, 0)  # H,W,Patch,3
    mask_patch = mask_patch.permute(2, 3, 1, 0)  # H,W,Patch,1
    vtx_patch = v_map_patch[valid_mask]  # M,Patch,3
    neighbor_counts = neighbor_counts[valid_mask]
    mask_patch = mask_patch[valid_mask]  # M,Patch,1

    # compute the curvature normal for the neighbor hood
    # fix several bug here: 1.) the centroid mean bug 2.) the local coordinate should be mask to zero for cov mat
    assert neighbor_counts.min() > 0
    centroid = vtx_patch.sum(dim=1, keepdim=True) / (neighbor_counts[:, None, None])
    vtx_patch = vtx_patch - centroid
    vtx_patch = vtx_patch * mask_patch
    # vtx_patch = vtx_patch.double()
    W = torch.matmul(vtx_patch.unsqueeze(-1), vtx_patch.unsqueeze(-2))
    # todo: here can use distance/confidence to weight the contribution
    W = W.sum(dim=1)  # M,3,3

    # # # one way to solve normal
    # curvature = torch.linalg.eigvalsh(W)
    # c_normal = curvature[:, :1]
    # I = torch.eye(3).to(W.device)[None, ...].expand_as(W)
    # A = W - I * c_normal[..., None]
    # _, _, _Vh = torch.linalg.svd(A)
    # normal = _Vh[:, 2, :]  # M,3

    curvature, local_frame = torch.linalg.eigh(W)
    normal = local_frame[..., 0]

    # # rotate normal towards the camera and filter out some surfels (75deg according to the surfelwarp paper)
    # ray_dir = Homo_cali_unilen[valid_mask]  # M,3
    # inner = (ray_dir * normal).sum(-1)  # the towards cam dir should have cos < 0.0
    # sign_multiplier = -torch.sign(inner)
    # oriented_normal = normal * sign_multiplier[:, None]
    # # debug
    # debug_inner = (ray_dir * oriented_normal).sum(-1)

    # ! warning, when selecting the grazing surfels, here we consider the angel to the principle axis, not the rays
    # inner = oriented_normal[..., 2]  # the z component
    # filter out the surfels whose normal are too tangent to the **ray dir**? or the principle axis? Now use the ray dir
    # valid_normal_mask = inner <= self.normal_valid_cos_th
    # valid_mask[valid_mask.clone()] = valid_normal_mask

    normal_map = torch.zeros_like(vtx_map)
    normal_map[valid_mask] = normal

    return normal_map, valid_mask


########################################################################
# ! node grow
########################################################################

from lib_render.render_helper import render


@torch.no_grad()
def error_grow_dyn_model(
    s2d,
    cams: MonocularCameras,
    s_model,
    d_model,
    optimizer_dynamic,
    step,
    dyn_error_grow_th,
    dyn_error_grow_num_frames,
    dyn_error_grow_subsample,
    viz_dir,
    open_k_size=3,
    opacity_init_factor=0.98,
):
    # * identify the error mask
    device = s2d.rgb.device
    error_list = identify_rendering_error(cams, s_model, d_model, s2d)
    T = len(error_list)
    imageio.mimsave(
        osp.join(viz_dir, f"error_raw_{step}.mp4"), error_list.cpu().numpy()
    )

    grow_fg_masks = (error_list > dyn_error_grow_th).to(device)
    open_kernel = torch.ones(open_k_size, open_k_size)
    # handle large time by chunk the time
    cur = 0
    chunk = 50
    grow_fg_masks_morph = []
    while cur < T:
        _grow_fg_masks = kornia.morphology.opening(
            grow_fg_masks[cur : cur + chunk, None].float(),
            kernel=open_kernel.to(grow_fg_masks.device),
        ).squeeze(1)
        grow_fg_masks_morph.append(_grow_fg_masks.bool())
        cur = cur + chunk
    grow_fg_masks = torch.cat(grow_fg_masks_morph, 0)
    grow_fg_masks = grow_fg_masks * s2d.dep_mask * s2d.dyn_mask
    # viz
    imageio.mimsave(
        osp.join(viz_dir, f"error_{step}.mp4"),
        grow_fg_masks.detach().cpu().float().numpy(),
    )

    if dyn_error_grow_num_frames < T:
        # sample some frames to grow
        grow_cnt = grow_fg_masks.reshape(T, -1).sum(-1)
        grow_cnt = grow_cnt.detach().cpu().numpy()
        grow_tids = np.argsort(grow_cnt)[-dyn_error_grow_num_frames:][::-1]
        # plot the grow_cnt with bars
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(grow_cnt)), grow_cnt)
        plt.savefig(osp.join(viz_dir, f"error_{step}_grow_cnt.jpg"))
        plt.close()
    else:
        grow_tids = [i for i in range(T)]
    logging.info(f"Grow Error at {step} on {grow_tids}")

    for grow_tid in tqdm(grow_tids):
        grow_mask = grow_fg_masks[grow_tid]
        if grow_mask.sum() == 0:
            continue
        # ! The append points must be in front of the model depth!!!
        grow_mu_cam, grow_mask = align_to_model_depth(
            s2d,
            working_mask=grow_mask.bool(),
            cams=cams,
            tid=grow_tid,
            s_model=s_model,
            d_model=d_model,
            dep_align_knn=-1,  # 400, #32,
            sub_sample=dyn_error_grow_subsample,
        )
        if len(grow_mu_cam) == 0:
            continue
        # convert to mu_w and get s_inti
        R_wc, t_wc = cams.Rt_wc(grow_tid)
        grow_mu_w = torch.einsum("ij,aj->ai", R_wc, grow_mu_cam) + t_wc[None]
        K = cams.K(s2d.H, s2d.W)
        grow_s = (
            grow_mu_cam[:, -1]
            / (0.5 * K[0, 0] + 0.5 * K[1, 1])
            * float(dyn_error_grow_subsample)
        )

        # ! special function that sample nodes from candidates and select attached nodes
        quat_w = torch.zeros(len(grow_mu_w), 4).to(grow_mu_w)
        quat_w[:, 0] = 1.0
        d_model.append_new_node_and_gs(
            optimizer_dynamic,
            tid=grow_tid,
            mu_w=grow_mu_w,
            quat_w=quat_w,
            scales=grow_s[:, None].expand(-1, 3),
            opacity=torch.ones_like(grow_s)[:, None] * opacity_init_factor,
            rgb=s2d.rgb[grow_tid][grow_mask],
        )
    return


@torch.no_grad()
def identify_rendering_error(
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    s2d,
):
    # render all frames and compare photo error
    logging.info("Compute rendering errors ...")
    error_list = []
    for t in tqdm(range(d_model.T)):  # ! warning, d_model.T may smaller than cams.T
        gs5 = [s_model(), d_model(t)]
        render_dict = render(gs5, s2d.H, s2d.W, cams.K(s2d.H, s2d.W), cams.T_cw(t))
        rgb_pred = render_dict["rgb"].permute(1, 2, 0)
        rgb_gt = s2d.rgb[t].clone()
        error = (rgb_pred - rgb_gt).abs().max(dim=-1).values
        error_list.append(error.detach().cpu())
    error_list = torch.stack(error_list, 0)
    return error_list


def get_subsample_mask_like(buffer, sub_sample):
    # buffer is H,W,...
    assert buffer.ndim >= 2
    ret = torch.zeros_like(buffer).bool()
    ret[::sub_sample, ::sub_sample, ...] = True
    return ret


@torch.no_grad()
def __align_pixel_depth_scale_backproject__(
    homo_map,
    src_align_mask,
    src,
    src_mask,
    dst,
    dst_mask,
    cams: MonocularCameras,
    knn=8,
    infrontflag=True,
):
    # src_align_mask: which pixel is going to be aligned
    # src_mask: the valid pixel in the src
    # dst_mask: the valid pixel in the dst
    # use 3D nearest knn nn to find the best scaling, local rigid warp

    # the base pts build correspondence between current frame and
    base_mask = src_mask * dst_mask * (~src_align_mask)
    query_mask = src_align_mask

    ratio = dst / (src + 1e-6)
    base_pts_ratio = ratio[base_mask]

    query_pts = cams.backproject(homo_map[query_mask], src[query_mask])

    # backproject src depth to 3D
    if knn > 0:
        base_pts = cams.backproject(homo_map[base_mask], src[base_mask])
        _, ind, _ = knn_points(query_pts[None], base_pts[None], K=knn)
        ind = ind[0]
        ratio = base_pts_ratio[ind]
    else:
        ratio = base_pts_ratio.mean()[None, None].expand(len(query_pts), -1)
    ret = ratio.mean(-1, keepdim=True) * query_pts

    if infrontflag:
        # ! make sure the ret-z is always smaller than the dst z
        ret_z = ret[:, -1]
        dst_z = dst[query_mask]
        assert (dst_z > -1e-6).all()
        new_z = torch.min(ret_z, dst_z - 1e-4)
        logging.info(f"Make sure the aligned points is in front of the dst depth")
        ratio = new_z / torch.clamp(ret_z, min=1e-6)
        # assert (ratio <= 1 + 1e-6).all()
    ret = ret * ratio[:, None]
    return ret


def align_to_model_depth(
    s2d,
    working_mask,
    cams,
    tid,
    s_model,
    d_model=None,
    dep_align_knn=9,
    sub_sample=1,
):
    gs5 = [s_model()]
    if d_model:
        gs5.append(d_model(tid))
    render_dict = render(gs5, s2d.H, s2d.W, cams.K(s2d.H, s2d.W), cams.T_cw(tid))
    model_alpha = render_dict["alpha"].squeeze(0)
    model_dep = render_dict["dep"].squeeze(0)
    # align prior depth to current depth
    sub_mask = get_subsample_mask_like(working_mask, sub_sample)
    ret_mask = working_mask * sub_mask
    new_mu_cam = __align_pixel_depth_scale_backproject__(
        homo_map=s2d.homo_map,
        src_align_mask=ret_mask,
        src=s2d.dep[tid].clone(),
        src_mask=s2d.dep_mask[tid] * sub_mask,
        dst=model_dep,
        dst_mask=(model_alpha > 0.5)
        & (
            ~working_mask
        ),  # ! warning, here manually use the original non-subsampled mask, because the dilated place is not reliable!
        cams=cams,
        knn=dep_align_knn,
    )
    return new_mu_cam, ret_mask


########################################################################
# ! end node grow
########################################################################


def __query_image_buffer__(uv, buffer):
    # buffer: H, W, C; uv: N, 2
    if buffer.ndim == 2:
        buffer = buffer[..., None]
    H, W = buffer.shape[:2]
    uv = torch.round(uv).long()
    uv[:, 0] = torch.clamp(uv[:, 0], 0, W - 1)
    uv[:, 1] = torch.clamp(uv[:, 1], 0, H - 1)
    uv = torch.round(uv).long()
    ind = uv[:, 1] * W + uv[:, 0]
    return buffer.reshape(-1, buffer.shape[-1])[ind]


def identify_traj_id(uv_trajs, visibs, idmap_list):
    # ! this is only for point odeyssey
    assert (
        len(uv_trajs) == len(visibs) == len(idmap_list)
    ), f"{len(uv_trajs)} vs {len(visibs)} vs {len(idmap_list)}"
    T, N = uv_trajs.shape[:2]
    collected_id = torch.ones_like(visibs, dtype=idmap_list.dtype) * -1
    for t in range(T):
        uv = uv_trajs[t]
        visib = visibs[t]
        idmap = idmap_list[t]
        traj_id = __query_image_buffer__(uv[visib], torch.as_tensor(idmap)).squeeze(-1)
        buff = collected_id[t]
        buff[visib] = traj_id
        collected_id[t] = buff

    valid_mask = collected_id >= 0
    ret = []
    for i in range(N):
        traj_id = collected_id[:, i]
        mask = valid_mask[:, i]
        if not mask.any():
            ret.append(-1)
            continue
        ids = traj_id[mask]
        # majority voting
        id_count = torch.bincount(ids)
        id_count[0] = 0
        max_id = torch.argmax(id_count)
        ret.append(max_id.item())
    ret = torch.as_tensor(ret).to(uv_trajs.device)
    return ret


def get_colorplate(n):
    hue = np.linspace(0, 1, n + 1)[:-1]
    color_plate = torch.Tensor([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue])
    return color_plate