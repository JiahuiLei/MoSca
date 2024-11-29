import torch
import os, os.path as osp
import logging
import numpy as np
from tqdm import tqdm
from typing import Literal, Optional, Tuple
from pytorch3d.ops import knn_points

from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian

from eval_utils.eval_nvidia import eval_nvidia_dir
from eval_utils.eval_dyncheck import eval_dycheck
from eval_utils.eval_sintel_cam import eval_sintel_campose
from eval_utils.eval_tum_cam import eval_metrics as eval_tum_campose
from eval_utils.eval_tum_cam import c2w_to_tumpose, load_traj as load_tum_traj

from eval_utils.campose_alignment import align_ate_c2b_use_a2b
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

import imageio
from omegaconf import OmegaConf
from data_utils.iphone_helpers import load_iphone_gt_poses
from data_utils.nvidia_helpers import load_nvidia_gt_pose, get_nvidia_dummy_test

from lib_render.render_helper import render, render_cam_pcl
from tqdm import tqdm
import imageio
from matplotlib import pyplot as plt
import cv2 as cv
from lib_render.render_helper import GS_BACKEND
import time

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


####################
# * DyCheck PCK Eval
####################


def compute_pck(
    kps0: np.ndarray,
    kps1: np.ndarray,
    img_wh: Tuple[int, int],
    ratio: float = 0.05,
    reduce: Optional[Literal["mean"]] = "mean",
) -> np.ndarray:
    """Compute PCK between two sets of keypoints given the threshold ratio.

    Canonical Surface Mapping via Geometric Cycle Consistency.
        Kulkarni et al., ICCV 2019.
        https://arxiv.org/abs/1907.10043

    Args:
        kps0 (jnp.ndarray): A set of keypoints of shape (J, 2) in float32.
        kps1 (jnp.ndarray): A set of keypoints of shape (J, 2) in float32.
        img_wh (Tuple[int, int]): Image width and height.
        ratio (float): A threshold ratios. Default: 0.05.
        reduce (Optional[Literal["mean"]]): Reduction method. Default: "mean".

    Returns:
        jnp.ndarray:
            if reduce == "mean", PCK of shape();
            if reduce is None, corrects of shape (J,).
    """
    dists = np.linalg.norm(kps0 - kps1, axis=-1)
    thres = ratio * max(img_wh)
    corrects = dists < thres
    if reduce == "mean":
        return corrects.mean()
    elif reduce is None:
        return corrects


def eval_pck(gt_list, pred_list, image_size, ratio):

    N = len(gt_list)
    assert N == len(pred_list)
    metrics = []
    for i in tqdm(range(N)):
        common_corrects = compute_pck(
            gt_list[i],
            pred_list[i],
            image_size,
            ratio,
            reduce=None,
        )
        metrics.append(common_corrects)
    mean_pck = np.mean(
        [it.mean() for it in metrics]
    )  # ! the teddy scene verified the mean is in this way, not the cat all mean, but first mean across all points and then across all paris
    return mean_pck, metrics


def load_gt_pck_data(gt_data_dict):
    gt_dst_pixel_list = [it["dst_pixel_gt"] for it in gt_data_dict]
    gt_src_pixel_list = [it["src_pixel"] for it in gt_data_dict]
    src_t_list = [it["src_t"] for it in gt_data_dict]
    dst_t_list = [it["dst_t"] for it in gt_data_dict]
    img_wh = gt_data_dict[0]["img_wh"]
    ratio = gt_data_dict[0]["ratio"]
    for it in gt_data_dict:
        assert (it["img_wh"] == img_wh).all()
        assert it["ratio"] == ratio
    return (
        gt_src_pixel_list,
        gt_dst_pixel_list,
        src_t_list,
        dst_t_list,
        img_wh,
        ratio,
    )


#########
# test helper
#########


@torch.no_grad()
def render_test(
    H,
    W,
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    save_dir=None,
    fn_list=None,
    focal=None,
    cxcy_ratio=None,
    # cover_factor=0.3,
):
    # prior2d: Prior2D = self.prior2d
    # device = self.device
    device = s_model.device

    # first align the camera
    solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
    aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
        traj_a=train_camera_T_wi,
        traj_b=solved_cam_T_wi.detach().cpu(),
        traj_c=test_camera_T_wi,
    )
    # render
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if focal is None:
        focal = cams.rel_focal
    if cxcy_ratio is None:
        cxcy_ratio = cams.cxcy_ratio

    L = min(H, W)
    fx = focal * L / 2.0
    fy = focal * L / 2.0
    cx = W * cxcy_ratio[0]
    cy = H * cxcy_ratio[1]
    K = torch.eye(3).to(device)
    K[0, 0] = K[0, 0] * 0 + fx
    K[1, 1] = K[1, 1] * 0 + fy
    K[0, 2] = K[0, 2] * 0 + cx
    K[1, 2] = K[1, 2] * 0 + cy

    test_ret = []
    for i in tqdm(range(len(test_camera_tid))):
        working_t = test_camera_tid[i]
        render_dict = render(
            [s_model(), d_model(working_t)],
            H,
            W,
            K,
            T_cw=torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device),
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)  # ! important
        test_ret.append(rgb)
        if save_dir:
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)
    return test_ret


def render_test_tto(
    H,
    W,
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    gt_rgb_dir,
    save_pose_fn,
    ##
    tto_steps=25,
    decay_start=15,
    lr_p=0.003,
    lr_q=0.003,
    lr_final=0.0001,
    ###
    gt_mask_dir=None,
    save_dir=None,
    fn_list=None,
    focal=None,
    cxcy_ratio=None,
    # dbg
    use_sgd=False,
    loss_type="psnr",
    # boost
    initialize_from_previous_camera=True,
    initialize_from_previous_step_factor=10,
    initialize_from_previous_lr_factor=0.1,
    fg_mask_th=0.1,
):
    # * Optimize the test camera pose, nost simply do the global sim(3) alignment
    s_model.eval()
    d_model.eval()

    assert gt_mask_dir is None, "THIS IS NOT CORRECT, SHOULD NOT USE GT MASK DURING TTO"

    device = s_model.device

    # first align the camera
    with torch.no_grad():
        solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
        aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=train_camera_T_wi,
            traj_b=solved_cam_T_wi.detach().cpu(),
            traj_c=test_camera_T_wi,
        )

    # render
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if focal is None:
        focal = cams.rel_focal
    if cxcy_ratio is None:
        cxcy_ratio = cams.cxcy_ratio

    L = min(H, W)
    fx = focal * L / 2.0
    fy = focal * L / 2.0
    cx = W * cxcy_ratio[0]
    cy = H * cxcy_ratio[1]
    cam_K = torch.eye(3).to(device)
    cam_K[0, 0] = cam_K[0, 0] * 0 + float(fx)
    cam_K[1, 1] = cam_K[1, 1] * 0 + float(fy)
    cam_K[0, 2] = cam_K[0, 2] * 0 + float(cx)
    cam_K[1, 2] = cam_K[1, 2] * 0 + float(cy)

    test_ret = []
    solved_pose_list = []
    for i in tqdm(range(len(test_camera_tid))):
        if initialize_from_previous_camera and i == 0:
            step_factor = initialize_from_previous_step_factor
            lr_factor = 1.0
        else:
            step_factor = 1
            lr_factor = initialize_from_previous_lr_factor

        working_t = test_camera_tid[i]
        # load gt rgb and mask
        gt_rgb = imageio.imread(osp.join(gt_rgb_dir, f"{fn_list[i]}.png")) / 255.0
        gt_rgb = gt_rgb[..., :3]
        if gt_mask_dir is None:
            gt_mask = np.ones_like(gt_rgb[..., 0])
        else:
            raise RuntimeError("Should not use this during TTO!!")
            gt_mask = imageio.imread(osp.join(gt_mask_dir, f"{fn_list[i]}.png")) / 255.0
        gt_rgb = torch.tensor(gt_rgb, device=device).float()
        gt_mask = torch.tensor(gt_mask, device=device).float()
        gt_mask_sum = gt_mask.sum()

        T_cw_init = torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device)
        T_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        t_init = torch.nn.Parameter(T_cw_init[:3, 3].detach())
        q_init = torch.nn.Parameter(matrix_to_quaternion(T_cw_init[:3, :3]).detach())
        if use_sgd:
            optimizer_type = torch.optim.SGD
        else:
            optimizer_type = torch.optim.Adam
        optimizer = optimizer_type(
            [
                {"params": t_init, "lr": lr_p * lr_factor},
                {"params": q_init, "lr": lr_q * lr_factor},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tto_steps * step_factor - decay_start,
            eta_min=lr_final * lr_factor,
        )

        loss_list = []

        with torch.no_grad():
            gs5 = [s_model(), d_model(working_t)]  # ! this does not change
        for _step in range(tto_steps * step_factor):
            optimizer.zero_grad()
            _T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
            T_cw = torch.cat([_T_cw, T_bottom[None]], 0)
            render_dict = render(gs5, H, W, cam_K, T_cw=T_cw)
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            rendered_mask = render_dict["alpha"].squeeze(-1).squeeze(0) > fg_mask_th

            if loss_type == "abs":
                raise RuntimeError("Should not use this")
                rgb_loss_i = torch.abs(pred_rgb - gt_rgb) * gt_mask[..., None]
                rgb_loss = rgb_loss_i.sum() / gt_mask_sum
            elif loss_type == "psnr":
                mse = ((pred_rgb - gt_rgb) ** 2)[rendered_mask].mean()
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                rgb_loss = -psnr

            else:
                raise ValueError(f"Unknown loss tyoe {loss_type}")

            loss = rgb_loss
            loss.backward()
            optimizer.step()
            if _step >= decay_start:
                scheduler.step()

            loss_list.append(loss.item())

        solved_T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
        solved_T_cw = torch.cat([solved_T_cw, T_bottom[None]], 0)
        solved_pose_list.append(solved_T_cw.detach().cpu().numpy())
        with torch.no_grad():

            render_dict = render(
                [s_model(), d_model(working_t)], H, W, cam_K, T_cw=T_cw
            )
            rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
            rgb = np.clip(rgb, 0, 1)  # ! important
            test_ret.append(rgb)
        if save_dir:
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)
        logging.info(f"TTO {fn_list[i]}: {loss_list[0]:.3f}->{loss_list[-1]:.3f}")
        if initialize_from_previous_camera and i < len(test_camera_tid) - 1:
            aligned_test_camera_T_wi[i + 1] = torch.linalg.inv(solved_T_cw).to(
                aligned_test_camera_T_wi
            )
    np.savez(save_pose_fn, poses=solved_pose_list)
    return test_ret


def test_main(
    cfg,
    saved_dir,
    data_root,
    device,
    tto_flag,
    eval_also_dyncheck_non_masked=False,
    skip_test_gen=False,
):
    # ! this func can be called at the end of running, or run seperately after trained

    # get cfg
    if data_root.endswith("/"):
        data_root = data_root[:-1]
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
        OmegaConf.set_readonly(cfg, True)

    dataset_mode = getattr(cfg, "mode", "iphone")
    # max_sph_order = getattr(cfg, "max_sph_order", 1)
    logging.info(f"Dataset mode: {dataset_mode}")

    ######################################################################
    ######################################################################

    cams = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(saved_dir, "photometric_cam.pth"))
    ).to(device)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_s_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_d_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )

    cams.to(device)
    cams.eval()
    d_model.to(device)
    d_model.eval()
    s_model.to(device)
    s_model.eval()

    ######################################################################
    ######################################################################

    if dataset_mode == "iphone":
        (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            _,
            gt_testing_cxcy_ratio_list,
        ) = load_iphone_gt_poses(data_root, getattr(cfg, "t_subsample", 1))
        gt_dir = osp.join(data_root, "test_images")
        # * cfg
        tto_steps = getattr(cfg, "tto_steps", 30)
        decay_start = getattr(cfg, "tto_decay_start", 15)
        lr_p = getattr(cfg, "tto_lr_p", 0.003)
        lr_q = getattr(cfg, "tto_lr_q", 0.003)
        lr_final = getattr(cfg, "tto_lr_final", 0.0001)
        sgd_flag = False
        tto_initialize_from_previous_step_factor = 10
        tto_initialize_from_previous_lr_factor = 0.1
        tto_fg_mask_th = 0.1

    elif dataset_mode == "nvidia":
        # ! always use the first training view
        gt_training_cam_T_wi = cams.T_wc_list().detach().cpu()
        gt_training_fov = cams.fov

        (
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_testing_fov_list,
            gt_testing_cxcy_ratio_list,
        ) = get_nvidia_dummy_test(gt_training_cam_T_wi, gt_training_fov)
        gt_dir = osp.join(
            # "./data/robust_dynrf/results/Nvidia/gt/", osp.basename(data_root)
            "./eval_utils/nvidia_rodynrf_gt",
            osp.basename(data_root),
        )
        # * cfg
        gt_testing_fov_list[0] = gt_testing_fov_list[0][0]
        tto_steps = getattr(cfg, "tto_steps", 100)
        decay_start = getattr(cfg, "tto_decay_start", 30)
        lr_p = getattr(cfg, "tto_lr_p", 0.0003)
        lr_q = getattr(cfg, "tto_lr_q", 0.0003)
        lr_final = getattr(cfg, "tto_lr_final", 0.000001)
        sgd_flag = False
        # ！ use original
        tto_initialize_from_previous_step_factor = 1
        tto_initialize_from_previous_lr_factor = 1.0
        tto_fg_mask_th = 0.1

    else:
        raise ValueError(
            f"Unknown dataset mode: {dataset_mode}, shouldn't call test funcs"
        )
    # id the image size
    sample_fn = [
        f for f in os.listdir(gt_dir) if f.endswith(".png") or f.endswith(".jpg")
    ][0]
    sample = imageio.imread(osp.join(gt_dir, sample_fn))
    H, W = sample.shape[:2]

    ######################################################################
    ######################################################################

    eval_prefix = "tto_" if tto_flag else ""

    if not skip_test_gen:
        for test_i in range(len(gt_testing_cam_T_wi_list)):
            testing_fov = gt_testing_fov_list[test_i]
            testing_focal = 1.0 / np.tan(np.deg2rad(testing_fov) / 2.0)

            if tto_flag:
                frames = render_test_tto(
                    gt_rgb_dir=gt_dir,
                    tto_steps=tto_steps,
                    decay_start=decay_start,
                    lr_p=lr_p,
                    lr_q=lr_q,
                    lr_final=lr_final,
                    use_sgd=sgd_flag,
                    #
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=gt_testing_cam_T_wi_list[test_i],
                    test_camera_tid=gt_testing_tids_list[test_i],
                    save_dir=osp.join(saved_dir, f"tto_test"),
                    save_pose_fn=osp.join(saved_dir, f"tto_test_pose_{test_i}"),
                    fn_list=gt_testing_fns_list[test_i],
                    focal=testing_focal,
                    cxcy_ratio=gt_testing_cxcy_ratio_list[test_i],
                    #
                    initialize_from_previous_camera=True,
                    initialize_from_previous_step_factor=tto_initialize_from_previous_step_factor,
                    initialize_from_previous_lr_factor=tto_initialize_from_previous_lr_factor,
                    fg_mask_th=tto_fg_mask_th,
                )
                imageio.mimsave(
                    osp.join(saved_dir, f"tto_test_cam{test_i}.mp4"), frames
                )
            else:
                frames = render_test(
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=gt_testing_cam_T_wi_list[test_i],
                    test_camera_tid=gt_testing_tids_list[test_i],
                    save_dir=osp.join(saved_dir, "test"),
                    fn_list=gt_testing_fns_list[test_i],
                    focal=testing_focal,
                    cxcy_ratio=gt_testing_cxcy_ratio_list[test_i],
                )
                imageio.mimsave(osp.join(saved_dir, f"test_cam{test_i}.mp4"), frames)

    # * Test
    if dataset_mode == "iphone":
        eval_dycheck(
            save_dir=saved_dir,
            gt_rgb_dir=gt_dir,
            gt_mask_dir=osp.join(data_root, "test_covisible"),
            pred_dir=osp.join(saved_dir, f"{eval_prefix}test"),
            save_prefix=eval_prefix,
            strict_eval_all_gt_flag=True,  # ! only support full len now!!
            eval_non_masked=eval_also_dyncheck_non_masked,
        )

    elif dataset_mode == "nvidia":
        if data_root.endswith("/"):
            data_root = data_root[:-1]
        eval_nvidia_dir(
            gt_dir=gt_dir,
            pred_dir=osp.join(saved_dir, f"{eval_prefix}test"),
            report_dir=osp.join(saved_dir, f"{eval_prefix}test_report"),
        )

    logging.info(f"Finished, saved to {saved_dir}")
    return


@torch.no_grad()
def test_pck(saved_dir, gt_npz_fn, device, save_fn=None):
    # laod gt
    src, dst_gt, src_t, dst_t, img_wh, ratio = load_gt_pck_data(
        np.load(gt_npz_fn, allow_pickle=True)["arr_0"]
    )

    cams = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(saved_dir, "photometric_cam.pth"))
    ).to(device)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_s_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_d_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )

    cams.to(device)
    cams.eval()
    d_model.to(device)
    d_model.eval()
    s_model.to(device)
    s_model.eval()

    H, W = int(cams.default_H), int(cams.default_W)
    assert img_wh[0] == W and img_wh[1] == H

    dst_pred = []
    for _st, _dt, _src in tqdm(zip(src_t, dst_t, src)):
        _st, _dt = int(_st), int(_dt)

        # ! use RGB to render xyz because should also work with native renderor
        # render world coordinate map
        d_gs5_src = d_model(_st)
        d_gs5_dst = d_model(_dt)
        s_gs5 = s_model()

        mu = torch.cat([s_gs5[0], d_gs5_src[0]], 0)
        fr = torch.cat([s_gs5[1], d_gs5_src[1]], 0)
        s = torch.cat([s_gs5[2], d_gs5_src[2]], 0)
        o = torch.cat([s_gs5[3], d_gs5_src[3]], 0)
        sph = torch.cat([s_gs5[4], d_gs5_src[4]], 0)

        T_cw = cams.T_cw(_st)
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
        mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
        fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)

        xyz_dst = torch.cat([s_gs5[0], d_gs5_dst[0]], 0)
        mu_dst_cam = cams.trans_pts_to_cam(_dt, xyz_dst)

        render_dict = render_cam_pcl(
            mu_cam,
            fr_cam,
            s,
            o,
            sph,
            cams.default_H,
            cams.default_W,
            CAM_K=cams.K(),
            bg_color=[0.0, 0.0, 0.0],
            colors_precomp=mu_dst_cam,
        )
        dst_xyz_map = render_dict["rgb"].permute(1, 2, 0)

        rounded_src = np.round(_src).astype(int)
        rounded_src[:, 0] = np.clip(rounded_src[:, 0], 0, W - 1)
        rounded_src[:, 1] = np.clip(rounded_src[:, 1], 0, H - 1)
        index = rounded_src[:, 1] * W + rounded_src[:, 0]

        dst_xyz = dst_xyz_map.reshape(-1, 3)[index]

        dst_uv = cams.project(dst_xyz)
        dst_x = (dst_uv[:, :1] + 1.0) / 2.0 * 360.0
        dst_y = (dst_uv[:, 1:] + 480.0 / 360.0) / 2.0 * 360.0
        _dst_pred = torch.cat([dst_x, dst_y], dim=1).cpu().numpy()
        dst_pred.append(_dst_pred)

    pck005, _ = eval_pck(dst_gt, dst_pred, img_wh, ratio)
    print(f"PCK@0.05: {pck005}")
    if save_fn is not None:
        with open(save_fn, "w") as fp:
            fp.write(f"PCK@0.05: {pck005:.10f}\n")
    return pck005


def test_sintel_cam(cam_pth_fn, ws, save_path="sintel_pose_metrics.txt"):
    cams = MonocularCameras.load_from_ckpt(torch.load(cam_pth_fn))
    pose_est = cams.T_wc_list().detach().cpu().numpy()
    # gt_dir = osp.join("./data/robust_dynrf/results/Sintel", sq)
    gt_dir = osp.join(ws, "gt_cameras")
    ate, rpe_trans, rpe_rot = eval_sintel_campose(pose_est[:, :3], gt_dir=gt_dir)
    logging.info(
        f"Sintel ATE: {ate}, RPE Translation: {rpe_trans}, RPE Rotation: {rpe_rot}"
    )
    # save to txt
    with open(save_path, "w") as fp:
        fp.write(f"ATE: {ate:.10f}\n")
        fp.write(f"RPE-trans: {rpe_trans:.10f}\n")
        fp.write(f"RPE-rot: {rpe_rot:.10f}\n")
    return ate, rpe_trans, rpe_rot


def test_tum_cam(cam_pth_fn, ws, save_path="tum_pose_metrics.txt"):

    cams = MonocularCameras.load_from_ckpt(torch.load(cam_pth_fn))
    pose_est = cams.T_wc_list().detach().cpu().numpy()
    tt = np.arange(len(pose_est)).astype(float)
    tum_poses = [c2w_to_tumpose(p) for p in pose_est]
    tum_poses = np.stack(tum_poses, 0)
    pred_traj = [tum_poses, tt]

    gt_traj = load_tum_traj(
        gt_traj_file=osp.join(ws, "groundtruth_90.txt"), traj_format="tum"
    )

    ate, rpe_trans, rpe_rot = eval_tum_campose(pred_traj, gt_traj)
    # plot_trajectory(
    #     pred_traj, gt_traj, title=seq, filename=f'{save_dir}/{seq}.png'
    # )
    logging.info(
        f"TUM ATE: {ate}, RPE Translation: {rpe_trans}, RPE Rotation: {rpe_rot}"
    )
    # save to txt
    with open(save_path, "w") as fp:
        fp.write(f"ATE: {ate:.10f}\n")
        fp.write(f"RPE-trans: {rpe_trans:.10f}\n")
        fp.write(f"RPE-rot: {rpe_rot:.10f}\n")
    return ate, rpe_trans, rpe_rot


def test_fps(saved_dir, rounds=1, device=torch.device("cuda:0")):
    cams = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(saved_dir, "photometric_cam.pth"))
    ).to(device)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_s_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_d_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )

    cams.to(device)
    cams.eval()
    d_model.to(device)
    d_model.eval()
    s_model.to(device)
    s_model.eval()

    d_model.set_inference_mode()

    sample_t = [0, cams.T // 2, cams.T - 1]

    s_gs5 = s_model()
    H, W = cams.default_H, cams.default_W
    K = cams.K(H, W)

    viz = []
    for t in sample_t:
        d_gs5 = d_model(t)
        rd = render([s_gs5, d_gs5], H, W, K, T_cw=cams.T_cw(t))
        rd_sample = rd["rgb"].permute(1, 2, 0).cpu().detach().numpy()
        viz.append(rd_sample)
    viz = np.concatenate(viz, 1)
    imageio.imsave(osp.join(saved_dir, "fps_eval_samples.jpg"), viz)

    cnt = cams.T * rounds
    with torch.no_grad():
        start_t = time.time()
        for t in tqdm(range(cnt)):
            t = t % d_model.T
            d_gs5 = d_model(t)
            rd = render([s_gs5, d_gs5], H, W, K, T_cw=cams.T_cw(t))
        end_t = time.time()
    duration = end_t - start_t
    fps = cnt / duration
    logging.info(f"FPS: {fps} tested in rounds {rounds}, rendered {cnt} frames")
    with open(osp.join(saved_dir, "fps_eval.txt"), "w") as fp:
        fp.write(f"FPS: {fps : .10f}\n")
    return


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", type=str, help="Source folder")
    parser.add_argument("--cfg", type=str, help="profile yaml file path")
    parser.add_argument("--logdir", type=str, help="log dir")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    logdir = args.logdir

    datamode = getattr(cfg, "mode", "iphone")
    if datamode == "sintel":
        test_func = test_sintel_cam
    elif datamode == "tum":
        test_func = test_tum_cam
    else:
        test_func = None
    if test_func is not None:
        test_func(
            cam_pth_fn=osp.join(logdir, "photometric_cam.pth"),
            ws=args.ws,
            save_path=osp.join(logdir, "final_cam_eval.txt"),
        )

    test_main(
        cfg,
        saved_dir=logdir,
        data_root=args.ws,
        device=torch.device("cuda"),
        tto_flag=True,
        eval_also_dyncheck_non_masked=False,
        skip_test_gen=False,
    )
