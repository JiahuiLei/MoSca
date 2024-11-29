import torch
import os, os.path as osp
import logging
import numpy as np
from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
import imageio
from omegaconf import OmegaConf
from viz_utils import *
from lib_render.render_helper import GS_BACKEND

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


def load_model_cfg(cfg, saved_dir, device=torch.device("cuda")):

    # get cfg
    if saved_dir.endswith("/"):
        saved_dir = saved_dir[:-1]
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
        OmegaConf.set_readonly(cfg, True)

    dataset_mode = getattr(cfg, "dataset_mode", "iphone")
    max_sph_order = getattr(cfg, "max_sph_order", 1)
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

    return cfg, d_model, s_model, cams


def viz_main(
    save_dir,
    log_dir,
    cfg_fn,
    N=5,
    move_angle_deg=20.0,
    H_3d=960,
    # H_3d=640,
    W_3d=960,
    fov_3d=70,
    back_ratio_3d=0.8,
    up_ratio=0.4,
    bg_color=[1.0, 1.0, 1.0],
):
    cfg, d_model, s_model, cams = load_model_cfg(cfg_fn, log_dir)
    H, W = cams.default_H, cams.default_W

    rel_focal_3d = 1.0 / np.tan(np.deg2rad(fov_3d) / 2.0)

    key_steps = [cams.T // 2, cams.T - 1, 0, cams.T // 4, 3 * cams.T // 4][:N]

    # * Get pose
    global_pose_list = get_global_3D_cam_T_cw(
        s_model,
        d_model,
        cams,
        H,
        W,
        cams.T // 2,
        back_ratio=back_ratio_3d,
        up_ratio=up_ratio,
    )
    global_pose_list = global_pose_list[None].expand(cams.T, -1, -1)
    training_pose_list = [cams.T_cw(t) for t in range(cams.T)]

    # * #############################################################################
    save_fn_prefix = osp.join(save_dir, f"3D_moving_cam")
    viz_single_2d_camera_video(
        H_3d,
        W_3d,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        global_pose_list,
        rel_focal=rel_focal_3d,
        bg_color=bg_color,
    )

    # viz 3D
    save_fn_prefix = osp.join(save_dir, f"3D_moving_node")
    viz_single_2d_node_video(
        H_3d,
        W_3d,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        global_pose_list,
        rel_focal=rel_focal_3d,
        bg_color=bg_color,
    )

    save_fn_prefix = osp.join(save_dir, f"3D_moving_flow")
    viz_single_2d_flow_video(
        H_3d,
        W_3d,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        global_pose_list,
        rel_focal=rel_focal_3d,
        bg_color=bg_color,
    )

    save_fn_prefix = osp.join(save_dir, f"3D_moving")
    viz_single_2d_video(
        H_3d,
        W_3d,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        global_pose_list,
        rel_focal=rel_focal_3d,
        bg_color=bg_color,
    )

    # flow
    save_fn_prefix = osp.join(save_dir, f"training_moving_flow")
    viz_single_2d_flow_video(
        H,
        W,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        training_pose_list,
        bg_color=bg_color,
    )
    # node
    save_fn_prefix = osp.join(save_dir, f"training_moving_node")
    viz_single_2d_node_video(
        H,
        W,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        training_pose_list,
        bg_color=bg_color,
    )
    # rgb
    save_fn_prefix = osp.join(save_dir, f"training_moving")
    viz_single_2d_video(
        H,
        W,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        training_pose_list,
        bg_color=bg_color,
    )

    # * #############################################################################
    # key_time_step = cams.T // 2
    for key_time_step in key_steps:
        fixed_pose_list = [cams.T_cw(key_time_step) for _ in range(cams.T)]
        round_pose_list = get_move_around_cam_T_cw(
            s_model,
            d_model,
            cams,
            H,
            W,
            np.deg2rad(move_angle_deg),
            total_steps=cams.T,  # cams.T
            center_id=key_time_step,
        )

        # viz flow
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving_flow")
        viz_single_2d_flow_video(
            H, W, cams, s_model, d_model, save_fn_prefix, fixed_pose_list
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving_flow")
        viz_single_2d_flow_video(
            H, W, cams, s_model, d_model, save_fn_prefix, round_pose_list
        )
        # Viz node
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving_node")
        viz_single_2d_node_video(
            H, W, cams, s_model, d_model, save_fn_prefix, round_pose_list
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_freezing_node")
        viz_single_2d_node_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            save_fn_prefix,
            round_pose_list,
            model_t=key_time_step,
            bg_color=bg_color,
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving_node")
        viz_single_2d_node_video(
            H, W, cams, s_model, d_model, save_fn_prefix, fixed_pose_list
        )
        # Viz rgb
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            save_fn_prefix,
            round_pose_list,
            bg_color=bg_color,
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_freezing")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            save_fn_prefix,
            round_pose_list,
            model_t=key_time_step,
            bg_color=bg_color,
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            save_fn_prefix,
            fixed_pose_list,
            bg_color=bg_color,
        )

    return


if __name__ == "__main__":

    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--cfg", "-c", type=str, required=True)
    args.add_argument("--logdir", "-r", type=str, required=True)
    args.add_argument("--savedir", "-s", type=str, required=True)
    args.add_argument("--N", "-n", type=int, default=1)
    args.add_argument("--move_angle_deg", "-m", type=float, default=10.0)
    args = args.parse_args()

    viz_main(
        args.savedir,
        args.logdir,
        args.cfg,
        N=args.N,
        move_angle_deg=args.move_angle_deg,
    )
