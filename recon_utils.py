import torch
import logging, glob, sys, os, shutil, os.path as osp
from datetime import datetime
import numpy as np, random, kornia
import imageio

from lib_render.render_helper import GS_BACKEND
from mosca_viz import viz_main, viz_list_of_colored_points_in_cam_frame
from lib_moca.bundle import query_buffers_by_track
from lib_moca.epi_helpers import analyze_track_epi, identify_tracks

SEED = 12345


def seed_everything(seed=SEED):
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_recon_ws(ws, fit_cfg, logdir="logs"):
    seed_everything(SEED)
    # get datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    name = getattr(fit_cfg, "exp_name", "default")
    name = f"{name}_{GS_BACKEND.lower()}_{dt_string}"
    log_path = osp.join(ws, logdir, name)
    os.makedirs(log_path, exist_ok=True)
    logging.info(f"Log dir set to: {log_path}")
    # backup the files
    logging.info(f"Backup files")
    backup_dir = osp.join(log_path, "src_backup")
    os.makedirs(backup_dir, exist_ok=True)
    for path in [
        "profile",
        "lib_prior",
        "lib_moca",
        "lib_mosca",
        "lite_moca_reconstruct.py",
        "mosca_reconstruct.py",
        "mosca_evaluate.py",
        "mosca_precompute.py",
        "mosca_viz.py",
    ]:
        os.system(f"cp -r {path} {backup_dir}")
    # reduce the backup size
    shutil.rmtree(osp.join(backup_dir, "lib_prior", "seg"))
    for root, dirs, files in os.walk(backup_dir):
        for file in files:
            if file.endswith(".pth") or file.endswith(".ckpt"):
                if osp.isfile(osp.join(root, file)):
                    os.remove(osp.join(root, file))
                else:
                    shutil.rmtree(osp.join(root, file))
    # backu the commandline args
    with open(osp.join(log_path, "fit_commandline_args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    return log_path


def auto_get_depth_dir_tap_mode(ws, fit_cfg):
    dep_dir = getattr(fit_cfg, "depth_dirname", None)
    if dep_dir is None:
        logging.info("Auto get depth dir")
        pattern = "*_depth"
        candidates = glob.glob(osp.join(ws, pattern))
        # ensure is dir
        candidates = [it for it in candidates if osp.isdir(it)]
        if len(candidates) > 1:
            # have a default order
            priority_key = ["gt", "sensor", "sharp", "depthcrafter"]
            for priority_it in priority_key:
                _candidates = [it for it in candidates if priority_it in it]
                if len(_candidates) == 1:
                    logging.warning(f"Multiple depth dir, use {priority_it} depth dir")
                    candidates = _candidates
                    break
        assert len(candidates) == 1, f"Found {len(candidates)} depth dir"
        dep_dir = osp.basename(candidates[0])
    tap_mode = getattr(fit_cfg, "tap_mode", None)
    if tap_mode is None:
        logging.info("Auto get tap mode")
        pattern = "*uniform*tap.npz"
        candidates = glob.glob(osp.join(ws, pattern))
        assert len(candidates) == 1, f"Found {len(candidates)} tap mode"
        tap_mode = osp.basename(candidates[0])
        tap_mode = tap_mode.split("_tap.npz")[0].split("_")[-1]
    return dep_dir, tap_mode


def viz_mosca_curves_before_optim(curve_xyz, curve_rgb, curve_mask, cams, viz_dir):
    # * viz
    os.makedirs(viz_dir, exist_ok=True)
    viz_list = viz_list_of_colored_points_in_cam_frame(
        [
            cams.trans_pts_to_cam(cams.T // 2, it).cpu()
            for t, it in enumerate(curve_xyz)
        ],
        curve_rgb,
        zoom_out_factor=1.0,
    )
    imageio.mimsave(osp.join(viz_dir, "curve.gif"), viz_list, loop=1000)
    if curve_mask.any(dim=1).all():
        viz_list = viz_list_of_colored_points_in_cam_frame(
            [
                cams.trans_pts_to_cam(t, it[curve_mask[t]]).cpu()
                for t, it in enumerate(curve_xyz)
            ],
            [curve_rgb[itm] for itm in curve_mask.cpu()],
            zoom_out_factor=0.2,
        )
        imageio.mimsave(osp.join(viz_dir, "cam_curve_masked.gif"), viz_list, loop=1000)
    viz_list = viz_list_of_colored_points_in_cam_frame(
        [cams.trans_pts_to_cam(t, it).cpu() for t, it in enumerate(curve_xyz)],
        curve_rgb,
        zoom_out_factor=0.2,
    )
    imageio.mimsave(osp.join(viz_dir, "cam_curve.gif"), viz_list, loop=1000)
    viz_valid_color = torch.tensor([0.0, 1.0, 0.0]).to(curve_xyz.device)
    viz_invalid_color = torch.tensor([1.0, 0.0, 0.0]).to(curve_xyz.device)
    # T,N,3
    viz_mask_color = (
        viz_valid_color[None, None] * curve_mask.float()[..., None]
        + viz_invalid_color[None, None] * (1 - curve_mask.float())[..., None]
    )
    viz_list = viz_list_of_colored_points_in_cam_frame(
        [cams.trans_pts_to_cam(t, it).cpu() for t, it in enumerate(curve_xyz)],
        [it for it in viz_mask_color],
        zoom_out_factor=0.2,
    )
    imageio.mimsave(osp.join(viz_dir, "cam_curve_valid.gif"), viz_list, loop=1000)
    return


def update_s2d_track_identification(
    s2d,
    log_path,
    epi_th,
    dyn_id_cnt,
    min_curve_num=32,
    photo_error_masks=None,
    photo_error_mode="only",
    photo_error_id_cnt=None,
):
    # identify the fg tack by EPI
    if s2d.has_epi:
        raft_epi = s2d.epi.clone()
        with torch.no_grad():
            epi_error_list = query_buffers_by_track(
                raft_epi[..., None], s2d.track, s2d.track_mask
            )
            epi_error_list = epi_error_list.squeeze(-1).cpu()
    else:
        epi_data = np.load(osp.join(log_path, "tracker_epi.npz"))
        pair_list = [tuple(it) for it in epi_data["continuous_pair_list"].tolist()]
        F_list = epi_data["F_list"]
        _, epi_error_list, _ = analyze_track_epi(
            pair_list, s2d.track, s2d.track_mask, s2d.H, s2d.W, F_list
        )
    epi_track_static_selection, epi_track_dynamic_selection = identify_tracks(
        epi_error_list, epi_th, static_cnt=1, dynamic_cnt=dyn_id_cnt
    )

    # * optionally: identify the fg track by photo error
    if photo_error_masks is not None:
        assert photo_error_mode in [
            "only",
            "and",
            "or",
        ], f"photo_error_mode={photo_error_mode}"
        with torch.no_grad():
            photo_error_list = query_buffers_by_track(
                photo_error_masks[..., None], s2d.track, s2d.track_mask
            )
            photo_error_list = photo_error_list.squeeze(
                -1
            ).cpu()  # ! this is 01 mask, not error
            if photo_error_id_cnt is None:
                photo_error_id_cnt = dyn_id_cnt
            photo_track_static_selection, photo_track_dynamic_selection = (
                identify_tracks(
                    photo_error_list, 0.5, static_cnt=1, dynamic_cnt=photo_error_id_cnt
                )
            )
        if photo_error_mode == "only":
            epi_track_static_selection = photo_track_static_selection
            epi_track_dynamic_selection = photo_track_dynamic_selection
        elif photo_error_mode == "and":
            epi_track_static_selection = (
                epi_track_static_selection & photo_track_static_selection
            )
            epi_track_dynamic_selection = (
                epi_track_dynamic_selection & photo_track_dynamic_selection
            )
        elif photo_error_mode == "or":
            epi_track_static_selection = (
                epi_track_static_selection | photo_track_static_selection
            )
            epi_track_dynamic_selection = (
                epi_track_dynamic_selection | photo_track_dynamic_selection
            )
        else:
            raise NotImplementedError(f"photo_error_mode={photo_error_mode}")

    if epi_track_dynamic_selection.sum() < min_curve_num:
        logging.warning(
            f"Dynamic curves identification get too few curves, select K={min_curve_num} highest epi curves"
        )
        epi_error_recover = epi_error_list.max(dim=0).values
        highest_k = epi_error_recover.topk(min_curve_num, largest=False).indices
        epi_track_dynamic_selection[highest_k] = True

    s2d.register_track_indentification(
        epi_track_static_selection, epi_track_dynamic_selection
    )
    return s2d  # ! warning, changed


def set_epi_mask_to_s2d_for_bg_render(s2d, epi_th, device):
    assert s2d.has_epi, "EPI is required for static warm"
    static_mask = s2d.epi < epi_th
    # erode the static mask
    kernel = torch.ones((3, 3), device=device)
    static_mask = kornia.morphology.erosion(
        static_mask.float().unsqueeze(1), kernel
    ).squeeze(1)
    dynamic_mask = kornia.morphology.erosion(
        (1.0 - static_mask).float().unsqueeze(1), kernel
    ).squeeze(1)
    s2d.register_2d_identification(
        static_2d_mask=static_mask > 0, dynamic_2d_mask=dynamic_mask > 0
    )
    logging.info(f"Set EPI maks to s2d with epi_th={epi_th}")
    return s2d
