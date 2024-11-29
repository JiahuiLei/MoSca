import torch
import imageio
import sys, os, os.path as osp
import numpy as np
import logging
import kornia
from omegaconf import OmegaConf

from lib_prior.prior_loading import Saved2D

from lib_render.render_helper import GS_BACKEND

from lib_moca.camera import MonocularCameras

from lib_mosca.mosca import MoSca
from lib_mosca.dynamic_solver import get_dynamic_curves
from lib_mosca.dynamic_solver import geometry_scf_init
from lib_mosca.photo_recon_utils import OptimCFG, GSControlCFG
from lib_mosca.mosca import MoSca
from lib_mosca.photo_recon import DynReconstructionSolver
from lib_mosca.static_gs import StaticGaussian
from lib_mosca.misc import seed_everything
from lib_mosca.dynamic_solver_utils import (
    round_int_coordinates,
    query_image_buffer_by_pix_int_coord,
)

from mosca_viz import viz_main, viz_list_of_colored_points_in_cam_frame
from mosca_evaluate import test_tum_cam, test_sintel_cam, test_main, test_pck, test_fps
from lite_moca_reconstruct import static_reconstruct

from recon_utils import (
    SEED,
    seed_everything,
    setup_recon_ws,
    auto_get_depth_dir_tap_mode,
    update_s2d_track_identification,
    viz_mosca_curves_before_optim,
    set_epi_mask_to_s2d_for_bg_render,
)


def get_static_render_error_mask(s2d, log_path, render_error_th, open_ksize=-1):
    assert osp.exists(
        osp.join(log_path, "photo_warmup_rendered.npz")
    ), "Photo warmup result not found"
    photo_data = np.load(osp.join(log_path, "photo_warmup_rendered.npz"))
    device = s2d.rgb.device
    rgb_rendered = torch.tensor(photo_data["rgb"]).to(device).permute(0, 2, 3, 1)
    render_error = abs(s2d.rgb - rgb_rendered).max(dim=-1).values
    render_error_mask = render_error > render_error_th
    render_error_mask_viz = (render_error_mask[..., None] * s2d.rgb).cpu().numpy()
    imageio.mimsave(
        osp.join(log_path, f"render_error_mask_th={render_error_th:.3f}.gif"),
        (render_error_mask_viz * 255).astype(np.uint8),
    )
    if open_ksize > 0:
        kernel = torch.ones(3, 3).to(render_error_mask)
        render_error_mask = kornia.morphology.opening(render_error_mask, kernel)
    return render_error_mask


def photometric_warmup(ws, log_path, fit_cfg):
    seed_everything(SEED)
    # ! here the warup do not need to start from low opa, only when mix two component we start from low opa!
    DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg)
    DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th", 1.0)
    DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0)
    EPI_TH = getattr(fit_cfg, "photo_warm_epi_th", getattr(fit_cfg, "epi_th", 1e-3))
    PHOTO_STATIC_WARM_STEPS = getattr(fit_cfg, "photo_static_warm_steps", -1)
    if PHOTO_STATIC_WARM_STEPS < 0:
        logging.info("No static warmup needed")
        return
    device = torch.device("cuda:0")
    logging.info(
        f"First run static bg GS warm up to save time before joint optimization"
    )

    s2d = (
        Saved2D(ws)
        .load_epi()
        .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=DEP_MEDIAN)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(
            TAP_MODE, min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4)
        )
        .rescale_perframe_depth_from_bundle(
            bundle_pth_fn=osp.join(log_path, "bundle", "bundle.pth")
        )
        .load_vos()
        .to(device)
    )
    s2d = set_epi_mask_to_s2d_for_bg_render(s2d, EPI_TH, device)
    cams: MonocularCameras = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(log_path, "bundle", "bundle_cams.pth"))
    ).to(device)

    photo_solver = DynReconstructionSolver(
        working_dir=log_path,
        device=device,
        radius_init_factor=getattr(fit_cfg, "gs_radius_init_factor", 4.0),
        opacity_init_factor=getattr(fit_cfg, "gs_opacity_init_factor", 0.95),
    )
    if GS_BACKEND == "gof":
        photo_solver.compute_normals_for_s2d(
            s2d, cams, patch_size=7, nn_dist_th=0.03, nn_min_cnt=4
        )

    s_model = photo_solver.get_static_model(
        s2d=s2d,
        cams=cams,
        n_init=getattr(fit_cfg, "gs_static_n_init", 30000),
        radius_max=getattr(fit_cfg, "gs_radius_max", 0.1),
        max_sph_order=getattr(fit_cfg, "gs_max_sph_order", 0),
        mask_type="static_depth",
    )

    # registrate the static mask
    photo_solver.photometric_fit(
        phase_name="static_warm",
        s2d=s2d,
        total_steps=PHOTO_STATIC_WARM_STEPS,
        optim_cam_after_steps=getattr(fit_cfg, "photo_warm_optim_cam_after_steps", 0),
        decay_start=getattr(fit_cfg, "photo_warm_decay_start", 2000),
        cams=cams,
        s_model=s_model,
        # losses
        lambda_rgb=getattr(fit_cfg, "photo_warm_lambda_rgb", 1.0),
        lambda_dep=getattr(fit_cfg, "photo_warm_lambda_dep", 0.1),
        lambda_mask=getattr(fit_cfg, "photo_warm_lambda_mask", 0.0),
        dep_st_invariant=getattr(fit_cfg, "photo_warm_dep_st_invariant", True),
        lambda_normal=getattr(fit_cfg, "photo_warm_lambda_normal", 0.05),
        lambda_depth_normal=getattr(fit_cfg, "photo_warm_lambda_depth_normal", 0.05),
        lambda_distortion=getattr(fit_cfg, "photo_warm_lambda_distortion", 100.0),
        optimizer_cfg=OptimCFG(
            lr_cam_f=0.0,
            lr_cam_q=0.00003,
            lr_cam_t=0.00003,
            # gs
            lr_p=getattr(fit_cfg, "photo_warm_lr_p", 0.00016),
            lr_q=getattr(fit_cfg, "photo_warm_lr_q", 0.001),
            lr_s=getattr(fit_cfg, "photo_warm_lr_s", 0.005),
            lr_o=getattr(fit_cfg, "photo_warm_lr_o", 0.05),
            lr_sph=getattr(fit_cfg, "photo_warm_lr_sph", 0.0025),
            lr_sph_rest_factor=getattr(fit_cfg, "photo_warm_lr_sph_rest_factor", 20.0),
            lr_p_final=getattr(fit_cfg, "photo_warm_lr_p_final", 0.00016 / 100),
        ),
        s_gs_ctrl_cfg=GSControlCFG(
            densify_steps=getattr(fit_cfg, "photo_warm_s_ctrl_densify_steps", 400),
            reset_steps=getattr(fit_cfg, "photo_warm_s_ctrl_reset_steps", 1001),
            prune_steps=getattr(fit_cfg, "photo_warm_s_ctrl_prune_steps", 200),
            densify_max_grad=getattr(
                fit_cfg, "photo_warm_s_ctrl_densify_max_grad", 0.0002
            ),
            densify_percent_dense=getattr(
                fit_cfg, "photo_warm_s_ctrl_densify_percent_dense", 0.01
            ),
            prune_opacity_th=getattr(
                fit_cfg, "photo_warm_s_ctrl_prune_opacity_th", 0.05
            ),
            reset_opacity=getattr(fit_cfg, "photo_warm_s_ctrl_reset_opacity", 0.01),
        ),
        s_gs_ctrl_start_ratio=getattr(fit_cfg, "photo_warm_s_ctrl_start_ratio", 0.01),
        s_gs_ctrl_end_ratio=getattr(fit_cfg, "photo_warm_s_ctrl_end_ratio", 0.9),
        # viz
        viz_skip_t=1 if cams.T < 120 else max(1, cams.T // 50),
        viz_interval=getattr(fit_cfg, "photo_warm_viz_interval", -1),
        viz_cheap_interval=getattr(fit_cfg, "photo_warm_viz_cheap_interval", -1),
        viz_move_angle_deg=getattr(fit_cfg, "photo_warm_viz_move_angle_deg", 15.0),
        random_bg=getattr(fit_cfg, "photo_warm_random_bg", True),
    )
    # update the bundle cam
    rgb, dep, alp = photo_solver.render_all(cams, s_model=s_model)
    np.savez(
        osp.join(log_path, "photo_warmup_rendered.npz"),
        rgb=rgb.cpu().numpy(),
        dep=dep.cpu().numpy(),
        alp=alp.cpu().numpy(),
    )
    os.rename(
        osp.join(log_path, "bundle", "bundle_cams.pth"),
        osp.join(log_path, "bundle", "bundle_cams_ba.pth"),
    )
    torch.save(cams.state_dict(), osp.join(log_path, "bundle", "bundle_cams.pth"))

    datamode = getattr(fit_cfg, "mode", "iphone")
    if datamode == "sintel":
        test_func = test_sintel_cam
    elif datamode == "tum":
        test_func = test_tum_cam
    else:
        test_func = None
    if test_func is not None:
        test_func(
            cam_pth_fn=osp.join(log_path, "bundle", "bundle_cams.pth"),
            ws=ws,
            save_path=osp.join(log_path, "cam_metrics_warmup.txt"),
        )

    return


def scaffold_reconstruct(ws, log_path, fit_cfg):
    seed_everything(SEED)
    DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg)
    DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th", 1.0)

    EPI_TH = getattr(fit_cfg, "epi_th", 1e-3)
    DYN_ID_CNT = getattr(fit_cfg, "dyn_id_cnt", 2 * 4)
    SCF_GEO_KEYFRAME_RATE = getattr(fit_cfg, "scf_geo_keyframe_rate", 4)
    DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0)
    device = torch.device("cuda:0")

    # load solved camera and s2d and rescale
    s2d = (
        Saved2D(ws)
        .load_epi()
        .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=DEP_MEDIAN)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(
            TAP_MODE, min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4)
        )
        .rescale_perframe_depth_from_bundle(
            bundle_pth_fn=osp.join(log_path, "bundle", "bundle.pth")
        )
        .load_vos()
        .to(device)
    )

    # re-identify the static and dynamic regions
    consider_photo_error_dyn_id_th = getattr(
        fit_cfg, "consider_photo_error_dyn_id_th", -1
    )
    if consider_photo_error_dyn_id_th > 0:
        photo_error_masks = get_static_render_error_mask(
            s2d,
            log_path,
            render_error_th=consider_photo_error_dyn_id_th,
            open_ksize=getattr(fit_cfg, "consider_photo_error_dyn_id_open_ksize", -1),
        )
    else:
        photo_error_masks = None

    s2d = update_s2d_track_identification(
        s2d,
        log_path,
        EPI_TH,
        DYN_ID_CNT,
        min_curve_num=getattr(fit_cfg, "min_curve_num", 32),
        photo_error_masks=photo_error_masks,
        photo_error_mode=getattr(fit_cfg, "consider_photo_error_dyn_id_mode", "and"),
        photo_error_id_cnt=getattr(
            fit_cfg, "consider_photo_error_dyn_id_cnt", DYN_ID_CNT
        ),
    )
    np.savez(
        osp.join(log_path, "track_identification.npz"),
        static_track_mask=s2d.static_track_mask.cpu().numpy(),
        dynamic_track_mask=s2d.dynamic_track_mask.cpu().numpy(),
    )

    if s2d.has_epi:
        viz_epi_mask = s2d.epi > EPI_TH
        viz_epi_mask = viz_epi_mask[..., None] * s2d.rgb
        imageio.mimsave(
            osp.join(log_path, f"epi_th={EPI_TH}_hardmask.gif"),
            (viz_epi_mask.cpu().numpy() * 255).astype(np.uint8),
        )

    cams: MonocularCameras = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(log_path, "bundle", "bundle_cams.pth"))
    ).to(device)

    sub_t_list = [
        t for t in range(s2d.T) if t % SCF_GEO_KEYFRAME_RATE == 0 or t == s2d.T - 1
    ]
    logging.info(f"Dyn GEO first work on len(sub_t_list)={len(sub_t_list)} key frames")

    get_dynamic_curves_filter_factor = (
        s2d.scale_nw
        if getattr(fit_cfg, "get_dynamic_curves_filter_factor_in_world", True)
        else 1.0
    )
    curve_xyz, curve_mask, curve_rgb, curve_filter_mask = get_dynamic_curves(
        s2d,
        cams,
        t_list=sub_t_list,
        refilter_2d_track_flag=True,
        refilter_min_valid_cnt=DYN_ID_CNT,
        refilter_shaking_th=getattr(
            fit_cfg, "get_curve_refilter_shaking_th_world", 0.15
        )
        * get_dynamic_curves_filter_factor,
        refilter_spatracker_consistency_th=getattr(
            fit_cfg, "get_curve_refilter_spatracker_consistency_th_world", 0.15
        )
        * get_dynamic_curves_filter_factor,
        refilter_remove_shaking_curve=getattr(
            fit_cfg, "get_curve_refilter_remove_shaking_curve", True
        ),
        enforce_line_init=getattr(
            fit_cfg, "get_curve_enforce_line_init", False
        ),  # for spatracker sometime set True is better, because spatracker invisible 3D posotion is not reliable
        min_num_curves=getattr(fit_cfg, "min_curve_num", 32),
    )
    curve_uv = s2d.track[:, s2d.dynamic_track_mask][:, curve_filter_mask][sub_t_list][
        ..., :2
    ]
    curve_rgb = (curve_rgb * curve_mask.unsqueeze(-1)).sum(0) / (
        curve_mask.sum(0).unsqueeze(-1) + 1e-3
    )

    # * refilter the curve by photo error if set
    refilter_curve_by_photo_error_cnt = getattr(
        fit_cfg, "refilter_curve_by_photo_error_cnt", -1
    )
    if refilter_curve_by_photo_error_cnt > 0:
        photo_error_masks = get_static_render_error_mask(
            s2d,
            log_path,
            render_error_th=getattr(fit_cfg, "refilter_curve_by_photo_error_th", 0.1),
        )
        # fetch the mask and count how many fg will each curve lies when its valid
        cnt = torch.zeros(curve_mask.shape[1], device=device)
        assert curve_uv.shape[:2] == curve_mask.shape
        for _t in range(len(curve_mask)):
            _uv = curve_uv[_t]
            _int_uv, _inside_mask = round_int_coordinates(_uv, s2d.H, s2d.W)
            _cnt = query_image_buffer_by_pix_int_coord(
                photo_error_masks[_t].clone(), _int_uv
            )
            cnt = cnt + (_inside_mask * curve_mask[_t] * _cnt).long()
        refilter_valid_curve_mask = cnt >= refilter_curve_by_photo_error_cnt
        logging.info(
            f"Photo refilter {(~refilter_valid_curve_mask).sum()} curves with th={refilter_curve_by_photo_error_cnt} cnt={cnt.max()}"
        )
        curve_xyz = curve_xyz[:, refilter_valid_curve_mask]
        curve_mask = curve_mask[:, refilter_valid_curve_mask]
        curve_rgb = curve_rgb[refilter_valid_curve_mask]
        curve_filter_mask[curve_filter_mask.clone()] = refilter_valid_curve_mask
        curve_uv = curve_uv[:, refilter_valid_curve_mask]

    viz_mosca_curves_before_optim(curve_xyz, curve_rgb, curve_mask, cams, log_path)

    # * get scaffold
    scaffold: MoSca = MoSca(
        node_xyz=curve_xyz.detach().clone(),
        node_certain=curve_mask,
        t_list=sub_t_list,
        spatial_unit_factor=getattr(fit_cfg, "mosca_unit_auto_factor", 1.0),
        spatial_unit_hard_set=getattr(
            fit_cfg, "mosca_unit_world", 0.02 * s2d.scale_nw
        ),  # ! SET NEGATIVE IF WANT TO USE AUTO
        sigma_init_ratio=getattr(fit_cfg, "mosca_sigma_init_ratio", 5.0),
        sigma_max_ratio=getattr(fit_cfg, "mosca_sigma_max_ratio", 10.0),
        topo_dist_top_k=getattr(fit_cfg, "mosca_dist_k", 3),
        topo_th_ratio=getattr(fit_cfg, "mosca_topo_th_ratio", 5.0),
        topo_sample_T=getattr(fit_cfg, "mosca_topo_sample_T", 100),
        skinning_k=getattr(fit_cfg, "mosca_skinning_k", 16),
        skinning_method=getattr(fit_cfg, "mosca_skinning_method", "dqb"),
        mlevel_list=getattr(fit_cfg, "mosca_mlevel_list", [1, 7, 15]),
        mlevel_k_list=getattr(fit_cfg, "mosca_mlevel_k_list", [16, 8, 8]),
        mlevel_w_list=getattr(fit_cfg, "mosca_mlevel_w_list", [0.4, 0.3, 0.3]),
        mlevel_detach_nn_flag=getattr(
            fit_cfg, "mosca_mlevel_detach_nn_flag", True
        ),  # ! this should be False but due to the old code behavior, set default to True to align with the submission version.
        mlevel_detach_self_flag=getattr(
            fit_cfg, "mosca_mlevel_detach_self_flag", False
        ),
        #
        w_corr_maintain_sum_flag=getattr(
            fit_cfg, "mosca_w_corr_maintain_sum_flag", False
        ),
        # node_grouping=curve_group_id if s2d.has_vos else None,
        # break_topo_between_group=False,  # ! dycheck one body has multiple seg, which is not good here
    )
    scaffold.compute_rotation_from_xyz()
    if getattr(fit_cfg, "mosca_resample_flag", True):
        sampled_inds = scaffold.resample_node(resample_factor=1.0, use_mask=True)
    else:
        logging.warning("Not resampling the scaffold")
        sampled_inds = torch.arange(scaffold.M).to(device)
    node_rgb = curve_rgb[sampled_inds]

    logging.info(
        f"MoSca: get scaffold with M={scaffold.M} and unit={scaffold.spatial_unit}"
    )

    logging.info("*" * 20 + "MoSca Geo" + "*" * 20)
    # * Optimize the curve with ARAP
    assert (
        getattr(fit_cfg, "geo_mosca_use_mask_topo", True) or s2d.track.shape[-1] == 3
    ), "Must use mask topo for 2D tracks"
    if getattr(fit_cfg, "geo_mosca_steps", 1500) > 0:
        scaffold = geometry_scf_init(
            viz_dir=osp.join(log_path, "mosca"),
            log_dir=osp.join(log_path, "mosca"),
            scf=scaffold,
            cams=cams,
            lr_q=getattr(fit_cfg, "geo_mosca_lr_q", 0.03),
            lr_p=getattr(fit_cfg, "geo_mosca_lr_p", 0.03),
            lr_sig=0.0,
            total_steps=getattr(fit_cfg, "geo_mosca_steps", 1500),
            max_time_window=cams.T + 1,
            temporal_diff_shift=getattr(fit_cfg, "geo_temporal_diff_shift", [2, 8, 16]),
            temporal_diff_weight=getattr(
                fit_cfg, "geo_temporal_diff_weight", [0.6, 0.4, 0.3]
            ),
            lambda_local_coord=getattr(fit_cfg, "geo_mosca_lambda_local_coord", 1.0),
            lambda_metric_len=getattr(fit_cfg, "geo_mosca_lambda_metric_len", 1.0),
            lambda_xyz_acc=getattr(fit_cfg, "geo_mosca_lambda_xyz_acc", 0.3),
            lambda_q_acc=getattr(fit_cfg, "geo_mosca_lambda_q_acc", 0.1),
            lambda_xyz_vel=getattr(fit_cfg, "geo_mosca_lambda_xyz_vel", 0.3),
            lambda_q_vel=getattr(fit_cfg, "geo_mosca_lambda_q_vel", 0.1),
            mlevel_resample_steps=getattr(fit_cfg, "geo_mosca_resample_steps", 100),
            update_full_topo=False,
            # use_mask_topo=True,  # ! must set true for 2D tracks
            use_mask_topo=getattr(fit_cfg, "geo_mosca_use_mask_topo", True),
            update_all_topo_steps=getattr(
                fit_cfg, "geo_mosca_update_all_topo_steps", []
            ),
            reline_steps=getattr(fit_cfg, "geo_mosca_reline_steps", []),
            decay_start=getattr(fit_cfg, "geo_mosca_decay_steps", 500),
            decay_factor=getattr(fit_cfg, "geo_mosca_decay_factor", 30.0),
            viz_debug_interval=getattr(fit_cfg, "geo_mosca_viz_debug_interval", -1),
            viz_interval=getattr(fit_cfg, "geo_mosca_viz_interval", -1),
            viz_node_rgb=node_rgb,
            viz_level_flag=getattr(fit_cfg, "geo_mosca_viz_level_flag", True),
        )
    viz_list = viz_list_of_colored_points_in_cam_frame(
        [cams.trans_pts_to_cam(t, it).cpu() for t, it in enumerate(scaffold._node_xyz)],
        node_rgb,
        zoom_out_factor=1.0,
    )
    imageio.mimsave(osp.join(log_path, "cam_curve_optimized.gif"), viz_list, loop=1000)

    # resampled time!
    if SCF_GEO_KEYFRAME_RATE > 1:
        fulltime_curve_mask = s2d.track_mask.detach().clone()[
            :, s2d.dynamic_track_mask
        ][:, curve_filter_mask][:, sampled_inds]
        scaffold.resample_time(
            new_tids=torch.arange(cams.T), new_node_certain=fulltime_curve_mask
        )
    os.makedirs(osp.join(log_path, "mosca"), exist_ok=True)
    torch.save(scaffold.state_dict(), osp.join(log_path, "mosca", "mosca.pth"))

    return s2d


def photometric_reconstruct(ws, log_path, fit_cfg):
    seed_everything(SEED)
    DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg)
    DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th", 1.0)
    DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0)

    EPI_TH = getattr(fit_cfg, "epi_th", 1e-3)
    DYN_ID_CNT = getattr(fit_cfg, "dyn_id_cnt", 2 * 4)

    STATIC_GS_START_OPA = getattr(fit_cfg, "gs_static_start_opacity", 0.01)
    DYNAMIC_GS_START_OPA = getattr(fit_cfg, "gs_dynamic_start_opacity", 0.02)

    PHOTO_STATIC_WARM_STEPS = getattr(fit_cfg, "photo_static_warm_steps", -1)

    device = torch.device("cuda:0")

    # load solved camera and s2d and rescale
    s2d = (
        Saved2D(ws)
        .load_epi()
        .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=DEP_MEDIAN)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(
            TAP_MODE, min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4)
        )
        .rescale_perframe_depth_from_bundle(
            bundle_pth_fn=osp.join(log_path, "bundle", "bundle.pth")
        )
        .load_vos()
        .load_flow()
        .to(device)
    )
    track_identification = np.load(osp.join(log_path, "track_identification.npz"))
    s2d.register_track_indentification(
        torch.from_numpy(track_identification["static_track_mask"]).to(device),
        torch.from_numpy(track_identification["dynamic_track_mask"]).to(device),
    )

    cams: MonocularCameras = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(log_path, "bundle", "bundle_cams.pth"))
    ).to(device)
    scaffold = MoSca.load_from_ckpt(
        torch.load(osp.join(log_path, "mosca", "mosca.pth"))
    ).to(device)

    # * reset the scaffold mlevel config
    scaffold.set_multi_level(
        mlevel_arap_flag=True,
        mlevel_list=getattr(fit_cfg, "photo_mlevel_list", [1, 6]),
        mlevel_k_list=getattr(fit_cfg, "photo_mlevel_k_list", [16, 8]),
        mlevel_w_list=getattr(fit_cfg, "photo_mlevel_w_list", [0.4, 0.3]),
    )

    # construct the GS models
    photo_solver = DynReconstructionSolver(
        working_dir=log_path,
        device=device,
        radius_init_factor=getattr(fit_cfg, "gs_radius_init_factor", 4.0),
        opacity_init_factor=getattr(fit_cfg, "gs_opacity_init_factor", 0.95),
    )
    # ! warning, this mask is only useful for constructing the model.
    photo_solver.identify_fg_mask_by_nearest_curve(
        s2d, cams, "gs_model_construct_fg_mask.mp4"
    )
    if GS_BACKEND == "gof":
        photo_solver.compute_normals_for_s2d(
            s2d,
            cams,
            patch_size=7,
            nn_dist_th=0.03,
            nn_min_cnt=4,
            viz_fn="gs_model_construct_normal.mp4",
        )
    include_fg_in_static = getattr(fit_cfg, "gs_include_fg_in_static", True)
    s_model_warmup_path = osp.join(
        log_path, f"static_warm_s_model_{GS_BACKEND.lower()}.pth"
    )
    if osp.exists(s_model_warmup_path):
        logging.info(f"Load static model from {s_model_warmup_path}")
        s_model = StaticGaussian.load_from_ckpt(torch.load(s_model_warmup_path)).to(
            device
        )
    else:
        s_model = photo_solver.get_static_model(
            s2d=s2d,
            cams=cams,
            n_init=getattr(fit_cfg, "gs_static_n_init", 30000),
            radius_max=getattr(fit_cfg, "gs_radius_max", 0.1),
            max_sph_order=getattr(fit_cfg, "gs_max_sph_order", 0),
            mask_type="depth" if include_fg_in_static else "static_depth",
        )

    if getattr(fit_cfg, "photo_d_model_fetch_only_error_th", -1) > 0:
        photo_data = np.load(osp.join(log_path, "photo_warmup_rendered.npz"))
        render_error_th = getattr(fit_cfg, "photo_d_model_fetch_only_error_th", -1)
        rgb_rendered = torch.tensor(photo_data["rgb"]).to(device).permute(0, 2, 3, 1)
        render_error = abs(s2d.rgb - rgb_rendered).max(dim=-1).values
        render_error_mask = render_error > render_error_th
        render_error_mask_viz = (render_error_mask[..., None] * s2d.rgb).cpu().numpy()
        imageio.mimsave(
            osp.join(
                log_path,
                f"d_model_fetch_mask_render_error_mask_th={render_error_th:.3f}.gif",
            ),
            (render_error_mask_viz * 255).astype(np.uint8),
        )
        d_model_add_mask = render_error_mask
    else:
        d_model_add_mask = None

    d_model = photo_solver.get_dynamic_model(
        s2d=s2d,
        cams=cams,
        scf=scaffold,
        n_init=getattr(fit_cfg, "gs_dynamic_n_init", 30000),
        radius_max=getattr(fit_cfg, "gs_radius_max", 0.1),
        max_sph_order=getattr(fit_cfg, "gs_max_sph_order", 0),
        leaf_local_flag=getattr(fit_cfg, "gs_leaf_local_flag", True),
        additional_mask=d_model_add_mask,
        nn_fusion=getattr(fit_cfg, "gs_nn_fusion", -1),
        # ! below is set to dyn_gs_model becaues it controls the densification
        max_node_num=getattr(fit_cfg, "gs_max_node_num", 100000),
    )
    with torch.no_grad():
        if DYNAMIC_GS_START_OPA > 0:
            d_model._opacity.data = d_model.o_inv_act(
                torch.min(
                    d_model.o_act(d_model._opacity),
                    torch.ones_like(d_model._opacity) * DYNAMIC_GS_START_OPA,
                )
            )
        if STATIC_GS_START_OPA > 0:
            s_model._opacity.data = s_model.o_inv_act(
                torch.min(
                    s_model.o_act(s_model._opacity),
                    torch.ones_like(s_model._opacity) * STATIC_GS_START_OPA,
                )
            )

    photo_solver.photometric_fit(
        s2d=s2d,
        total_steps=getattr(fit_cfg, "photo_total_steps", 6000),
        optim_cam_after_steps=getattr(fit_cfg, "photo_optim_cam_after_steps", 0),
        decay_start=getattr(fit_cfg, "photo_decay_start", 2000),
        skinning_corr_start_steps=getattr(
            fit_cfg, "photo_skinning_corr_start_steps", 10000000000
        ),
        cams=cams,
        s_model=s_model,
        d_model=d_model,
        # losses
        lambda_rgb=getattr(fit_cfg, "photo_lambda_rgb", 1.0),
        lambda_dep=getattr(fit_cfg, "photo_lambda_dep", 0.1),
        lambda_mask=getattr(fit_cfg, "photo_lambda_mask", 0.0),
        dep_st_invariant=getattr(fit_cfg, "photo_dep_st_invariant", True),
        lambda_normal=getattr(fit_cfg, "photo_lambda_normal", 0.05),
        lambda_depth_normal=getattr(fit_cfg, "photo_lambda_depth_normal", 0.05),
        lambda_distortion=getattr(fit_cfg, "photo_lambda_distortion", 100.0),
        lambda_vel_xyz_reg=getattr(fit_cfg, "photo_lambda_vel_xyz_reg", 5.0),
        lambda_vel_rot_reg=getattr(fit_cfg, "photo_lambda_vel_rot_reg", 5.0),
        lambda_acc_rot_reg=getattr(fit_cfg, "photo_lambda_acc_rot_reg", 5.0),
        lambda_acc_xyz_reg=getattr(fit_cfg, "photo_lambda_acc_xyz_reg", 5.0),
        lambda_arap_coord=getattr(fit_cfg, "photo_lambda_arap_coord", 10.0),
        lambda_arap_len=getattr(fit_cfg, "photo_lambda_arap_len", 10.0),
        lambda_small_w_reg=getattr(fit_cfg, "photo_lambda_small_w_reg", 0.0),
        # track loss
        lambda_track=getattr(fit_cfg, "photo_lambda_track", 0.0),
        track_flow_chance=getattr(fit_cfg, "photo_track_flow_chance", 0.0),
        track_flow_interval_candidates=getattr(
            fit_cfg, "photo_track_flow_interval_candidates", [1, 3]
        ),
        track_loss_interval=getattr(fit_cfg, "photo_track_loss_interval", 3),
        track_loss_start_step=getattr(fit_cfg, "photo_track_loss_start_step", -1),
        track_loss_end_step=getattr(fit_cfg, "photo_track_loss_end_step", 100000),
        temporal_diff_shift=getattr(
            fit_cfg, "photo_temporal_diff_shift", [1, 3, 6]
        ),  # ! warning, if not set, the default value between geo and photo are different
        temporal_diff_weight=getattr(
            fit_cfg, "photo_temporal_diff_weight", [0.6, 0.3, 0.1]
        ),
        geo_reg_start_steps=getattr(fit_cfg, "photo_geo_reg_start_steps", 0),
        optimizer_cfg=OptimCFG(
            lr_cam_f=0.0,
            lr_cam_q=0.00003,
            lr_cam_t=0.00003,
            # gs
            lr_p=getattr(fit_cfg, "photo_lr_p", 0.00016),
            lr_q=getattr(fit_cfg, "photo_lr_q", 0.001),
            lr_s=getattr(fit_cfg, "photo_lr_s", 0.005),
            lr_o=getattr(fit_cfg, "photo_lr_o", 0.05),
            lr_sph=getattr(fit_cfg, "photo_lr_sph", 0.0025),
            lr_sph_rest_factor=getattr(fit_cfg, "photo_lr_sph_rest_factor", 20.0),
            lr_p_final=getattr(fit_cfg, "photo_lr_p_final", 0.00016 / 100),
            # node
            lr_np=getattr(fit_cfg, "photo_lr_np", 0.00016),
            lr_nq=getattr(fit_cfg, "photo_lr_nq", 0.00016),
            lr_nsig=getattr(fit_cfg, "photo_lr_nsig", 0.003),
            lr_np_final=getattr(fit_cfg, "photo_lr_np_final", 0.00016 / 100.0),
            lr_nq_final=getattr(fit_cfg, "photo_lr_nq_final", 0.00016 / 100.0),
            lr_w=getattr(fit_cfg, "photo_lr_w", 0.0),
            lr_w_final=getattr(fit_cfg, "photo_lr_w_final", None),
        ),
        d_gs_ctrl_cfg=GSControlCFG(
            densify_steps=getattr(fit_cfg, "photo_d_ctrl_densify_steps", 200),
            reset_steps=getattr(fit_cfg, "photo_d_ctrl_reset_steps", 1001),
            prune_steps=getattr(fit_cfg, "photo_d_ctrl_prune_steps", 200),
            densify_max_grad=getattr(fit_cfg, "photo_d_ctrl_densify_max_grad", 0.0002),
            densify_percent_dense=getattr(
                fit_cfg, "photo_d_ctrl_densify_percent_dense", 0.01
            ),
            prune_opacity_th=getattr(fit_cfg, "photo_d_ctrl_prune_opacity_th", 0.05),
            reset_opacity=getattr(fit_cfg, "photo_d_ctrl_reset_opacity", 0.01),
        ),
        s_gs_ctrl_cfg=GSControlCFG(
            densify_steps=getattr(fit_cfg, "photo_s_ctrl_densify_steps", 400),
            reset_steps=getattr(fit_cfg, "photo_s_ctrl_reset_steps", 1001),
            prune_steps=getattr(fit_cfg, "photo_s_ctrl_prune_steps", 200),
            # densify_max_grad=getattr(fit_cfg.s_ctrl, "densify_max_grad", 0.0006),
            densify_max_grad=getattr(
                fit_cfg, "photo_s_ctrl_densify_max_grad", 0.0002
            ),  # ! changed here
            densify_percent_dense=getattr(
                fit_cfg, "photo_s_ctrl_densify_percent_dense", 0.01
            ),
            prune_opacity_th=getattr(fit_cfg, "photo_s_ctrl_prune_opacity_th", 0.05),
            reset_opacity=getattr(fit_cfg, "photo_s_ctrl_reset_opacity", 0.01),
        ),
        s_gs_ctrl_start_ratio=getattr(fit_cfg, "photo_s_ctrl_start_ratio", 0.01),
        d_gs_ctrl_start_ratio=getattr(fit_cfg, "photo_d_ctrl_start_ratio", 0.01),
        s_gs_ctrl_end_ratio=getattr(fit_cfg, "photo_s_ctrl_end_ratio", 0.9),
        d_gs_ctrl_end_ratio=getattr(fit_cfg, "photo_d_ctrl_end_ratio", 0.9),
        # NODE CONTROL
        dyn_error_grow_steps=getattr(fit_cfg, "photo_dyn_error_grow_steps", []),
        dyn_error_grow_th=getattr(fit_cfg, "photo_dyn_error_grow_th", 0.2),
        dyn_error_grow_num_frames=getattr(
            fit_cfg, "photo_dyn_error_grow_num_frames", 4
        ),
        dyn_node_densify_steps=getattr(fit_cfg, "photo_dyn_node_densify_steps", []),
        dyn_node_densify_grad_th=getattr(
            fit_cfg, "photo_dyn_node_densify_grad_th", 0.2
        ),
        dyn_node_densify_record_start_steps=getattr(
            fit_cfg, "photo_dyn_node_densify_record_start_steps", 2000
        ),
        dyn_node_densify_max_gs_per_new_node=getattr(
            fit_cfg, "photo_dyn_node_densify_max_gs_per_new_node", 100000
        ),
        # Dyn clean
        photo_s2d_trans_steps=getattr(fit_cfg, "photo_s2d_trans_steps", []),
        # SCF clean
        dyn_scf_prune_steps=getattr(fit_cfg, "photo_dyn_scf_prune_steps", []),
        dyn_scf_prune_sk_th=getattr(fit_cfg, "photo_dyn_scf_prune_sk_th", 0.02),
        ##################################################
        # viz
        viz_skip_t=1 if cams.T < 120 else max(1, cams.T // 50),
        viz_interval=getattr(fit_cfg, "photo_viz_interval", -1),
        viz_cheap_interval=getattr(fit_cfg, "photo_viz_cheap_interval", -1),
        viz_move_angle_deg=getattr(fit_cfg, "photo_viz_move_angle_deg", 10.0),
        random_bg=getattr(fit_cfg, "photo_random_bg", True),
        default_bg_color=getattr(fit_cfg, "photo_default_bg_color", [1.0, 1.0, 1.0]),
    )
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoSca-V2 Reconstruction")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument("--no_viz", action="store_true", help="no viz")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    logdir = setup_recon_ws(args.ws, fit_cfg=cfg)

    # * RUN
    static_reconstruct(args.ws, logdir, cfg)
    photometric_warmup(
        args.ws, logdir, cfg
    )  # this is optional, if not set, will directly skip and return.
    scaffold_reconstruct(args.ws, logdir, cfg)
    photometric_reconstruct(args.ws, logdir, cfg)

    # * EVAL AND VIZ
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

    if datamode in ["iphone"]:
        try:
            seq_name = osp.basename(args.ws)
            test_pck(
                saved_dir=logdir,
                gt_npz_fn=f"./eval_utils/pck_gt_packs/{seq_name}_train_pck.npz",
                device=torch.device("cuda"),
                save_fn=osp.join(logdir, "pck5.txt"),
            )
        except:
            logging.warning("PCK5 failed")
            pass

    test_fps(saved_dir=logdir, rounds=1 if datamode in ["iphone"] else 3)

    if datamode in ["iphone", "nvidia"]:
        test_main(
            cfg,
            saved_dir=logdir,
            data_root=args.ws,
            device=torch.device("cuda"),
            tto_flag=True,
            eval_also_dyncheck_non_masked=False,
            skip_test_gen=False,
        )

    if not args.no_viz and datamode in ["wild"]:
        from mosca_viz import viz_main

        viz_main(
            save_dir=osp.join(logdir, "viz"),
            log_dir=logdir,
            cfg_fn=args.cfg,
            N=getattr(cfg, "viz_N", 5),
            move_angle_deg=getattr(cfg, "viz_move_angle_deg", 10.0),
            H_3d=getattr(cfg, "viz_H_3d", 960),
            W_3d=getattr(cfg, "viz_W_3d", 960),
            fov_3d=getattr(cfg, "viz_fov_3d", 70),
            back_ratio_3d=getattr(cfg, "viz_back_ratio_3d", 1.5),
            up_ratio=getattr(cfg, "viz_up_ratio", 0.05),
            bg_color=getattr(cfg, "photo_default_bg_color", [0.0, 0.0, 0.0]),
        )
