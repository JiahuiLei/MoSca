import os, sys
import os.path as osp
import torch
import numpy as np
import imageio, cv2
import logging

sys.path.append(osp.abspath(osp.dirname(__file__)))

from epi_helpers import analyze_track_epi, identify_tracks
from intrinsic_helpers import find_initinsic
from intrinsic_helpers import compute_graph_energy, track2undistroed_homo
from camera import MonocularCameras
from bundle import (
    compute_static_ba,
    query_buffers_by_track,
    prepare_track_homo_dep_rgb_buffers,
)
from moca_misc import make_pair_list, Rt2T, configure_logging, get_all_world_pts_list

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))

from lib_prior.prior_loading import Saved2D


@torch.no_grad()
def rescale_camera_pose(
    s2d, cams, s_track, s_track_mask, robust_alignment_jitter_th_ratio=2.0
):
    device = s2d.dep.device
    homo_list, dep_list, rgb_list = prepare_track_homo_dep_rgb_buffers(
        s2d, s_track[...,:2], s_track_mask, torch.arange(s2d.T).to(device)
    )
    # ! this jitter removal is super important for robustly scale the Nvidia camera back to a correct scale!!
    # mask out large depth jitter
    neighbor_frame_mask = s_track_mask[1:] * s_track_mask[:-1]
    depth_diff = abs(dep_list[1:] - dep_list[:-1])
    large_depth_jitter_th = depth_diff[neighbor_frame_mask].median()
    jitter_mask = (
        depth_diff > large_depth_jitter_th * robust_alignment_jitter_th_ratio
    ) * neighbor_frame_mask
    logging.info(
        f"When solving optimal cam scale, ignore {jitter_mask.sum() / neighbor_frame_mask.sum()*100.0:.2f}% potential jitters"
    )
    neighbor_frame_mask = neighbor_frame_mask * (~jitter_mask)

    T, M = homo_list.shape[:2]
    point_cam = cams.backproject(homo_list.reshape(-1, 2), dep_list.reshape(-1))
    point_cam = point_cam.reshape(T, M, 3)
    R_wc, t_wc = cams.Rt_wc_list()
    point_ref_rot = torch.einsum("tij,tmj->tmi", R_wc, point_cam)
    point_ref = point_ref_rot + t_wc[:, None]
    # the optimal scale has a closed form solution from a quadratic form

    # we only consider the neighboring two frames for now!
    a = point_ref_rot[1:] - point_ref_rot[:-1]
    b = (t_wc[1:] - t_wc[:-1])[:, None].expand(-1, M, -1)
    a, b = a[neighbor_frame_mask], b[neighbor_frame_mask]
    # should be masked
    s_optimal = float(-(a * b).sum() / (b * b).sum())
    # avoid singular case
    if s_optimal < 0.00001:
        logging.warning(
            f"optimal rescaling of gt camera translation degenerate to {s_optimal}, use 1.0 instead!"
        )
        s_optimal = 1.0
    logging.info(
        f"Rescale the GT camera pose ot our depth scale (median=1) with a global scale factor {s_optimal} by closed form solution"
    )
    cams.t_wc.data = cams.t_wc.data * s_optimal
    return cams


def moca_solve(
    ws,
    s2d: Saved2D,
    device=torch.device("cuda:0"),
    # EPI
    epi_th=1e-4,
    # FOV
    fov_search_intervals=[10, 20],
    fov_min_valid_covalid=64,  # 512,
    fov_search_fallback=53.0,
    fov_search_N=100,
    fov_search_start=30.0,
    fov_search_end=90.0,
    init_cam_with_optimal_fov_results=True,
    # BA
    ba_total_steps=1000,
    ba_switch_to_ind_step=500,
    ba_depth_correction_after_step=500,
    ba_max_frames_per_step=200,
    ba_lr_cam_q=0.0003,
    ba_lr_cam_t=0.0003,
    ba_lr_cam_f=0.0003,
    ba_lr_dep_s=0.001,
    ba_lr_dep_c=0.001,
    ba_lambda_flow=1.0,
    ba_lambda_depth=0.1,
    ba_lambda_small_correction=0.03,
    ba_lambda_cam_smooth_trans=0.0,
    ba_lambda_cam_smooth_rot=0.0,
    # input a camera instance
    gt_cam=None,
    iso_focal=False,
    rescale_gt_cam_transl=False,
    static_id_mode="raft",
    # filter the tracks
    depth_filter_th=-1.0,
    depth_filter_min_cnt=4,  # after filter, if valid < cnt, remove
    # robust
    robust_huber_delta=-1,
    # ! warning, the robust th and sigma are factors with respect to the depth median
    robust_depth_decay_th=2.0,
    robust_depth_decay_sigma=1.0,
    robust_std_decay_th=0.2,
    robust_std_decay_sigma=0.2,
    # viz
    viz_valid_ba_points=False,
):
    configure_logging(osp.join(ws, "moca_solve.log"), debug=False)
    torch.cuda.empty_cache()
    assert static_id_mode in ["raft", "track"], f"{static_id_mode} not in [raft, track]"
    s2d.to(device)
    H, W, T = s2d.H, s2d.W, s2d.T

    track = s2d.track.cpu()
    track_mask = s2d.track_mask.cpu()
    continuous_pair_list = make_pair_list(T, interval=[1], dense_flag=True)

    depth_median = float(s2d.dep[s2d.dep_mask].median())
    logging.warning(
        f"All robust decay th and sigma are factors w.r.t the depth median, rescale them with median={depth_median:.2f}"
    )
    robust_depth_decay_th = robust_depth_decay_th * depth_median
    robust_depth_decay_sigma = robust_depth_decay_sigma * depth_median
    robust_std_decay_th = robust_std_decay_th * depth_median
    robust_std_decay_sigma = robust_std_decay_sigma * depth_median

    # * 1. mark static track
    if static_id_mode == "raft":
        logging.info(f"Use pre-computed 2D epi error to mark static tracks")
        # * mark by collecting epi error
        # collect the epi error from the pre-compute 2D epi from raft
        raft_epi = s2d.epi.clone()
        with torch.no_grad():
            epierr_list = query_buffers_by_track(raft_epi[..., None], track, track_mask)
            epierr_list = epierr_list.squeeze(-1).cpu()
    elif static_id_mode == "track":
        logging.info(f"Analyze the track epi to mark static tracks")
        # * mark by compute all epi for neighboring pairs and
        # first call for neighbor pairs, solve the epi, and get F, so later can use to initialize pose
        F_list, epierr_list, _ = analyze_track_epi(
            continuous_pair_list, s2d.track, s2d.track_mask, H=s2d.H, W=s2d.W
        )
        np.savez(
            osp.join(ws, "tracker_epi.npz"),
            continuous_pair_list=continuous_pair_list,
            F_list=F_list,
        )
    track_static_selection, track_dynamic_selection = identify_tracks(
        epierr_list, epi_th
    )

    s2d.register_track_indentification(track_static_selection, track_dynamic_selection)
    sta_track = track[:, track_static_selection]
    sta_track_mask = track_mask[:, track_static_selection]

    # if sta_track.shape[-1] == 3:
    #     logging.info(f"SpaT mode, direct use 3D Track")
    #     sta_track_dep = sta_track[..., -1].clone()
    # else:
    # ! looks like the spatracker depth is not reliable
    sta_track_dep = query_buffers_by_track(
        s2d.dep[..., None], sta_track, sta_track_mask
    ).to(sta_track_mask.device)

    assert not torch.isinf(sta_track_dep[sta_track_mask]).any()
    assert not torch.isnan(sta_track_dep[sta_track_mask]).any()

    # * verify the static track maximum visible depth and filter out
    if depth_filter_th > 0:
        logging.warning(
            f"MoCa BA set to filter out very far tracks with th={depth_filter_th}"
        )
        sta_track_dep = query_buffers_by_track(
            s2d.dep[..., None], sta_track, sta_track_mask
        ).to(sta_track_mask.device)
        invalid_mask = sta_track_dep.squeeze(-1) > depth_filter_th
        sta_track_mask = sta_track_mask * ~invalid_mask
        # remove the tracks if there is not enough valid
        valid_cnt = sta_track_mask.sum(dim=0)
        filter_mask = valid_cnt >= depth_filter_min_cnt
        # filter
        sta_track = sta_track[:, filter_mask]
        sta_track_mask = sta_track_mask[:, filter_mask]
        sta_track_dep = sta_track_dep[:, filter_mask]

    if gt_cam is None:
        # * 2. compute also for later fov inlier mask
        # rerun the epi analysis for later FOV init, with the pairs that count the static common visible mask
        fov_jump_pair_list = make_pair_list(
            T,
            interval=fov_search_intervals,
            dense_flag=True,
            track_mask=sta_track_mask,
            min_valid_num=fov_min_valid_covalid,
        )
        assert len(fov_jump_pair_list) > 0, f"no valid pair for FOV search"
        logging.info(f"Start analyzing {len(fov_jump_pair_list)} pairs for FOV serach")
        _, _, inlier_list_jumped = analyze_track_epi(
            fov_jump_pair_list, sta_track, sta_track_mask, H=s2d.H, W=s2d.W
        )
        checked_pair, checked_inlier = [], []
        for pair, inlier in zip(fov_jump_pair_list, inlier_list_jumped):
            if inlier.sum() > fov_min_valid_covalid:
                checked_pair.append(pair)
                checked_inlier.append(inlier)
        # collect the robsut mask inside the static
        fov_jump_pair_list = checked_pair
        inlier_list_jumped = torch.stack(checked_inlier, 0)

        # * 3. compute FOV
        optimal_fov = find_initinsic(
            H=s2d.H,
            W=s2d.W,
            pair_list=fov_jump_pair_list,
            pair_mask_list=inlier_list_jumped.to(
                device
            ),  # ! use the inlier mask to find the optimal fov
            track=sta_track.to(device),
            dep_list=sta_track_dep.to(device),
            viz_fn=osp.join(ws, "fov_search.jpg"),
            fallback_fov=fov_search_fallback,
            search_N=fov_search_N,
            search_start=fov_search_start,
            search_end=fov_search_end,
            depth_decay_th=robust_depth_decay_th,
            depth_decay_sigma=robust_depth_decay_th,
        )
        # solve the neighboring pair list as well
        E, E_i, opt_s_ij, opt_R_ij, opt_t_ij = compute_graph_energy(
            optimal_fov,
            continuous_pair_list,
            sta_track_mask[[it[0] for it in continuous_pair_list]]
            * sta_track_mask[[it[1] for it in continuous_pair_list]],
            track2undistroed_homo(sta_track, H, W),
            sta_track_dep,
            depth_decay_th=robust_depth_decay_th,
            depth_decay_sigma=robust_depth_decay_sigma,
        )  # ! this is in metric space

        # * 4. prepare camra
        # todo: ! warning, the scale is ignored during initalziation, let the BA to solve this
        cams = MonocularCameras(
            s2d.T,
            s2d.H,
            s2d.W,
            [optimal_fov, optimal_fov, 0.5, 0.5],
            delta_flag=True,
            init_camera_pose=(
                Rt2T(opt_R_ij, opt_t_ij) if init_cam_with_optimal_fov_results else None
            ),
            iso_focal=iso_focal,
        ).to(device)
    else:
        logging.info(
            f"MoCa solver use passed in GT cam as initialziaiton to start optimization"
        )

        # todo: rescale the camera

        cams = gt_cam
        if rescale_gt_cam_transl:
            logging.info(
                "Sometimes the GT pose and the metric depth are not in the same scale, rescale the camera"
            )
            cams = rescale_camera_pose(
                s2d, cams, s_track=sta_track, s_track_mask=sta_track_mask
            )
        cams.to(device)

    # * 5. bundle adjustment
    if ba_total_steps > 0:
        compute_static_ba(
            max_t_per_step=ba_max_frames_per_step,
            s2d=s2d,
            log_dir=osp.join(ws, "bundle"),
            s_track=sta_track.to(device),
            s_track_valid_mask=sta_track_mask.to(device),
            cams=cams,
            total_steps=ba_total_steps,
            switch_to_ind_step=ba_switch_to_ind_step,
            depth_correction_after_step=ba_depth_correction_after_step,
            lr_cam_q=ba_lr_cam_q,
            lr_cam_t=ba_lr_cam_t,
            lr_cam_f=ba_lr_cam_f,
            lr_dep_s=ba_lr_dep_s,
            lr_dep_c=ba_lr_dep_c,
            lambda_flow=ba_lambda_flow,
            lambda_depth=ba_lambda_depth,
            lambda_small_correction=ba_lambda_small_correction,
            lambda_cam_smooth_trans=ba_lambda_cam_smooth_trans,
            lambda_cam_smooth_rot=ba_lambda_cam_smooth_rot,
            #
            huber_delta=robust_huber_delta,
            depth_decay_th=robust_depth_decay_th,
            std_decay_th=robust_std_decay_th,
            std_decay_sigma=robust_std_decay_sigma,
            #
            viz_video_rgb=(
                (s2d.rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                if viz_valid_ba_points
                else None
            ),
        )
    torch.cuda.empty_cache()

    # * 6 finish
    viz_all_pts = get_all_world_pts_list(
        s2d.homo_map, s2d.dep, s2d.rgb, s2d.dep_mask, cams
    )
    viz_all_pts = torch.cat(viz_all_pts, 0).numpy()
    viz_all_sel = np.random.choice(len(viz_all_pts), 50_000, replace=False)
    viz_all_pts = viz_all_pts[viz_all_sel]
    viz_all_pts[:, 3:] = viz_all_pts[:, 3:] * 255
    print(viz_all_pts.shape)
    np.savetxt(osp.join(ws, "bundle", "all_pts.xyz"), viz_all_pts[:, :6], fmt="%.5f")

    return cams, s2d, track_static_selection
