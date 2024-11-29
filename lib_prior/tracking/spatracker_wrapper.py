import sys
import os, os.path as osp
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
import logging
import time
import glob
import imageio

sys.path.append(osp.dirname(osp.abspath(__file__)))
from cotracker_visualizer import Visualizer
from spatracker.spatracker.predictor import SpaTrackerPredictor
from tracking_utils import (
    seed_everything,
    convert_img_list_to_cotracker_input,
    get_sampling_mask,
    get_uniform_random_queries,
    load_epi_error,
    load_vos,
    viz_queries,
    viz_coverage,
    tracker_get_query_uv,
)


def get_spatracker(
    device,
    S_lenth=12,
    ckpt_path=osp.abspath(
        osp.join(osp.dirname(osp.abspath(__file__)), "../../weights", "spaT_final.pth")
    ),
):
    assert S_lenth in [8, 12, 16]  # [8, 12, 16] choose one you want
    model = SpaTrackerPredictor(
        checkpoint=ckpt_path, interp_shape=(384, 512), seq_length=S_lenth
    )
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def __infer_one_pass__(
    video,
    dep,
    queries,
    model: SpaTrackerPredictor,
    K=None,
    backward_tracking=False,
    visiblility_th=0.9,
):
    # video: 1,T,3,H,W, [0,255] with float
    # depth: T,1,H,W
    # queries: 1,N,3 t,x(W),y(H)
    torch.cuda.empty_cache()
    device = model.model.fnet.conv1.weight.device
    if device.type == "cuda":
        assert (
            device.index == 0
        ), "NOW SOFTSPLAT SEEMS HAVE BUG WHEN SETING DEVICE!=0, SO YOU MUST SET DEVICE OUTSIDE THE PYTHON SCRIPT AS ENV VAR INSTEAD OF SETTING INSIDE PYTHON CODE!"
    S_length = model.model.S

    assert (
        video.ndim == 5 and video.shape[0] == 1 and video.shape[2] == 3
    ), "video should have size: 1,T,3,H,W"
    assert dep.ndim == 4 and dep.shape[1] == 1, "depth should have size: T,1,H,W"
    assert dep.shape[0] == video.shape[1], "video and depth should have same length"
    assert (
        queries.ndim == 3 and queries.shape[0] == 1 and queries.shape[2] == 3
    ), "queries should have size: 1,N,3"
    if K is not None:
        K = torch.as_tensor(K).to(device)
        if K.ndim == 2:
            assert K.shape[0] == 3 and K.shape[1] == 3, "K should have size: 3,3"
            K = K[None].repeat(len(dep), 1, 1)
        elif K.ndim == 3:
            assert (
                K.shape[0] == len(dep) and K.shape[1] == 3 and K.shape[2] == 3
            ), "K should have size: T,3,3"
        else:
            raise ValueError("K should have size: 3,3 or T,3,3 or None")
        K = K[None]  # 1,T,3,3

    video = video.to(device)
    dep = dep.to(device)
    queries = queries.to(device)

    # # debug
    # from mde import MonoDEst
    # cfg = edict({"mde_name": "zoedepth_nk"})
    # MonoDEst_O = MonoDEst(cfg)
    # MonoDEst_M = MonoDEst_O.model
    # MonoDEst_M.to(device)
    # MonoDEst_M.eval()

    pred_tracks, pred_visibility, _, filter_id = model(
        video,
        queries=queries,
        video_depth=dep,
        wind_length=S_length,
        intrs=K,
        # depth_predictor=MonoDEst_M,
        backward_tracking=backward_tracking,
        thr=visiblility_th,
    )
    if filter_id is not None:
        filter_id = filter_id.cpu()
    torch.cuda.empty_cache()
    return pred_tracks.cpu(), pred_visibility.cpu(), filter_id


@torch.no_grad()
def infer_spa_tracker(video, dep, queries, model, K=None, visiblility_th=0.9):
    start_t = time.time()
    T = video.shape[1]

    torch.cuda.empty_cache()
    device = model.model.fnet.conv1.weight.device
    if device.type == "cuda":
        assert (
            device.index == 0
        ), "NOW SOFTSPLAT SEEMS HAVE BUG WHEN SETING DEVICE!=0, SO YOU MUST SET DEVICE OUTSIDE THE PYTHON SCRIPT AS ENV VAR INSTEAD OF SETTING INSIDE PYTHON CODE!"
    S_length = model.model.S

    assert (
        video.ndim == 5 and video.shape[0] == 1 and video.shape[2] == 3
    ), "video should have size: 1,T,3,H,W"
    assert dep.ndim == 4 and dep.shape[1] == 1, "depth should have size: T,1,H,W"
    assert dep.shape[0] == video.shape[1], "video and depth should have same length"
    assert (
        queries.ndim == 3 and queries.shape[0] == 1 and queries.shape[2] == 3
    ), "queries should have size: 1,N,3"
    if K is not None:
        K = torch.as_tensor(K).to(device).clone()
        if K.ndim == 2:
            assert K.shape[0] == 3 and K.shape[1] == 3, "K should have size: 3,3"
            K = K[None].repeat(len(dep), 1, 1)
        elif K.ndim == 3:
            assert (
                K.shape[0] == len(dep) and K.shape[1] == 3 and K.shape[2] == 3
            ), "K should have size: T,3,3"
        else:
            raise ValueError("K should have size: 3,3 or T,3,3 or None")
        K = K[None]  # 1,T,3,3

    video = video.to(device)
    dep = dep.to(device)
    queries = queries.to(device)

    pred_tracks, pred_visibility, filter_id = __infer_one_pass__(
        video.detach().clone(),
        dep.detach().clone(),
        queries.detach().clone(),
        model,
        K=K,
        backward_tracking=True,
        visiblility_th=visiblility_th,
    )
    pred_tracks = pred_tracks[0]  # T,N,3
    pred_visibility = pred_visibility[0]

    end_t = time.time()
    print(f"SpaT bi-directional time cost: {(end_t - start_t)/60.0:.3f} min")

    return pred_tracks, pred_visibility


@torch.no_grad()
def infer_spa_tracker_legacy_manual_bidirecitonal(video, dep, queries, model, K=None):
    # ! the old function infer the bi-directional tracking by hand, the new one use the built in bidirecitonal inference
    start_t = time.time()
    T = video.shape[1]

    # * forward
    logging.info("Forward tracking ...")
    pred_tracks_fwd, pred_visibility_fwd, filter_id_fwd = __infer_one_pass__(
        video, dep, queries, model, K=K
    )
    N = pred_tracks_fwd.shape[1]
    pred_tracks_fwd = pred_tracks_fwd[0]  # T,N,3
    pred_visibility_fwd = pred_visibility_fwd[0]  # T,N

    # * inverse manually
    logging.info("Backward tracking ...")
    video_inv = video.clone().flip(1)
    queries_inv = queries.clone()
    queries_inv[..., 0] = T - 1 - queries_inv[..., 0]
    dep_inv = dep.clone().flip(0)
    pred_tracks_bwd, pred_visibility_bwd, filter_id_bwd = __infer_one_pass__(
        video_inv, dep_inv, queries_inv, model, K=K
    )
    pred_tracks_bwd = pred_tracks_bwd.flip(1)[0]  # T,N,3
    pred_visibility_bwd = pred_visibility_bwd.flip(1)[0]  # T,N
    assert pred_tracks_bwd.shape == pred_tracks_fwd.shape
    if filter_id_bwd is not None:
        assert (filter_id_fwd == filter_id_bwd).all()

    # * fuse the forward and backward
    bwd_mask = torch.arange(T)[:, None] < queries[0, :, 0][None, :]  # T,N
    bwd_mask = bwd_mask[:, filter_id_bwd]

    pred_tracks = torch.where(bwd_mask[..., None], pred_tracks_bwd, pred_tracks_fwd)
    pred_visibility = torch.where(bwd_mask, pred_visibility_bwd, pred_visibility_fwd)

    end_t = time.time()
    print(f"SpaT bi-directional time cost: {(end_t - start_t)/60.0:.3f} min")

    return pred_tracks, pred_visibility


def make_spatracker_input(rgb_list, dep_list):
    # T,H,W,3; T,H,W
    assert rgb_list.ndim == 4 and rgb_list.shape[-1] == 3
    assert dep_list.ndim == 3
    assert len(rgb_list) == len(dep_list)

    input_video = (
        torch.from_numpy(rgb_list).permute(0, 3, 1, 2).float()[None].cuda()
    )  # 1,T,3,H,W
    input_depth = torch.from_numpy(dep_list).float().cuda()[:, None]  # T,1,H,W

    return input_video, input_depth


@torch.no_grad()
def spatracker_process_folder(
    working_dir,
    img_list,
    dep_list,
    sample_mask_list,
    model,
    total_n_pts,
    chunk_size=10000,  # designed for 16GB GPU
    K=None,
    save_name="",
    max_viz_cnt=512,
    support_ratio=0.2,
):
    viz_dir = osp.join(working_dir, "spatracker_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = working_dir
    os.makedirs(save_dir, exist_ok=True)
    vis = Visualizer(
        save_dir=working_dir,
        linewidth=2,
        draw_invisible=True,  # False
        tracks_leave_trace=4,
    )

    full_video_pt, full_dep_pt = make_spatracker_input(img_list, dep_list)
    _, T, _, H, W = full_video_pt.shape
    assert sample_mask_list.shape == (T, H, W), f"{sample_mask_list.shape} != {T,H,W}"
    depth_mask = full_dep_pt.squeeze(1) > 1e-6
    logging.info(f"T=[{T}], video shape: {full_video_pt.shape}")

    start_t = time.time()
    # viz the fg mask
    sample_mask_list = torch.as_tensor(sample_mask_list).cpu() > 0
    viz_sample_mask = sample_mask_list[..., None].cpu() * full_video_pt[
        0
    ].cpu().permute(0, 2, 3, 1)
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_sample_mask.mp4"),
        viz_sample_mask.cpu().numpy().astype(np.uint8),
    )
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_depth_boundary_mask.mp4"),
        (depth_mask.detach().cpu().float().numpy() * 255).astype(np.uint8),
    )

    video_pt = full_video_pt.clone()
    dep_pt = full_dep_pt.clone()

    tracks, visibility = [], []
    num_slice = int(np.ceil(total_n_pts / chunk_size))
    chunk_size = int(np.ceil(total_n_pts / num_slice))
    for round in range(num_slice):
        logging.info(f"Round {round+1}/{num_slice} ...")
        masks = sample_mask_list * depth_mask.to(sample_mask_list)
        queries = get_uniform_random_queries(
            video_pt, int(chunk_size * (1.0 - support_ratio)), mask_list=masks
        )
        queries_uniform = get_uniform_random_queries(
            video_pt,
            int(chunk_size * support_ratio),
            mask_list=depth_mask.to(sample_mask_list),
        )
        queries = torch.cat([queries, queries_uniform], 1)

        viz_list = viz_queries(queries.squeeze(0), H, W, T)
        imageio.mimsave(
            osp.join(viz_dir, f"{save_name}_r={round}_quries.mp4"), viz_list
        )
        _tracks, _visibility = infer_spa_tracker(
            video_pt, dep_pt, queries, model, K=K
        )  # T,N,3; T,N
        tracks.append(_tracks)
        visibility.append(_visibility)
    tracks = torch.cat(tracks, 1)
    visibility = torch.cat(visibility, 1)

    end_t = time.time()
    logging.info(f"Time cost: {(end_t - start_t)/60.0:.3f}min")

    # efficient viz
    viz_choice = np.random.choice(tracks.shape[1], min(tracks.shape[1], max_viz_cnt))
    vis.visualize(
        video=video_pt,
        tracks=tracks[None, :, viz_choice, :2],
        visibility=visibility[None, :, viz_choice],
        filename=f"{save_name}_spatracker_tap",
    )
    logging.info(f"Save to {save_dir} with tracks={tracks.shape}")

    np.savez_compressed(
        osp.join(save_dir, f"{save_name}_spatracker_tap.npz"),
        queries=queries.numpy(),  # useless
        tracks=tracks.numpy(),
        visibility=visibility.numpy(),
        K=K,  # also save intrinsic for later use if necessary, but seems because the depth is aligned to input depth, so it is not necessary
    )
    return


def get_pointermap_from_track(H, W, subsample, track, visibility):
    start_t = time.time()
    assert track.ndim == 2 and track.shape[1] == 3
    assert visibility.ndim == 1 and visibility.shape[0] == track.shape[0]

    assert H % subsample == 0 and W % subsample == 0
    ret_H, ret_W = H // subsample, W // subsample
    pointer_map = torch.ones(ret_H * ret_W, dtype=torch.long) * -1
    # quantize the 2D track from oroginal H,W index to ret_H, ret_W index
    track_uv = track[:, :2].clone()
    track_uv_int = (track_uv / subsample).round().long()
    valid_mask = visibility.bool()
    valid_mask = (
        valid_mask
        & (track_uv_int[:, 0] >= 0)
        & (track_uv_int[:, 0] < ret_W)
        & (track_uv_int[:, 1] >= 0)
        & (track_uv_int[:, 1] < ret_H)
    )
    visible_track_uv_int = track_uv_int[valid_mask]
    visible_track_ind = torch.arange(track_uv_int.shape[0])[valid_mask]
    # put the visible track ind into the pointer map by the uv int index
    selected_ind = visible_track_uv_int[:, 1] * ret_W + visible_track_uv_int[:, 0]
    pointer_map[selected_ind] = visible_track_ind
    pointer_map = pointer_map.view(ret_H, ret_W)
    print(f"Pointer map time cost: {time.time()-start_t:.3f}s")
    return pointer_map


@torch.no_grad()
def spatracker_process_dense(
    working_dir,
    query_t,
    subsample_factor: int,
    exsisting_tracks,
    exsisting_visibility,
    img_list,
    dep_list,
    model,
    chunk_size=8192,
    uniform_aug_n=1024,
    mask_aug=None,
    mask_aug_n=1024,
    K=None,
    max_viz_cnt=512,
    visiblility_th=0.9,
):
    # TODO: check the coverage of pointer not only by the visiblilty, but also by the depth ordering!!!
    # TODO: if we know the 3D, one can easily determine the visiblity by checking the order with the depth
    # TODO: the point is invisible if it is both too back or too low confidence!!

    viz_dir = osp.join(working_dir, "spatracker_dense_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = working_dir
    os.makedirs(save_dir, exist_ok=True)
    vis = Visualizer(
        save_dir=working_dir,
        linewidth=2,
        draw_invisible=True,  # False
        tracks_leave_trace=4,
    )

    full_video_pt, full_dep_pt = make_spatracker_input(img_list, dep_list)
    _, T, _, H, W = full_video_pt.shape
    logging.info(f"T=[{T}], video shape: {full_video_pt.shape}")

    pointer_buffer = get_pointermap_from_track(
        H, W, subsample_factor, exsisting_tracks[query_t], exsisting_visibility[query_t]
    )
    pointer_filled = pointer_buffer >= 0

    start_t = time.time()

    video_pt = full_video_pt.clone()
    dep_pt = full_dep_pt.clone()

    round = 0
    while not pointer_filled.all():
        logging.info(f"Round {round} {(~pointer_filled).sum()} unfilled ...")
        # viz
        imageio.imsave(
            osp.join(viz_dir, f"pointer_filled_q={query_t}_{round}.png"),
            pointer_filled.detach().cpu().numpy().astype(np.uint8) * 255,
        )

        # * use original backward tracking
        sample_queries = tracker_get_query_uv(~pointer_filled, query_t, num=chunk_size)
        aug_queries = tracker_get_query_uv(
            torch.ones_like(pointer_filled), query_t, num=uniform_aug_n
        )
        infer_queries = torch.cat([sample_queries, aug_queries], 0)
        infer_queries[:, 1:] = (
            infer_queries[:, 1:] * subsample_factor
        )  # ! scale back to original image scale

        # always append nearest and farthest depth point as query!
        dep = dep_pt[query_t].squeeze(0)
        min_depth_point = torch.argmin(dep.view(-1))
        max_depth_point = torch.argmax(dep.view(-1))
        min_depth_query = torch.tensor(
            [query_t, min_depth_point % W, min_depth_point // W]
        )
        max_depth_query = torch.tensor(
            [query_t, max_depth_point % W, max_depth_point // W]
        )
        depth_queries = torch.stack([min_depth_query, max_depth_query], dim=0).to(
            infer_queries
        )
        infer_queries = torch.cat([infer_queries, depth_queries], 0)

        # ! mask augmentation
        if mask_aug is not None:
            mask_aug_queries = tracker_get_query_uv(
                mask_aug[::subsample_factor, ::subsample_factor],
                query_t,
                num=mask_aug_n,
            )
            mask_aug_queries[:, 1:] = mask_aug_queries[:, 1:] * subsample_factor
            mask_aug_queries[:, 1] = torch.clamp(mask_aug_queries[:, 1], 0, W - 1)
            mask_aug_queries[:, 2] = torch.clamp(mask_aug_queries[:, 2], 0, H - 1)
            infer_queries = torch.cat([infer_queries, mask_aug_queries], 0)

        infer_queries = infer_queries[None].to(video_pt)
        pred_tracks, pred_visibility, filter_id = __infer_one_pass__(
            video_pt.detach().clone(),
            dep_pt.detach().clone(),
            infer_queries.detach().clone(),
            model,
            K=K,
            backward_tracking=True,
            visiblility_th=visiblility_th,
        )
        pred_tracks = pred_tracks[0]  # T,N,3
        pred_visibility = pred_visibility[0]
        if filter_id is not None:
            assert (
                len(filter_id) == infer_queries.shape[1]
            ), f"{len(filter_id)} != {infer_queries.shape[1]}"
        assert len(pred_tracks) == len(pred_visibility) == T

        exsisting_tracks = torch.cat([exsisting_tracks, pred_tracks], 1)
        exsisting_visibility = torch.cat([exsisting_visibility, pred_visibility], 1)

        pointer_buffer = get_pointermap_from_track(
            H,
            W,
            subsample_factor,
            exsisting_tracks[query_t],
            exsisting_visibility[query_t],
        )
        pointer_filled = pointer_buffer >= 0
        round += 1

    end_t = time.time()
    logging.info(f"Time cost: {(end_t - start_t)/60.0:.3f}min")

    # efficient viz
    if max_viz_cnt > 0:
        viz_choice = np.random.choice(
            exsisting_tracks.shape[1], min(exsisting_tracks.shape[1], max_viz_cnt)
        )
        vis.visualize(
            video=video_pt,
            tracks=exsisting_tracks[None, :, viz_choice, :2],
            visibility=exsisting_visibility[None, :, viz_choice],
            filename=f"dense_q={query_t}_spatracker_tap",
        )
    return exsisting_tracks, exsisting_visibility, pointer_buffer
