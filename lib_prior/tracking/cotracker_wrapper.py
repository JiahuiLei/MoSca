import torch
import os, sys
import os.path as osp
import imageio
from tqdm import tqdm
import numpy as np

sys.path.append(osp.dirname(osp.abspath(__file__)))
from cotracker_visualizer import Visualizer
import logging, time
import glob
from tracking_utils import (
    seed_everything,
    convert_img_list_to_cotracker_input,
    get_sampling_mask,
    get_uniform_random_queries,
    load_epi_error,
    load_vos,
    viz_queries,
    viz_coverage,
)


def get_cotracker(device, online_flag=True, cotrakcer_version="3"):
    if online_flag:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", f"cotracker{cotrakcer_version}_online"
        ).to(device)
    else:
        postfix = "_offline" if cotrakcer_version == "3" else ""
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", f"cotracker{cotrakcer_version}{postfix}"
        ).to(device)
    cotracker.eval()
    return cotracker


@torch.no_grad()
def __online_inference_one_pass__(
    video_pt_cpu, queries_cpu, model, device, add_support_grid=True
):
    T = video_pt_cpu.shape[1]
    first_flag = True
    queries = queries_cpu.to(device)
    for i in tqdm(range(T)):
        if i % model.step == 0 and i > 0:
            video_chunk = video_pt_cpu[:, max(0, i - model.step * 2) : i].to(device)
            pred_tracks, pred_visibility = model(
                video_chunk,
                is_first_step=first_flag,
                queries=queries,
                add_support_grid=add_support_grid,
            )
            first_flag = False
    pred_tracks, pred_visibility = model(
        video_pt_cpu[:, -(i % model.step) - model.step - 1 :].to(device),
        False,
        queries=queries,
        add_support_grid=add_support_grid,
    )
    torch.cuda.empty_cache()
    return pred_tracks.cpu(), pred_visibility.cpu()


@torch.no_grad()
def online_track_point(video_pt, queries, model, device, add_support_grid=True):
    T = video_pt.shape[1]
    N = queries.shape[1]
    # * forward
    pred_tracks_fwd, pred_visibility_fwd = __online_inference_one_pass__(
        video_pt, queries, model, device, add_support_grid
    )
    pred_tracks_fwd = pred_tracks_fwd[0, :, :N]  # T,N,2
    pred_visibility_fwd = pred_visibility_fwd[0, :, :N]  # T,N
    # * inverse manually
    video_pt_inv = video_pt.flip(1)
    queries_inv = queries.clone()
    queries_inv[..., 0] = T - 1 - queries_inv[..., 0]
    pred_tracks_bwd, pred_visibility_bwd = __online_inference_one_pass__(
        video_pt_inv, queries_inv, model, device, add_support_grid
    )
    pred_tracks_bwd = pred_tracks_bwd.flip(1)[0, :, :N]  # T,N,2
    pred_visibility_bwd = pred_visibility_bwd.flip(1)[0, :, :N]  # T,N
    # * fuse the forward and backward
    bwd_mask = torch.arange(T)[:, None] < queries[0, :, 0][None, :]  # T,N
    pred_tracks = torch.where(bwd_mask[..., None], pred_tracks_bwd, pred_tracks_fwd)
    pred_visibility = torch.where(bwd_mask, pred_visibility_bwd, pred_visibility_fwd)
    return pred_tracks, pred_visibility


@torch.no_grad()
def cotracker_process_folder(
    working_dir,
    img_list,
    sample_mask_list,
    model,
    total_n_pts,
    chunk_size=10000,  # designed for 16GB GPU
    save_name="",
    max_viz_cnt=512,
    # cotracker setting
    online_flag=True,
    device=torch.device("cuda:0"),
):
    viz_dir = osp.join(working_dir, "cotracker_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = working_dir
    os.makedirs(save_dir, exist_ok=True)
    full_video_pt = convert_img_list_to_cotracker_input(
        img_list
    ).cpu()  # [B,T,3,H,W] save on cpu
    _, full_T, _, H, W = full_video_pt.shape
    logging.info(f"T=[{full_T}], video shape: {full_video_pt.shape}")
    vis = Visualizer(
        save_dir=working_dir,
        linewidth=2,
        draw_invisible=True,  # False
        tracks_leave_trace=4,
    )

    sample_mask_list = torch.as_tensor(sample_mask_list).cpu() > 0
    viz_sample_mask = sample_mask_list[..., None].cpu() * full_video_pt[
        0
    ].cpu().permute(0, 2, 3, 1)
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_sample_mask.mp4"),
        viz_sample_mask.cpu().numpy().astype(np.uint8),
    )

    # sta cfg

    video_pt = full_video_pt.clone()
    T = full_T

    tracks = torch.zeros(T, 0, 2)
    visibility = torch.zeros(T, 0).bool()

    start_t = time.time()
    num_slice = int(np.ceil(total_n_pts / chunk_size))
    chunk_size = int(np.ceil(total_n_pts / num_slice))
    for round in range(num_slice):
        queries = get_uniform_random_queries(
            video_pt, chunk_size, mask_list=sample_mask_list
        )
        viz_list = viz_queries(queries.squeeze(0), H, W, T)
        imageio.mimsave(
            osp.join(viz_dir, f"{save_name}_r={round}_quries.mp4"), viz_list
        )
        if online_flag:
            _tracks, _visibility = online_track_point(
                video_pt, queries, model, device
            )  # T,N,2; T,N
        else:
            _tracks, _visibility = model(
                video_pt.to(device), queries.to(device), backward_tracking=True
            )
            _tracks, _visibility = _tracks[0].cpu(), _visibility[0].cpu()
        if tracks is None:
            tracks = _tracks
            visibility = _visibility
        else:
            tracks = torch.cat([tracks, _tracks], 1)
            visibility = torch.cat([visibility, _visibility], 1)
    end_t = time.time()
    logging.info(f"Time cost: {(end_t - start_t)/60.0:.3f}min")
    mode = "online" if online_flag else "offline"
    viz_choice = np.random.choice(tracks.shape[1], min(tracks.shape[1], max_viz_cnt))
    vis.visualize(
        video=video_pt,
        tracks=tracks[None, :, viz_choice, :2],
        visibility=visibility[None, :, viz_choice],
        filename=f"{save_name}_cotracker_global_{mode}",
    )
    logging.info(f"Save to {save_dir} with tracks={tracks.shape}")

    np.savez_compressed(
        osp.join(save_dir, f"{save_name}_cotracker_tap.npz"),
        queries=queries.numpy(),  # useless
        tracks=tracks.numpy(),
        visibility=visibility.numpy(),
        # sub_t_list=sub_t_list.numpy(),
    )
    return


if __name__ == "__main__":

    seed_everything(12345)

    # src = "../../data/iphone_5x_2/spin"
    # src = "../../data/iphone/spin"
    src = "../../data/debug/C2_N11_S212_s03_T2_2/"
    model = get_cotracker("cuda", online_flag=True)
    cotracker_process_folder(
        src,
        model,
        "cuda",
        dyn_total_n_pts=1000,
        sta_total_n_pts=1000,
        dyn_chunk_n_pts=1000,
        sta_chunk_n_pts=1000,
    )
    print()
