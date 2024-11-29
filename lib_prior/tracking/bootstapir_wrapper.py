# this is for boots_tap
# TODO: have to deal with super-long video like how to infer on dycheck. have to split and stick back toghter
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
import logging
import glob, sys, os, os.path as osp
import imageio

sys.path.append(osp.dirname(osp.abspath(__file__)))

from cotracker_visualizer import Visualizer

from tapnet_pt import tapir_model
from tapnet_pt import transforms
from tapnet_pt import viz_utils

from tracking_utils import (
    convert_img_list_to_tapnet_input,
    load_epi_error,
    load_vos,
    get_uniform_random_queries,
)


@torch.no_grad()
def get_bootstapir_model(
    model_path=osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "../../weights/tapnet/bootstapir_checkpoint_v2.pt",
        )
    ),
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
):
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    # model.eval() # ! debug
    return model


def prepare_fg_masks(
    src,
    T,
    tap_size,
    ori_H,
    ori_W,
    dyn_epi_th,
    consider_dyn_epi_mask,
    consider_dyn_seg_mask,
    vos_dir_name,
):
    full_fg_mask = torch.zeros(T, tap_size, tap_size) > 0
    if consider_dyn_epi_mask:
        logging.info(f"Use dynamic epi mask with th={dyn_epi_th}...")
        epi_error_cpu = load_epi_error(osp.join(src, "epi", "global", "error"))
        epi_th = (ori_H * ori_W) / (dyn_epi_th**2)  # ! note use original size
        logging.info(f"epi_th={epi_th:.6f}")
        epi_mask = epi_error_cpu > epi_th
        # nearest resize the epi_mask to target size
        resized_epi_mask = (
            F.interpolate(
                epi_mask.float()[:, None],
                size=(tap_size, tap_size),
                mode="nearest",
            ).squeeze(1)
            > 0
        )  # T,256,256
        full_fg_mask = full_fg_mask | resized_epi_mask

    if consider_dyn_seg_mask:
        logging.info("Use dynamic segmentation mask")
        img_fn_list = sorted(glob.glob(os.path.join(src, "images", "*")))
        seg_dir = osp.join(src, vos_dir_name)
        id_maps, unique_id = load_vos(seg_dir, img_fn_list)
        dyn_ids = np.load(osp.join(src, "epi", "analysis.npz"), allow_pickle=True)[
            "dyn_ids"
        ]
        for uid in dyn_ids:
            # nearest resize the seg_mask to target size
            seg_mask = torch.from_numpy(id_maps == uid).float()  # T,H,W
            resized_seg_mask = (
                F.interpolate(
                    seg_mask[:, None],
                    size=(tap_size, tap_size),
                    mode="nearest",
                ).squeeze(1)
                > 0
            )  # T,256,256

            full_fg_mask = full_fg_mask | resized_seg_mask

    return full_fg_mask


@torch.no_grad()
def bootstapir_process_folder(
    working_dir,
    img_list,
    sample_mask_list,
    model,
    total_n_pts,
    chunk_size=5000,  # designed for 16GB GPU
    save_name="",
    max_viz_cnt=512,
    device=torch.device("cuda:0"),
):

    viz_dir = osp.join(working_dir, "bootstapir_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = working_dir
    os.makedirs(save_dir, exist_ok=True)
    vis = Visualizer(
        save_dir=working_dir,
        linewidth=2,
        draw_invisible=True,
        tracks_leave_trace=4,
    )

    # * prepare video,
    tap_size = 256
    video_pt, ori_H, ori_W, ori_video_pt = convert_img_list_to_tapnet_input(
        img_list, tap_size, tap_size
    )
    video_pt = video_pt.to(device)
    logging.info(
        f"Loaded video frame: {video_pt.shape} and original size: {ori_H}x{ori_W}"
    )
    # frame size is T,256,256,3 uint8

    # * prepare and resize the dense mask
    sample_mask_list = torch.as_tensor(sample_mask_list).cpu() > 0
    viz_sample_mask = sample_mask_list[..., None].cpu() * ori_video_pt.cpu()
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_sample_mask.mp4"),
        viz_sample_mask.cpu().numpy().astype(np.uint8),
    )

    # ! legacy old version may need the sub-sample time, here set the dummy one
    T = len(video_pt)
    logging.info(f"Start tracking...")

    # ! resize the mask to tap_size
    sample_mask_list = (
        F.interpolate(
            sample_mask_list.float()[:, None],
            size=(tap_size, tap_size),
            mode="nearest",
        ).squeeze(1)
        > 0
    )  # T,256,256

    queries = get_uniform_random_queries(
        video_pt[None], total_n_pts, mask_list=sample_mask_list
    ).squeeze(
        0
    )  # N,3 [t,Wind, Hind]
    queries = queries[:, [0, 2, 1]]  # N,3 [t,Hind, Wind]
    queries = queries.to(torch.int32)
    queries[:, 0] = torch.clamp(queries[:, 0], 0, T - 1)
    queries[:, 1] = torch.clamp(queries[:, 1], 0, tap_size - 1)
    queries[:, 2] = torch.clamp(queries[:, 2], 0, tap_size - 1)
    queries = queries.to(device)

    # query: N,3, in the last dim is [t, [0-H=255], [0-W=255]] in int32
    # TODO: chunk wise save memory
    cur = 0
    tracks, visibility = [], []
    while cur < total_n_pts:
        logging.info(f"Processing {cur}-{cur+chunk_size}/{total_n_pts}")
        cur_queries = queries[cur : min(cur + chunk_size, len(queries))]
        _tracks, _visibility = inference(video_pt, cur_queries, model)
        tracks.append(_tracks)
        visibility.append(_visibility)
        cur = cur + chunk_size
    tracks = torch.cat(tracks, dim=0)
    visibility = torch.cat(visibility, dim=0)
    # tracks, visibility = inference(video_pt, queries, model)  # N,T,2; N,T

    tracks = transforms.convert_grid_coordinates(
        tracks.cpu(), (tap_size, tap_size), (ori_W, ori_H)
    )
    visibility = visibility.cpu()

    tracks = tracks.permute(1, 0, 2).cpu()
    visibility = visibility.permute(1, 0).cpu()

    viz_choice = np.random.choice(tracks.shape[1], min(tracks.shape[1], max_viz_cnt))
    vis.visualize(
        video=ori_video_pt.permute(0, 3, 1, 2)[None],  # 1,T,3,H,W
        tracks=tracks[None, :, viz_choice, :2],
        visibility=visibility[None, :, viz_choice],
        filename=f"{save_name}_bootstapir_global",
    )
    logging.info(f"Save to {save_dir} with tracks={tracks.shape}")
    np.savez_compressed(
        osp.join(save_dir, f"{save_name}_bootstapir_tap.npz"),
        queries=queries.cpu().numpy(),  # useless
        tracks=tracks.cpu().numpy(),
        visibility=visibility.cpu().numpy(),
        # sub_t_list=sub_t_list.cpu().numpy(),
    )

    return


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


def postprocess_occlusions(occlusions, expected_dist):
    visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
    return visibles


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


@torch.no_grad()
def inference(frames, query_points, model):
    assert frames.dim() == 4, frames.shape
    assert frames.shape[-1] == 3, frames.shape
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.float()
    frames, query_points = frames[None], query_points[None]

    # Model inference
    outputs = model(frames, query_points)
    tracks, occlusions, expected_dist = (
        outputs["tracks"][0],
        outputs["occlusion"][0],
        outputs["expected_dist"][0],
    )

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles  # N,T,2; N,T


if __name__ == "__main__":
    src = "../../data/debug/C2_N11_S212_s03_T2_2/"
    # src = "../../data/davis/horsejump-high"
    model = get_bootstapir_model()
    bootstapir_process_folder(src, model, "cuda", 4096, 4096)
