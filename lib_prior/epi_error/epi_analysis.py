# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import glob
import os

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool
import os, os.path as osp
import json, logging, time
from matplotlib import cm


def make_video(src_dir, dst_fn):
    print(f"export video to {dst_fn}...")
    img_fn = [
        f for f in os.listdir(src_dir) if f.endswith(".png") or f.endswith(".jpg")
    ]
    img_fn.sort()
    frames = []
    for fn in tqdm(img_fn):
        frames.append(imageio.imread(osp.join(src_dir, fn)))
    imageio.mimsave(dst_fn, frames)
    return


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def get_uv_grid(H, W, homo=False, align_corners=False, device=None):
    """
    Get uv grid renormalized from -1 to 1
    :returns (H, W, 2) tensor
    """
    if device is None:
        device = torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if align_corners:
        xx = 2 * xx / (W - 1) - 1
        yy = 2 * yy / (H - 1) - 1
    else:
        xx = 2 * (xx + 0.5) / W - 1
        yy = 2 * (yy + 0.5) / H - 1
    if homo:
        return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    return torch.stack([xx, yy], dim=-1)


def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = z**2 / (d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2)
    return err


def thread(
    idx,
    flow_dir,
    save_dir,
    image_mask,
    img_fn_list,
    H,
    W,
    step_list,
    verbose_flag=False,
):
    uv = get_uv_grid(H, W, align_corners=False)
    x1 = uv[image_mask].reshape(-1, 2)

    file_names = [osp.basename(f) for f in img_fn_list]
    weights = []
    err_list = []
    this_flow = 0
    counter = 0

    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    F_save_dir = osp.join(save_dir, "robust_F")
    if not osp.exists(F_save_dir):
        os.makedirs(F_save_dir, exist_ok=True)
    Err_save_dir = osp.join(save_dir, "error")
    if not osp.exists(Err_save_dir):
        os.makedirs(Err_save_dir, exist_ok=True)

    # handle the empty mask case
    if image_mask.sum() < 20:
        # early stop
        save_name = os.path.basename(img_fn_list[idx])
        np.save(
            os.path.join(Err_save_dir, save_name + ".npy"),
            np.zeros_like(image_mask).astype(np.float32),
        )
        if verbose_flag:
            print(f"Early finished {idx}.")
        return

    for step in step_list:
        if idx - step >= 0:
            # backward flow and mask
            bwd_flow_path = os.path.join(
                flow_dir, f"{file_names[idx]}_to_{file_names[idx-step]}.npz"
            )
            bwd_data = np.load(bwd_flow_path)
            bwd_flow, bwd_mask = bwd_data["flow"], bwd_data["mask"]
            this_flow = np.copy(this_flow - bwd_flow)
            counter += 1
            bwd_flow = torch.from_numpy(bwd_flow)
            bwd_mask = np.float32(bwd_mask)
            bwd_mask = torch.from_numpy(bwd_mask)
            flow = torch.from_numpy(
                np.stack(
                    [
                        2.0 * bwd_flow[..., 0] / (W - 1),
                        2.0 * bwd_flow[..., 1] / (H - 1),
                    ],
                    axis=-1,
                )
            )
            x2 = x1 + flow[image_mask].view(-1, 2)  # (H*W, 2)
            F, mask = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
            if F is None:
                continue
            F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
            _err = compute_sampson_error(x1, x2, F)
            err = torch.zeros(H, W)
            err[image_mask] = _err
            # * save the robust F matrix
            np.savez_compressed(
                osp.join(F_save_dir, f"{file_names[idx]}_to_{file_names[idx-1]}.npz"),
                F=F,
                mask=mask > 0.5,
            )
            # imageio.imsave("./debug/F_mask.png", (mask*255).reshape(H, W))

            # ! 2024.Sep.17 new version, no need to scale the error
            # fac = (H + W) / 2
            # err = err * fac**2
            err_list.append(err)
            weights.append(bwd_mask[image_mask].mean())

        if idx + step < len(img_fn_list):
            # forward flow and mask
            fwd_flow_path = os.path.join(
                flow_dir, f"{file_names[idx]}_to_{file_names[idx+step]}.npz"
            )
            fwd_data = np.load(fwd_flow_path)
            fwd_flow, fwd_mask = fwd_data["flow"], fwd_data["mask"]
            this_flow = np.copy(this_flow + fwd_flow)
            counter += 1
            fwd_flow = torch.from_numpy(fwd_flow)
            fwd_mask = np.float32(fwd_mask)
            fwd_mask = torch.from_numpy(fwd_mask)
            flow = torch.from_numpy(
                np.stack(
                    [
                        2.0 * fwd_flow[..., 0] / (W - 1),
                        2.0 * fwd_flow[..., 1] / (H - 1),
                    ],
                    axis=-1,
                )
            )
            x2 = x1 + flow[image_mask].view(-1, 2)  # (H*W, 2)
            F, mask = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
            if F is None:
                continue
            F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
            _err = compute_sampson_error(x1, x2, F)
            err = torch.zeros(H, W)
            err[image_mask] = _err
            # * save the robust F matrix
            np.savez_compressed(
                osp.join(F_save_dir, f"{file_names[idx]}_to_{file_names[idx+1]}.npz"),
                F=F,
                mask=mask > 0.5,
            )

            # ! 2024.Sep.17 new version, no need to scale the error
            # fac = (H + W) / 2
            # err = err * fac**2

            err_list.append(err)
            weights.append(fwd_mask[image_mask].mean())

    if len(err_list) == 0:
        save_name = os.path.basename(img_fn_list[idx])
        np.save(
            os.path.join(Err_save_dir, save_name + ".npy"),
            np.zeros_like(image_mask).astype(np.float32),
        )
        if verbose_flag:
            print(f"Early finished {idx}.")
        return

    err = torch.amax(torch.stack(err_list, 0), 0)

    # # ! old cold has a surpass
    # thresh = torch.quantile(err, 0.8)
    # err = torch.where(err <= thresh, torch.zeros_like(err), err)

    save_name = os.path.basename(img_fn_list[idx])
    np.save(os.path.join(Err_save_dir, save_name + ".npy"), err.numpy())
    if verbose_flag:
        print(f"finished {idx}.")
    return


def load_epi_error(root_dir):
    saved_dir = osp.join(root_dir, "error")
    epi_fns = glob.glob(osp.join(saved_dir, "*.npy"))
    epi_fns.sort()
    epi_list = []
    for fn in epi_fns:
        epi = np.load(fn)
        epi_list.append(epi)
    epi_list = np.stack(epi_list, 0)
    return epi_list


def viz_epi_error_folder(epi_list, save_fn):
    epi_list = epi_list / epi_list.max()
    epi_list = cm.viridis(epi_list)[..., :3]
    epi_list = (epi_list * 255).astype(np.uint8)
    imageio.mimsave(save_fn, epi_list)
    return


def get_seg_boundary_mask(id_maps, boundary_dilate_size):
    dilate_kernel = np.ones((boundary_dilate_size, boundary_dilate_size))
    boundary_mask_list = []
    for t in range(len(id_maps)):
        # do a laplacian on the id maps
        id_maps_lap = cv2.Laplacian(id_maps[t].astype(np.float32), cv2.CV_32F)
        id_maps_lap = np.abs(id_maps_lap)
        boundary_mask = id_maps_lap > 1e-6
        # dilate teh boundary mask
        boundary_mask = cv2.dilate(
            boundary_mask.astype(np.uint8), dilate_kernel, iterations=1
        )
        boundary_mask_list.append(boundary_mask)
    boundary_mask_list = np.stack(boundary_mask_list, 0) > 0
    return boundary_mask_list


def get_epi_list(epi_list, mask_list, epi_rank):
    T = len(epi_list)
    ret = []
    for t in range(T):
        epi = epi_list[t][mask_list[t]]
        if epi.size == 0:
            ret.append(0)
            continue
        sorted_epi = np.sort(epi)[::-1]
        ith = max(0, int(epi_rank * len(sorted_epi)))
        max_epi = sorted_epi[ith]
        ret.append(float(max_epi))
    ret = np.asarray(ret)
    return ret


def analyze_epi(working_dir, step_list=[1], num_threads=None):
    start_t = time.time()
    img_fn_list = sorted(glob.glob(os.path.join(working_dir, "images", "*")))
    H, W, _ = imageio.imread(img_fn_list[0]).shape

    params = []
    global_epi_dir = osp.join(working_dir, "epi")
    for idx, _ in tqdm(enumerate(img_fn_list)):
        params.append(
            (
                idx,
                osp.join(working_dir, "flow_raft"),
                global_epi_dir,
                np.ones((H, W)) > 0,
                img_fn_list,
                H,
                W,
                step_list,
            )
        )
    if num_threads is None or num_threads < 0:
        cores = os.cpu_count()
    else:
        cores = num_threads
    logging.info(f"Running {len(params)} sub-process with {cores} in parallel ...")
    with Pool(cores) as p:
        p.starmap(thread, params)

    epi_global_list = load_epi_error(global_epi_dir)
    viz_epi_error_folder(epi_global_list, osp.join(working_dir, "epi_error.mp4"))
    logging.info(f"EPI Total time: {(time.time()-start_t)/60.0:.2f}min")
    return


if __name__ == "__main__":
    analyze_epi("../../data/dragon")
    # analyze_epi("../../data/debug11")
    # analyze_epi("../../data/davis/breakdance-flare/")
    # analyze_epi("../../data/davis/train/")
