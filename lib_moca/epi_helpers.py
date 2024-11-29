# Copyright (c) Meta Platforms, Inc. and affiliates.
import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing.dummy import Pool
import multiprocessing
import os, os.path as osp
import json, logging, time
from matplotlib import cm


# cpu multi thread with return values
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


def epi_thread(send_end, i, j, uv1_normalzied, uv2_normalized, covis_mask, F=None):
    # ! note, this function uses normalzied grid (strenched) uv should both be [-1,1]
    x1 = uv1_normalzied[covis_mask][..., :2]
    x2 = uv2_normalized[covis_mask][..., :2]
    # * [p2;1]^T * F * [p1;1] = 0
    if F is None:  # statndard mode
        F, inlier = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
    else:  # providxed F, only compute epi error
        inlier = np.zeros(len(x1), dtype=np.uint8) > 0
    if F is None:
        # make a dummy output
        # ！ if no inlier， everything should be large epi error and dynamic!
        F = torch.zeros(3, 3)  # use all zero as dummy
        inlier = torch.zeros(len(x1), dtype=torch.bool)
        epi_error = torch.ones(len(x1), dtype=torch.float32) * 1e10
    else:
        F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
        inlier = torch.from_numpy(inlier).bool().squeeze()
        epi_error = compute_sampson_error(x1, x2, F)
    # ! note, different from the old version, this version epi error is not scaled by H,W
    # pad the inlier and epi error
    ret_epi = -torch.ones_like(covis_mask, dtype=torch.float32)
    ret_epi[covis_mask] = epi_error
    ret_inlier = torch.zeros_like(covis_mask).bool()
    ret_inlier[covis_mask] = inlier
    send_end.send(
        {
            "i": i,
            "j": j,
            "F": F,
            "padded_inlier": ret_inlier,
            "inlier_ratio": inlier.float().mean(),
            "inlier_count": inlier.sum(),
            "epi_error": ret_epi,
        }
    )

@torch.no_grad()
def analyze_track_epi(pair_list, track, track_mask, H, W, F_list=None):
    logging.info(f"Start analyzing {len(pair_list)} pairs")
    start_t = time.time()
    # track: tensor, (T, N, 2)
    # track_mask: tensor, (T, N)
    # H: int, height, W: int, width
    track_noramlized = track.clone().cpu().float()
    track_noramlized[..., 0] = 2.0 * track_noramlized[..., 0] / W - 1.0
    track_noramlized[..., 1] = 2.0 * track_noramlized[..., 1] / H - 1.0

    # * parallel solve
    jobs = []
    pipe_list = []
    for _ind in range(len(pair_list)):
        i, j = pair_list[_ind]
        recv_end, send_end = multiprocessing.Pipe(False)
        if F_list is not None:
            F = F_list[_ind]
        else:
            F = None
        args = (
            send_end,
            i,
            j,
            track_noramlized[i].detach().cpu().clone(),
            track_noramlized[j].detach().cpu().clone(),
            (track_mask[i] & track_mask[j]).cpu().clone(),
            F,
        )
        p = multiprocessing.dummy.Process(target=epi_thread, args=args)
        jobs.append(p)
        pipe_list.append(recv_end)
        p.start()
    for proc in jobs:
        proc.join()

    # * collect results
    result_list = [x.recv() for x in pipe_list]
    result_dict = {(it["i"], it["j"]): it for it in result_list}
    F_list, epi_error_list, inlier_list = [], [], []
    # todo: here may also use the inlier count to improve the robustness
    for pair in pair_list:
        sol = result_dict[pair]
        inlier_count = sol["inlier_count"]
        F = sol["F"]
        epi_error = sol["epi_error"]
        inlier = sol["padded_inlier"]
        F_list.append(F)
        epi_error_list.append(epi_error)
        inlier_list.append(inlier)
        # print(inlier_count)

    F_list = torch.stack(F_list, dim=0)
    epi_error_list = torch.stack(epi_error_list, dim=0)
    inlier_list = torch.stack(inlier_list, dim=0)
    run_time = time.time() - start_t
    logging.info(f"Finished epi for {len(pair_list)} pairs, time: {run_time:.2f}s")
    return F_list, epi_error_list, inlier_list


def compute_track_epi(pair_list, track, track_mask, F_list, H, W):
    logging.info(f"Start analyzing {len(pair_list)} pairs")
    start_t = time.time()
    # track: tensor, (T, N, 2)
    # track_mask: tensor, (T, N)
    # H: int, height, W: int, width
    track_noramlized = track.clone().cpu()
    track_noramlized[..., 0] = 2.0 * track_noramlized[..., 0] / W - 1.0
    track_noramlized[..., 1] = 2.0 * track_noramlized[..., 1] / H - 1.0

    # * parallel solve
    jobs = []
    pipe_list = []
    for i, j in pair_list:
        recv_end, send_end = multiprocessing.Pipe(False)
        args = (
            send_end,
            i,
            j,
            track_noramlized[i],
            track_noramlized[j],
            (track_mask[i] & track_mask[j]).cpu(),
        )
        p = multiprocessing.dummy.Process(target=epi_thread, args=args)
        jobs.append(p)
        pipe_list.append(recv_end)
        p.start()
    for proc in jobs:
        proc.join()

    # * collect results
    result_list = [x.recv() for x in pipe_list]
    result_dict = {(it["i"], it["j"]): it for it in result_list}
    F_list, epi_error_list, inlier_list = [], [], []
    # todo: here may also use the inlier count to improve the robustness
    for pair in pair_list:
        sol = result_dict[pair]
        inlier_count = sol["inlier_count"]
        F = sol["F"]
        epi_error = sol["epi_error"]
        inlier = sol["padded_inlier"]
        F_list.append(F)
        epi_error_list.append(epi_error)
        inlier_list.append(inlier)

    F_list = torch.stack(F_list, dim=0)
    epi_error_list = torch.stack(epi_error_list, dim=0)
    inlier_list = torch.stack(inlier_list, dim=0)
    run_time = time.time() - start_t
    logging.info(f"Finished epi for {len(pair_list)} pairs, time: {run_time:.2f}s")
    return F_list, epi_error_list, inlier_list

 
def identify_tracks(epierr_list, epi_th, static_cnt=1, dynamic_cnt=1):
    over_th_cnt = (epierr_list > epi_th).sum(0)
    dym_mask = over_th_cnt >= dynamic_cnt
    sta_mask = over_th_cnt < static_cnt
    logging.warning(
        f"Identify {sta_mask.sum()} static, {dym_mask.sum()} dynamic outof {len(sta_mask)} tracks"
    )
    return sta_mask, dym_mask
