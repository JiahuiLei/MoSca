import torch
import logging
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
import sys, os, os.path as osp
from tqdm import tqdm
from types import SimpleNamespace

sys.path.append(osp.dirname(osp.abspath(__file__)))


def fovdeg2focal(fov_deg):
    focal = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    return focal


def find_valid_view_pair(track_mask, interval=8, min_valid_num=512, early_break=True):
    T, M = track_mask.shape
    todo_list = []
    for i in range(T):
        mask_i = track_mask[i]
        if i + interval > T:
            break
        for j in range(i + interval, T):
            if (j - i) % interval != 0:
                continue
            mask_j = track_mask[j]
            joint_valid = mask_i * mask_j
            if joint_valid.sum() >= min_valid_num:
                todo_list.append((i, j))
                if early_break:
                    break
    logging.info(
        f"Find {len(todo_list)} valid view pair in total {T} frames (M={M}) with interval={interval} and min_valid_num={min_valid_num}"
    )
    return todo_list


def backproject(uv, d, cams):
    # uv: always be [-1,+1] on the short side
    assert uv.ndim == d.ndim + 1
    assert uv.shape[-1] == 2
    dep = d[..., None]
    rel_f = torch.as_tensor(cams.rel_focal).to(uv)
    cxcy = torch.as_tensor(cams.cxcy_ratio).to(uv) * 2.0 - 1.0
    xy = (uv - cxcy[None, :]) * dep / rel_f
    z = dep
    xyz = torch.cat([xy, z], dim=-1)
    return xyz


def project(xyz, cams, th=1e-5):
    assert xyz.shape[-1] == 3
    xy = xyz[..., :2]
    z = xyz[..., 2:]
    z_close_mask = abs(z) < th
    if z_close_mask.any():
        logging.warning(
            f"Projection may create singularity with a point too close to the camera, detected [{z_close_mask.sum()}] points, clamp it"
        )
        z_close_mask = z_close_mask.float()
        z = (
            z * (1 - z_close_mask) + (1.0 * th) * z_close_mask
        )  # ! always clamp to positive
        assert not (abs(z) < th).any()
    rel_f = torch.as_tensor(cams.rel_focal).to(xyz)
    cxcy = torch.as_tensor(cams.cxcy_ratio).to(xyz) * 2.0 - 1.0
    uv = (xy * rel_f / z) + cxcy[None, :]
    return uv  # [-1,1]


def compute_graph_energy(
    rel_focal, view_pair_list, track_mask, homo_list, dep_list, t_chunk=120
):
    # compute all the best aligned
    T, M = track_mask.shape
    cams = SimpleNamespace()
    cams.rel_focal = rel_focal
    cams.cxcy_ratio = [0.5, 0.5]
    point_cam = backproject(
        homo_list.reshape(-1, 2), dep_list.reshape(-1), cams
    ).reshape(T, M, 3)

    ti_list = torch.as_tensor([p[0] for p in view_pair_list])
    tj_list = torch.as_tensor([p[1] for p in view_pair_list])

    mask = track_mask[ti_list] * track_mask[tj_list]  # E,N
    xyz_cam_i, xyz_cam_j = point_cam[ti_list], point_cam[tj_list]
    # solve optimal
    # chunk operation

    s_ji, R_ji, t_ji = compute_batch_optimal_sRt_ji(xyz_cam_i, xyz_cam_j, mask)
    assert (s_ji > 0).all(), "scale solution error, should be non-negative"
    # compute cross coordinates
    xyz_cam_j_from_i = (
        torch.einsum("bij,bnj->bni", R_ji, s_ji[:, None, None] * xyz_cam_i)
        + t_ji[:, None]
    )
    xyz_cam_i_from_j = torch.einsum(
        "bji,bnj->bni", R_ji, (xyz_cam_j - t_ji[:, None]) / s_ji[:, None, None]
    )
    # project and compute flow error
    uv_cam_j_from_i = project(xyz_cam_j_from_i, cams)  # B,N,2
    uv_cam_i_from_j = project(xyz_cam_i_from_j, cams)  # B,N,2
    rel_uv_track = homo_list[..., :2]
    gt_uv_i, gt_uv_j = rel_uv_track[ti_list], rel_uv_track[tj_list]
    uv_diff_at_i = (gt_uv_i - uv_cam_i_from_j).norm(dim=-1)
    uv_diff_at_j = (gt_uv_j - uv_cam_j_from_i).norm(dim=-1)
    E_i = ((uv_diff_at_i + uv_diff_at_j) * mask).sum(1) / mask.sum(1) / 2.0
    E = E_i.mean()
    return E, E_i, s_ji, R_ji, t_ji


def compute_batch_optimal_sRt_ji(xyz_i, xyz_j, mask):
    # solve procrustes
    # j is q, i is p
    # q = sRp +t; xyz_j = sR xyz_i + t

    assert xyz_i.ndim == 3 and xyz_i.shape[-1] == 3
    assert xyz_j.ndim == 3 and xyz_j.shape[-1] == 3
    assert xyz_i.shape == xyz_j.shape
    assert mask.ndim == 2
    mask = mask.float()
    assert mask.any(-1).all()
    mask_weight = mask.float() / mask.sum(dim=1, keepdim=True)

    p_bar = (xyz_i * mask_weight[..., None]).sum(dim=1, keepdim=True)
    q_bar = (xyz_j * mask_weight[..., None]).sum(dim=1, keepdim=True)
    p = xyz_i - p_bar
    q = xyz_j - q_bar

    W = torch.einsum("bni,bnj->bnij", p, q)
    W = (W * mask[..., None, None]).sum(1)  # B,3,3

    U, s, V = torch.svd(W.double())
    # ! warning, torch's svd has W = U @ torch.diag(s) @ (V.T)
    U, s, V = U.float(), s.float(), V.float()
    # R_star = V @ (U.T)
    # ! handling flipping
    R_tmp = torch.einsum("nij,nkj->nik", V, U)
    det = torch.det(R_tmp)
    dia = torch.ones(len(det), 3).to(det)
    dia[:, -1] = det
    Sigma = torch.diag_embed(dia)
    V = torch.einsum("nij,njk->nik", V, Sigma)
    R_star = torch.einsum("nij,nkj->nik", V, U)

    pp = ((p**2).sum(-1) * mask).sum(-1)
    s_star = s.sum(-1) / pp
    t_star = q_bar.squeeze(1) - torch.einsum(
        "bij,bj->bi", R_star, s_star[:, None] * (p_bar.squeeze(1))
    )
    # q = sRp +t; xyz_j = sR xyz_i + t
    return s_star, R_star, t_star


@torch.no_grad()
def fov_graph_init(
    track_mask,
    homo_list,
    dep_list,
    min_interval=8,
    # min_valid_num=256,
    min_valid_num=20,  # ! debug, for box wild video, small overlap....
    viz_fn=None,
    fallback_fov=40.0,
    search_N=100,
    search_start=1.0,
    search_end=179.0,
    early_break=True,
):
    logging.info(f"Start FOV search")
    # * assume fov is known, solve the optimal s,R,t between view pair and form an energy under such case
    T, M = track_mask.shape
    # find time pair
    todo_list = find_valid_view_pair(
        track_mask, min_interval, min_valid_num, early_break=early_break
    )

    e_list, fov_list = [], []
    search_candidates = np.linspace(search_start, search_end, num=search_N)[1:-1]
    for fov in tqdm(search_candidates):
        E, E_i, optimal_s_ji, optimal_R_ji, optimal_t_ji = compute_graph_energy(
            fovdeg2focal(fov), todo_list, track_mask, homo_list, dep_list
        )
        e_list.append(E.item()), fov_list.append(fov)
    e_list = np.array(e_list)
    best_ind = e_list.argmin()
    ret = fov_list[best_ind]
    # ! detect mono case and use fallback fov if no optimal is found
    mono_flag = (e_list[1:] >= e_list[:-1]).all() or (e_list[1:] <= e_list[:-1]).all()
    if mono_flag:
        logging.warning(
            f"FOV search mono case encountered, fall back to FOV={fallback_fov}"
        )
        ret = fallback_fov
    # viz
    if viz_fn is not None:
        fig = plt.figure(figsize=(10, 3))
        plt.plot(fov_list, e_list)
        plt.plot([ret, ret], [min(e_list), max(e_list)], "r--")
        plt.plot([ret], [min(e_list)], "o")
        plt.title(
            f"FOV Linear Search Best={ret:.3f} with energy {min(e_list):.6f}"
        ), plt.xlabel("fov"), plt.ylabel("ReprojEnergy")
        plt.tight_layout()
        plt.savefig(viz_fn)
        plt.close()
    logging.info(f"FOV search done, find FOV={ret:.3f} deg")
    return ret
