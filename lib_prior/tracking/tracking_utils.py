# the helpers shared across cotracker, tapir .etc
import torch
import os, sys
import os.path as osp
import imageio
from tqdm import tqdm
import numpy as np

sys.path.append(osp.dirname(osp.abspath(__file__)))
from cotracker_visualizer import Visualizer, read_video_from_path
import logging, time
from torchvision.transforms import GaussianBlur
import random, json, glob
import cv2


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def tracker_get_query_uv(mask, fid, num=1024):
    mask = mask.float()
    H, W = mask.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # u=W ind, H ind
    uv = np.stack([u, v], -1)  # H,W,2
    fg_uv = uv[mask > 0]
    weight = mask[mask > 0]
    if num < fg_uv.shape[0] and num > 0:  # set num to -1 if use all
        choice = torch.multinomial(weight, num, replacement=False)
        # choice = np.random.choice(dyn_uv.shape[0], num, replace=False)
        fg_uv = fg_uv[choice]
    queries = torch.tensor(fg_uv).float()  # N,2
    queries = torch.cat([torch.ones(queries.shape[0], 1) * fid, queries], -1)  # N,3
    return queries


def load_video_pt(src):
    image_src = osp.join(src, "images")
    img_fns = os.listdir(image_src)
    img_fns.sort()
    img_frames = []
    for fn in tqdm(img_fns):
        img_frames.append(imageio.imread(osp.join(image_src, fn)))
    img_frames = np.stack(img_frames, 0)
    video_pt = torch.tensor(img_frames).permute(0, 3, 1, 2)[None].float()
    return video_pt  # B T C H W


def convert_img_list_to_cotracker_input(img_list: list):
    img_frames = [fr for fr in img_list]
    video_pt = torch.tensor(img_frames).permute(0, 3, 1, 2)[None].float()
    return video_pt  # B T C H W


def load_video_resize(src, target_H, target_W):
    image_src = osp.join(src, "images")
    img_fns = os.listdir(image_src)
    img_fns.sort()
    img_frames, ori_img_frames = [], []
    ori_H, ori_W = -1, -1
    for fn in tqdm(img_fns):
        _fr = imageio.imread(osp.join(image_src, fn))
        if ori_H < 0:
            ori_H, ori_W = _fr.shape[:2]
        else:
            assert ori_H == _fr.shape[0] and ori_W == _fr.shape[1]
        ori_img_frames.append(_fr)
        resized_fr = cv2.resize(
            _fr.copy(), (target_W, target_H), interpolation=cv2.INTER_LINEAR
        )
        img_frames.append(resized_fr)
    img_frames = np.stack(img_frames, 0)  # T,H,W,3
    video_pt = torch.from_numpy(img_frames).to(torch.uint8)

    ori_img_frames = np.stack(ori_img_frames, 0)  # T,H,W,3
    ori_video_pt = torch.from_numpy(ori_img_frames).to(torch.uint8)
    return video_pt, ori_H, ori_W, ori_video_pt  # T,H,W,C;


def convert_img_list_to_tapnet_input(img_list: list, target_H, target_W):
    img_frames, ori_img_frames = [], []
    ori_H, ori_W = -1, -1
    for _fr in tqdm(img_list):
        if ori_H < 0:
            ori_H, ori_W = _fr.shape[:2]
        else:
            assert ori_H == _fr.shape[0] and ori_W == _fr.shape[1]
        ori_img_frames.append(_fr)
        resized_fr = cv2.resize(
            _fr.copy(), (target_W, target_H), interpolation=cv2.INTER_LINEAR
        )
        img_frames.append(resized_fr)
    img_frames = np.stack(img_frames, 0)  # T,H,W,3
    video_pt = torch.from_numpy(img_frames).to(torch.uint8)

    ori_img_frames = np.stack(ori_img_frames, 0)  # T,H,W,3
    ori_video_pt = torch.from_numpy(ori_img_frames).to(torch.uint8)
    return video_pt, ori_H, ori_W, ori_video_pt  # T,H,W,C;


def get_sampling_mask(fg_mask, coverage_mask, sigma_ratio=0.02, scale=10.0):
    # sigma_ratio=0.02
    # scale = 10.0
    T, H, W = fg_mask.shape
    coverage_mask = coverage_mask.to(fg_mask)
    sigma = int(min(H, W) * sigma_ratio)
    ksize = int(4 * sigma + 1)
    blur_layer = GaussianBlur(kernel_size=ksize, sigma=sigma)
    # blur the coverage mask
    weight = blur_layer(coverage_mask.float().reshape(T, 1, H, W)).squeeze(1)
    weight = torch.clamp(weight * scale, 0, 1)
    weight = (1 - weight) * fg_mask
    # viz = [(f*255).cpu().numpy().astype(np.uint8) for f in weight]
    # imageio.mimsave("./debug/w.mp4", viz)
    return weight


def get_uniform_random_queries(
    video_pt, n_pts, t_max=-1, mask_list=None, interval=1, shift=0
):
    queries = []
    T = video_pt.shape[1]
    if t_max > 0:
        T = min(t_max, T)
    key_inds = [i for i in range(shift, T) if i % interval == 0]
    if shift == 0 and T - 1 not in key_inds:
        key_inds.append(T - 1)
    if shift == 0 and 0 not in key_inds:
        key_inds = [0] + key_inds

    T = len(key_inds)

    if mask_list is not None:
        mask_list = mask_list[key_inds]
        _count = (mask_list.reshape(T, -1) > 0).sum(-1)
        mask_weight = _count / _count.sum()
    for i, t in enumerate(key_inds):
        if mask_list is None:
            mask = torch.ones_like(video_pt[0, t, 0])
            target_num = n_pts / T
        else:
            mask = mask_list[i]
            target_num = n_pts * mask_weight[i]
        q = tracker_get_query_uv(mask, fid=t, num=int(target_num) * 3)
        queries.append(q)
    queries = torch.cat(queries, 0)
    choice = torch.randperm(queries.shape[0])[:n_pts]
    queries = queries[None, choice]
    return queries  # 1,N,3


def load_epi_error(save_dir):
    fns = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    fns.sort()
    epi_error = []
    for fn in fns:
        epi_error.append(np.load(osp.join(save_dir, fn)))
    epi_error = np.stack(epi_error, 0)
    epi_error = torch.tensor(epi_error).float()
    return epi_error


def load_vos(vos_dir, img_fn_list):
    logging.info(f"loading vos results from {vos_dir}...")
    img_name_list = [osp.basename(f)[:-4] for f in img_fn_list]
    id_mask_list = []
    for img_name in img_name_list:
        seg_fn = osp.join(vos_dir, "Annotations", img_name + ".png")
        seg = imageio.imread(seg_fn)
        id_map = seg[..., 0] + seg[..., 1] * 256 + seg[..., 2] * 256**2
        id_mask_list.append(id_map)
    id_mask_list = np.stack(id_mask_list, 0)
    unique_id = np.unique(id_mask_list)
    # remove 0 from unique id
    unique_id = unique_id[unique_id != 0]
    logging.info(f"loaded {len(unique_id)} unique ids with {len(id_mask_list)} frames.")
    return id_mask_list, unique_id


def viz_queries(queries, H, W, T):
    ret = []
    uv, t = queries[:, 1:], queries[:, 0]
    for _t in range(T):
        mask = _t == t
        _uv = uv[mask].int()
        buffer = torch.zeros(H * W)
        buffer[_uv[:, 1] * W + _uv[:, 0]] = 1
        ret.append((buffer.reshape(H, W).float().numpy() * 255).astype(np.uint8))
    return ret


def viz_coverage(track, track_mask, H, W):
    ret = []
    for t in range(track.shape[0]):
        mask = torch.zeros(H * W)
        uv_int = track[t].int()
        valid_mask = (
            (uv_int[:, 0] < W)
            * (uv_int[:, 0] >= 0)
            * (uv_int[:, 1] < H)
            * (uv_int[:, 1] >= 0)
        )
        vis_mask = track_mask[t][valid_mask]
        uv_int = uv_int[valid_mask]
        flat_ind = uv_int[:, 1] * W + uv_int[:, 0]
        mask[flat_ind[vis_mask]] = 1
        ret.append((mask.reshape(H, W).float().numpy() * 255).astype(np.uint8))
    return ret
