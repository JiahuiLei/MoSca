# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import imageio
from copy import deepcopy

from .RAFT.raft import RAFT
from .RAFT.utils import flow_viz
from .RAFT.utils.utils import InputPadder

from .flow_utils import *
from tqdm import tqdm
import os, os.path as osp


def load_image(input_img):
    long_dim = 768
    if isinstance(input_img, str):
        img = np.array(Image.open(input_img)).astype(np.uint8)
    else:
        assert isinstance(input_img, np.ndarray)
        assert input_img.ndim == 3 and input_img.shape[2] == 3, f"{input_img.shape}"
        img = input_img.astype(np.uint8).copy()
    ori_shape = img.shape[:2]

    # Portrait Orientation
    if img.shape[0] > img.shape[1]:
        input_h = long_dim
        input_w = int(round(float(input_h) / img.shape[0] * img.shape[1]))
    # Landscape Orientation
    else:
        input_w = long_dim
        input_h = int(round(float(input_w) / img.shape[1] * img.shape[0]))

    # print("flow input w %d h %d" % (input_w, input_h))
    img = cv2.resize(img, (input_w, input_h), cv2.INTER_LINEAR)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None], ori_shape


def resize_flow(flow, img_h, img_w):
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

    return flow


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = (
        fwd_lr_error
        < alpha_1
        * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1))
        + alpha_2
    )

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return fwd_mask, bwd_mask


def get_neighboring_pair_list(name_list):
    names = deepcopy(name_list)
    names.sort()
    pair_list = []
    for i in range(len(names) - 1):
        pair_list.append((names[i], names[i + 1]))
    return pair_list


def get_dense_pair_list(name_list, jump_steps=[1]):
    names = deepcopy(name_list)
    names.sort()
    pair_list = []
    for step in jump_steps:
        for i in range(len(names) - step):
            pair_list.append((names[i], names[i + step]))
    return pair_list


@torch.no_grad()
def raft_process_folder(
    model, img_list, img_name_list, dst_dir, pair_list=None, iters=20, step_list=[1]
):
    if pair_list is None:
        pair_list = get_dense_pair_list(img_name_list, jump_steps=step_list)
    device = next(model.parameters()).device
    os.makedirs(dst_dir, exist_ok=True)
    flow_viz_list, flow_mask_viz_list = [], []
    for vi, vj in tqdm(pair_list):
        image1, img_shape = load_image(img_list[img_name_list.index(vi)])
        image2, img_shape = load_image(img_list[img_name_list.index(vj)])
        image1, image2 = image1.to(device), image2.to(device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, flow_fwd = model(image1, image2, iters=iters, test_mode=True)
        _, flow_bwd = model(image2, image1, iters=iters, test_mode=True)

        flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0)
        flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0)

        flow_fwd = resize_flow(flow_fwd, img_shape[0], img_shape[1])
        flow_bwd = resize_flow(flow_bwd, img_shape[0], img_shape[1])

        mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

        # Save flow
        np.savez_compressed(
            os.path.join(dst_dir, f"{vi}_to_{vj}.npz"),
            flow=flow_fwd.astype(np.float16),
            mask=mask_fwd.astype(np.float16),
        )
        np.savez_compressed(
            os.path.join(dst_dir, f"{vj}_to_{vi}.npz"),
            flow=flow_bwd.astype(np.float16),
            mask=mask_bwd.astype(np.float16),
        )
        # Save flow_img
        if vi < vj:
            flow_viz_fwd = flow_viz.flow_to_image(flow_fwd)
            flow_viz_list.append(flow_viz_fwd)
            flow_mask_viz_list.append(mask_fwd.astype(np.uint8) * 255)
    imageio.mimsave(
        os.path.join(os.path.dirname(dst_dir), "flow_viz.mp4"),
        flow_viz_list,
    )
    imageio.mimsave(
        os.path.join(os.path.dirname(dst_dir), "flow_mask_viz.mp4"),
        flow_mask_viz_list,
    )
    return


def get_raft_model(ckpt_path, device, small=False, mixed_precision=False):
    args = argparse.Namespace()
    args.small = small
    args.model = ckpt_path
    args.mixed_precision = mixed_precision
    # model = torch.nn.DataParallel(RAFT(args))
    model = RAFT(args)
    _stat_dict = torch.load(args.model, map_location="cpu")
    # remove the module prefix
    state_dict = {}
    for k, v in _stat_dict.items():
        if k.startswith("module."):
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict)
    # model = model.module
    model.to(device)
    model.eval()
    return model
