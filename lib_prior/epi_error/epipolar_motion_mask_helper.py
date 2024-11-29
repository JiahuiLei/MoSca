# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import glob
import os

import cv2
import imageio
import numpy as np
import skimage.morphology
import torch
from PIL import Image
import torchvision
from tqdm import tqdm
from multiprocessing.dummy import Pool
import os, os.path as osp


def make_video(src_dir, dst_fn):
    print(f"export video to {dst_fn}...")
    # print(os.listdir(src_dir))
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


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def get_stats(X, norm=2):
    """
    :param X (N, C, H, W)
    :returns mean (1, C, 1, 1), scale (1)
    """
    mean = X.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
    if norm == 1:
        mag = torch.abs(X - mean).sum(dim=1)  # (N, H, W)
    else:
        mag = np.sqrt(2) * torch.sqrt(torch.square(X - mean).sum(dim=1))  # (N, H, W)
    scale = mag.mean() + 1e-6
    return mean, scale


def thread(idx, data_dir, images, error_factor=90.0, maskrcnn_mask=None):
    rgb = load_image(images[idx])
    H = rgb.shape[2]
    W = rgb.shape[3]
    uv = get_uv_grid(H, W, align_corners=False)
    x1 = uv.reshape(-1, 2)

    file_names = [osp.basename(f) for f in images]
    # print("processing idx=" + str(idx))
    weights = []
    err_list = []
    normalized_flow = []
    this_flow = 0
    counter = 0
    F_save_dir = osp.join(data_dir, "epipolar_F")
    os.makedirs(F_save_dir, exist_ok=True)
    for step in [1]:
        # print("step: " + str(step))
        if idx - step >= 0:
            # backward flow and mask
            bwd_flow_path = os.path.join(
                data_dir, "flows", f"{file_names[idx]}_to_{file_names[idx-1]}.npz"
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
            normalized_flow.append(flow)
            x2 = x1 + flow.view(-1, 2)  # (H*W, 2)
            F, mask = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
            F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
            err = compute_sampson_error(x1, x2, F).reshape(H, W)
            # * save the robust F matrix
            np.savez_compressed(
                osp.join(F_save_dir, f"{file_names[idx]}_to_{file_names[idx-1]}.npz"),
                F=F,
                mask=(mask > 0.5).reshape(H, W),
            )
            # imageio.imsave("./debug/F_mask.png", (mask*255).reshape(H, W))
            fac = (H + W) / 2
            err = err * fac**2
            err_list.append(err)
            weights.append(bwd_mask.mean())

        if idx + step < len(images):
            # forward flow and mask
            fwd_flow_path = os.path.join(
                data_dir, "flows", f"{file_names[idx]}_to_{file_names[idx+1]}.npz"
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
            normalized_flow.append(flow)
            x2 = x1 + flow.view(-1, 2)  # (H*W, 2)
            F, mask = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
            F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
            err = compute_sampson_error(x1, x2, F).reshape(H, W)
            # * save the robust F matrix
            np.savez_compressed(
                osp.join(F_save_dir, f"{file_names[idx]}_to_{file_names[idx+1]}.npz"),
                F=F,
                mask=(mask > 0.5).reshape(H, W),
            )
            fac = (H + W) / 2
            err = err * fac**2
            err_list.append(err)
            weights.append(fwd_mask.mean())

    err = torch.amax(torch.stack(err_list, 0), 0)
    thresh = torch.quantile(err, 0.8)
    err = torch.where(err <= thresh, torch.zeros_like(err), err)

    # ext = ".png"
    ext = ""

    if error_factor > 0:
        mask = torch.from_numpy(
            skimage.morphology.binary_opening(
                err.numpy() > ((H * W) / (error_factor**2)), skimage.morphology.disk(1)
            )
        )  # 64.0 for nvidia, 49.0 for DAVIS_480p, 256.0 for DAVIS_1080p
        # combine
        if maskrcnn_mask is not None:
            mask_rcnn_mask = torch.from_numpy(maskrcnn_mask)
            mask[mask_rcnn_mask > 1] = 1.0
        mask_dir = os.path.join(data_dir, "epipolar_mask")
        os.makedirs(mask_dir, exist_ok=True)
        mask = torch.from_numpy(
            skimage.morphology.dilation(mask.cpu(), skimage.morphology.disk(2))
        ).float()
        imageio.imwrite(
            os.path.join(mask_dir, save_name + ext),
            ((1.0 - mask).numpy() * 255.0).astype(np.uint8),
        )  # ! save the inverse
    npy_dir = os.path.join(data_dir, "epipolar_error_npy")
    err_dir = os.path.join(data_dir, "epipolar_error")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    save_name = os.path.basename(images[idx])
    np.save(os.path.join(npy_dir, save_name + ".npy"), err.numpy())
    imageio.imwrite(
        os.path.join(err_dir, save_name + ext),
        ((err / err.max()).cpu().numpy() * 255).astype(np.uint8),
    )
    print(f"finished {idx}.")
    return


def run_maskrcnn(model, img_path, device):
    threshold = 0.5
    o_image = Image.open(img_path).convert("RGB")
    width, height = o_image.size
    if width > height:
        intHeight = 576
        intWidth = 1024
    else:
        intHeight = 1024
        intWidth = 576

    image = o_image.resize((intWidth, intHeight), Image.ANTIALIAS)

    image_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

    tenHumans = torch.FloatTensor(intHeight, intWidth).fill_(1.0).to(device)

    objPredictions = model([image_tensor])[0]

    for intMask in range(objPredictions["masks"].size(0)):
        if objPredictions["scores"][intMask].item() > threshold:
            # person, vehicle, accessory, animal, sports
            if objPredictions["labels"][intMask].item() == 1:  # person
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if (
                objPredictions["labels"][intMask].item() >= 2
                and objPredictions["labels"][intMask].item() <= 9
            ):  # vehicle
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if (
                objPredictions["labels"][intMask].item() >= 26
                and objPredictions["labels"][intMask].item() <= 33
            ):  # accessory
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if (
                objPredictions["labels"][intMask].item() >= 16
                and objPredictions["labels"][intMask].item() <= 25
            ):  # animal
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if (
                objPredictions["labels"][intMask].item() >= 34
                and objPredictions["labels"][intMask].item() <= 43
            ):  # sports
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if objPredictions["labels"][intMask].item() == 88:  # teddy bear
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0

    npyMask = skimage.morphology.erosion(
        tenHumans.cpu().numpy(), skimage.morphology.disk(1)
    )
    npyMask = ((npyMask < 1e-3) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return npyMask


def get_maskrcnn_masks(data_dir, W, H):
    print("Running MaskRCNN ... ")
    img_path_list = sorted(glob.glob(os.path.join(data_dir, "images", "*")))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    netMaskrcnn = (
        torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        .to(device)
        .eval()
    )
    mask_rcnn_masks = []
    for i in tqdm(range(0, len(img_path_list))):
        img_path = img_path_list[i]
        img_name = img_path.split("/")[-1]
        # print(img_path)
        semantic_mask = run_maskrcnn(netMaskrcnn, img_path, device)
        semantic_mask = cv2.resize(
            semantic_mask, (W, H), interpolation=cv2.INTER_NEAREST
        )
        mask_rcnn_masks.append(semantic_mask)
    netMaskrcnn.cpu()
    del netMaskrcnn
    return mask_rcnn_masks


def epi_motion_mask_from_optical_flow_process_folder(
    data_dir, use_maskrcnn=False, error_factor=-1
):
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*")))
    if use_maskrcnn:
        rgb = load_image(images[0])
        H = rgb.shape[2]
        W = rgb.shape[3]
        maskrcnn_masks = get_maskrcnn_masks(data_dir, W, H)
    else:
        maskrcnn_masks = [None for _ in images]

    params = []
    for idx, _ in tqdm(enumerate(images)):
        params.append((idx, data_dir, images, error_factor, maskrcnn_masks[idx]))
    # get the number of available cores
    cores = os.cpu_count()
    cores = int(cores * 0.8)
    # create the process pool

    with Pool(cores) as p:
        p.starmap(thread, params)
    # for param in params:
    #     thread(*param)

    # mask_dir = os.path.join(data_dir, "epipolar_mask")
    err_dir = os.path.join(data_dir, "epipolar_error")
    # make_video(mask_dir, mask_dir + ".mp4")
    make_video(err_dir, err_dir + ".mp4")
