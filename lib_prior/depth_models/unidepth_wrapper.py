import os

os.environ["PATH"] += os.pathsep + "/sbin"

import torch
from PIL import Image
import numpy as np
import os, os.path as osp
from tqdm import tqdm
import cv2
from matplotlib import cm
import sys

sys.path.append(osp.abspath(osp.dirname(__file__)))

from depth_utils import viz_depth_list


def make_video(src_dir, dst_fn):
    import imageio

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


def load_image(fn):
    rgb = torch.from_numpy(np.array(Image.open(fn))).permute(2, 0, 1)  # C, H, W
    h, w = rgb.shape[-2:]
    return rgb, h, w


@torch.no_grad()
def process(image, out_fn, model):
    pred = model.infer(image)
    dep = pred["depth"]
    dep = dep.cpu()[0, 0].numpy().astype(np.float32)
    np.savez_compressed(out_fn, dep=dep)
    return dep


def get_unidepth_model(device):
    version = "v2"
    backbone = "vitl14"
    model = torch.hub.load(
        "lpiccinelli-eth/UniDepth",
        "UniDepth",
        version=version,
        backbone=backbone,
        pretrained=True,
        trust_repo=True,
        force_reload=True,
    )

    model.to(device)
    model.eval()
    return model


def unidepth_process_folder(
    model,
    img_list,
    fn_list,
    dst,
    invalid_mask_list=None,
):
    print("Generating UniDepth...")
    assert len(img_list) == len(fn_list)
    os.makedirs(dst, exist_ok=True)
    dep_list = []
    device = next(model.parameters()).device
    for i in tqdm(range(len(fn_list))):
        fn = fn_list[i]
        img = torch.from_numpy(img_list[i]).permute(2, 0, 1)  # C, H, W
        save_fn = osp.basename(fn).replace(".jpg", ".npz").replace(".png", ".npz")
        out_fn = os.path.join(dst, save_fn)
        dep = process(img.to(device), out_fn, model)
        if invalid_mask_list is not None:
            dep[invalid_mask_list[i] > 0] = 0
        dep_list.append(dep)
    viz_depth_list(dep_list, dst + ".mp4")
    return


def unidepth_process_folder_legacy(model, src, dst):
    print("Generating UniDepth...")
    os.makedirs(dst, exist_ok=True)
    viz_dir = dst + "_viz"
    os.makedirs(viz_dir, exist_ok=True)
    fns = os.listdir(src)
    fns.sort()
    device = next(model.parameters()).device
    for fn in tqdm(fns):
        in_fn = os.path.join(src, fn)
        save_fn = fn.replace(".jpg", ".npz")
        save_fn = save_fn.replace(".png", ".npz")
        out_fn = os.path.join(dst, save_fn)
        viz_fn = os.path.join(viz_dir, fn)
        process(in_fn, out_fn, viz_fn, model, device)
    make_video(viz_dir, dst + ".mp4")


if __name__ == "__main__":
    device = "cuda"
    # src = "../../data/nvidia_dev_N/Playground/"
    # src = "../../data/nvidia_dev_H/Playground/"
    src = "../../data/debug/"
    unidepth_model = get_unidepth_model(device=device)
    unidepth_process_folder(
        unidepth_model,
        src=osp.join(src, "images"),
        dst=osp.join(src, "depth_uni"),
    )
