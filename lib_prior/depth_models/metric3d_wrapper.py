import torch
from PIL import Image
import numpy as np
import os, os.path as osp
from tqdm import tqdm
import cv2
from matplotlib import cm
import imageio
from matplotlib import pyplot as plt
import sys

sys.path.append(osp.abspath(osp.dirname(__file__)))

from depth_utils import viz_depth_list


@torch.no_grad()
def __process_frame__(img, out_fn, model, fxfycxcy_pixel=None, default_fov_deg=53.13):
    rgb_origin = torch.from_numpy(img)
    input_size = (616, 1064)  # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    if fxfycxcy_pixel is None:
        # the default short side fov is 53.13 degree
        f = min(h, w) / 2 / np.tan(np.radians(default_fov_deg / 2))
        intrinsic = [f, f, w / 2, h / 2]
    else:
        intrinsic = fxfycxcy_pixel
    rgb = cv2.resize(
        rgb_origin.numpy().copy(),
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    # remember to scale intrinsic, hold depth
    intrinsic = [
        intrinsic[0] * scale,
        intrinsic[1] * scale,
        intrinsic[2] * scale,
        intrinsic[3] * scale,
    ]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({"input": rgb})
    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1],
        pad_info[2] : pred_depth.shape[1] - pad_info[3],
    ]

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
    ).squeeze()
    ###################### canonical camera space ######################

    #### de-canonical transform
    canonical_to_real_scale = (
        intrinsic[0] / 1000.0
    )  # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)
    dep = pred_depth.cpu().numpy().astype(np.float32)
    np.savez_compressed(out_fn, dep=dep)
    # save_viz(viz_fn, dep)
    return dep


def get_metric3dv2_model(device, version="giant2"):
    assert version in ["giant2", "large", "small"]
    model = torch.hub.load("yvanyin/metric3d", f"metric3d_vit_{version}", pretrain=True)
    model.to(device)
    model.eval()
    return model


def metric3d_process_folder(
    model,
    img_list,
    fn_list,
    dst,
    fxfycxcy_pixel=None,
    default_fov_deg=53.13,
    invalid_mask_list=None,
):
    print("Metric-3d processing...")
    assert len(img_list) == len(fn_list)
    os.makedirs(dst, exist_ok=True)
    dep_list = []
    # for fn, img in tqdm(zip(fn_list, img_list)):
    for i in tqdm(range(len(fn_list))):
        fn = fn_list[i]
        img = img_list[i]
        save_fn = osp.basename(fn).replace(".jpg", ".npz").replace(".png", ".npz")
        out_fn = os.path.join(dst, save_fn)
        dep = __process_frame__(
            img,
            out_fn,
            model,
            fxfycxcy_pixel=fxfycxcy_pixel,
            default_fov_deg=default_fov_deg,
        )
        if invalid_mask_list is not None:
            dep[invalid_mask_list[i] > 0] = 0
        dep_list.append(dep)
    # make_video(viz_dir, dst + ".mp4")
    # viz the depth in global consistent scale
    viz_depth_list(dep_list, dst + ".mp4")

    return


if __name__ == "__main__":
    device = "cuda"
    # src = "../../data/nvidia_dev_N/Playground/"
    # src = "../../data/nvidia_dev_H/Playground/"
    src = "../../data/debug/C2_N11_S212_s03_T2_new/images"

    model = get_metric3dv2_model(device=device)

    # load images and fns
    fns = os.listdir(src)
    fns.sort()
    img_list, fn_list = [], []
    for fn in fns:
        if fn.endswith(".jpg") or fn.endswith(".png"):
            img_list.append(cv2.imread(os.path.join(src, fn)))
            fn_list.append(fn)

    metric3d_process_folder(
        model,
        img_list,
        fn_list,
        dst=osp.join("../../debug", "depth_m3d"),
    )
