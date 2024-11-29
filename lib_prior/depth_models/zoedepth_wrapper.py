raise NotImplementedError("Refactor")
# get metric depth from torch hub zoedepth
# standalone script
import torch
from PIL import Image
import imageio
import numpy as np
import os, os.path as osp
from tqdm import tqdm
import cv2
from matplotlib import cm
import sys, os, os.path as ops

# sys.path.append(osp.dirname(osp.abspath(__file__)))

# from zoedepth.zoedepth.models.builder import build_model
# from zoedepth.zoedepth.utils.config import get_config

# torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo


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
    image = imageio.imread(fn)
    
    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    h1 = h1 - h1 % 8
    w1 = w1 - w1 % 8
    image2 = cv2.resize(image, (w1, h1))
    # image = image[: h1 - h1 % 8, : w1 - w1 % 8]
    # H, W, _ = image.shape
    
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None] / 255.0
    image2 = image2.contiguous()
    return image2, h0, w0


def save_viz(viz_fn, prediction):
    depth_min = prediction.min()
    depth_max = prediction.max()
    viz = cm.viridis((prediction - depth_min) / (depth_max - depth_min))[:, :, :3]
    cv2.imwrite(viz_fn, viz[..., ::-1] * 255)
    return

@torch.no_grad()
def process(in_fn, out_fn, viz_fn, model, device):
    image, H, W = load_image(in_fn)
    dep = model.infer(image.to(device))
    dep = dep.cpu()[0, 0].numpy().astype(np.float32)
    # resize back to H, W
    dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
    np.savez_compressed(out_fn, dep=dep)
    save_viz(viz_fn, dep)
    return


def get_zoedepth_model(device, type="NK"):
    repo = "isl-org/ZoeDepth"
    model = torch.hub.load(repo, f"ZoeD_{type}", pretrained=True)
    model.to(device)
    model.eval()
    return model


def zoedepth_process_folder(model, src, dst):
    print("Generating ZoeDepth...")
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
    # ! warning, try https://github.com/isl-org/ZoeDepth/issues/49 with latest timm!
    
    
    device = "cuda"
    src = "../../data/davis_dev/train"
    zoedepth_model = get_zoedepth_model(device=device, type="N")
    zoedepth_process_folder(
        zoedepth_model,
        src=osp.join(src, "images"),
        dst=osp.join(src, "zoe_depth"),
    )
