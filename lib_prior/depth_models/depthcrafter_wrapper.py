import sys, os, os.path as osp

sys.path.append(osp.abspath(osp.dirname(__file__)))

import numpy as np
import torch
from diffusers.training_utils import set_seed
from diffusers import EulerDiscreteScheduler

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from matplotlib import cm
import imageio
from depth_utils import viz_depth_list, save_depth_list
import cv2


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
        modify_scheduler: bool = True,
        local_files_only=False,
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=local_files_only,
        )

        if modify_scheduler:
            # ! modify the inference scheduler
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                local_files_only=local_files_only,
            )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                try:
                    self.pipe.enable_model_cpu_offload()
                except Exception as e:
                    print(e)
                    print("Model offload is not enabled")
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        frames: np.ndarray,  # T,H,W,3 in [0, 1]
        num_denoising_steps: int,
        guidance_scale: float,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = True,
    ):
        set_seed(seed)

        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]  # T.H.W
        return res


def get_depthcrafter_model(modify_scheduler=True):
    try:
        # model = DepthCrafterDemo(
        #     unet_path=osp.join(
        #         osp.abspath(osp.dirname(__file__)), "../../weights/depthcrafter/unet"
        #     ),
        #     pre_train_path=osp.join(
        #         osp.abspath(osp.dirname(__file__)), "../../weights/depthcrafter/svd"
        #     ),
        #     cpu_offload="model",
        #     modify_scheduler=modify_scheduler,
        #     local_files_only=True,
        # )
        model = DepthCrafterDemo(
            unet_path="tencent/DepthCrafter",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
            cpu_offload="model",
            modify_scheduler=modify_scheduler,
            local_files_only=True,
        )
        return model
    except:
        print("Failed to load model from cache, try to download from the internet")
        model = DepthCrafterDemo(
            unet_path="tencent/DepthCrafter",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
            cpu_offload="model",
            modify_scheduler=modify_scheduler,
        )
        return model


def depthcrafter_process_folder(
    model,
    img_list,
    fn_list,
    dst,
    invalid_mask_list=None,
    n_steps=25,  # * can tune
    # default
    guidance_scale=1.2,
    window_size=110,
    overlap=25,
    seed=42,
):
    img_list = np.asarray(img_list)
    if img_list.dtype == np.uint8:
        img_list = img_list.astype(np.float32) / 255.0
    T, H, W, C = img_list.shape

    # reshape to 64 base size
    new_H = H // 64 * 64
    new_W = W // 64 * 64
    # use opencv to resize
    working_img_list = []
    for img in img_list:
        img = cv2.resize(img.copy(), (new_W, new_H))
        working_img_list.append(img)
    working_img_list = np.asarray(working_img_list)

    _dep_list = model.infer(
        working_img_list,
        n_steps,
        guidance_scale=guidance_scale,
        window_size=window_size,
        overlap=overlap,
        seed=seed,
        track_time=True,
    )

    dep_list = []
    for dep in _dep_list:
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        dep_list.append(dep)
    dep_list = np.asarray(dep_list)

    save_depth_list(dep_list, fn_list, dst, invalid_mask_list)
    viz_depth_list(dep_list, dst + ".mp4")
    return


if __name__ == "__main__":
    import os, os.path as osp
    import imageio

    model = get_depthcrafter_model(modify_scheduler=True)

    src = "./data/DAVIS_raw/JPEGImages/480p/breakdance-flare/"
    src = osp.expanduser(src)
    fns = os.listdir(src)
    fns.sort()
    img_list, fn_list = [], []

    H, W = 448, 832

    for fn in fns:
        if fn.endswith(".jpg") or fn.endswith(".png"):
            img = imageio.imread(os.path.join(src, fn))
            # crop the image to 448x832
            ori_H, ori_W, _ = img.shape
            img = img[
                (ori_H - H) // 2 : (ori_H + H) // 2, (ori_W - W) // 2 : (ori_W + W) // 2
            ]
            img_list.append(img)
            fn_list.append(fn)

    depthcrafter_process_folder(
        model, img_list, fn_list, dst="./debug/depth_crafter25", n_steps=25
    )

    print()
