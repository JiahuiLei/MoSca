# ! warning this is from iclr24 paper, should calibrate with the original neurips dycheck metrics
# ! the name in robust dynerf is different, this script is to double check with robust dynrf

import os, sys, os.path as osp
import logging, imageio
from tqdm import tqdm
import numpy as np
import torch
import lpips, time
import pandas as pd
import cv2 as cv

sys.path.append(osp.dirname(osp.abspath(__file__)))


def eval_dycheck(
    save_dir,
    gt_rgb_dir,
    gt_mask_dir,
    pred_dir,
    strict_eval_all_gt_flag=False,
    eval_non_masked=False,
    save_prefix="",
):

    # from ml-pgdvs
    # https://github.com/apple/ml-pgdvs/blob/13f9875eb0b7de6068452ec9c37231eda48581df/pgdvs/engines/trainer_pgdvs.py#L89
    # https://github.com/apple/ml-pgdvs/blob/13f9875eb0b7de6068452ec9c37231eda48581df/pgdvs/engines/trainer_pgdvs.py#L139
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    start_t = time.time()
    lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(device)
    logging.info(f"lpips_fn init time: {time.time() - start_t:.2f}s")

    gt_rgb_fns = sorted(os.listdir(gt_rgb_dir))
    gt_mask_fns = sorted(os.listdir(gt_mask_dir))
    pred_fns = sorted(os.listdir(pred_dir))
    
    # # ! for calibrate robust-dynrf
    # pred_fns = [fn.split("_")[1][2:] for fn in gt_rgb_fns]

    # # debug
    # pred_fns = pred_fns[:5]

    if pred_dir.endswith("/"):
        pred_dir = pred_dir[:-1]
    eval_name = osp.basename(pred_dir)
    save_viz_dir = osp.join(save_dir, f"{eval_name}_viz")
    os.makedirs(save_viz_dir, exist_ok=True)

    if strict_eval_all_gt_flag:
        assert (
            len(gt_rgb_fns) == len(gt_mask_fns) == len(pred_fns)
        ), "Number of files must match"
    else:
        pred_ids = [f[:-4] for f in pred_fns]
        if len(gt_rgb_fns) != len(pred_fns):
            logging.warning(
                f"Only eval predicted images {len(pred_ids)} < all gt {len(gt_rgb_fns)}"
            )
            assert len(gt_rgb_fns) == len(gt_mask_fns)
            filtered_gt_rgb_fns, filtered_gt_mask_fns = [], []
            for i in range(len(gt_rgb_fns)):
                if gt_rgb_fns[i][:-4] in pred_ids:
                    filtered_gt_rgb_fns.append(gt_rgb_fns[i])
                    filtered_gt_mask_fns.append(gt_mask_fns[i])
            gt_rgb_fns = filtered_gt_rgb_fns
            gt_mask_fns = filtered_gt_mask_fns
        assert (
            len(gt_rgb_fns) == len(gt_mask_fns) == len(pred_fns)
        ), "Number of files must match"

    psnr_list, ssim_list, lpips_list = [], [], []
    mpsnr_list, mssim_list, mlpips_list = [], [], []
    for i in tqdm(range(len(gt_rgb_fns))):
        gt = imageio.imread(osp.join(gt_rgb_dir, gt_rgb_fns[i])).astype(float) / 255.0
        gt_mask = (imageio.imread(osp.join(gt_mask_dir, gt_mask_fns[i])) > 0).astype(
            float
        )[..., None]
        pred = imageio.imread(osp.join(pred_dir, pred_fns[i])).astype(float) / 255.0
        full_mask = np.ones_like(gt_mask)

        # pred / gt: [H, W, 3], float32, range [0, 1]
        # covis_mask: [H, W, 3], float32, range [0, 1]

        import jax

        device_cpu = jax.devices("cpu")[0]
        with jax.default_device(device_cpu):
            from dycheck_metrics import compute_psnr, compute_ssim, compute_lpips

            if eval_non_masked:
                tmp_psnr = compute_psnr(gt, pred, full_mask).item()
                tmp_ssim = compute_ssim(gt, pred, full_mask).item()
                tmp_lpips = compute_lpips(
                    lpips_fn, gt, pred, full_mask, device=device
                ).item()
            else:
                tmp_psnr = 0.0
                tmp_ssim = 0.0
                tmp_lpips = 0.0

            # with covis mask
            tmp_mpsnr = compute_psnr(gt, pred, gt_mask).item()
            tmp_mssim = compute_ssim(gt, pred, gt_mask).item()
            tmp_mlpips = compute_lpips(
                lpips_fn, gt, pred, gt_mask, device=device
            ).item()

        psnr_list.append(tmp_psnr)
        ssim_list.append(tmp_ssim)
        lpips_list.append(tmp_lpips)
        mpsnr_list.append(tmp_mpsnr)
        mssim_list.append(tmp_mssim)
        mlpips_list.append(tmp_mlpips)

        # viz
        m_error = abs(pred - gt).max(axis=-1) * gt_mask.squeeze(-1)
        m_error = cv.applyColorMap((m_error * 255).astype(np.uint8), cv.COLORMAP_JET)[
            ..., [2, 1, 0]
        ]
        error = abs(pred - gt).max(axis=-1)
        error = cv.applyColorMap((error * 255).astype(np.uint8), cv.COLORMAP_JET)[
            ..., [2, 1, 0]
        ]
        viz_img = np.concatenate([gt * 255, pred * 255, error, m_error], axis=1).astype(
            np.uint8
        )
        imageio.imwrite(osp.join(save_viz_dir, f"{gt_rgb_fns[i]}"), viz_img)

    ave_psnr = np.mean(psnr_list)
    ave_ssim = np.mean(ssim_list)
    ave_lpips = np.mean(lpips_list)

    ave_mpsnr = np.mean(mpsnr_list)
    ave_mssim = np.mean(mssim_list)
    ave_mlpips = np.mean(mlpips_list)

    logging.info(
        f"ave_psnr: {ave_psnr:.2f}, ave_ssim: {ave_ssim:.4f}, ave_lpips: {ave_lpips:.4f}"
    )
    logging.info(
        f"ave_mpsnr: {ave_mpsnr:.2f}, ave_mssim: {ave_mssim:.4f}, ave_mlpips: {ave_mlpips:.4f}"
    )

    # * save and viz
    # save excel with pandas, each row is a frame
    df = pd.DataFrame(
        {
            "fn": ["AVE"],
            "psnr": [ave_psnr],
            "ssim": [ave_ssim],
            "lpips": [ave_lpips],
            "mpsnr": [ave_mpsnr],
            "mssim": [ave_mssim],
            "mlpips": [ave_mlpips],
        }
    )
    for i in range(len(gt_rgb_fns)):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "fn": [gt_rgb_fns[i]],
                        "psnr": [psnr_list[i]],
                        "ssim": [ssim_list[i]],
                        "lpips": [lpips_list[i]],
                        "mpsnr": [mpsnr_list[i]],
                        "mssim": [mssim_list[i]],
                        "mlpips": [mlpips_list[i]],
                    }
                ),
            ],
            ignore_index=True,
        )
    df.to_excel(osp.join(save_dir, f"{save_prefix}dycheck_metrics.xlsx"), index=False)

    viz_fns = sorted(
        [f for f in os.listdir(save_viz_dir) if "tto" not in f and f.endswith("jpg")]
    )
    frames = [imageio.imread(osp.join(save_viz_dir, f)) for f in viz_fns]
    imageio.mimsave(save_viz_dir + ".mp4", frames)
    return


if __name__ == "__main__":
    # gt_rgb_dir = "../../data/iphone/spin/test_images/"
    # gt_mask_dir = "../../data/iphone/spin/test_covisible/"
    # pred_rgb_dir = "../../data/iphone/spin/log/20240401_104110/test/"
    # save_dir = "../../data/iphone/spin/log/20240401_104110/"
    
    gt_rgb_dir = "../../data/iphone_final/spin/test_images/"
    gt_mask_dir = "../../data/iphone_final/spin/test_covisible/"
    
    pred_rgb_dir = "../../data/robust_dynrf/results/iPhone/Ours/spin"
    save_dir = "../../debug/calibrate_dycheck/robust_dynrf_spin"

    eval_dycheck(
        save_dir, gt_rgb_dir, gt_mask_dir, pred_rgb_dir, strict_eval_all_gt_flag=False
    )
