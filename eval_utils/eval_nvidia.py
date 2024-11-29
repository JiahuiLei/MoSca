import os
import cv2
import lpips
import torch
import numpy as np
from skimage.metrics import structural_similarity
import os.path as osp
import pandas as pd
import logging


def im2tensor(img):
    return torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]


def readimage(data_dir, sequence, time, method):
    img = cv2.imread(
        os.path.join(data_dir, method, sequence, "v000_t" + str(time).zfill(3) + ".png")
    )
    return img


def eval_nvidia_dir(gt_dir, pred_dir, report_dir, fixed_view_id=0):
    lpips_loss = lpips.LPIPS(net="alex")  # best forward scores
    assert osp.exists(gt_dir)
    assert osp.exists(pred_dir)
    os.makedirs(report_dir, exist_ok=True)
    viz_dir = osp.join(report_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    assert "Yoon" not in pred_dir, "Check back the original eval code!"
    time_start = 0
    time_end = 12

    psnr_list, ssim_list, lpips_list = [], [], []
    results = []
    for time in range(time_start, time_end):  # Fix view v0, change time
        gt_fn = osp.join(gt_dir, f"v{fixed_view_id:03d}_t{time:03d}.png")
        pred_fn = osp.join(pred_dir, f"v{fixed_view_id:03d}_t{time:03d}.png")
        img_true = cv2.imread(gt_fn)
        img = cv2.imread(pred_fn)

        _psnr = cv2.PSNR(img_true, img)
        _ssim = structural_similarity(img_true, img, multichannel=True)
        _lpips = lpips_loss.forward(im2tensor(img_true), im2tensor(img)).item()

        psnr_list.append(_psnr)
        ssim_list.append(_ssim)
        lpips_list.append(_lpips)

        results.append({"time": time, "psnr": _psnr, "ssim": _ssim, "lpips": _lpips})

        # save the psnr error map to report dir
        errmap = np.sqrt(((img - img_true) ** 2).sum(-1)) / np.sqrt(3)
        # ! normalize
        factor = errmap.max()
        errmap = errmap / factor * 255
        errmap = cv2.applyColorMap((errmap).astype(np.uint8), cv2.COLORMAP_JET)
        # put the text normalization onto the error map
        errmap = cv2.putText(
            errmap,
            f"NORM[{factor:.3f}] PSNR={_psnr:.3f}, SSIM={_ssim:.3f}, LPIPS={_lpips:.3f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        save_viz = np.concatenate([img_true, img, errmap], axis=1)
        save_name = osp.join(
            viz_dir, f"v{fixed_view_id:03d}_t{time:03d}_psnr={_psnr:.3f}.png"
        )
        cv2.imwrite(save_name, save_viz)

    ave_psnr = np.mean(psnr_list)
    ave_ssim = np.mean(ssim_list)
    ave_lpips = np.mean(lpips_list)
    # save the txt of three metrics to report dir
    with open(osp.join(report_dir, "nvidia_render_metrics.txt"), "w") as f:
        f.write(f"PSNR: {ave_psnr:.6f}\n")
        f.write(f"SSIM: {ave_ssim:.6f}\n")
        f.write(f"LPIPS: {ave_lpips:.6f}\n")

    # build pandas dataframe and save to report dir an excel
    results = [
        {"time": "ave", "psnr": ave_psnr, "ssim": ave_ssim, "lpips": ave_lpips}
    ] + results
    # save each row as a time frame
    df = pd.DataFrame(results)
    save_report_fn = osp.join(report_dir, "nvidia_render_metrics.xlsx")
    df.to_excel(save_report_fn)
    logging.info(f"Saved the evaluation report to {report_dir}")
    logging.info(
        f"Metric: PSNR={ave_psnr:.6f}, SSIM={ave_ssim:.6f}, LPIPS={ave_lpips:.6f}"
    )
    return ave_psnr, ave_ssim, ave_lpips


def calculate_metrics(data_dir, sequence, methods, lpips_loss):

    PSNRs = np.zeros((len(methods)))
    SSIMs = np.zeros((len(methods)))
    LPIPSs = np.zeros((len(methods)))

    nFrame = 0

    # Yoon's results do not include v000_t000 and v000_t011. Omit these two
    # frames if evaluating Yoon's method.
    if "Yoon" in methods:
        time_start = 1
        time_end = 11
    else:
        time_start = 0
        time_end = 12

    for time in range(time_start, time_end):  # Fix view v0, change time

        nFrame += 1

        img_true = readimage(data_dir, sequence, time, "gt")

        for method_idx, method in enumerate(methods):

            if "Yoon" in methods and sequence == "Truck" and time == 10:
                break

            img = readimage(data_dir, sequence, time, method)
            PSNR = cv2.PSNR(img_true, img)
            SSIM = structural_similarity(img_true, img, multichannel=True)
            LPIPS = lpips_loss.forward(im2tensor(img_true), im2tensor(img)).item()

            PSNRs[method_idx] += PSNR
            SSIMs[method_idx] += SSIM
            LPIPSs[method_idx] += LPIPS

    PSNRs = PSNRs / nFrame
    SSIMs = SSIMs / nFrame
    LPIPSs = LPIPSs / nFrame

    return PSNRs, SSIMs, LPIPSs


# if __name__ == "__main__":

#     eval_nvidia_dir(
#         gt_dir="/home/ray/projects/vid24d/data/robust_dynrf/results/Nvidia/gt/Balloon1/",
#         pred_dir="/home/ray/projects/vid24d/data/robust_dynrf/results/Nvidia/RobustDynrf/Balloon1/",
#         report_dir="/home/ray/projects/vid24d/debug/eval_report",
#     )

#     lpips_loss = lpips.LPIPS(net="alex")  # best forward scores
#     data_dir = "/home/ray/projects/vid24d/data/robust_dynrf/results/Nvidia"
#     sequences = [
#         # "Jumping",
#         # "Skating",
#         # "Truck",
#         # "Umbrella",
#         "Balloon1",
#         # "Balloon2",
#         # "Playground",
#     ]

#     methods = [
#         # "NeRF",
#         # "NeRF_t",
#         # "DNeRF",
#         # "NR",
#         # "NSFF",
#         # "DynamicNeRF",
#         # "HyperNeRF",
#         # "TiNeuVox",
#         "RobustDynrf",
#         "RobustDynrf_wo_COLMAP",
#         "4Dyition",
#     ]

#     # sequences = ['Jumping', 'Skating', 'Truck', 'Umbrella', 'Balloon1', 'Balloon2', 'Playground', 'DynamicFace', 'Teadybear']
#     # methods = ['Ours_wo_COLMAP']

#     PSNRs_total = np.zeros((len(methods)))
#     SSIMs_total = np.zeros((len(methods)))
#     LPIPSs_total = np.zeros((len(methods)))
#     for sequence in sequences:
#         print(sequence)
#         PSNRs, SSIMs, LPIPSs = calculate_metrics(
#             data_dir, sequence, methods, lpips_loss
#         )
#         for method_idx, method in enumerate(methods):
#             print(
#                 method.ljust(7)
#                 + " %.2f" % (PSNRs[method_idx])
#                 + " / %.4f" % (SSIMs[method_idx])
#                 + " / %.3f" % (LPIPSs[method_idx])
#             )

#         PSNRs_total += PSNRs
#         SSIMs_total += SSIMs
#         LPIPSs_total += LPIPSs

#     PSNRs_total = PSNRs_total / len(sequences)
#     SSIMs_total = SSIMs_total / len(sequences)
#     LPIPSs_total = LPIPSs_total / len(sequences)
#     print("Avg.")
#     for method_idx, method in enumerate(methods):
#         print(
#             method.ljust(7)
#             + " %.2f" % (PSNRs_total[method_idx])
#             + " / %.4f" % (SSIMs_total[method_idx])
#             + " / %.3f" % (LPIPSs_total[method_idx])
#         )


if __name__ == "__main__":
    # PSNRs, SSIMs, LPIPSs = calculate_metrics(data_dir, sequence, methods, lpips_loss)
    # eval_nvidia_dir(
    #     gt_dir="../../data/robust_dynrf/results/Nvidia/gt/Umbrella/", pred_dir="../../data/robust_dynrf/results/Nvidia/RobustDynrf/Umbrella/", report_dir="../../debug/umbrella_repot", fixed_view_id=0
    # )

    # eval_nvidia_dir(
    #     gt_dir="../../data/robust_dynrf/results/Nvidia/gt/Skating/",
    #     pred_dir="../../data/robust_dynrf/results/Nvidia/DynamicNeRF/Skating/",
    #     report_dir="../../debug/skating_dynNeRF",
    #     fixed_view_id=0,
    # )

    # eval_nvidia_dir(
    #     gt_dir="../../data/robust_dynrf/results/Nvidia/gt/Playground/",
    #     pred_dir="../../data/robust_dynrf/results/Nvidia/RobustDynrf/Playground/",
    #     report_dir="../../debug/Playground_RobustDynrf",
    #     fixed_view_id=0,
    # )

    # eval_nvidia_dir(
    #     gt_dir="../../data/robust_dynrf/results/Nvidia/gt/Playground/",
    #     pred_dir="../../data/nvidia_dev_H/Playground/log/nvidia_balloon1_balloon2.yaml20240505_132627/test/",
    #     report_dir="../../debug/Playground_ours",
    #     fixed_view_id=0,
    # )

    # eval_nvidia_dir(
    #     gt_dir="../../data/robust_dynrf/results/Nvidia/gt/Skating/",
    #     pred_dir="../../data/nvidia_dev_K/Skating/log/nvidia_balloon1_balloon2.yaml20240505_050105/test/",
    #     report_dir="../../debug/skating_ours",
    #     fixed_view_id=0,
    # )

    eval_nvidia_dir(
        gt_dir="../../data/robust_dynrf/results/Nvidia/gt/Skating/",
        # pred_dir="../../data/nvidia_dev_old/Skating/log/20240505_221534/nvidia_eval",
        pred_dir="../../data/nvidia_dev_old/Skating/log/20240505_230711/nvidia_eval",
        report_dir="../../debug/skating_ours_old2",
        fixed_view_id=0,
    )

    # eval_nvidia_dir(
    #     # gt_dir="../../data/robust_dynrf/results/Nvidia/gt/Skating/",
    #     # gt_dir="../../data/nvidia_dev_old/Skating/log/20240505_221534_good/nvidia_eval",
    #     gt_dir="../../data/nvidia_dev_old/Skating/log/20240505_230711/nvidia_eval",
    #     pred_dir="../../data/nvidia_dev_H/Skating/log/nvidia.yaml20240505_234218/test/",
    #     report_dir="../../debug/skating_ours_old_as_gt3",
    #     fixed_view_id=0,
    # )
