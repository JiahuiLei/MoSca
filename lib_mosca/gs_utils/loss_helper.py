import torch, numpy as np
import os, sys, os.path as osp

sys.path.append(osp.dirname(osp.abspath(__file__)))

from ssim_helper import ssim


def compute_rgb_loss(
    gt_rgb, render_dict: dict, sup_mask: torch.Tensor, ssim_lambda=0.1
):
    gt_rgb = gt_rgb.detach()
    pred_rgb = render_dict["rgb"].permute(1, 2, 0)
    sup_mask = sup_mask.float()
    rgb_loss_i = torch.abs(pred_rgb - gt_rgb.detach()) * sup_mask[..., None]
    rgb_loss = rgb_loss_i.sum() / sup_mask.sum()
    if ssim_lambda > 0:
        ssim_loss = 1.0 - ssim(
            (render_dict["rgb"] * sup_mask[None, ...])[None],
            (gt_rgb.permute(2, 0, 1) * sup_mask[None, ...])[None],
        )
        rgb_loss = rgb_loss + ssim_loss * ssim_lambda
    return rgb_loss, rgb_loss_i, pred_rgb, gt_rgb


def compute_dep_loss(
    target_dep,
    render_dict: dict,
    sup_mask: torch.Tensor,
    st_invariant=True,
):
    # pred_dep = render_dict["dep"][0] / torch.clamp(render_dict["alpha"][0], min=1e-6)
    # ! warning, gof does not need divide alpha
    pred_dep = render_dict["dep"][0]
    target_dep = target_dep.detach()
    if st_invariant:
        prior_t = torch.median(target_dep[sup_mask > 0.5])
        pred_t = torch.median(pred_dep[sup_mask > 0.5])
        prior_s = (target_dep[sup_mask > 0.5] - prior_t).abs().mean()
        pred_s = (pred_dep[sup_mask > 0.5] - pred_t).abs().mean()
        prior_dep_norm = (target_dep - prior_t) / prior_s
        pred_dep_norm = (pred_dep - pred_t) / pred_s
    else:
        prior_dep_norm = target_dep
        pred_dep_norm = pred_dep
    sup_mask = sup_mask.float()
    loss_dep_i = torch.abs(pred_dep_norm - prior_dep_norm) * sup_mask
    loss_dep = loss_dep_i.sum() / sup_mask.sum()
    return loss_dep, loss_dep_i, pred_dep, target_dep


def compute_normal_loss(gt_normal, render_dict: dict, sup_mask: torch.Tensor):
    # ! below two normals are all in camera frame, pointing towards camera
    # gt_normal # H,W,3
    pred_normal = render_dict["normal"].permute(1, 2, 0)
    loss, error = __normal_loss__(gt_normal.detach(), pred_normal, sup_mask)
    return loss, error, pred_normal, gt_normal


def __normal_loss__(gt_normal, pred_normal, sup_mask):
    valid_gt_mask = gt_normal.norm(dim=-1) > 1e-6
    sup_mask = sup_mask * valid_gt_mask
    normal_error1 = 1 - (pred_normal * gt_normal).sum(-1)
    normal_error2 = 1 + (pred_normal * gt_normal).sum(-1)
    error = torch.min(normal_error1, normal_error2)
    loss = (error * sup_mask).sum() / sup_mask.sum()
    return loss, error


def compute_dep_reg_loss(gt_rgb, render_dict):
    # ! for now, the reg loss has nothing to do with mask .etc, everything is intrinsic
    gt_rgb = gt_rgb.permute(2, 0, 1)
    distortion_map = render_dict["distortion_map"][0]
    distortion_map = get_edge_aware_distortion_map(gt_rgb.detach(), distortion_map)
    distortion_loss = distortion_map.mean()
    return distortion_loss, distortion_map


def compute_normal_reg_loss(s2d, cams, render_dict):
    # ! for now, the reg loss has nothing to do with mask .etc, everything is intrinsic
    dep = render_dict["dep"][0]
    dep_mask = dep > 0
    pts = cams.backproject(s2d.homo_map[dep_mask].detach(), dep[dep_mask])
    v_map = torch.zeros_like(render_dict["rgb"]).permute(1, 2, 0)
    v_map[dep_mask] = v_map[dep_mask] + pts

    dep_normal = torch.zeros_like(v_map)
    dx = torch.cat([v_map[2:, 1:-1] - v_map[:-2, 1:-1]], dim=0)
    dy = torch.cat([v_map[1:-1, 2:] - v_map[1:-1, :-2]], dim=1)
    dep_normal[1:-1, 1:-1, :] = torch.nn.functional.normalize(
        torch.cross(dx, dy, dim=-1), dim=-1
    )
    pred_normal = render_dict["normal"].permute(1, 2, 0)
    loss, error = __normal_loss__(dep_normal, pred_normal, dep_mask)
    return loss, error, pred_normal, dep_normal


def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(
        torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0
    )
    grad_img_right = torch.mean(
        torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0
    )
    grad_img_top = torch.mean(
        torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0
    )
    grad_img_bottom = torch.mean(
        torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0
    )
    max_grad = torch.max(
        torch.stack(
            [grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1
        ),
        dim=-1,
    )[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad
