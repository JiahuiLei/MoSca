from matplotlib import pyplot as plt
import torch, numpy as np
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
from transforms3d.euler import mat2euler, euler2mat
import logging
import imageio
import os, sys, os.path as osp
from tqdm import tqdm

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))

from lib_render.gauspl_renderer_native import render_cam_pcl
from lib_render.sh_utils import RGB2SH, SH2RGB


from matplotlib import cm
import cv2 as cv
import glob

import torch.nn.functional as F

TEXTCOLOR = (255, 0, 0)
BORDER_COLOR = (100, 255, 100)
BG_COLOR1 = [1.0, 1.0, 1.0]


def __video_save__(fn, imgs, fps=10):
    logging.info(f"Saving video to {fn} ...")
    # H, W = imgs[0].shape[:2]
    # image_size = (W, H)
    # out = cv.VideoWriter(fn, cv.VideoWriter_fourcc(*"MP4V"), fps, image_size)
    # for img in tqdm(imgs):
    #     out.write(img[..., ::-1])
    # out.release()
    imageio.mimsave(fn, imgs)
    logging.info(f"Saved!")
    return


def make_video_from_pattern(pattern, dst):
    fns = glob.glob(pattern)
    fns.sort()
    frames = []
    for fn in fns:
        frames.append(imageio.imread(fn))
    __video_save__(dst, frames)
    return


def viz_global_ba(
    point_world,
    rgb,
    mask,
    cams,
    pts_size=0.001,
    res=480,
    error=None,
    text="",
):
    T, M = point_world.shape[:2]
    device = point_world.device
    mu = point_world.clone()[mask]
    sph = RGB2SH(rgb.clone()[mask])
    s = torch.ones_like(mu) * pts_size
    fr = torch.eye(3, device=device).expand(len(mu), -1, -1)
    o = torch.ones(len(mu), 1, device=device)
    viz_cam_R = quaternion_to_matrix(cams.q_wc)
    viz_cam_t = cams.t_wc
    viz_cam_R, viz_cam_t = cams.Rt_wc_list()
    viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
    frame_dict = viz_scene(
        res,
        res,
        viz_cam_R,
        viz_cam_t,
        viz_f=viz_f,
        gs5_param=(mu, fr, s, o, sph),
        bg_color=BG_COLOR1,
        draw_camera_frames=True,
    )
    frame = np.concatenate([v for v in frame_dict.values()], 1)
    # * also do color and error if provided
    id_color = cm.hsv(np.arange(M, dtype=np.float32) / M)[:, :3]
    id_color = torch.from_numpy(id_color).to(device)
    id_color = id_color[None].expand(T, -1, -1)
    sph = RGB2SH(id_color[mask])
    id_frame_dict = viz_scene(
        res,
        res,
        viz_cam_R,
        viz_cam_t,
        viz_f=viz_f,
        gs5_param=(mu, fr, s, o, sph),
        bg_color=BG_COLOR1,
        draw_camera_frames=True,
    )
    if error is None:
        id_frame = np.concatenate([v for v in id_frame_dict.values()], 1)
        frame = np.concatenate([frame, id_frame], 0)
    else:
        # render error as well
        error = error[mask]
        error_th = error.max()
        error_color = (error / (error_th + 1e-9)).detach().cpu().numpy()
        text = text + f" ErrorVizTh={error_th:.6f}"
        error_color = cm.viridis(error_color)[:, :3]
        error_color = torch.from_numpy(error_color).to(device)
        sph = RGB2SH(error_color)
        error_frame_dict = viz_scene(
            res,
            res,
            viz_cam_R,
            viz_cam_t,
            viz_f=viz_f,
            gs5_param=(mu, fr, s, o, sph),
            bg_color=BG_COLOR1,
            draw_camera_frames=True,
        )
        add_frame = np.concatenate(
            [
                id_frame_dict["scene_camera_20deg"],
                error_frame_dict["scene_camera_20deg"],
            ],
            1,
        )
        frame = np.concatenate([frame, add_frame], 0)
    # imageio.imsave("./debug/viz.jpg", frame)
    frame = frame.copy()
    if len(text) > 0:
        font = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (1.0, 1.0, 1.0)
        lineType = 2
        cv.putText(
            frame,
            text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
    return frame


@torch.no_grad()
def viz_curve(
    curve_world,
    mask,
    cams,
    res=480,
    pts_size=0.0003,
    viz_n=128,
    # n_line=8,
    # ! debug
    n_line=64,
    line_radius_factor=0.1,
    only_viz_last_frame=False,
    text="",
    time_window=10,
):
    # TODO: add error viz
    # draw the curve
    T, M = curve_world.shape[:2]
    device = curve_world.device
    if viz_n == -1:
        viz_n = M
        choice = torch.arange(M).to(device)
    else:
        step = max(1, M // viz_n)
        choice = torch.arange(M)[::step].to(device)
        viz_n = len(choice)
    mu = curve_world[:, choice].clone()
    mask = mask[:, choice].clone()
    id_sph = torch.from_numpy(
        cm.hsv(np.arange(viz_n, dtype=np.float32) / viz_n)[:, :3]
    ).to(mu)
    id_sph = RGB2SH(id_sph)

    frame_list = []

    line_mu = torch.zeros([0, 3], device=device)
    line_fr = torch.zeros([0, 3, 3], device=device)
    line_s = torch.zeros([0, 3], device=device)
    line_o = torch.zeros([0, 1], device=device)
    line_sph = torch.zeros([0, 3], device=device)

    # only draw stuff within a time window!
    for t in tqdm(range(T)):
        # draw the continuous geo
        _mu, _mask = (
            mu[max(0, t - time_window + 1) : t + 1].reshape(-1, 3),
            mask[max(0, t - time_window + 1) : t + 1].reshape(-1),
        )
        _s = torch.ones_like(_mu) * pts_size
        _fr = torch.eye(3, device=device).expand(len(_mu), -1, -1)
        _o = torch.ones(len(_mu), 1, device=device)

        if t == 0:
            line_mu = _mu
            line_fr = _fr
            line_s = _s
            line_o = _o
            line_sph = id_sph
        else:
            # ! this can't grow as time grows, for super long sequence !!
            prev_mu = mu[t - 1]
            cur_mu = mu[t]
            _line_mu = (
                prev_mu[None]
                + torch.linspace(0.0, 1.0, n_line, device=device)[:, None, None]
                * (cur_mu - prev_mu)[None]
            )
            _line_fr = torch.eye(3, device=device)[None, None].expand(
                n_line, viz_n, -1, -1
            )
            _line_s = (
                torch.ones(n_line, viz_n, 3, device=device)
                * pts_size
                * line_radius_factor
            )
            _line_s[0] = pts_size
            _line_s[-1] = pts_size
            _line_o = torch.ones(n_line, viz_n, 1, device=device)
            _line_sph = id_sph[None].expand(n_line, -1, -1)

            _N = viz_n * n_line * time_window
            line_mu = torch.cat([line_mu, _line_mu.reshape(-1, 3)], 0)[-_N:]
            line_fr = torch.cat([line_fr, _line_fr.reshape(-1, 3, 3)], 0)[-_N:]
            line_s = torch.cat([line_s, _line_s.reshape(-1, 3)], 0)[-_N:]
            line_o = torch.cat([line_o, _line_o.reshape(-1, 1)], 0)[-_N:]
            line_sph = torch.cat([line_sph, _line_sph.reshape(-1, 3)], 0)[-_N:]

        # render
        if only_viz_last_frame and t < T - 1:
            continue

        # viz_cam_R = quaternion_to_matrix(cams.q_wc)[: t + 1]
        # viz_cam_t = cams.t_wc[: t + 1]
        viz_cam_R, viz_cam_t = cams.Rt_wc_list()
        viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
        frame_dict = viz_scene(
            res,
            res,
            viz_cam_R,
            viz_cam_t,
            viz_f=viz_f,
            gs5_param=(line_mu, line_fr, line_s, line_o, line_sph),
            bg_color=BG_COLOR1,
            draw_camera_frames=False,
        )
        frame = np.concatenate([v for v in frame_dict.values()], 1).copy()
        # frame = np.concatenate([frame, frame_line], 0).copy()

        if len(text) > 0:
            font = cv.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = (1.0, 1.0, 1.0)
            lineType = 2
            cv.putText(
                frame,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
        frame_list.append(frame)
    return frame_list


def get_global_viz_cam_Rt(
    mu_w,
    param_cam_R_wc,
    param_cam_t_wc,
    viz_f,
    z_downward_deg=0.0,
    factor=1.0,
    auto_zoom_mask=None,
    scene_center_mode="mean",
    shift_margin_ratio=1.5,
):
    # always looking towards the scene center
    if scene_center_mode == "mean":
        scene_center = mu_w.mean(0)
    else:
        scene_bound_max, scene_bound_min = mu_w.max(0)[0], mu_w.min(0)[0]
        scene_center = (scene_bound_max + scene_bound_min) / 2.0
    cam_center = param_cam_t_wc.mean(0)

    cam_z_direction = F.normalize(scene_center - cam_center, dim=0)
    cam_y_direction = F.normalize(
        torch.cross(cam_z_direction, param_cam_R_wc[0, :, 0]),
        dim=0,
    )
    cam_x_direction = F.normalize(
        torch.cross(cam_y_direction, cam_z_direction),
        dim=0,
    )
    R_wc = torch.stack([cam_x_direction, cam_y_direction, cam_z_direction], 1)
    additional_R = euler2mat(-np.deg2rad(z_downward_deg), 0, 0, "rxyz")
    additional_R = torch.as_tensor(additional_R).to(R_wc)
    R_wc = R_wc @ additional_R
    # transform the mu to cam_R and then identify the distance
    mu_viz_cam = (mu_w - scene_center[None, :]) @ R_wc.T
    desired_shift = (
        viz_f / factor * mu_viz_cam[:, :2].abs().max(-1)[0] - mu_viz_cam[:, 2]
    )
    # # the nearest point should be in front of camera!
    # nearest_shift =   mu_viz_cam[:, :2].mean()
    # desired_shift = max(desired_shift)
    if auto_zoom_mask is not None:
        desired_shift = desired_shift[auto_zoom_mask]
    shift = desired_shift.max() * shift_margin_ratio
    t_wc = -R_wc[:, -1] * shift + scene_center
    return R_wc, t_wc


@torch.no_grad()
def viz_scene(
    H,
    W,
    param_cam_R_wc,
    param_cam_t_wc,
    model=None,
    viz_f=40.0,
    save_name=None,
    viz_first_n_cam=-1,
    gs5_param=None,
    bg_color=[1.0, 1.0, 1.0],
    draw_camera_frames=False,
    return_full=False,
):
    # auto select viewpoint
    # manually add the camera viz to to
    if model is None:
        assert gs5_param is not None
        mu_w, fr_w, s, o, sph = gs5_param
    else:
        mu_w, fr_w, s, o, sph = model()
    # add the cameras to the GS
    if draw_camera_frames:
        mu_w, fr_w, s, o, sph, cam_viz_mask = add_camera_frame(
            (mu_w, fr_w, s, o, sph), param_cam_R_wc, param_cam_t_wc, viz_first_n_cam
        )

    # * prepare the viz camera
    # * (1) global scene viz
    # viz camera set manually
    global_R_wc, global_t_wc = get_global_viz_cam_Rt(
        mu_w, param_cam_R_wc, param_cam_t_wc, viz_f
    )
    global_down20_R_wc, global_down20_t_wc = get_global_viz_cam_Rt(
        mu_w, param_cam_R_wc, param_cam_t_wc, viz_f, 20
    )
    if draw_camera_frames:
        camera_R_wc, camera_t_wc = get_global_viz_cam_Rt(
            mu_w,
            param_cam_R_wc,
            param_cam_t_wc,
            viz_f,
            factor=0.5,
            auto_zoom_mask=cam_viz_mask,
        )
        camera_down20_R_wc, camera_down20_t_wc = get_global_viz_cam_Rt(
            mu_w,
            param_cam_R_wc,
            param_cam_t_wc,
            viz_f,
            20,
            factor=0.5,
            auto_zoom_mask=cam_viz_mask,
        )

    ret = {}
    ret_full = {}
    todo = {  # "scene_global": (global_R_wc, global_t_wc),
        "scene_global_20deg": (global_down20_R_wc, global_down20_t_wc)
    }
    if draw_camera_frames:
        # todo["scene_camera"] = (camera_R_wc, camera_t_wc)
        todo["scene_camera_20deg"] = (camera_down20_R_wc, camera_down20_t_wc)
    for name, Rt in todo.items():
        viz_cam_R_wc, viz_cam_t_wc = Rt
        viz_cam_R_cw = viz_cam_R_wc.transpose(1, 0)
        viz_cam_t_cw = -viz_cam_R_cw @ viz_cam_t_wc
        viz_mu = torch.einsum("ij,nj->ni", viz_cam_R_cw, mu_w) + viz_cam_t_cw[None]
        viz_fr = torch.einsum("ij,njk->nik", viz_cam_R_cw, fr_w)

        pf = viz_f / 2 * min(H, W)
        assert len(viz_mu) == len(sph)
        render_dict = render_cam_pcl(
            viz_mu, viz_fr, s, o, sph, H=H, W=W, fx=pf, bg_color=bg_color
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        ret[name] = rgb
        if return_full:
            ret_full[name] = render_dict
        if save_name is not None:
            base_name = osp.basename(save_name)
            dir_name = osp.dirname(save_name)
            os.makedirs(dir_name, exist_ok=True)
            save_img = np.clip(ret[name] * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(osp.join(dir_name, f"{name}_{base_name}.jpg"), save_img)
    if return_full:
        return ret, ret_full
    return ret


def add_camera_frame(
    gs5_param, cam_R_wc, cam_t_wc, viz_first_n_cam=-1, add_global=False
):
    mu_w, fr_w, s, o, sph = gs5_param
    N_scene = len(mu_w)
    if viz_first_n_cam <= 0:
        viz_first_n_cam = len(cam_R_wc)
    for i in range(viz_first_n_cam):
        if cam_R_wc.ndim == 2:
            R_wc = quaternion_to_matrix(F.normalize(cam_R_wc[i : i + 1], dim=-1))[0]
        else:
            assert cam_R_wc.ndim == 3
            R_wc = cam_R_wc[i]
        t_wc = cam_t_wc[i]
        _mu, _fr, _s, _o, _sph = draw_frame(
            R_wc.clone(), t_wc.clone(), size=0.1, weight=0.0003
        )
        # pad the _sph to have same order with the input
        if sph.shape[1] > 3:
            _sph = torch.cat(
                [_sph, torch.zeros(len(_sph), sph.shape[1] - 3).to(_sph)], dim=1
            )
        mu_w, fr_w, s, o, sph = cat_gs(mu_w, fr_w, s, o, sph, _mu, _fr, _s, _o, _sph)
    if add_global:
        _mu, _fr, _s, _o, _sph = draw_frame(
            torch.eye(3).to(s),
            torch.zeros(3).to(s),
            size=0.3,
            weight=0.001,
            color=[0.5, 0.5, 0.5],
            opa=0.3,
        )
        mu_w, fr_w, s, o, sph = cat_gs(mu_w, fr_w, s, o, sph, _mu, _fr, _s, _o, _sph)
    cam_pts_mask = torch.zeros_like(o.squeeze(-1)).bool()
    cam_pts_mask[N_scene:] = True
    return mu_w, fr_w, s, o, sph, cam_pts_mask


def draw_frame(R_wc, t_wc, size=0.1, weight=0.01, color=None, opa=1.0):
    if not isinstance(R_wc, torch.Tensor):
        R_wc = torch.as_tensor(R_wc)
    if not isinstance(t_wc, torch.Tensor):
        t_wc = torch.as_tensor(t_wc)
    origin = t_wc
    for i in range(3):
        end = t_wc + size * R_wc[:, i]
        if color is None:
            _color = torch.eye(3)[i].to(R_wc)
        else:
            _color = torch.as_tensor(color).to(R_wc)
        _mu, _fr, _s, _o, _c = draw_line(origin, end, weight, _color, opa)
        if i == 0:
            mu, fr, s, o, rgb = _mu, _fr, _s, _o, _c
        else:
            mu, fr, s, o, rgb = cat_gs(mu, fr, s, o, rgb, _mu, _fr, _s, _o, _c)
    return mu, fr, s, o, rgb


def cat_gs(m1, f1, s1, o1, c1, m2, f2, s2, o2, c2):
    m = torch.cat([m1, m2], dim=0).contiguous()
    f = torch.cat([f1, f2], dim=0).contiguous()
    s = torch.cat([s1, s2], dim=0).contiguous()
    o = torch.cat([o1, o2], dim=0).contiguous()
    c = torch.cat([c1, c2], dim=0).contiguous()
    return m, f, s, o, c


def draw_line(start, end, radius, rgb, opa=1.0):
    if not isinstance(start, torch.Tensor):
        start = torch.as_tensor(start)
    if not isinstance(end, torch.Tensor):
        end = torch.as_tensor(end)
    line_len = torch.norm(end - start)
    assert line_len > 0
    N = line_len / radius * 3
    line_dir = (end - start) / line_len
    # draw even points on the line
    mu = torch.linspace(0, float(line_len), int(N)).to(start)
    mu = start + mu[:, None] * line_dir[None]
    fr = torch.eye(3)[None].to(mu).expand(len(mu), -1, -1)
    s = radius * torch.ones(len(mu), 3).to(mu)
    o = opa * torch.ones(len(mu), 1).to(mu)
    assert len(rgb) == 3
    c = torch.as_tensor(rgb)[None].to(mu) * torch.ones(len(mu), 3).to(mu)
    c = RGB2SH(c)
    return mu, fr, s, o, c
