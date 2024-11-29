from matplotlib import pyplot as plt
import torch, numpy as np
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
import logging
import imageio
import os, sys, os.path as osp
from tqdm import tqdm

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))

from lib_render.render_helper import render
from lib_mosca.gs_utils.loss_helper import (
    compute_rgb_loss,
    compute_dep_loss,
    compute_normal_loss,
)
from lib_mosca.scaffold_utils.viz_helper import viz_curve
from lib_mosca.camera import MonocularCameras
from lib_render.gauspl_renderer_native import render_cam_pcl
from lib_render.sh_utils import RGB2SH, SH2RGB
from lib_prior.prior_loading import Saved2D

import torch.nn.functional as F
from matplotlib import cm
import cv2 as cv
import glob
from transforms3d.euler import mat2euler, euler2mat
import imageio.core.util

TEXTCOLOR = (255, 0, 0)
BORDER_COLOR = (100, 255, 100)
BG_COLOR1 = [1.0, 1.0, 1.0]


def make_video_from_pattern(pattern, dst):
    fns = glob.glob(pattern)
    fns.sort()
    frames = []
    for fn in fns:
        frames.append(imageio.imread(fn))
    __video_save__(dst, frames)
    return


@torch.no_grad()
def make_viz_np(
    gt,
    pred,
    error,
    error_cm=cv.COLORMAP_WINTER,
    img_cm=cv.COLORMAP_VIRIDIS,
    text0="target",
    text1="pred",
    text2="error",
    gt_margin=5,
    print_text=True,
):
    assert error.ndim == 2
    error = (error / error.max()).detach().cpu().numpy()
    error = (error * 255).astype(np.uint8)
    error = cv.applyColorMap(error, error_cm)[:, :, ::-1]
    viz_frame = torch.cat([gt, pred], 1)
    if viz_frame.ndim == 2:
        viz_frame = viz_frame / viz_frame.max()
    viz_frame = viz_frame.detach().cpu().numpy()
    viz_frame = np.clip(viz_frame * 255, 0, 255).astype(np.uint8)
    if viz_frame.ndim == 2:
        viz_frame = cv.applyColorMap(viz_frame, img_cm)[:, :, ::-1]
    viz_frame = np.concatenate([viz_frame, error], 1)
    # split the image to 3 draw the text onto the image
    viz_frame_list = np.split(viz_frame, 3, 1)
    # draw green border of GT target, don't pad, draw inside

    if print_text:
        viz_frame_list[0] = cv.copyMakeBorder(
            viz_frame_list[0][gt_margin:-gt_margin, gt_margin:-gt_margin],
            gt_margin,
            gt_margin,
            gt_margin,
            gt_margin,
            cv.BORDER_CONSTANT,
            value=BORDER_COLOR,
        )
        for i, text in enumerate([text0, text1, text2]):
            if len(text) > 0:
                font = cv.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 30)
                fontScale = 1
                fontColor = TEXTCOLOR
                lineType = 2
                cv.putText(
                    viz_frame_list[i],
                    text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )
    viz_frame = np.concatenate(viz_frame_list, 1)
    return viz_frame


def __get_move_around_cam_T_cw__(
    s_model,
    d_model,
    cams: MonocularCameras,
    move_around_id,
    move_around_angle_deg,
    start_from,
    end_at,
):
    gs5 = [s_model()]
    if d_model is not None:
        gs5.append(d_model(move_around_id))
    render_dict = render(
        gs5,
        cams.default_H,
        cams.default_W,
        cams.default_K,
        cams.T_cw(move_around_id),
    )
    # depth = (render_dict["dep"] / (render_dict["alpha"] + 1e-6))[0]
    depth = render_dict["dep"][0]
    center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
    if center_dep < 1e-2:
        try:
            center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
        except:
            center_dep = 1.0
    focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)

    move_around_radius = np.tan(move_around_angle_deg) * focus_point[2].item()
    # in the xy plane, the new camera is forming a circle
    total_steps = end_at - start_from + 1
    move_around_view_list = []
    for i in range(total_steps):
        x = move_around_radius * np.cos(2 * np.pi * i / (total_steps - 1))
        y = move_around_radius * np.sin(2 * np.pi * i / (total_steps - 1))
        T_c_new = torch.eye(4).to(cams.T_wc(0))
        T_c_new[0, -1] = x
        T_c_new[1, -1] = y
        _z_dir = F.normalize(focus_point[:3] - T_c_new[:3, -1], dim=0)
        _x_dir = F.normalize(
            torch.cross(torch.Tensor([0.0, 1.0, 0.0]).to(_z_dir), _z_dir), dim=0
        )
        _y_dir = F.normalize(torch.cross(_z_dir, _x_dir), dim=0)
        T_c_new[:3, 0] = _x_dir
        T_c_new[:3, 1] = _y_dir
        T_c_new[:3, 2] = _z_dir
        T_w_new = cams.T_wc(move_around_id) @ T_c_new
        T_new_w = T_w_new.inverse()
        move_around_view_list.append(T_new_w)
    return move_around_view_list


def viz2d_total_video(
    s2d: Saved2D,
    viz_vid_fn,
    start_from,
    end_at,
    skip_t,
    cams,
    s_model,
    d_model,
    subsample=1,
    mask_type="all",
    move_around_angle_deg=60.0,
    move_around_id=None,
    remove_redundant_flag=True,
    max_num_frames=500,  # 150,
    print_text=True,
):
    logging.info(f"Viz 2D video from {start_from} to {end_at} ...")

    frame_list = []
    fix_cam_id = start_from

    # prepare the novel camera poses
    move_around_angle = np.deg2rad(move_around_angle_deg)
    if move_around_id is None:
        move_around_id = (start_from + end_at) // 2
    move_around_view_list = __get_move_around_cam_T_cw__(
        s_model,
        d_model,
        cams,
        move_around_id,
        move_around_angle,
        start_from,
        end_at,
    )
    for _ind, view_ind in tqdm(enumerate(range(start_from, min(end_at + 1, cams.T)))):
        if view_ind % skip_t != 0:
            continue
        frame = viz2d_one_frame(
            view_ind,
            s2d,
            cams,
            s_model,
            d_model,
            subsample=1,
            loss_mask_type=mask_type,
            print_text=print_text,
        )
        # fix cam, vary time
        _frame = viz2d_one_frame(
            view_ind,
            s2d,
            cams,
            s_model,
            d_model,
            subsample=1,
            loss_mask_type=mask_type,
            append_depth=False,
            append_dyn=False,
            append_graph=False,
            prefix_text=f"Fix={fix_cam_id} ",
            T_cw=cams.T_cw(fix_cam_id),
            print_text=print_text,
        )
        frame = np.concatenate([frame, _frame], 0)
        # fix time, vary cam
        _frame = viz2d_one_frame(
            move_around_id,
            s2d,
            cams,
            s_model,
            d_model,
            subsample=1,
            loss_mask_type=mask_type,
            append_depth=False,
            append_dyn=False,
            append_graph=False,
            prefix_text=f"T={move_around_id} ",
            T_cw=move_around_view_list[_ind],
            print_text=print_text,
        )
        frame = np.concatenate([frame, _frame], 0)
        # vary cam and time
        _frame = viz2d_one_frame(
            view_ind,
            s2d,
            cams,
            s_model,
            d_model,
            subsample=1,
            loss_mask_type=mask_type,
            append_depth=False,
            append_dyn=False,
            append_graph=False,
            prefix_text=f"Novel",
            T_cw=move_around_view_list[_ind],
            print_text=print_text,
        )
        frame = np.concatenate([frame, _frame], 0)

        frame_list.append(frame)

    # viz the flow
    if d_model is not None:
        flow_frame_list1, choice = viz2d_flow_video(
            start_from,
            end_at,
            cams,
            s_model,
            d_model,
            viz_bg=True,
            text="WithBG",
            skip_t=skip_t,
        )
        flow_frame_list2, choice = viz2d_flow_video(
            start_from,
            end_at,
            cams,
            s_model,
            d_model,
            viz_bg=False,
            text="NoBG",
            choice=choice,
            skip_t=skip_t,
        )
        flow_frame_list3, choice = viz2d_flow_video(
            start_from,
            end_at,
            cams,
            s_model,
            d_model,
            viz_bg=False,
            view_cam_id=fix_cam_id,
            text=f"FixCam={fix_cam_id} ",
            choice=choice,
            skip_t=skip_t,
        )
        flow_frame_list = [
            np.concatenate(
                [flow_frame_list1[i], flow_frame_list2[i], flow_frame_list3[i]], 1
            )
            for i in range(len(flow_frame_list1))
        ]
        final_frame_list = [
            np.concatenate([flow_frame_list[i], frame_list[i]], 0)
            for i in range(len(flow_frame_list))
        ]
    else:
        final_frame_list = frame_list

    # ! always split to two columns
    if (final_frame_list[0].shape[0] // cams.default_H) % 2 != 0:
        dummy_frame = np.zeros((cams.default_H, cams.default_W * 3, 3))
        final_frame_list = [
            np.concatenate([f, dummy_frame], 0) for f in final_frame_list
        ]
        assert (final_frame_list[0].shape[0] // cams.default_H) % 2 == 0
    # * save
    re_aranged_list = []
    for frame in final_frame_list:
        H = frame.shape[0]
        frame_top = frame[: H // 2]
        frame_bottom = frame[H // 2 :]
        if remove_redundant_flag:
            # check whether the right image has all redundant gt and error
            first_col = frame_bottom[:, 0]
            if (first_col == first_col[:1]).all() and first_col[0].tolist() == list(
                BORDER_COLOR
            ):
                frame_bottom_w = frame_bottom.shape[1]
                frame_bottom = frame_bottom[
                    :, frame_bottom_w // 3 : -frame_bottom_w // 3
                ]

        frame = np.concatenate([frame_top, frame_bottom], 1)
        re_aranged_list.append(frame)

    cnt = 0
    cur = 0
    T = len(re_aranged_list)
    while cur < T:
        __video_save__(
            viz_vid_fn[:-4] + f"_{cnt}.mp4",
            [
                f[::subsample, ::subsample, :]
                for f in re_aranged_list[cur : cur + max_num_frames]
            ],
        )
        cnt += 1
        cur += max_num_frames

    return


@torch.no_grad()
def viz2d_one_frame(
    model_tid,
    s2d: Saved2D,
    cams: MonocularCameras,
    s_model,
    d_model=None,
    subsample=1,
    loss_mask_type="all",
    view_cam_id=None,
    prefix_text="",
    save_path=None,
    append_depth=True,
    append_dyn=True,
    append_graph=True,
    rgb_mask=None,
    dep_mask=None,
    T_cw=None,
    print_text=True,
):
    if T_cw is None:
        if view_cam_id is None:
            T_cw = cams.T_cw(model_tid)
        else:
            T_cw = cams.T_cw(view_cam_id)

    # * normal viz
    gs5 = [s_model()]
    if d_model is not None:
        gs5.append(d_model(model_tid))
    render_dict = render(gs5, cams.default_H, cams.default_W, cams.default_K, T_cw)
    if rgb_mask is None:
        rgb_mask = s2d.get_mask_by_key(loss_mask_type)[model_tid].clone()
    _, rgb_loss_i, pred_rgb, gt_rgb = compute_rgb_loss(
        s2d.rgb[model_tid].clone(), render_dict, rgb_mask
    )
    viz_frame = make_viz_np(
        gt_rgb * rgb_mask[:, :, None],
        pred_rgb,
        rgb_loss_i.max(dim=-1).values,
        text0=f"{prefix_text}Fr={model_tid:03d} GT",
        text1=f"{prefix_text}Fr={model_tid:03d} Pred",
        text2=f"{prefix_text}Fr={model_tid:03d} Err",
        print_text=print_text,
    )

    # * ed viz
    if append_graph and d_model is not None:
        viz_frame_graph = make_viz_graph(d_model, model_tid, cams, view_cam_id)
        viz_frame = np.concatenate([viz_frame, viz_frame_graph], 0)

    # * depth viz
    if append_depth:
        if dep_mask is None:
            dep_mask = rgb_mask * s2d.dep_mask[model_tid]
        _, dep_loss_i, pred_dep, prior_dep = compute_dep_loss(
            s2d.dep[model_tid].clone(), render_dict, dep_mask
        )
        viz_frame_dep = make_viz_np(
            prior_dep * dep_mask,
            pred_dep,
            dep_loss_i,
            text0="DEP Target",
            text1="DEP Pred",
            text2="DEP Error",
            print_text=print_text,
        )
        viz_frame = np.concatenate([viz_frame, viz_frame_dep], 0)
        # * when append depth, also append normal
        try:
            _, normal_loss_i, pred_normal, gt_normal = compute_normal_loss(
                s2d.nrm[model_tid].clone(), render_dict, dep_mask
            )
            viz_frame_normal = make_viz_np(
                (1.0 - gt_normal) / 2.0,
                (1.0 - pred_normal) / 2.0,
                normal_loss_i,
                text0="Nrm GT",
                text1="Nrm Pred",
                text2="Nrm Error",
                print_text=print_text,
            )
            viz_frame = np.concatenate([viz_frame, viz_frame_normal], 0)
        except:
            # logging.warning("Failed to viz normal, skip...")
            pass

    # * fg only viz
    if d_model is not None and append_dyn:
        dyn_render_dict = render(
            [d_model(model_tid)],
            cams.default_H,
            cams.default_W,
            cams.default_K,
            T_cw,
            bg_color=[0.5, 0.5, 0.5],
        )
        _, dyn_rgb_loss_i, dyn_pred_rgb, dyn_gt_rgb = compute_rgb_loss(
            s2d.rgb[model_tid].clone(), dyn_render_dict, rgb_mask
        )
        viz_frame_dyn = make_viz_np(
            dyn_gt_rgb,
            dyn_pred_rgb,
            dyn_rgb_loss_i.max(dim=-1).values,
            text0="FG Only",
            text1="FG Pred",
            text2="FG Error",
            print_text=print_text,
        )
        viz_frame = np.concatenate([viz_frame, viz_frame_dyn], 0)

    viz_frame = viz_frame[::subsample, ::subsample, :]
    if save_path is not None:
        imageio.imwrite(save_path, viz_frame)
    return viz_frame


@torch.no_grad()
def make_viz_graph(d_model, view_ind, cams, view_cam_id=None, max_radius=0.001):
    node_mu_w = d_model.scf._node_xyz[d_model.get_tlist_ind(view_ind)]
    if view_cam_id is None:
        render_cam_id = view_ind
    else:
        render_cam_id = view_cam_id
    R_cw, t_cw = cams.Rt_cw(render_cam_id)
    node_mu = node_mu_w @ R_cw.T + t_cw[None]
    order = torch.arange(len(node_mu))
    c_id = torch.from_numpy(cm.hsv(order / len(node_mu))[:, :3]).to(node_mu)

    c_time = torch.from_numpy(cm.hsv(torch.rand(len(node_mu)))[:, :3]).to(node_mu)
    acc_w = d_model.get_node_sinning_w_acc().detach().cpu().numpy()
    acc_w_binary = (acc_w > float(d_model.scf.skinning_k) / 2.0).astype(np.float32)
    acc_w = acc_w / acc_w.max()
    c_w = torch.from_numpy(cm.viridis(acc_w)[:, :3]).to(node_mu)
    c_wb = torch.from_numpy(cm.viridis(acc_w_binary)[:, :3]).to(node_mu)

    H, W = cams.default_H, cams.default_W
    pf = cams.rel_focal.mean() / 2 * min(H, W)

    viz_frames = []
    # for color in [c_id, c_time, c_w]:
    viz_r = min(max_radius, d_model.scf.spatial_unit / 10.0)
    for color, text in zip(
        [c_id, c_time, c_w], ["Nodes-id", "Nodes-rand-color", "Nodes-acc-w"]
    ):
        sph = RGB2SH(color)
        fr = torch.eye(3).to(node_mu)[None].expand(len(node_mu), -1, -1)
        s = torch.ones(len(node_mu), 3).to(node_mu) * viz_r
        o = torch.ones(len(node_mu), 1).to(node_mu) * 1.0

        render_dict = render_cam_pcl(
            node_mu, fr, s, o, sph, H, W, CAM_K=cams.default_K, bg_color=BG_COLOR1
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8).copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = TEXTCOLOR
        lineType = 2
        cv.putText(
            rgb,
            text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
        viz_frames.append(rgb)
    ret = np.concatenate(viz_frames, 1).copy()

    return ret


@torch.no_grad()
def viz2d_flow_video(
    start_from,
    end_at,
    cams,
    s_model,
    d_model,
    subsample=1,
    view_cam_id=None,
    N=128,
    line_n=32,
    viz_bg=False,
    choice=None,
    text="",
    skip_t=1,
):
    H, W = cams.default_H, cams.default_W
    frame_list = []
    if viz_bg:
        s_mu_w, s_fr_w, s_s, s_o, s_sph = s_model(0)
    prev_mu, _, s_traj, o_traj, sph_traj = d_model(
        start_from, 0, nn_fusion=-1
    )  # get init coloring
    s_traj = torch.ones_like(s_traj) * 0.0015
    o_traj = torch.ones_like(o_traj) * 0.999

    # sph_int = torch.rand_like(sph_traj)
    # sph_traj = RGB2SH(sph_int)
    # make new color, first sort the prev mu and then use a HSV in the order
    color_id_coord = d_model._xyz
    color_id = (
        color_id_coord[:, 2] + color_id_coord[:, 1] * 1e6 + color_id_coord[:, 0] * 1e12
    )
    color_id = color_id.argsort().float() / len(color_id)
    # track_color = cm.rainbow(color_id.cpu())[:, :3]
    track_color = cm.hsv(color_id.cpu())[:, :3]
    track_color = torch.from_numpy(track_color).to(prev_mu)
    # _track_color = torch.zeros_like(sph_traj)
    # _track_color[:, :3] = track_color
    # track_color = _track_color
    sph_traj = RGB2SH(track_color)

    # only viz the first frame visible points!!
    render_dict = render(
        [s_model(0), d_model(start_from, 0, nn_fusion=-1)],
        H,
        W,
        cams.default_K,
        cams.T_cw(start_from),
    )
    visibility_mask = render_dict["visibility_filter"][-d_model.N :]
    valid_ind = torch.arange(len(visibility_mask)).to(visibility_mask.device)[
        visibility_mask
    ]
    if choice is None:
        choice = valid_ind[torch.randperm(len(valid_ind))[:N]]
    s_traj = s_traj[choice]
    o_traj = o_traj[choice]
    sph_traj = sph_traj[choice]
    prev_mu = prev_mu[choice]

    mu_w = torch.zeros(0, 3).to(prev_mu)
    fr_w = torch.zeros(0, 3, 3).to(prev_mu)
    s = torch.zeros(0, 3).to(prev_mu)
    o = torch.zeros(0, 1).to(prev_mu)
    sph = torch.zeros(0, sph_traj.shape[-1]).to(prev_mu)
    for view_ind in tqdm(range(start_from, min(end_at + 1, cams.T))):
        if view_ind % skip_t != 0:
            continue
        d_mu_w, d_fr_w, d_s, d_o, d_sph = d_model(view_ind, 0, nn_fusion=-1)
        # draw the line
        src_mu = prev_mu
        dst_mu = d_mu_w[choice]
        line_dir = dst_mu - src_mu  # N,3
        intermediate_mu = (
            src_mu[:, None]
            + torch.linspace(0, 1, line_n)[None, :, None].to(line_dir)
            * line_dir[:, None]
        ).reshape(-1, 3)
        intermediate_fr = (
            torch.eye(3)[None].expand(len(intermediate_mu), -1, -1).to(intermediate_mu)
        )
        intermediate_s = torch.ones_like(intermediate_mu) * 0.0015 * 0.3
        intermediate_o = torch.ones_like(intermediate_mu[:, :1]) * 0.999
        intermediate_sph = (
            sph_traj[:, None].expand(-1, line_n, -1).reshape(-1, sph_traj.shape[-1])
        )
        prev_mu = dst_mu

        mu_w = torch.cat([mu_w, intermediate_mu.clone(), d_mu_w[choice].clone()], 0)
        fr_w = torch.cat([fr_w, intermediate_fr.clone(), d_fr_w[choice].clone()], 0)
        s = torch.cat([s, intermediate_s.clone(), s_traj.clone()], 0)
        o = torch.cat([o, intermediate_o.clone(), o_traj.clone()], 0)
        sph = torch.cat([sph, intermediate_sph.clone(), sph_traj.clone()], 0)
        # transform
        if view_cam_id is None:
            _render_cam_id = view_ind
        else:
            _render_cam_id = view_cam_id
        if viz_bg:
            working_mu_w = torch.cat([s_mu_w, mu_w, d_mu_w], 0)
            working_fr_w = torch.cat([s_fr_w, fr_w, d_fr_w], 0)
            working_s = torch.cat([s_s, s, d_s], 0)
            working_o = torch.cat([s_o, o, d_o], 0)
            working_sph = torch.cat([s_sph, sph, d_sph], 0)
        else:
            working_mu_w = mu_w
            working_fr_w = fr_w
            working_s = s
            working_o = o
            working_sph = sph
        R_cw, t_cw = cams.Rt_cw(_render_cam_id)
        working_mu_cur = (
            torch.einsum("ij, nj->ni", R_cw, working_mu_w.clone()) + t_cw[None]
        )
        working_fr_cur = torch.einsum("ij, njk->nik", R_cw, working_fr_w.clone())
        # render
        assert (
            len(working_mu_cur)
            == len(working_fr_cur)
            == len(working_s)
            == len(working_o)
            == len(working_sph)
        )
        render_dict = render_cam_pcl(
            working_mu_cur.contiguous(),
            working_fr_cur.contiguous(),
            working_s.contiguous(),
            working_o.contiguous(),
            working_sph.contiguous(),
            H,
            W,
            CAM_K=cams.default_K,
            bg_color=BG_COLOR1,
        )
        pred_rgb = render_dict["rgb"].permute(1, 2, 0)
        viz_frame = pred_rgb.detach().cpu().numpy()
        viz_frame = (np.clip(viz_frame, 0.0, 1.0) * 255).astype(np.uint8).copy()
        if len(text) > 0:
            font = cv.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = TEXTCOLOR
            lineType = 2
            cv.putText(
                viz_frame,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
        viz_frame = viz_frame[::subsample, ::subsample, :]
        frame_list.append(viz_frame)
    return frame_list, choice


@torch.no_grad()
def viz3d_total_video(
    cams,
    d_model,
    start_tid,
    end_tid,
    save_path,
    res=240,
    s_model=None,
    time_interval=5,
    bg_color=BG_COLOR1,
    max_num_frames=500,  # 150,
):
    logging.info(f"Viz 3D scene from {start_tid} to {end_tid} ...")

    frames_node = viz_curve(
        d_model.scf._node_xyz,
        torch.ones_like(d_model.scf._node_xyz[..., 0]).bool(),
        cams,
        res=res,
        text="All ED Nodes",
        viz_n=-1,
        # viz_n=128,
        time_window=30,
    )

    # first row is all
    frames1 = _combine_frames(
        viz3d_scene_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            s_model=s_model,
            bg_color=bg_color,
        )
    )
    frames2 = _combine_frames(
        viz3d_scene_flow_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            s_model=s_model,
            bg_color=BG_COLOR1,
        )
    )
    frames3 = _combine_frames(
        viz3d_scene_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            bg_color=bg_color,
        )
    )
    frames4 = _combine_frames(
        viz3d_scene_flow_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            bg_color=BG_COLOR1,
        )
    )

    frames_up = [
        np.concatenate([frames1[i], frames2[i]], 1) for i in range(len(frames1))
    ]
    frames_down = [
        np.concatenate([frames3[i], frames4[i]], 1) for i in range(len(frames3))
    ]
    frames = [
        np.concatenate([frames_up[i], frames_down[i]], 0) for i in range(len(frames1))
    ]
    frames = [
        np.concatenate(
            [
                frames[i],
                np.concatenate([frames_node[i], np.ones_like(frames_node[i])], 0) * 255,
            ],
            1,
        )
        for i in range(len(frames1))
    ]

    # __video_save__(save_path, frames)
    cnt = 0
    cur = 0
    T = len(frames)
    while cur < T:
        __video_save__(
            save_path[:-4] + f"_{cnt}.mp4", frames[cur : cur + max_num_frames]
        )
        cnt += 1
        cur += max_num_frames
    return


def _combine_frames(frames: dict):
    T = len(frames[list(frames.keys())[0]])
    for v in frames.values():
        assert len(v) == T
    ret = []
    for i in range(T):
        ret.append(np.concatenate([frames[k][i] for k in frames.keys()], 1))
    return ret


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


@torch.no_grad()
def viz3d_scene_video(
    cams,
    d_model,
    start_tid,
    end_tid,
    res=480,
    prefix="",
    save_dir=None,
    s_model=None,
    bg_color=BG_COLOR1,
):
    viz_cam_R = q2R(cams.q_wc[start_tid : end_tid + 1])
    viz_cam_t = cams.t_wc[start_tid : end_tid + 1]
    viz_cam_R, viz_cam_t = cams.Rt_wc_list()
    viz_cam_R = viz_cam_R[start_tid : end_tid + 1].clone()
    viz_cam_t = viz_cam_t[start_tid : end_tid + 1].clone()

    viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
    frames = {}
    for viz_time in tqdm(range(start_tid, end_tid + 1)):
        gs5_param = d_model(viz_time)
        if s_model is not None:
            gs5_param = cat_gs(*gs5_param, *s_model())
        viz_dict = viz_scene(
            res,
            res,
            viz_cam_R,
            viz_cam_t,
            viz_f=viz_f,
            gs5_param=gs5_param,
            draw_camera_frames=False,
            bg_color=bg_color,
        )
        for k, v in viz_dict.items():
            if k not in frames.keys():
                frames[k] = []
            v = np.clip(v, 0.0, 1.0)
            v = (v * 255).astype(np.uint8)
            frames[k].append(v)
    if save_dir is not None:
        for k, v in frames.items():
            __video_save__(
                osp.join(save_dir, f"{prefix}dyn_{k}_{start_tid}-{end_tid}.mp4"),
                v,
            )
    return frames


@torch.no_grad()
def viz3d_scene_flow_video(
    cams,
    d_model,
    start_tid,
    end_tid,
    res=480,
    prefix="",
    save_dir=None,
    s_model=None,
    N=128,
    line_n=16,
    time_window=10,
    bg_color=BG_COLOR1,
):
    viz_R = quaternion_to_matrix(cams.q_wc[start_tid : end_tid + 1])
    viz_t = cams.t_wc[start_tid : end_tid + 1]
    viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
    frames = {}

    # prepare flow gaussians
    prev_mu, _, s_traj, o_traj, sph_traj = d_model(
        start_tid, 0, nn_fusion=-1
    )  # get init coloring
    s_traj = torch.ones_like(s_traj) * 0.0015
    o_traj = torch.ones_like(o_traj) * 0.999
    sph_int = torch.rand_like(sph_traj)
    sph_traj = RGB2SH(sph_int)

    choice = torch.randperm(len(prev_mu))[:N]
    s_traj = s_traj[choice]
    o_traj = o_traj[choice]
    sph_traj = sph_traj[choice]
    prev_mu = prev_mu[choice]

    # dummy
    mu_w, fr_w, s, o, sph = d_model(start_tid, 0, nn_fusion=-1)
    s = s * 0.0
    o = o * 0.0

    for viz_time in range(start_tid, end_tid + 1):

        _mu_w, _fr_w, _, _, _ = d_model(viz_time, 0, nn_fusion=-1)
        # draw the line
        src_mu = prev_mu
        dst_mu = _mu_w[choice]
        line_dir = dst_mu - src_mu  # N,3
        intermediate_mu = (
            src_mu[:, None]
            + torch.linspace(0, 1, line_n)[None, :, None].to(line_dir)
            * line_dir[:, None]
        ).reshape(-1, 3)
        intermediate_fr = (
            torch.eye(3)[None].expand(len(intermediate_mu), -1, -1).to(intermediate_mu)
        )
        intermediate_s = torch.ones_like(intermediate_mu) * 0.0015 * 0.3
        intermediate_o = torch.ones_like(intermediate_mu[:, :1]) * 0.999
        intermediate_sph = sph_traj[:, None].expand(-1, line_n, -1).reshape(-1, 3)
        prev_mu = dst_mu
        # pad
        if sph.shape[1] > 3:
            intermediate_sph = torch.cat(
                [
                    intermediate_sph,
                    torch.zeros(len(intermediate_sph), sph.shape[1] - 3).to(sph),
                ],
                1,
            )

        one_time_N = len(src_mu) * (line_n + 1)
        max_N = time_window * N

        mu_w = torch.cat([mu_w, intermediate_mu.clone(), _mu_w[choice].clone()], 0)[
            -max_N:
        ]
        fr_w = torch.cat([fr_w, intermediate_fr.clone(), _fr_w[choice].clone()], 0)[
            -max_N:
        ]
        s = torch.cat([s, intermediate_s.clone(), s_traj.clone()], 0)[-max_N:]
        o = torch.cat([o, intermediate_o.clone(), o_traj.clone()], 0)[-max_N:]
        sph = torch.cat([sph, intermediate_sph.clone(), sph_traj.clone()], 0)[-max_N:]
        gs5_param = (mu_w, fr_w, s, o, sph)

        if s_model is not None:
            gs5_param = cat_gs(*gs5_param, *s_model(0))
        viz_dict = viz_scene(
            res, res, viz_R, viz_t, viz_f=viz_f, gs5_param=gs5_param, bg_color=bg_color
        )
        for k, v in viz_dict.items():
            if k not in frames.keys():
                frames[k] = []
            v = np.clip(v, 0.0, 1.0)
            v = (v * 255).astype(np.uint8)
            frames[k].append(v)
    if save_dir is not None:
        for k, v in frames.items():
            __video_save__(
                osp.join(save_dir, f"{prefix}dyn_{k}_{start_tid}-{end_tid}.mp4"),
                v,
            )
    return frames


def cat_gs(m1, f1, s1, o1, c1, m2, f2, s2, o2, c2):
    m = torch.cat([m1, m2], dim=0).contiguous()
    f = torch.cat([f1, f2], dim=0).contiguous()
    s = torch.cat([s1, s2], dim=0).contiguous()
    o = torch.cat([o1, o2], dim=0).contiguous()
    c = torch.cat([c1, c2], dim=0).contiguous()
    return m, f, s, o, c


def viz_o_hist(model, save_path, title_text=""):
    o = model.get_o.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    plt.hist(o, bins=100)
    plt.title(f"{title_text} o hist")
    plt.savefig(save_path)
    plt.close()
    return


def viz_s_hist(model, save_path, title_text=""):
    s = model.get_s.detach()
    s = s.sort(dim=-1).values
    s = s.cpu().numpy()
    fig = plt.figure(figsize=(20, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(s[..., i], bins=100)
        plt.title(f"{title_text} s hist")
    plt.savefig(save_path)
    plt.close()
    return


def viz_sigma_hist(scf, save_path, title_text=""):
    sig = scf.node_sigma.abs().detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    # sig = sig.reshape(-1)
    if sig.shape[1] == 1:
        plt.hist(sig, bins=100)
        plt.title(f"{title_text} Node Sigma hist (Total {scf.M} nodes)")
    else:
        C = sig.shape[1]
        for i in range(C):
            plt.subplot(1, C, i + 1)
            plt.hist(sig[:, i], bins=100)
            plt.title(f"{title_text} Node Sigma [{scf.M}] dim={i}")
    plt.savefig(save_path)
    plt.close()
    return


def viz_dyn_o_hist(model, save_path, title_text=""):
    dyn_o = model.get_d.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    plt.hist(dyn_o, bins=100)
    plt.title(f"{title_text} dyn_o hist")
    plt.savefig(save_path)
    plt.close()
    return


def viz_hist(d_model, viz_dir, postfix):
    viz_s_hist(d_model, osp.join(viz_dir, f"s_hist_{postfix}.jpg"))
    viz_o_hist(d_model, osp.join(viz_dir, f"o_hist_{postfix}.jpg"))


def viz_dyn_hist(scf, viz_dir, postfix):
    viz_sigma_hist(scf, osp.join(viz_dir, f"sigma_hist_{postfix}.jpg"))
    # viz_dyn_o_hist(d_model, osp.join(viz_dir, f"dyn_o_hist_{postfix}.jpg"))
    # viz the skinning K count
    valid_sk_count = scf.topo_knn_mask.sum(-1).detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    plt.hist(valid_sk_count), plt.title(f"Valid node neighbors count {scf.M}")
    plt.savefig(osp.join(viz_dir, f"valid_sk_count_{postfix}.jpg"))
    plt.close()
    return


def viz_N_count(N_count_list, path):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(N_count_list), plt.title("Noodle Count")
    plt.savefig(path)
    plt.close()


def viz_depth_list(depth_list_pt, save_path):
    assert isinstance(depth_list_pt, torch.Tensor)

    # depth_min = depth_list_pt.min()
    # depth_max = depth_list_pt.max()
    # depth_list_pt = (depth_list_pt - depth_min) / (depth_max - depth_min)
    viz_list = []
    for dep in tqdm(depth_list_pt):
        depth_min = dep.min()
        depth_max = dep.max()
        dep = (dep - depth_min) / (depth_max - depth_min)
        viz = cm.viridis(dep.detach().cpu().numpy())[:, :, :3]
        viz_list.append(viz)
    __video_save__(save_path, viz_list)
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


def viz_plt_missing_slot(track_mask, path, max_viz=2048):
    # T,N
    T, N = track_mask.shape
    choice = torch.randperm(N)[:max_viz]

    viz_mask = track_mask[:, choice].clone().float()
    resort = torch.argsort(viz_mask.sum(0), descending=True)
    viz_mask = viz_mask[:, resort]
    plt.figure(figsize=(2.0 * max_viz / T, 3.0))
    plt.imshow((viz_mask * 255.0).cpu().numpy(), cmap="viridis")
    plt.title("MissingSlot=0"), plt.xlabel("Sorted Noodles"), plt.ylabel("T")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    # also viz how many valid slot in each frame
    valid_count = viz_mask.sum(1)
    plt.figure(figsize=(5, 3))
    # use bar plot
    plt.bar(range(T), valid_count.cpu().numpy())
    plt.title("ValidSlotCount"), plt.xlabel("T"), plt.ylabel("ValidCount")
    plt.tight_layout()
    plt.savefig(path.replace(".jpg", "_count_perframe.jpg"))
    plt.close()

    return


@torch.no_grad()
def viz_mv_model_frame(
    prior2d, cams, s_model, scf, mv_model, t, support_t, T_cw=None, support_m=None
):
    if s_model is None:
        gs5 = []
        order = 0
    else:
        gs5 = [s_model()]
        order = s_model.max_sph_order
    if support_m is not None:
        d_gs5, param_mask = mv_model.forward(
            scf,
            t,
            support_mask=support_m,
            pad_sph_order=order,
        )
    else:
        d_gs5, param_mask = mv_model.forward(
            scf,
            t,
            torch.Tensor([support_t]).long(),
            pad_sph_order=order,
        )
    gs5.append(d_gs5)
    if T_cw is None:
        T_cw = cams.T_cw(t)
    render_dict = render(
        gs5,
        prior2d.H,
        prior2d.W,
        cams.defualt_K,
        T_cw,
    )
    # do the rendering loss
    rgb_sup_mask = prior2d.get_mask_by_key("all", t)
    loss_rgb, rgb_loss_i, pred_rgb, gt_rgb = compute_rgb_loss(
        prior2d, t, render_dict, rgb_sup_mask
    )
    dep_sup_mask = prior2d.get_mask_by_key("all_dep", t)
    loss_dep, dep_loss_i, pred_dep, prior_dep = compute_dep_loss(
        prior2d, t, render_dict, dep_sup_mask
    )
    viz_figA = make_viz_np(
        gt_rgb,
        pred_rgb,
        rgb_loss_i.max(-1).values,
        text0=f"q={t},s={support_t}",
    )
    viz_figB = make_viz_np(
        prior_dep,
        pred_dep,
        dep_loss_i,
        text0=f"q={t},s={support_t}",
    )
    ret = np.concatenate([viz_figA, viz_figB], 0)
    return ret


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


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning


def make_video(src_pattern, dst):
    image_fns = glob(src_pattern)
    image_fns.sort()
    if len(image_fns) == 0:
        print(f"no image found in {src_pattern}")
        return
    frames = []
    for i, fn in enumerate(image_fns):
        img = cv.imread(fn)[..., ::-1]
        frames.append(img)
    imageio.mimwrite(dst, frames)


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


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


def look_at_R(look_at, cam_center, right_dir=None):
    if right_dir is None:
        right_dir = torch.tensor([1.0, 0.0, 0.0]).to(look_at)
    z_dir = F.normalize(look_at - cam_center, dim=0)
    y_dir = F.normalize(torch.cross(z_dir, right_dir), dim=0)
    x_dir = F.normalize(torch.cross(y_dir, z_dir), dim=0)
    R = torch.stack([x_dir, y_dir, z_dir], 1)
    return R


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
    # global_R_wc, global_t_wc = get_global_viz_cam_Rt(
    #     mu_w, param_cam_R_wc, param_cam_t_wc, viz_f
    # )
    global_down20_R_wc, global_down20_t_wc = get_global_viz_cam_Rt(
        mu_w, param_cam_R_wc, param_cam_t_wc, viz_f, 20, shift_margin_ratio=1.1
    )
    if draw_camera_frames:
        # camera_R_wc, camera_t_wc = get_global_viz_cam_Rt(
        #     mu_w,
        #     param_cam_R_wc,
        #     param_cam_t_wc,
        #     viz_f,
        #     factor=0.5,
        #     auto_zoom_mask=cam_viz_mask,
        # )
        camera_down20_R_wc, camera_down20_t_wc = get_global_viz_cam_Rt(
            mu_w,
            param_cam_R_wc,
            param_cam_t_wc,
            viz_f,
            20,
            factor=0.5,
            auto_zoom_mask=cam_viz_mask,
            shift_margin_ratio=1.1,
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
