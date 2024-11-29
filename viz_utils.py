from tqdm import tqdm
import torch, numpy as np
from transforms3d.euler import mat2euler, euler2mat
from lib_render.gauspl_renderer_native import render_cam_pcl
from lib_render.sh_utils import RGB2SH, SH2RGB
from lib_render.render_helper import GS_BACKEND
from matplotlib import pyplot as plt
import torch, numpy as np
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
from pytorch3d.ops import knn_points
import logging
import imageio
import os, sys, os.path as osp
from tqdm import tqdm

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
import open3d as o3d
from lib_render.render_helper import render

from lib_render.sh_utils import RGB2SH, SH2RGB
import cv2 as cv
import torch.nn.functional as F

TEXTCOLOR = (255, 0, 0)
BORDER_COLOR = (100, 255, 100)

from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA


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


def save_frame_list(frame_list, name):
    os.makedirs(name, exist_ok=True)
    imageio.mimsave(name + ".mp4", frame_list)
    for i, frame in enumerate(frame_list):
        imageio.imwrite(osp.join(name, f"{i:04d}.jpg"), frame)
    return


@torch.no_grad()
def get_global_3D_cam_T_cw(
    s_model,
    d_model,
    cams,
    H,
    W,
    ref_tid,
    back_ratio=1.0,
    up_ratio=0.2,
):
    render_dict = render(
        [s_model(), d_model(ref_tid)],
        H,
        W,
        K=cams.K(H, W),
        T_cw=cams.T_cw(ref_tid),
    )
    depth = render_dict["dep"][0]
    center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
    if center_dep < 1e-2:
        center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
    focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)  # in cam frame

    T_c_new = torch.eye(4).to(cams.T_wc(0))
    T_c_new[2, -1] = -center_dep * back_ratio  # z
    T_c_new[1, -1] = -center_dep * up_ratio  # y
    _z_dir = F.normalize(focus_point[:3] - T_c_new[:3, -1], dim=0)
    _x_dir = F.normalize(
        torch.cross(torch.Tensor([0.0, 1.0, 0.0]).to(_z_dir), _z_dir), dim=0
    )
    _y_dir = F.normalize(torch.cross(_z_dir, _x_dir), dim=0)
    T_c_new[:3, 0] = _x_dir
    T_c_new[:3, 1] = _y_dir
    T_c_new[:3, 2] = _z_dir
    T_base = cams.T_wc(ref_tid)
    T_w_new = T_base @ T_c_new
    T_new_w = T_w_new.inverse()
    return T_new_w


@torch.no_grad()
def get_move_around_cam_T_cw(
    s_model,
    d_model,
    cams,
    H,
    W,
    move_around_angle_deg,
    total_steps,
    center_id=None,
):

    # in the xy plane, the new camera is forming a circle
    move_around_view_list = []
    for i in tqdm(range(total_steps)):
        if center_id is None:
            move_around_id = i
            assert total_steps - 1 < cams.T
            render_dict = render(
                [s_model(), d_model(move_around_id)],
                H,
                W,
                K=cams.K(H, W),
                T_cw=cams.T_cw(move_around_id),
            )
            # depth = (render_dict["dep"] / (render_dict["alpha"] + 1e-6))[0]
            depth = render_dict["dep"][0]
            center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
            if center_dep < 1e-2:
                center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
            focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)
            move_around_radius = np.tan(move_around_angle_deg) * focus_point[2].item()
        else:
            move_around_id = center_id
            if i == 0:
                render_dict = render(
                    [s_model(), d_model(move_around_id)],
                    H,
                    W,
                    K=cams.K(H, W),
                    T_cw=cams.T_cw(move_around_id),
                )
                # depth = (render_dict["dep"] / (render_dict["alpha"] + 1e-6))[0]
                depth = render_dict["dep"][0]
                center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
                if center_dep < 1e-2:
                    center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
                focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)
                move_around_radius = (
                    np.tan(move_around_angle_deg) * focus_point[2].item()
                )

        x = (
            move_around_radius * np.cos(2 * np.pi * i / (total_steps - 1))
            - move_around_radius
        )
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

        T_base = cams.T_wc(move_around_id)

        T_w_new = T_base @ T_c_new
        T_new_w = T_w_new.inverse()
        move_around_view_list.append(T_new_w)
    return move_around_view_list


@torch.no_grad()
def draw_gs_point_line(start, end, n=32):
    # start, end is N,3 tensor
    line_dir = end - start
    xyz = (
        start[:, None]
        + torch.linspace(0, 1, n)[None, :, None].to(start) * line_dir[:, None]
    )
    return xyz


def map_colors(points, mod=1):
    # normalized_points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0))

    # do pca for the points
    pca = PCA(n_components=3)
    pca_points = pca.fit_transform(points)
    # normalzie
    pca_points = (pca_points - np.min(pca_points, axis=0)) / (
        np.max(pca_points, axis=0) - np.min(pca_points, axis=0)
    )

    # Map coordinates to HSV colors
    # # H: X-coordinate, S: 1 (high saturation), V: Z-coordinate
    hsv_colors = np.zeros_like(pca_points)
    hue = pca_points[:, 0]
    if mod > 1:
        # set periodical mod times
        hue = hue * mod
        hue = hue - np.floor(hue)
    hsv_colors[:, 0] = hue
    hsv_colors[:, 1] = 0.9
    hsv_colors[:, 2] = 0.9
    rgb_colors = hsv_to_rgb(hsv_colors)
    return rgb_colors


@torch.no_grad()
def viz_single_2d_flow_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    save_fn,
    pose_list,
    N_max=512,
    color_mod=5,
    max_T=1000,
    gray_scale_bg_flag=True,
    #
    node_r_factor=0.001,  # 0.05,
    # line
    line_N=32,
    line_opa=0.5,
    line_r_factor=0.001,
    rel_focal=None,
    bg_color=[0.0, 0.0, 0.0],
):
    rgb_viz_list = []

    # ! color the node
    pts_first = d_model(0)[0]
    if len(pts_first) > N_max:
        # ! do a filtering for viz purpose, only viz dense area
        # use open3d
        inlier_mask = outlier_removal_o3d(pts_first, std_ratio=1.0)
        print(f"Filtered {len(pts_first) - inlier_mask.sum()} points")
        candidates = torch.arange(len(pts_first))[inlier_mask.cpu()]
        step = max(1, len(candidates) // N_max)
        # viz_choice = candidates[torch.randperm(len(candidates))[:N_max]]
        viz_choice = candidates[::step][:N_max]
        pts_first = pts_first[viz_choice]
    node_colors = map_colors(pts_first.detach().cpu().numpy(), mod=color_mod)

    flow_sph = RGB2SH(torch.from_numpy(node_colors).to(pts_first.device).float())
    pad_sph_dim = s_model()[-1].shape[1]
    if pad_sph_dim > flow_sph.shape[1]:
        flow_sph = F.pad(flow_sph, (0, pad_sph_dim - flow_sph.shape[1], 0, 0))

    flow_mu = pts_first
    flow_fr = (
        torch.eye(3).to(flow_mu.device).unsqueeze(0).expand(flow_mu.shape[0], -1, -1)
    )
    flow_s = (
        torch.ones(len(flow_mu), 3).to(flow_mu)
        * d_model.scf.spatial_unit
        * node_r_factor
    )
    flow_o = torch.ones_like(flow_s[:, :1]) * 0.99
    last_flow_mu = flow_mu
    last_flow_sph = flow_sph

    # ! gray-scale the bg
    gs5_bg = list(s_model())
    if gray_scale_bg_flag:
        bg_rgb = SH2RGB(gs5_bg[-1][:, :3])
        bg_gray = torch.mean(bg_rgb, dim=1, keepdim=True).expand(-1, 3)
        # convert to gray scale
        bg_sph = RGB2SH(bg_gray)
        if pad_sph_dim > bg_sph.shape[1]:
            bg_sph = F.pad(bg_sph, (0, pad_sph_dim - bg_sph.shape[1], 0, 0))
        gs5_bg[-1] = bg_sph

    max_buffer_size = len(flow_mu) * (line_N + 1) * max_T

    for cam_tid in tqdm(range(len(pose_list))):
        # working_t = cam_tid if model_t is None else model_t
        working_t = cam_tid

        ##################################################
        # make GS
        gs5 = [gs5_bg]
        d_gs5 = list(d_model(working_t))
        d_gs5[-2] = 0.2 * d_gs5[-2]
        gs5.append(d_gs5)

        if cam_tid > 0:
            new_xyz = d_gs5[0][viz_choice]
            new_flow_sph = last_flow_sph
            # first draw lines of the flow
            if line_N > 0:
                line_xyz = draw_gs_point_line(new_xyz, last_flow_mu, n=line_N).reshape(
                    -1, 3
                )
                line_fr = (
                    torch.eye(3)
                    .to(flow_mu.device)
                    .unsqueeze(0)
                    .expand(line_xyz.shape[0], -1, -1)
                )
                line_s = (
                    torch.ones_like(line_xyz) * d_model.scf.spatial_unit * line_r_factor
                )
                line_o = torch.ones_like(line_s[:, :1]) * line_opa
                line_sph = draw_gs_point_line(
                    new_flow_sph,
                    last_flow_sph,
                    n=line_N,
                ).reshape(-1, flow_sph.shape[-1])
                flow_mu = torch.cat([flow_mu, line_xyz], dim=0)
                flow_fr = torch.cat([flow_fr, line_fr], dim=0)
                flow_s = torch.cat([flow_s, line_s], dim=0)
                flow_o = torch.cat([flow_o, line_o], dim=0)
                flow_sph = torch.cat([flow_sph, line_sph], dim=0)
                last_flow_mu = new_xyz
                last_flow_sph = new_flow_sph
            flow_mu = torch.cat([flow_mu, new_xyz], dim=0)
            new_fr = (
                torch.eye(3)
                .to(new_xyz.device)
                .unsqueeze(0)
                .expand(new_xyz.shape[0], -1, -1)
            )
            flow_fr = torch.cat([flow_fr, new_fr], dim=0)
            flow_s = torch.cat(
                [
                    flow_s,
                    torch.ones_like(new_xyz) * d_model.scf.spatial_unit * node_r_factor,
                ],
                dim=0,
            )
            flow_o = torch.cat([flow_o, torch.ones_like(flow_s[:, :1]) * 0.99], dim=0)
            flow_sph = torch.cat([flow_sph, new_flow_sph], dim=0)
        if len(flow_mu) > max_buffer_size:
            flow_mu = flow_mu[-max_buffer_size:]
            flow_fr = flow_fr[-max_buffer_size:]
            flow_s = flow_s[-max_buffer_size:]
            flow_o = flow_o[-max_buffer_size:]
            flow_sph = flow_sph[-max_buffer_size:]

        gs5.append([flow_mu, flow_fr, flow_s, flow_o, flow_sph])
        ##################################################
        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            bg_color=bg_color,
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)
    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    return


@torch.no_grad()
def viz_single_2d_node_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    save_fn,
    pose_list,
    model_t=None,
    gray_scale_bg_flag=True,
    #
    node_r1=0.003,
    node_opa1=1.0,
    node_r2=0.020,  # 0.01,
    node_opa2=0.012,
    fg_opa_factor=0.1,
    # line
    line_N=32,
    line_color=[0.7] * 3,
    line_opa=0.05,
    line_r_factor=0.0025,  # 0.08
    line_colorful_flag=True,
    rel_focal=None,
    bg_color=[0.0, 0.0, 0.0],
):
    rgb_viz_list = []

    # ! color the node
    node_first = d_model.scf._node_xyz[0]
    node_colors = map_colors(node_first.detach().cpu().numpy())
    node_sph = RGB2SH(torch.from_numpy(node_colors).to(node_first.device).float())
    pad_sph_dim = s_model()[-1].shape[1]
    if pad_sph_dim > node_sph.shape[1]:
        node_sph = F.pad(node_sph, (0, pad_sph_dim - node_sph.shape[1], 0, 0))

    # node_s1 = d_model.scf.node_sigma.expand(-1, 3) * node_r1_factor  # 0.333  # * 0.05
    node_s1 = (
        torch.ones_like(d_model.scf.node_sigma.expand(-1, 3)) * node_r1
    )  # 0.333  # * 0.05
    node_s1 = torch.clamp(node_s1, 1e-8, d_model.scf.spatial_unit * 3)
    node_o1 = torch.ones_like(node_s1[:, :1]) * node_opa1

    node_s2 = torch.ones_like(node_s1) * node_r2
    # node_o2 = torch.ones_like(node_s2[:, :1]) * 0.003
    node_o2 = torch.ones_like(node_s2[:, :1]) * node_opa2

    line_sph = torch.tensor(line_color).to(node_sph.device).float()[None]
    line_sph = RGB2SH(line_sph)
    if pad_sph_dim > line_sph.shape[1]:
        line_sph = F.pad(line_sph, (0, pad_sph_dim - line_sph.shape[1], 0, 0))

    # ! gray-scale the bg
    gs5_bg = list(s_model())
    if gray_scale_bg_flag:
        bg_rgb = SH2RGB(gs5_bg[-1][:, :3])
        bg_gray = torch.mean(bg_rgb, dim=1, keepdim=True).expand(-1, 3)
        # convert to gray scale
        bg_sph = RGB2SH(bg_gray)
        if pad_sph_dim > bg_sph.shape[1]:
            bg_sph = F.pad(bg_sph, (0, pad_sph_dim - bg_sph.shape[1], 0, 0))
        gs5_bg[-1] = bg_sph

    for cam_tid in tqdm(range(len(pose_list))):
        working_t = cam_tid if model_t is None else model_t

        ##################################################
        # make GS
        gs5 = [gs5_bg]
        d_gs5 = list(d_model(working_t))
        d_gs5[-2] = fg_opa_factor * d_gs5[-2]
        gs5.append(d_gs5)
        node_mu = d_model.scf._node_xyz[working_t]
        node_fr = (
            torch.eye(3)
            .to(node_mu.device)
            .unsqueeze(0)
            .expand(node_mu.shape[0], -1, -1)
        )
        gs5.append([node_mu, node_fr, node_s1, node_o1, node_sph * 0.3])
        gs5.append([node_mu, node_fr, node_s2, node_o2, node_sph])
        ##################################################
        if line_N > 0:
            scf = d_model.scf
            dst_xyz = node_mu[scf.topo_knn_ind]
            src_xyz = node_mu[:, None].expand(-1, scf.topo_knn_ind.shape[1], -1)
            line_xyz = draw_gs_point_line(
                src_xyz[scf.topo_knn_mask], dst_xyz[scf.topo_knn_mask], n=line_N
            ).reshape(-1, 3)
            line_fr = (
                torch.eye(3)
                .to(node_mu.device)
                .unsqueeze(0)
                .expand(line_xyz.shape[0], -1, -1)
            )
            line_s = torch.ones_like(line_xyz) * scf.spatial_unit * line_r_factor
            line_o = torch.ones_like(line_s[:, :1]) * line_opa
            if line_colorful_flag:
                src_sph = node_sph[:, None].expand(-1, scf.topo_knn_ind.shape[1], -1)
                dst_sph = node_sph[scf.topo_knn_ind]
                l_sph = draw_gs_point_line(
                    src_sph[scf.topo_knn_mask], dst_sph[scf.topo_knn_mask], n=line_N
                ).reshape(-1, node_sph.shape[-1])
            else:
                l_sph = line_sph.expand(len(line_xyz), -1)
            gs5.append([line_xyz, line_fr, line_s, line_o, l_sph])

        ##################################################
        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            bg_color=bg_color,
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)

        # imageio.imsave("./debug/dbg.jpg", rgb_viz)

        rgb_viz_list.append(rgb_viz)
    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    return


@torch.no_grad()
def viz_single_2d_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    save_fn,
    pose_list,
    model_t=None,
    rel_focal=None,
    bg_flag=True,
    fg_flag=True,
    bg_color=[0.0, 0.0, 0.0],
    d_mask=None,
):
    rgb_viz_list, dep_viz_list, normal_viz_list = [], [], []
    if rel_focal is None:
        rel_focal = cams.rel_focal
    for cam_tid in tqdm(range(len(pose_list))):
        gs5 = []
        assert bg_flag or fg_flag
        if bg_flag:
            gs5.append(s_model())
        # if fg_flag:
        #     gs5.append(d_model(cam_tid if model_t is None else model_t))
        if fg_flag:
            if d_mask is None:
                gs5.append(d_model(cam_tid if model_t is None else model_t))
            else:
                _d_gs5 = d_model(cam_tid if model_t is None else model_t)
                gs5.append([it[d_mask] for it in _d_gs5])
        render_dict = render(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            bg_color=bg_color,
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)
        dep = render_dict["dep"].detach().cpu().numpy().squeeze(0)
        dep_viz_list.append(dep)
        if "normal" in render_dict:
            normal = render_dict["normal"].detach().cpu().numpy()
            normal_viz = (1 - normal) / 2
            normal_viz_list.append(normal_viz.transpose(1, 2, 0))

    # # use disp map to viz the depth!
    # viz_dep = np.stack(dep_viz_list, axis=0)
    # valid_mask = viz_dep > 0
    # max_dep, min_dep = viz_dep[valid_mask].max(), viz_dep[valid_mask].min()
    # viz_dep[valid_mask] = (viz_dep[valid_mask] - min_dep) / (max_dep - min_dep)
    # # viz_dep = [plt.cm.plasma(it)[:,:,:3] * 255 for it in viz_dep]
    # viz_dep = [plt.cm.viridis(it)[:, :, :3] * 255 for it in viz_dep]
    # save_frame_list(viz_dep, save_fn + "_dep")

    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    if len(normal_viz_list) > 0:
        print(normal_viz_list[0].shape)
        save_frame_list(normal_viz_list, save_fn + "_normal")
    return


@torch.no_grad()
def viz_single_2d_camera_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    save_fn,
    pose_list,
    model_t=None,
    rel_focal=None,
    bg_flag=True,
    fg_flag=True,
    bg_color=[0.0, 0.0, 0.0],
    invisble_opa_factor=1.0,  # 0.05,
    cam_draw_scale=0.2,
    inivisble_red_ratio=0.8,
    # K=32,
):
    device = cams.T_wc(0).device
    rgb_viz_list, dep_viz_list, normal_viz_list = [], [], []
    if rel_focal is None:
        rel_focal = cams.rel_focal

    cam_H, cam_W = cams.default_H, cams.default_W
    L = float(max(cam_H, cam_W))
    cam_F = float(cams.K()[0, 0] / L)
    cam_H, cam_W = float(cam_H / L), float(cam_W / L)
    camera_mu = __draw_camera_pyramid__(H=cam_H, W=cam_W, F=cam_F)
    camera_mu = camera_mu * cam_draw_scale
    camera_mu = torch.from_numpy(camera_mu).to(device).float()

    # middle_T = cams.T // 2
    # _T_cw = pose_list[middle_T]
    # mid_cam_ori_w = cams.T_wc(middle_T)[:3, -1]
    # mid_cam_ori_c = _T_cw[:3,:3] @ mid_cam_ori_w + _T_cw[:3,-1]
    # # distance to the camera

    for cam_tid in tqdm(range(len(pose_list))):
        working_t = cam_tid if model_t is None else model_t

        gs5 = []
        assert bg_flag or fg_flag
        if bg_flag:
            gs5.append(s_model())
        if fg_flag:
            gs5.append(d_model(working_t))

        # * identyfy the visible GS
        visible_render_dict = render(
            gs5,
            cams.default_H,
            cams.default_W,
            K=cams.K(),
            T_cw=cams.T_cw(cam_tid),
            bg_color=bg_color,
        )
        # mu_cat = torch.cat([it[0] for it in gs5], 0)
        # dep = visible_render_dict["dep"].detach()[0]
        # mask = visible_render_dict["alpha"].detach()[0] > 0.5
        # back_pts = cams.backproject(cams.homo()[mask], dep[mask])
        # back_pts_world = cams.trans_pts_to_world(working_t, back_pts)
        # dist_sq, nearest_id, _ = knn_points(back_pts_world[None], mu_cat[None], K=K)
        # dist_sq = dist_sq[0, :].reshape(-1)
        # nearest_id = nearest_id[0, :].reshape(-1)
        # valid_nn_mask = dist_sq < (d_model.scf.spatial_unit * 3.0) ** 2
        # nearest_id = nearest_id[valid_nn_mask]
        # visibl_emask = torch.zeros_like(mu_cat[:, 0]).bool()
        # if len(nearest_id) > 0:
        #     visible_mask[nearest_id] = True
        visible_mask = visible_render_dict["visibility_filter"]

        gs5_cat = []
        for i in range(5):
            gs5_cat.append(torch.cat([it[i] for it in gs5], 0))
        new_opa = gs5_cat[-2]
        new_opa[~visible_mask] = new_opa[~visible_mask] * invisble_opa_factor
        gs5_cat[-2] = new_opa

        # convert to gray scale
        gray_sph = RGB2SH(
            torch.mean(
                SH2RGB(gs5_cat[-1][~visible_mask, :3]), dim=1, keepdim=True
            ).expand(-1, 3)
        )
        gray_sph[:, 0] = (
            gray_sph[:, 0] * (1.0 - inivisble_red_ratio) + inivisble_red_ratio
        )
        gray_sph[:, 1:] = gray_sph[:, 1:] * (1.0 - inivisble_red_ratio) + 0.0
        pad_sph_dim = s_model()[-1].shape[1]
        if pad_sph_dim > gray_sph.shape[1]:
            gray_sph = F.pad(gray_sph, (0, pad_sph_dim - gray_sph.shape[1], 0, 0))
        gs5_cat[-1][~visible_mask] = gray_sph

        # * draw also the current camera frame in the scene
        add_mu = cams.trans_pts_to_world(working_t, camera_mu)
        add_fr = (
            torch.eye(3).to(add_mu.device).unsqueeze(0).expand(add_mu.shape[0], -1, -1)
        )
        add_s = torch.ones_like(add_mu) * 0.001
        add_o = torch.ones_like(add_s[:, :1]) * 1.0  # 0.4
        add_sph = torch.ones_like(add_s) * 0.0
        add_sph[:, 1] = 1.0
        if pad_sph_dim > add_sph.shape[1]:
            add_sph = F.pad(add_sph, (0, pad_sph_dim - add_sph.shape[1], 0, 0))

        render_dict = render(
            [
                gs5_cat,
                [
                    add_mu.to(device),
                    add_fr.to(device),
                    add_s.to(device),
                    add_o.to(device),
                    add_sph.to(device),
                ],
            ],
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            bg_color=bg_color,
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)
        dep = render_dict["dep"].detach().cpu().numpy().squeeze(0)
        dep_viz_list.append(dep)
        if "normal" in render_dict:
            normal = render_dict["normal"].detach().cpu().numpy()
            normal_viz = (1 - normal) / 2
            normal_viz_list.append(normal_viz.transpose(1, 2, 0))

        # imageio.imsave("./debug/dbg.jpg", rgb_viz)

    # # use disp map to viz the depth!
    # viz_dep = np.stack(dep_viz_list, axis=0)
    # valid_mask = viz_dep > 0
    # max_dep, min_dep = viz_dep[valid_mask].max(), viz_dep[valid_mask].min()
    # viz_dep[valid_mask] = (viz_dep[valid_mask] - min_dep) / (max_dep - min_dep)
    # # viz_dep = [plt.cm.plasma(it)[:,:,:3] * 255 for it in viz_dep]
    # viz_dep = [plt.cm.viridis(it)[:, :, :3] * 255 for it in viz_dep]

    # save_frame_list(viz_dep, save_fn + "_dep")

    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    if len(normal_viz_list) > 0:
        print(normal_viz_list[0].shape)
        save_frame_list(normal_viz_list, save_fn + "_normal")
    return


def outlier_removal_o3d(xyz, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    _, inlier_ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    inlier_mask = torch.zeros_like(xyz[:, 0]).bool()
    inlier_mask[inlier_ind] = True
    return inlier_mask


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


def __draw_camera_pyramid__(n_pts_per_line=100, H=1.0, W=1.0, F=1.0):
    # get a list of xyz position of opencv camera pyramid
    # the forward z is facing the scene
    cam_points = np.array(
        [
            [0, 0, 0],  # camera center
            [W / 2, H / 2, F],  # top-right
            [W / 2, -H / 2, F],  # bottom-right
            [-W / 2, -H / 2, F],  # bottom-left
            [-W / 2, H / 2, F],  # top-left
        ]
    )
    lines = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),  # from center to corners
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),  # between corners
    ]
    xyz = []
    for start, end in lines:
        line_dir = cam_points[end] - cam_points[start]
        line_points = (
            cam_points[start][None, :]
            + np.linspace(0, 1, n_pts_per_line)[:, None] * line_dir[None, :]
        )
        xyz.append(line_points)
    return np.concatenate(xyz, axis=0)


@torch.no_grad()
def viz_list_of_colored_points_in_cam_frame(
    xyz_list,  # list[N,3]
    color,  # list[N,3] or N,3
    bg_color=[1.0, 1.0, 1.0],
    device=torch.device("cuda:0"),
    # camera control
    H=480,
    W=480,
    rel_focal=1.0,
    zoom_out_factor=0.0,
    pitch_deg=30.0,
):
    # the saved momap is the 3D position in current view frame
    T = len(xyz_list)
    if not isinstance(color, list):
        color = [color] * T
    else:
        assert len(color) == T

    # the momap is in world frame, where the world is the camera frame of reference (src_t)

    # compute a camera pose, all the mu are in world frame
    T_wc_viz = torch.eye(4)
    # manipulate the pose
    dep = torch.cat(xyz_list, 0)[:, 2]
    dep_median = np.median(dep)
    dist_to_center = dep_median * (1 + zoom_out_factor)
    lift_height = np.sin(np.deg2rad(pitch_deg)) * dist_to_center
    lift_back = np.cos(np.deg2rad(pitch_deg)) * dist_to_center - dep_median
    T_wc_viz[1, 3] -= lift_height
    T_wc_viz[2, 3] -= lift_back
    T_wc_viz[:3, :3] = T_wc_viz[:3, :3] @ euler2mat(
        -np.deg2rad(pitch_deg), 0, 0, "sxyz"
    )
    T_cw_viz = np.linalg.inv(T_wc_viz)
    T_cw_viz = torch.from_numpy(T_cw_viz).float().to(device)

    viz_list = []
    for t in tqdm(range(T)):
        mu_w = xyz_list[t].float().to(device)
        mu = torch.einsum("ij,nj->ni", T_cw_viz[:3, :3], mu_w) + T_cw_viz[:3, 3]
        rgb = color[t].float().to(device)

        dep = mu_w[:, 2]
        scale = dep / (rel_focal * min(H, W)) * 2.0

        fr = torch.eye(3)[None].expand(len(mu_w), -1, -1).to(device)
        s = scale.reshape(-1, 1).float().to(device).expand(-1, 3)
        o = torch.ones(len(mu_w), 1).to(device)
        sph = RGB2SH(rgb.reshape(-1, 3).float().to(device))

        render_dict = render_cam_pcl(
            mu,
            fr,
            s,
            o,
            sph,
            H=H,
            W=W,
            fx=rel_focal / 2.0 * min(H, W),
            fy=rel_focal / 2.0 * min(H, W),
            bg_color=bg_color,
        )
        _viz = render_dict["rgb"].cpu().permute(1, 2, 0).numpy() * 255
        _viz = np.clip(_viz, 0, 255).astype(np.uint8)
        viz_list.append(_viz)
        # imageio.imsave("./debug/rgb.jpg", _viz)
    # imageio.mimsave("./debug/momap.mp4", viz_list, fps=10)
    return viz_list
