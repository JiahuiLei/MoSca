# Single File
from matplotlib import pyplot as plt
import torch, numpy as np
import os, sys, os.path as osp
from tqdm import tqdm
import logging, imageio
from pytorch3d.ops import knn_points
from matplotlib import cm
import cv2

sys.path.append(osp.dirname(osp.abspath(__file__)))


from camera import MonocularCameras
from viz_helper import make_video_from_pattern, viz_global_ba
from robust_utils import positive_th_gaussian_decay


def compute_static_ba(
    s2d,
    log_dir,
    s_track,
    s_track_valid_mask,
    cams: MonocularCameras,
    max_t_per_step=10000,
    total_steps=2000,  # 6000
    switch_to_ind_step=1000,  # this is also the scheduler start!
    max_num_of_tracks=10000,
    depth_correction_after_step=1000,
    # lr and lambda
    lr_cam_q=0.0003,
    lr_cam_t=0.0003,
    lr_cam_f=0.0003,
    lr_dep_s=0.001,
    lr_dep_c=0.001,
    lambda_flow=1.0,
    lambda_depth=0.1,
    lambda_small_correction=0.03,
    # camera pose smoothness
    lambda_cam_smooth_trans=0.0,
    lambda_cam_smooth_rot=0.0,
    # viz
    viz_verbose_n=300,
    viz_fig_n=300,
    viz_denser_range=[],  # [[0, 10]],  # [[0, 40], [1000, 1040]],
    viz_denser_interval=1,
    save_more_flag=False,
    viz_video_rgb=None,
    # robustify
    huber_delta=-1,
    # ! all these robust weights are computed on the fly, which assume that the initializaiton is good
    depth_decay_th=2.0,
    depth_decay_sigma=1.0,
    std_decay_th=0.2,
    std_decay_sigma=0.2,
    #
    optimizer_class=torch.optim.Adam,
):

    viz_dir = osp.join(log_dir, "static_ba_viz")
    os.makedirs(viz_dir, exist_ok=True)

    # prepare dense track
    # s_track = s2d.track[:, s2d.static_track_mask, :2].clone()
    s_track = s_track[..., :2].clone()
    s_track_valid_mask = s_track_valid_mask.clone()
    device = s_track.device
    if max_num_of_tracks < s_track.shape[1]:
        logging.info(
            f"Track is too dense {s_track.shape[1]}, radom sample to {max_num_of_tracks}"
        )
        choice = torch.randperm(s_track.shape[1])[:max_num_of_tracks]
        s_track = s_track[:, choice]
        s_track_valid_mask = s_track_valid_mask[:, choice]

    homo_list, dep_list, rgb_list = prepare_track_homo_dep_rgb_buffers(
        s2d, s_track, s_track_valid_mask, torch.arange(s2d.T).to(device)
    )

    if viz_video_rgb is not None:
        logging.info(f"Viz BA points on each frame...")
        viz_frames = viz_ba_point(viz_video_rgb, s_track, s_track_valid_mask)
        imageio.mimsave(osp.join(log_dir, "BA_points.mp4"), viz_frames)

    # * start solve global init of the camera
    logging.info(
        f"Static Scaffold BA: Depth correction after {depth_correction_after_step}; Lr Scheduling and Ind after {switch_to_ind_step} steps (total {total_steps})"
    )
    param_scale = torch.ones(cams.T).to(device)
    param_scale.requires_grad_(True)
    param_dep_corr = torch.zeros_like(dep_list).clone()
    param_dep_corr.requires_grad_(True)
    optim_list = cams.get_optimizable_list(lr_f=lr_cam_f, lr_q=lr_cam_q, lr_t=lr_cam_t)
    if lr_dep_s > 0:
        optim_list.append(
            {"params": [param_scale], "lr": lr_dep_s, "name": "cam_scale"}
        )
    if lr_dep_c > 0:
        optim_list.append(
            {"params": [param_dep_corr], "lr": lr_dep_c, "name": "dep_correction"}
        )
    optimizer = optimizer_class(optim_list)
    scheduler = None
    s_track_valid_mask_w = s_track_valid_mask.float()
    s_track_valid_mask_w = s_track_valid_mask_w / s_track_valid_mask_w.sum(0)

    if huber_delta > 0:
        logging.info(f"Use Huber Loss with delta={huber_delta}")
        huber_loss = torch.nn.HuberLoss(reduction="none", delta=huber_delta)

    loss_list, std_list, fovx_list, fovy_list = [], [], [], []
    flow_loss_list, dep_loss_list, dep_corr_loss_list = [], [], []
    cam_rot_loss_list, cam_trans_loss_list = [], []

    logging.info(f"Start Static BA with {cams.T} frames and {dep_list.shape[1]} points")

    for step in tqdm(range(total_steps)):
        if step == switch_to_ind_step:
            logging.info(
                "Switch to Independent Camera Optimization and Start Scheduling"
            )
            cams.disable_delta()
            optim_list = cams.get_optimizable_list(
                lr_f=lr_cam_f, lr_q=lr_cam_q, lr_t=lr_cam_t
            )
            if lr_dep_s > 0:
                optim_list.append(
                    {"params": [param_scale], "lr": lr_dep_s, "name": "cam_scale"}
                )
            if lr_dep_c > 0:
                optim_list.append(
                    {
                        "params": [param_dep_corr],
                        "lr": lr_dep_c,
                        "name": "dep_correction",
                    }
                )
            optimizer = optimizer_class(optim_list)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                total_steps - switch_to_ind_step,
                eta_min=min(lr_cam_q, lr_cam_t) / 10.0,
            )

        optimizer.zero_grad()

        ########################
        dep_scale = param_scale.abs()
        dep_scale = dep_scale / dep_scale.mean()

        scaled_depth_list = dep_list * dep_scale[:, None]
        if step > depth_correction_after_step:
            scaled_depth_list = scaled_depth_list + param_dep_corr
            dep_corr_loss = abs(param_dep_corr).mean()
        else:
            dep_corr_loss = torch.zeros_like(dep_scale[0])
        point_ref = get_world_points(homo_list, scaled_depth_list, cams)  # T,N,3

        # transform to each frame!
        if cams.T > max_t_per_step:
            tgt_inds = torch.randperm(cams.T)[:max_t_per_step].to(device)
        else:
            tgt_inds = torch.arange(cams.T).to(device)
        R_cw, t_cw = cams.Rt_cw_list()
        R_cw, t_cw = R_cw[tgt_inds], t_cw[tgt_inds]

        point_ref_to_every_frame = (
            torch.einsum("tij,snj->stni", R_cw, point_ref) + t_cw[None, :, None]
        )  # Src,Tgt,N,3
        uv_src_to_every_frame = cams.project(point_ref_to_every_frame)  # Src,Tgt,N,3

        # * robusitify the loss by down weight some curves
        with torch.no_grad():
            projection_singular_mask = abs(point_ref_to_every_frame[..., -1]) < 1e-5
            # no matter where the src is, it should be mapped to every frame with the gt tracking
            cross_time_mask = (
                s_track_valid_mask[:, None] * s_track_valid_mask[None, tgt_inds]
            ).float()
            cross_time_mask = (
                cross_time_mask * (~projection_singular_mask).float()
            )  # Src,Tgt,N

            depth_robust_w = positive_th_gaussian_decay(
                abs(scaled_depth_list), depth_decay_th, depth_decay_sigma
            )

            point_ref_mean = (point_ref * s_track_valid_mask_w[:, :, None]).sum(0)
            point_ref_std = (point_ref - point_ref_mean[None]).norm(dim=-1, p=2)
            point_ref_std_robust_w = (point_ref_std * s_track_valid_mask_w).sum(0)
            point_ref_std_robust_w = positive_th_gaussian_decay(
                point_ref_std_robust_w, std_decay_th, std_decay_sigma
            )

            robust_w = depth_robust_w * point_ref_std_robust_w[None]
            cross_robust_time_mask = robust_w[:, None] * robust_w[None, tgt_inds]
            cross_time_mask = cross_time_mask * cross_robust_time_mask.detach()

        uv_target = homo_list[None, tgt_inds].expand(
            len(uv_src_to_every_frame), -1, -1, -1
        )
        uv_loss_i = (uv_src_to_every_frame - uv_target).norm(dim=-1)

        if huber_delta > 0:
            uv_loss_i = huber_loss(uv_loss_i, torch.zeros_like(uv_loss_i))
        uv_loss = (uv_loss_i * cross_time_mask).sum() / (cross_time_mask.sum() + 1e-6)

        # compute depth loss
        dep_target = scaled_depth_list[None, tgt_inds].expand(
            len(uv_src_to_every_frame), -1, -1
        )
        warped_depth = point_ref_to_every_frame[..., -1]

        dep_consistency_i = 0.5 * abs(
            dep_target / torch.clamp(warped_depth, min=1e-6) - 1
        ) + 0.5 * abs(warped_depth / torch.clamp(dep_target, min=1e-6) - 1)
        # todo: this may be unstable... for fare away depth points!!!
        if huber_delta > 0:
            dep_consistency_i = huber_loss(
                dep_consistency_i, torch.zeros_like(dep_consistency_i)
            )
        dep_loss = (dep_consistency_i * cross_time_mask).sum() / (
            cross_time_mask.sum() + 1e-6
        )

        # camera smoothness reg
        if lambda_cam_smooth_rot > 0 or lambda_cam_smooth_trans > 0:
            cam_trans_loss, cam_rot_loss = cams.smoothness_loss()
        else:
            cam_trans_loss = torch.zeros_like(dep_loss)
            cam_rot_loss = torch.zeros_like(dep_loss)

        loss = (
            lambda_depth * dep_loss
            + lambda_flow * uv_loss
            + lambda_small_correction * dep_corr_loss
            + lambda_cam_smooth_rot * cam_rot_loss
            + lambda_cam_smooth_trans * cam_trans_loss
        )

        # loss = 0.0
        # if lambda_depth > 0:
        #     loss = loss + lambda_depth * dep_loss
        # if lambda_flow > 0:
        #     loss = loss + lambda_flow * uv_loss
        # if lambda_small_correction > 0:
        #     loss = loss + lambda_small_correction * dep_corr_loss
        assert torch.isnan(loss).sum() == 0 and torch.isinf(loss).sum() == 0
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # viz
        with torch.no_grad():
            point_ref_mean = (point_ref * s_track_valid_mask_w[:, :, None]).sum(0)
            std = (point_ref - point_ref_mean[None]).norm(dim=-1, p=2)
            metric_std = (std * s_track_valid_mask_w).sum(0).mean()
            loss_list.append(loss.item())
            dep_corr_loss_list.append(dep_corr_loss.item())
            flow_loss_list.append(uv_loss.item())
            dep_loss_list.append(dep_loss.item())
            cam_rot_loss_list.append(cam_rot_loss.item())
            cam_trans_loss_list.append(cam_trans_loss.item())
            std_list.append(metric_std.item())
            fov = cams.fov
            fovx_list.append(float(fov[0]))
            fovy_list.append(float(fov[1]))
            if step % viz_verbose_n == 0 or step == total_steps - 1:
                logging.info(f"loss={loss:.6f}, fov={cams.fov}")
                logging.info(f"scale max={param_scale.max()} min={param_scale.min()}")

            viz_flag = (
                np.array([step >= r[0] and step <= r[1] for r in viz_denser_range])
                .any()
                .item()
            )
            viz_flag = viz_flag and step % viz_denser_interval == 0
            viz_flag = viz_flag or step % viz_fig_n == 0 or step == total_steps - 1
            if viz_flag:
                # viz the 3D aggregation as well as the pcl in 3D!
                viz_frame = viz_global_ba(
                    point_ref,
                    rgb_list,
                    s_track_valid_mask,
                    cams,
                    error=std,
                    text=f"Step={step}",
                )
                imageio.imsave(
                    osp.join(viz_dir, f"static_scaffold_init_{step:06d}.jpg"),
                    (viz_frame * 255).astype(np.uint8),
                )

    # viz
    make_video_from_pattern(
        osp.join(viz_dir, "static_scaffold_init_*.jpg"),
        osp.join(log_dir, "static_scaffold_init.mp4"),
    )

    if total_steps > 0:
        fig = plt.figure(figsize=(21, 3))
        for plt_i, plt_pack in enumerate(
            [
                ("loss", loss_list),
                ("loss_flow", flow_loss_list),
                ("loss_dep", dep_loss_list),
                ("loss_dep_corr", dep_corr_loss_list),
                ("cam_rot", cam_rot_loss_list),
                ("cam_trans", cam_trans_loss_list),
                ("std", std_list),
                ("fov-x", fovx_list),
                ("fov-y", fovy_list),
            ]
        ):
            plt.subplot(1, 9, plt_i + 1)
            plt.plot(plt_pack[1]), plt.title(
                plt_pack[0] + f" End={plt_pack[1][-1]:.6f}"
            )
            if plt_pack[0].startswith("loss"):
                plt.yscale("log")
        plt.tight_layout()
        plt.savefig(osp.join(log_dir, f"static_scaffold_init.jpg"))
        plt.close()

    # update the depth scale
    # dep_scale = param_scale.abs()
    dep_scale = param_scale.abs()
    dep_scale = dep_scale / dep_scale.mean()
    logging.info(f"Update the S2D depth scale with {dep_scale}")
    s2d.rescale_depth(dep_scale)
    torch.save(cams.state_dict(), osp.join(log_dir, "bundle_cams.pth"))
    torch.save(
        {
            # sol
            "dep_scale": dep_scale,  # ! important, to later rescale the depth
            "dep_correction": param_dep_corr,
            "s_track": s_track,
            "s_track_mask": s_track_valid_mask,
        },
        osp.join(log_dir, "bundle.pth"),
    )
    # also save a reconstructed point cloud
    if save_more_flag:
        np.savetxt(
            osp.join(log_dir, "static_scaffold_pcl_unmerged.xyz"),
            torch.cat([point_ref, rgb_list], -1).reshape(-1, 6).detach().cpu().numpy(),
            fmt="%.6f",
        )
    return cams, s_track, s_track_valid_mask, param_dep_corr.detach().clone()


@torch.no_grad()
def deform_depth_map(
    depth_list,
    mask_list,
    cams: MonocularCameras,
    track_uv_list,
    track_mask_list,
    dep_correction,
    K=16,
    rbf_factor=0.333,
    viz_fn=None,
):
    # depth_list: T,H,W; track_uv_list: T,N,2; track_mask_list: T,N; src_buffer: T,N,C
    logging.info("Deforming depth")
    T = len(track_mask_list)
    H, W = depth_list[0].shape
    assert depth_list.shape == mask_list.shape
    assert T == len(track_uv_list) == len(dep_correction)
    assert T == len(depth_list) == len(mask_list)
    homo_map = torch.from_numpy(get_homo_coordinate_map(H, W)).to(depth_list[0])

    dep_corr_map_list, dep_new_map_list = [], []
    for tid in tqdm(range(T)):
        mask2d = mask_list[tid]
        scf_mask = track_mask_list[tid]
        dep_map = depth_list[tid]
        scf_uv = track_uv_list[tid][scf_mask]
        scf_int_uv, scf_inside_mask = round_int_coordinates(scf_uv, H, W)
        if not scf_inside_mask.all():
            logging.warning(
                f"Warning, {(~scf_inside_mask).sum()} invalid uv in t={tid}! may due to round accuracy"
            )

        scf_dep = query_image_buffer_by_pix_int_coord(depth_list[tid], scf_int_uv)
        scf_homo = query_image_buffer_by_pix_int_coord(homo_map, scf_int_uv)
        # this pts is used to distribute the carrying interp_src in 3D cam frame
        scf_cam_pts = cams.backproject(scf_homo, scf_dep)
        dst_cam_pts = cams.backproject(homo_map[mask2d], dep_map[mask2d])
        scf_buffer = dep_correction[tid][scf_mask]

        interp_dep_corr = spatial_interpolation(
            src_xyz=scf_cam_pts,
            src_buffer=scf_buffer[:, None],
            query_xyz=dst_cam_pts,
            K=K,
            rbf_sigma_factor=rbf_factor,
        )

        # viz
        dep_corr_map = torch.zeros_like(dep_map)
        dep_corr_map[mask2d] = interp_dep_corr.squeeze(-1)
        scf_corr_interp = query_image_buffer_by_pix_int_coord(dep_corr_map, scf_int_uv)
        # check_interp_error = (
        #     abs(scf_corr_interp - scf_buffer.squeeze(-1)).median()
        #     / abs(scf_buffer).median()
        # )s
        dep_corr_map_list.append(dep_corr_map.detach())
        dep_new_map_list.append((dep_corr_map + dep_map).detach())

    if viz_fn is not None:
        viz_corr_list, viz_dep_list = [], []
        for tid in tqdm(range(T)):
            viz_corr = dep_corr_map_list[tid]
            viz_dep = dep_new_map_list[tid]
            viz_corr_radius = abs(viz_corr).max()
            viz_corr = (viz_corr / viz_corr_radius) + 0.5
            viz_dep = (viz_dep - viz_dep.min()) / (viz_dep.max() - viz_dep.min())
            viz_corr = cm.viridis(viz_corr.cpu().numpy())
            viz_dep = cm.viridis(viz_dep.cpu().numpy())
            viz_corr = (viz_corr * 255).astype(np.uint8)
            viz_dep = (viz_dep * 255).astype(np.uint8)
            viz_corr_list.append(viz_corr)
            viz_dep_list.append(viz_dep)
        imageio.mimsave(viz_fn.replace(".mp4", "_corr.mp4"), viz_corr_list)
        imageio.mimsave(viz_fn.replace(".mp4", "_dep_corr.mp4"), viz_dep_list)

    dep_new_map_list = torch.stack(dep_new_map_list, 0)
    return dep_new_map_list


def spatial_interpolation(src_xyz, src_buffer, query_xyz, K=16, rbf_sigma_factor=0.333):
    # src_xyz: M,3 src_buffer: M,C query_xyz: N,3
    # build RBG on each src and smoothly interpolate the buffer to query
    # first construct src_xyz nn graph
    _dist_sq_to_nn, _, _ = knn_points(src_xyz[None], src_xyz[None], K=2)
    dist_to_nn = torch.sqrt(torch.clamp(_dist_sq_to_nn[0, :, 1:], min=1e-8)).squeeze(-1)
    rbf_sigma = dist_to_nn * rbf_sigma_factor  # M
    # find the nearest K neighbors for each query point to the src
    dist_sq, ind, _ = knn_points(query_xyz[None], src_xyz[None], K=K)
    dist_sq, ind = dist_sq[0], ind[0]

    w = torch.exp(-dist_sq / (2.0 * (rbf_sigma[ind] ** 2)))  # N,K
    w = w / torch.clamp(w.sum(-1, keepdim=True), min=1e-8)

    value = src_buffer[ind]  # N,K,C
    ret = torch.einsum("nk, nkc->nc", w, value)
    return ret


def get_homo_coordinate_map(H, W):
    # the grid take the short side has (-1,+1)
    if H > W:
        u_range = [-1.0, 1.0]
        v_range = [-float(H) / W, float(H) / W]
    else:  # H<=W
        u_range = [-float(W) / H, float(W) / H]
        v_range = [-1.0, 1.0]
    # make uv coordinate
    u, v = np.meshgrid(np.linspace(*u_range, W), np.linspace(*v_range, H))
    uv = np.stack([u, v], axis=-1)  # H,W,2
    return uv


def round_int_coordinates(coord, H, W):
    ret = coord.round().long()
    valid_mask = (
        (ret[..., 0] >= 0) & (ret[..., 0] < W) & (ret[..., 1] >= 0) & (ret[..., 1] < H)
    )
    ret[..., 0] = torch.clamp(ret[..., 0], 0, W - 1)
    ret[..., 1] = torch.clamp(ret[..., 1], 0, H - 1)
    return ret, valid_mask


def query_image_buffer_by_pix_int_coord(buffer, pixel_int_coordinate):
    assert pixel_int_coordinate.ndim == 2 and pixel_int_coordinate.shape[-1] == 2
    assert (pixel_int_coordinate[..., 0] >= 0).all()
    assert (pixel_int_coordinate[..., 0] < buffer.shape[1]).all()
    assert (pixel_int_coordinate[..., 1] >= 0).all()
    assert (pixel_int_coordinate[..., 1] < buffer.shape[0]).all()
    # u is the col, v is the row
    col_id, row_id = pixel_int_coordinate[:, 0], pixel_int_coordinate[:, 1]
    H, W = buffer.shape[:2]
    index = col_id + row_id * W
    ret = buffer.reshape(H * W, *buffer.shape[2:])[index]
    if isinstance(ret, np.ndarray):
        ret = ret.copy()
    return ret


def prepare_track_homo_dep_rgb_buffers(s2d, track, track_mask, t_list):
    # track: T,N,2, track_mask: T,N
    device = track.device
    homo_list, ori_dep_list, rgb_list = [], [], []
    for ind, tid in enumerate(t_list):
        _uv = track[ind]
        _int_uv, _inside_mask = round_int_coordinates(_uv, s2d.H, s2d.W)
        _dep = query_image_buffer_by_pix_int_coord(
            s2d.dep[tid].clone().to(device), _int_uv
        )
        _homo = query_image_buffer_by_pix_int_coord(
            s2d.homo_map.clone().to(device), _int_uv
        )
        ori_dep_list.append(_dep.to(device))
        homo_list.append(_homo.to(device))
        # for viz purpose
        _rgb = query_image_buffer_by_pix_int_coord(
            s2d.rgb[tid].clone().to(device), _int_uv
        )
        rgb_list.append(_rgb.to(device))
    rgb_list = torch.stack(rgb_list, 0)
    ori_dep_list = torch.stack(ori_dep_list, 0)
    homo_list = torch.stack(homo_list)
    ori_dep_list[~track_mask] = -1
    homo_list[~track_mask] = 0.0
    return homo_list, ori_dep_list, rgb_list


def query_buffers_by_track(image_buffer, track, track_mask, default_value=0.0):
    # image_buffer: T,H,W,C; track: T,N,2, track_mask: T,N
    assert image_buffer.ndim == 4 and track.ndim == 3 and track_mask.ndim == 2
    assert len(image_buffer) == len(track) == len(track_mask)
    T, H, W, C = image_buffer.shape
    N = track.shape[1]
    ret_buffer = torch.ones(T, N, C).to(image_buffer) * default_value

    for i in range(T):
        _uv = track[i][..., :2]
        _int_uv, _inside_mask = round_int_coordinates(_uv, H, W)
        _value = query_image_buffer_by_pix_int_coord(image_buffer[i].clone(), _int_uv)
        valid_mask = track_mask[i] & _inside_mask
        # for outside, put default value
        _value[~valid_mask] = default_value
        ret_buffer[i] = _value
    return ret_buffer


def get_world_points(homo_list, dep_list, cams, cam_t_list=None):
    T, M = dep_list.shape
    if cam_t_list is None:
        cam_t_list = torch.arange(T).to(homo_list.device)
    point_cam = cams.backproject(homo_list.reshape(-1, 2), dep_list.reshape(-1))
    point_cam = point_cam.reshape(T, M, 3)
    R_wc, t_wc = cams.Rt_wc_list()
    R_wc, t_wc = R_wc[cam_t_list], t_wc[cam_t_list]
    point_world = torch.einsum("tij,tmj->tmi", R_wc, point_cam) + t_wc[:, None]
    return point_world


def fovdeg2focal(fov_deg):
    focal = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    return focal


def track2undistroed_homo(track, H, W):
    # the short side is -1,1, the long side may exceed
    H, W = float(H), float(W)
    L = min(H, W)
    u, v = track[..., 0], track[..., 1]
    u = 2.0 * u / L - W / L
    v = 2.0 * v / L - H / L
    uv = torch.stack([u, v], -1)
    return uv


def viz_ba_point(viz_video_rgb, s_track, s_track_valid_mask):
    # todo: color the points by importance robust weight
    viz_frames = []
    for t in tqdm(range(len(viz_video_rgb))):
        frame_rgb = viz_video_rgb[t].copy()
        uv = s_track[t].cpu().numpy()
        _viz_valid_mask = s_track_valid_mask[t].cpu().numpy()
        for i in range(len(uv)):
            if _viz_valid_mask[i]:
                u, v = int(uv[i, 0]), int(uv[i, 1])
                if 0 <= u < frame_rgb.shape[1] and 0 <= v < frame_rgb.shape[0]:
                    _color = np.array(cm.hsv(float(i) / len(uv)))[:3] * 255
                    # put a circel with color
                    cv2.circle(frame_rgb, (u, v), 3, _color, 1)
        # put total valid num as text valid_mask.sum()
        cv2.putText(
            frame_rgb,
            f"Visible BA points: {_viz_valid_mask.sum()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        viz_frames.append(frame_rgb)
    return viz_frames
