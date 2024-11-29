# Jiahui 2024.Feb.27
# * copied and verified from NOPE-NERF, produce exactly the same numbers with their origial code
# ! use PNG not JPG to avoid the jpeg compression misalignment

import torch, numpy as np
import torch.nn.functional as F
import os, os.path as osp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import lpips as lpips_lib
from tqdm import tqdm
import imageio
from scipy.spatial.transform import Rotation as RotLib
import math
import logging


def eval_pose(gt_path, pred_path):
    gt_poses = torch.from_numpy(np.load(gt_path))
    pred_pose = torch.from_numpy(np.load(pred_path))

    c2ws_est_aligned = align_ate_c2b_use_a2b(pred_pose, gt_poses)  # (N, 4, 4)
    # compute ate
    ate = compute_ATE(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    rpe_trans, rpe_rot = compute_rpe(
        gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy()
    )
    # logging.info(
    #     "{0:.3f}".format(rpe_trans * 100),
    #     "&" "{0:.3f}".format(rpe_rot * 180 / np.pi),
    #     "&",
    #     "{0:.3f}".format(ate),
    # )
    logging.info(
        f"RPE_T={rpe_trans*100:.3f} RPE_R={rpe_rot*180/np.pi:.3f} ATE={ate:.3f}"
    )
    return rpe_trans * 100, rpe_rot * 180 / np.pi, ate


def eval_rendering_dir(gt_dir, pred_dir, device=torch.device("cuda")):
    # * Verified with original code!
    lpips_vgg_fn = lpips_lib.LPIPS(net="vgg").to(device)
    gt_fns = os.listdir(gt_dir)
    pred_fns = os.listdir(pred_dir)
    gt_fns.sort()
    pred_fns.sort()

    assert len(gt_fns) == len(pred_fns)

    psnr_list, ssim_list, lpips_list = [], [], []
    for i in tqdm(range(len(gt_fns))):
        gt_fn = osp.join(gt_dir, gt_fns[i])
        pred_fn = osp.join(pred_dir, pred_fns[i])
        img_gt = torch.from_numpy(imageio.imread(gt_fn).astype(np.float32) / 255.0).to(
            device
        )
        if pred_fn.endswith(
            "npy"
        ):  # ! to check whether this eval aligns with the original code!
            img_out = torch.from_numpy(np.load(pred_fn)).to(device)
        elif pred_fn.endswith("png"):
            img_out = torch.from_numpy(imageio.imread(pred_fn) / 255.0).to(device)
        else:
            raise ValueError(f"Unknown file format: {pred_fn}, Don't use JPG!!!")
        # debug
        # dbg = np.load("./dbg.npz")["gt"] - img_gt.cpu().numpy()
        img_gt = img_gt.float()
        img_out = img_out.float()
        metric_psnr = mse2psnr(F.mse_loss(img_out, img_gt).item())
        metric_ssim = ssim(
            img_out.permute(2, 0, 1).unsqueeze(0), img_gt.permute(2, 0, 1).unsqueeze(0)
        ).item()
        metric_lpips = lpips_vgg_fn(
            img_out.permute(2, 0, 1).unsqueeze(0).contiguous(),
            img_gt.permute(2, 0, 1).unsqueeze(0).contiguous(),
            normalize=True,
        ).item()
        logging.info(
            "[{}] PSNR: {:.2f}, SSIM: {:.2f}, LPIPS: {:.2f}".format(
                i, metric_psnr, metric_ssim, metric_lpips
            )
        )
        psnr_list.append(metric_psnr)
        ssim_list.append(metric_ssim)
        lpips_list.append(metric_lpips)
    ave_psnr = np.mean(psnr_list)
    ave_ssim = np.mean(ssim_list)
    ave_lpips = np.mean(lpips_list)
    logging.info(
        f"Average PSNR: {ave_psnr:.2f}, SSIM: {ave_ssim:.2f}, LPIPS: {ave_lpips:.2f}"
    )

    return psnr_list, ssim_list, lpips_list


def compute_rpe(gt, pred):
    trans_errors = []
    rot_errors = []
    for i in range(len(gt) - 1):
        gt1 = gt[i]
        gt2 = gt[i + 1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i + 1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel

        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))
    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot


def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2 + dy**2 + dz**2)
    return trans_error


def compute_ATE(gt, pred):
    """Compute RMSE of ATE
    Args:
        gt: ground-truth poses
        pred: predicted poses
    """
    errors = []

    for i in range(len(pred)):
        # cur_gt = np.linalg.inv(gt_0) @ gt[i]
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3]

        # cur_pred = np.linalg.inv(pred_0) @ pred[i]
        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz

        errors.append(np.sqrt(np.sum(align_err**2)))
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
    return ate


def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    device = traj_a.device
    if traj_c is None:
        traj_c = traj_a.clone()

    traj_a = traj_a.float().cpu().numpy()
    traj_b = traj_b.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)
    quat_a = SO3_to_quat(R_a)  # (N0, 4)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)
    quat_b = SO3_to_quat(R_b)  # (N0, 4)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method="sim3")

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)

    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

    # append the last row
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
    return traj_c_aligned  # (N1, 4, 4)


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat(
                [input, torch.zeros_like(input[:, 0:1])], dim=1
            )  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat(
                [
                    input,
                    torch.tensor(
                        [[0, 0, 0, 1]], dtype=input.dtype, device=input.device
                    ),
                ],
                dim=0,
            )  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate(
                [input, np.zeros_like(input[:, 0:1])], axis=1
            )  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0
            )  # (4, 4)
            output[3, 3] = 1.0
    return output


# a general interface
def alignTrajectory(p_es, p_gt, q_es, q_gt, method, n_aligned=-1):
    """
    calculate s, R, t so that:
        gt = R * s * est + t
    method can be: sim3, se3, posyaw, none;
    n_aligned: -1 means using all the frames
    """
    assert p_es.shape[1] == 3
    assert p_gt.shape[1] == 3
    assert q_es.shape[1] == 4
    assert q_gt.shape[1] == 4

    s = 1
    R = None
    t = None
    if method == "sim3":
        assert n_aligned >= 2 or n_aligned == -1, "sim3 uses at least 2 frames"
        s, R, t = alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == "se3":
        R, t = alignSE3(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == "posyaw":
        R, t = alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == "none":
        R = np.identity(3)
        t = np.zeros((3,))
    else:
        assert False, "unknown alignment method"

    return s, R, t


def alignPositionYawSingle(p_es, p_gt, q_es, q_gt):
    """
    calcualte the 4DOF transformation: yaw R and translation t so that:
        gt = R * est + t
    """

    p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
    p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]
    g_rot = quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]

    C_R = np.dot(est_rot, g_rot.transpose())
    theta = get_best_yaw(C_R)
    R = rot_z(theta)
    t = p_gt_0 - np.dot(R, p_es_0)

    return R, t


def alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned=1):
    if n_aligned == 1:
        R, t = alignPositionYawSingle(p_es, p_gt, q_es, q_gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        _, R, t = align_umeyama(
            gt_pos, est_pos, known_scale=True, yaw_only=True
        )  # note the order
        t = np.array(t)
        t = t.reshape((3,))
        R = np.array(R)
        return R, t


_EPS = np.finfo(float).eps * 4.0


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> np.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


# align by a SE3 transformation
def alignSE3Single(p_es, p_gt, q_es, q_gt):
    """
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    Using only the first poses of est and gt
    """

    p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
    p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]

    g_rot = quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]

    R = np.dot(g_rot, np.transpose(est_rot))
    t = p_gt_0 - np.dot(R, p_es_0)

    return R, t


def alignSE3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
    """
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    """
    if n_aligned == 1:
        R, t = alignSE3Single(p_es, p_gt, q_es, q_gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        s, R, t = align_umeyama(gt_pos, est_pos, known_scale=True)  # note the order
        t = np.array(t)
        t = t.reshape((3,))
        R = np.array(R)
        return R, t


def get_best_yaw(C):
    """
    maximize trace(Rz(theta) * C)
    """
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta


def rot_z(theta):
    R = rotation_matrix(theta, [0, 0, 1])
    R = R[0:3, 0:3]

    return R


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis.

    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3), dtype=np.float64)
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1.0]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2., np.trace(rotation_matrix(math.pi/2,
    ...                                                direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float64
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float64,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def align_umeyama(model, data, known_scale=False, yaw_only=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)

    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0 / n * np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0 / n * np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)

    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0:
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0 / sigma2 * np.trace(np.dot(D_svd, S))

    t = mu_M - s * np.dot(R, mu_D)

    return s, R, t


# align by similarity transformation
def alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
    """
    calculate s, R, t so that:
        gt = R * s * est + t
    """
    idxs = _getIndices(n_aligned, p_es.shape[0])
    est_pos = p_es[idxs, 0:3]
    gt_pos = p_gt[idxs, 0:3]
    s, R, t = align_umeyama(gt_pos, est_pos)  # note the order
    return s, R, t


def _getIndices(n_aligned, total_n):
    if n_aligned == -1:
        idxs = np.arange(0, total_n)
    else:
        assert n_aligned <= total_n and n_aligned >= 1
        idxs = np.arange(0, n_aligned)
    return idxs


def SO3_to_quat(R):
    """
    :param R:  (N, 3, 3) or (3, 3) np
    :return:   (N, 4, ) or (4, ) np
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat


def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, use_padding, size_average=True):

    if use_padding:
        padding_size = window_size // 2
    else:
        padding_size = 0

    mu1 = F.conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, use_padding=True, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.use_padding = use_padding
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(
            img1,
            img2,
            window,
            self.window_size,
            channel,
            self.use_padding,
            self.size_average,
        )


def ssim(img1, img2, use_padding=True, window_size=11, size_average=True):
    """SSIM only defined at intensity channel. For RGB or YUV or other image format, this function computes SSIm at each
    channel and averge them.
    :param img1:  (B, C, H, W)  float32 in [0, 1]
    :param img2:  (B, C, H, W)  float32 in [0, 1]
    :param use_padding: we use conv2d when we compute mean and var for each patch, this use_padding is for that conv2d.
    :param window_size: patch size
    :param size_average:
    :return:  a tensor that contains only one scalar.
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, use_padding, size_average)


if __name__ == "__main__":

    eval_pose(
        gt_path="./out/Tanks/Ignatius/gt/train_T_wc.npy",
        pred_path="./out/Tanks/Ignatius/extraction/eval/pre/pred_cameras.npy",
    )

    eval_rendering_dir(
        gt_dir="./out/Tanks/Ignatius/gt/test_images/",
        pred_dir="./out/Tanks/Ignatius/extraction/eval/pre/img_out_np/",
    )
