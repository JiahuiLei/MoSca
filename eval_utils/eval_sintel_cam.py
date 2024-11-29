# ! This is modified from RoDyNRF

import os
import numpy as np
import argparse
import glob
from scipy.spatial.transform import Rotation
import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation
from evo.core.metrics import Unit
import sintel_io
from scipy.spatial.transform import Rotation


def eval_sintel_campose(poses_est, gt_dir):
    # pre-process the sintel-format poses to tum format poses
    gt_pose_lists = sorted(glob.glob(os.path.join(gt_dir, "*.cam")))
    tstamps = [float(x.split("/")[-1][:-4].split("_")[-1]) for x in gt_pose_lists]
    gt_poses = [sintel_io.cam_read(f)[1] for f in gt_pose_lists]
    xyzs, wxyzs = [], []
    tum_gt_poses = []
    for gt_pose in gt_poses:
        gt_pose = np.concatenate([gt_pose, np.array([[0, 0, 0, 1]])], 0)
        gt_pose_inv = np.linalg.inv(gt_pose)  # world2cam -> cam2world
        xyz = gt_pose_inv[:3, -1]
        xyzs.append(xyz)
        R = Rotation.from_matrix(gt_pose_inv[:3, :3])
        xyzw = R.as_quat()  # scalar-last for scipy
        wxyz = np.array([xyzw[-1], xyzw[0], xyzw[1], xyzw[2]])
        wxyzs.append(wxyz)
        tum_gt_pose = np.concatenate([xyz, xyzw], 0)
        tum_gt_poses.append(tum_gt_pose)

    tum_gt_poses = np.stack(tum_gt_poses, 0)
    tum_gt_poses[:, :3] = tum_gt_poses[:, :3] - np.mean(
        tum_gt_poses[:, :3], 0, keepdims=True
    )
    tt = np.expand_dims(np.stack(tstamps, 0), -1)
    tum_gt_poses = np.concatenate([tt, tum_gt_poses], -1)
    traj_ref = PoseTrajectory3D(
        positions_xyz=np.stack(xyzs, 0),
        orientations_quat_wxyz=np.stack(wxyzs, 0),
        timestamps=np.array(tstamps),
    )

    poses, traj_est = convert_to_tum(poses_est, tstamps)

    # if less than 80% images got valid poses, we treat this sequence as failed.
    if len(poses) < 0.8 * len(tstamps):
        return None, None, None
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    
    # ATE
    result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    ate = result.stats["rmse"]

    # RPE rotation and translation
    delta_list = [1]
    rpe_rots, rpe_transs = [], []
    for delta in delta_list:
        result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
        rot = result.stats["rmse"]
        rpe_rots.append(rot)

    for delta in delta_list:
        result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
        trans = result.stats["rmse"]
        rpe_transs.append(trans)
    rpe_trans, rpe_rot = np.mean(rpe_transs), np.mean(rpe_rots)

    return ate, rpe_trans, rpe_rot


def load_est_pose_rodynrf(input_dir, method):
    if method == "ParticleSfM":
        # ParticleSfM
        all_pose_files = sorted(
            glob.glob(input_dir + "/particleSfM/colmap_outputs_converted/poses/*.txt")
        )
        all_poses = []
        for pose_file in all_pose_files:
            pose = np.loadtxt(pose_file)
            if pose.shape[0] == 3:
                pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], 0)
            cam2world = np.linalg.inv(pose)
            all_poses.append(cam2world[:3, :])
        poses_est = np.stack(all_poses)

    elif method == "BARF":
        # BARF
        poses_est = np.load(input_dir + "/BARF_poses.npy")
        poses_est = np.concatenate(
            [poses_est[..., 1:2], -poses_est[..., :1], poses_est[..., 2:4]], -1
        )

    elif method == "nerfmm":
        # NeRF--
        poses_est = np.load(input_dir + "/nerfmm_poses.npy")[:, :3, :]
        poses_est = np.concatenate(
            [poses_est[..., 1:2], -poses_est[..., :1], poses_est[..., 2:4]], -1
        )

    elif method == "Ours":
        # Ours
        poses_est = np.load(input_dir + "/poses_bounds_ours.npy")[:, :15].reshape(
            -1, 3, 5
        )[:, :, :4]
        poses_est = np.concatenate(
            [poses_est[..., 1:2], -poses_est[..., :1], poses_est[..., 2:4]], -1
        )
    return poses_est


def convert_to_tum(poses_est, tstamps):
    eye = np.stack([np.eye(4)] * poses_est.shape[0], 0)
    eye[:, :3, :] = poses_est
    poses_est = np.copy(eye)
    # pre-process the colmap-format poses to tum format poses
    poses = []
    tum_poses = []
    for pose_instance in poses_est:
        R, t = pose_instance[:3, :3], pose_instance[:3, 3]
        rot = Rotation.from_matrix(R)
        quad = rot.as_quat()  # xyzw
        quad_wxyz = np.array([quad[3], quad[0], quad[1], quad[2]])
        pose_t = np.concatenate([t, quad_wxyz], 0)  # [tx, ty, tz, qw, qx, qy, qz]
        # tum pose format
        tum_pose_t = np.concatenate([t, quad], 0)
        poses.append(pose_t)
        tum_poses.append(tum_pose_t)
    poses = np.stack(poses, 0)
    tum_poses = np.stack(tum_poses, 0)
    # np.savetxt("./Ours.txt", tum_poses)
    traj_est = PoseTrajectory3D(
        positions_xyz=poses[:, 0:3],
        orientations_quat_wxyz=poses[:, 3:],
        timestamps=np.array(tstamps),
    )
    return poses, traj_est


def main_rodynrf(base_dir):
    scenes = [
        "alley_2",
        "ambush_4",
        "ambush_5",
        "ambush_6",
        "cave_2",
        "cave_4",
        "market_2",
        "market_5",
        "market_6",
        "shaman_3",
        "sleeping_1",
        "sleeping_2",
        "temple_2",
        "temple_3",
    ]
    method = "Ours"  # ParticleSfM, BARF, nerfmm, Ours
    ate_list = []
    rpe_trans_list = []
    rpe_rot_list = []
    for sq in scenes:
        # gt_dir = os.path.join(base_dir, sq)
        # debug
        gt_dir = os.path.join("./data/sintel_dev", sq, "gt_cameras")
        est_pose = load_est_pose_rodynrf(os.path.join(base_dir, sq), method)

        ate, rpe_trans, rpe_rot = eval_sintel_campose(est_pose, gt_dir=gt_dir)

        ate_list.append(ate)
        rpe_trans_list.append(rpe_trans)
        rpe_rot_list.append(rpe_rot)
        print(sq)
        print(ate, rpe_trans, rpe_rot)
        print("-" * 20)
    print("=" * 20)
    print("mean")
    print("=" * 20)
    print("ATE", np.mean(np.array(ate_list)))
    print("RPE", np.mean(np.array(rpe_trans_list)))
    print("ROT", np.mean(np.array(rpe_rot_list)))
