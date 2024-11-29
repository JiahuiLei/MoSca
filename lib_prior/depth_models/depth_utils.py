import numpy as np
from matplotlib import cm
import imageio
import os, os.path as osp


def viz_depth_list(depths, save_fn, viz_quantile=3):

    dep_list = np.stack(depths, axis=0)

    # use disparity to viz, not depth
    dep_valid_mask = dep_list > 1e-6
    # use robust min and max to visualize
    dep_max = np.percentile(dep_list[dep_valid_mask], 100 - viz_quantile)
    dep_min = np.percentile(dep_list[dep_valid_mask], viz_quantile)
    dep_list = np.clip(dep_list, dep_min, dep_max)
    dep_list = (dep_list - dep_min) / (dep_max - dep_min)
    dep_list[~dep_valid_mask] = 0
    # h_dep = dep_list.reshape(-1)
    # plt.hist(h_dep, bins=300), plt.title("Depth Histogram")
    # plt.savefig(save_fn.replace(".mp4", ".jpg"))
    # plt.close()
    viz_list = []
    for dep in dep_list:
        viz = cm.viridis(dep)[:, :, :3]
        viz_list.append((viz * 255).astype(np.uint8))
    imageio.mimsave(save_fn, viz_list)
    return


def save_depth_list(dep_list, fn_list, dst, invalid_mask_list=None):
    assert len(dep_list) == len(fn_list)
    os.makedirs(dst, exist_ok=True)
    for i in range(len(dep_list)):
        if invalid_mask_list is not None:
            dep_list[i][invalid_mask_list[i] > 0] = 0
        save_fn = ".".join(fn_list[i].split(".")[:-1]) + ".npz"
        np.savez_compressed(osp.join(dst, save_fn), dep=dep_list[i])
    return
