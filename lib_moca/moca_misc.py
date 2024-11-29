import logging
import torch
import platform


def make_pair_list(N, interval=[1], dense_flag=False, track_mask=None, min_valid_num=0):
    # N: int, number of frames
    # interval: list of int, interval between frames
    # dense_flag: bool, whether to use dense pair
    pair_list = []
    for T in interval:
        for i in range(N - T):
            if T > 1 and not dense_flag:
                if i % T != 0:
                    continue
            if track_mask is not None:
                # check the common visib
                valid_num = (track_mask[i] & track_mask[i + T]).sum()
                if valid_num < min_valid_num:
                    continue
                    logging.info(
                        f"skip pair {i} {i+T} due to not enough valid num {valid_num}"
                    )
            pair_list.append((i, i + T))
    return pair_list


def Rt2T(R_list, t_list):
    # R_list: N,3,3, t_list: N,3
    assert len(R_list) == len(t_list)
    assert R_list.ndim == 3 and t_list.ndim == 2
    N = len(R_list)
    ret = torch.cat([R_list, t_list[:, :, None]], dim=2)
    bottom = torch.Tensor([0, 0, 0, 1.0]).to(ret)
    ret = torch.cat([ret, bottom[None, None].expand(N, -1, -1)], dim=1)
    return ret


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


def configure_logging(log_path, debug=False):
    """
    https://github.com/facebookresearch/DeepSDF
    """
    logging.getLogger().handlers.clear()
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    logger_handler.addFilter(HostnameFilter())
    formatter = logging.Formatter(
        "| %(hostname)s | %(levelname)s | %(asctime)s | %(message)s   [%(filename)s:%(lineno)d]",
        "%b-%d-%H:%M:%S",
    )
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    file_logger_handler = logging.FileHandler(log_path)

    file_logger_handler.setFormatter(formatter)
    logger.addHandler(file_logger_handler)


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


@torch.no_grad()
def get_all_world_pts_list(homo, dep_list, rgb_list, mask_list, cams):
    ret = []
    T = len(dep_list)
    assert len(rgb_list) == T and len(mask_list) == T and cams.T == T
    for t in range(T):
        mask = mask_list[t]
        cam_xyz = cams.backproject(homo[mask], dep_list[t][mask])

        world_xyz = cams.trans_pts_to_world(t, cam_xyz)
        rgb = rgb_list[t][mask]
        world_xyz = torch.cat([world_xyz, rgb], -1)
        ret.append(world_xyz.detach().cpu())
    return ret
