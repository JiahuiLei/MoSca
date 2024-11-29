import torch, numpy as np


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


def prepare_track_buffers(s2d, track, track_mask, t_list):
    # track: T,N,2, track_mask: T,N
    homo_list, ori_dep_list, rgb_list = [], [], []
    for ind, tid in enumerate(t_list):
        _uv = track[ind]
        _int_uv, _inside_mask = round_int_coordinates(_uv, s2d.H, s2d.W)
        _dep = query_image_buffer_by_pix_int_coord(s2d.dep[tid].clone(), _int_uv)
        _homo = query_image_buffer_by_pix_int_coord(s2d.homo_map.clone(), _int_uv)
        ori_dep_list.append(_dep)
        homo_list.append(_homo)
        # for viz purpose
        _rgb = query_image_buffer_by_pix_int_coord(s2d.rgb[tid].clone(), _int_uv)
        rgb_list.append(_rgb)
    rgb_list = torch.stack(rgb_list, 0)
    ori_dep_list = torch.stack(ori_dep_list, 0)
    homo_list = torch.stack(homo_list)
    ori_dep_list[~track_mask] = -1
    homo_list[~track_mask] = 0.0
    return homo_list, ori_dep_list, rgb_list


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
