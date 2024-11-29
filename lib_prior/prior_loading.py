# helpers for loading saved priors
import logging, sys, os, os.path as osp
from glob import glob
import torch, numpy as np
from torch import nn
import imageio, cv2
from tqdm import tqdm
import time

sys.path.append(osp.dirname(osp.abspath(__file__)))
from epi_error.epi_analysis import load_epi_error  # , load_vos
from tracking.cotracker_visualizer import Visualizer


def RGB2INST(x):
    assert x.shape[-1] == 3
    y = x.astype(np.int32)
    y = y[..., 0] + y[..., 1] * 256 + y[..., 2] * 256**2
    return y


@torch.no_grad()
def gather_track_from_buffer(track, track_base_mask, buffer_list):
    # buffer_list: T, H, W, C
    if buffer_list.ndim == 3:
        buffer_list = buffer_list.unsqueeze(3)
    T, N = track_base_mask.shape
    # fill the ret with nan
    ret = torch.zeros(T, N, buffer_list.shape[-1], dtype=buffer_list.dtype)
    ret = ret.to(track.device)
    ret.fill_(np.nan)
    ret_mask = torch.zeros_like(track_base_mask) > 0
    for tid in range(T):
        _mask = track_base_mask[tid]
        _uv_int = track[tid][_mask]
        _value = query_image_buffer_by_pix_int_coord(buffer_list[tid], _uv_int)
        ret[tid][_mask] = _value
        ret_mask[tid][_mask] = True
    return ret, ret_mask


@torch.no_grad()
def filter_track(track, track_vis, dep_mask_list, min_valid_cnt=4):
    # query valid depth and visible mask
    track_dep_mask, _ = gather_track_from_buffer(
        track[..., :2].long(), track_vis, dep_mask_list
    )
    track_mask = track_dep_mask.squeeze(-1) * track_vis
    # check whether a track is visible and has valid depth more more thant min_valid_cnt times
    valid_cnt = track_mask.sum(0)
    filter_track_mask = valid_cnt >= min_valid_cnt
    logging.info(
        f"Valid check: min_cnt={min_valid_cnt} {(~filter_track_mask).sum()} tracks are removed!"
    )
    track = track[:, filter_track_mask]
    track_mask = track_mask[:, filter_track_mask].clone()
    return track, track_mask


def laplacian_filter_depth(depths, threshold_ratio=0.5, ksize=5, open_ksize=3):
    # logging.info("Filtering depth maps...")
    # filter the depth changing boundary, they are not reliable
    dep_boundary_errors, dep_valid_masks = [], []
    ellip_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
    )
    for dep in depths:
        # detect the edge boundary of depth
        dep = dep.astype(np.float32)
        # ! to handle different scale, the threshold should be adaptive
        threshold = np.median(dep) * threshold_ratio
        mask, error = detect_depth_occlusion_boundaries(dep, threshold, ksize)
        mask = mask > 0.5
        mask = ~mask  # valid mask
        # ! do a morph operator to remove outliers
        mask_opened = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, ellip_kernel
        )
        mask_opened = mask_opened > 0
        # mask_opened = mask
        dep_valid_masks.append(mask_opened)
        dep_boundary_errors.append(error)
    dep_valid_masks = np.stack(dep_valid_masks, axis=0)
    dep_boundary_errors = np.stack(dep_boundary_errors, axis=0)
    return dep_valid_masks, dep_boundary_errors


def detect_depth_occlusion_boundaries(depth_map, threshold=10, ksize=5):
    error = cv2.Laplacian(depth_map, cv2.CV_64F, ksize=ksize)
    error = np.abs(error)
    _, occlusion_boundaries = cv2.threshold(error, threshold, 255, cv2.THRESH_BINARY)
    return occlusion_boundaries.astype(np.uint8), error


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


def visualize_track(save_fn, tracks, visibility, video_pt, max_viz_cnt=512):
    # video_pt (T,3,H,W)
    # (T,N,2)
    # (T,N)
    video_pt = torch.as_tensor(video_pt)[None]
    tracks = torch.as_tensor(tracks)[None]
    visibility = torch.as_tensor(visibility)[None]
    assert video_pt.ndim == 5 and video_pt.shape[0] == 1
    assert tracks.ndim == 4 and tracks.shape[0] == 1
    assert visibility.ndim == 3 and visibility.shape[0] == 1
    _step = max(1, tracks.shape[2] // max_viz_cnt)
    viz_choice = torch.arange(0, tracks.shape[2], _step)
    vis = Visualizer(
        save_dir=osp.dirname(save_fn),
        linewidth=2,
        draw_invisible=True,  # False
        tracks_leave_trace=4,
    )
    vis.visualize(
        video=video_pt,
        tracks=tracks[:, :, viz_choice, :2],
        visibility=visibility[:, :, viz_choice],
        filename=osp.basename(save_fn),
    )


class Saved2D(nn.Module):
    def __init__(self, ws) -> None:
        super().__init__()
        self.ws = ws

        # this init only load the RGBs and build basic attrs
        self.load_rgb("images")

        # * some old attr, not sure whether useful
        homo_map = torch.from_numpy(get_homo_coordinate_map(H=self.H, W=self.W))
        self.register_gradfree_buffer("homo_map", homo_map.float().detach())
        pixel_int_map = torch.from_numpy(
            np.stack(np.meshgrid(np.arange(self.W), np.arange(self.H)), -1)
        ).int()
        self.register_gradfree_buffer("pixel_int_map", pixel_int_map.detach())

        # * other flags
        self.has_epi = False
        self.has_vos = False
        return

    def register_gradfree_buffer(self, name, buffer):
        buf = torch.as_tensor(buffer).detach().contiguous()
        buf.requires_grad = False
        self.register_buffer(name, buf)
        return

    @torch.no_grad
    def load_rgb(self, rgb_dirname="images"):
        img_dir = osp.join(self.ws, rgb_dirname)
        img_npz = img_dir + ".npz"
        if osp.exists(img_npz):
            images = np.load(img_npz)["images"]  # ! in [0,255]
            images = torch.from_numpy(images).float() / 255.0  # T,H,W,3
            img_names = [f"{i:05d}" for i in range(images.shape[0])]
        elif osp.exists(img_dir):
            img_fns = [
                f
                for f in os.listdir(img_dir)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
            img_fns.sort()
            img_names = [osp.splitext(f)[0] for f in img_fns]
            images = [imageio.imread(osp.join(img_dir, img_fn)) for img_fn in img_fns]
            images = torch.Tensor(np.stack(images)) / 255.0  # T,H,W,3
        else:
            raise ValueError(f"Cannot find images in {img_dir}")
        # assign
        self.frame_names = img_names
        images = images[..., :3]
        self.register_gradfree_buffer("rgb", images.detach())
        return self

    @property
    def T(self):
        return self.rgb.shape[0]

    @property
    def H(self):
        return self.rgb.shape[1]

    @property
    def W(self):
        return self.rgb.shape[2]

    # load more data
    # todo: register the dyn mask on 2D

    def load_epi(self, epi_dirname="epi"):
        epi_dir = osp.join(self.ws, epi_dirname)
        if not osp.exists(epi_dir):
            logging.warning(
                f"Calling 2D EPI loading, but {epi_dir} not found, usually this means use track epi not optical flow epi, so skip!"
            )
            return self
        epipolar_errs = load_epi_error(osp.join(self.ws, epi_dirname))
        epipolar_errs = torch.Tensor(np.stack(epipolar_errs)).float()  # T,H,W
        self.register_gradfree_buffer("epi", epipolar_errs)
        self.has_epi = True
        return self

    @torch.no_grad()
    def load_dep(
        self,
        depth_dirname="depth_metric3d",
        depth_boundary_th=0.1,
        depth_min=1e-3,
        depth_max=1000.0,
        mask_depth_flag=True,
    ):
        assert hasattr(self, "frame_names"), "Load RGB before DEP!"
        dep_dir = osp.join(self.ws, depth_dirname)
        if osp.isdir(dep_dir):
            dep = [
                np.load(osp.join(dep_dir, f"{img_name}.npz"))["dep"]
                for img_name in self.frame_names
            ]
            dep = np.stack(dep)  # T,H,W
        else:
            dep_fn = dep_dir + ".npz"
            assert osp.exists(dep_fn)
            dep = np.load(dep_fn)["dep"]
        if mask_depth_flag:
            dep_mask, _ = laplacian_filter_depth(dep, depth_boundary_th, 5)
            logging.info(f"Dep Boundary Mask {dep_mask.mean()*100:.2f}%")
            dep_mask = dep_mask * (dep > depth_min) * (dep < depth_max)
            dep = np.clip(dep, depth_min, depth_max)
        else:
            dep_mask = np.ones_like(dep) > 0

        # assign
        self.register_gradfree_buffer("dep", torch.Tensor(dep))
        self.register_gradfree_buffer("dep_mask", torch.Tensor(dep_mask).bool())
        return self

    @torch.no_grad()
    def replace_depth(self, dep, dep_mask):
        if hasattr(self, "dep"):
            assert dep.shape == self.dep.shape, f"{dep.shape} vs {self.dep.shape}"
            assert (
                dep_mask.shape == self.dep_mask.shape
            ), f"{dep_mask.shape} vs {self.dep_mask.shape}"
        self.register_gradfree_buffer("dep", dep)
        self.register_gradfree_buffer("dep_mask", dep_mask)
        return self

    @torch.no_grad()
    def recompute_dep_mask(
        self, depth_boundary_th=0.1, depth_min=1e-3, depth_max=1000.0
    ):
        dep = self.dep.cpu().numpy()
        dep_mask, _ = laplacian_filter_depth(dep, depth_boundary_th, 5)
        logging.info(f"Dep Boundary Mask {dep_mask.mean()*100:.2f}%")
        dep_mask = dep_mask * (dep > depth_min) * (dep < depth_max)
        # dep[~dep_mask] = np.inf
        dep = np.clip(dep, depth_min, depth_max)
        self.register_gradfree_buffer("dep", torch.Tensor(dep))
        self.register_gradfree_buffer("dep_mask", torch.Tensor(dep_mask).bool())
        return self

    @torch.no_grad()
    def load_track(self, track_npz_key="spatracker", min_valid_cnt=4):
        assert hasattr(self, "dep_mask"), "Load DEP before TAP!"
        # * load from saved files
        long_track_fns = glob(osp.join(self.ws, f"*{track_npz_key}*.npz"))
        logging.info(f"Loading TAP from {long_track_fns}...")
        assert len(long_track_fns) > 0, "no TAP found!"
        track, track_mask = [], []
        for track_fn in long_track_fns:
            track_data = np.load(track_fn, allow_pickle=True)
            track.append(track_data["tracks"])
            track_mask.append(track_data["visibility"])
        # ! explicitly round to long
        track = torch.from_numpy(np.concatenate(track, 1)).float()  # T,N,2/3
        track[:, :, :2] = track[:, :, :2].long().float()
        track_mask = torch.from_numpy(np.concatenate(track_mask, 1)).bool()  # T,N
        track_mask = track_mask * (track[..., 0] >= 0) * (track[..., 1] >= 0)
        track_mask = track_mask * (track[..., 0] < self.W) * (track[..., 1] < self.H)
        assert track.shape[:2] == track_mask.shape
        assert len(track) == self.T
        # * filter the load tracks
        if min_valid_cnt > 0:
            track, track_mask = filter_track(
                track,
                track_mask,
                self.dep_mask.to(track.device),
                min_valid_cnt=min_valid_cnt,
            )

        # append the tracks
        if hasattr(self, "track"):
            self.track = torch.cat([self.track, track.to(self.track.device)], 1)
            self.track_mask = torch.cat(
                [self.track_mask, track_mask.to(self.track_mask.device)], 1
            )
        else:
            self.register_gradfree_buffer("track", track.float())  # 2D/3D track
        self.register_gradfree_buffer("track_mask", track_mask)
        # ! warning, the track static mask is not saved here
        # assert (
        #     self.track.dtype == torch.long
        # ), "Must use Long type for track! to avoid later roudning error, otherwise the system will fail."

        # re-scale the 3rd depth if the depth is rescaled
        if self.track.shape[-1] == 3 and hasattr(self, "scale_nw"):
            logging.info(f"Also align the 3D track with the depth scale")
            self.track[:, :, 2] = self.track[:, :, 2].clone() * self.scale_nw

        return self

    @torch.no_grad()
    def load_vos(self, vos_dirname="vos_deva/Annotations"):
        vos_dir = osp.join(self.ws, vos_dirname)
        if not osp.exists(vos_dir):
            logging.warning(
                f"Calling 2D VOS loading, but {vos_dir} not found, usually this means the data is not available, so skip!"
            )
            return self
        logging.info(f"loading vos results from {vos_dirname}...")
        id_mask_list = []
        fn_list = sorted(os.listdir(vos_dir))
        for fn in fn_list:
            seg_fn = osp.join(vos_dir, fn)
            seg = imageio.imread(seg_fn)
            id_map = RGB2INST(seg)
            id_mask_list.append(id_map)
        id_mask_list = np.stack(id_mask_list, 0)
        unique_id = np.unique(id_mask_list)
        # remove 0 from unique id, which is unknown
        unique_id = unique_id[unique_id != 0]
        logging.info(
            f"loaded {len(unique_id)} unique ids with {len(id_mask_list)} frames."
        )
        self.register_gradfree_buffer("vos", torch.from_numpy(id_mask_list))
        self.register_gradfree_buffer("vos_id", torch.from_numpy(unique_id))
        self.has_vos = True
        return self

    @torch.no_grad()
    def load_flow(self, flow_dirname="flow_raft"):
        flow_list, flow_mask_list, src_t_list, dst_t_list = [], [], [], []
        flow_ij_to_listind_dict = {}
        flow_dir = osp.join(self.ws, flow_dirname)
        if not osp.exists(flow_dir):
            logging.warning(
                f"Calling 2D Flow loading, but {flow_dir} not found, usually this means the data is not available, so skip!"
            )
            return self
        for fn in tqdm(sorted(os.listdir(flow_dir))):
            if not fn.endswith(".npz"):
                continue
            flow_fn = osp.join(flow_dir, fn)
            flow_data = np.load(flow_fn, allow_pickle=True)
            flow, mask = flow_data["flow"], flow_data["mask"]
            src_t, dst_t = fn[:-4].split("_to_")
            src_t = self.frame_names.index(src_t[:-4])
            dst_t = self.frame_names.index(dst_t[:-4])
            flow_list.append(flow)
            flow_mask_list.append(mask)
            src_t_list.append(src_t)
            dst_t_list.append(dst_t)
            flow_ij_to_listind_dict[(src_t, dst_t)] = len(flow_list) - 1
        flow_list = np.stack(flow_list, 0)
        flow_mask_list = np.stack(flow_mask_list, 0)
        src_t_list = np.array(src_t_list)
        dst_t_list = np.array(dst_t_list)
        self.register_gradfree_buffer("flow", torch.from_numpy(flow_list))
        self.register_gradfree_buffer("flow_mask", torch.from_numpy(flow_mask_list))
        self.register_gradfree_buffer("flow_src_t", torch.from_numpy(src_t_list))
        self.register_gradfree_buffer("flow_dst_t", torch.from_numpy(dst_t_list))
        self.flow_ij_to_listind_dict = flow_ij_to_listind_dict
        return self

    @torch.no_grad()
    def register_2d_identification(self, static_2d_mask, dynamic_2d_mask):
        device = self.rgb.device
        self.register_gradfree_buffer("sta_mask", static_2d_mask.float().to(device))
        self.register_gradfree_buffer("dyn_mask", dynamic_2d_mask.float().to(device))
        logging.info(
            f"Saved2d register 2d identification: {static_2d_mask.sum()} static, {dynamic_2d_mask.sum()} dynamic; Unused: {(~static_2d_mask & ~dynamic_2d_mask).sum()}"
        )
        return

    @torch.no_grad()
    def register_track_indentification(self, static_track_mask, dynamic_track_mask):
        assert hasattr(self, "track"), "Load track before identify!"
        assert len(static_track_mask) == self.track.shape[1]
        device = self.track.device
        self.register_gradfree_buffer(
            "static_track_mask", static_track_mask.bool().to(device)
        )
        self.register_gradfree_buffer(
            "dynamic_track_mask", dynamic_track_mask.bool().to(device)
        )
        logging.warning(
            f"Saved2d register track identification: {static_track_mask.sum()} static, {dynamic_track_mask.sum()} dynamic; Unused: {(~static_track_mask & ~dynamic_track_mask).sum()}"
        )
        return self

    @torch.no_grad()
    def get_mask_by_key(self, key):
        if key == "all":
            return torch.ones_like(self.dep_mask)
        elif key == "static":
            return self.sta_mask
        elif key == "dynamic":
            return self.dyn_mask
        elif key == "dep":
            return self.dep_mask
        else:
            raise ValueError(f"Unknown key={key}")

    @torch.no_grad()
    def rescale_depth(self, dep_scale):
        dep_scale = dep_scale.to(self.dep.device)
        assert len(dep_scale) == self.T, f"dep_scale:{len(dep_scale)} vs T:{self.T}"
        self.dep = self.dep.clone() * dep_scale[:, None, None]
        if self.track.shape[-1] == 3:
            logging.info(f"Also align the 3D track with the depth scale")
            self.track[:, :, 2] = self.track[:, :, 2].clone() * dep_scale[:, None]
        return

    @torch.no_grad()
    def rescale_perframe_depth_from_bundle(self, bundle_pth_fn=None):
        if bundle_pth_fn is None:
            bundle_pth_fn = osp.join(self.ws, "bundle", "bundle.pth")
        bundle_data = torch.load(bundle_pth_fn)
        dep_scale = bundle_data["dep_scale"]
        logging.info(f"Rescale depth with {bundle_pth_fn} and scale={dep_scale}")
        self.rescale_depth(dep_scale)
        return self

    @torch.no_grad()
    def normalize_depth(self, median_depth: float = 1.0):
        # align the median of the depth to 1.0
        if median_depth < 0:
            scale_nw = 1.0
        else:
            world_depth = self.dep.clone()
            world_depth_fg = world_depth[self.dep_mask]
            world_depth_median = torch.median(world_depth_fg)
            scale_nw = (
                median_depth / world_depth_median
            )  # depth_normalized = scale_nw * depth_world
        self.scale_nw = float(scale_nw)
        assert hasattr(self, "dep_mask"), "Load DEP before noramlization!"
        # assert hasattr(self, "track"), "Load TAP before noramlization!"

        self.dep = self.dep.clone() * scale_nw
        if hasattr(self, "track") and self.track.shape[-1] == 3:
            logging.info(f"Also align the 3D track with the depth scale")
            self.track[:, :, 2] = self.track[:, :, 2].clone() * scale_nw
        return self


if __name__ == "__main__":
    src = "../data/debug/C2_N11_S212_s03_T2/"

    s2d = Saved2D(src)
