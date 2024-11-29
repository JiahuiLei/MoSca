# Stand alone warping field with MoSca
import torch
from torch import nn
import torch.nn.functional as F
import logging, time
import sys, os, os.path as osp
from pytorch3d.ops import knn_points
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scaffold_utils.dualquat_helper import Rt2dq, dq2unitdq, dq2Rt
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from gs_utils.gs_optim_helper import prune_optimizer, cat_tensors_to_optimizer

# DQ_EPS = 0.00001
DQ_EPS = 0.001
TOPO_CHUNK_SIZE = 65536 // 2


class MoSca(nn.Module):
    def __init__(
        self,
        node_xyz,  # T,M,3
        node_certain,  # T,M # if valid (visible, inside depth) = 1
        node_grouping=None,  # M # ! warning edges can only be built with in a group, usually in real case, there must be edges across different object, so, use a dummy group here
        node_sigma_logit=None,
        spatial_unit_factor: float = 5.0,  # ! important to tune
        spatial_unit_hard_set=None,  # will overwrite other config of the spa unit
        # Topology
        skinning_k: int = 8,
        topo_dist_top_k: int = 1,
        topo_sample_T: int = 1000,
        topo_th_ratio: float = 8.0,
        skinning_method: str = "dqb",
        sigma_max_ratio: float = 10.0,
        sigma_init_ratio: float = 1.0,  # independent to the max sigma
        break_topo_between_group=True,  # by default don't construct topology edges between two groups
        # mlevel reg
        mlevel_arap_flag=True,
        mlevel_list=[1, 6],
        mlevel_k_list=[16, 8],
        mlevel_w_list=[0.4, 0.3],
        mlevel_detach_nn_flag=True,
        mlevel_detach_self_flag=False,
        # w corr additional config
        w_corr_maintain_sum_flag=False,
        # others
        init_device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        # for better viz
        sort_node=False,
        # Buffer
        t_list=None,
        # min num of nodes
        min_node_num=32,
        max_node_num=16384,
    ):
        # T,N,3; T,N,4
        super().__init__()

        # * xyz
        if node_grouping is None:
            node_grouping = torch.zeros_like(node_certain[0], dtype=torch.int)  # M
        if len(torch.unique(node_grouping)) > 1 and sort_node:
            # sort the node by the grouping
            logging.warning(f"Re-sort the node order!")
            sort_id = torch.argsort(node_grouping)
            node_xyz = node_xyz[:, sort_id]
            node_certain = node_certain[:, sort_id]
            node_grouping = node_grouping[sort_id]

        self._node_xyz = nn.Parameter(node_xyz.to(init_device))
        self.register_buffer(
            "_node_certain", node_certain.to(init_device)
        )  # if certain True
        self.register_buffer("_node_grouping", node_grouping.to(init_device))
        self.register_buffer("unique_grouping", torch.unique(self._node_grouping))

        # * compute topology with grouping
        # first identify the spatial unit
        if spatial_unit_hard_set is None or spatial_unit_hard_set <= 0.0:
            logging.info(
                f"Auto set spatial unit with factor={spatial_unit_factor} from curve median dist"
            )
            spatial_unit = spatial_unit_factor * __identify_spatial_unit_from_curves__(
                self._node_xyz, self._node_certain, K=skinning_k
            )
            logging.info(f"Auto set spatial unit with factor={spatial_unit}")
        else:
            logging.info(f"Hard set spatial unit with factor={spatial_unit_hard_set}")
            spatial_unit = spatial_unit_hard_set
        self.register_buffer("spatial_unit", torch.tensor(spatial_unit))
        self.register_buffer("skinning_k", torch.tensor(skinning_k).long())
        self.register_buffer("topo_dist_top_k", torch.tensor(topo_dist_top_k).long())
        self.register_buffer("topo_sample_T", torch.tensor(topo_sample_T).long())
        self.register_buffer("topo_th_ratio", torch.tensor(topo_th_ratio))
        self.register_buffer("fixed_topology_flag", torch.tensor(False))
        self.set_multi_level(
            mlevel_arap_flag, mlevel_list, mlevel_k_list, mlevel_w_list
        )
        self.register_buffer("mlevel_detach_nn_flag", torch.tensor(mlevel_detach_nn_flag))
        self.register_buffer("mlevel_detach_self_flag", torch.tensor(mlevel_detach_self_flag))
        self.register_buffer("topo_knn_ind", torch.zeros(0).long())
        self.register_buffer("topo_knn_mask", torch.zeros(0).bool())
        self.register_buffer(
            "break_topo_between_group", torch.tensor(break_topo_between_group)
        )
        logging.info(f"mosca break topo between group: {break_topo_between_group}")
        self.update_topology(curve_mask=self._node_certain)

        # * compute optimal frame and init the rotation
        self.compute_rotation_from_xyz()

        # * RBF interpolation
        self.register_buffer("max_sigma", torch.tensor(sigma_max_ratio * spatial_unit))
        self.register_buffer(
            "init_sigma", torch.tensor(sigma_init_ratio * spatial_unit)
        )
        if node_sigma_logit is None:
            node_sigma_logit = self.sig_invact(torch.ones(self.M) * self.init_sigma)
        else:
            assert len(node_sigma_logit) == self.M
            assert node_sigma_logit.ndim <= 2
        if node_sigma_logit.ndim == 1:
            node_sigma_logit = node_sigma_logit[:, None]
        self._node_sigma_logit = nn.Parameter(node_sigma_logit)  # M,1

        # * Skinning config
        # self.blending_method = skinning_method
        self.blending_mehtod_int_dict = {"dqb": 0, "lbs": 1}
        self.register_buffer(
            "blending_method",
            torch.Tensor([self.blending_mehtod_int_dict[skinning_method]]).long(),
        )
        self.set_skinning_method()
        
        self.register_buffer("w_corr_maintain_sum_flag", torch.tensor(w_corr_maintain_sum_flag))

        # * Buffers
        if t_list is None:
            t_list = torch.arange(self.T).long()
            logging.info(f"[t_list] is not provided and set to default")
        else:
            t_list = torch.as_tensor(t_list).long()
            logging.info(f"[t_list] is provided: {t_list[:3]} ... {t_list[-3:]}")
        self.register_buffer("_t_list", t_list)

        self.min_node_num = min_node_num
        self.max_node_num = max_node_num
        assert (
            self.M >= self.min_node_num
        ), f"Node num {self.M} is less than {self.min_node_num}"
        self.to(init_device)
        self.summary()
        return

    def set_skinning_method(self):
        if self.blending_method == 0:
            self.__BLEND_FUNC__ = __DQB_warp__
        elif self.blending_method == 1:
            logging.warning("USE LBS")
            self.__BLEND_FUNC__ = __LBS_warp__
        else:
            raise NotImplementedError(f"Unknown blending")
        return

    @torch.no_grad()
    def compute_rotation_from_xyz(self):
        _node_quat = self.__compute_R_from_xyz__()
        if hasattr(self, "_node_rotation"):
            self._node_rotation.data = _node_quat
        else:
            self._node_rotation = nn.Parameter(_node_quat)
        torch.cuda.empty_cache()
        return

    @property
    def device(self):
        return self._node_xyz.device

    @property
    def M(self):
        return self._node_xyz.shape[1]

    @property
    def T(self):
        return self._node_xyz.shape[0]

    def summary(self):
        logging.info(
            f"MoSca Summary: T={self.T}; M={self.M}; K={self.skinning_k}; spatial-unit={self.spatial_unit}"
        )

        logging.info(
            f"MoSca Summary[1]: T={self.T}; M={self.M}; K={self.skinning_k}; Multi-level={self.mlevel_list}"
        )
        logging.info(
            f"MoSca Summary[2]: curve-dist-K={self.topo_dist_top_k}; spatial-unit={self.spatial_unit}; topo-th-ratio={self.topo_th_ratio}"
        )
        logging.info(
            f"MoSca Summary[3]: SK-method={self.blending_method}; fixed-topology={self.fixed_topology_flag}; sigma-max={self.max_sigma}; sigma-init={self.init_sigma}"
        )
        return

    def sig_act(self, x):
        return self.max_sigma * torch.sigmoid(x)

    def sig_invact(self, x):
        return torch.logit(torch.clamp(x / self.max_sigma, 1e-6, 1.0 - 1e-6))

    @property
    def node_sigma(self):
        return self.sig_act(self._node_sigma_logit)

    @classmethod
    def load_from_ckpt(
        cls,
        ckpt,
        device=torch.device("cuda:0"),
    ):
        logging.info(f"Loading MoSca from ckpt to {device}")
        node_xyz = ckpt["_node_xyz"]
        node_certain = ckpt["_node_certain"]
        node_grouping = ckpt["_node_grouping"]
        node_sigma_logit = ckpt["_node_sigma_logit"]
        skinning_k = ckpt["skinning_k"]
        # ! handle the old version
        if "blending_method" not in ckpt.keys():
            logging.warning("Old ckpt, set blending_method to 0 (DQB)")
            ckpt["blending_method"] = torch.tensor(0)
        if "break_topo_between_group" not in ckpt.keys():
            ckpt["break_topo_between_group"] = torch.tensor(True)
        unique_grouping = torch.unique(node_grouping)
        if "unique_grouping" not in ckpt.keys():
            ckpt["unique_grouping"] = torch.unique(node_grouping)
        else:
            if len(unique_grouping) < len(ckpt["unique_grouping"]):
                # ! this bug is fixed in later version, but to load old ckpt, still handle here
                ckpt["unique_grouping"] = unique_grouping
        if "mlevle_detach_nn_flag" not in ckpt.keys():
            # old ckpt
            ckpt["mlevel_detach_nn_flag"] = torch.tensor(True)
        if "mlevel_detach_self_flag" not in ckpt.keys():
            # old ckpt
            ckpt["mlevel_detach_self_flag"] = torch.tensor(False)
        if "w_corr_maintain_sum_flag" not in ckpt.keys():
            # old ckpt
            ckpt["w_corr_maintain_sum_flag"] = torch.tensor(False)
        
        scf = cls(
            node_xyz=node_xyz,
            node_certain=node_certain,
            node_grouping=node_grouping,
            node_sigma_logit=node_sigma_logit,
            skinning_k=skinning_k,
        )
        scf.load_state_dict(ckpt, strict=True)
        scf.set_skinning_method()
        scf.to(device)
        return scf

    def set_multi_level(
        self,
        mlevel_arap_flag=True,
        mlevel_list=[1, 8],
        mlevel_k_list=[16, 8],
        mlevel_w_list=[0.4, 0.3],
    ):
        self.mlevel_arap_flag = mlevel_arap_flag
        if self.mlevel_arap_flag:
            self.mlevel_list = mlevel_list
            self.mlevel_k_list = mlevel_k_list
            self.mlevel_w_list = mlevel_w_list
            logging.info(
                f"Set MoSca with multi-level arap topo reg with level-list={self.mlevel_list}, k-list={self.mlevel_k_list}, w-list={self.mlevel_w_list}"
            )
        else:
            self.mlevel_list = []  # dummy
        return

    @torch.no_grad()
    def decremental_topology(
        self,
        node_prune_mask,  # if prune, mark as Ture
        verbose=False,
        multilevel_update_flag=True,
    ):
        ind_convert = torch.ones_like(node_prune_mask).long() * -1
        ind_convert[~node_prune_mask] = torch.arange(self.M, device=self.device)

        start_t = time.time()
        assert hasattr(self, "_D_topo")
        assert self._D_topo.shape[1] > self.M, "No need to decrement!"
        old_M = self._D_topo.shape[1]
        logging.info(
            f"Decremental topology from {old_M} to {self.M} (specially handle the knn)"
        )

        # update the D_topo
        self._D_topo = _compute_curve_topo_dist_(
            curve_xyz=self._node_xyz,
            curve_mask=None,
            top_k=self.topo_dist_top_k,
            max_subsample_T=self.topo_sample_T,
        )
        if multilevel_update_flag:
            self.update_multilevel_arap_topo(verbose=verbose)

        # manually update the knn ind and mask, for fixed topology update
        can_change_mask = node_prune_mask[self.topo_knn_ind[~node_prune_mask]]
        new_topo_knn_ind = ind_convert[self.topo_knn_ind[~node_prune_mask]]
        assert (can_change_mask == (new_topo_knn_ind == -1)).all()
        new_topo_knn_mask = self.topo_knn_mask[~node_prune_mask]
        logging.info(
            f"During topo decremental, only {can_change_mask.sum()} slot of knn ind and mask are changed"
        )

        # only change the can change place with a naive for loop
        topo_th = self.topo_th_ratio * self.spatial_unit
        any_change_id = torch.arange(len(can_change_mask))[
            can_change_mask.any(dim=1).cpu()
        ]
        for node_id in tqdm(any_change_id.cpu()):
            d = self._D_topo[node_id].clone()
            change_mask = can_change_mask[node_id]
            unchanged_ind = new_topo_knn_ind[node_id][~change_mask]
            d[unchanged_ind] = d.max()
            new_nn_id = d.topk(self.skinning_k, largest=False).indices
            new_topo_knn_ind[node_id][change_mask] = new_nn_id[: change_mask.sum()]
            new_topo_knn_mask[node_id][change_mask] = d[
                new_nn_id[: change_mask.sum()]
            ] < (topo_th)
        self.topo_knn_ind = new_topo_knn_ind.clone()
        self.topo_knn_mask = new_topo_knn_mask.clone()
        if verbose:
            logging.info(f"Topology updated in {time.time()-start_t:.2f}s")
        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def incremental_topology(
        self,
        verbose=False,
        multilevel_update_flag=True,
    ):
        # ! special func for error grow
        start_t = time.time()
        assert hasattr(self, "_D_topo")
        assert self._D_topo.shape[1] < self.M, "No need to increment!"
        old_M = self._D_topo.shape[1]
        logging.info(
            f"Incremental topology from {old_M} to {self.M} (not changing old {old_M})"
        )

        # update the D_topo
        old_M = len(self._D_topo)
        append_M = self.M - old_M
        if append_M > 0:
            bottom = __query_distance_to_curve__(
                q_curve_xyz=self._node_xyz[:, old_M:],
                b_curve_xyz=self._node_xyz[:, :old_M],
                top_k=self.topo_dist_top_k,
                max_subsample_T=self.topo_sample_T,
            )
            square = __query_distance_to_curve__(
                q_curve_xyz=self._node_xyz[:, old_M:],
                b_curve_xyz=self._node_xyz[:, old_M:],
                top_k=self.topo_dist_top_k,
                max_subsample_T=self.topo_sample_T,
            )
            # all diag elements of square should be huge 1e10
            square = square + torch.eye(append_M).to(square) * 1e10
           
            new_D1 = torch.cat([self._D_topo, bottom.T], 1)
            new_D2 = torch.cat([bottom, square], 1)
            new_D = torch.cat([new_D1, new_D2], 0)
            self._D_topo = new_D
           

        # update the knn and mask
        assert len(self.topo_knn_ind) == old_M
        new_topo_dist, new_topo_knn_ind = __compute_topo_ind_from_dist__(
            new_D[old_M:], self.skinning_k - 1
        )
        new_self_ind = torch.arange(self.M).to(self.device)[old_M:]
        topo_ind = torch.cat([new_self_ind[:, None], new_topo_knn_ind], dim=1)
        topo_th = self.topo_th_ratio * self.spatial_unit
        topo_mask = new_topo_dist < topo_th
        topo_mask = torch.cat([torch.ones_like(topo_mask[:, :1]), topo_mask], 1)
        self.topo_knn_ind = torch.cat([self.topo_knn_ind, topo_ind], 0)
        self.topo_knn_mask = torch.cat([self.topo_knn_mask, topo_mask], 0)

        if multilevel_update_flag:
            self.update_multilevel_arap_topo(verbose=verbose)
        if verbose:
            logging.info(f"Topology updated in {time.time()-start_t:.2f}s")
        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def update_topology(
        self,
        verbose=False,
        curve_mask=None,
        multilevel_update_flag=True,
    ):
        # * do some other checks here
        if len(self._node_grouping.unique()) != len(self.unique_grouping):
            print()
        assert len(self._node_grouping.unique()) == len(
            self.unique_grouping
        ), f"{len(self._node_grouping.unique())} vs {len(self.unique_grouping)}"

        # get eval-train state
        assert self.training, "Topology update must be in training mode"
        start_t = time.time()
        if self.fixed_topology_flag:
            logging.warning(
                "Topology update is disabled due to the fixed_topology_flag, but multi-level arap topo is still updated without recompute D"
            )
            self.update_multilevel_arap_topo(verbose=verbose)
            return

        unique_group_id = torch.unique(self._node_grouping)
        if len(unique_group_id) == 1 or not self.break_topo_between_group:
            self._D_topo = _compute_curve_topo_dist_(
                curve_xyz=self._node_xyz,
                curve_mask=curve_mask,
                top_k=self.topo_dist_top_k,
                max_subsample_T=self.topo_sample_T,
            )
            torch.cuda.empty_cache()
        else:
            assert len(unique_group_id) > 1
            self._D_topo = torch.ones(self.M, self.M).to(self.device) * 1e6
            for gid in tqdm(unique_group_id):
                mask = self._node_grouping == gid
                mask2d = mask[:, None] & mask[None]
                _D = _compute_curve_topo_dist_(
                    curve_xyz=self._node_xyz[:, mask],
                    curve_mask=curve_mask[:, mask] if curve_mask is not None else None,
                    top_k=self.topo_dist_top_k,
                    max_subsample_T=self.topo_sample_T,
                )
                self._D_topo[mask2d] = _D.reshape(-1)

        topo_dist, topo_ind = __compute_topo_ind_from_dist__(
            self._D_topo, self.skinning_k - 1
        )

        self_ind = torch.arange(self.M).to(self.device)
        topo_ind = torch.cat([self_ind[:, None], topo_ind], dim=1)
        topo_th = self.topo_th_ratio * self.spatial_unit
        topo_mask = topo_dist < topo_th
        topo_mask = torch.cat([torch.ones_like(topo_mask[:, :1]), topo_mask], 1)
        self.topo_knn_ind = topo_ind
        self.topo_knn_mask = topo_mask

        if multilevel_update_flag:
            self.update_multilevel_arap_topo(verbose=verbose)
        if verbose:
            logging.info(f"Topology updated in {time.time()-start_t:.2f}s")
        torch.cuda.empty_cache()
        return

    def update_multilevel_arap_topo(self, verbose=False):
        if not self.mlevel_arap_flag:
            return
        topo_th = self.topo_th_ratio * self.spatial_unit
        self.multilevel_arap_edge_list, self.multilevel_arap_dist_list = (
            __compute_multilevel_topo_ind_from_dist__(
                self._D_topo,
                K_list=self.mlevel_k_list,
                subsample_units=[self.spatial_unit * l for l in self.mlevel_list],
                verbose=verbose,
            )
        )
        multilevel_arap_topo_w = []
        for l, w in zip(self.mlevel_list, self.multilevel_arap_dist_list):
            multilevel_arap_topo_w.append(w < topo_th * l)
            if verbose:
                logging.info(
                    f"MultiRes l={l} {multilevel_arap_topo_w[-1].float().mean() * 100.0:.2f}% valid edges"
                )
        self.multilevel_arap_topo_w = multilevel_arap_topo_w
        return

    def warp(
        self,
        attach_node_ind,
        query_xyz,
        query_tid,
        target_tid,
        query_dir=None,
        skinning_w_corr=None,
        dyn_o_flag=False,
    ):
        # query_xyz: (N, 3) in live world frame, time: N,
        # query_dir: (N,3,C), attach_node_ind: N, must specify outside which curve is the nearest, so the topology is decided there
        # note, the query_tid and target_tid can be different for each query

        # * check
        if isinstance(query_tid, int) or query_tid.ndim == 0:
            query_tid = torch.ones_like(query_xyz[:, 0]).long() * query_tid
        N = len(query_tid)
        assert len(query_xyz) == N and query_xyz.shape == (N, 3)
        if query_dir is not None:
            assert query_dir.shape[:2] == (N, 3) and query_dir.ndim == 3
        if isinstance(target_tid, int) or target_tid.ndim == 0:
            target_tid = target_tid * torch.ones_like(query_tid)

        # * identify skinning weights
        sk_ind, sk_w, sk_w_sum, sk_ref_node_xyz, sk_ref_node_quat = (
            self.get_skinning_weights(
                query_xyz=query_xyz,
                query_t=query_tid,
                attach_ind=attach_node_ind,
                skinning_weight_correction=skinning_w_corr,
            )
        )

        # * blending
        sk_dst_node_xyz, sk_dst_node_quat = self.get_async_knns(target_tid, sk_ind)
        if dyn_o_flag:
            dyn_o = torch.clamp(sk_w_sum, min=0.0, max=1.0)
        else:
            dyn_o = torch.ones_like(sk_w_sum) * (1.0 - DQ_EPS)
        ret_xyz, ret_dir = self.__BLEND_FUNC__(
            sk_w=sk_w,
            src_xyz=query_xyz,
            src_R=query_dir,
            sk_src_node_xyz=sk_ref_node_xyz,
            sk_src_node_quat=sk_ref_node_quat,
            sk_dst_node_xyz=sk_dst_node_xyz,
            sk_dst_node_quat=sk_dst_node_quat,
            dyn_o=dyn_o,
        )
        return ret_xyz, ret_dir

    def fast_warp(
        self,
        target_tid,
        # all below are baked
        sk_ind,
        sk_w,
        sk_ref_node_xyz,
        sk_ref_node_quat,
        dyn_o,
        query_xyz,
        query_dir,
    ):
        # query_xyz: (N, 3) in live world frame, time: N,
        # query_dir: (N,3,C), attach_node_ind: N, must specify outside which curve is the nearest, so the topology is decided there
        # note, the query_tid and target_tid can be different for each query
        sk_dst_node_xyz = self._node_xyz[target_tid][sk_ind]
        sk_dst_node_quat = self._node_rotation[target_tid][sk_ind]
        ret_xyz, ret_dir = self.__BLEND_FUNC__(
            sk_w=sk_w,
            src_xyz=query_xyz,
            src_R=query_dir,
            sk_src_node_xyz=sk_ref_node_xyz,
            sk_src_node_quat=sk_ref_node_quat,
            sk_dst_node_xyz=sk_dst_node_xyz,
            sk_dst_node_quat=sk_dst_node_quat,
            dyn_o=dyn_o,
        )
        return ret_xyz, ret_dir

    def get_async_knns(self, t, knn_ind):
        assert t.ndim == 1 and knn_ind.ndim == 2 and len(t) == len(knn_ind)
        # self._node_XXXX[t,knn_ind]
        with torch.no_grad():
            flat_sk_ind = t[:, None] * self.M + knn_ind
        sk_ref_node_xyz = self._node_xyz.reshape(-1, 3)[flat_sk_ind, :]  # N,K,3
        sk_ref_node_quat = self._node_rotation.reshape(-1, 4)[flat_sk_ind, :]  # N,K,4
        return sk_ref_node_xyz, sk_ref_node_quat

    def get_async_knns_buffer(self, buffer, t, knn_ind):
        assert buffer.ndim == 3  # T,M,C
        assert t.ndim == 1 and knn_ind.ndim == 2 and len(t) == len(knn_ind)
        # self._node_XXXX[t,knn_ind]
        with torch.no_grad():
            flat_sk_ind = t[:, None] * self.M + knn_ind
        C = buffer.shape[2]
        ret = buffer.reshape(-1, C)[flat_sk_ind, :]  # N,K,C
        return ret

    def get_skinning_weights(
        self,
        query_xyz,
        query_t,
        attach_ind,
        skinning_weight_correction=None,
    ):
        assert query_xyz.ndim == 2
        sk_ind = self.topo_knn_ind[attach_ind]
        sk_mask = self.topo_knn_mask[attach_ind]

        if isinstance(query_t, int) or query_t.ndim == 0:
            sk_ref_node_xyz = self._node_xyz[query_t][sk_ind]
            sk_ref_node_quat = self._node_rotation[query_t][sk_ind]
        else:
            sk_ref_node_xyz, sk_ref_node_quat = self.get_async_knns(query_t, sk_ind)

        sq_dist_to_sk_node = (query_xyz[:, None, :] - sk_ref_node_xyz) ** 2
        sq_dist_to_sk_node = sq_dist_to_sk_node.sum(-1)  # N,K
        sk_w_un = (
            torch.exp(
                -sq_dist_to_sk_node / (2 * (self.node_sigma.squeeze(-1) ** 2)[sk_ind])
            )
            + 1e-6
        )  # N,K

        sk_w_un = sk_w_un * sk_mask.float()
        if skinning_weight_correction is not None:
            assert len(skinning_weight_correction) == len(query_xyz)
            assert skinning_weight_correction.shape[1] == self.skinning_k
            if self.w_corr_maintain_sum_flag:
                tmp_sk_w_sum = abs(sk_w_un).sum(-1)[:, None] # N,1
                sk_w_un = abs(sk_w_un + skinning_weight_correction)
                new_sk_w_sum = torch.clamp(abs(sk_w_un).sum(-1), min=1e-6)[:, None] # N,1
                factor = tmp_sk_w_sum / new_sk_w_sum
                sk_w_un = sk_w_un * factor
            else:
                sk_w_un = abs(sk_w_un + skinning_weight_correction)
        sk_w_sum = sk_w_un.sum(-1)
        sk_w = sk_w_un / torch.clamp(sk_w_sum, min=1e-6)[:, None]
        return sk_ind, sk_w, sk_w_sum, sk_ref_node_xyz, sk_ref_node_quat

    ###############################################################
    # * Node Control
    ###############################################################

    @torch.no_grad()
    def resample_node(self, resample_factor=1.0, use_mask=False, resample_ind=None):
        old_M = self.M
        if resample_ind is None:
            resample_ind = resample_curve(
                D=_compute_curve_topo_dist_(
                    self._node_xyz,
                    curve_mask=self._node_certain if use_mask else None,
                    top_k=self.topo_dist_top_k,
                    max_subsample_T=self.topo_sample_T,
                ),
                sample_margin=resample_factor * self.spatial_unit,
                mask=self._node_certain,
                verbose=True,
            )
        if len(resample_ind) < self.min_node_num:
            logging.warning(
                f"Resample node num {len(resample_ind)} is less than {self.min_node_num}, skip node resample"
            )
            self.update_topology()
            return torch.arange(self.M).to(resample_ind)
        new_node_xyz = self._node_xyz[:, resample_ind]
        new_node_quat = self._node_rotation[:, resample_ind]
        new_node_sigma_logit = self._node_sigma_logit[resample_ind]
        with torch.no_grad():
            self._node_xyz = nn.Parameter(new_node_xyz)
            self._node_rotation = nn.Parameter(new_node_quat)
            self._node_sigma_logit = nn.Parameter(new_node_sigma_logit)
            self._node_certain = self._node_certain[:, resample_ind]
            self._node_grouping = self._node_grouping[resample_ind]
            if len(self._node_grouping.unique()) < len(self.unique_grouping):
                logging.warning(
                    f"Resample node totally removed some groups {self._node_grouping.unique()} <- {self.unique_grouping}"
                )
                self.unique_grouping = torch.unique(self._node_grouping)

        self.update_topology()
        logging.info(
            f"Resample node from {old_M} to {self.M}, with curve_mask={use_mask}"
        )
        self.summary()
        return resample_ind

    @torch.no_grad()
    def append_nodes_pnt(
        self,
        optimizer,
        new_node_xyz,
        new_node_quat,
        new_tid,
        chunk_size=512,  # ! disable gradually mode, set to a large number to speed up
    ):
        if self.M + len(new_node_xyz) > self.max_node_num:
            logging.warning(
                f"Node num is more than {self.max_node_num}, skip node append"
            )
            return False
        # grow a node only knowing the time to grow it
        start_t = time.time()
        while len(new_node_xyz) > 0:
            # identify the nearest chunk_size nodes to append and remove it from the new_node attrs
            existing_node_xyz = self._node_xyz[new_tid]
            dist_sq, nn_exsiting_ind, _ = knn_points(
                new_node_xyz[None], existing_node_xyz[None], K=1
            )
            dist_sq = dist_sq[0, :, 0]
            nearest_ind = torch.argsort(dist_sq)[:chunk_size]
            nearest_mask = torch.zeros(len(new_node_xyz), dtype=bool)
            nearest_mask[nearest_ind] = True
            working_node_xyz = new_node_xyz[nearest_mask]
            working_node_quat = new_node_quat[nearest_mask]
            new_node_xyz = new_node_xyz[~nearest_mask]
            new_node_quat = new_node_quat[~nearest_mask]
            nn_exsiting_ind = nn_exsiting_ind[0, nearest_ind, 0]
            new_group_id = self._node_grouping[nn_exsiting_ind]
            node_xyz_list, node_quat_list = [], []
            # no need parallel, this is super fast for the loop
            for _t in tqdm(range(self.T)):
                _, nearest_node_ind, _ = knn_points(
                    working_node_xyz[None], self._node_xyz[new_tid, None], K=1
                )
                nearest_node_ind = nearest_node_ind[0, :, 0]
                working_node_xyz_t, working_node_fr_t = self.warp(
                    attach_node_ind=nearest_node_ind,
                    query_xyz=working_node_xyz,
                    query_dir=q2R(working_node_quat),
                    query_tid=new_tid * torch.ones_like(nearest_node_ind),
                    target_tid=_t,
                    dyn_o_flag=True,  # ! important, ensure fare away append does not move dramastically!
                )
                if _t == new_tid:
                    check_error = (working_node_xyz_t - working_node_xyz).norm(dim=-1)
                    assert check_error.max() < 1e-6
                node_xyz_list.append(working_node_xyz_t)
                node_quat_list.append(matrix_to_quaternion(working_node_fr_t))

            to_append_node_xyz = torch.stack(node_xyz_list, 0)
            to_append_node_quat = torch.stack(node_quat_list, 0)
            self.append_nodes_traj(
                optimizer, to_append_node_xyz, to_append_node_quat, new_group_id
            )
            # assert not self.fixed_topology_flag, "Not support fixed_topology_flag"
            # self.update_topology()  # !
            self.incremental_topology()
        logging.info(f"Grow nodes in {time.time()-start_t:.2f}s")
        return True

    @torch.no_grad()
    def append_nodes_traj(
        self,
        optimizer,
        new_node_xyz_traj,
        new_node_q_traj,
        new_group_id,
        new_node_sigma_logit=None,
    ):
        # grow a node when know the full trajectory
        N = new_node_q_traj.shape[1]
        if new_node_sigma_logit is None:
            new_node_sigma_logit = self.sig_invact(
                torch.ones(N, device=self.device) * self.init_sigma
            )
            new_node_sigma_logit = new_node_sigma_logit[:, None].expand(
                -1, self._node_sigma_logit.shape[1]
            )
        d = {
            "node_xyz": new_node_xyz_traj,
            "node_rotation": new_node_q_traj,
            "node_sigma": new_node_sigma_logit,
        }
        spec = {"node_xyz": 1, "node_rotation": 1}
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d, spec)
        self._node_xyz = optimizable_tensors["node_xyz"]
        self._node_rotation = optimizable_tensors["node_rotation"]
        self._node_sigma_logit = optimizable_tensors["node_sigma"]

        # ! always append invalid
        append_valid_mask = torch.zeros_like(new_node_xyz_traj[..., 0]).bool()
        self._node_certain = torch.cat(
            [self._node_certain, append_valid_mask], 1
        )  # T,M'
        self._node_grouping = torch.cat([self._node_grouping, new_group_id], 0)
        return

    @torch.no_grad()
    def remove_nodes(self, optimizer, node_prune_mask):
        if self.M <= self.min_node_num:
            logging.warning(
                f"Node num is less than {self.min_node_num}, skip node prune"
            )
            return
        # node_prune_mask: if remove -> True
        spec = ["node_xyz", "node_rotation", "node_sigma"]
        optimizable_tensors = prune_optimizer(
            optimizer,
            ~node_prune_mask,
            specific_names=spec,
        )
        self._node_xyz = optimizable_tensors["node_xyz"]
        self._node_rotation = optimizable_tensors["node_rotation"]
        self._node_sigma_logit = optimizable_tensors["node_sigma"]
        logging.info(
            f"Node prune: [-{(node_prune_mask).sum()}]; now has {self.M} nodes"
        )
        self._node_certain = self._node_certain[:, ~node_prune_mask]
        self._node_grouping = self._node_grouping[~node_prune_mask]
        self.decremental_topology(node_prune_mask, verbose=True)
        return

    ###############################################################
    # * reg
    ###############################################################

    def compute_vel_acc_loss(
        self, tids=None, detach_mask=None, reduce_type="mean", square=False
    ):
        if tids is None:
            tids = torch.arange(self.T).to(self.device)
        assert tids.max() <= self.T - 1
        xyz = self._node_xyz[tids]
        R_wi = q2R(self._node_rotation[tids])
        if detach_mask is not None:
            detach_mask = detach_mask.float()[:, None, None]
            xyz = xyz.detach() * detach_mask + xyz * (1 - detach_mask)
            R_wi = (
                R_wi.detach() * detach_mask[..., None]
                + R_wi * (1 - detach_mask)[..., None]
            )
        xyz_vel, ang_vel, xyz_acc, ang_acc = compute_vel_acc(xyz, R_wi)
        if square:
            xyz_vel, ang_vel, xyz_acc, ang_acc = (
                xyz_vel**2,
                ang_vel**2,
                xyz_acc**2,
                ang_acc**2,
            )
        if reduce_type == "mean":
            loss_p_vel, loss_q_vel = xyz_vel.mean(), ang_vel.mean()
            loss_p_acc, loss_q_acc = xyz_acc.mean(), ang_acc.mean()
        elif reduce_type == "sum":
            loss_p_vel, loss_q_vel = xyz_vel.sum(), ang_vel.sum()
            loss_p_acc, loss_q_acc = xyz_acc.sum(), ang_acc.sum()
        else:
            raise NotImplementedError()
        return loss_p_vel, loss_q_vel, loss_p_acc, loss_q_acc

    def compute_arap_loss(
        self,
        tids=None,
        temporal_diff_weight=[0.75, 0.25],
        temporal_diff_shift=[1, 4],
        # * used for only change the latest append frame during appending loop
        detach_tids_mask=None,
        reduce_type="mean",
        square=False,
    ):
        assert len(temporal_diff_weight) == len(temporal_diff_shift)
        if tids is None:
            tids = torch.arange(self.T).to(self.device)
        assert tids.max() <= self.T - 1
        xyz = self._node_xyz[tids]
        R_wi = q2R(self._node_rotation[tids])
        topo_ind = self.topo_knn_ind
        topo_w = self.topo_knn_mask.float()  # N,K, binary mask
        topo_w = topo_w / (topo_w.sum(dim=-1, keepdim=True) + 1e-6)  # normalize
        local_coord = get_local_coord(xyz, topo_ind, R_wi)

        if detach_tids_mask is not None:
            detach_tids_mask = detach_tids_mask.float()
            local_coord = (
                local_coord.detach() * detach_tids_mask[:, None, None, None]
                + local_coord * (1 - detach_tids_mask)[:, None, None, None]
            )
        loss_coord, loss_len, _, _ = compute_arap(
            local_coord,
            topo_w,
            temporal_diff_shift,
            temporal_diff_weight,
            reduce=reduce_type,
            square=square,
        )

        # topo: speed up this
        if self.mlevel_arap_flag:
            for l in range(len(self.multilevel_arap_edge_list)):
                # ! in this case, self is from the larger set
                _local_coord = get_local_coord(
                    xyz,
                    self.multilevel_arap_edge_list[l][:, 1:],
                    R_wi,
                    self_ind=self.multilevel_arap_edge_list[l][:, :1],
                    detach_nn=self.mlevel_detach_nn_flag.item(),
                    detach_self=self.mlevel_detach_self_flag.item(),
                )  # T,N,1,3

                _loss_coord, _loss_len, _, _ = compute_arap(
                    _local_coord,
                    self.multilevel_arap_topo_w[l][:, None],
                    temporal_diff_shift,
                    temporal_diff_weight,
                    reduce=reduce_type,
                    square=square,
                )
                loss_coord = loss_coord + _loss_coord * self.mlevel_w_list[l]
                loss_len = loss_len + _loss_len * self.mlevel_w_list[l]

        return loss_coord, loss_len

    def get_optimizable_list(
        self,
        lr_np=0.0001,
        lr_nq=0.0001,
        lr_nsig=0.00001,
    ):
        ret = []
        if lr_np is not None:
            ret.append({"params": [self._node_xyz], "lr": lr_np, "name": "node_xyz"})
        if lr_nq is not None:
            ret.append(
                {"params": [self._node_rotation], "lr": lr_nq, "name": "node_rotation"}
            )
        if lr_nsig is not None:
            ret.append(
                {
                    "params": [self._node_sigma_logit],
                    "lr": lr_nsig,
                    "name": "node_sigma",
                }
            )
        return ret

    @torch.no_grad()
    def mask_xyz_grad(self, mask):
        # for init stage maintain the observed xyz unchanged
        assert mask.shape == self._node_xyz.shape[:2]
        assert self._node_xyz.grad is not None
        mask = mask.to(self._node_xyz)
        self._node_xyz.grad = self._node_xyz.grad * mask[..., None]
        return

    ##################################################
    # warp helper
    ##################################################
    @torch.no_grad()
    def identify_nearest_node_id(self, query_xyz, query_tid, query_group_id=None):
        # ! naive for loop, actually can do batchwise parallel of time
        if isinstance(query_tid, int) or query_tid.ndim == 0:
            query_tid = torch.ones_like(query_xyz[:, 0]).int() * query_tid
        N = len(query_tid)
        assert query_tid.ndim == 1
        assert len(query_xyz) == N and query_xyz.shape == (N, 3)
        ret_id = -torch.ones_like(query_tid).int()
        for t in torch.unique(query_tid):
            mask = query_tid == t
            if query_group_id is not None:
                assert len(query_group_id) == len(query_tid)
                # ! there is a possibility that some small part is removed when resample the nodes, but if you still want to query the pixel with a group id that never exsit in the current scaffold, will fall back to non-grouping mode, i.e. find the nearest node in the whole scaffold
                group_id_this_time = query_group_id[mask]
                unqiue_query_id = torch.unique(group_id_this_time)
                nearest_node_ind = -torch.ones(mask.sum()).to(ret_id)
                # for each group do saperately
                _node_all_id = torch.arange(self.M).to(self.device)
                for gid in unqiue_query_id:
                    mask_gid = group_id_this_time == gid
                    if mask_gid.any():
                        node_mask = self._node_grouping == gid
                        if not node_mask.any():
                            # logging.warning(
                            #     f"Group {gid} not found in scaffold, fallback to non-grouping mode"
                            # )
                            node_mask = torch.ones_like(self._node_grouping).bool()
                        _, nearest_node_ind_gid, _ = knn_points(
                            query_xyz[mask][mask_gid][None],
                            self._node_xyz[t][node_mask][None],
                            K=1,
                        )
                        nearest_node_ind_gid = nearest_node_ind_gid[0, :, 0]
                        nearest_node_ind[mask_gid] = _node_all_id[node_mask][
                            nearest_node_ind_gid
                        ].int()
                assert (nearest_node_ind >= 0).all(), f"{(nearest_node_ind<0).sum()}"
                ret_id[mask] = nearest_node_ind.int()
            else:
                _, nearest_node_ind, _ = knn_points(
                    query_xyz[mask][None], self._node_xyz[t, None], K=1
                )
                ret_id[mask] = nearest_node_ind[0, :, 0].int()
        assert (ret_id >= 0).all()
        return ret_id

    @torch.no_grad()
    def __compute_R_from_xyz__(self):
        init_node_R = torch.eye(3).to(self.device)[None].repeat(self.M, 1, 1)
        new_R_list = [init_node_R]
        nn_ind, nn_mask = self.topo_knn_ind, self.topo_knn_mask
        for new_tid in tqdm(range(1, self.T)):
            # ! do to the first frame!

            src_xyz, dst_xyz = self._node_xyz[new_tid - 1], self._node_xyz[new_tid]

            # DEBUG: seems the old version does not normalize the centroid, may affect??
            # DEBUG: the old version has bug, seems the w is not used anymore ...

            src_p = src_xyz[nn_ind] - src_xyz[:, None]
            dst_q = dst_xyz[nn_ind] - dst_xyz[:, None]
            w = nn_mask
            w = w / w.sum(dim=-1, keepdim=True)  # M,K

            src_centroid = (src_p * w[..., None]).sum(1)
            dst_centroid = (dst_q * w[..., None]).sum(1)
            src_p = src_p - src_centroid[:, None]
            dst_q = dst_q - dst_centroid[:, None]

            W = torch.einsum("nki,nkj->nkij", src_p, dst_q)

            W = W * w[:, :, None, None]
            W = W.sum(1)

            U, s, V = torch.svd(W.double())
            # ! warning, torch's svd has W = U @ torch.diag(s) @ (V.T)
            # U, s, V = U.float(), s.float(), V.float()
            # R_star = V @ (U.T) # ! handling flipping
            R_tmp = torch.einsum("nij,nkj->nik", V, U)
            det = torch.det(R_tmp)
            dia = torch.ones(len(det), 3).to(det)
            dia[:, -1] = det
            Sigma = torch.diag_embed(dia)
            V = torch.einsum("nij,njk->nik", V, Sigma)
            R_star = torch.einsum("nij,nkj->nik", V, U)
            # dst = R_star @ src
            next_R = torch.einsum(
                "nij,njk->nik", R_star.double(), new_R_list[-1].double()
            )
            new_R_list.append(next_R.float())
        new_R_list = torch.stack(new_R_list, dim=0)
        new_q_list = matrix_to_quaternion(new_R_list)
        return new_q_list

    ##################################################
    # Time densify
    ##################################################

    @torch.no_grad()
    def resample_time(self, new_tids, new_node_certain, mode="linear"):
        assert new_tids.max() <= self._t_list.max(), "no extrapolate"
        assert new_tids.min() >= self._t_list.min(), "no extrapolate"
        new_xyz_list, new_quat_list = [], []
        for new_t in tqdm(new_tids):
            left_t = self._t_list[self._t_list <= new_t].max()
            left_ind = (self._t_list == left_t).float().argmax()
            assert left_ind >= 0 and left_ind < self.T
            l_xyz, l_quat = self._node_xyz[left_ind], self._node_rotation[left_ind]
            if left_t == new_t:
                new_xyz_list.append(l_xyz)
                new_quat_list.append(l_quat)
                continue
            right_t = self._t_list[self._t_list >= new_t].min()
            assert left_t < new_t < right_t
            right_ind = (self._t_list == right_t).float().argmax()
            assert right_ind >= 0 and right_ind < self.T
            r_xyz, r_quat = self._node_xyz[right_ind], self._node_rotation[right_ind]
            if right_t == new_t:
                new_xyz_list.append(r_xyz)
                new_quat_list.append(r_quat)
                continue
            # print(left_t, right_t)
            # print(left_ind, right_ind)
            l_w = float(right_t - new_t) / float(right_t - left_t)
            r_w = float(new_t - left_t) / float(right_t - left_t)

            if mode == "dq":
                l_dq = Rt2dq(q2R(l_quat), l_xyz)
                r_dq = Rt2dq(q2R(r_quat), r_xyz)
                dq = l_dq * l_w + r_dq * r_w
                dq = dq2unitdq(dq)
                new_R, new_xyz = dq2Rt(dq)
                new_quat = matrix_to_quaternion(new_R)
            elif mode == "linear":
                new_xyz = l_xyz * l_w + r_xyz * r_w
                new_quat = l_quat * l_w + r_quat * r_w
            else:
                raise NotImplementedError()

            new_xyz_list.append(new_xyz)
            new_quat_list.append(new_quat)

        new_xyz_list = torch.stack(new_xyz_list, 0)
        new_quat_list = torch.stack(new_quat_list, 0)

        self._node_xyz.data = new_xyz_list
        self._node_rotation.data = new_quat_list
        assert new_node_certain.shape[1] == self.M and len(new_tids) == len(
            new_node_certain
        )
        self._node_certain.data = new_node_certain

        # update t_list
        remap = [torch.where(new_tids == t)[0][0].item() for t in self._t_list.cpu()]
        self._t_list = new_tids.to(self._t_list)
        return remap  # new index in the new list


def __compute_delta_Rt_ji__(R_wi, t_wi, R_wj, t_wj):
    # R: N,K,3,3; t: N,K,3
    # the stored node R,t are R_wi, t_wi
    # p_t=i_world = R_wi @ p_local + t_wi
    # p_local = R_wi.T @ (p_t=i_world - t_wi)
    # p_t=j_world = R_wj @ p_local + t_wj
    # p_t=j_world = R_wj @ R_wi.T @ (p_t=i_world - t_wi) + t_wj
    # p_t=j_world = (R_wj @ R_wi.T) @ p_t=i_world + t_wj - (R_wj @ R_wi.T) @ t_wi
    assert R_wi.ndim == 4 and R_wi.shape[2:] == (3, 3)
    assert t_wi.ndim == 3 and t_wi.shape[2] == 3
    assert R_wj.ndim == 4 and R_wj.shape[2:] == (3, 3)
    assert t_wj.ndim == 3 and t_wj.shape[2] == 3

    R_ji = torch.einsum("nsij,nskj->nsik", R_wj, R_wi)
    t_ji = t_wj - torch.einsum("nsij,nsj->nsi", R_ji, t_wi)
    return R_ji, t_ji


def __DQB_warp__(
    sk_w,
    src_xyz,
    sk_src_node_xyz,
    sk_src_node_quat,
    sk_dst_node_xyz,
    sk_dst_node_quat,
    dyn_o,
    src_R=None,
):
    sk_R_tq, sk_t_tq = __compute_delta_Rt_ji__(
        R_wj=q2R(sk_dst_node_quat),
        t_wj=sk_dst_node_xyz,
        R_wi=q2R(sk_src_node_quat),
        t_wi=sk_src_node_xyz,
    )
    sk_dq_tq = Rt2dq(sk_R_tq, sk_t_tq)  # N,K,8
    # * Dual Quaternion skinning
    dq = torch.einsum("nki,nk->ni", sk_dq_tq, sk_w)  # N,8
    # use dyn mask to blend a unit dq into
    unit_dq = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0]).to(sk_w)[None].expand(len(dq), -1)
    dq = dq * dyn_o[:, None] + unit_dq * (1 - dyn_o)[:, None]
    with torch.no_grad():
        assert dq.max() < 1e6, f"{dq.max()}"
    dq = dq2unitdq(dq)
    R_tq, t_tq = dq2Rt(dq)  # N,3,3; N,3
    # * apply the transformation to the leaf
    mu_dst = torch.einsum("nij,nj->ni", R_tq, src_xyz) + t_tq
    if src_R is not None:
        fr_dst = torch.einsum("nij,njk->nik", R_tq, src_R)
    else:
        fr_dst = None
    return mu_dst, fr_dst


def __LBS_warp__(
    sk_w,
    src_xyz,
    sk_src_node_xyz,
    sk_src_node_quat,
    sk_dst_node_xyz,
    sk_dst_node_quat,
    dyn_o,
    src_R=None,
):
    sk_R_tq, sk_t_tq = __compute_delta_Rt_ji__(
        R_wj=q2R(sk_dst_node_quat),
        t_wj=sk_dst_node_xyz,
        R_wi=q2R(sk_src_node_quat),
        t_wi=sk_src_node_xyz,
    )
    sk_quat_tq = matrix_to_quaternion(sk_R_tq)
    quat_tq = torch.einsum("nki,nk->ni", sk_quat_tq, sk_w)  # N,4
    t_tq = torch.einsum("nki,nk->ni", sk_t_tq, sk_w)  # N,3
    q_unit = torch.Tensor([1, 0, 0, 0]).to(sk_w)[None]
    quat_tq = quat_tq * dyn_o[:, None] + (1 - dyn_o)[:, None] * q_unit
    t_tq = t_tq * dyn_o[:, None]
    R_tq = q2R(quat_tq)
    mu_dst = torch.einsum("nij,nj->ni", R_tq, src_xyz) + t_tq
    if src_R is not None:
        fr_dst = torch.einsum("nij,njk->nik", R_tq, src_R)
    else:
        fr_dst = None
    # logging.warning("Should NOT use LBS!")
    return mu_dst, fr_dst


##################################################################################
##################################################################################
def robust_curve_dist_kernel(src, dst, src_m, dst_m, top_k: int):
    # src, dst: T,N,C
    # src_m, dst_m: T,N bool
    _d = (src - dst).norm(dim=-1)
    T = len(src)
    _m = src_m.bool() & dst_m.bool()  # T, chunk
    _d = _d * _m
    _m_cnt = _m.sum(dim=0)
    # shrink the top-k ind if the visible count is less than T
    top_k_select_id = torch.ceil(top_k * _m_cnt / T).long() - 1  # - 1 is for 0-index
    top_k_select_id = torch.clamp(top_k_select_id, 0, top_k - 1)
    top_k_value, _ = torch.topk(_d, k=top_k, dim=0, largest=True)  # K,Chunk
    # * because top_k always < T, so the selected top_k value should always be valid!
    assert top_k_select_id.max() < top_k
    d = torch.gather(top_k_value, 0, top_k_select_id[None]).squeeze(0)  # Chunk
    return d


@torch.no_grad()
def __query_distance_to_curve__(
    q_curve_xyz,
    b_curve_xyz,
    q_mask=None,
    b_mask=None,
    max_subsample_T=80,
    top_k=8,
    chunk=TOPO_CHUNK_SIZE,
):
    # q_curve_xyz: T,N,3; b_curve_xyz: T,M,3, usually N << M
    # return: N,M

    T, N, _ = q_curve_xyz.shape
    M = b_curve_xyz.shape[1]
    assert q_curve_xyz.shape[0] == b_curve_xyz.shape[0]

    if q_mask is None:
        q_mask = torch.ones(T, N).bool().to(q_curve_xyz.device)
    else:
        q_mask = q_mask.bool().to(q_curve_xyz.device)
    if b_mask is None:
        b_mask = torch.ones(T, M).bool().to(b_curve_xyz.device)
    else:
        b_mask = b_mask.bool().to(b_curve_xyz.device)

    if T > max_subsample_T:
        t_choice = torch.randperm(T)[:max_subsample_T]
        t_choice = torch.sort(t_choice)[0]

    q_curve_xyz = q_curve_xyz[t_choice] if T > max_subsample_T else q_curve_xyz
    q_mask = q_mask[t_choice] if T > max_subsample_T else q_mask
    b_curve_xyz = b_curve_xyz[t_choice] if T > max_subsample_T else b_curve_xyz
    b_mask = b_mask[t_choice] if T > max_subsample_T else b_mask
    T = len(q_curve_xyz)

    cur = 0
    ret = torch.zeros(0).to(q_curve_xyz)

    # not triangle, but all pairs
    all_pair_ind = torch.meshgrid(torch.arange(N), torch.arange(M))
    all_pair_ind = torch.stack(all_pair_ind, -1)  # N,M,2, the first is the N index
    all_pair_ind = all_pair_ind.reshape(-1, 2)

    while cur < len(all_pair_ind):
        ind = all_pair_ind[cur : cur + chunk]
        src = q_curve_xyz[:, ind[:, 0]]  # T,Chunk,3
        dst = b_curve_xyz[:, ind[:, 1]]
        src_m = q_mask[:, ind[:, 0]]  # T, chunk
        dst_m = b_mask[:, ind[:, 1]]  # T, chunk
        d = robust_curve_dist_kernel(src, dst, src_m, dst_m, top_k)
        ret = torch.cat([ret, d])
        cur = cur + chunk
    ret = ret.reshape(N, M)
    return ret


@torch.no_grad()
def _compute_curve_topo_dist_(
    curve_xyz,
    curve_mask=None,
    max_subsample_T=80,
    top_k=8,
    chunk=65536,
):
    # * this  function have to handle size N~10k-30k, T<max_subsample_T
    # curve_xyz: T,N,3

    T, N = curve_xyz.shape[:2]
    if curve_mask is None:
        curve_mask = torch.ones(T, N).bool().to(curve_xyz.device)
    else:
        curve_mask = curve_mask.bool().to(curve_xyz.device)

    if T > max_subsample_T:
        t_choice = torch.randperm(T)[:max_subsample_T]
        t_choice = torch.sort(t_choice)[0]

    xyz = curve_xyz[t_choice] if T > max_subsample_T else curve_xyz
    mask = curve_mask[t_choice] if T > max_subsample_T else curve_mask

    T = len(xyz)

    # prepare the ind pair to compute the dist (half pairs because of symmetric)
    ind_pair = torch.triu_indices(N, N).permute(1, 0)  # N,2
    P = len(ind_pair)
    cur = 0
    flat_D = torch.zeros(0).to(xyz)
    while cur < P:
        ind = ind_pair[cur : cur + chunk]

        src = xyz[:, ind[:, 0]]  # T,Chunk,3
        dst = xyz[:, ind[:, 1]]  # T,Chunk,3
        src_m = mask[:, ind[:, 0]]  # T, chunk
        dst_m = mask[:, ind[:, 1]]  # T, chunk

        d = robust_curve_dist_kernel(src, dst, src_m, dst_m, top_k)

        flat_D = torch.cat([flat_D, d])
        cur = cur + chunk
    # convert the list to upper and lower triangle
    _d = torch.zeros(N, N).to(xyz)
    _d[ind_pair[:, 0], ind_pair[:, 1]] = flat_D
    _d[ind_pair[:, 1], ind_pair[:, 0]] = flat_D
    # make distance that is zero to be Huge
    # ! set distance to self as inf as well, outside this will handle skinning to self
    invalid_mask = _d == 0
    _d[invalid_mask] = 1e10
    return _d


##################################################################################
##################################################################################


@torch.no_grad()
def __compute_topo_ind_from_dist__(dist, K):
    knn_dist, knn_ind = torch.topk(dist, K, dim=-1, largest=False)
    return knn_dist, knn_ind


@torch.no_grad()
def __compute_multilevel_topo_ind_from_dist__(
    dist, K_list: list, subsample_units: list, shrink_level=False, verbose=False
):
    torch.cuda.empty_cache()
    assert (
        not shrink_level
    ), "Warning, shrink the level will makes the multi-level arap not helping!, shoudl always do dense!"
    # dist: N,N, K_list and subsample_unit list are list of knn and subsample units
    N, _ = dist.shape
    current_set = torch.arange(N, device=dist.device)

    edge_list, dist_list = [], []
    for k, unit in zip(K_list, subsample_units):
        torch.cuda.empty_cache()
        if not shrink_level:
            # everytime the source coordinate is the original one
            current_set = torch.arange(N, device=dist.device)
        # subsample the dist curve by the units
        assert current_set.max() < N and current_set.min() >= 0
        cur_D = dist[current_set][:, current_set]
        resample_ind = resample_curve(cur_D, unit, mask=None, verbose=verbose)
        if len(resample_ind) < 1:
            logging.info("No resampled nodes, early stop!")
            break
        assert resample_ind.max() < len(current_set) and resample_ind.min() >= 0
        sub_D = cur_D[
            :, resample_ind
        ]  # ! before convert to global ind, first subsample the sub_D
        resample_ind = current_set[resample_ind].clone()  # ! in original N len

        # for all previous set curve, find the K nearest subset curve
        nearest_ind = torch.topk(
            sub_D, min(k, len(resample_ind)), dim=1, largest=False
        ).indices
        nearest_ind = resample_ind[nearest_ind]  # ! in original N len
        src_ind = current_set[:, None].expand(-1, min(k, len(resample_ind)))
        edge = torch.stack([src_ind, nearest_ind], dim=-1).reshape(-1, 2)  # N,k,2
        assert edge.max() < N and edge.min() >= 0
        dist_list.append(dist[edge[..., 0], edge[..., 1]])  # N,k
        edge_list.append(edge)

        # save the subset as current set
        current_set = resample_ind
        if verbose:
            logging.info(
                f"level {len(edge_list)} with margin={unit:.4f} k={k} |Set|={len(current_set)} |E|={len(dist_list[-1])}"
            )
    torch.cuda.empty_cache()
    return edge_list, dist_list


@torch.no_grad()
def resample_curve(D, sample_margin, mask=None, verbose=False):
    N = D.shape[0]
    assert D.shape == (N, N)
    if mask is None:
        rank_inds = torch.randperm(N)
    else:
        rank_inds = torch.argsort(mask.sum(0), descending=True)
    sampled_inds = rank_inds[:1]
    # for ind in tqdm(rank_inds[1:]):
    for ind in rank_inds[1:]:
        if D[ind][sampled_inds].min() > sample_margin:
            sampled_inds = torch.cat([sampled_inds, ind[None]])
    # sort sampled_inds
    sampled_inds = torch.sort(sampled_inds).values
    if verbose:
        logging.info(
            f"SCF Resample with th={sample_margin:.4f} N={len(sampled_inds)} out of {N} ({len(sampled_inds)/N * 100.0:.2f}%)"
        )
    assert sampled_inds.max() < N and sampled_inds.min() >= 0
    return sampled_inds


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


def compute_vel_acc(xyz, R_wi):
    xyz_vel = (xyz[1:] - xyz[:-1]).norm(dim=-1)
    xyz_acc = (xyz[2:] - 2 * xyz[1:-1] + xyz[:-2]).norm(dim=-1)

    delta_R = torch.einsum("tnij,tnkj->tnik", R_wi[1:], R_wi[:-1])
    ang_vel = matrix_to_axis_angle(delta_R).norm(dim=-1)
    ang_acc_mag = abs(ang_vel[1:] - ang_vel[:-1])
    return xyz_vel, ang_vel, xyz_acc, ang_acc_mag


def compute_arap(
    local_coord,  # T,N,K,3
    topo_w,  # N,K
    temporal_diff_shift,
    temporal_diff_weight,
    reduce="mean",
    square=False,
):
    local_coord_len = local_coord.norm(dim=-1, p=2)  # T,N,K
    # the coordinate should be similar
    # the metric should be similar
    loss_coord, loss_len = torch.tensor(0.0).to(local_coord), torch.tensor(0.0).to(
        local_coord
    )
    for shift, _w in zip(temporal_diff_shift, temporal_diff_weight):
        diff_coord = (local_coord[:-shift] - local_coord[shift:]).norm(dim=-1)
        if len(diff_coord) < 1:
            continue
        diff_len = (local_coord_len[:-shift] - local_coord_len[shift:]).abs()
        if square:
            logging.warning(
                "Use square loss, but initial exp shows non-square loss is better!"
            )
            diff_coord = diff_coord**2
            diff_len = diff_len**2
        diff_coord = (diff_coord * topo_w[None]).sum(-1)
        diff_len = (diff_len * topo_w[None]).sum(-1)
        if reduce == "sum":
            loss_coord = loss_coord + _w * diff_coord.sum()
            loss_len = loss_len + _w * diff_len.sum()
        elif reduce == "mean":
            loss_coord = loss_coord + _w * diff_coord.mean()
            loss_len = loss_len + _w * diff_len.mean()
        else:
            raise NotImplementedError()
    loss_coord_global = (local_coord.std(0) * topo_w[..., None]).sum()
    loss_len_global = (local_coord_len.std(0) * topo_w).sum()
    assert not torch.isnan(loss_coord) and not torch.isnan(loss_len)
    return loss_coord, loss_len, loss_coord_global, loss_len_global


def get_local_coord(
    xyz, topo_ind, R_wi, self_ind=None, detach_self=False, detach_nn=False
):
    assert not (detach_self and detach_nn), "detach_self and detach_nn are exclusive"
    # * self will be expressed in nn coordinate frame
    # xyz: T,N,3; topo_ind: N,K; R_wi: T,N,3,3
    nn_xyz = xyz[:, topo_ind, :]
    nn_R_wi = R_wi[:, topo_ind, :]
    if self_ind is None:
        self_xyz = xyz[:, :, None]  # T,N,1,3
    else:
        assert self_ind.shape == topo_ind.shape
        self_xyz = xyz[:, self_ind, :]
    if detach_self:
        self_xyz = self_xyz.detach()
    if detach_nn:
        nn_xyz = nn_xyz.detach()
    local_coord = torch.einsum(
        "tnkji,tnkj->tnki", nn_R_wi, self_xyz - nn_xyz
    )  # T,N,K,3
    return local_coord


@torch.no_grad()
def __identify_spatial_unit_from_curves__(xyz, mask, K=10):
    T = len(xyz)
    dist_list = []
    for t in tqdm(range(T)):
        _p = xyz[t][mask[t]]
        dist_sq, _, _ = knn_points(_p[None], _p[None], K=min(K, mask[t].sum()))
        if dist_sq.shape[1] == 0:
            continue
        dist_list.append(dist_sq.squeeze(0).median())
    dist_list = torch.tensor(dist_list)
    dist_list = torch.clamp(dist_list, 1e-6, 1e6)
    unit = torch.sqrt(dist_list.median())
    assert torch.isnan(unit) == False, f"{dist_list}"
    return float(unit)


if __name__ == "__main__":
    # test dummy init
    xyz = torch.zeros(0, 0, 3)
    quat = torch.zeros(0, 0, 4)
