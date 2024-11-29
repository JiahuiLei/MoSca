# Jiahui Lei 2024.Jan.31 Dual Quaternion Skinning Helper
import torch
import numpy as np
from torch.nn import functional as F
from pytorch3d.transforms import (
    # quaternion_multiply, # ! for unknown reason, the pt3d multiplication will get the results wrong with a negative sign!!! use our own implementation
    quaternion_invert,  # the conjugate of a unit quaternion is its inverse
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
import logging


def quaternion_multiply(p, q):
    assert p.shape[-1] == 4 and q.shape[-1] == 4
    ps, pv = p[..., :1], p[..., 1:]
    qs, qv = q[..., :1], q[..., 1:]
    s = ps * qs - (pv * qv).sum(dim=-1, keepdim=True)
    v = ps * qv + qs * pv + torch.cross(pv, qv, dim=-1)
    ret = torch.cat([s, v], dim=-1)
    return ret


def Rt2dq(R, t):
    # convert R, t to dual quaternion
    # R: [...,3,3], t: [...,3]
    # return: [...,8] p + e*q, p is the rotation quat, q is the translation quat
    # ! the first 4 elements of dq is the rotation quaternion (real), the last 4 elements is the translation quaternion (dual)
    assert R.shape[-2:] == (3, 3) and t.shape[-1] == 3
    p = matrix_to_quaternion(R.clone())
    t = torch.cat([torch.zeros_like(t[..., :1]), t.clone()], dim=-1)
    q = 0.5 * quaternion_multiply(t, p)
    dq = torch.cat([p, q], dim=-1)
    # with torch.no_grad():
    #     assert ((dq[..., :4] * dq[..., 4:]).sum(-1).abs() < 1e-6).all()
    return dq


def dq2dualnorm(dq):
    # input dq: [...,8], return norm in dual number [..., 2]
    assert dq.shape[-1] == 8
    real = dq[..., :4].norm(dim=-1, p=2)
    denom = torch.clamp(real, min=1e-8)
    dual = (dq[..., :4] * dq[..., 4:]).sum(-1) / denom
    ret = torch.stack([real, dual], dim=-1)
    return ret  # is a dual number


def dual_inverse(x):
    # input x: [...,2], return 1/x [...,2]
    assert x.shape[-1] == 2
    assert x[..., 0].min() > 0, "real part should be positive"
    ret = x.clone()
    ret[..., 0] = 1 / x[..., 0]
    ret[..., 1] = -x[..., 1] / (x[..., 0] ** 2)
    return ret


def dq_multiply_with_dual_number(dq, x):
    # input dq1, dq2: [...,8], return dq1*dq2
    assert dq.shape[-1] == 8 and x.shape[-1] == 2
    p, q = dq[..., :4], dq[..., 4:]
    a, b = x[..., :1], x[..., 1:]
    ret_p = p * a
    ret_q = p * b + q * a
    ret = torch.cat([ret_p, ret_q], dim=-1)
    return ret


def dq2Rt(dq):
    # ! there is no normalzation in this function
    assert dq.shape[-1] == 8
    with torch.no_grad():
        dq_len = dq2dualnorm(dq.double())
        assert dq_len[..., -1].max() < 1e-3, "dual quaternion is not normalized"
    p, q = dq[..., :4], dq[..., 4:]
    R = quaternion_to_matrix(p)
    t = 2 * quaternion_multiply(q, quaternion_invert(p))[..., 1:]
    return R, t


def dq2T(dq):
    R, t = dq2Rt(dq)
    T = torch.cat([R, t[..., None]], -1)
    bottom = torch.zeros_like(T[..., :1, :])
    bottom[..., -1] = 1
    T = torch.cat([T, bottom], -2)
    return T


##################################################################


def dq2unitdq(dq):
    # ! handle corner case, if the input dq real part is too small, replace with a I quaternion
    invalid_mask = dq[..., :4].norm(dim=-1) < 1e-4
    _dnorm = dq2dualnorm(dq)
    _dinv = dual_inverse(_dnorm)
    unit = dq_multiply_with_dual_number(dq, _dinv)
    unit[invalid_mask] = 0.0 * dq[invalid_mask] + torch.tensor(
        [1, 0, 0, 0, 0, 0, 0, 0], device=dq.device, dtype=dq.dtype
    )
    # verify whether the real and dual are orthogonal
    with torch.no_grad():
        inner = (unit[..., :4] * unit[..., 4:]).sum(-1).abs()
        if inner.max() > 1e-5:
            logging.warning(f"not orthogonal {inner.max()}")
    return unit


if __name__ == "__main__":
    u = F.normalize(torch.randn(1000, 3), dim=1)
    theta = torch.rand(1000) * 2 * np.pi
    print(theta.min(), theta.max())
    axis_angle = u * theta[:, None]
    R = axis_angle_to_matrix(axis_angle)
    t = torch.randn(1000, 3)
    t_len = t.norm(dim=-1)
    print(t_len.min(), t_len.max())

    dq = Rt2dq(R.double(), t.double())
    dq = dq2unitdq(dq)
    R_recon, t_recon = dq2Rt(dq)

    # print(abs(R_recon - R).max())
    # print(abs(t_recon - t).max(), abs(t_recon - t).mean())

    T = torch.eye(4).expand(1000, -1, -1).to(dq.device).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T_inv = torch.inverse(T)

    dq_inv = Rt2dq(T_inv[..., :3, :3], T_inv[..., :3, 3])

    sk_w = torch.rand(100, 10).to(dq)
    sk_w = sk_w / sk_w.sum(dim=1, keepdim=True)

    dq = dq.reshape(100, 10, 8)
    dq_inv = dq_inv.reshape(100, 10, 8)

    dq_b = (dq * sk_w[:, :, None]).sum(dim=1)
    dq_b = dq2unitdq(dq_b)
    T_b = dq2T(dq_b)

    dq_inv_b = (dq_inv * sk_w[:, :, None]).sum(dim=1)
    dq_inv_b = dq2unitdq(dq_inv_b)
    R_inv_b, t_inv_b = dq2Rt(dq_inv_b)
    T_inv_b = dq2T(dq_inv_b)

    error = torch.einsum("nij,njk->nik", T_b, T_inv_b)
    angle_error = torch.acos(
        torch.clamp(
            (error[..., :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1, 1
        )
    ).max()
    trans_error = (error[..., :3, 3].norm(dim=-1)).max()
    print(angle_error, trans_error)

    T_b_recon = torch.inverse(T_inv_b)

    print()

    # T = torch.eye(4).expand(1000, -1, -1).to(R)
    # T[..., :3, :3] = R
    # T[..., :3, 3] = t
    # dq2 = T2dq(T)
    # R2, t2 = dq2Rt(dq2)
    # T2 = dq2T(dq2)
    # print()
