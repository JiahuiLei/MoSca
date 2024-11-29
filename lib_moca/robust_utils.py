import torch
import numpy
from matplotlib import pyplot as plt


@torch.no_grad()
def positive_th_gaussian_decay(x, th, sigma, viz_flag=False):
    # x is a tensor
    # if x > th put a gaussian decay weight, other wise put 1.0
    # if viz_flag is try, return a numpy array ploted from plt show the distribution hist and the weight

    weights = torch.ones_like(x)
    mask = x > th
    weights[mask] = torch.exp(-((x[mask] - th) ** 2) / (2 * sigma**2))

    return weights
