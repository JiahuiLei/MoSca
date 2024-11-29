# import torch
# import torch.nn.functional as F
# import numpy as np

# def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
#     """Create a 1D Gaussian kernel."""
#     hw = size // 2
#     shift = (2 * hw - size + 1) / 2
#     x = np.arange(size) - hw + shift
#     g = np.exp(-0.5 * (x / sigma) ** 2)
#     g /= g.sum()
#     return torch.tensor(g, dtype=torch.float32)

# def apply_filter(img: torch.Tensor, kernel: torch.Tensor, dim: int) -> torch.Tensor:
#     """Apply Gaussian filter along a specific dimension (x or y)."""
#     channels = img.shape[1]
#     if dim == 0:
#         # Convolve along width (x-direction)
#         kernel = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
#         padding = (kernel.size(2) // 2, 0)
#     else:
#         # Convolve along height (y-direction)
#         kernel = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
#         padding = (0, kernel.size(3) // 2)

#     return F.conv2d(img, kernel, padding=padding, groups=channels)

# def compute_ssim(
#     img0: torch.Tensor,
#     img1: torch.Tensor,
#     mask: torch.Tensor = None,
#     max_val: float = 1.0,
#     filter_size: int = 11,
#     filter_sigma: float = 1.5,
#     k1: float = 0.01,
#     k2: float = 0.03,
#     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
# ) -> torch.Tensor:
#     """Computes SSIM between two images.

#     Args:
#         img0 (torch.Tensor): An image of size (1, 3, H, W) in float32.
#         img1 (torch.Tensor): An image of size (1, 3, H, W) in float32.
#         mask (Optional[torch.Tensor]): An optional foreground mask of shape (1, 1, H, W) in float32 {0, 1}.
#         max_val (float): The dynamic range of the images.
#         filter_size (int): Size of the Gaussian blur kernel used to smooth the input images.
#         filter_sigma (float): Standard deviation of the Gaussian blur kernel used to smooth the input images.
#         k1 (float): One of the SSIM dampening parameters.
#         k2 (float): One of the SSIM dampening parameters.

#     Returns:
#         torch.Tensor: SSIM in range [0, 1] of shape ().
#     """
#     img0, img1 = img0.to(device), img1.to(device)
#     if mask is None:
#         mask = torch.ones_like(img0[:, :1, :, :], device=device)
#     else:
#         mask = mask[:, 0, :, :].unsqueeze(1).to(device)

#     kernel = gaussian_kernel(filter_size, filter_sigma).to(device)

#     def convolve2d(img, mask, kernel):
#         img_ = apply_filter(img * mask, kernel, dim=0)
#         img_ = apply_filter(img_, kernel, dim=1)
#         mask_ = apply_filter(mask, kernel, dim=0)
#         mask_ = apply_filter(mask_, kernel, dim=1)
#         return torch.where(mask_ != 0, img_ * kernel.sum() / mask_, torch.zeros_like(img_)), (mask_ != 0).float()

#     mu0, _ = convolve2d(img0, mask, kernel)
#     mu1, _ = convolve2d(img1, mask, kernel)
#     mu00 = mu0 * mu0
#     mu11 = mu1 * mu1
#     mu01 = mu0 * mu1
#     sigma00, _ = convolve2d(img0 ** 2, mask, kernel)
#     sigma11, _ = convolve2d(img1 ** 2, mask, kernel)
#     sigma01, _ = convolve2d(img0 * img1, mask, kernel)
#     sigma00 -= mu00
#     sigma11 -= mu11
#     sigma01 -= mu01

#     # Clip the variances and covariances to valid values.
#     sigma00 = torch.maximum(torch.tensor(0.0, device=device), sigma00)
#     sigma11 = torch.maximum(torch.tensor(0.0, device=device), sigma11)
#     sigma01 = torch.sign(sigma01) * torch.minimum(torch.sqrt(sigma00 * sigma11), torch.abs(sigma01))

#     c1 = (k1 * max_val) ** 2
#     c2 = (k2 * max_val) ** 2
#     numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
#     denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
#     ssim_map = numer / denom
#     ssim = ssim_map.mean()

#     return ssim

# # Example usage
# if __name__ == "__main__":
#     img0 = torch.rand(1, 3, 256, 256, dtype=torch.float32)
#     img1 = torch.rand(1, 3, 256, 256, dtype=torch.float32)
#     mask = torch.ones(1, 1, 256, 256, dtype=torch.float32)
#     ssim_value = compute_ssim(img0, img1, mask)
#     print(f"SSIM: {ssim_value.item()}")
