import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import numpy as np

def rmse(img1, img2):
    """
    Compute Root Mean Square Error (RMSE) between two images.
    """
    return torch.sqrt(F.mse_loss(img1, img2))

def psnr(img1, img2, max_pixel_value=255.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(max_pixel_value / torch.sqrt(mse))

def compute_ssim(img1, img2, default_win_size=7):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    """
    img1_np = img1.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  # Convert to numpy (H, W, C)
    img2_np = img2.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  

    # Get the smallest image dimension
    min_dim = min(img1_np.shape[0], img1_np.shape[1])  # (Height, Width)

    # Ensure win_size is <= min_dim and an odd number
    win_size = min(default_win_size, min_dim)
    if win_size % 2 == 0:
        win_size -= 1  # Make sure it's odd

    return ssim(img1_np, img2_np, data_range=img2_np.max() - img2_np.min(), multichannel=True, win_size=win_size)


# Example Usage
if __name__ == "__main__":
    # Dummy images (batch_size=1, channels=3, height=256, width=256)
    img1 = torch.rand((1, 3, 256, 256)) * 255  # Simulating an image
    img2 = torch.rand((1, 3, 256, 256)) * 255  # Simulating a distorted image

    print(f"RMSE: {rmse(img1, img2).item():.4f}")
    print(f"PSNR: {psnr(img1, img2).item():.2f} dB")