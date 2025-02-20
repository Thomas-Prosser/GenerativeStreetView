import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import pandas as pd  # Added for saving metrics

# Import custom modules.
from Height_Estimation import HeightEstimationNetwork
from DiffusionModel import DiffusionModel
from GeometryProjection import GeometryProjection
from metrics import rmse, psnr, compute_ssim
from PerceptualLoss import PerceptualLoss

# -------------------------------
# Data Loading
# -------------------------------
data = torch.load("output_tensors.pt")
satellite_tensors = data['satview']      # (N, 3, 256, 256)
streetview_tensors = data['streetview']  # (N, 3, 256, 256)

# Create dataset and dataloader
dataset = TensorDataset(satellite_tensors, streetview_tensors)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# -------------------------------
# Model Initialization
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

height_model = HeightEstimationNetwork(num_height_levels=64).to(device)
geometry_projector = GeometryProjection(target_height=256, target_width=512, 
                                        grd_height=-2, max_height=6, 
                                        num_planes=64, device=device)
diffusion_model = DiffusionModel(timesteps=1000).to(device)
perceptual_loss_module = PerceptualLoss(device=device).to(device)

optimizer = optim.Adam(list(height_model.parameters()) + list(diffusion_model.parameters()), lr=1e-4)
mse_loss_fn = nn.MSELoss()

# -------------------------------
# Training Loop
# -------------------------------
num_epochs = 10

# Lists to store metrics
perceptual_losses = []
rmse_values = []
psnr_values = []

for epoch in range(num_epochs):
    for batch_idx, (sat_img, gt_street) in enumerate(dataloader):
        sat_img = sat_img.to(device)
        gt_street = gt_street.to(device)

        optimizer.zero_grad()

        # 1. Height Estimation
        height_probs = height_model(sat_img)
        
        # 2. Geometry Projection
        projected = geometry_projector(sat_img, height_probs)

        # 3. Diffusion Model (refinement)
        projected = projected.permute(0, 3, 1, 2)
        t = torch.randint(0, diffusion_model.timesteps, (sat_img.shape[0],), device=device)
        noisy = diffusion_model.forward_process(projected, t)
        refined = diffusion_model.reverse_process(noisy)

        # 4. Compute Losses (MSE + Perceptual Loss)
        loss_mse = mse_loss_fn(refined, gt_street)
        loss_perceptual = perceptual_loss_module(gt_street, refined)
        loss = loss_mse + loss_perceptual

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Recording metrics
        perceptual_losses.append(loss_perceptual.item())
        rmse_values.append(rmse(gt_street, refined).item())
        psnr_values.append(psnr(gt_street, refined).item())

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}, "
                  f"Perceptual Loss: {loss_perceptual.item():.4f}, RMSE: {rmse_values[-1]:.4f}, PSNR: {psnr_values[-1]:.2f}")

# Save models
torch.save(height_model.state_dict(), "height_model.pth")
torch.save(diffusion_model.state_dict(), "diffusion_model.pth")

# Save metrics to a CSV file
df = pd.DataFrame({
    "Perceptual Loss": perceptual_losses,
    "RMSE": rmse_values,
    "PSNR": psnr_values
})

df.to_csv("training_metrics.csv", index=False)
print("Metrics have been saved to 'training_metrics.csv'")

# -------------------------------
# Inference Example: Generate Final Street-View Image
# -------------------------------
height_model.eval()
diffusion_model.eval()
with torch.no_grad():
    sample_sat = satellite_tensors[0:1].to(device)
    height_probs = height_model(sample_sat)
    projected = geometry_projector(sample_sat, height_probs)
    projected = projected.permute(0, 3, 1, 2)
    t = torch.randint(0, diffusion_model.timesteps, (1,), device=device)
    noisy = diffusion_model.forward_process(projected, t)
    final_img = diffusion_model.reverse_process(noisy)
    
    # Rescale from [-1,1] to [0,255] and save
    final_img = (final_img.squeeze(0).cpu().clamp(-1, 1) + 1) / 2.0
    final_img = (final_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(final_img).save("final_streetview.png")
