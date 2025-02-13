import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# -------------------------------
# UNet Architecture for Diffusion Model
# -------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._conv_block(feature * 2, feature))

        # Final Output Layer
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_layer(x)

# --------------------------------
# Diffusion Model Implementation
# --------------------------------
class DiffusionModel(nn.Module):
    def __init__(self, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.unet = UNet(in_channels=3, out_channels=3)

        # Noise Schedule (Beta values)
        self.beta_schedule = torch.linspace(0.0001, 0.02, timesteps)
        self.alpha_schedule = 1.0 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)

    def forward_process(self, x, t):
        """ Adds noise to the image at a given timestep `t` """
        noise = torch.randn_like(x)
        alpha_t = self.alpha_cumprod[t].to(x.device)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

    def reverse_process(self, x):
        """ Removes noise step by step using U-Net """
        for t in reversed(range(self.timesteps)):
            predicted_noise = self.unet(x)
            alpha_t = self.alpha_cumprod[t].to(x.device)
            x = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        return torch.clamp(x, -1, 1)  # Ensure pixel values remain valid

# --------------------------------
# Dataset & Image Loader for MPI Output
# --------------------------------
class MPIDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --------------------------------
# Training the Diffusion Model
# --------------------------------
def train_diffusion_model():
    folder_path = "path_to_mpi_output_images"  # 🔹 This folder should contain MPI-generated images
    transform = transforms.Compose([
        transforms.Resize((512, 256)),  # 🔹 Ensure consistency with the expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 🔹 Normalize between [-1, 1]
    ])

    dataset = MPIDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Loss Function (MSE for Image Reconstruction)
    mse_loss = nn.MSELoss()

    for epoch in range(10):  # 🔹 Adjust epochs as needed
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)

            # Apply Forward Diffusion (Noise Addition)
            t = torch.randint(0, model.timesteps, (images.shape[0],), device=device)
            noisy_images = model.forward_process(images, t)

            # Reverse Process (Denoising)
            generated_images = model.reverse_process(noisy_images)

            # Compute Loss (MSE)
            loss = mse_loss(generated_images, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/10], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "diffusion_model.pth")

# --------------------------------
# Inference (Generating Final Street View)
# --------------------------------
def generate_images():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load("diffusion_model.pth"))
    model.eval()

    img_path = "path_to_noisy_image.png"  # 🔹 MPI-generated noisy street view
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0).to(device)

    # Generate cleaned street-view image
    with torch.no_grad():
        generated_image = model.reverse_process(image)

    # Convert back to RGB format
    generated_image = (generated_image.squeeze(0).cpu() * 255).permute(1, 2, 0).numpy().astype("uint8")
    Image.fromarray(generated_image).save("generated_streetview.png")

# --------------------------------
# Run Training or Inference
# --------------------------------
if __name__ == "__main__":
    train_diffusion_model()
    # generate_images()  # Uncomment to run inference
