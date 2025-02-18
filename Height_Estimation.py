import torch
import torch.nn as nn
import torch.nn.functional as F

# Model definition
class HeightEstimationNetwork(nn.Module):
    def __init__(self, num_height_levels=64):
        super(HeightEstimationNetwork, self).__init__()
        
        self.num_height_levels = num_height_levels

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, self.num_height_levels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Decoder
        x = F.relu(self.bn5(self.deconv1(x)))
        x = F.relu(self.bn6(self.deconv2(x)))
        x = F.relu(self.bn7(self.deconv3(x)))
        x = self.deconv4(x)

        # Apply softmax across the height levels (channel dimension)
        height_probabilities = F.softmax(x, dim=1)

        # Rearrange dimensions to (Batch, Height, Width, HeightLevels)
        height_probabilities = height_probabilities.permute(0, 2, 3, 1)

        return height_probabilities


# Main Execution (Pure Tensor Input)
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and move to device
    model = HeightEstimationNetwork(num_height_levels=64).to(device)
    model.train()  # Ensure gradients are tracked

    # Define input tensor (Example: Random Tensor with Batch=2)
    batch_size = 2
    input_tensor = torch.randn((batch_size, 3, 256, 256), device=device)  # Simulated input

    # Forward pass through the model
    height_probabilities = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")  # Expected: [Batch, 3, 256, 256]
    print(f"Output shape: {height_probabilities.shape}")  # Expected: [Batch, 256, 256, 64]


    # Process each batch of images
    for batch_idx, images in enumerate(dataloader):
        height_probabilities = model(images)
        print(f"Batch {batch_idx + 1}: Output shape: {height_probabilities.shape}")  # Should be [1, 256, 256, 64] [BS, H, W, N]
