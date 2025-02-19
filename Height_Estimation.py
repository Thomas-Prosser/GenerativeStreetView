import torch
import torch.nn as nn
import torch.nn.functional as F

class HeightEstimationNetwork(nn.Module):
    def __init__(self, num_height_planes=64):
        super(HeightEstimationNetwork, self).__init__()
        
        self.num_height_planes = num_height_planes

        # Encoder: Multi-scale feature extraction with skip connections
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Bottleneck (Dense Feature Extraction)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Decoder: Transpose Convolutions with Skip Connections
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)  # Skip connection from conv3
        self.bn7 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)   # Skip from conv2
        self.bn8 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(128, self.num_height_planes, kernel_size=4, stride=2, padding=1)  # Skip from conv1

    def forward(self, x):
        # Encoder with skip connections
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))  # Bottleneck

        # Decoder with skip connections
        x = F.relu(self.bn6(self.deconv1(x5)))
        x = torch.cat((x, x3), dim=1)  # Skip connection

        x = F.relu(self.bn7(self.deconv2(x)))
        x = torch.cat((x, x2), dim=1)  # Skip connection

        x = F.relu(self.bn8(self.deconv3(x)))
        x = torch.cat((x, x1), dim=1)  # Skip connection

        x = self.deconv4(x)  # Output multi-plane height representation

        # Softmax across height planes
        height_probabilities = F.softmax(x, dim=1)

        return height_probabilities  # Shape: (Batch, HeightLevels, Height, Width)


