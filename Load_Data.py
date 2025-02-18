import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')

# Define paths
satview_path = "C:\\Users\\immar\\Downloads\\Desktop\\CVACT\\SatView" #Change the folder location
streetview_path = "C:\\Users\\immar\\Downloads\\Desktop\\CVACT\\GroundTruth" #Change the folder location

# Define transforms: Resize, Convert to Tensor, Normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (values in range [0,1])
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
])

# Get filenames and extract the first 15 characters as keys
satview_files = {f[:15]: os.path.join(satview_path, f) for f in os.listdir(satview_path) if f.endswith(('jpg', 'png'))}
streetview_files = {f[:15]: os.path.join(streetview_path, f) for f in os.listdir(streetview_path) if f.endswith(('jpg', 'png'))}

# Find common keys (i.e., matching images)
common_keys = sorted(set(satview_files.keys()) & set(streetview_files.keys()))

# Check for mismatches
satview_only = set(satview_files.keys()) - set(streetview_files.keys())
streetview_only = set(streetview_files.keys()) - set(satview_files.keys())

if satview_only or streetview_only:
    print("Mismatched Files Found!")
    print(f"Images only in Satview: {satview_only}")
    print(f"Images only in Streetview: {streetview_only}")
else:
    print("All images match correctly between folders!")

# Initialize lists to store tensors
satview_tensors = []
streetview_tensors = []

# Load images, transform them, and convert to tensors
for key in common_keys:
    sat_img = Image.open(satview_files[key]).convert("RGB")
    street_img = Image.open(streetview_files[key]).convert("RGB")

    satview_tensors.append(transform(sat_img))
    streetview_tensors.append(transform(street_img))

# Convert list of tensors to a single tensor batch (N, C, H, W)
satview_tensors = torch.stack(satview_tensors)
streetview_tensors = torch.stack(streetview_tensors)

# Output tensor shapes
print(f"Satview Tensor Shape: {satview_tensors.shape}")  # Expected: (N, 3, 256, 256)
print(f"Streetview Tensor Shape: {streetview_tensors.shape}")

torch.save({'satview': satview_tensors, 'streetview': streetview_tensors}, "output_tensors.pt")
