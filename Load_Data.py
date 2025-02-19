import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')

# Define paths (adjust these paths to your local setup)
satview_path = "C:\\Users\\immar\\Downloads\\Desktop\\CVACT\\SatView"
streetview_path = "C:\\Users\\immar\\Downloads\\Desktop\\CVACT\\GroundTruth"

# Define transforms: Convert to tensor and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Get filenames and extract the first 15 characters as keys
satview_files = {f[:15]: os.path.join(satview_path, f) for f in os.listdir(satview_path) if f.endswith(('jpg', 'png'))}
streetview_files = {f[:15]: os.path.join(streetview_path, f) for f in os.listdir(streetview_path) if f.endswith(('jpg', 'png'))}

# Find common keys (matching images)
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

satview_tensors = []
streetview_tensors = []

# Load images, transform them, and stack into tensors.
for key in common_keys:
    sat_img = Image.open(satview_files[key]).convert("RGB")
    street_img = Image.open(streetview_files[key]).convert("RGB")

    satview_tensors.append(transform(sat_img))
    streetview_tensors.append(transform(street_img))

satview_tensors = torch.stack(satview_tensors)
streetview_tensors = torch.stack(streetview_tensors)

print(f"Satview Tensor Shape: {satview_tensors.shape}")  # e.g., (N, 3, 256, 256)
print(f"Streetview Tensor Shape: {streetview_tensors.shape}")

torch.save({'satview': satview_tensors, 'streetview': streetview_tensors}, "output_tensors.pt")

