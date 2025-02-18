import os
from PIL import Image
import torch
import torchvision.transforms as transforms

# Define paths
satview_path = "C:\\Users\\immar\\Downloads\\Desktop\\CVACT\\SatView"
streetview_path = "C:\\Users\\immar\\Downloads\\Desktop\\CVACT\\GroundTruth"

# Define transforms: Convert to Tensor, Normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (values in range [0,1])
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
])

# Get filenames and extract the first 15 characters as keys
satview_files = {f[:15]: os.path.join(satview_path, f) for f in os.listdir(satview_path) if f.endswith(('jpg', 'png'))}
streetview_files = {f[:15]: os.path.join(streetview_path, f) for f in os.listdir(streetview_path) if
                    f.endswith(('jpg', 'png'))}

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

# Define batch size
batch_size = 32  # Adjust this value as needed


# Function to process and save a batch of images
def process_and_save_batch(batch_keys, batch_index):
    satview_tensors = []
    streetview_tensors = []

    for key in batch_keys:
        sat_img = Image.open(satview_files[key]).convert("RGB")
        street_img = Image.open(streetview_files[key]).convert("RGB")

        satview_tensors.append(transform(sat_img))
        streetview_tensors.append(transform(street_img))

    # Convert list of tensors to a single tensor batch (N, C, H, W)
    satview_batch = torch.stack(satview_tensors)
    streetview_batch = torch.stack(streetview_tensors)

    # Print the shape of the tensors
    print(f"Batch {batch_index} - Satellite View Tensor Shape: {satview_batch.shape}")
    print(f"Batch {batch_index} - Street View Tensor Shape: {streetview_batch.shape}")

    # Save the batch
    torch.save({'satview': satview_batch, 'streetview': streetview_batch}, f"output_tensors_batch_{batch_index}.pt")
    print(f"Saved batch {batch_index} with {len(batch_keys)} pairs.")


# Process images in batches
for i in range(0, len(common_keys), batch_size):
    batch_keys = common_keys[i:i + batch_size]
    process_and_save_batch(batch_keys, i // batch_size + 1)
