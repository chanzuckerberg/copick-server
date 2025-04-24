import zarr
import requests
import numpy as np
from fsspec import get_mapper

# First read the tomogram to get the shape
store = get_mapper("http://localhost:8012/16463/Tomograms/VoxelSpacing10.012/wbp-denoised-denoiset-ctfdeconv.zarr")
tomo = zarr.open(store, mode='r')
full_shape = tomo["0"].shape
print(f"Tomogram shape: {full_shape}")

# Example parameters
voxel_size = 10.012
user_id = "kisharrington"
session_id = "copickServer"
name = "membrane"  # This must match a pickable object in your config
is_multilabel = False

# Construct the segmentation filename
seg_filename = f"{voxel_size}_{user_id}_{session_id}_{name}"
if is_multilabel:
    seg_filename += "-multilabel"
seg_filename += ".zarr"

# Create segmentation data matching tomogram dimensions
data = np.zeros(full_shape, dtype=np.uint8)  # for single label

# Ensure the data is in C-contiguous memory layout
data = np.ascontiguousarray(data)

# Convert to bytes, preserving the shape information
shape_header = np.array(data.shape, dtype=np.int64).tobytes()
data_bytes = shape_header + data.tobytes()

# Send to server as a single request
url = f"http://localhost:8012/16463/Segmentations/{seg_filename}"
response = requests.put(url, data=data_bytes)

if response.status_code == 200:
    print("Successfully wrote segmentation")
else:
    print(f"Failed to write segmentation: {response.status_code} - {response.text}")
