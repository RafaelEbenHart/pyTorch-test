import torch
import numpy as np
## membuat objek tensor pada GPU / CUDA

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
## Informasi tentang GPU ##
print(f"jumlah GPU yang tersedia: {torch.cuda.device_count()}")
print(f"GPU yang digunakan: {torch.cuda.get_device_name(0)}")
##

tensor = torch.tensor([1, 2, 3], device=device)
print(tensor)

## convert tensor dari CPU ke GPU
tensorCPU = torch.tensor([1, 2, 3])
print(tensorCPU, tensorCPU.device)
tensorGPU = tensorCPU.to(device)
print(tensorGPU, tensorGPU.device)

## conevert tensor dari GPU ke CPU
## hal ini dilakukan karna numpy defaultnya hanya bisa di CPU
## print(tensorGPU.numpy())
##

tensorGPUtoCPU = tensorGPU.to("cpu")
tensorGPUtoCPU = tensorGPU.cpu()  # alternatif cara untuk mengubah tensor GPU ke CPU
print(tensorGPUtoCPU, tensorGPUtoCPU.device)
print(tensorGPUtoCPU.numpy())  # mengubah tensor GPU ke numpy array
