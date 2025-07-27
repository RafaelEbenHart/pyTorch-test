import torch

print("Versi PyTorch:", torch.__version__)
print("CUDA tersedia:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Tidak ada GPU")
