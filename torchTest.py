import torch

# print("Versi PyTorch:", torch.__version__)
# print("CUDA tersedia:", torch.cuda.is_available())
# print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Tidak ada GPU")

a = torch.tensor([1,3])
b = torch.zeros(1,2)
print(a)
print(b)