import torch
import numpy as np

# numpy adalah library untuk komputasi numerik di Python
# data di numpy disimpan dalam bentuk array dan bisa diubah menjadi tensor PyTorch
# menggunakan torch.from_numpy("nama_array") untuk numpy ke tensor
# menggunakan .numpy() atau torch.Tensor.numpy() untuk mengubah tensor ke numpy


# numpy array ke tensor PyTorch
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor)
# default datatype untuk array adalah float64 dan jika di ubah menjadi tensor maka dtype akan menjadi float64 bkn float32
print(tensor.dtype)
# mencoba manipulasi array
array = array + 1
print(array, tensor)
# Kesimpulan  = tensor akan berubah jika nilai pada array yang di konversi tidak berubah

# tensor ke numpy
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor, numpy_tensor)
# jika tensor diubah ke numpy maka numpy array akan mengikuti dtype tensor yakni float32
# jika tensor di manipulasi maka tensor yang diubah menjadi numpy nilainya tidak akan berubah
