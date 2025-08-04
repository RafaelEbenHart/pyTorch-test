import torch

#Random Tensor
#random tensor sangat penting karena penggunaan nueral network dilakukan dengan
# tensor yang acak dan merangkainya menjadi data yang bisa terima manusia

#Random Tensor
random_tensor = torch.rand(3, 4) #membuat tensor acak dengan ukuran tergantung parameter
print(random_tensor)
print(random_tensor.ndim) #jumlah dimensi tensor acak

##Random tensor dengan tampilan dengan image tensor
random_image_size_tensor = torch.rand(3,224,224) #tinggi, lebar, channel warna (RGB)
# print(random_image_size_tensor)
print(random_image_size_tensor.ndim) #jumlah dimensi tensor acak
print(random_image_size_tensor.shape) #menampilkan bentuk tensor acak

#########################################################################################

# range tensor dan tensor like
##range
range = torch.range(0,10) #range membuat tensor sebanyak dari parameter
print(range) #range membuat tensor sebanyak dari parameter
print(torch.arange(0,10)) #arange membuat tensor sebanyak dari parameter -1
# range dimulai dari start, diakhiri di end, dan step adalah jarak antar angka
print(torch.arange(0,10,2))


#like
satu_sepuluh = torch.zeros_like(input = range)
print(satu_sepuluh) #membuat tensor nol dengan ukuran yang sama dengan "range"

#########################################################################################

##tensor datatype

float_32_tensor = torch.tensor([3.0,6.0,9.0],
                               dtype=None, #datatype adalah data type yang ingin digunakan pada tensor biasa di gunakana adalah 32 dan 16
                               device=None, #tensor akan dibuat di perangkat yang ditentukan, bisa "cpu" atau "cuda" (GPU)
                               requires_grad=False) #apakah tensor membutuhkan gradien untuk backpropagation, defaultnya adalah False
# Datatyepe 16 membuat presisi pemrosesan rendah namun lebih cepat
# Datatype 32 membuat presisi pemrosesan baik
# Datatype 64 membuat presisi pemrosesan tinggi dan lebih lambat

# ##pengembangan tensor memiliki 3 masalah yang mungkin terjadi
# # 1. Ukuran tensor yang tidak sesuai / shape
# # 2. Data type tensor yang tidak sesuai
# # 3. Perangkat yang tidak sesuai (CPU/GPU)

print(float_32_tensor)
print(float_32_tensor.dtype) #default datatype tensor adalah float32

float_16_tensor = float_32_tensor.type(torch.float16) #mengubah tipe data tensor menjadi float16
print(float_16_tensor)

kali = float_16_tensor * float_32_tensor #perkalian tensor float16 dengan float32
print(kali)
print(kali.dtype) #menampilkan tipe data hasil perkalian tensor float16 dengan float32

int_32_tensor = torch.tensor([3 ,6 ,9], dtype=torch.int32)
print(int_32_tensor)
print(int_32_tensor * float_32_tensor)
# operasi antara tensor int32 dan float32 masih dapat menghasilkan ouput tanpa error,
# namun perlu di ingat bahwa untuk melakukan operasi pastikan tensor yang di operasikan dalam datatype yang sama untuk menghindari error

#############################################################################################################################################

#tensor atrributes
# Tensor memiliki beberapa atribut yang dapat digunakan untuk memeriksa kondisi tensor
# 1. Ukuran tensor yang tidak sesuai / shape,untuk mengecek ukuran tensor, gunakan "tensor.shape"
# 2. Data type tensor yang tidak sesuai, untuk mengecek tipe data tensor, gunakan "tensor.dtype"
# 3. Perangkat yang tidak sesuai (CPU/GPU), untuk mengecek perangkat tensor, gunakan "tensor.device"

a_tensor = torch.rand([3, 4])
print(a_tensor)
#detail dari a_tensor
print(f"shape dari a_tensor: {a_tensor.shape}") # atau {a_tensor.size()}
print(f"dtype dari a_tensor: {a_tensor.dtype}")
print(f"device dari a_tensor: {a_tensor.device}")