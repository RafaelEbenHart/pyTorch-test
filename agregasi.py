import torch

## AGREGASI TENSOR

tensor = torch.arange(0, 100, 10)
# torch.arange() membuat dtype dari tensor menjadi long / int64
print(tensor)

# Minimum dari tensor
print(torch.min(tensor), tensor.min())

# Maximum dari tensor
print(torch.max(tensor), tensor.max())

# Rata-rata dari tensor(avg)
print(torch.mean(tensor.float()) , tensor.float().mean)
# tensor.float() digunakan untuk mengubah tipe data tensor menjadi float
# lalu tensor yang telah menjadi float dapat berikan method mean()
tensor.type(torch.float) # atau bisa menggunakan method type() default dtype nya akan tetap 32

# sum dari tensor
print(torch.sum(tensor), tensor.sum())

####################################################################################
