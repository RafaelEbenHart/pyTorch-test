import torch

# manipulasi tensor
# 1.penjumlahan tensor
# 2.perkalian tensor
# 3.perkalian matriks
# 4.pembagian tensor
# 5.perkalian elemen tensor
# 6. pengurangan tensor

# 1.penjumlahan tensor
tensorTambah = torch.tensor([1, 2, 3])
print(tensorTambah + 10)

# 2.perkalian tensor
tensorKali = torch.tensor([1, 2, 3])
print(tensorKali * 10)

# 3.pengurangan tensor
tensorKurang = torch.tensor([1, 2, 3])
print(tensorKurang - 10)

# 4.pembagian tensor
tensorBagi = torch.tensor([1, 2, 3])
print(tensorBagi / 10)

#pytorch in built function

# Perkalian tensor dengan in built function
torch.mul(tensorKali,10) #perkalian tensor dengan in built function
print(torch.mul(tensorKali,10))

# Penjumlahan tensor dengan in built function
torch.add(tensorTambah,10) #penjumlahan tensor dengan in built function
print(torch.add(tensorTambah,10))

# Pengurangan tensor dengan in built function
torch.sub(tensorKurang,10) #pengurangan tensor dengan in built function
print(torch.sub(tensorKurang,10))

# Pembagian tensor dengan in built function
torch.div(tensorKurang,10) #pembagian tensor dengan in built function
print(torch.div(tensorKurang,10))