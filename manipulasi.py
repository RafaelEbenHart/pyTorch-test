import torch
# import timeit

# # manipulasi tensor
# # 1.penjumlahan tensor
# # 2.perkalian tensor
# # 3.perkalian matriks
# # 4.pembagian tensor
# # 5.perkalian elemen tensor
# # 6. pengurangan tensor

# # 1.penjumlahan tensor
# tensorTambah = torch.tensor([1, 2, 3])
# print(tensorTambah + 10)

# # 2.perkalian tensor
# tensorKali = torch.tensor([1, 2, 3])
# print(tensorKali * 10)

# # 3.pengurangan tensor
# tensorKurang = torch.tensor([1, 2, 3])
# print(tensorKurang - 10)

# # 4.pembagian tensor
# tensorBagi = torch.tensor([1, 2, 3])
# print(tensorBagi / 10)

# # #pytorch in built function

# # Perkalian tensor dengan in built function
# torch.mul(tensorKali,10) #perkalian tensor dengan in built function
# print(torch.mul(tensorKali,10))

# # Penjumlahan tensor dengan in built function
# torch.add(tensorTambah,10) #penjumlahan tensor dengan in built function
# print(torch.add(tensorTambah,10))

# # Pengurangan tensor dengan in built function
# torch.sub(tensorKurang,10) #pengurangan tensor dengan in built function
# print(torch.sub(tensorKurang,10))

# # Pembagian tensor dengan in built function
# torch.div(tensorKurang,10) #pembagian tensor dengan in built function
# print(torch.div(tensorKurang,10))

# ################################################################################

# # Perkalian matriks
# # ada 2 cara perkalian matriks di neural network dan deep learning
# # 1 perkalian elemen matriks
# # 2 perkalian matriks (perkalian dot product)

# # perkalian elemen matriks
# print(tensorTambah ,"*", tensorTambah)
# print(f"Sama dengan: {tensorTambah * tensorTambah}")

# # # perkalian matriks (perkalian dot product)
# torch.matmul(tensorTambah, tensorTambah) #perkalian matriks dengan in built function
# print(torch.matmul(tensorTambah , tensorTambah))

# #perbandungan method perkalian matrik dan for loop perkalian matriks

# # Mengukur waktu eksekusi torch.matmul sebanyak 10000 kali
# waktumatmul = timeit.timeit(
#     stmt="torch.matmul(tensorTambah, tensorTambah)",
#     setup="import torch; tensorTambah = torch.tensor([1, 2, 3])",
#     number=10000
# )
# print(waktumatmul)

# waktuForLoop = timeit.timeit(
#     stmt="""
# value = 0
# for i in range(len(tensorTambah)):
#     value += tensorTambah[i] * tensorTambah[i]
# """,
#     setup="import torch; tensorTambah = torch.tensor([1, 2, 3])",
#     number=10000
# )
# print(waktuForLoop)

# # matmul lebih cepat daripada for loop

# #####################################################################################

# # 2 Aturan pada perkalian matriks

# # 1. Jumlah kolom pada matriks pertama harus sama dengan jumlah baris pada matriks kedua
# # @ adalah operator perkalian matriks
# print(tensorTambah @ tensorTambah)

# # print(torch.matmul(torch.rand(3, 2), torch.rand(3, 2))) # error karena shape tidak sesuai
# print(torch.matmul(torch.rand(2, 3), torch.rand(3, 2)))
# print(torch.matmul(torch.rand(3, 2) , torch.rand(2, 3))) # ,2) , (2 inner dimension
# #2 contoh diatas adalah contoh benar dengan inner dimension yang sama yang 3 dan 3 , dan 2 dan 2

# # 2. Hasil perkalian matriks akan memiliki jumlah baris dari matriks pertama dan jumlah kolom dari matriks kedua
# print(torch.rand(2, 4) @ torch.rand(4, 2)) # matriks yang dikalikan akan berbentuk 2x2 karena outer dimension nya 2
# print(torch.rand(2, 3) @ torch.rand(3, 4)) # matriks yang dikalikan akan berbentuk 2x4 karena outer dimension nya 2 dan 4 dan tidak akan terjadi eror
# print(torch.matmul(torch.rand(2, 3) , torch.rand(3, 2)).shape)

# ## mengatasi error pada perkalian matriks dalam kasus shape tidak sesuai
# ##TRANSPOSE dapat berfungsi untuk 2D(matriks) saja
# tensorA = torch.tensor([[1, 2],
#                         [3, 4],
#                         [5, 6]])
# tensorB = torch.tensor([[7, 10],
#                         [8, 11],
#                         [9, 12]])
# # torch.mm adalah fungsi untuk perkalian matriks
# print(tensorA, tensorB)
# print(tensorA.shape, tensorB.shape)

# # untuk mengatasi eror shape kita bisa gunakan *transpose*
# #transpose akan mengubah baris menjadi kolom dan kolom menjadi baris / dimensi
# tensorB.T # transpose
# tensorA.T
# print(tensorB.T.shape)

# print(torch.mm(tensorA, tensorB.T)) #matrik A dan B sakarang bisa di operasikan
# print(torch.mm(tensorA, tensorB.T).shape)
# print(torch.mm(tensorA.T, tensorB))
# print(torch.mm(tensorA.T, tensorB).shape)


##################################################################################

# ## Reshape, stacking, squeezing dan unsqueezing tensor

tensorEX = torch.arange(1., 10)
print(tensorEX , tensorEX.shape)

## view
tensorV = tensorEX.view(1, 9) # mengubah shape tensor A menjadi 1 baris dan 9 kolom
## penggunaan view untuk mengubah shape tensor,tidak membuat salinan tensor namun saat tensor sudah di transpose/permute gunakan contiguous lalu view
tensorTV = tensorEX.T.contiguous().view(1, 9)
## view harus sesuai dengan jumlah elemen pada tensor
## saat view digunakan untuk mengubah shape tensor, tensor yang dihasilkan akan memiliki elemen yang sama dengan tensor awal dan
## mengikuti perubahan yang terjadi satu sama lain jika di ubah,karena kedua tensosr berbagi memori yang sama

tensorV[:, 0] = 3 # mengubah index 0 pada tensorV menjadi 3
print(tensorEX, tensorV) # tensorEX dan tensorV akan berubah karena berbagi memori yang sama

## Reshape
tensorR = tensorEX.reshape(1, 9) # mengubah shape tensor A menjadi 1 baris dan 9 kolom
# reshape juga tidak membuat salinan tensor tapi saat fleksibel daripada view
# reshape harus sesuai dengan jumlah elemen pada tensor

## Stacking
# stacking digunakan untuk menggabungkan beberapa tensor menjadi satu tensor
# dibagi menjadi 2 yaitu vstacking dan hstacking yakni vertikal dan horizontal
tensorTumpuk  = torch.stack([tensorEX, tensorEX, tensorEX, tensorEX], dim=1) # menggabungkan tensorEX menjadi 4 baris
print(tensorTumpuk)
# dim=0 untuk menggabungkan tensor secara vertikal
# dim=1 untuk menggabungkan tensor secara horizontal

## squeezing
# squeezing digunakan untuk menghapus dimensi yang berukuran 1
print(tensorV) # tensor V sebelum di squeeze
tensorSQ = tensorV.squeeze()
print(f"tensor awal adalah {tensorV}")
print(f"bentuk tensor awal adalah {tensorV.shape}")
##  Menghapus dimensi pada tensor
print(f"tensor setelah di squeeze adalah {tensorSQ}")
print(f"bentuk tensor setelah di squeeze adalah {tensorSQ.shape}")

## unsqueezing
# unsqueezing digunakan untuk menambahkan dimensi baru pada tensor dengan dimensi yang spesifik
print(f"target tensor adalah {tensorSQ}")
print(f"bentuk tensor target adalah {tensorSQ.shape}")
## Menambahkan dimensi baru pada tensor
tensorUSQ = tensorSQ.unsqueeze(dim=0) # menambah dimensi sesuai indexing
print(f"tensor setelah unsqueeze adalah {tensorUSQ}")
print(f"bentuk tensor setelah unsqueeze adalah {tensorUSQ.shape}")

##Permute
# permute digunakan untuk mengubah urutan dimensi 3D(tensor)
tensor3D = torch.rand(2, 3, 4)
tensor3DPerm = tensor3D.permute(2, 0, 1) # mengubah urutan dimensi menjadi (4, 2, 3) dengan indexing
print(tensor3DPerm.shape)

tensorIMG = torch.rand(size=(224, 224, 3)) # tinggi , lebar dan color channel
tensorIMG_perm = tensorIMG.permute(2, 0 ,1) # color channel, tinggi, lebar
print(f"hasil permute dari tensorIMG {tensorIMG_perm.shape}")


