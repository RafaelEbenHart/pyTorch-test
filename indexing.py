import torch

## Indexing / Slicing

# tensorIMG = torch.rand(size=(224, 224, 3)) # tinggi , lebar dan color channel
# tensorIMG_perm = tensorIMG.permute(2, 0 ,1) # color channel, tinggi, lebar
# print(f"hasil permute dari tensorIMG {tensorIMG_perm.shape}")
# print(tensorIMG[1, 0, 1])
# print(tensorIMG[0, 0, 0] ,tensorIMG_perm[0, 0, 0])

tensor = torch.arange(1, 10).reshape(1, 3, 3)
print(tensor, tensor.shape)
print(tensor[0]) #indexing bracket pertama
print(tensor[0, 0], tensor[0][0]) # indexing bracket kedua
print(tensor[0, 0, 0], tensor[0][2][2]) # indexing bracket ketiga
## kamu bisa juga menggunakan ":" untuk mengambil semua elemen pada dimensi tertentu
print(tensor[:, 1, :]) # mengambil semua elemen pada dimensi kedua
               #> #v
