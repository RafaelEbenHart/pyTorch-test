import torch

# 1
soal1 = torch.rand(7, 7)
print(soal1)

# soal 2
soal2a = torch.rand(1, 7)
soal2b = torch.rand(1, 7)
# kali
transposeSoal2 = soal2a.T @ soal2b
print(transposeSoal2)

# soal 3
random_seed = 0
torch.manual_seed(random_seed)
soal3a = torch.rand(7, 7)
print(soal3a)

torch.manual_seed(random_seed)
soal3b = torch.rand(1,7)
torch.manual_seed(random_seed)
soal3c = torch.rand(1,7)
transposeSoal3 = soal3b.T @ soal3c
print(transposeSoal3)

# soal 4
torch.cuda.manual_seed(1234) ## seed untuk GPU
soal4a = torch.rand(1, 2).cuda()
print(soal4a.device)


# soal 5
torch.manual_seed(1234)
soal5a = torch.rand(2, 3).cuda()

torch.manual_seed(1234)
soal5b = torch.rand(2, 3).cuda()
print(soal5a == soal5b)

# soal 6
soal6a = torch.rand(2, 3)
soal6b = torch.rand(2, 3)
matmul = torch.matmul(soal6a.T, soal6b)
print(matmul)

# soal 7
soal7a = torch.arange(1, 8)
print(soal7a.max(), soal7a.min())

# soal 8
soal8a = torch.arange(1, 8)
print(soal8a.argmax(), soal8a.argmin())

# soal 9
torch.manual_seed(7)
soal9a = torch.rand(1, 1, 1, 10)
print(soal9a, soal9a.shape)
soal9b = soal9a.squeeze(0)
print(soal9b, soal9b.shape)
soal9c = soal9b.squeeze(0)
print(soal9c, soal9c.shape)
soal9d = soal9c.squeeze(0)
print(soal9d, soal9d.shape)
