import torch
import numpy as np #logika perhitungan
import pandas as pd #data manipulasi
import matplotlib.pyplot as plt #visualisasi data

##scalar
scalar = torch.tensor([7]) #tensor skalar
print(scalar)
print(scalar.ndim) #jumlah dimensi tensor
print(scalar.shape) #menampilkan bentuk tensor skalar
print(scalar.item()) #mengambil nilai dari tensor skalar menjadi int


##vector
vector = torch.tensor([7, 3]) #tensor vektor
print(vector)
print(vector.ndim) #jumlah dimensi tensor vektor
print(vector.shape) #menampilkan bentuk tensor vektor


#MATRIX
matrix = torch.tensor([[7, 3], [2, 1]]) #tensor matriks
print(matrix)
print(matrix) #menampilkan tensor matriks setelah diubah
print(matrix.ndim) #jumlah dimensi tensor matriks
print(matrix.shape) #menampilkan bentuk tensor matriks
print(matrix[1]) #menampilkan baris ke-2 tensor matriks


##TENSOR
tensor133 = torch.tensor([[[1,2,3],
                          [4,5,6],
                          [7,8,9]]]) #tensor 3 dimensi
print(tensor133)
print(tensor133.ndim) #jumlah dimensi tensor
print(tensor133.shape) #menampilkan bentuk tensor
(1,3,3) #artinya tensor memiliki 1 layer, 3 baris, dan 3 kolom

tensor233 = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9]],
                        [[10,11,12],
                         [13,14,15],
                         [16,17,18]]]) #tensor 3 dimensi
print(tensor233)
print(tensor233.ndim) #jumlah dimensi tensor
print(tensor233.shape) #menampilkan bentuk tensor
(2,3,3) #artinya tensor memiliki 2 layer, 3 baris, dan 3 kolom

##zeros and ones tensor
zeros = torch.zeros(3, 4) #membuat tensor nol dengan ukuran tergantung parameter
print(zeros)
sum = zeros * tensor133
print(sum) #menampilkan hasil perkalian tensor nol dengan tensor acak

ones = torch.ones(3, 4) #membuat tensor satu dengan ukuran tergantung parameter
print(ones)
print(ones.dtype) #dtype adalah default data type


