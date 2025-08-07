import torch

### Test unsqueeze
# tensorTest = torch.arange(1, 10)
# tensorUn1 = tensorTest.unsqueeze(0)
# print(tensorUn1.shape)
# print(tensorUn1.ndim)
# tensorUn2 = tensorUn1.unsqueeze(2)
# print(tensorUn2.shape)
# print(tensorUn2.ndim)
# print(tensorUn2.permute(2, 0, 1).shape)


## test materi slicing / indexing
tensor = torch.arange(1, 51).reshape(2, 5, 5)
print(tensor.shape)
print(tensor)
print(tensor[0, :, 4])
print(tensor[0, 0, :])