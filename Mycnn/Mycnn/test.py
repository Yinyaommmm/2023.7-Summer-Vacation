import torch

x = torch.tensor([[0,1.0]])
y = torch.tensor([1])
loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
print(loss_fn(x,y))

# print(x.shape)
# print(y.shape)

