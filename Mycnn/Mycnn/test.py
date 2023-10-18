import torch

# x = torch.tensor([[0,1.0]])
# y = torch.tensor([1])
# loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
# print(loss_fn(x,y))

x = torch.tensor([1, 2, 3, 4, 5.0])
print(x.mean())
