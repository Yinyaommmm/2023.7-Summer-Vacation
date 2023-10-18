import torch

x = torch.tensor([[-100,1.0,-100],
                  [0.33,0.33,0.33],
                  [0.7,0.3,0.001]])

y = torch.tensor([1,
                  2,
                  0])
loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
print(loss_fn(x,y))

# # x = torch.tensor([1, 2, 3, 4, 5.0])
# # print(x.mean())
