Classification is a problem of predicting whether something is one thing or another.

#### A easy way to create a model

```python
model = nn.Sequential(
	nn.Linear(in_features=2,out_features=5),	
    nn.Linear(in_features=5,out_features=1),	
).to(device)
```

but less flexibility.

#### Binary Classification

loss function:  BCELoss or BCEWithLogitsLoss，

BCELoss requires input to have gone through sigmoid layer prior to itself。

BCEWithLogitsLoss has already combined sigmoid layer in it self, namely 

BCEWithLogitsLoss = (Sigmoid => BCELoss)

For optimizer, we still use SGD.

#### Get Accuracy

```py
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_true) * 100
```

torch.eq returns a tensor whose certain element is True  when y_true and y_pred **have the same data in the corresponding place** otherwise False.

tensor.sum returns the sum of all the value in tensor **in the form of a Scala tensor**. If it gets a 'dim param'，it will reshape the result into corresponding dimension.

tensor.item returns a standard python number.

