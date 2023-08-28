## Online Book

https://www.learnpytorch.io/

## Machine Learning

#### Concept

 turning things(data) into numbers and finding patterns in those numbers.

& finding patterns in those numbers

#### Compare with DL

![image-20230710154814064](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230710154814064.png)

#### Compare with Traditional Programming

![image-20230710155237381](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230710155237381.png)

In traditional programming, you give the inputs and instructions to the computer,and then the computer give you the final result.

But in ML,you need to give both the inputs and the right outputs, and then the computer will give you the rules -- the bridge between inputs and outputs.

#### Where to apply ML or DL

![image-20230710161437897](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230710161437897.png)

All in all , apply ML when rules are so complex that humans can hardly figure out. (Just like what is a banana , namely the PATTERN).

#### Where not to apply ML or DL

![image-20230710162235158](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230710162235158.png)

The pattern learned by ML is typically uninterpretable by human, so the result produced by ML is unpredictable for human. When errors are unacceptable, ML is not reliable.

#### Difference

ML is better for Structured Data (columns and rows like data sheet).

DL is better for unstructured data (diagram , chat msg, post , tweet, image, audio)

#### Neural Network

Input ----[Numerical encoding]--> Numbers -> Neural Layers -> Output Numbers -> Human understandable items

#### Types of Learning

Supervised Learning : You give the data(photos) and labels (answers, dog or cat?)， pass then to computer.

Unsupervised and self-supervised Learning: You only have the data, and you don't have the labels. It will automatically tell the similarity or difference between the unlabeled data. So it is best for clustering(group things) and association.

Transfer Learning: Pass the knowledge acquired by one model to another model.

Reinforcement Learning: give praise for some action to encourage your model do something.

#### Application Area

recommendation ; Speech Recognition ; translation ; CV ; NLP

## PyTorch

the most popular machine learning framework.

1. Provide you some pre-built ML model.
2. Help you preprocess data , model data, and deploy model in your application or cloud.

CUDA: an api NVIDA provide for programmers to be able to use GPU to accelerate numerical calculation.

#### What is tensor

A rank-n tensor in m-dimensions is a mathematical object that has n indices and m^n componens and obeys certain transformation rules.

![image-20230710232039481](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230710232039481.png)

rank: how many information you need to figure specific component, (here is x and y)

dimension: the cols and rows of matrix

In pytorch, we refer the numbers derived from images/audio/text to TENSORS.

Maybe a matrix of vectors.  

![image-20230713000044284](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230713000044284.png)

TESNSOR = pytorch.tensor()  to create a tenser.

TENSOR.ndim = the number you need to locate a number

TENSOR.shape= the vector of number of each bracket

```python
torch.rand(size=(a,b))  # Create tensor with specific size
torch.arange(start = 1,end = 10 , step = 1) # Create a tensor which resembles vector
# output tensor([1,2,3,4,5,6,7,8,9])

torch.zeros/ones_like(inputTesor)  #return a tensor filled with zeros or ones in the same shape as the input respectively

```

#### Torch DataType

torch.float16 / torch.half

torch.float32 / torch.float

torch.float64 / torch.double

#### Torch get random

torch.rand(size, require_grad,dtype) ; random num between 0~1

torch.randn(size, require_grad,dtype) normal distribution between 0 ~ 1

#### Torch Operation

**\+ \- \\ \*** is element-wise. 

```
[1,2,3] + 10 = [11,12,13] ; 
[1,2,3] * [1,2,3] = [1,4,9]
[1,2,3] * 10 = [10,20,30] ; 
```

#### torch.matmul(matrix1,matrix2)

 Matrix Multiplication

#### MATRIX.T 

 Matrix Transpose

#### torch.min/max/mean/sum(myTensor)

aggregation function

#### torch.argmin/argmax()

find the position of the number that has the min/max value

#### tensor.type(newType)

transform original type into new type , return the new one.

#### tensor.reshape(newShape)

transform original shape to new shape, return the new one

e.g. 10 -> 2 * 5

#### tensor.view(newSize)

transform original shape to new shape, return the new one with the same memory.

z = x.view（3，4）

z[0] = 5  ;      result that x[0] =5

#### torch.stack([tensor1,tensor2,...], dim = 0)

concat the tensor list along a new dimension

output.ndim = tensor.ndim+1. e.g. 2 -> 3

dim: which dim is the new tensor placed , so it must be strict smaller than the output.ndim

```py
x = [1,2,3]
y = [3,4,5]
z = torch.stack([x,y]) ; # z = [[1,2,3],
						 #	    [3,4,5]]
```

#### torch.squeeze(x)

remove all the 1 dim from x , quiet awesome

```py
x = torch.zeros(2,1,2,1,2)
print(f"before squeeze ,x = {x}")
print(f"before squeeze ,shape = {x.shape}")
x = torch.squeeze(x)
print(f"after squeeze ,x = {x}")
print(f"after squeeze ,shape = {x.shape}")
```

```
before squeeze ,x = tensor(
[
	[[   [[0., 0.]],
      [[0., 0.]]
    ]],
    [[    [[0., 0.]],
          [[0., 0.]]
    ]]
])
before squeeze ,shape = torch.Size([2, 1, 2, 1, 2])
after squeeze ,x = tensor(
[
	[	[0., 0.],
    	[0., 0.]
    ],
    [	[0., 0.],
    	[0., 0.]
   	]
])
after squeeze ,shape = torch.Size([2, 2, 2])
```

#### torch.unsqueeze(x, dim = y)

insert a new dim at y of x

```python
x = torch.tensor([1,2,3])
x = torch.unsqueeze(x,dim=0) ; # [[1,2,3]]
x = torch.unsqueeze(x,dim=1) ; # [[1],[2],[3]]
```

#### torch.permute(x,(oldDim1,oldDim2,oldDim3))

rearrange the dimension of x, **the result shares the same memory of x**

```py
x = torch.rand(2,3,5)
y = torch.permute(x,(2,0,1));
y.size  # torch.Size([5,2,3])
```

#### torch.nn.model.to(device)

when model receive the data, put the data on the specific device

#### Index

using : to get the results quickly

```py
x = torch.arange(1,13,1);
x = x.reshape(2,2,3)
# tensor([
#         [
# 		   [ 1,  2,  3],
#          [ 4,  5,  6]
# 		  ],
#         [
#	       [ 7,  8,  9],
#          [10, 11, 12]
#      	  ]
#		 ])
#
x[:,:,0]  -> [ [1,4],[7,10] ]  the depth of bracket equals the number of colons
x[:,0]    -> [ [1,2,3],[7,8,9]]
```

#### Reproducibility

Reproducibility means trying to take random out of random

```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED) 
x = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED) 
y = torch.rand(3,4)
print(x == y)
```

both x and y are the tensors created by random seed 42.

#### tensor.to("deviceName")

we usually use it to pass tensor from original device to new device.

```
y = x.to("cuda")
```

## PyTorch Workflow

1. prepare and load the data
2. build model
3. fitting the model to data (train)
4. making predictions and evaluating the model

```python
import torch
from torch import nn. # nn contains all the building block for neural networ
```

#### Visualize your data

```python
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(10, 7))
    # draw dot 
    plt.scatter(train_data, train_labels, c="b", s=4, label="Train Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test Data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r",
                    s=4, label="Prediction Data")
    # set the font size of legend
    plt.legend(prop={"size": 14})
    # draw the figure
    plt.show()
```

![image-20230725001924445](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230725001924445.png)

Attention to the inference mode,

```python
with torch.inference_mode():
    y_predict = model(standardX)
print(y_predict)
y_predict = model(standardX)
print(y_predict)
```

![image-20230809212427147](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230809212427147.png)

The two results are different, the second one has a grad_fn.

The inference mode turns off gradient tracking & a couple of things behind the scenes.

**The sad news is that we can't use the y_predict below to plot.**

#### Build Your own Model

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # use nn.Parameter to allow 'weight' param to be trained / adjusted
        self.weight = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float))
        # use Parameter too.
        self.bias = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias
```

MyModel inherit nn.Module. nn Module is the base class for all neural network.It defines the base structure of a model.

**nn.Parameter** needs a torch.Tensor to be its param.

**nn.forward** [must] be override

#### Check the Parameter

```python
model = mm.LinearRegressionModel()
t = model.state_dict()
print(t)
```

**model.state_dict()** returns a dict where the key is name of parameter and the value is the tensor

#### Train your Model

First you need to do is to create a 【**Loss Function**】, which is one of the ways to describe how wrong your model's predictions are from the ideal outputs.

Then you need a 【**Optimizer**】,which takes into account the loss of a model and adjusts the parameters.

Specifically for pytorch, we need training loop & testing loop.

#### Loss Function in Pytorch

- **nn.L1loss** : mean absolute error calculating from the input tensor x & y
- **nn.MSEloss**：mean square error from x & y

#### Optimizer in Pytorch

lr = learning rate , which is a hyper param we can set. It affects how big  the params change in one hit.

- **torch.optim.SGD**(params, lr) : stochastic gradient decent

#### Pick loss function & optimzer

```python
loss_fn = nn.L1Loss()
optimzer = torch.optim.SGD()
```

#### Building a training loop (and a testing loop)

<img src="C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230809185620659.png" alt="image-20230809185620659" style="zoom:100%;" />

```python
epochs = 1
for epoch in range(epochs):
    # tell your model that you are training ; a switch for specific layers
    model.train()
	# Forward pass
    y_predict = model(standardX)
	# Calculate the loss
    loss = loss_fn(y_predict, standardY)
	# Set the gradients of each param to zero
    optimzer.zero_grad()
	# Calculate the gradients for each param which requires_grad = true
	# i.e. x.grad += dloss / dx
    loss.backward()
	# x += -lr * x.grad
    optimzer.step()
	# Start to evaluate your model, turn off some layers.
	# (Dropout Layer & Barchnorm)
    model.eval()
    # The evaluation section
    if epoch % 10 == 0:
        with torch.inference_mode():
            test_pred = model(standardX)
            test_loss = loss_fn(test_pred, standardY)
            print("=========================")
            print(f"In epoch{epoch} | Loss: {loss}")
            print(model.state_dict())
```

#### Draw the loss curves

Before entering the traing loops, we prepare data for the loss curves

```python
epoch_value = []
train_loss_value = []
test_loss_value = []
```

Then at  some point in the loop, we need record these values

e.g. the test loop

```python
if epoch % 50 == 0:
    with torch.inference_mode():
        test_pred = model(standardX)
        test_loss = loss_fn(test_pred, standardY)
        epoch_value.append(epoch)
        train_loss_value.append(loss)
        test_loss_value.append(test_loss)
        print("=========================")
        print(f"In epoch{epoch} | Loss: {loss}")
        print(model.state_dict())
```

Finally, after the loop, we draw the figure.

**ATTENTION! ** we need to **transform the tensor into normal array or numpy array**, so that it will fit the param type in plt.plot()

just using `np.array(torch.tensor(your_record).numpy())` the np.array are unnecessarily needed.

```python
# draw the loss curves
plt.plot(epoch_value, np.array(torch.tensor(
    train_loss_value).numpy()), label="Train Loss")
plt.plot(epoch_value, np.array(torch.tensor(
    test_loss_value).numpy()), label="Test Loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
```

#### Save your model

Three main methonds.

`torch.save()` - allows you save a PyTorch object in Python's **pickle format**

```python
path = "./model_save/test_model.pth"
print(f"Saving model to: {path}")
torch.save(obj=model.state_dict(),f=path)
```

`torch.load()` - load a object



`torch.nn.Module.load_state_dict()` load a state_dict

It is more recommended to save state_dict rather than the whole model.

```python
path = "model_save/test_model.pth"
load_model = mm.LinearRegressionModel()
print(load_model.state_dict())

load_model.load_state_dict(torch.load(f=path))
print(load_model.state_dict())
```

![image-20230811180314530](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230811180314530.png)
