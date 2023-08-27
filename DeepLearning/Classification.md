Classification is a problem of predicting whether something is one thing or another.

#### A easy way to create a model

```python
model = nn.Sequential(
	nn.Linear(in_features=2,out_features=5),	
    nn.Linear(in_features=5,out_features=1),	
).to(device)
```

but less flexibility.