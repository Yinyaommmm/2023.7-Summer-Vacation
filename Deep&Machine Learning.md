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