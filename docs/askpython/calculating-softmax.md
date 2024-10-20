# 在 Python 中计算 Softmax

> 原文：<https://www.askpython.com/python/examples/calculating-softmax>

你好，学习者！！在本教程中，我们将学习 Softmax 函数，以及如何使用 [NumPy](https://www.askpython.com/python-modules/numpy/numpy-universal-functions) 在 Python 中计算 softmax 函数。我们还将了解具有 Softmax 内置方法的框架。所以让我们开始吧。

## 什么是 Softmax 函数？

Softmax 是一个数学函数，它将数字向量作为输入，并将其归一化为[概率分布](https://www.askpython.com/python/examples/probability-distributions)，其中每个值的概率与[向量](https://www.askpython.com/python-modules/numpy/numpy-vectorization)中每个值的相对比例成比例。

在对矢量应用 softmax 函数之前，矢量的元素可以在`(-∞, ∞)`的范围内。

一些元素可以是负的，而一些可以是正的。

应用 softmax 函数后，每个值都将在`[0, 1]`的范围内，并且这些值的总和将为 1，以便可以将它们解释为概率。

softmax 的计算公式为

![\sigma(\vec{z})_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}](img/1a8bd6a051ad1d623c3c036cfd34f9a8.png)

其中我们首先找到向量中每个元素的指数，并将它们除以所计算的指数之和。

Softmax 函数最常用作多类分类问题的激活函数，在这种情况下，您有一系列值，并且需要找到它们出现的概率。softmax 函数用于预测多项式概率分布的神经网络模型的输出层。

## 用 Python 实现 Softmax 函数

现在我们知道了在一个数字向量上计算 softmax 的公式，让我们来实现它。我们将使用 NumPy `exp()`方法计算向量的指数，使用 NumPy `sum()`方法计算分母和。

```py
import numpy as np

def softmax(vec):
  exponential = np.exp(vec)
  probabilities = exponential / np.sum(exponential)
  return probabilities

vector = np.array([1.0, 3.0, 2.0])
probabilities = softmax(vector)
print("Probability Distribution is:")
print(probabilities)

```

```py
Probability Distribution is:
[0.09003057 0.66524096 0.24472847]

```

## 使用框架计算 softmax

许多框架提供了在各种数学模型中使用的向量上计算 softmax 的方法。

### 1\. Tensorflow

您可以使用`tensorflow.nn.softmax`计算矢量上的 softmax，如图所示。

```py
import tensorflow as tf
import numpy as np

vector = np.array([5.5, -13.2, 0.5])

probabilities = tf.nn.softmax(vector).numpy()

print("Probability Distribution is:")
print(probabilities)

```

```py
Probability Distribution is:
[9.93307142e-01 7.51236614e-09 6.69285087e-03]

```

### 2.我的天啊

[Scipy](https://www.askpython.com/python-modules/python-scipy) 库可用于使用如下所示的`scipy.special.softmax`计算 softmax。

```py
import scipy
import numpy as np

vector = np.array([1.5, -3.5, 2.0])
probabilities = scipy.special.softmax(vector)
print("Probability Distribution is:")
print(probabilities)

```

```py
Probability Distribution is:
[0.3765827  0.00253739 0.62087991]

```

### 3\. PyTorch

您可以使用 [Pytorch](https://www.askpython.com/python-modules/pytorch) `torch.nn.Softmax(dim)`来计算 softmax，指定您想要计算的尺寸，如图所示。

```py
import torch

vector = torch.tensor([1.5, -3.5, 2.0])
probabilities = torch.nn.Softmax(dim=-1)(vector)
print("Probability Distribution is:")
print(probabilities)

```

```py
Probability Distribution is:
tensor([0.3766, 0.0025, 0.6209])

```

## 结论

恭喜你！！现在，您已经了解了 softmax 函数以及如何使用各种方式来实现它，您可以使用它来解决机器学习中的多类分类问题。

感谢阅读！！