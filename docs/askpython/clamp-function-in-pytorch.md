# PyTorch 中的 Clamp()函数–完整指南

> 原文：<https://www.askpython.com/python/examples/clamp-function-in-pytorch>

各位编码员，你们好吗？因此，在本教程中，我们将尝试使用 **PyTorch clamp()函数**。我们将从理论和实践两个角度来看待它。

让我们开始吧。

* * *

## Python PyTorch 中 clamp()介绍

**clamp()函数**用于约束指定范围内的数值。这意味着什么？

首先，让我们搞清楚。

假设你已经得到了一个从 60 到 110 的数字范围，你正在寻找数字 85(T2，T3)。因此，clamp()函数将其值限制为 85。在这种情况下，85 介于 60 和 110 之间，很容易计算。

但是，如果你选择了 **35** ，你就在范围之外了。在这种情况下，它被限制为 60，因为它最接近下限，而不是在范围的中间。

同样，如果输入一个大于 110 的数，比如 **132** ，它会返回 110，因为 132 接近最大限制，也就是 110。

* * *

## 在 PyTorch 中实现 clamp()函数

让我们开始在 PyTorch 中实现 clamp()函数。

### 使用 clamp()功能

Python clamp 功能没有内置到语言中，但可以使用以下代码进行定义:

```py
def clamp_fucntion (no , min_no , max_no ):
        n = max(min(no, max_no), min_no)
        return n
print( "Find 10 in 20 to 30 : ", clamp_fucntion(10 ,20 ,30) )
print( "Find 25 in 20 to 30 : ", clamp_fucntion(25 ,20 ,30 ) )
print( "Find 115  in 20 to 30 : ",  clamp_fucntion(115 ,20 ,30 ) )

```

```py
Find 10 in 20 to 30 :  20
Find 25 in 20 to 30 :  25
Find 115  in 20 to 30 :  30

```

还有一些实现**箝位功能**的其他方法。让我们在下一节看看其中的一些。

### pyker 夹点()

然而，虽然这个函数在核心 Python 中并不常用，但它在许多 Python 库中被广泛使用，比如 Pytorch 和 Wand ImageMagick 库。

此外，这个函数已经包含在这些库中。您只需要导入它并根据需要使用它。

让我们来看一些例子。

```py
import torch

T = torch.FloatTensor([3,12,15,18,21])
print("Input Tensor: ", T)

output = torch.clamp(T,min=10,max=20)
print("Output Tensor: ",output)

```

```py
Input Tensor:  tensor([ 3., 12., 15., 18., 21.])
Output Tensor:  tensor([10., 12., 15., 18., 20.])

```

* * *

## **结论**

恭喜你！您刚刚学习了 Clamp 函数及其在 Python 中的实现。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Numpy average()函数——简要概述](https://www.askpython.com/python-modules/numpy/numpy-average-function)
2.  [Pandas isin()函数-完整指南](https://www.askpython.com/python-modules/pandas/pandas-isin)
3.  [Python 中的 4 个激活函数要知道！](https://www.askpython.com/python/examples/activation-functions-python)
4.  [Python 中损失函数概述](https://www.askpython.com/python/examples/loss-functions)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *