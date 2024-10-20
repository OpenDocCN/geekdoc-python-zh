# Python 中的无穷大–将 Python 变量值设置为无穷大

> 原文：<https://www.askpython.com/python/examples/infinity-in-python>

一个简单的数字不能代表你的数据集？在 Python 中把你的变量值设置为无穷大怎么样？今天我们要说的就是这个！

在用 Python 编码时，我们经常需要用一个大的正值或大的负值来初始化一个变量。这在比较变量以计算集合中的最小值或最大值时非常常见。

**Python 中的正无穷大**被认为是最大的正值，**负无穷大**被认为是最大的负数。

在本教程中，我们将学习用正负无穷大初始化变量的三种方法。除此之外，我们还将学习如何检查一个变量是否无穷大，并对这些变量执行一些算术运算。

让我们开始吧。

## 在 Python 中用无穷大初始化浮点变量

在不使用任何模块的情况下，将变量设置为正无穷大或负无穷大的最简单方法是使用 [float](https://www.askpython.com/python/built-in-methods/python-float-method) 。

您可以使用以下代码行将变量设置为正无穷大:

```py
p_inf = float('inf')

```

要打印变量值，请使用:

```py
print('Positive Infinity = ',p_inf)

```

输出:

```py
Positive Infinity =  inf

```

要用负无穷大初始化变量，请使用:

```py
n_inf = float('-inf')
print('Negative Infinity = ',n_inf)

```

输出:

```py
Negative Infinity =  -inf

```

## 使用 Numpy 模块初始化具有无穷大的变量

你也可以使用流行的 [Numpy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-module)来初始化具有正或负无穷大的变量。

让我们从导入 Numpy 模块开始。

```py
import Numpy as np

```

现在，我们可以使用模块将变量初始化为正无穷大，如下所示:

```py
p_inf = np.inf
print('Positive Infinity = ',p_inf)

```

输出结果如下:

```py
Positive Infinity =  inf

```

要用负无穷大初始化变量，请使用:

```py
n_inf = -np.inf
print('Negative Infinity = ',n_inf)

```

输出结果如下:

```py
Negative Infinity =  -inf

```

### 完全码

下面给出了本节的完整代码。

```py
import Numpy as np

#positive inf
p_inf = np.inf
print('Positive Infinity = ',p_inf)

#negative inf
n_inf = -np.inf
print('Negative Infinity = ',n_inf)

```

## 使用数学模块在 Python 中用无穷大初始化变量

将变量初始化为无穷大的第三种方法是使用 python 中的[数学模块](https://www.askpython.com/python-modules/python-math-module)。

让我们从导入模块开始。

```py
import math

```

要使用数学模块将变量设置为正无穷大，请使用以下代码行:

```py
p_inf = math.inf
print('Positive Infinity = ',p_inf)

```

输出:

```py
Positive Infinity =  inf

```

要使用数学模块将变量设置为负无穷大，请使用以下代码行:

```py
n_inf = -math.inf
print('Negative Infinity = ',n_inf)

```

输出:

```py
Negative Infinity =  -inf

```

除此之外，Math 模块还给出了一个方法，让你检查一个变量是否被设置为无穷大。

您可以使用下面一行代码来检查它:

```py
math.isinf(p_inf)

```

输出:

```py
True

```

```py
math.isinf(n_inf) 

```

输出:

```py
True 

```

### 完全码

本节的完整代码如下所示:

```py
import math

#positive inf
p_inf = math.inf
print('Positive Infinity = ',p_inf)

#negative inf
n_inf = -math.inf
print('Negative Infinity = ',n_inf)

#check
print(math.isinf(p_inf))
print(math.isinf(n_inf))

```

## Python 中关于无穷大的算术运算

让我们试着用设置为正负无穷大的变量做一些算术运算。

### 1.对无穷大值的加法运算

让我们看看当我们把一个数加到正无穷大和负无穷大时会发生什么。

```py
a = p_inf + 100
print(a)

```

输出:

```py
inf

```

```py
a = n_inf + 100
print(a)

```

输出:

```py
-inf

```

### 2.无穷大值的减法

让我们试着从正负无穷大中减去一个数。

```py
a = p_inf - 100
print(a)

```

输出:

```py
inf

```

```py
a = n_inf - 100
print(a)

```

输出:

```py
-inf

```

我们可以看到，在无穷大上加上或减去一个值没有任何影响。

让我们看看当我们在两个无穷大之间执行算术运算时会发生什么。

### 3.两个无穷大之间的算术运算

让我们试着把正负无穷大相加，看看结果。

```py
a = p_inf + n_inf 
print(a)

```

输出:

```py
nan

```

我们得到的输出是 **nan** ，它是**的缩写，而不是一个数字**。因此，我们可以说这个操作没有被很好地定义。要了解南更多，请阅读这个[教程。](https://www.askpython.com/python/examples/nan-in-numpy-and-pandas)

让我们看看正负无穷大相乘会发生什么。

```py
a = p_inf * n_inf 
print(a)

```

输出:

```py
-inf

```

我们得到负的无穷大，因为数值相乘，符号是负的，因为负的无穷大。

让我们看看当我们在正负无穷大之间进行除法运算时会发生什么。

```py
a = p_inf / n_inf 
print(a)

```

输出:

```py
nan

```

无限除以无限也没有定义，因此我们得到 nan。

## 结论

本教程是关于用正负无穷大初始化变量的不同方法。我们还讲述了一些涉及正负无穷大的算术运算。