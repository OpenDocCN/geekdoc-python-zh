# 如何使用 Python sum()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-sum-method>

## 介绍

在本教程中，我们将讨论 Python sum()方法。

Python `sum()`方法是一个内置方法，它返回传递的 iterable 的所有元素的总和。

## Python sum()函数

让我们看看在 Python 中使用 sum()方法的语法。

```py
sum(iterable[, start])

```

这里，

*   **iterable** 可以是任何包含我们需要计算总和的值的 iterable 对象。它可以是列表、元组或字典，
*   **start** 是将要进行加法运算的初始值。默认情况下，起始值设置为 **0** 。

现在让我们看一些例子来清楚地理解这个方法的使用和工作。

## 对列表、元组、复数、浮点数等使用 sum()。

如前所述，`sum()`方法可以计算以 list、 [tuple](https://www.askpython.com/python/tuple/python-tuple) 或 [dictionary](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial) 即任意 **iterable** 形式传递的值的总和。但条件是，iterable 应该包含某个值，否则，会引发一个**错误**。

让我们通过寻找整数列表、元组列表、字典列表和复数列表的元素的和来熟悉该方法。

```py
#using built-in sum() in Python

list1=[1,2,3,4,5]
tuple1=(10,20,30)
dict1={ 0:11, 1:22, 2:33, 3:44 }
complx_nums=[ 4+3j, 7+5j, 8+3j]

print("sum of list elements:",sum(list1,5))
print("sum of tuple elements:", sum(tuple1))
print("sum of dictionary keys:", sum(dict1))
print("sum of complex numbers:", sum(complx_nums))

```

**输出**:

```py
sum of list elements: 20
sum of tuple elements: 60
sum of dictionary keys: 6
sum of complex numbers: (19+11j)

```

这里，

*   我们首先用一些值初始化**列表 1** 、**元组 1** 、**字典 1** 和 **complx_num** ，
*   然后我们简单地将这些 iterable 分别传递给`sum()`方法，
*   为了计算**列表 1** 元素的总和，我们设置了一个起始值 **5** ，对于其余的，我们不传递任何起始参数(默认设置为 **0** )。

因此，我们得到了期望的输出。对于每种情况，我们得到各自的可迭代元素的和。

## Python sum()与 NumPy sum()

Python 中的 **NumPy** 模块中定义了一个`sum()`方法。它用于计算 NumPy 数组元素的和。但是，这个方法也可以找到 python 中任何其他包含一些值的 iterable 的元素之和。

让我们比较一下 Python 中对于一些可迭代对象的`sum()`和`numpy.sum()`的输出。

```py
#numpy sum() vs python sum()

import numpy as np

list1=[1,2,3,4,5]
np_arr = np.array([1,2,3,4,5])

print("Built-in sum() result:",sum(list1))
print("numpy sum() for np array:",np.sum(np_arr))
print("numpy sum() for list:",np.sum(list1))

```

**输出**:

```py
Built-in sum() result: 15
numpy sum() for np array: 15
numpy sum() for list: 15

```

正如我们所见，`sum()`和`np.sum()`方法为列表 **list1** 返回相同的结果。

对于计算列表、元组和字典等可迭代对象的总和，内置的`sum()`方法比 **numpy 的** `sum()`方法更快更容易使用。

但是当涉及到 **numpy 数组**时，`np.sum()`方法似乎更快更可靠。这仅仅是因为 NumPy 使用了一种**矢量化**的方法。

## 结论

因此，在本教程中，我们了解了 **Python** 中内置的`sum()`方法，以及 Python sum()和 numpy.sum()方法之间的比较。我希望你现在对这个方法有一个清楚的了解。

对于任何进一步的问题，请随意共同以下。

## 参考

*   Python 文档中关于 [sum()的 Python](https://docs.python.org/3/library/functions.html#sum) ，
*   [Python 的 sum vs. NumPy 的 NumPy . sum](https://stackoverflow.com/questions/10922231/pythons-sum-vs-numpys-numpy-sum)–stack overflow 问题，
*   python sum()–日志开发帖子。