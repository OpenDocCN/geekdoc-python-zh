# 如何存进去？npy 格式？

> 原文：<https://www.askpython.com/python-modules/numpy/save-in-npy-format>

遇到过. npy 文件吗？在这篇文章中，我们将回顾保存为 npy 格式的步骤。NPY 是 Numpy 的二进制数据存储格式。

[Numpy](https://www.askpython.com/python-modules/numpy/python-numpy-module) 是高效执行数据科学操作的必备模块。数据的导入、保存和处理占据了数据科学领域的大部分时间。在导入和导出数据时，CSV 文件是一个不错的选择。

但是，有时您需要保存数据，以便在 Python 中再次使用。对于这种情况，Numpy 提供了。npy 格式。

**导入和导出数据。与其他选项相比，npy 文件更有效。**

Numpy 提供了 [numpy.save()](https://numpy.org/doc/stable/reference/generated/numpy.save.html) 方法，允许您将文件保存到。npy 格式。它只允许您保存数组格式的数据。它在保存之前将数组转换为二进制文件。最终保存的是这个二进制文件。

在本教程中，我们将使用 numpy 数组并保存在。npy 格式。接下来我们还将导入该文件。

让我们开始吧。

## 使用 Numpy save()以 npy 格式保存

让我们从创建一个示例数组开始。

```py
import numpy as np 
arr = np.arange(10)
print("arr :) 
print(arr)

```

将此数组保存到。npy 文件，我们将使用。Numpy 中的 save()方法。

```py
np.save('ask_python', arr)
print("Your array has been saved to ask_python.npy")

```

运行这行代码会将数组保存到一个名为 *'ask_python.npy'* 的二进制文件中。

**输出:**

```py
arr:
[0 1 2 3 4 5 6 7 8 9 10]
Your array has been saved to ask_python.npy

```

## 进口。Python 中的 npy 文件

为了将数据加载回 python，我们将使用。Numpy 下的 load()方法。

```py
data = np.load('ask_python.npy')
print("The data is:")
print(data)

```

输出结果如下:

```py
The data is:
[0 1 2 3 4 5 6 7 8 9 10]

```

## 结论

本教程是关于将 Python 中数组的数据保存到. npy 二进制文件中，并将其加载回 Python。希望你和我们一起学习愉快！