# 如何在 Python 中求数组长度

> 原文：<https://www.askpython.com/python/array/array-length-in-python>

嘿，伙计们！我希望你们都过得很好。在本文中，我们将揭示 Python 中数组长度的 **3 种变体。**

众所周知，Python 并不直接支持或为我们提供数组数据结构。相反，Python 在这里为我们提供了使用数组数据结构的 3 种不同变体。

让我们先来看看创建 Python 数组的不同方法。

此外，在接下来的章节中，我们将讨论如何使用 Python len()方法来获取每个变量中数组的长度。

* * *

## 使用 len()方法在 Python 中查找数组长度

Python 为我们提供了以下形式的数组数据结构:

*   [Python 列表](https://www.askpython.com/python/list/python-list)
*   [Python 数组模块](https://www.askpython.com/python/array/initialize-a-python-array)
*   NumPy 模块

我们可以使用上面的任何变体创建一个数组，并使用不同的函数来处理和操作数据。

`Python len() method`使我们能够找到数组/对象中元素的总数。也就是说，它返回数组/对象中元素的计数。

**语法:**

```py
len(array)

```

现在让我们来了解一下 Python array 的上述变种中求数组长度的方法。

* * *

## 查找 Python 列表的长度

`Python len() method`可与列表一起使用，获取并显示列表中元素的数量。

在下面的例子中，我们创建了一个异构元素列表。此外，我们使用了 len()方法来显示列表的长度。

```py
lst = [1,2,3,4,'Python']
print("List elements: ",lst)
print("Length of the list:",len(lst))

```

**输出:**

```py
List elements:  [1, 2, 3, 4, 'Python']
Length of the list: 5

```

* * *

## 查找 Python 数组的长度

`Python Array module`使用模块的各种功能帮助我们创建数组并操纵数组。len()方法可以用来计算数组的长度。

```py
import array as A 
arr = A.array('i',[1,2,3,4,5])
print("Array elements: ",arr)
print("Length of array:",len(arr))

```

```py
Array elements:  array('i', [1, 2, 3, 4, 5])
Length of array: 5

```

* * *

## 寻找 Python NumPy 数组的长度

众所周知，我们可以使用`NumPy module`创建一个数组，并将其用于任何数学目的。len()方法帮助我们找出 NumPy 数组中存在的数据值的数量。

```py
import numpy as np
arr = np.arange(5)
len_arr = len(arr)
print("Array elements: ",arr)
print("Length of NumPy array:",len_arr)

```

**输出:**

```py
Array elements:  [0 1 2 3 4]
Length of NumPy array: 5

```

* * *

## 结论

这个话题到此结束。如果你遇到任何问题，欢迎在下面评论。在那之前，学习愉快！

* * *

## 参考

*   Python len() with List — JournalDev