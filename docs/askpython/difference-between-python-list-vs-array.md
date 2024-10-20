# Python 列表与数组——要知道的 4 个区别！

> 原文：<https://www.askpython.com/python/difference-between-python-list-vs-array>

嘿，伙计们！希望你们都过得好。在本文中，我们将详细关注 Python 列表和数组之间的**差异。**

* * *

Python 列表和 Python 数组之间的主要区别在于，列表是 Python 标准包的一部分，而对于数组，需要导入“数组”模块。Python 中的列表用一些例外情况代替了数组数据结构。

## 1.列表和数组如何存储数据

众所周知，数据结构是用来有效存储数据的。

在这种情况下，一个[列表](https://www.askpython.com/python/list/python-list)可以将异构数据值存储到其中。也就是说，不同数据类型的数据项可以容纳在一个 Python 列表中。

**举例:**

```py
lst = [1,2,3,4,'Python']
print(lst)

```

**输出:**

```py
[1,2,3,4,'Python']

```

另一方面，[数组](https://www.askpython.com/python/array/python-array-examples)存储同类元素，即它们存储属于同一类型的元素。

**举例:**

```py
import array

arr = array.array('i', [10, 20, 30, 40])
print(arr)

```

**输出:**

```py
array('i', [10, 20, 30, 40])

```

* * *

## 2.数组与列表的声明

Python 将“列表”作为内置的数据结构。这就是为什么列表不需要在 Python 中声明。

```py
lst = [1, 2, 3, 4]

```

另一方面，Python 中的数组需要声明。我们可以使用以下技术声明一个数组:

**阵列模块**

```py
import array
array-name = array.array('format-code', [elements])

```

**NumPy 模块**

```py
import numpy
array-name = numpy.array([elements])

```

* * *

## 3.使用数组的高级数学运算

在执行数学运算时，数组占了上风。 [NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)为我们提供了数组结构来存储数据值并方便地操作它们。

**数组示例:**

```py
import numpy
arr = numpy.array([1,2,3,4])
pr = arr*5
print(pr)

```

**输出:**

```py
[ 5 10 15 20]

```

与列表不同，在列表中执行的操作不会反映到结果中，如下面的列表操作示例所示。

在这里，我们尝试将常量值(5)与列表相乘，这不会反映输出中的任何内容。因为列表不能对任何数据值进行直接的数学运算。

所以，如果我们想用列表中的元素乘以 5，我们必须分别用列表中的每个元素乘以 5。

**列表示例:**

```py
lst = [1,2,3,4]
pr = lst*5
print(lst)

```

**输出:**

```py
[1, 2, 3, 4]

```

* * *

## 4.调整数据结构的大小

Python 列表是一种内置的数据结构，可以非常容易和有效地调整大小。

另一方面，在调整数组内存大小时，数组的性能很差。相反，我们将不得不把数组复制到另一个数组中来缩放和调整它的大小。

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

在那之前，学习愉快！！

* * *

## 参考

*   [Python 数组 vs List — StackOverflow](https://stackoverflow.com/questions/176011/python-list-vs-array-when-to-use)