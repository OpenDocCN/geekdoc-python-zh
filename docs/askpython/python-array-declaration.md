# Python 数组声明

> 原文：<https://www.askpython.com/python/array/python-array-declaration>

嘿，读者们。希望你们都过得好。在本文中，我们将主要关注 **Python 数组声明**的变体。

* * *

## 什么是 Python 数组？

众所周知，Python 提供了各种数据结构来操作和处理数据值。

当谈到作为数据结构的数组时，Python 没有提供创建或使用数组的直接方法。相反，它为我们提供了以下数组变体:

*   [Python 数组模块](https://www.askpython.com/python/array/python-array-examples):数组模块包含各种创建和处理值的方法。
*   [Python List](https://www.askpython.com/python/list/python-list) : List 可以认为是一个动态数组。此外，与数组不同，异构元素可以存储在列表中。
*   Python NumPy 数组:NumPy 数组最适合对大量数据执行数学运算。

了解了 Python 数组之后，现在让我们来了解在 Python 中声明数组的方法。

* * *

## Python 数组声明–Python 数组的变体

在下一节中，我们将了解使用 Python 数组的变体来声明数组的技术。

* * *

### 类型 1: Python 数组模块

`Python Array module` 包含 `array() function`，使用它我们可以在 python 环境中创建一个数组。

**语法:**

```py
array.array('format code',[data])

```

*   `format_code`:表示数组接受的元素类型。代码“I”代表数值。

**举例:**

```py
import array
arr = array.array('i', [10,20,30,40,50])
print(arr)

```

**输出:**

```py
array('i', [10, 20, 30, 40, 50])

```

* * *

### 类型 2:作为数组的 Python 列表

`Python list`可以用来动态创建和存储像数组一样的元素。

**语法:**

```py
list = [data]

```

**举例:**

```py
lst = [10,20,30,40, 'Python']
print(lst)

```

**输出:**

```py
[10, 20, 30, 40, 'Python']

```

如上所述，不同数据类型的元素可以一起存储在 List 中。

* * *

### 类型 3: Python NumPy 数组

`NumPy module`包含各种创建和使用数组作为数据结构的函数。

在 Python 中,`numpy.array() function`可以用来创建一维和多维数组。它创建一个数组对象作为“ndarray”。

```py
np.array([data])

```

**示例:使用 numpy.array()函数创建数组**

```py
import numpy
arr = numpy.array([10,20])
print(arr)

```

**输出:**

```py
[10 20]

```

此外，我们可以使用`numpy.arange() function`在数据值的特定范围内创建一个数组。

```py
numpy.arange(start,stop,step)

```

*   `start`:数组的开始元素。
*   `end`:数组的最后一个元素。
*   `step`:数组元素之间的间隔或步数。

**举例:**

```py
import numpy
arr = numpy.arange(1,10,2)
print(arr)

```

**输出:**

```py
[1 3 5 7 9]

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

* * *

## 参考

*   [StackOverflow — Python 数组声明](https://stackoverflow.com/questions/1514553/how-do-i-declare-an-array-in-python)