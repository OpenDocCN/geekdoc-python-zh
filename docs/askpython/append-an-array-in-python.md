# 如何在 Python 中追加数组？

> 原文：<https://www.askpython.com/python/array/append-an-array-in-python>

嘿，伙计们！在本文中，我们将关注在 Python 中添加数组的**方法。**

* * *

## Python 数组是什么？

在编程术语中，数组是一个存储相似类型元素的线性数据结构。

众所周知，Python 并没有为我们提供一种特定的数据类型——“数组”。相反，我们可以使用 Python Array 的以下变体——

*   [Python List](https://www.askpython.com/python/list/python-list) :包含了一个数组的所有功能。
*   [Python Array](https://www.askpython.com/python/array/python-array-examples) 模块:这个模块用来创建一个数组，并用指定的函数操作数据。
*   [Python NumPy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays):NumPy 模块创建一个数组，用于数学目的。

现在，让我们来理解将元素附加到 Python 数组的上述变体的方法。

* * *

## 使用 Append()函数在 Python 中追加数组

`Python append() function`允许我们将一个元素或一个数组添加到另一个数组的末尾。也就是说，指定的元素被追加到输入数组的末尾。

根据上面提到的 Python 数组的变体，append()函数具有不同的结构。

现在让我们了解 Python append()方法在 Python 数组的每个变体上的功能。

* * *

## 变体 1:带有列表的 Python append()函数

列表被认为是动态数组。Python append()方法可以在这里构造，以将元素添加/追加到列表的末尾。

**语法:**

```py
list.append(element or list)

```

列表或元素被添加到列表的末尾，列表用添加的元素更新。

**举例:**

```py
lst = [10,20,30,40] 
x = [0,1,2] 
lst.append(x) 
print(lst) 

```

**输出:**

```py
[10, 20, 30, 40, [0, 1, 2]]

```

* * *

## 变体 2:带有数组模块的 Python append()方法

我们可以使用 array 模块创建一个数组，然后应用 append()函数向其中添加元素。

**使用数组模块初始化 Python 数组:**

```py
import array
array.array('unicode',elements)

```

*   `unicode`:表示数组要占用的元素类型。例如，“d”表示双精度/浮点元素。

此外，append()函数的操作方式与 Python 列表相同。

**举例:**

```py
import array 
x = array.array('d', [1.4, 3.4])
y = 10
x.append(y)
print(x)

```

**输出:**

```py
array('d', [1.4, 3.4, 10.0])

```

* * *

## 变体 3:带有 NumPy 数组的 Python append()方法

NumPy 模块可用于创建一个数组，并根据各种数学函数操作数据。

**语法:Python numpy.append()函数**

```py
numpy.append(array,value,axis)

```

*   `array`:要追加数据的 numpy 数组。
*   `value`:要添加到数组中的数据。
*   `axis`(可选):指定按行或按列操作。

在下面的示例中，我们使用 numpy.arange()方法在指定的值范围内创建了一个数组。

**举例:**

```py
import numpy as np 

x = np.arange(3) 
print("Array x : ", x) 

y = np.arange(10,15) 
print("\nArray y : ", y) 

res = np.append(x, y)
print("\nResult after appending x and y: ", res) 

```

**输出:**

```py
Array x :  [0 1 2]

Array y :  [10 11 12 13 14]

Result after appending x and y:  [ 0  1  2 10 11 12 13 14]

```

* * *

## 结论

这个题目到此为止。如果你有任何疑问，欢迎在下面评论。更多关于 Python 的帖子，请访问 [【邮件保护】](https://www.askpython.com/) 。

* * *

## 参考

*   Python 添加到数组— JournalDev
*   NumPy append()方法— JournalDev