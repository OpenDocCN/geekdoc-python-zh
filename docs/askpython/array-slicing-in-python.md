# 理解 Python 中的数组切片

> 原文：<https://www.askpython.com/python/array/array-slicing-in-python>

## 介绍

在本教程中，我们将了解 Python 中**数组切片的概念。**

## 数组切片

**Python** 支持数组切片。它是在用户定义的起始和结束索引的基础上，从给定的数组创建一个新子数组。我们可以通过以下任一种方法对数组进行切片。

遵循 Python 切片方法可以很容易地完成数组切片。其语法如下所示。

```py
arr[ start : stop : step ]

```

同样，Python 还提供了一个名为 [slice()](https://www.askpython.com/python/built-in-methods/python-slice-function) 的函数，该函数返回一个包含要切片的索引的 **slice** 对象。下面给出了使用该方法的语法。

```py
slice(start, stop[, step])

```

对于这两种情况，

*   **start** 是我们需要对数组 arr 进行切片的起始索引。默认设置为 0，
*   **stop** 是结束索引，在此之前切片操作将结束。默认情况下等于数组的长度，
*   **步骤**是切片过程从开始到停止的步骤。默认设置为 1。

## Python 中的数组切片方法

现在我们知道了使用这两种方法的语法，让我们看一些例子，并试着理解**切片过程**。

在下面的例子中，我们将考虑数组模块中的两个数组以及 [NumPy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)。

## 1.只有一个参数

**开始**、**停止**和**步进**的默认值分别等于 0、数组长度和 1。因此，指定 start 或 stop 中的一个，我们就可以对数组进行切片。

让我们看看如何。

```py
import array
import numpy as np

#array initialisation
array_arr= array.array('i',[1,2,3,4,5])
np_arr = np.array([6,7,8,9,10])

#slicing array with 1 parameter
print("Sliced array: ", array_arr[:3])
print("Sliced NumPy array: ", np_arr[:4])

```

**输出**:

```py
Sliced array:  array('i', [1, 2, 3])
Sliced NumPy array:  [6 7 8 9]

```

这里，我们已经初始化了两个数组，一个来自`array`模块，另一个来自`NumPy`数组。输出中显示了使用一个参数对它们进行切片的结果。正如我们所看到的，对于这两种情况，**开始**和**步骤**被默认设置为 **0** 和 **1** 。切片数组包含索引为 **0** 到 **(stop-1)** 的元素。这是 Python 中最快的数组切片方法之一。

## 2.Python 中带两个参数的数组切片

同样，指定 start、stop 和 end 中的任意两个参数，可以通过考虑第三个参数的默认值来执行 Python 中的数组切片。

让我们举一个例子。

```py
import array
import numpy as np

#array initialisation
array_arr= array.array('i',[1,2,3,4,5])
np_arr = np.array([6,7,8,9,10])

#slicing array with 2 parameters
print("Sliced array: ", array_arr[2:5])
print("Sliced NumPy array: ", np_arr[1:4])

```

**输出**:

```py
Sliced array:  array('i', [3, 4, 5])
Sliced NumPy array:  [7 8 9]

```

在这种情况下，分片的`array`模块数组和 **`NumPy`** 数组也包含指定为**开始**到`(stop-1)`的索引的元素，其中步长设置为 **1** 。因此输出是合理的。

## 3.使用步长参数

当提到所有这三个参数时，你可以在 Python 中从索引**开始**到 **(stop-1)** 执行数组切片，每个索引跳转等于给定的**步**。

看看下面的例子就有了清晰的认识。

```py
import array
import numpy as np

#array initialisation
array_arr= array.array('i',[1,2,3,4,5,6,7,8,9,10])
np_arr = np.array([11,12,13,14,15,16,17,18,19,20])

#slicing array with step parameter
print("Sliced array: ", array_arr[1:8:2])
print("Sliced NumPy array: ", np_arr[5:9:3])

```

**输出**:

```py
Sliced array:  array('i', [2, 4, 6, 8])
Sliced NumPy array:  [16 19]

```

类似地，这里我们从给定索引 **start** 到 **stop-1** 的数组中得到切片数组的值。这里唯一的区别是步长值，这次对于`array`模块数组和`NumPy`数组分别指定为 **2** 和 **3** 。因此，这次每个步进跳转都是给定**步骤**的值。

## 4.Python 中使用 slice()方法的数组切片

Python 中的`slice()`方法使用给定的**步骤**值返回一系列索引，范围从**开始**到**停止-1** 。

与前面的情况类似，这里 start 和 stop 的默认值也是 0，步长等于 1。

```py
import array
import numpy as np

#array initialisation
array_arr = array.array('i',[1,2,3,4,5,6,7,8,9,10])
np_arr = np.array([11,12,13,14,15,16,17,18,19,20])

s = slice(3,9,3)

#slicing array with slice()
print("Sliced array: ", array_arr[s])
print("Sliced NumPy array: ", np_arr[s])

```

**输出**:

```py
Sliced array:  array('i', [4, 7])
Sliced NumPy array:  [14 17]

```

这里，首先我们初始化了两个数组，一个来自`array`模块，另一个来自`NumPy`模块。`slice()`方法的开始、停止和步进分别称为 **3** 、 **9** 和 **3** 。因此，当我们将这个序列`s`传递给数组时，我们得到切片数组，其值包含索引 **3** 和 **6** 处的元素。

因此，输出是合理的。

**注意**:原始数组总是保持完整，保持不变。如果需要，切片数组可以存储在某个变量中。

## 结论

所以在本教程中，我们要学习 Python 中的**数组切片**的概念。任何进一步的问题，请在下面随意评论。

## 参考

*   [Python 数组教程](https://www.askpython.com/python/array/python-array-examples)，
*   [NumPy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)在 Python 中，
*   [Python slice()函数](https://www.askpython.com/python/built-in-methods/python-slice-function)。