# Python np.argmax()函数

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-argmax

NumPy (np)是最流行的数学和科学计算库之一。它提供了许多处理多维数组的函数。在本文中，我们将重点介绍 **Python np.argmax()函数**。

* * *

## Python np.argmax()函数

顾名思义， **argmax()** 函数返回 NumPy 数组中最大值的索引。如果有多个索引具有相同的最大值，将返回第一个索引。

**argmax()语法:**

np.argmax( *a* **，** *axis=None* **，** *out=None* **，** *** **，***keep dims =<no value>***)**

第一个参数是输入数组。如果没有提供轴，数组**变平**，然后返回最大值的索引。

如果我们指定 ***轴*** ，它返回沿给定轴的索引值。

第三个参数用于传递数组参数来存储结果，它应该具有正确的形状和数据类型才能正常工作。

如果 ***keepdims*** 被传为真，则被缩减的轴作为尺寸为 1 的尺寸留在结果中。

让我们看一些使用 argmax()函数的例子来正确理解不同参数的用法。

* * *

### 1.使用 np.argmax()找到最大值的索引

```py
>>> import numpy as np
>>> arr = np.array([[4,2,3], [1,6,2]])
>>> arr
array([[4, 2, 3],
       [1, 6, 2]])
>>> np.ndarray.flatten(arr)
array([4, 2, 3, 1, 6, 2])
>>> np.argmax(arr)
4

```

np.argmax()返回 4，因为数组首先被展平，然后返回最大值的索引。因此，在这种情况下，最大值为 6，其在展平数组中的索引为 4。

但是，我们希望索引值在一个普通的数组中，而不是扁平的数组中。所以，我们必须使用***【arg max()***和***underline _ index()***函数来获得正确格式的索引值。

```py
>>> np.unravel_index(np.argmax(arr), arr.shape)
(1, 1)
>>>

```

* * *

### 2.沿着轴寻找最大值的索引

如果您想要沿不同轴的最大值的索引，请传递轴参数值。如果我们传递 axis=0，则返回列中最大值的索引。对于轴=1，返回沿行最大值的索引。

```py
>>> arr
array([[4, 2, 3],
       [1, 6, 2]])
>>> np.argmax(arr, axis=0)
array([0, 1, 0])
>>> np.argmax(arr, axis=1)
array([0, 1])

```

对于轴= 0，第一列值是 4 和 1。所以最大值索引为 0。类似地，对于第二列，值是 2 和 6，因此最大值索引是 1。对于第三列，值为 3 和 2，因此最大值索引为 0。这就是为什么我们得到的输出是一个数组([0，1，0])。

对于轴= 1，第一行值是(4，2，3)，因此最大值索引是 0。对于第二行，值为(1，6，2)，因此最大值索引为 1。因此输出数组([0，1])。

* * *

### 3.使用具有多个最大值的 np.argmax()

```py
>>> import numpy as np
>>> arr = np.arange(6).reshape(2,3)
>>> arr
array([[0, 1, 2],
       [3, 4, 5]])
>>> arr[0][1] = 5
>>> arr
array([[0, 5, 2],
       [3, 4, 5]])
>>> np.argmax(arr)
1
>>> arr[0][2] = 5
>>> arr
array([[0, 5, 5],
       [3, 4, 5]])
>>> np.argmax(arr)
1
>>> np.argmax(arr, axis=0)
array([1, 0, 0])
>>> np.argmax(arr, axis=1)
array([1, 2])
>>> 

```

我们使用 [arange()函数](https://www.askpython.com/python-modules/numpy/numpy-arange-method-in-python)创建一个带有一些默认值的 2d 数组。然后，我们更改其中一个值，使多个索引具有最大值。从输出中可以清楚地看到，当有多个位置具有最大值时，返回最大值的第一个索引。

* * *

## 摘要

NumPy argmax()函数很好理解，只需要记住在找到最大值的索引之前，数组是展平的。此外，axis 参数对于查找行和列的最大值的索引非常有帮助。

## 下一步是什么？

*   [NumPy 教程](https://www.askpython.com/python-modules/numpy/python-numpy-module)
*   [南在 NumPy](https://www.askpython.com/python/examples/nan-in-numpy-and-pandas)
*   [NumPy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)
*   [NumPy reshape()函数](https://www.askpython.com/python-modules/numpy/python-numpy-reshape-function)
*   [NumPy 排序数组](https://www.askpython.com/python/sorting-techniques-in-numpy)

## 资源

*   [正式文件](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)