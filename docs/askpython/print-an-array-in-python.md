# 如何在 Python 中打印数组

> 原文：<https://www.askpython.com/python/array/print-an-array-in-python>

## 介绍

在本教程中，我们将学习如何用 Python 打印一个数组。

所以在我们进入主题之前，让我们了解一下 Python 中的**数组**。

## Python 数组

[数组](https://www.askpython.com/python/array/python-array-examples)是同名同类型数据元素的集合。在 Python 中，我们可以使用**列表**或 **NumPy** 模块来实现数组。NumPy 模块为我们提供了类型为`ndarray` (NumPy Array)的数组。

此外，阵列可以是多维的。我们知道，多维数组最简单的形式是[二维数组](https://www.askpython.com/python/two-dimensional-array-in-python)。因此，在本教程中，我们将同时考虑 1D 和 2D 阵列。

## 用 Python 打印数组的方法

现在，让我们看看用 Python 打印 1D 和 2D 数组的一些方法。**注意**:这些数组将使用列表来实现。

### 使用 print()方法直接打印

我们可以直接将包含要打印的值的**数组** (list)的名称传递给 Python 中的`print()`方法进行打印。

但是在这种情况下，数组以列表**的形式打印出来，即带有括号和逗号分隔的值。**

```py
arr = [2,4,5,7,9]
arr_2d = [[1,2],[3,4]]

print("The Array is: ", arr) #printing the array
print("The 2D-Array is: ", arr_2d) #printing the 2D-Array

```

**输出**:

```py
The Array is:  [2, 4, 5, 7, 9]
The 2D-Array is:  [[1, 2], [3, 4]]

```

这里，`arr`是一维数组。而`arr_2d`是二维的。我们直接将它们各自的名字传递给`print()`方法，分别以**列表**和**列表列表**的形式打印出来。

### 在 Python 中使用 for 循环

我们也可以用 Python 打印一个数组，方法是使用`for`循环遍历所有相应的元素。

让我们看看如何。

```py
arr = [2,4,5,7,9]
arr_2d = [[1,2],[3,4]]

#printing the array
print("The Array is : ")
for i in arr:
    print(i, end = ' ')

#printing the 2D-Array
print("\nThe 2D-Array is:")
for i in arr_2d:
    for j in i:
        print(j, end=" ")
    print()

```

**输出**:

```py
The Array is : 
2 4 5 7 9 
The 2D-Array is:
1 2 
3 4

```

在上面的代码中，我们使用 for 循环遍历了一个 **1D** 以及一个 **2D** 数组的元素，并以我们想要的形式打印出相应的元素。

## 用 Python 打印 NumPy 数组的方法

如前所述，我们还可以使用 **NumPy** 模块在 Python 中实现数组。该模块带有一个预定义的数组类，可以保存相同类型的值。

这些 [NumPy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)也可以是多维的。所以，让我们看看如何用 Python 打印出 **1D** 以及 **2D** NumPy 数组。

### 使用 print()方法

类似于使用列表实现数组的情况，我们可以直接将 NumPy 数组名传递给`print()`方法来打印数组。

```py
import numpy as np

arr_2d = np.array([[21,43],[22,55],[53,86]])
arr = np.array([1,2,3,4])

print("Numpy array is: ", arr) #printing the 1d numpy array

print("Numpy 2D-array is: ", arr_2d) #printing the 2d numpy array

```

**输出**:

```py
Numpy array is:  [1 2 3 4]
Numpy 2D-array is:  [[21 43]
 [22 55]
 [53 86]]

```

这里，`arr`和`arr_2d`分别是一个 **1D** 和一个 **2D** NumPy 数组。我们将它们的名字传递给`print()`方法并打印它们。**注意:**这次数组也以带括号的 NumPy 数组的形式打印。

### 使用 for 循环

同样，我们也可以使用**循环**结构遍历 Python 中的 NumPy 数组。这样我们就可以访问数组的每个元素并打印出来。这是用 Python 打印数组的另一种方式。

仔细看下面的例子。

```py
import numpy as np

arr = np.array([11,22,33,44])
arr_2d = np.array([[90,20],[76,45],[44,87],[73,81]])

#printing the numpy array
print("The Numpy Array is : ")
for i in arr:
    print(i, end = ' ')

#printing the numpy 2D-Array
print("\nThe Numpy 2D-Array is:")
for i in arr_2d:
    for j in i:
        print(j, end=" ")
    print()

```

**输出**:

```py
The Numpy Array is : 
11 22 33 44 
The Numpy 2D-Array is:
90 20 
76 45 
44 87 
73 81

```

这里我们也通过分别访问 **1D** 和 **2D** 数组的元素，以我们想要的方式打印 **NumPy 数组**的元素(没有括号)。

## 结论

所以在本教程中，我们学习了如何用 Python 打印一个数组。我希望你现在对这个话题有一个清晰的理解。关于这个话题的任何进一步的问题，请随意使用评论。

## 参考

*   [Python 中的数组](https://www.askpython.com/python/array)–ask Python 教程，
*   [NumPy 数组介绍](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)–ask python 帖子。