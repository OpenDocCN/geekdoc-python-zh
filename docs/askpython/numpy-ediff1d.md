# numpy ediff 1d–数组中连续元素之间的差异

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-ediff1d

在本文中，我们理解并实现了 NumPy 函数`numpy.ediff1d()`,因为 NumPy 是一个数组处理包 numpy.ediff1d()用于计算数组元素之间的差。

## Numpy.ediff1d()是什么？

在 python 中使用`numpy.ediff1d()`来计算给定数组元素之间的连续差。

我们可以通过`to_begin`提到预置值，它在一个数组的开始预置一个值，同样我们可以通过`to_end`在一个数组的结尾附加一个值。

这些值可以包含多个要前置或附加的元素。

## ediff1d()的语法

```py
numpy.ediff1d(ary, to_end=None, to_begin=None)

```

**参数**

| **参数** | **描述** | **必需/可选** |
| ary: array_like | 输入数组 | 需要 |
| to_end:数组 _like | 这些是要追加到包含差异的数组末尾的数字。 | 可选择的 |
| to_begin:类似数组 | 这些数字将被添加到包含差值的数组的开头。 | 可选择的 |

Numpy -ediff1d syntax paramter

**返回值**

返回一个数组，其中包含:

*   在数组中预先考虑和追加值(如果给定的话)。
*   数组元素之间的连续差。

## ediff1d()方法的示例

导入 numpy 库并声明和打印数组。

```py
import numpy as np
arr=np.array([10,20,30,40,50])
print("Input array: \n",arr)

```

```py
Input array:
 [10 20 30 40 50]

```

### 示例 1:实现`numpy.ediff1d()`

下面的代码使用 np.ediff1d()函数来计算给定数组“arr”中每个元素之间的差。

```py
x=np.ediff1d(arr)
print ("Output array: \n",x)

```

输出是一个数组，包含给定数组中每个元素之间的差。在这种情况下，输出数组包含四个元素，每个元素的值都等于 10。

**输出:**

```py
Output array:
 [10 10 10 10]

```

### 例 2 **:** 将元素前置和追加到输出中。

这段代码创建了一个带有附加开始和结束值的输出数组。它使用 NumPy ediff1d 函数来实现这一点，该函数接受一个数组(在本例中为 arr)并将 to_begin 和 to_end 值(在本例中为 1 和 100)相加。

```py
y=np.ediff1d(arr,to_begin=[1],to_end=[100])
print ("Output array with begin and end value: \n",y)

```

**输出:**

```py
Output array with begin and end value:
 [  1  10  10  10  10 100]

```

### 示例 3:预先计划和附加多个元素到输出

下面的代码使用 numpy.ediff1d()函数来计算数组(arr)中元素之间的差。to_begin 和 to_end 参数用于在输出数组中预先计划和追加元素。

```py
z=np.ediff1d(arr, to_begin=[0,1], to_end=[90,100])
print ("Output array with multiple begin and end value: \n",z)

```

在这种情况下，输出数组将以值 1 开始，以值 100 结束。结果存储在变量 y 中，并打印到控制台。

**输出:**

```py
Output array with multiple begin and end value:
 [  0   1  10  10  10  10  90 100]

```

### 例 4:返回的数组总是 1D。

这段代码使用 NumPy 库来计算二维数组中每个元素之间的差。二维数组存储在名为“w”的变量中。然后使用 NumPy ediff1d()函数计算数组中每个元素之间的差异。

```py
w = [[1, 2, 4], [1, 6, 24]]
print ("Input array: \n",w)
print("Output array: \n",np.ediff1d(w))

```

ediff1d()函数的输出是一个一维数组，包含数组中每个元素之间的差异。在此示例中，输出数组包含差 1、2、-3、5 和 18。

**输出计算为**:

*   2 – 1 = 1
*   4 – 2 = 2
*   1 – 4 = -3
*   6 – 1 = 5
*   24 – 6 = 18

**输出:**

```py
Input array:
 [[1, 2, 4], [1, 6, 24]]
Output array:
[ 1  2 -3  5 18]

```

## 结论

我们已经理解并实现了带有和不带有可选参数的`numpy.ediff1d()`函数，即`to_end`和`to_begin`具有单个或多个元素作为它们的值。`numpy.ediff1d()`用于返回一个由输入数组元素的连续差值组成的数组。

## 参考

[https://HET . as . ute xas . edu/HET/Software/Numpy/reference/generated/Numpy . ediff 1d . html](https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.ediff1d.html)