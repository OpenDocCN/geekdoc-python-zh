# NumPy conj–返回输入数字的复共轭

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-conj

嘿大家，欢迎来到另一个 NumPy 数学函数教程。在本教程中，我们将详细了解 **`NumPy conj`** 功能。

通过简单地改变虚部的符号，可以获得复数的共轭。

比如复数 10-8j 的共轭是 **10+8j** 。我们可以使用 **`numpy.conj()`** 函数获得复数的共轭。

那么，我们开始吧。

***也读作:[NumPy fmax–数组元素的逐元素最大值](https://www.askpython.com/python-modules/numpy/numpy-fmax)***

## 关于 NumPy conj

NumPy conj 是 NumPy 库的一个数学函数，它计算输入复数的复共轭。

### 句法

根据定义，这听起来很简单，对吗？现在，让我们看看函数的语法。

```py
numpy.conj(input)

```

这里，输入可以是单个复数，也可以是复数的 NumPy 数组。

## 使用 NumPy conj

现在，让我们用 python 做一些编程。

### 单个复数的 NumPy conj

```py
import numpy as np

# Complex Conjugate of a Complex number with real and imaginary parts
print("The complex conjugate of 1+6j is:",np.conj(1+6j))
print("The complex conjugate of 1-6j is:",np.conj(1-6j))

# Complex Conjugate of a Complex number with only imaginary part
print("The complex conjugate of 0+6j is:",np.conj(0+6j))
print("The complex conjugate of 0-6j is:",np.conj(0-6j))

# Complex Conjugate of a Complex number with only real part
print("The complex conjugate of 1 is:",np.conj(1))
print("The complex conjugate of -1 is:",np.conj(-1))

```

### 输出

```py
The complex conjugate of 1+6j is: (1-6j)
The complex conjugate of 1-6j is: (1+6j)
The complex conjugate of 0+6j is: -6j
The complex conjugate of 0-6j is: 6j
The complex conjugate of 1 is: 1
The complex conjugate of -1 is: -1

```

在上面的代码片段中，NumPy 库是使用 **`import`** 语句导入的，函数 **`np.conj()`** 用于计算输入复数的复共轭。

让我们了解一下这些值是如何计算的。

对于复数 **`1+6j`** ，通过改变虚部的符号得到共轭，因此输出为 **`1-6j`** 。

对于复数 **`1`** ，共轭将与输入的复数相同。这是因为数字 1 可以写成 **`1+0j`** ，其中虚部为 0，因此输出与输入的复数相同。

现在，让我们传递复数的 NumPy 数组，并计算复共轭。

### 复数的 NumPy 数组的 NumPy conj

```py
import numpy as np

a = np.array((1+3j , 0+6j , 5-4j))

b = np.conj(a)

print("Input Array:\n",a)
print("Output Array:\n",b)

```

**输出**

```py
Input Array:
 [1.+3.j 0.+6.j 5.-4.j]
Output Array:
 [1.-3.j 0.-6.j 5.+4.j]

```

在上面的代码片段中，使用存储在变量 **`a`** 中的 **`np.array()`** 创建了一个复数数组。变量 **`b`** 存储输入数组的共轭值，该数组也是一个 NumPy 数组。

**`np.conj()`** 计算输入数组中每个元素的共轭。

在接下来的几行中，我们使用了 print 语句来打印输入数组和输出数组。

### 使用 NumPy eye 函数的 NumPy 数组的 NumPy conj

在这段代码中，我们将使用 **`numpy.eye()`** 创建一个 NumPy 数组。

```py
import numpy as np

a = np.eye(2) + 1j * np.eye(2)

b = np.conj(a)

print("Input Array:\n",a)
print("Conjugated Values:\n",b)

```

**输出**

```py
Input Array:
 [[1.+1.j 0.+0.j]
 [0.+0.j 1.+1.j]]
Conjugated Values:
 [[1.-1.j 0.-0.j]
 [0.-0.j 1.-1.j]]

```

让我们试着理解上面的代码片段。

*   在第一行中，我们使用`**import**`语句导入 NumPy 库。
*   函数`**np.eye(2)**`创建一个 2×2 的数组，其中**对角元素**为 1，**其他元素**为 0。
*   类似地，表达式 **`1j * np.eye(2)`** 创建一个 2×2 的数组，其中**对角元素**为 1j，**其他元素**为 0。
*   然后，表达式`**np.eye(2)**` **`+`** **`1j * np.eye(2)`** 将两个数组对应的元素相加，存储在变量 **a** 中。
*   在下一行中，我们使用了 **`np.conj()`** 函数来计算共轭值。

这就是使用 **NumPy conj** 函数的全部内容。

## 摘要

在本教程中，您学习了 NumPy conj 函数，并练习了不同类型的示例。还有一个功能 **`numpy.conjugate()`** ，其工作方式与 **`numpy.conj()`** 功能完全相同。快乐学习，坚持编码。

## 参考

[num py documentation–num py conj](https://numpy.org/doc/stable/reference/generated/numpy.conj.html)