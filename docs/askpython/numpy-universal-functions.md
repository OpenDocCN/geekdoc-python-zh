# NumPy 万能函数要知道！

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-universal-functions>

读者朋友们，你们好！在本文中，我们将关注 Python 编程中的 **NumPy 通用函数**。所以，让我们开始吧！🙂

* * *

## 我们所说的 NumPy 万能函数是什么意思？

NumPy 通用函数实际上是数学函数。NumPy 中的 NumPy 数学函数被构造为通用函数。这些通用(数学 NumPy 函数)对 [NumPy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)进行运算，而[对数据值执行元素式运算](https://www.askpython.com/python-modules/numpy/numpy-broadcasting)。

在 Python 中，通用 NumPy 函数属于 numpy.ufunc 类。当我们调用某些运算符时，一些基本的数学运算会在内部调用。例如，当我们构造 x + y 时，它在内部调用 numpy.add()通用函数。

我们甚至可以使用 frompyfunc()方法创建自己的通用函数。

**语法:**

```py
numpy.frompyfunc(function-name, input, output)

```

*   **函数名**:作为通用函数的函数名
*   **输入**:输入数组的个数
*   **输出**:输出数组的个数

**举例:**

在本例中，我们使用 **frompyfunc()** 方法将函数 **product** 转换为通用函数。

因此，现在 product()方法的行为就像一个通用的数学函数，当数组作为参数传递给它时，它执行元素级乘法。

```py
import numpy as np

def product(a, b):
  return a*b

product = np.frompyfunc(product, 2, 1)

res = product([1, 2, 3, 4], [1,1,1,1])
print(res)

```

**输出:**

```py
[1 2 3 4]

```

* * *

## 1.NumPy 中的泛三角函数

在这个概念的过程中，我们现在将看看 NumPy 中的一些通用[三角函数。](https://www.askpython.com/python/numpy-trigonometric-functions)

1.  **numpy。deg2raf()** :这个函数帮助我们将度数转换成弧度。
2.  **numpy.sinh()函数**:计算双曲正弦值。
3.  **numpy.sin()函数**:计算正弦双曲值的倒数。
4.  **numpy.hypot()函数**:计算直角三角形结构的斜边。

**举例:**

```py
import numpy as np

data = np.array([0, 30, 45])

rad = np.deg2rad(data)

# hyperbolic sine value
print('Sine hyperbolic values:')
hy_sin = np.sinh(rad)
print(hy_sin)

# inverse sine hyperbolic
print('Inverse Sine hyperbolic values:')
print(np.sin(hy_sin))

# hypotenuse
b = 3
h = 6
print('hypotenuse value for the right angled triangle:')
print(np.hypot(b, h))

```

**输出:**

```py
Sine hyperbolic values:
[0\.         0.54785347 0.86867096]
Inverse Sine hyperbolic values:
[0\.         0.52085606 0.76347126]
hypotenuse value for the right angled triangle:
6.708203932499369

```

* * *

## 2.通用统计函数

除了三角函数， [Python NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-module) 还为我们提供了通用的统计函数。其中一些列举如下:

1.  **numpy.amin()函数**:表示数组中的最小值。
2.  **numpy.amax()函数**:表示数组中的最大值。
3.  **numpy.ptp()函数**:表示一个数组的值在一个轴上的范围，通过从最大值中减去最小值来计算。
4.  **numpy.average()函数**:计算数组元素的平均值。

**举例:**

```py
import numpy as np

data = np.array([10.2,34,56,7.90])

print('Minimum and maximum data values from the array: ')
print(np.amin(data))
print(np.amax(data))

print('Range of the data: ')
print(np.ptp(data))

print('Average data value of the array: ')
print(np.average(data))

```

**输出:**

```py
Minimum and maximum data values from the array:
7.9
56.0
Range of the data:
48.1
Average data value of the array:
27.025000000000002

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，敬请关注我们！

在那之前，学习愉快！！🙂