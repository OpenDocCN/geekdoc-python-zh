# NumPy 展开——完整指南

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-unwrap>

你好读者，欢迎来到另一个关于 [**NumPy 数学函数**](https://www.askpython.com/python/numpy-trigonometric-functions) 的教程。在本教程中，我们将涵盖 NumPy 展开功能，以及实践的例子。这是一个需要理解的有趣函数。

没有别的事了，让我们开始吧。

***亦读:[NumPy hypot-完全指南](https://www.askpython.com/python-modules/numpy/numpy-hypot)***

## 什么是 NumPy unwrap？

`numpy.unwrap()`函数是 [**NumPy 库**](https://www.askpython.com/python-modules/numpy/python-numpy-module) 提供的数学函数之一。这个函数将给定的数组展开成一组新的值。简单地说，它将原来的元素数组转换成一组新的元素。

现在，让我们来看看 numpy.unwrap()函数的语法，这样会使事情更清楚。

### NumPy unwrap 的语法

```py
numpy.unwrap ( p,
               discont=None,
               axis=- 1,
               *,
               period=6.283185307179586 )

```

该函数的参数是:

*   **`p`**–是输入数组。
*   **`discont`** –它是输入数组的值之间的最大不连续性。默认情况下，`discont`值是 pi。
*   **`axis`**–这是一个可选参数。它指定展开功能将沿其运行的轴。
*   **`period`**–该参数为浮点型，可选。它表示输入换行范围的大小。默认值是 2*pi。

### 为什么解开一个数组？

如果一个 [**NumPy 数组**](https://www.askpython.com/python-modules/numpy/python-numpy-arrays) 有不连续的值，那么我们使用`numpy.unwrap()`函数来转换输入数组的值。如果数组元素之间的跳跃大于其沿给定轴的 **2*pi** 补码值的`discont`值，则展开函数展开或转换输入数组。

关于 NumPy unwrap 函数的理论，这就是你需要了解的全部内容。

**注意:**如果输入数组中的不连续性小于 **pi** 但大于 discont，则不进行展开，因为取 **2*p** i 补码只会使不连续性更大。

## 使用 NumPy 展开功能

让我们直接进入使用数组的例子

### 没有属性的默认 numpy 展开

```py
import numpy as np

a = np.array((1 , 2 , 3 , 4 , 5))
print("Result1 :\n",np.unwrap(a))

b = np.array((0 , 0.78 , 5.55 , 7.89))
print("Result2 :\n",np.unwrap(b))

```

**输出**

```py
Result1 :
 [1\. 2\. 3\. 4\. 5.]
Result2 :
 [ 0\.          0.78       -0.73318531  1.60681469]

```

在上面的输出中，您可以观察到输入数组 **a** 的元素之间没有不连续，并且没有对这个 NumPy 数组进行展开。

在第二个输出中，当 NumPy 数组 **b** 作为一个参数传递给 np.unwrap()函数时，展开就完成了。这是因为 0.78 和 5.55 之间的不连续性大于默认折扣值，即 pi。

现在，让我们尝试更多的例子，我们将设置自定义值。

### 使用不连续属性展开数字

```py
import numpy as np

a = np.array((5, 7, 10, 14, 19, 25, 32))
print("Result1 :\n",np.unwrap(a , discont=4))

b = np.array((0, 1.34237486723, 4.3453455, 8.134654756, 9.3465456542))
print("Result2 :\n",np.unwrap(b , discont=3.1))

```

**输出**

```py
Result1 :
 [ 5\.          7\.         10\.          7.71681469  6.43362939  6.15044408
  6.86725877]
Result2 :
 [0\.         1.34237487 4.3453455  1.85146945 3.06336035]

```

对于 Result1，我们已经将 NumPy 数组 **`a`** 作为输入传递给 NumPy 展开函数，并且将 **`discont`** 值设置为 4，即如果数组 a 的元素之间的跳转将大于或等于 4，则值将被转换。这里，10 和 14 之间的差是 4，因此值被改变。

对于 Result2，我们传递了一个 NumPy 数组 **b** 作为 NumPy 展开函数的输入，并且将 **`discont`** 的值设置为 3.1。这里，4.3453455 和 8.134654756 之间的差大于 3.1，因此展开完成。

## 摘要

这就是关于 NumPy 展开函数的全部内容。功能真的很有趣，也很好用。请查看 NumPy 官方文档链接，获取更多示例。请继续关注更多关于 Python 主题的有趣文章🙂

## 参考

[NumPy 文档–NumPy 展开](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html)