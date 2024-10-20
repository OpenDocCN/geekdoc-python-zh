# NumPy angle–返回复杂参数的角度

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-angle>

嘿，你有没有想过一个复杂论点的角度是怎么计算的？这就是 Python NumPy 库发挥作用的地方。NumPy 库中有各种各样的函数，有一个函数 **NumPy angle** 执行计算角度的任务。

我们将通过它的语法和不同类型的例子来让你熟悉这个函数。

因此，没有任何进一步的到期让我们开始吧。

## 关于数字角度

你们一定都熟悉复数，对吧？我们来快速复习一下复数。

复数由两部分组成，第一部分是实部，第二部分是虚部。当复数绘制在复平面上时，实部和虚部位于相互垂直的轴上(就像 X 轴和 Y 轴一样)。

现在，复数的**角**就是复数与实轴的夹角。这就是 NumPy 角度函数发挥作用的地方。

*注:*复数 **z** 表示为 **z = x+yi** 其中 x 和 y 为实数。复数 z 的**角度**由 *tan^(-1) (y/x)* 给出。

**`numpy.angle()`** 是 NumPy 库的数学函数之一，返回复数的**角度**。

***也读作:*[NumPy Arctan——完全指南](https://www.askpython.com/python-modules/numpy/numpy-arctan)**

现在，让我们理解函数的语法。

## 数字角度的语法

```py
numpy.angle(z , deg = "")

```

*   第一个参数是复数。它可以是单个复数，也可以是复数的 NumPy 数组。
*   第二个参数是类型为**布尔**的 **deg** 值(真或假)。默认情况下，该函数返回以弧度表示的角度。为了获得以度为单位的角度，将**角度**值设置为**真**。

这是关于函数的理论和语法。

## 使用数字角度

让我们写一些代码来更好地理解这个函数。

### 单复数的 NumPy 角

```py
import numpy as np

print("The angle of the complex number 1+3j is:",np.angle(1+3j),"radians")

print("The angle of the complex number 3j is:",np.angle(3j),"radians")

print("The angle of the complex number 1 is:",np.angle(1),"radians")

print("The angle of the complex number -1-1j is:",np.angle(-1-1j),"radians")

```

**输出**

```py
The angle of the complex number 1+3j is: 1.2490457723982544 radians
The angle of the complex number 3j is: 1.5707963267948966 radians
The angle of the complex number 1 is: 0.0 radians
The angle of the complex number -1-1j is: -2.356194490192345 radians

```

在上面所有的例子中，我们已经将一个**单复数**作为参数传递给计算输入复数的角度的函数 **`np.angle()`** 。输出角度以**弧度**为单位。

让我们了解一下对于复数 **3j** 和 **1** 输出角度是如何计算的。取复数 **3j** ，实部为 0，虚部为 3，暗示在 tan^(-1) (y/x)中，y 为 3，x 为 0。这意味着 tan^(-1) (y/x)等于 tan^(-1) (无穷大)，其值为 90 度或 1.5707 弧度。

对于复数 **1** ，实部为 1，虚部为 0，这意味着在 tan^(-1) (y/x)中，y 为 0，x 为 1。这意味着 tan^(-1) (y/x)等于 tan^(-1) (0)，其值为 0 弧度。

我们的数学计算和程序的输出完全吻合。多酷啊🙂

**注:**始终在复平面上画复数，以明确复数与实轴所成的角度。对你真的会有帮助。

### 复数数组的 NumPy 角度

```py
import numpy as np

a = np.array((1+3j , -1+0.5j , 4-2j , 0.5+0.5j))

b = np.angle(a)

print("Input Array of Complex Numbers:\n",a)
print("Angles in radians:\n",b)

```

**输出**

```py
Input Array of Complex Numbers:
 [ 1\. +3.j  -1\. +0.5j  4\. -2.j   0.5+0.5j]
Angles in radians:
 [ 1.24904577  2.67794504 -0.46364761  0.78539816]

```

**注意:**输出数组与输入数组具有相同的维数和形状。

在上面的示例中，复数的 NumPy 数组作为参数传递给函数。这里，所有复数的输出角度也是以**弧度**表示的。

这里，该函数也以类似的方式工作。计算出**输入数组**的每个复元素的**角度**，并存储在上述程序的变量`**b**`中。然后，我们使用两个 **print** 语句来打印输入的 NumPy 数组和角度的 NumPy 数组。

但是，我们能得到以度为单位的输出角度吗？让我们看看如何做到这一点🙂

### 具有 deg 属性的 NumPy 角度

```py
import numpy as np

a = np.array((1+3j, -1j, 0.5+0.5j))

b = np.angle(a , deg="true")

print("Input Array:\n",a)
print("Angle in degrees:\n",b)

```

**输出**

```py
Input Array:
 [ 1\. +3.j  -0\. -1.j   0.5+0.5j]
Angle in degrees:
 [ 71.56505118 -90\.          45\.        ]

```

因此，如果我们将 **`deg`** 值设置为**真值**，就可以得到以**度**为单位的输出角度。

其余所有的事情都类似于前面的代码。你们都有一个任务，你们要计算单个复数的输出角度，以度为单位。

## 摘要

在本文中，我们学习了 NumPy 角度函数。我们练习了单个复数的代码以及复数的 NumPy 数组，我们还使用 deg 属性将输出角度转换为度数。请在阅读文章的同时写下代码。

敬请关注，点击这里继续探索更多文章[。](https://www.askpython.com/)

## 参考

[数字文件–数字角度](https://numpy.org/doc/stable/reference/generated/numpy.angle.html)