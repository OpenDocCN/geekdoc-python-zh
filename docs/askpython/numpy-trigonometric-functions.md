# 要知道普适的 NumPy 三角函数

> 原文：<https://www.askpython.com/python/numpy-trigonometric-functions>

读者朋友们，你们好！在本文中，我们将学习**万能的 NumPy 三角函数**来认识！

所以，让我们开始吧！🙂

为了与一起，NumPy 中的数学函数被框架为通用函数。这些通用(数学 NumPy 函数)对 [NumPy 数组类](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)进行操作，并对数据值执行元素操作。通用 NumPy 函数属于 Python 中的 **numpy.ufunc 类**。

在本主题的背景下，我们将重点关注以下类型的通用三角函数

1.  **通用三角函数**
2.  **帮助我们进行度数和弧度值相互转换的函数**
3.  **双曲线函数**
4.  **计算斜边值**
5.  **根据三角函数确定角度值**

* * *

## 1。Numpy 三角函数

在本教程中，我们将使用以下通用数字三角函数——

1.  **numpy.sin()函数**:计算数组值的正弦分量。
2.  **numpy.cos()函数**:计算数组值的余弦分量。
3.  **numpy.tan()函数**:计算数组数据元素的正切值。

**举例:**

```py
import numpy as np
arr = np.array([30,60,90])

val_sin = np.sin(arr)
print("Sine value",val_sin)

val_cos = np.cos(arr)
print("Cosine value",val_cos)

val_tan = np.tan(arr)
print("Tangent value",val_tan)

```

**输出:**

```py
Sine value [-0.98803162 -0.30481062  0.89399666]
Cosine value [ 0.15425145 -0.95241298 -0.44807362]
Tangent value [-6.4053312   0.32004039 -1.99520041]

```

* * *

## 2.度数和弧度值之间的相互转换

在任何语言中执行三角函数运算时，我们都会遇到需要将角度转换为弧度的情况，反之亦然。

同样，NumPy 为我们提供了通用功能

1.  **deg2rad** :将角度的度数转换为弧度。
2.  **rad2deg** :将弧度角度转换为度数。

**举例:**

```py
import numpy as np
arr = np.array([30,60,90])

rad = np.deg2rad(arr)
print("Radian values for the array having degree values:", rad)

arr_rad = np.array([0.52359878, 1.04719755, 1.57079633])
degree = np.rad2deg(arr_rad)
print("Degree values for the array having radian values:", degree)

```

**输出:**

```py
Radian values for the array having degree values: [0.52359878 1.04719755 1.57079633]
Degree values for the array having radian values: [30.00000025 59.99999993 90.00000018]

```

* * *

## 3.根据三角值确定角度

以逆向工程的形式，我们现在给下面的函数输入三角值，并试图从中获得角度值

1.  **反正弦()函数**:根据正弦值计算角度值。
2.  **arccos()函数**:根据余弦值计算角度值。
3.  **arctan()函数**:从正切值计算角度值。

**举例:**

```py
import numpy as np
arr = np.array([1,0.5])

sin_ang = np.arcsin(arr)
print("Angle from the sin function:", sin_ang)

cos_ang = np.arccos(arr)
print("Angle from the cos function:", cos_ang)

tan_ang = np.arctan(arr)
print("Angle from the tan function:", tan_ang)

```

**输出:**

```py
Angle from the sin function: [1.57079633 0.52359878]
Angle from the cos function: [0\.         1.04719755]
Angle from the tan function: [0.78539816 0.46364761]

```

* * *

## 4。斜边

使用 **numpy.hypot()函数**，我们可以通过向函数提供底值和高值来根据毕达哥拉斯的标准计算斜边值。

**语法:**

```py
numpy.hypot() function

```

**举例:**

```py
import numpy as np

b = 5
h = 8

hy = np.hypot(b, h)

print(hy)

```

**输出:**

```py
9.433981132056603

```

* * *

## 5.双曲函数

NumPy 为我们提供了以下函数来计算给定值的双曲三角值:

1.  **numpy.sinh()函数**:计算数组值的双曲正弦值。
2.  **numpy.cosh()函数**:计算数组值的双曲余弦值。
3.  **numpy.tanh()函数**:计算数组值的双曲正切值。

**举例:**

```py
import numpy as np
arr = np.array([30,60,90])

val_sin = np.sinh(arr)
print("Hyperbolic Sine value",val_sin)

val_cos = np.cosh(arr)
print("Hyperbolic Cosine value",val_cos)

val_tan = np.tanh(arr)
print("Hyperbolic Tangent value",val_tan)

```

**输出:**

```py
Hyperbolic Sine value [5.34323729e+12 5.71003695e+25 6.10201647e+38]
Hyperbolic Cosine value [5.34323729e+12 5.71003695e+25 6.10201647e+38]
Hyperbolic Tangent value [1\. 1\. 1.]

```

* * *

## 结论

至此，我们已经结束了 NumPy 三角函数这篇文章。如果你遇到任何问题，欢迎在下面评论。更多关于 [Python 编程](https://www.askpython.com/python/oops/object-oriented-programming-python)的帖子，敬请关注我们！

在那之前，学习愉快！！🙂