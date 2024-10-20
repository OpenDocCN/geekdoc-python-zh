# Python colorsys 模块

> 原文：<https://www.askpython.com/python-modules/colorsys-module>

读者朋友们，你们好！在本文中，我们将详细关注 **Python colorsys 模块**。所以，让我们开始吧！🙂

* * *

## 什么是 colorsys 模块？

Python 为我们提供了[不同的模块](https://www.askpython.com/python-modules/python-modules)来测试数据值的功能，并执行操作和表示。Python colorsys 模块就是这样一个模块。

colorsys 模块帮助我们对以下颜色值进行双向转换

1.  **(色调明度饱和度)**
2.  **YIQ(亮度(Y)同相正交)**
3.  **HSV(色调饱和度值)**
4.  **RGB(红、绿、蓝)**

所有这些颜色的坐标表示都是浮点值。转换值的允许范围通常分别在 0–1 之间。

## 如何使用 colorsys 模块？

现在让我们在下一节看看它们之间的相互转换。

### 1。RGB 到 YIQ 的相互转换

colorsys 模块为我们提供了 **rgb_to_yiq()方法**，该方法启动 rgb 到亮度(Y)同相正交颜色范围之间的转换。同样，我们需要将三个颜色值作为参数传递给函数，如下所示:

1.  **红色**
2.  **绿色**
3.  **蓝色**

看看下面的语法！🙂

**语法:**

```py
colorsys.rgb_to_yiq(Red, Green, Blue)

```

**例 1:** RGB- > YIQ

在下面的例子中，我们将红色、绿色和蓝色这三个颜色值传递给了 rgb_to_yiq()函数，并实现了 rgb 到 yiq 色阶的转换。

```py
import colorsys 

R = 0.1
G = 0.3
B = 0.3

YIQ = colorsys.rgb_to_yiq(R, G, B) 

print(YIQ) 

```

**输出:**

```py
(0.24, -0.11979999999999999, -0.0426)

```

Python colorsys 模块包括 **yiq_to_rgb()函数**，用于将亮度(Y)同相正交颜色值转换为 rgb 模式。

**语法:**

```py
yiq_to_rgb(Y, I, Q) 

```

**例二:** YIQ- > RGB

我们已经执行了 YIQ 色标值到红绿蓝色标的转换。

```py
import colorsys 

Y = 0.1
I = 0.3
Q = 0.3

RGB = colorsys.yiq_to_rgb(Y, I, Q) 

print(RGB) 

```

**输出:**

```py
(0.5711316397228637, 0.0, 0.28013856812933025)

```

* * *

### 2。HSV 到 RGB 的相互转换

除了 YIQ 和 rgb，colorsys 模块还为我们提供了 **hsv_to_rgb(H，S，V)函数**来执行 hsv 比例数据到 RGB 比例的转换。

**语法:**

```py
hsv_to_rgb(H,S,V)

```

**例 1:** HSV- > RGB

```py
import colorsys 

H = 0.1
S = 0.3
V = 0.3

RGB = colorsys.hsv_to_rgb(H, S, V) 

print(RGB) 

```

**输出:**

```py
(0.3, 0.264, 0.21)

```

除此之外，colorsys 模块还为我们提供了 rgb_to_hsv(R，G，B)函数来执行 rgb 比例到 hsv 颜色值格式的转换。

**例 2:** HSV- > RGB

我们利用 rgb_to_hsv()函数实现了 rgb 色标到 hsv 色标的转换。

```py
import colorsys 

R = 0.1
G = 0.3
B = 0.3

HSV = colorsys.rgb_to_hsv(R, G, B) 

print(HSV) 

```

**输出:**

```py
(0.5, 0.6666666666666666, 0.3)

```

* * *

### 3。RGB 到 HLS 的相互转换

使用 Python colorsys 模块，您可以使用 rgb_to_hls()函数轻松执行 RGB 色标到 HLS 色标的转换。

**语法:**

```py
rgb_to_hls(R, G, B)

```

**举例:**

在本例中，我们将 RGB 色阶值转换为 HLS 格式。

```py
import colorsys 

R = 0.1
G = 0.3
B = 0.3

HLS = colorsys.rgb_to_hls(R, G, B) 

print(HLS) 

```

**输出:**

如下所示，在上面的例子中也是如此，转换范围通常只限于 0 到 1 的范围。

```py
(0.5, 0.2, 0.49999999999999994)

```

* * *

## 结论

如果你遇到任何问题，欢迎在下面评论。更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂

* * *

## 参考

*   Python colorsys 模块— [文档](https://docs.python.org/3/library/colorsys.html)