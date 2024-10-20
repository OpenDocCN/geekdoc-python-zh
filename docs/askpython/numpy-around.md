# NumPy around–完整指南

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-around>

欢迎来到另一个关于 [**NumPy 数学函数**](https://www.askpython.com/python/numpy-trigonometric-functions) 的教程。在本教程中，我们将详细学习如何使用`NumPy around`功能，我们还将练习各种例子来使我们的理解清晰。

我们一定都解决过不同类型的数学或物理问题，这些问题的最终答案精确到小数点后两位。假设我们必须对 1.7 进行舍入，我们会认为 1.7 是最接近 2 的 T2，因此在对 1.7 进行舍入后，值将是 2。

我们也可以通过编程来做到这一点，我们将在本教程中学习编程。因此，没有任何进一步的到期让我们开始。

[NumPy exp–完整指南](https://www.askpython.com/python-modules/numpy/numpy-exp)

## NumPy 在哪里？

NumPy around 是 NumPy 库的数学函数之一，它对作为函数输入的数字进行舍入。

让我们看看 NumPy around 函数的语法。

## NumPy around 的语法

```py
numpy.around(a, decimals=0, out=None)

```

我们来了解一下这个函数的参数。

*   `**a**`–需要四舍五入的输入数字。它可以是一个**的单个数字**，也可以是一个**的数字数组**。
*   **`decimals`**–小数的值总是一个**整数**。它指定了我们希望将输入数字四舍五入到的小数位数。这是一个可选参数。其默认值为 0。
*   **`out`**–是 numpy.around()函数的输出。这也是一个可选参数。

## 和周围的人一起工作

现在让我们写一些代码来更好地理解这个函数。

### 用一个数字作为输入

```py
import numpy as np

# Rounding off some integer values
print("Around of 1 is:",np.around(1))

print("Around of 5 is:",np.around(5))

# Rounding off some decimal values
print("Around of 5.5 is:",np.around(5.5))

print("Around of 9.54 is:",np.around(9.54))

print("Around of 12.70 is:",np.around(12.70))

print("Around of 9.112 is:",np.around(9.112))

print("Around of 10.112 is:",np.around(10.112))

```

**输出**

```py
Around of 1 is: 1
Around of 5 is: 5
Around of 5.5 is: 6.0
Around of 9.54 is: 10.0
Around of 12.70 is: 13.0
Around of 9.112 is: 9.0
Around of 10.112 is: 10.0

```

在上面的输出中，整数即使经过舍入也保持不变。每隔一个输出四舍五入到小数点后 1 位的数字。

### 带小数参数的数字

**`decimals`** 参数允许我们指定输入数字的小数位数。

```py
import numpy as np

# Round to ones place
print("Around 5.145 is:",np.around(5.145 , 0))

# Round to tenths place
print("Around 5.145 is:",np.around(5.145 , 1))

# Round to hundredths place
print("Around 5.145 is:",np.around(5.145 , 2))

# Round to thousandths place
print("Around 5.145 is:",np.around(5.145 , 3))

# Returns the same number
print("Around 5.145 is:",np.around(5.145 , 10))

```

**输出**

```py
Around 5.145 is: 5.0
Around 5.145 is: 5.1
Around 5.145 is: 5.14
Around 5.145 is: 5.145
Around 5.145 is: 5.145

```

上面的输出非常清楚，其中的数字被舍入到指定的 **`decimals`** 值。

### NumPy 与小数的负值有关

```py
import numpy as np

# Round to tenths place
print("Around 455.56 is:",np.around(455.56 , -1))

# Round to hundredths place
print("Around 455.56 is:",np.around(455.56 , -2))

# Round to thousandths place
print("Around 455.56 is:",np.around(455.56 , -3))

# Round to tenths place    
print("Around 455 is:",np.around(455 , -1))

```

**输出**

```py
Around 455.56 is: 460.0
Around 455.56 is: 500.0
Around 455.56 is: 0.0
Around 455 is: 460

```

如果 **`decimals`** 的值被设置为某个负值，则输入数字的**非十进制**位被舍入。

让我们来了解一下将 455.56 的小数位数设置为-2 时的舍入。这里，-2 表示数字中的第一百位。现在，计算数字中小数点左边的第一百位，相应地，数字被四舍五入。

**注意:**超出输入数字最左边的数字四舍五入将得到 0。

### 用 NumPy 数组来表示 NumPy

```py
import numpy as np

a = np.array((1 , 3 , 5 , 100 , 145 , 344 , 745))

print("\n")
print("Input Array:\n",a)
print("Result :\n",np.around(a))

print("\n")
print("Input Array:\n",a)
print("Rounded Values:\n",np.around(a , -1))

b = np.array((0.5 , 1.5 , 1.7 , 3.5 , 7.5 , 9.8))

print("\n")
print("Input Array:\n",b)
print("Rounded Values:\n",np.around(b))

c = np.array((4.567 , 13.564 , 12.334 , 1.567 , 9.485 , 4.444))

print("\n")
print("Input Array:\n",c)
print("Rounded Values:\n",np.around(c , 2))

```

**输出**

```py
Input Array:
 [  1   3   5 100 145 344 745]
Result :
 [  1   3   5 100 145 344 745]

Input Array:
 [  1   3   5 100 145 344 745]
Rounded Values:
 [  0   0   0 100 140 340 740]

Input Array:
 [0.5 1.5 1.7 3.5 7.5 9.8]
Rounded Values:
 [ 0\.  2\.  2\.  4\.  8\. 10.]

Input Array:
 [ 4.567 13.564 12.334  1.567  9.485  4.444]
Rounded Values:
 [ 4.57 13.56 12.33  1.57  9.48  4.44]

```

有一件有趣的事情需要注意，`**np.around(**` **`)`** 舍入后返回一个**新的 ndarray** 并且不影响原来的 ndarray。

这就是第三个参数发挥作用的地方。

### 在不带参数的情况下使用 NumPy

```py
import numpy as np

a = np.array((1.34 , 2.56 , 3.99 , 45.45 , 100.01))

print("Original Array:\n",a)

np.around(a , out=a)

print("Array after Rounding the values:\n",a)

```

**输出**

```py
Original Array:
 [  1.34   2.56   3.99  45.45 100.01]
Array after Rounding the values:
 [  1\.   3\.   4\.  45\. 100.]

```

在上面的程序中，舍入后的值存储在原始数组中。

以上都是我的观点，请执行本文中讨论的代码，并尝试使用带有不同输入的函数。请务必查看 NumPy around 函数的官方文档。

## 参考

[NumPy 文档–NumPy 大约](https://numpy.org/doc/stable/reference/generated/numpy.around.html)