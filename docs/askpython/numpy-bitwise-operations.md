# 5 NumPy 位运算就知道了！

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-bitwise-operations>

读者朋友们，你们好！在本文中，我们将重点关注我们应该知道的 **5 NumPy 位操作**！

所以，让我们开始吧！

首先，按位[操作符](https://www.askpython.com/python/python-operators)帮助我们执行位级操作，即通过函数中包含的抽象层进行逐位操作。

在主题课程中，我们将在本文中讨论以下主题——

1.  **和操作**
2.  **或操作**
3.  **异或运算**
4.  **反转操作**
5.  **整数到二进制表示法**

让我们开始吧！🙂

* * *

## 1.NumPy 位操作–AND

**NumPy 位 AND 运算符**使我们能够像输入值一样对数组执行位 AND 运算。也就是说，它对输入整数值的二进制表示执行 AND 运算。

**语法:**

```py
numpy.bitwise_and(num1,num2)

```

**举例:**

在下面的示例中，bitwise_and()函数将整数值 2 和 3 转换为等效的二进制值，即 2 ~ 010 和 3 ~ 011。此外，它执行 AND 运算，如果两个等价位都是 1，则返回 1 作为结果位，否则返回 0。

```py
import numpy as np
x = 2
y = 3

print ("x=",x)
print ("y=",y)
res_and = np.bitwise_and(x, y) 
print ("Bitwise AND result: ", res_and) 

```

**输出:**

```py
x= 2
y= 3
Bitwise AND result:  2

```

* * *

## 2.按位或运算

与 AND 运算一样， **[NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)** 也为我们提供了`numpy.bitwise_or() function`，使我们能够对数据值执行 NumPy 位“或”运算。

**语法:**

```py
numpy.bitwise_or(num1,num2)

```

**举例:**

在本例中，bitwise_or()函数对两个整数值执行 or 运算。在 OR 运算中，如果两位相同，即 0/0，则返回 0，否则返回 1。

```py
import numpy as np
x = 2
y = 3

print ("x=",x)
print ("y=",y)
res_or = np.bitwise_or(x, y) 
print ("Bitwise OR result: ", res_or) 

```

**输出:**

```py
x= 2
y= 3
Bitwise OR result:  3

```

* * *

## 3.逐位异或运算

XOR 运算是 NumPy 位运算之一。我们可以使用 numpy.bitwise_xor()函数来执行运算。这样，我们可以很容易地对使用的逐位数据执行逐位 XOR 运算。

**举例:**

```py
import numpy as np
x = 2
y = 3

print ("x=",x)
print ("y=",y)
res_xor = np.bitwise_xor(x, y) 
print ("Bitwise XOR result: ", res_xor) 

```

**输出:**

```py
x= 2
y= 3
Bitwise XOR result:  1

```

* * *

## 4.逐位反转操作

使用 numpy.invert()函数执行按位反转操作。我们的意思是，它对内部处理为二进制表示格式的数据位执行逐位 NOT 运算。

对于有符号整数，返回二进制补码值。

**举例:**

```py
import numpy as np
x = 2
y = 3

print ("x=",x)
res = np.invert(x) 
print ("Bitwise Invert operation result: ", res) 

```

**输出:**

```py
x= 2
Bitwise Invert operation result:  -3

```

* * *

## 5.二进制表示法

使用 NumPy 模块，我们可以显式地将整数值转换为二进制数。 `binary_repr()` 函数使我们能够轻松地将整数数据值转换为二进制值。

**语法:**

```py
numpy.binary_repr()

```

**举例:**

```py
import numpy as np
x = 7

print ("x=",x)
res = np.binary_repr(x) 
print ("Bitwise representation of x: ", res) 

```

**输出:**

在本例中，我们已经将 int 值“7”转换为其等效的二进制表示形式。

```py
x= 7
Bitwise representation of x:  111

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂