# 整数数据的 Python 位函数[简单解释]

> 原文：<https://www.askpython.com/python/built-in-methods/python-bit-functions>

读者朋友们，你们好！在本文中，我们将重点介绍一些针对整数数据 **的 **Python 位函数** **。****

所以，让我们开始吧！

## 什么是 Python 位函数？

在开始学习整数的 Python 位函数之前，让我们先了解一下整数的相互转换。

现在，当我们自动化一些手动步骤或者处理系统级信息时，数据值的转换就变得重要了。

当我们处理包括不同数字形式(如十六进制、数字、八进制等)的数据时，位函数在分析整数的位级数据中起着重要的作用。

Python 为我们提供了以下一组位级函数，帮助我们分析有关位级信息和表示的整数数据:

1.  **bit _ length()函数**
2.  **to _ bytes()函数**
3.  **int . from _ bytes()函数**

* * *

## 了解 Python 位函数

现在，让我们在下一节中逐一查看上述功能。

### 1.Python bit_length()函数

**bit_length()函数**计算并返回二进制表示所传递的整数数据值所需的位数。该函数不考虑数据值的符号以及前导零。

**举例:**

在本例中，我们最初将 data = 3 传递给了 bit_length()函数。它返回值为 2。但是二进制形式的整数值 3 的实际表示包括 2 个零，即总共 4 个比特(0011)。

但是由于该函数不考虑前导零，所以它只计算相同的非零位置。

此外，我们向函数传递了一个负 7，即(-7)。但是当函数忽略符号值时，它会像对待其他正整数一样对待。

```py
data = 3
print(data.bit_length()) 
num = 9
print(num.bit_length())

num = -7
print(num.bit_length()) 

```

**输出:**

```py
2
4
3

```

* * *

### 2.Python to_bytes()函数

**int.to_bytes()函数**也将整数值表示为字节数组的序列。

**语法:**

```py
int.to_bytes(length, byteorder, signed=False)

```

1.  **length:** 表示结果数组的长度。
2.  **byteorder:** 如果设置为‘big’，最高有效字节放在数组的开头。如果设置为' little '，最高有效字节位于字节数组的末尾。
3.  **signed:** 如果设置为 True，它使用二进制补码将整数表示为字节数组。

**举例:**

在本例中，我们将整数值 2048 表示为长度等于 4 的字节数组，最高有效字节位于数组的开头。

```py
print((2048).to_bytes(4, byteorder ='big')) 

```

**输出:**

```py
b'\x00\x00\x08\x00'

```

* * *

### 3.Python from_bytes()函数

**int.from_bytes()函数**与 int.to_bytes()函数完全相反。

也就是说，from_bytes()函数将一个字节数组作为参数，并带有 byteorder 参数，然后返回与之对应的整数值。

**语法:**

```py
int.from_bytes(bytes, byteorder, signed=False)

```

**举例:**

```py
print(int.from_bytes(b'\x00\x04', byteorder ='big')) 

```

**输出:**

```py
4

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂