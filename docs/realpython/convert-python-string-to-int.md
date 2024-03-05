# 如何将 Python 字符串转换成 int

> 原文：<https://realpython.com/convert-python-string-to-int/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**将一个 Python 字符串转换成 int**](/courses/convert-python-string-int/)

[整数](https://en.wikipedia.org/wiki/Integer)是整数。换句话说，它们没有小数部分。Python 中可以用来存储整数的两种数据类型是 [`int`](https://realpython.com/python-data-types/#integers) 和 [`str`](https://realpython.com/python-data-types/#strings) 。这些类型为在不同环境下处理整数提供了灵活性。在本教程中，您将学习如何将 Python [字符串](https://realpython.com/python-strings/)转换为`int`。您还将学习如何将`int`转换成字符串。

本教程结束时，您将了解:

*   如何使用`str`和`int`存储整数
*   如何将 Python 字符串转换成`int`
*   如何将 Python `int`转换成字符串

我们开始吧！

**Python 中途站:**本教程是一个**快速**和**实用**的方法来找到你需要的信息，所以你会很快回到你的项目！

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 用 Python 表示整数

整数可以用不同的类型存储。表示整数的两种可能的 [Python 数据类型](https://realpython.com/python-data-types/)是:

1.  [T2`str`](https://docs.python.org/3/library/stdtypes.html#textseq)
2.  [T2`int`](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)

例如，您可以使用字符串文字来表示整数:

>>>

```py
>>> s = "110"
```

在这里，Python 理解您的意思是希望将整数`110`存储为字符串。您可以对整数数据类型执行相同的操作:

>>>

```py
>>> i = 110
```

在上面的例子中，考虑一下`"110"`和`110`的具体含义是很重要的。作为一个一生都在使用十进制数字系统的人，很明显你指的是数字*一百一十*。不过还有其他几个[数制](https://realpython.com/python-encodings-guide/#covering-all-the-bases-other-number-systems)，比如**二进制**和**十六进制**，用不同的[基](https://simple.wikipedia.org/wiki/Base_(mathematics))来表示一个整数。

比如你可以把数字*一百一十*用二进制和十六进制分别表示为 *1101110* 和 *6e* 。

您还可以使用`str`和`int`数据类型用 Python 中的其他数字系统来表示您的整数:

>>>

```py
>>> binary = 0b1010
>>> hexadecimal = "0xa"
```

注意`binary`和`hexadecimal`使用[前缀](https://docs.python.org/reference/lexical_analysis.html#integers)来标识数字系统。所有的整数前缀都是以`0?`的形式出现，在这个形式中，您用一个表示数字系统的字符替换`?`:

*   **b:** 二进制(基数 2)
*   **o:** 八进制(基数 8)
*   **d:** decimal (base 10)
*   **x:** 十六进制(基数 16)

**技术细节:**当前缀可以被推断时，它在整数或字符串表示中都不是必需的。

`int`假设文字整数为**十进制**:

>>>

```py
>>> decimal = 303
>>> hexadecimal_with_prefix = 0x12F
>>> hexadecimal_no_prefix = 12F
  File "<stdin>", line 1
    hexadecimal_no_prefix = 12F
                              ^
SyntaxError: invalid syntax
```

整数的字符串表示形式更加灵活，因为字符串包含任意文本数据:

>>>

```py
>>> decimal = "303"
>>> hexadecimal_with_prefix = "0x12F"
>>> hexadecimal_no_prefix = "12F"
```

这些字符串中的每一个都代表同一个整数。

现在您已经有了一些关于如何使用`str`和`int`表示整数的基础知识，您将学习如何将 Python 字符串转换成`int`。

[*Remove ads*](/account/join/)

## 将 Python 字符串转换为`int`

如果您有一个表示为字符串的十进制整数，并且您想要将 Python 字符串转换为`int`，那么您只需将该字符串传递给`int()`，它将返回一个十进制整数:

>>>

```py
>>> int("10")
10
>>> type(int("10"))
<class 'int'>
```

默认情况下，`int()`假定字符串参数表示十进制整数。然而，如果您将一个十六进制字符串传递给`int()`，那么您将看到一个`ValueError`:

>>>

```py
>>> int("0x12F")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: invalid literal for int() with base 10: '0x12F'
```

错误消息指出该字符串不是有效的十进制整数。

**注:**

重要的是要认识到向`int()`传递字符串的两种失败结果之间的区别:

1.  **语法错误:**当`int()`不知道如何使用提供的基数(默认为 10)解析字符串时，会出现`ValueError`。
2.  **逻辑错误:** `int()`确实知道如何解析字符串，但不是你预期的方式。

这是一个逻辑错误的例子:

>>>

```py
>>> binary = "11010010"
>>> int(binary)  # Using the default base of 10, instead of 2
11010010
```

在本例中，您希望结果是 *210* ，这是二进制字符串的十进制表示。不幸的是，因为您没有指定这种行为，`int()`假定该字符串是一个十进制整数。

对此行为的一个很好的保护措施是始终使用显式基来定义字符串表示:

>>>

```py
>>> int("0b11010010")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: invalid literal for int() with base 10: '0b11010010'
```

这里，您得到一个`ValueError`，因为`int()`不知道如何将二进制字符串解析为十进制整数。

当您将一个字符串传递给`int()`时，您可以指定用来表示整数的数字系统。指定数字系统的方法是使用`base`:

>>>

```py
>>> int("0x12F", base=16)
303
```

现在，`int()`知道您传递的是一个十六进制字符串，而期望的是一个十进制整数。

**技术细节:**你传递给`base`的参数不限于 2、8、10、16:

>>>

```py
>>> int("10", base=3)
3
```

太好了！现在您已经熟悉了将 Python 字符串转换成`int`的细节，您将学习如何进行相反的操作。

## 将 Python `int`转换成字符串

在 Python 中，可以使用`str()`将 Python `int`转换成字符串:

>>>

```py
>>> str(10)
'10'
>>> type(str(10))
<class 'str'>
```

默认情况下，`str()`的行为类似于`int()`,因为它产生十进制表示:

>>>

```py
>>> str(0b11010010)
'210'
```

在这个例子中，`str()`足够聪明，可以解释二进制文本并将其转换为十进制字符串。

如果您希望一个字符串在另一个数字系统中表示一个整数，那么您可以使用一个格式化的字符串，比如一个 [f-string](https://realpython.com/python-f-strings/) (在 Python 3.6+中)，以及一个指定基数的[选项](https://docs.python.org/library/string.html#format-specification-mini-language):

>>>

```py
>>> octal = 0o1073
>>> f"{octal}"  # Decimal
'571'
>>> f"{octal:x}"  # Hexadecimal
'23b'
>>> f"{octal:b}"  # Binary
'1000111011'
```

`str`是一种在各种不同的数字系统中表示整数的灵活方式。

[*Remove ads*](/account/join/)

## 结论

恭喜你！您已经学习了很多关于整数的知识，以及如何在 Python 字符串和`int`数据类型之间表示和转换它们。

在本教程中，您学习了:

*   如何使用`str`和`int`存储整数
*   如何为整数表示指定显式数字系统
*   如何将 Python 字符串转换成`int`
*   如何将 Python `int`转换成字符串

现在你已经知道了这么多关于`str`和`int`的知识，你可以学习更多关于使用 [`float()`](https://docs.python.org/3/library/functions.html#float) 、 [`hex()`](https://docs.python.org/3/library/functions.html#hex) 、 [`oct()`](https://docs.python.org/3/library/functions.html#oct) 、 [`bin()`](https://docs.python.org/3/library/functions.html#bin) 来表示数值类型的知识！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**将一个 Python 字符串转换成 int**](/courses/convert-python-string-int/)****