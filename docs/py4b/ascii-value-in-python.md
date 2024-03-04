# Python 中的 ASCII 值

> 原文：<https://www.pythonforbeginners.com/basics/ascii-value-in-python>

世界上有许多语言，因此有无数的符号。所有的符号都在计算机中用不同类型的编码表示，如 ASCII 和 Unicode。在本文中，我们将看到如何在 Python 中找到给定字符的 ASCII 值。

## 什么是 ASCII 值？

ASCII 代表“美国信息交换标准代码”。它仅包含 128 个字符，用于在计算机中表示不同的符号、十进制数字和英文字母。

在 ASCII 编码中，每个字符都被赋予一个 0 到 127 之间的数值。这个与字符相关的数值称为字符的 ASCII 值。

举个例子，

*   “A”被赋予 ASCII 值 65。所有大写字母都按照各自的顺序被赋予了 65 之后的 ASCII 值。即“B”的 ASCII 值为 66，“C”的 ASCII 值为 67，依此类推。
*   “a”被赋予了 ASCII 值 97。所有小写字母在 97 之后都被赋予了相应的 ASCII 值。即“b”的 ASCII 值为 98，“c”的 ASCII 值为 99，依此类推。

## 如何在 Python 中打印一个字符的 ASCII 值？

要打印给定字符的 ASCII 值，我们可以使用 python 中的 ord()函数。order()函数将给定的字符作为输入，并返回该字符的 ASCII 值。您可以在下面的示例中观察到这一点。

```py
char1 = "A"
result1 = ord(char1)
print("ASCII value of the character {} is {}.".format(char1, result1))
char2 = "B"
result2 = ord(char2)
print("ASCII value of the character {} is {}.".format(char2, result2))
char3 = "C"
result3 = ord(char3)
print("ASCII value of the character {} is {}.".format(char3, result3))
char4 = "Z"
result4 = ord(char4)
print("ASCII value of the character {} is {}.".format(char4, result4))
char5 = "a"
result5 = ord(char5)
print("ASCII value of the character {} is {}.".format(char5, result5))
char6 = "b"
result6 = ord(char6)
print("ASCII value of the character {} is {}.".format(char6, result6))
char7 = "c"
result7 = ord(char7)
print("ASCII value of the character {} is {}.".format(char7, result7))
char8 = "z"
result8 = ord(char8)
print("ASCII value of the character {} is {}.".format(char8, result8))
```

输出:

```py
ASCII value of the character A is 65.
ASCII value of the character B is 66.
ASCII value of the character C is 67.
ASCII value of the character Z is 90.
ASCII value of the character a is 97.
ASCII value of the character b is 98.
ASCII value of the character c is 99.
ASCII value of the character z is 122.
```

如果我们向`ord()`函数传递一个字符以外的值，它将引发一个 TypeError 异常。例如，如果我们传递一个包含多个字符而不是单个字符的字符串，`ord()`函数将引发如下的 TypeError。

```py
char1 = "ABC"
result1 = ord(char1)
print("ASCII value of the character {} is {}.".format(char1, result1))
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 2, in <module>
    result1 = ord(char1)
TypeError: ord() expected a character, but string of length 3 found
```

类似地，如果我们将一个整数而不是一个字符传递给 ord()函数，它将引发 TypeError 异常，如下所示。

```py
num1 = 123
result1 = ord(num1)
print("ASCII value of the character {} is {}.".format(num1, result1))
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 2, in <module>
    result1 = ord(num1)
TypeError: ord() expected string of length 1, but int found 
```

## 结论

在本文中，我们讨论了 ASCII 编码。我们还看到了如何在 Python 中找到字符的 ASCII 值。要了解 python 中不同的值和允许的字符，可以阅读这些关于 python 中的 [python 文字](https://www.pythonforbeginners.com/basics/python-literals)和[数据类型的文章。](https://www.pythonforbeginners.com/basics/numeric-types-python)