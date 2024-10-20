# 如何使用 Python hex()函数？

> 原文：<https://www.askpython.com/python/built-in-methods/python-hex-function>

在本文中，我们将介绍 Python **hex()** 函数。

如果你想把一个整数转换成十六进制的[字符串](https://www.askpython.com/python/string/python-string-functions)，前缀为**“0x”**，这个函数很有用。

让我们看看如何使用这个函数。

* * *

## 使用 Python hex()函数

Python hex()函数有一个非常简单的语法:

```py
hex_string = hex(val)

```

这里，`val`可以是整数、二进制、八进制或十六进制数。

让我们快速看一些例子。

```py
print(hex(1000))  # decimal
print(hex(0b111))  # binary
print(hex(0o77))  # octal
print(hex(0XFF))  # hexadecimal

```

**输出**

```py
0x3e8
0x7
0x3f
0xff

```

## 在自定义对象上使用 Python hex()

我们也可以在自定义对象上使用 hex()。但是，如果我们想成功地使用它，我们必须为我们的类定义 __index__() dunder 方法。

hex()方法会调用`__index__()`，所以一定要实现。这必须返回一个值，可以是十进制/二进制/八进制/十六进制数。

```py
class MyClass:
    def __init__(self, value):
        self.value = value
    def __index__(self):
        print('__index__() dunder method called')
        return self.value

my_obj = MyClass(255)

print(hex(my_obj))

```

**输出**

```py
__index__() dunder method called
0xff

```

事实上，正如您所看到的，它返回了我们所期望的结果。

首先，`hex()`在我们的自定义类上调用 __index__ 方法。

然后，它将返回值转换为十六进制字符串(**255->“0x ff”**)

* * *

## 结论

在本文中，我们学习了如何使用 **hex()** 函数，将数值转换成十六进制字符串。

## 参考

*   关于 Python hex()的 JournalDev 文章

* * *