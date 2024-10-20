# 如何使用 Python complex()方法

> 原文：<https://www.askpython.com/python/built-in-methods/python-complex-method>

Python **complex()** 方法用于从其他原始数据类型创建一个复数。当您希望快速执行复杂的算术和转换时，这非常有用。

让我们来看看如何使用这个方法。

* * *

## Python 复合体的语法()

该函数返回一个包含实部和虚部的复数。复数的数据类型是`class complex`。

因此，函数调用语法是:

```py
class complex([real[, imag]])

```

复数的形式为:

```py
complex_number = real + imag*j

```

这里，`j`是虚数(sqrt(-1))

Python `complex()`方法将返回这个复数。

让我们看一些例子，来熟悉这个方法。

你可能认为我们可以只用整数和浮点数来构造复数，但事实并非如此！

我们也可以用一个字符串，十六进制，二进制，甚至是另一个复数来构造它！

我们将看一些例子来说明这一点。

* * *

## 使用 Python complex()方法

### 调用不带任何参数的 complex()

我们可以调用这个方法而不传递任何参数给它。这将返回零复数`0j`。

```py
a = complex()
print(type(a))
print(a)

```

**输出**

```py
<class 'complex'>
0j

```

### 使用数字参数调用 Python complex()

这将构造所需的形式为`a + bj`的复数，其中`a`和`b`是数字参数。

```py
a = complex(1, 2)
print(a)

b = complex(1, 1.5)
print(b)

c = complex(-1.5, 3.414)
print(c)

```

**输出**

```py
(1+2j)
(1+1.5j)
(-1.5+3.414j)

```

### 使用十六进制/二进制参数调用 complex()

我们也可以直接把十六进制或二进制数传入这个，而不用把它转换成整数。

```py
a = complex(0xFF)  # hexadecimal
print(a)

b = complex(0b1010, -1)  # binary
print(b)

```

**输出**

```py
(255+0j)
(10-1j)

```

### 使用另一个复数调用 complex()

当使用`complex()`构造一个复数时，我们也可以传递另一个复数

```py
a = 1 + 2j
b = complex(a, -4) # Construct using another complex number
print(b)
c = complex(1+2j, 1+2J)
print(c)

```

**输出**

```py
(1-2j)
(-1+3j)

```

因为两个参数都是按照形式`a + b*j`相加的，所以在第一种情况下，结果将是:1+2j+(-4 * j)=**1–2j**

第二种情况，我们有(因为 j * j =-1):1+2j+(1+2j)* j =(1+2j+j–2)=**-1+3j**

### 使用字符串参数调用 Python complex()

我们也可以用它来传递 string，只要它**和**之间没有任何空格。该字符串只能是“a+bj”形式，即可以表示复数的字符串。

如果我们使用一个字符串，我们只能使用**一个参数**。

```py
a = complex("1+2j")
print(a)

```

**输出**

```py
(1+2j)

```

如果字符串包含空格或任何其他无关字符，将会引发一个`ValueError`异常。

```py
b = complex("2 + 4j")
print(b)

```

这将引发一个`ValueError`，因为字符串中有空格。

类似地，如果我们传递两个参数给 Python `complex()`，它将引发一个`TypeError`异常，因为如果第一个参数是字符串，我们不允许传递多个参数。

* * *

## 结论

在本文中，我们学习了如何使用内置的`complex()`方法从不同的数据类型中构造一个复数。如果你想了解其他 [Python 内置函数，请访问这里](https://www.askpython.com/python/built-in-methods)。

## 参考

*   [Python 文档](https://docs.python.org/3.8/library/functions.html#complex)关于 complex()方法
*   关于使用 complex()方法的 JournalDev 文章

* * *