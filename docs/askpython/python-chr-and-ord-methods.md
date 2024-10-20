# Python chr()和 ord()

> 原文：<https://www.askpython.com/python/built-in-methods/python-chr-and-ord-methods>

Python 的内置函数 **chr()** 用于将*整数*转换为*字符*，而函数 **ord()** 用于进行相反的操作，即将*字符*转换为*整数*。

让我们快速浏览一下这两个函数，并了解如何使用它们。

* * *

## chr()函数

### 句法

它接受一个整数`i`并将其转换成一个字符`c`，因此它返回一个字符串。

格式:

```py
c = chr(i)

```

这里有一个例子来说明这一点:

```py
# Convert integer 65 to ASCII Character ('A')
y = chr(65)
print(type(y), y)

# Print A-Z
for i in range(65, 65+25):
    print(chr(i), end = " , ")

```

**输出**

```py
<class 'str'> A
A , B , C , D , E , F , G , H , I , J , K , L , M , N , O , P , Q , R , S , T , U , V , W , X , Y , Z 

```

该参数的有效范围是从 0 到 1，114，111(十六进制的 0x10FFFF)。如果整数 *i* 在该范围之外，则 [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) 将被提升。

让我们用一些例子来验证这一点

```py
print(chr(-1))

```

这将引发一个`ValueError`。

```py
ValueError: chr() arg not in range(0x110000)

```

```py
start = 0
end = 1114111

try:
    for i in range(start, end+2):
        a = chr(i)
except ValueError:
    print("ValueError for i =", i)

```

**输出**

```py
ValueError for i = 1114112

```

* * *

## order()函数

**ord()** 函数接受单个 Unicode 字符的字符串参数，并返回其整数 Unicode 码位值。它与`chr()`相反。

### 句法

这采用单个 Unicode 字符(长度为 1 的字符串)并返回一个整数，因此格式为:

```py
i = ord(c)

```

为了验证它是否与`chr()`相反，让我们用一些例子来测试这个函数。

```py
# Convert ASCII Unicode Character 'A' to 65
y = ord('A')
print(type(y), y)

alphabet_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Print 65-90
for i in alphabet_list:
    print(ord(i), end = " , ")

```

**输出**

```py
<class 'int'> 65
65 , 66 , 67 , 68 , 69 , 70 , 71 , 72 , 73 , 74 , 75 , 76 , 77 , 78 , 79 , 80 , 81 , 82 , 83 , 84 , 85 , 86 , 87 , 88 , 89 , 90 , 

```

如果输入字符串的长度不等于 1，就会引发一个`TypeError`。

```py
y = ord('Hi')

```

**输出**

```py
TypeError: ord() expected a character, but string of length 2 found

```

* * *

## 传递十六进制数据

我们还可以将其他常用基数表示的整数传递给 **chr()** 和 **ord()** 等十六进制格式(基数 16)。

在 Python 中，我们可以通过在整数前面加上`0x`来使用十六进制，只要它在整数值的 32/64 位范围内。

```py
>>> print(hex(18))
'0x12'
>>> print(chr(0x12))
'\x12'
>>> print(ord('\x12'))
18
>>> print(int('\x12'))
18

```

我们将十六进制格式的整数 **18** 传递给`chr()`，后者返回十六进制的`0x12`。我们将它传递给`chr()`，并使用`ord()`来取回我们的整数。

注意，我们也可以使用`int()`获得整数，因为单个字符串也是一个字符串，它可以是上述函数的有效参数。

* * *

## 结论

在本文中，我们学习了如何使用`chr()`和`ord()`将整数转换成字符，反之亦然。

* * *

## 参考

*   [用于 chr()的 Python 文档](https://docs.python.org/3.8/library/functions.html#chr)
*   [Python 文档](https://docs.python.org/3.8/library/functions.html#ord) for ord()
*   JournalDev 文章

* * *