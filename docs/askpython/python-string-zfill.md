# Python 字符串 zfill()

> 原文：<https://www.askpython.com/python/string/python-string-zfill>

Python string **zfill()** 方法用于在字符串左边填充零，直到达到特定的宽度。这是向字符串添加填充的最“Pythonic 式”的方法，而不是使用 for 循环手动迭代。

* * *

## Python 字符串 zfill()方法的语法

Python string **zfill()** 函数属于 string 类，只能在 String 对象上使用。

它将**宽度**作为参数，并在字符串的左边(字符串的开始)填充零，直到达到指定的宽度。

如果 **width < len(str)** ，则返回原字符串。否则，由于 Python 字符串是不可变的，这个函数返回一个新的字符串。

```py
new_str = str.zfill(width)

```

如果有前导符号( **+/-** )，则在符号后添加填充，即使在新字符串中也保持前导符号的位置。

现在我们来看一些例子。

* * *

如果没有前导符号，则将零填充到左侧。

```py
>>> a = "AskPython"
>>> a.zfill(15)
'00000AskPython'

```

如您所见，零被填充到左侧，新字符串的长度为 **15** 。

如果指定的宽度小于 **len("AskPython")** ，那么我们得到原始字符串。

```py
>>> a = "AskPython"
>>> a.zfill(len(a) - 1)
'AskPython'
>>> a.zfill(len(a) - 1) == a
True

```

如果有一个前导 **+/-** 符号，则在符号后补零。

```py
>>> a = "+123456"
>>> a.zfill(len(a) - 1)     
'+123456'
>>> a.zfill(10)
'+000123456'

>>> a = "-123456"
>>> a.zfill(10)
'-000123456'

```

* * *

## Python 字符串 zfill()的替代

除了 **zfill()** 之外，我们还可以使用一些其他方法在字符串中引入零填充。

### 使用 str.rjust()

与 **str.zfill()** 类似， **str.rjust()** 也可以给字符串填充，但也允许我们指定填充字符串的字符。看看下面的例子。

```py
>>> a = "AskPython"
>>> a.rjust(15, "0")
'00000AskPython'
>>> a.rjust(15, "1")
'11111AskPython'

```

这里，我们介绍了使用`rjust(width, '0')`填充零，使用`rjust(width, '1')`填充**一个**。您可以用任何字符替换这些数字。

### 使用打印字符串格式

如果你正在处理实数，你也可以使用 **Python f-strings** (来自 **Python 3.6** 和更高版本)返回一个零填充字符串。

```py
>>> a = 1.414
>>> print(f'{a:013}')
'000000001.414'

```

您可以在[官方文档](https://docs.python.org/2/library/string.html#formatexamples)中了解更多关于打印字符串格式的信息。

* * *

## 参考

*   zfill()上的 [Python 文档](https://docs.python.org/3.8/library/stdtypes.html?highlight=zfill#str.zfill)
*   [zfill()上的 StackOverflow 问题](https://stackoverflow.com/questions/339007/how-do-i-pad-a-string-with-zeroes)
*   zfill()上的 JournalDev 文章

* * *