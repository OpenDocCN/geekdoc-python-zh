# Python 字符串 lower()方法

> 原文：<https://www.askpython.com/python/string/python-string-lower>

Python String lower()方法将字符串对象转换成小写字符串。这是 Python 中内置的[字符串函数](https://www.askpython.com/python/string/python-string-functions)之一。因为在 Python 中字符串是*不可变的*，所以这个方法只返回原始字符串的一个**副本**。

## Python String lower()方法的语法和用法

格式:

```py
str_copy = str_orig.lower()

```

这里，`str_copy`是`str_orig`的小写字符串。

```py
a = "HELLO FROM ASKPYTHON"

b = a.lower()

print(a)
print(b)

```

**输出**

```py
HELLO FROM ASKPYTHON
hello from askpython

```

这将使整个输出字符串变成小写，即使输入字符串只有一部分是大写的。

```py
a = "Hello from AskPython"
b = a.lower()

print(a)
print(b)

```

**输出**

```py
Hello from AskPython
hello from askpython

```

由于 Python3 将任何字符串文字作为 Unicode 处理，因此它也可以小写不同的语言。

```py
>>> string = 'Километр'
>>> string
'Километр'
>>> string.lower()
'километр'

```

## 熊猫模块–下部()

熊猫[模块](https://www.askpython.com/python-modules/python-modules)中也有一个`lower()`方法，其功能与原生 Python 方法相同，但用于熊猫对象。

**格式:**

```py
pandas_copy = pandas_object.str.lower()

```

这里有一个例子可以说明这一点:

```py
>>> import pandas as pd
>>> 
>>> s = pd.Series(['Hello', 'from', 'ASKPYTHON'])
>>> print(s)
0        Hello
1         from
2    ASKPYTHON
dtype: object
>>> 
>>> print(s.str.lower())
0        hello
1         from
2    askpython
dtype: object
>>> print(s)
0        Hello
1         from
2    ASKPYTHON
dtype: object

```

正如您所观察到的，原始对象保持不变，我们得到一个全部是小写字符串的新对象！

* * *

## 参考

*   [Python 文档`str.lower()`上的](https://docs.python.org/3/library/stdtypes.html?highlight=str.lower#str.lower)
*   [StackOverflow 问题](https://stackoverflow.com/questions/6797984/how-do-i-lowercase-a-string-in-python)

* * *