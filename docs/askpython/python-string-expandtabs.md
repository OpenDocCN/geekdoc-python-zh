# 了解 Python 字符串 expandtabs()函数

> 原文：<https://www.askpython.com/python/string/python-string-expandtabs>

Python 有一种有效的方法来处理字符串之间的空白。让我们在本文中了解一下 **Python String expandtabs()方法**。

## Python 字符串 expandtabs()方法的工作原理

如上所述，Python String 内置了 expandtabs()方法来处理字符串之间的空格。

Python String `expandtabs()`函数基本上用用户提供的空格数量作为参数来替换和扩展字符串之间的' \t '字符。

**语法:**

```py
string.expandtabs(size)

```

*   `size(optional)`:该参数指定了字符串之间用户需要的空间大小。默认大小为 8。

**例 1:**

```py
inp_str = "JournalDev\tprovides\ttutorials\ton\tPython\tand\tJava."
#print(inp_str)
res = inp_str.expandtabs()
print(res)

```

在上面的例子中，Python 字符串 expandtabs()函数的使用不带任何参数，即大小。因此' \t '被替换为默认尺寸**，即 8** 。

**输出:**

```py
JournalDev      provides        tutorials       on      Python  and     Java.

```

**例 2:**

```py
inp_str = "JournalDev\tprovides\ttutorials\ton\tPython\tand\tJava."
print("Original string:\n",inp_str)
res = inp_str.expandtabs(4)
print("String after using expandtabs:\n",res)

```

在上面的代码片段中，size = 4 作为参数传递给 Python 字符串 expandtabs()函数。因此，整个字符串中的' \t '字符被大小为 4 个单位的空格替换。

**输出:**

```py
Original string:
 JournalDev	provides	tutorials	on	Python	and	Java.
String after using expandtabs:
 JournalDev  provides    tutorials   on  Python  and Java.

```

**例 3:**

```py
inp_str = "JournalDev\tprovides\ttutorials\ton\tPython\tand\tJava."
print("Original string:\n",inp_str)
res = inp_str.expandtabs(18)
print("String after using expandtabs:\n",res)

```

**输出:**

```py
Original string:
 JournalDev	provides	tutorials	on	Python	and	Java.
String after using expandtabs:
 JournalDev        provides          tutorials         on                Python            and               Java.

```

* * *

## python expand tabs()–错误和异常

如果我们试图将非整数类型的值(比如浮点值)作为参数传递给 Python string expandtabs()函数，它会引发一个`TypeError exception`。

因此，我们可以理解 expandtabs()函数只接受整型值作为参数。

**举例:**

```py
inp_str = "JournalDev\tprovides\ttutorials\ton\tPython\tand\tJava."
res = inp_str.expandtabs(18.17)
print("String after using expandtabs:\n",res)

```

**输出:**

```py
TypeError                                 Traceback (most recent call last)
<ipython-input-15-f2418be436bf> in <module>
      1 inp_str = "JournalDev\tprovides\ttutorials\ton\tPython\tand\tJava."
----> 2 res = inp_str.expandtabs(18.17)
      3 print("String after using expandtabs:\n",res)

TypeError: integer argument expected, got float

```

* * *

## Python numpy.expandtabs()函数

[Python Numpy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)具有 numpy.char.expandtabs()函数，该函数提供与内置 expandtabs()函数相同的功能。

`numpy.char.expandtabs()`函数接受一个数组作为参数和空格的大小，以防用户希望更具体。

**语法:**

```py
numpy.char.expandtabs(array,size)

```

*   `array`:包含函数必须执行的元素。
*   `size`(可选):可选参数，指定通过替换制表符提供的空格大小，即' \t '。

**例 1:**

```py
import numpy
inp_arr = numpy.array("JournalDev\tprovides\ttutorials\ton\tPython\tand\tJava.")
res = numpy.char.expandtabs(inp_arr)
print("String after using expandtabs:\n",res)

```

**输出:**

```py
Array of string after using expandtabs:
 JournalDev      provides        tutorials       on      Python  and     Java.

```

**例 2:**

```py
import numpy
inp_arr = numpy.array("JournalDev\tprovides\ttutorials\ton\tPython\tand\tJava.")
res = numpy.char.expandtabs(inp_arr,20)
print("Array of string after using expandtabs:\n",res)

```

**输出:**

```py
Array of string after using expandtabs:
 JournalDev          provides            tutorials           on                  Python              and                 Java.

```

* * *

## 摘要

*   Python 字符串 **expandtabs()函数**用于处理在运行时在字符串之间提供空格。它**用提到的空间量**替换' \t '字符，默认大小是 8。
*   替换' \t '的大小应该是一个**整型值**。
*   如果一个非整数类型的值被传递给函数，它会引发一个 **TypeError 异常**。
*   因此，expandtabs()被证明是一种在运行时处理添加空格的有效技术。

* * *

## 结论

因此，在本文中，我们已经了解了 Python string expandtabs()函数与 Python 字符串和 NumPy 数组的工作原理。

## 参考

*   Python 扩展表— JournalDev