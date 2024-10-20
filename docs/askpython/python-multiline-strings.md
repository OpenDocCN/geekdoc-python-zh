# 创建 Python 多行字符串的 4 种技术

> 原文：<https://www.askpython.com/python/string/python-multiline-strings>

Python 有不同的方式来表示字符串。一个 **Python** **多行字符串**是以格式化和优化的方式呈现多个字符串语句的最有效方式。

在本文中，我们将重点关注可用于创建 Python 多行字符串的不同技术。

## 技巧 1:用三重引号在 Python 中创建多行字符串

`triple quotes`可用于一起显示多个字符串，即 Python 中的多行字符串。

**语法:**

```py
variable = """ strings """

```

*   如果我们的输入包含带有太多字符的**字符串语句，那么三重引号可以满足我们以格式化的方式显示它的需要。**
*   三重引号下的所有内容都被视为字符串本身。

**举例:**

```py
inp_str = """You can find the entire set of tutorials for Python and R on JournalDev.
Adding to it, AskPython has got a very detailed version of Python understanding through its easy to understand articles."""
print(inp_str)

```

**输出:**

```py
You can find the entire set of tutorials for Python and R on JournalDev.
Adding to it, AskPython has got a very detailed version of Python understanding through its easy to understand articles.

```

* * *

## 技术 2:使用反斜杠(\)创建多行字符串

转义序列— `backslash ('\')`用于在 Python 中创建多行字符串。

**语法:**

```py
variable = "string1"\"string2"\"stringN"

```

*   当使用反斜杠(\)创建多行字符串时，**用户需要明确地提到字符串之间的空格**。

**举例:**

```py
inp_str = "You can find the entire set of tutorials for Python and R on JournalDev."\
"Adding to it, AskPython has got a very detailed version of Python understanding through its easy to understand articles."\
"Welcome to AskPython!!"
print(inp_str)

```

从下面可以清楚地看到，它不管理语句之间的空格。用户必须在声明多行字符串时提及它。

**输出:**

```py
You can find the entire set of tutorials for Python and R on JournalDev.Adding to it, AskPython has got a very detailed version of Python understanding through its easy to understand articles.Welcome to AskPython!!

```

* * *

## 技巧 3:构建 Python 多行字符串的 string.join()方法

[Python string.join()方法](https://www.askpython.com/python/string/python-string-join-method)被证明是创建 Python 多行字符串的有效技术。

`string.join() method`处理和操纵字符串之间的所有空间，用户不需要担心同样的问题。

**语法:**

```py
string.join(("string1","string2","stringN"))

```

**举例:**

```py
inp_str =' '.join(("You can find the entire set of tutorials for Python and R on JournalDev.",
                   "Adding to it", 
                   "AskPython has got a very detailed version",
                   "of Python understanding through its easy to understand articles.",
                   "Welcome to AskPython!!"))
print(inp_str)

```

**输出:**

```py
You can find the entire set of tutorials for Python and R on JournalDev. Adding to it AskPython has got a very detailed version of Python understanding through its easy to understand articles. Welcome to AskPython!!

```

* * *

## 技术 4: Python 圆括号()创建多行字符串

`Python brackets`可用于创建多行字符串并将字符串拆分在一起。

这种技术的唯一缺点是，用户需要明确地指出多行字符串之间的空格。

**语法:**

```py
variable = ("string1""string2""stringN")

```

**举例:**

```py
inp_str =("You can find the entire set of tutorials for Python and R on JournalDev."
                   "Adding to it "
                   "AskPython has got a very detailed version "
                   "of Python understanding through its easy to understand articles."
                   "Welcome to AskPython!!")
print(inp_str)

```

**输出:**

```py
You can find the entire set of tutorials for Python and R on JournalDev.Adding to it AskPython has got a very detailed version of Python understanding through its easy to understand articles.Welcome to AskPython!!

```

* * *

## 摘要

*   Python 多行字符串是分成多行的字符串，以增强代码对用户的可读性。
*   Python 括号、反斜杠和三重引号可以用来创建多行字符串，但是在这里，用户需要提到字符串之间空格的使用。
*   Python string.join()方法被认为是构建多行字符串的非常有效的方法，而且字符串之间的空格由该方法隐式处理。
*   **Python 缩进**规则**不适用于**多行字符串。
*   如果使用三重引号创建多行字符串，则所有转义序列(如换行符(\n)、制表符(\t)都被视为字符串的一部分。

* * *

## 结论

因此，在本文中，我们已经了解了创建 Python 多行字符串的各种方法。

* * *

## 参考

*   Python 多行字符串— JournalDev