# 在 Python 中以字符串形式读取文件

> 原文：<https://www.askpython.com/python/examples/read-file-as-string-in-python>

在本文中，我们将尝试理解如何使用 Python 中不同的内置函数和方法将文本文件作为不同格式的字符串读取。

* * *

## 使用 read()方法

我们可以使用 read()方法读取存储在文本文件中的数据。此方法将文本文件中的数据转换为字符串格式。但是首先，我们需要使用 open()函数打开文件。永远记住添加 replace()函数和 read()函数，用指定的字符替换新行字符，这样返回的数据看起来更均匀，可读性更好。

```py
#without replace()

with open("AskPython.txt") as file:
    data = f.read()

print(data)

```

**输出:**

```py
AskPython Website is very useful

Python Programming language

How to read files as strings in python?
```

```py
#using replace() everything is returned in one line.

with open("AskPython.txt") as file:
    data = file.read().replace('\n',' ')

print(data)

```

**输出:**

AskPython 网站很有用。Python 编程语言。如何在 python 中将文件读取为字符串？

## 使用 pathlib 模块

pathlib 是 python 3.2 或更高版本中可用的 Python 模块。它使得文件和文件系统的整体工作更加有效。e 不需要使用 os 和 os.path 函数，有了 pathlib，一切都可以通过操作符、属性访问和方法调用轻松完成。我们使用 read.text()函数从文件中读取字符串格式的数据。如果需要，我们还可以添加 replace()方法和 read.text()，就像前面的示例中解释的那样。

```py
from pathlib import Path

data = Path("AskPython.txt").read_text()
print(data)

```

**输出:**

```py
AskPython Website is very useful
Python Programming language
How to read files as strings in python?
```

## 结论

在工作和开发不同的项目时，很多时候文件都要包含在编程中。为了使处理文件更容易，我们可以通过将数据提取为字符串格式来读取文件中的数据。本文讨论了在 Python 中将文件作为字符串读取的不同方法。

另外，了解如何通过 ***[点击此处解决 Python filenotfounderror 错误。](https://www.askpython.com/python/examples/python-filenotfounderror)***