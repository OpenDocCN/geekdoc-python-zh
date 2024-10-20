# python–将字符串转换为列表

> 原文：<https://www.askpython.com/python/string/python-convert-string-to-list>

在 Python 中，如果您需要处理对其他 API 执行各种调用的代码库，可能会出现这样的情况:您可能会收到类似列表格式的字符串，但仍然不是显式的列表。在这种情况下，您可能希望将字符串转换为列表。

在本文中，我们将研究在 Python 上实现相同功能的一些方法。

* * *

## 转换列表类型字符串

一个*列表类型的*字符串可以是一个在列表中有左括号和右括号并且列表元素有逗号分隔字符的字符串。它和列表之间的唯一区别是左引号和右引号，这表示它是一个字符串。

示例:

```py
str_inp = '["Hello", "from", "AskPython"]'

```

让我们看看如何将这些类型的字符串转换成列表。

### 方法 1:使用 ast 模块

Python 的`ast`(抽象语法树)模块是一个方便的工具，可以用来处理这样的字符串，相应地处理给定字符串的内容。

我们可以使用`ast.literal_eval()`来评估文字，并将其转换为列表。

```py
import ast

str_inp = '["Hello", "from", "AskPython"]'
print(str_inp)
op = ast.literal_eval(str_inp)
print(op)

```

**输出**

```py
'["Hello", "from", "AskPython"]'
['Hello', 'from', 'AskPython']

```

### 方法 2:使用 json 模块

Python 的`json`模块也为我们提供了可以操作字符串的方法。

特别是，`json.loads()`方法用于解码 JSON 类型的字符串并返回一个列表，我们可以相应地使用它。

```py
import json

str_inp = '["Hello", "from", "AskPython"]'
print(str_inp)
op = json.loads(str_inp)
print(op)

```

输出和以前一样。

### 方法 3:使用 str.replace()和 str.split()

我们可以使用 Python 内置的`str.replace()`方法，手动迭代输入字符串。

我们可以在使用`str.split(",")`向新形成的列表添加元素时删除左括号和右括号，手动解析列表类型的字符串。

```py
str_inp = '["Hello", "from", "AskPython"]'
str1 = str_inp.replace(']','').replace('[','')
op = str1.replace('"','').split(",")
print(op)

```

**输出**:

```py
['Hello', ' from', ' AskPython']

```

* * *

## 转换逗号分隔的字符串

*逗号分隔的字符串是指*有一系列字符，用逗号分隔，并包含在 Python 的字符串引号中。

示例:

```py
str_inp = "Hello,from,AskPython'

```

要将这些类型的字符串转换为元素列表，我们有一些其他的方法来执行这项任务。

### 方法 1:使用 str.split('，')

我们可以通过使用`str.split(',')`分隔逗号来直接将其转换成列表。

```py
str_inp = "Hello,from,AskPython"
op = str_inp.split(",")
print(op)

```

**输出**:

```py
['Hello', 'from', 'AskPython']

```

### 方法 2:使用 eval()

如果输入字符串是可信的，我们可以启动一个交互式 shell，并使用`eval()`直接评估该字符串。

然而，由于运行潜在的不可信代码的安全隐患，这是**而不是**推荐的，并且应该避免。

即便如此，如果你还想用这个，那就用吧。我们警告过你！

```py
str_inp = "potentially,untrusted,code"

# Convert to a quoted string so that
# we can use eval() to convert it into
# a normal string
str_inp = "'" + str_inp + "'"
str_eval = ''

# Enclose every comma within single quotes
# so that eval() can separate them
for i in str_inp:
    if i == ',':
        i = "','"
    str_eval += i

op = eval('[' + str_eval + ']')
print(op)

```

输出将是一个列表，因为字符串已经被求值，并且插入了一个括号来表示它`op`是一个列表。

**输出**

```py
['potentially', 'untrusted', 'code']

```

这很长，不建议解析出逗号分隔的字符串。在这种情况下，使用`str.split(',')`是显而易见的选择。

* * *

## 结论

在本文中，我们学习了一些将列表转换成字符串的方法。我们处理了列表类型的字符串和逗号分隔的字符串，并将它们转换成 Python 列表。

## 参考

*   [StackOverflow 关于将字符串转换为列表的帖子](https://stackoverflow.com/questions/5387208/how-to-convert-a-string-with-comma-delimited-items-to-a-list-in-python)

* * *