# Python 字符串引号

> 原文：<https://www.pythonforbeginners.com/basics/strings-quotes>

## 概观

字符串可以用单引号或双引号括起来。

单引号字符串可以包含双引号，双引号字符串可以
包含单引号。字符串也可以以多种方式跨越多行。

换行符可以用反斜杠转义，但是每行的
结尾必须有一个反斜杠来转义换行符。

字符串可以用一对匹配的三重引号括起来:" "(双引号)
或" '(单引号)

```py
print """

Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
"""

will show this output:

Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to 
```

最后，我们有原始字符串。

原始字符串禁止转义，所以
序列不会被转换成
换行符，但是行尾的反斜杠和源中的
换行符都作为数据包含在字符串中。

“r”可以是小写或大写，并且必须放在第一个引号
之前。

```py
hello = r"This is a rather long string containing

several lines of text."
print hello 
```

将显示以下输出:

这是一个相当长的字符串，包含

几行文字。

使用单引号还是双引号并不重要，只要确保如果用双引号开始一个值，就必须用双引号结束它。

我通常对正则表达式使用原始字符串，对文档字符串使用三重引号

请看一下[字符串备忘单](https://www.pythonforbeginners.com/basics/strings "strings")了解更多关于 Python 中字符串
的事情。