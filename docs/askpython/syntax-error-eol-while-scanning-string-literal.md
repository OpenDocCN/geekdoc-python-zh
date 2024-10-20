# 语法错误:扫描字符串文字时出现 EOL

> 原文：<https://www.askpython.com/python/syntax-error-eol-while-scanning-string-literal>

Python 是一种解释型语言，本质上意味着每一行代码都是一个一个执行的，而不是一次把整个程序转换成更低级别的代码。

当 Python 解释器扫描每一行代码并发现异常时，它会引发一个错误，称为**语法错误**。这些错误可能是由“缺少括号”、“缺少结束引号”和语法中的其他基本异常引起的。

本文中我们将要讨论的语法错误是“扫描字符串文字时的 EOL”。

## 这个错误是什么意思？

我们不能解决一个问题，除非我们有效地理解它。 **EOL** 代表“生产线末端”。该错误意味着 Python 解释器在尝试扫描字符串文字时到达了行尾。

字符串文字(常量)必须用单引号和双引号括起来。扫描时到达*“行尾”*是指到达字符串的最后一个字符，没有遇到结束引号。

```py
# String value
s = "This is a string literal...

# Printing the string 
print(s)

```

运行上述代码会产生以下输出:

```py
  File "EOL.py", line 2
    s = "This is a string literal...
                                   ^
SyntaxError: EOL while scanning string literal

```

小箭头指向字符串的最后一个字符，表示在解析语句的该部分时发生了错误。

既然我们已经理解了这个问题，那么让我们来看一些在运行 python 代码时会出现这个问题的例子。

* * *

## 如何修复“扫描字符串文字时出现语法错误:EOL”

可能会在四种主要情况下遇到此错误:

### 缺少结束引号

正如上面的代码片段所解释的，当 Python 解释器到达字符串文字的末尾并发现引号丢失时，它会引发一个语法错误。

```py
# Situation #1

# Missing the ending quotation mark
s = "This is a string literal...

# Printing the string 
print(s)

```

这个语法错误的原因很明显。每种语言都有一些基本的语法规则，违反这些规则会导致错误。

**解决方案:**

简单的解决方法是遵守语法规则并加上引号。

```py
# Solution #1

# Place the ending quotation mark
s = "This is a string literal..."

# Printing the string 
print(s)

```

* * *

### 使用不正确的结束引号

Python 允许使用`' '`和`" "`来封装字符串常量。有时程序员使用不正确的引用来结束字符串值。

```py
# Situation #2

# Incorrect ending quotation mark
s = "This is a string literal...'

# Printing the string 
print(s)

```

尽管字符串看起来是封闭的，但事实并非如此。Python 解释器在字符串末尾搜索匹配的引号。

**解决方案:**

基本的解决方案是匹配开始和结束引号。

```py
#		Solution #2

# Match the quotation marks
s = "This is a string literal..."

# Printing the string 
print(s)

```

* * *

### 字符串常量拉伸到多行

许多初学 Python 的程序员会犯将语句拉长到多行的错误。Python 认为新的一行是语句的结尾，不像 C++和 Java 认为`';'`是语句的结尾。

```py
#		Situation #3

# String extending to multiple lines
s = "This is a string literal...
		  Going to the next line"

# Printing the string 
print(s)

```

起初，代码可能看起来很普通，但是一旦新行开始，Python 解释器就结束该语句，并因没有包含字符串常量而引发错误。

**解决方案 1:**

转义序列`'\n'`可用于向字符串常量提供新行的效果。访问[这里](https://docs.python.org/3/reference/lexical_analysis.html#index-18)了解其他逃脱序列。

```py
#		Solution #3.1

# Using the escape sequences \n -> Newline
s = "This is a string literal... \n Going to the next line"

# Printing the string 
print(s)

```

**解决方案 2:**

另一个解决方案是使用三重引号、`'''`或`"""`来存储多行字符串。

```py
#		Solution #3.2

# Using triple quotation marks 
s = """This is a string literal...
		  Going to the next line"""

# Printing the string 
print(s)

```

* * *

### 在结束引号前使用反斜杠

反斜杠`'\'`负责转义字符串并导致语法错误。

```py
#		Situation #4

# Storing a directory path 
s = "\home\User\Desktop\"

# Printing the string 
print(s)

```

引号前的最后一个反斜杠对字符串常量进行转义，Python 解释器将`\"`视为单个字符。这个转义序列被翻译成引号`(")`。

**解决方案:**

解决方案是用转义序列替换反斜杠`(\\)`。

```py
#		Solution #4

# Storing a directory path 
s = "\\home\\User\\Desktop\\"

# Printing the string 
print(s)

```

* * *

## 结论

长达一千行的代码中的一个错误可能会花费数小时来调试。因此，建议编写这样的代码时要非常专注，并使用正确的语法。

我们希望这篇文章在解决读者的错误方面有所收获。感谢您的阅读。