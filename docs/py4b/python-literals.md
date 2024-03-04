# Python 文字

> 原文：<https://www.pythonforbeginners.com/basics/python-literals>

在阅读 python 编程材料时，您一定遇到过某些短语，如关键字、变量、常量和文字。在本文中，我们将研究 python 中文字的定义和用法的基本概念。

## 什么是 Python 文字？

文字是编程时赋给变量或常量的原始数据。在 python 中，我们有不同类型的文字，如字符串文字、数字文字、布尔文字和特殊文字 None。在接下来的章节中，我们将研究每种类型的 python 文字，还将研究文字集合。

## Python 中的字符串文字是什么？

字符串是用单引号、双引号或三引号括起来的字符序列。在 python 中，我们有两种类型的字符串，即单行字符串和多行字符串。

单行字符串是遇到换行符时终止的字符串文字。它可以通过在单引号或双引号中包含一个或多个字符来定义，如下所示。

```py
myString="This is a single line string"
anotherString='This is also a single line string'
```

多行字符串可以通过用三重引号将扩展为多行的字符括起来来定义，如下所示。

```py
myString="""This is
a
multiline string.
"""
```

## 什么是数字文字？

数字文字用于表示程序中的数字。在 python 中，我们有不同类型的数值，比如整数、浮点数和复数。

python 中的整数是没有小数部分的数字。表示十进制数的整数可以定义如下。

```py
myNum=1117
```

我们也可以定义其他数系的整数。二进制数表示为以“0b”开头的数字文字，由数字 0 和 1 组成，如下所示。

```py
myNum=0b10101
```

十六进制数字系统中的数字以“0x”开头，可以用 python 表示如下。

```py
myNum=0x123ab
```

八进制数字系统中的整数文字以“0o”开头，可以用 python 表示如下。

```py
myNum=0o124
```

python 中的浮点文字表示由整数和小数组成的实数。十进制数字系统中的浮点数可以表示如下。

```py
myNum=123.345
```

复数的形式是 a+bj，其中“a”代表复数的实部，“b”代表复数的虚部。表示复数的数字文字可以写成如下形式。

```py
myNum=3+4j
```

## 什么是布尔文字？

在 python 中，有两种类型的布尔文字，即 True 和 False。它们可以在程序中定义如下。

```py
myVar=True
myVar1=False
```

## 特殊文字–无

文字“None”用于指定它所赋给的变量不引用任何对象。值为“无”的变量可定义如下。

```py
myObj=None
```

## Python 中的文字集合

有不同类型的集合对象，如 python 字典、列表、集合和元组，我们可以在其中存储 python 文本。

我们可以在 python 列表中存储不同类型的数据，如下所示。

```py
myList=[1,2,"pfb",3.14]
```

我们甚至可以使用 [python string split](https://www.pythonforbeginners.com/dictionary/python-split) 操作在空格处断开字符串文字，并将子字符串保存在一个列表中，如下所示。

```py
myStr="I am a single line string"
myList=myStr.split()
print("String literal is:",myStr)
print("Created list is:",myList)
```

输出:

```py
String literal is: I am a single line string
Created list is: ['I', 'am', 'a', 'single', 'line', 'string']
```

元组也像列表一样包含数据，但是元组是不可变的，新元素不能添加到元组中，或者任何元素不能从元组中删除。我们可以如下定义一个元组。

```py
myTuple=(1,2,"pfb",3.14)
```

一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)包含如下键值对形式的数据。

```py
myDict={1:1,2:"pfb","name":'Chris', "Value":100.5}
```

python 中的集合可以保存无序和不重复的数据。集合中的每个元素都是唯一的。我们可以如下定义一个包含几个元素的集合。

```py
mySet={1,2,"pfb",3.14}
```

## 结论

在这篇文章中，我们已经了解了不同类型的 python 文本，我们也看到了在 python 中如何使用列表、字典、元组和集合等集合来存储文本。

要了解更多关于 python 编程的知识，你可以阅读这篇关于[字符串操作](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)的文章。你可能也会喜欢这篇关于如何在 Python 中使用 [sys.argv list 的文章。](https://www.pythonforbeginners.com/argv/how-to-use-sys-arv-in-python)

请继续关注更多内容丰富的文章。