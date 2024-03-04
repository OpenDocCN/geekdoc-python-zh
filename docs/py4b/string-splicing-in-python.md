# Python 中的字符串拼接

> 原文：<https://www.pythonforbeginners.com/python-strings/string-splicing-in-python>

Python 字符串是用单引号、双引号或三引号括起来的字符序列。python 中的字符串是不可变的。我们可以使用 python 中的字符串拼接来访问字符串中的每个字符。拼接也被称为索引。

## Python 中的字符串拼接是什么？

字符串拼接或索引是一种方法，通过这种方法，我们可以从 python 字符串中访问任何字符或字符组。在 python 中，可以在方括号`[ ]`的帮助下访问字符串的字符或子字符串，就像我们在 python 中访问列表中的元素一样。我们可以使用正索引和负索引来访问字符串中的字符。

在 python 中使用正索引进行字符串拼接时，字符串的第一个字符的索引为 0，后续字符的索引增加 1，直到结束。

例如，我们可以使用下面的程序打印一个字符串的第一个字符、第三个字符和第十一个字符。注意在 python 中**索引是基于 0 的。即第一个字符被赋予索引 0 而不是 1。**

```py
myString="PythonForBeginners"
x=myString[0]
print(x)
x=myString[2]
print(x)
x=myString[10]
print(x)
```

输出

```py
P
t
e
```

在 python 中使用负索引进行字符串拼接时，python 字符串的最后一个字符的索引为-1，向后移动时，每个字符的索引都比前一个字符小 1。

在下面的例子中，我们使用负索引来打印 python 字符串的字符。

```py
myString="PythonForBeginners"
x=myString[-1]
print(x)
x=myString[-5]
print(x)
x=myString[-10]
print(x)
```

输出

```py
s
n
r
```

## 如何从 python 字符串中捕获子字符串？

子字符串是 python 字符串的连续部分。它可以从任何索引开始，在任何索引结束。

使用正索引，我们可以使用方括号 `[ ]`操作符捕获一个子字符串。我们可以指定要包含在子字符串中的字符串的起始字符的索引和最后字符的索引。取出子字符串的语法是`string_name[start_index:last_index]`。位于`start_index`的字符包含在子字符串中，但不包含位于`last_index`的字符。仅包含索引`last_index-1`之前的字符。因此`start_index`是包含性的，而`last_index`是排他性的。

在下面给出的例子中，我们可以看到，`start_index`处的字符已经包含在输出中，而`last_index`处的字符没有包含在输出中。

```py
 myString="PythonForBeginners"
x=myString[0:5]
print(x)
x=myString[6:9]
print(x)
x=myString[0:6]
print(x) 
```

输出

```py
Pytho
For
Python
```

为了捕获从开始到给定索引的子字符串，我们可以将`start_index`值留空。

```py
myString="PythonForBeginners"
x=myString[:6]
print(x)
x=myString[:9]
print(x)
x=myString[:18]
print(x)
```

输出

```py
Python
PythonFor
PythonForBeginners
```

要捕获从给定索引开始到最后一个索引结束的字符串，我们可以简单地将`last_index`值留空。

```py
myString="PythonForBeginners"
x=myString[0:]
print(x)
x=myString[6:]
print(x)
x=myString[9:]
print(x) 
```

输出

```py
PythonForBeginners
ForBeginners
Beginners
```

我们也可以使用负索引从 python 字符串中捕获子字符串，方法同上。

```py
myString="PythonForBeginners"
x=myString[-10:-1]
print(x)
x=myString[:-1]
print(x)
x=myString[-5:-1]
print(x)
```

输出

```py
rBeginner
PythonForBeginner
nner
```

python 中的子字符串也是字符串，我们可以执行类似于[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)、 [python 字符串分割](https://www.pythonforbeginners.com/dictionary/python-split)等操作

例如，我们可以执行字符串连接，如下例所示。

```py
 myString="PythonForBeginners"
x1=myString[:6]
x2=myString[6:9]
x3=myString[9:]

x=x1+x2+x3
print(x) 
```

输出:

```py
PythonForBeginners
```

## 如何从 python 字符串中捕获一个子序列？

python 字符串的子序列是从字符串中取出的字符序列，不会打乱字符在字符串中的顺序。子序列中的字符可能是输入 python 字符串的连续字符，也可能不是。

为了捕获子序列，我们使用方括号 `[ ]` 操作符。从字符串中捕获子序列的语法是`string_name[start_index,end_index,difference]`。`difference`表示要加到`start_index`上的数字，以获得要包含在子序列中的下一个字符的索引，在子序列中包含一个字符后，跳过`difference-1`字符。

```py
myString="PythonForBeginners"
x=myString[0:10:2]
print(x)
x=myString[0:10:3]
print(x) 
```

输出

```py
PtoFr
PhFB
```

## 结论

在本文中，我们研究了 python 中的字符串拼接。我们还看到了如何使用字符串拼接从 python 字符串中捕获子字符串和子序列。请继续关注更多内容丰富的文章。