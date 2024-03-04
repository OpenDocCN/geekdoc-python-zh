# Python 中的字符串索引

> 原文：<https://www.pythonforbeginners.com/strings/string-indexing-in-python>

在 Python 中，字符串用于处理文本数据。在处理字符串时，我们经常需要访问字符串的某一部分。在本文中，我们将看到如何使用 Python 中的索引来提取字符串的各个部分。

## 什么是字符串索引？

如果我们有一个有序序列或容器对象，如字符串、列表或元组，我们可以使用它们在序列中的相对位置来访问对象的元素。有序序列中元素的相对位置称为索引。通过索引，我们可以使用索引访问有序序列中的任何元素。

在 python 中，字符串索引是从零开始的。这意味着我们从 0 开始计数，字符串的第一个字符被赋予索引 0，第二个字符被赋予索引 1，第三个字符被赋予索引 2，依此类推。

我们可以通过下面的例子来理解这一点。

假设我们有一个字符串“PythonForBeginners”

这里，字母“P”的索引是 0。字母“y”的索引是 1。字母“t”的索引是 2，字母“h”的索引是 3，依此类推。最后一个字母“s”的索引是 17。

在 python 中，我们可以使用正数和负数进行字符串索引。让我们逐一讨论。

## 使用正数的字符串索引

正如我们在上面看到的，字符串使用正数从 0 到字符串长度-1 进行索引。我们可以使用如下的正索引来访问 0 到(字符串长度)-1 之间任何位置的字符。

```py
myString = "PythonForbeginners"
index = 0
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character))
index = 1
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character))
index = 2
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character))
index = 3
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character))
index = 17
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character))
```

输出:

```py
Character at index 0 in the string 'PythonForbeginners' is P.
Character at index 1 in the string 'PythonForbeginners' is y.
Character at index 2 in the string 'PythonForbeginners' is t.
Character at index 3 in the string 'PythonForbeginners' is h.
Character at index 17 in the string 'PythonForbeginners' is s.
```

请记住，大于或等于字符串长度的索引将导致如下 IndexError 异常。

```py
myString = "PythonForbeginners"
index = 20
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character)) 
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    character = myString[index]
IndexError: string index out of range
```

您可以通过在访问字符串中的任何字符之前检查 index 的值来避免 IndexError 异常。或者，您可以使用 python try except 块来处理出现的异常。

在这里，我会建议你使用 try except 块。如果我们使用小于字符串长度的索引进行访问，那么每次访问一个字符时检查索引可能是多余的，而且代价很高。当使用 try except 块时，程序不会在我们每次访问字符串中的一个字符时检查 index 的值。如果发生 IndexError，它将由 except 块中的代码处理。

## 使用负数进行索引

我们也可以使用负索引来访问字符串中的字符。在 python 中，字符串的最后一个字符被赋予索引-1。倒数第二个字符的索引为-2。类似地，字符串的第一个字符被赋予一个索引-(字符串的长度)。

我们可以通过下面的例子来理解这一点。

假设我们有一个字符串“PythonForBeginners”

这里，字母“s”的索引是-1。字母“r”的索引是-2。字母“n”的索引是-3，字母“n”的索引是-4，依此类推。第一个字母“P”的索引是-18。

您可以使用下面的程序来验证这一点。

```py
myString = "PythonForbeginners"
index = -1
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character))
index = -2
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character))
index = -3
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character))
index = -4
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character))
index = -18
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character)) 
```

输出:

```py
Character at index -1 in the string 'PythonForbeginners' is s.
Character at index -2 in the string 'PythonForbeginners' is r.
Character at index -3 in the string 'PythonForbeginners' is e.
Character at index -4 in the string 'PythonForbeginners' is n.
Character at index -18 in the string 'PythonForbeginners' is P.
```

使用负数作为索引时，请确保传递的索引不小于-(字符串长度)。否则，您的程序将会遇到如下的 IndexError。

```py
myString = "PythonForbeginners"
index = -20
character = myString[index]
print("Character at index {} in the string '{}' is {}.".format(index, myString, character)) 
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    character = myString[index]
IndexError: string index out of range
```

## 结论

在本文中，我们研究了 python 中的字符串索引。我们已经看到了如何使用负数和正数来访问字符串中的字符。要学习更多关于 python 中字符串的知识，你可以阅读这篇关于[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)的文章。