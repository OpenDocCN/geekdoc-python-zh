# 在 Python 中追加多个字符串的 4 种简单方法

> 原文：<https://www.askpython.com/python/string/append-multiple-strings-in-python>

在这篇文章中，我们将看看在 Python 中插入和追加多个字符串的不同方法。字符串插值涉及在特定语句中注入字符串。让我们开始吧！

* * *

## 技巧 1: f-string 在 Python 中追加多个字符串

Python f-string 也被称为**格式——string**已经被证明是处理字符串的一种有效和最佳的方式。f 字符串在 [PEP 498](https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep498) 下作为**文字字符串插补**引入。

`f-string`主要用于`string interpolation`的目的，即将多个字符串注入到一个语句或结构中。

**语法**:

```py
f'{string1} {string2} {stringN}'

```

*   **{}** :要插值的**字符串**放在花括号之间。

**例 1:**

```py
str1 = 'Python'
str2 = '@'
str3 = 'JournalDev'
res = f'{str1} {str2} {str3}'
print("Appending multiple strings using f-string:\n")
print(res)

```

**输出:**

```py
Appending multiple strings using f-string:

Python @ JournalDev

```

**例 2:**

```py
str1 = 'Python'
str2 = 'and'
str3 = 'R'
str4 = '@ JournalDev'
res = f'{str1} {str2} {str3} {str4}'
print("Appending multiple strings using f-string:\n")
print(res)      

```

**输出:**

```py
Appending multiple strings using f-string:

Python and R @ JournalDev

```

* * *

## 技术 2:追加多个字符串的 Python format()方法

Python string.format()函数也可以用来高效地格式化字符串。

**语法:**

**1。**使用 format()函数进行单字符串格式化

```py
{}.format(string)

```

**2。**使用 format()函数格式化多个字符串

```py
{} {}.format(string1, string2)

```

`string.format() function`格式化字符串，并通过位置格式化帮助**替换字符串，即根据字符串在函数参数列表中的位置。**

**例 1:**

```py
str1 = 'Python'
str2 = '@'
str3 = 'JournalDev'
res = "{} {} {}".format(str1, str2, str3)
print("Appending multiple strings using format():\n")
print(res)      

```

**输出:**

```py
Appending multiple strings using format():

Python @ JournalDev

```

**例 2:**

```py
str1 = 'Python'
str2 = 'and'
str3 = 'Java'
str4 = '@ JournalDev'
res = "{} {} {} {}".format(str1, str2, str3, str4)
print("Appending multiple strings using format():\n")
print(res)      

```

**输出:**

```py
Appending multiple strings using format():

Python and Java @ JournalDev

```

让我们学习更多在 Python 中追加多个字符串的方法。

* * *

## 技巧 3:使用“+”操作符追加多个字符串

Python **串联运算符**即`'+' operator`可用于将多个字符串追加在一起。

**语法:**

```py
string1 + string2 + ..... + stringN

```

**例 1:**

```py
str1 = 'Python'
str2 = '@'
str3 = 'JournalDev'

res = str1 + str2 + str3
print("Appending multiple strings using Python '+' operator:\n")
print(res)      

```

**输出**:

```py
Appending multiple strings using Python '+' operator:

[email protected]

```

**例 2:**

```py
str1 = 'Python'
str2 = '+'
str3 = 'R'
str4 = "@JournalDev"

res = str1 + str2 + str3 + str4
print("Appending multiple strings using Python '+' operator:\n")
print(res)      

```

**输出:**

```py
Appending multiple strings using Python '+' operator:

[email protected]

```

* * *

## 技巧 4:Python“%”操作符追加多个字符串

Python `'%' operator`的作用是字符串格式化和插值。

**语法:**

```py
"%s" % (string)

```

' **%s'** 作为一个**占位符**，用**括号()**中传递的字符串替换它。

**例 1:**

```py
str1 = 'Python'
str2 = '+'
str3 = 'Java'

res = "%s %s %s" % (str1, str2, str3)
print("Appending multiple strings using Python '%' operator:\n")
print(res)      

```

**输出:**

```py
Appending multiple strings using Python '%' operator:

Python + Java

```

**例 2:**

```py
str1 = 'Python'
str2 = '+'
str3 = 'Java'
str4 = '@ journalDev'

res = "%s %s %s %s" % (str1, str2, str3, str4)
print("Appending multiple strings using Python '%' operator:\n")
print(res)

```

**输出:**

```py
Appending multiple strings using Python '%' operator:

Python + Java @ journalDev

```

* * *

## 结论

因此，在本文中，我们已经了解了在 Python 中插入和追加多个字符串的不同方法。

* * *

## 参考

*   Python 字符串串联–journal dev