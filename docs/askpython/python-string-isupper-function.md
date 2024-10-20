# Python String isupper()函数

> 原文：<https://www.askpython.com/python/string/python-string-isupper-function>

Python 中的 String 有内置的函数，可以对字符串执行几乎所有的操作。Python String isupper()函数检查一个字符串中的所有字符是否都是大写的，然后返回 true，否则返回 false。

**要点:**

*   **返回类型:**布尔型，即真或假
*   **参数值:**不需要参数
*   它不区分空间，但区分大小写
*   空字符串也会返回 False。

* * *

## String isupper()语法

```py
str_name.isupper()

```

这里的 str_name 指的是输入字符串。并且，isupper()是 python 中内置的字符串函数。

```py
str_name = "WELCOME"
print(str_name.isupper())   # True

```

* * *

## String isupper()示例

下面给出不同的案例。

### 情况 1:字符串中的每个字符都是大写的，还包含空格/数字/特殊字符

```py
str_name = "WELCOME PYTHON USER"
print(str_name.isupper())   # True

str_name = "WELCOME 2019"
print(str_name.isupper())   # True

str_name = "WELCOME @ 2020"
print(str_name.isupper())   # True

```

### 情况 2:字符串只包含数字或特殊字符

```py
str_name = "2020"
print(str_name.isupper())   # False

str_name = "@$&"
print(str_name.isupper())   # False

```

### 情况 3:字符串中的每个字符都是小写的，还包含空格/数字/特殊字符

```py
str_name = "welcome python user"
print(str_name.isupper())   # False

str_name = "welcome 2019"
print(str_name.isupper())   # False

str_name = "welcome @ 2020"
print(str_name.isupper())   # False

```

### 情况 4:只有每个单词的第一个字符是大写的，还包含空格/数字/特殊字符

```py
str_name = "Welcome"
print(str_name.isupper())   # False

str_name = "Welcome Python User"
print(str_name.isupper())   # False

str_name = "Welcome 2019"
print(str_name.isupper())   # False

str_name = "Welcome @ 2020"
print(str_name.isupper())   # False

```

### 情况 5:字符串为空

```py
str_name = ' '
print(str_name.isupper())   # False

```

* * *

## 打印 Python 中所有可能的大写字符列表的程序

Unicode 模块可用于检查大写字符。该计划是打印所有大写 Unicode 字符。

```py
import unicodedata

total_count = 0
for i in range(2 ** 16):
    charac = chr(i)
    if charac.isupper():
        print(u'{:04x}: {} ({})'.format(i, charac, unicodedata.name(charac, 'UNNAMED')))
        total_count = total_count + 1
print("Total Count of Unicode UPPERCASE Characters = ",total_count)

```

![Output All Uppercase Unicode Characters](img/8372df5b0338a2cebb8dbf60735cc0dc.png)

Output All Uppercase Unicode Characters

这只是输出的一瞥，因为实际输出很长。Unicode 中有 999 个大写字符。

## 参考

*   Python 字符串 isupper()
*   [Python 内置函数](https://docs.python.org/3/library/stdtypes.html)