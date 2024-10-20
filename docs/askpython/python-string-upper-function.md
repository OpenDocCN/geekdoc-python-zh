# Python 字符串 upper()函数

> 原文：<https://www.askpython.com/python/string/python-string-upper-function>

Python String upper()函数将完整的字符串转换成大写，并返回一个新的字符串。字符串在 String 中是不可变的，所以原来的字符串值保持不变。

**要点:**

*   **返回类型:**字符串
*   **参数值:**没有参数可以传入 upper()函数。
*   将整个字符串转换成大写
*   它不会修改原始字符串。修改后的字符串可以用新的变量名保存。

**示例:**给定字符串-“祝您愉快”或“祝您愉快”或“祝您愉快”或“祝您愉快”

使用 upper()函数后的新字符串:“祝您有美好的一天”(针对以上给出的所有字符串)

* * *

## String upper()语法

```py
str_name.upper()

```

这里 str_name 指的是输入字符串。并且，upper()是 python 中内置的字符串函数。

```py
str_name = "welcome"
print(str_name.upper())   #  WELCOME

```

* * *

## String upper()示例

### 情况 1:字符串是小写的，可能包含数字/特殊字符/空格

```py
str_name = "welcome 2020"
print(str_name.upper())   #  WELCOME  2020

str_name = "welcome @2020"
print(str_name.upper())   #  WELCOME @2020

```

### 情况 2:字符串是大写的，可能包含数字/特殊字符/空格

```py
str_name = "WELCOME 2020"
print(str_name.upper())   #  WELCOME  2020

str_name = "WELCOME @2020"
print(str_name.upper())   #  WELCOME @2020

```

### 情况 3:字符串中每个单词只有第一个字母是大写的

```py
str_name = "Python"
print(str_name.upper())   #  PYTHON

str_name = "Python 2020"
print(str_name.upper())   #  PYTHON 2020

```

### 情况 4:字符串只包含数字或特殊字符

```py
str_name = "2020"
print(str_name.upper())   #  2020

str_name = "@$&"
print(str_name.upper())   #  @$&

```

### 情况 5:字符串为空

```py
str_name = ' '
print(str_name.upper())   #  (Will not give any error and show empty space as output)

```

* * *

## 参考

*   Python 字符串转换为大写–str . upper()
*   [Python 字符串内置函数](https://docs.python.org/3/library/stdtypes.html)