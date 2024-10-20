# Python 字符串大写()函数

> 原文：<https://www.askpython.com/python/string/python-string-capitalize-function>

Python 中的 String 有内置的函数，可以对字符串执行几乎所有的操作。Python 的 String capitalize()函数用于只将第一个字符转换成大写字母，其余所有字符都是小写的。

**要点:**

*   **返回类型:**字符串
*   **参数值:**没有参数可以解析到 capitalize()函数上。
*   仅将字符串的第一个字符转换为大写。
*   它不会修改原始字符串。修改后的字符串以新的变量名保存。

**示例:**给定字符串-“祝您愉快”或“祝您愉快”或“祝您愉快”或“祝您愉快”

大写的字符串:“祝你有美好的一天”(适用于上述所有字符串)

* * *

## 语法:

```py
str_name.capitalize()

```

这里的 str_name 指的是要大写的字符串。并且，capitalize()是 python 中内置的字符串函数。

### 基本示例

```py
str_name = "hi there!"
new_str = str_name.capitalize()
print('The New Capitalized String is ',new_str)

```

**输出:**你好！

* * *

## 不同情况:

不同情况的示例如下

### 情况 1:字符串中的所有字符都是大写的

```py
str_name = "HI THERE"
new_str = str_name.capitalize()
print('The New Capitalized String is ',new_str)

```

**输出:**你好！

### 情况 2:包含多个单词的字符串中每个单词的第一个字母都是大写的

```py
str_name = "Hi There!"
new_str = str_name.capitalize()
print('The New Capitalized String is ',new_str)

```

**输出:**你好！

### 情况 3:字符串中的任意字符都是大写的

```py
str_name = "hI tHeRE!"
new_str = str_name.capitalize()
print('The New Capitalized String is ',new_str)

```

**输出:**你好！

### 情况 4:非字母数字或数字的第一个字符

```py
str_name = "! hi there"
new_str = str_name.capitalize()
print('The New Capitalized String is ',new_str)

```

**输出:**！你好。

* * *

## 参考

[蟒](https://www.askpython.com/python/python-functions) [月功能](https://www.askpython.com/python/python-functions)