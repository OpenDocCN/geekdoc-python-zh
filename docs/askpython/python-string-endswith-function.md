# Python 字符串 endswith()函数

> 原文：<https://www.askpython.com/python/string/python-string-endswith-function>

如果输入字符串以特定后缀结尾，Python string `**endswith()**`函数返回 True，否则返回 False。

**要点**:

*   **返回类型:**布尔型，即真或假
*   **参数值:**有 3 个参数:后缀，开始，结束

| 参数 | 描述 |
| 后缀 | 它可以是要检查的字符串或字符串元组。它区分大小写 |
| 开始 | 它是可选的，用于指定开始检查的起始索引 |
| 结束 | 它是可选的，用于指定检查结束的结束索引 |

* * *

## Python 字符串 endswith()语法

`**string.endswith(suffix[, start[, end]])**`

* * *

## String endswith()示例

**例 1:**

```py
str= 'Engineering Discipline'

print(str.endswith('Discipline'))  # True

```

**示例 2:** 提供偏移

```py
str = 'Engineering is an interesting discipline'

print(str.endswith('discipline', 2))  # True
print(str.endswith('Engineering', 10))  # False

```

**示例 3:** 使用 len()函数和 endswith()函数

```py
str = 'Engineering is an interesting discipline'

print(str.endswith('discipline', 11, len(str)))  # True
print(str.endswith('Engineering', 0, 11))  # True
print(str.endswith('Python', 8))  # False

```

**例 4:**

```py
str = 'C++ Java Python'

print(str.endswith(('Perl', 'Python')))  # True
print(str.endswith(('Java', 'Python'), 3, 8))  # True

```

* * *

## 结论

Python String endswith()函数是一个实用程序，用于检查字符串是否以给定的后缀结尾。

* * *

## 参考

*   Python endswith()函数
*   [Python 字符串文档](https://docs.python.org/3/library/string.html)