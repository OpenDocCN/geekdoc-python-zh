# Python 字符串到 int 和 int 到 string

> 原文：<https://www.askpython.com/python/built-in-methods/python-string-int-conversion>

在本文中，我们将了解 Python 字符串到 int 的转换以及 Int 到 String 的转换。在 Python 中，这些值从不进行隐式类型转换。让我们看看如何显式地对变量进行类型转换。

## 1.Python 字符串到整数的转换

Python `int()`方法使我们能够将**字符串**类型的任何值转换为**整数**值。

**语法:**

```py
int(string_variable)

```

**举例:**

```py
string_num = '75846'
print("The data type of the input variable is:\n")
print(type(string_num))

result = int(string_num)
print("The data type of the input value after conversion:\n")
print(type(result))
print("The converted variable from string to int:\n")
print(result)

```

**输出:**

```py
The data type of the input variable is:
<class 'str'>
The data type of the input value after conversion:
<class 'int'>
The converted variable from string to int:
75846

```

### 使用不同的基数将 Python 字符串转换为 int

Python 还为我们提供了一个高效的选项，可以根据数字系统将字符串类型的数字/值转换为特定**基**下的整数值。

**语法:**

```py
int(string_value, base = val)

```

**举例:**

```py
string_num = '100'
print("The data type of the input variable is:\n")
print(type(string_num))

print("Considering the input string number of base 8....")
result = int(string_num, base = 8)
print("The data type of the input value after conversion:\n")
print(type(result))
print("The converted variable from string(base 8) to int:\n")
print(result) 

print("Considering the input string number of base 16....")
result = int(string_num, base = 16)
print("The data type of the input value after conversion:\n")
print(type(result))
print("The converted variable from string(base 16) to int:\n")
print(result) 

```

在上面的代码片段中，我们已经将**‘100’**分别转换为以**为基数 8** 和以**为基数 16** 的整数值。

**输出:**

```py
The data type of the input variable is:

<class 'str'>
Considering the input string number of base 8....
The data type of the input value after conversion:

<class 'int'>
The converted variable from string(base 8) to int:

64
Considering the input string number of base 16....
The data type of the input value after conversion:

<class 'int'>
The converted variable from string(base 16) to int:

256

```

### Python 字符串到 int 转换时出现 ValueError 异常

**场景:如果任何输入字符串包含不属于十进制数字系统**的数字。

在下面的例子中，如果您希望**将字符串‘A’转换为基数为 16 的整数值 A，并且我们没有将 base=16 作为** **参数传递给 int()方法**，那么**将引发 ValueError 异常**。

因为即使' **A** 是一个十六进制值，但由于它不属于十进制数系统，它不会认为 A 等同于十进制数 10，除非我们不将 **base = 16** 作为参数传递给 int()函数。

**举例:**

```py
string_num = 'A'
print("The data type of the input variable is:\n")
print(type(string_num))
result = int(string_num)

print(type(result))
print("The converted variable from string(base 16) to int:\n")
print(result) 

```

**输出:**

```py
The data type of the input variable is:

<class 'str'>
Traceback (most recent call last):
  File "main.py", line 4, in <module>
    result = int(string_num)
ValueError: invalid literal for int() with base 10: 'A'

```

### 将 Python 整数列表转换为字符串列表

包含整数元素的 [**Python 列表**](https://www.askpython.com/python/list/python-list) 可以使用 **int()** 方法和**列表理解**转换成字符串值列表。

**举例:**

```py
st_lst = ['121', '144', '111', '564']

res_lst = [int(x) for x in st_lst]
print (res_lst)

```

**输出:**

```py
[121, 144, 111, 564]

```

* * *

## 2.Python 整数到字符串的转换

Python 的`str()`方法使我们能够将**整数**类型的任何值转换为**字符串**值。

**语法:**

```py
str(integer_value)

```

**举例:**

```py
int_num = 100
print("The data type of the input variable is:\n")
print(type(int_num))
result = str(int_num)
print("The data type of the input value after conversion:\n")
print(type(result))
print("The converted variable from int to string:\n")
print(result) 

```

**输出:**

```py
The data type of the input variable is:
<class 'int'>
The data type of the input value after conversion:
<class 'str'>
The converted variable from int to string:
100

```

* * *

## 结论

在本文中，我们已经了解了 Python 字符串到整数的转换，反之亦然。

* * *

## 参考

*   [**Python str()函数文档**](https://docs.python.org/3/library/stdtypes.html#str)
*   [**Python int()函数文档**](https://docs.python.org/3/library/functions.html#int)