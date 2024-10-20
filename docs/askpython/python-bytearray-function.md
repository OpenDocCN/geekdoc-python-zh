# Python bytearray()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-bytearray-function>

在本文中，我们将了解 Python 的内置函数之一— **Python bytearray()函数**。

## 了解 Python bytearray()函数

[Python](https://www.askpython.com/) 有内置的 bytearray()方法，**创建一个字节数组**，**返回一个创建的数组的 bytearray 对象**，其大小由特定的 iterable 或值定义。

**语法:**

```py
bytearray(source_input, encoding_scheme, error)

```

**参数:**

*   `source_input`:可选参数。它主要用于初始化数组数据元素。source_input 可以是一个**可迭代的**、**值**等。
*   `encoding_scheme`(可选):用于定义字符串的编码模式。
*   `error`(可选):定义编码失败时需要采取的动作。

`bytearray() method`以一个 iterable 如 list、string、tuple 等或 value 作为参数，用 size 初始化一个数组，并返回它的一个 **bytearray 对象**。

**例子:** **无参数的 Python bytearray()函数**

```py
inp = bytearray()
print(inp)

```

当**没有参数**传递给 bytearray()函数时，该函数返回一个**空 bytearray 对象**。

**输出:**

```py
bytearray(b'')

```

### 1.Python bytearray()函数，使用字符串作为参数

当一个**字符串值**作为参数传递给 bytearray()函数时，它会将**字符串转换成一个字节数组**。

强制条件继续保持，使得无论何时在参数列表中传递字符串，都必须在参数列表中为相同的字符串定义**编码**，否则引发`TypeError exception`。

**例 1:**

```py
inp_str = "JournalDev"

arr_8 = bytearray(inp_str, 'utf-8') 

print(arr_8) 

```

这里，我们将 encoding_scheme 作为' **utf-8** '传递，以便将输入字符串转换为字节数组。

**输出:**

```py
bytearray(b'JournalDev')

```

**例 2:**

```py
inp_str = "JournalDev"

arr_16 = bytearray(inp_str, 'utf-16') 

print(arr_16) 

```

在上面的例子中，编码方案被定义为' **utf-16** '。

**输出:**

```py
bytearray(b'\xff\xfeJ\x00o\x00u\x00r\x00n\x00a\x00l\x00D\x00e\x00v\x00')

```

* * *

### 2.将 iterable 作为参数传递的 Python bytearray()函数

当一个 iterable 如 [list](https://www.askpython.com/python/list/python-list) 、 [set](https://www.askpython.com/python/set/python-set) 、 [tuple](https://www.askpython.com/python/tuple/python-tuple) 等作为参数传递给 bytearray()函数时，它返回一个包含**初始内容的字节数组作为 bytearray 对象中的数组元素**。

**强制条件:**如果将 iterable 作为参数传递给 bytearray()函数，则 iterable 的所有元素都必须是类型`integer`以避免类型错误。

**例子:** **Python bytearray()带列表**

```py
inp_lst = [2, 4, 6, 8]
res_byte = bytearray(inp_lst)
print(res_byte)

```

如清楚理解的，**列表的内容，即[2，4，6，8]** 已经被用于创建 bytearray 对象。

**输出:**

```py
bytearray(b'\x02\x04\x06\x08')

```

* * *

### 3.Python bytearray()函数，以整数值作为参数

如果 bytearray()函数遇到一个整数值作为参数，它就用`size = integer value`创建一个 bytearray 对象，然后**用**空值(' \0')** 初始化**。

**举例:**

```py
inp = 14
res_byte = bytearray(inp)
print(res_byte)

```

在上面的例子中，创建了一个数组对象，数组大小为“14”，并用空值初始化。

**输出:**

```py
bytearray(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')

```

* * *

## 摘要

*   Python bytearray()函数返回一个 **bytearray 对象**，即**生成一个提供给该函数的源输入类型的字节数组**。
*   bytearray()函数可以有一个**可迭代的**、**值**等作为它的参数。
*   每当向 iterable 传递函数时，iterable 的元素必须是整数类型。

* * *

## 结论

因此，在本文中，我们已经理解了 Python bytearray()方法使用各种类型的参数的工作原理。

* * *

## 参考

*   python bytearray()–journal dev
*   [Python bytearray()函数——官方文档](https://docs.python.org/3.1/library/functions.html#bytearray)