# Python oct()函数是什么？

> 原文：<https://www.askpython.com/python/built-in-methods/python-oct-function>

嘿，伙计们！在本文中，我们将关注使用 Python oct()函数的整数值的**八进制表示。**

* * *

## Python oct()函数入门

Python 有各种内置函数来操纵和操作数据值。`Python oct() method`用于将表示整数值转换成八进制格式的表示法。

**语法:**

```py
oct(number)

```

*   `number`:必须传递给函数的整数值。它可以是一个**十进制**、**二进制**或**十六进制**的值。

**举例:**

```py
dec_num = 2
print("Octal representation of decimal number:", oct(dec_num))
bin_num = 0b10
print("Octal representation of binary number:", oct(bin_num))
hex_num = 0x17
print("Octal representation of decimal number:", oct(hex_num))

```

**输出:**

```py
Octal representation of decimal number: 0o2
Octal representation of binary number: 0o2
Octal representation of decimal number: 0o27

```

* * *

### Python oct()函数的错误和异常

如果将一个**浮点类型值**传递给 Python oct()函数，它会引发一个`TypeError`异常，即 oct()函数只接受整数值作为参数。

**举例:**

```py
dec_num = 2.4
print("Octal representation of decimal number:", oct(dec_num))

```

**输出:**

```py
TypeError                                 Traceback (most recent call last)
<ipython-input-3-75c901e342e0> in <module>
      1 dec_num = 2.4
----> 2 print("Octal representation of decimal number:", oct(dec_num))

TypeError: 'float' object cannot be interpreted as an integer

```

* * *

## NumPy 模块中数组元素的八进制表示

使用 [NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-arrays) 的内置函数，可以将数据结构中包含的元素(如 NumPy 数组、列表等)转换为其**八进制**格式。

`numpy.base_repr() function`用于以元素方式将数组的每个元素转换成八进制形式。

**语法:**

```py
numpy.base_repr(number, base, padding)

```

*   `number`:需要表示八进制格式的数组元素。
*   `base`:表示要转换的值的合成数字系统的值。对于八进制表示，我们需要放置 **base = 8** 。
*   `padding`:要加到结果数左轴的零的个数。

**举例:**

```py
import numpy as N
arr = [23,15,36,20]
res_arr = N.base_repr(arr[0],base = 8)
print("The Octal representation of",arr[0],":",res_arr)

```

**输出:**

```py
The Octal representation of 23 : 27

```

* * *

## 使用 Pandas 模块的数据值的八进制表示

[Python Pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)用于以数据帧的形式构建元素，并对数据集进行操作。Python `apply(oct) function`可用于将数据集的整数数据值表示成八进制格式。

**语法:**

```py
DataFrame.apply(oct)

```

**举例:**

```py
import pandas as pd
dec_data=pd.DataFrame({ 'Decimal-values':[20,32,7,23] })
oct_data=dec_data['Decimal-values'].apply(oct)
print(oct_data)

```

**输出:**

```py
0    0o24
1    0o40
2     0o7
3    0o27
Name: Decimal-values, dtype: object

```

* * *

## 结论

因此，在本文中，我们已经理解了使用 Python oct()函数将整数值表示为八进制格式的方法。

* * *

## 参考

*   Python oct()函数— JournalDev