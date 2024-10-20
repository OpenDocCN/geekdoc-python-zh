# Python 字符串 isspace()方法

> 原文：<https://www.askpython.com/python/string/python-string-isspace>

在本文中，我们将深入研究 **Python String isspace()方法**的工作原理。 **[Python String](https://www.askpython.com/python/string/python-string-functions)** 有生动的内置方法来处理字符串输入。

## Python String isspace()入门

Python string `isspace()`方法用于检查输入字符串中是否存在**空格**。

**空白字符包括**:

*   \n
*   \t
*   \v
*   \f
*   ' '
*   \r
*   等等。

如果输入字符串只包含**和**空格，则返回 **True** 。否则，当字符串包含一个或多个非空白字符时，函数返回 **False** 。

**语法:**

```py
input_string.isspace()

```

**例 1:**

```py
inp = 'Engineering_Discipline'

print(inp.isspace()) 

```

在上面的示例中，由于输入字符串不包含空格，该函数返回 False。

**输出:**

```py
False

```

**例 2:**

```py
inp = '\t \v \n'

print(inp.isspace()) 

```

在本例中，输入字符串只包含空格。因此，该函数的计算结果为 True。

**输出:**

```py
True

```

**例 3:**

```py
inp = '\thello everyone!!\n'

print(inp.isspace()) 

```

在此示例中，输入字符串包含空白字符和非空白字符的组合，即它包含一个或多个非空白字符。

因此，该函数返回 False。

**输出:**

```py
False

```

* * *

## NumPy isspace()方法

Python **[NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)** 为我们提供了 **numpy.char.isspace()** **方法**来检查数组的输入元素中是否存在空格。

**语法**:

```py
numpy.char.isspace(input_array)

```

`numpy.char.isspace()`函数以元素方式检查输入数组中是否存在空格。

也就是说，它检查数组中的每个元素，并为出现的每个元素返回 true 或 false。

**举例:**

```py
import numpy

inp_arr1 = numpy.array([ 'Science', 'Commerce', 'Arts'] ) 
print ("Elements of array1:\n", inp_arr1)  

res1 = numpy.char.isspace(inp_arr1) 
print ("Array1 after using isspace():\n", res1) 

inp_arr2 = numpy.array([ 'Sci\nence', 'Commerce\t', 'Arts'] ) 
print ("Elements of array2:\n", inp_arr2)  

res2 = numpy.char.isspace(inp_arr2) 
print ("Array2 after using isspace():\n", res2) 

inp_arr3 = numpy.array([ '\n\r', '\f\t', ' '] ) 
print ("Elements of array3:\n", inp_arr3)  

res3 = numpy.char.isspace(inp_arr3) 
print ("Array3 after using isspace():\n", res3) 

```

**输出:**

```py
Elements of array1:
 ['Science' 'Commerce' 'Arts']
Array1 after using isspace():
 [False False False]

Elements of array2:
 ['Sci\nence' 'Commerce\t' 'Arts']
Array2 after using isspace():
 [False False False]

Elements of array3:
 ['\n\r' '\x0c\t' ' ']
Array3 after using isspace():
 [ True  True  True]

```

* * *

## Pandas isspace()方法

**[Pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)** 包含 isspace()函数，用于检查序列或数据帧中所有数据的空白字符串。

**语法:**

```py
Series.str.isspace()

```

`Series.str.isspace()`方法检查每个元素的空白字符串是否存在，并只为这些元素返回 **True** 。

**注意** : Series.str.isspace()方法只对字符串类型元素有效。如果解释器遇到任何非字符串值，它会引发 ValueError 异常。

**上述异常可以使用`.astype()`功能进行控制。的。astype()函数将非字符串类型的数据转换为字符串类型。**

**举例:**

```py

import pandas 

inp_data = pandas.Series(['Jim', 'Jonny', ' ', '\t', 'Daisy', '\n']) 

res = inp_data.str.isspace() 

print(res) 

```

**输出:**

```py
0    False
1    False
2     True
3     True
4    False
5     True
dtype: bool

```

* * *

## 结论

在本文中，我们已经了解了 Python string isspace()方法的功能。

* * *

## 参考

Python isspace()方法