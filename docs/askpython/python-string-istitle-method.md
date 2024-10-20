# Python 字符串 istitle()方法

> 原文：<https://www.askpython.com/python/string/python-string-istitle-method>

在本文中，我们将揭示 Python 字符串 istitle()函数。 **[Python 字符串](https://www.askpython.com/python/string/python-string-functions)** 有内置函数对输入字符串进行操作。Python String **istitle()** 就是这样一种方法。

## Python String istitle()方法入门

String `istitle()`方法用于检查输入字符串的**标题大小写**，即如果字符串中每个单词的`first character`都是**大写**并且字符串中每个单词的所有剩余字符都是**小写**，则检查并返回 **True** 。

**例 1:**

```py
inp = 'Taj Mahal'
print(inp.istitle()) 

```

在上面的例子中，istitle()函数返回 **True** ，因为对于上面输入的每个单词，只有第一个字符是大写的。

**输出:**

```py
True

```

**例 2:**

```py
inp = 'Taj mahal'
print(inp.istitle()) 

```

在本例中，istitle()方法导致 **False** ，因为输入字符串的第二个单词，即“mahal”没有大写的第一个字符。

**输出:**

```py
False

```

**例 3:**

```py
inp = 'TAJ MAHAL'
print(inp.istitle()) 

```

在本例中，输入字符串的每个字符都是大写的。因此，该函数返回**假**。

**输出:**

```py
False

```

* * *

## NumPy istitle()方法

**[NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)** 内置 istitle()方法，检查输入数组的标题 case 。

`**numpy.char.istitle()**`方法以一种基于**元素的**方式工作。它单独检查数组中每个元素的标题大小写，并返回**真/假**。

**注意:**如果输入字符串包含**零字符**，默认情况下，函数返回 **False** 。

**语法:**

```py
numpy.char.istitle(input_array)

```

**举例:**

```py
import numpy 

inp_arr1 = numpy.array(['TAJ', 'Mahal', '14Pen', '20eraser', 'aMAZON', 'F21Ever']) 

print ("Elements of the array:\n", inp_arr1) 

res1 = numpy.char.istitle(inp_arr1) 
print ("Array after istitle():\n", res1 ) 

```

**输出:**

```py
Elements of the array:
 ['TAJ' 'Mahal' '14Pen' '20eraser' 'aMAZON' 'F21Ever']
Array after istitle():
 [False  True  True False False  True]

```

* * *

## Pandas istitle()方法

**[熊猫模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)** 由 **Series.str.istitle()** 方法组成，用于检查输入数据的标题情况。

**`Series.str.istitle()`** 方法以**元素方式**检查数据集/输入中的所有字符串是否都是标题大小写。

**语法**:

```py
Series.str.istitle()

```

**举例:**

```py
import pandas
res = pandas.Series(['TAJ', 'Mahal', '14Pen', '20eraser', 'aMAZON', 'F21Ever'])
print(res.str.istitle())

```

**如上所述，输入数据中数字的存在不会给函数的输出带来任何变化。**

**输出:**

```py
0    False
1    True
2    True
3    False
4    False
5    True
dtype: bool

```

* * *

## 结论

在本文中，我们确实了解了 Python istitle()函数在各种场景下的工作。

* * *

## 参考

**Python 字符串 istitle()方法**