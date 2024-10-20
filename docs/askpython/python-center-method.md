# 如何使用 Python center()方法？

> 原文：<https://www.askpython.com/python/string/python-center-method>

大家好！在本文中，我们将了解带有 String、NumPy 和 Pandas 模块的 **Python center()函数**。

* * *

## 带字符串的 Python center()方法

[Python String](https://www.askpython.com/python/string) 有很多内置的函数来操作和处理字符串。

`string.center() method`以集中的形式在输入字符串的两边(左右两边)用特定的字符对字符串进行填充。

**语法:**

```py
string.center(width,fillchar)

```

*   `width`:该值决定字符串周围的填充区域。
*   `fillchar`:填充区域由特定字符填充。默认字符是空格。

**例 1:**

```py
inp_str = "Python with JournalDev"
print("Input string: ", inp_str)
res_str = inp_str.center(36) 
print("String after applying center() function: ", res_str)

```

在上面的代码片段中，使用 center()函数，我们用默认的 fillchar 填充了字符串，即参数列表中定义的宽度(36)的空格。

**输出:**

```py
Input string:  Python with JournalDev
String after applying center() function:         Python with JournalDev    

```

**例 2:** **用 center()函数通过特定的 fillchar 填充字符串**

```py
inp_str = "Python with JournalDev"
print("Input string: ", inp_str)
res_str = inp_str.center(36,"$") 
print("String after applying center() function: ", res_str)

```

**输出:**

```py
Input string:  Python with JournalDev
String after applying center() function:  $$$$$$$Python with JournalDev$$$$$$$

```

* * *

## 带有熊猫模块的 Python center()函数

Python center()函数也可以和 [Pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)的数据帧一起使用。

`DataFrame.str.center() function`用传递给函数的字符沿着字符串两边的特定宽度(包括字符串的宽度)填充输入字符串。

**语法:**

```py
DataFrame.str.center(width,fillchar)

```

**输入数据集:**

```py
        contact
1	cellular
2	telephone
3	cellular
4	cellular
5	telephone
6	telephone
7	telephone
8	cellular
9	cellular

```

**举例:**

```py
import pandas
info=pandas.read_csv("C:/marketing_tr.csv")
info_con=pandas.DataFrame(info['contact'].iloc[1:10])
info_con['contact']=info_con['contact'].str.center(width = 15, fillchar = '%') 
print(info_con['contact'])

```

**输出:**

```py
1    %%%%cellular%%%
2    %%%telephone%%%
3    %%%%cellular%%%
4    %%%%cellular%%%
5    %%%telephone%%%
6    %%%telephone%%%
7    %%%telephone%%%
8    %%%%cellular%%%
9    %%%%cellular%%%
Name: contact, dtype: object

```

* * *

## 带有 NumPy 模块的 Python center()函数

Python center()函数可以和 [NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)一起使用，对数组的每个元素进行中心填充。

`numpy.char.center() method`用于将元素居中，甚至通过数组元素两侧的特定字符对其进行填充。

**语法:**

```py
numpy.char.center(array,width,fillchar)

```

**举例:**

```py
import numpy as np 

inp_arr = ['Python', 'Java', 'Kotlin', 'C'] 
print ("Input Array : ", inp_arr) 
res_arr = np.char.center(inp_arr, 15, fillchar = '%')
print("Array after applying center() function: ", res_arr)

```

在上面的例子中，我们已经将 center()函数应用于数组的每个元素，从而使元素居中，并根据两边的宽度用 fillchar 填充数组元素。

**输出:**

```py
Input Array :  ['Python', 'Java', 'Kotlin', 'C']
Array after applying center() function:  ['%%%%%Python%%%%' '%%%%%%Java%%%%%' '%%%%%Kotlin%%%%' '%%%%%%%C%%%%%%%']

```

* * *

## 结论

因此，在本文中，我们已经分别理解了 Python center()函数以及 NumPy 和 Pandas 模块的工作原理。

* * *

## 参考

*   Python center()函数— JournalDev