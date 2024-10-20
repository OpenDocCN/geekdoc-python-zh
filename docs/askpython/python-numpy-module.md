# Python NumPy 模块–需要了解的 4 种重要函数类型

> 原文：<https://www.askpython.com/python-modules/numpy/python-numpy-module>

嘿，伙计们！希望你们都过得好。在本文中，我们将关注于 **Python NumPy 模块的重要功能。**

所以，让我们开始吧！

* * *

## Python NumPy 模块简介

[Python NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)集成了多种函数，轻松执行不同的科学和数学运算。

因此，NumPy 模块可以被认为是一个所有程序员都可以方便地执行所有数学和复杂计算任务的模块。

现在，让我们了解并实现 NumPy 模块的一些重要功能。

* * *

### 1.NumPy 数组操作函数

NumPy 模块的数组操作功能帮助我们对数组元素进行修改。

看看下面的功能——

*   [numpy . shape()](https://www.askpython.com/python-modules/numpy/python-numpy-reshape-function):这个函数允许我们在不影响数组值的情况下改变数组的维数。
*   numpy.concatenate():以行或列的方式连接两个相同形状的数组。

现在让我们把注意力集中在上述功能的实现上。

**举例:**

```py
import numpy

arr1 = numpy.arange(4)
print('Elements of an array1:\n',arr1)

arr2 = numpy.arange(4,8)
print('Elements of an array2:\n',arr2)

res1 = arr1.reshape(2,2)
print('Reshaped array with 2x2 dimensions:\n',res1)

res2 = arr2.reshape(2,2)
print('Reshaped array with 2x2 dimensions:\n',res2)

print("Concatenation two arrays:\n")
concat = numpy.concatenate((arr1,arr2),axis=1)
print(concat)

```

理解形状很重要，即阵列的尺寸需要相同才能执行上述功能。

**输出:**

```py
Elements of an array1:
 [0 1 2 3]
Elements of an array2:
 [4 5 6 7]
Reshaped array with 2x2 dimensions:
 [[0 1]
 [2 3]]
Reshaped array with 2x2 dimensions:
 [[4 5]
 [6 7]]
Concatenation two arrays:

[0 1 2 3 4 5 6 7]

```

* * *

### 2.NumPy 字符串函数

使用 NumPy 字符串函数，我们可以操作数组中包含的字符串值。下面提到了一些最常用的字符串函数:

*   `numpy.char.add() function`:连接两个数组的数据值，合并它们，并作为结果表示一个新的数组。
*   `numpy.char.capitalize() function`:将整个单词/字符串的第一个字符大写。
*   `numpy.char.lower() function`:将字符串字符的大小写转换为小写字符串。
*   `numpy.char.upper() function`:将字符串字符的大小写转换为大写字符串。
*   `numpy.char.replace() function`:用另一个字符串值替换一个字符串或字符串的一部分。

**举例:**

```py
import numpy

res =  numpy.char.add(['Python'],[' JournalDev'])

print("Concatenating two strings:\n",res)

print("Capitalizing the string: ",numpy.char.capitalize('python data'))

print("Converting to lower case: ",numpy.char.lower('PYTHON'))

print("Converting to UPPER case: ",numpy.char.upper('python'))

print("Replacing string within a string: ",numpy.char.replace ('Python Tutorials with AA', 'AA', 'JournalDev'))

```

**输出:**

```py
Concatenating two strings:
 ['Python JournalDev']
Capitalizing the string:  Python data
Converting to lower case:  python
Converting to UPPER case:  PYTHON
Replacing string within a string:  Python Tutorials with JournalDev

```

* * *

### 3.NumPy 算术函数

下面提到的 NumPy 函数用于对数组的数据值执行基本的算术运算

*   `numpy.add() function`:将两个数组相加，返回结果。
*   `numpy.subtract() function`:从 array1 中减去 array2 的元素，返回结果。
*   `numpy.multiply() function`:将两个数组的元素相乘并返回乘积。
*   `numpy.divide() function`:用 array1 除以 array2，返回数组值的商。
*   `numpy.mod() function`:执行取模运算，返回余数数组。
*   `numpy.power() function`:返回数组 1 ^数组 2 的指数值。

**举例:**

```py
import numpy as np 
x = np.arange(4) 
print("Elements of array 'x':\n",x)

y = np.arange(4,8) 
print("Elements of array 'y':\n",y)

add = np.add(x,y)
print("Addition of x and y:\n",add)

subtract = np.subtract(x,y)
print("Subtraction of x and y:\n",subtract)

mul = np.multiply(x,y)
print("Multiplication of x and y:\n",mul)

div = np.divide(x,y)
print("Division of x and y:\n",div)

mod = np.mod(x,y)
print("Remainder array of x and y:\n",mod)

pwr = np.power(x,y)
print("Power value of x^y:\n",pwr)

```

**输出:**

```py
Elements of array 'x':
 [0 1 2 3]
Elements of array 'y':
 [4 5 6 7]
Addition of x and y:
 [ 4  6  8 10]
Subtraction of x and y:
 [-4 -4 -4 -4]
Multiplication of x and y:
 [ 0  5 12 21]
Division of x and y:
 [ 0\.          0.2         0.33333333  0.42857143]
Remainder array of x and y:
 [0 1 2 3]
Power value of x^y:
 [   0    1   64 2187]

```

* * *

### 4.NumPy 统计函数

NumPy 统计函数在数据挖掘和分析数据中的大量特征方面非常有用。

让我们来看看一些常用的函数

*   `numpy.median()`:计算传递数组的中值。
*   `numpy.mean()`:返回数组数据值的平均值。
*   `numpy.average()`:返回所传递数组的所有数据值的平均值。
*   `numpy.std()`:计算并返回数组数据值的标准差。

**举例:**

```py
import numpy as np 
x = np.array([10,20,30,4,50,60]) 

med = np.median(x)
print("Median value of array: \n",med)

mean = np.mean(x)
print("Mean value of array: \n",mean)

avg = np.average(x)
print("Average value of array: \n",avg)

std = np.std(x)
print("Standard deviation value of array: \n",std)

```

**输出:**

```py
Median value of array: 
 25.0
Mean value of array: 
 29.0
Average value of array: 
 29.0
Standard deviation value of array: 
 20.2895703914

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

在那之前，学习愉快！！

* * *

**参考文献**

*   Python NumPy 模块— JournalDev