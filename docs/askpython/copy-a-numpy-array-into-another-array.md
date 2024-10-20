# 如何将一个 Numpy 数组复制到另一个数组中？

> 原文：<https://www.askpython.com/python-modules/numpy/copy-a-numpy-array-into-another-array>

数组是 Python 中的一种数据结构，它存储相似数据类型的对象。但是有时可能会出现需要将一个数组复制到另一个数组中的情况。在本文中，我们将学习如何将一个 Numpy 数组复制到另一个 Numpy 数组中。所以让我们开始吧！

## 什么是 Numpy 数组？

数组是 Python 中的一种数据结构，它存储相似数据类型的对象。除了列表可以存储多种数据类型的对象之外，它几乎与列表相似。

例如:

```py
import numpy as np
my_array = np.array([1,2,3,4])
print(my_array)

```

输出:

```py
[1 ,2 , 3, 4]

```

## 方法将 Numpy 数组复制到另一个数组中

所以让我们马上看看你可以使用的方法或函数。

### 1.使用 np.copy()函数

这个内置函数将返回与输入数组完全相同的副本。

该函数的语法如下:

numpy.copy(a，order='K ')

让我们看看下面的例子。

```py
import numpy as np

# Creating a numpy array using np.array()
my_array = np.array([1.63, 7.92, 5.46, 66.8, 7.89,
                      3.33, 6.56, 50.60, 100.11])

print("The original array is: ")

print(my_array)

# Now copying the org_array to copy_array using np.copy() function
copy = np.copy(my_array)

print("\nCopied array is: ")

print(copy)

```

输出:

```py
The original array is: 
[  1.63   7.92   5.46  66.8    7.89   3.33   6.56  50.6  100.11]

Copied array is: 
[  1.63   7.92   5.46  66.8    7.89   3.33   6.56  50.6  100.11]

```

### 2.使用赋值运算符

赋值操作符通常在 python 中用来给变量赋值。但是我们也可以用它们将一个数组复制到另一个数组中。

**例如:**

```py
import numpy as np

my_array = np.array([[100, 55, 66 ,44, 77]])

# Copying the original array to copy using Assignment operator
copy = my_array

print('The Original Array: \n', my_array)

print('\nCopied Array: \n', copy)

```

**输出:**

```py
The Original Array: 
 [[100  55  66  44  77]]

Copied Array: 
 [[100  55  66  44  77]]

```

这里，我们简单地将原始数组分配给复制的数组。

### 3.使用 np.empty_like 函数

在这个方法中，我们将首先创建一个类似于原始数组的空数组，然后将原始数组赋给这个空数组。

**该函数的语法如下:**

```py
numpy.empty_like(a, dtype = None, order = ‘K’)

```

让我们来看看下面的例子。

```py
import numpy as np

my_ary = np.array([34, 65, 11, 
                66, 80, 630, 50])

print("The original array is:")

print(my_ary)

# Creating an empty Numpy array similar to the original array
copy = np.empty_like(my_ary)

# Assigning my_ary to copy
copy[:] = my_ary

print("\nCopy of the original array is: ")

print(copy)

```

**输出:**

```py
The original array is:
[ 34  65  11  66  80 630  50]

Copy of the original array is: 
[ 34  65  11  66  80 630  50]

```

## 结论

总之，我们学习了不同的方法和函数，可以用来将一个数组复制到另一个数组中。数组是一种非常有用的数据结构，了解可以对数组执行的不同操作非常重要。