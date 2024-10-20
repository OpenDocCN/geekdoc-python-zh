# Numpy 布尔数组——初学者简易指南

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-boolean-array>

Numpy 布尔数组是一种数组(值的集合)，可用于表示 Python 编程语言中数组数据结构中存储的逻辑“真”或“假”值。

当需要来自一个或多个复杂变量的单个逻辑值时，结合逻辑运算符使用布尔数组可以是减少运行时计算需求的有效方式。布尔数组在执行某些运算时，在结果数组中也很有用。

虽然乍一看这种结构似乎没什么用处，但它对初学者来说尤其重要，因为初学者在熟悉其他更灵活的复杂 Python 数据类型之前，常常会发现自己在使用布尔变量和数组。

Python 中的布尔数组是使用 [NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-module) python 库实现的。Numpy 包含一种特殊的数据类型，称为
numpy。BooleanArray(count，dtype=bool)。这会产生一个布尔值数组(与位整数相反),其中的值为 0 或 1。

***也读:[Python——NumPy 数组简介](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)***

## 声明 Numpy 布尔数组

可以手动使用 **dtype=bool，**创建布尔数组。在布尔数组中，除“0”、“False”、“None”或空字符串之外的所有值都被视为 True。

```py
import numpy as np
arr = np.array([5, 0.001, 1, 0, 'g', None, True, False, '' "], dtype=bool)
print(bool_arr)

#Output: [True True True False True False True False False]

```

### Numpy 布尔数组–关系运算

当在 numpy 布尔数组上执行关系运算时，在条件匹配的情况下，所有值被打印为**真**，否则其他值被打印为**假**。在下面的代码示例中演示了等效操作，其中检查布尔数组的值是否等于 2。

```py
import numpy as np
A = np.array([2, 5, 7, 3, 2, 10, 2])
print(A == 2)

#Output: [True False False False True False True]

```

对于计算来说，诸如:""、"<=”, and “> = "之类的关系运算也同样有效。

**该操作也适用于高维数组:**

```py
import numpy as np
# a 4x3 numpy array
A = np.array([[35, 67, 23, 90],   [89, 101, 55, 12],   [45, 2, 72, 33]])
print (A>=35)

#Output: [[ True  True  False  True] [ True  True  True False] [ True  False True False]]

```

同样，**真/假**可以用 **0/1** 代替，使用 [**astype()** 对象](https://www.askpython.com/python/built-in-methods/python-astype)将其转换为 int 类型。

```py
import numpy as np
A = np.array([[90, 11, 9, 2, 34, 3, 19, 100,  41], [21, 64, 12, 65, 14, 16, 10, 122, 11], [10, 5, 12, 15, 14, 16, 10, 12, 12], [ 49, 51, 60, 75, 43, 86, 25, 22, 30]])
B = A < 20
B.astype(np.int)

#Output: array([[0, 1, 1, 1, 0, 1, 1, 0, 0],       [0, 0, 1, 0, 1, 1, 1, 0, 1],       [1, 1, 1, 1, 1, 1, 1, 1, 1],       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

```

其中，在 int 类型中，0 表示 False，1 表示 True。

### Numpy 布尔数组–逻辑运算

逻辑运算，例如:AND、OR、NOT、XOR，也可以用下面的语法方法在布尔数组上运算。

```py
numpy.logical_and(a,b)
numpy.logical_or(a,b)
numpy.logical_not(a,b)

# a and b are single variables or a list/array.

#Output: Boolean value

```

### Numpy 布尔数组索引

它是 Numpy 的一个属性，您可以使用它来访问使用布尔数组的数组的特定值。还准备了更多关于[数组索引在这里](https://www.askpython.com/python/array/array-indexing-in-python)。

```py
import numpy as np
# 1D Boolean indexing
A = np.array([1, 2, 3])B = np.array([True, False, True])
print(A[B])
# Output: [1, 3] 

# 2D Boolean indexing
A = np.array([4, 3, 7],  [1, 2, 5])
B = np.array([True, False, True], [False, False, True])
print(A[B])

#Output: [4, 7, 5]

```

### 结论

使用 Numpy 的布尔数组是一种简单的方法，可以确保数组的内容是您所期望的，而不必检查每个元素。希望您已经很好地了解了 numpy 布尔数组，以及如何实现它和对它执行操作。