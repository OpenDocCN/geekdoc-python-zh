# Python 中的矢量化——完整指南

> 原文：<https://www.askpython.com/python-modules/numpy/vectorization-numpy>

在本文中，我们将学习矢量化。如今许多复杂的系统都要处理大量的数据。与 C/C++等其他语言相比，在 python 中处理如此大量的数据可能会很慢。这就是矢量化发挥作用的地方。在本教程中，我们将学习 NumPy 中对数组的矢量化操作，通过比较 Python 程序的执行时间来加速它们的执行。

## Python 中的矢量化

矢量化是一种不使用 for 循环实现[数组操作](https://www.askpython.com/python/array/python-array-declaration)的技术。相反，我们使用由各种模块定义的函数，这些函数经过高度优化，减少了代码的运行和执行时间。矢量化数组操作将比纯 Python 操作更快，对任何类型的数值计算都有最大的影响。

Python for-loops 比 C/C++要慢。Python 是一种解释型语言，大部分实现都很慢。这种缓慢计算的主要原因归结于 Python 的动态特性和编译器级优化的缺乏，这导致了内存开销。NumPy 是 Python 中数组的 C 实现，它提供了对 NumPy 数组的矢量化操作。

## 使用 NumPy 的矢量化运算

### 1.标量加/减/乘/除

当用给定的标量更新数组的所有元素时，数组与标量的加、减、乘和除会产生相同维数的数组。我们应用这个操作就像我们处理变量一样。与 for 循环实现相比，代码既小又快。

为了计算执行时间，我们将使用执行语句的`timeit`模块中的`Timer`类，然后调用重复执行语句的次数的 [timeit()方法](https://www.askpython.com/python-modules/python-timeit-module)。请注意，输出计算时间并不总是完全相同，而是取决于硬件和其他因素。

```py
import numpy as np
from timeit import Timer

# Creating a large array of size 10**6
array = np.random.randint(1000, size=10**6)

# method that adds elements using for loop
def add_forloop():
  new_array = [element + 1 for element in array]

# method that adds elements using vectorization
def add_vectorized():
  new_array = array + 1

# Finding execution time using timeit
computation_time_forloop = Timer(add_forloop).timeit(1)
computation_time_vectorized = Timer(add_vectorized).timeit(1)

print("Computation time is %0.9f using for-loop"%execution_time_forloop)
print("Computation time is %0.9f using vectorization"%execution_time_vectorized)

```

```py
Computation time is 0.001202600 using for-loop
Computation time is 0.000236700 using vectorization

```

### 2.数组的和与最大值

为了找到数组中的 sum 和 maximum 元素，我们可以分别使用 For 循环以及 python 内置方法`sum()`和`max()`。让我们用 numpy 操作来比较这两种方法。

```py
import numpy as np
from timeit import Timer

# Creating a large array of size 10**5
array = np.random.randint(1000, size=10**5)

def sum_using_forloop():
  sum_array=0
  for element in array:
    sum_array += element

def sum_using_builtin_method():
  sum_array = sum(array)

def sum_using_numpy():
  sum_array = np.sum(array)

time_forloop = Timer(sum_using_forloop).timeit(1)
time_builtin = Timer(sum_using_builtin_method).timeit(1)
time_numpy = Timer(sum_using_numpy).timeit(1)

print("Summing elements takes %0.9f units using for loop"%time_forloop)
print("Summing elements takes %0.9f units using builtin method"%time_builtin)
print("Summing elements takes %0.9f units using numpy"%time_numpy)

print()

def max_using_forloop():
  maximum=array[0]
  for element in array:
    if element > maximum:
      maximum = element

def max_using_builtin_method():
  maximum = max(array)

def max_using_numpy():
  maximum = np.max(array)

time_forloop = Timer(max_using_forloop).timeit(1)
time_builtin = Timer(max_using_built-in_method).timeit(1)
time_numpy = Timer(max_using_numpy).timeit(1)

print("Finding maximum element takes %0.9f units using for loop"%time_forloop)
print("Finding maximum element takes %0.9f units using built-in method"%time_builtin)
print("Finding maximum element takes %0.9f units using numpy"%time_numpy)

```

```py
Summing elements takes 0.069638600 units using for loop
Summing elements takes 0.044852800 units using builtin method
Summing elements takes 0.000202500 units using numpy

Finding maximum element takes 0.034151200 units using for loop
Finding maximum element takes 0.029331300 units using builtin method
Finding maximum element takes 0.000242700 units using numpy

```

这里我们可以看到 numpy 操作比内置方法快得多，内置方法比循环快得多。

### 3.点积

也称为内积，两个向量的点积是一种代数运算，它采用两个长度相同的向量，并返回单个标量。它被计算为两个向量的元素乘积之和。就一个矩阵而言，给定大小为`nx1`的两个矩阵 a 和 b，通过取第一个矩阵的转置，然后进行`a^T`(`a`的转置)和`b`的数学矩阵乘法来完成点积。

在 NumPy 中，我们使用`dot()`方法来寻找 2 个向量的点积，如下所示。

```py
import numpy as np
from timeit import Timer

# Create 2 vectors of same length
length = 100000
vector1 = np.random.randint(1000, size=length)
vector2 = np.random.randint(1000, size=length)

# Finds dot product of vectors using for loop
def dotproduct_forloop():
  dot = 0.0
  for i in range(length):
    dot += vector1[i] * vector2[i]

# Finds dot product of vectors using numpy vectorization
def dotproduct_vectorize():
  dot = np.dot(vector1, vector2)

# Finding execution time using timeit
time_forloop = Timer(dotproduct_forloop).timeit(1)
time_vectorize = Timer(dotproduct_vectorize).timeit(1)

print("Finding dot product takes %0.9f units using for loop"%time_forloop)
print("Finding dot product takes %0.9f units using vectorization"%time_vectorize)

```

```py
Finding dot product takes 0.155011500 units using for loop
Finding dot product takes 0.000219400 units using vectorization

```

### 4.外部产品

两个向量的外积产生一个矩形矩阵。给定大小为 ***nx1*** 和 ***mx1*** 的两个向量 ***a*** 和***b***b 和【nxm】，这些向量的外积产生大小为 ***的矩阵。***

在 NumPy 中，我们使用`outer()`方法来寻找两个向量的外积，如下所示。

```py
import numpy as np
from timeit import Timer

# Create 2 vectors of same length
length1 = 1000
length2 = 500
vector1 = np.random.randint(1000, size=length1)
vector2 = np.random.randint(1000, size=length2)

# Finds outer product of vectors using for loop
def outerproduct_forloop():
  outer_product = np.zeros((length1, length2), dtype='int')
  for i in range(length1):
    for j in range(length2):
      outer_product[i, j] = vector1[i] * vector2[j]

# Finds outer product of vectors using numpy vectorization
def outerproduct_vectorize():
  outer_product = np.outer(vector1, vector2)

# Finding execution time using timeit
time_forloop = Timer(outerproduct_forloop).timeit(1)
time_vectorize = Timer(outerproduct_vectorize).timeit(1)

print("Finding outer product takes %0.9f units using for loop"%time_forloop)
print("Finding outer product takes %0.9f units using vectorization"%time_vectorize)

```

```py
Finding outer product takes 0.626915200 units using for loop
Finding outer product takes 0.002191900 units using vectorization

```

### 5.矩阵乘法

[矩阵乘法](https://www.askpython.com/python/python-matrix-tutorial)是第一个矩阵的行乘以第二个矩阵的列的代数运算。对于 2 维矩阵 ***p x q*** 和 ***r x s，*** 一个必要条件是 ***q == r*** 对于 2 个矩阵相乘。乘法后得到的矩阵将具有维度 ***p x s*** 。

矩阵乘法是机器学习等数学模型中广泛使用的运算。计算矩阵乘法是计算成本很高的操作，并且需要快速处理以使系统快速执行。在 NumPy 中，我们用`matmul()`方法求 2 个矩阵的矩阵乘法，如下图。

```py
import numpy as np
from timeit import Timer

# Create 2 vectors of same length
n = 100
k = 50
m = 70
matrix1 = np.random.randint(1000, size=(n, k))
matrix2 = np.random.randint(1000, size=(k, m))

# Multiply 2 matrices using for loop
def matrixmultiply_forloop():
  product = np.zeros((n, m), dtype='int')
  for i in range(n):
    for j in range(m):
      for z in range(k):
        product[i, j] += matrix1[i, z] * matrix2[z, j]

# Multiply 2 matrices using numpy vectorization
def matrixmultiply_vectorize():
  product = np.matmul(matrix1, matrix2)

# Finding execution time using timeit
time_forloop = Timer(matrixmultiply_forloop).timeit(1)
time_vectorize = Timer(matrixmultiply_vectorize).timeit(1)

print("Multiplying matrices takes %0.9f units using for loop"%time_forloop)
print("Multiplying matrices takes %0.9f units using vectorization"%time_vectorize)

```

```py
Multiplying matrices takes 0.777318300 units using for loop
Multiplying matrices takes 0.000984900 units using vectorization

```

### 6.矩阵中的元素乘积

两个矩阵的元素乘积是一种代数运算，其中第一个矩阵的每个元素都乘以第二个矩阵中相应的元素。矩阵的维数应该相同。

在 NumPy 中，我们使用`*`运算符来查找 2 个向量的元素乘积，如下所示。

```py
import numpy as np
from timeit import Timer

# Create 2 vectors of same length
n = 500
m = 700
matrix1 = np.random.randint(1000, size=(n, m))
matrix2 = np.random.randint(1000, size=(n, m))

# Multiply 2 matrices using for loop
def multiplication_forloop():
  product = np.zeros((n, m), dtype='int')
  for i in range(n):
    for j in range(m):
      product[i, j] = matrix1[i, j] * matrix2[i, j]

# Multiply 2 matrices using numpy vectorization
def multiplication_vectorize():
  product = matrix1 * matrix2

# Finding execution time using timeit
time_forloop = Timer(multiplication_forloop).timeit(1)
time_vectorize = Timer(multiplication_vectorize).timeit(1)

print("Element Wise Multiplication takes %0.9f units using for loop"%time_forloop)
print("Element Wise Multiplication takes %0.9f units using vectorization"%time_vectorize)

```

```py
Element Wise Multiplication takes 0.543777400 units using for loop
Element Wise Multiplication takes 0.001439500 units using vectorization

```

## 结论

由于执行速度更快、代码量更少，矢量化被广泛用于复杂系统和数学模型中。现在您已经知道如何在 python 中使用矢量化，您可以应用它来使您的项目执行得更快。所以恭喜你！

感谢阅读！