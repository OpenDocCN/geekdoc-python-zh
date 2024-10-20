# Python 矩阵教程

> 原文：<https://www.askpython.com/python/python-matrix-tutorial>

我们可以用一个**二维列表**或者一个**二维数组**的形式实现一个 Python 矩阵。**要对 Python 矩阵执行操作，我们需要导入 Python NumPy 模块。**

Python 矩阵在统计学、数据处理、图像处理等领域是必不可少的。

* * *

## 创建 Python 矩阵

Python 矩阵可使用以下技术之一创建:

*   **通过使用列表**
*   **通过使用 arange()方法**
*   **利用矩阵()方法**

### 1.使用列表创建矩阵

使用**列表作为**的输入，函数`numpy.array()`可以用来创建一个数组。

**举例:**

```py
import numpy
input_arr = numpy.array([[ 10, 20, 30],[ 40, 50, 60]])
print(input_arr)

```

**输出:**

```py
[[10 20 30]
 [40 50 60]]

```

如上所述，输出以列表的形式表示一个带有给定输入集的二维矩阵。

### 2.使用“numpy.arange()”函数创建矩阵

在 Python 中，`numpy.arange()`函数和列表输入可以用来创建一个矩阵。

**举例:**

```py
import numpy

print(numpy.array([numpy.arange(10,15), numpy.arange(15,20)]))

```

**输出:**

```py
[[10 11 12 13 14]
 [15 16 17 18 19]]

```

### 3.使用“numpy.matrix()函数”创建矩阵

`numpy.matrix()`函数使我们能够用 Python 创建一个矩阵。

**语法:**

```py
numpy.matrix(input,dtype)

```

*   **输入:将元素输入形成矩阵。**
*   **dtype:对应输出的数据类型。**

**举例:**

```py
import numpy as p

matA = p.matrix([[10, 20], [30, 40]])  
print('MatrixA:\n', matA)

matB = p.matrix('[10,20;30,40]', dtype=p.int32)  # Setting the data-type to int
print('\nMatrixB:\n', matB)

```

**输出:**

```py
MatrixA:
 [[10 20]
 [30 40]]

MatrixB:
 [[10 20]
 [30 40]]

```

* * *

## Python 中矩阵的添加

矩阵的加法运算可以通过以下方式执行:

*   **传统方法**
*   **通过使用'+'运算符**

### 1.传统方法

在这种传统方法中，我们基本上从用户那里获取输入，然后使用循环的**(遍历矩阵的元素)和 **'+'操作符**执行加法操作。**

****举例:****

```py
import numpy as p

ar1 = p.matrix([[11, 22], [33, 44]])  
ar2 = p.matrix([[55, 66], [77, 88]])  
res = p.matrix(p.zeros((2,2)))  
print('Matrix ar1 :\n', ar1)
print('\nMatrix ar2 :\n', ar2)

# traditional code
for x in range(ar1.shape[1]):
    for y in range(ar2.shape[0]):
        res[x, y] = ar1[x, y] + ar2[x, y]

print('\nResult :\n', res) 
```

****注意** : `Matrix.shape`返回特定矩阵的维数。**

****输出:****

```py
Matrix ar1 :
 [[11 22]
 [33 44]]

Matrix ar2 :
 [[55 66]
 [77 88]]

Result :
 [[  66\.   88.]
 [ 110\.  132.]] 
```

### **2.使用“+”运算符**

**这种方法为代码提供了更好的效率，因为它减少了 LOC(代码行),从而优化了代码。**

****举例:****

```py
import numpy as p

ar1 = p.matrix([[11, 22], [33, 44]])  
ar2 = p.matrix([[55, 66], [77, 88]])  
res = p.matrix(p.zeros((2,2)))  
print('Matrix ar1 :\n', ar1)
print('\nMatrix ar2 :\n', ar2)

res = ar1 + ar2 # using '+' operator

print('\nResult :\n', res) 
```

****输出:****

```py
Matrix ar1 :
 [[11 22]
 [33 44]]

Matrix ar2 :
 [[55 66]
 [77 88]]

Result :
 [[ 66  88]
 [110 132]] 
```

* * *

## **Python 中的矩阵乘法**

**Python 中的矩阵乘法可通过以下方式实现:**

*   ****标量积****
*   ****矩阵乘积****

### **数积**

**在标量积中，**标量/常数值**乘以矩阵的每个元素。**

****'* '运算符**用于将标量值乘以输入矩阵元素。**

****举例:****

```py
import numpy as p

matA = p.matrix([[11, 22], [33, 44]])  

print("Matrix A:\n", matA)
print("Scalar Product of Matrix A:\n", matA * 10) 
```

****输出:****

```py
Matrix A:
 [[11 22]
 [33 44]]
Scalar Product of Matrix A:
 [[110 220]
 [330 440]] 
```

### **矩阵积**

**如上所述，我们可以使用**“*”操作符仅用于标量乘法**。为了进行矩阵乘法，我们需要使用`numpy.dot()`函数。**

**`numpy.dot()`函数以 **NumPy 数组作为参数**值，根据矩阵乘法的基本规则进行乘法运算。**

****举例:****

```py
import numpy as p

matA = p.matrix([[11, 22], [33, 44]])  
matB = p.matrix([[2,2], [2,2]])

print("Matrix A:\n", matA)
print("Matrix B:\n", matB)
print("Dot Product of Matrix A and Matrix B:\n", p.dot(matA, matB)) 
```

****输出:****

```py
Matrix A:
 [[11 22]
 [33 44]]
Matrix B:
 [[2 2]
 [2 2]]
Dot Product of Matrix A and Matrix B:
 [[ 66  66]
 [154 154]] 
```

* * *

## **Python 矩阵的减法**

****'-'运算符**用于对 Python 矩阵进行减法运算。**

****举例:****

```py
import numpy as p

matA = p.matrix([[11, 22], [33, 44]])  
matB = p.matrix([[2,2], [2,2]])

print("Matrix A:\n", matA)
print("Matrix B:\n", matB)
print("Subtraction of Matrix A and Matrix B:\n",(matA - matB)) 
```

****输出:****

```py
Matrix A:
 [[11 22]
 [33 44]]
Matrix B:
 [[2 2]
 [2 2]]
Subtraction of Matrix A and Matrix B:
 [[ 9 20]
 [31 42]] 
```

* * *

## **Python 矩阵的除法**

****在 Python 中可以使用 **'/'运算符**对矩阵的元素进行标量除法**。**

**“/”运算符用标量/常数值除矩阵的每个元素。**

****举例**:**

```py
import numpy as p

matB = p.matrix([[2,2], [2,2]])

print("Matrix B:\n", matB)
print("Matrix B after Scalar Division operation:\n",(matB/2)) 
```

****输出:****

```py
Matrix B:
 [[2 2]
 [2 2]]
Matrix B after Scalar Division operation:
 [[ 1\.  1.]
 [ 1\.  1.]] 
```

* * *

## **Python 矩阵的转置**

**矩阵的转置基本上包括在相应对角线上矩阵的**翻转，即交换输入矩阵的行和列。行变成列，反之亦然。****

**例如:让我们考虑一个维数为 3×2 的矩阵 A，即 3 行 2 列。在执行转置操作之后，矩阵 A 的维数将是 2×3，即 2 行 3 列。**

**`Matrix.T`基本上执行输入矩阵的转置，并产生一个**新矩阵**作为转置操作的结果。**

****举例:****

```py
import numpy

matA = numpy.array([numpy.arange(10,15), numpy.arange(15,20)])
print("Original Matrix A:\n")
print(matA)
print('\nDimensions of the original MatrixA: ',matA.shape)
print("\nTranspose of Matrix A:\n ")
res = matA.T
print(res)
print('\nDimensions of the Matrix A after performing the Transpose Operation:  ',res.shape) 
```

****输出:****

```py
Original Matrix A:

[[10 11 12 13 14]
 [15 16 17 18 19]]

Dimensions of the original MatrixA: (2, 5)

Transpose of Matrix A:

[[10 15]
 [11 16]
 [12 17]
 [13 18]
 [14 19]]

Dimensions of the Matrix A after performing the Transpose Operation: (5, 2) 
```

**在上面的代码片段中，我创建了一个 2×5 的矩阵，即 2 行 5 列。**

**在执行转置操作之后，结果矩阵的维数是 5×2，即 5 行 2 列。**

* * *

## **Python 矩阵的指数**

**矩阵的指数是按元素计算的**，即每个元素的指数是通过将元素提升到输入标量/常数值的幂来计算的。****

******举例:******

```py
**import numpy

matA = numpy.array([numpy.arange(0,2), numpy.arange(2,4)])
print("Original Matrix A:\n")
print(matA)
print("Exponent of the input matrix:\n")
print(matA ** 2) # finding the exponent of every element of the matrix** 
```

******输出:******

```py
**Original Matrix A:

[[0 1]
 [2 3]]

Exponent of the input matrix:

[[0 1]
 [4 9]]** 
```

****在上面的代码片段中，我们通过将输入矩阵的每个元素的指数提升到 2 的幂，找到了它的指数。****

* * *

## ****使用 NumPy 方法的矩阵乘法运算****

****以下技术可用于执行 NumPy 矩阵乘法:****

*   ******使用乘法()方法******
*   ******使用 matmul()方法******
*   ******使用 dot()方法**——本文已经介绍过了****

### ****方法 1:使用 multiply()方法****

****`numpy.multiply()`方法对输入矩阵执行逐元素乘法。****

******举例:******

```py
**import numpy as p

matA = p.matrix([[10, 20], [30, 40]])  
print('MatrixA:\n', matA)

matB = p.matrix('[10,20;30,40]', dtype=p.int32)  # Setting the data-type to int
print('\nMatrixB:\n', matB)

print("Matrix multplication using numpy.matrix() method")
res = p.multiply(matA,matB)
print(res)** 
```

******输出:******

```py
**MatrixA:
 [[10 20]
 [30 40]]

MatrixB:
 [[10 20]
 [30 40]]
Matrix multplication using numpy.matrix() method
[[ 100  400]
 [ 900 1600]]** 
```

### ****方法 2:使用 matmul()方法****

****`numpy.matmul()`方法对输入矩阵执行矩阵乘积。****

******举例:******

```py
**import numpy as p

matA = p.matrix([[10, 20], [30, 40]])  
print('MatrixA:\n', matA)

matB = p.matrix('[10,20;30,40]', dtype=p.int32)  # Setting the data-type to int
print('\nMatrixB:\n', matB)

print("Matrix multplication using numpy.matmul() method")
res = p.matmul(matA,matB)
print(res)** 
```

******输出:******

```py
**MatrixA:
 [[10 20]
 [30 40]]

MatrixB:
 [[10 20]
 [30 40]]
Matrix multplication using numpy.matmul() method
[[ 700 1000]
 [1500 2200]]** 
```

****我强烈推荐所有的读者阅读下面的教程，以彻底理解 NumPy 矩阵乘法:NumPy 矩阵乘法****

* * *

## ****NumPy 矩阵转置****

****`numpy.transpose()`函数对输入矩阵进行转置，产生一个新矩阵。****

******举例:******

```py
**import numpy

matA = numpy.array([numpy.arange(10,15), numpy.arange(15,20)])
print("Original Matrix A:\n")
print(matA)
print('\nDimensions of the original MatrixA: ',matA.shape)
print("\nTranspose of Matrix A:\n ")
res = matA.transpose()
print(res)
print('\nDimensions of the Matrix A after performing the Transpose Operation:  ',res.shape)** 
```

******输出:******

```py
**Original Matrix A:

[[10 11 12 13 14]
 [15 16 17 18 19]]

Dimensions of the original MatrixA: (2, 5)

Transpose of Matrix A:

[[10 15]
 [11 16]
 [12 17]
 [13 18]
 [14 19]]

Dimensions of the Matrix A after performing the Transpose Operation: (5, 2)** 
```

****推荐阅读:NumPy 矩阵转置()函数****

* * *

## ****结论****

****因此，在本文中，我们已经理解了在 Python Matrix 上执行的操作，并且还了解了 NumPy Matrix 操作。****

* * *

## ****参考****

*   ****Python 矩阵****
*   ****[NumPy 文档](https://numpy.org/doc/stable/reference/)****
*   ****Python NumPy****