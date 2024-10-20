# 计算一个正方形数组的特征值和右特征向量

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-linalg-EIG

numpy linalg.eig 函数是一个强大的数学工具，使用户能够计算一个正方形数组的特征值和右特征向量。科学家和数学家在从物理和工程到经济和金融的各个领域都使用这个函数。

通过利用这一功能，用户可以深入了解系统的底层结构，并发现变量之间的关系。在本文中，我们将探讨什么是特征值和特征向量，以及 numpy linalg.eig 函数如何计算它们。

***也读作: [Numpy fabs-计算绝对值元素明智。](https://www.askpython.com/python-modules/numpy/numpy-fabs)***

## 什么是特征值和特征向量？

特征值是与矩阵的线性方程相关的唯一标量值。特征值告诉我们一个特定的数据值在一个特定的方向上有多少方差。方向主要由特征向量给出。特征向量是非零向量，经过线性变换后，它最多可以被一个标量改变。

### 特征值和特征向量有什么用？

需要特征值和特征向量来压缩数据和移除或减少维度空间。

### 计算特征值和特征向量的必要条件

要计算特征值和特征向量的矩阵必须是方阵，即矩阵的维数必须是“nXn”的形式，其中 n =行数=列数。

## Numpy linalg 库和 linalg.eig()函数

numpy 线性代数库 numpy linalg 包含一系列函数，使复杂的线性代数计算变得更加容易。

### linalg.eig()函数的语法

下面给出了函数的语法，其中 x 是初始参数或输入:

```py
linalg.eig(x)

```

**numpy linalg . EIG()函数的参数**

下面给出了该函数所需的参数:

**输入**–`x : array`->要计算特征值和右特征向量的初始方阵。

**输出**--

`y : array` - >特征值按其重数无序重复。数组的类型是复数，除非复数部分为零，在这种情况下，返回类型是实数。

`z : array` - >列 z[:，i]对应特征值 y[i]的单位特征向量。

## 如何使用 numpy linalg.eig()函数

第一步是安装 numpy 包，如果你还没有安装的话。运行以下代码在您的系统中安装 numpy。

```py
pip install numpy

```

接下来，将库导入到您的项目中，然后按照给出的示例进行操作。

### 示例 1–计算预定义矩阵的特征值和特征向量

这个代码计算预定义矩阵的特征值和特征向量。它首先从 numpy 导入必要的模块 numpy(作为 py)和 linalg。然后定义一个矩阵为 x，接下来使用 linalg.eig()函数计算矩阵的特征值和特征向量，并存储在变量 y 和 z 中，最后显示计算出的特征值和特征向量。

```py
import numpy as py #importing required modules
from numpy import linalg as L #importing the linalg function
x=[[1,6],[4,9]] #pre defined matrix
y, z = L.eig(x) #computing the eigenvalues and eigenvectors
#displaying the values
print("the eigenvalues are:",y) 
print("the eigenvectors are:",z)

```

**输出:**

```py
the eigenvalues are: [-1.32455532 11.32455532]
the eigenvectors are: [[-0.93246475 -0.50245469]
 [ 0.36126098 -0.86460354]]

```

### 示例 2–计算矩阵的特征值和特征向量

获取用户输入的矩阵并计算特征值和特征向量。这段代码被设计成接受用户输入的矩阵，计算它的特征值和特征向量，然后显示结果。它使用 NumPy 和 linalg 模块来执行计算。首先要求用户输入矩阵的行数和列数。如果行数不等于列数，将显示一条错误消息。然后提示用户逐行输入矩阵的值。一旦输入了矩阵，就会显示出来，然后使用 linalg.eig()函数计算特征值和特征向量。最后，显示结果。

```py
#importing the required modules
import numpy as py
from numpy import linalg as L
#taking user input
row = int(input("Enter the number of rows:"))
col= int(input("Enter the number of columns:"))
print("NOTE!! The number of rows should be equal to the number of columns")

# Initializing the required matrix
x = []
print("enter the values rowwise:")

# For user input
for i in range(row):          # loop for row entries
    b =[]
    for j in range(col):      # loop for column entries
         b.append(int(input()))
    x.append(b)

# For displaying the matrix
print("The matrix is as follows:")
print("[")
for i in range(row):
    for j in range(col):
        print(x[i][j], end = " ")
    print()
print("]")
y, z = L.eig(x) #computing the values and vectors

#displaying the result
print("the eigenvalues are:",y)
print("the eigenvectors are:",z)

```

**输出:**

```py
Enter the number of rows:2
Enter the number of columns:2
NOTE!! The number of rows should be equal to the number of columns
enter the values rowwise:
1
-1
-1
1
The matrix is as follows:
[
1 -1
-1 1
]
the eigenvalues are: [2\. 0.]
the eigenvectors are: [[ 0.70710678  0.70710678]
 [-0.70710678  0.70710678]]

```

## 特征值和特征向量的缺点

1.  特征值和特征向量只适用于线性变换，所以不能用来解决非线性问题。
2.  它们只能用来研究线性变换，不能用来研究非线性变换。
3.  计算给定矩阵的特征值和特征向量可能是困难的，并且计算可能是耗时的。
4.  特征向量通常不代表关于数据集的最直观或最有意义的信息。

如果矩阵的特征向量是线性相关的，则表明存在一个代数重数大于几何重数的特征值。在这种情况下，结果会有缺陷。要了解更多关于特征值的多重性，点击这里。

## 结论:

eig 函数是计算方阵的特征值和右特征向量的强大工具。通过使用该函数，科学家和数学家可以深入了解系统的底层结构，并发现变量之间的关系。

这可以帮助他们压缩数据，减少维度空间。然而，特征值和特征向量仅适用于线性变换，并且可能难以计算，因此用户在使用该函数时必须确保理解其局限性。