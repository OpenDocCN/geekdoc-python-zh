# NumPy degrees()、deg2rad()、rad2deg()和 radians()函数

> 原文：<https://www.askpython.com/python-modules/numpy/degrees-deg2rad-rad2deg-radians>

读者你好！欢迎来到另一个关于 NumPy 函数的教程。在本教程中，我们将详细了解 numpy.degrees()、numpy.deg2rad()、numpy.rad2deg()和 numpy.radians()三个重要函数。

那么，我们开始吧。

## 1.numpy.degrees()

`numpy.degrees()`是 NumPy 库的一个函数，将角度从弧度转换成度数。

**语法:** `numpy.degrees(input)`其中输入可以是以弧度表示的单个角度或以弧度表示的角度的 NumPy 数组。

### 使用 numpy.degrees()

为了更好地理解，让我们尝试一些例子。

#### 使用 numpy.degrees()和单个角度作为输入

```py
import numpy as np

# Converting pi/6 radians to equivalent angle in degrees
print("PI/6 radians is ",np.degrees(np.pi/6),"degrees \n")

# Converting pi/4 radians to equivalent angle in degrees
print("PI/4 radians is ",np.degrees(np.pi/4),"degrees \n")

# Converting pi/3 radians to equivalent angle in degrees
print("PI/3 radians is ",np.degrees(np.pi/3),"degrees \n")

# Converting pi/2 radians to equivalent angle in degrees
print("PI/2 radians is ",np.degrees(np.pi/2),"degrees \n")

# Converting pi radians to equivalent angle in degrees
print("PI radians is ",np.degrees(np.pi),"degrees ")

```

**输出**

```py
PI/6 radians is  29.999999999999996 degrees 

PI/4 radians is  45.0 degrees

PI/3 radians is  59.99999999999999 degrees

PI/2 radians is  90.0 degrees

PI radians is  180.0 degrees

```

上面的代码片段非常清楚地显示了一个角度(以弧度为单位)作为参数传递给了`np.degrees()`函数。

#### 对 numpy 数组使用 numpy.degrees()

```py
import numpy as np

a = np.array((np.pi/3 , np.pi/2 , np.pi/4 , np.pi))
b = np.degrees(a)

print("Angles in radians: \n",a)

print("Corresponding angles in degrees: \n",b)

```

**输出**

```py
Angles in radians: 
 [1.04719755 1.57079633 0.78539816 3.14159265]
Corresponding angles in degrees:
 [ 60\.  90\.  45\. 180.]

```

在上面的代码片段中，创建了一个 NumPy 数组，并将其分配给变量`**a**`。在这个数组中，所有的角度都是以弧度表示的，这个数组传递给 np.degrees()函数。

## 2\. numpy.deg2rad()

`numpy.deg2rad()`是一个数学函数，它将以度为单位的角度转换为以弧度为单位的角度。

**语法:** `numpy.deg2rad(input)`其中输入可以是以度为单位的单个角度，也可以是以度为单位的角度的 NumPy 数组。

### 使用 numpy.deg2rad()函数

现在让我们尝试几个 2 度弧度的例子。

#### 使用 numpy.deg2rad()和单个角度作为输入

```py
import numpy as np

print("30 degrees is equal to ",np.deg2rad(30),"radians\n")

print("45 degrees is equal to ",np.deg2rad(45),"radians\n")

print("60 degrees is equal to ",np.deg2rad(60),"radians\n")

print("90 degrees is equal to ",np.deg2rad(90),"radians\n")

print("360 degrees is equal to ",np.deg2rad(360),"radians")

```

**输出**

```py
30 degrees is equal to  0.5235987755982988 radians

45 degrees is equal to  0.7853981633974483 radians

60 degrees is equal to  1.0471975511965976 radians

90 degrees is equal to  1.5707963267948966 radians

360 degrees is equal to  6.283185307179586 radians

```

#### 对 numpy 数组使用 numpy.deg2rad()

```py
import numpy as np

a = np.array((30 , 45 , 60 , 90 , 180 , 270 , 360))

b = np.deg2rad(a)

print("Angles in Degrees :\n",a)

print("Angles in Radians :\n",b)

```

**输出**

```py
Angles in Degrees :
 [ 30  45  60  90 180 270 360]
Angles in Radians :
 [0.52359878 0.78539816 1.04719755 1.57079633 3.14159265 4.71238898
 6.28318531]

```

## 3\. numpy.rad2deg()

`numpy.rad2deg()`NumPy 库的函数相当于 numpy.degrees()函数。它将以弧度为单位的角度值转换为以度为单位的角度值。

**语法:** `numpy.rad2deg(input)`其中输入可以是以弧度表示的单个角度或以弧度表示的角度的 NumPy 数组。

让我们尝试一些例子来更好地理解它。

### 使用 numpy.rad2deg()

现在让我们来看看 Numpy 的弧度转度数函数。

#### 对 numpy 数组使用 numpy.rad2deg()

```py
import numpy as np

a = np.array((-np.pi , np.pi/2 , np.pi/3 , np.pi/4 , np.pi))

b = np.rad2deg(a)

print("Angles in Radians:\n",a)

print("Angles in Degrees:\n",b)

```

**输出**

```py
Angles in Radians:
 [-3.14159265  1.57079633  1.04719755  0.78539816  3.14159265]
Angles in Degrees:
 [-180\.   90\.   60\.   45\.  180.]

```

## 4.numpy.radians()

**`numpy.radians()`也是将角度从度转换为弧度的数学函数之一。**

语法:`numpy.radians(input)`其中输入可以是单个角度，也可以是角度的 NumPy 数组。

### 使用 numpy.radians()

最后来看看 numpy 的弧度函数。

#### 对 numpy 数组使用 numpy.radians()

```py
import numpy as np

a = np.array((60 , 90 , 45 , 180))
b = np.radians(a)

print("Angles in degrees: \n",a)

print("Corresponding angles in radians: \n",b)

```

#### 输出

```py
Angles in degrees: 
 [ 60  90  45 180]
Corresponding angles in radians: 
 [1.04719755 1.57079633 0.78539816 3.14159265]

```

这就是关于 NumPy degrees()、deg2rad()、rad2deg()和 radians()函数的全部内容。这些功能用起来真的很简单，也很容易理解。一定要把这篇文章读两遍，在阅读的同时练习这些代码。有一个**任务**给你们所有人，你们要用`**numpy.rad2deg(**` **`)`** 和 **`numpy.radians()`** 功能单输入。

## 参考

*   [NumPy 文档–NumPy 度](https://numpy.org/doc/stable/reference/generated/numpy.degrees.html)
*   [num py documentation–num py deg2 rad](https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html)
*   [num py documentation–num py rad 2 deg](https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html)
*   [NumPy 文档–NumPy 弧度](https://numpy.org/doc/stable/reference/generated/numpy.radians.html)