# 如何使用 Python 对图像执行算术运算

> 原文：<https://www.askpython.com/python/examples/arithmetic-operations-on-images>

在本教程中，我们将学习如何使用 Python 对图像执行不同的算术运算。我们将执行不同的运算，如加、减、乘、除。

* * *

## 我们对图像的算术运算是什么意思？

**图像运算**是指对图像的算术运算。每当我们对图像执行任何算术运算时，它都是对单个像素值执行的。**例如:**如果图像是彩色的，那么加法是这样执行的:

```py
f_img(i, j, k) = img1(i, j, k) + img2(i, j, k) 
or
f_img(i, j, k) = img1(i, j, k) + constant

```

如果图像是一个 **[灰度图像](https://www.askpython.com/python/examples/image-processing-in-python)** ，那么加法是这样执行的:

```py
f_img(i, j) = img1(i, j) + img2(i, j)
or
f_img(i, j) = img1(i, j) + constant

```

类似地，其他算术运算也在图像上执行。要首先对图像执行任何算术运算，我们必须使用 cv2.imread()方法加载图像。

正如我们所知，图像被加载为 NumPy N 维数组，因此对它们执行不同的算术运算变得非常容易。**注意:**如果对两幅或多幅图像进行算术运算，那么所有图像都应该是相同的*类型*，如 jpeg、jpg、png 等。、 *^(**) 深度*、*尺寸*。

^(******) **深度:**用于表示每个像素的比特数，如每通道 8 比特，通常被称为 24 比特彩色图像(8 比特×3 通道)。

## 使用 OpenCV 对图像进行算术运算

首先，我们必须安装 **OpenCV-Python** 库，然后在 Python 程序中导入 **cv2** 模块。以下是安装 OpenCV-Python 和导入 cv2 模块的命令:

```py
# Installing OpenCV-Python library
pip install opencv-python

```

```py
# Importing cv2 module
import cv2

```

## 1.图像添加

我们既可以添加两个图像，也可以为一个图像添加一个常量值。图像添加通常用作一些复杂过程中的中间步骤，而不是作为其本身的有用操作。

在进行适当的掩蔽后，它可用于将一幅图像叠加到另一幅图像上。我们可以通过两种方式执行图像添加:

*   **NumPy 加法:**在这里，我们简单地加载图像文件，并使用(+)操作符添加加载图像后返回的 NumPy N-d 数组。这是一个**模运算**，这意味着如果输入(加载)图像的像素值相加后所得像素值大于 255，则计算所得像素值与 256(对于 8 位图像格式)的模(%)，并将其分配给所得像素值，以保持其低于 255 或 255，因为任何像素值都不能超过 255。**例如:** **`250+10 = 260 => 260 % 256 = 4`**

![arithmetic operations on images](img/c9e458db08e629308ac9e86e4f10dee7.png)

Sample Image 1

![arithmetic operations on images](img/f5bf77c3a4dc5de86b48859fbe7ae2bd.png)

Sample Image 2

```py
# Reading image files
img1 = cv2.imread('sample-img-1.jpg')
img2 = cv2.imread('sample-img-2.jpg')

# Applying NumPy addition on images
fimg = img1 + img2

# Saving the output image
cv2.imwrite('output.jpg', fimg)

```

****O** 输出:**

![arithmetic operations on images](img/c46da6b501301f2475e3632458c0cf55.png)

Output Image

**OpenCV 添加:**在这里，我们简单地加载图像文件，并将加载图像后返回的 NumPy N-d 数组作为参数传递给`cv2.add()`方法。这是一个**饱和** **操作**，这意味着如果在输入(加载)图像的像素值相加之后得到的像素值大于 255，则它饱和到 255，使得任何像素值都不能超过 255。这叫做^(**) *饱和度*。**例如:** `250+10 = 260 => 255`

^(******) **饱和度**是一种用于处理像素溢出的图像处理技术，其中我们将所有溢出的像素设置为最大可能值。

```py
# Reading image files
img1 = cv2.imread('sample-img-1.jpg')
img2 = cv2.imread('sample-img-2.jpg')

# Applying OpenCV addition on images
fimg = cv2.add(img1, img2)

# Saving the output image
cv2.imwrite('output.jpg', fimg)

```

**输出:**

![Output Image 2](img/2357709c7ae804b9f5330d06089d6694.png)

Output Image

**注意:**坚持使用 OpenCV 函数对图像执行不同的操作总是明智的，因为从上面两个例子的输出可以看出，它们提供了更好的结果。

## 2.图像减影

图像减法就是像素减法，它将两幅图像作为输入，产生第三幅图像作为输出，第三幅图像的像素值就是第一幅图像的像素值减去第二幅图像的相应像素值。我们也可以使用单个图像作为输入，并从其所有像素值中减去一个常数值。某些版本的运算符将输出像素值之间的绝对差值，而不是直接的有符号输出。

如果输出像素值为负，图像相减的实现方式会有所不同。如果图像格式支持负值像素，在这种情况下，负值是好的。如果图像格式不支持负像素值，那么这种像素通常被设置为零(即通常为黑色)。或者

如果图像减法计算使用相同像素值类型的两个输入图像的绝对差，则输出像素值不会超出由输入图像像素类型表示的指定范围，因此不会出现这个问题。这就是为什么使用绝对差异是好的。同样，我们可以通过两种方式执行图像相减:

**NumPy 减法和 OpenCV 减法。**

我们将只使用 OpenCV 减法，因为它产生更好的结果，并且被广泛使用。使用`cv2.subtract()`方法进行图像相减，结果将类似于`res = img1 - img2`，其中 *img1* & *img2* 是相同深度和类型的图像。

图像相减既可以作为复杂图像处理技术的中间步骤，也可以作为一种独立的重要操作。图像减法的一个最常见的用途是从场景中减去背景照明的变化，以便可以更容易和更清楚地分析前景中的物体。

**注意:**我们也将使用相同的样本图像进行图像减影。

```py
# Reading image files
img1 = cv2.imread('sample-img-1.jpg')
img2 = cv2.imread('sample-img-2.jpg')

# Applying OpenCV subtraction on images
fimg = cv2.subtract(img1, img2)

# Saving the output image
cv2.imwrite('output.jpg', fimg)

```

**输出:**

![Output Image 3](img/ec5b67cbd15da41d51f3a2f1d39b88e6.png)

Output Image

## 3.图像倍增

像对图像的其他算术运算一样，图像乘法也可以在表单中实现。第一种形式的图像乘法采用两个输入图像并产生输出图像，其中像素值是输入图像的相应像素值的乘积。

第二种形式采用单个输入图像并产生输出，其中每个像素值是输入图像的相应像素值和指定常数(比例因子)的乘积。这第二种形式的图像乘法应用更广泛，通常称为**缩放**。

[图像缩放](https://www.askpython.com/python-modules/pygame-creating-interactive-shapes)有多种用途，但一般来说，缩放系数大于 1 时，图像会变亮，缩放系数小于 1 时，图像会变暗。

与简单地将偏移添加到像素值相比，缩放通常会在图像中产生更自然的亮或暗效果，因为它可以更好地保留图像的相对对比度。

**注意:**常数值往往是一个浮点数，根据它可以增加或减少图像强度。如果图像格式支持，它可以是负数。如果计算出的输出值大于最大允许像素值，则在该最大允许像素值处将其截断。

让我们使用 NumPy 图像乘法来增加下面给出的样本图像的亮度。

![Sample Image 3](img/ec27b4b4c7e9f76d6c176b3b4ee2c702.png)

Sample Image

```py
# Reading image file
img = cv2.imread('sample_img.jpg')

# Applying NumPy scalar multiplication on image
fimg = img * 1.5

# Saving the output image
cv2.imwrite('output.jpg', fimg)

```

**输出:**

![Output Image 4](img/ac2e82673321195bca8640f6db5e5014.png)

Output Image

现在让我们看看这个示例图像在使用`cv2.multiply()`方法应用 OpenCV 图像乘法时的变化，该方法通常采用两个图像数组或一个图像数组和一个指定的常数。

```py
# Reading image file
img = cv2.imread('sample_img.jpg')

# Applying OpenCV scalar multiplication on image
fimg = cv2.multiply(img, 1.5)

# Saving the output image
cv2.imwrite('output.jpg', fimg)

```

**输出:**

![Output Image 5](img/10fbc4dcd8b8e70f90c5e515f9088398.png)

Output Image

## 4.**图像分割**

图像分割操作通常将两幅图像作为输入，并产生第三幅图像，其像素值是第一幅图像的像素值除以第二幅图像的相应像素值。

它也可以用于单个输入图像，在这种情况下，图像的每个像素值都除以指定的常数。

图像除法运算可以像减法一样用于变化检测，但是除法运算给出对应像素值之间的分数变化或比率，而不是给出每个像素值从一个图像到另一个图像的绝对变化。

这就是为什么它通常被称为配给制。

让我们使用图像分割来降低上面样本图像的亮度，使用`cv2.divide()`方法，通常采用两个图像数组或一个图像数组和一个指定的常数。

```py
# Reading image file
img = cv2.imread('sample_img.jpg')

# Applying OpenCV scalar division on image
fimg = cv2.divide(img, 2)

# Saving the output image
cv2.imwrite('output.jpg', fimg)

```

**输出:**

![Output Image 5](img/57962d442877fdc4b5c68b7bb67119c6.png)

Output Image

或者，我们也可以使用[数字除法](https://www.askpython.com/python-modules/numpy/python-numpy-module)来降低上述样本图像的亮度，如下所示:

```py
# Reading image file
img = cv2.imread('sample_img.jpg')

# Applying NumPy scalar division on image
fimg = img / 2

# Saving the output image
cv2.imwrite('output.jpg', fimg)

```

**输出:**

![Output Image 6](img/754e89b36af6c242fa763e697e610bc2.png)

Output Image

## **结论**

在本教程中，我们学习了如何对图像执行不同的算术运算，分析了用于执行图像算术运算的不同 OpenCV 方法的工作原理，并学习了这些图像算术运算的使用位置，如饱和度、**、缩放、**等。