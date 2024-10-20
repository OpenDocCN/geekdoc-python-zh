# Python imread():使用 OpenCV.imread()方法加载图像的不同方式

> 原文：<https://www.askpython.com/python-modules/python-imread-opencv>

* * *

在本教程中，我们将详细学习如何使用 OpenCV-Python 的`imread()`方法，以及使用`imread()`方法加载图像的不同方式。

## **什么是 Python imread()？**

`imread()`是 **OpenCV-Python** 库中最有用和最常用的方法之一。它用于从指定文件加载 Python 程序中的图像。它在成功加载图像后返回一个`numpy.ndarray` (NumPy N 维数组)。当加载的图像是彩色图像时，该`numpy.ndarray`是一个**三维**数组，当加载的图像是灰度图像时，该**二维**数组。

## **导入 OpenCV 以使用 Python imread()**

为了使用 Python `imread()`方法，我们需要 [opencv-python 库](https://www.askpython.com/python/examples/image-processing-in-python)的`cv2`模块。为此，我们必须首先在[虚拟环境](https://www.askpython.com/python/examples/virtual-environments-in-python)或本地系统中安装`opencv-python`库，然后在 Python 程序中导入`cv2`模块。以下是安装和导入它的命令:

```py
# Installing the opencv-python library
pip install opencv-python

```

```py
# Importing the cv2 module
import cv2

```

## **Python imread()方法的语法**

以下是 Python `imread()`方法的正确语法:

```py
cv2.imread(filename, flag)

```

**参数:** `cv2.imread()`方法需要两个参数。这两个参数如下:

1.  **`filename`** 是第一个要传递的强制参数，它采用一个字符串值来表示*图像文件*(或带扩展名的图像名称)的路径。**注意:**如果不在工作目录中，我们必须传递*图像文件*的完整路径。
2.  `flag`是第二个要传递的可选参数，通常有三种类型的值:`cv2.IMREAD_COLOR`、`cv2.IMREAD_GRAYSCALE`和`cv2.IMREAD_UNCHANGED`。实际上，这个`flag`定义了读取图像的模式。**注:**默认情况下，该`flag`参数的值为`cv2.IMREAD_COLOR`或`1`。

**返回值:** `cv2.imread()`如果图片加载成功，方法返回一个`numpy.ndarray` (NumPy N 维数组)。**注意:**如果图像由于任何原因(如文件丢失、权限不当、不支持或格式无效)无法读取，则返回一个空矩阵(Mat::data==NULL)。

## **Python imread()方法支持的图像格式**

以下是`cv2.imread()`方法支持的图像格式:

*   **便携式网络显卡**–`*.png`
*   **便携图像格式**–`*.pbm`、`*.pgm`、`*.ppm`、`*.pxm`、`*.pnm`
*   **Windows 位图**–`*.bmp`
*   **JPEG 文件**–`*.jpeg`、`*.jpg`、`*.jpe`
*   **JPEG 2000 文件**–`*.jp2`
*   **WebP**–`*.webp`
*   **PFM 档案**–`*.pfm`
*   **太阳光栅**–`*.sr`、`*.ras`
*   **OpenEXR 镜像文件**–`*.exr`
*   **HDR 辐射**——`*.hdr`，`*.pic`
*   **TIFF 文件**–`*.tiff`，`*.tif`

**注意:**对`.JPEG`格式图像的读取取决于系统、平台或环境(如 x86/ARM)上安装的 [OpenCV 库](https://www.askpython.com/python/examples/edge-detection-in-images)的版本等。最重要的是图像的类型不是由*图像文件的*扩展名决定的，而是由`cv2.imread()`方法返回的`numpy.ndarray`的内容决定的。

*让我们用 Python 代码实现一切**……*

![Sample Image](img/c4d12cc22b5b7dd4f95d6f5e7d5d9961.png)

Sample Image

## **使用“flag = cv2”加载图像。IMREAD_COLOR"**

当`flag`以值`cv2.IMREAD_COLOR`传递时，图像首先被转换为没有透明通道的三通道 **BGR** 彩色图像，然后被加载到程序中。

这是`flag`参数的默认值。`cv2.IMREAD_COLOR`对应的整数值是`1`。我们也可以用`1`来代替`cv2.IMREAD_COLOR`。**注意:**我们正在使用`.shape`方法来访问图像的形状。它返回一个*行数*、*列数*和*通道数*的**元组**。

```py
img = cv2.imread('sample_image.png', cv2.IMREAD_COLOR) 
print("Shape of the loaded image is", img.shape)

```

**输出:**

```py
Shape of the loaded image is (512, 512, 3)

```

输出元组有三个值`512`是样本图像中的行数(图像的高度)`512`是列数(图像的宽度)`3`是通道数。

这里加载的图像只有三个通道**蓝绿色&红色**，因为标志值是`cv2.IMREAD_COLOR`。

第四个通道是透明或 alpha 通道，即使它出现在样本图像中，也会被忽略。

## **使用“flag = cv2”加载图像。im read _ gray**

当用值`cv2.IMREAD_GRAYSCALE`传递`flag` 时，图像首先被转换成单通道灰度图像，然后加载到程序中。`cv2.IMREAD_GRAYSCALE`对应的整数值是`0`我们也可以用`0`代替`cv2.IMREAD_GRAYSCALE`。

```py
img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
print("Shape of the loaded image is", img.shape)

```

**输出:**

```py
Shape of the loaded image is (512, 512)

```

输出元组只有两个值`512`是样本图像中的行数，`512`是列数。当`flag`值为`0`或`cv2.IMREAD_GRAYSCALE`时，无论传递给`cv2.imread()`方法的输入样本图像如何，图像都将被加载为灰度图像。

## **使用“flag = cv2”加载图像。IMREAD_UNCHANGED**

当用值`cv2.IMREAD_UNCHANGED`传递`flag` 时，图像就和 alpha 或透明通道(如果有的话)一起加载到程序中。`cv2.IMREAD_UNCHANGED`对应的整数值是`-1`我们也可以用`-1`代替`cv2.IMREAD_UNCHANGED`。

```py
img = cv2.imread('sample_image.png', cv2.IMREAD_UNCHANGED)
print("Shape of the loaded image is",img.shape)

```

**输出:**

```py
Shape of the loaded image is (512, 512, 4)

```

输出元组有三个值`512`是样本图像中的行数(图像的高度)`512`是列数(图像的宽度)`4`是通道数。

这里加载的图像有四个通道**蓝色、绿色、红色&透明度**，因为标志值是`cv2.IMREAD_UNCHANGED`。第四个通道是透明或 alpha 通道，如果它出现在样本图像中，它将被包括在内。

## **结论**

在本教程中，您已经学习了通过使用不同的`flag`参数值来加载图像的不同方法。请记住两件事:如果您当前的工作目录中没有示例图像文件，您必须传递它的完整路径；您还可以将整数值`[1, 0, & -1]`传递给对应于`[cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, & cv2.IMREAD_UNCHANGED]`的`flag`参数。

希望您对使用您自己的示例图像更多地尝试 Python `imread()`方法和`opencv-python`库的其他方法感到兴奋！