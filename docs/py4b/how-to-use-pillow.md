# 如何使用枕头，PIL 的一把叉子

> 原文：<https://www.pythonforbeginners.com/gui/how-to-use-pillow>

### 概观

在上一篇文章中，我写了关于 PIL，也称为 Python 图像库，这个库可以很容易地操作图像。自 2009 年以来，PIL 没有任何发展。因此，该网站的好心用户建议看一看枕头。这篇文章将告诉你如何使用枕头。

## 枕头是什么？

Pillow 是 PIL (Python Image Library)的一个分支，由 Alex Clark 和贡献者发起并维护。它以 PIL 法典为基础，然后演变成一个更好、更现代、更友好的 PIL 版本。它增加了对打开、操作和保存许多不同图像文件格式的支持。很多东西的工作方式和最初的 PIL 一样。

## 下载和安装枕头

在我们开始使用枕头之前，我们必须先下载并安装它。Pillow 适用于 Windows、Mac OS X 和 Linux。最新版本为“2.2.1”，受 python 2.6 及以上版本支持。要在 Windows 机器上安装 Pillow，您可以使用 easy_install:

```py
easy_install Pillow
```

要在 Linux 机器上安装枕头，只需使用:

```py
sudo pip install Pillow
```

要在 Mac OS X 上安装 Pillow，我必须先安装 XCode，然后通过自制软件安装先决条件。家酿安装后，我运行:

```py
$ brew install libtiff libjpeg webp littlecms
```

```py
$ sudo pip install Pillow
```

如果你知道在 Mac 上做这件事的更简单的方法，请告诉我。

## 验证枕头是否已安装

要验证 Pillow 是否已安装，请打开一个终端并键入以下内容:

```py
$ python
Python 2.7.5 (default, Aug 25 2013, 00:04:04)
[GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.0.68)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from PIL import Image
```

如果系统返回“> > >”，则说明枕形模块安装正确。

## 文件格式

在我们开始使用 Pillow 模块之前，让我们提一下支持的一些文件类型。BMP EPS GIF IM JPEG MSP PCX PNG PPM TIFF WebP ICO PSD PDF 有些文件类型，你只有读取的可能，其他的你只能写入。要查看支持的文件类型的完整列表，以及关于它们的更多信息，请查看 Pillow 的手册。

## 如何使用枕头来处理图像

既然我们要处理图像，让我们先下载一个。如果您已经有一张图片可以使用，请跳过这一步。在我们的示例中，我们将使用名为“Lenna”或“Lena”的标准测试图像。该标准测试图像用于许多图像处理实验。就去[这里](https://en.wikipedia.org/wiki/Lenna "lenna")下载图片。如果你点击图像，它会把它保存为 512×512 像素的图像。

## 使用枕头

让我们看看这个库的可能用途。基本功能可在图像模块中找到。您可以通过多种方式创建该类的实例:从文件中加载图像、处理其他图像或从头开始创建图像。导入您想要使用的枕头模块。

```py
from PIL import Image
```

然后，您可以像往常一样访问功能，例如

```py
myimage = Image.open(filename)
myimage.load()
```

## 加载图像

要从您的计算机中加载一个图像，您可以使用 use "open "方法来识别该文件，然后使用 myfile.load()加载所识别的文件。一旦图像被加载，你可以用它做很多事情。我经常在处理文件时使用 try/except 块。使用 try/except 加载我们的图像:

```py
from PIL import Image, ImageFilter
try:
    original = Image.open("Lenna.png")
except:
    print "Unable to load image"
```

当我们使用 open()函数从磁盘读取文件时，我们不需要知道文件的格式。该库根据文件内容自动确定格式。现在，当您拥有一个图像对象时，您可以使用可用的属性来检查文件。例如，如果您想查看图像的大小，您可以调用“格式”属性。

```py
print "The size of the Image is: "
print(original.format, original.size, original.mode)
```

“size”属性是一个包含宽度和高度(以像素为单位)的二元组。常见的“模式”是灰度图像的“L”，真彩色图像的“RGB”，以及印前图像的“CMYK”。上面的输出应该给你这个:

```py
The size of the Image is:
('PNG', (512, 512), 'RGB')
```

## 模糊图像

这个例子将从硬盘上加载一个图像并模糊它。[ [来源](https://en.wikipedia.org/wiki/Python_Imaging_Library "wiki_pil")

```py
# Import the modules
from PIL import Image, ImageFilter

try:
    # Load an image from the hard drive
    original = Image.open("Lenna.png")

    # Blur the image
    blurred = original.filter(ImageFilter.BLUR)

    # Display both images
    original.show()
    blurred.show()

    # save the new image
    blurred.save("blurred.png")

except:
    print "Unable to load image" 

To display the image, we used the "show()" methods. If you don't see anything, you could try installing ImageMagick first and run the example again.
```

## 创建缩略图

一个很常见的事情是为图像创建缩略图。缩略图是图片的缩小版本，但仍然包含图像的所有最重要的方面。

```py
from PIL import Image

size = (128, 128)
saved = "lenna.jpeg"

try:
    im =  Image.open("Lenna.png")
except:
    print "Unable to load image"

im.thumbnail(size)
im.save(saved)
im.show()
```

我们程序的结果，显示缩略图:

## 枕头中的过滤器

枕头模块提供以下一组预定义的图像增强过滤器:

```py
BLUR
CONTOUR
DETAIL
EDGE_ENHANCE
EDGE_ENHANCE_MORE
EMBOSS
FIND_EDGES
SMOOTH
SMOOTH_MORE
SHARPEN
```

在我们今天的最后一个例子中，我们将展示如何将“轮廓”滤镜应用到图像中。下面的代码将获取我们的图像并应用

```py
from PIL import Image, ImageFilter

im = Image.open("Lenna.png")
im = im.filter(ImageFilter.CONTOUR)

im.save("lenna" + ".jpg")
im.show()
```

我们应用了“轮廓”滤镜的图像:

我很喜欢试用枕头，以后我会写更多关于它的帖子。

##### 更多阅读

[http://pillow . readthe docs . org/en/latest/handbook/tutorial . html](https://pillow.readthedocs.org/en/latest/handbook/tutorial.html "pillow")