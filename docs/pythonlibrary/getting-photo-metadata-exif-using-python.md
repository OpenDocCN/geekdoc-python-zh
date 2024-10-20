# 使用 Python 获取照片元数据(EXIF)

> 原文：<https://www.blog.pythonlibrary.org/2010/03/28/getting-photo-metadata-exif-using-python/>

上周，我试图找出如何获得我的照片的元数据。我注意到 Windows 可以在我的照片上显示相机型号、创建日期和许多其他数据，但我不记得这些数据叫什么了。我终于找到了我要找的东西。术语是 EXIF(可交换图像文件格式)。在本帖中，我们将看看各种第三方软件包，它们让您可以访问这些信息。

我的第一个想法是 [Python 图像库](http://www.pythonware.com/products/pil/)会有这个功能，但是我还没有找到 EXIF 术语，如果没有它，在 PIL 的手册中也找不到这个信息。幸运的是，我最终通过一个 [stackoverflow 线程](http://stackoverflow.com/questions/765396/exif-manipulation-library-for-python)找到了使用 PIL 的方法。这是它展示的方法:

```py

from PIL import Image
from PIL.ExifTags import TAGS

def get_exif(fn):
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
    return ret

```

这工作得很好，并返回一个很好的字典对象。我发现有几个字段没有用，比如“MakerNote”字段，它看起来像许多十六进制值，所以您可能只想使用某些数据。以下是我得到的一些信息的例子:

```py

{'YResolution': (180, 1), 
 'ResolutionUnit': 2, 
 'Make': 'Canon', 
 'Flash': 16, 
 'DateTime': '2009:09:11 11:29:10', 
 'MeteringMode': 5, 
 'XResolution': (180, 1), 
 'ColorSpace': 1, 
 'ExifImageWidth': 3264, 
 'DateTimeDigitized': '2009:09:11 11:29:10', 
 'ApertureValue': (116, 32), 
 'FocalPlaneYResolution': (2448000, 169), 
 'CompressedBitsPerPixel': (3, 1), 
 'SensingMethod': 2, 
 'FNumber': (35, 10), 
 'DateTimeOriginal': '2009:09:11 11:29:10', 
 'FocalLength': (26000, 1000), 
 'FocalPlaneXResolution': (3264000, 225), 
 'ExifOffset': 196, 
 'ExifImageHeight': 2448, 
 'ISOSpeedRatings': 100, 
 'Model': 'Canon PowerShot S5 IS', 
 'Orientation': 1, 
 'ExposureTime': (1, 200), 
 'FileSource': '\x03', 
 'MaxApertureValue': (116, 32), 
 'ExifInteroperabilityOffset': 3346, 
 'FlashPixVersion': '0100', 
 'FocalPlaneResolutionUnit': 2, 
 'YCbCrPositioning': 1, 
 'ExifVersion': '0220'}

```

我真的不知道所有这些值意味着什么，但我知道我可以使用其中的一些。我想要这些数据的目的是扩展我的简单的[图像浏览器](https://www.blog.pythonlibrary.org/2010/03/26/creating-a-simple-photo-viewer-with-wxpython/),这样它可以向用户显示更多关于他们照片的信息。

以下是我发现的其他几个可以访问 EXIF 数据的图书馆:

*   [Python 的媒体元数据](http://sourceforge.net/projects/mmpython/files/)
*   [EXIF.py](http://sourceforge.net/projects/exif-py/)
*   [Python Exif 解析器](http://sourceforge.net/projects/pyexif/)
*   一个博主的 [Exif 解析器](http://fetidcascade.com/pyexif.html)
*   [pyexiv2](http://tilloy.net/dev/pyexiv2/)

我尝试了 Python Exif 解析器，它工作得相当好。当我在工作中试图在我的 Python 2.5 机器上安装 pyexiv2 时，我收到了一条关于 Python 2.6 未找到的错误消息，然后安装程序退出了。pyexiv2 网站上没有提到它需要特定版本的 Python 才能工作，所以这有点令人沮丧。这些模块中的大多数很少或者没有文档，这也非常令人沮丧。据我所知，EXIF.py 应该通过命令行使用，而不是作为一个可导入的模块。

总之，回到 Python Exif 解析器。它实际上比 PIL 更容易使用。将 exif.py 文件复制到 Python 路径后，您需要做的就是:

```py

import exif
photo_path = "somePath\to\a\photo.jpg"
data = exif.parse(photo_path)

```

上面的代码返回的信息与 PIL 代码片段返回的信息基本相同，尽管它对“MakersNote”使用了整数而不是十六进制，并且它有几个“Tag0xa406”字段，而 PIL 数据有一些数字字段(我在上面排除了这些字段)。我假设他们以不同的方式引用相同的信息。

无论如何，当你试图发现这些信息时，如果你发现自己在网上游荡，希望你会偶然发现这篇文章，它会给你指出正确的方向。