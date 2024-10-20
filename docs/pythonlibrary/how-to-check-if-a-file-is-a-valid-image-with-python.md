# 如何用 Python 检查文件是否是有效图像

> 原文：<https://www.blog.pythonlibrary.org/2020/02/09/how-to-check-if-a-file-is-a-valid-image-with-python/>

Python 的标准库中有许多模块。一个经常被忽视的是 [imghdr](https://docs.python.org/3.8/library/imghdr.html) ，它可以让你识别文件、字节流或类路径对象中包含的图像类型。

**imghdr** 可以识别以下图像类型:

*   rgb
*   可交换的图像格式
*   pbm
*   precision-guided munition 精密制导武器
*   百万分率
*   一口
*   拉斯特
*   xbm
*   jpeg / jpg
*   位图文件的扩展名
*   png
*   webp
*   遗嘱执行人

下面是如何使用 imghdr 来检测文件的图像类型:

```py
>>> import imghdr
>>> path = 'python.jpg'
>>> imghdr.what(path)
'jpeg'
>>> path = 'python.png'
>>> imghdr.what(path)
'png'

```

你所需要做的就是传递一个路径到 **imghdr.what(path)** 它会告诉你它认为图像类型是什么。

另一种方法是使用[枕头包](https://pillow.readthedocs.io/en/stable/)，如果你还没有 pip，你可以安装它。

以下是枕头的使用方法:

```py
>>> from PIL import Image
>>> img = Image.open('/home/mdriscoll/Pictures/all_python.jpg')
>>> img.format
'JPEG'

```

这个方法几乎和使用 **imghdr** 一样简单。在这种情况下，您需要创建一个**图像**对象，然后调用它的**格式**属性。Pillow 支持的图像类型比 **imghdr** 多[，但是文档并没有真正说明**格式**属性是否适用于所有这些图像类型。](https://pillow.readthedocs.io/en/latest/handbook/image-file-formats.html)

无论如何，我希望这有助于您识别文件的图像类型。