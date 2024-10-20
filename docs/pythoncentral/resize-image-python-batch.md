# 使用 Python 2.x 调整图像大小(批量调整)

> 原文：<https://www.pythoncentral.io/resize-image-python-batch/>

The module we use in this recipe to resize an image with Python is `PIL`. At the time of writing, it is only available for Python 2.x. If you want to do more with image editing, be sure to checkout our article on how to [watermark an image in Python](https://www.pythoncentral.io/watermark-images-python-2x/ "Watermark an Image in Python")

Python 是一种非常强大的脚本语言，你会惊喜地发现，你想要构建的许多常用函数都以库的形式存在。Python 生态系统非常活跃，充满了库。

举个例子，今天我将向你展示如何轻松地构建一个 Python 脚本，它将使用 Python 来*调整*一个图像的大小，我们将扩展它来将一个文件夹中的所有图像调整到你选择的尺寸。这是用 Python 的`PIL` (Python 映像库)。首先我们需要安装这个。

## 在 Windows 上安装 PIL

你可以在这里下载并安装 PIL 图书馆。

## 在 Mac 上安装 PIL

对于 MacPorts，您可以使用以下命令安装 PIL(对于 Python 2.7):

```py

$ port install py27-pil

```

## 在 Ubuntu 上安装 PIL

在 Linux 上安装 PIL 往往因发行版而异，所以我们将只讨论 Ubuntu。要在 Ubuntu 上安装 PIL，请使用以下命令:

```py

$ sudo apt-get install libjpeg libjpeg-dev libfreetype6 libfreetype6-dev zlib1g-dev

```

也可以使用`pip`，比如:

```py

$ pip install PIL

```

好了，现在来看看调整图像大小的脚本！

## 用 Python 调整图像大小

要使用 Python 的 PIL 调整一幅图像的大小，我们可以使用以下命令:

```py

from PIL import Image
宽度= 500 
高度= 500
#打开图像文件。
 img = Image.open(os.path.join(目录，图像))
#调整大小。
 img = img.resize((宽度，高度)，图像。双线性)
#保存回磁盘。
 img.save(os.path.join(目录，'调整大小-' + image)) 

```

假设我们已经知道了宽度和高度。但是如果我们只知道宽度，需要计算高度呢？使用 Python 调整图像大小时，我们可以如下计算高度:

```py

from PIL import Image
基底宽度= 500
#打开图像文件。
 img = Image.open(os.path.join(目录，图像))
#使用相同的纵横比计算高度
width percent =(base width/float(img . size[0])
height = int((float(img . size[1])* float(width percent)))
#调整大小。
 img = img.resize((baseWidth，height)，Image。双线性)
#保存回磁盘。
 img.save(os.path.join(目录，'调整大小-' + image)) 

```

简单！现在，我们向您展示如何使用 Python 批量调整图像大小。

## 使用 Python 批量调整图像大小

这是调用脚本的方式:

```py

python image_resizer.py -d 'PATHHERE' -w 448 -h 200

```

首先，让我们导入让这个脚本工作所需的内容:

```py

import os

import getopt

import sys

from PIL import Image

```

*   操作系统:让我们访问与电脑交互的功能，在这种情况下，从文件夹中获取文件。
*   getopt :让我们轻松地访问最终用户传入的命令行参数。
*   *图像*:将允许我们调用`resize`函数，该函数将执行应用程序的重载。

### 我们的批处理图像缩放器命令行参数

接下来，让我们继续处理命令行参数。我们还必须考虑到一个论点丢失的微小可能性。在这种情况下，我们将显示一条错误消息并终止程序。

```py

# Let's parse the arguments.

opts, args = getopt.getopt(sys.argv[1:], 'd:w:h:')
#为需要的变量设置一些默认值。
目录= '' 
宽度= -1 
高度= -1
#如果传入了参数，请将其赋给正确的变量。
对于 opt，opts 中的 arg:
if opt = = '-d ':
directory = arg
elif opt = = '-w ':
width = int(arg)
elif opt = = '-h ':
height = int(arg)
#我们必须确保所有的参数都通过了。
如果 width == -1 或 height == -1 或 directory == '': 
打印('无效的命令行参数。-d[目录]' \
'-w[宽度]-h[高度]都是必需的')
#如果缺少参数，请退出应用程序。
退出()

```

上面的评论是不言自明的。我们解析参数，用默认值设置变量的用法，并给它们赋值。如果一个或多个变量丢失，我们将终止应用程序。

很好，现在我们可以专注于这个脚本的目的了。让我们获取文件夹中的每个图像并对其进行处理。

```py

# Iterate through every image given in the directory argument and resize it.

for image in os.listdir(directory):

    print('Resizing image ' + image)
#打开图像文件。
 img = Image.open(os.path.join(目录，图像))
#调整大小。
 img = img.resize((宽度，高度)，图像。双线性)
#保存回磁盘。
 img.save(os.path.join(目录，'调整大小-' + image))
打印(“批处理完成。”)

```

[Image.open](http://effbot.org/imagingbook/image.htm) 函数正在返回一个`Image`对象，这反过来让我们对它应用`resize`方法。为了简单起见，我们使用`Image.BILINEAR`算法。

这就是全部了。正如你所看到的，Python 是一种非常敏捷的语言，允许开发者专注于解决业务需求。