# 如何用 Python 给你的照片添加边框

> 原文：<https://www.blog.pythonlibrary.org/2017/10/26/how-to-add-a-border-to-your-photos-with-python/>

有时给照片添加简单的边框很有趣。Pillow 软件包有一个非常简单的方法，通过它的 **ImageOps** 模块给你的图像添加这样的边框。像往常一样，您需要安装 Pillow 来完成本文中的任何示例。如果您还没有它，您可以使用 pip 安装它:

```py

pip install Pillow

```

现在，我们已经处理好了那部分内务，让我们学习如何添加一个边框！

* * *

### 添加边框

![](img/c2f99eb8191f7bab08b18399cd1fb3fb.png)

本文的重点是使用 [ImageOps](http://pillow.readthedocs.io/en/4.3.x/reference/ImageOps.html) 模块来添加我们的边框。对于这个例子，我们将使用我拍摄的这张整洁的蝴蝶照片。我们写点代码吧！

```py

from PIL import Image, ImageOps

def add_border(input_image, output_image, border):
    img = Image.open(input_image)

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border)
    else:
        raise RuntimeError('Border is not an integer or tuple!')

    bimg.save(output_image)

if __name__ == '__main__':
    in_img = 'butterfly.jpg'

    add_border(in_img, output_image='butterfly_border.jpg',
               border=100)

```

在上面的代码中，我们创建了一个可以接受三个参数的函数:

*   输入图像路径
*   输出图像路径
*   边框，可以是一个整数或最多 4 个整数的元组，表示图像四个边中每一个边的像素

我们打开输入图像，然后检查边界的类型。是 int 还是 tuple 还是别的？如果是前两者之一，我们通过调用 **expand()** 函数来添加边框。否则我们会引发一个错误，因为我们传入了一个无效的类型。最后，我们保存图像。这是我得到的结果:

![](img/168302cf0b38ac8660e75bdff4fab044.png)

正如你所看到的，当你只是传入一个整数作为你的边界，它适用于图像的所有四边。如果我们希望顶部和底部的边界不同于右边和左边，我们需要指定。让我们更新代码，看看会发生什么！

```py

from PIL import Image, ImageOps

def add_border(input_image, output_image, border):
    img = Image.open(input_image)

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border)
    else:
        raise RuntimeError('Border is not an integer or tuple!')

    bimg.save(output_image)

if __name__ == '__main__':
    in_img = 'butterfly.jpg'

    add_border(in_img,
               output_image='butterfly_border_top_bottom.jpg',
               border=(10, 50))

```

在这里，我们想添加一个 10 像素的边界到左边和右边，一个 50 像素的边界到图片的顶部和底部。如果运行此代码，您应该会得到以下结果:

![](img/a15f3773a6da3767a1995291710b1952.png)

现在让我们尝试为所有四个边指定不同的值！

```py

from PIL import Image, ImageOps

def add_border(input_image, output_image, border):
    img = Image.open(input_image)

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border)
    else:
        raise RuntimeError('Border is not an integer or tuple!')

    bimg.save(output_image)

if __name__ == '__main__':
    in_img = 'butterfly.jpg'

    add_border(in_img,
               output_image='butterfly_border_all_different.jpg',
               border=(10, 30, 20, 50))

```

在这个例子中，我们告诉 Pillow 我们想要一个左边是 10 像素，上边是 30 像素，右边是 20 像素，下边是 50 像素宽的边框。当我运行这段代码时，我得到了这个:

![](img/ee3467bb47eda3631aba29c6c2389a9c.png)

坦白地说，我不知道为什么你希望所有的四个边都有不同大小的边框，但是如果你这样做了，就很容易应用了。

* * *

### 更改边框颜色

您还可以设置正在添加的边框的颜色。默认明显是黑色的。Pillow package 支持通过“rgb ”(即红色为“ff0000”)、“rgb(红、绿、蓝)”指定颜色，其中 RGB 值为 0 到 255 之间的整数、HSL 值或 HTML 颜色名称。让我们更新代码并给边框添加一些颜色:

```py

from PIL import Image, ImageOps

def add_border(input_image, output_image, border, color=0):
    img = Image.open(input_image)

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border, fill=color)
    else:
        raise RuntimeError('Border is not an integer or tuple!')

    bimg.save(output_image)

if __name__ == '__main__':
    in_img = 'butterfly.jpg'

    add_border(in_img,
               output_image='butterfly_border_indianred.jpg',
               border=100,
               color='indianred')

```

你会注意到我们在这里添加了一个新的参数来指定我们想要的边框颜色。默认为黑色，即零(0)。在这个例子中，我们传入了 HTML 颜色‘Indian red’。结果如下:

![](img/356447f6110d9a31439ef3b472dc6aa2.png)

通过将传入的值从' indianred '更改为' #CD5C5C '，可以获得相同的效果。

只是为了好玩，试着把你传入的值改成‘RGB(255，215，0)’,这是一种金色。如果你这样做，你可以让你的边界看起来像这样:

![](img/8f9a01840f98569c0f2540f88ac04c15.png)

请注意，您也可以传入' gold '、' Gold '或' #FFD700 '，结果会是相同的颜色。

* * *

### 包扎

此时，你应该知道如何给你的照片添加简单的边框。如您所见，您可以单独或成对更改每条边的边框颜色像素数量，甚至可以同时更改所有四条边的边框颜色像素数量。你也可以把边框的颜色改成任何颜色。花点时间研究一下代码，看看你能想出什么来！

* * *

### 相关阅读

*   Kanoki - [为你的照片画铅笔素描](http://kanoki.org/2017/08/15/draw-pencil-sketches-of-your-photo/)
*   如何用 Python 给你的[照片加水印](https://www.blog.pythonlibrary.org/2017/10/17/how-to-watermark-your-photos-with-python/)
*   [如何用 Python 调整照片大小](https://www.blog.pythonlibrary.org/2017/10/12/how-to-resize-a-photo-with-python/)
*   用 Python 将一张[照片转换成黑白](https://www.blog.pythonlibrary.org/2017/10/11/convert-a-photo-to-black-and-white-in-python/)
*   [如何用 Python 旋转/镜像照片](https://www.blog.pythonlibrary.org/2017/10/05/how-to-rotate-mirror-photos-with-python/)
*   [如何用 Python 裁剪照片](https://www.blog.pythonlibrary.org/2017/10/03/how-to-crop-a-photo-with-python/)
*   使用 Python 增强照片