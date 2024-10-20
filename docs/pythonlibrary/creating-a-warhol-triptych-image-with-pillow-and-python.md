# 用枕头和蟒蛇皮创作沃霍尔三联画

> 原文：<https://www.blog.pythonlibrary.org/2021/06/17/creating-a-warhol-triptych-image-with-pillow-and-python/>

安迪·沃霍尔是一位著名的艺术家，他创造了一幅著名的图像，其中有同一张脸的多个副本，但背景颜色不同。

你可以用 Python 和 [Pillow package](https://pillow.readthedocs.io/en/stable/) 的软件做类似的事情。您还需要安装 NumPy 来完成这项工作。

让我们来看看这一切是如何运作的！

## 入门指南

你需要做的第一件事是确保你有枕头和 NumPy 安装。如果您使用 pip，可以尝试运行以下命令:

```py
python3 -m pip install numpy Pillow
```

如果您还没有安装 NumPy 和 Pillow，这将安装它们。

如果您运行的是 Anaconda，那么这两个包应该已经安装好了。

现在你已经准备好创作一些艺术品了！

## 创作三联画

用 Python 创建三联画并不需要很多代码。你所需要的只是一点知识和一些实验。第一步是了解你为什么需要 NumPy。

NumPy 与其说是枕头的替代品，不如说是增强枕头功能的一种方式。您可以使用 NumPy 来做 Pillow 本身做的一些事情。对于本节中的示例，您将使用作者的这张照片:

![Michael Driscoll](img/e86f094059dca34aaa28a74a93c699a1.png)

迈克尔·德里斯科尔

为了体验如何在 Pillow 中使用 NumPy，您将创建一个 Python 程序，将几个图像连接在一起。这将创建您的三联图像！打开 Python 编辑器，创建一个名为`concatenating.py`的新文件。然后在其中输入以下代码:

```py
# concatenating.py

import numpy as np
from PIL import Image

def concatenate(input_image_path, output_path):
    image = np.array(Image.open(input_image_path))

    red = image.copy()
    red[:, :, (1, 2)] = 0

    green = image.copy()
    green[:, :, (0, 2)] = 0

    blue = image.copy()
    blue[:, :, (0, 1)] = 0

    rgb = np.concatenate((red, green, blue), axis=1)
    output = Image.fromarray(rgb)
    output.save(output_path)

if __name__ == "__main__":
    concatenate("author.jpg", "stacked.jpg")
```

这段代码将使用 Pillow 打开图像。然而，不是将图像保存为一个`Image`对象，而是将该对象传递到一个 Numpy `array()`中。然后创建数组的三个副本，并使用一些矩阵数学将其他颜色通道清零。例如，对于`red`，你将绿色和蓝色通道归零，只留下红色通道。

当你这样做时，它将创建原始图像的三个着色版本。你现在会有一个红色，绿色和蓝色版本的照片。然后使用 NumPy 将三幅图像连接成一幅。

为了保存这个新图像，您使用 Pillow 的`Image.fromarray()`方法将 NumPy 数组转换回 Pillow `Image`对象。

运行此代码后，您将看到以下结果:

![Python Triptych](img/b652b63e1a28174bde760db65803395a.png)

这是一个整洁的效果！

NumPy 可以做 Pillow 不容易做的其他事情，比如二值化或者去噪。当您将 NumPy 与其他科学 Python 包(如 SciPy 或 Pandas)结合使用时，您可以做得更多。

## 包扎

Python 功能强大。你可以用 Python，Pillow 和 NumPy 做很多事情。你应该试着用你自己的图像做这件事，看看你能想出什么。例如，您可以创建四个图像并将它们放在一个正方形中，而不是从左到右排列图像！

## 相关阅读

*   如何使用 Python 改变图像的颜色？
*   StackOverflow: [Python PIL 图像分割成 RGB](https://stackoverflow.com/questions/51325224/python-pil-image-split-to-rgb)
*   枕头:[用 Python 进行图像处理](https://gum.co/pypillow)