# python patchly——从大型图像中提取补丁

> 原文：<https://www.askpython.com/python-modules/python-patchify>

在本教程中，我们将看看如何使用 Python 编程语言从巨大的照片中提取补丁。

* * *

## 介绍

当训练任何深度学习算法时，我们更喜欢使用小图像，因为它们会产生更好的结果。但是如果我们有巨大的图像呢？一种解决方案是将较大的照片分成较小的碎片，允许我们训练任何算法。

你可能想知道`patch`是什么意思？顾名思义，图像块是图片中的一组像素。假设我有一个 20×20 像素的图像。它可以被分成 1000 个 2 × 2 像素的正方形小块。

* * *

## Python Patchify 简介

Python `Patchify`是一个包，用于裁剪照片并将裁剪或修补的图像保存在一个`Numpy`数组中。但是首先，使用 [pip 命令](https://www.askpython.com/python-modules/python-pip)确保您的系统中已经安装了 patchify。

```py
pip install patchify

```

Patchify 可以根据指定的补丁单元大小将一张图片分成小的重叠部分，然后将这些区域与原始图像融合。

* * *

## 利用 python 分片提取图像分片

现在让我们开始使用这个模块，从这里开始提取图像补丁。

### 1.导入模块

我们从导入将大图像转换为补丁所需的模块开始。这里的`Numpy`用于创建图像数据，`patchify`模块用于将图像转换为图像补丁。

```py
import numpy as np
from patchify import patchify

```

### 2.创建图像数据

我们以 numpy `array`的形式创建图像数据，让我们看看图像的初始形状。

```py
image = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13, 14, 15, 16]])
print(image.shape)

```

### 3.从图像中提取补丁

```py
patches = patchify(image, (2,2), step=2) 
print(patches.shape)

```

* * *

## 完整的代码和输出

```py
import numpy as np
from patchify import patchify
image = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13, 14, 15, 16]])
print(image.shape)
patches = patchify(image, (2,2), step=2) 
print(patches.shape)

```

```py
(4, 4)
(2, 2, 2, 2)

```

* * *

我希望你清楚这个概念，也明白如何生成补丁。这同样适用于 3D 图像！试试吧！

编码快乐！😇

* * *