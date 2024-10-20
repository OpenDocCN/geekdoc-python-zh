# OpenCV putText()–在图像上书写文本

> 原文：<https://www.askpython.com/python-modules/opencv-puttext>

你好，学习伙伴！在本教程中，我们将学习如何使用 OpenCV putText()方法在 Python 中的图像上编写字符串文本。所以让我们开始吧。

## OpenCV putText()方法是什么？

[OpenCV](https://www.askpython.com/python-modules/read-images-in-python-opencv) Python 是一个主要针对实时计算机视觉和[图像处理](https://www.askpython.com/python/examples/image-processing-in-python)问题的编程函数库。

OpenCV 包含用于在任何图像上放置文本的`putText()`方法。该方法使用以下参数。

*   ***img:*** 你要在上面写文字的图像。
*   ***文字:*** 您要在图像上书写的文字。
*   ***org:*** 是你的文字左下角的坐标。它被表示为一个由两个值(X，Y)组成的元组。x 表示距图像左边缘的距离，Y 表示距图像上边缘的距离。
*   ***字体:*** 表示你要使用的字体类型。OpenCV 只支持[好时字体](https://en.wikipedia.org/wiki/Hershey_fonts)的子集。
    *   字体 _ 好时 _ 单纯形
    *   FONT_HERSHEY_PLAIN
    *   FONT_HERSHEY_DUPLEX
    *   FONT _ HERSHEY _ 复杂
    *   FONT_HERSHEY_TRIPLEX
    *   FONT _ 好时 _COMPLEX_SMALL
    *   字体 _ 好时 _ 脚本 _ 单纯形
    *   FONT _ HERSHEY _ SCRIPT _ 复杂
    *   字体 _ 斜体
*   ***fontScale:*** 它用来增加/减少你的文本的大小。字体比例因子乘以特定字体的基本大小。
*   ***颜色:*** 它代表你要给的文本的颜色。它采用`BGR`格式的值，即首先是蓝色值，然后是绿色值，红色值都在 0 到 255 的范围内。
*   ***粗细(可选):*** 表示用来绘制文本的线条粗细。默认值为 1。
*   ***线型(可选):*** 表示您想要使用的线型。4 可用的[线型](https://docs.opencv.org/3.4/d0/de1/group__core.html#gaf076ef45de481ac96e0ab3dc2c29a777)有
    *   装满
    *   第 4 行
    *   LINE_8(默认)
    *   线性的
*   ***bottomLeftOrigin(可选):*** 为真时，图像数据原点在左下角。否则，它在左上角。默认值为 False。

## 使用 OpenCV–cv2 . puttext()方法在图像上添加文本

让我们使用下面的图片，用 OpenCV putText()方法写一个“早安”消息。

![OpenCV PutText Initial Image](img/a990dba548312772d8bd16446a15f5de.png)

OpenCV PutText Initial Image

```py
# importing cv2 library
import cv2

# Reading the image
image = cv2.imread("Wallpaper.jpg")

# Using cv2.putText()
new_image = cv2.putText(
  img = image,
  text = "Good Morning",
  org = (200, 200),
  fontFace = cv2.FONT_HERSHEY_DUPLEX,
  fontScale = 3.0,
  color = (125, 246, 55),
  thickness = 3
)

# Saving the new image
cv2.imwrite("New Wallpaper.jpg", new_image)

```

![OpenCV PutText Final Image](img/f9b1171a6ea1ad4265929cca46300cdf.png)

OpenCV PutText Final Image

## 结论

在本教程中，您学习了如何使用 OpenCV putText()方法在图像上书写文本。感谢阅读！！