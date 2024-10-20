# 使用网络摄像头和 Python OpenCV 进行实时素描[简易指南]

> 原文：<https://www.askpython.com/python/examples/sketch-using-webcam>

在今天的教程中，你将学习 OpenCV 的一个应用，这将让你意识到 OpenCV 有多强大。

在该项目中，我们将采用一个实时网络摄像头，并在 numpy 和 OpenCV 库的帮助下将其转换为一个实时草图。

让我们从这个惊人的项目开始吧！

* * *

## 步骤 1:导入模块

首先，我们需要导入`OpenCV`和`[Numpy](https://www.askpython.com/python/numpy-trigonometric-functions)`(假设您已经安装了库)。我们在代码中定义 OpenCV 和 Numpy 如下:

```py
import cv2
import numpy as np

```

* * *

## 步骤 2:定义一个函数将框架转换为草图

为了将一个框架转换成草图，我们将遵循下面列出的一些步骤:

1.  将图像转换成`gray`图像
2.  对获得的灰度图像应用`Gaussian Blur`
3.  将`Canny Edge Detection`应用到高斯图像
4.  最后，反转图像，得到`Binary Inverted Image`

该函数的代码如下所示。

```py
def sketch(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask

```

* * *

## 步骤 3:打开网络摄像头并应用功能

我们需要使用网络摄像头，并从视频中提取图像帧。为了达到同样的效果，我们将使用`VideoCapture`和`read`函数一个接一个地提取帧。

现在使用`imshow`功能显示实时网络摄像头，并应用上一步创建的草图功能。

最后一步是为窗口创建一个退出条件。这里我们保留了键`Enter Key`作为窗口的退出键。最后，摧毁程序中所有打开的和将要关闭的窗口。

```py
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Our Live Sketcher', sketch(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()

```

* * *

## 完整的代码

```py
import cv2
import numpy as np

def sketch(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Our Live Sketcher', sketch(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()

```

* * *

## 最终输出

下面的小视频显示了运行上一节提到的全部代码后得到的最终输出。

* * *

## 结论

我希望你理解的概念，并喜欢输出。自己尝试简单的代码，并观察 OpenCV 库的强大功能。

编码快乐！😇

想了解更多？查看下面提到的教程:

1.  [如何使用 Python OpenCV 从视频中提取图像？](https://www.askpython.com/python/examples/extract-images-from-video)
2.  [Python 和 OpenCV:对图像应用滤镜](https://www.askpython.com/python/examples/filters-to-images)
3.  [Python 中的动画](https://www.askpython.com/python-modules/animation-in-python-celluloid)

* * *