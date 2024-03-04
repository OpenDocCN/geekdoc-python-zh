# 用 Python 截图

> 原文：<https://www.pythonforbeginners.com/basics/take-screenshots-in-python>

我们经常需要在我们的 PC 或笔记本电脑上截图，以捕捉视频或网站上的图像。在本文中，我们将讨论 python 中截图的两种方式。

## 使用 Python 中的 pyautogui 模块截图

我们可以使用 pyautogui 模块在 python 中截图。除了 pyautogui 模块，我们还需要 OpenCV 和 numpy 模块来捕获屏幕截图。您可以使用 PIP 安装这些模块，如下所示。

```py
pip install pyautogui numpy opencv-python
```

安装模块后，您可以在程序中使用它们。`pyautogui`模块为我们提供了`screenshot()`函数，在这个函数的帮助下我们可以截图。执行`screenshot()` 函数时，返回 RGB 格式的 PIL(python 图像库)文件。首先，我们将使用`numpy.array()` 函数将 PIL 文件转换成一个 numpy 数组。

创建 numpy 数组后，我们将把图像的 RGB 格式转换成 BGR 格式，然后再存储到存储器中。为此，我们将使用`opencv.cvtColor()`函数来执行该操作。`opencv.cvtColor() `将前一步中创建的 numpy 数组作为其第一个输入参数，并将常量`cv2.COLOR_RGB2BGR`作为其第二个输入参数，以表明我们正在将 RGB 格式转换为 BGR 格式。执行后，`opencv.cvtColor()`返回最终图像。

为了存储最终的截图，我们将使用 `cv2.imwrite()`函数。它将图像文件的名称作为第一个输入参数，将表示图像的数组作为第二个输入参数。执行`imwrite()`功能后，屏幕截图保存在永久存储器中。

下面是使用`pyautogui`和`opencv`模块在 python 中截屏的代码。

```py
import cv2
import pyautogui
import numpy

pil_file = pyautogui.screenshot()
numpy_arr = numpy.array(pil_file)
image = cv2.cvtColor(numpy_arr, cv2.COLOR_RGB2BGR)
cv2.imwrite('screenshot.png', image)
```

输出:

![](img/0c765ada43744c6aa54b334e5c248206.png)



## 使用 Python 中的 pyscreenshot 模块截图

在使用`pyautogui`模块截图时，我们必须执行各种操作来生成输出图像。为了避免麻烦，我们可以使用 Python 中的`pyscreenshot`模块截图。

`pyscreenshot`模块为我们提供了`grab()`函数，借助它可以截图。`grab()`函数在执行时以数组的形式返回屏幕截图。您可以使用`pyscreenshot`模块中定义的`save()`方法将截图保存在存储器中。当在包含使用`grab()` 函数捕获的图像的数组上调用`save()`方法时，该方法将文件名作为输入参数，并将图像保存在存储器中。

```py
import pyscreenshot

image = pyscreenshot.grab()
image.save('Output_screenshot.png')
```

输出:

![](img/0c765ada43744c6aa54b334e5c248206.png)



您还可以使用在`pyscreenshot`模块中定义的`show()`方法来查看截图。当在包含使用`grab()` 函数捕获的图像的数组上调用`show()`方法时，会在屏幕上显示图像。

## 结论

在本文中，我们讨论了 python 中截图的两种方式。要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中的[列表理解的文章。你可能也会喜欢这篇关于 python 中的](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)[集合理解的文章。](https://www.pythonforbeginners.com/basics/set-comprehension-in-python)