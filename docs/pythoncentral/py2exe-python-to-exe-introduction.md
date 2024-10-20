# py2exe: Python 转 exe 简介

> 原文：<https://www.pythoncentral.io/py2exe-python-to-exe-introduction/>

py2exe 是一种将 Python 脚本转换成 Windows 的简单方法。exe 应用程序。它是一个基于 Distutils 的实用程序，允许您在 Windows 计算机上运行用 Python 编写的应用程序，而无需用户安装 Python。当您需要将程序作为独立的应用程序分发给最终用户时，这是一个很好的选择。py2exe 目前只能在 Python 2.x 中运行。

首先你需要从 sourceforge 官方网站下载并安装 py2exe

现在，为了能够创建可执行文件，我们需要在您希望可执行的脚本所在的同一文件夹中创建一个名为`setup.py`的文件:

```py

# setup.py

from distutils.core import setup

import py2exe

setup(console=['myscript.py'])

```

在上面的代码中，我们将为`myscript.py`创建一个可执行文件。`setup`函数接收一个参数`console=['myscript.py']`，告诉 py2exe 我们有一个名为`myscript.py`的控制台应用程序。

然后，为了创建可执行文件，只需从 Windows 命令提示符(cmd)运行`python setup.py py2exe`。你会看到很多输出，然后会创建两个文件夹:`dist`和`build`。py2exe 使用`build`文件夹作为临时文件夹来创建可执行文件所需的文件。`dist`文件夹存储可执行文件和运行该可执行文件所需的所有文件。删除`build`文件夹是安全的。注意:运行`python setup.py py2exe`假设您的 path 环境变量中有 Python。如果不是这样，就使用`C:\Python27\python.exe setup.py py2exe`。

现在测试您的可执行文件是否有效:

```py

cd dist

myscript.exe

```

## GUI 应用程序

现在是时候创建一个 GUI 应用程序了。在这个例子中，我们将使用 Tkinter:
【python】
# tkexample . py
' ' '一个非常基本的 Tkinter 例子'
从 Tkinter import *
root = Tk()
root . title('一个 Tk 应用')
Label(text= '我是标签')。pack(pady = 15)
root . main loop()

然后创建`setup.py`，我们可以使用以下代码:
【python】
# setup . py
从 distutils.core 导入 setup
导入 py2exe

setup(windows =[' tk example . py '])

`setup`函数现在正在接收一个参数`windows=['tkexample.py']`,告诉 py2exe 这是一个 GUI 应用程序。再次创建在 Windows 命令提示符下运行`python setup.py py2exe`的可执行文件。要运行该应用程序，只需在 Windows 资源管理器中导航到`dist`文件夹，然后双击`tkexample.exe`。

## 使用外部模块

前面的例子是从 Python 标准库中导入模块。py2exe 默认包含标准库模块。然而，如果我们安装了第三方库，py2exe 很可能不包括它。在大多数情况下，我们需要显式地包含它。例如，一个应用程序使用`ReportLab`库来制作 PDF 文件:

```py

# invoice.py

from reportlab.pdfgen import canvas

from reportlab.lib.pagesizes import letter

from reportlab.lib.units import mm
if _ _ name _ _ = = ' _ _ main _ _ ':
name = u ' Mr。约翰·多伊'
城市= '佩雷拉'
地址= '榆树街'
电话= '555-7241' 
 c =画布。Canvas(filename='invoice.pdf '，pagesize= (letter[0]，letter[1]/2))
c . set font(' Helvetica '，10) 
 #打印客户数据
 c.drawString(107*mm，120*mm，姓名)
 c.drawString(107*mm，111*mm，城市)
 c.drawString(107*mm，106*mm，地址)
 c.drawString(107*mm
```

为了包含`ReportLab`模块，我们创建一个`setup.py`文件，将一个选项字典传递给`setup`函数:

```py

# setup.py

from distutils.core import setup

import py2exe
setup(
console =[' invoice . py ']，
options = {
' py2exe ':{
' packages ':[' reportlab ']
}
}
)

```

能够在其他计算机上运行可执行文件的最后一步是，运行可执行文件的计算机需要安装 Microsoft Visual C++ 2008 可再发行软件包。在这里可以找到一个很好的指南来解释如何做到这一点。然后只需将`dist`文件夹复制到另一台计算机并执行。exe 文件。

最后，请考虑以下建议:

*   大多数时候创建的可执行文件是向前兼容的:如果您在 Windows XP 中创建可执行文件，它将在 Vista 和 7 中运行。然而，它不是向后兼容的:如果你在 Windows 7 中创建可执行文件，它将不能在 Windows XP 上运行。
*   如果您导入了第三方库，请确保在交付软件之前测试所有的应用程序功能，因为有时会创建可执行文件，但缺少一些库。在这种情况下，当您尝试访问使用外部库的功能时，将会出现运行时错误。
*   py2exe 自 2008 年以来就没有更新过，所以它不适合 Python 3。

如果你需要更多关于 py2exe 的信息，请访问[官方网站](https://www.py2exe.org/ "py2exe")。