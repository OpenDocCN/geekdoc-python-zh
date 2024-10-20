# 使用 Python 获得屏幕分辨率

> 原文：<https://www.blog.pythonlibrary.org/2015/08/18/getting-your-screen-resolution-with-python/>

我最近在寻找用 Python 获得我的屏幕分辨率的方法，以帮助诊断一个不能正常运行的应用程序的问题。在这篇文章中，我们将看看获得屏幕分辨率的一些方法。并非所有的解决方案都是跨平台的，但是在讨论这些方法时，我一定会提到这一点。我们开始吧！

### 使用 Linux 命令行

在 Linux 中有几种方法可以获得你的屏幕分辨率。如果你在谷歌上搜索，你会看到人们使用各种 Python GUI 工具包。我想找到一种不用安装第三方模块就能获得屏幕分辨率的方法。我最终找到了以下命令:

```py

xrandr | grep '*'

```

然后我必须把这些信息翻译成 Python。这是我想到的:

```py

import subprocess

cmd = ['xrandr']
cmd2 = ['grep', '*']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
p.stdout.close()

resolution_string, junk = p2.communicate()
resolution = resolution_string.split()[0]
width, height = resolution.split('x')

```

每当需要使用 Python 传输数据时，都需要创建两个不同的子流程实例。以上就是我做的。我通过 stdin 将来自 **xrandr** 的输出传送到我的第二个子流程。然后，我关闭了第一个进程的 stdout，基本上清除了它返回给第二个进程的所有内容。剩下的代码只是解析出监视器的宽度和高度。

### 使用 PyGTK

当然，上面的方法只适用于 Linux。如果你碰巧安装了 [PyGTK](http://www.pygtk.org/) ，那么你可以用它来获得你的屏幕分辨率。让我们来看看:

```py

import gtk

width = gtk.gdk.screen_width()
height = gtk.gdk.screen_height()

```

这非常简单，因为 PyGTK 内置了这些方法。请注意，PyGTK 适用于 Windows 和 Linux。应该还有一个正在开发中的 Mac 版本。

### 使用 wxPython

正如您所料， [wxPython 工具包](http://wxpython.org/)也提供了一种获得屏幕分辨率的方法。它的用处不大，因为您实际上需要创建一个 App 对象，然后才能获得解决方案。

```py

import wx

app = wx.App(False)
width, height = wx.GetDisplaySize()

```

这仍然是获得您想要的分辨率的简单方法。还应该注意的是，wxPython 运行在所有三个主要平台上。

### 使用 Tkinter

Tkinter 库通常包含在 Python 中，所以您应该将这个库作为默认库。它还提供屏幕分辨率，尽管它也要求您创建一个“app”对象:

```py

import Tkinter

root = Tkinter.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()

```

幸运的是，Tkinter 也可以在所有 3 个主要平台上使用，所以你几乎可以在任何地方使用这种方法。

### 使用 PySide / PyQt

正如你可能已经猜到的，你也可以使用 [PySide](https://pypi.python.org/pypi/PySide/1.2.2) 和 [PyQt](https://www.riverbankcomputing.com/software/pyqt/intro) 来获得屏幕分辨率。以下是 PySide 版本:

```py

from PySide import QtGui

app = QtGui.QApplication([])
screen_resolution = app.desktop().screenGeometry()
width, height = screen_resolution.width(), screen_resolution.height()

```

如果您使用 PyQt4，那么您需要将开头的导入更改为:

```py

from PyQt4 import QtGui

```

其余都一样。你可能知道，这两个库都可以在 Windows、Linux 和 Mac 上使用。

### 包扎

此时，您应该能够在任何操作系统上获得屏幕分辨率。还有其他方法可以获得这些信息，这些方法依赖于平台。例如，在 Windows 上，您可以使用 PyWin32 的 win32api 或 ctypes。在 Mac 上，有 AppKit。但是这里列出的工具包方法可以在大多数平台上很好地工作，因为它们是跨平台的，所以您不需要使用特殊的情况来导入特定的包来使其工作。

### 附加阅读

*   [在 Ubuntu 上用 Python 获取显示器分辨率](http://stackoverflow.com/questions/3597965/getting-monitor-resolution-in-python-on-ubuntu)
*   如何在 Python 中获得显示器分辨率？