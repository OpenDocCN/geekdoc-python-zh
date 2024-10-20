# wxPython:在标题栏中嵌入图像

> 原文：<https://www.blog.pythonlibrary.org/2008/05/23/wxpython-embedding-an-image-in-your-title-bar/>

我收到了一个关于如何在 Windows 上的框架工具栏中放置图像的问题。实际上，工具栏只是使用了一个通用的图标。我知道有三种方式。首先是从可执行文件中获取嵌入的图像。第二种方法是把你有的一些图片嵌入其中。最后一种方法是将您的图像转换成可以导入的 Python 文件。我敢肯定你也可以和 PIL 乱搞，或者甚至使用油漆处理器，但我不知道这些东西。

我先说说从可执行文件中获取嵌入图像。这其实很简单。基本想法是这样的:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, 'Image Extractor') 

        # Add a panel so it looks the correct on all platforms
        self.panel = wx.Panel(self, wx.ID_ANY)

        loc = wx.IconLocation(r'L:\Python25\python.exe', 0)
        self.SetIcon(wx.IconFromLocation(loc))

# Run the program
if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

在本例中，我使用以下代码行从 python.exe 中抓取 Python 2.5 图标:

```py

loc = wx.IconLocation(r'L:\Python25\python.exe', 0)

```

然后，我使用 SetIcon()设置框架的图标。注意，我也需要使用 wx。IconFromLocation(loc)，我将它嵌套在 SetIcon()调用中。

接下来，我将讨论如何使用你手头的任何图像。这段代码和上面代码的唯一区别是，我去掉了对 wx 的调用。图标位置和 wx。IconFromLocation 并添加了一个 wx。图标对象。wx。图标对象只需要一个指向图标和 wx 的路径。位图类型图标标志。参见下面的完整代码:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, 'Image Extractor')

        # Add a panel so it looks the correct on all platforms
        self.panel = wx.Panel(self, wx.ID_ANY)

        ico = wx.Icon('py.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(ico)

# Run the program
if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

我最终会采取的方式可能是最好的。在其中，我使用 wx 的 img2py 实用程序将一个图标或图像转换成 python 文件。为什么这可能是最好的？因为通过在 python 文件中嵌入图像文件，可以简化用 py2exe 发布应用程序的过程。至少，这是我的经历。

在我的电脑上，可以在以下位置找到 img2py 实用程序:

c:\ python 25 \ Lib \ site-packages \ wx-2.8-MSW-unicode \ wx \ tools >

根据您的设置需要进行调整。打开命令窗口并导航到此目录。然后键入以下命令:

python img 2 py . py-I path/to/your/icon . ico myicon . py

img2py.py 的第一个参数是-i，它告诉实用程序您正在嵌入一个图标。接下来是图标文件的路径。最后，给出希望 img2py 创建的文件的名称(即，将图标嵌入其中)。现在，将您刚刚创建的 python 文件复制到包含您的 wxPython 脚本的文件夹中，以便它可以导入它(或者您可以将代码从 Python 文件复制到您正在创建的应用程序的文本中)。

我将为这个示例导入它。要获得图标，您需要调用我导入的图标文件的 getIcon()方法。查看代码，看看我在做什么:

```py

import wx
import myIcon

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, 'Image Extractor')

        # Add a panel so it looks the correct on all platforms
        self.panel = wx.Panel(self, wx.ID_ANY)

        ico = myIcon.getIcon()
        self.SetIcon(ico)

# Run the program
if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

希望这篇教程已经帮助你学会了如何在你的应用程序中使用你的图标。请记住，您可以对任何想要插入的图像使用这些技术；不仅仅是标题栏图标，还包括应用程序中使用的任何静态图像，比如任务栏图标或工具栏图标。祝你好运！

**延伸阅读:**
[wxPython Wiki -闪烁任务栏图标](http://wiki.wxpython.org/index.cgi/FlashingTaskbarIcon)

**下载量:**

[embedded-icon-code . zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2008/05/wx-tutorials.zip)【embedded-icon-code.tar】
T3