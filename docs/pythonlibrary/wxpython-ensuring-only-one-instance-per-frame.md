# wxPython:确保每帧只有一个实例

> 原文：<https://www.blog.pythonlibrary.org/2015/07/15/wxpython-ensuring-only-one-instance-per-frame/>

前几天，我遇到了一个有趣的 [StackOverflow 问题](http://stackoverflow.com/q/31386570/393194),这个人试图找出如何只打开一次子框架。基本上，他想要子帧(和其他子帧)的单个实例。在谷歌上挖了一点之后，我发现了 wxPython 谷歌组的一个旧的[线程](https://groups.google.com/forum/#!topic/wxpython-users/VTPpXYZYHmM)，它有一个有趣的方法来做需要做的事情。

基本上它需要一点元编程，但这是一个有趣的小练习，我想我的读者会感兴趣的。代码如下:

```py

import wx

########################################################################
class MyPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)

########################################################################
class SingleInstanceFrame(wx.Frame):
    """"""

    instance = None
    init = 0

    #----------------------------------------------------------------------
    def __new__(self, *args, **kwargs):
        """"""
        if self.instance is None:
            self.instance = wx.Frame.__new__(self)
        elif isinstance(self.instance, wx._core._wxPyDeadObject):
            self.instance = wx.Frame.__new__(self)
        return self.instance

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        print id(self)
        if self.init:
            return
        self.init = 1

        wx.Frame.__init__(self, None, title="Single Instance Frame")
        panel = MyPanel(self)
        self.Show()

########################################################################
class MainFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Main Frame")
        panel = MyPanel(self)
        btn = wx.Button(panel, label="Open Frame")
        btn.Bind(wx.EVT_BUTTON, self.open_frame)
        self.Show()

    #----------------------------------------------------------------------
    def open_frame(self, event):
        frame = SingleInstanceFrame()

if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

```

这段代码的核心在 **SingleInstanceFrame** 类中，具体来说是在 **__new__** 方法中。这里我们检查变量 **self.instance** 是否被设置为 None。如果是这样，我们创建一个新的实例。如果用户关闭框架，我们还将创建一个新的实例，这将使它成为一个 **wxPyDeadObject** 。这就是 if 语句的第二部分的作用。它检查实例是否已被删除，如果已被删除，它将创建一个新的实例。

您还会注意到我们有一个名为 **self.init** 的变量。这用于检查实例是否已经初始化。如果是这样的话， **__init__** 只会返回，而不是重新实例化一切。

* * *

### wxPython 4 /凤凰

在 wxPython 4 / Phoenix 中，没有 **wx。_core。_wxPyDeadObject** ，所以我们必须稍微修改一下我们的代码，使它能在 wxPython 的新版本中工作。方法如下:

```py

import wx

class MyPanel(wx.Panel):
    """"""

    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)

class SingleInstanceFrame(wx.Frame):
    """"""

    instance = None
    init = 0

    def __new__(self, *args, **kwargs):
        """"""
        if self.instance is None:
            self.instance = wx.Frame.__new__(self)
        elif not self.instance:
            self.instance = wx.Frame.__new__(self)

        return self.instance

    def __init__(self):
        """Constructor"""
        print(id(self))
        if self.init:
            return
        self.init = 1

        wx.Frame.__init__(self, None, title="Single Instance Frame")
        panel = MyPanel(self)
        self.Show()

class MainFrame(wx.Frame):
    """"""

    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Main Frame")
        panel = MyPanel(self)
        btn = wx.Button(panel, label="Open Frame")
        btn.Bind(wx.EVT_BUTTON, self.open_frame)
        self.Show()

    def open_frame(self, event):
        frame = SingleInstanceFrame()

if __name__ == '__main__':
    print wx.version()
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

```

您会注意到，唯一的区别是在 **__new__** 方法中，我们稍微改变了条件语句。

我希望这篇教程对你有用。开心快乐编码！

### 相关阅读

*   [一个实例运行](http://wiki.wxpython.org/OneInstanceRunning)
*   wx。单实例检查器[文档](http://wxpython.org/Phoenix/docs/html/SingleInstanceChecker.html)