# wxPython 的上下文管理器

> 原文：<https://www.blog.pythonlibrary.org/2015/10/22/wxpythons-context-managers/>

几年前，wxPython 工具包在其代码库中添加了上下文管理器，但是由于某种原因，您并没有看到很多使用它们的例子。在本文中，我们将看看 wxPython 中上下文管理器的三个例子。一个 wxPython 用户是第一个建议在 wxPython 的邮件列表中使用上下文管理器的人。我们将从使用我们自己的上下文管理器开始，然后看几个 wxPython 中内置上下文管理器的例子。

* * *

### 创建自己的 wxPython 上下文管理器

在 wxPython 中创建自己的上下文管理器非常容易。我们将使用 **wx。FileDialog** 是我们的上下文管理器的例子。

```py

import os
import wx

########################################################################
class ContextFileDialog(wx.FileDialog):
    """"""

    #----------------------------------------------------------------------
    def __enter__(self):
        """"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Destroy()

########################################################################
class MyPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        btn = wx.Button(self, label='Open File')
        btn.Bind(wx.EVT_BUTTON, self.onOpenFile)

    #----------------------------------------------------------------------
    def onOpenFile(self, event):
        """"""
        wildcard = "Python source (*.py)|*.py|" \
            "All files (*.*)|*.*"
        kwargs = {'message':"Choose a file",
                  'defaultDir':os.path.dirname(os.path.abspath( __file__ )), 
                  'defaultFile':"",
                  'wildcard':wildcard,
                  'style':wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
                  }
        with ContextFileDialog(self, **kwargs) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                paths = dlg.GetPaths()
                print "You chose the following file(s):"
                for path in paths:
                    print path

########################################################################
class MyFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title='wxPython Contexts')
        panel = MyPanel(self)
        self.Show()

if __name__ == '__main__':
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

```

在这个例子中，我们子类化了 **wx。FileDialog** 和我们所做的就是覆盖 **__enter__** 和 **__exit__** 方法。当我们使用 python**和**语句调用 FileDialog 实例时，它将变成一个上下文管理器。您可以在 **MyPanel** 类中的 **onOpenFile** 事件处理程序中看到这一点。现在让我们继续看一些 wxPython 的内置例子！

* * *

### wxPython 的上下文管理器

wxPython 包支持任何子类化 **wx 的上下文管理器。对话框**以及以下小部件:

*   wx.BusyInfo
*   wx。忙碌光标
*   wx。windows 禁用程序
*   wx.LogNull
*   wx。DCTextColourChanger
*   wx。DCPenChanger
*   wx(地名)。DCBrushChanger
*   wx。DCClipper
*   wx。定格/ wx。融雪

可能还有更多小部件，但这是我在撰写本文时唯一能找到的清单。让我们看几个例子:

```py

import time
import wx

########################################################################
class MyPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.frame = parent

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        dlg_btn = wx.Button(self, label='Open ColorDialog')
        dlg_btn.Bind(wx.EVT_BUTTON, self.onOpenColorDialog)
        main_sizer.Add(dlg_btn, 0, wx.ALL|wx.CENTER)

        busy_btn = wx.Button(self, label='Open BusyInfo')
        busy_btn.Bind(wx.EVT_BUTTON, self.onOpenBusyInfo)
        main_sizer.Add(busy_btn,0, wx.ALL|wx.CENTER)

        self.SetSizer(main_sizer)

    #----------------------------------------------------------------------
    def onOpenColorDialog(self, event):
        """
        Creates and opens the wx.ColourDialog
        """
        with wx.ColourDialog(self) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                data = dlg.GetColourData()
                color = str(data.GetColour().Get())
                print 'You selected: %s\n' % color

    #----------------------------------------------------------------------
    def onOpenBusyInfo(self, event):
        """
        Creates and opens an instance of BusyInfo
        """
        msg = 'This app is busy right now!'
        self.frame.Hide()
        with wx.BusyInfo(msg) as busy:
            time.sleep(5)
        self.frame.Show()

########################################################################
class MyFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title='Context Managers')
        panel = MyPanel(self)

        self.Show()

#----------------------------------------------------------------------
if __name__ == '__main__':
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

```

在上面的代码中，我们有两个 wxPython 的上下文管理器的例子。第一个是在 **onOpenColorDialog** 事件处理程序中。这里我们创建一个 **wx 的实例。ColourDialog** 然后如果用户按下 OK 按钮，抓取所选颜色。第二个例子稍微复杂一点，因为它在显示 BusyInfo 实例之前隐藏了框架。坦率地说，我认为这个例子可以通过将框架的隐藏和显示放入上下文管理器本身来改进，但是我将把它作为一个练习留给读者去尝试。

* * *

### 包扎

wxPython 的上下文管理器非常方便，使用起来也很有趣。我希望你很快会在自己的代码中使用它们。请务必尝试 wxPython 中的其他一些上下文管理器，看看它们是否适合您的代码库，或者只是让您的代码更整洁一些。