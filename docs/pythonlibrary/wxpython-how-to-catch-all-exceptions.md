# wxPython:如何捕捉所有异常

> 原文：<https://www.blog.pythonlibrary.org/2014/01/31/wxpython-how-to-catch-all-exceptions/>

我在 wxPython Google 组的一个朋友[问](https://groups.google.com/forum/#!topic/wxpython-users/IXFsH9vKQMQ)如何捕捉 wxPython 中发生的任何异常。这个问题有点复杂，因为 wxPython 是 C++库(wxWidgets)上的一个包装器。你可以在 [wxPython wiki](http://wiki.wxpython.org/C%2B%2B%20%26%20Python%20Sandwich?action=show&redirect=CppAndPythonSandwich) 上了解这个问题。几个 wxPython 用户提到使用 Python 的 **sys.excepthook** 来捕捉错误。因此，我决定根据 Andrea Gavana 在前面提到的帖子中发布的内容，写一个例子来说明这是如何工作的。我们还将看看 wiki 链接中的解决方案。

* * *

### 用 sys.excepthook 捕获所有错误

这比我预期的要多一点，因为我最终需要导入 Python 的 **traceback** 模块，我决定显示错误，所以我也创建了一个对话框。让我们看一下代码:

```py

import sys
import traceback
import wx
import wx.lib.agw.genericmessagedialog as GMD

########################################################################
class Panel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        btn = wx.Button(self, label="Raise Exception")
        btn.Bind(wx.EVT_BUTTON, self.onExcept)

    #----------------------------------------------------------------------
    def onExcept(self, event):
        """
        Raise an error
        """
        1/0

########################################################################
class Frame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Exceptions")
        sys.excepthook = MyExceptionHook
        panel = Panel(self)
        self.Show()

########################################################################
class ExceptionDialog(GMD.GenericMessageDialog):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, msg):
        """Constructor"""
        GMD.GenericMessageDialog.__init__(self, None, msg, "Exception!",
                                          wx.OK|wx.ICON_ERROR)

#----------------------------------------------------------------------
def MyExceptionHook(etype, value, trace):
    """
    Handler for all unhandled exceptions.

    :param `etype`: the exception type (`SyntaxError`, `ZeroDivisionError`, etc...);
    :type `etype`: `Exception`
    :param string `value`: the exception error message;
    :param string `trace`: the traceback header, if any (otherwise, it prints the
     standard Python header: ``Traceback (most recent call last)``.
    """
    frame = wx.GetApp().GetTopWindow()
    tmp = traceback.format_exception(etype, value, trace)
    exception = "".join(tmp)

    dlg = ExceptionDialog(exception)
    dlg.ShowModal()
    dlg.Destroy()    

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = Frame()
    app.MainLoop()

```

这段代码有点复杂，所以我们将在每一部分花点时间。面板代码上有一个按钮，该按钮将调用一个方法，该方法将导致一个**zerodisvisionerror**。在 Frame 类中，我们将 **sys.excepthook** 设置为自定义函数 **MyExceptionHook** 。让我们来看看这个:

```py

#----------------------------------------------------------------------
def MyExceptionHook(etype, value, trace):
    """
    Handler for all unhandled exceptions.

    :param `etype`: the exception type (`SyntaxError`, `ZeroDivisionError`, etc...);
    :type `etype`: `Exception`
    :param string `value`: the exception error message;
    :param string `trace`: the traceback header, if any (otherwise, it prints the
     standard Python header: ``Traceback (most recent call last)``.
    """
    frame = wx.GetApp().GetTopWindow()
    tmp = traceback.format_exception(etype, value, trace)
    exception = "".join(tmp)

    dlg = ExceptionDialog(exception)
    dlg.ShowModal()
    dlg.Destroy()

```

这个函数接受 3 个参数:etype、value 和 traceback。我们使用 **traceback** 模块将这些片段放在一起，得到一个完整的回溯，我们可以将它传递给我们的消息对话框。

* * *

### 使用原始的错误捕获方法

robin Dunn(wxPython 的创建者)提到在上面的同一个线程中有一个关于 [wiki](http://wiki.wxpython.org/C%2B%2B%20%26%20Python%20Sandwich?action=show&redirect=CppAndPythonSandwich) 的解决方案，他希望看到它被用作装饰器。以下是我的实现:

```py

import logging
import wx
import wx.lib.agw.genericmessagedialog as GMD

########################################################################
class ExceptionLogging(object):

    #----------------------------------------------------------------------
    def __init__(self, fn):
        self.fn = fn

        # create logging instance
        self.log = logging.getLogger("wxErrors")
        self.log.setLevel(logging.INFO)

        # create a logging file handler / formatter
        log_fh = logging.FileHandler("error.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        log_fh.setFormatter(formatter)
        self.log.addHandler(log_fh)

    #----------------------------------------------------------------------
    def __call__(self,evt):
        try:
            self.fn(self, evt)
        except Exception, e:
            self.log.exception("Exception")

########################################################################
class Panel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        btn = wx.Button(self, label="Raise Exception")
        btn.Bind(wx.EVT_BUTTON, self.onExcept)

    #----------------------------------------------------------------------
    @ExceptionLogging
    def onExcept(self, event):
        """
        Raise an error
        """
        1/0

########################################################################
class Frame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Exceptions")
        panel = Panel(self)
        self.Show()

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = Frame()
    app.MainLoop()

```

我们使用自定义异常类来记录错误。为了将该类应用于我们的事件处理程序，我们使用@classname 用该类来修饰它们，在本例中该类被翻译成 **@ExceptionLogging** 。因此，无论何时调用这个事件处理程序，它都会通过装饰器运行，装饰器将事件处理程序包装在 try/except 中，并将所有异常记录到磁盘中。我不完全确定本文中提到的两种方法是否能捕捉到相同的错误。欢迎在评论中告诉我。

### 相关信息

*   PyQt 邮件列表上的[捕获错误线程](http://www.riverbankcomputing.com/pipermail/pyqt/2009-May/022961.html)