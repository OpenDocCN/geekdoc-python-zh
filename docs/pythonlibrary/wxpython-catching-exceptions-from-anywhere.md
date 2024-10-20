# wxPython:从任何地方捕捉异常

> 原文：<https://www.blog.pythonlibrary.org/2014/03/14/wxpython-catching-exceptions-from-anywhere/>

前几天，wxPython Google Group 正在讨论捕捉 wxPython 中异常的不同方法。如果您经常使用 wxPython，您将很快意识到有些异常很难捕捉。 [wxPython Wiki](http://wiki.wxpython.org/C%2B%2B%20%26%20Python%20Sandwich?action=show&redirect=CppAndPythonSandwich) 解释了原因。无论如何，名单上的人都推荐使用 **sys.excepthook** 。所以我采用了他们提到的一种方法，创建了一个小例子:

```py

import sys
import traceback
import wx

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

在这个例子中，我们创建了一个带有按钮的面板，这个按钮会故意引发一个异常。我们通过将 **sys.excepthook** 重定向到我们的 **MyExceptionHook** 函数来捕获异常。这个函数将格式化异常的回溯，格式化它使它可读，然后显示一个带有异常信息的对话框。wxPython 的创建者 Robin Dunn 认为，如果有人提出一个装饰器，我们可以用它来捕捉异常，然后作为一个例子添加到 [wiki 页面](http://wiki.wxpython.org/C%2B%2B%20%26%20Python%20Sandwich?action=show&redirect=CppAndPythonSandwich)，这将是一件好事。

我对装潢师的第一个想法如下:

```py

import logging
import wx

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
    def __call__(self, evt):
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

在这段代码中，我们创建了一个创建日志实例的类。然后我们覆盖 **__call__** 方法，将方法调用包装在异常处理程序中，这样我们就可以捕捉异常。基本上我们在这里做的是创建一个类装饰器。接下来，我们用异常日志记录类修饰一个事件处理程序。这并不完全是邓恩先生想要的，因为装饰者也需要能够包装其他功能。所以我编辑了一下，做了如下的小调整:

```py

import logging
import wx

class ExceptionLogging(object):
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn

        # create logging instance
        self.log = logging.getLogger("wxErrors")
        self.log.setLevel(logging.INFO)

        # create a logging file handler / formatter
        log_fh = logging.FileHandler("error.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        log_fh.setFormatter(formatter)
        self.log.addHandler(log_fh)

    def __call__(self, *args, **kwargs):
        try:
            self.fn(self, *args, **kwargs)
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

这一次， **__call__** 方法可以接受任意数量的参数或关键字参数，这给了它更多的灵活性。这仍然不是罗宾·邓恩想要的，所以他写了下面的例子:

```py

from __future__ import print_function

import logging
import wx

print(wx.version())

def exceptionLogger(func, loggerName=''):
    """
    A simple decorator that will catch and log any exceptions that may occur
    to the root logger.
    """
    assert callable(func)
    mylogger = logging.getLogger(loggerName)

    # wrap a new function around the callable
    def logger_func(*args, **kw):
        try:
            if not kw:
                return func(*args)
            return func(*args, **kw)
        except Exception:
            mylogger.exception('Exception in %s:', func.__name__)

    logger_func.__name__ = func.__name__
    logger_func.__doc__ = func.__doc__
    if hasattr(func, '__dict__'):
        logger_func.__dict__.update(func.__dict__)
    return logger_func    

def exceptionLog2Logger(loggerName):
    """
    A decorator that will catch and log any exceptions that may occur
    to the named logger.
    """
    import functools
    return functools.partial(exceptionLogger, loggerName=loggerName)    

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

    @exceptionLog2Logger('testLogger')
    def onExcept(self, event):
        """
        Raise an error
        """
        print(self, event)
        print(isinstance(self, wx.Panel))

        #trigger an exception
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

    # set up the default logger
    log = logging.getLogger('testLogger')
    log.setLevel(logging.INFO)

    # create a logging file handler / formatter
    log_fh = logging.FileHandler("error.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    log_fh.setFormatter(formatter)
    log.addHandler(log_fh)

    app = wx.App(False)
    frame = Frame()
    app.MainLoop()

```

这展示了几个不同的装饰器示例。这个例子展示了更传统的装饰器构造方法。虽然它有更多的元编程。第一个示例检查以确保传递给它的内容实际上是可调用的。然后，它创建一个记录器，并用一个异常处理程序包装可调用的。在它返回包装的函数之前，包装的函数被修改，以便它与传递给它的原始函数具有相同的名称和 docstring。我相信你可以放弃它，使用 [functools.wraps](http://docs.python.org/2/library/functools.html#functools.wraps) 来代替，但是在教程中明确一点可能更好。

* * *

### 包扎

现在你知道如何用几种不同的方法捕捉异常了。希望这对您自己的应用程序设计有所帮助。玩得开心！