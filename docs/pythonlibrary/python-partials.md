# Python Partials

> 原文：<https://www.blog.pythonlibrary.org/2016/02/11/python-partials/>

Python 附带了一个有趣的模块，叫做 **functools** 。它的一个类是**分部**类。您可以使用它创建一个新函数，部分应用您传递给它的参数和关键字。您可以使用 partial 来“冻结”函数的一部分参数和/或关键字，从而生成一个新对象。另一种说法是，partial 用一些默认值创建了一个新函数。我们来看一个例子！

```py

>>> from functools import partial
>>> def add(x, y):
...     return x + y
... 
>>> p_add = partial(add, 2)
>>> p_add(4)
6

```

这里，我们创建一个简单的加法函数，返回其参数 x 和 y 相加的结果。接下来，我们创建一个新的 callable，方法是创建一个 partial 实例，并将我们的函数和该函数的一个参数传递给它。换句话说，我们基本上将我们的 **add** 函数的 x 参数默认为数字 2。最后，我们调用新的可调用函数， **p_add** ，参数为数字 4，结果为 6，因为 2 + 4 = 6。

片段的一个方便的用例是将参数传递给回调。让我们用 wxPython 来看看:

```py

import wx

from functools import partial 

########################################################################
class MainFrame(wx.Frame):
    """
    This app shows a group of buttons
    """

    #----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """Constructor"""
        super(MainFrame, self).__init__(parent=None, title='Partial')
        panel = wx.Panel(self)

        sizer = wx.BoxSizer(wx.VERTICAL)
        btn_labels = ['one', 'two', 'three']
        for label in btn_labels:
            btn = wx.Button(panel, label=label)
            btn.Bind(wx.EVT_BUTTON, partial(self.onButton, label=label))
            sizer.Add(btn, 0, wx.ALL, 5)

        panel.SetSizer(sizer)
        self.Show()

    #----------------------------------------------------------------------
    def onButton(self, event, label):
        """
        Event handler called when a button is pressed
        """
        print 'You pressed: ', label

if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

```

这里我们使用 partial 调用带有额外参数的 **onButton** 事件处理程序，该参数恰好是按钮的标签。这对你来说可能没什么用，但是如果你经常做 GUI 编程，你会看到很多人问如何做这种事情。当然，你也可以使用 lambda 来传递参数给回调函数。

我们在工作中使用的一个用例是我们的自动化测试框架。我们用 Python 测试了一个 UI，我们希望能够传递一个函数来关闭某些对话框。基本上你可以传递一个函数和对话框的名字来关闭对话框，但是在这个过程中的某个时刻需要调用这个函数才能正常工作。由于我不能展示这些代码，这里有一个传递部分函数的基本例子:

```py

from functools import partial

#----------------------------------------------------------------------
def add(x, y):
    """"""
    return x + y

#----------------------------------------------------------------------
def multiply(x, y):
    """"""
    return x * y

#----------------------------------------------------------------------
def run(func):
    """"""
    print func()

#----------------------------------------------------------------------
def main():
    """"""
    a1 = partial(add, 1, 2)
    m1 = partial(multiply, 5, 8)
    run(a1)
    run(m1)

if __name__ == "__main__":
    main()

```

这里，我们在主函数中创建了几个部分函数。接下来，我们将这些片段传递给我们的 **run** 函数，调用它，然后打印出被调用函数的结果。

* * *

### 包扎

至此，您应该知道如何使用 functools partial 来创建自己的“冻结”可调用程序。偏音有许多用途，但它们并不总是显而易见的。我建议您开始尝试使用它们，您可能会看到自己代码的用途。玩得开心！

* * *

### 相关阅读

*   关于 [functools](https://docs.python.org/2/library/functools.html) 的 Python 文档
*   pydanny - [Python Partials 很有趣](http://www.pydanny.com/python-partials-are-fun.html)
*   本周 Python 模块- [functools](https://pymotw.com/2/functools/)
*   wxPython Wiki - [向回调传递参数](http://wiki.wxpython.org/Passing%20Arguments%20to%20Callbacks)
*   wxPython - [从任何地方捕获异常](https://www.blog.pythonlibrary.org/2014/03/14/wxpython-catching-exceptions-from-anywhere/)