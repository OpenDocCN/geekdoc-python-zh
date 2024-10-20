# wxPython:如何触发多个事件处理程序

> 原文：<https://www.blog.pythonlibrary.org/2012/07/24/wxpython-how-to-fire-multiple-event-handlers/>

今天在 [StackOverflow](http://stackoverflow.com/q/11621833/393194) 上，我看到有人想知道如何在 wxPython 中将两个函数/方法绑定到同一个事件。这真的很容易。这里有一个例子:

```py

import wx

########################################################################
class MyPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        btn = wx.Button(self, label="Press Me")
        btn.Bind(wx.EVT_BUTTON, self.HandlerOne)
        btn.Bind(wx.EVT_BUTTON, self.HandlerTwo)

    #----------------------------------------------------------------------
    def HandlerOne(self, event):
        """"""
        print "handler one fired!"
        event.Skip()

    #----------------------------------------------------------------------
    def HandlerTwo(self, event):
        """"""
        print "handler two fired!"
        event.Skip()

########################################################################
class MyFrame(wx.Frame):
    """."""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Test")
        panel = MyPanel(self)
        self.Show()

if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

```

正如您所看到的，您所要做的就是调用小部件的 Bind 方法两次，并向它传递相同的事件，但传递不同的处理程序。下一个关键是你必须使用**事件。Skip()** 。Skip 将导致 wxPython 寻找可能需要处理该事件的其他处理程序。事件沿着层次结构向上传递到父级，直到它们被处理或什么都没发生。罗宾·邓恩的《Wxpython in Action[Wxpython " target = " _ blank ">Wxpython in Action](http://www.amazon.com/gp/product/1932394621/ref=as_li_ss_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=1932394621&linkCode=as2&tag=thmovsthpy-20)一书很好地解释了这个概念。