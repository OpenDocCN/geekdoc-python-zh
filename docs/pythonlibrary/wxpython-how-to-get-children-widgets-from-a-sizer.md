# wxPython:如何从 Sizer 中获取子部件

> 原文：<https://www.blog.pythonlibrary.org/2012/08/24/wxpython-how-to-get-children-widgets-from-a-sizer/>

前几天，我在 StackOverflow 上偶然发现了一个问题，询问如何获得 BoxSizer 的子窗口部件。在 wxPython 中，您可能会调用 sizer 的 GetChildren()方法。但是，这将返回 SizerItems 对象的列表，而不是实际小部件本身的列表。如果你调用一个 wx，你就能看出区别。Panel 的 GetChildren()方法。现在我不会在 wxPython 用户组列表上问很多问题，但我对这个很好奇，并最终收到了来自 Cody Precord 的快速[回答](https://groups.google.com/forum/?fromgroups=#!topic/wxpython-users/d8yzkP8MPyU),[wxPython 食谱](http://www.amazon.com/gp/product/1849511780/ref=as_li_ss_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=1849511780&linkCode=as2&tag=thmovsthpy-2)和 Editra 的作者。总之，他最终给我指出了正确的方向，我想出了下面的代码:

```py

import wx

########################################################################
class MyApp(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Example")
        panel = wx.Panel(self)

        lbl = wx.StaticText(panel, label="I'm a label!")
        txt = wx.TextCtrl(panel, value="blah blah")
        btn = wx.Button(panel, label="Clear")
        btn.Bind(wx.EVT_BUTTON, self.onClear)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(lbl, 0, wx.ALL, 5)
        self.sizer.Add(txt, 0, wx.ALL, 5)
        self.sizer.Add(btn, 0, wx.ALL, 5)

        panel.SetSizer(self.sizer)

    #----------------------------------------------------------------------
    def onClear(self, event):
        """"""
        children = self.sizer.GetChildren()

        for child in children:
            widget = child.GetWindow()
            print widget
            if isinstance(widget, wx.TextCtrl):
                widget.Clear()

if __name__ == "__main__":
    app = wx.App(False)
    frame = MyApp()
    frame.Show()
    app.MainLoop()

```

重要的位在 **onClear** 方法中。这里我们需要调用 SizerItem 的 GetWindow()方法来返回实际的小部件实例。一旦我们有了它，我们就可以对小部件做一些事情，比如改变标签、值，或者在这个例子中，清除文本控件。现在您也知道如何访问 sizer 的子部件了。