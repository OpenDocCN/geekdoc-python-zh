# wxPython:冻结和解冻

> 原文：<https://www.blog.pythonlibrary.org/2014/11/11/wxpython-freeze-and-thaw/>

wxPython 库附带了两个方便的方法，分别叫做 **Freeze()** 和 **Thaw()** 。调用 Freeze()防止窗口在冻结时更新。当您添加或删除小部件，并希望减少 UI 的闪烁时，这很有用。更新完 UI 后，调用解冻()方法，以便用户能够看到更新。

让我们看一个简单的例子。

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial", size=(500,500))
        self.btnNum = 1

        self.panel = wx.Panel(self, wx.ID_ANY)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.onClick)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.sizer)

    def onClick(self, event):
        self.Freeze()
        btn = wx.Button(self.panel, label="Button #%s" % self.btnNum)
        self.sizer.Add(btn, 0, wx.ALL, 5)
        self.sizer.Layout()
        self.Thaw()

        self.btnNum += 1

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

这里我们有一个 wx 的实例。包含面板的框架。每次左键单击面板时，我们调用框架的 Freeze()方法并添加一个按钮。然后我们解冻()它，按钮就出现了。我们跟踪有多少个按钮，这样我们可以保持按钮的标签不同。我以前在更新 ListCtrl 或 ObjectListView 小部件时使用过这些方法，在这些情况下效果很好。我相信我已经看到一些人提到它与网格小部件一起使用。你可能需要尝试一下，看看这些方法是否对你有帮助。

### 相关文章

*   wxPython: [动态添加和移除小部件](https://www.blog.pythonlibrary.org/2012/05/05/wxpython-adding-and-removing-widgets-dynamically/)
*   [无闪烁绘图](https://wiki.wxwidgets.org/Flicker-Free_Drawing)