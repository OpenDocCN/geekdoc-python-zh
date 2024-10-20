# wxPython:如何以编程方式更改 wx。笔记本页面

> 原文：<https://www.blog.pythonlibrary.org/2012/07/18/wxpython-how-to-programmatically-change-wx-notebook-pages/>

偶尔我会在 wxPython [用户组](https://groups.google.com/forum/?fromgroups#!topic/wxpython-users/uVPl73Gv9eQ)上看到有人询问如何制作 wx。笔记本以编程方式更改页面(或选项卡)。所以我决定是时候弄清楚了。下面是一些适合我的代码:

```py

import random
import wx

########################################################################
class TabPanel(wx.Panel):
    #----------------------------------------------------------------------
    def __init__(self, parent, page):
        """"""
        wx.Panel.__init__(self, parent=parent)
        self.page = page

        colors = ["red", "blue", "gray", "yellow", "green"]
        self.SetBackgroundColour(random.choice(colors))

        btn = wx.Button(self, label="Change Selection")
        btn.Bind(wx.EVT_BUTTON, self.onChangeSelection)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(btn, 0, wx.ALL, 10)
        self.SetSizer(sizer)

    #----------------------------------------------------------------------
    def onChangeSelection(self, event):
        """
        Change the page!
        """
        notebook = self.GetParent()
        notebook.SetSelection(self.page)

########################################################################
class DemoFrame(wx.Frame):
    """
    Frame that holds all other widgets
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""        
        wx.Frame.__init__(self, None, wx.ID_ANY, 
                          "Notebook Tutorial",
                          size=(600,400)
                          )
        panel = wx.Panel(self)

        notebook = wx.Notebook(panel)
        tabOne = TabPanel(notebook, 1)
        notebook.AddPage(tabOne, "Tab 1")

        tabTwo = TabPanel(notebook, 0)
        notebook.AddPage(tabTwo, "Tab 2")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(notebook, 1, wx.ALL|wx.EXPAND, 5)
        panel.SetSizer(sizer)
        self.Layout()

        self.Show()

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = DemoFrame()
    app.MainLoop()

```

需要知道的主要事情是，您需要使用 SetSelection(或 ChangeSelection)来强制 Notebook 小部件改变页面。就是这样！这段代码在 Windows 7 上用 Python 2.7.3 和 Python 2.7.3(经典)进行了测试。另请参见关于[启用](http://wxpython-users.1045709.n5.nabble.com/wxNotebook-Programatically-change-page-td2302391.html)的讨论。