# 在 wxPython 中跨平台获取正确的笔记本选项卡

> 原文：<https://www.blog.pythonlibrary.org/2019/06/05/getting-the-correct-notebook-tab-across-platforms-in-wxpython/>

我最近在开发一个 GUI 应用程序，它有一个`wx.Notebook in it. When the user changed tabs in the notebook, I wanted the application to do an update based on the newly shown (i.e. selected) tab. I quickly discovered that while it is easy to catch the tab change event, getting the right tab is not as obvious.`

这篇文章将带你了解我的错误，并向你展示两个解决问题的方法。

下面是我最初做的一个例子:

```py

# simple_note.py

import random
import wx

class TabPanel(wx.Panel):

    def __init__(self, parent, name):
        """"""
        super().__init__(parent=parent)
        self.name = name

        colors = ["red", "blue", "gray", "yellow", "green"]
        self.SetBackgroundColour(random.choice(colors))

        btn = wx.Button(self, label="Press Me")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(btn, 0, wx.ALL, 10)
        self.SetSizer(sizer)

class DemoFrame(wx.Frame):
    """
    Frame that holds all other widgets
    """

    def __init__(self):
        """Constructor"""
        super().__init__(None, wx.ID_ANY,
                         "Notebook Tutorial",
                         size=(600,400)
                         )
        panel = wx.Panel(self)

        self.notebook = wx.Notebook(panel)
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_tab_change)
        tabOne = TabPanel(self.notebook, name='Tab 1')
        self.notebook.AddPage(tabOne, "Tab 1")

        tabTwo = TabPanel(self.notebook, name='Tab 2')
        self.notebook.AddPage(tabTwo, "Tab 2")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.ALL|wx.EXPAND, 5)
        panel.SetSizer(sizer)
        self.Layout()

        self.Show()

    def on_tab_change(self, event):
        # Works on Windows and Linux, but not Mac
        current_page = self.notebook.GetCurrentPage()
        print(current_page.name)
        event.Skip()

if __name__ == "__main__":
    app = wx.App(False)
    frame = DemoFrame()
    app.MainLoop()

```

这段代码可以在 Linux 和 Windows 上正常运行。然而，当你在 Mac OSX 上运行它时，报告的当前页面总是你在选择当前页面之前所在的标签。这有点像一个错误，但是是在 GUI 中。

在尝试了我自己的一些想法后，我决定向 wxPython Google group 寻求帮助。

他们有两种解决方法:

*   使用`GetSelection() along with the notebook's `GetPage() method``
*   使用平板笔记本小部件

* * *

### 使用 GetSelection()

使用事件对象的`GetSelection() method will return the index of the currently selected tab. Then you can use the notebook's `GetPage() method to get the actual page. This was the suggestion that Robin Dunn, the maintainer of wxPython, gave to me.``

下面是更新后使用该修复程序的代码:

```py

# simple_note2.py

import random
import wx

class TabPanel(wx.Panel):

    def __init__(self, parent, name):
        """"""
        super().__init__(parent=parent)
        self.name = name

        colors = ["red", "blue", "gray", "yellow", "green"]
        self.SetBackgroundColour(random.choice(colors))

        btn = wx.Button(self, label="Press Me")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(btn, 0, wx.ALL, 10)
        self.SetSizer(sizer)

class DemoFrame(wx.Frame):
    """
    Frame that holds all other widgets
    """

    def __init__(self):
        """Constructor"""
        super().__init__(None, wx.ID_ANY,
                         "Notebook Tutorial",
                         size=(600,400)
                         )
        panel = wx.Panel(self)

        self.notebook = wx.Notebook(panel)
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_tab_change)
        tabOne = TabPanel(self.notebook, name='Tab 1')
        self.notebook.AddPage(tabOne, "Tab 1")

        tabTwo = TabPanel(self.notebook, name='Tab 2')
        self.notebook.AddPage(tabTwo, "Tab 2")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.ALL|wx.EXPAND, 5)
        panel.SetSizer(sizer)
        self.Layout()

        self.Show()

    def on_tab_change(self, event):
        # Works on Windows, Linux and Mac
        current_page = self.notebook.GetPage(event.GetSelection())
        print(current_page.name)
        event.Skip()

if __name__ == "__main__":
    app = wx.App(False)
    frame = DemoFrame()
    app.MainLoop()

```

这是一个相当简单的修复，但有点烦人，因为不清楚为什么需要这么做。

* * *

### 使用平板笔记本

另一个选择是换掉`wx.Notebook for the FlatNotebook. Let's see how that looks:`

```py

# simple_note.py

import random
import wx
import wx.lib.agw.flatnotebook as fnb

class TabPanel(wx.Panel):

    def __init__(self, parent, name):
        """"""
        super().__init__(parent=parent)
        self.name = name

        colors = ["red", "blue", "gray", "yellow", "green"]
        self.SetBackgroundColour(random.choice(colors))

        btn = wx.Button(self, label="Press Me")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(btn, 0, wx.ALL, 10)
        self.SetSizer(sizer)

class DemoFrame(wx.Frame):
    """
    Frame that holds all other widgets
    """

    def __init__(self):
        """Constructor"""
        super().__init__(None, wx.ID_ANY,
                         "Notebook Tutorial",
                         size=(600,400)
                         )
        panel = wx.Panel(self)

        self.notebook = fnb.FlatNotebook(panel)
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_tab_change)
        tabOne = TabPanel(self.notebook, name='Tab 1')
        self.notebook.AddPage(tabOne, "Tab 1")

        tabTwo = TabPanel(self.notebook, name='Tab 2')
        self.notebook.AddPage(tabTwo, "Tab 2")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.ALL|wx.EXPAND, 5)
        panel.SetSizer(sizer)
        self.Layout()

        self.Show()

    def on_tab_change(self, event):
        # Works on Windows, Linux and Mac
        current_page = self.notebook.GetCurrentPage()
        print(current_page.name)
        event.Skip()

if __name__ == "__main__":
    app = wx.App(False)
    frame = DemoFrame()
    app.MainLoop()

```

现在你可以回到使用笔记本的`GetCurrentPage() method. You can also use `self.notebook.GetPage(event.GetSelection()) like you do in the other workaround, but I feel like `GetCurrentPage() is just more obvious what it is that you are doing.

* * *

### 包扎

这是我在 wxPython 中被一个奇怪的陷阱抓住的少数几次之一。当您编写旨在跨多个平台运行的代码时，您会不时地遇到这类事情。检查文档以确保您没有使用并非所有平台都支持的方法总是值得的。然后你会想自己做一些研究和测试。但是一旦你做了你的尽职调查，不要害怕寻求帮助。我会一直寻求帮助，避免浪费我自己的时间，尤其是当我的解决方案在三分之二的情况下都有效的时候。