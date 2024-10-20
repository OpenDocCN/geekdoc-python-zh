# wxPython:关于加速器

> 原文：<https://www.blog.pythonlibrary.org/2017/09/28/wxpython-all-about-accelerators/>

wxPython 工具包通过加速器和加速器表的概念支持使用键盘快捷键。您也可以直接绑定到按键，但在很多情况下，您会希望使用加速器。加速器提供了向应用程序添加键盘快捷键的能力，比如大多数应用程序用来保存文件的无处不在的“CTRL+S”。只要您的应用程序有焦点，就可以轻松地添加这个键盘快捷键。

请注意，您通常会将一个加速表添加到您的 **wx 中。框架**实例。如果您的应用程序中碰巧有多个帧，那么您可能需要根据您的设计向多个帧添加一个加速器表。

我们来看一个简单的例子:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title="Accelerator Tutorial", 
                          size=(500,500))

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)

        randomId = wx.NewId()
        self.Bind(wx.EVT_MENU, self.onKeyCombo, id=randomId)
        accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL,  ord('Q'), 
                                          randomId )])
        self.SetAcceleratorTable(accel_tbl)

    def onKeyCombo(self, event):
        """"""
        print "You pressed CTRL+Q!"

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

如果你有很多键盘快捷键需要添加到你的应用程序中，这可能看起来有点难看，因为你最终会得到一个看起来有点奇怪的元组列表。你会发现这种方式或者写一个**加速表**更常见。然而，还有其他方法可以添加条目到你的**加速表**。让我们来看看 wxPython 的[文档](https://docs.wxpython.org/wx.AcceleratorTable.html)中的一个例子:

```py

entries = [wx.AcceleratorEntry() for i in xrange(4)]

entries[0].Set(wx.ACCEL_CTRL, ord('N'), ID_NEW_WINDOW)
entries[1].Set(wx.ACCEL_CTRL, ord('X'), wx.ID_EXIT)
entries[2].Set(wx.ACCEL_SHIFT, ord('A'), ID_ABOUT)
entries[3].Set(wx.ACCEL_NORMAL, wx.WXK_DELETE, wx.ID_CUT)

accel = wx.AcceleratorTable(entries)
frame.SetAcceleratorTable(accel)

```

这里我们创建了一个包含四个 wx 的列表。AcceleratorEntry() 使用列表理解对象。然后我们使用 Python 列表的索引来访问列表中的每个条目，以调用每个条目的 **Set** 方法。代码的其余部分与您之前看到的非常相似。让我们花点时间让这段代码实际上可以运行:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title="AcceleratorEntry Tutorial", 
                          size=(500,500))

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)

        exit_menu_item = wx.MenuItem(id=wx.NewId(), text="Exit",
                               helpString="Exit the application")
        about_menu_item = wx.MenuItem(id=wx.NewId(), text='About')

        ID_NEW_WINDOW = wx.NewId()
        ID_ABOUT = wx.NewId()

        self.Bind(wx.EVT_MENU, self.on_new_window, id=ID_NEW_WINDOW)
        self.Bind(wx.EVT_MENU, self.on_about, id=ID_ABOUT)

        entries = [wx.AcceleratorEntry() for i in range(4)]

        entries[0].Set(wx.ACCEL_CTRL, ord('N'),
                       ID_NEW_WINDOW, exit_menu_item)
        entries[1].Set(wx.ACCEL_CTRL, ord('X'), wx.ID_EXIT)
        entries[2].Set(wx.ACCEL_SHIFT, ord('A'), ID_ABOUT, 
                       about_menu_item)
        entries[3].Set(wx.ACCEL_NORMAL, wx.WXK_DELETE, wx.ID_CUT)

        accel_tbl = wx.AcceleratorTable(entries)
        self.SetAcceleratorTable(accel_tbl)

    def on_new_window(self, event):
        """"""
        print("You pressed CTRL+N!")

    def on_about(self, event):
        print('You pressed SHIFT+A')

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

首先，我要指出的是，我没有把所有的加速器都连接上。例如，“CTRL+X”实际上不会退出程序。但是我确实把“CTRL+N”和“SHIFT+A”连接起来了。尝试运行代码，看看它是如何工作的。

您还可以稍微明确一点，逐个创建 AcceleratorEntry()对象，而不是使用列表理解。让我们稍微修改一下代码，看看它是如何工作的:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, 
                          title="AcceleratorEntry Tutorial", 
                          size=(500,500))

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)

        exit_menu_item = wx.MenuItem(id=wx.NewId(), text="Exit",
                               helpString="Exit the application")
        about_menu_item = wx.MenuItem(id=wx.NewId(), text='About')

        ID_NEW_WINDOW = wx.NewId()
        ID_ABOUT = wx.NewId()

        self.Bind(wx.EVT_MENU, self.on_new_window, id=ID_NEW_WINDOW)
        self.Bind(wx.EVT_MENU, self.on_about, id=ID_ABOUT)

        entry_one = wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('N'),
                                        ID_NEW_WINDOW, 
                                        exit_menu_item)
        entry_two = wx.AcceleratorEntry(wx.ACCEL_SHIFT, ord('A'), 
                                        ID_ABOUT, 
                                        about_menu_item)
        entries = [entry_one, entry_two]

        accel_tbl = wx.AcceleratorTable(entries)
        self.SetAcceleratorTable(accel_tbl)

    def on_new_window(self, event):
        """"""
        print("You pressed CTRL+N!")

    def on_about(self, event):
        print('You pressed SHIFT+A')

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

坦白地说，我认为这个版本是最好的，因为它是最明确的。“Python 的禅”总是提倡显式而非隐式地做事，所以我认为这也很好地遵循了这一范式。

* * *

### 包扎

现在，您已经知道了为您的应用程序创建键盘快捷键(加速器)的几种不同方法。它们非常方便，可以增强应用程序的有用性。

* * *

### 相关阅读

*   wxPython: [键盘快捷键](https://www.blog.pythonlibrary.org/2010/12/02/wxpython-keyboard-shortcuts-accelerators/)
*   wxPython: [菜单、工具栏和加速器](https://www.blog.pythonlibrary.org/2008/07/02/wxpython-working-with-menus-toolbars-and-accelerators/)
*   关于 [wx 的 wxPython 文档。加速表](https://docs.wxpython.org/wx.AcceleratorTable.html)
*   关于 [wx 的 wxPython 文档。加速器入口](https://docs.wxpython.org/wx.AcceleratorEntry.html)