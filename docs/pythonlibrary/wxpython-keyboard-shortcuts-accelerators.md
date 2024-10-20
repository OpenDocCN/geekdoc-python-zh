# wxPython:键盘快捷键(加速器)

> 原文：<https://www.blog.pythonlibrary.org/2010/12/02/wxpython-keyboard-shortcuts-accelerators/>

几乎所有电脑超级用户都想使用键盘快捷键(又名:加速器)来完成工作。对我们来说幸运的是，wxPython 提供了一种通过 wx 使用加速器表非常容易地实现这一点的方法。可加速的类。在本文中，我们将通过几个例子来了解这是如何实现的。

## 入门指南

对于我们的第一个技巧，我们将从一个非常简单的例子开始。查看下面的代码！

```py

import wx

class MyForm(wx.Frame):

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial", size=(500,500))

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)

        randomId = wx.NewId()
        self.Bind(wx.EVT_MENU, self.onKeyCombo, id=randomId)
        accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL,  ord('Q'), randomId )])
        self.SetAcceleratorTable(accel_tbl)

    #----------------------------------------------------------------------
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

上面的代码只是一个带有 **wx 的面板。AcceleratorTable** 对象，其中包含一个快捷键组合，即 CTRL+Q，wxPython 中的快捷键实际上是菜单事件(即 wx。EVT _ 菜单)，可能是因为快捷键通常也是一个菜单项。因此，他们依赖于某种身份。如果我们有一个菜单项，我们想给一个快捷方式，我们将使用菜单项的 id。在这里，我们只是创建一个新的 id。注意，我们必须使用 **wx。ACCEL_CTRL** 来“捕捉”CTRL 键的按下。最后，在创建 wx 之后。AcceleratorTable，我们需要将它添加到 wx 中。调用 Frame 对象的**setAcceleratorTable**方法并传入 acceleratotable 实例。唷！你都明白了吗？很好！那我们继续吧。

## 了解多种快捷方式及更多

在这个例子中，我们将学习如何添加多个键盘快捷键，我们还将学习如何创建一个多键快捷键。让我们来看看:

```py

import wx

class MyForm(wx.Frame):

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial")

        panel = wx.Panel(self, wx.ID_ANY)

        # Create a menu
        menuBar = wx.MenuBar()
        fileMenu = wx.Menu()

        refreshMenuItem = fileMenu.Append(wx.NewId(), "Refresh",
                                          "Refresh app")
        self.Bind(wx.EVT_MENU, self.onRefresh, refreshMenuItem)

        exitMenuItem = fileMenu.Append(wx.NewId(), "E&xit\tCtrl+X", "Exit the program")
        self.Bind(wx.EVT_MENU, self.onExit, exitMenuItem)

        menuBar.Append(fileMenu, "File")
        self.SetMenuBar(menuBar)

        # Create an accelerator table
        xit_id = wx.NewId()
        yit_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.onAltX, id=xit_id)
        self.Bind(wx.EVT_MENU, self.onShiftAltY, id=yit_id)

        self.accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL, ord('R'), refreshMenuItem.GetId()),
                                              (wx.ACCEL_ALT, ord('X'), xit_id),
                                              (wx.ACCEL_SHIFT|wx.ACCEL_ALT, ord('Y'), yit_id)
                                             ])
        self.SetAcceleratorTable(self.accel_tbl)

    #----------------------------------------------------------------------
    def onRefresh(self, event):
        print "refreshed!"

    #----------------------------------------------------------------------
    def onAltX(self, event):
        """"""
        print "You pressed ALT+X!"

    #----------------------------------------------------------------------
    def onShiftAltY(self, event):
        """"""
        print "You pressed SHIFT+ALT+Y!"

    #----------------------------------------------------------------------
    def onExit(self, event):
        """"""
        self.Close()

#----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

在这个例子中，我们创建了一个只有一个菜单项的超级简单的菜单。正如您所看到的，当我们在加速器表中创建快捷方式时，我们使用了刷新菜单项的 id。我们还将菜单项本身绑定到同一个事件处理程序。加速器表中的其他两个项目以与之前相同的方式绑定。但是，请注意，表中的最后一项是两个标志:wx。ACCEL_SHIFT 和 wx。ACCEL ALT。那是什么意思？这意味着我们必须按 SHIFT+ALT+SomeKey 来触发事件。在这种情况下，我们想要的键是“Y”。

如果你密切注意，你会注意到有一个退出菜单项，声称如果你按下 CTRL+X，应用程序将关闭。但是，我们的加速器表中没有这种快捷方式。幸运的是，wxPython 会自动执行这个快捷方式，因为它被添加到了菜单项中:“\tCtrl+X”。“\t”是一个制表符(以防您不知道)，正因为如此，它告诉 wxPython 解析它后面的内容，并将其添加到一个加速表中。这不是很棒吗？

*注意:我无法确认 wxPython 是否会将该快捷方式添加到一个单独的快捷键表中，或者是否会将其添加到程序员创建的快捷键表中，但是这两种方式都无关紧要，因为它“只是工作”。*

## 包扎

到目前为止，您应该知道如何在 wxPython 中创建键盘快捷键。恭喜你！您就离创建一个很酷的应用程序更近了！我经常使用键盘快捷键，如果一个应用程序有好的、直观的、容易记住的快捷键，我会非常感激。如果你也是一个键盘迷，那么你会明白我的意思。玩得开心！

## 进一步阅读

*   [wxPython:使用菜单、工具栏和加速器](https://www.blog.pythonlibrary.org/2008/07/02/wxpython-working-with-menus-toolbars-and-accelerators/)
*   wxPython: [捕捉键和字符事件](https://www.blog.pythonlibrary.org/2009/08/29/wxpython-catching-key-and-char-events/)
*   wx。加速器表[文档](http://www.wxpython.org/docs/api/wx.AcceleratorTable-class.html)
*   wx。关键事件[文档](http://www.wxpython.org/docs/api/wx.KeyEvent-class.html)

*注意:这段代码在 Windows 上用 Python 2.5.4 和 wxPython 2.8.10.1 进行了测试*