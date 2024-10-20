# wxPython:捕获键和字符事件

> 原文：<https://www.blog.pythonlibrary.org/2009/08/29/wxpython-catching-key-and-char-events/>

在这篇文章中，我将详细说明如何捕捉特定的按键，以及为什么这很有用。这是我的“要求”系列教程中的另一个。捕捉按键真的没什么大不了的，但是当一个小部件的行为与另一个小部件稍有不同时，可能会有点混乱。当你需要捕捉 EVT _ 查尔时，真正复杂的东西就来了。

首先我将报道关键事件，wx。EVT 向下键和 wx。EVT 向上，然后我会去看看 wx 的复杂性。EVT 夏尔。我认为，如果您看到一些示例代码，编程是最容易理解的，所以我将从一个简单的示例开始:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Key Press Tutorial")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        btn = wx.Button(panel, label="OK")

        btn.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)

    def onKeyPress(self, event):
        keycode = event.GetKeyCode()
        print keycode
        if keycode == wx.WXK_SPACE:
            print "you pressed the spacebar!"
        event.Skip()

# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

您会注意到这段代码中唯一重要的小部件是一个面板和一个按钮。我将按钮绑定到 EVT 键，并在处理程序中检查用户是否按下了空格键。该事件仅在按钮有焦点时触发。你会注意到我也称之为“事件”。跳过结尾的。如果你不调用 Skip，那么这个键将被“吃掉”,并且不会有相应的 char 事件。这在按钮上并不重要，但是在文本控件中可能会很重要，因为 char 事件是捕捉大小写、重音、元音变音等的正确方式。

在我的一个电子表格类型的应用程序中，我使用了类似的方法来捕捉箭头键。我希望能够检测到这些键，这样，如果我正在编辑一个单元格，按下箭头键将使选择更改为不同的单元格。这不是默认行为。在网格中，每个单元格都有自己的编辑器，按箭头键只是在单元格内移动光标。

只是为了好玩，我创建了一个与上面类似的例子，其中我绑定了 key 和 key down 事件，但是使用了两个不同的小部件。请查看以下内容:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Key Press Tutorial 2")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        sizer = wx.BoxSizer(wx.VERTICAL)

        btn = self.onWidgetSetup(wx.Button(panel, label="OK"), 
                                 wx.EVT_KEY_UP,
                                 self.onButtonKeyEvent, sizer)
        txt = self.onWidgetSetup(wx.TextCtrl(panel, value=""),
                                 wx.EVT_KEY_DOWN, self.onTextKeyEvent,
                                 sizer)
        panel.SetSizer(sizer)

    def onWidgetSetup(self, widget, event, handler, sizer):
        widget.Bind(event, handler)
        sizer.Add(widget, 0, wx.ALL, 5)
        return widget

    def onButtonKeyEvent(self, event):
        keycode = event.GetKeyCode()
        print keycode
        if keycode == wx.WXK_SPACE:
            print "you pressed the spacebar!"
        event.Skip()

    def onTextKeyEvent(self, event):
        keycode = event.GetKeyCode()
        print keycode
        if keycode == wx.WXK_DELETE:
            print "you pressed the delete key!"
        event.Skip()

# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

诚然，这主要是为了说明。主要要知道的是，你真的不用 EVT_KEY_UP，除非你需要跟踪多按键组合，比如 CTRL+K+Y 之类的(关于半相关的注意事项，参见 wx。可加速)。虽然在我的例子中我没有这样做，但是需要注意的是，如果您正在检查 CTRL 键，那么最好使用 event。CmdDown()而不是 event.ControlDown .原因是 CmdDown 在 Windows 和 Linux 上相当于 ControlDown，但在 Mac 上它模拟 Command 键。因此，CmdDown 是跨平台检查 CTRL 键是否被按下的最佳方式。

这就是你需要知道的关于关键事件的全部内容。让我们继续，看看我们能从 char 事件中学到什么。这里有一个简单的例子:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Char Event Tutorial")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        btn = wx.TextCtrl(panel, value="")

        btn.Bind(wx.EVT_CHAR, self.onCharEvent)

    def onCharEvent(self, event):
        keycode = event.GetKeyCode()
        controlDown = event.CmdDown()
        altDown = event.AltDown()
        shiftDown = event.ShiftDown()

        print keycode
        if keycode == wx.WXK_SPACE:
            print "you pressed the spacebar!"
        elif controlDown and altDown:
            print keycode
        event.Skip()

# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

我认为最主要的不同是你想检查口音或国际字符。因此，您将有复杂的条件来检查某些键是否被按下以及按下的顺序。robin Dunn(wxPython 的创建者)说 wxSTC 检查键和字符事件。如果你计划支持美国以外的用户，你可能想了解这一切是如何工作的。

罗宾·邓恩接着说*如果你想获得按键事件以便在应用程序中处理“命令”,那么在 EVT _ 按键 _ 按下处理程序中使用原始值是合适的。然而，如果目的是处理“文本”的输入，那么应用程序应该使用 EVT_CHAR 事件处理程序中的熟值，以便获得非美国键盘和输入法编辑器的正确处理。*(注意:向上键和向下键事件被认为是“未加工的”，而 char 事件已经为您“加工”好了。)正如 Robin Dunn 向我解释的那样，*在非美国键盘上，将键事件烹饪成字符事件的一部分是将物理键映射到国家键盘映射，以产生带有重音、元音等的字符。*

很抱歉，本教程没有涉及更多关于 char 事件的内容，但是我找不到太多的例子。

这些代码样本在以下方面进行了测试

*   Windows Vista SP2、wxPython 2.8.9.2(unicode)、Python 2.5.2

**下载量**

*   [按键和字符示例(zip)](https://www.blog.pythonlibrary.org/wp-content/uploads/2009/08/key.zip)
*   [键和字符示例(tar)](https://www.blog.pythonlibrary.org/wp-content/uploads/2009/08/key.tar)

**延伸阅读**

*   [关键事件列表](http://www.wxpython.org/docs/api/wx.KeyEvent-class.html)
*   【Windows 下的字符编码和键盘(wxPython Wiki)
*   [加速器表文档](http://www.wxpython.org/docs/api/wx.AcceleratorTable-class.html)