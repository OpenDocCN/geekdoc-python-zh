# wxPython:学习专注

> 原文：<https://www.blog.pythonlibrary.org/2009/08/27/wxpython-learning-to-focus/>

这周我接到一个请求，要写一个关于 wxPython 中焦点事件的快速教程。幸运的是，在这个主题上没有太多的材料，所以这应该是一个非常快速和肮脏的小帖子。跳完之后再见！

我见过的焦点事件真的只有两个:wx。SET _ 设置 _ 焦点和 wx。EVT _ 杀死 _ 聚焦。当小部件获得焦点时，例如当您单击空白面板或将光标放在 TextCtrl 小部件中时，将触发 EVT_SET_FOCUS 事件。当你点击一个有焦点的小部件时，EVT_KILL_FOCUS 就会被触发。

我在 wxPython 邮件列表中看到的少数几个“陷阱”之一是 wx。只有当*而不是*有一个可以接受焦点的子部件时，面板才接受焦点。最好的解释方式是用一系列的例子。

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Focus Tutorial 1")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        panel.Bind(wx.EVT_SET_FOCUS, self.onFocus)        

    def onFocus(self, event):
        print "panel received focus!"

# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

现在这段代码将显示一个空白面板，上面什么也没有。您会注意到 stdout 立即得到“panel received focus！”打印出来。现在，如果我们添加一个 TextCtrl 或一个按钮，那么它们将获得焦点，而 OnFocus 事件处理程序将不会被触发。试着运行下面的代码来看看这是怎么回事:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Focus Tutorial 1a")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        panel.Bind(wx.EVT_SET_FOCUS, self.onFocus)
        txt = wx.TextCtrl(panel, wx.ID_ANY, "")

    def onFocus(self, event):
        print "panel received focus!"

# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

只是为了好玩，试着在那里放一个 StaticText 控件，而不是 TextCtrl。你期待什么会得到关注？如果你猜是 StaticText 控件或面板，那你就错了！事实上，它是接收焦点的框架！Robin Dunn 通过在我的代码中添加一个计时器向我说明了这一点。看看这个:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Focus Finder")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        panel.Bind(wx.EVT_SET_FOCUS, self.onFocus)
        txt = wx.StaticText(panel, wx.ID_ANY, 
                   "This label cannot receive focus")

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimer)
        self.timer.Start(1000)

    def onFocus(self, event):
        print "panel received focus!"

    def onTimer(self, evt):
        print 'Focused window:', wx.Window.FindFocus()

# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

你可能想知道为什么你会想知道什么时候帧在焦点上。如果您需要知道某人何时点击了您的应用程序或将某个帧放到了前台，这可能会很有帮助。当然，有些人更希望知道鼠标何时进入框架，这些信息可以通过 EVT _ 回车 _ 窗口获得(我不会在这里讨论)。

现在我们来快速看一下 wx。EVT _ 杀死 _ 聚焦。我创建了一个只有两个控件的简单示例。试着猜猜如果你在它们之间切换会发生什么。

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Focus Tutorial 1a")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)

        txt = wx.TextCtrl(panel, wx.ID_ANY, "")
        txt.Bind(wx.EVT_SET_FOCUS, self.onFocus)
        txt.Bind(wx.EVT_KILL_FOCUS, self.onKillFocus)
        btn = wx.Button(panel, wx.ID_ANY, "Test")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(txt, 0, wx.ALL, 5)
        sizer.Add(btn, 0, wx.ALL, 5)
        panel.SetSizer(sizer)

    def onFocus(self, event):
        print "widget received focus!"

    def onKillFocus(self, event):
        print "widget lost focus!"

# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

正如您可能已经猜到的，当您在它们之间切换时，TextCtrl 要么触发一个 kill focus 事件，要么触发一个 set focus 事件。您如何知道哪个小部件触发了这些事件？看看我的绑定方法。只有文本控件绑定到焦点事件。作为一个练习，试着将按钮也绑定到这些处理程序，并打印出哪个小部件触发了什么。

我要提到的最后一个焦点事件是 wx。EVT 儿童对焦，我从来没用过的。该事件用于确定子部件何时获得焦点，并确定它是哪个子部件。根据 [Robin Dunn](http://wxpython.org/blog/) ，wx.lib.scrolledpanel 使用这个事件。我的一个读者告诉我一个关于 wx 的便利用例。EVT _ 儿童 _ 焦点:*你可以在框架上使用它，当你点击任何其他子部件时，简单地清除状态栏。这样
当你点击一个不同的子部件时，你就不会有一个旧的“错误”消息或者在状态栏中总结这样的文本。* (hattip @devplayer)

还要注意，一些更复杂的小部件有自己的 focus hokus pokus。有关如何在 wx.grid.Grid 单元格中获得焦点，请参见以下主题:http://www . velocity reviews . com/forums/t 352017-wx grid-and-focus-event . html

我希望本教程能够帮助您更好地理解 wxPython 中焦点事件的工作方式。如果您有问题或其他反馈，请随时通过评论给我留言，或者在[邮件列表](http://wxpython.org/maillist.php)上询问其他 wxPython 开发人员。

**附加信息**

*   [微软 Windows 下与 EVT_KILL_FOCUS 一起生存](http://wiki.wxpython.org/Surviving%20with%20wxEVT%20KILL%20FOCUS%20under%20Microsoft%20Windows)
*   [wxPython 中的事件](http://zetcode.com/wxpython/events/)
*   [wx。焦点事件](http://www.wxpython.org/docs/api/wx.FocusEvent-class.html)

**下载量**

*   [焦点示例(zip)](https://www.blog.pythonlibrary.org/wp-content/uploads/2009/08/focusExamples.zip)
*   [焦点示例(焦油)](https://www.blog.pythonlibrary.org/wp-content/uploads/2009/08/focusExamples.tar)