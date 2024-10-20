# wxPython 和 PubSub:一个简单的教程

> 原文：<https://www.blog.pythonlibrary.org/2010/06/27/wxpython-and-pubsub-a-simple-tutorial/>

我在 wxPython 邮件列表或其 IRC 频道上看到许多关于框架间通信的问题，大多数时候开发人员需要的是 PubSub 模块。发布者/订阅者模型是向一个或多个侦听器发送消息的一种方式。你可以在这里阅读[。据说](http://en.wikipedia.org/wiki/Publish/subscribe)[观察者模式](http://en.wikipedia.org/wiki/Observer_pattern)是基于发布/订阅模式的。在 wxPython land 中，我们有 pubsub 模块，可以从 wx.lib.pubsub 中访问。它实际上包含在 wxPython 中，但你也可以从其[源 Forge](http://pubsub.sourceforge.net/) 中下载它作为一个独立的模块。pubsub 的替代方案是 [PyDispatcher](http://pypi.python.org/pypi/PyDispatcher/2.0.1) 模块。

无论如何，在这篇文章中，我们不会研究这些模块背后的理论。相反，我们将在 wxPython 中使用一个半实用的示例来展示如何使用内置版本的 pubsub 在两个框架之间进行通信。如果你还同意我的观点，那么我鼓励你继续读下去！
 **更新:本文为 wxPython 2.8。如果你碰巧在使用 wxPython 的新版本，那么你会想要阅读我这篇文章的新版本[这里](https://www.blog.pythonlibrary.org/2019/03/28/wxpython-4-and-pubsub/)**

## 如何在两个框架之间传递信息

我发现有时我需要打开一个非模态框架来获取用户的信息，然后将这些信息传递回应用程序的主框架。其他时候，我只需要告诉我的一个框架，另一个已经关闭。在这两种情况下，pubsub 来拯救。下面的例子将实际演示这两个问题的解决方案。

```py

import wx
from wx.lib.pubsub import Publisher

########################################################################
class OtherFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, wx.ID_ANY, "Secondary Frame")
        panel = wx.Panel(self)

        msg = "Enter a Message to send to the main frame"
        instructions = wx.StaticText(panel, label=msg)
        self.msgTxt = wx.TextCtrl(panel, value="")
        closeBtn = wx.Button(panel, label="Send and Close")
        closeBtn.Bind(wx.EVT_BUTTON, self.onSendAndClose)

        sizer = wx.BoxSizer(wx.VERTICAL)
        flags = wx.ALL|wx.CENTER
        sizer.Add(instructions, 0, flags, 5)
        sizer.Add(self.msgTxt, 0, flags, 5)
        sizer.Add(closeBtn, 0, flags, 5)
        panel.SetSizer(sizer)

    #----------------------------------------------------------------------
    def onSendAndClose(self, event):
        """
        Send a message and close frame
        """
        msg = self.msgTxt.GetValue()
        Publisher().sendMessage(("show.mainframe"), msg)
        self.Close()

########################################################################
class MainPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        self.frame = parent

        Publisher().subscribe(self.showFrame, ("show.mainframe"))

        self.pubsubText = wx.TextCtrl(self, value="")
        hideBtn = wx.Button(self, label="Hide")
        hideBtn.Bind(wx.EVT_BUTTON, self.hideFrame)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.pubsubText, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(hideBtn, 0, wx.ALL|wx.CENTER, 5)
        self.SetSizer(sizer)

    #----------------------------------------------------------------------
    def hideFrame(self, event):
        """"""
        self.frame.Hide()
        new_frame = OtherFrame()
        new_frame.Show()

    #----------------------------------------------------------------------
    def showFrame(self, msg):
        """
        Shows the frame and shows the message sent in the
        text control
        """
        self.pubsubText.SetValue(msg.data)
        frame = self.GetParent()
        frame.Show()

########################################################################
class MainFrame(wx.Frame):

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Pubsub Tutorial")
        panel = MainPanel(self)

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()

```

这段代码的第一站是 MainPanel 类。请注意下面一行:

```py

Publisher().subscribe(self.showFrame, ("show.mainframe"))

```

这创建了一个订阅“show.mainframe”主题的监听器 singleton(又名:接收器)。程序的其他部分可以发布到该主题，监听器将获取它们并调用“showFrame”方法。要了解这一点，请查看“OtherFrame”类的“onSendAndClose”方法。

```py

# OtherFrame Class
def onSendAndClose(self, event):
    """
    Send a message and close frame
    """
    msg = self.msgTxt.GetValue()
    Publisher().sendMessage(("show.mainframe"), msg)
    self.Close()

```

这里，我们获取文本控件的值，并将其发送回主框架。要发送它，我们调用*发布者*对象的 *sendMessage* 方法，并向其传递主题字符串和消息。该消息可以是对象列表，也可以只是单个对象。在这种情况下，它只是一个字符串。回到*主面板*，调用 *showFrame* 方法。看起来是这样的:

```py

# MainPanel class
def showFrame(self, msg):
    """
    Shows the frame and shows the message sent in the
    text control
    """
    self.pubsubText.SetValue(msg.data)
    frame = self.GetParent()
    frame.Show()

```

在这个方法中，我们通过 pubsub 的*数据*属性提取通过 pubsub 发送的数据。如果我们使用一个列表发送多个条目，我们需要做一些类似 msg.data[0]的事情来获得正确的条目(假设字符串在元素一中)。最新的 pubsub 有一个稍微不同的 API，你可以在它的[食谱](http://pubsub.sourceforge.net/recipes/upgrade_v1tov3.html)中查看。最新的 API 是从 wxPython 2.8.11.0 开始提供的。*注意:我在用最新的 pubsub 创建二进制文件时遇到了一些麻烦，因为我使用的是稍旧的 API。参见此[线程](http://groups.google.com/group/wxpython-users/browse_thread/thread/d448a42abdae3e69/318cc65f2b54348f?lnk=gst&q=pubsub#318cc65f2b54348f)以了解详细信息和一些可能的解决方法。*

现在您已经了解了在项目中使用 pubsub 的基本知识。这个例子展示了如何在两个框架之间进行通信，即使其中一个是隐藏的！它还展示了如何将信息从一个框架传递到另一个框架。玩得开心！

## 附加阅读

*   [创建一个简单的照片浏览器](https://www.blog.pythonlibrary.org/2010/03/26/creating-a-simple-photo-viewer-with-wxpython/)
*   [wxPython 和线程](https://www.blog.pythonlibrary.org/2010/05/22/wxpython-and-threads/)
*   [Pubsub wxPython wiki 页面](http://wiki.wxpython.org/WxLibPubSub)
*   官方发布主页