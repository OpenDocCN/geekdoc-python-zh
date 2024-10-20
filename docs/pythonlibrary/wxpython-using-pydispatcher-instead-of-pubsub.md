# wxPython:使用 PyDispatcher 而不是 Pubsub

> 原文：<https://www.blog.pythonlibrary.org/2013/09/06/wxpython-using-pydispatcher-instead-of-pubsub/>

前几天，我为 wxPython 2.9 写了一篇 wxPython pubsub [文章](https://www.blog.pythonlibrary.org/2013/09/05/wxpython-2-9-and-the-newer-pubsub-api-a-simple-tutorial/)的更新版本，并意识到我从未尝试过 PyDispatcher 来看看它与 pubsub 有何不同。我仍然不确定它在内部有什么不同，但我认为将上一篇文章中的 pubsub 代码“移植”到 PyDispatcher 会很有趣。看看变化有多大！

### 入门指南

首先，您需要获取 PyDispatcher 并将其安装在您的系统上。如果安装了 pip，您可以执行以下操作:

```py

pip install PyDispatcher

```

否则，转到项目的 [sourceforge 页面](http://pydispatcher.sourceforge.net/)并从那里下载。在 wxPython 中使用 pubsub 的好处之一是它已经包含在标准的 wxPython 发行版中。但是，如果您想在 wxPython 之外使用 pubsub，您必须下载它的独立代码库并安装它。我只是觉得我应该提一下。大多数开发者不喜欢在其他包之上下载额外的包。

无论如何，现在我们有了 PyDispatcher，让我们移植代码，看看我们最终会得到什么！

```py

import wx
from pydispatch import dispatcher 

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
        dispatcher.send("panelListener", message=msg)
        dispatcher.send("panelListener", message="test2", arg2="2nd argument!")
        self.Close()

########################################################################
class MyPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        dispatcher.connect(self.myListener, signal="panelListener",
                           sender=dispatcher.Any)

        btn = wx.Button(self, label="Open Frame")
        btn.Bind(wx.EVT_BUTTON, self.onOpenFrame)

    #----------------------------------------------------------------------
    def myListener(self, message, arg2=None):
        """
        Listener function
        """
        print "Received the following message: " + message
        if arg2:
            print "Received another arguments: " + str(arg2)

    #----------------------------------------------------------------------
    def onOpenFrame(self, event):
        """
        Opens secondary frame
        """
        frame = OtherFrame()
        frame.Show()

########################################################################
class MyFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="New PubSub API Tutorial")
        panel = MyPanel(self)
        self.Show()

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

```

这应该不会花太长时间来解释。所以我们从 **pydispatch** 导入**dispatch**。然后我们编辑另一个框架的 **onSendAndClose** 方法，这样它就会向我们的面板监听器发送消息。怎么会？通过执行以下操作:

```py

dispatcher.send("panelListener", message=msg)
dispatcher.send("panelListener", message="test2", arg2="2nd argument!")

```

然后在 MyPanel 类中，我们像这样设置一个侦听器:

```py

dispatcher.connect(self.myListener, signal="panelListener",
                   sender=dispatcher.Any)

```

这个代码告诉 pydispatcher 监听任何有**panel ener**信号的发送者。如果它有那个信号，那么它将调用面板的 **myListener** 方法。这就是我们从 pubsub 到 pydispatcher 的所有工作。那不是很容易吗？