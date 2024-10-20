# wxPython 2.9 和更新的 Pubsub API:一个简单的教程

> 原文：<https://www.blog.pythonlibrary.org/2013/09/05/wxpython-2-9-and-the-newer-pubsub-api-a-simple-tutorial/>

**注**:本文针对 wxPython 2.9-3.0。如果你用的是 **wxPython 4** ，你应该去我的[更新文章](https://www.blog.pythonlibrary.org/2019/03/28/wxpython-4-and-pubsub/)

几年前，我写了一篇关于 wxPython 2.8 及其内置 pubsub 模块的教程，你可以在这里阅读。当时，wxPython 2.8.11.0 中为 pubsub 添加了一个新的 API，可以通过执行以下操作来启用它:

```py

import wx.lib.pubsub.setupkwargs
from wx.lib.pubsub import pub

```

导入 pubsub 的旧方法是执行以下操作:

```py

from wx.lib.pubsub import Publisher

```

现在在 wxPython 2.9 中，它变成了这样:

```py

from wx.lib.pubsub import pub

```

因此，您不能再使用我的旧教程中的代码，并期望它在 wxPython 的最新版本中工作。所以是时候稍微更新一下教程了。

### 新的 pubsub API

让我们从旧文章中提取原始代码，并使用 pubsub 较新的 API 对其进行修饰。不会花很长时间。事实上，这是一个非常小的代码更改。我们来看看吧！

```py

import wx
from wx.lib.pubsub import pub 

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
        pub.sendMessage("panelListener", message=msg)
        pub.sendMessage("panelListener", message="test2", arg2="2nd argument!")
        self.Close()

########################################################################
class MyPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        pub.subscribe(self.myListener, "panelListener")

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

正如我们已经讨论过的，进口是不同的。我们来看看还有什么变化。在 panel 类中，我们像这样创建一个侦听器:

```py

pub.subscribe(self.myListener, "panelListener")

```

myListener 方法可以接受一个或多个参数。在这种情况下，我们将其设置为总是需要一个参数(message)和一个可选参数(arg2)。接下来我们转向 **OtherFrame** 类，在这里我们需要看一下 **onSendAndClose** 方法。在这个方法中，我们发现它发出了两个信息:

```py

msg = self.msgTxt.GetValue()
pub.sendMessage("panelListener", message=msg)
pub.sendMessage("panelListener", message="test2", arg2="2nd argument!")
self.Close()

```

第一个只发送所需的信息，而第二个发送两者。您会注意到新的 API 需要使用显式的关键字参数。如果您将第一个 sendMessage 命令更改为**pub . sendMessage(" panel ener "，msg)** ，您将收到一个 TypeError 异常。

### 包扎

这是一个相当简单的变化，是吧？我认为新的 pubsub API 实际上比原来的可读性更好，而且不那么“神奇”。希望你也会。开心快乐编码！

### 附加阅读

*   wxPython 和 PubSub: [一个简单的教程](https://www.blog.pythonlibrary.org/2010/06/27/wxpython-and-pubsub-a-simple-tutorial/)
*   在 [wxPython 邮件列表](https://groups.google.com/forum/#!searchin/wxpython-users/pubsub/wxpython-users/tKbfaVr-URk/2uZDkfd0k34J)的 pubsub 上的众多主题之一
*   pubsub [网站](http://pubsub.sourceforge.net/)
*   wxPython: [如何从线程更新进度条](https://www.blog.pythonlibrary.org/2013/09/04/wxpython-how-to-update-a-progress-bar-from-a-thread/)