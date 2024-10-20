# wxPython 4 和 PubSub

> 原文：<https://www.blog.pythonlibrary.org/2019/03/28/wxpython-4-and-pubsub/>

[发布-订阅](https://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern)模式在计算机科学中非常普遍，也非常有用。wxPython GUI 工具包在 **wx.lib.pubsub** 中已经实现了很长时间。这个实现基于 [PyPubSub](https://pypubsub.readthedocs.io/en/v4.0.3/) 包。虽然您总是可以下载 PyPubSub 并直接使用它，但是能够在没有额外依赖的情况下运行 wxPython 还是不错的。

然而，从 **wxPython 4.0.4** ， **wx.lib.pubsub** 现在已经被弃用，并将在 wxPython 的未来版本中删除。因此，如果您想在 wxPython 中轻松使用发布-订阅模式，现在您需要下载 PyPubSub 或 [PyDispatcher](https://pypi.org/project/PyDispatcher/) 。

* * *

### 正在安装 PyPubSub

您可以使用 pip 安装 PyPubSub。

以下是如何做到这一点:

```py

pip install pypubsub

```

PyPubSub 应该可以很快安装。一旦完成，让我们看看如何使用它！

* * *

### 使用 PyPubSub

让我们从我以前关于这个主题的文章中取一个例子，并更新它以使用 PyPubSub:

```py

import wx
from pubsub import pub

class OtherFrame(wx.Frame):
    """"""

    def __init__(self):
        """Constructor"""
        super().__init__(None, title="Secondary Frame")
        panel = wx.Panel(self)

        msg = "Enter a Message to send to the main frame"
        instructions = wx.StaticText(panel, label=msg)
        self.msg_txt = wx.TextCtrl(panel, value="")
        close_btn = wx.Button(panel, label="Send and Close")
        close_btn.Bind(wx.EVT_BUTTON, self.on_send_and_slose)

        sizer = wx.BoxSizer(wx.VERTICAL)
        flags = wx.ALL|wx.CENTER
        sizer.Add(instructions, 0, flags, 5)
        sizer.Add(self.msg_txt, 0, flags, 5)
        sizer.Add(close_btn, 0, flags, 5)
        panel.SetSizer(sizer)

    def on_send_and_slose(self, event):
        """
        Send a message and close frame
        """
        msg = self.msg_txt.GetValue()
        pub.sendMessage("panel_listener", message=msg)
        pub.sendMessage("panel_listener", message="test2",
                        arg2="2nd argument!")
        self.Close()

class MyPanel(wx.Panel):
    """"""

    def __init__(self, parent):
        """Constructor"""
        super().__init__(parent)
        pub.subscribe(self.my_listener, "panel_listener")

        btn = wx.Button(self, label="Open Frame")
        btn.Bind(wx.EVT_BUTTON, self.on_open_frame)

    def my_listener(self, message, arg2=None):
        """
        Listener function
        """
        print(f"Received the following message: {message}")
        if arg2:
            print(f"Received another arguments: {arg2}")

    def on_open_frame(self, event):
        """
        Opens secondary frame
        """
        frame = OtherFrame()
        frame.Show()

class MyFrame(wx.Frame):
    """"""

    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None,
                          title="New PubSub API Tutorial")
        panel = MyPanel(self)
        self.Show()

if __name__ == "__main__":
    app = wx.App(False)

```

这里使用内置 PubSub 的主要区别是导入。

你需要做的就是替换这个:

```py

from wx.lib.pubsub import pub 

```

有了这个:

```py

from pubsub import pub

```

只要您使用的是 wxPython 2.9 或更高版本。如果您一直在使用 wxPython 2.8，那么您可能想看看我以前关于这个主题的一篇文章，看看 PubSub API 是如何变化的。

如果您使用的是 wxPython 2.9 或更高版本，那么这种改变非常容易，几乎没有痛苦。

像往常一样，你订阅一个话题:

```py

pub.subscribe(self.myListener, "panelListener")

```

然后你发布到那个主题:

```py

pub.sendMessage("panelListener", message=msg)

```

试一试，看看添加到您自己的代码中是多么容易！

* * *

### 包扎

我个人非常喜欢使用 **wx.lib.pubsub** ，所以我可能会继续使用 PyPubSub。然而，如果您曾经想尝试另一个包，如 PyDispatcher，这将是一个很好的时机。

* * *

### 相关阅读

*   wxPython 2.9 和更新的 Pubsub API: [一个简单的教程](https://www.blog.pythonlibrary.org/2013/09/05/wxpython-2-9-and-the-newer-pubsub-api-a-simple-tutorial/)
*   wxPython 和 PubSub: [一个简单的教程](https://www.blog.pythonlibrary.org/2010/06/27/wxpython-and-pubsub-a-simple-tutorial/)
*   wxPython: [使用 PyDispatcher 代替 Pubsub](https://www.blog.pythonlibrary.org/2013/09/06/wxpython-using-pydispatcher-instead-of-pubsub/)