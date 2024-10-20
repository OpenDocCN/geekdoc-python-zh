# wxPython:获取事件名称而不是整数

> 原文：<https://www.blog.pythonlibrary.org/2011/07/05/wxpython-get-the-event-name-instead-of-an-integer/>

StackOverflow 上最近有一个帖子，我觉得很有趣。它询问如何从事件对象中获取事件名称，比如 EVT 按钮，而不是事件的 id 号。所以我对这个主题做了一些调查，wxPython 没有内置任何东西来完成这项任务。wxPython 的创建者 Robin Dunn 建议我应该创建一个事件及其 id 的字典来完成这个壮举。因此，在本教程中，我们将看看如何去做。

我试图自己解决这个问题，但后来我决定确保别人还没有这样做。简单的谷歌搜索后，我找到了一个论坛帖子，里面罗宾·邓恩描述了如何做到这一点。以下是基本要点:

```py

import wx

eventDict = {}
for name in dir(wx):
    if name.startswith('EVT_'):
        evt = getattr(wx, name)
        if isinstance(evt, wx.PyEventBinder):
            eventDict[evt.typeId] = name

```

不过，这只能得到一般事件。在 wx 的一些子库中有一些特殊的事件，比如在 wx.grid 中。你必须考虑这类事情。我还没有想出一个通用的解决方案。但是在下面的 runnable 示例中，我也展示了如何添加这些事件。我们来看看吧！

```py

import wx
import wx.grid

########################################################################
class MyForm(wx.Frame):

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, title="Tutorial")

        self.eventDict = {}
        evt_names = [x for x in dir(wx) if x.startswith("EVT_")]
        for name in evt_names:
            evt = getattr(wx, name)
            if isinstance(evt, wx.PyEventBinder):
                self.eventDict[evt.typeId] = name

        grid_evt_names = [x for x in dir(wx.grid) if x.startswith("EVT_")]
        for name in grid_evt_names:
            evt = getattr(wx.grid, name)
            if isinstance(evt, wx.PyEventBinder):
                self.eventDict[evt.typeId] = name

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        btn = wx.Button(panel, wx.ID_ANY, "Get POS")

        btn.Bind(wx.EVT_BUTTON, self.onEvent)
        panel.Bind(wx.EVT_LEFT_DCLICK, self.onEvent)
        panel.Bind(wx.EVT_RIGHT_DOWN, self.onEvent)

    #---------------------------------------------------------------------- 
    def onEvent(self, event):
        """
        Print out what event was fired
        """
        evt_id = event.GetEventType()
        print self.eventDict[evt_id]

#----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm().Show()
    app.MainLoop()

```

如你所见，我们稍微改变了循环。我们采用了第一个例子中的循环，并将其与第一个 IF 语句结合起来，以创建一个列表理解。这将返回事件名称字符串列表。然后我们循环使用其他条件句添加到字典中。我们做两次，一次是常规事件，另一次是 wx.grid 事件。然后我们绑定一些事件来测试我们的事件字典。如果您运行这个程序，您将会看到，如果您执行任何绑定事件，它将会把这些事件名称打印到 stdout。在大多数系统中，这将是一个控制台窗口或调试窗口。

## 包扎

现在您知道如何获取事件的事件名称，而不仅仅是整数。这在调试时很有帮助，因为有时您希望将多个事件绑定到一个处理程序，并且需要检查并查看哪个事件被触发。编码快乐！

## 来源

*   [eventId2Name.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2011/07/eventId2Name.tar)
*   [eventId2Name.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2011/07/eventId2Name.zip)