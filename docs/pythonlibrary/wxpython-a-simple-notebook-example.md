# wxPython:一个简单的笔记本例子

> 原文：<https://www.blog.pythonlibrary.org/2010/09/15/wxpython-a-simple-notebook-example/>

前几天，我接到一个投诉，说我的[书控系列](https://www.blog.pythonlibrary.org/2009/12/03/the-book-controls-of-wxpython-part-1-of-2/)里我原来的笔记本例子太复杂了。我并不真的只写对 n00b 友好的文章，也从来没有这样说过，但是这个评论引起了我的不满，所以我决定为 wxPython 新手写一个超级简单的例子。希望你喜欢！

```py

import random
import wx

########################################################################
class TabPanel(wx.Panel):
    #----------------------------------------------------------------------
    def __init__(self, parent):
        """"""
        wx.Panel.__init__(self, parent=parent)

        colors = ["red", "blue", "gray", "yellow", "green"]
        self.SetBackgroundColour(random.choice(colors))

        btn = wx.Button(self, label="Press Me")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(btn, 0, wx.ALL, 10)
        self.SetSizer(sizer)

########################################################################
class DemoFrame(wx.Frame):
    """
    Frame that holds all other widgets
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""        
        wx.Frame.__init__(self, None, wx.ID_ANY, 
                          "Notebook Tutorial",
                          size=(600,400)
                          )
        panel = wx.Panel(self)

        notebook = wx.Notebook(panel)
        tabOne = TabPanel(notebook)
        notebook.AddPage(tabOne, "Tab 1")

        tabTwo = TabPanel(notebook)
        notebook.AddPage(tabTwo, "Tab 2")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(notebook, 1, wx.ALL|wx.EXPAND, 5)
        panel.SetSizer(sizer)
        self.Layout()

        self.Show()

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = DemoFrame()
    app.MainLoop()

```

通常我写 wx 的时候。笔记本的例子，我希望每个标签看起来不同的方式。如果您查看前面提到的系列的第一篇文章，您会看到笔记本的每一页中都嵌入了一些复杂的小部件。然而，在这个例子中，我只是随机地给每一页涂上不同的颜色。让我们花点时间来解开这段代码。

在 **DemoFrame** 类中，我们有一个主面板，它是框架的唯一子部件。在那里面，我们有笔记本控制。笔记本的每一页都是我们的 **TabPanel** 类的一个实例，它应该有一个“随机”的背景颜色和一个不做任何事情的按钮。我们将笔记本添加到 sizer 中，并将其设置为以 1 的比例扩展。这意味着它将填充面板，并且由于面板填充框架，笔记本也将填充框架。说实话，真的就是这么回事。

另一个值得注意的话题是，笔记本事件，如 EVT _ 笔记本 _ 页面 _ 改变，可能需要有一个“事件”。Skip()"调用它们的事件处理程序，使它们正常工作。wxPython 中的事件层次有点难以理解，但可以把它想象成池塘中的气泡。如果您将一个小部件绑定到一个特定的事件，并且不调用 Skip()，那么您的事件只在那个特定的处理程序中处理。这就像让泡沫从池塘底部部分破裂一样。然而，有时你需要事件在更高的层次上处理，比如在部件的父级或祖父级。如果是这样，调用 Skip()并且您的事件“bubble”将上升到下一个处理程序。wxPython wiki 有更多关于这方面的内容，正如“ [wxPython in Action](http://amzn.to/95Gcln) ”一书一样。

好了，这涵盖了简单的笔记本电脑的例子。如果您需要了解关于笔记本或其他书籍控件的更多信息，请参阅本文的第一个链接或下载 wxPython 演示。