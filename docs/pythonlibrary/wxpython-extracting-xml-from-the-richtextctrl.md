# wxPython:从 RichTextCtrl 中提取 XML

> 原文：<https://www.blog.pythonlibrary.org/2015/07/10/wxpython-extracting-xml-from-the-richtextctrl/>

我最近遇到了一个 StackOverflow 问题,这个人问如何获取 wxPython 的 RichTextCtrl 的 XML 数据，以便保存到数据库中。我不太了解这个控件，但在谷歌上快速搜索后，我找到了 2008 年的一篇[文章](http://play.pixelblaster.ro/blog/archive/2008/10/08/richtext-control-with-wxpython-saving-and-loading)，它给了我需要的信息。我把这个例子简化成下面的例子:

```py

import wx
import wx.richtext

from StringIO import StringIO

########################################################################
class MyFrame(wx.Frame):

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, title='Richtext Test')

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.rt = wx.richtext.RichTextCtrl(self)
        self.rt.SetMinSize((300,200))

        save_button = wx.Button(self, label="Save")
        save_button.Bind(wx.EVT_BUTTON, self.on_save)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rt, 1, wx.EXPAND|wx.ALL, 6)
        sizer.Add(save_button, 0, wx.EXPAND|wx.ALL, 6)

        self.SetSizer(sizer)
        self.Show()

    #----------------------------------------------------------------------
    def on_save(self, event):
        out = StringIO()
        handler = wx.richtext.RichTextXMLHandler()
        rt_buffer = self.rt.GetBuffer()
        handler.SaveStream(rt_buffer, out)
        self.xml_content = out.getvalue()
        print self.xml_content

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

```

让我们把它分解一下。首先，我们创建了可爱的应用程序，并在框架中添加了一个 **RichTextCtrl** 小部件的实例，以及一个按钮，用于保存我们在该小部件中写入的内容。接下来，我们为按钮设置绑定并布局小部件。最后，我们创建事件处理程序。这就是奇迹发生的地方。这里我们创建了 **RichTextXMLHandler** 并获取了 RichTextCtrl 这样我们就可以写出数据了。但是我们写的不是一个文件，而是一个类似文件的对象，也就是我们的 **StringIO** 实例。我们这样做是为了将数据写入内存，然后再读取出来。我们这样做的原因是因为 StackOverflow 上的人想要一种方法来提取 RichTextCtrl 生成的 XML 并将其写入数据库。我们可以先把它写到磁盘上，然后再读取那个文件，但是这样做不那么麻烦，速度也更快。

但是请注意，如果有人在 RichTextCtrl 中写了一部小说，那么这将是一个坏主意！虽然我们不太可能用尽空间，但肯定有大量文本文件超出了您计算机的内存。如果你知道你正在加载的文件会占用很多内存，那么你就不会走这条路。相反，你应该成块地读写数据。无论如何，这段代码为我们想要做的事情工作。我希望你觉得这很有用。弄清楚这一点当然很有趣。

不幸的是，这个代码示例不能在**wxPython Phoenix**中运行。在下一节中，我们将更新该示例，以便它能够！

* * *

### Phoenix / wxPython 4 的更新

在 wxPython 4(又名 Phoenix)中运行上述示例时，您将遇到的第一个问题是, **SaveStream** 方法不再存在。你需要使用**保存文件**来代替。另一个问题实际上是 Python 3 引入的一个问题。如果你在 Python 3 中运行这段代码，你会发现 **StringIO** 模块并不存在，你需要使用 **io** 来代替。所以对于我们的下一个例子，我更新了代码以支持 Python 3 和 wxPython Phoenix。让我们看看它有什么不同:

```py

# wxPython 4 (Phoenix) / Python 3 Version

import wx
import wx.richtext

from io import BytesIO

class MyFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title='Richtext Test')

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.rt = wx.richtext.RichTextCtrl(self)
        self.rt.SetMinSize((300,200))

        save_button = wx.Button(self, label="Save")
        save_button.Bind(wx.EVT_BUTTON, self.on_save)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rt, 1, wx.EXPAND|wx.ALL, 6)
        sizer.Add(save_button, 0, wx.EXPAND|wx.ALL, 6)

        self.SetSizer(sizer)
        self.Show()

    def on_save(self, event):
        out = BytesIO()
        handler = wx.richtext.RichTextXMLHandler()
        rt_buffer = self.rt.GetBuffer()
        handler.SaveFile(rt_buffer, out)
        self.xml_content = out.getvalue()
        print(self.xml_content)

if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

```

主要区别在于开头的 imports 部分和 **on_save** 方法。你会注意到我们正在使用 io 模块的**字节序**类。然后，我们以与之前相同的方式获取其余数据，只是我们将存储流与存储文件进行了交换。输出的 XML 是一个二进制字符串，所以如果您打算解析它，那么您可能需要将结果转换成一个字符串。我有过一些不能正确处理二进制字符串的 XML 解析器。

* * *

### 包装材料

虽然本文只讨论了提取 XML，但是您可以很容易地将其扩展到提取 RichTextCtrl 支持的其他格式，比如 HTML 或 RTF 本身。如果您需要将应用程序中的数据保存到数据库或其他数据存储中，这可能是一个有用的工具。