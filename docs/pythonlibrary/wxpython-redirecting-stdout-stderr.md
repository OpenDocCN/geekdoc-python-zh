# wxPython -重定向 stdout / stderr

> 原文：<https://www.blog.pythonlibrary.org/2009/01/01/wxpython-redirecting-stdout-stderr/>

每周都有新的程序员开始使用 Python 和 wxPython。因此，每隔几个月，我就会看到有人询问如何将 stdout 重定向到 wx。comp.lang.python 上的 TextCtrl 或 wxPython 邮件列表。既然这是一个如此普遍的问题，我想我应该写一篇关于它的文章。普通读者会知道我在之前的一篇文章中提到过这个概念。

### **更新于 2015-10-06**

最初我认为我们需要创建一个类，它可以 [duck-type](http://en.wikipedia.org/wiki/Duck_typing) 编写 wx.TextCtrl 的 API。注意，我使用了所谓的[“新风格”类](http://www.python.org/doc/newstyle/)，它子类化了**对象**(参见下面的代码)。

```py

class RedirectText(object):
    def __init__(self,aWxTextCtrl):
        self.out=aWxTextCtrl

    def write(self,string):
        self.out.WriteText(string)

```

注意，这个类中只有一个方法(当然，除了初始化方法)。它允许我们将文本从 stdout 或 stderr 写入文本控件。应该注意的是， *write* 方法不是线程安全的。如果您想要重定向线程中的文本，请将 write 语句更改如下:

```py

def write(self, string):
    wx.CallAfter(self.out.WriteText, string)

```

现在让我们深入研究我们需要的 wxPython 代码:

```py

import sys
import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "wxPython Redirect Tutorial")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        log = wx.TextCtrl(panel, wx.ID_ANY, size=(300,100),
                          style = wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)
        btn = wx.Button(panel, wx.ID_ANY, 'Push me!')
        self.Bind(wx.EVT_BUTTON, self.onButton, btn)

        # Add widgets to a sizer        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(log, 1, wx.ALL|wx.EXPAND, 5)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)
        panel.SetSizer(sizer)

        # redirect text here
        redir=RedirectText(log)
        sys.stdout=redir

    def onButton(self, event):        
        print "You pressed the button!"

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm().Show()
    app.MainLoop()

```

在上面的代码中，我创建了一个只读的多行文本控件和一个按钮，其唯一目的是将一些文本打印到 stdout。我将它们添加到一个 BoxSizer 中，以防止小部件相互堆叠，并更好地处理框架的大小调整。接下来，我通过向它传递我的文本控件的实例来实例化 RedirectText 类。最后，我将 stdout 设置为 RediectText 实例， **redir** (即 sys.stdout=redir)。

如果您还想重定向 stderr，那么只需在“sys.stdout=redir”之后添加以下内容:sys.stderr=redir

可以对此进行改进，即颜色编码(或预挂起)哪些消息来自 stdout，哪些来自 stderr，但我将把它留给读者作为练习。

最近有人向我指出，我不应该需要经历所有这些困难。相反，您可以将 stdout 直接重定向到 TextCtrl 小部件，因为它有自己的 **write** 方法。这里有一个例子:

```py

import sys
import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, 
                          title="wxPython Redirect Tutorial")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        style = wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL
        log = wx.TextCtrl(panel, wx.ID_ANY, size=(300,100),
                          style=style)
        btn = wx.Button(panel, wx.ID_ANY, 'Push me!')
        self.Bind(wx.EVT_BUTTON, self.onButton, btn)

        # Add widgets to a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(log, 1, wx.ALL|wx.EXPAND, 5)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)
        panel.SetSizer(sizer)

        # redirect text here
        sys.stdout=log

    def onButton(self, event):
        print "You pressed the button!"

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm().Show()
    app.MainLoop()

```

你会注意到上面的代码不再引用 **RedirectText** 类，因为我们不需要它。我很确定，如果你想使用线程，这样做是不安全的。为了安全起见，您需要以前面提到的类似方式覆盖 TextCtrl 的 write 方法。特别感谢 carandraug 为我指出了这一点。

#### 相关阅读

*   Python: [运行 Ping、Traceroute 等](https://www.blog.pythonlibrary.org/2010/06/05/python-running-ping-traceroute-and-more/)

#### 下载源代码

*   [redirectText.py](https://www.blog.pythonlibrary.org/wp-content/uploads/2009/01/redirecttext.txt)