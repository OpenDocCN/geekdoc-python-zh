# wxPython:摆弄鼠标光标

> 原文：<https://www.blog.pythonlibrary.org/2008/08/02/wxpython-messing-with-mouse-cursors/>

最近在 wxPython 邮件列表上，关于改变鼠标图标有相当多的流量。在本文中，我将描述用 wxPython 操作光标的不同方法。为了跟进，我建议您下载 [Python 2.4 或更高版本](http://www.python.org)和 [wxPython 2.8.x](http://www.wxpython.org) 。

wxPython(以及一般的 Python)的一个很酷的地方是它“包括电池”。在这种情况下，wxPython 在 [wxPython 文档](http://wxpython.org/docs/api/wx.Cursor-class.html)中提供了可用股票光标的列表。以下是设定股票图标的方法:

```py

myCursor= wx.StockCursor(wx.CURSOR_POINT_LEFT)
myFrame.SetCursor(myCursor)

```

当然，有时您可能希望使用自定义光标。如果是这种情况，那么您可以使用 wx 创建一个游标对象。光标()。以下是我在 Windows 中的做法。您必须根据操作系统的需要进行调整。

```py

# wx.Cursor(path\to\file, wx.BITMAP_TYPE* constant)
myCursor= wx.Cursor(r"C:\WINDOWS\Cursors\3dgarro.cur",
                    wx.BITMAP_TYPE_CUR)

# self is a wx.Frame in my example
myFrame.SetCursor(myCursor)

```

请注意，您可能无法在 Mac 上使用自定义光标。至少，罗宾·邓恩在他的博客上是这么说的。这就是你真正需要知道的使这个工作。如果你有问题，一定要发邮件给我，地址是“python library . org 的 mike”或 [wxPython 用户组](http://wxpython.org/maillist.php)。