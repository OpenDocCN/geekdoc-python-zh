# 用 wxPython 做淡入

> 原文：<https://www.blog.pythonlibrary.org/2008/04/14/doing-a-fade-in-with-wxpython/>

今天我们将讨论如何让你的应用程序做一个“淡入”。Windows 用户通常会在 Microsoft Outlook 的电子邮件通知中看到这一点。它淡入淡出。wxPython 提供了一种设置任何顶层窗口的 alpha 透明度的方法，这会影响放置在顶层小部件上的小部件。

在这个例子中，我将使用一个框架对象作为顶层对象，并使用一个计时器来改变 alpha 透明度，单位为每秒 5。计时器的事件处理程序将使帧淡入视图，然后再次退出。值的范围是 0 - 255，0 表示完全透明，255 表示完全不透明。

代码如下:

```py

import wx

class Fader(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title='Test')
        self.amount = 5
        self.delta = 5
        panel = wx.Panel(self, wx.ID_ANY)

        self.SetTransparent(self.amount)

        ## ------- Fader Timer -------- ##
        self.timer = wx.Timer(self, wx.ID_ANY)
        self.timer.Start(60)
        self.Bind(wx.EVT_TIMER, self.AlphaCycle)
        ## ---------------------------- ##

    def AlphaCycle(self, evt):
        self.amount += self.delta
        if self.amount >= 255:
            self.delta = -self.delta
            self.amount = 255
        if self.amount <= 0:
            self.amount = 0
        self.SetTransparent(self.amount)

if __name__ == '__main__':
    app = wx.App(False)
    frm = Fader()
    frm.Show()
    app.MainLoop()

```

如您所见，要更改顶级小部件的透明度，您只需调用该小部件的 SetTransparent()方法，并向其传递要设置的数量。实际上，我在自己的一个应用程序中使用了这种方法，它会在一个对话框中淡入提醒我 Zimbra 电子邮件帐户中有新邮件。

**欲了解更多信息，请查看以下资源:**

[计时器](http://wiki.wxpython.org/Timer)
[透明相框](http://wiki.wxpython.org/Transparent%20Frames)

**对以下代码进行了测试:**

操作系统:Windows XP
Python:2 . 5 . 2
wxPython:2.8.8.1 和 2.8.9.1