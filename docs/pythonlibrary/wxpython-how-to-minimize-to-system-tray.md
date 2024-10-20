# wxPython:如何最小化到系统托盘

> 原文：<https://www.blog.pythonlibrary.org/2013/07/12/wxpython-how-to-minimize-to-system-tray/>

我在网上各个地方看到有人时不时的问这个话题。让 wxPython 最小化到托盘非常简单，但是至少有一件事需要注意。我们会谈到这一点，但首先我们需要花一些时间来看一些代码。事实上，我在几年前写过关于 TaskBarIcons 的文章。首先我们来看看任务栏图标代码，它大致基于前面提到的文章:

```py

import wx

########################################################################
class CustomTaskBarIcon(wx.TaskBarIcon):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, frame):
        """Constructor"""
        wx.TaskBarIcon.__init__(self)
        self.frame = frame

        img = wx.Image("24x24.png", wx.BITMAP_TYPE_ANY)
        bmp = wx.BitmapFromImage(img)
        self.icon = wx.EmptyIcon()
        self.icon.CopyFromBitmap(bmp)

        self.SetIcon(self.icon, "Restore")
        self.Bind(wx.EVT_TASKBAR_LEFT_DOWN, self.OnTaskBarLeftClick)

    #----------------------------------------------------------------------
    def OnTaskBarActivate(self, evt):
        """"""
        pass

    #----------------------------------------------------------------------
    def OnTaskBarClose(self, evt):
        """
        Destroy the taskbar icon and frame from the taskbar icon itself
        """
        self.frame.Close()

    #----------------------------------------------------------------------
    def OnTaskBarLeftClick(self, evt):
        """
        Create the right-click menu
        """
        self.frame.Show()
        self.frame.Restore()

```

如您所见，我们需要继承 wx 的子类。TaskBarIcon，然后给它一个图标。对于本文，我们将使用来自 [deviantart](http://theg-force.deviantart.com/art/Social-Icons-hand-drawned-109467069) 的免费图标。好的，在 init 中，我们必须通过几道关卡将 PNG 文件转换成 wx 的 icon 方法可以使用的格式。当子类化 wx.TaskBarIcon 时，其余的方法是必需的。你会注意到我们绑定到 LEFT _ 任务栏 _ 左 _ 下，这样当用户点击图标时，我们可以恢复窗口。

现在我们准备看看框架的代码。

```py

import custTray
import wx

########################################################################
class MainFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Minimize to Tray")
        panel = wx.Panel(self)
        self.tbIcon = custTray.CustomTaskBarIcon(self)

        self.Bind(wx.EVT_ICONIZE, self.onMinimize)
        self.Bind(wx.EVT_CLOSE, self.onClose)

        self.Show()

    #----------------------------------------------------------------------
    def onClose(self, evt):
        """
        Destroy the taskbar icon and the frame
        """
        self.tbIcon.RemoveIcon()
        self.tbIcon.Destroy()
        self.Destroy()

    #----------------------------------------------------------------------
    def onMinimize(self, event):
        """
        When minimizing, hide the frame so it "minimizes to tray"
        """
        if self.IsIconized():
            self.Hide()

#----------------------------------------------------------------------
def main():
    """"""
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

if __name__ == "__main__":
    main()

```

这里我们有两个事件绑定。一个是 EVT 附近的，另一个是 EVT 附近的。后者在用户最小化框架时触发，所以我们用它来最小化托盘，实际上只是隐藏框架。另一个事件在您关闭框架时触发，它更重要一点。为什么？你需要捕捉关闭事件，以防用户试图通过托盘图标关闭应用程序。你需要确保移除图标并销毁它，否则你的应用看起来会关闭，但实际上只是挂在后台。

### 包扎

现在您知道了如何将 wxPython 应用程序最小化到系统托盘区域。我以前用它做过一个简单的邮件检查程序。你可以用它来做很多其他的事情，比如一个监视器，它通过提升框架来响应事件。

### 附加阅读

*   [wxpython:如何最小化到任务栏](http://bytes.com/topic/python/answers/699757-wxpython-how-minimize-taskbar)
*   wxPython 101: [创建任务栏图标](https://www.blog.pythonlibrary.org/2011/12/13/wxpython-101-creating-taskbar-icons/)