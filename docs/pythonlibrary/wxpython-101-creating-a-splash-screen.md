# wxPython 101:创建闪屏

> 原文：<https://www.blog.pythonlibrary.org/2018/09/11/wxpython-101-creating-a-splash-screen/>

你过去经常看到的一个常见的 UI 元素是闪屏。闪屏只是一个对话框，上面有一个标志或图案，有时还会包含一条消息，告诉你应用程序已经加载了多长时间。一些开发人员使用闪屏来告诉用户应用程序正在加载，这样他们就不会多次尝试打开它。

wxPython 支持创建闪屏。在版本 4 之前的 wxPython 版本中，您可以在 **wx 中找到闪屏小部件。闪屏**。然而在 wxPython 的最新版本中，它被移到了 **wx.adv.SplashScreen** 。

让我们来看一个简单的闪屏示例:

```py

import wx
import wx.adv

class MyFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial", size=(500,500))

        bitmap = wx.Bitmap('py_logo.png')
        splash = wx.adv.SplashScreen(
                     bitmap, 
                     wx.adv.SPLASH_CENTER_ON_SCREEN|wx.adv.SPLASH_TIMEOUT, 
                     5000, self)
        splash.Show()

        self.Show()

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

```

这里我们创建了 wx 的一个子类。框架，我们用 wx.Bitmap 加载一个图像。位图实际上并不要求你只加载位图，因为我在这里使用的是 PNG。不管怎样，下一行实例化了我们的闪屏实例。在这里，我们传递给它我们想要显示的位图，一个告诉它如何定位自己的标志，一个以毫秒为单位的启动屏幕应该显示多长时间的超时，以及它的父对象应该是什么。这些都是必需的参数。

闪屏小部件还可以接受另外三个参数:位置、大小和样式。你会注意到，在这个例子中，我们告诉闪屏在屏幕上居中。我们也可以通过 SPLASH_CENTRE_ON_PARENT 告诉它以它的父节点为中心。

当然，您需要修改这个例子来使用您自己的图像。

* * *

### 包扎

如果你有一个需要很长时间加载的应用程序，闪屏实际上是非常有用的。你可以很容易地用它来分散用户的注意力，给人一种你的应用程序即使还没有完全加载也能响应的错觉。试一试，看看你的想法。

* * *

### 相关阅读

*   wxPython 闪屏小工具[文档](https://wxpython.org/Phoenix/docs/html/wx.adv.SplashScreen.html)
*   关于该主题的原始 wxPython [wiki 页面(使用旧的小部件)](https://wiki.wxpython.org/SplashScreen)
*   一个来自闪屏上态度为的[极客的旧教程(也使用旧版本的小部件)](https://geekswithlatitude.readme.io/docs/wxpython-splash-screen)