# wxPython:设置框架在哪个显示器上

> 原文：<https://www.blog.pythonlibrary.org/2018/06/13/wxpython-set-which-display-the-frame-is-on/>

前几天在 wxPython IRC 频道看到一个有趣的问题。他们询问是否有办法设置他们的应用程序将出现在哪个显示器上。wxPython 的创建者 Robin Dunn 给了提问者一些提示，但我决定继续写一篇关于这个主题的快速教程。

wxPython 工具包实际上包含了这类事情所需的所有部分。第一步是获得组合屏幕尺寸。我的意思是问 wxPython 它认为屏幕的总尺寸是多少。这将是所有显示器的总宽度和高度的总和。你可以通过调用 **wx 得到这个。DisplaySize()** ，返回一个元组。如果你想获得单独的显示分辨率，那么你必须调用 **wx。显示**并传入显示器的索引。因此，如果您有两台显示器，那么第一台显示器的分辨率可以这样获得:

```py

index = 0
display = wx.Display(index)
geo = display.GetGeometry()

```

让我们写一个快速的小应用程序，它有一个按钮，可以切换应用程序的显示。

```py

import wx

class MyPanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.frame = parent

        mainsizer = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        btn = wx.Button(self, label='Switch Display')
        btn.Bind(wx.EVT_BUTTON, self.switch_displays)

        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)
        mainsizer.AddStretchSpacer(prop=1)
        mainsizer.Add(sizer, 0, wx.ALL|wx.CENTER, 5)
        mainsizer.AddStretchSpacer(prop=1)
        self.SetSizer(mainsizer)

    def switch_displays(self, event):
        combined_screen_size = wx.DisplaySize()
        for index in range(wx.Display.GetCount()):
            display = wx.Display(index)
            geo = display.GetGeometry()
            print(geo)

        current_w, current_h = self.frame.GetPosition()

        screen_one = wx.Display(0)
        _, _, screen_one_w, screen_one_h = screen_one.GetGeometry()
        screen_two = wx.Display(1)
        _, _, screen_two_w, screen_two_h = screen_two.GetGeometry()

        if current_w > combined_screen_size[0] / 2:
            # probably on second screen
            self.frame.SetPosition((int(screen_one_w / 2),
                                   int(screen_one_h / 2)))
        else:
            self.frame.SetPosition((int(screen_one_w + (screen_two_w / 2)),
                                   int(screen_two_h / 2)))

class MainFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title='Display Change')
        panel = MyPanel(self)
        self.Show()

if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

```

这里我们创建了两个类。第一个包含了几乎所有的代码，并定义了按钮及其事件处理程序。另一个类用于创建框架本身。不过，事件处理程序才是最有趣的地方，所以让我们来看看。作为背景，我碰巧有两台相同品牌、型号和方向的显示器。

```py

def switch_displays(self, event):
    combined_screen_size = wx.DisplaySize()
    for index in range(wx.Display.GetCount()):
        display = wx.Display(index)
        geo = display.GetGeometry()
        print(geo)

    x, y = self.frame.GetPosition()

    screen_one = wx.Display(0)
    _, _, screen_one_w, screen_one_h = screen_one.GetGeometry()
    screen_two = wx.Display(1)
    _, _, screen_two_w, screen_two_h = screen_two.GetGeometry()

    if x > combined_screen_size[0] / 2:
        # probably on second screen
        self.frame.SetPosition((int(screen_one_w / 2),
                                    int(screen_one_h / 2)))
    else:
        self.frame.SetPosition((int(screen_one_w + (screen_two_w / 2)),
                                    int(screen_two_h / 2)))

```

这里我们得出两个显示器的总分辨率。然后为了演示的目的，我们循环显示并打印出它们的几何图形。您可以将这些行注释掉，因为它们除了有助于调试之外什么也不做。

然后，我们通过调用它的 **GetPosition** 方法来获取框架的当前位置。接下来，我们通过调用每个显示对象的 **GetGeometry** 方法来提取两个显示器的分辨率。接下来，我们检查框架的 X 坐标是否大于显示器的组合宽度除以 2。因为我知道我的两个显示器都是相同的分辨率和方向，我知道这将工作。无论如何，如果它更大，那么我们通过调用 **SetPosition** 来尝试将应用程序移动到对面的监视器。

* * *

### 包扎

您应该尝试一下这个代码，看看它是否能在您的多显示器设置上工作。如果没有，你可能需要调整一下算法，或者试着找出你的操作系统认为你的显示器在哪里，这样你就可以相应地修改代码。

* * *

### 附加阅读

*   wx 上的 wxPython 文档页面。显示