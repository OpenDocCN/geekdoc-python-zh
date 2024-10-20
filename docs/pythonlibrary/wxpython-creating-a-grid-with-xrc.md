# wxPython:用 XRC 创建网格

> 原文：<https://www.blog.pythonlibrary.org/2013/07/24/wxpython-creating-a-grid-with-xrc/>

我最近试图帮助某人(在 [wxPython 邮件列表](https://groups.google.com/forum/?fromgroups=#!topic/wxpython-users/IfjW9f7LEhQ)上)弄清楚如何通过 XRC 使用网格小部件(wx.grid.Grid)。这应该很简单，但是如果您运行下面的代码，您会发现一个奇怪的问题:

```py

import wx
from wx import xrc

########################################################################
class MyApp(wx.App):
    def OnInit(self):
        self.res = xrc.XmlResource("grid.xrc")

        frame = self.res.LoadFrame(None, 'MyFrame')
        panel = xrc.XRCCTRL(frame, "MyPanel")
        grid = xrc.XRCCTRL(panel, "MyGrid")
        print type(grid)
        grid.CreateGrid(25, 6)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(grid, 1, wx.EXPAND|wx.ALL, 5)

        panel.SetSizer(sizer)

        frame.Show()
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()

```

您会注意到，当您运行这个命令时，输出的类型是“wx”。_windows。ScrolledWindow”，不是网格对象。因此，您将得到以下回溯结果:

```py

AttributeError: 'ScrolledWindow' object has no attribute 'CreateGrid'
File "c:\Users\mdriscoll\Desktop\xrcGridDemo.py", line 26, in app = MyApp(False)
File "C:\Python26\Lib\site-packages\wx-2.8-msw-unicode\wx\_core.py", line 7981, in __init__
  self._BootstrapApp()
File "C:\Python26\Lib\site-packages\wx-2.8-msw-unicode\wx\_core.py", line 7555, in _BootstrapApp
  return _core_.PyApp__BootstrapApp(*args, **kwargs)
File "c:\Users\mdriscoll\Desktop\xrcGridDemo.py", line 14, in OnInit
  grid.CreateGrid(25, 6) 
```

现在你可能想知道 XRC 文件里有什么，以下是它的内容:

 `<resource class=""><object class="wxFrame" name="MyFrame">如您所见，您应该会得到一个 wxGrid。有什么解决办法？需要导入 wx.grid！更多信息见此[线程](http://wxpython-users.1045709.n5.nabble.com/xrc-wxGrid-problems-fetching-widget-using-XRCCTRL-td2363160.html)。根据 wxPython 的创建者 Robin Dunn 的说法，你需要这么做的原因如下:

*您需要在 python 代码中导入 wx.grid。当你这样做的时候，
一些内部数据结构被更新为
网格类的类型信息，并且这个信息被用于计算如何将一个
C++指针转换为一个正确类型的 Python 对象，用于 XRCCTRL 返回
值。*

因此，更新后的代码如下所示:

```py

import wx
import wx.grid
from wx import xrc

########################################################################
class MyApp(wx.App):
    def OnInit(self):
        self.res = xrc.XmlResource("grid.xrc")

        frame = self.res.LoadFrame(None, 'MyFrame')
        panel = xrc.XRCCTRL(frame, "MyPanel")
        grid = xrc.XRCCTRL(panel, "MyGrid")
        print type(grid)
        grid.CreateGrid(25, 6)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(grid, 1, wx.EXPAND|wx.ALL, 5)

        panel.SetSizer(sizer)

        frame.Show()
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()

```

现在，如果你遇到这个奇怪的问题，你也会知道该怎么做。

### 相关文章

*   wxPython:[XRC 简介](https://www.blog.pythonlibrary.org/2010/05/11/wxpython-an-introduction-to-xrc/)
*   wxPython: [一个 XRCed 教程](https://www.blog.pythonlibrary.org/2010/10/28/wxpython-an-xrced-tutorial/)