# wxPython:如何制作“闪烁文本”

> 原文：<https://www.blog.pythonlibrary.org/2012/08/09/wxpython-how-to-make-flashing-text/>

人们不断在 StackOverflow 上提出有趣的 wxPython 问题。今天他们想知道如何在 wxPython 中制作"[闪烁文本](http://stackoverflow.com/q/11849632/393194)。这其实很容易做到。让我们来看看一些简单的代码:

```py

import random
import time
import wx

########################################################################
class MyPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        self.font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.flashingText = wx.StaticText(self, label="I flash a LOT!")
        self.flashingText.SetFont(self.font)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(1000)

    #----------------------------------------------------------------------
    def update(self, event):
        """"""
        now = int(time.time())
        mod = now % 2
        print now
        print mod
        if mod:
            self.flashingText.SetLabel("Current time: %i" % now)
        else:
            self.flashingText.SetLabel("Oops! It's mod zero time!")
        colors = ["blue", "green", "red", "yellow"]
        self.flashingText.SetForegroundColour(random.choice(colors))

########################################################################
class MyFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Flashing text!")
        panel = MyPanel(self)
        self.Show()

#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

```

基本上你需要的就是一个 wx。StaticText 实例和一个 wx.Timer。对于 flash，我们的意思是它会改变颜色，文本本身也会改变。最初提出这个问题的人想知道如何使用 Python 的 time.time()方法显示时间，他们希望消息根据时间除以 2 的模数是否等于零而变化。我知道这看起来有点奇怪，但是我实际上在我自己的一些代码中使用了这个想法。无论如何，这在我使用 Python 2.6.6 和 wxPython 2.8.12.1 的 Windows 7 上工作。

*请注意，有时 SetForegroundColour 方法并不适用于所有平台上的所有小部件，因为原生小部件并不总是允许改变颜色，所以您的收益可能会有所不同。*