# wxPython:如何创建通用向导

> 原文：<https://www.blog.pythonlibrary.org/2012/07/12/wxpython-how-to-create-a-generic-wizard/>

前几天在 StackOverflow 上，我看到有人正在努力使用 wxPython 的向导小部件。当涉及到按钮时，向导不允许太多的定制，所以我决定看看编写自己的向导有多难。这段代码非常有限，但这是我的第一个测试版本:

```py

import wx

########################################################################
class WizardPage(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        if title:
            title = wx.StaticText(self, -1, title)
            title.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
            sizer.Add(title, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
            sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND|wx.ALL, 5)

########################################################################
class WizardPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        self.pages = []
        self.page_num = 0

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.panelSizer = wx.BoxSizer(wx.VERTICAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)

        # add prev/next buttons
        self.prevBtn = wx.Button(self, label="Previous")
        self.prevBtn.Bind(wx.EVT_BUTTON, self.onPrev)
        btnSizer.Add(self.prevBtn, 0, wx.ALL|wx.ALIGN_RIGHT, 5)

        self.nextBtn = wx.Button(self, label="Next")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        btnSizer.Add(self.nextBtn, 0, wx.ALL|wx.ALIGN_RIGHT, 5)

        # finish layout
        self.mainSizer.Add(self.panelSizer, 1, wx.EXPAND)
        self.mainSizer.Add(btnSizer, 0, wx.ALIGN_RIGHT)
        self.SetSizer(self.mainSizer)

    #----------------------------------------------------------------------
    def addPage(self, title=None):
        """"""
        panel = WizardPage(self, title)
        self.panelSizer.Add(panel, 2, wx.EXPAND)
        self.pages.append(panel)
        if len(self.pages) > 1:
            # hide all panels after the first one
            panel.Hide()
            self.Layout()

    #----------------------------------------------------------------------
    def onNext(self, event):
        """"""
        pageCount = len(self.pages)
        if pageCount-1 != self.page_num:
            self.pages[self.page_num].Hide()
            self.page_num += 1
            self.pages[self.page_num].Show()
            self.panelSizer.Layout()
        else:
            print "End of pages!"

        if self.nextBtn.GetLabel() == "Finish":
            # close the app
            self.GetParent().Close()

        if pageCount == self.page_num+1:
            # change label
            self.nextBtn.SetLabel("Finish")

    #----------------------------------------------------------------------
    def onPrev(self, event):
        """"""
        pageCount = len(self.pages)
        if self.page_num-1 != -1:
            self.pages[self.page_num].Hide()
            self.page_num -= 1
            self.pages[self.page_num].Show()
            self.panelSizer.Layout()
        else:
            print "You're already on the first page!"

########################################################################
class MainFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Generic Wizard", size=(800,600))

        self.panel = WizardPanel(self)
        self.panel.addPage("Page 1")
        self.panel.addPage("Page 2")
        self.panel.addPage("Page 3")

        self.Show()

if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

```

主框架实例化了我们的主面板(WizardPanel)。这里是我们大部分代码的位置。它毕竟控制着向导页面的来回分页。您可以随意定义向导页面。事实上，我在第二个版本中可能会做的是这样做，这样我就可以传入一个我自己制作的面板类，因为只使用我想出的简单页面真的很有限。总之，我添加了 3 页，然后我有一些检查来遍历它们。我希望其他人也会对此感兴趣。玩得开心！

### 进一步阅读

*   wxPython: [一个向导教程](https://www.blog.pythonlibrary.org/2011/01/27/wxpython-a-wizard-tutorial/)