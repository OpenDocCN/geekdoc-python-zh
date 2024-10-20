# wxPython Sizers 教程:使用 GridSizer

> 原文：<https://www.blog.pythonlibrary.org/2008/05/19/wxpython-sizers-tutorial-using-a-gridsizer/>

在我上一篇[文章](https://www.blog.pythonlibrary.org/?p=22)中，我只用 wx 在 wxPython 中创建了一个通用表单。用于自动调整我的部件大小的 BoxSizers。这一次，我将使用一个 wx 来添加到我之前的示例中。GridSizer 显示以下内容:

*   如何右对齐图标和标签
*   如何将标签与文本控件垂直对齐
*   无论标签有多长，如何保持文本控件对齐？

首先，我要感谢来自 [wxPython 用户组](http://wxpython.org/maillist.php)的 Malcolm，他向我提出了改进上一个例子的想法，我还需要感谢 Karsten 告诉我一个解决 Malcolm 右对齐困境的好方法。如你所知，这不是唯一的方法，可能有更好或更简单的方法。任何愚蠢的代码都是我的错。

现在，继续表演！首先要注意的是 wx。我用的 GridSizer。让我们来看看引擎盖下:

```py

gridSizer = wx.GridSizer(rows=4, cols=2, hgap=5, vgap=5)

```

这告诉我们，我正在创建一个小部件，它将有 4 行 2 列，垂直和水平的空白间隔为 5 个像素。您以从右向左的方式向其中添加小部件，这样当您添加的小部件数量等于列数时，它会自动“换行”到下一行。换句话说，如果我添加两个小部件，这就构成了一整行，我添加的下一个小部件将自动转到第二行(以此类推)。

在我们给 wx 添加任何东西之前。不过，我们需要看看我是如何改变我的 wx.BoxSizers 的。比例为 1 的 BoxSizer。这意味着垫片将占据 sizer 中所有剩余的空间。我还为每个 wx 添加了图标和标签。BoxSizer，然后添加 wx。BoxSizer 作为 wx.GridSizer 中的第一项。

```py

inputOneSizer.Add((20,20), proportion=1)

```

在这种情况下，间隔符只是一个元组。你实际上可以使用任何尺寸。元组的第一个元素是宽度，第二个元素是高度。如果我没有将比例设置为 1，那么我可以改变宽度的大小来放置图标和标签。如果您想使用默认高度，请使用-1。这告诉 wxPython 使用缺省值。

你会注意到我已经添加了 wx。当我把我的标签添加到 wx.BoxSizer 时，ALIGN_CENTER_VERTICAL 标志。这是为了强制标签垂直居中，因此看起来它也相对于文本控件居中。在我的一些真实的程序中，我只是把字体放大了一些，以达到一种非常相似的效果。

现在我们添加 wx。BoxSizer 到我的 wx。GridSizer:

```py

gridSizer.Add(inputOneSizer, 0, wx.ALIGN_RIGHT)

```

这里我设置了 wx。ALIGN_RIGHT 使所有的项都“推”向文本控件旁边单元格的不可见墙。当我添加文本控件时，我确保我传递了 wx。扩展标志，使其在调整窗口大小时扩展。

最后，我加上 wx。GridSizer 到我的 topSizer，这是一个垂直 wx。盒子尺寸:

```py

topSizer.Add(gridSizer, 0, wx.ALL|wx.EXPAND, 5) 

```

在这一点上，你应该注意到我告诉它也要扩展。如果我不这样做，那么 topSizer 将阻止它所包含的 Sizer 膨胀。试着拿出那面旗子自己看看。

下面是完整的代码:

```py

import wx

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, title='My Form')

        # Add a panel so it looks correct on all platforms
        self.panel = wx.Panel(self, wx.ID_ANY)

        bmp = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, wx.ART_OTHER, (16, 16))
        titleIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        title = wx.StaticText(self.panel, wx.ID_ANY, 'My Title')

        lblSize = (50, -1)

        bmp = wx.ArtProvider.GetBitmap(wx.ART_TIP, wx.ART_OTHER, (16, 16))
        inputOneIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelOne = wx.StaticText(self.panel, wx.ID_ANY, 'Name')
        inputTxtOne = wx.TextCtrl(self.panel, wx.ID_ANY,'')

        inputTwoIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelTwo = wx.StaticText(self.panel, wx.ID_ANY, 'Address')
        inputTxtTwo = wx.TextCtrl(self.panel, wx.ID_ANY,'')

        inputThreeIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelThree = wx.StaticText(self.panel, wx.ID_ANY, 'Email')
        inputTxtThree = wx.TextCtrl(self.panel, wx.ID_ANY, '')

        inputFourIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelFour = wx.StaticText(self.panel, wx.ID_ANY, 'Phone')
        inputTxtFour = wx.TextCtrl(self.panel, wx.ID_ANY, '')

        okBtn = wx.Button(self.panel, wx.ID_ANY, 'OK')
        cancelBtn = wx.Button(self.panel, wx.ID_ANY, 'Cancel')
        self.Bind(wx.EVT_BUTTON, self.onOK, okBtn)
        self.Bind(wx.EVT_BUTTON, self.onCancel, cancelBtn)

        topSizer        = wx.BoxSizer(wx.VERTICAL)
        titleSizer      = wx.BoxSizer(wx.HORIZONTAL)
        gridSizer       = wx.GridSizer(rows=4, cols=2, hgap=5, vgap=5)
        inputOneSizer   = wx.BoxSizer(wx.HORIZONTAL)
        inputTwoSizer   = wx.BoxSizer(wx.HORIZONTAL)
        inputThreeSizer = wx.BoxSizer(wx.HORIZONTAL)
        inputFourSizer  = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer        = wx.BoxSizer(wx.HORIZONTAL)

        titleSizer.Add(titleIco, 0, wx.ALL, 5)
        titleSizer.Add(title, 0, wx.ALL, 5)

        # each input sizer will contain 3 items
        # A spacer (proportion=1),
        # A bitmap (proportion=0),
        # and a label (proportion=0)
        inputOneSizer.Add((20,-1), proportion=1)  # this is a spacer
        inputOneSizer.Add(inputOneIco, 0, wx.ALL, 5)
        inputOneSizer.Add(labelOne, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5) 

        inputTwoSizer.Add((20,20), 1, wx.EXPAND) # this is a spacer
        inputTwoSizer.Add(inputTwoIco, 0, wx.ALL, 5)
        inputTwoSizer.Add(labelTwo, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        inputThreeSizer.Add((20,20), 1, wx.EXPAND) # this is a spacer
        inputThreeSizer.Add(inputThreeIco, 0, wx.ALL, 5)
        inputThreeSizer.Add(labelThree, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        inputFourSizer.Add((20,20), 1, wx.EXPAND) # this is a spacer
        inputFourSizer.Add(inputFourIco, 0, wx.ALL, 5)
        inputFourSizer.Add(labelFour, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        # Add the 3-item sizer to the gridsizer and
        # Right align the labels and icons
        gridSizer.Add(inputOneSizer, 0, wx.ALIGN_RIGHT)
        # Set the TextCtrl to expand on resize
        gridSizer.Add(inputTxtOne, 0, wx.EXPAND)
        gridSizer.Add(inputTwoSizer, 0, wx.ALIGN_RIGHT)
        gridSizer.Add(inputTxtTwo, 0, wx.EXPAND)
        gridSizer.Add(inputThreeSizer, 0, wx.ALIGN_RIGHT)
        gridSizer.Add(inputTxtThree, 0, wx.EXPAND)
        gridSizer.Add(inputFourSizer, 0, wx.ALIGN_RIGHT)
        gridSizer.Add(inputTxtFour, 0, wx.EXPAND)

        btnSizer.Add(okBtn, 0, wx.ALL, 5)
        btnSizer.Add(cancelBtn, 0, wx.ALL, 5)

        topSizer.Add(titleSizer, 0, wx.CENTER)
        topSizer.Add(wx.StaticLine(self.panel), 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(gridSizer, 0, wx.ALL|wx.EXPAND, 5)        
        topSizer.Add(wx.StaticLine(self.panel), 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(btnSizer, 0, wx.ALL|wx.CENTER, 5)

        # SetSizeHints(minW, minH, maxW, maxH)
        self.SetSizeHints(250,300,500,400)

        self.panel.SetSizer(topSizer)
        topSizer.Fit(self)        

    def onOK(self, event):
        # Do something
        print 'onOK handler'

    def onCancel(self, event):
        self.closeProgram()

    def closeProgram(self):
        self.Close()

# Run the program
if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

今天到此为止。尽情享受吧！

**下载**
wxPython Sizer 教程 2