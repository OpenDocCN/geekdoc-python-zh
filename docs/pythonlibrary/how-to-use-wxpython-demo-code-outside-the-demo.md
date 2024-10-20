# 如何在演示之外使用 wxPython 演示代码

> 原文：<https://www.blog.pythonlibrary.org/2018/01/23/how-to-use-wxpython-demo-code-outside-the-demo/>

有时，有人会问他们如何在演示之外运行来自 wxPython 的演示代码。换句话说，他们想知道如何从演示中提取代码并在自己的程序中运行。我想我很久以前在 [wxPython wiki](https://wiki.wxpython.org/Using%20wxPython%20Demo%20Code) 上写过这个主题，但是我想我也应该在这里写这个主题。

* * *

### 如何处理日志

我经常看到的第一个问题是，演示代码中充满了对某种日志的调用。它总是写入该日志，以帮助开发人员了解不同的事件是如何触发的，或者不同的方法是如何调用的。这一切都很好，但这使得从演示中复制代码变得很困难。让我们从 **wx 中取出代码。ListBox** 演示作为一个例子，看看我们是否可以让它在演示之外工作。下面是演示代码:

```py

import wx

#----------------------------------------------------------------------
# BEGIN Demo Code
class FindPrefixListBox(wx.ListBox):
    def __init__(self, parent, id, pos=wx.DefaultPosition, size=wx.DefaultSize,
                 choices=[], style=0, validator=wx.DefaultValidator):
        wx.ListBox.__init__(self, parent, id, pos, size, choices, style, validator)
        self.typedText = ''
        self.log = parent.log
        self.Bind(wx.EVT_KEY_DOWN, self.OnKey)

    def FindPrefix(self, prefix):
        self.log.WriteText('Looking for prefix: %s\n' % prefix)

        if prefix:
            prefix = prefix.lower()
            length = len(prefix)

            # Changed in 2.5 because ListBox.Number() is no longer supported.
            # ListBox.GetCount() is now the appropriate way to go.
            for x in range(self.GetCount()):
                text = self.GetString(x)
                text = text.lower()

                if text[:length] == prefix:
                    self.log.WriteText('Prefix %s is found.\n' % prefix)
                    return x

        self.log.WriteText('Prefix %s is not found.\n' % prefix)
        return -1

    def OnKey(self, evt):
        key = evt.GetKeyCode()

        if key >= 32 and key <= 127:
            self.typedText = self.typedText + chr(key)
            item = self.FindPrefix(self.typedText)

            if item != -1:
                self.SetSelection(item)

        elif key == wx.WXK_BACK:   # backspace removes one character and backs up
            self.typedText = self.typedText[:-1]

            if not self.typedText:
                self.SetSelection(0)
            else:
                item = self.FindPrefix(self.typedText)

                if item != -1:
                    self.SetSelection(item)
        else:
            self.typedText = ''
            evt.Skip()

    def OnKeyDown(self, evt):
        pass

#---------------------------------------------------------------------------

class TestListBox(wx.Panel):
    def __init__(self, parent, log):
        self.log = log
        wx.Panel.__init__(self, parent, -1)

        sampleList = ['zero', 'one', 'two', 'three', 'four', 'five',
                      'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                      'twelve', 'thirteen', 'fourteen']

        wx.StaticText(self, -1, "This example uses the wx.ListBox control.", (45, 10))
        wx.StaticText(self, -1, "Select one:", (15, 50))
        self.lb1 = wx.ListBox(self, 60, (100, 50), (90, 120), sampleList, wx.LB_SINGLE)
        self.Bind(wx.EVT_LISTBOX, self.EvtListBox, self.lb1)
        self.Bind(wx.EVT_LISTBOX_DCLICK, self.EvtListBoxDClick, self.lb1)
        self.lb1.Bind(wx.EVT_RIGHT_UP, self.EvtRightButton)
        self.lb1.SetSelection(3)
        self.lb1.Append("with data", "This one has data");
        self.lb1.SetClientData(2, "This one has data");

        wx.StaticText(self, -1, "Select many:", (220, 50))
        self.lb2 = wx.ListBox(self, 70, (320, 50), (90, 120), sampleList, wx.LB_EXTENDED)
        self.Bind(wx.EVT_LISTBOX, self.EvtMultiListBox, self.lb2)
        self.lb2.Bind(wx.EVT_RIGHT_UP, self.EvtRightButton)
        self.lb2.SetSelection(0)

        sampleList = sampleList + ['test a', 'test aa', 'test aab',
                                   'test ab', 'test abc', 'test abcc',
                                   'test abcd' ]
        sampleList.sort()
        wx.StaticText(self, -1, "Find Prefix:", (15, 250))
        fp = FindPrefixListBox(self, -1, (100, 250), (90, 120), sampleList, wx.LB_SINGLE)
        fp.SetSelection(0)

    def EvtListBox(self, event):
        self.log.WriteText('EvtListBox: %s, %s, %s\n' %
                           (event.GetString(),
                            event.IsSelection(),
                            event.GetSelection()
                            # event.GetClientData()
                            ))

        lb = event.GetEventObject()
        # data = lb.GetClientData(lb.GetSelection())

        # if data is not None:
            # self.log.WriteText('\tdata: %s\n' % data)

    def EvtListBoxDClick(self, event):
        self.log.WriteText('EvtListBoxDClick: %s\n' % self.lb1.GetSelection())
        self.lb1.Delete(self.lb1.GetSelection())

    def EvtMultiListBox(self, event):
        self.log.WriteText('EvtMultiListBox: %s\n' % str(self.lb2.GetSelections()))

    def EvtRightButton(self, event):
        self.log.WriteText('EvtRightButton: %s\n' % event.GetPosition())

        if event.GetEventObject().GetId() == 70:
            selections = list(self.lb2.GetSelections())
            selections.reverse()

            for index in selections:
                self.lb2.Delete(index)
#----------------------------------------------------------------------
# END Demo Code
#----------------------------------------------------------------------

```

我不打算解释演示代码本身。相反，当我想尝试在演示之外运行它时，我将把重点放在这段代码出现的问题上。在演示的最后有一个 **runTest** 函数，我没有复制它，因为如果你在演示之外复制它，代码不会做任何事情。你看，演示代码有某种包装来使它工作。如果你想使用演示代码，你需要添加你自己的“包装器”。

这段代码呈现的主要问题是许多方法都调用了 **self.log.WriteText** 。您不能从代码中看出 log 对象是什么，但是您知道它有一个 **WriteText** 方法。在演示中，您会注意到，当其中一个方法触发时，WriteText 调用似乎会写入演示底部的文本控件。所以日志必须是一个文本控件！

有许多不同的方法可以解决日志问题。以下是我最喜欢的三个:

*   移除对 self.log.WriteText 的所有调用
*   创建我自己的文本控件并将其传入
*   用 WriteText 方法创建一个简单的类

我在很多场合都选择了第一种，因为这是一种简单的开始方式。但是对于教程来说，这有点无聊，所以我们将选择第三个选项，用 WriteText 方法创建一个类！将以下代码添加到包含上述代码的同一文件中:

```py

#----------------------------------------------------------------------
# Start Your own code here           
class FakeLog:
    """
    The log in the demo is a text control, so just create a class
    with an overridden WriteText function
    """

    def WriteText(self, string):
        print(string)

# Create a frame that can wrap your demo code (works in most cases)

class MyFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title='Listbox demo', 
                          size=(800,600))
        log = FakeLog()
        panel = TestListBox(self, log=log)

        self.Show()

if __name__ == '__main__':
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()

```

这里我们只是用 WriteText 方法创建了一个 **FakeLog** ，它接受一个字符串作为唯一的参数。该方法所做的就是将字符串打印到 stdout。然后我们创建 wx 的一个子类。框架，初始化我们的假日志和演示代码，并显示我们的框架。现在我们有了一个不在演示中的演示代码！如果你愿意，你可以在 [Github](https://gist.github.com/driscollis/88636400caf6f593901982781bddda76) 上获得完整的代码。

* * *

### 其他演示问题

还有一些其他的演示没有遵循与**列表框**演示完全相同的 API。例如，如果你尝试使用我在上面为 **wx 创建的类。按钮**演示，你会发现它的 log 对象调用的是 write()方法而不是 WriteText()方法。在这种情况下，解决方案是显而易见的，因为我们只需要向我们的假日志记录类添加第二个方法:

```py

class FakeLog:
    """
    The log in the demo is a text control, so just create a class
    with an overridden WriteText function
    """

    def WriteText(self, string):
        print(string)

    def write(self, string):
        print(string)

```

现在我们的演示运行代码更加灵活了。然而，当我让我的一个读者测试这段代码时，他们注意到了一个关于 **wx 的问题。ListCtrl** 演示。问题是它导入了一个名为“images”的模块。实际上有几个演示引用了这个模块。你只需要从演示中复制 **images.py** ，并把它放在你正在编写的脚本所在的位置，这样你就可以导入它了。

*注意:我收到一份报告，说 wxPython 4 最新测试版中包含的 **images.py** 文件对他们不适用，他们不得不从旧版本的演示中获取一份副本。我自己没有遇到过这个问题，但请记住这一点。*

* * *

### 包扎

现在，您应该有了让 wxPython 演示中的大多数演示在您自己的代码中工作所需的工具。去抓些代码来试试吧！编码快乐！