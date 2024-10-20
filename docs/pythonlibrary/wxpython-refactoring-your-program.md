# wxPython:重构您的程序

> 原文：<https://www.blog.pythonlibrary.org/2009/11/11/wxpython-refactoring-your-program/>

世界上有很多糟糕的代码。我在本文中的目标是帮助 wxPython 程序员学习如何使他们的应用程序更容易维护和修改。需要注意的是，本文中的内容并不一定是重构程序的所谓“最佳”方式；相反，以下是我从自己的经历中学到的一些东西，并得到了罗宾·邓恩的书 *[wxPython in Action](http://www.amazon.com/wxPython-Action-Noel-Rappin/dp/1932394621/ref=sr_1_1?ie=UTF8&s=books&qid=1257861660&sr=8-1)* 和 wxPython 社区的一些帮助。

为了便于说明，我创建了一个流行的计算器程序的 GUI 框架，计算机科学教授喜欢用它来吸引毫无戒心的新生。代码只创建用户界面。它实际上不会进行任何计算。然而，我在这个网站的[上找到了一些代码，应该很容易适应这个程序。我将把它留给读者作为练习。让我们来看一段粗略的、未重构的代码:](http://www.peterbe.com/plog/calculator-in-python-for-dummies)

```py

import wx

class PyCalc(wx.App):
    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)

    def OnInit(self):
        # create frame here
        self.frame = wx.Frame(None, wx.ID_ANY, title="Calculator")
        panel = wx.Panel(self.frame, wx.ID_ANY)
        self.displayTxt = wx.TextCtrl(panel, wx.ID_ANY, "0", 
                                      size=(155,-1),
                                      style=wx.TE_RIGHT|wx.TE_READONLY)
        size=(35, 35)
        zeroBtn = wx.Button(panel, wx.ID_ANY, "0", size=size)
        oneBtn = wx.Button(panel, wx.ID_ANY, "1", size=size)
        twoBtn = wx.Button(panel, wx.ID_ANY, "2", size=size)
        threeBtn = wx.Button(panel, wx.ID_ANY, "3", size=size)
        fourBtn = wx.Button(panel, wx.ID_ANY, "4", size=size)
        fiveBtn = wx.Button(panel, wx.ID_ANY, "5", size=size)
        sixBtn = wx.Button(panel, wx.ID_ANY, "6", size=size)
        sevenBtn = wx.Button(panel, wx.ID_ANY, "7", size=size)
        eightBtn = wx.Button(panel, wx.ID_ANY, "8", size=size)
        nineBtn = wx.Button(panel, wx.ID_ANY, "9", size=size)
        zeroBtn.Bind(wx.EVT_BUTTON, self.method1)
        oneBtn.Bind(wx.EVT_BUTTON, self.method2)
        twoBtn.Bind(wx.EVT_BUTTON, self.method3)
        threeBtn.Bind(wx.EVT_BUTTON, self.method4)
        fourBtn.Bind(wx.EVT_BUTTON, self.method5)
        fiveBtn.Bind(wx.EVT_BUTTON, self.method6)
        sixBtn.Bind(wx.EVT_BUTTON, self.method7)
        sevenBtn.Bind(wx.EVT_BUTTON, self.method8)
        eightBtn.Bind(wx.EVT_BUTTON, self.method9)
        nineBtn.Bind(wx.EVT_BUTTON, self.method10)
        divBtn = wx.Button(panel, wx.ID_ANY, "/", size=size)
        multiBtn = wx.Button(panel, wx.ID_ANY, "*", size=size)
        subBtn = wx.Button(panel, wx.ID_ANY, "-", size=size)
        addBtn = wx.Button(panel, wx.ID_ANY, "+", size=(35,100))
        equalsBtn = wx.Button(panel, wx.ID_ANY, "Enter", size=(35,100))
        divBtn.Bind(wx.EVT_BUTTON, self.method11)
        multiBtn.Bind(wx.EVT_BUTTON, self.method12)
        addBtn.Bind(wx.EVT_BUTTON, self.method13)
        subBtn.Bind(wx.EVT_BUTTON, self.method14)
        equalsBtn.Bind(wx.EVT_BUTTON, self.method15)
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        masterBtnSizer = wx.BoxSizer(wx.HORIZONTAL)
        vBtnSizer = wx.BoxSizer(wx.VERTICAL)
        numSizer  = wx.GridBagSizer(hgap=5, vgap=5)
        numSizer.Add(divBtn, pos=(0,0), flag=wx.CENTER)
        numSizer.Add(multiBtn, pos=(0,1), flag=wx.CENTER)
        numSizer.Add(subBtn, pos=(0,2), flag=wx.CENTER)
        numSizer.Add(sevenBtn, pos=(1,0), flag=wx.CENTER)
        numSizer.Add(eightBtn, pos=(1,1), flag=wx.CENTER)
        numSizer.Add(nineBtn, pos=(1,2), flag=wx.CENTER)
        numSizer.Add(fourBtn, pos=(2,0), flag=wx.CENTER)
        numSizer.Add(fiveBtn, pos=(2,1), flag=wx.CENTER)
        numSizer.Add(sixBtn, pos=(2,2), flag=wx.CENTER)
        numSizer.Add(oneBtn, pos=(3,0), flag=wx.CENTER)
        numSizer.Add(twoBtn, pos=(3,1), flag=wx.CENTER)
        numSizer.Add(threeBtn, pos=(3,2), flag=wx.CENTER)
        numSizer.Add(zeroBtn, pos=(4,1), flag=wx.CENTER)        
        vBtnSizer.Add(addBtn, 0)
        vBtnSizer.Add(equalsBtn, 0)
        masterBtnSizer.Add(numSizer, 0, wx.ALL, 5)
        masterBtnSizer.Add(vBtnSizer, 0, wx.ALL, 5)
        mainSizer.Add(self.displayTxt, 0, wx.ALL, 5)
        mainSizer.Add(masterBtnSizer)
        panel.SetSizer(mainSizer)
        mainSizer.Fit(self.frame)
        self.frame.Show()
        return True

    def method1(self, event):
        pass

    def method2(self, event):
        pass

    def method3(self, event):
        pass

    def method4(self, event):
        pass

    def method5(self, event):
        pass

    def method6(self, event):
        pass

    def method7(self, event):
        pass

    def method8(self, event):
        pass

    def method9(self, event):
        pass

    def method10(self, event):
        pass

    def method13(self, event):
        pass

    def method14(self, event):
        pass

    def method12(self, event):
        pass

    def method11(self, event):
        pass

    def method15(self, event):
        pass

def main():
    app = PyCalc()
    app.MainLoop()

if __name__ == "__main__":
    main()

```

我把这段代码建立在一些非常讨厌的 VBA 代码的基础上，在过去的几年里，我不得不维护这些代码。这种代码很可能来自为程序员自动生成代码的程序，如 Visual Studio 或 Microsoft Office 中的宏生成器。请注意，函数只是编号，而不是描述性的，许多代码看起来都是一样的。当你看到两行或多行代码看起来相同或似乎有相同的目的时，它们通常符合重构的条件。这种现象的一个术语是“复制粘贴”或[意大利面代码](http://en.wikipedia.org/wiki/Spaghetti_code)(不要与其他与意大利面相关的代码委婉语混淆)。是的，复制粘贴代码是邪恶的！当您需要进行更改时，您需要找到复制代码的每个实例，并对其进行更改。

有鉴于此，让我们开始重构这个烂摊子吧！我认为将框架、应用程序和面板对象分开会使代码更容易处理，所以这是我们首先要做的。通过查看小部件的父控件，我们看到文本控件和所有按钮都使用 panel 作为它们的父控件，所以让我们把它们放在一个类中。我还将把实际的小部件创建和布局放到一个函数中，这个函数可以从 panel 类的 __init__ 中调用(见下文)。

```py

class MainPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.formula = []
        self.currentVal = "0"
        self.previousVal = "0"
        self.operator = None
        self.operatorFlag = False
        self.createAndlayoutWidgets()

    def createAndlayoutWidgets(self):
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        masterBtnSizer = wx.BoxSizer(wx.HORIZONTAL)
        vBtnSizer = wx.BoxSizer(wx.VERTICAL)
        numSizer  = wx.GridBagSizer(hgap=5, vgap=5)

        self.displayTxt = wx.TextCtrl(self, wx.ID_ANY, "0", 
                                      size=(155,-1),
                                      style=wx.TE_RIGHT|wx.TE_READONLY)

        # number buttons
        size=(45, 45)
        zeroBtn = wx.Button(self, wx.ID_ANY, "0", size=size)
        oneBtn = wx.Button(self, wx.ID_ANY, "1", size=size)
        twoBtn = wx.Button(self, wx.ID_ANY, "2", size=size)
        threeBtn = wx.Button(self, wx.ID_ANY, "3", size=size)
        fourBtn = wx.Button(self, wx.ID_ANY, "4", size=size)
        fiveBtn = wx.Button(self, wx.ID_ANY, "5", size=size)
        sixBtn = wx.Button(self, wx.ID_ANY, "6", size=size)
        sevenBtn = wx.Button(self, wx.ID_ANY, "7", size=size)
        eightBtn = wx.Button(self, wx.ID_ANY, "8", size=size)
        nineBtn = wx.Button(self, wx.ID_ANY, "9", size=size)

        numBtnLst = [zeroBtn, oneBtn, twoBtn, threeBtn, fourBtn, fiveBtn,
                     sixBtn, sevenBtn, eightBtn, nineBtn]
        for button in numBtnLst:
            button.Bind(wx.EVT_BUTTON, self.onButton)

        # operator buttons
        divBtn = wx.Button(self, wx.ID_ANY, "/", size=size)
        multiBtn = wx.Button(self, wx.ID_ANY, "*", size=size)
        subBtn = wx.Button(self, wx.ID_ANY, "-", size=size)
        addBtn = wx.Button(self, wx.ID_ANY, "+", size=(45,100))
        equalsBtn = wx.Button(self, wx.ID_ANY, "Enter", size=(45,100))
        equalsBtn.Bind(wx.EVT_BUTTON, self.onCalculate)

        opBtnLst = [divBtn, multiBtn, subBtn, addBtn]
        for button in opBtnLst:
            button.Bind(wx.EVT_BUTTON, self.onOperation)

        numSizer.Add(divBtn, pos=(0,0), flag=wx.CENTER)
        numSizer.Add(multiBtn, pos=(0,1), flag=wx.CENTER)
        numSizer.Add(subBtn, pos=(0,2), flag=wx.CENTER)
        numSizer.Add(sevenBtn, pos=(1,0), flag=wx.CENTER)
        numSizer.Add(eightBtn, pos=(1,1), flag=wx.CENTER)
        numSizer.Add(nineBtn, pos=(1,2), flag=wx.CENTER)
        numSizer.Add(fourBtn, pos=(2,0), flag=wx.CENTER)
        numSizer.Add(fiveBtn, pos=(2,1), flag=wx.CENTER)
        numSizer.Add(sixBtn, pos=(2,2), flag=wx.CENTER)
        numSizer.Add(oneBtn, pos=(3,0), flag=wx.CENTER)
        numSizer.Add(twoBtn, pos=(3,1), flag=wx.CENTER)
        numSizer.Add(threeBtn, pos=(3,2), flag=wx.CENTER)
        numSizer.Add(zeroBtn, pos=(4,1), flag=wx.CENTER)

        vBtnSizer.Add(addBtn, 0)
        vBtnSizer.Add(equalsBtn, 0)

        masterBtnSizer.Add(numSizer, 0, wx.ALL, 5)
        masterBtnSizer.Add(vBtnSizer, 0, wx.ALL, 5)
        mainSizer.Add(self.displayTxt, 0, wx.ALL, 5)
        mainSizer.Add(masterBtnSizer)
        self.SetSizer(mainSizer)
        mainSizer.Fit(self.parent)

```

您会注意到添加了一个函数和一些空白使这部分代码看起来更好。它还使我们能够将这些代码放到一个单独的文件中，如果我们愿意的话，我们可以导入这个文件。这有助于提升整个 MVC 做事方式。如果您注意的话，您会看到我已经将数字按钮绑定到一个处理程序，将操作按钮绑定到另一个处理程序。以下是我采用的方法:

```py

def onOperation(self, event):
    """
    Add an operator to the equation
    """
    print "onOperation handler fired"

def onButton(self, event):
    """
    Keeps the display up to date
    """
    # Get the button object
    buttonObj = event.GetEventObject()
    # Get the label of the button object
    buttonLbl = buttonObj.GetLabel()

def onCalculate(self):
    """
    Calculate the total
    """
    print 'in onCalculate'

```

我在 **onButton** 事件处理程序中添加了一些代码，以便您可以看到如何获得调用它的按钮对象的句柄。否则，这个方法实际上没有任何作用。其他方法只是存根。现在让我们看看框架和应用程序对象代码:

```py

class PyCalcFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, parent=None, id=wx.ID_ANY, 
                          title="Calculator")
        panel = MainPanel(self)

class PyCalc(wx.App):
    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)

    def OnInit(self):
        # create frame here
        frame = PyCalcFrame()
        frame.Show()
        return True

def main():
    app = PyCalc()
    app.MainLoop()

if __name__ == "__main__":
    main()

```

正如你所看到的，它们非常简短扼要。如果你赶时间，那么这一点点重构可能就足够了。然而，我认为我们可以做得更好。向上滚动，再次查看重构后的 panel 类。看到有十几个基本相同的按钮创建行了吗？“sizer 也有很多。把“台词也加进去。那些将是我们的下一个目标！

在今年春天(2009 年)的 wxPython 邮件列表上，有一个关于这个主题的大讨论。我看到了许多有趣的解决方案，但流传最广的是创建某种小部件构建方法。这就是我要告诉你怎么做的。以下是我的限量版:

```py

def onWidgetSetup(self, widget, event, handler, sizer, pos=None, flags=[]):
    """
    Accepts a widget, the widget's default event and its handler,
    the sizer for the widget, the position of the widget inside 
    the sizer (if applicable) and the sizer flags (if applicable)
    """
    widget.Bind(event, handler)        
    if not pos:
        sizer.Add(widget, 0, wx.ALL, 5)
    elif pos and flags:
        sizer.Add(widget, pos=pos, flag=wx.CENTER) 
    else:
        sizer.Add(widget, pos=pos)

    return widget

```

这是一段非常简单的代码，它获取一个小部件，将其绑定到一个事件，并将结果组合添加到一个 sizer 中。这个脚本可以扩展做额外的绑定，嵌套大小，设置字体等等。发挥你的想象力。现在我们可以看看这是如何改变面板类代码的:

```py

class MainPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.formula = []
        self.currentVal = "0"
        self.previousVal = "0"
        self.operator = None
        self.operatorFlag = False
        self.createAndlayoutWidgets()

    def createAndlayoutWidgets(self):
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        masterBtnSizer = wx.BoxSizer(wx.HORIZONTAL)
        vBtnSizer = wx.BoxSizer(wx.VERTICAL)
        numSizer  = wx.GridBagSizer(hgap=5, vgap=5)

        self.displayTxt = wx.TextCtrl(self, wx.ID_ANY, "0", 
                                      size=(155,-1),
                                      style=wx.TE_RIGHT|wx.TE_READONLY)
        # number buttons
        size=(45, 45)
        zeroBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "0", size=size),
                                     wx.EVT_BUTTON, self.onButton, numSizer,
                                     pos=(4,1))
        oneBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "1", size=size),
                                    wx.EVT_BUTTON, self.onButton, numSizer,
                                    pos=(3,0))
        twoBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "2", size=size),
                                    wx.EVT_BUTTON, self.onButton, numSizer,
                                    pos=(3,1))
        threeBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "3", size=size),
                                      wx.EVT_BUTTON, self.onButton, numSizer,
                                      pos=(3,2))
        fourBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "4", size=size),
                                     wx.EVT_BUTTON, self.onButton, numSizer,
                                     pos=(2,0))
        fiveBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "5", size=size),
                                     wx.EVT_BUTTON, self.onButton, numSizer,
                                     pos=(2,1))
        sixBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "6", size=size),
                                    wx.EVT_BUTTON, self.onButton, numSizer,
                                    pos=(2,2))
        sevenBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "7", size=size),
                                      wx.EVT_BUTTON, self.onButton, numSizer,
                                      pos=(1,0))
        eightBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "8", size=size),
                                      wx.EVT_BUTTON, self.onButton, numSizer,
                                      pos=(1,1))
        nineBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "9", size=size),
                                     wx.EVT_BUTTON, self.onButton, numSizer,
                                     pos=(1,2))
        # operator buttons
        divBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "/", size=size),
                                    wx.EVT_BUTTON, self.onOperation, numSizer,
                                    pos=(0,0), flags=wx.CENTER)
        multiBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "*", size=size),
                                      wx.EVT_BUTTON, self.onOperation, numSizer,
                                      pos=(0,1), flags=wx.CENTER)
        subBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "-", size=size),
                                    wx.EVT_BUTTON, self.onOperation, numSizer,
                                    pos=(0,2), flags=wx.CENTER)
        addBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "+", size=(45,100)),
                                    wx.EVT_BUTTON, self.onOperation, vBtnSizer)
        equalsBtn = self.onWidgetSetup(wx.Button(self, wx.ID_ANY, "Enter", size=(45,100)),
                                       wx.EVT_BUTTON, self.onCalculate, vBtnSizer)
        masterBtnSizer.Add(numSizer, 0, wx.ALL, 5)
        masterBtnSizer.Add(vBtnSizer, 0, wx.ALL, 5)
        mainSizer.Add(self.displayTxt, 0, wx.ALL, 5)
        mainSizer.Add(masterBtnSizer)
        self.SetSizer(mainSizer)
        mainSizer.Fit(self.parent)

    def onWidgetSetup(self, widget, event, handler, sizer, pos=None, flags=[]):
        """
        Accepts a widget, the widget's default event and its handler,
        the sizer for the widget, the position of the widget inside 
        the sizer (if applicable) and the sizer flags (if applicable)
        """
        widget.Bind(event, handler)        
        if not pos:
            sizer.Add(widget, 0, wx.ALL, 5)
        elif pos and flags:
            sizer.Add(widget, pos=pos, flag=wx.CENTER) 
        else:
            sizer.Add(widget, pos=pos)

        return widget

```

这段代码最棒的地方在于，它把所有按钮创建的东西都放在了一个方法中，所以我们不必编写“wx。Button()”一遍又一遍。这次迭代还删除了大部分“sizer”。添加“通话。当然，在它的位置上，我们有许多“self.onWidgetSetup()”方法调用。看起来这个还是可以重构的，但是怎么重构呢！？对于我的下一个技巧，我浏览了罗宾·邓恩书中的重构部分，并得出结论，他的想法值得一试。(他毕竟是 wxPython 的创造者。)

在他的书中，他有一个按钮构建器方法，看起来类似于我的小部件构建器，尽管他的要简单得多。他还有一个只返回按钮数据的方法。我已经采纳了这些想法，并将其应用到这个程序中，正如你在我最终的面板代码中看到的:

```py

class MainPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.formula = []
        self.currentVal = "0"
        self.previousVal = "0"
        self.operator = None
        self.operatorFlag = False
        self.createDisplay()

    def createDisplay(self):
        """
        Create the calculator display
        """
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        masterBtnSizer = wx.BoxSizer(wx.HORIZONTAL)
        vBtnSizer = wx.BoxSizer(wx.VERTICAL)
        numSizer  = wx.GridBagSizer(hgap=5, vgap=5)

        self.displayTxt = wx.TextCtrl(self, wx.ID_ANY, "0", 
                                      size=(155,-1),
                                      style=wx.TE_RIGHT|wx.TE_READONLY)

        for eachLabel, eachSize, eachHandler, eachPos in self.buttonData():
            button = self.buildButton(eachLabel, eachSize, eachHandler)
            if eachPos:
                numSizer.Add(button, pos=eachPos, flag=wx.CENTER) 
            else:
                vBtnSizer.Add(button)

        masterBtnSizer.Add(numSizer, 0, wx.ALL, 5)
        masterBtnSizer.Add(vBtnSizer, 0, wx.ALL, 5)
        mainSizer.Add(self.displayTxt, 0, wx.ALL, 5)
        mainSizer.Add(masterBtnSizer)
        self.SetSizer(mainSizer)
        mainSizer.Fit(self.parent)

    def buttonData(self):
        size=(45, 45)
        return (("0", size, self.onButton, (4,1)), 
                ("1", size, self.onButton, (3,0)),
                ("2", size, self.onButton, (3,1)), 
                ("3", size, self.onButton, (3,2)),
                ("4", size, self.onButton, (2,0)), 
                ("5", size, self.onButton, (2,1)),
                ("6", size, self.onButton, (2,2)), 
                ("7", size, self.onButton, (1,0)),
                ("8", size, self.onButton, (1,1)), 
                ("9", size, self.onButton, (1,2)),
                ("/", size, self.onOperation, (0,0)), 
                ("*", size, self.onOperation, (0,1)),
                ("-", size, self.onOperation, (0,2)),
                ("+", (45,100), self.onOperation, None),
                ("Enter", (45,100), self.onCalculate, None))

    def buildButton(self, label, size, handler):
        """
        Builds a button and binds it to an event handler.
        Returns the button object
        """
        button = wx.Button(self, wx.ID_ANY, label, size=size)
        self.Bind(wx.EVT_BUTTON, handler, button)
        return button

```

在这一点上，我们应该退一步，看看这给我们带来了什么。原始代码是 132 行，第一次重构将行数减少到 128 行，第二次将行数增加到 144 行，最后一次将行数减少到 120 行。愤世嫉俗的人可能会说我们只保存了 12 行代码。我不同意。我们最终得到的(不管它是否比原始代码多)是一个更容易维护的代码库。这可以修改，并保持清洁比原来容易得多。

我希望这篇文章已经帮助你看到了如何将你的代码重构为类和方法可以使你的程序更易读，更容易维护——并且与其他贡献者分享，毫无羞耻！

**下载量**

*   [calc.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2009/11/calc.zip)
*   [calc.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2009/11/calc.tar)

**延伸阅读**

*   [wxPython 风格指南](http://wiki.wxpython.org/wxPython%20Style%20Guide)
*   [wxPython 在行动](http://www.manning.com/rappin/)