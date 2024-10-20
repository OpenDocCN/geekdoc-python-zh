# Python Lambda

> 原文：<https://www.blog.pythonlibrary.org/2010/07/19/the-python-lambda/>

当我第一次开始学习 Python 时，最让我困惑的概念之一是 lambda 语句。我敢肯定其他新程序员也会被它弄糊涂，你们中的一些人可能想知道我在说什么。所以，本着教育的精神，让我们来个突击测验:

什么是λ？

A.希腊字母中的第 11 个字母
B .颅骨矢状缝和人字缝交界处的颅骨测量点
C .可以让它将用户的想法变为现实的手臂奴隶机甲中的驾驶员
D .一系列日本火箭的名称
E .匿名(未绑定)功能

如果你猜对了以上所有的问题，你就答对了！当然，在这篇文章的上下文中，“E”确实是正确的答案。Python lambda 语句是一个匿名或未绑定的函数，而且是一个非常有限的函数。让我们看几个典型的例子，看看我们是否能找到它的用例。

人们通常看到的讲授 lambda 的典型例子是某种无聊的加倍函数。恰恰相反，我们的简单例子将显示如何找到平方根。首先我们将展示一个普通函数，然后是 lambda 等价函数:

```py

import math

#----------------------------------------------------------------------
def sqroot(x):
    """
    Finds the square root of the number passed in
    """
    return math.sqrt(x)

square_rt = lambda x: math.sqrt(x)

```

如果您尝试这些函数中的每一个，您将得到一个 float。这里有几个例子:

 `>>> sqroot(49)
7.0
>>> square_rt(64)
8.0` 

很圆滑，对吧？但是在现实生活中，我们实际上在哪里使用λ呢？也许是计算器程序？好吧，那是可行的，但是对于 Python 的内置来说，这是一个非常有限的应用！lambda 示例经常应用的 Python 的主要部分之一是 Tkinter 回调。我们将对此进行研究，但是我们也将获取这些信息，并使用 wxPython 进行尝试，看看是否也能在那里工作。

## Tkinter + lambda

我们将从 Tkinter 开始，因为它包含在标准 Python 包中。这是一个非常简单的脚本，有三个按钮，其中两个使用 lambda 绑定到它们的事件处理程序:

```py

import Tkinter as tk

########################################################################
class App:
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        frame = tk.Frame(parent)
        frame.pack()

        btn22 = tk.Button(frame, text="22", command=lambda: self.printNum(22))
        btn22.pack(side=tk.LEFT)
        btn44 = tk.Button(frame, text="44", command=lambda: self.printNum(44))
        btn44.pack(side=tk.LEFT)

        quitBtn = tk.Button(frame, text="QUIT", fg="red", command=frame.quit)
        quitBtn.pack(side=tk.LEFT)

    #----------------------------------------------------------------------
    def printNum(self, num):
        """"""
        print "You pressed the %s button" % num

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

```

注意 btn22 和 btn44 变量。这是行动的地方。我们创建了一个传统知识。按钮实例并绑定到我们的 *printNum* 方法。λ被分配给按钮的*命令*参数。这意味着我们正在为这个命令创建一个一次性的函数，就像在退出按钮中我们调用框架的退出方法一样。这里的区别在于，这个特定的 lambda 是一个调用另一个方法并向后者传递一个整数的方法。在 *printNum* 方法中，我们通过使用从 lambda 函数传递给它的信息，将哪个按钮被按下打印到 stdout。你都明白了吗？如果是这样，我们可以继续...如果没有，根据需要反复阅读这一段，直到信息被理解或者你发疯，无论哪一个先出现。

## wxPython+λ

我们的 wxPython 示例与 Tkinter 示例非常相似，只是更详细一点:

```py

import wx

########################################################################
class DemoFrame(wx.Frame):
    """
    Frame that holds all other widgets
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""        
        wx.Frame.__init__(self, None, wx.ID_ANY, 
                          "wx lambda tutorial",
                          size=(600,400)
                          )
        panel = wx.Panel(self)

        button8 = wx.Button(panel, label="8")
        button8.Bind(wx.EVT_BUTTON, lambda evt, name=button8.GetLabel(): self.onButton(evt, name))
        button10 = wx.Button(panel, label="10")
        button10.Bind(wx.EVT_BUTTON, lambda evt, name=button10.GetLabel(): self.onButton(evt, name))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(button8, 0, wx.ALL, 5)
        sizer.Add(button10, 0, wx.ALL, 5)
        panel.SetSizer(sizer)

    #----------------------------------------------------------------------
    def onButton(self, event, buttonLabel):
        """"""
        print "You pressed the %s button!" % buttonLabel

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = DemoFrame().Show()
    app.MainLoop()

```

在这种情况下，我们用 lambda 语句创建一个双参数匿名函数。第一个参数是 *evt* ，第二个是按钮的标签。这些被传递给 *onButton* 事件处理程序，当用户点击两个按钮中的一个时，这个事件处理程序被调用。此示例还将按钮的标签打印到 stdout。

## 包扎

lambda 语句也用于各种其他项目。如果你在谷歌上搜索一个 Python 项目名和 lambda，你可以找到很多活代码。例如，如果您搜索“django lambda”，您会发现 django 有一个利用 lambdas 的 modelformset 工厂。SqlAlchemy 的 Elixir 插件也使用 lambdas。睁大你的眼睛，你会惊奇地发现有多少次你会偶然发现这个方便的小函数生成器。

## 进一步阅读

*   深入 Python: [使用 lambda 函数](http://diveintopython.org/power_of_introspection/lambda_functions.html)
*   [Effbot](http://effbot.org/zone/tkinter-callbacks.htm) Tkinter lambda 示例
*   wxPython Wiki: [向回调传递参数](http://wiki.wxpython.org/Passing%20Arguments%20to%20Callbacks)