# wxPython:如何打开第二个窗口/框架

> 原文：<https://www.blog.pythonlibrary.org/2018/10/19/wxpython-how-to-open-a-second-window-frame/>

我经常看到与本文标题相关的问题。如何打开第二个框架/窗口？当我关闭主应用程序时，如何关闭所有的框架？当您第一次学习 wxPython 时，这类问题可能很难找到答案，因为您对框架或术语不够熟悉，不知道如何寻找答案。

希望这篇文章能有所帮助。我们将学习如何打开多个框架，以及如何关闭它们。我们开始吧！

* * *

### 打开多个框架

创建多个帧实际上非常简单。你只需要创建一个 **wx 的子类。Frame** 用于您想要创建的每个新帧。或者如果新的框架看起来相同，那么您只需要多次实例化第二个类。你还需要一个 wx 的子类。主应用程序框架的框架。让我们写一些代码，因为我认为这将使事情更清楚:

```py

import wx

class OtherFrame(wx.Frame):
    """
    Class used for creating frames other than the main one
    """

    def __init__(self, title, parent=None):
        wx.Frame.__init__(self, parent=parent, title=title)
        self.Show()

class MyPanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        btn = wx.Button(self, label='Create New Frame')
        btn.Bind(wx.EVT_BUTTON, self.on_new_frame)
        self.frame_number = 1

    def on_new_frame(self, event):
        title = 'SubFrame {}'.format(self.frame_number)
        frame = OtherFrame(title=title)
        self.frame_number += 1

class MainFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title='Main Frame', size=(800, 600))
        panel = MyPanel(self)
        self.Show()

if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

```

所以我们在这里创建了两个类来继承 wx。框架:**其他框架**和**主机**。OtherFrame 是我们额外使用的框架，而 MainFrame 是我们的主应用程序窗口。我们还创建了一个 **wx 的子类。面板**容纳一个按钮和一个按钮事件处理程序。该按钮将用于创建额外的框架。

试着运行这段代码，你会发现它可以通过多次点击按钮来创建任意数量的额外帧。然而，如果你试图在一个或多个其他框架仍在运行的情况下关闭主应用程序，你会发现该应用程序将继续运行。

让我们学习如何修改这段代码，以便在主框架关闭时关闭所有框架。

* * *

### 关闭所有框架

您可能已经注意到了这一点，但是 MainFrame 类和 OtherFrame 类都设置了父类。MainFrame 类被硬编码为相当微妙的 None，而 OtherFrame 的父类则默认为 None。这实际上是我们问题的关键。当你设置一个 wx。Frame 的父级设置为 None，它将成为没有依赖关系的独立实体。所以它不在乎另一个框架是否关闭。但是，如果我们将 OtherFrame 实例的所有父实例都设置为 MainFrame 实例，那么当我们关闭主框架时，它们也会关闭。

要进行这一更改，我们需要做的就是将 **on_new_frame** 函数更改为以下内容:

```py

def on_new_frame(self, event):
    title = 'SubFrame {}'.format(self.frame_number)
    frame = OtherFrame(title=title, parent=wx.GetTopLevelParent(self))
    self.frame_number += 1

```

这里我们通过使用 wxPython 的 **wx 将父节点设置为大型机。GetTopLevelParent** 函数。另一种方法是修改我的面板的 **__init__** 来保存对父参数的引用:

```py

def __init__(self, parent):
    wx.Panel.__init__(self, parent)

    btn = wx.Button(self, label='Create New Frame')
    btn.Bind(wx.EVT_BUTTON, self.on_new_frame)
    self.frame_number = 1
    self.parent = parent

def on_new_frame(self, event):
    title = 'SubFrame {}'.format(self.frame_number)
    frame = OtherFrame(title=title, parent=self.parent)
    self.frame_number += 1

```

在这种情况下，我们只使用 instance 属性来设置我们的 parent。为了彻底起见，我想提一个我们可以为大型机实例设置父实例的方法，那就是调用 **GetParent()** :

```py

def on_new_frame(self, event):
    title = 'SubFrame {}'.format(self.frame_number)
    frame = OtherFrame(title=title, parent=self.GetParent())
    self.frame_number += 1

```

这里我们只是调用 MyPanel 的 GetParent()方法来获取面板的父面板。这是可行的，因为我的面板只被主机使用。但是，如果我们碰巧在其他框架中使用了 panel 类，事情就会变得混乱，所以我个人更喜欢使用**gettoplevelparant**方法，因为它很大程度上保证了我们将获得正确的小部件。

* * *

### 包扎

使用多个框架应该不难。我希望这篇教程已经向你展示了做起来是多么容易。您也可以使用本文中的概念来创建对话框，因为它们的工作方式与框架非常相似，尽管它们往往是模态的。感谢阅读和快乐编码！