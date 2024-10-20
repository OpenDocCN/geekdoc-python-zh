# wxPython:将多个小部件绑定到同一个处理程序

> 原文：<https://www.blog.pythonlibrary.org/2011/09/20/wxpython-binding-multiple-widgets-to-the-same-handler/>

如果您已经在 wxPython 社区呆了几个月以上，您可能会认识到以下问题:“如何将多个按钮绑定到同一个事件处理程序，并让它们做不同的事情？”那么，这篇文章将告诉你如何做到这一点。

*注意:这篇文章基于这篇博客上一篇关于按钮的文章中的一些代码！*

## 我们开始吧

首先，我们需要编写一些实际包含多个按钮的代码。我们将通过一个例子来展示获取按钮对象的两种不同方法，这样你就可以根据需要操作你的程序。这是您一直在等待的代码:

```py

import wx

########################################################################
class MyForm(wx.Frame):

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Button Tutorial")
        panel = wx.Panel(self, wx.ID_ANY)

        sizer = wx.BoxSizer(wx.VERTICAL)
        buttonOne = wx.Button(panel, label="One", name="one")
        buttonTwo = wx.Button(panel, label="Two", name="two")
        buttonThree = wx.Button(panel, label="Three", name="three")
        buttons = [buttonOne, buttonTwo, buttonThree]

        for button in buttons:
            self.buildButtons(button, sizer)

        panel.SetSizer(sizer)

    #----------------------------------------------------------------------
    def buildButtons(self, btn, sizer):
        """"""
        btn.Bind(wx.EVT_BUTTON, self.onButton)
        sizer.Add(btn, 0, wx.ALL, 5)

    #----------------------------------------------------------------------
    def onButton(self, event):
        """
        This method is fired when its corresponding button is pressed
        """
        button = event.GetEventObject()
        print "The button you pressed was labeled: " + button.GetLabel()
        print "The button's name is " + button.GetName()

        button_id = event.GetId()
        button_by_id = self.FindWindowById(button_id)
        print "The button you pressed was labeled: " + button_by_id.GetLabel()
        print "The button's name is " + button_by_id.GetName()

#----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

首先，我们创建三个按钮对象。然后，为了使事情不那么混乱，我们将它们放入一个列表中，并遍历列表，将按钮添加到一个 sizer 中，并将它们绑定到一个事件处理程序。这是一个减少杂乱代码(即复制和粘贴代码)的好方法，使代码更整洁，更容易调试。有些人继续前进，创建一些精心制作的助手方法，如 **buildButtons** 可以处理其他小部件，并且更加灵活。

不过，我们真正关心的是事件处理程序本身。在事件处理程序中获取小部件最简单的方法是调用事件对象的 **GetEventObject** ()方法。这将返回小部件，然后你可以做任何你喜欢的事情。有些人会更改小部件的值或标签，其他人会使用小部件 ID 或唯一名称，并设置一些条件结构，以便在按下该按钮时执行某些操作，而在按下不同的按钮时执行其他操作。功能由您决定。

获取小部件的第二种方法是一个两步过程，我们需要使用事件的 **GetID** ()方法从事件中提取 ID。然后我们将结果传递给我们的框架对象的 **FindWindowById** ()方法，我们又一次有了这个小部件。

## 包扎

现在您知道了将多个小部件绑定到同一个事件处理程序的“秘密”。勇往直前，像没有明天一样编码，创造出令人惊叹的东西！代码可以在新博客的 Mercurial [库](https://bitbucket.org/driscollis/mousevspython/overview)上下载。

## 额外资源

*   [wxPython:按钮之旅(第 1 部分，共 2 部分)](https://www.blog.pythonlibrary.org/2010/06/09/wxpython-a-tour-of-buttons-part-1-of-2/)
*   [自我。Bind vs. self.button.Bind](http://wiki.wxpython.org/self.Bind%20vs.%20self.button.Bind)
*   wx。按钮[文档](http://www.wxpython.org/docs/api/wx.Button-class.html)
*   在 [Youtube](http://www.youtube.com/watch?v=cp1ZeMisTNo) 上创建按钮教程