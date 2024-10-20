# wxPython 和线程

> 原文：<https://www.blog.pythonlibrary.org/2010/05/22/wxpython-and-threads/>

如果你经常使用 Python 中的 GUI，你就会知道有时你需要时不时地执行一些长时间运行的过程。当然，如果您像使用命令行程序那样做，那么您将会大吃一惊。在大多数情况下，你最终会阻塞 GUI 的事件循环，用户会看到你的程序冻结。你能做些什么来避免不幸呢？当然，在另一个线程或进程中启动任务！在本文中，我们将看看如何使用 wxPython 和 Python 的线程模块来实现这一点。

## wxPython 的线程安全方法

在 wxPython 世界中，有三种相关的“线程安全”方法。如果你在更新用户界面时没有使用这三个中的一个，那么你可能会遇到奇怪的问题。有时你的图形用户界面会工作得很好。其他时候，它会莫名其妙地使 Python 崩溃。因此需要线程安全的方法。下面是 wxPython 提供的三种线程安全方法:

*   wx。事件后
*   wx.CallAfter
*   wx.calllater 后期版本

据罗宾·邓恩(wxPython 的创作者)说，wx。CallAfter 使用 wx。向应用程序对象发送事件。应用程序将有一个绑定到该事件的事件处理程序，并在收到事件时根据程序员编写的代码做出反应。我的理解是 wx。CallLater 调用 wx。带有指定时间限制的 CallAfter，以便您可以告诉它在发送事件之前要等待多长时间。

Robin Dunn 还指出，Python 全局解释器锁(GIL)将阻止多个线程同时执行 Python 字节码，这可能会限制程序使用多少 CPU 内核。另一方面，他还说“wxPython 在调用 wx APIs 时释放 GIL，这样其他线程就可以在那时运行”。换句话说，在多核机器上使用线程时，您的收益可能会有所不同。我发现这个讨论既有趣又令人困惑...

无论如何，这对于三个 wx 方法来说意味着。CallLater 是 wx 中最抽象的线程安全方法。CallAfter next 和 wx。PostEvent 是最低级的。在下面的例子中，您将看到如何使用 wx。CallAfter 和 wx。更新您的 wxPython 程序。

## wxPython，线程，wx。CallAfter 和 PubSub

在 wxPython 邮件列表上，您会看到专家告诉其他人使用 wx。CallAfter 和 PubSub 一起从另一个线程与 wxPython 应用程序通信。我可能已经告诉人们这么做了。所以在下面的例子中，这正是我们要做的:

```py

import time
import wx

from threading import Thread
from wx.lib.pubsub import Publisher

########################################################################
class TestThread(Thread):
    """Test Worker Thread Class."""

    #----------------------------------------------------------------------
    def __init__(self):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.start()    # start the thread

    #----------------------------------------------------------------------
    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        for i in range(6):
            time.sleep(10)
            wx.CallAfter(self.postTime, i)
        time.sleep(5)
        wx.CallAfter(Publisher().sendMessage, "update", "Thread finished!")

    #----------------------------------------------------------------------
    def postTime(self, amt):
        """
        Send time to GUI
        """
        amtOfTime = (amt + 1) * 10
        Publisher().sendMessage("update", amtOfTime)

########################################################################
class MyForm(wx.Frame):

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        self.displayLbl = wx.StaticText(panel, label="Amount of time since thread started goes here")
        self.btn = btn = wx.Button(panel, label="Start Thread")

        btn.Bind(wx.EVT_BUTTON, self.onButton)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.displayLbl, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)
        panel.SetSizer(sizer)

        # create a pubsub receiver
        Publisher().subscribe(self.updateDisplay, "update")

    #----------------------------------------------------------------------
    def onButton(self, event):
        """
        Runs the thread
        """
        TestThread()
        self.displayLbl.SetLabel("Thread started!")
        btn = event.GetEventObject()
        btn.Disable()

    #----------------------------------------------------------------------
    def updateDisplay(self, msg):
        """
        Receives data from thread and updates the display
        """
        t = msg.data
        if isinstance(t, int):
            self.displayLbl.SetLabel("Time since thread started: %s seconds" % t)
        else:
            self.displayLbl.SetLabel("%s" % t)
            self.btn.Enable()

#----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

我们将使用 Python 的 *time* 模块来模拟我们的长时间运行过程。然而，请随意将更好的东西放在它的位置上。在一个真实的例子中，我使用一个线程打开 Adobe Reader 并将 PDF 发送到打印机。这可能看起来没什么特别的，但是当我不使用线程时，当文档被发送到打印机时，我的应用程序中的打印按钮会保持按下状态，我的 GUI 会一直挂起，直到这个操作完成。甚至一两秒钟对用户来说都是显而易见的！

无论如何，让我们看看这是如何工作的。在我们的线程类中(如下所示)，我们覆盖了“run”方法，这样它就能做我们想做的事情。这个线程是在我们实例化它时启动的，因为我们在它的 __init__ 方法中有“self.start()”。在“run”方法中，我们在 6 的范围内循环，在迭代之间休息 10 秒，然后使用 wx 更新我们的用户界面。CallAfter 和 PubSub。当循环结束时，我们向应用程序发送一条最终消息，让用户知道发生了什么。

```py

########################################################################
class TestThread(Thread):
    """Test Worker Thread Class."""

    #----------------------------------------------------------------------
    def __init__(self):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.start()    # start the thread

    #----------------------------------------------------------------------
    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        for i in range(6):
            time.sleep(10)
            wx.CallAfter(self.postTime, i)
        time.sleep(5)
        wx.CallAfter(Publisher().sendMessage, "update", "Thread finished!")

    #----------------------------------------------------------------------
    def postTime(self, amt):
        """
        Send time to GUI
        """
        amtOfTime = (amt + 1) * 10
        Publisher().sendMessage("update", amtOfTime)

```

您会注意到，在我们的 wxPython 代码中，我们使用一个按钮事件处理程序来启动线程。我们还禁用了该按钮，这样我们就不会意外地启动额外的线程。如果我们有一堆程序在运行，UI 会随机地说它已经完成了，而实际上并没有完成，这将会非常令人困惑。这对读者来说是一个很好的练习。你可以显示线程的 PID，这样你就知道哪个是哪个了...您可能希望将这些信息输出到一个滚动文本控件，这样您就可以看到各种线程的活动。

这里最后一个有趣的部分可能是 PubSub 接收器及其事件处理程序:

```py

def updateDisplay(self, msg):
    """
    Receives data from thread and updates the display
    """
    t = msg.data
    if isinstance(t, int):
        self.displayLbl.SetLabel("Time since thread started: %s seconds" % t)
    else:
        self.displayLbl.SetLabel("%s" % t)
        self.btn.Enable()

```

看看我们如何从线程中提取消息并使用它来更新我们的显示？我们还使用接收到的数据类型来告诉我们应该向用户显示什么。很酷吧。现在让我们往下一层，看看如何用 wx 来做这件事。改为 PostEvent。

## wx。事件和线程

以下代码基于来自 [wxPython wiki](http://wiki.wxpython.org/LongRunningTasks) 的一个例子。它比 wx 稍微复杂一点。我们刚才看到的 CallAfter 代码，但我有信心我们可以解决它。

```py

import time
import wx

from threading import Thread

# Define notification event for thread completion
EVT_RESULT_ID = wx.NewId()

def EVT_RESULT(win, func):
    """Define Result Event."""
    win.Connect(-1, -1, EVT_RESULT_ID, func)

class ResultEvent(wx.PyEvent):
    """Simple event to carry arbitrary result data."""
    def __init__(self, data):
        """Init Result Event."""
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_RESULT_ID)
        self.data = data

########################################################################
class TestThread(Thread):
    """Test Worker Thread Class."""

    #----------------------------------------------------------------------
    def __init__(self, wxObject):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.wxObject = wxObject
        self.start()    # start the thread

    #----------------------------------------------------------------------
    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        for i in range(6):
            time.sleep(10)
            amtOfTime = (i + 1) * 10
            wx.PostEvent(self.wxObject, ResultEvent(amtOfTime))
        time.sleep(5)
        wx.PostEvent(self.wxObject, ResultEvent("Thread finished!"))

########################################################################
class MyForm(wx.Frame):

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        self.displayLbl = wx.StaticText(panel, label="Amount of time since thread started goes here")
        self.btn = btn = wx.Button(panel, label="Start Thread")

        btn.Bind(wx.EVT_BUTTON, self.onButton)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.displayLbl, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)
        panel.SetSizer(sizer)

        # Set up event handler for any worker thread results
        EVT_RESULT(self, self.updateDisplay)

    #----------------------------------------------------------------------
    def onButton(self, event):
        """
        Runs the thread
        """
        TestThread(self)
        self.displayLbl.SetLabel("Thread started!")
        btn = event.GetEventObject()
        btn.Disable()

    #----------------------------------------------------------------------
    def updateDisplay(self, msg):
        """
        Receives data from thread and updates the display
        """
        t = msg.data
        if isinstance(t, int):
            self.displayLbl.SetLabel("Time since thread started: %s seconds" % t)
        else:
            self.displayLbl.SetLabel("%s" % t)
            self.btn.Enable()

#----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

让我们把它分解一下。对我来说，最令人困惑的是前三部分:

```py

# Define notification event for thread completion
EVT_RESULT_ID = wx.NewId()

def EVT_RESULT(win, func):
    """Define Result Event."""
    win.Connect(-1, -1, EVT_RESULT_ID, func)

class ResultEvent(wx.PyEvent):
    """Simple event to carry arbitrary result data."""
    def __init__(self, data):
        """Init Result Event."""
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_RESULT_ID)
        self.data = data

```

EVT 结果标识是这里的关键。它将线程链接到 wx。PyEvent 和奇怪“EVT 结果”函数。在 wxPython 代码中，我们将一个事件处理程序绑定到 EVT _ 结果函数。这使我们能够使用 wx。将事件发送到我们的自定义事件类 ResultEvent。这是做什么的？它通过发出我们绑定的自定义 EVT 结果，将数据发送到 wxPython 程序。我希望这一切都有意义。

一旦你在脑子里想通了，继续读下去。你准备好了吗？很好！你会注意到我们的 *TestThread* 类和之前的几乎一样，除了我们使用了 wx。将我们的消息发送到 GUI 而不是 PubSub。我们 GUI 的显示更新器中的 API 没有改变。我们仍然只是使用消息的数据属性来提取我们想要的数据。这就是全部了！

## 包扎

希望您现在知道如何在 wxPython 程序中使用基本的线程技术。还有其他几种线程方法，我们没有机会在这里介绍，比如使用 wx。产量或队列。幸运的是，wxPython wiki 很好地涵盖了这些主题，所以如果您对这些方法感兴趣，请务必查看下面的链接。

## 进一步阅读

*   [LongRunningTasks](http://wiki.wxpython.org/LongRunningTasks) 维基页面
*   [非阻塞图形用户界面](http://wiki.wxpython.org/Non-Blocking%20Gui)维基页面
*   [工作线程](http://wiki.wxpython.org/WorkingWithThreads)维基页面
*   [Python:运行 Ping、Traceroute 等](https://www.blog.pythonlibrary.org/2010/06/05/python-running-ping-traceroute-and-more/)

## 下载

*   [wxthreads.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/05/wxthreads.zip)
*   [wxthreads.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/05/wxthreads.tar)