# wxPython:在组合框或列表框小部件中存储对象

> 原文：<https://www.blog.pythonlibrary.org/2010/12/16/wxpython-storing-object-in-combobox-or-listbox-widgets/>

本周早些时候，wxPython IRC 频道上有一个关于如何在 wx.ListBox 中存储对象的讨论。然后在那天晚些时候， [StackOverflow](http://stackoverflow.com/questions/4433715/how-can-i-store-objects-other-than-strings-in-a-wxpython-combobox) 上有一个关于同样事情的问题，但与 wx.ComboBox 有关。幸运的是，这两个小部件都继承了 wx。ItemContainer 并包含 Append 方法，该方法允许您将对象与这些小部件中的项目相关联。在本文中，您将了解这是如何做到的。

## 向 wx 添加对象。列表框

我们先从列表框开始。让我们直接进入代码，因为我认为这样你会学得更快。

```py

import wx

class Car:
    """"""

    def __init__(self, id, model, make, year):
        """Constructor"""
        self.id = id
        self.model = model
        self.make = make
        self.year = year

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)

        cars = [Car(0, "Ford", "F-150", "2008"),
                Car(1, "Chevrolet", "Camaro", "2010"),
                Car(2, "Nissan", "370Z", "2005")]

        sampleList = []
        self.cb = wx.ComboBox(panel,
                              size=wx.DefaultSize,
                              choices=sampleList)
        self.widgetMaker(self.cb, cars)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.cb, 0, wx.ALL, 5)
        panel.SetSizer(sizer)

    def widgetMaker(self, widget, objects):
        """"""
        for obj in objects:
            widget.Append(obj.make, obj)
        widget.Bind(wx.EVT_COMBOBOX, self.onSelect)

    def onSelect(self, event):
        """"""
        print("You selected: " + self.cb.GetStringSelection())
        obj = self.cb.GetClientData(self.cb.GetSelection())
        text = """
        The object's attributes are:
        %s  %s    %s  %s

        """ % (obj.id, obj.make, obj.model, obj.year)
        print(text)

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

现在，这是如何工作的？让我们花点时间来解开这个例子。首先，我们创建了一个非常简单的汽车类，其中定义了四个属性:id、型号、品牌和年份。然后我们创建一个带有面板和**列表框**小部件的简单框架。如您所见，我们使用 ListBox 继承的 Append 方法添加每个 Car 对象的“make”字符串，然后添加对象本身。这允许我们将列表框中的每一项与一个对象相关联。最后，我们将列表框绑定到 **EVT 列表框**，这样我们就可以知道当我们从小部件中选择一个项目时如何访问该对象。

要了解这是如何完成的，请查看 **onSelect** 方法。这里我们可以看到，我们需要调用 ListBox 的 **GetClientData** 方法，并将当前选择传递给它。这将返回我们之前关联的对象。现在我们可以访问该方法的每个属性。在本例中，我们只是将所有内容输出到 stdout。现在让我们看看如何使用 wx.ComboBox 来完成这个任务。

## 向 wx 添加对象。组合框

wx 的代码。ComboBox 实际上是相同的，所以为了好玩，我们将做一点重构。看一看:

```py

import wx

class Car:
    """"""

    def __init__(self, id, model, make, year):
        """Constructor"""
        self.id = id
        self.model = model
        self.make = make
        self.year = year       

class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)

        cars = [Car(0, "Ford", "F-150", "2008"),
                Car(1, "Chevrolet", "Camaro", "2010"),
                Car(2, "Nissan", "370Z", "2005")]

        sampleList = []
        self.cb = wx.ComboBox(panel,
                              size=wx.DefaultSize,
                              choices=sampleList)
        self.widgetMaker(self.cb, cars)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.cb, 0, wx.ALL, 5)
        panel.SetSizer(sizer)

    def widgetMaker(self, widget, objects):
        """"""
        for obj in objects:
            widget.Append(obj.make, obj)
        widget.Bind(wx.EVT_COMBOBOX, self.onSelect)

    def onSelect(self, event):
        """"""
        print("You selected: " + self.cb.GetStringSelection())
        obj = self.cb.GetClientData(self.cb.GetSelection())
        text = """
        The object's attributes are:
        %s  %s    %s  %s

        """ % (obj.id, obj.make, obj.model, obj.year)
        print(text)

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()

```

在本例中，步骤完全相同。但是我们有多个组合框，为什么要做这种事情呢？那会有很多多余的代码。因此，我们编写了一个名为 **widgetMaker** 的简单助手方法，它将为我们完成追加和事件绑定。我们可以让它构建小部件，将它添加到 sizer 和其他东西中，但在这个例子中我们将保持简单。无论如何，为了让它工作，我们传入 ComboBox 小部件以及我们想要添加到小部件的对象列表。widgetMaker 将为我们把这些对象追加到 ComboBox 中。代码的其余部分是相同的，除了我们需要绑定到的稍微不同的事件。

## 包扎

如您所见，这是一个非常简单的小练习，但是它使您的 GUI 更加健壮。对于数据库应用程序，您可能会这样做。我可以想象自己将它用于 SqlAlchemy 结果集。发挥创造力，我相信你也会发现它的用处。

## 附加阅读

*   wx。项目容器[文档](http://www.wxpython.org/docs/api/wx.ItemContainer-class.html)
*   wx。组合框[文档](http://www.wxpython.org/docs/api/wx.ComboBox-class.html)
*   wx。列表框[文档](http://www.wxpython.org/docs/api/wx.ListBox-class.html)