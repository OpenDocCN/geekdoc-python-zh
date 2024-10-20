# 从文件生成对话框

> 原文：<https://www.blog.pythonlibrary.org/2010/01/20/generating-a-dialog-from-a-file/>

前几天写了一篇关于 wxPython 使用 ConfigObj 的[文章](https://www.blog.pythonlibrary.org/2010/01/17/configobj-wxpython-geek-happiness/)。关于这篇文章，我被问到的第一个问题是关于使用配置文件来生成对话框。我认为这是一个有趣的想法，所以我尝试实现这个功能。我个人认为，使用 XRC 创建对话框并使用 ConfigObj 帮助管理以这种方式加载的对话框文件可能会更好。然而，这对我来说是一个有趣的练习，我想你也会发现它很有启发性。

免责声明:这是一个总的黑客，可能会或可能不会满足您的需求。我给出了各种扩展示例的建议，所以我希望这有所帮助！

既然已经解决了这个问题，让我们创建一个超级简单的配置文件。为了方便起见，我们称它为“config.ini ”:

config . ini

```py

[Labels]
server = Update Server:
username = Username:
password = Password:
update interval = Update Interval:
agency = Agency Filter:
filters = ""

[Values]
server = http://www.someCoolWebsite/hackery.php
username = ""
password = ""
update interval = 2
agency_choices = Include all agencies except, Include all agencies except, Exclude all agencies except
filters = ""

```

这个配置文件有两个部分:标签和值。*标签*部分有我们将用来创建 wx 的标签。StaticText 控件。*值*部分有一些样本值，我们可以将它们用于相应的文本控件和一个组合框。请注意，*机构选择*字段是一个列表。列表中的第一项将是组合框中的默认选项，另外两项是小部件的实际内容。

现在，让我们来看看构建该对话框的代码:

**偏好 sDlg.py**

```py

import configobj
import wx

########################################################################
class PreferencesDialog(wx.Dialog):
    """
    Creates and displays a preferences dialog that allows the user to
    change some settings.
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """
        Initialize the dialog
        """
        wx.Dialog.__init__(self, None, wx.ID_ANY, 'Preferences', size=(550,300))
        self.createWidgets()

    #----------------------------------------------------------------------
    def createWidgets(self):
        """
        Create and layout the widgets in the dialog
        """
        lblSizer = wx.BoxSizer(wx.VERTICAL)
        valueSizer = wx.BoxSizer(wx.VERTICAL)
        btnSizer = wx.StdDialogButtonSizer()
        colSizer = wx.BoxSizer(wx.HORIZONTAL)
        mainSizer = wx.BoxSizer(wx.VERTICAL)

        iniFile = "config.ini"
        self.config = configobj.ConfigObj(iniFile)

        labels = self.config["Labels"]
        values = self.config["Values"]
        self.widgetNames = values
        font = wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD)

        for key in labels:
            value = labels[key]
            lbl = wx.StaticText(self, label=value)
            lbl.SetFont(font)
            lblSizer.Add(lbl, 0, wx.ALL, 5)

        for key in values:
            print key
            value = values[key]
            if isinstance(value, list):
                default = value[0]
                choices = value[1:]
                cbo = wx.ComboBox(self, value=value[0],
                                  size=wx.DefaultSize, choices=choices, 
                                  style=wx.CB_DROPDOWN|wx.CB_READONLY, 
                                  name=key)
                valueSizer.Add(cbo, 0, wx.ALL, 5)
            else:
                txt = wx.TextCtrl(self, value=value, name=key)
                valueSizer.Add(txt, 0, wx.ALL|wx.EXPAND, 5)

        saveBtn = wx.Button(self, wx.ID_OK, label="Save")
        saveBtn.Bind(wx.EVT_BUTTON, self.onSave)
        btnSizer.AddButton(saveBtn)

        cancelBtn = wx.Button(self, wx.ID_CANCEL)
        btnSizer.AddButton(cancelBtn)
        btnSizer.Realize()

        colSizer.Add(lblSizer)
        colSizer.Add(valueSizer, 1, wx.EXPAND)
        mainSizer.Add(colSizer, 0, wx.EXPAND)
        mainSizer.Add(btnSizer, 0, wx.ALL | wx.ALIGN_RIGHT, 5)
        self.SetSizer(mainSizer)

    #----------------------------------------------------------------------
    def onSave(self, event):
        """
        Saves values to disk
        """
        for name in self.widgetNames:
            widget = wx.FindWindowByName(name)
            if isinstance(widget, wx.ComboBox):
                selection = widget.GetValue()
                choices = widget.GetItems()
                choices.insert(0, selection)
                self.widgetNames[name] = choices
            else:
                value = widget.GetValue()
                self.widgetNames[name] = value
        self.config.write()
        self.EndModal(0)

########################################################################
class MyApp(wx.App):
    """"""

    #----------------------------------------------------------------------
    def OnInit(self):
        """Constructor"""
        dlg = PreferencesDialog()
        dlg.ShowModal()
        dlg.Destroy()

        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()

```

首先，我们创建 wx 的子类。对话框及其所有的 *createWidgets* 方法。这个方法将读取我们的配置文件，并使用其中的数据来创建显示。一旦配置被读取，我们循环遍历*标签*部分中的键，并根据需要创建静态文本控件。接下来，我们遍历另一部分中的值，并使用一个条件来检查小部件的类型。在这种情况下，我们只关心 wx。这就是 ConfigObj 有帮助的地方，因为它实际上可以对我们的配置文件中的一些条目进行类型转换。如果您使用 configspec，您可以得到更细粒度的，这可能是您想要扩展本教程的方式。注意，对于文本控件和组合框，我设置了 name 字段。这对保存数据很重要，我们一会儿就会看到。

无论如何，在这两个循环中，我们使用垂直的 BoxSizers 来存放我们的小部件。对于您的专用接口，您可能希望将其替换为 GridBagSizer 或 FlexGridSizer。我个人真的很喜欢 BoxSizers。在 Steven Sproat ( [Whyteboard](https://launchpad.net/whyteboard) )的建议下，我还为按钮使用了 StdDialogButtonSizer。如果您为按钮使用正确的标准 id，这个 sizer 会以跨平台的方式将它们按正确的顺序放置。它相当方便，虽然它不需要很多参数。还要注意，这个 sizer 的[文档](http://www.wxpython.org/docs/api/wx.StdDialogButtonSizer-class.html)暗示您可以指定方向，但实际上您不能。我和 Robin Dunn(wxPython 的创建者)在 IRC 上讨论了这个问题，他说 [epydoc](http://epydoc.sourceforge.net/) 抓取了错误的文档字符串。

我们关心的下一个方法是 *onSave* 。这里是我们保存用户输入内容的地方。在程序的早些时候，我从配置中获取了小部件的名称，现在我们对它们进行循环。我们叫 wx。FindWindowByName 按名称查找小部件。然后我们再次使用 *isinstance* 来检查我们有哪种小部件。完成后，我们使用 GetValue 获取小部件保存的值，并将该值分配给配置中的正确字段。当循环结束时，我们将数据写入磁盘。**立即改进警告:我在这里没有任何验证！这是你要做的事情来扩展这个例子**。最后一步是调用 EndModal(0)关闭对话框，然后关闭应用程序。

现在您已经知道了从配置文件生成对话框的基本知识。我认为使用某种带有小部件类型名称(可能是字符串)的字典可能是让这个脚本与其他小部件一起工作的简单方法。发挥你的想象力，让我知道你想到了什么。

*注意:所有代码都是在 Windows XP 上用 Python 2.5、ConfigObj 4.6.0 和 Validate 1.0.0 测试的。*

**延伸阅读**

*   [ConfigObj 教程](http://www.voidspace.org.uk/python/articles/configobj.shtml)
*   [XRC 和 wxPython](http://wiki.wxpython.org/UsingXmlResources)
*   [wx。对话框

    证明文件

    ](http://www.wxpython.org/docs/api/wx.Dialog-class.html)

**下载量**

*   [dialog_from_config.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/01/dialog_from_config.zip)
*   [dialog_from_config.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/01/dialog_from_config.tar)