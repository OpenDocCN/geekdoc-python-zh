# ConfigObj + wxPython =极客快乐

> 原文：<https://www.blog.pythonlibrary.org/2010/01/17/configobj-wxpython-geek-happiness/>

我最近开始在工作中使用 [Michael Foord 的](http://www.voidspace.org.uk/) [ConfigObj](http://www.voidspace.org.uk/python/configobj.html) 用于我们内部的 [wxPython](http://www.wxpython.org) 应用程序。当我写我的另一个 ConfigObj [教程](https://www.blog.pythonlibrary.org/2010/01/01/a-brief-configobj-tutorial/)时，我想向你展示我是如何在我的首选项对话框中使用 ConfigObj 的，但我不希望我所有的帖子都包含 wx。在本文中，我将向您展示在 ConfigObj 中添加一个新的首选项设置而不删除原来的设置是多么容易，以及如何用 wxPython 对话框加载和保存它们。现在，让我们开始吧！

首先，我们将创建一个简单的控制器，用于使用 ConfigObj 创建和访问配置文件:

```py

import configobj
import os
import sys
import wx
from wx.lib.buttons import GenBitmapTextButton

appPath = os.path.abspath(os.path.dirname(os.path.join(sys.argv[0])))
inifile = os.path.join(appPath, "example.ini")

########################################################################
class CloseBtn(GenBitmapTextButton):
    """
    Creates a reusuable close button with a bitmap
    """

    #----------------------------------------------------------------------
    def __init__(self, parent, label="Close"):
        """Constructor"""
        font = wx.Font(16, wx.SWISS, wx.NORMAL, wx.BOLD)
        img = wx.Bitmap(r"%s\images\cancel.png" % appPath)
        GenBitmapTextButton.__init__(self, parent, wx.ID_CLOSE, img, 
                                     label=label, size=(110, 50))
        self.SetFont(font)

#----------------------------------------------------------------------
def createConfig():
    """
    Create the configuration file
    """
    config = configobj.ConfigObj()
    config.filename = inifile
    config['update server'] = "http://www.someCoolWebsite/hackery.php"    
    config['username'] = ""
    config['password'] = ""
    config['update interval'] = 2
    config['agency filter'] = 'include'
    config['filters'] = ""
    config.write()

#----------------------------------------------------------------------
def getConfig():
    """
    Open the config file and return a configobj
    """
    if not os.path.exists(inifile):
        createConfig()
    return configobj.ConfigObj(inifile)

```

这段代码非常简单。在 *createConfig* 函数中，它在运行该脚本的目录中创建一个“example.ini”文件。配置文件有六个字段，但没有节。在 *getConfig* 函数中，代码检查配置文件是否存在，如果不存在就创建它。无论如何，该函数都会向调用者返回一个 ConfigObj 对象。我们将把这个脚本放到“controller.py”中。现在我们将 wx 子类化。对话框类来创建首选项对话框。

```py

# -----------------------------------------------------------
# preferencesDlg.py
#
# Created 10/20/2009 by mld
# -----------------------------------------------------------

import controller
import wx
from wx.lib.buttons import GenBitmapTextButton

########################################################################
class PreferencesDialog(wx.Dialog):
    """
    Creates and displays a preferences dialog that allows the user to
    change some settings.
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """
        """
        wx.Dialog.__init__(self, None, wx.ID_ANY, 'Preferences', size=(550,300))
        appPath = controller.appPath

        # ---------------------------------------------------------------------
        # Create widgets
        font = wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD)
        serverLbl = wx.StaticText(self, wx.ID_ANY, "Update Server:")
        self.serverTxt = wx.TextCtrl(self, wx.ID_ANY, "")
        self.serverTxt.Disable()

        usernameLbl = wx.StaticText(self, wx.ID_ANY, "Username:")
        self.usernameTxt = wx.TextCtrl(self, wx.ID_ANY, "")
        self.usernameTxt.Disable()

        passwordLbl = wx.StaticText(self, wx.ID_ANY, "Password:")
        self.passwordTxt = wx.TextCtrl(self, wx.ID_ANY, "", style=wx.TE_PASSWORD)
        self.passwordTxt.Disable()

        updateLbl = wx.StaticText(self, wx.ID_ANY, "Update Interval:")
        self.updateTxt = wx.TextCtrl(self, wx.ID_ANY, "")
        minutesLbl = wx.StaticText(self, wx.ID_ANY, "minutes")

        agencyLbl = wx.StaticText(self, wx.ID_ANY, "Agency Filter:")
        choices = ["Include all agencies except", "Exclude all agencies except"]
        self.agencyCbo = wx.ComboBox(self, wx.ID_ANY, "Include all agencies except",
                                     None, wx.DefaultSize, choices, wx.CB_DROPDOWN|wx.CB_READONLY)
        self.agencyCbo.SetFont(font)
        self.filterTxt = wx.TextCtrl(self, wx.ID_ANY, "")

        img = wx.Bitmap(r"img/filesave.png" % appPath)
        saveBtn = GenBitmapTextButton(self, wx.ID_ANY, img, "Save", size=(110, 50))
        saveBtn.Bind(wx.EVT_BUTTON, self.savePreferences)
        cancelBtn = controller.CloseBtn(self, label="Cancel")
        cancelBtn.Bind(wx.EVT_BUTTON, self.onCancel)

        widgets = [serverLbl, usernameLbl, passwordLbl, updateLbl, agencyLbl, minutesLbl,
                   self.serverTxt, self.usernameTxt, self.passwordTxt, self.updateTxt,
                   self.agencyCbo, self.filterTxt, saveBtn, cancelBtn]
        for widget in widgets:
            widget.SetFont(font)

        # ---------------------------------------------------------------------
        # layout widgets
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        updateSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        prefSizer = wx.FlexGridSizer(cols=2, hgap=5, vgap=5)
        prefSizer.AddGrowableCol(1)

        prefSizer.Add(serverLbl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        prefSizer.Add(self.serverTxt, 0, wx.EXPAND)
        prefSizer.Add(usernameLbl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        prefSizer.Add(self.usernameTxt, 0, wx.EXPAND)
        prefSizer.Add(passwordLbl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        prefSizer.Add(self.passwordTxt, 0, wx.EXPAND)
        prefSizer.Add(updateLbl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        updateSizer.Add(self.updateTxt, 0, wx.RIGHT, 5)
        updateSizer.Add(minutesLbl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        prefSizer.Add(updateSizer)
        prefSizer.Add(agencyLbl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        prefSizer.Add(self.agencyCbo, 0, wx.EXPAND)
        prefSizer.Add((20,20))
        prefSizer.Add(self.filterTxt, 0, wx.EXPAND)

        mainSizer.Add(prefSizer, 0, wx.EXPAND|wx.ALL, 5)
        btnSizer.Add(saveBtn, 0, wx.ALL, 5)
        btnSizer.Add(cancelBtn, 0, wx.ALL, 5)
        mainSizer.Add(btnSizer, 0, wx.ALL | wx.ALIGN_RIGHT, 10)
        self.SetSizer(mainSizer)

        # ---------------------------------------------------------------------
        # load preferences
        self.loadPreferences()

    #----------------------------------------------------------------------
    def loadPreferences(self):
        """
        Load the preferences and fill the text controls
        """
        config = controller.getConfig()
        updateServer = config['update server']
        username = config['username']
        password = config['password']
        interval = config['update interval']
        agencyFilter = config['agency filter']
        filters = config['filters']

        self.serverTxt.SetValue(updateServer)
        self.usernameTxt.SetValue(username)
        self.passwordTxt.SetValue(password)
        self.updateTxt.SetValue(interval)
        self.agencyCbo.SetValue(agencyFilter)
        self.filterTxt.SetValue(filters)

    #----------------------------------------------------------------------
    def onCancel(self, event):
        """
        Closes the dialog
        """
        self.EndModal(0)

    #----------------------------------------------------------------------
    def savePreferences(self, event):
        """
        Save the preferences
        """
        config = controller.getConfig()

        config['update interval'] = self.updateTxt.GetValue()
        config['agency filter'] = str(self.agencyCbo.GetValue())
        data = self.filterTxt.GetValue()
        if "," in data:
            filters = [i.strip() for i in data.split(',')]
        elif " " in data:
            filters = [i.strip() for i in data.split(' ')]
        else:
            filters = [data]
        text = ""
        for f in filters:
            text += " " + f
        text = text.strip()
        config['filters'] = text
        config.write()

        dlg = wx.MessageDialog(self, "Preferences Saved!", 'Information',  
                               wx.OK|wx.ICON_INFORMATION)
        dlg.ShowModal()        
        self.EndModal(0)

if __name__ == "__main__":
    app = wx.PySimpleApp()
    dlg = PreferencesDialog()
    dlg.ShowModal()
    dlg.Destroy()

```

上面的代码创建了一个基本的首选项对话框，并使用 ConfigObj 从一个文件中加载它的配置。您可以通过阅读 *loadPreferences* 方法中的代码来了解其工作原理。我们关心的另一件事是，当用户更改参数时，代码如何保存它们。为此，我们需要看看 *savePreferences* 方法。这是一个非常简单的方法，因为它所做的就是使用 wx 的特定 getter 函数从小部件中获取各种值。还有一个条件，对筛选字段做一些小的检查。主要原因是，在我的原始程序中，我使用空格作为分隔符，程序需要将逗号等转换为空格。这个代码仍然是一个正在进行的工作，虽然它没有涵盖用户可以输入的所有情况。

无论如何，一旦我们在 ConfigObj 的类似 dict 的接口中有了值，我们就把 ConfigObj 实例的数据写到文件中。然后程序显示一个简单的对话框让用户知道已经保存了。

现在，假设我们的程序规范发生了变化，我们需要添加或删除一个偏好。这样做所需要的只是在配置文件中添加或删除它。ConfigObj 将获取更改，我们只需要记住在 GUI 中添加或删除适当的小部件。ConfigObj 最好的一点是，它不会重置文件中的数据，它只会在适当的时候添加更改。试一试，发现它是多么简单！

*注意:所有代码都是在 Windows XP 上用 Python 2.5、ConfigObj 4.6.0 和 Validate 1.0.0 测试的。*

**下载量**

*   [ConfigObj-GUI-Ex.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/01/ConfigObj-GUI-Ex.zip)
*   [ConfigObj-GUI-Ex.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/01/ConfigObj-GUI-Ex.tar)