# 使用 Python 读取 OpenVPN 状态数据(第 1 页，共 3 页)

> 原文：<https://www.blog.pythonlibrary.org/2008/04/03/reading-openvpn-status-data-with-python/>

我正在做一个关于使用 wxPython 和 PyWin32 从正在运行的 OpenVPN 会话中捕获输出的 3 部分系列文章。

我在工作中使用 OpenVPN 连接电脑。我注意到我们当前启动 OpenVPN 的方法是在控制台窗口中，这样就可以监控程序的输出。如果用户碰巧关闭了所述窗口，它将结束 VPN 会话。我认为这是愚蠢的，所以我决定尝试使用 wxPython 包装接口，这样我可以最小化它到系统托盘中，如果我有问题的话，可以根据需要将它带回来检查输出。如果你想跟着做，你需要以下东西:

*   [Python](http://www.python.org)
*   [wxPython](http://www.wxpython.org)
*   [PyWin32](http://sourceforge.net/projects/pywin32/)
*   [开放 VPN](http://openvpn.net)
*   [观看目录脚本](http://timgolden.me.uk/python/win32_how_do_i/watch_directory_for_changes.html)

都拿到了吗？好的。我们继续。首先，创建一个文件夹来存放您的脚本。我们实际上需要一对夫妇来做这件事。

首先，我们要创建一个系统托盘图标。

第一步:选择一个图标(我用的是塔玛林系列中的一个)

步骤 2:一旦有了图标，我们将使用一个名为 img2py 的 wxPython 实用程序，它将把图标或图片转换成 Python 文件。安装 wxPython:\ \ path \ to \ Python 25 \ Lib \ site-packages \ wx-2.8-MSW-unicode \ wx \ tools 后，可以在 Python 文件夹中找到它(根据您的系统需要进行调整)

步骤 3:将图标文件移动到步骤 2 中的目录，并通过单击开始、运行和键入 cmd 打开命令窗口。导航到上面的目录(使用 cd 命令)并运行以下命令:python img 2 py . py-I myicon . ico icon . py

步骤 4:完成后，将 icon.py 文件复制到您创建的保存脚本的文件夹中。这将与一些处理图标化和右键菜单的代码结合在一起。

现在我们将创建系统托盘图标响应鼠标事件所需的逻辑。我在 wxPython 演示中找到了一些代码，它们完成了我所做的大部分工作。所以我复制了一下，稍微修改了一下，适合我的需求。您可以在下面看到最终结果:

```py

import wx
from vpnIcon import getIcon

class VPNIconCtrl(wx.TaskBarIcon):
    TBMENU_RESTORE = wx.NewId()
    TBMENU_CLOSE   = wx.NewId()
    TBMENU_CHANGE  = wx.NewId()

    def __init__(self, frame):
        wx.TaskBarIcon.__init__(self)
        self.frame = frame        

        # Set the image
        tbIcon = getIcon()

        # Give the icon a tooltip
        self.SetIcon(tbIcon, "VPN Status")
        self.imgidx = 1

        # bind some events
        self.Bind(wx.EVT_TASKBAR_LEFT_DCLICK, self.OnTaskBarActivate)
        self.Bind(wx.EVT_MENU, self.OnTaskBarActivate, id=self.TBMENU_RESTORE)
        self.Bind(wx.EVT_MENU, self.OnTaskBarClose, id=self.TBMENU_CLOSE)        

    def CreatePopupMenu(self):
        """
        This method is called by the base class when it needs to popup
        the menu for the default EVT_RIGHT_DOWN event.  Just create
        the menu how you want it and return it from this function,
        the base class takes care of the rest.
        """
        menu = wx.Menu()
        menu.Append(self.TBMENU_RESTORE, "View Status")
        menu.AppendSeparator()
        menu.Append(self.TBMENU_CLOSE, "Close Program")

        return menu

    def OnTaskBarActivate(self, evt):
        if self.frame.IsIconized():
            self.frame.Iconize(False)
        if not self.frame.IsShown():
            self.frame.Show(True)
        self.frame.Raise()

    def OnTaskBarClose(self, evt):
        self.Destroy()
        self.frame.Close()

```

下一次，我们将讨论您需要了解的 win32 代码，在最后一部分，我们将创建 GUI 并将其余部分放在一起。