# 用 Python 锁定窗口

> 原文：<https://www.blog.pythonlibrary.org/2010/02/06/lock-down-windows-with-python/>

大约四年前，我的任务是将一个 Kixtart 脚本转换成 Python。这个特殊的脚本被用来锁定 Windows XP 机器，以便它们可以被用作信息亭。显然，你不需要 Python 来做这些。任何可以访问 Windows 注册表的编程语言都可以做到这一点，或者您可以只使用组策略。但是这是一个 Python 博客，所以这就是你在这篇文章中将要得到的！

## 入门指南

本教程需要的只是标准的 Python 发行版和 [PyWin32 包](http://sourceforge.net/projects/pywin32/files/)。我们将使用的模块是 *_winreg* 和*子进程*，它们内置在标准发行版和 PyWin32 的 *win32api* 和 *win32con* 中。

我们将把代码分成两半来看。前半部分将使用 *_winreg* 模块:

```py

import subprocess, win32con
from win32api import SetFileAttributes
from _winreg import *

# Note: 1 locks the machine, 0 opens the machine.      
UserPolicySetting = 1

# Connect to the correct Windows Registry key and path, then open the key
reg = ConnectRegistry(None, HKEY_CURRENT_USER)
regpath = r"Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer"
key = OpenKey(reg, regpath, 0, KEY_WRITE)

# Edit registry key by adding new values (Lock-down the PC)
SetValueEx(key, "NoRecentDocsMenu", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoRun", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoFavoritesMenu", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoFind", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoSetFolders", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoSetTaskbar", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoSetActiveDesktop", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoWindowsUpdate", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoSMHelp", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoCloseDragDropBands", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoActiveDesktopChanges", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoMovingBands", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoViewContextMenu", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoChangeStartMenu", 0, REG_DWORD, UserPolicySetting)
SetValueEx(key, "NoTrayContextMenu", 0, REG_DWORD, UserPolicySetting)
CloseKey(key)

```

首先，我们导入我们需要的模块。然后，我们连接到 Windows 注册表，打开以下注册表项进行写入:

HKEY _ 当前用户\软件\微软\ Windows \当前版本\策略\资源管理器

**警告:在继续之前，请注意，如果操作不当，篡改 Windows 注册表可能会对您的电脑造成损害。如果你不知道你在做什么，在做任何事情之前备份你的注册表或者使用一个你可以恢复的虚拟机(比如 VirtualBox 或者 VMWare)。上面的脚本将使你的电脑除了 kiosk 之外几乎不能用。**

无论如何，打开密钥后，我们设置了十几个隐藏运行选项、收藏夹、最近和开始菜单中的查找的设置。我们还禁用了桌面上的右键点击、Windows Update 以及对各种菜单的所有更改(等等)。这都是由我们的*用户策略设置*变量控制的。如果它被设置为 1(即布尔真)，它将锁定所有这些值；如果是零，那么它会再次启用所有设置。最后，我们调用*关闭键*来应用设置。

我们应该在这里停一会儿，并考虑这是一些真正丑陋的代码(正如在下面的评论中指出的)。我写这篇文章的时候，我只有一个月的编程经验，我是从一个 Kixtart 脚本移植过来的，这个脚本看起来和这篇一样差，甚至更差。让我们试着让它更干净:

```py

from _winreg import *

UserPolicySetting = 1

# Connect to the correct Windows Registry key and path, then open the key
reg = ConnectRegistry(None, HKEY_CURRENT_USER)
regpath = r"Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer"
key = OpenKey(reg, regpath, 0, KEY_WRITE)

sub_keys = ["NoRecentDocsMenu", "NoRun", "NoFavoritesMenu",
            "NoFind", "NoSetFolders", "NoSetTaskbar",
            "NoSetActiveDesktop", "NoWindowsUpdate",
            "NoSMHelp", "NoCloseDragDropBands",
            "NoActiveDesktopChanges", "NoMovingBands",
            "NoViewContextMenu", "NoChangeStartMenu",
            "NoTrayContextMenu"]

# Edit registry key by adding new values (Lock-down the PC)
for sub_key in sub_keys:
    SetValueEx(key, sub_key, 0, REG_DWORD, UserPolicySetting)

CloseKey(key)

```

这里的主要区别是，我将所有的 sub_key 名称放入一个 Python 列表中，然后我们可以对其进行迭代，并将每个名称设置为正确的值。现在，如果我们在这些代码行中有多处不同，比如混合了 REG_DWORD 和 REG_SZ，那么我们需要做一些不同的事情。例如，我们需要迭代元组列表，或者创建一个函数来传递信息。如果你想要最大的灵活性，你可以创建一个类，除了错误处理和键的打开和关闭之外，还能为你做这些。不过，我将把它留给读者作为练习。

我们脚本的另一半将隐藏各种图标和文件夹:

```py

# Sets the "Hidden" attribute for specified files/folders.
if UserPolicySetting == 1:
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Set Program Access and Defaults.lnk", win32con.FILE_ATTRIBUTE_HIDDEN)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Windows Catalog.lnk", win32con.FILE_ATTRIBUTE_HIDDEN)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Open Office Document.lnk", win32con.FILE_ATTRIBUTE_HIDDEN)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\New Office Document.lnk", win32con.FILE_ATTRIBUTE_HIDDEN)
    # NOTE: Two backslashes are required before the last directory only
    subprocess.Popen('attrib +h "C:\Documents and Settings\All Users\Start Menu\Programs\\vnc"')
    subprocess.Popen('attrib +h "C:\Documents and Settings\All Users\Start Menu\Programs\\Outlook Express"')
    subprocess.Popen('attrib +h "C:\Documents and Settings\All Users\Start Menu\Programs\\Java Web Start"')
    subprocess.Popen('attrib +h "C:\Documents and Settings\All Users\Start Menu\Programs\\Microsoft Office"')
    subprocess.Popen('attrib +h "C:\Documents and Settings\All Users\Start Menu\Programs\\Microsoft SQL Server"')
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Programs\Adobe Reader 6.0.lnk", win32con.FILE_ATTRIBUTE_HIDDEN)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Programs\Windows Media Player.lnk", win32con.FILE_ATTRIBUTE_HIDDEN)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Programs\Windows Movie Maker.lnk", win32con.FILE_ATTRIBUTE_HIDDEN)
else:
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Set Program Access and Defaults.lnk", win32con.FILE_ATTRIBUTE_NORMAL)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Windows Catalog.lnk", win32con.FILE_ATTRIBUTE_NORMAL)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Open Office Document.lnk", win32con.FILE_ATTRIBUTE_NORMAL)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\New Office Document.lnk", win32con.FILE_ATTRIBUTE_NORMAL)
    subprocess.Popen('attrib -h "C:\Documents and Settings\All Users\Start Menu\Programs\\vnc"')
    subprocess.Popen('attrib -h "C:\Documents and Settings\All Users\Start Menu\Programs\\Outlook Express"')
    subprocess.Popen('attrib -h "C:\Documents and Settings\All Users\Start Menu\Programs\\Java Web Start"')
    subprocess.Popen('attrib -h "C:\Documents and Settings\All Users\Start Menu\Programs\\Microsoft Office"')
    subprocess.Popen('attrib -h "C:\Documents and Settings\All Users\Start Menu\Programs\\Microsoft SQL Server"')
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Programs\Adobe Reader 6.0.lnk", win32con.FILE_ATTRIBUTE_NORMAL)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Programs\Windows Media Player.lnk", win32con.FILE_ATTRIBUTE_NORMAL)
    SetFileAttributes("C:\Documents and Settings\All Users\Start Menu\Programs\Windows Movie Maker.lnk", win32con.FILE_ATTRIBUTE_NORMAL)

```

该代码片段隐藏了以下内容

*   通常出现在“开始”菜单中的指向 Microsoft Office 文档的各种链接
*   Windows 目录和设置程序访问和默认值快捷方式
*   “开始”菜单的“程序”子菜单中我们不希望用户访问的各种文件夹，如 VNC、Office、Microsoft SQL Connector 等。
*   程序子菜单中的其他链接

这是通过 win32api 的 SetFileAttributes 和 win32con 的快捷方式文件属性的组合来实现的。使用调用 Windows [attrib 命令](http://www.microsoft.com/resources/documentation/windows/xp/all/proddocs/en-us/attrib.mspx?mfr=true)的子进程来切换文件夹的隐藏状态。

不幸的是，这是另一个重复代码的例子。让我们花点时间来尝试重构它。看起来我们设置的所有内容都在“开始”菜单或它的“程序”子文件夹中。我们可以像前面的例子一样，把这些路径放入一个循环中。我们还可以使用标志来告诉我们何时隐藏或显示文件夹。我们将把整个东西放入一个函数中，这样也更容易重用。让我们看看这是什么样子:

```py

import os
import subprocess
import win32con
from win32api import SetFileAttributes

def toggleStartItems(flag=True):
    items = ["Set Program Access and Defaults.lnk", "Windows Catalog.lnk",
             "Open Office Document.lnk", "New Office Document.lnk",
             "Programs\Adobe Reader 6.0.lnk", "Programs\Windows Media Player.lnk",
             "Programs\Windows Movie Maker.lnk", "Programs\\vnc",
             "Programs\\Outlook Express", "Programs\\Java Web Start",
             "Programs\\Microsoft Office", "Programs\\Microsoft SQL Server"]
    path = r'C:\Documents and Settings\All Users\Start Menu'
    if flag:
        toggle = "+h"
    else:
        toggle = "-h"

    for item in items:
        p = os.path.join(path, item)
        if os.path.isdir(p):
            subprocess.Popen('attrib %s "%s"' % (toggle, p))
        elif flag:
            SetFileAttributes(p, win32con.FILE_ATTRIBUTE_HIDDEN)
        else:
            SetFileAttributes(p, win32con.FILE_ATTRIBUTE_NORMAL)

```

现在，看起来是不是好多了？它还使添加和删除项目变得更加容易。这使得将来的维护更简单，麻烦也更少。

## 包扎

现在你知道如何用 Windows 和 Python 创建你自己的 kiosk 了。我希望这篇文章对你有所帮助。

*注意:这些脚本是在 Windows XP 上使用 Python 2.4+测试的*

**延伸阅读**

*   [官方 _winreg 文档](http://docs.python.org/library/_winreg.html)
*   【PyWin32 官方文档