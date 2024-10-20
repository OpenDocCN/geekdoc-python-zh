# 使用 Winshell 或 PyWin32 在 Windows 中创建快捷方式

> 原文：<https://www.blog.pythonlibrary.org/2008/05/16/create-a-shortcut-in-windows-using-winshell-or-pywin32/>

在过去的几天里，我需要一种方法在登录过程中在用户的桌面上创建快捷方式。我有一个适用于大多数捷径的方法，但是我就是不知道如何去做这个。

设置如下:我需要一种方法来创建一个快捷方式，使用特定的 web 浏览器访问特定的网站。我的第一次尝试如下:

```py

import win32com.client
import winshell

userDesktop = winshell.desktop()
shell = win32com.client.Dispatch('WScript.Shell')

shortcut = shell.CreateShortCut(userDesktop + '\\Zimbra Webmail.lnk')
shortcut.Targetpath = r'C:\Program Files\Mozilla Firefox\firefox.exe'
shortcut.Arguments = 'http://mysite.com/auth/preauth.php'
shortcut.WorkingDirectory = r'C:\Program Files\Mozilla Firefox'
shortcut.save()

```

这段代码的问题在于它创建了以下内容作为快捷方式的目标路径:

" C:\ " C:\ Program Files \ Mozilla Firefox \ Firefox . exe " http://mysite.com/auth/preauth.php

我们一会儿将回到解决方案，但是首先我猜想你们中的一些人可能想知道我怎么知道 Mozilla 会在哪里。嗯，在我工作的地方，我们将 Mozilla 放在硬盘上的一个特定位置，如果我的一个登录脚本没有检测到该位置，那么该脚本将自动重新安装该程序。在与 pywin32 邮件列表上的知识渊博的人谈论这个话题时，他们提醒我应该通过注册表获取位置。

蒂姆·罗伯茨指出了这种发现方法:

```py

import _winreg
ffkey = _winreg.OpenKey( _winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Mozilla\\Mozilla Firefox')
ffver = _winreg.QueryValueEx( ffkey, 'CurrentVersion' )[0]
print ffver
ffmainkey = _winreg.OpenKey( ffkey, sub + "\\Main" )
ffpath = _winreg.QueryValueEx( ffmainkey, 'PathToExe' )[0]
_winreg.CloseKey( ffkey )
_winreg.CloseKey( ffmainkey ) 

```

而邓肯·布斯在 [c.l.py](http://groups.google.com/group/comp.lang.python/browse_frm/thread/19fe19cf2fb89dea#) 上指出了这种方法:

```py

import _winreg
print _winreg.QueryValue(_winreg.HKEY_CLASSES_ROOT,
    'FirefoxURL\shell\open\command') 

```

事实证明，有三种方法可以做到这一点，所有这些方法都是相关的。

*   你可以用这个[食谱](http://aspn.activestate.com/ASPN/docs/ActivePython/2.3/pywin32/win32com.shell_and_Windows_Shell_Links.html)，它是加布里埃尔·吉纳林娜在 c.l.py 上和我分享的
*   您可以添加“更改我的代码”以包含参数或
*   你可以使用 Tim Golden 的 winshell 模块来完成这一切。

我们来看看最后两个。首先，我们将修改我的代码。c.l.py 的 Chris 和 pywin32 列表上的 Roger Upole 向我指出了这种添加“Arguments”参数的方法。见下文:

```py

import win32com.client
import winshell

shortcut = shell.CreateShortCut(userDesktop + '\\MyShortcut.lnk')
shortcut.TargetPath = r'Program Files\Mozilla Firefox\firefox.exe'
shortcut.Arguments = r'http://mysite.com/auth/preauth.php'
shortcut.WorkingDirectory = r'C:\Program Files\Mozilla Firefox'
shortcut.save() 

```

最后，Tim Golden 指出他的 [winshell 模块](http://pypi.python.org/pypi/winshell/0.2)可以做我想做的事情。他告诉我 winshell 在 pywin32 列表上包装了“IShellLink
功能”。您可以在下面看到结果:

```py

import os
import winshell

winshell.CreateShortcut (
Path=os.path.join (winshell.desktop (), "Zimbra Monkeys.lnk"),
Target=r"c:\Program Files\Mozilla Firefox\firefox.exe",
Arguments="http://mysite.com/auth/preauth.php",
Description="Open http://localhost with Firefox",
StartIn=r'C:\Program Files\Mozilla Firefox'
)

```

现在，您也有一些新工具可以在项目中使用了。

**附加资源:**

*   蒂姆·戈登的网站
*   [PyWin32 文档](http://aspn.activestate.com/ASPN/docs/ActivePython/2.5/pywin32/PyWin32.html)