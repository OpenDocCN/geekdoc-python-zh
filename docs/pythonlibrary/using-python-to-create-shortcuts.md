# 使用 Python 创建快捷方式

> 原文：<https://www.blog.pythonlibrary.org/2010/01/23/using-python-to-create-shortcuts/>

在我的工作中，我用 Python 编写了大量的系统管理脚本。例如，几乎所有的登录脚本都是用 Python 编写的(其中一些是从 [Kixtart](http://www.kixtart.org/) 移植过来的)。多年来，我一直负责创建新应用程序的快捷方式，这些应用程序需要放在用户的桌面上或开始菜单中，或者两者都放。在本文中，我将向您展示如何完成这项任务。

注意:这是一篇只适用于 Windows 的文章，所以如果你不使用那个操作系统，那么这篇文章可能会让你感到厌烦。见鬼，它可能会这样做！

在 Windows 上你需要做的第一件事就是 Mark Hammond 的 PyWin32 包(又名:Python for Windows extensions)。我还推荐蒂姆·戈尔登的 [winshell](http://timgolden.me.uk/python/winshell.html) 模块，因为它可以更容易地找到特定用户的文件夹。我将在本文中使用 Python 2.5，但据我所知，PyWin32 与 Python 3 兼容，因此本教程也适用于使用 Python 3 的读者。

我必须做的最简单的任务是在用户的桌面上创建一个 URL 链接。除非您需要指定特定的浏览器，否则这是一个非常简单的任务:

```py

import os, winshell

desktop = winshell.desktop()
path = os.path.join(desktop, "myNeatWebsite.url")
target = "http://www.google.com/"

shortcut = file(path, 'w')
shortcut.write('[InternetShortcut]\n')
shortcut.write('URL=%s' % target)
shortcut.close()

```

在上面的代码中，我们导入了 os 和 winshell 模块。我们使用 winshell 来抓取当前用户的桌面路径，然后使用 os.path 的 join()函数来连接路径和 url 快捷方式的名称。因为这只是 Windows，所以您真的不需要这样做。字符串连接也同样有效。请注意，我们需要提供一个“url”扩展名，以便 Windows 知道该做什么。然后我们用之前创建的路径写一个文件，把*【internet shortcut】*写成第一行。在第二行，我们写下 url 目标，然后关闭文件。就是这样！

下一个例子会稍微复杂一点。在其中，我们将使用 PyWin32 包中的 win32com 模块创建一个到 [Media Player Classic](http://mpc-hc.sourceforge.net/) 的快捷方式，这是一个不错的开源媒体播放器。让我们来看看一些代码，这样我们就可以看到如何做这件事:

```py

import os, winshell
from win32com.client import Dispatch

desktop = winshell.desktop()
path = os.path.join(desktop, "Media Player Classic.lnk")
target = r"P:\Media\Media Player Classic\mplayerc.exe"
wDir = r"P:\Media\Media Player Classic"
icon = r"P:\Media\Media Player Classic\mplayerc.exe"

shell = Dispatch('WScript.Shell')
shortcut = shell.CreateShortCut(path)
shortcut.Targetpath = target
shortcut.WorkingDirectory = wDir
shortcut.IconLocation = icon
shortcut.save()

```

这里主要带走的是外壳部分。首先从 *win32com.client* 导入 *Dispatch* ，然后调用 *Dispatch('WScript。获取一个 Shell 对象(类似于),然后用它来创建一个快捷方式对象。一旦有了这些，就可以给快捷方式的属性赋值了，这些属性是 Targetpath、WorkingDirectory 和 IconLocation。注意，WorkingDirectory 对应于普通快捷方式属性对话框中的“开始于”字段。在上面的脚本中，IconLocation 可以是一个图标文件，也可以直接从可执行文件中提取图标。这里有一个小问题，如果你不调用*保存*，图标就不会被创建。在前面的例子中，我们不需要显式地调用 file 对象上的 *close* ，因为 Python 会在脚本完成时为我们处理这些。*

让我们拿这两个例子来做一个可重用的函数，这样我们就可以在任何我们想要的登录脚本中使用它们:

```py

from win32com.client import Dispatch

def createShortcut(path, target='', wDir='', icon=''):    
    ext = path[-3:]
    if ext == 'url':
        shortcut = file(path, 'w')
        shortcut.write('[InternetShortcut]\n')
        shortcut.write('URL=%s' % target)
        shortcut.close()
    else:
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.WorkingDirectory = wDir
        if icon == '':
            pass
        else:
            shortcut.IconLocation = icon
        shortcut.save()

```

我们可以添加的一个明显的改进是使用 *os.path.dirname* 方法从目标中提取工作目录，并消除传递该信息的需要。当然，我见过一些奇怪的快捷方式，它们根本没有指定目标或工作目录！无论如何，我希望这篇文章对您的脚本编写有所帮助。下次见！