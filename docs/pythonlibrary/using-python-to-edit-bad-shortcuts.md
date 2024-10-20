# 使用 Python 编辑错误的快捷方式

> 原文：<https://www.blog.pythonlibrary.org/2010/02/13/using-python-to-edit-bad-shortcuts/>

几周前，我写了一些我们在工作中使用的脚本，用于为 Windows 中的各种程序创建快捷方式。嗯，我们也推出了一些程序的更新，改变了程序的路径，然后我们需要改变用户的快捷方式来匹配。不幸的是，一些用户会更改快捷方式的名称，这使得查找变得困难。Python 使得找到我需要改变的快捷方式变得很容易，在这篇文章中，我将向你展示如何去做。

对于这个脚本，你需要 Python 2.4-3.x、 [PyWin32 包](http://sourceforge.net/projects/pywin32/files/)和蒂姆·戈登的 [winshell 模块](http://timgolden.me.uk/python/winshell.html)。现在，让我们看一下代码:

```py

import glob
import win32com.client
import winshell

paths = glob.glob(winshell.desktop() + "\\*.lnk")

shell = win32com.client.Dispatch("WScript.Shell")
old_program_path = r"\\SomeUNC\path\old.exe"

# find the old links and change their target to the new executable
for path in paths:
    shortcut = shell.CreateShortCut(path)
    target = shortcut.Targetpath
    if target == old_program_path:
        shortcut.Targetpath = r"\\newUNC\path\new.exe"
        shortcut.save()

```

首先，我们导入 glob、 *win32com.client* 和 *winshell* ，然后我们创建一个路径列表，其中包含我们需要更改的快捷方式。在这种情况下，我们只查看用户的桌面，但是您可以修改它以包括其他目录(比如启动)。然后我们创建一个 *shell* 对象，并设置一个变量来保存我们想要找到的旧程序的链接路径。

最后，我们遍历链接路径，并使用 shell 对象来查看快捷方式的属性。我们关心的唯一属性是快捷方式的目标，所以我们获取它并使用条件来检查它是否是正确的。如果不是，那么我们继续循环到下一个链接。如果是正确的，那么我们改变目标指向新的位置。

幸运的是，这种情况不常发生；但是当它出现时，您将知道 Python 会支持您。