# 用 Python 创建 Windows 快捷方式(第二部分)

> 原文：<https://www.blog.pythonlibrary.org/2010/02/25/creating-windows-shortcuts-with-python-part-ii/>

当我上个月第一次写关于用 Python 创建快捷方式的时候，我一直在想我有第三种方法。今天，我不得不维护我的一些快捷方式代码，我又一次偶然发现了它。我还注意到我的帖子收到了 Tim Golden 关于创建快捷方式的另一种方法的评论，所以我也将把它写在这篇帖子里。

奇怪的是，我的另一个方法碰巧使用了蒂姆·戈尔登的一些东西；也就是他的 [winshell 模块](http://timgolden.me.uk/python/winshell.html)。请查看以下内容:

```py

import os
import winshell

program_files = winshell.programs()
winshell.CreateShortcut (
               Path=os.path.join (program_files, 'My Shortcut.lnk'),
               Target=r'C:\Program Files\SomeCoolProgram\prog.exe')

```

让我们把这个打开，这样你就知道发生了什么。首先，我们使用 winshell 找到用户的“程序”文件夹，这是用户的“启动”文件夹的子文件夹。然后我们使用 winshell 的 *CreateShortcut* 方法来实际创建快捷方式。如果您查看 CreateShortcut 的 docstring，您会发现它还可以接受以下关键字参数:arguments、StartIn、Icon 和 Description。

奇怪的是，这不是戈尔登在他的评论中建议的方法。他还有另一个更复杂的想法。让我们快速看一下:

```py

import os, sys
import pythoncom
from win32com.shell import shell, shellcon

shortcut = pythoncom.CoCreateInstance (
  shell.CLSID_ShellLink,
  None,
  pythoncom.CLSCTX_INPROC_SERVER,
  shell.IID_IShellLink
)
program_location = r'C:\Program Files\SomeCoolProgram\prog.exe'
shortcut.SetPath (program_location)
shortcut.SetDescription ("My Program Does Something Really Cool!")
shortcut.SetIconLocation (program_location, 0)

desktop_path = shell.SHGetFolderPath (0, shellcon.CSIDL_DESKTOP, 0, 0)
persist_file = shortcut.QueryInterface (pythoncom.IID_IPersistFile)
persist_file.Save (os.path.join (desktop_path, "My Shortcut.lnk"), 0)

```

在我看来，这对于我的口味来说有点太低级了。然而，这样做也很酷。如果你看看 Tim Golden 的[链接页面](http://timgolden.me.uk/python/win32_how_do_i/create-a-shortcut.html)，你会发现这个例子实际上只是一个复制粘贴的工作。我认为 winshell 的*桌面*方法和“shell”做了同样的事情。SHGetFolderPath (0，shellcon。CSIDL _ 桌面，0，0)”而且更容易阅读。不管怎样，上面的代码做的和前面的代码一样，只是它把快捷方式放在了用户的桌面上。我真的不明白戈尔登在“pythoncom”里干什么。CoCreateInstance”部分，而不是创建一个我们可以用来创建快捷方式的对象。

还要注意的是，Tim Golden 谈到使用与上面详述的方法非常相似的方法来创建 URL 快捷方式。你可以去他的[网站](http://timgolden.me.uk/python/win32_how_do_i/create-a-url-shortcut.html)了解所有有趣的细节。

无论如何，我觉得要完整，我最好写下这另外两个在 Windows 中创建快捷方式的方法。希望你觉得这很有趣，甚至有教育意义。