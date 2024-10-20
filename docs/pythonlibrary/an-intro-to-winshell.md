# winshell 简介

> 原文：<https://www.blog.pythonlibrary.org/2014/11/25/an-intro-to-winshell/>

今天我们将看看 Tim Golden 的便捷软件包 [winshell](http://winshell.readthedocs.org/en/latest/) 。winshell 软件包允许您在 Windows 上查找特殊文件夹，轻松创建快捷方式，通过“结构化存储”处理元数据，使用 Windows shell 完成文件操作，以及使用 Windows 回收站。

在本文中，我们将重点介绍 winshell 的特殊文件夹、快捷方式和回收站功能。

* * *

### 入门指南

winshell 包依赖于安装了 [PyWin32](http://sourceforge.net/projects/pywin32/files/pywin32/) 。确保你已经安装了。完成后，您可以使用 pip 安装 winshell:

```py

pip install winshell

```

现在您已经安装了 winshell，我们可以继续了。

* * *

### 访问特殊文件夹

winshell 包公开了对 Windows 中特殊文件夹路径的访问。暴露的路径有:

*   应用程序 _ 数据
*   收藏
*   书签(收藏夹的别名)
*   开始菜单
*   程序
*   启动
*   个人 _ 文件夹
*   我的文档(个人文件夹的别名)
*   最近的
*   sendto

让我们看几个例子:

```py

>>> import winshell
>>> winshell.application_data()
'C:\\Users\\mdriscoll\\AppData\\Roaming'
>>> winshell.desktop()
'C:\\Users\\mdriscoll\\Desktop'
>>> winshell.start_menu()
'C:\\Users\\mdriscoll\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu'

```

这是不言自明的。让我们看看另一个相关的方法，叫做**文件夹**。根据文档，它使用*CSIDL 数值常量或相应的名称，例如“appdata”代表 CSIDL_APPDATA，或者“desktop”代表 CSIDL _ 桌面*。让我们看一个简单的例子:

```py

>>> import winshell
>>> winshell.folder("desktop")
'C:\\Users\\mdriscoll\\Desktop'

```

这使用了一些我从未听说过的 Windows 内部机制。您可能需要在 MSDN 上查找 CSIDL 数值常量，以便有效地使用 winshell 的这一部分。否则，我会建议坚持使用前面提到的功能。

* * *

### 使用快捷方式

您可以使用 winshell 来获取有关快捷方式的信息。让我们来看一个例子，看看谷歌浏览器的快捷方式:

```py

>>> import winshell
>>> import os
>>> link_path = os.path.join(winshell.desktop(), "Google Chrome.lnk")
>>> sh = winshell.shortcut(link_path)
>>> sh.path
'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
>>> sh.description
'Access the Internet'
>>> sh.arguments
''

```

这让我们了解一下快捷方式的属性。我们还可以调用很多其他的方法。我建议阅读完整的文档，看看还能做些什么。现在让我们尝试使用 winshell 创建一个快捷方式:

```py

>>> winshell.CreateShortcut(
    Path=os.path.join(winshell.desktop(), "Custom Python.lnk"),
    Target=r"c:\python34\python.exe",
    Icon=(r"c:\python34\python.exe", 0),
    Description="The Python Interpreter")

```

如果你是这个博客的长期读者，你可能还记得几年前我曾经写过关于用 winshell 创建快捷方式的文章。这里的功能与以前并没有什么不同，而且是不言自明的。您可能想看看那篇旧文章，因为它也展示了如何使用 PyWin32 创建快捷方式。

* * *

### winshell 和回收站

您还可以使用 winshell 来访问 Windows 回收站。让我们看看你能做什么:

```py

>>> import winshell
>>> recycle_bin = winshell.recycle_bin()
>>>
>>> # undelete a file
>>> recycle_bin.undelete(filepath)
>>>
>>> # empty the recycle bin
>>> recycle_bin.empty(confirm=False)

```

如果您在同一路径上多次调用 **undelete** 方法，您将会取消删除文件的先前版本，如果适用的话。您也可以通过**清空**方法清空回收箱。还有一些未记录的方法，如**条目**或**文件夹**，它们似乎返回一个生成器对象，我假设您可以迭代该对象以发现当前回收站中的所有内容。

* * *

### 包扎

至此，您应该能够很好地使用 winshell 包了。您刚刚学习了如何使用回收站、读写快捷方式以及获取 Windows 上的特殊文件夹。我希望你喜欢这篇教程，并能很快在你自己的代码中使用它。