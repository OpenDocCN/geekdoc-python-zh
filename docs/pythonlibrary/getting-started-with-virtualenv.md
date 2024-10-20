# virtualenv 入门

> 原文：<https://www.blog.pythonlibrary.org/2012/07/17/getting-started-with-virtualenv/>

虚拟环境对于测试软件来说非常方便。在编程界也是如此。Ian Bicking 创建了 virtualenv 项目，这是一个用于创建隔离 Python 环境的工具。你可以使用这些环境来测试你的软件的新版本，你所依赖的软件包的新版本，或者只是作为一个沙箱来测试一些新的软件包。当您不能将文件复制到站点包中时，您也可以使用 virtualenv 作为工作空间，因为它位于共享主机上。当您使用 virtualenv 创建一个虚拟环境时，它会创建一个文件夹，并将 Python 与一个 site-packages 文件夹和几个其他文件夹一起复制到其中。它还安装 pip。一旦你的虚拟环境被激活，就像使用普通的 Python 一样。完成后，您可以删除要清理的文件夹。没有混乱，没有大惊小怪。或者，您可以继续使用它进行开发。

在本文中，我们将花一些时间来了解 virtualenv 以及如何使用它来制作我们自己的魔术。

### 装置

首先你大概需要安装 virtualenv。你可以使用 pip 或 easy_install 来安装它，或者你可以从他们的[网站](http://www.virtualenv.org/en/latest/index.html)下载 **virtualenv.py** 文件，然后这样安装。此时，假设您的 Python 文件夹在系统路径上，您应该能够在命令行上调用 virtualenv

### 创建虚拟环境

用 virtualenv 包创建一个虚拟沙箱是相当容易的。你只需要做以下事情:

```py

python virtualenv.py FOLDER_NAME

```

其中，FOLDER_NAME 是您希望沙盒所在的文件夹的名称。在我的 Windows 7 机器上，我将 C:\Python26\Scripts 添加到我的路径中，这样我就可以只调用 **virtualenv.py FOLDER_NAME** 而不用 Python 部分。如果你没有传递任何东西，那么你会在屏幕上看到一个选项列表。假设我们创建了一个名为**沙盒**的项目。我们如何使用它？我们需要激活它。方法如下:

在 Posix 上你可以做 **source bin/activate** 而在 Windows 上，你可以在命令行上做**\ path \ to \ env \ Scripts \ activate**。让我们实际经历这些步骤。我们将在桌面上创建 sandbox 文件夹，这样您就可以看到一个示例。这是它在我的机器上的样子:

```py

C:\Users\mdriscoll\Desktop>virtualenv sandbox
New python executable in sandbox\Scripts\python.exe
Installing setuptools................done.
Installing pip...................done.

C:\Users\mdriscoll\Desktop>sandbox\Scripts\activate
(sandbox) C:\Users\mdriscoll\Desktop>

```

您将会注意到，一旦您的虚拟环境被激活，您将会看到您的提示更改为包含您创建的文件夹名称的前缀，在本例中是“sandbox”。这让你知道你正在使用你的沙箱。现在，您可以使用 pip 将其他包安装到您的虚拟环境中。完成后，您只需调用 deactivate 脚本来退出环境。

在创建虚拟游乐场时，有几个标志可以传递给 virtualenv，您应该知道。例如，您可以使用 **- system-site-packages** 来继承默认 Python 站点包中的包。如果你想使用 distribute 而不是 setuptools，你可以给 virtualenv 传递 **- distribute** 标志。

virtualenv 还为您提供了一种只安装库，但使用系统 Python 本身来运行它们的方法。根据文档，您只需创建一个特殊的脚本来完成它。你可以在这里阅读更多

还有一个简洁的(实验性的)标志叫做 **-可重定位的**，可以用来使文件夹可重定位。然而，在我写这篇文章的时候，它还不能在 Windows 上运行，所以我无法测试它。

最后，还有一个 **- extra-search-dir** 标志，您可以使用它来保持您的虚拟环境离线。基本上，它允许您在搜索路径中添加一个目录，以便安装 pip 或 easy_install。这样，您就不需要访问互联网来安装软件包。

### 包扎

至此，你应该可以自己使用 virtualenv 了。在这一点上，有几个其他项目值得一提。有道格·赫尔曼的 [virtualenvwrapper](http://www.doughellmann.com/projects/virtualenvwrapper/) 库，它使创建、删除和管理虚拟环境变得更加容易，还有 [zc.buildout](http://www.buildout.org/) ，它可能是最接近 virtualenv 的东西，可以被称为竞争对手。我建议把它们都看看，因为它们可能对你的编程冒险有所帮助。