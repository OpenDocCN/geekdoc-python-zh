# Python 101: easy_install 或如何创建 eggs

> 原文：<https://www.blog.pythonlibrary.org/2012/07/12/python-101-easy_install-or-how-to-create-eggs/>

今天我们将看看有争议的安装 Python 模块和包的 **easy_install** 方法。我们还将学习如何创建我们自己的*。鸡蛋文件。你需要去拿 [SetupTools 包](http://pypi.python.org/pypi/setuptools/)来跟进。这个包不支持 Python 3.x，所以如果你需要，请查看 [pip](http://pypi.python.org/pypi/pip/) 或 [distribute](https://pypi.python.org/pypi/distribute/0.7.3) 。将来会有关于这些项目的文章。现在，我们将从 SetupTools 和 easy_install 开始。

为什么会有争议？我不完全确定，但人们不喜欢它部分安装软件包的方式，因为它不会等待下载完成。我还听说作者对更新它不太感兴趣，但也不允许任何人更新它。参见本文末尾的 Ziade 文章。

SetupTools 是通过命令行从 PyPI 和其他来源下载和安装 Python 包的原始主流方法，有点像 Python 的 apt-get。当您安装 SetupTools 时，它会安装一个名为 easy_install 的脚本或 exe，您可以在命令行上调用它来安装或升级软件包。它还提供了一种创建 Python 蛋的方法。让我们花一点时间来了解这个实用程序。

### 使用 easy_install 安装软件包

一旦安装了 SetupTools，您的路径上应该有 easy_install。这意味着您可以打开终端(Linux)或命令行(Windows ),然后简单地运行 easy_install。下面是一个调用示例:

```py

easy_install sqlalchemy

```

这将发送到 PyPI，并尝试从那里或者从 PyPI 包页面指向的任何位置下载最新的 SQLAlchemy。easy_install 脚本也将安装它。easy_install 出错的一个常见问题是，它会在下载完软件包之前尝试开始安装，这可能会导致错误的安装。easy_install 很酷的一点是，如果你正确设置了 setup.py 文件，它也可以下载并安装依赖项。所以如果你安装一些复杂的东西，比如 TurboGears，你会看到它安装了很多包。这就是为什么你想使用 virtualenv 的一个原因，因为你可以确定你喜欢新的软件包，并且它们可以正常工作。如果没有，你就删除 virtualenv 文件夹。否则，你将不得不进入你的 Python 文件夹，并尝试自己“卸载”(即删除文件夹)。easy_install 在安装 egg 时做的另一件事是，它将 egg 添加到 site-packages 中的一个 **easy-install.pth** 文件中，所以当你卸载它时，你也需要编辑它。幸运的是，如果你不喜欢自己动手，你可以用 pip 卸载它。有一个-uninstall (-u)命令，但是我听到了关于它如何工作的混合报告。

您可以通过将 url 直接传递给 easy_install 来安装软件包。另一个有趣的功能是，你可以告诉 easy_install 你想要哪个版本，它会尝试安装它。最后，easy_install 可以从源文件或 eggs 安装。要获得完整的命令列表，您应该阅读[文档](http://peak.telecommunity.com/DevCenter/setuptools#command-reference)

### 创造一个卵子

egg 文件是 Python 包的分发格式。它只是一个源代码发行版或 Windows 可执行文件的替代品，但需要注意的是，对于纯 Python 来说，egg 文件是完全跨平台的。我们将看看如何使用我们在[之前的教程](https://www.blog.pythonlibrary.org/2012/07/08/python-201-creating-modules-and-packages/)中创建的包来创建我们自己的蛋。创建一个新文件夹，并将 mymath 文件夹放入其中。然后在 mymath 的父目录中创建一个 setup.py 文件，内容如下:

```py

from setuptools import setup, find_packages

setup(
    name = "mymath",
    version = "0.1",
    packages = find_packages()
    )

```

注意，我们没有使用 Python 的 distutils 的 setup 函数，而是使用 setuptools 的 setup。我们还使用 setuptools 的 **find_packages** 函数，该函数将自动查找当前目录中的任何包，并将它们添加到 egg 中。要创建上述 egg，您需要从命令行执行以下操作:

```py

python setup.py bdist_egg

```

这将生成大量输出，但是当它完成时，您将看到有三个新文件夹:build、dist 和 mymath.egg-info。我们唯一关心的是 **dist** 文件夹，在这个文件夹中您可以填充并找到您的 egg 文件， **mymath-0.1-py2.6.egg** 。请注意，在我的机器上，它选择了我的默认 Python，即 2.6 版本，并针对该版本的 Python 创建了 egg。egg 文件本身基本上是一个 zip 文件。如果您将扩展名改为“zip ”,您可以看到它有两个文件夹:mymath 和 EGG-INFO。此时，您应该能够将 easy_install 指向文件系统上的 egg，并让它安装您的包。

如果您愿意，还可以使用 easy_install 将您的 egg 或源代码直接上传到 Python 包索引(PyPI)，方法是使用以下命令(从 docs 复制):

```py

setup.py bdist_egg upload         # create an egg and upload it
setup.py sdist upload             # create a source distro and upload it
setup.py sdist bdist_egg upload   # create and upload both

```

### 包扎

此时，您应该能够使用 easy_install，或者对尝试其中一种替代方法有足够的了解。就我个人而言，我很少使用它，也不介意使用它。然而，我们将会关注 pip 并很快发布，因此在接下来的一两篇文章中，您将能够学习如何使用它们。与此同时，试试 easy_install 吧，看看你有什么想法，或者在评论中说出你的恐怖故事！

### 进一步阅读

*   设置工具[文档](http://peak.telecommunity.com/DevCenter/setuptools)
*   EasyInstall [文档](http://peak.telecommunity.com/DevCenter/EasyInstall)
*   [搭便车的包装指南](http://guide.python-distribute.org/installation.html)
*   [Python 在 Windows 上的开发【第二部分】:安装 easy_install...可能更容易](http://blog.sadphaeton.com/2009/01/20/python-development-windows-part-2-installing-easyinstallcould-be-easier.html)
*   如何安装 Python Easy_install 来配合 Siri 服务器使用( [youtube](www.youtube.com/watch?v=c96fTX1w_e0) )
*   包装的奇异世界 Tarek Ziade 的分叉设置工具