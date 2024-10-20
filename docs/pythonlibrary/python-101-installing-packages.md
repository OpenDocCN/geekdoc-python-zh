# Python 101 -安装包

> 原文：<https://www.blog.pythonlibrary.org/2014/03/27/python-101-installing-packages/>

当您第一次开始成为 Python 程序员时，您不会考虑您可能需要如何安装外部包或模块。但是当这种需求出现时，你会很快想知道如何去做！Python 包在互联网上随处可见。大部分流行的都可以在 [Python 包索引(PyPI)](https://pypi.python.org/pypi) 上找到。你还会在 github、bitbucket、Google code 上找到很多 Python 包。在本文中，我们将介绍以下安装 Python 包的方法:

*   从源安装
*   简单安装
*   点
*   安装软件包的其他方式

* * *

### 从源安装

从源代码安装是一项很好的技能。还有更简单的方法，我们将在本文后面讨论。然而，有一些软件包您必须从源代码安装。例如，要使用 **easy_install** ，您需要首先安装 [setuptools](https://pypi.python.org/pypi/setuptools) 。为此，您需要下载 tar 或 zip 文件，并将其解压缩到系统中的某个位置。然后寻找 **setup.py** 文件。打开终端会话，将目录切换到包含安装文件的文件夹。然后运行以下命令:

```py

python setup.py install

```

如果 python 不在您的系统路径上，您将收到一条错误消息，指出“Python”命令未找到或者是一个未知的应用程序。您可以使用 Python 的完整路径来调用此命令。如果你在 Windows 上，你可以这样做:

```py

c:\python34\python.exe setup.py install

```

如果您安装了多个版本的 Python，并且需要将包安装到不同的版本上，这种方法尤其方便。您所需要做的就是输入正确 Python 版本的完整路径，并根据它安装软件包。

有些软件包包含 C 代码，例如 C 头文件，需要编译这些文件才能正确安装软件包。在 Linux 上，你通常已经安装了一个 C/C++编译器，你可以轻松地安装这个包。在 Windows 上，您需要安装正确版本的 Visual Studio 才能正确编译该包。有些人说你也可以使用 MingW，但是我还没有找到一种方法让它工作。如果软件包中已经预先制作了 Windows installer，请使用它。那你就完全不用乱搞编译了。

* * *

### 使用简易安装

一旦安装了 setuptools，您就可以使用 **easy_install** 。你可以在你的 Python 安装的**脚本**文件夹中找到它。请确保将 Scripts 文件夹添加到您的系统路径中，这样您就可以在命令行上调用 easy_install，而无需指定其完整路径。尝试运行以下命令来了解 easy_install 的所有选项:

```py

easy_install -h

```

当你想用 easy_install 安装包时，你所要做的就是:

```py

easy_install package_name

```

easy_install 将尝试从 PyPI 下载软件包，编译它(如果需要)并安装它。如果你进入你的 Python 的 **site-packages** 目录，你会发现一个名为 **easy-install.pth** 的文件，它将包含所有用 easy_install 安装的包的条目。Python 使用这个文件来帮助导入模块或包。

您还可以告诉 easy_install 从 URL 或您计算机上的路径进行安装。它还可以直接从 tar 文件安装软件包。你可以使用 easy_install 通过使用 **- upgrade** (或者-U)来升级一个包。最后，您可以使用 easy_install 来安装 Python eggs。您可以在 PyPI 和其他位置找到 egg 文件。egg 基本上是一个特殊的 zip 文件。事实上，如果您将扩展名更改为。zip，可以解压 egg 文件。

以下是一些例子:

```py

easy_install -U SQLAlchemy
easy_install http://example.com/path/to/MyPackage-1.2.3.tgz
easy_install /path/to/downloaded/package

```

easy_install 存在一些问题。它会在下载完成前尝试安装一个软件包。无法使用 easy_install 卸载软件包。您必须自己删除软件包，并通过删除软件包中的条目来更新 easy-install.pth 文件。出于这些和其他原因，Python 社区中出现了创造不同事物的运动，这导致了 **pip** 的诞生。

* * *

### 使用画中画

安装 pip 与我们之前讨论的稍有不同。您仍然可以访问 PyPI，但是不是下载包并运行它的 setup.py 脚本，而是要求您下载一个名为 **get-pip.py** 的脚本。然后，您需要通过执行以下操作来执行它:

```py

python get-pip.py

```

这将安装 setuptools 或 setuptools 的替代产品 **distribute** ，如果其中一个尚未安装的话。它还将安装 pip。pip 与 CPython 版本 2.6、2.7、3.1、3.2、3.3、3.4 以及 pypy 一起工作。您可以使用 pip 来安装 easy_install 可以安装的任何东西，但是调用有点不同。要安装软件包，请执行以下操作:

```py

pip install PackageName

```

要升级软件包，您需要执行以下操作:

```py

pip install -U PackageName

```

您可能想调用“pip -h”来获得 pip 可以做的所有事情的完整列表。pip 可以安装而 easy_install 不能安装的一个东西是 Python wheel 格式。wheel 是一个 ZIP 格式的归档文件，具有特殊格式的文件名和。whl 分机。您也可以通过自己的命令行实用程序安装轮子。另一方面，pip 不能安装鸡蛋。如果你需要安装一个 egg，你会希望使用 easy_install。

* * *

### 关于依赖性的一个注记

使用 easy_install 和 pip 的众多好处之一是，如果软件包在其 setup.py 脚本中指定了依赖项，easy_install 和 pip 也会尝试下载并安装它们。当您尝试新的软件包时，这可以减轻很多挫折，您没有意识到软件包 A 依赖于软件包 B、C 和 d。

* * *

### 包扎

现在，您应该能够安装您需要的任何包了，假设这个包支持您的 Python 版本。Python 程序员可以使用很多工具。虽然现在 Python 中的打包有点混乱，但是一旦你知道如何使用合适的工具，你通常可以得到你想要安装或打包的东西。

* * *

### 附加阅读

*   PyPI [安装工具页](https://pypi.python.org/pypi/setuptools)
*   [简易安装指南](https://pythonhosted.org/setuptools/easy_install.html)
*   PyPI [pip 页面](https://pypi.python.org/pypi/pip)
*   车轮[文档](http://wheel.readthedocs.org/en/latest/)
*   [Python 打包用户指南](https://python-packaging-user-guide.readthedocs.org/en/latest/index.html)
*   [分发包](https://pypi.python.org/pypi/distribute/0.7.3)