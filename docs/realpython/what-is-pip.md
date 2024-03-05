# 使用 Python 的 pip 管理项目的依赖关系

> 原文：<https://realpython.com/what-is-pip/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Pip 入门**](/courses/what-is-pip/)

[Python](https://www.python.org/) 的标准包管理器是 [`pip`](https://pip.pypa.io/en/stable/) 。它允许你安装和管理不属于 [Python 标准库](https://docs.python.org/3/py-modindex.html)的包。如果你正在寻找关于`pip`的介绍，那么你来对地方了！

**在本教程中，您将学习如何:**

*   **在您的工作环境中设置`pip`**
*   修复与使用`pip`相关的**常见错误**
*   **用`pip`安装和卸载**软件包
*   使用**需求文件**管理项目的依赖关系

您可以使用`pip`做很多事情，但是 Python 社区非常活跃，已经创建了一些简洁的替代方案来代替`pip`。您将在本教程的后面部分了解这些内容。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## `pip` 入门

那么，`pip`到底是做什么的呢？ [`pip`](https://pip.pypa.io/en/stable/) 是 Python 的一个**包管理器**。这意味着它是一个工具，允许您安装和管理不作为标准库的一部分分发的库和依赖项。伊恩·比克在 2008 年引入了 pip 这个名字:

> 我已经将 pyinstall 重命名为新名称:pip。pip 这个名字是一个缩写和声明:pip 安装软件包。([来源](https://www.ianbicking.org/blog/2008/10/pyinstall-is-dead-long-live-pip.html))

包管理是如此重要，以至于 Python 的安装程序从版本 3.4 和 2.7.9 开始就包含了`pip`，分别针对 Python 3 和 Python 2。许多 Python 项目使用`pip`，这使得它成为每个 Python 爱好者的必备工具。

如果您来自另一种编程语言，那么您可能对包管理器的概念很熟悉。 [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript) 使用 [npm](https://www.npmjs.com/) 进行包管理， [Ruby](https://www.ruby-lang.org/en/) 使用 [gem](https://rubygems.org/) ，而[。NET 平台](https://dotnet.microsoft.com/languages)使用 [NuGet](https://www.nuget.org/) 。在 Python 中，`pip`已经成为标准的包管理器。

[*Remove ads*](/account/join/)

### 在您的系统上找到`pip`

在您的系统上安装 Python 时，Python 3 安装程序为您提供了安装`pip`的选项。事实上，用 Python 安装`pip`的选项是默认勾选的，所以安装完 Python 后`pip`应该就可以使用了。

**注意:**在 Ubuntu 等一些 Linux (Unix)系统上，`pip`是一个名为`python3-pip`的单独的包，你需要用`sudo apt install python3-pip`来安装。默认情况下，它不会随解释器一起安装。

您可以通过在您的系统上查找`pip3`可执行文件来验证`pip`是否可用。从下面选择您的操作系统，并相应地使用特定于平台的命令:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
C:\> where pip3
```

Windows 上的`where`命令会告诉你在哪里可以找到`pip3`的可执行文件。如果 Windows 找不到名为`pip3`的可执行文件，那么你也可以尝试寻找末尾没有三个(`3`)的`pip`。

```py
$ which pip3
```

Linux 系统和 macOS 上的`which`命令显示了`pip3`二进制文件的位置。

在 Windows 和 Unix 系统中，`pip3`可能位于多个位置。当您安装了多个 Python 版本时，可能会发生这种情况。如果您在系统的任何位置都找不到`pip`，那么您可以考虑[重新安装 pip](#reinstalling-pip-when-errors-occur) 。

除了直接运行您的系统`pip`，您还可以将它作为 Python 模块来运行。在下一节中，您将了解如何操作。

### 将`pip`作为模块运行

当你直接运行你的系统`pip`时，命令本身并不会透露`pip`属于哪个 Python 版本。不幸的是，这意味着您可以使用`pip`将一个包安装到旧 Python 版本的站点包中，而不会注意到。为了防止这种情况发生，您可以将`pip`作为 Python 模块运行:

```py
$ python3 -m pip
```

注意，您使用了`python3 -m`来运行`pip`。`-m`开关告诉 Python 运行一个模块作为`python3`解释器的可执行文件。这样，您可以确保您的系统默认 Python 3 版本运行`pip`命令。如果你想进一步了解这种运行`pip`的方式，那么你可以阅读 Brett Cannon 关于[使用`python3 -m pip`](https://snarky.ca/why-you-should-use-python-m-pip/) 的优势的颇有见地的文章。

有时您可能想要更明确地将包限制在特定的项目中。在这种情况下，你应该在一个虚拟环境中运行`pip`。

### 在 Python 虚拟环境中使用`pip`

为了避免将包直接安装到您的系统 Python 安装中，您可以使用一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)。虚拟环境为您的项目提供了一个独立的 Python 解释器。您在这个环境中使用的任何包都将独立于您的系统解释器。这意味着您可以将项目的依赖项与其他项目和整个系统分开。

在虚拟环境中使用`pip`有三个主要优势。您可以:

1.  确保您正在为手头的项目使用正确的 Python 版本
2.  确信你在运行`pip`或`pip3`时引用的是**正确的`pip`实例**
3.  在不影响其他项目的情况下，为您的项目使用一个特定的包版本

Python 3 内置了 [`venv`](https://docs.python.org/3/library/venv.html) 模块，用于创建虚拟环境。本模块帮助您使用独立的 Python 安装创建虚拟环境。一旦激活了虚拟环境，就可以将软件包安装到该环境中。您安装到一个虚拟环境中的软件包与系统上的所有其他环境是隔离的。

您可以按照以下步骤创建一个虚拟环境，并验证您在新创建的环境中使用了`pip`模块:

*   [*视窗*](#windows-2)
**   [**Linux + macOS**](#linux-macos-2)*

```py
C:\> python -m venv venv
C:\> venv\Scripts\activate.bat
(venv) C:\>  pip3 --version
pip 21.2.3 from ...\lib\site-packages\pip (python 3.10)
(venv) C:\>  pip --version
pip 21.2.3 from ...\lib\site-packages\pip (python 3.10)
```

```py
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip3 --version
pip 21.2.3 from .../python3.10/site-packages/pip (python 3.10)
(venv) $ pip --version
pip 21.2.3 from .../python3.10/site-packages/pip (python 3.10)
```

在这里，您通过使用 Python 的内置`venv`模块创建一个名为`venv`的虚拟环境。然后你用`source`命令激活它。您的`venv`名字周围的括号(`()`)表示您成功激活了虚拟环境。

最后，检查激活的虚拟环境中的`pip3`和`pip`可执行文件的版本。两者都指向同一个`pip`模块，所以一旦你的虚拟环境被激活，你可以使用`pip`或者`pip3`。

[*Remove ads*](/account/join/)

### 出现错误时重新安装`pip`

当运行`pip`命令时，在某些情况下可能会出现错误。具体的错误消息取决于您的操作系统:

| 操作系统 | 出错信息 |
| --- | --- |
| Windows 操作系统 | `'pip' is not recognized as an internal or external command,`
T1】 |
| Linux 操作系统 | `bash: pip: command not found` |
| 马科斯 | `zsh: command not found: pip` |

类似这样的错误消息表明`pip`的安装出现了问题。

**注意:**当`pip`命令不起作用时，在您开始任何故障排除之前，您可以尝试使用结尾带有三个(`3`)的`pip3`命令。

得到如上所示的错误可能会令人沮丧，因为`pip`对于安装和管理外部包是至关重要的。`pip`的一些常见问题与该工具在您的系统上的安装方式有关。

尽管不同系统的错误信息不同，但它们都指向同一个问题:您的系统无法在您的 [`PATH`](https://realpython.com/add-python-to-path/) 变量中列出的位置找到`pip`。在 Windows 上，`PATH`是**系统变量**的一部分。在 macOS 和 Linux 上，`PATH`是**环境变量**的一部分。你可以用这个命令检查你的`PATH`变量的内容:

*   [*视窗*](#windows-3)
**   [**Linux + macOS**](#linux-macos-3)*

```py
C:\> echo %PATH%
```

```py
$ echo $PATH
```

该命令的输出将显示操作系统在磁盘上查找可执行程序的位置(目录)列表。根据您的系统，位置可以用冒号(`:`)或分号(`;`)分隔。

默认情况下，在安装 Python 或创建虚拟环境之后，包含`pip`可执行文件的目录应该出现在`PATH`中。然而，失踪的`pip`是一个常见的问题。两个[支持的方法](https://pip.pypa.io/en/stable/installation/#supported-methods)可以帮助你再次安装`pip`并将其添加到你的`PATH`中:

1.  [`ensurepip`](https://docs.python.org/3/library/ensurepip.html#module-ensurepip) 模块
2.  [`get-pip.py`](https://github.com/pypa/get-pip) 剧本

从 Python 3.4 开始，`ensurepip`模块就是标准库的一部分。它被添加到[中，为你重新安装`pip`提供了一种直接的方式](https://www.python.org/dev/peps/pep-0453/)，例如，如果你在安装 Python 时跳过了它或者在某个时候卸载了`pip`。在下面选择您的操作系统，并相应地运行`ensurepip`:

*   [*视窗*](#windows-4)
**   [**Linux + macOS**](#linux-macos-4)*

```py
C:\> python -m ensurepip --upgrade
```

```py
$ python3 -m ensurepip --upgrade
```

如果`pip`还没有安装，那么这个命令会将它安装到您当前的 Python 环境中。如果您在一个活动的虚拟环境中，那么该命令会将`pip`安装到该环境中。否则，它会在你的系统上全局安装`pip`。`--upgrade`选项确保`pip`版本与`ensurepip`中声明的版本相同。

**注意:**`ensurepip`模块不接入互联网。`ensurepip`可以安装的最新版本的`pip`是捆绑在您环境的 Python 安装中的版本。比如用 Python 3.10.0 运行`ensurepip`，安装`pip` 21.2.3。如果你想要一个更新的`pip`版本，那么你需要首先运行`ensurepip`。之后，您可以手动将`pip`更新到其最新版本。

另一种修复`pip`安装的方法是使用`get-pip.py`脚本。`get-pip.py`文件包含一个完整的`pip`副本，作为一个编码的 [ZIP 文件](https://realpython.com/python-zip-import/)。你可以直接从 [PyPA 引导页面](https://bootstrap.pypa.io/get-pip.py)下载`get-pip.py`。一旦你在你的机器上有了脚本，你就可以像这样运行 Python 脚本:

*   [*视窗*](#windows-5)
**   [**Linux + macOS**](#linux-macos-5)*

```py
C:\> python get-pip.py
```

```py
$ python3 get-pip.py
```

该脚本将在您当前的 Python 环境中安装最新版本的`pip`、[、`setuptools`、](https://setuptools.pypa.io/en/latest/)和[、`wheel`、](https://realpython.com/python-wheels/)。如果您只想安装`pip`，那么您可以在您的命令中添加`--no-setuptools`和`--no-wheel`选项。

如果以上方法都不起作用，那么可能值得尝试为您当前的平台下载最新的 [Python 版本](https://www.python.org/downloads/)。你可以按照 [Python 3 安装&设置指南](https://realpython.com/installing-python/)来确保`pip`被正确安装并且工作无误。

[*Remove ads*](/account/join/)

## 用`pip` 安装包

Python 被认为是一门包含[电池的](https://www.python.org/dev/peps/pep-0206/#id3)语言。这意味着 [Python 标准库](https://docs.python.org/3/py-modindex.html)包含了一套广泛的[包和模块](https://realpython.com/python-modules-packages/)来帮助开发者完成他们的编码项目。

同时，Python 有一个活跃的社区，它提供了一组更广泛的包，可以帮助您满足开发需求。这些包被发布到 [Python 包索引](https://pypi.org/)，也称为 **PyPI** (读作*派豌豆眼*)。

PyPI 托管了大量的包，包括开发框架、工具和库。其中许多包都提供了 Python 标准库功能的友好接口。

### 使用 Python 包索引(PyPI)

PyPI 托管的众多包中有一个叫做 [`requests`](https://realpython.com/python-requests/) 。`requests`库通过抽象 [HTTP](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol) 请求的复杂性来帮助[与 web 服务](https://realpython.com/api-integration-in-python/)交互。你可以在官方的[文档网站](http://docs.python-requests.org/en/master/)上了解关于`requests`的一切。

当您想在项目中使用`requests`包时，您必须首先将它安装到您的环境中。如果您不想将它安装在您的系统 Python 站点包中，那么您可以首先创建一个虚拟环境，如上所示。

一旦创建并激活了虚拟环境，命令行提示符就会在括号中显示虚拟环境的名称。从现在起，您执行的任何`pip`命令都将发生在您的虚拟环境中。

为了安装软件包，`pip`提供了一个`install`命令。你可以运行它来安装`requests`包:

*   [*视窗*](#windows-6)
**   [**Linux + macOS**](#linux-macos-6)*

```py
(venv) C:\> python -m pip install requests
```

```py
(venv) $ python3 -m pip install requests
```

在这个例子中，您运行`pip`,带有`install`命令，后跟您想要安装的软件包的名称。`pip`命令在 PyPI 中寻找包，解析它的依赖项，并在当前的 Python 环境中安装所有东西，以确保`requests`能够工作。

`pip install <package>`命令总是寻找软件包的最新版本并安装它。它还搜索软件包元数据中列出的依赖项并安装它们，以确保软件包具有它需要的所有要求。

也可以在一个命令中安装多个软件包:

*   [*视窗*](#windows-7)
**   [**Linux + macOS**](#linux-macos-7)*

```py
(venv) C:\> python -m pip install rptree codetiming
```

```py
(venv) $ python3 -m pip install rptree codetiming
```

通过在`pip install`命令中链接软件包`rptree`和`codetiming`，你可以一次安装两个软件包。您可以向`pip install`命令添加任意数量的包。在这种情况下，一个`requirements.txt`文件可以派上用场。在本教程的后面，您将学习如何使用一个`requirements.txt`文件一次安装多个包。

**注意:**除非软件包的具体版本号与本教程相关，否则您会注意到版本字符串采用了`x.y.z`的通用形式。这是一种占位符格式，可以代表`3.1.4`、`2.9`或任何其他版本号。当您继续操作时，终端中的输出将显示您实际的包版本号。

您可以使用`list`命令来显示您的环境中安装的软件包，以及它们的版本号:

*   [*视窗*](#windows-8)
**   [**Linux + macOS**](#linux-macos-8)*

```py
(venv) C:\> python -m pip list
Package            Version
------------------ ---------
certifi            x.y.z
charset-normalizer x.y.z
codetiming         x.y.z
idna               x.y.z
pip                x.y.z
requests           x.y.z
rptree             x.y.z
setuptools         x.y.z
urllib3            x.y.z
```

```py
(venv) $ python3 -m pip list
Package            Version
------------------ ---------
certifi            x.y.z
charset-normalizer x.y.z
idna               x.y.z
pip                x.y.z
requests           x.y.z
setuptools         x.y.z
urllib3            x.y.z
```

`pip list`命令呈现一个表格，显示当前环境中所有已安装的软件包。上面的输出显示了使用`x.y.z`占位符格式的包的版本。当您在您的环境中运行`pip list`命令时，`pip`会显示您为每个包安装的特定版本号。

要获得关于特定包的更多信息，您可以通过使用`pip`中的`show`命令来查看包的元数据:

*   [*视窗*](#windows-9)
**   [**Linux + macOS**](#linux-macos-9)*

```py
(venv) C:\> python -m pip show requests
Name: requests
Version: x.y.z
Summary: Python HTTP for Humans.
 ...
Requires: certifi, idna, charset-normalizer, urllib3
Required-by:
```

```py
(venv) $ python3 -m pip show requests
Name: requests
Version: x.y.z
Summary: Python HTTP for Humans.
 ...
Requires: certifi, idna, charset-normalizer, urllib3
Required-by:
```

您系统上的这个命令的输出将列出包的元数据。`Requires`行列出了包，比如[`certifi`](https://pypi.org/project/certifi/)[`idna`](https://pypi.org/project/idna/)[`charset-normalizer`](https://charset-normalizer.readthedocs.io/en/latest/)[`urllib3`](https://urllib3.readthedocs.io/en/stable/)。安装这些是因为`requests`依赖它们才能正常工作。

既然已经安装了`requests`及其依赖项，就可以像 Python 代码中的任何其他常规包一样[导入](https://realpython.com/python-modules-packages/)了。启动[交互式 Python 解释器](https://realpython.com/interacting-with-python/)，导入`requests`包；

>>>

```py
>>> import requests
>>> requests.__version__
"x.y.z"
```

启动交互式 Python 解释器后，您导入了`requests`模块。通过调用`requests.__version__`，您验证了您正在虚拟环境中使用`requests`模块。

[*Remove ads*](/account/join/)

### 使用自定义包索引

默认情况下，`pip`使用 PyPI 来查找包。但是`pip`也给了你定义自定义包索引的选项。

当 PyPI 域在您的网络上被阻止时，或者如果您想使用不公开的包，使用带有自定义索引的`pip`会很有帮助。有时系统管理员也会创建他们自己的内部包索引，以便更好地控制哪些包版本对公司网络上的`pip`用户可用。

自定义包索引必须符合[PEP 503–简单存储库 API](https://www.python.org/dev/peps/pep-0503/) 才能与`pip`一起工作。您可以通过访问 [PyPI 简单索引](https://pypi.org/simple/)来了解这样一个 [API(应用程序编程接口)](https://en.wikipedia.org/wiki/API)是什么样子的——但是要注意这是一个很大的页面，有很多难以解析的内容。任何遵循相同 API 的定制索引都可以使用`--index-url`选项。除了键入`--index-url`，你还可以使用`-i`速记。

例如，要从 [TestPyPI](https://test.pypi.org/) 包索引中安装 [`rptree`](https://realpython.com/directory-tree-generator-python/) 工具，可以运行以下命令:

*   [*视窗*](#windows-10)
**   [**Linux + macOS**](#linux-macos-10)*

```py
(venv) C:\> python -m pip install -i https://test.pypi.org/simple/ rptree
```

```py
(venv) $ python3 -m pip install -i https://test.pypi.org/simple/ rptree
```

使用`-i`选项，您告诉`pip`查看不同的包索引，而不是默认的 PyPI。这里，你是从 TestPyPI 而不是从 PyPI 安装`rptree`。您可以使用 TestPyPI 来微调 Python 包的[发布过程，而不会弄乱 PyPI 上的产品包索引。](https://realpython.com/pypi-publish-python-package/)

如果需要永久使用替代索引，那么可以在`pip` [配置文件](https://pip.pypa.io/en/stable/topics/configuration/)中设置`index-url`选项。这个文件叫做`pip.conf`，您可以通过运行以下命令找到它的位置:

*   [*视窗*](#windows-11)
**   [**Linux + macOS**](#linux-macos-11)*

```py
(venv) C:\> python -m pip config list -vv
```

```py
(venv) $ python3 -m pip config list -vv
```

使用`pip config list`命令，您可以列出活动配置。只有当您设置了自定义配置时，此命令才会输出一些内容。否则，输出为空。这时，附加的`--verbose`或`-vv`选项会有所帮助。当您添加`-vv`，`pip`会向您显示它在哪里寻找不同的配置级别。

如果你想添加一个`pip.conf`文件，那么你可以选择一个`pip config list -vv`列出的位置。带有自定义包索引的`pip.conf`文件如下所示:

```py
# pip.conf

[global]
index-url = https://test.pypi.org/simple/
```

当你有一个像这样的`pip.conf`文件时，`pip`将使用定义好的`index-url`来寻找包。有了这个配置，您不需要在您的`pip install`命令中使用`--index-url`选项来指定您只想要可以在 TestPyPI 的[简单 API](https://test.pypi.org/simple/) 中找到的包。

### 从你的 GitHub 库安装包

您不局限于托管在 PyPI 或其他包索引上的包。`pip`还提供了从 [GitHub 库](https://realpython.com/python-git-github-intro/)安装包的选项。但是即使一个包托管在 PyPI 上，比如[真正的 Python 目录树生成器](https://pypi.org/project/rptree/)，你也可以选择从它的 [Git 库](https://github.com/realpython/rptree)安装它:

*   [*视窗*](#windows-12)
**   [**Linux + macOS**](#linux-macos-12)*

```py
(venv) C:\> python -m pip install git+https://github.com/realpython/rptree
```

```py
(venv) $ python3 -m pip install git+https://github.com/realpython/rptree
```

使用`git+https`方案，您可以指向包含可安装包的 Git 存储库。您可以通过运行交互式 Python 解释器并导入`rptree`来验证您是否正确安装了包:

>>>

```py
>>> import rptree
>>> rptree.__version__
"x.y.z"
```

启动交互式 Python 解释器后，导入`rptree`模块。通过调用`rptree.__version__`，您验证了您正在使用基于虚拟环境的`rptree`模块。

**注意:**如果你使用的是版本控制系统(VCS)而不是 Git，`pip`你已经覆盖了。要了解如何将`pip`与 Mercurial、Subversion 或 Bazaar 一起使用，请查看`pip`文档的 [VCS 支持章节](https://pip.pypa.io/en/stable/topics/vcs-support/)。

如果包不是托管在 PyPI 上，而是有一个远程 Git 存储库，那么从 Git 存储库安装包会很有帮助。您指向的远程存储库甚至可以托管在您公司内部网的内部 Git 服务器上。当您处于防火墙之后或 Python 项目有其他限制时，这可能会很有用。

[*Remove ads*](/account/join/)

### 以可编辑模式安装软件包以简化开发

当处理您自己的包时，以可编辑的模式安装它是有意义的。通过这样做，您可以在处理源代码的同时仍然像在任何其他包中一样使用命令行。典型的工作流程是首先克隆存储库，然后使用`pip`将其作为可编辑的包安装到您的环境中:

*   [*视窗*](#windows-13)
**   [**Linux + macOS**](#linux-macos-13)*

```py
 1C:\> git clone https://github.com/realpython/rptree
 2C:\> cd rptree
 3C:\rptree> python3 -m venv venv
 4C:\rptree> venv\Scripts\activate.bat
 5(venv) C:\rptree> python -m pip install -e .
```

```py
 1$ git clone https://github.com/realpython/rptree
 2$ cd rptree
 3$ python3 -m venv venv
 4$ source venv/bin/activate
 5(venv) $ python3 -m pip install -e .
```

使用上面的命令，您安装了作为可编辑模块的`rptree`包。以下是您刚刚执行的操作的详细步骤:

1.  **第 1 行**克隆了`rptree`包的 Git 库。
2.  **第 2 行**将工作目录更改为`rptree/`。
3.  **第 3 行和第 4 行**创建并激活了一个虚拟环境。
4.  **第 5 行**将当前目录的内容安装为可编辑包。

`-e`选项是`--editable`选项的简写。当你使用`-e`选项和`pip install`时，你告诉`pip`你想在可编辑模式下安装软件包。不使用包名，而是使用一个点(`.`)将`pip`指向当前目录。

如果您没有使用`-e`标志，`pip`将会把这个包正常安装到您环境的`site-packages/`文件夹中。当您在可编辑模式下安装软件包时，您会在站点软件包中创建一个到本地项目路径的链接:

```py
 ~/rptree/venv/lib/python3.10/site-packages/rptree.egg-link
```

使用带有`-e`标志的`pip install`命令只是`pip install`提供的众多选项之一。你可以在`pip`文档中查看 [`pip install`示例](https://pip.pypa.io/en/stable/cli/pip_install/#examples)。在那里你将学习如何安装一个包的特定版本，或者将`pip`指向一个不同的不是 PyPI 的索引。

在下一节中，您将了解需求文件如何帮助您的`pip`工作流。

## 使用需求文件

`pip install`命令总是安装软件包的最新发布版本，但是有时您的代码需要特定的软件包版本才能正常工作。

您希望创建用于开发和测试应用程序的依赖项和版本的规范，以便在生产中使用应用程序时不会出现意外。

### 固定要求

当您与其他开发人员共享 Python 项目时，您可能希望他们使用与您相同版本的外部包。也许某个软件包的特定版本包含了您所依赖的新特性，或者您正在使用的软件包版本与以前的版本不兼容。

这些外部依赖也被称为需求。您经常会发现 Python 项目将它们的需求固定在一个名为`requirements.txt`或类似的文件中。[需求文件格式](https://pip.pypa.io/en/stable/reference/requirements-file-format/)允许您精确地指定应该安装哪些包和版本。

运行`pip help`显示有一个`freeze`命令以需求格式输出已安装的包。您可以使用这个命令，将输出重定向到一个文件来生成一个需求文件:

*   [*视窗*](#windows-14)
**   [**Linux + macOS**](#linux-macos-14)*

```py
(venv) C:\> python -m pip freeze > requirements.txt
```

```py
(venv) $ python3 -m pip freeze > requirements.txt
```

该命令在您的工作目录中创建一个包含以下内容的`requirements.txt`文件:

```py
certifi==x.y.z
charset-normalizer==x.y.z
idna==x.y.z
requests==x.y.z
urllib3==x.y.z
```

记住上面显示的`x.y.z`是包版本的占位符格式。您的`requirements.txt`文件将包含真实的版本号。

`freeze`命令将当前安装的软件包的名称和版本转储到标准输出中。您可以将输出重定向到一个文件，稍后您可以使用该文件将您的确切需求安装到另一个系统中。您可以随意命名需求文件。然而，一个广泛采用的惯例是将其命名为`requirements.txt`。

当您想要在另一个系统中复制环境时，您可以运行`pip install`，使用`-r`开关来指定需求文件:

*   [*视窗*](#windows-15)
**   [**Linux + macOS**](#linux-macos-15)*

```py
(venv) C:\> python -m pip install -r requirements.txt
```

```py
(venv) $ python3 -m pip install -r requirements.txt
```

在上面的命令中，您告诉`pip`将`requirements.txt`中列出的包安装到您当前的环境中。包版本将匹配`requirements.txt`文件包含的版本约束。您可以运行`pip list`来显示您刚刚安装的包，以及它们的版本号:

*   [*视窗*](#windows-16)
**   [**Linux + macOS**](#linux-macos-16)*

```py
(venv) C:\> python -m pip list

Package            Version
------------------ ---------
certifi            x.y.z
charset-normalizer x.y.z
idna               x.y.z
pip                x.y.z
requests           x.y.z
setuptools         x.y.z
urllib3            x.y.z
```

```py
(venv) $ python3 -m pip list

Package            Version
------------------ ---------
certifi            x.y.z
charset-normalizer x.y.z
idna               x.y.z
pip                x.y.z
requests           x.y.z
setuptools         x.y.z
urllib3            x.y.z
```

现在，您可以分享您的项目了！您可以将`requirements.txt`提交到 Git 这样的版本控制系统中，并使用它在其他机器上复制相同的环境。但是等等，如果这些包发布了新的更新会怎么样呢？

[*Remove ads*](/account/join/)

### 微调要求

硬编码软件包的版本和依赖关系的问题是，软件包会频繁地更新错误和安全补丁。您可能希望这些更新一发布就加以利用。

需求文件格式允许您使用比较操作符来指定依赖版本，这为您提供了一些灵活性，以确保在定义包的基本版本的同时更新包。

在您最喜欢的[编辑器](https://realpython.com/python-ides-code-editors-guide/)中打开`requirements.txt`，将等式运算符(`==`)转换为大于或等于运算符(`>=`，如下例所示:

```py
# requirements.txt

certifi>=x.y.z
charset-normalizer>=x.y.z
idna>=x.y.z
requests>=x.y.z
urllib3>=x.y.z
```

您可以将[比较运算符](https://realpython.com/python-operators-expressions/#comparison-operators)改为`>=`来告诉`pip`安装一个已经发布的精确或更高版本。当您使用`requirements.txt`文件设置一个新环境时，`pip`会寻找满足需求的最新版本并安装它。

接下来，您可以通过运行带有`--upgrade`开关或`-U`速记的`install`命令来升级您的需求文件中的包:

*   [*视窗*](#windows-17)
**   [**Linux + macOS**](#linux-macos-17)*

```py
(venv) C:\> python -m pip install -U -r requirements.txt
```

```py
(venv) $ python3 -m pip install -U -r requirements.txt
```

如果列出的软件包有新版本可用，则该软件包将被升级。

在理想的情况下，新版本的包应该是向后兼容的，并且不会引入新的错误。不幸的是，新版本可能会引入一些会破坏应用程序的变化。为了微调您的需求，需求文件语法支持额外的[版本说明符](https://www.python.org/dev/peps/pep-0440/#version-specifiers)。

假设`requests`的一个新版本`3.0`已经发布，但是引入了一个不兼容的变更，导致应用程序崩溃。您可以修改需求文件以防止安装`3.0`或更高版本:

```py
# requirements.txt

certifi==x.y.z
charset-normalizer==x.y.z
idna==x.y.z
requests>=x.y.z, <3.0 urllib3==x.y.z
```

更改`requests`包的版本说明符可以确保任何大于或等于`3.0`的版本都不会被安装。`pip`文档提供了关于[需求文件格式](https://pip.pypa.io/en/stable/reference/requirements-file-format/)的大量信息，你可以参考它来了解更多。

### 分离生产和开发依赖关系

并非所有在应用程序开发过程中安装的包都是生产依赖项。例如，您可能想要测试您的应用程序，因此您需要一个测试框架。一个流行的测试框架是 [`pytest`](https://realpython.com/pytest-python-testing/) 。您希望将它安装在您的开发环境中，但不希望将其安装在您的生产环境中，因为它不是生产依赖项。

您创建了第二个需求文件`requirements_dev.txt`，列出了设置开发环境的附加工具:

```py
# requirements_dev.txt

pytest>=x.y.z
```

拥有两个需求文件将要求你使用`pip`来安装它们，`requirements.txt`和`requirements_dev.txt`。幸运的是，`pip`允许您在需求文件中指定额外的参数，因此您可以修改`requirements_dev.txt`来安装来自生产`requirements.txt`文件的需求:

```py
# requirements_dev.txt

-r requirements.txt
pytest>=x.y.z
```

注意，您使用同一个`-r`开关来安装生产`requirements.txt`文件。现在，在您的开发环境中，您只需运行这个命令就可以安装所有需求:

*   [*视窗*](#windows-18)
**   [**Linux + macOS**](#linux-macos-18)*

```py
(venv) C:\> python -m pip install -r requirements_dev.txt
```

```py
(venv) $ python3 -m pip install -r requirements_dev.txt
```

因为`requirements_dev.txt`包含了`-r requirements.txt`行，所以您不仅要安装`pytest`，还要安装`requirements.txt`的固定需求。在生产环境中，只安装生产需求就足够了:

*   [*视窗*](#windows-19)
**   [**Linux + macOS**](#linux-macos-19)*

```py
(venv) C:\> python -m pip install -r requirements.txt
```

```py
(venv) $ python3 -m pip install -r requirements.txt
```

使用这个命令，您可以安装`requirements.txt`中列出的需求。与您的开发环境相比，您的生产环境不会安装`pytest`。

[*Remove ads*](/account/join/)

### 生产冻结要求

您创建了生产和开发需求文件，并将它们添加到源代码控制中。这些文件使用灵活的版本说明符来确保您利用由您的依赖项发布的错误修复。您还测试了您的应用程序，现在准备将它部署到生产环境中。

您知道所有的测试都通过了，并且应用程序使用了您在开发过程中使用的依赖项，因此您可能希望确保您将相同版本的依赖项部署到生产环境中。

当前的版本说明符不能保证相同的版本将被部署到生产中，所以您希望在发布项目之前冻结生产需求。

在您根据当前需求完成开发之后，创建当前项目的新版本的工作流可能如下所示:

| 步骤 | 命令 | 说明 |
| --- | --- | --- |
| one | `pytest` | 运行您的测试并验证您的代码工作正常。 |
| Two | `pip install -U -r requirements.txt` | 将您的需求升级到与您的`requirements.txt`文件中的约束相匹配的版本。 |
| three | `pytest` | 运行您的测试，并考虑降级任何给代码带来错误的依赖项。 |
| four | `pip freeze > requirements_lock.txt` | 一旦项目正常工作，将依赖关系冻结到一个`requirements_lock.txt`文件中。 |

有了这样的工作流，`requirements_lock.txt`文件将包含精确的版本说明符，并可用于复制您的环境。您已经确保当您的用户将`requirements_lock.txt`中列出的软件包安装到他们自己的环境中时，他们将使用您希望他们使用的版本。

冻结您的需求是确保您的 Python 项目在您的用户环境中与在您的环境中一样工作的重要一步。

## 用`pip` 卸载软件包

偶尔，你需要卸载一个软件包。要么你找到了更好的库来代替它，要么它是你不需要的东西。卸载软件包可能有点棘手。

请注意，当您安装了`requests`时，您也让`pip`安装其他依赖项。安装的软件包越多，多个软件包依赖于同一个依赖项的可能性就越大。这就是`pip`中的`show`命令派上用场的地方。

在卸载软件包之前，请确保运行该软件包的`show`命令:

*   [*视窗*](#windows-20)
**   [**Linux + macOS**](#linux-macos-20)*

```py
(venv) C:\> python -m pip show requests

Name: requests
Version: 2.26.0
Summary: Python HTTP for Humans.
Home-page: https://requests.readthedocs.io
Author: Kenneth Reitz
Author-email: me@kennethreitz.org
License: Apache 2.0
Location: .../python3.9/site-packages
Requires: certifi, idna, charset-normalizer, urllib3 Required-by:
```

```py
(venv) $ python3 -m pip show requests

Name: requests
Version: 2.26.0
Summary: Python HTTP for Humans.
Home-page: https://requests.readthedocs.io
Author: Kenneth Reitz
Author-email: me@kennethreitz.org
License: Apache 2.0
Location: .../python3.9/site-packages
Requires: certifi, idna, charset-normalizer, urllib3 Required-by:
```

注意最后两个字段，`Requires`和`Required-by`。`show`命令告诉你`requests`需要`certifi`、`idna`、`charset-normalizer`和`urllib3`。你可能也想卸载它们。注意`requests`不是任何其他包所必需的。所以卸载它是安全的。

您应该对所有的`requests`依赖项运行`show`命令，以确保没有其他库也依赖于它们。一旦理解了要卸载的软件包的依赖顺序，就可以使用`uninstall`命令删除它们:

*   [*视窗*](#windows-21)
**   [**Linux + macOS**](#linux-macos-21)*

```py
(venv) C:\> python -m pip uninstall certifi
```

```py
(venv) $ python3 -m pip uninstall certifi
```

`uninstall`命令显示将要删除的文件并要求确认。如果您确定要删除这个包，因为您已经检查了它的依赖关系，并且知道没有其他东西在使用它，那么您可以传递一个`-y`开关来抑制文件列表和确认对话框:

*   [*视窗*](#windows-22)
**   [**Linux + macOS**](#linux-macos-22)*

```py
(venv) C:\> python -m pip uninstall urllib3 -y
```

```py
(venv) $ python3 -m pip uninstall urllib3 -y
```

这里你卸载`urllib3`。使用`-y`开关，您抑制询问您是否要卸载该软件包的确认对话框。

在一次调用中，您可以指定要卸载的所有软件包:

*   [*视窗*](#windows-23)
**   [**Linux + macOS**](#linux-macos-23)*

```py
(venv) C:\> python -m pip uninstall -y charset-normalizer idna requests
```

```py
(venv) $ python3 -m pip uninstall -y charset-normalizer idna requests
```

您可以向`pip uninstall`命令传递多个包。如果您没有添加任何额外的开关，那么您需要确认卸载每个软件包。通过`-y`开关，你可以卸载它们，而不需要任何确认对话框。

您也可以通过提供`-r <requirements file>`选项来卸载需求文件中列出的所有包。该命令将提示对每个包的确认请求，但您可以使用`-y`开关抑制它:

*   [*视窗*](#windows-24)
**   [**Linux + macOS**](#linux-macos-24)*

```py
(venv) C:\> python -m pip uninstall -r requirements.txt -y
```

```py
(venv) $ python3 -m pip uninstall -r requirements.txt -y
```

请记住，始终检查要卸载的软件包的依赖关系。您可能希望卸载所有依赖项，但是卸载其他人使用的包会破坏您的工作环境。因此，您的项目可能不再正常工作。

如果您在虚拟环境中工作，只需创建一个新的虚拟环境就可以减少工作量。然后，您可以安装您需要的软件包，而不是尝试卸载您不需要的软件包。然而，当您需要从您的系统 Python 安装中卸载一个包时，`pip uninstall`会非常有用。如果您不小心在系统范围内安装了一个软件包，使用`pip uninstall`是清理系统的好方法。

[*Remove ads*](/account/join/)

## 探索`pip`的替代品

Python 社区提供了优秀的工具和库供您在`pip`之外使用。这些包括试图简化和改进包管理的`pip`的替代方案。

以下是 Python 可用的一些其他包管理工具:

| 工具 | 描述 |
| --- | --- |
| [康达](https://conda.io/en/latest/) | Conda 是许多语言的包、依赖和环境管理器，包括 Python。它来自 [Anaconda](https://www.anaconda.com/) ，最初是 Python 的一个数据科学包。因此，它被广泛用于数据科学和[机器学习应用](https://realpython.com/python-windows-machine-learning-setup/)。Conda 运行自己的[索引](https://repo.continuum.io/)来托管兼容的包。 |
| [诗歌](https://python-poetry.org/) | 如果你来自 [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript) 和 [npm](https://www.npmjs.com/) ，那么诗歌对你来说会非常熟悉。poem 超越了[包管理](https://realpython.com/dependency-management-python-poetry/)，帮助您为您的应用程序和库构建发行版，并将它们部署到 PyPI。 |
| [Pipenv](https://github.com/pypa/pipenv) | **Pipenv** 是另一个软件包管理工具，它将虚拟环境和包管理合并在一个工具中。 [Pipenv:新 Python 打包工具指南](https://realpython.com/pipenv-guide/)是开始学习 Pipenv 及其包管理方法的好地方。 |

只有`pip`捆绑在标准 Python 安装中。如果您想使用上面列出的任何替代方法，那么您必须遵循文档中的安装指南。有了这么多的选项，您一定会找到适合您编程之旅的工具！

## 结论

许多 Python 项目使用`pip`包管理器来管理它们的依赖关系。它包含在 Python 安装程序中，是 Python 中依赖性管理的基本工具。

**在本教程中，您学习了如何:**

*   **在您的工作环境中设置**并运行`pip`
*   修复与使用`pip`相关的**常见错误**
*   **用`pip`安装和卸载**软件包
*   **为您的项目和应用定义需求**
*   **需求文件**中的引脚依赖关系

此外，您还了解了保持依赖关系最新的重要性，以及可以帮助您管理这些依赖关系的`pip`的替代方案。

通过仔细查看`pip`，您已经探索了 Python 开发工作流中的一个基本工具。使用`pip`，你可以安装和管理你在 [PyPI](https://pypi.org/) 上找到的任何额外的包。您可以使用其他开发人员的外部包作为需求，并专注于使您的项目独一无二的代码。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Pip 入门**](/courses/what-is-pip/)**********************************************************************************