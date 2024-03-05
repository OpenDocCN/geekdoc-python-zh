# Python 虚拟环境:入门

> 原文：<https://realpython.com/python-virtual-environments-a-primer/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。与书面教程一起观看，以加深您的理解: [**使用 Python 虚拟环境**](/courses/working-python-virtual-environments/)

在本教程中，你将学习如何使用 [Python 的`venv`模块](https://docs.python.org/3/library/venv.html)为你的 Python 项目创建和管理单独的[虚拟环境](https://docs.python.org/3/library/venv.html#venv-def)。每个环境都可以使用不同版本的包依赖项和 Python。在您学会使用虚拟环境之后，您将知道如何帮助其他程序员复制您的开发设置，并且您将确保您的项目不会导致相互之间的依赖冲突。

**本教程结束时，你将知道如何:**

*   **创建**和**激活**一个 **Python 虚拟环境**
*   解释**为什么**你想**隔离外部依赖**
*   **当你创建一个虚拟环境时，想象 Python 做了什么**
*   **使用**可选参数**到`venv`定制**你的虚拟环境
*   **停用**和**移除**虚拟环境
*   选择**用于管理**您的 Python 版本和虚拟环境的附加工具

虚拟环境是 Python 开发中常用且有效的技术。更好地理解它们是如何工作的，为什么需要它们，以及可以用它们做什么，将有助于您掌握 Python 编程工作流。

**免费奖励:** ，向您展示如何使用 Pip、PyPI、Virtualenv 和需求文件等工具避免常见的依赖管理问题。

在整个教程中，您可以为 Windows、Ubuntu Linux 或 macOS 选择代码示例。在相关代码块的右上角选择您的平台，以获得您需要的命令，如果您想了解如何在其他操作系统上使用 Python 虚拟环境，可以随意在选项之间切换。

***参加测验:****通过我们的交互式“Python 虚拟环境:入门”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-virtual-environments-a-primer/)

## 如何使用 Python 虚拟环境？

如果您只是需要启动并运行一个 Python 虚拟环境来继续您最喜欢的项目，那么这一部分是您的最佳选择。

本教程中的指令使用 [Python 的`venv`模块](https://docs.python.org/3/library/venv.html)来创建虚拟环境。这个模块是 Python 标准库的一部分，也是自 Python 3.5 以来官方推荐的创建虚拟环境的方法。

**注意:**还有其他很棒的第三方工具来创建虚拟环境，比如 [conda](#the-conda-package-and-environment-manager) 和 [virtualenv](#the-virtualenv-project) ，你可以在本教程的后面了解更多。这些工具都可以帮助您设置 Python 虚拟环境。

对于基本用法，`venv`是一个很好的选择，因为它已经打包在 Python 安装中。记住这一点，您就可以在本教程中创建您的第一个虚拟环境了。

[*Remove ads*](/account/join/)

### 创建它

任何时候，当你在使用外部依赖项的 Python 项目上工作时，最好先创建一个虚拟环境，这些外部依赖项是你用`pip` 安装的[:](https://realpython.com/what-is-pip/)

*   [*视窗*](#windows-1)
**   [*Linux*](#linux-1)**   [*macOS*](#macos-1)**

```py
PS> python -m venv venv
```

如果您正在 Windows 上[使用 Python，并且还没有配置`PATH`和`PATHEXT`变量，那么您可能需要提供 Python 可执行文件的完整路径:](https://docs.python.org/3/using/windows.html#using-on-windows)

```py
PS> C:\Users\Name\AppData\Local\Programs\Python\Python310\python -m venv venv
```

上面显示的系统路径假设您使用 [Python 下载页面](https://www.python.org/downloads/)提供的 Windows installer 安装了 Python 3.10。您系统上 Python 可执行文件的路径可能不同。使用 PowerShell，您可以使用`where.exe python`命令找到路径。

```py
$ python3 -m venv venv
```

许多 Linux 操作系统都附带了 Python 3 版本。如果`python3`不起作用，那么您必须首先[安装 Python](https://realpython.com/installing-python/) ，并且您可能需要使用您安装的可执行版本的特定名称，例如 Python 3.10.x 的`python3.10`，如果您是这种情况，请记住将代码块中提到的`python3`替换为您的特定版本号。

```py
$ python3 -m venv venv
```

旧版本的 macOS 系统安装了 Python 2.7.x，你应该*永远不要*用它来运行你的脚本。如果你在 macOS < 12.3 上工作，用`python`而不是`python3`调用 Python 解释器，那么你可能会意外地启动过时的系统 Python 解释器。

如果运行`python3`不起作用，那么你必须首先[安装一个现代版本的 Python](https://realpython.com/installing-python/) 。

### 激活它

太好了！现在，您的项目有了自己的虚拟环境。通常，在您开始使用它之前，您将首先通过执行安装附带的脚本来激活环境:

*   [*视窗*](#windows-2)
**   [**Linux + macOS**](#linux-macos-2)*

```py
PS> venv\Scripts\activate
(venv) PS>
```

```py
$ source venv/bin/activate
(venv) $
```

在运行此命令之前，请确保您位于包含刚刚创建的虚拟环境的文件夹中。

**注意:**您也可以在不激活虚拟环境的情况下使用它。为此，您将在执行命令时[向其 Python 解释器提供完整路径](#it-runs-from-anywhere-with-absolute-paths)。然而，最常见的情况是，您希望在创建虚拟环境之后激活它，以节省重复输入长路径的努力。

一旦您可以在命令提示符中看到您的虚拟环境的名称，那么您就知道您的虚拟环境是活动的。您已经准备好安装您的外部软件包了！

### 将软件包安装到其中

创建并激活虚拟环境后，现在可以安装项目所需的任何外部依赖项:

*   [*视窗*](#windows-3)
**   [**Linux + macOS**](#linux-macos-3)*

```py
(venv) PS> python -m pip install <package-name>
```

```py
(venv) $ python -m pip install <package-name>
```

这个命令是默认命令，您应该使用它来安装带有`pip`的外部 Python 包。因为您首先创建并激活了虚拟环境，`pip`将在一个隔离的位置安装软件包。

**注意:**因为您已经使用 Python 3 版本创建了您的虚拟环境，所以您不需要显式调用`python3`或`pip3`。只要你的虚拟环境是活动的，`python`和`pip`链接到与`python3`和`pip3`相同的可执行文件。

恭喜，您现在可以将软件包安装到您的虚拟环境中了。为了达到这一点，首先创建一个名为`venv`的 Python 虚拟环境，然后在当前的 shell 会话中激活它。

只要你不关闭你的终端，你将要安装的每一个 [Python 包](https://realpython.com/python-modules-packages/)将会在这个隔离的环境中结束，而不是你的全局 Python 站点包。这意味着您现在可以在 Python 项目中工作，而不用担心依赖性冲突。

### 停用它

一旦您使用完这个虚拟环境，您就可以停用它:

*   [*视窗*](#windows-4)
**   [**Linux + macOS**](#linux-macos-4)*

```py
(venv) PS> deactivate
PS>
```

```py
(venv) $ deactivate
$
```

执行`deactivate`命令后，您的命令提示符恢复正常。这一变化意味着您已经退出了虚拟环境。如果您现在与 Python 或`pip`交互，您将与您的全局配置的 Python 环境交互。

如果您想回到之前创建的虚拟环境，您需要再次[运行该虚拟环境的激活脚本](#activate-it)。

**注意:**在安装软件包之前，在命令提示符前面的括号中查找您的虚拟环境的名称。在上面的例子中，环境的名称是`venv`。

如果名称出现了，那么您知道您的虚拟环境是活动的，并且您可以安装您的外部依赖项。如果在命令提示符下没有看到这个名称，记得在安装任何包之前激活 Python 虚拟环境。

至此，您已经了解了使用 Python 虚拟环境的基本知识。如果这就是你所需要的，那么当你继续创作的时候，祝你快乐！

但是，如果你想知道刚刚到底发生了什么，为什么那么多教程一开始就要求你创建一个虚拟环境，真正的 Python 虚拟环境*是什么*，那就继续看下去吧！你要深入了！

[*Remove ads*](/account/join/)

## 为什么需要虚拟环境？

Python 社区中的几乎每个人都建议您在所有项目中使用虚拟环境。但是为什么呢？如果您想知道为什么首先需要建立一个 Python 虚拟环境，那么这是适合您的部分。

简单的回答是 Python 不擅长依赖管理。如果您不指定，那么`pip`会将您安装的所有外部包放在 Python 基础安装中的一个名为`site-packages/`的文件夹中。



从技术上讲，Python 附带了两个站点包文件夹:

1.  **`purelib/`** 应该只包含纯 Python 代码编写的模块。
2.  **`platlib/`** 应该包含非纯 Python 编写的二进制文件，例如`.dll`、`.so`或`.pydist`文件。

如果您正在使用 Fedora 或 RedHat Linux 发行版，您可以在不同的位置找到这些文件夹。

然而，大多数操作系统实现 Python 的 site-packages 设置，以便两个位置指向相同的路径，有效地创建单个 site-packages 文件夹。

您可以使用`sysconfig`检查路径:

*   [*视窗*](#windows-5)
**   [*Linux*](#linux-5)**   [*macOS*](#macos-5)**

***>>>

```py
>>> import sysconfig
>>> sysconfig.get_path("purelib")
'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages'
>>> sysconfig.get_path("platlib")
'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages'
```

>>>

```py
>>> import sysconfig
>>> sysconfig.get_path("purelib")
'/home/name/path/to/venv/lib/python3.10/site-packages'
>>> sysconfig.get_path("platlib")
'/home/name/path/to/venv/lib/python3.10/site-packages'
```

>>>

```py
>>> import sysconfig
>>> sysconfig.get_path("purelib")
'/Users/name/path/to/venv/lib/python3.10/site-packages'
>>> sysconfig.get_path("platlib")
'/Users/name/path/to/venv/lib/python3.10/site-packages'
```

最有可能的是，两个输出将向您显示相同的路径。如果两个输出相同，那么你的操作系统不会把`purelib`模块放到与`platlib`模块不同的文件夹中。如果出现两个不同的路径，那么您的操作系统会进行区分。

即使你的操作系统区分了这两者，依赖冲突仍然会出现，因为所有的`purelib`模块将进入`purelib`模块的单一位置，同样的情况也会发生在`platlib`模块上。

要使用虚拟环境，您不需要担心单个 site-packages 文件夹或两个独立文件夹的实现细节。事实上，你可能再也不需要去想它了。然而，如果你愿意，你可以记住，当有人提到 Python 的站点包目录时，他们*可能*在谈论两个不同的目录。***  ***如果所有的外部包都在同一个文件夹中，会出现几个问题。在本节中，您将了解更多关于它们的信息，以及虚拟环境可以缓解的其他问题。

### 避免系统污染

Linux 和 macOS 预装了 Python 的一个版本，操作系统使用该版本执行内部任务。

如果将包安装到操作系统的全局 Python 中，这些包将与系统相关的包混合在一起。这种混淆可能会对操作系统正常运行的关键任务产生意想不到的副作用。

此外，如果您更新操作系统，那么您安装的软件包可能会被覆盖并丢失。你不会希望这两种头痛发生的！

### 回避依赖冲突

您的一个项目可能需要不同版本的外部库。如果你只有一个地方可以安装软件包，那么你就不能使用同一个库的两个不同版本。这是推荐使用 Python 虚拟环境的最常见原因之一。

为了更好地理解为什么这如此重要，想象一下你正在为两个不同的客户构建 Django 网站。一个客户对他们现有的 web 应用程序很满意，这是你最初使用 [Django 2.2.26](https://pypi.org/project/Django/2.2.26/) 构建的，而那个客户拒绝将他们的项目更新到现代的 Django 版本。另一个客户想让你在他们的网站上包含异步功能，这个功能只能从 Django 4.0 开始使用。

如果您全局安装了 Django，那么您只能安装两个版本中的一个:

*   [*视窗*](#windows-6)
**   [**Linux + macOS**](#linux-macos-6)*

```py
PS> python -m pip install django==2.2.26
PS> python -m pip list
Package    Version
---------- -------
Django     2.2.26
pip        22.0.4
pytz       2022.1
setuptools 58.1.0
sqlparse   0.4.2

PS> python -m pip install django==4.0.3
PS> python -m pip list
Package    Version
---------- -------
asgiref    3.5.0
Django     4.0.3
pip        22.0.4
pytz       2022.1
setuptools 58.1.0
sqlparse   0.4.2
tzdata     2022.1
```

```py
$ python3 -m pip install django==2.2.26
$ python3 -m pip list
Package    Version
---------- -------
Django     2.2.26
pip        22.0.4
pytz       2022.1
setuptools 58.1.0
sqlparse   0.4.2

$ python3 -m pip install django==4.0.3
$ python3 -m pip list
Package    Version
---------- -------
asgiref    3.5.0
Django     4.0.3
pip        22.0.4
pytz       2022.1
setuptools 58.1.0
sqlparse   0.4.2
```

如果您将同一个包的两个不同版本安装到您的全局 Python 环境中，第二个安装将覆盖第一个。出于同样的原因，为两个客户端提供一个虚拟环境也是行不通的。在一个 Python 环境中，不能有同一个包的两个不同版本。

看起来你将无法在这两个项目中的任何一个上工作！但是，如果您为每个客户的项目创建一个虚拟环境，那么您可以在每个项目中安装不同版本的 Django:

*   [*视窗*](#windows-7)
**   [**Linux + macOS**](#linux-macos-7)*

```py
PS> mkdir client-old
PS> cd client-old
PS> python -m venv venv --prompt="client-old"
PS> venv\Scripts\activate
(client-old) PS> python -m pip install django==2.2.26
(client-old) PS> python -m pip list
Package    Version
---------- -------
Django     2.2.26
pip        22.0.4
pytz       2022.1
setuptools 58.1.0
sqlparse   0.4.2
(client-old) PS> deactivate

PS> cd ..
PS> mkdir client-new
PS> cd client-new
PS> python -m venv venv --prompt="client-new"
PS> venv\Scripts\activate
(client-new) PS> python -m pip install django==4.0.3
(client-new) PS> python -m pip list
Package    Version
---------- -------
asgiref    3.5.0
Django     4.0.3
pip        22.0.4
setuptools 58.1.0
sqlparse   0.4.2
tzdata     2022.1
(client-new) PS> deactivate
```

```py
$ mkdir client-old
$ cd client-old
$ python3 -m venv venv --prompt="client-old"
$ source venv/bin/activate
(client-old) $ python -m pip install django==2.2.26
(client-old) $ python -m pip list
Package    Version
---------- -------
Django     2.2.26
pip        22.0.4
pytz       2022.1
setuptools 58.1.0
sqlparse   0.4.2
(client-old) $ deactivate

$ cd ..
$ mkdir client-new
$ cd client-new
$ python3 -m venv venv --prompt="client-new"
$ source venv/bin/activate
(client-new) $ python -m pip install django==4.0.3
(client-new) $ python -m pip list
Package    Version
---------- -------
asgiref    3.5.0
Django     4.0.3
pip        22.0.4
setuptools 58.1.0
sqlparse   0.4.2
(client-new) $ deactivate
```

如果您现在激活这两个虚拟环境中的任何一个，那么您会注意到它仍然拥有自己特定的 Django 版本。这两个环境也有不同的依赖项，每个环境都只包含 Django 版本所必需的依赖项。

通过这种设置，您可以在处理一个项目时激活一个环境，而在处理另一个项目时激活另一个环境。现在，您可以同时让任意数量的客户满意！

[*Remove ads*](/account/join/)

### 最小化再现性问题

如果你所有的包都在一个地方，那么就很难找到与单个项目相关的依赖关系。

如果您使用 Python 已经有一段时间了，那么您的全局 Python 环境可能已经包含了各种第三方包。如果不是这样，那就拍拍自己的背吧！你可能最近安装了新版本的 Python，或者你已经知道如何处理虚拟环境来避免系统污染。

为了阐明在多个项目间共享 Python 环境时可能遇到的再现性问题，接下来我们将看一个例子。想象一下，在过去的一个月里，你从事两个独立的项目:

1.  [一个美汤的网页抓取项目](https://realpython.com/beautiful-soup-web-scraper-python/)
2.  [烧瓶应用](https://realpython.com/python-web-applications-with-flask-part-i/)

在不知道虚拟环境的情况下，您将所有必需的包安装到了您的全局 Python 环境中:

*   [*视窗*](#windows-8)
**   [**Linux + macOS**](#linux-macos-8)*

```py
PS> python -m pip install beautifulsoup4 requests
PS> python -m pip install flask
```

```py
$ python3 -m pip install beautifulsoup4 requests
$ python3 -m pip install flask
```

你的 Flask 应用程序非常有用，所以其他开发人员也想开发它。他们需要重现你工作时的环境。您想要继续并[固定您的依赖关系](#pin-your-dependencies)，以便您可以在线共享您的项目:

*   [*视窗*](#windows-9)
**   [**Linux + macOS**](#linux-macos-9)*

```py
PS> python -m pip freeze
beautifulsoup4==4.10.0
certifi==2021.10.8
charset-normalizer==2.0.12
click==8.0.4
colorama==0.4.4
Flask==2.0.3
idna==3.3
itsdangerous==2.1.1
Jinja2==3.0.3
MarkupSafe==2.1.1
requests==2.27.1
soupsieve==2.3.1
urllib3==1.26.9
Werkzeug==2.0.3
```

```py
$ python3 -m pip freeze
beautifulsoup4==4.10.0
certifi==2021.10.8
charset-normalizer==2.0.12
click==8.0.4
Flask==2.0.3
idna==3.3
itsdangerous==2.1.1
Jinja2==3.0.3
MarkupSafe==2.1.1
requests==2.27.1
soupsieve==2.3.1
urllib3==1.26.9
Werkzeug==2.0.3
```

这些包中哪些与你的 Flask 应用相关，哪些是因为你的网络抓取项目而出现的？很难判断什么时候所有的外部依赖都存在于一个桶中。

对于像这样的单一环境，您必须手动检查依赖项，并知道哪些是您的项目所必需的，哪些不是。充其量，这种方法是乏味的，但更有可能的是，它容易出错。

如果您为您的每个项目使用一个单独的虚拟环境，那么从您的固定依赖项中读取项目需求会更简单。这意味着当你开发一个伟大的应用程序时，你可以分享你的成功，让其他人有可能与你合作！

### 躲避安装特权锁定

最后，您可能需要计算机上的管理员权限才能将包安装到主机 Python 的 site-packages 目录中。在公司的工作环境中，您很可能无法访问您正在使用的机器。

如果您使用虚拟环境，那么您可以在您的用户权限范围内创建一个新的安装位置，这允许您安装和使用外部软件包。

无论你是把在自己的机器上编程作为一种爱好，还是为客户开发网站，或者在公司环境中工作，从长远来看，使用虚拟环境将会为你省去很多麻烦。

## 什么是 Python 虚拟环境？

此时，您确信想要使用虚拟环境。很好，但是当您使用虚拟环境时，您在使用什么呢？如果您想了解什么是 Python 虚拟环境，那么这是适合您的部分。

简而言之，Python 虚拟环境是一个文件夹结构，它为您提供了运行轻量级且独立的 Python 环境所需的一切。

[*Remove ads*](/account/join/)

### 文件夹结构

当您使用`venv`模块创建一个新的虚拟环境时，Python 会创建一个自包含的文件夹结构，并将 Python [可执行文件](https://en.wikipedia.org/wiki/Executable)复制或[符号链接](https://en.wikipedia.org/wiki/Symbolic_link)到该文件夹结构中。

你不需要深入研究这个文件夹结构来了解更多关于虚拟环境是由什么组成的。一会儿，你会小心翼翼地刮掉表层土，调查你发现的高层结构。

然而，如果你已经准备好铲子，并且你渴望挖掘，那么打开下面的可折叠部分:



欢迎，勇敢的人。您已经接受了挑战，更深入地探索虚拟环境的文件夹结构！在这个可折叠的部分，你会找到如何深入黑暗深渊的指导。

在命令行中，导航到包含虚拟环境的文件夹。深呼吸，振作精神，然后执行`tree`命令，显示目录的所有内容:

*   [*视窗*](#windows-10)
**   [*Linux*](#linux-10)**   [*macOS*](#macos-10)**

```py
PS> tree venv /F
```

```py
$ tree venv
```

你可能需要先[安装`tree`](https://gitlab.com/OldManProgrammer/unix-tree) ，比如用`sudo apt install tree`。

```py
$ tree venv
```

你可能需要先用[安装`tree`](https://gitlab.com/OldManProgrammer/unix-tree) ，比如用[自制](https://formulae.brew.sh/formula/tree#default)。

`tree`命令以一个*非常*长的树形结构显示你的`venv`目录的内容。

**注意:**或者，你可以通过创建一个新的虚拟环境来磨练你的技能，在其中安装 [`rptree`包](https://github.com/realpython/rptree)，并使用它来显示文件夹的树形结构。你甚至可以走一个更大的弯路，然后[自己构建目录树生成器](https://realpython.com/directory-tree-generator-python/)！

然而，当你最终显示出`venv/`文件夹的所有内容时，你可能会对你的发现感到惊讶。许多开发人员在第一次看的时候都会有轻微的震惊。那里有很多 T2 的文件！

如果这是你的第一次，你也有这种感觉，那么欢迎加入这个看了一眼就不知所措的群体。***  ***虚拟环境文件夹包含许多文件和文件夹，但您可能会注意到，使这种树形结构如此之长的大部分内容都在`site-packages/`文件夹中。如果您减少其中的子文件夹和文件，最终会得到一个不太大的树形结构:

*   [*视窗*](#windows-11)
**   [*Linux*](#linux-11)**   [*macOS*](#macos-11)**

```py
venv\
│
├── Include\
│
├── Lib\
│   │
│   └── site-packages\
│       │
│       ├── _distutils_hack\
│       │
│       ├── pip\
│       │
│       ├── pip-22.0.4.dist-info\
│       │
│       ├── pkg_resources\
│       │
│       ├── setuptools\
│       │
│       ├── setuptools-58.1.0.dist-info\
│       │
│       └── distutils-precedence.pth
│
│
├── Scripts\
│   ├── Activate.ps1
│   ├── activate
│   ├── activate.bat
│   ├── deactivate.bat
│   ├── pip.exe
│   ├── pip3.10.exe
│   ├── pip3.exe
│   ├── python.exe
│   └── pythonw.exe
│
└── pyvenv.cfg
```

```py
venv/
│
├── bin/
│   ├── Activate.ps1
│   ├── activate
│   ├── activate.csh
│   ├── activate.fish
│   ├── pip
│   ├── pip3
│   ├── pip3.10
│   ├── python
│   ├── python3
│   └── python3.10
│
├── include/
│
├── lib/
│   │
│   └── python3.10/
│       │
│       └── site-packages/
│           │
│           ├── _distutils_hack/
│           │
│           ├── pip/
│           │
│           ├── pip-22.0.4.dist-info/
│           │
│           ├── pkg_resources/
│           │
│           ├── setuptools/
│           │
│           ├── setuptools-58.1.0.dist-info/
│           │
│           └── distutils-precedence.pth
│
├── lib64/
│   │
│   └── python3.10/
│       │
│       └── site-packages/
│           │
│           ├── _distutils_hack/
│           │
│           ├── pip/
│           │
│           ├── pip-22.0.4.dist-info/
│           │
│           ├── pkg_resources/
│           │
│           ├── setuptools/
│           │
│           ├── setuptools-58.1.0.dist-info/
│           │
│           └── distutils-precedence.pth
│
└── pyvenv.cfg
```

```py
venv/
│
├── bin/
│   ├── Activate.ps1
│   ├── activate
│   ├── activate.csh
│   ├── activate.fish
│   ├── pip
│   ├── pip3
│   ├── pip3.10
│   ├── python
│   ├── python3
│   └── python3.10
│
├── include/
│
├── lib/
│   │
│   └── python3.10/
│       │
│       └── site-packages/
│           │
│           ├── _distutils_hack/
│           │
│           ├── pip/
│           │
│           ├── pip-22.0.4.dist-ino/
│           │
│           ├── pkg_resources/
│           │
│           ├── setuptools/
│           │
│           ├── setuptools-58.1.0.dist-info/
│           │
│           └── distutils-precedence.pth
│
└── pyvenv.cfg
```

这种简化的树结构让您可以更好地了解虚拟环境文件夹中的情况:

*   [*视窗*](#windows-12)
**   [*Linux*](#linux-12)**   [*macOS*](#macos-12)*

***   **`Include\`** 是一个最初为空的文件夹，Python 使用它来[包含 C 头文件](https://docs.python.org/3/c-api/intro.html#include-files)，这些文件是你可能安装的依赖于 C 扩展的包。

*   **`Lib\`** 包含`site-packages\`文件夹，这是创建你的虚拟环境的主要原因之一。此文件夹是您安装要在虚拟环境中使用的外部包的位置。默认情况下，您的虚拟环境预装了两个依赖项，`pip`和 setuptools。过一会儿你会学到更多关于它们的知识。

*   **`Scripts\`** 包含你的虚拟环境的可执行文件。最值得注意的是 Python 解释器(`python.exe`)、`pip`可执行文件(`pip.exe`)和用于虚拟环境的激活脚本，它们有几种不同的风格，允许您使用不同的 shells。在本教程中，您已经使用了`activate`，它在大多数 shells 中处理 Windows 虚拟环境的激活。

*   **`pyvenv.cfg`** 对于你的虚拟环境来说是一个至关重要的文件。它只包含几个键值对，Python 使用这些键值对来设置`sys`模块中的变量，这些变量决定当前 Python 会话将使用哪个 Python 解释器和哪个站点包目录。当您阅读虚拟环境如何工作时，您将了解到关于该文件中设置的更多信息。

*   **`bin/`** 包含你的虚拟环境的可执行文件。最值得注意的是 Python 解释器(`python`)和`pip`可执行文件(`pip`，以及它们各自的符号链接(`python3`、`python3.10`、`pip3`、`pip3.10`)。该文件夹还包含虚拟环境的激活脚本。具体的激活脚本取决于您使用的 shell。例如，在本教程中，您运行了适用于 Bash 和 Zsh shells 的`activate`。

*   **`include/`** 是一个最初为空的文件夹，Python 使用它来[包含 C 头文件](https://docs.python.org/3/c-api/intro.html#include-files)，这些文件是你可能安装的依赖于 C 扩展的包。

*   **`lib/`** 包含嵌套在指定 Python 版本(`python3.10/`)的文件夹中的`site-packages/`目录。`site-packages/`是创建虚拟环境的主要原因之一。此文件夹是您安装要在虚拟环境中使用的外部包的位置。默认情况下，您的虚拟环境预装了两个依赖项，`pip`和 setuptools。过一会儿你会学到更多关于它们的知识。

*   出于兼容性的考虑，许多 Linux 系统中的 **`lib64/`** 是`lib/` [的符号链接。一些 Linux 系统可能使用`lib/`和`lib64/`之间的区别来安装不同版本的库，这取决于它们的架构。](https://stackoverflow.com/a/11370995/5717580)

*   **`pyvenv.cfg`** 对于你的虚拟环境来说是一个至关重要的文件。它只包含几个键值对，Python 使用这些键值对来设置`sys`模块中的变量，这些变量决定当前 Python 会话将使用哪个 Python 解释器和哪个站点包目录。当您阅读虚拟环境如何工作时，您将了解到关于该文件中设置的更多信息。

*   **`bin/`** 包含你的虚拟环境的可执行文件。最值得注意的是 Python 解释器(`python`)和`pip`可执行文件(`pip`，以及它们各自的符号链接(`python3`、`python3.10`、`pip3`、`pip3.10`)。该文件夹还包含虚拟环境的激活脚本。具体的激活脚本取决于您使用的 shell。例如，在本教程中，您运行了适用于 Bash 和 Zsh shells 的`activate`。

*   **`include/`** 是一个最初为空的文件夹，Python 使用它来[包含 C 头文件](https://docs.python.org/3/c-api/intro.html#include-files)，这些文件是你可能安装的依赖于 C 扩展的包。

*   **`lib/`** 包含嵌套在指定 Python 版本(`python3.10/`)的文件夹中的`site-packages/`目录。`site-packages/`是创建虚拟环境的主要原因之一。此文件夹是您安装要在虚拟环境中使用的外部包的位置。默认情况下，您的虚拟环境预装了两个依赖项，`pip`和 setuptools。过一会儿你会学到更多关于它们的知识。

*   **`pyvenv.cfg`** 对于你的虚拟环境来说是一个至关重要的文件。它只包含几个键值对，Python 使用这些键值对来设置`sys`模块中的变量，这些变量决定当前 Python 会话将使用哪个 Python 解释器和哪个站点包目录。当您阅读虚拟环境如何工作时，您将了解到关于该文件中设置的更多信息。

从这个虚拟环境文件夹内容的鸟瞰图中，您可以进一步缩小以发现 Python 虚拟环境有三个基本部分:

1.  **Python 二进制文件**的副本或符号链接
2.  一个 **`pyvenv.cfg`文件**
3.  一个**站点包目录**

在`site-packages/`中安装的软件包是可选的，但是作为一个合理的缺省值。然而，如果这个目录是空的，你的虚拟环境仍然是一个有效的虚拟环境，并且有办法创建它[而不安装任何依赖](#avoid-installing-pip)。

在默认设置下，`venv`将同时安装`pip`和 setuptools。使用`pip`是 Python 中安装包的推荐方式，setuptools 是[对`pip`](https://github.com/pypa/pip/blob/main/pyproject.toml) 的依赖。因为安装其他包是 Python 虚拟环境中最常见的用例，所以您会想要访问`pip`。

您可以使用`pip list`仔细检查 Python 是否在您的虚拟环境中安装了`pip`和 setuptools:

*   [*视窗*](#windows-13)
**   [**Linux + macOS**](#linux-macos-13)*

```py
(venv) PS> python -m pip list
Package    Version
---------- -------
pip        22.0.4
setuptools 58.1.0
```

```py
(venv) $ python -m pip list
Package    Version
---------- -------
pip        22.0.4
setuptools 58.1.0
```

您的版本号可能会有所不同，但是这个输出确认了当您使用默认设置创建虚拟环境时，Python 安装了这两个包。

**注意:**在这个输出下面，`pip`可能还会显示一个警告，提示您没有使用最新版本的模块。先别担心这个。稍后，您将了解为什么会发生这种情况，以及如何在创建虚拟环境时[自动更新`pip`](#update-the-core-dependencies) 。

这两个已安装的包构成了新虚拟环境的大部分内容。然而，您会注意到在`site-packages/`目录中还有几个其他的文件夹:

*   **`_distutils_hack/`模块**，[以一种名副其实的方式](https://github.com/pypa/setuptools/blob/main/_distutils_hack/)确保当执行包安装时，Python 选择 setuptools 的本地`._distutils`子模块而不是标准库的`distutils`模块。

*   **`pkg_resources/`模块**帮助应用程序自动发现插件并允许 Python 包访问它们的资源文件。和`setuptools` 一起分发的是[。](https://setuptools.pypa.io/en/latest/pkg_resources.html)

*   **`pip`和 setuptools 的`{name}-{version}.dist-info/`目录**包含[包分发](https://realpython.com/pypi-publish-python-package/)信息，存在[中记录已安装包](https://packaging.python.org/en/latest/specifications/recording-installed-packages/#the-dist-info-directory)的信息。

最后，还有一个名为 **`distutils-precedence.pth`** 的文件。该文件帮助设置`distutils`导入的路径优先级，并与`_distutils_hack`一起工作，以确保 Python 更喜欢与 setuptools 捆绑在一起的`distutils`版本，而不是内置版本。

**注意:**为了完整起见，你正在学习这些额外的文件和文件夹。您不需要记住它们就能有效地在虚拟环境中工作。记住你的`site-packages/`目录中的任何预装软件包都是标准工具，它们使得安装*其他*软件包更加用户友好。

至此，如果您已经使用内置的`venv`模块安装了 Python 虚拟环境，那么您已经看到了组成它的所有文件和文件夹。

请记住，您的虚拟环境只是一个文件夹结构，这意味着您可以随时删除和重新创建它。但是为什么*会有这种特定的*文件夹结构，以及它使什么成为可能？

[*Remove ads*](/account/join/)

### 一个独立的 Python 安装

Python 虚拟环境旨在提供一个轻量级、隔离的 Python 环境，您可以快速创建该环境，然后在不再需要它时将其丢弃。您在上面看到的文件夹结构通过提供三个关键部分使之成为可能:

1.  **Python 二进制文件**的副本或符号链接
2.  一个 **`pyvenv.cfg`文件**
3.  一个**站点包目录**

您希望实现一个隔离的环境，这样您安装的任何外部包都不会与全局站点包冲突。`venv`所做的就是复制标准 Python 安装创建的文件夹结构。

这个结构说明了 Python 二进制文件的副本或符号链接的位置，以及 Python 安装外部包的站点包目录。

**注意:**您的虚拟环境中的 Python 可执行文件是您的环境所基于的 Python 可执行文件的副本还是符号链接主要取决于您的操作系统。

Windows 和 Linux 可能创建符号链接而不是副本，而 macOS 总是创建副本。在创建虚拟环境时，您可以尝试用可选参数[影响默认行为](#copy-or-link-your-executables)。然而，在大多数标准情况下，您不必担心这个问题。

除了 Python 二进制文件和 site-packages 目录之外，还会得到`pyvenv.cfg`文件。这是一个小文件，只包含几个键值对。但是，这些设置对于虚拟环境的运行至关重要:

*   [*视窗*](#windows-14)
**   [*Linux*](#linux-14)**   [*macOS*](#macos-14)**

```py
home = C:\Users\Name\AppData\Local\Programs\Python\Python310
include-system-site-packages = false
version = 3.10.3
```

```py
home = /usr/local/bin
include-system-site-packages = false
version = 3.10.3
```

```py
home = /Library/Frameworks/Python.framework/Versions/3.10/bin
include-system-site-packages = false
version = 3.10.3
```

当阅读关于虚拟环境如何工作的时，您将在后面的章节中了解关于这个文件的更多信息。

假设您仔细检查了新创建的虚拟环境的[文件夹结构](#a-folder-structure)。在这种情况下，您可能会注意到这个轻量级安装不包含任何可信的[标准库](https://docs.python.org/3/library/)模块。有人可能会说，没有标准库的 Python 就像一辆没有电池的玩具车！

然而，如果您从虚拟环境中启动 Python 解释器，那么您仍然可以访问标准库中的所有好东西:

>>>

```py
>>> import urllib
>>> from pprint import pp
>>> pp(dir(urllib))
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__']
```

在上面的示例代码片段中，您已经成功地从[漂亮打印模块](https://realpython.com/python-pretty-print/)中 [`urllib`模块](https://realpython.com/urllib-request/)和`pp()`快捷方式[导入了](https://realpython.com/python-import/)。然后你用`dir()`检查`urllib`模块。

这两个模块都是标准库的一部分，那么即使它们不在您的 Python 虚拟环境的文件夹结构中，您怎么能访问它们呢？

您可以访问 Python 的标准库模块，因为您的虚拟环境重用 Python 的内置模块和 Python 安装中的标准库模块，您可以从 Python 安装中创建您的虚拟环境。在后面的部分中，您将了解虚拟环境如何实现链接到 Python 标准库的 T2。

**注意:**因为您总是需要现有的 Python 安装来创建您的虚拟环境，`venv`选择重用现有的标准库模块，以避免将它们复制到您的新虚拟环境中的开销。这个有意的决定加速了虚拟环境的创建，并使它们更加轻量级，正如 PEP 405 的[动机中所描述的。](https://www.python.org/dev/peps/pep-0405/#motivation)

除了标准的库模块，您还可以在创建环境时通过一个参数[让您的虚拟环境访问基本安装的站点包](#include-the-system-site-packages):

*   [*视窗*](#windows-15)
**   [**Linux + macOS**](#linux-macos-15)*

```py
PS C:\> python -m venv venv --system-site-packages
```

```py
$ python3 -m venv venv --system-site-packages
```

如果在调用`venv`时加上`--system-site-packages`，Python 会将`pyvenv.cfg`到`true`中的值设置为`include-system-site-packages`。这个设置意味着您可以使用安装到您的基本 Python 中的任何外部包，就像您已经将它们安装到您的虚拟环境中一样。

这种连接只在一个方向起作用。即使您允许虚拟环境访问源 Python 的 site-packages 文件夹，您安装到虚拟环境中的任何新包都不会与那里的包混合在一起。Python 将尊重虚拟环境中安装的隔离特性，并将它们放入虚拟环境中单独的站点包目录中。

您知道 Python 虚拟环境只是一个带有设置文件的文件夹结构。它可能预装了`pip`,也可能没有，它可以访问源代码 Python 的 site-packages 目录，同时保持隔离。但是你可能想知道所有这些是如何工作的。

[*Remove ads*](/account/join/)

## 虚拟环境是如何工作的？

如果您知道什么是 Python 虚拟环境，但是想知道它如何设法创建它所提供的轻量级隔离，那么您就在正确的部分。在这里，您将了解到文件夹结构和`pyvenv.cfg`文件中的设置如何与 Python 交互，从而为安装外部依赖项提供一个可再现的隔离空间。

### 它复制结构和文件

当您使用`venv`创建虚拟环境时，该模块会在您的操作系统上重新创建标准 Python 安装的文件和[文件夹结构](#a-folder-structure)。Python 还将 Python 可执行文件复制或符号链接到那个文件夹结构中，您已经用它调用了`venv`:

*   [*视窗*](#windows-16)
**   [*Linux*](#linux-16)**   [*macOS*](#macos-16)**

```py
venv\
│
├── Include\
│
├── Lib\
│   │
│   └── site-packages\
│
├── Scripts\
│   ├── Activate.ps1
│   ├── activate
│   ├── activate.bat
│   ├── deactivate.bat
│   ├── pip.exe
│   ├── pip3.10.exe
│   ├── pip3.exe
│   ├── python.exe
│   └── pythonw.exe
│
└── pyvenv.cfg
```

```py
venv/
│
├── bin/
│   ├── Activate.ps1
│   ├── activate
│   ├── activate.csh
│   ├── activate.fish
│   ├── pip
│   ├── pip3
│   ├── pip3.10
│   ├── python
│   ├── python3
│   └── python3.10
│
├── include/
│
├── lib/
│   │
│   └── python3.10/
│       │
│       └── site-packages/
│
├── lib64/
│   │
│   └── python3.10/
│       │
│       └── site-packages/
│
└── pyvenv.cfg
```

```py
venv/
│
├── bin/
│   ├── Activate.ps1
│   ├── activate
│   ├── activate.csh
│   ├── activate.fish
│   ├── pip
│   ├── pip3
│   ├── pip3.10
│   ├── python
│   ├── python3
│   └── python3.10
│
├── include/
│
├── lib/
│   │
│   └── python3.10/
│       │
│       └── site-packages/
│
└── pyvenv.cfg
```

如果您在操作系统上找到系统范围的 Python 安装，并检查那里的文件夹结构，那么您将看到您的虚拟环境类似于该结构。

通过导航到在`pyvenv.cfg`中的`home`键下找到的路径，可以找到虚拟环境所基于的 Python 基础安装。

**注意:**在 Windows 上，你可能会注意到 Python 基础安装中的`python.exe`不在`Scripts\`中，而是在上一级文件夹中。在您的虚拟环境中，可执行文件被有意放置在`Scripts\`文件夹中。

这个对文件夹结构的小改变意味着你只需要添加一个目录到你的 shell [`PATH`](https://realpython.com/add-python-to-path/) 变量来激活虚拟环境。

虽然您可能会在 Python 的基础安装中找到一些额外的文件和文件夹，但是您会注意到标准的文件夹结构与虚拟环境中的相同。创建这个文件夹结构是为了确保 Python 可以像预期的那样独立工作，而不需要应用很多额外的修改。

### 它采用前缀查找过程

有了标准的文件夹结构，虚拟环境中的 Python 解释器可以理解所有相关文件的位置。它只根据 [`venv`规范](https://www.python.org/dev/peps/pep-0405/#specification)对其前缀查找过程做了微小的修改。

Python 解释器首先寻找一个`pyvenv.cfg`文件，而不是寻找`os`模块来确定标准库的位置。如果解释器找到这个文件，并且它包含一个`home`键，那么解释器将使用这个键来设置两个变量的值:

1.  [`sys.base_prefix`](https://docs.python.org/3/library/sys.html#sys.base_prefix) 将保存用于创建这个虚拟环境的 Python 可执行文件的路径，你可以在`pyvenv.cfg`中的`home`键下定义的路径中找到。
2.  [`sys.prefix`](https://docs.python.org/3/library/sys.html#sys.prefix) 会指向包含`pyvenv.cfg`的目录。

如果解释器没有找到一个`pyvenv.cfg`文件，那么它确定它没有在虚拟环境中运行，然后`sys.base_prefix`和`sys.prefix`将指向相同的路径。



您可以确认这是按照描述的方式工作的。在活动的虚拟环境中启动 Python 解释器，并检查两个变量:

*   [*视窗*](#windows-17)
**   [*Linux*](#linux-17)**   [*macOS*](#macos-17)**

***>>>

```py
>>> import sys
>>> sys.prefix
'C:\\Users\\Name\\path\\to\\venv'
>>> sys.base_prefix
'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310'
```

>>>

```py
>>> import sys
>>> sys.prefix
'/home/name/path/to/venv'
>>> sys.base_prefix
'/usr/local'
```

>>>

```py
>>> import sys
>>> sys.prefix
'/Users/name/path/to/venv'
>>> sys.base_prefix
'/Library/Frameworks/Python.framework/Versions/3.10'
```

您可以看到变量指向系统中的不同位置。

现在继续并停用虚拟环境，进入一个新的解释器会话，并重新运行相同的代码:

*   [*视窗*](#windows-18)
**   [*Linux*](#linux-18)**   [*macOS*](#macos-18)**

***>>>

```py
>>> import sys
>>> sys.prefix
'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310'
>>> sys.base_prefix
'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310'
```

>>>

```py
>>> import sys
>>> sys.prefix
'/usr/local'
>>> sys.base_prefix
'/usr/local'
```

>>>

```py
>>> import sys
>>> sys.prefix
'/Library/Frameworks/Python.framework/Versions/3.10'
>>> sys.base_prefix
'/Library/Frameworks/Python.framework/Versions/3.10'
```

您应该看到`sys.prefix`和`sys.base_prefix`现在都指向相同的路径。******  *****如果这两个变量有不同的值，那么 Python 会适应它寻找模块的地方:

> 对`site`和`sysconfig`标准库模块进行了修改，使得标准库和头文件相对于`sys.base_prefix` […]找到，而站点包目录[…]仍然相对于`sys.prefix` […]找到。([来源](https://www.python.org/dev/peps/pep-0405/#specification))

这一改变有效地允许虚拟环境中的 Python 解释器使用来自基本 Python 安装的标准库模块，同时指向其内部 site-packages 目录来安装和访问外部包。

### 它链接回你的标准库

Python 虚拟环境旨在成为一种轻量级方式，为您提供一个隔离的 Python 环境，您可以快速创建该环境，然后在不再需要它时将其删除。为了实现这一点，`venv`只复制最不必要的文件:

> 最简单形式的 Python 虚拟环境只包含 Python 二进制文件的副本或符号链接，以及一个`pyvenv.cfg`文件和一个站点包目录。([来源](https://www.python.org/dev/peps/pep-0405/#specification))

虚拟环境中的 Python 可执行文件可以访问环境所基于的 Python 安装的标准库模块。Python 通过在`pyvenv.cfg`中的`home`设置中指向基本 Python 可执行文件的路径来实现这一点:

*   [*视窗*](#windows-19)
**   [*Linux*](#linux-19)**   [*macOS*](#macos-19)**

```py
home = C:\Users\Name\AppData\Local\Programs\Python\Python310 include-system-site-packages = false
version = 3.10.3
```

```py
home = /usr/local/bin include-system-site-packages = false
version = 3.10.3
```

```py
home = /Library/Frameworks/Python.framework/Versions/3.10/bin include-system-site-packages = false
version = 3.10.3
```

如果您导航到`pyvenv.cfg`中高亮显示的行的路径值，并列出文件夹的内容，那么您会找到用于创建虚拟环境的基本 Python 可执行文件。从那里，您可以导航找到包含标准库模块的文件夹:

*   [*视窗*](#windows-20)
**   [*Linux*](#linux-20)**   [*macOS*](#macos-20)**

```py
PS> ls C:\Users\Name\AppData\Local\Programs\Python\Python310

 Directory: C:\Users\Name\AppData\Local\Programs\Python\Python310

Mode              LastWriteTime      Length Name
----              -------------      ------ ----
d-----     12/19/2021   5:09 PM             DLLs
d-----     12/19/2021   5:09 PM             Doc
d-----     12/19/2021   5:09 PM             include
d-----     12/19/2021   5:09 PM             Lib
d-----     12/19/2021   5:09 PM             libs
d-----     12/21/2021   2:04 PM             Scripts
d-----     12/19/2021   5:09 PM             tcl
d-----     12/19/2021   5:09 PM             Tools
-a----      12/7/2021   4:28 AM       32762 LICENSE.txt
-a----      12/7/2021   4:29 AM     1225432 NEWS.txt
-a----      12/7/2021   4:28 AM       98544 python.exe
-a----      12/7/2021   4:28 AM       61680 python3.dll
-a----      12/7/2021   4:28 AM     4471024 python310.dll
-a----      12/7/2021   4:28 AM       97008 pythonw.exe
-a----      12/7/2021   4:29 AM       97168 vcruntime140.dll
-a----      12/7/2021   4:29 AM       37240 vcruntime140_1.dll

PS> ls C:\Users\Name\AppData\Local\Programs\Python\Python310\Lib

 Directory: C:\Users\Name\AppData\Local\Programs\Python\Python310\Lib

Mode              LastWriteTime      Length Name
----              -------------      ------ ----
d-----     12/19/2021   5:09 PM             asyncio
d-----     12/19/2021   5:09 PM             collections

# ...

-a----      12/7/2021   4:27 AM        5302 __future__.py
-a----      12/7/2021   4:27 AM          65 __phello__.foo.py
```

```py
$ ls /usr/local/bin

2to3-3.10         pip3.10           python3.10
idle3.10          pydoc3.10         python3.10-config

$ ls /usr/local/lib/python3.10

$ ls
abc.py                   hmac.py            shelve.py
aifc.py                  html               shlex.py
_aix_support.py          http               shutil.py
antigravity.py           idlelib            signal.py

# ...

graphlib.py              runpy.py           zipimport.py
gzip.py                  sched.py           zoneinfo
hashlib.py               secrets.py
heapq.py                 selectors.py
```

```py
$ ls /Library/Frameworks/Python.framework/Versions/3.10/bin

2to3               pip3.10            python3-intel64
2to3-3.10          pydoc3             python3.10
idle3              pydoc3.10          python3.10-config
idle3.10           python3            python3.10-intel64
pip3               python3-config

$ ls /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/

LICENSE.txt              fnmatch.py             rlcompleter.py
__future__.py            fractions.py           runpy.py
__phello__.foo.py        ftplib.py              sched.py
__pycache__              functools.py           secrets.py

# ...

ensurepip                quopri.py              zipimport.py
enum.py                  random.py              zoneinfo
filecmp.py               re.py
fileinput.py             reprlib.py
```

Python 被设置为通过将相关路径添加到`sys.path`来查找这些模块。在初始化过程中，Python 自动导入[的`site`模块](https://docs.python.org/3.10/library/site.html)，为这个参数设置默认值。

您的 Python 会话在`sys.path`中可以访问的路径决定了 Python 可以从哪些位置导入模块。

如果激活虚拟环境并输入 Python 解释器，则可以确认基本 Python 安装的标准库文件夹的路径可用:

*   [*视窗*](#windows-21)
**   [*Linux*](#linux-21)**   [*macOS*](#macos-21)**

***>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\python310.zip',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\DLLs',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\lib', 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310',
 'C:\\Users\\Name\\path\\to\\venv',
 'C:\\Users\\Name\\path\\to\\venv\\lib\\site-packages']
```

>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 '/usr/local/lib/python310.zip',
 '/usr/local/lib/python3.10', '/usr/local/lib/python3.10/lib-dynload',
 '/home/name/path/to/venv/lib/python3.10/site-packages']
```

>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python310.zip',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10', '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/lib-dynload',
 '/Users/name/path/to/venv/lib/python3.10/site-packages']
```

因为包含标准库模块的目录路径在`sys.path`中可用，所以当您在虚拟环境中使用 Python 时，您将能够导入它们中的任何一个。

[*Remove ads*](/account/join/)

### 它修改你的`PYTHONPATH`

为了确保您想要运行的脚本在您的虚拟环境中使用 Python 解释器，`venv`修改了您可以使用 [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path) 访问的 [`PYTHONPATH`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH) 环境变量。

如果在没有活动虚拟环境的情况下检查变量*，您将看到默认 Python 安装的默认路径位置:*

*   [*视窗*](#windows-22)
**   [*Linux*](#linux-22)**   [*macOS*](#macos-22)**

***>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\python310.zip',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\DLLs',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\lib',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310',
 'C:\\Users\\Name\\AppData\\Roaming\\Python\\Python310\\site-packages', 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages']
```

>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 '/usr/local/lib/python310.zip',
 '/usr/local/lib/python3.10',
 '/usr/local/lib/python3.10/lib-dynload',
 '/usr/local/lib/python3.10/site-packages']
```

>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python310.zip',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/lib-dynload',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages']
```

请注意突出显示的行，它们表示站点包目录的路径。该文件夹包含您要安装的外部模块，例如，使用`pip`安装的模块。如果没有激活的虚拟环境，该目录将嵌套在与 Python 可执行文件相同的文件夹结构中。

**注意:**Windows 上的`Roaming`文件夹包含一个额外的 site-packages 目录，该目录与使用带有`pip`的`--user`标志的安装相关。这个文件夹提供了小程度的虚拟化，但是它仍然在一个地方收集了所有的`--user`安装包。

但是，如果您在启动另一个解释器会话之前激活虚拟环境，并重新运行相同的命令，那么您将得到不同的输出:

*   [*视窗*](#windows-23)
**   [*Linux*](#linux-23)**   [*macOS*](#macos-23)**

***>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\python310.zip',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\DLLs',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\lib',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310',
 'C:\\Users\\Name\\path\\to\\venv', 'C:\\Users\\Name\\path\\to\\venv\\lib\\site-packages']
```

>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 '/usr/local/lib/python310.zip',
 '/usr/local/lib/python3.10',
 '/usr/local/lib/python3.10/lib-dynload',
 '/home/name/path/to/venv/lib/python3.10/site-packages']
```

>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python310.zip',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/lib-dynload',
 '/Users/name/path/to/venv/lib/python3.10/site-packages']
```

Python 用虚拟环境中的路径替换了默认的站点包目录路径。这一变化意味着 Python 将加载虚拟环境中安装的任何外部包。相反，因为到您的基本 Python 的 site-packages 目录的路径不再在这个列表中，Python 不会从那里加载模块。

**注意:**在 Windows 系统上，Python 额外添加了你的虚拟环境的根文件夹路径到`sys.path`。

Python 路径设置的这一变化有效地在您的虚拟环境中创建了外部包的隔离。

可选地，通过在创建虚拟环境时传递一个参数，您可以获得对基本 Python 安装的系统站点包目录的只读[访问权。](#include-the-system-site-packages)

### 它在激活时改变你的 Shell `PATH`变量

为了方便起见，您通常会在工作之前激活您的虚拟环境，尽管您不必这样做。

要激活您的虚拟环境，您需要执行激活脚本:

*   [*视窗*](#windows-24)
**   [**Linux + macOS**](#linux-macos-24)*

```py
PS> venv\Scripts\activate
(venv) PS>
```

```py
$ source venv/bin/activate
(venv) $
```

您必须运行哪个激活脚本取决于您的操作系统和您使用的 shell。

如果您深入了解虚拟环境的文件夹结构，您会发现它附带了一些不同的激活脚本:

*   [*视窗*](#windows-25)
**   [*Linux*](#linux-25)**   [*macOS*](#macos-25)**

```py
venv\
│
├── Include\
│
├── Lib\
│
├── Scripts\
│   ├── Activate.ps1 │   ├── activate │   ├── activate.bat │   ├── deactivate.bat
│   ├── pip.exe
│   ├── pip3.10.exe
│   ├── pip3.exe
│   ├── python.exe
│   └── pythonw.exe
│
└── pyvenv.cfg
```

```py
venv/
│
├── bin/
│   ├── Activate.ps1 │   ├── activate │   ├── activate.csh │   ├── activate.fish │   ├── pip
│   ├── pip3
│   ├── pip3.10
│   ├── python
│   ├── python3
│   └── python3.10
│
├── include/
│
├── lib/
│
├── lib64/
│
└── pyvenv.cfg
```

```py
venv/
│
├── bin/
│   ├── Activate.ps1 │   ├── activate │   ├── activate.csh │   ├── activate.fish │   ├── pip
│   ├── pip3
│   ├── pip3.10
│   ├── python
│   ├── python3
│   └── python3.10
│
├── include/
│
├── lib/
│
└── pyvenv.cfg
```

这些激活脚本都有相同的目的。然而，由于用户使用的操作系统和外壳各不相同，他们需要提供不同的实现方式。

**注意:**您可以在[您最喜欢的代码编辑器](https://realpython.com/python-ides-code-editors-guide/)中打开任何高亮显示的文件，以检查虚拟环境激活脚本的内容。请随意深入研究该文件，以便更深入地了解它的作用，或者继续阅读，以便快速了解它的要点。

激活脚本中会发生两个关键操作:

1.  **路径:**它将`VIRTUAL_ENV`变量设置为虚拟环境的根文件夹路径，并将 Python 可执行文件的相对位置添加到`PATH`中。
2.  **命令提示符:**它将命令提示符更改为您在创建虚拟环境时传递的名称。它将这个名字放在括号中，例如`(venv)`。

这些变化在您的 shell 中实现了虚拟环境的便利性:

1.  **路径:**因为虚拟环境中所有可执行文件的路径现在都在`PATH`的前面，所以当你输入`pip`或`python`时，你的 shell 将调用`pip`或 Python 的内部版本。
2.  **命令提示符:**因为脚本改变了你的命令提示符，你会很快知道你的虚拟环境是否被激活。

这两个变化都是小的改动，纯粹是为了方便您而存在的。它们不是绝对必要的，但是它们让使用 Python 虚拟环境变得更加愉快。

您可以在虚拟环境激活前后检查您的`PATH`变量。如果您已经激活了您的虚拟环境，那么您将在`PATH`的开头看到包含您的内部可执行文件的文件夹的路径:

*   [*视窗*](#windows-26)
**   [*Linux*](#linux-26)**   [*macOS*](#macos-26)**

```py
PS> $Env:Path
C:\Users\Name\path\to\venv\Scripts;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Users\Name\AppData\Local\Programs\Python\Python310\Scripts\;C:\Users\Name\AppData\Local\Programs\Python\Python310\;c:\users\name\.local\bin;c:\users\name\appdata\roaming\python\python310\scripts
```

```py
$ echo $PATH
/home/name/path/to/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/name/.local/bin
```

```py
$ echo $PATH
/Users/name/path/to/venv/bin:/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/name/.local/bin
```

请记住，打印您的`PATH`变量的输出很可能看起来非常不同。重要的一点是，激活脚本已经在`PATH`变量的开头添加了虚拟环境的路径。

当您使用`deactivate`停用虚拟环境时，您的 shell 会撤销这些更改，并将`PATH`和您的命令提示符恢复到原来的状态。

**注意:**在 Windows 上，`deactivate`命令执行一个名为`deactivate.bat`的单独文件。在 UNIX 系统上，用于激活虚拟环境的同一脚本也提供了用于停用虚拟环境的代码逻辑。

试一试，检查一下变化。对您的`PATH`变量的这一小小的改变为您提供了在虚拟环境中运行可执行文件的便利，而无需提供完整的路径。

[*Remove ads*](/account/join/)

### 它从任何有绝对路径的地方运行

你不需要*激活你的虚拟环境来使用它。您可以在不激活虚拟环境的情况下使用它，即使激活它是您经常看到的推荐的常见操作。*

如果你只给你的 shell 提供一个可执行文件的名字，它会在记录在`PATH`中的位置寻找一个带有这个名字的可执行文件。然后，它会挑选并运行第一个符合该标准的。

激活脚本[改变你的`PATH`变量](#it-changes-your-shell-path-variable-on-activation)，这样你的虚拟环境的二进制文件夹就是你的 shell 寻找可执行文件的第一个地方。这个改变允许你只输入`pip`或`python`来运行位于你的虚拟环境中的相应程序。

如果您*没有*激活您的虚拟环境，那么您可以转而在您的虚拟环境中传递 Python 可执行文件的**绝对路径**，以便从您的虚拟环境中运行任何脚本:

*   [*视窗*](#windows-27)
**   [*Linux*](#linux-27)**   [*macOS*](#macos-27)**

```py
PS> C:\Users\Name\path\to\venv\Scripts\python.exe
```

```py
$ /home/name/path/to/venv/bin/python
```

```py
$ /Users/name/path/to/venv/bin/python
```

这个命令将在您的虚拟环境中启动 Python 解释器，就像您首先激活虚拟环境，然后用`python`调用它一样。



如何确认在不激活虚拟环境的情况下使用绝对路径会启动与激活虚拟环境并运行`python`时相同的解释器？

记下一些可能的检查方法，然后尝试下面的*解决方案*模块中提到的一些解决方案。



如本教程前面部分所述，您可以:

*   [打印`sys.path`](#it-modifies-your-pythonpath) 并确认您的虚拟环境中的站点包目录已列出
*   确认 [`sys.prefix`已经改变](#it-adapts-the-prefix-finding-process)，现在指向虚拟环境文件夹结构中的一个文件夹
*   激活虚拟环境，然后检查 [`PATH`外壳变量](#it-changes-your-shell-path-variable-on-activation)，找到您的虚拟环境的二进制可执行文件的路径

如果您不确定为什么这些方法中的任何一种能够确认这种工作方式，请点击本教程中相关部分的链接来刷新您的记忆。

或者，您可以通过启动解释器并运行`import sys; sys.executable`来确认您正在使用哪个 Python 可执行文件。这些命令将返回当前 Python 解释器的绝对路径。该路径是否通向您的虚拟环境文件夹结构？

您通常会在使用虚拟环境之前激活它，并在使用完毕后停用它。然而，在日常使用中，使用绝对路径是一种很有帮助的方法。

**注意:**绝对路径有助于在远程服务器上运行预定的脚本，或者在 Docker 容器中运行[。具体来说，如果脚本需要外部依赖项，并且希望在 Python 虚拟环境中将其与服务器的其余部分隔离开来，那么您将希望使用绝对路径。](https://pythonspeed.com/articles/activate-virtualenv-dockerfile/)

将虚拟环境的激活嵌入到脚本中是一项繁琐的工作，出错的几率比不出错的几率要高。相反，利用您在本教程中获得的知识，您可以在运行脚本时在虚拟环境中使用 Python 解释器的绝对路径。

例如，如果您在远程 Linux 服务器上设置了一个每小时一次的 CRON 作业，它使用您在虚拟环境中安装的外部`aiohttp`包异步检查[站点连接](https://realpython.com/site-connectivity-checker-python/),那么您可以使用这个方法:

```py
0 * * * *
 /home/name/Documents/connectivity-checker/venv/bin/python
 -m rpchecker
 -u google.com twitter.com
 -a
```

您不需要激活您的虚拟环境来使用正确的 Python 解释器，它可以访问您在虚拟环境中安装的依赖项。相反，您只需将绝对路径传递给解释器的二进制文件。Python 会在初始化期间为您处理剩下的事情。

只要提供 Python 可执行文件的路径，就不需要激活虚拟环境来享受使用虚拟环境的好处。

## 如何定制虚拟环境？

如果您对什么是 Python 虚拟环境很有信心，并且想要为特定的用例定制它，那么您就在正确的地方。在本节中，您将了解在使用`venv`创建虚拟环境时可以传递的可选参数，以及这些定制如何帮助您精确地获得您需要的虚拟环境。

### 改变命令提示符

创建虚拟环境时，您可以通过传递除 *venv* 之外的名称来更改包含虚拟环境的文件夹名称。事实上，你经常会在不同的项目中看到不同的名字。其中一些是常用的:

*   `venv`
*   `env`
*   `.venv`

您可以为您为虚拟环境创建的文件夹命名任何您想要的名称。

**注意:**命名你的虚拟环境文件夹`venv`只是一个惯例。坚持这个约定可以帮助你使用一个 [`.gitignore`](https://realpython.com/python-git-github-intro/#gitignore) 文件可靠地将你的虚拟环境从版本控制中排除。

激活虚拟环境后，您选择的任何名称都会显示在您的命令提示符中:

*   [*视窗*](#windows-28)
**   [**Linux + macOS**](#linux-macos-28)*

```py
PS> python -m venv your-fancy-name
PS> your-fancy-name\Scripts\activate
(your-fancy-name) PS>
```

```py
$ python3 -m venv your-fancy-name
$ source your-fancy-name/bin/activate
(your-fancy-name) $
```

如果您为虚拟环境文件夹提供了一个备用名称，那么在运行激活脚本时也需要考虑这个名称，如上面的代码示例所示。

如果您希望方便地看到不同的命令提示符，但是您希望保持文件夹名称的描述性，以便您知道它包含一个虚拟环境，那么您可以将您想要的命令提示符名称传递给`--prompt`:

*   [*视窗*](#windows-29)
**   [**Linux + macOS**](#linux-macos-29)*

```py
PS> python -m venv venv --prompt="dev-env"
PS> venv\Scripts\activate
(dev-env) PS>
```

```py
$ python3 -m venv venv --prompt="dev-env"
$ source venv/bin/activate
(dev-env) $
```

使用可选的`--prompt`参数，您可以将虚拟环境活动时显示的命令提示符设置为描述性字符串，而无需更改虚拟环境文件夹的名称。

在上面的代码片段中，您可以看到您仍然在调用文件夹`venv`，这意味着您将能够使用熟悉的路径访问激活脚本。同时，激活后显示的命令提示符会是你传给`--prompt`的东西。

[*Remove ads*](/account/join/)

### 覆盖现有环境

您可能希望随时删除并重新创建一个虚拟环境。如果您经常这样做，那么您可能会很高兴知道，在 Python 创建新环境之前，您可以添加`--clear`参数来删除现有环境的内容。

在您尝试之前，了解一下运行命令创建一个新的虚拟环境*而不使用*会有所帮助，该参数不会覆盖同名的现有虚拟环境:

*   [*视窗*](#windows-30)
**   [**Linux + macOS**](#linux-macos-30)*

```py
PS> python -m venv venv
PS> venv\Scripts\pip.exe install requests
PS> venv\Scripts\pip.exe list
Package            Version
------------------ ---------
certifi            2021.10.8
charset-normalizer 2.0.12
idna               3.3
pip                22.0.4
requests           2.27.1
setuptools         58.1.0
urllib3            1.26.9

PS> python -m venv venv PS> venv\Scripts\pip.exe list
Package            Version
------------------ ---------
certifi            2021.10.8
charset-normalizer 2.0.12
idna               3.3
pip                22.0.4
requests           2.27.1
setuptools         58.1.0
urllib3            1.26.9
```

```py
$ python3 -m venv venv
$ venv/bin/pip install requests
$ venv/bin/pip list
Package            Version
------------------ ---------
certifi            2021.10.8
charset-normalizer 2.0.12
idna               3.3
pip                22.0.4
requests           2.27.1
setuptools         58.1.0
urllib3            1.26.9

$ python3 -m venv venv $ venv/bin/pip list
Package            Version
------------------ ---------
certifi            2021.10.8
charset-normalizer 2.0.12
idna               3.3
pip                22.0.4
requests           2.27.1
setuptools         58.1.0
urllib3            1.26.9
```

在这个代码示例中，您首先创建了一个名为 *venv* 的虚拟环境，然后使用 environment-internal `pip`可执行文件将`requests`安装到虚拟环境的 site-packages 目录中。然后使用`pip list`来确认它已经安装，以及它的依赖项。

**注意:**您在没有激活虚拟环境的情况下运行了所有这些命令。相反，您[使用内部`pip`可执行文件的完整路径](#it-runs-from-anywhere-with-absolute-paths)来安装到您的虚拟环境中。或者，你可以[激活虚拟环境](#activate-it)。

在突出显示的行中，您试图使用相同的名称 *venv* 创建另一个虚拟环境*。*

您可能期望`venv`通知您在相同的路径上有一个现有的虚拟环境，但是它没有。您可能希望`venv`自动删除同名的现有虚拟环境，并用一个新的替换它，但是它也没有这样做。相反，当`venv`在你提供的路径上找到一个同名的现有虚拟环境时，它不会做任何事情——同样，它也不会向你传达这一点。

如果您在第二次运行虚拟环境创建命令后列出已安装的包，那么您会注意到`requests`及其依赖项仍然会出现。这可能不是你想要实现的。

您可以使用`--clear`显式地*覆盖*一个现有的虚拟环境，而不是导航到您的虚拟环境文件夹并首先删除它:

*   [*视窗*](#windows-31)
**   [**Linux + macOS**](#linux-macos-31)*

```py
PS> python -m venv venv
PS> venv\Scripts\pip.exe install requests
PS> venv\Scripts\pip.exe list
Package            Version
------------------ ---------
certifi            2021.10.8
charset-normalizer 2.0.12
idna               3.3
pip                22.0.4
requests           2.27.1
setuptools         58.1.0
urllib3            1.26.9

PS> python -m venv venv --clear PS> venv\Scripts\pip.exe list
Package    Version ---------- ------- pip        22.0.4 setuptools 58.1.0
```

```py
$ python3 -m venv venv
$ venv/bin/pip install requests
$ venv/bin/pip list
Package            Version
------------------ ---------
certifi            2021.10.8
charset-normalizer 2.0.12
idna               3.3
pip                22.0.4
requests           2.27.1
setuptools         58.1.0
urllib3            1.26.9

$ python3 -m venv venv --clear $ venv/bin/pip list
Package    Version ---------- ------- pip        22.0.4 setuptools 58.1.0
```

使用与前面相同的示例，您在第二次运行 creation 命令时添加了可选的`--clear`参数。

然后，您确认 Python 自动丢弃了同名的现有虚拟环境，并创建了一个新的默认虚拟环境，而没有先前安装的包。

### 一次创建多个虚拟环境

如果一个虚拟环境不够，您可以通过向命令传递多个路径来一次创建多个独立的虚拟环境:

*   [*视窗*](#windows-32)
**   [*Linux*](#linux-32)**   [*macOS*](#macos-32)**

```py
PS> python -m venv venv C:\Users\Name\Documents\virtualenvs\venv-copy
```

```py
$ python3 -m venv venv /home/name/virtualenvs/venv-copy
```

```py
$ python3 -m venv venv /Users/name/virtualenvs/venv-copy
```

运行此命令会在两个不同的位置创建两个独立的虚拟环境。这两个文件夹是独立的虚拟环境文件夹。因此，传递多个路径可以节省您多次键入创建命令的精力。

在上面的例子中，您可能会注意到第一个参数`venv`代表一个相对路径。相反，第二个参数使用绝对路径指向新的文件夹位置。在创建虚拟环境时，这两种方法都有效。你甚至可以混合搭配，就像你在这里做的那样。

**注意:**创建虚拟环境最常用的命令是`python3 -m venv venv`，它使用 shell 中当前位置的相对路径，并在该目录中创建新的文件夹`venv`。

你没必要这么做。相反，您可以提供一个指向系统任何地方的绝对路径。如果您的任何路径目录尚不存在，`venv`将为您创建它们。

您也不局限于同时创建两个虚拟环境。您可以传递任意数量的有效路径，用空白字符分隔。Python 将努力在每个位置建立一个虚拟环境，甚至在途中创建任何丢失的文件夹。

[*Remove ads*](/account/join/)

### 更新核心依赖关系

当您使用`venv`及其默认设置创建了一个 Python 虚拟环境，然后使用`pip`安装了一个外部包时，您很可能会遇到一条消息，告诉您您安装的`pip`已经过期:

*   [*视窗*](#windows-33)
**   [**Linux + macOS**](#linux-macos-33)*

```py
WARNING: You are using pip version 21.2.4; however, version 22.0.4 is available.
You should consider upgrading via the
'C:\Users\Name\path\to\venv\Scripts\python.exe -m pip install --upgrade pip' command.
```

```py
WARNING: You are using pip version 21.2.4; however, version 22.0.4 is available.
You should consider upgrading via the
'/path/to/venv/python -m pip install --upgrade pip' command.
```

创造新的东西却发现它已经过时了，这可能会令人沮丧。为什么会这样？

在创建默认配置为`venv`的虚拟环境时，您将收到的`pip`安装可能已经过时，因为`venv`使用 [`ensurepip`](https://docs.python.org/3/library/ensurepip.html) 将`pip`引导到您的虚拟环境中。

`ensurepip`有意地[不连接互联网](https://www.python.org/dev/peps/pep-0453/#explicit-bootstrapping-mechanism)，而是使用每一个新的 CPython 版本附带的`pip`轮子。因此，捆绑的`pip`比独立的`pip`项目有不同的更新周期。

一旦你使用`pip`安装了一个外部包，这个程序就会连接到 PyPI，并且还会识别`pip`本身是否过时。如果`pip`过时了，那么你会收到如上所示的警告。

虽然使用引导版本的`pip`在某些情况下会有所帮助，但是您可能希望使用最新的`pip`来避免旧版本中仍然存在的潜在安全问题或错误。对于现有的虚拟环境，您可以按照`pip`打印到您的终端上的指令，使用`pip`进行自我升级。

如果您想节省手动操作的工作量，您可以通过传递参数`--upgrade-deps`来指定您想要`pip`联系 PyPI 并在安装后立即更新它自己:

*   [*视窗*](#windows-34)
**   [**Linux + macOS**](#linux-macos-34)*

```py
PS> python -m venv venv --upgrade-deps
PS> venv\Scripts\activate
(venv) PS> python -m pip install --upgrade pip
Requirement already satisfied: pip in c:\users\name\path\to\venv\lib\site-packages (22.0.4)
```

```py
$ python3 -m venv venv --upgrade-deps
$ source venv/bin/activate
(venv) $ python -m pip install --upgrade pip
Requirement already satisfied: pip in ./venv/lib/python3.10/site-packages (22.0.4)
```

假设您在创建虚拟环境时使用可选的`--upgrade-deps`参数。在这种情况下，它会自动向 PyPI 轮询最新版本的`pip`和 setuptools，如果本地轮不是最新的，就安装它们。

讨厌的警告信息消失了，你可以放心使用最新版本的`pip`。

### 避免安装`pip`

您可能想知道为什么要花一些时间来设置 Python 虚拟环境，而它所做的只是创建一个文件夹结构。时间延迟的原因主要是`pip`的安装。`pip`它的依赖性很大，会将虚拟环境的大小从几千字节扩大到几兆字节！

在大多数用例中，您会希望在您的虚拟环境中安装`pip`,因为您可能会使用它来安装来自 PyPI 的外部包。然而，如果你出于某种原因不需要`pip`，那么你可以使用`--without-pip`来创建一个没有它的虚拟环境:

*   [*视窗*](#windows-35)
**   [*Linux*](#linux-35)**   [*macOS*](#macos-35)**

```py
PS> python -m venv venv --without-pip
PS> Get-ChildItem venv | Measure-Object -Property length -Sum

Count    : 1
Average  :
Sum      : 120
Maximum  :
Minimum  :
Property : Length
```

```py
$ python3 -m venv venv --without-pip
$ du -hs venv
52K venv
```

```py
$ python3 -m venv venv --without-pip
$ du -hs venv
28K venv
```

通过使用单独的 Python 可执行文件提供轻量级的隔离，您的虚拟环境仍然可以做任何符合虚拟环境条件的事情。

**注意:**即使你没有安装`pip`，运行`pip install <package-name>`可能仍然*看起来*可以工作。但是不要这样做，因为运行该命令不会得到您想要的结果。您将从系统的其他地方使用一个`pip`可执行文件，并且您的包将位于与该`pip`可执行文件相关联的任何 Python 安装的 site-packages 文件夹中。

要使用没有安装`pip`的虚拟环境，您可以手动将软件包安装到您的站点软件包目录中，或者将您的 ZIP 文件放在那里，然后使用 [Python ZIP imports](https://realpython.com/python-zip-import/) 导入它们。

### 包括系统站点包

在某些情况下，您可能希望保持对基本 Python 的 site-packages 目录的访问，而不是切断这种联系。例如，您可能已经在您的全局 Python 环境中设置了一个在安装期间编译的包，比如 [Bokeh](https://realpython.com/python-data-visualization-bokeh/) 。

Bokeh 恰好是您选择的数据探索库，您在所有项目中都使用它。您仍然希望将客户的项目保存在单独的环境中，但是将散景安装到每个环境中可能需要几分钟的时间。为了快速迭代，您需要访问现有的散景安装，而不需要为您创建的每个虚拟环境重新安装。

在创建虚拟环境时，通过添加`--system-site-packages`标志，可以访问*所有已经安装到基本 Python 的站点包目录中的*模块。

**注意:**如果您安装任何额外的外部包，那么 Python 会将它们放入您的虚拟环境的 site-packages 目录中。您只能读取系统站点包目录。

传递此参数时创建一个新的虚拟环境。您将会看到，除了您的本地站点包目录之外，到您的基本 Python 站点包目录的路径将会停留在`sys.path`中。

为了测试这一点，您可以使用`--system-site-packages`参数创建并激活一个新的虚拟环境:

*   [*视窗*](#windows-36)
**   [**Linux + macOS**](#linux-macos-36)*

```py
PS> python -m venv venv --system-site-packages
PS> venv\Scripts\activate
(venv) PS>
```

```py
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
(venv) $
```

您又一次创建了一个名为`venv`的新虚拟环境，但是这次您传递了`--system-site-packages`参数。添加这个可选参数会导致您的`pyvenv.cfg`文件中出现不同的设置:

*   [*视窗*](#windows-37)
**   [*Linux*](#linux-37)**   [*macOS*](#macos-37)**

```py
home = C:\Users\Name\AppData\Local\Programs\Python\Python310
include-system-site-packages = true version = 3.10.3
```

```py
home = /usr/local/bin
include-system-site-packages = true version = 3.10.3
```

```py
home = /Library/Frameworks/Python.framework/Versions/3.10/bin
include-system-site-packages = true version = 3.10.3
```

现在，`include-system-site-packages`配置被设置为`true`，而不是显示默认值`false`。

这一变化意味着您将看到一个额外的`sys.path`条目，它允许您的虚拟环境中的 Python 解释器也能访问系统站点包目录。确保您的虚拟环境是活动的，然后启动 Python 解释器来检查路径变量:

*   [*视窗*](#windows-38)
**   [*Linux*](#linux-38)**   [*macOS*](#macos-38)**

***>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\python310.zip',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\DLLs',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\lib',
 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310',
 'C:\\Users\\Name\\path\\to\\venv',
 'C:\\Users\\Name\\path\\to\\venv\\lib\\site-packages',
 'C:\\Users\\Name\\AppData\\Roaming\\Python\\Python310\\site-packages', 'C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages']
```

>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 '/usr/local/lib/python310.zip',
 '/usr/local/lib/python3.10',
 '/usr/local/lib/python3.10/lib-dynload',
 '/home/name/path/to/venv/lib/python3.10/site-packages',
 '/home/name/.local/lib/python3.10/site-packages', '/usr/local/lib/python3.10/site-packages']
```

>>>

```py
>>> import sys
>>> from pprint import pp
>>> pp(sys.path)
['',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python310.zip',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/lib-dynload',
 '/Users/name/path/to/venv/lib/python3.10/site-packages',
 '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages']
```

突出显示的线条显示了当您使用`--system-site-packages`创建虚拟环境时，虚拟环境中存在的附加路径。它们指向 Python 基础安装的 site-packages 目录，并让虚拟环境中的解释器访问这些包。

### 复制或链接您的可执行文件

您收到的是 Python 二进制文件的副本还是符号链接取决于您使用的操作系统:

*   Windows 可以创建符号链接或副本，但有些版本不支持符号链接。创建符号链接可能需要您拥有管理员权限。
*   Linux 发行版可能会创建一个符号链接或者一个副本，并且通常选择符号链接而不是副本。
*   macOS 总是会创建二进制文件的副本。

PEP 405 提到了创建符号链接的优点:

> 如果可能，最好使用符号链接，因为在升级到底层 Python 安装的情况下，在 venv 中复制的 Python 可执行文件可能会与安装的标准库不同步，需要手动升级。([来源](https://www.python.org/dev/peps/pep-0405/#copies-versus-symlinks))

虽然对可执行文件进行符号链接会有所帮助，这样即使您升级了基本 Python 安装，它们也会自动保持同步，但是这种方法增加的脆弱性可能会超过它的好处。比如你在 Windows 中双击`python.exe`，操作系统会急切的解析符号链接，忽略你的虚拟环境。

最有可能的是，您永远都不需要接触这些参数，但是如果您有一个很好的理由来尝试强制使用操作系统的默认符号链接或副本，那么您可以这样做:

*   **`--symlinks`** 将尝试创建符号链接而不是副本。此选项对 macOS 构件没有任何影响。
*   **`--copies`** 将尝试创建您的 Python 二进制文件的副本，而不是将它们链接到基本 Python 安装的可执行文件。

创建虚拟环境时，您可以传递这些可选参数中的任何一个。

### 升级你的 Python 以匹配系统 Python

如果您已经使用副本而不是符号链接构建了您的虚拟环境[，并且后来在您的操作系统上更新了您的基本 Python 版本，您可能会遇到与标准库模块不匹配的版本。](#copy-or-link-your-executables)

`venv`模块提供了一个解决方案。可选的`--upgrade`参数保持站点包目录不变，同时将二进制文件更新到系统上的新版本:

*   [*视窗*](#windows-39)
**   [**Linux + macOS**](#linux-macos-39)*

```py
PS> python -m venv venv --upgrade
```

```py
$ python3 -m venv venv --upgrade
```

如果您运行该命令，并且在最初创建虚拟环境后更新了 Python 版本，那么您将保留已安装的库，但是`venv`将更新`pip`和 Python 的可执行文件。

在本节中，您已经了解到可以对使用`venv`模块构建的虚拟环境进行大量定制。这些调整可以是纯粹的便利性更新，例如将命令提示符命名为与环境文件夹不同的名称，覆盖现有的环境，或者用一个命令创建多个环境。其他定制在您的虚拟环境中创建不同的功能，例如，跳过`pip`及其依赖项的安装，或者链接回基本 Python 的 site-packages 文件夹。

但是如果你想做更多的事情呢？在下一节中，您将探索内置`venv`模块的替代方案。

## 除了`venv`之外，还有哪些流行的选择？

`venv`模块是一种处理 Python 虚拟环境的好方法。它的一个主要优势是`venv`从 3.3 版本开始预装 Python。但这不是你唯一的选择。您可以使用其他工具在 Python 中创建和处理虚拟环境。

在本节中，您将了解两个流行的工具。它们有不同的作用域，但都通常用于与`venv`模块相同的目的:

1.  **Virtualenv** 是`venv`的超集，并为其实现提供基础。这是一个强大的、可扩展的工具，用于创建隔离的 Python 环境。
2.  Conda 为 Python 和其他语言提供了包、依赖和环境管理。

它们相对于`venv`有一些优势，但是它们不随标准 Python 安装一起提供，所以你必须单独安装它们。

### Virtualenv 项目

[Virtualenv](https://virtualenv.pypa.io/en/latest/) 是一款专门用于创建独立 Python 环境的工具。它一直是 Python 社区中最受欢迎的，并且先于内置的`venv`模块。

这个包是`venv`的超集，它允许你使用`venv`做所有你能做的事情，甚至更多。Virtualenv 允许您:

*   更快地创建虚拟环境
*   [无需提供绝对路径即可发现](https://virtualenv.pypa.io/en/latest/user_guide.html#python-discovery)已安装的 Python 版本
*   使用`pip`升级工具
*   自己扩展工具的功能

当您处理 Python 项目时，这些额外的功能都会派上用场。您甚至可能希望将代码中的 virtualenv 蓝图与项目一起保存，以帮助再现。Virtualenv 有一个丰富的[编程 API](https://virtualenv.pypa.io/en/latest/user_guide.html#programmatic-api) ，允许你描述虚拟环境，而无需创建它们。

在[将`virtualenv`](https://virtualenv.pypa.io/en/latest/installation.html) 安装到您的系统上之后，您可以创建并激活一个新的虚拟环境，类似于您使用`venv`的方式:

*   [*视窗*](#windows-40)
**   [*Linux*](#linux-40)**   [*macOS*](#macos-40)**

```py
PS> virtualenv venv
created virtual environment CPython3.10.3.final.0-64 in 312ms
 creator CPython3Windows(dest=C:\Users\Name\path\to\venv, clear=False, no_vcs_ignore=False, global=False)
 seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=C:\Users\Name\AppData\Local\pypa\virtualenv)
 added seed packages: pip==22.0.4, setuptools==60.10.0, wheel==0.37.1
 activators BashActivator,BatchActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator
PS> Set-ExecutionPolicy Unrestricted -Scope Process
PS> venv\Scripts\activate
(venv) PS>
```

**注意:**为了避免在激活虚拟环境时遇到执行策略问题，您首先使用`Set-ExecutionPolicy Unrestricted -Scope Process`更改了当前 PowerShell 会话的执行策略。

```py
$ virtualenv venv
created virtual environment CPython3.10.3.final.0-64 in 214ms
 creator CPython3Posix(dest=/home/name/path/to/venv, clear=False, no_vcs_ignore=False, global=False)
 seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/home/name/.local/share/virtualenv)
 added seed packages: pip==22.0.4, setuptools==60.10.0, wheel==0.37.1
 activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator
$ source venv/bin/activate
(venv) $
```

```py
$ virtualenv venv
created virtual environment CPython3.10.3.final.0-64 in 389ms
 creator CPython3Posix(dest=/Users/name/path/to/venv, clear=False, no_vcs_ignore=False, global=False)
 seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/Users/name/Library/Application Support/virtualenv)
 added seed packages: pip==22.0.4, setuptools==60.10.0, wheel==0.37.1
 activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator
$ source venv/bin/activate
(venv) $
```

像使用`venv`一样，您可以传递一个相对或绝对路径，并命名您的虚拟环境。在使用 virtualenv 之前，您通常会使用提供的脚本之一激活它。

**注意:**您可能会注意到 virtualenv 创建隔离环境的速度比内置的`venv`模块快得多，这是可能的，因为工具[缓存了特定于平台的应用数据](https://discuss.python.org/t/virtualenv-20-0-0-beta1-is-available/3077)，它可以快速读取这些数据。

与`venv`相比，virtualenv 有两个主要的用户优势:

1.  速度: Virtualenv 创建环境的速度要快得多。
2.  **更新:**由于 virtualenv 的[嵌入式轮子](https://virtualenv.pypa.io/en/latest/user_guide.html#wheels)，你将获得最新的`pip`和设置工具，而不需要在你第一次设置虚拟环境时就连接到互联网。

如果您需要使用 Python 2.x 的遗留版本，那么 virtualenv 也可以提供帮助。它支持使用 Python 2 可执行文件构建 Python 虚拟环境，而使用`venv`是不可能的。

**注意:**如果你想尝试使用 virtualenv，但你没有安装它的权限，你可以使用 [Python 的`zipapp`](https://realpython.com/python-zipapp/) 模块来规避。遵循文档中关于[通过 zipapp](https://virtualenv.pypa.io/en/latest/installation.html#via-zipapp) 安装 virtualenv 的说明。

如果您刚刚开始使用 Python 的虚拟环境，那么您可能希望继续使用内置的`venv`模块。然而，如果你已经使用了一段时间，并且遇到了该工具的局限性，那么[开始使用 virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) 是个好主意。

### Conda 包和环境管理器

[Conda](https://docs.conda.io/en/latest/) 为您提供了另一种包装和环境管理方法。虽然该工具主要与数据科学社区和 [Anaconda Python 发行版](https://anaconda.org)相关联，但它的潜在用例超越了该社区，不仅仅是安装 Python 包:

> 任何语言的包、依赖和环境管理—Python、R、Ruby、Lua、Scala、Java、JavaScript、C/ C++、FORTRAN 等等。([来源](https://docs.conda.io/en/latest/))

虽然您也可以使用 conda 建立一个隔离的环境来安装 Python 包，但这只是该工具的一个特性:

> pip 在*和*环境中安装 *python* 包；康达在*康达*环境中安装*任何*包。([来源](https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/#Myth-#3:-Conda-and-pip-are-direct-competitor))

正如你可能从这句话中了解到的，conda 实现这种隔离的方式不同于`venv`模块和 virtualenv 项目。

**注意:**关于 conda 包和环境管理器的完整讨论超出了本教程的范围。您将忽略其中的差异，并查看专门用于创建和使用 Python 虚拟环境的 conda。

Conda 是自己的项目，与`pip`无关。你可以使用 [Miniconda 安装程序](https://docs.conda.io/en/latest/miniconda.html)在你的系统上设置它，它带来了在你的系统上运行`conda`的最低要求。

在它的默认配置中，conda 从[repo.anaconda.com](https://repo.anaconda.com/)而不是 PyPI 获得它的包。这个备选包索引由 Anaconda 项目维护，类似于 PyPI，但不完全相同。

因为 conda 并不局限于 *Python* 包，你会在 conda 的包索引中发现其他的，通常是数据科学相关的包，它们是用不同的语言编写的。相反，PyPI 上有一些 Python 包是不能用 conda 安装的，因为它们不在那个包库中。如果您的 conda 环境中需要这样一个包，那么您可以使用`pip`将它安装在那里。

如果您在数据科学领域工作，并且使用 Python 和其他数据科学项目，那么 conda 是跨平台和跨语言工作的绝佳选择。

安装 Anaconda 或 Miniconda 之后，您可以[创建一个 conda 环境](https://docs.conda.io/projects/conda/en/latest/commands/create.html):

*   [*视窗*](#windows-41)
**   [*Linux*](#linux-41)**   [*macOS*](#macos-41)**

```py
PS> conda create -n <venv-name>
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

 environment location: C:\Users\Name\miniconda3\envs\<venv-name>

Proceed ([y]/n)? y

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate <venv-name>
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

假设您的标准 PowerShell 会话在成功安装 Anaconda 后没有识别出`conda`命令。在这种情况下，您可以在您的程序中寻找 *Anaconda PowerShell 提示符*,并使用它来代替。

```py
$ conda create -n <venv-name>
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

 environment location: /home/name/anaconda3/envs/<venv-name>

Proceed ([y]/n)? y

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
# $ conda activate <venv-name>
#
# To deactivate an active environment, use
#
# $ conda deactivate
```

```py
$ conda create -n <venv-name>
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

 environment location: /Users/name/opt/anaconda3/envs/<venv-name>

Proceed ([y]/n)? y

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
# $ conda activate <venv-name>
#
# To deactivate an active environment, use
#
# $ conda deactivate
```

该命令在计算机的中心位置创建一个新的 conda 环境。

**注意:**因为所有的 conda 环境都位于同一个位置，所以所有的环境名称都需要是唯一的。因此，最好给它们起一个描述性的名字，而不是调用任何 conda 环境`venv`。

要在新的 conda 环境中工作，您需要激活它:

*   [*视窗*](#windows-42)
**   [**Linux + macOS**](#linux-macos-42)*

```py
PS> conda activate <venv-name>
(<venv-name>) PS>
```

```py
$ conda activate <venv-name>
(<venv-name>) $
```

激活环境后，您可以从 conda 的软件包存储库将软件包安装到该环境中:

*   [*视窗*](#windows-43)
**   [**Linux + macOS**](#linux-macos-43)*

```py
(<venv-name>) PS> conda install numpy
```

```py
(<venv-name>) $ conda install numpy
```

`install`命令将第三方软件包从 conda 的软件包库中安装到您的活动 conda 环境中。

当您在环境中完成工作后，您必须将其停用:

*   [*视窗*](#windows-44)
**   [**Linux + macOS**](#linux-macos-44)*

```py
(<venv-name>) PS> conda deactivate
PS>
```

```py
(<venv-name>) $ conda deactivate
$
```

您可能会注意到，总体思路类似于使用使用`venv`创建的 Python 虚拟环境。命令略有不同，但您将获得在隔离环境中工作的相同好处，您可以在必要时删除并重新创建。

如果您主要从事数据科学项目，并且已经使用过 Anaconda，那么您可能永远都不需要使用`venv`。在这种情况下，你可以阅读更多关于[康达环境](https://realpython.com/python-windows-machine-learning-setup/#understanding-conda-environments)以及如何在你的机器上有效地使用它们。

如果您只有纯 Python 依赖，并且以前没有使用过 Anaconda，那么您最好直接使用更轻量级的`venv`模块，或者尝试一下 virtualenv。

## 您如何管理您的虚拟环境？

如果您已经吸收了前几节中的所有信息，但是您不确定如何处理已经开始聚集在您的系统上的大量环境文件夹，请继续阅读这里。

在本节中，您将学习如何将虚拟环境的基本信息提取到一个文件中，以便您可以随时在任何计算机上快速删除和重新创建虚拟环境文件夹。

您还将了解组织虚拟环境文件夹存放位置的两种不同方法，以及可以帮助您管理虚拟环境的一些流行的第三方工具。

### 决定在哪里创建环境文件夹

Python 虚拟环境只是一个文件夹结构。您可以将它放在系统的任何地方。但是，一致的结构会有所帮助，关于在何处创建虚拟环境文件夹，有两种主要观点:

1.  在每个单独的**项目文件夹内**
2.  在**单个位置**，例如在您主目录的子文件夹中

这两种方法各有优缺点，您的偏好最终取决于您的工作流程。

在**项目-文件夹方法**方法中，您在项目的根文件夹中创建一个新的虚拟环境，该虚拟环境将用于:

```py
project_name/
│
├── venv/ │
└── src/
```

虚拟环境文件夹与您为该项目编写的任何代码并存。

这种结构的优点是，您将知道哪个虚拟环境属于哪个项目，并且一旦导航到项目文件夹，您就可以使用一个短的相对路径来激活您的虚拟环境。

在**单文件夹方法**中，您将*所有*虚拟环境保存在一个文件夹中，例如您主目录的子文件夹中:

*   [*视窗*](#windows-45)
**   [*Linux*](#linux-45)**   [*macOS*](#macos-45)**

```py
C:\USERS\USERNAME\
│
├── .local\
│
├── Contacts\
│
├── Desktop\
│
├── Documents\
│   │
│   └── Projects\
│       │
│       ├── django-project\
│       │
│       ├── flask-project\
│       │
│       └── pandas-project\
│
├── Downloads\
│
├── Favorites\
│
├── Links\
│
├── Music\
│
├── OneDrive\
│
├── Pictures\
│
├── Searches\
│
├── venvs\ │   │ │   ├── django-venv\ │   │ │   ├── flask-venv\ │   │ │   └── pandas-venv\ │
└── Videos\
```

```py
name/
│
├── Desktop/
│
├── Documents/
│   │
│   └── projects/
│       │
│       ├── django-project/
│       │
│       ├── flask-project/
│       │
│       └── pandas-project/
│
├── Downloads/
│
├── Music/
│
├── Pictures/
│
├── Public/
│
├── Templates/
│
├── venvs │   │ │   ├── django-venv/ │   │ │   ├── flask-venv/ │   │ │   └── pandas-venv/ │
└── Videos/
```

```py
name/
│
├── Applications/
│
├── Desktop/
│
├── Documents/
│   │
│   └── projects/
│       │
│       ├── django-project/
│       │
│       ├── flask-project/
│       │
│       └── pandas-project/
│
├── Downloads/
│
├── Library/
│
├── Movies/
│
├── Music/
│
├── Pictures/
│
├── Public/
│
├── opt/
│
└── venvs
 │ ├── django-venv/ │ ├── flask-venv/ │ └── pandas-venv/
```

如果您使用这种方法，跟踪您创建的虚拟环境可能会更容易。您可以在操作系统上的一个位置检查所有虚拟环境，并决定保留和删除哪些虚拟环境。

另一方面，当您已经导航到您的项目文件夹时，您将无法使用相对路径快速激活您的虚拟环境。相反，最好使用相应虚拟环境文件夹中激活脚本的绝对路径来激活它。

**注意:**您可以使用这两种方法中的任何一种，甚至可以混合使用。

你可以在系统的任何地方创建你的虚拟环境。请记住，清晰的结构会让你更容易知道在哪里可以找到文件夹。

第三种选择是将这个决定留给您的[集成开发环境(IDE)](https://realpython.com/python-ides-code-editors-guide/) 。这些程序中的许多都包括在您开始新项目时自动为您创建虚拟环境的选项。

要了解您最喜欢的 IDE 如何处理虚拟环境的更多信息，请查看它的在线文档。例如， [VS Code](https://code.visualstudio.com/docs/python/environments) 和 [PyCharm](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html) 都有自己创建虚拟环境的方法。

### 将它们视为一次性物品

虚拟环境是可任意处理的文件夹结构，您应该能够在任何时候安全地删除和重新创建，而不会丢失有关代码项目的信息。

这意味着您通常不会将任何额外的代码或信息手动放入您的虚拟环境中。任何放入其中的东西都应该由您的包管理器来处理，通常是`pip`或`conda`。

你也不应该将你的虚拟环境提交给[版本控制](https://realpython.com/python-git-github-intro/#version-control)，也不应该将它与你的项目一起发布。

因为虚拟环境不是完全自给自足的 Python 安装，而是依赖于基本 Python 的标准库，所以您不会通过将虚拟环境与代码一起分发来创建可移植的应用程序。

**注意:**如果你想学习如何发布你的 Python 项目，那么你可以阅读关于[发布一个开源包给 PyPI](https://realpython.com/pypi-publish-python-package/) 或者[使用 PyInstaller 发布 Python 应用](https://realpython.com/pyinstaller-python/)的文章。

虚拟环境是轻量级的、一次性的、隔离的环境，可以在其中开发项目。

但是，您应该能够在不同的计算机上重新创建您的 Python 环境，以便您可以在那里运行您的程序或继续开发它。当您将虚拟环境视为可任意处置的，并且不将其提交给版本控制时，您如何实现这一点呢？

### 固定您的依赖关系

为了使您的虚拟环境可再现，您需要一种方法来描述它的内容。最常见的方法是在虚拟环境处于活动状态时创建一个 [`requirements.txt`文件](https://realpython.com/what-is-pip/#using-requirements-files):

*   [*视窗*](#windows-46)
**   [**Linux + macOS**](#linux-macos-46)*

```py
(venv) PS> python -m pip freeze > requirements.txt
```

```py
(venv) $ python -m pip freeze > requirements.txt
```

这个命令将`pip freeze`的输出传输到一个名为`requirements.txt`的新文件中。如果您打开该文件，您会注意到它包含当前安装在您的虚拟环境中的外部依赖项的列表。

这个列表是`pip`知道要安装哪个包的哪个版本的诀窍。只要您保持这个`requirements.txt`文件是最新的，您就可以随时重新创建您正在工作的虚拟环境，甚至在删除了`venv/`文件夹或完全移动到不同的计算机之后:

*   [*视窗*](#windows-47)
**   [**Linux + macOS**](#linux-macos-47)*

```py
(venv) PS> deactivate
PS> python -m venv new-venv
PS> new-venv\Scripts\activate
(new-venv) PS> python -m pip install -r requirements.txt
```

```py
(venv) $ deactivate
$ python3 -m venv new-venv
$ source new-venv/bin/activate
(new-venv) $ python -m pip install -r requirements.txt
```

在上面的示例代码片段中，您创建了一个名为`new-venv`的新虚拟环境，激活了它，并安装了之前记录在`requirements.txt`文件中的所有外部依赖项。

如果您使用`pip list`来检查当前安装的依赖项，那么您将看到两个虚拟环境`venv`和`new-venv`现在包含相同的外部包。

**注意:**通过将您的`requirements.txt`文件提交到版本控制，您可以将您的项目代码与允许您的用户和合作者在他们的机器上重新创建相同的虚拟环境的方法一起发布。

请记住，虽然这是在 Python 中传递代码项目依赖信息的一种普遍方式，但它不是确定性的:

1.  **Python 版本:**这个需求文件不包括创建虚拟环境时使用哪个版本的 Python 作为基础 Python 解释器的信息。
2.  **子依赖关系:**根据您创建需求文件的方式，它可能不包含关于依赖关系的子依赖关系的版本信息。这意味着如果在您创建您的需求文件之后，这个包被悄悄地更新了，那么有人可以得到一个不同版本的子包。

单靠`requirements.txt`无法轻松解决这些问题，但是许多第三方依赖管理工具试图解决它们以保证确定性的构建:

*   [`requirements.txt`利用`pip-tools`](https://pip-tools.readthedocs.io/en/latest/#example-usage-for-pip-compile)
*   [`Pipfile.lock`使用 Pipenv](https://pipenv.pypa.io/en/latest/basics/#example-pipfile-pipfile-lock)
*   [`poetry.lock`用诗](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock)

将虚拟环境工作流集成到其功能中的项目，除此之外，通常还会包括创建锁定文件的方法，以允许您的环境的确定性构建。

### 避免生产中的虚拟环境

您可能想知道在将项目部署到生产环境时，如何包含和激活您的虚拟环境。在大多数情况下，您不希望将虚拟环境文件夹包含在远程在线位置:

*   **GitHub:** 不要把`venv/`文件夹推送给 GitHub。
*   **CI/CD 管道:**不要将您的虚拟环境文件夹包含在您的[持续集成](https://realpython.com/python-continuous-integration/)或持续交付管道中。
*   **服务器部署:**不要在您的部署服务器上设置虚拟环境，除非您自己管理该服务器并在其上运行多个独立的项目。

您仍然需要隔离的环境和代码项目的可再现性。您将通过固定您的依赖项来实现这一点，而不是包括您在本地使用的虚拟环境文件夹。

大多数远程托管提供商，包括 CI/CD 管道工具和平台即服务(PaaS)提供商，如 [Heroku](https://realpython.com/django-hosting-on-heroku/) 或[谷歌应用引擎(GAE)](https://realpython.com/python-web-applications/) ，将自动为您创建这种隔离。

当您将代码项目推送到这些托管服务之一时，该服务通常会将服务器的虚拟部分分配给您的应用程序。这种虚拟化服务器在设计上是隔离的环境，这意味着默认情况下您的代码将在其独立的环境中运行。

在大多数托管解决方案中，您不需要处理创建隔离，但是您仍然需要提供关于在远程环境中安装什么的信息。为此，您将经常在您的`requirements.txt`文件中使用固定的依赖关系。

**注意:**如果您在自己托管的服务器上运行多个项目，那么您可能会从在该服务器上设置虚拟环境中受益。

在这种情况下，您可以像对待本地计算机一样对待服务器。即使这样，您也不会复制虚拟环境文件夹。相反，您将从您的固定依赖项在您的远程服务器上重新创建虚拟环境。

大多数托管平台提供商还会要求您创建特定于您正在使用的工具的设置文件。这个文件将包含没有记录在`requirements.txt`中的信息，但是平台需要为你的代码建立一个运行环境。你需要仔细阅读你打算使用的托管服务文档中的这些特定文件。

一个流行的选项是 [Docker](https://realpython.com/python-versions-docker/) ，它将虚拟化提升到一个新的水平，并且仍然允许您自己创建许多设置。

### 使用第三方工具

Python 社区创建了许多附加工具，这些工具将虚拟环境作为其功能之一，并允许您以用户友好的方式管理多个虚拟环境。

因为许多工具出现在在线讨论和教程中，所以您可能想知道每个工具是关于什么的，以及它们如何帮助您管理虚拟环境。

虽然讨论每个项目超出了本教程的范围，但是您将大致了解存在哪些流行的项目，它们做什么，以及在哪里可以了解更多信息:

*   **[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)**是 virtualenv 项目的扩展，它使得创建、删除和管理虚拟环境变得更加简单。它将您的所有虚拟环境保存在一个地方，引入了用户友好的 CLI 命令来管理和切换虚拟环境，并且是可配置和可扩展的。 [`virtualenvwrapper-win`](https://github.com/davidmarble/virtualenvwrapper-win/) 是这个项目的一个 Windows 端口。

*   **[诗歌](https://python-poetry.org)** 是 [Python 依赖管理](https://realpython.com/dependency-management-python-poetry/)和打包的工具。有了诗歌，你可以声明你的项目所依赖的包，类似于`requirements.txt`但是具有确定性。然后，poems 会在一个自动生成的虚拟环境中安装这些依赖项，并帮助您[管理虚拟环境](https://python-poetry.org/docs/managing-environments/)。

*   **[Pipenv](https://pipenv.pypa.io/en/latest/)** 旨在改进 Python 中的打包。它在后台使用`virtualenv`为你的项目创建和管理虚拟环境。像诗歌一样， [Pipenv 旨在改进依赖管理](https://realpython.com/pipenv-guide/#dependency-management-with-requirementstxt)以允许确定性构建。这是一个相对较慢的高级工具，已经得到了 [Python 打包权威(PyPA)](https://www.pypa.io/en/latest/) 的支持。

*   **[pipx](https://github.com/pypa/pipx)** 允许你安装 Python 包，你习惯在隔离环境中作为独立的*应用*运行。它为每个工具创建了一个虚拟环境，并使其可以在全球范围内访问。除了帮助使用代码质量工具，如 black、isort、flake8、pylint 和 mypy，它对于安装替代的 Python 解释器也很有用，如 bpython、ptpython 或 ipython。

*   **[pipx-in-pipx](https://github.com/mattsb42-meta/pipx-in-pipx)** 是一个您可以用来安装 pipx 的包装器，它通过允许您使用 pipx 本身来安装和管理 pipx，将`pip` 的[递归缩写提升到了一个新的水平。](https://en.wikipedia.org/wiki/Pip_(package_manager)#History)

*   **[pyenv](https://github.com/pyenv/pyenv)** 与虚拟环境并没有内在联系，尽管它经常被提到与这个概念有关。您可以使用 pyenv 管理多个 Python 版本，这允许您在新版本和旧版本之间切换，这是您正在进行的项目所需要的。pyenv 还有一个名为 [pyenv-win](https://github.com/pyenv-win/pyenv-win) 的 Windows 端口。

*   **[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)** 是 pyenv 的一个插件，结合了 pyenv 和 virtualenv，允许你在 UNIX 系统上为 pyenv 管理的 Python 版本创建虚拟环境。甚至还有一个混合 pyenv 和 virtualenvwrapper 的插件，叫做 [pyenv-virtualenvwrapper](https://github.com/pyenv/pyenv-virtualenvwrapper) 。

Python 社区构建了一整套第三方项目，可以帮助您以用户友好的方式管理 Python 虚拟环境。

请记住，这些项目旨在使您的过程更加方便，而不是在 Python 中处理虚拟环境所必需的。

## 结论

祝贺您完成了 Python 虚拟环境教程。在整个教程中，您已经对什么是虚拟环境、为什么需要虚拟环境、虚拟环境的内部功能以及如何在您的系统上管理虚拟环境有了全面的了解。

**在本教程中，您学习了如何:**

*   **创建**和**激活**一个 **Python 虚拟环境**
*   解释**为什么**你想**隔离外部依赖**
*   **当你创建一个虚拟环境时，想象 Python 做了什么**
*   **使用**可选参数**到`venv`定制**你的虚拟环境
*   **停用**和**移除**虚拟环境
*   选择**用于管理**您的 Python 版本和虚拟环境的附加工具

下次教程告诉你创建和激活一个虚拟环境，你会更好地理解为什么这是一个好建议，以及 Python 在幕后为你做了什么。

***参加测验:****通过我们的交互式“Python 虚拟环境:入门”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-virtual-environments-a-primer/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。与书面教程一起观看，以加深您的理解: [**使用 Python 虚拟环境**](/courses/working-python-virtual-environments/)**********************************************************************************************************************************************************