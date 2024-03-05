# 一个有效的 Python 环境:宾至如归

> 原文：<https://realpython.com/effective-python-environment/>

当你第一次学习一门新的编程语言时，你会花费大量的时间和精力去理解语法、代码风格和内置工具。对于 Python 和其他任何语言来说都是如此。一旦您对 Python 的来龙去脉有了足够的了解，您就可以开始投入时间来构建一个能够提高您的工作效率的 Python 环境。

您的 shell 不仅仅是按原样提供给您的预构建程序。这是一个框架，你可以在这个框架上建立一个生态系统。这个生态系统将会满足你的需求，这样你就可以花更少的时间去思考你正在做的下一个大项目。

**注意:**如果你正在 Windows 上设置一个全新的 Python 环境，那么你可能想看看这个[综合指南](https://realpython.com/python-coding-setup-windows/)，它将带你完成整个过程。

尽管没有两个开发人员有相同的设置，但是在开发 Python 环境时，每个人都面临许多选择。理解每一个决定和你可用的选择是很重要的！

到本文结束时，你将能够回答如下问题:

*   我应该使用什么外壳？我应该使用什么终端？
*   我可以使用什么版本的 Python？
*   如何管理不同项目的依赖关系？
*   我如何让我的工具为我做一些工作？

一旦你自己回答了这些问题，你就可以开始创建属于你自己的 Python 环境了。我们开始吧！

**免费奖励:** ，向您展示如何使用 Pip、PyPI、Virtualenv 和需求文件等工具避免常见的依赖管理问题。

## 炮弹

当您使用[命令行界面](https://realpython.com/command-line-interfaces-python-argparse/#what-is-a-command-line-interface) (CLI)时，您执行命令并查看它们的输出。一个 **shell** 就是为你提供这个(通常是基于文本的)界面的程序。Shells 通常提供自己的编程语言，您可以用它来操作文件、安装软件等等。

独特的贝壳比这里合理列出的要多，所以你会看到几个突出的。其他的在语法或增强特性上有所不同，但是它们通常提供相同的核心功能。

[*Remove ads*](/account/join/)

### UNIX shell

Unix 是一个操作系统家族，最初是在计算的早期发展起来的。Unix 的流行一直持续到今天，极大地鼓舞了 Linux 和 macOS。第一批 shells 是为 Unix 和类似 Unix 的操作系统开发的。

#### 伯恩·谢尔(`sh` )

Bourne shell——由 Stephen Bourne 于 1979 年为 Bell Labs 开发——是第一个包含环境变量、条件和循环概念的 shell。它为今天使用的许多其他 shells 提供了一个强大的基础，并且在大多数系统上仍然可用。

#### 伯恩-再贝(`bash` )

基于最初的 Bourne shell 的成功，`bash`引入了改进的用户交互功能。使用`bash`，您可以获得 `Tab` 完成、历史以及命令和路径的通配符搜索。`bash`编程语言提供了更多的数据类型，比如数组。

#### z 壳(`zsh` )

将其他 shells 的许多最佳特性以及一些自己的技巧结合到一次体验中。`zsh`提供拼写错误命令的自动纠正，操作多个文件的速记，以及定制命令提示符的高级选项。

`zsh`还提供了深度定制的框架。 [Oh My Zsh](https://ohmyz.sh) 项目提供了丰富的主题和插件，通常与`zsh`一起使用。

[macOS 将从 Catalina](https://support.apple.com/en-us/HT208050) 开始以`zsh`作为默认外壳，这说明了该外壳的受欢迎程度。现在就考虑让自己熟悉`zsh`，这样你会对它的发展感到舒适。

#### Xonsh

如果你特别喜欢冒险，你可以试一试。Xonsh 是一个外壳，它结合了其他类 Unix 外壳的一些特性和 Python 语法的强大功能。您可以使用您已经知道的语言来完成文件系统上的任务等等。

尽管 Xonsh 功能强大，但它缺乏其他 shells 所共有的兼容性。因此，您可能无法在 Xonsh 中运行许多现有的 shell 脚本。如果你发现你喜欢 Xonsh，但是兼容性是个问题，那么你可以在一个更广泛使用的 shell 中使用 Xonsh 作为你活动的补充。

### 视窗外壳

与类 Unix 操作系统类似，Windows 在 shells 方面也提供了许多选项。Windows 中提供的 shells 在特性和语法上各不相同，因此您可能需要尝试几种来找到您最喜欢的一种。

#### CMD ( `cmd.exe` )

CMD(“命令”的缩写)是 Windows 的默认 CLI shell。它是 COMMAND.COM 的继任者，为 DOS(磁盘操作系统)构建的 shell。

因为 DOS 和 Unix 是独立发展的，所以 CMD 中的命令和语法与为类 Unix 系统构建的 shells 明显不同。然而，CMD 仍然为浏览和操作文件、运行命令以及查看输出提供了相同的核心功能。

#### PowerShell

PowerShell 于 2006 年发布，也随 Windows 一起发布。它为大多数命令提供了类似 Unix 的别名，因此，如果您从 macOS 或 Linux 进入 Windows，或者必须使用这两者，那么 PowerShell 可能非常适合您。

PowerShell 比 CMD 强大得多。使用 PowerShell，您可以:

*   将一个命令的输出通过管道传输到另一个命令的输入
*   通过公开的 Windows 管理功能自动执行任务
*   使用脚本语言完成复杂的任务

#### 用于 Linux 的 Windows 子系统

微软发布了一个用于 Linux 的 [Windows 子系统](https://docs.microsoft.com/en-us/windows/wsl/install-win10) (WSL)，可以直接在 Windows 上运行 Linux。如果您安装了 WSL，那么您可以使用`zsh`、`bash`或任何其他类似 Unix 的 shell。如果您想要跨 Windows 和 macOS 或 Linux 环境的强兼容性，那么一定要试试 WSL。你也可以考虑[双启动 Linux 和 Windows](https://opensource.com/article/18/5/dual-boot-linux) 作为替代。

参见这个命令 shell 的[比较，以获得详尽的覆盖范围。](https://en.wikipedia.org/wiki/Comparison_of_command_shells)

[*Remove ads*](/account/join/)

## 终端仿真器

早期的开发者使用终端与中央主机进行交互。这些设备带有键盘、屏幕或打印机，可以显示计算结果。

今天，计算机是便携式的，不需要单独的设备与它们交互，但术语仍然存在。尽管 shell 提供了提示和解释器，用于与基于文本的 CLI 工具进行交互，但是终端**仿真器**(通常简称为**终端**)是您运行来访问 shell 的图形应用程序。

您遇到的几乎所有终端都应该支持相同的基本功能:

*   **文本颜色**用于在代码中突出显示语法，或者在命令输出中区分有意义的文本
*   **滚动**查看之前的命令或其输出
*   **复制/粘贴**用于将文本从其他程序传入或传出外壳
*   **选项卡**用于同时运行多个程序或将您的工作分成不同的会话

### macOS 终端

适用于 macOS 的终端选项都是全功能的，主要区别在于美观和与其他工具的特定集成。

#### 终端

如果你用的是 Mac，那么你之前可能用过内置的[终端](https://support.apple.com/guide/terminal/welcome/mac)应用。终端支持所有常用的功能，您也可以自定义配色方案和一些热键。如果你不需要很多花里胡哨的东西，这是一个足够好的工具。你可以在 macOS 上的*应用→实用程序→终端*中找到终端 app。

#### iTerm2

我是 [iTerm2](https://iterm2.com) 的长期用户。它将开发人员在 Mac 上的体验向前推进了一步，提供了更广泛的定制和生产力选项，使您能够:

*   与 shell 集成以快速跳转到先前输入的命令
*   在命令的输出中创建自定义搜索词高亮显示
*   打开终端中显示的网址和文件用 `Cmd` + `click`

iTerm2 的最新版本附带了一个 Python API，因此您甚至可以通过开发更复杂的定制来提高您的 Python 能力！

iTerm2 足够受欢迎，可以与其他几个工具进行一流的集成，并拥有健康的社区构建插件等等。这是一个很好的选择，因为与终端相比，它的发布周期更频繁，而终端的更新频率只相当于 macOS。

#### 超级

相对来说， [Hyper](https://hyper.is/) 是一个基于 [Electron](https://electronjs.org/) 的终端，这是一个使用网络技术构建桌面应用程序的框架。电子应用程序是高度可定制的，因为它们“只是罩下的 JavaScript”。您可以创建任何您可以为其编写 JavaScript 的功能。

另一方面，JavaScript 是一种高级编程语言，并不总是像 Objective-C 或 Swift 这样的低级语言那样表现良好。当心你安装或创建的插件！

### Windows 终端

与 shell 选项一样，Windows 终端选项在实用程序方面也有很大不同。有些还与特定的外壳紧密结合。

#### 命令提示符

命令提示符是一个图形应用程序，您可以在 Windows 中使用 CMD。像 CMD 一样，它是完成一些小事情的基本工具。尽管 Command Prompt 和 CMD 提供的功能比其他替代产品少，但您可以确信，它们将在几乎每个 Windows 安装中可用，并且位于一个一致的位置。

#### Cygwin

Cygwin 是一个用于 Windows 的第三方工具套件，它提供了一个类似 Unix 的包装器。当我使用 Windows 时，这是我的首选设置，但是您可以考虑为 Linux 采用 Windows 子系统，因为它得到了更多的关注和改进。

#### Windows 终端

微软最近发布了一款名为 [Windows 终端](https://github.com/Microsoft/Terminal)的 Windows 10 开源终端。它允许您在 CMD、PowerShell 甚至 Linux 的 Windows 子系统中工作。如果您需要在 Windows 中做大量的外壳工作，那么 Windows 终端可能是您的最佳选择！Windows 终端仍处于后期测试阶段，因此它尚未随 Windows 一起发布。查看文档以获取有关访问权限的说明。

[*Remove ads*](/account/join/)

## Python 版本管理

选择好终端和 shell 后，您就可以将注意力集中在 Python 环境上了。

你最终会遇到的事情是需要运行 Python 的多个**版本**。您使用的项目可能只在某些版本上运行，或者您可能对创建支持多个 Python 版本的项目感兴趣。您可以配置您的 Python 环境来满足这些需求。

macOS 和大多数 Unix 操作系统都默认安装了 Python 版本。这通常被称为**系统 Python** 。Python 系统工作得很好，但是它通常是过时的。在撰写本文时，macOS High Sierra 仍然附带 Python 2.7.10 作为系统 Python。

**注意**:你几乎肯定希望[至少安装最新版本的 Python](https://realpython.com/installing-python/) ，所以你至少已经有了两个版本的 Python。

将系统 Python 作为默认的很重要，因为系统的许多部分依赖于特定版本的默认 Python。这是定制 Python 环境的众多理由之一！

你如何驾驭它？工装是来帮忙的。

### `pyenv`

[`pyenv`](https://github.com/pyenv/pyenv) 是在 macOS 上安装和管理多个 Python 版本的成熟工具。我推荐[用家酿](https://github.com/pyenv/pyenv#homebrew-on-macos)安装。如果你用的是 Windows，可以用 [`pyenv-win`](https://github.com/pyenv-win/pyenv-win#installation) 。安装了`pyenv`之后，您可以用几个简短的命令将多个版本的 Python 安装到您的 Python 环境中:

```py
$ pyenv versions
* system
$ python --version
Python 2.7.10
$ pyenv install 3.7.3  # This may take some time
$ pyenv versions
* system
 3.7.3
```

您可以管理您希望在当前会话中使用的 Python，可以是全局的，也可以是基于每个项目的。`pyenv`将使`python`命令指向您指定的 Python。请注意，这些都不会覆盖其他应用程序的默认系统 Python，因此您可以安全地使用它们，但是它们在您的 Python 环境中最适合您:

```py
$ pyenv global 3.7.3
$ pyenv versions
 system
* 3.7.3 (set by /Users/dhillard/.pyenv/version)

$ pyenv local 3.7.3
$ pyenv versions
 system
* 3.7.3 (set by /Users/dhillard/myproj/.python-version)

$ pyenv shell 3.7.3
$ pyenv versions
 system
* 3.7.3 (set by PYENV_VERSION environment variable)

$ python --version
Python 3.7.3
```

因为我在工作中使用特定版本的 Python，在个人项目中使用最新版本的 Python，在测试开源项目中使用多个版本，`pyenv`已经被证明是我在自己的 Python 环境中管理所有这些不同版本的一种相当顺畅的方式。关于该工具的详细概述，请参见[使用`pyenv`T4 管理多个 Python 版本。你也可以使用`pyenv`来](https://realpython.com/intro-to-pyenv/)[安装 Python](https://realpython.com/python-pre-release/) 的预发布版本。

### `conda`

如果你在数据科学社区，你可能已经在使用 [Anaconda](https://www.anaconda.com/distribution/) (或者 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) )。Anaconda 是一种数据科学软件的一站式商店，它不仅仅支持 Python。

如果你不需要数据科学[包](https://realpython.com/effective-python-environment/)或者 Anaconda 预打包的所有东西，`pyenv`可能是你更好的轻量级解决方案。不过，管理 Python 版本在每个版本中都非常相似。您可以使用`conda`命令安装类似于`pyenv`的 Python 版本:

```py
$ conda install python=3.7.3
```

你会看到一个详细的清单，列出了所有`conda`将要安装的相关软件，并要求你确认。

没有办法设置“默认”Python 版本，甚至没有好办法查看你已经安装了哪些版本的 Python。相反，它取决于“环境”的概念，您可以在接下来的章节中了解更多。

## 虚拟环境

现在您知道了如何管理多个 Python 版本。通常，你会从事多个项目，这些项目需要*相同的* Python 版本。

因为每个项目都有自己的依赖项集，所以避免混淆它们是一个好的做法。如果所有的依赖项都安装在一个 Python 环境中，那么就很难辨别每个依赖项的来源。在最坏的情况下，两个不同的项目可能依赖于一个包的两个不同版本，但是使用 Python，一次只能安装一个包的一个版本。真是一团糟！

进入**虚拟环境**。你可以把虚拟环境想象成 Python 基础版本的翻版。例如，如果您安装了 Python 3.7.3，那么您可以基于它创建许多虚拟环境。当您在虚拟环境中安装一个包时，您可以将它与您可能拥有的其他 Python 环境隔离开来。每个虚拟环境都有自己的`python`可执行文件副本。

**提示**:大多数虚拟环境工具都提供了一种方法来更新您的 shell 的命令提示符，以显示当前活动的虚拟环境。如果您经常在项目之间切换，请确保这样做，这样您就可以确保在正确的虚拟环境中工作。

[*Remove ads*](/account/join/)

### `venv`

[`venv`](https://docs.python.org/3/library/venv.html) 搭载 Python 版本 3.3+。您可以创建虚拟环境，只需向它传递一个路径，在这个路径上存储环境的`python`、已安装的包等等:

```py
$ python -m venv ~/.virtualenvs/my-env
```

您可以通过获取其`activate`脚本来激活虚拟环境:

```py
$ source ~/.virtualenvs/my-env/bin/activate
```

您可以使用`deactivate`命令退出虚拟环境，该命令在您激活虚拟环境时可用:

```py
(my-env)$ deactivate
```

`venv`是建立在独立的 [`virtualenv`](https://virtualenv.pypa.io/en/stable/) 项目的出色工作和成功之上的。`virtualenv`仍然提供了一些有趣的特性，但是`venv`很好，因为它提供了虚拟环境的效用，而不需要你安装额外的软件。如果您在自己的 Python 环境中主要使用单个 Python 版本，那么您可能已经做得很好了。

如果您已经在管理多个 Python 版本(或计划管理多个版本)，那么集成该工具来简化使用特定 Python 版本创建新虚拟环境的过程是有意义的。`pyenv`和`conda`生态系统都提供了在创建新的虚拟环境时指定 Python 版本的方法，这将在下面的章节中介绍。

### `pyenv-virtualenv`

如果您使用的是`pyenv`，那么 [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv) 用一个管理虚拟环境的子命令增强了`pyenv`:

```py
// Create virtual environment
$ pyenv virtualenv 3.7.3 my-env

// Activate virtual environment
$ pyenv activate my-env

// Exit virtual environment
(my-env)$ pyenv deactivate
```

我每天在大量项目之间切换环境。因此，在我的 Python 环境中，我至少要管理十几个不同的虚拟环境。`pyenv-virtualenv`真正的好处在于，你可以使用`pyenv local`命令配置虚拟环境，并让`pyenv-virtualenv`在你切换到不同目录时自动激活正确的环境:

```py
$ pyenv virtualenv 3.7.3 proj1
$ pyenv virtualenv 3.7.3 proj2
$ cd /Users/dhillard/proj1
$ pyenv local proj1
(proj1)$ cd ../proj2
$ pyenv local proj2
(proj2)$ pyenv versions
 system
 3.7.3
 3.7.3/envs/proj1
 3.7.3/envs/proj2
 proj1
* proj2 (set by /Users/dhillard/proj2/.python-version)
```

`pyenv`和`pyenv-virtualenv`在我的 Python 环境中提供了特别流畅的工作流。

### `conda`

您之前看到过,`conda`将环境而不是 Python 版本作为主要的工作方法。 [`conda`内置管理虚拟环境的支持](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```py
// Create virtual environment
$ conda create --name my-env python=3.7.3

// Activate virtual environment
$ conda activate my-env

// Exit virtual environment
(my-env)$ conda deactivate
```

`conda`将安装指定版本的 Python(如果尚未安装),因此您不必先运行`conda install python=3.7.3`。

### `pipenv`

[`pipenv`](https://docs.pipenv.org/en/latest/) 是一个相对较新的工具，旨在将包管理(稍后将详细介绍)与虚拟环境管理结合起来。它主要是从您那里抽象出虚拟环境管理，只要事情进展顺利，这就很好:

```
$ cd /Users/dhillard/myproj

// Create virtual environment
$ pipenv install
Creating a virtualenv for this project…
Pipfile: /Users/dhillard/myproj/Pipfile
Using /path/to/pipenv/python3.7 (3.7.3) to create virtualenv…
✔ Successfully created virtual environment!
Virtualenv location: /Users/dhillard/.local/share/virtualenvs/myproj-nAbMEAt0
Creating a Pipfile for this project…
Pipfile.lock not found, creating…
Locking [dev-packages] dependencies…
Locking [packages] dependencies…
Updated Pipfile.lock (a65489)!
Installing dependencies from Pipfile.lock (a65489)…
 🐍   ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉ 0/0 — 00:00:00
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.

// Activate virtual environment (uses a subshell)
$ pipenv shell
Launching subshell in virtual environment…
 . /Users/dhillard/.local/share/virtualenvs/test-nAbMEAt0/bin/activate

// Exit virtual environment (by exiting subshell)
(myproj-nAbMEAt0)$ exit
```py

为您完成创建虚拟环境并激活它的所有繁重工作。如果你仔细观察，你会发现它还创建了一个名为`Pipfile`的文件。在您第一次运行`pipenv install`之后，这个文件只包含几样东西:

```
[[source]] name  =  "pypi" url  =  "https://pypi.org/simple" verify_ssl  =  true [dev-packages] [packages] [requires] python_version  =  "3.7"
```py

特别注意，它显示的是`python_version = "3.7"`。默认情况下，`pipenv`创建一个虚拟 Python 环境，使用的 Python 版本与它安装时的版本相同。如果您想使用不同的 Python 版本，那么您可以在运行`pipenv install`之前自己创建`Pipfile`，并指定您想要的版本。如果你已经安装了`pyenv`，那么`pipenv`会在必要的时候用它来安装指定的 Python 版本。

抽象虚拟环境管理是`pipenv`的一个崇高目标，但是它偶尔会因为难以理解的错误而被挂起。试一试，但如果你感到困惑或不知所措，不要担心。随着它的成熟，工具、文档和社区将围绕它成长和改进。

要深入了解虚拟环境，请务必阅读 [Python 虚拟环境:初级读本](https://realpython.com/python-virtual-environments-a-primer)。

[*Remove ads*](/account/join/)

## 包装管理

对于您从事的许多项目，您可能需要一些第三方包。这些包可能依次有自己的依赖项。在 Python 的早期，使用包需要手动下载文件，并让 Python 指向它们。今天，我们很幸运有各种各样的包管理工具可供我们使用。

大多数包管理器与虚拟环境协同工作，将您在一个 Python 环境中安装的包与另一个环境隔离开来。将这两者结合使用，您将真正开始看到可用工具的威力。

### `pip`

[`pip`](https://realpython.com/courses/what-is-pip/)(**p**IP**I**installs**p**packages)几年来一直是 Python 中包管理事实上的标准。它很大程度上受到了一个叫做`easy_install`的早期工具的启发。Python 从 3.4 版本开始将 [`pip`](https://realpython.com/what-is-pip/) 合并到标准发行版中。`pip`自动化下载包并让 Python 知道它们的过程。

如果您有多个虚拟环境，那么您可以看到它们是通过在一个环境中安装几个包来隔离的:

```
$ pyenv virtualenv 3.7.3 proj1
$ pyenv activate proj1
(proj1)$ pip list
Package    Version
---------- ---------
pip        19.1.1
setuptools 40.8.0

(proj1)$ python -m pip install requests
Collecting requests
 Downloading .../requests-2.22.0-py2.py3-none-any.whl (57kB)
 100% |████████████████████████████████| 61kB 2.2MB/s
Collecting chardet<3.1.0,>=3.0.2 (from requests)
 Downloading .../chardet-3.0.4-py2.py3-none-any.whl (133kB)
 100% |████████████████████████████████| 143kB 1.7MB/s
Collecting certifi>=2017.4.17 (from requests)
 Downloading .../certifi-2019.6.16-py2.py3-none-any.whl (157kB)
 100% |████████████████████████████████| 163kB 6.0MB/s
Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 (from requests)
 Downloading .../urllib3-1.25.3-py2.py3-none-any.whl (150kB)
 100% |████████████████████████████████| 153kB 1.7MB/s
Collecting idna<2.9,>=2.5 (from requests)
 Downloading .../idna-2.8-py2.py3-none-any.whl (58kB)
 100% |████████████████████████████████| 61kB 26.6MB/s
Installing collected packages: chardet, certifi, urllib3, idna, requests
Successfully installed packages

$ pip list
Package    Version
---------- ---------
certifi    2019.6.16
chardet    3.0.4
idna       2.8
pip        19.1.1
requests   2.22.0
setuptools 40.8.0
urllib3    1.25.3
```py

`pip`已安装`requests`，以及它所依赖的几个包。`pip list`显示所有当前安装的软件包及其版本。

**警告**:例如，你可以使用`pip uninstall requests`卸载软件包，但是这将*只*卸载`requests`——而不是它的任何依赖项。

为`pip`指定项目依赖关系的一种常见方式是使用`requirements.txt`文件。文件中的每一行都指定了一个包名，并且可以选择要安装的版本:

```
scipy==1.3.0
requests==2.22.0
```py

然后，您可以运行`python -m pip install -r requirements.txt`来一次安装所有指定的依赖项。有关`pip`的更多信息，请参见[什么是 Pip？新蟒蛇指南](https://realpython.com/what-is-pip/)。

### `pipenv`

[`pipenv`](https://docs.pipenv.org/en/latest/) 与`pip`有着大部分相同的基本操作，但是考虑包的方式有点不同。还记得`pipenv`创造的`Pipfile`吗？当你安装一个包时，`pipenv`会将这个包添加到`Pipfile`中，同时也会将更多的详细信息添加到一个名为`Pipfile.lock`的新**锁文件**中。锁定文件充当已安装软件包的精确集合的快照，包括直接依赖项及其子依赖项。

你可以看到`pipenv`在你安装包的时候整理包管理:

```
$ pipenv install requests
Installing requests…
Adding requests to Pipfile's [packages]…
✔ Installation Succeeded
Pipfile.lock (444a6d) out of date, updating to (a65489)…
Locking [dev-packages] dependencies…
Locking [packages] dependencies…
✔ Success!
Updated Pipfile.lock (444a6d)!
Installing dependencies from Pipfile.lock (444a6d)…
 🐍   ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉ 5/5 — 00:00:00
```py

`pipenv`将使用这个锁文件，如果存在的话，来安装同一套软件包。您可以确保在使用这种方法创建的任何 Python 环境中，您总是拥有相同的工作依赖集。

`pipenv`还区分**开发依赖**和**生产(常规)依赖**。在开发过程中，您可能需要一些工具，例如 [`black`](https://github.com/python/black) 或 [`flake8`](http://flake8.pycqa.org/en/latest/) ，而在生产中运行您的应用程序时，您并不需要这些工具。您可以在安装软件包时指定该软件包用于开发:

```
$ pipenv install --dev flake8
Installing flake8…
Adding flake8 to Pipfile's [dev-packages]…
✔ Installation Succeeded
...
```py

默认情况下，`pipenv install`(没有任何参数)将只安装您的产品包，但是您也可以使用`pipenv install --dev`告诉它安装开发依赖项。

[*Remove ads*](/account/join/)

### `poetry`

[`poetry`](https://poetry.eustace.io) 解决了包管理的其他方面，包括创建和发布你自己的包。安装`poetry`后，您可以使用它创建一个新项目:

```
$ poetry new myproj
Created package myproj in myproj
$ ls myproj/
README.rst    myproj    pyproject.toml    tests
```py

类似于`pipenv`如何创建`Pipfile`，`poetry`创建一个 [`pyproject.toml`](https://realpython.com/courses/packaging-with-pyproject-toml/) 文件。这个[最新标准](https://www.python.org/dev/peps/pep-0518/#file-format)包含关于项目的元数据以及依赖版本:

```
[tool.poetry] name  =  "myproj" version  =  "0.1.0" description  =  "" authors  =  ["Dane Hillard <github@danehillard.com>"] [tool.poetry.dependencies] python  =  "^3.7" [tool.poetry.dev-dependencies] pytest  =  "^3.0" [build-system] requires  =  ["poetry>=0.12"] build-backend  =  "poetry.masonry.api"
```py

可以用`poetry add`安装包(或者用`poetry add --dev`作为开发依赖):

```
$ poetry add requests
Using version ^2.22 for requests

Updating dependencies
Resolving dependencies... (0.2s)

Writing lock file

Package operations: 5 installs, 0 updates, 0 removals

 - Installing certifi (2019.6.16)
 - Installing chardet (3.0.4)
 - Installing idna (2.8)
 - Installing urllib3 (1.25.3)
 - Installing requests (2.22.0)
```py

`poetry`也维护一个锁文件，它比`pipenv`有优势，因为它跟踪哪些包是子依赖包。这样一来，你就可以卸载`requests` *和*与其`poetry remove requests`的依赖关系。

### `conda`

有了`conda`，你可以照常使用`pip`安装包，但是你也可以使用`conda install`安装包来自不同的**通道**，这些通道是 Anaconda 或者其他提供者提供的包的集合。要从`conda-forge`通道安装`requests`，可以运行`conda install -c conda-forge requests`。

在[中的`conda`中了解更多关于包管理的信息在 Windows](https://realpython.com/python-windows-machine-learning-setup/) 上设置 Python 进行机器学习。

## Python 解释器

如果您对 Python 环境的进一步定制感兴趣，可以选择与 Python 交互时的命令行体验。Python 解释器提供了一个**读取-评估-打印循环** (REPL)，这是当您在 shell 中键入不带参数的`python`时出现的情况:

>>>

```
Python 3.7.3 (default, Jun 17 2019, 14:09:05)
[Clang 10.0.1 (clang-1001.0.46.4)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 2 + 2
4
>>> exit()
```py

REPL **读取**您键入的内容，**将**评估为 Python 代码，**打印**结果。然后它等待重新做一遍。这大约是默认的 Python REPL 提供的数量，足以完成大部分典型工作。

### IPython

像 Anaconda 一样， [IPython](https://ipython.org/) 是一套不仅仅支持 Python 的工具，但是它的主要特性之一是一个替代的 Python REPL。IPython 的 REPL 对每个命令进行编号，并明确标记每个命令的输入和输出。安装 IPython ( `python -m pip install ipython`)后，您可以运行`ipython`命令代替`python`命令来使用 IPython REPL:

>>>

```
Python 3.7.3
Type 'copyright', 'credits' or 'license' for more information
IPython 6.0.0.dev -- An enhanced Interactive Python. Type '?' for help.

In [1]: 2 + 2
Out[1]: 4

In [2]: print("Hello!")
Out[2]: Hello!
```

IPython 还支持 `Tab` 完成，更强大的帮助特性，以及与其他工具如 [`matplotlib`](https://matplotlib.org/) 的强大集成，用于绘图。IPython 为 [Jupyter](https://jupyter.org/) 提供了基础，两者都因为与其他工具的集成而在数据科学社区中得到了广泛的应用。

IPython REPL 也是高度可配置的，所以尽管它还算不上一个完整的开发环境，但它仍然可以提高您的工作效率。它内置的可定制的[魔法命令](https://ipython.org/ipython-doc/3/interactive/tutorial.html#magic-functions)值得一试。

[*Remove ads*](/account/join/)

### `bpython`

[`bpython`](https://bpython-interpreter.org) 是另一个可选择的 REPL，它提供了行内语法高亮显示、制表符补全，甚至在您键入时提供自动建议。它提供了 IPython 的许多快捷的好处，而没有对接口做太多改变。如果没有集成等等的权重，`bpython`可能会很好地添加到您的清单中一段时间，看看它如何改进您对 REPL 的使用。

## 文本编辑器

你一生中有三分之一的时间在睡觉，所以投资一张好床是有意义的。作为一名开发人员，您花了大量时间阅读和编写代码，因此您应该投入时间按照您喜欢的方式设置 Python 环境的文本编辑器。

每个编辑器都提供了一组不同的按键绑定和操作文本的模型。一些需要鼠标来有效地与它们交互，而另一些只需要键盘就可以控制。有些人认为他们选择的文本编辑器和定制是他们做出的最个人的决定！

在这个舞台上有如此多的选择，所以我不会试图在这里详细介绍它。查看[Python ide 和代码编辑器(指南)](https://realpython.com/python-ides-code-editors-guide/)获得广泛的概述。一个好的策略是找一个简单的小文本编辑器来快速修改，找一个全功能的 IDE 来完成更复杂的工作。 [Vim](https://www.vim.org/) 和 [PyCharm](https://www.jetbrains.com/pycharm/) 分别是我选择的编辑器。

## Python 环境提示和技巧

一旦您做出了关于 Python 环境的重大决定，剩下的路就由一些小的调整铺就，让您的生活变得更加轻松。这些调整每个都可以节省几分钟或几秒钟，但是它们合起来可以节省你几个小时的时间。

让某项活动变得更容易可以减少你的认知负荷，这样你就可以专注于手头的任务，而不是围绕它的后勤工作。如果你注意到自己一遍又一遍地执行一个动作，那么考虑自动化它。使用 XKCD 的这张美妙的图表来决定是否值得自动化一项特定的任务。

这里是一些最后的提示。

**了解您当前的虚拟环境**

如前所述，在命令提示符中显示活动的 Python 版本或虚拟环境是一个好主意。大多数工具会为您做到这一点，但是如果不这样做(或者如果您想要定制提示)，该值通常包含在`VIRTUAL_ENV`环境变量中。

**禁用不必要的临时文件**

你有没有注意到`*.pyc`文件遍布你的项目目录？这些文件是预编译的 Python 字节码——它们帮助 Python 更快地启动应用程序。在生产中，这是一个很好的主意，因为它们会给你一些性能增益。然而，在本地开发过程中，它们很少有用。设置`PYTHONDONTWRITEBYTECODE=1`禁用该行为。如果您以后发现了它们的用例，那么您可以很容易地从 Python 环境中移除它们。

**定制您的 Python 解释器**

您可以使用一个**启动文件**来影响 REPL 的行为。Python 将在进入 REPL 之前读取这个启动文件并执行其中包含的代码。将`PYTHONSTARTUP`环境变量设置为启动文件的路径。(我的在`~/.pystartup`。)如果您想像您的 shell 提供的那样点击 `Up` 查看命令历史，点击 `Tab` 查看完成，那么尝试一下[这个启动文件](https://github.com/daneah/dotfiles/blob/master/source/pystartup)。

## 结论

您了解了典型 Python 环境的许多方面。有了这些知识，您可以:

*   选择具有您喜欢的美感和增强功能的终端
*   根据您的需要选择带有任意多(或少)定制选项的 shell
*   管理系统上多个版本的 Python
*   使用虚拟 Python 环境管理使用单一版本 Python 的多个项目
*   在您的虚拟环境中安装软件包
*   选择适合您的交互式编码需求的 REPL

当你已经有了自己的 Python 环境，我希望你能分享关于你的完美设置✨的截图、截屏或博客帖子*******