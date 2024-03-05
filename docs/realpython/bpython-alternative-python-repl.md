# 探索 bpython:具有类似 IDE 特性的 Python REPL

> 原文：<https://realpython.com/bpython-alternative-python-repl/>

标准的 Python 解释器让你[从文件中运行脚本](https://realpython.com/run-python-scripts/)或者[在所谓的**读取-评估-打印循环(REPL)** 中动态交互执行代码](https://realpython.com/interacting-with-python/)。虽然这是一个通过对代码输入的即时反馈来探索语言和发现其库的强大工具，但 Python 附带的默认 REPL 有几个限制。幸运的是，像 **bpython** 这样的替代品提供了一种对程序员更加友好和方便的体验。

您可以使用 bpython 来试验您的代码或快速测试一个想法，而无需在不同程序之间切换上下文，就像在**集成开发环境(IDE)** 中一样。此外，bpython 可能是虚拟或物理教室中一个有价值的教学工具。

**在本教程中，您将学习如何:**

*   安装并使用 bpython 作为您的**替代 Python REPL**
*   由于 bpython 的独特特性，提高您的**生产力**
*   调整 bpython 的**配置**和它的**颜色主题**
*   使用通用的**键盘快捷键**更快地编码
*   在 GitHub 上为 bpython 的**开源项目**做出贡献

在开始本教程之前，请确保您已经熟悉了 [Python 基础知识](https://realpython.com/products/python-basics-book/)，并且知道如何在命令行中启动标准的 Python REPL。此外，你应该能够用`pip`[安装包，理想情况下进入一个](https://realpython.com/what-is-pip/)[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)。

要下载您将在本教程中使用的配置文件和示例脚本，请单击下面的链接:

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/bpython-alternative-python-repl-code/)，您将使用它来驾驭 bpython 的力量。

## 开始使用 bpython

与独立的 python 发行版不同，例如 [CPython](https://realpython.com/cpython-source-code-guide/) 、 [PyPy](https://realpython.com/pypy-faster-python/) 或 [Anaconda](https://www.anaconda.com/products/distribution) 、 [bpython](https://pypi.org/project/bpython/) 仅仅是一个**纯 Python 包**，作为所选 Python 解释器的轻量级包装器。因此，您可以在任何特定的 python 发行版、版本甚至虚拟环境上使用 bpython，这为您提供了很大的灵活性。

**注意:**bpython 中的字母 *b* 代表[鲍勃·法雷尔](https://github.com/bobf)，他是该工具的原作者和维护者。

与此同时，bpython 仍然是一个熟悉的 Python REPL，只有一些基本特性，如语法高亮和自动完成，是从成熟的[Python ide](https://realpython.com/python-ides-code-editors-guide/)借鉴来的。这种**极简方法**与 [IPython](https://ipython.org/) 等工具形成对比，后者是标准 Python REPL 的另一种替代方案，在数据科学界很流行。IPython 引入了许多定制命令和其他额外的功能，这些功能在 [vanilla](https://en.wikipedia.org/wiki/Vanilla_software) Python 中是没有的。

有几种方法可以在您的计算机上安装 bpython。像[家酿](https://brew.sh/)或 [APT](https://en.wikipedia.org/wiki/APT_(software)) 这样的包管理器为你的操作系统提供预构建版本的 bpython。然而，它们很可能已经过时，并被硬连接到系统范围的 Python 解释器中。虽然您可以手工从其源代码构建最新的 bpython 版本，但最好将其安装到一个带有 [`pip`](https://realpython.com/what-is-pip/) 的 [**虚拟环境**](https://realpython.com/python-virtual-environments-a-primer/) :

```py
(venv) $ python -m pip install bpython
```

在许多虚拟环境中，将 bpython 安装在多个副本中是很常见的，这很好。这允许您将 bpython 包装在您最初用来创建虚拟环境的特定 python 解释器周围。

**注意:**不幸的是，bpython 在 Windows 上没有本地支持，因为它依赖于 [curses 库](https://en.wikipedia.org/wiki/Curses_(programming_library))，而这个库只在类似 Unix 的系统上可用，比如 macOS 和 Linux。官方文档[提到了一个变通办法](https://docs.bpython-interpreter.org/en/latest/windows.html)，它依赖于一个非官方的 Windows 二进制文件，但似乎不再管用了。如果您使用的是 Windows，那么您最好的选择是安装 [Windows 子系统 for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) 并从那里使用 bpython。

安装完成后，您可以使用以下两个命令之一启动 bpython:

1.  `bpython`
2.  `python -m bpython`

最好选择更明确的第二个命令，它将 bpython 作为一个**可运行的 python 模块**来调用。这样，您将确保运行安装在当前活动虚拟环境中的 bpython 程序。

另一方面，使用简单的`bpython`命令可以悄悄地回到全局安装的程序，如果有的话。它也可以在您的 shell 中别名为不同的可执行文件，优先于本地`bpython`模块。

下面是一个示例，展示了如何针对封装在隔离虚拟环境中的几个不同的 python 解释器使用 bpython:

```py
(py2.7) $ python -m bpython
bpython version 0.20.1 on top of Python 2.7.18
 ⮑ /home/realpython/py2.7/bin/python WARNING: You are using `bpython` on Python 2\. Support for Python 2
 ⮑ has been deprecated in version 0.19 and might disappear
 ⮑ in a future version.
>>> import platform
>>> platform.python_version()
'2.7.18'
>>> platform.python_implementation()
'CPython'

(py3.11) $ python -m bpython
bpython version 0.23 on top of Python 3.11.0
 ⮑ /home/realpython/py3.11/bin/python >>> import platform
>>> platform.python_version()
'3.11.0'
>>> platform.python_implementation()
'CPython'

(pypy) $ python -m bpython
bpython version 0.23 on top of Python 3.9.12
 ⮑ /home/realpython/pypy/bin/python >>> import platform
>>>> platform.python_version()
'3.9.12'
>>> platform.python_implementation()
'PyPy'
```

注意，您使用相同的命令从不同的虚拟环境运行 bpython。每一个突出显示的行都指出了解释器版本以及 bpython 在当前 REPL 会话中包装的 Python 可执行文件的路径。可以通过标准库中的 [`platform`](https://docs.python.org/3/library/platform.html) 模块确认 Python 版本及其实现。

**注意:**[Django web 框架](https://realpython.com/django-setup/)可以检测到安装在虚拟环境中的 bpython。当您执行 [shell 命令](https://docs.djangoproject.com/en/4.1/ref/django-admin/#shell)来调用 python 交互式解释器以及模块搜索路径上的项目文件时，框架将自动运行 bpython。

好了，现在你已经学习了如何安装和运行 bpython 作为一个**替代 Python REPL** ，是时候探索它的关键特性了。在接下来的几节中，无论您的技能水平如何，您都将发现 bpython 可以提高您作为 python 程序员的生产率的几种方式。

## 一眼看出错别字

![](img/ffcd460964ede470a1b18d280ef88bda.png)

加入我们，访问数以千计的教程和 Pythonistas 专家社区。

[*解锁本文*](/account/join/?utm_source=rp_article_preview&utm_content=bpython-alternative-python-repl)

*已经是会员了？[签到](/account/login/)

![](img/ffcd460964ede470a1b18d280ef88bda.png)

全文仅供会员阅读。加入我们，访问数以千计的教程和 Pythonistas 专家社区。

[*解锁本文*](/account/join/?utm_source=rp_article_preview&utm_content=bpython-alternative-python-repl)

*已经是会员了？[签到](/account/login/)**