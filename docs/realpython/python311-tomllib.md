# Python 3.11 预览版:TOML 和 tomllib

> 原文：<https://realpython.com/python311-tomllib/>

Python 3.11 离最终发布越来越近，最终发布将在 2022 年 10 月。新版本目前正在进行 beta 测试，你可以自己安装它来预览和测试一些新功能，包括支持使用新的`tomllib`模块读取 TOML。

TOML 是一种配置文件格式，在 Python 生态系统中越来越流行。这是由采用`pyproject.toml`作为 Python 打包中的中央配置文件所驱动的。其他重要的工具，像[黑色](https://pypi.org/project/black/)、 [mypy](https://mypy.readthedocs.io/en/stable/) 和 [pytest](https://realpython.com/pytest-python-testing/) 也使用 TOML 进行配置。

**在本教程中，您将:**

*   在你的电脑上安装 Python 3.11 测试版，就在你当前安装的 Python 旁边
*   熟悉 **TOML 格式**的基础知识
*   **使用新的`tomllib`模块读取 TOML** 文件
*   **用第三方库编写 TOML** ，了解为什么这个功能**没有包含在`tomllib`的**中
*   探索 Python 3.11 新的**类型特性**，包括`Self`和`LiteralString`类型以及可变泛型

Python 3.11 中还有许多其他新特性和改进。查看变更日志中的[新增内容](https://docs.python.org/3.11/whatsnew/3.11.html)以获得最新列表，并在 Real Python 上阅读其他 [Python 3.11 预览版](https://realpython.com/search?kind=article&q=python+3.11)以了解其他特性。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 3.11 测试版

Python 的新版本在每年 10 月发布。代码是在发布日期前[经过 17 个月的时间](https://www.python.org/dev/peps/pep-0602/)开发和测试的。新功能在 [alpha 阶段](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)实现。对于 Python 3.11，在 2021 年 10 月至 2022 年 4 月期间，共发布了七个 **alpha 版本**。

Python 3.11 的第一个 **beta 版本**发生在 2022 年 5 月 8 日[凌晨](https://twitter.com/pyblogsal/status/1523636192587423744)。每个这样的预发布都由一个发布经理协调——目前是[Pablo ga lindo Salgado](https://twitter.com/pyblogsal)——并将来自 Python 核心开发者和其他志愿者的数百个提交集合在一起。

这个版本也标志着新版本的**功能冻结**。换句话说，Python 3.11.0b1 中没有的新功能不会添加到 Python 3.11 中。相反，功能冻结和发布日期(2022 年 10 月 3 日)之间的时间用于测试和巩固代码。

大约每月[一次](https://www.python.org/dev/peps/pep-0664/)在[测试阶段](https://en.wikipedia.org/wiki/Software_release_life_cycle#Beta)，Python 的核心开发者发布一个新的**测试版本**，继续展示新特性，测试它们，并获得早期反馈。目前 Python 3.11 的最新测试版是 **3.11.0b3** ，发布于[2022 年](https://www.python.org/downloads/release/python-3110b3/)6 月 1 日。

**注:**本教程使用的是 Python 3.11 的第三个 beta 版本。如果您使用更高版本，可能会遇到一些小的差异。然而，`tomllib`建立在一个成熟的库之上，你可以预期你在本教程中学到的东西将在 Python 3.11 的测试阶段和最终版本中保持不变。

如果你在维护你自己的 Python 包，那么测试阶段是一个重要的时期，你应该开始用新版本测试你的包。核心开发人员与社区一起，希望在最终发布之前找到并修复尽可能多的 bug。

[*Remove ads*](/account/join/)

### 很酷的新功能

Python 3.11 的一些亮点包括:

*   **增强的错误消息**，帮助您更有效地调试代码
*   **任务和异常组**，它们简化了异步代码的使用，并允许程序同时引发和处理多个异常
*   **TOML 支持**，它允许你使用标准库解析 TOML 文档
*   静态类型改进，让你更精确地注释你的代码
*   **优化**，承诺让 Python 3.11 比以前的版本快很多

Python 3.11 有很多值得期待的地方！你已经可以在早期的 [Python 3.11 预览版](https://realpython.com/search?kind=article&q=python+3.11)文章中读到关于[增强的错误消息](https://realpython.com/python311-error-messages/)和[任务和异常组](https://realpython.com/python311-exception-groups/)。要获得全面的概述，请查看 [Python 3.11:供您尝试的酷新功能](https://realpython.com/python311-new-features/)。

在本教程中，您将关注如何使用新的`tomllib`库来读取和解析 TOML 文件。您还将看到 Python 3.11 中的一些打字改进。

### 安装

要使用本教程中的代码示例，您需要在系统上安装 Python 3.11 版本。在这一小节中，你将学习几种不同的方法:使用 **Docker** ，使用 **pyenv** ，或者从**源**安装。选择最适合您和您的系统的一个。

**注意:**测试版是即将推出的功能的预览。虽然大多数特性都可以很好地工作，但是你不应该依赖任何 Python 3.11 beta 版本的产品，或者任何潜在错误会带来严重后果的地方。

如果您可以在您的系统上访问 [Docker](https://docs.docker.com/get-docker/) ，那么您可以通过拉取并运行`python:3.11-rc-slim` [Docker 镜像](https://hub.docker.com/_/python)来下载最新版本的 Python 3.11:

```py
$ docker pull python:3.11-rc-slim
3.11-rc-slim: Pulling from library/python
[...]
docker.io/library/python:3.11-rc-slim

$ docker run -it --rm python:3.11-rc-slim
```

这会将您带入 Python 3.11 REPL。查看 Docker 中的[运行 Python 版本，了解更多关于通过 Docker 使用 Python 的信息，包括如何运行脚本。](https://realpython.com/python-versions-docker/#running-python-in-a-docker-container)

[pyenv](https://realpython.com/intro-to-pyenv/) 工具非常适合管理系统上不同版本的 Python，如果你愿意，你可以用它来安装 Python 3.11 beta。它有两个不同的版本，一个用于 Windows，一个用于 Linux 和 macOS。使用下面的切换器选择您的平台:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)

**在 Windows 上，你可以使用 [pyenv-win](https://pyenv-win.github.io/pyenv-win/) 。首先更新您的`pyenv`安装:

```py
PS> pyenv update
:: [Info] ::  Mirror: https://www.python.org/ftp/python
[...]
```

进行更新可以确保您可以安装最新版本的 Python。你也可以[手动更新`pyenv`](https://pyenv-win.github.io/pyenv-win/#how-to-update-pyenv)。

在 Linux 和 macOS 上，可以使用 [pyenv](https://github.com/pyenv/pyenv) 。首先使用 [`pyenv-update`](https://github.com/pyenv/pyenv-update) 插件更新您的`pyenv`安装:

```py
$ pyenv update
Updating /home/realpython/.pyenv...
[...]
```

进行更新可以确保您可以安装最新版本的 Python。如果你不想使用更新插件，那么你可以[手动更新`pyenv`](https://github.com/pyenv/pyenv#upgrading)。

使用`pyenv install --list`查看 Python 3.11 有哪些版本。然后，安装最新版本:

```py
$ pyenv install 3.11.0b3
Downloading Python-3.11.0b3.tar.xz...
[...]
```

安装可能需要几分钟时间。一旦安装了新的测试版，你就可以创建一个虚拟环境，在这里你可以玩它:

*   [*视窗*](#windows-2)
**   [**Linux + macOS**](#linux-macos-2)*

```py
PS> pyenv local 3.11.0b3
PS> python --version
Python 3.11.0b3

PS> python -m venv venv
PS> venv\Scripts\activate
```

您使用`pyenv local`激活您的 Python 3.11 版本，然后使用`python -m venv`设置虚拟环境。

```py
$ pyenv virtualenv 3.11.0b3 311_preview
$ pyenv activate 311_preview
(311_preview) $ python --version
Python 3.11.0b3
```

在 Linux 和 macOS 上，你使用 [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv) 插件来设置虚拟环境并激活它。

你也可以从[python.org](https://www.python.org/)的预发布版本中安装 Python。选择[最新预发布](https://www.python.org/download/pre-releases/)，向下滚动到页面底部的*文件*部分。下载并安装与您的系统对应的文件。更多信息参见 [Python 3 安装&设置指南](https://realpython.com/installing-python/)。

本教程中的大多数示例都依赖于新特性，因此您应该使用 Python 3.11 可执行文件来运行它们。具体如何运行可执行文件取决于您的安装方式。如果你需要帮助，那么看看关于 [Docker](https://realpython.com/python-versions-docker/#running-python-in-a-docker-container) 、 [pyenv](https://realpython.com/intro-to-pyenv/) 、[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)或者[从源码](https://realpython.com/installing-python/)安装的相关教程。

[*Remove ads*](/account/join/)

## `tomllib`Python 3.11 中的 TOML 解析器

Python 是一门成熟的语言。Python 的第一个公共版本发布于 30 多年前的 1991 年。Python 的许多独特特性，包括显式异常处理、对空白的依赖以及丰富的数据结构，如列表和字典，甚至在早期的[就已经存在。](https://en.wikipedia.org/wiki/History_of_Python)

然而，Python 的第一个版本缺少的一个特性是共享社区包和模块的便捷方式。这并不奇怪。事实上，Python 和[万维网](https://en.wikipedia.org/wiki/World_Wide_Web)几乎是同时发明的。1991 年底，全世界只有[12 台网络服务器](https://w3.org/2012/08/history-of-the-web/origins.htm#c6p4)，而且没有一台是专门用于发布 Python 代码的。

随着时间的推移，Python 和 T2 互联网变得越来越流行。几个[倡议](https://realpython.com/pypi-publish-python-package/#get-to-know-python-packaging)旨在允许共享 Python 代码。这些特性有机地发展，导致 Python 与打包的关系有些混乱。

在过去的几十年里，这个问题已经通过几个[打包 pep](https://peps.python.org/topic/packaging/)(Python 增强提案)得到了解决，对于[库维护者](https://realpython.com/pypi-publish-python-package/)和[最终用户](https://pypi.org/)来说，情况已经有了很大的改善。

一个挑战是构建包依赖于执行一个`setup.py`文件，但是没有机制知道该文件依赖于哪些依赖项。这就产生了一种[先有鸡还是先有蛋](https://en.wikipedia.org/wiki/Chicken_or_the_egg)的问题，你需要运行`setup.py`来发现如何运行`setup.py`。

实际上，[`pip`](https://realpython.com/what-is-pip/)——Python 的包管理器——假设它应该使用 [Setuptools](https://setuptools.pypa.io/) 来构建包，并且 Setuptools 在你的计算机上是可用的。这使得使用像 [Flit](https://flit.pypa.io) 和[poems](https://realpython.com/dependency-management-python-poetry/)这样的替代构建系统变得更加困难。

为了解决这种情况， [PEP 518](https://peps.python.org/pep-0518/) 引入了 **`pyproject.toml`** 配置文件，它指定了 Python 项目构建依赖关系。2016 年接受了 PEP 518。当时，TOML 仍然是一种相当新的格式，而且 Python 或其标准库中没有对解析 TOML 的内置支持。

随着 TOML 格式的成熟和`pyproject.toml`文件的使用，Python 3.11 增加了对解析 TOML 文件的支持。在这一节中，您将了解更多关于什么是 TOML 格式，如何使用新的`tomllib`来解析 TOML 文档，以及为什么`tomllib`不支持编写 TOML 文件。

### 学习基本的 TOML

[Tom Preston-Werner](https://tom.preston-werner.com/) 先是[宣布](https://github.com/toml-lang/toml/releases/tag/v0.1.0) **Tom 的显而易见、极简的语言**——俗称**TOML**——并于 2013 年发布了其规范的[版本 0.1.0](https://toml.io/en/v0.1.0) 。从一开始，TOML 的目标就是提供一种“最小化的配置文件格式，由于语义明显，易于阅读”( [Source](https://toml.io/en/v0.1.0#objectives) )。TOML 规范的稳定[版本 1.0.0](https://toml.io/en/v1.0.0) 于 2021 年 1 月发布。

TOML 文件是一个 [UTF-8](https://realpython.com/python-encodings-guide/#enter-unicode) 编码的，区分大小写的文本文件。TOML 中的主要构件是**键-值对**，其中键与值用等号(`=`)隔开:

```py
version  =  3.11
```

在这个最小的 TOML 文档中，`version`是一个具有相应值`3.11`的键。TOML 中的值有类型。`3.11`解释为**浮点数**。您可以利用的其他基本类型有[字符串](https://realpython.com/python-strings/)、[布尔值](https://realpython.com/python-boolean/)、[整数](https://realpython.com/python-numbers/#integers)和日期:

```py
version  =  3.11 release_manager  =  "Pablo Galindo Salgado" is_beta  =  true beta_release  =  3 release_date  =  2022-06-01
```

这个例子展示了其中的大部分类型。语法类似于 Python 的语法，除了有小写布尔和一个特殊的日期文字。在其基本形式中，TOML 键值对类似于 Python 变量赋值，因此它们应该看起来很熟悉。关于这些和其他相似之处的更多细节，请查看 [TOML 文档](https://toml.io/)。

从本质上讲，TOML 文档是键值对的集合。您可以通过将它们包装在数组和表中，为这些对添加一些结构。一个**数组**是一个值列表，类似于一个 Python `list`。一个**表**是一个键值对的嵌套集合，类似于 Python `dict`。

使用方括号将数组的元素括起来。表格从命名表格的`[key]`行开始:

```py
[python] version  =  3.11 release_manager  =  "Pablo Galindo Salgado" is_beta  =  true beta_release  =  3 release_date  =  2022-06-01 peps  =  [657,  654,  678,  680,  673,  675,  646,  659] [toml] version  =  1.0 release_date  =  2021-01-12
```

这个 TOML 文档可以用 Python 表示如下:

```py
{
    "python": {
        "version": 3.11,
        "release_manager": "Pablo Galindo Salgado",
        "is_beta": True,
        "beta_release": 3,
        "release_date": datetime.date(2022, 6, 1),
        "peps": [657, 654, 678, 680, 673, 675, 646, 659],
    },
    "toml": {
        "version": 1.0,
        "release_date": datetime.date(2021, 1, 12),
    },
}
```

TOML 中的`[python]`键在 Python 中由字典中的`"python"`键表示，指向包含 TOML 部分中所有键值对的嵌套字典。TOML 表可以任意嵌套，一个 TOML 文档可以包含几个 TOML 表。

这就结束了对 TOML 语法的简短介绍。虽然 TOML 的设计有一个相当简单的语法，但是这里还有一些细节没有涉及到。要深入了解，请查看 [Python 和 TOML:新的最好的朋友](https://realpython.com/python-toml)或 [TOML 规范](https://toml.io/en/latest)。

除了语法之外，您还应该考虑如何解释 TOML 文件中的值。TOML 文档通常用于配置。最终，其他一些应用程序会使用 TOML 文档中的信息。因此，该应用程序对 TOML 文件的内容有一些期望。这意味着一个 TOML 文档可能有两种不同的错误:

1.  **语法错误:**TOML 文档不是有效的 TOML。TOML 解析器通常会捕捉到这一点。
2.  **模式错误:**TOML 文档是有效的 TOML，但是它的结构不是应用程序所期望的。应用程序本身必须处理这个问题。

TOML 规范目前还不包括一种可以用来验证 TOML 文档结构的模式语言，尽管有几个提案存在。这种模式将检查给定的 TOML 文档是否包含给定用例的正确的表、键和值类型。

作为一个非正式模式的例子， [PEP 517](https://peps.python.org/pep-0517/#source-trees) 和 [PEP 518](https://peps.python.org/pep-0518/#build-system-table) 说一个`pyproject.toml`文件应该定义`build-system`表，该表必须包括关键字`requires`和`build-backend`。此外，`requires`的值必须是字符串数组，而`build-backend`的值必须是字符串。下面是一个满足这个模式的 TOML 文档的[示例](https://realpython.com/pypi-publish-python-package/#configure-your-package):

```py
# pyproject.toml [build-system] requires  =  ["setuptools>=61.0.0",  "wheel"] build-backend  =  "setuptools.build_meta"
```

本例遵循 PEP 517 和 PEP 518 的要求。然而，验证通常由构建者前端完成。

**注意:**如果你想了解更多关于用 Python 构建自己的包的知识，请查看[如何向 PyPI 发布开源 Python 包](https://realpython.com/pypi-publish-python-package/)。

您可以自己检查这个验证。创建以下错误的`pyproject.toml`文件:

```py
# pyproject.toml [build-system] requires  =  "setuptools>=61.0.0" backend  =  "setuptools.build_meta"
```

这是有效的 TOML，因此该文件可以被任何 TOML 解析器读取。但是，根据 PEPs 中的要求，它不是有效的`build-system`表。为了确认这一点，安装 [`build`](https://pypa-build.readthedocs.io/) ，这是一个符合 PEP 517 的构建前端，并基于您的`pyproject.toml`文件执行构建:

```py
(venv) $ python -m pip install build
(venv) $ python -m build
ERROR Failed to validate `build-system` in pyproject.toml:
 `requires` must be an array of strings
```

错误消息指出`requires`必须是一个字符串数组，如 PEP 518 中所指定的。尝试其他版本的`pyproject.toml`文件，注意`build`为你做的其他验证。您可能需要在自己的应用程序中实现类似的验证。

到目前为止，您已经看到了一些 TOML 文档的例子，但是您还没有探索如何在您自己的项目中使用它们。在下一小节中，您将了解如何使用标准库中新的`tomllib`包来读取和解析 Python 3.11 中的 TOML 文件。

[*Remove ads*](/account/join/)

### 用`tomllib`读 TOML】

Python 3.11 在标准库中新增了一个模块，名为 [`tomllib`](https://docs.python.org/3.11/library/tomllib.html) 。您可以使用`tomllib`来读取和解析任何符合 TOML v1.0 的文档。在这一小节中，您将学习如何直接从文件和包含 TOML 文档的字符串中加载 TOML。

[PEP 680](https://peps.python.org/pep-0680/) 描述了`tomllib`和一些导致 TOML 支持被添加到标准库中的过程。在 Python 3.11 中包含`tomllib`的两个决定性因素是`pyproject.toml`在 Python 打包生态系统中扮演的核心角色，以及 TOML 规范将在 2021 年初达到 1.0 版本。

`tomllib`的实现或多或少是直接从[的](https://github.com/hukkin) [`tomli`](https://pypi.org/project/tomli/) 中剽窃来的，他也是 PEP 680 的合著者之一。

`tomllib`模块非常简单，因为它只包含两个函数:

1.  **`load()`** 从文件中读取 TOML 文件。
2.  **`loads()`** 从字符串中读取 TOML 文件。

您将首先看到如何使用`tomllib`来读取下面的`pyproject.toml`文件，它是 [`tomli`](https://github.com/hukkin/tomli) 项目中相同文件的简化版本:

```py
# pyproject.toml [build-system] requires  =  ["flit_core>=3.2.0,<4"] build-backend  =  "flit_core.buildapi" [project] name  =  "tomli" version  =  "2.0.1"  # DO NOT EDIT THIS LINE MANUALLY. LET bump2version DO IT description  =  "A lil' TOML parser" requires-python  =  ">=3.7" readme  =  "README.md" keywords  =  ["toml"] [project.urls] "Homepage"  =  "https://github.com/hukkin/tomli" "PyPI"  =  "https://pypi.org/project/tomli"
```

复制该文档，并将其保存在本地文件系统上名为`pyproject.toml`的文件中。现在，您可以开始 REPL 会话，探索 Python 3.11 的 TOML 支持:

>>>

```py
>>> import tomllib
>>> with open("pyproject.toml", mode="rb") as fp: ...     tomllib.load(fp) ...
{'build-system': {'requires': ['flit_core>=3.2.0,<4'],
 'build-backend': 'flit_core.buildapi'},
 'project': {'name': 'tomli',
 'version': '2.0.1',
 'description': "A lil' TOML parser",
 'requires-python': '>=3.7',
 'readme': 'README.md',
 'keywords': ['toml'],
 'urls': {'Homepage': 'https://github.com/hukkin/tomli',
 'PyPI': 'https://pypi.org/project/tomli'}}}
```

通过向函数传递一个文件指针，使用`load()`来读取和解析 TOML 文件。注意，文件指针必须指向二进制流。确保这一点的一种方法是使用`open()`和`mode="rb"`，其中`b`表示二进制模式。

**注意:**根据 [PEP 680](https://peps.python.org/pep-0680/#types-accepted-as-the-first-argument-of-tomllib-load) 的规定，文件必须以二进制模式打开，这样`tomllib`才能确保 UTF-8 编码在所有系统上都得到正确处理。

将原始的 TOML 文档与生成的 Python 数据结构进行比较。文档由 Python 字典表示，其中所有的键都是字符串，TOML 中的不同表表示为嵌套字典。注意，原始文件中关于`version`的注释被忽略，并且不是结果的一部分。

您可以使用`loads()`来加载已经用字符串表示的 TOML 文档。以下示例解析来自[前面的子节](#learn-basic-toml)的示例:

>>>

```py
>>> import tomllib
>>> document = """
... [python]
... version = 3.11
... release_manager = "Pablo Galindo Salgado"
... is_beta = true
... beta_release = 3
... release_date = 2022-06-01
... peps = [657, 654, 678, 680, 673, 675, 646, 659]
... ... [toml]
... version = 1.0
... release_date = 2021-01-12
... """

>>> tomllib.loads(document)
{'python': {'version': 3.11,
 'release_manager': 'Pablo Galindo Salgado',
 'is_beta': True,
 'beta_release': 3,
 'release_date': datetime.date(2022, 6, 1),
 'peps': [657, 654, 678, 680, 673, 675, 646, 659]},
 'toml': {'version': 1.0,
 'release_date': datetime.date(2021, 1, 12)}}
```

与`load()`类似，`loads()`返回一个字典。一般来说，表示基于基本的 Python 类型:`str`、`float`、`int`、`bool`，以及[字典](https://realpython.com/python-dicts/)、[列表](https://realpython.com/python-lists-tuples/#python-lists)和[、`datetime`对象](https://realpython.com/python-datetime/)。`tomllib`文档包括一个[转换表](https://docs.python.org/3.11/library/tomllib.html#conversion-table)，它展示了如何用 Python 表示 TOML 类型。

如果您愿意，那么您可以使用`loads()`结合`pathlib`从文件中读取 TOML:

>>>

```py
>>> import pathlib
>>> import tomllib

>>> path = pathlib.Path("pyproject.toml")
>>> with path.open(mode="rb") as fp:
...     from_load = tomllib.load(fp) ...
>>> from_loads = tomllib.loads(path.read_text()) 
>>> from_load == from_loads
True
```

在这个例子中，你使用`load()`和`loads()`来加载`pyproject.toml`。然后确认无论如何加载文件，Python 表示都是相同的。

`load()`和`loads()`都接受一个可选参数: [`parse_float`](https://docs.python.org/3.11/library/tomllib.html#tomllib.load) 。这允许您控制如何用 Python 解析和表示浮点数。默认情况下，它们被解析并存储为`float`对象，在大多数 Python 实现中，这些对象是 64 位的，精度为[精度](https://en.wikipedia.org/wiki/Double-precision_floating-point_format)的大约 16 位十进制数字。

另一种方法是，如果你需要用更精确的数字，用 [`decimal.Decimal`](https://realpython.com/python-rounding/#the-decimal-class) 代替:

>>>

```py
>>> import tomllib
>>> from decimal import Decimal
>>> document = """
... small = 0.12345678901234567890
... large = 9999.12345678901234567890
... """

>>> tomllib.loads(document)
{'small': 0.12345678901234568,
 'large': 9999.123456789011}

>>> tomllib.loads(document, parse_float=Decimal)
{'small': Decimal('0.12345678901234567890'),
 'large': Decimal('9999.12345678901234567890')}
```

这里加载一个带有两个键值对的 TOML 文档。默认情况下，当使用`load()`或`loads()`时，您会损失一些精度。通过使用`Decimal`类，您可以保持输入的精确性。

如上所述，`tomllib`模块改编自流行的`tomli`模块。如果你想在需要支持旧版本 Python 的代码库上使用 TOML 和`tomllib`，那么你可以依靠`tomli`。为此，请在您的需求文件中添加以下行:

```py
tomli >= 1.1.0 ; python_version < "3.11"
```

这将在 3.11 之前的 Python 版本上使用时安装`tomli`。在您的源代码中，您可以适当地使用`tomllib`或`tomli`和下面的[导入](https://realpython.com/python-import/#handle-packages-across-python-versions):

```py
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
```

这段代码将在 Python 3.11 和更高版本中导入`tomllib`。如果`tomllib`不可用，那么`tomli`被导入并别名为`tomllib`名。

您已经看到了如何使用`tomllib`来读取 TOML 文档。您可能想知道如何编写 TOML 文件。原来不能用`tomllib`写 TOML。请继续阅读，了解原因，并查看一些替代方案。

[*Remove ads*](/account/join/)

### 写入 toml〔t0〕

类似的现有库如`json`和`pickle`包括`load()`和`dump()`函数，后者用于写数据。`dump()`功能，以及相应的`dumps()`，被故意排除在`tomllib`之外。

根据 [PEP 680](https://peps.python.org/pep-0680/#including-an-api-for-writing-toml) 和围绕它的讨论，这样做有几个原因:

*   将`tomllib`包含在标准库中的主要动机是为了能够*读取生态系统中使用的* TOML 文件。

*   TOML 格式被设计成一种对人友好的配置格式，所以许多 TOML 文件都是手工编写的。

*   TOML 格式不是像 JSON 或 pickle 那样的数据序列化格式，所以没有必要完全与`json`和`pickle`API 保持一致。

*   TOML 文档可能包含在写入文件时应该保留的注释和格式。这与将 TOML 表示为基本 Python 类型不兼容。

*   关于如何布局和格式化 TOML 文件有不同的观点。

*   没有一个核心开发人员表示有兴趣为`tomllib`维护一个写 API。

一旦某些东西被添加到标准库中，就很难更改或删除，因为有人依赖它。这是一件好事，因为这意味着 Python 在很大程度上保持了向后兼容:在 Python 3.10 上运行的 Python 程序很少会在 Python 3.11 上停止工作。

另一个后果是，核心团队对添加新功能持保守态度。如果有明确的需求，可以在以后添加对编写 TOML 文档的支持。

不过，这不会让你空手而归。有几个第三方 TOML 编写器可用。`tomllib`文档提到了两个包:

*   [**`tomli-w`**](https://pypi.org/project/tomli-w/) 顾名思义就是可以写 TOML 文档的`tomli`的兄弟姐妹。这是一个简单的模块，没有很多选项来控制输出。
*   [**`tomlkit`**](https://pypi.org/project/tomlkit/) 是一个强大的处理 TOML 文档的软件包，它支持读写。它保留注释、缩进和其他空白。TOML 工具包是为[诗](https://realpython.com/dependency-management-python-poetry/)开发和使用的。

根据您的用例，其中一个包可能会满足您的 TOML 编写需求。

如果你不想仅仅为了写一个 TOML 文件而添加一个外部依赖，那么你也可以试着滚动你自己的 writer。以下示例显示了一个不完整的 TOML 编写器的示例。它不支持 TOML v1.0 的所有特性，但是它支持编写您之前看到的`pyproject.toml`示例:

```py
# tomllib_w.py

from datetime import date

def dumps(toml_dict, table=""):
    document = []
    for key, value in toml_dict.items():
        match value:
            case dict():
                table_key = f"{table}.{key}" if table else key
                document.append(
                    f"\n[{table_key}]\n{dumps(value, table=table_key)}"
                )
            case _:
                document.append(f"{key} = {_dumps_value(value)}")
    return "\n".join(document)

def _dumps_value(value):
    match value:
        case bool():
            return "true" if value else "false"
        case float() | int():
            return str(value)
        case str():
            return f'"{value}"'
        case date():
            return value.isoformat()
        case list():
            return f"[{', '.join(_dumps_value(v) for v in value)}]"
        case _:
            raise TypeError(
                f"{type(value).__name__}  {value!r} is not supported"
            )
```

`dumps()`函数接受一个代表 TOML 文档的字典。它通过遍历字典中的键值对，将字典转换为字符串。你很快就会更仔细地了解细节。首先，您应该检查代码是否有效。打开 REPL 并导入`dumps()`:

>>>

```py
>>> from tomllib_w import dumps
>>> print(dumps({"version": 3.11, "module": "tomllib_w", "stdlib": False}))
version = 3.11
module = "tomllib_w"
stdlib = false
```

你用不同类型的值编写一个简单的字典。它们被正确地写成 TOML 类型:数字是普通的，字符串用双引号括起来，布尔值是小写的。

回头看看代码。大多数对 TOML 类型的序列化发生在助手函数`_dumps_value()`中。它使用[结构模式匹配](https://realpython.com/python310-new-features/#structural-pattern-matching)基于`value`的类型构造不同种类的 TOML 字符串。

主`dumps()`函数与字典一起工作。它遍历每个键值对。如果值是另一个字典，那么它通过添加一个表头来构造一个 TOML 表，然后递归地调用自己来处理表中的键值对。如果值不是一个字典，那么`_dumps_value()`用于正确地将键-值对转换成 TOML。

如上所述，这个编写器不支持完整的 TOML 规范。例如，它不支持 TOML 中可用的所有日期和时间类型，也不支持内嵌或数组表等嵌套结构。在字符串处理中也有一些不被支持的边缘情况。但是，对于许多应用程序来说，这已经足够了。

例如，您可以尝试加载并转储您之前使用的`pyproject.toml`文件:

>>>

```py
>>> import tomllib
>>> from tomllib_w import dumps
>>> with open("pyproject.toml", mode="rb") as fp:
...     pyproject = tomllib.load(fp)
...
>>> print(dumps(pyproject))

[build-system]
requires = ["flit_core>=3.2.0,<4"]
build-backend = "flit_core.buildapi"

[project]
name = "tomli"
version = "2.0.1"
description = "A lil' TOML parser"
requires-python = ">=3.7"
readme = "README.md"
keywords = ["toml"]

[project.urls]
Homepage = "https://github.com/hukkin/tomli"
PyPI = "https://pypi.org/project/tomli"
```

这里你先用`tomllib`读`pyproject.toml`。然后使用自己的`tomllib_w`模块将 TOML 文档写回控制台。

如果你需要更好的支持来编写 TOML 文档，你可以扩展一下`tomllib_w`。然而，在大多数情况下，你应该依赖一个现有的包，比如`tomli_w`或者`tomlkit`。

虽然 Python 3.11 不支持编写 TOML 文件，但是包含的 TOML 解析器对许多项目都很有用。接下来，您可以将 TOML 用于您的配置文件，因为您知道在 Python 中读取它们将获得一流的支持。

[*Remove ads*](/account/join/)

## 其他新功能

TOML 支持当然值得庆祝，但是 Python 3.11 中也有一些小的改进。在很长一段时间里，Python 的[类型检查](https://realpython.com/python-type-checking/)领域已经出现了这样的增量变化。

[PEP 484](https://peps.python.org/pep-0484/) 引入了类型提示。它们从 Pyhon 3.5 开始就可用了，每一个新的 Python 版本[都为静态类型系统增加了功能](https://dafoster.net/articles/2021/01/26/python's-type-checking-renaissance/)。[在](https://twitter.com/llanga) [PyCon US 2022](https://realpython.com/python-news-april-2022/#pycon-us-2022) 大会的[主题演讲中，茹卡兹·兰加](https://www.youtube.com/watch?v=wbohVjhqg7c)谈到了类型检查。

Python 3.11 接受了几个新的与类型相关的 pep。您将很快了解到更多关于`Self`类型、`LiteralString`类型和可变泛型的知识。

**注意:**类型检查增强有点特殊，因为它们依赖于您的 Python 版本和类型检查工具的版本。最新的测试版支持一些新的 Python 3.11 类型系统特性，但是还没有在所有的类型检查器中实现。

例如，你可以在他们的 [GitHub 页面](https://github.com/python/mypy/issues/12840)上监控 [mypy 的](https://mypy.readthedocs.io/)对新功能的支持状态。

甚至还有一些新的与打字相关的特性，下面就不介绍了。 [PEP 681](https://peps.python.org/pep-0681/) 增加了`@dataclass_transform` [装饰器](https://realpython.com/primer-on-python-decorators/)，可以标记语义类似于[数据类](https://realpython.com/python-data-classes/)的类。此外， [PEP 655](https://peps.python.org/pep-0655/) 允许您在[类型化词典](https://realpython.com/python38-new-features/#more-precise-types)中标记必填和可选字段。

### 自身类型

[PEP 673](https://peps.python.org/pep-0673/) 引入了一个新的`Self`类型，它动态地引用当前的类。当您用返回类实例的方法实现类时，这很有用。考虑由[极坐标](https://en.wikipedia.org/wiki/Polar_coordinate_system)表示的二维点的以下部分实现:

```py
# polar_point.py

import math
from dataclasses import dataclass

@dataclass
class PolarPoint:
    r: float
    φ: float

    @classmethod
    def from_xy(cls, x, y):
        return cls(r=math.hypot(x, y), φ=math.atan2(y, x))
```

您添加了`.from_xy()`构造函数，这样您就可以方便地从相应的[笛卡尔坐标](https://en.wikipedia.org/wiki/Cartesian_coordinate_system)创建`PolarPoint`实例。

**注意:**属性名`.r`和`.φ`是特意选择来模仿[公式](https://en.wikipedia.org/wiki/Polar_coordinate_system#Converting_between_polar_and_Cartesian_coordinates)中使用的数学符号。

一般来说，[建议](https://peps.python.org/pep-0008/#naming-conventions)为属性使用更长更具描述性的名称。然而，有时候遵循你的问题领域的惯例也是有用的。如果你愿意的话，可以随意用`.radius`替换`.r`，用`.phi`或`.angle`替换`.φ`。

Python 源代码由[默认](https://docs.python.org/3/reference/lexical_analysis.html)编码在 [UTF-8](https://realpython.com/python-encodings-guide/#enter-unicode) 中。然而，[标识符](https://peps.python.org/pep-3131/)像变量和属性[不能使用](https://github.com/python/cpython/pull/1686)完整的 Unicode 字母表。例如，在你的变量和属性名中，你必须[远离](https://www.youtube.com/watch?v=Wtm7Iy-wEUI&t=52m43s)和[表情符号](https://pypi.org/project/pythonji)。

您可以按如下方式使用新类:

>>>

```py
>>> from polar_point import PolarPoint
>>> point = PolarPoint.from_xy(3, 4)
>>> point
PolarPoint(r=5.0, φ=0.9272952180016122)

>>> from math import cos
>>> point.r * cos(point.φ)
3.0000000000000004
```

这里，首先创建一个表示笛卡尔点(3，4)的点。在极坐标中，这个点用半径`r` = 5.0，角度`φ` ≈ 0.927 来表示。您可以使用公式`x = r * cos(φ)`转换回笛卡尔`x`坐标。

现在，您想给`.from_xy()`添加类型提示。它返回一个`PolarPoint`对象。然而，在这一点上你不能直接使用`PolarPoint`作为注释，因为那个类还没有被完全定义。相反，您可以使用带引号的`"PolarPoint"`或者添加一个 [PEP 563](https://peps.python.org/pep-0563/) future import，使[推迟注释的求值](https://realpython.com/python37-new-features/#typing-enhancements)。

这两种变通方法都有其缺点，目前推荐的是用一个`TypeVar`代替。这种方法即使在子类中也能工作，但是它很麻烦并且容易出错。

使用新的`Self`类型，您可以向您的类添加类型提示，如下所示:

```py
import math
from dataclasses import dataclass
from typing import Self 
@dataclass
class PolarPoint:
    r: float
    φ: float

    @classmethod
 def from_xy(cls, x: float, y: float) -> Self:        return cls(r=math.hypot(x, y), φ=math.atan2(y, x))
```

注释`-> Self`表明`.from_xy()`将返回当前类的一个实例。如果你创建了一个`PolarPoint`的子类，这也可以正常工作。

工具箱中有了`Self`类型，就可以更方便地使用类和面向对象的特性(如继承)向项目添加静态类型。

[*Remove ads*](/account/join/)

### 任意文字字符串类型

Python 3.11 中的另一个新类型是`LiteralString`。虽然这个名字可能会让你想起 Python 3.8 中添加的`Literal`，但是`LiteralString`的主要用例有点不同。要理解将它添加到类型系统的动机，首先退一步考虑字符串。

一般来说，Python 不关心如何构造字符串:

>>>

```py
>>> s1 = "Python"
>>> s2 = "".join(["P", "y", "t", "h", "o", "n"])
>>> s3 = input()
Python 
>>> s1 == s2 == s3
True
```

在这个例子中，您以三种不同的方式创建字符串`"Python"`。首先，将它指定为一个文字字符串。接下来，将六个单字符字符串连接起来，形成字符串`"Python"`。最后，您使用 [`input()`](https://docs.python.org/3/library/functions.html#input) 从用户输入中读取字符串。

最后的测试显示每个字符串的值是相同的。在大多数应用程序中，您不需要关心特定的字符串是如何构造的。但是，有些时候您需要小心，尤其是在处理用户输入时。

不幸的是，针对数据库的攻击非常普遍。 [Java Log4j 漏洞](https://nvd.nist.gov/vuln/detail/CVE-2021-44228)同样利用日志系统执行任意代码。

回到上面的例子。虽然`s1`和`s3`的值恰好相同，但是您对这两个字符串的信任应该是完全不同的。假设您需要构建一个 SQL 语句，从数据库中读取关于用户的信息:

>>>

```py
>>> def get_user_sql(user_id):
...     return f"SELECT * FROM users WHERE user_id = '{user_id}'"
...

>>> user_id = "Bobby"
>>> get_user_sql(user_id)
"SELECT * FROM users WHERE user_id = 'Bobby'"

>>> user_id = input()
Robert'; DROP TABLE users; -- 
>>> get_user_sql(user_id)
"SELECT * FROM users WHERE user_id = 'Robert'; DROP TABLE users; --'"
```

这是对一个经典的 SQL 注入例子的改编。恶意用户可以利用编写任意 SQL 代码的能力进行破坏。如果最后一条 SQL 语句被执行，那么它将删除`users`表。

有许多机制可以抵御这类攻击。 [PEP 675](https://peps.python.org/pep-0675/) 名单上又多了一个。一种新的类型被添加到了`typing`模块中:`LiteralString`是一种特殊的字符串类型，它是在您的代码中定义的。

您可以使用`LiteralString`来标记易受用户控制字符串攻击的函数。例如，执行 SQL 查询的函数可以注释如下:

```py
from typing import LiteralString

def execute_sql(query: LiteralString):
    # ...
```

类型检查器会特别注意在这个函数中作为`query`传递的值的类型。以下字符串将全部被允许作为`execute_sql`的参数:

>>>

```py
>>> execute_sql("SELECT * FROM users")

>>> table = "users"
>>> execute_sql("SELECT * FROM " + table)

>>> execute_sql(f"SELECT * FROM {table}")
```

最后两个例子没问题，因为`query`是从文字字符串构建的。如果字符串的所有部分都是按字面定义的，则该字符串仅被识别为`LiteralString`。例如，以下示例将无法通过类型检查:

>>>

```py
>>> user_input = input()
users

>>> execute_sql("SELECT * FROM " + user_input)
```

即使`user_input`的值恰好与前面的`table`的值相同，类型检查器也会在这里产生一个错误。用户控制着`user_input`的值，并有可能将其更改为对您的应用程序不安全的值。如果您使用`LiteralString`标记这些易受攻击的函数，类型检查器将帮助您跟踪需要格外小心的情况。

### 可变泛型类型

一个**通用类型**指定了一个用其他类型参数化的类型，例如一个字符串列表或一个由一个整数、一个字符串和另一个整数组成的元组。Python 使用方括号来参数化泛型。你把这两个例子分别写成`list[str]`和`tuple[int, str, int]`。

一个**变量**是一个接受可变数量参数的实体。例如，`print()`在 Python 中是一个[变量函数](https://en.wikipedia.org/wiki/Variadic_function):

>>>

```py
>>> print("abc", 123, "def")
abc 123 def
```

通过使用 [`*args`和`**kwargs`](https://realpython.com/python-kwargs-and-args/) 来捕获多个位置和关键字参数，您可以定义自己的变量函数。

如果你想指定你自己的类是泛型的，你可以使用`typing.Generic`。下面是一个向量的例子，也称为一维数组:

```py
# vector.py

from typing import Generic, TypeVar

T = TypeVar("T")

class Vector(Generic[T]):
    ...
```

[类型变量](https://realpython.com/python-type-checking/#type-variables) `T`用作任何类型的替身。可以在类型注释中使用`Vector`,如下所示:

>>>

```py
>>> from vector import Vector
>>> position: Vector[float]
```

在这个特定的例子中，`T`将是`float`。为了让你的代码更加清晰和类型安全，你也可以使用[类型别名](https://realpython.com/python-type-checking/#type-aliases)甚至[专用的派生类型](https://mypy.readthedocs.io/en/stable/more_types.html#newtypes):

>>>

```py
>>> from typing import NewType
>>> from vector import Vector

>>> Coordinate = NewType("Coordinate", float)
>>> Coordinate(3.11)
3.11
>>> type(Coordinate(3.11))
<class 'float'>

>>> position: Vector[Coordinate]
```

这里，`Coordinate`在运行时的行为类似于`float`，但是静态类型检查将区分`Coordinate`和`float`。

现在，假设您创建了一个更通用的数组类，它可以处理可变数量的维度。直到现在，还没有好的方法来指定这样的**可变泛型**。

[PEP 646](https://peps.python.org/pep-0646/) 引入 [`typing.TypeVarTuple`](https://docs.python.org/3.11/library/typing.html#typing.TypeVarTuple) 来处理这个用例。这些**类型变量元组**本质上是包装在元组中的任意数量的类型变量。您可以使用它们来定义任意维数的数组:

```py
# ndarray.py

from typing import Generic, TypeVarTuple

Ts = TypeVarTuple("Ts")

class Array(Generic[*Ts]):
    ...
```

注意解包操作符(`*`)的使用。这是语法的必要部分，表明`Ts`代表可变数量的类型。

**注意:**在 3.11 之前的 Python 版本上可以从`typing_extensions`导入`TypeVarTuple`。然而，`*Ts`语法在这些版本上不起作用。作为等价的替代，你可以用 [`typing_extensions.Unpack`](https://docs.python.org/3.11/library/typing.html#typing.Unpack) 并写成`Unpack[Ts]`。

您可以使用`NewType`来标记数组中的尺寸，或者使用 [`Literal`](https://realpython.com/python38-new-features/#more-precise-types) 来指定精确的形状:

>>>

```py
>>> from typing import Literal, NewType
>>> from ndarray import Array

>>> Height = NewType("Height", int)
>>> Width = NewType("Width", int)
>>> Channels = NewType("Channels", int)
>>> image: Array[Height, Width, Channels]

>>> video_frame: Array[Literal[1920], Literal[1080], Literal[3]]
```

您将`image`标注为一个三维数组，其维度标记为`Height`、`Width`和`Channels`。您不需要指定这些维度的大小。第二个例子，`video_frame`，用文字值注释。实际上，这意味着`video_frame`必须是一个特定形状为 1920 × 1080 × 3 的数组。

可变泛型的主要动机是对数组进行类型化，就像你在上面的例子中看到的那样。然而，也有[其他用例](https://realpython.com/python-news-february-2022/#pep-646-variadic-generics)。一旦工具到位， [NumPy](https://realpython.com/numpy-tutorial/) 和[其他](https://peps.python.org/pep-0646/#endorsements)数组库[计划实现](https://github.com/numpy/numpy/issues/16544#issuecomment-1058675773)可变泛型。

[*Remove ads*](/account/join/)

## 结论

在本教程中，您了解了 Python 3.11 中的一些新特性。虽然最终版本将于 2022 年 10 月发布，但你已经可以下载测试版并尝试新功能。在这里，您已经探索了新的`tomllib`模块，并逐渐熟悉了 TOML 格式。

**您已经完成了以下操作:**

*   **在您的计算机上安装了** Python 3.11 测试版，就在您当前安装的 Python 旁边
*   **使用新的`tomllib`模块读取 TOML** 文件
*   用第三方库编写了 TOML 并创建了自己的函数来编写 TOML 的子集
*   探索 Python 3.11 新的**类型特性**，包括`Self`和`LiteralString`类型以及可变泛型

你已经在你的项目中使用 TOML 了吗？尝试新的 TOML 解析器，并在下面的评论中分享你的经验。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。*************