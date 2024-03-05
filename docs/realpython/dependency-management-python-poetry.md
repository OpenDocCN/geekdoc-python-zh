# 使用 Python 诗歌进行依赖管理

> 原文：<https://realpython.com/dependency-management-python-poetry/>

当您的 Python 项目依赖于外部包时，您需要确保使用每个包的正确版本。更新后，包可能不会像更新前那样工作。像 Python**poem**这样的依赖管理器可以帮助您指定、安装和解析项目中的外部包。这样，您可以确保在每台机器上始终使用正确的依赖版本。

**在本教程中，您将学习如何:**

*   开始一个新的诗歌项目
*   向现有的项目添加诗歌
*   使用 **`pyproject.toml`** 文件
*   引脚**依赖关系**
*   安装依赖 **`poetry.lock`**
*   执行基本的**诗歌 CLI** 命令

使用[诗歌](https://python-poetry.org/)将帮助你开始新项目，维护现有项目，并掌握**依赖管理**。您将准备好使用`pyproject.toml`文件，这将是定义 Python 项目中构建需求的[标准](https://www.python.org/dev/peps/pep-0518/)。

为了完成本教程并充分利用它，您应该对[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)、[模块和包](https://realpython.com/python-modules-packages/)和 [`pip`](https://realpython.com/what-is-pip/) 有一个基本的了解。

虽然本教程关注的是依赖管理，但是诗歌也可以帮助你构建和打包项目。如果你想分享你的作品，那么你甚至可以[发布](https://python-poetry.org/docs/cli/#publish)你的诗歌项目到 [Python 打包索引(PyPI)](https://pypi.org/) 。

**免费奖励:** ，向您展示如何使用 Pip、PyPI、Virtualenv 和需求文件等工具避免常见的依赖管理问题。

## 满足先决条件

在深入 Python 诗歌的本质之前，您需要考虑一些先决条件。首先，您将阅读本教程中会遇到的术语的简短概述。接下来，您将安装诗歌本身。

[*Remove ads*](/account/join/)

### 相关术语

如果你曾经在你的 Python 脚本中使用过`import`语句，那么你就使用过**模块**。其中一些模块可能是您自己编写的 Python 文件。其他的可能是**内置的**模块，比如[日期时间](https://realpython.com/python-datetime/)。然而，有时 Python 提供的还不够。这时你可能会求助于外部的打包模块。当你的 Python 代码依赖外部模块时，你可以说这些**包**是你的项目的**依赖**。

你可以在 [PyPI](https://pypi.org/) 中找到不属于 [Python 标准库](https://docs.python.org/3/py-modindex.html)的包。在了解如何工作之前，您需要在您的系统上安装诗歌。

### Python 诗歌装置

要在命令行中使用诗歌，您应该在系统范围内安装它。如果你只是想尝试一下，那么你可以使用 [`pip`](https://pypi.org/project/poetry/) 将其安装到虚拟环境中。但是您应该小心地尝试这种方法，因为诗歌会安装它自己的依赖项，这可能会与您在项目中使用的其他包冲突。

推荐使用官方的脚本来安装诗歌 T2。您可以手动下载并运行这个 [Python 文件](https://github.com/python-poetry/poetry/blob/master/install-poetry.py)，或者选择下面的操作系统来使用适当的命令:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
PS C:\> (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py -UseBasicParsing).Content | python -
```

如果您使用的是 Windows，那么您可以使用带有`-UseBasicParsing`选项的`Invoke-Webrequest` cmdlet 将请求的 URL 内容下载到**标准输出流(stdout)** 。使用管道字符(`|`，您将把输出交给`python`的**标准输入流(stdin)** 。在这种情况下，您将`install-poetry.py`的内容通过*管道*传输到您的 Python 解释器。

**注意:**部分用户[在 Windows 10 上使用 PowerShell 命令时](https://github.com/python-poetry/poetry/issues/2795)报错。

```py
$ curl https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
```

使用`curl`，您将请求的 URL 的内容输出到**标准输出流(stdout)** 。通过使用带有管道字符(`|`)的 Unix 管道，您将把输出交给`python3`的**标准输入流(stdin)** 。在这种情况下，您将*将`install-poetry.py`的内容通过管道*传输到您的 Python 解释器。

**注意:**如果你在 macOS 上，那么你可能会得到一个`ssl.SSLCertVerificationError`。如果没有为 SSL 模块安装默认的**根证书**，就会出现这个错误。您可以通过运行 Python 文件夹中的命令脚本来安装它们:

```py
$ open "/Applications/Python 3.9/Install Certificates.command"
```

根据您安装的 Python 版本，Python interpeter 的具体路径可能会有所不同。在这种情况下，您需要相应地调整上面命令中的路径。

运行该命令后，上面的`curl`命令应该没有任何错误。

在输出中，您应该会看到安装完成的消息。你可以在你的终端中运行`poetry --version`，看看`poetry`是否工作。这个命令将显示你当前的诗歌版本。如果你想更新诗歌，那么你可以运行`poetry self update`。

## Python 诗歌入门

装了诗，就该看看诗是怎么做的了。在本节中，您将学习如何开始一个新的诗歌项目，以及如何将诗歌添加到现有项目中。您还将看到项目结构并检查`pyproject.toml`文件。

### 创建新的诗歌项目

您可以使用`new`命令和项目名称作为参数来创建一个新的诗歌项目。在本教程中，该项目被称为`rp-poetry`。创建项目，然后移动到新创建的目录中:

```py
$ poetry new rp-poetry
$ cd rp-poetry
```

通过运行`poetry new rp-poetry`，您创建了一个名为`rp-poetry/`的新文件夹。当您查看文件夹内部时，您会看到一个结构:

```py
rp-poetry/
│
├── rp_poetry/
│   └── __init__.py
│
├── tests/
│   ├── __init__.py
│   └── test_rp_poetry.py
│
├── README.rst
└── pyproject.toml
```

诗自动为你规范包名。它将项目名称中的破折号(`-`)转换成文件夹名称`rp_poetry/`中的下划线(`_`)。否则，这个名称在 Python 中是不允许的，所以您不能将其作为模块导入。为了对创建包名有更多的控制，您可以使用`--name`选项来命名它，不同于项目文件夹:

```py
$ poetry new rp-poetry --name realpoetry
```

如果您喜欢将您的源代码存储在一个额外的`src/`父文件夹中，那么 poems 可以让您通过使用`--src`标志来遵守这个约定:

```py
$ poetry new --src rp-poetry
$ cd rp-poetry
```

通过添加`--src`标志，您已经创建了一个名为`src/`的文件夹，其中包含您的`rp_poetry/`目录:

```py
rp-poetry/
│
├── src/
│   │
│   └── rp_poetry/
│       └── __init__.py
│
├── tests/
│   ├── __init__.py
│   └── test_rp_poetry.py
│
├── README.rst
└── pyproject.toml
```

当创建一个新的诗歌项目时，你会马上收到一个基本的文件夹结构。

[*Remove ads*](/account/join/)

### 检查项目结构

`rp_poetry/`子文件夹本身还不是很壮观。在这个目录中，您将找到一个包含您的软件包版本的`__init__.py`文件:

```py
# rp_poetry/__init__.py

__version__ = "0.1.0"
```

当你跳到`tests/`文件夹并打开`test_rp_poetry.py`时，你会注意到`rp_poetry`已经可以导入了:

```py
# tests/test_rp_poetry.py

from rp_poetry import __version__

def test_version():
    assert __version__ == "0.1.0"
```

诗歌也为这个项目增加了第一个测试。`test_version()`函数检查`rp_poetry/__init__.py`的`__version__`变量是否包含期望的版本。然而，`__init__.py`文件并不是您定义软件包版本的唯一地方。另一个位置是`pyproject.toml`文件。

### 使用`pyproject.toml`文件

处理诗歌最重要的文件之一是`pyproject.toml`文件。这个文件不是诗歌的发明。这是一个在 PEP 518 中定义的**配置文件**标准:

> 这个 PEP 指定了 Python 软件包应该如何指定它们有什么构建依赖，以便执行它们选择的构建系统。作为本规范的一部分，引入了一个新的配置文件，供软件包用来指定它们的构建依赖关系(预期相同的配置文件将用于未来的配置细节)。([来源](https://www.python.org/dev/peps/pep-0518/#abstract))

作者考虑了上面引用的“新配置文件”的几种文件格式。最终，他们决定采用 **TOML** 格式，它代表[汤姆明显的最小语言](https://toml.io/en/)。在他们看来，TOML 足够灵活，比 YAML、JSON、CFG 或 INI 等其他选项具有更好的可读性和更低的复杂性。要查看 TOML 的外观，请打开`pyproject.toml`文件:

```py
 1# pyproject.toml 2
 3[tool.poetry] 4name  =  "rp-poetry" 5version  =  "0.1.0" 6description  =  "" 7authors  =  ["Philipp <philipp@realpython.com>"] 8
 9[tool.poetry.dependencies] 10python  =  "^3.9" 11
12[tool.poetry.dev-dependencies] 13pytest  =  "^5.2" 14
15[build-system] 16requires  =  ["poetry-core>=1.0.0"] 17build-backend  =  "poetry.core.masonry.api"
```

您可以在`pyproject.toml`文件中看到四个部分。这些部分被称为**表格**。它们包含像 poems 这样的工具识别并用于**依赖性管理**或**构建例程**的指令。

如果表名是特定于刀具的，则必须以`tool`为前缀。通过使用这样的**子表**，您可以为项目中的不同工具添加指令。这种情况下，只有`tool.poetry`。但是你可能会在其他项目中看到像 [pytest](https://docs.pytest.org/en/6.2.x/customize.html#pyproject-toml) 的`[tool.pytest.ini_options]`这样的例子。

在上面第 3 行的`[tool.poetry]`子表中，您可以存储关于您的诗歌项目的一般信息。你可用的键是由诗歌定义的[。虽然有些键是可选的，但有四个键是必须指定的:](https://python-poetry.org/docs/pyproject/)

1.  **`name`** :您的包的名称
2.  **`version`** :你的包的版本，理想情况下遵循[语义版本](https://semver.org/)
3.  **`description`** :您的包裹的简短描述
4.  **`authors`** :作者列表，格式`name <email>`

第 9 行的子表`[tool.poetry.dependencies]`和第 12 行的子表`[tool.poetry.dev-dependencies]`对于您的依赖管理是必不可少的。在下一节中，当您将依赖项添加到您的诗歌项目时，您将了解到关于这些子表的更多信息。现在，重要的事情是认识到在包依赖和开发依赖之间有*和*的区别。

`pyproject.toml`文件的最后一个表是第 15 行的`[build-system]`。这个表定义了诗歌和其他构建工具可以使用的数据，但是因为它不是特定于工具的，所以没有前缀。诗歌创建了`pyproject.toml`文件，其中有两个关键点:

1.  **`requires`** :构建包所需的依赖项列表，使这个键成为强制键
2.  **`build-backend`** :用于执行构建过程的 Python 对象

如果你想了解更多关于`pyproject.toml`文件的这一部分，那么你可以通过阅读 PEP 517 中的[源代码树来找到更多。](https://www.python.org/dev/peps/pep-0517/#source-trees)

当你用诗歌开始一个新项目时，这是你开始用的`pyproject.toml`文件。随着时间的推移，您将添加关于您的包和您正在使用的工具的配置详细信息。随着 Python 项目的增长，您的`pyproject.toml`文件也会随之增长。对于子表`[tool.poetry.dependencies]`和`[tool.poetry.dev-dependencies]`来说尤其如此。在下一节中，您将了解如何展开这些子表。

[*Remove ads*](/account/join/)

## 用 Python 写诗

一旦你建立了一个诗歌项目，真正的工作就可以开始了。一旦诗歌到位，你就可以开始编码了。一路上，你会发现诗歌如何为你提供一个虚拟的环境，并照顾你的依赖。

### 使用诗歌的虚拟环境

当您开始一个新的 Python 项目时，创建一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)是一个很好的实践。否则，您可能会混淆来自不同项目的不同依赖项。使用虚拟环境是 poem 的核心特性之一，它永远不会干扰您的全局 Python 安装。

然而，当你开始一个项目时，诗歌不会马上创造一个虚拟的环境。您可以通过让 poems 列出所有连接到当前项目的虚拟环境来确认 poems 没有创建虚拟环境。如果你还没有把`cd`变成`rp-poetry/`然后运行一个命令:

```py
$ poetry env list
```

目前，不应该有任何输出。

当你运行某些命令时，诗歌会在途中创建一个虚拟环境。如果您想要更好地控制虚拟环境的创建，那么您可能会决定明确地告诉 poems 您想要为它使用哪个 Python 版本，并从那里开始:

```py
$ poetry env use python3
```

使用这个命令，您使用的 Python 版本与您用来安装诗歌的版本相同。当你的`PATH` 中有 [Python 可执行文件时，使用`python3`就可以了。](https://realpython.com/add-python-to-path/)

**注意:**或者，你可以传递一个 Python 可执行文件的绝对路径。它应该与您可以在`pyproject.toml`文件中找到的 Python 版本约束相匹配。如果没有，那么您可能会遇到麻烦，因为您使用的 Python 版本不同于您的项目所需的版本。在您的环境中工作的代码在另一台机器上可能会出错。

更糟糕的是，外部包通常依赖于特定的 Python 版本。因此，安装您的包的用户可能会收到一个错误，因为您的依赖项版本与其 Python 版本不兼容。

当您运行`env use`时，您会看到一条消息:

```py
Creating virtualenv rp-poetry-AWdWY-py3.9 in ~/Library/Caches/pypoetry/virtualenvs
Using virtualenv: ~/Library/Caches/pypoetry/virtualenvs/rp-poetry-AWdWY-py3.9
```

如您所见，诗歌为您的项目环境构建了一个独特的名称。该名称包含项目名称和 Python 版本。中间看似随机的字符串是父目录的散列。有了这个唯一的字符串在中间，poems 就可以在您的系统上处理多个同名且 Python 版本相同的项目。这很重要，因为默认情况下，诗歌在同一个文件夹中创建所有的虚拟环境。

在没有任何其他配置的情况下，poem 在 poem 的**缓存目录**的`virtualenvs/`文件夹中创建虚拟环境:

| 操作系统 | 小路 |
| --- | --- |
| 马科斯 | `~/Library/Caches/pypoetry` |
| Windows 操作系统 | `C:\Users\<username>\AppData\Local\pypoetry\Cache` |
| Linux 操作系统 | `~/.cache/pypoetry` |

如果你想改变默认的缓存目录，那么你可以编辑[poems 的配置](https://python-poetry.org/docs/configuration)。当您已经在使用 [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) 或另一个第三方工具来管理您的虚拟环境时，这可能会很有用。要查看当前配置，包括已配置的`cache-dir`，您可以运行一个命令:

```py
$ poetry config --list
```

通常情况下，您不必更改这条路径。如果你想了解更多关于与诗歌的虚拟环境交互的知识，那么诗歌文档中有一章是关于管理环境的。

只要你在你的项目文件夹中，诗歌就会使用与之相关的虚拟环境。如果您有疑问，可以通过再次运行`env list`命令来检查虚拟环境是否被激活:

```py
$ poetry env list
```

这将显示类似于`rp-poetry-AWdWY-py3.9 (Activated)`的内容。有了激活的虚拟环境，您就可以开始管理一些依赖关系，并看到诗歌的光芒。

[*Remove ads*](/account/join/)

### 声明您的依赖关系

诗歌的一个关键要素是它对你的依赖的处理。在开始之前，看一下`pyproject.toml`文件中的两个依赖表:

```py
# rp_poetry/pyproject.toml (Excerpt) [tool.poetry.dependencies] python  =  "^3.9" [tool.poetry.dev-dependencies] pytest  =  "^5.2"
```

当前为您的项目声明了两个依赖项。一个是 Python 本身。另一个是 [pytest](https://docs.pytest.org) ，一个广泛使用的测试框架。正如您之前看到的，您的项目包含一个`tests/`文件夹和一个`test_rp_poetry.py`文件。有了 **pytest** 作为依赖，poems 可以在安装后立即运行您的测试。

**注:**在写这篇教程的时候，用 [Python 3.10](https://realpython.com/python310-new-features/) 运行`pytest`带诗是不行的。poem 安装的 pytest 版本与 Python 3.10 不兼容。

诗歌开发者[已经意识到这个问题](https://github.com/python-poetry/poetry/issues/4652)，随着诗歌 1.2 的发布，这个问题将会得到解决。

确保您在`rp-poetry/`项目文件夹中，并运行一个命令:

```py
$ poetry install
```

使用`install`命令，poems 检查您的`pyproject.toml`文件的依赖项，然后解析并安装它们。当您有许多依赖项需要不同版本的第三方包时，解析部分尤其重要。在安装任何包之前，poems 会计算出包的哪个版本满足了其他包按照他们的需求设置的版本约束。

除了`pytest`和它的需求，诗诗还安装项目本身。这样，您可以立即将`rp_poetry`导入到您的测试中:

```py
# tests/test_rp_poetry.py

from rp_poetry import __version__

def test_version():
    assert __version__ == "0.1.0"
```

安装好项目的包后，您可以将`rp_poetry`导入到您的测试中，并检查`__version__`字符串。安装了`pytest`之后，您可以使用`poetry run`命令来执行测试:

```py
 1$ poetry run pytest
 2========================= test session starts ==========================
 3platform darwin -- Python 3.9.1, pytest-5.4.3, py-1.10.0, pluggy-0.13.1
 4rootdir: /Users/philipp/Real Python/rp-poetry
 5collected 1 item
 6
 7tests/test_rp_poetry.py .                                        [100%]
 8
 9========================== 1 passed in 0.01s ===========================
```

您当前的测试运行成功，因此您可以放心地继续编码。然而，如果仔细观察第 3 行，有些东西看起来有点奇怪。上面写着`pytest-5.4.3`，而不是像`pyproject.toml`文件中写的`5.2`。接得好！

概括地说，您的`pyproject.toml`文件中的`pytest`依赖项如下所示:

```py
# rp_poetry/pyproject.toml (Excerpt) [tool.poetry.dev-dependencies] pytest  =  "^5.2"
```

`5.2`前面的插入符号(`^`)有特定的含义，它是诗歌提供的[版本约束](https://python-poetry.org/docs/dependency-specification/#version-constraints)之一。这意味着诗歌可以安装任何版本匹配最左边的非零数字的版本字符串。这意味着使用`5.4.3`是允许的。版本`6.0`将不被允许。

当诗歌试图解析依赖版本时，像插入符号这样的符号将变得很重要。如果只有两个要求，这并不太难。声明的依赖项越多，就越复杂。让我们看看 poem 如何通过在项目中安装新的包来处理这个问题。

### 用诗歌安装一个包

您可能以前使用过 [`pip`](https://realpython.com/what-is-pip/) 来安装不属于 Python 标准库的包。如果使用包名作为参数运行`pip install`，那么`pip`会在 [Python 包索引](https://pypi.org/)中查找包。你可以用同样的方式使用诗歌。

如果你想添加一个像 [`requests`](https://pypi.org/project/requests/) 这样的外部包到你的项目中，那么你可以运行一个命令:

```py
$ poetry add requests
```

通过运行`poetry add requests`，您将最新版本的`requests`库添加到您的项目中。如果你想更具体一些，你可以使用像`requests<=2.1`或`requests==2.24`这样的版本约束。当你不添加任何约束的时候，诗总会尝试安装最新版本的包。

有时，有些包您只想在您的开发环境中使用。通过`pytest`，你已经发现了其中一个。另一个公共库包括像 [Black](https://black.readthedocs.io) 这样的代码格式化程序，像 [Sphinx](https://www.sphinx-doc.org/en/master/) 这样的文档生成器，以及像 [Pylint](https://pylint.org/) 、 [Flake8](https://flake8.pycqa.org/en/latest/) 、 [mypy](http://mypy-lang.org/) 或 [coverage.py](https://coverage.readthedocs.io/en/6.1.2/) 这样的静态分析工具。

为了明确地告诉 poem 一个包是一个开发依赖，您可以运行带有`--dev`选项的`poetry add`。也可以使用简写的`-D`选项，和`--dev`一样:

```py
$ poetry add black -D
```

您添加了`requests`作为项目依赖项，添加了`black`作为开发依赖项。诗歌在后台为你做了几件事。首先，它将您声明的依赖项添加到了`pyproject.toml`文件中:

```py
# rp_poetry/pyproject.toml (Excerpt) [tool.poetry.dependencies] python  =  "^3.9" requests  =  "^2.26.0"  
[tool.poetry.dev-dependencies] pytest  =  "^5.2" black  =  "^21.9b0"
```

poems 将`requests`包作为项目依赖添加到`tool.poetry.dependencies`表中，同时将`black`作为开发依赖添加到`tool.poetry.dev-dependencies`表中。

区分项目依赖和开发依赖可以防止安装用户不需要运行程序的需求。开发依赖关系只与你的包的其他开发人员相关，他们想用`pytest`运行测试，并确保代码用`black`正确格式化。当用户安装你的软件包时，他们只安装`requests`。

**注意:**您可以更进一步，声明**可选依赖项**。当您想让用户选择安装一个特定的数据库适配器时，这是很方便的，这个适配器不是必需的，但是可以增强您的包。你可以在[诗歌文档](https://python-poetry.org/docs/pyproject/#extras)中了解更多关于可选依赖项的信息。

除了对`pyproject.toml`文件的修改，poems 还创建了一个名为`poetry.lock`的新文件。在这个文件中，poems 跟踪所有的包以及您在项目中使用的确切版本。

[*Remove ads*](/account/join/)

## 手柄`poetry.lock`

当您运行`poetry add`命令时，poems 会自动更新`pyproject.toml`并将解析后的版本固定在`poetry.lock`文件中。然而，你不必让诗歌做所有的工作。您可以手动将依赖项添加到`pyproject.toml`文件中，然后锁定它们。

### `poetry.lock`中的引脚依赖关系

如果你想用 Python[构建一个 web scraper，那么你可能想用](https://realpython.com/beautiful-soup-web-scraper-python/)[美汤](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)解析你的数据。将其添加到`pyproject.toml`文件的`tool.poetry.dependencies`表中:

```py
# rp_poetry/pyproject.toml (Excerpt) [tool.poetry.dependencies] python  =  "^3.9" requests  =  "^2.26.0" beautifulsoup4  =  "4.10.0"
```

通过添加`beautifulsoup4 = "4.10.0"`，你告诉 poem 它应该安装这个版本。当您向`pyproject.toml`文件添加需求时，它还没有安装。只要您的项目中没有`poetry.lock`文件，您就可以在手动添加依赖项后运行`poetry install`，因为 poems 会先查找一个`poetry.lock`文件。如果找不到，poems 会解析`pyproject.toml`文件中列出的依赖项。

一旦出现一个`poetry.lock`文件，poems 就会依赖这个文件来安装依赖项。只运行`poetry install`会触发两个文件不同步的警告，并且会产生一个错误，因为 poem 还不知道项目中有任何`beautifulsoup4`版本。

要将手动添加的依赖项从您的`pyproject.toml`文件固定到`poetry.lock`，您必须首先运行`poetry lock`命令:

```py
$ poetry lock
Updating dependencies
Resolving dependencies... (1.5s)

Writing lock file
```

通过运行`poetry lock`，poems 处理你的`pyproject.toml`文件中的所有依赖项，并将它们锁定到`poetry.lock`文件中。而诗歌并不止于此。当你运行`poetry lock`时，诗歌也递归地遍历并锁定你的直接依赖的所有依赖。

**注意:**`poetry lock`命令还会更新您现有的依赖项，如果符合您的版本约束的新版本可用的话。如果您不想更新任何已经在`poetry.lock`文件中的依赖项，那么您必须将`--no-update`选项添加到`poetry lock`命令中:

```py
$ poetry lock --no-update
Resolving dependencies... (0.1s)
```

在这种情况下，poem 只解析新的依赖项，而不影响`poetry.lock`文件中任何现有的依赖项版本。

既然您已经锁定了所有的依赖项，那么是时候安装它们了，这样您就可以在您的项目中使用它们了。

### 从`poetry.lock`安装依赖项

如果您遵循了上一节中的步骤，那么您已经通过使用`poetry add`命令安装了`pytest`和`black`。你也锁定了`beautifulsoup4`，但是还没有安装美汤。为了验证`beautifulsoup4`还没有安装，用`poetry run`命令打开 **Python 解释器**:

```py
$ poetry run python3
```

执行`poetry run python3`将在诗歌环境中打开一个交互式 [REPL](https://realpython.com/interacting-with-python/) 会话。首先尝试导入`requests`。这应该可以完美地工作。然后尝试导入`bs4`，这是美汤的模块名。这应该会抛出一个错误，因为 Beautiful Soup 还没有安装:

>>>

```py
>>> import requests
>>> import bs4
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'bs4'
```

不出所料，可以毫不费力地导入`requests`，模块`bs4`找不到了。通过键入`exit()`并点击 `Enter` ，退出交互式 Python 解释器。

在使用`poetry lock`命令锁定依赖项之后，您必须运行`poetry install`命令，以便您可以在您的项目中实际使用它们:

```py
$ poetry install
Installing dependencies from lock file

Package operations: 2 installs, 0 updates, 0 removals

 • Installing soupsieve (2.2.1)
 • Installing beautifulsoup4 (4.10.0)

Installing the current project: rp-poetry (0.1.0)
```

通过运行`poetry install`，poem 读取`poetry.lock`文件并安装其中声明的所有依赖项。现在，`bs4`已经准备好供您在项目中使用了。为了测试这一点，输入`poetry run python3`并将`bs4`导入 Python 解释器:

>>>

```py
>>> import bs4
>>> bs4.__version__
'4.10.0'
```

完美！这一次没有错误，并且您得到了您声明的确切版本。这意味着 Beautiful Soup 被正确地钉在您的`poetry.lock`文件中，被安装在您的项目中，并且可以使用了。要列出项目中可用的包并检查它们的细节，您可以使用`show`命令。当您使用`--help`标志运行它时，您将看到如何使用它:

```py
$ poetry show --help
```

要检查一个包，您可以使用`show`并将包名作为参数，或者您可以使用`--tree`选项将所有依赖项作为一个树列出。这将有助于您看到项目中嵌套的需求。

[*Remove ads*](/account/join/)

### 更新依赖关系

为了更新您的依赖关系，poems 根据两种情况提供了不同的选项:

1.  更新版本约束内的依赖项。
2.  更新版本约束之外的依赖项。

您可以在您的`pyproject.toml`文件中找到您的版本约束。当依赖项的新版本仍然满足您的版本约束时，您可以使用`update`命令:

```py
$ poetry update
```

`update`命令将在它们的版本约束内更新所有的包和它们的依赖项。之后，诗歌会更新你的`poetry.lock`文件。

如果要更新一个或多个特定的软件包，可以将它们作为参数列出:

```py
$ poetry update requests beautifulsoup4
```

使用这个命令，poems 将搜索满足您的`pyproject.toml`文件中列出的版本约束的`requests`的新版本和`beautifulsoup4`的新版本。然后，它将解析项目的所有依赖项，并将版本固定到您的`poetry.lock`文件中。您的`pyproject.toml`文件将保持不变，因为列出的约束仍然有效。

如果您想要更新一个版本高于在`pyproject.toml`文件中定义的版本的依赖项，您需要预先调整`pyproject.toml`文件。另一个选择是运行带有版本约束或`latest`标签的`add`命令:

```py
$ poetry add pytest@latest --dev
```

当您运行带有`latest`标签的`add`命令时，它会查找最新版本的包并更新您的`pyproject.toml`文件。在使用`add`命令时，包含`latest`标签或版本约束是至关重要的。如果没有它，您会得到一条消息，提示您项目中已经存在该包。另外，不要忘记为开发依赖项添加`--dev`标志。否则，您需要将该包添加到常规依赖项中。

添加新版本后，您必须运行您在上一节中了解到的`install`命令。只有这样，您的更新才会被锁定到`poetry.lock`文件中。

如果您不确定更新会给依赖项带来哪些基于版本的变化，您可以使用`--dry-run`标志。该标志对`update`和`add`命令都有效。它显示终端中的操作，而不执行任何操作。这样，您可以安全地发现版本变化，并决定哪种更新方案最适合您。

### 区分`pyproject.toml`和`poetry.lock`

虽然`pyproject.toml`文件中的版本要求可能不严格，但是 poems 锁定了您在`poetry.lock`文件中实际使用的版本。这就是为什么如果你正在使用 [Git](https://realpython.com/python-git-github-intro/) 你应该提交这个文件。通过在一个 **Git 库**中提供一个`poetry.lock`文件，你可以确保所有开发人员都将使用相同版本的必需包。当您遇到包含`poetry.lock`文件的存储库时，使用诗歌是个好主意。

有了`poetry.lock`，你可以确保你使用的是其他开发者正在使用的版本。如果其他开发人员没有使用诗歌，您可以将它添加到一个没有设置诗歌的现有项目中。

## 向现有项目添加诗歌

很有可能，您有一些项目不是用`poetry new`命令启动的。或者你继承了一个不是用诗歌创建的项目，但是现在你想用诗歌来进行你的依赖管理。在这种情况下，您可以向现有的 Python 项目中添加诗歌。

### 将`pyproject.toml`添加到脚本文件夹

如果您的项目只包含一些 Python 文件，那么您仍然可以添加诗歌作为未来构建的基础。在这个例子中，只有一个文件，`hello.py`:

```py
# rp-hello/hello.py

print("Hello World!")
```

这个脚本唯一做的事情就是输出字符串`"Hello World!"`。但也许这只是一个宏大项目的开始，所以你决定在你的项目中加入诗歌。您将使用`poetry init`命令，而不是之前的`poetry new`命令:

```py
$ poetry init

This command will guide you through creating your pyproject.toml config.

Package name [rp-hello]: rp-hello
Version [0.1.0]:
Description []: My Hello World Example
Author [Philipp <philipp@realpython.com>, n to skip]:
License []:
Compatible Python versions [^3.9]:

Would you like to define your main dependencies interactively? (yes/no) [yes] no
Would you like to define your development dependencies interactively? (yes/no) [yes] no
Generated file
```

`poetry init`命令将启动一个[交互会话](https://realpython.com/interacting-with-python/)来创建一个`pyproject.toml`文件。诗给你推荐了你需要设置的大部分配置，你可以按 `Enter` 来使用它们。当您没有声明任何依赖项时，poem 创建的`pyproject.toml`文件如下所示:

```py
# rp-hello/pyproject.toml [tool.poetry] name  =  "rp-hello" version  =  "0.1.0" description  =  "My Hello World Example" authors  =  ["Philipp <philipp@realpython.com>"] [tool.poetry.dependencies] python  =  "^3.9" [tool.poetry.dev-dependencies] [build-system] requires  =  ["poetry-core>=1.0.0"] build-backend  =  "poetry.core.masonry.api"
```

内容看起来类似于您在前面章节中所经历的例子。

现在，您可以使用诗歌项目提供的所有命令。有了`pyproject.toml`文件，您现在可以运行脚本:

```py
$ poetry run python3 hello.py
Creating virtualenv rp-simple-UCsI2-py3.9 in ~/Library/Caches/pypoetry/virtualenvs
Hello World!
```

因为 poem 没有找到任何可以使用的虚拟环境，所以它在执行脚本之前创建了一个新的虚拟环境。完成后，它会显示您的`Hello World!`消息，没有任何错误。这意味着你现在有一个工作的诗歌项目。

[*Remove ads*](/account/join/)

### 使用现有的`requirements.txt`文件

有时候你的项目已经有了一个`requirements.txt`文件。看看这个 [Python web scraper](https://realpython.com/beautiful-soup-web-scraper-python/) 的`requirements.txt`文件:

```py
$ cat requirements.txt
beautifulsoup4==4.9.3
certifi==2020.12.5
chardet==4.0.0
idna==2.10
requests==2.25.1
soupsieve==2.2.1
urllib3==1.26.4
```

使用 [`cat`实用程序](https://en.wikipedia.org/wiki/Cat_(Unix))，可以读取一个文件并将内容写入**标准输出**。在这种情况下，它显示了 web scraper 项目的依赖关系。一旦用`poetry init`创建了诗歌项目，就可以将`cat`实用程序与`poetry add`命令结合起来:

```py
$ poetry add `cat requirements.txt`
Creating virtualenv rp-require-0ubvZ-py3.9 in ~/Library/Caches/pypoetry/virtualenvs

Updating dependencies
Resolving dependencies... (6.2s)

Writing lock file

Package operations: 7 installs, 0 updates, 0 removals

 • Installing certifi (2020.12.5)
 • Installing chardet (4.0.0)
 • Installing idna (2.10)
 • Installing soupsieve (2.2.1)
 • Installing urllib3 (1.26.4)
 • Installing beautifulsoup4 (4.9.3)
 • Installing requests (2.25.1)
```

当一个需求文件如此简单时，使用`poetry add`和`cat`可以为您节省一些手工工作。

然而，有时`requirements.txt`文件有点复杂。在这些情况下，您可以执行一次测试运行，看看结果如何，或者手工将需求添加到`pyproject.toml`文件的`[tool.poetry.dependencies]`表中。要查看您的`pyproject.toml`的结构是否有效，您可以稍后运行`poetry check`。

### 从`poetry.lock`创建`requirements.txt`

在某些情况下，您必须有一个`requirements.txt`文件。例如，也许你想在 Heroku 上[主持你的 Django 项目。对于这种情况，诗歌提供了](https://realpython.com/django-hosting-on-heroku/) [`export`命令](https://python-poetry.org/docs/cli/#export)。如果你有一个诗歌项目，你可以从你的`poetry.lock`文件创建一个`requirements.txt`文件:

```py
$ poetry export --output requirements.txt
```

以这种方式使用`poetry export`命令创建一个`requirements.txt`文件，其中包含[散列](https://pip.pypa.io/en/stable/cli/pip_install/#hash-checking-mode)和[环境标记](https://www.python.org/dev/peps/pep-0508/#environment-markers)。这意味着你可以确保按照非常严格的要求工作，就像你的`poetry.lock`文件的内容一样。如果您还想包含您的开发依赖项，您可以将`--dev`添加到命令中。要查看所有可用选项，您可以勾选`poetry export --help`。

## 命令参考

本教程已经向您介绍了诗歌的依赖管理。在这个过程中，您已经使用了一些 poem 的命令行界面(CLI)命令:

| 诗歌命令 | 说明 |
| --- | --- |
| `$ poetry --version` | 显示您的诗歌安装版本。 |
| `$ poetry new` | 创建一个新的诗歌项目。 |
| `$ poetry init` | 向现有项目添加诗歌。 |
| `$ poetry run` | 用诗歌执行给定的命令。 |
| `$ poetry add` | 给`pyproject.toml`添加一个包并安装。 |
| `$ poetry update` | 更新项目的依赖项。 |
| `$ poetry install` | 安装依赖项。 |
| `$ poetry show` | 列出已安装的软件包。 |
| `$ poetry lock` | 将您的依赖项的最新版本固定到`poetry.lock`。 |
| `$ poetry lock --no-update` | 刷新`poetry.lock`文件，不更新任何依赖版本。 |
| `$ poetry check` | 验证`pyproject.toml`。 |
| `$ poetry config --list` | 展示诗词配置。 |
| `$ poetry env list` | 列出项目的虚拟环境。 |
| `$ poetry export` | 将`poetry.lock`导出为其他格式。 |

您可以查看[poems CLI 文档](https://python-poetry.org/docs/cli/)来了解更多关于上面的命令和 poems 提供的其他命令。您还可以运行`poetry --help`在您的终端上查看信息！

## 结论

在本教程中，您了解了如何创建新的 Python 诗歌项目，以及如何向现有项目添加诗歌。诗歌的一个关键部分是`pyproject.toml`文件。结合使用`poetry.lock`，您可以确保安装项目所需的每个包的精确版本。当您在 Git 存储库中跟踪`poetry.lock`文件时，您也要确保项目中的所有其他开发人员在他们的机器上安装了相同的依赖版本。

**在本教程中，您学习了如何:**

*   开始一个新的诗歌项目
*   向现有的项目添加诗歌
*   使用 **`pyproject.toml`** 文件
*   引脚**依赖关系**
*   安装依赖 **`poetry.lock`**
*   执行基本的**诗歌 CLI** 命令

本教程关注的是诗歌依赖管理的基础，但是诗歌也可以帮助你**构建和上传**你的包。如果您想体验一下这种能力，那么您可以阅读当[向 PyPI](https://realpython.com/pypi-publish-python-package/#poetry) 发布开源 Python 包时如何使用诗歌。**********