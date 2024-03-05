# 使用 Python 和 Typer 构建命令行待办事项应用程序

> 原文：<https://realpython.com/python-typer-cli/>

当你正在学习一门新的编程语言或试图将你的技能提升到一个新的水平时，构建一个管理你的**待办事项列表**的应用程序可能是一个有趣的项目。在本教程中，您将使用 Python 和 [Typer](https://typer.tiangolo.com/) 为命令行构建一个功能性的待办事项应用程序，这是一个相对年轻的库，几乎可以立即创建强大的命令行界面(CLI)应用程序。

有了这样一个项目，您将应用广泛的核心编程技能，同时构建一个具有真实特性和需求的真实应用程序。

**在本教程中，您将学习如何:**

*   用 Python 中的**类型器 CLI** 构建一个功能性的**待办应用程序**
*   使用 Typer 将**命令**、**参数**和**选项**添加到你的待办事项应用中
*   用 Typer 的 **`CliRunner`** 和 **pytest** 测试你的 Python 待办应用

此外，您将通过使用 Python 的`json`模块和使用 Python 的`configparser`模块管理[配置文件](https://en.wikipedia.org/wiki/Configuration_file)来练习与处理 [JSON 文件](https://realpython.com/python-json/)相关的技能。有了这些知识，您就可以马上开始创建 CLI 应用程序了。

您可以点击下面的链接并转到`source_code_final/`目录，下载该待办事项 CLI 应用程序的完整代码和所有附加资源:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-typer-cli-project-code/)使用 Python 和 Typer 为您的命令行构建一个待办事项应用程序。

## 演示

在这个循序渐进的项目中，您将构建一个[命令行界面(CLI)](https://en.wikipedia.org/wiki/Command-line_interface) 应用程序来管理待办事项列表。您的应用程序将提供一个基于 Typer 的 CLI，这是一个用于创建 CLI 应用程序的现代化通用库。

在你开始之前，看看这个演示，看看你的待办事项应用程序在本教程结束后会是什么样子。演示的第一部分展示了如何获得使用该应用程序的帮助。它还展示了如何初始化和配置应用程序。视频的其余部分演示了如何与基本功能进行交互，例如添加、删除和列出待办事项:

[https://player.vimeo.com/video/591043158](https://player.vimeo.com/video/591043158)

不错！该应用程序有一个用户友好的 CLI，允许您设置待办事项数据库。在那里，你可以使用适当的**命令**、**参数**和**选项**来添加、删除和完成待办事项。如果你遇到困难，你可以使用`--help`选项和适当的参数来寻求帮助。

你想开始这个待办事项应用程序项目吗？酷！在下一节中，您将计划如何构建项目的布局，以及您将使用什么工具来构建它。

[*Remove ads*](/account/join/)

## 项目概述

当你想启动一个新的应用程序时，你通常会首先考虑你希望这个应用程序如何工作。在本教程中，您将为命令行构建一个待办事项应用程序。您将把该应用程序称为`rptodo`。

您希望您的应用程序有一个用户友好的命令行界面，允许您的用户与应用程序交互并管理他们的待办事项列表。

首先，您希望 CLI 提供以下全局选项:

*   **`-v`** 或 **`--version`** 显示当前版本并退出应用程序。
*   **`--help`** 显示整个应用程序的全局帮助信息。

您将在许多其他 CLI 应用程序中看到这些相同的选项。提供它们是一个好主意，因为大多数使用命令行的用户希望在每个应用程序中都找到它们。

关于管理待办事项列表，您的应用程序将提供初始化应用程序、添加和删除待办事项以及管理待办事项完成状态的命令:

| 命令 | 描述 |
| --- | --- |
| `init` | 初始化应用程序的待办事项数据库 |
| `add DESCRIPTION` | 向数据库中添加新的待办事项及其说明 |
| `list` | 列出数据库中的所有待办事项 |
| `complete TODO_ID` | 通过使用待办事项的 ID 将其设置为已完成来完成待办事项 |
| `remove TODO_ID` | 使用待办事项的 ID 从数据库中删除待办事项 |
| `clear` | 通过清除数据库来删除所有待办事项 |

这些命令提供了所有你需要的功能，将你的待办事项应用程序转化为一个[最小可行产品(MVP)](https://en.wikipedia.org/wiki/Minimum_viable_product) ，这样你就可以[将它发布到 PyPI](https://realpython.com/pypi-publish-python-package/) 或者你选择的平台，并开始从你的用户那里获得反馈。

要在待办事项应用程序中提供所有这些功能，您需要完成几项任务:

1.  构建一个能够接受和处理命令、选项和参数的命令行界面
2.  选择合适的**数据类型**来表示您的待办事项
3.  实现一种方法来**持久存储**你的待办事项列表
4.  定义一种方法来**连接**用户界面和待办数据

这些任务与所谓的模型-视图-控制器设计密切相关，这是一种[架构模式](https://en.wikipedia.org/wiki/Architectural_pattern)。在这个模式中，**模型**处理数据，**视图**处理用户界面，**控制器**连接两端以使应用程序工作。

在您的应用程序和项目中使用这种模式的主要原因是提供[关注点分离(SoC)](https://en.wikipedia.org/wiki/Separation_of_concerns) ，使您代码的不同部分独立处理特定的概念。

您需要做出的下一个决定是关于您将用来处理您进一步定义的每个任务的工具和库。换句话说，你需要决定你的[软件栈](https://en.wikipedia.org/wiki/Solution_stack)。在本教程中，您将使用以下堆栈:

*   键入以构建待办事项应用程序的命令行界面
*   [命名元组](https://realpython.com/python-namedtuple/)和[字典](https://realpython.com/python-dicts/)来处理待办数据
*   Python 的 [`json`](https://docs.python.org/3/library/json.html) 模块管理持久数据存储

您还将使用 Python [标准库](https://docs.python.org/3/library/index.html)中的 [`configparser`](https://docs.python.org/3/library/configparser.html#module-configparser) 模块来处理配置文件中应用程序的初始设置。在配置文件中，您将在文件系统中存储待办事项数据库的路径。最后，您将使用 [pytest](https://realpython.com/pytest-python-testing/) 作为工具来[测试您的 CLI 应用程序](https://realpython.com/python-cli-testing/)。

## 先决条件

要完成本教程并从中获得最大收益，您应该熟悉以下主题:

*   [模型-视图-控制器模式](https://realpython.com/the-model-view-controller-mvc-paradigm-summarized-with-legos/)
*   [Command-line interfaces (CLI)](https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/)
*   [Python 类型提示，也称为类型注释](https://realpython.com/python-type-checking/)
*   [使用 pytest 进行单元测试](https://realpython.com/pytest-python-testing/)
*   [Python 中的面向对象编程](https://realpython.com/python3-object-oriented-programming/)
*   [配置文件用`configparser`](https://docs.python.org/3/library/configparser.html#module-configparser)
*   [JSON 文件与 Python 的`json`](https://realpython.com/python-json/)
*   [文件系统路径操作同`pathlib`](https://realpython.com/python-pathlib/)

就是这样！如果你已经准备好动手创建你的待办事项应用，那么你可以开始设置你的工作环境和项目布局。

[*Remove ads*](/account/join/)

## 第一步:建立待办项目

要开始编写您的待办应用程序，您需要设置一个[工作 Python 环境](https://realpython.com/effective-python-environment/)，其中包含您将在这个过程中使用的所有工具、库和依赖项。然后你需要给项目一个连贯的 Python [应用布局](https://realpython.com/python-application-layouts/)。这就是你在接下来的小节中要做的。

要下载您将在本节中创建的所有文件和项目结构，请单击下面的链接并转到`source_code_step_1/`目录:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-typer-cli-project-code/)使用 Python 和 Typer 为您的命令行构建一个待办事项应用程序。

### 设置工作环境

在本节中，您将创建一个 [Python 虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)来处理您的待办项目。为每个独立的项目使用虚拟环境是 Python 编程中的最佳实践。它允许您隔离项目的依赖关系，而不会扰乱您的系统 Python 安装或破坏使用相同工具和库的不同版本的其他项目。

**注意:**这个项目是用 [Python 3.9.5](https://realpython.com/python39-new-features/) 构建和测试的，代码应该在大于等于 3.6 的 Python 版本上工作。

要创建 Python 虚拟环境，请转到您最喜欢的工作目录，并创建一个名为`rptodo_project/`的文件夹。然后启动终端或命令行，运行以下命令:

```py
$ cd rptodo_project/
$ python -m venv ./venv
$ source venv/bin/activate
(venv) $
```

这里，首先使用`cd`进入`rptodo_project/`目录。该目录将是您项目的根目录。然后使用标准库中的 [`venv`](https://docs.python.org/3/library/venv.html#module-venv) 创建一个 Python 虚拟环境。`venv`的参数是托管虚拟环境的目录的路径。一种常见的做法是根据您的喜好将该目录命名为`venv`、`.venv`或`env`。

第三个命令激活您刚刚创建的虚拟环境。您知道环境是活动的，因为您的提示会变成类似于`(venv) $`的内容。

**注意:**要在 Windows 上创建和激活虚拟环境，您将遵循类似的过程。

继续运行以下命令:

```py
c:\> python -m venv venv
c:\> venv\Scripts\activate.bat
```

如果您在不同的平台上，那么您可能需要查看 Python 官方文档中关于[创建虚拟环境](https://docs.python.org/3/library/venv.html#creating-virtual-environments)的内容。

现在您已经有了一个工作的虚拟环境，您需要[安装 Typer](https://typer.tiangolo.com/#installation) 来创建 CLI 应用程序和 [pytest](https://realpython.com/pytest-python-testing/#how-to-install-pytest) 来测试您的应用程序的代码。要安装 Typer 及其所有当前的[可选依赖项](https://typer.tiangolo.com/#optional-dependencies)，请运行以下命令:

```py
(venv) $ python -m pip install typer==0.3.2 colorama==0.4.4 shellingham==1.4.0
```

该命令安装 Typer 及其所有推荐的依赖项，例如 [Colorama](https://pypi.org/project/colorama/) ，它确保颜色在命令行窗口中正确工作。

要安装 pytest(稍后您将使用它来测试您的待办事项应用程序),请运行以下命令:

```py
(venv) $ python -m pip install pytest==6.2.4
```

使用这最后一个命令，您成功地安装了开始开发您的待办事项应用程序所需的所有工具。您将使用的其余库和工具是 Python 标准库的一部分，因此您不必安装任何东西就可以使用它们。

### 定义项目布局

完成待办事项应用项目设置的最后一步是创建[包、模块](https://realpython.com/python-modules-packages/)和构建应用布局的文件。该应用的核心包将位于`rptodo_project/`内的`rptodo/`目录中。

以下是对该包内容的描述:

| 文件 | 描述 |
| --- | --- |
| `__init__.py` | 使`rptodo/`成为一个 Python 包 |
| `__main__.py` | 提供一个入口点脚本，使用`python -m rptodo`命令从包中运行应用程序 |
| `cli.py` | 为应用程序提供 Typer 命令行界面 |
| `config.py` | 包含处理应用程序配置文件的代码 |
| `database.py` | 包含处理应用程序的待办事项数据库的代码 |
| `rptodo.py` | 提供将 CLI 与待办事项数据库连接起来的代码 |

您还需要一个包含一个`__init__.py`文件的`tests/`目录来将该目录转换成一个包，还需要一个`test_rptodo.py`文件来保存应用程序的[单元测试](https://realpython.com/python-testing/#unit-tests-vs-integration-tests)。

继续使用以下结构创建项目布局:

```py
rptodo_project/
│
├── rptodo/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── database.py
│   └── rptodo.py
│
├── tests/
│   ├── __init__.py
│   └── test_rptodo.py
│
├── README.md
└── requirements.txt
```

[`README.md`](https://dbader.org/blog/write-a-great-readme-for-your-github-project) 文件将提供项目的描述以及安装和运行应用程序的说明。向您的项目添加一个描述性的详细的`README.md`文件是编程中的一个最佳实践，尤其是如果您计划将该项目作为开放源代码发布的话。

`requirements.txt`文件将为您的待办应用程序提供依赖项列表。继续填写以下内容:

```py
typer==0.3.2
colorama==0.4.4
shellingham==1.4.0
pytest==6.2.4
```

现在，您的用户可以通过运行以下命令自动安装列出的依赖项:

```py
(venv) $ python -m pip install -r requirements.txt
```

像这样提供一个`requirements.txt`可以确保您的用户将安装您用来构建项目的依赖项的精确版本，避免意外的问题和行为。

除了`requirements.txt`之外，此时您的项目的所有文件都应该是空的。在本教程中，您将使用必要的内容填充每个文件。在下一节中，您将使用 Python 和 Typer 编写应用程序的 CLI。

[*Remove ads*](/account/join/)

## 第二步:用 Python 和 Typer 设置待办事项 CLI 应用

至此，您应该有了待办事项应用程序的完整项目布局。您还应该有一个工作的 Python 虚拟环境，其中包含所有必需的工具和库。在这一步结束时，您将拥有一个功能型 CLI 应用程序。然后，您将能够在其最小功能的基础上进行构建。

您可以通过点击下面的链接并转到`source_code_step_2/`目录来下载您将在本节中添加的代码、单元测试和资源:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-typer-cli-project-code/)使用 Python 和 Typer 为您的命令行构建一个待办事项应用程序。

启动代码编辑器，从`rptodo/`目录中打开`__init__.py`文件。然后向其中添加以下代码:

```py
"""Top-level package for RP To-Do."""
# rptodo/__init__.py

__app_name__ = "rptodo"
__version__ = "0.1.0"

(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    DB_READ_ERROR,
    DB_WRITE_ERROR,
    JSON_ERROR,
    ID_ERROR,
) = range(7)

ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    DB_READ_ERROR: "database read error",
    DB_WRITE_ERROR: "database write error",
    ID_ERROR: "to-do id error",
}
```

这里，首先定义两个[模块级名称](https://www.python.org/dev/peps/pep-0008/#module-level-dunder-names)来保存应用程序的名称和版本。然后定义一系列的[返回和错误代码](https://en.wikipedia.org/wiki/Error_code)，并使用 [`range()`](https://realpython.com/python-range/) 给它们分配整数。`ERROR`是一个[字典](https://realpython.com/python-dicts/)，它将错误代码映射到人类可读的错误消息。您将使用这些消息告诉用户应用程序正在发生什么。

有了这些代码，就可以创建 Typer CLI 应用程序的框架了。这就是你在下一节要做的。

### 创建 Typer CLI 应用程序

在这一节中，您将创建一个支持`--help`、`-v`和`--version`选项的最小 Typer CLI 应用程序。为此，您将使用一个[显式类型应用程序](https://typer.tiangolo.com/tutorial/commands/#explicit-application)。这种类型的应用程序适用于包含[多个命令](https://typer.tiangolo.com/tutorial/commands/#a-cli-application-with-multiple-commands)和几个[选项](https://typer.tiangolo.com/tutorial/options/)和[参数](https://typer.tiangolo.com/tutorial/arguments/)的大型项目。

继续在文本编辑器中打开`rptodo/cli.py`,输入以下代码:

```py
 1"""This module provides the RP To-Do CLI."""
 2# rptodo/cli.py
 3
 4from typing import Optional
 5
 6import typer
 7
 8from rptodo import __app_name__, __version__
 9
10app = typer.Typer()
11
12def _version_callback(value: bool) -> None:
13    if value:
14        typer.echo(f"{__app_name__} v{__version__}")
15        raise typer.Exit()
16
17@app.callback()
18def main(
19    version: Optional[bool] = typer.Option(
20        None,
21        "--version",
22        "-v",
23        help="Show the application's version and exit.",
24        callback=_version_callback,
25        is_eager=True,
26    )
27) -> None:
28    return
```

Typer 广泛使用 Python 类型提示，因此在本教程中，您也将使用它们。这就是为什么你从 [`typing`](https://docs.python.org/3/library/typing.html#module-typing) 导入 [`Optional`](https://docs.python.org/3/library/typing.html?highlight=typing#typing.Optional) 开始。接下来，你进口`typer`。最后，你从你的`rptodo`包中导入`__app_name__`和`__version__`。

下面是其余代码的工作方式:

*   **第 10 行**创建了一个显式类型应用程序`app`。

*   **第 12 到 15 行**定义了`_version_callback()`。这个函数采用一个名为`value`的[布尔](https://realpython.com/python-boolean/)参数。如果`value`是`True`，那么该函数使用 [`echo()`](https://typer.tiangolo.com/tutorial/printing/) 打印应用程序的名称和版本。之后，它引发一个 [`typer.Exit`](https://typer.tiangolo.com/tutorial/terminating/#exit-a-cli-program) 异常来干净地退出应用程序。

*   **第 17 行和第 18 行**使用`@app.callback()`装饰器将 [`main()`](https://realpython.com/python-main-function/) 定义为[类型回调](https://typer.tiangolo.com/tutorial/commands/callback/)。

*   **第 19 行**定义了`version`，其类型为`Optional[bool]`。这意味着它可以是 [`bool`](https://realpython.com/python-boolean/) 或 [`None`](https://realpython.com/null-in-python/) 类型。`version`参数默认为一个`typer.Option`对象，它允许您在 Typer 中创建命令行选项。

*   **第 20 行**将`None`作为第一个参数传递给`Option`的初始化器。此参数是必需的，并提供选项的默认值。

*   **第 21 行和第 22 行**为`version`选项设置命令行名称:`-v`和`--version`。

*   **第 23 行**为`version`选项提供了一条`help`消息。

*   **第 24 行**将一个回调函数`_version_callback()`附加到`version`选项上，这意味着运行该选项会自动调用该函数。

*   **第 25 行**将`is_eager`参数设置为`True`。这个参数告诉 Typer[`version`](https://typer.tiangolo.com/tutorial/options/version/)命令行选项优先于当前应用程序中的其他命令。

有了这些代码，就可以创建应用程序的入口点脚本了。这就是你在下一节要做的。

### 创建一个入口点脚本

您几乎已经准备好第一次运行您的待办事项应用程序了。在此之前，您应该为应用程序创建一个入口点脚本。您可以用几种不同的方式创建这个脚本。在本教程中，您将使用`rptodo`包中的 [`__main__.py`](https://docs.python.org/3/library/__main__.html#module-__main__) 模块来完成。在 Python 包中包含一个`__main__.py`模块使您能够使用命令`python -m rptodo`将包作为可执行程序运行。

回到代码编辑器，从`rptodo/`目录中打开`__main__.py`。然后添加以下代码:

```py
"""RP To-Do entry point script."""
# rptodo/__main__.py

from rptodo import cli, __app_name__

def main():
    cli.app(prog_name=__app_name__)

if __name__ == "__main__":
    main()
```

在`__main__.py`中，你首先从`rptodo`导入`cli`和`__app_name__`。然后你定义`main()`。在这个函数中，您用`cli.app()`调用 Typer 应用程序，将应用程序的名称传递给`prog_name`参数。向`prog_name`提供一个值可以确保用户在命令行上运行`--help`选项时获得正确的应用程序名称。

有了这最后一项，您就可以第一次运行您的待办事项应用程序了。转到您的终端窗口，执行以下命令:

```py
(venv) $ python -m rptodo -v
rptodo v0.1.0

(venv) $ python -m rptodo --help
Usage: rptodo [OPTIONS] COMMAND [ARGS]...

Options:
 -v, --version         Show the application's version and exit.
 --install-completion  Install completion for the current shell.
 --show-completion     Show completion for the current shell, to copy it
 or customize the installation.

 --help                Show this message and exit.
```

第一个命令运行`-v`选项，显示应用程序的版本。第二个命令运行`--help`选项，为整个应用程序显示用户友好的帮助消息。Typer 会自动为您生成并显示此帮助消息。

[*Remove ads*](/account/join/)

### 使用 pytest 设置初始 CLI 测试

在本节中，您将运行的最后一个操作是为您的待办应用程序设置一个初始的[测试套件](https://en.wikipedia.org/wiki/Test_suite)。为此，您已经用一个名为`test_rptodo.py`的模块创建了`tests`包。正如您在前面所学的，您将使用 pytest 来编写和运行您的单元测试。

[测试一个 Typer 应用程序很简单，因为这个库与 pytest 集成得很好。您可以使用一个名为](https://typer.tiangolo.com/tutorial/testing/) [`CliRunner`](https://click.palletsprojects.com/en/8.0.x/api/#click.testing.CliRunner) 的 Typer 类来测试应用程序的 CLI。`CliRunner`允许您创建一个运行程序，用于测试您的应用程序的 CLI 如何响应实际命令。

回到代码编辑器，从`tests/`目录中打开`test_rptodo.py`。键入以下代码:

```py
 1# tests/test_rptodo.py
 2
 3from typer.testing import CliRunner
 4
 5from rptodo import __app_name__, __version__, cli
 6
 7runner = CliRunner()
 8
 9def test_version():
10    result = runner.invoke(cli.app, ["--version"])
11    assert result.exit_code == 0
12    assert f"{__app_name__} v{__version__}\n" in result.stdout
```

下面是这段代码的作用:

*   **三号线**从`typer.testing`进口`CliRunner`。
*   **第 5 行**从你的`rptodo`包中导入一些需要的对象。
*   **第 7 行**通过实例化`CliRunner`创建一个 CLI 运行器。
*   第 9 行定义了测试应用程序版本的第一个单元测试。
*   **第 10 行**调用`runner`上的`.invoke()`来运行带有`--version`选项的应用程序。您将这次调用的结果存储在`result`中。
*   **第 11 行**断言应用程序的[退出代码](https://en.wikipedia.org/wiki/Exit_status) ( `result.exit_code`)等于`0`，以检查应用程序是否成功运行。
*   **第 12 行**断言应用程序的版本出现在标准输出中，可通过`result.stdout`获得。

Typer 的`CliRunner`是 [Click 的`CliRunner`](https://click.palletsprojects.com/en/8.0.x/api/?highlight=clirunner#click.testing.CliRunner) 的子类。因此，它的`.invoke()`方法返回一个 [`Result`](https://click.palletsprojects.com/en/8.0.x/api/#click.testing.Result) 对象，该对象保存使用目标参数和选项运行 CLI 应用程序的结果。`Result`对象提供了几个有用的属性和[特性](https://realpython.com/python-property/)，包括应用程序的退出代码和标准输出。更多细节请看一下类文档。

现在，您已经为 Typer CLI 应用程序设置了第一个单元测试，您可以使用 pytest 运行测试。回到命令行，从项目的根目录执行`python -m pytest tests/`:

```py
========================= test session starts =========================
platform linux -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: .../rptodo
plugins: Faker-8.1.1, cov-2.12.0, celery-4.4.7
collected 1 item

tests/test_rptodo.py .                                          [100%]
========================== 1 passed in 0.07s ==========================
```

就是这样！您第一次成功运行了您的测试套件！是的，到目前为止你只有一个测试。但是，您将在接下来的章节中添加更多的内容。如果你想挑战你的测试技巧，你也可以添加你自己的测试。

有了 to-do 应用程序的框架，现在您可以考虑设置 to-do 数据库以准备使用。这就是你在下一节要做的。

## 步骤 3:准备待办事项数据库以供使用

到目前为止，您已经为您的待办事项应用程序构建了一个 CLI，创建了一个入口点脚本，并且第一次运行了该应用程序。您还为应用程序设置并运行了一个最小的测试套件。下一步是定义应用程序如何初始化并连接到待办事项数据库。

您将使用一个 [JSON](https://www.json.org/json-en.html) 文件来存储关于您的待办事项的数据。JSON 是一种轻量级的数据交换格式，可读可写。Python 的标准库包括`json`，这是一个提供开箱即用的 JSON 文件格式支持的模块。这就是你要用来管理你的待办事项数据库。

您可以通过点击下面的链接并转到`source_code_step_3/`目录来下载本节的完整代码:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-typer-cli-project-code/)使用 Python 和 Typer 为您的命令行构建一个待办事项应用程序。

在本节结束时，您已经编写了创建、连接和初始化待办事项数据库的代码，这样它就可以使用了。然而，第一步是定义应用程序如何在文件系统中找到待办事项数据库。

### 设置应用程序的配置

您可以使用不同的技术来定义应用程序如何连接以及如何在您的文件系统上打开文件。您可以动态地提供文件路径，创建一个环境变量来保存文件路径，创建一个用于存储文件路径的[配置文件](https://en.wikipedia.org/wiki/Configuration_file)，等等。

**注:**配置文件，也称为**配置文件**，是程序员用来为给定程序或应用提供初始参数和设置的一种文件。

在本教程中，您将在个人目录中为待办事项应用程序提供一个配置文件来存储数据库的路径。为此，您将使用 [`pathlib`](https://realpython.com/python-pathlib/) 处理文件系统路径，使用`configparser`处理配置文件。这两个包都可以在 Python 标准库中找到。

现在回到你的代码编辑器，从`rptodo/`打开`config.py`。键入以下代码:

```py
 1"""This module provides the RP To-Do config functionality."""
 2# rptodo/config.py
 3
 4import configparser
 5from pathlib import Path
 6
 7import typer
 8
 9from rptodo import (
10    DB_WRITE_ERROR, DIR_ERROR, FILE_ERROR, SUCCESS, __app_name_
11)
12
13CONFIG_DIR_PATH = Path(typer.get_app_dir(__app_name__))
14CONFIG_FILE_PATH = CONFIG_DIR_PATH / "config.ini"
15
16def init_app(db_path: str) -> int:
17    """Initialize the application."""
18    config_code = _init_config_file()
19    if config_code != SUCCESS:
20        return config_code
21    database_code = _create_database(db_path)
22    if database_code != SUCCESS:
23        return database_code
24    return SUCCESS
25
26def _init_config_file() -> int:
27    try:
28        CONFIG_DIR_PATH.mkdir(exist_ok=True)
29    except OSError:
30        return DIR_ERROR
31    try:
32        CONFIG_FILE_PATH.touch(exist_ok=True)
33    except OSError:
34        return FILE_ERROR
35    return SUCCESS
36
37def _create_database(db_path: str) -> int:
38    config_parser = configparser.ConfigParser()
39    config_parser["General"] = {"database": db_path}
40    try:
41        with CONFIG_FILE_PATH.open("w") as file:
42            config_parser.write(file)
43    except OSError:
44        return DB_WRITE_ERROR
45    return SUCCESS
```

下面是这段代码的详细内容:

*   **四号线**进口`configparser`。这个模块提供了 [`ConfigParser`](https://docs.python.org/3/library/configparser.html#configparser.ConfigParser) 类，允许你处理结构类似于 [INI 文件](https://en.wikipedia.org/wiki/INI_file)的配置文件。

*   **5 号线**从`pathlib`进口 [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) 。这个类提供了一种跨平台的方式来处理系统路径。

*   **7 号线**进口`typer`。

*   **第 9 到 11 行**从`rptodo`导入一堆需要的对象。

*   **第 13 行**创建`CONFIG_DIR_PATH`来保存 [app 的目录路径](https://typer.tiangolo.com/tutorial/app-dir/)。为了获得这个路径，您调用`get_app_dir()`,将应用程序的名称作为参数。此函数返回一个字符串，表示存储配置的目录的路径。

*   **第 14 行**定义`CONFIG_FILE_PATH`来保存配置文件本身的路径。

*   **第 16 行**定义`init_app()`。这个函数初始化应用程序的配置文件和数据库。

*   第 18 行调用第 26 到 35 行定义的`_init_config_file()`助手函数。调用此函数使用 [`Path.mkdir()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir) 创建配置目录。它还使用 [`Path.touch()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.touch) 创建配置文件。最后，如果在创建目录和文件的过程中发生了错误，`_init_config_file()`会返回正确的错误代码。如果一切顺利，它将返回`SUCCESS`。

*   **第 19 行**检查在创建目录和配置文件的过程中是否出现错误，第 20 行相应地返回错误代码。

*   **第 21 行**调用`_create_database()`助手函数，创建待办事项数据库。如果在创建数据库时发生了什么，这个函数将返回相应的错误代码。如果流程成功，它将返回`SUCCESS`。

*   **第 22 行**检查数据库创建过程中是否出现错误。如果是，那么第 23 行返回相应的错误代码。

*   如果一切运行正常，第 24 行返回`SUCCESS`。

使用这段代码，您已经完成了设置应用程序的配置文件来存储 to-do 数据库的路径。您还添加了代码来将待办事项数据库创建为 JSON 文件。现在，您可以编写代码来初始化数据库并准备好使用它。这就是你在下一节要做的。

[*Remove ads*](/account/join/)

### 准备好待办事项数据库

要准备好待办事项数据库，您需要执行两个操作。首先，您需要一种从应用程序的配置文件中检索数据库文件路径的方法。其次，需要初始化数据库来保存 JSON 内容。

在您的代码编辑器中从`rptodo/`打开`database.py`，并编写以下代码:

```py
 1"""This module provides the RP To-Do database functionality."""
 2# rptodo/database.py
 3
 4import configparser
 5from pathlib import Path
 6
 7from rptodo import DB_WRITE_ERROR, SUCCESS
 8
 9DEFAULT_DB_FILE_PATH = Path.home().joinpath(
10    "." + Path.home().stem + "_todo.json"
11)
12
13def get_database_path(config_file: Path) -> Path:
14    """Return the current path to the to-do database."""
15    config_parser = configparser.ConfigParser()
16    config_parser.read(config_file)
17    return Path(config_parser["General"]["database"])
18
19def init_database(db_path: Path) -> int:
20    """Create the to-do database."""
21    try:
22        db_path.write_text("[]")  # Empty to-do list
23        return SUCCESS
24    except OSError:
25        return DB_WRITE_ERROR
```

在这个文件中，第 4 行到第 7 行执行所需的导入。下面是代码的其余部分:

*   **第 9 到 11 行**定义`DEFAULT_DB_FILE_PATH`来保存默认的数据库文件路径。如果用户没有提供自定义路径，应用程序将使用该路径。

*   **第 13 到 17 行**定义了`get_database_path()`。该函数将应用程序配置文件的路径作为参数，使用 [`ConfigParser.read()`](https://docs.python.org/3/library/configparser.html#configparser.ConfigParser.read) 读取输入文件，并返回一个`Path`对象，表示文件系统上待办事项数据库的路径。`ConfigParser`实例将数据存储在一个字典中。`"General"`键代表存储所需信息的文件部分。`"database"`键检索数据库路径。

*   **第 19 到 25 行**定义`init_database()`。这个函数获取一个数据库路径，并写入一个表示空列表的字符串。你在数据库路径上调用 [`.write_text()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.write_text) ，列表用一个空的待办列表初始化 JSON 数据库。如果流程运行成功，那么`init_database()`返回`SUCCESS`。否则，它返回适当的错误代码。

酷！现在，您有了从应用程序的配置文件中检索数据库文件路径的方法。您还可以用 JSON 格式的空待办事项列表来初始化数据库。是时候用 Typer 实现`init`命令了，这样用户就可以从 CLI 初始化他们的待办事项数据库。

### 执行`init` CLI 命令

将本节中编写的所有代码放在一起的最后一步是将`init`命令添加到应用程序的 CLI 中。该命令将采用可选的数据库文件路径。然后它会创建应用程序的配置文件和待办事项数据库。

继续将`init()`添加到您的`cli.py`文件中:

```py
 1"""This module provides the RP To-Do CLI."""
 2# rptodo/cli.py
 3
 4from pathlib import Path 5from typing import Optional
 6
 7import typer
 8
 9from rptodo import ERRORS, __app_name__, __version__, config, database 10
11app = typer.Typer()
12
13@app.command() 14def init( 15    db_path: str = typer.Option(
16        str(database.DEFAULT_DB_FILE_PATH),
17        "--db-path",
18        "-db",
19        prompt="to-do database location?",
20    ),
21) -> None:
22    """Initialize the to-do database."""
23    app_init_error = config.init_app(db_path)
24    if app_init_error:
25        typer.secho(
26            f'Creating config file failed with "{ERRORS[app_init_error]}"',
27            fg=typer.colors.RED,
28        )
29        raise typer.Exit(1)
30    db_init_error = database.init_database(Path(db_path))
31    if db_init_error:
32        typer.secho(
33            f'Creating database failed with "{ERRORS[db_init_error]}"',
34            fg=typer.colors.RED,
35        )
36        raise typer.Exit(1)
37    else:
38        typer.secho(f"The to-do database is {db_path}", fg=typer.colors.GREEN)
39
40def _version_callback(value: bool) -> None:
41    # ...
```

下面是新代码的工作原理:

*   **第 4 行和第 9 行**更新所需的导入。

*   **第 13 行和第 14 行**使用`@app.command()`装饰器将`init()`定义为一个键入命令。

*   **第 15 到 20 行**定义了一个 Typer `Option`实例，并将其作为默认值赋给`db_path`。要为该选项提供一个值，您的用户需要使用`--db-path`或`-db`，后跟一个数据库路径。`prompt`参数显示一个询问数据库位置的提示。它还允许您通过按下 `Enter` 来接受默认路径。

*   **第 23 行**调用`init_app()`创建应用程序的配置文件和待办事项数据库。

*   **第 24 到 29 行**检查对`init_app()`的调用是否返回错误。如果是这样，第 25 到 28 行打印一条错误消息。第 29 行用一个`typer.Exit`异常和一个退出代码`1`退出应用程序，表示应用程序因出错而终止。

*   **第 30 行**调用`init_database()`用一个空的待办事项列表初始化数据库。

*   **第 31 到 38 行**检查对`init_database()`的调用是否返回错误。如果是，那么第 32 到 35 行显示一条错误消息，第 36 行退出应用程序。否则，第 38 行用绿色文本打印一条成功消息。

使用 [`typer.secho()`](https://typer.tiangolo.com/tutorial/printing/#typersecho-style-and-print) 打印该代码中的信息。这个函数有一个前景参数`fg`，当[将](https://typer.tiangolo.com/tutorial/printing/)文本打印到屏幕上时，它允许你使用不同的颜色。Typer 在`typer.colors`中提供了几种内置颜色。在那里你会发现`RED`、`BLUE`、`GREEN`等等。你可以像这里一样用`secho()`使用这些颜色。

**注意:**本教程中代码示例中的行号是出于解释的目的。大多数情况下，它们不会与最终模块或脚本中的行号相匹配。

不错！有了所有这些代码，现在可以尝试一下`init`命令了。回到您的终端，运行以下命令:

```py
(venv) $ python -m rptodo init
to-do database location? [/home/user/.user_todo.json]:
The to-do database is /home/user/.user_todo.json
```

该命令提示您输入数据库位置。可以按 `Enter` 接受方括号内的默认路径，也可以输入自定义路径后再按 `Enter` 。该应用程序创建了待办事项数据库，并告诉您从现在开始它将驻留在哪里。

或者，您可以通过使用带有`-db`或`--db-path`选项的`init`来直接提供一个定制的数据库路径，后跟所需的路径。在所有情况下，您的自定义路径都应该包括数据库文件名。

一旦你运行了上面的命令，看看你的主目录。您将拥有一个 JSON 文件，该文件以您在`init`中使用的文件名命名。在您的主文件夹中，您还会有一个包含一个`config.ini`文件的`rptodo/`目录。该文件的具体路径取决于您当前的操作系统。比如在 Ubuntu 上，文件会在`/home/user/.config/rptodo/`。

[*Remove ads*](/account/join/)

## 第四步:设置待办 App 后端

到目前为止，您已经找到了创建、初始化和连接 to-do 数据库的方法。现在您可以开始考虑您的数据模型了。换句话说，你需要考虑如何表示和存储关于你的待办事项的数据。您还需要定义应用程序将如何处理 CLI 和数据库之间的通信。

您可以通过点击下面的链接并转到`source_code_step_4/`目录来下载代码和您将在本节中使用的所有其他资源:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-typer-cli-project-code/)使用 Python 和 Typer 为您的命令行构建一个待办事项应用程序。

### 定义一个单独的待办事项

首先，考虑定义一个待办事项所需的数据。在这个项目中，待办事项将由以下信息组成:

*   **描述:**如何描述这个待办事项？
*   优先级:这个待办事项比你的其他待办事项优先级高多少？
*   **完成:**这个待办事项完成了吗？

要存储这些信息，可以使用常规的 Python 字典:

```py
todo = {
    "Description": "Get some milk.",
    "Priority": 2,
    "Done": True,
}
```

`"Description"`键存储描述当前待办事项的字符串。`"Priority"`键可以有三个可能的值:`1`表示高优先级，`2`表示中优先级，`3`表示低优先级。当您完成待办事项时，`"Done"`键会按住`True`，否则会按住`False`。

### 与 CLI 通信

为了与 CLI 通信，您将使用两个包含所需信息的数据:

1.  **`todo`** :保存当前待办事项信息的字典
2.  **`error`** :确认当前操作是否成功的返回或错误代码

为了存储这些数据，您将使用一个名为 tuple 的[,并带有适当命名的字段。从`rptodo`打开`rptodo.py`模块，创建所需的命名元组:](https://realpython.com/python-namedtuple/)

```py
 1"""This module provides the RP To-Do model-controller."""
 2# rptodo/rptodo.py
 3
 4from typing import Any, Dict, NamedTuple
 5
 6class CurrentTodo(NamedTuple):
 7    todo: Dict[str, Any]
 8    error: int
```

在`rptodo.py`中，首先从`typing`导入一些需要的对象。在第 6 行，您创建了一个名为`CurrentTodo`的 [`typing.NamedTuple`的子类，它有两个字段`todo`和`error`。](https://realpython.com/python-namedtuple/#namedtuple-vs-typingnamedtuple)

子类化`NamedTuple`允许您为命名字段创建带有类型提示的命名元组。例如，上面的`todo`字段保存了一个字典，其中键的类型为`str`，值的类型为`Any`。`error`字段保存一个 [`int`](https://realpython.com/python-numbers/#integers) 值。

### 与数据库通信

现在，您需要另一个数据容器，它允许您向待办事项数据库发送数据和从中检索数据。在这种情况下，您将使用具有以下字段的另一个命名元组:

1.  **`todo_list`** :你将从数据库中写入和读取的待办事项列表
2.  **`error`** :表示当前数据库操作相关的返回码的整数

最后，您将创建一个名为`DatabaseHandler`的类来读写 to-do 数据库中的数据。继续打开`database.py`。一旦你到了那里，输入以下代码:

```py
 1# rptodo/database.py
 2
 3import configparser
 4import json 5from pathlib import Path
 6from typing import Any, Dict, List, NamedTuple 7
 8from rptodo import DB_READ_ERROR, DB_WRITE_ERROR, JSON_ERROR, SUCCESS 9
10# ...
11
12class DBResponse(NamedTuple): 13    todo_list: List[Dict[str, Any]]
14    error: int
15
16class DatabaseHandler: 17    def __init__(self, db_path: Path) -> None:
18        self._db_path = db_path
19
20    def read_todos(self) -> DBResponse:
21        try:
22            with self._db_path.open("r") as db:
23                try:
24                    return DBResponse(json.load(db), SUCCESS)
25                except json.JSONDecodeError:  # Catch wrong JSON format
26                    return DBResponse([], JSON_ERROR)
27        except OSError:  # Catch file IO problems
28            return DBResponse([], DB_READ_ERROR)
29
30    def write_todos(self, todo_list: List[Dict[str, Any]]) -> DBResponse:
31        try:
32            with self._db_path.open("w") as db:
33                json.dump(todo_list, db, indent=4)
34            return DBResponse(todo_list, SUCCESS)
35        except OSError:  # Catch file IO problems
36            return DBResponse(todo_list, DB_WRITE_ERROR)
```

下面是这段代码的作用:

*   **第 4、6 和 8 行**添加了一些必需的导入。

*   **第 12 到 14 行**将`DBResponse`定义为一个`NamedTuple`子类。`todo_list`字段是代表单个待办事项的字典列表，而`error`字段保存一个整数返回代码。

*   **第 16 行**定义了`DatabaseHandler`，它允许你使用标准库中的`json`模块向待办数据库读写数据。

*   **第 17 行和第 18 行**定义了类初始化器，它接受一个表示文件系统上数据库路径的参数。

*   **第 20 行**定义`.read_todos()`。这个方法从数据库中读取待办事项列表，[反序列化](https://en.wikipedia.org/wiki/Serialization)它。

*   **第 21 行**开始一个`try` … `except`语句来捕捉你打开数据库时发生的任何错误。如果出现错误，那么第 28 行返回一个带有空待办事项列表和一个`DB_READ_ERROR`的`DBResponse`实例。

*   **第 22 行**使用 [`with`语句](https://realpython.com/python-with-statement/)打开数据库进行读取。

*   **第 23 行**开始另一个`try` … `except`语句，捕捉从待办数据库加载和反序列化 JSON 内容时发生的任何错误。

*   **第 24 行**返回一个`DBResponse`实例，保存调用`json.load()`的结果，以待办数据库对象作为参数。这个结果由一个字典列表组成。每本词典都代表一项任务。`DBResponse`的`error`字段按住`SUCCESS`表示操作成功。

*   **第 25 行**在从数据库加载 JSON 内容时捕获任何`JSONDecodeError`，第 26 行返回一个空列表和一个`JSON_ERROR`。

*   **第 27 行**在加载 JSON 文件时捕获任何文件 IO 问题，第 28 行返回一个带有空待办事项列表和`DB_READ_ERROR`的`DBResponse`实例。

*   **第 30 行**定义了`.write_todos()`，它获取待办字典列表并将其写入数据库。

*   **第 31 行**开始一个`try` … `except`语句来捕捉你打开数据库时发生的任何错误。如果出现错误，那么第 36 行返回一个带有原始待办事项列表和一个`DB_READ_ERROR`的`DBResponse`实例。

*   **第 32 行**使用一个`with`语句打开数据库进行写操作。

*   第 33 行将待办事项列表作为 JSON 负载转储到数据库中。

*   **第 34 行**返回一个保存待办事项列表和`SUCCESS`代码的`DBResponse`实例。

哇！太多了！既然您已经完成了编码`DatabaseHandler`并设置了数据交换机制，那么您可以考虑如何将它们连接到应用程序的 CLI。

[*Remove ads*](/account/join/)

### 写控制器类，`Todoer`，

为了将`DatabaseHandler`逻辑与应用程序的 CLI 连接起来，您将编写一个名为`Todoer`的类。这个类的工作方式类似于模型-视图-控制器模式中的控制器。

现在回到`rptodo.py`并添加以下代码:

```py
# rptodo/rptodo.py
from pathlib import Path from typing import Any, Dict, NamedTuple

from rptodo.database import DatabaseHandler 
# ...

class Todoer:
    def __init__(self, db_path: Path) -> None:
        self._db_handler = DatabaseHandler(db_path)
```

这段代码包括一些导入和`Todoer`的定义。这个类使用了[组合](https://realpython.com/inheritance-composition-python/#whats-composition)，所以它有一个`DatabaseHandler`组件来促进与待办事项数据库的直接通信。在接下来的部分中，您将向该类添加更多的代码。

在这一节中，你已经完成了许多设置，这些设置决定了你的待办事项应用程序的后端将如何工作。您已经决定了使用什么数据结构来存储待办事项数据。您还定义了将使用哪种数据库来保存待办事项信息，以及如何对其进行操作。

所有这些设置就绪后，您就可以开始通过允许用户填充他们的待办事项列表来为他们提供价值了。您还将实现一种在屏幕上显示待办事项的方法。

## 步骤 5:编写添加和列出待办功能的代码

在本节中，您将编写待办事项应用程序的一个主要特性。您将为您的用户提供一个命令，将新的待办事项添加到他们当前的列表中。您还可以允许用户在屏幕上以表格形式列出他们的待办事项。

在使用这些特性之前，您将为您的代码设置一个最小的测试套件。在写代码之前写一个测试套件会帮助你理解[测试驱动开发(TDD)](https://en.wikipedia.org/wiki/Test-driven_development) 是关于什么的。

要下载代码、单元测试和您将在本节中添加的所有附加资源，只需点击下面的链接并转到`source_code_step_5/`目录:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-typer-cli-project-code/)使用 Python 和 Typer 为您的命令行构建一个待办事项应用程序。

### 为`Todoer.add()`定义单元测试

在本节中，您将使用 pytest 为`Todoer.add()`编写并运行一个最小的测试套件。这种方法会将新的待办事项添加到数据库中。测试套件就绪后，您将编写通过测试所需的代码，这是 TDD 背后的一个基本思想。

**注意:**如果您下载了本教程每一节的源代码和资源，那么您会发现本节和接下来的几节中有额外的单元测试。

看一看它们，试着理解它们的逻辑。运行它们以确保应用程序正常工作。扩展它们以添加新的测试用例。在这个过程中你会学到很多东西。

在为`.add()`编写测试之前，想想这个方法需要做什么:

1.  获取待办事项**描述**和**优先级**
2.  创建一个字典来保存**待办信息**
3.  从**数据库**中读取待办事项列表
4.  将**新待办事项**追加到当前待办事项列表中
5.  将**更新后的待办事项列表**写回数据库
6.  将新添加的**待办事项**连同返回码一起返回给调用者

代码测试中的一个常见实践是从给定方法或函数的主要功能开始。您将通过创建测试用例来检查`.add()`是否正确地向数据库添加了新的待办事项。

为了测试`.add()`，您必须创建一个`Todoer`实例，用一个合适的 JSON 文件作为目标数据库。为了提供该文件，您将使用 pytest [夹具](https://realpython.com/pytest-python-testing/#fixtures-managing-state-and-dependencies)。

回到代码编辑器，从`tests/`目录中打开`test_rptodo.py`。向其中添加以下代码:

```py
# tests/test_rptodo.py
import json 
import pytest from typer.testing import CliRunner 
from rptodo import (
 DB_READ_ERROR, SUCCESS,    __app_name__,
    __version__,
    cli,
 rptodo, )

# ...

@pytest.fixture def mock_json_file(tmp_path):
    todo = [{"Description": "Get some milk.", "Priority": 2, "Done": False}]
    db_file = tmp_path / "todo.json"
    with db_file.open("w") as db:
        json.dump(todo, db, indent=4)
    return db_file
```

在这里，您首先更新您的导入来完成一些需求。fixture`mock_json_file()`创建并返回一个临时 JSON 文件`db_file`，其中有一个单项待办事项列表。在这个 fixture 中，您使用了 [`tmp_path`](https://docs.pytest.org/en/6.2.x/tmpdir.html#the-tmp-path-fixture) ，这是一个`pathlib.Path`对象，pytest 使用它来提供一个用于测试目的的临时目录。

您已经有一个临时待办事项数据库可以使用。现在你需要一些数据来创建你的[测试用例](https://en.wikipedia.org/wiki/Test_case):

```py
# tests/test_rptodo.py
# ...

test_data1 = {
    "description": ["Clean", "the", "house"],
    "priority": 1,
    "todo": {
        "Description": "Clean the house.",
        "Priority": 1,
        "Done": False,
    },
}
test_data2 = {
    "description": ["Wash the car"],
    "priority": 2,
    "todo": {
        "Description": "Wash the car.",
        "Priority": 2,
        "Done": False,
    },
}
```

这两个字典提供了测试`Todoer.add()`的数据。前两个键表示您将用作`.add()`的参数的数据，而第三个键保存方法的预期返回值。

现在是时候为`.add()`编写你的第一个**测试函数**了。使用 pytest，您可以使用[参数化](https://docs.pytest.org/en/6.2.x/parametrize.html#parametrize-basics)为单个测试函数提供多组参数和预期结果。这是一个非常好的特性。它使一个单一的测试函数表现得像运行不同测试用例的几个测试函数一样。

以下是在 pytest 中使用参数化创建测试函数的方法:

```py
 1# tests/test_rptodo.py
 2# ...
 3
 4@pytest.mark.parametrize( 5    "description, priority, expected",
 6    [
 7        pytest.param( 8            test_data1["description"],
 9            test_data1["priority"],
10            (test_data1["todo"], SUCCESS),
11        ),
12        pytest.param( 13            test_data2["description"],
14            test_data2["priority"],
15            (test_data2["todo"], SUCCESS),
16        ),
17    ],
18)
19def test_add(mock_json_file, description, priority, expected): 20    todoer = rptodo.Todoer(mock_json_file)
21    assert todoer.add(description, priority) == expected
22    read = todoer._db_handler.read_todos()
23    assert len(read.todo_list) == 2
```

`@pytest.mark.parametrize()`装饰器标记`test_add()`用于参数化。当 pytest 运行这个测试时，它调用`test_add()`两次。每个调用使用第 7 行到第 11 行以及第 12 行到第 16 行中的一个参数集。

第 5 行的字符串包含两个必需参数的描述性名称，以及一个描述性的返回值名称。注意`test_add()`有那些相同的参数。此外，`test_add()`的第一个参数与您刚刚定义的夹具同名。

在`test_add()`中，代码执行以下操作:

*   **第 20 行**用`mock_json_file`作为参数创建了一个`Todoer`的实例。

*   **第 21 行**断言使用`description`和`priority`作为参数对`.add()`的调用应该返回`expected`。

*   **第 22 行**从临时数据库中读取待办事项列表并存储在`read`中。

*   **第 23 行**断言待办事项列表的长度为`2`。为什么是`2`？因为`mock_json_file()`返回了一个带有待办事项的列表，现在你又添加了第二个。

酷！你有一个覆盖了`.add()`主要功能的测试。现在是时候再次运行您的测试套件了。回到你的命令行并运行`python -m pytest tests/`。您将得到类似如下的输出:

```py
======================== test session starts ==========================
platform linux -- Python 3.8.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: .../rptodo
plugins: Faker-8.1.1, cov-2.12.0, celery-4.4.7
collected 3 items

tests/test_rptodo.py .FF                                        [100%] ============================== FAILURES ===============================

# Output cropped
```

突出显示行中的字母 *F* 意味着您的两个测试用例失败了。测试失败是 TDD 的第一步。第二步是编写通过这些测试的代码。这就是你接下来要做的。

[*Remove ads*](/account/join/)

### 执行`add` CLI 命令

在本节中，您将在`Todoer`类中编写`.add()`代码。您还将在您的 Typer CLI 中编写`add`命令。有了这两段代码，您的用户将能够向他们的待办事项列表添加新项目。

待办应用每次运行都需要访问`Todoer`类，将 CLI 与数据库连接。为了满足这个需求，您将实现一个名为`get_todoer()`的函数。

回到你的代码编辑器，打开`cli.py`。键入以下代码:

```py
 1# rptodo/cli.py
 2
 3from pathlib import Path
 4from typing import List, Optional 5
 6import typer
 7
 8from rptodo import (
 9    ERRORS, __app_name__, __version__, config, database, rptodo 10)
11
12app = typer.Typer()
13
14@app.command()
15def init(
16    # ...
17
18def get_todoer() -> rptodo.Todoer: 19    if config.CONFIG_FILE_PATH.exists():
20        db_path = database.get_database_path(config.CONFIG_FILE_PATH)
21    else:
22        typer.secho(
23            'Config file not found. Please, run "rptodo init"',
24            fg=typer.colors.RED,
25        )
26        raise typer.Exit(1)
27    if db_path.exists():
28        return rptodo.Todoer(db_path)
29    else:
30        typer.secho(
31            'Database not found. Please, run "rptodo init"',
32            fg=typer.colors.RED,
33        )
34        raise typer.Exit(1)
35
36def _version_callback(value: bool) -> None:
37    # ...
```

更新导入后，在第 18 行定义`get_todoer()`。第 19 行定义了一个[条件](https://realpython.com/python-conditional-statements/)，它检查应用程序的配置文件是否存在。为此，它使用了 [`Path.exists()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.exists) 。

如果配置文件存在，那么第 20 行从中获得数据库的路径。如果文件不存在，则运行`else`子句。该子句将一条错误消息打印到屏幕上，并使用退出代码`1`退出应用程序，以发出错误信号。

第 27 行检查到数据库的路径是否存在。如果是这样，那么第 28 行创建一个`Todoer`的实例，并将路径作为参数。否则，从第 29 行开始的`else`子句打印一条错误消息并退出应用程序。

现在您已经有了一个具有有效数据库路径的`Todoer`实例，您可以编写`.add()`了。回到`rptodo.py`模块并更新`Todoer`:

```py
 1# rptodo/rptodo.py
 2from pathlib import Path
 3from typing import Any, Dict, List, NamedTuple 4
 5from rptodo import DB_READ_ERROR 6from rptodo.database import DatabaseHandler
 7
 8# ...
 9
10class Todoer:
11    def __init__(self, db_path: Path) -> None:
12        self._db_handler = DatabaseHandler(db_path)
13
14    def add(self, description: List[str], priority: int = 2) -> CurrentTodo: 15        """Add a new to-do to the database."""
16        description_text = " ".join(description)
17        if not description_text.endswith("."):
18            description_text += "."
19        todo = {
20            "Description": description_text,
21            "Priority": priority,
22            "Done": False,
23        }
24        read = self._db_handler.read_todos()
25        if read.error == DB_READ_ERROR:
26            return CurrentTodo(todo, read.error)
27        read.todo_list.append(todo)
28        write = self._db_handler.write_todos(read.todo_list)
29        return CurrentTodo(todo, write.error)
```

下面是`.add()`一行一行的工作方式:

*   **第 14 行**定义了`.add()`，它以`description`和`priority`为自变量。描述是一个字符串列表。Typer 根据您在命令行输入的单词来创建这个列表，以描述当前的待办事项。在`priority`的情况下，它是一个表示待办事项优先级的整数值。默认值为`2`，表示中等优先级。

*   **第 16 行**使用 [`.join()`](https://realpython.com/python-string-split-concatenate-join/) 将描述组件连接成一个字符串。

*   **第 17 行和第 18 行**如果用户没有添加句点(`"."`)，则在描述的末尾添加一个句点。

*   **第 19 行到第 23 行**根据用户的输入创建一个新的待办事项。

*   **第 24 行**通过调用数据库处理器上的`.read_todos()`从数据库中读取待办事项列表。

*   **第 25 行**检查`.read_todos()`是否返回一个`DB_READ_ERROR`。如果是，那么第 26 行返回一个命名的元组，`CurrentTodo`，包含当前的待办事项和错误代码。

*   第 27 行将新的待办事项添加到列表中。

*   **第 28 行**通过调用数据库处理程序上的`.write_todos()`将更新后的待办事项列表写回数据库。

*   **第 29 行**返回一个`CurrentTodo`的实例，带有当前的待办事项和一个适当的返回代码。

现在您可以再次运行您的测试套件来检查`.add()`是否正常工作。继续运行`python -m pytest tests/`。您将得到类似如下的输出:

```py
========================= test session starts =========================
platform linux -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
plugins: Faker-8.1.1, cov-2.12.0, celery-4.4.7
rootdir: .../rptodo
collected 2 items

tests/test_rptodo.py ...                                        [100%] ========================== 3 passed in 0.09s ==========================
```

三个绿点意味着你通过了三项测试。如果您从 GitHub 上的项目 repo 中下载了代码，那么您会得到一个包含更多成功测试的输出。

一旦你写完了`.add()`，你就可以去`cli.py`为你的应用程序的命令行界面写`add`命令:

```py
 1# rptodo/cli.py
 2# ...
 3
 4def get_todoer() -> rptodo.Todoer:
 5    # ...
 6
 7@app.command() 8def add( 9    description: List[str] = typer.Argument(...),
10    priority: int = typer.Option(2, "--priority", "-p", min=1, max=3),
11) -> None:
12    """Add a new to-do with a DESCRIPTION."""
13    todoer = get_todoer()
14    todo, error = todoer.add(description, priority)
15    if error:
16        typer.secho(
17            f'Adding to-do failed with "{ERRORS[error]}"', fg=typer.colors.RED
18        )
19        raise typer.Exit(1)
20    else:
21        typer.secho(
22            f"""to-do: "{todo['Description']}" was added """
23            f"""with priority: {priority}""",
24            fg=typer.colors.GREEN,
25        )
26
27def _version_callback(value: bool) -> None:
28    # ...
```

下面是对`add`命令功能的分析:

*   **第 7 行和第 8 行**使用`@app.command()` Python decorator 将`add()`定义为 Typer 命令。

*   **第 9 行**将`description`定义为`add()`的参数。此参数包含表示待办事项描述的字符串列表。为了建立论点，你可以使用`typer.Argument`。当您将一个[省略号](https://realpython.com/python-ellipsis/) ( `...`)作为第一个参数传递给`Argument`的构造函数时，您是在告诉 Typer】是必需的。此参数是必需的这一事实意味着用户必须在命令行提供待办事项描述。

*   **第 10 行**将`priority`定义为 Typer 选项，默认值为`2`。选项名为`--priority`和`-p`。正如您之前所决定的，`priority`只接受三个可能的值:`1`、`2`或`3`。为了保证这一条件，您将`min`设置为`1`并将`max`设置为`3`。这样，Typer 会自动验证用户的输入，并且只接受指定区间内的数字。

*   第 13 行得到一个要使用的`Todoer`实例。

*   **第 14 行**在`todoer`上调用`.add()`，并将结果解包到`todo`和`error`中。

*   **第 15 行到第 25 行**定义了一个条件语句，如果在向数据库添加新的待办事项时出现错误，则打印一条错误消息并退出应用程序。如果没有错误发生，那么第 20 行的`else`子句在屏幕上显示一条成功消息。

现在，您可以回到您的终端，尝试一下您的`add`命令:

```py
(venv) $ python -m rptodo add Get some milk -p 1
to-do: "Get some milk." was added with priority: 1

(venv) $ python -m rptodo add Clean the house --priority 3
to-do: "Clean the house." was added with priority: 3

(venv) $ python -m rptodo add Wash the car
to-do: "Wash the car." was added with priority: 2

(venv) $ python -m rptodo add Go for a walk -p 5
Usage: rptodo add [OPTIONS] DESCRIPTION...
Try 'rptodo add --help' for help.

Error: Invalid value for '--priority' / '-p': 5 is not in the valid range...
```

在第一个例子中，您执行带有描述`"Get some milk"`和优先级`1`的`add`命令。要设置优先级，您可以使用`-p`选项。按下 `Enter` 后，应用程序会添加待办事项并通知您添加成功。第二个例子非常相似。这次您使用`--priority`将待办事项优先级设置为`3`。

在第三个示例中，您提供了一个待办事项描述，但没有提供优先级。在这种情况下，应用程序使用默认的优先级值，即`2`。

在第四个例子中，您尝试添加一个优先级为`5`的新待办事项。由于这个优先级值超出了允许的范围，Typer 显示一个[用法消息](https://en.wikipedia.org/wiki/Usage_message)以及一个错误消息。请注意，Typer 会自动为您显示这些消息。您不需要添加额外的代码来实现这一点。

太好了！你的待办事项已经有了一些很酷的功能。现在你需要一种方法来列出你所有的待办事项，以了解你有多少工作要做。在下一节中，您将实现`list`命令来帮助您完成这项任务。

[*Remove ads*](/account/join/)

### 执行`list`命令

在本节中，您将把`list`命令添加到应用程序的 CLI 中。这个命令将允许你的用户列出他们当前所有的待办事项。在向 CLI 添加任何代码之前，您需要一种从数据库中检索整个待办事项列表的方法。为了完成这个任务，您将把`.get_todo_list()`添加到`Todoer`类中。

在代码编辑器或 IDE 中打开`rptodo.py`,添加以下代码:

```py
# rptodo/rptodo.py
# ...

class Todoer:
    # ...
 def get_todo_list(self) -> List[Dict[str, Any]]:        """Return the current to-do list."""
        read = self._db_handler.read_todos()
        return read.todo_list
```

在`.get_todo_list()`中，首先通过调用数据库处理程序上的`.read_todos()`从数据库中获得整个待办事项列表。对`.read_todos()`的调用返回一个命名的元组`DBResponse`，其中包含待办事项列表和一个返回代码。然而，您只需要待办事项列表，所以`.get_todo_list()`只返回`.todo_list`字段。

有了`.get_todo_list()`,您现在可以在应用程序的 CLI 中实现`list`命令。继续将`list_all()`添加到`cli.py`:

```py
 1# rptodo/cli.py
 2# ...
 3
 4@app.command()
 5def add(
 6    # ...
 7
 8@app.command(name="list") 9def list_all() -> None: 10    """List all to-dos."""
11    todoer = get_todoer()
12    todo_list = todoer.get_todo_list()
13    if len(todo_list) == 0:
14        typer.secho(
15            "There are no tasks in the to-do list yet", fg=typer.colors.RED
16        )
17        raise typer.Exit()
18    typer.secho("\nto-do list:\n", fg=typer.colors.BLUE, bold=True)
19    columns = (
20        "ID.  ",
21        "| Priority  ",
22        "| Done  ",
23        "| Description  ",
24    )
25    headers = "".join(columns)
26    typer.secho(headers, fg=typer.colors.BLUE, bold=True)
27    typer.secho("-" * len(headers), fg=typer.colors.BLUE)
28    for id, todo in enumerate(todo_list, 1):
29        desc, priority, done = todo.values()
30        typer.secho(
31            f"{id}{(len(columns[0]) - len(str(id))) * ' '}"
32            f"| ({priority}){(len(columns[1]) - len(str(priority)) - 4) * ' '}"
33            f"| {done}{(len(columns[2]) - len(str(done)) - 2) * ' '}"
34            f"| {desc}",
35            fg=typer.colors.BLUE,
36        )
37    typer.secho("-" * len(headers) + "\n", fg=typer.colors.BLUE)
38
39def _version_callback(value: bool) -> None:
40    # ...
```

下面是`list_all()`的工作原理:

*   **第 8 行和第 9 行**使用`@app.command()`装饰器将`list_all()`定义为一个类型命令。这个装饰器的`name`参数为命令设置了一个自定义名称，这里是`list`。注意`list_all()`没有任何参数或选项。它只是列出了用户从命令行运行`list`时的待办事项。

*   **第 11 行**获取您将使用的`Todoer`实例。

*   **第 12 行**通过调用`todoer`上的`.get_todo_list()`从数据库中获取待办事项列表。

*   **第 13 到 17 行**定义了一个条件语句来检查列表中是否至少有一个待办事项。如果没有，那么`if`代码块将错误信息打印到屏幕上并退出应用程序。

*   **第 18 行**打印一个顶层标题来呈现待办事项列表。在这种情况下，`secho()`接受一个名为`bold`的额外布尔参数，这使您能够以粗体格式显示文本。

*   **第 19 到 27 行**定义并打印所需的列，以表格格式显示待办事项列表。

*   **第 28 行到第 36 行**运行一个 [`for`循环](https://realpython.com/python-for-loop/)用适当的填充和分隔符将每个待办事项打印到自己的行上。

*   **第 37 行**打印一行破折号，最后一个[换行符](https://realpython.com/python-data-types/#applying-special-meaning-to-characters)(`\n`)可视地将待办事项列表与下一个命令行提示符分开。

如果您使用`list`命令运行应用程序，那么您会得到以下输出:

```py
(venv) $ python -m rptodo list

to-do list:

ID.  | Priority  | Done  | Description
----------------------------------------
1    | (1)       | False | Get some milk.
2    | (3)       | False | Clean the house.
3    | (2)       | False | Wash the car.
----------------------------------------
```

这个输出在一个格式良好的表格中显示了当前所有的待办事项。这样，您的用户可以跟踪他们的任务列表的状态。请注意，输出应该在您的终端窗口中以蓝色字体显示。

## 步骤 6:编写待办事项完成功能的代码

您将添加到待办事项应用程序的下一个特性是一个 Typer 命令，它允许您的用户将一个给定的待办事项设置为完成。这样，您的用户可以跟踪他们的进度，并知道还有多少工作要做。

同样，您可以通过点击下面的链接并转到`source_code_step_6/`目录来下载本节的代码和所有资源，包括额外的单元测试:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-typer-cli-project-code/)使用 Python 和 Typer 为您的命令行构建一个待办事项应用程序。

像往常一样，您将从在`Todoer`中编码所需的功能开始。在这种情况下，您需要一个方法，它接受一个待办事项 ID 并将相应的待办事项标记为完成。回到代码编辑器中的`rptodo.py`，添加以下代码:

```py
 1# rptodo/rptodo.py
 2# ...
 3from rptodo import DB_READ_ERROR, ID_ERROR 4from rptodo.database import DatabaseHandler
 5
 6# ...
 7
 8class Todoer:
 9    # ...
10    def set_done(self, todo_id: int) -> CurrentTodo: 11        """Set a to-do as done."""
12        read = self._db_handler.read_todos()
13        if read.error:
14            return CurrentTodo({}, read.error)
15        try:
16            todo = read.todo_list[todo_id - 1]
17        except IndexError:
18            return CurrentTodo({}, ID_ERROR)
19        todo["Done"] = True
20        write = self._db_handler.write_todos(read.todo_list)
21        return CurrentTodo(todo, write.error)
```

您的新`.set_done()`方法完成了所需的工作。方法如下:

*   **第 10 行**定义`.set_done()`。该方法采用一个名为`todo_id`的参数，它保存一个整数，表示您想要标记为完成的待办事项的 ID。当你使用`list`命令列出你的待办事项时，待办事项 ID 是与给定的待办事项相关联的数字。因为您使用 Python [list](https://realpython.com/python-lists-tuples/) 来存储待办事项，所以您可以将这个 ID 转换成从零开始的索引，并使用它从列表中检索所需的待办事项。

*   **第 12 行**通过调用数据库处理程序上的`.read_todos()`来读取所有的待办事项。

*   **第 13 行**检查读取过程中是否出现错误。如果是，那么第 14 行返回一个命名的元组`CurrentTodo`，带有一个空的待办事项和错误。

*   **第 15 行**开始一个`try` … `except`语句来捕捉无效的待办事项 id，这些 id 转换成底层待办事项列表中的无效索引。如果发生了一个`IndexError`，那么第 18 行返回一个`CurrentTodo`实例，带有一个空的待办事项和相应的错误代码。

*   **第 19 行**将`True`分配给目标待办字典中的`"Done"`键。这样，你就把待办事项设置为完成。

*   **第 20 行**通过调用数据库处理程序上的`.write_todos()`将更新写回数据库。

*   **第 21 行**返回一个`CurrentTodo`实例，带有目标待办事项和指示操作进行情况的返回代码。

当`.set_done()`就位后，你可以移动到`cli.py`并编写`complete`命令。下面是所需的代码:

```py
 1# rptodo/cli.py
 2# ...
 3
 4@app.command(name="list")
 5def list_all() -> None:
 6    # ...
 7
 8@app.command(name="complete") 9def set_done(todo_id: int = typer.Argument(...)) -> None: 10    """Complete a to-do by setting it as done using its TODO_ID."""
11    todoer = get_todoer()
12    todo, error = todoer.set_done(todo_id)
13    if error:
14        typer.secho(
15            f'Completing to-do # "{todo_id}" failed with "{ERRORS[error]}"',
16            fg=typer.colors.RED,
17        )
18        raise typer.Exit(1)
19    else:
20        typer.secho(
21            f"""to-do # {todo_id} "{todo['Description']}" completed!""",
22            fg=typer.colors.GREEN,
23        )
24
25def _version_callback(value: bool) -> None:
26    # ...
```

看看这段代码是如何一行一行地工作的:

*   **第 8 行和第 9 行**用通常的`@app.command()`装饰器将`set_done()`定义为一个类型命令。在这种情况下，您使用`complete`作为命令名。`set_done()`函数接受一个名为`todo_id`的参数，默认为`typer.Argument`的一个实例。该实例将作为必需的命令行参数。

*   **第 11 行**得到通常的`Todoer`实例。

*   **第 12 行**通过调用`todoer`上的`.set_done()`来设置特定`todo_id`的待办事项。

*   **第 13 行**检查过程中是否出现错误。如果是这样，那么第 14 到 18 行打印一个适当的错误消息，并使用退出代码`1`退出应用程序。如果没有错误发生，那么第 20 到 23 行用绿色字体打印一条成功消息。

就是这样！现在你可以试试你的新`complete`命令了。回到终端窗口，运行以下命令:

```py
(venv) $ python -m rptodo list

to-do list:

ID.  | Priority  | Done  | Description
----------------------------------------
1    | (1)       | False | Get some milk.
2    | (3)       | False | Clean the house.
3    | (2)       | False | Wash the car.
----------------------------------------

(venv) $ python -m rptodo complete 1
to-do # 1 "Get some milk." completed!

(venv) $ python -m rptodo list

to-do list:

ID.  | Priority  | Done  | Description
----------------------------------------
1    | (1)       | True  | Get some milk.
2    | (3)       | False | Clean the house.
3    | (2)       | False | Wash the car.
----------------------------------------
```

首先，您列出所有的待办事项，以可视化对应于每个待办事项的 ID。然后使用`complete`将 ID 为`1`的待办事项设置为完成。当你再次列出待办事项时，你会看到第一个待办事项在*完成*栏中被标记为`True`。

关于`complete`命令和底层`Todoer.set_done()`方法需要注意的一个重要细节是，待办事项 ID 不是一个固定值。如果您从列表中删除一个或多个待办事项，那么一些剩余待办事项的 id 将会改变。说到删除待办事项，这就是你在接下来的部分要做的。

[*Remove ads*](/account/join/)

## 步骤 7:编写删除待办功能的代码

从列表中删除待办事项是你可以添加到待办事项应用程序中的另一个有用的功能。在本节中，您将使用 Python 向应用程序的 CLI 添加两个新的 Typer 命令。第一个命令将是`remove`。它将允许您的用户通过 ID 删除待办事项。第二个命令是`clear`，它将允许用户从数据库中删除所有当前的待办事项。

您可以通过点击下面的链接并转到`source_code_step_7/`目录来下载本节的代码、单元测试和其他资源:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-typer-cli-project-code/)使用 Python 和 Typer 为您的命令行构建一个待办事项应用程序。

### 执行`remove` CLI 命令

要在应用程序的 CLI 中实现`remove`命令，首先需要在`Todoer`中编写底层的`.remove()`方法。该方法将提供使用待办事项 ID 从列表中删除单个待办事项的所有功能。请记住，您将待办事项 ID 设置为与特定待办事项相关联的整数。要显示待办事项 id，运行`list`命令。

以下是如何在`Todoer`中编写`.remove()`的方法:

```py
 1# rptodo/rptodo.py
 2# ...
 3
 4class Todoer:
 5    # ...
 6    def remove(self, todo_id: int) -> CurrentTodo: 7        """Remove a to-do from the database using its id or index."""
 8        read = self._db_handler.read_todos()
 9        if read.error:
10            return CurrentTodo({}, read.error)
11        try:
12            todo = read.todo_list.pop(todo_id - 1)
13        except IndexError:
14            return CurrentTodo({}, ID_ERROR)
15        write = self._db_handler.write_todos(read.todo_list)
16        return CurrentTodo(todo, write.error)
```

这里，您的代码执行以下操作:

*   **第 6 行**定义`.remove()`。此方法将待办事项 ID 作为参数，并从数据库中删除相应的待办事项。

*   **第 8 行**通过调用数据库处理程序上的`.read_todos()`从数据库中读取待办事项列表。

*   **第 9 行**检查读取过程中是否出现错误。如果是，那么第 10 行返回一个命名的 tuple，`CurrentTodo`，包含一个空的 to-do 和相应的错误代码。

*   **第 11 行**开始一个`try` … `except`语句来捕捉任何来自用户输入的无效 id。

*   **第 12 行**从待办事项列表中删除索引`todo_id - 1`处的待办事项。如果在这个操作过程中出现了一个`IndexError`，那么第 14 行返回一个`CurrentTodo`实例，带有一个空的待办事项和相应的错误代码。

*   **第 15 行**将更新后的待办事项列表写回数据库。

*   **第 16 行**返回一个`CurrentTodo`元组，保存被移除的待办事项和一个指示操作成功的返回码。

现在你已经在`Todoer`中完成了`.remove()`的编码，你可以去`cli.py`并添加`remove`命令:

```py
 1# rptodo/cli.py
 2# ...
 3
 4@app.command()
 5def set_done(todo_id: int = typer.Argument(...)) -> None:
 6    # ...
 7
 8@app.command() 9def remove( 10    todo_id: int = typer.Argument(...),
11    force: bool = typer.Option(
12        False,
13        "--force",
14        "-f",
15        help="Force deletion without confirmation.",
16    ),
17) -> None:
18    """Remove a to-do using its TODO_ID."""
19    todoer = get_todoer()
20
21    def _remove():
22        todo, error = todoer.remove(todo_id)
23        if error:
24            typer.secho(
25                f'Removing to-do # {todo_id} failed with "{ERRORS[error]}"',
26                fg=typer.colors.RED,
27            )
28            raise typer.Exit(1)
29        else:
30            typer.secho(
31                f"""to-do # {todo_id}: '{todo["Description"]}' was removed""",
32                fg=typer.colors.GREEN,
33            )
34
35    if force:
36        _remove()
37    else:
38        todo_list = todoer.get_todo_list()
39        try:
40            todo = todo_list[todo_id - 1]
41        except IndexError:
42            typer.secho("Invalid TODO_ID", fg=typer.colors.RED)
43            raise typer.Exit(1)
44        delete = typer.confirm(
45            f"Delete to-do # {todo_id}: {todo['Description']}?"
46        )
47        if delete:
48            _remove()
49        else:
50            typer.echo("Operation canceled")
51
52def _version_callback(value: bool) -> None:
53    # ...
```

哇！代码太多了。它是这样工作的:

*   第 8 行和第 9 行将`remove()`定义为一个键入 CLI 命令。

*   **第 10 行**将`todo_id`定义为`int`类型的参数。在这种情况下，`todo_id`是`typer.Argument`的必需实例。

*   **第 11 行**将`force`定义为`remove`命令的一个选项。这是一个布尔选项，允许你在没有确认的情况下删除待办事项。该选项默认为`False`(第 12 行)，其标志为`--force`和`-f`(第 13 行和第 14 行)。

*   **第 15 行**定义了`force`选项的帮助信息。

*   **第 19 行**创建所需的`Todoer`实例。

*   **第 21 到 33 行**定义了一个叫做`_remove()`的[内部函数](https://realpython.com/inner-functions-what-are-they-good-for/)。这是一个助手功能，允许您重用删除功能。该函数使用待办事项的 ID 删除待办事项。为此，它在`todoer`上调用`.remove()`。

*   **第 35 行**检查`force`的值。一个`True`值意味着用户想要在没有确认的情况下删除待办事项。在这种情况下，第 36 行调用`_remove()`来运行删除操作。

*   **第 37 行**开始一个`else`子句，如果`force`是`False`则运行该子句。

*   第 38 行从数据库中获取整个待办事项列表。

*   **第 39 到 43 行**定义了一个`try` … `except`语句，从列表中检索所需的待办事项。如果发生了`IndexError`，那么第 42 行将打印一条错误消息，第 43 行将退出应用程序。

*   **第 44 到 46 行**调用 Typer 的 [`confirm()`](https://typer.tiangolo.com/tutorial/prompt/#confirm) 并将结果存储在`delete`中。该功能提供了另一种要求确认的方式。它允许您使用动态创建的确认提示，如第 45 行所示。

*   **47 线**检查`delete`是否为`True`，如果是，48 线调用`_remove()`。否则，第 50 行告知操作被取消。

您可以通过在命令行上运行以下命令来尝试使用`remove`命令:

```py
(venv) $ python -m rptodo list

to-do list:

ID.  | Priority  | Done  | Description
----------------------------------------
1    | (1)       | True  | Get some milk.
2    | (3)       | False | Clean the house.
3    | (2)       | False | Wash the car.
----------------------------------------

(venv) $ python -m rptodo remove 1
Delete to-do # 1: Get some milk.? [y/N]:
Operation canceled

(venv) $ python -m rptodo remove 1
Delete to-do # 1: Get some milk.? [y/N]: y
to-do # 1: 'Get some milk.' was removed

(venv) $ python -m rptodo list

to-do list:

ID.  | Priority  | Done  | Description
----------------------------------------
1    | (3)       | False | Clean the house.
2    | (2)       | False | Wash the car.
----------------------------------------
```

在这组命令中，首先用`list`命令列出当前所有的待办事项。然后你尝试用 ID 号`1`删除待办事项。这将向您显示是(`y`)或否(`N`)的确认提示。如果您按下 `Enter` ，则应用程序运行默认选项`N`，并取消移除操作。

**注意:**如果您使用的是高于 0.3.2 的 Typer 版本，那么上例中的确认提示可能会有所不同。

例如，在 macOS 上，确认提示没有默认答案:

```py
$ # Typer version 0.4.0 on macOS
$ python -m rptodo remove 1
Delete to-do # 1: Get some milk.? [y/n]:
Error: invalid input
```

如果您的情况就是这样，那么您需要在命令行明确提供一个答案，然后按下 `Enter` 。

在第三个命令中，您显式地提供了一个`y`答案，因此应用程序删除了 ID 号为`1`的待办事项。如果你再次列出所有的待办事项，你会发现待办事项`"Get some milk."`已经不在列表中了。作为实验，继续尝试使用`--force`或`-f`选项，或者尝试删除列表中没有的待办事项。

### 执行`clear` CLI 命令

在本节中，您将实现`clear`命令。此命令将允许您的用户从数据库中删除所有待办事项。在`clear`命令下面是来自`Todoer`的`.remove_all()`方法，它提供后端功能。

回到`rptodo.py`，在`Todoer`的末尾加上`.remove_all()`:

```py
# rptodo/rptodo.py
# ...

class Todoer:
    # ...
 def remove_all(self) -> CurrentTodo:        """Remove all to-dos from the database."""
        write = self._db_handler.write_todos([])
        return CurrentTodo({}, write.error)
```

在`.remove_all()`中，通过用一个空列表替换当前的待办事项列表，从数据库中删除所有的待办事项。为了一致性，该方法返回一个带有空字典和适当的返回或错误代码的`CurrentTodo`元组。

现在，您可以在应用程序的 CLI 中实现`clear`命令:

```py
 1# rptodo/cli.py
 2# ...
 3
 4@app.command()
 5def remove(
 6    # ...
 7
 8@app.command(name="clear") 9def remove_all( 10    force: bool = typer.Option(
11        ...,
12        prompt="Delete all to-dos?",
13        help="Force deletion without confirmation.",
14    ),
15) -> None:
16    """Remove all to-dos."""
17    todoer = get_todoer()
18    if force:
19        error = todoer.remove_all().error
20        if error:
21            typer.secho(
22                f'Removing to-dos failed with "{ERRORS[error]}"',
23                fg=typer.colors.RED,
24            )
25            raise typer.Exit(1)
26        else:
27            typer.secho("All to-dos were removed", fg=typer.colors.GREEN)
28    else:
29        typer.echo("Operation canceled")
30
31def _version_callback(value: bool) -> None:
32    # ...
```

下面是这段代码的工作原理:

*   **第 8 行和第 9 行**使用带有`clear`的`@app.command()`装饰器将`remove_all()`定义为一个类型命令。

*   **第 10 到 14 行**将`force`定义为一个类型器`Option`。这是布尔类型的必需选项。`prompt`参数要求用户为`force`输入一个合适的值，可以是`y`或`n`。

*   **第 13 行**为`force`选项提供帮助信息。

*   **第 17 行**得到通常的`Todoer`实例。

*   **第 18 行**检查`force`是否为`True`。如果是，那么`if`代码块使用`.remove_all()`从数据库中删除所有待办事项。如果在此过程中出错，应用程序会打印一条错误消息并退出(第 21 到 25 行)。否则，它会在第 27 行打印一条成功消息。

*   **如果用户通过向`force`提供一个假值，指示*否*，取消移除操作，则第 29 行**运行。

要尝试这个新的`clear`命令，请在您的终端上运行以下命令:

```py
(venv) $ python -m rptodo clear
Delete all to-dos? [y/N]:
Operation canceled

(venv) $ python -m rptodo clear
Delete all to-dos? [y/N]: y
All to-dos were removed

(venv) $ python -m rptodo list
There are no tasks in the to-do list yet
```

在第一个例子中，您运行`clear`。一旦你按下 `Enter` ，你会得到一个要求确认是(`y`)还是否(`N`)的提示。大写的`N`表示*否*是默认答案，所以如果你按下 `Enter` ，就有效取消了`clear`操作。

在第二个例子中，您再次运行`clear`。这一次，您显式地输入`y`作为提示的答案。这个答案使应用程序从数据库中删除整个待办事项列表。当您运行`list`命令时，您会收到一条消息，告知当前待办事项列表中没有任务。

就是这样！现在，您有了一个用 Python 和 Typer 构建的功能性 CLI 待办事项应用程序。您的应用程序提供了创建新待办事项、列出所有待办事项、管理待办事项完成情况以及根据需要删除待办事项的命令和选项。是不是很酷？

## 结论

构建用户友好的[命令行界面(CLI)](https://en.wikipedia.org/wiki/Command-line_interface) 应用程序是 Python 开发人员的一项基本技能。在 Python 生态系统中，您会发现一些创建这种应用程序的工具。诸如 [`argparse`](https://realpython.com/command-line-interfaces-python-argparse/) 、 [Click](https://palletsprojects.com/p/click/) 和 [Typer](https://typer.tiangolo.com/) 等库是 Python 中这些工具的很好的例子。这里，您使用 Python 和 Typer 构建了一个 CLI 应用程序来管理待办事项列表。

**在本教程中，您学习了如何:**

*   用 Python 和 **Typer** 构建一个**待办应用**
*   使用 Typer 将**命令**、**参数**和**选项**添加到您的待办事项应用程序中
*   使用 Python 中的 **Typer 的`CliRunner`** 和 **pytest** 来测试你的待办应用

您还练习了一些额外的技能，比如使用 Python 的`json`模块处理 **JSON 文件**，使用 Python 的`configparser`模块管理**配置文件**。现在，您已经准备好构建命令行应用程序了。

您可以通过点击下面的链接并转到`source_code_final/`目录来下载这个项目的全部代码和所有资源:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-typer-cli-project-code/)使用 Python 和 Typer 为您的命令行构建一个待办事项应用程序。

## 接下来的步骤

在本教程中，您已经使用 Python 和 Typer 为命令行构建了一个功能性的待办事项应用程序。尽管应用程序只提供了最少的一组功能，但这是一个很好的起点，可以让您继续添加功能，并在这个过程中不断学习。这将帮助您将 Python 技能提升到一个新的水平。

以下是一些你可以用来继续扩展你的待办事项应用程序的想法:

*   **添加对日期和截止日期的支持:**您可以使用 [`datetime`](https://realpython.com/python-datetime/) 模块来完成这项工作。该功能将允许用户更好地控制他们的任务。

*   **编写更多的单元测试:**你可以使用 pytest 为你的代码编写更多的测试。这将增加[代码覆盖率](https://en.wikipedia.org/wiki/Code_coverage)，并帮助您提高测试技能。你可能会在这个过程中发现一些错误。如果是这样的话，请在评论中发表吧。

*   **打包应用程序并发布到 PyPI:** 你可以使用[诗歌](https://python-poetry.org/)或其他类似的工具打包你的待办应用程序并由[发布到 PyPI](https://realpython.com/pypi-publish-python-package/) 。

这些只是一些想法。接受挑战，在这个项目的基础上建立一些很酷的东西！在这个过程中你会学到很多东西。**********