# 用 Python 的 argparse 构建命令行界面

> 原文：<https://realpython.com/command-line-interfaces-python-argparse/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 argparse**](/courses/python-argparse-command-line-interfaces/) 构建命令行接口

命令行应用在普通用户的空间中可能并不常见，但它们存在于开发、数据科学、系统管理和许多其他操作中。每个命令行应用程序都需要一个用户友好的[命令行界面(CLI)](https://en.wikipedia.org/wiki/Command-line_interface) ，这样你就可以与应用程序本身进行交互。在 Python 中，可以用[标准库](https://docs.python.org/3/library/index.html)中的 **`argparse`** 模块创建全功能 CLI。

**在这篇文章中，你将学习如何:**

*   从**命令行界面**开始
*   **组织**和**用 Python 布局**一个命令行 app 项目
*   用 Python 的 **`argparse`** 创建**命令行界面**
*   使用`argparse`的一些**强大功能**深度定制您的 CLI

为了充分利用本教程，您应该熟悉 Python 编程，包括诸如[面向对象编程](https://realpython.com/python3-object-oriented-programming/)、[脚本开发和执行](https://realpython.com/run-python-scripts/)以及 Python [包和模块](https://realpython.com/python-modules-packages/)等概念。如果您熟悉与使用命令行或终端相关的一般概念和主题，这也会很有帮助。

**源代码:** [点击这里下载源代码](https://realpython.com/bonus/command-line-interfaces-python-argparse-code/)，您将使用它来构建与`argparse`的命令行界面。

## 了解命令行界面

自从计算机发明以来，人类一直需要并找到与这些机器交互和共享信息的方法。信息交换在人类、[计算机软件](https://en.wikipedia.org/wiki/Software)和[硬件组件](https://en.wikipedia.org/wiki/Computer_hardware)之间流动。这些元素中的任何两个之间的共享边界一般被称为[接口](https://en.wikipedia.org/wiki/Interface_(computing))。

在软件开发中，接口是给定软件的一个特殊部分，它允许计算机系统的组件之间进行交互。当涉及到人和软件的交互时，这个重要的组件被称为用户界面。

你会在编程中发现不同类型的用户界面。大概，[图形用户界面(GUI)](https://realpython.com/python-gui-tkinter/)是当今最常见的。然而，你也会发现为用户提供[命令行界面(CLIs)](https://en.wikipedia.org/wiki/Command-line_interface) 的应用和程序。在本教程中，您将了解 CLI 以及如何用 Python 创建它们。

[*Remove ads*](/account/join/)

### 命令行界面

**命令行界面**允许你通过操作系统命令行、终端或控制台与应用程序或程序进行交互。

要理解命令行界面及其工作原理，请考虑这个实际的例子。假设您有一个名为`sample`的目录，其中包含三个示例文件。如果您使用的是类似于 [Unix 的](https://en.wikipedia.org/wiki/Unix-like)操作系统，比如 Linux 或 macOS，那么在父目录中打开一个命令行窗口或终端，然后执行以下命令:

```py
$ ls sample/
hello.txt     lorem.md      realpython.md
```

[`ls` Unix 命令](https://en.wikipedia.org/wiki/Ls)列出了目标目录下包含的文件和子目录，默认为当前工作目录。上面的命令调用没有显示太多关于`sample`内容的信息。它只在屏幕上显示文件名。

**注意:**如果你在 Windows 上，那么你会有一个`ls`命令，它的工作方式类似于 Unix 的`ls`命令。但是，在普通形式下，该命令会显示不同的输出:

```py
PS> ls .\sample\

 Directory: C:\sample

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---          11/10/2022 10:06 AM             88 hello.txt
-a---          11/10/2022 10:06 AM           2629 lorem.md
-a---          11/10/2022 10:06 AM            429 realpython.md
```

PowerShell `ls`命令发出一个表，其中包含目标目录下每个文件和子目录的详细信息。因此，接下来的例子在 Windows 系统上不会像预期的那样工作。

假设您想要关于您的目录及其内容的更丰富的信息。在这种情况下，你不需要四处寻找除了`ls`之外的程序，因为这个命令有一个全功能的命令行界面，它有一组有用的**选项**，你可以用它来定制命令的行为。

例如，使用`-l`选项继续执行`ls`:

```py
$ ls -l sample/
total 24
-rw-r--r--@ 1 user  staff    83 Aug 17 22:15 hello.txt
-rw-r--r--@ 1 user  staff  2609 Aug 17 22:15 lorem.md
-rw-r--r--@ 1 user  staff   428 Aug 17 22:15 realpython.md
```

现在`ls`的输出已经大不一样了。该命令显示了关于`sample`中文件的更多信息，包括权限、所有者、组、日期和大小。它还显示这些文件在您的计算机磁盘上使用的总空间。

**注意:**要获得作为 CLI 一部分的`ls`提供的所有选项的详细列表，请在命令行或终端中运行`man ls`命令。

这个更丰富的输出来自于使用`-l`选项，它是 Unix `ls`命令行界面的一部分，支持详细的输出格式。

### 命令、自变量、选项、参数和子命令

在本教程中，你将学习到**命令**和**子命令**。您还将了解命令行**参数**、**选项**和**参数**，因此您应该将这些术语纳入您的技术词汇表:

*   **命令:**在命令行或终端窗口运行的程序或例程。您通常会用底层程序或例程的名称来标识命令。

*   **参数:**命令用来执行其预期动作的必需或可选信息。命令通常接受一个或多个参数，您可以在命令行中以空格分隔或逗号分隔的列表形式提供这些参数。

*   **选项**，也称为**标志**或**开关:**修改命令行为的可选参数。选项使用特定的名称传递给命令，就像前面例子中的`-l`。

*   **参数:**选项用来执行其预定操作或动作的自变量。

*   **子命令:**预定义的名称，可以传递给应用程序来运行特定的动作。

考虑上一节中的示例命令结构:

```py
$ ls -l sample/
```

在此示例中，您组合了 CLI 的以下组件:

*   **`ls`** :命令名或应用名
*   **`-l`** :启用详细输出的选项、开关或标志
*   **`sample`** :为命令的执行提供附加信息的参数

现在考虑下面的命令结构，它展示了 Python 的包管理器的 CLI，称为 [`pip`](https://realpython.com/what-is-pip/) :

```py
$ pip install -r requirements.txt
```

这是一个常见的`pip`命令结构，您可能以前见过。它允许您使用一个`requirements.txt`文件来安装给定 Python 项目的需求。在本例中，您使用了以下 CLI 组件:

*   **`pip`** :命令的名称
*   **`install`** :一条`pip`子命令的名称
*   **`-r`** :是`install`子命令的一个选项
*   **`requirements.txt`** :实参，具体是`-r`选项的一个参数

现在你知道什么是命令行界面，它们的主要部分或组件是什么。是时候学习如何用 Python 创建自己的 CLI 了。

[*Remove ads*](/account/join/)

## Python 中的 CLIs 入门:`sys.argv` vs `argparse`

Python 附带了一些工具，您可以使用它们为您的程序和应用程序编写命令行界面。如果你需要为一个小程序快速创建一个最小化的 CLI，那么你可以使用 [`sys`](https://docs.python.org/3/library/sys.html#module-sys) 模块中的 [`argv`](https://docs.python.org/3/library/sys.html#sys.argv) 属性。该属性自动存储您在命令行传递给给定程序的参数。

### 使用`sys.argv`构建一个最小的 CLI

作为使用`argv`创建最小 CLI 的例子，假设您需要编写一个小程序，列出给定目录中的所有文件，类似于`ls`所做的。在这种情况下，你可以这样写:

```py
# ls_argv.py

import sys
from pathlib import Path

if (args_count := len(sys.argv)) > 2:
    print(f"One argument expected, got {args_count - 1}")
    raise SystemExit(2)
elif args_count < 2:
    print("You must specify the target directory")
    raise SystemExit(2)

target_dir = Path(sys.argv[1])

if not target_dir.is_dir():
    print("The target directory doesn't exist")
    raise SystemExit(1)

for entry in target_dir.iterdir():
    print(entry.name)
```

这个程序通过手动处理命令行提供的参数来实现一个最小的 CLI，这些参数自动存储在`sys.argv`中。`sys.argv`中的第一项总是程序名。第二项将是目标目录。app 不应该接受一个以上的目标目录，所以`args_count`不能超过`2`。

检查完`sys.argv`的内容后，创建一个 [`pathlib.Path`](https://realpython.com/python-pathlib/) 对象来存储目标目录的路径。如果这个目录不存在，那么你通知用户并退出应用程序。 [`for`循环](https://realpython.com/python-for-loop/)列出了目录内容，每行一个条目。

如果您从命令行[运行脚本](https://realpython.com/run-python-scripts/)，那么您将得到以下结果:

```py
$ python ls_argv.py sample/
hello.txt
lorem.md
realpython.md

$ python ls_argv.py
You must specify the target directory

$ python ls_argv.py sample/ other_dir/
One argument expected, got 2

$ python ls_argv.py non_existing/
The target directory doesn't exist
```

您的程序将一个目录作为参数，并列出其内容。如果您运行不带参数的命令，那么您会得到一条错误消息。如果在多个目标目录下运行该命令，也会出现错误。使用不存在的目录运行该命令会产生另一条错误消息。

即使您的程序运行良好，使用`sys.argv`属性手动解析命令行参数对于更复杂的 CLI 应用程序来说也不是一个可伸缩的解决方案。如果您的应用程序需要更多的参数和选项，那么解析`sys.argv`将是一项复杂且容易出错的任务。你需要更好的东西，你可以在 Python 的`argparse`模块中得到它。

### 使用`argparse`和创建 CLI

用 Python 创建 CLI 应用程序的一个更方便的方法是使用 [`argparse`](https://docs.python.org/3/library/argparse.html?highlight=argparse#module-argparse) 模块，它来自[标准库](https://docs.python.org/3/library/index.html)。这个模块最初是在 PEP [389](https://www.python.org/dev/peps/pep-0389/) 的 [Python 3.2](https://docs.python.org/3/whatsnew/3.2.html#pep-389-argparse-command-line-parsing-module) 中发布的，是一种在 Python 中创建 CLI 应用程序的快捷方式，无需安装第三方库，如 [Typer](https://realpython.com/python-typer-cli/) 或 [Click](https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/) 。

这个模块是作为旧的 [`getopt`](https://docs.python.org/3/library/getopt.html) 和 [`optparse`](https://docs.python.org/3/library/optparse.html) 模块的替代品发布的，因为它们缺少一些重要的功能。

Python 的`argparse`模块允许您:

*   解析命令行**参数**和**选项**
*   在单个选项中取一个**可变数量的参数**
*   在您的 CLI 中提供**子命令**

这些特性将`argparse`变成了一个强大的 CLI 框架，您可以在创建 CLI 应用程序时放心地依赖它。要使用 Python 的`argparse`，您需要遵循四个简单的步骤:

1.  导入`argparse`。
2.  通过实例化 [`ArgumentParser`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser) 创建一个**参数解析器**。
3.  使用 [`.add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument) 方法将**参数**和**选项**添加到解析器中。
4.  在解析器上调用 [`.parse_args()`](https://docs.python.org/3/library/argparse.html?highlight=argparse#argparse.ArgumentParser.parse_args) 来得到 [`Namespace`](https://docs.python.org/3/library/argparse.html#namespace) 的参数。

举个例子，你可以用`argparse`来改进你的`ls_argv.py`脚本。继续用下面的代码创建`ls.py`:

```py
# ls.py v1

import argparse from pathlib import Path

parser = argparse.ArgumentParser() 
parser.add_argument("path") 
args = parser.parse_args() 
target_dir = Path(args.path)

if not target_dir.exists():
    print("The target directory doesn't exist")
    raise SystemExit(1)

for entry in target_dir.iterdir():
    print(entry.name)
```

随着`argparse`的引入，您的代码发生了显著的变化。与前一版本最显著的不同是，检查用户提供的参数的[条件语句](https://realpython.com/python-conditional-statements/)不见了。那是因为`argparse`会自动检查参数的存在。

在这个新的实现中，首先导入`argparse`并创建一个参数解析器。要创建解析器，可以使用`ArgumentParser`类。接下来，定义一个名为`path`的参数来获取用户的目标目录。

下一步是调用`.parse_args()`来解析输入参数，并获得一个包含所有用户参数的`Namespace`对象。请注意，现在`args` [变量](https://realpython.com/python-variables/)持有一个`Namespace`对象，该对象拥有从命令行收集的每个参数的属性。

在这个例子中，你只有一个参数，叫做`path`。`Namespace`对象允许你使用`args`上的**点符号**来访问`path`。代码的其余部分与第一个实现中的一样。

现在继续从命令行运行这个新脚本:

```py
$ python ls.py sample/
lorem.md
realpython.md
hello.txt

$ python ls.py
usage: ls.py [-h] path
ls.py: error: the following arguments are required: path

$ python ls.py sample/ other_dir/
usage: ls.py [-h] path
ls.py: error: unrecognized arguments: other_dir/

$ python ls.py non_existing/
The target directory doesn't exist
```

第一个命令[打印](https://realpython.com/python-print/)与原始脚本`ls_argv.py`相同的输出。相比之下，第二个命令显示的输出与`ls_argv.py`中的完全不同。程序现在显示一条用法信息，并发出一个错误，告诉你必须提供`path`参数。

在第三个命令中，您传递两个目标目录，但是应用程序没有为此做好准备。因此，它会再次显示用法信息，并抛出一个错误，让您了解潜在的问题。

最后，如果您使用一个不存在的目录作为参数来运行脚本，那么您会得到一个错误，告诉您目标目录不存在，因此程序无法工作。

您现在可以使用一个新的隐式功能。现在你的程序接受一个可选的`-h`标志。来吧，试一试:

```py
$ python ls.py -h
usage: ls.py [-h] path

positional arguments:
 path

options:
 -h, --help  show this help message and exit
```

太好了，现在你的程序自动响应`-h`或`--help`标志，为你显示一个帮助信息和使用说明。这是一个非常好的特性，您可以通过在代码中引入`argparse`来免费获得它！

有了这个用 Python 创建 CLI 应用程序的快速介绍，您现在就可以更深入地研究`argparse`模块及其所有很酷的特性了。

[*Remove ads*](/account/join/)

## 用 Python 的`argparse` 创建命令行界面

您可以使用`argparse`模块为您的应用程序和项目编写用户友好的命令行界面。此模块允许您定义应用程序需要的参数和选项。然后,`argparse`将负责为您解析`sys.argv`的那些参数和选项。

`argparse`的另一个很酷的特性是它会自动为你的 CLI 应用程序生成用法和帮助信息。该模块还会发出错误以响应无效的参数等。

在深入研究`argparse`之前，您需要知道模块的[文档](https://docs.python.org/3/library/argparse.html#module-argparse)识别两种不同类型的命令行参数:

1.  **位置自变量**，也就是你所知道的自变量
2.  **可选参数**，即选项、标志或开关

在`ls.py`示例中，`path`是一个**位置自变量**。这样的参数被称为*位置*，因为它在命令结构中的相对位置定义了它的用途。

**可选的**参数不是强制的。它们允许您修改命令的行为。在`ls` Unix 命令示例中，`-l`标志是一个可选参数，它使命令显示详细的输出。

有了这些清晰的概念，你就可以开始用 Python 和`argparse`构建你自己的 CLI 应用了。

### 创建命令行参数解析器

命令行参数解析器是任何`argparse` CLI 中最重要的部分。您在命令行中提供的所有参数和选项都将通过这个解析器，它将为您完成繁重的工作。

要用`argparse`创建命令行参数解析器，需要实例化 [`ArgumentParser`](https://docs.python.org/3/library/argparse.html#argumentparser-objects) 类:

>>>

```py
>>> from argparse import ArgumentParser

>>> parser = ArgumentParser()
>>> parser
ArgumentParser(
 prog='',
 usage=None,
 description=None,
 formatter_class=<class 'argparse.HelpFormatter'>,
 conflict_handler='error',
 add_help=True
)
```

`ArgumentParser`的[构造器](https://realpython.com/python-class-constructor/)接受许多不同的参数，您可以用它们来调整您的 CLI 的一些特性。它的所有参数都是可选的，所以您可以创建的最简单的解析器是通过实例化没有任何参数的`ArgumentParser`得到的。

在本教程中，你会学到更多关于`ArgumentParser`构造函数的参数，尤其是在关于[定制参数解析器](#customizing-your-command-line-argument-parser)的部分。现在，您可以使用`argparse`处理创建 CLI 的下一步。这一步是通过解析器对象添加参数和选项。

### 添加参数和选项

要向一个`argparse` CLI 添加参数和选项，您将使用您的`ArgumentParser`实例的 [`.add_argument()`](https://docs.python.org/3/library/argparse.html#the-add-argument-method) 方法。请注意，该方法对于参数和选项是通用的。请记住，在`argparse`术语中，参数被称为**位置参数**，选项被称为**可选参数**。

`.add_argument()`方法的第一个参数设置了参数和选项之间的区别。该自变量被标识为 [`name`或`flag`](https://docs.python.org/3/library/argparse.html?highlight=argparse#name-or-flags) 。所以，如果你提供一个`name`，那么你将定义一个参数。相反，如果你使用一个`flag`，那么你将增加一个选项。

您已经在`argparse`中使用了命令行参数。因此，考虑下面的定制`ls`命令的增强版本，它向 CLI 添加了一个`-l`选项:

```py
 1# ls.py v2
 2
 3import argparse
 4import datetime
 5from pathlib import Path
 6
 7parser = argparse.ArgumentParser()
 8
 9parser.add_argument("path")
10
11parser.add_argument("-l", "--long", action="store_true") 12
13args = parser.parse_args()
14
15target_dir = Path(args.path)
16
17if not target_dir.exists():
18    print("The target directory doesn't exist")
19    raise SystemExit(1)
20
21def build_output(entry, long=False): 22    if long:
23        size = entry.stat().st_size
24        date = datetime.datetime.fromtimestamp(
25            entry.stat().st_mtime).strftime(
26            "%b %d %H:%M:%S"
27        )
28        return f"{size:>6d}  {date}  {entry.name}"
29    return entry.name
30
31for entry in target_dir.iterdir():
32    print(build_output(entry, long=args.long))
```

在这个例子中，第 11 行创建了一个带有标志`-l`和`--long`的选项。参数和选项在语法上的区别在于，选项名以`-`开头表示简写标志，以`--`开头表示长标志。

注意，在这个特定的例子中，设置为`"store_true"`的`action`参数伴随着`-l`或`--long`选项，这意味着这个选项将存储一个[布尔值](https://realpython.com/python-boolean/)。如果您在命令行提供选项，那么它的值将是`True`。如果你错过了选项，那么它的值将是`False`。您将在[设置选项](#setting-the-action-behind-an-option)后面的动作一节中了解关于`.add_argument()`的`action`参数的更多信息。

当`long`为`True`时，第 21 行[上的`build_output()`函数向](https://realpython.com/python-return-statement/)返回详细输出，否则返回最小输出。详细的输出将包含目标目录中所有条目的大小、修改日期和名称。它使用的工具有 [`Path.stat()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.stat) 和一个 [`datetime.datetime`](https://realpython.com/python-datetime/) 对象，带有自定义的[字符串](https://realpython.com/python-strings/)格式。

继续在`sample`执行您的程序，检查`-l`选项是如何工作的:

```py
$ python ls.py -l sample/
 2609 Oct 28 14:07:04 lorem.md
 428 Oct 28 14:07:04 realpython.md
 83 Oct 28 14:07:04 hello.txt
```

新的`-l`选项允许您生成并显示关于目标目录内容的更详细的输出。

既然您已经知道了如何向 CLI 添加命令行参数和选项，那么是时候开始解析这些参数和选项了。这就是您将在下一节中探索的内容。

[*Remove ads*](/account/join/)

### 解析命令行参数和选项

解析命令行参数是任何基于`argparse`的 CLI 应用程序的另一个重要步骤。一旦解析了参数，就可以开始响应它们的值采取行动了。在您的定制`ls`命令示例中，参数解析发生在包含`args = parser.parse_args()`语句的行上。

该语句调用 [`.parse_args()`](https://docs.python.org/3/library/argparse.html#the-parse-args-method) 方法，并将其返回值赋给`args`变量。`.parse_args()`的返回值是一个 [`Namespace`](https://docs.python.org/3/library/argparse.html#the-namespace-object) 对象，包含命令行提供的所有参数和选项及其对应的值。

考虑下面的玩具例子:

>>>

```py
>>> from argparse import ArgumentParser

>>> parser = ArgumentParser()

>>> parser.add_argument("site")
_StoreAction(...)

>>> parser.add_argument("-c", "--connect", action="store_true")
_StoreTrueAction(...)

>>> args = parser.parse_args(["Real Python", "-c"])
>>> args
Namespace(site='Real Python', connect=True)

>>> args.site
'Real Python'
>>> args.connect
True
```

在命令行参数解析器上调用`.parse_args()`产生的`Namespace`对象通过使用**点符号**让您可以访问所有的输入参数、选项及其相应的值。这样，您可以检查输入参数和选项的列表，以响应用户在命令行中的选择。

您将在应用程序的主代码中使用这个`Namespace`对象。这就是您在自定义的`ls`命令示例中的`for`循环下所做的。

至此，您已经了解了创建`argparse`CLI 的主要步骤。现在，您可以花一些时间来学习如何用 Python 组织和构建 CLI 应用程序的基础知识。

### 设置您的 CLI 应用程序的布局和构建系统

在继续你的`argparse`学习冒险之前，你应该停下来想想你将如何组织你的代码和[设计](https://realpython.com/python-application-layouts/)一个 CLI 项目。首先，你应该注意以下几点:

*   你可以创建[模块和包](https://realpython.com/python-modules-packages/)来组织你的代码。
*   您可以用应用程序本身来命名 Python 应用程序的核心包。
*   您将根据特定的内容或功能来命名每个 Python 模块。
*   如果您想让这个包直接可执行，您可以在任何 Python 包中添加一个`__main__.py`模块。

记住这些想法，并考虑到[模型-视图-控制器(MVC)](https://realpython.com/the-model-view-controller-mvc-paradigm-summarized-with-legos/) 模式是构建应用程序的有效方式，您可以在设计 CLI 项目时使用以下目录结构:

```py
hello_cli/
│
├── hello_cli/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   └── model.py
│
├── tests/
│   ├── __init__.py
│   ├── test_cli.py
│   └── test_model.py
│
├── pyproject.toml
├── README.md
└── requirements.txt
```

`hello_cli/`目录是项目的根目录。在那里，您将放置以下文件:

*   [`pyproject.toml`](https://realpython.com/pypi-publish-python-package/#configure-your-package) 是一个 [TOML](https://realpython.com/python-toml/) 文件，指定了项目的**构建系统**和其他**配置**。
*   [`README.md`](https://dbader.org/blog/write-a-great-readme-for-your-github-project) 为安装和运行应用程序提供项目**描述**和**指令**。向您的项目添加一个描述性的详细的`README.md`文件是编程中的最佳实践，尤其是如果您计划将项目作为开源解决方案发布的话。
*   [`requirements.txt`](https://realpython.com/what-is-pip/#using-requirements-files) 提供了一个常规文件，列出了项目的**外部依赖**。您将使用这个文件通过使用带有`-r`选项的`pip`来自动安装依赖项。

然后是保存应用核心包的`hello_cli/`目录，它包含以下模块:

*   **`__init__.py`** 启用`hello_cli/`作为 Python **包**。
*   **`__main__.py`** 提供了应用程序的**入口点脚本**或者可执行文件。
*   **`cli.py`** 提供了应用程序的命令行界面。这个文件中的代码将在基于 MVC 的架构中扮演**视图控制器**的角色。
*   **`model.py`** 包含支持应用程序主要功能的代码。这段代码将在你的 MVC 布局中扮演**模型**的角色。

你还会有一个`tests/`包，其中包含对你的应用组件进行[单元测试](https://realpython.com/python-testing/)的文件。在这个特定的项目布局示例中，您有`test_cli.py`用于检查 CLI 功能的单元测试，还有`test_model.py`用于检查模型代码的单元测试。

`pyproject.toml`文件允许您定义应用程序的构建系统以及许多其他通用配置。下面是如何为您的示例`hello_cli`项目填写该文件的一个简单示例:

```py
# pyproject.toml [build-system] requires  =  ["setuptools>=64.0.0",  "wheel"] build-backend  =  "setuptools.build_meta" [project] name  =  "hello_cli" version  =  "0.0.1" description  =  "My awesome Hello CLI application" readme  =  "README.md" authors  =  [{  name  =  "Real Python",  email  =  "info@realpython.com"  }] [project.scripts] hello_cli  =  "hello_cli.__main__:main"
```

`[build-system]` [表头](https://realpython.com/python-toml/#tables)将`setuptools`设置为你的应用的[构建系统](https://realpython.com/pypi-publish-python-package/#build-your-package)，并指定 Python 需要安装哪些依赖项来构建你的应用。`[project]`头为你的应用程序提供了通用的元数据。当您想要[将您的应用](https://realpython.com/pypi-publish-python-package/)发布到 Python 包索引( [PyPI](https://pypi.org/) )时，这些元数据非常有用。最后，`[project.scripts]`标题定义了应用程序的入口点。

通过快速浏览布局和构建 CLI 项目，您可以继续学习`argparse`，尤其是如何定制您的命令行参数解析器。

[*Remove ads*](/account/join/)

## 定制您的命令行参数解析器

在前面的章节中，您学习了使用 Python 的`argparse`为您的程序或应用程序实现命令行接口的基础知识。您还学习了如何按照 MVC 模式组织和布局 CLI 应用程序项目。

在接下来的几节中，您将深入了解`argparse`的许多其他优秀特性。具体来说，您将学习如何在`ArgumentParser`构造函数中使用一些最有用的参数，这将允许您定制 CLI 应用程序的一般行为。

### 调整程序的帮助和使用内容

向您的 CLI 应用程序的用户提供使用说明和帮助是一种最佳实践，它将使您的用户的生活更加愉快，带来出色的[用户体验(UX)](https://en.wikipedia.org/wiki/User_experience) 。在本节中，您将了解如何利用`ArgumentParser`的一些参数来微调您的 CLI 应用程序如何向用户显示帮助和使用消息。您将学习如何:

*   设置程序的名称
*   定义程序的描述和结尾信息
*   显示参数和选项的分组帮助

首先，您将设置程序的名称，并指定该名称在帮助或用法消息的上下文中的外观。

#### 设置程序名

默认情况下，`argparse`使用`sys.argv`中的第一个值来设置程序的名称。第一项保存了您刚刚执行的 Python 文件的名称。这个文件名在使用信息中看起来很奇怪。

例如，继续运行带有`-h`选项的自定义`ls`命令:

```py
$ python ls.py -h
usage: ls.py [-h] [-l] path 
positional arguments:
 path

options:
 -h, --help  show this help message and exit
 -l, --long
```

命令输出中突出显示的一行显示了`argparse`正在使用文件名`ls.py`作为程序的名称。这看起来很奇怪，因为应用程序名称在使用信息中显示时很少包含文件扩展名。

幸运的是，您可以通过使用`prog`参数来指定程序的名称，如下面的代码片段所示:

```py
# ls.py v3

import argparse
import datetime
from pathlib import Path

parser = argparse.ArgumentParser(prog="ls") 
# ...

for entry in target_dir.iterdir():
    print(build_output(entry, long=args.long))
```

使用`prog`参数，您可以指定将在用法消息中使用的程序名。在这个例子中，您使用了`"ls"`字符串。现在，继续运行您的应用程序:

```py
$ python ls.py -h
usage: ls [-h] [-l] path 
positional arguments:
 path

options:
 -h, --help  show this help message and exit
 -l, --long
```

太好了！此输出的第一行中的应用程序使用信息显示程序名称为`ls`，而不是`ls.py`。

除了设置程序的名称，`argparse`让你定义应用程序的描述和结束信息。在下一节中，您将学习如何做到这两点。

#### 定义程序的描述和结尾消息

您还可以为您的应用程序定义一个一般描述和一个结尾或结束消息。为此，您将分别使用`description`和`epilog`参数。继续更新`ls.py`文件，在`ArgumentParser`构造函数中添加以下内容:

```py
# ls.py v4

import argparse
import datetime
from pathlib import Path

parser = argparse.ArgumentParser(
    prog="ls",
 description="List the content of a directory", epilog="Thanks for using %(prog)s! :)", )

# ...

for entry in target_dir.iterdir():
    print(build_output(entry, long=args.long))
```

在此更新中，`description`允许您为您的应用程序提供一个通用描述。此描述将显示在帮助消息的开头。`epilog`参数允许你定义一些文本作为你的应用程序的结尾或结束消息。注意，您可以使用[旧式的字符串格式化操作符(`%` )](https://realpython.com/python-string-formatting/#1-old-style-string-formatting-operator) 将`prog`参数插入到结束字符串中。

**注:**帮助消息支持`%(specifier)s`格式的**格式说明符**。这些说明符使用字符串格式化操作符`%`，而不是流行的 [f 字符串](https://realpython.com/python-f-strings/)。这是因为 f 字符串在运行时会立即用它们的值替换名称。

因此，在上面对`ArgumentParser`的调用中，将`prog`插入到`epilog`中，如果使用 f 字符串，将会失败，并出现 [`NameError`](https://realpython.com/python-traceback/#nameerror) 。

如果您再次运行该应用程序，您将得到如下输出:

```py
$ python ls.py -h
usage: ls [-h] [-l] path

List the content of a directory 
positional arguments:
 path

options:
 -h, --help  show this help message and exit
 -l, --long

Thanks for using ls! :)
```

现在，输出在用法消息之后显示描述消息，在帮助文本的末尾显示结束消息。

#### 显示参数和选项的分组帮助

**帮助小组**是`argparse`的另一个有趣的特色。它们允许您将相关的命令和参数分组，这将帮助您组织应用程序的帮助信息。要创建这些帮助组，您将使用`ArgumentParser`的 [`.add_argument_group()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument_group) 方法。

例如，考虑以下定制`ls`命令的更新版本:

```py
# ls.py v5
# ...

parser = argparse.ArgumentParser(
    prog="ls",
    description="List the content of a directory",
    epilog="Thanks for using %(prog)s! :)",
)

general = parser.add_argument_group("general output") general.add_argument("path") 
detailed = parser.add_argument_group("detailed output") detailed.add_argument("-l", "--long", action="store_true") 
args = parser.parse_args()

# ...

for entry in target_dir.iterdir():
    print(build_output(entry, long=args.long))
```

在此更新中，您将为显示常规输出的参数和选项创建一个帮助组，并为显示详细输出的参数和选项创建另一个帮助组。

**注意:**在这个具体的例子中，像这样对参数进行分组似乎是不必要的。然而，如果你的应用有几个参数和选项，那么使用帮助组可以显著改善你的用户体验。

如果您在命令行运行带有`-h`选项的应用程序，那么您将得到以下输出:

```py
python ls.py -h
usage: ls [-h] [-l] path

List the content of a directory

options:
 -h, --help  show this help message and exit

general output:
 path

detailed output:
 -l, --long

Thanks for using ls! :)
```

现在，您的应用程序的参数和选项被方便地分组在帮助消息的描述性标题下。这个简洁的功能将帮助你为你的用户提供更多的上下文，并提高他们对应用程序如何工作的理解。

[*Remove ads*](/account/join/)

### 为参数和选项提供全局设置

除了定制用法和帮助信息，`ArgumentParser`还允许你对你的 CLI 应用程序进行一些其他有趣的调整。这些调整包括:

*   为参数和选项定义全局默认值
*   从外部文件加载参数和选项
*   允许或禁止选项缩写

有时，你可能需要为你的应用程序的参数和选项指定一个单一的**全局默认值**。您可以通过在调用`ArgumentParser`构造函数时将默认值传递给 [`argument_default`](https://docs.python.org/3/library/argparse.html#argument-default) 来做到这一点。

此功能可能很少有用，因为参数和选项通常具有不同的数据类型或含义，并且很难找到满足所有要求的值。

然而，`argument_default`的一个常见用例是当您想要避免向`Namespace`对象添加参数和选项时。在这种情况下，可以使用`SUPPRESS` [常量](https://realpython.com/python-constants/)作为默认值。这个默认值将使得只有命令行提供的参数和选项最终存储在参数`Namespace`中。

例如，继续修改您的定制`ls`命令，如下面的代码片段所示:

```py
# ls.py v6

import argparse
import datetime
from pathlib import Path

parser = argparse.ArgumentParser(
    prog="ls",
    description="List the content of a directory",
    epilog="Thanks for using %(prog)s! :)",
 argument_default=argparse.SUPPRESS, )

# ...

for entry in target_dir.iterdir():
 try: long = args.long except AttributeError: long = False    print(build_output(entry, long=long))
```

通过将`SUPPRESS`传递给`ArgumentParser`构造函数，可以防止未提供的参数存储在参数`Namespace`对象中。这就是为什么你要在调用`build_output()`之前检查`-l`或者`--long`选项是否真的通过了。否则，您的代码将因`AttributeError`而中断，因为`long`不会出现在`args`中。

`ArgumentParser`的另一个很酷的特性是它允许你从外部文件中加载参数值。当您的应用程序具有很长或复杂的命令行结构，并且希望自动加载参数值时，这种可能性就很方便了。

在这种情况下，您可以将参数值存储在一个外部文件中，并要求您的程序从中加载它们。要尝试这个特性，请继续创建以下 toy CLI 应用程序:

```py
# fromfile.py

import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

parser.add_argument("one")
parser.add_argument("two")
parser.add_argument("three")

args = parser.parse_args()

print(args)
```

这里，您将`@`符号传递给`ArgumentParser`的 [`fromfile_prefix_chars`](https://docs.python.org/3/library/argparse.html#fromfile-prefix-chars) 参数。然后创建三个必须在命令行提供的参数。

现在假设您经常使用具有相同参数值集的应用程序。为了方便和简化您的工作，您可以创建一个包含所有必要参数的适当值的文件，每行一个，如下面的`args.txt`文件所示:

```py
first
second
third
```

有了这个文件，您现在可以调用您的程序，并指示它从`args.txt`文件加载值，如下面的命令运行所示:

```py
$ python fromfile.py @args.txt
Namespace(one='first', two='second', three='third')
```

在这个命令的输出中，你可以看到`argparse`已经读取了`args.txt`的内容，并依次给你的`fromfile.py`程序的每个参数赋值。所有参数及其值都成功地存储在`Namespace`对象中。

接受**缩写选项名**的能力是`argparse` CLIs 的另一个很酷的特性。默认情况下，这个特性是启用的，当您的程序有很长的选项名时，这个特性会很方便。例如，考虑下面的程序，它打印出您在命令行的`--argument-with-a-long-name`选项下指定的值:

```py
# abbreviate.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--argument-with-a-long-name")

args = parser.parse_args()

print(args.argument_with_a_long_name)
```

这个程序打印您传递的任何内容作为`--argument-with-a-long-name`选项的参数。继续运行以下命令，检查 Python `argparse`模块如何为您处理缩写:

```py
$ python abbreviate.py --argument-with-a-long-name 42
42

$ python abbreviate.py --argument 42
42

$ python abbreviate.py --a 42
42
```

这些例子展示了如何简化`--argument-with-a-long-name`选项的名称，并且仍然让应用程序正常工作。默认情况下，此功能处于启用状态。如果您想禁用它并禁止缩写，那么您可以使用 [`allow_abbrev`](https://docs.python.org/3/library/argparse.html#allow-abbrev) 参数到`ArgumentParser`:

```py
# abbreviate.py

import argparse

parser = argparse.ArgumentParser(allow_abbrev=False) 
parser.add_argument("--argument-with-a-long-name")

args = parser.parse_args()

print(args.argument_with_a_long_name)
```

将`allow_abbrev`设置为`False`会禁用命令行选项中的缩写。从现在开始，您必须提供完整的选项名称，程序才能正常工作。否则，您会得到一个错误:

```py
$ python abbreviate.py --argument-with-a-long-name 42
42

$ python abbreviate.py --argument 42
usage: abbreviate.py [-h] [--argument-with-a-long-name ...]
abbreviate.py: error: unrecognized arguments: --argument 42
```

第二个例子中的错误消息告诉您,`--argument`选项没有被识别为有效选项。要使用该选项，您需要提供它的全名。

[*Remove ads*](/account/join/)

## 微调您的命令行参数和选项

到目前为止，您已经学习了如何定制`ArgumentParser`类的几个特性来改善 CLI 的用户体验。现在你知道如何调整应用程序的用法和帮助信息，以及如何微调命令行参数和选项的一些全局方面。

在本节中，您将了解如何自定义 CLI 命令行参数和选项的其他几个功能。在这种情况下，您将使用`.add_argument()`方法及其一些最相关的参数，包括[`action`](https://docs.python.org/3/library/argparse.html#action)[`type`](https://docs.python.org/3/library/argparse.html?highlight=argparse#type)[`nargs`](https://docs.python.org/3/library/argparse.html?highlight=argparse#nargs)[`default`](https://docs.python.org/3/library/argparse.html?highlight=argparse#default)[`help`](https://docs.python.org/3/library/argparse.html?highlight=argparse#help)，以及其他一些参数。

### 设置选项后的动作

当您向命令行界面添加一个选项或标志时，您通常需要定义如何在调用`.parse_args()`得到的`Namespace`对象中存储选项的值。为此，您将对`.add_argument()`使用`action`参数。`action`参数默认为`"store"`，这意味着为当前选项提供的值将按原样存储在`Namespace`中。

`action`参数可以取几个可能值中的一个。以下是这些可能值及其含义的列表:

| 容许值 | 描述 |
| --- | --- |
| `store` | 将输入值存储到`Namespace`对象 |
| `store_const` | 当指定选项时，存储一个常数值 |
| `store_true` | 当选项被指定时存储`True` [布尔值](https://realpython.com/python-boolean/)，否则存储`False` |
| `store_false` | 指定选项时存储`False`，否则存储`True` |
| `append` | 每次提供选项时，[将当前值追加](https://realpython.com/python-append/)到[列表](https://realpython.com/python-lists-tuples/) |
| `append_const` | 每次提供选项时，将常数值追加到列表中 |
| `count` | 存储当前选项被提供的次数 |
| `version` | 显示应用程序的版本并终止执行 |

在此表中，名称中包含`_const`后缀的值要求您在调用`.add_argument()`方法时使用 [`const`](https://docs.python.org/3/library/argparse.html#const) 参数提供所需的常数值。类似地，`version`动作要求您通过将`version`参数传递给`.add_argument()`来提供应用程序的版本。您还应该注意到，只有`store`和`append`动作可以而且必须在命令行接受参数。

要尝试这些操作，您可以使用以下实现创建一个玩具应用程序:

```py
# actions.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--name", action="store"
)  # Equivalent to parser.add_argument("--name")
parser.add_argument("--pi", action="store_const", const=3.14)
parser.add_argument("--is-valid", action="store_true")
parser.add_argument("--is-invalid", action="store_false")
parser.add_argument("--item", action="append")
parser.add_argument("--repeated", action="append_const", const=42)
parser.add_argument("--add-one", action="count")
parser.add_argument(
    "--version", action="version", version="%(prog)s 0.1.0"
)

args = parser.parse_args()

print(args)
```

该程序为上面讨论的每种类型的`action`实现了一个选项。然后程序打印结果参数`Namespace`。下面总结了这些选项的工作方式:

*   `--name`将存储传递的值，无需任何进一步考虑。

*   当提供选项时,`--pi`将自动存储目标常数。

*   `--is-valid`将在提供时存储`True`，否则存储`False`。如果您需要相反的行为，在本例中使用类似于`--is-invalid`的`store_false`动作。

*   `--item`将让你创建一个所有值的列表。您必须为每个值重复该选项。在底层，`argparse`会将项目添加到一个以选项本身命名的列表中。

*   `--repeated`的工作方式与`--item`相似。然而，它总是附加相同的常量值，您必须使用`const`参数提供该常量值。

*   `--add-one`统计选项在命令行传递的次数。当您想在程序中实现几个详细级别时，这种类型的选项非常有用。例如，`-v`可以表示详细程度的第一级，`-vv`可以表示详细程度的第二级，等等。

*   `--version`显示应用的版本并立即终止执行。注意，您必须预先提供版本号，这可以在使用`.add_argument()`创建选项时通过使用`version`参数来实现。

继续运行带有以下命令结构的脚本，尝试所有这些选项:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
PS> python actions.py `
>   --name Python `
>   --pi `
>   --is-valid `
>   --is-invalid `
>   --item 1 --item 2 --item 3 `
>   --repeat --repeat --repeat `
>   --add-one --add-one --add-one
Namespace(
 name='Python',
 pi=3.14,
 is_valid=True,
 is_invalid=False,
 item=['1', '2', '3'],
 repeated=[42, 42, 42],
 add_one=3
)

PS> python actions.py --version
actions.py 0.1.0
```

```py
$ python actions.py \
    --name Python \
    --pi \
    --is-valid \
    --is-invalid \
    --item 1 --item 2 --item 3 \
    --repeat --repeat --repeat \
    --add-one --add-one --add-one
Namespace(
 name='Python',
 pi=3.14,
 is_valid=True,
 is_invalid=False,
 item=['1', '2', '3'],
 repeated=[42, 42, 42],
 add_one=3
)

$ python actions.py --version
actions.py 0.1.0
```

使用这个命令，您可以展示所有动作是如何工作的，以及它们是如何存储在最终的`Namespace`对象中的。`version`动作是您使用的最后一个动作，因为这个选项只是显示程序的版本，然后结束执行。它不会存储在`Namespace`对象中。

即使默认的动作集已经很完整了，你也可以通过子类化 [`argparse.Action`](https://docs.python.org/3/library/argparse.html#argparse.Action) 类来创建自定义的动作。如果您决定这样做，那么您必须覆盖 [`.__call__()`](https://docs.python.org/3/reference/datamodel.html#object.__call__) 方法，该方法将实例转换成可调用的对象。或者，您可以根据需要覆盖 [`.__init__()`](https://realpython.com/python-class-constructor/#object-initialization-with-__init__) 和`.format_usage()`方法。

要覆盖`.__call__()`方法，您需要确保该方法的签名包括`parser`、`namespace`、`values`和`option_string`参数。

在以下示例中，您实现了一个最小且详细的`store`操作，您可以在构建 CLI 应用程序时使用该操作:

```py
# custom_action.py

import argparse

class VerboseStore(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(f"Storing {values} in the {option_string} option...")
        setattr(namespace, self.dest, values)

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--name", action=VerboseStore)

args = parser.parse_args()

print(args)
```

在这个例子中，您定义了从`argparse.Action`继承而来的`VerboseStore`。然后重写`.__call__()`方法来打印信息性消息，并在命令行参数的名称空间中设置目标选项。最后，应用程序打印名称空间本身。

继续运行以下命令来尝试您的自定义操作:

```py
$ python custom_action.py --name Python
Storing Python in the --name option... Namespace(name='Python')
```

太好了！您的程序现在在命令行存储提供给`--name`选项的值之前打印出一条消息。像上面例子中的自定义动作允许你微调程序选项的存储方式。

为了继续微调您的`argparse`CLI，您将在下一节学习如何定制命令行参数和选项的输入值。

[*Remove ads*](/account/join/)

### 自定义参数和选项中的输入值

构建 CLI 应用程序时的另一个常见需求是自定义参数和选项在命令行接受的输入值。例如，您可能要求给定的参数接受整数值、值列表、字符串等。

默认情况下，命令行中提供的任何参数都将被视为字符串。幸运的是，`argparse`有内部机制来检查给定的参数是否是有效的整数、字符串、列表等等。

在本节中，您将学习如何定制`argparse`处理和存储输入值的方式。具体来说，您将学习如何:

*   设置参数和选项输入值的数据类型
*   在参数和选项中取多个输入值
*   为参数和选项提供默认值
*   为参数和选项定义一系列允许的输入值

首先，您将从定制您的参数和选项在命令行接受的数据类型开始。

#### 设置输入值的类型

创建`argparse`CLI 时，您可以定义在`Namespace` [对象](https://realpython.com/learning-paths/object-oriented-programming-oop-python/)中存储命令行参数和选项时要使用的类型。为此，您可以使用`.add_argument()`的`type`参数。

例如，假设您想编写一个用于划分两个[数字](https://realpython.com/python-numbers/)的示例 CLI 应用程序。该应用程序将采用两个选项，`--dividend`和`--divisor`。这些选项在命令行中只接受整数:

```py
# divide.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dividend", type=int)
parser.add_argument("--divisor", type=int)

args = parser.parse_args()

print(args.dividend / args.divisor)
```

在本例中，您将`--dividend`和`--divisor`的类型设置为 [`int`](https://realpython.com/python-numbers/#integers) 。此设置将使您的选项只接受有效的整数值作为输入。如果输入值不能被转换成`int`类型而不丢失信息，那么您将得到一个错误:

```py
$ python divide.py --dividend 42 --divisor 2
21.0

$ python divide.py --dividend "42" --divisor "2"
21.0

$ python divide.py --dividend 42 --divisor 2.0
usage: divide.py [-h] [--dividend DIVIDEND] [--divisor DIVISOR]
divide.py: error: argument --divisor: invalid int value: '2.0'

$ python divide.py --dividend 42 --divisor two
usage: divide.py [-h] [--dividend DIVIDEND] [--divisor DIVISOR]
divide.py: error: argument --divisor: invalid int value: 'two'
```

前两个示例工作正常，因为输入值是整数。第三个示例失败并出现错误，因为除数是浮点数。最后一个例子也失败了，因为`two`不是一个数值。

#### 取多个输入值

在一些 CLI 应用程序中，可能需要在参数和选项中采用多个值。默认情况下，`argparse`假设每个参数或选项都有一个值。您可以使用`.add_argument()`的 [`nargs`](https://docs.python.org/3/library/argparse.html#nargs) 参数修改此行为。

`nargs`参数告诉`argparse`底层参数可以接受零个或多个输入值，这取决于分配给`nargs`的特定值。如果您希望参数或选项接受固定数量的输入值，那么您可以将`nargs`设置为一个整数。如果你需要更灵活的行为，那么`nargs`已经满足了你，因为它也接受以下价值观:

| 容许值 | 意义 |
| --- | --- |
| `?` | 接受单个输入值，这可以是可选的 |
| `*` | 接受零个或多个输入值，这些值将存储在一个列表中 |
| `+` | 接受一个或多个输入值，这些值将存储在一个列表中 |
| `argparse.REMAINDER` | 收集命令行中剩余的所有值 |

值得注意的是，`nargs`的允许值列表对命令行参数和选项都有效。

要开始尝试`nargs`的允许值，请使用以下代码创建一个`point.py`文件:

```py
# point.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--coordinates", nargs=2)

args = parser.parse_args()

print(args)
```

在这个小应用程序中，您创建了一个名为`--coordinates`的命令行选项，它接受两个输入值，分别代表笛卡尔坐标`x`和`y`。准备好这个脚本后，继续运行以下命令:

```py
$ python point.py --coordinates 2 3
Namespace(coordinates=['2', '3'])

$ python point.py --coordinates 2
usage: point.py [-h] [--coordinates COORDINATES COORDINATES]
point.py: error: argument --coordinates: expected 2 arguments

$ python point.py --coordinates 2 3 4
usage: point.py [-h] [--coordinates COORDINATES COORDINATES]
point.py: error: unrecognized arguments: 4

$ python point.py --coordinates
usage: point.py [-h] [--coordinates COORDINATES COORDINATES]
point.py: error: argument --coordinates: expected 2 arguments
```

在第一个命令中，您将两个数字作为输入值传递给`--coordinates`。在这种情况下，程序正常工作，将值存储在`Namespace`对象的`coordinates`属性下的列表中。

在第二个示例中，您传递了一个输入值，程序失败了。错误消息告诉您应用程序需要两个参数，但您只提供了一个。第三个例子非常相似，但是在这种情况下，您提供了比所需更多的输入值。

最后一个例子也失败了，因为您根本没有提供输入值，而`--coordinates`选项需要两个值。在本例中，两个输入值是必需的。

要测试`nargs`的`*`值，假设您需要一个 CLI 应用程序，它在命令行中获取一系列数字并返回它们的总和:

```py
# sum.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("numbers", nargs="*", type=float)

args = parser.parse_args()

print(sum(args.numbers))
```

因为已经将`nargs`设置为`*`，所以`numbers`参数在命令行接受零个或多个浮点数。这个脚本是这样工作的:

```py
$ python sum.py 1 2 3
6.0

$ python sum.py 1 2 3 4 5 6
21.0

$ python sum.py
0
```

前两个命令显示`numbers`在命令行接受不确定数量的值。这些值将存储在一个以`Namespace`对象中的参数命名的列表中。如果你没有传递任何值给`sum.py`，那么对应的值列表将会是空的，总和将会是`0`。

接下来，你可以用另一个小例子试试`nargs`的`+`值。这一次，假设您需要一个在命令行接受一个或多个文件的应用程序。您可以像下面的例子一样编写这个应用程序:

```py
# files.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("files", nargs="+")

args = parser.parse_args()

print(args)
```

本例中的`files`参数将在命令行接受一个或多个值。您可以通过运行以下命令来尝试一下:

```py
$ python files.py hello.txt
Namespace(files=['hello.txt'])

$ python files.py hello.txt realpython.md README.md
Namespace(files=['hello.txt', 'realpython.md', 'README.md'])

$ python files.py
usage: files.py [-h] files [files ...]
files.py: error: the following arguments are required: files
```

前两个例子表明`files`在命令行接受不确定数量的文件。最后一个例子表明，如果不提供文件，就不能使用`files`，因为会出错。这种行为迫使您至少向`files`参数提供一个文件。

`nargs`的最终允许值是`REMAINDER`。此常数允许您捕获命令行中提供的其余值。如果你把这个值传递给`nargs`，那么底层的参数将像一个袋子一样收集所有额外的输入值。作为一个练习，你可以自己编写一个小应用程序，探索一下`REMAINDER`是如何工作的。

尽管`nargs`参数为您提供了很大的灵活性，但有时在多个命令行选项和参数中正确使用该参数是相当具有挑战性的。例如，在同一个 CLI 中，当`nargs`设置为`*`、`+`或`REMAINDER`时，很难可靠地组合参数和选项:

```py
# cooking.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("veggies", nargs="+")
parser.add_argument("fruits", nargs="*")

args = parser.parse_args()

print(args)
```

在这个例子中，`veggies`参数将接受一种或多种蔬菜，而`fruits`参数应该在命令行接受零种或多种水果。不幸的是，这个例子并不像预期的那样工作:

```py
$ python cooking.py pepper tomato apple banana
Namespace(veggies=['pepper', 'tomato', 'apple', 'banana'], fruits=[])
```

该命令的输出显示所有提供的输入值都存储在了`veggies`属性中，而`fruits`属性保存了一个空列表。发生这种情况是因为`argparse`解析器没有可靠的方法来确定哪个值属于哪个参数或选项。在这个特定的示例中，您可以通过将两个参数转换为选项来解决问题:

```py
# cooking.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--veggies", nargs="+") parser.add_argument("--fruits", nargs="*") 
args = parser.parse_args()

print(args)
```

通过这个小的更新，您可以确保解析器有一个安全的方法来解析命令行提供的值。继续运行以下命令来确认这一点:

```py
$ python cooking.py --veggies pepper tomato --fruits apple banana
Namespace(veggies=['pepper', 'tomato'], fruits=['apple', 'banana'])
```

现在每个输入值都已经存储在结果`Namespace`的正确列表中。`argparse`解析器已经使用选项名来正确解析每个提供的值。

为了避免类似于上例中讨论的问题，当试图将参数和选项与设置为`*`、`+`或`REMAINDER`的`nargs`组合时，您应该始终小心。

#### 提供默认值

`.add_argument()`方法可以接受一个 [`default`](https://docs.python.org/3/library/argparse.html#default) 参数，该参数允许您为各个参数和选项提供适当的默认值。当您需要目标参数或选项始终有一个有效值，以防止用户在命令行中不提供任何输入时，此功能会很有用。

例如，回到您的定制`ls`命令，当用户没有提供目标目录时，您需要让命令列出当前目录的内容。您可以通过将`default`设置为`"."`来实现这一点，如下面的代码所示:

```py
# ls.py v7

import argparse
import datetime
from pathlib import Path

# ...

general = parser.add_argument_group("general output")
general.add_argument("path", nargs="?", default=".") 
# ...
```

这段代码中突出显示的一行很神奇。在对`.add_argument()`的调用中，您使用带问号(`?`)的`nargs`作为它的值。您需要这样做，因为`argparse`中的所有命令行参数都是必需的，将`nargs`设置为`?`、`*`或`+`是跳过所需输入值的唯一方法。在这个具体的例子中，您使用`?`,因为您需要一个输入值或者不需要。

然后将`default`设置为代表当前工作目录的`"."`字符串。有了这些更新，你现在可以运行`ls.py`而不需要提供目标目录。它会列出默认目录的内容。要进行试验，请继续运行以下命令:

```py
$ cd sample/

$ python ../ls.py
lorem.md
realpython.md
hello.txt
```

现在，如果您没有在命令行中提供目标目录，您的定制`ls`命令会列出当前目录的内容。是不是很酷？

#### 指定允许输入值的列表

在`argparse` CLIs 中另一个有趣的可能性是，您可以为特定的参数或选项创建一个允许值的域。您可以通过使用`.add_argument()`的 [`choices`](https://docs.python.org/3/library/argparse.html#choices) 参数提供一个可接受值的列表来做到这一点。

下面是一个带有`--size`选项的小应用程序的例子，它只接受几个预定义的输入值:

```py
# size.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--size", choices=["S", "M", "L", "XL"], default="M")

args = parser.parse_args()

print(args)
```

在这个例子中，您使用`choices`参数为`--size`选项提供一个允许值的列表。此设置将导致该选项仅接受预定义的值。如果您尝试使用不在列表中的值，则会出现错误:

```py
$ python size.py --size S
Namespace(size='S')

$ python choices.py --size A
usage: choices.py [-h] [--size {S,M,L,XL}]
choices.py: error: argument --size: invalid choice: 'A'
 (choose from 'S', 'M', 'L', 'XL')
```

如果您使用允许值列表中的输入值，那么您的应用程序可以正常工作。如果您使用一个无关的值，那么应用程序将失败并出现错误。

`choices`参数可以保存允许值的列表，这些值可以是不同的数据类型。对于整数值，一个有用的技巧是使用一系列可接受的值。为此，您可以使用 [`range()`](https://realpython.com/python-range/) ，如下例所示:

```py
# weekdays.py

import argparse

my_parser = argparse.ArgumentParser()

my_parser.add_argument("--weekday", type=int, choices=range(1, 8))

args = my_parser.parse_args()

print(args)
```

在本例中，命令行提供的值将自动对照作为`choices`参数提供的`range`对象进行检查。继续运行以下命令，尝试一下这个示例:

```py
$ python days.py --weekday 2
Namespace(weekday=2)

$ python days.py --weekday 6
Namespace(weekday=6)

$ python days.py --weekday 9
usage: days.py [-h] [--weekday {1,2,3,4,5,6,7}]
days.py: error: argument --weekday: invalid choice: 9
 (choose from 1, 2, 3, 4, 5, 6, 7)
```

前两个示例工作正常，因为输入数字在允许的值范围内。然而，如果输入的数字超出了定义的范围，就像上一个例子，那么你的应用程序就会失败，显示使用和错误信息。

[*Remove ads*](/account/join/)

### 在参数和选项中提供和定制帮助消息

正如你已经知道的，`argparse`的一个伟大特性是它为你的应用程序自动生成使用和帮助信息。您可以使用任何`argparse` CLI 中默认包含的`-h`或`--help`标志来访问这些消息。

至此，您已经了解了如何为您的应用程序提供描述和 epilog 消息。在本节中，您将通过为各个命令行参数和选项提供增强的消息来继续改进应用程序的帮助和使用消息。为此，您将使用`.add_argument()`的 [`help`](https://docs.python.org/3/library/argparse.html#help) 和 [`metavar`](https://docs.python.org/3/library/argparse.html#metavar) 参数。

回到您的自定义`ls`命令，使用`-h`开关运行脚本，检查其当前输出:

```py
$ python ls.py -h
usage: ls [-h] [-l] [path]

List the content of a directory

options:
 -h, --help  show this help message and exit

general output:
 path

detailed output:
 -l, --long

Thanks for using ls! :)
```

这个输出看起来不错，是一个很好的例子，说明了`argparse`如何通过提供现成的用法和帮助消息来节省您的大量工作。

注意，只有`-h`或`--help`选项显示描述性帮助信息。相比之下，您自己的参数`path`和`-l`或`--long`不会显示帮助信息。要解决这个问题，您可以使用`help`参数。

打开您的`ls.py`并更新它，如以下代码所示:

```py
# ls.py v8

import argparse
import datetime
from pathlib import Path

# ...

general = parser.add_argument_group("general output")
general.add_argument(
    "path",
    nargs="?",
    default=".",
 help="take the path to the target directory (default: %(default)s)", )

detailed = parser.add_argument_group("detailed output")
detailed.add_argument(
    "-l",
    "--long",
    action="store_true",
 help="display detailed directory content", )

# ...
```

在这次对`ls.py`的更新中，您使用`.add_argument()`的`help`参数来为您的参数和选项提供特定的帮助消息。

**注意:**正如你已经知道的，帮助消息支持像`%(prog)s`这样的格式说明符。您可以使用`add_argument()`的大多数参数作为格式说明符。例如，`%(default)s`，`%(type)s`等等。

现在继续运行带有`-h`标志的应用程序:

```py
$ python ls.py -h
usage: ls [-h] [-l] [path]

List the content of a directory

options:
 -h, --help  show this help message and exit

general output:
 path        take the path to the target directory (default: .) 
detailed output:
 -l, --long  display detailed directory content 
Thanks for using ls! :)
```

现在，当你运行带有`-h`标志的应用时，`path`和`-l`都会显示描述性的帮助信息。请注意，`path`在它的帮助消息中包含了它的默认值，这为您的用户提供了有价值的信息。

另一个期望的特性是在你的 CLI 应用程序中有一个好的和可读的使用信息。`argparse`的默认使用信息已经很不错了。不过，你可以用`.add_argument()`的`metavar`论证稍微改进一下。

当命令行参数或选项接受输入值时，`metavar`参数就派上了用场。它允许您给这个输入值一个描述性的名称，解析器可以用它来生成帮助消息。

作为何时使用`metavar`的示例，回到您的`point.py`示例:

```py
# point.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--coordinates", nargs=2)

args = parser.parse_args()

print(args)
```

如果您使用`-h`开关从命令行运行这个应用程序，那么您将得到如下所示的输出:

```py
$ python point.py -h
usage: point.py [-h] [--coordinates COORDINATES COORDINATES] 
options:
 -h, --help            show this help message and exit
 --coordinates COORDINATES COORDINATES
```

默认情况下，`argparse`使用命令行选项的原始名称来指定它们在用法和帮助消息中对应的输入值，正如您在突出显示的行中所看到的。在这个具体的例子中，复数形式的名称`COORDINATES`可能会引起混淆。你的用户应该提供点的坐标两次吗？

您可以通过使用`metavar`参数来消除这种歧义:

```py
# point.py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--coordinates",
 nargs=2,    metavar=("X", "Y"),
 help="take the Cartesian coordinates %(metavar)s", )

args = parser.parse_args()

print(args)
```

在这个例子中，您使用一个元组作为`metavar`的值。元组包含人们通常用来指定一对笛卡尔坐标的两个坐标名称。您还为`--coordinates,`提供了一个定制的帮助消息，包括一个带有`metavar`参数的格式说明符。

如果您运行带有`-h`标志的脚本，那么您将得到以下输出:

```py
$ python coordinates.py -h
usage: coordinates.py [-h] [--coordinates X Y]

options:
 -h, --help         show this help message and exit
 --coordinates X Y  take the Cartesian coordinates ('X', 'Y')
```

现在，你的应用程序的用法和帮助信息比以前更加清晰。现在您的用户将立即知道他们需要提供两个数值，`X`和`Y`，以便`--coordinates`选项正确工作。

[*Remove ads*](/account/join/)

## 定义互斥的参数和选项组

另一个有趣的特性是，您可以将它整合到您的`argparse`CLI 中，创建互斥的参数和选项组。当参数或选项不能在同一个命令结构中共存时，这个特性就很方便了。

考虑以下 CLI 应用程序，它具有不能在同一个命令调用中共存的`--verbose`和`--silent`选项:

```py
# groups.py

import argparse

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-s", "--silent", action="store_true")

args = parser.parse_args()

print(args)
```

对`--verbose`和`--silent`使用互斥的组使得不可能在同一个命令调用中使用这两个选项:

```py
$ python groups.py -v -s
usage: groups.py [-h] (-v | -s)
groups.py: error: argument -s/--silent: not allowed with argument -v/--verbose
```

不能在同一个命令调用中指定`-v`和`-s`标志。如果您尝试这样做，那么您会得到一个错误，告诉您不允许同时使用这两个选项。

请注意，应用程序的使用消息显示，`-v`和`-s`是互斥的，使用竖线符号(`|`)来分隔它们。这种呈现选项的方式必须解释为*使用`-v`或`-s`，而不能同时解释为*。

## 向您的 CLI 添加子命令

一些命令行应用程序利用子命令来提供新的特性和功能。像 [`pip`](https://realpython.com/what-is-pip/) 、 [pyenv](https://realpython.com/intro-to-pyenv/) 、[poems](https://realpython.com/dependency-management-python-poetry/)和 [`git`](https://realpython.com/python-git-github-intro/) 这些在 Python 开发者中相当流行的应用程序，大量使用了子命令。

例如，如果您使用`--help`开关运行`pip`，那么您将获得应用程序的使用和帮助消息，其中包括子命令的完整列表:

```py
$ pip --help

Usage:
 pip <command> [options]

Commands:
 install                     Install packages.
 download                    Download packages.
 uninstall                   Uninstall packages.
 ...
```

要使用其中一个子命令，您只需将其列在应用程序名称之后。例如，以下命令将列出您在当前 Python 环境中安装的所有包:

```py
$ pip list
Package    Version
---------- -------
pip        x.y.z
setuptools x.y.z
 ...
```

在 CLI 应用程序中提供子命令是一个非常有用的特性。幸运的是，`argparse`也提供了实现这个特性所需的工具。如果你想用子命令武装你的命令行程序，那么你可以使用`ArgumentParser`的 [`.add_subparsers()`](https://docs.python.org/3/library/argparse.html#sub-commands) 方法。

作为使用`.add_subparsers()`的一个例子，假设您想要创建一个 CLI 应用程序来执行基本的算术运算，包括加、减、乘和除。您希望在应用程序的 CLI 中将这些操作作为子命令来实现。

要构建这个应用程序，首先要编写应用程序的核心功能，或者算术运算本身。然后，将相应的参数添加到应用程序的 CLI 中:

```py
 1# calc.py
 2
 3import argparse
 4
 5def add(a, b):
 6    return a + b
 7
 8def sub(a, b):
 9    return a - b
10
11def mul(a, b):
12    return a * b
13
14def div(a, b):
15    return a / b
16
17global_parser = argparse.ArgumentParser(prog="calc")
18subparsers = global_parser.add_subparsers(
19    title="subcommands", help="arithmetic operations"
20)
21
22arg_template = {
23    "dest": "operands",
24    "type": float,
25    "nargs": 2,
26    "metavar": "OPERAND",
27    "help": "a numeric value",
28}
29
30add_parser = subparsers.add_parser("add", help="add two numbers a and b")
31add_parser.add_argument(**arg_template)
32add_parser.set_defaults(func=add)
33
34sub_parser = subparsers.add_parser("sub", help="subtract two numbers a and b")
35sub_parser.add_argument(**arg_template)
36sub_parser.set_defaults(func=sub)
37
38mul_parser = subparsers.add_parser("mul", help="multiply two numbers a and b")
39mul_parser.add_argument(**arg_template)
40mul_parser.set_defaults(func=mul)
41
42div_parser = subparsers.add_parser("div", help="divide two numbers a and b")
43div_parser.add_argument(**arg_template)
44div_parser.set_defaults(func=div)
45
46args = global_parser.parse_args()
47
48print(args.func(*args.operands))
```

下面是代码的工作原理:

*   **第 5 行到第 15 行**定义了执行加、减、乘、除基本算术运算的四个函数。这些函数将提供应用程序的每个子命令背后的操作。

*   第 17 行照常定义命令行参数解析器。

*   **第 18 到 20 行**通过调用`.add_subparsers()`定义了一个子参数。在这个调用中，您提供一个标题和一条帮助消息。

*   **第 22 到 28 行**为您的命令行参数定义了一个模板。这个模板是一个字典，包含了必需参数`.add_argument()`的敏感值。每个参数将被称为`operands`，并将由两个浮点值组成。定义此模板可以让您在创建命令行参数时避免重复代码。

*   **第 30 行**给 subparser 对象添加一个解析器。这个子命令的名字是`add`，它将代表加法操作的子命令。`help`参数特别为这个解析器定义了一个帮助消息。

*   **第 31 行**使用带有参数模板的`.add_argument()`将`operands`命令行参数添加到`add`子参数中。注意，您需要使用[字典解包操作符(`**` )](https://realpython.com/iterate-through-dictionary-python/#using-the-dictionary-unpacking-operator) 从`arg_template`中提取参数模板。

*   **第 32 行**使用 [`.set_defaults()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.set_defaults) 将`add()`回调函数分配给`add`子用户或子命令。

第 34 行到第 44 行执行类似于第 30 行到第 32 行的操作，用于其余的三个子命令，`sub`、`mul`和`div`。最后，第 48 行从`args`调用`func`属性。该属性将自动调用与子命令相关联的函数。

继续运行以下命令，试用您的新 CLI 计算器:

```py
$ python calc.py add 3 8
11.0

$ python calc.py sub 15 5
10.0

$ python calc.py mul 21 2
42.0

$ python calc.py div 12 2
6.0

$ python calc.py -h
usage: calc [-h] {add,sub,mul,div} ...

options:
 -h, --help         show this help message and exit

subcommands:
 {add,sub,mul,div}  arithmetic operations
 add              add two numbers a and b
 sub              subtract two numbers a and b
 mul              multiply two numbers a and b
 div              divide two numbers a and b

$ python calc.py div -h
usage: calc div [-h] OPERAND OPERAND

positional arguments:
 OPERAND     a numeric value

options:
 -h, --help  show this help message and exit
```

酷！您所有的子命令都像预期的那样工作。它们接受两个数字，并用它们执行目标算术运算。请注意，现在您已经有了应用程序和每个子命令的用法和帮助消息。

## 处理您的 CLI 应用程序的执行如何终止

创建 CLI 应用程序时，您会发现由于错误或异常而需要终止应用程序执行的情况。在这种情况下，常见的做法是退出应用程序，同时发出一个[错误代码](https://en.wikipedia.org/wiki/Error_code)或[退出状态](https://en.wikipedia.org/wiki/Exit_status)，以便其他应用程序或操作系统可以了解应用程序因执行错误而终止。

通常，如果命令以零代码退出，那么它已经成功。同时，非零退出状态表示失败。这个系统的缺点是，虽然你有一个单一的、定义明确的方式来表示成功，但你有各种方式来表示失败，这取决于手头的问题。

不幸的是，错误代码或退出状态没有明确的标准。操作系统和编程语言使用不同的风格，包括十进制或[十六进制](https://en.wikipedia.org/wiki/Hexadecimal)数字、字母数字代码，甚至描述错误的短语。Unix 程序通常使用`2`表示命令行语法错误，使用`1`表示所有其他错误。

在 Python 中，通常使用整数值来指定 CLI 应用程序的系统退出状态。如果你的代码返回`None`，那么退出状态为零，这被认为是**成功终止**。任何非零值表示**异常终止**。大多数系统要求退出代码在从`0`到`127`的范围内，否则会产生未定义的结果。

用`argparse`构建 CLI apps 时，不需要担心返回成功操作的退出代码。然而，当你的应用程序由于一个错误而不是命令语法错误突然终止执行时，你应该返回一个适当的退出代码，在这种情况下`argparse`为你做了开箱即用的工作。

`argparse`模块，特别是`ArgumentParser`类，有两个专用的方法来在出现问题时终止应用程序:

| 方法 | 描述 |
| --- | --- |
| [T2`.exit(status=0, message=None)`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.exit) | 终止应用程序，返回指定的`status`并打印`message`(如果给定) |
| [T2`.error(message)`](https://docs.python.org/3/library/argparse.html?highlight=argparse#argparse.ArgumentParser.error) | 打印包含所提供的`message`的使用信息，并使用状态代码`2`终止应用程序 |

两种方法都直接打印到专用于错误报告的[标准错误流](https://realpython.com/python-subprocess/#the-standard-io-streams)。当您需要完全控制返回哪个状态代码时，`.exit()`方法是合适的。另一方面，`.error()`方法由`argparse`在内部用于处理命令行语法错误，但是您可以在必要和适当的时候使用它。

作为何时使用这些方法的示例，考虑对您的自定义`ls`命令的以下更新:

```py
# ls.py v9

import argparse
import datetime
from pathlib import Path

# ...

target_dir = Path(args.path)

if not target_dir.exists():
 parser.exit(1, message="The target directory doesn't exist") 
# ...
```

在检查目标目录是否存在的条件语句中，不使用`raise SystemExit(1)`，而是使用`ArgumentParser.exit()`。这使得你的代码更加关注于所选择的技术栈，也就是`argparse`框架。

要检查您的应用程序现在的行为，请继续运行以下命令:

*   [*视窗*](#windows-2)
**   [**Linux + macOS**](#linux-macos-2)*

```py
PS> python ls.py .\non_existing\
The target directory doesn't exist

PS> echo $LASTEXITCODE
1
```

```py
$ python ls.py non_existing/
The target directory doesn't exist

$ echo $?
1
```

当目标目录不存在时，应用程序立即终止执行。如果你在一个类似 Unix 的系统上，比如 Linux 或 macOS，那么你可以检查`$?` shell 变量来确认你的应用程序已经返回了`1`来通知它执行中的一个错误。如果你在 Windows 上，那么你可以检查`$LASTEXITCODE`变量的内容。

在您的 CLI 应用程序中提供一致的状态代码是一种最佳实践，它将允许您和您的用户成功地将您的应用程序集成到他们的 shell 脚本和命令管道中。

## 结论

现在您知道了什么是命令行界面，以及它的主要组件是什么，包括参数、选项和子命令。您还学习了如何使用 Python 标准库中的 **`argparse`** 模块创建全功能的 **CLI 应用程序**。

**在本教程中，您已经学会了如何:**

*   从**命令行界面**开始
*   **组织**和**用 Python 布置**一个命令行项目
*   使用 Python 的 **`argparse`** 创建**命令行界面**
*   使用`argparse`的一些**强大功能**定制 CLI 的大多数方面

作为开发人员，知道如何编写有效且直观的命令行界面是一项非常重要的技能。为你的应用程序编写好的 CLI 可以让你在与你的应用程序交互时给你的用户一个愉快的用户体验。

**源代码:** [点击这里下载源代码](https://realpython.com/bonus/command-line-interfaces-python-argparse-code/)，您将使用它来构建与`argparse`的命令行界面。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 argparse**](/courses/python-argparse-command-line-interfaces/) 构建命令行接口******************