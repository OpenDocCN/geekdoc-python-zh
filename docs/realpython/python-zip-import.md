# Python Zip 导入:快速分发模块和包

> 原文：<https://realpython.com/python-zip-import/>

Python 允许你直接通过 **Zip imports** 从 ZIP 文件导入代码。这个有趣的内置特性使您能够压缩 Python 代码以供分发。如果您经常使用 Zip 文件中的 Python 代码，ZIP 导入也会有所帮助。在这两种情况下，学习创建可导入的 ZIP 文件并从中导入代码将是一项很有价值的技能。

即使您的日常工作流程不涉及包含 Python 代码的 Zip 文件，您仍然可以通过本教程探索 ZIP 导入来学习一些有趣的新技能。

在本教程中，您将学习:

*   什么是 **Zip 导入**
*   何时在代码中使用 Zip 导入
*   如何用`zipfile`创建**可导入的压缩文件**
*   如何使您的 ZIP 文件对**导入代码**可用

您还将学习如何使用`zipimport`模块从 ZIP 文件中动态导入代码，而无需将它们添加到 Python 的模块搜索路径中。为此，您将编写一个从 ZIP 文件加载 Python 代码的最小插件系统。

为了从本教程中获得最大收益，你应该事先了解 Python 的[导入系统](https://realpython.com/python-import/)是如何工作的。你还应该知道用[`zipfile`](https://realpython.com/python-zipfile/)[操作 ZIP 文件的基本知识，用](https://realpython.com/working-with-files-in-python/)操作文件，使用 [`with`语句](https://realpython.com/python-with-statement/)。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 了解 Python Zip 导入

从 Python [2.3](https://docs.python.org/3/whatsnew/2.3.html#pep-273-importing-modules-from-zip-archives) 开始，可以从 [ZIP 文件](https://realpython.com/python-zipfile/#what-is-a-zip-file)里面导入[模块和包](https://realpython.com/python-modules-packages/)。这个特性被称为 **Zip imports** ，当您需要将一个完整的包作为单个文件分发时，这是非常有用的，这是它最常见的用例。

PEP 273 引入了 Zip 导入作为内置特性。这个特性被 Python 社区广泛认为是必备的，因为分发几个独立的`.py`、`.pyc`和`.pyo`文件并不总是合适和有效的。

Zip 导入可以简化共享和分发代码的过程，这样您的同事和最终用户就不必四处摸索，试图将文件提取到正确的位置来让代码工作。

**注意:**从 [Python 3.5](https://docs.python.org/3/whatsnew/3.5.html#pep-488-elimination-of-pyo-files) 开始，`.pyo`文件扩展名不再使用。详见 [PEP 488](https://www.python.org/dev/peps/pep-0488/) 。

[PEP 302](https://www.python.org/dev/peps/pep-0302/) 增加了一系列的**导入[挂钩](https://en.wikipedia.org/wiki/Hooking)** ，为 Zip 导入提供内置支持。如果你想从一个 ZIP 文件中导入模块和包，那么你只需要这个文件出现在 Python 的[模块搜索路径](https://realpython.com/python-modules-packages/#the-module-search-path)中。

模块搜索路径是目录和 ZIP 文件的列表。它住在 [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path) 。当您在代码中运行 [`import`](https://realpython.com/python-import/) 语句时，Python 会自动搜索列表中的项目。

在接下来的几节中，您将学习如何使用不同的 Python 工具和技术创建准备导入的 ZIP 文件。您还将了解一些将这些文件添加到当前 Python 的模块搜索路径中的方法。最后，您将深入研究`zipimport`，它是在幕后支持 Zip 导入特性的模块。

[*Remove ads*](/account/join/)

## 创建您自己的可导入 ZIP 文件

Zip 导入允许您将组织在几个模块和包中的代码作为单个文件快速分发。在创建**可导入的 ZIP 文件**时，Python 已经帮你搞定了。来自[标准库](https://docs.python.org/3/library/index.html)的 [`zipfile`](https://docs.python.org/3/library/zipfile.html) 模块包含一个名为 [`ZipFile`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile) 的类，用于操作 ZIP 文件。它还包含了一个更专业的类，叫做 [`PyZipFile`](https://docs.python.org/3.9/library/zipfile.html#zipfile.PyZipFile) ，可以方便地创建可导入的 ZIP 文件。

`PyZipFile`让您快速高效地将 Python 代码捆绑到 ZIP 文件中。该类继承自`ZipFile`，因此它共享同一个基本接口。但是，这些类别之间有两个主要区别:

1.  `PyZipFile`的初始化器带有一个名为`optimize`的可选参数，它允许你在归档之前通过编译成[字节码](https://docs.python.org/3/glossary.html#term-bytecode)来优化 Python 代码。
2.  `PyZipFile`类提供了一个名为 [`.writepy()`](https://docs.python.org/3/library/zipfile.html#pyzipfile-objects) 的方法，该方法接受 Python 模块或包作为参数，并将其添加到目标 ZIP 文件中。

如果`optimize`是其默认值`-1`，那么输入的`.py`文件会自动编译成`.pyc`文件，然后添加到目标档案中。为什么会这样？通过跳过编译步骤，打包`.pyc`文件而不是原始的`.py`文件使得导入过程更加有效。在接下来的章节中，您将了解到关于这个主题的更多信息。

在接下来的两节中，您将亲自动手创建自己的包含模块和包的可导入 ZIP 文件。

### 将 Python 模块捆绑成 ZIP 文件

在这一节中，您将使用`PyZipFile.writepy()`将一个`.py`文件编译成字节码，并将生成的`.pyc`文件添加到一个 ZIP 存档中。要试用`.writepy()`，假设您有一个`hello.py`模块:

```py
"""Print a greeting message."""
# hello.py

def greet(name="World"):
    print(f"Hello, {name}! Welcome to Real Python!")
```

这个模块定义了一个名为`greet()`的[函数](https://realpython.com/defining-your-own-python-function/)，它将`name`作为参数，[将](https://realpython.com/python-print/)友好的问候信息打印到屏幕上。

现在假设您想将这个模块打包成一个 ZIP 文件，以便以后导入。为此，您可以运行以下代码:

>>>

```py
>>> import zipfile

>>> with zipfile.PyZipFile("hello.zip", mode="w") as zip_module:
...     zip_module.writepy("hello.py")
...

>>> with zipfile.PyZipFile("hello.zip", mode="r") as zip_module:
...     zip_module.printdir()
...
File Name                                             Modified             Size
hello.pyc                                      2021-10-18 05:40:04          313
```

运行这段代码后，您将在当前工作目录中拥有一个`hello.zip`文件。对`zip_module`上的`.writepy()`的调用自动将`hello.py`编译成`hello.pyc`，并存储在底层的 ZIP 文件`hello.zip`中。这就是为什么 [`.printdir()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.printdir) 显示`hello.pyc`而不是你原来的`hello.py`文件。这种自动编译确保了高效的导入过程。

**注意:**`PyZipFile`类默认不压缩你的 Python 模块和包。它只是将它们存储在一个 ZIP 文件容器中。如果你想压缩你的源文件，你需要通过`PyZipFile`的`compression`参数显式地提供一个压缩方法。目前，Python 支持 [Deflate](https://en.wikipedia.org/wiki/Deflate) 、 [bzip2](https://en.wikipedia.org/wiki/Bzip2) 和 [LZMA](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm) 压缩方法。

在本教程中，您将依赖于默认值`compression`、[、`ZIP_STORED`、](https://docs.python.org/3/library/zipfile.html#zipfile.ZIP_STORED)，这意味着您的源文件不会被压缩。压缩源文件会影响导入操作的性能，您将在本教程的后面部分了解到这一点。

您也可以使用任何常规的[文件归档器](https://en.wikipedia.org/wiki/File_archiver)手动将`.py`和`.pyc`文件打包成 ZIP 文件。如果生成的档案包含没有相应的`.pyc`文件的`.py`文件，那么 Python 将在您第一次从特定的 ZIP 文件导入时编译它们。

Python 不会修改底层的 ZIP 文件来添加新编译的`.pyc`文件。所以下次运行导入时，Python 会再次编译代码。这种行为会使导入过程变慢。

您还可以将一个目录作为第一个参数传递给`.writepy()`。如果输入目录不是 Python 包，那么该方法扫描它寻找`.py`文件，将它们编译成`.pyc`文件，并将这些`.pyc`文件添加到目标 ZIP 文件的顶层。扫描步骤不是递归的，这意味着不扫描子目录中的源文件。

您可以通过将`PyZipFile`的`optimize`参数设置为以下值之一来进一步调整编译过程:

| 价值 | 最佳化 |
| --- | --- |
| `0` | 不执行任何优化 |
| `1` | 删除 [`assert`语句](https://realpython.com/python-assert-statement/) |
| `2` | 删除`assert`语句和[文档字符串](https://realpython.com/documenting-python-code/) |

有了这些值，当`.writepy()`在归档之前将`.py`文件编译成`.pyc`文件时，您可以微调您想要使用的优化级别。

到目前为止，您已经学习了如何将一个或多个模块捆绑到一个 ZIP 文件中。在日常编码中，您可能还需要压缩一个完整的 Python 包。您将在下一节中学习如何做到这一点。

[*Remove ads*](/account/join/)

### 将 Python 包打包成 ZIP 文件

还可以通过使用`PyZipFile`及其`.writepy()`方法将 Python 包捆绑到 ZIP 文件中。正如您已经了解到的，如果您将一个常规目录作为第一个参数传递给`.writepy()`，那么该方法将扫描目录中的`.py`文件，编译它们，并将相应的`.pyc`文件添加到结果 ZIP 文件中。

另一方面，如果输入目录是一个 Python 包，那么`.writepy()`编译所有的`.py`文件，并将它们添加到 ZIP 文件中，保持包的内部结构。

要使用 Python 包来尝试`.writepy()`，创建一个新的`hello/`目录，并将您的`hello.py`文件复制到其中。然后添加一个空的`__init__.py`模块，把目录变成一个包。您最终应该得到以下结构:

```py
hello/
|
├── __init__.py
└── hello.py
```

现在假设您想要将这个包打包成一个 ZIP 文件，以便分发。如果是这种情况，那么您可以运行以下代码:

>>>

```py
>>> import zipfile

>>> with zipfile.PyZipFile("hello_pkg.zip", mode="w") as zip_pkg:
...     zip_pkg.writepy("hello")
...

>>> with zipfile.PyZipFile("hello_pkg.zip", mode="r") as zip_pkg:
...     zip_pkg.printdir()
...
File Name                                             Modified             Size
hello/__init__.pyc                             2021-10-18 05:56:00          110
hello/hello.pyc                                2021-10-18 05:56:00          319
```

对`.writepy()`的调用以`hello`包为参数，在其中搜索`.py`文件，编译成`.pyc`文件，最后添加到目标 ZIP 文件中，保持相同的包结构。

### 了解 Zip 导入的局限性

当您使用 Zip 文件分发 Python 代码时，您需要考虑 ZIP 导入的一些限制:

*   无法加载**动态文件**，如`.pyd`、`.dll`、`.so`、**。**
***   从 **`.py`文件**中导入代码意味着**性能妥协**。*   如果**解压缩库**不可用，从**压缩文件**导入代码将会失败。*

*您可以在 ZIP 存档中包含任何类型的文件。然而，当您的用户从这些档案中导入代码时，只读取了`.py`、`.pyw`、`.pyc`和`.pyo`文件。从动态文件中导入代码是不可能的，比如`.pyd`、`.dll`和`.so`，如果它们存在于 ZIP 文件中。比如，你不能从 ZIP 存档中加载用 [C](https://realpython.com/c-for-python-programmers/) 编写的共享库和扩展模块。

您可以通过从 ZIP 文件中提取动态模块，将它们写入文件系统，然后加载它们的代码来解决这个限制。然而，这意味着您需要创建临时文件并处理可能的错误和安全风险，这可能会使事情变得复杂。

正如您在本教程前面所学的，Zip 导入也可能意味着性能的降低。如果您的档案包含`.py`模块，那么 Python 将编译它们以满足导入。但是，它不会保存相应的`.pyc`文件。这种行为可能会降低导入操作的性能。

最后，如果你需要从一个压缩的 ZIP 文件中导入代码，那么 [`zlib`](https://docs.python.org/3/library/zlib.html) 必须在你的工作环境中可用以进行解压缩。如果这个库不可用，从压缩的归档文件中导入代码会失败，并显示一个丢失的`zlib`消息。此外，解压缩步骤会给导入过程增加额外的性能开销。出于这些原因，您将在本教程中使用未压缩的 ZIP 文件。

## 从 ZIP 文件导入 Python 代码

到目前为止，您已经学会了如何创建自己的可导入 ZIP 文件以供分发。现在假设您在另一端，并且您正在获得带有 Python 模块和包的 ZIP 文件。如何从它们那里导入代码呢？在本节中，您将获得这个问题的答案，并了解如何使 ZIP 文件可用于导入其内容。

为了让 Python 从 ZIP 文件导入代码，该文件必须在 Python 的模块搜索路径中可用，该路径存储在`sys.path`中。这个模块级变量包含一个由指定模块搜索路径的[字符串](https://realpython.com/python-strings/)组成的[列表](https://realpython.com/python-lists-tuples/)。`path`的内容包括:

*   包含您正在运行的脚本的目录
*   当前目录，如果你已经交互式地运行了解释器
*   [`PYTHONPATH`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH) 环境变量中的目录，如果设置的话
*   取决于您的特定 Python 安装的目录列表
*   该目录中列出了任意路径的配置文件(`.pth`文件)

下表指出了几种将 ZIP 文件添加到`sys.path`的方法:

| [计]选项 | 目标代码或解释程序 |
| --- | --- |
| `list.insert()`、`list.append()`和`list.extend()`方法 | 您正在编写和运行的 Python 代码 |
| `PYTHONPATH`环境变量 | 您系统上运行的每个 Python 解释器 |
| 一个 Python 路径配置文件，或`.pth`文件 | 包含`.pth`文件的 Python 解释器 |

在接下来的部分中，您将探索这三种向`sys.path`添加项目的方法，这样您就可以使您的 ZIP 文件可用于导入它们的内容。

[*Remove ads*](/account/join/)

### 动态使用`sys.path`进行 Zip 导入

因为`sys.path`是一个`list`对象，所以可以通过使用常规的`list`方法从 Python 代码中操纵它。一般来说，要向`list`对象添加新项目，可以使用`.insert()`、[、](https://realpython.com/python-append/)或`.extend()`。

通常，您将使用`.insert(0, item)`从您的 Python 代码向`sys.path`添加新项目。以这种方式调用`.insert()`会在列表的开头插入`item`，确保新添加的条目优先于已有的条目。在名字冲突可能发生的时候，让`item`在开头使你能够[隐藏](https://en.wikipedia.org/wiki/Variable_shadowing)现有的模块和包。

现在假设您需要将包含您的`hello.py`模块的`hello.zip`文件添加到您当前 Python 的`sys.path`中。在这种情况下，您可以运行下面示例中的代码。注意，为了在您的机器上运行这个例子，您需要提供到`hello.zip`的正确路径:

>>>

```py
>>> import sys

>>> # Insert the hello.zip into sys.path
>>> sys.path.insert(0, "/path/to/hello.zip")

>>> sys.path[0]
'/path/to/hello.zip'

>>> # Import and use the code
>>> import hello

>>> hello.greet("Pythonista")
Hello, Pythonista! Welcome to Real Python!
```

一旦你将`hello.zip`的路径添加到你的`sys.path`中，那么你就可以从`hello.py`中导入对象，就像对待任何常规模块一样。

如果像`hello_pkg.zip`一样，您的 ZIP 文件包含 Python 包，那么您也可以将它添加到`sys.path`中。在这种情况下，导入应该是包相关的:

>>>

```py
>>> import sys

>>> sys.path.insert(0, "/path/to/hello_pkg.zip")

>>> from hello import hello

>>> hello.greet("Pythonista")
Hello, Pythonista! Welcome to Real Python!
```

因为您的代码现在在一个包中，所以您需要从`hello`包中导入`hello`模块。然后您可以像往常一样访问`greet()`功能。

向`sys.path`添加项目的另一个选项是使用`.append()`。此方法将单个对象作为参数，并将其添加到基础列表的末尾。重启您的 Python 交互式会话，并运行提供`hello.zip`路径的代码:

>>>

```py
>>> import sys

>>> sys.path.append("/path/to/hello.zip")

>>> # The hello.zip file is at the end of sys.path
>>> sys.path[-1] '/path/to/hello.zip'

>>> from hello import greet
>>> greet("Pythonista")
Hello, Pythonista! Welcome to Real Python!
```

这种技术的工作原理类似于使用`.insert()`。然而，ZIP 文件的路径现在位于`sys.path`的末尾。如果列表中前面的任何一项包含一个名为`hello.py`的模块，那么 Python 将从该模块导入，而不是从您新添加的`hello.py`模块导入。

你也可以循环使用`.append()`来添加几个文件到`sys.path`，或者你可以只使用`.extend()`。该方法接受 iterable 项，并将其内容添加到基础列表的末尾。和`.append()`一样，记住`.extend()`会把你的文件添加到`sys.path`的末尾，所以现有的名字可以隐藏你的 ZIP 文件中的模块和包。

### 使用`PYTHONPATH`进行系统范围的 Zip 导入

在某些情况下，您可能需要一个给定的 ZIP 文件，以便从计算机上运行的任何脚本或程序中导入其内容。在这些情况下，您可以使用`PYTHONPATH`环境变量让 Python 在您运行解释器时自动将您的档案加载到`sys.path`中。

`PYTHONPATH`使用与 [`PATH`](https://realpython.com/add-python-to-path/) 环境变量相同的格式，由 [`os.pathsep`](https://docs.python.org/3/library/os.html#os.pathsep) 分隔的目录路径列表。在 [Unix](https://en.wikipedia.org/wiki/Unix) 系统上，比如 Linux 和 macOS，这个函数返回一个冒号(`:`)，而在 Windows 上，它返回一个分号(`;`)。

例如，如果您在 Linux 或 macOS 上，那么您可以通过运行以下命令将您的`hello.zip`文件添加到`PYTHONPATH`:

```py
$ export PYTHONPATH="$PYTHONPATH:/path/to/hello.zip"
```

该命令将`/path/to/hello.zip`添加到当前的`PYTHONPATH`中，并导出它，以便它在当前的终端会话中可用。

**注意:**上面的命令导出了一个定制版本的`PYTHONPATH`，其中包含了到`hello.zip`的路径。该变量的自定义版本仅在当前命令行会话中可用，一旦关闭该会话，该版本将会丢失。

如果您正在运行 [Bash](https://en.wikipedia.org/wiki/Bash_(Unix_shell)) 作为您当前的 [shell](https://en.wikipedia.org/wiki/Unix_shell) ，那么您可以通过将以下代码添加到您的`.bashrc`文件中，使这个自定义版本的`PYTHONPATH`可用于您的所有命令行会话:

```py
# .bashrc

if [ -f /path/to/hello.zip ]; then
    export PYTHONPATH="$PYTHONPATH:/path/to/hello.zip"
fi
```

这段代码检查`hello.zip`是否存在于您的文件系统中。如果是，那么它将文件添加到`PYTHONPATH`变量并导出它。因为每次启动新的命令行实例时，Bash 都会运行这个文件，所以定制的`PYTHONPATH`将在每个会话中可用。

现在您可以发出`python`命令来运行解释器。一旦你到达那里，像往常一样检查`sys.path`的内容:

>>>

```py
>>> import sys

>>> sys.path
[..., '/path/to/hello.zip', ...]
```

酷！您的`hello.zip`文件在列表中。从这一点开始，您将能够像在上一节中一样从`hello.py`导入对象。来吧，试一试！

在上面的输出中需要注意的重要一点是，你的`hello.zip`文件不在`sys.path`的开头，这意味着根据 Python 如何处理其[模块搜索路径](https://docs.python.org/3/tutorial/modules.html#the-module-search-path)，较早出现的同名模块将优先于你的`hello`模块。

要在 Windows 系统上向`PYTHONPATH`添加项目，您可以在`cmd.exe`窗口中执行命令:

```py
C:\> set PYTHONPATH=%PYTHONPATH%;C:\path\to\hello.zip
```

该命令将`C:\path\to\hello.zip`添加到 Windows 机器上`PYTHONPATH`变量的当前内容中。要检查它，在同一个命令提示符会话中运行 Python 解释器，并像以前一样查看`sys.path`的内容。

**注意:**同样，您用上面的命令设置的`PYTHONPATH`变量将只在您当前的终端会话中可用。要在 Windows 上永久设置`PYTHONPATH`变量，学习[如何在 Windows](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages) 中添加 PYTHONPATH

将目录和 ZIP 文件添加到`PYTHONPATH`环境变量中，可以让您在终端会话下运行的任何 Python 解释器都可以使用这些条目。最后，需要注意的是 Python 会忽略`PYTHONPATH`中列出的不存在的目录和 ZIP 文件，所以请密切关注。

[*Remove ads*](/account/join/)

### 使用`.pth`文件进行首选范围的 zip 导入

有时，只有在运行特定的 Python 解释器时，您可能希望从给定的 ZIP 文件中导入代码。当您的项目使用该 ZIP 文件中的代码，并且您不希望该代码可用于您的其他项目时，这是非常有用的。

Python 的**路径配置文件**允许你用自定义的目录和 ZIP 文件来扩展给定解释器的`sys.path`。

路径配置文件使用`.pth`文件扩展名，可以保存目录和 ZIP 文件的路径列表，每行一个。每次运行提供`.pth`文件的 Python 解释器时，这个路径列表都会被添加到`sys.path`中。

Python 的`.pth`文件有一个简单明了的格式:

*   每行必须包含一个路径条目。
*   空行和以数字符号(`#`)开头的行被跳过。
*   执行以`import`开头的行。

一旦你有了一个合适的`.pth`文件，你需要把它复制到一个**站点目录**中，这样 Python 就可以找到它并加载它的内容。要获得当前 Python 环境的站点目录，可以从 [`site`](https://docs.python.org/3/library/site.html) 模块中调用 [`getusersitepackages()`](https://docs.python.org/3/library/site.html#site.getusersitepackages) 。如果您在当前机器上没有管理员权限，那么您可以使用位于 [`site.USER_SITE`](https://docs.python.org/3/library/site.html#site.USER_SITE) 的用户站点目录。

**注意:**用户网站目录可能不在您的个人文件夹中。如果这是您的情况，那么请按照所需的路径结构随意创建它。

例如，以下命令为 Ubuntu 上的全系统 Python 3 解释器创建了一个`hello.pth`路径配置文件:

```py
$ sudo nano /usr/lib/python3/dist-packages/hello.pth
```

该命令创建`hello.pth`，使用 [GNU nano](https://en.wikipedia.org/wiki/GNU_nano) 文本编辑器作为`root`。在那里，输入你的`hello.zip`文件的路径。按 `Ctrl` + `X` ，然后按 `Y` ，最后按 `Enter` 保存文件。现在，当您再次启动系统 Python 解释器时，这个 ZIP 文件将在`sys.path`中可用:

>>>

```py
>>> import sys

>>> sys.path
[..., '/path/to/hello.zip', ...]
```

就是这样！从这一点开始，只要使用系统范围的 Python 解释器，就可以从`hello.py`导入对象。

同样，当 Python 读取和加载给定的`.pth`文件的内容时，不存在的目录和 ZIP 文件不会被添加到`sys.path`中。最后，`.pth`文件中的重复条目只添加一次到`sys.path`。

## 探索 Python 的`zipimport`:Zip 导入背后的工具

你已经在不知不觉中使用了标准库中的 [`zipimport`](https://docs.python.org/3/library/zipimport.html#module-zipimport) 模块。在幕后，当一个`sys.path`项包含一个 ZIP 文件的路径时，Python 的内置导入机制会自动使用这个模块。在这一节中，您将通过一个实际的例子学习`zipimport`是如何工作的，以及如何在您的代码中显式地使用它。

### 了解`zipimport`的基础知识

`zipimport`的主要成分是 [`zipimporter`](https://docs.python.org/3/library/zipimport.html#zipimport.zipimporter) 。这个类将 ZIP 文件的路径作为参数，并创建一个导入器实例。下面是一个如何使用`zipimporter`及其一些属性和方法的例子:

>>>

```py
>>> from zipimport import zipimporter

>>> importer = zipimporter("/path/to/hello.zip")

>>> importer.is_package("hello")
False

>>> importer.get_filename("hello")
'/path/to/hello.zip/hello.pyc'

>>> hello = importer.load_module("hello")
>>> hello.__file__
'/path/to/hello.zip/hello.pyc'

>>> hello.greet("Pythonista")
Hello, Pythonista! Welcome to Real Python!
```

在这个例子中，首先从`zipimport`导入`zipimporter`。然后您创建一个带有您的`hello.zip`文件路径的`zipimporter`实例。

`zipimporter`类提供了几个有用的属性和方法。例如，如果输入名称是一个包，则 [`.is_package()`](https://docs.python.org/3/library/zipimport.html#zipimport.zipimporter.is_package) 返回`True`，否则返回`False`。 [`.get_filename()`](https://docs.python.org/3/library/zipimport.html#zipimport.zipimporter.get_filename) 方法返回归档文件中给定模块的路径( [`.__file__`](https://docs.python.org/3/reference/import.html#file__) )。

如果您想将模块的名称放入当前的[名称空间](https://realpython.com/python-namespaces-scope/)，那么您可以使用`.load_module()`，它返回对输入模块的引用。有了这个引用，您就可以像往常一样从模块中访问任何代码对象。

[*Remove ads*](/account/join/)

### 用`zipimport` 构建一个插件系统

如上所述，Python 内部使用`zipimport`从 ZIP 文件加载代码。您还了解了本模块提供的工具，您可以在一些实际的编码情况下使用。例如，假设您想要实现一个定制的插件系统，其中每个插件都位于自己的 ZIP 文件中。您的代码应该在给定的文件夹中搜索 ZIP 文件，并自动导入插件的功能。

要实际体验这个例子，您将实现两个玩具插件，它们接受一条消息和一个标题，并在您的默认 web 浏览器和一个 [Tkinter](https://realpython.com/python-gui-tkinter/) 消息框中显示它们。每个插件都应该在自己的目录中，在一个叫做`plugin.py`的模块中。这个模块应该实现插件的功能，并提供一个 [`main()`](https://realpython.com/python-main-function/) 函数作为插件的入口点。

继续创建一个名为`web_message/`的文件夹，其中包含一个`plugin.py`文件。在您最喜欢的[代码编辑器或 IDE](https://realpython.com/python-ides-code-editors-guide/) 中打开文件，并为 web 浏览器插件键入以下代码:

```py
"""A plugin that displays a message using webbrowser."""
# web_message/plugin.py

import tempfile
import webbrowser

def main(text, title="Alert"):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False
    ) as home:
        html = f"""
 <html>
 <head>
 <title>{title}</title>
 </head>
 <body>
 <h1>
  {text} </h1>
 </body>
 </html>
 """
        home.write(html)
        path = "file://" + home.name
    webbrowser.open(path)
```

这段代码中的`main()`函数接受一条`text`消息和一个窗口`title`。然后在一个`with`语句中创建一个 [`NamedTemporaryFile`](https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile) 。该文件将包含一个在页面上显示`title`和`text`的最小 HTML 文档。要在默认的 web 浏览器中打开这个文件，可以使用`webbrowser.open()`。

下一个插件提供了类似的功能，但是使用了`Tkinter`工具包。这个插件的代码也应该存在于一个名为`plugin.py`的模块中。您可以将该模块放在文件系统中一个名为`tk_message/`的目录下:

```py
"""A plugin that displays a message using Tkinter."""
# tk_message/plugin.py

import tkinter
from tkinter import messagebox

def main(text, title="Alert"):
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showinfo(title, text)
```

遵循与网络浏览器插件相同的模式，`main()`使用`text`和`title`。在这种情况下，该函数创建一个 [`Tk`](https://docs.python.org/3/library/tkinter.html#tkinter.Tk) 实例来保存插件的顶层窗口。但是，您不需要显示那个窗口，只需要一个消息框。所以，你使用`.withdraw()`来隐藏根窗口，然后调用`messagebox`上的`.showinfo()`来显示一个带有输入`text`和`title`的对话框。

现在您需要将每个插件打包到它自己的 ZIP 文件中。为此，在包含`web_message/`和`tk_message/`文件夹的目录中启动一个 Python [交互会话](https://realpython.com/interacting-with-python/)，并运行以下代码:

>>>

```py
>>> import zipfile

>>> plugins = ("web_message", "tk_message")

>>> for plugin in plugins:
...     with zipfile.PyZipFile(f"{plugin}.zip", mode="w") as zip_plugin:
...         zip_plugin.writepy(plugin)
...
```

下一步是为你的插件系统创建一个根文件夹。该文件夹必须包含一个`plugins/`目录，其中包含新创建的 ZIP 文件。您的目录应该是这样的:

```py
rp_plugins/
│
├── plugins/
│   │
│   ├── tk_message.zip
│   └── web_message.zip
│
└── main.py
```

在`main.py`中，您将为您的插件系统放置客户端代码。继续用下面的代码填充`main.py`:

```py
 1# main.py
 2
 3import zipimport
 4from pathlib import Path
 5
 6def load_plugins(path):
 7    plugins = []
 8    for zip_plugin in path.glob("*.zip"):
 9        importer = zipimport.zipimporter(zip_plugin)
10        plugin_module = importer.load_module("plugin")
11        plugins.append(getattr(plugin_module, "main"))
12    return plugins
13
14if __name__ == "__main__":
15    path = Path("plugins/")
16    plugins = load_plugins(path)
17    for plugin in plugins:
18        plugin("Hello, World!", "Greeting!")
```

下面是这段代码的逐行工作方式:

*   **第 3 行**导入`zipimport`从相应的 ZIP 文件中动态加载你的插件。
*   **第 4 行**导入 [`pathlib`](https://realpython.com/python-pathlib/) 来管理系统路径。
*   **第 6 行**定义了`load_plugins()`，它获取包含插件档案的目录的路径。
*   第 7 行创建一个空列表来保存当前的插件。
*   **第 8 行**定义了一个 [`for`循环](https://realpython.com/python-for-loop/)，它遍历插件目录中的`.zip`文件。
*   **第 9 行**为系统中的每个插件创建一个`zipimporter`实例。
*   **第 10 行**从每个插件的 ZIP 文件中加载`plugin`模块。
*   **第 11 行**将每个插件的`main()`函数添加到`plugins`列表中。
*   **第 12 行**T3】将`plugins`列表返回给调用者。

第 14 到 18 行调用`load_plugins()`来生成当前可用插件列表，并循环执行它们。

如果您从命令行运行`main.py`脚本，那么您首先会得到一个 Tkinter 消息框，显示`Hello, World!`消息和`Greeting!`标题。关闭该窗口后，您的 web 浏览器将在新页面上显示相同的消息和标题。来吧，试一试！

## 结论

Python 可以直接从 ZIP 文件导入代码，如果它们在模块搜索路径中可用的话。这个特性被称为 **Zip 导入**。您可以利用 Zip 导入将模块和包捆绑到一个归档文件中，这样您就可以快速有效地将它们分发给最终用户。

如果您经常将 Python 代码捆绑到 ZIP 文件中，并且需要在日常任务中使用这些代码，那么您也可以利用 Zip 导入。

**在本教程中，您学习了:**

*   什么是 **Zip 导入**
*   何时以及如何使用 Zip 导入
*   如何用`zipfile`构建**可导入的压缩文件**
*   如何使 ZIP 文件对**导入机制**可用

您还编写了一个关于如何用`zipimport`构建一个最小插件系统的实践示例。通过这个例子，您学习了如何用 Python 从 ZIP 文件中动态导入代码。*******