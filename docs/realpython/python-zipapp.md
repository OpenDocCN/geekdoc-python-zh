# Python 的 zipapp:构建可执行的 Zip 应用程序

> 原文：<https://realpython.com/python-zipapp/>

Python Zip 应用程序是一个快速而又酷的选择，您可以将可执行的应用程序捆绑并分发到一个单独的**准备运行的文件**中，这将使您的最终用户体验更加愉快。如果您想了解 Python 应用程序以及如何使用标准库中的`zipapp`创建它们，那么本教程就是为您准备的。

您将能够创建 Python Zip 应用程序，作为向最终用户和客户分发您的软件产品的一种快速且可访问的方式。

在本教程中，您将学习:

*   什么是 Python Zip 应用程序
*   Zip 应用程序如何工作**内部**
*   如何用**`zipapp`****构建** Python Zip 应用
*   什么是独立的 Python Zip 应用程序以及如何创建它们
*   如何使用命令行工具手动创建 Python Zip 应用程序

 **您还将了解一些用于创建 Zip 应用程序的第三方库，它们克服了`zipapp`的一些限制。

为了更好地理解本教程，你需要知道如何[构造 Python 应用程序布局](https://realpython.com/python-application-layouts/)、[运行 Python 脚本](https://realpython.com/run-python-scripts/)、[构建 Python 包](https://realpython.com/pypi-publish-python-package/)，使用 [Python 虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)，以及使用 [`pip`](https://realpython.com/what-is-pip/) 安装和管理依赖关系。您还需要熟练使用命令行或终端。

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## Python Zip 应用入门

Python 生态系统中最具挑战性的问题之一是找到一种有效的方法来分发可执行的应用程序，如[图形用户界面(GUI)](https://realpython.com/python-pyqt-gui-calculator/) 和[命令行界面(CLI)](https://realpython.com/command-line-interfaces-python-argparse/) 程序。

编译后的编程语言，比如 [C](https://realpython.com/c-for-python-programmers/) 、 [C++](https://realpython.com/python-vs-cpp/) 、 [Go](https://golang.org/) ，可以生成你可以直接在不同操作系统和架构上运行的可执行文件。这种能力使您可以轻松地向最终用户分发软件。

然而，Python 不是那样工作的。Python 是一种[解释语言](https://docs.python.org/3/glossary.html#term-interpreted)，这意味着你需要一个合适的 Python 解释器来运行你的应用程序。没有直接的方法生成一个不需要解释器就能运行的独立的可执行文件。

有许多解决方案可以解决这个问题。你会发现诸如 [PyInstaller](https://realpython.com/pyinstaller-python/) 、 [py2exe](http://www.py2exe.org/) 、 [py2app](https://py2app.readthedocs.io/en/latest/) 、 [Nuitka](https://nuitka.net/) 等工具。这些工具允许您创建可分发给最终用户的自包含可执行应用程序。然而，设置这些工具可能是一个复杂且具有挑战性的过程。

有时候你不需要额外的复杂性。你只需要从一个脚本或者一个小程序中构建一个可执行的应用程序，这样你就可以快速的把它分发给你的终端用户。如果您的应用程序足够小，并且使用纯 Python 代码，那么使用一个 **Python Zip 应用程序**就足够了。

[*Remove ads*](/account/join/)

### 什么是 Python Zip 应用程序？

[PEP 441——改进 Python ZIP 应用程序支持](https://www.python.org/dev/peps/pep-0441/)围绕 **Python Zip 应用程序**形成了概念、术语和规范。这种类型的应用程序由一个使用 [ZIP 文件格式](https://en.wikipedia.org/wiki/ZIP_(file_format))的文件组成，其中包含 Python 可以作为程序执行的代码。这些应用程序依靠 Python 从 ZIP 文件中运行代码的能力，这些 ZIP 文件的根目录下有一个 [`__main__.py`](https://docs.python.org/3/library/__main__.html#module-__main__) 模块，它作为一个入口点脚本工作。

从版本 [2.6 和 3.0](https://bugs.python.org/issue1739468) 开始，Python 已经能够从 ZIP 文件运行脚本。实现这一目标的步骤非常简单。您只需要一个 ZIP 文件，其根目录下有一个`__main__.py`模块。然后你可以把那个文件传递给 Python，Python 把它添加到 [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path) 并把`__main__.py`作为一个程序执行。在`sys.path`中保存应用程序的档案允许你通过 Python 的[导入系统](https://realpython.com/python-import/#the-python-import-system)访问它的代码。

举个简单的例子，假设你在一个类似于 Unix 的操作系统上，比如 Linux 或者 macOS，你运行下面的命令:

```py
$ echo 'print("Hello, World!")' > __main__.py

$ zip hello.zip __main__.py
 adding: __main__.py (stored 0%)

$ python ./hello.zip
Hello, World!
```

您使用 [`echo`命令](https://en.wikipedia.org/wiki/Echo_(command))创建一个包含代码`print("Hello, World!")`的`__main__.py`文件。然后你使用 [`zip`](https://en.wikipedia.org/wiki/Info-ZIP) 命令将`__main__.py`存档到`hello.zip`。一旦你完成了这些，你就可以通过将文件名作为参数传递给`python`命令来运行`hello.zip`了。

为了完善 Python Zip 应用程序的内部结构，您需要一种方法来告诉操作系统如何执行它们。ZIP 文件格式允许您在 ZIP 存档文件的开头添加任意数据。Python Zip 应用程序利用该特性在应用程序的归档中包含一个标准的 Unix she bang 行:

```py
#!/usr/bin/env python3
```

在 Unix 系统上，这一行告诉操作系统使用哪个程序来执行手头的文件，这样您就可以直接运行文件，而无需使用`python`命令。在 Windows 系统上，Python 启动器正确理解 shebang 行并为您运行 Zip 应用程序。

即使使用 shebang 行，也可以通过将应用程序的文件名作为参数传递给`python`命令来执行 Python Zip 应用程序。

总之，要构建 Python Zip 应用程序，您需要:

*   一个使用**标准 ZIP 文件格式**并在其根包含一个 **`__main__.py`模块**的档案
*   一个可选的 **shebang 行**,指定适当的 **Python 解释器**来运行应用程序

除了`__main__.py`模块，您的应用程序的 ZIP 文件可以包含 Python [模块和包](https://realpython.com/python-modules-packages/)以及任何其他任意文件。但是，只有`.py`、`.pyc`和`.pyo`文件可以通过导入系统直接使用。换句话说，您可以将`.pyd`、`.so`和`.dll`文件打包到您的应用程序文件中，但是除非您将它们解压缩到您的文件系统中，否则您将无法使用它们。

**注意:**无法执行存储在 ZIP 文件中的`.pyd`、`.so`和`.dll`文件的代码是操作系统的限制。这个限制使得创建运送和使用`.pyd`、`.so`和`.dll`文件的 Zip 应用程序变得困难。

Python 生态系统充满了用 C 或 C++编写的有用的库和工具，以保证速度和效率。即使您可以将这些库捆绑到一个 Zip 应用程序的归档文件中，您也不能从那里直接使用它们。您需要将这个库解压缩到您的文件系统中，然后从这个新位置访问它的组件。

PEP 441 提议将`.pyz`和`.pyzw`作为 Python Zip 应用的[文件扩展名。`.pyz`扩展标识控制台或命令行应用程序，而`.pyzw`扩展指窗口或](https://www.python.org/dev/peps/pep-0441/#a-new-python-zip-application-extension) [GUI 应用程序](https://realpython.com/python-gui-tkinter/)。

在 Unix 系统上，如果您更喜欢 CLI 应用程序的简单命令名，可以删除`.pyz`扩展名。在 Windows 上，`.pyz`和`.pyzw`文件是可执行文件，因为 Python 解释器将它们注册为可执行文件。

### 为什么使用 Python Zip 应用程序？

假设你有一个程序，你的团队在他们的内部工作流程中经常使用它。该程序已经从一个单文件脚本发展成为一个拥有多个包、模块和文件的成熟应用程序。

此时，一些团队成员努力安装和设置每个新版本。他们不断要求您提供一种更快、更简单的方式来设置和运行程序。在这种情况下，您应该考虑创建一个 Python Zip 应用程序，将您的程序捆绑到一个文件中，并作为一个准备运行的应用程序分发给您的同事。

Python Zip 应用程序是发布软件的绝佳选择，您必须将这些软件作为单个可执行文件进行分发。这也是一种使用非正式渠道分发软件的便捷方式，例如通过计算机网络发送或托管在 FTP 服务器上。

Python Zip 应用程序是以现成的格式打包和分发 Python 应用程序的方便快捷的方式，可以让您的最终用户的生活更加愉快。

[*Remove ads*](/account/join/)

### 如何构建 Python Zip 应用程序？

正如您已经了解到的，Python Zip 应用程序由一个标准 Zip 文件组成，该文件包含一个`__main__.py`模块，该模块作为应用程序的入口点。当您运行应用程序时，Python 会自动将其容器(ZIP 文件本身)添加到`sys.path`中，这样`__main__.py`就可以从塑造应用程序的模块和包中导入对象。

要构建 Python Zip 应用程序，您可以运行以下常规步骤:

1.  创建包含`__main__.py`模块的应用程序源目录。
2.  压缩应用程序的源目录。
3.  添加一个可选的 Unix shebang 行来定义运行应用程序的解释器。
4.  使应用程序的 ZIP 文件可执行。此步骤仅适用于类似 Unix 的操作系统。

这些步骤非常简单，运行起来也很快。有了它们，如果您拥有所需的工具和知识，您可以在几分钟内手动构建一个 Python Zip 应用程序。然而，Python [标准库](https://docs.python.org/3/library/index.html)为您提供了更方便、更快捷的解决方案。

PEP 441 提议在标准库中增加一个名为 [`zipapp`](https://docs.python.org/3/library/zipapp.html) 的新模块。这个模块方便了 Zip 应用程序的创建，它从 [Python 3.5](https://docs.python.org/3/whatsnew/3.5.html#whatsnew-zipapp) 开始就可用了。

在本教程中，您将关注使用`zipapp`创建 Python Zip 应用程序。然而，您还将学习如何使用不同的工具手动运行整个系列的步骤。这些额外的知识可以帮助您更深入地理解创建 Python Zip 应用程序的整个过程。如果您使用的是低于 3.5 的 Python 版本，这也会很有帮助。

## 设置 Python Zip 应用程序

到目前为止，您已经了解了什么是 Python Zip 应用程序，如何构建它们，为什么使用它们，以及创建它们时需要遵循的步骤。你已经准备好开始建造你自己的了。不过，首先，您需要有一个用于 Python Zip 应用程序的功能性应用程序或脚本。

对于本教程，您将使用一个名为 [`reader`](https://github.com/realpython/reader#real-python-feed-reader) 的示例应用程序，它是一个最小的 [web 提要](https://en.wikipedia.org/wiki/Web_feed)阅读器，从 [*真实 Python 提要*中读取最新的文章和资源。](https://realpython.com/contact/#rss-atom-feed)

接下来，您应该将`reader`的存储库克隆到您的本地机器上。在您选择的工作目录中打开命令行，并运行以下命令:

```py
$ git clone https://github.com/realpython/reader.git
```

该命令将`reader`存储库的全部内容下载到当前目录下的`reader/`文件夹中。

**注意:**如果你不熟悉 [Git](https://git-scm.com/) 和 [GitHub](https://github.com/) ，请查看[Git 和 GitHub 介绍给 Python 开发者](https://realpython.com/python-git-github-intro/)。

一旦克隆了存储库，就需要安装应用程序的依赖项。首先，你应该创建一个 Python [虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)。继续运行以下命令:

```py
$ cd reader/
$ python3 -m venv ./venv
$ source venv/bin/activate
```

这些命令在`reader/`目录中创建和激活一个新的 Python 虚拟环境，该目录是`reader`项目的根目录。

**注意:**要在 Windows 上创建和激活虚拟环境，您可以运行以下命令:

```py
C:\> python -m venv venv
C:\> venv\Scripts\activate.bat
```

如果你在一个不同的平台上，那么你可能需要查看 Python 的官方文档关于[创建虚拟环境](https://docs.python.org/3/library/venv.html#creating-virtual-environments)。

现在您可以使用 [`pip`](https://realpython.com/what-is-pip/) 安装`reader`的依赖项:

```py
(venv) $ python -m pip install feedparser html2text importlib_resources
```

运行上面的命令将在您的活动 Python 虚拟环境中安装应用程序的所有依赖项。

**注:**自 Python 3.7 起， [`importlib_resources`](https://importlib-resources.readthedocs.io/en/latest/) 在标准库中可用为 [`importlib.resources`](https://docs.python.org/3/library/importlib.html#module-importlib.resources) 。所以，如果你使用的是高于或等于 3.7 的版本，你不需要安装这个库。只需在定义了`reader`包的`__init__.py`文件中修改相应的导入。

下面是一个使用`reader`从 *Real Python* 获取最新[文章](https://realpython.com/)、[课程](https://realpython.com/courses/)、[播客剧集](https://realpython.com/podcasts/rpp/)和其他学习资源的例子:

```py
(venv) $ python -m reader
The latest tutorials from Real Python (https://realpython.com/)
 0 The Django Template Language: Tags and Filters
 1 Pass by Reference in Python: Best Practices
 2 Using the "and" Boolean Operator in Python
 ...
```

由于`reader`在提要中列出了 30 个最新的学习资源，因此您的输出会有所不同。每个学习资源都有一个 ID 号。要从这些学习资源中获取一个项目的内容，您可以将相应的 ID 号作为命令行参数传递给`reader`:

```py
(venv) $ python -m reader 2
Using the "and" Boolean Operator in Python

Python has three Boolean operators, or **logical operators** : `and`, `or`,
and `not`. You can use them to check if certain conditions are met before
deciding the execution path your programs will follow. In this tutorial,
you'll learn about the `and` operator and how to use it in your code.
 ...
```

该命令使用 Python 中的“and”布尔运算符将文章[的部分内容打印到使用](https://realpython.com/python-and-operator/) [Markdown](https://en.wikipedia.org/wiki/Markdown) 文本格式的屏幕上。您可以通过更改 ID 号来阅读任何可用的内容。

**注意:**`reader`如何工作的细节与本教程无关。如果你对实现感兴趣，那么看看[如何向 PyPI](https://realpython.com/pypi-publish-python-package/) 发布开源 Python 包。特别是，你可以阅读名为[的部分，快速浏览代码](https://realpython.com/pypi-publish-python-package/#a-quick-look-at-the-code)。

要从`reader`存储库创建一个 Zip 应用程序，您将主要使用`reader/`文件夹。该文件夹具有以下结构:

```py
reader/
|
├── config.cfg
├── feed.py
├── __init__.py
├── __main__.py
└── viewer.py
```

从`reader/`目录中要注意的最重要的事实是，它包括一个`__main__.py`文件。这个文件使您能够像以前一样使用`python -m reader`命令来执行这个包。

拥有一个`__main__.py`文件提供了创建 Python Zip 应用程序所需的入口点脚本。在这个例子中，`__main__.py`文件在`reader`包中。如果您使用这个目录结构创建您的 Zip 应用程序，那么您的应用程序将不会运行，因为`__main__.py`将无法从`reader`导入对象。

要解决这个问题，将`reader`包复制到一个名为`realpython/`的外部目录，并将`__main__.py`文件放在其根目录下。然后删除运行`python -m reader`产生的`__pycache__/`文件夹，就像你之前做的那样。您最终应该得到以下目录结构:

```py
realpython/ │
├── reader/ │   ├── __init__.py
│   ├── config.cfg
│   ├── feed.py
│   └── viewer.py
│
└── __main__.py
```

有了这个新的目录结构，您就可以用`zipapp`创建您的第一个 Python Zip 应用程序了。这就是你在下一节要做的。

[*Remove ads*](/account/join/)

## 用`zipapp` 构建 Python Zip 应用程序

要创建您的第一个 Python Zip 应用程序，您将使用`zipapp`。这个模块实现了一个用户友好的[命令行界面](https://docs.python.org/3/library/zipapp.html#command-line-interface)，它提供了用一个命令构建一个完整的 Zip 应用程序所需的选项。你也可以通过模块的 [Python API](https://docs.python.org/3/library/zipapp.html#python-api) 从你的代码中使用`zipapp`，它主要由一个单一的函数组成。

在接下来的两节中，您将了解使用`zipapp`构建 Zip 应用程序的两种方法。

### 从命令行使用`zipapp`

`zipapp`的命令行界面简化了将 Python 应用程序打包成 ZIP 文件的过程。在内部，`zipapp`通过运行您之前学习的步骤，从源代码创建一个 Zip 应用程序。

要从命令行运行`zipapp`，您应该使用以下命令语法:

```py
$ python -m zipapp <source> [OPTIONS]
```

如果`source`是一个目录，那么这个命令从该目录的内容创建一个 Zip 应用程序。如果`source`是一个文件，那么这个文件应该是一个包含应用程序代码的 ZIP 文件。然后，输入 ZIP 文件的内容被复制到目标应用程序档案中。

下面是`zipapp`接受的命令行选项的总结:

| 选择 | 描述 |
| --- | --- |
| `-o <output_filename>`或`--output=<output_filename>` | 将 Zip 应用程序写入名为`output_filename`的文件。此选项使用您提供的输出文件名。如果你不提供这个选项，那么`zipapp`使用带有`.pyz`扩展名的`source`的名字。 |
| `-p <interpreter>`或`--python=<interpreter>` | 将 shebang 行添加到应用程序的存档中。如果你在一个 [POSIX](https://en.wikipedia.org/wiki/POSIX) 系统上，那么`zipapp`使应用程序的归档文件可执行。如果您不提供此选项，那么您的应用程序的存档将不会有 shebang，也不会是可执行的。 |
| `-m <main_function>`或`--main=<main_function>` | 生成并写入一个适当的执行`main_function`的`__main__.py`文件。`main_function`参数的形式应该是`"package.module:callable"`。如果你已经有一个`__main__.py`模块，你不需要这个选项。 |
| `-c`或`--compress` | 使用 [Deflate](https://en.wikipedia.org/wiki/Deflate) 压缩方法压缩`source`的内容。默认情况下，`zipapp`只存储`source`的内容而不压缩它，这可以让你的应用程序运行得更快。 |

此表提供了对`zipapp`命令行选项的简要描述。有关每个选项的具体行为的更多细节，请查看[官方文档](https://docs.python.org/3/library/zipapp.html#command-line-interface)。

现在您已经知道了从命令行使用`zipapp`的基本知识，是时候构建`reader` Zip 应用程序了。返回终端窗口，运行以下命令:

```py
(venv) $ python -m zipapp realpython/ \
-o realpython.pyz \
-p "/usr/bin/env python3"
```

在这个命令中，您将`realpython/`目录设置为 Zip 应用程序的源。使用`-o`选项，您可以为应用程序的档案提供一个名称`realpython.pyz`。最后，`-p`选项让您设置解释器，`zipapp`将使用它来构建 shebang 行。

就是这样！现在，您将在当前目录中拥有一个`realpython.pyz`文件。稍后您将学习如何执行该文件。

为了展示`zipapp`的`-m`和`--main`命令行选项，假设您决定更改`reader`项目布局并将`__main__.py`重命名为`cli.py`，同时将文件移回`reader`包。继续创建您的`realpython/`目录的副本，并进行建议的更改。之后，`realpython/`的文案应该是这样的:

```py
realpython_copy/
│
└── reader/
    ├── __init__.py
 ├── cli.py    ├── config.cfg
    ├── feed.py
    └── viewer.py
```

目前，您的应用程序的源目录没有一个`__main__.py`模块。`zipapp`的`-m`命令行选项允许你自动生成:

```py
$ python -m zipapp realpython_copy/ \
-o realpython.pyz \
-p "/usr/bin/env python3" \
-m "reader.cli:main"
```

该命令使用带有`"reader.cli:main"`的`-m`选项作为参数。这个输入值告诉`zipapp`Zip 应用程序可调用的入口点是`reader`包中`cli.py`模块的 [`main()`](https://realpython.com/python-main-function/) 。

生成的`__main__.py`文件包含以下内容:

```py
# -*- coding: utf-8 -*-
import reader.cli
reader.cli.main()
```

然后，这个`__main__.py`文件与您的应用程序源代码一起打包成一个名为`realpython.pyz`的 ZIP 存档文件。

[*Remove ads*](/account/join/)

### 使用 Python 代码中的`zipapp`

Python 的`zipapp`也有一个[应用编程接口(API)](https://en.wikipedia.org/wiki/API) ，你可以从你的 Python 代码中使用它。这个 API 主要由一个名为 [`create_archive()`](https://docs.python.org/3/library/zipapp.html#zipapp.create_archive) 的函数组成。使用该函数，您可以快速创建 Python Zip 应用程序:

>>>

```py
>>> import zipapp

>>> zipapp.create_archive(
...     source="realpython/",
...     target="realpython.pyz",
...     interpreter="/usr/bin/env python3",
... )
```

这个对`create_archive()`的调用需要一个名为`source`的第一个参数，它代表您的 Zip 应用程序的源代码。第二个参数，`target`，保存应用程序存档的文件名。最后，`interpreter`保存解释器来构建并作为 shebang 行添加到应用程序的 ZIP 存档中。

以下是`create_archive()`可以提出的论点的总结:

*   **`source`** 可以带以下对象:
    *   现有源目录的基于字符串的路径
    *   引用现有源目录的类似于路径的对象
    *   现有 Zip 应用程序归档的基于字符串的路径
    *   引用现有 Zip 应用程序档案的类似路径的对象
    *   一个类似于[文件的对象](https://docs.python.org/3/glossary.html#term-file-object)被打开用于读取并指向一个现有的 Zip 应用程序档案
*   **`target`** 接受以下对象:
    *   目标 Zip 应用程序文件的基于字符串的路径
    *   目标 Zip 应用程序文件的类似路径的对象
*   **`interpreter`** 指定一个 Python 解释器，作为 shebang 行写在生成的应用程序归档文件的开头。省略此参数会导致没有 shebang 行，也没有应用程序的执行权限。
*   **`main`** 指定`zipapp`将用作目标归档入口点的可调用文件的名称。当您没有一个`__main__.py`文件时，您为`main`提供一个值。
*   **`filter`** 采用一个[布尔值函数](https://en.wikipedia.org/wiki/Boolean-valued_function)，如果源目录中的给定文件应该被添加到最终的 Zip 应用程序文件中，该函数应该返回`True`。
*   **`compressed`** 接受一个决定是否要压缩源文件的布尔值。

这些参数中的大多数在`zipapp`的命令行界面中都有等价的选项。上面的例子只使用了前三个参数。根据您的具体需要，您也可以使用其他参数。

## 运行 Python Zip 应用程序

到目前为止，您已经学习了如何从命令行和 Python 代码使用`zipapp`创建 Python Zip 应用程序。现在是时候运行你的`realpython.pyz`应用程序了，以确保它能正常工作。

如果您在一个类似 Unix 的系统上，那么您可以通过执行以下命令来运行您的应用程序:

```py
(venv) $ ./realpython.pyz
The latest tutorials from Real Python (https://realpython.com/)
 0 The Django Template Language: Tags and Filters
 1 Pass by Reference in Python: Best Practices
 2 Using the "and" Boolean Operator in Python
 ...
```

酷！有用！现在，您有了一个可以快速与朋友和同事共享的应用程序文件。

您不再需要从命令行调用 Python 来运行应用程序。因为您的 Zip 应用程序档案文件在开头有一个 shebang 行，所以操作系统将自动使用您的活动 Python 解释器来运行目标档案文件的内容。

**注意:**为了让您的应用程序运行，您需要在 Python 环境中安装所有必需的依赖项。否则，你会得到一个`ImportError`。

如果您在 Windows 上，那么您的 Python 安装应该已经注册了`.pyz`和`.pyzw`文件，并且应该能够运行它们:

```py
C:\> .\realpython.pyz
The latest tutorials from Real Python (https://realpython.com/)
 0 The Django Template Language: Tags and Filters
 1 Pass by Reference in Python: Best Practices
 2 Using the "and" Boolean Operator in Python
 ...
```

本教程中使用的`reader`应用程序有一个命令行界面，所以从命令行或终端窗口运行它是有意义的。然而，如果你有一个图形用户界面应用程序，那么你将能够从你最喜欢的文件管理器中运行它，就像你通常运行可执行程序一样。

同样，您可以通过调用适当的 Python 解释器来执行任何 Zip 应用程序，并将应用程序的文件名作为参数:

```py
$ python3 realpython.pyz
The latest tutorials from Real Python (https://realpython.com/)
 0 The Django Template Language: Tags and Filters
 1 Pass by Reference in Python: Best Practices
 2 Using the "and" Boolean Operator in Python
 ...
```

在这个例子中，您使用系统 Python 3.x 安装来运行`realpython.pyz`。如果您的系统上有许多 Python 版本，那么您可能需要更具体一些，使用类似于`python3.9 realpython.pyz`的命令。

注意，无论您使用什么解释器，您都需要安装应用程序的依赖项。否则，您的应用程序将会失败。不满足依赖关系是 Python Zip 应用程序的常见问题。要解决这种恼人的情况，您可以创建一个独立的应用程序，这是下一节的主题。

[*Remove ads*](/account/join/)

## 使用`zipapp` 创建独立的 Python Zip 应用程序

您还可以使用`zipapp`来创建独立的 Python Zip 应用程序。这种类型的应用程序将其所有依赖项捆绑到应用程序的 ZIP 文件中。这样，您的最终用户只需要一个合适的 Python 解释器来运行应用程序。他们不需要担心依赖性。

要创建一个独立的 Python Zip 应用程序，首先需要使用`pip`将其依赖项安装到源目录中。继续创建一个名为`realpython_sa/`的`realpython/`目录的副本。然后运行以下命令来安装应用程序的依赖项:

```py
(venv) $ python -m pip install feedparser html2text importlib_resources \
--target realpython_sa/
```

这个命令使用带有`--target`选项的`pip install`来安装`reader`的所有依赖项。`pip`的文档说这个选项允许你将包安装到一个*目标*目录中。在本例中，该目录必须是您的应用程序的源目录，`realpython_sa/`。

**注意:**如果你的应用程序有一个`requirements.txt`文件，那么你可以通过一个快捷方式来安装依赖项。

您可以改为运行以下命令:

```py
(venv) $ python -m pip install \
-r requirements.txt \ --target app_directory/
```

使用这个命令，您可以将应用程序的`requirements.txt`文件中列出的所有依赖项安装到`app_directory/`文件夹中。

一旦将`reader`的依赖项安装到`realpython_sa/`中，就可以随意删除`pip`创建的`*.dist-info`目录。这些目录包含几个带有元数据的文件，`pip`用它们来管理相应的包。既然你不再需要这些信息，你可以把它们扔掉。

这个过程的最后一步是像往常一样使用`zipapp`构建 Zip 应用程序:

```py
(venv) $ python -m zipapp realpython_sa/ \
-p "/usr/bin/env python3" \
-o realpython_sa.pyz \
-c
```

该命令在`realpython_sa.pyz`中生成一个独立的 Python Zip 应用程序。要运行这个应用程序，您的最终用户只需要在他们的机器上安装一个合适的 Python 3 解释器。与常规的 Zip 应用程序相比，这种应用程序的优势在于您的最终用户不需要安装任何依赖项来运行应用程序。来吧，试一试！

在上面的例子中，您使用了`zipapp`的`-c`选项来压缩`realpython_sa/`的内容。对于具有许多依赖项、占用大量磁盘空间的应用程序来说，这个选项相当方便。

## 手动创建 Python Zip 应用程序

正如您已经了解到的，从 Python 3.5 开始，`zipapp`就在标准库中可用。如果您使用的是低于这个版本的 Python，那么您仍然可以手动构建您的 Python Zip 应用程序，而不需要`zipapp`。

在接下来的两节中，您将学习如何使用 Python 标准库中的 [`zipfile`](https://realpython.com/python-zipfile/) 创建一个 Zip 应用程序。您还将学习如何使用一些命令行工具来完成相同的任务。

### 使用 Python 的`zipfile`

您已经有了包含`reader`应用程序源文件的`realpython/`目录。从该目录手动构建 Python Zip 应用程序的下一步是将其归档到一个 Zip 文件中。为此，你可以使用`zipfile`。这个模块提供了创建、读取、写入、添加和列出 ZIP 文件内容的便利工具。

下面的代码展示了如何使用`zipfile.ZipFile`和一些其他工具创建`reader` Zip 应用程序。例如，代码依靠 [`pathlib`](https://realpython.com/python-pathlib/) 和 [`stat`](https://docs.python.org/3/library/stat.html#module-stat) 来读取源目录的内容，并对结果文件设置执行权限:

```py
# build_app.py

import pathlib
import stat
import zipfile

app_source = pathlib.Path("realpython/")
app_filename = pathlib.Path("realpython.pyz")

with open(app_filename, "wb") as app_file:
 # 1\. Prepend a shebang line    shebang_line = b"#!/usr/bin/env python3\n"
    app_file.write(shebang_line)

 # 2\. Zip the app's source    with zipfile.ZipFile(app_file, "w") as zip_app:
        for file in app_source.rglob("*"):
            member_file = file.relative_to(app_source)
            zip_app.write(file, member_file)

# 3\. Make the app executable (POSIX systems only) current_mode = app_filename.stat().st_mode
exec_mode = stat.S_IEXEC
app_filename.chmod(current_mode | exec_mode)
```

这段代码运行所需的三个步骤，最终得到一个成熟的 Python Zip 应用程序。第一步是在应用程序文件中添加一个 shebang 行。它使用 [`with`语句](https://realpython.com/python-with-statement/)中的 [`open()`](https://realpython.com/read-write-files-python/#opening-and-closing-a-file-in-python) 创建一个文件对象(`app_file`)来处理应用程序。然后调用`.write()`在`app_file`的开头写 shebang 行。

**注意:**如果你在 Windows 上，你应该在 [UTF-8](https://en.wikipedia.org/wiki/UTF-8) 中编码 shebang 行。如果你在一个 [POSIX](https://en.wikipedia.org/wiki/POSIX) 系统上，比如 Linux 和 macOS，你应该用 [`sys.getfilesystemencoding()`](https://docs.python.org/3/library/sys.html#sys.getfilesystemencoding) 返回的任何文件系统编码对它进行编码。

第二步使用嵌套的`with`语句中的 [`ZipFile`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile) 压缩应用程序的源目录内容。`for`循环使用`pathlib.Path.rglob()`遍历`realpython/`中的文件，并将它们写入`zip_app`。注意`.rglob()`通过目标文件夹`app_source`递归搜索文件和目录。

最终 ZIP 存档中每个文件的文件名`member_file`需要相对于应用程序的源目录，以确保应用程序 ZIP 文件的内部结构与源文件的结构`realpython/`相匹配。这就是为什么你在上面的例子中使用`pathlib.Path.relative_to()`。

最后，第三步使用 [`pathlib.Path.chmod()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.chmod) 使应用程序的文件可执行。为此，首先使用 [`pathlib.Path.stat()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.stat) 获取文件的当前模式，然后使用[按位](https://realpython.com/python-bitwise-operators/)或运算符(`|`)将该模式与 [`stat.S_IEXEC`](https://docs.python.org/3/library/stat.html?highlight=stat#stat.S_IEXEC) 结合起来。注意，这个步骤只对 POSIX 系统有影响。

运行完这些步骤后，您的`realpython.pyz`应用程序就可以运行了。请从命令行尝试一下。

[*Remove ads*](/account/join/)

### 使用 Unix 命令行工具

如果您使用的是类 Unix 系统，比如 Linux 和 macOS，那么您也可以在命令行中使用特定的工具来运行上一节中的三个步骤。例如，您可以使用`zip`命令压缩应用程序源目录的内容:

```py
$ cd realpython/
$ zip -r ../realpython.zip *
```

在这个例子中，你先将 [`cd`](https://en.wikipedia.org/wiki/Cd_(command)) 放入`realpython/`目录。然后使用带有`-r`选项的`zip`命令将`realpython/`的内容压缩到`realpython.zip`中。该选项递归遍历目标目录。

**注意:**另一个选择是从命令行使用 Python 的`zipfile`。

为此，从`realpython/`目录外运行以下命令:

```py
$ python -m zipfile --create realpython.zip realpython/*
```

`zipfile`的`--create`命令行选项允许您从源目录创建一个 ZIP 文件。追加到`realpython/`目录的星号(`*`)告诉`zipfile`将该目录的内容放在生成的 ZIP 文件的根目录下。

下一步是将 shebang 行添加到 ZIP 文件`realpython.zip`，并将其保存为`realpython.pyz`。为此，您可以在管道中使用`echo`和 [`cat`](https://en.wikipedia.org/wiki/Cat_(Unix)) 命令:

```py
$ cd ..
$ echo '#!/usr/bin/env python3' | cat - realpython.zip > realpython.pyz
```

`cd ..`命令让你退出`realpython/`。`echo`命令将`'#!/usr/bin/env python3'`发送到标准输出。管道字符(`|`)将标准输出的内容传递给`cat`命令。然后`cat`将标准输出(`-`)与`realpython.zip`的内容连接起来。最后，大于号(`>`)将`cat`输出重定向到`realpython.pyz`文件。

最后，您可能希望使用 [`chmod`](https://en.wikipedia.org/wiki/Chmod) 命令使应用程序的文件可执行:

```py
$ chmod +x realpython.pyz
```

这里，`chmod`给`realpython.pyz`增加了执行权限(`+x`)。现在，您已经准备好再次运行您的应用程序，这可以像往常一样从命令行完成。

## 使用第三方工具创建 Python 应用程序

在 Python 生态系统中，您会发现一些第三方库的工作方式与`zipapp`类似。它们提供了更多的特性，对探索这些特性很有帮助。在本节中，您将了解其中两个第三方库: [`pex`](https://pex.readthedocs.io/en/latest/index.html) 和 [`shiv`](https://shiv.readthedocs.io/en/latest/index.html) 。

项目提供了一个创建 PEX 文件的工具。PEX 代表 **Python 可执行文件**，是一种存储自包含可执行 Python 虚拟环境的文件格式。`pex`工具用一个 shebang 行和一个`__main__.py`模块将这些环境打包成 ZIP 文件，这允许您直接执行生成的 PEX 文件。`pex`工具是对 PEP 441 中概述的思想的扩展。

要用`pex`创建一个可执行的应用程序，首先需要安装它:

```py
(venv) $ python -m pip install pex
(venv) $ pex --help
pex [-o OUTPUT.PEX] [options] [-- arg1 arg2 ...]

pex builds a PEX (Python Executable) file based on the given specifications:
sources, requirements, their dependencies and other options.
Command-line options can be provided in one or more files by prefixing the
filenames with an @ symbol. These files must contain one argument per line.
 ...
```

`pex`工具提供了丰富的选项，允许你微调你的 PEX 文件。以下命令显示了如何为`reader`项目创建一个 PEX 文件:

```py
(venv) $ pex realpython-reader -c realpython -o realpython.pex
```

这个命令在你的当前目录中创建`realpython.pex`。这个文件是用于`reader`的 Python 可执行文件。注意，`pex`处理`reader`的安装和所有来自 [PyPI](https://pypi.org/) 的依赖项。在 PyPI 上可以获得名为`realpython-reader`的`reader`项目，这就是为什么您使用这个名称作为`pex`的第一个参数。

`-c`选项允许您定义应用程序将使用哪个控制台脚本。在这种情况下，控制台脚本是`reader`的`setup.py`文件中定义的`realpython`。`-o`选项指定输出文件。像往常一样，您可以从命令行执行`./realpython.pex`来运行应用程序。

由于`.pex`文件的内容在执行前被解压缩，PEX 应用程序解决了`zipapp`应用程序的限制，允许您执行来自`.pyd`、`.so`和`.dll`文件的代码。

需要注意的最后一个细节是`pex`在生成的 PEX 文件中创建和打包了一个 Python 虚拟环境。这种行为使您的 Zip 应用程序比用`zipapp`创建的常规应用程序大很多。

在本节中，您将学习的第二个工具是`shiv`。它是一个命令行工具，用于构建自包含的 Python Zip 应用程序，如 PEP 441 中所述。与`zipapp`相比，`shiv`的优势在于`shiv`会自动将应用程序的所有依赖项包含在最终文档中，并使它们在 Python 的[模块搜索路径](https://realpython.com/python-modules-packages/#the-module-search-path)中可用。

要使用`shiv`，您需要从 PyPI 安装它:

```py
(venv) $ python -m pip install shiv
(venv) $ shiv --help
Usage: shiv [OPTIONS] [PIP_ARGS]...

 Shiv is a command line utility for building fully self-contained Python
 zipapps as outlined in PEP 441, but with all their dependencies included!
 ...
```

`--help`选项显示了一个完整的使用信息，您可以通过检查来快速了解`shiv`是如何工作的。

要用`shiv`构建 Python Zip 应用程序，您需要一个可安装的 Python 应用程序，带有一个`setup.py`或`pyproject.toml`文件。幸运的是，GitHub 最初的`reader`项目满足了这个要求。回到包含克隆的`reader/`文件夹的目录，运行以下命令:

```py
(venv) $ shiv -c realpython \
-o realpython.pyz reader/ \
-p "/usr/bin/env python3"
```

像`pex`工具一样，`shiv`有一个`-c`选项来定义应用程序的控制台脚本。`-o`和`-p`选项允许您分别提供输出文件名和合适的 Python 解释器。

**注意:**上面的命令按预期工作。然而，`shiv` (0.5.2)的当前版本让`pip`显示一条关于它如何构建包的反对消息。由于`shiv` [直接接受`pip`参数](https://shiv.readthedocs.io/en/latest/cli-reference.html#cmdoption-shiv-arg-PIP_ARGS)，所以可以将 [`--use-feature=in-tree-build`](https://pip.pypa.io/en/stable/cli/pip/#cmdoption-use-feature) 放在命令的末尾，这样`shiv`就可以安全地使用`pip`。

与`zipapp`不同，`shiv`允许您使用存储在应用程序档案中的`.pyd`、`.so`和`.dll`文件。为此，`shiv`在归档中包含了一个特殊的引导功能。这个函数将应用程序的依赖项解压到您的主文件夹的`.shiv/`目录中，并将它们添加到 Python 的`sys.path`目录中。

这个特性允许您创建独立的应用程序，其中包括部分用 C 和 C++编写的库，以提高速度和效率，例如 [NumPy](https://realpython.com/numpy-tutorial/) 。

[*Remove ads*](/account/join/)

## 结论

拥有一种快速有效的方法来分发 Python 可执行应用程序，可以在满足最终用户需求方面发挥重要作用。 **Python Zip applications** 为您捆绑和分发现成的应用程序提供了一个有效且可访问的解决方案。您可以使用 Python 标准库中的 [`zipapp`](https://docs.python.org/3/library/zipapp.html) 来快速创建自己的可执行 Zip 应用程序，并将它们传递给最终用户。

**在本教程中，您学习了:**

*   什么是 Python Zip 应用程序
*   Zip 应用程序如何工作**内部**
*   如何用 **`zipapp`** 构建自己的 Python Zip 应用
*   什么是**独立 Zip 应用**以及如何使用`pip`和`zipapp`创建它们
*   如何使用命令行工具手动创建 Python Zip 应用程序

 **有了这些知识，您就可以快速创建 Python Zip 应用程序，作为向最终用户分发 Python 程序和脚本的便捷方式。***********