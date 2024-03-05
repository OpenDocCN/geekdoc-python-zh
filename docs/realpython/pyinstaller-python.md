# 使用 PyInstaller 轻松发布 Python 应用程序

> 原文:# t0]https://realython . com/pyinstaller-python/

你是否嫉妒开发人员构建一个可执行程序并轻松地将其发布给用户？如果你的用户不需要安装任何东西就可以**运行你的应用程序，这不是很好吗？这就是梦想，而 [PyInstaller](https://pyinstaller.readthedocs.io/en/stable/) 是在 Python 生态系统中实现梦想的一种方式。**

关于如何[建立虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)、[管理依赖关系](https://realpython.com/courses/managing-python-dependencies/)、[避免依赖陷阱](https://realpython.com/dependency-pitfalls/)以及[发布到 PyPI](https://realpython.com/pypi-publish-python-package/) 有无数的教程，这在你创建 Python 库时很有用。对于构建 Python 应用程序的**开发者来说，信息少得多**。本教程是为那些想将应用程序分发给用户的开发人员编写的，这些用户可能是也可能不是 Python 开发人员。

在本教程中，您将学习以下内容:

*   PyInstaller 如何简化应用程序分发
*   如何在自己的项目中使用 PyInstaller
*   如何调试 PyInstaller 错误
*   PyInstaller 做不到的

PyInstaller 让你能够创建一个文件夹或可执行文件，用户无需额外安装就可以立即运行。为了充分理解 PyInstaller 的强大功能，回顾一下 PyInstaller 帮助您避免的一些发行版问题是很有用的。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 分配问题

建立一个 Python 项目可能会令人沮丧，尤其是对于非开发人员来说。通常，设置从打开终端开始，这对于一大群潜在用户来说是不可能的。这个障碍甚至在安装指南深入研究虚拟环境、Python 版本和无数潜在依赖项的复杂细节之前就阻止了用户。

考虑一下在为 Python 开发设置一台新机器时通常会经历什么。大概是这样的:

*   下载并安装特定版本的 Python
*   设置 pip
*   设置虚拟环境
*   获取代码的副本
*   安装依赖项

停下来想一想，如果你不是开发人员，更不用说 Python 开发人员，上述步骤是否有意义。大概不会。

如果您的用户足够幸运地到达安装的依赖部分，这些问题就会爆发。随着 wheels 的流行，这在过去几年中已经变得更好了，但是一些依赖项仍然需要 C/C++甚至 FORTRAN 编译器！

如果你的目标是让尽可能多的用户使用一个应用程序，那么这个门槛太高了。正如 Raymond Hettinger 在他的[精彩演讲](https://pyvideo.org/speaker/raymond-hettinger.html)中经常说的，“一定有更好的方法。”

[*Remove ads*](/account/join/)

## PyInstaller

PyInstaller 通过找到你所有的依赖项并把它们捆绑在一起，从用户那里抽象出这些细节。你的用户甚至不会知道他们正在运行一个 [Python 项目](https://realpython.com/intermediate-python-project-ideas/)，因为 Python 解释器本身就捆绑在你的应用程序中。再见复杂的安装说明！

PyInstaller 通过[自省](https://en.wikipedia.org/wiki/Type_introspection)您的 Python 代码，检测您的依赖项，然后根据您的操作系统将它们打包成合适的格式，来完成这一惊人的壮举。

关于 PyInstaller 有很多有趣的细节，但是现在您将学习它如何工作以及如何使用它的基础知识。如果您想了解更多细节，您可以随时参考[优秀的 PyInstaller 文档](https://pyinstaller.readthedocs.io/en/stable/operating-mode.html#analysis-finding-the-files-your-program-needs)。

此外，PyInstaller 可以为 Windows、Linux 或 macOS 创建可执行文件。这意味着 Windows 用户将获得一个`.exe`，Linux 用户将获得一个常规的可执行文件，macOS 用户将获得一个`.app`包。对此有一些警告。更多信息参见[限制](#limitations)部分。

## 准备您的项目

PyInstaller 要求您的应用程序符合一些最小的结构，即您有一个 CLI 脚本来启动您的应用程序。通常，这意味着在你的 Python 包的之外创建一个小脚本*，它简单地导入你的包并运行`main()`。*

入口点脚本是一个 Python 脚本。从技术上讲，您可以在入口点脚本中做任何您想做的事情，但是您应该避免使用[显式相对导入](https://realpython.com/absolute-vs-relative-python-imports/#relative-imports)。如果您喜欢的话，您仍然可以在应用程序的其余部分使用相对导入。

注意:入口点是启动项目或应用程序的代码。

您可以在自己的项目中尝试一下，或者跟随 [Real Python feed reader 项目](https://github.com/realpython/reader)。关于[阅读器项目](https://github.com/realpython/reader)的更多详细信息，请查看关于[在 PyPI](https://realpython.com/pypi-publish-python-package/) 上发布包的教程。

构建该项目的可执行版本的第一步是添加入口点脚本。幸运的是，feed reader 项目结构良好，所以你只需要在包外面有一个简短的脚本*来运行它。例如，您可以使用以下代码在 reader 包旁边创建一个名为`cli.py`的文件:*

```py
from reader.__main__ import main

if __name__ == '__main__':
    main()
```

这个`cli.py`脚本调用`main()`来启动提要阅读器。

当您在自己的项目中工作时，创建这个入口点脚本很简单，因为您对代码很熟悉。然而，找到另一个人的代码的入口点并不容易。在这种情况下，你可以从查看第三方项目中的`setup.py`文件开始。

在项目的`setup.py`中查找对`entry_points`参数的引用。例如，这是读者项目的`setup.py`:

```py
setup(
    name="realpython-reader",
    version="1.0.0",
    description="Read the latest Real Python tutorials",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/realpython/reader",
    author="Real Python",
    author_email="info@realpython.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    packages=["reader"],
    include_package_data=True,
    install_requires=[
        "feedparser", "html2text", "importlib_resources", "typing"
    ],
    entry_points={"console_scripts": ["realpython=reader.__main__:main"]},
)
```

如您所见，入口点`cli.py`脚本调用了`entry_points`参数中提到的同一个函数。

更改之后，reader 项目目录应该如下所示，假设您将它签出到一个名为`reader`的文件夹中:

```py
reader/
|
├── reader/
|   ├── __init__.py
|   ├── __main__.py
|   ├── config.cfg
|   ├── feed.py
|   └── viewer.py
|
├── cli.py
├── LICENSE
├── MANIFEST.in
├── README.md
├── setup.py
└── tests
```

注意，阅读器代码本身没有变化，只是一个名为`cli.py`的新文件。这个入口点脚本通常是在 PyInstaller 中使用您的项目所必需的。

然而，您还需要注意在函数内部使用`__import__()`或导入。在 PyInstaller 术语中，这些被称为[隐藏导入](https://pyinstaller.readthedocs.io/en/stable/when-things-go-wrong.html?highlight=Hidden#listing-hidden-imports)。

如果在您的应用程序中更改导入非常困难，您可以手动指定隐藏的导入来强制 PyInstaller 包含这些依赖项。在本教程的后面，您将看到如何做到这一点。

一旦您可以在包的之外使用 Python 脚本*启动您的应用程序，您就可以尝试让 PyInstaller 创建一个可执行文件了。*

[*Remove ads*](/account/join/)

## 使用 PyInstaller

第一步是从 [PyPI](https://pypi.org) 安装 PyInstaller。你可以像使用其他 [Python 包](https://realpython.com/python-modules-packages)一样使用`pip`来完成这项工作:

```py
$ pip install pyinstaller
```

`pip`将安装 PyInstaller 的依赖项以及一个新命令:`pyinstaller`。PyInstaller 可以导入到 Python 代码中，并作为一个库使用，但您可能只会将它用作 CLI 工具。

如果你[创建你自己的钩子文件](https://pyinstaller.readthedocs.io/en/stable/hooks.html#)，你将使用库接口。

如果您只有纯 Python 依赖，您将增加 PyInstaller 默认创建可执行文件的可能性。然而，如果你对 C/C++扩展有更复杂的依赖，不要太紧张。

PyInstaller 支持许多流行的软件包，如 [NumPy](http://www.numpy.org) 、 [PyQt](https://pypi.org/project/PyQt5/) 和 [Matplotlib](https://matplotlib.org) ，而不需要您做任何额外的工作。通过参考 [PyInstaller 文档](https://github.com/pyinstaller/pyinstaller/wiki/Supported-Packages)，可以看到更多关于 PyInstaller 官方支持的包列表。

如果您的一些依赖项没有列在官方文档中，也不用担心。许多 Python 包工作良好。事实上，PyInstaller 非常受欢迎，许多项目都解释了如何使用 PyInstaller。

简而言之，你的项目打破常规的几率很高。

要尝试使用所有默认设置创建可执行文件，只需给 PyInstaller 主入口点脚本的名称。

首先，将`cd`放入带有您的入口点的文件夹中，并将其作为参数传递给安装 PyInstaller 时添加到您的 [`PATH`](https://realpython.com/add-python-to-path/) 中的`pyinstaller`命令。

例如，如果您正在跟踪 feed reader 项目，请在顶层`reader`目录中的`cd`之后键入以下内容:

```py
$ pyinstaller cli.py
```

如果您在构建可执行文件时看到大量输出，不要惊慌。默认情况下，PyInstaller 是冗长的，可以通过调试来提高冗长度，稍后您将看到这一点。

## 挖掘 PyInstaller 工件

PyInstaller 很复杂，会产生大量输出。所以，知道先关注什么很重要。也就是可以分发给用户的可执行文件和潜在的调试信息。默认情况下，`pyinstaller`命令会创建一些感兴趣的东西:

*   一个`*.spec`文件
*   一个`build/`文件夹
*   一个`dist/`文件夹

### 规格文件

默认情况下，规范文件将以您的 CLI 脚本命名。继续我们之前的例子，您会看到一个名为`cli.spec`的文件。在对`cli.py`文件运行 PyInstaller 之后，默认的 spec 文件看起来是这样的:

```py
# -*- mode: python -*-

block_cipher = None

a = Analysis(['cli.py'],
             pathex=['/Users/realpython/pyinstaller/reader'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='cli',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='cli')
```

该文件将由`pyinstaller`命令自动创建。您的版本将有不同的路径，但大多数应该是相同的。

不要担心，您不需要理解上面的代码就可以有效地使用 PyInstaller！

这个文件可以修改，并在以后创建可执行文件时重用。通过向`pyinstaller`命令提供这个 spec 文件而不是入口点脚本，可以使未来的构建更快一些。

PyInstaller 规范文件有一些特定的用例。然而，对于简单的项目，您不需要担心这些细节，除非您想要大量定制您的项目是如何构建的。

[*Remove ads*](/account/join/)

### 构建文件夹

文件夹`build/`是 PyInstaller 放置大部分元数据和内部簿记的地方，用于构建您的可执行文件。默认内容如下所示:

```py
build/
|
└── cli/
    ├── Analysis-00.toc
    ├── base_library.zip
    ├── COLLECT-00.toc
    ├── EXE-00.toc
    ├── PKG-00.pkg
    ├── PKG-00.toc
    ├── PYZ-00.pyz
    ├── PYZ-00.toc
    ├── warn-cli.txt
    └── xref-cli.html
```

build 文件夹对于调试很有用，但是除非您遇到问题，否则这个文件夹很大程度上可以被忽略。在本教程的后面，您将了解更多关于调试的内容。

### Dist Folder

构建完成后，您将得到一个类似如下的`dist/`文件夹:

```py
dist/
|
└── cli/
    └── cli
```

`dist/`文件夹包含您想要发送给用户的最终工件。在`dist/`文件夹中，有一个以您的入口点命名的文件夹。所以在这个例子中，您将有一个`dist/cli`文件夹，其中包含我们应用程序的所有依赖项和可执行文件。运行的可执行文件是`dist/cli/cli`或`dist/cli/cli.exe`，如果你在 Windows 上的话。

根据您的操作系统，您还会发现许多扩展名为`.so`、`.pyd`和`.dll`的文件。这些共享库代表 PyInstaller 创建和收集的项目的依赖项。

**注意:**如果您使用`git`进行版本控制，您可以将`*.spec`、`build/`和`dist/`添加到您的`.gitignore`文件中，以保持`git status`的整洁。Python 项目的默认 GitHub gitignore 文件[已经为您完成了这项工作。](https://github.com/github/gitignore/blob/master/Python.gitignore)

您将希望分发整个`dist/cli`文件夹，但是您可以将`cli`重命名为任何适合您的名称。

此时，如果您遵循提要阅读器示例，您可以尝试运行`dist/cli/cli`可执行文件。

您会注意到，运行可执行文件会导致提及`version.txt`文件的错误。这是因为提要阅读器及其依赖项需要一些 PyInstaller 不知道的额外数据文件。要解决这个问题，您必须告诉 PyInstaller 需要`version.txt`，您将在[测试您的新可执行文件](#testing-your-new-executable)时了解到这一点。

## 定制您的构建

PyInstaller 附带了许多选项，可以作为规范文件或普通 CLI 选项提供。下面，你会发现一些最常见和最有用的选项。

`--name`

> 更改可执行文件的名称。

这是一种避免您的可执行文件、规范文件和构建工件文件夹以您的入口点脚本命名的方法。如果您像我一样习惯于将您的入口点脚本命名为类似于`cli.py`的名称，那么`--name`是非常有用的。

您可以使用如下命令从`cli.py`脚本构建一个名为`realpython`的可执行文件:

```py
$ pyinstaller cli.py --name realpython
```

`--onefile`

> 将整个应用程序打包成一个可执行文件。

默认选项创建一个依赖项*和*以及可执行文件的文件夹，而`--onefile`通过只创建*和*一个可执行文件来简化发布。

该选项没有参数。要将您的项目捆绑到一个文件中，您可以使用如下命令进行构建:

```py
$ pyinstaller cli.py --onefile
```

使用上面的命令，您的`dist/`文件夹将只包含一个可执行文件，而不是一个包含所有独立文件的文件夹。

`--hidden-import`

> 列出 PyInstaller 无法自动检测的多个顶级导入。

这是使用`import`内部函数和`__import__()`解决代码问题的一种方法。你也可以在同一个命令中多次使用`--hidden-import`。

此选项需要您要包含在可执行文件中的包的名称。例如，如果您的项目在一个函数中导入了[请求](https://realpython.com/python-requests/)库，那么 PyInstaller 不会自动将`requests`包含在您的可执行文件中。您可以使用以下命令强制包含`requests`:

```py
$ pyinstaller cli.py --hiddenimport=requests
```

您可以在构建命令中多次指定该选项，每个隐藏导入指定一次。

`--add-data`和`--add-binary`

> 指示 PyInstaller 将附加数据或二进制文件插入到您的构建中。

当您想要捆绑配置文件、示例或其他非代码数据时，这很有用。如果您一直关注 feed reader 项目，稍后将会看到一个这样的例子。

`--exclude-module`

> 从可执行文件中排除一些模块

这有助于排除开发人员专用的需求，如测试框架。这是一个让你给用户的工件尽可能小的好方法。例如，如果您使用 [pytest](https://realpython.com/pytest-python-testing/) ，您可能希望将其从您的可执行文件中排除:

```py
$ pyinstaller cli.py --exclude-module=pytest
```

`-w`

> 避免自动打开控制台窗口进行`stdout`记录。

这只有在构建支持 GUI 的应用程序时才有用。通过允许用户永远看不到终端，这有助于隐藏实现的细节。

类似于`--onefile`选项，`-w`没有参数:

```py
$ pyinstaller cli.py -w
```

`.spec file`

如前所述，您可以重用自动生成的`.spec`文件来进一步定制您的可执行文件。`.spec`文件是一个普通的 Python 脚本，它隐式地使用 PyInstaller 库 API。

因为它是一个普通的 Python 脚本，所以你可以在里面做任何事情。你可以参考官方的 PyInstaller 规范文档来获得更多关于这个 API 的信息。

[*Remove ads*](/account/join/)

## 测试新的可执行文件

测试新的可执行文件的最好方法是在一台新机器上。新机器应该与您的构建机器具有相同的操作系统。理想情况下，这台机器应该尽可能与你的用户使用的相似。这并不总是可能的，所以下一个最好的方法是在您自己的机器上进行测试。

关键是在没有激活开发环境的情况下运行生成的可执行文件*。这意味着在没有`virtualenv`、`conda`或任何其他能够访问您的 Python 安装的**环境**的情况下运行。记住，PyInstaller 创建的可执行文件的主要目标之一是让用户不需要在他们的机器上安装任何东西。*

以提要阅读器为例，您会注意到在`dist/cli`文件夹中运行默认的`cli`可执行文件会失败。幸运的是，这个错误指出了问题所在:

```py
FileNotFoundError: 'version.txt' resource not found in 'importlib_resources'
[15110] Failed to execute script cli
```

`importlib_resources`包需要一个`version.txt`文件。您可以使用`--add-data`选项将这个文件添加到构建中。下面是如何包含所需的`version.txt`文件的示例:

```py
$ pyinstaller cli.py \
    --add-data venv/reader/lib/python3.6/site-packages/importlib_resources/version.txt:importlib_resources
```

这个命令告诉 PyInstaller 将`importlib_resources`文件夹中的`version.txt`文件包含在您的构建中一个名为`importlib_resources`的新文件夹中。

**注意:**`pyinstaller`命令使用`\`字符使命令更容易阅读。如果您使用相同的路径，您可以在自己运行命令时省略`\`,或者复制并粘贴下面的命令。

您需要调整上述命令中的路径，以匹配安装提要阅读器依赖项的位置。

现在运行新的可执行文件将导致一个关于`config.cfg`文件的新错误。

这个文件是 feed reader 项目所必需的，因此您需要确保将它包含在您的构建中:

```py
$ pyinstaller cli.py \
    --add-data venv/reader/lib/python3.6/site-packages/importlib_resources/version.txt:importlib_resources \
    --add-data reader/config.cfg:reader
```

同样，您需要根据 feed reader 项目所在的位置来调整文件的路径。

此时，您应该有一个可以直接提供给用户的工作可执行文件了！

## 调试 PyInstaller 可执行文件

正如您在上面看到的，您可能会在运行可执行文件时遇到问题。根据项目的复杂程度，修复可以简单到包括像提要阅读器示例这样的数据文件。然而，有时你需要更多的调试技巧。

以下是一些常见的策略，排名不分先后。通常情况下，这些策略中的一个或一个组合会在艰难的调试过程中带来突破。

### 使用终端

首先，尝试从终端运行可执行文件，这样就可以看到所有的输出。

记得移除`-w`构建标志，以便在控制台窗口中查看所有的`stdout`。通常，如果缺少依赖项，您会看到`ImportError`异常。

[*Remove ads*](/account/join/)

### 调试文件

检查`build/cli/warn-cli.txt`文件是否有问题。PyInstaller 创建了*大量的*输出来帮助你理解它到底在创建什么。在`build/`文件夹中挖掘是一个很好的起点。

### 单一目录构建

使用`--onedir`分发模式创建分发文件夹，而不是单个可执行文件。同样，这是默认模式。用`--onedir`构建让你有机会检查所有包含的依赖项，而不是所有东西都隐藏在一个可执行文件中。

`--onedir`对于调试很有用，但是`--onefile`通常更容易让用户理解。调试之后，你可能想切换到`--onefile`模式来简化发布。

### 其他 CLI 选项

PyInstaller 还可以控制构建过程中打印的信息量。使用 PyInstaller 的`--log-level=DEBUG`选项重新构建可执行文件，并查看输出。

当用`--log-level=DEBUG`增加详细程度时，PyInstaller 将创建*大量的*输出。将此输出保存到一个文件中很有用，您可以稍后参考，而不是在您的终端中滚动。为此，您可以使用您的 shell 的[重定向功能](https://en.wikipedia.org/wiki/Redirection_(computing))。这里有一个例子:

```py
$ pyinstaller --log-level=DEBUG cli.py 2> build.txt
```

通过使用上面的命令，您将拥有一个名为`build.txt`的文件，其中包含许多额外的`DEBUG`消息。

**注意:**带`>`的标准重定向是不够的。PyInstaller 打印到`stderr`流，*而不是* `stdout`。这意味着您需要将`stderr`流重定向到一个文件，这可以使用前面命令中的`2`来完成。

下面是您的`build.txt`文件的一个示例:

```py
67 INFO: PyInstaller: 3.4
67 INFO: Python: 3.6.6
73 INFO: Platform: Darwin-18.2.0-x86_64-i386-64bit
74 INFO: wrote /Users/realpython/pyinstaller/reader/cli.spec
74 DEBUG: Testing for UPX ...
77 INFO: UPX is not available.
78 DEBUG: script: /Users/realptyhon/pyinstaller/reader/cli.py
78 INFO: Extending PYTHONPATH with paths
['/Users/realpython/pyinstaller/reader',
 '/Users/realpython/pyinstaller/reader']
```

这个文件包含了很多详细的信息，比如你的构建中包含了什么，为什么没有包含什么，以及可执行文件是如何打包的。

除了使用`--log-level`选项获取更多信息之外，您还可以使用`--debug`选项重新构建您的可执行文件。

**注意:**`-y`和`--clean`选项在重建时很有用，尤其是在最初配置您的构建或者用[持续集成](https://realpython.com/python-continuous-integration/)构建时。这些选项删除了旧的构建，并且在构建过程中不需要用户输入。

### 附加 PyInstaller 文档

PyInstaller GitHub Wiki 有很多有用的链接和调试技巧。最值得注意的是关于[确保所有东西都正确包装](https://github.com/pyinstaller/pyinstaller/wiki/How-to-Report-Bugs#make-sure-everything-is-packaged-correctly)和如果事情出错该怎么做[的部分](https://github.com/pyinstaller/pyinstaller/wiki/If-Things-Go-Wrong)。

### 协助依赖性检测

如果 PyInstaller 不能正确地检测到所有的依赖项，您将看到的最常见的问题是`ImportError`异常。如前所述，如果你正在使用`__import__()`，函数内部的导入，或者其他类型的[隐藏导入](https://pyinstaller.readthedocs.io/en/stable/when-things-go-wrong.html?highlight=Hidden#listing-hidden-imports)，就会发生这种情况。

许多这类问题可以通过使用`--hidden-import` PyInstaller CLI 选项来解决。这告诉 PyInstaller 包含一个模块或包，即使它没有自动检测到它。这是在你的应用程序中解决大量动态导入魔法的最简单的方法。

另一种解决问题的方法是[钩子文件](https://pyinstaller.readthedocs.io/en/stable/hooks.html)。这些文件包含帮助 PyInstaller 打包依赖项的附加信息。您可以编写自己的钩子，并告诉 PyInstaller 通过`--additional-hooks-dir` CLI 选项来使用它们。

钩子文件是 PyInstaller 内部工作的方式，所以你可以在 PyInstaller 源代码中找到很多钩子文件的例子。

[*Remove ads*](/account/join/)

## 局限性

PyInstaller 非常强大，但是它也有一些限制。前面已经讨论过一些限制:入口点脚本中的隐藏导入和相对导入。

PyInstaller 支持为 Windows、Linux 和 macOS 制作可执行文件，但是它不能[交叉编译](https://en.wikipedia.org/wiki/Cross_compiler)。因此，您不能从一个操作系统创建针对另一个操作系统的可执行文件。因此，要为多种类型的操作系统分发可执行文件，您需要为每种支持的操作系统创建一台构建机器。

与交叉编译限制相关，知道 PyInstaller 在技术上不完全捆绑您的应用程序运行所需的一切是有用的。您的可执行文件仍然依赖于用户的 [`glibc`](https://en.wikipedia.org/wiki/GNU_C_Library) 。通常情况下，您可以通过构建每一个目标操作系统的最老版本来绕过`glibc`限制。

例如，如果您想要针对各种各样的 Linux 机器，那么您可以构建一个旧版本的 [CentOS](https://www.centos.org) 。这将使您能够兼容比您构建的版本更新的大多数版本。这与 [PEP 0513](https://www.python.org/dev/peps/pep-0513/) 中描述的策略相同，也是 [PyPA](https://www.pypa.io/en/latest/) 对制造兼容车轮的建议。

事实上，您可能希望使用 [PyPA 的 manylinux docker 映像](https://github.com/pypa/manylinux)来研究您的 linux 构建环境。您可以从基本映像开始，然后安装 PyInstaller 以及所有的依赖项，并拥有一个支持大多数 Linux 变体的构建映像。

## 结论

PyInstaller 可以使复杂的安装文档变得不必要。相反，您的用户可以简单地运行您的可执行文件来尽快开始。PyInstaller 工作流可以总结为执行以下操作:

1.  创建一个调用主函数的入口点脚本。
2.  安装 PyInstaller。
3.  在您的入口点上运行 PyInstaller。
4.  测试新的可执行文件。
5.  将生成的`dist/`文件夹发送给用户。

您的用户根本不需要知道您使用的是哪个版本的 Python，或者您的应用程序使用的是 Python！******