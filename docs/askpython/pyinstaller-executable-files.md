# py installer–创建可执行 Python 文件

> 原文：<https://www.askpython.com/python/pyinstaller-executable-files>

嘿伙计们！在本教程中，我们将学习 Python 的 ***PyInstaller*** 的用途和基础知识。所以，让我们开始吧！

## PyInstaller 是什么？

PyInstaller 是 Python 中的一个包，它将 Python 应用程序的所有依赖项捆绑在一个包中。

我们不需要为不同的应用程序安装不同的包或模块。

PyInstaller 读取并分析我们的代码，然后发现程序执行所需的模块。然后将它们打包到一个文件夹或一个可执行文件中。

它用于创建；用于 windows 的 exe 文件。适用于 Mac 的应用程序文件和适用于 Linux 的可分发包。

## 如何安装 PyInstaller？

我们可以从 PyPI 下载 PyInstaller。我们可以使用 [pip 包管理器](https://www.askpython.com/python-modules/python-pip)来安装它。

建议[创建一个虚拟环境](https://www.askpython.com/python/examples/virtual-environments-in-python)并在那里安装 PyInstaller。

在 windows 命令提示符下输入以下命令-

```py
pip install pyinstaller

```

现在，将当前目录设置为您的程序 *program.py* 的位置。

```py
cd CurrentDirectory

```

运行下面给出的代码。这里， *program.py* 是我们给定的 python 脚本的名称。

```py
pyinstaller program.py

```

PyInstaller 分析我们的代码并执行以下操作-

1.  创建一个 *program.spec* 文件，其中包含关于应该打包的文件的信息。
2.  创建一个包含一些日志文件和工作文件的构建文件夹。
3.  还将创建一个名为 *dist* 的文件夹，其中包含一个*。与给定的 python 脚本名称同名的 exe* 文件。

现在，使用 PyInstaller 有 3 个重要元素:

*   的。规格文件
*   构建文件夹
*   dist 文件夹

## 什么是等级库文件？

规格文件，简称规格文件，是执行`pyinstaller program.py`后建立的第一个文件。它存储在`--specpath= directory`中。spec 文件是一个可执行的 python 代码，它告诉 PyIstaller 如何处理我们的 Python 脚本。

不需要修改规范文件，除非在极少数情况下，当我们希望:

1.  将我们的数据文件与应用程序捆绑在一起。
2.  包含 PyInstaller 未知的运行时库。
3.  向可执行文件添加 Python 运行时选项。
4.  用合并的公用模块创建多道程序包。

**要创建一个规范文件，运行以下命令-**

```py
pyi-makespec program.py

```

该命令创建 *program.py* 规格文件。为了构建应用程序，我们将规范文件传递给 pyinstaller 命令:

```py
pyinstaller program.spec

```

## 什么是构建文件夹？

构建文件夹存储元数据，对**调试**很有用。**

**build/*文件夹中的*文件 build/program/warn-program.txt 包含了很多输出，可以更好的理解事情。

要查看输出，使用`--log-level=DEBUG`选项重建可执行文件。建议保存此输出，以便以后参考。我们可以通过-

```py
pyinstaller --log-level=DEBUG program.py

```

这将在 build 文件夹中创建一个 build.txt 文件，其中将包含许多调试消息。

## 什么是 dist 文件夹？

dist 文件夹包含应用程序的需求和可执行文件。它包含一个与我们的脚本同名的. exe 文件。

要运行的可执行文件在 windows 上是 *dist/program/program.exe* 。

## 什么是*导入错误*？

如果 PyInstaller 无法正确检测所有的依赖项，就会出现一个 ***导入错误*** 错误。当我们使用`__import__( )` *时就会出现这种情况。*函数内部导入或隐藏导入。

为了解决这个问题，我们使用了`--hidden-import`。该命令自动包含软件包和模块。

另一种方法是使用包含附加信息的钩子文件来帮助 PyInstaller 打包一个依赖项。

```py
pyinstaller --additional-hooks-dir=. program.py

```

## 如何使用 PyInstaller 更改可执行文件的名称？

为了避免我们的规范、构建和可执行文件被命名为与我们的 python 脚本名称相同，我们可以使用*–name*命令。

```py
pyinstaller program.py --name pythonproject

```

## 如何创建单个可执行文件？

PyInstaller 还可以为我们的 Python 脚本创建一个单文件应用程序。它包含我们的 Python 程序所需的所有 [Python 模块](https://www.askpython.com/python-modules/python-modules)的档案。

要创建 python 脚本的一个可执行文件，运行下面给出的代码-

```py
pyinstaller --onefile --windowed program.py

```

## 测试我们的可执行文件

我们应该总是在一台没有任何开发环境的新机器上测试我们的可执行程序，比如 *virtualev* 、 *conda* 等等。因为 PyInstaller 可执行文件的主要目的是用户不需要在他们的系统上安装任何东西。

现在，在运行可执行文件之后，出现了以下错误。

```py
FileNotFoundError: 'version.txt' resource not found in 'importlib_resources'

```

这是因为 *'importlib_resources'* 需要 *version.txt* 文件。我们可以通过下面给出的命令添加这个文件。

```py
pyinstaller program.py \
    --add-data venv/reader/lib/python3.6/site-packages/importlib_resources/version.txt:importlib_resources

```

这将包括 *importlib_resources* 文件夹下新文件夹中的 *version.txt* 文件。

## 结论

在本文中，我们学习了如何安装 PyInstaller、运行、调试和测试我们的可执行文件。敬请期待！

## 参考

[PyInstaller 官方文档](https://pyinstaller.org/en/stable/)