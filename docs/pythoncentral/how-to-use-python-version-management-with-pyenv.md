# 如何通过 Pyenv 使用 Python 版本管理

> 原文：<https://www.pythoncentral.io/how-to-use-python-version-management-with-pyenv/>

有没有想过开发团队如何为支持多个 Python 版本的项目做出贡献？仔细想想，跨多个 Python 版本测试一个项目似乎是一项繁重的工作。

但是谢天谢地，在现实中，这很容易做到。使用 pyenv 可以相对容易地管理多个 Python 版本。

在本指南中，我们将带您了解什么是 pnenv，它是如何工作的，以及如何使用它来管理 Python 版本。

## **pyenv 是什么？它是如何工作的？**

pyenv 最初以“Pythonbrew”的名字发布，它是一个版本管理工具，使 Python 用户能够管理项目中的 Python 版本。如果您的机器安装了多个 Python 版本，那么您可能已经在机器上安装了 pyenv。

该工具使您能够利用新的 Python 特性，并为在各种 Python 版本上工作的项目做出贡献。需要注意的是，在 Python 版本之间切换时，可能会出现几个问题。

例如，如果您从 Python 3.7 切换到 3.9，特性会有很大的不同，3.7 的特性集要小得多。pyenv 的好处在于，它有助于弥合这些特性之间的差距，并克服因缺乏特性而产生的问题。

pyenv 工作背后的基本理念是直接使用垫片。

Shims 是轻量级的可执行文件，pyenv 使用它们将您的命令从您的机器上安装的所有版本传递到您的项目设计运行的正确 Python 版本。

当您安装 pyenv 时，垫片被引入到环境路径所在的目录中。这样，当 Python 命令运行时，pyenv 会拦截它并将其定向到适当的填充程序。

在这个过程的这一点上，它识别你的项目需要的 Python 版本，然后将命令传递给正确的 Python 版本。

## pyenv 的安装要求

像任何其他应用程序一样，编程语言的解释器也接收更新。更新可能会改进一系列功能，涉及补丁，引入错误修复，或添加新功能。

要在机器上使用 pyenv 进行 Python 版本管理，机器需要安装 Python。此外，pyenv 需要 shell 路径才能正常工作。

让我们回顾一下激活 pyenv 以便在您的机器上安装的步骤:

*   访问 [官方 Python 网站](https://www.python.org/) 并安装在你的机器上。
*   根据需要设置 shell——要使 pyenv 能够导航 Python 版本并选择合适的版本，它必须使用 shell 的 PATH 变量。PATH 的作用是根据命令确定 shell 需要在哪里搜索文件。因此，您必须确保 shell 找到 pyenv 正在运行的 Python 版本，而不是它默认检测到的版本(通常是系统版本)。
*   当 shell 准备好，并且路径设置正确时，您必须激活环境。我们将在接下来的两节中探讨如何做到这一点。

### **pyenv global**

pyenv global 的主要功能是确保 pyenv 可以在使用不同操作系统的机器上使用所有 Python 版本，开发团队使用这些操作系统来创建项目。

你可以为你正在处理的任何项目设置一个目录，根据它正在处理的 Python 版本来命名。pyenv 还为开发人员提供了创建和管理虚拟环境的灵活性。

这些特性适用于 Linux 发行版和 OS X，不依赖于 Python。因此，采取全球方法在用户一级有效。换句话说，不需要使用任何 sudo 命令来使用它。

需要注意的重要一点是，可以使用其他命令覆盖 global。但是，您可以使用一个命令来确保默认情况下使用特定的 Python 版本。例如，如果开发人员想默认使用 3.9，他们可以运行下面的命令:

```py
$ pyenv global 3.9
```

上面一行会改变~/中的版本。pyenv/版本到 3.9。正如您可能已经猜到的那样，pyenv 全局并不是专门为某些依赖项和应用程序而设置的，而是为整个项目而设置的。

您也可以运行上面的命令来查看项目是否在 Python 版本上运行。

### **pyenv 本地**

该命令有助于检查哪个 Python 版本适用于特定的应用程序。因此，如果您运行以下命令:

```py
$ pyenv local 2.7.
```

该命令将在当前目录下创建一个. python 版本的文件。

如果在机器和环境上安装并激活了 pyenv，运行该命令将为您创建 2.7 版本。

## **使用 pyenv**

参与 Python 项目的开发人员和测试人员都需要利用几个 Python 版本来发布一个项目。一遍又一遍地改变 Python 版本是耗时的、具有挑战性的，而且是彻头彻尾的烦人，会减慢进度。

由于这个原因，pyenv 被开发人员和测试人员视为必不可少的工具。它自动在 Python 版本之间切换，使得构建和测试 Python 项目更加方便。

如前所述，该工具使管理 Python 版本变得容易。以下是使用 pyenv 的关键步骤:

*   安装正在为其构建项目的所有 Python 版本。
*   在计算机内设置机器的默认 Python 版本。
*   针对项目设置本地 Python 版本。
*   创建一个虚拟环境，以便正确使用 pyenv。

### **常见错误:“pyenv 不更新 Python 版本”**

当您设置 pyenv 环境时，您可能会遇到一个错误，显示一条类似“pyenv global x.x.x 没有更新 Python 版本”的消息

不要担心，因为当有人设置 pyenv 时，这种错误在 macOS 和 Linux 机器上出现是很常见的。

通过在~/中添加下面一行代码，很容易解决这个错误。zshrc 文件:

```py
eval "$(pyenv init -)"eval "$(pyenv init --path)"
```

### **使用 pyenv 迁移包**

迁移包包括在 Python 版本之间移动 Python 库或包。

该过程包括根据选择标准转移所有相关的设置、依赖关系和程序。有了 pyenv，就没有什么可担心的了，因为它允许您通过内置的二进制包来完成它。

如果你在 Windows 或 Linux 机器上，你所要做的就是在你的终端上输入以下命令:

```py
git clone https://github.com/pyenv/pyenv-pip-migrate.git $(pyenv root)/plugins/pyenv-pip-migrate
```

该命令会将最新的 pyenv 版本安装到上述目录中。您可以在 macOS 机器上通过启动 Homebrew 并运行命令来做同样的事情:

```py
$ brew install pyenv-pip-migrate
```

### **用 pyenv 列出并安装所有 Python 版本**

我们已经在这篇文章中简要介绍了如何安装 pyenv，所以让我们来看看如何用这个工具安装 Python。

请记住，如果您按如下所述安装 Python，您将看到一个 Python 版本列表，您可以在项目中进行切换。

pyenv 提供了一些命令，向您展示您可以安装的 Python 版本。要获得在您的机器上使用 pyenv 的 Python 版本列表，您可以运行命令:

```py
$ pyenv install --list | grep " 3\.[678]"
```

您可以期待类似这样的输出:

| 3.6.03.6-开发3.6.13.6.23.6.33.6.43.6.53.6.63.6.73.6.83.7.03.7-开发3.7.13.7.23.8-开发 |

开发人员可能会意识到上面的 Python 版本列表包含了“Cython”，这是 Python 的一个超集，旨在执行类似 C 语言的功能。

超集从 C 语言中获取语法，可以像 C 语言一样快速运行。默认情况下，当您安装 Python 时，您正在安装它的 Cython 版本。

在命令中," 3\ "part 定义了您想要显示的 pyenv 的子版本。

假设您想要安装 Python 解释器的 Jython 版本。您可以稍微修改一下我们之前使用的命令并运行它。应该是这样的:

```py
$ pyenv install --list | grep "jython"
```

运行该命令的结果如下:

| jython-devjython-2.5.0jython-2.5-devjython-2.5.1jython-2.5.2jython-2.5.3jython-2.5.4-rcljython-2.7.0jython-2.7.1 |

Jython 是 Python 的一个实现，设计用于在 Java 上运行。它以前被称为 JPython，提供了在 Python 上运行基于类的程序的好处，这些程序本来是要在 JVM 上运行的。

您也可以不使用附加参数而运行 pyenv - list。这将允许您找到 Python 的实现，并查看 pyenv 工具能够获取的所有版本。

### **用 pyenv 切换 Python 版本**

当您准备更改 Python 版本或安装一个不在您机器上的新版本时，您可以运行命令:

```py
$ pyenv install -v 3.9.3
```

该命令将下载您指定的 Python 版本，将它与收集到的包一起安装，并让您知道它是否安装成功以及安装在哪里。

### **你应该知道的 pyenv 命令**

pyenv 可以使用许多命令，这些命令允许您对许多 Python 版本执行一系列检查。当测试人员和开发人员团队需要在 Python 版本之间来回切换时，pyenv 提供的命令变得非常有价值。

这里快速浏览了许多可用的命令:

*   **pyenv 命令:** 该命令显示 pyenv 可以使用的所有命令和子命令的列表。
*   **pyenv global:** 您可以使用这个命令来设置将在所有 shells 中使用的全局 Python 版本。使用命令会将版本名称组合到~/中。pyenv/版本文件。可以用特定于应用程序的覆盖全局 Python 版本。python 版本文件。或者，您也可以指定环境变量 PYENV_VERSION 来完成此操作。
*   pyenv help: 它输出一个你可以用 pyenv 使用的所有命令的列表，以及这些命令要完成的任务的简短解释。如果您需要某个命令的详细信息，您应该运行这个命令。
*   **pyenv install:** 在这篇文章的前面，我们已经介绍了在用 pyenv 列出 Python 版本时如何使用 install 命令。该命令允许您安装一个特定的 Python 版本，您可以使用以下标志属性:
    *   它显示了所有可安装 Python 版本的列表。
    *   -g:你可以用这个标志构建一个调试 Python 版本。
    *   -v:打开详细模式，允许您将编译状态打印到 stdout。
*   **pyenv local:**local 命令允许你设置一个特定于应用的本地 Python 版本。它通过向。python 版本文件。该命令允许您通过设置 PYENV_VERSION 或使用 pyenv shell 命令来覆盖全局 Python 版本。
*   **pyenv shell:** 使用 shell 中的 PYENV_VERSION 环境变量，您可以仅为 shell 设置 Python 版本。该命令不仅会覆盖全局版本，还会覆盖特定于应用程序的 Python 版本。
*   **pyenv 版本:** 你可以使用该命令显示当前安装在机器上的所有 Python 版本。
*   **pyenv which:** 这个命令允许你找到你的机器的可执行文件的完整路径。如前所述，pyenv 利用了垫片，该命令允许您查看可执行 pyenv 运行的路径。

## **结论**

有了这个指南，你应该能够通过最大限度地使用 pyenv 工具，毫不费力地为 Python 项目做出贡献。它允许您在几个 Python 版本之间移动，并在最新和最老的 Python 版本上测试您的项目——所有这些都不会干扰开发系统。

您可以在 Linux、Windows 和 macOS 上使用 pyenv，并检查您机器的当前 Python 版本。还可以用 pyenv 安装新版本的 Python。我们已经介绍了如何在 Python 版本之间切换并利用它们不同的特性。