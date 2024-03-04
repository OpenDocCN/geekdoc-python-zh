# 如何运行您的 Python 脚本

> 原文：<https://www.pythonforbeginners.com/development/how-run-your-python-scripts>

您的 Python 代码可以位于代码编辑器、IDE 或文件中。而且，除非你知道如何执行你的 Python 脚本，否则它不会起作用。

在这篇博文中，我们将看看执行 Python 代码和脚本的 7 种方式。无论您的操作系统是什么，您的 Python 环境是什么，或者您的代码在什么位置，我们都将向您展示如何执行这段代码！

## 目录

1.  交互式运行 Python 代码
2.  Python 脚本是如何执行的
3.  如何运行 Python 脚本
4.  如何使用命令行运行 Python 脚本
5.  如何交互式运行 Python 代码
6.  从文本编辑器运行 Python 代码
7.  从 IDE 中运行 Python 代码
8.  如何从文件管理器运行 Python 脚本
9.  如何从另一个 Python 脚本运行 Python 脚本

## 在哪里运行 Python 脚本以及如何运行？

您可以从以下位置运行 Python 脚本:

1.  操作系统命令行(也称为外壳或终端)
2.  在 Anaconda 上运行特定 Python 版本的 Python 脚本
3.  使用 Crontab
4.  使用另一个 Python 脚本运行一个 Python 脚本
5.  使用文件管理器
6.  使用 Python 交互模式
7.  使用 IDE 或代码编辑器

## 交互式运行 Python 代码

要启动 Python 代码的交互式会话，只需打开终端或命令行并输入 Python(或 Python 3，取决于您的 Python 版本)。一旦你按下回车键，你将进入交互模式。

以下是在 Windows、Linux 和 MacOS 中进入交互模式的方法。

### Linux 上的交互式 Python 脚本模式

打开你的终端。

它应该看起来像这样

```py
$ python
   Python 3.7.3 (default, Mar 27 2019, 22:11:17)
   [GCC 7.3.0] :: Anaconda, Inc. on linux
   Type "help", "copyright", "credits" or "license" for more information.
```

按“回车”后进入 Python 脚本交互模式。

### Mac OSX 上的交互式 Python 脚本模式

在 Mac OS 上启动交互式 Python 脚本模式与 Linux 非常相似。下图显示了 Mac OS 上的交互模式。

![Python on MacOSx](img/e8a57f4972516a1d3218894eabc89576.png)



### Windows 上的交互式 Python 脚本模式

在 Windows 上，进入你的命令提示符，写“python”。一旦您按下 enter 键，您应该会看到类似这样的内容:

![](img/e281f4ce9d91221c0fea7e2c2cd3713f.png)



### 交互式运行 Python 脚本

使用交互式 Python 脚本模式，您可以编写代码片段并执行它们，以查看它们是否给出所需的输出或是否失败。

以下面的 for 循环为例。

![Python Loop example](img/e8a57f4972516a1d3218894eabc89576.png)



我们的代码片段被编写为打印所有内容，包括 0 到 5。所以，你在 print(i)之后看到的是这里的输出。

要退出交互式 Python 脚本模式，请编写以下内容:

```py
>>>exit() 
```

然后，按回车键。您应该会回到最初启动的命令行屏幕。

还有其他方法可以退出交互式 Python 脚本模式。在 Linux 上，你可以简单地按 Ctrl + D，而在 Windows 上，你需要按 Ctrl + Z + Enter 来退出。

请注意，当您退出交互模式时，您的 Python 脚本不会保存到本地文件。

## Python 脚本是如何执行的？

一种直观显示执行 Python 脚本时发生的情况的好方法是使用下图。该块代表我们编写的 Python 脚本(或函数)，其中的每个块代表一行代码。

![Python function block](img/e8a57f4972516a1d3218894eabc89576.png)



当您运行这个 Python 脚本时，Python 解释器从上到下执行每一行。

这就是 Python 解释器执行 Python 脚本的方式。

但不是这样的！还有很多事情发生。

### Python 解释器如何运行代码的流程图

第一步:你的脚本或。py 文件编译并生成二进制格式。这种新的形式要么是。pyc 或. pyo。

![python compiler](img/e8a57f4972516a1d3218894eabc89576.png)



步骤 2:生成的二进制文件，现在由解释器读取以执行指令。

把它们想象成一堆导致最终结果的指令。

检查字节码有一些好处。而且，如果你的目标是把自己变成一个专业的 Python 高手，你可能想要学习和理解字节码来编写高度优化的 Python 脚本。

您还可以使用它来理解和指导您的 Python 脚本的设计决策。你可以看看某些因素，理解为什么有些函数/数据结构比其他的快。

## 如何运行 Python 脚本？

要使用命令行运行 Python 脚本，首先需要将代码保存为本地文件。

让我们再次以本地 Python 文件为例。如果你把它保存到本地。名为 python_script.py 的 py 文件。

有许多方法可以做到这一点:

1.  从命令行创建一个 Python 脚本并保存它
2.  使用文本编辑器或 IDE 创建 Python 脚本并保存它

从代码编辑器中保存 Python 脚本非常容易。基本上就像保存一个文本文件一样简单。

但是，要通过命令行来完成，需要几个步骤。

首先，转到命令行，将工作目录更改为您希望保存 Python 脚本的位置。

进入正确的目录后，在终端中执行以下命令:

```py
$ sudo nano python_script.py 
```

一旦您按下 enter，您将进入一个命令行界面，如下所示:

![python script example](img/e8a57f4972516a1d3218894eabc89576.png)



现在，您可以在这里编写 Python 代码，并使用命令行轻松运行它。

## 如何使用命令行运行 Python 脚本？

Python 脚本可以通过命令行界面使用 Python 命令运行。确保您指定了脚本的路径或具有相同的工作目录。要执行您的 Python 脚本(python_script.py ),请打开命令行并编写 python3 python_script.py

如果您的 python 版本是 python2.x，请用 Python 替换 python3。

以下是我们保存在 python_script.py 中的内容

```py
for i in range(0,5):
               print(i) 
```

并且，命令行上的输出如下所示

![execute python script](img/e8a57f4972516a1d3218894eabc89576.png)



比方说，我们想要保存 Python 代码的输出，即 0，1，2，3，4——我们使用了一种叫做管道操作符的东西。

在我们的情况下，我们要做的就是:

```py
$python python_script.py > newfile.txt 
```

并且，会创建一个名为“newfile.txt”的文件，其中保存了我们的输出。

## 如何交互式运行 Python 代码

有 4 种以上的方法可以交互运行 Python 脚本。在接下来的几节中，我们将看到执行 Python 脚本的所有主要方式。

### 使用导入运行 Python 脚本

我们都非常频繁地使用导入模块来加载脚本和库。您可以编写自己的 Python 脚本(比如 code1.py)并将其导入到另一个代码中，而无需再次在新脚本中编写整个代码。

下面是如何在新的 Python 脚本中导入 code1.py。

```py
>>> import code1 
```

但是，这样做意味着将 code1.py 中的所有内容都导入到 Python 代码中。这不是问题，直到你开始在你的代码必须针对性能、可伸缩性和可维护性进行优化的情况下工作。

比方说，我们在 code1 中有一个小函数，它可以绘制一个漂亮的图表，例如 chart_code1()。而且，该函数是我们希望导入整个 code1.py 脚本的唯一原因。我们不必调用整个 Python 脚本，而是可以简单地调用函数。

这是你通常会怎么做

```py
>>> from code1 import chart_code1 
```

而且，您应该能够在新的 Python 脚本中使用 chart_code1，就像它出现在您当前的 Python 代码中一样。

接下来，让我们看看导入 Python 代码的其他方法。

### 使用和 importlib 运行 Python 代码

importlib 的 import_module()允许您导入和执行其他 Python 脚本。

它的工作方式非常简单。对于我们的 Python 脚本 code1.py，我们要做的就是:

```py
import importlib
   import.import_module(‘code1’) 
```

没必要补充。py in import_module()。

让我们看一个例子，我们有复杂的目录结构，我们希望使用 importlib。我们要运行的 Python 代码的目录结构如下:

级别 1

|

+–_ _ init _ _。巴拉圭

–二级

|

+–_ _ init _ _。巴拉圭

–level 3 . py

在这种情况下，如果你认为你可以做 import lib . import _ module(" level 3 ")，你会得到一个错误。这被称为相对导入，其方法是使用带有 anchor explicit 的相对名称。

因此，要运行 Python 脚本 level3.py，您可以

```py
importlib.import_module(“.level3”, “level1.level”) 
```

或者你可以

```py
importlib.import_module(“level1.level2.level3”). 
```

### 使用 runpy 运行 Python 代码

Runpy 模块定位并执行 Python 脚本，而不导入它。用法非常简单，因为您可以轻松地在 run_module()中调用模块名。

使用 runpy 执行我们的 code1.py 模块。这是我们要做的。

```py
>>> import runpy
   >>> runpy.run_module(mod_name=”code1”) 
```

### 动态运行 Python 代码

我们将看看 exec()函数来动态执行 Python 脚本。在 Python 2 中，exec 函数实际上是一个语句。

下面是它如何帮助你在字符串的情况下动态执行 Python 代码。

```py
>>> print_the_string  = ‘print(“Dynamic Code Was Executed”)’
   >>>  exec(print_the_string) 
```

动态代码已执行

但是，使用 exec()应该是最后的手段。由于速度慢且不可预测，请尝试看看是否有其他更好的替代方案。

## 从文本编辑器运行 Python 脚本

要使用 Python 文本编辑器运行 Python 脚本，您可以使用默认的“运行”命令或使用热键，如 Function + F5 或简单的 F5(取决于您的操作系统)。

下面是一个在空闲状态下执行 Python 脚本的例子。

![execute python from text editor](img/e8a57f4972516a1d3218894eabc89576.png)



资料来源:pitt.edu

但是，请注意，您不能像通常从命令行界面执行那样控制虚拟环境。

这就是 ide 和高级文本编辑器远胜于代码编辑器的地方。

## 从 IDE 中运行 Python 脚本

当从 IDE 执行脚本时，您不仅可以运行您的 Python 代码，还可以调试它并选择您想要运行它的 Python 环境。

虽然 IDE 的 UI 界面可能会有所不同，但是保存、运行和编辑代码的过程非常相似。

## 如何从文件管理器运行 Python 脚本

如果有一种只需双击就能运行 Python 脚本的方法呢？您实际上可以通过创建代码的可执行文件来做到这一点。例如，在 Windows 操作系统的情况下，您可以简单地创建 Python 脚本的. exe 扩展名，并通过双击它来运行它。

## 如何从另一个 Python 脚本运行 Python 脚本

虽然我们还没有提到这一点，但是，如果你回去阅读，你会发现你可以:

1.  通过调用另一个 Python 脚本的命令行运行 Python 脚本
2.  使用类似 import 的模块加载 Python 脚本

就是这样！

## 关键外卖

1.  您可以在交互和非交互模式下编写 Python 代码。一旦退出交互模式，数据就会丢失。所以，sudo nano your _ python _ filename . py 吧！
2.  您还可以通过 IDE、代码编辑器或命令行运行 Python 代码
3.  有不同的方法可以导入 Python 代码并将其用于另一个脚本。明智地挑选，看看优缺点。
4.  Python 读取您编写的代码，将其翻译成字节码，然后用作指令——所有这些都发生在您运行 Python 脚本时。因此，学习如何使用字节码来优化您的 Python 代码。