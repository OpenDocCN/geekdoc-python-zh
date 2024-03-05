# 如何运行您的 Python 脚本

> 原文：<https://realpython.com/run-python-scripts/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**运行 Python 脚本**](/courses/running-python-scripts/)

作为 Python 开发人员，您需要掌握的最重要的技能之一是能够运行 Python 脚本和代码。这将是你知道你的代码是否如你计划的那样工作的唯一方法。这甚至是知道你的代码是否工作的唯一方法！

这个循序渐进的教程将根据您的环境、平台、需求和程序员的技能，引导您通过一系列方式来运行 Python 脚本。

**您将有机会通过使用**来学习如何运行 Python 脚本

*   操作系统命令行或终端
*   Python 交互模式
*   您最喜欢的 IDE 或文本编辑器
*   系统的文件管理器，双击脚本图标

这样，您将获得所需的知识和技能，使您的开发周期更加高效和灵活。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

***参加测验:****通过我们的交互式“如何运行您的 Python 脚本”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/run-python-scripts/)

## 脚本与模块

在计算技术中，脚本一词用来指包含命令逻辑序列的文件或批处理文件。这通常是一个简单的程序，存储在一个纯文本文件中。

脚本总是由某种解释器处理，解释器负责按顺序执行每个命令。

打算由用户直接执行的包含 Python 代码的纯文本文件通常被称为**脚本**，这是一个非正式术语，意思是**顶层程序文件**。

另一方面，包含 Python 代码的纯文本文件被称为**模块**，该代码被设计为从另一个 Python 文件导入并使用。

所以，[模块](https://realpython.com/python-modules-packages/)和脚本的主要区别在于，**模块是为了导入**，而**脚本是为了直接执行**。

无论是哪种情况，重要的是知道如何运行您编写到模块和脚本中的 Python 代码。

[*Remove ads*](/account/join/)

## Python 解释器是什么？

Python 是一种优秀的编程语言，允许你在[广泛的领域](https://realpython.com/what-can-i-do-with-python/)中高效工作。

Python 也是一个叫做**解释器**的软件。解释器是运行 Python 代码和脚本所需的程序。从技术上讲，解释器是一层软件，它在你的程序和你的计算机硬件之间工作，使你的代码运行。

根据您使用的 Python 实现，解释器可以是:

*   用 [C](https://realpython.com/c-for-python-programmers/) 写的程序，像 [CPython](https://www.python.org/about/) ，这是语言的核心实现
*   用 [Java](https://realpython.com/oop-in-python-vs-java/) 写的程序，像 [Jython](http://www.jython.org/index.html)
*   用 Python 自己写的程序，像 [PyPy](https://realpython.com/pypy-faster-python/)
*   在中实现的程序。NET，像 [IronPython](http://ironpython.net/)

无论解释器采用什么形式，你写的代码总是由这个程序运行。因此，能够运行 Python 脚本的首要条件是[在您的系统](https://realpython.com/installing-python/)上正确安装解释器。

解释器能够以两种不同的方式运行 Python 代码:

*   作为脚本或模块
*   作为在交互式会话中键入的一段代码

## 如何交互式运行 Python 代码

一种广泛使用的运行 Python 代码的方式是通过交互式会话。要启动 Python 交互式会话，只需打开命令行或终端，然后根据您的 Python 安装键入`python`或`python3`，然后点击 `Enter` 。

下面是一个在 Linux 上如何做到这一点的例子:

```py
$ python3
Python 3.6.7 (default, Oct 22 2018, 11:32:17)
[GCC 8.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

交互模式的标准提示是`>>>`，所以只要你看到这些字符，你就知道你已经进入了。

现在，您可以随心所欲地编写和运行 Python 代码，唯一的缺点是当您关闭会话时，您的代码将会消失。

当您以交互方式工作时，您键入的每个表达式和语句都会被立即计算和执行:

>>>

```py
>>> print('Hello World!')
Hello World!
>>> 2 + 5
7
>>> print('Welcome to Real Python!')
Welcome to Real Python!
```

交互式会话将允许您测试您编写的每一段代码，这使它成为一个非常棒的开发工具，也是一个试验语言和动态测试 Python 代码的绝佳场所。

要退出交互模式，您可以使用以下选项之一:

*   `quit()`或`exit()`，内置函数
*   Windows 上的 `Ctrl` + `Z` 和 `Enter` 组合键，或者类 Unix 系统上的 `Ctrl` + `D`

**注意:**使用 Python 时要记住的第一条经验法则是，如果你对一段 Python 代码的作用有疑问，那么就启动一个交互式会话，尝试一下看看会发生什么。

如果您从未使用过命令行或终端，那么您可以尝试这样做:

*   在 Windows 上，命令行通常被称为命令提示符或 MS-DOS 控制台，它是一个名为`cmd.exe`的程序。该程序的路径会因系统版本的不同而有很大差异。

    快速访问它的方法是按下 `Win` + `R` 组合键，这将把你带到*运行*对话框。一旦你到了那里，输入`cmd`并按下 `Enter` 。

*   在 GNU/Linux(和其他 Unixes)上，有几个应用程序可以让您访问系统命令行。一些最流行的是 xterm、Gnome 终端和 Konsole。这些工具运行 shell 或终端，如 Bash、ksh、csh 等等。

    在这种情况下，到这些应用程序的路径要多得多，并且取决于发行版，甚至取决于您使用的桌面环境。所以，你需要阅读你的系统文档。

*   在 Mac OS X 上，您可以从*应用程序→实用程序→终端*访问系统终端。

[*Remove ads*](/account/join/)

## 解释器如何运行 Python 脚本？

当您尝试运行 Python 脚本时，一个多步骤的过程开始了。在此过程中，口译员将:

1.  **按顺序处理脚本语句**

2.  将源代码编译成称为字节码的中间格式

    这个字节码是将代码翻译成独立于平台的低级语言。其目的是优化代码执行。所以，下一次解释器运行你的代码时，它会绕过这个编译步骤。

    严格来说，这种代码优化只针对模块(导入的文件)，不针对可执行脚本。

3.  **发送要执行的代码**

    此时，一种称为 Python 虚拟机(PVM)的东西开始发挥作用。PVM 是 Python 的运行时引擎。这是一个循环，它遍历字节码中的指令，一个接一个地运行它们。

    PVM 不是 Python 的一个独立组件。它只是你安装在机器上的 Python 系统的一部分。从技术上讲，PVM 是 Python 解释器的最后一步。

运行 Python 脚本的整个过程被称为 **Python 执行模型**。

**注意:**Python 执行模型的这个描述对应的是语言的核心实现，也就是 CPython。由于这不是一个语言要求，它可能会受到未来的变化。

## 如何使用命令行运行 Python 脚本

Python 交互式会话将允许您编写许多行代码，但是一旦您关闭会话，您将丢失您所编写的所有内容。这就是为什么编写 Python 程序的通常方式是使用纯文本文件。按照惯例，这些文件将使用`.py`扩展名。(在 Windows 系统上，扩展名也可以是`.pyw`。)

Python 代码文件可以用任何纯文本编辑器创建。如果你是 Python 编程新手，可以试试 [Sublime Text](https://realpython.com/products/sublime-python/) ，这是一个功能强大且易于使用的编辑器，但是你可以使用任何你喜欢的编辑器。

为了继续学习本教程，您需要创建一个测试脚本。打开您最喜欢的文本编辑器，编写以下代码:

```py
 1#!/usr/bin/env python3
 2
 3print('Hello World!')
```

用名称`hello.py`将文件保存在您的工作目录中。准备好测试脚本后，您可以继续阅读。

### 使用`python`命令

要使用`python`命令运行 Python 脚本，您需要打开一个命令行，键入单词`python`，或者如果您有两个版本，键入`python3`，后跟脚本的路径，如下所示:

```py
$ python3 hello.py
Hello World!
```

如果一切正常，按下 `Enter` 后，你会在屏幕上看到短语`Hello World!`。就是这样！您刚刚运行了您的第一个 Python 脚本！

如果这不能正常工作，也许你需要检查你的系统 [`PATH`](https://realpython.com/add-python-to-path/) ，你的 Python 安装，你创建`hello.py`脚本的方式，你保存它的地方，等等。

这是运行 Python 脚本的最基本和最实用的方法。

### 重定向输出

有时保存脚本的输出供以后分析会很有用。你可以这样做:

```py
$ python3 hello.py > output.txt
```

这个操作将脚本的输出重定向到`output.txt`，而不是标准的系统输出(`stdout`)。这个过程通常被称为流重定向，在 Windows 和类 Unix 系统上都可用。

如果`output.txt`不存在，那么它是自动创建的。另一方面，如果文件已经存在，那么它的内容将被新的输出替换。

最后，如果你想把连续执行的输出加到`output.txt`的末尾，那么你必须用两个尖括号(`>>`而不是一个，就像这样:

```py
$ python3 hello.py >> output.txt
```

现在，输出将被附加到`output.txt`的末尾。

[*Remove ads*](/account/join/)

### 使用`-m`选项运行模块

Python 提供了一系列命令行选项，您可以根据自己的需要来使用。例如，如果你想运行一个 Python 模块，你可以使用命令`python -m <module-name>`。

`-m`选项在`sys.path`中搜索模块名，并以`__main__`的身份运行其内容:

```py
$ python3 -m hello
Hello World!
```

**注意:** `module-name`需要是模块对象的名称，而不是字符串。

### 使用脚本文件名

在最新版本的 Windows 上，只需在命令提示符下输入包含代码的文件名，就可以运行 Python 脚本:

```py
C:\devspace> hello.py
Hello World!
```

这是可能的，因为 Windows 使用系统注册表和文件关联来确定使用哪个程序来运行特定文件。

在类似 Unix 的系统上，比如 GNU/Linux，您可以实现类似的东西。您只需添加第一行文本`#!/usr/bin/env python`，就像您对`hello.py`所做的那样。

对于 Python 来说，这是一个简单的注释，但是对于操作系统来说，这一行表示必须使用什么程序来运行文件。

这一行以`#!`字符组合开始，通常称为**哈希邦**或**舍邦**，并继续到解释器的路径。

有两种方法可以指定解释器的路径:

*   **`#!/usr/bin/python` :** 书写绝对路径
*   **`#!/usr/bin/env python` :** 使用操作系统`env`命令，通过搜索`PATH`环境变量定位并执行 Python

如果您记住不是所有的类 Unix 系统都将解释器放在同一个地方，那么最后一个选项是有用的。

最后，要执行这样的脚本，您需要为它分配执行权限，然后在命令行输入文件名。

下面是一个如何做到这一点的示例:

```py
$ # Assign execution permissions
$ chmod +x hello.py
$ # Run the script by using its filename
$ ./hello.py
Hello World!
```

有了执行权限和适当配置的 shebang 行，您只需在命令行输入文件名就可以运行该脚本。

最后，您需要注意，如果您的脚本不在您当前的工作目录中，您将不得不使用文件路径来使这个方法正确工作。

[*Remove ads*](/account/join/)

## 如何交互式运行 Python 脚本

也可以从交互式会话中运行 Python 脚本和模块。这个选项为您提供了多种可能性。

### 利用`import`

当您[导入一个模块](https://realpython.com/absolute-vs-relative-python-imports/)时，真正发生的是您加载它的内容供以后访问和使用。这个过程有趣的地方在于 [`import`](https://realpython.com/absolute-vs-relative-python-imports/) 运行代码作为它的最后一步。

当模块只包含类、函数、变量和常量定义时，您可能不会意识到代码实际上在运行，但是当模块包含对函数、方法或其他生成可见结果的语句的调用时，您将会看到它的执行。

这为您提供了另一个运行 Python 脚本的选项:

>>>

```py
>>> import hello
Hello World!
```

您必须注意，这个选项在每个会话中只起一次作用。在第一个`import`之后，连续的`import`执行什么都不做，即使你修改了模块的内容。这是因为`import`操作成本很高，因此只运行一次。这里有一个例子:

>>>

```py
>>> import hello  # Do nothing
>>> import hello  # Do nothing again
```

这两个`import`操作什么都不做，因为 Python 知道`hello`已经被导入了。

这种方法的工作有一些要求:

*   包含 Python 代码的文件必须位于您当前的工作目录中。
*   该文件必须位于 [Python 模块搜索路径(PMSP)](https://realpython.com/python-modules-packages/#the-module-search-path) 中，Python 会在这里查找您导入的模块和包。

要了解您当前的 PMSP 中有什么，您可以运行以下代码:

>>>

```py
>>> import sys
>>> for path in sys.path:
...     print(path)
...
/usr/lib/python36.zip
/usr/lib/python3.6
/usr/lib/python3.6/lib-dynload
/usr/local/lib/python3.6/dist-packages
/usr/lib/python3/dist-packages
```

运行这段代码，您将获得 Python 在其中搜索您导入的模块的目录和`.zip`文件的列表。

### 使用`importlib`和`imp`

在 [Python 标准库](https://docs.python.org/3/library/index.html)中可以找到 [`importlib`](https://docs.python.org/3/library/importlib.html) ，这是一个提供`import_module()`的模块。

使用`import_module()`，您可以模拟一个`import`操作，并因此执行任何模块或脚本。看一下这个例子:

>>>

```py
>>> import importlib
>>> importlib.import_module('hello')
Hello World!
<module 'hello' from '/home/username/hello.py'>
```

一旦你第一次导入了一个模块，你将不能继续使用`import`来运行它。在这种情况下，您可以使用`importlib.reload()`，这将强制解释器再次重新导入模块，就像下面的代码一样:

>>>

```py
>>> import hello  # First import
Hello World!
>>> import hello  # Second import, which does nothing
>>> import importlib
>>> importlib.reload(hello)
Hello World!
<module 'hello' from '/home/username/hello.py'>
```

这里需要注意的重要一点是，`reload()`的参数必须是模块对象的名称，而不是字符串:

>>>

```py
>>> importlib.reload('hello')
Traceback (most recent call last):
    ...
TypeError: reload() argument must be a module
```

如果你使用一个字符串作为参数，那么`reload()`将引发一个`TypeError`异常。

**注意:**为了节省空间，前面代码的输出被缩写为(`...`)。

当您正在修改一个模块，并且想要测试您的更改是否有效，而又不离开当前的交互会话时，这个功能就派上了用场。

最后，如果您使用的是 Python 2.x，那么您将拥有`imp`，这是一个提供名为`reload()`的函数的模块。`imp.reload()`的工作原理与`importlib.reload()`类似。这里有一个例子:

>>>

```py
>>> import hello  # First import
Hello World!
>>> import hello  # Second import, which does nothing
>>> import imp
>>> imp.reload(hello)
Hello World!
<module 'hello' from '/home/username/hello.py'>
```

在 Python 2.x 中，`reload()`是一个内置函数。在 2.6 和 2.7 版本中，它也被包含在`imp`中，以帮助过渡到 3.x。

**注意:** `imp`自该语言 3.4 版本起已被弃用。`imp`方案正等待有利于`importlib`的否决。

[*Remove ads*](/account/join/)

### 使用`runpy.run_module()`和`runpy.run_path()`

标准库包括一个名为 [`runpy`](https://docs.python.org/3/library/runpy.html) 的模块。在这个模块中，你可以找到`run_module()`，这是一个可以让你不用先导入模块就可以运行模块的功能。这个函数返回被执行模块的`globals`字典。

这里有一个你如何使用它的例子:

>>>

```py
>>> runpy.run_module(mod_name='hello')
Hello World!
{'__name__': 'hello',
 ...
'_': None}}
```

使用标准的`import`机制定位模块，然后在新的模块[命名空间](https://realpython.com/python-namespaces-scope/)上执行。

`run_module()`的第一个参数必须是带有模块绝对名称的字符串(不带`.py`扩展名)。

另一方面，`runpy`还提供了`run_path()`，这将允许您通过提供模块在文件系统中的位置来运行模块:

>>>

```py
>>> import runpy
>>> runpy.run_path(path_name='hello.py')
Hello World!
{'__name__': '<run_path>',
 ...
'_': None}}
```

像`run_module()`，`run_path()`返回被执行模块的`globals`字典。

`path_name`参数必须是一个字符串，可以引用以下内容:

*   Python 源文件的位置
*   编译后的字节码文件的位置
*   `sys.path`中有效条目的值，包含一个`__main__`模块(`__main__.py`文件)

### 黑客`exec()`

到目前为止，您已经看到了运行 Python 脚本最常用的方法。在本节中，您将看到如何通过使用 [`exec()`](https://realpython.com/python-exec/) 来实现这一点，这是一个支持 Python 代码动态执行的内置函数。

`exec()`提供了运行脚本的另一种方式:

>>>

```py
>>> exec(open('hello.py').read())
'Hello World!'
```

该语句打开`hello.py`，读取其内容，并发送给`exec()`，最后运行代码。

上面的例子有点离谱。这只是一个“黑客”，它向您展示了 Python 是多么的通用和灵活。

### 使用`execfile()`(仅限 Python 2.x】

如果您喜欢使用 Python 2.x，可以使用一个名为`execfile()`的内置函数，它能够运行 Python 脚本。

`execfile()`的第一个参数必须是一个字符串，包含您要运行的文件的路径。这里有一个例子:

>>>

```py
>>> execfile('hello.py')
Hello World!
```

这里，`hello.py`被解析和评估为一系列 Python 语句。

[*Remove ads*](/account/join/)

## 如何从 IDE 或文本编辑器运行 Python 脚本

当开发更大更复杂的应用时，建议您使用[集成开发环境(IDE)或高级文本编辑器](https://realpython.com/python-ides-code-editors-guide/)。

这些程序中的大多数都提供了从环境内部运行脚本的可能性。它们通常包含一个*运行*或*构建*命令，这通常可以从工具栏或主菜单中获得。

Python 的标准发行版包括作为默认 IDE 的 [IDLE](https://realpython.com/python-idle/) ，你可以用它来编写、调试、修改和运行你的模块和脚本。

其他 ide 如 Eclipse-PyDev、PyCharm、Eric 和 NetBeans 也允许您从环境内部运行 Python 脚本。

像 [Sublime Text](https://www.sublimetext.com) 和 [Visual Studio Code](https://code.visualstudio.com/docs) 这样的高级文本编辑器也允许你运行你的脚本。

为了掌握如何从您喜欢的 IDE 或编辑器中运行 Python 脚本的细节，您可以看一下它的文档。

## 如何从文件管理器运行 Python 脚本

通过双击文件管理器中的图标来运行脚本是运行 Python 脚本的另一种可能方式。这个选项可能不会在开发阶段广泛使用，但是当您发布代码用于生产时可能会用到。

为了能够双击运行您的脚本，您必须满足一些取决于您的操作系统的条件。

例如，Windows 将扩展名`.py`和`.pyw`分别与程序`python.exe`和`pythonw.exe`相关联。这允许您通过双击脚本来运行它们。

当您有一个带有命令行界面的脚本时，您很可能只能在屏幕上看到一个黑色的闪烁窗口。为了避免这种恼人的情况，您可以在脚本末尾添加类似于`input('Press Enter to Continue...')`的语句。这样，程序会停止，直到你按下 `Enter` 。

不过，这种技巧也有缺点。例如，如果您的脚本有任何错误，执行将在到达`input()`语句之前中止，您仍然无法看到结果。

在类似 Unix 的系统上，您可以通过在文件管理器中双击脚本来运行它们。要实现这一点，您的脚本必须有执行权限，并且您需要使用您已经看到的 shebang 技巧。同样，对于命令行界面脚本，您可能在屏幕上看不到任何结果。

因为通过双击执行脚本有一些限制，并且取决于许多因素(例如操作系统、文件管理器、执行权限、文件关联)，所以建议您将它视为已经调试好并准备投入生产的脚本的一个可行选项。

## 结论

通过阅读本教程，您已经掌握了在各种情况和开发环境下以多种方式运行 Python 脚本和代码所需的知识和技能。

**您现在可以从**运行 Python 脚本了

*   操作系统命令行或终端
*   Python 交互模式
*   您最喜欢的 IDE 或文本编辑器
*   系统的文件管理器，双击脚本图标

这些技能将使你的开发过程更快，更有效率和灵活性。

***参加测验:****通过我们的交互式“如何运行您的 Python 脚本”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/run-python-scripts/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**运行 Python 脚本**](/courses/running-python-scripts/)**********