# IPython(替代 Python Shell)综述

> 原文：<https://www.pythoncentral.io/review-of-ipython-alternative-python-shell/>

关于 IPython 的介绍，请务必阅读文章[IPython 简介:一个增强的 Python 解释器](https://www.pythoncentral.io/ipython-introduction-enhanced-python-interpreter/ "Introduction to IPython: An Enhanced Python Interpreter")。

使用 Python 这样的解释型语言的好处之一是使用交互式 Python shell 进行探索性编程。它让您可以快速、轻松地进行尝试，而无需编写脚本并执行它，并且您可以在进行过程中轻松地检查值。shell 也有一些便利之处，包括逐行历史记录和用于阅读帮助的内置分页器。

不过，标准 Python shell 有一些限制:您最终会频繁地使用`dir(x)`来检查对象并找到您正在寻找的方法名。将您从 shell 中编写的代码保存到一个格式良好的方便文件中也很有挑战性。最后，如果您习惯了 IDE 的便利，您将会错过诸如语法高亮、代码完成、函数签名帮助等功能。此外，当使用 TkInter 以外的工具包测试 GUI 开发时，GUI 工具包的主循环会阻塞交互式解释器。

然而，Python 有一些替代的 shells，它们为基本接口的行为提供了一些扩展。IPython 可能是其中最古老的，它提供了一些非常深刻的功能，但忽略了一些肤浅的功能，而 bpython 和 DreamPie 是较新的，在 IPython 忽略的一些功能上做得很好。我们将从年龄先于美貌开始，逐一探讨。—使用 IPython。

## IPython 概述

IPython 旨在实现几个目的:它希望成为常规 Python shell 的更强大的替代方案；作为您自己的 Python 程序的调试界面或交互式控制台；作为使用 Python 的编程系统的基础，例如科学编程或类似用途；并作为 GUI 工具包的交互界面。IPython 还开发了用于协作、交互式并行编程的工具。

IPython 运行在 Linux、Windows 和 OS X 上，可以使用`easy_install`或`pip`轻松安装。(如果您想使用可选的 Qt 接口，您还需要安装 PyQt)。一旦安装完成，您就可以通过从命令行执行`ipython`(或者，对于 Qt 版本，`ipython qtconsole`来启动它。以下是其功能概述:

## IPython 代码完成和语法突出显示

IPython 提供了区分大小写的制表符补全，不仅包括对象和关键字的名称，还包括可用模块(包括当前工作目录中的模块)以及文件名和目录名。除了键入以外，没有其他方法可以选择其中一个选项，直到它成为唯一的选项。选项要么显示在光标下方的单行中，要么如果选项太多，则显示在几列中。

在 Qt 控制台中，当屏幕上有太多选择时，这种显示就成了一个问题。例如，如果你这样做:`import gtk gtk.`它会给你一个无法管理的可能选择的长列表，并把你的光标推离屏幕。不过，您可以通过按下`q`或`Esc`来恢复，这将清除选项视图并返回到您的光标处。在常规终端版本中，选项列表被打印到屏幕上，然后光标所在的行再次出现，这更容易使用，尽管很难浏览选项列表。

总而言之，使用 IPython 完成代码是准确和可行的，但没有想象中那么方便。

正在输入的代码的语法高亮显示在常规的控制台视图中不可用，但是在 Qt 控制台中提供了。然而，它不是特别可定制的。有三个选项，NoColor、Linux 和 LightBG，你可以使用 IPython 的`%colors`魔法在它们之间切换，例如:`%colors linux`切换到 Linux 着色方案。在页面中显示 Python 文件内容的`%pycat%`魔术在两个界面中都进行语法高亮显示。

## IPython 魔法

你可能已经注意到上面提到的“魔法”；IPython 提供了各种控制或改变外壳行为的神奇功能。上面我们看到了`%colors`和`%pycat`，但是还有更多。以下是一些亮点:

*   **%自动呼叫**:自动在呼叫中插入括号，如`range 3 5`
*   **%debug** :调试当前环境
*   **%edit** :运行文本编辑器并执行其输出
*   **%gui** :指定一个 gui 工具包，允许在其事件循环运行时进行交互
*   **%**
*   **%loadpy** :从文件名或 URL(！)
*   **%登录**和**%注销**:开启和关闭登录
*   **%macro** :为历史中的一系列行命名，以便重复
*   **%pylab** :加载`numpy`和`matplotlib`交互使用
*   **%quickref** :加载快速参考指南
*   **%**
*   **%**
*   **%**
*   **%**
*   **%timeit** :使用 Python 的`timeit`来计时语句、表达式或块的执行

而那些，就像我说的，只是亮点。IPython 有很多功能，其中很多功能并不是立即可见的。不过，在线和内置的文档都很棒。

## IPython 的帮助系统

IPython 的帮助系统特别方便——你只需键入你想看到描述的对象或函数的名字，然后加上`?`；IPython 将显示有问题的对象的文档字符串、定义、源、源文件等等。不够？用两个问号再试一次，你会得到更多。

例如，导入`urllib2`然后输入`urllib2?`会产生以下结果:

```py

Type: module

Base Class: <type 'module'>

String Form:<module 'urllib2' from '/usr/lib/python2.7/urllib2.pyc'>

Namespace: Interactive

File: /usr/lib/python2.7/urllib2.py

Docstring:  An extensible library for opening URLs using a variety of protocols
# ...删减了 350 个单词...
这是它的用法的一个例子:
import urllib2
#设置认证信息
 authinfo = urllib2。HTTPBasicAuthHandler()
authinfo . add _ password(realm = ' PDQ Application '，uri = ' https://localhost:8092/site-updates . py '，user='username '，passwd = ' password ')
proxy _ support = URL lib 2。proxy handler({ ' http ':' http://proxy hostname:3128 ' })
#构建一个新的 opener，增加认证和缓存 FTP 处理程序
opener = urllib2 . Build _ opener(proxy _ support，authinfo，URL lib 2。CacheFTPHandler)
# Install it
URL lib 2 . Install _ opener(开启器)
f = URL lib 2 . urlopen(' http://www . python . org/')

```

请注意，我从结果中截取了 350 个单词的描述！输入`urllib2??`提供了上述所有内容以及该模块的完整源代码。

IPython 的帮助也适用于内置的神奇命令，它有一个内置的 IPython 概述(只需通过`?`访问)，以及一个快速参考指南(`%quickref`)。简单来说，IPython 的帮助很大。

## 其他工具

IPython 提供了许多其他 Python shells 所没有的工具，包括用于并行编程和与多个 GUI 工具包交互工作的工具。

### IPython 中的并行编程

IPython 的一个主要部分是它的并行编程结构，这是内置的，看起来非常强大——但这在很大程度上超出了本文的范围。(附带说明:默认情况下，`IPython.parallel`包的一些依赖项不会随 IPython 一起安装。)

### 交互式 GUI 编程

交互式 GUI 编程是 IPython 允许的活动之一，它非常适合这种活动。要启用它，请使用带有参数的`%gui`魔法函数，指定您正在使用哪个 GUI 工具包:

*   **%gui wx** :启用 wxPython 事件循环集成
*   **%gui qt4|qt** :启用 PyQt4 事件循环集成
*   **%gui gtk** :启用 PyGTK 事件循环集成
*   **%gui tk** :启用 tk 事件循环集成
*   **%gui OSX** :启用 Cocoa 事件循环集成(需要%matplotlib 1.1)
*   **%gui** :禁用所有事件循环集成

然后，您可以像往常一样构建您的 UI，除了当您调用工具包的主循环(例如 TkInter 中的`root.mainloop()`或 PyGTK 中的`gtk.main()`)时，您的用户界面将在您继续交互发出命令时运行。

## IPython 审查摘要

IPython 可以做的比我在这里描述的多得多——它包含很多功能，尽管它有一个相当长的学习曲线。它可能不适合所有人；对于许多人来说，我将讨论的其他选项之一可能提供了他们实际使用的所有功能，但复杂性要低得多，但它确实值得一看。让我们来讨论一下 [bpython 和 DreamPie](https://www.pythoncentral.io/review-of-bpython-and-dreampie-alternative-python-shells/ "Review of bpython and DreamPie (alternative Python shells)") 。