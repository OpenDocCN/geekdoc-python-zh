# Java vs Python:面向 Java 开发人员的基础 Python

> 原文：<https://realpython.com/java-vs-python/>

Python 是一种通用编程语言。通过考虑它对学习的可接近性和它对数据分析、机器学习和 web 开发的高度适用性，你可以理解它在过去几年中的增长。但是它是一种什么样的编程语言呢？当你比较 Java 和 Python 时，有什么不同？你能用它做什么？而且真的像有些人宣称的那样“简单易学”吗？

在本教程中，您将从 Java 的角度探索 Python。阅读完之后，您将能够决定 Python 是否是解决您的用例的可行选项，并了解何时可以将 Python 与 Java 结合使用来解决某些类型的问题。

在本教程中，您将了解到:

*   通用 Python 编程**语言语法**
*   最相关的标准**数据类型**
*   Java 与 Python 的**差异**和**相似之处**
*   高质量 Python 的资源**文档**和**教程**
*   一些 Python 社区最喜欢的**框架**和**库**
*   **从头开始**Python 编程的方法

本教程面向熟悉 Java 内部工作原理、概念、术语、类、类型、集合框架等的软件开发人员。

你根本不需要有任何 Python 经验。

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## Python 从何而来？

Python 是由吉多·范·罗苏姆开发的一种编程语言。1989 年圣诞节期间，他一直在寻找一个业余编程项目，让自己有事可做，于是他开始开发 Python 解释器。

Python 起源于多种语言: [ABC](https://homepages.cwi.nl/~steven/abc/ "The ABC Programming Language: a short introduction") 、 [C](https://en.wikipedia.org/wiki/C_(programming_language) "C (programming language) - Wikipedia") 和 [Modula-3](https://en.wikipedia.org/wiki/Modula-3 "Modula-3 - Wikipedia") 。它基本上是一种面向对象的命令式编程语言。

根据您的偏好和所需的功能，它可以应用于完全面向对象的风格，也可以应用于带有函数的过程化编程风格。面向对象的功能将在本教程的[后](#class-based-object-orientation)讨论。

**注意:**为了清楚起见，从 Java 的角度来看，Python 函数就像**静态方法**，你不一定需要在一个类内定义它们。[稍后](#indentation-for-code-block-grouping)，你会看到一个 Python 函数定义的例子。

此外，更函数式的编程风格也是完全可能的。要了解更多，您需要探索 [Python 的函数式编程能力](https://realpython.com/python-functional-programming/ "Functional Programming in Python: When and How to Use It – Real Python")。

2021 年初，TIOBE 第四次宣布 Python 为年度编程语言[。截至 2021 年](https://www.tiobe.com/tiobe-index/python/ "Tiobe: Python") [Octoverse 报道](https://octoverse.github.com/#top-languages-over-the-years "Octoverse: Top languages over the years")，Python 在 GitHub 上被知识库贡献者评为第二受欢迎的语言。

[*Remove ads*](/account/join/)

## Python 的哲学是什么？

很快，您将在本节之后的[小节中亲身体验 Python。但是，首先，您将通过研究一些可以追溯到 Python 哲学的特性来探索为什么更好地了解 Python 是值得的。](#how-can-you-start-discovering-python)

Java 和 Python 背后的一些思想是相似的，但是每种编程语言都有自己独特的特点。Python 的哲学被捕获为十九个指导原则的集合，即 Python 的禅。Python 藏了几个复活节彩蛋，其中一个就是 Python 的禅。考虑当您在 Python[read–eval–print 循环(REPL)](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) 中发出以下命令时会发生什么:

>>>

```py
>>> import this
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

虽然你不应该把上面的陈述看得太重，但是其中一些与你接下来要讲的特征直接相关。

**注意:**Python read–eval–print 循环将在本教程的后面解释。

通过考虑 Python 禅宗的指导原则，您会对如何使用这种语言有一个很好的想法。

### Python 代码可读

如果您来自 Java 背景，并且看到一段典型的 Python 代码，您可能会认为您看到的是伪代码。有几个因素促成了这一点:

*   **缩进**用于语句分组。这使得代码块更短，并促进了统一的编码风格。稍后你会发现更多关于这个主题的内容。
*   一些内置的高级数据结构，加上一组适当的操作符符号，使得 Python 非常具有表现力。
*   选择使用[异常](https://realpython.com/python-exceptions/ "Python Exceptions: An Introduction – Real Python")作为处理错误的主要方式保持了代码的整洁。
*   Python 程序员更喜欢一种编码风格，这种风格的灵感来自于请求原谅比请求许可更容易(EAFP)的概念，而不是三思而后行(LBYL)的概念。这种风格将重点放在程序的正常的、[愉快的路径](https://en.wikipedia.org/wiki/Happy_path "Happy path - Wikipedia")上，并且您将在之后弄清楚如何处理任何异常。有关这两种编码风格的更多信息，请查看 [LBYL vs EAFP:防止或处理 Python 中的错误](https://realpython.com/python-lbyl-vs-eafp/)。

在本教程中，以及在其他链接资源中，您可以找到一些例子来说明这一点。

### Python 自带电池

Python 的目标是你可以用 Python 的标准发行版解决大多数日常问题。为此，Python 包含了所谓的[标准库](https://docs.python.org/3/library/index.html "The Python Standard Library — Python 3 documentation")。就像 [Java 类库](https://en.wikipedia.org/wiki/Java_(programming_language)#Class_libraries "Java (programming language) - Wikipedia: Class libraries")，它是由常量、函数、类和框架组成的有用工具的广泛集合。

要进一步了解 Python 标准库，请查看 Python 文档的 [Python 教程](https://docs.python.org/3/tutorial/ "The Python Tutorial — Python 3 documentation")中标准库简介的[第一部分](https://docs.python.org/3/tutorial/stdlib.html "10\. Brief Tour of the Standard Library — Python 3 documentation")和[第二部分](https://docs.python.org/3/tutorial/stdlib2.html "11\. Brief Tour of the Standard Library — Part II — Python 3 documentation")。

### Python 提倡代码重用

Python 提供了几个特性，使您能够开发可以在不同地方重用的代码，以应用[不要重复自己(DRY)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself "Don't repeat yourself - Wikipedia") 原则。

一个特点是，你通常将代码分解成 Python 中的[模块](https://docs.python.org/3/glossary.html#term-module "Glossary — Python 3 documentation: module")和[包](https://docs.python.org/3/glossary.html#term-package "Glossary — Python 3 documentation: package")。但是，请注意，Python 模块和包不同于 Java 模块和包。如果你想从 Python 开发者的角度了解更多这些概念，你可以阅读关于 [Python 模块和包](https://realpython.com/python-modules-packages/ "Python Modules and Packages – An Introduction – Real Python")的内容。

Python 中可以使用的另一种技术是面向对象编程。您将在本教程的部分探索这个[。](#class-based-object-orientation)

您还可以使用[decorator](https://realpython.com/primer-on-python-decorators/ "Primer on Python Decorators – Real Python")来修改 Python 函数、类或方法。这是另一种技术，因此您可以只对功能进行一次编程，之后就可以从您修饰的任何函数、类或方法中使用它。

### Python 很容易扩展

对模块和包的支持是使 Python 易于扩展新功能的要素之一。您还可以通过重载[Python 标准操作符和函数来定义新的或适应的行为。甚至有可能影响](https://en.wikipedia.org/wiki/Operator_overloading "Operator overloading - Wikipedia")[类是如何被创建的](https://realpython.com/python-metaclasses/ "Python Metaclasses – Real Python")。

扩展 Python 最直接的方法是用纯 Python 编写代码。您还可以使用 Python 的一种简化方言(称为 Cython)或 C 或 C++中的绑定通过[来定义模块。](https://realpython.com/python-bindings-overview/ "Python Bindings: Calling C or C++ From Python – Real Python")

[*Remove ads*](/account/join/)

## 如何开始发现 Python？

在本教程中，您会发现一些例子，这些例子可能会鼓励您探索某些东西或者亲自尝试 Python 代码片段。作为一名 Java 开发人员，您可能还记得熟悉 Java 和安装第一个 Java 开发工具包的第一步。同样，如果您想开始使用 Python，您首先需要安装它，然后创建一个沙箱，在那里您可以安全地进行实验。

已经有几个教程解释了如何做到这一点，所以接下来的小节将向您介绍这些资源。

### 安装 Python

第一步是安装 Python 的最新版本。为此，请遵循本 [Python 3 安装和设置指南](https://realpython.com/installing-python/ "Python 3 Installation & Setup Guide – Real Python")。

另一个可以找到安装说明的地方是[官方 Python 下载页面](https://www.python.org/downloads/ "Download Python | Python.org")。

**注意:**确保你已经安装了最新版本的 Python。在撰写本教程时，最新版本是 [3.10.x 系列](https://realpython.com/python310-new-features/ "Python 3.10: Cool New Features for You to Try")的最新补丁版本。本教程中显示的代码片段应该都适用于这个版本的 Python。

许多 Python 开发人员贡献了支持各种 Python 版本的库，他们通常更喜欢尝试 Python 的预发布版本，而不会干扰他们的常规 Python 工作。在这些情况下，在同一台机器上访问多个版本的 Python 非常方便。提供那个功能的一个工具是 [`pyenv`](https://realpython.com/intro-to-pyenv/ "Managing Multiple Python Versions With pyenv – Real Python") ，堪比 Java 的 [jEnv](https://www.jenv.be/ "jEnv - Manage your Java environment") 。

### 创建沙盒并使用它

第二步，您应该建立一个虚拟环境，这样您就可以安全地利用开源 Python 生态系统。本节解释了您应该如何以及为什么这样做。

尽管 Python 附带了一个具有各种功能的广泛的[标准库](https://docs.python.org/3/library/index.html "The Python Standard Library — Python 3 documentation")，还有更多功能以**外部包**的形式提供，其中绝大多数是开源的。 [Python 包索引](https://pypi.org/ "PyPI · The Python Package Index")，或简称为 **PyPI** ，是收集和提供这些包的主要中央存储库。您可以使用 [`pip`](https://realpython.com/what-is-pip/ "What Is Pip? A Guide for New Pythonistas – Real Python") 命令安装软件包。但是，在此之前，请先阅读下面两段。

为了避免依赖版本冲突，通常不应该在项目之间共享您的全局或个人 Python 安装。实际上，每个项目或实验沙箱都有一个虚拟环境。

这样，您的项目保持相互独立。这种方法还可以防止包之间的版本冲突。如果你想更多地了解这个过程，那么你可以详细阅读如何[创建和激活你的虚拟环境](https://realpython.com/python-virtual-environments-a-primer/ "Python Virtual Environments: A Primer – Real Python")。

### 选择编辑器或集成开发环境

作为设置的最后一步，决定您想要使用哪个编辑器或 ide。如果你习惯了 IntelliJ，那么 [PyCharm](https://realpython.com/pycharm-guide/ "PyCharm for Productive Python Development (Guide) – Real Python") 似乎是合乎逻辑的选择，因为它属于同一系列的产品。另一个正在崛起的流行编辑器是 [Visual Studio Code](https://realpython.com/python-development-visual-studio-code/ "Python Development in Visual Studio Code – Real Python") ，但是你也可以从[中选择许多其他选项](https://realpython.com/python-ides-code-editors-guide/ "Python IDEs and Code Editors (Guide) – Real Python")。

在您安装了 Python，学习了如何将外部包安装到虚拟环境中，并选择了编辑器或 IDE 之后，您就可以开始尝试这种语言了。当您通读本教程的其余部分时，您会发现大量的实验和实践机会。

## Python 和 Java 有什么不同？

通过查看最显著的不同之处，您可以快速了解 Python 是哪种编程语言。在接下来的小节中，您将了解 Python 与 Java 最重要的不同之处。

### 代码块分组的缩进

或许 Python 最引人注目的特征是它的语法。特别是，您指定它的函数、类、流控制结构和代码块的方式与您可能习惯的方式非常不同。在 Java 中，用众所周知的花括号(`{`和`}`)来表示代码块。然而，在 Python 中，通过**缩进级别**来指示代码块。这里您可以看到一个演示缩进如何确定代码块分组的示例:

```py
 1def parity(number):
 2    result = "odd"                              # Function body
 3    if number % 2 == 0:
 4        result = "even"                         # Body of if-block
 5    return result                               # Not part of if-block
 6
 7for num in range(4):                            # Not part of function
 8    print("Number", num, "is", parity(num))     # Body of for-loop
 9print("This is not part of the loop")           # Not part of for-loop
```

该代码展示了一些新概念:

*   **第 1 行:**`def`语句开始定义一个名为`parity()`的新[函数](https://realpython.com/defining-your-own-python-function/ "Defining Your Own Python Function – Real Python")，它接受一个名为`number`的参数。注意，如果`def`语句出现在一个类定义块中，它将会启动一个[方法](https://realpython.com/instance-class-and-static-methods-demystified/ "Python's Instance, Class, and Static Methods Demystified – Real Python")定义。
*   **第 2 行:**在`parity()`内，函数体从**缩进层次**开始。第一条语句是将`"odd"`字符串赋给`result`变量。
*   **第 3 行:**这里你看到一个 [`if`语句](https://realpython.com/python-conditional-statements/#introduction-to-the-if-statement "Conditional Statements in Python – Real Python - Introduction to the if Statement")的开始。
*   **第 4 行:**额外的缩进开始一个新的块。当`if`语句的条件表达式`number % 2 == 0`评估为真时，执行该块。在这个例子中，它只由一行代码组成，在这里你将`"even"`赋值给`result`变量。
*   **第 5 行:**[`return`语句](https://realpython.com/python-return-statement/ "The Python return Statement: Usage and Best Practices – Real Python")之前的 **dedent** 标志着`if`语句及其相关块的结束。
*   **第 7 行:**同样，你可以看到在 [`for`循环](https://realpython.com/python-for-loop/ "Python 'for' Loops (Definite Iteration) – Real Python")开始之前的数据。因此，`for`循环从与函数定义块的第一行相同的缩进级别开始。它标志着函数定义块的结束。
*   **第 8 行:**你会看到同样的事情在`for`循环中再次发生。第一个 [`print()`函数](https://realpython.com/python-print/ "Your Guide to the Python print() Function – Real Python")调用是`for`循环块的一部分。
*   **第 9 行:**这个第二个定向的`print()`函数调用不是`for`循环块的一部分。

您可能已经注意到，行尾的冒号(`:`)引入了一个新的代码子块，应该缩进一级。当下一条语句再次重复时，该代码块结束。

代码块必须至少包含一条语句。空代码块是不可能的。在极少数不需要任何语句的情况下，你可以使用 [`pass`语句](https://realpython.com/python-pass/ "The pass Statement: How to Do Nothing in Python – Real Python")，它什么也不做。

最后，您可能还注意到，您可以使用散列符号(`#`)进行注释。

上述示例将产生以下输出:

```py
Number 0 is even
Number 1 is odd
Number 2 is even
Number 3 is odd
This is not part of the loop
```

虽然这种定义块的方式乍一看可能很奇怪，甚至可能会吓到你，但经验表明，人们会比你现在想象的更快地习惯它。

有一个对 Python 代码很有帮助的[风格指南](https://www.python.org/dev/peps/pep-0008/ "PEP 8: Style Guide for Python Code")，叫做 **PEP 8** 。它建议使用四个位置的缩进级别，使用空格。样式指南不建议在源代码文件中使用制表符。原因是不同的编辑器和系统终端可能会使用不一致的制表位位置，并为不同的用户或不同的操作系统呈现不同的代码。

**注意:**风格指南是一个 **Python 增强提议**的例子，或者简称为 **PEP** 。pep 不仅包含提议，还反映了实现的规范，因此您可以将 pep 比作 Java 的 jep 和 JSR 的联合。 [PEP 0](https://www.python.org/dev/peps/ "PEP 0 -- Index of Python Enhancement Proposals (PEPs) | Python.org") 列出了 PEP 的索引。

你可以在 PEP 8 中找到很多有趣的信息，包括 Python [命名约定](https://realpython.com/python-pep8/ "How to Write Beautiful Python Code With PEP 8 – Real Python")。如果你仔细阅读它们，你会发现它们与 Java 的略有不同。

[*Remove ads*](/account/join/)

### 从头开始的读取-评估-打印循环

从一开始，Python 就有一个内置的[读取-评估-打印循环(REPL)](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop "Read–eval–print loop - Wikipedia") 。REPL 读取尽可能短的完整语句、表达式或块，将其编译成字节码，然后进行求值。如果被评估的代码返回一个不同于`None`对象的对象，它输出这个对象的一个明确的[表示](https://docs.python.org/3/library/functions.html#repr "Built-in Functions — Python 3 documentation: repr()")。在本教程的后面你会找到对`None` [的解释。](#none)

**注:**你可以把 Python REPL 和 [Java 的 JShell (JEP 222)](https://openjdk.java.net/jeps/222 "JEP 222: jshell: The Java Shell (Read-Eval-Print Loop)") 对比一下，后者从 JDK 9 开始就有了。

以下片段展示了 Python 的 REPL 是如何工作的:

>>>

```py
>>> zero = int(0)
>>> zero
0
>>> float(zero)
0.0
>>> complex(zero)
0j
>>> bool(zero)
False
>>> str(zero)
'0'
```

正如您所看到的，解释器总是试图明确地显示结果表达式的值。在上面的例子中，您可以看到[整数](https://realpython.com/python-numbers/ "Numbers in Python – Real Python")、浮点、[复数](https://realpython.com/python-complex-numbers/ "Simplify Complex Numbers With Python – Real Python")、[布尔](https://realpython.com/python-boolean/ "Python Booleans: Optimize Your Code With Truth Values – Real Python")和[字符串](https://realpython.com/python-strings/ "Strings and Character Data in Python – Real Python")的值是如何以不同的方式显示的。

Java 和 Python 的区别在于赋值操作符(`=`)。使用单个等号的常规 Python 赋值是一个[语句](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements "7\. Simple statements — Python 3 documentation: Assignment statements")，而不是一个[表达式](https://docs.python.org/3/reference/expressions.html "6\. Expressions — Python 3 documentation")，后者会产生一些值或对象。

这解释了为什么 REPL 不打印对变量`zero`的赋值:语句总是评估为`None`。赋值后的一行包含变量表达式`zero`，指示 REPL 显示结果变量。

**注:** Python 3.8 引入了一个[赋值表达式](https://docs.python.org/3/reference/expressions.html#assignment-expressions "6\. Expressions — Python 3 documentation: Assignment expressions")运算符(`:=`)，也被称为[海象运算符](https://realpython.com/python-walrus-operator/ "The Walrus Operator: Python 3.8 Assignment Expressions")。它给变量赋值，但与常规赋值不同的是，它像在表达式中一样对变量求值。这类似于 Java 中的赋值操作符。但是，请注意，它不能与常规赋值语句完全互换。它的范围实际上相当有限。

在 REPL 中，下划线特殊变量(`_`)保存最后一个表达式的值，前提是它不是`None`。下面的片段展示了如何使用这个特殊变量:

>>>

```py
>>> 2 + 2
4
>>> _
4
>>> _ + 2
6
>>> some_var = _ + 1
>>> _
6
>>> some_var
7
```

将值`7`赋给`some_var`后，特殊变量`_`仍然保存值`6`。那是因为赋值语句评估为`None`。

### 动态类型和强类型

编程语言的一个重要特征是语言解释器或编译器何时、如何以及在何种程度上执行[类型验证](https://en.wikipedia.org/wiki/Type_system#Type_checking "Type system - Wikipedia: Type checking")。

Python 是一种**动态类型化的**语言。这意味着变量、函数参数和函数返回值的类型是在运行时检查的，而不是像 Java 那样在编译时检查。

Python 同时也是一种强类型语言:

*   每个对象都有一个与之相关联的特定类型。
*   不兼容类型之间需要显式转换。

在这里，您将探索 Python 如何在运行时检查类型兼容性，以及如何使类型兼容:

>>>

```py
>>> 40 + "2"
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
TypeError: unsupported operand type(s) for +: 'int' and 'str'
>>> 40 + int("2")  # Add two numbers
42
>>> str(40) + "2"  # Concatenate strings
'402'
>>> 40 * "2"       # Repeat string "2" forty times
'2222222222222222222222222222222222222222'
```

你会注意到，你不能只把一个整数值加到一个字符串值上。这在运行时受到保护。当解释器检测到运行时错误时，它会生成一个异常。REPL 捕捉`Exception`个实例，并显示导致错误表达式的[回溯](https://realpython.com/python-traceback/ "Understanding the Python Traceback – Real Python")。

要解决这个问题，您需要将一种类型转换为另一种类型。如果想将两个对象作为数字相加，可以使用`int()`构造函数将代表数字的字符串转换成普通数字。如果您想将两个对象连接成字符串，那么您可以使用`str()`构造函数将数字转换成字符串。

上面 Python 会话的最后一行显示了另一个特性。通过将一个[序列](https://docs.python.org/3/glossary.html#term-sequence "Glossary — Python 3 documentation: sequence")乘以一个数字，可以得到原始序列的串联结果，并按照给定的数字重复。

尽管 Python 是一种动态类型的语言，但是可以为代码提供[类型的注释](https://realpython.com/python-type-checking/ "Python Type Checking (Guide) – Real Python")。

在运行时，Python 不会对这些注释做任何事情，除了让它们可用于自省。但是，静态类型检查器工具可以检测类型声明和带类型注释的函数、类和变量的实际使用之间的不一致。

**注意:**如上所述，Python 运行时使类型注释可供代码自省。一些图书馆利用这些信息，比如  。

类型注释帮助您在代码开发周期的早期阶段检测错误。尤其是在大型软件项目中，它们帮助您使代码更易于维护，并保持代码库的良好状态。您通常调用静态类型检查器作为构建管道中验证步骤的一部分。大多数 ide 也使用类型注释。

[*Remove ads*](/account/join/)

### CPython vs JIT 编译器

与 Java 不同，Python 的参考实现没有 JIT 编译器。到目前为止，使用最多的 Python 实现是 **CPython** 。这也是**参考实现**。 [CPython](https://realpython.com/cpython-source-code-guide/) 是一个用 C 语言编写的编译器和解释器，几乎可以在任何可以想到的平台上使用。

CPython 分两步加载一个被称为[模块](https://docs.python.org/3/glossary.html#term-module "Glossary — Python 3 documentation: module")的源文件:

1.  **编译:**首先，CPython 读取代码，编译成[字节码](https://en.wikipedia.org/wiki/Bytecode "Bytecode - Wikipedia")，这是 CPython 字节码解释器可以执行的指令序列。在有限的范围内，您可以将编译阶段与 Java 的 [`javac`](https://docs.oracle.com/en/java/javase/16/docs/specs/man/javac.html "The javac Command") 如何将一个`.java`文件编译成一个`.class`文件进行比较。
2.  **执行:**CPython 字节码解释器——换句话说，CPython 的[虚拟机(VM)](https://en.wikipedia.org/wiki/Virtual_machine#Process_virtual_machines "Virtual machine - Wikipedia: Process virtual machines")——随后从第一步开始执行字节码。

**注意:**不像在 Java 中，你不能假设同一个 Python 字节码可以与其他 Python 实现一起工作，甚至可以在同一个 Python 实现的不同版本之间工作。但是，它有助于减少加载模块所需的时间。

如果可能的话，[编译后的模块](https://docs.python.org/3/tutorial/modules.html#compiled-python-files "6\. Modules — Python 3 documentation: "Compiled" Python Files")存储在[缓存](https://en.wikipedia.org/wiki/Cache_(computing) "Cache (computing) - Wikipedia")目录中。

与主流的 Java VM 实现不同，CPython 不会*而不是*随后将字节码编译成[本机目标代码](https://en.wikipedia.org/wiki/Object_code "Object code - Wikipedia")。然而，还有其他不同的 Python 实现:

*   有一个用于 Java 平台的 Python 实现叫做 [Jython](https://www.jython.org/ "Home | Jython") 。它运行在 JVM 中，Java 和 Python 之间有直接的互操作性。
*   同样，有一个名为 [IronPython](https://ironpython.net/ "IronPython.net /") 的版本运行在。NET 平台。
*   有一个实现使用了名为 [PyPy](https://www.pypy.org/ "PyPy") 的[实时(JIT)](https://en.wikipedia.org/wiki/Just-in-time_compilation "Just-in-time compilation - Wikipedia") 编译器。平均来说，PyPy 比 CPython 快 4.2 倍。
*   最后， [GraalVM](https://www.graalvm.org/docs/introduction/ "GraalVM Documentation") 是一个支持许多编程语言的高性能运行时。它为最近的 Python 版本提供了[实验支持](https://www.graalvm.org/docs/getting-started/#run-python "Getting Started with GraalVM: Run Python")。

上面的列表并不详尽。Python 站点包含了一个[备选实现](https://www.python.org/download/alternatives/ "Alternative Python Implementations | Python.org")和发行版的列表。

### 内置函数和运算符重载

作为一名 Java 开发人员，你可能知道术语**重载**和[方法重载](https://docs.oracle.com/javase/specs/jls/se14/html/jls-8.html#jls-8.4.9)。虽然 Python 中有一个动态等价函数 [`@singledispatchmethod`](https://realpython.com/python-multiple-constructors/#providing-multiple-constructors-with-singledispatchmethod) 提供了类似的功能，但 Python 中还有另一种重载，您可能会发现它更有用。

您可以为任何符合条件的 Python 内置函数和运算符定义自定义构建类的新行为。

**注意:**在这个上下文中，您可以认为合格的函数和操作符是那些允许您重载它们的行为的函数和操作符。

Python 提供了一种实现[函数和运算符重载](https://realpython.com/operator-function-overloading/ "Operator and Function Overloading in Custom Python Classes – Real Python")的便捷方式。

你可以通过在你的类中定义[特别命名的方法](https://docs.python.org/3/reference/datamodel.html#special-method-names "Data model / Special Method Names — Python 3 documentation")来尝试。这种方法的名称以两个下划线开始和结束，如`.__len__()`或`.__add__()`。具有这种名称的标识符被称为**双下划线**，是**双下划线** ( `__`)的缩写。

当您使用一个对象调用符合条件的内置函数时，对应的 dunder 方法会出现，Python 会将行为委托给该方法。同样，当您使用一个或多个操作数包含相应的 dunder 方法的操作符时，Python 会将行为委托给该方法。

例如，您可以定义`.__len__()`为内置的 [`len()`](https://realpython.com/len-python-function/) 函数提供行为。同样，您可以定义`.__add__()`来为加法运算符(`+`)提供行为。

这个特性使得将 Python 代码的漂亮、富于表现力和简洁的语法不仅应用于标准对象，也应用于定制对象成为可能。

### 集合函数处理的良好语法

在 Java 中，您可能已经通过组合调用`map()`、`filter()`和 lambda 表达式构建了列表。使用相同的函数和技术，您可以在 Python 中做同样的事情。使用这些结构并不总是能产生可读性最强的代码。

Python 为列表和其他集合的基本功能操作提供了一种优雅的替代语法。你可以对列表使用[列表理解](https://docs.python.org/3/glossary.html#term-list-comprehension "Glossary — Python 3 documentation: list comprehension"),对其他集合使用其他类型的理解。如果你想了解更多关于 Python 中的理解，那么你可以探索一下[何时使用列表理解](https://realpython.com/list-comprehension-python/ "When to Use a List Comprehension in Python – Real Python")。

[*Remove ads*](/account/join/)

### 一切都是物体

在 Java 中，并不是所有的东西都是对象，尽管事实上唯一可以放置代码的地方是在 Java 类内部。例如，Java 原语`42`不是一个对象。

就像 Java 一样，Python 也完全支持面向对象的编程风格。与 Java 不同的是**在 Python 中一切都是对象**。Python 对象的一些示例有:

*   [数值](https://realpython.com/python-numbers/ "Numbers in Python")
*   [文档字符串](https://realpython.com/documenting-python-code/ "Documenting Python Code: A Complete Guide – Real Python")
*   [功能和方法](https://realpython.com/defining-your-own-python-function/ "Defining Your Own Python Function")
*   [模块](https://realpython.com/python-modules-packages/ "Python Modules and Packages – An Introduction")
*   堆栈回溯
*   字节编译的代码对象
*   [类本身](https://realpython.com/python3-object-oriented-programming/ "Object-Oriented Programming (OOP) in Python 3")

因为它们是对象，所以您可以将所有这些存储在变量中，传递它们，并在运行时自省它们。

**注:**正如你在上面读到的，类是对象。因为根据定义，对象是类的实例，所以类*也必须*是某些东西的实例。

事实上，这些是一个[元类](https://docs.python.org/3/glossary.html#term-metaclass "Glossary — Python 3 documentation: metaclass")的实例。标准元类是`type`，但是你可以创建替代的[元类](https://realpython.com/python-metaclasses/ "Python Metaclasses – Real Python")，通常是从`type`派生而来，以改变类的创建方式。

元类，加上重载内置函数和操作符的能力，是 Python 成为多功能编程工具包的一部分。它们允许你创建你自己的可编程的额外的或者可选择的类和实例的行为。

## Java 和 Python 有哪些方面相似？

尽管存在差异，但您可能已经发现了 Java 和 Python 之间的一些相似之处。这是因为 Python 和 Java 都受到了 C 编程语言的启发。在继续探索 Python 的过程中，您会发现它与 Java 有更多的相似之处。

### 基于类的面向对象

Python 是一种基于[类的](https://en.wikipedia.org/wiki/Class-based_programming "Class-based programming - Wikipedia")、[面向对象的](https://en.wikipedia.org/wiki/Object-oriented_programming "Object-oriented programming - Wikipedia")编程语言，这也是 Java 的主要特点之一。然而，这两种语言的面向对象特性集是不同的，要足够详细地解决这些问题，需要一个单独的教程。

幸运的是，您可以更深入地研究 Python 和 Java 中的面向对象编程，从而了解 Java 和 Python 在面向对象编程结构方面的区别。您还可以查看 Python 3 中的[面向对象编程概述，以扩展您对该主题的了解。](https://realpython.com/python3-object-oriented-programming/ "Object-Oriented Programming (OOP) in Python 3 – Real Python")

### 操作员

您可能会注意到这两种语言的共同遗产的一个方面是它们如何使用运算符。它们中的许多在两种语言中都有相同的意思。

首先，比较 Java 和 Python 中众所周知的算术运算符。加法运算符(`+`)、减法运算符(`-`)、乘法运算符(`*`)、除法运算符(`/`)和模运算符(`%`)在两种语言中几乎具有相同的用途——除了用于对类似整数的操作数进行除法运算。

这同样适用于[位运算符](https://realpython.com/python-bitwise-operators/ "Bitwise Operators in Python – Real Python"):位 or 运算符(`|`)、位 AND 运算符(`&`)、位 XOR 运算符(`^`)和一元位 NOT 运算符(`~`)，以及用于左移(`<<`)和右移(`>>`)的位移位运算符。

你可以在 Python 中使用方括号语法(`[]`)来访问一个序列的元素，就像你如何使用 Java 的[数组访问](https://docs.oracle.com/javase/specs/jls/se16/html/jls-10.html#jls-10.4 "Chapter 10\. Arrays: Array Access")一样。

后面关于数据类型的部分提供了关于这些操作符和一些附加操作符的更多细节。或者，如果你现在想了解更多，你可以阅读 Python 中的[操作符和表达式。](https://realpython.com/python-operators-expressions/ "Operators and Expressions in Python – Real Python")

### 字符串格式化

最初，Python 提供了字符串格式化功能，这是基于 C 编程语言中的`printf`函数家族如何处理这一功能。这个类似于 Java 的`String.format()`。在 Python 中，`%`操作符执行这个功能。运算符的左侧包含格式字符串，右侧包含位置参数的[元组](https://realpython.com/python-lists-tuples/ "Lists and Tuples in Python – Real Python")或键控参数的[字典](https://realpython.com/python-dicts/ "Dictionaries in Python – Real Python")。

注意:在本教程的后面，你会看到更多关于元组和字典的内容。

以下进程显示了一些示例:

>>>

```py
>>> "Hello, %s!" % "world"             # %-style, single argument
'Hello, world!'
>>> "The %s is %d." % ("answer", 42)   # %-style, positional
'The answer is 42.'
>>> "The %(word)s is %(value)d." \
... % dict(word="answer", value=42)    # %-style, key-based
'The answer is 42.'
```

最近，Python 已经采用了其他的[格式化字符串的方式](https://realpython.com/python-formatted-output/ "A Guide to the Newer Python String Format Techniques – Real Python")。一种是使用`.format()`字符串方法，替换字段用花括号(`{}`)表示。这方面的一个例子是`"The {word} is {value}.".format(word="answer", value=42)`。

从 Python 3.6 开始，还可以使用[格式的字符串文字](https://realpython.com/python-f-strings/ "Python 3's f-Strings: An Improved String Formatting Syntax (Guide) – Real Python")，也称为 **f 字符串**。假设在作用域中有两个名为`word`和`value`的变量。在这种情况下，表达式`f"The {word} is {value}."`为您呈现与上面的示例`.format()`相同的字符串。

[*Remove ads*](/account/join/)

### 控制流构造

比较 Java 和 Python 时，控制流结构是相似的。这意味着您可能直观地认识到许多控制流结构。然而，在更详细的层面上，也存在差异。

一个 Python [`while`](https://realpython.com/python-while-loop/ "Python 'while' Loops (Indefinite Iteration) – Real Python") 循环类似于 Java 的:

```py
while (word := input("Enter word: ")) != "END":
    print(word)

print("READY")
```

代码片段一行一行地将标准输入复制到标准输出，直到该行等于`"END"`。那一行没有被复制，但是文本`"READY"`被写入，后跟一个换行符。

您可能已经注意到了像这样的构造中的 walrus 操作符的附加价值。该赋值*表达式*运算符的[优先级](https://docs.python.org/3/reference/expressions.html#operator-precedence "6\. Expressions — Python 3 documentation: Operator precedence")是所有运算符中最低的。这意味着当赋值表达式是一个更大的表达式的一部分时，你经常需要在它周围加上括号，就像在 Java 中一样。

**注意:** Python 没有`do {...} while (...)`循环结构。

Python [`for`](https://realpython.com/python-for-loop/ "Python 'for' Loops (Definite Iteration) – Real Python") 循环类似于 Java for-each 循环。这意味着，例如，如果您想要迭代前五个罗马数字的列表，您可以使用类似的逻辑对其进行编码:

>>>

```py
>>> roman_numerals = "I II III IV V".split()
>>> roman_numerals
['I', 'II', 'III', 'IV', 'V']
>>> for numeral in roman_numerals:
...     print(numeral)
...
I
II
III
IV
V
```

您可能会注意到，使用`str.split()`是创建单词列表的一种便捷方式。

**注意:**不仅是`list`实例可以这样迭代，任何[可迭代的](https://docs.python.org/3/glossary.html#term-iterable "Glossary — Python 3 documentation: iterable")都可以。

有时，您可能需要一个运行计数器来代替。在这种情况下，您可以使用`range()`:

>>>

```py
>>> for i in range(5):
...     print(i)
...
0
1
2
3
4
```

在本例中，`i`在每次迭代中引用请求范围的下一个值。随后打印该值。

在极少数情况下，您希望迭代一个集合，同时希望有一个运行计数器，您可以使用 [`enumerate()`](https://realpython.com/python-enumerate/) :

>>>

```py
>>> for i, numeral in enumerate("I II III IV V".split(), start=1):
...     print(i, numeral)
...
1 I
2 II
3 III
4 IV
5 V
```

上面的例子显示了前面两个例子在一个循环中的功能。默认情况下，伴随的计数器从零开始，但是使用可选的关键字参数`start`，您可以指定另一个值。

**注:**查看 [Python enumerate():用计数器简化循环](https://realpython.com/python-enumerate/ "Python enumerate(): Simplify Looping With Counters – Real Python")如果你想了解更多关于 Python 循环和`enumerate()`的知识。

Python 也理解 [`break`和`continue`](https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops "4\. More Control Flow Tools — Python 3 documentation: break and continue Statements, and else Clauses on Loops") 语句。

另一个类似于 Java 的控制流结构是 **`if`语句**:

>>>

```py
>>> for n in range(3):
...     if n <= 0:
...         adjective = "not enough"
...     elif n == 1:
...         adjective = "just enough"
...     else:
...         adjective = "more than enough"
...     print(f"You have {adjective} items ({n:d})")
...
You have not enough items (0)
You have just enough items (1)
You have more than enough items (2)
```

如上所述，Python `if ... else`构造还支持一个`elif`关键字，这很有帮助，因为没有简单的`switch ... case`语句。

**注意:**最近发布的 Python 3.10 版本包含了一个名为[结构模式匹配](https://www.python.org/dev/peps/pep-0636/ "PEP 636 -- Structural Pattern Matching: Tutorial | Python.org")的新特性，它引入了`match`和`case`关键字，但其行为与 Java 的 [`switch`语句](https://docs.oracle.com/javase/tutorial/java/nutsandbolts/switch.html "The switch Statement - The Java Tutorials - Learning the Java Language - Language Basics")截然不同。

这个新的语言特性受到 Scala 的[模式匹配](https://docs.scala-lang.org/tour/pattern-matching.html "Pattern Matching | Tour of Scala | Scala Documentation")语句的启发，Scala 是另一种运行在 JVM 上的编程语言。

虽然许多编码结构表面上看起来很相似，但仍有许多不同之处。例如，Python 循环，以及[异常捕获构造](https://realpython.com/python-exceptions/ "Python Exceptions: An Introduction – Real Python")，支持`else:`部分。此外，Python 为上下文管理器提供了一个 [`with`语句](https://realpython.com/python-with-statement/ "Context Managers and Python's with Statement – Real Python")。

[*Remove ads*](/account/join/)

## Java vs Python:什么是高级本地数据类型？

在接下来的小节中，您将看到 Python 标准类型的简要概述。重点是这些类型或它们相关的操作符与 Java 的不同之处，或者它们与相应的 Java 集合类相比如何。

### 数字类型及其运算符

Python 提供了多种数值类型供选择，以适应您的特定应用领域。它内置了三种数值类型:

| 类型 | 类型名 | 示例文字 |
| --- | --- | --- |
| 整数 | `int` | `42` |
| 浮点型 | `float` | `3.14` |
| 复杂的 | `complex` | `1+2j` |

如果您比较这两种语言，那么您会发现 Python 整数可以包含任意长值，只受您的机器可用的(虚拟)内存量的限制。您可以将它们想象成固定精度的本地整数(或者 Java 称之为原始整数类型)和 Java 的 [`BigInteger`](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/math/BigInteger.html "BigInteger (Java SE 11 & JDK 11 )") 数字的智能混合，结果如下:

1.  您拥有任意精度整数的所有**便利，并且可以对它们使用所有众所周知的符号操作符。**
2.  当值足够小时，Python 对提供的值应用**快速固定精度整数运算**。

通过使用前缀`0x`、`0o`和`0b`，您可以将 Python 整数分别指定为十六进制、八进制和二进制常量。

**注意:**这意味着八进制数是*而不是*，只有一个或多个前导零(`0`)，这与 Java 不同。

比较 Java 和 Python 中众所周知的算术运算符`+`、`-`、`*`、`/`和`%`，它们在两种语言中具有相同的含义，除了类似整数类型上的除法运算符。在 Python 中，应用于`int`操作数的 [truediv](https://docs.python.org/3/library/operator.html#operator.truediv "operator — Standard operators as functions — Python 3 documentation: operator.truediv()") 运算符(`/`)产生一个`float`值，这与 Java 不同。Python 使用 [floordiv](https://docs.python.org/3/library/operator.html#operator.floordiv "operator — Standard operators as functions — Python 3 documentation: operator.floordiv()") 运算符(`//`)进行除法运算，向下舍入到最接近的整数，类似于 Java 中的除法运算符(`/`):

>>>

```py
>>> 11 / 4          # truediv
2.75
>>> 11.0 // 4       # floordiv, despite float operand(s)
2.0
```

此外，Python 提供了双星号运算符(`**`)来求幂，作为双参数 [`pow()`](https://docs.python.org/3/library/functions.html#pow "Built-in Functions — Python 3 documentation: pow()") 函数的替代。 [matmul](https://docs.python.org/3/library/operator.html#operator.matmul "operator — Standard operators as functions — Python 3 documentation: operator.matmul()") 操作符(`@`)是为外部包提供的类型保留的附加操作符，旨在为矩阵乘法提供方便的符号。

Python 和 Java 都采用了 C 编程语言中的[位操作符](https://realpython.com/python-bitwise-operators/ "Bitwise Operators in Python – Real Python")。这意味着按位运算符(`|`、`&`、`^`和一元`~`)在两种编程语言中具有相同的含义。

如果您想对负值使用这些操作符，那么最好知道在 Python 中，整数被表示为概念上无限大的空间中的[二进制补码](https://en.wikipedia.org/wiki/Two%27s_complement "Two's complement - Wikipedia")值来保存位。这意味着负值在概念上有无限多的前导`1`位，就像正数在概念上有无限多的前导`0`位一样:

>>>

```py
>>> bin(~0)
'-0b1'
>>> bin(~0 & 0b1111)       # ~0 is an "infinite" sequence of ones
'0b1111'
```

上面的代码片段表明，不管您选择什么值，如果您将这个值与常数`~0`进行位与运算，那么结果值等于所选择的值。这意味着常数`~0`在概念上是一个无限的`1`比特序列。

也可以使用**位移**运算符(`<<`和`>>`)。然而，Java 的按位[零填充右移](https://docs.oracle.com/javase/tutorial/java/nutsandbolts/op3.html "Bitwise and Bit Shift Operators (The Java™ Tutorials > Learning the Java Language > Language Basics)")操作符(`>>>`)没有对等物，因为这在任意长整数的数字系统中没有意义。没有最重要的**位。这两种语言的另一个区别是 Python 不允许用负移位数来移位。**

Python 标准库也提供了其他数值类型。十进制定点和浮点运算有 [`decimal.Decimal`](https://docs.python.org/3/library/decimal.html "decimal — Decimal fixed point and floating point arithmetic — Python 3 documentation") ，堪比 Java 的 [`BigDecimal`](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/math/BigDecimal.html "BigDecimal (Java SE 11 & JDK 11 )") 。有理数有一个 [`fractions.Fraction`](https://docs.python.org/3/library/fractions.html "fractions — Rational numbers — Python 3 documentation") 类，类似于 [Apache Commons 数学分数](http://commons.apache.org/proper/commons-math/userguide/fraction.html "Math – The Commons Math User Guide - Fractions")。请注意，这些类型是*而不是*分类为内置数值类型。

### 基本序列类型

序列类型是容器，您可以在其中使用整数索引来访问它们的元素。[字符串](#strings)和[字节序列](#bytes)也是序列类型。这些将在几节后介绍。Python 内置了三种基本序列类型:

| 类型 | 类型名 | 示例文字 |
| --- | --- | --- |
| 目录 | `list` | `[41, 42, 43]` |
| 元组 | `tuple` | `("foo", 42, 3.14)` |
| 范围 | `range` | `range(0, 9, 2)` |

如您所见，列表和元组初始化器之间的语法差异是方括号(`[]`)和圆括号(`()`)。

一个 Python **list** 类似 Java 的 [`ArrayList`](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/ArrayList.html "ArrayList (Java SE 11 & JDK 11 )") ，是 [mutable](https://docs.python.org/3/glossary.html#term-mutable "Glossary — Python 3 documentation: mutable") 。您通常将这种容器用于同构集合，就像在 Java 中一样。然而，在 Python 中，存储不相关类型的对象是可能的。

另一方面，**元组**更类似于 Java 中类似 [`Pair`](https://www.javatuples.org/ "javatuples - Main") 类的[不可变](https://docs.python.org/3/glossary.html#term-immutable "Glossary — Python 3 documentation: immutable")版本，除了它用于任意数量的条目而不是两个。空括号(`()`)表示空元组。像`(3,)`这样的结构表示包含单个元素的元组。在这种情况下，单个元素是`3`。

**注意:**你注意到只有一个元素的`tuple`的特殊语法了吗？请参见下面的进程，了解不同之处:

>>>

```py
>>> (3,)
(3,)
>>> type(_)
<class 'tuple'>
>>> (3)
3
>>> type(_)
<class 'int'>
```

如果没有添加尾随逗号，那么表达式将被解释为括号内的对象。

一个**范围**产生一个通常在循环中使用的数字序列。在本教程的前面，您已经看到了一个这样的例子。如果你想学习更多关于 Python 中范围的知识，你可以查看[这个关于 range()函数](https://realpython.com/python-range/ "The Python range() Function (Guide) – Real Python")的指南。

要从序列中选择一个**元素**，可以在方括号中指定从零开始的索引，如`some_sequence[some_index]`所示。负索引从末尾向后计数，因此`-1`表示最后一个元素。

您也可以从序列中选择一个**切片**。这是对零个、一个或多个元素的选择，产生与原始序列同类的对象。您可以指定一个`start`、`stop`和`step`值，也称为步幅。切片语法的一个例子是`some_sequence[<start>:<stop>:<step>]`。

所有这些值都是可选的，如果没有另外指定，则采用实际的默认值。如果您想了解更多关于列表、元组、索引和切片的信息，阅读 Python 中的[列表和元组会很有帮助。](https://realpython.com/python-lists-tuples/ "Lists and Tuples in Python – Real Python")

对于正索引，Python 语法类似于在 Java 数组中选择元素的方式。

您可以使用加号运算符(`+`)连接大多数序列，并使用星号运算符(`*`)重复它们:

>>>

```py
>>> ["testing"] + ["one", "two"] * 2
['testing', 'one', 'two', 'one', 'two']
```

您不能用范围完成串联或序列重复，但是您可以对它们进行切片。例如，尝试`range(6, 36, 3)[7:2:-2]`并考虑你得到的结果。

[*Remove ads*](/account/join/)

### 字典

Python 中的一个字典， [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "Built-in Types — Python 3 documentation: class dict()") ，类似于 Java 的 [`LinkedHashMap`](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/LinkedHashMap.html "LinkedHashMap (Java SE 11 & JDK 11 )") 。dict 初始化器常量的语法是花括号(`{}`)之间逗号分隔的`key: value`条目序列。这方面的一个例子是`{"pi": 3.14, "e": 2.71}`。

要从 dict 或任何其他[映射](https://docs.python.org/3/glossary.html#term-mapping "Glossary — Python 3 documentation: mapping")中选择一个元素，可以在方括号(`[]`)中指定键，如`math_symbols["pi"]`。键和值都可以是任何对象，但是键需要是[可散列的](https://docs.python.org/3/glossary.html#term-hashable "Glossary — Python 3 documentation: hashable")，这意味着它们通常是[不可变的](https://docs.python.org/3/glossary.html#term-immutable "Glossary — Python 3 documentation: immutable")——或者至少应该表现得像不可变的对象。键不一定需要都是相同的类型，尽管它们通常是相同的。这同样适用于价值观。

要了解更多信息，您可以阅读更多关于 Python 中的[字典，或者查看关于](https://realpython.com/python-dicts/ "Dictionaries in Python – Real Python")[映射类型](https://docs.python.org/3/library/stdtypes.html#typesmapping "Built-in Types — Python 3 documentation: Mapping Types --- dict")的 Python 文档。

### 设置

Python 还提供了[集](https://docs.python.org/3/library/stdtypes.html#set "Built-in Types — Python 3 documentation: set")。您可以使用类似于`{"pi", "e"}`的语法或者使用`set()`构造函数语法初始化集合，使用 iterable 作为参数。要创建一个空集，可以使用表达式`set()`，因为字面量`{}`已经被赋予了字典。

集合在底层实现中使用哈希。当你迭代一个集合时，要考虑到条目将会以明显随机的顺序出现，就像在 Java 中一样。此外，不同 Python 调用之间的顺序甚至可能会改变。

对于集合运算，某些运算符已被重载。详情可以阅读 Python 中关于[集合的更多内容。](https://realpython.com/python-sets/ "Sets in Python – Real Python")

### 字符串

就像在 Java 中一样，Python 中的字符串是 Unicode 元素的不可变序列。字符串文字在双引号(`"`)之间指定，也可以在单引号(`'`)之间指定，这与 Java 不同。

字符串的两个例子是`"foo"`和`'bar'`。定义字符串时，不同的引号不一定有不同的含义。但是，你应该注意，如果你使用引号作为字符串的一部分，如果它们碰巧也是字符串的分隔符，你就必须对它们进行转义。

和 Java 一样，Python 中的反斜杠(`\`)是引入一个[转义序列](https://en.wikipedia.org/wiki/Escape_sequence "Escape sequence - Wikipedia")的字符。Python 解释器识别转义序列，这些序列在 Java 中也是已知的，比如`\b`、`\n`、`\t`，以及 C 编程语言中的一些其他序列。

默认情况下，Python 假设 Python 源文件采用 UTF-8 编码。这意味着您可以将 Unicode 文字直接放在字符串中，就像在`"é"`的情况中一样。您还可以使用其码位的 16 位或 32 位十六进制表示形式。对于`"é"`，你可以通过使用`\u00E9`或`\U000000E9`转义序列来实现。注意小写`\u`和大写`\U`转义序列的区别。最后，也可以提供它的 Unicode 描述，比如`\N{Latin Small Letter E with acute}`。

即使在 Python 标识符中也可以使用 Unicode 字符，但问题是这样做是否明智。

如果你在字符串前面加上了`r`，就像在`r"raw\text"`中一样，那么反斜杠就失去了它的特殊意义。当您想要指定[正则表达式](https://realpython.com/regex-python/ "Regular Expressions: Regexes in Python (Part 1) – Real Python")时，这尤其方便。

您还可以用三重引号将字符串括起来，以方便地创建多行字符串，如下例所示:

>>>

```py
>>> s = """This is a
...        multiline
...        string.
...        """
...
>>> for line in s.splitlines():
...     print(repr(line))
...
'This is a'
'       multiline'
'       string.'
'       '
```

您可以将这种类型的字符串与 Java [文本块(JEP 378)](https://openjdk.java.net/jeps/378 "JEP 378: Text Blocks") 进行比较，尽管有其他语法限制和另一种空白保留(制表符、空格和换行符)。

### 字节

在 Java 中，如果你需要存储**二进制数据**而不是**文本**，你可能会使用 [`ByteBuffer`](https://docs.oracle.com/en/java/javase/16/docs/api/java.base/java/nio/ByteBuffer.html "ByteBuffer (Java SE 16 & JDK 16)") ，这给了你可变的对象。在 Python 中， [`bytearray`](https://docs.python.org/3/library/stdtypes.html#bytearray-objects "Built-in Types — Python 3 documentation: Bytearray Objects") 对象提供了类似的功能。

与 Java 不同，Python 还提供了一个 [`bytes`](https://realpython.com/python-strings/#bytes-objects "Strings and Character Data in Python – Real Python: bytes Objects") 类型来存储不可变的二进制数据。字节字面量看起来非常类似于字符串字面量，除了您在字面量前面加了一个`b`。字符串包含一个将它们转换成字节序列的`.encode()`方法，一个`bytes`对象包含一个将它转换成字符串的`.decode()`方法:

>>>

```py
>>> bytes(4)                     # Zero-filled with a specified length
b'\x00\x00\x00\x00'
>>> bytes(range(4))              # From an iterable of integers
b'\x00\x01\x02\x03'
>>> b = "Attaché case".encode()  # Many different codecs are available
>>> b
b'Attach\xc3\xa9 case'
>>> b.decode()                   # Decode back into a string again
'Attaché case'
```

如果未指定编解码器，则默认的 UTF-8 编解码器用于编码字符串和解码字节。当你需要的时候，你可以从提供各种文本和字节转换的编解码器的一个大的[列表中选择。](https://docs.python.org/3/library/codecs.html#standard-encodings "codecs — Codec registry and base classes — Python 3 documentation: Standard Encodings")

Python `bytes`对象也有一个`.hex()`方法，它产生一个字符串，以十六进制形式列出内容。对于相反的操作，您可以使用`.fromhex()` [类方法](https://realpython.com/instance-class-and-static-methods-demystified/#class-methods "Python's Instance, Class, and Static Methods Demystified – Real Python: Class Methods")从十六进制字符串表示中构造一个`bytes`对象。

[*Remove ads*](/account/join/)

### 布尔型

`False`和`True`是 Python 中的两个`bool`实例对象。在[数字上下文](https://docs.python.org/3/library/stdtypes.html#typesnumeric "Built-in Types — Python 3 documentation: Numeric Types")中，`True`计算为`1`，而`False`计算为`0`。这意味着`True + True`评估为`2`。

Python 中的布尔逻辑运算符不同于 Java 的`&&`、`||`和`!`运算符。在 Python 中，这些是保留的关键字`and`、`or`和`not`。

下表总结了这一点:

| Java 语言(一种计算机语言，尤用于创建网站) | 计算机编程语言 | 描述 |
| --- | --- | --- |
| `a && b` | `a and b` | 逻辑与 |
| `a &#124;&#124; b` | `a or b` | 逻辑或 |
| `!a` | `not a` | 逻辑非 |

与 Java 类似，布尔运算符`and`和`or`有一个短路求值行为，Python 解释器从左到右缓慢地对操作数求值，直到它可以确定整个表达式的真值。

与 Java 的另一个相似之处是解释器产生最后一次求值的子表达式作为结果。因此，你应该知道一个`and`或`or`表达式的结果不一定产生一个`bool`实例对象。

所有 Python 对象要么有一个**假值**要么有**真值**值。换句话说，当您将 Python 对象转换为`bool`时，结果是明确定义的:

*   **等于`0`** 的数值转换为`False`，否则转换为`True`。
*   **空容器**、**集合**、**字符串**和**字节对象**转换为`False`，否则转换为`True`。
*   **中的`None`对象**也会转换成`False`。
*   **所有其他对象**评估为`True`。

**注意:**用户定义的类可以提供一个`.__bool__()` dunder 方法来定义它们的类实例的真实性。

如果您想测试一个[容器](https://docs.python.org/3/library/collections.abc.html#collections.abc.Container "collections.abc — Abstract Base Classes for Containers — Python 3 documentation: class collections.abc.Container")或字符串是否非空，那么您只需在一个布尔上下文中提供该对象。这被认为是一种[蟒](https://docs.python.org/3/glossary.html#term-pythonic "Glossary — Python 3 documentation: Pythonic")的做法。

查看以下检查非空字符串的不同方法:

>>>

```py
>>> s = "some string"
>>> if s != "":                 # Comparing two strings
...     print('s != ""')
...
s != ""

>>> if len(s) != 0:             # Asking for the string length
...     print("len(s) != 0")
...
len(s) != 0

>>> if len(s):                  # Close, but no cigar
...     print("len(s)")
...
len(s)

>>> if s:                       # Pythonic code!
...     print("s")
...
s
```

在最后一个示例中，您只需在布尔上下文中提供字符串。如果字符串不为空，则计算结果为 true。

**注意:**以上并不意味着所有类型都依赖隐式`bool`转换。关于 [`None`](#none) 的下一节将对此进行更详细的阐述。

如果您想了解更多关于最典型的 Python 结构的知识，可以遵循[编写更多 Python 代码](https://realpython.com/learning-paths/writing-pythonic-code/ "Write More Pythonic Code (Learning Path) – Real Python")的学习路径。

在 Python 中，你用 Java 编写的带有[条件运算符(`? :` )](https://docs.oracle.com/javase/specs/jls/se16/html/jls-6.html#jls-6.3.1.4 "Chapter 6\. Names: Conditional operator ? :") 的[条件表达式](https://realpython.com/python-conditional-statements/#conditional-expressions-pythons-ternary-operator "Conditional Statements in Python – Real Python: Conditional Expression (Python's Ternary Operator)")，作为带有关键词`if`和`else`的表达式:

| Java 语言(一种计算机语言，尤用于创建网站) | 计算机编程语言 |
| --- | --- |
| `cond ? a : b` | `a if cond else b` |

考虑 Python 中这种类型表达式的一个示例:

>>>

```py
>>> for n in range(3):
...     word = "item" if n == 1 else "items"
...     print(f"Amount: {n:d}  {word}")
...
Amount: 0 items
Amount: 1 item
Amount: 2 items
```

只有当`n`等于`1`时，REPL 才输出`"item"`。在所有其他情况下，REPL 输出`"items"`。

[*Remove ads*](/account/join/)

### 无

在 Python 中，`None`是一个单例对象，可以用来标识[类似空值](https://realpython.com/null-in-python/ "Null in Python: Understanding Python's NoneType Object – Real Python")。在 Java 中，出于类似的目的，你可以使用文字 [`null`](https://docs.oracle.com/javase/specs/jls/se16/html/jls-3.html#jls-3.10.8 "Chapter 3\. Lexical Structure: 3.10.8\. The Null Literal") 。

Python 中最常用的`None`是作为函数或方法定义中的默认参数值。此外，不返回任何值的函数或方法实际上会隐式返回`None`对象。

一般来说，当你在一个布尔上下文中依赖于`None`的隐式转换时，它被认为是一种[代码味道](https://en.wikipedia.org/wiki/Code_smell "Code smell - Wikipedia")，因为你可能会为其他类型的对象编写无意的行为代码，而这些对象恰好返回一个 falsy 值。

因此，如果您想测试一个对象是否真的是`None`对象，那么您应该显式地这样做。因为只有一个`None`对象，所以可以通过使用对象标识操作符`is`或相反的操作符`is not`来实现:

>>>

```py
>>> some_value = "All" or None

>>> if some_value is None:
...     print(f"is None: {some_value}")

>>> if some_value is not None:
...     print(f"is not None: {some_value}")
...
is not None: All
```

请记住，这里的单词`not`是`is not`操作符不可分割的一部分，而且它明显不同于逻辑`not`操作符。在这个例子中，字符串`"All"`在布尔上下文中有一个真值。您可能还记得，`or`操作符有这种短路行为，只要结果已知，就返回最后一个表达式，在本例中是`"All"`。

### 更多容器数据类型

Java 通过它的[集合框架](https://docs.oracle.com/en/java/javase/16/docs/api/java.base/java/util/doc-files/coll-overview.html "Collections Framework Overview (Java SE 16 & JDK 16)")提供它的标准容器类型。

Python 采用了不同的方法。它以内置类型的形式提供了您在本节前面已经探索过的基本容器类型，然后 Python 的标准库通过 [`collections`模块](https://realpython.com/python-collections-module/ "Python's collections: A Buffet of Specialized Data Types – Real Python")提供了更多的容器数据类型。您可以通过`collections`模块访问许多有用的容器类型示例:

*   [`namedtuple`](https://realpython.com/python-namedtuple/ "Write Pythonic and Clean Code With namedtuple – Real Python") 提供了元组，您还可以通过字段名访问元组中的元素。
*   [`deque`](https://docs.python.org/3/library/collections.html#collections.deque "collections — Container datatypes — Python 3 documentation: class collections.deque") 提供双端队列，在集合两端都有快速追加和移除。
*   [`ChainMap`](https://docs.python.org/3/library/collections.html#collections.ChainMap "collections — Container datatypes — Python 3 documentation: class collections.ChainMap") 让你将多个贴图对象折叠成一个单独的贴图视图。
*   [`Counter`](https://realpython.com/python-counter/ "Python's Counter: The Pythonic Way to Count Objects – Real Python") 为计数可散列对象提供了映射。
*   [`defaultdict`](https://realpython.com/python-defaultdict/ "Using the Python defaultdict Type for Handling Missing Keys – Real Python") 提供了调用工厂函数来提供缺失值的映射。

这些数据容器类型已经用普通 Python 实现了。

至此，您已经有了理解 Java 和 Python 在特性、语法和数据类型上的异同的良好基础。现在是时候后退一步，探索可用的 Python 库和框架，并找出它们对特定用例的适用性。

## 具体用法有哪些资源？

您可以在许多领域使用 Python。下面您将找到其中的一些领域，以及它们最有用和最受欢迎的相关 Python 库或框架:

*   **命令行脚本:** [`argparse`](https://docs.python.org/3/library/argparse.html "argparse — Parser for command-line options, arguments and sub-commands — Python 3 documentation") 提供创建命令行参数解析器的功能。
*   **网络框架:**
    *   在开发完整且可能复杂的网站时，Django 提供了一种更简单的方法。它包括定义模型的能力，提供自己的 ORM 解决方案，并提供完整的管理功能。您可以添加额外的插件来进一步扩展管理。
    *   Flask 宣称自己是一个专注于做好一件事的微框架，那就是为 web 请求服务。您可以将这个核心功能与您自己选择的其他已经存在的组件结合起来，比如 ORM 和表单验证。许多被称为 Flask 插件的扩展可以很好地为您集成这些组件。
    *   [请求](https://requests.readthedocs.io/en/master/ "Requests: HTTP for Humans™ — Requests 2.x.x documentation")使得发送 HTTP 请求变得极其方便。
*   **数据建模与分析:** [pandas](https://pandas.pydata.org/ "pandas - Python Data Analysis Library") 基于 [NumPy](https://numpy.org/ "NumPy") ，是一款快速、强大、灵活、直观的开源数据分析与操作工具。有些人把熊猫称为“类固醇上的可编程电子表格”
*   **机器学习:** [TensorFlow](https://www.tensorflow.org/ "TensorFlow") 、 [Keras](https://keras.io/ "Keras: the Python deep learning API") 和 [PyTorch](https://pytorch.org/ "PyTorch") 是机器学习领域的几个流行框架。
*   **SQL 工具包和对象关系映射器(ORM):** [SQLAlchemy](https://www.sqlalchemy.org/ "SQLAlchemy - The Database Toolkit for Python") 是一个非常流行的 Python SQL 工具包和 ORM 框架。
*   **工作量分配:** [芹菜](https://docs.celeryproject.org/en/stable/ "Celery - Distributed Task Queue — Celery 4.x.x documentation")是一个分布式任务队列系统。

Python 还有一些与**质量保证**相关的值得注意的工具:

*   [pytest](https://docs.pytest.org/en/stable/ "pytest: helps you write better programs — pytest documentation") 是标准 [`unittest`](https://docs.python.org/3/library/unittest.html "unittest — Unit testing framework — Python 3 documentation") 库的一个很好的替代品。
*   [behavior](https://behave.readthedocs.io/en/latest/index.html "Welcome to behave! — behave 1.x.x documentation")是一个流行的[行为驱动开发(BDD)](https://en.wikipedia.org/wiki/Behavior-driven_development "Behavior-driven development – Wikipedia") 工具。您可以将它与 [PyHamcrest](https://github.com/hamcrest/PyHamcrest "hamcrest/PyHamcrest: Hamcrest matchers for Python") 结合使用，以获得更具表现力的断言检查。
*   [Flake8](https://flake8.pycqa.org/en/latest/ "Flake8: Your Tool For Style Guide Enforcement — flake8 3.x.x documentation") 是一个编码风格向导检查器。
*   [Pylint](https://www.pylint.org/ "Pylint - code analysis for Python | www.pylint.org") 是一个工具，可以检查 Python 代码中的错误，识别代码气味和编码标准偏差。
*   黑色的是不妥协的、固执己见的、难以配置的代码重组者。虽然这听起来很可怕，但在现实生活中，对于任何大型软件项目来说，这都是一个很好的工具。
*   mypy 是使用最广泛的静态类型检查器。
*   [Bandit](https://bandit.readthedocs.io/en/latest/ "Welcome to Bandit's developer documentation! — Bandit documentation") 发现常见安全问题。
*   [安全](https://github.com/pyupio/safety "pyupio/safety: Safety checks your installed dependencies for known security vulnerabilities")检查您已安装的依赖项是否存在已知的安全漏洞。
*   [tox](https://tox.readthedocs.io/en/latest/ "Welcome to the tox automation project — tox 3.x.x documentation") 是一个命令行工具，有助于在一个命令中运行为您的项目以及多个 Python 版本和依赖配置定义的自动化测试和 QA 工具检查。

上面的列表只是众多可用包和框架中的一小部分。您可以浏览和搜索 Python 包索引(PyPI)来找到您正在寻找的特殊包。

## Python 什么时候会比 Java 更有用，为什么？

通常，您希望为一个用例选择一种编程语言，而为另一个用例选择不同的编程语言。在比较 Java 和 Python 时，您应该考虑以下几个方面:

*   Java 和 Python 都成功地用于世界上最大的 web 应用程序中。
*   您还可以使用 Python 编写 shell 工具。
*   Python 优雅的语法、代码可读性、丰富的库和大量的外部包允许快速开发。您可能只需要不到一半的代码行就可以实现与 Java 相同的功能。
*   因为标准 Python 不需要编译或链接步骤，所以当您更新代码时会立即看到结果。这进一步加快了开发周期。
*   在大多数情况下，对于一般的应用程序，标准 Java 的执行速度要高于 Python。
*   你可以用 C 或 C++毫不费力地扩展 Python。这在一定程度上缓解了执行速度的差异。

对于某些用途，如数据建模、分析、机器学习和人工智能，执行速度真的很重要。为此功能创建的流行的第三方包是用编译成本机代码的编程语言定义的。对于这些领域，Python 似乎是最符合逻辑的选择。

## 结论

在本教程中，您熟悉了 Python，并对这种编程语言的特性有了清晰的了解。您已经探索了 Java 和 Python 之间的相似之处和不同之处。

现在，您已经有了一些快速入门 Python 的经验。您也有了一个很好的基础，可以了解在哪些情况下以及在哪些问题领域应用 Python 是有用的，并且对接下来可以查看哪些有用的资源有了一个大致的了解。

**在本教程中，您学习了:**

*   Python 编程语言的**语法**
*   Python 中有相当多的标准数据类型
*   Python 与 Java 有何不同
*   Java 和 Python 在哪些方面**相似**
*   在那里你可以找到 Python **文档**和**特定主题教程**
*   如何开始使用 Python
*   如何通过使用 Python REPL 查看**即时结果**
*   一些**喜欢的框架**和**库**

也许你确定将来会更多地使用 Python，或者也许你还在决定是否要更深入地研究这门语言。无论哪种方式，有了上面总结的信息，您已经准备好探索 Python 和本教程中介绍的一些框架。

## 额外资源

当您准备好学习更多关于 Python 及其包的知识时，web 上有大量的可用资源:

*   你可以找到[书面教程](https://realpython.com/tutorials/all/ "All Python Tutorial Topics – Real Python")、[视频课程](https://realpython.com/courses/ "Python Video Courses – Real Python")、[测验](https://realpython.com/quizzes/ "Python Quizzes – Real Python")和[学习路径](https://realpython.com/learning-paths/ "Python Learning Paths – Real Python")，它们涵盖了 [*真实 Python*](https://realpython.com/ "Real Python") 的许多主题。
*   外部 Python 实用程序、库和框架通常会提供良好的文档。
*   PyVideo.org 提供了一个庞大的索引集合，可以免费获取来自世界各地 Python 相关会议的演示。
*   最后但同样重要的是，官方的 [Python 文档](https://docs.python.org/3/ "Python 3 Documentation")维护了关于 Python 编程语言、其标准库及其生态系统的高标准的准确和完整的信息。

既然您已经了解了 Python 实际上是一种什么样的编程语言，那么您很有希望对它产生热情，并考虑在您的下一个项目中使用它，无论项目是大是小。快乐的蟒蛇！**********