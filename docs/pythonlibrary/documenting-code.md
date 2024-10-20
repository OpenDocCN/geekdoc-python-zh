# Python 101 -记录您的代码

> 原文：<https://www.blog.pythonlibrary.org/2021/09/12/documenting-code/>

在早期记录代码比大多数新开发人员意识到的要重要得多。软件开发中的文档是指给你的变量、函数和其他标识符起一个描述性的名字。也指添加好的评论。当你沉浸于开发你的最新作品时，用非描述性的名字创建变量和函数是很容易的。一个月或一年后，当你不可避免地回到你的代码时，你将花费大量的时间试图弄清楚你的代码做什么。

通过使您的代码自文档化(即，使用描述性名称)并在必要时添加注释，您将使您的代码对于您自己和任何可能使用您代码的人来说更具可读性。这也将使更新代码和重构代码变得更加容易！

在本章中，您将了解以下主题:

*   评论
*   文档字符串
*   pep 8-Python 风格指南
*   用于记录代码的其他工具

让我们从了解评论开始。

## 什么是评论？

注释是为您编写的代码，不是为您的计算机编写的。我的意思是，注释基本上是给你自己的一个注解，解释在你的代码部分发生了什么。你使用注释来解释你为什么做某事或者一段代码是如何工作的。当你开始作为一个新的开发人员时，最好给自己留下大量的评论以供参考。但是一旦你学会了如何正确命名你的函数和变量，你会发现你不再需要注释了。

但是仍然建议使用注释，尤其是对于复杂的、乍一看不容易理解的代码。根据你工作的公司，你也可以使用注释来记录错误修复。例如，如果您正在修复一个 bug，您可以在注释中提到您正在修复的 bug，以帮助解释您为什么必须更改它。

您可以使用`#`符号后跟一些描述性文本来创建注释。

这里有一个例子:

```py
# This is a bad comment
x = 10
```

在上面的代码中，第一行演示了如何创建一个简单的注释。当 Python 执行这段代码时，它会看到`#`符号，并忽略它后面的所有文本。实际上，Python 将跳过这一行，尝试执行第二行。

此评论被标记为“差评”。虽然它有利于演示，但它根本没有描述它后面的代码。这就是为什么它不是一个好的评论。好的注释描述了后面的代码。一个好的注释可以描述 Python 脚本的目的、代码行或其他内容。注释是代码的文档。如果他们不提供信息，那么他们应该被删除。

您还可以创建内嵌注释:

```py
x = 10  # 10 is being assigned to x
```

这里您再次将 10 赋给变量`x`，但是您添加了两个空格和一个`#`符号，这允许您添加关于代码的注释。当您可能需要解释特定的代码行时，这很有用。如果你给你的变量起了一个描述性的名字，那么你很可能根本不需要注释。

## 注释掉

你会经常听到“注释掉代码”这个术语。这是将`#`符号添加到代码开头的做法。这将有效地禁用您的代码。

例如，您可能有这样一行代码:

```py
number_of_people = 10
```

如果您想将其注释掉，可以执行以下操作:

```py
# number_of_people = 10
```

当您尝试不同的解决方案或调试代码时，您可以注释掉代码，但您不想删除代码。Python 将忽略被注释掉的代码，允许您尝试其他东西。大多数 Python 代码编辑器(和文本编辑器)提供了一种方法来突出显示多行代码，并注释掉或取消注释掉整个代码块。

## 多行注释

一些编程语言，比如 C++，提供了创建多行注释的能力。Python 风格指南(PEP8)说英镑符号是首选。但是，您可以使用带三重引号的字符串作为多行注释。

这里有一个例子:

```py
>>> '''This is a 
multiline comment'''
>>> """This is also a 
multiline comment"""

```

当您创建三重引号字符串时，您可能会创建一个**文档字符串**。

让我们看看什么是 docstrings 以及如何使用它们！

## 了解文档字符串

Python 有 PEP 或 Python 增强提案的概念。这些 pep 是 Python 指导委员会讨论并同意的 Python 语言的建议或新特性。

PEP 257 描述了文档字符串约定。如果你想知道完整的故事，你可以去看看。可以说，docstring 是一个字符串文字，应该作为模块、函数、类或方法定义中的第一条语句出现。你现在不需要理解所有这些术语。事实上，在本书的后面你会学到更多。

docstring 是使用三重双引号创建的。

这里有一个例子:

```py
"""
This is a docstring
with multiple lines
"""

```

Python 会忽略文档字符串。他们不能被处决。但是，当您使用 docstring 作为模块、函数等的第一条语句时，docstring 将成为一个特殊的属性，可以通过`__doc__`访问。在关于类的章节中，你会学到更多关于属性和文档字符串的知识。

文档字符串可以用于单行或多行字符串。

下面是一个一行程序的示例:

```py
"""This is a one-liner"""

```

单行 docstring 就是只有一行文本的 docstring。

以下是函数中使用的文档字符串的示例:

```py
def my_function():
    """This is the function's docstring"""
    pass

```

上面的代码展示了如何向函数中添加 docstring。你可以在第 14 章学到更多关于函数的知识。一个好的 docstring 描述了函数应该完成什么。

**注意**:虽然三个双引号是推荐标准，但是三个单引号、单个双引号和单个单引号都可以(但是单个双引号和单个单引号只能包含一行，不能包含多行)。

现在让我们根据 Python 的风格指南来学习编码。

## Python 的风格指南:PEP8

风格指南是描述好的编程实践的文档，通常是关于单一语言的。一些公司有特定的公司风格指南，开发人员无论使用什么编程语言都必须遵循。

早在 2001 年，Python 风格指南被创建为 [PEP8](https://www.python.org/dev/peps/pep-0008/) 。它记录了 Python 编程语言的编码约定，这些年来已经更新了几次。

如果你打算经常使用 Python，你真的应该看看这个指南。它会帮助你写出更好的 Python 代码。

此外，如果你想为 Python 语言本身做出贡献，你的所有代码必须符合风格指南，否则你的代码将被拒绝。

遵循风格指南将使你的代码更容易阅读和理解。这将有助于您和将来使用您代码的任何人。

不过，记住所有的规则可能很难。幸运的是，一些勇敢的开发人员已经创建了一些实用程序来提供帮助！

## 有帮助的工具

有很多优秀的工具可以帮助你写出优秀的代码。以下是几个例子:

*   pycodestyle-[https://pypi.org/project/pycodestyle/](https://pypi.org/project/pycodestyle/)-检查你的代码是否遵循 PEP8
*   皮林特-[https://www.pylint.org/](https://www.pylint.org/)-一个深入的静态代码测试工具，可以发现代码中的常见问题
*   py flakes-[https://pypi.org/project/pyflakes/](https://pypi.org/project/pyflakes/)-Python 的另一个静态代码测试工具
*   flake 8-[https://pypi.org/project/flake8/](https://pypi.org/project/flake8/)-一个包裹着 PyFlakes、pycodestyle 和 McCabe 脚本的包装
*   布莱克-[https://black.readthedocs.io/en/stable/](https://black.readthedocs.io/en/stable/)-一个主要遵循 PEP8 的代码格式化程序

您可以针对您的代码运行这些工具，以帮助您找到代码中的问题。我发现 Pylint 和 PyFlakes / flake8 是最有用的。如果您在团队中工作，并且希望每个人的代码都遵循相同的格式，黑色会很有帮助。可以将 Black 添加到您的工具链中，为您格式化代码。

更高级的 Python IDEs 提供了 Pylint 等的一些检查。实时提供。例如，PyCharm 会自动检查这些工具会发现的许多问题。WingIDE 和 VS 代码也提供了一些静态代码检查。您应该查看各种 ide，看看哪一个最适合您。

## 包扎

Python 提供了几种不同的方法来记录代码。你可以使用**注释**来解释一行或多行代码。这些应该在适当的时候适度使用。您还可以使用 **docstrings** 来记录您的模块、函数、方法和类。

您还应该查看一下 PEP8 中的 Python 风格指南。这将帮助您开发良好的 Python 编码实践。Python 还有其他几个风格指南。例如，你可能想要查找 Google 的风格指南或者 NumPy 的 Python 风格指南。有时看看不同的风格指南也会帮助你发展良好的实践。

最后，您了解了几个可以用来帮助您改进代码的工具。如果你有时间，我鼓励你去看看 PyFlakes 或 Flake8，特别是因为它们在指出你的代码中常见的编码问题时非常有帮助。

## 相关阅读

想了解更多关于 Python 的功能吗？查看这些教程:

*   matplotlib–[用 Python 创建图表的介绍](https://www.blog.pythonlibrary.org/2021/09/07/matplotlib-an-intro-to-creating-graphs-with-python/)

*   # 

    Python 101: [使用 JSON 的介绍](https://www.blog.pythonlibrary.org/2020/09/15/python-101-an-intro-to-working-with-json/)

*   Python 101 - [创建多个流程](https://www.blog.pythonlibrary.org/2020/07/15/python-101-creating-multiple-processes/)
*   python 101-[用 pdb 调试你的代码](https://www.blog.pythonlibrary.org/2020/07/07/python-101-debugging-your-code-with-pdb/)

*   Python 101—[使用 Python 启动子流程](https://www.blog.pythonlibrary.org/2020/06/30/python-101-launching-subprocesses-with-python/)