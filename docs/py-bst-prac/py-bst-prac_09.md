# 文档

可读性是 Python 开发者需关注的重点，这包括项目和代码文档的可读性。遵循一些 简单的最佳实践能够帮助你和其他人节省很多时间。

## 项目文档

根目录下的 `README` 文件应该告诉用户和维护者关于项目的基本信息。它应该是 原始文本，用一些非常容易阅读的标记，比如 reStructuredText 或 Markdown。 它应该包含几行内容用来解释项目或者库的目的（假设用户不用知道任何关于该项目的事）， 软件主要源的 URL，以及一些基本的信用信息。此文件是代码阅读者的主要入口。

`INSTALL` 文件对 Python 来说并不必要。安装指令通常少至一条命令，比如说 `pip install module` 或 `python setup.py install` ，这被添加到 `README` 文件中。

`LICENSE` 文件应该 *总是* 存在的，而且应该指定向公众开放的软件许可。

`TODO` 文件或者位于 `README` 中的 `TODO` 区域应该列出代码的开发计划。

`CHANGELOG` 文件或者位于 `README` 中的 `CHANGELOG` 区域应该呈现对 代码库的最新修改的简短概述。

## 项目发布

根据项目，你的文档中可能包含下列部分或所有的内容：

*   一份 *介绍* 应该用一两个极其简化的用例，来非常简短地概述该产品 能够用来做什么。这是你的项目的 30 秒的自我陈述（thirty-second pitch）。
*   一份 *教程* 应该展示一些主要的用例，它要有更多细节。读者将会跟着一步步搭建 工作原型。
*   一份 *API 参考* 通常从代码（参见 docstrings）中产生。 它会列出所有的可供公共访问的接口、参数和返回值。
*   *开发人员文档* 适用于潜在贡献者。这可以包括项目的代码惯例和通用设计策略。

 ### Sphinx

[Sphinx](http://sphinx.pocoo.org) [http://sphinx.pocoo.org] 无疑是最流行的 Python 文档工具。**请使用它吧。** 它能把 :ref:`restructuredtext-ref`标记语言转换为广泛的输出格式，包括 HTML、LaTeX(可打印 PDF 版本)、手册页和纯文本。

[Read The Docs](http://readthedocs.org) [http://readthedocs.org] 是一个 *超棒的* 并且 *免费的* 文档托管，可以托管你的 [Sphinx](http://sphinx.pocoo.org) [http://sphinx.pocoo.org] 文档。 请使用它。你可以为它配置提交钩子到你的源库中，这样文档的重新构建将会自动进行。  ### reStructuredText

大多数 Python 文档是用 [reStructuredText](http://docutils.sourceforge.net/rst.html) [http://docutils.sourceforge.net/rst.html] 编写的。它就像是内建了所有可选扩展的 Markdown。

[reStructuredText Primer](http://sphinx.pocoo.org/rest.html) [http://sphinx.pocoo.org/rest.html] 和 [reStructuredText Quick Reference](http://docutils.sourceforge.net/docs/user/rst/quickref.html) [http://docutils.sourceforge.net/docs/user/rst/quickref.html] 应该会帮助 你熟悉它的语法。 

## 代码文档建议

注释能使代码清晰，将其加入到代码中是为了理解代码起来更容易。在 Python 中，注意以一个 hash（数字符号）（“#”）开始。

在 Python 中， *docstrings* 用来描述模块、类和函数：

```py
def square_and_rooter(x):
    """Returns the square root of self times self."""
    ... 
```

一般来说，要遵循 [**PEP 8#comments**](https://www.python.org/dev/peps/pep-0008#comments) [https://www.python.org/dev/peps/pep-0008#comments] （”Python 风格指南”）的注释部分。

### 注释代码块

*不要使用三引号去注释代码*。 这不是好的实践，因为面向行的命令行工具， 比如说 grep，不会知道注释过的代码是没有激活的。对每一个注释行，使用带有 合适缩进的井号会更好。你的编辑器可能很容易做到这一点，并能切换注释/取消注释。

### 文档字符串和魔法

一些工具使用文档字符串来嵌入超过文档（more-than-documentation）的行为， 比如说单元测试逻辑。这些可以很好，但你也不能排除“这就是这样做的”。 （译者注：原句是 Those can be nice, but you won’t ever go wrong with vanilla “here’s what this does.” 对此句的翻译是意译的，我也可能理解 不到位~）

### 文档字符串 vs 块注释

这些不可互换。对于函数或类，开头的注释区是程序员的注解。而文档字符串 描述了函数或类的 *操作*:

```py
# 因为某种原因这个函数减慢程序执行。
def square_and_rooter(x):
    """返回自己乘以自己的平方根。"""
    ... 
```

## 其他工具

你可能在其他场景看到过这些。使用 Sphinx。

[Pycco](http://fitzgen.github.com/pycco) [http://fitzgen.github.com/pycco]

Pycco 是一个 “文学编程风格的文档生成器”，它是 node.js [Docco](http://jashkenas.github.com/docco) [http://jashkenas.github.com/docco] 的 移植版本。它将代码生成为一个并排的 HTML 代码和文档。

[Ronn](https://github.com/rtomayko/ronn) [https://github.com/rtomayko/ronn]

Ronn 用来构建 Unix 手册。它将人可读的文本文件转换成用于终端显示的 roff 文件, 以及用于 web 的 HTML 文件。

[Epydoc](http://epydoc.sourceforge.net) [http://epydoc.sourceforge.net]

Epydoc 已经中断开发。使用 Sphinx 来替代。

[MkDocs](http://www.mkdocs.org/) [http://www.mkdocs.org/]

MkDocs 是一个快速简单的静态网站生成器，它适合于构建使用 Markdown 的项目文档。

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.