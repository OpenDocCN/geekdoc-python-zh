# 本周 PyDev:杰森·r·库姆斯

> 原文：<https://www.blog.pythonlibrary.org/2020/08/17/pydev-of-the-week-jason-r-coombs/>

本周我们欢迎 Jason Coombs ( [@jaraco](https://twitter.com/jaraco) )成为我们本周的 PyDev！Jason 是 [twine](https://pypi.org/project/twine/) ，SetupTools，CherryPy 和 140+其他 Python 包的维护者。你可以通过查看 [Github](https://github.com/jaraco) 来了解他目前在做什么。

让我们花一些时间来更好地了解 Jason！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我热爱计算，因为它赋予了人类超凡的力量。我学会了在公告栏系统上社交，并从软件开发提供的持续改进中获得乐趣。我的编程技能是在 Borland C++、K&R C 以及我在 Scheme 的第一门正式编程课程中锻炼出来的。我在终极(光盘)中找到了娱乐，我从索科罗到 DC 到匹兹堡都玩过。

![Jason Coombs](img/6e5843de32e54b6d45023d50217d24c6.png)

**你为什么开始使用 Python？**

1997 年，在我的本科学习中，我和一个 3 人小组一起，为 Python 编写了一个编译器。在 Lex、Yacc 和 C++的基础上，我们学习了 Python 编程语言，并努力克服了对空格敏感的语法的困难，但最终对利用正式上下文的良好格式惯例的价值有了强烈的认识。对 Python 的介绍来得正是时候，当时我仍然依赖 Maple 或 Mathematica 之类的语言进行高级构造，这些构造允许在没有内存管理和电子特性的持续干扰的情况下对问题空间进行建模。我被迷住了。

你还知道哪些编程语言，你最喜欢哪一种？

我涉猎了几十种语言，但是用 Scheme、C/C++、Java、Maple、Matlab、Perl、cmd.exe、Bash、Powershell、XML/XSLT 和 Javascript 开发了不平凡的解决方案。我从事过 C++、Java 和 Python 方面的专业开发，但在 2008 年，我对 Python 如此着迷，以至于我开始专攻 Python，2010 年，我找了一份全职工作，在 Python 上开发全球规模的 web 应用程序。

你现在在做什么项目？

我在 PyPI 中维护了 140 多个包，其中有一部分被大量采用:Setuptools、keyring、CherryPy、IRC、path、twine。我正在积极地改进 Python 上“打包”的故事。特别是，我正在帮助将 Setuptools 的一些复杂特性分解到单独的包中，例如 importlib.metadata 和 import lib . resources(Python stdlib ),并将 distutils 与 Setuptools 统一起来。我更热衷于为常见的计算问题提供越来越复杂的抽象，例如使用 ITER tools(more _ ITER tools/jara co . ITER tools)和 functools (jaraco.functools)，但在实践中，我渴望消除最阻碍社区发展的任何障碍——让每个程序员能够最容易和可持续地解决他们的问题。

哪些 Python 库是你最喜欢的(核心或第三方)？

当我第一次接触 Ruby on Rails 时，我对它如何轻易地使编程领域适应 HTTP 协议印象深刻。我在 Python 生态系统中寻找了一个类似的努力，这使我找到了 Turbogears 和代理 CherryPy。与 Flask 和 Django 等其他框架不同，CherryPy 采用了一种通过 HTTP 域来反映编程域的方法，通过 HTTP 将对象及其属性反映为实体。CherryPy 没有将斜杠分隔的字符串映射到函数，而是采用了更类似于 Zope 的方法，将分层组织的实体暴露到可通过 HTTP 路径遍历的树中。使用语言本身的方面直接建模问题空间的简单优雅是一种真正的乐趣，尽管我真的很欣赏像 Django 和 Flask 以及现在的 FastAPI 这样的库提供的价值，但我渴望分享 CherryPy 方法的好处和乐趣。

我也是像 itertools、more-itertools、functools 这样的库的狂热爱好者，或者任何使开发人员能够避免简单分支(if/else)而使用类似于函数式编程的抽象的库。

**您是如何开始 twine 项目的？**

我很早就作为 setuptools 的用户参与了打包工作，当时 0.6c7(或其左右)的文件发现在 Subversion 的最新版本中工作得不太好。后来，当我的项目需要 Python 3 支持时(依赖于 Setuptools)，我参与了 Distribute 项目。后来，当我着手将 Distribute fork 重新合并到 Setuptools 中时，我采用了 Setuptools，我继续支持打包项目，并帮助它们以一种不那么单一、更具可组合性的方式满足 Setuptools 遗留用户的用例。

作为 Setuptools 和 **setup.py upload** 函数的维护者，我为 twine 项目提供了关于支持上传功能的意见。另外，作为密匙环库的维护者，我对启用像 twine 这样的工具感兴趣，这些工具利用密匙环来允许用户安全地存储他们的密码和令牌，而不是明文。

我对 twine 的支持主要是简化维护过程，这是我在匹兹堡 Python 小组上介绍过的一个主题。

为什么要使用 twine 而不是其他发布工具？

Twine 是 PyPA 认可的官方工具，不仅可以上传到 warehouse(官方 PyPI 索引)，还可以上传测试索引和第三方索引。它是围绕精炼和受支持的标准而设计的，就像一个健康的开源项目一样，欢迎任何善意的贡献，并与维护者合作以避免意想不到的后果。

你还有什么想说的吗？

我真的很想看到 PyPy 成为 Python 的参考实现。它有助于解决我所认为的 Python 的三大挑战之一，即函数调用的性能(另外两个是健壮的多核支持和简单的打包)。对 Python 最大的批评之一是它的性能不如其他语言，如 C++或 Go。实际上，这通常没问题。我在这个问题上只受过很少的教育，所以这可能是我的催款——Kruger 说的，但是我想象一个世界，PyPy 和它的 JIT 编译器可以把 Python 带入一个在整体性能上与 Java 竞争的世界。

杰森，谢谢你接受采访！