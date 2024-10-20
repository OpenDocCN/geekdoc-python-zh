# 本周 PyDev:Pablo ga lindo Salgado

> 原文：<https://www.blog.pythonlibrary.org/2020/04/06/pydev-of-the-week-pablo-galindo-salgado/>

本周我们欢迎 Pablo ga lindo Salgado([@ pyblogsal](https://twitter.com/pyblogsal))成为我们本周的 PyDev！Pablo 是 Python 编程语言的核心开发人员。他也是几个 Python 相关会议的发言人。如果你想知道他参与了哪些项目，你可以看看他的 [Github 简介](https://github.com/pablogsal)。

让我们花些时间更好地了解 Pablo！

**能否简单介绍一下自己(爱好、学历等):** 我目前在 Bloomberg L.P .的 Python 基础架构团队工作，支持我们所有的 Python 开发者，提供关键的基础架构和库，确保每个人都有更好的 Python 编程体验。但是在从事软件行业之前，我曾经是学术界的一名理论物理学家，研究广义相对论，特别是黑洞物理学。这是我仍然作为爱好做的事情(尽管没有出版和资金的压力),因为我仍然热爱它！例如，我在一些 Python 会议上发表过与这款
([https://www.youtube.com/<wbr>手表相关的演讲？v=p0Fc2jWVbrk](https://www.youtube.com/watch?v=p0Fc2jWVbrk) )和我继续开发和研究改进的算法来模拟和可视化不同的时空。例如，这里有一些模拟的克尔纽曼黑洞，它们周围有我最近研究过的吸积盘:

![Accretion Disks](img/d168643bd7e66d9a1ab694db91c0cd73.png)
这里用一些纹理映射到天空球体和吸积盘:
![Texture mapped accretion disks](img/4a5e3e2bc75ebdf133874527ca30abb8.png)

当我不在 CPython 中燃烧 CPU 内核进行内核开发工作时，我会燃烧它们进行越来越多的模拟。我喜欢继续从事这项工作，因为它结合了我的两个爱好:理论物理和编码！有时候，为了优化你的代码，你需要一个更好的等式，而不是一个更好的算法！

**你为什么开始使用 Python？** 当我开始攻读博士学位时，我开始使用 Python 来编排和粘贴一些 C 语言的模拟代码(有时是 FORTRAN 77！) )并做数据处理和分析。我立刻爱上了这门语言，后来又爱上了这个社区和这个生态系统(正如一句名言“为语言而来，为社区而留”)。我以前接触过许多其他语言，但我开始越来越多地使用 Python，因为它有一些迷人的东西，使编程变得非常非常有趣。此外，在科学界(不包括数据科学和机器学习的庞大世界)，它允许快速建立想法原型，并以其他人可以轻松直观地使用和扩展的方式集成现有系统
。此外，与许多其他科学工具和套件相反:是免费和开源的，我认为这是使科学更加透明、可用和可复制的基础！

你还知道哪些编程语言，哪种是你最喜欢的？

除了 Python，我精通 C 和 Fortran(真的！)而且我有信心用 Rust 和 C++编码。除此之外，我可以自豪地说，我以某种形式的 Javascript 编写了几次代码，而没有让事情爆炸，我可以复制一些预先存在的 HTML 和 CSS，并修改它以制作一些很酷的前端。我用 Wolfram 语言(Mathematica)写了很多年的代码，但这是我不再做的事情，尽管有时我会想念它的一些功能模式。

即使现在不是很常用，并且有相当数量的逻辑批评，我仍然喜欢 c。这可能是斯德哥尔摩综合症，但我发现它非常简单而且(通常不是)优雅。我还发现，当我需要推理一些低级效果或者我需要“更接近金属”时，我有很好的抽象水平。同样，当我开始用 C 语言编码时，我有相当多的 FORTRAN(也叫 FORTRAN77)经验，让我告诉你一些事情:第一次发现你可以不用每行的前五列为空来编码，这是一次改变一生的经历。另一个改变人生的经历是，当你在很久以后才发现，这种胡说八道是为了与打孔卡兼容。
 **你现在在做什么项目？** 我所有的开源工作主要是在 CPython 上，但是除了我作为核心开发人员的一般工作之外，从去年开始，我还与吉多·范·罗苏姆、莱桑德罗斯·尼古拉、艾米丽·莫尔豪斯托和其他人一起在一个项目中做了很多工作，以替换 CPython 中当前的解析器，获得一个新的闪亮的 PEG 解析器！

我对解析器和形式语言非常感兴趣(我已经做了几次关于它们的演讲，以及 CPython 中的解析器是如何作为[https://www.youtube.com/watch?<wbr>v = 1 _ 23 avsiqec&t = 2018s](https://www.youtube.com/watch?v=1_23AVsiQEc&t=2018s)工作的),所以这个项目非常令人兴奋，因为有了它，我们不仅能够消除当前 LL(1)语法中的几个丑陋的漏洞，而且它将允许编写一些现在不可能的结构。特别是，我一直在努力尝试在上下文管理器中允许分组括号，比如

```py
with (
    something_very_long_here as A,
   something_else_ very_long_here as B,
   more_long_lines as C,
):
   ...
```

但遗憾的是，这在当前的解析机制下是不可能的。相信我:书上的每一招我都试过了。此外，我们希望有了新的解析器，现有语法的许多解析器将以更易维护和可读性更好的方式编写。敬请关注了解更多！

**您还从事 Python core 的其他哪些工作？**

作为 Python 核心开发人员，我主要从事 Python 运行时和 VM 方面的工作，尤其是在解析器、AST、编译器和垃圾收集器方面。但是除了这些重点领域，我还在标准库中的所有地方工作过:posix、多重处理、functools、数学、内建....我还花了大量的时间来消除错误和竞争条件，并照顾 CPython CI 和 buildbot 系统(请查看 PSF 博客中关于它的这篇博文[http://py found . blogspot .<wbr>com/2019/06/pablo-ga lindo-<wbr>salgado-nights-watch-is.html](http://pyfound.blogspot.com/2019/06/pablo-galindo-salgado-nights-watch-is.html))。

和其他一些核心开发人员交谈，最近我发现自从我被提升以来，我是 3 个“最活跃”的核心开发人员之一([https://github.com/python/<wbr>cpython/graphs/contributors？<wbr>from = 2018-05-07&to = 2020-03-25&type = c](https://github.com/python/cpython/graphs/contributors?from=2018-05-07&to=2020-03-25&type=c))！

我还致力于让贡献者和未来的核心开发人员更容易接触到 CPython。例如，我最近为 CPython 最没有文档记录的领域之一([https://devguide.python.org/<wbr>垃圾收集器/](https://devguide.python.org/garbage_collector/) )写了一份非常完整的设计文档。我也花了很多时间指导贡献者，希望他们中的一些成为未来的核心开发者！指导要求很高，但我认为这是确保 Python 保持活力和我们有一个开放和包容的社区的一个非常重要的部分。

我也非常感谢在核心开发团队中有一些最不可思议、最有才华和最坦率的人在我身边，因为他们是我每天做出贡献的主要原因之一！

哪些 Python 库是你最喜欢的(核心或第三方)？

这是个很难的问题！我将滥用这一事实，该问题并没有限制图书馆列出几个:

来自标准库:gc，ast，posix ( \_(？)_/)、functools、tracemalloc 和 itertools。第三方:Cython 和 numpy。

我知道你是 PEP 570 背后团队的一员。你是怎么卷进来的？

我完成了完整的实现，主持了讨论，并用不太令人印象深刻的英语编写了 PEP 文档的大部分内容，但多亏了我的其他合作者和 Carol Willing(她是我在几件事情上的榜样，包括以简单的方式记录和解释复杂的事情),自我编写第一个版本以来，该文档有了很大的改进。

你有最喜欢的晦涩的 Python 特性或模块吗？

我喜欢连锁比较！例如，当你写下:

```py
if a < c < d:
   ...
```

而不是:

```py
if a < c and d < d:
   ...
```

我还喜欢你用不太直观的方式使用它们，比如:

```py
>>> x = "'Hola'"
>>> x[-1] == x[0] in {"'", "other", "string"}
True
```

这个特性有一个稍微阴暗的一面，可能会让很多人感到困惑，尤其是在与其他操作符一起编写时:

```py
>>> False == False in [False]
True
```

遗憾的是，使用链式比较会稍微慢一些🙁

你还有什么想说的吗？

非常感谢邀请我做这次采访，也感谢每一个坚持到采访结束的人🙂

Pablo，谢谢你接受采访！