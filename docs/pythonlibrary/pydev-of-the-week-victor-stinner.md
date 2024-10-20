# 本周 PyDev:维克多·斯坦纳

> 原文：<https://www.blog.pythonlibrary.org/2017/02/27/pydev-of-the-week-victor-stinner/>

本周我们欢迎维克多·斯坦纳成为我们的本周 PyDev！Victor 在 Python 社区非常活跃，是一名核心 Python 开发人员。你可以在这里看到他的一些贡献。他是八个公认的 pep 的作者，你也可以在前面的链接中读到。如果你有兴趣看看 Victor 还做了什么，那么我强烈推荐你去看看 [Github](https://github.com/haypo) 和 [ReadTheDocs](http://haypo-notes.readthedocs.io/) 。Victor 还为 CPython 和 FASTCALL 优化收集了一些有趣的基准测试。您可能还想在这里查看他关于 Python 基准的最新演讲:[https://fosdem . org/2017/schedule/event/Python _ stable _ benchmark/](https://fosdem.org/2017/schedule/event/python_stable_benchmark/)

现在，让我们花些时间来更好地了解 Victor！

![](img/50298313266c421d08ab84f079751038.png)

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

*大家好，我叫 Victor Stinner，我在 OpenStack 上为 Red Hat 工作，从 2010 年开始就是一名 CPython 核心开发人员。*

我是一名来自法国贝尔福-蒙彼利亚德大学(UTBM)工程学院的工程师。当我不黑 CPython 的时候，我就和我的两个可爱的小女儿一起玩🙂

你还知道哪些编程语言，你最喜欢哪一种？
 *我一直在编程。我尝试了各种编程语言，从最低级别的 Intel x86 汇编程序到高级语言，如 Javascript 和 BASIC。即使我现在真的喜欢编写 C 代码以获得最佳性能，Python 更适合我的日常工作需求。由于编写 Python 代码很容易，而且我也不讨厌内存管理或分析崩溃，所以我利用“空闲”时间编写更多的单元测试，关注编码风格
，以及所有使软件成为“好软件”的微小事情。*

经过 10 年的专业编程，我现在可以说，我花在阅读“旧”代码和修复旧的复杂问题上的时间比从头开始编写新代码的时间还多。拥有一个可扩展的测试套件让我更酷。不得不在没有束缚的压力下工作很可能会导致精疲力竭，或者更简单地说，辞职。

你现在在做什么项目？
 *在红帽，我有一个把 OpenStack 移植到 Python 3 的大项目。OpenStack 由 300 多万条 Python 代码组成，并且每天都在增长！超过 90%的单元测试已经在 Python 3 上通过，我们现在正致力于解决功能和
集成测试的最后问题。*

在 CPython 上，我在 Python 3 的童年花了很多时间修复 Unicode。如今，我正在从事多个项目，以使 CPython 更快。非常好的消息是，在大多数基准测试中，CPython 3.6 现在比 2.7 快，CPython 3.7 已经比 CPython 3.6 快了！总之 Python 3 终于比 Python 2 快了！

去年，我花了很多时间进行“FASTCALL”优化，避免创建临时元组来传递位置参数，创建临时字典来传递关键字参数。我现在 3/4 以上的 FASTCALL 工作都合并到 CPython 里了。当一个函数被转换为 FASTCALL 时，它通常会快 20%,而且转换非常简单。

在进行 FASTCALL 和其他优化时，我被不可靠的基准测试阻塞了。你可以看到我刚刚在 FOSDEM(比利时布鲁塞尔)上发表的“如何运行稳定的基准测试”演讲，其中列出了我的所有发现，并解释了如何获得可重复且可靠的结果:[https://FOSDEM . org/2017/schedule/event/python _ stable _ benchmark/](https://fosdem.org/2017/schedule/event/python_stable_benchmark/)

另请参见我为使基准测试更加可靠而创建的 [perf 项目](https://pypi.python.org/pypi/perf)。这是一个用两行代码编写基准的 Python 模块。该模块提供了许多工具来检查一个基准是否可靠，比较两个基准和检查一个优化是否重要，等等。

哪些 Python 库是你最喜欢的(核心或第三方)？

在 Python 标准库中，我喜欢 asyncio、argparse 和 datetime 模块。

日期时间做一件事，而且做得很好。它最近得到了增强，以支持夏令时(DST):https://www.python.org/dev/peps/pep-0495/

argparse 模块非常完整，它允许构建高级命令行界面。我在我的 perf 模块中使用它来获得子命令，如“python3 -m perf timeit stmt”、“python 3-m perf show-metadata file . JSON”，...

asyncio 很好地集成了一些很酷的东西:网络服务器的高效事件循环和 Python 3 新的 async/await 关键字。不仅 asyncio 有很好的 API(不再有回调地狱！)，但它也有很好的实现。例如，很少有事件循环库支持子进程，尤其是在 Windows IOCP 上(在 Windows 上进行异步编程的最有效方式)。

作为一名核心开发人员，我最关心的是标准库的模块，但事实上最好的库都在 PyPI 上！仅举几个例子:pip，jinja2，django 等。抱歉，列表太长，这里放不下🙂

作为一门编程语言，你认为 Python 将何去何从？

我希望 Python 停止进化，我说的是语言本身。在花了几年时间缓慢过渡到 Python 3 的过程中，我意识到有多少像 Python 2.7 这样的用户停止了进化。与快速移动的库甚至编程语言相比，不必接触他们的代码被视为一种优势。

由于打包现在可以用 pip 流畅地运行，拥有外部依赖变得很容易。外部代码的优势在于它可以比 Python 标准库移动得更快，Python 标准库基本上每两年更新一次主要的 Python 版本。

即使我不喜欢进化，我也不得不承认这种语言最近增加的功能真的很酷:通用解包、async/await 关键字、f-string、允许数字中有下划线等等。

你还有什么想说的吗？

*我听 Twitter 的时候，Go，Rust，Javascript，Elm 等。似乎比其他任何语言都要活跃得多。*

同时，我总是对每个 Python 版本中所做的工作印象深刻。甚至 Python 语言也在不断发展。脸书决定只使用 Python 3.5 来获得新的 async 和 await 关键字和 asyncio！Python 3.6 增加了更多东西:f-string (PEP 498)、变量注释语法(PEP 526)和数字文字中的下划线(PEP 515)。

顺便说一下，许多人抱怨类型暗示。有些人认为它们是“非 pythonic 式的”。其他人担心 Python 会变成一种乏味的类似 Java 的语言。我也知道类型提示已经在像 Dropbox 和脸书这样的大公司的 Python 中使用了，它们对于非常大的代码库非常有帮助。

Python 很酷的一点是它不强制任何东西。例如，您可以不使用对象来设计整个应用程序。你也可以完全忽略类型提示，它们是完全可选的。那可是巨蟒的一个实力！

非常感谢你接受采访！