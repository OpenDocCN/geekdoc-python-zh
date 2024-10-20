# 本周 PyDev:法比奥·扎德罗兹尼

> 原文：<https://www.blog.pythonlibrary.org/2015/12/28/pydev-of-the-week-fabio-zadrozny/>

本周我们欢迎[法比奥·扎德罗兹尼](http://pydev.blogspot.com/)([@法比奥夫兹](https://twitter.com/fabiofz))成为我们本周的 PyDev。他是 [PyDev](http://www.pydev.org/) 项目的幕后推手，这是一个用于 Eclipse 的 Python IDE。法比奥还参与了许多其他项目，如果你查看他的 github 简介，你会发现这一点。让我们花些时间更好地了解他。

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

当然可以...我是一名软件工程师，拥有计算机科学学士学位(2002 年毕业)。我住在巴西的弗洛里亚纳波利斯。除了编程，我真的很喜欢打乒乓球(我十几岁的时候几乎就开始打职业乒乓球了，现在这是一个很好的爱好)。尽管如此，我现在除了工作之外的主要职业是抚养一个 6 个月大和一个 6 岁大的女儿。

**你为什么开始使用 Python？**

*我在 2003 年开始使用 Python，在 http://www.esss.com.br ESSS()从事科学计算项目——当时，我对 Python 的主要不满是没有一个合适的 IDE，所以， 这些年来，我在 PyDev([http://pydev.org](http://pydev.org))上工作了很长时间，以获得一个我喜欢的工作环境——有点讽刺的是，为了获得一个很棒的 Python 工作环境，我在 Java 上工作了很长时间——尽管我也参与了许多 Python 项目，并且肯定在这里吃了自己的狗粮！*

你还知道哪些编程语言，你最喜欢哪一种？

我研究的主要语言是 Python 和 Java——所以，在 Eclipse 中开发像 PyDev 这样的 Python IDE 对我来说很自然😉

除此之外，我已经用 C/C++完成了我的一部分工作，最近我开始用 Javascript 做一些项目(我也尝试过一些其他语言，因为我真的很喜欢尝试新的语言，但这些是我目前使用最多的语言)。

Python 成为了我最喜欢的语言，因为它非常有表现力，并且使程序非常简洁。除了 Python 之外，我要说 Java 获得了第二名，尽管它的优点与 Python 非常不同，并且 ide 等工具发挥了更大的作用，因此，您必须掌握 IDE 才能令人愉快(没有 IDE 编写 Java 代码肯定不会令人愉快，而在 Python 中这是可行的-尽管 IMHO，PyDev 等 IDE 即使在 Python 中也有很大帮助，并且当项目变大时是必不可少的)。

你现在在做什么项目？

*嗯，我总是有一大堆令人兴奋的事情要做——目前我正在做:*

*   ***PyDev**([http://pydev.org](http://pydev.org))目前是 Python 的顶级 ide 之一——我每天都在使用它，我已经为它工作了 12 年，通过社区贡献/众筹活动让它继续发展。此外，我对 PyDev 现在的状态非常满意——有很多用户，近年来，随着市场变得更加拥挤，有了更多的选择，我只找到了真正喜欢使用 PyDev/Eclipse 的人(LiClipse 使用户更容易启动和运行)。*
*   ***PyDev。调试器**([https://github.com/fabioz/PyDev.Debugger/](https://github.com/fabioz/PyDev.Debugger/))是最初只在 PyDev 中的调试器的一个分离，以便它可以在其他 ui 中使用(目前它由 PyDev 和 PyCharm 使用)；*
*   ***LiClipse**([http://liclipse.com](http://liclipse.com))，为 PyDev 提供了一个单机，在 Eclipse 中增加了对多种语言的轻量级支持，使其整体上成为一个好得多的包；*
*   ***PyVmMonitor**([http://www.pyvmmonitor.com](http://www.pyvmmonitor.com))，Python 可用的最好的剖析和 vm 监控体验；*
*   ***speed tin**:[https://www.speedtin.com](https://www.speedtin.com)，这是我开始参与的最新项目——它的目标是在性能退化上线之前更容易发现它——或者就 CPython 2.7 而言，希望一些核心提交者能够看看诸如[https://www . speed tin . com/reports/1 _ CPython 27 x _ Performance _ Over _ Time](https://www.speedtin.com/reports/1_CPython27x_Performance_Over_Time/)之类的东西，并为以前版本上损失的性能提供一些修复😉*
*   ***mu-repo**([https://fabioz.github.io/mu-repo/](https://fabioz.github.io/mu-repo/)):这是用 Python 制作的一个实用程序，用于处理多个 git 库。*
*   ***洛基**([http://rocky-dem.com](http://rocky-dem.com))，这是一个顶尖的粒子模拟器(模拟器实际上是用 C/C++完成的，在 GPU 中用 CUDA 工作，或者在 CPU 中用 OpenMP 工作，但我主要研究它的后处理能力，这些都是用 Python 完成的)。*

如你所见，我从不感到无聊😉

哪些 Python 库是你最喜欢的(核心或第三方)？

如果我必须挑选一个最喜欢的库，我可能会选择 greenlets，这是异步编程的神奇解决方案(我个人认为用它编程比用 asyncio 类的解决方案或基于回调的编程容易得多)。

尽管如此，我真的很喜欢 Python 对 yield/generators 的支持，并且，在核心库中，我认为 collections 和 itertools 非常好，提供了组织和迭代数据结构的简洁方法。

此外，我总是惊讶于 numpy 如何以一种高性能的方式处理大型数组，而 py.test 使测试变得非常简单。

作为一门编程语言，你认为 Python 将何去何从？

*嗯，希望 Python 能克服来自 Python 2/3 的破损，重新拥有同页的社区。*

我一直对此持否定态度，但最近我认为有更好的迹象表明会选择 Python 3，尽管像 Pyston 这样的东西仍然让我有点担心——也就是说:这是一个全新的 Python 实现，目标是 Python 2——不幸的是，我认为它真正表明了社区仍然是多么的破碎(因为，你知道，移植大型代码库肯定不是有趣的，昂贵的，并且有添加可能难以发现的 bug 的额外风险)。

时间会证明一切，但我真的希望有一天我不用在程序中添加 cruft 来支持 CPython 2 和 3😉

非常感谢！

### 一周的最后 10 个 PyDevs

*   马赫什·文基塔查拉姆
*   弗洛里安·布鲁欣
*   马克-安德烈·莱姆堡
*   尼克·科格兰
*   阿伦·拉文德兰
*   [什么场](https://www.blog.pythonlibrary.org/2015/11/16/pydev-of-the-week-amit-saha/)
*   布莱恩·施拉德
*   克雷格·布鲁斯
*   海梅·费尔南德斯·德尔罗
*   [瑞安·米切尔](https://www.blog.pythonlibrary.org/2015/10/19/pydev-of-the-week-ryan-mitchell/)