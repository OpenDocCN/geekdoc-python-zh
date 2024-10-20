# 本周 PyDev:马克-安德烈·莱姆堡

> 原文：<https://www.blog.pythonlibrary.org/2015/12/07/pydev-of-the-week-marc-andre-lemburg/>

本周，我们欢迎马克-安德烈·莱姆堡([@马林堡](https://twitter.com/malemburg))成为我们本周的 PyDev。Lemburg 先生是 Python 核心开发人员和企业家。他拥有自己的企业 eGenix，并在他的[博客](http://www.malemburg.com/)上写关于 Python 的文章。他是 Python 软件基金会的创始成员之一，目前是该组织的董事会成员。他还是欧洲蟒蛇协会的董事会成员。让我们花些时间去更好地了解他！

[![ma_lemburg](img/15a883a3bdd9b968d531207528b5b3d2.png)](https://www.blog.pythonlibrary.org/wp-content/uploads/2015/10/ma_lemburg.jpg)

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我和妻子及两个孩子住在德国的 DÃ杜塞尔多夫，拥有一家名为[eGenix.com](http://www.eGenix.com)的咨询和项目公司，喜欢创意、艺术、摄影、舞蹈、音乐和任何形式的灵感。我原本想学艺术，但后来决定转而学我的另一个爱好，物理。然而，在找到理论物理模型来处理三阶甚至二阶近似后，我转而重新关注数学，后来专攻最优化和逻辑。我的梦想是有一天重返艺术，但那一天还没有到来。

**你为什么开始使用 Python？**

我第一次了解 Python 是在 1994 年，当时我在 OS/2 免费软件 CD 上找到了一个发行版(Hobbes)。一天晚上，我读了 Guido 的教程，立刻就被这个想法打动了，那就是这将成为我未来几年最喜欢的编程语言。

几年后，我成为了 Python 核心开发人员，设计了最初的 Unicode 集成，为 Python 贡献了一些模块，并编写了许多 Python C 扩展和包，有些是开源的，有些是商业的或为客户开发的。

你还知道哪些编程语言，你最喜欢哪一种？

当我第一次学习 Python 时，我是一名 C 程序员，经历了 BASIC、汇编、Pascal(按此顺序)。我试过 C++，但从来没有真正喜欢过。Java 也一样。我还学了 Lisp，一些 Smalltalk，后来又学了 Javascript，开始做网站的时候。

我最喜欢的两种编程语言仍然是 Python 和 C。我用 Python 写所有的高级代码，为了速度，我转向 C。

你现在在做什么项目？

*现在，我正在考虑将 [mx 扩展](http://www.egenix.com/products/python)移植到 Python 3。鉴于代码库的规模，这是一个相当具有挑战性的项目，也是我为什么没有早点开始这个探索的主要原因。最近开始移植 mxDateTime 后，我发现有些事情对于移植代码以在 Python 2 和 3 上运行的人来说肯定会变得更容易，同时保持向后兼容。*

与此同时，我正在研究一些其他项目，例如实现 Windows 版本的[eGenix pyron](http://www.egenix.com/products/python/PyRun/)，并找到一种比序列化数据更有效地在运行 Python 的进程或线程之间共享数据的方法。

后一个项目是解决全局解释器锁问题的一种方法，这种锁阻止了充分利用当今的处理器来处理 CPU 受限的工作负载。

除了这些项目，我还继续从事其他几个与 Python 相关的技术和社区项目:[http://www.malemburg.com/summary](http://www.malemburg.com/summary)

哪些 Python 库是你最喜欢的(核心或第三方)？

我非常喜欢标准的 lib difflib 模块。它并不知名，但实现了一些相当惊人的算法来查找文本数据中的差异。

Python 的列表排序和字典实现也非常值得一读。两者都有很多值得学习和欣赏的地方——你不会经常看到如此精心制作的代码。

作为一门编程语言，你认为 Python 将何去何从？

我看到 Python 变得越来越受欢迎。考虑到它的年龄和不断增长的受欢迎程度，我认为 Python 已经达到了主流编程语言的地位。它与 C/C++、Java 和 Javascript 并列。今天，在与客户交谈时，您不必再解释 Python 是什么，这是一个好现象。

随着 Python 进入教育系统，这种趋势将会更加明显。

Python 应用程序并行执行的问题是我们必须投入一些认真思考来保持趋势的一个领域。Python 3 中的异步支持是 I/O 绑定工作负载的解决方案，但是对于 CPU 绑定工作负载，我们还没有一个好的解决方案。我看到的另一个问题是 Python 标准库的老化。我相信这将受益于一个完整的重写，使其更加一致和易于使用。

你对当前的 Python 程序员市场有什么看法？

鉴于 Python 在几个重要行业的应用，这是一个不断增长的市场:网络、大数据、科学(除此之外，参见[http://brochure.getpython.info/](http://brochure.getpython.info/))。

这是 Python 程序员的大好时光。

你还有什么想说的吗？

我希望我们能找到一种方法，在不破坏所有 Python C 扩展的情况下，消除或解决 Python 4 中的 GIL 限制。

Python 2 到 3 的过渡是一个痛苦的经历，并导致许多 Python 用户开始认真寻找替代品。我认为没有人会期待另一次这样的转变。

非常感谢！

### 一周的最后 10 个 PyDevs

*   尼克·科格兰
*   阿伦·拉文德兰
*   [什么场](https://www.blog.pythonlibrary.org/2015/11/16/pydev-of-the-week-amit-saha/)
*   布莱恩·施拉德
*   克雷格·布鲁斯
*   海梅·费尔南德斯·德尔罗
*   [瑞安·米切尔](https://www.blog.pythonlibrary.org/2015/10/19/pydev-of-the-week-ryan-mitchell/)
*   [卡罗尔心甘情愿](https://www.blog.pythonlibrary.org/2015/10/12/pydev-of-the-week-carol-willing/)
*   迈克尔·福格曼
*   [特雷西·奥斯本](https://www.blog.pythonlibrary.org/2015/09/28/pydev-of-the-week-tracy-osborn/)