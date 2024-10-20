# 本周 PyDev:Julien Danjou

> 原文：<https://www.blog.pythonlibrary.org/2015/05/11/pydev-of-the-week-julien-danjou/>

本周，我们欢迎 Julien Danjou ( [@juldanjou](https://twitter.com/juldanjou) )成为我们本周的 PyDev。Julien 是《Python 黑客指南》的作者。他还写有趣的 Python 博客。让我们花些时间更好地了解朱利安！

**能简单介绍一下自己吗(爱好、学历等):**

当然可以！我是一个 31 岁的自由软件黑客。我在 1998 年左右开始从事 Linux 系统和开源项目，因为我发现它们非常酷，在技术上也很有趣。2001 年，我开始在大学学习，并于 2006 年获得了计算机工程硕士学位。

从那以后，我一直为不同的公司开发自由软件——有时是作为自由职业者——最终在 Red Hat 工作。最近 3 年来，我一直在从事 OpenStack 项目——一个用 Python 编写的 2M·SLOC 云计算项目。

我住在法国巴黎，但我喜欢四处旅行，在不同的地方工作——这是从事分布式开源项目的额外津贴。我喜欢弹吉他，看好看的电视剧，做饭(我是个吃货)，跑步(为几天后的第一次半程马拉松做准备！)和玩第一人称射击游戏(我是个老贵格会教徒)。

**你为什么开始使用 Python？**

好奇心。10 年前看到越来越多的人用。我习惯了 Perl。但是我并不真正喜欢 Perl，也没有很好地理解它的对象系统。

*一旦我找到一个可以工作的想法——如果我没记错的话，那是[重建的](https://julien.danjou.info/projects/rebuildd)——我就开始用 Python 编程，同时学习这门语言。*

我喜欢 Python 的工作方式，以及我开发和学习它的速度，所以我决定在我的下一个项目中继续使用它。由于某些原因，我最终投身于 Python core，甚至在某个时候短暂地黑过 Cython，最终从事 OpenStack。

OpenStack 是一个完全用 Python 编写的云计算平台。从做 OpenStack 开始，每天都在写 Python。

这促使我在 2013 年写了《黑客的 Python 指南》,并在一年后的 2014 年自行出版，在那里我谈到了如何做智能高效的 Python。

它取得了巨大的成功，甚至被翻译成了中文和韩文，所以我目前正在写这本书的第二版。这是一次惊人的冒险！

你还知道哪些编程语言，你最喜欢哪一种？

在 2007-2010 年期间，我一直在从事 C 和 Lua 方面的工作，当时我在开发令人敬畏的窗口管理器和一些 X11 库，所以我对它们都非常了解。

我也做过很多 Emacs Lisp 开发人员——我是 Lisp 迷——和一些 Common Lisp 开发人员。当你知道 Lisp 时，学习任何新语言都很容易，但你会变得懒惰，因为他们似乎都不如它。🙂

你现在在做什么项目？

我 100%的时间都在 OpenStack 上工作。更具体地说，我关注云高仪和汤团。云测仪负责计量 OpenStack 云平台及其周边。Gnocchi 是我几个月前开始的一个新项目，它提供了一个 REST API 来操作资源目录和时间序列数据库。它旨在以分布式和高度可扩展的方式存储您的资源(例如虚拟机)及其所有指标(CPU 使用率、带宽等)。

我还帮助维护我们用于 OpenStack 的 Python 工具链和库。我们分解了很多代码，并向 PyPI 发布了很多库。我写补丁，修复 bug，审查代码。

哪些 Python 库是你最喜欢的(核心或第三方)？

我喜欢 functools、operator 和 itertools 但那是因为我喜欢函数式编程。我也对 asyncio 寄予厚望，它是新的 async Python 3 库。我喜欢使用重试而不是编写 *我的 try/except/retry 循环。*

对于第三方，我喜欢 Flask 和 Jinja2，因为它们简单易用。我喜欢 stevedore 是因为它让在你的程序中使用插件变得超级简单，我喜欢 pbr 是因为它让分发你的包变得更容易。

现在我依靠 Pecan 以 OpenStack 的方式构建 REST API。我也很喜欢在团子里用熊猫来操纵时间序列——如果你需要操纵和计算统计数据，这是一个很好的工具箱！

你还有什么想说的吗？

Python 是一门伟大的语言，即使它有一些缺点和设计问题。我真的鼓励人们看一看 Lisp，学习它，使用它一点，对 Python 语言本身有一些后知之明。它会让你更聪明。

*黑客快乐！*

非常感谢！

### 一周的最后 10 个 PyDevs

*   马特·哈里森
*   阿迪娜·豪
*   [诺亚礼物](https://www.blog.pythonlibrary.org/2015/04/20/pydev-of-the-week-noah-gift/)
*   道格拉斯·斯塔内斯
*   [可降解的脊椎动物](https://www.blog.pythonlibrary.org/2015/04/06/pydev-of-the-week-lennart-regebro/)
*   迈克·弗莱彻
*   丹尼尔·格林菲尔德
*   伊莱·本德斯基
*   [Ned Batchelder](https://www.blog.pythonlibrary.org/2015/03/09/pydev-of-the-week-ned-batchelder/)
*   布莱恩·奥克利