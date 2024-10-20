# 本周 PyDev:Yusuke Tsutsumi

> 原文：<https://www.blog.pythonlibrary.org/2018/03/26/pydev-of-the-week-yusuke-tsutsumi/>

本周我们欢迎 Yusuke Tsutsumi 成为我们的本周 PyDev！Yusuke 是 Zillow 的 web 开发人员，从事开源项目已经有几年了。他有一个[博客](http://y.tsutsumi.io/)——在那里你可以了解更多关于他以及各种编程语言的信息。你也可以在 [Github](https://github.com/toumorokoshi?tab=repositories) 上查看他参与了哪些项目。让我们花些时间去更好地了解他！

![](img/6811a5f5208295238b58ae3b81386112.png)

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我在西雅图长大，热爱西北太平洋。在高中涉猎了一点音乐之后，我在大学里选择了电子工程。我研究模拟和生物医学系统(心电图仍然是我所知道的最酷的设备之一)。我还最终了解了嵌入式系统，加上在大学期间帮助改进内容管理系统的一些经验，帮助我在 Zillow 获得了一个职位，也就是我今天工作的地方。

在工作中，我为用于持续交付、测试和监控的工具和基础设施做出贡献，这当然是我的热情所在。在家里，我喜欢阅读、远足和在城市里骑自行车。

**你为什么开始使用 Python？**

Python 不是一个有意的选择:我加入了华盛顿大学的一个开发团队，他们主要在 Plone 上工作，Plone 是用 Python 编写的。我记得我的第一个项目是写一个 Plone 插件，只上了几门 CS 课程。我不知道我在做什么！在那之后，我得到了一份实习工作，在那里我可以自由地使用任何我想要的工具，我选择了 Python。令人敬畏的社区和语言库帮助我解决了所有抛给我的问题。从自动化部署过程到构建分布式 web 服务，Python 一直是一个很好的工具。

你还知道哪些编程语言，你最喜欢哪一种？

我已经涉猎了一些。我用 C#和 Java 开发过游戏，用 Javascript 做过前端工作，在我黑我的 Emacs 系统的时候还学过一点 lisp。我一直在 Rust 做一些副业，我真的很喜欢。很难找到一种具有强大社区、开发人员友好的设计原则和出色性能的语言。结合 CFFI，Rust 也可以与 Python 集成。我认为这两个系统是彼此强大的补充:使用 Python 实现富于表现力的语法和快速原型，使用 Rust 处理高性能是关键的情况。

你现在在做什么项目？

在 Python 中，我主要改进了嬗变核心。readthedocs.io/en/latest/<wbr>，一个使用 Python 的类型模块提供模式验证和生成 API 文档的库。我们已经构建了与 Flask 和 aiohttp 等流行的 web 框架的集成，并在继续改进更多 web 框架的适配器。

去年，我决定停止使用 Emacs，而选择 Atom ( [atom.io](http://atom.io/) )作为我的文本编辑器。构建一个文本编辑器，结合 VIM 的模态编辑能力、Emacs 的可扩展性，并专注于消除使用鼠标，这一直是我的梦想。我一直在构建一个名为 chimera([https://atom.io/packages/<wbr>chimera](https://atom.io/packages/chimera))的 atom 插件，它将 atom 扩展到编辑器中，使这个梦想成为现实。

哪些 Python 库是你最喜欢的(核心或第三方)？

在标准图书馆，我是 asyncio 的超级粉丝。事件驱动的编程范式与受 GIL 约束的 Python 解释器配合得很好，能够显著提高内存利用率，并且在面对长网络连接时也能很好地扩展。我做了一些基准测试来说明它也有显著的性能提升([http://y.tsutsumi.io/aiohttp-<wbr>vs .<wbr>high-io-applications.html](http://y.tsutsumi.io/aiohttp-vs-multithreaded-flask-for-high-io-applications.html))。Python3.5 对我来说是一个转折点，因为异步故事是如此引人注目。我知道我不仅要在家里使用它，而且要让它成为我公司的可行选择。

第三方，我对面料很欣赏([http://www.fabfile.org/](http://www.fabfile.org/))。界面设计得很好，实现稳定的 SSH 连接和命令执行的技术复杂性是巨大的。我喜欢能帮助你解决大问题的库，fabric 使用 Python 的力量实现了大规模编排的执行。

**Is there anything else youâ€™d like to say?** 
A little random, but as you know, I was not a computer science major. I really respect those who have a degree in the field: it's tough, and requires a lot of hard work and dedication. I played a lot of catch up my first few years to match my peers.If you are interested in software and do not have a degree, sometimes the job market can be tough. If you find it difficult to land your first job, don't lose hope. In my experience, the way to getting into the industry is to get really involved. Go to local meetups, blog like crazy, and write code and contribute to open source constantly. Coding is one of the few professions where ability is really the only requirement. So if you have the skills, the next step is to get yourself out there.

感谢您接受采访！