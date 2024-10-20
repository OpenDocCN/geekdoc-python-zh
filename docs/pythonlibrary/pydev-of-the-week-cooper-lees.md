# 本周 PyDev:库珀·利斯

> 原文：<https://www.blog.pythonlibrary.org/2021/05/24/pydev-of-the-week-cooper-lees/>

本周我们欢迎库珀·里斯([@库珀·里斯](https://twitter.com/cooperlees))成为我们本周的 PyDev！库珀对 Python 编程语言做出了贡献。他也是 [bandersnatch](https://github.com/pypa/bandersnatch) 和 [black](https://github.com/psf/black) 等的维护者。

你可以在库珀的网站或 T2 的 GitHub 网站上看到他还在做什么。让我们花些时间更好地了解库珀！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我是一个重新定居的澳大利亚人，住在美国加利福尼亚州的南太浩湖。我现在受雇于脸书公司，已经在那里工作了 8 年多。总体来说，我喜欢运动，板球+澳大利亚规则足球是我最喜欢的运动，既可以观看也可以玩。我也滑雪和滑雪板(住在山里有帮助)。夏天，我喜欢公路自行车赛。

我在澳大利亚新南威尔士州伍伦贡大学学习“互联网科学”。它原本是一个新的基于互联网的计算机科学/电子工程学位，我相信它已经退休了。从大学开始，我参加了许多 Cisco + Juniper 网络课程和考试+大量的自学。

**你为什么开始使用 Python？**

我在澳大利亚核科技组织的第一份工作开始了我有意义的 python 生活。一位同事告诉我用 Python 编写更多“文明”的代码，而不是我为了完成系统管理任务而拼凑的 perl。我的第一个有意义的 Python 代码是 Python 2.4 中用于 Solaris 10 邮件中继主机的 LDAP to Sendmail 别名文件生成器。我从未体验过< 2.4。

你还知道哪些编程语言，你最喜欢哪一种？

我懂一点 C，Perl，还有围棋。但是我真的很努力地学习了 Python，而且学得相当好。到目前为止，Python 是我最喜欢的语言，因为我的大多数工作负载都需要为 asyncio 编写代码，所以线程池和进程池允许我完成工作。

你现在在做什么项目？

在 OSS 世界中，我目前帮助维护以下软件包:

*   [banders natch](https://github.com/pypa/bandersnatch)-Python PyPI PEP X 镜像软件
*   [黑色](https://github.com/psf/black) -固执己见的 AST 安全型 Python 代码格式
*   [flake 8-bugbear](https://github.com/PyCQA/flake8-bugbear)-AST flake 8 插件发现不良代码气味

在我的日常工作中，我是脸书的一名生产工程师，负责我们内部的路由协议守护进程，这些守护进程运行我们自己的 FBOSS 交换机。我们开发和维护的主要软件是:

*   [开/关](https://github.com/facebook/openr)
*   内部 BGP 守护进程
*   服务器 VIP 注入(通过 thrift + BGP)

是我的主要项目。我还在帮助消除我们数据中心网络中剩余的传统 IPv4 的长尾效应。我们正在慢慢地把它推到我们的边缘。99.我们 99%的内部流量都是 IPv6。

哪些 Python 库是你最喜欢的(核心或第三方)？

*   核心库:偏向于我所贡献的那些库:pathlib + venv

第三方:

*   点击-使用子命令+颜色输出等轻松剪辑。
*   aioprometheus:我喜欢度量，所以我喜欢把普罗米修斯的出口放在我写的所有东西上
*   aiohttp:运行服务器或客户端到基于 http 的服务的最佳异步 http 库
*   uvloop:加速 asyncio 程序的最好方法是使用基于 C libuv 的快速事件循环

你还有什么想说的吗？

贡献开源。从分类，写代码和 CI，都有帮助！概念证明公关是值得的，让你的观点。自动消除你的痛苦。总是添加单元测试！

库珀，谢谢你接受采访！