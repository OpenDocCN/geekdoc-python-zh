# 本周 PyDev:Vinay Sajip

> 原文：<https://www.blog.pythonlibrary.org/2022/10/31/pydev-of-the-week-vinay-sajip/>

本周我们欢迎 Vinay Sajip 成为我们本周的 PyDev！Vinay 是 Python 语言的核心开发者。Vinay 还是 [python-gnupg、](https://pypi.org/project/python-gnupg/)GNU Privacy Guard 的 Python API 以及其他优秀软件包的核心贡献者。

如果你需要 Python 或者工程方面的咨询，可以查看 [Vinay 的网站](https://www.red-dove.com/)。你可以通过访问 Vinay 的 [GitHub 简介](https://github.com/vsajip/)来了解 Vinay 最近在做什么。

让我们花一些时间来更好地了解 Vinay！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我在大学学的是化学工程，在一家建造化工厂的公司找到了一份过程系统工程师的工作。几年后，该行业产能过剩，新工厂建设不多，我感到厌倦，加入了一家制作 CAD 软件的小公司，用定制的操纵杆开发 Apple II。这个软件是用 6502 汇编语言编写的，我学到了很多关于接近金属编程的知识。编程是我的一种爱好，我基本上是自学的——所以拥有一份基于爱好的工作对我来说是最理想的！我没有回头，很快就进入了咨询——做软件开发，项目管理，对别人的软件质量提出建议。

我的其他爱好是阅读、电影和戏剧，我偶尔玩电子游戏。为了锻炼，我步行和远足。

**你为什么开始使用 Python？**

1999 年左右，我在当地图书馆浏览时，偶然发现了马克·卢茨的《编程 Python》。作为一个对编程语言有点痴迷的人，我对这种语言非常着迷，并且看到了它如何在使用起来很有趣的同时极大地提高了我的工作效率。尽管当时 Python 鲜为人知，我的大多数客户都使用 Java 或 C/C++，但我能够开发出让我非常高效的工具，而且(因为我通常从事固定价格的项目),使用 Python 有助于我的咨询底线。

你还知道哪些编程语言，你最喜欢哪一种？

我对相当多的语言非常了解，足以用它们进行生产。Python 是我最喜欢的，但是我不确定我能从其他语言中找到我最喜欢的——它们在不同的场景中有不同的优势和适用性。我相当擅长使用 C/C++、JavaScript/TypeScript、Java、Kotlin、C#、Rust、Go、Dart、Ruby、D、Elixir、Pascal/Delphi 和 Nim。当然，内存管理语言比其他语言更容易使用。

你现在在做什么项目？

我目前正在开发一个解析器生成器，它用 Python 生成词法分析器和递归下降解析器，没有外部依赖性。它仍然是一个进行中的工作，还没有准备好发布。除此之外，我还有许多项目处于维护模式，但我认为自己仍在从事这些项目，因为我偶尔会收到错误报告和功能请求:

*   我维护的 Python 部分——日志、venv 和用于 Windows 的 Python 启动器。

*   Python-gnupg——GNU 隐私保护(GnuPG)的 Python API

*   page sign——age 和 minisign 的 Python 包装器，是 GnuPG 更现代的等价物

*   dist lib——实现一些 Python 打包标准(PEPs)的库，由 pip 使用

*   CFG——一种分层的配置格式，是 JSON 的超集，在 Python、JVM。NET，Go，Rust，D，JavaScript，Ruby，Elixir，Dart 和 Nim

*   simple _ launcher——一个用于 Windows 的可执行启动器，可以用来部署用 Python 和其他语言编写的应用程序。它由 pip 通过 distlib 使用

*   sarge——提供命令管道功能的子进程包装器

哪些 Python 库是你最喜欢的(核心或第三方)？

我喜欢 Django，也喜欢 Bottle web framework(非常被低估)，Trio for async，BeautifulSoup for parsing HTML，Whoosh for the full-text search and requests for HTTP client work。我也使用 Supervisor 进行流程管理，尽管这更多的是应用程序而不是库。

 **你是如何入门 python-gnupg 的？**

我需要一个客户项目的加密/解密和签名/验证功能，该项目涉及核对信用卡交易和与银行系统接口。我意识到 GnuPG 符合要求，但是没有可用的可编程接口。所以我建了一个(虽然是用 C#，不是用 Python——客户端是微软商店)。它工作得很好，所以我在 Python 中寻找一个类似的库，并修改了我找到的一个库——以便与 stdlib 子进程一起工作，并具有扩展的功能。我发布了 python-gnupg。

作为开源软件的创造者或维护者，你学到了什么？

喜欢你的开源软件的人有时会发来感谢的信息——当我做了大量工作将 Django 移植到 Python 3 时(这样它可以在 2.x 和 3.x 上用一个代码库工作),我甚至收到了小礼物作为感谢。但是很少有人会特意公开表示感谢(尽管人们有时会这样做，例如，关于 requests 库或 Flask)。另一方面，不喜欢你的软件的人(不管出于什么原因)会公开指责它，谈论他们如何“讨厌它”，但不会给出任何深思熟虑的理由。您会发现这种情绪经常与 Python 的日志记录有关。负面反馈往往会被放大，我已经学会不理会任何情绪化的、没有具体例子和理由来解释为什么会有问题的东西。

我还发现，人们经常不先讨论问题/功能就直接创建 pr。有时他们会错过目标，导致他们和维护人员浪费时间。我通常认为在制作公关之前提出一个潜在的问题进行讨论更有用。

你还有什么想说的吗？

我希望 Python 打包在一个更好的地方。在这个领域，每个人都有自己的观点，很难在重要的事情上达成共识，所以这可能是一个费力不讨好的领域。但是我喜欢更广泛的 Python 社区和 Python 核心开发社区，因为他们非常友好和乐于助人。

Vinay，谢谢你接受采访！