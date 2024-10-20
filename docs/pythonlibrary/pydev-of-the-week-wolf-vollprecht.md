# 本周 PyDev:Wolf Vollprecht

> 原文：<https://www.blog.pythonlibrary.org/2022/08/08/pydev-of-the-week-wolf-vollprecht/>

本周我们欢迎 Wolf Vollprecht([@ woulf](https://twitter.com/wuoulf))成为我们本周的 PyDev！Wolf 是快速跨平台软件包管理器 [mamba](https://github.com/mamba-org/mamba) 以及用于多维数组表达式数值分析的 C++库 [xtensor](https://github.com/xtensor-stack/xtensor) 的核心维护者。

你可以在 GitHub 上看到一些 Wolf 正在做的事情。让我们花一些时间来更好地了解 Wolf！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我是 Wolf，目前是 QuantStack 的 CTO。我在一台 TI 83-Plus 上开始编程，用的是 Basic，然后继续学习 PHP 去一家网络代理公司工作。上大学的时候，我参加了 Ubuntu App 挑战赛(那是 2012 年的事了)。为此，我开发了一个基于 GTK 的 Markdown 编辑器(叫做 Uberwriter，现在是撇号)，那是我第一次认真使用 Python。后来，我研究了一个叫做 [Beachbot](https://www.youtube.com/watch?v=eBRrQBPtdak) 的小机器人，它可以在沙子上画画，然后我攻读了机器人学硕士学位。我有一个为开源公司工作的愿景，也为 xtensor 做出了贡献，这是一个具有 Python 绑定的 C++库，也是首批 QuantStack 项目之一。这让我在 5 年前加入了 QuantStack。

**你为什么开始使用 Python？**

我开始用 Python 做 Uberwriter！我非常喜欢语言和强大的表现力。在 Uberwriter 之后，我涉足了其他一些个人项目，例如“SilverFlask”，这是一个基于 Flask 和 SQLAlchemy 的 CMS，它的灵感很大程度上来自于 PHPs SilverStripe。SilverFlask 从未去过任何地方，但我学到了很多关于 Python 中抽象基类和元类的知识，这些知识至今仍在帮助我。后来，我基于表达式图开发了一个名为 PyJet 的 Python 到 C++的编译器，它也教会了我很多关于 Python 的知识，以及捕捉这种语言的动态本质有多难。

你还知道哪些编程语言，你最喜欢哪一种？

我经常使用 C++，但是我不太喜欢它。它非常冗长，是一种非常古老的学校语言(尽管人们可以用它做非常快速和复杂的事情)。尤其是当涉及到模板时，错误消息并不重要！我也非常了解 Javascript，但也不太喜欢它。它不是一种非常“深思熟虑”的语言，也不支持运算符重载。我喜欢 Javascript 的一点是它有非常强大的实时编译引擎。有一些新兴的编程语言看起来像是“编译”语言& Python 的有趣交叉——例如 Nim 和 Zig。我梦想中的语言是可编译的，但也是可解释的(这样人们可以在编译时用语言本身运行宏)。我想齐格已经接近了。还有一个新的领先于时代的 Python 编译器 LPython，看起来很有前途(但仍然很早！).

你现在在做什么项目？

我目前正在开发跨所有平台的最终包管理解决方案。这个项目叫做“mamba”，是一个使用“conda”包的包管理器。Mamba 可以在所有平台上运行(Windows、macOS 和 Linux)。我也是提供这些包的最大社区 conda-forge 社区的一员。我们很自豪有超过 4000 名个人贡献者为 6 个不同的平台提供软件包:Windows，3 个 Linux 版本(英特尔，Arm 和 PowerPC)，以及 macOS(英特尔和苹果芯片)。大多数 conda-forge 包都与 Python 和数据科学相关。然而，我们也有大量的 R 编程语言的包，以及许多 C 和 C++，Go 和 Rust 的包。我们真的试图把每个人都集中在一个单一的包管理系统！

我也在“mamba-生态系统”中做两个包:一个是 boa，一个包构建器，另一个是 quetz，一个包服务器。两者都是纯 Python 包。Quetz 正在使用 FastAPI，这对我们来说非常有用。

哪些 Python 库是你最喜欢的(核心或第三方)？

非常好的问题。我可以利用这个机会宣传我们在 QuantStack 工作的一些 Python 库:ipywidgets、ipycanvas、ipyleaflet、bq plot——所有这些都有助于在 Jupyter 笔记本中制作强大的用户界面。JupyterLab 和 Voila 当然也很棒！

总的来说，我真的很喜欢我们在 boa 中使用的“rich”和“prompt_toolkit ”,它对漂亮的终端用户界面帮助很大。有许多经过深思熟虑的 Python 库！

你是如何结束曼巴的工作的？

回到过去，conda 真的很慢(那是在他们添加一些有帮助的优化之前)。康达锻造厂发展得越来越快。我有一个疯狂的想法来构建 ROS 生态系统的所有机器人包(这仍在 robostack 项目中进行)。在任何给定的环境中都有许多 ROS 包，在 conda 的限制下让它工作似乎是不可行的。

因此，我开始摆弄 libsolv，看看它是否会更快(作为另一种 SAT 求解器实现，因为那是真正慢的部分。openSUSE 的 Michael schroder 在 libsolv 中实现了大部分必要的部分，剩下的就是历史了！

**你认为 Python 在数据科学领域最擅长什么？**

我认为 Python 之所以强大，不仅是因为它是高级动态代码(Python)和低级 FORTRAn、C 或 C++代码之间的一种伟大的粘合语言。使用 pybind11 用 C++或者使用 pyo3 用 Rust 编写自己的高速 Python 扩展非常简单。这使得许多流行的数据科学库成为可能，如 NumPy、Pandas、scikit-learn 和 scikit-image 库、Tensorflow、PyTorch 等

你还有什么想说的吗？

我们最近开始认真考虑 Webassembly！我们开发了 JupyterLite(一个在浏览器中运行 Python 的非常棒的工具，例如用于交互式文档！).此外，我们正在尝试为 Webassembly 包准备 conda-forge。这确实是一项令人兴奋的工作，正在 Github 上的“emscripten-forge”组织中进行。

谢谢你接受采访，Wolf！