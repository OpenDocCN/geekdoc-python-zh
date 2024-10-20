# 本周 PyDev:托马斯·罗比塔耶

> 原文：<https://www.blog.pythonlibrary.org/2016/04/04/pydev-of-the-week-thomas-robitaille/>

本周，我们欢迎托马斯·罗比塔耶( [@astrofrog](http://twitter.com/astrofrog) )成为我们本周的 PyDev！托马斯是[胶水数据探索包](http://www.glueviz.org/en/stable/)的主要开发者，也是 [Astropy](http://astropy.org/) 项目的主要开发者之一，该项目是一个用于天文学的 Python 库。他还写了一个有趣的 [Python 博客](http://astrofrog.github.io/)。让我们花一点时间来更好地了解我们的同胞 Pythonista！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我目前是一名自由职业的科学软件开发人员，与北美和欧洲的团队一起从事一些项目。然而，这是最近的发展:直到 2015 年底，我还是一名天体物理学家，我主要致力于研究恒星的形成和[辐射转移](https://en.wikipedia.org/wiki/Radiative_transfer)。2008 年，我在英国完成了博士学位，然后在哈佛大学担任研究员，随后在德国马克斯·普朗克天文研究所工作。在我做研究员期间，我学习了 Python，并参与了许多开源科学软件项目。特别是，我积极参与构建用于天文学的 Python 工具，并鼓励天文学家学习 Python，我成为了 [Astropy](http://astropy.org/) 项目的协调员和首席开发人员之一。我最终决定，我想全职从事科学软件开发，并成为一名自由开发者！

—

**你为什么开始使用 Python？**

我从 2008 年开始使用 Python，那时我开始在哈佛工作。在我读博士期间，我遇到过几次 Python，当我搬到美国后，我决定在新工作的头几周花大量时间学习 Python。在我读博士期间，我主要使用 Fortran，虽然它对高性能计算(我当时正在做的)很好，但对于交互式分析和快速开发来说，它并不是理想的语言。当时 Python 在天文学领域的主要竞争对手是[交互式数据语言](https://en.wikipedia.org/wiki/IDL_(programming_language)) (IDL)。虽然 IDL 确实有相当丰富的天文学函数库，但我不想与商业语言捆绑在一起，并且发现 Python 是一种更加灵活和优雅的语言。尽管当时可用的天文学软件包数量很少，但其他科学软件包的巨大生态系统，以及大量其他软件包，对我来说非常有吸引力。

—

你还知道哪些编程语言，你最喜欢哪一种？

作为一名研究人员，我在 Fortran 95 中做了大量高性能计算工作，特别是开发了一个名为 [Hyperion](http://hyperion-rt.org/) 的辐射传输包，以模拟哪些天文物体(如正在形成的恒星或星系)以不同的波长(例如红外线)观察。Fortran 被认为是一种老式的语言，虽然它确实不是一种非常灵活和广泛适用的语言，也不是面向对象编程的最佳选择，但它确实非常适合高性能计算。根据项目的不同，我也使用过许多其他语言，包括 C/C++、R、Perl、Javascript、PHP 等等。但是到目前为止，Python 仍然是我最喜欢的语言，不仅因为这种语言的优雅或简单，还因为可用包的大型生态系统以及用户和开发人员的巨大社区。

—

你现在在做什么项目？

我目前是 [Glue](http://www.glueviz.org/) 包的首席开发人员，这是一个用于数据探索和可视化的交互式 Python 包。这个包最初是由[Chris Beaumont](https://chrisbeaumont.org/)开发的，你可以在这里看到他谈论这个包的早期版本[。我也是 Astropy 的协调员和首席开发人员之一，这是一个为天文学开发一套核心 Python 包的项目。](https://vimeo.com/53378575)[core astropy 包](https://pypi.python.org/pypi/astropy)包含了许多天文学家和科学家的核心功能:例如，有一个非常强大的[unit conversion framework](http://docs.astropy.org/en/stable/units/index.html)，以及一个处理[tabular data](http://docs.astropy.org/en/stable/table/index.html)的框架，它与 [pandas](http://pandas.pydata.org/) 接口良好。除了 Glue 和 Astropy 之外，我还维护了相当多的小 Python 包——其中很多你可以通过[我的 GitHub 配置文件](https://github.com/astrofrog?tab=repositories)找到。

—

哪些 Python 库是你最喜欢的(核心或第三方)？

*我非常喜欢 [requests](http://docs.python-requests.org/en/latest/) 用于简单和 Python 式的网络访问， [Jupyter](http://jupyter.org/) 笔记本(以前的 IPython 笔记本)用于交互式分析， [Vispy](http://vispy.org/) 用于 OpenGL 的 3d 可视化， [scikit-learn](http://scikit-learn.org/) 用于机器学习， [Numpy](http://www.numpy.org/) 用于快速数组操作。我真的很喜欢 [conda](http://conda.pydata.org/docs/) 包管理器，尤其是它能够在一瞬间创建不同的环境。就帮助我进行开发的 Python 工具而言，我从事的大多数项目都有我用 [Sphinx](http://www.sphinx-doc.org/) 构建的文档，虽然严格来说不是 Python 包，但通过 [ReadTheDocs](https://readthedocs.org/) 使用 Sphinx 使事情变得非常容易。我真的很喜欢使用 pytest 框架编写测试，并使用 pytest 插件的大型生态系统。Python 标准库也有很多不错的功能，例如包括[多重处理](https://docs.python.org/3.5/library/multiprocessing.html)、[并发.未来](https://docs.python.org/3.5/library/concurrent.futures.html)和[集合](https://docs.python.org/3.5/library/collections.html)模块。*

—

你还有什么想说的吗？

以我的经验来看，Python 最大的优点之一就是用户和开发人员非常友好的社区，而且很容易找到愿意提供帮助的人。在这个编程社区有时以[咄咄逼人的](https://lkml.org/lkml/2015/9/3/428)或[排外的](http://ironholds.org/blog/down-and-out-in-statistical-computing/)方式行事的时代，我真诚地希望我们继续共同努力，使 Python 社区尽可能友好和包容，无论是在个人社区还是在我们的在线社区！

非常感谢你接受采访！