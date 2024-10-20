# 本周 PyDev:吉姆·克里斯特·哈里夫

> 原文：<https://www.blog.pythonlibrary.org/2020/07/20/pydev-of-the-week-jim-crist-harif/>

本周，我们欢迎 Jim Crist-Harif([@ jcristharif](https://twitter.com/jcristharif))成为我们本周的 PyDev！Jim 是 Dask、Skein 和其他几个数据科学/机器学习 Python 包的贡献者。吉姆[也写了关于 Python](https://jcristharif.com/blog.html) 的博客。你可以在 [Github](https://github.com/jcrist) 上看到他正在做的事情。

让我们花些时间去了解吉姆吧！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

嗨，我是吉姆！我在明尼苏达州的明尼阿波利斯附近长大。在我们的成长过程中，我们没有太多的时间看屏幕，所以直到上了大学，我才真正开始接触电脑。我更喜欢制作实物，并在我父亲的工作室里呆了很长时间。

在大学里，我学的是机械工程，我非常喜欢它，于是继续读研究生，专注于系统动力学和控制。读研最终对我的精神健康相当有害，所以两年后我辞职了，搬到了德克萨斯州，并在 Anaconda 找了份工作。这原来是一个伟大的决定！我在那里的工作是改善 Python 生态系统，这让我可以在[从事各种有趣的](https://dask.org/) [项目](https://jupyterhub-on-hadoop.readthedocs.io/)(这也让我给[做了几次演讲](https://jcristharif.com/talks.html))。

在过去的 5 年里，我的大部分工作都与 [Dask](https://dask.org/) 有关，这是一个灵活的并行计算库。当我开始在 Dask 上工作时，它只是一个小项目——看到它成长为一个大的生态系统，多家公司在它的开发和维护上投入巨资，这是值得的。

工作之余，我热衷于攀岩、骑自行车和木雕。

**你为什么开始使用 Python？**

像任何优秀的研究生一样，我开始研究开源 Python，以此来拖延我的研究工作:)。我正在使用 [SymPy](https://www.sympy.org/) 来帮助推导我的研究的运动方程，并开始反馈我继续工作所需的特征。这逐渐级联，直到 2014 年夏天，我被接受参加谷歌代码之夏，致力于[改进 SymPy 的经典机制和代码生成](https://jcristharif.com/gsoc-week-1.html)。那个夏天，我学到了很多关于软件开发最佳实践的知识(我以前的学术代码很难维护)，到最后，我被深深吸引住了。

你还知道哪些编程语言，你最喜欢哪一种？

我做的大部分工作都是用 Python 写的(还有一些用 Cython 或 C 写的扩展)。我也用 Julia、C++、Java ( [skein](https://jcristharif.com/skein/) )和 Go ( [dask-gateway](https://gateway.dask.org/) )编写和维护过项目。Python 无疑是我最喜欢的面向用户的语言——如果我用另一种语言编写代码，通常是为了向 Python 开发者展示一些东西。

你现在在做什么项目？

目前，我是新一代工作流管理系统[perfect](https://www.prefect.io/)的软件工程师。完美的工作流经常在 Dask 上运行，所以我最近做的很多工作都是在改进他们的 Dask 集成。

作为核心维护团队的一员，我还在 Dask 上工作。我最近在 Dask 的大部分工作都是帮助回应 PRs 和问题，不再是开发了。当我有开发时间时，我主要关注于 Dask 网关，这是一个用于部署和管理 Dask 集群的集中式服务。

哪些 Python 库是你最喜欢的(核心或第三方)？

如果没有以下工具，我的工作效率将会大打折扣:

*   [Conda](https://docs.conda.io/) -跨平台(和语言)依赖管理。在为多个平台(和 Python 版本)开发时，conda 使得创建测试环境和在测试环境之间切换变得很容易。使用一个健壮的依赖解算器，你可以升级软件包而不用(或者至少不用)担心破坏东西。
*   [IPython Shell](https://ipython.org/) -花哨的 Python Shell，我开发的时候总有一个在运行。
*   [黑色](https://black.readthedocs.io/)——不再争论 Python 格式化，自动化吧！
*   可读的测试，强大的夹具，我在任何语言中使用过的最好的测试库。
*   [flake8](https://flake8.pycqa.org/) -除了 pep8 问题之外，flake8 还发现了缺失/不必要的导入、错别字等等！Flake8 作为 CI 的一部分运行在我维护的所有项目中。

非常感谢所有帮助开发这些工具的开发者。

你还有什么想说的吗？

我们一直在为 Dask 寻找新的贡献者——如果有人想开始贡献，我们有很多[好的开始问题](https://github.com/dask/dask/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)。如果你不知道从哪里开始，请随时联系我们的 [gitter](https://gitter.im/dask/dask) ，我们会给你一些指点。

吉姆，谢谢你接受采访！