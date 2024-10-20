# 本周 PyDev:Tom Augspurger

> 原文：<https://www.blog.pythonlibrary.org/2017/10/02/pydev-of-the-week-tom-augspurger/>

本周，我们欢迎 Tom Augspurger([@ TomAugspurger](https://twitter.com/tomaugspurger))成为我们本周的 PyDev！汤姆是[熊猫](http://pandas.pydata.org/)、 [dask](https://dask.pydata.org/en/latest/) 和[分布式](https://pypi.python.org/pypi/distributed) Python 包的核心开发者。你可以通过查看他的[博客](http://tomaugspurger.github.io/archives.html)或者在 [Github](https://github.com/TomAugspurger) 上看到汤姆在做什么。让我们花些时间更好地了解汤姆吧！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我开始对 2008 年席卷全球的金融危机感兴趣。这促使我学习经济学，试图弄清楚到底发生了什么，以及如何解决这个问题。我继续在爱荷华大学的研究生院学习经济学。我花了 3 年时间才意识到自己不适合读博士，所以我提前退出了硕士项目。

我的爱好曾经围绕着数据分析和为开源项目做贡献。自从去年有了儿子，养他就成了我唯一的爱好🙂

**你为什么开始使用 Python？**

大约 5 年前，我学习了 Python。在研究生院，我需要一些东西来帮助数据分析和模拟。他们教了我们一些 Matlab，这很好，但是我很快就去寻找从语言设计的角度来看能提供更多东西的替代品。我喜欢研究的编程和软件工程方面，就像我喜欢分析本身一样。

你还知道哪些编程语言，你最喜欢哪一种？

SQL 可能是我唯一能声称懂的其他语言。我已经学习了 R、Matlab、Javascript、Haskell、C 和 C++的一些知识，但 Python 无疑是我的最爱。我确实认为与其他社区互动很重要，这样我们就可以从他们解决问题的方式中获得灵感。我的一些最重要的贡献来自复制 R 开发人员 Hadley Wickham 的 API 设计。

你现在在做什么项目？

我帮助维护 pandas、dask 和 distributed，我忽略了一些我写的兼职项目，比如 engarde 和 stitch。

Pandas 提供了一些对数据分析有用的高性能数据结构，以及一些对这些数据结构进行操作的方法。它是 Python 对 dataframe 的首选实现，由 r。

Dask 和 distributed 是并行化现有科学 python 堆栈的相关项目对。Dask 提供了处理数组或数据帧的 API，这些数组或数据帧看起来像 NumPy 或 Pandas，但是是在大于内存的数据集上并行操作的。Dask 将使用您的单台机器上的所有内核；分布式做同样的事情，但是是针对整个机器集群。

哪些 Python 库是你最喜欢的(核心或第三方)？

当然是在达斯克和熊猫之后🙂我认为 seaborn 值得一提。这是一个基于 Matplotlib 的数据集统计可视化库。Pandas 内置了一些绘图功能，但大多数时候我告诉人们使用 seaborn，因为它做得更好。

你是如何开始熊猫和 dask 项目的？

对于 pandas，我从一个小的 pull 请求开始，修复一个远程数据提供使用的缺失值指示器。尽管只修改了 3-4 行，我还是设法完全弄乱了 git 命令，以创建一个干净的 PR。在维护人员平静地向我介绍了修复一切的过程后，我知道这是一个我想成为其中一员的社区。我一直在帮助解决问题，提交 PRs，并在 StackOverflow 上提交答案。

Dask 与科学 python 生态系统中的其他项目有着紧密的关系。我能够提交 PRs 来实现在 pandas 中工作的特性，但是在 Dask 中还没有实现。现在我在 Anaconda 工作，我的一些工作时间花在了 Dask 和熊猫上。

在开源工作中，你遇到过哪些挑战？

早期，很难将我的代码公开给所有人看。我是编程新手，不仅仅是 Python。至少对我来说，让别人评判你(和你)代码的恐惧已经消失了。

作为维护者，最困难的部分可能是熊猫的问题规模。在撰写本文时，我们有 2，038 个未决问题和 100 个未决拉动请求。很难知道对每个问题给予多少关注。

对于想加入开源项目的新开发者，你有什么建议吗？

找到一个你感兴趣的问题。单单利他主义不会促使我贡献出时间来写代码或解决问题。我需要投入到构建数据分析工具的大目标中(并自己使用这些工具)。

你还有什么想说的吗？

Pandas 和 dask 总是向新的贡献者开放，所以如果你对数据科学充满热情，请随时在 Github 上提出问题。

我目前正在考虑 python 的“可扩展机器学习”。基本上，采用科学家用于日常机器学习项目的工作流数据，并将其扩展到更复杂的模型和更大的数据集。我正在写博客(第一部分在[http://tomaugspurger.github.io/scalable-ml-01.html](http://tomaugspurger.github.io/scalable-ml-01.html))。如果你已经考虑过这个问题，那么请联系我们。

感谢您接受采访！