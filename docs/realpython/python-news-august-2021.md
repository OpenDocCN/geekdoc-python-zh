# Python 新闻:2021 年 8 月有什么新消息

> 原文：<https://realpython.com/python-news-august-2021/>

暑假结束了，又回到了学校。虽然对我们许多人来说，这是一个休闲的时代，脱离了虚拟世界，但 Python 的维护者和贡献者在同一时期一直忙于工作。就在 2021 年 8 月的**，Python 社区看到了三个新的 Python 版本，带来了一些重要的安全修复、优化和全新的特性。**

让我们深入了解过去一个月最大的 Python 新闻！

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## Python 3.10 差不多准备好了

8 月 3 日，Python 3.10.0 的[首个预览版终于出来了。](https://pythoninsider.blogspot.com/2021/08/python-3100rc1-is-available.html)

几个月来，Python 社区一直屏住呼吸，期待着他们最喜欢的语言的下一个小版本。Python 3.10.0 将包含大量令人兴奋的新特性和改进。有些甚至引发了一点[的争议](https://lwn.net/Articles/845480/)，但这通常是开创性的变化。

如果你迫不及待地想在 Python 3.10.0 于 10 月 4 日正式发布之前进行一次测试，那么你可以抓住上个月初发布的候选版本。最终版本将只包括错误修复，不会增加额外的功能，所以预览版是相当完整的。

您可以通过几种不同的方式访问预览版:

*   您可以通过 web 浏览器导航到官方的 [Python 3.10.0rc1 下载页面](https://www.python.org/downloads/release/python-3100rc1/)，并获取要编译的源代码或适用于您的操作系统的 Python 安装程序。
*   或者，您可以在 [pyenv](https://realpython.com/intro-to-pyenv/) 的帮助下，与其他 Python 解释器一起运行候选版本。
*   最后，您可以通过在一个 [Docker 容器](https://realpython.com/python-versions-docker/)中运行解释器来尝试最新的 Python 版本，而无需安装它。

最具革命性的即将到来的变化，值得在这里简单提一下，是在语言的语法中增加了与匹配的[结构模式。](https://www.python.org/dev/peps/pep-0634/)[模式匹配](https://en.wikipedia.org/wiki/Pattern_matching)是一些函数式编程语言中的强大构造，比如 Scala，它可以让你的代码更加简洁可靠。在某些情况下，它还会让你模仿 [`switch`语句](https://en.wikipedia.org/wiki/Switch_statement)，Python 因为没有而被[批评。](https://www.python.org/dev/peps/pep-3103/)

但这只是皮毛而已！还会有很多其他的改进，所以请继续关注未来的*真正的 Python* 教程，它将为你分解 Python 3.10 中的大多数新特性。

[*Remove ads*](/account/join/)

## Python 3.9 和 3.8 变得更安全

Python 3.9.7 和 Python 3.8.12 都是 8 月 30 日发布的[。](https://pythoninsider.blogspot.com/2021/08/python-397-and-3812-are-now-available.html)

尽管 Python 3.10 将很快成为该语言的最新版本，并将提供一些前沿特性，但它要得到第三方库供应商的广泛支持还需要一段时间。因此，大多数商业使用 Python 的公司可能会坚持使用稍旧的版本，因为旧版本更稳定，更经得起考验。

[Python 3.9.7](https://www.python.org/downloads/release/python-397/) 现在是你应该考虑安装的最新**稳定版**。这个版本包括几十个安全和[漏洞修复](https://realpython.com/python-bugfix-version/)以及小的优化和改进。Python 3.9 将被支持到大约 2025 年 10 月。

[Python 3.8.12](https://www.python.org/downloads/release/python-3812/) 是遗留 Python 3.8 系列的第二个**纯安全**补丁。与此同时，尽管没有定期的维护版本，它仍将支持到 2024 年 10 月。

## PyCharm 2021.2.1 变得更快更好

在 8 月 27 日的一篇博客文章中，JetBrains 宣布发布 PyCharm 2021.2.1，这是他们为 Python 开发者开发的非常受欢迎的 IDE 的最新版本。

随着 Python 3.10 的出现，为 Python 生态系统中的软件开发人员提供工具的公司必须为这个新的 Python 版本将带来的巨大变化做好准备。 [PyCharm](https://realpython.com/pycharm-guide/) 现在为 Python 3.10 中引入的新语法结构提供支持，比如**结构模式匹配**和[联合类型](https://www.python.org/dev/peps/pep-0604/)。

除了[无数的错误修复](https://confluence.jetbrains.com/display/PYH/PyCharm+2021.2.1+Release+Notes)，以及性能和可用性的改进，PyCharm 中另一个有趣的创新是从 Python 的内置`venv`模块转移到用于[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)创建的`virtualenv`库。这一小小的改变通过利用缓存以及链接到目录而不是复制目录来显著提高速度。

如果您是 PyCharm 的新手，那么您会喜欢 IDE 中改进的**特性训练器**插件。它是最近增加的，通过**互动课程**教你如何使用图形用户界面。现在，它有新的课程致力于直接从 IDE 中使用 Git 存储库。

## 2021 年 Django 开发者调查正式启动

8 月 4 日，Django 软件基金会(DSF) [宣布](https://www.djangoproject.com/weblog/2021/aug/04/2021-django-developers-survey/)他们已经开始收集来自世界各地 Django 开发者的意见，以更好地了解社区如何使用他们的 web 框架和相关工具。在与 JetBrains 的合作中，这项调查的目标是收集知识，以帮助他们选择正确的发展方向。

该调查现已结束，但它花了大约 10 分钟完成，并且主要由多项选择题组成。希望很多 Django 用户有机会分享你的宝贵反馈。

调查结果将被汇总、匿名并向公众公开。要了解结果何时出来，你可以在社交媒体上关注 Django 软件基金会和 T2 的 JetBrains。

## Python 有一个打包项目经理

8 月 18 日，Python 软件基金会(PSF) [宣布](https://pyfound.blogspot.com/2021/08/shamika-mohanan-has-joined-psf-as.html)**Shamika Mohan**已经接受了一个新的职位，担任**包装项目经理**。

在上个月的新闻中，你了解到 [PSF 雇佣了 ukasz Langa](https://lukasz.langa.pl/a072a74b-19d7-41ff-a294-e6b1319fdb6e/) 作为第一个全职的 CPython 常驻开发者。由于赞助商的持续支持，这家非营利组织得以聘用另一名全职员工，并签订了一份有保障的两年合同。

Shamika 将负责从 Python 社区收集关于打包生态系统中的挑战和正在进行的项目和计划的反馈，重点是改进 [Python 包索引(PyPI)](https://pypi.org/) 。这是个好消息，考虑到 Python 中打包工具的[前景一直是支离破碎的，没有遵循一个特定的标准。](https://realpython.com/pypi-publish-python-package/)

恭喜你，沙米卡👏

## PyCon US 2022 正在寻找志愿者

8 月 30 日，PyCon 美国组织者[宣布](https://pycon.blogspot.com/2021/08/join-pycon-us-2022-team.html)他们开始寻找愿意为明年的会议贡献时间和知识的志愿者。

在 PyCon 通常有很多机会作为志愿者工作。然而，目前 PSF 特别感兴趣的是为负责**提案征集**流程的委员会招募成员。它涉及三个广泛的任务，在这个[谷歌文档](https://docs.google.com/document/d/1Kxi1oHIT4oyhRk4Syd1o7YjKaRzsfnp5T5B5Pd2rAQA/edit?usp=sharing)中有更详细的描述。

[*Remove ads*](/account/join/)

## Python 的下一步是什么？

8 月，Python 出现了一些令人兴奋的发展。在*真实 Python* 展会上，我们对 Python 的未来感到兴奋，迫不及待地想看看在**9 月**会有什么新东西等着我们。

来自**8 月**的 **Python 新闻**你最喜欢的片段是什么？我们错过了什么值得注意的吗？请在评论中告诉我们，我们可能会在下个月的 Python 新闻综述中介绍您。

快乐的蟒蛇！**