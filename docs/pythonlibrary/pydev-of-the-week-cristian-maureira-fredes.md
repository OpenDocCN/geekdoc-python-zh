# 本周 PyDev:克里斯蒂安·毛雷拉-弗雷德斯

> 原文：<https://www.blog.pythonlibrary.org/2022/07/25/pydev-of-the-week-cristian-maureira-fredes/>

本周，我们欢迎克里斯蒂安·毛雷拉-弗雷德斯( [@cmaureir](https://twitter.com/cmaureir) )成为我们本周的 PyDev！Cristián 是 Python 项目(又名 PySide6)的 [Qt 的核心开发人员。你可以在](https://doc.qt.io/qtforpython/) [GitHub](https://github.com/cmaureir) 或者 Cristián 的[网站](https://maureira.xyz/)上追上 Cristián。

让我们花些时间更好地了解克里斯蒂安！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我的名字是 [Cristián Maureira-Fredes](https://maureira.xyz) ，我来自智利圣安东尼奥，但自 2013 年以来我一直住在德国柏林。

在学习计算机科学之后，我来到德国完成了我的硕士学位，并开始攻读天体物理学博士学位。完成学业后，我决定进入这个行业，并从 2018 年开始在 Qt 公司工作。

在工作中，我目前的头衔是“R&D 高级经理”，这意味着我是几个团队的团队领导，一个专注于 [Qt 框架](https://qt.io)的核心方面，另一个是 Qt 框架的官方 Python 绑定集，Python 项目的 [Qt。](https://doc.qt.io/qtforpython/)

我的大部分业余时间都花在了 Python 社区上，我帮助组织了在 [Python 智利](https://pythonchile.cl)、 [Python 西班牙](https://es.python.org)以及一个名为[Python en espa ol’](https://hablemospython.dev)的新全球社区的活动，该社区包括了所有说西班牙语的国家。

**你为什么开始使用 Python？**

我很幸运地在我就读的[大学](https://www.usm.cl)的计算机实验室(LabComp)找到了一份学生工作，在那里我接触到了许多与系统管理员相关的任务，其中包括任务自动化。我开始用 Bash 编码，然后转向 Perl，在某个时候，有人提到了 Python，当时是 Python 2.5(我想)，但我一开始并不喜欢它，我对 Bash、Perl、Awk 和其他命令行语言和工具很熟悉。

在 2009 年(Python 2.6...也许吧？)，我给了它第二次尝试，并开始更经常地使用它，从那时起，我一直是一个快乐的 Pythonista。

你还知道哪些编程语言，你最喜欢哪一种？

在我的日常工作和学习中，我一直都在使用 C++，这是我非常喜欢的。在学会了用 C 编码之后，C++真的是一个飞跃，让我爱上了编程。在我学习期间，我同时使用了 C++和 Python，从那以后，我的心在它们之间分裂，所以我不能决定。然而，如果你问我关于“编程语言社区”的问题，我会毫不犹豫地选择 Python。

你现在在做什么项目？

我是 Qt for Python 项目的维护者之一，所以这是我的主要职责，这很酷，因为这是一个开源项目，但我是有偿开发和领导的。

除此之外，我真的很幸运成为了将官方 Python 文档翻译成西班牙语的团队的一员，现在我们正在等待 3.11 的发布，继续新的 Python 小版本。

我做的其他事情，但不确定是否应该称之为项目，是我与社区互动的许多平台的机器人，如与 Github、通知、discord 任务自动化等的集成，会议或倡议的小静态网站，数据仪表板等。

在社区方面，我目前正在与前面提到的社区一起组织活动、会议等等，老实说，这有时比编写软件项目的代码要多得多！但是我真的很喜欢这样做。

哪些 Python 库是你最喜欢的(核心或第三方)？

PySide6 因为我一直在非常努力地研究它，并不时用它来创建讲座、研讨会、网络研讨会等等🙂

除此之外，自从我第一次发现 NumPy 以来，它一直是一个令人惊叹的模块，它解决了我在学习期间遇到的一个真正的问题，并且大多数从 NumPy“继承”来的模块也很酷，比如熊猫。

从核心...不确定我是否有喜欢的模块，但我确实喜欢不同内置对象的实现。在过去的几年里，我一直在谈论 CPython 的实现，但是很幽默，所以人们不会感到太害怕而不敢进入语言的核心。例如[我的 2021 年欧洲 Python 演讲](https://www.youtube.com/watch?v=WThKZDUt_UM)或者[我在 PyConUS 2022](https://www.youtube.com/watch?v=xQ0-aSmn9ZA&t=25s) 做的闪电演讲，让说英语的人感受一下非本地人学习 Python 的感受

**你是如何加入 Python 团队的？**

前面说过，我刚读完博士就加入了 Qt 公司，甚至能找到一个需要 Python 和 C++的招聘广告都是一个惊喜，因为我知道 Qt 是纯 C++的。

从大学时代就认识 Qt，这真是梦想成真。一年后，我甚至写了[关于我的经历](https://maureira.xyz/posts/one-year-working-at-the-qt-company.html)，因为这是我从来没有预料到的。

你认为 Qt 对 Python 最好的 3 个特性是什么？

我在这里尽量不要有偏见，但大多数人都提到 PySide(Python 的 Qt)相当于 PyQt，但我们决定做的一个转变是，不要为 Python 用户“只做 Qt 绑定”,所以我们决定为 Python 生态系统添加更多东西，比如:

1.启用 snake_case Qt API 而不是 camelCase 的选项，删除 setters 和 getters 并直接访问 Qt 属性。[更多信息](https://www.qt.io/blog/qt-for-python-6-released)

2.启用像 PyPy 这样的其他解释器。[更多信息](https://www.qt.io/blog/qt-for-python-details-on-the-new-6.3-release)

3.把 Qt-API 分解好，为 NumPy 数组、[不透明容器](https://www.qt.io/blog/qt-for-python-release-6.2)提供 [API，直接访问 C++对象，无需复制](https://www.qt.io/blog/qt-for-python-6.1)

我认为 PySide 的这些方面确实有所不同，我们可以为 Python 用户提供更多的东西来开始使用 Qt。

你还有什么想说的吗？

我要感谢你给我这个机会，我真的希望在 Python 社区分享我的一些生活经验可以激励更多的人加入他们的当地社区，为他们喜欢的项目做出贡献，甚至申请他们一直梦想工作的公司。

克里斯汀，谢谢你接受采访！