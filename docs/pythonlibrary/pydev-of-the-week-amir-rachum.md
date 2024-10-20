# 本周 PyDev:阿米尔·拉丘姆

> 原文：<https://www.blog.pythonlibrary.org/2017/06/12/pydev-of-the-week-amir-rachum/>

本周我们欢迎 Amir Rachum 成为我们本周的 PyDev。阿米尔是 [pydocstyle](https://github.com/PyCQA/pydocstyle) 和 [yieldfrom](https://github.com/Nurdok/yieldfrom) 的作者/维护者。阿米尔还写了一个有趣的关于 Python 的小[博客](http://amir.rachum.com/)。让我们花一些时间来更好地了解 Amir！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我是一名来自特拉维夫地区的以色列软件开发人员。我有软件工程学士学位。我在大学四年中花了三年时间在学生岗位上工作，以获得一些现实世界的经验，我相信这一点直到今天都产生了巨大的影响(对我的技能来说是积极的，对我的成绩来说不是那么积极)。

在我的业余时间，我喜欢和朋友一起玩棋盘游戏——到目前为止，我的收藏中有超过 200 种棋盘游戏。

**你为什么开始使用 Python？**

我在学生时代玩过一点 Python，到处都是实用程序脚本，但大多数其他开发人员都使用 Tcl，所以我也跟着学。直到我的下一份工作，我才开始全职用 Python 开发——从 Python 2.4 开始，一直到 2.7。不幸的是，我们仍然没有迁移到 Python 3。

我花了大约 3 年的时间与 Django 合作开发 web 应用程序后端以及异步任务运行服务，该服务与定制的嵌入式板进行交互(并在其上运行)。

你还知道哪些编程语言，你最喜欢哪一种？

我也有在桌面和嵌入式系统中使用 C++的经验。我也涉猎了一些 Java 开发。到目前为止，Python 是我最喜欢的——在没有“官僚”障碍的情况下，我将想法付诸实施的速度是惊人的。

你现在在做什么项目？
 *我是 [pydocstyle](https://github.com/PyCQA/pydocstyle) 的作者和维护者，这是 Python 的一个静态 docstring linter。大多数人都熟悉 PEP8，它是一个定义 Python 编码约定的 PEP。PEP257 是一个类似的 PEP，它定义了 *docstring* 约定——间距、内容等。我发现自己留下了很多关于 docstring 约定的评论，所以我寻找一个工具来自动执行它，就像 [pep8](https://pypi.python.org/pypi/pep8) 为 pep8 所做的那样。我偶然发现了 Vladimir Keleshev 的 pep257，并开始投稿，直到我成为维护者。*

后来，吉多·范·罗苏姆要求这两个项目更改我们的名字，以便不暗示我们的实现是正式的。所以我们决定为“pep8”选择 [pycodestyle](https://pypi.python.org/pypi/pycodestyle) ,为“pep257”选择“pydocstyle”。我们认为这一变化是“pep257”的分支。

随着名称的改变，我们也将项目转移到了 PyCQA 框架中。PyCQA 是 Python 代码质量(未授权)权威，它收集了几个代码质量工具，包括“pydocstyle”、“pycodestyle”、“pylint”、“flake8”等等。

我也刚刚发布了一个名为[yields from](https://github.com/Nurdok/yieldfrom)的小库。这是从 Python 3 到 Python 2 的语义“屈服”的反向移植。它专注于一个非常具体的问题，所以我怀疑它会占用我更多的时间，但是我发现在 Python 2 中使用嵌套生成器时它非常有用。

当我有一个好主意和一些空闲时间时，我也写博客，在那里我写编程(主要是 Python)和相关主题。我特别自豪我在[知识债](http://amir.rachum.com/blog/2016/09/15/knowledge-debt/)上的帖子。

哪些 Python 库是你最喜欢的(核心或第三方)？
 *我是‘pathlib’的忠实粉丝，它突然让文件系统操作变得令人愉快。我个人的任务是将我们的遗留代码一点一点地转换成使用‘path lib’。*

我也喜欢用‘mock’进行测试，用‘doc opt’进行命令行解析。

作为一门编程语言，你认为 Python 将何去何从？
 *向 Python 3 的转移正在发生。我记得当[http://py3readiness.org/](http://py3readiness.org/)有几十个 Python 3 兼容库的时候，今天它显示 PyPI 中前 360 个包中有 343 个支持 Python 3。如果你仔细看看，大多数没有的包，只是它们的 Python 3 分支有不同的名字。*

我在每个新项目中都使用 Python 3，并且我鼓励每个人都这样做。

非常感谢你接受采访！