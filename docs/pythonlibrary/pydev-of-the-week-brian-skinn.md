# 本周 PyDev:布莱恩·斯金

> 原文：<https://www.blog.pythonlibrary.org/2022/05/23/pydev-of-the-week-brian-skinn/>

本周我们欢迎布莱恩·斯金( [@btskinn](https://twitter.com/btskinn) ) 成为我们本周的 PyDev！Brian 在 python 新闻/个人博客上维护来自 Python 导入日志的[RSS 提要。Brian 在 Python 社区也很活跃。](https://bskinn.github.io/)

让我们花些时间来更好地了解 Brian！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

当然可以！我的背景是化学工程——我拥有凯斯西储大学的学士学位和麻省理工学院的博士学位。在过去的十年里，我一直在俄亥俄州代顿地区的一家小企业工作，做 R&D 的电化学过程，尽管我最近在 [OpenTeams](https://www.openteams.com) 以开发人员和服务合作伙伴关系职位的形式跳到了 Python 世界。我真的很兴奋能够开始更直接地使用 Python 和开源软件。

为了好玩，我喜欢读小说，玩音乐和棋盘游戏，并在我的副业项目上做黑客。我是《德累斯顿档案》的忠实粉丝，刚刚读完斯卡齐的*开居保护协会*，并且一直在研究简·奥斯汀的小说。我演奏单簧管、钢琴和吉他，并在男子四重奏中唱男高音。

**你为什么开始使用 Python？**

从高中开始，我就一直在用各种语言编程(TI-BASIC，一点 C++和 Java，还有很多 Excel VBA)，在研究生院的时候，我还对量子化学产生了兴趣。大约在 2014 年，我决定尝试实施一种我在书上读到的量子化学方法。然而，我只真正了解 Excel VBA 足以建立一些实质性的东西，所以这就是我开始。它...非常可怕。从技术上来说，VBA 可以进行面向对象编程，但这很难，也很难。在大约一千行代码之后，我意识到我必须找到别的东西。我在使用 Linux 的过程中已经知道了 Python，所以我对它进行了一些研究，认为它似乎是一个很有前途的选择，并开始学习...并且从未回头。

你还知道哪些编程语言，你最喜欢哪一种？

各种风格的 VBA 绝对是我最熟悉的另一种语言，它在我职业生涯的工程部分非常有价值——将 Excel、Word 和 Outlook 中所有枯燥的东西自动化极大地提高了工作效率。除此之外，我只知道足够多的 Javascript、Java 和 C 是危险的，但不会用它们构建任何实质性的东西。

你现在在做什么项目？

在开源领域，我最积极的工作是开发用于检查/操作 Sphinx objects.inv 文件的工具。短期来看，CLI 有一些方面需要改进；从中期来看，我想研究多重处理，以加速内部的一些关键点；从长远来看，我需要改进异常模型，我想尝试改善核心库存 API 的用户体验。

至于其他项目，我刚刚砍掉了第一版的 jupyter-tempvars，这是一个 jupyter 扩展，提供了简单的临时变量管理。我对此做了一些调整，对底层的 [tempvars](https://github.com/bskinn/tempvars) 库做了一些更新，我想(a)为 conda-forge 打包它,( b)为 JupyterLab 改编它。我还想彻底修改 [pent](https://github.com/bskinn/pent) 的解析器构造模型，它从长格式字符串文本中提取结构化数据(例如，数字数组)。

哪些 Python 库是你最喜欢的(核心或第三方)？

我经常使用标准库中的 pathlib 和 argparse。我最近发现了 pprint(漂亮的印刷)和 calendar，它们非常有用。itertools(以及第三方 more-itertools)非常适合编写简洁、易读、实用的代码。

就第三方包而言，我的大多数项目都使用 [attrs](https://github.com/python-attrs/attrs) ，甚至在引入 dataclasses 之后——我喜欢转换、验证等的灵活性。特色。我最近选了 beautiful soup([beautiful soup 4](https://pypi.org/project/beautifulsoup4/))和 [sqlalchemy](https://pypi.org/project/SQLAlchemy/) ，我真的很喜欢它们。科学栈( [numpy](https://pypi.org/project/numpy/) 、 [scipy](https://pypi.org/project/scipy/) 、[熊猫](https://pypi.org/project/pandas/)等。)当然是关键。

对于工具，我大量使用了 [Sphinx](https://pypi.org/project/Sphinx/) 、 [pytest](https://pypi.org/project/pytest/) 、 [tox](https://pypi.org/project/tox/) 、 [coverage](https://pypi.org/project/coverage/) 和 [flake8](https://pypi.org/project/flake8/) ，并且我最近开始在我的项目中添加[预提交](https://pypi.org/project/pre-commit/)。我使用 [setuptools](https://pypi.org/project/setuptools/) 构建包。

你的 pylogging feed 是怎么来的，进展如何？

我以前写过一些博客，在用 Python 写了几年代码后，重新开始关注我用 Python 做的事情是有意义的。到目前为止，博客的目标主要分为两部分:( I)在相对较高的层次上描述我一直在做的事情，以及(ii)详细描述具体的技术元素，试图解释事情是如何工作的。我也有一个早期的暗示，我可能想从化学工程转向 Python，并且知道博客的投资组合方面可能是有益的。博客在过去的一年里一直萎靡不振——另外，不幸的是，因为我从事代码工作的空闲时间非常少，我想专注于推进项目，而不是写博客。我希望在不久的将来唤醒它。

你还有什么想说的吗？

If you've ever thought about contributing to an open-source project, but haven't done it yet -- go for it! It's definitely intimidating at first, because there's a lot to learn about the tooling, the processes, the etiquette, and so on. But, most maintainers are very welcoming to new contributors, and are happy to guide them through everything. I would recommend looking into making your first contributions to small- to medium-sized projects, though, and ones with at least a measure of visibility (a dozen or more Github stars, say). This will hopefully guide you toward projects with engaged maintainer(s) that will have bandwidth to engage with you in some detail. (I will note, a larger project may still work for this; you can monitor the flow of issues/PRs on the repo, and if new issues and PRs are getting steady engagement from maintainers, then it might work well.) Be aware that your first contribution doesn't necessarily need to involve code -- clarifying something in the documentation, fixing a typo in the README or contributor's guide, and other buffs to a project are quite valuable, too!

**Thanks for doing the interview, Brian!**