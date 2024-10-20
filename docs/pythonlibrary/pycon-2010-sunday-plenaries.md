# PyCon 2010:周日全体会议

> 原文：<https://www.blog.pythonlibrary.org/2010/02/22/pycon-2010-sunday-plenaries/>

2010 年 PyCon】的最后一次全体会议是在周日。在这篇文章中，Van Lindberg 告诉我们，如果我们包括所有的供应商，我们的会议就有超过 1100 人参加。这意味着，对于 2011 年的 PyCon，他们可能不得不将出席人数限制在 1500 人，这样我们就不会在目前的场地上用完房间。这样好吗？我真的不知道。有时候感觉它已经太大了。时间会证明一切。

上午的第一次全体会议由[酱实验室](http://saucelabs.com/)的 [Frank Wierzbicki](http://fwierzbicki.blogspot.com/) 主讲，他谈到了“Jython 的现状”，Python 的 Java 实现。事实证明，2.5.1 版本与普通的 Python 实现有很好的兼容性，因此 [Jython](http://www.jython.org/) 通过了几乎所有的 CPython 测试套件。Jython 还可以与大多数 Java 库和应用程序一起工作，因此您可以两全其美。

Wierzbicki 接着说，任何纯 Python 代码都应该在 Jython 中工作。他给出了 SqlAlchemy、Django、Pylons、pip、web2py 和 distribute 在 Jython 中工作的例子。Jython 目前的计划是争取在今年夏天发布 2.6 版本，让 Jython 达到 Python 2.6 的水平，然后，根据 Python 开发人员将他们的应用程序移植到 Python 3 的进度，他希望 Jython 也开始移植到 3.x。

他呼吁对 Jython 项目提供帮助，因为他们不再有赞助商。然后他用麻省理工学院的 Joseph Chang 用 Jython 玩 Bejeweled 的脚本做了一个演示。很奇怪，但是很酷！

第二次全体会议是关于“空载燕子的状态”，由谷歌工作人员[科林·温特](http://oakwinter.com/code/)主讲。他没有使用幻灯片，因为他说如果我们需要视觉效果，我们可以参考他周六的一次演讲。温特告诉我们他们的解释器如何比 Jython 和 [PyPy](http://codespeak.net/pypy/dist/pypy/doc/) 更快，但它可以在 wazoo 上进行优化。他宣布 Guido 已经批准将 Unladen Swallow 合并到 Python 3.x 代码库，希望能赶上 3.3。温特说，他希望通过合并代码，他们将获得更多的开发人员，他们可以使优化过程达到最快速度，并做出真正快速的解释器。最后，Winter 指出，Unladen Swallow 与所有当前的 Python 代码 100%兼容，并举例说明 Unladen Swallow 使 Django 的运行速度提高了 20%。

最后一场全体会议由企业家安东尼奥·罗德里格斯(Antonio Rodriguez)主持，他是 [tabblo](http://www.tabblo.com/studio) 的创始人(后来他将该公司卖给了惠普)。以下是我在演讲中的一些笔记:

*   success =[e . hack()for e in employees]
*   每台机器都可以运行完整的堆栈。任何人都可以检查完整的树并构建完整的产品。任何人都可以对源代码树的任何部分进行更改。每个人都有承诺位。任何人都可以推进生产。
*   98%的公司从 10 人左右开始
*   商业与技术是一个错误的二分法
*   精益创业应该是瘦弱的创业

他认为 Python 面临的挑战是让人们承诺移植到 3.x，这样分裂就不会继续，标准库中需要更多的电池以及打包问题。我强烈建议等待视频并观看，因为我没有很好地解释他的演讲，我认为他的演讲是我参加过的最好的演讲。