# py flakes——Python 程序的被动检查器

> 原文：<https://www.blog.pythonlibrary.org/2012/06/13/pyflakes-the-passive-checker-of-python-programs/>

Python 有几个代码分析工具。最著名的是 pylint。接下来是 pychecker，现在我们开始讨论 pyflakes。pyflakes 项目是 Divmod 项目的一部分。与 pychecker 不同，Pyflakes 实际上并不执行它检查的代码。当然，pylint 也不执行代码。不管怎样，我们将快速浏览一下，看看 pyflakes 是如何工作的，以及它是否比竞争对手更好。

### 入门指南

正如您可能已经猜到的，pyflakes 不是 Python 发行版的一部分。你需要从 [PyPI](http://pypi.python.org/pypi/pyflakes) 或者从项目的[启动页面](https://launchpad.net/pyflakes)下载它。一旦你安装了它，你就可以在你自己的代码上运行它。或者您可以跟随我们的测试脚本，看看它是如何工作的。

### 运行 pyflakes

我们将使用一个超级简单且相当愚蠢的示例脚本。事实上，它与我们在 pylint 和 pychecker 文章中使用的是同一个。这又是为了你的观赏乐趣:

```py

import sys

########################################################################
class CarClass:
    """"""

    #----------------------------------------------------------------------
    def __init__(self, color, make, model, year):
        """Constructor"""
        self.color = color
        self.make = make
        self.model = model
        self.year = year

        if "Windows" in platform.platform():
            print "You're using Windows!"

        self.weight = self.getWeight(1, 2, 3)

    #----------------------------------------------------------------------
    def getWeight(this):
        """"""
        return "2000 lbs"

```

正如在其他文章中提到的，这段愚蠢的代码有 4 个问题，其中 3 个会使程序停止运行。让我们看看 pyflakes 能找到什么！尝试运行以下命令，您将看到以下输出:

 `C:\Users\mdriscoll\Desktop>pyflakes crummy_code.py
crummy_code.py:1: 'sys' imported but unused
crummy_code.py:15: undefined name 'platform'` 

虽然 pyflakes 返回这个输出的速度非常快，但是它没有找到所有的错误。getWeight 方法调用传递了太多参数，getWeight 方法本身的定义不正确，因为它没有“self”参数。实际上，你可以把第一个论点称为任何你想要的东西，但是按照惯例，它通常被称为“自我”。如果你按照 pyflakes 告诉你的去修改你的代码，你的代码仍然不会工作。

### 包扎

pyflakes 网站声称 pyflakes 比 pychecker 和 pylint 更快。我没有对此进行测试，但是任何想要这样做的人都可以很容易地对一些大文件进行测试。也许可以获取 BeautifulSoup 文件，或者将它(以及其他文件)与 PySide 或 SQLAlchemy 等复杂的文件进行比较，看看它们之间的差别。我个人感到失望的是，它没有抓住我所寻找的所有问题。我想出于我的目的，我会坚持使用 pylint。这可能是一个方便的工具，用于快速和肮脏的测试，或者只是在 pylint 扫描的结果特别差之后让你感觉好一些。

### 进一步阅读

*   我在 [PyChecker](https://www.blog.pythonlibrary.org/2011/01/26/pychecker-python-code-analysis/) 上的文章
*   PyChecker 官方[网站(类似 pyflakes 的项目)](http://pychecker.sourceforge.net/)
*   来自 Python 杂志的 Doug Hellman 对 Python 静态代码分析器的[评论](http://www.doughellmann.com/articles/pythonmagazine/completely-different/2008-03-linters/index.html)