# PyLint:分析 Python 代码

> 原文：<https://www.blog.pythonlibrary.org/2012/06/12/pylint-analyzing-python-code/>

Python 代码分析可能是一个沉重的主题，但它对改进您的程序非常有帮助。有几个 Python 代码分析器可以用来检查您的代码，看看它们是否符合标准。pylint 可能是最受欢迎的。它的可配置性、可定制性和可插拔性都很强。它还检查你的代码是否符合 Python Core 的官方风格指南 [PEP8](http://www.python.org/dev/peps/pep-0008/) ，并且它还寻找编程错误。我们将花几分钟时间来看看您可以用这个方便的工具做些什么。

### 入门指南

遗憾的是，pylint 没有包含在 Python 中，所以你需要从 [Logilab](https://www.pylint.org/) 或 [PyPI](http://pypi.python.org/pypi/pylint) 下载。如果您安装了 pip，那么您可以像这样安装它:

 `pip install pylint` 

现在您应该已经安装了 pylint 及其所有依赖项。现在我们准备好出发了！

### 分析你的代码

撰写本文时的最新版本是 0.25.1。一旦安装了 pylint，您就可以在命令行上不带任何参数地运行它，以查看它接受哪些选项。现在我们需要一些代码来测试。由于我去年为我的 PyChecker 文章写了一些糟糕的代码，我们将在这里重用它们，看看 pylint 是否会发现同样的问题。应该有四个问题。代码如下:

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

现在让我们对这段代码运行 pylint，看看它会发现什么。您应该会得到如下所示的内容:

 `C:\Users\mdriscoll\Desktop>pylint crummy_code.py
No config file found, using default configuration
************* Module crummy_code
C: 1,0: Missing docstring
C: 4,0:CarClass: Empty docstring
E: 15,24:CarClass.__init__: Undefined variable 'platform'
E: 18,22:CarClass.__init__: Too many positional arguments for function call
E: 21,4:CarClass.getWeight: Method should have "self" as first argument
C: 21,4:CarClass.getWeight: Invalid name "getWeight" (should match [a-z_][a-z0-9
_]{2,30}$)
C: 21,4:CarClass.getWeight: Empty docstring
R: 21,4:CarClass.getWeight: Method could be a function
R: 4,0:CarClass: Too few public methods (1/2)
W: 1,0: Unused import sys`

报告
= = = = =
分析了 13 个报表。

按类别分类的消息

+---+--+
| type | number | previous | difference |
+= = = = = = = = = = = = = = = = =+= = = = = = = = = = = = = = = = = =+
|约定| 4 | NC | NC |
+---+----+
|重构| 2 | NC | NC |
+----+
|警告| 1 | NC | NC |
+-----+--+--+

消息

+--+
|消息 id |出现次数|
+= = = = = = = = = = = = = = = = = = = = =+
| c 0112 | 2 |
+---+
| w 0611 | 1 |
+--+
+--+
| 1 |+-+
| r 0201 | 1 |
+-+-+
| e 1121 | 1 |
。
+--+
| c 0111 | 1 |
+--+
| c 0103 | 1 |
+-+-+

全局评估
-
您的代码被评为-6.92/10

按类型统计
-

+--+--+---+--+
|类型|编号|旧编号|差异| %已记录| % bad name |
+= = = = = = = = = = = = = = = =+= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =+= = = = = = = = = = = =+
|模块| 1 | NC | NC | NC | 0.00 | 0.00 |
+---+----+--+----+-+-+
|类| 1 | NC | NC | NC | 0
+-+-+-+-+-+
|函数| 0 | NC | NC | 0 |
+-+-+-+-+-+

原始指标

+---+--+--+
| type | number | % | previous | difference |
+= = = = = = = = = =+= = = = = = = = = = = = = = = = = = = = = = =+= = = = = = =+
| code | 12 | 63.16 | NC | NC |
+---+-+--+
| docstring | 4 | 21.05 | NC |
+-----+---+---+--+--+--+-+-+
。

重复

+--+--+--+
| |现在|以前|差别|
+= = = = = = = = = = = = = = = = = = = = = =+= = = = = = = = = = = =+= = = = = = = =+
| nb 重复行| 0 | NC | NC |
+---+--+-+
|百分比重复行| 0.000 | NC | NC |
+-------+--+

如果您想知道 Messages 部分中的这些项目的含义，您可以通过在命令行上执行以下操作让 pylint 告诉您:

 `pylint --help-msg=C0112` 

然而，我们实际上只关心报告的第一部分，因为其余部分基本上只是以更模糊的方式显示相同信息的表格。让我们更仔细地看看这一部分:

 `C: 1,0: Missing docstring
C: 4,0:CarClass: Empty docstring
E: 15,24:CarClass.__init__: Undefined variable 'platform'
E: 18,22:CarClass.__init__: Too many positional arguments for function call
E: 21,4:CarClass.getWeight: Method should have "self" as first argument
C: 21,4:CarClass.getWeight: Invalid name "getWeight" (should match [a-z_][a-z0-9
_]{2,30}$)
C: 21,4:CarClass.getWeight: Empty docstring
R: 21,4:CarClass.getWeight: Method could be a function
R: 4,0:CarClass: Too few public methods (1/2)
W: 1,0: Unused import sys` 

首先我们需要弄清楚字母代表什么:C 代表约定，R 代表重构，W 代表警告，E 代表错误。pylint 发现了 3 个错误、4 个约定问题、2 行值得重构的代码和 1 个警告。这 3 个错误加上警告正是我想要的。我们应该努力让这个糟糕的代码变得更好，减少问题的数量。我们将修复导入并将 getWeight 函数改为 get_weight，因为 camelCase 不允许用于方法名。我们还需要修复对 get_weight 的调用，以便它传递正确数量的参数，并修复它，以便它将“self”作为第一个参数。下面是新代码:

```py

# crummy_code_fixed.py
import platform

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

        self.weight = self.get_weight(3)

    #----------------------------------------------------------------------
    def get_weight(self, this):
        """"""
        return "2000 lbs"

```

让我们对 pylint 运行一下，看看我们对结果的改进有多大。为简洁起见，我们只展示下面的第一部分:

 `C:\Users\mdriscoll\Desktop>pylint crummy_code_fixed.py
No config file found, using default configuration
************* Module crummy_code_fixed
C: 1,0: Missing docstring
C: 4,0:CarClass: Empty docstring
C: 21,4:CarClass.get_weight: Empty docstring
W: 21,25:CarClass.get_weight: Unused argument 'this'
R: 21,4:CarClass.get_weight: Method could be a function
R: 4,0:CarClass: Too few public methods (1/2)` 

那帮了大忙！如果我们添加 docstrings，我们可以将问题的数量减半。

### 包扎

下一步将是尝试对您自己的一些代码或像 SQLAlchemy 这样的 Python 包运行 pylint，看看会得到什么输出。使用这个工具，您可以学到很多关于您自己的代码的知识。如果你有 Wingware 的 Python IDE，你可以安装 pylint 作为一个工具，只需点击几下鼠标就可以运行。你可能会发现有些警告很烦人，甚至不太适用。有一些方法可以通过命令行选项来抑制诸如不推荐使用警告这样的事情。或者您可以使用 **- generate-rcfile** 来创建一个示例配置文件，帮助您控制 pylint。注意，pylint 不会导入您的代码，所以您不必担心不良副作用。

此时，您应该准备好开始改进您自己的代码库。向前迈进，让您的代码令人惊叹！

### 进一步阅读

*   [PyLint](http://www.logilab.org/857) 官网
*   我在 [PyChecker](https://www.blog.pythonlibrary.org/2011/01/26/pychecker-python-code-analysis/) 上的文章
*   PyChecker 官方[网站(与 PyLint 类似的项目)](http://pychecker.sourceforge.net/)
*   另一个类似的项目
*   来自 Python 杂志的 Doug Hellman 对 Python 静态代码分析器的[评论](http://www.doughellmann.com/articles/pythonmagazine/completely-different/2008-03-linters/index.html)