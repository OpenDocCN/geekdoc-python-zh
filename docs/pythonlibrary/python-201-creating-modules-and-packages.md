# Python 201:创建模块和包

> 原文：<https://www.blog.pythonlibrary.org/2012/07/08/python-201-creating-modules-and-packages/>

创建 Python 模块是大多数 Python 程序员每天都要做的事情。每当您保存一个新的 Python 脚本时，您就创建了一个新的模块。您可以将您的模块导入到其他模块中。包是模块的集合。从标准库中导入到脚本中的东西是模块或包。在本文中，我们将看看如何创建模块和包。我们将在软件包上花更多的时间，因为它们比模块更复杂。

### 如何创建 Python 模块

我们将从创建一个超级简单的模块开始。这个模块将为我们提供基本的算术和没有错误的处理。这是我们的第一个例子:

```py

#----------------------------------------------------------------------
def add(x, y):
    """"""
    return x + y

#----------------------------------------------------------------------
def division(x, y):
    """"""
    return x / y

#----------------------------------------------------------------------
def multiply(x, y):
    """"""
    return x * y

#----------------------------------------------------------------------
def subtract(x, y):
    """"""
    return x - y

```

当然，这段代码有问题。如果您将两个整数传递给 **division** 方法，那么您将最终得到一个整数(在 Python 2.x 中)，这可能不是您所期望的。也没有被零除或混合字符串和数字的错误检查。但这不是重点。关键是，如果您保存了这段代码，您就拥有了一个完全合格的模块。姑且称之为**算术. py** 。现在你能用一个模块做什么呢？您可以导入它并使用任何已定义的模块。我们可以稍加修改使其“可执行”。让我们双管齐下！

首先，我们将编写一个小脚本来导入我们的模块并运行其中的函数:

```py

import arithmetic

print arithmetic.add(5, 8)
print arithmetic.subtract(10, 5)
print arithmetic.division(2, 7)
print arithmetic.multiply(12, 6)

```

现在让我们修改原始脚本，以便我们可以从命令行运行它。下面是一个蹩脚的方法:

```py

#----------------------------------------------------------------------
def add(x, y):
    """"""
    return x + y

#----------------------------------------------------------------------
def division(x, y):
    """"""
    return x / y

#----------------------------------------------------------------------
def multiply(x, y):
    """"""
    return x * y

#----------------------------------------------------------------------
def subtract(x, y):
    """"""
    return x - y

#----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print sys.argv
    v = sys.argv[1].lower()
    valOne = int(sys.argv[2])
    valTwo = int(sys.argv[3])
    if v == "a":
        print add(valOne, valTwo)
    elif v == "d":
        print division(valOne, valTwo)
    elif v == "m":
        print multiply(valOne, valTwo)
    elif v == "s":
        print subtract(valOne, valTwo)
    else:
        pass

```

完成这个脚本的正确方法是使用 Python 的[optparse](http://docs.python.org/library/optparse.html)(2.7 之前)或 [argparse](http://docs.python.org/library/argparse.html#module-argparse) (2.7 以上)模块。不过，你也可以把它当作一种练习。我们需要继续包装！

### 如何创建 Python 包

模块和包之间的主要区别在于，包是模块的集合，它有一个 __init__。py 文件。根据包的复杂程度，它可能有不止一个 __init__.py。让我们看一个简单的文件夹结构来使这一点更明显，然后我们将创建一些简单的代码来遵循该结构。

```py

myMath/
    __init__.py
    adv/
        __init__.py
        sqrt.py
        fib.py
    add.py
    subtract.py
    multiply.py
    divide.py

```

现在我们只需要在我们自己的包中复制这个结构。让我们试一试！在如上所示的文件夹树中创建这些文件。对于加减乘除文件，您可以使用我们在前面的示例中创建的函数。对于另外两个，我们将使用下面的代码。

对于斐波那契数列，我们将使用来自 StackOverflow 的简单代码:

```py

# fib.py
from math import sqrt

#----------------------------------------------------------------------
def fibonacci(n):
    """
    http://stackoverflow.com/questions/494594/how-to-write-the-fibonacci-sequence-in-python
    """
    return ((1+sqrt(5))**n-(1-sqrt(5))**n)/(2**n*sqrt(5))

```

对于 sqrt.py 文件，我们将使用以下代码:

```py

# sqrt.py
import math

#----------------------------------------------------------------------
def squareroot(n):
    """"""
    return math.sqrt(n)

```

您可以保留这两个 __init__。py 文件是空白的，但是这样你就不得不编写类似于 **mymath.add.add(x，y)** 的代码，这有点糟糕，所以我们将下面的代码添加到 outer __init__。py 使使用我们的包更容易。

```py

# outer __init__.py
from add import add
from divide import division
from multiply import multiply
from subtract import subtract
from adv.fib import fibonacci
from adv.sqrt import squareroot

```

现在，一旦我们在 Python 路径上有了模块，我们就应该能够使用它了。为此，您可以将该文件夹复制到 Python 的 **site-packages** 文件夹中。在 Windows 上，它位于以下位置:**C:\ python 26 \ Lib \ site-packages**。或者，您可以在测试代码中动态编辑路径。让我们看看这是怎么做到的:

```py

import sys

sys.path.append('C:\Users\mdriscoll\Documents')

import mymath

print mymath.add(4,5)
print mymath.division(4, 2)
print mymath.multiply(10, 5)
print mymath.fibonacci(8)
print mymath.squareroot(48)

```

注意，我的路径不包括 **mymath** 文件夹。您希望追加包含新模块的父文件夹，而不是模块文件夹本身。如果你这样做，那么上面的代码应该工作。恭喜你！您刚刚创建了一个 Python 包！

### 进一步阅读

*   关于[模块](http://docs.python.org/tutorial/modules.html)的官方 Python 文档
*   Python 征服宇宙[关于包的文章](http://pythonconquerstheuniverse.wordpress.com/2009/10/15/python-packages/)

### 源代码

*   [mymath.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/07/mymath.zip)