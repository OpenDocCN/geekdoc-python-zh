# Python 101:关于导入的一切

> 原文：<https://www.blog.pythonlibrary.org/2016/03/01/python-101-all-about-imports/>

作为一名初学 Python 的程序员，您首先要学习的一件事就是如何导入其他模块或包。但是，我注意到，即使是随便使用 Python 多年的人，也不总是知道 Python 的导入基础设施有多灵活。在本文中，我们将探讨以下主题:

*   常规进口
*   使用来自
*   相对进口
*   可选进口
*   本地进口
*   进口陷阱

* * *

### 常规进口

一个常规的导入，很可能是最流行的导入是这样的:

```py

import sys

```

您所需要做的就是使用单词“import ”,然后指定您想要实际导入的模块或包。不过，import 的好处是它还可以一次导入多个包:

```py

import os, sys, time

```

虽然这样可以节省空间，但这违背了 Python 风格指南的建议，即把每个导入放在自己的行上。

有时当您导入一个模块时，您想要重命名它。Python 很容易支持这一点:

```py

import sys as system

print(system.platform)

```

这段代码只是将我们的导入重命名为“system”。我们可以像以前一样调用所有的模块方法，但是使用新的名称。还有一些必须使用点符号导入的子模块:

```py

import urllib.error

```

你不会经常看到这些，但了解它们是有好处的。

* * *

### 使用从模块导入内容

很多时候你只想导入模块或库的一部分。让我们看看 Python 是如何做到这一点的:

```py

from functools import lru_cache

```

上面的代码允许你直接调用 **lru_cache** 。如果您以正常方式只导入了 **functools** ，那么您将不得不像这样调用 lru_cache:

```py

functools.lru_cache(*args)

```

根据你正在做的事情，以上事实上可能是一件好事。在复杂的代码库中，知道某些东西是从哪里导入的非常好。然而，如果你的代码被很好地维护和模块化，从模块中导入一部分代码会非常方便和简洁。

当然你也可以使用中的**方法来导入所有内容，就像这样:**

```py

from os import *

```

这在极少数情况下很方便，但也可能会搞乱名称空间。问题是你可能定义了你自己的函数或一个顶级变量，它与你导入的一个项目同名，如果你试图使用来自 **os** 模块的那个，它将使用你的。所以你最终会得到一个相当混乱的逻辑错误。Tkinter 模块真的是我见过的标准库中唯一推荐全部导入的。

如果你碰巧编写了自己的模块或包，有些人建议把所有东西都导入到你的 **__init__。py** 让你的模块或者包更容易使用。我个人更喜欢显性的，而不是隐性的，而是各取所需。

您也可以通过从一个包中导入多个项目来折中:

```py

from os import path, walk, unlink
from os import uname, remove

```

在上面的代码中，我们从 os 模块导入了五个函数。您还会注意到，我们可以通过多次从同一个模块导入来实现这一点。如果您愿意，也可以使用括号来导入大量项目:

```py

from os import (path, walk, unlink, uname, 
                remove, rename)

```

这是一种有用的技术，但是您也可以用另一种方式来实现:

```py

from os import path, walk, unlink, uname, \
                remove, rename

```

上面看到的反斜杠是 Python 的行继续符，它告诉 Python 这行代码在下一行继续。

* * *

### 相对进口

PEP 328 描述了相对导入是如何产生的，以及选择了什么特定的语法。其背后的想法是使用句点来确定如何相对导入其他包/模块。原因是为了防止标准库模块的意外隐藏。让我们使用 PEP 328 建议的示例文件夹结构，看看我们是否能让它工作:

```py

my_package/
    __init__.py
    subpackage1/
        __init__.py
        module_x.py
        module_y.py
    subpackage2/
        __init__.py
        module_z.py
    module_a.py

```

在硬盘上的某个地方创建上面的文件和文件夹。在顶层 **__init__。py** ，将以下代码放在适当的位置:

```py

from . import subpackage1
from . import subpackage2

```

接下来在**子包 1** 中向下导航，并编辑它的 **__init__。py** 要有以下内容:

```py

from . import module_x
from . import module_y

```

现在编辑 **module_x.py** ，使其具有以下代码:

```py

from .module_y import spam as ham

def main():
    ham()

```

最后编辑 **module_y.py** 来匹配这个:

```py

def spam():
    print('spam ' * 3)

```

打开一个终端，将 **cd** 放到有 my_package 的文件夹中，但不要放到 my_package 中。运行该文件夹中的 Python 解释器。我使用下面的 **iPython** 主要是因为它的自动完成非常方便:

```py

In [1]: import my_package

In [2]: my_package.subpackage1.module_x
Out[2]: In [3]: my_package.subpackage1.module_x.main()
spam spam spam 
```

相对导入对于创建您可以转换成包的代码非常有用。如果您已经创建了许多相关的代码，那么这可能是一条可行之路。你会发现在 Python 包索引(PyPI)上很多流行的包中都使用了相对导入。另请注意，如果您需要进入多个级别，您可以只使用额外的期间。但是，根据 PEP 328，你真的不应该超过两个。

还要注意的是，如果您要向 **module_x.py** 添加一个“ **if __name__ == '__main__'** ”部分并试图运行它，您将会以一个相当令人困惑的错误结束。让我们编辑文件并尝试一下吧！

```py

from . module_y import spam as ham

def main():
    ham()

if __name__ == '__main__':
    # This won't work!
    main()

```

现在导航到终端中的 subpackage1 文件夹，并运行以下命令:

```py

python module_x.py

```

对于 Python 2，您应该会在屏幕上看到以下错误:

```py

Traceback (most recent call last):
  File "module_x.py", line 1, in from . module_y import spam as ham
ValueError: Attempted relative import in non-package 
```

如果你试着用 Python 3 运行它，你会得到这个:

```py

Traceback (most recent call last):
  File "module_x.py", line 1, in from . module_y import spam as ham
SystemError: Parent module '' not loaded, cannot perform relative import 
```

这意味着 module_x.py 是包内的一个模块，你试图将它作为脚本运行，这与相对导入不兼容。

如果您想在代码中使用这个模块，您必须将它添加到 Python 的导入搜索路径中。最简单的方法如下:

```py

import sys
sys.path.append('/path/to/folder/containing/my_package')
import my_package

```

请注意，您想要的是位于 **my_package** 正上方的文件夹的路径(例如。subpackage1)，而不是 **my_package** 本身。原因是 my_package 是这个包，所以如果你添加了它，你在使用这个包的时候会有问题。让我们转到可选导入！

* * *

### 可选进口

当您有一个想要使用的首选模块或包，但是您还想在它不存在的情况下有一个后备时，可以使用可选导入。例如，您可以使用可选导入来支持软件的多个版本或加速。这里有一个来自包 [github2](http://pythonhosted.org/github2/_modules/github2/request.html) 的例子，演示了如何使用可选的导入来支持不同版本的 Python:

```py

try:
    # For Python 3
    from http.client import responses
except ImportError:  # For Python 2.5-2.7
    try:
        from httplib import responses  # NOQA
    except ImportError:  # For Python 2.4
        from BaseHTTPServer import BaseHTTPRequestHandler as _BHRH
        responses = dict([(k, v[0]) for k, v in _BHRH.responses.items()])

```

[lxml 包](https://github.com/lxml/lxml/blob/master/src/lxml/ElementInclude.py)也使用可选的导入:

```py

try:
    from urlparse import urljoin
    from urllib2 import urlopen
except ImportError:
    # Python 3
    from urllib.parse import urljoin
    from urllib.request import urlopen

```

正如你所看到的，它一直被用来产生巨大的效果，是一个方便的工具来增加你的曲目。

* * *

### 本地进口

局部导入是指将模块导入局部范围。当您在 Python 脚本文件的顶部进行导入时，也就是将模块导入到全局范围内，这意味着后面的任何函数或方法都可以使用它。让我们看看导入到本地范围是如何工作的:

```py

import sys  # global scope

def square_root(a):
    # This import is into the square_root functions local scope
    import math
    return math.sqrt(a)

def my_pow(base_num, power):
    return math.pow(base_num, power)

if __name__ == '__main__':
    print(square_root(49))
    print(my_pow(2, 3))

```

这里我们将 **sys** 模块导入到全局作用域中，但是我们实际上并没有使用它。然后在**平方根**函数中，我们将 Python 的**数学**模块导入到函数的局部范围内，这意味着数学模块只能在平方根函数内部使用。如果我们试图在 **my_pow** 函数中使用它，我们将会收到一个 **NameError** 。继续尝试运行代码，看看这是怎么回事！

使用局部作用域的好处之一是，您可能会使用一个需要很长时间才能加载的模块。如果是这样，将它放入一个很少被调用的函数而不是模块的全局作用域中可能是有意义的。这真的取决于你想做什么。坦白地说，我几乎从未在局部范围内使用导入，主要是因为如果导入分散在整个模块中，很难判断会发生什么。按照惯例，所有的导入都应该在模块的顶部。

* * *

### 进口陷阱

程序员会陷入一些非常常见的导入陷阱。我们将在这里讨论两个最常见的问题:

*   循环进口
*   隐藏的导入

让我们从循环导入开始

#### 循环进口

当您创建两个相互导入的模块时，就会发生循环导入。让我们看一个例子，因为这将使我所指的非常清楚。将以下代码放入名为 **a.py** 的模块中

```py

# a.py
import b

def a_test():
    print("in a_test")
    b.b_test()

a_test()

```

然后在与上面相同的文件夹中创建另一个模块，并将其命名为 **b.py**

```py

import a

def b_test():
    print('In test_b"')
    a.a_test()

b_test()

```

如果您运行这些模块中的任何一个，您应该会收到一个 **AttributeError** 。发生这种情况是因为两个模块都试图相互导入。基本上，这里发生的是模块 a 试图导入模块 b，但是它不能这样做，因为模块 b 试图导入已经在执行的模块 a。我读过一些黑客的解决方法，但是一般来说，你应该重构你的代码来防止这种事情发生

#### 隐藏的导入

当程序员创建一个与 Python 模块同名的模块时，就会发生影子导入(又名名称屏蔽)。让我们创造一个人为的例子！在这种情况下，创建一个名为 **math.py** 的文件，并将以下代码放入其中:

```py

import math

def square_root(number):
    return math.sqrt(number)

square_root(72)

```

现在打开一个终端，试着运行这段代码。当我尝试这样做时，我得到了以下回溯:

```py

Traceback (most recent call last):
  File "math.py", line 1, in import math
  File "/Users/michael/Desktop/math.py", line 6, in <module>square_root(72)
  File "/Users/michael/Desktop/math.py", line 4, in square_root
    return math.sqrt(number)
AttributeError: module 'math' has no attribute 'sqrt'
```

这里发生了什么？当你运行这段代码时，Python 首先会在当前运行脚本的文件夹中寻找一个名为“math”的模块。在这种情况下，它会找到我们正在运行的模块并尝试使用它。但是我们的模块没有一个名为 **sqrt** 的函数或属性，所以产生了一个 **AttributeError** 。

* * *

### 包扎

我们已经在本文中介绍了很多内容，关于 Python 的导入系统还有很多需要学习的地方。有 [PEP 302](https://www.python.org/dev/peps/pep-0302/) 涵盖了导入挂钩，允许你做一些非常酷的事情，比如直接从 github 导入。还有 Python 的 [importlib](https://docs.python.org/3/library/importlib.html) ，非常值得一看。走出去，开始挖掘源代码，了解更多巧妙的技巧。编码快乐！

* * *

### 相关阅读

*   导入[陷阱](http://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html)
*   [Python 2 和 Python 3 中的循环导入](https://gist.github.com/datagrok/40bf84d5870c41a77dc6)
*   Stackoverflow - [Python 第十亿次相对导入](http://stackoverflow.com/questions/14132789/python-relative-imports-for-the-billionth-time)