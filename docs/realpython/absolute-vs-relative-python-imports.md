# Python 中的绝对导入与相对导入

> 原文：<https://realpython.com/absolute-vs-relative-python-imports/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解:[**Python 中的绝对 vs 相对导入**](/courses/absolute-vs-relative-imports-python/)

如果您曾经处理过包含多个文件的 Python 项目，那么您很可能曾经使用过 import 语句。

即使对于拥有几个项目的 Pythonistas 来说，导入也会令人困惑！您可能正在阅读这篇文章，因为您想更深入地了解 Python 中的导入，尤其是绝对和相对导入。

在本教程中，您将了解两者之间的差异，以及它们的优缺点。让我们开始吧！

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 进口快速回顾

你需要对 [Python 模块和包](https://realpython.com/python-modules-packages/)有很好的理解，才能知道导入是如何工作的。Python 模块是一个扩展名为`.py`的文件，Python 包是任何包含模块的文件夹(或者，在 Python 2 中，包含`__init__.py`文件的文件夹)。

当一个模块中的代码需要访问另一个模块或包中的代码时会发生什么？你进口的！

[*Remove ads*](/account/join/)

### 进口如何运作

但是进口到底是怎么运作的呢？假设您像这样导入一个模块`abc`:

```py
import abc
```

Python 会做的第一件事就是在 [`sys.modules`](https://docs.python.org/3/library/sys.html#sys.modules) 中查找名字`abc`。这是以前导入的所有模块的缓存。

如果在模块缓存中没有找到该名称，Python 将继续搜索内置模块列表。这些是 Python 预装的模块，可以在 [Python 标准库](https://docs.python.org/3/library/)中找到。如果在内置模块中仍然没有找到这个名字，Python 就会在由 [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path) 定义的目录列表中搜索它。该列表通常包括当前目录，首先搜索该目录。

当 Python 找到该模块时，它会将其绑定到本地范围内的一个名称。这意味着`abc`现在已经被定义，可以在当前文件中使用，而不用抛出`NameError`。

如果名字没有找到，你会得到一个`ModuleNotFoundError`。你可以在 Python 文档中找到更多关于导入的信息[这里](https://docs.python.org/3/reference/import.html)！

**注意:安全问题**

请注意，Python 的导入系统存在一些重大的安全风险。这很大程度上是因为它的灵活性。例如，模块缓存是可写的，并且可以使用导入系统覆盖核心 Python 功能。从第三方包导入也会使您的应用程序面临安全威胁。

以下是一些有趣的资源，可以帮助您了解更多关于这些安全问题以及如何缓解这些问题的信息:

*   Anthony Shaw 的 Python 中的 10 个常见安全陷阱以及如何避免它们(第 5 点讨论了 Python 的导入系统。)
*   [第 168 集:10 Python 安全漏洞和如何堵塞漏洞](https://talkpython.fm/episodes/show/168/10-python-security-holes-and-how-to-plug-them)来自 TalkPython 播客(小组成员在大约 27:15 开始谈论进口。)

### 导入语句的语法

现在您已经知道了 import 语句是如何工作的，让我们来研究一下它们的语法。您可以导入包和模块。(注意，导入一个包实质上是将包的`__init__.py`文件作为一个模块导入。)您还可以从包或模块中导入特定的对象。

通常有两种类型的导入语法。当您使用第一个时，您直接导入资源，就像这样:

```py
import abc
```

`abc`可以是包，也可以是模块。

当使用第二种语法时，从另一个包或模块中导入资源。这里有一个例子:

```py
from abc import xyz
```

`xyz`可以是模块，子包，或者对象，比如类或者函数。

您还可以选择重命名导入的资源，如下所示:

```py
import abc as other_name
```

这将在脚本中将导入的资源`abc`重命名为`other_name`。它现在必须被称为`other_name`，否则它将不会被识别。

### 导入报表的样式

Python 的官方[风格指南 PEP 8](https://realpython.com/python-code-quality/) ，在编写导入语句时有一些提示。这里有一个总结:

1.  导入应该总是写在文件的顶部，在任何模块注释和[文档字符串](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)之后。

2.  进口应该根据进口的内容来划分。通常有三组:

    *   标准库导入(Python 的内置模块)
    *   相关的第三方导入(已安装但不属于当前应用程序的模块)
    *   本地应用程序导入(属于当前应用程序的模块)
3.  每组导入都应该用空格隔开。

在每个导入组中按字母顺序排列导入也是一个好主意。这使得查找特定的导入更加容易，尤其是当一个文件中有许多导入时。

以下是如何设计导入语句样式的示例:

```py
"""Illustration of good import statement styling.

Note that the imports come after the docstring.

"""

# Standard library imports
import datetime
import os

# Third party imports
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy

# Local application imports
from local_module import local_class
from local_package import local_function
```

上面的导入语句分为三个不同的组，用空格隔开。它们在每个组中也是按字母顺序排列的。

[*Remove ads*](/account/join/)

## 绝对进口

您已经掌握了如何编写 import 语句以及如何像专家一样设计它们的风格。现在是时候多了解一点绝对进口了。

绝对导入使用项目根文件夹的完整路径指定要导入的资源。

### 语法和实例

假设您有以下目录结构:

```py
└── project
    ├── package1
    │   ├── module1.py
    │   └── module2.py
    └── package2
        ├── __init__.py
        ├── module3.py
        ├── module4.py
        └── subpackage1
            └── module5.py
```

有一个目录，`project`，包含两个子目录，`package1`和`package2`。`package1`目录下有两个文件，`module1.py`和`module2.py`。

`package2`目录有三个文件:两个模块`module3.py`和`module4.py`，以及一个初始化文件`__init__.py`。它还包含一个目录`subpackage`，该目录又包含一个文件`module5.py`。

让我们假设以下情况:

1.  `package1/module2.py`包含一个函数，`function1`。
2.  `package2/__init__.py`包含一个类，`class1`。
3.  `package2/subpackage1/module5.py`包含一个函数，`function2`。

以下是绝对进口的实际例子:

```py
from package1 import module1
from package1.module2 import function1
from package2 import class1
from package2.subpackage1.module5 import function2
```

请注意，您必须给出每个包或文件的详细路径，从顶层包文件夹开始。这有点类似于它的文件路径，但是我们用一个点(`.`)代替斜线(`/`)。

### 绝对进口的利弊

绝对导入是首选，因为它们非常清晰和直接。通过查看语句，很容易准确地判断导入的资源在哪里。此外，即使 import 语句的当前位置发生变化，绝对导入仍然有效。事实上，PEP 8 明确建议绝对进口。

然而，根据目录结构的复杂性，有时绝对导入会变得非常冗长。想象一下这样的陈述:

```py
from package1.subpackage2.subpackage3.subpackage4.module5 import function6
```

太荒谬了，对吧？幸运的是，在这种情况下，相对进口是一个很好的选择！

## 相对进口

相对导入指定相对于当前位置(即导入语句所在的位置)要导入的资源。相对导入有两种类型:隐式和显式。Python 3 中不赞成隐式相对导入，所以我不会在这里讨论它们。

[*Remove ads*](/account/join/)

### 语法和实例

相对导入的语法取决于当前位置以及要导入的模块、包或对象的位置。以下是一些相对进口的例子:

```py
from .some_module import some_class
from ..some_package import some_function
from . import some_class
```

你可以看到上面的每个 import 语句中至少有一个点。相对导入使用点符号来指定位置。

单个点意味着所引用的模块或包与当前位置在同一个目录中。两个点表示它在当前位置的父目录中，也就是上面的目录。三个点表示它在祖父母目录中，依此类推。如果您使用的是类似 Unix 的操作系统，您可能会对此很熟悉！

让我们假设您有和以前一样的目录结构:

```py
└── project
    ├── package1
    │   ├── module1.py
    │   └── module2.py
    └── package2
        ├── __init__.py
        ├── module3.py
        ├── module4.py
        └── subpackage1
            └── module5.py
```

回忆文件内容:

1.  `package1/module2.py`包含一个函数，`function1`。
2.  `package2/__init__.py`包含一个类，`class1`。
3.  `package2/subpackage1/module5.py`包含一个函数，`function2`。

您可以这样将`function1`导入到`package1/module1.py`文件中:

```py
# package1/module1.py

from .module2 import function1
```

这里只使用一个点，因为`module2.py`和当前模块`module1.py`在同一个目录中。

您可以这样将`class1`和`function2`导入到`package2/module3.py`文件中:

```py
# package2/module3.py

from . import class1
from .subpackage1.module5 import function2
```

在第一个 import 语句中，单个点意味着您正在从当前包中导入`class1`。记住，导入一个包实际上是将包的`__init__.py`文件作为一个模块导入。

在第二个 import 语句中，您将再次使用一个点，因为`subpackage1`与当前模块`module3.py`在同一个目录中。

### 相对进口的利弊

相对导入的一个明显优势是它们非常简洁。根据当前的位置，他们可以将您之前看到的长得离谱的导入语句变成像这样简单的语句:

```py
from ..subpackage4.module5 import function6
```

不幸的是，相对导入可能会很混乱，特别是对于目录结构可能会改变的共享项目。相对导入也不像绝对导入那样可读，并且不容易判断导入资源的位置。

[*Remove ads*](/account/join/)

## 结论

在这个绝对和相对进口的速成课程结束时做得很好！现在，您已经了解了导入是如何工作的。您已经学习了编写导入语句的最佳实践，并且知道绝对导入和相对导入之间的区别。

凭借您的新技能，您可以自信地从 Python 标准库、第三方包和您自己的本地包中导入包和模块。请记住，您通常应该选择绝对导入而不是相对导入，除非路径很复杂并且会使语句太长。

感谢阅读！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解:[**Python 中的绝对 vs 相对导入**](/courses/absolute-vs-relative-imports-python/)******