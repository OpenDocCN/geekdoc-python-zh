# 高级 Python——如何动态加载模块或类

> 原文：<https://www.blog.pythonlibrary.org/2012/07/31/advanced-python-how-to-dynamically-load-modules-or-classes/>

时不时地，你会发现自己需要动态地加载模块或类。换句话说，您希望能够在事先不知道要导入哪个模块的情况下导入模块。在本文中，我们将研究用 Python 实现这一壮举的两种方法。

### 使用 __import__ Magic 方法

做这类事情最简单的方法是使用“神奇”的方法 __import__。事实上，如果你在谷歌上搜索这个话题，这可能是你找到的第一个方法。下面是基本的方法:

```py

module = __import__(module_name)
my_class = getattr(module, class_name)
instance = my_class()

```

在上面的代码中，**模块名**和**类名**都必须是字符串。如果您正在导入的类需要一些传递给它的参数，那么您也必须添加那个逻辑。这里有一个更具体的例子来帮助你理解这是如何工作的:

```py

########################################################################
class DynamicImporter:
    """"""

    #----------------------------------------------------------------------
    def __init__(self, module_name, class_name):
        """Constructor"""
        module = __import__(module_name)
        my_class = getattr(module, class_name)
        instance = my_class()
        print instance

if __name__ == "__main__":
    DynamicImporter("decimal", "Context")

```

如果运行这段代码，您应该会在 stdout 中看到类似下面的输出:

```py

Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=-999999999, Emax=999999999, capitals=1, flags=[], traps=[DivisionByZero, Overflow, InvalidOperation])

```

这表明代码像预期的那样工作，它导入了十进制的**并返回了**上下文**类的一个实例。那是非常直接的。让我们看看另一种方法！**

### 使用 Python 的 imp 模块

使用 imp 模块稍微复杂一点。您最终需要做一点递归调用，并且您还希望将所有东西都包装在异常处理程序中。让我们看一下代码，然后我会解释为什么:

```py

import imp
import sys

#----------------------------------------------------------------------
def dynamic_importer(name, class_name):
    """
    Dynamically imports modules / classes
    """
    try:
        fp, pathname, description = imp.find_module(name)
    except ImportError:
        print "unable to locate module: " + name
        return (None, None)

    try:
        example_package = imp.load_module(name, fp, pathname, description)
    except Exception, e:
        print e

    try:
        myclass = imp.load_module("%s.%s" % (name, class_name), fp, pathname, description)
        print myclass
    except Exception, e:
        print e

    return example_package, myclass

if __name__ == "__main__":
    module, modClass = dynamic_importer("decimal", "Context")

```

imp 模块带有一个 **find_module** 方法，它将为您查找模块。虽然我不能让它总是可靠地工作，所以我把它包在一个尝试/例外中。例如，它根本找不到 SQLAlchemy，当我试图找到 wx 时。相框，我拿到了，但不是那个。我不知道后一个问题是 wxPython 还是 imp 的问题。无论如何，在 imp 找到模块后，它会返回一个打开的文件处理程序、imp 找到的模块的路径以及排序的描述(参见下面的文档或 PyMOTW 文章)。接下来，您需要加载该模块，使其“导入”。如果你想在模块中取出一个类，那么你可以再次使用 **load_module** 。此时，您应该拥有与第一个方法中相同的对象。

**更新**:我已经有一个评论者完全否定了这篇文章。所以作为回应，我想澄清一些事情。首先，简单的尝试/例外通常是不好的。我修正了代码，但是我总是在博客上，在产品代码中，甚至在书中看到它。这是个好主意吗？不。如果您知道会出现什么错误，您应该处理它们或重新引发错误。其次，有人告诉我“__import__”不是一个“方法”。确实如此。它实际上是 Python 内部的一个函数。然而，我见过的 Python 中的每一个双下划线函数都被称为“神奇方法”。参见[福德的书](http://www.ironpythoninaction.com/magic-methods.html)或者这个[的博客](http://pythonconquerstheuniverse.wordpress.com/2012/03/09/pythons-magic-methods/)，上面列出了一堆关于这个主题的其他资源。最后，我的 IDE (Wingware，如果有人关心的话)自己给类和函数/方法添加了一些愚蠢的东西，比如空的 docstring 或者在 __init__ docstring 中，它会添加“构造函数”。

### 附加阅读

*   [StackOverflow](http://stackoverflow.com/q/4821104/393194) :动态导入模块中类的字符串名的 Python 动态实例化
*   [StackOverflow](http://stackoverflow.com/a/1057898/393194) :只有在运行时才知道包名的情况下，如何使用 __import__()导入包？
*   [PyMOTW](http://www.doughellmann.com/PyMOTW/imp/):imp-模块导入机制的接口
*   [imp 模块](http://docs.python.org/library/imp.htm)的 Python 官方文档