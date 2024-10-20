# 绕过隐藏函数

> 原文：<https://www.blog.pythonlibrary.org/2014/06/16/working-around-shadowed-functions/>

最近，我遇到了一个问题，调用 Python 的应用程序会将 **int** 插入 Python 的名称空间，这会覆盖 Python 的内置 **int** 函数。因为我必须使用这个工具，而且我需要使用 Python 的 **int** 函数，所以我需要一种方法来解决这个麻烦。

幸运的是，这很容易解决。你所需要做的就是从 **__builtin__** 导入 **int** 并重命名它，这样你就不会覆盖插入的版本:

```py

from __builtin__ import int as py_int

```

这使您可以再次访问 Python 的 **int** 函数，尽管它现在被称为 **py_int** 。只要不叫 **int** ，随便你怎么叫。

隐藏内置变量或其他变量最常见的情况是当开发人员从包或模块中导入所有内容时:

```py

from something import *

```

当你像上面一样进行导入时，你并不总是知道你导入了什么，你可能最终会写下你自己的变量或函数来隐藏你导入的变量或函数。这就是强烈反对从一个包或模块中导入任何东西的主要原因。

无论如何，我希望这个小提示对你有所帮助。在 Python 3 中，核心开发者添加了一个[内置模块](https://docs.python.org/3/library/builtins.html)基本上就是为了这个目的。