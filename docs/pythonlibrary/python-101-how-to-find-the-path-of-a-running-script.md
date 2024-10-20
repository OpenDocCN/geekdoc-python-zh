# Python 101:如何找到运行脚本的路径

> 原文：<https://www.blog.pythonlibrary.org/2013/10/29/python-101-how-to-find-the-path-of-a-running-script/>

这个话题实际上比它第一次出现时更复杂。在这篇文章中，我们将花一点时间来看看这个问题和一些解决方案。

几年前，我在 wxPython 用户邮件列表中的一个朋友告诉我，他使用了以下方法:

```py

import os
script_path = os.path.dirname(os.path.abspath( __file__ ))

```

这对我很有效，也是我目前使用的方法。顺便说一下，上面的代码返回了绝对路径。根据[文档](http://docs.python.org/2/library/os.path.html#os.path.abspath)，它相当于

```py

import os
os.path.normpath(join(os.getcwd(), path))

```

我也看到有人推荐以下类似的解决方案:

```py

import os
os.path.dirname(os.path.realpath(__file__))

```

文档声明 realpath 将*返回指定文件名的规范路径，消除路径*中遇到的任何符号链接，这听起来可能比我一直使用的解决方案更好。

不管怎样，正如一些人可能指出的，你不能在 IDLE /解释器中使用 **__file__** 。如果这样做，您将得到以下错误:

```py

Traceback (most recent call last):
  File "", line 1, in <module>__file__
NameError: name '__file__' is not defined
```

如果您碰巧通过创建一个类似 py2exe 的可执行文件来“冻结”您的应用程序，您将会得到同样的错误。对于这种情况，有些人会推荐以下替代方案:

```py

import os
os.path.dirname(sys.argv[0])

```

现在，如果您碰巧从另一个脚本调用您的脚本，这将不起作用。我也很确定，当我用一个冻结的应用程序尝试这样做并从一个快捷方式调用可执行文件时，它返回的是快捷方式的路径，而不是可执行文件的路径。然而，我可能会和 os.getcwd()混淆，后者肯定不会可靠地工作。

我用 py2exe 创建的可执行文件的最终解决方案是这样的:

```py

import os, sys
os.path.abspath(os.path.dirname(sys.argv[0]))

```

我很确定 wxPython 的一个核心开发人员推荐过使用它，但是我不能确定，因为我好像已经没有那封邮件了。不管怎样，《深入 Python 的作者 Mark Pilgrim 也推荐使用 os.path.abspath。

现在，我想我将继续使用 os.path.abspath 或 os.path.realpath 来编写脚本，并为我冻结的 Windows 应用程序使用上面的变体。不过，我很想听听你的解决方案。让我知道你是否发现了任何跨平台和/或用于冻结脚本的东西。

### 进一步阅读

*   Python: [如何找到脚本的目录](http://stackoverflow.com/questions/4934806/python-how-to-find-scripts-directory</li>
    <p>)
*   Python 的 os.path [文档](http://docs.python.org/2/library/os.path.html)
*   python，[脚本路径](http://stackoverflow.com/questions/595305/python-path-of-script)
*   深入 Python: [寻找路径](http://www.diveintopython.net/functional_programming/finding_the_path.html)