# Python 101:内省

> 原文：<https://www.blog.pythonlibrary.org/2010/10/14/python-101-introspection/>

无论您是 Python 新手、使用了几年还是专家，了解如何使用 Python 的自省功能都有助于理解您的代码和您刚刚下载的带有蹩脚文档的新包。自省是一个花哨的词，意思是观察自己，思考自己的思想、感觉和欲望。在 Python 世界中，自省实际上有点类似。这种情况下的自省就是用 Python 来搞清楚 Python。在本文中，您可以学习如何使用 Python 来给自己一些关于您正在处理或试图学习的代码的线索。有些人甚至称之为调试的一种形式。

以下是我们将要讲述的内容:

*   类型
*   目录
*   帮助
*   [计]系统复制命令（system 的简写）

需要注意的是，这不是一篇有深度的文章。它会给你一些工具，让你开始行动。但废话少说，我们需要继续下去！

## Python 类型

您可能不知道这一点，但 Python 可能正是您喜欢的类型。是的，Python 可以告诉你你有什么类型的变量，或者从一个函数返回什么类型。这是一个非常方便的小工具。让我们看几个例子来说明这一点:

```py
>>> x = "test"
>>> y = 7
>>> z = None
>>> type(x)
<type 'str'>
>>> type(y)
<type 'int'>
>>> type(z)
<type 'NoneType'>

```

如你所见，Python 有一个名为 **type** 的关键字，可以告诉你什么是什么。在我的实际经验中，我使用 type 来帮助我弄清楚当我的数据库数据损坏或者不是我所期望的时候发生了什么。我只是添加了几行，并打印出每一行的数据及其类型。当我被自己写的一些愚蠢的代码弄糊涂的时候，这给了我很大的帮助。

## Python 目录

什么是 **dir** ？它是当某人说或做一些愚蠢的事情时你说的话吗？不是在这种背景下！不，在 Python 这个星球上， **dir** 关键字(又名:builtin)是用来告诉程序员传入的对象有什么属性的。如果您忘记传入一个对象，dir 将返回当前范围内的名称列表。和往常一样，这用几个例子就比较好理解了。

```py
>>> dir("test")
['__add__', '__class__', '__contains__', '__delattr__',
 '__doc__', '__eq__', '__ge__', '__getattribute__',
 '__getitem__', '__getnewargs__', '__getslice__', '__gt__',
 '__hash__', '__init__', '__le__', '__len__', '__lt__',
 '__mod__', '__mul__', '__ne__', '__new__', '__reduce__',
 '__reduce_ex__', '__repr__', '__rmod__', '__rmul__',
 '__setattr__', '__str__', 'capitalize', 'center',
 'count', 'decode', 'encode', 'endswith', 'expandtabs',
 'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower',
 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower',
 'lstrip', 'replace', 'rfind', 'rindex', 'rjust', 'rsplit',
 'rstrip', 'split', 'splitlines', 'startswith', 'strip',
 'swapcase', 'title', 'translate', 'upper', 'zfill'] 

```

由于 Python 中的一切都是对象，我们可以将一个字符串传递给 dir，并找出它有哪些方法。很整洁，是吧？现在让我们用一个导入的模块来尝试一下:

```py
>>> import sys
>>> dir(sys)
['__displayhook__', '__doc__', '__egginsert', '__excepthook__',
 '__name__', '__plen', '__stderr__', '__stdin__', '__stdout__',
 '_getframe', 'api_version', 'argv', 'builtin_module_names',
 'byteorder', 'call_tracing', 'callstats', 'copyright',
 'displayhook', 'dllhandle', 'exc_clear', 'exc_info',
 'exc_traceback', 'exc_type', 'exc_value', 'excepthook',
 'exec_prefix', 'executable', 'exit', 'exitfunc',
 'getcheckinterval', 'getdefaultencoding', 'getfilesystemencoding',
 'getrecursionlimit', 'getrefcount', 'getwindowsversion', 'hexversion',
 'maxint', 'maxunicode', 'meta_path', 'modules', 'path', 'path_hooks',
 'path_importer_cache', 'platform', 'prefix', 'setcheckinterval',
 'setprofile', 'setrecursionlimit', 'settrace', 'stderr', 'stdin',
 'stdout', 'version', 'version_info', 'warnoptions', 'winver']

```

现在，这很方便！如果您还没有弄明白，dir 函数对于那些您已经下载(或即将下载)的几乎没有文档的第三方包来说是非常方便的。在这些情况下，你如何找到可用的方法？嗯，dir 会帮你搞清楚的。当然，有时文档就在代码本身中，这就把我们带到了内置的帮助实用程序中。

## Python 救命！

Python 附带了一个方便的**帮助**工具。只需在 Python shell 中键入“help()”(去掉引号)，您将看到以下说明(Python 版本可能有所不同...)

```py
>>> help()

Welcome to Python 2.6!  This is the online help utility.

If this is your first time using Python, you should definitely check out
the tutorial on the Internet at http://www.python.org/doc/tut/.

Enter the name of any module, keyword, or topic to get help on writing
Python programs and using Python modules.  To quit this help utility and
return to the interpreter, just type "quit".

To get a list of available modules, keywords, or topics, type "modules",
"keywords", or "topics".  Each module also comes with a one-line summary
of what it does; to list the modules whose summaries contain a given word
such as "spam", type "modules spam".

help>

```

请注意，您现在有了一个**帮助>** 提示，而不是 **> > >** 。在帮助模式下，您可以探索 Python 中的各种模块、关键字和主题。还要注意，当输入单词**模块**时，当 Python 搜索其库文件夹以获取列表时，您会看到一个延迟。如果你已经安装了很多第三方模块，这可能需要一段时间，所以准备好在你等待的时候给自己弄杯摩卡吧。一旦完成，只需按照说明进行操作，我想你会掌握要点的。

## Python sys 模块

是的，如果你大声读出来，标题听起来像是在嘶嘶作响，但我们在这里谈论的是 Python。无论如何，我们关心 Python 的 sys 模块的主要原因是因为它可以告诉我们关于 Python 环境的所有事情。看看它在我的机器上发现了什么:

```py
>>> import sys
>>> sys.executable
'L:\\Python24\\pythonw.exe'
>>> sys.platform
'win32'
>>> sys.version
'2.4.3 (#69, Mar 29 2006, 17:35:34) [MSC v.1310 32 bit (Intel)]'
>>> sys.argv
['']
>>> sys.path
['L:\\Python24\\Lib\\idlelib', 'L:\\Python24\\lib\\site-packages\\icalendar-1.2-py2.4.egg', 'C:\\WINDOWS\\system32\\python24.zip', 'L:\\Python24', 'L:\\Python24\\DLLs', 'L:\\Python24\\lib', 'L:\\Python24\\lib\\plat-win', 'L:\\Python24\\lib\\lib-tk', 'L:\\Python24\\lib\\site-packages', 'L:\\Python24\\lib\\site-packages\\win32', 'L:\\Python24\\lib\\site-packages\\win32\\lib', 'L:\\Python24\\lib\\site-packages\\Pythonwin', 'L:\\Python24\\lib\\site-packages\\wx-2.8-msw-unicode']

```

仅供参考:我现在很少使用 Python 2.4。出于某种原因，它只是碰巧是我在写作时手边的 IDLE 版本。无论如何，正如你所看到的，sys 模块对于弄清楚你的机器和 Python 本身是非常方便的。例如，当您通过 sys.path.add()或 sys.path.remove()键入“import”时，可以使用 sys 在 Python 搜索模块的路径列表中添加或移除路径。

## 包扎

好吧，我希望你能从这篇文章中学到一些东西。几年前，我的老板做了一个关于自省的讲座，我认为这真的很有趣，尽管我知道他在说什么。当我开始写这篇文章时，我决定看看是否有其他人写过关于这个主题的东西，其中最臭名昭著的是帕特里克·奥布莱恩(PyCrust 的作者)2002 年的一篇旧文章。它比这个长得多，但相当有趣。如果你有时间，我也推荐你读一读。