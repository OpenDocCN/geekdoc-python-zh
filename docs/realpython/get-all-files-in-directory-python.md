# 如何用 Python 获得一个目录中所有文件的列表

> 原文：<https://realpython.com/get-all-files-in-directory-python/>

获得一个[目录](https://en.wikipedia.org/wiki/Directory_(computing))中所有[文件](https://en.wikipedia.org/wiki/Computer_file)和文件夹的[列表](https://realpython.com/python-lists-tuples/)是 Python 中许多**文件相关操作**的第一步。然而，当你深入研究它的时候，你可能会惊讶地发现有各种各样的方法去实现它。

当你面对做某事的许多方法时，这可能是一个很好的迹象，表明没有一个放之四海而皆准的解决方案。最有可能的是，每个解决方案都有自己的优势和权衡。在 Python 中获取一个目录的内容列表就是这种情况。

在本教程中，您将重点关注在 [`pathlib`模块](https://realpython.com/python-pathlib/)中列出目录中的项目的最通用的技术，但是您也将了解一些替代工具。

**源代码:** [点击这里下载免费的源代码、目录和额外材料](https://realpython.com/bonus/get-all-files-in-directory-python-code/)，它们展示了用 Python 列出目录中的文件和文件夹的不同方式。

在 Python 3.4 的`pathlib`出现之前，如果你想处理文件路径，那么你可以使用 [`os`](https://docs.python.org/3/library/os.html) 模块。虽然这在性能方面非常高效，但您必须将所有路径作为[字符串](https://realpython.com/python-strings/)来处理。

起初，将路径作为字符串处理似乎还可以，但是一旦您开始将多个操作系统混合在一起，事情就变得更加棘手了。您还会得到一堆与字符串操作相关的代码，这些代码可以从文件路径中抽象出来。事情很快就会变得神秘起来。

**注意:**查看可下载的材料，了解一些可以在您的机器上运行的测试。测试将比较使用来自`pathlib`模块、`os`模块、甚至未来 [Python 3.12](https://realpython.com/python-pre-release/) 版本的`pathlib`的方法返回一个目录中所有条目的列表所花费的时间。这个新版本包含了众所周知的`walk()`功能，这在本教程中不会涉及。

这并不是说将路径作为字符串工作是不可行的——毕竟，开发人员在没有`pathlib`的情况下也能很好地工作很多年！`pathlib`模块只是负责许多棘手的事情，让您专注于代码的主要逻辑。

这一切都是从创建一个`Path`对象开始的，这个对象会因操作系统(OS)的不同而不同。在 Windows 上，你会得到一个`WindowsPath`对象，而 Linux 和 macOS 会返回`PosixPath`:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

***>>>

```py
>>> import pathlib
>>> desktop = pathlib.Path("C:/Users/RealPython/Desktop")
>>> desktop
WindowsPath("C:/Users/RealPython/Desktop")
```

>>>

```py
>>> import pathlib
>>> desktop = pathlib.Path("/home/RealPython/Desktop")
>>> desktop
PosixPath('/home/RealPython/Desktop')
```

有了这些支持操作系统的对象，您可以利用许多可用的方法和属性，比如获取文件和文件夹列表的方法和属性。

**注:**如果你有兴趣了解更多关于`pathlib`及其特性的信息，那么请查看 [Python 3 的 pathlib 模块:驯服文件系统](https://realpython.com/python-pathlib/)和 [`pathlib`文档](https://docs.python.org/3/library/pathlib.html)。

现在，是时候开始列出文件夹内容了。请注意，有几种方法可以做到这一点，选择正确的方法将取决于您的特定用例。

## 用 Python 获取一个目录中所有文件和文件夹的列表

在开始列出清单之前，您需要一组与本教程中遇到的内容相匹配的文件。在补充资料中，你会找到一个名为 *Desktop* 的文件夹。如果你打算跟随，下载这个文件夹并导航到*父文件夹*，在那里启动你的 [Python REPL](https://realpython.com/interacting-with-python/) :

**源代码:** [点击这里下载免费的源代码、目录和额外材料](https://realpython.com/bonus/get-all-files-in-directory-python-code/)，它们展示了用 Python 列出目录中的文件和文件夹的不同方式。

你也可以使用自己的桌面。只需在桌面的父目录中启动 Python REPL，示例应该可以工作，但是输出中会有您自己的文件。

**注意:**在本教程中，你将主要看到作为输出的`WindowsPath`对象。如果你继续使用 Linux 或 macOS，那么你会看到`PosixPath`。这是唯一的区别。你写的代码在所有平台上都是一样的。

如果你只需要列出一个给定目录的内容，而不需要得到每个*子目录*的内容，那么你可以使用`Path`对象的`.iterdir()`方法。如果你的目标是递归地浏览目录和子目录，那么你可以跳到递归列表的[部分。](#recursively-listing-with-rglob)

当在一个`Path`对象上调用`.iterdir()`方法时，该方法返回一个[生成器](https://realpython.com/introduction-to-python-generators/)，该生成器生成代表子项的`Path`对象。如果您将生成器包装在一个`list()`构造函数中，那么您可以看到您的文件和文件夹列表:

>>>

```py
>>> import pathlib
>>> desktop = pathlib.Path("Desktop")

>>> # .iterdir() produces a generator
>>> desktop.iterdir()
<generator object Path.iterdir at 0x000001A8A5110740>

>>> # Which you can wrap in a list() constructor to materialize
>>> list(desktop.iterdir())
[WindowsPath('Desktop/Notes'),
 WindowsPath('Desktop/realpython'),
 WindowsPath('Desktop/scripts'),
 WindowsPath('Desktop/todo.txt')]
```

将由`.iterdir()`生成的生成器传递给`list()`构造函数会为您提供一个表示*桌面*目录中所有项目的`Path`对象列表。

与所有生成器一样，您也可以使用一个`for`循环来迭代生成器生成的每个项目。这使您有机会探索每个对象的一些属性:

>>>

```py
>>> desktop = pathlib.Path("Desktop")
>>> for item in desktop.iterdir():
...     print(f"{item} - {'dir' if item.is_dir() else 'file'}")
...
Desktop\Notes - dir
Desktop\realpython - dir
Desktop\scripts - dir
Desktop\todo.txt - file
```

在 [`for`循环](https://realpython.com/python-for-loop/)主体中，您使用一个 [f 字符串](https://realpython.com/python-f-strings/)来显示每个项目的一些信息。

在 f 字符串的第二组花括号(`{}`)中，如果项目是一个目录，您使用一个[条件表达式](https://realpython.com/python-conditional-statements/#conditional-expressions-pythons-ternary-operator)来打印*目录*，如果不是，则打印*文件*。要获得这些信息，您使用 [`.is_dir()`](https://docs.python.org/3/library/pathlib.html?highlight=is_dir#pathlib.Path.is_dir) 方法。

将一个`Path`对象放在 f 字符串中会自动将该对象转换成一个字符串，这就是为什么不再有`WindowsPath`或`PosixPath`注释的原因。

像这样用一个`for`循环反复遍历对象，对于按文件或目录过滤来说非常方便，如下例所示:

>>>

```py
>>> desktop = pathlib.Path("Desktop")
>>> for item in desktop.iterdir():
...     if item.is_file():
...         print(item)
...
Desktop\todo.txt
```

这里，您使用一个条件语句和 [`.is_file()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.is_file) 方法只打印文件项。

您还可以将生成器放入[理解](https://realpython.com/list-comprehension-python/)中，这可以产生非常简洁的代码:

>>>

```py
>>> desktop = pathlib.Path("Desktop")
>>> [item for item in desktop.iterdir() if item.is_dir()]
[WindowsPath('Desktop/Notes'),
 WindowsPath('Desktop/realpython'),
 WindowsPath('Desktop/scripts')]
```

这里，您通过在理解中使用一个条件表达式来过滤结果列表，以检查项目是否是一个目录。

但是，如果您也需要文件夹子目录中的所有文件和目录，该怎么办呢？您可以将`.iterdir()`改编为递归函数，就像您将在教程的后面做的[一样，但是使用`.rglob()`可能会更好，您将在接下来进行讨论。](#creating-a-recursive-iterdir-function)

[*Remove ads*](/account/join/)

## 用`.rglob()`和递归列表

由于目录的递归性质，目录经常被比作树。在树木中，主干分裂成各种各样的主枝。每个主枝又分成更多的次枝。每个子分支也从自身分支，等等。同样，目录包含子目录，子目录包含子目录，子目录包含更多子目录，等等。

递归地列出目录中的项目意味着不仅要列出目录的内容，还要列出子目录及其子目录的内容，等等。

有了`pathlib`，遍历一个目录出奇的容易。您可以使用`.rglob()`返回所有内容:

>>>

```py
>>> import pathlib
>>> desktop = pathlib.Path("Desktop")

>>> # .rglob() produces a generator too
>>> desktop.rglob("*")
<generator object Path.glob at 0x000001A8A50E2F00>

>>> # Which you can wrap in a list() constructor to materialize
>>> list(desktop.rglob("*"))
[WindowsPath('Desktop/Notes'),
 WindowsPath('Desktop/realpython'),
 WindowsPath('Desktop/scripts'),
 WindowsPath('Desktop/todo.txt'),
 WindowsPath('Desktop/Notes/hash-tables.md'),
 WindowsPath('Desktop/realpython/iterate-dict.md'),
 WindowsPath('Desktop/realpython/tictactoe.md'),
 WindowsPath('Desktop/scripts/rename_files.py'),
 WindowsPath('Desktop/scripts/request.py')]
```

以`"*"`作为参数的`.rglob()`方法产生一个生成器，该生成器递归地从`Path`对象产生所有文件和文件夹。

但是`.rglob()`的星号参数是什么？在下一节中，您将研究 glob 模式，看看除了列出目录中的所有项目之外，您还能做些什么。

## 使用 Python Glob 模式进行条件列表

有时候你不想要所有的文件。有时候，您只需要一种类型的文件或目录，或者名称中包含某种字符模式的所有项目。

与`.rglob()`相关的一种方法是`.glob()`方法。这两种方法都利用了 [glob](https://en.wikipedia.org/wiki/Glob_(programming)) 模式。glob 模式表示路径的集合。Glob 模式利用[通配符](https://en.wikipedia.org/wiki/Wildcard_character)来匹配某些标准。例如，单个星号`*`匹配目录中的所有内容。

您可以利用许多不同的 glob 模式。查看以下 glob 模式选择，了解一些想法:

| 球状图案 | 比赛 |
| --- | --- |
| `*` | 每次 |
| `*.txt` | 以`.txt`结尾的每一项，如`notes.txt`或`hello.txt` |
| `??????` | 名称长度为六个字符的每一项，如`01.txt`、`A-01.c`或`.zshrc` |
| `A*` | 以字符 *A* 开头的每一项，如`Album`、`A.txt`或`AppData` |
| `[abc][abc][abc]` | 名称为三个字符但仅由字符 *a* 、 *b* 、 *c* 组成的项目，如`abc`、`aaa`或`cba` |

使用这些模式，您可以灵活地匹配许多不同类型的文件。查看关于`fnmatch` 的[文档，这是控制`.glob()`行为的底层模块，感受一下 Python 中可以使用的其他模式。](https://docs.python.org/3/library/fnmatch.html#module-fnmatch)

注意，在 Windows 上，glob 模式是不区分大小写的，因为路径通常是不区分大小写的。在像 Linux 和 macOS 这样的类 Unix 系统上，glob 模式是区分大小写的。

### 条件清单使用`.glob()`

一个`Path`对象的`.glob()`方法的行为与`.rglob()`非常相似。如果您传递了`"*"`参数，那么您将获得目录中的条目列表，但是*没有递归*:

>>>

```py
>>> import pathlib
>>> desktop = pathlib.Path("Desktop")

>>> # .glob() produces a generator too
>>> desktop.glob("*")
<generator object Path.glob at 0x000001A8A50E2F00>

>>> # Which you can wrap in a list() constructor to materialize
>>> list(desktop.glob("*"))
[WindowsPath('Desktop/Notes'),
 WindowsPath('Desktop/realpython'),
 WindowsPath('Desktop/scripts'),
 WindowsPath('Desktop/todo.txt')]
```

在一个`Path`对象上使用带有`"*"` glob 模式的`.glob()`方法会产生一个生成器，该生成器生成由`Path`对象表示的目录中的所有项目，而不进入子目录。这样，它产生与`.iterdir()`相同的结果，你可以在`for`循环或理解中使用生成的生成器，就像你使用`iterdir()`一样。

但是正如您已经了解到的，真正使 glob 方法与众不同的是可以用来匹配特定路径的不同模式。例如，如果您只想要以`.txt`结尾的路径，那么您可以执行以下操作:

>>>

```py
>>> desktop = pathlib.Path("Desktop")
>>> list(desktop.glob("*.txt"))
[WindowsPath('Desktop/todo.txt')]
```

因为这个目录只有一个文本文件，所以您得到的列表只有一项。例如，如果您只想获得以 *real* 开头的项目，那么您可以使用下面的 glob 模式:

>>>

```py
>>> list(desktop.glob("real*"))
[WindowsPath('Desktop/realpython')]
```

这个示例也只生成一个项目，因为只有一个项目的名称以字符`real`开头。请记住，在类 Unix 系统上，glob 模式是区分大小写的。

**注意:***名称*在这里指的是路径的最后一部分，而不是路径的其他部分，在这种情况下，其他部分将从`Desktop`开始。

您还可以通过包含子目录的名称、正斜杠(`/`)和星号来获取子目录的内容。这种类型的模式将产生目标目录中的所有内容:

>>>

```py
>>> list(desktop.glob("realpython/*"))
[WindowsPath('Desktop/realpython/iterate-dict.md'),
 WindowsPath('Desktop/realpython/tictactoe.md')]
```

在这个例子中，使用`"realpython/*"`模式产生了`realpython`目录中的所有文件。它会给你与创建一个代表`Desktop/realpython`路径的路径对象并在其上调用`.glob("*")`相同的结果。

接下来，您将进一步研究使用`.rglob()`进行过滤，并了解它与`.glob()`的不同之处。

[*Remove ads*](/account/join/)

### 条件清单使用`.rglob()`

就像使用`.glob()`方法一样，你可以调整`.rglob()`的 glob 模式，只给你一个特定的文件扩展名，除了`.rglob()`将总是递归搜索:

>>>

```py
>>> list(desktop.rglob("*.md"))
[WindowsPath('Desktop/Notes/hash-tables.md'),
 WindowsPath('Desktop/realpython/iterate-dict.md'),
 WindowsPath('Desktop/realpython/tictactoe.md')]
```

通过将`.md`添加到 glob 模式中，现在`.rglob()`只在不同的目录和子目录中生成`.md`文件。

您实际上可以使用`.glob()`,通过调整作为参数传递的 glob 模式，让它以与`.rglob()`相同的方式运行:

>>>

```py
>>> list(desktop.glob("**/*.md"))
[WindowsPath('Desktop/Notes/hash-tables.md'),
 WindowsPath('Desktop/realpython/iterate-dict.md'),
 WindowsPath('Desktop/realpython/tictactoe.md')]
```

在这个例子中，你可以看到对`.glob("**/*.md")`的调用等同于`.rglob(*.md)`。同样，对`.glob("**/*")`的调用相当于`.rglob("*")`。

`.rglob()`方法是使用递归模式调用`.glob()`的一个稍微更显式的版本，所以使用更显式的版本可能比使用普通`.glob()`的递归模式更好。

### 使用 Glob 方法进行高级匹配

glob 方法的一个潜在缺点是，您只能根据 glob 模式选择文件。如果你想在物品的属性上做更高级的匹配或过滤，那么你需要额外的东西。

要运行更复杂的匹配和过滤，您至少可以遵循三种策略。您可以使用:

1.  带有条件检查的`for`循环
2.  有条件表达的理解
3.  内置的`filter()`功能

方法如下:

>>>

```py
>>> import pathlib
>>> desktop = pathlib.Path("Desktop")

>>> # Using a for loop
>>> for item in desktop.rglob("*"):
...     if item.is_file():
...         print(item)
...
Desktop\todo.txt
Desktop\Notes\hash-tables.md
Desktop\realpython\iterate-dict.md
Desktop\realpython\tictactoe.md
Desktop\scripts\rename_files.py
Desktop\scripts\request.py

>>> # Using a comprehension
>>> [item for item in desktop.rglob("*") if item.is_file()]
[WindowsPath('Desktop/todo.txt'),
 WindowsPath('Desktop/Notes/hash-tables.md'),
 WindowsPath('Desktop/realpython/iterate-dict.md'),
 WindowsPath('Desktop/realpython/tictactoe.md'),
 WindowsPath('Desktop/scripts/rename_files.py'),
 WindowsPath('Desktop/scripts/request.py')]

>>> # Using the filter() function
>>> list(filter(lambda item: item.is_file(), desktop.rglob("*")))
[WindowsPath('Desktop/todo.txt'),
 WindowsPath('Desktop/Notes/hash-tables.md'),
 WindowsPath('Desktop/realpython/iterate-dict.md'),
 WindowsPath('Desktop/realpython/tictactoe.md'),
 WindowsPath('Desktop/scripts/rename_files.py'),
 WindowsPath('Desktop/scripts/request.py')]
```

在这些示例中，您首先使用`"*"`模式调用了`.rglob()`方法，以递归方式获取所有项目。这将生成目录及其子目录中的所有项目。然后使用上面列出的三种不同的方法来过滤掉不是文件的项目。注意，在 [`filter()`](https://realpython.com/python-filter-function/) 的例子中，你使用了一个[λ](https://realpython.com/python-lambda/)函数。

glob 方法非常通用，但是对于大型目录树，它们可能有点慢。在下一节中，您将研究一个例子，在这个例子中，使用`.iterdir()`来实现更可控的迭代可能是一个更好的选择。

## 选择不列出垃圾目录

比方说，你想找到你系统上的所有文件，但是你有各种各样的子目录，这些子目录有很多很多的子目录和文件。一些最大的子目录是你不感兴趣的临时文件。

例如，检查这个目录树，它有很多垃圾目录！实际上，这个完整的目录树有 1850 行长。无论你在哪里看到一个省略号(`...`)，这意味着在那个位置有数百个垃圾文件:



```py
large_dir/
├── documents/
│   ├── notes/
│   │   ├── temp/
│   │   │   ├── 2/
│   │   │   │   ├── 0.txt
│   │   │   │   ...
│   │   │   │
│   │   │   ├── 0.txt
│   │   │   ...
│   │   │
│   │   ├── 0.txt
│   │   └── find_me.txt
│   │
│   ├── tools/
│   │   ├── temporary_files/
│   │   │   ├── logs/
│   │   │   │   ├──0.txt
│   │   │   │   ...
│   │   │   │
│   │   │   ├── temp/
│   │   │   │   ├──0.txt
│   │   │   │   ...
│   │   │   │
│   │   │   ├── 0.txt
│   │   │   ...
│   │   │
│   │   ├── 33.txt
│   │   ├── 34.txt
│   │   ├── 36.txt
│   │   ├── 37.txt
│   │   └── real_python.txt
│   │
│   ├── 0.txt
│   ├── 1.txt
│   ├── 2.txt
│   ├── 3.txt
│   └── 4.txt
│
├── temp/
│   ├── 0.txt
│   ...
│
└── temporary_files/
    ├── 0.txt
    ...
```

这里的问题是你有垃圾目录。垃圾目录有时叫做`temp`，有时叫做`temporary files`，有时叫做`logs`。更糟糕的是，它们无处不在，可以在任何层次筑巢。好消息是您不必列出它们，因为您将在接下来学习。

[*Remove ads*](/account/join/)

### 使用`.rglob()`过滤整个目录

如果使用`.rglob()`，只要在`.rglob()`生产出来之后就可以过滤掉了。要正确丢弃垃圾目录中的路径，您可以检查路径中的任何元素是否与[目录列表中的任何元素匹配，以跳过](https://en.wikipedia.org/wiki/Blacklist_(computing)):

>>>

```py
>>> SKIP_DIRS = ["temp", "temporary_files", "logs"]
```

这里，您将`SKIP_DIRS`定义为一个列表，其中包含您想要排除的路径字符串。

用一个星号作为参数调用`.rglob()`将产生所有项目，甚至是那些您不感兴趣的目录中的项目。因为您必须遍历所有项目，所以如果您只查看路径的名称，可能会有一个问题:

```py
large_dir/documents/notes/temp/2/0.txt
```

由于*名称*只是`0.txt`，它不会匹配`SKIP_DIRS`中的任何项目。您需要检查被阻止名称的整个路径。

您可以使用`.parts`属性获取路径中的所有元素，该属性包含路径中所有元素的元组:

>>>

```py
>>> import pathlib
>>> temp_file = pathlib.Path("large_dir/documents/notes/temp/2/0.txt")
>>> temp_file.parts
('large_dir', 'documents', 'notes', 'temp', '2', '0.txt')
```

然后，您需要做的就是检查`.parts`元组中的任何元素是否在要跳过的目录列表中。

你可以通过利用[集合](https://realpython.com/python-sets/)来检查任意两个可重复项是否有一个公共项。如果您将其中一个 iterables 转换为一个集合，那么您可以使用`.isdisjoint()`方法来确定它们是否有任何共同的元素:

>>>

```py
>>> {"documents", "notes", "find_me.txt"}.isdisjoint({"temp", "temporary"})
True

>>> {"documents", "temp", "find_me.txt"}.isdisjoint({"temp", "temporary"})
False
```

如果两个集合没有共同的元素，那么`.isdisjoint()`返回`True`。如果两个集合至少有一个元素相同，那么`.isdisjoint()`返回`False`。您可以将该检查合并到一个`for`循环中，该循环遍历由`.rglob("*")`返回的所有项目:

>>>

```py
>>> SKIP_DIRS = ["temp", "temporary_files", "logs"]
>>> large_dir = pathlib.Path("large_dir")

>>> # With a for loop
>>> for item in large_dir.rglob("*"):
...     if set(item.parts).isdisjoint(SKIP_DIRS):
...         print(item)
...
large_dir\documents
large_dir\documents\0.txt
large_dir\documents\1.txt
large_dir\documents\2.txt
large_dir\documents\3.txt
large_dir\documents\4.txt
large_dir\documents\notes
large_dir\documents\tools
large_dir\documents\notes\0.txt
large_dir\documents\notes\find_me.txt
large_dir\documents\tools\33.txt
large_dir\documents\tools\34.txt
large_dir\documents\tools\36.txt
large_dir\documents\tools\37.txt
large_dir\documents\tools\real_python.txt
```

在这个例子中，您打印了`large_dir`中不在任何垃圾目录中的所有项目。

要检查路径是否在某个不想要的文件夹中，您将`item.parts`转换为一个集合，并使用`.isdisjoint()`来检查`SKIP_DIRS`和`.parts` *是否没有任何共同的项目。如果是这种情况，则打印该项目。*

您也可以使用`filter()`和理解来实现相同的效果，如下所示:

>>>

```py
>>> # With a comprehension
>>> [
...     item
...     for item in large_dir.rglob("*")
...     if set(item.parts).isdisjoint(SKIP_DIRS)
... ]

>>> # With filter()
>>> list(
...     filter(
...         lambda item: set(item.parts).isdisjoint(SKIP_DIRS),
...         large_dir.rglob("*")
...     )
... )
```

不过，这些方法已经变得有点晦涩难懂了。不仅如此，它们的效率也不是很高，因为`.rglob()`生成器必须生成*所有的*项，这样匹配操作才能丢弃那个结果。

你肯定可以用`.rglob()`过滤掉整个文件夹，但是你不能逃避这样一个事实，即生成的生成器将产生*所有的项目*，然后一个接一个地过滤掉不需要的项目。这可能会使 glob 方法非常慢，这取决于您的用例。这就是为什么您可能选择递归`.iterdir()`函数，您将在接下来探索它。

[*Remove ads*](/account/join/)

### 创建递归`.iterdir()`函数

在垃圾目录的例子中，如果给定子目录中的所有文件与`SKIP_DIRS`中的某个名称匹配，那么理想情况下，您希望能够*选择退出*来迭代这些文件:

```py
# skip_dirs.py

import pathlib

SKIP_DIRS = ["temp", "temporary_files", "logs"]

def get_all_items(root: pathlib.Path, exclude=SKIP_DIRS):
    for item in root.iterdir():
        if item.name in exclude:
            continue
        yield item
        if item.is_dir():
            yield from get_all_items(item)
```

在这个模块中，您定义了一个字符串列表`SKIP_DIRS`，它包含了您想要忽略的目录的名称。然后定义一个[生成器函数](https://realpython.com/introduction-to-python-generators/#using-generators)，它使用`.iterdir()`遍历每一项。

生成器函数在第一个参数后使用了[类型注释](https://realpython.com/python-type-checking/#annotations) `: pathlib.Path`来表示不能只传入代表路径的字符串。参数需要是一个`Path`对象。

如果项目名称在`exclude`列表中，那么您只需移动到下一个项目，一次性跳过整个子目录树。

如果这个项目不在列表中，那么您就放弃这个项目，如果它是一个目录，那么您就在这个目录上再次调用这个函数。也就是说，在函数体内，函数有条件地再次调用同一个函数。这是递归函数的标志。

这个递归函数可以有效地产生您想要的所有文件和目录，排除您不感兴趣的所有文件和目录:

>>>

```py
>>> import pathlib
>>> import skip_dirs
>>> large_dir = pathlib.Path("large_dir")

>>> list(skip_dirs.get_all_items(large_dir))
[WindowsPath('large_dir/documents'),
 WindowsPath('large_dir/documents/0.txt'),
 WindowsPath('large_dir/documents/1.txt'),
 WindowsPath('large_dir/documents/2.txt'),
 WindowsPath('large_dir/documents/3.txt'),
 WindowsPath('large_dir/documents/4.txt'),
 WindowsPath('large_dir/documents/notes'),
 WindowsPath('large_dir/documents/notes/0.txt'),
 WindowsPath('large_dir/documents/notes/find_me.txt'),
 WindowsPath('large_dir/documents/tools'),
 WindowsPath('large_dir/documents/tools/33.txt'),
 WindowsPath('large_dir/documents/tools/34.txt'),
 WindowsPath('large_dir/documents/tools/36.txt'),
 WindowsPath('large_dir/documents/tools/37.txt'),
 WindowsPath('large_dir/documents/tools/real_python.txt')]
```

至关重要的是，您已经设法避免了检查不需要的目录中的所有文件。一旦您的生成器识别出该目录在`SKIP_DIRS`列表中，它就会跳过整个过程。

因此，在这种情况下，使用`.iterdir()`将比同等的 glob 方法更有效。

事实上，如果您需要过滤比 glob 模式更复杂的东西，您会发现`.iterdir()`通常比 glob 方法更有效。然而，如果您需要做的只是递归地获得所有`.txt`文件的列表，那么 glob 方法会更快。

查看一些测试的可下载资料，这些测试展示了用 Python 列出文件的不同方法的相对速度:

**源代码:** [点击这里下载免费的源代码、目录和额外材料](https://realpython.com/bonus/get-all-files-in-directory-python-code/)，它们展示了用 Python 列出目录中的文件和文件夹的不同方式。

有了这些信息，您就可以选择列出所需文件和文件夹的最佳方式了！

## 结论

在本教程中，您已经研究了 Python `pathlib`模块中的`.glob()`、`.rglob()`和`.iterdir()`方法，以便将给定目录中的所有文件和文件夹放入一个列表中。您已经讨论了列出目录的**直接后代**的文件和文件夹，并且您还查看了**递归列表**。

总的来说，您已经看到，如果您只需要目录中的基本条目列表，而不需要递归，那么`.iterdir()`是最干净的方法，这要归功于它的描述性名称。这项工作效率也更高。然而，如果你需要一个递归列表，那么你最好使用`.rglob()`，这将比一个等价的递归`.iterdir()`更快。

您还研究了一个例子，在这个例子中，使用`.iterdir()`递归地列出可以产生巨大的性能优势——当您有垃圾文件夹而您想选择不迭代时。

在可下载的资料中，您会发现从`pathlib`和`os`模块中获取基本文件列表的方法的各种实现，以及对它们进行计时的几个脚本:

**源代码:** [点击这里下载免费的源代码、目录和额外材料](https://realpython.com/bonus/get-all-files-in-directory-python-code/)，它们展示了用 Python 列出目录中的文件和文件夹的不同方式。

检查它们，修改它们，并在评论中分享你的发现！*******