# Python 3 的 pathlib 模块:驯服文件系统

> 原文：<https://realpython.com/python-pathlib/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 Python 的 pathlib 模块**](/courses/pathlib-python/)

你是否纠结于 Python 中的文件路径处理？在 Python 3.4 及以上版本中，斗争现在已经结束了！您不再需要为代码而绞尽脑汁，比如:

>>>

```py
>>> path.rsplit('\\', maxsplit=1)[0]
```

或畏缩于以下的冗长:

>>>

```py
>>> os.path.isfile(os.path.join(os.path.expanduser('~'), 'realpython.txt'))
```

在本教程中，您将了解如何在 Python 中使用文件路径(目录和文件的名称)。您将学习读写文件、操作路径和底层文件系统的新方法，还将看到一些如何列出文件和遍历文件的示例。使用`pathlib`模块，上面的两个例子可以用优雅的、可读的 Pythonic 代码重写，比如:

>>>

```py
>>> path.parent
>>> (pathlib.Path.home() / 'realpython.txt').is_file()
```

**免费 PDF 下载:** [Python 3 备忘单](https://realpython.com/bonus/python-cheat-sheet-short/)

## Python 文件路径处理的问题

由于许多不同的原因，使用文件和与文件系统交互是很重要的。最简单的情况可能只涉及读取或写入文件，但有时更复杂的任务就在手边。也许您需要[列出给定类型的目录](https://realpython.com/get-all-files-in-directory-python/)中的所有文件，找到给定文件的父目录，或者创建一个尚不存在的唯一文件名。

传统上，Python 使用常规的[文本字符串](https://realpython.com/python-strings/)来表示文件路径。在 [`os.path`](https://docs.python.org/3/library/os.path.html) 标准库的支持下，这已经足够了，尽管有点麻烦(如简介中的第二个例子所示)。然而，由于[路径不是字符串](https://snarky.ca/why-pathlib-path-doesn-t-inherit-from-str/)，重要的功能遍布标准库，包括像 [`os`](https://docs.python.org/3/library/os.html) 、 [`glob`](https://docs.python.org/3/library/glob.html) 和 [`shutil`](https://docs.python.org/3/library/shutil.html) 这样的库。下面的例子需要三个 [`import`语句](https://realpython.com/absolute-vs-relative-python-imports/)来将所有文本文件移动到一个归档目录中:

```py
import glob
import os
import shutil

for file_name in glob.glob('*.txt'):
    new_path = os.path.join('archive', file_name)
    shutil.move(file_name, new_path)
```

对于用字符串表示的路径，使用常规的字符串方法是可能的，但通常不是一个好主意。例如，不要像常规字符串那样用`+`来连接两个路径，而应该使用`os.path.join()`，它使用操作系统上正确的路径分隔符来连接路径。回想一下，Windows 使用`\`，而 Mac 和 Linux 使用`/`作为分隔符。这种差异会导致难以发现的错误，比如我们在引言中的第一个例子只适用于 Windows 路径。

Python 3.4 ( [PEP 428](https://www.python.org/dev/peps/pep-0428/) )中引入了`pathlib`模块来应对这些挑战。它将必要的功能集中在一个地方，并通过一个易于使用的`Path`对象上的方法和属性使其可用。

早期，其他包仍然使用字符串作为文件路径，但是从 Python 3.6 开始，整个标准库都支持`pathlib`模块，部分原因是添加了一个[文件系统路径协议](https://www.python.org/dev/peps/pep-0519/)。如果你被困在传统的 Python 上，Python 2 也有一个 [backport 可用。](https://github.com/mcmtroffaes/pathlib2)

行动的时间到了:让我们看看`pathlib`在实践中是如何工作的。

[*Remove ads*](/account/join/)

## 创建路径

你真正需要知道的是`pathlib.Path`类。创建路径有几种不同的方法。首先有 `.cwd()`(当前工作目录)`.home()`(你用户的主目录)这样的[类方法:](https://realpython.com/instance-class-and-static-methods-demystified/)

>>>

```py
>>> import pathlib
>>> pathlib.Path.cwd()
PosixPath('/home/gahjelle/realpython/')
```

> **注意:**在整个教程中，我们将假设`pathlib`已经被导入，而不像上面那样拼出`import pathlib`。因为你将主要使用`Path`类，你也可以做`from pathlib import Path`并写`Path`而不是`pathlib.Path`。

路径也可以从其字符串表示形式显式创建:

>>>

```py
>>> pathlib.Path(r'C:\Users\gahjelle\realpython\file.txt')
WindowsPath('C:/Users/gahjelle/realpython/file.txt')
```

处理 Windows 路径的一个小技巧:在 Windows 上，路径分隔符是反斜杠，`\`。然而，在许多上下文中，反斜杠也被用作一个*转义字符*，以表示不可打印的字符。为了避免问题，使用*原始字符串文字*来表示 Windows 路径。这些是前面有一个`r`的字符串。在原始字符串文字中，`\`代表一个文字反斜杠:`r'C:\Users'`。

构建路径的第三种方法是使用特殊操作符`/`连接路径的各个部分。正斜杠运算符的使用独立于平台上的实际路径分隔符:

>>>

```py
>>> pathlib.Path.home() / 'python' / 'scripts' / 'test.py'
PosixPath('/home/gahjelle/python/scripts/test.py')
```

只要至少有一个`Path`对象，`/`就可以连接几个路径或者路径和字符串的混合(如上)。如果你不喜欢特殊的`/`符号，你可以用`.joinpath()`方法做同样的事情:

>>>

```py
>>> pathlib.Path.home().joinpath('python', 'scripts', 'test.py')
PosixPath('/home/gahjelle/python/scripts/test.py')
```

注意，在前面的例子中，`pathlib.Path`由`WindowsPath`或`PosixPath`表示。表示路径的实际对象取决于底层操作系统。(也就是说，`WindowsPath`示例是在 Windows 上运行的，而`PosixPath`示例是在 Mac 或 Linux 上运行的。)更多信息参见[操作系统差异](#operating-system-differences)一节。

## 读写文件

传统上，[在 Python](https://realpython.com/read-write-files-python/) 中读写文件的方法是使用内置的`open()`函数。这仍然是正确的，因为`open()`函数可以直接使用`Path`对象。下面的例子在一个 Markdown 文件中找到所有的头，然后[打印它们](https://realpython.com/python-print/):

```py
path = pathlib.Path.cwd() / 'test.md'
with open(path, mode='r') as fid:
    headers = [line.strip() for line in fid if line.startswith('#')]
print('\n'.join(headers))
```

一个等价的替代方法是在`Path`对象上调用`.open()`:

```py
with path.open(mode='r') as fid:
    ...
```

实际上，`Path.open()`是在幕后调用内置的`open()`。你使用哪个选项主要是个人喜好的问题。

对于简单的文件读写，在`pathlib`库中有一些方便的方法:

*   `.read_text()`:以文本方式打开路径，以字符串形式返回内容。
*   `.read_bytes()`:以二进制/字节模式打开路径，以字节字符串的形式返回内容。
*   `.write_text()`:打开路径，写入字符串数据。
*   `.write_bytes()`:以二进制/字节模式打开路径，向其中写入数据。

这些方法中的每一个都处理文件的打开和关闭，使得它们使用起来很简单，例如:

>>>

```py
>>> path = pathlib.Path.cwd() / 'test.md'
>>> path.read_text()
<the contents of the test.md-file>
```

路径也可以指定为简单的文件名，在这种情况下，它们被解释为相对于当前工作目录。以下示例等同于上一个示例:

>>>

```py
>>> pathlib.Path('test.md').read_text()
<the contents of the test.md-file>
```

`.resolve()`方法将找到完整的路径。下面，我们确认当前工作目录用于简单文件名:

>>>

```py
>>> path = pathlib.Path('test.md')
>>> path.resolve()
PosixPath('/home/gahjelle/realpython/test.md')

>>> path.resolve().parent == pathlib.Path.cwd()
True

>>> path.parent == pathlib.Path.cwd()
False
```

注意，当比较路径时，比较的是它们的表示。在上例中，`path.parent`不等于`pathlib.Path.cwd()`，因为`path.parent`用`'.'`表示，而`pathlib.Path.cwd()`用`'/home/gahjelle/realpython/'`表示。

[*Remove ads*](/account/join/)

## 挑选路径的组成部分

路径的不同部分可以方便地作为属性使用。基本示例包括:

*   `.name`:没有目录的文件名
*   `.parent`:包含文件的目录，如果 path 是目录，则为父目录
*   `.stem`:不带后缀的文件名
*   `.suffix`:文件扩展名
*   `.anchor`:目录前的路径部分

下面是这些正在运行的属性:

>>>

```py
>>> path
PosixPath('/home/gahjelle/realpython/test.md')
>>> path.name
'test.md'
>>> path.stem
'test'
>>> path.suffix
'.md'
>>> path.parent
PosixPath('/home/gahjelle/realpython')
>>> path.parent.parent
PosixPath('/home/gahjelle')
>>> path.anchor
'/'
```

注意，`.parent`返回一个新的`Path`对象，而其他属性返回字符串。这意味着，例如，`.parent`可以像上一个例子那样被链接，或者甚至与`/`结合来创建全新的路径:

>>>

```py
>>> path.parent.parent / ('new' + path.suffix)
PosixPath('/home/gahjelle/new.md')
```

出色的 [Pathlib Cheatsheet](https://github.com/chris1610/pbpython/blob/master/extras/Pathlib-Cheatsheet.pdf) 提供了这些以及其他属性和方法的可视化表示。

## 移动和删除文件

通过`pathlib`，您还可以访问基本的文件系统级操作，比如移动、更新甚至删除文件。在大多数情况下，这些方法不会在信息或文件丢失之前发出警告或等待确认。使用这些方法时要小心。

要移动文件，使用`.replace()`。注意，如果目的地已经存在，`.replace()`将覆盖它。不幸的是，`pathlib`并没有明确支持文件的安全移动。为了避免可能覆盖目标路径，最简单的方法是在替换之前测试目标是否存在:

```py
if not destination.exists():
    source.replace(destination)
```

然而，这确实为可能的竞争条件敞开了大门。另一个进程可能会在执行`if`语句和`.replace()`方法之间的`destination`路径添加一个文件。如果这是一个问题，一个更安全的方法是为[独占创建](https://docs.python.org/3/library/functions.html#open)打开目标路径，并显式复制源数据:

```py
with destination.open(mode='xb') as fid:
    fid.write(source.read_bytes())
```

如果`destination`已经存在，上面的代码将引发一个`FileExistsError`。从技术上讲，这是复制一个文件。要执行移动，只需在复制完成后删除`source`(见下文)。但是要确保没有引发异常。

重命名文件时，有用的方法可能是`.with_name()`和`.with_suffix()`。它们都返回原始路径，但分别替换了名称或后缀。

例如:

>>>

```py
>>> path
PosixPath('/home/gahjelle/realpython/test001.txt')
>>> path.with_suffix('.py')
PosixPath('/home/gahjelle/realpython/test001.py')
>>> path.replace(path.with_suffix('.py'))
```

可以分别使用`.rmdir()`和`.unlink()`删除目录和文件。(还是那句话，小心！)

## 示例

在本节中，您将看到一些如何使用`pathlib`处理简单挑战的例子。

[*Remove ads*](/account/join/)

### 清点文件

有几种不同的方法来列出许多文件。最简单的是`.iterdir()`方法，它遍历给定目录中的所有文件。下面的例子结合了`.iterdir()`和`collections.Counter`类来计算当前目录中每种文件类型有多少个文件:

>>>

```py
>>> import collections
>>> collections.Counter(p.suffix for p in pathlib.Path.cwd().iterdir())
Counter({'.md': 2, '.txt': 4, '.pdf': 2, '.py': 1})
```

使用方法`.glob()`和`.rglob()`(递归 glob)可以创建更灵活的文件列表。例如，`pathlib.Path.cwd().glob('*.txt')`返回当前目录中所有带有`.txt`后缀的文件。以下仅统计以`p`开头的文件类型:

>>>

```py
>>> import collections
>>> collections.Counter(p.suffix for p in pathlib.Path.cwd().glob('*.p*'))
Counter({'.pdf': 2, '.py': 1})
```

### 显示目录树

下一个例子定义了一个函数`tree()`，它将打印一个表示文件层次结构的可视化树，以给定的目录为根。这里，我们也想列出子目录，所以我们使用了`.rglob()`方法:

```py
def tree(directory):
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')
```

注意，我们需要知道一个文件离根目录有多远。为此，我们首先使用`.relative_to()`来表示相对于根目录的路径。然后，我们计算表示中目录的数量(使用`.parts`属性)。运行时，此函数会创建如下所示的可视化树:

>>>

```py
>>> tree(pathlib.Path.cwd())
+ /home/gahjelle/realpython
 + directory_1
 + file_a.md
 + directory_2
 + file_a.md
 + file_b.pdf
 + file_c.py
 + file_1.txt
 + file_2.txt
```

> **注意:**[f 串](https://realpython.com/python-f-strings/)只在 Python 3.6 及更高版本中有效。在更老的蟒蛇身上，表达式`f'{spacer}+ {path.name}'`可以写成`'{0}+ {1}'.format(spacer, path.name)`。

### 找到最后修改的文件

`.iterdir()`、`.glob()`和`.rglob()`方法非常适合[生成器表达式](https://realpython.com/introduction-to-python-generators/)和[列表理解](https://realpython.com/list-comprehension-python/)。要在目录中找到最后修改的文件，您可以使用`.stat()`方法来获取关于底层文件的信息。例如，`.stat().st_mtime`给出了文件的最后修改时间:

>>>

```py
>>> from datetime import datetime
>>> time, file_path = max((f.stat().st_mtime, f) for f in directory.iterdir())
>>> print(datetime.fromtimestamp(time), file_path)
2018-03-23 19:23:56.977817 /home/gahjelle/realpython/test001.txt
```

您甚至可以使用类似的表达式来获取最后修改的文件内容:

>>>

```py
>>> max((f.stat().st_mtime, f) for f in directory.iterdir())[1].read_text()
<the contents of the last modified file in directory>
```

从不同的`.stat().st_`属性返回的时间戳表示自 1970 年 1 月 1 日以来的秒数。除了`datetime.fromtimestamp`之外，`time.localtime`或`time.ctime`可以用来将时间戳转换成更有用的东西。

### 创建一个唯一的文件名

最后一个例子将展示如何基于模板构造一个唯一的编号文件名。首先，为文件名指定一个模式，并为计数器留出空间。然后，检查通过连接目录和文件名(带有计数器值)创建的文件路径是否存在。如果它已经存在，增加计数器并重试:

```py
def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path

path = unique_path(pathlib.Path.cwd(), 'test{:03d}.txt')
```

如果目录中已经包含了文件`test001.txt`和`test002.txt`，上面的代码会将`path`设置为`test003.txt`。

[*Remove ads*](/account/join/)

## 操作系统差异

前面，我们注意到当我们实例化`pathlib.Path`时，或者返回一个`WindowsPath`或者一个`PosixPath`对象。对象的种类取决于您使用的操作系统。这个特性使得编写跨平台兼容的代码变得相当容易。显式地请求一个`WindowsPath`或者一个`PosixPath`是可能的，但是你只会把你的代码限制在那个系统中，没有任何好处。像这样的具体路径不能在不同的系统上使用:

>>>

```py
>>> pathlib.WindowsPath('test.md')
NotImplementedError: cannot instantiate 'WindowsPath' on your system
```

有时候，您可能需要一个无法访问底层文件系统的路径表示(在这种情况下，在非 Windows 系统上表示 Windows 路径也是有意义的，反之亦然)。这可以通过`PurePath`对象来完成。这些对象支持在路径组件的[部分中讨论的操作，但不支持访问文件系统的方法:](#picking-out-components-of-a-path)

>>>

```py
>>> path = pathlib.PureWindowsPath(r'C:\Users\gahjelle\realpython\file.txt')
>>> path.name
'file.txt'
>>> path.parent
PureWindowsPath('C:/Users/gahjelle/realpython')
>>> path.exists()
AttributeError: 'PureWindowsPath' object has no attribute 'exists'
```

可以在所有系统上直接实例化`PureWindowsPath`或者`PurePosixPath`。根据您使用的操作系统，实例化`PurePath`将返回这些对象中的一个。

## 作为适当对象的路径

在[简介](#the-problem-with-python-file-path-handling)中，我们简要地提到了路径不是字符串，`pathlib`背后的一个动机是用适当的对象来表示文件系统。事实上，`pathlib` 的[官方文档名为 *`pathlib` —面向对象文件系统路径*。在上面的例子中，](https://docs.python.org/3/library/pathlib.html)[面向对象的方法](https://realpython.com/python3-object-oriented-programming/)已经很明显了(特别是如果你将它与旧的`os.path`做事方式对比的话)。然而，让我给你留下一些其他的花絮。

与您使用的操作系统无关，路径以 Posix 样式表示，用正斜杠作为路径分隔符。在 Windows 上，您会看到类似这样的内容:

>>>

```py
>>> pathlib.Path(r'C:\Users\gahjelle\realpython\file.txt')
WindowsPath('C:/Users/gahjelle/realpython/file.txt')
```

尽管如此，当路径被转换为字符串时，它将使用本机形式，例如在 Windows 上使用反斜杠:

>>>

```py
>>> str(pathlib.Path(r'C:\Users\gahjelle\realpython\file.txt'))
'C:\\Users\\gahjelle\\realpython\\file.txt'
```

如果你正在使用一个不知道如何处理`pathlib.Path`对象的库，这是非常有用的。这在 3.6 之前的 Python 版本上是一个更大的问题。例如，在 Python 3.5 中，[`configparser`标准库](https://docs.python.org/3/library/configparser.html)只能使用字符串路径来读取文件。处理这种情况的方法是显式转换为字符串:

>>>

```py
>>> from configparser import ConfigParser
>>> path = pathlib.Path('config.txt')
>>> cfg = ConfigParser()
>>> cfg.read(path)                     # Error on Python < 3.6
TypeError: 'PosixPath' object is not iterable
>>> cfg.read(str(path))                # Works on Python >= 3.4
['config.txt']
```

在 Python 3.6 和更高版本中，如果需要进行显式转换，建议使用`os.fspath()`而不是`str()`。这稍微安全一点，因为如果你不小心试图转换一个不是[路径的对象，它会引发一个错误。](https://docs.python.org/3/library/os.html#os.PathLike)

`pathlib`库最不寻常的部分可能是使用了`/`操作符。让我们看一下它是如何实现的。这是操作符重载的一个例子:操作符的行为根据上下文而改变。你以前见过这个。想想`+`对于字符串和数字来说意味着什么。Python 通过使用*双下划线*方法(又名 *dunder* 方法)来实现操作符重载。

`/`操作符由`.__truediv__()`方法定义。事实上，如果你看一下`pathlib` 的[源代码，你会看到这样的内容:](https://github.com/python/cpython/blob/master/Lib/pathlib.py)

```py
class PurePath(object):

    def __truediv__(self, key):
        return self._make_child((key,))
```

## 结论

从 Python 3.4 开始，`pathlib`已经可以在标准库中使用了。有了`pathlib`，文件路径可以用合适的`Path`对象来表示，而不是像以前一样用普通的字符串。这些对象构成了处理文件路径的代码:

*   更容易阅读，尤其是因为`/`用于将路径连接在一起
*   更强大，大多数必需的方法和属性都可以直接在对象上使用
*   跨操作系统更加一致，因为不同系统的特性被`Path`对象隐藏了

在本教程中，您已经看到了如何创建`Path`对象、读写文件、操作路径和底层文件系统，以及如何迭代多个文件路径的一些示例。

**免费 PDF 下载:** [Python 3 备忘单](https://realpython.com/bonus/python-cheat-sheet-short/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 Python 的 pathlib 模块**](/courses/pathlib-python/)******