# 在 Python 中使用文件

> 原文：<https://realpython.com/working-with-files-in-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**用 Python 处理文件的实用菜谱**](/courses/practical-recipes-files/)

Python 有几个处理文件的内置模块和函数。这些功能分布在几个模块上，例如`os`、`os.path`、`shutil`和`pathlib`等等。本文集中了在 Python 中对文件执行最常见操作所需的许多函数。

**在本教程中，您将学习如何:**

*   检索文件属性
*   创建目录
*   匹配文件名中的模式
*   遍历目录树
*   创建临时文件和目录
*   删除文件和目录
*   复制、移动或重命名文件和目录
*   创建和提取 ZIP 和 TAR 归档文件
*   使用`fileinput`模块打开多个文件

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 的“with open(…) as …”模式

使用 Python 读写数据非常简单。为此，您必须首先以适当的模式打开文件。下面是如何使用 Python 的“with open(…) as …”模式打开文本文件并读取其内容的示例:

```py
with open('data.txt', 'r') as f:
    data = f.read()
```

`open()`以文件名和模式作为参数。`r`以只读模式打开文件。要将数据写入文件，请将`w`作为参数传入:

```py
with open('data.txt', 'w') as f:
    data = 'some data to be written to the file'
    f.write(data)
```

在上面的例子中，`open()`打开文件进行读取或写入，并返回一个文件句柄(在本例中为`f`),它提供了可用于读取或写入文件数据的方法。查看[在 Python 中读写文件](https://realpython.com/read-write-files-python/)和[在 Python 中使用文件 I/O](https://dbader.org/blog/python-file-io)，了解更多关于如何读写文件的信息。

[*Remove ads*](/account/join/)

## 获取目录列表

假设您当前的工作目录有一个名为`my_directory`的子目录，该子目录包含以下内容:

```py
my_directory/
|
├── sub_dir/
|   ├── bar.py
|   └── foo.py
|
├── sub_dir_b/
|   └── file4.txt
|
├── sub_dir_c/
|   ├── config.py
|   └── file5.txt
|
├── file1.py
├── file2.csv
└── file3.txt
```

内置的`os`模块有许多有用的功能，可以用来列出目录内容和过滤结果。要获得文件系统中特定目录下所有文件和文件夹的列表，在 Python 的旧版本中使用`os.listdir()`或在 Python 3.x 中使用`os.scandir()`。如果您还想获得文件和目录属性，如文件大小和修改日期，最好使用`os.scandir()`方法。

### 传统 Python 版本中的目录列表

在 Python 3 之前的 Python 版本中，`os.listdir()`是用于获取目录列表的方法:

>>>

```py
>>> import os
>>> entries = os.listdir('my_directory/')
```

`os.listdir()`返回一个 [Python 列表](https://realpython.com/python-lists-tuples/)，其中包含由 path 参数给出的目录中的文件和子目录的名称:

>>>

```py
>>> os.listdir('my_directory/')
['sub_dir_c', 'file1.py', 'sub_dir_b', 'file3.txt', 'file2.csv', 'sub_dir']
```

像那样的目录列表不容易阅读。使用循环打印出对`os.listdir()`的调用输出有助于清理:

>>>

```py
>>> entries = os.listdir('my_directory/')
>>> for entry in entries:
...     print(entry)
...
...
sub_dir_c
file1.py
sub_dir_b
file3.txt
file2.csv
sub_dir
```

### 现代 Python 版本中的目录列表

在现代版本的 Python 中，`os.listdir()`的一个替代方法是使用`os.scandir()`和`pathlib.Path()`。

`os.scandir()`是在 Python 3.5 中引入的，在 [PEP 471](https://www.python.org/dev/peps/pep-0471/) 中有记载。`os.scandir()`调用时返回迭代器，而不是列表:

>>>

```py
>>> import os
>>> entries = os.scandir('my_directory/')
>>> entries
<posix.ScandirIterator object at 0x7f5b047f3690>
```

`ScandirIterator`指向当前目录中的所有条目。您可以循环遍历迭代器的内容并打印出文件名:

```py
import os

with os.scandir('my_directory/') as entries:
    for entry in entries:
        print(entry.name)
```

这里，`os.scandir()`与`with`语句一起使用，因为它支持上下文管理器协议。使用上下文管理器关闭迭代器，并在迭代器用尽后自动释放获取的资源。结果是打印出`my_directory/`中的文件名，就像您在`os.listdir()`示例中看到的一样:

```py
sub_dir_c
file1.py
sub_dir_b
file3.txt
file2.csv
sub_dir
```

获取目录列表的另一种方法是使用`pathlib`模块:

```py
from pathlib import Path

entries = Path('my_directory/')
for entry in entries.iterdir():
    print(entry.name)
```

根据操作系统的不同，`Path`返回的对象或者是`PosixPath`或者是`WindowsPath`对象。

`pathlib.Path()`对象有一个`.iterdir()`方法，用于创建一个目录中所有文件和文件夹的[迭代器。由`.iterdir()`生成的每个条目包含关于文件或目录的信息，比如它的名称和文件属性。`pathlib`最初是在 Python 3.4 中引入的，是对 Python 的一个很好的补充，为文件系统提供了一个面向对象的接口。](https://realpython.com/get-all-files-in-directory-python/)

在上面的例子中，您调用`pathlib.Path()`并传递一个路径参数给它。接下来是对`.iterdir()`的调用，以获取`my_directory`中所有文件和目录的列表。

以一种简单的、面向对象的方式，提供了一组以路径上大多数常见操作为特色的类。使用`pathlib`比使用`os`中的函数更有效。使用`pathlib`而不是`os`的另一个好处是，它减少了操作文件系统路径所需的导入次数。更多信息，请阅读 [Python 3 的 pathlib 模块:驯服文件系统](https://realpython.com/python-pathlib/)。

运行上面的代码会产生以下结果:

```py
sub_dir_c
file1.py
sub_dir_b
file3.txt
file2.csv
sub_dir
```

使用`pathlib.Path()`或`os.scandir()`而不是`os.listdir()`是获取目录列表的首选方式，尤其是当您处理需要文件类型和文件属性信息的代码时。`pathlib.Path()`提供了很多在`os`和`shutil`中找到的文件和路径处理功能，它的方法比这些模块中的一些更有效。我们将很快讨论如何获取文件属性。

下面是目录列表函数:

| 功能 | 描述 |
| --- | --- |
| `os.listdir()` | 返回目录中所有文件和文件夹的列表 |
| `os.scandir()` | 返回目录中所有对象的迭代器，包括文件属性信息 |
| `pathlib.Path.iterdir()` | 返回目录中所有对象的迭代器，包括文件属性信息 |

这些函数返回目录中所有内容的列表，包括子目录。这可能并不总是你想要的行为。下一节将向您展示如何从目录列表中过滤结果。

[*Remove ads*](/account/join/)

### 列出目录中的所有文件

本节将向您展示如何使用`os.listdir()`、`os.scandir()`和`pathlib.Path()`打印出目录中的文件名。要从`os.listdir()`生成的目录列表中过滤出目录并仅列出文件，请使用`os.path`:

```py
import os

# List all files in a directory using os.listdir
basepath = 'my_directory/'
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        print(entry)
```

这里，对`os.listdir()`的调用返回指定路径中所有内容的列表，然后这个列表被`os.path.isfile()`过滤，只打印出文件而不是目录。这会产生以下输出:

```py
file1.py
file3.txt
file2.csv
```

列出目录中的文件的一种更简单的方法是使用`os.scandir()`或`pathlib.Path()`:

```py
import os

# List all files in a directory using scandir()
basepath = 'my_directory/'
with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_file():
            print(entry.name)
```

使用`os.scandir()`比使用`os.listdir()`有看起来更干净、更容易理解的优点，尽管它比使用`os.listdir()`多了一行代码。如果对象是文件，对`ScandirIterator`中的每一项调用`entry.is_file()`将返回`True`。打印出目录中所有文件的名称会得到以下输出:

```py
file1.py
file3.txt
file2.csv
```

下面是如何使用`pathlib.Path()`列出目录中的文件:

```py
from pathlib import Path

basepath = Path('my_directory/')
files_in_basepath = basepath.iterdir()
for item in files_in_basepath:
    if item.is_file():
        print(item.name)
```

在这里，您对由`.iterdir()`产生的每个条目调用`.is_file()`。产生的输出是相同的:

```py
file1.py
file3.txt
file2.csv
```

如果将 [`for`循环](https://realpython.com/python-for-loop/)和 [`if`语句](https://realpython.com/python-conditional-statements/)组合成一个生成器表达式，上面的代码会更简洁。Dan Bader 有一篇关于[生成器表达式](https://realpython.com/introduction-to-python-generators/)和列表理解的[优秀文章](https://dbader.org/blog/python-generator-expressions)。

修改后的版本如下所示:

```py
from pathlib import Path

# List all files in directory using pathlib
basepath = Path('my_directory/')
files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_file())
for item in files_in_basepath:
    print(item.name)
```

这与之前的示例产生了完全相同的输出。本节展示了使用`os.scandir()`和`pathlib.Path()`过滤文件或目录比结合使用`os.listdir()`和`os.path`感觉更直观，看起来更干净。

### 列出子目录

要列出子目录而不是文件，请使用下面的方法之一。下面是如何使用`os.listdir()`和`os.path()`:

```py
import os

# List all subdirectories using os.listdir
basepath = 'my_directory/'
for entry in os.listdir(basepath):
    if os.path.isdir(os.path.join(basepath, entry)):
        print(entry)
```

当您多次调用`os.path.join()`时，以这种方式操作文件系统路径会变得很麻烦。在我的计算机上运行该程序会产生以下输出:

```py
sub_dir_c
sub_dir_b
sub_dir
```

下面是如何使用`os.scandir()`:

```py
import os

# List all subdirectories using scandir()
basepath = 'my_directory/'
with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_dir():
            print(entry.name)
```

与文件列表示例一样，这里您对由`os.scandir()`返回的每个条目调用`.is_dir()`。如果条目是一个目录，`.is_dir()`返回`True`，并打印出目录名。输出与上面相同:

```py
sub_dir_c
sub_dir_b
sub_dir
```

下面是如何使用`pathlib.Path()`:

```py
from pathlib import Path

# List all subdirectory using pathlib
basepath = Path('my_directory/')
for entry in basepath.iterdir():
    if entry.is_dir():
        print(entry.name)
```

在`basepath`迭代器的每个条目上调用`.is_dir()`,检查条目是文件还是目录。如果条目是一个目录，它的名称将打印到屏幕上，并且产生的输出与上一个示例中的输出相同:

```py
sub_dir_c
sub_dir_b
sub_dir
```

[*Remove ads*](/account/join/)

## 获取文件属性

Python 使得检索文件属性(如文件大小和修改时间)变得很容易。这是通过`os.stat()`、`os.scandir()`或`pathlib.Path()`完成的。

`os.scandir()`和`pathlib.Path()`检索结合了文件属性的目录列表。这可能比使用`os.listdir()`列出文件，然后获取每个文件的文件属性信息更有效。

下面的例子显示了如何获取`my_directory/`中的文件最后被修改的时间。输出以秒为单位:

>>>

```py
>>> import os
>>> with os.scandir('my_directory/') as dir_contents:
...     for entry in dir_contents:
...         info = entry.stat()
...         print(info.st_mtime)
...
1539032199.0052035
1539032469.6324475
1538998552.2402923
1540233322.4009316
1537192240.0497339
1540266380.3434134
```

`os.scandir()`返回一个`ScandirIterator`对象。一个`ScandirIterator`对象中的每一个条目都有一个`.stat()`方法来检索关于它所指向的文件或目录的信息。`.stat()`提供文件大小和上次修改时间等信息。在上面的例子中，代码打印出了`st_mtime`属性，这是文件内容最后一次被修改的时间。

`pathlib`模块有相应的方法来检索文件信息，得到相同的结果:

>>>

```py
>>> from pathlib import Path
>>> current_dir = Path('my_directory')
>>> for path in current_dir.iterdir():
...     info = path.stat()
...     print(info.st_mtime)
...
1539032199.0052035
1539032469.6324475
1538998552.2402923
1540233322.4009316
1537192240.0497339
1540266380.3434134
```

在上面的例子中，代码遍历由`.iterdir()`返回的对象，并通过对目录列表中每个文件的`.stat()`调用来检索文件属性。`st_mtime`属性返回一个浮点值，表示从纪元开始的[秒。为了显示的目的，要转换由`st_mtime`返回的值，您可以编写一个助手函数来将秒转换成一个`datetime`对象:](https://en.wikipedia.org/wiki/Epoch_(reference_date)#Computing)

```py
from datetime import datetime
from os import scandir

def convert_date(timestamp):
    d = datetime.utcfromtimestamp(timestamp)
    formated_date = d.strftime('%d %b %Y')
    return formated_date

def get_files():
    dir_entries = scandir('my_directory/')
    for entry in dir_entries:
        if entry.is_file():
            info = entry.stat()
            print(f'{entry.name}\t Last Modified: {convert_date(info.st_mtime)}')
```

这将首先获得`my_directory`中文件及其属性的列表，然后调用`convert_date()`将每个文件的最后修改时间转换成人类可读的形式。`convert_date()`使用`.strftime()`将秒的时间转换成字符串。

传递给`.strftime()`的参数如下:

*   **`%d` :** 一月中的某一天
*   **`%b` :** 月份，缩写形式
*   **`%Y` :** 年份

这些指令一起产生如下所示的输出:

>>>

```py
>>> get_files()
file1.py        Last modified:  04 Oct 2018
file3.txt       Last modified:  17 Sep 2018
file2.txt       Last modified:  17 Sep 2018
```

将日期和时间转换成字符串的语法可能会很混乱。要了解更多信息，请查看上面的官方文档。另一个容易记住的方便参考是 http://strftime.org/的 T2。

## 制作目录

迟早，你写的程序将不得不创建目录来存储数据。`os`和`pathlib`包括创建目录的功能。我们会考虑这些:

| 功能 | 描述 |
| --- | --- |
| `os.mkdir()` | 创建一个子目录 |
| `pathlib.Path.mkdir()` | 创建单个或多个目录 |
| `os.makedirs()` | 创建多个目录，包括中间目录 |

### 创建单个目录

要创建单个目录，请将该目录的路径作为参数传递给`os.mkdir()`:

```py
import os

os.mkdir('example_directory/')
```

如果一个目录已经存在，`os.mkdir()`引发`FileExistsError`。或者，您可以使用`pathlib`创建一个目录:

```py
from pathlib import Path

p = Path('example_directory/')
p.mkdir()
```

如果路径已经存在，`mkdir()`会引发一个`FileExistsError`:

>>>

```py
>>> p.mkdir()
Traceback (most recent call last):
 File '<stdin>', line 1, in <module>
 File '/usr/lib/python3.5/pathlib.py', line 1214, in mkdir
 self._accessor.mkdir(self, mode)
 File '/usr/lib/python3.5/pathlib.py', line 371, in wrapped
 return strfunc(str(pathobj), *args)
FileExistsError: [Errno 17] File exists: '.'
[Errno 17] File exists: '.'
```

为了避免这样的错误，[在错误发生时捕获错误](https://realpython.com/python-exceptions/),并让您的用户知道:

```py
from pathlib import Path

p = Path('example_directory')
try:
    p.mkdir()
except FileExistsError as exc:
    print(exc)
```

或者，您可以通过将`exist_ok=True`参数传递给`.mkdir()`来忽略`FileExistsError`:

```py
from pathlib import Path

p = Path('example_directory')
p.mkdir(exist_ok=True)
```

如果目录已经存在，这不会引发错误。

[*Remove ads*](/account/join/)

### 创建多个目录

`os.makedirs()`类似于`os.mkdir()`。两者的区别在于,`os.makedirs()`不仅可以创建单独的目录，还可以用来创建目录树。换句话说，它可以创建任何必要的中间文件夹，以确保完整路径的存在。

`os.makedirs()`类似于在 Bash 中运行`mkdir -p`。例如，要创建一组类似于`2018/10/05`的目录，您所要做的就是以下这些:

```py
import os

os.makedirs('2018/10/05')
```

这将创建一个包含文件夹 2018、10 和 05 的嵌套目录结构:

```py
.
|
└── 2018/
    └── 10/
        └── 05/
```

`.makedirs()`用默认权限创建目录。如果您需要创建具有不同权限的目录，请调用`.makedirs()`并传递您希望创建目录的模式:

```py
import os

os.makedirs('2018/10/05', mode=0o770)
```

这将创建`2018/10/05`目录结构，并授予所有者和组用户读、写和执行权限。默认模式为`0o777`，现有父目录的文件权限位不变。关于文件权限以及如何应用模式的更多细节，[参见文档](https://docs.python.org/3/library/os.html#os.makedirs)。

运行`tree`以确认应用了正确的权限:

```py
$ tree -p -i .
.
[drwxrwx---]  2018
[drwxrwx---]  10
[drwxrwx---]  05
```

这将打印出当前目录的目录树。`tree`通常用于以树状格式列出目录的内容。向它传递`-p`和`-i`参数会在一个垂直列表中打印出目录名及其文件权限信息。`-p`打印出文件权限，`-i`让`tree`产生一个没有缩进线的垂直列表。

如您所见，所有目录都有`770`权限。创建目录的另一种方法是使用来自`pathlib.Path`的`.mkdir()`:

```py
import pathlib

p = pathlib.Path('2018/10/05')
p.mkdir(parents=True)
```

将`parents=True`传递给`Path.mkdir()`会让它创建目录`05`和任何使路径有效所需的父目录。

默认情况下，如果目标目录已经存在，`os.makedirs()`和`Path.mkdir()`会引发一个`OSError`。在调用每个函数时，通过将`exist_ok=True`作为关键字参数传递，可以覆盖这种行为(从 Python 3.2 开始)。

运行上面的代码会一次性生成如下所示的目录结构:

```py
.
|
└── 2018/
    └── 10/
        └── 05/
```

我更喜欢在创建目录时使用`pathlib`,因为我可以使用相同的函数来创建单个或嵌套的目录。

## 文件名模式匹配

使用上述方法之一获得目录中的文件列表后，您很可能想要搜索与特定模式匹配的文件。

以下是您可以使用的方法和功能:

*   `endswith()`和`startswith()`字符串方法
*   `fnmatch.fnmatch()`
*   `glob.glob()`
*   `pathlib.Path.glob()`

下面将逐一讨论。本节中的示例将在名为`some_directory`的目录上执行，该目录具有以下结构:

```py
.
|
├── sub_dir/
|   ├── file1.py
|   └── file2.py
|
├── admin.py
├── data_01_backup.txt
├── data_01.txt
├── data_02_backup.txt
├── data_02.txt
├── data_03_backup.txt
├── data_03.txt
└── tests.py
```

如果您正在使用 Bash shell，您可以使用以下命令创建上面的目录结构:

```py
$ mkdir some_directory
$ cd some_directory/
$ mkdir sub_dir
$ touch sub_dir/file1.py sub_dir/file2.py
$ touch data_{01..03}.txt data_{01..03}_backup.txt admin.py tests.py
```

这将创建`some_directory/`目录，进入该目录，然后创建`sub_dir`。下一行在`sub_dir`中创建`file1.py`和`file2.py`，最后一行使用扩展创建所有其他文件。要了解关于 shell 扩展的更多信息，请访问[这个站点](http://linuxcommand.org/lc3_lts0080.php)。

[*Remove ads*](/account/join/)

### 使用字符串方法

Python 有几个用于[修改和操作字符串](https://realpython.com/python-strings/)的内置方法。当您在文件名中搜索模式时，`.startswith()`和`.endswith()`这两种方法非常有用。为此，首先获取一个目录列表，然后遍历它:

>>>

```py
>>> import os

>>> # Get .txt files
>>> for f_name in os.listdir('some_directory'):
...     if f_name.endswith('.txt'):
...         print(f_name)
```

上面的代码找到了`some_directory/`中的所有文件，遍历它们并使用`.endswith()`打印出扩展名为`.txt`的文件名。在我的计算机上运行该程序会产生以下输出:

```py
data_01.txt
data_03.txt
data_03_backup.txt
data_02_backup.txt
data_02.txt
data_01_backup.txt
```

### 使用`fnmatch` 进行简单的文件名模式匹配

字符串方法的匹配能力有限。`fnmatch`具有更高级的模式匹配功能和方法。我们将考虑`fnmatch.fnmatch()`，一个支持使用通配符如`*`和`?`来匹配文件名的函数。例如，为了使用`fnmatch`找到一个目录中的所有`.txt`文件，您可以执行以下操作:

>>>

```py
>>> import os
>>> import fnmatch

>>> for file_name in os.listdir('some_directory/'):
...     if fnmatch.fnmatch(file_name, '*.txt'):
...         print(file_name)
```

这将遍历`some_directory`中的文件列表，并使用`.fnmatch()`对扩展名为`.txt`的文件执行通配符搜索。

### 更高级的模式匹配

让我们假设您想要找到符合特定标准的`.txt`文件。例如，您可能只对查找文件名中包含单词`data`、一组下划线之间的数字和单词`backup`的`.txt`文件感兴趣。类似于`data_01_backup`、`data_02_backup`或者`data_03_backup`的东西。

使用`fnmatch.fnmatch()`，你可以这样做:

>>>

```py
>>> for filename in os.listdir('.'):
...     if fnmatch.fnmatch(filename, 'data_*_backup.txt'):
...         print(filename)
```

这里，您只打印与`data_*_backup.txt`模式匹配的文件名。模式中的星号将匹配任何字符，因此运行该命令将找到文件名以单词`data`开头并以`backup.txt`结尾的所有文本文件，正如您从下面的输出中看到的:

```py
data_03_backup.txt
data_02_backup.txt
data_01_backup.txt
```

### 文件名模式匹配使用`glob`

模式匹配的另一个有用模块是`glob`。

`glob`模块中的`.glob()`就像`fnmatch.fnmatch()`一样工作，但与`fnmatch.fnmatch()`不同，它把以句点(`.`)开头的文件视为特殊文件。

UNIX 和相关系统将带有通配符`?`和`*`的名称模式转换成文件列表。这叫做 globbing。

例如，在 UNIX shell 中键入`mv *.py python_files/`会将扩展名为`.py`的所有文件从当前目录移动(`mv`)到目录`python_files`。`*`字符是一个通配符，表示“任意数量的字符”，而`*.py`是 glob 模式。此外壳功能在 Windows 操作系统中不可用。`glob`模块在 Python 中增加了这个功能，使得 Windows 程序能够使用这个特性。

下面是一个如何使用`glob`在当前目录中搜索所有 Python ( `.py`)源文件的例子:

>>>

```py
>>> import glob
>>> glob.glob('*.py')
['admin.py', 'tests.py']
```

`glob.glob('*.py')`在当前目录中搜索所有扩展名为`.py`的文件，并将它们作为列表返回。`glob`还支持 shell 风格的通配符来匹配模式:

>>>

```py
>>> import glob
>>> for name in glob.glob('*[0-9]*.txt'):
...     print(name)
```

这将查找文件名中包含数字的所有文本(`.txt`)文件:

```py
data_01.txt
data_03.txt
data_03_backup.txt
data_02_backup.txt
data_02.txt
data_01_backup.txt
```

`glob`也使得在子目录中递归搜索文件变得容易:

>>>

```py
>>> import glob
>>> for file in glob.iglob('**/*.py', recursive=True):
...     print(file)
```

这个例子使用`glob.iglob()`在当前目录和子目录中搜索`.py`文件。将`recursive=True`作为参数传递给`.iglob()`，使其在当前目录和任何子目录中搜索`.py`文件。`glob.iglob()`和`glob.glob()`的区别在于`.iglob()`返回的是迭代器而不是列表。

运行上面的程序会产生以下结果:

```py
admin.py
tests.py
sub_dir/file1.py
sub_dir/file2.py
```

`pathlib`包含制作灵活文件列表的类似方法。下面的例子展示了如何使用`.Path.glob()`来列出以字母`p`开头的文件类型:

>>>

```py
>>> from pathlib import Path
>>> p = Path('.')
>>> for name in p.glob('*.p*'):
...     print(name)

admin.py
scraper.py
docs.pdf
```

调用`p.glob('*.p*')`返回一个生成器对象，该对象指向当前目录中所有文件扩展名以字母`p`开头的文件。

`Path.glob()`类似于上面讨论的`os.glob()`。正如你所看到的，`pathlib`将`os`、`os.path`和`glob`模块的许多最好的特性结合到一个单独的模块中，这使得它使用起来非常有趣。

概括来说，下面是我们在本节中介绍的功能的表格:

| 功能 | 描述 |
| --- | --- |
| `startswith()` | 测试字符串是否以指定的模式开始，并返回`True`或`False` |
| `endswith()` | 测试字符串是否以指定的模式结束，并返回`True`或`False` |
| `fnmatch.fnmatch(filename, pattern)` | 测试文件名是否与模式匹配，并返回`True`或`False` |
| `glob.glob()` | 返回与模式匹配的文件名列表 |
| `pathlib.Path.glob()` | 查找路径名中的模式并返回一个生成器对象 |

[*Remove ads*](/account/join/)

## 遍历目录和处理文件

一个常见的编程任务是遍历目录树并处理树中的文件。让我们来探索如何使用内置的 Python 函数`os.walk()`来实现这一点。`os.walk()`用于通过自顶向下或自底向上遍历目录树来生成目录树中的文件名。出于本节的目的，我们将操作以下目录树:

```py
.
|
├── folder_1/
|   ├── file1.py
|   ├── file2.py
|   └── file3.py
|
├── folder_2/
|   ├── file4.py
|   ├── file5.py
|   └── file6.py
|
├── test1.txt
└── test2.txt
```

下面的例子展示了如何使用`os.walk()`列出目录树中的所有文件和目录。

`os.walk()`默认以自顶向下的方式遍历目录:

```py
# Walking a directory tree and printing the names of the directories and files
for dirpath, dirnames, files in os.walk('.'):
    print(f'Found directory: {dirpath}')
    for file_name in files:
        print(file_name)
```

`os.walk()`在循环的每次迭代中返回三个值:

1.  当前文件夹的名称

2.  当前文件夹中的文件夹列表

3.  当前文件夹中的文件列表

在每次迭代中，它打印出找到的子目录和文件的名称:

```py
Found directory: .
test1.txt
test2.txt
Found directory: ./folder_1
file1.py
file3.py
file2.py
Found directory: ./folder_2
file4.py
file5.py
file6.py
```

要以自下而上的方式遍历目录树，请向`os.walk()`传递一个`topdown=False`关键字参数:

```py
for dirpath, dirnames, files in os.walk('.', topdown=False):
    print(f'Found directory: {dirpath}')
    for file_name in files:
        print(file_name)
```

传递`topdown=False`参数将使`os.walk()`首先打印出它在*子目录*中找到的文件:

```py
Found directory: ./folder_1
file1.py
file3.py
file2.py
Found directory: ./folder_2
file4.py
file5.py
file6.py
Found directory: .
test1.txt
test2.txt
```

如您所见，该程序首先列出子目录的内容，然后列出根目录的内容。这在您想要递归删除文件和目录的情况下非常有用。您将在下面几节中学习如何做到这一点。默认情况下，`os.walk`不会进入解析到目录的符号链接。这个行为可以通过用一个`followlinks=True`参数调用它来覆盖。

## 制作临时文件和目录

Python 为创建临时文件和目录提供了一个方便的模块`tempfile`。

`tempfile`可用于在程序运行时打开数据并将其临时存储在文件或目录中。当你的程序处理完临时文件时，处理它们的删除。

下面是创建临时文件的方法:

```py
from tempfile import TemporaryFile

# Create a temporary file and write some data to it
fp = TemporaryFile('w+t')
fp.write('Hello universe!')

# Go back to the beginning and read data from file
fp.seek(0)
data = fp.read()

# Close the file, after which it will be removed
fp.close()
```

第一步是从`tempfile`模块导入`TemporaryFile`。接下来，使用`TemporaryFile()`方法创建一个类似文件的对象，方法是调用它并传递您想要打开文件的模式。这将创建并打开一个可用作临时存储区域的文件。

在上面的例子中，模式是`'w+t'`，这使得`tempfile`以写模式创建一个临时文本文件。不需要给临时文件一个文件名，因为它会在脚本运行后被销毁。

写入文件后，您可以读取它，并在完成处理后关闭它。一旦文件关闭，它将从文件系统中删除。如果您需要命名使用`tempfile`生成的临时文件，请使用`tempfile.NamedTemporaryFile()`。

使用`tempfile`创建的临时文件和目录存储在用于存储临时文件的特殊系统目录中。Python 搜索一个标准的目录列表，以找到一个用户可以在其中创建文件的目录。

在 Windows 上，这些目录依次为`C:\TEMP`、`C:\TMP`、`\TEMP`和`\TMP`。在所有其他平台上，目录依次为`/tmp`、`/var/tmp`和`/usr/tmp`。作为最后的手段，`tempfile`会在当前目录下保存临时文件和目录。

`.TemporaryFile()`也是一个上下文管理器，所以它可以与`with`语句结合使用。使用上下文管理器可以在文件被读取后自动关闭和删除文件:

```py
with TemporaryFile('w+t') as fp:
    fp.write('Hello universe!')
    fp.seek(0)
    fp.read()
# File is now closed and removed
```

这将创建一个临时文件并从中读取数据。一旦文件的内容被读取，临时文件就被关闭并从文件系统中删除。

`tempfile`也可以用来创建临时目录。让我们看看如何使用`tempfile.TemporaryDirectory()`来实现这一点:

>>>

```py
>>> import tempfile
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     print('Created temporary directory ', tmpdir)
...     os.path.exists(tmpdir)
...
Created temporary directory  /tmp/tmpoxbkrm6c
True

>>> # Directory contents have been removed
...
>>> tmpdir
'/tmp/tmpoxbkrm6c'
>>> os.path.exists(tmpdir)
False
```

调用`tempfile.TemporaryDirectory()`在文件系统中创建一个临时目录，并返回一个表示这个目录的对象。在上面的例子中，目录是使用上下文管理器创建的，目录名存储在`tmpdir`中。第三行打印出临时目录的名称，`os.path.exists(tmpdir)`确认该目录是否确实是在文件系统中创建的。

在上下文管理器脱离上下文之后，临时目录被删除，对`os.path.exists(tmpdir)`的调用返回`False`，这意味着目录被成功删除。

[*Remove ads*](/account/join/)

## 删除文件和目录

您可以使用`os`、`shutil`和`pathlib`模块中的方法删除单个文件、目录和整个目录树。以下各节介绍了如何删除不再需要的文件和目录。

### 在 Python 中删除文件

要删除单个文件，使用`pathlib.Path.unlink()`、`os.remove()`。或者`os.unlink()`。

`os.remove()`和`os.unlink()`语义相同。要使用`os.remove()`删除文件，请执行以下操作:

```py
import os

data_file = 'C:\\Users\\vuyisile\\Desktop\\Test\\data.txt'
os.remove(data_file)
```

使用`os.unlink()`删除文件类似于使用`os.remove()`删除文件:

```py
import os

data_file = 'C:\\Users\\vuyisile\\Desktop\\Test\\data.txt'
os.unlink(data_file)
```

在文件上调用`.unlink()`或`.remove()`会从文件系统中删除该文件。如果传递给这两个函数的路径指向一个目录而不是一个文件，它们将抛出一个`OSError`。为了避免这种情况，您可以检查您试图删除的实际上是一个文件，如果是就删除它，或者您可以使用异常处理来处理`OSError`:

```py
import os

data_file = 'home/data.txt'

# If the file exists, delete it
if os.path.isfile(data_file):
    os.remove(data_file)
else:
    print(f'Error: {data_file} not a valid filename')
```

`os.path.isfile()`检查`data_file`是否实际上是一个文件。如果是，则通过调用`os.remove()`将其删除。如果`data_file`指向一个文件夹，一条错误信息被打印到控制台。

下面的示例显示了如何使用异常处理来处理删除文件时的错误:

```py
import os

data_file = 'home/data.txt'

# Use exception handling
try:
    os.remove(data_file)
except OSError as e:
    print(f'Error: {data_file} : {e.strerror}')
```

上面的代码试图在检查文件类型之前先删除文件。如果`data_file`实际上不是一个文件，抛出的`OSError`在`except`子句中被处理，一条错误消息被打印到控制台。使用 [Python f-strings](https://realpython.com/python-f-strings/) 格式化打印出来的错误消息。

最后，您还可以使用`pathlib.Path.unlink()`删除文件:

```py
from pathlib import Path

data_file = Path('home/data.txt')

try:
    data_file.unlink()
except IsADirectoryError as e:
    print(f'Error: {data_file} : {e.strerror}')
```

这创建了一个名为`data_file`的`Path`对象，它指向一个文件。在`data_file`上呼叫`.remove()`会删除`home/data.txt`。如果`data_file`指向一个目录，则产生一个`IsADirectoryError`。值得注意的是，上面的 Python 程序与运行它的用户拥有相同的权限。如果用户没有删除文件的权限，就会引发一个`PermissionError`。

### 删除目录

标准库提供以下删除目录的功能:

*   `os.rmdir()`
*   `pathlib.Path.rmdir()`
*   `shutil.rmtree()`

要删除单个目录或文件夹，使用`os.rmdir()`或`pathlib.rmdir()`。这两个功能只有在你试图删除的目录为空时才有效。如果目录不为空，就会引发一个`OSError`。以下是删除文件夹的方法:

```py
import os

trash_dir = 'my_documents/bad_dir'

try:
    os.rmdir(trash_dir)
except OSError as e:
    print(f'Error: {trash_dir} : {e.strerror}')
```

这里，`trash_dir`目录通过将其路径传递给`os.rmdir()`而被删除。如果目录不为空，屏幕上会显示一条错误消息:

>>>

```py
Traceback (most recent call last):
 File '<stdin>', line 1, in <module>
OSError: [Errno 39] Directory not empty: 'my_documents/bad_dir'
```

或者，您可以使用`pathlib`删除目录:

```py
from pathlib import Path

trash_dir = Path('my_documents/bad_dir')

try:
    trash_dir.rmdir()
except OSError as e:
    print(f'Error: {trash_dir} : {e.strerror}')
```

在这里，您创建了一个指向要删除的目录的`Path`对象。在`Path`对象上调用`.rmdir()`将删除它，如果它是空的。

[*Remove ads*](/account/join/)

### 删除整个目录树

为了删除非空目录和整个目录树，Python 提供了`shutil.rmtree()`:

```py
import shutil

trash_dir = 'my_documents/bad_dir'

try:
    shutil.rmtree(trash_dir)
except OSError as e:
    print(f'Error: {trash_dir} : {e.strerror}')
```

当调用`shutil.rmtree()`时，`trash_dir`中的所有内容都会被删除。有些情况下，您可能希望递归删除空文件夹。你可以结合`os.walk()`使用上面讨论的方法之一:

```py
import os

for dirpath, dirnames, files in os.walk('.', topdown=False):
    try:
        os.rmdir(dirpath)
    except OSError as ex:
        pass
```

这将遍历目录树，并尝试删除它找到的每个目录。如果目录不为空，则引发`OSError`并跳过该目录。下表列出了本节涵盖的功能:

| 功能 | 描述 |
| --- | --- |
| `os.remove()` | 删除文件，但不删除目录 |
| `os.unlink()` | 与`os.remove()`相同，删除一个文件 |
| `pathlib.Path.unlink()` | 删除文件，不能删除目录 |
| `os.rmdir()` | 删除一个空目录 |
| `pathlib.Path.rmdir()` | 删除一个空目录 |
| `shutil.rmtree()` | 删除整个目录树，并可用于删除非空目录 |

## 复制、移动和重命名文件和目录

Python 附带了`shutil`模块。`shutil`是 shell utilities 的简称。它提供了许多对文件的高级操作，以支持文件和目录的复制、存档和删除。在本节中，您将学习如何移动和复制文件和目录。

### 用 Python 复制文件

`shutil`提供了几个复制文件的功能。最常用的功能是`shutil.copy()`和`shutil.copy2()`。要使用`shutil.copy()`将文件从一个位置复制到另一个位置，请执行以下操作:

```py
import shutil

src = 'path/to/file.txt'
dst = 'path/to/dest_dir'
shutil.copy(src, dst)
```

`shutil.copy()`相当于基于 UNIX 的系统中的`cp`命令。`shutil.copy(src, dst)`会将文件`src`复制到`dst`指定的位置。如果`dst`是一个文件，该文件的内容将被替换为`src`的内容。如果`dst`是一个目录，那么`src`将被复制到那个目录中。`shutil.copy()`仅复制文件的内容和文件的权限。文件的创建和修改时间等其他元数据不会被保留。

要在复制时保留所有文件元数据，请使用`shutil.copy2()`:

```py
import shutil

src = 'path/to/file.txt'
dst = 'path/to/dest_dir'
shutil.copy2(src, dst)
```

使用`.copy2()`保存文件的细节，比如最后访问时间、许可位、最后修改时间和标志。

### 复制目录

虽然`shutil.copy()`只复制单个文件，但是`shutil.copytree()`会复制整个目录以及其中包含的所有内容。`shutil.copytree(src, dest)`有两个参数:一个源目录和文件和文件夹将被复制到的目标目录。

以下是如何将一个文件夹的内容复制到不同位置的示例:

>>>

```py
>>> import shutil
>>> shutil.copytree('data_1', 'data1_backup')
'data1_backup'
```

在这个例子中，`.copytree()`将`data_1`的内容复制到一个新位置`data1_backup`，并返回目标目录。目标目录不能已经存在。它将被创建并丢失父目录。`shutil.copytree()`是备份文件的好方法。

[*Remove ads*](/account/join/)

### 移动文件和目录

要将文件或目录移动到另一个位置，使用`shutil.move(src, dst)`。

`src`是要移动的文件或目录，`dst`是目的地:

>>>

```py
>>> import shutil
>>> shutil.move('dir_1/', 'backup/')
'backup'
```

如果`backup/`存在，则`shutil.move('dir_1/', 'backup/')`将`dir_1/`移动到`backup/`。如果`backup/`不存在，`dir_1/`将被重命名为`backup`。

### 重命名文件和目录

Python 包含用于重命名文件和目录的`os.rename(src, dst)`:

>>>

```py
>>> os.rename('first.zip', 'first_01.zip')
```

上面的行将`first.zip`重命名为`first_01.zip`。如果目标路径指向一个目录，它将引发一个`OSError`。

重命名文件或目录的另一种方法是使用`pathlib`模块中的`rename()`:

>>>

```py
>>> from pathlib import Path
>>> data_file = Path('data_01.txt')
>>> data_file.rename('data.txt')
```

要使用`pathlib`重命名文件，首先要创建一个`pathlib.Path()`对象，其中包含要替换的文件的路径。下一步是调用 path 对象上的`rename()`,并为要重命名的文件或目录传递一个新文件名。

## 存档

归档是将几个文件打包成一个文件的便捷方式。两种最常见的归档类型是 ZIP 和 TAR。您编写的 Python 程序可以创建、读取和提取档案中的数据。在本节中，您将学习如何读写这两种归档格式。

### 读取 ZIP 文件

[`zipfile`](https://realpython.com/python-zipfile/) 模块是一个低级模块，是 Python 标准库的一部分。`zipfile`具有打开和解压 ZIP 文件的功能。要读取一个 ZIP 文件的内容，首先要做的是创建一个`ZipFile`对象。`ZipFile`对象类似于使用`open()`创建的文件对象。`ZipFile`也是一个上下文管理器，因此支持`with`语句:

```py
import zipfile

with zipfile.ZipFile('data.zip', 'r') as zipobj:
```

在这里，您创建了一个`ZipFile`对象，传入以读取模式打开的 ZIP 文件的名称。打开一个 ZIP 文件后，可以通过`zipfile`模块提供的函数访问关于档案的信息。上例中的`data.zip`归档文件是从名为`data`的目录中创建的，该目录总共包含 5 个文件和 1 个子目录:

```py
.
|
├── sub_dir/
|   ├── bar.py
|   └── foo.py
|
├── file1.py
├── file2.py
└── file3.py
```

要获得归档中的文件列表，请对`ZipFile`对象调用`namelist()`:

```py
import zipfile

with zipfile.ZipFile('data.zip', 'r') as zipobj:
    zipobj.namelist()
```

这会产生一个列表:

```py
['file1.py', 'file2.py', 'file3.py', 'sub_dir/', 'sub_dir/bar.py', 'sub_dir/foo.py']
```

`.namelist()`返回档案中文件和目录的名称列表。要检索归档中文件的信息，请使用`.getinfo()`:

```py
import zipfile

with zipfile.ZipFile('data.zip', 'r') as zipobj:
    bar_info = zipobj.getinfo('sub_dir/bar.py')
    bar_info.file_size
```

以下是输出结果:

```py
15277
```

`.getinfo()`返回一个`ZipInfo`对象，该对象存储关于档案中一个成员的信息。要获得档案中某个文件的信息，可以将其路径作为参数传递给`.getinfo()`。使用`getinfo()`，您可以检索关于归档成员的信息，比如文件的最后修改日期、压缩大小和完整文件名。访问`.file_size`以字节为单位获取文件的原始大小。

以下示例显示了如何在 Python REPL 中检索有关归档文件的更多详细信息。假设`zipfile`模块已经被导入，并且`bar_info`是您在前面的例子中创建的同一个对象:

>>>

```py
>>> bar_info.date_time
(2018, 10, 7, 23, 30, 10)
>>> bar_info.compress_size
2856
>>> bar_info.filename
'sub_dir/bar.py'
```

`bar_info`包含关于`bar.py`的细节，比如压缩时的大小和完整路径。

第一行显示了如何检索文件的最后修改日期。下一行显示了如何获得压缩后文件的大小。最后一行显示了归档文件中`bar.py`的完整路径。

`ZipFile`支持上下文管理器协议，这就是为什么您可以将它与`with`语句一起使用。这样做可以在你完成后自动关闭`ZipFile`对象。试图从关闭的`ZipFile`对象中打开或提取文件将导致错误。

[*Remove ads*](/account/join/)

### 解压压缩文件

`zipfile`模块允许您通过`.extract()`和`.extractall()`从 ZIP 存档中提取一个或多个文件。

默认情况下，这些方法将文件提取到当前目录。它们都带有一个可选的`path`参数，允许您指定一个不同的目录来提取文件。如果该目录不存在，则会自动创建。要从归档中提取文件，请执行以下操作:

>>>

```py
>>> import zipfile
>>> import os

>>> os.listdir('.')
['data.zip']

>>> data_zip = zipfile.ZipFile('data.zip', 'r')

>>> # Extract a single file to current directory
>>> data_zip.extract('file1.py')
'/home/terra/test/dir1/zip_extract/file1.py'

>>> os.listdir('.')
['file1.py', 'data.zip']

>>> # Extract all files into a different directory
>>> data_zip.extractall(path='extract_dir/')

>>> os.listdir('.')
['file1.py', 'extract_dir', 'data.zip']

>>> os.listdir('extract_dir')
['file1.py', 'file3.py', 'file2.py', 'sub_dir']

>>> data_zip.close()
```

第三行代码是对`os.listdir()`的调用，显示当前目录只有一个文件`data.zip`。

接下来，在读取模式下打开`data.zip`，并调用`.extract()`从中提取`file1.py`。`.extract()`返回提取文件的完整文件路径。由于没有指定路径，`.extract()`将`file1.py`提取到当前目录。

下一行打印一个目录列表，显示当前目录除了原始归档文件之外，还包括提取的文件。之后的一行显示了如何将整个归档文件提取到`zip_extract`目录中。`.extractall()`创建`extract_dir`并将`data.zip`的内容提取到其中。最后一行关闭 ZIP 存档。

### 从受密码保护的档案中提取数据

`zipfile`支持提取密码保护的拉链。要提取受密码保护的 ZIP 文件，请将密码作为参数传递给`.extract()`或`.extractall()`方法:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile('secret.zip', 'r') as pwd_zip:
...     # Extract from a password protected archive
...     pwd_zip.extractall(path='extract_dir', pwd='Quish3@o')
```

这将在读取模式下打开`secret.zip`档案。向`.extractall()`提供密码，并将档案内容提取到`extract_dir`。由于使用了`with`语句，在提取完成后，归档会自动关闭。

### 创建新的 ZIP 存档文件

要创建一个新的 ZIP 存档，您需要以写模式(`w`)打开一个`ZipFile`对象，并添加您想要存档的文件:

>>>

```py
>>> import zipfile

>>> file_list = ['file1.py', 'sub_dir/', 'sub_dir/bar.py', 'sub_dir/foo.py']
>>> with zipfile.ZipFile('new.zip', 'w') as new_zip:
...     for name in file_list:
...         new_zip.write(name)
```

在本例中，`new_zip`以写模式打开，`file_list`中的每个文件都被添加到归档文件中。当`with`语句组完成后，`new_zip`关闭。以写入模式打开 ZIP 文件会删除归档文件的内容，并创建一个新的归档文件。

要将文件添加到现有档案中，在追加模式下打开一个`ZipFile`对象，然后添加文件:

>>>

```py
>>> # Open a ZipFile object in append mode
>>> with zipfile.ZipFile('new.zip', 'a') as new_zip:
...     new_zip.write('data.txt')
...     new_zip.write('latin.txt')
```

在这里，您在 append 模式下打开您在前一个例子中创建的`new.zip`档案。在追加模式下打开`ZipFile`对象允许您在不删除当前内容的情况下向 ZIP 文件添加新文件。在将文件添加到 ZIP 文件后，`with`语句脱离上下文并关闭 ZIP 文件。

### 打开 TAR 档案

TAR 文件是像 ZIP 一样的未压缩文件存档。它们可以使用 gzip、bzip2 和 lzma 压缩方法进行压缩。`TarFile`类允许读写 TAR 文档。

执行此操作以从归档中读取:

```py
import tarfile

with tarfile.open('example.tar', 'r') as tar_file:
    print(tar_file.getnames())
```

对象像大多数类似文件的对象一样打开。他们有一个`open()`函数，它采用一种模式来决定文件如何打开。

使用`'r'`、`'w'`或`'a'`模式分别打开一个未压缩的 TAR 文件进行读取、写入和附加。要打开压缩的 TAR 文件，向`tarfile.open()`传递一个模式参数，格式为`filemode[:compression]`。下表列出了打开 TAR 文件的可能模式:

| 方式 | 行动 |
| --- | --- |
| `r` | 使用透明压缩打开档案进行读取 |
| `r:gz` | 使用 gzip 压缩打开存档文件进行阅读 |
| `r:bz2` | 使用 bzip2 压缩打开存档文件进行阅读 |
| `r:xz` | 使用 lzma 压缩打开档案进行读取 |
| `w` | 打开归档文件进行未压缩的写入 |
| `w:gz` | 打开归档文件进行 gzip 压缩写入 |
| `w:xz` | 为 lzma 压缩写打开存档 |
| `a` | 打开归档文件进行无压缩附加 |

`.open()`默认为`'r'`模式。要读取一个未压缩的 TAR 文件并检索其中的文件名，使用`.getnames()`:

>>>

```py
>>> import tarfile

>>> tar = tarfile.open('example.tar', mode='r')
>>> tar.getnames()
['CONTRIBUTING.rst', 'README.md', 'app.py']
```

这将返回一个包含归档内容名称的列表。

**注意:**为了向您展示如何使用不同的`tarfile`对象方法，示例中的 TAR 文件是在交互式 REPL 会话中手动打开和关闭的。

通过这种方式与 TAR 文件交互，您可以看到运行每个命令的输出。通常，您会希望使用上下文管理器来打开类似文件的对象。

可以使用特殊属性访问归档中每个条目的元数据:

>>>

```py
>>> for entry in tar.getmembers():
...     print(entry.name)
...     print(' Modified:', time.ctime(entry.mtime))
...     print(' Size    :', entry.size, 'bytes')
...     print()
CONTRIBUTING.rst
 Modified: Sat Nov  1 09:09:51 2018
 Size    : 402 bytes

README.md
 Modified: Sat Nov  3 07:29:40 2018
 Size    : 5426 bytes

app.py
 Modified: Sat Nov  3 07:29:13 2018
 Size    : 6218 bytes
```

在这个例子中，您遍历由`.getmembers()`返回的文件列表，并打印出每个文件的属性。由`.getmembers()`返回的对象具有可以以编程方式访问的属性，比如档案中每个文件的名称、大小和最后修改时间。在读取或写入归档文件后，必须将其关闭以释放系统资源。

### 从 TAR 存档中提取文件

在本节中，您将学习如何使用以下方法从 TAR 归档中提取文件:

*   `.extract()`
*   `.extractfile()`
*   `.extractall()`

要从 TAR 归档文件中提取一个文件，使用`extract()`，传入文件名:

>>>

```py
>>> tar.extract('README.md')
>>> os.listdir('.')
['README.md', 'example.tar']
```

`README.md`文件从归档文件中提取到文件系统中。调用`os.listdir()`确认`README.md`文件被成功提取到当前目录。要从归档中解压或提取所有内容，请使用`.extractall()`:

>>>

```py
>>> tar.extractall(path="extracted/")
```

`.extractall()`有一个可选的`path`参数来指定提取的文件应该放在哪里。在这里，归档文件被解压到`extracted`目录中。以下命令显示归档文件已成功提取:

```py
$ ls
example.tar  extracted  README.md

$ tree
.
├── example.tar
├── extracted
|   ├── app.py
|   ├── CONTRIBUTING.rst
|   └── README.md
└── README.md

1 directory, 5 files

$ ls extracted/
app.py  CONTRIBUTING.rst  README.md
```

要提取一个文件对象来读或写，使用`.extractfile()`，它接受一个文件名或要提取的`TarInfo`对象作为参数。`.extractfile()`返回一个可以读取和使用的类似文件的对象:

>>>

```py
>>> f = tar.extractfile('app.py')
>>> f.read()
>>> tar.close()
```

打开的档案在被读取或写入后应该关闭。要关闭一个归档文件，调用归档文件句柄上的`.close()`，或者在创建`tarfile`对象时使用`with`语句，以便在完成后自动关闭归档文件。这将释放系统资源，并将您对归档文件所做的任何更改写入文件系统。

### 创建新的 TAR 归档文件

你可以这样做:

>>>

```py
>>> import tarfile

>>> file_list = ['app.py', 'config.py', 'CONTRIBUTORS.md', 'tests.py']
>>> with tarfile.open('packages.tar', mode='w') as tar:
...     for file in file_list:
...         tar.add(file)

>>> # Read the contents of the newly created archive
>>> with tarfile.open('package.tar', mode='r') as t:
...     for member in t.getmembers():
...         print(member.name)
app.py
config.py
CONTRIBUTORS.md
tests.py
```

首先，列出要添加到归档中的文件列表，这样就不必手动添加每个文件。

下一行使用`with`上下文管理器以写模式打开一个名为`packages.tar`的新档案。以写模式(`'w'`)打开一个归档文件，使您能够向归档文件写入新文件。归档中的任何现有文件都将被删除，并创建一个新的归档。

在创建并填充归档文件后，`with`上下文管理器自动关闭它并将其保存到文件系统中。最后三行打开您刚刚创建的归档文件，并打印出其中包含的文件名。

要向现有档案添加新文件，请在追加模式(`'a'`)下打开档案:

>>>

```py
>>> with tarfile.open('package.tar', mode='a') as tar:
...     tar.add('foo.bar')

>>> with tarfile.open('package.tar', mode='r') as tar:
...     for member in tar.getmembers():
...         print(member.name)
app.py
config.py
CONTRIBUTORS.md
tests.py
foo.bar
```

在追加模式下打开归档文件允许您在不删除已有文件的情况下向其中添加新文件。

### 使用压缩档案

`tarfile`还可以读写使用 gzip、bzip2 和 lzma 压缩的 TAR 文件。要读取或写入压缩的归档文件，使用`tarfile.open()`，为压缩类型传入适当的模式。

例如，要读取或写入使用 gzip 压缩的 TAR 归档文件，分别使用`'r:gz'`或`'w:gz'`模式:

>>>

```py
>>> files = ['app.py', 'config.py', 'tests.py']
>>> with tarfile.open('packages.tar.gz', mode='w:gz') as tar:
...     tar.add('app.py')
...     tar.add('config.py')
...     tar.add('tests.py')

>>> with tarfile.open('packages.tar.gz', mode='r:gz') as t:
...     for member in t.getmembers():
...         print(member.name)
app.py
config.py
tests.py
```

`'w:gz'`模式打开存档进行 gzip 压缩写入，而`'r:gz'`打开存档进行 gzip 压缩读取。无法在附加模式下打开压缩档案。要将文件添加到压缩的归档文件中，您必须创建一个新的归档文件。

## 创建档案的简单方法

Python 标准库还支持使用`shutil`模块中的高级方法创建 TAR 和 ZIP 归档。`shutil`中的归档工具允许您创建、读取和提取 ZIP 和 TAR 归档文件。这些实用程序依赖于较低级别的`tarfile`和`zipfile`模块。

**使用`shutil.make_archive()`处理档案**

`shutil.make_archive()`至少有两个参数:归档文件的名称和归档文件的格式。

默认情况下，它将当前目录中的所有文件压缩成在`format`参数中指定的存档格式。您可以传入一个可选的`root_dir`参数来压缩不同目录中的文件。`.make_archive()`支持`zip`、`tar`、`bztar`和`gztar`存档格式。

这就是如何使用`shutil`创建 TAR 归档文件:

```py
import shutil

# shutil.make_archive(base_name, format, root_dir)
shutil.make_archive('data/backup', 'tar', 'data/')
```

这将复制`data/`中的所有内容，并在文件系统中创建一个名为`backup.tar`的归档文件，并返回其名称。要提取存档文件，请调用`.unpack_archive()`:

```py
shutil.unpack_archive('backup.tar', 'extract_dir/')
```

调用`.unpack_archive()`并传入一个档案名称和目标目录，将`backup.tar`的内容提取到`extract_dir/`中。可以用同样的方式创建和解压缩 ZIP 存档。

## 读取多个文件

Python 支持通过`fileinput`模块从多个输入流或文件列表中读取数据。这个模块允许你快速简单地循环一个或多个文本文件的内容。下面是使用`fileinput`的典型方式:

```py
import fileinput
for line in fileinput.input()
    process(line)
```

默认情况下，`fileinput`从传递给`sys.argv`的[命令行参数](https://realpython.com/python-command-line-arguments/)中获取输入。

**使用`fileinput`循环多个文件**

让我们使用`fileinput`构建一个普通 UNIX 实用程序`cat`的原始版本。`cat`实用程序按顺序读取文件，将它们写入标准输出。当在命令行参数中给出多个文件时，`cat`将连接文本文件并在终端中显示结果:

```py
# File: fileinput-example.py
import fileinput
import sys

files = fileinput.input()
for line in files:
    if fileinput.isfirstline():
        print(f'\n--- Reading {fileinput.filename()} ---')
    print(' -> ' + line, end='')
print()
```

对我当前目录中的两个文本文件运行此命令会产生以下输出:

```py
$ python3 fileinput-example.py bacon.txt cupcake.txt
--- Reading bacon.txt ---
 -> Spicy jalapeno bacon ipsum dolor amet in in aute est qui enim aliquip,
 -> irure cillum drumstick elit.
 -> Doner jowl shank ea exercitation landjaeger incididunt ut porchetta.
 -> Tenderloin bacon aliquip cupidatat chicken chuck quis anim et swine.
 -> Tri-tip doner kevin cillum ham veniam cow hamburger.
 -> Turkey pork loin cupidatat filet mignon capicola brisket cupim ad in.
 -> Ball tip dolor do magna laboris nisi pancetta nostrud doner.

--- Reading cupcake.txt ---
 -> Cupcake ipsum dolor sit amet candy I love cheesecake fruitcake.
 -> Topping muffin cotton candy.
 -> Gummies macaroon jujubes jelly beans marzipan.
```

`fileinput`允许您检索每一行的更多信息，例如它是否是第一行(`.isfirstline()`)、行号(`.lineno()`)和文件名(`.filename()`)。你可以在这里了解更多关于[的信息。](https://docs.python.org/3/library/fileinput.html)

## 结论

您现在知道如何使用 Python 对文件和文件组执行最常见的操作。您已经了解了用于读取、查找和操作它们的不同内置模块。

现在，您已经准备好使用 Python 来:

*   获取目录内容和文件属性
*   创建目录和目录树
*   在文件名中查找模式
*   创建临时文件和目录
*   移动、重命名、复制和删除文件或目录
*   从不同类型的档案中读取和提取数据
*   使用`fileinput`同时读取多个文件

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**用 Python 处理文件的实用菜谱**](/courses/practical-recipes-files/)************