# Python 的 zipfile:有效地操作你的 ZIP 文件

> 原文:# t0]https://realython . com/python-zipfile/

Python 的 [`zipfile`](https://docs.python.org/3/library/zipfile.html) 是一个标准的库模块，用来操作 **ZIP 文件**。这种文件格式是归档和压缩数字数据时广泛采用的行业标准。你可以用它来打包几个相关的文件。它还允许您减少文件的大小并节省磁盘空间。最重要的是，它促进了计算机网络上的数据交换。

作为 Python 开发人员或 DevOps 工程师，知道如何使用`zipfile`模块创建、读取、写入、填充、提取和列出 ZIP 文件是一项有用的技能。

**在本教程中，您将学习如何:**

*   **用 Python 的`zipfile`从 ZIP 文件中读取、写入和提取**文件
*   使用`zipfile`读取关于 ZIP 文件内容的**元数据**
*   使用`zipfile`到**操作现有 ZIP 文件中的成员文件**
*   创建**新的 ZIP 文件**来归档和压缩文件

如果您经常处理 ZIP 文件，那么这些知识可以帮助您简化工作流程，自信地处理您的文件。

为了从本教程中获得最大收益，你应该知道[处理文件](https://realpython.com/working-with-files-in-python/)，使用 [`with`语句](https://realpython.com/python-with-statement/)，用 [`pathlib`](https://realpython.com/python-pathlib/) 处理文件系统路径，以及处理类和[面向对象编程](https://realpython.com/python3-object-oriented-programming/)的基础知识。

要获取您将用于编写本教程中的示例的文件和档案，请单击下面的链接:

**获取资料:** [单击此处获取文件和档案的副本](https://realpython.com/bonus/python-zipfile-materials/)，您将使用它们来运行本 zipfile 教程中的示例。

## ZIP 文件入门

**ZIP 文件**是当今数字世界中众所周知的流行工具。这些文件相当流行，广泛用于计算机网络(尤其是互联网)上的跨平台数据交换。

您可以使用 ZIP 文件将常规文件打包成一个归档文件，压缩数据以节省磁盘空间，分发您的数字产品，等等。在本教程中，您将学习如何使用 Python 的`zipfile`模块操作 ZIP 文件。

因为关于 ZIP 文件的术语有时会令人困惑，所以本教程将遵循以下术语约定:

| 学期 | 意义 |
| --- | --- |
| ZIP 文件、ZIP 存档或存档 | 使用 [ZIP 文件格式](https://en.wikipedia.org/wiki/ZIP_(file_format))的物理文件 |
| 文件 | 一个普通的[电脑文件](https://en.wikipedia.org/wiki/Computer_file) |
| 成员文件 | 作为现有 ZIP 文件一部分的文件 |

清楚地记住这些术语将有助于你在阅读接下来的章节时避免混淆。现在，您已经准备好继续学习如何在 Python 代码中有效地操作 ZIP 文件了！

[*Remove ads*](/account/join/)

### 什么是 ZIP 文件？

您可能已经遇到并使用过 ZIP 文件。没错，那些带`.zip`文件扩展名的到处都是！ZIP 文件，又称 **ZIP 存档**，是使用 **ZIP 文件格式**的文件。

PKWARE 是创建并首先实现这种文件格式的公司。该公司制定并维护了当前的[格式规范](https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT)，该规范公开发布，允许创建使用 ZIP 文件格式读写文件的产品、程序和进程。

ZIP 文件格式是一种跨平台、可互操作的文件存储和传输格式。它结合了[无损数据压缩](https://en.wikipedia.org/wiki/Lossless_compression)，文件管理，数据[加密](https://en.wikipedia.org/wiki/Encryption)。

数据压缩不是将归档文件视为 ZIP 文件的必要条件。因此，您可以在 ZIP 存档中压缩或解压缩成员文件。ZIP 文件格式支持几种压缩算法，尽管最常见的是 [Deflate](https://en.wikipedia.org/wiki/Deflate) 。该格式还支持用 [CRC32](https://en.wikipedia.org/wiki/Cyclic_redundancy_check) 进行信息完整性检查。

尽管有其他类似的存档格式，如 RAR 文件和 T2 文件，ZIP 文件格式已经迅速成为高效数据存储和计算机网络数据交换的通用标准。

ZIP 文件到处都是。比如[微软 Office](https://en.wikipedia.org/wiki/Microsoft_Office) 和 [Libre Office](https://en.wikipedia.org/wiki/LibreOffice) 等办公套件都依赖 ZIP 文件格式作为它们的[文档容器文件](https://www.iso.org/standard/60101.html)。这意味着`.docx`、`.xlsx`、`.pptx`、`.odt`、`.ods`、`.odp`文件实际上是包含组成每个文档的几个文件和文件夹的 ZIP 存档。其他使用 ZIP 格式的常见文件包括 [`.jar`](https://en.wikipedia.org/wiki/JAR_(file_format)) 、 [`.war`](https://en.wikipedia.org/wiki/WAR_(file_format)) 、 [`.epub`](https://en.wikipedia.org/wiki/EPUB) 文件。

你可能对 [GitHub](https://realpython.com/python-git-github-intro/) 很熟悉，它使用 [Git](https://realpython.com/advanced-git-for-pythonistas/) 为软件开发和[版本控制](https://en.wikipedia.org/wiki/Version_control)提供 web 托管。当你下载软件到你的本地电脑时，GitHub 使用 ZIP 文件打包软件项目。例如，您可以下载 ZIP 文件形式的[练习解决方案](https://github.com/realpython/python-basics-exercises/archive/refs/heads/master.zip) for [*Python 基础知识:Python 3*](https://realpython.com/products/python-basics-book/) 实用介绍一书，或者您可以下载您选择的任何其他项目。

ZIP 文件允许您将文件聚合、压缩和加密到一个可互操作的便携式容器中。您可以传输 ZIP 文件，将它们分割成段，使它们能够自解压，等等。

### 为什么使用 ZIP 文件？

对于使用计算机和数字信息的开发人员和专业人员来说，知道如何创建、读取、写入和提取 ZIP 文件是一项有用的技能。除其他好处外，ZIP 文件允许您:

*   **在不丢失信息的情况下，减小文件**的大小及其存储要求
*   **由于尺寸减小和单文件传输，提高了网络传输速度**
*   **将几个相关的文件打包在一起**到一个归档中，以便高效管理
*   **将您的代码打包到一个单独的归档文件**中，以便分发
*   **使用加密技术**保护您的数据，这是当今的普遍要求
*   **保证您信息的完整性**避免对您数据的意外和恶意更改

如果您正在寻找一种灵活、可移植且可靠的方法来归档您的数字文件，这些功能使 ZIP 文件成为您的 Python 工具箱的有用补充。

### Python 可以操作 ZIP 文件吗？

是啊！Python 有几个工具可以让你操作 ZIP 文件。其中一些工具在 Python 标准库中可用。它们包括使用特定压缩算法压缩和解压缩数据的低级库，如[`zlib`](https://docs.python.org/3/library/zlib.html#module-zlib)[`bz2`](https://docs.python.org/3/library/bz2.html#module-bz2)[`lzma`](https://docs.python.org/3/library/lzma.html#module-lzma)[其他](#using-other-libraries-to-manage-zip-files)。

Python 还提供了一个名为`zipfile`的高级模块，专门用于创建、读取、写入、提取和列出 ZIP 文件的内容。在本教程中，您将了解 Python 的`zipfile`以及如何有效地使用它。

## 用 Python 的`zipfile` 操作现有的 ZIP 文件

Python 的`zipfile`提供了方便的类和函数，允许您创建、读取、写入、提取和列出 ZIP 文件的内容。以下是`zipfile`支持的一些附加功能:

*   大于 4 GiB 的 ZIP 文件( [ZIP64 文件](https://en.wikipedia.org/wiki/ZIP_(file_format)#ZIP64))
*   数据解密
*   几种压缩算法，如 Deflate、 [Bzip2](https://en.wikipedia.org/wiki/Bzip2) 和 [LZMA](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm)
*   使用 CRC32 进行信息完整性检查

要知道`zipfile`确实有一些限制。例如，当前的数据解密功能可能相当慢，因为它使用纯 Python 代码。该模块无法处理加密 ZIP 文件的创建。最后，也不支持使用多磁盘 ZIP 文件。尽管有这些限制，`zipfile`仍然是一个伟大而有用的工具。继续阅读，探索它的能力。

[*Remove ads*](/account/join/)

### 打开 ZIP 文件进行读写

在`zipfile`模块中，你会找到 [`ZipFile`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile) 类。这个类的工作很像 Python 内置的 [`open()`](https://realpython.com/read-write-files-python/#opening-and-closing-a-file-in-python) 函数，允许你使用不同的模式打开你的 ZIP 文件。默认为读取模式(`"r"`)。您也可以使用写入(`"w"`)、追加(`"a"`)和独占(`"x"`)模式。稍后，您将了解更多关于这些内容的信息。

`ZipFile`实现了**上下文管理器协议**，这样你就可以在 [`with`语句](https://realpython.com/python-with-statement/)中使用该类。该功能允许您快速打开并处理 ZIP 文件，而不用担心在完成工作后[会关闭文件](https://realpython.com/why-close-file-python/)。

在编写任何代码之前，请确保您有一份将要使用的文件和归档的副本:

**获取资料:** [单击此处获取文件和档案的副本](https://realpython.com/bonus/python-zipfile-materials/)，您将使用它们来运行本 zipfile 教程中的示例。

为了准备好您的工作环境，将下载的资源放在您的主文件夹中名为`python-zipfile/`的目录中。将文件放在正确的位置后，移动到新创建的目录，并在那里启动一个 Python 交互式会话。

为了热身，您将从阅读名为`sample.zip`的 ZIP 文件开始。为此，您可以在阅读模式下使用`ZipFile`:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     archive.printdir()
...
File Name                                        Modified             Size
hello.txt                                 2021-09-07 19:50:10           83
lorem.md                                  2021-09-07 19:50:10         2609
realpython.md                             2021-09-07 19:50:10          428
```

`ZipFile`初始化器的第一个参数可以是一个[字符串](https://realpython.com/python-strings/),代表你需要打开的 ZIP 文件的路径。这个参数也可以接受类似文件的和类似路径的对象。在本例中，您使用基于字符串的路径。

`ZipFile`的第二个参数是一个单字母字符串，表示您将用来打开文件的模式。正如您在本节开始时了解到的，`ZipFile`可以根据您的需要接受四种可能的模式。`mode` [位置参数](https://realpython.com/defining-your-own-python-function/#positional-arguments)默认为`"r"`，如果想打开档案只读可以去掉。

在`with`语句里面，你在`archive`上调用 [`.printdir()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.printdir) 。`archive` [变量](https://realpython.com/python-variables/)现在保存了`ZipFile`本身的实例。这个函数提供了一种在屏幕上显示底层 ZIP 文件内容的快速方法。该函数的输出具有用户友好的表格格式，包含三个信息栏:

*   `File Name`
*   `Modified`
*   `Size`

如果您想在尝试打开一个有效的 ZIP 文件之前确保它，那么您可以将`ZipFile`包装在一个 [`try` … `except`](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions) 语句中，并捕捉任何 [`BadZipFile`](https://docs.python.org/3/library/zipfile.html#zipfile.BadZipFile) 异常:

>>>

```py
>>> import zipfile

>>> try:
...     with zipfile.ZipFile("sample.zip") as archive:
...         archive.printdir()
... except zipfile.BadZipFile as error:
...     print(error)
...
File Name                                        Modified             Size
hello.txt                                 2021-09-07 19:50:10           83
lorem.md                                  2021-09-07 19:50:10         2609
realpython.md                             2021-09-07 19:50:10          428

>>> try:
...     with zipfile.ZipFile("bad_sample.zip") as archive:
...         archive.printdir()
... except zipfile.BadZipFile as error:
...     print(error)
...
File is not a zip file
```

第一个例子成功地打开了`sample.zip`而没有引发`BadZipFile`异常。这是因为`sample.zip`有一个有效的 ZIP 格式。另一方面，第二个例子没有成功打开`bad_sample.zip`，因为这个文件不是一个有效的 ZIP 文件。

要检查有效的 ZIP 文件，您也可以使用 [`is_zipfile()`](https://docs.python.org/3/library/zipfile.html#zipfile.is_zipfile) 功能:

>>>

```py
>>> import zipfile

>>> if zipfile.is_zipfile("sample.zip"):
...     with zipfile.ZipFile("sample.zip", "r") as archive:
...         archive.printdir()
... else:
...     print("File is not a zip file")
...
File Name                                        Modified             Size
hello.txt                                 2021-09-07 19:50:10           83
lorem.md                                  2021-09-07 19:50:10         2609
realpython.md                             2021-09-07 19:50:10          428

>>> if zipfile.is_zipfile("bad_sample.zip"):
...     with zipfile.ZipFile("bad_sample.zip", "r") as archive:
...         archive.printdir()
... else:
...     print("File is not a zip file")
...
File is not a zip file
```

在这些例子中，您使用了一个带有`is_zipfile()`的条件语句作为条件。这个函数接受一个`filename`参数，它保存文件系统中 ZIP 文件的路径。该参数可以接受字符串、类似文件或类似路径的对象。如果`filename`是一个有效的 ZIP 文件，该函数将返回`True`。否则，它返回`False`。

现在假设您想使用`ZipFile`将`hello.txt`添加到`hello.zip`档案中。为此，您可以使用写入模式(`"w"`)。这种模式打开一个 ZIP 文件进行写入。如果目标 ZIP 文件存在，那么`"w"`模式会截断它，并写入您传入的任何新内容。

**注意:**如果你对现有文件使用`ZipFile`，那么你应该小心使用`"w"`模式。您可以截断您的 ZIP 文件并丢失所有原始内容。

如果目标 ZIP 文件不存在，那么`ZipFile`会在您关闭归档时为您创建一个:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("hello.zip", mode="w") as archive:
...     archive.write("hello.txt")
...
```

运行这段代码后，在您的`python-zipfile/`目录中会有一个`hello.zip`文件。如果您使用`.printdir()`列出文件内容，那么您会注意到`hello.txt`会出现在那里。在这个例子中，你在`ZipFile`对象上调用 [`.write()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.write) 。这种方法允许您将成员文件写入 ZIP 存档。请注意，`.write()`的参数应该是一个现有的文件。

**注意:** `ZipFile`足够聪明，当你以写模式使用该类并且目标档案不存在时，它可以创建一个新的档案。但是，如果这些目录不存在，该类不会在目标 ZIP 文件的路径中创建新目录。

这解释了为什么下面的代码不起作用:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("missing/hello.zip", mode="w") as archive:
...     archive.write("hello.txt")
...
Traceback (most recent call last):
    ...
FileNotFoundError: [Errno 2] No such file or directory: 'missing/hello.zip'
```

因为目标`hello.zip`文件路径中的`missing/`目录不存在，你得到一个 [`FileNotFoundError`](https://docs.python.org/3/library/exceptions.html#FileNotFoundError) 异常。

追加模式(`"a"`)允许您*向现有的 ZIP 文件追加*新成员文件。这种模式不会截断存档，因此其原始内容是安全的。如果目标 ZIP 文件不存在，那么`"a"`模式会为您创建一个新文件，然后将您作为参数传递的任何输入文件追加到`.write()`中。

要尝试`"a"`模式，继续将`new_hello.txt`文件添加到您新创建的`hello.zip`档案中:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("hello.zip", mode="a") as archive:
...     archive.write("new_hello.txt")
...

>>> with zipfile.ZipFile("hello.zip") as archive:
...     archive.printdir()
...
File Name                                        Modified             Size
hello.txt                                 2021-09-07 19:50:10           83
new_hello.txt                             2021-08-31 17:13:44           13
```

这里，您使用 append 模式将`new_hello.txt`添加到`hello.zip`文件中。然后运行`.printdir()`来确认这个新文件存在于 ZIP 文件中。

`ZipFile`还支持独占模式(`"x"`)。这种模式允许你*独占*创建新的 ZIP 文件，并向其中写入新的成员文件。当您想要创建一个新的 ZIP 文件而不覆盖现有文件时，您将使用独占模式。如果目标文件已经存在，那么你得到 [`FileExistsError`](https://docs.python.org/3/library/exceptions.html#FileExistsError) 。

最后，如果您使用`"w"`、`"a"`或`"x"`模式创建一个 ZIP 文件，然后关闭归档文件而不添加任何成员文件，那么`ZipFile`会创建一个具有适当 ZIP 格式的空归档文件。

[*Remove ads*](/account/join/)

### 从 ZIP 文件读取元数据

你已经将`.printdir()`付诸行动。这是一个非常有用的方法，可以用来快速列出 ZIP 文件的内容。与`.printdir()`一起，`ZipFile`类提供了几种从现有 ZIP 文件中提取元数据的简便方法。

以下是这些方法的总结:

| 方法 | 描述 |
| --- | --- |
| [T2`.getinfo(filename)`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.getinfo) | 返回一个包含由`filename`提供的成员文件信息的 [`ZipInfo`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipInfo) 对象。注意，`filename`必须保存底层 ZIP 文件中目标文件的路径。 |
| [T2`.infolist()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.infolist) | 返回一个[列表中](https://realpython.com/python-lists-tuples/)的 [`ZipInfo`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipInfo) 对象，每个成员一个文件。 |
| [T2`.namelist()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.namelist) | 返回包含基础存档中所有成员文件名称的列表。此列表中的名称是`.getinfo()`的有效参数。 |

使用这三个工具，您可以检索大量关于 ZIP 文件内容的有用信息。例如，看看下面的例子，它使用了`.getinfo()`:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     info = archive.getinfo("hello.txt")
...

>>> info.file_size
83

>>> info.compress_size
83

>>> info.filename
'hello.txt'

>>> info.date_time
(2021, 9, 7, 19, 50, 10)
```

正如您在上表中所了解到的，`.getinfo()`将一个成员文件作为参数，并返回一个带有相关信息的`ZipInfo`对象。

**注意:** `ZipInfo`不打算直接实例化。当你调用`.getinfo()`和`.infolist()`方法时，它们会自动返回`ZipInfo`对象。然而，`ZipInfo`包含了一个名为 [`.from_file()`](https://docs.python.org/3/library/zipfile.html?highlight=zipfile#zipfile.ZipInfo.from_file) 的[类方法](https://realpython.com/instance-class-and-static-methods-demystified/)，如果你需要的话，它允许你显式地实例化这个类。

`ZipInfo`对象有几个属性，允许你检索关于目标成员文件的有价值的信息。例如， [`.file_size`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipInfo.file_size) 和 [`.compress_size`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipInfo.compress_size) 分别保存原始文件和压缩文件的大小，以字节为单位。该类还有一些其他有用的属性，如 [`.filename`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipInfo.filename) 和 [`.date_time`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipInfo.date_time) ，它们返回文件名和上次修改日期。

**注意:**默认情况下，`ZipFile`不会对输入文件进行压缩以将其添加到最终的归档文件中。这就是为什么在上面的例子中，大小和压缩后的大小是相同的。在下面的[压缩文件和目录](#compressing-files-and-directories)部分，你会学到更多关于这个主题的知识。

使用`.infolist()`，您可以从给定档案中的所有文件中提取信息。下面是一个使用这种方法生成最小报告的例子，该报告包含关于您的`sample.zip`档案中所有成员文件的信息:

>>>

```py
>>> import datetime
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     for info in archive.infolist():
...         print(f"Filename: {info.filename}")
...         print(f"Modified: {datetime.datetime(*info.date_time)}")
...         print(f"Normal size: {info.file_size} bytes")
...         print(f"Compressed size: {info.compress_size} bytes")
...         print("-" * 20)
...
Filename: hello.txt
Modified: 2021-09-07 19:50:10
Normal size: 83 bytes
Compressed size: 83 bytes
--------------------
Filename: lorem.md
Modified: 2021-09-07 19:50:10
Normal size: 2609 bytes
Compressed size: 2609 bytes
--------------------
Filename: realpython.md
Modified: 2021-09-07 19:50:10
Normal size: 428 bytes
Compressed size: 428 bytes
--------------------
```

[`for`循环](https://realpython.com/python-for-loop/)遍历来自`.infolist()`的`ZipInfo`对象，检索每个成员文件的文件名、最后修改日期、正常大小和压缩大小。在本例中，您使用了 [`datetime`](https://realpython.com/python-datetime/) 以人类可读的方式格式化日期。

**注:**上例改编自 [zipfile — ZIP 存档访问](https://pymotw.com/3/zipfile/)。

如果您只需要对一个 ZIP 文件执行快速检查并列出其成员文件的名称，那么您可以使用`.namelist()`:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     for filename in archive.namelist():
...         print(filename)
...
hello.txt
lorem.md
realpython.md
```

因为这个输出中的文件名是`.getinfo()`的有效参数，所以可以结合这两种方法来只检索关于所选成员文件的信息。

例如，您可能有一个 ZIP 文件，其中包含不同类型的成员文件(`.docx`、`.xlsx`、`.txt`等等)。不需要用`.infolist()`得到完整的信息，你只需要得到关于`.docx`文件的信息。然后您可以通过扩展名过滤文件，并只对您的`.docx`文件调用`.getinfo()`。来吧，试一试！

### 读写成员文件

有时您有一个 ZIP 文件，需要读取给定成员文件的内容，而不需要提取它。为此，您可以使用 [`.read()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.read) 。这个方法获取一个成员文件的`name`并将该文件的内容作为[字节](https://realpython.com/python-data-structures/#bytes-immutable-arrays-of-single-bytes)返回:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     for line in archive.read("hello.txt").split(b"\n"):
...         print(line)
...
b'Hello, Pythonista!'
b''
b'Welcome to Real Python!'
b''
b"Ready to try Python's zipfile module?"
b''
```

要使用`.read()`，需要打开 ZIP 文件进行读取或追加。注意`.read()`以字节流的形式返回目标文件的内容。在这个例子中，您使用`.split()`将流分成行，使用[换行](https://realpython.com/python-data-types/#applying-special-meaning-to-characters)字符`"\n"`作为分隔符。因为`.split()`正在对一个字节对象进行操作，所以您需要在用作参数的字符串前添加一个前导`b`。

`ZipFile.read()`也接受名为`pwd`的第二个位置参数。此参数允许您提供读取加密文件的密码。要尝试这个特性，您可以依赖与本教程的材料一起下载的`sample_pwd.zip`文件:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample_pwd.zip", mode="r") as archive:
...     for line in archive.read("hello.txt", pwd=b"secret").split(b"\n"):
...         print(line)
...
b'Hello, Pythonista!'
b''
b'Welcome to Real Python!'
b''
b"Ready to try Python's zipfile module?"
b''

>>> with zipfile.ZipFile("sample_pwd.zip", mode="r") as archive:
...     for line in archive.read("hello.txt").split(b"\n"):
...         print(line)
...
Traceback (most recent call last):
    ...
RuntimeError: File 'hello.txt' is encrypted, password required for extraction
```

在第一个例子中，您提供密码`secret`来读取您的加密文件。`pwd`参数接受 bytes 类型的值。如果您在没有提供所需密码的情况下对一个加密文件使用`.read()`，那么您会得到一个`RuntimeError`，正如您在第二个示例中注意到的。

**注意:** Python 的`zipfile`支持解密。但是，它不支持加密 ZIP 文件的创建。这就是为什么你需要使用一个外部文件归档来加密你的文件。

一些流行的文件归档器包括 Windows 的 [7z](https://en.wikipedia.org/wiki/7-Zip) 和 [WinRAR](https://en.wikipedia.org/wiki/WinRAR) ，Linux 的 [Ark](https://en.wikipedia.org/wiki/Ark_(software)) 和 [GNOME Archive Manager](https://en.wikipedia.org/wiki/GNOME_Archive_Manager) ，macOS 的 [Archiver](https://archiverapp.com/) 。

对于大型加密 ZIP 文件，请记住，解密操作可能会非常慢，因为它是在纯 Python 中实现的。在这种情况下，考虑使用一个专门的程序来处理你的档案，而不是使用`zipfile`。

如果您经常使用加密文件，那么您可能希望避免每次调用`.read()`或另一个接受`pwd`参数的方法时都提供解密密码。如果是这种情况，可以使用 [`ZipFile.setpassword()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.setpassword) 来设置全局密码:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample_pwd.zip", mode="r") as archive:
...     archive.setpassword(b"secret")
...     for file in archive.namelist():
...         print(file)
...         print("-" * 20)
...         for line in archive.read(file).split(b"\n"):
...             print(line)
...
hello.txt
--------------------
b'Hello, Pythonista!'
b''
b'Welcome to Real Python!'
b''
b"Ready to try Python's zipfile module?"
b''
lorem.md
--------------------
b'# Lorem Ipsum'
b''
b'Lorem ipsum dolor sit amet, consectetur adipiscing elit.
 ...
```

使用`.setpassword()`，您只需提供一次密码。`ZipFile`使用该唯一密码解密所有成员文件。

相比之下，如果 ZIP 文件的各个成员文件具有不同的密码，那么您需要使用`.read()`的`pwd`参数为每个文件提供特定的密码:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample_file_pwd.zip", mode="r") as archive:
...     for line in archive.read("hello.txt", pwd=b"secret1").split(b"\n"):
...         print(line)
...
b'Hello, Pythonista!'
b''
b'Welcome to Real Python!'
b''
b"Ready to try Python's zipfile module?"
b''

>>> with zipfile.ZipFile("sample_file_pwd.zip", mode="r") as archive:
...     for line in archive.read("lorem.md", pwd=b"secret2").split(b"\n"):
...         print(line)
...
b'# Lorem Ipsum'
b''
b'Lorem ipsum dolor sit amet, consectetur adipiscing elit.
 ...
```

在这个例子中，您使用`secret1`作为密码来读取`hello.txt`，使用`secret2`来读取`lorem.md`。要考虑的最后一个细节是，当您使用`pwd`参数时，您将覆盖您可能已经用`.setpassword()`设置的任何归档级密码。

**注意:**在使用不支持的压缩方法的 ZIP 文件上调用`.read()`会引发一个 [`NotImplementedError`](https://docs.python.org/3/library/exceptions.html#NotImplementedError) 。如果所需的压缩模块在 Python 安装中不可用，也会出现错误。

如果您正在寻找一种更灵活的方式来读取成员文件，并创建和添加新的成员文件到一个档案，那么 [`ZipFile.open()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.open) 是为您准备的。像内置的`open()`函数一样，这个方法实现了上下文管理器协议，因此它支持`with`语句:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     with archive.open("hello.txt", mode="r") as hello:
...         for line in hello:
...             print(line)
...
b'Hello, Pythonista!\n'
b'\n'
b'Welcome to Real Python!\n'
b'\n'
b"Ready to try Python's zipfile module?\n"
```

在这个例子中，你打开`hello.txt`进行阅读。`.open()`的第一个参数是`name`，表示您想要打开的成员文件。第二个参数是模式，照常默认为`"r"`。`ZipFile.open()`也接受一个`pwd`参数来打开加密文件。该参数的工作原理与`.read()`中的等效`pwd`参数相同。

您也可以将`.open()`与`"w"`模式配合使用。此模式允许您创建一个新的成员文件，向其中写入内容，最后将文件追加到底层归档文件中，您应该在追加模式下打开该归档文件:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="a") as archive:
...     with archive.open("new_hello.txt", "w") as new_hello:
...         new_hello.write(b"Hello, World!")
...
13

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     archive.printdir()
...     print("------")
...     archive.read("new_hello.txt")
...
File Name                                        Modified             Size
hello.txt                                 2021-09-07 19:50:10           83
lorem.md                                  2021-09-07 19:50:10         2609
realpython.md                             2021-09-07 19:50:10          428
new_hello.txt                             1980-01-01 00:00:00           13
------
b'Hello, World!'
```

在第一段代码中，您以追加模式(`"a"`)打开`sample.zip`。然后你通过用`"w"`模式调用`.open()`来创建`new_hello.txt`。这个函数返回一个类似文件的对象，支持 [`.write()`](https://docs.python.org/3.9/library/io.html#io.BufferedIOBase.write) ，允许你向新创建的文件中写入字节。

**注意:**你需要给`.open()`提供一个不存在的文件名。如果您使用的文件名已经存在于底层归档文件中，那么您将得到一个重复的文件和一个 [`UserWarning`](https://docs.python.org/3.9/library/exceptions.html#UserWarning) 异常。

在这个例子中，你将`b'Hello, World!'`写入`new_hello.txt`。当执行流退出内部的`with`语句时，Python 将输入字节写入成员文件。当外部的`with`语句退出时，Python 将`new_hello.txt`写入底层的 ZIP 文件`sample.zip`。

第二段代码确认了`new_hello.txt`现在是`sample.zip`的成员文件。在这个例子的输出中需要注意的一个细节是，`.write()`将新添加文件的`Modified`日期设置为`1980-01-01 00:00:00`，这是一个奇怪的行为，在使用这个方法时应该记住。

[*Remove ads*](/account/join/)

### 以文本形式读取成员文件的内容

正如您在上一节中了解到的，您可以使用`.read()`和`.write()`方法来读取和写入成员文件，而不需要从包含它们的 ZIP 存档中提取它们。这两种方法都专门处理字节。

但是，当您有一个包含文本文件的 ZIP 存档时，您可能希望将它们的内容作为文本而不是字节来读取。至少有两种方法可以做到这一点。您可以使用:

1.  [T2`bytes.decode()`](https://docs.python.org/3/library/stdtypes.html#bytes.decode)
2.  [T2`io.TextIOWrapper`](https://docs.python.org/3/library/io.html#io.TextIOWrapper)

因为`ZipFile.read()`以字节的形式返回目标成员文件的内容，`.decode()`可以直接对这些字节进行操作。`.decode()`方法使用给定的[字符编码](https://en.wikipedia.org/wiki/Character_encoding)格式将`bytes`对象解码成字符串。

以下是如何使用`.decode()`从`sample.zip`档案中的`hello.txt`文件中读取文本:

>>>

```py
>>> import zipfile

>>>  with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     text = archive.read("hello.txt").decode(encoding="utf-8")
...

>>> print(text)
Hello, Pythonista!

Welcome to Real Python!

Ready to try Python's zipfile module?
```

在这个例子中，你以字节的形式读取`hello.txt`的内容。然后你调用`.decode()`将字节解码成一个字符串，使用 [UTF-8](https://en.wikipedia.org/wiki/UTF-8) 作为[编码](https://realpython.com/python-encodings-guide/)。要设置`encoding`参数，可以使用`"utf-8"`字符串。但是，您可以使用任何其他有效的编码，例如 [UTF-16](https://en.wikipedia.org/wiki/UTF-16) 或 [cp1252](https://en.wikipedia.org/wiki/Windows-1252) ，它们可以表示为不区分大小写的字符串。注意，`"utf-8"`是`encoding`参数对`.decode()`的默认值。

记住这一点很重要，您需要预先知道想要使用`.decode()`处理的任何成员文件的字符编码格式。如果您使用了错误的字符编码，那么您的代码将无法正确地将底层的字节解码成文本，并且您最终会得到大量无法辨认的字符。

从成员文件中读取文本的第二个选项是使用一个`io.TextIOWrapper`对象，它提供了一个缓冲的文本流。这次你需要使用`.open()`而不是`.read()`。下面是一个使用`io.TextIOWrapper`将`hello.txt`成员文件的内容作为文本流读取的例子:

>>>

```py
>>> import io
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     with archive.open("hello.txt", mode="r") as hello:
...         for line in io.TextIOWrapper(hello, encoding="utf-8"):
...             print(line.strip())
...
Hello, Pythonista!

Welcome to Real Python!

Ready to try Python's zipfile module?
```

在本例的内部`with`语句中，您从您的`sample.zip`档案中打开了`hello.txt`成员文件。然后将生成的类似二进制文件的对象`hello`作为参数传递给`io.TextIOWrapper`。这通过使用 UTF-8 字符编码格式解码`hello`的内容来创建一个缓冲的文本流。因此，您可以直接从目标成员文件中获得文本流。

就像使用`.encode()`一样，`io.TextIOWrapper`类接受一个`encoding`参数。您应该始终为该参数指定一个值，因为[默认文本编码](https://realpython.com/python310-new-features/#default-text-encodings)取决于运行代码的系统，并且可能不是您试图解码的文件的正确值。

### 从您的 ZIP 存档中提取成员文件

提取给定归档文件的内容是对 ZIP 文件最常见的操作之一。根据您的需要，您可能希望一次提取一个文件或一次提取所有文件。

[`ZipFile.extract()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.extract) 让你完成第一个任务。这个方法获取一个`member`文件的名称，并将其提取到一个由`path`指示的给定目录中。目的地`path`默认为当前目录:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     archive.extract("new_hello.txt", path="output_dir/")
...
'output_dir/new_hello.txt'
```

现在`new_hello.txt`将在你的`output_dir/`目录中。如果目标文件名已经存在于输出目录中，那么`.extract()`会覆盖它而不要求确认。如果输出目录不存在，那么`.extract()`会为您创建一个。注意，`.extract()`返回解压文件的路径。

成员文件的名称必须是由`.namelist()`返回的文件的全名。它也可以是一个包含文件信息的`ZipInfo`对象。

您也可以对加密文件使用`.extract()`。在这种情况下，您需要提供必需的`pwd`参数或者用`.setpassword()`设置归档级别的密码。

当从一个档案中提取所有成员文件时，可以使用 [`.extractall()`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.extractall) 。顾名思义，该方法将所有成员文件提取到目标路径，默认情况下是当前目录:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     archive.extractall("output_dir/")
...
```

运行这段代码后，`sample.zip`的所有当前内容都将在您的`output_dir/`目录中。如果您将一个不存在的目录传递给`.extractall()`，那么这个方法会自动创建这个目录。最后，如果任何成员文件已经存在于目标目录中，那么`.extractall()`将会覆盖它们而不需要你的确认，所以要小心。

如果您只需要从给定的档案中提取一些成员文件，那么您可以使用`members`参数。该参数接受一个成员文件列表，该列表应该是手头存档中整个文件列表的子集。最后，和`.extract()`一样，`.extractall()`方法也接受一个`pwd`参数来提取加密文件。

[*Remove ads*](/account/join/)

### 使用后关闭 ZIP 文件

有时，不使用`with`语句打开给定的 ZIP 文件会很方便。在这些情况下，您需要在使用后手动关闭归档，以完成任何写入操作并释放获得的资源。

为此，您可以在您的`ZipFile`对象上调用 [`.close()`](https://docs.python.org/3/library/zipfile.html?highlight=zipfile#zipfile.ZipFile.close) :

>>>

```py
>>> import zipfile

>>> archive = zipfile.ZipFile("sample.zip", mode="r")

>>> # Use archive in different parts of your code
>>> archive.printdir()
File Name                                        Modified             Size
hello.txt                                 2021-09-07 19:50:10           83
lorem.md                                  2021-09-07 19:50:10         2609
realpython.md                             2021-09-07 19:50:10          428
new_hello.txt                             1980-01-01 00:00:00           13

>>> # Close the archive when you're done
>>> archive.close()
>>> archive
<zipfile.ZipFile [closed]>
```

对`.close()`的调用会为您关闭`archive`。在退出你的程序之前，你必须呼叫`.close()`。否则，一些写操作可能无法执行。例如，如果您打开一个 ZIP 文件来追加(`"a"`)新的成员文件，那么您需要关闭归档文件来写入这些文件。

## 创建、填充和解压你自己的 ZIP 文件

到目前为止，您已经学习了如何使用现有的 ZIP 文件。您已经学会了通过使用不同的`ZipFile`模式来读取、写入和添加成员文件。您还学习了如何读取相关元数据以及如何提取给定 ZIP 文件的内容。

在这一节中，您将编写一些实用的例子，帮助您学习如何使用`zipfile`和其他 Python 工具从几个输入文件和整个目录创建 ZIP 文件。您还将学习如何使用`zipfile`进行文件压缩等等。

### 从多个常规文件创建一个 ZIP 文件

有时您需要从几个相关的文件创建一个 ZIP 存档。这样，您可以将所有文件放在一个容器中，以便通过计算机网络分发或与朋友或同事共享。为此，您可以创建一个目标文件列表，并使用`ZipFile`和一个循环将它们写入归档文件:

>>>

```py
>>> import zipfile

>>> filenames = ["hello.txt", "lorem.md", "realpython.md"]

>>> with zipfile.ZipFile("multiple_files.zip", mode="w") as archive:
...     for filename in filenames:
...         archive.write(filename)
...
```

在这里，您创建了一个`ZipFile`对象，将所需的档案名称作为它的第一个参数。`"w"`模式允许您将成员文件写入最终的 ZIP 文件。

`for`循环遍历输入文件列表，并使用`.write()`将它们写入底层 ZIP 文件。一旦执行流退出了`with`语句，`ZipFile`会自动关闭存档，为您保存更改。现在您有了一个`multiple_files.zip`档案，其中包含了您的原始文件列表中的所有文件。

### 从目录构建 ZIP 文件

将一个目录的内容捆绑到一个归档中是 ZIP 文件的另一个日常用例。Python 有几个工具可以和`zipfile`一起使用来完成这个任务。例如，您可以使用 [`pathlib`](https://realpython.com/python-pathlib/) 到[读取给定目录](https://realpython.com/get-all-files-in-directory-python/)的内容。有了这些信息，您可以使用`ZipFile`创建一个容器档案。

在`python-zipfile/`目录下，有一个名为`source_dir/`的子目录，内容如下:

```py
source_dir/
│
├── hello.txt
├── lorem.md
└── realpython.md
```

在`source_dir/`中，你只有三个常规文件。因为目录不包含子目录，所以可以使用 [`pathlib.Path.iterdir()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.iterdir) 直接迭代其内容。有了这个想法，下面是如何从`source_dir/`的内容构建一个 ZIP 文件:

>>>

```py
>>> import pathlib
>>> import zipfile

>>> directory = pathlib.Path("source_dir/")

>>> with zipfile.ZipFile("directory.zip", mode="w") as archive:
...    for file_path in directory.iterdir():
...        archive.write(file_path, arcname=file_path.name)
...

>>> with zipfile.ZipFile("directory.zip", mode="r") as archive:
...     archive.printdir()
...
File Name                                        Modified             Size
realpython.md                             2021-09-07 19:50:10          428
hello.txt                                 2021-09-07 19:50:10           83
lorem.md                                  2021-09-07 19:50:10         2609
```

在这个例子中，你从你的源目录中创建一个 [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) 对象。第一个`with`语句创建一个准备写入的`ZipFile`对象。然后对`.iterdir()`的调用返回底层目录中条目的迭代器。

因为在`source_dir/`中没有任何子目录，所以`.iterdir()`函数只产生文件。`for`循环遍历文件并将它们写入存档。

在这种情况下，您将`file_path.name`传递给`.write()`的第二个参数。这个参数被称为`arcname`,它保存了结果档案中成员文件的名称。到目前为止，您看到的所有示例都依赖于默认值`arcname`，这是您作为第一个参数传递给`.write()`的相同文件名。

如果您没有将`file_path.name`传递给`arcname`，那么您的源目录将位于您的 ZIP 文件的根目录，根据您的需要，这也可以是一个有效的结果。

现在检查工作目录中的`root_dir/`文件夹。在这种情况下，您会发现以下结构:

```py
root_dir/
│
├── sub_dir/
│   └── new_hello.txt
│
├── hello.txt
├── lorem.md
└── realpython.md
```

您有常用的文件和一个包含单个文件的子目录。如果你想创建一个具有相同内部结构的 ZIP 文件，那么你需要一个工具，让[递归地](https://realpython.com/python-recursion/)遍历[目录树中`root_dir/`下的](https://realpython.com/directory-tree-generator-python/)。

下面是如何压缩一个完整的目录树，如上图所示，使用来自`pathlib`模块的`zipfile`和`Path.rglob()`:

>>>

```py
>>> import pathlib
>>> import zipfile

>>> directory = pathlib.Path("root_dir/")

>>> with zipfile.ZipFile("directory_tree.zip", mode="w") as archive:
...     for file_path in directory.rglob("*"):
...         archive.write(
...             file_path,
...             arcname=file_path.relative_to(directory)
...         )
...

>>> with zipfile.ZipFile("directory_tree.zip", mode="r") as archive:
...     archive.printdir()
...
File Name                                        Modified             Size
sub_dir/                                  2021-09-09 20:52:14            0
realpython.md                             2021-09-07 19:50:10          428
hello.txt                                 2021-09-07 19:50:10           83
lorem.md                                  2021-09-07 19:50:10         2609
sub_dir/new_hello.txt                     2021-08-31 17:13:44           13
```

在这个例子中，您使用 [`Path.rglob()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob) 递归遍历`root_dir/`下的[目录树](https://realpython.com/directory-tree-generator-python/)。然后，将每个文件和子目录写入目标 ZIP 存档。

这一次，您使用 [`Path.relative_to()`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.is_relative_to) 获得每个文件的相对路径，然后将结果传递给第二个参数`.write()`。这样，最终得到的 ZIP 文件的内部结构与原始源目录相同。同样，如果您希望您的源目录位于 ZIP 文件的根目录，您可以去掉这个参数。

[*Remove ads*](/account/join/)

### 压缩文件和目录

如果您的文件占用了太多的磁盘空间，那么您可以考虑压缩它们。Python 的`zipfile`支持几种流行的压缩方法。但是，默认情况下，该模块不会压缩您的文件。如果您想让您的文件更小，那么您需要显式地为`ZipFile`提供一个压缩方法。

通常，您将使用术语**存储的**来指代未经压缩就写入 ZIP 文件的成员文件。这就是为什么`ZipFile`的默认压缩方法被称为 [ZIP_STORED](https://docs.python.org/3/library/zipfile.html#zipfile.ZIP_STORED) ，它实际上指的是*未压缩的*成员文件，它们被简单地存储在包含的归档文件中。

`compression`方法是`ZipFile`初始化器的第三个参数。如果要在将文件写入 ZIP 存档文件时对其进行压缩，可以将该参数设置为下列常量之一:

| 常数 | 压缩法 | 所需模块 |
| --- | --- | --- |
| `zipfile.ZIP_DEFLATED` | 紧缩 | `zlib` |
| `zipfile.ZIP_BZIP2` | Bzip2 | `bz2` |
| `zipfile.ZIP_LZMA` | 伊玛 | `lzma` |

这些是您目前可以在`ZipFile`中使用的压缩方法。不同的方法会养出一个 [`NotImplementedError`](https://docs.python.org/3/library/exceptions.html#NotImplementedError) 。从 Python 3.10 开始，`zipfile`没有额外的压缩方法。

作为附加要求，如果您选择这些方法中的一种，那么 Python 安装中必须有支持它的压缩模块。否则，你会得到一个 [`RuntimeError`](https://docs.python.org/3/library/exceptions.html#RuntimeError) 异常，你的代码就会中断。

当涉及到压缩文件时，`ZipFile`的另一个相关参数是`compresslevel`。此参数控制您使用的压缩级别。

使用 Deflate 方法，`compresslevel`可以从`0`到`9`取整数[。使用 Bzip2 方法，您可以从`1`到`9`传递整数。在这两种情况下，当压缩级别增加时，您将获得更高的压缩和更低的压缩速度。](https://realpython.com/python-numbers/)

**注意:**PNG、JPG、MP3 等二进制文件已经使用了某种压缩方式。因此，将它们添加到 ZIP 文件中可能不会使数据变得更小，因为它已经被压缩到一定程度。

现在假设您想使用 Deflate 方法归档和压缩给定目录的内容，这是 ZIP 文件中最常用的方法。为此，您可以运行以下代码:

>>>

```py
>>> import pathlib
>>> from zipfile import ZipFile, ZIP_DEFLATED

>>> directory = pathlib.Path("source_dir/")

>>> with ZipFile("comp_dir.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
...     for file_path in directory.rglob("*"):
...         archive.write(file_path, arcname=file_path.relative_to(directory))
...
```

在这个例子中，您将`9`传递给`compresslevel`以获得最大的压缩。为了提供这个参数，您使用了一个[关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-arguments)。你需要这么做，因为`compresslevel`不是`ZipFile`初始化器的第四个[位置参数](https://realpython.com/defining-your-own-python-function/#positional-arguments)。

**注意:**`ZipFile`的初始化器接受第四个参数，称为`allowZip64`。这是一个[布尔](https://realpython.com/python-boolean/)参数，告诉`ZipFile`为大于 4 GB 的文件创建扩展名为`.zip64`的 ZIP 文件。

运行这段代码后，您将在当前目录中拥有一个`comp_dir.zip`文件。如果您将该文件的大小与您原来的`sample.zip`文件的大小进行比较，那么您会注意到文件大小显著减小。

### 按顺序创建 ZIP 文件

按顺序创建 ZIP 文件可能是日常编程中的另一个常见需求。例如，您可能需要创建一个包含或不包含内容的初始 ZIP 文件，然后在新成员文件可用时立即追加它们。在这种情况下，您需要多次打开和关闭目标 ZIP 文件。

要解决这个问题，您可以在追加模式(`"a"`)中使用`ZipFile`，就像您已经做的那样。此模式允许您将新成员文件安全地附加到 ZIP 存档，而不会截断其当前内容:

>>>

```py
>>> import zipfile

>>> def append_member(zip_file, member):
...     with zipfile.ZipFile(zip_file, mode="a") as archive:
...         archive.write(member)
...

>>> def get_file_from_stream():
...     """Simulate a stream of files."""
...     for file in ["hello.txt", "lorem.md", "realpython.md"]:
...         yield file
...

>>> for filename in get_file_from_stream():
...     append_member("incremental.zip", filename)
...

>>> with zipfile.ZipFile("incremental.zip", mode="r") as archive:
...     archive.printdir()
...
File Name                                        Modified             Size
hello.txt                                 2021-09-07 19:50:10           83
lorem.md                                  2021-09-07 19:50:10         2609
realpython.md                             2021-09-07 19:50:10          428
```

在这个例子中，`append_member()`是一个[函数](https://realpython.com/defining-your-own-python-function/)，它将一个文件(`member`)附加到输入 ZIP 存档(`zip_file`)中。要执行此操作，该函数会在您每次调用它时打开和关闭目标归档。使用一个函数来执行这个任务允许您根据需要多次重用代码。

`get_file_from_stream()`函数是一个[生成器函数](https://realpython.com/introduction-to-python-generators/)，模拟要处理的文件流。同时，`for`循环使用`append_number()`将成员文件依次添加到`incremental.zip`中。如果您在运行完这段代码后检查您的工作目录，那么您会发现一个`incremental.zip`档案，包含您传递到循环中的三个文件。

[*Remove ads*](/account/join/)

### 提取文件和目录

对 ZIP 文件执行的最常见的操作之一是将它们的内容提取到文件系统中的给定目录。您已经学习了使用`.extract()`和`.extractall()`从归档中提取一个或所有文件的基础知识。

作为一个额外的例子，回到你的`sample.zip`文件。此时，归档包含四个不同类型的文件。你有两个`.txt`档和两个`.md`档。现在假设您只想提取`.md`文件。为此，您可以运行以下代码:

>>>

```py
>>> import zipfile

>>> with zipfile.ZipFile("sample.zip", mode="r") as archive:
...     for file in archive.namelist():
...         if file.endswith(".md"):
...             archive.extract(file, "output_dir/")
...
'output_dir/lorem.md'
'output_dir/realpython.md'
```

`with`语句打开`sample.zip`进行读取。该循环使用`namelist()`遍历归档中的每个文件，而条件语句检查文件名是否以扩展名`.md`结尾。如果是这样，那么使用`.extract()`将手头的文件提取到目标目录`output_dir/`。

## 从`zipfile`到探索附加类

到目前为止，你已经学习了`ZipFile`和`ZipInfo`，这是`zipfile`中可用的两个职业。这个模块还提供了另外两个类，在某些情况下会很方便。那些类是 [`zipfile.Path`](https://docs.python.org/3/library/zipfile.html#zipfile.Path) 和 [`zipfile.PyZipFile`](https://docs.python.org/3/library/zipfile.html#zipfile.PyZipFile) 。在接下来的两节中，您将学习这些类的基础知识和它们的主要特性。

### 在 ZIP 文件中查找`Path`

当你用你最喜欢的归档程序打开一个 ZIP 文件时，你会看到归档文件的内部结构。您可能在归档的根目录下有文件。您还可以拥有包含更多文件的子目录。归档文件看起来像文件系统中的一个普通目录，每个文件都位于一个特定的路径中。

`zipfile.Path`类允许您构造 path 对象来快速创建和管理给定 ZIP 文件中成员文件和目录的路径。该类有两个参数:

*   **`root`** 接受一个 ZIP 文件，作为一个`ZipFile`对象或者一个物理 ZIP 文件的基于字符串的路径。
*   **`at`** 保存着特定成员文件的位置或归档内的目录。它默认为空字符串，代表归档文件的根目录。

以你的老朋友`sample.zip`为目标，运行下面的代码:

>>>

```py
>>> import zipfile

>>> hello_txt = zipfile.Path("sample.zip", "hello.txt")

>>> hello_txt
Path('sample.zip', 'hello.txt')

>>> hello_txt.name
'hello.txt'

>>> hello_txt.is_file()
True

>>> hello_txt.exists()
True

>>> print(hello_txt.read_text())
Hello, Pythonista!

Welcome to Real Python!

Ready to try Python's zipfile module?
```

这段代码显示了`zipfile.Path`实现了几个对`pathlib.Path`对象通用的特性。你可以用`.name`得到文件的名字。可以用`.is_file()`检查路径是否指向常规文件。您可以检查给定的文件是否存在于特定的 ZIP 文件中，等等。

`Path`还提供了一个`.open()`方法来使用不同的模式打开一个成员文件。例如，下面的代码打开`hello.txt`进行阅读:

>>>

```py
>>> import zipfile

>>> hello_txt = zipfile.Path("sample.zip", "hello.txt")

>>> with hello_txt.open(mode="r") as hello:
...     for line in hello:
...         print(line)
...
Hello, Pythonista!

Welcome to Real Python!

Ready to try Python's zipfile module?
```

使用`Path`，您可以在给定的 ZIP 文件中快速创建一个指向特定成员文件的 path 对象，并使用`.open()`立即访问其内容。

就像使用`pathlib.Path`对象一样，您可以通过在`zipfile.Path`对象上调用 [`.iterdir()`](https://docs.python.org/3/library/zipfile.html#zipfile.Path.iterdir) 来列出 ZIP 文件的内容:

>>>

```py
>>> import zipfile

>>> root = zipfile.Path("sample.zip")
>>> root
Path('sample.zip', '')

>>> root.is_dir()
True

>>> list(root.iterdir())
[
 Path('sample.zip', 'hello.txt'),
 Path('sample.zip', 'lorem.md'),
 Path('sample.zip', 'realpython.md')
]
```

很明显，`zipfile.Path`提供了许多有用的特性，您可以用它们来管理 ZIP 存档中的成员文件。

[*Remove ads*](/account/join/)

### 用`PyZipFile` 构建可导入的 ZIP 文件

`zipfile`中另一个有用的类是 [`PyZipFile`](https://docs.python.org/3.9/library/zipfile.html#zipfile.PyZipFile) 。这个类非常类似于`ZipFile`，当您需要将 Python 模块和包打包成 ZIP 文件时，它尤其方便。与`ZipFile`的主要区别在于`PyZipFile`的初始化器带有一个名为`optimize`的可选参数，它允许你在归档之前通过编译成[字节码](https://docs.python.org/3/glossary.html#term-bytecode)来优化 Python 代码。

`PyZipFile`提供与`ZipFile`相同的接口，增加了 [`.writepy()`](https://docs.python.org/3/library/zipfile.html#pyzipfile-objects) 。这个方法可以将一个 Python 文件(`.py`)作为参数，并将其添加到底层的 ZIP 文件中。如果`optimize`是`-1`(默认)，那么`.py`文件会自动编译成`.pyc`文件，然后添加到目标档案中。为什么会这样？

从 2.3 版本开始，Python 解释器支持从 ZIP 文件导入 Python 代码[，这一功能被称为](https://docs.python.org/3/whatsnew/2.3.html#pep-273-importing-modules-from-zip-archives) [Zip 导入](https://realpython.com/python-zip-import/)。这个功能相当方便。它允许你创建**可导入的 ZIP 文件**来分发你的[模块和包](https://realpython.com/python-modules-packages/)作为一个单独的存档。

**注意:**还可以使用 ZIP 文件格式创建和分发 Python 可执行应用程序，也就是俗称的 Python Zip 应用程序。要了解如何创建它们，请查看 [Python 的 zipapp:构建可执行的 Zip 应用程序](https://realpython.com/python-zipapp/)。

当您需要生成可导入的 ZIP 文件时,`PyZipFile`非常有用。打包`.pyc`文件而不是`.py`文件使得导入过程更加有效，因为它跳过了编译步骤。

在`python-zipfile/`目录中，有一个包含以下内容的`hello.py`模块:

```py
"""Print a greeting message."""
# hello.py

def greet(name="World"):
    print(f"Hello, {name}! Welcome to Real Python!")
```

这段代码定义了一个名为`greet()`的函数，它将`name`作为一个参数，[将一条问候消息打印到](https://realpython.com/python-print/)屏幕上。现在假设您想要将这个模块打包成一个 ZIP 文件，以便于分发。为此，您可以运行以下代码:

>>>

```py
>>> import zipfile

>>> with zipfile.PyZipFile("hello.zip", mode="w") as zip_module:
...     zip_module.writepy("hello.py")
...

>>> with zipfile.PyZipFile("hello.zip", mode="r") as zip_module:
...     zip_module.printdir()
...
File Name                                        Modified             Size
hello.pyc                                 2021-09-13 13:25:56          311
```

在这个例子中，对`.writepy()`的调用自动将`hello.py`编译成`hello.pyc`，并存储在`hello.zip`中。当您使用`.printdir()`列出档案的内容时，这就变得很清楚了。

一旦将`hello.py`打包成一个 ZIP 文件，就可以使用 Python 的 [import](https://realpython.com/python-import/) 系统从其包含的归档文件中导入这个模块:

>>>

```py
>>> import sys

>>> # Insert the archive into sys.path
>>> sys.path.insert(0, "/home/user/python-zipfile/hello.zip")
>>> sys.path[0]
'/home/user/python-zipfile/hello.zip'

>>> # Import and use the code
>>> import hello

>>> hello.greet("Pythonista")
Hello, Pythonista! Welcome to Real Python!
```

从 ZIP 文件导入代码的第一步是使该文件在 [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path) 中可用。这个[变量](https://realpython.com/python-variables/)保存了一个字符串列表，该列表指定了 Python 对模块的**搜索路径**。要向`sys.path`添加新项目，可以使用`.insert()`。

为了让这个示例工作，您需要更改占位符路径，并将路径传递给文件系统上的`hello.zip`。一旦您的可导入 ZIP 文件出现在这个列表中，您就可以像对待常规模块一样导入代码了。

最后，考虑工作文件夹中的`hello/`子目录。它包含一个具有以下结构的小 Python 包:

```py
hello/
|
├── __init__.py
└── hello.py
```

`__init__.py`模块将`hello/`目录变成一个 Python 包。`hello.py`模块就是你在上面的例子中使用的那个。现在假设您想将这个包打包成一个 ZIP 文件。如果是这种情况，您可以执行以下操作:

>>>

```py
>>> import zipfile

>>> with zipfile.PyZipFile("hello.zip", mode="w") as zip_pkg:
...     zip_pkg.writepy("hello")
...

>>> with zipfile.PyZipFile("hello.zip", mode="r") as zip_pkg:
...     zip_pkg.printdir()
...
File Name                                        Modified             Size
hello/__init__.pyc                        2021-09-13 13:39:30          108
hello/hello.pyc                           2021-09-13 13:39:30          317
```

对`.writepy()`的调用以`hello`包为参数，在其中搜索`.py`文件，编译成`.pyc`文件，最后添加到目标 ZIP 文件`hello.zip`。同样，您可以按照之前学习的步骤从这个归档文件中导入您的代码:

>>>

```py
>>> import sys

>>> sys.path.insert(0, "/home/user/python-zipfile/hello.zip")

>>> from hello import hello

>>> hello.greet("Pythonista")
Hello, Pythonista! Welcome to Real Python!
```

因为您的代码现在在一个包中，所以您首先需要从`hello`包中导入`hello`模块。然后您可以正常访问您的`greet()`功能。

[*Remove ads*](/account/join/)

## 从命令行运行`zipfile`

Python 的`zipfile`还提供了一个最小的[命令行接口](https://realpython.com/command-line-interfaces-python-argparse/)，允许你快速访问模块的主要功能。例如，您可以使用`-l`或`--list`选项来列出现有 ZIP 文件的内容:

```py
$ python -m zipfile --list sample.zip
File Name                                         Modified             Size
hello.txt                                  2021-09-07 19:50:10           83
lorem.md                                   2021-09-07 19:50:10         2609
realpython.md                              2021-09-07 19:50:10          428
new_hello.txt                              1980-01-01 00:00:00           13
```

该命令显示的输出与对`sample.zip`档案中的`.printdir()`的等效调用相同。

现在假设您想要创建一个包含几个输入文件的新 ZIP 文件。在这种情况下，您可以使用`-c`或`--create`选项:

```py
$ python -m zipfile --create new_sample.zip hello.txt lorem.md realpython.md

$ python -m zipfile -l new_sample.zip
File Name                                         Modified             Size
hello.txt                                  2021-09-07 19:50:10           83
lorem.md                                   2021-09-07 19:50:10         2609
realpython.md                              2021-09-07 19:50:10          428
```

这个命令创建一个包含您的`hello.txt`、`lorem.md`、`realpython.md`文件的`new_sample.zip`文件。

如果您需要创建一个 ZIP 文件来归档整个目录，该怎么办？例如，您可能有自己的`source_dir/`，它包含与上面例子相同的三个文件。您可以使用以下命令从该目录创建一个 ZIP 文件:

```py
$ python -m zipfile -c source_dir.zip source_dir/

$ python -m zipfile -l source_dir.zip
File Name                                         Modified             Size
source_dir/                                2021-08-31 08:55:58            0
source_dir/hello.txt                       2021-08-31 08:55:58           83
source_dir/lorem.md                        2021-08-31 09:01:08         2609
source_dir/realpython.md                   2021-08-31 09:31:22          428
```

使用这个命令，`zipfile`将`source_dir/`放在生成的`source_dir.zip`文件的根目录下。像往常一样，您可以通过使用`-l`选项运行`zipfile`来列出归档内容。

**注意:**当您从命令行使用`zipfile`创建归档文件时，库[在归档您的文件时隐式地使用](https://github.com/python/cpython/blob/5c65834d801d6b4313eef0684a30e12c22ccfedd/Lib/zipfile.py#L2408)Deflate 压缩算法。

您还可以从命令行使用`-e`或`--extract`选项提取给定 ZIP 文件的所有内容:

```py
$ python -m zipfile --extract sample.zip sample/
```

运行这个命令后，您的工作目录中将会有一个新的`sample/`文件夹。新文件夹将包含您的`sample.zip`档案中的当前文件。

您可以从命令行使用`zipfile`的最后一个选项是`-t`或`--test`。此选项允许您测试给定文件是否是有效的 ZIP 文件。来吧，试一试！

## 使用其他库管理 ZIP 文件

Python 标准库中还有一些其他工具，可以用来在较低的级别上归档、压缩和解压缩文件。Python 的`zipfile`在内部使用了其中一些，主要用于压缩目的。以下是其中一些工具的摘要:

| 组件 | 描述 |
| --- | --- |
| [T2`zlib`](https://docs.python.org/3/library/zlib.html) | 允许使用 [zlib](https://www.zlib.net/) 库进行压缩和解压缩 |
| [T2`bz2`](https://docs.python.org/3/library/bz2.html) | 提供使用 Bzip2 压缩算法压缩和解压缩数据的接口 |
| [T2`lzma`](https://docs.python.org/3/library/lzma.html) | 提供使用 LZMA 压缩算法压缩和解压缩数据的类和函数 |

与`zipfile`不同，这些模块中的一些允许你压缩和解压缩来自内存和数据流的数据，而不是常规文件和存档。

在 Python 标准库中，您还会找到 [`tarfile`](https://docs.python.org/3/library/tarfile.html) ，它支持 [TAR](https://en.wikipedia.org/wiki/Tar_(computing)) 归档格式。还有一个模块叫做 [`gzip`](https://docs.python.org/3/library/gzip.html) ，它提供了一个压缩和解压缩数据的接口，类似于 [GNU Gzip](https://www.gnu.org/software/gzip/) 程序的做法。

例如，您可以使用`gzip`创建一个包含一些文本的压缩文件:

>>>

```py
>>> import gzip

>>> with gzip.open("hello.txt.gz", mode="wt") as gz_file:
...     gz_file.write("Hello, World!")
...
13
```

一旦运行了这段代码，在当前目录中就会有一个包含压缩版本的`hello.txt`的`hello.txt.gz`档案。在`hello.txt`里面，你会找到文本`Hello, World!`。

不使用`zipfile`创建 ZIP 文件的一种快速高级方法是使用 [`shutil`](https://docs.python.org/3/library/shutil.html) 。此模块允许您对文件和文件集合执行一些高级操作。说到[归档操作](https://docs.python.org/3/library/shutil.html#archiving-operations)，你有 [`make_archive()`](https://docs.python.org/3/library/shutil.html#shutil.make_archive) ，可以创建归档，比如 ZIP 或者 TAR 文件:

>>>

```py
>>> import shutil

>>> shutil.make_archive("shutil_sample", format="zip", root_dir="source_dir/")
'/home/user/sample.zip'
```

这段代码在您的工作目录中创建一个名为`sample.zip`的压缩文件。这个 ZIP 文件将包含输入目录`source_dir/`中的所有文件。当你需要一种快速和高级的方式在 Python 中创建你的 ZIP 文件时，`make_archive()`函数是很方便的。

[*Remove ads*](/account/join/)

## 结论

当你需要从 **ZIP 档案**中读取、写入、压缩、解压缩和提取文件时，Python 的 **`zipfile`** 是一个方便的工具。ZIP 文件格式已经成为行业标准，允许你打包或者压缩你的数字数据。

使用 ZIP 文件的好处包括将相关文件归档在一起、节省磁盘空间、便于通过计算机网络传输数据、捆绑 Python 代码以供分发等等。

**在本教程中，您学习了如何:**

*   使用 Python 的`zipfile`来**读取、写入和提取**现有的 ZIP 文件
*   用`zipfile`阅读**元数据**关于你的 ZIP 文件的内容
*   使用`zipfile`到**操作现有 ZIP 文件中的成员文件**
*   创建**您自己的 ZIP 文件**来归档和压缩您的数字数据

您还学习了如何在命令行中使用`zipfile`来列出、创建和解压缩 ZIP 文件。有了这些知识，您就可以使用 ZIP 文件格式高效地归档、压缩和操作您的数字数据了。**********