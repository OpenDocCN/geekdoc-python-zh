# Python mmap:通过内存映射改进了文件 I/O

> 原文：<https://realpython.com/python-mmap/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python mmap:用内存映射做文件 I/O**](/courses/python-mmap-io/)

Python 的[禅](https://www.python.org/dev/peps/pep-0020/)有很多智慧可以提供。一个特别有用的想法是“应该有一个——最好只有一个——显而易见的方法去做。”然而，用 Python 做大多数事情有多种方法，而且通常都有很好的理由。比如在 Python 中有[多种方式读取一个文件，包括很少使用的`mmap`模块。](https://realpython.com/read-write-files-python/)

Python 的`mmap`提供了内存映射的文件输入和输出(I/O)。它允许你利用底层操作系统的功能来读取文件，就好像它们是一个大的[字符串](https://realpython.com/python-strings/)或[数组](https://dbader.org/blog/python-arrays)。这可以显著提高需要大量文件 I/O 的代码的性能。

在本教程中，您将学习:

*   电脑内存有哪些种类
*   用 **`mmap`** 可以解决什么问题
*   如何使用内存映射来**更快地读取大文件**
*   如何改变文件的**部分而不重写整个文件**
*   如何使用`mmap`到**在多个进程间共享信息**

**免费下载:** [从 CPython Internals:您的 Python 3 解释器指南](https://realpython.com/bonus/cpython-internals-sample/)获得一个示例章节，向您展示如何解锁 Python 语言的内部工作机制，从源代码编译 Python 解释器，并参与 CPython 的开发。

## 了解计算机内存

**内存映射**是一种使用低级操作系统 API 将文件直接加载到计算机内存中的技术。它可以显著提高程序中的文件 I/O 性能。为了更好地理解内存映射如何提高性能，以及如何以及何时可以使用`mmap`模块来利用这些性能优势，首先学习一点关于计算机内存的知识是很有用的。

[计算机内存](https://en.wikipedia.org/wiki/Computer_memory)是一个大而复杂的话题，但是本教程只关注你需要知道的如何有效地使用`mmap`模块。出于本教程的目的，术语**存储器**指的是[随机存取存储器](https://en.wikipedia.org/wiki/Random-access_memory)，或 RAM。

有几种类型的计算机内存:

1.  身体的
2.  虚拟的
3.  共享的

当您使用内存映射时，每种类型的内存都会发挥作用，所以让我们从较高的层次来回顾一下每种类型的内存。

[*Remove ads*](/account/join/)

### 物理内存

物理内存是理解起来最简单的一种内存，因为它通常是与你的电脑相关的市场营销的一部分。(你可能还记得，当你买电脑时，它宣传的是 8g 内存。)物理内存通常位于连接到计算机主板的卡上。

物理内存是程序运行时可用的易失性内存总量。不应将物理内存与存储混淆，如硬盘或固态硬盘。

### 虚拟内存

[虚拟内存](https://en.wikipedia.org/wiki/Virtual_memory)是一种处理[内存管理](https://realpython.com/python-memory-management/)的方式。操作系统使用虚拟内存使你看起来比实际拥有的内存多，这样你就不用担心在任何给定的时间有多少内存可供你的程序使用。在幕后，您的操作系统使用部分非易失性存储(如固态硬盘)来模拟额外的 RAM。

为此，您的操作系统必须维护物理内存和虚拟内存之间的映射。每个操作系统都使用自己的复杂算法，通过一种叫做[页表](https://en.wikipedia.org/wiki/Page_table)的数据结构将虚拟内存地址映射到物理内存地址。

幸运的是，这种复杂性大部分隐藏在您的程序中。用 Python 编写高性能 I/O 代码不需要理解页表或逻辑到物理的映射。然而，了解一点内存会让你更好地理解计算机和库在为你做什么。

`mmap`使用虚拟内存，让您看起来好像已经将一个非常大的文件加载到内存中，即使该文件的内容太大而不适合您的物理内存。

### 共享内存

共享内存是操作系统提供的另一种技术，允许多个程序同时访问相同的数据。在使用[并发](https://realpython.com/python-concurrency/)的程序中，共享内存是处理数据的一种非常有效的方式。

Python 的`mmap`使用共享内存在多个 Python 进程、[线程](https://realpython.com/intro-to-python-threading/)和并发发生的任务之间高效地共享大量数据。

## 深入挖掘文件 I/O

现在，您已经对不同类型的内存有了一个较高的认识，是时候了解什么是内存映射以及它解决什么问题了。内存映射是执行文件 I/O 的另一种方式，可以提高性能和内存效率。

为了充分理解内存映射的作用，从底层角度考虑常规文件 I/O 是很有用的。当读取文件时，许多事情在幕后发生:

1.  **通过系统调用将**控制权转移给内核或核心操作系统代码
2.  **与文件所在的物理磁盘交互**
3.  **将**数据复制到[用户空间](https://en.wikipedia.org/wiki/User_space)和[内核空间](https://en.wikipedia.org/wiki/Kernel_(operating_system))之间的不同缓冲区

考虑以下执行常规 Python 文件 I/O 的代码:

```py
def regular_io(filename):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        text = file_obj.read()
        print(text)
```

这段代码将整个文件读入物理内存，如果运行时有足够的内存可用的话，然后[将它打印到屏幕上。](https://realpython.com/python-print/)

这种类型的文件 I/O 您可能在 Python 之旅的早期就已经了解过了。代码不是很密集或复杂。然而，在像`read()`这样的函数调用的掩盖下发生的事情是非常复杂的。请记住，Python 是一种高级编程语言，所以很多复杂性对程序员来说是隐藏的。

### 系统调用

实际上，对`read()`的调用意味着操作系统要做大量复杂的工作。幸运的是，操作系统提供了一种方法，通过[系统调用](https://en.wikipedia.org/wiki/System_call)，从你的程序中抽象出每个硬件设备的具体细节。每个操作系统将不同地实现这个功能，但是至少，`read()`必须执行几次系统调用来从文件中检索数据。

所有对物理硬件的访问都必须在一个名为**内核空间**的受保护环境中进行。系统调用是操作系统提供的 API，允许你的程序从用户空间进入内核空间，在内核空间管理物理硬件的底层细节。

在`read()`的情况下，操作系统需要几次系统调用才能与物理存储设备交互并返回数据。

同样，你不需要牢牢掌握系统调用和计算机架构的细节来理解内存映射。要记住的最重要的事情是，从计算上来说，系统调用相对昂贵，所以系统调用越少，代码可能执行得越快。

除了系统调用之外，对`read()`的调用还包括在数据返回到你的程序之前，在多个[数据缓冲区](https://en.wikipedia.org/wiki/Data_buffer)之间进行大量潜在的不必要的数据复制。

通常情况下，这一切发生得如此之快，以至于人们察觉不到。但是所有这些层都增加了**延迟**并且会减慢你的程序。这就是内存映射发挥作用的地方。

[*Remove ads*](/account/join/)

### 内存映射优化

避免这种开销的一种方法是使用一个[内存映射文件](https://en.wikipedia.org/wiki/Memory-mapped_file)。您可以将内存映射想象成一个过程，在这个过程中，读写操作跳过上面提到的许多层，将请求的数据直接映射到物理内存中。

内存映射文件 I/O 方法牺牲内存使用来换取速度，这被经典地称为[空间-时间权衡](https://en.wikipedia.org/wiki/Space-time_tradeoff)。然而，内存映射并不需要比传统方法使用更多的内存。操作系统非常聪明。它将[根据请求缓慢地加载](https://en.wikipedia.org/wiki/Lazy_loading)数据，类似于 [Python 生成器](https://realpython.com/introduction-to-python-generators/)的工作方式。

此外，由于虚拟内存，您可以加载比物理内存更大的文件。然而，当没有足够的物理内存存储文件时，您不会看到内存映射带来的巨大性能提升，因为操作系统将使用较慢的物理存储介质(如固态磁盘)来模拟它缺少的物理内存。

## 用 Python 的`mmap` 读取内存映射文件

现在，所有这些理论都已过时，您可能会问自己，“我如何使用 Python 的`mmap`来创建内存映射文件？”

下面是您之前看到的文件 I/O 代码的内存映射等价物:

```py
import mmap

def mmap_io(filename):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            text = mmap_obj.read()
            print(text)
```

这段代码将整个文件作为一个字符串读入内存，并将其打印到屏幕上，就像早期的常规文件 I/O 方法一样。

简而言之，使用`mmap`与读取文件的传统方式非常相似，只有一些小的变化:

1.  用`open()`打开文件是不够的。您还需要使用`mmap.mmap()`向操作系统发送信号，表示您希望将文件映射到 RAM 中。

2.  你需要确保你和`open()`使用的模式和`mmap.mmap()`兼容。`open()`的默认模式是读，而`mmap.mmap()`的默认模式是读*和*写。所以，在打开文件时，你必须明确。

3.  您需要使用`mmap`对象而不是由`open()`返回的标准[文件对象](https://docs.python.org/3/glossary.html#term-file-object)来执行所有的读写操作。

### 性能影响

内存映射方法比典型的文件 I/O 稍微复杂一些，因为它需要创建另一个对象。然而，当读取一个只有几兆字节的文件时，这一小小的改变可以带来巨大的性能优势。下面是读著名小说 [*《堂吉诃德的历史》*](http://www.gutenberg.org/files/996/996-0.txt) 的原文对比，大致是 2.4 兆:

>>>

```py
>>> import timeit
>>> timeit.repeat(
...     "regular_io(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import regular_io, filename")
[0.02022400000000002, 0.01988580000000001, 0.020257300000000006]
>>> timeit.repeat(
...     "mmap_io(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import mmap_io, filename")
[0.006156499999999981, 0.004843099999999989, 0.004868600000000001]
```

这是使用常规文件 I/O 和内存映射文件 I/O 读取整个 2.4 兆字节文件所需的时间。如您所见，内存映射方法大约需要 0.005 秒，而常规方法大约需要 0.02 秒。当读取更大的文件时，这种性能提升甚至会更大。

**注意:**这些结果是使用 Windows 10 和 Python 3.8 收集的。因为内存映射非常依赖于操作系统的实现，所以您的结果可能会有所不同。

Python 的`mmap`文件对象提供的 API 与传统文件对象非常相似，除了一个额外的超级能力:Python 的`mmap`文件对象可以像[字符串](https://realpython.com/python-strings/)对象一样被[切片](https://realpython.com/python-strings/#string-slicing)！

### `mmap`对象创建

在创建`mmap`对象的过程中，有一些细微之处值得仔细观察:

```py
mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ)
```

`mmap`需要一个[文件描述符](https://en.wikipedia.org/wiki/File_descriptor)，它来自一个常规文件对象的`fileno()`方法。文件描述符是一个内部标识符，通常是一个[整数](https://realpython.com/python-data-types/#integers)，操作系统用它来跟踪打开的文件。

`mmap`的第二个参数是`length=0`。这是存储器映射的字节长度。`0`是一个特殊的值，表示系统应该创建一个足够大的内存映射来保存整个文件。

`access`参数告诉操作系统你将如何与映射内存交互。选项有`ACCESS_READ`、`ACCESS_WRITE`、`ACCESS_COPY`和`ACCESS_DEFAULT`。这些有点类似于内置`open()`的`mode`参数:

*   **`ACCESS_READ`** 创建一个只读内存映射。
*   **`ACCESS_DEFAULT`** 默认为可选`prot`参数中指定的模式，用于[内存保护](https://en.wikipedia.org/wiki/Memory_protection)。
*   **`ACCESS_WRITE`** 和 **`ACCESS_COPY`** 是两种写模式，在下面你会了解到[。](#write-modes)

文件描述符、`length`和`access`参数表示创建一个内存映射文件所需的最低要求，该文件将在 Windows、Linux 和 macOS 等操作系统上工作。上面的代码是跨平台的，这意味着它将通过所有操作系统上的内存映射接口读取文件，而不需要知道代码运行在哪个操作系统上。

另一个有用的参数是`offset`，这是一种节省内存的技术。这指示`mmap`从文件中指定的偏移量开始创建一个内存映射。

[*Remove ads*](/account/join/)

### `mmap`字符串形式的对象

如前所述，内存映射将文件内容作为字符串透明地加载到内存中。因此，一旦你打开文件，你就可以执行许多与使用[字符串](https://realpython.com/python-strings/)相同的操作，比如[切片](https://realpython.com/python-strings/#string-slicing):

```py
import mmap

def mmap_io(filename):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            print(mmap_obj[10:20])
```

这段代码将十个字符从`mmap_obj`打印到屏幕上，并将这十个字符读入物理内存。同样，数据被缓慢地读取。

切片不会提升内部文件位置。所以，如果你在一个片后调用`read()`，那么你仍然会从文件的开始读取。

### 搜索内存映射文件

除了切片之外，`mmap`模块还允许其他类似字符串的行为，比如使用`find()`和`rfind()`在文件中搜索特定的文本。例如，有两种方法可以找到文件中第一次出现的`" the "`:

```py
import mmap

def regular_io_find(filename):
    with open(filename, mode="r", encoding="utf-8") as file_obj:
        text = file_obj.read()
        print(text.find(" the "))

def mmap_io_find(filename):
    with open(filename, mode="r", encoding="utf-8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            print(mmap_obj.find(b" the "))
```

这两个函数都在文件中搜索第一次出现的`" the "`,它们之间的主要区别是第一个函数在字符串对象上使用`find()`,而第二个函数在内存映射文件对象上使用`find()`。

**注意:** `mmap`操作的是字节，不是字符串。

以下是性能差异:

>>>

```py
>>> import timeit
>>> timeit.repeat(
...     "regular_io_find(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import regular_io_find, filename")
[0.01919180000000001, 0.01940510000000001, 0.019157700000000027]
>>> timeit.repeat(
...     "mmap_io_find(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import mmap_io_find, filename")
[0.0009397999999999906, 0.0018005999999999855, 0.000826699999999958]
```

那可是差了好几个数量级啊！同样，您的结果可能会因操作系统而异。

内存映射文件也可以直接和[正则表达式](https://realpython.com/regex-python/)一起使用。考虑下面的示例，该示例查找并打印出所有五个字母的单词:

```py
import re
import mmap

def mmap_io_re(filename):
    five_letter_word = re.compile(rb"\b[a-zA-Z]{5}\b")

    with open(filename, mode="r", encoding="utf-8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            for word in five_letter_word.findall(mmap_obj):
                print(word)
```

这段代码读取整个文件，并打印出其中正好有五个字母的每个单词。请记住，内存映射文件使用[字节字符串](https://realpython.com/python-strings/#bytes-objects)，因此正则表达式也必须使用字节字符串。

下面是使用常规文件 I/O 的等效代码:

```py
import re

def regular_io_re(filename):
    five_letter_word = re.compile(r"\b[a-zA-Z]{5}\b")

    with open(filename, mode="r", encoding="utf-8") as file_obj:
        for word in five_letter_word.findall(file_obj.read()):
            print(word)
```

这段代码还打印出文件中所有五个字符的单词，但是它使用传统的文件 I/O 机制，而不是内存映射文件。和以前一样，这两种方法的性能不同:

>>>

```py
>>> import timeit
>>> timeit.repeat(
...     "regular_io_re(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import regular_io_re, filename")
[0.10474110000000003, 0.10358619999999996, 0.10347820000000002]
>>> timeit.repeat(
...     "mmap_io_re(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import mmap_io_re, filename")
[0.0740976000000001, 0.07362639999999998, 0.07380980000000004]
```

内存映射方法仍然要快一个数量级。

[*Remove ads*](/account/join/)

### 作为文件的内存映射对象

内存映射文件是部分字符串和部分文件，因此`mmap`也允许您执行常见的文件操作，如`seek()`、`tell()`和`readline()`。这些函数的工作方式与常规的文件对象完全一样。

例如，下面是如何查找文件中的特定位置，然后执行单词搜索:

```py
import mmap

def mmap_io_find_and_seek(filename):
    with open(filename, mode="r", encoding="utf-8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            mmap_obj.seek(10000)
            mmap_obj.find(b" the ")
```

这段代码将寻找文件中的位置`10000`，然后找到第一次出现`" the "`的位置。

`seek()`对内存映射文件的作用与对常规文件的作用完全相同:

```py
def regular_io_find_and_seek(filename):
    with open(filename, mode="r", encoding="utf-8") as file_obj:
        file_obj.seek(10000)
        text = file_obj.read()
        text.find(" the ")
```

这两种方法的代码非常相似。让我们看看他们的表现如何比较:

>>>

```py
>>> import timeit
>>> timeit.repeat(
...     "regular_io_find_and_seek(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import regular_io_find_and_seek, filename")
[0.019396099999999916, 0.01936059999999995, 0.019192100000000045]
>>> timeit.repeat(
...     "mmap_io_find_and_seek(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import mmap_io_find_and_seek, filename")
[0.000925100000000012, 0.000788299999999964, 0.0007854999999999945]
```

同样，只需对代码进行一些小的调整，您的内存映射方法就会快得多。

## 用 Python 的`mmap` 写内存映射文件

内存映射对于读取文件最有用，但是您也可以使用它来写入文件。用于写文件的 API 与常规的文件 I/O 非常相似，除了一些不同之处。

下面是一个将文本写入内存映射文件的示例:

```py
import mmap

def mmap_io_write(filename, text):
    with open(filename, mode="w", encoding="utf-8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_WRITE) as mmap_obj:
            mmap_obj.write(text)
```

这段代码将文本写入内存映射文件。但是，如果在创建`mmap`对象时文件是空的，它将引发一个`ValueError`异常。

Python 的`mmap`模块不允许空文件的内存映射。这是合理的，因为从概念上讲，一个空的内存映射文件只是一个内存缓冲区，所以不需要内存映射对象。

通常，内存映射用于读取或读/写模式。例如，下面的代码演示了如何快速读取文件并只修改其中的一部分:

```py
import mmap

def mmap_io_write(filename):
    with open(filename, mode="r+") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_WRITE) as mmap_obj:
            mmap_obj[10:16] = b"python"
            mmap_obj.flush()
```

该功能将打开一个至少包含 16 个字符的文件，并将字符 10 至 15 更改为`"python"`。

写入`mmap_obj`的更改在磁盘上的文件和内存中都是可见的。官方 Python 文档建议总是调用`flush()`来保证数据被写回磁盘。

[*Remove ads*](/account/join/)

### 写入模式

写操作的语义由`access`参数控制。编写内存映射文件和普通文件的一个区别是`access`参数的选项。有两个选项可以控制如何将数据写入内存映射文件:

1.  **`ACCESS_WRITE`** 指定直写语义，意味着数据将通过内存写入并持久存储在磁盘上。
2.  **`ACCESS_COPY`** 不将更改写入磁盘，即使`flush()`被调用。

换句话说，`ACCESS_WRITE`写入内存和文件，而`ACCESS_COPY`只写入内存，*不写入底层文件。*

### 搜索和替换文本

内存映射文件将数据公开为一个字节字符串，但是这个字节字符串与常规字符串相比还有一个重要的优势。内存映射文件数据是一个由**个可变**字节组成的字符串。这意味着编写在文件中搜索和替换数据的代码要简单和高效得多:

```py
import mmap
import os
import shutil

def regular_io_find_and_replace(filename):
    with open(filename, "r", encoding="utf-8") as orig_file_obj:
        with open("tmp.txt", "w", encoding="utf-8") as new_file_obj:
            orig_text = orig_file_obj.read()
            new_text = orig_text.replace(" the ", " eht ")
            new_file_obj.write(new_text)

    shutil.copyfile("tmp.txt", filename)
    os.remove("tmp.txt")

def mmap_io_find_and_replace(filename):
    with open(filename, mode="r+", encoding="utf-8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_WRITE) as mmap_obj:
            orig_text = mmap_obj.read()
            new_text = orig_text.replace(b" the ", b" eht ")
            mmap_obj[:] = new_text
            mmap_obj.flush()
```

这两个函数都将给定文件中的单词`" the "`更改为`" eht "`。如您所见，内存映射方法大致相同，但是它不需要手动跟踪额外的临时文件来进行适当的替换。

在这种情况下，对于这种文件长度，内存映射方法实际上会稍慢一些。因此，对内存映射文件进行完全搜索和替换可能是也可能不是最有效的方法。这可能取决于许多因素，如文件长度、机器的内存速度等。也可能有一些操作系统缓存扭曲了时间。正如您所看到的，常规 IO 方法在每次调用时都会加快速度。

>>>

```py
>>> import timeit
>>> timeit.repeat(
...     "regular_io_find_and_replace(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import regular_io_find_and_replace, filename")
[0.031016973999996367, 0.019185273000005054, 0.019321329999996806]
>>> timeit.repeat(
...     "mmap_io_find_and_replace(filename)",
...     repeat=3,
...     number=1,
...     setup="from __main__ import mmap_io_find_and_replace, filename")
[0.026475408999999672, 0.030173652999998524, 0.029132930999999473]
```

在这个基本的搜索-替换场景中，内存映射会使代码稍微简洁一些，但并不总是能大幅提高速度。正如他们所说，“你的里程可能会有所不同。”

## 用 Python 的`mmap` 在进程间共享数据

到目前为止，您只对磁盘上的数据使用内存映射文件。然而，你也可以创建没有物理存储的匿名内存映射。这可以通过传递`-1`作为文件描述符来实现:

```py
import mmap

with mmap.mmap(-1, length=100, access=mmap.ACCESS_WRITE) as mmap_obj:
    mmap_obj[0:100] = b"a" * 100
    print(mmap_obj[0:100])
```

这在 RAM 中创建了一个匿名的内存映射对象，其中包含字母`"a"`的`100`个副本。

匿名内存映射对象本质上是内存中特定大小的缓冲区，由参数`length`指定。缓冲区类似于标准库中的 [`io.StringIO`](https://docs.python.org/3/library/io.html#io.StringIO) 或 [`io.BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO) 。然而，一个匿名的内存映射对象支持跨多个进程的共享，`io.StringIO`和`io.BytesIO`都不允许。

这意味着您可以使用匿名内存映射对象在进程之间交换数据，即使这些进程具有完全独立的内存和堆栈。下面是一个创建匿名内存映射对象来共享可以从两个进程中读写的数据的示例:

```py
import mmap

def sharing_with_mmap():
    BUF = mmap.mmap(-1, length=100, access=mmap.ACCESS_WRITE)

    pid = os.fork()
    if pid == 0:
        # Child process
        BUF[0:100] = b"a" * 100
    else:
        time.sleep(2)
        print(BUF[0:100])
```

使用这段代码，您创建了一个`100`字节的内存映射缓冲区，并允许从两个进程中读取和写入该缓冲区。如果您希望节省内存，同时仍能在多个进程间共享大量数据，这种方法会很有用。

使用内存映射共享内存有几个优点:

*   数据不必在进程间复制。
*   操作系统透明地处理内存。
*   数据不必在进程间[酸洗](https://realpython.com/python-pickle-module/)，节省了 CPU 时间。

说到**酸洗**，值得指出的是`mmap`与更高级、更全功能的 API 如内置`multiprocessing`模块不兼容。`multiprocessing`模块需要在进程间传递数据来支持 pickle 协议，而`mmap`不需要。

您可能会尝试使用`multiprocessing`而不是`os.fork()`，如下所示:

```py
from multiprocessing import Process

def modify(buf):
    buf[0:100] = b"xy" * 50

if __name__ == "__main__":
    BUF = mmap.mmap(-1, length=100, access=mmap.ACCESS_WRITE)
    BUF[0:100] = b"a" * 100
    p = Process(target=modify, args=(BUF,))
    p.start()
    p.join()
    print(BUF[0:100])
```

在这里，您试图创建一个新的进程，并将内存映射缓冲区传递给它。这段代码将立即引发一个 [`TypeError`](https://realpython.com/python-traceback/#typeerror) ，因为`mmap`对象不能被酸洗，这是将数据传递给第二个进程所必需的。因此，要使用内存映射共享数据，您需要坚持使用底层的`os.fork()`。

如果您使用的是 Python 3.8 或更新版本，那么您可以使用新的 [`shared_memory`模块](https://docs.python.org/3/library/multiprocessing.shared_memory.html)来更有效地跨 Python 进程共享数据:

```py
from multiprocessing import Process
from multiprocessing import shared_memory

def modify(buf_name):
    shm = shared_memory.SharedMemory(buf_name)
    shm.buf[0:50] = b"b" * 50
    shm.close()

if __name__ == "__main__":
    shm = shared_memory.SharedMemory(create=True, size=100)

    try:
        shm.buf[0:100] = b"a" * 100
        proc = Process(target=modify, args=(shm.name,))
        proc.start()
        proc.join()
        print(bytes(shm.buf[:100]))
    finally:
        shm.close()
        shm.unlink()
```

这个小程序创建了一个`100`字符列表，并从另一个进程中修改前 50 个字符。

注意，只有缓冲区的名称被传递给第二个进程。然后，第二个进程可以使用该唯一名称检索同一个内存块。这是由`mmap`供电的`shared_memory`模块的一个特殊功能。在幕后，`shared_memory`模块使用每个操作系统独特的 API 为您创建命名的内存映射。

现在您已经知道了新的共享内存 Python 3.8 特性的一些底层实现细节，以及如何直接使用`mmap`！

[*Remove ads*](/account/join/)

## 结论

内存映射是文件 I/O 的另一种方法，Python 程序可以通过`mmap`模块使用它。内存映射使用低级操作系统 API 将文件内容直接存储在物理内存中。这种方法通常会提高 I/O 性能，因为它避免了许多昂贵的系统调用，并减少了昂贵的数据缓冲区传输。

**在本教程中，您学习了:**

*   **物理**、**虚拟**和**共享内存**有什么区别
*   如何优化**内存使用**与内存映射
*   如何使用 Python 的 **`mmap`模块**在你的代码中实现内存映射

`mmap` API 类似于常规的文件 I/O API，所以测试起来相当简单。在您自己的代码中尝试一下，看看您的程序是否能从内存映射提供的性能改进中受益。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python mmap:用内存映射做文件 I/O**](/courses/python-mmap-io/)********