# 第十一章 文件和流

> 来源：[`www.cnblogs.com/Marlowes/p/5519591.html`](http://www.cnblogs.com/Marlowes/p/5519591.html)
> 
> 作者：Marlowes

到目前为止，本书介绍过的内容都是和解释器自带的数据结构打交道。我们的程序与外部的交互只是通过`input`、`raw_input`和`print`函数，与外部的交互很少。本章将更进一步，让程序能接触更多领域：文件和流。本章介绍的函数和对象可以让你在程序调用时存储数据，并且可以处理来自其他程序的数据。

## 11.1 打开文件

`open`函数用来打开文件，语法如下：

```py
open(name[, mode[, buffering]]) 
```

`open`函数使用一个文件名作为唯一的强制参数，然后返回一个文件对象。模式(`mode`)和缓冲(`buffering`)参数都是可选的，我会在后面的内容中对它们进行解释。

因此，假设有一个名为`somefile.txt`的文本文件(可能是用文本编辑器创建的)，其存储路径是`c:\text`(或者在 UNIX 下的`~/text`)，那么可以像下面这样打开文件。

```py
>>> f = open(r"C:\text\somefile.txt") 
```

如果文件不存在，则会看到一个类似下面这样的异常回溯：

```py
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> 
  IOError: [Errno 2] No such file or directory: 'C:\\text\\somefile.txt' 
```

稍后会介绍文件对象的用处。在此之前，先来看看`open`函数的其他两个参数。

### 11.1.1 文件模式

如果`open`函数只带一个文件名参数，那么我们可以获得能读取文件内容的文件对象。如果要向文件内写入内容，则必须提供一个模式参数(稍后会具体地说明读和写方式)来显式声明。

`open`函数中的模式参数只有几个值，如表 11-1 所示。

明确地指出读模式和什么模式参数都不用的效果是一样的。使用写模式可以向文件写入内容。

`'+'`参数可以用到其他任何模式中，指明读和写都是允许的。比如`'r+'`能在打开一个文本文件用来读写时使用(也可以使用 seek 方法来实现，请参见本章后面的"随机访问"部分)。

表 11-1 `open`函数中模式参数的常用值

```py
'r'　　　　　　　　　　读模式
'w'　　　　　　　　　　写模式
'a'　　　　　　　　　　追加模式
'b'　　　　　　　　　　二进制模式(可添加到其他模式中使用)
'+'　　　　　　　　　　读/写模式(可添加到其他模式中使用) 
```

`'b'`模式改变处理文件的方法。一般来说，Python 假定处理的是文本文件(包含字符)。通常这样做不会有任何问题。但是如果处理的是一些其他类型的文件(二进制文件)，比如声音剪辑或者图像，那么应该在模式中增加`'b'`。参数`'rb'`可以用来读取一个二进制文件。

**为什么使用二进制模式**

如果使用二进制模式来读取(写入)文件的话，与使用文本模式不会有很大区别。仍然能读一定数量的字节(基本上和字符一样)，并且能执行和文本文件有关的操作。关键是，在使用二进制模式时，Python 会原样给出文件中的内容——在文本模式下则不一定。

Python 对于文本文件的操作方式令人有些惊讶，但不必担心。其中唯一要用到的技巧就是标准化换行符。一般来说，在 Python 中，换行符(`\n`)表示结束一行并另起一行，这也是 UNIX 系统中的规范。但在 Windows 中一行结束的标志是`\r\n`。为了在程序中隐藏这些区别(这样的程序就能跨平台运行)，Python 在这里做了一些自动转换：当在 Windows 下用文本模式读取文件中的文本时，Python 将`\r\n`转换成`\n`。相反地，当在 Windows 下用文本模式向文件写文本时，Python 会把`\n`转换成`\r\n`(Macintosh 系统上的处理也是如此，只是转换是在`\r`和`\n`之间进行)。

在使用二进制文件(比如声音剪辑)时可能会产生问题，因为文件中可能包含能被解释成前面提及的换行符的字符，而使用文本模式，Python 能自动转换。但是这样会破坏二进制数据。因此为了避免这样的事发生，要使用二进制模式，这样就不会发生转换了。

需要注意的是，在 UNIX 这种以换行符为标准行结束标志的平台上，这个区别不是很重要，因为不会发生任何转换。

*注：通过在模式参数中使用`U`参数能够在打开文件时使用通用的换行符支持模式，在这种模式下，所有的换行符/字符串(`\r\n`、`\r`或者是`\n`)都被转换成`\n`，而不用考虑运行的平台。*

### 11.1.2 缓冲

`open`函数的第 3 个参数(可选)控制着文件的缓冲。如果参数是`0`(或者是`False`)，I/O(输入/输出)就是无缓冲的(所有的读写操作都直接针对硬盘)；如果是 1(或者`True`)，I/O 就是有缓冲的(意味着 Python 使用内存来代替硬盘，让程序更快，只有使用`flush`或者`close`时才会更新硬盘上的数据——参见 11.2.4 节)。大于 1 的数字代表缓冲区的大小(单位是字节)，`-1`(或者是任何负数)代表使用默认的缓冲区大小。

## 11.2 基本的文件方法

打开文件的方法已经介绍了，那么下一步就是用它们做些有用的事情。接下来会介绍文件对象(和一些类文件对象，有时称为*流*)的一些基本方法。

*注：你可能会在 Python 的职业生涯多次遇到类文件这个术语(我已经使用了好几次了)。类文件对象是支持一些`file`类方法的对象，最重要的是支持`read`方法或者`write`方法，或者两者兼有。那些由`urllib.urlopen`(参见第十四章)返回的对象是一个很好的例子。它们支持的方法有`read`、`readline`和`readlines`。但(在本书写作期间)也有一些方法不支持，如`isatty`方法。*

**三种标准的流**

第十章中关于`sys`模块的部分曾经提到过 3 种流。它们实际上是文件(或者是类文件对象)：大部分文件对象可用的操作它们也可以使用。

数据输入的标准源是`sys.stdin`。当程序从标准输入读取数据时，你可以通过输入或者使用管道把它和其他程序的标准输出链接起来提供文本(管道是标准的 UNIX 概念)。

要打印的文本保存在`sys.stdout`内。`input`和`raw_input`函数的提示文字也是写入在`sys.stdout`中的。写入`sys.stdout`的数据一般是出现在屏幕上，但也能使用管道连接到其他程序的标准输入。

错误信息(如栈追踪)被写入`sys.stderr`。它和`sys.stdout`在很多方面都很像。

### 11.2.1 读和写

文件(或流)最重要的能力是提供或者接受数据。如果有一个名为 f 的类文件对象，那么就可以用`f.write`方法和`f.read`方法(以字符串形式)写入和读取数据。

每次调用`f.write(string)`时，所提供的参数`string`会被追加到文件中已存在部分的后面。

```py
>>> f = open("somefile.txt", "w") 
>>> f.write("Hello, ") 
>>> f.write("World!") >>> f.close() 
```

在完成了对一个文件的操作时，调用`close`。这个方法会在 11.2.4 节进行详细的介绍。

读取很简单，只要记得告诉流要读多少字符(字节)即可。例子(接上例)如下：

```py
>>> f = open("somefile.txt", "r") 
>>> f.read(4) 'Hell'
>>> f.read() 
'o, World!' 
```

首先指定了我要读取的字符数`"4"`，然后(通过不提供要读取的字符数的方式)读取了剩下的文件。注意，在调用`open`时可以省略模式，因为`'r'`是默认的。

### 11.2.2 管式输出

在 UNIX 的 shell(就像 GUN bash)中，使用*管道*可以在一个命令后面续写其他的多个命令，就像下面这个例子(假设是 GUN bash)。

```py
$ cat somefile.txt | python somescript.py | sort 
```

*注：GUN bash 在 Windows 中也是存在的。 [`www.cygwin.com`](http://www.cygwin.com) 上面有更多的信息。在 Mac OS X 中，是通过 Terminal 程序，可以使用 shell 文件。*

这个管道由以下三 3 个命令组成。

☑ `cat somefile.txt`：只是把`somefile.txt`的内容写到标准输出(`sys.stdout`)。

☑ `python somescript.py`：这个命令运行了 Python 脚本`somescript`。脚本应该是从标准输入读，把结果写入到标准输出。

☑ `sort`：这条命令从标准输入(`sys.stdin`)读取所有的文本，按字母排序，然后把结果写入标准输出。

但管道符号(`|`)的作用是什么？`somescript.py`的作用又是什么呢？

管道符号讲一个命令的标准输出和下一个命令的标准输入连接在一起。明白了吗？这样，就知道`somescript.py`会从它的`sys.stdin`中读取数据(`cat somefile.txt`写入的)，并把结果写入它的`sys.stdout`(`sort`在此得到数据)中。

使用`sys.stdin`的一个简单的脚本(`somescript`)如代码清单 11-1 所示。`somefile.txt`文件的内容如代码清单 11-2 所示。

```py
# 代码清单 11-1 统计`sys.stdin`中单词数的简单脚本 
# somescript.py

import sys
text = sys.stdin.read()
words = text.split()
wordcount = len(words) 
print "Wordcount:", wordcount 
# 代码清单 11-2 包含示例文本的文件
Your mother was a hamster and your father smelled of elderberries. 
```

下面是`cat somefile.txt | python somescript.py`的结果。

```py
Wordcount: 11 
```

**随机访问**

本章内的例子把文件都当成流来操作，也就是说只能按照从头到尾的顺序读数据。实际上，在文件中随意移动读取位置也是可以的，可以使用类文件对象的方法`seek`和`tell`来直接访问感兴趣的部分(这种做法称为随机访问)。

```py
seek(offset[, whence]) 
```

这个方法把当前位置(进行读和写的位置)移动到由`offset`和`whence`定义的位置。`Offset`类是一个字节(字符)数，表示偏移量。`whence`默认是 0，表示偏移量是从文件开头开始计算的(偏移量必须是非负的)。`whence`可能被设置为 1(相对于当前位置的移动，此时偏移量`offset``可以是负的)或者 2(相对于文件结尾的移动)。

考虑下面这个例子：

```py
>>> f = open(r"c:\text\somefile.txt", "w") 
>>> f.write("01234567890123456789") 
>>> f.seek(5) 
>>> f.write("Hello, World!") 
>>> f.close() 
>>> f = open(r"c:\text\somefile.txt") 
>>> f.read() 
>>> '01234Hello, World!89'
# tell 方法返回当前文件的位置如下例所示：
>>> f = open(r"c:\text\somefile.txt") 
>>> f.read(3) 
>>> '012'
>>> f.read(2) 
>>> '34'
>>> f.tell() 
>>> 5L 
```

### 11.2.3 读写行

实际上，程序到现在做的工作都是很不实用的。通常来说，逐个字符串读取文件也是没问题的，进行逐行的读取也可以。还可以使用 file.readline 读取单独的一行(从当前位置开始直到一个换行符出现，也读取这个换行符)。不使用任何参数(这样，一行就被读取和返回)或者使用一个非负数的整数作为`readline`可以读取的字符(或字节)的最大值。因此，如果`someFile.readline()`返回`"Hello, World!\n"`，`someFile.readline(5)`返回`"Hello"`。`readlines`方法可以读取一个文件中的所有行并将其作为列表返回。

`writelines`方法和`readlines`相反：传给它一个字符串的列表(实际上任何序列或者可迭代的对象都行)，它会把所有的字符串写入文件(或流)。注意，程序不会增加新行，需要自己添加。没有`writeline`方法，因为能使用`write`。

*注：在使用其他的符号作为换行符的平台上，用`\r`(Mac 中)和`\r\n`(Windows 中)代替`\n`(有`os.linesep`决定)。*

### 11.2.4 关闭文件

应该牢记使用`close`方法关闭文件。通常来说，一个文件对象在退出程序后(也可能在退出前)自动关闭，尽管是否关闭文件不是很重要，但关闭文件是没有什么害处的，可以避免在某些操作系统或设置中进行无用的修改，这样做也会避免用完系统中所打开文件的配额。

写入过的文件总是应该关闭，是因为 Python 可能会缓存(出于效率的考虑而把数据临时地存储在某处)写入的数据，如果程序因为某些原因崩溃了，那么数据根本就不会被写入文件。为了安全起见，要在使用完文件后关闭。

如果想确保文件被关闭了，那么应该使用`try/finally`语句，并且在`finally`子句中调用`close`方法。

```py
# Open your file here
try: 
    # Write data to your file
finally:
    file.close() 
```

事实上，有专门为这种情况设计的语句(在 Python2.5 中引入)，即`with`语句：

```py
with open("somefile.txt") as somefile:
    do_something(somefile) 
```

`with`语句可以打开文件并且将其赋值到变量上(本例是`somefile`)。之后就可以将数据写入语句体中的文件(或许执行其他操作)。文件在语句结束后会被自动关闭，即使是处于异常引起的结束也是如此。

在 Python2.5 中，`with`语句只有在导入如下的模块后才可以用：

```py
from __future__ import with_statement 
```

而 2.5 之后的版本中，`with`语句可以直接使用。

注：在写入了一些文件的内容后，通常的想法是希望这些改变会立刻体现在文件中，这样一来其他读取这个文件的程序也能知道这个改变。哦，难道不是这样吗？不一定。数据可能被缓存了(在内存中临时性地存储)，直到关闭文件才会被写入到文件。如果需要继续使用文件(不关闭文件)，又想将磁盘上的文件进行更新，以反映这些修改，那么就要调用文件对象的`flush`方法(注意，`flush`方法不允许其他程序使用该文件的同时访问文件，具体的情况依据使用的操作系统和设置而定。不管在什么时候，能关闭文件时最好关闭文件)。

**上下文管理器**

`with`语句实际上是很通用的结构，允许使用所谓的上下文管理器(context manager)。上下文管理器是一种支持`__enter__`和`__exit__`这两个方法的对象。

`__enter__`方法不带参数，它在进入`with`语句块的时候被调用，返回值绑定到在`as`关键字之后的变量。

`__exit__`方法带有 3 个参数：异常类型、异常对象和异常回溯。在离开方法(通过带有参数提供的、可引发的异常)时这个函数被调用。如果`__exit__`返回`false`，那么所有的异常都不会被处理。

文件可以被用作上下文管理器。它们的`__enter__`方法返回文件对象本身，`__exit__`方法关闭文件。有关这个强大且高级的特性的更多信息，请参看 Python 参考手册中的上下文管理器部分。或者可以在 Python 库参考中查看上下文管理器和`contextlib`部分。

### 11.2.5 使用基本文件方法

假设`somefile.txt`包含如代码清单 11-3 所示的内容，能对它进行什么操作？

```py
# 代码清单 11-3 一个简单的文本文件
Welcome to this file
There is nothing here except This stupid haiku 
```

让我们试试已经知道的方法，首先是`read(n)`：

```py
>>> f = open(r"C:\text\somefile.txt") 
>>> f.read(7) 'Welcome'
>>> f.read(4) ' to '
>>> f.close() 
```

然后是`read()`：

```py
>>> f = open(r"C:\text\somefile.txt") 
>>> print f.read()
Welcome to this file
There is nothing here except This stupid haiku 
>>> f.close() 
```

接着是`readline()`：

```py
>>> f = open(r"C:\text\somefile.txt") 
>>> for i in range(3):
...     print str(i) + ": " + f.readline(),
...     
0: Welcome to this file 
1: There is nothing here except
2: This stupid haiku 
>>> f.close() 
```

以及`readlines()`：

```py
>>> import pprint 
>>> pprint.pprint(open(r"C:\text\somefile.txt").readlines())
['Welcome to this file\n', 'There is nothing here except\n', 'This stupid haiku'] 
```

注意，本例中我所使用的是文件对象自动关闭的方式。

下面是写文件，首先是`write(string)`：

```py
>>> f = open(r"C:\text\somefile.txt", "w") 
>>> f.write("this\nis no\nhaiku") 
>>> f.close() 
```

在运行这个程序后，文件包含的内容如代码清单 11-4 所示。

```py
# 代码清单 11-4 修改了的文本文件
this is no
haiku 
```

最后是`writelines(list)`：

```py
>>> f = open(r"C:\text\somefile.txt") 
>>> lines = f.readlines() 
>>> f.close() 
>>> lines[1] = "isn't a\n"
>>> f = open(r"C:\text\somefile.txt", "w") 
>>> f.writelines(lines) 
>>> f.close() 
```

运行这个程序后，文件包含的文本如代码清单 11-5 所示。

```py
# 代码清单 11-5 再次修改的文本文件
this
isn't a
haiku 
```

## 11.3 对文件内容进行迭代

前面介绍了文件对象提供的一些方法，以及如何获取这样的文件对象。对文件内容进行迭代以及重复执行一些操作，是最常见的文件操作之一。尽管有很多方法可以实现这个功能，或者可能有人会偏爱某一种并坚持只使用那种方法，但是还有一些人使用其他的方法，为了能理解他们的程序，你就应该了解所有的基本技术。其中的一些技术是使用曾经见过的方法(如`read`、`readline`和`readlines`)，另一些方法是我即将介绍的(比如`xreadlines`和文件迭代器)。

在这部分的所有例子中都使用了一个名为`process`的函数，用来表示每个字符或每行的处理过程。读者也可以用你喜欢的方法自行实现这个函数。下面就是一个例子：

```py
def process(string): 
    print "Processing: ", string 
```

更有用的实现是在数据结构中存储数据，计算和值，用`re`模块来代替模式或者增加行号。

如果要尝试实现以上功能，则应该把`filename`变量设置为一个实际的文件名。

### 11.3.1 按字节处理

最常见的对文件内容进行迭代的方法是在`while`循环中使用`read`方法。例如，对每个字符(字节)进行循环，可以用代码清单 11-6 所示的方法实现。

```py
# 代码清单 11-6 用 read 方法对每个字符进行循环
f = open(filename)
char = f.read(1) 
while char:
    process(char)
    char = f.read(1)
f.close() 
```

这个程序可以使用是因为当到达文件的末尾时，`read`方法返回一个空的字符串，但在那之前返回的字符串会包含一个字符(这样布尔值是真)。如果`char`是真，则表示还没有到文件末尾。

可以看到，赋值语句`char = f.read(1)`被重复地使用，代码重复通常被认为是一件坏事。(懒惰是美德，还记得吗？)为了避免发生这种情况，可以使用在第五章介绍过的`while true/break`语句。最终的代码如代码清单 11-7 所示。

```py
# 代码清单 11-7 用不同的方式写循环
f = open(filename) 
while True:
    char = f.read() 
    if not char: 
    break 
    process(char)
f.close 
```

如在第五章提到的，`break`语句不应该频繁地使用(因为这样会让代码很难懂)；尽管如此，代码清单 11-7 中使用的方法比代码清单 11-6 中的方法要好，因为前者避免了重复的代码。

### 11.3.2 按行操作

当处理文本文件时，经常会对文件的行进行迭代而不是处理单个字符。处理行使用的方法和处理字符一样，即使用`readline`方法(先前在 11.2.3 节介绍过)，如代码清单 11-8 所示。

```py
# 代码清单 11-8 在 while 循环中使用 readline
f = open(filename) 
while True:
    line = f.readline() 
    if not line: 
    break 
    process(line)
f.close() 
```

### 11.3.3 读取所有内容

如果文件不是很大，那么可以使用不带参数的`read`方法一次读取整个文件(把整个文件当做一个字符串来读取)，或者使用`readlines`方法(把文件读入一个字符串列表，在列表中每个字符串就是一行)。代码清单 11-9 和代码清单 11-10 展示了在读取这样的文件时，在字符串和行上进行迭代是多么容易。注意，将文件的内容读入一个字符串或者是读入列表在其他时候也很有用。比如在读取后，就可以对字符串使用正则表达式操作，也可以将行列表存入一些数据结构中，以备将来使用。

```py
# 代码清单 11-9 用 read 迭代每个字符
f = open(filename) 
for char in f.read():
    process(char)
f.close() 
# 代码清单 11-10 用 readlines 迭代行
f = open(filename) 
for line in f.readlines():
    process(line)
f.close() 
```

### 11.3.4 使用`fileinput`实现懒惰行迭代

在需要对一个非常大的文件进行行迭代的操作时，`readlines`会占用太多的内存。这个时候可以使用`while`循环和`readline`方法来替代。当然，在 Python 中如果能使用`for`循环，那么它就是首选。本例恰好可以使用`for`循环可以使用一个名为*懒惰行迭代*的方法：说它懒惰是因为它只是读取实际需要的文件部分。

第十章内已经介绍过`fileinput`，代码清单 11-11 演示了它的用法。注意，`fileinput`模块包含了打开文件的函数，只需要传一个文件名给它。

```py
# 代码清单 11-11 用 fileinput 来对行进行迭代

import fileinput 
for line in fileinput.input(filename):
    process(line) 
```

*注：在旧式代码中，可使用`xreadlines`实现懒惰行迭代。它的工作方式和`readlines`很类似，不同点在于，它不是将全部的行读到列表中而是创建了一个`xreadlines`对象。注意，`xreadlines`是旧式的，在你自己的代码中最好用`fileinput`或文件迭代器(下面来介绍)。*

### 11.3.5 文件迭代器

现在是展示所有最酷的技术的时候了，在 Python 中如果一开始就存在这个特性的话，其他很多方法(至少包括`xreadlines`)可能就不会出现了。那么这种技术到底是什么？在 Python 的近几个版本中(从 2.2 开始)，文件对象是*可迭代*的，这就意味着可以直接在`for`循环中使用它们，从而对它们进行迭代。如代码清单 11-12 所示，很优雅，不是吗？

```py
# 代码清单 11-12 迭代文件
f = open(filename) for line in f:
    process(line)
f.close() 
```

在这些迭代的例子中，都没有显式的关闭文件的操作，尽管在使用完以后，文件的确应该关闭，但是只要没有向文件内写入内容，那么不关闭文件也是可以的。如果希望由 Python 来负责关闭文件(也就是刚才所做的)，那么例子应该进一步简化，如代码清单 11-13 所示。在那个例子中并没有把一个打开的文件赋给变量(就像我在其他例子中使用的变量`f`)，因此也就没办法显式地关闭文件。

```py
# 代码清单 11-13 对文件进行迭代而不使用变量存储文件对象

for line in open(filename):
    process(line) 
```

注意`sys.stdin`是可迭代的，就像其他的文件对象。因此如果想要迭代标准输入中的所有行，可以按如下形式使用`sys.stdin`。

```py
import sys 
for line in sys.stdin:
    process(line) 
```

可以对文件迭代器执行和普通迭代器相同的操作。比如将它们转换为字符串列表(使用 list(open(filename)))，这样所达到的效果和使用 readlines 一样。

考虑下面的例子：

```py
>>> f = open("somefile.txt", "w") 
>>> f.write("First line\n") 
>>> f.write("Second line\n") 
>>> f.write("Third line\n") 
>>> f.close() 
>>> lines = list(open("somefile.txt")) 
>>> lines
['First line\n', 'Second line\n', 'Third line\n'] 
>>> first, second, third = open("somefile.txt") 
>>> first 'First line\n'
>>> second 'Second line\n'
>>> third 'Third line\n' 
```

在这个例子中，注意下面的几点很重要。

☑ 在使用`print`来向文件内写入内容，这会在提供的字符串后面增加新的行。

☑ 使用序列来对一个打开的文件进行解包操作，把每行都放入一个单独的变量中(这么做是很有实用性的，因为一般不知道文件中有多少行，但它演示了文件对象的"迭代性")。

☑ 在写文件后关闭了文件，是为了确保数据被更新到硬盘(你也看到了，在读取文件后没有关闭文件，或许是太马虎了，但并没有错)。

## 11.4 小结

本章中介绍了如何通过文件对象和类文件对象与环境互动，I/O 也是 Python 中最重要的技术之一。下面是本章的关键知识。

☑ 类文件对象：类文件对象是支持`read`和`readline`方法(可能是`write`和`writelines`)的非正式对象。

☑ 打开和关闭文件：通过提供一个文件名，使用`open`函数打开一个文件(在新版的 Python 中实际上是`file`的别名)。如果希望确保文件被正常关闭，即使发生错误时也是如此可以使用`with`语句。

☑ 模式和文件类型：当打开一个文件时，也可以提供一个*模式*，比如`'r'`代表读模式，`'w'`代表写模式。还可以将文件作为二进制文件打开(这个只在 Python 进行换行符转换的平台上才需要，比如 Windows，或许其他地方也应该如此)。

☑ 标准流：3 个标准文件对象(在`sys`模块中的`stdin`、`stdout`和`stderr`)是一个类文件对象，该对象实现了 UNIX 标准的 I/O 机制(Windows 中也能用)。

☑ 读和写：使用`read`或是`write`方法可以对文件对象或类文件对象进行读写操作。

☑ 读写行：使用`readline`和`readlines`和(用于有效迭代的)`xreadlines`方法可以从文件中读取行，使用`writelines`可以写入数据。

☑ 迭代文件内容：有很多方法可以迭代文件的内容。一般是迭代文本中的行，通过迭代文件对象本身可以轻松完成，也有其他的方法，就像`readlines`和`xreadlines`这两个倩兼容 Python 老版本的方法。

### 11.4.1 本章的新函数

本章涉及的新函数如表 11-2 所示。

表 11-2 本章的新函数

```py
file(name[, mode[, buffering]])  打开一个文件并返回一个文件对象
open(name[, mode[, buffering]])  file 的别名；在打开文件时，使用 open 而不是 file 
```

### 11.4.2 接下来学什么

现在你已经知道了如何通过文件与环境交互，但怎么和用户交互呢？到现在为止，程序已经使用的只有`input`、`raw_input`和`print`函数，除非用户在程序能够读取的文件中写入一些内容，否则没有任何其他工具能创建用户界面。下一章会介绍图形用户界面(graphical user interface)中的窗口、按钮等。