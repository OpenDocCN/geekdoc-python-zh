# 如何在 Python 中打开文件

> 原文：<https://www.pythoncentral.io/how-to-open-a-file-in-python/>

Python 是 [最流行的编程语言](https://insights.stackoverflow.com/survey/2020) 之一，与其他一些语言不同，它不需要你导入一个库来处理文件。

文件是在 Python 中原生处理的[；然而，处理文件的方法与其他语言不同。](https://docs.python.org/3/tutorial/inputoutput.html)

在这个 Python 教程中，你将学习如何打开、读取、写入和关闭文件。在这篇文章中，我们还将介绍如何使用“with”语句。

## **如何在 Python 中打开文件？**

Python 内置了创建、打开、关闭、读取和写入文件的功能。在 Python 中打开一个文件就像使用 open()函数一样简单，该函数在每个 Python 版本中都可用。该函数返回一个“文件对象”

文件对象由方法和属性组成，这些方法和属性可以用来收集关于你打开的文件的信息。此外，当您想要操作您打开的文件时，这些细节也会派上用场。

例如，“模式”是每个文件对象都关联的少数属性之一。它描述了文件打开的模式。

同样，file 对象的“name”属性揭示了打开的文件的名称。

你必须记住，虽然文件和文件对象在根本上是相互关联的，但它们是分开的。

## **什么是文件？**

当您使用像 Windows 这样的操作系统时，“文件”意味着图像、文档、视频、音频剪辑、可执行文件等等。在技术层面上，文件只是磁盘驱动器上存储相关数据的命名位置。

磁盘驱动器用于永久存储数据，因为 RAM 是易失性的，当计算机关机时会丢失数据。

在操作系统中，你可以操作这些文件，并把它们组织在文件夹中，形成复杂的目录。本质上，Windows 中的文件是您可以创建、操作和删除的任何项目。

然而，Python 处理文件的方式不同于 Windows 或任何其他操作系统。

### **Python 中的文件类型**

Python 将文件分为两类:文本或二进制。在学习如何使用 Python 文件之前，理解它们之间的差异是至关重要的。

文本文件是具有数据序列的文件，数据是字母数字字符。在这些文件中，每一行信息都以称为 EOL 或行尾字符的特殊字符结束。

一个文本文件可以使用许多不同的 EOL 字符。然而，新行字符(" \n ")是 Python 的默认 EOL 字符，因此是最常用的字符。

新行字符中的反斜杠向 Python 解释器表明当前行已经结束。反斜杠后面的“n”向解释器表明后面的数据必须被视为新的一行。

二进制文件与文本文件截然不同。顾名思义，二进制文件包含二进制数据 0 和 1，只能由理解二进制的应用程序处理。

这些文件不使用 EOL 字符或任何终止符。数据在转换成二进制后被存储。

-

**注:** Python 字符串不同于文件，但学习如何使用字符串可以帮助更好地理解 Python 文件的工作原理。要了解更多关于在 Python 中使用字符串的知识，请查看我们的 [关于字符串](https://www.pythoncentral.io/what-is-the-string-in-python/) 的综合指南。

-

## **打开文本文件**

在写入或读取文件之前，必须先打开文件。为此，您可以使用 Python 内置的 open()函数。

这个函数有两个参数:一个接受文件名，另一个保存访问模式。它返回一个文件对象，语法如下:

```py
file_object = open("File_Name", "Access_Mode")
```

你必须注意，你要打开的文件必须和 Python 脚本在同一个文件夹/目录中。如果文件不在同一个目录中，在写入文件名参数时，必须提到文件的完整路径。

例如，要打开与脚本在同一个目录中的“example.txt”文件，您需要编写以下代码:

```py
f = open("example.txt")
```

另一方面，要在桌面上打开“example.txt”文件，您需要编写以下代码:

```py
f = open("C:\Users\Krish\Desktop\example.txt")
```

### **访问模式**

Python 有几种访问模式，这些模式控制着你可以对一个打开的文件执行的操作。换句话说，每种访问模式都是指当文件以不同方式打开时如何使用它。

Python 中的访问模式也指定了文件句柄的位置。您可以将文件句柄想象成一个光标，指示必须从哪里读取或写入数据。

也就是说，使用访问模式参数是可选的，如前面的示例所示。

Python 中有六种访问模式:

1.  **只读(' r'):** 这是默认的访问模式。它打开文本文件进行读取，并将文件句柄定位在文件的开头。如果 Python 发现该文件不存在，它会抛出 I/O 错误，并且不会创建具有指定名称的文件。
2.  **只写(' w'):** 用于写文件。如果文件不在文件夹中，则会创建一个具有指定名称的文件。如果文件存在，数据将被截断，然后被覆盖。文件句柄位于文件的开头。
3.  **读写(' r+'):** 你可以读写用这种访问模式打开的文件。它将句柄定位在文件的开头。如果文件不在目录中，就会引发 I/O 错误。
4.  **读写(' w+'):** 与前一种模式一样，你可以读写用这种访问模式打开的文件。它将句柄定位在文件的开头。如果文件不在文件夹中，它会创建一个新文件。此外，如果指定的文件存在，其中的数据将被截断和覆盖。
5.  **仅追加(' a'):** 此模式打开文件进行写入。如果特定文件不在目录中，则会创建一个同名的新文件。但是，与其他模式不同，句柄位于文件的末尾。因此，输入的数据会附加在所有现有数据之后。
6.  **追加和读取(' a+'):** 在 Python 中用来读写文件。它的工作方式与前面的模式类似。如果目录中没有该文件，则会创建一个同名的文件。或者，如果文件可用，则输入的数据会附加在现有数据之后。

访问模式参数中的“+”实质上表示文件正在被打开以进行更新。

还有另外两个参数可以用来指定文件的打开模式。

Python 默认以文本模式读取文件，文本模式由参数“t”指定，open()函数在读取文件时返回字符串。相反，在使用 open()函数时使用二进制模式(“b”)会返回字节。二进制模式通常用于处理非文本文件，如图像和可执行文件。

这里有几个在 Python 中使用这些访问模式的例子:

```py
f = open("example.txt")      # Uses default access mode 'r'
f = open("example.txt",'w')  # Opens the file for writing in text mode
f = open("img.bmp",'r+b')    # Opens the file for reading and writing in binary mode 
```

**With 语句**

除了有更清晰的语法之外,“with”语句还使得处理文件对象时的异常处理更加容易。在处理文件时，只要适用，最好使用 with 语句。

使用 with 语句的一个显著优点是，它会在操作完成后自动关闭你打开的所有文件。因此，您不必担心在对文件执行任何操作后关闭文件并进行清理。

with 语句的语法是:

```py
with open("File_Name") as file:
```

要使用 with 语句读取文件，您需要编写以下代码行:

```py
with open("example.txt") as f: 
data = f.readlines() 
```

当第二行代码运行时，存储在“example.txt”中的所有数据都将存储在一个名为“data”的字符串中

如果你想将数据写入同一个文件，你可以运行下面的代码:

```py
with open("example.txt", "w") as f: 
f.write("Hello World") 
```

## **阅读内容**

在 Python 中读取一个文件很简单——你必须使用“r”参数或者根本不提及参数，因为这是默认的访问模式。

也就是说，在 Python 中还有其他从文件中读取数据的方法。例如，您可以使用 Python 内置的 read(size)方法来读取指定大小的数据。如果不指定 size 参数，该方法将读取文件，直到文件结束。

我们假设“example.txt”有以下文字: `Hello world.
This file has two lines.` 

要使用 read()方法读取文件，可以使用下面的代码:

```py
f = open("example.txt",'r',encoding = 'utf-8')

f.read(5)    # reads the first five characters of data 'Hello'

f.read(7)    # reads the next seven characters in the file ' world.'

f.read()     # reads the rest till EOF '\nThis file has two lines.'

f.read()     # returns empty string
```

在上面的例子中，read()方法在行结束时返回新的行字符。此外，一旦到达文件末尾，它将返回一个空字符串。

在 seek()方法的帮助下，无论使用何种访问模式，都可以改变光标位置。要检查光标的当前位置，可以使用 tell()函数。

你也可以使用 for 循环逐行读取文件，就像这样:

```py
 for line in f:
    print(line, end = '')
...
Hello world.
This file has two lines. 
```

readline()和 readlines()方法是另外两种从文件中读取数据的可靠技术。readline()方法在读取一个新的行字符后停止读取该行。

```py
f.readline() # reads the first line 'Hello world.'
```

另一方面，readlines()读取文件，直到到达 EOF。

```py
f.readlines() # reads the first line 'Hello world.'
```

 **追加新文本**

要写入 Python 文件，必须以写(w)、追加(a)或独占创建(x)模式打开。

使用写模式时必须小心，因为它会覆盖文件中已经存在的数据。

无论是文本文件还是二进制文件，你都可以使用 Python 的 write()方法。

```py
with open("example.txt",'w',encoding = 'utf-8') as f:
   f.write("Hello world.\n\n")
   f.write("This file has two lines.")
```

如果目录中不存在“example.txt ”,将使用写入的指定文本创建它。另一方面，如果文件与脚本存在于同一个目录中，则文件先前保存的数据将被删除并用新文本覆盖。

## **关闭文件**

在读取或写入文件后，您必须关闭文件。这是释放文件占用资源的唯一方法。

在 Python 中关闭一个文件很容易——你所要做的就是使用 close()方法。运行方法可以释放文件占用的内存。语法是:

```py
File_object.close()
```

要在读写后关闭“example.txt ”,只需运行以下代码即可:

```py
f.close()
```

close()方法有时会抛出异常，导致代码在文件没有关闭的情况下退出。

操作后关闭文件的更好方法是使用 try…finally 块。

```py
try:
   f = open("example.txt", encoding = 'utf-8')
   # perform operations
finally:
   f.close()
```

使用 try… finally 块可以确保打开的文件被正确关闭，而不管是否出现异常。

# **结论**

学习文件处理和复制/移动文件和目录将帮助你掌握 Python 中的文件操作。

接下来，看看我们关于使用 Python 使用进度条 [复制/移动文件的简单教程](https://www.pythoncentral.io/how-to-movecopy-a-file-or-directory-folder-with-a-progress-bar-in-python/) 。