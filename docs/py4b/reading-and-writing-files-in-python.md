# 用 Python 读写文件

> 原文：<https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python>

## 概观

当您使用 Python 时，不需要为了读写文件而导入库。它是在语言中自然处理的，尽管是以一种独特的方式。下面，我们概述了用 Python 读写文件的简单步骤。

首先你需要做的是使用 python 内置的 ***打开*** 文件函数得到一个 ***文件对象*** 。

***打开*** 功能打开一个文件。很简单。这是用 python 读写文件的第一步。

当你使用 ***打开*** 函数时，它返回一个叫做 ***的文件对象*** 。 ***文件对象*** 包含方法和属性，可用于收集关于您打开的文件的信息。它们也可以用来操作所述文件。

例如， ***文件对象*** 的 ***模式*** 属性告诉你文件是以哪种模式打开的。而 ***名称*** 属性告诉你文件的名称。

你必须明白一个 ***文件*** 和 ***文件对象*** 是两个完全独立却又相关的东西。

## 文件类型

您可能知道的文件在 Python 中略有不同。

例如，在 Windows 中，文件可以是由用户/操作系统操作、编辑或创建的任何项目。这意味着文件可以是图像、文本文档、可执行文件和 excel 文件等等。大多数文件都是通过保存在单独的文件夹中来组织的。

Python 中的文件分为文本文件和二进制文件，这两种文件类型之间的区别很重要。

文本文件由一系列行组成，每一行都包含一系列字符。这就是你所知道的代码或语法。

每一行都以一个特殊字符结束，称为 EOL 或**行尾**字符。有几种类型，但最常见的是逗号{，}或换行符。它结束当前行，并告诉解释器新的一行已经开始。

也可以使用反斜杠字符，它告诉解释器斜杠后面的下一个字符应该被当作新的一行。当您不想在文本本身而是在代码中开始一个新行时，这个字符很有用。

二进制文件是非文本文件的任何类型的文件。由于其性质，二进制文件只能由知道或理解文件结构的应用程序处理。换句话说，它们必须是可以读取和解释二进制的应用程序。

## 用 Python 读取文件

在 Python 中，使用 **open()** 方法读取文件。这是 Python 的内置方法之一，用于打开文件。

**open()** 函数有两个参数:文件名和文件打开模式。文件名指向文件在你电脑上的路径，而文件打开模式用来告诉 **open()** 函数我们计划如何与文件交互。

默认情况下，文件打开模式设置为只读，这意味着我们只有打开和检查文件内容的权限。

在我的电脑上有一个名为 PythonForBeginners 的文件夹。那个文件夹里有三个文件。一个是名为 emily_dickinson.txt 的文本文件，另外两个是 python 文件:read.py 和 write.py。

该文本文件包含以下由诗人艾米莉·狄金森所写的诗。也许我们正在做一个诗歌程序，并把我们的诗歌作为文件储存在电脑上。

从未成功的人认为成功是最甜蜜的。
*领悟一种甘露*
*需要最迫切的需要。*

*所有的紫主持人*
*中没有一个人今天接过了*
*的旗帜能把*
*的定义说得如此清晰的胜利*

*随着他的战败，*
*在谁的禁耳上*
*远处胜利的旋律*
*迸发出极度痛苦和清晰。*

在我们对诗歌文件的内容做任何事情之前，我们需要告诉 Python 打开它。read.py 文件包含阅读这首诗所需的所有 python 代码。

任何文本编辑器都可以用来编写代码。我使用的是 Atom 代码编辑器，这是我使用 python 时的首选编辑器。

![](img/e5b5567db48477b77cd05451a24b6ce4.png)



This screenshot shows my setup in Atom.

```py
# read.py
# loading a file with open()
myfile = open(“emily_dickinson.txt”)

# reading each line of the file and printing to the console
for line in myfile:
	print(line)
```

我用 Python 注释解释了代码中的每一步。点击这个链接了解更多关于什么是 [Python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)的信息。

上面的例子说明了如何在 Python 中使用一个简单的[循环来读取文件的内容。](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)

在读取文件时，Python 会在幕后处理这些麻烦。通过使用命令提示符或终端导航到该文件，然后键入“python”后跟文件名，来运行脚本。

**Windows 用户**:在命令提示符下使用 python 关键字之前，您需要设置环境变量。当您安装 Python 时，这应该是自动发生的，但如果没有，您可能需要手动进行。

```py
>python read.py
```

由 **open()** 方法提供的数据通常存储在一个新的变量中。在这个例子中，诗歌的内容存储在变量“myfile”中。

创建文件后，我们可以使用 for 循环读取文件中的每一行，并将其内容打印到命令行。

这是如何用 Python 打开文件的一个非常简单的例子，但是学生应该知道 **open()** 方法非常强大。对于一些项目来说，这将是用 Python 读写文件所需要的唯一东西。

## 用 Python 写文件

在我们可以用 Python 写一个文件之前，必须先用不同的文件打开模式打开它。我们可以通过提供带有特殊参数的 ***open()*** 方法来做到这一点。

在 [Python 中，使用 **open()** 方法写入文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)。您需要传递一个文件名和一个特殊字符，告诉 Python 我们打算写入该文件。

将以下代码添加到 write.py。我们将告诉 Python 查找名为“sample.txt”的文件，并用新消息覆盖其内容。

```py
# open the file in write mode
myfile = open(“sample.txt”,’w’)

myfile.write(“Hello from Python!”)
```

向 **open()** 方法传递‘w’告诉 Python 以写模式打开文件。在这种模式下，当写入新数据时，文件中已有的任何数据都会丢失。

如果文件不存在，Python 将创建一个新文件。在这种情况下，程序运行时将创建一个名为“sample.txt”的新文件。

使用命令提示符运行程序:

```py
>python write.py
```

Python 也可以在一个文件中写入多行。最简单的方法是使用 *writelines()* 方法。

```py
# open the file in write mode
myfile = open(“sample.txt”,’w’)

myfile.writelines(“Hello World!”,”We’re learning Python!”)

# close the file
myfile.close()
```

我们还可以使用特殊字符将多行写入一个文件:

```py
# open the file in write mode
myfile = open("poem.txt", 'w')

line1 = "Roses are red.\n"
line2 = "Violets are blue.\n"
line3 = "Python is great.\n"
line4 = "And so are you.\n"

myfile.write(line1 + line2 + line3 + line4)
```

使用[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)使得 Python 能够以多种方式保存文本数据。

然而，如果我们想避免覆盖文件中的数据，而是追加或更改它，我们必须使用另一种文件打开模式打开文件。

## 文件打开模式

默认情况下，Python 将以只读模式打开文件。如果我们想做除了读取文件之外的任何事情，我们需要手动告诉 Python 我们打算用它做什么。

*   r’–读取模式:这是 ***open()*** 的默认模式。文件被打开，指针位于文件内容的开头。
*   w '–写入模式:使用此模式将覆盖文件中的任何现有内容。如果给定的文件不存在，将创建一个新文件。
*   r+'–读/写模式:如果您需要同时读写文件，请使用此模式。
*   a '–追加模式:在这种模式下，用户可以追加数据，而不会覆盖文件中任何已经存在的数据。
*   a+'–追加和读取模式:在这种模式下，您可以读取和追加数据，而不会覆盖原始文件。
*   x '–独占创建模式:该模式仅用于创建新文件。如果您事先知道要写入的文件不存在，请使用此模式。

注意:这些例子假设用户正在处理文本文件类型。如果意图是读取或写入二进制文件类型，则必须向 **open()** 方法传递一个额外的参数:字符“b”。

```py
# binary files need a special argument: ‘b’
binary_file = open(“song_data.mp3”,’rb’)
song_data = binary_file.read()

# close the file
binary_file.close()
```

## **用 Python 关闭文件**

在 Python 中打开一个文件后，重要的是在完成后关闭它。关闭文件可以确保程序无法再访问其内容。

用 **close()** 方法关闭一个文件。

```py
# open a file
myfile = open(“poem.txt”)
# an array to store the contents of the file
lines = []
For line in myfile:
	lines.append(line)

# close the file
myfile.close()

For line in liens:
	print(line)
```

## **打开其他文件类型**

**open()** 方法可以读写许多不同的文件类型。我们已经看到了如何打开二进制文件和文本文件。Python 还可以打开图像，允许您查看和编辑它们的像素数据。

在 Python 可以打开图像文件之前，必须安装 **Pillow** 库(Python 图像库)。使用 pip 安装这个模块是最简单的。

```py
pip install Pillow
```

安装了 **Pillow** 后，Python 可以打开图像文件并读取其内容。

```py
From PIL import Image

# tell Pillow to open the image file
img = Image.open(“your_image_file.jpg”)
img.show()
img.close()
```

**Pillow** 库包含强大的图像编辑工具。这使得它成为最受欢迎的 Python 库之一。

## With 语句

您还可以使用带有语句的[来处理文件对象。它旨在当您处理代码时提供更清晰的语法和异常处理。这解释了为什么在适当的地方使用 with 语句是一种好的做法。](https://www.pythonforbeginners.com/files/with-statement-in-python)

使用这种方法的一个好处是，任何打开的文件都会在您完成后自动关闭。这样在清理过程中就不用太担心了。

使用 with 语句打开文件:

```py
***with open(“filename”) as file:* **
```

现在你明白了如何调用这个语句，让我们来看几个例子。

```py
***with open(“poem.txt”) as file: * **
***data = file.read()* **
***do something with data* **
```

使用此语句时，还可以调用其他方法。例如，您可以对文件对象执行类似循环的操作:

```py
**w*ith open(“poem.txt”) as f:* **
***for line in f:* **
***print line,* **
```

你还会注意到，在上面的例子中，我们没有使用“***”file . close()***”方法，因为 with 语句会在执行时自动调用它。这真的让事情变得简单多了，不是吗？

## 在文本文件中拆分行

作为最后一个例子，让我们探索一个独特的函数，它允许您从文本文件中分割行。这样做的目的是，每当解释器遇到一个空格字符时，就分割变量数据中包含的字符串。

但是仅仅因为我们要用它来在一个空格字符后分行，并不意味着这是唯一的方法。实际上，你可以使用任何字符来拆分你的文本，比如冒号。

执行此操作的代码(也使用 with 语句)是:

```py
***with* *open(**“**hello.text**”, “r”) as f:***
***data =* *f.readlines**()***

***for line in data:***
***words =* *line.split**()***
***print words***
```

如果您想使用冒号而不是空格来拆分文本，只需将 line.split()改为 line.split(":")。

其输出将是:

```py
***[“hello”, “world”, “how”, “are”, “you”, “today?”]***
***[“today”, “is”, “Saturday”]***
```

以这种方式显示单词的原因是因为它们以数组的形式存储和返回。使用拆分功能时，请务必记住这一点。

## **结论**

在 Python 中读写文件涉及到对 ***open()*** 方法的理解。通过利用这种方法的通用性，可以在 Python 中读取、写入和创建文件。

Python 文件可以是文本文件，也可以是二进制文件。还可以使用枕头模块打开和编辑图像数据。

一旦一个文件的数据被加载到 Python 中，实际上可以用它做的事情是无穷无尽的。程序员经常处理大量的文件，使用程序自动生成它们。

与任何课程一样，在所提供的空间内只能涵盖这么多内容。希望您已经掌握了足够的知识，可以开始用 Python 读写文件。

##### **更多阅读**

[官方 Python 文档——读写文件](https://docs.python.org/2/tutorial/inputoutput.html#reading-and-writing-files "python_readwrite")

[Python 文件处理备忘单](/cheatsheet/python-file-handling "files")