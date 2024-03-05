# 用 Python 创建和修改 PDF 文件

> 原文：<https://realpython.com/creating-modifying-pdf/>

知道如何在 Python 中创建和修改 PDF 文件真的很有用。 **PDF** 或 **P** 或表格 **D** 文档 **F** 格式，是在互联网上共享文档最常见的格式之一。[pdf](https://realpython.com/pdf-python/)可以包含文本、图像、表格、表单以及视频和动画等富媒体，所有这些都在一个文件中。

如此丰富的内容类型会使处理 pdf 变得困难。当打开一个 PDF 文件时，有许多不同种类的数据要解码！幸运的是，Python 生态系统有一些很棒的包，可以用来读取、操作和创建 PDF 文件。

**在本教程中，您将学习如何:**

*   **阅读 PDF 中的**文本
*   **将一个 PDF 文件分割成多个文件**
*   **串联**和**合并** PDF 文件
*   **旋转**和**裁剪 PDF 文件中的**页面
*   **用密码加密**和**解密** PDF 文件
*   **从头开始创建**PDF 文件

**注:**本教程改编自 [*Python 基础知识:Python 实用入门 3*](https://realpython.com/products/python-basics-book/) 中“创建和修改 PDF 文件”一章。

这本书使用 Python 内置的 [IDLE](https://realpython.com/python-idle/) 编辑器来创建和编辑 Python 文件，并与 Python shell 进行交互，因此在整个教程中你会偶尔看到对 IDLE 的引用。但是，从您选择的编辑器和环境中运行示例代码应该没有问题。

在这个过程中，您将有几次机会通过跟随示例来加深理解。您可以点击下面的链接下载示例中使用的材料:

**下载示例材料:** [单击此处获取您将在本教程中使用](https://realpython.com/bonus/create-modify-pdf/)学习创建和修改 PDF 文件的材料。

## 从 PDF 中提取文本

在本节中，您将学习如何阅读 PDF 文件并使用 [`PyPDF2`](https://pypi.org/project/PyPDF2/) 包提取文本。不过，在你这么做之前，你需要用`pip` 安装[:](https://realpython.com/what-is-pip/)

```py
$ python3 -m pip install PyPDF2
```

通过在终端中运行以下命令来验证安装:

```py
$ python3 -m pip show PyPDF2
Name: PyPDF2
Version: 1.26.0
Summary: PDF toolkit
Home-page: http://mstamy2.github.com/PyPDF2
Author: Mathieu Fenniak
Author-email: biziqe@mathieu.fenniak.net
License: UNKNOWN
Location: c:\\users\\david\\python38-32\\lib\\site-packages
Requires:
Required-by:
```

请特别注意版本信息。在撰写本文时，`PyPDF2`的最新版本是`1.26.0`。如果你有 IDLE open，那么你需要重启它才能使用`PyPDF2`包。

[*Remove ads*](/account/join/)

### 打开 PDF 文件

让我们先打开一个 PDF 文件，阅读一些相关信息。您将使用位于配套存储库中的`practice_files/`文件夹中的`Pride_and_Prejudice.pdf`文件。

打开 IDLE 的交互窗口，[从`PyPDF2`包中导入](https://realpython.com/lessons/import-statement/)类`PdfFileReader`:

>>>

```py
>>> from PyPDF2 import PdfFileReader
```

要创建一个新的`PdfFileReader`类的实例，您将需要您想要打开的 PDF 文件的[路径](https://realpython.com/read-write-files-python/#file-paths)。现在让我们使用`pathlib`模块来获取:

>>>

```py
>>> from pathlib import Path
>>> pdf_path = (
...     Path.home()
...     / "creating-and-modifying-pdfs"
...     / "practice_files"
...     / "Pride_and_Prejudice.pdf"
... )
```

`pdf_path` [变量](https://realpython.com/python-variables/)现在包含了简·奥斯汀的*傲慢与偏见*的 PDF 版本的路径。

**注意:**您可能需要更改`pdf_path`，使其对应于您计算机上`creating-and-modifying-pdfs/`文件夹的位置。

现在创建`PdfFileReader`实例:

>>>

```py
>>> pdf = PdfFileReader(str(pdf_path))
```

您将`pdf_path`转换为[字符串](https://realpython.com/python-data-types/#strings)，因为`PdfFileReader`不知道如何从`pathlib.Path`对象中读取。

回想一下[第 12 章](https://realpython.com/products/python-basics-book/)“文件输入和输出”，在程序终止之前[所有打开的文件都应该关闭](https://realpython.com/why-close-file-python/)。对象为你做了所有这些，所以你不需要担心打开或关闭 PDF 文件！

现在您已经创建了一个`PdfFileReader`实例，您可以使用它来收集关于 PDF 的信息。例如，`.getNumPages()`返回 PDF 文件中包含的页数:

>>>

```py
>>> pdf.getNumPages()
234
```

注意`.getNumPages()`是用 mixedCase 写的，而不是像 [PEP 8](https://pep8.org) 中推荐的 lower _ case _ with _ 下划线。记住，PEP 8 是一套指导方针，而不是规则。就 Python 而言，mixedCase 是完全可以接受的。

**注:** `PyPDF2`改编自`pyPdf`包。`pyPdf`写于 2005 年，仅在 PEP 8 出版后四年。

当时，许多 Python 程序员正在从 mixedCase 更常见的语言中迁移。

您还可以使用`.documentInfo`属性访问一些文档信息:

>>>

```py
>>> pdf.documentInfo
{'/Title': 'Pride and Prejudice, by Jane Austen', '/Author': 'Chuck',
'/Creator': 'Microsoft® Office Word 2007',
'/CreationDate': 'D:20110812174208', '/ModDate': 'D:20110812174208',
'/Producer': 'Microsoft® Office Word 2007'}
```

由`.documentInfo`返回的对象看起来像一个[字典](https://realpython.com/courses/dictionaries-python/)，但它实际上不是同一个东西。您可以将`.documentInfo`中的每个项目作为[属性](https://realpython.com/lessons/class-and-instance-attributes/)进行访问。

例如，要获得标题，使用`.title`属性:

>>>

```py
>>> pdf.documentInfo.title
'Pride and Prejudice, by Jane Austen'
```

`.documentInfo`对象包含 PDF **元数据**，它在创建 PDF 时设置。

`PdfFileReader`类提供了访问 PDF 文件中的数据所需的所有方法和属性。让我们来探索一下您可以用 PDF 文件做什么，以及如何做！

[*Remove ads*](/account/join/)

### 从页面中提取文本

PDF 页面在`PyPDF2`中用`PageObject`类表示。您可以使用`PageObject`实例与 PDF 文件中的页面进行交互。您不需要直接创建自己的`PageObject`实例。相反，你可以通过`PdfFileReader`对象的`.getPage()`方法来访问它们。

从单个 PDF 页面中提取文本有两个步骤:

1.  用`PdfFileReader.getPage()`得到一个`PageObject`。
2.  用`PageObject`实例的`.extractText()`方法将文本提取为字符串。

`Pride_and_Prejudice.pdf`有`234`页。每一页在`0`和`233`之间都有一个索引。通过将页面的索引传递给`PdfFileReader.getPage()`，可以获得代表特定页面的`PageObject`:

>>>

```py
>>> first_page = pdf.getPage(0)
```

`.getPage()`返回一个`PageObject`:

>>>

```py
>>> type(first_page)
<class 'PyPDF2.pdf.PageObject'>
```

您可以使用`PageObject.extractText()`提取页面文本:

>>>

```py
>>> first_page.extractText()
'\\n \\nThe Project Gutenberg EBook of Pride and Prejudice, by Jane
Austen\\n \\n\\nThis eBook is for the use of anyone anywhere at no cost
and with\\n \\nalmost no restrictions whatsoever.  You may copy it,
give it away or\\n \\nre\\n-\\nuse it under the terms of the Project
Gutenberg License included\\n \\nwith this eBook or online at
www.gutenberg.org\\n \\n \\n \\nTitle: Pride and Prejudice\\n \\n
\\nAuthor: Jane Austen\\n \\n \\nRelease Date: August 26, 2008
[EBook #1342]\\n\\n[Last updated: August 11, 2011]\\n \\n \\nLanguage:
Eng\\nlish\\n \\n \\nCharacter set encoding: ASCII\\n \\n \\n***
START OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\\n \\n
\\n \\n \\n \\nProduced by Anonymous Volunteers, and David Widger\\n
\\n \\n \\n \\n \\n \\n \\nPRIDE AND PREJUDICE \\n \\n \\nBy Jane
Austen \\n \\n\\n \\n \\nContents\\n \\n'
```

请注意，此处显示的输出已经过格式化，以更好地适应此页面。您在电脑上看到的输出格式可能会有所不同。

每个`PdfFileReader`对象都有一个`.pages`属性，您可以使用它来按顺序遍历 PDF 中的所有页面。

例如，下面的 [`for`循环](https://realpython.com/courses/python-for-loop/)打印*傲慢与偏见* PDF 中每一页的文本:

>>>

```py
>>> for page in pdf.pages:
...     print(page.extractText())
...
```

让我们结合你所学的一切，编写一个程序，从`Pride_and_Prejudice.pdf`文件中提取所有文本，并保存到`.txt`文件中。

### 将所有这些放在一起

在空闲状态下打开一个新的编辑器窗口，并键入以下代码:

```py
from pathlib import Path
from PyPDF2 import PdfFileReader

# Change the path below to the correct path for your computer.
pdf_path = (
    Path.home()
    / "creating-and-modifying-pdfs"
    / "practice-files"
    / "Pride_and_Prejudice.pdf"
)

# 1
pdf_reader = PdfFileReader(str(pdf_path))
output_file_path = Path.home() / "Pride_and_Prejudice.txt"

# 2
with output_file_path.open(mode="w") as output_file:
    # 3
    title = pdf_reader.documentInfo.title
    num_pages = pdf_reader.getNumPages()
    output_file.write(f"{title}\\nNumber of pages: {num_pages}\\n\\n")

    # 4
    for page in pdf_reader.pages:
        text = page.extractText()
        output_file.write(text)
```

让我们来分解一下:

1.  首先，将一个新的`PdfFileReader`实例分配给`pdf_reader` [变量](https://realpython.com/python-variables/)。您还创建了一个新的`Path`对象，它指向您的主目录中的文件`Pride_and_Prejudice.txt`，并将它赋给变量`output_file_path`。

2.  接下来，以写模式打开`output_file_path`，并将`.open()`返回的文件对象赋给变量`output_file`。你在[第 12 章](https://realpython.com/products/python-basics-book/)“文件输入和输出”中了解到的 [`with`语句](https://realpython.com/python-with-statement/)确保当`with`块退出时文件被关闭。

3.  然后，在`with`块中，使用`output_file.write()`将 PDF 标题和页数写入文本文件。

4.  最后，使用一个`for`循环迭代 PDF 中的所有页面。在循环的每一步，下一个`PageObject`被分配给`page`变量。用`page.extractText()`提取每页的文本，并写入`output_file`。

当您保存并运行该程序时，它将在您的主目录中创建一个名为`Pride_and_Prejudice.txt`的新文件，其中包含了`Pride_and_Prejudice.pdf`文档的全文。打开看看吧！

[*Remove ads*](/account/join/)

### 检查你的理解能力

展开下面的方框，检查您的理解程度:



在本文的配套存储库中的`practice_files/`文件夹中，有一个名为`zen.pdf`的文件。创建一个读取 PDF 的`PdfFileReader`实例，并使用它打印第一页的文本。

您可以展开下面的方框查看解决方案:



设置 PDF 文件的路径:

```py
# First, import the needed classes and libraries
from pathlib import Path
from PyPDF2 import PdfFileReader

# Then create a `Path` object to the PDF file.
# You might need to change this to match the path
# on your computer.
pdf_path = (
    Path.home()
    / "creating-and-modifying-pdfs"
    / "practice_files"
    / "zen.pdf"
)
```

现在您可以创建`PdfFileReader`实例了:

```py
pdf_reader = PdfFileReader(str(pdf_path))
```

记住`PdfFileReader`对象只能用路径字符串实例化，不能用`Path`对象实例化！

现在使用`.getPage()`来获得第一页:

```py
first_page = pdf_reader.getPage(0)
```

记住，页面索引是从 0 开始的！

然后使用`.extractText()`提取文本:

```py
text = first_page.extractText()
```

最后，[打印](https://realpython.com/python-print/)文本:

```py
print(text)
```

当你准备好了，你可以进入下一部分。

## 从 PDF 中提取页面

在上一节中，您学习了如何从 PDF 文件中提取所有文本并保存到一个`.txt`文件中。现在，您将了解如何从现有 PDF 中提取一个页面或一系列页面，并将它们保存到新的 PDF 中。

您可以使用`PdfFileWriter`创建一个新的 PDF 文件。让我们探索这个课程，学习使用`PyPDF2`创建 PDF 所需的步骤。

### 使用`PdfFileWriter`类

`PdfFileWriter`类创建新的 PDF 文件。在 IDLE 的交互窗口中，导入`PdfFileWriter`类并创建一个名为`pdf_writer`的新实例:

>>>

```py
>>> from PyPDF2 import PdfFileWriter
>>> pdf_writer = PdfFileWriter()
```

对象就像空白的 PDF 文件。在将它们保存到文件之前，您需要向它们添加一些页面。

继续给`pdf_writer`添加一个空白页:

>>>

```py
>>> page = pdf_writer.addBlankPage(width=72, height=72)
```

`width`和`height`参数是必需的，它们以称为**点**的单位确定页面的尺寸。一点等于 1/72 英寸，所以上面的代码为`pdf_writer`添加了一个一英寸见方的空白页。

`.addBlankPage()`返回一个新的`PageObject`实例，代表您添加到`PdfFileWriter`中的页面:

>>>

```py
>>> type(page)
<class 'PyPDF2.pdf.PageObject'>
```

在这个例子中，您已经将由`.addBlankPage()`返回的`PageObject`实例赋给了`page`变量，但是实际上您通常不需要这样做。也就是说，您通常调用`.addBlankPage()`而不将返回值赋给任何东西:

>>>

```py
>>> pdf_writer.addBlankPage(width=72, height=72)
```

要将`pdf_writer`的内容写入 PDF 文件，以二进制写入模式将文件对象传递给`pdf_writer.write()`:

>>>

```py
>>> from pathlib import Path
>>> with Path("blank.pdf").open(mode="wb") as output_file:
...     pdf_writer.write(output_file)
...
```

这将在当前工作目录中创建一个名为`blank.pdf`的新文件。如果你用 PDF 阅读器打开文件，比如 Adobe Acrobat，你会看到一个只有一个一英寸见方的空白页面的文档。

**技术细节:**注意，您保存 PDF 文件的方法是将 file 对象传递给`PdfFileWriter`对象的`.write()`方法，将*而不是*传递给 file 对象的`.write()`方法。

特别是，下面的代码将不起作用:

>>>

```py
>>> with Path("blank.pdf").open(mode="wb") as output_file:
...     output_file.write(pdf_writer)
```

对于许多新程序员来说，这种方法似乎是倒退的，所以确保你避免这个错误！

`PdfFileWriter`对象可以写入新的 PDF 文件，但是除了空白页之外，不能从头开始创建新的内容。

这似乎是一个大问题，但是在许多情况下，您不需要创建新的内容。通常，您会处理从 PDF 文件中提取的页面，这些文件是用`PdfFileReader`实例打开的。

**注意:**您将在下面的“从头创建 PDF 文件”一节中学习如何从头创建 PDF 文件

在上面的例子中，使用`PyPDF2`创建一个新的 PDF 文件有三个步骤:

1.  创建一个`PdfFileWriter`实例。
2.  向`PdfFileWriter`实例添加一个或多个页面。
3.  使用`PdfFileWriter.write()`写入文件。

随着您学习向`PdfFileWriter`实例添加页面的各种方法，您将会一遍又一遍地看到这种模式。

[*Remove ads*](/account/join/)

### 从 PDF 中提取单个页面

让我们重温一下你在上一节处理过的*傲慢与偏见* PDF。您将打开 PDF，提取第一页，并创建一个新的 PDF 文件，其中只包含一个提取的页面。

打开 IDLE 的交互窗口，从`PyPDF2`导入`PdfFileReader`和`PdfFileWriter`，从`pathlib`模块导入`Path`类；

>>>

```py
>>> from pathlib import Path
>>> from PyPDF2 import PdfFileReader, PdfFileWriter
```

现在用一个`PdfFileReader`实例打开`Pride_and_Prejudice.pdf`文件:

>>>

```py
>>> # Change the path to work on your computer if necessary
>>> pdf_path = (
...     Path.home()
...     / "creating-and-modifying-pdfs"
...     / "practice_files"
...     / "Pride_and_Prejudice.pdf"
... )
>>> input_pdf = PdfFileReader(str(pdf_path))
```

将索引`0`传递给`.getPage()`以获得代表 PDF 第一页的`PageObject`:

>>>

```py
>>> first_page = input_pdf.getPage(0)
```

现在创建一个新的`PdfFileWriter`实例，并用`.addPage()`将`first_page`添加到其中:

>>>

```py
>>> pdf_writer = PdfFileWriter()
>>> pdf_writer.addPage(first_page)
```

与`.addBlankPage()`一样，`.addPage()`方法将页面添加到`pdf_writer`对象的页面集中。不同的是，它需要一个已有的`PageObject`。

现在将`pdf_writer`的内容写入一个新文件:

>>>

```py
>>> with Path("first_page.pdf").open(mode="wb") as output_file:
...     pdf_writer.write(output_file)
...
```

您现在有一个新的 PDF 文件保存在您当前的工作目录中，名为`first_page.pdf`，它包含了`Pride_and_Prejudice.pdf`文件的封面。相当整洁！

### 从 PDF 中提取多个页面

让我们从`Pride_and_Prejudice.pdf`中提取第一章并保存到一个新的 PDF 中。

如果你用 PDF 浏览器打开`Pride_and_Prejudice.pdf`，那么你可以看到第一章在 PDF 的第二、第三和第四页。因为页面是从`0`开始索引的，所以您需要提取索引`1`、`2`和`3`处的页面。

您可以通过导入所需的类并打开 PDF 文件来设置所有内容:

>>>

```py
>>> from PyPDF2 import PdfFileReader, PdfFileWriter
>>> from pathlib import Path
>>> pdf_path = (
...     Path.home()
...     / "creating-and-modifying-pdfs"
...     / "practice_files"
...     / "Pride_and_Prejudice.pdf"
... )
>>> input_pdf = PdfFileReader(str(pdf_path))
```

您的目标是提取索引为`1`、`2`和`3`的页面，将它们添加到一个新的`PdfFileWriter`实例，然后将它们写入一个新的 PDF 文件。

一种方法是在从`1`开始到`3`结束的数字范围内循环，在循环的每一步提取页面并将其添加到`PdfFileWriter`实例:

>>>

```py
>>> pdf_writer = PdfFileWriter()
>>> for n in range(1, 4):
...     page = input_pdf.getPage(n)
...     pdf_writer.addPage(page)
...
```

因为`range(1, 4)`不包括右边的端点，所以循环迭代数字`1`、`2`和`3`。在循环的每一步，使用`.getPage()`提取当前索引处的页面，并使用`.addPage()`将其添加到`pdf_writer`。

现在`pdf_writer`有三页，你可以用`.getNumPages()`检查:

>>>

```py
>>> pdf_writer.getNumPages()
3
```

最后，您可以将提取的页面写入新的 PDF 文件:

>>>

```py
>>> with Path("chapter1.pdf").open(mode="wb") as output_file:
...     pdf_writer.write(output_file)
...
```

现在你可以打开当前工作目录中的`chapter1.pdf`文件来阅读*傲慢与偏见*的第一章。

另一种从 PDF 中提取多页的方法是利用`PdfFileReader.pages`支持[切片标记](https://realpython.com/lessons/indexing-and-slicing/)的事实。让我们使用`.pages`重复前面的例子，而不是在一个 [`range`对象](https://realpython.com/python-range/)上循环。

首先初始化一个新的`PdfFileWriter`对象:

>>>

```py
>>> pdf_writer = PdfFileWriter()
```

现在从开始于`1`结束于`4`的索引开始循环一段`.pages`:

>>>

```py
>>> for page in input_pdf.pages[1:4]:
...    pdf_writer.addPage(page)
...
```

请记住，切片中的值范围是从切片中第一个索引处的项目到切片中第二个索引处的项目，但不包括这两个项目。所以`.pages[1:4]`返回一个包含索引为`1`、`2`和`3`的页面的 iterable。

最后，将`pdf_writer`的内容写入输出文件:

>>>

```py
>>> with Path("chapter1_slice.pdf").open(mode="wb") as output_file:
...     pdf_writer.write(output_file)
...
```

现在打开当前工作目录中的`chapter1_slice.pdf`文件，并将其与通过循环`range`对象创建的`chapter1.pdf`文件进行比较。它们包含相同的页面！

有时你需要从 PDF 中提取每一页。您可以使用上面举例说明的方法来做到这一点，但是`PyPDF2`提供了一个快捷方式。`PdfFileWriter`实例有一个`.appendPagesFromReader()`方法，可以用来从`PdfFileReader`实例追加页面。

要使用`.appendPagesFromReader()`，向方法的`reader`参数传递一个`PdfFileReader`实例。例如，以下代码将*傲慢与偏见* PDF 中的每一页复制到`PdfFileWriter`实例中:

>>>

```py
>>> pdf_writer = PdfFileWriter()
>>> pdf_writer.appendPagesFromReader(pdf_reader)
```

`pdf_writer`现在包含了`pdf_reader`中的每一页！

[*Remove ads*](/account/join/)

### 检查你的理解能力

展开下面的方框，检查您的理解程度:



从`Pride_and_Prejudice.pdf`文件中提取最后一页，并将其保存到主目录中一个名为`last_page.pdf`的新文件中。

您可以展开下面的方框查看解决方案:



设置`Pride_and_Prejudice.pdf`文件的路径:

```py
# First, import the needed classes and libraries
from pathlib import Path
from PyPDF2 import PdfFileReader, PdfFileWriter

# Then create a `Path` object to the PDF file.
# You might need to change this to match the path
# on your computer.
pdf_path = (
    Path.home()
    / "creating-and-modifying-pdfs"
    / "practice_files"
    / "Pride_and_Prejudice.pdf"
)
```

现在您可以创建`PdfFileReader`实例了:

```py
pdf_reader = PdfFileReader(str(pdf_path))
```

记住`PdfFileReader`对象只能用路径字符串实例化，不能用`Path`对象实例化！

使用`.pages`属性获取 PDF 中所有页面的 iterable。最后一页可以用索引`-1`访问:

```py
last_page = pdf_reader.pages[-1]
```

现在您可以创建一个`PdfFileWriter`实例，并将最后一个页面添加到其中:

```py
pdf_writer = PdfFileWriter()
pdf_writer.addPage(last_page)
```

最后，将`pdf_writer`的内容写入主目录中的文件`last_page.pdf`:

```py
output_path = Path.home() / "last_page.pdf"
with output_path.open(mode="wb") as output_file:
    pdf_writer.write(output_file)
```

当你准备好了，你可以进入下一部分。

## 连接和合并 pdf 文件

处理 PDF 文件时的两个常见任务是将几个 PDF 连接并合并到一个文件中。

当您**连接**两个或更多 pdf 时，您将文件一个接一个地合并成一个文档。例如，一家公司可能会在月末将几份每日报告合并成一份月度报告。

**合并**两个 pdf 也会合并成一个文件。但是合并允许您在第一个 PDF 的特定页面之后插入它，而不是将第二个 PDF 连接到第一个 PDF 的末尾。然后，它将插入点之后的第一个 PDF 的所有页面推到第二个 PDF 的结尾。

在本节中，您将学习如何使用`PyPDF2`包的`PdfFileMerger`来连接和合并 pdf。

### 使用`PdfFileMerger`类

`PdfFileMerger`类很像您在上一节中了解的`PdfFileWriter`类。您可以使用这两个类来编写 PDF 文件。在这两种情况下，都将页添加到类的实例中，然后将它们写入文件。

两者的主要区别在于，`PdfFileWriter`只能将页面追加或连接到已经包含在编写器中的页面列表的末尾，而`PdfFileMerger`可以在任何位置插入或合并页面。

继续创建您的第一个`PdfFileMerger`实例。在 IDLE 的交互窗口中，键入以下代码以导入`PdfFileMerger`类并创建一个新实例:

>>>

```py
>>> from PyPDF2 import PdfFileMerger
>>> pdf_merger = PdfFileMerger()
```

对象第一次实例化时是空的。在对对象进行任何操作之前，您需要向对象添加一些页面。

有几种方法可以将页面添加到`pdf_merger`对象，使用哪一种取决于您需要完成的任务:

*   **`.append()`** 将现有 PDF 文档中的每一页连接到当前`PdfFileMerger`中页面的末尾。
*   **`.merge()`** 将现有 PDF 文档中的所有页面插入到`PdfFileMerger`中的特定页面之后。

在本节中，您将从`.append()`开始查看这两种方法。

[*Remove ads*](/account/join/)

### 将 pdf 与`.append()` 连接

`practice_files/`文件夹有一个名为`expense_reports`的子目录，其中包含名为 Peter Python 的雇员的三份费用报告。

Peter 需要将这三个 PDF 文件连接起来，作为一个 PDF 文件提交给他的雇主，这样他就可以报销一些与工作相关的费用。

首先，您可以使用`pathlib`模块获取`expense_reports/`文件夹中三份费用报告的`Path`对象列表:

>>>

```py
>>> from pathlib import Path
>>> reports_dir = (
...     Path.home()
...     / "creating-and-modifying-pdfs"
...     / "practice_files"
...     / "expense_reports"
... )
```

导入`Path`类后，您需要构建到`expense_reports/`目录的路径。请注意，您可能需要修改上面的代码，以便在您的计算机上获得正确的路径。

一旦将`expense_reports/`目录的路径分配给了`reports_dir`变量，您就可以使用`.glob()`来获取目录中 PDF 文件的路径。

看看目录中有什么:

>>>

```py
>>> for path in reports_dir.glob("*.pdf"):
...     print(path.name)
...
Expense report 1.pdf
Expense report 3.pdf
Expense report 2.pdf
```

列出了三个文件的名称，但它们没有按顺序排列。此外，您在计算机输出中看到的文件顺序可能与此处显示的输出不一致。

一般来说，`.glob()`返回的路径顺序是不确定的，所以你需要自己排序。您可以通过创建一个包含三个文件路径的列表，然后在该列表上调用 [`.sort()`](https://realpython.com/python-sort/) 来实现:

>>>

```py
>>> expense_reports = list(reports_dir.glob("*.pdf"))
>>> expense_reports.sort()
```

记住`.sort()`就地对列表进行排序，所以不需要将返回值赋给变量。在`.list()`被调用后，`expense_reports`列表将按文件名的字母顺序排序。

为了确认排序成功，再次循环`expense_reports`并打印出文件名:

>>>

```py
>>> for path in expense_reports:
...     print(path.name)
...
Expense report 1.pdf
Expense report 2.pdf
Expense report 3.pdf
```

看起来不错！

现在您可以连接这三个 pdf 文件。为此，您将使用`PdfFileMerger.append()`，它需要一个表示 PDF 文件路径的字符串参数。当您调用`.append()`时，PDF 文件中的所有页面都会被追加到`PdfFileMerger`对象中的页面集合中。

让我们来看看实际情况。首先，导入`PdfFileMerger`类并创建一个新实例:

>>>

```py
>>> from PyPDF2 import PdfFileMerger
>>> pdf_merger = PdfFileMerger()
```

现在遍历排序后的`expense_reports`列表中的路径，并将它们附加到`pdf_merger`:

>>>

```py
>>> for path in expense_reports:
...     pdf_merger.append(str(path))
...
```

注意，`expense_reports/`中的每个`Path`对象在被传递给`pdf_merger.append()`之前都被转换成一个带有`str()`的字符串。

将`expense_reports/`目录中的所有 PDF 文件连接到`pdf_merger`对象中，您需要做的最后一件事就是将所有内容写入一个输出 PDF 文件。`PdfFileMerger`实例有一个`.write()`方法，就像`PdfFileWriter.write()`一样工作。

以二进制写模式打开一个新文件，然后将 file 对象传递给`pdf_merge.write()`方法:

>>>

```py
>>> with Path("expense_reports.pdf").open(mode="wb") as output_file:
...     pdf_merger.write(output_file)
...
```

您现在在当前工作目录中有一个名为`expense_reports.pdf`的 PDF 文件。用 PDF 阅读器打开它，您会发现所有三份费用报告都在同一个 PDF 文件中。

[*Remove ads*](/account/join/)

### 使用`.merge()`和合并 pdf

要合并两个或多个 pdf，请使用`PdfFileMerger.merge()`。该方法类似于`.append()`，除了您必须指定在输出 PDF 中的什么位置插入您正在合并的 PDF 中的所有内容。

看一个例子。Goggle，Inc .准备了一份季度报告，但忘记包括目录。Peter Python 注意到了这个错误，并很快创建了一个缺少目录的 PDF。现在，他需要将 PDF 文件合并到原始报告中。

报告 PDF 和目录 PDF 都可以在`practice_files`文件夹的`quarterly_report/`子文件夹中找到。报告在名为`report.pdf`的文件中，目录在名为`toc.pdf`的文件中。

在 IDLE 的交互窗口中，导入`PdfFileMerger`类，为`report.pdf`和`toc.pdf`文件创建`Path`对象:

>>>

```py
>>> from pathlib import Path
>>> from PyPDF2 import PdfFileMerger
>>> report_dir = (
...     Path.home()
...     / "creating-and-modifying-pdfs"
...     / "practice_files"
...     / "quarterly_report"
... )
>>> report_path = report_dir / "report.pdf"
>>> toc_path = report_dir / "toc.pdf"
```

您要做的第一件事是使用`.append()`将报告 PDF 附加到一个新的`PdfFileMerger`实例:

>>>

```py
>>> pdf_merger = PdfFileMerger()
>>> pdf_merger.append(str(report_path))
```

现在`pdf_merger`中已经有了一些页面，您可以将目录 PDF 合并到它的正确位置。如果您用 PDF 阅读器打开`report.pdf`文件，那么您会看到报告的第一页是一个标题页。第二个是简介，其余的页面包含不同的报告部分。

您希望在标题页之后、简介部分之前插入目录。由于 PDF 页面索引从`PyPDF2`中的`0`开始，您需要在索引`0`处的页面之后和索引`1`处的页面之前插入目录。

为此，用两个参数调用`pdf_merger.merge()`:

1.  整数`1`，表示应该插入目录的页面索引
2.  包含目录的 PDF 文件路径的字符串

看起来是这样的:

>>>

```py
>>> pdf_merger.merge(1, str(toc_path))
```

目录 PDF 中的每一页都在索引`1`处的页面之前*插入。因为目录 PDF 只有一页，所以它被插入到索引`1`处。当前在索引`1`的页面然后被转移到索引`2`。当前在索引`2`的页面被转移到索引`3`，等等。*

现在将合并的 PDF 写入输出文件:

>>>

```py
>>> with Path("full_report.pdf").open(mode="wb") as output_file:
...     pdf_merger.write(output_file)
...
```

您现在在当前工作目录中有一个`full_report.pdf`文件。用 PDF 阅读器打开它，检查目录是否插入正确的位置。

连接和合并 pdf 是常见的操作。虽然本节中的示例确实有些做作，但是您可以想象一个程序对于合并成千上万的 pdf 或自动化日常任务是多么有用，否则这些任务将需要花费大量的时间来完成。

### 检查你的理解能力

展开下面的方框，检查您的理解程度:



在本文的配套存储库中的`practice_files/`文件夹中，有两个名为`merge1.pdf`和`merge2.pdf`的文件。

使用一个`PdfFileMerge`实例，通过`.append()`连接两个文件。如果您的计算机的主目录，将连接的 pdf 保存到一个名为`concatenated.pdf`的新文件中。

您可以展开下面的方框查看解决方案:



设置 PDF 文件的路径:

```py
# First, import the needed classes and libraries
from pathlib import Path
from PyPDF2 import PdfFileMerger

# Then create a `Path` objects to the PDF files.
# You might need to change this to match the path
# on your computer.
BASE_PATH = (
    Path.home()
    / "creating-and-modifying-pdfs"
    / "practice_files"
)

pdf_paths = [BASE_PATH / "merge1.pdf", BASE_PATH / "merge2.pdf"]
```

现在您可以创建`PdfFileMerger`实例了:

```py
pdf_merger = PdfFileMerger()
```

现在将每个文件的`pdf_paths`和`.append()`中的路径循环到`pdf_merger`:

```py
for path in pdf_paths:
    pdf_merger.append(str(path))
```

最后，将`pdf_merger`的内容写入主目录中名为`concatenated.pdf`的文件:

```py
output_path = Path.home() / "concatenated.pdf"
with output_path.open(mode="wb") as output_file:
    pdf_merger.write(output_file)
```

当你准备好了，你可以进入下一部分。

[*Remove ads*](/account/join/)

## 旋转和裁剪 PDF 页面

到目前为止，您已经学习了如何从 PDF 中提取文本和页面，以及如何连接和合并两个或多个 PDF 文件。这些都是 pdf 的常见操作，但是`PyPDF2`还有许多其他有用的特性。

**注:**本教程改编自 [*Python 基础知识:Python 实用入门 3*](https://realpython.com/products/python-basics-book/) 中“创建和修改 PDF 文件”一章。如果你喜欢你正在阅读的东西，那么一定要看看这本书的其余部分。

在本节中，您将学习如何旋转和裁剪 PDF 文件中的页面。

### 旋转页面

您将从学习如何翻页开始。对于这个例子，您将使用`practice_files`文件夹中的`ugly.pdf`文件。这个`ugly.pdf`文件包含了安徒生的*丑小鸭*的可爱版本，除了每一个奇数页都被逆时针旋转了 90 度。

让我们解决这个问题。在一个新的空闲交互窗口中，开始从`PyPDF2`导入`PdfFileReader`和`PdfFileWriter`类，以及从`pathlib`模块导入`Path`类:

>>>

```py
>>> from pathlib import Path
>>> from PyPDF2 import PdfFileReader, PdfFileWriter
```

现在为`ugly.pdf`文件创建一个`Path`对象:

>>>

```py
>>> pdf_path = (
...     Path.home()
...     / "creating-and-modifying-pdfs"
...     / "practice_files"
...     / "ugly.pdf"
... )
```

最后，创建新的`PdfFileReader`和`PdfFileWriter`实例:

>>>

```py
>>> pdf_reader = PdfFileReader(str(pdf_path))
>>> pdf_writer = PdfFileWriter()
```

您的目标是使用`pdf_writer`创建一个新的 PDF 文件，其中所有页面都具有正确的方向。PDF 中偶数页的方向已经正确，但是奇数页逆时针旋转了 90 度。

要纠正这个问题，您将使用`PageObject.rotateClockwise()`。该方法采用一个整数参数，以度数为单位，将页面顺时针旋转该度数。例如，`.rotateClockwise(90)`顺时针旋转 PDF 页面 90 度。

**注:**除了`.rotateClockwise()`，`PageObject`类还有逆时针旋转页面的`.rotateCounterClockwise()`。

有几种方法可以在 PDF 中旋转页面。我们将讨论做这件事的两种不同方法。它们都依赖于`.rotateClockwise()`，但是它们采用不同的方法来决定哪些页面被旋转。

第一种技术是遍历 PDF 中页面的索引，并检查每个索引是否对应于需要旋转的页面。如果是这样，那么您将调用`.rotateClockwise()`来旋转页面，然后将页面添加到`pdf_writer`。

看起来是这样的:

>>>

```py
>>> for n in range(pdf_reader.getNumPages()):
...     page = pdf_reader.getPage(n)
...     if n % 2 == 0:
...         page.rotateClockwise(90)
...     pdf_writer.addPage(page)
...
```

请注意，如果索引是偶数，页面会旋转。这可能看起来很奇怪，因为 PDF 中奇数页是旋转不正确的页面。但是，PDF 中的页码以`1`开始，而页面索引以`0`开始。这意味着奇数编号的 PDF 页面具有偶数索引。

如果这让你头晕，不要担心！即使在多年处理这类事情之后，职业程序员仍然会被这类事情绊倒！

**注意:**当你执行上面的`for`循环时，你会在 IDLE 的交互窗口看到一堆输出。这是因为`.rotateClockwise()`返回了一个`PageObject`实例。

您现在可以忽略这个输出。当你从 IDLE 的编辑器窗口执行程序时，这个输出是不可见的。

现在您已经旋转了 PDF 中的所有页面，您可以将`pdf_writer`的内容写入一个新文件，并检查一切是否正常:

>>>

```py
>>> with Path("ugly_rotated.pdf").open(mode="wb") as output_file:
...     pdf_writer.write(output_file)
...
```

现在，在当前工作目录中应该有一个名为`ugly_rotated.pdf`的文件，来自`ugly.pdf`文件的页面都被正确旋转。

您刚刚使用的旋转`ugly.pdf`文件中页面的方法的问题是，它依赖于提前知道哪些页面需要旋转。在现实世界中，浏览整个 PDF 并注意要旋转哪些页面是不现实的。

事实上，您可以在没有先验知识的情况下确定哪些页面需要旋转。嗯，有时候你可以。

让我们看看如何从一个新的`PdfFileReader`实例开始:

>>>

```py
>>> pdf_reader = PdfFileReader(str(pdf_path))
```

您需要这样做，因为您通过旋转页面改变了旧的`PdfFileReader`实例中的页面。所以，通过创建一个新的实例，你可以从头开始。

实例维护一个包含页面信息的值字典:

>>>

```py
>>> pdf_reader.getPage(0)
{'/Contents': [IndirectObject(11, 0), IndirectObject(12, 0),
IndirectObject(13, 0), IndirectObject(14, 0), IndirectObject(15, 0),
IndirectObject(16, 0), IndirectObject(17, 0), IndirectObject(18, 0)],
'/Rotate': -90, '/Resources': {'/ColorSpace': {'/CS1':
IndirectObject(19, 0), '/CS0': IndirectObject(19, 0)}, '/XObject':
{'/Im0': IndirectObject(21, 0)}, '/Font': {'/TT1':
IndirectObject(23, 0), '/TT0': IndirectObject(25, 0)}, '/ExtGState':
{'/GS0': IndirectObject(27, 0)}}, '/CropBox': [0, 0, 612, 792],
'/Parent': IndirectObject(1, 0), '/MediaBox': [0, 0, 612, 792],
'/Type': '/Page', '/StructParents': 0}
```

呀！混杂在这些看起来毫无意义的东西中的是一个名为`/Rotate`的键，您可以在上面的第四行输出中看到它。这个键的值是`-90`。

您可以使用下标符号访问`PageObject`上的`/Rotate`键，就像您可以访问 Python `dict`对象一样:

>>>

```py
>>> page = pdf_reader.getPage(0)
>>> page["/Rotate"]
-90
```

如果您查看`pdf_reader`中第二页的`/Rotate`键，您会看到它的值为`0`:

>>>

```py
>>> page = pdf_reader.getPage(1)
>>> page["/Rotate"]
0
```

这意味着索引`0`处的页面旋转值为`-90`度。换句话说，它逆时针旋转了 90 度。索引`1`处的页面旋转值为`0`，因此根本没有旋转。

如果使用`.rotateClockwise()`旋转第一页，则`/Rotate`的值从`-90`变为`0`:

>>>

```py
>>> page = pdf_reader.getPage(0)
>>> page["/Rotate"]
-90
>>> page.rotateClockwise(90)
>>> page["/Rotate"]
0
```

现在您已经知道如何检查`/Rotate`键，您可以使用它来旋转`ugly.pdf`文件中的页面。

你需要做的第一件事是重新初始化你的`pdf_reader`和`pdf_writer`对象，这样你就有了一个新的开始:

>>>

```py
>>> pdf_reader = PdfFileReader(str(pdf_path))
>>> pdf_writer = PdfFileWriter()
```

现在编写一个循环，遍历`pdf_reader.pages` iterable 中的页面，检查`/Rotate`的值，如果该值为`-90`，则旋转页面:

>>>

```py
>>> for page in pdf_reader.pages:
...     if page["/Rotate"] == -90:
...         page.rotateClockwise(90)
...     pdf_writer.addPage(page)
...
```

这个循环不仅比第一个解决方案中的循环稍短，而且它不依赖于需要旋转哪些页面的任何先验知识。您可以使用这样的循环来旋转任何 PDF 中的页面，而不必打开它查看。

要完成解决方案，将`pdf_writer`的内容写入一个新文件:

>>>

```py
>>> with Path("ugly_rotated2.pdf").open(mode="wb") as output_file:
...     pdf_writer.write(output_file)
...
```

现在您可以打开当前工作目录中的`ugly_rotated2.pdf`文件，并将其与您之前生成的`ugly_rotated.pdf`文件进行比较。它们应该看起来一样。

**注意:**关于`/Rotate`键的一个警告:它不能保证存在于页面中。

如果`/Rotate`键不存在，那么通常意味着页面没有被旋转。然而，这并不总是一个安全的假设。

如果一个`PageObject`没有`/Rotate`键，那么当你试图访问它的时候会出现一个 [`KeyError`](https://realpython.com/python-keyerror/) 。你可以用一个`try...except`块来[捕捉这个异常。](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions)

`/Rotate`的值可能不总是你所期望的。例如，如果您将页面逆时针旋转 90 度来扫描纸质文档，那么 PDF 的内容将会旋转。然而，`/Rotate`键可能具有值`0`。

这是使处理 PDF 文件令人沮丧的许多怪癖之一。有时你只需要在 PDF 阅读器程序中打开一个 PDF，然后手动解决问题。

[*Remove ads*](/account/join/)

### 裁剪页面

pdf 的另一个常见操作是裁剪页面。您可能需要这样做来将单个页面分割成多个页面，或者只提取页面的一小部分，例如签名或图形。

例如，`practice_files`文件夹包含一个名为`half_and_half.pdf`的文件。这个 PDF 包含了安徒生的*小美人鱼*的一部分。

此 PDF 中的每一页都有两栏。让我们把每一页分成两页，每一栏一页。

首先，从`PyPDF2`导入`PdfFileReader`和`PdfFileWriter`类，从`pathlib`模块导入`Path`类:

>>>

```py
>>> from pathlib import Path
>>> from PyPDF2 import PdfFileReader, PdfFileWriter
```

现在为`half_and_half.pdf`文件创建一个`Path`对象:

>>>

```py
>>> pdf_path = (
...     Path.home()
...     / "creating-and-modifying-pdfs"
...     / "practice_files"
...     / "half_and_half.pdf"
... )
```

接下来，创建一个新的`PdfFileReader`对象并获取 PDF 的第一页:

>>>

```py
>>> pdf_reader = PdfFileReader(str(pdf_path))
>>> first_page = pdf_reader.getPage(0)
```

要裁剪页面，您首先需要对页面的结构有更多的了解。像`first_page`这样的`PageObject`实例有一个`.mediaBox`属性，代表一个定义页面边界的矩形区域。

您可以使用 IDLE 的交互窗口浏览`.mediaBox`，然后使用它裁剪页面:

>>>

```py
>>> first_page.mediaBox
RectangleObject([0, 0, 792, 612])
```

`.mediaBox`属性返回一个`RectangleObject`这个对象在`PyPDF2`包中定义，代表页面上的一个矩形区域。

输出中的列表`[0, 0, 792, 612]`定义了矩形区域。前两个数字是矩形左下角的 x 和 y 坐标。第三和第四个数字分别代表矩形的宽度和高度。所有数值的单位都是磅，等于 1/72 英寸。

`RectangleObject([0, 0, 792, 612])`表示一个矩形区域，左下角为原点，宽度为`792`点，即 11 英寸，高度为 612 点，即 8.5 英寸。这是一个标准信纸大小的横向页面的尺寸，用于《小美人鱼的 PDF 示例。纵向的信纸大小的 PDF 页面将返回输出`RectangleObject([0, 0, 612, 792])`。

一个`RectangleObject`有四个返回矩形角坐标的属性:`.lowerLeft`、`.lowerRight`、`.upperLeft`和`.upperRight`。就像宽度和高度值一样，这些坐标以磅为单位给出。

你可以使用这四个属性来获得`RectangleObject`的每个角的坐标:

>>>

```py
>>> first_page.mediaBox.lowerLeft
(0, 0)
>>> first_page.mediaBox.lowerRight
(792, 0)
>>> first_page.mediaBox.upperLeft
(0, 612)
>>> first_page.mediaBox.upperRight
(792, 612)
```

每个属性返回一个 [`tuple`](https://realpython.com/python-lists-tuples/#python-tuples) 包含指定角的坐标。您可以像访问任何其他 Python 元组一样，使用方括号访问各个坐标:

>>>

```py
>>> first_page.mediaBox.upperRight[0]
792
>>> first_page.mediaBox.upperRight[1]
612
```

您可以通过给一个属性分配一个新的元组来改变一个`mediaBox`的坐标:

>>>

```py
>>> first_page.mediaBox.upperLeft = (0, 480)
>>> first_page.mediaBox.upperLeft
(0, 480)
```

当您更改`.upperLeft`坐标时，`.upperRight`属性会自动调整以保持矩形形状:

>>>

```py
>>> first_page.mediaBox.upperRight
(792, 480)
```

当您更改由`.mediaBox`返回的`RectangleObject`的坐标时，您有效地裁剪了页面。`first_page`对象现在只包含新`RectangleObject`边界内的信息。

继续将裁剪后的页面写入新的 PDF 文件:

>>>

```py
>>> pdf_writer = PdfFileWriter()
>>> pdf_writer.addPage(first_page)
>>> with Path("cropped_page.pdf").open(mode="wb") as output_file:
...     pdf_writer.write(output_file)
...
```

如果你打开当前工作目录中的`cropped_page.pdf`文件，你会看到页面的顶部已经被移除。

如何裁剪页面，以便只看到页面左侧的文本？你需要将页面的水平尺寸减半。您可以通过改变`.mediaBox`对象的`.upperRight`坐标来实现这一点。让我们看看它是如何工作的。

首先，您需要获得新的`PdfFileReader`和`PdfFileWriter`对象，因为您刚刚修改了`pdf_reader`中的第一页并将其添加到`pdf_writer`:

>>>

```py
>>> pdf_reader = PdfFileReader(str(pdf_path))
>>> pdf_writer = PdfFileWriter()
```

现在获取 PDF 的第一页:

>>>

```py
>>> first_page = pdf_reader.getPage(0)
```

这一次，让我们使用第一页的副本，这样您刚刚提取的页面保持不变。您可以通过从 Python 的标准库中导入`copy`模块并使用`deepcopy()`来制作页面的副本:

>>>

```py
>>> import copy
>>> left_side = copy.deepcopy(first_page)
```

现在你可以改变`left_side`而不改变`first_page`的属性。这样，你可以稍后使用`first_page`来提取页面右侧的文本。

现在你需要做一点数学。您已经知道需要将`.mediaBox`的右上角移动到页面的顶部中央。为此，您将创建一个新的`tuple`，其第一个组件等于原始值的一半，并将其赋给`.upperRight`属性。

首先，获取`.mediaBox`右上角的当前坐标。

>>>

```py
>>> current_coords = left_side.mediaBox.upperRight
```

然后创建一个新的`tuple`，其第一个坐标是当前坐标的一半，第二个坐标与原始坐标相同:

>>>

```py
>>> new_coords = (current_coords[0] / 2, current_coords[1])
```

最后，将新坐标分配给`.upperRight`属性:

>>>

```py
>>> left_side.mediaBox.upperRight = new_coords
```

现在，您已经裁剪了原始页面，只包含左侧的文本！接下来让我们提取页面的右侧。

首先获取`first_page`的新副本:

>>>

```py
>>> right_side = copy.deepcopy(first_page)
```

移动`.upperLeft`角而不是`.upperRight`角:

>>>

```py
>>> right_side.mediaBox.upperLeft = new_coords
```

这会将左上角设置为提取页面左侧时将右上角移动到的相同坐标。所以，`right_side.mediaBox`现在是一个矩形，它的左上角在页面的顶部中心，它的右上角在页面的右上角。

最后，将`left_side`和`right_side`页面添加到`pdf_writer`，并将其写入一个新的 PDF 文件:

>>>

```py
>>> pdf_writer.addPage(left_side)
>>> pdf_writer.addPage(right_side)
>>> with Path("cropped_pages.pdf").open(mode="wb") as output_file:
...     pdf_writer.write(output_file)
...
```

现在用 PDF 阅读器打开`cropped_pages.pdf`文件。您应该看到一个有两页的文件，第一页包含原始第一页左侧的文本，第二页包含原始第二页右侧的文本。

[*Remove ads*](/account/join/)

### 检查你的理解能力

展开下面的方框，检查您的理解程度:



在本文的配套存储库中的`practice_files/`文件夹中，有一个名为`split_and_rotate.pdf`的文件。

在您计算机的主目录中创建一个名为`rotated.pdf`的新文件，其中包含来自`split_and_rotate.pdf`的所有页面，但是每一页都逆时针旋转 90 度。

您可以展开下面的方框查看解决方案:



设置 PDF 文件的路径:

```py
# First, import the needed classes and libraries
from pathlib import Path
from PyPDF2 import PdfFileReader

# Then create a `Path` object to the PDF file.
# You might need to change this to match the path
# on your computer.
pdf_path = (
    Path.home()
    / "creating-and-modifying-pdfs"
    / "practice_files"
    / "split_and_rotate.pdf"
)
```

现在您可以创建`PdfFileReader`和`PdfFileWriter`实例:

```py
pdf_reader = PdfFileReader(str(pdf_path))
pdf_writer = PdfFileWriter()
```

循环浏览`pdf_reader`中的页面，使用`.rotateCounterClockwise()`将其全部旋转 90 度，并添加到`pdf_writer`:

```py
for page in pdf_reader.pages:
    rotated_page = page.rotateCounterClockwise(90)
    pdf_writer.addPage(rotated_page)
```

最后，将`pdf_writer`的内容写入计算机主目录中名为`rotated.pdf`的文件:

```py
output_path = Path.home() / "rotated.pdf"
with output_path.open(mode="wb") as output_file:
    pdf_writer.write(output_file)
```

## 加密和解密 pdf

有时 PDF 文件受密码保护。使用`PyPDF2`包，您可以处理加密的 PDF 文件，并为现有的 PDF 文件添加密码保护。

**注:**本教程改编自 [*Python 基础知识:Python 实用入门 3*](https://realpython.com/products/python-basics-book/) 中“创建和修改 PDF 文件”一章。如果你喜欢你正在阅读的东西，那么一定要看看这本书的其余部分。

### 加密 pdf

您可以使用`PdfFileWriter()`实例的`.encrypt()`方法为 PDF 文件添加密码保护。它有两个主要参数:

1.  **`user_pwd`** 设置用户密码。这允许打开和阅读 PDF 文件。
2.  **`owner_pwd`** 设置所有者密码。这使得打开 PDF 没有任何限制，包括编辑。

让我们使用`.encrypt()`为 PDF 文件添加密码。首先，打开`practice_files`目录下的`newsletter.pdf`文件:

>>>

```py
>>> from pathlib import Path
>>> from PyPDF2 import PdfFileReader, PdfFileWriter
>>> pdf_path = (
...     Path.home()
...     / "creating-and-modifying-pdfs"
...     / "practice_files"
...     / "newsletter.pdf"
... )
>>> pdf_reader = PdfFileReader(str(pdf_path))
```

现在创建一个新的`PdfFileWriter`实例，并将来自`pdf_reader`的页面添加到其中:

>>>

```py
>>> pdf_writer = PdfFileWriter()
>>> pdf_writer.appendPagesFromReader(pdf_reader)
```

接下来，用`pdf_writer.encrypt()`添加密码`"SuperSecret"`:

>>>

```py
>>> pdf_writer.encrypt(user_pwd="SuperSecret")
```

当您只设置了`user_pwd`时，`owner_pwd`参数默认为相同的字符串。因此，上面的代码行设置了用户和所有者密码。

最后，将加密的 PDF 写到主目录中的输出文件`newsletter_protected.pdf`:

>>>

```py
>>> output_path = Path.home() / "newsletter_protected.pdf"
>>> with output_path.open(mode="wb") as output_file:
...     pdf_writer.write(output_file)
```

当您使用 PDF 阅读器打开 PDF 时，系统会提示您输入密码。输入`"SuperSecret"`打开 PDF。

如果您需要为 PDF 设置单独的所有者密码，那么将第二个字符串传递给`owner_pwd`参数:

>>>

```py
>>> user_pwd = "SuperSecret"
>>> owner_pwd = "ReallySuperSecret"
>>> pdf_writer.encrypt(user_pwd=user_pwd, owner_pwd=owner_pwd)
```

在本例中，用户密码是`"SuperSecret"`，所有者密码是`"ReallySuperSecret"`。

当您使用密码加密 PDF 文件并试图打开它时，您必须提供密码才能查看其内容。这种保护扩展到在 Python 程序中读取 PDF。接下来我们来看看如何用`PyPDF2`解密 PDF 文件。

### 解密 pdf 文件

要解密加密的 PDF 文件，请使用`PdfFileReader`实例的`.decrypt()`方法。

`.decrypt()`有一个名为`password`的参数，可以用来提供解密的密码。您在打开 PDF 时拥有的权限取决于您传递给`password`参数的参数。

让我们打开您在上一节中创建的加密的`newsletter_protected.pdf`文件，并使用`PyPDF2`来解密它。

首先，用受保护 PDF 的路径创建一个新的`PdfFileReader`实例:

>>>

```py
>>> from pathlib import Path
>>> from PyPDF2 import PdfFileReader, PdfFileWriter
>>> pdf_path = Path.home() / "newsletter_protected.pdf"
>>> pdf_reader = PdfFileReader(str(pdf_path))
```

在您解密 PDF 之前，请检查如果您尝试获取第一页会发生什么:

>>>

```py
>>> pdf_reader.getPage(0)
Traceback (most recent call last):
  File "/Users/damos/github/realpython/python-basics-exercises/venv/
 lib/python38-32/site-packages/PyPDF2/pdf.py", line 1617, in getObject
    raise utils.PdfReadError("file has not been decrypted")
PyPDF2.utils.PdfReadError: file has not been decrypted
```

出现一个`PdfReadError`异常，通知您 PDF 文件还没有被解密。

**注:**上述[追溯](https://realpython.com/python-traceback/)已被缩短以突出重要部分。你在电脑上看到的[回溯](https://realpython.com/courses/python-traceback/)会更长。

现在开始解密文件:

>>>

```py
>>> pdf_reader.decrypt(password="SuperSecret")
1
```

`.decrypt()`返回一个表示解密成功的整数:

*   **`0`** 表示密码不正确。
*   **`1`** 表示用户密码匹配。
*   **`2`** 表示主人密码被匹配。

解密文件后，您可以访问 PDF 的内容:

>>>

```py
>>> pdf_reader.getPage(0)
{'/Contents': IndirectObject(7, 0), '/CropBox': [0, 0, 612, 792],
'/MediaBox': [0, 0, 612, 792], '/Parent': IndirectObject(1, 0),
'/Resources': IndirectObject(8, 0), '/Rotate': 0, '/Type': '/Page'}
```

现在你可以提取文本和作物或旋转页面到你的心的内容！

### 检查你的理解能力

展开下面的方框，检查您的理解程度:



在本文的配套存储库中的`practice_files/`文件夹中，有一个名为`top_secret.pdf`的文件。

使用`PdfFileWriter.encrypt()`，用用户密码`Unguessable`加密文件。将加密文件保存为电脑主目录中的`top_secret_encrypted.pdf`。

您可以展开下面的方框查看解决方案:



设置 PDF 文件的路径:

```py
# First, import the needed classes and libraries
from pathlib import Path
from PyPDF2 import PdfFileReader

# Then create a `Path` object to the PDF file.
# You might need to change this to match the path
# on your computer.
pdf_path = (
    Path.home()
    / "creating-and-modifying-pdfs"
    / "practice_files"
    / "top_secret.pdf"
)
```

现在创建`PdfFileReader`和`PdfFileWriter`实例:

```py
pdf_reader = PdfFileReader(str(pdf_path))
pdf_writer = PdfFileWriter()
```

您可以使用`.appendPagesFromReader()`添加从`pdf_reader`到`pdf_writer`的所有页面:

```py
pdf_writer.appendPagesFromReader(pdf_reader)
```

现在使用`encrypt()`将用户密码设置为`"Unguessable"`:

```py
pdf_writer.encrypt(user_pwd="Unguessable")
```

最后，将`pdf_writer`的内容写入计算机主目录中名为`top_secret_encrypted.pdf`的文件:

```py
output_path = Path.home() / "top_secret_encrypted.pdf"
with output_path.open(mode="wb") as output_file:
    pdf_writer.write(output_file)
```

## 从头开始创建 PDF 文件

`PyPDF2`包非常适合阅读和修改现有的 PDF 文件，但是它有一个主要的限制:你不能用它来创建一个新的 PDF 文件。在本节中，您将使用 [ReportLab 工具包](http://www.reportlab.com/software/opensource/rl-toolkit/)从头开始生成 PDF 文件。

ReportLab 是用于创建 pdf 的全功能解决方案。有一个付费使用的商业版本，但也有一个功能有限的开源版本。

**注意:**这一部分并不是对 ReportLab 的详尽介绍，而是一个可能的示例。

更多示例，请查看 ReportLab 的[代码片段页面](http://www.reportlab.com/snippets/)。

### 安装`reportlab`

要开始使用，您需要安装带有`pip`的`reportlab`:

```py
$ python3 -m pip install reportlab
```

您可以使用`pip show`来验证安装:

```py
$ python3 -m pip show reportlab
Name: reportlab
Version: 3.5.34
Summary: The Reportlab Toolkit
Home-page: http://www.reportlab.com/
Author: Andy Robinson, Robin Becker, the ReportLab team
 and the community
Author-email: reportlab-users@lists2.reportlab.com
License: BSD license (see license.txt for details),
 Copyright (c) 2000-2018, ReportLab Inc.
Location: c:\users\davea\venv\lib\site-packages
Requires: pillow
Required-by:
```

在撰写本文时，`reportlab`的最新版本是 3.5.34。如果你有 IDLE open，那么你需要重新启动它才能使用`reportlab`包。

### 使用`Canvas`类

用`reportlab`创建 pdf 的主界面是`Canvas`类，它位于`reportlab.pdfgen.canvas`模块中。

打开一个新的空闲交互窗口，键入以下内容导入`Canvas`类:

>>>

```py
>>> from reportlab.pdfgen.canvas import Canvas
```

当您创建一个新的`Canvas`实例时，您需要提供一个字符串，其中包含您正在创建的 PDF 的文件名。继续为文件`hello.pdf`创建一个新的`Canvas`实例:

>>>

```py
>>> canvas = Canvas("hello.pdf")
```

现在您有了一个`Canvas`实例，它被赋予了变量名`canvas`，并且与当前工作目录中的一个名为`hello.pdf`的文件相关联。但是文件`hello.pdf`还不存在。

让我们给 PDF 添加一些文本。为此，您可以使用`.drawString()`:

>>>

```py
>>> canvas.drawString(72, 72, "Hello, World")
```

传递给`.drawString()`的前两个参数决定了文本在画布上的书写位置。第一个指定距画布左边缘的距离，第二个指定距下边缘的距离。

传递给`.drawString()`的值以磅为单位。因为一个点等于 1/72 英寸，所以`.drawString(72, 72, "Hello, World")`将字符串`"Hello, World"`绘制在页面左侧一英寸和底部一英寸处。

要将 PDF 保存到文件，请使用`.save()`:

>>>

```py
>>> canvas.save()
```

您现在在当前工作目录中有一个名为`hello.pdf`的 PDF 文件。可以用 PDF 阅读器打开，看到页面底部的文字`Hello, World`！

对于您刚刚创建的 PDF，有一些事情需要注意:

1.  默认页面尺寸是 A4，这与标准的美国信函页面尺寸不同。
2.  字体默认为 Helvetica，字号为 12 磅。

你不会被这些设置束缚住。

### 设置页面尺寸

当实例化一个`Canvas`对象时，可以用可选的`pagesize`参数改变页面大小。该参数接受一个由[浮点值](https://realpython.com/python-data-types/#floating-point-numbers)组成的元组，以磅为单位表示页面的宽度和高度。

例如，要将页面大小设置为宽`8.5`英寸，高`11`英寸，您可以创建下面的`Canvas`:

```py
canvas = Canvas("hello.pdf", pagesize=(612.0, 792.0))
```

`(612, 792)`代表信纸大小的纸张，因为`8.5`次`72`是`612`，而`11`次`72`是`792`。

如果你不喜欢计算将磅转换成英寸或厘米，那么你可以使用`reportlab.lib.units`模块来帮助你转换。`.units`模块包含几个助手对象，比如`inch`和`cm`，它们简化了你的转换。

继续从`reportlab.lib.units`模块导入`inch`和`cm`对象:

>>>

```py
>>> from reportlab.lib.units import inch, cm
```

现在，您可以检查每个对象，看看它们是什么:

>>>

```py
>>> cm
28.346456692913385
>>> inch
72.0
```

`cm`和`inch`都是浮点值。它们代表每个单元中包含的点数。`inch`是`72.0`点，`cm`是`28.346456692913385`点。

要使用单位，请将单位名称乘以要转换为点的单位数。例如，下面是如何使用`inch`将页面尺寸设置为`8.5`英寸宽乘`11`英寸高:

>>>

```py
>>> canvas = Canvas("hello.pdf", pagesize=(8.5 * inch, 11 * inch))
```

通过向`pagesize`传递一个 tuple，您可以创建任意大小的页面。然而，`reportlab`包有一些更容易使用的标准内置页面大小。

页面尺寸位于`reportlab.lib.pagesizes`模块中。例如，要将页面大小设置为 letter，可以从`pagesizes`模块导入`LETTER`对象，并在实例化`Canvas`时将其传递给`pagesize`参数:

>>>

```py
>>> from reportlab.lib.pagesizes import LETTER
>>> canvas = Canvas("hello.pdf", pagesize=LETTER)
```

如果您检查`LETTER`对象，那么您会看到它是一个浮点元组:

>>>

```py
>>> LETTER
(612.0, 792.0)
```

`reportlab.lib.pagesize`模块包含许多标准页面尺寸。以下是一些尺寸:

| 页面大小 | 规模 |
| --- | --- |
| `A4` | 210 毫米 x 297 毫米 |
| `LETTER` | 8.5 英寸 x 11 英寸 |
| `LEGAL` | 8.5 英寸 x 14 英寸 |
| `TABLOID` | 11 英寸 x 17 英寸 |

除此之外，该模块还包含所有 [ISO 216 标准纸张尺寸](https://en.wikipedia.org/wiki/ISO_216)的定义。

### 设置字体属性

当您向`Canvas`写入文本时，您还可以更改字体、字体大小和字体颜色。

要更改字体和字体大小，可以使用`.setFont()`。首先，用文件名`font-example.pdf`和信纸大小创建一个新的`Canvas`实例:

>>>

```py
>>> canvas = Canvas("font-example.pdf", pagesize=LETTER)
```

然后将字体设置为 Times New Roman，大小为`18`磅:

>>>

```py
>>> canvas.setFont("Times-Roman", 18)
```

最后，将字符串`"Times New Roman (18 pt)"`写入画布并保存:

>>>

```py
>>> canvas.drawString(1 * inch, 10 * inch, "Times New Roman (18 pt)")
>>> canvas.save()
```

使用这些设置，文本将被写在离页面左侧 1 英寸，离页面底部 10 英寸的地方。打开当前工作目录中的`font-example.pdf`文件并检查它！

默认情况下，有三种字体可用:

1.  `"Courier"`
2.  `"Helvetica"`
3.  `"Times-Roman"`

每种字体都有粗体和斜体两种变体。以下是`reportlab`中所有可用字体的列表:

*   `"Courier"`
*   `"Courier-Bold`
*   `"Courier-BoldOblique"`
*   `"Courier-Oblique"`
*   `"Helvetica"`
*   `"Helvetica-Bold"`
*   `"Helvetica-BoldOblique"`
*   `"Helvetica-Oblique"`
*   `"Times-Bold"`
*   `"Times-BoldItalic`
*   `"Times-Italic"`
*   `"Times-Roman"`

您也可以使用`.setFillColor()`设置字体颜色。在下面的示例中，您创建了一个名为`font-colors.pdf`的带有蓝色文本的 PDF 文件:

```py
from reportlab.lib.colors import blue
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas

canvas = Canvas("font-colors.pdf", pagesize=LETTER)

# Set font to Times New Roman with 12-point size
canvas.setFont("Times-Roman", 12)

# Draw blue text one inch from the left and ten
# inches from the bottom
canvas.setFillColor(blue)
canvas.drawString(1 * inch, 10 * inch, "Blue text")

# Save the PDF file
canvas.save()
```

`blue`是从`reportlab.lib.colors`模块导入的对象。这个模块包含几种常见的颜色。完整的颜色列表可以在 [`reportlab`源代码](https://realpython.com/pybasics-reportlab-source)中找到。

本节中的例子强调了使用`Canvas`对象的基础。但你只是触及了表面。使用`reportlab`，您可以从头开始创建表格、表单，甚至高质量的图形！

ReportLab 用户指南包含了大量如何从头开始生成 PDF 文档的例子。如果您有兴趣了解更多关于使用 Python 创建 pdf 的内容，这是一个很好的起点。

### 检查你的理解能力

展开下面的方框，检查您的理解程度:



在您计算机的主目录中创建一个名为`realpython.pdf`的 PDF，其中包含文本`"Hello, Real Python!"`的信纸大小的页面放置在距离页面左边缘 2 英寸和下边缘 8 英寸的位置。

您可以展开下面的方框查看解决方案:



用信纸大小的页面设置`Canvas`实例:

```py
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas

canvas = Canvas("font-colors.pdf", pagesize=LETTER)
```

现在画一条线`"Hello, Real Python!"`，距离左边两英寸，距离底部八英寸:

```py
canvas.drawString(2 * inch, 8 * inch, "Hello, Real Python!")
```

最后，保存`canvas`来编写 PDF 文件:

```py
canvas.save()
```

当你准备好了，你可以进入下一部分。

## 结论:用 Python 创建和修改 PDF 文件

在本教程中，您学习了如何使用`PyPDF2`和`reportlab`包创建和修改 PDF 文件。如果您想了解刚才看到的示例，请务必点击下面的链接下载材料:

**下载示例材料:** [单击此处获取您将在本教程中使用](https://realpython.com/bonus/create-modify-pdf/)学习创建和修改 PDF 文件的材料。

**通过`PyPDF2`，你学会了如何:**

*   **读取** PDF 文件，**使用`PdfFileReader`类提取**文本
*   **使用`PdfFileWriter`类编写**新的 PDF 文件
*   **连接**和**使用`PdfFileMerger`类合并** PDF 文件
*   **旋转**和**裁剪** PDF 页面
*   **用密码加密**和**解密** PDF 文件

您还了解了如何使用`reportlab`包从头开始创建 PDF 文件。**你学会了如何:**

*   使用`Canvas`类
*   **用`.drawString()`写**文本到`Canvas`
*   用`.setFont()`设置**字体**和**字体大小**
*   用`.setFillColor()`改变**字体颜色**

`reportlab`是一个强大的 PDF 创建工具，而你只是触及了它的表面。如果你喜欢在这个例子中从 [*Python 基础知识:Python 3*](https://realpython.com/products/python-basics-book/) 实用介绍中学到的东西，那么一定要看看本书的其余部分。

编码快乐！**********