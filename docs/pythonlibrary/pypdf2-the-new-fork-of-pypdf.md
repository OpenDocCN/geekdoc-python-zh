# pyPdf 2:pyPdf 的新分支

> 原文：<https://www.blog.pythonlibrary.org/2012/07/11/pypdf2-the-new-fork-of-pypdf/>

今天我得知 pyPDF 项目并没有像我最初认为的那样死去。事实上，它已经被分为 PyPDF2(注意拼写略有不同)。也有可能其他人已经接管了最初的 pyPDF 项目并正在积极地工作。如果你愿意，你可以在 [reddit](http://www.reddit.com/r/Python/comments/wcsl7/an_intro_to_pyfpdf_a_simple_python_pdf_generation/) 上关注这些。与此同时，我决定试一试 PyPDF2，看看它与原版有什么不同。如果你有一两个空闲时间，请随意跟随。

### PyPDF2 简介

我最初在两年前写了关于 pyPDF 的文章，最近我一直在深入研究各种 Python PDF 相关的库，所以偶然发现 pyPDF 的一个新分支是非常令人兴奋的。我们将采用我的一些旧示例，并在新的 PyPDF2 中运行它们，看看它们是否以相同的方式工作。

```py

# Merge two PDFs
from PyPDF2 import PdfFileReader, PdfFileWriter

output = PdfFileWriter()
pdfOne = PdfFileReader(file( "some\path\to\a\PDf", "rb"))
pdfTwo = PdfFileReader(file("some\other\path\to\a\PDf", "rb"))

output.addPage(pdfOne.getPage(0))
output.addPage(pdfTwo.getPage(0))

outputStream = file(r"output.pdf", "wb")
output.write(outputStream)
outputStream.close()

```

这在我的 Windows 7 机器上运行得非常好。正如您可能已经猜到的，代码所做的只是创建 **PdfFileReader** 对象，并读取每个对象的第一页。接下来，它将这两个页面添加到我们的 PdfFileWriter 中。最后，我们打开一个新文件，写出我们的 PDF 页面。就是这样！您刚刚从两个单独的 pdf 创建了一个新文档！

现在让我们试试我的另一篇文章中的页面旋转脚本:

```py

from PyPDF2 import PdfFileWriter, PdfFileReader

output = PdfFileWriter()
input1 = PdfFileReader(file("document1.pdf", "rb"))
output.addPage(input1.getPage(1).rotateClockwise(90))
# output.addPage(input1.getPage(2).rotateCounterClockwise(90))

outputStream = file("output.pdf", "wb")
output.write(outputStream)
outputStream.close()

```

那也在我的机器上工作。到目前为止一切顺利。我对奇偶校验的最后一个测试是看它是否能提取与原始 pyPdf 相同的数据。我们将尝试从最新的 Reportlab 用户手册中读取元数据:

```py

>>> from PyPDF2 import PdfFileReader

>>> p = r'C:\Users\mdriscoll\Documents\reportlab-userguide.pdf'

>>> pdf = PdfFileReader(open(p, 'rb'))

>>> pdf.documentInfo

{'/ModDate': u'D:20120629155504', '/CreationDate': u'D:20120629155504', '/Producer': u'GPL Ghostscript 8.15', '/Title': u'reportlab-userguide.pdf', '/Creator': u'Adobe Acrobat 10.1.3', '/Author': u'mdriscoll'}
>>> pdf.getNumPages()

120
>>> info = pdf.getDocumentInfo()

>>> info.author

u'mdriscoll'
>>> info.creator

u'Adobe Acrobat 10.1.3'
>>> info.producer

u'GPL Ghostscript 8.15'
>>> info.title

u'reportlab-userguide.pdf'

```

这一切看起来都很好，除了作者那一点。我当然不是那份文件的作者，我也不知道为什么它认为我是。否则，它看起来工作正常。现在让我们来看看有什么新内容！

### PyPDF2 中的新增功能

在查看 PyPDF2 的源代码时，我注意到的第一件事是它为 PdfFileReader 和 PdfFileWriter 添加了一些新方法。我还注意到有一个全新的模块叫做 **merger.py** ，它包含了类:PdfFileMerger。因为在撰写本文时没有真正的文档，所以让我们看一看幕后的情况。添加到阅读器中的唯一新方法是 **getOutlines** ，它检索文档大纲(如果存在的话)。在 writer 中，支持添加**书签**和**命名目的地**。不多，但是要饭的不能挑肥拣瘦。我认为我最兴奋的部分是新的 PdfFileMerger 类，它让我想起了死去的[订书机项目](https://github.com/hellerbarde/stapler)。PDF merger 允许程序员通过连接、切片、插入或三者的任意组合将多个 PDF 合并成一个 PDF。

让我们用几个示例脚本来尝试一下，好吗？

```py

import PyPDF2

path = open('path/to/hello.pdf', 'rb')
path2 = open('path/to/another.pdf', 'rb')

merger = PyPDF2.PdfFileMerger()

merger.merge(position=0, fileobj=path2)
merger.merge(position=2, fileobj=path)
merger.write(open("test_out.pdf", 'wb'))

```

这是将两个文件合并在一起。第一个将从第 3 页开始插入第二个文件(注意逐个插入)，并在插入后继续。这比遍历两个文档的页面并将它们放在一起要容易得多。 **merge** 命令有以下签名和文档字符串，这很好地概括了它:

```py

>>> merge(position, file, bookmark=None, pages=None, import_bookmarks=True)

        Merges the pages from the source document specified by "file" into the output
        file at the page number specified by "position".

        Optionally, you may specify a bookmark to be applied at the beginning of the 
        included file by supplying the text of the bookmark in the "bookmark" parameter.

        You may prevent the source document's bookmarks from being imported by
        specifying "import_bookmarks" as False.

        You may also use the "pages" parameter to merge only the specified range of 
        pages from the source document into the output document.

```

还有一个 **append** 方法，它与 merge 命令相同，只是它假设您想要将所有页面追加到 PDF 的末尾。为了完整起见，这里有一个示例脚本:

```py

import PyPDF2

path = open('path/to/hello.pdf', 'rb')
path2 = open('path/to/another.pdf', 'rb')

merger = PyPDF2.PdfFileMerger()

merger.append(fileobj=path2)
merger.append(fileobj=path)
merger.write(open("test_out2.pdf", 'wb'))

```

这是相当无痛的，也非常好！

### 包扎

我想我已经为 PDF 黑客找到了一个很好的替代方案。我可以用 PyPDF2 合并和拆分 PDF，比使用原始 PyPDF 更容易。我也希望 PyPDF 能坚持下去，因为它有一个赞助者付钱给人们来开发它。根据 reddit 的帖子，最初的 pyPdf 有可能会复活，这两个项目可能会一起工作。不管发生什么，我只是很高兴它又回到了开发阶段，并希望能保持一段时间。让我知道你对这个话题的想法。

### 进一步阅读

*   [github](https://github.com/knowah/PyPDF2/) 上的 PyPDF2 源代码库
*   PyPDF2 网站也在 [github](http://knowah.github.com/PyPDF2/) 上
*   我看到的关于 PyPDF2 的两个 reddit 线程:[线程一](http://www.reddit.com/r/Python/comments/wcsl7/an_intro_to_pyfpdf_a_simple_python_pdf_generation/)和[线程二](http://www.reddit.com/r/Python/comments/qsvfm/pypdf2_updates_pypdf_pypdf2_is_an_opensource/)
*   [用 Python 和 pyPdf 操作 Pdf](https://www.blog.pythonlibrary.org/2010/05/15/manipulating-pdfs-with-python-and-pypdf/)