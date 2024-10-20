# 使用 Python 拆分和合并 pdf

> 原文：<https://www.blog.pythonlibrary.org/2018/04/11/splitting-and-merging-pdfs-with-python/>

PyPDF2 包允许你在现有的 PDF 上做很多有用的操作。在这篇文章中，我们将学习如何将一个 PDF 文件分割成多个更小的文件。我们还将学习如何把一系列的 PDF 文件组合成一个 PDF 文件。

* * *

### 入门指南

PyPDF2 不是 Python 标准库的一部分，所以您需要自己安装它。这样做的首选方法是使用 [pip](https://packaging.python.org/tutorials/installing-packages/) 。

```py

pip install pypdf2

```

现在我们已经安装了 PyPDF2，让我们学习如何分割和合并 PDF！

* * *

### 分割 pdf

PyPDF2 包让您能够将一个 PDF 分割成多个 PDF。你只需要告诉它你想要多少页。对于这个例子，我们将从 IRS 下载一个 [W9 表单](https://www.irs.gov/pub/irs-pdf/fw9.pdf),并遍历它的所有六个页面。我们将分裂出每一页，把它变成自己的独立的 PDF 文件。

让我们来看看如何实现:

```py

# pdf_splitter.py

import os
from PyPDF2 import PdfFileReader, PdfFileWriter

def pdf_splitter(path):
    fname = os.path.splitext(os.path.basename(path))[0]

    pdf = PdfFileReader(path)
    for page in range(pdf.getNumPages()):
        pdf_writer = PdfFileWriter()
        pdf_writer.addPage(pdf.getPage(page))

        output_filename = '{}_page_{}.pdf'.format(
            fname, page+1)

        with open(output_filename, 'wb') as out:
            pdf_writer.write(out)

        print('Created: {}'.format(output_filename))

if __name__ == '__main__':
    path = 'w9.pdf'
    pdf_splitter(path)

```

对于这个例子，我们需要导入 **PdfFileReader** 和**pdffilerwriter**。然后我们创建一个有趣的小函数，叫做 **pdf_splitter** 。它接受输入 PDF 的路径。这个函数的第一行将获取输入文件的名称，减去扩展名。接下来，我们打开 PDF 并创建一个 reader 对象。然后我们使用 reader 对象的 **getNumPages** 方法遍历所有页面。

在循环的**内部，我们创建了一个 **PdfFileWriter** 的实例。然后，我们使用其 **addPage** 方法向 writer 对象添加一个页面。这个方法接受一个 page 对象，所以为了获取 page 对象，我们调用 reader 对象的 **getPage** 方法。现在我们已经向 writer 对象添加了一个页面。下一步是创建一个唯一的文件名，我们使用原始文件名加上单词“page”加上页码+ 1。我们添加 1 是因为 PyPDF2 的页码是从零开始的，所以第 0 页实际上是第 1 页。**

最后，我们以写入二进制模式打开新文件名，并使用 PDF writer 对象的 **write** 方法将对象的内容写入磁盘。

* * *

### 将多个 pdf 合并在一起

现在我们有了一堆 pdf 文件，让我们学习如何把它们合并在一起。这样做的一个有用的用例是企业将他们的日报合并成一个 PDF。为了工作和娱乐，我需要合并 pdf。我脑海中浮现的一个项目是扫描文档。根据您使用的扫描仪，您可能最终会将一个文档扫描成多个 pdf，因此能够将它们再次合并在一起会非常棒。

当最初的 PyPdf 问世时，让它将多个 Pdf 合并在一起的唯一方法是这样的:

```py

# pdf_merger.py

import glob
from PyPDF2 import PdfFileWriter, PdfFileReader

def merger(output_path, input_paths):
    pdf_writer = PdfFileWriter()

    for path in input_paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))

    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)

if __name__ == '__main__':
    paths = glob.glob('w9_*.pdf')
    paths.sort()
    merger('pdf_merger.pdf', paths)

```

这里我们创建了一个 **PdfFileWriter** 对象和几个 **PdfFileReader** 对象。对于每个 PDF 路径，我们创建一个 **PdfFileReader** 对象，然后遍历它的页面，将每个页面添加到我们的 writer 对象中。然后，我们将 writer 对象的内容写到磁盘上。

PyPDF2 通过创建一个 **PdfFileMerger** 对象使这变得简单了一点:

```py

# pdf_merger2.py

import glob
from PyPDF2 import PdfFileMerger

def merger(output_path, input_paths):
    pdf_merger = PdfFileMerger()
    file_handles = []

    for path in input_paths:
        pdf_merger.append(path)

    with open(output_path, 'wb') as fileobj:
        pdf_merger.write(fileobj)

if __name__ == '__main__':
    paths = glob.glob('w9_*.pdf')
    paths.sort()
    merger('pdf_merger2.pdf', paths)

```

这里我们只需要创建 **PdfFileMerger** 对象，然后遍历 PDF 路径，将它们添加到我们的合并对象中。PyPDF2 将自动追加整个文档，因此您不需要自己遍历每个文档的所有页面。然后我们把它写到磁盘上。

**PdfFileMerger** 类也有一个可以使用的**合并**方法。它的代码定义如下:

```py

def merge(self, position, fileobj, bookmark=None, pages=None, import_bookmarks=True):
        """
        Merges the pages from the given file into the output file at the
        specified page number.

        :param int position: The *page number* to insert this file. File will
            be inserted after the given number.

        :param fileobj: A File Object or an object that supports the standard read
            and seek methods similar to a File Object. Could also be a
            string representing a path to a PDF file.

        :param str bookmark: Optionally, you may specify a bookmark to be applied at
            the beginning of the included file by supplying the text of the bookmark.

        :param pages: can be a :ref:`Page Range ` or a ``(start, stop[, step])`` tuple
            to merge only the specified range of pages from the source
            document into the output document.

        :param bool import_bookmarks: You may prevent the source document's bookmarks
            from being imported by specifying this as ``False``.
        """ 
```

基本上，merge 方法允许您通过页码告诉 PyPDF 将页面合并到哪里。因此，如果您已经创建了一个包含 3 页的合并对象，您可以告诉合并对象在特定位置合并下一个文档。这允许开发者做一些非常复杂的合并操作。试试看，看看你能做什么！

* * *

### 包扎

PyPDF2 是一个强大而有用的软件包。多年来，我一直断断续续地使用它来处理各种家庭和工作项目。如果您需要操作现有的 pdf，那么这个包可能正好适合您！

* * *

### 相关阅读

*   简单的分步报告实验室[教程](https://www.blog.pythonlibrary.org/2010/03/08/a-simple-step-by-step-reportlab-tutorial/)
*   ReportLab 101: [文本对象](https://www.blog.pythonlibrary.org/2018/02/06/reportlab-101-the-textobject/)
*   ReportLab - [如何添加图表和图形](https://www.blog.pythonlibrary.org/2016/02/18/reportlab-how-to-add-charts-graphs/)
*   [用 Python 提取 PDF 元数据](https://www.blog.pythonlibrary.org/2018/04/10/extracting-pdf-metadata-and-text-with-python/)和文本