# 如何在 Python 中使用 PDF

> 原文：<https://realpython.com/pdf-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**如何用 Python 处理 PDF**](/courses/pdf-python/)

可移植文档格式(PDF)是一种文件格式，可用于跨操作系统可靠地呈现和交换文档。虽然 PDF 最初是由 Adobe 发明的，但它现在是由国际标准化组织(ISO)维护的一个[开放标准](https://www.iso.org/standard/51502.html)。您可以使用 **`PyPDF2`** 包来处理 Python 中预先存在的 PDF。

`PyPDF2`是一个[纯 Python](https://stackoverflow.com/questions/45976946/what-is-pure-python) 包，可以用于许多不同类型的 PDF 操作。

到本文结束时，你将知道如何做以下事情:

*   用 Python 从 PDF 中提取文档信息
*   旋转页面
*   合并 pdf
*   分割 pdf
*   添加水印
*   加密 PDF

我们开始吧！

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## `pyPdf`、`PyPDF2`、`PyPDF4`、T3 的历史

最初的 [`pyPdf`](https://pypi.org/project/pyPdf/) 套装发布于 2005 年。上一次正式发布`pyPdf`是在 2010 年。时隔一年左右，一家名为 [Phasit](http://phaseit.net/) 的公司赞助了一款名为 [`PyPDF2`](https://pypi.org/project/PyPDF2/) 的`pyPdf`叉子。该代码被编写为向后兼容原始代码，并在几年内运行良好，其最后一次发布是在 2016 年。

有一个名为 [`PyPDF3`](https://pypi.org/project/PyPDF3/) 的包的简短系列发布，然后该项目被重命名为 [`PyPDF4`](https://pypi.org/project/PyPDF2/) 。所有这些项目做的都差不多，但是`pyPdf`和 PyPDF2+之间最大的区别是后者版本增加了 Python 3 支持。Python 3 有一个和原来[T3 不一样的 Python 3 分叉，但是那个已经很多年没有维护了。](http://github.com/mfenniak/pyPdf/tree/py3)

虽然`PyPDF2`在 2016 年被废弃，但它在 2022 年被重新启用，目前正在积极维护中。新的 [`PyPDF4`](https://github.com/claird/PyPDF4) 并不完全向后兼容`PyPDF2`。本文中的大多数例子都可以很好地与`PyPDF4`一起工作，但也有一些不能，这就是为什么`PyPDF4`在本文中没有被重点介绍的原因。随意用`PyPDF4`替换`PyPDF2`的导入，看看它如何为你工作。

[*Remove ads*](/account/join/)

## `pdfrw`:替代方案

帕特里克·莫平创造了一个名为 [`pdfrw`](https://github.com/pmaupin/pdfrw) 的包，它可以做很多和`PyPDF2`一样的事情。您可以使用`pdfrw`完成所有与您将在本文中学习如何为`PyPDF2`完成的任务相同的任务，加密是一个明显的例外。

最大的不同是它集成了 [ReportLab](https://www.reportlab.com/) 包，这样你就可以获取一个预先存在的 PDF，并使用部分或全部预先存在的 PDF 构建一个新的。

## 安装

如果你碰巧用的是 Anaconda 而不是普通的 Python，那么安装`PyPDF2`可以用`pip`或`conda`来完成。

下面是如何安装带有`pip`的`PyPDF2`:

```py
$ pip install pypdf2
```

安装非常快，因为`PyPDF2`没有任何依赖关系。你可能会花和安装包一样多的时间来下载它。

现在让我们继续学习如何从 PDF 中提取一些信息。

## 如何用 Python 从 PDF 中提取文档信息

您可以使用`PyPDF2`从 PDF 中提取元数据和一些文本。当您对预先存在的 PDF 文件进行某些类型的自动化时，这很有用。

以下是当前可以提取的数据类型:

*   作者
*   创造者
*   生产者
*   科目
*   标题
*   页数

你需要去找一个 PDF 来用于这个例子。你可以在你的机器上使用任何你手边的 PDF 文件。为了让事情变得简单，我去了 [Leanpub](https://leanpub.com/reportlab) 并拿了一本我的书作为这个练习的样本。你要下载的样本叫做`reportlab-sample.pdf`。

让我们使用该 PDF 编写一些代码，并了解如何访问这些属性:

```py
# extract_doc_info.py

from PyPDF2 import PdfFileReader

def extract_information(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PdfFileReader(f)
        information = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()

    txt = f"""
 Information about {pdf_path}: 

 Author: {information.author} Creator: {information.creator} Producer: {information.producer} Subject: {information.subject} Title: {information.title} Number of pages: {number_of_pages} """

    print(txt)
    return information

if __name__ == '__main__':
    path = 'reportlab-sample.pdf'
    extract_information(path)
```

这里您从`PyPDF2`包中[导入](https://realpython.com/absolute-vs-relative-python-imports/) `PdfFileReader`。`PdfFileReader`是一个有几个与 PDF 文件交互的方法的类。在这个例子中，您调用`.getDocumentInfo()`，它将返回一个`DocumentInformation`的实例。这包含了你感兴趣的大部分信息。您还调用 reader 对象上的`.getNumPages()`，它返回文档中的页数。

**注意:**最后一个代码块使用 Python 3 新的 f 字符串进行字符串格式化。如果你想了解更多，你可以查看 [Python 3 的 f-Strings:一个改进的字符串格式语法(指南)](https://realpython.com/python-f-strings/)。

`information` [变量](https://realpython.com/python-variables/)有几个实例属性，您可以使用它们从文档中获得想要的其余元数据。你[把信息](https://realpython.com/python-print/)打印出来，然后还回去以备将来使用。

虽然`PyPDF2`有`.extractText()`，可以在它的页面对象上使用(在这个例子中没有显示)，但是它不能很好地工作。一些 pdf 将返回文本，一些将返回空字符串。当你想从 PDF 中提取文本时，你应该选择 [`PDFMiner`](https://github.com/euske/pdfminer) 项目。 **`PDFMiner`** 更加健壮，专门用于从 pdf 中提取文本。

现在您已经准备好了解旋转 PDF 页面。

[*Remove ads*](/account/join/)

## 如何翻页

有时，您会收到包含处于[横向模式](https://techterms.com/definition/pageorientation)而非纵向模式页面的 pdf。或者它们甚至是上下颠倒的。当有人将文档扫描成 PDF 或电子邮件时，可能会发生这种情况。您可以将文档打印出来并阅读纸质版本，也可以使用 Python 的强大功能来旋转有问题的页面。

对于这个例子，你可以选择一篇真正的 Python 文章并打印成 PDF。

让我们学习如何用`PyPDF2`旋转文章的几页:

```py
# rotate_pages.py

from PyPDF2 import PdfFileReader, PdfFileWriter

def rotate_pages(pdf_path):
    pdf_writer = PdfFileWriter()
    pdf_reader = PdfFileReader(pdf_path)
    # Rotate page 90 degrees to the right
    page_1 = pdf_reader.getPage(0).rotateClockwise(90)
    pdf_writer.addPage(page_1)
    # Rotate page 90 degrees to the left
    page_2 = pdf_reader.getPage(1).rotateCounterClockwise(90)
    pdf_writer.addPage(page_2)
    # Add a page in normal orientation
    pdf_writer.addPage(pdf_reader.getPage(2))

    with open('rotate_pages.pdf', 'wb') as fh:
        pdf_writer.write(fh)

if __name__ == '__main__':
    path = 'Jupyter_Notebook_An_Introduction.pdf'
    rotate_pages(path)
```

对于这个例子，除了`PdfFileReader`之外，您还需要[导入](https://realpython.com/python-import/)T0 】,因为您将需要写出一个新的 PDF。`rotate_pages()`获取您想要修改的 PDF 的路径。在这个函数中，您需要创建一个名为`pdf_writer`的 writer 对象和一个名为`pdf_reader`的 reader 对象。

接下来，您可以使用`.GetPage()`来获得想要的页面。这是第 0 页，也就是第一页。然后调用 page 对象的`.rotateClockwise()`方法，90 度传入。然后对于第二页，你调用`.rotateCounterClockwise()`并且同样 90 度通过它。

**注意:**`PyPDF2`包只允许你以 90 度为增量旋转一页。否则你会收到一个`AssertionError`。

在每次调用旋转方法之后，调用`.addPage()`。这将把页面的旋转版本添加到 writer 对象中。添加到 writer 对象的最后一页是第 3 页，没有进行任何旋转。

最后，使用`.write()`写出新的 PDF。它将一个类似于文件的对象作为它的参数。这个新的 PDF 将包含三页。前两页将向彼此相反的方向旋转，并且是横向的，而第三页是普通页。

现在让我们学习如何将多个 pdf 合并成一个。

## 如何合并 pdf 文件

在许多情况下，您会想要将两个或多个 PDF 合并成一个 PDF。例如，您可能有一个标准的封面，需要转到许多类型的报告。你可以用 Python 来帮你做这类事情。

对于本例，您可以打开一个 PDF，并将页面作为单独的 PDF 打印出来。然后再做一次，但是用不同的页面。这将为您提供一些输入，用于示例目的。

让我们继续编写一些代码，您可以使用它们将 pdf 合并在一起:

```py
# pdf_merging.py

from PyPDF2 import PdfFileReader, PdfFileWriter

def merge_pdfs(paths, output):
    pdf_writer = PdfFileWriter()

    for path in paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            # Add each page to the writer object
            pdf_writer.addPage(pdf_reader.getPage(page))

    # Write out the merged PDF
    with open(output, 'wb') as out:
        pdf_writer.write(out)

if __name__ == '__main__':
    paths = ['document1.pdf', 'document2.pdf']
    merge_pdfs(paths, output='merged.pdf')
```

当您有想要合并在一起的 pdf 列表时，您可以使用`merge_pdfs()`。您还需要知道保存结果的位置，因此这个函数接受一个输入路径列表和一个输出路径。

然后循环输入，并为每个输入创建一个 PDF 阅读器对象。接下来，您将遍历 PDF 文件中的所有页面，并使用`.addPage()`将这些页面中的每一个添加到它自身。

一旦你完成了列表中所有 pdf 的所有页面的迭代，你将在最后写出结果。

我想指出的一点是，如果您不想合并每个 PDF 的所有页面，您可以通过添加一系列要添加的页面来增强这个脚本。如果你喜欢挑战，你也可以使用 [Python 的`argparse`](https://realpython.com/command-line-interfaces-python-argparse/) 模块为这个函数创建一个命令行界面。

让我们来了解一下如何做与合并相反的事情！

[*Remove ads*](/account/join/)

## 如何分割 pdf 文件

有时，您可能需要将一个 PDF 拆分成多个 PDF。对于包含大量扫描内容的 PDF 来说尤其如此，但是有太多好的理由想要分割 PDF。

以下是如何使用`PyPDF2`将 PDF 分割成多个文件:

```py
# pdf_splitting.py

from PyPDF2 import PdfFileReader, PdfFileWriter

def split(path, name_of_split):
    pdf = PdfFileReader(path)
    for page in range(pdf.getNumPages()):
        pdf_writer = PdfFileWriter()
        pdf_writer.addPage(pdf.getPage(page))

        output = f'{name_of_split}{page}.pdf'
        with open(output, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)

if __name__ == '__main__':
    path = 'Jupyter_Notebook_An_Introduction.pdf'
    split(path, 'jupyter_page')
```

在本例中，您再次创建了一个 PDF reader 对象，并对其页面进行循环。对于 PDF 中的每个页面，您将创建一个新的 PDF writer 实例并向其添加一个页面。然后，您将把该页面写出到一个唯一命名的文件中。当脚本运行完成时，您应该将原始 PDF 的每一页分割成单独的 PDF。

现在，让我们花一点时间来学习如何添加水印到您的 PDF。

## 如何添加水印

水印是印刷和数字文档上的识别图像或图案。有些水印只有在特殊的光照条件下才能看到。水印之所以重要，是因为它允许您保护您的知识产权，如您的图像或 pdf。水印的另一个术语是覆盖。

你可以用 Python 和`PyPDF2`给你的文档加水印。您需要一个只包含您的水印图像或文本的 PDF。

现在让我们学习如何添加水印:

```py
# pdf_watermarker.py

from PyPDF2 import PdfFileWriter, PdfFileReader

def create_watermark(input_pdf, output, watermark):
    watermark_obj = PdfFileReader(watermark)
    watermark_page = watermark_obj.getPage(0)

    pdf_reader = PdfFileReader(input_pdf)
    pdf_writer = PdfFileWriter()

    # Watermark all the pages
    for page in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page)
        page.mergePage(watermark_page)
        pdf_writer.addPage(page)

    with open(output, 'wb') as out:
        pdf_writer.write(out)

if __name__ == '__main__':
    create_watermark(
        input_pdf='Jupyter_Notebook_An_Introduction.pdf', 
        output='watermarked_notebook.pdf',
        watermark='watermark.pdf')
```

`create_watermark()`接受三个论点:

1.  **`input_pdf` :** 要加水印的 PDF 文件路径
2.  **`output` :** 您想要保存 PDF 水印版本的路径
3.  **`watermark` :** 包含水印图像或文本的 PDF

在代码中，您打开水印 PDF 并从文档中抓取第一页，因为这是您的水印应该驻留的位置。然后使用`input_pdf`和一个通用的`pdf_writer`对象创建一个 PDF 阅读器对象，用于写出带水印的 PDF。

下一步是迭代`input_pdf`中的页面。这就是奇迹发生的地方。你需要调用`.mergePage()`并传递给它`watermark_page`。当你这样做时，它将覆盖当前页面顶部的`watermark_page`。然后你把新合并的页面添加到你的`pdf_writer`对象中。

最后，你把新加水印的 PDF 写到磁盘上，你就完成了！

您将了解的最后一个主题是`PyPDF2`如何处理加密。

## 如何加密 PDF 文件

`PyPDF2`目前仅支持将用户密码和所有者密码添加到预先存在的 PDF 中。在 PDF land 中，所有者密码基本上会授予您 PDF 的管理员权限，并允许您设置文档的权限。另一方面，用户密码只允许您打开文档。

据我所知，`PyPDF2`实际上不允许您在文档上设置任何权限，尽管它允许您设置所有者密码。

无论如何，这就是你如何添加密码，这也将固有地加密 PDF:

```py
# pdf_encrypt.py

from PyPDF2 import PdfFileWriter, PdfFileReader

def add_encryption(input_pdf, output_pdf, password):
    pdf_writer = PdfFileWriter()
    pdf_reader = PdfFileReader(input_pdf)

    for page in range(pdf_reader.getNumPages()):
        pdf_writer.addPage(pdf_reader.getPage(page))

    pdf_writer.encrypt(user_pwd=password, owner_pwd=None, 
                       use_128bit=True)

    with open(output_pdf, 'wb') as fh:
        pdf_writer.write(fh)

if __name__ == '__main__':
    add_encryption(input_pdf='reportlab-sample.pdf',
                   output_pdf='reportlab-encrypted.pdf',
                   password='twofish')
```

`add_encryption()`接收输入和输出 PDF 路径以及您想要添加到 PDF 的密码。然后像以前一样，它打开一个 PDF writer 和一个 reader 对象。因为您想要加密整个输入 PDF，所以您需要循环所有页面并将它们添加到 writer 中。

最后一步是调用`.encrypt()`，它接受用户密码、所有者密码以及是否应该添加 128 位加密。默认情况下，打开 128 位加密。如果您将其设置为`False`，那么将改为应用 40 位加密。

**注:** PDF 加密根据[pdflib.com](https://www.pdflib.com/pdf-knowledge-base/pdf-password-security/encryption/)使用 RC4 或 AES(高级加密标准)加密 PDF。

仅仅因为你加密了你的 PDF 并不意味着它一定是安全的。有工具可以删除 pdf 中的密码。如果你想了解更多，卡耐基·梅隆大学有一篇关于主题的有趣的[论文。](https://www.cs.cmu.edu/~dst/Adobe/Gallery/PDFsecurity.pdf)

[*Remove ads*](/account/join/)

## 结论

这个包非常有用，而且通常非常快。您可以使用`PyPDF2`来自动化大型工作，并利用其功能来帮助您更好地完成工作！

在本教程中，您学习了如何执行以下操作:

*   从 PDF 中提取元数据
*   旋转页面
*   合并和分割 pdf
*   添加水印
*   添加加密

同时也要关注新的`PyPDF4`包，因为它可能很快就会取代`PyPDF2`。你可能还想看看 [`pdfrw`](https://github.com/pmaupin/pdfrw) ，它可以做很多和`PyPDF2`一样的事情。

## 延伸阅读

如果您想了解更多关于使用 Python 处理 pdf 的信息，您应该查看以下资源以获取更多信息:

*   [`PyPDF2`网站](https://pythonhosted.org/PyPDF2/)
*   [Github 页面为`PyPDF4`](https://github.com/claird/PyPDF4)
*   [Github 页面为`pdfrw`](https://github.com/pmaupin/pdfrw)
*   [报告实验室网站](https://www.reportlab.com/)
*   [Github 页面为`PDFMiner`](https://github.com/euske/pdfminer)
*   [卡米洛特:人类 PDF 表格提取](https://github.com/socialcopsdev/camelot)
*   [在 Python 中创建和修改 PDF 文件(教程)](https://realpython.com/creating-modifying-pdf/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**如何用 Python 处理 PDF**](/courses/pdf-python/)******