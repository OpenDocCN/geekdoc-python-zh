# 用 Python 提取 PDF 元数据和文本

> 原文：<https://www.blog.pythonlibrary.org/2018/04/10/extracting-pdf-metadata-and-text-with-python/>

有很多与 Python 相关的 PDF 包。我最喜欢的一个是 [PyPDF2](https://pythonhosted.org/PyPDF2/) 。您可以使用它来提取元数据、旋转页面、分割或合并 pdf 等。这就像是现有 pdf 的瑞士军刀。在本文中，我们将学习如何使用 PyPDF2 提取 PDF 的基本信息

* * *

### 入门指南

PyPDF2 不是 Python 标准库的一部分，所以您需要自己安装它。这样做的首选方法是使用 [pip](https://packaging.python.org/tutorials/installing-packages/) 。

```py

pip install pypdf2

```

现在我们已经安装了 PyPDF2，让我们学习如何从 PDF 中获取元数据！

* * *

### 提取元数据

您可以使用 PyPDF2 从任何 PDF 中提取大量有用的数据。例如，您可以了解文档的作者、标题和主题以及有多少页。让我们从位于 https://leanpub.com/reportlab 的 Leanpub 下载这本书的样本来了解一下。我下载的样本名为“reportlab-sample.pdf”。

代码如下:

```py

# get_doc_info.py

from PyPDF2 import PdfFileReader

def get_info(path):
    with open(path, 'rb') as f:
        pdf = PdfFileReader(f)
        info = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()

    print(info)

    author = info.author
    creator = info.creator
    producer = info.producer
    subject = info.subject
    title = info.title

if __name__ == '__main__':
    path = 'reportlab-sample.pdf'
    get_info(path)

```

这里我们从 **PyPDF2** 导入 **PdfFileReader** 类。这个类让我们能够使用各种访问器方法读取 PDF 并从中提取数据。我们做的第一件事是创建我们自己的 **get_info** 函数，它接受 PDF 文件路径作为唯一的参数。然后，我们以只读二进制模式打开文件。接下来，我们将该文件处理程序传递给 PdfFileReader，并创建它的一个实例。

现在我们可以通过使用 **getDocumentInfo** 方法从 PDF 中提取一些信息。这将返回一个**pypdf 2 . pdf . document information**的实例，它具有以下有用的属性:

*   作者
*   创造者
*   生产者
*   科目
*   标题

如果您打印出 **DocumentInformation** 对象，您将会看到:

```py

{'/Author': 'Michael Driscoll',
 '/CreationDate': "D:20180331023901-00'00'",
 '/Creator': 'LaTeX with hyperref package',
 '/Producer': 'XeTeX 0.99998',
 '/Title': 'ReportLab - PDF Processing with Python'}

```

我们还可以通过调用 **getNumPages** 方法来获得 PDF 中的页数。

* * *

### 从 pdf 中提取文本

PyPDF2 对从 PDF 中提取文本的支持有限。不幸的是，它没有提取图像的内置支持。我在 StackOverflow 上看到过一些使用 PyPDF2 提取图像的菜谱，但是代码示例似乎很随意。

让我们尝试从上一节下载的 PDF 的第一页中提取文本:

```py

# extracting_text.py

from PyPDF2 import PdfFileReader

def text_extractor(path):
    with open(path, 'rb') as f:
        pdf = PdfFileReader(f)

        # get the first page
        page = pdf.getPage(1)
        print(page)
        print('Page type: {}'.format(str(type(page))))

        text = page.extractText()
        print(text)

if __name__ == '__main__':
    path = 'reportlab-sample.pdf'
    text_extractor(path)

```

您会注意到，这段代码的开始方式与我们之前的示例非常相似。我们仍然需要创建一个 **PdfFileReader** 的实例。但是这一次，我们使用 **getPage** 方法抓取页面。PyPDF2 是从零开始的，很像 Python 中的大多数东西，所以当你给它传递一个 1 时，它实际上抓取了第二页。在这种情况下，第一页只是一个图像，所以它不会有任何文本。

有趣的是，如果你运行这个例子，你会发现它没有返回任何文本。相反，我得到的是一系列换行符。不幸的是，PyPDF2 对提取文本的支持非常有限。即使它能够提取文本，它也可能不会按照您期望的顺序排列，并且间距也可能不同。

要让这个示例代码工作，您需要尝试在不同的 PDF 上运行它。我在美国国税局的网站上找到了一个:[https://www.irs.gov/pub/irs-pdf/fw9.pdf](https://www.irs.gov/pub/irs-pdf/fw9.pdf)

这是为个体经营者或合同工准备的 W9 表格。它也可以用在其他场合。无论如何，我以**w9.pdf**的名字下载了它，并把它添加到了 Github 库。如果您使用 PDF 文件而不是示例文件，它会很高兴地从第 2 页中提取一些文本。我不会在这里复制输出，因为它有点长。

您可能会发现 [pdfminer](https://github.com/euske/pdfminer) 包比 PyPDF2 更适合提取文本。

* * *

### 包扎

PyPDF2 包非常有用。使用它，我们能够从 pdf 中获得一些有用的信息。我可以在 PDF 文件夹中使用 PyPDF，并使用元数据提取技术按照创建者名称、主题等对 pdf 进行分类。试试看，看你怎么想！

* * *

### 相关阅读

*   简单的分步报告实验室[教程](https://www.blog.pythonlibrary.org/2010/03/08/a-simple-step-by-step-reportlab-tutorial/)
*   ReportLab 101: [文本对象](https://www.blog.pythonlibrary.org/2018/02/06/reportlab-101-the-textobject/)
*   ReportLab - [如何添加图表和图形](https://www.blog.pythonlibrary.org/2016/02/18/reportlab-how-to-add-charts-graphs/)