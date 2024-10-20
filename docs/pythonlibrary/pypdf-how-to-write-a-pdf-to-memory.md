# PyPdf:如何将 Pdf 写入内存

> 原文：<https://www.blog.pythonlibrary.org/2013/07/16/pypdf-how-to-write-a-pdf-to-memory/>

在我的工作中，我们有时需要将 PDF 写入内存而不是磁盘，因为我们需要将一个覆盖图合并到它上面。通过写入内存，我们可以加快这个过程，因为我们没有额外的步骤将文件写入磁盘，然后再将它读回内存。遗憾的是，pyPdf 的 PdfFileWriter()类不支持提取二进制字符串，所以我们只能用 StringIO 来代替。这里有一个例子，我将两个 pdf 合并到内存中:

```py

import pyPdf
from StringIO import StringIO

#----------------------------------------------------------------------
def mergePDFs(pdfOne, pdfTwo):
    """
    Merge PDFs
    """
    tmp = StringIO()

    output = pyPdf.PdfFileWriter()

    pdfOne = pyPdf.PdfFileReader(file(pdfOne, "rb"))
    for page in range(pdfOne.getNumPages()):
        output.addPage(pdfOne.getPage(page))
    pdfTwo = pyPdf.PdfFileReader(file(pdfTwo, "rb"))
    for page in range(pdfTwo.getNumPages()):
        output.addPage(pdfTwo.getPage(page))

    output.write(tmp)
    return tmp.getvalue()

if __name__ == "__main__":
    pdfOne = '/path/to/pdf/one'
    pdfTwo = '/path/to/pdf/two'
    pdfObj = mergePDFs(pdfOne, pdfTwo)

```

如您所见，您需要做的就是创建一个 **StringIO** ()对象，向 **PdfFileWriter** ()对象添加一些页面，然后将数据写入 StringIO 对象。然后要提取二进制字符串，就得调用 StringIO 的 **getvalue** ()方法。简单吧？现在你在内存中有了一个类似文件的对象，你可以用它来添加更多的页面或者覆盖 OMR 标记等等。

### 相关文章

*   [用 Python 和 pyPdf 操作 Pdf](https://www.blog.pythonlibrary.org/2010/05/15/manipulating-pdfs-with-python-and-pypdf/)
*   pyPdf 2:[pyPdf 的新分支](https://www.blog.pythonlibrary.org/2012/07/11/pypdf2-the-new-fork-of-pypdf/)