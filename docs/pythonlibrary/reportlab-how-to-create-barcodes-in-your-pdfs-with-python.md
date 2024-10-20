# Reportlab -如何使用 Python 在 pdf 中创建条形码

> 原文：<https://www.blog.pythonlibrary.org/2013/03/25/reportlab-how-to-create-barcodes-in-your-pdfs-with-python/>

Reportlab 库是用 Python 生成 pdf 的一个很好的方式。最近注意到它有做条形码的能力。我听说过它能够生成二维码，但我并没有真正深入了解它还能做什么。在本教程中，我们将了解 Reportlab 可以生成的一些条形码。如果你还没有 Reportlab，在进入本文之前，去他们的[网站](http://www.reportlab.com/)获取。

### Reportlab 的条形码库

Reportlab 提供了几种不同类型的条形码:code39(即 code 3 of 9)、code93、code 128、EANBC、QR 和 USPS。我也看到了一个叫做“fourstate”的，但是我不知道如何让它工作。在其中一些类型下，还有子类型，如标准、扩展或多宽度。我没有太多的运气让多宽度的那个为 code128 条形码工作，因为它一直给我一个属性错误，所以我们就忽略那个。如果你知道怎么做，请在评论中或通过我的联系方式告诉我。如果有人能告诉我如何添加那个或 fourstate 条形码，我会更新这篇文章。

不管怎样，最好的学习方法就是写一些代码。这里有一个非常简单的例子:

```py

from reportlab.graphics.barcode import code39, code128, code93
from reportlab.graphics.barcode import eanbc, qr, usps
from reportlab.graphics.shapes import Drawing 
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF

#----------------------------------------------------------------------
def createBarCodes():
    """
    Create barcode examples and embed in a PDF
    """
    c = canvas.Canvas("barcodes.pdf", pagesize=letter)

    barcode_value = "1234567890"

    barcode39 = code39.Extended39(barcode_value)
    barcode39Std = code39.Standard39(barcode_value, barHeight=20, stop=1)

    # code93 also has an Extended and MultiWidth version
    barcode93 = code93.Standard93(barcode_value)

    barcode128 = code128.Code128(barcode_value)
    # the multiwidth barcode appears to be broken 
    #barcode128Multi = code128.MultiWidthBarcode(barcode_value)

    barcode_usps = usps.POSTNET("50158-9999")

    codes = [barcode39, barcode39Std, barcode93, barcode128, barcode_usps]

    x = 1 * mm
    y = 285 * mm
    x1 = 6.4 * mm

    for code in codes:
        code.drawOn(c, x, y)
        y = y - 15 * mm

    # draw the eanbc8 code
    barcode_eanbc8 = eanbc.Ean8BarcodeWidget(barcode_value)
    bounds = barcode_eanbc8.getBounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    d = Drawing(50, 10)
    d.add(barcode_eanbc8)
    renderPDF.draw(d, c, 15, 555)

    # draw the eanbc13 code
    barcode_eanbc13 = eanbc.Ean13BarcodeWidget(barcode_value)
    bounds = barcode_eanbc13.getBounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    d = Drawing(50, 10)
    d.add(barcode_eanbc13)
    renderPDF.draw(d, c, 15, 465)

    # draw a QR code
    qr_code = qr.QrCodeWidget('www.mousevspython.com')
    bounds = qr_code.getBounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    d = Drawing(45, 45, transform=[45./width,0,0,45./height,0,0])
    d.add(qr_code)
    renderPDF.draw(d, c, 15, 405)

    c.save()

if __name__ == "__main__":
    createBarCodes()

```

让我们把它分解一下。代码 39。Extended39 除了价值本身并没有接受太多东西。另一方面，代码 39。标准 39，代码 93。标准 93 和代码 128。Code128 都有基本相同的 API。你可以改变酒吧宽度，酒吧高度，打开开始/停止符号，并添加“安静”区。usps 条形码模块提供两种类型的条形码:FIM 和 POSTNET。FIM 或 Facing ID 标志只编码一个字母(A-D)，我个人并不觉得很有趣。所以我只展示 POSTNET 版本，这应该是美国人非常熟悉的，因为它出现在大多数信封的底部。POSTNET 对邮政编码进行编码！

接下来的三个条形码使用不同的 API 在我通过 [StackOverflow](http://stackoverflow.com/questions/13129015/generate-multiple-qr-codes-in-one-pdf-file-using-reportlab-and-django-framework) 发现的 PDF 上绘制它们。基本上，你创建一个一定大小的**图形**对象，然后将条形码添加到图形中。最后，您使用 **renderPDF** 模块将绘图放到 PDF 上。这很复杂，但是效果很好。EANBC 代码是你会在一些制成品上看到的代码，比如纸巾盒。

如果你想看看上面代码的结果，你可以在这里下载 PDF。

### 包扎

此时，您应该能够在 pdf 中创建自己的条形码。Reportlab 非常方便，我希望您会发现这个额外的工具对您的工作有所帮助。

### 附加阅读

*   [一步一步的报告实验室辅导](https://www.blog.pythonlibrary.org/2010/03/08/a-simple-step-by-step-reportlab-tutorial/)
*   Reportlab: [混合固定内容和流动内容](https://www.blog.pythonlibrary.org/2012/06/27/reportlab-mixing-fixed-content-and-flowables/)
*   Reportlab 表格-使用 Python 在 pdf 中创建表格
*   [用 Python 创建二维码](https://www.blog.pythonlibrary.org/2012/05/18/creating-qr-codes-with-python/)
*   关于 Python 条形码生成的 StackOverflow [问题](http://stackoverflow.com/questions/2179269/python-barcode-generation-library)
*   StackOverflow [关于 reportlab、QR 码和 django 的问题](http://stackoverflow.com/questions/13129015/generate-multiple-qr-codes-in-one-pdf-file-using-reportlab-and-django-framework)

### 获取来源！

*   [barcodes.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2013/03/barcodes.tar)