# 用 Python 解析 XML 和创建 PDF 发票

> 原文：<https://www.blog.pythonlibrary.org/2012/07/18/parsing-xml-and-creating-a-pdf-invoice-with-python/>

*注:以下帖子最初发表在 [Dzone](http://python.dzone.com/articles/use-case-xml-parsing-python) 上。我更改了标题，因为我已经写了几篇 XML 解析文章，不希望我的读者将这篇文章与其他文章混淆。*

我日常工作中的一项常见任务是获取一些数据格式输入，并对其进行解析以创建报告或其他文档。今天，我们将查看一些 XML 输入，用 Python 编程语言对其进行解析，然后使用 report lab(Python 的第三方包)创建一封 PDF 格式的信件。比方说，我的公司收到了一份三件商品的订单，我需要履行订单。这样的 XML 可以看起来像下面的代码:

```py

 <order_number>456789</order_number>
    <customer_id>789654</customer_id>
    <address1>John Doe</address1>
    <address2>123 Dickens Road</address2>
    <address3>Johnston, IA 55555</address3>

    <order_items><item><id>11123</id>
            <name>Expo Dry Erase Pen</name>
            <price>1.99</price>
            <quantity>5</quantity></item> 
        <item><id>22245</id>
            <name>Cisco IP Phone 7942</name>
            <price>300</price>
            <quantity>1</quantity></item> 
        <item><id>33378</id>
            <name>Waste Basket</name>
            <price>9.99</price>
            <quantity>1</quantity></item></order_items> 

```

将上面的代码保存为 order.xml，现在我只需要用 Python 写一个解析器和 PDF 生成器脚本。您可以使用 Python 内置的 XML 解析库，其中包括 SAX、minidom 或 ElementTree，或者您可以出去下载许多用于 XML 解析的外部包中的一个。我最喜欢的是 lxml，它包括 ElementTree 的一个版本以及一段非常好的代码，他们称之为“objectify”。后一部分主要是将 XML 转换成点符号 Python 对象。我将用它来做我们的解析，因为它非常简单，易于实现和理解。如前所述，我将使用 Reportlab 来创建 PDF 文件。

下面是一个简单的脚本，它将完成我们需要的一切:

```py

from decimal import Decimal
from lxml import etree, objectify

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

########################################################################
class PDFOrder(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, xml_file, pdf_file):
        """Constructor"""
        self.xml_file = xml_file
        self.pdf_file = pdf_file

        self.xml_obj = self.getXMLObject()

    #----------------------------------------------------------------------
    def coord(self, x, y, unit=1):
        """
        # http://stackoverflow.com/questions/4726011/wrap-text-in-a-table-reportlab
        Helper class to help position flowables in Canvas objects
        """
        x, y = x * unit, self.height -  y * unit
        return x, y  

    #----------------------------------------------------------------------
    def createPDF(self):
        """
        Create a PDF based on the XML data
        """
        self.canvas = canvas.Canvas(self.pdf_file, pagesize=letter)
        width, self.height = letter
        styles = getSampleStyleSheet()
        xml = self.xml_obj

        address = """ SHIP TO:

        %s

        %s

        %s

        %s 
        """ % (xml.address1, xml.address2, xml.address3, xml.address4)
        p = Paragraph(address, styles["Normal"])
        p.wrapOn(self.canvas, width, self.height)
        p.drawOn(self.canvas, *self.coord(18, 40, mm))

        order_number = '**Order #%s** ' % xml.order_number
        p = Paragraph(order_number, styles["Normal"])
        p.wrapOn(self.canvas, width, self.height)
        p.drawOn(self.canvas, *self.coord(18, 50, mm))

        data = []
        data.append(["Item ID", "Name", "Price", "Quantity", "Total"])
        grand_total = 0
        for item in xml.order_items.iterchildren():
            row = []
            row.append(item.id)
            row.append(item.name)
            row.append(item.price)
            row.append(item.quantity)
            total = Decimal(str(item.price)) * Decimal(str(item.quantity))
            row.append(str(total))
            grand_total += total
            data.append(row)
        data.append(["", "", "", "Grand Total:", grand_total])
        t = Table(data, 1.5 * inch)
        t.setStyle(TableStyle([
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
            ('BOX', (0,0), (-1,-1), 0.25, colors.black)
        ]))
        t.wrapOn(self.canvas, width, self.height)
        t.drawOn(self.canvas, *self.coord(18, 85, mm))

        txt = "Thank you for your business!"
        p = Paragraph(txt, styles["Normal"])
        p.wrapOn(self.canvas, width, self.height)
        p.drawOn(self.canvas, *self.coord(18, 95, mm))

    #----------------------------------------------------------------------
    def getXMLObject(self):
        """
        Open the XML document and return an lxml XML document
        """
        with open(self.xml_file) as f:
            xml = f.read()
        return objectify.fromstring(xml)

    #----------------------------------------------------------------------
    def savePDF(self):
        """
        Save the PDF to disk
        """
        self.canvas.save()

#----------------------------------------------------------------------
if __name__ == "__main__":
    xml = "order.xml"
    pdf = "letter.pdf"
    doc = PDFOrder(xml, pdf)
    doc.createPDF()
    doc.savePDF()

```

下面是 PDF 输出:[letter.pdf](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/07/letter.pdf)

让我们花几分钟来看一下这段代码。首先是一批进口货。这只是用来自 Reportlab 和 lxml 的所需内容设置了我们的环境。我还导入了十进制模块，因为我将添加数量，这对于浮点数学来说比仅仅使用普通的 Python 数学要精确得多。接下来，我们创建接受两个参数的 PDFOrder 类:一个 xml 文件和一个 pdf 文件路径。在我们的初始化方法中，我们创建两个类属性，读取 XML 文件并返回一个 XML 对象。coord 方法用于定位 Reportlab 流，这些流是动态对象，能够跨页面拆分并接受各种样式。

createPDF 方法是程序的核心。canvas 对象用于创建我们的 PDF 并在其上“绘图”。我将它设置为 letter 大小，还获取了一个默认样式表。接下来，我创建一个送货地址，并将其放置在页面顶部附近，距离左侧 18 毫米，距离顶部 40 毫米。之后，我创建并下订单编号。最后，我对订单中的项目进行迭代，并将它们放在一个嵌套列表中，然后将该列表放在 Reportlab 的表 flowable 中。最后，我定位表格并传递一些样式给它一个边框和内部网格。最后，我们将文件保存到磁盘。

文档已经创建好了，现在我已经有了一个很好的原型来展示给我的同事们。在这一点上，我需要做的就是通过为文本传递不同的样式(即粗体、斜体、字体大小)或稍微改变布局来调整文档的外观。这通常取决于管理层或客户，所以你必须等待，看看他们想要什么。

现在您知道了如何用 Python 解析 XML 文档并从解析的数据创建 PDF。

### 源代码

*   [xml2pdfex.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/07/xml2pdfex.zip)