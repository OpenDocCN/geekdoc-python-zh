# Reportlab:混合固定内容和流动内容

> 原文：<https://www.blog.pythonlibrary.org/2012/06/27/reportlab-mixing-fixed-content-and-flowables/>

最近，我需要能够使用 Reportlab 的可流动内容，但要将它们放在固定的位置。你们中的一些人可能想知道我为什么要这么做。关于 flowables 的好处，就像段落一样，是它们很容易被设计。如果我能加粗某样东西或把某样东西放在中心，并把它放在一个固定的位置，那将会很棒！这花了很多谷歌和试验和错误，但我终于得到了一个像样的模板放在一起，我可以使用邮件。在本文中，我也将向您展示如何做到这一点。

### 入门指南

你需要确保你有报告实验室，否则你最终会一无所获。可以去[这里](http://www.reportlab.com/software/opensource/rl-toolkit/download/)抢。在你等待下载的时候，你可以继续阅读这篇文章或者去做一些其他有意义的事情。你现在准备好了吗？那就让我们开始吧！

现在我们只需要举个例子。幸运的是，我在工作中一直在做一些事情，所以我可以用下面这种愚蠢且不完整的格式信来掩饰。仔细研究代码，因为你永远不知道什么时候会有测试。

```py

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm, inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, Paragraph, Table

########################################################################
class LetterMaker(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, pdf_file, org, seconds):
        self.c = canvas.Canvas(pdf_file, pagesize=letter)
        self.styles = getSampleStyleSheet()
        self.width, self.height = letter
        self.organization = org
        self.seconds  = seconds

    #----------------------------------------------------------------------
    def createDocument(self):
        """"""
        voffset = 65

        # create return address
        address = """ Jack Spratt

        222 Ioway Blvd, Suite 100

        Galls, TX 75081-4016
        """
        p = Paragraph(address, self.styles["Normal"])        

        # add a logo and size it
        logo = Image("snakehead.jpg")
        logo.drawHeight = 2*inch
        logo.drawWidth = 2*inch
##        logo.wrapOn(self.c, self.width, self.height)
##        logo.drawOn(self.c, *self.coord(140, 60, mm))
##        

        data = [[p, logo]]
        table = Table(data, colWidths=4*inch)
        table.setStyle([("VALIGN", (0,0), (0,0), "TOP")])
        table.wrapOn(self.c, self.width, self.height)
        table.drawOn(self.c, *self.coord(18, 60, mm))

        # insert body of letter
        ptext = "Dear Sir or Madam:"
        self.createParagraph(ptext, 20, voffset+35)

        ptext = """
        The document you are holding is a set of requirements for your next mission, should you
        choose to accept it. In any event, this document will self-destruct %s seconds after you
        read it. Yes, %s can tell when you're done...usually.
        """ % (self.seconds, self.organization)
        p = Paragraph(ptext, self.styles["Normal"])
        p.wrapOn(self.c, self.width-70, self.height)
        p.drawOn(self.c, *self.coord(20, voffset+48, mm))

    #----------------------------------------------------------------------
    def coord(self, x, y, unit=1):
        """
        # http://stackoverflow.com/questions/4726011/wrap-text-in-a-table-reportlab
        Helper class to help position flowables in Canvas objects
        """
        x, y = x * unit, self.height -  y * unit
        return x, y    

    #----------------------------------------------------------------------
    def createParagraph(self, ptext, x, y, style=None):
        """"""
        if not style:
            style = self.styles["Normal"]
        p = Paragraph(ptext, style=style)
        p.wrapOn(self.c, self.width, self.height)
        p.drawOn(self.c, *self.coord(x, y, mm))

    #----------------------------------------------------------------------
    def savePDF(self):
        """"""
        self.c.save()   

#----------------------------------------------------------------------
if __name__ == "__main__":
    doc = LetterMaker("example.pdf", "The MVP", 10)
    doc.createDocument()
    doc.savePDF()

```

现在您已经看到了代码，所以我们将花一点时间来看看它是如何工作的。首先，我们创建一个 **Canvas** 对象，不用 LetterMaker 类也可以使用。我们还创建了一个**风格**字典，并设置了一些其他的类变量。在 **createDocument** 方法中，我们使用一些类似 HTML 的标签创建一个段落(一个地址)来控制字体和换行行为。然后，我们创建一个徽标，并在将两个项目放入 Reportlab **Table** 对象之前调整其大小。你会注意到，我留下了几行注释，展示了如何在没有桌子的情况下放置徽标。我们使用**坐标**方法来帮助定位可流动的。我在 StackOverflow 上找到了它，觉得它非常方便。

信的正文使用了一点字符串替换，并将结果放入另一个段落。我们还使用存储的偏移量来帮助我们定位。我发现为代码的某些部分存储几个偏移量非常有用。如果你小心地使用它们，那么你可以只改变几个偏移量来移动文档中的内容，而不必编辑每个元素的位置。如果您需要绘制线条或形状，您可以使用 canvas 对象以通常的方式来完成。

### 包扎

我希望这段代码能够帮助您创建 PDF。我不得不承认，我把它贴在这里，既是为了你自己，也是为了我自己的未来。我有点难过，我不得不从它身上剥离这么多，但我的组织不会很喜欢它，如果我张贴原件。不管怎样，现在您已经有了用 Python 创建一些漂亮的 PDF 文档的工具。现在你只需要走出去，做到这一点！

### 进一步阅读

*   简单的分步指南[报告实验室教程](https://www.blog.pythonlibrary.org/2010/03/08/a-simple-step-by-step-reportlab-tutorial/)
*   Reportlab 表格-使用 Python 在 pdf 中创建表格
*   [rst 2 pdf 简介](https://www.blog.pythonlibrary.org/2012/06/17/an-intro-to-rst2pdf-changing-restructured-text-into-pdfs-with-python/)使用 Python 将重构文本更改为 pdf
*   用 [Python 和 pyPdf](https://www.blog.pythonlibrary.org/2010/05/15/manipulating-pdfs-with-python-and-pypdf/) 操作 Pdf