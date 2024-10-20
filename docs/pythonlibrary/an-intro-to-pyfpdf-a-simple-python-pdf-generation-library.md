# pyfpdf 简介——一个简单的 Python PDF 生成库

> 原文：<https://www.blog.pythonlibrary.org/2012/07/10/an-intro-to-pyfpdf-a-simple-python-pdf-generation-library/>

今天我们将看到一个简单的 PDF 生成库，名为 [pyfpdf](https://github.com/reingart/pyfpdf) ，它是一个 php 库 [FPDF](http://www.fpdf.org/) 的一个端口。这不是 Reportlab 的替代品，但它确实为您提供了创建简单 pdf 的足够能力，并且可能满足您的需求。让我们来看看它能做什么！

**注:不再维护 PyFPDF。已经换成了[fpdf 2](https://pypi.org/project/fpdf2/)**

### 安装 pyfpdf

您可以只使用 pip 来安装 pyfpdf:

```py
pip install pyfpdf

```

### 试用 pyfpdf

与任何新的库一样，您需要实际编写一些代码来查看它是如何工作的。下面是创建 PDF 的最简单的脚本之一:

```py
import pyfpdf

pdf = pyfpdf.FPDF(format='letter')
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Welcome to Python!", align="C")
pdf.output("tutorial.pdf")

```

注意，当我们初始化我们的 FPDF 对象时，我们需要告诉它我们希望结果是“字母”大小。默认为“A4”。接下来，我们需要添加一个页面，设置字体，并把一些文本。 **pdf.cell** 调用有点不直观。前两个参数是宽度和高度，指定了传入文本的位置。 **align** 参数只接受单个字母缩写，您必须查看源代码才能弄清楚。在这种情况下，我们通过传递一个“C”来使文本居中。最后一行接受两个参数:pdf 名称和目的地。如果没有目的地，那么 PDF 将输出到脚本运行的目录。

如果您想添加另一行呢？如果你编辑单元格的大小，然后创建另一个单元格，你可以在末尾添加更多的文本。如果您需要换行符，那么您可以将代码更改为以下内容:

```py
import pyfpdf

pdf = pyfpdf.FPDF(format='letter')
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Welcome to Python!", ln=1, align="C")
pdf.cell(200,10,'Powered by FPDF',0,1,'C')
pdf.output("tutorial.pdf")

```

这产生了下面的 PDF:[tutorial.pdf](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/07/tutorial.pdf)

### 添加页眉、页脚和分页符

pyfpdf 的 Google 代码站点上的教程展示了如何做页眉、页脚和分页符。它不能运行是因为一个方法名肯定已经改变了，而且代码是用“this”而不是“self”写的，所以我重写了它并稍微清理了一下。代码如下:

```py
import pyfpdf

########################################################################
class MyPDF(pyfpdf.FPDF):
    """"""

    #----------------------------------------------------------------------
    def header(self):
        """
        Header on each page
        """
        # insert my logo
        self.image("logo.png", x=10, y=8, w=23)
        # position logo on the right
        self.cell(w=80)

        # set the font for the header, B=Bold
        self.set_font("Arial", style="B", size=15)
        # page title
        self.cell(40,10, "Python Rules!", border=1, ln=0, align="C")
        # insert a line break of 20 pixels
        self.ln(20)

    #----------------------------------------------------------------------
    def footer(self):
        """
        Footer on each page
        """
        # position footer at 15mm from the bottom
        self.set_y(-15)

        # set the font, I=italic
        self.set_font("Arial", style="I", size=8)

        # display the page number and center it
        pageNum = "Page %s/{nb}" % self.page_no()
        self.cell(0, 10, pageNum, align="C")

#----------------------------------------------------------------------
if __name__ == "__main__":
    pdf = MyPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Times", size=12)

    # put some lines on the page
    for i in range(1, 50):
        pdf.cell(0, 10, "Line number %s" % i, border=0, ln=1)
    pdf.output("tutorial2.pdf")

```

这创建了下面的 PDF:[tutorial2.pdf](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/07/tutorial2.pdf)

这一次，我们创建了 FPDF 的一个子类，并覆盖了它的 **header** 和 **footer** 方法，因为它们实际上不做任何事情(也就是说，它们是存根)。在我们的头中，我们创建一个图像对象，并传入一个徽标文件以及徽标的 x/y 位置和宽度(23)。如果你关心保持长宽比正常，你也可以通过一个高度。然后我们定位它，并为标题插入一串文本。最后，我们放入一个换行符。

页脚设置为距离底部 15 毫米。它也是 8 磅的 Arial 斜体。官方教程的错误是它调用了 self。PageNo()，但那并不存在。然而有一个 **page_no** 方法似乎可以替代它，所以我使用了它。然后在脚本的底部，我们实际上创建了我们的 pdf 对象，并告诉它写一串行。如果您运行这个脚本，您应该得到一个 3 页的文档。

### 使用 pyfpdf 从 HTML 生成 pdf

我的一个[读者](http://www.reddit.com/r/Python/comments/wcsl7/an_intro_to_pyfpdf_a_simple_python_pdf_generation/)指出 pyfpdf 也可以从基本 HTML 生成 pdf。我不知道我怎么没发现，但这是真的！你可以！你可以在该项目的[维基](https://code.google.com/p/pyfpdf/wiki/WriteHTML)上读到它。我在下面复制一个稍加修改的例子:

```py
html = """

```

# html2fpdf

## 基本用法

您现在可以轻松打印文本，同时混合不同的样式:**粗体**、*斜体*、下划线或 ***一次全部*** ！
也可以插入类似这样的超链接 [www.mousevspython.comg](http://www.mousevspython.com) ，或者在图片中包含超链接。只需点击下面的一个。

<center>[![](img/b9b9ee79fb04412affe4c8204df56070.png)](http://www.mousevspython.com)</center>

### 样本列表

*   选项 3

标题 1 标题 2

单元格 1 单元格 2

细胞 2 细胞 3

"""从 pyfpdf 导入 fpdf，HTMLMixin 类 my pdf(FPDF，html mixin):pass pdf = my pdf()# First page pdf . add _ page()pdf . write _ html(html)pdf . output(' html . pdf '，' F ')

这将创建以下 PDF:[html.pdf](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/07/html.pdf)

### 包扎

他们的网站上还有一个关于颜色和换行符的教程，但我会把它留给你作为练习。我在这里没有看到任何关于绘图、插入表格或图形、自定义字体嵌入或 Reportlab 中可用的许多其他功能的内容，但这应该是一个简单的 PDF 生成库。如果您需要更高级的特性，那么您当然应该看看 Reportlab，甚至是基于它的一些分支项目(我想到了 rst2pdf 或 xhtml2pdf)。

### 进一步阅读

*   官方 [pyfpdf 教程](http://code.google.com/p/pyfpdf/wiki/Tutorial)
*   pyfpdf [参考手册](http://code.google.com/p/pyfpdf/wiki/ReferenceManual)

### 源代码

*   [pyfpdf_examples.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/07/pyfpdf_examples.zip)