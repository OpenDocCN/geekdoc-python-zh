# pdfrw 快速介绍

> 原文：<https://www.blog.pythonlibrary.org/2012/07/07/a-quick-intro-to-pdfrw/>

我一直在寻找 Python PDF 库，前几天我偶然发现了 [pdfrw](http://code.google.com/p/pdfrw/) 。它看起来像是 [pyPDF](http://pybrary.net/pyPdf/) 的替代品，因为它可以读写 PDF，连接 pdf，并可以使用 Reportlab 进行连接和水印等操作。这个项目看起来也有点死气沉沉，因为它的最后一次更新是在 2011 年，但话又说回来，pyPDF 的最后一次更新是在 2010 年，所以它有点新鲜。在本文中，我们将对 pdfrw 进行一点测试，看看它是否有用。快来凑热闹吧！

***安装注意:**遗憾的是没有 **setup.py** 脚本，所以你必须从谷歌代码中检查它，只需将 pdfrw 文件夹复制到 site-packages 或你的 virtualenv。*

### 使用 pdfrw 将 pdf 连接在一起

使用 pdfrw 将两个 PDF 文件合并成一个文件实际上非常简单。见下文:

```py

from pdfrw import PdfReader, PdfWriter

pages = PdfReader(r'C:\Users\mdriscoll\Desktop\1.pdf', decompress=False).pages
other_pages = PdfReader(r'C:\Users\mdriscoll\Desktop\2.pdf', decompress=False).pages

writer = PdfWriter()
writer.addpages(pages)
writer.addpages(other_pages)
writer.write(r'C:\Users\mdriscoll\Desktop\out.pdf')

```

我觉得有趣的是，在写出文件之前，你还可以通过这样的方式对文件进行元数据:

```py

writer.trailer.Info = IndirectPdfDict(
    Title = 'My Awesome PDF',
    Author = 'Mike',
    Subject = 'Python Rules!',
    Creator = 'myscript.py',
)

```

还有一个包含的示例显示了如何使用 pdfrw 和 reportlab 组合 pdf。我在这里重复一下:

```py

# http://code.google.com/p/pdfrw/source/browse/trunk/examples/rl1/subset.py
import sys
import os

from reportlab.pdfgen.canvas import Canvas

import find_pdfrw
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl

def go(inpfn, firstpage, lastpage):
    firstpage, lastpage = int(firstpage), int(lastpage)
    outfn = 'subset_%s_to_%s.%s' % (firstpage, lastpage, os.path.basename(inpfn))

    pages = PdfReader(inpfn, decompress=False).pages
    pages = [pagexobj(x) for x in pages[firstpage-1:lastpage]]
    canvas = Canvas(outfn)

    for page in pages:
        canvas.setPageSize(tuple(page.BBox[2:]))
        canvas.doForm(makerl(canvas, page))
        canvas.showPage()

    canvas.save()

if __name__ == '__main__':
    inpfn, firstpage, lastpage = sys.argv[1:]
    go(inpfn, firstpage, lastpage)

```

我只是觉得这很酷。无论如何，它为您提供了 pyPDF 编写器的几种替代方案。这个包中还有许多其他有趣的例子，包括

1.  如何使用 pdf(第一页)作为所有其他页面的背景。
2.  如何添加一个[水印](http://code.google.com/p/pdfrw/source/browse/trunk/examples/watermark.py)

我认为这个项目有潜力。希望我们能产生足够的兴趣来再次启动这个项目，或者可能得到一些新的东西。