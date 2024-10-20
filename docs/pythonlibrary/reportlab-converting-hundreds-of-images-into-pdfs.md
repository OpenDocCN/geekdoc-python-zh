# Reportlab:将数百张图像转换为 pdf

> 原文：<https://www.blog.pythonlibrary.org/2012/01/07/reportlab-converting-hundreds-of-images-into-pdfs/>

我最近被要求将几百张图片转换成 PDF 页面。我的一个朋友画漫画，我的兄弟希望能够在平板电脑上阅读。唉，如果你有一堆像这样命名的文件:

 `'Jia_01.Jpg', 'Jia_02.Jpg', 'Jia_09.Jpg', 'Jia_10.Jpg', 'Jia_11.Jpg', 'Jia_101.Jpg'` 

安卓平板电脑会把它们重新排序成这样:

 `'Jia_01.Jpg', 'Jia_02.Jpg', 'Jia_09.Jpg', 'Jia_10.Jpg', 'Jia_101.Jpg', 'Jia_11.Jpg'` 

你拥有的无序文件越多，就越令人困惑。可悲的是，即使 Python 也是这样分类文件的。我尝试直接在上使用 **glob** 模块，然后对结果进行排序，得到了完全相同的问题。所以我要做的第一件事就是找到某种排序算法，能够正确地对它们进行排序。需要注意的是，Windows 7 可以在其文件系统中正确地对文件进行排序，尽管 Python 不能。

在 Google 上搜索了一下，我在 [StackOverflow](//stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python) 上找到了下面的脚本:

```py

import re

#----------------------------------------------------------------------
def sorted_nicely( l ): 
    """     
    Sort the given iterable in the way that humans expect.
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

```

效果非常好！现在我只需要找到一种方法将每个漫画页面放在他们自己的 PDF 页面上。幸运的是， [reportlab](http://www.reportlab.com/software/opensource/) 库使得这个任务很容易完成。您只需要迭代这些图像，然后一次一个地将它们插入到页面中。只看代码更容易，所以让我们这样做:

```py

import glob
import os
import re

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak
from reportlab.lib.units import inch

#----------------------------------------------------------------------
def sorted_nicely( l ): 
    """ 
    # http://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python

    Sort the given iterable in the way that humans expect.
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

#----------------------------------------------------------------------
def create_comic(fname, front_cover, back_cover, path):
    """"""
    filename = os.path.join(path, fname + ".pdf")
    doc = SimpleDocTemplate(filename,pagesize=letter,
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=18)
    Story=[]
    width = 7.5*inch
    height = 9.5*inch    

    pictures = sorted_nicely(glob.glob(path + "\\%s*" % fname))

    Story.append(Image(front_cover, width, height))
    Story.append(PageBreak())

    x = 0
    page_nums = {100:'%s_101-200.pdf', 200:'%s_201-300.pdf',
                 300:'%s_301-400.pdf', 400:'%s_401-500.pdf',
                 500:'%s_end.pdf'}
    for pic in pictures:
        parts = pic.split("\\")
        p = parts[-1].split("%s" % fname)
        page_num = int(p[-1].split(".")[0])
        print "page_num => ", page_num

        im = Image(pic, width, height)
        Story.append(im)
        Story.append(PageBreak())

        if page_num in page_nums.keys():
            print "%s created" % filename 
            doc.build(Story)
            filename = os.path.join(path, page_nums[page_num] % fname)
            doc = SimpleDocTemplate(filename,
                                    pagesize=letter,
                                    rightMargin=72,leftMargin=72,
                                    topMargin=72,bottomMargin=18)
            Story=[]
        print pic
        x += 1

    Story.append(Image(back_cover, width, height))
    doc.build(Story)
    print "%s created" % filename

#----------------------------------------------------------------------
if __name__ == "__main__":
    path = r"C:\Users\Mike\Desktop\Sam's Comics"
    front_cover = os.path.join(path, "FrontCover.jpg")
    back_cover = os.path.join(path, "BackCover2.jpg")
    create_comic("Jia_", front_cover, back_cover, path) 

```

让我们把它分解一下。像往常一样，您需要一些必要的导入，这些导入是代码工作所必需的。你会注意到我们之前提到的**排序良好的**函数也在这段代码中。主函数名为 **create_comic** ，接受四个参数:fname、front_cover、back_cover、path。如果您以前使用过 reportlab 工具包，那么您会认出 SimpleDocTemplate 和 Story list，因为它们直接来自 reportlab 教程。

无论如何，您循环遍历排序后的图片，并将图像和 PageBreak 对象一起添加到文章中。循环中有一个条件的原因是，我发现如果我试图用所有 400 多张图片构建 PDF，我会遇到内存错误。所以我把它分成一系列不超过 100 页的 PDF 文档。在文档的最后，您必须调用 **doc** 对象的 **build** 方法来实际创建 PDF 文档。

现在你知道我是如何将一大堆图片写入多个 PDF 文档的了。理论上，您可以使用 PyPdf 将所有生成的 Pdf 编织成一个 PDF，但是我没有尝试。您可能会遇到另一个内存错误。我将把它留给读者作为练习。

## 源代码

*   [comic_maker.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/01/comic_maker.zip)
*   [comic_maker.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/01/comic_maker.tar)