# Python PDF 系列 metaPDF 简介

> 原文：<https://www.blog.pythonlibrary.org/2012/07/21/python-pdf-series-an-intro-to-metapdf/>

在研究 Python 的 PDF 库时，我偶然发现了另一个名为 [metaPDF](https://github.com/aanari/metaPdf) 的小项目。根据其网站，metaPDF 是一个*轻量级 Python 库，针对元数据提取和插入进行了优化，它是优秀 pyPdf 库的快速包装器。它的工作原理是在解析 xref 表之前快速搜索 PDF 的最后 2048 个字节，与直接逐行解析表相比，性能提高了 50-60%。我不确定这有多有用，但是让我们试一试，看看 metaPDF 能做什么。*

### 获取和使用 metaPDF

metaPDF 的安装过程非常简单。用 easy_install 或者 pip 安装就可以了。接下来我们需要写一个小脚本来看看它是如何工作的。这里有一个基于 metaPDF 的 github 页面:

```py

from metapdf import MetaPdfReader

pdfOne = r'C:\Users\mdriscoll\Documents\reportlab-userguide.pdf'
x = MetaPdfReader()
metadata = x.read_metadata(open(pdfOne, 'rb'))
print metadata

```

在这里，我根据 Reportlab 用户指南 PDF 运行它。请注意，原始文件有一个打印错误，它使用了一个叫做“read”的东西来打开文件。我想，除非你跟踪了**打开**，否则没用。总之，这个脚本的输出如下:

```py

{'/ModDate': u'D:20120629155504', '/CreationDate': u'D:20120629155504', '/Producer': u'GPL Ghostscript 8.15', '/Title': u'reportlab-userguide.pdf', '/Creator': u'Adobe Acrobat 10.1.3', '/Author': u'mdriscoll'}

```

我真的不明白这份文件的作者是怎么被改的，但我确定我不是作者。我也不太明白为什么关键字段会有正斜杠。查看这个模块的源代码，似乎这就是它所能做的一切。这有点令人失望。也许通过吸引人们对这个库的注意，我们可以让开发人员在其中写入更多的功能？