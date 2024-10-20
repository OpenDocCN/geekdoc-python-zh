# rst2pdf 简介——用 Python 将重构的文本转换成 pdf

> 原文：<https://www.blog.pythonlibrary.org/2012/06/17/an-intro-to-rst2pdf-changing-restructured-text-into-pdfs-with-python/>

用 Python 创建 pdf 有几种很酷的方法。在本文中，我们将关注一个叫做 rst2pdf 的很酷的小工具，它获取一个包含重构文本的文本文件，并将其转换为 pdf。这个 [rst2pd](http://rst2pdf.ralsina.com.ar/) f 包需要 [Reportlab](http://www.reportlab.com/software/opensource/) 才能运行。这不会是一个关于重构文本的教程，尽管我们将不得不在某种程度上讨论它，只是为了理解发生了什么。

### 入门指南

首先，我们需要创建一个带有所需标记的文档。让我们先做那件事。这里有一些简单的重组文本，混合了一些指令。我们将在你有机会阅读代码后解释一切:

 `.. header::
Python Rules! - page ###Page###`

=====
Title
=====

这是一些废话

.. raw:: pdf

一列分页

新章节
=========

或者说

..代码块::python

import urllib
import urllib2
import webbrowser

URL = " http://duck duck go . com/html "
data = urllib . urlencode({ ' q ':' Python ' })
results = urllib 2 . urlopen(URL，data)
with open(" results . html "，" w ")as f:
f . write(results . read())

webbrowser . open(" results . html ")

前几行定义了每页的标题。在这种情况下，我们将有“Python 规则！”打印在每一页的顶部，并附有页码。还有其他几种特殊的散列标记插入指令可用。你应该查看官方文档以获得更多相关信息。然后我们有了一个**称号**。请注意，它的前面和后面有一串等号，与文本的长度相同。这告诉我们，这个文本将被样式化和居中。下面这一行只是出于演示目的的蹩脚句子。接下来是另一个特殊的指令，它告诉 rst2pdf 插入一个分页符。第二页包含一个章节标题、一个蹩脚的句子和一个用颜色标记的代码示例。

要生成 PDF，您需要在命令行上执行如下操作:

 `rst2pdf test.rst -o out.pdf` 

您还可以对配置文件运行 rst2pdf 来控制一些特殊的 pdf 指令，如页眉和页脚等。关于如何读取配置文件的信息有点令人困惑。听起来好像必须将文件放在一个特定的位置:/etc/rst2pdf.conf 和~/.rst2pdf/config。也有一个- config 标志可以传递，但我在网上发现了各种报告，这不起作用，所以您的里程可能会有所不同。在项目的 [repo](http://code.google.com/p/rst2pdf/source/browse/trunk/doc/config.sample?r=2180) 中有一个示例配置文件，您会发现它很有指导意义。

### 包扎

我希望 rst2pdf 能够提供一种简单的方法来指定绝对位置并创建行和框，这样我就可以用更简单的东西来替换我正在处理的 XSL / XML 项目。唉，在撰写本文时，rst2pdf 只是不支持 reportlab 本身所支持的行和框。然而，如果你需要一些易于使用的东西来创建你的文档，并且你已经知道重构的文本，我认为这是一个非常好的方法。您还可以将您重新构建的文本技能用于 [Sphinx 文档](http://sphinx.pocoo.org/index.html)项目。

### 进一步阅读

*   rst2pdf [手册](http://rst2pdf.ralsina.com.ar/handbook.html#id10)
*   重组文本[快速参考](http://docutils.sourceforge.net/docs/user/rst/quickref.html)
*   Pisa/XHTML 2 pdf-HTML/CSS 技能的替代者