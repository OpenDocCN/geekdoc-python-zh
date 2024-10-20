# Python:用 lxml 解析 XML

> 原文：<https://www.blog.pythonlibrary.org/2010/11/20/python-parsing-xml-with-lxml/>

上次，我们看了 Python 的一个内置 XML 解析器。在本文中，我们将看看有趣的第三方包，来自 [codespeak](http://codespeak.net/lxml/) 的 lxml。它使用了 ElementTree API 等。lxml 包支持 XPath 和 XSLT，包括一个用于 SAX 的 API 和一个与 C/Pyrex 模块兼容的 C 级 API。我们只是用它做一些简单的事情。

无论如何，对于本文，我们将使用 minidom 解析示例中的例子，看看如何用 lxml 解析这些例子。下面是一个 XML 示例，它来自一个为跟踪约会而编写的程序:

```py

 <appointment><begin>1181251680</begin>
        <uid>040000008200E000</uid>
        <alarmtime>1181572063</alarmtime>
        <state><location><duration>1800</duration>
        <subject>Bring pizza home</subject></location></state></appointment> 
    <appointment><begin>1234360800</begin>
        <duration>1800</duration>
        <subject>Check MS Office website for updates</subject>
        <location><uid>604f4792-eb89-478b-a14f-dd34d3cc6c21-1234360800</uid>
        <state>dismissed</state></location></appointment> 

```

上面的 XML 显示了两个约会。从纪元开始的开始时间以秒为单位；uid 是基于开始时间和一个密钥(我认为)的哈希生成的；报警时间是自该时期以来的秒数，但应该小于开始时间；而状态就是任命有没有被打盹儿，有没有被辞退。其余的就不言自明了。现在让我们看看如何解析它。

```py

from lxml import etree
from StringIO import StringIO

#----------------------------------------------------------------------
def parseXML(xmlFile):
    """
    Parse the xml
    """
    f = open(xmlFile)
    xml = f.read()
    f.close()

    tree = etree.parse(StringIO(xml))
    context = etree.iterparse(StringIO(xml))
    for action, elem in context:
        if not elem.text:
            text = "None"
        else:
            text = elem.text
        print elem.tag + " => " + text   

if __name__ == "__main__":
    parseXML("example.xml")

```

首先，我们导入所需的模块，即来自 **lxml** 包的 **etree** 模块和来自内置 StringIO 模块的 **StringIO** 函数。我们的 **parseXML** 函数接受一个参数:所讨论的 XML 文件的路径。我们打开文件，阅读并关闭它。现在有趣的部分来了！我们使用 etree 的 parse 函数来解析从 StringIO 模块返回的 XML 代码。出于我不完全理解的原因，parse 函数需要一个类似文件的对象。

无论如何，接下来我们迭代上下文(即 lxml.etree.iterparse 对象)并提取标记元素。我们添加条件语句 **if** 来用单词“None”替换空字段，以使输出更加清晰。仅此而已。

## 解析图书示例

这个例子的结果有点蹩脚。大多数情况下，您希望保存提取的数据并对其进行处理，而不仅仅是将其输出到 stdout。因此，对于我们的下一个例子，我们将创建一个数据结构来包含结果。这个例子的数据结构将是一个字典列表。我们将在这里使用 MSDN 图书的例子:

```py

 <book id="bk101"><author>Gambardella, Matthew</author>
      <title>XML Developer's Guide</title>
      <genre>Computer</genre>
      <price>44.95</price>
      <publish_date>2000-10-01</publish_date>
      <description>An in-depth look at creating applications 
      with XML.</description></book> 
   <book id="bk102"><author>Ralls, Kim</author>
      <title>Midnight Rain</title>
      <genre>Fantasy</genre>
      <price>5.95</price>
      <publish_date>2000-12-16</publish_date>
      <description>A former architect battles corporate zombies, 
      an evil sorceress, and her own childhood to become queen 
      of the world.</description></book> 
   <book id="bk103"><author>Corets, Eva</author>
      <title>Maeve Ascendant</title>
      <genre>Fantasy</genre>
      <price>5.95</price>
      <publish_date>2000-11-17</publish_date>
      <description>After the collapse of a nanotechnology 
      society in England, the young survivors lay the 
      foundation for a new society.</description></book> 
   <book id="bk104"><author>Corets, Eva</author>
      <title>Oberon's Legacy</title>
      <genre>Fantasy</genre>
      <price>5.95</price>
      <publish_date>2001-03-10</publish_date>
      <description>In post-apocalypse England, the mysterious 
      agent known only as Oberon helps to create a new life 
      for the inhabitants of London. Sequel to Maeve 
      Ascendant.</description></book> 
   <book id="bk105"><author>Corets, Eva</author>
      <title>The Sundered Grail</title>
      <genre>Fantasy</genre>
      <price>5.95</price>
      <publish_date>2001-09-10</publish_date>
      <description>The two daughters of Maeve, half-sisters, 
      battle one another for control of England. Sequel to 
      Oberon's Legacy.</description></book> 
   <book id="bk106"><author>Randall, Cynthia</author>
      <title>Lover Birds</title>
      <genre>Romance</genre>
      <price>4.95</price>
      <publish_date>2000-09-02</publish_date>
      <description>When Carla meets Paul at an ornithology 
      conference, tempers fly as feathers get ruffled.</description></book> 
   <book id="bk107"><author>Thurman, Paula</author>
      <title>Splish Splash</title>
      <genre>Romance</genre>
      <price>4.95</price>
      <publish_date>2000-11-02</publish_date>
      <description>A deep sea diver finds true love twenty 
      thousand leagues beneath the sea.</description></book> 
   <book id="bk108"><author>Knorr, Stefan</author>
      <title>Creepy Crawlies</title>
      <genre>Horror</genre>
      <price>4.95</price>
      <publish_date>2000-12-06</publish_date>
      <description>An anthology of horror stories about roaches,
      centipedes, scorpions  and other insects.</description></book> 
   <book id="bk109"><author>Kress, Peter</author>
      <title>Paradox Lost</title>
      <genre>Science Fiction</genre>
      <price>6.95</price>
      <publish_date>2000-11-02</publish_date>
      <description>After an inadvertant trip through a Heisenberg
      Uncertainty Device, James Salway discovers the problems 
      of being quantum.</description></book> 
   <book id="bk110"><author>O'Brien, Tim</author>
      <title>Microsoft .NET: The Programming Bible</title>
      <genre>Computer</genre>
      <price>36.95</price>
      <publish_date>2000-12-09</publish_date>
      <description>Microsoft's .NET initiative is explored in 
      detail in this deep programmer's reference.</description></book> 
   <book id="bk111"><author>O'Brien, Tim</author>
      <title>MSXML3: A Comprehensive Guide</title>
      <genre>Computer</genre>
      <price>36.95</price>
      <publish_date>2000-12-01</publish_date>
      <description>The Microsoft MSXML3 parser is covered in 
      detail, with attention to XML DOM interfaces, XSLT processing, 
      SAX and more.</description></book> 
   <book id="bk112"><author>Galos, Mike</author>
      <title>Visual Studio 7: A Comprehensive Guide</title>
      <genre>Computer</genre>
      <price>49.95</price>
      <publish_date>2001-04-16</publish_date>
      <description>Microsoft Visual Studio 7 is explored in depth,
      looking at how Visual Basic, Visual C++, C#, and ASP+ are 
      integrated into a comprehensive development 
      environment.</description></book> 

```

现在让我们解析它，并把它放到我们的数据结构中！

```py

from lxml import etree
from StringIO import StringIO

#----------------------------------------------------------------------
def parseBookXML(xmlFile):

    f = open(xmlFile)
    xml = f.read()
    f.close()

    tree = etree.parse(StringIO(xml))
    print tree.docinfo.doctype
    context = etree.iterparse(StringIO(xml))
    book_dict = {}
    books = []
    for action, elem in context:
        if not elem.text:
            text = "None"
        else:
            text = elem.text
        print elem.tag + " => " + text
        book_dict[elem.tag] = text
        if elem.tag == "book":
            books.append(book_dict)
            book_dict = {}
    return books

if __name__ == "__main__":
    parseBookXML("example2.xml")

```

这个例子与上一个非常相似，所以我们只关注这里的不同之处。在开始迭代上下文之前，我们创建了一个空的 dictionary 对象和一个空的 list。然后在循环内部，我们像这样创建字典:

```py

book_dict[elem.tag] = text

```

文本为 elem.text 或“无”。最后，如果标签碰巧是“book ”,那么我们在一本书的末尾，需要将字典添加到我们的列表中，并为下一本书重置字典。如你所见，这正是我们所做的。更现实的例子是将提取的数据放入 Book 类。我以前用 json 提要做过后者。

## 重构代码

正如我警惕的读者所指出的，我写了一些相当糟糕的代码。所以我对代码进行了一些清理，希望这样会好一点:

```py

from lxml import etree

#----------------------------------------------------------------------
def parseBookXML(xmlFile):
    """"""

    context = etree.iterparse(xmlFile)
    book_dict = {}
    books = []
    for action, elem in context:
        if not elem.text:
            text = "None"
        else:
            text = elem.text
        print elem.tag + " => " + text
        book_dict[elem.tag] = text
        if elem.tag == "book":
            books.append(book_dict)
            book_dict = {}
    return books

if __name__ == "__main__":
    parseBookXML("example.xml")

```

如您所见，我们完全放弃了 StringIO 模块，将所有文件 I/O 内容放在 lxml 方法调用中。其余都一样。很酷吧？像往常一样，巨蟒摇滚！

## 包扎

你从这篇文章中学到什么了吗？我当然希望如此。Python 在其标准库内外都有很多很酷的解析库。一定要检查它们，看看哪一个最适合你的编程方式。

## 进一步阅读

*   lxml 官方[网站](http://codespeak.net/lxml/index.html)
*   一篇关于 lxml 的 IBM 文章
*   StringIO [文档](http://docs.python.org/library/stringio.html)