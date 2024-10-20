# Python:用 minidom 解析 XML

> 原文：<https://www.blog.pythonlibrary.org/2010/11/12/python-parsing-xml-with-minidom/>

如果你是一个长期读者，你可能记得我在 2006 年开始编程 Python。在一年左右的时间里，我的雇主决定从 Microsoft Exchange 迁移到开源的 Zimbra 客户端。Zimbra 是一个不错的客户端，但它缺少一种好的方式来提醒用户他们有一个约会，所以我必须创建一种方法来查询 Zimbra 的信息并显示一个对话框。但是，所有这些晦涩难懂的东西与 XML 有什么关系呢？嗯，我认为使用 XML 是跟踪哪些约会被添加、删除、暂停或其他什么的好方法。结果证明我错了，但这不是这个故事的重点。

在本文中，我们将看到我第一次尝试用 Python 解析 XML。如果您对这个主题做一点研究，您很快就会发现 Python 在它的 **xml** 模块中内置了一个 XML 解析器。我最终使用了那个模块的 **minidom** 子组件...至少一开始是这样。最终我改用 lxml，它使用 ElementTree，但这超出了本文的范围。让我们快速看一下我想到的一些难看的 XML:

```py

 <appointment><begin>1181251680</begin>        
        <uid>040000008200E000</uid>
        <alarmtime>1181572063</alarmtime>
        <state><location><duration>1800</duration>
        <subject>Bring pizza home</subject></location></state></appointment> 

```

现在我们知道我需要解析什么了。让我们看看在 Python 中使用 minidom 解析类似内容的典型方式。

```py

import xml.dom.minidom
import urllib2

class ApptParser(object):

    def __init__(self, url, flag='url'):
        self.list = []
        self.appt_list = []        
        self.flag = flag
        self.rem_value = 0
        xml = self.getXml(url) 
        print "xml"
        print xml
        self.handleXml(xml)

    def getXml(self, url):
        try:
            print url
            f = urllib2.urlopen(url)
        except:
            f = url
        #print f
        doc = xml.dom.minidom.parse(f)
        node = doc.documentElement        
        if node.nodeType == xml.dom.Node.ELEMENT_NODE:
            print 'Element name: %s' % node.nodeName
            for (name, value) in node.attributes.items():
                #print '    Attr -- Name: %s  Value: %s' % (name, value)
                if name == 'reminder':
                    self.rem_value = value                    

        return node

    def handleXml(self, xml):
        rem = xml.getElementsByTagName('zAppointments')        
        appointments = xml.getElementsByTagName("appointment")
        self.handleAppts(appointments)

    def getElement(self, element):
        return self.getText(element.childNodes)

    def handleAppts(self, appts):
        for appt in appts:
            self.handleAppt(appt)
            self.list = []

    def handleAppt(self, appt):
        begin     = self.getElement(appt.getElementsByTagName("begin")[0])
        duration  = self.getElement(appt.getElementsByTagName("duration")[0])
        subject   = self.getElement(appt.getElementsByTagName("subject")[0])
        location  = self.getElement(appt.getElementsByTagName("location")[0])
        uid       = self.getElement(appt.getElementsByTagName("uid")[0])

        self.list.append(begin)
        self.list.append(duration)
        self.list.append(subject)
        self.list.append(location)
        self.list.append(uid)
        if self.flag == 'file':

            try:
                state     = self.getElement(appt.getElementsByTagName("state")[0])
                self.list.append(state)
                alarm     = self.getElement(appt.getElementsByTagName("alarmTime")[0])
                self.list.append(alarm)
            except Exception, e:
                print e

        self.appt_list.append(self.list)        

    def getText(self, nodelist):
        rc = ""
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc = rc + node.data
        return rc

```

如果我没记错的话，这段代码是基于 Python 文档中的一个例子(或者是深入 Python 中的一个章节)。我还是不喜欢这个代码。您在 **ApptParser** 类中看到的 **url** 参数可以是 url，也可以是文件。我有一个来自 Zimbra 的 XML 提要，我会定期检查它的变化，并将其与我下载的 XML 的最后一个副本进行比较。如果有新的东西，我会把修改添加到下载的副本中。无论如何，让我们稍微解开这个代码。

在 **getXml** 中，我们使用一个异常处理程序来尝试打开 url。如果它碰巧引发了一个错误，那么我们假设这个 url 实际上是一个文件路径。接下来，我们使用 minidom 的**解析**方法来解析 XML。然后我们从 XML 中取出一个节点。我们将忽略条件句，因为它对这个讨论不重要(它与我的程序有关)。最后，我们返回节点对象。

从技术上讲，节点是 XML，我们将它传递给 **handleXml** 。为了获取 XML 中所有的**约会**实例，我们这样做: *xml.getElementsByTagName("约会")*。然后，我们将该信息传递给**处理设备**方法。是的，到处都在传递各种价值观。试图跟踪它并在以后调试它让我发疯。总之，*handle appt*方法所做的就是循环每个约会，并调用 **handleAppt** 方法从中提取一些附加信息，将数据添加到一个列表，并将该列表添加到另一个列表。我的想法是以一个包含所有与我的约会相关的数据的列表结束。

你会注意到 **handleAppt** 方法调用了 **getElement** 方法，后者调用了 **getText** 方法。我不知道原作者为什么要这样做。我会直接调用 **getText** 方法，跳过 getElement 方法。亲爱的读者，我想这对你是一种锻炼。

现在您已经了解了使用 minidom 进行解析的基础知识。我个人不喜欢这种方法，所以我决定尝试用 minidom 提出一种更简洁的解析 XML 的方法。

## 让 minidom 更容易关注

我不会说我的代码有什么好的，但是我会说我认为我想出了一些更容易理解的东西。我肯定有些人会认为代码不够灵活，但是没关系。下面是我们将要解析的一个新的 XML 示例(在 [MSDN](http://msdn.microsoft.com/en-us/library/ms762271%28VS.85%29.aspx) 上找到的):

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

对于这个例子，我们只需要解析 XML，提取书名并打印到 stdout。你准备好了吗？开始了。

```py

import xml.dom.minidom as minidom

#----------------------------------------------------------------------
def getTitles(xml):
    """
    Print out all titles found in xml
    """
    doc = minidom.parse(xml)
    node = doc.documentElement
    books = doc.getElementsByTagName("book")

    titles = []
    for book in books:
        titleObj = book.getElementsByTagName("title")[0]
        titles.append(titleObj)

    for title in titles:
        nodes = title.childNodes
        for node in nodes:
            if node.nodeType == node.TEXT_NODE:
                print node.data

if __name__ == "__main__":
    document = 'example.xml'
    getTitles(document)

```

这段代码只是一个接受一个参数(XML 文件)的简短函数。我们导入 minidom 模块，并给它相同的名称，以便更容易引用。然后我们解析 XML。函数中的前两行与前一个例子非常相似。我们使用 **getElementsByTagName** 获取我们想要的 XML 部分，然后迭代结果并从中提取书名。这实际上提取了标题对象，所以我们也需要对其进行迭代并提取纯文本，这就是第二个嵌套的 **for** 循环的目的。

就是这样。再也没有了。

## 包扎

好吧，我希望这篇漫无边际的文章能够教会您一些关于使用 Python 内置的 XML 解析器解析 XML 的知识。在以后的文章中，我们将会看到更多的 XML 解析。如果您有自己喜欢的方法或模块，请随意给我指出来，我会看一看。

## 附加阅读

*   Python minidom [官方文档](http://docs.python.org/library/xml.dom.minidom.html)
*   Python 和 XML [维基页面](http://wiki.python.org/moin/PythonXml)
*   Python 的其他内置 XML 解析器: [ElementTree](http://docs.python.org/library/xml.etree.elementtree.html) 、 [sax、](http://docs.python.org/library/xml.sax.html)、 [expat](http://docs.python.org/library/pyexpat.html) 和 [dom](http://docs.python.org/library/xml.dom.html)