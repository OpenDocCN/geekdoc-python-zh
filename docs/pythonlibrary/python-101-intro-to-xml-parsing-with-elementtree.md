# Python 101 -用 ElementTree 解析 XML 简介

> 原文：<https://www.blog.pythonlibrary.org/2013/04/30/python-101-intro-to-xml-parsing-with-elementtree/>

如果您关注这个博客已经有一段时间了，您可能还记得我们已经介绍了 Python 中包含的几个 XML 解析库。在本文中，我们将通过快速浏览 ElementTree 库来继续这个系列。您将学习如何创建 XML 文件、编辑 XML 和解析 XML。为了便于比较，我们将使用在之前的 [minidom 文章](https://www.blog.pythonlibrary.org/2010/11/12/python-parsing-xml-with-minidom/)中使用的相同 XML 来说明使用 minidom 和 ElementTree 之间的区别。下面是原始的 XML:

```py

 <appointment><begin>1181251680</begin>        
        <uid>040000008200E000</uid>
        <alarmtime>1181572063</alarmtime>
        <state><location><duration>1800</duration>
        <subject>Bring pizza home</subject></location></state></appointment> 

```

现在让我们深入研究一下 Python！

### 如何用 ElementTree 创建 XML

用 ElementTree 创建 XML 非常简单。在本节中，我们将尝试用 Python 创建上面的 XML。代码如下:

```py

import xml.etree.ElementTree as xml

#----------------------------------------------------------------------
def createXML(filename):
    """
    Create an example XML file
    """
    root = xml.Element("zAppointments")
    appt = xml.Element("appointment")
    root.append(appt)

    # add appointment children
    begin = xml.SubElement(appt, "begin")
    begin.text = "1181251680"

    uid = xml.SubElement(appt, "uid")
    uid.text = "040000008200E000"

    alarmTime = xml.SubElement(appt, "alarmTime")
    alarmTime.text = "1181572063"

    state = xml.SubElement(appt, "state")

    location = xml.SubElement(appt, "location")

    duration = xml.SubElement(appt, "duration")
    duration.text = "1800"

    subject = xml.SubElement(appt, "subject")

    tree = xml.ElementTree(root)
    with open(filename, "w") as fh:
        tree.write(fh)

#----------------------------------------------------------------------
if __name__ == "__main__":
    createXML("appt.xml")

```

如果您运行这段代码，您应该会得到如下所示的内容(可能都在一行中):

```py

 <appointment><begin>1181251680</begin>
        <uid>040000008200E000</uid>
        <alarmtime>1181572063</alarmtime>
        <state><location><duration>1800</duration></location></state></appointment> 

```

这非常接近原始的 XML，当然也是有效的 XML，但又不完全相同。不过，已经够近了。让我们花点时间回顾一下代码，确保我们理解了它。首先，我们使用 ElementTree 的**元素**函数创建根元素。然后，我们创建一个约会元素，并将其附加到根元素上。接下来，我们通过将约会元素对象(appt)和一个名称(如“begin ”)传递给子元素来创建子元素。然后，对于每个子元素，我们设置它的**文本**属性来赋予它一个值。在脚本的最后，我们创建了一个 ElementTree，并用它将 XML 写出到一个文件中。

令人恼火的是，它将 XML 全部写在一行上，而不是以一种易读的格式(即“漂亮的打印”)。Effbot 上有一个[秘方](http://effbot.org/zone/element-lib.htm#prettyprint)，但似乎没有内部实现的方法。你可能还想看看 [StackOverflow](http://stackoverflow.com/questions/749796/pretty-printing-xml-in-python) 上的其他一些解决方案。应该注意的是，lxml 支持开箱即用的“漂亮打印”。

现在我们准备学习如何编辑文件！

### 如何用 ElementTree 编辑 XML

用 ElementTree 编辑 XML 也很容易。为了让事情变得更有趣，我们将在 XML 中添加另一个约会块:

```py

 <appointment><begin>1181251680</begin>        
        <uid>040000008200E000</uid>
        <alarmtime>1181572063</alarmtime>
        <state><location><duration>1800</duration>
        <subject>Bring pizza home</subject></location></state></appointment> 
        <appointment><begin>1181253977</begin>        
        <uid>sdlkjlkadhdakhdfd</uid>
        <alarmtime>1181588888</alarmtime>
        <state>TX</state>
        <location>Dallas</location>
        <duration>1800</duration>
        <subject>Bring pizza home</subject></appointment> 

```

现在，让我们编写一些代码，将每个 **begin** 标签的值从 epoch 以来的秒数更改为可读性更好的值。我们将使用 Python 的时间模块来实现这一点:

```py

import time
import xml.etree.cElementTree as ET

#----------------------------------------------------------------------
def editXML(filename):
    """
    Edit an example XML file
    """
    tree = ET.ElementTree(file=filename)
    root = tree.getroot()

    for begin_time in root.iter("begin"):
        begin_time.text = time.ctime(int(begin_time.text))

    tree = ET.ElementTree(root)
    with open("updated.xml", "w") as f:
        tree.write(f)

#----------------------------------------------------------------------
if __name__ == "__main__":
    editXML("original_appt.xml")

```

这里我们创建了一个 ElementTree 对象(树),并从中提取了根。然后我们使用 ElementTree 的 **iter** ()方法来查找所有标记为“begin”的标签。注意，iter()方法是在 Python 2.7 中添加的。在我们的 **for** 循环中，我们通过 **time.ctime** ()，将每一项的 **text** 属性设置为更易于阅读的时间格式。您会注意到，在将字符串传递给 ctime 时，我们必须将其转换为整数。输出应该如下所示:

```py

 <appointment><begin>Thu Jun 07 16:28:00 2007</begin>        
        <uid>040000008200E000</uid>
        <alarmtime>1181572063</alarmtime>
        <state><location><duration>1800</duration>
        <subject>Bring pizza home</subject></location></state></appointment> 
    <appointment><begin>Thu Jun 07 17:06:17 2007</begin>        
        <uid>sdlkjlkadhdakhdfd</uid>
        <alarmtime>1181588888</alarmtime>
        <state>TX</state>
        <location>Dallas</location>
        <duration>1800</duration>
        <subject>Bring pizza home</subject></appointment> 

```

还可以使用 ElementTree 的 **find** ()或 **findall** ()方法在 XML 中搜索特定的标签。find()方法将只查找第一个实例，而 findall()将查找带有指定标签的所有标签。这些对于编辑和解析都很有帮助，这是我们的下一个主题！

### 如何用 ElementTree 解析 XML

现在我们开始学习如何用 ElementTree 做一些基本的解析。首先，我们将通读代码，然后我们将一点一点地浏览，以便我们能够理解它。注意，这段代码是基于最初的例子，但是它也应该适用于第二个例子。

```py

import xml.etree.cElementTree as ET

#----------------------------------------------------------------------
def parseXML(xml_file):
    """
    Parse XML with ElementTree
    """
    tree = ET.ElementTree(file=xml_file)
    print tree.getroot()
    root = tree.getroot()
    print "tag=%s, attrib=%s" % (root.tag, root.attrib)

    for child in root:
        print child.tag, child.attrib
        if child.tag == "appointment":
            for step_child in child:
                print step_child.tag

    # iterate over the entire tree
    print "-" * 40
    print "Iterating using a tree iterator"
    print "-" * 40
    iter_ = tree.getiterator()
    for elem in iter_:
        print elem.tag

    # get the information via the children!
    print "-" * 40
    print "Iterating using getchildren()"
    print "-" * 40
    appointments = root.getchildren()
    for appointment in appointments:
        appt_children = appointment.getchildren()
        for appt_child in appt_children:
            print "%s=%s" % (appt_child.tag, appt_child.text)

#----------------------------------------------------------------------
if __name__ == "__main__":
    parseXML("appt.xml")

```

您可能已经注意到了这一点，但是在这个例子和最后一个例子中，我们一直在导入 cElementTree，而不是普通的 ElementTree。两者的主要区别在于 cElementTree 是基于 C 而不是基于 Python 的，所以速度要快得多。无论如何，我们再次创建一个 ElementTree 对象并从中提取根。您会注意到 e 打印出了根以及根的标签和属性。接下来，我们展示几种迭代标签的方法。第一个循环只是一个接一个地遍历 XML。不过这只会打印出顶层的子节点(约会),所以我们添加了一个 if 语句来检查这个子节点，并遍历它的子节点。

接下来，我们从树对象本身获取一个迭代器，并以这种方式遍历它。您将获得相同的信息，但是没有第一个示例中的额外步骤。第三种方法使用根的 **getchildren** ()函数。这里我们再次需要一个内部循环来获取每个约会标记中的所有子元素。最后一个例子使用根的 iter()方法遍历所有匹配字符串“begin”的标签。

正如上一节所提到的，您还可以使用 find()或 findall()分别帮助您找到特定的标签或标签集。还要注意，每个元素对象都有一个**标签**和一个**文本**属性，您可以使用它们来获取准确的信息。

### 包扎

现在您知道了如何使用 ElementTree 来创建、编辑和解析 XML。您可以将这些信息添加到 XML 解析工具包中，并将其用于娱乐或盈利目的。您可以在下面找到以前关于其他 XML 解析工具的文章的链接，以及关于 ElementTree 本身的其他信息。

### 来自鼠标 Vs Python 的相关文章

*   [用 minidom 解析 XML](https://www.blog.pythonlibrary.org/2010/11/12/python-parsing-xml-with-minidom/)
*   Python: [用 lxml 解析 xml](https://www.blog.pythonlibrary.org/2010/11/20/python-parsing-xml-with-lxml/)
*   [使用 lxml.objectify 用 Python 解析 XML](https://www.blog.pythonlibrary.org/2012/06/06/parsing-xml-with-python-using-lxml-objectify/)

### 附加阅读

*   [元素树](http://docs.python.org/2/library/xml.etree.elementtree.html)的官方 Python 文档
*   Effbot 的[元素树概述](http://effbot.org/zone/element-index.htm)
*   [用 elementree 在 Python 中处理 XML](http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
*   查克博士的[简单元素树示例](http://www.dr-chuck.com/csev-blog/2008/09/a-simple-python-elementtree-example/)

### 下载源代码

*   [ETXMLParsing.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2013/04/ETXMLParsing.zip)