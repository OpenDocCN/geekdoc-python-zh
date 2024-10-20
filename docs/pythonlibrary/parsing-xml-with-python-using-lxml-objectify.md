# 使用 lxml.objectify 用 Python 解析 XML

> 原文：<https://www.blog.pythonlibrary.org/2012/06/06/parsing-xml-with-python-using-lxml-objectify/>

几年前，我开始撰写一系列关于 XML 解析的文章。我介绍了 lxml 的 [etree](https://www.blog.pythonlibrary.org/2010/11/20/python-parsing-xml-with-lxml/) 和 Python 的 included [minidom](https://www.blog.pythonlibrary.org/2010/11/12/python-parsing-xml-with-minidom/ "Python: Parsing XML with minidom") XML 解析库。不管出于什么原因，我没有注意到 lxml 的 objectify 子包，但我最近看到了它，并决定应该检查一下。在我看来，objectify 模块似乎比 etree 更“Pythonic 化”。让我们花点时间回顾一下我以前使用 objectify 的 XML 例子，看看它有什么不同！

### 让我们开始派对吧！

如果你还没有，出去[下载 lxml](http://lxml.de/installation.html) ,否则你会跟不上。一旦你拿到了，我们可以继续。为了解析的方便，我们将使用下面这段 XML:

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

现在我们需要编写一些可以解析和修改 XML 的代码。让我们来看看这个小演示，它展示了 objectify 提供的一系列简洁的功能。

```py

from lxml import etree, objectify

#----------------------------------------------------------------------
def parseXML(xmlFile):
    """"""
    with open(xmlFile) as f:
        xml = f.read()

    root = objectify.fromstring(xml)

    # returns attributes in element node as dict
    attrib = root.attrib

    # how to extract element data
    begin = root.appointment.begin
    uid = root.appointment.uid

    # loop over elements and print their tags and text
    for appt in root.getchildren():
        for e in appt.getchildren():
            print "%s => %s" % (e.tag, e.text)
        print

    # how to change an element's text
    root.appointment.begin = "something else"
    print root.appointment.begin

    # how to add a new element
    root.appointment.new_element = "new data"

    # print the xml
    obj_xml = etree.tostring(root, pretty_print=True)
    print obj_xml

    # remove the py:pytype stuff
    #objectify.deannotate(root)
    etree.cleanup_namespaces(root)
    obj_xml = etree.tostring(root, pretty_print=True)
    print obj_xml

    # save your xml
    with open("new.xml", "w") as f:
        f.write(obj_xml)

#----------------------------------------------------------------------
if __name__ == "__main__":
    f = r'path\to\sample.xml'
    parseXML(f)

```

代码被很好地注释了，但是我们还是会花一点时间来检查它。首先，我们将样本 XML 文件传递给它，**将它对象化**。如果你想访问一个标签的属性，使用 **attrib** 属性。它将返回标签属性的字典。要访问子标签元素，只需使用点符号。正如你所看到的，要得到 **begin** 标签的值，我们可以这样做:

```py

begin = root.appointment.begin

```

如果需要迭代子元素，可以使用 **iterchildren** 。您可能必须使用嵌套的 **for** 循环结构来获取所有内容。改变一个元素的值就像给它分配一个新值一样简单。如果你需要创建一个新元素，只需添加一个句点和新元素的名称(如下所示):

```py

root.appointment.new_element = "new data"

```

当我们使用 objectify 添加或更改项目时，它会给 XML 添加一些注释，比如*xmlns:py = " http://code speak . net/lxml/objectify/pytype " py:py type = " str "*。您可能不希望包含这些内容，所以您必须调用以下方法来删除这些内容:

```py

etree.cleanup_namespaces(root)

```

您还可以使用“objectify.deannotate(root)”来做一些 deannotation 杂务，但是我无法让它在这个例子中工作。为了保存新的 XML，实际上似乎需要 lxml 的 etree 模块将它转换成一个字符串以便保存。

至此，您应该能够解析大多数 XML 文档，并使用 lxml 的 objectify 有效地编辑它们。我觉得很直观，很容易上手。希望你也会发现它对你的努力有用。

### 进一步阅读

*   lxml 对象化[文档](http://lxml.de/objectify.html)
*   [用 lxml objectify 解析 XML 的例子](http://www.saltycrane.com/blog/2011/07/example-parsing-xml-lxml-objectify/)
*   [用 lxml etree 解析 XML](https://www.blog.pythonlibrary.org/2010/11/20/python-parsing-xml-with-lxml/)