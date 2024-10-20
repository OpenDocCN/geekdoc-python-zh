# Python:用 lxml.objectify 创建 XML

> 原文：<https://www.blog.pythonlibrary.org/2014/03/26/python-creating-xml-with-lxml-objectify/>

lxml.objectify 子包对于解析和创建 xml 非常方便。在本文中，我们将展示如何使用 lxml 包创建 XML。我们将从一些简单的 XML 开始，然后尝试复制它。我们开始吧！

在过去的[篇文章](https://www.blog.pythonlibrary.org/2012/06/06/parsing-xml-with-python-using-lxml-objectify/)中，我使用了以下愚蠢的 XML 示例进行演示:

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

让我们看看如何使用 lxml.objectify 来重新创建这个 xml:

```py

from lxml import etree, objectify

#----------------------------------------------------------------------
def create_appt(data):
    """
    Create an appointment XML element
    """
    appt = objectify.Element("appointment")
    appt.begin = data["begin"]
    appt.uid = data["uid"]
    appt.alarmTime = data["alarmTime"]
    appt.state = data["state"]
    appt.location = data["location"]
    appt.duration = data["duration"]
    appt.subject = data["subject"]
    return appt

#----------------------------------------------------------------------
def create_xml():
    """
    Create an XML file
    """

    xml = '''

    '''

    root = objectify.fromstring(xml)
    root.set("reminder", "15")

    appt = create_appt({"begin":1181251680,
                        "uid":"040000008200E000",
                        "alarmTime":1181572063,
                        "state":"",
                        "location":"",
                        "duration":1800,
                        "subject":"Bring pizza home"}
                       )
    root.append(appt)

    uid = "604f4792-eb89-478b-a14f-dd34d3cc6c21-1234360800"
    appt = create_appt({"begin":1234360800,
                        "uid":uid,
                        "alarmTime":1181572063,
                        "state":"dismissed",
                        "location":"",
                        "duration":1800,
                        "subject":"Check MS Office website for updates"}
                       )
    root.append(appt)

    # remove lxml annotation
    objectify.deannotate(root)
    etree.cleanup_namespaces(root)

    # create the xml string
    obj_xml = etree.tostring(root,
                             pretty_print=True,
                             xml_declaration=True)

    try:
        with open("example.xml", "wb") as xml_writer:
            xml_writer.write(obj_xml)
    except IOError:
        pass

#----------------------------------------------------------------------
if __name__ == "__main__":
    create_xml()

```

让我们把它分解一下。我们将从 **create_xml** 函数开始。在其中，我们使用 objectify 模块的 **fromstring** 函数创建了一个 XML 根对象。根对象将包含 **zAppointment** 作为它的标签。我们设置根的**提醒**属性，然后使用字典作为参数调用我们的 **create_appt** 函数。在 **create_appt** 函数中，我们创建了一个元素的实例(从技术上讲，它是一个 **ObjectifiedElement** ),我们将它分配给了我们的 **appt** 变量。这里我们使用点符号来创建这个元素的标签。最后，我们返回 appt 元素并将其附加到我们的根对象中。我们对第二个约会实例重复这个过程。

**create_xml** 函数的下一部分将删除 lxml 注释。如果您不这样做，您的 XML 将看起来像下面这样:

```py

 <appointment py:pytype="TREE"><begin py:pytype="int">1181251680</begin>
        <uid py:pytype="str">040000008200E000</uid>
        <alarmtime py:pytype="int">1181572063</alarmtime>
        <state py:pytype="str"><location py:pytype="str"><duration py:pytype="int">1800</duration>
        <subject py:pytype="str">Bring pizza home</subject></location></state></appointment> <appointment py:pytype="TREE"><begin py:pytype="int">1234360800</begin>
        <uid py:pytype="str">604f4792-eb89-478b-a14f-dd34d3cc6c21-1234360800</uid>
        <alarmtime py:pytype="int">1181572063</alarmtime>
        <state py:pytype="str">dismissed</state>
        <location py:pytype="str"><duration py:pytype="int">1800</duration>
        <subject py:pytype="str">Check MS Office website for updates</subject></location></appointment> 

```

为了删除所有不需要注释，我们调用以下两个函数:

```py

objectify.deannotate(root)
etree.cleanup_namespaces(root)

```

难题的最后一部分是让 lxml 自己生成 xml。这里我们使用 lxml 的 etree 模块来完成这项艰巨的工作:

```py

obj_xml = etree.tostring(root, pretty_print=True)

```

**tostring** 函数将返回一个漂亮的 XML 字符串，如果您将 **pretty_print** 设置为 True，它通常也会以漂亮的格式返回 XML。

#### Python 3 的 2020 年 11 月更新

在 Python 3 中，上一节中的代码没有将“修饰过的”XML 输出到文件中。要使它正常工作，你必须经历更多的困难。这里是一个更新版本的代码。在 Mac OS 上的 **Python 3.9 中测试**:

```py

from lxml import etree, objectify
from io import BytesIO

def create_appt(data):
    """
    Create an appointment XML element
    """
    appt = objectify.Element("appointment")
    appt.begin = data["begin"]
    appt.uid = data["uid"]
    appt.alarmTime = data["alarmTime"]
    appt.state = data["state"]
    appt.location = data["location"]
    appt.duration = data["duration"]
    appt.subject = data["subject"]
    return appt

def create_xml():
    """
    Create an XML file
    """

    xml = '''

    '''

    root = objectify.fromstring(xml)
    root.set("reminder", "15")

    appt = create_appt({"begin":1181251680,
                        "uid":"040000008200E000",
                        "alarmTime":1181572063,
                        "state":"",
                        "location":"",
                        "duration":1800,
                        "subject":"Bring pizza home"}
                       )
    root.append(appt)

    uid = "604f4792-eb89-478b-a14f-dd34d3cc6c21-1234360800"
    appt = create_appt({"begin":1234360800,
                        "uid":uid,
                        "alarmTime":1181572063,
                        "state":"dismissed",
                        "location":"",
                        "duration":1800,
                        "subject":"Check MS Office website for updates"}
                       )
    root.append(appt)

    # remove lxml annotation
    objectify.deannotate(root)
    etree.cleanup_namespaces(root)

    # create the xml string
    parser = etree.XMLParser(remove_blank_text=True)
    file_obj = BytesIO(etree.tostring(root))
    tree = etree.parse(file_obj, parser)

    try:
        with open("example.xml", "wb") as xml_writer:
            tree.write(xml_writer, pretty_print=True)
    except IOError:
        pass

if __name__ == "__main__":
    create_xml()

```

这是基于在 [StackOverflow](https://stackoverflow.com/questions/5086922/python-pretty-xml-printer-with-lxml) 上找到的解决方案。

您需要在文件的顶部添加一个新的导入来获得**字节数**。然后在代码的结尾，你需要修改你的代码，看起来像这样:

```py

# create the xml string
parser = etree.XMLParser(remove_blank_text=True)
file_obj = BytesIO(etree.tostring(root))
tree = etree.parse(file_obj, parser)

try:
    with open("example.xml", "wb") as xml_writer:
        tree.write(xml_writer, pretty_print=True)
except IOError:
    pass

```

这将添加一个新的 XML 解析器对象，从根目录中删除空白文本。这将发生在你把根变成一个字节串，而这个字节串本身又用 BytesIO 变成一个类似文件的对象之后。试一试，您应该得到一个包含适当缩进的 XML 代码的文件。

* * *

### 包扎

现在您知道了如何使用 lxml 的 objectify 模块来创建 xml。这是一个非常方便的界面，并且在很大程度上相当 Pythonic 化。

* * *

### 相关阅读

*   [使用 lxml.objectify 用 Python 解析 XML](https://www.blog.pythonlibrary.org/2012/06/06/parsing-xml-with-python-using-lxml-objectify/)
*   Python: [用 lxml.etree 解析 XML](https://www.blog.pythonlibrary.org/2010/11/20/python-parsing-xml-with-lxml/)
*   Python 101 - [元素树 XML 解析简介](https://www.blog.pythonlibrary.org/2013/04/30/python-101-intro-to-xml-parsing-with-elementtree/)