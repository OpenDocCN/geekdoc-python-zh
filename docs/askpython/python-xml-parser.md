# Python XML 解析器

> 原文：<https://www.askpython.com/python/examples/python-xml-parser>

有没有遇到过令人讨厌的 XML 文件，需要解析它才能获得重要的值？让我们学习如何创建一个 Python XML 解析器。

```py
<page>
    <header>
        <type heading="XML Parsing in Python"/>
        <type text="Hello from AskPython. We'll be parsing XML"/>
    </header>
</page>

```

我们将看看如何使用 Python 来解析这样的 XML 文件，以获得相关的属性和值。

我们开始吧！

* * *

## 方法 1:使用 ElementTree(推荐)

我们可以使用 [ElementTree](https://docs.python.org/3/library/xml.etree.elementtree.html) Python 库来实现这个任务。

这是构建 Python XML 解析器的最简单和推荐的选择，因为这个库在默认情况下与 Python *捆绑在一起。*

它不仅提供了方便的访问，因为它已经安装，而且它也相当快。让我们看看如何从测试文件中提取属性。

```py
<page>
    <header>
        <type heading="XML Parsing in Python"/>
        <type text="Hello from AskPython. We'll be parsing XML"/>
    </header>
</page>

```

我们将在核心`xml`包中使用`xml.etree.ElementTree`接口。

```py
import xml.etree.ElementTree as ET

```

### 构建 Python XML 解析器树

让我们首先构造这个解析树的根节点。这是树的最顶层节点，是我们开始解析所必需的。

谢天谢地，这个 API 已经为我们提供了下面的方法:

```py
import xml.etree.ElementTree as ET
root_node = ET.parse('sample.xml').getroot()
print(root_node)

```

这将自动读取 XML 输入文件，并为我们获取根节点。

**输出**

```py
<Element 'page' at 0x7f885836b2f0>

```

好的，看起来它已经被解析了。但是我们还无法证实。因此，让我们解析其他属性，并尝试获取其值。

### 获取相关属性的值

所以现在，我们的任务是使用 Python XML 解析器获取`<heading>`属性中的值。

它在根节点`<page>`的位置是`<header/type>`，所以我们需要遍历树的该层的所有匹配。

我们可以使用`root_node.findall(level)`来实现，其中**电平**是期望的位置(在我们的例子中是`<header/type>`)。

```py
for tag in root_node.find_all(level):
    value = tag.get(attribute)
    if value is not None: print(value)

```

在我们正在搜索的级别上，`tag.get(attribute)`将获得我们的`<attribute>`标签的值。所以，我们只需要在`<header/type>`做这件事，并获得`<heading>`和`<text>`属性的值。就是这样！

```py
import xml.etree.ElementTree as ET

# We're at the root node (<page>)
root_node = ET.parse('sample.xml').getroot()

# We need to go one level below to get <header>
# and then one more level from that to go to <type>
for tag in root_node.findall('header/type'):
    # Get the value of the heading attribute
    h_value = tag.get('heading')
    if h_value is not None:
        print(h_value)
    # Get the value of the text attribute
    t_value = tag.get('text')
    if t_value is not None:
        print(t_value)

```

**输出**

```py
XML Parsing in Python
Hello from AskPython. We'll be parsing XML

```

我们已经检索了 XML 解析树中该层的所有值！我们已经成功解析了 XML 文件。

再举一个例子，为了搞清楚一切。

现在，假设 XML 文件如下所示:

```py
<data>
    <items>
        <item name="item1">10</item>
        <item name="item2">20</item>
        <item name="item3">30</item>
        <item name="item4">40</item>
    </items>
</data>

```

这里，我们不仅要获得`name`的属性值，还要获得该级别的每个元素的文本值 10、20、30 和 40。

要得到`name`的属性值，我们可以像之前一样。我们也可以使用`tag.attrib[name]`来获取值。这与`tag.get(name)`相同，除了它使用字典查找。

```py
attr_value = tag.get(attr_name)
# Both methods are the same. You can
# choose any approach
attr_value = tag.attrib[attr_name]

```

要得到文本值，很简单！只需使用:

```py
tag.text

```

因此，我们这个解析器的完整程序将是:

```py
import xml.etree.ElementTree as ET

# We're at the root node (<page>)
root_node = ET.parse('sample.xml').getroot()

# We need to go one level below to get <items>
# and then one more level from that to go to <item>
for tag in root_node.findall('items/item'):
    # Get the value from the attribute 'name'
    value = tag.attrib['name']
    print(value)
    # Get the text of that tag
    print(tag.text)

```

**输出**

```py
item1
10
item2
20
item3
30
item4
40

```

对于任意长的 XML 文件，您也可以将这种逻辑扩展到任意级别！您还可以向另一个 XML 文件中写入一个新的解析树。

但是，我将让您从[文档](https://docs.python.org/3/library/xml.etree.elementtree.html)中找出答案，因为我已经为您提供了一个构建的起点！

## 方法 2:使用 BeautifulSoup(可靠)

如果由于某种原因，源 XML 的格式不正确，这也是另一个不错的选择。如果不对文件做一些预处理，XML 可能不能很好地工作。

事实证明， *BeautifulSoup* 对所有这些类型的文件都非常适用，所以如果您想解析任何类型的 XML 文件，请使用这种方法。

要安装它，使用`pip`并安装`bs4`模块:

```py
pip3 install bs4

```

我将为您提供我们之前的 XML 文件的一个小片段:

```py
<data>
    <items>
        <item name="item1">10</item>
        <item name="item2">20</item>
        <item name="item3">30</item>
        <item name="item4">40</item>
    </items>
</data>

```

我将传递这个文件，然后使用`bs4`解析它。

```py
from bs4 import BeautifulSoup

fd = open('sample.xml', 'r')

xml_file = fd.read()

soup = BeautifulSoup(xml_file, 'lxml')

for tag in soup.findAll("item"):
    # print(tag)
    print(tag["name"])
    print(tag.text)

fd.close()

```

语法类似于我们的`xml`模块，所以我们仍然使用`value = tag['attribute_name']`和`text = tag.text`来获取属性名。和以前一模一样！

**输出**

```py
item1
10
item2
20
item3
30
item4
40

```

我们现在已经用`bs4`解析过了！如果您的源文件`XML`格式不正确，这种方法是可行的，因为 BeautifulSoup 有不同的规则来处理这样的文件。

* * *

## 结论

希望您现在已经很好地掌握了如何轻松构建 Python XML 解析器。我们向您展示了两种方法:一种使用`xml`模块，另一种使用 **BeautifulSoup** 。

## 参考

*   关于解析 XML 的 StackOverflow 问题
*   [XML 模块](https://docs.python.org/3/library/xml.etree.elementtree.html)文档

* * *