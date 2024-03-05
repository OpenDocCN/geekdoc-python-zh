# Python 中 XML 解析器的路线图

> 原文：<https://realpython.com/python-xml-parser/>

如果你以前曾经尝试过用 Python 解析一个 **XML 文档**,那么你就会知道这样的任务有多困难。一方面，Python 的[禅只承诺一个显而易见的方法来实现你的目标。与此同时，标准库遵循](https://www.python.org/dev/peps/pep-0020/)[电池内置](https://docs.python.org/3/tutorial/stdlib.html#batteries-included)的格言，让您从不止一个而是几个 XML 解析器中进行选择。幸运的是，Python 社区通过创建更多的 XML 解析库解决了这个多余的问题。

玩笑归玩笑，在这个充满或大或小挑战的世界里，所有 XML 解析器都有自己的位置。熟悉可用的工具是值得的。

**在本教程中，您将学习如何:**

*   选择正确的 XML **解析模型**
*   使用**标准库中的 XML 解析器**
*   使用主要的 XML 解析**库**
*   使用**数据绑定**以声明方式解析 XML 文档
*   使用安全的 XML 解析器消除安全漏洞

您可以将本教程作为路线图来引导您穿过 Python 中令人困惑的 XML 解析器世界。结束时，您将能够为给定的问题选择正确的 XML 解析器。为了从本教程中获得最大收益，您应该已经熟悉了 [XML](https://en.wikipedia.org/wiki/XML) 及其构建模块，以及如何在 Python 中使用[处理文件。](https://realpython.com/working-with-files-in-python/)

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 选择正确的 XML 解析模型

事实证明，您可以使用一些与语言无关的策略来处理 XML 文档。每一种都展示了不同的内存和速度权衡，这可以部分地证明 Python 中可用的 XML 解析器的多样性。在接下来的部分，你会发现他们的不同和优势。

[*Remove ads*](/account/join/)

### 文档对象模型

历史上，解析 XML 的第一个也是最广泛的模型是 DOM，或最初由万维网联盟(W3C)定义的[文档对象模型](https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model)。你可能已经听说过 DOM，因为网络浏览器通过 [JavaScript](https://realpython.com/python-vs-javascript/) 公开了一个 DOM 接口，让你操作你网站的 HTML 代码。XML 和 HTML 都属于同一家族的[标记语言](https://en.wikipedia.org/wiki/Markup_language)，这使得用 DOM 解析 XML 成为可能。

DOM 可以说是最简单和最通用的模型。它定义了一些标准操作**来遍历和修改对象层次结构中的文档元素。整个文档树的抽象表示存储在内存中，让您可以对单个元素进行**随机访问**。**

虽然 DOM 树允许快速和全方位的导航，但首先构建它的抽象表示可能很耗时。此外，作为一个整体，XML 会被立刻**解析，所以它必须足够小以适应可用的内存。这使得 DOM 只适合中等大小的配置文件，而不是几千兆字节的 [XML 数据库](https://en.wikipedia.org/wiki/XML_database)。**

当便利性比处理时间更重要，并且内存不是问题时，使用 DOM 解析器。一些典型的用例是当您需要解析一个相对较小的文档时，或者当您只需要偶尔进行解析时。

### XML 的简单应用编程接口(SAX)

为了解决 DOM 的缺点，Java 社区通过合作开发出了一个库，这个库后来成为用其他语言解析 XML 的替代模型。没有正式的规范，只有邮件列表上的有机讨论。最终结果是一个基于事件的流 API ,它对单个元素而不是整个树进行顺序操作。

元素按照它们在文档中出现的顺序从上到下进行处理。解析器触发用户定义的[回调](https://en.wikipedia.org/wiki/Callback_(computer_programming))来处理在文档中找到的特定 XML 节点。这种方法被称为**“推”解析**，因为元素是由解析器推送到函数中的。

SAX 还允许您丢弃不感兴趣的元素。这意味着它比 DOM 占用的内存少得多，并且可以处理任意大的文件，这对于**单遍处理**来说非常好，比如索引、转换成其他格式等等。

然而，查找或修改随机的树节点很麻烦，因为它通常需要多次遍历文档并跟踪被访问的节点。SAX 也不方便处理深度嵌套的元素。最后，SAX 模型只允许**只读**解析。

简而言之，SAX 在空间和时间上很便宜，但是在大多数情况下比 DOM 更难使用。它非常适合解析非常大的文档或实时解析输入的 XML 数据。

### XML 流应用编程接口(StAX)

虽然在 Python 中不太流行，但这第三种解析 XML 的方法是建立在 SAX 之上的。它扩展了**流**的概念，但是使用了一个**“拉”解析**模型，这给了你更多的控制。您可以将 StAX 想象成一个[迭代器](https://docs.python.org/3/glossary.html#term-iterator)，通过 XML 文档推进一个**光标对象**，其中自定义处理程序按需调用解析器，而不是相反。

**注意:**可以组合多个 XML 解析模型。例如，可以使用 SAX 或 StAX 在文档中快速找到感兴趣的数据，然后在内存中构建该特定分支的 DOM 表示。

使用 StAX 可以让您更好地控制解析过程，并允许更方便的状态管理。流中的事件只有在被请求时才被使用，启用了[惰性评估](https://en.wikipedia.org/wiki/Lazy_evaluation)。除此之外，它的性能应该与 SAX 相当，这取决于解析器的实现。

## 了解 Python 标准库中的 XML 解析器

在这一节中，您将了解 Python 的内置 XML 解析器，几乎每个 Python 发行版中都提供了这些解析器。您将把这些解析器与一个示例[可伸缩矢量图形(SVG)](https://en.wikipedia.org/wiki/Scalable_Vector_Graphics) 图像进行比较，这是一种基于 XML 的格式。通过用不同的解析器处理同一个文档，您将能够选择最适合您的解析器。

您将要保存在本地文件中以供参考的示例图像描绘了一个笑脸。它由以下 XML 内容组成:

```py
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [
 <!ENTITY custom_entity "Hello">
]>
<svg xmlns="http://www.w3.org/2000/svg"
  xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  viewBox="-105 -100 210 270" width="210" height="270">
  <inkscape:custom x="42" inkscape:z="555">Some value</inkscape:custom>
  <defs>
    <linearGradient id="skin" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="yellow" stop-opacity="1.0"/>
      <stop offset="75%" stop-color="gold" stop-opacity="1.0"/>
      <stop offset="100%" stop-color="orange" stop-opacity="1"/>
    </linearGradient>
  </defs>
  <g id="smiley" inkscape:groupmode="layer" inkscape:label="Smiley">
    <!-- Head -->
    <circle cx="0" cy="0" r="50"
      fill="url(#skin)" stroke="orange" stroke-width="2"/>
    <!-- Eyes -->
    <ellipse cx="-20" cy="-10" rx="6" ry="8" fill="black" stroke="none"/>
    <ellipse cx="20" cy="-10" rx="6" ry="8" fill="black" stroke="none"/>
    <!-- Mouth -->
    <path d="M-20 20 A25 25 0 0 0 20 20"
      fill="white" stroke="black" stroke-width="3"/>
  </g>
  <text x="-40" y="75">&custom_entity; &lt;svg&gt;!</text>
  <script>
    <![CDATA[
 console.log("CDATA disables XML parsing: <svg>")
 const smiley = document.getElementById("smiley")
 const eyes = document.querySelectorAll("ellipse")
 const setRadius = r => e => eyes.forEach(x => x.setAttribute("ry", r))
 smiley.addEventListener("mouseenter", setRadius(2))
 smiley.addEventListener("mouseleave", setRadius(8))
 ]]>
  </script>
</svg>
```

它以一个 **XML 声明**开始，接着是一个[文档类型定义(DTD)](https://en.wikipedia.org/wiki/Document_type_definition) 和`<svg>` **根元素**。DTD 是可选的，但是如果您决定使用 XML 验证器，它可以帮助验证您的文档结构。根元素为编辑器特定的元素和属性指定了**默认名称空间** `xmlns`以及**前缀名称空间** `xmlns:inkscape`。该文件还包含:

*   嵌套元素
*   属性
*   评论
*   字符数据(`CDATA`)
*   预定义和自定义实体

继续，将 XML 保存在名为 *smiley.svg* 的文件中，并使用现代 web 浏览器打开它，浏览器将运行最后出现的 JavaScript 片段:

[![Smiley Face (SVG)](img/5fc63dff7e1e83d118fc6bc072fc186b.png)](https://files.realpython.com/media/smiley_face.8067146671ea.gif)

该代码向图像添加了一个交互式组件。当你将鼠标悬停在笑脸上时，它会眨眼睛。如果你想使用方便的图形用户界面(GUI)来编辑笑脸，那么你可以使用矢量图形编辑器来打开文件，比如 Adobe Illustrator 或 T2 Inkscape。

**注意:**与 JSON 或 YAML 不同，XML 的一些特性可能会被黑客利用。Python 中的`xml`包中可用的标准 XML 解析器是不安全的，容易受到一系列[攻击](https://docs.python.org/3/library/xml.html#xml-vulnerabilities)。为了安全地解析来自不可信来源的 XML 文档，最好使用安全的替代方法。更多细节可以跳转到本教程的[最后一节](#defuse-the-xml-bomb-with-secure-parsers)。

值得注意的是，Python 的标准库定义了抽象接口**来解析 XML 文档，同时让您提供具体的解析器实现。实际上，您很少这样做，因为 Python 为 [Expat](https://en.wikipedia.org/wiki/Expat_(library)) 库捆绑了一个绑定，Expat 库是一个广泛使用的用 c 编写的开源 XML 解析器。标准库中的所有以下 Python 模块默认使用 Expat。**

不幸的是，虽然 Expat 解析器可以告诉您文档是否**格式良好**，但它不能**根据[XML 模式定义(XSD)](https://en.wikipedia.org/wiki/XML_Schema_(W3C))或[文档类型定义(DTD)](https://en.wikipedia.org/wiki/Document_type_definition)来验证**文档的结构。为此，您必须使用稍后讨论的第三方库之一。

[*Remove ads*](/account/join/)

### `xml.dom.minidom`:最小 DOM 实现

考虑到使用 DOM 解析 XML 文档可以说是最简单的，所以在 Python 标准库中找到 DOM 解析器也就不足为奇了。然而，令人惊讶的是，实际上有两个 DOM 解析器。

`xml.dom`包包含了两个模块来处理 Python 中的 DOM:

1.  `xml.dom.minidom`
2.  `xml.dom.pulldom`

第一个是 DOM 接口的精简实现，它符合 W3C 规范的一个相对较旧的版本。它提供了由 DOM API 定义的常见对象，如`Document`、`Element`和`Attr`。正如您将会发现的那样，这个模块没有得到很好的记录，并且用处非常有限。

第二个模块有一个稍微容易让人误解的名字，因为它定义了一个**流拉解析器**，它可以*或者*生成文档树中当前节点的 DOM 表示。稍后您将找到关于`pulldom`解析器[的更多信息。](#xmldompulldom-streaming-pull-parser)

`minidom`中有两个函数可以让您解析来自各种数据源的 XML 数据。一个接受文件名或文件对象的[，而另一个期望一个](https://docs.python.org/3/glossary.html#term-file-object) [Python 字符串](https://realpython.com/python-strings/):

>>>

```py
>>> from xml.dom.minidom import parse, parseString

>>> # Parse XML from a filename
>>> document = parse("smiley.svg")

>>> # Parse XML from a file object
>>> with open("smiley.svg") as file:
...     document = parse(file)
...

>>> # Parse XML from a Python string
>>> document = parseString("""\
... <svg viewBox="-105 -100 210 270">
...   <!-- More content goes here... -->
... </svg>
... """)
```

[三重引号字符串](https://docs.python.org/3/glossary.html#term-triple-quoted-string)有助于嵌入多行字符串文字，而无需在每行末尾使用延续字符(`\`)。在任何情况下，您都会得到一个`Document`实例，它展示了熟悉的 DOM 接口，允许您遍历树。

除此之外，您将能够访问 XML 声明、DTD 和根元素:

>>>

```py
>>> document = parse("smiley.svg")

>>> # XML Declaration
>>> document.version, document.encoding, document.standalone
('1.0', 'UTF-8', False)

>>> # Document Type Definition (DTD)
>>> dtd = document.doctype
>>> dtd.entities["custom_entity"].childNodes
[<DOM Text node "'Hello'">]

>>> # Document Root
>>> document.documentElement
<DOM Element: svg at 0x7fc78c62d790>
```

如您所见，尽管 Python 中的默认 XML 解析器不能验证文档，但它仍然允许您检查 DTD`.doctype`(如果它存在的话)。注意，XML 声明和 DTD 是可选的。如果 XML 声明或给定的 XML 属性缺失，那么对应的 Python 属性将是 [`None`](https://realpython.com/null-in-python/) 。

要通过 ID 查找元素，必须使用`Document`实例，而不是特定的父元素`Element`。示例 SVG 图像有两个带有`id`属性的节点，但是您找不到它们中的任何一个:

>>>

```py
>>> document.getElementById("skin") is None
True
>>> document.getElementById("smiley") is None
True
```

对于只使用过 HTML 和 JavaScript，但以前没有使用过 XML 的人来说，这可能会令人惊讶。虽然 HTML 为某些元素和属性定义了语义，比如`<body>`或`id`，但是 XML 并没有为其构建块赋予任何意义。您需要使用 DTD 或者通过调用 Python 中的`.setIdAttribute()`来显式地将属性标记为 ID，例如:

| 定义样式 | 履行 |
| --- | --- |
| 文档类型定义（Document Type Definition 的缩写） | `<!ATTLIST linearGradient id ID #IMPLIED>` |
| 计算机编程语言 | `linearGradient.setIdAttribute("id")` |

但是，如果您的文档有默认的名称空间，使用 DTD 不足以解决问题，示例 SVG 图像就是这种情况。为了解决这个问题，您可以递归地访问 Python 中的所有元素，检查它们是否具有`id`属性，并一次性地将其指定为它们的 ID:

>>>

```py
>>> from xml.dom.minidom import parse, Node

>>> def set_id_attribute(parent, attribute_name="id"):
...     if parent.nodeType == Node.ELEMENT_NODE:
...         if parent.hasAttribute(attribute_name):
...             parent.setIdAttribute(attribute_name)
...     for child in parent.childNodes:
...         set_id_attribute(child, attribute_name)
...
>>> document = parse("smiley.svg")
>>> set_id_attribute(document)
```

您的自定义函数`set_id_attribute()`接受一个父元素和 identity 属性的可选名称，默认为`"id"`。当在 SVG 文档中调用该函数时，所有具有`id`属性的子元素都可以通过 DOM API 访问:

>>>

```py
>>> document.getElementById("skin")
<DOM Element: linearGradient at 0x7f82247703a0>

>>> document.getElementById("smiley")
<DOM Element: g at 0x7f8224770940>
```

现在，您将获得对应于`id`属性值的预期 XML 元素。

使用 ID 最多可以找到一个唯一的元素，但是您也可以通过它们的**标记名**找到一组相似的元素。与`.getElementById()`方法不同，您可以在文档或特定父元素上调用`.getElementsByTagName()`来缩小搜索范围:

>>>

```py
>>> document.getElementsByTagName("ellipse")
[
 <DOM Element: ellipse at 0x7fa2c944f430>,
 <DOM Element: ellipse at 0x7fa2c944f4c0>
]

>>> root = document.documentElement
>>> root.getElementsByTagName("ellipse")
[
 <DOM Element: ellipse at 0x7fa2c944f430>,
 <DOM Element: ellipse at 0x7fa2c944f4c0>
]
```

注意，`.getElementsByTagName()`总是返回元素的[列表](https://realpython.com/python-lists-tuples/)，而不是单个元素或`None`。当您在两种方法之间切换时忘记它是一个常见的错误来源。

不幸的是，像`<inkscape:custom>`这样以名称空间标识符作为**前缀的元素将不会被包含在内。必须使用`.getElementsByTagNameNS()`来搜索它们，它需要不同的参数:**

>>>

```py
>>> document.getElementsByTagNameNS(
...     "http://www.inkscape.org/namespaces/inkscape",
...     "custom"
... )
...
[<DOM Element: inkscape:custom at 0x7f97e3f2a3a0>]

>>> document.getElementsByTagNameNS("*", "custom")
[<DOM Element: inkscape:custom at 0x7f97e3f2a3a0>]
```

第一个参数必须是 XML 名称空间，通常采用域名的形式，而第二个参数是标记名。请注意，名称空间前缀是不相关的！要搜索所有名称空间，可以提供一个通配符(`*`)。

**注意:**要找到 XML 文档中声明的名称空间，可以检查根元素的属性。理论上，它们可以在任何元素上声明，但是顶级元素通常是你可以找到它们的地方。

一旦你找到你感兴趣的元素，你就可以用它来遍历树。然而，`minidom`的另一个不和谐之处是它如何处理元素之间的空白字符:

>>>

```py
>>> element = document.getElementById("smiley")

>>> element.parentNode
<DOM Element: svg at 0x7fc78c62d790>

>>> element.firstChild
<DOM Text node "'\n    '">

>>> element.lastChild
<DOM Text node "'\n  '">

>>> element.nextSibling
<DOM Text node "'\n  '">

>>> element.previousSibling
<DOM Text node "'\n  '">
```

换行符和前导缩进被捕获为单独的树元素，这是规范所要求的。一些解析器允许您忽略这些，但 Python 不允许。但是，您可以手动折叠此类节点中的空白:

>>>

```py
>>> def remove_whitespace(node):
...     if node.nodeType == Node.TEXT_NODE:
...         if node.nodeValue.strip() == "":
...             node.nodeValue = ""
...     for child in node.childNodes:
...         remove_whitespace(child)
...
>>> document = parse("smiley.svg")
>>> set_id_attribute(document)
>>> remove_whitespace(document)
>>> document.normalize()
```

注意，你还必须将 [`.normalize()`](https://docs.python.org/3/library/xml.dom.html#xml.dom.Node.normalize) 文档中相邻的文本节点组合起来。否则，您可能会得到一堆只有空格的冗余 XML 元素。同样，递归是访问树元素的唯一方式，因为不能用循环遍历文档及其元素。最后，这应该会给您带来预期的结果:

>>>

```py
>>> element = document.getElementById("smiley")

>>> element.parentNode
<DOM Element: svg at 0x7fc78c62d790>

>>> element.firstChild
<DOM Comment node "' Head '">

>>> element.lastChild
<DOM Element: path at 0x7f8beea0f670>

>>> element.nextSibling
<DOM Element: text at 0x7f8beea0f700>

>>> element.previousSibling
<DOM Element: defs at 0x7f8beea0f160>

>>> element.childNodes
[
 <DOM Comment node "' Head '">,
 <DOM Element: circle at 0x7f8beea0f4c0>,
 <DOM Comment node "' Eyes '">,
 <DOM Element: ellipse at 0x7fa2c944f430>,
 <DOM Element: ellipse at 0x7fa2c944f4c0>,
 <DOM Comment node "' Mouth '">,
 <DOM Element: path at 0x7f8beea0f670>
]
```

元素公开了一些有用的方法和属性，让您可以查询它们的详细信息:

>>>

```py
>>> element = document.getElementsByTagNameNS("*", "custom")[0]

>>> element.prefix
'inkscape'

>>> element.tagName
'inkscape:custom'

>>> element.attributes
<xml.dom.minidom.NamedNodeMap object at 0x7f6c9d83ba80>

>>> dict(element.attributes.items())
{'x': '42', 'inkscape:z': '555'}

>>> element.hasChildNodes()
True

>>> element.hasAttributes()
True

>>> element.hasAttribute("x")
True

>>> element.getAttribute("x")
'42'

>>> element.getAttributeNode("x")
<xml.dom.minidom.Attr object at 0x7f82244a05f0>

>>> element.getAttribute("missing-attribute")
''
```

例如，您可以检查元素的名称空间、标记名或属性。如果您要求一个缺失的属性，那么您将得到一个空字符串(`''`)。

处理命名空间属性没什么不同。您只需要记住在属性名前面加上相应的前缀或提供域名:

>>>

```py
>>> element.hasAttribute("z")
False

>>> element.hasAttribute("inkscape:z")
True

>>> element.hasAttributeNS(
...     "http://www.inkscape.org/namespaces/inkscape",
...     "z"
... )
...
True

>>> element.hasAttributeNS("*", "z")
False
```

奇怪的是，通配符(`*`)在这里并不像以前在`.getElementsByTagNameNS()`方法中那样起作用。

因为本教程只是关于 XML 解析，所以您需要查看`minidom`文档中修改 DOM 树的方法。它们大多遵循 W3C 规范。

如您所见，`minidom`模块并不十分方便。它的主要优势来自于作为标准库的一部分，这意味着您不必在项目中安装任何外部依赖项来使用 DOM。

[*Remove ads*](/account/join/)

### `xml.sax`:Python 的 SAX 接口

要开始使用 Python 中的 SAX，您可以像以前一样使用相同的`parse()`和`parseString()`便利函数，但是要从`xml.sax`包中获取。您还必须提供至少一个必需的参数，它必须是一个**内容处理程序**实例。本着 Java 的精神，您可以通过对特定的基类进行子类化来提供一个:

```py
from xml.sax import parse
from xml.sax.handler import ContentHandler

class SVGHandler(ContentHandler):
    pass

parse("smiley.svg", SVGHandler())
```

在解析文档时，内容处理程序接收与文档中的元素相对应的事件流。运行这段代码不会做任何有用的事情，因为您的处理程序类是空的。为了让它工作，你需要从超类中重载一个或多个[回调方法](https://docs.python.org/3/library/xml.sax.handler.html#contenthandler-objects)。

启动您最喜欢的编辑器，键入以下代码，并将其保存在名为`svg_handler.py`的文件中:

```py
# svg_handler.py

from xml.sax.handler import ContentHandler

class SVGHandler(ContentHandler):

    def startElement(self, name, attrs):
        print(f"BEGIN: <{name}>, {attrs.keys()}")

    def endElement(self, name):
        print(f"END: </{name}>")

    def characters(self, content):
        if content.strip() != "":
            print("CONTENT:", repr(content))
```

这个修改后的内容处理程序[在标准输出中打印出一些事件。SAX 解析器将为您调用这三个方法，以响应找到开始标记、结束标记以及它们之间的一些文本。当您打开 Python 解释器的交互式会话时，导入您的内容处理程序并进行测试。它应该产生以下输出:](https://realpython.com/python-print/)

>>>

```py
>>> from xml.sax import parse
>>> from svg_handler import SVGHandler
>>> parse("smiley.svg", SVGHandler())
BEGIN: <svg>, ['xmlns', 'xmlns:inkscape', 'viewBox', 'width', 'height']
BEGIN: <inkscape:custom>, ['x', 'inkscape:z']
CONTENT: 'Some value'
END: </inkscape:custom>
BEGIN: <defs>, []
BEGIN: <linearGradient>, ['id', 'x1', 'x2', 'y1', 'y2']
BEGIN: <stop>, ['offset', 'stop-color', 'stop-opacity']
END: </stop>
⋮
```

这本质上是[观察者设计模式](https://sourcemaking.com/design_patterns/observer)，它允许您将 XML 逐步转换成另一种分层格式。假设您想将 SVG 文件转换成简化的 [JSON](https://realpython.com/python-json/) 表示。首先，您希望将内容处理程序对象存储在一个单独的变量中，以便以后从中提取信息:

>>>

```py
>>> from xml.sax import parse
>>> from svg_handler import SVGHandler
>>> handler = SVGHandler() >>> parse("smiley.svg", handler)
```

因为 SAX 解析器发出事件时没有提供任何关于它所找到的元素的上下文，所以您需要跟踪您在树中的位置。因此，将当前元素推入并弹出到一个[堆栈](https://realpython.com/how-to-implement-python-stack/)是有意义的，您可以通过一个常规的 [Python 列表](https://realpython.com/python-lists-tuples/)来模拟这个堆栈。您还可以定义一个助手属性`.current_element`，它将返回放置在堆栈顶部的最后一个元素:

```py
# svg_handler.py

# ...

class SVGHandler(ContentHandler):

    def __init__(self):
        super().__init__()
        self.element_stack = []

    @property
    def current_element(self):
        return self.element_stack[-1]

    # ...
```

当 SAX 解析器找到一个新元素时，您可以立即捕获它的标记名和属性，同时为子元素和值创建占位符，这两者都是可选的。现在，您可以将每个元素存储为一个`dict`对象。用新的实现替换您现有的`.startElement()`方法:

```py
# svg_handler.py

# ...

class SVGHandler(ContentHandler):

    # ...

    def startElement(self, name, attrs):
        self.element_stack.append({
            "name": name,
            "attributes": dict(attrs),
            "children": [],
            "value": ""
        })
```

SAX 解析器将属性作为[映射](https://docs.python.org/3/glossary.html#term-mapping)提供给你，你可以通过调用`dict()`函数将其转换成普通的 [Python 字典](https://realpython.com/python-dicts/)。元素值通常分布在多个片段上，您可以使用加号运算符(`+`)或相应的增强赋值语句将这些片段连接起来:

```py
# svg_handler.py

# ...

class SVGHandler(ContentHandler):

    # ...

    def characters(self, content):
        self.current_element["value"] += content
```

以这种方式聚合文本将确保多行内容出现在当前元素中。例如，样本 SVG 文件中的`<script>`标记包含六行 JavaScript 代码，它们分别触发对`characters()`回调的调用。

最后，一旦解析器发现了结束标记，就可以从堆栈中弹出当前元素，并将其附加到父元素的子元素中。如果只剩下一个元素，那么它将是你的文档的根，你以后应该保留它。除此之外，您可能希望通过删除具有空值的键来清除当前元素:

```py
# svg_handler.py

# ...

class SVGHandler(ContentHandler):

    # ...

    def endElement(self, name):
        clean(self.current_element)
        if len(self.element_stack) > 1:
            child = self.element_stack.pop()
            self.current_element["children"].append(child)

def clean(element):
    element["value"] = element["value"].strip()
    for key in ("attributes", "children", "value"):
        if not element[key]:
            del element[key]
```

注意`clean()`是在类体外部定义的函数。清理必须在最后完成，因为没有办法预先知道可能有多少文本片段要连接。您可以展开下面的可折叠部分，查看完整的内容处理程序代码。



```py
# svg_handler.py

from xml.sax.handler import ContentHandler

class SVGHandler(ContentHandler):

    def __init__(self):
        super().__init__()
        self.element_stack = []

    @property
    def current_element(self):
        return self.element_stack[-1]

    def startElement(self, name, attrs):
        self.element_stack.append({
            "name": name,
            "attributes": dict(attrs),
            "children": [],
            "value": ""
        })

    def endElement(self, name):
        clean(self.current_element)
        if len(self.element_stack) > 1:
            child = self.element_stack.pop()
            self.current_element["children"].append(child)

    def characters(self, content):
        self.current_element["value"] += content

def clean(element):
    element["value"] = element["value"].strip()
    for key in ("attributes", "children", "value"):
        if not element[key]:
            del element[key]
```

现在，是时候通过解析 XML、从内容处理程序中提取根元素并将其转储到 JSON 字符串来测试一切了:

>>>

```py
>>> from xml.sax import parse
>>> from svg_handler import SVGHandler
>>> handler = SVGHandler()
>>> parse("smiley.svg", handler)
>>> root = handler.current_element

>>> import json
>>> print(json.dumps(root, indent=4))
{
 "name": "svg",
 "attributes": {
 "xmlns": "http://www.w3.org/2000/svg",
 "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
 "viewBox": "-105 -100 210 270",
 "width": "210",
 "height": "270"
 },
 "children": [
 {
 "name": "inkscape:custom",
 "attributes": {
 "x": "42",
 "inkscape:z": "555"
 },
 "value": "Some value"
 },
⋮
```

值得注意的是，这个实现并没有比 DOM 增加多少内存，因为它像以前一样构建了整个文档的抽象表示。不同之处在于，您制作了一个定制的字典表示，而不是标准的 DOM 树。但是，您可以想象在接收 SAX 事件时直接写入文件或数据库，而不是内存。这将有效地解除你的计算机内存限制。

如果您想解析 XML 名称空间，那么您需要用一些样板代码自己创建和配置 SAX 解析器，并实现稍微不同的回调:

```py
# svg_handler.py

from xml.sax.handler import ContentHandler

class SVGHandler(ContentHandler):

    def startPrefixMapping(self, prefix, uri):
        print(f"startPrefixMapping: {prefix=}, {uri=}")

    def endPrefixMapping(self, prefix):
        print(f"endPrefixMapping: {prefix=}")

    def startElementNS(self, name, qname, attrs):
        print(f"startElementNS: {name=}")

    def endElementNS(self, name, qname):
        print(f"endElementNS: {name=}")
```

这些回调接收关于元素名称空间的附加参数。要让 SAX 解析器真正触发这些回调而不是一些早期的回调，必须显式启用 **XML 名称空间**支持:

>>>

```py
>>> from xml.sax import make_parser
>>> from xml.sax.handler import feature_namespaces >>> from svg_handler import SVGHandler

>>> parser = make_parser()
>>> parser.setFeature(feature_namespaces, True) >>> parser.setContentHandler(SVGHandler())

>>> parser.parse("smiley.svg")
startPrefixMapping: prefix=None, uri='http://www.w3.org/2000/svg'
startPrefixMapping: prefix='inkscape', uri='http://www.inkscape.org/namespaces/inkscape'
startElementNS: name=('http://www.w3.org/2000/svg', 'svg')
⋮
endElementNS: name=('http://www.w3.org/2000/svg', 'svg')
endPrefixMapping: prefix='inkscape'
endPrefixMapping: prefix=None
```

设置这个特性会将元素`name`变成一个由名称空间的域名和标记名组成的元组。

`xml.sax`包提供了一个体面的基于事件的 XML 解析器接口，它模仿了原始的 Java API。与 DOM 相比，它有些局限，但应该足以实现一个基本的 XML 流推送解析器，而不需要借助第三方库。考虑到这一点，Python 中提供了一个不太冗长的 pull 解析器，您将在接下来探索它。

[*Remove ads*](/account/join/)

### `xml.dom.pulldom`:流拉解析器

Python 标准库中的解析器经常一起工作。例如，`xml.dom.pulldom`模块包装了来自`xml.sax`的解析器，以利用缓冲并分块读取文档。同时，它使用来自`xml.dom.minidom`的默认 DOM 实现来表示文档元素。但是，这些元素一次处理一个，没有任何关系，直到您明确要求它。

**注意:**在`xml.dom.pulldom`中默认启用 XML 名称空间支持。

虽然 SAX 模型遵循[观察者模式](https://en.wikipedia.org/wiki/Observer_pattern)，但是您可以将 StAX 视为[迭代器设计模式](https://sourcemaking.com/design_patterns/iterator)，它允许您在事件的**平面流**上循环。同样，您可以调用从模块导入的熟悉的`parse()`或`parseString()`函数来解析 SVG 图像:

>>>

```py
>>> from xml.dom.pulldom import parse
>>> event_stream = parse("smiley.svg")
>>> for event, node in event_stream:
...     print(event, node)
...
START_DOCUMENT <xml.dom.minidom.Document object at 0x7f74f9283e80>
START_ELEMENT <DOM Element: svg at 0x7f74fde18040>
CHARACTERS <DOM Text node "'\n'">
⋮
END_ELEMENT <DOM Element: script at 0x7f74f92b3c10>
CHARACTERS <DOM Text node "'\n'">
END_ELEMENT <DOM Element: svg at 0x7f74fde18040>
```

解析文档只需要几行代码。`xml.sax`和`xml.dom.pulldom`最显著的区别是缺少回调，因为你驱动整个过程。在构建代码时，你有更多的自由，如果你不想使用[类](https://realpython.com/python3-object-oriented-programming/)，你就不需要使用它们。

注意，从流中提取的 XML 节点具有在`xml.dom.minidom`中定义的类型。但是如果你检查他们的父母、兄弟姐妹和孩子，你会发现他们对彼此一无所知:

>>>

```py
>>> from xml.dom.pulldom import parse, START_ELEMENT
>>> event_stream = parse("smiley.svg")
>>> for event, node in event_stream:
...     if event == START_ELEMENT:
...         print(node.parentNode, node.previousSibling, node.childNodes)
<xml.dom.minidom.Document object at 0x7f90864f6e80> None []
None None []
None None []
None None []
⋮
```

相关属性为空。无论如何，拉解析器可以帮助以混合方式快速查找某个父元素，并只为以它为根的分支构建一个 DOM 树:

```py
from xml.dom.pulldom import parse, START_ELEMENT

def process_group(parent):
    left_eye, right_eye = parent.getElementsByTagName("ellipse")
    # ...

event_stream = parse("smiley.svg")
for event, node in event_stream:
    if event == START_ELEMENT:
        if node.tagName == "g":
 event_stream.expandNode(node)            process_group(node)
```

通过在事件流上调用`.expandNode()`,您实际上是向前移动迭代器并递归解析 XML 节点，直到找到父元素的匹配结束标记。结果节点将有正确初始化属性的子节点。此外，您将能够对它们使用 DOM 方法。

pull 解析器结合了两者的优点，为 DOM 和 SAX 提供了一个有趣的替代品。它使用起来高效、灵活、简单，导致代码更紧凑、可读性更好。您还可以使用它更容易地同时处理多个 XML 文件。也就是说，到目前为止提到的 XML 解析器没有一个可以与 Python 标准库中最后一个解析器的优雅、简单和完整性相媲美。

### `xml.etree.ElementTree`:一个轻量级的 Pythonic 替代品

到目前为止，您已经了解的 XML 解析器完成了这项工作。然而，它们不太符合 Python 的哲学，这不是偶然的。虽然 DOM 遵循 W3C 规范，而 SAX 是在 Java API 的基础上建模的，但这两者都不太像 Pythonic。

更糟糕的是，DOM 和 SAX 解析器都感觉过时了，因为它们在 [CPython](https://realpython.com/cpython-source-code-guide/) 解释器中的一些代码已经二十多年没有改变了！在我写这篇文章的时候，它们的实现还没有完成，并且有[丢失的打字存根](https://github.com/python/typeshed/issues/3787)，这破坏了[代码编辑器](https://realpython.com/python-ides-code-editors-guide/)中的代码完成。

同时，Python 2.5 带来了解析*和*编写 XML 文档的新视角——元素树 API**。它是一个轻量级的、高效的、优雅的、功能丰富的接口，甚至一些第三方库都是基于它构建的。要入门，必须导入`xml.etree.ElementTree`模块，有点拗口。因此，习惯上是这样定义**别名**的:**

```py
import xml.etree.ElementTree as ET
```

在稍微旧一点的代码中，您可能会看到导入了`cElementTree`模块。这是一个比用 c 编写的相同接口快几倍的实现。今天，只要有可能，常规模块就使用快速实现，所以您不再需要费心了。

您可以通过采用不同的解析策略来使用 ElementTree API:

|  | 非增量 | 增量(阻塞) | 增量(非阻塞) |
| --- | --- | --- | --- |
| `ET.parse()` | ✔️ |  |  |
| `ET.fromstring()` | ✔️ |  |  |
| `ET.iterparse()` |  | ✔️ |  |
| `ET.XMLPullParser` |  |  | ✔️ |

非增量策略以类似 DOM 的方式将整个文档加载到内存中。模块中有两个适当命名的函数，用于解析包含 XML 内容的文件或 Python 字符串:

>>>

```py
>>> import xml.etree.ElementTree as ET

>>> # Parse XML from a filename
>>> ET.parse("smiley.svg")
<xml.etree.ElementTree.ElementTree object at 0x7fa4c980a6a0>

>>> # Parse XML from a file object
>>> with open("smiley.svg") as file:
...     ET.parse(file)
...
<xml.etree.ElementTree.ElementTree object at 0x7fa4c96df340>

>>> # Parse XML from a Python string
>>> ET.fromstring("""\
... <svg viewBox="-105 -100 210 270">
...   <!-- More content goes here... -->
... </svg>
... """)
<Element 'svg' at 0x7fa4c987a1d0>
```

用`parse()`解析文件对象或文件名会返回一个 [`ET.ElementTree`](https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.ElementTree) 类的实例，它代表整个元素层次结构。另一方面，用`fromstring()`解析字符串将返回特定的根 [`ET.Element`](https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.Element) 。

或者，您可以使用流**拉解析器**递增地读取 XML 文档，这将产生一系列事件和元素:

>>>

```py
>>> for event, element in ET.iterparse("smiley.svg"):
...     print(event, element.tag)
...
end {http://www.inkscape.org/namespaces/inkscape}custom
end {http://www.w3.org/2000/svg}stop
end {http://www.w3.org/2000/svg}stop
end {http://www.w3.org/2000/svg}stop
end {http://www.w3.org/2000/svg}linearGradient
⋮
```

默认情况下，`iterparse()`只发出与结束 XML 标记相关联的`end`事件。但是，您也可以订阅其他活动。你可以用字符串常量找到它们，比如`"comment"`:

>>>

```py
>>> import xml.etree.ElementTree as ET
>>> for event, element in ET.iterparse("smiley.svg", ["comment"]):
...     print(element.text.strip())
...
Head
Eyes
Mouth
```

以下是所有可用事件类型的列表:

*   **`start` :** 元素的开始
*   **`end` :** 一个元素结束
*   **`comment` :** 评论元素
*   **`pi` :** 加工指令，如 [XSL](https://en.wikipedia.org/wiki/XSL)
*   **`start-ns` :** 命名空间的开始
*   **`end-ns` :** 一个名称空间的结束

`iterparse()`的缺点是它使用**阻塞调用**来读取下一个数据块，这可能不适合在单个执行线程上运行的[异步代码](https://realpython.com/python-async-features/)。为了缓解这种情况，您可以查看一下 [`XMLPullParser`](https://docs.python.org/3/library/xml.etree.elementtree.html#pull-api-for-non-blocking-parsing) ，这稍微有点冗长:

```py
import xml.etree.ElementTree as ET

async def receive_data(url):
    """Download chunks of bytes from the URL asynchronously."""
    yield b"<svg "
    yield b"viewBox=\"-105 -100 210 270\""
    yield b"></svg>"

async def parse(url, events=None):
    parser = ET.XMLPullParser(events)
    async for chunk in receive_data(url):
        parser.feed(chunk)
        for event, element in parser.read_events():
            yield event, element
```

这个假设的例子向解析器提供几秒钟后到达的 XML 块。一旦有了足够的内容，就可以迭代解析器缓冲的一系列事件和元素。这种**非阻塞的**增量解析策略允许在下载多个 XML 文档的同时对它们进行真正的并行解析。

树中的元素是可变的、可迭代的和可索引的[序列](https://docs.python.org/3/glossary.html#term-sequence)。它们的长度对应于其直接子代的数量:

>>>

```py
>>> import xml.etree.ElementTree as ET
>>> tree = ET.parse("smiley.svg")
>>> root = tree.getroot()

>>> # The length of an element equals the number of its children.
>>> len(root)
5

>>> # The square brackets let you access a child by an index.
>>> root[1]
<Element '{http://www.w3.org/2000/svg}defs' at 0x7fe05d2e8860>
>>> root[2]
<Element '{http://www.w3.org/2000/svg}g' at 0x7fa4c9848400>

>>> # Elements are mutable. For example, you can swap their children.
>>> root[2], root[1] = root[1], root[2] 
>>> # You can iterate over an element's children.
>>> for child in root:
...     print(child.tag)
...
{http://www.inkscape.org/namespaces/inkscape}custom
{http://www.w3.org/2000/svg}g
{http://www.w3.org/2000/svg}defs
{http://www.w3.org/2000/svg}text
{http://www.w3.org/2000/svg}script
```

标记名可能以一对花括号(`{}`)中的可选名称空间为前缀。定义时，默认的 XML 名称空间也会出现在那里。注意突出显示的行中的交换赋值是如何使`<g>`元素出现在`<defs>`之前的。这显示了序列的可变性质。

这里还有一些值得一提的元素属性和方法:

>>>

```py
>>> element = root[0]

>>> element.tag
'{http://www.inkscape.org/namespaces/inkscape}custom'

>>> element.text
'Some value'

>>> element.attrib
{'x': '42', '{http://www.inkscape.org/namespaces/inkscape}z': '555'}

>>> element.get("x")
'42'
```

这个 API 的好处之一是它如何使用 Python 的原生数据类型。上面，它为元素的属性使用了 Python 字典。在前面的模块中，它们被包装在不太方便的适配器中。与 DOM 不同的是，ElementTree API 不公开任何方向遍历树的方法或属性，但是有一些更好的替代方法。

正如您之前看到的，`Element`类的实例实现了**序列协议**，允许您通过一个循环迭代它们的直接子类:

>>>

```py
>>> for child in root:
...     print(child.tag)
...
{http://www.inkscape.org/namespaces/inkscape}custom
{http://www.w3.org/2000/svg}defs
{http://www.w3.org/2000/svg}g
{http://www.w3.org/2000/svg}text
{http://www.w3.org/2000/svg}script
```

您将获得根的直接子元素的序列。然而，要深入嵌套的后代，您必须调用祖先元素上的`.iter()`方法:

>>>

```py
>>> for descendant in root.iter():
...     print(descendant.tag)
...
{http://www.w3.org/2000/svg}svg
{http://www.inkscape.org/namespaces/inkscape}custom
{http://www.w3.org/2000/svg}defs
{http://www.w3.org/2000/svg}linearGradient
{http://www.w3.org/2000/svg}stop
{http://www.w3.org/2000/svg}stop
{http://www.w3.org/2000/svg}stop
{http://www.w3.org/2000/svg}g
{http://www.w3.org/2000/svg}circle
{http://www.w3.org/2000/svg}ellipse
{http://www.w3.org/2000/svg}ellipse
{http://www.w3.org/2000/svg}path
{http://www.w3.org/2000/svg}text
{http://www.w3.org/2000/svg}script
```

根元素只有五个子元素，但总共有十三个后代。还可以通过使用可选的`tag`参数**仅过滤**特定的标签名称来缩小后代的范围:

>>>

```py
>>> tag_name = "{http://www.w3.org/2000/svg}ellipse"
>>> for descendant in root.iter(tag_name):
...     print(descendant)
...
<Element '{http://www.w3.org/2000/svg}ellipse' at 0x7f430baa03b0>
<Element '{http://www.w3.org/2000/svg}ellipse' at 0x7f430baa0450>
```

这一次，你只得到两个`<ellipse>`元素。记得在标签名中包含 **XML 名称空间**，比如`{http://www.w3.org/2000/svg}`——只要它已经被定义了。否则，如果您只提供标记名而没有正确的名称空间，那么您可能会得到比最初预期的更少或更多的后代元素。

使用`.iterfind()`处理名称空间更方便，它接受前缀到域名的可选映射。要指示**默认名称空间**，您可以将键留空或分配一个任意前缀，这个前缀必须在后面的标记名中使用:

>>>

```py
>>> namespaces = {
...     "": "http://www.w3.org/2000/svg",
...     "custom": "http://www.w3.org/2000/svg"
... }

>>> for descendant in root.iterfind("g", namespaces):
...     print(descendant)
...
<Element '{http://www.w3.org/2000/svg}g' at 0x7f430baa0270>

>>> for descendant in root.iterfind("custom:g", namespaces):
...     print(descendant)
...
<Element '{http://www.w3.org/2000/svg}g' at 0x7f430baa0270>
```

名称空间映射允许您用不同的前缀引用同一个元素。令人惊讶的是，如果您像以前一样尝试查找那些嵌套的`<ellipse>`元素，那么`.iterfind()`不会返回任何内容，因为它需要一个 **XPath 表达式**，而不是一个简单的标记名:

>>>

```py
>>> for descendant in root.iterfind("ellipse", namespaces):
...     print(descendant)
...

>>> for descendant in root.iterfind("g/ellipse", namespaces):
...     print(descendant)
...
<Element '{http://www.w3.org/2000/svg}ellipse' at 0x7f430baa03b0>
<Element '{http://www.w3.org/2000/svg}ellipse' at 0x7f430baa0450>
```

巧合的是，字符串`"g"`恰好是相对于当前`root`元素的有效路径，这也是函数之前返回非空结果的原因。但是，要找到嵌套在 XML 层次结构中更深一层的省略号，您需要一个更详细的路径表达式。

ElementTree 对 [XPath 小型语言](https://www.w3.org/TR/xpath/)有[有限的语法支持](https://docs.python.org/3/library/xml.etree.elementtree.html#supported-xpath-syntax)，可以用来查询 XML 中的元素，类似于 HTML 中的 CSS 选择器。还有其他方法接受这样的表达式:

>>>

```py
>>> namespaces = {"": "http://www.w3.org/2000/svg"}

>>> root.iterfind("defs", namespaces)
<generator object prepare_child.<locals>.select at 0x7f430ba6d190>

>>> root.findall("defs", namespaces)
[<Element '{http://www.w3.org/2000/svg}defs' at 0x7f430ba09e00>]

>>> root.find("defs", namespaces)
<Element '{http://www.w3.org/2000/svg}defs' at 0x7f430ba09e00>
```

当`.iterfind()`产生匹配元素时，`.findall()`返回一个列表，`.find()`只返回第一个匹配元素。类似地，您可以使用`.findtext()`提取元素的开始和结束标记之间的文本，或者使用`.itertext()`获取整个文档的内部文本:

>>>

```py
>>> namespaces = {"i": "http://www.inkscape.org/namespaces/inkscape"}

>>> root.findtext("i:custom", namespaces=namespaces)
'Some value'

>>> for text in root.itertext():
...     if text.strip() != "":
...         print(text.strip())
...
Some value
Hello <svg>!
console.log("CDATA disables XML parsing: <svg>")
⋮
```

首先查找嵌入在特定 XML 元素中的文本，然后查找整个文档中的所有文本。按文本搜索是 ElementTree API 的一个强大功能。可以使用其他内置的解析器来复制它，但是代价是增加了代码的复杂性，降低了便利性。

ElementTree API 可能是其中最直观的一个。它是 Pythonic 式的、高效的、健壮的、通用的。除非您有特定的理由使用 DOM 或 SAX，否则这应该是您的默认选择。

[*Remove ads*](/account/join/)

## 探索第三方 XML 解析器库

有时候，接触标准库中的 XML 解析器可能感觉像是拿起一把大锤敲碎一颗坚果。在其他时候，情况正好相反，您希望解析器能做更多的事情。例如，您可能希望根据模式或使用高级 XPath 表达式来验证 XML。在这些情况下，最好检查一下在 [PyPI](https://pypi.org/) 上可用的外部库。

下面，您将找到一系列复杂程度不同的外部库。

### `untangle`:将 XML 转换成 Python 对象

如果您正在寻找一个可以将 XML 文档转换成 Python 对象的一行程序，那么不用再找了。虽然已经有几年没有更新了，但是 [`untangle`](https://pypi.org/project/untangle/) 库可能很快就会成为您最喜欢的用 Python 解析 XML 的方式。只需要记住一个函数，它接受 URL、文件名、文件对象或 XML 字符串:

>>>

```py
>>> import untangle

>>> # Parse XML from a URL
>>> untangle.parse("http://localhost:8000/smiley.svg")
Element(name = None, attributes = None, cdata = )

>>> # Parse XML from a filename
>>> untangle.parse("smiley.svg")
Element(name = None, attributes = None, cdata = )

>>> # Parse XML from a file object
>>> with open("smiley.svg") as file:
...     untangle.parse(file)
...
Element(name = None, attributes = None, cdata = )

>>> # Parse XML from a Python string
>>> untangle.parse("""\
... <svg viewBox="-105 -100 210 270">
...   <!-- More content goes here... -->
... </svg>
... """)
Element(name = None, attributes = None, cdata = )
```

在每种情况下，它都返回一个`Element`类的实例。您可以使用**点操作符**访问其子节点，使用**方括号**语法通过索引获取 XML 属性或其中一个子节点。例如，要获取文档的根元素，您可以像访问对象的属性一样访问它。要获取元素的一个 XML 属性，可以将其名称作为字典键传递:

>>>

```py
>>> import untangle
>>> document = untangle.parse("smiley.svg")

>>> document.svg
Element(name = svg, attributes = {'xmlns': ...}, ...)

>>> document.svg["viewBox"]
'-105 -100 210 270'
```

不需要记住函数或方法的名字。相反，每个被解析的对象都是唯一的，所以您真的需要知道底层 XML 文档的结构才能用`untangle`遍历它。

要找出根元素的名称，在文档上调用`dir()`:

>>>

```py
>>> dir(document)
['svg']
```

这显示了元素的直接子元素的名称。注意，`untangle`为其解析的文档重新定义了`dir()`的含义。通常，您调用这个内置函数来检查一个类或一个 Python 模块。默认实现将返回属性名列表，而不是 XML 文档的子元素。

如果有多个子元素具有给定的标记名，那么您可以用一个循环迭代它们，或者通过索引引用一个子元素:

>>>

```py
>>> dir(document.svg)
['defs', 'g', 'inkscape_custom', 'script', 'text']

>>> dir(document.svg.defs.linearGradient)
['stop', 'stop', 'stop']

>>> for stop in document.svg.defs.linearGradient.stop:
...     print(stop)
...
Element <stop> with attributes {'offset': ...}, ...
Element <stop> with attributes {'offset': ...}, ...
Element <stop> with attributes {'offset': ...}, ...

>>> document.svg.defs.linearGradient.stop[1]
Element(name = stop, attributes = {'offset': ...}, ...)
```

您可能已经注意到了，`<inkscape:custom>`元素被重命名为`inkscape_custom`。不幸的是，这个库不能很好地处理 **XML 名称空间**，所以如果这是你需要依赖的东西，那么你必须去别处看看。

由于点符号，XML 文档中的元素名必须是有效的 [Python 标识符](https://docs.python.org/3/reference/lexical_analysis.html#identifiers)。如果不是，那么`untangle`将自动重写它们的名字，用下划线替换被禁止的字符:

>>>

```py
>>> dir(untangle.parse("<com:company.web-app></com:company.web-app>"))
['com_company_web_app']
```

子标签名称不是您可以访问的唯一对象属性。元素有一些预定义的对象属性，可以通过调用`vars()`来显示:

>>>

```py
>>> element = document.svg.text

>>> list(vars(element).keys())
['_name', '_attributes', 'children', 'is_root', 'cdata']

>>> element._name
'text'

>>> element._attributes
{'x': '-40', 'y': '75'}

>>> element.children
[]

>>> element.is_root
False

>>> element.cdata
'Hello <svg>!'
```

在幕后，`untangle`使用内置的 SAX 解析器，但是因为这个库是用纯 Python 实现的，并且创建了许多重量级对象，所以它的性能**相当差**。虽然它旨在读取微小的文档，但是您仍然可以将它与另一种方法结合起来读取数千兆字节的 XML 文件。

以下是方法。如果你去[维基百科档案馆](https://dumps.wikimedia.org/enwiki/latest/)，你可以下载他们的一个压缩 XML 文件。顶部的一个应该包含文章摘要的快照:

```py
<feed>
  <doc>
    <title>Wikipedia: Anarchism</title>
    <url>https://en.wikipedia.org/wiki/Anarchism</url>
    <abstract>Anarchism is a political philosophy...</abstract>
    <links>
      <sublink linktype="nav">
        <anchor>Etymology, terminology and definition</anchor>
        <link>https://en.wikipedia.org/wiki/Anarchism#Etymology...</link>
      </sublink>
      <sublink linktype="nav">
        <anchor>History</anchor>
        <link>https://en.wikipedia.org/wiki/Anarchism#History</link>
      </sublink>
      ⋮
    </links>
  </doc>
  ⋮
</feed>
```

下载后大小超过 6 GB，非常适合这个练习。这个想法是扫描文件，找到连续的开始和结束标签`<doc>`，然后为了方便起见，使用`untangle`解析它们之间的 XML 片段。

内置的 [`mmap`](https://realpython.com/python-mmap/) 模块可以让您创建文件内容的**虚拟视图**，即使它不适合可用内存。这给人一种使用支持搜索和常规切片语法的巨大字节串的印象。如果您对如何将这个逻辑封装在一个 [Python 类](https://realpython.com/python3-object-oriented-programming/#define-a-class-in-python)中并利用一个[生成器](https://realpython.com/introduction-to-python-generators/)进行惰性评估感兴趣，那么请展开下面的可折叠部分。



下面是`XMLTagStream`类的完整代码:

```py
import mmap
import untangle

class XMLTagStream:
    def __init__(self, path, tag_name, encoding="utf-8"):
        self.file = open(path)
        self.stream = mmap.mmap(
            self.file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.tag_name = tag_name
        self.encoding = encoding
        self.start_tag = f"<{tag_name}>".encode(encoding)
        self.end_tag = f"</{tag_name}>".encode(encoding)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.stream.close()
        self.file.close()

    def __iter__(self):
        end = 0
        while (begin := self.stream.find(self.start_tag, end)) != -1:
            end = self.stream.find(self.end_tag, begin)
            yield self.parse(self.stream[begin: end + len(self.end_tag)])

    def parse(self, chunk):
        document = untangle.parse(chunk.decode(self.encoding))
        return getattr(document, self.tag_name)
```

这是一个定制的[上下文管理器](https://realpython.com/python-with-statement/)，它使用被定义为内嵌**生成器函数**的**迭代器协议**。生成的生成器对象在 XML 文档中循环，就好像它是一长串字符一样。

注意，`while`循环利用了相当新的 Python 语法，即 [walrus 操作符(`:=` )](https://realpython.com/python-walrus-operator/) ，来简化代码。您可以在**赋值表达式**中使用该操作符，表达式可以被求值并赋值给变量。

无需深入细节，下面介绍如何使用这个定制类快速浏览一个大型 XML 文件，同时使用`untangle`更彻底地检查特定元素:

>>>

```py
>>> with XMLTagStream("abstract.xml", "doc") as stream:
...     for doc in stream:
...         print(doc.title.cdata.center(50, "="))
...         for sublink in doc.links.sublink:
...             print("-", sublink.anchor.cdata)
...         if "q" == input("Press [q] to exit or any key to continue..."):
...             break
...
===============Wikipedia: Anarchism===============
- Etymology, terminology and definition
- History
- Pre-modern era
⋮
Press [q] to exit or any key to continue...
================Wikipedia: Autism=================
- Characteristics
- Social development
- Communication
⋮
Press [q] to exit or any key to continue...
```

首先，您打开一个文件进行读取，并指出您想要查找的标记名。然后，迭代这些元素，得到 XML 文档的解析片段。这几乎就像透过一个在无限长的纸上移动的小窗口看一样。这是一个相对肤浅的例子，忽略了一些细节，但是它应该让您对如何使用这种混合解析策略有一个大致的了解。

[*Remove ads*](/account/join/)

### `xmltodict`:将 XML 转换成 Python 字典

如果你喜欢 JSON，但不是 XML 的粉丝，那么看看 [`xmltodict`](https://pypi.org/project/xmltodict/) ，它试图在两种数据格式之间架起一座桥梁。顾名思义，该库可以解析 XML 文档并将其表示为 Python 字典，这也恰好是 Python 中 JSON 文档的目标数据类型。这使得 XML 和 JSON T4 之间的转换成为可能。

**注意:**字典是由键-值对组成的，而 XML 文档本来就是层次化的，这可能会导致转换过程中的一些信息丢失。最重要的是，XML 有属性、注释、处理指令和其他定义元数据的方式，这些都是字典中没有的。

与迄今为止的其他 XML 解析器不同，这个解析器期望以*二进制*模式打开一个 Python 字符串或类似文件的对象进行读取:

>>>

```py
>>> import xmltodict

>>> xmltodict.parse("""\
... <svg viewBox="-105 -100 210 270">
...   <!-- More content goes here... -->
... </svg>
... """)
OrderedDict([('svg', OrderedDict([('@viewBox', '-105 -100 210 270')]))])

>>> with open("smiley.svg", "rb") as file: ...     xmltodict.parse(file)
...
OrderedDict([('svg', ...)])
```

默认情况下，库返回一个 [`OrderedDict`](https://realpython.com/python-ordereddict/) 集合的实例来保留**元素顺序**。然而，从 Python 3.6 开始，普通字典也保持插入顺序。如果您想使用常规词典，那么将`dict`作为`dict_constructor`参数传递给`parse()`函数:

>>>

```py
>>> import xmltodict

>>> with open("smiley.svg", "rb") as file:
...     xmltodict.parse(file, dict_constructor=dict)
...
{'svg': ...}
```

现在，`parse()`返回一个普通的旧字典，带有熟悉的文本表示。

为了避免 XML 元素和它们的属性之间的**名称冲突**，库自动为后者加上前缀`@`字符。您也可以通过适当地设置`xml_attribs`标志来完全忽略属性:

>>>

```py
>>> import xmltodict

>>> # Rename attributes by default
>>> with open("smiley.svg", "rb") as file:
...     document = xmltodict.parse(file)
...     print([x for x in document["svg"] if x.startswith("@")])
...
['@xmlns', '@xmlns:inkscape', '@viewBox', '@width', '@height']

>>> # Ignore attributes when requested
>>> with open("smiley.svg", "rb") as file:
...     document = xmltodict.parse(file, xml_attribs=False) ...     print([x for x in document["svg"] if x.startswith("@")])
...
[]
```

默认情况下，另一条被忽略的信息是 **XML 名称空间**声明。这些被视为常规属性，而相应的前缀成为标记名的一部分。但是，如果需要，您可以扩展、重命名或跳过一些命名空间:

>>>

```py
>>> import xmltodict

>>> # Ignore namespaces by default
>>> with open("smiley.svg", "rb") as file:
...     document = xmltodict.parse(file)
...     print(document.keys())
...
odict_keys(['svg'])

>>> # Process namespaces when requested
>>> with open("smiley.svg", "rb") as file:
...     document = xmltodict.parse(file, process_namespaces=True)
...     print(document.keys())
...
odict_keys(['http://www.w3.org/2000/svg:svg'])

>>> # Rename and skip some namespaces
>>> namespaces = {
...     "http://www.w3.org/2000/svg": "svg",
...     "http://www.inkscape.org/namespaces/inkscape": None,
... }
>>> with open("smiley.svg", "rb") as file:
...     document = xmltodict.parse(
...         file, process_namespaces=True, namespaces=namespaces
...     )
...     print(document.keys())
...     print("custom" in document["svg:svg"])
...     print("inkscape:custom" in document["svg:svg"])
...
odict_keys(['svg:svg'])
True
False
```

在上面的第一个例子中，标记名不包括 XML 名称空间前缀。在第二个例子中，它们是因为您请求处理它们。最后，在第三个示例中，您将默认名称空间折叠为`svg`，同时用`None`取消 Inkscape 的名称空间。

Python 字典的默认字符串表示可能不够清晰。为了改善它的表现，你可以[美化](https://realpython.com/python-print/#pretty-printing-nested-data-structures)它或者将其转换成另一种格式，如 **JSON** 或 **YAML** :

>>>

```py
>>> import xmltodict
>>> with open("smiley.svg", "rb") as file:
...     document = xmltodict.parse(file, dict_constructor=dict)
...

>>> from pprint import pprint as pp
>>> pp(document)
{'svg': {'@height': '270',
 '@viewBox': '-105 -100 210 270',
 '@width': '210',
 '@xmlns': 'http://www.w3.org/2000/svg',
 '@xmlns:inkscape': 'http://www.inkscape.org/namespaces/inkscape',
 'defs': {'linearGradient': {'@id': 'skin',
 ⋮

>>> import json
>>> print(json.dumps(document, indent=4, sort_keys=True))
{
 "svg": {
 "@height": "270",
 "@viewBox": "-105 -100 210 270",
 "@width": "210",
 "@xmlns": "http://www.w3.org/2000/svg",
 "@xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
 "defs": {
 "linearGradient": {
 ⋮

>>> import yaml  # Install first with 'pip install PyYAML'
>>> print(yaml.dump(document))
svg:
 '@height': '270'
 '@viewBox': -105 -100 210 270
 '@width': '210'
 '@xmlns': http://www.w3.org/2000/svg
 '@xmlns:inkscape': http://www.inkscape.org/namespaces/inkscape
 defs:
 linearGradient:
 ⋮
```

`xmltodict`库允许反过来转换文档——也就是说，从 Python 字典转换回 XML 字符串:

>>>

```py
>>> import xmltodict

>>> with open("smiley.svg", "rb") as file:
...     document = xmltodict.parse(file, dict_constructor=dict)
...

>>> xmltodict.unparse(document)
'<?xml version="1.0" encoding="utf-8"?>\n<svg...'
```

如果需要的话，在将数据从 JSON 或 YAML 转换成 XML 时，该字典作为一种中间格式可能会派上用场。

在`xmltodict`库中还有很多特性，比如流媒体，所以你可以自由探索它们。然而，这个图书馆也有点过时了。此外，如果您真的在寻找高级 XML 解析特性，那么它是您应该考虑的下一个库。

### `lxml`:使用类固醇元素树

如果你想把最好的性能、最广泛的功能和最熟悉的界面都打包在一个包里，那么就安装 [`lxml`](https://pypi.org/project/lxml/) ，忘掉其余的库。它是 C 库 [libxml2](https://en.wikipedia.org/wiki/Libxml2) 和 [libxslt](https://en.wikipedia.org/wiki/Libxslt) 的 **Python 绑定**，支持多种标准，包括 XPath、XML Schema 和 xslt。

该库与 Python 的 **ElementTree API** 兼容，您在本教程的前面已经了解过。这意味着您可以通过只替换一条 import 语句来重用现有代码:

```py
import lxml.etree as ET
```

这将给你带来巨大的**性能提升**。最重要的是，`lxml`库提供了一组广泛的特性，并提供了使用它们的不同方式。例如，它让您**根据几种模式语言来验证**您的 XML 文档，其中之一是 XML 模式定义:

>>>

```py
>>> import lxml.etree as ET

>>> xml_schema = ET.XMLSchema(
...     ET.fromstring("""\
...         <xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
...             <xsd:element name="parent"/>
...             <xsd:complexType name="SomeType">
...                 <xsd:sequence>
...                     <xsd:element name="child" type="xsd:string"/>
...                 </xsd:sequence>
...             </xsd:complexType>
...         </xsd:schema>"""))

>>> valid = ET.fromstring("<parent><child></child></parent>")
>>> invalid = ET.fromstring("<child><parent></parent></child>")

>>> xml_schema.validate(valid)
True

>>> xml_schema.validate(invalid)
False
```

Python 标准库中的 XML 解析器都没有验证文档的能力。同时，`lxml`允许您定义一个`XMLSchema`对象并通过它运行文档，同时保持与 ElementTree API 的大部分兼容性。

除了 ElementTree API 之外，`lxml`还支持另一种接口 [lxml.objectify](https://lxml.de/objectify.html) ，这将在后面的[数据绑定](#bind-xml-data-to-python-objects)部分中介绍。

[*Remove ads*](/account/join/)

### `BeautifulSoup`:处理格式错误的 XML

在这个比较中，您通常不会使用最后一个库来解析 XML，因为您最常遇到的是 [web 抓取](https://realpython.com/beautiful-soup-web-scraper-python/) HTML 文档。也就是说，它也能够解析 XML。 [BeautifulSoup](https://pypi.org/project/beautifulsoup4/) 带有一个**可插拔架构**，可以让你选择底层解析器。前面描述的`lxml`实际上是官方文档推荐的，也是目前该库唯一支持的 XML 解析器。

根据您想要解析的文档类型、期望的效率和特性可用性，您可以选择以下解析器之一:

| 文件类型 | 分析器名称 | Python 库 | 速度 |
| --- | --- | --- | --- |
| 超文本标记语言 | `"html.parser"` | - | 温和的 |
| 超文本标记语言 | `"html5lib"` | [T2`html5lib`](https://pypi.org/project/html5lib/) | 慢的 |
| 超文本标记语言 | `"lxml"` | `lxml` | 快的 |
| 可扩展标记语言 | `"lxml-xml"`或`"xml"` | `lxml` | 快的 |

除了速度，各个解析器之间还有明显的差异。例如，当涉及到畸形元素时，它们中的一些比另一些更宽容，而另一些则更好地模拟了 web 浏览器。

**趣闻:**库名指的是[标签汤](https://en.wikipedia.org/wiki/Tag_soup)，描述语法或结构不正确的 HTML 代码。

假设您已经将`lxml`和`beautifulsoup4`库安装到活动的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中，那么您可以立即开始解析 XML 文档。你只需要导入`BeautifulSoup`:

```py
from bs4 import BeautifulSoup

# Parse XML from a file object
with open("smiley.svg") as file:
    soup = BeautifulSoup(file, features="lxml-xml")

# Parse XML from a Python string
soup = BeautifulSoup("""\
<svg viewBox="-105 -100 210 270">
 <!-- More content goes here... -->
</svg>
""", features="lxml-xml")
```

如果您不小心指定了一个不同的解析器，比如说`lxml`，那么这个库会为您将缺少的 HTML 标签，比如`<body>`添加到解析后的文档中。在这种情况下，这可能不是您想要的，所以在指定解析器名称时要小心。

BeautifulSoup 是一个强大的解析 XML 文档的工具，因为它可以处理无效内容，并且有一个丰富的 API 来提取信息。看看它是如何处理不正确的嵌套标签、禁用字符和放置不当的文本的:

>>>

```py
>>> from bs4 import BeautifulSoup

>>> soup = BeautifulSoup("""\
... <parent>
...     <child>Forbidden < character </parent>
...     </child>
... ignored
... """, features="lxml-xml")

>>> print(soup.prettify())
<?xml version="1.0" encoding="utf-8"?>
<parent>
 <child>
 Forbidden
 </child>
</parent>
```

另一个不同的解析器会引发一个[异常](https://realpython.com/python-exceptions/)，并在检测到文档有问题时立即放弃。在这里，它不仅忽略了问题，而且还找到了修复其中一些问题的明智方法。这些元素现在已经正确嵌套，并且没有无效内容。

用 BeautifulSoup 定位元素的方法太多了，这里无法一一介绍。通常，您会在 soup 元素上调用`.find()`或`.findall()`的变体:

>>>

```py
>>> from bs4 import BeautifulSoup

>>> with open("smiley.svg") as file:
...     soup = BeautifulSoup(file, features="lxml-xml")
...

>>> soup.find_all("ellipse", limit=1)
[<ellipse cx="-20" cy="-10" fill="black" rx="6" ry="8" stroke="none"/>]

>>> soup.find(x=42)
<inkscape:custom inkscape:z="555" x="42">Some value</inkscape:custom>

>>> soup.find("stop", {"stop-color": "gold"})
<stop offset="75%" stop-color="gold" stop-opacity="1.0"/>

>>> soup.find(text=lambda x: "value" in x).parent
<inkscape:custom inkscape:z="555" x="42">Some value</inkscape:custom>
```

`limit`参数类似于 MySQL 中的`LIMIT`子句，它让您决定最多希望接收多少个结果。它将返回指定数量或更少的结果。这不是巧合。您可以将这些搜索方法看作是一种简单的查询语言，带有强大的过滤器。

搜索界面非常灵活，但超出了本教程的范围。你可以查看[库的文档](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)以了解更多细节，或者阅读另一篇关于 Python 中的 [web 抓取](https://realpython.com/python-web-scraping-practical-introduction/)的教程，该教程涉及 BeautifulSoup。

## 将 XML 数据绑定到 Python 对象

假设您想通过一个低延迟的 WebSocket 连接使用一个实时数据馈送，并以 XML 格式交换消息。出于本演示的目的，您将使用 web 浏览器向 Python 服务器广播您的鼠标和键盘事件。您将构建一个**定制协议**，并使用**数据绑定**将 XML 转换成本地 Python 对象。

数据绑定背后的想法是声明性地定义一个数据模型*，同时让程序弄清楚如何在运行时从 XML 中提取有价值的信息。如果你曾经和 [Django models](https://docs.djangoproject.com/en/3.2/topics/db/models/) 一起工作过，那么这个概念应该听起来很熟悉。

首先，从设计数据模型开始。它将由两种类型的事件组成:

1.  `KeyboardEvent`
2.  `MouseEvent`

每一个都可以代表一些特殊的子类型，比如键盘的按键或释放键以及鼠标的单击或右键。下面是响应按住 `Shift` + `2` 组合键时生成的示例 XML 消息:

```py
<KeyboardEvent>
    <Type>keydown</Type>
    <Timestamp>253459.17999999982</Timestamp>
    <Key>
        <Code>Digit2</Code>
        <Unicode>@</Unicode>
    </Key>
    <Modifiers>
        <Alt>false</Alt>
        <Ctrl>false</Ctrl>
        <Shift>true</Shift>
        <Meta>false</Meta>
    </Modifiers>
</KeyboardEvent>
```

该消息包含特定的键盘事件类型、时间戳、键码及其 [Unicode](https://realpython.com/python-encodings-guide/) ，以及修改键，如 `Alt` 、 `Ctrl` 或 `Shift` 。[元键](https://en.wikipedia.org/wiki/Meta_key)通常是 `Win` 或 `Cmd` 键，这取决于你的键盘布局。

类似地，鼠标事件可能如下所示:

```py
<MouseEvent>
    <Type>mousemove</Type>
    <Timestamp>52489.07000000145</Timestamp>
    <Cursor>
        <Delta x="-4" y="8"/>
        <Window x="171" y="480"/>
        <Screen x="586" y="690"/>
    </Cursor>
    <Buttons bitField="0"/>
    <Modifiers>
        <Alt>false</Alt>
        <Ctrl>true</Ctrl>
        <Shift>false</Shift>
        <Meta>false</Meta>
    </Modifiers>
</MouseEvent>
```

然而，代替键的是鼠标光标位置和一个对事件中按下的[鼠标按钮](https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/buttons#return_value)进行编码的[位域](https://en.wikipedia.org/wiki/Bit_field)。零位域表示没有按钮被按下。

一旦客户端建立连接，它将开始向服务器发送大量消息。该协议不会包含任何握手、心跳、正常关机、主题订阅或控制消息。通过注册事件处理程序并在不到 50 行代码中创建一个`WebSocket`对象，您可以用 JavaScript 对此进行编码。

然而，实现客户机并不是本练习的重点。因为你不需要理解它，只需展开下面可折叠的部分来显示嵌入了 JavaScript 的 HTML 代码，并将其保存在一个名为随便你喜欢的文件中。



```py
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Real-Time Data Feed</title>
</head>
<body>
    <script> const  ws  =  new  WebSocket("ws://localhost:8000") ws.onopen  =  event  =>  { ["keydown",  "keyup"].forEach(name  => window.addEventListener(name,  event  => ws.send(`\
<KeyboardEvent>
 <Type>${event.type}</Type>
 <Timestamp>${event.timeStamp}</Timestamp>
 <Key>
 <Code>${event.code}</Code>
 <Unicode>${event.key}</Unicode>
 </Key>
 <Modifiers>
 <Alt>${event.altKey}</Alt>
 <Ctrl>${event.ctrlKey}</Ctrl>
 <Shift>${event.shiftKey}</Shift>
 <Meta>${event.metaKey}</Meta>
 </Modifiers>
</KeyboardEvent>`)) ); ["mousedown",  "mouseup",  "mousemove"].forEach(name  => window.addEventListener(name,  event  => ws.send(`\
<MouseEvent>
 <Type>${event.type}</Type>
 <Timestamp>${event.timeStamp}</Timestamp>
 <Cursor>
 <Delta x="${event.movementX}" y="${event.movementY}"/>
 <Window x="${event.clientX}" y="${event.clientY}"/>
 <Screen x="${event.screenX}" y="${event.screenY}"/>
 </Cursor>
 <Buttons bitField="${event.buttons}"/>
 <Modifiers>
 <Alt>${event.altKey}</Alt>
 <Ctrl>${event.ctrlKey}</Ctrl>
 <Shift>${event.shiftKey}</Shift>
 <Meta>${event.metaKey}</Meta>
 </Modifiers>
</MouseEvent>`)) ) } </script>
</body>
</html>
```

客户端连接到侦听端口 8000 的本地服务器。一旦你将 HTML 代码保存在一个文件中，你就可以用你最喜欢的浏览器打开它。但是在此之前，您需要实现服务器。

Python 没有 WebSocket 支持，但是您可以将 [`websockets`](https://pypi.org/project/websockets/) 库安装到您的活动虚拟环境中。稍后您还将需要`lxml`,因此这是一个一次性安装两个依赖项的好时机:

```py
$ python -m pip install websockets lxml
```

最后，您可以搭建一个最小的异步 web 服务器:

```py
# server.py

import asyncio
import websockets

async def handle_connection(websocket, path):
    async for message in websocket:
        print(message)

if __name__ == "__main__":
    future = websockets.serve(handle_connection, "localhost", 8000)
    asyncio.get_event_loop().run_until_complete(future)
    asyncio.get_event_loop().run_forever()
```

当您启动服务器并在 web 浏览器中打开保存的 HTML 文件时，您应该看到 XML 消息出现在标准输出中，以响应您的鼠标移动和按键。可以在多个标签页甚至多个浏览器同时打开客户端！

[*Remove ads*](/account/join/)

### 用 XPath 表达式定义模型

现在，您的消息以纯字符串格式到达。使用这种格式的信息不太方便。幸运的是，您可以使用`lxml.objectify`模块，通过一行代码将它们转换成复合 Python 对象:

```py
# server.py

import asyncio
import websockets
import lxml.objectify 
async def handle_connection(websocket, path):
    async for message in websocket:
        try:
 xml = lxml.objectify.fromstring(message)        except SyntaxError:
            print("Malformed XML message:", repr(message))
        else:
            if xml.tag == "KeyboardEvent":
                if xml.Type == "keyup":
                    print("Key:", xml.Key.Unicode)
            elif xml.tag == "MouseEvent":
                screen = xml.Cursor.Screen
                print("Mouse:", screen.get("x"), screen.get("y"))
            else:
                print("Unrecognized event type")

# ...
```

只要 XML 解析成功，就可以检查根元素的常见属性，比如标记名、属性、内部文本等等。您将能够使用点运算符导航到元素树的深处。在大多数情况下，库会识别合适的 Python 数据类型，并为您转换值。

保存这些更改并重新启动服务器后，您需要在 web 浏览器中重新加载页面，以建立新的 WebSocket 连接。下面是修改后的程序的输出示例:

```py
$ python server.py
Mouse: 820 121
Mouse: 820 122
Mouse: 820 123
Mouse: 820 124
Mouse: 820 125
Key: a
Mouse: 820 125
Mouse: 820 125
Key: a
Key: A
Key: Shift
Mouse: 821 125
Mouse: 821 125
Mouse: 820 123
⋮
```

有时，XML 可能包含不是有效 Python 标识符的标记名，或者您可能希望调整消息结构以适应您的数据模型。在这种情况下，一个有趣的选择是用声明如何使用 XPath 表达式查找信息的[描述符](https://realpython.com/python-descriptors/)定义定制的**模型类**。这是开始类似 Django 模型或 [Pydantic](https://github.com/samuelcolvin/pydantic) 模式定义的部分。

您将使用一个定制的`XPath`描述符和一个附带的`Model`类，为您的数据模型提供可重用的属性。描述符要求在收到的消息中使用 XPath 表达式进行元素查找。底层实现有点高级，所以可以随意从下面的可折叠部分复制代码。



```py
import lxml.objectify

class XPath:
    def __init__(self, expression, /, default=None, multiple=False):
        self.expression = expression
        self.default = default
        self.multiple = multiple

    def __set_name__(self, owner, name):
        self.attribute_name = name
        self.annotation = owner.__annotations__.get(name)

    def __get__(self, instance, owner):
        value = self.extract(instance.xml)
        instance.__dict__[self.attribute_name] = value
        return value

    def extract(self, xml):
        elements = xml.xpath(self.expression)
        if elements:
            if self.multiple:
                if self.annotation:
                    return [self.annotation(x) for x in elements]
                else:
                    return elements
            else:
                first = elements[0]
                if self.annotation:
                    return self.annotation(first)
                else:
                    return first
        else:
            return self.default

class Model:
    """Abstract base class for your models."""
    def __init__(self, data):
        if isinstance(data, str):
            self.xml = lxml.objectify.fromstring(data)
        elif isinstance(data, lxml.objectify.ObjectifiedElement):
            self.xml = data
        else:
            raise TypeError("Unsupported data type:", type(data))
```

假设您的模块中已经有了期望的`XPath`描述符和`Model`抽象基类，您可以使用它们来定义`KeyboardEvent`和`MouseEvent`消息类型以及可重用的构建块以避免重复。有无数种方法可以做到这一点，但这里有一个例子:

```py
# ...

class Event(Model):
    """Base class for event messages with common elements."""
    type_: str = XPath("./Type")
    timestamp: float = XPath("./Timestamp")

class Modifiers(Model):
    alt: bool = XPath("./Alt")
    ctrl: bool = XPath("./Ctrl")
    shift: bool = XPath("./Shift")
    meta: bool = XPath("./Meta")

class KeyboardEvent(Event):
    key: str = XPath("./Key/Code")
    modifiers: Modifiers = XPath("./Modifiers")

class MouseEvent(Event):
    x: int = XPath("./Cursor/Screen/@x")
    y: int = XPath("./Cursor/Screen/@y")
    modifiers: Modifiers = XPath("./Modifiers")
```

`XPath`描述符允许**惰性评估**，因此 XML 消息的元素只有在被请求时才被查找。更具体地说，只有当您访问事件对象的属性时，才会查找它们。此外，结果被**缓存**，以避免多次运行相同的 XPath 查询。描述符还考虑到了[类型注释](https://realpython.com/python-type-checking/)，并将反序列化的数据自动转换为正确的 Python 类型。

使用这些事件对象与之前由`lxml.objectify`自动生成的没有太大区别:

```py
if xml.tag == "KeyboardEvent":
 event = KeyboardEvent(xml)    if event.type_ == "keyup":
        print("Key:", event.key)
elif xml.tag == "MouseEvent":
 event = MouseEvent(xml)    print("Mouse:", event.x, event.y)
else:
    print("Unrecognized event type")
```

还有一个创建特定事件类型的新对象的额外步骤。但是除此之外，在独立于 XML 协议构建模型方面，它给了您更多的灵活性。此外，可以基于接收到的消息中的属性派生出新的模型属性，并在此基础上添加更多的方法。

### 从 XML 模式生成模型

实现模型类是一项乏味且容易出错的任务。然而，只要您的模型反映了 XML 消息，您就可以利用一个自动化的工具来基于 XML Schema 为您生成必要的代码。这种代码的缺点是通常比手写的可读性差。

最古老的第三方模块之一是 [PyXB](https://pypi.org/project/PyXB/) ，它模仿了 Java 流行的 [JAXB](https://www.baeldung.com/jaxb) 库。不幸的是，它最后一次发布是在几年前，目标是遗留的 Python 版本。您可以研究一种类似但仍被积极维护的 [`generateDS`](https://pypi.org/project/generateDS/) 替代方案，它从 XML 模式生成数据结构。

假设您有这个描述您的`KeyboardEvent`消息的`models.xsd`模式文件:

```py
<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
    <xsd:element name="KeyboardEvent" type="KeyboardEventType"/>
    <xsd:complexType name="KeyboardEventType">
        <xsd:sequence>
            <xsd:element type="xsd:string" name="Type"/>
            <xsd:element type="xsd:float" name="Timestamp"/>
            <xsd:element type="KeyType" name="Key"/>
            <xsd:element type="ModifiersType" name="Modifiers"/>
        </xsd:sequence>
    </xsd:complexType>
    <xsd:complexType name="KeyType">
        <xsd:sequence>
            <xsd:element type="xsd:string" name="Code"/>
            <xsd:element type="xsd:string" name="Unicode"/>
        </xsd:sequence>
    </xsd:complexType>
    <xsd:complexType name="ModifiersType">
        <xsd:sequence>
            <xsd:element type="xsd:string" name="Alt"/>
            <xsd:element type="xsd:string" name="Ctrl"/>
            <xsd:element type="xsd:string" name="Shift"/>
            <xsd:element type="xsd:string" name="Meta"/>
        </xsd:sequence>
    </xsd:complexType>
</xsd:schema>
```

模式告诉 XML 解析器预期的元素、它们的顺序以及它们在树中的级别。它还限制了 XML 属性的允许值。这些声明和实际的 XML 文档之间的任何差异都会使它无效，并使解析器拒绝该文档。

此外，一些工具可以利用这些信息生成一段代码，对您隐藏 XML 解析的细节。安装完库之后，您应该能够在您的活动虚拟环境中运行`generateDS`命令:

```py
$ generateDS -o models.py models.xsd
```

它将在与生成的 Python 源代码相同的目录中创建一个名为`models.py`的新文件。然后，您可以导入该模块并使用它来解析传入的消息:

>>>

```py
>>> from models import parseString

>>> event = parseString("""\
... <KeyboardEvent>
...     <Type>keydown</Type>
...     <Timestamp>253459.17999999982</Timestamp>
...     <Key>
...         <Code>Digit2</Code>
...         <Unicode>@</Unicode>
...     </Key>
...     <Modifiers>
...         <Alt>false</Alt>
...         <Ctrl>false</Ctrl>
...         <Shift>true</Shift>
...         <Meta>false</Meta>
...     </Modifiers>
... </KeyboardEvent>""", silence=True)

>>> event.Type, event.Key.Code
('keydown', 'Digit2')
```

它看起来类似于前面显示的`lxml.objectify`示例。不同之处在于，使用数据绑定强制符合模式，而`lxml.objectify`动态地产生对象，不管它们在语义上是否正确。

[*Remove ads*](/account/join/)

## 用安全解析器化解 XML 炸弹

Python 标准库中的 XML 解析器容易受到大量安全威胁的攻击，这些威胁最多会导致[拒绝服务(DoS)](https://en.wikipedia.org/wiki/Denial-of-service_attack) 或数据丢失。公平地说，那不是他们的错。他们只是遵循 XML 标准的规范，这比大多数人知道的更复杂和强大。

**注意:**请注意，您应该明智地使用您将要看到的信息。您不希望最终成为攻击者，将自己暴露在法律后果之下，或者面临终身禁止使用某个特定服务。

最常见的攻击之一是 **XML 炸弹**，也被称为[亿笑攻击](https://en.wikipedia.org/wiki/Billion_laughs_attack)。攻击利用 DTD 中的**实体扩展**来炸毁内存，尽可能长时间占用 CPU。要阻止未受保护的 web 服务器接收新流量，您只需几行 XML 代码:

```py
import xml.etree.ElementTree as ET
ET.fromstring("""\
<?xml version="1.0"?>
<!DOCTYPE lolz [
 <!ENTITY lol "lol">
 <!ELEMENT lolz (#PCDATA)>
 <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
 <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
 <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
 <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
 <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
 <!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">
 <!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">
 <!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;">
 <!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">
]>
<lolz>&lol9;</lolz>""")
```

一个天真的解析器将试图通过检查 DTD 来解析放置在文档根中的定制实体`&lol9;`。但是，该实体本身多次引用另一个实体，后者又引用另一个实体，依此类推。当您[运行上面的脚本](https://realpython.com/run-python-scripts/)时，您会注意到内存和处理单元有些令人不安的地方:

[https://player.vimeo.com/video/563603395?background=1](https://player.vimeo.com/video/563603395?background=1)

看看当其中一个 CPU 以 100%的容量工作时，主内存和交换分区是如何在几秒钟内耗尽的。当系统内存变满时，记录会突然停止，然后在 Python 进程被终止后恢复。

另一种被称为 [XXE](https://en.wikipedia.org/wiki/XML_external_entity_attack) 的流行攻击利用**通用外部实体**读取本地文件并发出网络请求。然而，从 Python 3.7.1 开始，这个特性被默认禁用，以增加安全性。如果您信任您的数据，那么您可以告诉 SAX 解析器处理外部实体:

>>>

```py
>>> from xml.sax import make_parser
>>> from xml.sax.handler import feature_external_ges

>>> parser = make_parser()
>>> parser.setFeature(feature_external_ges, True)
```

这个解析器将能够读取你的计算机上的本地文件。它可能会在类似 Unix 的操作系统上提取用户名，例如:

>>>

```py
>>> from xml.dom.minidom import parseString

>>> xml = """\
... <?xml version="1.0" encoding="UTF-8"?>
... <!DOCTYPE root [
...     <!ENTITY usernames SYSTEM "/etc/passwd">
... ]>
... <root>&usernames;</root>"""

>>> document = parseString(xml, parser)
>>> print(document.documentElement.toxml())
<root>root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
bin:x:2:2:bin:/bin:/usr/sbin/nologin
⋮
realpython:x:1001:1001:Real Python,,,:/home/realpython:/bin/bash
</root>
```

将数据通过网络发送到远程服务器是完全可行的！

现在，你如何保护自己免受这种攻击呢？Python 官方文档明确警告您使用内置 XML 解析器的风险，并建议在关键任务应用程序中切换到外部包。虽然没有随 Python 一起发布， [`defusedxml`](https://pypi.org/project/defusedxml/) 是标准库中所有解析器的**替代者**。

该库施加了严格的限制，并禁用了许多危险的 XML 特性。它应该可以阻止大多数众所周知的攻击，包括刚才描述的两种攻击。要使用它，从 PyPI 获取库并相应地替换您的导入语句:

>>>

```py
>>> import defusedxml.ElementTree as ET
>>> ET.parse("bomb.xml")
Traceback (most recent call last):
  ...
    raise EntitiesForbidden(name, value, base, sysid, pubid, notation_name)
defusedxml.common.EntitiesForbidden:
 EntitiesForbidden(name='lol', system_id=None, public_id=None)
```

就是这样！被禁止的功能不会再通过了。

## 结论

XML 数据格式是一种成熟的、功能惊人的标准，至今仍在使用，尤其是在企业环境中。选择正确的 XML 解析器对于在性能、安全性、合规性和便利性之间找到最佳平衡点至关重要。

本教程为您提供了一个详细的**路线图**,帮助您在 Python 中的 XML 解析器迷宫中导航。你知道在哪里走捷径，如何避免死胡同，节省你很多时间。

**在本教程中，您学习了如何:**

*   选择正确的 XML **解析模型**
*   使用**标准库中的 XML 解析器**
*   使用主要的 **XML 解析库**
*   使用**数据绑定**以声明方式解析 XML 文档
*   使用安全的 XML 解析器消除安全漏洞

现在，您已经理解了解析 XML 文档的不同策略以及它们的优缺点。有了这些知识，您就能够为您的特定用例选择最合适的 XML 解析器，甚至可以组合多个解析器来更快地读取几千兆字节的 XML 文件。************