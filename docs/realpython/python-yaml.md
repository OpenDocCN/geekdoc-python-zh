# YAML:Python 中缺失的电池

> 原文：<https://realpython.com/python-yaml/>

Python 经常被宣传为一种包含电池的语言，因为它几乎具备了你对编程语言的所有期望。这种说法基本上是正确的，因为标准库和外部模块涵盖了广泛的编程需求。然而，Python 缺乏对通常用于配置和序列化的 **YAML 数据格式**的内置支持，尽管这两种语言之间有明显的相似之处。

在本教程中，您将学习如何使用可用的第三方库在 Python 中使用 YAML，重点是 **PyYAML** 。如果你刚到 YAML 或者有一段时间没有使用它，那么在深入研究这个主题之前，你将有机会参加一个快速速成班。

**在本教程中，您将学习如何:**

*   **用 Python 读**和**写** YAML 文档
*   **序列化** Python 的**内置的**和**自定义的**数据类型到 YAML
*   **安全**读取来自**不可信来源的 YAML 文件**
*   控制**在较低层次解析 YAML** 文档

稍后，您将了解 YAML 的高级、潜在危险功能以及如何保护自己免受其害。为了在底层解析 YAML，您将在 HTML 中构建一个**语法高亮工具**和一个**交互式预览**。最后，您将利用自定义 YAML 标记来扩展数据格式的语法。

为了充分利用本教程，您应该熟悉 Python 中的[面向对象编程](https://realpython.com/python3-object-oriented-programming/)，并且知道如何创建一个类。如果您已经准备好了，那么您可以通过下面的链接获得您将在本教程中编写的示例的源代码:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-yaml-project-code/)在 Python 中使用 YAML。

## 在 YAML 参加速成班

在这一节中，您将了解关于 YAML 的基本事实，包括它的用法、语法以及一些独特而强大的功能。如果你以前和 YAML 一起工作过，那么你可以跳到下一节继续阅读，这一节涵盖了在 Python 中使用 YAML。

[*Remove ads*](/account/join/)

### 历史背景

YAML，与 *camel* 押韵，是一个[递归首字母缩略词](https://en.wikipedia.org/wiki/Recursive_acronym)，代表 **YAML 不是标记语言**，因为它是*而不是*标记语言！有趣的是，YAML 规范的最初草案[将该语言定义为*另一种标记语言*，但是后来采用了当前的](https://yaml.org/spec/history/2001-05-26.html) [backronym](https://en.wikipedia.org/wiki/Backronym) 来更准确地描述该语言的用途。

一种真正的[标记语言](https://en.wikipedia.org/wiki/Markup_language)，比如 Markdown 或 HTML，可以让你用混合在内容中的格式或处理指令来注释文本。因此，标记语言主要关注文本文档，而 YAML 是一种 **[数据序列化格式](https://en.wikipedia.org/wiki/Comparison_of_data-serialization_formats)** ，它与许多编程语言固有的常见数据类型集成得很好。在 YAML 没有固有的文本，只有数据来代表。

YAML 原本是为了简化[可扩展标记语言(XML)](https://en.wikipedia.org/wiki/XML) ，但实际上，它与 [JavaScript 对象符号(JSON)](https://en.wikipedia.org/wiki/JSON) 有很多共同之处。事实上，它是 JSON 的超集。

尽管 XML 最初被设计成一种为文档创建标记语言的元语言，但人们很快就将其作为标准的数据序列化格式。尖括号类似 HTML 的语法让 XML 看起来很熟悉。突然间，每个人都想使用 XML 作为他们的配置、持久性或消息格式。

作为这个领域的第一个孩子，XML 统治了这个领域很多年。它成为了一种成熟可靠的数据交换格式，并帮助形成了一些新概念，比如构建交互式网络应用。毕竟， [AJAX](https://en.wikipedia.org/wiki/Ajax_(programming)) 中的字母 *X* ，一种无需重新加载页面就能从服务器获取数据的技术，代表的正是 XML。

具有讽刺意味的是，正是 AJAX 最终导致了 XML 受欢迎程度的下降。当数据通过网络发送时，冗长、复杂和冗余的 XML 语法浪费了大量带宽。用 JavaScript 解析 XML 文档既慢又乏味，因为 XML 的固定[文档对象模型(DOM)](https://realpython.com/python-xml-parser/#document-object-model-dom) 与应用程序的数据模型不匹配。社区最终承认他们在工作中使用了错误的工具。

那就是 [JSON](https://realpython.com/python-json/) 进入画面的时候。它是在考虑数据序列化的基础上从头开始构建的。Web 浏览器可以毫不费力地解析它，因为 JSON 是他们已经支持的 JavaScript 的子集。JSON 的极简语法不仅吸引了开发人员，而且比 XML 更容易移植到其他平台。直到今天，JSON 仍然是互联网上最瘦、最快、最通用的文本数据交换格式。

YAML 与 JSON 在同一年出现，纯属巧合，它在语法和语义层面上几乎是 JSON 的完整超集。从 YAML 1.2 开始，该格式正式成为 JSON 的严格**超集，这意味着每一个有效的 JSON 文档也恰好是 YAML 文档。**

然而，实际上，这两种格式看起来不同，因为 [YAML 规范](https://yaml.org/spec/1.2.2/)通过在 JSON 之上添加更多的[语法糖](https://en.wikipedia.org/wiki/Syntactic_sugar)和特性，更加强调人类可读性。因此，YAML 更适用于手工编辑的配置文件，而不是作为一个[传输层](https://en.wikipedia.org/wiki/Transport_layer)。

### 与 XML 和 JSON 的比较

如果你熟悉 [XML](https://realpython.com/python-xml-parser/) 或 [JSON](https://realpython.com/python-json/) ，那么你可能想知道 YAML 带来了什么。这三种都是主要的数据交换格式，它们共享一些重叠的特性。例如，它们都是基于**文本的**，或多或少具有可读性。同时，它们在许多方面有所不同，这一点你接下来会发现。

**注意:**还有其他不太流行的文本数据格式，比如 [TOML](https://realpython.com/python-toml/) ，Python 中的新构建系统就是基于这种格式。目前，只有像[poems](https://realpython.com/dependency-management-python-poetry/)这样的外部打包和依赖管理工具可以读取 TOML，但是 Python 3.11 很快就会在标准库中有一个 [TOML 解析器](https://realpython.com/python311-tomllib/)。

常见的二进制数据序列化格式包括谷歌的[协议缓冲区](https://en.wikipedia.org/wiki/Protocol_Buffers)和阿帕奇的 [Avro](https://en.wikipedia.org/wiki/Apache_Avro) 。

现在来看一个样本文档，它用所有三种数据格式表示，但是表示的是同一个人。您可以单击以展开可折叠部分，并显示以这些格式序列化的数据:



```py
<?xml version="1.0" encoding="UTF-8" ?>
<person firstName="John" lastName="Doe">
    <dateOfBirth>1969-12-31</dateOfBirth>
    <married>true</married>
    <spouse>
        <person firstName="Jane" lastName="Doe">
            <dateOfBirth/> <!- This is a comment -->
        </person>
    </spouse>
</person>
```



```py
{ "person":  { "dateOfBirth":  "1969-12-31", "firstName":  "John", "lastName":  "Doe", "married":  true, "spouse":  { "dateOfBirth":  null, "firstName":  "Jane", "lastName":  "Doe" } } }
```



```py
%YAML  1.2 --- person: dateOfBirth:  1969-12-31 firstName:  John lastName:  Doe married:  true spouse: dateOfBirth:  null  # This is a comment firstName:  Jane lastName:  Doe
```

乍一看，XML 似乎具有最令人生畏的语法，这增加了许多噪音。JSON 在简单性方面极大地改善了这种情况，但是它仍然将信息隐藏在强制分隔符之下。另一方面，YAML 使用 Python 风格的**块缩进**来定义结构，使它看起来干净和简单。缺点是，在通过网络传输消息时，不能通过折叠空白来减小大小。

**注意:** JSON 是唯一不支持注释的数据格式。它们被从规范中删除是为了简化解析器，防止人们滥用它们来定制处理指令。

这里有一个对 XML、JSON 和 YAML 的主观比较，让您了解它们作为当今的数据交换格式是如何相互比较的:

|  | 可扩展标记语言 | JSON | 亚姆 |
| --- | --- | --- | --- |
| 采用和支持 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 可读性 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 读写速度 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| 文件大小 | ⭐ | ⭐⭐⭐ | ⭐⭐ |

当你查看 [Google Trends](https://trends.google.com/trends/explore?date=all&q=XML,JSON,YAML) 来追踪对这三个搜索短语的兴趣时，你会得出结论，JSON 是当前的赢家。然而，XML 紧随其后，YAML 吸引了最不感兴趣的 T2。此外，自从 Google 开始收集数据以来，XML 的受欢迎程度似乎一直在稳步下降。

**注意:**在这三种格式中，XML 和 JSON 是 Python 开箱即用支持的唯一格式，而如果您希望使用 YAML，那么您必须找到并安装相应的第三方库。然而，Python 并不是唯一一种比 YAML 更好地支持 XML 和 JSON 的语言。您可能会在各种编程语言中发现这种趋势。

YAML 可以说是最容易看的，因为可读性一直是它的核心原则之一，但是 JSON 也不错。有些人甚至会发现 JSON 不那么杂乱和嘈杂，因为它的语法非常简洁，并且与 Python 列表和字典非常相似。XML 是最冗长的，因为它需要将每一条信息包装在一对开始和结束标记中。

在 Python 中，处理这些数据格式时的性能会有所不同，并且对您选择的实现非常敏感。纯 Python 实现总会输给编译过的 C 库。除此之外，使用不同的 XML 处理模型——([DOM](https://realpython.com/python-xml-parser/#document-object-model-dom)、 [SAX](https://realpython.com/python-xml-parser/#simple-api-for-xml-sax) 或 [StAX](https://realpython.com/python-xml-parser/#streaming-api-for-xml-stax) )也会影响性能。

除了实现之外，YAML 的通用、自由和复杂的语法使它成为迄今为止解析和序列化最慢的。另一方面，你会发现 JSON，它的语法[可以放在一张名片上](https://twitter.com/jeresig/status/2875994605)。相比之下，YAML 自己的[语法文档](https://github.com/yaml/yaml-grammar)声称创建一个完全兼容的解析器几乎是不可能的。

**趣闻:**yaml.org[官方](https://yaml.org/)网站被写成有效的 YAML 文件。

说到文档大小，JSON 再次成为明显的赢家。虽然多余的引号、逗号和花括号占用了宝贵的空间，但是您可以删除各个元素之间的所有空白。您可以对 XML 文档做同样的事情，但是它不会克服开始和结束标记的开销。YAML 位于中间，拥有相对中等的面积。

从历史上看，XML 在广泛的技术中得到了最好的支持。JSON 是在互联网上传输数据的无与伦比的全方位交换格式。那么，谁在使用 YAML，为什么？

[*Remove ads*](/account/join/)

### YAML 的实际用途

如前所述，YAML 最受称赞的地方是它的可读性，这使得它非常适合以人类可读的格式存储各种配置数据。它在开发工程师中变得特别受欢迎，他们围绕它开发了自动化工具。这些工具的几个例子包括:

*   **[Ansible](https://www.ansible.com/) :** 使用 YAML 来描述远程基础设施的期望状态，管理配置，并协调 IT 流程
*   **[Docker Compose](https://docs.docker.com/compose/) :** 使用 YAML 来描述组成您的 Docker 化应用程序的微服务
*   **[Kubernetes](https://kubernetes.io/) :** 使用 YAML 来定义计算机集群中的各种对象以进行编排和管理

除此之外，一些通用工具、库和服务为您提供了通过 YAML 配置它们的选项，您可能会发现这比其他数据格式更方便。例如，像 [CircleCI](https://circleci.com/) 和 [GitHub](https://github.com/features/actions) 这样的平台经常求助于 YAML 来定义[持续集成、部署和交付(CI/CD)](https://en.wikipedia.org/wiki/CI/CD) 管道。 [OpenAPI 规范](https://swagger.io/specification/)允许基于[RESTful API](https://realpython.com/api-integration-in-python/)的 YAML 描述生成代码存根。

**注意:**Python 的[日志](https://realpython.com/python-logging/)框架的文档提到了 YAML，尽管该语言本身并不支持 YAML。

完成本教程后，也许你会决定在未来的项目中采用 YAML！

### YAML 语法

YAML 从你以前可能听说过的其他数据格式和语言中汲取了很多灵感。也许 YAML 语法中最引人注目和熟悉的元素是它的**块缩进**，它类似于 Python 代码。每行的前导空格定义了块的范围，不需要任何特殊字符或标记来表示它的开始或结束位置:

```py
grandparent: parent: child: name:  Bobby sibling: name:  Molly
```

这个示例文档定义了一个以`grandparent`为根元素的家庭树，它的直接子元素是`parent`元素，在树的最底层有两个带有`name`属性的子元素。在 Python 中，您可以将每个元素想象成一个开始语句，后跟一个冒号(`:`)。

**注意:**YAML 规范禁止使用[制表符](https://en.wikipedia.org/wiki/Tab_key)进行缩进，并认为它们的使用是一种语法错误。这与 Python 的 [PEP 8](https://realpython.com/python-pep8/) 推荐的更喜欢空格而不是制表符不谋而合。

同时，YAML 允许你利用从 JSON 借鉴来的另一种**内联块**语法。您可以用以下方式重写同一文档:

```py
grandparent: parent: child:  {name:  Bobby} sibling:  {'name':  "Molly"}
```

请注意如何在一个文档中混合缩进块和内联块。此外，如果您愿意，可以自由地将属性及其值用单引号(`'`)或双引号(`"`)括起来。这样做可以启用[特殊字符序列](https://realpython.com/python-data-types/#escape-sequences-in-strings)的两种插值方法之一，否则会用另一个反斜杠(`\`)为您转义。下面，您将在 YAML 旁边找到 Python 的等价物:

| 亚姆 | 计算机编程语言 | 描述 |
| --- | --- | --- |
| `Don''t\n` | `Don''t\\n` | 不加引号的字符串被逐字解析，这样像`\n`这样的转义序列就变成了`\\n`。 |
| `'Don''t\n'` | `Don't\\n` | 单引号字符串只内插双撇号(`''`)，而不内插`\n`这样的传统转义序列。 |
| `"Don''t\n"` | `Don''t\n` | 双引号(`"`)字符串插入转义序列，如`\n`、`\r`或`\t`，这在 C 编程语言中是已知的，但不是双撇号(`''`)。 |

如果这看起来令人困惑，不要担心。无论如何，在 YAML 的大部分情况下，您都希望指定不带引号的字符串文字。一个值得注意的例外是声明一个字符串，解析器可能会将其误解为错误的数据类型。例如，不带任何引号的`True`可能会被视为一个 [Python 布尔值](https://realpython.com/python-boolean/)。

YAML 的三种基本数据结构(T0)与 T2 的 Perl(T3)基本相同，Perl 曾是一种流行的脚本语言。它们是:

1.  **标量:**像数字、字符串或布尔这样的简单值
2.  **数组:**标量或其他集合的序列
3.  **散列:**关联数组，也称为由键值对组成的映射、字典、对象或记录

您可以定义一个 YAML 标量，类似于对应的 [Python 文字](https://docs.python.org/3/reference/lexical_analysis.html#literals)。这里有几个例子:

| 数据类型 | 亚姆 |
| --- | --- |
| 空 | `null`，`~` |
| 布尔代数学体系的 | `true`、`false`、 <mark>( **前 YAML 1.2:** 、`no`、`on`、`off` )</mark> |
| 整数 | `10`、`0b10`、`0x10`、`0o10`、 <mark>( **前 YAML 1.2:** 、`010` )</mark> |
| 浮点型 | `3.14`、`12.5e-9`、`.inf`、`.nan` |
| 线 | `Lorem ipsum` |
| 日期和时间 | `2022-01-16`、`23:59`、`2022-01-16 23:59:59` |

您可以用小写(`null`)、大写(`NULL`)或大写(`Null`)在 YAML 中编写保留字，以便将它们解析为所需的数据类型。此类单词的任何其他大小写变体都将被视为纯文本。`null`常量或其代字号(`~`)别名允许您显式声明缺少值，但是您也可以将该值留空以获得相同的效果。

**注:**YAML 的这种隐式打字看似方便，实则如同玩火，在边缘情况下会造成[严重问题](https://hitchdev.com/strictyaml/why/implicit-typing-removed/)。结果，YAML 1.2 规范放弃了对一些内置文字的支持，比如`yes`和`no`。

YAML 中的序列就像 Python 列表或 JSON 数组一样。它们在内嵌块模式中使用标准的方括号语法(`[]`)或者在块缩进时在每行的开头使用前导破折号(`-`):

```py
fruits:  [apple,  banana,  orange] veggies: -  tomato -  cucumber -  onion mushrooms: -  champignon -  truffle
```

您可以将列表项保持在与其属性名相同的缩进级别，或者添加更多的缩进，如果这样可以提高可读性的话。

最后，YAML 有类似于 [Python 字典](https://realpython.com/python-dicts/)或 JavaScript 对象的散列。它们由键组成，也称为属性或**属性**名称，后跟一个冒号(`:`)和一个值。在本节中，您已经看到了一些 YAML 散列的例子，但是这里有一个更复杂的例子:

```py
person: firstName:  John lastName:  Doe dateOfBirth:  1969-12-31 married:  true spouse: firstName:  Jane lastName:  Smith children: -  firstName:  Bobby dateOfBirth:  1995-01-17 -  firstName:  Molly dateOfBirth:  2001-05-14
```

你刚刚定义了一个人，约翰·多伊，他娶了简·史密斯，有两个孩子，鲍比和莫莉。注意孩子列表是如何包含**匿名对象**的，不像，例如，在名为`"spouse"`的属性下定义的配偶。当匿名或未命名对象作为列表项出现时，您可以通过它们的属性来识别它们，这些属性与一个短划线(`-`)对齐。

**注意:**YAML 的属性名非常灵活，因为它们可以包含空白字符，并且可以跨越多行。此外，您不仅限于使用字符串。与 JSON 不同，但与 Python 字典类似，YAML 散列允许您使用几乎任何数据类型作为键！

当然，这里你只是触及了皮毛，因为 YAML 还有很多更先进的功能。现在您将了解其中的一些。

[*Remove ads*](/account/join/)

### 独特的功能

在这一部分，你将了解 YAML 一些最独特的特色，包括:

*   数据类型
*   标签
*   锚和别名
*   合并属性
*   流动和块状样式
*   多文档流

虽然 XML 都是关于文本的，JSON 继承了 JavaScript 的少数数据类型，但 YAML 的定义特性是与现代编程语言的类型系统紧密集成。例如，您可以使用 YAML 来序列化和反序列化 Python 中内置的数据类型，如日期和时间:

| 亚姆 | 计算机编程语言 |
| --- | --- |
| `2022-01-16 23:59:59` | `datetime.datetime(2022, 1, 16, 23, 59, 59)` |
| `2022-01-16` | `datetime.date(2022, 1, 16)` |
| `23:59:59` | `86399` |
| `59:59` | `3599` |

YAML 理解各种日期和时间格式，包括 ISO 8601 标准，并且可以在任意时区工作。时间戳(如 23:59:59)被反序列化为自午夜以来经过的秒数。

为了解决潜在的歧义，您可以通过使用以双感叹号(`!!`)开头的 **YAML 标签**，将值转换为特定的数据类型。有一些[独立于语言的标签](https://yaml.org/type/index.html)，但是不同的解析器可能会提供只与你的编程语言相关的附加扩展。例如，您稍后将使用的库允许您将值转换为本机 Python 类型，甚至序列化您的自定义类:

```py
text:  !!str  2022-01-16 numbers:  !!set ?  5 ?  8 ?  13 image:  !!binary R0lGODdhCAAIAPAAAAIGAfr4+SwAA AAACAAIAAACDIyPeWCsClxDMsZ3CgA7 pair:  !!python/tuple -  black -  white center_at:  !!python/complex  3.14+2.72j person:  !!python/object:package_name.module_name.ClassName age:  42 first_name:  John last_name:  Doe
```

在日期对象旁边使用了`!!str`标记，这使得 YAML 把它当作一个普通的字符串。问号(`?`)表示 YAML 的映射键。它们通常是不必要的，但可以帮助您从另一个集合中定义一个复合键或包含保留字符的键。在这种情况下，您想要定义空白键来创建一个[集合数据结构](https://realpython.com/python-sets/)，这相当于一个没有键的映射。

而且，你可以使用`!!binary`标签来嵌入 [Base64 编码的](https://en.wikipedia.org/wiki/Base64)二进制文件比如图像或者其他资源，这些文件在 Python 中将成为 [`bytes`](https://realpython.com/python-strings/#bytes-objects) 的实例。以`!!python/`为前缀的标签由 PyYAML 提供。

上面的 YAML 文档将被翻译成下面的 Python 字典:

```py
{
    "text": "2022-01-16",
    "numbers": {8, 13, 5},
    "image": b"GIF87a\x08\x00\x08\x00\xf0\x00…",
    "pair": ("black", "white"),
    "center_at": (3.14+2.72j),
    "person": <package_name.module_name.ClassName object at 0x7f08bf528fd0>
}
```

请注意，在 YAML 标记的帮助下，解析器如何将属性值转换成各种 Python 数据类型，包括字符串、集合、字节对象、元组、复数，甚至自定义类实例。

YAML 的其他强大特性是**锚和别名**，它允许你定义一个元素一次，然后在同一个文档中多次引用它。潜在的使用案例包括:

*   重复使用发货地址进行开票
*   膳食计划中的轮换膳食
*   参考培训计划中的练习

要声明一个锚，你可以把它看作一个命名的变量，你可以使用[符号(`&` )](https://en.wikipedia.org/wiki/Ampersand) ，而稍后要取消引用这个锚，你可以使用[星号(`*` )](https://en.wikipedia.org/wiki/Asterisk) 符号:

```py
recursive:  &cycle  [*cycle] exercises: -  muscles:  &push-up -  pectoral -  triceps -  biceps -  muscles:  &squat -  glutes -  quadriceps -  hamstrings -  muscles:  &plank -  abs -  core -  shoulders schedule: monday: -  *push-up -  *squat tuesday: -  *plank wednesday: -  *push-up -  *plank
```

在这里，您已经根据之前定义的练习创建了一个锻炼计划，并在各种日常活动中重复进行。另外，`recursive`属性展示了一个[循环引用](https://en.wikipedia.org/wiki/Circular_reference)的例子。这个属性是一个序列，它的唯一元素是序列本身。换句话说，`recursive[0]`与`recursive`相同。

**注意:**与普通的 XML 和 JSON 不同，它们只能用单个根元素来表示[树状层次](https://en.wikipedia.org/wiki/Tree_structure)，YAML 也使得用[递归](https://realpython.com/python-recursion/)循环来描述[有向图](https://en.wikipedia.org/wiki/Directed_graph)结构成为可能。不过，在定制扩展或方言的帮助下，XML 和 JSON 中的交叉引用是可能的。

您还可以通过组合两个或更多对象来**合并** ( `<<`)或覆盖在其他地方定义的属性:

```py
shape:  &shape color:  blue square:  &square a:  5 rectangle: << :  *shape << :  *square b:  3 color:  green
```

`rectangle`对象继承了`shape`和`square`的属性，同时添加了一个新属性`b`，并更改了`color`的值。

YAML 的标量支持**流样式**或**块样式**，这给了你对多行字符串中换行符处理的不同级别的控制。流标量可以在其属性名所在的同一行开始，也可以跨多行:

```py
text:  Lorem ipsum dolor sit amet Lorem ipsum dolor sit amet
```

在这种情况下，每一行的前导和尾随空格将总是被折叠成一个空格，将段落变成行。这有点像 HTML 或 Markdown，会产生下面这段文本:

```py
Lorem ipsum dolor sit amet
Lorem ipsum dolor sit amet
```

如果你想知道， *[Lorem ipsum](https://en.wikipedia.org/wiki/Lorem_ipsum)* 是写作和网页设计中用来填充可用空间的常见占位符文本。它没有任何意义，因为它故意是无意义的，而且是用不恰当的拉丁文写的，让你专注于形式而不是内容。

与流标量相反，块标量允许改变如何处理[换行符](https://yaml.org/spec/1.2-old/spec.html#id2795688)、[尾随换行符](https://yaml.org/spec/1.2-old/spec.html#id2794534)或[缩进](https://yaml.org/spec/1.2-old/spec.html#id2793979)。例如，位于属性名之后的竖线(`|`)指示符按字面意思保留了换行符，这对于在您的 YAML 文件中嵌入 [shell 脚本](https://en.wikipedia.org/wiki/Shell_script)非常方便:

```py
script:  | #!/usr/bin/env python def main(): print("Hello world") if __name__ == "__main__": main()
```

上面的 YAML 文档定义了一个名为`script`的属性，它包含一个由几行代码组成的简短 Python 脚本。如果没有管道指示符，YAML 解析器会将下面几行视为嵌套元素，而不是整体。Ansible 是一个利用 YAML 这一特点的著名例子。

如果您只想折叠由段落中第一行确定缩进的行，则使用大于号(`>`)指示符:

```py
text:  > Lorem ipsum dolor sit amet Lorem ipsum dolor sit amet
```

这将产生以下输出:

```py
Lorem
  ipsum
dolor sit amet
Lorem ipsum dolor sit amet
```

最后，您可以将多个 YAML 文档存储在一个文件中，用三重破折号(`---`)分隔。您可以选择使用三点符号(`...`)来标记每个文档的结尾。

[*Remove ads*](/account/join/)

## Python YAML 入门

正如您在简介中了解到的，在 Python 中使用 YAML 需要一些额外的步骤，因为该语言不支持这种现成的数据格式。您将需要一个第三方库来将 Python 对象序列化为 YAML，反之亦然。

除此之外，您可能会发现将这些带有 [pip](https://realpython.com/what-is-pip/) 的命令行工具安装到您的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中有助于调试:

*   **[yamllint](https://pypi.org/project/yamllint/):**YAML 的一个贴子，里面可以检查语法等等
*   **[yq](https://pypi.org/project/yq/) :** 基于 [jq](https://stedolan.github.io/jq/) 的命令行 YAML 处理器，用于过滤数据
*   **[shyaml](https://pypi.org/project/shyaml/)** :替代命令行 yaml 处理器

这些都是 Python 工具，但是还有 yq 的一个广泛使用的 [Go 实现，它有一个稍微不同的命令行接口。如果您不能或不想安装这些程序，您可以随时使用在线工具，例如:](https://github.com/mikefarah/yq)

*   [YAML 解析器](https://jsonformatter.org/yaml-parser)
*   [YAML 格式化程序](https://jsonformatter.org/yaml-formatter)
*   [YAML 验证器](https://jsonformatter.org/yaml-validator)

请注意，您只需要在下面的小节中用到其中的一些工具，而在本教程的其余部分，您将会接触到纯 Python 中的 YAML。

### 将 YAML 文档序列化为 JSON

即使 Python 没有提供专用的 YAML 解析器或序列化器，您也可以借助内置的`json`模块在一定程度上避开这个问题。毕竟，您已经知道 YAML 是 JSON 的超集，所以您可以将数据转储为 Python 中的常规 JSON 格式，并期望外部 YAML 解析器接受它。

首先，制作一个示例 Python 脚本，以便在标准输出上[打印出](https://realpython.com/python-print/) JSON:

```py
# print_json.py

import datetime
import json

person = {
    "firstName": "John",
    "dateOfBirth": datetime.date(1969, 12, 31),
    "married": False,
    "spouse": None,
    "children": ["Bobby", "Molly"],
}

print(json.dumps(person, indent=4, default=str))
```

您创建了一个字典，然后对它调用`json.dumps()`来转储一个字符串。参数`default`指定当 Python 不能将对象序列化为 JSON 时要调用的函数，本例中的出生日期就是这种情况。内置的`str()`函数将把一个`datetime.date`对象转换成一个 ISO 8601 字符串。

现在，[运行您的脚本](https://realpython.com/run-python-scripts/)，并通过 Unix 管道(`|`)将其输出提供给前面提到的命令行 YAML 解析器之一，比如`yq`或`shyaml`:

```py
$ python print_json.py | yq -y .
firstName: John
dateOfBirth: '1969-12-31'
married: false
spouse: null
children:
 - Bobby
 - Molly

$ python print_json.py | shyaml get-value
firstName: John
dateOfBirth: '1969-12-31'
married: false
spouse: null
children:
- Bobby
- Molly
```

不错！两种解析器都以更规范的 YAML 格式格式化数据，而没有抱怨。然而，因为`yq`是 JSON 的`jq`的一个薄薄的包装，所以您必须请求它使用`-y`选项和一个尾随点作为过滤表达式来进行代码转换。另外，请注意`yq`和`shyaml`之间产生的缩进略有不同。

**注意:**要使用`yq`，你必须先在你的操作系统中安装`jq`，如果它还没有的话。

好吧，这感觉像是作弊，而且只有一种方式，因为你不能使用`json`模块将 YAML 文件读回 Python。谢天谢地，有办法做到这一点。

### 安装 PyYAML 库

Python 目前最流行的第三方 YAML 库是 [PyYAML](https://pypi.org/project/PyYAML/) ，它一直是从 [PyPI](https://pypi.org/) 下载的[顶级包](https://pypistats.org/top)之一。它的界面看起来有点类似于内置的 JSON 模块，它得到了积极的维护，它得到了 YAML 官方网站的支持，该网站将它与一些不太受欢迎的竞争者列在一起。

要将 PyYAML 安装到您的活动虚拟环境中，请在您的终端中键入以下命令:

```py
(venv) $ python -m pip install pyyaml
```

该库是自包含的，不需要任何进一步的依赖，因为它是用纯 Python 编写的。然而，大多数发行版为 [LibYAML](https://github.com/yaml/libyaml) 库捆绑了一个编译好的 [C 绑定](https://realpython.com/python-bindings-overview/)，这使得 PyYAML 运行得更快。要确认 PyYAML 安装是否附带了 C 绑定，请打开交互式 Python 解释器并运行以下代码片段:

>>>

```py
>>> import yaml
>>> yaml.__with_libyaml__
True
```

尽管 PyYAML 是您已经安装的库的名称，但是您将导入 Python 代码中的`yaml`包。另外，请注意，您需要明确请求 PyYAML 利用明显更快的共享 C 库，否则它将退回到默认的纯 Python。请继续阅读，了解如何改变这种默认行为。

尽管 PyYAML 很受欢迎，但它也有一些缺点。例如，如果您需要使用 YAML 1.2 中引入的特性，比如完全 JSON 兼容性或更安全的文字，那么您最好使用从更早的 PyYAML 版本派生而来的 [ruamel.yaml](https://pypi.org/project/ruamel.yaml/) 库。另外，它可以进行**往返解析**，以在需要时保留注释和原始格式。

另一方面，如果**类型安全**是您主要关心的问题，或者您想要根据**模式**来验证 YAML 文档，那么看看 [StrictYAML](https://pypi.org/project/strictyaml/) ，它通过忽略其最危险的特性来有意限制 YAML 规范。请记住，它不会像其他两个库一样运行得那么快。

现在，在本教程的剩余部分，您将继续使用 PyYAML，因为它是大多数 Python 项目的标准选择。请注意，前面列出的工具——YAML int、yq 和 shy AML——在表面下使用 PyYAML！

[*Remove ads*](/account/join/)

### 阅读并编写您的第一份 YAML 文档

假设您想要读取并解析一封假想的电子邮件，该邮件已经序列化为 YAML 格式，并存储在 Python 中的一个[字符串](https://realpython.com/python-strings/)变量中:

>>>

```py
>>> email_message = """\
... message:
...   date: 2022-01-16 12:46:17Z
...   from: john.doe@domain.com
...   to:
...     - bobby@domain.com
...     - molly@domain.com
...   cc:
...     - jane.doe@domain.com
...   subject: Friendly reminder
...   content: |
...     Dear XYZ,
... ...     Lorem ipsum dolor sit amet...
...   attachments:
...     image1.gif: !!binary
...         R0lGODdhCAAIAPAAAAIGAfr4+SwAA
...         AAACAAIAAACDIyPeWCsClxDMsZ3CgA7
... """
```

将这样一段 YAML 反序列化为 Python 字典的最快方法是通过`yaml.safe_load()`函数:

>>>

```py
>>> import yaml
>>> yaml.safe_load(email_message)
{
 'message': {
 'date': datetime.datetime(2022, 1, 16, 12, 46, 17, tzinfo=(...)),
 'from': 'john.doe@domain.com',
 'to': ['bobby@domain.com', 'molly@domain.com'],
 'cc': ['jane.doe@domain.com'],
 'subject': 'Friendly reminder',
 'content': 'Dear XYZ,\n\nLorem ipsum dolor sit amet...\n',
 'attachments': {
 'image1.gif': b'GIF87a\x08\x00\x08\x00\xf0\x00\x00\x02...'
 }
 }
}
```

调用`safe_load()`是目前推荐的处理来自**不可信来源**的内容的方式，这些内容可能包含恶意代码。YAML 有一个富有表现力的语法，充满了方便的特性，不幸的是，这为大量的漏洞打开了大门。稍后你会学到更多关于[利用 YAML](#explore-loaders-insecure-features)的弱点。

**注意:**在 PyYAML 库 6.0 版本之前，解析 YAML 文档的默认方式一直是`yaml.load()`函数，默认使用不安全的解析器。在最新的版本中，您仍然可以使用这个函数，但是它要求您显式地指定一个特定的 loader 类作为第二个参数。

引入这个额外的参数是一个**突破性的变化**,导致了许多维护依赖 PyYAML 的软件的人的抱怨。关于这种向后不兼容，在该库的 GitHub 库上仍然有一个[固定的问题](https://github.com/yaml/pyyaml/issues/576)。

在撰写本教程时，官方的 [PyYAML 文档](https://pyyaml.org/wiki/PyYAMLDocumentation)以及捆绑的[文档字符串](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)还没有更新以反映当前的代码库，它们包含不再工作的示例。

`safe_load()`函数是几个**速记函数**中的一个，这些函数封装了各种 YAML 加载器类的使用。在这种情况下，单个函数调用会转化为以下更显式但等效的代码片段:

>>>

```py
>>> from yaml import load, SafeLoader
>>> load(email_message, SafeLoader)
{
 'message': {
 'date': datetime.datetime(2022, 1, 16, 12, 46, 17, tzinfo=(...)),
 'from': 'john.doe@domain.com',
 'to': ['bobby@domain.com', 'molly@domain.com'],
 'cc': ['jane.doe@domain.com'],
 'subject': 'Friendly reminder',
 'content': 'Dear XYZ,\n\nLorem ipsum dolor sit amet...\n',
 'attachments': {
 'image1.gif': b'GIF87a\x08\x00\x08\x00\xf0\x00\x00\x02...'
 }
 }
}
```

使用速记函数时要记住的一点是，它们硬编码了纯 Python 实现。如果你想使用更快的 **C 实现**，那么你必须自己写一点点[样板代码](https://en.wikipedia.org/wiki/Boilerplate_code):

>>>

```py
>>> try:
...     from yaml import CSafeLoader as SafeLoader
... except ImportError:
...     from yaml import SafeLoader

>>> SafeLoader
<class 'yaml.cyaml.CSafeLoader'>
```

首先，您尝试导入一个以字母 *C* 为前缀的 loader 类来表示 C 库绑定的使用。如果失败，那么导入一个用 Python 实现的相应类。不幸的是，这使您的代码看起来更加冗长，并阻止您使用上面提到的快捷函数。

**注意:**如果你的 YAML 包含多个文档，那么`load()`或者它的包装器会抛出一个[异常](https://realpython.com/python-exceptions/)。

您之前已经通过滥用内置的`json`模块将一个 Python 对象序列化到 YAML，但是结果不是 YAML 的规范形式。现在，您将利用已安装的第三方 PyYAML 库来解决这个问题。有一个对应的`yaml.safe_dump()`函数，它接受一个 Python 对象并将其转换成一个字符串。您可以将`yaml.safe_load()`的输出提供给它，以便逆转解析过程:

>>>

```py
>>> yaml.safe_dump(yaml.safe_load(email_message))
"message:\n  attachments:\n    image1.gif: !!binary |\n  (...)

>>> print(yaml.safe_dump(yaml.safe_load(email_message)))
message:
 attachments:
 image1.gif: !!binary |
 R0lGODdhCAAIAPAAAAIGAfr4+SwAAAAACAAIAAACDIyPeWCsClxDMsZ3CgA7
 cc:
 - jane.doe@domain.com
 content: 'Dear XYZ,

 Lorem ipsum dolor sit amet...

 '
 date: 2022-01-16 12:46:17+00:00
 from: john.doe@domain.com
 subject: Friendly reminder
 to:
 - bobby@domain.com
 - molly@domain.com
```

结果是一个字符串对象，您的电子邮件再次序列化到 YAML。然而，这与你最初开始时的 YAML 不太一样。如您所见，`safe_dump()`为您排序了字典键，引用了多行字符串，并使用了略有不同的缩进。你可以通过几个关键字参数改变其中的一些内容，并对格式进行更多的调整，这将在下一节的[中探讨。](#tweak-the-formatting-with-optional-parameters)

## 用 Python 加载 YAML 文档

加载 YAML 归结为读取一段文本并根据数据格式的语法对其进行解析。由于可供选择的函数和类过多，PyYAML 可能会使这种情况变得混乱。另外，该库的文档并没有清楚地解释它们的区别和有效的用例。为了避免您调试底层代码，您将在本节中找到关于使用 PyYAML 加载文档的最重要的事实。

### 选择加载器类别

如果您想要最好的解析性能，那么您需要手动导入合适的 loader 类，并将其传递给通用的`yaml.load()`函数，如前所示。但是你应该选择哪一个呢？

要找到答案，请看一下您可以使用的装载机的高层次概述。简短的描述应该让您对可用的选项有一个大致的了解:

| 加载程序类 | 功能 | 描述 |
| --- | --- | --- |
| `BaseLoader` | - | 不解析或支持任何标签，只构造基本的 Python 对象(`str`、`list`、`dict`) |
| `Loader` | - | 保持向后兼容性，其他方面与`UnsafeLoader`相同 |
| `UnsafeLoader` | `unsafe_load()` | 支持所有标准、库和自定义标签，并且可以构造任意的 Python 对象 |
| `SafeLoader` | `safe_load()` | 只支持像`!!str`这样的标准 YAML 标签，不构造类实例 |
| `FullLoader` | `full_load()` | 应该可以安全地装载几乎整个 YAML |

您最有可能使用的三个加载器都有相应的速记函数，您可以调用这些函数，而不是将加载器类传递给通用的`yaml.load()`函数。记住，这些都是用 Python 写的，所以为了提高性能，你需要导入一个合适的以字母 *C* 为前缀的加载器类，比如`CSafeLoader`，并调用`yaml.load()`。

有关各个加载器类支持的特性的更详细的分类，请查看下表:

| 加载程序类 | 锚，别名 | YAML 标签 | PyYAML 标记 | 辅助类型 | 自定义类型 | 代码执行 |
| --- | --- | --- | --- | --- | --- | --- |
| `UnsafeLoader` ( `Loader`) | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| `FullLoader` | ✔️ | ✔️ | ✔️ | ✔️ | 错误 | 错误 |
| `SafeLoader` | ✔️ | ✔️ | 错误 | ✔️ | 错误 | 错误 |
| `BaseLoader` | ✔️ | 忽视 | 忽视 | 忽视 | 忽视 | 忽视 |

`UnsafeLoader`支持所有可用功能，允许任意代码执行。除了代码执行和反序列化定制 Python 类的能力(这会导致解析错误)之外,`FullLoader`是类似的。`SafeLoader`py YAML 提供的特定于 Python 的标签也有错误，比如`!!python/tuple`。另一方面，`BaseLoader`通过忽略大多数特性来保持不可知论。

[*Remove ads*](/account/join/)

### 比较装载机的特性

下面，您将获得上述特性的快速演示。首先，导入`yaml`模块并检查一个**锚和别名**示例:

>>>

```py
>>> import yaml

>>> yaml.safe_load("""
... Shipping Address: &shipping |
...     1111 College Ave
...     Palo Alto
...     CA 94306
...     USA
... Invoice Address: *shipping
... """)
{
 'Shipping Address': '1111 College Ave\nPalo Alto\nCA 94306\nUSA\n',
 'Invoice Address': '1111 College Ave\nPalo Alto\nCA 94306\nUSA\n'
}
```

您在发货地址附近定义了一个锚点`&shipping`，然后在别名`*shipping`的帮助下为发票重新使用同一个地址。因此，您只需指定一次地址。此功能适用于所有类型的装载机。

下一个例子展示了一个标准的 **YAML 标签**的作用:

>>>

```py
>>> yaml.safe_load("""
... number: 3.14
... string: !!str 3.14
... """)
{'number': 3.14, 'string': '3.14'}

>>> from yaml import BaseLoader
>>> yaml.load("""
... number: 3.14
... string: !!str 3.14
... """, BaseLoader)
{'number': '3.14', 'string': '3.14'}
```

默认情况下，像`3.14`这样的数字文字被视为浮点数，但是您可以使用`!!str`标签请求将类型转换为字符串。几乎所有装载机都遵守标准的 YAML 标签。唯一的例外是`BaseLoader`类，它用字符串表示标量，不管你是否标记它们。

为了利用库提供的 **PyYAML 标签**，可以使用`FullLoader`或`UnsafeLoader`，因为它们是唯一可以处理特定于 Python 的标签的加载器:

>>>

```py
>>> yaml.full_load("""
... list: [1, 2]
... tuple: !!python/tuple [1, 2]
... """)
{'list': [1, 2], 'tuple': (1, 2)}
```

上面例子中的`!!python/tuple`标签将一个内联列表转换成一个 Python 元组。前往 PyYAML 文档以获得支持标签的完整[列表，但是一定要交叉检查 GitHub](https://pyyaml.org/wiki/PyYAMLDocumentation#yaml-tags-and-python-types) 上的[源代码，因为文档可能不是最新的。](https://github.com/yaml/pyyaml/blob/8cdff2c80573b8be8e8ad28929264a913a63aa33/lib/yaml/constructor.py#L662)

大多数加载器都善于将标量反序列化为辅助类型，比基本的字符串、列表或字典更加具体:

>>>

```py
>>> yaml.safe_load("""
... married: false
... spouse: null
... date_of_birth: 1980-01-01
... age: 42
... kilograms: 80.7
... """)
{
 'married': False,
 'spouse': None,
 'date_of_birth': datetime.date(1980, 1, 1),
 'age': 42,
 'kilograms': 80.7
}
```

在这里，您有一个混合的类型，包括一个`bool`、一个`None`、一个`datetime.date`实例、一个`int`和一个`float`。同样，`BaseLoader`是唯一一个始终将所有标量视为字符串的加载器类。

假设您想从 YAML 反序列化一个**自定义类**，在您的 Python 代码中调用一个**函数**，或者甚至在解析 YAML 时执行一个 **shell 命令**。在这种情况下，您唯一的选择是`UnsafeLoader`，它接受一些特殊的库标签。其他加载程序要么抛出异常，要么忽略这些标签。现在您将了解更多关于 PyYAML 标记的内容。

### 探索装载机的不安全特性

PyYAML 允许您通过接入接口来序列化和反序列化任何可选择的 Python 对象。请记住，这允许任意代码执行，您很快就会发现这一点。然而，如果您不在乎损害您的应用程序的安全性，那么这个功能会非常方便。

该库提供了几个由`UnsafeLoader`识别的 YAML 标签来完成对象创建:

*   `!!python/object`
*   `!!python/object/new`
*   `!!python/object/apply`

它们后面都必须跟一个要实例化的类的完全限定名，包括[包和模块名](https://realpython.com/python-modules-packages/)。第一个标记期望一个键-值对的映射，要么是流样式，要么是块样式。这里有一个例子:

```py
# Flow style: !!python/object:models.Person  {first_name:  John, last_name:  Doe} # Block style: !!python/object:models.Person first_name:  John last_name:  Doe
```

其他两个标签更复杂，因为每个标签都有两种风格。然而，这两个标签几乎是相同的，因为`!!python/object/new`将处理委托给了`!!python/object/apply`。唯一的区别是，`!!python/object/new`在指定的类上调用特殊方法 [`.__new__()`](https://realpython.com/python-class-constructor/#object-creation-with-__new__) ，而不调用 [`.__init__()`](https://realpython.com/python-class-constructor/#object-initialization-with-__init__) ，而`!!python/object/apply`调用类本身，这是您在大多数情况下想要的。

该语法的一种风格允许通过一列[位置参数](https://realpython.com/defining-your-own-python-function/#positional-arguments)来设置对象的初始状态，如下所示:

```py
# Flow style: !!python/object/apply:models.Person  [John,  Doe] # Block style: !!python/object/apply:models.Person -  John -  Doe
```

这两种风格都通过调用`Person`类中的`.__init__()`方法来实现类似的效果，这两个方法将两个值作为位置参数传递。或者，您可以使用稍微冗长一点的语法，允许您混合位置和[关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-arguments)，以及一些为了简洁而忽略的更高级的技巧:

```py
!!python/object/apply:models.Person args:  [John] kwds:  {last_name:  Doe}
```

这仍然会调用你的类上的`.__init__()`,但是其中一个参数会作为关键字参数传递。在任何情况下，您都可以手动定义`Person`类，或者利用 Python 中的[数据类](https://realpython.com/python-data-classes/):

```py
# models.py

from dataclasses import dataclass

@dataclass
class Person:
   first_name: str
   last_name: str
```

这种简洁的语法将使 Python 生成类初始化器以及其他一些您必须自己编码的方法。

注意，您可以对任何**可调用对象**使用`!!python/object/apply`，包括常规函数，并指定要传递的参数。这允许您执行一个内置函数、一个定制函数，甚至是一个模块级函数，PyYAML 会很乐意为您导入这些函数。那是一个*巨大的*安全漏洞！想象一下使用`os`或`subprocess`模块运行一个 shell 命令来检索您的私有 SSH 密钥(如果您已经定义了一个):

>>>

```py
>>> import yaml
>>> yaml.unsafe_load("""
... !!python/object/apply:subprocess.getoutput
...     - cat ~/.ssh/id_rsa
... """)
'-----BEGIN RSA PRIVATE KEY-----\njC7PbMIIEow...
```

创建对象时，通过网络用窃取的数据发出 HTTP 请求并不困难。不良行为者可能会利用这些信息，使用您的身份访问敏感资源。

有时，这些标记会绕过正常的对象创建路径，这通常是对象序列化机制的典型特征。假设您想从 YAML 加载一个用户对象，并使其成为以下类的实例:

```py
# models.py

class User:
    def __init__(self, name):
        self.name = name
```

您将`User`类放在一个名为`models.py`的单独的源文件中，以保持有序。用户对象只有一个属性—名称。通过只使用一个属性并显式实现初始化器，您将能够观察 PyYAML 调用各个方法的方式。

当你决定在 YAML 使用`!!python/object`标签时，那么库调用`.__new__()`而没有任何参数，并且从不调用`.__init__()`。相反，它直接操纵新创建对象的`.__dict__`属性，这可能会产生一些不良影响:

>>>

```py
>>> import yaml

>>> user = yaml.unsafe_load("""
... !!python/object:models.User
... no_such_attribute: 42
... """)

>>> user
<models.User object at 0x7fe8adb12050>

>>> user.no_such_attribute
42

>>> user.name
Traceback (most recent call last):
  ...
AttributeError: 'User' object has no attribute 'name'
```

虽然您无疑已经创建了一个新的`User`实例，但是它没有正确初始化，因为缺少了`.name`属性。然而，它确实有一个意想不到的`.no_such_attribute`，这在类体中是无处可寻的。

您可以通过在您的类中添加一个 [`__slots__`](https://realpython.com/python-data-classes/#optimizing-data-classes) 声明来解决这个问题，一旦对象存在于内存中，它将禁止动态添加或删除属性:

```py
# models.py

class User:
 __slots__ = ["name"] 
    def __init__(self, name):
        self.name = name
```

现在，你的用户对象根本不会有 [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 属性。因为没有固有的`.__dict__`，所以对于空白对象上的每个键-值对，这个库只能调用 [`setattr()`](https://docs.python.org/3/library/functions.html#setattr) 。这确保了只有`__slots__`中列出的属性会通过。

这些都很好，但是如果`User`类接受了密码参数会怎么样呢？为了减少数据泄漏，您肯定不希望以明文形式序列化密码。那么[序列化有状态属性](https://docs.python.org/3/library/pickle.html#pickle-state)怎么样呢，比如文件描述符或者数据库连接？好吧，如果恢复对象的状态需要运行一些代码，那么您可以在您的类中使用特殊的 [`.__setstate__()`](https://docs.python.org/3/library/pickle.html#object.__setstate__) 方法定制序列化过程:

```py
# models.py

import codecs

class User:
    __slots__ = ["name"]

    def __init__(self, name):
        self.name = name

    def __setstate__(self, state):
        self.name = codecs.decode(state["name"], "rot13")
```

您使用原始的 [ROT-13](https://en.wikipedia.org/wiki/ROT13) 密码对持久化的用户名进行解码，该密码将字符在字母表中旋转十三个位置。但是，对于严格的加密，您必须超越标准库。请注意，如果您想安全地存储密码，也可以使用内置 [`hashlib`](https://docs.python.org/3/library/hashlib.html#module-hashlib) 模块中的[哈希算法](https://realpython.com/python-hash-table/#understand-the-hash-function)。

这里有一种从 YAML 加载编码状态的方法:

>>>

```py
>>> user = yaml.unsafe_load("""
... !!python/object:models.User
... name: Wbua Qbr
... """)

>>> user.name
'John Doe'
```

只要你已经定义了`.__setstate__()`方法，它总是优先的，并且给你设置对象状态的控制权。这就是为什么您能够从上面的编码文本中恢复原来的名称`'John Doe'`。

在继续之前，值得注意的是 PyYAML 提供了两个更不安全的标签:

*   `!!python/name`
*   `!!python/module`

第一种方法允许您在代码中加载对 Python 对象的引用，如类、函数或变量。第二个标记允许引用给定的 Python 模块。在下一节中，您将看到 PyYAML 允许您从中加载文档的不同数据源。

[*Remove ads*](/account/join/)

### 从字符串、文件或流中加载文档

一旦选择了 loader 类或使用了其中一个速记函数，就不再局限于只解析字符串了。PyYAML 公开的`safe_load()`和其他函数接受单个参数，这是字符或字节的通用流。这种流最常见的例子是字符串和 Python `bytes`对象:

>>>

```py
>>> import yaml

>>> yaml.safe_load("name: Иван")
{'name': 'Иван'}

>>> yaml.safe_load(b"name: \xd0\x98\xd0\xb2\xd0\xb0\xd0\xbd")
{'name': 'Иван'}
```

根据 YAML 1.2 规范，为了与 JSON 兼容，解析器应该支持用 [UTF-8](https://en.wikipedia.org/wiki/UTF-8) 、 [UTF-16](https://en.wikipedia.org/wiki/UTF-16) 或 [UTF-32](https://en.wikipedia.org/wiki/UTF-32) 编码的 [Unicode](https://realpython.com/python-encodings-guide/) 。但是，因为 PyYAML 库只支持 YAML 1.1，所以您唯一的选择是 UTF-8 和 UTF-16:

>>>

```py
>>> yaml.safe_load("name: Иван".encode("utf-8"))
{'name': 'Иван'}

>>> yaml.safe_load("name: Иван".encode("utf-16"))
{'name': 'Иван'}

>>> yaml.safe_load("name: Иван".encode("utf-32"))
Traceback (most recent call last):
  ...
yaml.reader.ReaderError: unacceptable character #x0000:
special characters are not allowed
 in "<byte string>", position 1
```

如果你尝试从 UTF-32 编码的文本中加载 YAML，你会得到一个错误。然而，这在实践中几乎不成问题，因为 UTF-32 不是一种常见的编码。在任何情况下，您都可以在加载 YAML 之前，使用 Python 的 [`str.encode()`和`str.decode()`](https://realpython.com/python-encodings-guide/#encoding-and-decoding-in-python-3) 方法自己进行适当的代码转换。或者，您可以尝试前面提到的其他 YAML 解析库。

你也可以直接从文件中读取 YAML 的内容。继续操作，[创建一个包含示例 YAML 内容的文件](https://realpython.com/working-with-files-in-python/),并使用 PyYAML 将其加载到 Python 中:

>>>

```py
>>> with open("sample.yaml", mode="wb") as file:
...     file.write(b"name: \xd0\x98\xd0\xb2\xd0\xb0\xd0\xbd")
...
14

>>> with open("sample.yaml", mode="rt", encoding="utf-8") as file:
...     print(yaml.safe_load(file))
...
{'name': 'Иван'}

>>> with open("sample.yaml", mode="rb") as file:
...     print(yaml.safe_load(file))
...
{'name': 'Иван'}
```

您在当前工作目录下创建一个名为`sample.yaml`的本地文件，并编写 14 个字节来表示一个示例 YAML 文档。接下来，您打开该文件进行读取，并使用`safe_load()`获取相应的字典。该文件可以以文本或二进制模式打开。事实上，您可以传递任何类似文件的字符流或字节流，比如内存中的 [io。StringIO](https://docs.python.org/3/library/io.html#io.StringIO) 文本缓冲区或二进制 [io。字节流](https://docs.python.org/3/library/io.html#io.BytesIO):

>>>

```py
>>> import io

>>> yaml.safe_load(io.StringIO("name: Иван"))
{'name': 'Иван'}

>>> yaml.safe_load(io.BytesIO(b"name: \xd0\x98\xd0\xb2\xd0\xb0\xd0\xbd"))
{'name': 'Иван'}
```

如您所见，PyYAML 中的加载函数非常通用。比较一下`json`模块，它根据输入参数的类型提供不同的函数。然而，PyYAML 捆绑了另一组函数，可以帮助您从一个流中读取多个文档。现在您将了解这些功能。

### 加载多个文档

PyYAML 中的所有四个加载函数都有它们的 iterable 对应物，可以从单个流中读取多个 YAML 文档。他们仍然只需要一个参数，但是他们没有立即将它解析成 Python 对象，而是用一个可以迭代的[生成器迭代器](https://docs.python.org/3/glossary.html#term-generator-iterator)将它包装起来:

>>>

```py
>>> import yaml

>>> stream = """\
... ---
... 3.14
... ---
... name: John Doe
... age: 53
... ---
... - apple
... - banana
... - orange
... """

>>> for document in yaml.safe_load_all(stream):
...     print(document)
...
3.14
{'name': 'John Doe', 'age': 53}
['apple', 'banana', 'orange']
```

单个文档必须以三连破折号(`---`)开头，也可以选择以三个点(`...`)结尾。

在本节中，您了解了 PyYAML 中用于加载文档的高级函数。不幸的是，他们试图急切地一口气读完整个流，这并不总是可行的。以这种方式读取大文件会花费太长时间，甚至由于有限的内存而失败。如果您想以类似于 XML 中 SAX 接口的流式方式处理 YAML，那么您必须使用 PyYAML 提供的[低级 API](#parsing-yaml-documents-at-a-low-level) 。

## 将 Python 对象转储到 YAML 文档中

如果您以前在 Python 中使用过 JSON，那么将 Python 对象序列化或“转储”到 YAML 看起来会比加载它们更熟悉。PyYAML 库有一个有点类似于内置`json`模块的接口。它还提供了比加载器更少的转储器类和包装器函数可供选择，因此您不必处理那么多选项。

### 选择转储器类别

PyYAML 中全面的 YAML 序列化函数是`yaml.dump()`，它以一个可选的 dumper 类作为参数。如果在函数调用过程中没有指定，那么它会使用特性最丰富的`yaml.Dumper`。其他选择如下:

| 转储器类别 | 功能 | 描述 |
| --- | --- | --- |
| `BaseDumper` | `dump(Dumper=BaseDumper)` | 不支持任何标签，只对子类化有用 |
| `SafeDumper` | `safe_dump()` | 只产生像`!!str`这样的标准 YAML 标签，并且不能表示类实例，这使得它与其他 YAML 解析器更加兼容 |
| `Dumper` | `dump()` | 支持所有标准、库和自定义标签，可以序列化任意 Python 对象，因此它可能会生成其他 YAML 解析器无法加载的文档 |

实际上，你真正的选择是在`Dumper`和`SafeDumper`之间，因为`BaseDumper`只是作为子类扩展的基类。通常，在大多数情况下，您会希望坚持使用默认的`yaml.Dumper`，除非您需要生成没有 Python 特有的怪癖的可移植 YAML。

同样，记住导入相应的以字母 *C* 为前缀的转储器类，以获得最佳的序列化性能，并且记住 Python 和 C 实现之间可能会有细微的差别:

>>>

```py
>>> import yaml
>>> print(yaml.dump(3.14, Dumper=yaml.Dumper))
3.14
...

>>> print(yaml.dump(3.14, Dumper=yaml.CDumper))
3.14
```

例如，pure Python dumper 在 YAML 文档的末尾添加了可选的点，而 LibYAML 库的一个类似的包装器类没有这样做。但是，这些只是表面上的差异，对序列化或反序列化的数据没有实际影响。

[*Remove ads*](/account/join/)

### 转储到字符串、文件或流

在 Python 中序列化 JSON 需要您根据您希望内容被转储到哪里来选择是调用`json.dump()`还是`json.dumps()`。另一方面，PyYAML 提供了一个二合一的转储函数，根据您对它的调用方式，它的行为会有所不同:

>>>

```py
>>> data = {"name": "John"}

>>> import yaml
>>> yaml.dump(data)
'name: John\n'

>>> import io
>>> stream = io.StringIO()
>>> print(yaml.dump(data, stream))
None

>>> stream.getvalue()
'name: John\n'
```

当用单个参数调用时，该函数返回一个表示序列化对象的字符串。但是，您可以选择传递第二个参数来指定要写入的目标流。它可以是一个文件或任何类似文件的对象。当您传递这个可选参数时，函数返回 [`None`](https://realpython.com/null-in-python/) ，您需要根据需要从流中提取数据。

如果你想把你的 YAML 转储到一个文件中，那么一定要在**写模式**下打开这个文件。此外，当文件以二进制模式打开时，必须通过可选的关键字参数将字符编码指定给`yaml.dump()`函数:

>>>

```py
>>> with open("/path/to/file.yaml", mode="wt", encoding="utf-8") as file:
...     yaml.dump(data, file)

>>> with open("/path/to/file.yaml", mode="wb") as file:
...     yaml.dump(data, file, encoding="utf-8")
```

当你在文本模式下打开一个文件时，显式地设置字符编码总是一个好习惯。否则，Python 将采用您平台的默认编码，这可能会降低可移植性。字符编码在二进制模式下没有意义，二进制模式处理的是已经编码的字节。不过，您应该通过`yaml.dump()`函数设置编码，该函数接受更多可选参数，您很快就会了解到。

### 转储多个文档

PyYAML 中的两个 YAML 转储函数`dump()`和`safe_dump()`无法知道您是想要序列化多个单独的文档还是包含一个元素序列的单个文档:

>>>

```py
>>> import yaml
>>> print(yaml.dump([
...     {"title": "Document #1"},
...     {"title": "Document #2"},
...     {"title": "Document #3"},
... ]))
- title: 'Document #1'
- title: 'Document #2'
- title: 'Document #3'
```

他们总是假设后者，转储一个带有元素列表的 YAML 文档。要转储多个文档，请使用`dump_all()`或`safe_dump_all()`:

>>>

```py
>>> print(yaml.dump_all([
...     {"title": "Document #1"},
...     {"title": "Document #2"},
...     {"title": "Document #3"},
... ]))
title: 'Document #1'
---
title: 'Document #2'
---
title: 'Document #3'
```

现在，您得到一个包含多个 YAML 文档的字符串，用三重破折号(`---`)分隔。

请注意，`dump_all()`是唯一使用的函数，因为所有其他函数，包括`dump()`和`safe_dump()`，都将处理委托给它。所以，不管你调用哪个函数，它们都有相同的形参列表。

### 用可选参数调整格式

PyYAML 中的转储函数接受一些位置参数和一些可选的关键字参数，这使您可以控制输出的格式。唯一需要的参数是 Python 对象或要序列化的对象序列，在所有转储函数中作为第一个参数传递。您将仔细查看本节中的可用参数。

委托给`yaml.dump_all()`的三个包装器有下面的[函数签名](https://en.wikipedia.org/wiki/Type_signature#Signature)，揭示了它们的位置参数:

```py
def dump(data, stream=None, Dumper=Dumper, **kwargs): ...
def safe_dump(data, stream=None, **kwargs): ...
def safe_dump_all(documents, stream=None, **kwargs): ...
```

第一个函数需要一到三个位置参数，因为其中两个有[可选值](https://realpython.com/python-optional-arguments/)。另一方面，上面列出的第二个和第三个函数只需要两个位置参数，因为它们都使用预定义的`SafeDumper`。要找到可用的关键字参数，您必须查看`yaml.dump_all()`函数的签名。

您可以在所有四个转储函数中使用相同的关键字参数。它们都是可选的，因为它们的默认值等于`None`或`False`，除了`sort_keys`参数，其默认值为`True`。总共有六个布尔标志，您可以打开和关闭它们来更改生成的 YAML 的外观:

| 布尔标志 | 意义 |
| --- | --- |
| `allow_unicode` | 不要对 Unicode 进行转义，也不要使用双引号。 |
| `canonical` | 以标准形式输出 YAML。 |
| `default_flow_style` | 更喜欢流动风格而不是块状风格。 |
| `explicit_end` | 以三点(`...`)结束每个文档。 |
| `explicit_start` | 每个文档以三连破折号(`---`)开始。 |
| `sort_keys` | 按关键字对字典的输出进行排序。 |

其他数据类型也有几个参数可以给你更多的自由:

| 参数 | 类型 | 意义 |
| --- | --- | --- |
| `indent` | `int` | 块缩进级别，必须大于 1 小于 10 |
| `width` | `int` | 线宽，必须大于缩进的两倍 |
| `default_style` | `str` | 标量报价样式，必须是下列之一:`None`、`"'"`或`'"'` |
| `encoding` | `str` | 字符编码，设置时产生`bytes`而不是`str` |
| `line_break` | `str` | 换行符，必须是下列字符之一:`'\r'`、`'\n'`或`'\r\n'` |
| `tags` | `dict` | 由标记句柄组成的附加标记指令 |
| `version` | `tuple` | 主要和次要 YAML 版本，如`(1, 2)`为版本 1.2 |

大部分都是不言自明的。然而，`tags`参数必须是一个字典，它将定制的**标签句柄**映射到 YAML 解析器识别的有效的 [URI 前缀](https://en.wikipedia.org/wiki/Uniform_Resource_Identifier):

```py
{"!model!": "tag:yaml.org,2002:python/object:models."}
```

指定这样的映射会将相关的标记指令添加到转储的文档中。标签句柄总是以感叹号开始和结束。它们是完整标记名的简写符号。例如，在 YAML 文档中使用相同标签的所有等效方式如下:

```py
%TAG  !model!  tag:yaml.org,2002:python/object:models. --- -  !model!Person first_name:  John last_name:  Doe -  !!python/object:models.Person first_name:  John last_name:  Doe -  !<tag:yaml.org,2002:python/object:models.Person> first_name:  John last_name:  Doe
```

通过在 YAML 文档上面使用一个`%TAG`指令，您声明了一个名为`!model!`的定制标记句柄，它被扩展成下面的前缀。双感叹号(`!!`)是默认名称空间的内置快捷方式，对应于前缀`tag:yaml.org,2002:`。

您可以通过更改关键字参数的值并重新运行您的代码来查看结果，从而试验可用的关键字参数。然而，这听起来像是一个乏味的任务。本教程的辅助材料附带了一个交互式应用程序，可让您在 web 浏览器中测试参数及其值的不同组合:

[https://player.vimeo.com/video/673144581?background=1](https://player.vimeo.com/video/673144581?background=1)

这是一个动态网页，使用 [JavaScript](https://realpython.com/python-vs-javascript/) 通过网络与用 [FastAPI](https://realpython.com/fastapi-python-web-apis/) 编写的最小 HTTP 服务器通信。服务器需要一个 JSON 对象，除了关键字参数`tags`之外，其他参数都包含在内，并对下面的测试对象调用`yaml.dump()`:

```py
{
    "person": {
        "name_latin": "Ivan",
        "name": "Иван",
        "age": 42,
    }
}
```

上面的示例对象是一个包含整数和字符串字段的字典，其中包含 Unicode 字符。要运行服务器，您必须首先在您的虚拟环境中安装 FastAPI 库和一个 [ASGI](https://en.wikipedia.org/wiki/Asynchronous_Server_Gateway_Interface) web 服务器，比如[uvicon](https://pypi.org/project/uvicorn/)，您之前已经在那里安装了 PyYAML:

```py
(venv) $ python -m pip install fastapi uvicorn
(venv) $ uvicorn server:app
```

要运行服务器，必须提供模块名，后跟一个冒号和该模块中 ASGI 兼容的可调用程序的名称。实现这样一个服务器和一个客户机的细节远远超出了本教程的范围，但是您可以随意下载示例材料来自己学习:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-yaml-project-code/)在 Python 中使用 YAML。

接下来，您将了解更多关于使用 PyYAML 转储自定义类的内容。

[*Remove ads*](/account/join/)

### 转储自定义数据类型

正如您已经知道的，此时，您可以使用 PyYAML 提供的特定于 Python 的标记之一来序列化和反序列化您的自定义数据类型的对象，比如类。您还知道，这些标记只被不安全的加载程序和转储程序识别，这明确地允许潜在危险的代码执行。该库将拒绝序列化特定于 Python 的类型，如[复数](https://realpython.com/python-complex-numbers/)，除非您选择不安全的转储器类:

>>>

```py
>>> import yaml
>>> yaml.safe_dump(complex(3, 2))
Traceback (most recent call last):
  ...
yaml.representer.RepresenterError: ('cannot represent an object', (3+2j))

>>> yaml.dump(complex(3, 2))
"!!python/complex '3.0+2.0j'\n"
```

在第一种情况下，安全转储器不知道如何在 YAML 表示您的复数。另一方面，调用`yaml.dump()`在幕后隐式使用不安全的`Dump`类，这利用了`!!python/complex`标签。当您尝试转储自定义类时，情况类似:

>>>

```py
>>> class Person:
...     def __init__(self, first_name, last_name):
...         self.first_name = first_name
...         self.last_name = last_name
...
>>> yaml.safe_dump(Person("John", "Doe"))
Traceback (most recent call last):
  ...
yaml.representer.RepresenterError: ('cannot represent an object',
 <__main__.Person object at 0x7f55a671e8f0>)

>>> yaml.dump(Person("John", "Doe"))
!!python/object:__main__.Person
first_name: John
last_name: Doe
```

你唯一的选择就是不安全的`yaml.dump()`。然而，可以将您的类标记为可安全解析的，这样即使是安全的加载程序也能够在以后处理它们。为此，您必须对您的类进行一些更改:

>>>

```py
>>> class Person(yaml.YAMLObject): ...     yaml_tag = "!Person" ...     yaml_loader = yaml.SafeLoader ...     def __init__(self, first_name, last_name):
...         self.first_name = first_name
...         self.last_name = last_name
```

首先，让类从`yaml.YAMLObject`继承。然后指定两个类属性。一个属性将代表与您的类相关联的自定义 YAML 标记，而第二个属性将是要使用的加载器类。现在，当你把一个`Person`对象转储到 YAML 时，你将能够用`yaml.safe_load()`把它装载回来:

>>>

```py
>>> print(jdoe := yaml.dump(Person("John", "Doe")))
!Person
first_name: John
last_name: Doe

>>> yaml.safe_load(jdoe)
<__main__.Person object at 0x7f6fb7ba9ab0>
```

[Walrus 操作符(`:=` )](https://realpython.com/python-walrus-operator/) 允许您定义一个变量，并在一个步骤中将其用作`print()`函数的参数。将类标记为安全是一个很好的妥协，允许您通过忽略安全性并允许它们进入来对一些类进行例外处理。当然，在尝试加载关联的 YAML 之前，您必须绝对确定它们没有任何可疑之处。

## 底层解析 YAML 文档

到目前为止，您使用的类和一些包装函数构成了一个高级 PyYAML 接口，它隐藏了使用 YAML 文档的实现细节。这涵盖了大多数用例，并允许您将注意力集中在数据上，而不是数据的表示上。但是，有时您可能希望对解析和序列化过程有更多的控制。

在那些罕见的情况下，该库通过几个低级函数向您公开其内部工作方式。有四种方法可以阅读 YAML 流:

| 阅读功能 | 返回值 | 懒？ |
| --- | --- | --- |
| `yaml.scan()` | 代币 | ✔️ |
| `yaml.parse()` | 事件 | ✔️ |
| `yaml.compose()` | 结节 |  |
| `yaml.compose_all()` | 节点 | ✔️ |

所有这些函数都接受一个流和一个可选的 loader 类，默认为`yaml.Loader`。除此之外，它们中的大多数返回一个生成器对象，让你以一种**流的方式**处理 YAML，这在这一点上是不可能的。稍后您将了解令牌、事件和节点之间的区别。

还有一些对应的函数可以将 YAML 写到流中:

| 书写功能 | 投入 | 例子 |
| --- | --- | --- |
| `yaml.emit()` | 事件 | `yaml.emit(yaml.parse(data))` |
| `yaml.serialize()` | 结节 | `yaml.serialize(yaml.compose(data))` |
| `yaml.serialize_all()` | 节点 | `yaml.serialize_all(yaml.compose_all(data))` |

请注意，无论您选择什么功能，您都可能比以前有更多的工作要做。例如，处理 YAML 标签或将字符串值解释为正确的原生数据类型现在就在您的法庭上。但是，根据您的使用情况，其中一些步骤可能是不必要的。

在本节中，您将在 PyYAML 中实现这些低级函数的三个实际例子。请记住，您可以通过下面的链接下载它们的源代码:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-yaml-project-code/)在 Python 中使用 YAML。

[*Remove ads*](/account/join/)

### 标记一个 YAML 文档

通过扫描 YAML 文档获得一串[令牌](https://en.wikipedia.org/wiki/Lexical_analysis#Token)，您将获得最细粒度的控制。每个标记都有独特的含义，并告诉您它在哪里开始，在哪里结束，包括确切的行号和列号，以及从文档开始的偏移量:

>>>

```py
>>> import yaml
>>> for token in yaml.scan("Lorem ipsum", yaml.SafeLoader):
...     print(token)
...     print(token.start_mark)
...     print(token.end_mark)
...
StreamStartToken(encoding=None)
 in "<unicode string>", line 1, column 1:
 Lorem ipsum
 ^
 in "<unicode string>", line 1, column 1:
 Lorem ipsum
 ^
ScalarToken(plain=True, style=None, value='Lorem ipsum')
 in "<unicode string>", line 1, column 1:
 Lorem ipsum
 ^
 in "<unicode string>", line 1, column 12:
 Lorem ipsum
 ^
StreamEndToken()
 in "<unicode string>", line 1, column 12:
 Lorem ipsum
 ^
 in "<unicode string>", line 1, column 12:
 Lorem ipsum
 ^
```

令牌的`.start_mark`和`.end_mark`属性包含所有相关信息。例如，如果你想为你最喜欢的[代码编辑器](https://realpython.com/python-ides-code-editors-guide/)实现一个 YAML 语法荧光笔插件，这是完美的。事实上，您为什么不继续构建一个用于彩色打印 YAML 内容的基本命令行工具呢？

首先，您需要缩小标记类型的范围，因为您只对着色标量值、映射键和 YAML 标记感兴趣。创建一个名为`colorize.py`的新文件，并将以下函数放入其中:

```py
 1# colorize.py
 2
 3import yaml
 4
 5def tokenize(text, loader=yaml.SafeLoader):
 6    last_token = yaml.ValueToken(None, None)
 7    for token in yaml.scan(text, loader):
 8        start = token.start_mark.index
 9        end = token.end_mark.index
10        if isinstance(token, yaml.TagToken):
11            yield start, end, token
12        elif isinstance(token, yaml.ScalarToken):
13            yield start, end, last_token
14        elif isinstance(token, (yaml.KeyToken, yaml.ValueToken)):
15            last_token = token
```

它是 PyYAML 的`yaml.scan()`函数的一个瘦包装器，该函数生成包含起始索引、结束索引和一个令牌实例的元组。以下是更详细的分类:

*   第 6 行定义了一个变量来保存最后一个令牌实例。只有标量和标记令牌包含值，所以您必须记住它们的上下文，以便以后选择正确的颜色。当文档只包含一个标量而没有任何上下文时，使用初始值。
*   **第 7 行**在扫描的代币上循环。
*   **第 8 行和第 9 行**从所有标记上可用的索引标记中提取标记在文本中的位置。令牌的位置由`start`和`end`界定。
*   **第 10 到 13 行**检查当前的令牌类型，并产生索引和一个令牌实例。如果令牌是一个标签，那么它就会被放弃。如果令牌是标量，那么就会产生`last_token`,因为标量可以出现在不同的上下文中，您需要知道当前的上下文是什么才能选择适当的颜色。
*   **第 14 行和第 15 行**如果当前令牌是映射键或值，则更新上下文。其他标记类型被忽略，因为它们没有有意义的可视化表示。

当您将函数导入到交互式 Python 解释器会话中时，您应该能够开始迭代带有相关索引的标记子集:

>>>

```py
>>> from colorize import tokenize
>>> for token in tokenize("key: !!str value"):
...     print(token)
...
(0, 3, KeyToken())
(5, 10, TagToken(value=('!!', 'str')))
(11, 16, ValueToken())
```

整洁！您可以利用这些元组，使用第三方库或 [ANSI 转义序列](https://realpython.com/python-print/#adding-colors-with-ansi-escape-sequences)来注释原始文本中的标记，只要您的终端支持它们。以下是一些带有转义序列的颜色示例:

| 颜色 | 字体粗细 | 换码顺序 |
| --- | --- | --- |
| 蓝色 | 大胆的 | `ESC[34;1m` |
| 蓝绿色 | 规则的 | `ESC[36m` |
| 红色 | 规则的 | `ESC[31m` |

例如，键可能变成蓝色，值可能变成青色，YAML 标签可能变成红色。记住你不能在迭代的时候修改元素序列，因为那样会改变它们的索引。然而，您可以做的是从另一端开始迭代。这样，插入转义序列不会影响文本的其余部分。

现在返回代码编辑器，向 Python 源文件添加另一个函数:

```py
# colorize.py

import yaml

def colorize(text):
    colors = {
        yaml.KeyToken: lambda x: f"\033[34;1m{x}\033[0m",
        yaml.ValueToken: lambda x: f"\033[36m{x}\033[0m",
        yaml.TagToken: lambda x: f"\033[31m{x}\033[0m",
    }

    for start, end, token in reversed(list(tokenize(text))):
        color = colors.get(type(token), lambda text: text)
        text = text[:start] + color(text[start:end]) + text[end:]

    return text

# ...
```

这个新函数在 [reverse](https://realpython.com/python-reverse-list/) 中遍历一个标记化的文本，并在由`start`和`end`指示的地方插入转义码序列。请注意，这不是最有效的方法，因为由于切片和连接，您最终会制作大量文本副本。

拼图的最后一块是从[标准输入](https://en.wikipedia.org/wiki/Standard_streams#Standard_input_(stdin))中取出 YAML，并将其呈现到标准输出流中:

```py
# colorize.py

import sys import yaml

# ...

if __name__ == "__main__":
 print(colorize("".join(sys.stdin.readlines())))
```

从 Python 的标准库中导入`sys`模块，并将`sys.stdin`引用传递给刚刚创建的`colorize()`函数。现在，您可以在终端中运行您的脚本，并享受彩色编码的 YAML 代币:

[https://player.vimeo.com/video/673193047?background=1](https://player.vimeo.com/video/673193047?background=1)

请注意，`cat`命令在 Windows 上不可用。如果那是你的操作系统，那么使用它的 [`type`](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/type) 对应物，并确保通过[终端](https://en.wikipedia.org/wiki/Terminal_(Windows))应用程序运行命令，而不是通过[命令提示符(`cmd.exe` )](https://en.wikipedia.org/wiki/Cmd.exe) 或 [Windows PowerShell](https://en.wikipedia.org/wiki/PowerShell) 来默认启用 ANSI 转义码支持。

展开下面的可折叠部分，查看脚本的完整源代码:



```py
# colorize.py

import sys
import yaml

def colorize(text):
    colors = {
        yaml.KeyToken: lambda x: f"\033[34;1m{x}\033[0m",
        yaml.ValueToken: lambda x: f"\033[36m{x}\033[0m",
        yaml.TagToken: lambda x: f"\033[31m{x}\033[0m",
    }

    for start, end, token in reversed(list(tokenize(text))):
        color = colors.get(type(token), lambda text: text)
        text = text[:start] + color(text[start:end]) + text[end:]

    return text

def tokenize(text, loader=yaml.SafeLoader):
    last_token = yaml.ValueToken(None, None)
    for token in yaml.scan(text, loader):
        start = token.start_mark.index
        end = token.end_mark.index
        if isinstance(token, yaml.TagToken):
            yield start, end, token
        elif isinstance(token, yaml.ScalarToken):
            yield start, end, last_token
        elif isinstance(token, (yaml.KeyToken, yaml.ValueToken)):
            last_token = token

if __name__ == "__main__":
    print(colorize("".join(sys.stdin.readlines())))
```

标记化对于实现语法高亮器非常有用，它必须能够引用源 YAML 文件中的符号。但是，对于其他不关心输入数据的确切布局的应用程序来说，这可能有点太低级了。接下来，您将了解处理 YAML 的另一种方法，这也涉及到流。

### 解析事件流

PyYAML 提供的另一个底层接口是一个**事件驱动的流** API，它的工作方式类似于 XML 中的 SAX。它将 YAML 转化为由单个元素触发的事件的平面序列。事件被延迟评估，而不需要将整个文档加载到内存中。你可以把它想象成透过一扇移动的窗户偷窥。

这有助于绕过试图读取大文件时可能面临的内存限制。它还可以大大加快在噪音海洋中搜索特定信息的速度。除此之外，流式传输还可以为您的数据逐步构建另一种表示方式。在本节中，您将创建一个 HTML 构建器，以一种粗略的方式可视化 YAML。

当您使用 PyYAML 解析文档时，该库会产生一系列事件:

>>>

```py
>>> import yaml
>>> for event in yaml.parse("[42, {pi: 3.14, e: 2.72}]", yaml.SafeLoader):
...     print(event)
...
StreamStartEvent()
DocumentStartEvent()
SequenceStartEvent(anchor=None, tag=None, implicit=True)
ScalarEvent(anchor=None, tag=None, implicit=(True, False), value='42')
MappingStartEvent(anchor=None, tag=None, implicit=True)
ScalarEvent(anchor=None, tag=None, implicit=(True, False), value='pi')
ScalarEvent(anchor=None, tag=None, implicit=(True, False), value='3.14')
ScalarEvent(anchor=None, tag=None, implicit=(True, False), value='e')
ScalarEvent(anchor=None, tag=None, implicit=(True, False), value='2.72')
MappingEndEvent()
SequenceEndEvent()
DocumentEndEvent()
StreamEndEvent()
```

如您所见，有各种类型的事件对应于 YAML 文档中的不同元素。其中一些事件公开了额外的属性，您可以检查这些属性以了解更多关于当前元素的信息。

您可以想象这些事件如何自然地转化为 HTML 等分层标记语言中的开始和结束标记。例如，您可以用下面的标记片段表示上面的结构:

```py
<ul>
  <li>42</li>
  <li>
    <dl>
      <dt>pi</dt>
      <dd>3.14</dd>
      <dt>e</dt>
      <dd>2.72</dd>
    </dl>
  </li>
</ul>
```

单个列表项被包装在`<li>`和`</li>`标签之间，而键值映射利用了[描述列表(`<dl>` )](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/dl) ，它包含交替的[术语(`<dt>` )](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/dt) 和[定义(`<dd>` )](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/dd) 。这是一个棘手的部分，因为它需要在给定的嵌套层次上计算后续的 YAML 事件，以确定一个事件是否应该成为 HTML 中的一个术语或定义。

最终，您希望设计一个`HTMLBuilder`类来帮助您以一种懒惰的方式解析来自流的多个 YAML 文档。假设您已经定义了这样一个类，那么您可以在一个名为`yaml2html.py`的文件中创建下面的帮助函数:

```py
# yaml2html.py

import yaml

# ...

def yaml2html(stream, loader=yaml.SafeLoader):
    builder = HTMLBuilder()
    for event in yaml.parse(stream, loader):
        builder.process(event)
        if isinstance(event, yaml.DocumentEndEvent):
            yield builder.html
            builder = HTMLBuilder()
```

代码在一系列解析器事件上循环，并将它们交给您的类，该类通过增量构建其表示将 YAML 转换为 HTML。一旦该函数检测到流中 YAML 文档的结尾，它就会产生一个 HTML 片段，并创建一个新的空构建器来重新开始。这避免了在处理可能无限长的 YAML 文档流的过程中发生阻塞，这些文档可能通过网络到达:

>>>

```py
>>> from yaml2html import yaml2html
>>> for document in yaml2html("""
... ---
... title: "Document #1"
... ---
... title: "Document #2"
... ---
... title: "Document #3"
... """):
...     print(document)
...
<dl><dt>title</dt><dd>Document #1</dd></dl>
<dl><dt>title</dt><dd>Document #2</dd></dl>
<dl><dt>title</dt><dd>Document #3</dd></dl>
```

上面的例子演示了一个由三个 YAML 文档组成的流，helper 函数将这些文档转换成单独的 HTML 片段。现在您已经理解了预期的行为，是时候实现`HTMLBuilder`类了。

构建器类中的初始化器方法将定义两个私有字段来跟踪当前的上下文和到目前为止构建的 HTML 内容:

```py
# yaml2html.py

import yaml

class HTMLBuilder:
    def __init__(self):
        self._context = []
        self._html = []

    @property
    def html(self):
        return "".join(self._html)

# ...
```

上下文是一个作为 Python 列表实现的[栈](https://realpython.com/how-to-implement-python-stack/)，它存储了到目前为止处理的给定级别上的键值对的数量。堆栈还可以包含指示`SequenceStartEvent`和`SequenceEndEvent`之间状态的列表标记。另一个字段是 HTML 标签及其内容的列表，由一个公共类属性连接。

有几个 YAML 事件你会想要处理:

```py
 1# yaml2html.py
 2
 3import yaml
 4
 5from yaml import (
 6    ScalarEvent,
 7    SequenceStartEvent,
 8    SequenceEndEvent,
 9    MappingStartEvent,
10    MappingEndEvent,
11)
12
13OPEN_TAG_EVENTS = (ScalarEvent, SequenceStartEvent, MappingStartEvent)
14CLOSE_TAG_EVENTS = (ScalarEvent, SequenceEndEvent, MappingEndEvent)
15
16class HTMLBuilder:
17    # ...
18
19    def process(self, event):
20
21        if isinstance(event, OPEN_TAG_EVENTS):
22            self._handle_tag()
23
24        if isinstance(event, ScalarEvent):
25            self._html.append(event.value)
26        elif isinstance(event, SequenceStartEvent):
27            self._html.append("<ul>")
28            self._context.append(list)
29        elif isinstance(event, SequenceEndEvent):
30            self._html.append("</ul>")
31            self._context.pop()
32        elif isinstance(event, MappingStartEvent):
33            self._html.append("<dl>")
34            self._context.append(0)
35        elif isinstance(event, MappingEndEvent):
36            self._html.append("</dl>")
37            self._context.pop()
38
39        if isinstance(event, CLOSE_TAG_EVENTS):
40            self._handle_tag(close=True)
41# ...
```

通过检查堆栈上是否有任何打开的标记等待某个操作来开始处理一个事件。您将该检查委托给另一个助手方法`._handle_tag()`，稍后您将添加该方法。然后，添加对应于当前事件的 HTML 标记，并再次更新上下文。

下面是上面片段的快速逐行摘要:

*   **第 5 行到第 11 行**从 PyYAML 导入所需的事件类型。
*   **第 13 行和第 14 行**指定了对应于 HTML 开始和结束标签的事件类型。
*   **第 24 行到第 37 行**附加相应的 HTML 标签，并根据需要更新堆栈。
*   **第 21、22、39 和 40 行**打开或关闭堆栈上的挂起标签，并可选地更新被处理的键值对的数量。

缺少的部分是在必要时负责打开和关闭匹配标记的 helper 方法:

```py
# yaml2html.py

import yaml

# ...

class HTMLBuilder:
    # ...

    def _handle_tag(self, close=False):
        if len(self._context) > 0:
            if self._context[-1] is list:
                self._html.append("</li>" if close else "<li>")
            else:
                if self._context[-1] % 2 == 0:
                    self._html.append("</dt>" if close else "<dt>")
                else:
                    self._html.append("</dd>" if close else "<dd>")
                if close:
                    self._context[-1] += 1
# ...
```

如果堆栈上已经有东西了，那么你就检查最后一个被放进去的东西。如果是列表，则打开或关闭列表项。否则，根据键-值映射的数量的[奇偶性，是时候打开或关闭描述列表中的术语或定义了。](https://en.wikipedia.org/wiki/Parity_(mathematics))

您可以通过在底部添加 [`if __name__`习语](https://docs.python.org/3/library/__main__.html#idiomatic-usage)来将您的 Python 模块转换成可执行脚本:

```py
# yaml2html.py

import sys

# ...

if __name__ == "__main__":
    print("".join(yaml2html("".join(sys.stdin.readlines()))))
```

当您将 HTML 输出通过管道传输到基于文本的网络浏览器(如 T2 的 Lynx 或 T4 的 html2text 转换器)时，它可以让您在终端上预览 YAML 的视觉表现:

```py
$ echo '[42, {pi: 3.14, e: 2.72}]' | python yaml2html.py | html2text
 * 42
 *   pi
 3.14
 e
 2.72
```

命令应该可以在所有主要的操作系统上运行。它在终端中打印一段文本，您可以使用竖线字符(`|`)将这段文本连接到另一个命令[管道](https://en.wikipedia.org/wiki/Pipeline_(Unix))。在这种情况下，您用您的`yaml2html.py`脚本处理一个简短的 YAML 文档，然后将生成的 HTML 转换成简化的文本形式，您可以在终端中预览，而无需启动一个成熟的 web 浏览器。

单击下面的可折叠部分以显示完整的源代码:



```py
# yaml2html.py

import sys
import yaml

from yaml import (
    ScalarEvent,
    SequenceStartEvent,
    SequenceEndEvent,
    MappingStartEvent,
    MappingEndEvent,
)

OPEN_TAG_EVENTS = (ScalarEvent, SequenceStartEvent, MappingStartEvent)
CLOSE_TAG_EVENTS = (ScalarEvent, SequenceEndEvent, MappingEndEvent)

class HTMLBuilder:
    def __init__(self):
        self._context = []
        self._html = []

    @property
    def html(self):
        return "".join(self._html)

    def process(self, event):

        if isinstance(event, OPEN_TAG_EVENTS):
            self._handle_tag()

        if isinstance(event, ScalarEvent):
            self._html.append(event.value)
        elif isinstance(event, SequenceStartEvent):
            self._html.append("<ul>")
            self._context.append(list)
        elif isinstance(event, SequenceEndEvent):
            self._html.append("</ul>")
            self._context.pop()
        elif isinstance(event, MappingStartEvent):
            self._html.append("<dl>")
            self._context.append(0)
        elif isinstance(event, MappingEndEvent):
            self._html.append("</dl>")
            self._context.pop()

        if isinstance(event, CLOSE_TAG_EVENTS):
            self._handle_tag(close=True)

    def _handle_tag(self, close=False):
        if len(self._context) > 0:
            if self._context[-1] is list:
                self._html.append("</li>" if close else "<li>")
            else:
                if self._context[-1] % 2 == 0:
                    self._html.append("</dt>" if close else "<dt>")
                else:
                    self._html.append("</dd>" if close else "<dd>")
                if close:
                    self._context[-1] += 1

def yaml2html(stream, loader=yaml.SafeLoader):
    builder = HTMLBuilder()
    for event in yaml.parse(stream, loader):
        builder.process(event)
        if isinstance(event, yaml.DocumentEndEvent):
            yield builder.html
            builder = HTMLBuilder()

if __name__ == "__main__":
    print("".join(yaml2html("".join(sys.stdin.readlines()))))
```

干得好！你现在可以在你的网页浏览器中看到 YAML 了。然而，呈现是静态的。用一点点互动来增加趣味不是很好吗？接下来，您将使用不同的方法来解析 YAML，这将允许！

### 构建节点树

有时候，你确实需要将整个文档保存在内存中，以便向前看，并根据接下来的内容做出明智的决定。PyYAML 可以构建 YAML 元素层次结构的对象表示，类似于 XML 中的 DOM。通过调用`yaml.compose()`，您将获得一个元素树的根节点:

>>>

```py
>>> import yaml
>>> root = yaml.compose("[42, {pi: 3.14, e: 2.72}]", yaml.SafeLoader)
>>> root
SequenceNode(
 tag='tag:yaml.org,2002:seq',
 value=[
 ScalarNode(tag='tag:yaml.org,2002:int', value='42'),
 MappingNode(
 tag='tag:yaml.org,2002:map',
 value=[
 (
 ScalarNode(tag='tag:yaml.org,2002:str', value='pi'),
 ScalarNode(tag='tag:yaml.org,2002:float', value='3.14')
 ),
 (
 ScalarNode(tag='tag:yaml.org,2002:str', value='e'),
 ScalarNode(tag='tag:yaml.org,2002:float', value='2.72')
 )
 ]
 )
 ]
)
```

根可通过方括号语法进行遍历。您可以使用 node 的`.value`属性和下标访问树中的任何后代元素:

>>>

```py
>>> key, value = root.value[1].value[0]

>>> key
ScalarNode(tag='tag:yaml.org,2002:str', value='pi')

>>> value
ScalarNode(tag='tag:yaml.org,2002:float', value='3.14')
```

因为只有三种节点(`ScalarNode`、`SequenceNode`和`MappingNode`)，所以可以用递归函数自动遍历它们:

```py
# tree.py

import yaml

def visit(node):
    if isinstance(node, yaml.ScalarNode):
        return node.value
    elif isinstance(node, yaml.SequenceNode):
        return [visit(child) for child in node.value]
    elif isinstance(node, yaml.MappingNode):
        return {visit(key): visit(value) for key, value in node.value}
```

将这个函数放在名为`tree.py`的 Python 脚本中，因为您将开发代码。该函数采用单个节点，并根据其类型返回其值或进入相关的子树。注意，映射键也必须被访问，因为它们在 YAML 中可以是非标量值。

然后，在交互式 Python 解释器会话中导入您的函数，并针对您之前创建的根元素进行测试:

>>>

```py
>>> from tree import visit
>>> visit(root)
['42', {'pi': '3.14', 'e': '2.72'}]
```

结果得到一个 Python 列表，但是其中包含的单个标量值都是字符串。PyYAML 检测与标量值相关联的数据类型，并将其存储在节点的`.tag`属性中，但是您必须自己进行类型转换。这些类型使用 **YAML 全局标签**进行编码，比如`"tag:yaml.org,2002:float"`，所以您可以提取第二个冒号(`:`之后的最后一位。

通过调用新的`cast()`函数包装标量的返回值来修改函数:

```py
# tree.py

import base64 import datetime import yaml

def visit(node):
    if isinstance(node, yaml.ScalarNode):
 return cast(node.value, node.tag)    elif isinstance(node, yaml.SequenceNode):
        return [visit(child) for child in node.value]
    elif isinstance(node, yaml.MappingNode):
        return {visit(key): visit(value) for key, value in node.value}

def cast(value, tag):
    match tag.split(":")[-1]:
        case "null":
            return None
        case "bool":
            return bool(value)
        case "int":
            return int(value)
        case "float":
            return float(value)
        case "timestamp":
            return datetime.datetime.fromisoformat(value)
        case "binary":
            return base64.decodebytes(value.encode("utf-8"))
        case _:
            return str(value)
```

您可以利用 Python 3.10 中引入的新的`match`和`case`关键字以及[结构模式匹配](https://realpython.com/python310-new-features/#structural-pattern-matching)语法，或者您可以使用普通的旧`if`语句重写这个示例。底线是，当您在交互式解释器会话中[重新加载模块](https://docs.python.org/3/library/importlib.html#importlib.reload)时，您现在应该获得原生 Python 类型的值:

>>>

```py
>>> import importlib, tree
>>> importlib.reload(tree)
<module 'tree' from '/home/realpython/tree.py'>

>>> visit(root)
[42, {'pi': 3.14, 'e': 2.72}]

>>> visit(yaml.compose("when: 2022-01-16 23:59:59"))
{'when': datetime.datetime(2022, 1, 16, 23, 59, 59)}
```

您已经准备好生成 HTML 字符串，而不是 Python 对象。将`visit()`中的返回值替换为对更多助手函数的调用:

```py
# tree.py

import base64
import datetime
import yaml

def visit(node):
    if isinstance(node, yaml.ScalarNode):
        return cast(node.value, node.tag)
    elif isinstance(node, yaml.SequenceNode):
 return html_list(node)    elif isinstance(node, yaml.MappingNode):
 return html_map(node) 
# ...

def html_list(node):
    items = "".join(f"<li>{visit(child)}</li>" for child in node.value)
    return f'<ul class="sequence">{items}</ul>'

def html_map(node):
    pairs = "".join(
        f'<li><span class="key">{visit(key)}:</span> {visit(value)}</li>'
        if isinstance(value, yaml.ScalarNode) else (
            "<li>"
            "<details>"
            f'<summary class="key">{visit(key)}</summary> {visit(value)}'
            "</details>"
            "</li>"
        )
        for key, value in node.value
    )
    return f"<ul>{pairs}</ul>"
```

两个助手函数都接受一个节点实例并返回一段 HTML 字符串。`html_list()`函数期望用[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)迭代`SequenceNode`，而`html_map()`迭代`MappingNode`的键和值。这就是提前了解整个树结构的好处。如果映射值是一个`ScalarNode`，那么用一个`<span>`元素替换它。其他节点类型被包装在一个可折叠的 [`<details>`](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/details) 标签中。

因为您将生成 HTML 输出，所以您可以通过只返回普通字符串来简化类型转换函数。同时，您可以为 Base64 编码的数据返回一个 HTML `<img>`元素，并显示该元素，而不是显示原始字节。除此之外，常规标量可以包装在一个`<span>`或一个适当样式的`<div>`元素中，这取决于它们是包含单行内容还是多行内容:

```py
# tree.py

import yaml

# ...

def cast(value, tag):
    match tag.split(":")[-1]:
        case "binary":
            return f'<img src="data:image/png;base64, {value}" />'
        case _:
            if "\n" in value:
                return f'<div class="multiline">{value}</div>'
            else:
                return f"<span>{value}</span>"
```

HTML `<img>`元素的`src`属性识别编码数据。请注意，您不再需要`base64`或`datetime`导入，所以继续操作并从文件顶部删除它们。

和往常一样，您希望通过从标准输入中读取内容来使您的脚本可运行。您还可以在一个新的`html_tree()`函数中用一些样板文件包装生成的 HTML 主体:

```py
# tree.py

import sys import yaml

def html_tree(stream, loader=yaml.SafeLoader):
    body = visit(yaml.compose(stream, loader))
    return (
        "<!DOCTYPE html>"
        "<html>"
        "<head>"
        "  <meta charset=\"utf-8\">"
        "  <title>YAML Tree Preview</title>"
        "  <link href=\"https://fonts.googleapis.com/css2"
        "?family=Roboto+Condensed&display=swap\" rel=\"stylesheet\">"
        "  <style>"
        "    * { font-family: 'Roboto Condensed', sans-serif; }"
        "    ul { list-style: none; }"
        "    ul.sequence { list-style: '- '; }"
        "    .key { font-weight: bold; }"
        "    .multiline { white-space: pre; }"
        "  </style>"
        "</head>"
        f"<body>{body}</body></html>"
    )

# ...

if __name__ == "__main__":
 print(html_tree("".join(sys.stdin.readlines())))
```

这个 HTML 使用了一个嵌入的谷歌字体，看起来更舒服。内联 [CSS 样式](https://en.wikipedia.org/wiki/CSS)从常规无序列表中移除项目符号，因为您使用项目符号进行键值映射。但是，显式标记为序列的列表在每个项目前使用破折号。映射键以粗体显示，多行字符串保留空白。

当您针对一些测试数据运行脚本时，它会输出一段 HTML 代码，您可以将这段代码重定向到一个本地文件，您可以使用默认的 web 浏览器打开该文件:

*   [*视窗*](#windows-1)
**   [*Linux*](#linux-1)**   [*macOS*](#macos-1)**

```py
C:\> type data.yaml | python tree.py > index.html
C:\> start index.html
```

```py
$ cat data.yaml | python tree.py > index.html
$ xdg-open index.html
```

```py
$ cat data.yaml | python tree.py > index.html
$ open index.html
```

在 web 浏览器中预览时，生成的页面将允许您交互式地展开和折叠各个键-值对:

[https://player.vimeo.com/video/691778178?background=1](https://player.vimeo.com/video/691778178?background=1)

<figcaption class="figure-caption text-center">Interactive HTML Tree of YAML Nodes</figcaption>

请注意 web 浏览器如何呈现描绘笑脸的 Base64 编码图像。您将在下面的可折叠部分找到最终代码:



```py
# tree.py

import sys
import yaml

def html_tree(stream, loader=yaml.SafeLoader):
    body = visit(yaml.compose(stream, loader))
    return (
        "<!DOCTYPE html>"
        "<html>"
        "<head>"
        "  <meta charset=\"utf-8\">"
        "  <title>YAML Tree Preview</title>"
        "  <link href=\"https://fonts.googleapis.com/css2"
        "?family=Roboto+Condensed&display=swap\" rel=\"stylesheet\">"
        "  <style>"
        "    * { font-family: 'Roboto Condensed', sans-serif; }"
        "    ul { list-style: none; }"
        "    ul.sequence { list-style: '- '; }"
        "    .key { font-weight: bold; }"
        "    .multiline { white-space: pre; }"
        "  </style>"
        "</head>"
        f"<body>{body}</body></html>"
    )

def visit(node):
    if isinstance(node, yaml.ScalarNode):
        return cast(node.value, node.tag)
    elif isinstance(node, yaml.SequenceNode):
        return html_list(node)
    elif isinstance(node, yaml.MappingNode):
        return html_map(node)

def cast(value, tag):
    match tag.split(":")[-1]:
        case "binary":
            return f'<img src="data:image/png;base64, {value}" />'
        case _:
            if "\n" in value:
                return f'<div class="multiline">{value}</div>'
            else:
                return f"<span>{value}</span>"

def html_list(node):
    items = "".join(f"<li>{visit(child)}</li>" for child in node.value)
    return f'<ul class="sequence">{items}</ul>'

def html_map(node):
    pairs = "".join(
        f'<li><span class="key">{visit(key)}:</span> {visit(value)}</li>'
        if isinstance(value, yaml.ScalarNode) else (
            "<li>"
            "<details>"
            f'<summary class="key">{visit(key)}</summary> {visit(value)}'
            "</details>"
            "</li>"
        )
        for key, value in node.value
    )
    return f"<ul>{pairs}</ul>"

if __name__ == "__main__":
    print(html_tree("".join(sys.stdin.readlines())))
```

好了，这就是使用 PyYAML 库在底层解析 YAML 文档的全部内容。相应的`yaml.emit()`和`yaml.serialize()`函数以相反的方式工作，分别获取一系列事件或根节点，并将它们转换成 YAML 表示。但是你很少需要使用它们。

## 结论

您现在知道在哪里可以找到 Python 中读写 YAML 文档所缺少的电池了。您已经创建了一个 YAML 语法荧光笔和一个 HTML 格式的交互式 YAML 预览。在这个过程中，您了解了这种流行的数据格式中强大而危险的特性，以及如何在 Python 中利用它们。

**在本教程中，您学习了如何:**

*   **用 Python 读**和**写** YAML 文档
*   **序列化** Python 的**内置的**和**自定义的**数据类型到 YAML
*   **安全**读取来自**不可信来源的 YAML 文件**
*   控制**解析下级的 YAML** 文档

要获得本教程中示例的源代码，请访问下面的链接:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/python-yaml-project-code/)在 Python 中使用 YAML。*************