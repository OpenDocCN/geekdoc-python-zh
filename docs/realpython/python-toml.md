# Python 和 TOML:新的好朋友

> 原文:# t0]https://realython . com/python-toml/

TOML——Tom 显而易见的最小语言——是一种相当新的配置文件格式，Python 社区在过去几年里已经接受了这种格式。TOML 在 Python 生态系统中扮演着重要的角色。许多您喜欢的工具依赖于 TOML 进行配置，当您构建和发布自己的包时，您将使用`pyproject.toml`。

在本教程中，你会学到更多关于 TOML 的知识以及如何使用它。特别是，您将:

*   学习并理解 TOML 的**语法**
*   使用`tomli`和`tomllib`来**解析** TOML 文档
*   使用`tomli_w`到**将**数据结构写成 TOML
*   当你需要对你的 TOML 文件有更多的控制时，使用`tomlkit`

在 Python 3.11 中，一个新的 TOML 解析模块被添加到 Python 的标准库中。[稍后](#read-toml-documents-with-tomli-and-tomllib)在本教程中，你将学习如何使用这个新模块。如果你想了解更多关于`tomllib`为什么被加入 Python 的信息，那么看看配套教程， [Python 3.11 预览:TOML 和`tomllib`T5。](https://realpython.com/python311-tomllib/)

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 使用 TOML 作为配置格式

TOML 是**汤姆的显而易见的最小语言**的缩写，并以它的创造者[汤姆·普雷斯顿-沃纳](https://tom.preston-werner.com/)的名字谦逊地命名。它被明确设计成一种**配置文件格式**，应该“易于解析成各种语言的数据结构”([来源](https://toml.io/en/v0.1.0))。

在这一节中，您将开始考虑配置文件，并看看 TOML 带来了什么。

[*Remove ads*](/account/join/)

### 配置和配置文件

配置几乎是任何应用程序或系统的重要组成部分。它允许你在不改变源代码的情况下改变设置或行为。有时，您将使用配置来指定连接到另一个服务(如数据库或云存储)所需的信息。其他时候，您将使用配置设置来允许用户自定他们对项目的体验。

为您的项目使用一个[配置文件](https://en.wikipedia.org/wiki/Configuration_file)是将您的代码与其设置分开的一个好方法。它还鼓励您意识到系统的哪些部分是真正可配置的，为您提供了一个在源代码中命名神奇值的工具。现在，考虑一个假想的[井字游戏](https://realpython.com/tic-tac-toe-python/)的配置文件:

```py
player_x_color  =  blue player_o_color  =  green board_size  =  3 server_url  =  https://tictactoe.example.com/
```

您可以直接在源代码中编写这种代码。但是，通过将设置移动到一个单独的文件中，您可以实现一些目标:

*   你给值一个明确的名字。
*   你提供这些值更多的**可见性**。
*   你使得**改变**这些值变得更简单。

更仔细地看看您假设的配置文件。这些值在概念上是不同的。颜色是您的框架可能支持更改的值。换句话说，如果您将`blue`替换为`red`，那么在代码中不会有任何特殊处理。您甚至可以考虑是否值得通过您的前端向您的最终用户公开此配置。

然而，电路板尺寸可以配置，也可以不配置。井字游戏是在一个三乘三的格子上玩的。不确定你的逻辑是否还适用于其他尺寸的电路板。将该值保存在配置文件中仍然是有意义的，这既是为了给该值命名，也是为了使其可见。

最后，在部署应用程序时，项目 URL 通常是必不可少的。这不是一个普通用户会改变的事情，但是一个超级用户可能想把你的游戏重新部署到一个不同的服务器上。

为了更清楚地了解这些不同的用例，您可能希望在您的配置中添加一些组织。一种流行的选择是将您的配置分成附加文件，每个文件处理不同的问题。另一个选择是以某种方式对配置值进行分组。例如，您可以按如下方式组织假设的配置文件:

```py
[user] player_x_color  =  blue player_o_color  =  green [constant] board_size  =  3 [server] url  =  https://tictactoe.example.com
```

文件的组织使得每个配置项的角色更加清晰。您还可以向配置文件添加注释，并向任何想对其进行更改的人提供说明。

**注意:**配置文件的实际格式对于这个讨论并不重要。上述原则与您如何指定配置值无关。碰巧的是，到目前为止你看到的例子都可以被 Python 的 [`ConfigParser`](https://docs.python.org/3/library/configparser.html) 类解析。

您可以通过多种方式来指定配置。Windows 传统上使用 [INI 文件](https://en.wikipedia.org/wiki/INI_file)，它类似于上面的配置文件。Unix 系统也依赖于纯文本、人类可读的[配置文件](https://en.wikipedia.org/wiki/Configuration_file#Unix_and_Unix-like_operating_systems)，尽管不同服务之间的实际格式有所不同。

随着时间的推移，越来越多的应用程序开始使用定义良好的格式，如 [XML](https://realpython.com/python-xml-parser/) 、 [JSON](https://realpython.com/python-json/) 或 [YAML](https://realpython.com/python-yaml/) 来满足它们的配置需求。这些格式被设计成数据**交换**或**串行化**格式，通常用于计算机通信。

另一方面，配置文件通常是由人编写或编辑的。许多开发人员在更新他们的 [Visual Studio 代码设置](https://realpython.com/advanced-visual-studio-code-python/#setting-up-your-terminal)时对 JSON 严格的逗号规则感到失望，或者在建立[云服务](https://realpython.com/python-boto3-aws-s3/)时对 YAML 的嵌套缩进感到失望。尽管它们无处不在，但这些文件格式并不是最容易手写的。

### 汤姆:汤姆明显的最小语言

TOML 是一种相当新的格式。第一个格式规范版本 0.1.0 于 2013 年发布。从一开始，它就专注于成为人类可读的最小配置文件格式。根据 TOML 的网页，TOML 的目标如下:

> TOML 的目标是成为一种最小化的配置文件格式，由于明显的语义，这种格式**易于阅读**。TOML 被设计成**明确地将**映射到散列表。TOML 应该**容易解析**成各种语言的数据结构。([来源](https://toml.io/en/)，重点添加)

当您阅读本教程时，您将会看到 TOML 是如何达到这些目标的。不过，很明显，TOML 在其短暂的生命周期中变得非常流行。越来越多的 Python 工具，包括 [Black](https://black.readthedocs.io/) 、 [pytest](https://docs.pytest.org/) 、 [mypy](https://mypy.readthedocs.io/) 和 [isort](https://black.readthedocs.io/) ，都使用 TOML 进行配置。对于大多数流行的编程语言来说，TOML 解析器是可用的。

回忆一下上一小节中的配置。用 TOML 表达它的一种方法如下:

```py
[user] player_x.color  =  "blue" player_o.color  =  "green" [constant] board_size  =  3 [server] url  =  "https://tictactoe.example.com"
```

在下一节的[中，您将了解更多关于 TOML 格式的细节。现在，试着自己阅读和解析这些信息。注意，和早前没太大区别。最大的变化是在一些值中添加了引号(`"`)。](#get-to-know-toml-key-value-pairs)

TOML 的语法受到传统配置文件的启发。与 Windows INI 文件和 Unix 配置文件相比，它的一个主要优势是 TOML 有一个**规范**，它精确地说明了 TOML 文档中允许的内容以及不同的值应该如何解释。[规范](https://toml.io/en/v1.0.0)在 2021 年初达到 1.0.0 版本后稳定成熟。

相比之下，INI 格式没有正式的规范。相反，有许多变体和方言，其中大部分是由一个实现定义的。Python 附带了对标准库中读取 INI 文件的[支持。虽然`ConfigParser`相当宽松，但它并不支持所有类型的 INI 文件。](https://realpython.com/build-a-python-weather-app-cli/#access-your-api-key-in-your-python-code)

TOML 和许多传统格式的另一个区别是 TOML 值有类型。在上面的例子中，`"blue"`被解释为一个字符串，而`3`是一个数字。对 TOML 的一个潜在的批评是，编写 TOML 的人需要知道类型。在更简单的格式中，这个责任在于程序员解析配置。

TOML 不是像 JSON 或 YAML 那样的数据序列化格式。换句话说，您不应该试图将一般数据存储在 TOML 中以便以后恢复。TOML 在几个方面有限制:

*   所有键都被解释为字符串。你不能轻易使用，比如说，一个数字作为密钥。
*   TOML 没有空类型。
*   一些空白很重要，这会降低压缩 TOML 文档大小的效率。

即使 TOML 是一把好锤子，但并不是所有的数据文件都是钉子。您应该主要使用 TOML 进行配置。

[*Remove ads*](/account/join/)

### TOML 模式验证

在下一节中，您将更深入地研究 TOML 语法。在那里，您将了解一些 TOML 文件的语法要求。然而，在实践中，给定的 TOML 文件也可能带有一些非语法要求。

这些是**模式需求**。例如，您的井字游戏应用程序可能要求配置文件包含服务器 URL。另一方面，播放器颜色可以是可选的，因为应用程序定义了默认颜色。

目前，TOML 不包括一种可以在 TOML 文档中指定必填和可选字段的模式语言。有几个提议存在，尽管还不清楚它们中的任何一个是否会很快被接受。

在简单的应用程序中，您可以手动验证 TOML 配置。比如可以使用[结构模式匹配](https://realpython.com/python310-new-features/#structural-pattern-matching)，这是在 [Python 3.10](https://realpython.com/python310-new-features/) 中引入的。假设您已经将配置解析成 Python，并将其命名为`config`。然后，您可以按如下方式检查其结构:

```py
match config:
    case {
 "user": {"player_x": {"color": str()}, "player_o": {"color": str()}}, "constant": {"board_size": int()}, "server": {"url": str()},    }:
        pass
    case _:
        raise ValueError(f"invalid configuration: {config}")
```

第一个`case`语句详细说明了您期望的结构。如果`config`匹配，那么你使用 [`pass`](https://realpython.com/python-pass/) 继续你的代码。否则，您会引发一个错误。

如果您的 TOML 文档更复杂，这种方法可能不太适用。如果想提供好的错误消息，还需要做更多的工作。更好的替代方法是使用 [pydantic](https://pydantic-docs.helpmanual.io/) ，它利用[类型注释](https://realpython.com/python-type-checking/)在运行时进行数据验证。pydantic 的一个优点是它内置了精确而有用的错误消息。

还有一些工具可以利用针对 JSON 等格式的现有模式验证。例如， [Taplo](https://taplo.tamasfe.dev/) 是一个 TOML 工具包，可以根据 JSON 模式验证 TOML 文档。Taplo 也可用于 [Visual Studio 代码](https://realpython.com/advanced-visual-studio-code-python/)，捆绑到[更好的 TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml) 扩展中。

在本教程的其余部分中，您不必担心模式验证。相反，您将更加熟悉 TOML 语法，并了解您可以使用的所有不同的数据类型。稍后，您将看到如何在 Python 中与 TOML 交互的示例，并且您将探索 TOML 非常适合的一些用例。

## 了解 TOML:键值对

TOML 是围绕键值对构建的，这些键值对可以很好地映射到[哈希表](https://realpython.com/python-hash-table/)数据结构。TOML 值有不同的类型。每个值必须具有以下类型之一:

*   [字符串](https://realpython.com/python-strings/)
*   [整数](https://realpython.com/python-numbers/#integers)
*   [浮动](https://realpython.com/python-numbers/#floating-point-numbers)
*   [布尔型](https://realpython.com/python-boolean/)
*   偏移日期时间
*   当地[日期时间](https://realpython.com/python-datetime/)
*   当地日期
*   当地[时间](https://realpython.com/python-time-module/)
*   排列
*   内嵌表格

此外，您可以使用**表**和**表数组**作为组织几个键值对的[集合](https://realpython.com/python-collections-module/)。在本节的剩余部分，您将了解到更多关于所有这些的内容，以及如何在 TOML 中指定它们。

**注意:** TOML 支持与 Python 相同语法的注释。散列符号(`#`)将该行的其余部分标记为注释。使用注释可以使您的配置文件更容易理解，便于您和您的用户使用。

在本教程中，您将看到 TOML 的所有不同元素。然而，一些细节和边缘情况将被掩盖。如果你对细则感兴趣，请查阅[文档](https://toml.io/)。

如前所述，**键值对**是 TOML 文档中的基本构件。您用一个`<key> = <value>`语法指定它们，其中键用等号与值分开。以下是具有一个键值对的有效 TOML 文档:

```py
greeting  =  "Hello, TOML!"
```

在这个例子中，`greeting`是键，而`"Hello, TOML!"`是值。值有类型。在本例中，该值是一个文本字符串。您将在下面的小节中了解不同的值类型。

键总是被解释为字符串，即使引号没有将它们括起来。考虑下面的例子:

```py
greeting  =  "Hello, TOML!" 42  =  "Life, the universe, and everything"
```

这里，`42`是一个有效的键，但是它被解释为一个字符串，而不是一个数字。通常，你要使用**光杆键**。这些密钥仅由 ASCII 字母和数字以及下划线和破折号组成。所有这样的键都可以不用引号，就像上面的例子一样。

TOML 文档必须用 UTF-8 Unicode 编码。这让你在表达自己的价值观时有很大的灵活性。尽管对空键有限制，但是在拼写键时也可以使用 Unicode。然而，这是有代价的。要使用 Unicode 键，必须在它们周围加上引号:

```py
"realpython.com"  =  "Real Python" "blåbærsyltetøy"  =  "blueberry jam" "Tom Preston-Werner"  =  "creator"
```

所有这些键都包含裸键中不允许的字符:点(`.`)、挪威语字符(`å`、`æ`和`ø`)和一个空格。您可以在任何键周围使用引号，但是一般来说，您希望坚持使用不使用或不需要引号的简单键。

点(`.`)在 TOML 键中起着特殊的作用。您可以在未加引号的键中使用点，但在这种情况下，它们会通过在每个点处拆分带点的键来触发分组。考虑下面的例子:

```py
player_x.symbol  =  "X" player_x.color  =  "purple"
```

这里，您指定了两个点键。因为它们都以`player_x`开始，所以键`symbol`和`color`将被组合在一个名为`player_x`的部分中。当你开始探索[表](#tables)时，你会学到更多关于点键的知识。

接下来，把注意力转向价值观。在下一节中，您将了解 TOML 中最基本的数据类型。

[*Remove ads*](/account/join/)

### 字符串、数字和布尔值

TOML 对基本数据类型使用熟悉的语法。从 Python 中，您可以识别字符串、整数、浮点数和布尔值:

```py
string  =  "Text with quotes" integer  =  42 float  =  3.11 boolean  =  true
```

TOML 和 Python 最直接的区别就是 TOML 的布尔值是小写的:`true`和`false`。

一个 TOML **字符串**通常应该使用双引号(`"`)。在字符串内部，可以借助反斜杠对特殊字符进行转义:`"\u03c0 is less than four"`。这里，`\u03c0`表示带有 [codepoint U+03c0](https://unicodeplus.com/U+03C0) 的 Unicode 字符，恰好是希腊字母π。该字符串将被解释为`"π is less than four"`。

还可以使用单引号(`'`)指定 TOML 字符串。单引号字符串被称为**文字字符串**，其行为类似于 Python 中的[原始字符串](https://realpython.com/python-data-types/#raw-strings)。在文字字符串中没有任何东西被转义和解释，所以`'\u03c0 is the Unicode codepoint of π'`从文字`\u03c0`字符开始。

最后，还可以使用**三重引号** ( `"""`或`'''`)来指定 TOML 字符串。三重引号字符串允许您在多行上编写一个字符串，类似于 Python 多行字符串:

```py
partly_zen  =  """
Flat is better than nested.
Sparse is better than dense.
"""
```

基本字符串中不允许出现控制字符，包括文字换行符。不过，您可以使用`\n`来表示基本字符串中的换行符。如果要将字符串格式化为多行，必须使用多行字符串。您也可以使用三重引号文字字符串。除了多行之外，这是在文字字符串中包含单引号的唯一方法:`'''Use '\u03c0' to represent π'''`。

**注意:**在 Python 代码中创建 TOML 文档时要小心特殊字符，因为 Python 也会解释这些特殊字符。例如，下面是一个有效的 TOML 文档:

```py
numbers  =  "one\ntwo\nthree"
```

在这里，`numbers`的值是一个分成三行的字符串。您可以尝试用 Python 表示同一个文档，如下所示:

>>>

```py
>>> 'numbers = "one\ntwo\nthree"'
'numbers = "one\ntwo\nthree"'
```

这是行不通的，因为 Python 解析了`\n`字符并创建了一个无效的 TOML 文档。您需要让特殊字符远离 Python，例如使用原始字符串:

>>>

```py
>>> r'numbers = "one\ntwo\nthree"'
'numbers = "one\\ntwo\\nthree"'
```

该字符串表示与原始文档相同的 TOML 文档。

TOML 中的数字要么是整数，要么是浮点数。**整数**代表整数，被指定为普通的数字字符。与 Python 中一样，您可以使用下划线来增强可读性:

```py
number  =  42 negative  =  -8 large  =  60_481_729
```

**浮点数**代表十进制数，包括整数部分、代表小数点的点和小数部分。浮点数可以使用科学记数法来表示非常小或非常大的数字。TOML 还支持特殊的浮点值，比如无穷大和[非数字(NaN)](https://realpython.com/python-math-module/#not-a-number-nan) :

```py
number  =  3.11 googol  =  1e100 mole  =  6.22e23 negative_infinity  =  -inf not_a_number  =  nan
```

注意，TOML 规范要求整数至少要表示为 64 位有符号整数。Python 处理任意大的整数，但是只有大约 19 位数的整数才能保证在所有的 TOML 实现中工作。

**注意:** TOML 是一种配置文件格式，不是编程语言。不支持类似`1 + 2`的表达式，只支持文字数字。

非负整数值也可以分别用前缀`0x`、`0o`或`0b`表示为十六进制、八进制或二进制值。例如，`0xffff00`是十六进制表示，`0b00101010`是二进制表示。

**布尔**值表示为`true`和`false`。这些必须是小写的。

TOML 还包括几种时间和日期类型。但是，在探索这些之前，您将看到如何使用表来组织和结构化您的键值对。

### 表格

您已经了解了 TOML 文档由一个或多个键值对组成。当用编程语言表示时，这些应该存储在一个[散列表](https://realpython.com/python-hash-table/)数据结构中。在 Python 中，这将是一个[字典](https://realpython.com/python-dicts/)或另一个类似于[字典的](https://docs.python.org/3/glossary.html#term-mapping)数据结构。为了组织键值对，可以使用**表**。

TOML 支持三种不同的指定表格的方式。您将很快看到这些例子。最终结果将是相同的，与您如何表示您的表无关。尽管如此，不同的表确实有稍微不同的用例:

*   在大多数情况下，使用带有**标题**的常规**表**。
*   当您需要指定一些与其父表紧密相关的键值对时，使用**点状键表**。
*   将**内联表**仅用于最多有三个键值对的非常小的表，其中的数据构成了一个明确定义的实体。

不同的表格表示通常是可以互换的。您应该默认使用常规表，只有当您认为这样可以提高配置的可读性或阐明您的意图时，才切换到点键表或内联表。

这些不同的表格类型在实践中看起来如何？从普通桌子开始。它们是通过在键值对上方添加一个表头来定义的。头是一个没有值的**键**，用方括号(`[]`)括起来。您之前遇到的以下示例定义了三个表:

```py
[user]  player_x.color  =  "blue" player_o.color  =  "green" [constant]  board_size  =  3 [server]  url  =  "https://tictactoe.example.com"
```

突出显示的三行是表格标题。它们指定了三个表，分别命名为`user`、`constant`和`server`。表的内容或值是列在标题下面和下一个标题上面的所有键值对。例如，`constant`和`server`各包含一个嵌套的键值对。

你也可以在上面的配置中找到**虚线键表**。在`user`中，您有以下内容:

```py
[user] player_x.color  =  "blue"  player_o.color  =  "green"
```

键中的句点或点(`.`)创建一个由点之前的键部分命名的表。您也可以通过嵌套常规表来表示配置的相同部分:

```py
[user] [user.player_x] color  =  "blue" [user.player_o] color  =  "green"
```

缩进在 TOML 中并不重要。这里用它来表示表格的嵌套。您可以看到，`user`表包含两个子表，`player_x`和`player_o`。每个子表都包含一个键值对。

注意:你可以任意深度的嵌套 TOML 表。例如，`player.x.color.name`这样的键或表头表示`color`表中的`name`和`player`表中的`x`。

请注意，您需要在嵌套表的标题中使用点键，并命名所有中间表。这使得 TOML 头规范相当冗长。例如，在 JSON 或 YAML 的类似规范中，您只需指定子表的名称，而不必重复外部表的名称。同时，这使得 TOML 非常显式，在深度嵌套的结构中更难迷失方向。

现在，您将在`user`桌面上扩展一点，为每个玩家添加一个**标签**或**符号**。您将用三种不同的形式来表示这个表，首先只使用常规表，然后使用点键表，最后使用内联表。您还没有看到后者，所以这将是对内联表以及如何表示它们的介绍。

从嵌套的常规表格开始:

```py
[user] [user.player_x] symbol  =  "X" color  =  "blue" [user.player_o] symbol  =  "O" color  =  "green"
```

这种表示非常清楚地表明，你有两个不同的球员表。您不需要显式定义只包含子表而不包含任何常规键的表。在前面的例子中，您可以删除线`[user]`。

将嵌套表与点键配置进行比较:

```py
[user] player_x.symbol  =  "X" player_x.color  =  "blue" player_o.symbol  =  "O" player_o.color  =  "green"
```

这比上面的嵌套表更短更简洁。然而，结构现在不太清楚了，在您意识到在`user`中嵌套了两个玩家表之前，您需要花费一些精力来解析这些键。当您有几个嵌套表，每个表有一个键时，点键表会更有用，就像前面的例子中只有`color`个子键一样。

接下来，您将使用**内联表格**来表示`user`:

```py
[user] player_x  =  {  symbol  =  "X",  color  =  "blue"  }  player_o  =  {  symbol  =  "O",  color  =  "green"  }
```

内联表是用花括号(`{}`)定义的，用逗号分隔的键值对。在这个例子中，内联表带来了可读性和紧凑性的良好平衡，因为玩家表的分组变得清晰。

不过，您应该谨慎地使用内联表，主要是在这种情况下，一个表代表一个小型的、定义良好的实体，比如一个播放器。与常规表格相比，内联表格被有意地限制。特别是，内联表必须写在 TOML 文件中的一行上，并且不能使用像[结尾逗号](https://docs.python.org/3/faq/design.html#why-does-python-allow-commas-at-the-end-of-lists-and-tuples)这样的便利。

在结束 TOML 中的表之旅之前，您将简要地看一下几个小问题。一般来说，您可以按任何顺序定义您的表，并且您应该努力以一种对您的用户有意义的方式来排列您的配置。

TOML 文档由一个包含所有其他表和键值对的无名根表表示。您在 TOML 配置的顶部，在任何表头之前编写的键-值对直接存储在根表中:

```py
title  =  "Tic-Tac-Toe" [constant] board_size  =  3
```

在这个例子中，`title`是根表中的一个键，`constant`是嵌套在根表中的一个表，`board_size`是`constant`表中的一个键。

请注意，一个表包括所有写在它的表头和下一个表头之间的键值对。实际上，这意味着您必须在属于该表的键值对下定义嵌套子表。考虑这份文件:

```py
[user] [user.player_x] color  =  "blue" [user.player_o] color  =  "green" background_color  =  "white"
```

缩进表明`background_color`应该是`user`表中的一个键。但是，TOML 忽略缩进，只检查表头。在这个例子中，`background_color`是`user.player_o`表的一部分。要纠正这一点，`background_color`应该在嵌套表之前定义:

```py
[user] background_color  =  "white" [user.player_x] color  =  "blue" [user.player_o] color  =  "green"
```

在这种情况下，`background_color`是`user`中的一个键。如果你使用点键表，那么你可以更自由地使用任何顺序的键:

```py
[user] player_x.color  =  "blue" player_o.color  =  "green" background_color  =  "white"
```

现在除了`[user]`之外没有显式的表头，所以`background_color`将是`user`表中的一个键。

您已经了解了 TOML 中的基本数据类型，以及如何使用表来组织数据。在接下来的小节中，您将看到可以在 TOML 文档中使用的最终数据类型。

[*Remove ads*](/account/join/)

### 时间和日期

TOML 支持直接在文档中定义时间和日期。您可以在四种不同的表示法之间进行选择，每种表示法都有其特定的用例:

*   **偏移日期时间**是一个带有时区信息的时间戳，代表特定的时刻。
*   **本地日期时间**是没有时区信息的时间戳。
*   **本地日期**是没有任何时区信息的日期。你通常用它来代表一整天。
*   **本地时间**是具有任何日期或时区信息的时间。您使用本地时间来表示一天中的某个时间。

TOML 基于 RFC 3339 来表示时间和日期。本文档定义了一种时间和日期格式，通常用于表示互联网上的时间戳。一个完整定义的时间戳应该是这样的:`2021-01-12T01:23:45.654321+01:00`。时间戳由几个字段组成，由不同的分隔符分隔:

| 田 | 例子 | 细节 |
| --- | --- | --- |
| 年 | `2021` |  |
| 月 | `01` | 从`01`(1 月)到`12`(12 月)的两位数 |
| 一天 | `12` | 两位数，十位以下用零填充 |
| 小时 | `01` | 从`00`到`23`的两位数 |
| 分钟 | `23` | 从`00`到`59`的两位数 |
| 第二 | `45` | 从`00`到`59`的两位数 |
| 微秒 | `654321` | 从`000000`到`999999`的六位数字 |
| 抵消 | `+01:00` | 时区相对于 UTC 的偏移量，`Z`代表 UTC |

偏移日期时间是包含偏移信息的[时间戳](https://realpython.com/python-datetime/)。本地日期时间是不包括这个的时间戳。本地时间戳也称为简单时间戳。

在 TOML 中，微秒字段对于所有日期时间和时间类型都是可选的。您还可以用空格替换分隔日期和时间的`T`。在这里，您可以看到每种时间戳相关类型的示例:

```py
offset_date-time  =  2021-01-12 01:23:45+01:00 offset_date-time_utc  =  2021-01-12 00:23:45Z local_date-time  =  2021-01-12  01:23:45 local_date  =  2021-01-12 local_time  =  01:23:45 local_time_with_us  =  01:23:45.654321
```

注意，不能用引号将时间戳值括起来，因为这会将它们转换成文本字符串。

这些不同的时间和日期类型为您提供了相当大的灵活性。如果您有这些没有涵盖的用例—例如，如果您想要指定一个像`1 day`这样的时间间隔—那么您可以使用字符串并使用您的应用程序来正确地处理它们。

TOML 支持的最后一种数据类型是数组。这些允许您在一个列表中组合其他几个值。请继续阅读，了解更多信息。

### 数组

TOML 数组表示一个有序的值列表。您使用方括号(`[]`)来指定它们，以便它们类似于 Python 的列表:

```py
packages  =  ["tomllib",  "tomli",  "tomli_w",  "tomlkit"]
```

在这个例子中，`packages`的值是一个包含四个字符串元素的数组:`"tomllib"`、`"tomli"`、`"tomli_w"`和`"tomlkit"`。

**注:**你会学到更多关于 [`tomllib`](https://docs.python.org/3.11/library/tomllib.html) ， [`tomli`](https://pypi.org/project/tomli/) ， [`tomli_w`](https://pypi.org/project/tomli_w/) ， [`tomlkit`](https://pypi.org/project/tomlkit/) 的知识，以及它们在 Python 的 TOML 版图中所扮演的角色，在本教程后面的的实际章节中。

您可以在数组中使用任何 TOML 数据类型，包括其他数组，并且一个数组可以包含不同的数据类型。您可以在几行中指定一个数组，并且可以在数组的最后一个元素后使用尾随逗号。以下所有示例都是有效的 TOML 数组:

```py
potpourri  =  ["flower",  1749,  {  symbol  =  "X",  color  =  "blue"  },  1994-02-14] skiers  =  ["Thomas",  "Bjørn",  "Mika"] players  =  [ {  symbol  =  "X",  color  =  "blue",  ai  =  true  }, {  symbol  =  "O",  color  =  "green",  ai  =  false  }, ]
```

这定义了三个数组。`potpourri`是包含四个不同数据类型元素的数组，而`skiers`是包含三个字符串的数组。最后一个数组`players`修改了前面的例子，将两个内联表格表示为一个数组中的元素。注意`players`是在四行中定义的，在最后一个行内表格后面有一个可选的逗号。

最后一个例子展示了创建表格数组的一种方法。您可以将内联表格放在方括号内。但是，正如您之前看到的，内联表的伸缩性不好。如果您想要表示一个表的数组，其中的表比较大，那么您应该使用不同的语法。

一般来说，您应该通过在双方括号(`[[]]`)内编写表格标题来表达表格的**数组。语法不一定漂亮，但相当有效。您可以用下面的例子来表示`players`:**

```py
[[players]] symbol  =  "X" color  =  "blue" ai  =  true [[players]] symbol  =  "O" color  =  "green" ai  =  false
```

这个表格数组相当于您上面编写的内联表格数组。双方括号定义了一个表格数组，而不是一个常规的表格。您需要为数组中的每个嵌套表重复数组名称。

作为一个更广泛的例子，考虑下面的 TOML 文档摘录，该文档列出了一个[测验应用程序](https://realpython.com/python-quiz-application/)的问题:

```py
[python] label  =  "Python" [[python.questions]] question  =  "Which built-in function can get information from the user" answers  =  ["input"] alternatives  =  ["get",  "print",  "write"] [[python.questions]] question  =  "What's the purpose of the built-in zip() function" answers  =  ["To iterate over two or more sequences at the same time"] alternatives  =  [ "To combine several strings into one", "To compress several files into one archive", "To get information from the user", ]
```

在这个例子中，`python`表有两个键，`label`和`questions`。`questions`的值是一个包含两个元素的表格数组。每个元素是一个带有三个键的表格:`question`、`answers`和`alternatives`。

您现在已经看到了 TOML 必须提供的所有数据类型。除了简单的数据类型，如字符串、数字、布尔值、时间和日期，您还可以用表和数组来组合和组织您的键和值。在这篇概述中，您忽略了一些细节和边缘情况。你可以在 [TOML 规范](https://toml.io/en/latest)中了解所有细节。

在接下来的章节中，当您学习如何在 Python 中使用 TOML 时，您会变得更加实际。您将了解如何读写 TOML 文档，并探索如何组织您的应用程序来有效地使用配置文件。

[*Remove ads*](/account/join/)

## 用 Python 加载 TOML】

是时候把手弄脏了。在本节中，您将启动 Python 解释器并将 TOML 文档加载到 Python 中。您已经看到了 TOML 格式的主要用例是配置文件。这些通常是手工编写的，因此在这一节中，您将了解如何使用 Python 读取这样的配置文件，并在您的项目中使用它们。

### 用`tomli`和`tomllib` 读取 TOML 文件

自从 TOML 规范在 2013 年首次出现以来，已经有几个包可以使用这种格式。随着时间的推移，这些包中的一些变得不可维护。一些曾经流行的库不再兼容最新版本的 TOML。

在本节中，您将使用一个相对较新的包，名为 [`tomli`](https://pypi.org/project/tomli/) 及其兄弟包 [`tomllib`](https://docs.python.org/3.11/library/tomllib.html) 。当您只想将一个 TOML 文档加载到 Python 中时，这些是很好的库。在以后的章节中，您还将探索`tomlkit`。该包为桌面带来了更高级的功能，并为您开辟了一些新的使用案例。

**注意:**在 Python 3.11 中，TOML 支持被添加到 Python 标准库中。新的`tomllib`模块可以帮助你读取和解析 TOML 文档。关于添加库的动机和原因详见 [Python 3.11 预览版:TOML 和`tomllib`](https://realpython.com/python311-tomllib/) 。

这个新的`tomllib`模块实际上是[通过将现有的`tomli`库复制到 CPython 代码库中而创建的](https://github.com/python/cpython/pull/31498)。这样做的结果是，你可以在 Python 版本 [3.7](https://realpython.com/python37-new-features/) 、 [3.8](https://realpython.com/python38-new-features/) 、 [3.9](https://realpython.com/python39-new-features/) 和 [3.10](https://realpython.com/python310-new-features/) 上使用`tomli`作为兼容的后端口。

按照下面的说明，你将学会如何使用`tomli`。如果你使用的是 Python 3.11，那么你可以跳过`tomli`的安装，用`tomllib`替换任何提到`tomli`的代码。

是时候探索如何读取 TOML 文件了。首先创建以下 TOML 文件，并将其另存为`tic_tac_toe.toml`:

```py
# tic_tac_toe.toml [user] player_x.color  =  "blue" player_o.color  =  "green" [constant] board_size  =  3 [server] url  =  "https://tictactoe.example.com"
```

这与您在上一节中使用的配置相同。接下来，使用 [`pip`](https://realpython.com/what-is-pip/) 将`tomli`安装到您的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中:

```py
(venv) $ python -m pip install tomli
```

`tomli`模块只公开了两个函数:`load()`和`loads()`。您可以使用它们分别从 file 对象和 string 加载 TOML 文档。首先使用`load()`读取您在上面创建的文件:

>>>

```py
>>> import tomli
>>> with open("tic_tac_toe.toml", mode="rb") as fp:
...     config = tomli.load(fp) ...
```

你首先打开文件，使用一个[上下文管理器](https://realpython.com/python-with-statement/)来处理[可能出现的任何问题](https://realpython.com/why-close-file-python/)。重要的是，您需要通过指定`mode="rb"`以二进制模式打开文件。这允许`tomli`正确处理您的 TOML 文件的编码。

您将 TOML 配置存储在一个名为`config`的变量中。继续探索它的内容:

>>>

```py
>>> config
{'user': {'player_x': {'color': 'blue'}, 'player_o': {'color': 'green'}},
 'constant': {'board_size': 3},
 'server': {'url': 'https://tictactoe.example.com'}}

>>> config["user"]["player_o"]
{'color': 'green'}

>>> config["server"]["url"]
'https://tictactoe.example.com'
```

在 Python 中，TOML 文档被表示为字典。TOML 文件中的所有表和子表都显示为`config`中的嵌套字典。您可以通过跟踪嵌套字典中的键来挑选单个值。

如果您已经将 TOML 文档表示为字符串，那么您可以使用`loads()`代替`load()`。可以把函数名后面的`s`看作是*字符串*的助记符。以下示例解析存储为`toml_str`的 TOML 文档:

>>>

```py
>>> import tomli
>>> toml_str = """
... offset_date-time_utc = 2021-01-12 00:23:45Z
... potpourri = ["flower", 1749, { symbol = "X", color = "blue" }, 1994-02-14]
... """

>>> tomli.loads(toml_str) {'offset_date-time_utc': datetime.datetime(2021, 1, 12, 0, 23, 45,
 tzinfo=datetime.timezone.utc),
 'potpourri': ['flower',
 1749,
 {'symbol': 'X', 'color': 'blue'},
 datetime.date(1994, 2, 14)]}
```

同样，您将生成一个字典，其中的键和值对应于 TOML 文档中的键值对。注意，TOML 时间和日期类型由 Python 的`datetime`类型表示，TOML 数组被转换成 Python 列表。您可以看到，正如预期的那样，在`.tzinfo`属性中表示的时区信息被附加到了`offset_date-time_utc`。

**注意:**偏移日期时间是具有指定时区的日期时间。将时区添加到`datetime`意味着您提供了足够的信息来描述一个确切的时刻，这在许多处理真实世界数据的应用程序中非常重要。

看看 [Python 3.9:很酷的新特性供你尝试](https://realpython.com/python39-new-features/#proper-time-zone-support)阅读更多关于 Python 如何处理时区的信息，并查看 Python 3.9 版本中添加的`zoneinfo`模块。

`load()`和`loads()`都将 TOML 文档转换成 Python 字典，并且可以互换使用。选择最适合您的使用情形的一个。作为最后一个例子，您将结合`loads()`和`pathlib`来重建井字游戏配置示例:

>>>

```py
>>> from pathlib import Path
>>> import tomli
>>> tomli.loads(Path("tic_tac_toe.toml").read_text(encoding="utf-8")) {'user': {'player_x': {'color': 'blue'}, 'player_o': {'color': 'green'}},
 'constant': {'board_size': 3},
 'server': {'url': 'https://tictactoe.example.com'}}
```

`load()`和`loads()`的一个区别是当你使用后者时，你使用常规的字符串而不是字节。在这种情况下，`tomli`假设您已经正确处理了编码。

**注:**这些例子都用了`tomli`。然而，如上所述，如果您使用的是 Python 3.11 或更新版本，您可以用`tomllib`替换任何提到`tomli`的代码。

你可能想在你的应用程序中自动执行这个决定。您可以通过将下面一行添加到您的`requirements.txt`依赖项规范中来实现这一点:

```py
tomli >= 1.1.0 ; python_version < "3.11"
```

这将确保`tomli`只安装在 3.11 之前的 Python 版本上。此外，您应该用一个稍微复杂一点的咒语替换您导入的`tomli`:

```py
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
```

这段代码将首先尝试导入`tomllib`。如果失败，它将导入`tomli`，但是将`tomli`模块的别名改为`tomllib`名称。由于这两个库是兼容的，你现在可以在你的代码中引用`tomllib`，它将在所有 Python 版本 3.7 和更高版本上工作。

您已经开始使用 Python 加载并解析了您的第一个 TOML 文档。在下一小节中，您将更仔细地查看 TOML 数据类型和来自`tomli`的输出之间的对应关系。

[*Remove ads*](/account/join/)

### 比较 TOML 类型和 Python 类型

在前一小节中，您加载了一些 TOML 文档，并学习了`tomli`和`tomllib`如何表示，例如，将 TOML 字符串表示为 Python 字符串，将 TOML 数组表示为 Python 列表。TOML 规范没有明确定义 Python 应该如何表示 TOML 对象，因为这超出了它的范围。然而，TOML 规范提到了对其自身类型的一些要求。例如:

*   TOML 文件必须是有效的 UTF-8 编码的 Unicode 文档。
*   应该无损地接受和处理任意 64 位有符号整数(从−2^63 到 2^63−1)。
*   浮点应该实现为 IEEE 754 二进制 64 值。

总的来说，TOML 的需求与 Python 的相应类型的实现匹配得很好。Python [通常](https://realpython.com/python310-new-features/#default-text-encodings)在处理文件时默认使用 UTF-8，一个 [Python `float`](https://realpython.com/python-numbers/#floating-point-numbers) 遵循 IEEE 754。Python 的`int` 类实现了任意精度的整数，可以处理所需的范围和更大的数字。

对于像`tomli`和`tomllib`这样的基本库，TOML 的数据类型和 Python 的数据类型之间的映射是相当自然的。您可以在`tomllib`的文档[中找到以下换算表:](https://docs.python.org/3.11/library/tomllib.html#conversion-table)

| 汤姆 | 计算机编程语言 |
| --- | --- |
| 线 | `str` |
| 整数 | `int` |
| 漂浮物 | `float` |
| 布尔型 | `bool` |
| 桌子 | `dict` |
| 偏移日期时间 | `datetime.datetime` ( `.tzinfo`是`datetime.timezone`的一个实例) |
| 当地日期时间 | `datetime.datetime` ( `.tzinfo`是`None`) |
| 当地日期 | `datetime.date` |
| 当地时间 | `datetime.time` |
| 排列 | `list` |

所有的 Python 数据类型要么是内置的[，要么是标准库中](https://docs.python.org/3/library/stdtypes.html) [`datetime`](https://docs.python.org/3/library/datetime.html) 的一部分。重申一下，并不要求 TOML 类型必须映射到本地 Python 类型。这是`tomli`和`tomllib`选择实现的便利。

仅使用标准类型也是一种限制。实际上，您只能表示值，而不能表示 TOML 文档中编码的其他信息，如注释或缩进。您的 Python 表示也没有区分在常规表或内联表中定义的值。

在许多用例中，这个元信息是不相关的，所以不会丢失任何东西。然而，有时这很重要。例如，如果您试图在现有的 TOML 文件中插入一个表格，那么您不希望所有的注释都消失。稍后你会了解到`tomlkit`。这个库将 TOML 类型表示为定制的 Python 对象，这些对象保留了恢复完整的 TOML 文档所必需的信息。

`load()`和`loads()`函数有一个参数，可以用来定制 TOML 解析。您可以向`parse_float`提供一个参数来指定应该如何解析浮点数。默认实现满足了使用 64 位浮点数的要求，这通常精确到大约 16 位有效数字。

但是，如果您的应用程序依赖于非常精确的数字，16 位数字可能不够。作为例子，考虑天文学中使用的[儒略日](https://en.wikipedia.org/wiki/Julian_day)的概念。这是一个时间戳的表示，它是一个计数自 6700 多年前的儒略历开始以来的天数的数字。例如，UTC 时间 2022 年 7 月 11 日中午是儒略日 2，459，772。

天文学家有时需要在非常小的时间尺度上工作，比如纳秒甚至皮秒。要以纳秒的精度表示一天中的时间，在小数的小数点后需要大约 14 位数字。例如，UTC 时间 2022 年 7 月 11 日下午 2:01，表示为具有纳秒精度的儒略日，即 245。58661 . 86768678671

像这样的数字，既有很大的值，又精确到许多小数位，不太适合表示为浮点数。如果你用`tomli`读这个儒略日，你会损失多少精度？打开 REPL，体验一下:

>>>

```py
>>> import tomli
>>> ts = tomli.loads("ts = 2_459_772.084027777777778")["ts"]
>>> ts
2459772.084027778

>>> seconds = (ts - int(ts)) * 86_400
>>> seconds
7260.000009834766

>>> seconds - 7260
9.834766387939453e-06
```

首先使用`tomli`解析儒略日，挑选出值，并将其命名为`ts`。您可以看到`ts`的值被截断了几个小数位。为了弄清楚截断的效果有多糟糕，您计算由`ts`的小数部分表示的秒数，并将其与 7260 进行比较。

整数儒略日代表某一天的中午。下午 2:01 是中午之后的两小时零一分钟，两小时零一分钟等于 7260 秒，所以`seconds - 7260`向您展示了您的解析引入了多大的误差。

在这种情况下，您的时间戳大约有 10 微秒的误差。这听起来可能不多，但在许多天文应用中，信号以光速传播。在这种情况下，10 微秒可能会导致大约 3 公里的误差！

这个问题的一个常见解决方案是不将非常精确的时间戳存储为儒略日。取而代之的是许多具有更高精度的[变体](https://en.wikipedia.org/wiki/Julian_day#Variants)。然而，您也可以通过使用 Python 的`Decimal`类来修复您的示例，该类提供任意精度的十进制数。

回到你的 REPL，重复上面的例子:

>>>

```py
>>> import tomli
>>> from decimal import Decimal
>>> ts = tomli.loads(
...     "ts = 2_459_772.084027777777778",
...     parse_float=Decimal, ... )["ts"]
>>> ts
Decimal('2459772.084027777777778')

>>> seconds = (ts - int(ts)) * 86_400
>>> seconds
Decimal('7260.000000000019200')

>>> seconds - 7260
Decimal('1.9200E-11')
```

现在，剩下的小误差来自你的原始表示，大约是 19 皮秒，相当于光速下的亚厘米误差。

当你知道你需要精确的浮点数时，你可以使用`Decimal`。在更具体的用例中，您还可以将数据存储为字符串，并在读取 TOML 文件后解析应用程序中的字符串。

到目前为止，您已经看到了如何用 Python 读取 TOML 文件。接下来，您将讨论如何将配置文件合并到您自己的项目中。

[*Remove ads*](/account/join/)

### 在项目中使用配置文件

您有一个项目，其中包含一些您想要提取到配置文件中的设置。回想一下，配置可以通过多种方式改进您的代码库:

*   它**命名**价值观和概念。
*   它为特定值提供了更多**可见性**。
*   它使得**改变**的值更简单。

配置文件可以帮助您了解源代码，并增加用户与应用程序交互的灵活性。您知道如何阅读基于 TOML 的配置文件，但是如何在您的项目中使用它呢？

特别是，你如何确保配置文件只被**解析一次**，你如何从不同的模块访问配置**？**

原来 Python 的[导入系统](https://realpython.com/python-import/#the-python-import-system)已经支持这两个开箱即用的特性。当您导入一个模块时，它会被缓存以备后用。换句话说，如果您将您的配置包装在一个模块中，您知道该配置将只被读取一次，即使您从几个地方导入该模块。

是时候举个具体的例子了。调用前面的`tic_tac_toe.toml`配置文件:

```py
# tic_tac_toe.toml [user] player_x.color  =  "blue" player_o.color  =  "green" [constant] board_size  =  3 [server] url  =  "https://tictactoe.example.com"
```

创建一个名为`config/`的目录，并将`tic_tac_toe.toml`保存在该目录中。另外，在`config/`中创建一个名为`__init__.py`的空文件。您的小型目录结构应该如下所示:

```py
config/
├── __init__.py
└── tic_tac_toe.toml
```

名为`__init__.py`的文件在 Python 中起着特殊的作用。它们将包含目录标记为包。此外，在`__init__.py`中定义的名字通过包公开。您将很快看到这在实践中意味着什么。

现在，向`__init__.py`添加代码以读取配置文件:

```py
# __init__.py

import pathlib
import tomli

path = pathlib.Path(__file__).parent / "tic_tac_toe.toml"
with path.open(mode="rb") as fp:
    tic_tac_toe = tomli.load(fp)
```

像前面一样，使用`load()`读取 TOML 文件，并将 TOML 数据存储到名称`tic_tac_toe`中。你使用 [`pathlib`](https://realpython.com/python-pathlib/) 和特殊的 [`__file__`](https://docs.python.org/3/reference/import.html#file__) 变量来设置`path`，TOML 文件的完整路径。实际上，这指定了 TOML 文件存储在与`__init__.py`文件相同的目录中。

通过从`config/`的父目录启动 REPL 会话来试用您的小软件包:

>>>

```py
>>> import config
>>> config.path
PosixPath('/home/realpython/config/tic_tac_toe.toml')

>>> config.tic_tac_toe
{'user': {'player_x': {'color': 'blue'}, 'player_o': {'color': 'green'}},
 'constant': {'board_size': 3},
 'server': {'url': 'https://tictactoe.example.com'}}
```

您可以检查配置的路径并访问配置本身。要读取特定值，可以使用常规项目访问:

>>>

```py
>>> config.tic_tac_toe["server"]["url"]
'https://tictactoe.example.com'

>>> config.tic_tac_toe["constant"]["board_size"]
3

>>> config.tic_tac_toe["user"]["player_o"]
{'color': 'green'}

>>> config.tic_tac_toe["user"]["player_o"]["color"]
'green'
```

现在，您可以通过将`config/`目录复制到您的项目中，并用您自己的设置替换井字游戏配置，来将配置集成到您现有的项目中。

在代码文件中，您可能希望为配置导入设置别名，以便更方便地访问您的设置:

>>>

```py
>>> from config import tic_tac_toe as CFG 
>>> CFG["user"]["player_x"]["color"]
'blue'
```

在这里，您可以在导入过程中将配置命名为`CFG`，这使得访问配置设置既高效又易读。

这个菜谱为您提供了一种在您自己的项目中使用配置的快速而可靠的方法。

[*Remove ads*](/account/join/)

## 将 Python 对象转储为 TOML

您现在知道如何用 Python 读取 TOML 文件了。怎么能反其道而行之呢？TOML 文档通常是手写的，因为它们主要用作配置。尽管如此，有时您可能需要将嵌套字典转换成 TOML 文档。

在这一节中，您将从手工编写一个基本的 TOML 编写器开始。然后，您会看到哪些工具已经可用，并使用第三方的`tomli_w`库将您的数据转储到 TOML。

### 将字典转换为 TOML

回想一下您之前使用的井字游戏配置。您可以将其稍加修改的版本表示为嵌套的 Python 字典:

```py
{
    "user": {
        "player_x": {"symbol": "X", "color": "blue", "ai": True},
        "player_o": {"symbol": "O", "color": "green", "ai": False},
        "ai_skill": 0.85,
    },
    "board_size": 3,
    "server": {"url": "https://tictactoe.example.com"},
}
```

在这一小节中，您将编写一个简化的 TOML 编写器，它能够将本词典编写为 TOML 文档。你不会实现 TOML 的所有特性。特别是，您忽略了一些值类型，如时间、日期和表格数组。您也没有处理需要加引号的键或多行字符串。

尽管如此，您的实现将处理 TOML 的许多典型用例。在下一小节中，您将看到如何使用一个库来处理规范的其余部分。打开编辑器，创建一个名为`to_toml.py`的新文件。

首先，编写一个名为`_dumps_value()`的助手函数。该函数将接受某个值，并基于值类型返回其 TOML 表示。您可以通过`isinstance()`检查来实现这一点:

```py
# to_toml.py

def _dumps_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, list):
        return f"[{', '.join(_dumps_value(v) for v in value)}]"
    else:
        raise TypeError(f"{type(value).__name__}  {value!r} is not supported")
```

您为布尔值返回`true`或`false`,并在字符串两边加上双引号。如果你的值是一个列表，你可以通过递归调用`_dumps_value()`来创建一个 TOML 数组。如果你正在使用 Python 3.10 或更新版本，那么你可以用一个`match` … `case`语句来[替换](https://realpython.com/python311-tomllib/#write-toml)你的`isinstance()`检查。

接下来，您将添加处理这些表的代码。您的 main 函数循环遍历一个字典，并将每个条目转换成一个键值对。如果值碰巧是一个字典，那么您将添加一个表头并递归地填写该表:

```py
# to_toml.py

# ...

def dumps(toml_dict, table=""):
    toml = []
    for key, value in toml_dict.items():
        if isinstance(value, dict):
            table_key = f"{table}.{key}" if table else key
            toml.append(f"\n[{table_key}]\n{dumps(value, table_key)}")
        else:
            toml.append(f"{key} = {_dumps_value(value)}")
    return "\n".join(toml)
```

为了方便起见，可以使用一个列表，在添加表或键值对时跟踪它们。在返回之前，将这个列表转换成一个字符串。

除了前面提到的限制，这个函数中还隐藏着一个微妙的错误。考虑一下如果您尝试转储前面的示例会发生什么:

>>>

```py
>>> import to_toml
>>> config = {
...     "user": {
...         "player_x": {"symbol": "X", "color": "blue", "ai": True},
...         "player_o": {"symbol": "O", "color": "green", "ai": False},
...         "ai_skill": 0.85,
...     },
...     "board_size": 3,
...     "server": {"url": "https://tictactoe.example.com"},
... }

>>> print(to_toml.dumps(config))

[user]

[user.player_x]
symbol = "X"
color = "blue"
ai = true

[user.player_o]
symbol = "O"
color = "green"
ai = false
ai_skill = 0.85 board_size = 3 
[server]
url = "https://tictactoe.example.com"
```

请特别注意突出显示的行。看起来`ai_skill`和`board_size`是`user.player_o`表中的键。但根据原始数据，它们应该分别是`user`和根表的成员。

问题是没有办法标记 TOML 表的结束。相反，常规键必须列在任何子表之前。修复代码的一种方法是对字典项进行排序，使字典值排在所有其他值之后。按如下方式更新您的函数:

```py
# to_toml.py

# ...

def dumps(toml_dict, table=""):
 def tables_at_end(item): _, value = item return isinstance(value, dict) 
    toml = []
 for key, value in sorted(toml_dict.items(), key=tables_at_end):        if isinstance(value, dict):
            table_key = f"{table}.{key}" if table else key
            toml.append(f"\n[{table_key}]\n{dumps(value, table_key)}")
        else:
            toml.append(f"{key} = {_dumps_value(value)}")
    return "\n".join(toml)
```

实际上，`tables_at_end()`为所有非字典值返回`False`或`0`，为所有字典值返回`True`，T3 相当于`1`。使用它作为排序关键字可以确保嵌套字典在其他类型的值之后被处理。

现在，您可以重做上面的示例。当您将结果打印到您的终端屏幕时，您将看到下面的 TOML 文档:

```py
board_size  =  3 [user] ai_skill  =  0.85 [user.player_x] symbol  =  "X" color  =  "blue" ai  =  true [user.player_o] symbol  =  "O" color  =  "green" ai  =  false [server] url  =  "https://tictactoe.example.com"
```

这里，`board_size`作为根表的一部分列在顶部，这是意料之中的。另外，`ai_skill`现在是`user`中的一个键，就像它应该的那样。

尽管 TOML 不是一种复杂的格式，但是在创建自己的 TOML 编写器时，您需要考虑一些细节。您不再继续这个任务，而是转而研究如何使用现有的库将数据转储到 TOML 中。

[*Remove ads*](/account/join/)

### 用`tomli_w` 编写 TOML 文档

在本节中，您将使用 [`tomli_w`](https://pypi.org/project/tomli_w/) 库。顾名思义，`tomli_w`与`tomli`有关。它有两个功能，`dump()`和`dumps()`，其设计或多或少与`load()`和`loads()`相反。

**注意:**Python 3.11 中新增的`tomllib`库[不包括](https://realpython.com/python311-tomllib/#write-toml) `dump()`和`dumps()`，也没有`tomllib_w`。相反，你可以使用`tomli_w`在 Python 3.7 以后的所有版本上编写 TOML。

您必须将`tomli_w`安装到您的虚拟环境中，然后才能使用它:

```py
(venv) $ python -m pip install tomli_w
```

现在，尝试重复上一小节中的示例:

>>>

```py
>>> import tomli_w
>>> config = {
...     "user": {
...         "player_x": {"symbol": "X", "color": "blue", "ai": True},
...         "player_o": {"symbol": "O", "color": "green", "ai": False},
...         "ai_skill": 0.85,
...     },
...     "board_size": 3,
...     "server": {"url": "https://tictactoe.example.com"},
... }

>>> print(tomli_w.dumps(config))
board_size = 3

[user]
ai_skill = 0.85

[user.player_x]
symbol = "X"
color = "blue"
ai = true

[user.player_o]
symbol = "O"
color = "green"
ai = false

[server]
url = "https://tictactoe.example.com"
```

毫无疑问:`tomli_w`编写了与您在上一节中手写的`dumps()`函数相同的 TOML 文档。此外，第三方库支持您没有实现的所有功能，包括时间和日期、内联表格和表格数组。

`dumps()`写入可以继续处理的字符串。如果您想将新的 TOML 文档直接存储到磁盘，那么您可以调用`dump()`来代替。与`load()`一样，您需要传入一个以二进制模式打开的文件指针。继续上面的例子:

>>>

```py
>>> with open("tic-tac-toe-config.toml", mode="wb") as fp:
...     tomli_w.dump(config, fp)
...
```

这将把`config`数据结构存储到文件`tic-tac-toe-config.toml`中。查看一下您新创建的文件:

```py
# tic-tac-toe-config.toml board_size  =  3 [user] ai_skill  =  0.85 [user.player_x] symbol  =  "X" color  =  "blue" ai  =  true [user.player_o] symbol  =  "O" color  =  "green" ai  =  false [server] url  =  "https://tictactoe.example.com"
```

您可以在需要的地方找到所有熟悉的表和键值对。

`tomli`和`tomli_w`都很基本，功能有限，同时实现了对 TOML v1.0.0 的完全支持。一般来说，只要它们兼容，您就可以通过 TOML 往返处理您的数据结构:

>>>

```py
>>> import tomli, tomli_w
>>> data = {"fortytwo": 42}
>>> tomli.loads(tomli_w.dumps(data)) == data
True
```

在这里，您确认您能够在第一次转储到 TOML 然后加载回 Python 之后恢复`data`。

**注意:**不应该使用 TOML 进行数据序列化的一个原因是有许多数据类型不受支持。例如，如果您有一个带数字键的字典，那么`tomli_w`理所当然地拒绝将它转换成 TOML:

>>>

```py
>>> import tomli, tomli_w
>>> data = {1: "one", 2: "two"}
>>> tomli.loads(tomli_w.dumps(data)) == data
Traceback (most recent call last):
  ...
TypeError: 'int' object is not iterable
```

这个错误消息不是很有描述性，但是问题是 TOML 不支持像`1`和`2`这样的非字符串键。

之前，您已经了解到`tomli`会丢弃评论。此外，您无法在字典中区分由`load()`或`loads()`返回的文字字符串、多行字符串和常规字符串。总的来说，这意味着当您解析一个 TOML 文档并将其写回时，您会丢失一些元信息:

>>>

```py
>>> import tomli, tomli_w
>>> toml_data = """
... [nested]  # Not necessary
... ...     [nested.table]
...     string       = "Hello, TOML!"
...     weird_string = '''Literal
...         Multiline'''
... """
>>> print(tomli_w.dumps(tomli.loads(toml_data)))
[nested.table]
string = "Hello, TOML!"
weird_string = "Literal\n        Multiline"
```

TOML 内容保持不变，但是您的输出与您传入的完全不同！父表`nested`没有明确地包含在输出中，注释也不见了。此外，`nested.table`中的等号不再对齐，`weird_string`也不再表示为多行字符串。

**注意:**您可以使用`multiline_strings`参数来指示`tomli_w`在适当的时候使用多行字符串。

总之，`tomli_w`是编写 TOML 文档的一个很好的选择，只要您不需要对输出进行很多控制。在下一节中，您将使用`tomlkit`，如果需要的话，它会给您更多的控制权。您将从头开始创建一个专用的 TOML 文档对象，而不是简单地将字典转储到 TOML。

[*Remove ads*](/account/join/)

## 创建新的 TOML 文档

你知道如何用`tomli`和`tomli_w`快速读写 TOML 文档。您还注意到了`tomli_w`的一些局限性，尤其是在格式化生成的 TOML 文件时。

在这一节中，您将首先探索如何格式化 TOML 文档，使它们更易于用户使用。然后，您将尝试另一个名为`tomlkit`的库，您可以用它来完全控制您的 TOML 文档。

### TOML 文档的格式和样式

一般来说，空白在 TOML 文件中会被忽略。您可以利用这一点来使您的配置文件组织良好、易读和直观。此外，散列符号(`#`)将该行的其余部分标记为注释。自由地使用它们。

没有针对 TOML 文档的样式指南，也就是说 [PEP 8](https://realpython.com/python-pep8/) 是针对 Python 代码的样式指南。然而，[规范](https://toml.io/en/v1.0.0)确实包含了一些建议，同时也为你留下了一些风格方面的选择。

TOML 中的一些特性非常灵活。例如，您可以任意顺序定义表格。因为表名是完全限定的，所以您甚至可以在父表之前定义子表。此外，键周围的空白被忽略。标题`[nested.table]`和`[ nested . table]`从同一个嵌套表开始。

TOML 规范中的建议可以总结为**不要滥用灵活性。保持你对一致性和可读性的关注，你和你的用户会更开心！**

要查看样式选项列表，您可以根据个人偏好做出合理的选择，请查看 Taplo 格式化程序可用的[配置选项](https://taplo.tamasfe.dev/configuration/formatter-options.html)。以下是一些你可以思考的问题:

*   **缩进子表**还是只依靠**表头**来表示结构？
*   **在每个表中对齐键-值对中的等号**还是始终坚持在等号的每一侧**留一个空格？**
*   **将长数组**分割成多行还是总是将它们集中在**一行**？
*   **在多行数组的最后一个值后添加一个尾随逗号**或**让它保持空白**？
*   对表和键**按语义**或**按字母顺序**排序？

每一个选择都取决于个人品味，所以请随意尝试，找到你觉得舒服的东西。

尽管如此，努力保持一致还是有好处的。为了保持一致性，您可以在项目中使用类似于 [Taplo](https://taplo.tamasfe.dev/) 的格式化程序，并将其配置文件包含在您的[版本控制](https://realpython.com/python-git-github-intro/)中。你也可以将[集成到你的编辑器中。](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml)

回头看看上面的问题。如果使用`tomli_w`编写 TOML 文档，那么唯一可以选择的问题就是如何对表和键进行排序。如果你想更好地控制你的文档，那么你需要一个不同的工具。在下一小节中，您将开始关注`tomlkit`，它赋予您更多的权力和责任。

### 用`tomlkit` 从头开始创建 TOML

[TOML Kit](https://pypi.org/project/tomlkit/) 最初是为[诗歌](https://realpython.com/dependency-management-python-poetry/)项目打造的。作为其依赖性管理的一部分，poem 操作`pyproject.toml`文件。然而，由于这个文件用于[多种用途](https://peps.python.org/pep-0518/#tool-table)，诗歌必须保留文件中的风格和注释。

在这一小节中，您将使用`tomlkit`从头开始创建一个 TOML 文档，以便使用它的一些功能。首先，您需要将软件包安装到您的虚拟环境中:

```py
(venv) $ python -m pip install tomlkit
```

你可以从确认`tomlkit`比`tomli`和`tomli_w`更强大开始。重复前面的往返示例，注意所有的格式都保留了下来:

>>>

```py
>>> import tomlkit
>>> toml_data = """
... [nested]  # Not necessary
... ...     [nested.table]
...     string       = "Hello, TOML!"
...     weird_string = '''Literal
...         Multiline'''
... """
>>> print(tomlkit.dumps(tomlkit.loads(toml_data)))

[nested]  # Not necessary

 [nested.table]
 string       = "Hello, TOML!"
 weird_string = '''Literal
 Multiline'''

>>> tomlkit.dumps(tomlkit.loads(toml_data)) == toml_data
True
```

你可以像前面一样使用`loads()`和`dumps()`——`load()`和`dump()`来读写 TOML。但是，现在所有的字符串类型、缩进、注释和对齐方式都保留了下来。

为了实现这一点，`tomlkit`使用了定制的数据类型，其行为或多或少类似于您的本地 Python 类型。稍后你会学到更多关于这些数据类型的知识。首先，您将看到如何从头开始创建一个 TOML 文档:

>>>

```py
>>> from tomlkit import comment, document, nl, table

>>> toml = document()
>>> toml.add(comment("Written by TOML Kit"))
>>> toml.add(nl())
>>> toml.add("board_size", 3)
```

一般来说，你需要通过调用`document()`来创建一个 TOML 文档实例。然后，您可以使用`.add()`向这个文档添加不同的对象，比如注释、换行符、键值对和表格。

**注意:**调用`.add()`返回更新后的对象。在本节的示例中您不会看到这一点，因为额外的输出会分散示例流的注意力。[稍后](#read-and-write-toml-losslessly)您将看到如何利用这一设计，并将几个调用链接到一起`.add()`。

你可以使用上面的`dump()`或`dumps()`将`toml`转换成一个实际的 TOML 文档，或者你可以使用`.as_string()`方法:

>>>

```py
>>> print(toml.as_string())
# Written by TOML Kit

board_size = 3
```

在本例中，您开始重新创建之前使用过的井字游戏配置的各个部分。注意输出中的每一行如何对应到代码中的一个`.add()`方法。首先是注释，然后是代表空行的`nl()`，然后是键值对。

继续您的示例，添加几个表格:

>>>

```py
>>> player_x = table()
>>> player_x.add("symbol", "X")
>>> player_x.add("color", "blue")
>>> player_x.comment("Start player")
>>> toml.add("player_x", player_x)

>>> player_o = table()
>>> player_o.update({"symbol": "O", "color": "green"})
>>> toml["player_o"] = player_o
```

您可以通过调用`table()`来创建表格，并向其中添加内容。创建了一个表格后，就可以将它添加到 TOML 文档中。您可以坚持使用`.add()`来组合您的文档，但是这个例子也展示了一些添加内容的替代方法。例如，您可以使用`.update()`直接从字典中添加键和值。

当您将文档转换为 TOML 字符串时，它将如下所示:

>>>

```py
>>> print(toml.as_string())
# Written by TOML Kit

board_size = 3

[player_x] # Start player
symbol = "X"
color = "blue"

[player_o]
symbol = "O"
color = "green"
```

将此输出与您用来创建文档的命令进行比较。如果您正在创建一个具有固定结构的 TOML 文档，那么将文档写成一个 TOML 字符串并用`tomlkit`加载它可能更容易。然而，您在上面看到的命令在动态组合配置时为您提供了很大的灵活性。

在下一节中，您将更深入地研究`tomlkit`,看看如何使用它来更新现有的配置。

## 更新现有的 TOML 文档

假设您已经花了一些时间将一个组织良好的配置和良好的注释放在一起，指导您的用户如何更改它。然后一些其他的应用程序出现并把它的配置存储在同一个文件中，同时破坏你精心制作的艺术品。

这可能是将您的配置保存在一个其他人不会接触到的专用文件中的一个理由。然而，有时使用公共配置文件也很方便。 [`pyproject.toml`](https://realpython.com/courses/packaging-with-pyproject-toml/) 文件就是这样一个通用文件，尤其是对于开发和构建包时使用的工具。

在这一节中，您将深入了解`tomlkit`如何表示 TOML 对象，以及如何使用这个包来更新现有的 TOML 文件。

### 将 TOML 表示为`tomlkit`对象

在前面，您看到了`tomli`和`tomllib`将 TOML 文档解析成本地 Python 类型，如字符串、整数和字典。你已经看到一些迹象表明`tomlkit`是不同的。现在，是时候仔细看看`tomlkit`如何表示一个 TOML 文档了。

首先，复制并保存下面的 TOML 文件为`tic-tac-toe-config.toml`:

```py
# tic-tac-toe-config.toml board_size  =  3 [user] ai_skill  =  0.85  # A number between 0 (random) and 1 (expert) [user.player_x] symbol  =  "X" color  =  "blue" ai  =  true [user.player_o] symbol  =  "O" color  =  "green" ai  =  false # Settings used when deploying the application [server] url  =  "https://tictactoe.example.com"
```

打开 REPL 会话并用`tomlkit`加载此文档:

>>>

```py
>>> import tomlkit
>>> with open("tic-tac-toe-config.toml", mode="rt", encoding="utf-8") as fp:
...     config = tomlkit.load(fp)
...
>>> config
{'board_size': 3, 'user': {'ai_skill': 0.85, 'player_x': { ... }}}

>>> type(config)
<class 'tomlkit.toml_document.TOMLDocument'>
```

使用`load()`从文件中加载 TOML 文档。看`config`的时候，第一眼就像一本字典。然而，深入挖掘，你会发现这是一种特殊的`TOMLDocument`类型。

**注意:**与`tomli`不同，`tomlkit`希望你以文本模式打开文件。你还应该记得指定文件应该使用`utf-8`编码来打开。

这些自定义数据类型的行为或多或少类似于您的本地 Python 类型。例如，您可以使用方括号(`[]`)访问文档中的子表和值，就像字典一样。继续上面的例子:

>>>

```py
>>> config["user"]["player_o"]["color"]
'green'

>>> type(config["user"]["player_o"]["color"])
<class 'tomlkit.items.String'>

>>> config["user"]["player_o"]["color"].upper()
'GREEN'
```

尽管这些值也是特殊的`tomlkit`数据类型，但是您可以像处理普通的 Python 类型一样处理它们。例如，您可以使用`.upper()`字符串方法。

特殊数据类型的一个优点是，它们允许您访问关于文档的元信息，包括注释和缩进:

>>>

```py
>>> config["user"]["ai_skill"]
0.85

>>> config["user"]["ai_skill"].trivia.comment
'# A number between 0 (random) and 1 (expert)'

>>> config["user"]["player_x"].trivia.indent
'    '
```

例如，您可以通过`.trivia`访问器恢复注释和缩进信息。

正如您在上面看到的，您可以将这些特殊对象视为本地 Python 对象。事实上，他们从本地的同类那里继承了 T2。但是，如果您真的需要，您可以使用`.unwrap()`将它们转换成普通的 Python:

>>>

```py
>>> config["board_size"] ** 2
9

>>> isinstance(config["board_size"], int)
True

>>> config["board_size"].unwrap()
3

>>> type(config["board_size"].unwrap())
<class 'int'>
```

在调用了`.unwrap()`之后，`3`现在是一个普通的 Python 整数。总之，这个调查让你对`tomlkit`如何能够保持 TOML 文档的风格有了一些了解。

在下一小节中，您将了解如何使用`tomlkit`数据类型来定制 TOML 文档，而不影响现有的样式。

### 无损读写 TOML】

您知道`tomlkit`表示使用定制类的 TOML 文档，并且您已经看到如何从头开始创建这些对象，以及如何读取现有的 TOML 文档。在这一小节中，您将加载一个现有的 TOML 文件，并在将其写回磁盘之前对其进行一些更改。

首先加载您在上一小节中使用的同一个 TOML 文件:

>>>

```py
>>> import tomlkit
>>> with open("tic-tac-toe-config.toml", mode="rt", encoding="utf-8") as fp:
...     config = tomlkit.load(fp)
...
```

正如您之前看到的，`config`现在是一个`TOMLDocument`。您可以使用`.add()`向其中添加新元素，就像您从头开始创建文档时一样。但是，您不能使用`.add()`来更新现有键的值:

>>>

```py
>>> config.add("app_name", "Tic-Tac-Toe")
{'board_size': 3, 'app_name': 'Tic-Tac-Toe', 'user': { ... }}

>>> config["user"].add("ai_skill", 0.6)
Traceback (most recent call last):
  ...
KeyAlreadyPresent: Key "ai_skill" already exists.
```

你试图降低人工智能的技能，这样你就有一个更容易对付的对手。但是，你不能用`.add()`做到这一点。相反，您可以分配新值，就像`config`是一个常规字典一样:

>>>

```py
>>> config["user"]["ai_skill"] = 0.6 >>> print(config["user"].as_string())
ai_skill = 0.6  # A number between 0 (random) and 1 (expert) 
 [user.player_x]
 symbol = "X"
 color = "blue"
 ai = true

 [user.player_o]
 symbol = "O"
 color = "green"
 ai = false
```

当您像这样更新一个值时，`tomlkit`仍然会注意保留样式和注释。如你所见，关于`ai_skill`的评论没有被改动。

部分`tomlkit`支持所谓的[流畅界面](https://en.wikipedia.org/wiki/Fluent_interface)。实际上，这意味着像`.add()`这样的操作会返回更新后的对象，这样你就可以在其上链接另一个对`.add()`的调用。当您需要构造包含多个字段的表时，可以利用这一点:

>>>

```py
>>> from tomlkit import aot, comment, inline_table, nl, table
>>> player_data = [
...     {"user": "gah", "first_name": "Geir Arne", "last_name": "Hjelle"},
...     {"user": "tompw", "first_name": "Tom", "last_name": "Preston-Werner"},
... ]

>>> players = aot()
>>> for player in player_data:
...     players.append(
...         table()
...         .add("username", player["user"])
...         .add("name",
...             inline_table()
...             .add("first", player["first_name"])
...             .add("last", player["last_name"])
...         )
...     )
...
>>> config.add(nl()).add(comment("Players")).add("players", players)
```

在本例中，您创建了一个包含球员信息的表数组。首先用`aot()`构造函数创建一个空的表数组。然后循环遍历玩家数据，将每个玩家添加到数组中。

您使用[方法链接](https://en.wikipedia.org/wiki/Method_chaining)来创建每个玩家表。实际上，您的调用是`table().add().add()`,它将两个元素添加到一个新表中。最后，在配置的底部，在一个简短的注释下面添加新的玩家表数组。

对配置的更新完成后，您现在可以将它写回同一个文件:

>>>

```py
>>> with open("tic-tac-toe-config.toml", mode="wt", encoding="utf-8") as fp:
...     tomlkit.dump(config, fp)
```

打开`tic-tac-toe-config.toml`，注意你的更新已经包含在内。与此同时，原有的风格得以保留:

```py
# tic-tac-toe-config.toml board_size  =  3 app_name  =  "Tic-Tac-Toe"  
[user] ai_skill  =  0.6  # A number between 0 (random) and 1 (expert)  
  [user.player_x] symbol  =  "X" color  =  "blue" ai  =  true [user.player_o] symbol  =  "O" color  =  "green" ai  =  false # Settings used when deploying the application [server] url  =  "https://tictactoe.example.com" # Players  
[[players]]  username  =  "gah"  name  =  {first  =  "Geir Arne",  last  =  "Hjelle"}  
[[players]]  username  =  "tompw"  name  =  {first  =  "Tom",  last  =  "Preston-Werner"}
```

请注意，`app_name`已经被添加，`user.ai_skill`的值已经被更新，`players`表的数组已经被附加到您的配置的末尾。您已经成功地以编程方式更新了您的配置。

## 结论

这是您对 TOML 格式以及如何在 Python 中使用它的广泛探索的结束。您已经看到了一些使 TOML 成为一种灵活方便的配置文件格式的特性。同时，您还发现了一些限制其在其他应用程序(如数据序列化)中使用的局限性。

**在本教程中，您已经:**

*   了解了 **TOML 语法**及其支持的数据类型
*   **用`tomli`和`tomllib`解析** TOML 文档
*   **用`tomli_w`编写** TOML 文档
*   无损的**用`tomlkit`更新了**的 TOML 文件

您是否有需要方便配置的应用程序？汤姆可能就是你要找的人。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。**********