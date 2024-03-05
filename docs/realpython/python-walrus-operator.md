# Walrus 运算符:Python 3.8 赋值表达式

> 原文：<https://realpython.com/python-walrus-operator/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python 赋值表达式和使用海象运算符**](/courses/python-assignment-expressions-walrus-operator/)

Python 的每个新版本都为该语言添加了新的特性。对于 Python 3.8，最大的变化是增加了**赋值表达式**。具体来说，`:=`操作符为在表达式中间分配变量提供了一种新的语法。这位操作员俗称**海象操作员**。

本教程是对 walrus 操作符的深入介绍。您将了解语法更新的一些动机，并探索赋值表达式有用的一些例子。

**在本教程中，您将学习如何:**

*   识别**海象运算符**并理解其含义
*   了解海象运营商的**用例**
*   **使用 walrus 运算符避免重复代码**
*   在使用 walrus 运算符的代码和使用**其他赋值方法**的代码之间转换
*   理解使用 walrus 操作符时对**向后兼容性**的影响
*   在赋值表达式中使用合适的**样式**

请注意，本教程中的所有 walrus 操作符示例都需要使用 [Python 3.8](https://realpython.com/python38-new-features/) 或更高版本。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 海象运营商基础知识

让我们从程序员用来指代这种新语法的一些不同术语开始。您已经在本教程中看到了一些。

`:=`运算符的正式名称是**赋值表达式运算符**。在早期的讨论中，它被称为**海象操作符**，因为`:=`语法类似于侧卧的[海象](https://en.wikipedia.org/wiki/Walrus)的眼睛和长牙。您可能还会看到被称为**冒号等于运算符**的`:=`运算符。用于赋值表达式的另一个术语是**命名表达式**。

[*Remove ads*](/account/join/)

### 你好，海象！

为了获得关于赋值表达式的第一印象，请启动您的 REPL，使用下面的代码:

>>>

```py
 1>>> walrus = False
 2>>> walrus
 3False
 4
 5>>> (walrus := True)
 6True
 7>>> walrus
 8True
```

第 1 行显示了传统的[赋值语句](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements)，其中值`False`被赋值给`walrus`。接下来，在第 5 行，使用赋值表达式将值`True`赋给`walrus`。在第 1 行和第 5 行之后，您可以通过使用变量名`walrus`来引用赋值。

您可能想知道为什么在第 5 行使用括号，稍后在本教程中您将了解为什么需要括号[。](#walrus-operator-syntax)

**注意:**Python 中的一个**语句**是一个代码单位。一个**表达式**是一个可以被赋值的特殊语句。

例如，`1 + 2`是一个计算值为`3`的表达式，而`number = 1 + 2`是一个不计算值的赋值语句。虽然运行语句`number = 1 + 2`不会计算出`3`，但是*会将值`3`分配给`number`。*

在 Python 中，你经常会看到[简单语句](https://docs.python.org/3/reference/simple_stmts.html)像 [`return`语句](https://realpython.com/python-return-statement/)[`import`语句](https://realpython.com/python-import/)，还有[复合语句](https://docs.python.org/3/reference/compound_stmts.html)像 [`if`语句](https://realpython.com/python-conditional-statements/)和[函数定义](https://realpython.com/defining-your-own-python-function/)。这些都是陈述，不是表达。

前面看到的使用`walrus`变量的两种类型的赋值之间有一个微妙但重要的区别。赋值表达式返回值，而传统的赋值不返回值。当 REPL 在第 1 行的`walrus = False`后不打印任何值，而在第 5 行的赋值表达式后打印出`True`时，您可以看到这一点。

在这个例子中，您可以看到关于 walrus 操作符的另一个重要方面。虽然看起来很新，但是`:=`操作符做了*而不是*没有它就不可能做的任何事情。它只是使某些构造更加方便，有时可以更清楚地传达代码的意图。

**注意:**你至少需要 [Python 3.8](https://realpython.com/python38-new-features/) 来试用本教程中的例子。如果你还没有安装 Python 3.8，并且你有可用的 [Docker](https://docs.docker.com/install/) ，开始使用 Python 3.8 的一个快速方法是运行官方 Docker 镜像中的[:](https://hub.docker.com/_/python/)

```py
$ docker container run -it --rm python:3.8-slim
```

这将下载并运行 Python 3.8 的最新稳定版本。有关更多信息，请参见 Docker 中的[运行 Python 版本:如何尝试最新的 Python 版本](https://realpython.com/python-versions-docker/)。

现在您对`:=`操作符是什么以及它能做什么有了一个基本的概念。它是赋值表达式中使用的操作符，可以返回被赋值的值，不像传统的赋值语句。要更深入地真正了解 walrus 操作符，请继续阅读，看看哪些地方应该使用，哪些地方不应该使用。

### 实施

像 Python 中的大多数新特性一样，赋值表达式是通过 **Python 增强提案** (PEP)引入的。 [PEP 572](https://www.python.org/dev/peps/pep-0572) 描述了引入 walrus 操作符的动机、语法细节，以及可以使用`:=`操作符改进代码的例子。

这个 PEP 最初是由[克里斯·安吉利科](https://twitter.com/Rosuav)在 2018 年 2 月写的[。经过一番激烈的讨论，PEP 572 于 2018 年 7 月被](https://mail.python.org/archives/list/python-ideas@python.org/message/H64ZNZ3T4RRJKMXR6UFNX3FK62IRPVOT/)[的](https://mail.python.org/archives/list/python-dev@python.org/message/J6EBK6ZEHZXTVWYSUO5N5XCUS45UQSB3/)[吉多·范·罗苏姆](https://twitter.com/gvanrossum)接受。从那时起，圭多[宣布](https://mail.python.org/archives/list/python-committers@python.org/message/GQONAGWBBFRHVRUPU7RNBM75MHKGUFJN/)他将辞去[BDFL](https://en.wikipedia.org/wiki/Benevolent_dictator_for_life)终身仁慈独裁者的角色。从 2019 年初开始，Python 已经由选举产生的[指导委员会](https://realpython.com/python38-new-features/#the-python-steering-council)管理[。](https://www.python.org/dev/peps/pep-0013/)

walrus 操作符是由 Emily Morehouse 实现的，并在 Python 3.8 的第一个 T2 alpha 版本中可用。

### 动机

在许多语言中，包括 C 语言及其派生语言，赋值语句的作用相当于表达式。这可能是非常强大的，也是令人困惑的错误的来源。例如，下面的代码是有效的 C，但没有按预期执行:

```py
int  x  =  3,  y  =  8; if  (x  =  y)  { printf("x and y are equal (x = %d, y = %d)",  x,  y); }
```

在这里，`if (x = y)`将计算为 true，代码片段将打印出`x and y are equal (x = 8, y = 8)`。这是你期待的结果吗？你试图比较`x`和`y`。`x`的值是怎么从`3`变成`8`的？

问题是您使用了赋值操作符(`=`)而不是相等比较操作符(`==`)。在 C 语言中，`x = y`是一个计算结果为`y`的表达式。在本例中，`x = y`被评估为`8`，这在`if`语句的上下文中被视为[真值](https://realpython.com/python-operators-expressions/#evaluation-of-non-boolean-values-in-boolean-context)。

看看 Python 中相应的例子。这段代码引出了 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) :

```py
x, y = 3, 8
if x = y:
    print(f"x and y are equal ({x = }, {y = })")
```

与 C 示例不同，这段 Python 代码给出了一个显式错误，而不是 bug。

Python 中赋值语句和赋值表达式之间的区别对于避免这类难以发现的错误非常有用。PEP 572 [认为](https://www.python.org/dev/peps/pep-0572/#frequently-raised-objections)Python 更适合赋值语句和表达式有不同的语法，而不是把现有的赋值语句变成表达式。

支撑 walrus 操作符的一个设计原则是，使用`=`操作符的赋值语句和使用`:=`操作符的赋值表达式在不同的代码上下文中都是有效的。例如，您不能用 walrus 运算符进行简单的赋值:

>>>

```py
>>> walrus := True
  File "<stdin>", line 1
    walrus := True
           ^
SyntaxError: invalid syntax
```

在许多情况下，您可以在赋值表达式两边添加括号(`()`)，使其成为有效的 Python:

>>>

```py
>>> (walrus := True)  # Valid, but regular statements are preferred
True
```

在这样的圆括号中不允许写带有`=`的传统赋值语句。这有助于您捕捉潜在的错误。

[在本教程](#walrus-operator-syntax)的后面，你将了解更多关于不允许使用 walrus 操作符的情况，但首先你将了解你可能想要使用它们的情况。

[*Remove ads*](/account/join/)

## 海象运营商用例

在本节中，您将看到 walrus 操作符可以简化代码的几个例子。所有这些例子的一个普遍主题是，你将避免不同类型的重复:

*   **重复的函数调用**会让你的代码比必要的要慢。
*   重复的语句会让你的代码难以维护。
*   重复调用穷举迭代器会使你的代码过于复杂。

您将看到海象操作员如何在这些情况下提供帮助。

### 调试

可以说，walrus 操作符的最佳用例之一是调试复杂表达式。假设您想要找出地球表面上两个位置之间的距离。一种方法是使用[哈弗辛公式](https://en.wikipedia.org/wiki/Haversine_formula):

![The haversine formula](img/e0c4cda05d3ebebb5086af20cac40d74.png)

*ϕ* 代表纬度 *λ* 代表每个位置的经度。为了演示这个公式，你可以计算出[奥斯陆](https://en.wikipedia.org/wiki/Oslo) (59.9 N 10.8 E)和[温哥华](https://en.wikipedia.org/wiki/Vancouver) (49.3 N 123.1 W)之间的距离如下:

>>>

```py
>>> from math import asin, cos, radians, sin, sqrt

>>> # Approximate radius of Earth in kilometers
>>> rad = 6371

>>> # Locations of Oslo and Vancouver
>>> ϕ1, λ1 = radians(59.9), radians(10.8)
>>> ϕ2, λ2 = radians(49.3), radians(-123.1)

>>> # Distance between Oslo and Vancouver
>>> 2 * rad * asin(
...     sqrt(
...         sin((ϕ2 - ϕ1) / 2) ** 2
...         + cos(ϕ1) * cos(ϕ2) * sin((λ2 - λ1) / 2) ** 2
...     )
... )
...
7181.7841229421165
```

正如你所看到的，从奥斯陆到温哥华的距离不到 7200 公里。

**注意:** Python 源代码通常使用 [UTF-8 Unicode](https://realpython.com/python-encodings-guide/#python-3-all-in-on-unicode) 编写。这允许您在代码中使用类似于`ϕ`和`λ`的希腊字母，这在翻译数学公式时可能很有用。Wikipedia 展示了在您的系统上使用 Unicode 的一些替代方法。

虽然支持 UTF-8(例如，在字符串中)，Python 的变量名使用更受[限制的字符集](https://www.python.org/dev/peps/pep-3131/#specification-of-language-changes)。例如，你[不能](https://github.com/python/cpython/pull/1686)在给你的变量命名时使用表情符号。那是一个[好的限制](https://github.com/gahjelle/pythonji/)！

现在，假设您需要仔细检查您的实现，并想看看[哈弗辛项](https://en.wikipedia.org/wiki/Versine#Haversine)对最终结果有多大贡献。您可以从您的主代码中复制并粘贴该术语，以单独评估它。但是，您也可以使用`:=`操作符为您感兴趣的子表达式命名:

>>>

```py
>>> 2 * rad * asin(
...     sqrt(
...         (ϕ_hav := sin((ϕ2 - ϕ1) / 2) ** 2) ...         + cos(ϕ1) * cos(ϕ2) * sin((λ2 - λ1) / 2) ** 2
...     )
... )
...
7181.7841229421165

>>> ϕ_hav
0.008532325425222883
```

这里使用 walrus 操作符的好处是，您可以计算完整表达式的值，同时跟踪`ϕ_hav`的值。这允许您确认在调试时没有引入任何错误。

### 列表和词典

[列表](https://realpython.com/python-lists-tuples/)是 Python 中强大的数据结构，通常表示一系列相关的属性。类似地，[字典](https://realpython.com/python-dicts/)在 Python 中广泛使用，对于结构化信息非常有用。

有时，在建立这些数据结构时，您最终会多次执行相同的操作。作为第一个例子，计算一列数字的一些基本的[描述性统计数据](https://realpython.com/python-statistics/)，并将它们存储在字典中:

>>>

```py
>>> numbers = [2, 8, 0, 1, 1, 9, 7, 7]

>>> description = {
...     "length": len(numbers),
...     "sum": sum(numbers),
...     "mean": sum(numbers) / len(numbers),
... }

>>> description
{'length': 8, 'sum': 35, 'mean': 4.375}
```

注意,`numbers`列表的总和和长度都被计算了两次。在这个简单的例子中，结果并不太糟糕，但是如果列表更大或者计算更复杂，您可能需要优化代码。为此，您可以首先将函数调用移出字典定义:

>>>

```py
>>> numbers = [2, 8, 0, 1, 1, 9, 7, 7]

>>> num_length = len(numbers)
>>> num_sum = sum(numbers)

>>> description = {
...     "length": num_length,
...     "sum": num_sum,
...     "mean": num_sum / num_length,
... }

>>> description
{'length': 8, 'sum': 35, 'mean': 4.375}
```

变量`num_length`和`num_sum`仅用于优化字典内的计算。通过使用 walrus 操作符，这个角色可以变得更加清晰:

>>>

```py
>>> numbers = [2, 8, 0, 1, 1, 9, 7, 7]

>>> description = {
...     "length": (num_length := len(numbers)),
...     "sum": (num_sum := sum(numbers)),
...     "mean": num_sum / num_length,
... }

>>> description
{'length': 8, 'sum': 35, 'mean': 4.375}
```

`num_length`和`num_sum`现在被定义在`description`的定义内。对于阅读这段代码的人来说，这是一个明确的暗示，这些变量只是用来优化这些计算，以后不会再使用。

**注意:**`num_length`和`num_sum`变量的[范围](https://realpython.com/python-scope-legb-rule/)在有 walrus 操作符的例子和没有 walrus 操作符的例子中是相同的。这意味着在这两个例子中，变量都是在定义了`description`之后才可用的。

尽管这两个例子在功能上非常相似，但是使用赋值表达式的一个好处是，`:=`操作符传达了这些变量的**意图**作为一次性优化。

在下一个例子中，您将使用 [`wc`实用程序](https://en.wikipedia.org/wiki/Wc_%28Unix%29)的基本实现来计算文本文件中的行、单词和字符:

```py
 1# wc.py
 2
 3import pathlib
 4import sys
 5
 6for filename in sys.argv[1:]:
 7    path = pathlib.Path(filename)
 8    counts = (
 9        path.read_text().count("\n"),  # Number of lines
10        len(path.read_text().split()),  # Number of words
11        len(path.read_text()),  # Number of characters
12    )
13    print(*counts, path)
```

这个脚本可以读取一个或几个文本文件，并报告每个文件包含多少行、单词和字符。下面是代码中发生的事情的分类:

*   **第 6 行**遍历用户提供的每个文件名。`sys.argv`是一个列表，包含命令行中给出的每个参数，以脚本名开始。关于`sys.argv`的更多信息，可以查看 [Python 命令行参数](https://realpython.com/python-command-line-arguments/#the-sysargv-array)。
*   **第 7 行**将每个文件名字符串翻译成一个[对象](https://realpython.com/python-pathlib/)。在一个`Path`对象中存储一个文件名可以让你方便地阅读下一行的文本文件。
*   **第 8 行到第 12 行**构建一个计数元组来表示一个文本文件中的行数、单词数和字符数。
*   **Line 9** 读取一个文本文件，通过计算新行来计算行数。
*   **第 10 行**读取一个文本文件，通过分割空白来计算字数。
*   **第 11 行**读取一个文本文件，通过查找字符串的长度来计算字符数。
*   **第 13 行**将所有三个计数连同文件名一起打印到控制台。`*counts`语法[解包`counts`元组](https://realpython.com/python-kwargs-and-args/#unpacking-with-the-asterisk-operators)。在这种情况下， [`print()`语句](https://realpython.com/python-print/)相当于`print(counts[0], counts[1], counts[2], path)`。

要查看`wc.py`的运行，您可以使用脚本本身，如下所示:

```py
$ python wc.py wc.py
13 34 316 wc.py
```

换句话说，`wc.py`文件由 13 行、34 个单词和 316 个字符组成。

如果您仔细观察这个实现，您会注意到它远非最佳。特别是，对`path.read_text()`的调用重复了三次。这意味着每个文本文件被读取三次。您可以使用 walrus 运算符来避免重复:

```py
# wc.py

import pathlib
import sys

for filename in sys.argv[1:]:
    path = pathlib.Path(filename)
    counts = [
 (text := path.read_text()).count("\n"),  # Number of lines        len(text.split()),  # Number of words
        len(text),  # Number of characters
    ]
    print(*counts, path)
```

文件的内容被分配给`text`，在接下来的两次计算中被重用。该程序的功能仍然相同:

```py
$ python wc.py wc.py
13 36 302 wc.py
```

与前面的例子一样，另一种方法是在定义`counts`之前定义`text`:

```py
# wc.py

import pathlib
import sys

for filename in sys.argv[1:]:
    path = pathlib.Path(filename)
 text = path.read_text()    counts = [
        text.count("\n"),  # Number of lines
        len(text.split()),  # Number of words
        len(text),  # Number of characters
    ]
    print(*counts, path)
```

虽然这比前一个实现多了一行，但它可能提供了可读性和效率之间的最佳平衡。赋值表达式操作符并不总是可读性最好的解决方案，即使它使你的代码更加简洁。

[*Remove ads*](/account/join/)

### 列出理解

[列表理解](https://realpython.com/list-comprehension-python/)对于构建和过滤列表非常有用。它们清楚地陈述了代码的意图，并且通常运行得相当快。

有一个列表理解用例，其中 walrus 操作符特别有用。假设您想要对列表中的元素应用一些计算量很大的函数`slow()`，并对结果值进行过滤。您可以像下面这样做:

```py
numbers = [7, 6, 1, 4, 1, 8, 0, 6]

results = [slow(num) for num in numbers if slow(num) > 0]
```

在这里，您过滤了`numbers`列表，留下了应用`slow()`的正面结果。这段代码的问题是这个昂贵的函数被调用了两次。

对于这种情况，一个非常常见的解决方案是重写您的代码，使用一个显式的`for`循环:

```py
results = []
for num in numbers:
    value = slow(num)
    if value > 0:
        results.append(value)
```

这个只会调用`slow()`一次。不幸的是，代码现在变得更加冗长，代码的意图也更加难以理解。列表理解清楚地表明您正在创建一个新列表，而这更多地隐藏在显式的`for`循环中，因为几行代码将列表创建和`.append()`的使用分开。此外，列表理解比重复调用`.append()`运行得更快。

你可以通过使用一个 [`filter()`表达式](https://realpython.com/python-filter-function/)或者一种双列表理解来编写一些其他的解决方案:

```py
# Using filter
results = filter(lambda value: value > 0, (slow(num) for num in numbers))

# Using a double list comprehension
results = [value for num in numbers for value in [slow(num)] if value > 0]
```

好消息是每个号码只能调用一次`slow()`。坏消息是代码的可读性在两个表达式中都受到了影响。

弄清楚在双列表理解中实际发生了什么需要相当多的挠头。本质上，第二个`for`语句仅用于给`slow(num)`的返回值命名为`value`。幸运的是，这听起来像是可以用赋值表达式来执行的事情！

您可以使用 walrus 运算符重写列表理解，如下所示:

```py
results = [value for num in numbers if (value := slow(num)) > 0]
```

请注意，`value := slow(num)`两边的括号是必需的。这个版本是有效的、可读的，并且很好地传达了代码的意图。

**注意:**你需要在列表理解的`if`子句上添加赋值表达式。如果您试图用对`slow()`的另一个调用来定义`value`，那么它将不起作用:

>>>

```py
>>> results = [(value := slow(num)) for num in numbers if value > 0]
NameError: name 'value' is not defined
```

这将引发一个`NameError`,因为在理解开始时，在表达式之前评估了`if`子句。

让我们看一个稍微复杂一点的实际例子。说要用 [*真蟒*提要](https://realpython.com/contact/#rss-atom-feed)找[真蟒播客](https://realpython.com/podcasts/rpp/)最后几集的标题。

您可以使用 [Real Python Feed 阅读器](https://pypi.org/project/realpython-reader/)下载关于最新 *Real Python* 出版物的信息。为了找到播客的剧集标题，你将使用第三方[解析](https://realpython.com/python-packages/#parse-for-matching-strings)包。首先将两者安装到您的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中:

```py
(venv) $ python -m pip install realpython-reader parse
```

您现在可以阅读由 *Real Python* 发布的最新标题:

>>>

```py
>>> from reader import feed

>>> feed.get_titles()
['The Walrus Operator: Python 3.8 Assignment Expressions',
 'The Real Python Podcast – Episode #63: Create Web Applications Using Anvil',
 'Context Managers and Python's with Statement',
 ...]
```

播客标题以`"The Real Python Podcast"`开头，所以您可以在这里创建一个模式，Parse 可以使用它来识别它们:

>>>

```py
>>> import parse

>>> pattern = parse.compile(
...     "The Real Python Podcast – Episode #{num:d}: {name}"
... )
```

预先编译模式可以加快以后的比较，尤其是当您想要反复匹配相同的模式时。您可以使用`pattern.parse()`或`pattern.search()`来检查字符串是否匹配您的模式:

>>>

```py
>>> pattern.parse(
...     "The Real Python Podcast – Episode #63: "
...     "Create Web Applications Using Anvil"
... )
...
<Result () {'num': 63, 'name': 'Create Web Applications Using Anvil'}>
```

注意，Parse 能够挑选出播客的集号和集名。因为您使用了`:d` [格式说明符](https://github.com/r1chardj0n3s/parse#format-specification)，所以剧集编号被转换为[整数](https://realpython.com/python-numbers/#integers)数据类型。

让我们回到手头的任务上来。为了列出所有最近的播客标题，您需要检查每个字符串是否匹配您的模式，然后解析出剧集标题。第一次尝试可能是这样的:

>>>

```py
>>> import parse
>>> from reader import feed

>>> pattern = parse.compile(
...     "The Real Python Podcast – Episode #{num:d}: {name}"
... )

>>> podcasts = [ ...     pattern.parse(title)["name"]
...     for title in feed.get_titles() ...     if pattern.parse(title)
... ]

>>> podcasts[:3]
['Create Web Applications Using Only Python With Anvil',
 'Selecting the Ideal Data Structure & Unravelling Python\'s "pass" and "with"',
 'Scaling Data Science and Machine Learning Infrastructure Like Netflix']
```

尽管它可以工作，但您可能会注意到之前看到的相同问题。您对每个标题进行了两次解析，因为您过滤掉了与您的模式匹配的标题，然后使用相同的模式来挑选剧集标题。

就像你之前做的那样，你可以通过使用一个显式的`for`循环或者一个双列表理解来重写列表理解，从而避免双重工作。然而，使用 walrus 操作符更加简单:

>>>

```py
>>> podcasts = [
...     podcast["name"]
...     for title in feed.get_titles()
...     if (podcast := pattern.parse(title))
... ]
```

赋值表达式可以很好地简化这类列表理解。它们帮助您保持代码的可读性，同时避免两次执行潜在的昂贵操作。

**注意:**真正的 Python 播客有自己独立的 [RSS 提要](https://realpython.com/podcasts/rpp/feed)，如果你只想了解播客的信息，你应该使用它。你可以用下面的代码得到所有的剧集标题:

```py
from reader import feed

podcasts = feed.get_titles("https://realpython.com/podcasts/rpp/feed")
```

请参见[真正的 Python 播客](https://realpython.com/podcasts/rpp/)，了解使用您的播客播放器收听该播客的选项。

在本节中，您已经关注了使用 walrus 操作符重写列表理解的例子。如果你发现你需要在一个[字典理解](https://realpython.com/iterate-through-dictionary-python/#using-comprehensions)，一个[集合理解](https://realpython.com/list-comprehension-python/#using-set-and-dictionary-comprehensions)，或者一个[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)中重复一个操作，同样的原则也适用。

以下示例使用生成器表达式来计算长度超过`50`个字符的剧集标题的平均长度:

>>>

```py
>>> import statistics

>>> statistics.mean(
...     title_length
...     for title in podcasts
...     if (title_length := len(title)) > 50
... )
65.425
```

生成器表达式使用赋值表达式来避免两次计算每个剧集标题的长度。

[*Remove ads*](/account/join/)

### While 循环

Python 有两种不同的循环构造: [`for`循环](https://realpython.com/python-for-loop/)和 [`while`循环](https://realpython.com/python-while-loop/)。当您需要迭代一个已知的元素序列时，通常会使用`for`循环。另一方面，当你事先不知道需要循环多少次时，就使用`while`循环。

在`while`循环中，你需要定义并检查循环顶部的结束条件。当您需要在执行检查之前做一些设置时，这有时会导致一些笨拙的代码。下面是一个选择题测验程序的片段，它要求用户从几个有效答案中选择一个来回答问题:

```py
question = "Will you use the walrus operator?"
valid_answers = {"yes", "Yes", "y", "Y", "no", "No", "n", "N"}

user_answer = input(f"\n{question} ") while user_answer not in valid_answers:
    print(f"Please answer one of {', '.join(valid_answers)}")
 user_answer = input(f"\n{question} ")
```

这是可行的，但不幸的是重复了相同的`input()`行。在检查它是否有效之前，需要从用户那里获得至少一个答案。然后在`while`循环中再次调用`input()`来请求第二个答案，以防最初的`user_answer`无效。

如果你想让你的代码更容易维护，用一个`while True`循环重写这种逻辑是很常见的。不是让检查成为主`while`语句的一部分，而是稍后在循环中与显式`break`一起执行检查:

```py
while True:
    user_answer = input(f"\n{question} ")
    if user_answer in valid_answers:
        break
    print(f"Please answer one of {', '.join(valid_answers)}")
```

这具有避免重复的优点。然而，实际的支票现在更难发现了。

赋值表达式通常可以用来简化这类循环。在本例中，您现在可以将支票与`while`放在一起，这样更有意义:

```py
while (user_answer := input(f"\n{question} ")) not in valid_answers:
    print(f"Please answer one of {', '.join(valid_answers)}")
```

`while`语句有点密集，但代码现在更清楚地传达了意图，没有重复的行或看似无限的循环。

您可以展开下面的框来查看多项选择测验程序的完整代码，并自己尝试几个关于 walrus 操作员的问题。



此脚本运行一个多项选择测验。您将按顺序回答每个问题，但每次回答的顺序都会改变:

```py
# walrus_quiz.py

import random
import string

QUESTIONS = {
    "What is the name of PEP 572?": [
        "Assignment Expressions",
        "Named Expressions",
        "The Walrus Operator",
        "The Colon Equals Operator",
    ],
    "Which one of these is an invalid use of the walrus operator?": [
        "[y**2 for x in range(10) if y := f(x) > 0]",
        "print(y := f(x))",
        "(y := f(x))",
        "any((y := f(x)) for x in range(10))",
    ],
}

num_correct = 0
for question, answers in QUESTIONS.items():
    correct = answers[0]
    random.shuffle(answers)

    coded_answers = dict(zip(string.ascii_lowercase, answers))
    valid_answers = sorted(coded_answers.keys())

    for code, answer in coded_answers.items():
        print(f" {code}) {answer}")

 while (user_answer := input(f"\n{question} ")) not in valid_answers: print(f"Please answer one of {', '.join(valid_answers)}") 
    if coded_answers[user_answer] == correct:
        print(f"Correct, the answer is {user_answer!r}\n")
        num_correct += 1
    else:
        print(f"No, the answer is {correct!r}\n")

print(f"You got {num_correct} correct out of {len(QUESTIONS)} questions")
```

请注意，第一个答案被认为是正确的。您可以自己在测验中添加更多问题。欢迎在教程下面的评论区与社区分享您的问题！

您通常可以通过使用赋值表达式来简化`while`循环。最初的 PEP 向[展示了来自标准库](https://www.python.org/dev/peps/pep-0572/#sysconfig-py)的一个例子，它表达了同样的观点。

### 证人和反例

在迄今为止看到的例子中，`:=`赋值表达式操作符与旧代码中的`=`赋值操作符做的工作基本相同。您已经看到了如何简化代码，现在您将了解一种不同类型的用例，这种新的操作符使之成为可能。

在本节中，您将学习如何在调用 [`any()`](https://realpython.com/any-python/) 时找到**见证人**，使用一个巧妙的技巧，不使用 walrus 操作符是不可能的。在这个上下文中，见证是满足检查并导致`any()`返回`True`的元素。

通过应用类似的逻辑，你还将学习如何在使用 [`all()`](https://realpython.com/python-all/) 时找到**反例**。在这个上下文中，反例是不满足检查并导致`all()`返回`False`的元素。

为了处理一些数据，请定义以下城市名称列表:

>>>

```py
>>> cities = ["Vancouver", "Oslo", "Houston", "Warsaw", "Graz", "Holguín"]
```

您可以使用`any()`和`all()`来回答关于您的数据的问题:

>>>

```py
>>> # Does ANY city name start with "H"?
>>> any(city.startswith("H") for city in cities)
True

>>> # Does ANY city name have at least 10 characters?
>>> any(len(city) >= 10 for city in cities)
False

>>> # Do ALL city names contain "a" or "o"?
>>> all(set(city) & set("ao") for city in cities)
True

>>> # Do ALL city names start with "H"?
>>> all(city.startswith("H") for city in cities)
False
```

在每一种情况下，`any()`和`all()`给你简单的`True`或`False`答案。如果你也有兴趣看一个城市名称的例子或反例呢？看看是什么导致了你的`True`或`False`结果会很好:

*   有没有以`"H"`开头的**城市名**？

    是的，因为`"Houston"`是从`"H"`开始的。

*   所有的城市名称都是以`"H"`开头吗？

    不会，因为`"Oslo"`不是以`"H"`开头的。

换句话说，你想要一个证人或反例来证明答案。

在早期版本的 Python 中，捕捉一个`any()`表达式的见证并不直观。如果你在一个列表上调用`any()`，然后意识到你还需要一个见证，你通常需要重写你的代码:

>>>

```py
>>> witnesses = [city for city in cities if city.startswith("H")]

>>> if witnesses:
...     print(f"{witnesses[0]} starts with H")
... else:
...     print("No city name starts with H")
...
Houston starts with H
```

在这里，首先捕获所有以`"H"`开头的城市名称。然后，如果至少有一个这样的城市名，就打印出以`"H"`开头的第一个城市名。注意，这里你实际上没有使用`any()`,即使你在列表理解中做了类似的操作。

通过使用`:=`运算符，您可以在`any()`表达式中直接找到见证:

>>>

```py
>>> if any((witness := city).startswith("H") for city in cities):
...     print(f"{witness} starts with H")
... else:
...     print("No city name starts with H")
...
Houston starts with H
```

您可以在`any()`表达式中捕获一个见证。这个工作原理有点微妙，依赖于`any()`和`all()`使用[短路评估](https://realpython.com/python-operators-expressions/#compound-logical-expressions-and-short-circuit-evaluation):他们只检查必要的项目来确定结果。

**注意:**如果你想检查*是否所有的*城市名称都以字母`"H"`开头，那么你可以通过用`all()`替换`any()`并更新`print()`函数来报告第一个没有通过检查的项目来寻找反例。

通过将`.startswith("H")`封装在一个函数中，您可以更清楚地看到发生了什么，该函数还打印出正在检查的项目:

>>>

```py
>>> def starts_with_h(name):
...     print(f"Checking {name}: {name.startswith('H')}")
...     return name.startswith("H")
...

>>> any(starts_with_h(city) for city in cities)
Checking Vancouver: False
Checking Oslo: False
Checking Houston: True
True
```

注意`any()`实际上并没有检查`cities`中的所有项目。它只检查项目，直到找到满足条件的项目。组合`:=`操作符和`any()`通过迭代地将每个被检查的条目分配给`witness`来工作。然而，只有最后一个这样的项目存在，并显示哪个项目是最后由`any()`检查的。

即使当`any()`返回`False`时，也会发现一个见证:

>>>

```py
>>> any(len(witness := city) >= 10 for city in cities)
False

>>> witness
'Holguín'
```

然而，在这种情况下，`witness`没有给出任何见解。`'Holguín'`不包含十个或更多字符。见证只显示最后评估的项目。

[*Remove ads*](/account/join/)

## Walrus 运算符语法

在 Python 中赋值不是表达式的一个主要原因是赋值操作符(`=`)和相等比较操作符(`==`)的视觉相似性可能会导致错误。在引入赋值表达式时，我们花了很多心思来避免 walrus 操作符的类似错误。正如前面提到的，一个重要的特点是`:=`操作符永远不允许直接替代`=`操作符，反之亦然。

正如您在本教程开始时看到的，您不能使用普通的赋值表达式来赋值:

>>>

```py
>>> walrus := True
  File "<stdin>", line 1
    walrus := True
           ^
SyntaxError: invalid syntax
```

使用赋值表达式只赋值在语法上是合法的，但前提是要添加括号:

>>>

```py
>>> (walrus := True)
True
```

尽管这是可能的，但是，这确实是一个最好的例子，说明您应该远离 walrus 操作符，而使用传统的赋值语句。

PEP 572 显示了其他几个例子，其中`:=`操作符要么是非法的，要么是不被鼓励的。下面的例子都举了一个`SyntaxError`:

>>>

```py
>>> lat = lon := 0
SyntaxError: invalid syntax

>>> angle(phi = lat := 59.9)
SyntaxError: invalid syntax

>>> def distance(phi = lat := 0, lam = lon := 0):
SyntaxError: invalid syntax
```

在所有这些情况下，使用`=`会更好。接下来的例子类似，都是法律代码。然而，在以下任何情况下，walrus 操作符都不会改进您的代码:

>>>

```py
>>> lat = (lon := 0)  # Discouraged

>>> angle(phi = (lat := 59.9))  # Discouraged

>>> def distance(phi = (lat := 0), lam = (lon := 0)):  # Discouraged
...     pass
...
```

这些例子都没有让你的代码更易读。相反，您应该使用传统的赋值语句单独完成额外的赋值。有关推理的更多细节，请参见 [PEP 572](https://www.python.org/dev/peps/pep-0572/#exceptional-cases) 。

在一个用例中,`:=`字符序列已经是有效的 Python。在 [f 字符串](https://realpython.com/python-f-strings/)中，冒号(`:`)用于将值与它们的**格式规范**分开。例如:

>>>

```py
>>> x = 3
>>> f"{x:=8}"
'       3'
```

本例中的`:=`看起来确实像一个 walrus 操作符，但是效果完全不同。为了解释 f 弦内部的`x:=8`，表达式被分解为三个部分:`x`、`:`和`=8`。

这里，`x`是数值，`:`作为分隔符，`=8`是格式规范。根据 Python 的[格式规范迷你语言](https://docs.python.org/3/library/string.html#format-specification-mini-language)，在这个上下文中`=`指定了一个对齐选项。在这种情况下，该值在宽度为`8`的字段中用空格填充。

要在 f 字符串中使用赋值表达式，需要添加括号:

>>>

```py
>>> x = 3
>>> f"{(x := 8)}"
'8'

>>> x
8
```

这将按预期更新`x`的值。然而，你最好使用 f 弦之外的传统赋值。

让我们看看赋值表达式非法的其他一些情况:

*   **属性和项目分配:**您只能分配给简单的名称，不能分配给带点或索引的名称:

    >>>

    ```py
    >>> (mapping["hearts"] := "♥")
    SyntaxError: cannot use assignment expressions with subscript

    >>> (number.answer := 42)
    SyntaxError: cannot use assignment expressions with attribute` 
    ```

    这将失败，并显示一条描述性错误消息。没有简单的解决方法。

*   **Iterable 解包:**使用 walrus 运算符时无法解包:

    >>>

    ```py
    >>> lat, lon := 59.9, 10.8
    SyntaxError: invalid syntax` 
    ```

    如果您在整个表达式周围添加括号，它将被解释为一个包含三个元素`lat`、`59.9`和`10.8`的三元组。

*   **增广赋值:**你不能像`+=`一样使用 walrus 操作符结合增广赋值操作符。这就引出了一个`SyntaxError`:

    >>>

    ```py
    >>> count +:= 1
    SyntaxError: invalid syntax` 
    ```

    最简单的解决方法是显式地进行增强。例如，你可以做`(count := count + 1)`。 [PEP 577](https://www.python.org/dev/peps/pep-0577/) 最初描述了如何给 Python 添加增强赋值表达式，但是这个提议被撤回了。

当您使用 walrus 操作符时，它的行为在许多方面与传统的赋值语句相似:

*   任务目标的**范围**与任务相同。它将遵循 [LEGB 规则](https://realpython.com/python-scope-legb-rule/)。通常情况下，赋值将发生在局部范围内，但是如果目标名称已经声明为 [`global`](https://realpython.com/python-scope-legb-rule/#the-global-statement) 或 [`nonlocal`](https://realpython.com/python-scope-legb-rule/#the-nonlocal-statement) ，那么这将被接受。

*   walrus 操作符的**优先级**可能会造成一些混乱。除了逗号之外，它没有其他所有运算符绑定得紧密，所以您可能需要括号来分隔所分配的表达式。例如，请注意不使用括号时会发生什么:

    >>>

    ```py
    >>> number = 3
    >>> if square := number ** 2 > 5:
    ...     print(square)
    ...
    True` 
    ```

    `square`绑定到整个表达式`number ** 2 > 5`。换句话说，`square`得到的是值`True`，而不是`number ** 2`的值，这正是我们的意图。在这种情况下，可以用括号分隔表达式:

    >>>

    ```py
    >>> number = 3
    >>> if (square := number ** 2) > 5:
    ...     print(square)
    ...
    9` 
    ```

    括号使得`if`语句更加清晰，并且实际上是正确的。

    还有最后一个问题。当使用 walrus 操作符分配元组时，您总是需要在元组周围使用括号。比较以下分配:

    >>>

    ```py
    >>> walrus = 3.7, False
    >>> walrus
    (3.7, False)

    >>> (walrus := 3.8, True)
    (3.8, True)
    >>> walrus
    3.8

    >>> (walrus := (3.8, True))
    (3.8, True)
    >>> walrus
    (3.8, True)` 
    ```

    注意，在第二个例子中，`walrus`采用值`3.8`，而不是整个元组`3.8, True`。这是因为`:=`操作符比逗号绑定得更紧密。这可能看起来有点烦人。然而，如果`:=`操作符的约束没有逗号紧，那么在带有多个参数的函数调用中就不可能使用 walrus 操作符。

*   针对海象操作符的**风格建议**与用于赋值的`=`操作符基本相同。首先，在代码中总是在`:=`操作符周围添加空格。第二，必要时在表达式两边使用括号，但避免添加不需要的额外括号。

赋值表达式的一般设计是在它们有用的时候使它们易于使用，但是在它们可能使你的代码混乱的时候避免过度使用它们。

[*Remove ads*](/account/join/)

## 海象运营商的陷阱

walrus 运算符是一种新语法，仅在 Python 3.8 及更高版本中可用。这意味着您编写的任何使用`:=`语法的代码都只能在最新版本的 Python 上运行。

如果需要支持旧版本的 Python，就不能发布使用赋值表达式的代码。有一些项目，像 [`walrus`](https://github.com/pybpc/walrus) ，可以自动将 walrus 操作符翻译成与旧版本 Python 兼容的代码。这允许您在编写代码时利用赋值表达式，并且仍然分发与更多 Python 版本兼容的代码。

海象运营商的经验表明`:=`不会彻底改变 Python。相反，在有用的地方使用赋值表达式可以帮助您对代码进行一些小的改进，这对您的整体工作有好处。

很多时候你可以使用 walrus 操作符，但是这并不一定能提高代码的可读性和效率。在这种情况下，您最好以更传统的方式编写代码。

## 结论

现在您已经知道了新的 walrus 操作符是如何工作的，以及如何在自己的代码中使用它。通过使用`:=`语法，您可以避免代码中不同类型的重复，并使您的代码更有效、更易于阅读和维护。同时，你不应该到处使用赋值表达式。它们只会在某些用例中帮助你。

**在本教程中，您学习了如何:**

*   识别**海象运算符**并理解其含义
*   了解海象运营商的**用例**
*   **使用 walrus 运算符避免重复代码**
*   在使用 walrus 运算符的代码和使用**其他赋值方法**的代码之间转换
*   理解使用 walrus 操作符时对**向后兼容性**的影响
*   在赋值表达式中使用合适的**样式**

要了解更多关于赋值表达式的细节，请参见 [PEP 572](https://www.python.org/dev/peps/pep-0572/) 。您还可以查看 PyCon 2019 演讲 [PEP 572:海象运营商](https://pyvideo.org/pycon-us-2019/pep-572-the-walrus-operator.html)，其中[达斯汀·英格拉姆](https://twitter.com/di_codes)概述了海象运营商以及围绕新 PEP 的讨论。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python 赋值表达式和使用海象运算符**](/courses/python-assignment-expressions-walrus-operator/)********