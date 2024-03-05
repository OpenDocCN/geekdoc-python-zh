# 如何检查 Python 字符串是否包含子串

> 原文：<https://realpython.com/python-string-contains-substring/>

如果您是编程新手，或者来自 Python 之外的编程语言，您可能正在寻找在 Python 中检查一个字符串是否包含另一个字符串的最佳方法。

当您处理来自文件的[文本内容时，或者在](https://realpython.com/read-write-files-python/)[收到用户输入](https://realpython.com/python-input-output/)后，识别这样的[子字符串](https://en.wikipedia.org/wiki/Substring)会很方便。根据子字符串是否存在，您可能希望在程序中执行不同的操作。

在本教程中，您将关注处理这项任务的最 Pythonic 化的方法，使用**成员操作符`in`** 。此外，您将学习如何**为相关但不同的用例识别正确的字符串方法**。

最后，您还将学习如何**在 pandas 列**中查找子字符串。如果您需要搜索 CSV 文件中的数据，这很有帮助。您*可以*使用您将在下一节学习的方法，但是如果您正在使用**表格数据**，最好将数据加载到 pandas 数据框架中，然后[在 pandas](#find-a-substring-in-a-pandas-dataframe-column) 中搜索子字符串。

**免费下载:** [单击此处下载示例代码](https://realpython.com/bonus/python-string-contains-substring-code/)，您将使用它来检查字符串是否包含子字符串。

## 如何确认一个 Python 字符串包含另一个字符串

如果需要检查一个字符串是否包含子串，使用 Python 的成员操作符`in`。在 Python 中，这是确认字符串中子串存在的推荐方法:

>>>

```py
>>> raw_file_content = """Hi there and welcome.
... This is a special hidden file with a SECRET secret.
... I don't want to tell you The Secret,
... but I do want to secretly tell you that I have one."""

>>> "secret" in raw_file_content True
```

`in`成员操作符为您提供了一种快速、易读的方法来检查一个子串是否存在于一个字符串中。您可能会注意到，这一行代码读起来几乎像英语。

**注意:**如果要检查子串是否是字符串中的*而不是*，那么可以使用`not in`:

>>>

```py
>>> "secret" not in raw_file_content
False
```

因为子串`"secret"`出现在`raw_file_content`中，所以`not in`操作符返回`False`。

当使用`in`时，表达式返回一个[布尔值](https://realpython.com/python-boolean/):

*   `True`如果 Python 找到了子串
*   `False`如果 Python 没有找到子串

您可以在[条件语句](https://realpython.com/python-conditional-statements/)中使用这种直观的语法在您的代码中做出决定:

>>>

```py
>>> if "secret" in raw_file_content:
...    print("Found!")
...
Found!
```

在这个代码片段中，您使用成员操作符来检查`"secret"`是否是`raw_file_content`的子串。如果是，那么您将向终端打印一条消息。任何缩进的代码只有在您检查的 Python 字符串包含您提供的子字符串时才会执行。

**注意:** Python 认为[空字符串总是作为任何其他字符串的子串](https://docs.python.org/3.12/reference/expressions.html#membership-test-operations)，所以检查字符串中的空字符串会返回`True`:

>>>

```py
>>> "" in "secret"
True
```

这可能令人惊讶，因为 Python 认为[em tty 字符串为 false](https://docs.python.org/3/library/stdtypes.html#truth-value-testing) ，但是记住这种极端情况是有帮助的。

如果您只需要检查一个 Python 字符串是否包含子串，那么成员操作符`in`是您最好的朋友。

但是，如果你想知道更多关于子串的信息呢？如果您通读存储在`raw_file_content`中的文本，那么您会注意到子字符串不止一次出现，甚至出现在不同的变体中！

Python 找到了这些事件中的哪一个？大写有区别吗？子字符串在文本中出现的频率是多少？这些子串的位置在哪里？如果你需要这些问题的答案，请继续阅读。

[*Remove ads*](/account/join/)

## 通过消除大小写敏感性来概括您的检查

Python 字符串区分大小写。如果您提供的子字符串与文本中的同一个单词使用不同的大写字母，Python 将找不到它。例如，如果您在原文的[标题大小写版本](https://realpython.com/python-strings/#case-conversion)上检查小写单词`"secret"`，成员操作符 check 返回`False`:

>>>

```py
>>> title_cased_file_content = """Hi There And Welcome.
... This Is A Special Hidden File With A Secret Secret.
... I Don't Want To Tell You The Secret,
... But I Do Want To Secretly Tell You That I Have One."""

>>> "secret" in title_cased_file_content
False
```

尽管单词 *secret* 在标题文本`title_cased_file_content`中多次出现，但它*从未以小写形式出现。这就是为什么用成员资格操作符执行的检查会返回`False`。Python 在提供的文本中找不到全小写的字符串`"secret"`。*

人类对待语言的方式与计算机不同。这就是为什么在 Python 中检查一个字符串是否包含子串时，通常会忽略大小写。

您可以通过将整个输入文本转换为小写来概括您的子字符串检查:

>>>

```py
>>> file_content = title_cased_file_content.lower()

>>> print(file_content)
hi there and welcome.
this is a special hidden file with a secret secret.
i don't want to tell you the secret,
but i do want to secretly tell you that i have one.

>>> "secret" in file_content
True
```

将输入文本转换为小写是一种常见的方式，因为人类认为只有大小写不同的单词才是同一个单词，而计算机不会。

**注意:**在下面的例子中，您将继续使用小写版本的文本`file_content`。

如果您使用原始字符串(`raw_file_content`)或大写字符串(`title_cased_file_content`)，那么您会得到不同的结果，因为它们不是小写的。在您完成示例时，请随意尝试一下！

既然您已经将字符串转换为小写，以避免由区分大小写引起的意外问题，那么是时候进一步挖掘和了解更多关于子字符串的内容了。

## 了解有关子字符串的更多信息

成员操作符`in`是描述性检查字符串中是否有子串的好方法，但是它并没有提供更多的信息。它非常适合条件检查——但是如果您需要了解更多关于子字符串的信息，该怎么办呢？

Python 提供了许多额外的字符串方法，允许您检查字符串包含多少个目标子字符串，根据复杂的条件搜索子字符串，或者在文本中定位子字符串的索引。

在这一节中，您将讨论一些额外的字符串方法，这些方法可以帮助您了解更多关于子字符串的信息。

**注意:**你可能见过下面这些用来检查一个字符串是否包含子串的方法。这是可能的——但是它们不应该被用于这个目的！

编程是一项创造性的活动，你总能找到不同的方法来完成同一项任务。但是，为了提高代码的可读性，最好按照您正在使用的语言的意图来使用方法。

通过使用`in`，您确认了字符串包含子字符串。但是你没有得到子串所在的的*的任何信息。*

如果您需要知道子串出现在字符串中的什么位置，那么您可以在 string 对象上使用`.index()`:

>>>

```py
>>> file_content = """hi there and welcome.
... this is a special hidden file with a secret secret.
... i don't want to tell you the secret,
... but i do want to secretly tell you that i have one."""

>>> file_content.index("secret")
59
```

当您在字符串上调用`.index()`并将子字符串作为参数传递给它时，您将获得子字符串第一次出现的第一个字符的索引位置。

**注意:**如果 Python 找不到子串，那么`.index()`会引发一个`ValueError` [异常](https://realpython.com/python-exceptions/)。

但是，如果您想找到子字符串的其他出现，该怎么办呢？`.index()`方法还接受第二个参数，该参数可以定义从哪个索引位置开始查找。因此，通过传递特定的索引位置，可以跳过已经识别的子字符串:

>>>

```py
>>> file_content.index("secret", 60)
66
```

当您传递一个超过子串第一次出现的起始索引时，Python 会从那里开始搜索。在这种情况下，您会得到另一个匹配，而不是一个`ValueError`。

这意味着文本不止一次包含子字符串。但是多久一次呢？

您可以使用描述性和惯用的 Python 代码使用`.count()`快速得到您的答案:

>>>

```py
>>> file_content.count("secret")
4
```

您在小写字符串上使用了`.count()`，并将子字符串`"secret"`作为参数传递。Python 统计了子串在字符串中出现的频率，并返回答案。该文本包含子字符串四次。但是这些子串是什么样子的呢？

您可以检查所有的子字符串，方法是在默认的单词边界处拆分您的文本，并使用 [`for`循环](https://realpython.com/python-for-loop/)将单词打印到您的终端:

>>>

```py
>>> for word in file_content.split():
...    if "secret" in word:
...        print(word)
...
secret
secret.
secret,
secretly
```

在这个例子中，您使用 [`.split()`](https://realpython.com/python-string-split-concatenate-join/) 将空白处的文本分成字符串，Python 将这些字符串打包成一个列表。然后遍历这个列表，在每个字符串上使用`in`，看看它是否包含子串`"secret"`。

**注意:**除了打印子字符串之外，您还可以将它们保存在一个新的列表中，例如通过使用带有条件表达式的列表理解:

>>>

```py
>>> [word for word in file_content.split() if "secret" in word]
['secret', 'secret.', 'secret,', 'secretly']
```

在这种情况下，您只从包含子字符串的单词中构建一个列表，这实质上是过滤文本。

既然您可以检查 Python 识别的所有子字符串，您可能会注意到 Python 并不关心子字符串`"secret"`之后是否有任何字符。无论单词后面是空格还是标点符号，它都会找到该单词。它甚至可以找到像`"secretly"`这样的单词。

知道这一点很好，但是如果想要对子串检查施加更严格的条件，该怎么办呢？

[*Remove ads*](/account/join/)

## 使用正则表达式查找带有条件的子字符串

您可能只想匹配出现的子串和标点符号，或者识别包含子串和其他字母的单词，例如`"secretly"`。

对于这种需要更复杂的字符串匹配的情况，您可以将[正则表达式](https://realpython.com/regex-python/)或 regex 与 Python 的`re`模块一起使用。

例如，如果您想查找以`"secret"`开头但后面至少还有一个字母的所有单词，那么您可以使用 regex [单词字符](https://realpython.com/regex-python/#metacharacters-that-match-a-single-character) ( `\w`)，后面跟着[加量词](https://realpython.com/regex-python/#quantifiers) ( `+`):

>>>

```py
>>> import re

>>> file_content = """hi there and welcome.
... this is a special hidden file with a secret secret.
... i don't want to tell you the secret,
... but i do want to secretly tell you that i have one."""

>>> re.search(r"secret\w+", file_content)
<re.Match object; span=(128, 136), match='secretly'>
```

`re.search()`函数返回匹配条件的子串及其起始和结束索引位置——而不仅仅是`True`!

然后，您可以通过`Match`对象上的[方法来访问这些属性，用`m`表示:](https://realpython.com/regex-python-part-2/#match-object-methods-and-attributes)

>>>

```py
>>> m = re.search(r"secret\w+", file_content)

>>> m.group()
'secretly'

>>> m.span()
(128, 136)
```

这些结果为您继续处理匹配的子字符串提供了很大的灵活性。

例如，您可以只搜索后跟逗号(`,`)或句点(`.`)的子字符串:

>>>

```py
>>> re.search(r"secret[\.,]", file_content)
<re.Match object; span=(66, 73), match='secret.'>
```

您的文本中有两个潜在匹配项，但您只匹配了符合您的查询的第一个结果。当您使用`re.search()`时，Python 再次只找到*的第一个*匹配。如果您想让*符合某个条件的`"secret"`的所有*提及会怎样？

要使用`re`查找所有匹配，您可以使用`re.findall()`:

>>>

```py
>>> re.findall(r"secret[\.,]", file_content)
['secret.', 'secret,']
```

通过使用`re.findall()`，您可以在您的文本中找到该模式的所有匹配。Python 将所有匹配作为字符串保存在一个列表中。

当您使用[捕获组](https://docs.python.org/3/howto/regex.html#grouping)时，您可以通过将该部分括在括号中来指定您想要在列表中保留的匹配部分:

>>>

```py
>>> re.findall(r"(secret)[\.,]", file_content)
['secret', 'secret']
```

通过将*秘密*括在括号中，您定义了一个单独的捕获组。 [`findall()`函数](https://docs.python.org/3/library/re.html#re.findall)返回匹配捕获组的字符串列表，只要模式中正好有一个捕获组。通过在*秘密*周围加上括号，你成功的去掉了标点符号！

**注意:**记住，子串`"secret"`在您的文本中出现了四次，通过使用`re`，您过滤出了两个您根据特殊条件匹配的特定出现。

将`re.findall()`与匹配组一起使用是从文本中提取子字符串的有效方法。但是你只得到一个*字符串*的列表，这意味着你已经丢失了你在使用`re.search()`时可以访问的索引位置。

如果你想保留这些信息，那么`re`可以给你一个[迭代器](https://dbader.org/blog/python-iterators)中的所有匹配:

>>>

```py
>>> for match in re.finditer(r"(secret)[\.,]", file_content):
...    print(match)
...
<re.Match object; span=(66, 73), match='secret.'>
<re.Match object; span=(103, 110), match='secret,'>
```

当您使用`re.finditer()`并将搜索模式和文本内容作为参数传递给它时，您可以访问包含子字符串的每个`Match`对象，以及它的开始和结束索引位置。

您可能会注意到标点符号出现在这些结果中，即使您仍然使用捕获组。这是因为一个 [`Match`对象](https://docs.python.org/3/library/re.html#match-objects)的字符串表示显示了整个匹配，而不仅仅是第一个捕获组。

但是`Match`对象是一个强大的信息容器，就像您之前看到的那样，您可以挑选出您需要的信息:

>>>

```py
>>> for match in re.finditer(r"(secret)[\.,]", file_content):
...    print(match.group(1))
...
secret
secret
```

通过调用 [`.group()`](https://docs.python.org/3/library/re.html#re.Match.group) 并指定您想要的第一个捕获组，您从每个匹配的子串中选择了不带标点的单词 *secret* 。

当您使用正则表达式时，您可以使用子串匹配进行更详细的描述。除了检查一个字符串是否包含另一个字符串，您还可以根据复杂的条件搜索子字符串。

**注意:**如果你想学习更多关于使用捕获组和组成更复杂的正则表达式模式的知识，那么你可以深入研究 Python 中的[正则表达式。](https://realpython.com/regex-python/)

如果您需要有关子字符串的信息，或者如果您在文本中找到子字符串后需要继续使用它们，那么使用带有`re`的正则表达式是一个很好的方法。但是如果您正在处理表格数据呢？为此，你会求助于熊猫。

[*Remove ads*](/account/join/)

## 在熊猫数据帧列中查找子串

如果您处理的数据不是来自纯文本文件或用户输入，而是来自 [CSV 文件](https://realpython.com/python-csv/)或 [Excel 表格](https://realpython.com/openpyxl-excel-spreadsheets-python/)，那么您可以使用上面讨论的相同方法。

然而，有一种更好的方法来识别列中的哪些单元格包含子串:您将使用**熊猫**！在本例中，您将使用一个包含虚假公司名称和标语的 CSV 文件。如果您想继续工作，可以下载下面的文件:

**免费下载:** [单击此处下载示例代码](https://realpython.com/bonus/python-string-contains-substring-code/)，您将使用它来检查字符串是否包含子字符串。

当你在 Python 中处理表格数据时，通常最好先把它加载到一个[熊猫`DataFrame`](https://realpython.com/pandas-dataframe/) 中:

>>>

```py
>>> import pandas as pd

>>> companies = pd.read_csv("companies.csv")

>>> companies.shape
(1000, 2)

>>> companies.head()
 company                                     slogan
0      Kuvalis-Nolan      revolutionize next-generation metrics
1  Dietrich-Champlin  envisioneer bleeding-edge functionalities
2           West Inc            mesh user-centric infomediaries
3         Wehner LLC               utilize sticky infomediaries
4      Langworth Inc                 reinvent magnetic networks
```

在这个代码块中，您将一个包含一千行虚假公司数据的 CSV 文件加载到 pandas DataFrame 中，并使用`.head()`检查了前五行。

**注意:**你将需要[创建一个虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)和[安装熊猫](https://realpython.com/pandas-read-write-files/#installing-pandas)以便使用图书馆。

将数据加载到 DataFrame 中后，可以快速查询整个 pandas 列来筛选包含子字符串的条目:

>>>

```py
>>> companies[companies.slogan.str.contains("secret")]
 company                                  slogan
7          Maggio LLC                    target secret niches
117      Kub and Sons              brand secret methodologies
654       Koss-Zulauf              syndicate secret paradigms
656      Bernier-Kihn  secretly synthesize back-end bandwidth
921      Ward-Shields               embrace secret e-commerce
945  Williamson Group             unleash secret action-items
```

您可以在 pandas 列上使用 [`.str.contains()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html) ,并将子串作为参数传递给它，以过滤包含子串的行。

**注意:**索引操作符(`[]`)和属性操作符(`.`)提供了[获取](https://pandas.pydata.org/docs/user_guide/10min.html#getting)数据帧的单个列或片的直观方法。

然而，如果您正在处理与性能有关的生产代码，pandas 建议使用优化的数据访问方法来索引和选择数据。

当你使用`.str.contains()`并且需要更复杂的匹配场景时，你也可以使用正则表达式！您只需要传递一个符合 regex 的搜索模式作为 substring 参数:

>>>

```py
>>> companies[companies.slogan.str.contains(r"secret\w+")]
 company                                  slogan
656  Bernier-Kihn  secretly synthesize back-end bandwidth
```

在这个代码片段中，您使用了之前使用的相同模式，只匹配包含*秘密*的单词，但是继续使用一个或多个单词字符(`\w+`)。这个假数据集中只有一家公司似乎在秘密运营*！

您可以编写任何复杂的正则表达式模式，并将其传递给`.str.contains()`,以便从 pandas 列中切割出您分析所需的行。

## 结论

就像一个坚持不懈的寻宝者，你找到了每一个`"secret"`，不管它藏得多好！在这个过程中，您了解到在 Python 中检查字符串是否包含子串的最佳方式是使用 **`in`成员操作符**。

您还学习了如何描述性地使用另外两个**字符串方法**，它们经常被误用来检查子字符串:

*   `.count()`统计子串在字符串中出现的次数
*   `.index()`获取子串开头的索引位置

之后，您探索了如何使用**正则表达式**和 Python 的`re`模块中的一些函数根据更高级的条件查找子字符串。

最后，您还学习了如何使用 DataFrame 方法`.str.contains()`来检查 **pandas DataFrame** 中的哪些条目包含子串。

现在，您知道了在 Python 中处理子字符串时如何选择最惯用的方法。继续使用最具描述性的方法，你会写出令人愉悦的代码，让别人很快就能理解。

**免费下载:** [单击此处下载示例代码](https://realpython.com/bonus/python-string-contains-substring-code/)，您将使用它来检查字符串是否包含子字符串。****