# 正则表达式:Python 中的正则表达式(第 1 部分)

> 原文：<https://realpython.com/regex-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**正则表达式和用 Python 构建正则表达式**](/courses/building-regexes-python/)

在本教程中，您将探索 Python 中的**正则表达式**，也称为**正则表达式**。正则表达式是一种特殊的字符序列，它为复杂的字符串匹配功能定义了一种模式。

在本系列的前面，在教程[Python 中的字符串和字符数据](https://realpython.com/python-strings)中，您学习了如何定义和操作字符串对象。从那时起，您已经看到了一些确定两个字符串是否匹配的方法:

*   您可以使用[相等(`==` )](https://realpython.com/python-operators-expressions/#comparison-operators) 运算符来测试两个字符串是否相等。

*   你可以用 [`in`](https://realpython.com/python-strings/#string-operators) 操作符或者[内置的字符串方法](https://realpython.com/python-strings/#built-in-string-methods) `.find()`和`.index()`来测试[是否是另一个字符串的子串](https://realpython.com/python-string-contains-substring/)。

像这样的字符串匹配是编程中的常见任务，使用字符串操作符和内置方法可以完成很多工作。但是，有时您可能需要更复杂的模式匹配功能。

在本教程中，您将学习:

*   如何访问 Python 中实现正则表达式匹配的 **`re`模块**
*   如何使用 **`re.search()`** 来匹配字符串的模式
*   如何用正则表达式元字符创建复杂的匹配模式

系好安全带！正则表达式语法需要一点时间来适应。但是一旦你熟悉了它，你会发现正则表达式在你的 Python 编程中几乎是不可或缺的。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## Python 中的正则表达式及其用途

假设你有一个字符串对象`s`。现在假设您需要编写 Python 代码来找出`s`是否包含子串`'123'`。至少有几种方法可以做到这一点。您可以使用`in`操作符:

>>>

```py
>>> s = 'foo123bar'
>>> '123' in s
True
```

如果你不仅想知道*`'123'`是否存在于`s`中，还想知道*存在于哪里，那么你可以使用`.find()`或者`.index()`。每个函数都返回子字符串所在的字符在`s`中的位置:**

**>>>

```py
>>> s = 'foo123bar'
>>> s.find('123')
3
>>> s.index('123')
3
```

在这些例子中，匹配是通过简单的逐字符比较来完成的。在许多情况下，这将完成工作。但有时候，问题比这更复杂。

例如，与其搜索像`'123'`这样的固定子串，不如假设您想确定一个字符串是否包含*任何*三个连续的十进制数字字符，如字符串`'foo123bar'`、`'foo456bar'`、`'234baz'`和`'qux678'`。

严格的字符比较在这里是行不通的。这就是 Python 中正则表达式的用武之地。

[*Remove ads*](/account/join/)

### 正则表达式的(非常简短的)历史

1951 年，数学家斯蒂芬·科尔·克莱尼描述了[正则语言](https://en.wikipedia.org/wiki/Regular_language)的概念，这是一种可被有限自动机识别并可使用[正则表达式](https://en.wikipedia.org/wiki/Regular_expression)进行正式表达的语言。20 世纪 60 年代中期，计算机科学先驱[肯·汤普森](https://en.wikipedia.org/wiki/Ken_Thompson)，Unix 的最初设计者之一，使用克莱尼符号在 [QED 文本编辑器](https://en.wikipedia.org/wiki/QED_(text_editor))中实现了模式匹配。

从那以后，正则表达式出现在许多编程语言、编辑器和其他工具中，作为确定字符串是否匹配指定模式的一种手段。Python、 [Java](https://realpython.com/oop-in-python-vs-java/) 和 Perl 都支持 regex 功能，大多数 Unix 工具和许多文本编辑器也是如此。

### `re`模块

Python 中的正则表达式功能驻留在一个名为`re`的模块中。`re`模块包含许多有用的函数和方法，其中大部分将在本系列的下一篇教程中学习。

现在，您将主要关注一个功能，`re.search()`。

`re.search(<regex>, <string>)`

> 扫描字符串中的正则表达式匹配。

`re.search(<regex>, <string>)`扫描`<string>`寻找模式`<regex>`匹配的第一个位置。如果找到匹配，那么`re.search()`返回一个**匹配对象**。否则，返回 [`None`](https://realpython.com/null-in-python/) 。

`re.search()`接受可选的第三个`<flags>`参数，您将在本教程结束时了解到。

### 如何导入`re.search()`

因为`search()`驻留在`re`模块中，你需要[导入](https://realpython.com/lessons/import-statement/)才能使用它。一种方法是导入整个模块，然后在调用函数时使用模块名作为前缀:

```py
import re
re.search(...)
```

或者，您可以通过名称从模块中导入函数，然后在没有模块名称前缀的情况下引用它:

```py
from re import search
search(...)
```

在你能够使用它之前，你总是需要以某种方式导入`re.search()`。

本教程剩余部分中的例子将假设所示的第一种方法——导入`re`模块，然后引用带有模块名前缀的函数:`re.search()`。为了简洁起见，`import re`语句通常会被省略，但是记住它总是必要的。

有关从模块和包导入的更多信息，请查看 [Python 模块和包——简介](https://realpython.com/python-modules-packages/)。

### 第一个模式匹配示例

现在您已经知道如何访问`re.search()`，您可以试一试:

>>>

```py
 1>>> s = 'foo123bar'
 2
 3>>> # One last reminder to import!
 4>>> import re
 5
 6>>> re.search('123', s)
 7<_sre.SRE_Match object; span=(3, 6), match='123'>
```

这里，搜索模式`<regex>`是`123`，`<string>`是`s`。返回的匹配对象出现在**第 7 行**。Match 对象包含大量有用的信息，您将很快了解这些信息。

目前，重要的一点是`re.search()`实际上返回了一个匹配对象，而不是`None`。告诉你它找到了一个匹配。换句话说，指定的`<regex>`图案`123`存在于`s`中。

匹配对象是 [**真值**](https://realpython.com/python-data-types/#boolean-type-boolean-context-and-truthiness) ，所以你可以像条件语句一样在[布尔上下文](https://realpython.com/python-boolean/)中使用它:

>>>

```py
>>> if re.search('123', s):
...     print('Found a match.')
... else:
...     print('No match.')
...
Found a match.
```

解释器将匹配对象显示为`<_sre.SRE_Match object; span=(3, 6), match='123'>`。这包含了一些有用的信息。

`span=(3, 6)`表示在`<string>`中找到匹配的部分。这与在[切片符号](https://realpython.com/python-strings/#string-slicing)中的意思相同:

>>>

```py
>>> s[3:6]
'123'
```

在这个例子中，匹配从字符位置`3`开始，延伸到但不包括位置`6`。

`match='123'`表示来自`<string>`的哪些字符匹配。

这是一个好的开始。但是在这种情况下，`<regex>`模式只是普通的字符串`'123'`。这里的模式匹配仍然只是逐字符的比较，与前面显示的`in`操作符和`.find()`示例非常相似。match 对象告诉您匹配的字符是`'123'`，这很有帮助，但是这并不能说明什么，因为这些正是您要搜索的字符。

你才刚开始热身。

[*Remove ads*](/account/join/)

### Python 正则表达式元字符

当`<regex>`包含称为**元字符**的特殊字符时，Python 中正则表达式匹配的真正威力就显现出来了。这些对于正则表达式匹配引擎有着独特的意义，并极大地增强了搜索能力。

再次考虑如何确定一个字符串是否包含任何三个连续的十进制数字字符的问题。

在正则表达式中，方括号(`[]`)中指定的一组字符组成了一个**字符类**。此元字符序列匹配类中的任何单个字符，如以下示例所示:

>>>

```py
>>> s = 'foo123bar'
>>> re.search('[0-9][0-9][0-9]', s)
<_sre.SRE_Match object; span=(3, 6), match='123'>
```

`[0-9]`匹配任何单个十进制数字字符——包括`'0'`和`'9'`之间的任何字符。完整表达式`[0-9][0-9][0-9]`匹配三个十进制数字字符的任意序列。在这种情况下，`s`匹配，因为它包含三个连续的十进制数字字符，`'123'`。

这些字符串也匹配:

>>>

```py
>>> re.search('[0-9][0-9][0-9]', 'foo456bar')
<_sre.SRE_Match object; span=(3, 6), match='456'>

>>> re.search('[0-9][0-9][0-9]', '234baz')
<_sre.SRE_Match object; span=(0, 3), match='234'>

>>> re.search('[0-9][0-9][0-9]', 'qux678')
<_sre.SRE_Match object; span=(3, 6), match='678'>
```

另一方面，不包含三个连续数字的字符串不匹配:

>>>

```py
>>> print(re.search('[0-9][0-9][0-9]', '12foo34'))
None
```

使用 Python 中的正则表达式，您可以识别字符串中使用`in`操作符或字符串方法无法找到的模式。

看看另一个正则表达式元字符。点(`.`)元字符匹配除换行符之外的任何字符，因此它的功能类似于通配符:

>>>

```py
>>> s = 'foo123bar'
>>> re.search('1.3', s)
<_sre.SRE_Match object; span=(3, 6), match='123'>

>>> s = 'foo13bar'
>>> print(re.search('1.3', s))
None
```

在第一个例子中，正则表达式`1.3`匹配`'123'`，因为`'1'`和`'3'`完全匹配，而`.`匹配`'2'`。在这里，您实际上是在问，“`s`是否包含一个`'1'`，然后是任何字符(除了换行符)，然后是一个`'3'`？”对于`'foo123bar'`答案是肯定的，对于`'foo13bar'`答案是否定的。

这些例子快速展示了正则表达式元字符的强大功能。字符类和点只是`re`模块支持的两个元字符。还有很多。接下来，您将全面探索它们。

## `re`模块支持的元字符

下表简要总结了`re`模块支持的所有元字符。有些角色有多种用途:

| 字符 | 意义 |
| --- | --- |
| `.` | 匹配除换行符以外的任何单个字符 |
| `^` | 在一个字符串的开头锚定一个匹配
√补充一个字符类 |
| `$` | 在字符串末尾锚定一个匹配 |
| `*` | 匹配零次或多次重复 |
| `+` | 匹配一个或多个重复 |
| `?` | 匹配零个或一个重复
∏指定`*`、`+`和`?`、
的非贪婪版本∏引入前视或后视断言
∏创建命名组 |
| `{}` | 匹配明确指定的重复次数 |
| `\` | 转义具有特殊含义的元字符
√引入特殊字符类
√引入分组反向引用 |
| `[]` | 指定了字符类 |
| `&#124;` | 指定变更 |
| `()` | 创建一个组 |
| `:`
`#`
`=`
 | 指定一个专门小组 |
| `<>` | 创建命名组 |

这似乎是一个巨大的信息量，但是不要惊慌！接下来的章节将详细介绍其中的每一项。

regex 解析器将上面没有列出的任何字符视为只匹配自身的普通字符。例如，在上面显示的[第一个模式匹配示例](#first-pattern-matching-example)中，您会看到:

>>>

```py
>>> s = 'foo123bar'
>>> re.search('123', s)
<_sre.SRE_Match object; span=(3, 6), match='123'>
```

在这种情况下，`123`从技术上来说是一个正则表达式，但它并不是一个非常有趣的正则表达式，因为它不包含任何元字符。它只是匹配字符串`'123'`。

当您将元字符融入其中时，事情会变得更加令人兴奋。以下部分详细解释了如何使用每个元字符或元字符序列来增强模式匹配功能。

[*Remove ads*](/account/join/)

### 匹配单个字符的元字符

本节中的元字符序列尝试匹配搜索字符串中的单个字符。当正则表达式解析器遇到这些元字符序列中的一个时，如果当前解析位置的字符符合该序列描述的描述，就会发生匹配。

`[]`

> 指定要匹配的特定字符集。

方括号(`[]`)中包含的字符代表一个**字符类**——一组要匹配的枚举字符。字符类元字符序列将匹配该类中包含的任何单个字符。

您可以像这样逐个列举字符:

>>>

```py
>>> re.search('ba[artz]', 'foobarqux')
<_sre.SRE_Match object; span=(3, 6), match='bar'>
>>> re.search('ba[artz]', 'foobazqux')
<_sre.SRE_Match object; span=(3, 6), match='baz'>
```

元字符序列`[artz]`匹配任何单个`'a'`、`'r'`、`'t'`或`'z'`字符。在这个例子中，正则表达式`ba[artz]`匹配`'bar'`和`'baz'`(也将匹配`'baa'`和`'bat'`)。

字符类还可以包含由连字符(`-`)分隔的字符范围，在这种情况下，它匹配范围内的任何单个字符。例如，`[a-z]`匹配`'a'`和`'z'`之间的任何小写字母字符，包括:

>>>

```py
>>> re.search('[a-z]', 'FOObar')
<_sre.SRE_Match object; span=(3, 4), match='b'>
```

`[0-9]`匹配任何数字字符:

>>>

```py
>>> re.search('[0-9][0-9]', 'foo123bar')
<_sre.SRE_Match object; span=(3, 5), match='12'>
```

在这种情况下，`[0-9][0-9]`匹配两位数的序列。匹配的字符串`'foo123bar'`的第一部分是`'12'`。

`[0-9a-fA-F]`匹配任意[十六进制](https://en.wikipedia.org/wiki/Hexadecimal)数字字符:

>>>

```py
>>> re.search('[0-9a-fA-f]', '--- a0 ---')
<_sre.SRE_Match object; span=(4, 5), match='a'>
```

这里，`[0-9a-fA-F]`匹配搜索字符串中的第一个十六进制数字字符，`'a'`。

**注意:**在上面的例子中，返回值总是最左边的可能匹配。`re.search()`从左到右扫描搜索字符串，一旦找到与`<regex>`匹配的内容，它就停止扫描并返回匹配内容。

您可以通过将`^`指定为第一个字符来补充字符类，在这种情况下，它匹配集合中*不属于*的任何字符。在以下示例中，`[^0-9]`匹配任何不是数字的字符:

>>>

```py
>>> re.search('[^0-9]', '12345foo')
<_sre.SRE_Match object; span=(5, 6), match='f'>
```

这里，match 对象表示字符串中第一个不是数字的字符是`'f'`。

如果一个`^`字符出现在一个字符类中，但不是第一个字符，那么它没有特殊的含义，并且匹配一个文字`'^'`字符:

>>>

```py
>>> re.search('[#:^]', 'foo^bar:baz#qux')
<_sre.SRE_Match object; span=(3, 4), match='^'>
```

如您所见，您可以通过用连字符分隔字符来指定字符类中的字符范围。如果您希望字符类包含文字连字符，该怎么办？您可以将其作为第一个或最后一个字符，或者用反斜杠(`\`)将其转义:

>>>

```py
>>> re.search('[-abc]', '123-456')
<_sre.SRE_Match object; span=(3, 4), match='-'>
>>> re.search('[abc-]', '123-456')
<_sre.SRE_Match object; span=(3, 4), match='-'>
>>> re.search('[ab\-c]', '123-456')
<_sre.SRE_Match object; span=(3, 4), match='-'>
```

如果你想在一个字符类中包含一个文字`']'`，那么你可以把它作为第一个字符或者用反斜杠对它进行转义:

>>>

```py
>>> re.search('[]]', 'foo[1]')
<_sre.SRE_Match object; span=(5, 6), match=']'>
>>> re.search('[ab\]cd]', 'foo[1]')
<_sre.SRE_Match object; span=(5, 6), match=']'>
```

其他正则表达式元字符在字符类中失去了它们的特殊意义:

>>>

```py
>>> re.search('[)*+|]', '123*456')
<_sre.SRE_Match object; span=(3, 4), match='*'>
>>> re.search('[)*+|]', '123+456')
<_sre.SRE_Match object; span=(3, 4), match='+'>
```

如上表所示，`*`和`+`在 Python 的正则表达式中有特殊的含义。它们表示重复，稍后你会学到更多。但是在这个例子中，它们在一个字符类中，所以它们在字面上匹配它们自己。

点(`.`)

> 指定通配符。

`.`元字符匹配除换行符之外的任何单个字符:

>>>

```py
>>> re.search('foo.bar', 'fooxbar')
<_sre.SRE_Match object; span=(0, 7), match='fooxbar'>

>>> print(re.search('foo.bar', 'foobar'))
None
>>> print(re.search('foo.bar', 'foo\nbar'))
None
```

作为一个正则表达式，`foo.bar`本质上意味着字符`'foo'`，然后是除换行符之外的任何字符，然后是字符`'bar'`。上面显示的第一个字符串`'fooxbar'`符合要求，因为`.`元字符与`'x'`匹配。

第二个和第三个字符串不匹配。在最后一种情况下，尽管在`'foo'`和`'bar'`之间有一个字符，但它是一个换行符，默认情况下，`.`元字符不匹配换行符。然而，有一种方法可以强制`.`匹配一个换行符，您将在本教程的末尾了解到这一点。

`\w`
T1】

> 基于字符是否是单词字符进行匹配。

`\w`匹配任何字母数字单词字符。单词字符是大小写字母、数字和下划线(`_`)字符，所以`\w`实际上是`[a-zA-Z0-9_]`的简写:

>>>

```py
>>> re.search('\w', '#(.a$@&')
<_sre.SRE_Match object; span=(3, 4), match='a'>
>>> re.search('[a-zA-Z0-9_]', '#(.a$@&')
<_sre.SRE_Match object; span=(3, 4), match='a'>
```

在这种情况下，字符串`'#(.a$@&'`中的第一个单词字符是`'a'`。

`\W`正相反。它匹配任何非单词字符，相当于`[^a-zA-Z0-9_]`:

>>>

```py
>>> re.search('\W', 'a_1*3Qb')
<_sre.SRE_Match object; span=(3, 4), match='*'>
>>> re.search('[^a-zA-Z0-9_]', 'a_1*3Qb')
<_sre.SRE_Match object; span=(3, 4), match='*'>
```

这里，`'a_1*3!b'`中的第一个非文字字符是`'*'`。

`\d`
T1】

> 基于字符是否为十进制数字进行匹配。

`\d`匹配任何十进制数字字符。`\D`则相反。它匹配*不是*十进制数字的任何字符:

>>>

```py
>>> re.search('\d', 'abc4def')
<_sre.SRE_Match object; span=(3, 4), match='4'>

>>> re.search('\D', '234Q678')
<_sre.SRE_Match object; span=(3, 4), match='Q'>
```

`\d`本质上相当于`[0-9]`，`\D`相当于`[^0-9]`。

`\s`
T1】

> 基于字符是否代表空白进行匹配。

`\s`匹配任何空白字符:

>>>

```py
>>> re.search('\s', 'foo\nbar baz')
<_sre.SRE_Match object; span=(3, 4), match='\n'>
```

注意，与点通配符元字符不同，`\s`匹配换行符。

`\S`是`\s`的反义词。它匹配*不是*空格的任何字符:

>>>

```py
>>> re.search('\S', ' \n foo \n ')
<_sre.SRE_Match object; span=(4, 5), match='f'>
```

同样，`\s`和`\S`认为换行符是空白。在上面的例子中，第一个非空白字符是`'f'`。

字符类序列`\w`、`\W`、`\d`、`\D`、`\s`和`\S`也可以出现在方括号字符类中:

>>>

```py
>>> re.search('[\d\w\s]', '---3---')
<_sre.SRE_Match object; span=(3, 4), match='3'>
>>> re.search('[\d\w\s]', '---a---')
<_sre.SRE_Match object; span=(3, 4), match='a'>
>>> re.search('[\d\w\s]', '--- ---')
<_sre.SRE_Match object; span=(3, 4), match=' '>
```

在这种情况下，`[\d\w\s]`匹配任何数字、单词或空白字符。并且由于`\w`包括`\d`，相同的字符类也可以表示为略短的`[\w\s]`。

[*Remove ads*](/account/join/)

### 转义元字符

偶尔，你会想在你的正则表达式中包含一个元字符，除非你不想让它有特殊的含义。相反，您会希望它将自己表示为一个文字字符。

反斜杠(`\`)

> 删除元字符的特殊含义。

正如您刚才看到的，反斜杠字符可以引入特殊的字符类，如单词、数字和空格。还有一些特殊的元字符序列，称为以反斜杠开头的**锚**，您将在下面了解到。

当反斜杠**不满足这两个目的时，它会对**元字符进行转义。以反斜杠开头的元字符失去了它的特殊含义，而是匹配原义字符。考虑下面的例子:

>>>

```py
 1>>> re.search('.', 'foo.bar') 2<_sre.SRE_Match object; span=(0, 1), match='f'>
 3
 4>>> re.search('\.', 'foo.bar') 5<_sre.SRE_Match object; span=(3, 4), match='.'>
```

在第 1 行**的`<regex>`中，点(`.`)作为通配符元字符，匹配字符串中的第一个字符(`'f'`)。第 4 行**的`<regex>`中的`.`字符由反斜杠转义，所以它不是通配符。它被逐字解释并匹配搜索字符串索引`3`处的`'.'`。

使用反斜杠进行转义会变得混乱。假设您有一个包含单个反斜杠的字符串:

>>>

```py
>>> s = r'foo\bar'
>>> print(s)
foo\bar
```

现在假设您想要创建一个`<regex>`，它将匹配`'foo'`和`'bar'`之间的反斜杠。反斜杠本身是正则表达式中的特殊字符，所以要指定字面反斜杠，需要用另一个反斜杠对其进行转义。如果是这种情况，那么应该执行以下操作:

>>>

```py
>>> re.search('\\', s)
```

不完全是。这是你尝试的结果:

>>>

```py
>>> re.search('\\', s)
Traceback (most recent call last):
  File "<pyshell#3>", line 1, in <module>
    re.search('\\', s)
  File "C:\Python36\lib\re.py", line 182, in search
    return _compile(pattern, flags).search(string)
  File "C:\Python36\lib\re.py", line 301, in _compile
    p = sre_compile.compile(pattern, flags)
  File "C:\Python36\lib\sre_compile.py", line 562, in compile
    p = sre_parse.parse(p, flags)
  File "C:\Python36\lib\sre_parse.py", line 848, in parse
    source = Tokenizer(str)
  File "C:\Python36\lib\sre_parse.py", line 231, in __init__
    self.__next()
  File "C:\Python36\lib\sre_parse.py", line 245, in __next
    self.string, len(self.string) - 1) from None
sre_constants.error: bad escape (end of pattern) at position 0
```

哎呀。发生了什么事？

这里的问题是反斜杠转义发生了两次，第一次是由 Python 解释器对字符串进行转义，第二次是由 regex 解析器对收到的 regex 进行转义。

事情的顺序是这样的:

1.  Python 解释器首先处理字符串文字`'\\'`。它将其解释为转义反斜杠，并且只将一个反斜杠传递给`re.search()`。
2.  正则表达式解析器只接收一个反斜杠，这不是一个有意义的正则表达式，所以混乱的错误随之而来。

有两种方法可以解决这个问题。首先，您可以对原始字符串中的两个反斜杠进行转义:

>>>

```py
>>> re.search('\\\\', s)
<_sre.SRE_Match object; span=(3, 4), match='\\'>
```

这样做会导致以下情况发生:

1.  解释器将`'\\\\'`视为一对转义反斜杠。它将每一对简化为一个反斜杠，并将`'\\'`传递给 regex 解析器。
2.  然后，正则表达式解析器将`\\`视为一个转义反斜杠。作为一个`<regex>`，它匹配一个反斜杠字符。您可以从 match 对象中看到，它按照预期匹配了`s`中索引`3`处的反斜杠。很繁琐，但是很管用。

第二种，也可能是更干净的处理方式是使用一个[原始字符串](https://realpython.com/lessons/raw-strings/)来指定`<regex>`:

>>>

```py
>>> re.search(r'\\', s)
<_sre.SRE_Match object; span=(3, 4), match='\\'>
```

这抑制了解释器级别的转义。字符串`'\\'`被原封不动地传递给 regex 解析器，解析器再次根据需要看到一个转义反斜杠。

在 Python 中，每当包含反斜杠时，使用原始字符串来指定正则表达式是一个好习惯。

[*Remove ads*](/account/join/)

### 锚

锚点是零宽度匹配。它们不匹配搜索字符串中的任何实际字符，并且在解析过程中不消耗任何搜索字符串。相反，锚点指示搜索字符串中必须出现匹配的特定位置。

`^`
T1】

> 将一场比赛锚定到`<string>`的开始。

当正则表达式解析器遇到`^`或`\A`时，解析器的当前位置必须在搜索字符串的开头，以便找到匹配。

换句话说，regex `^foo`规定`'foo'`不仅要出现在搜索字符串中的任何旧位置，还要出现在开头:

>>>

```py
>>> re.search('^foo', 'foobar')
<_sre.SRE_Match object; span=(0, 3), match='foo'>
>>> print(re.search('^foo', 'barfoo'))
None
```

`\A`功能相似:

>>>

```py
>>> re.search('\Afoo', 'foobar')
<_sre.SRE_Match object; span=(0, 3), match='foo'>
>>> print(re.search('\Afoo', 'barfoo'))
None
```

`^`和`\A`在`MULTILINE`模式下的表现略有不同。在下面关于[标志](https://realpython.com/regex-python/#modified-regular-expression-matching-with-flags)的章节中，你会了解到更多关于`MULTILINE`模式的信息。

`$`
T1】

> 将一场比赛锚定到`<string>`的结尾。

当正则表达式解析器遇到`$`或`\Z`时，解析器的当前位置必须在搜索字符串的末尾，这样它才能找到匹配。位于`$`或`\Z`之前的内容必须构成搜索字符串的结尾:

>>>

```py
>>> re.search('bar$', 'foobar')
<_sre.SRE_Match object; span=(3, 6), match='bar'>
>>> print(re.search('bar$', 'barfoo'))
None

>>> re.search('bar\Z', 'foobar')
<_sre.SRE_Match object; span=(3, 6), match='bar'>
>>> print(re.search('bar\Z', 'barfoo'))
None
```

作为一个特例，`$`(而不是`\Z`)也匹配搜索字符串末尾的一个换行符之前的内容:

>>>

```py
>>> re.search('bar$', 'foobar\n')
<_sre.SRE_Match object; span=(3, 6), match='bar'>
```

在这个例子中，`'bar'`在技术上并不在搜索字符串的末尾，因为它后面跟了一个额外的换行符。但是 regex 解析器让它滑动，不管怎样都称它为匹配。这个例外不适用于`\Z`。

`$`和`\Z`在`MULTILINE`模式下的表现略有不同。关于`MULTILINE`模式的更多信息，参见下面关于[标志](https://realpython.com/regex-python/#modified-regular-expression-matching-with-flags)的部分。

`\b`

> 将匹配锚定到单词边界。

断言正则表达式解析器的当前位置必须在单词的开头或结尾。一个单词由一系列字母数字字符或下划线(`[a-zA-Z0-9_]`)组成，与`\w`字符类相同:

>>>

```py
 1>>> re.search(r'\bbar', 'foo bar')
 2<_sre.SRE_Match object; span=(4, 7), match='bar'>
 3>>> re.search(r'\bbar', 'foo.bar')
 4<_sre.SRE_Match object; span=(4, 7), match='bar'>
 5
 6>>> print(re.search(r'\bbar', 'foobar'))
 7None
 8
 9>>> re.search(r'foo\b', 'foo bar')
10<_sre.SRE_Match object; span=(0, 3), match='foo'>
11>>> re.search(r'foo\b', 'foo.bar')
12<_sre.SRE_Match object; span=(0, 3), match='foo'>
13
14>>> print(re.search(r'foo\b', 'foobar'))
15None
```

在上面的例子中，匹配发生在**的第 1 行和第 3 行**，因为在`'bar'`的开头有一个单词边界。在第 6 的**线上情况并非如此，所以匹配在那里失败。**

同样的，在**第 9 行和第 11 行**也有匹配，因为在`'foo'`结尾有一个字边界，而在**第 14 行**没有。

在`<regex>`的两端使用`\b`锚将使它作为一个完整的单词出现在搜索字符串中时匹配:

>>>

```py
>>> re.search(r'\bbar\b', 'foo bar baz')
<_sre.SRE_Match object; span=(4, 7), match='bar'>
>>> re.search(r'\bbar\b', 'foo(bar)baz')
<_sre.SRE_Match object; span=(4, 7), match='bar'>

>>> print(re.search(r'\bbar\b', 'foobarbaz'))
None
```

这是另一个将`<regex>`指定为原始字符串的例子，就像上面的例子一样。

因为`'\b'`是 Python 中字符串文字和正则表达式的转义序列，所以如果不使用原始字符串，上面的每次使用都需要双转义为`'\\b'`。这不会是世界末日，但原始字符串更整齐。

`\B`

> 将匹配锚定到非单词边界的位置。

`\B`做与`\b`相反的事。它断言正则表达式解析器的当前位置必须*而不是*在单词的开头或结尾:

>>>

```py
 1>>> print(re.search(r'\Bfoo\B', 'foo'))
 2None
 3>>> print(re.search(r'\Bfoo\B', '.foo.'))
 4None
 5
 6>>> re.search(r'\Bfoo\B', 'barfoobaz')
 7<_sre.SRE_Match object; span=(3, 6), match='foo'>
```

在这种情况下，匹配发生在**第 7 行**，因为在搜索字符串`'barfoobaz'`中的`'foo'`的开头或结尾不存在单词边界。

[*Remove ads*](/account/join/)

### 量词

一个**量词**元字符紧跟在一个`<regex>`的一部分之后，并指示该部分必须出现多少次才能成功匹配。

`*`

> 匹配前面正则表达式的零次或多次重复。

例如，`a*`匹配零个或多个`'a'`字符。这意味着它将匹配一个空字符串，`'a'`，`'aa'`，`'aaa'`，等等。

考虑这些例子:

>>>

```py
 1>>> re.search('foo-*bar', 'foobar')                     # Zero dashes 2<_sre.SRE_Match object; span=(0, 6), match='foobar'>
 3>>> re.search('foo-*bar', 'foo-bar')                    # One dash 4<_sre.SRE_Match object; span=(0, 7), match='foo-bar'>
 5>>> re.search('foo-*bar', 'foo--bar')                   # Two dashes 6<_sre.SRE_Match object; span=(0, 8), match='foo--bar'>
```

在**第 1 行**，在`'foo'`和`'bar'`之间有零个`'-'`字符。在**线 3** 上有一个，在**线 5** 上有两个。元字符序列`-*`在所有三种情况下都匹配。

你可能会在 Python 程序中遇到正则表达式`.*`。这匹配任何字符的零次或多次出现。换句话说，它基本上匹配任何字符序列，直到一个换行符。(记住`.`通配符元字符不匹配换行符。)

在本例中，`.*`匹配`'foo'`和`'bar'`之间的所有内容:

>>>

```py
>>> re.search('foo.*bar', '# foo $qux@grault % bar #')
<_sre.SRE_Match object; span=(2, 23), match='foo $qux@grault % bar'>
```

注意到 match 对象中包含的`span=`和`match=`信息了吗？

到目前为止，示例中的正则表达式已经指定了可预测长度的匹配。一旦你开始使用像`*`这样的量词，匹配的字符数量可能会有很大的变化，match 对象中的信息变得更加有用。

在本系列的下一篇教程中，您将了解更多关于如何访问存储在 match 对象中的信息。

`+`

> 匹配前面正则表达式的一次或多次重复。

这类似于`*`，但是量化的正则表达式必须至少出现一次:

>>>

```py
 1>>> print(re.search('foo-+bar', 'foobar'))              # Zero dashes 2None
 3>>> re.search('foo-+bar', 'foo-bar')                    # One dash 4<_sre.SRE_Match object; span=(0, 7), match='foo-bar'>
 5>>> re.search('foo-+bar', 'foo--bar')                   # Two dashes 6<_sre.SRE_Match object; span=(0, 8), match='foo--bar'>
```

记住上面的内容，因为`*`元字符允许`'-'`不出现，所以`foo-*bar`匹配字符串`'foobar'`。另一方面，`+`元字符要求至少出现一次`'-'`。这意味着在这种情况下**行 1** 上没有匹配。

`?`

> 匹配前面正则表达式的零次或一次重复。

同样，这类似于`*`和`+`，但是在这种情况下，只有前面的正则表达式出现一次或者根本没有出现，才匹配:

>>>

```py
 1>>> re.search('foo-?bar', 'foobar')                     # Zero dashes 2<_sre.SRE_Match object; span=(0, 6), match='foobar'>
 3>>> re.search('foo-?bar', 'foo-bar')                    # One dash 4<_sre.SRE_Match object; span=(0, 7), match='foo-bar'>
 5>>> print(re.search('foo-?bar', 'foo--bar'))            # Two dashes 6None
```

在这个例子中，在**线 1 和 3** 上有匹配。但是在第 5 的**行，有两个`'-'`字符，匹配失败。**

以下是展示所有三个量词元字符用法的更多示例:

>>>

```py
>>> re.match('foo[1-9]*bar', 'foobar')
<_sre.SRE_Match object; span=(0, 6), match='foobar'>
>>> re.match('foo[1-9]*bar', 'foo42bar')
<_sre.SRE_Match object; span=(0, 8), match='foo42bar'>

>>> print(re.match('foo[1-9]+bar', 'foobar'))
None
>>> re.match('foo[1-9]+bar', 'foo42bar')
<_sre.SRE_Match object; span=(0, 8), match='foo42bar'>

>>> re.match('foo[1-9]?bar', 'foobar')
<_sre.SRE_Match object; span=(0, 6), match='foobar'>
>>> print(re.match('foo[1-9]?bar', 'foo42bar'))
None
```

这次量化的正则表达式是字符类`[1-9]`而不是简单字符`'-'`。

`*?`
`+?`
T2】

> `*`、`+`和`?`量词的非贪婪(或懒惰)版本。

单独使用时，量词元字符`*`、`+`和`?`都是**贪婪**，这意味着它们产生最长的可能匹配。考虑这个例子:

>>>

```py
>>> re.search('<.*>', '%<foo> <bar> <baz>%')
<_sre.SRE_Match object; span=(1, 18), match='<foo> <bar> <baz>'>
```

正则表达式`<.*>`实际上意味着:

*   一个`'<'`人物
*   那么任何字符序列
*   然后是一个`'>'`字符

但是哪个`'>'`人物呢？有三种可能性:

1.  就在`'foo'`之后的那个
2.  就在`'bar'`之后的那个
3.  就在`'baz'`之后的那个

由于`*`元字符是贪婪的，它规定了可能的最长匹配，这包括了从`'>'`字符到`'baz'`之后的所有内容。从 match 对象可以看出，这是产生的匹配。

如果您想要最短的可能匹配，那么使用非贪婪元字符序列`*?`:

>>>

```py
>>> re.search('<.*?>', '%<foo> <bar> <baz>%')
<_sre.SRE_Match object; span=(1, 6), match='<foo>'>
```

在这种情况下，匹配以跟在`'foo'`后面的`'>'`字符结束。

**注意:**您可以用正则表达式`<[^>]*>`完成同样的事情，这意味着:

*   一个`'<'`人物
*   然后是除`'>'`以外的任何字符序列
*   然后是一个`'>'`字符

这是一些不支持惰性量词的旧解析器的唯一选项。令人高兴的是，Python 的`re`模块中的正则表达式解析器却不是这样。

还有懒惰版本的`+`和`?`量词:

>>>

```py
 1>>> re.search('<.+>', '%<foo> <bar> <baz>%')
 2<_sre.SRE_Match object; span=(1, 18), match='<foo> <bar> <baz>'>
 3>>> re.search('<.+?>', '%<foo> <bar> <baz>%')
 4<_sre.SRE_Match object; span=(1, 6), match='<foo>'>
 5
 6>>> re.search('ba?', 'baaaa')
 7<_sre.SRE_Match object; span=(0, 2), match='ba'>
 8>>> re.search('ba??', 'baaaa')
 9<_sre.SRE_Match object; span=(0, 1), match='b'>
```

**1 号线和 3 号线**的前两个例子与上面所示的例子类似，只是使用了`+`和`+?`而不是`*`和`*?`。

第 6 行**和第 8 行**的最后一个例子略有不同。一般来说，`?`元字符匹配零个或一个前面的正则表达式。贪婪的版本`?`匹配一个事件，所以`ba?`匹配`'b'`，后跟一个`'a'`。非贪婪版本`??`匹配零个出现，所以`ba??`只匹配`'b'`。

`{m}`

> 精确匹配前面正则表达式的`m`次重复。

这类似于`*`或`+`，但是它精确地指定了前面的正则表达式必须出现多少次才能成功匹配:

>>>

```py
>>> print(re.search('x-{3}x', 'x--x'))                # Two dashes
None

>>> re.search('x-{3}x', 'x---x')                      # Three dashes
<_sre.SRE_Match object; span=(0, 5), match='x---x'> 
>>> print(re.search('x-{3}x', 'x----x'))              # Four dashes
None
```

在这里，`x-{3}x`与`'x'`匹配，紧接着是`'-'`角色的三个实例，接着是另一个`'x'`。当`'x'`字符之间的破折号少于或多于三个时，匹配失败。

`{m,n}`

> 匹配前面正则表达式从`m`到`n`的任意重复次数，包括 T0 和 T1。

在下面的例子中，量化的`<regex>`是`-{2,4}`。当`'x'`字符之间有两个、三个或四个破折号时，匹配成功，否则匹配失败:

>>>

```py
>>> for i in range(1, 6):
...     s = f"x{'-' * i}x"
...     print(f'{i}  {s:10}', re.search('x-{2,4}x', s))
...
1  x-x        None
2  x--x       <_sre.SRE_Match object; span=(0, 4), match='x--x'> 3  x---x      <_sre.SRE_Match object; span=(0, 5), match='x---x'> 4  x----x     <_sre.SRE_Match object; span=(0, 6), match='x----x'> 5  x-----x    None
```

省略`m`意味着一个`0`的下界，省略`n`意味着一个无限的上界:

| 正则表达式 | 比赛 | 相同 |
| --- | --- | --- |
| `<regex>{,n}` | 小于或等于`n`的`<regex>`的任何重复次数 | `<regex>{0,n}` |
| `<regex>{m,}` | 大于或等于`m`的`<regex>`的任意重复次数 | `----` |
| `<regex>{,}` | 重复任意次数的`<regex>` | `<regex>{0,}`
T1】 |

如果您省略了所有的`m`、`n`和逗号，那么花括号不再作为元字符。`{}`仅匹配文字字符串`'{}'`:

>>>

```py
>>> re.search('x{}y', 'x{}y')
<_sre.SRE_Match object; span=(0, 4), match='x{}y'>
```

事实上，要有任何特殊意义，带花括号的序列必须符合以下模式之一，其中`m`和`n`为非负整数:

*   `{m,n}`
*   `{m,}`
*   `{,n}`
*   `{,}`

否则，它匹配字面意思:

>>>

```py
>>> re.search('x{foo}y', 'x{foo}y')
<_sre.SRE_Match object; span=(0, 7), match='x{foo}y'>
>>> re.search('x{a:b}y', 'x{a:b}y')
<_sre.SRE_Match object; span=(0, 7), match='x{a:b}y'>
>>> re.search('x{1,3,5}y', 'x{1,3,5}y')
<_sre.SRE_Match object; span=(0, 9), match='x{1,3,5}y'>
>>> re.search('x{foo,bar}y', 'x{foo,bar}y')
<_sre.SRE_Match object; span=(0, 11), match='x{foo,bar}y'>
```

在本教程的后面，当您了解到`DEBUG`标志时，您将看到如何确认这一点。

`{m,n}?`

> `{m,n}`的非贪(懒)版本。

`{m,n}`将匹配尽可能多的字符，`{m,n}?`将匹配尽可能少的字符:

>>>

```py
>>> re.search('a{3,5}', 'aaaaaaaa')
<_sre.SRE_Match object; span=(0, 5), match='aaaaa'>

>>> re.search('a{3,5}?', 'aaaaaaaa')
<_sre.SRE_Match object; span=(0, 3), match='aaa'>
```

在这种情况下，`a{3,5}`产生最长的可能匹配，因此它匹配五个`'a'`字符。`a{3,5}?`产生最短的匹配，所以它匹配三个。

[*Remove ads*](/account/join/)

### 分组构造和反向引用

分组构造将 Python 中的正则表达式分解成子表达式或组。这有两个目的:

1.  **分组:**一个组代表一个单一的句法实体。附加元字符作为一个单元应用于整个组。
2.  **捕获:**一些分组结构也捕获搜索字符串中匹配组中子表达式的部分。您可以稍后通过几种不同的机制检索捕获的匹配。

下面来看看分组和采集是如何工作的。

`(<regex>)`

> 定义子表达式或组。

这是最基本的分组结构。括号中的正则表达式只匹配括号中的内容:

>>>

```py
>>> re.search('(bar)', 'foo bar baz')
<_sre.SRE_Match object; span=(4, 7), match='bar'>

>>> re.search('bar', 'foo bar baz')
<_sre.SRE_Match object; span=(4, 7), match='bar'>
```

作为一个正则表达式，`(bar)`匹配字符串`'bar'`，与不带括号的正则表达式`bar`一样。

#### 将一个组视为一个单元

组后面的量词元字符将组中指定的整个子表达式作为一个单元进行操作。

例如，以下示例匹配一个或多个字符串`'bar'`:

>>>

```py
>>> re.search('(bar)+', 'foo bar baz')
<_sre.SRE_Match object; span=(4, 7), match='bar'>
>>> re.search('(bar)+', 'foo barbar baz')
<_sre.SRE_Match object; span=(4, 10), match='barbar'>
>>> re.search('(bar)+', 'foo barbarbarbar baz')
<_sre.SRE_Match object; span=(4, 16), match='barbarbarbar'>
```

下面是有分组括号和没有分组括号的两种正则表达式之间的区别:

| 正则表达式 | 解释 | 比赛 | 例子 |
| --- | --- | --- | --- |
| `bar+` | `+`元字符仅适用于字符`'r'`。 | `'ba'`之后是一次或多次出现的`'r'` | `'bar'`
`'barr'`
T2】 |
| `(bar)+` | `+`元字符适用于整个字符串`'bar'`。 | 一次或多次出现`'bar'` | `'bar'`
`'barbar'`
T2】 |

现在看一个更复杂的例子。正则表达式`(ba[rz]){2,4}(qux)?`将`2`与`'bar'`或`'baz'`的`4`匹配，可选地后跟`'qux'`:

>>>

```py
>>> re.search('(ba[rz]){2,4}(qux)?', 'bazbarbazqux')
<_sre.SRE_Match object; span=(0, 12), match='bazbarbazqux'>
>>> re.search('(ba[rz]){2,4}(qux)?', 'barbar')
<_sre.SRE_Match object; span=(0, 6), match='barbar'>
```

以下示例显示您可以嵌套分组括号:

>>>

```py
>>> re.search('(foo(bar)?)+(\d\d\d)?', 'foofoobar')
<_sre.SRE_Match object; span=(0, 9), match='foofoobar'>
>>> re.search('(foo(bar)?)+(\d\d\d)?', 'foofoobar123')
<_sre.SRE_Match object; span=(0, 12), match='foofoobar123'>
>>> re.search('(foo(bar)?)+(\d\d\d)?', 'foofoo123')
<_sre.SRE_Match object; span=(0, 9), match='foofoo123'>
```

正则表达式`(foo(bar)?)+(\d\d\d)?`相当复杂，所以让我们把它分成更小的部分:

| 正则表达式 | 比赛 |
| --- | --- |
| `foo(bar)?` | `'foo'`可选后接`'bar'` |
| `(foo(bar)?)+` | 上述一个或多个事件 |
| `\d\d\d` | 三个十进制数字字符 |
| `(\d\d\d)?` | 上述情况出现零次或一次 |

把它们串在一起，你会得到:至少出现一次`'foo'`，后面可选地跟着`'bar'`，后面可选地跟着三个十进制数字字符。

如您所见，您可以使用分组括号在 Python 中构造非常复杂的正则表达式。

#### 捕捉组

分组并不是分组构造服务的唯一有用的目的。大多数(但不是全部)分组结构也捕获搜索字符串中与组匹配的部分。您可以检索捕获的部分，或者以后以几种不同的方式引用它。

还记得`re.search()`返回的 match 对象吗？为 match 对象定义了两个方法来提供对捕获组的访问:`.groups()`和`.group()`。

`m.groups()`

> 返回一个元组，其中包含从正则表达式匹配中捕获的所有组。

考虑这个例子:

>>>

```py
>>> m = re.search('(\w+),(\w+),(\w+)', 'foo,quux,baz')
>>> m
<_sre.SRE_Match object; span=(0, 12), match='foo:quux:baz'>
```

三个`(\w+)`表达式中的每一个都匹配一个单词字符序列。完整的正则表达式`(\w+),(\w+),(\w+)`将搜索字符串分成三个逗号分隔的标记。

因为`(\w+)`表达式使用分组括号，所以对应的匹配标记是**捕获的**。要访问捕获的匹配，您可以使用`.groups()`，它返回一个[元组](https://realpython.com/python-lists-tuples/#python-tuples)，其中按顺序包含所有捕获的匹配:

>>>

```py
>>> m.groups()
('foo', 'quux', 'baz')
```

请注意，元组包含标记，但不包含搜索字符串中出现的逗号。这是因为组成标记的单词字符在分组括号内，但逗号不在。您在返回的标记之间看到的逗号是用于分隔元组中的值的标准分隔符。

`m.group(<n>)`

> 返回一个包含`<n>` <sup>`th`</sup> 捕获匹配的字符串。

对于一个参数，`.group()`返回一个捕获的匹配。请注意，参数是从 1 开始的，而不是从 0 开始的。所以，`m.group(1)`指第一个被捕获的匹配，`m.group(2)`指第二个，依此类推:

>>>

```py
>>> m = re.search('(\w+),(\w+),(\w+)', 'foo,quux,baz')
>>> m.groups()
('foo', 'quux', 'baz')

>>> m.group(1)
'foo'
>>> m.group(2)
'quux'
>>> m.group(3)
'baz'
```

由于捕获的匹配项的编号是从 1 开始的，并且没有任何编号为 0 的组，`m.group(0)`具有特殊的含义:

>>>

```py
>>> m.group(0)
'foo,quux,baz'
>>> m.group()
'foo,quux,baz'
```

`m.group(0)`返回整个匹配，`m.group()`也做同样的事情。

`m.group(<n1>, <n2>, ...)`

> 返回包含指定的捕获匹配项的元组。

使用多个参数，`.group()`返回一个元组，该元组包含按给定顺序捕获的指定匹配项:

>>>

```py
>>> m.groups()
('foo', 'quux', 'baz')

>>> m.group(2, 3)
('quux', 'baz')
>>> m.group(3, 2, 1)
('baz', 'quux', 'foo')
```

这只是方便的速记。您可以自己创建匹配元组:

>>>

```py
>>> m.group(3, 2, 1)
('baz', 'qux', 'foo')
>>> (m.group(3), m.group(2), m.group(1))
('baz', 'qux', 'foo')
```

所示的两条语句在功能上是等效的。

#### 反向引用

您可以稍后在同一个正则表达式中使用一个称为**反向引用**的特殊元字符序列来匹配先前捕获的组。

`\<n>`

> 匹配以前捕获的组的内容。

在 Python 的正则表达式中，序列`\<n>`，其中`<n>`是从`1`到`99`的整数，匹配`<n>` <sup>`th`</sup> 捕获组的内容。

下面是一个正则表达式，它匹配一个单词，后跟一个逗号，再跟同一个单词:

>>>

```py
 1>>> regex = r'(\w+),\1'
 2
 3>>> m = re.search(regex, 'foo,foo') 4>>> m
 5<_sre.SRE_Match object; span=(0, 7), match='foo,foo'>
 6>>> m.group(1)
 7'foo'
 8
 9>>> m = re.search(regex, 'qux,qux') 10>>> m
11<_sre.SRE_Match object; span=(0, 7), match='qux,qux'>
12>>> m.group(1)
13'qux'
14
15>>> m = re.search(regex, 'foo,qux') 16>>> print(m)
17None
```

在第一个示例中，在**第 3 行**，`(\w+)`匹配字符串`'foo'`的第一个实例，并将其保存为第一个捕获的组。逗号完全匹配。那么`\1`是对第一个捕获组的反向引用，并再次匹配`'foo'`。第二个例子，在第 9 行的**上，除了`(\w+)`匹配`'qux'`之外是相同的。**

最后一个例子，在第 15 行的**上，没有匹配，因为逗号前面的和后面的不一样，所以`\1`反向引用不匹配。**

**注意:**任何时候在 Python 中使用带有编号反向引用的正则表达式时，最好将其指定为原始字符串。否则，解释器可能会将反向引用与八进制值的[混淆。](https://en.wikipedia.org/wiki/Octal)

考虑这个例子:

>>>

```py
>>> print(re.search('([a-z])#\1', 'd#d'))
None
```

正则表达式`([a-z])#\1`匹配一个小写字母，后面是`'#'`，后面是相同的小写字母。本例中的字符串是`'d#d'`，应该匹配。但是匹配失败了，因为 Python 将反向引用`\1`误解为八进制值为 1 的字符:

>>>

```py
>>> oct(ord('\1'))
'0o1'
```

如果将正则表达式指定为原始字符串，您将获得正确的匹配:

>>>

```py
>>> re.search(r'([a-z])#\1', 'd#d')
<_sre.SRE_Match object; span=(0, 3), match='d#d'>
```

请记住，只要您的正则表达式包含包含反斜杠的元字符序列，就要考虑使用原始字符串。

编号的反向引用是从 1 开始的，就像`.group()`的参数一样。通过反向引用只能访问前九十九个捕获的组。解释器会将`\100`视为`'@'`字符，其八进制值为 100。

#### 其他分组结构

上面显示的`(<regex>)`元字符序列是在 Python 的正则表达式中执行分组的最直接的方式。下一节将向您介绍一些增强的分组构造，这些构造允许您调整分组发生的时间和方式。

`(?P<name><regex>)`

> 创建命名的捕获组。

这个元字符序列类似于分组括号，因为它创建了一个组匹配`<regex>`,可通过 match 对象或后续反向引用访问。这种情况下的不同之处在于，您通过给定的符号`<name>`而不是它的编号来引用匹配的组。

之前，您看到了这个示例，其中有三个被捕获的组，编号分别为`1`、`2`和`3`:

>>>

```py
>>> m = re.search('(\w+),(\w+),(\w+)', 'foo,quux,baz')
>>> m.groups()
('foo', 'quux', 'baz')

>>> m.group(1, 2, 3)
('foo', 'quux', 'baz')
```

除了这些组具有符号名称`w1`、`w2`和`w3`之外，下面的代码实际上做了同样的事情:

>>>

```py
>>> m = re.search('(?P<w1>\w+),(?P<w2>\w+),(?P<w3>\w+)', 'foo,quux,baz')
>>> m.groups()
('foo', 'quux', 'baz')
```

您可以通过符号名称来引用这些捕获的组:

>>>

```py
>>> m.group('w1')
'foo'
>>> m.group('w3')
'baz'
>>> m.group('w1', 'w2', 'w3')
('foo', 'quux', 'baz')
```

如果您愿意，仍然可以通过数字访问带有符号名称的组:

>>>

```py
>>> m = re.search('(?P<w1>\w+),(?P<w2>\w+),(?P<w3>\w+)', 'foo,quux,baz')

>>> m.group('w1')
'foo'
>>> m.group(1)
'foo'

>>> m.group('w1', 'w2', 'w3')
('foo', 'quux', 'baz')
>>> m.group(1, 2, 3)
('foo', 'quux', 'baz')
```

用这个构造指定的任何`<name>`必须符合一个 [Python 标识符](https://realpython.com/python-variables/#variable-names)的规则，并且每个`<name>`在每个正则表达式中只能出现一次。

`(?P=<name>)`

> 匹配以前捕获的命名组的内容。

`(?P=<name>)`元字符序列是一个反向引用，类似于`\<n>`，除了它引用一个命名的组而不是一个编号的组。

这又是上面的例子，它使用一个带编号的反向引用来匹配一个单词，后跟一个逗号，再跟同一个单词:

>>>

```py
>>> m = re.search(r'(\w+),\1', 'foo,foo')
>>> m
<_sre.SRE_Match object; span=(0, 7), match='foo,foo'>
>>> m.group(1)
'foo'
```

下面的代码使用命名组和反向引用来做同样的事情:

>>>

```py
>>> m = re.search(r'(?P<word>\w+),(?P=word)', 'foo,foo')
>>> m
<_sre.SRE_Match object; span=(0, 7), match='foo,foo'>
>>> m.group('word')
'foo'
```

`(?P=<word>\w+)`匹配`'foo'`并保存为一个名为`word`的捕获组。同样，逗号字面上匹配。那么`(?P=word)`是对已命名捕获的反向引用，并再次匹配`'foo'`。

**注意:**当创建一个命名的组时，在`name`周围需要尖括号(`<`和`>`),但当以后引用它时，无论是通过反向引用还是通过`.group()`:

>>>

```py
>>> m = re.match(r'(?P<num>\d+)\.(?P=num)', '135.135') >>> m
<_sre.SRE_Match object; span=(0, 7), match='135.135'>

>>> m.group('num') '135'
```

在这里，`(?P` **`<num>`** `\d+)`创建了被捕获的组。但是对应的反向引用是`(?P=` **`num`** `)`不带尖括号。

`(?:<regex>)`

> 创建非捕获组。

`(?:<regex>)`就像`(<regex>)`一样，它匹配指定的`<regex>`。但是`(?:<regex>)`没有捕获匹配供以后检索:

>>>

```py
>>> m = re.search('(\w+),(?:\w+),(\w+)', 'foo,quux,baz')
>>> m.groups()
('foo', 'baz')

>>> m.group(1)
'foo'
>>> m.group(2)
'baz'
```

在这个例子中，中间的单词`'quux'`位于非捕获括号内，所以它在捕获组的元组中是缺失的。它不能从 match 对象中检索，也不能通过反向引用进行引用。

为什么要定义一个组，而不是捕获它？

记住，正则表达式解析器将把分组括号内的`<regex>`视为一个单元。您可能会遇到这样的情况，您需要这个分组特性，但是您不需要在以后对该值做任何事情，所以您不需要捕获它。如果您使用非捕获分组，那么捕获组的元组不会被您实际上不需要保留的值弄得乱七八糟。

此外，捕获一个组需要一些时间和内存。如果执行匹配的代码执行了多次，并且您没有捕获以后不打算使用的组，那么您可能会看到一点性能优势。

`(?(<n>)<yes-regex>|<no-regex>)`
T1】

> 指定条件匹配。

条件匹配根据给定组是否存在来匹配两个指定正则表达式之一:

*   如果编号为`<n>`的组存在，则`(?(<n>)<yes-regex>|<no-regex>)`与`<yes-regex>`匹配。否则匹配`<no-regex>`。

*   如果名为`<name>`的组存在，`(?(<name>)<yes-regex>|<no-regex>)`与`<yes-regex>`匹配。否则匹配`<no-regex>`。

通过一个例子可以更好地说明条件匹配。考虑这个正则表达式:

```py
regex = r'^(###)?foo(?(1)bar|baz)'
```

以下是这个正则表达式的各个部分，并附有一些解释:

1.  `^(###)?`表示搜索字符串可选地以`'###'`开头。如果是，那么`###`周围的分组括号将创建一个编号为`1`的组。否则，这样的团体将不复存在。
2.  下一部分`foo`，字面上匹配字符串`'foo'`。
3.  最后，如果组`1`存在，`(?(1)bar|baz)`与`'bar'`匹配，如果不存在，`'baz'`与组`'bar'`匹配。

以下代码块演示了上述正则表达式在几个不同的 Python 代码片段中的用法:

示例 1:

>>>

```py
>>> re.search(regex, '###foobar')
<_sre.SRE_Match object; span=(0, 9), match='###foobar'>
```

搜索字符串`'###foobar'`确实以`'###'`开头，所以解析器创建了一个编号为`1`的组。条件匹配然后与匹配的`'bar'`相对。

示例 2:

>>>

```py
>>> print(re.search(regex, '###foobaz'))
None
```

搜索字符串`'###foobaz'`确实以`'###'`开头，所以解析器创建了一个编号为`1`的组。条件匹配然后对`'bar'`，不匹配。

示例 3:

>>>

```py
>>> print(re.search(regex, 'foobar'))
None
```

搜索字符串`'foobar'`不是以`'###'`开头，所以没有编号为`1`的组。条件匹配然后对`'baz'`，不匹配。

示例 4:

>>>

```py
>>> re.search(regex, 'foobaz')
<_sre.SRE_Match object; span=(0, 6), match='foobaz'>
```

搜索字符串`'foobaz'`不是以`'###'`开头，所以没有编号为`1`的组。条件匹配然后与匹配的`'baz'`相对。

下面是另一个条件匹配，它使用命名组而不是编号组:

>>>

```py
>>> regex = r'^(?P<ch>\W)?foo(?(ch)(?P=ch)|)$'
```

这个正则表达式匹配字符串`'foo'`，前面是一个非单词字符，后面是相同的非单词字符，或者字符串`'foo'`本身。

同样，让我们把它分成几个部分:

| 正则表达式 | 比赛 |
| --- | --- |
| `^` | 字符串的开头 |
| `(?P<ch>\W)` | 在名为`ch`的组中捕获的单个非单词字符 |
| `(?P<ch>\W)?` | 上述情况出现零次或一次 |
| `foo` | 文字字符串`'foo'` |
| `(?(ch)(?P=ch)&#124;)` | 名为`ch`的组的内容，如果存在，则为空字符串 |
| `$` | 字符串的结尾 |

如果一个非单词字符在`'foo'`之前，那么解析器会创建一个包含该字符的名为`ch`的组。条件匹配然后与`<yes-regex>`匹配，也就是`(?P=ch)`，同样的角色。这意味着同一个角色也必须跟随`'foo'`才能赢得整个比赛。

如果`'foo'`前面没有非单词字符，那么解析器不会创建组`ch`。`<no-regex>`是空字符串，这意味着为了整个匹配成功，在`'foo'`之后不能有任何东西。因为`^`和`$`锚定了整个正则表达式，所以字符串必须正好等于`'foo'`。

以下是在 Python 代码中使用此正则表达式进行搜索的一些示例:

>>>

```py
 1>>> re.search(regex, 'foo')
 2<_sre.SRE_Match object; span=(0, 3), match='foo'>
 3>>> re.search(regex, '#foo#')
 4<_sre.SRE_Match object; span=(0, 5), match='#foo#'>
 5>>> re.search(regex, '@foo@')
 6<_sre.SRE_Match object; span=(0, 5), match='@foo@'>
 7
 8>>> print(re.search(regex, '#foo'))
 9None
10>>> print(re.search(regex, 'foo@'))
11None
12>>> print(re.search(regex, '#foo@'))
13None
14>>> print(re.search(regex, '@foo#'))
15None
```

在**1 号线**，`'foo'`是单独的。在**第 3 行和第 5 行**上，相同的非文字字符在`'foo'`之前和之后。正如广告所说，这些比赛成功了。

在其余情况下，匹配失败。

Python 中的条件正则表达式非常深奥，很难理解。如果你找到了使用它的理由，那么你可以通过多次单独的`re.search()`调用来完成同样的目标，并且你的代码阅读和理解起来也不会那么复杂。

[*Remove ads*](/account/join/)

### 前视和后视断言

**Lookahead** 和**look ahead**断言根据搜索字符串中解析器当前位置的正后方(左侧)或正前方(右侧)来确定 Python 中正则表达式匹配的成功或失败。

像锚一样，前视和后视断言是零宽度断言，所以它们不消耗任何搜索字符串。此外，即使它们包含括号并执行分组，它们也不会捕获它们匹配的内容。

`(?=<lookahead_regex>)`

> 创建一个正向前瞻断言。

`(?=<lookahead_regex>)`断言正则表达式解析器当前位置之后的内容必须与`<lookahead_regex>`匹配:

>>>

```py
>>> re.search('foo(?=[a-z])', 'foobar')
<_sre.SRE_Match object; span=(0, 3), match='foo'>
```

前瞻断言`(?=[a-z])`指定跟随`'foo'`的必须是小写字母字符。在本例中，是人物`'b'`，所以找到了匹配。

另一方面，在下一个示例中，前瞻失败。`'foo'`之后的下一个字符是`'1'`，所以没有匹配:

>>>

```py
>>> print(re.search('foo(?=[a-z])', 'foo123'))
None
```

lookahead 的独特之处在于，搜索字符串中匹配`<lookahead_regex>`的部分不会被消耗，它也不是返回的 match 对象的一部分。

再看一下第一个例子:

>>>

```py
>>> re.search('foo(?=[a-z])', 'foobar')
<_sre.SRE_Match object; span=(0, 3), match='foo'>
```

正则表达式解析器只向前看跟在`'foo'`后面的`'b'`，但是还没有跳过它。您可以看出`'b'`不被认为是匹配的一部分，因为匹配对象显示的是`match='foo'`。

与之相比，一个类似的例子使用分组括号，但没有前瞻:

>>>

```py
>>> re.search('foo([a-z])', 'foobar')
<_sre.SRE_Match object; span=(0, 4), match='foob'>
```

这一次，正则表达式使用了`'b'`，它成为最终匹配的一部分。

下面是另一个例子，说明了 Python 中的前瞻与传统正则表达式的区别:

>>>

```py
 1>>> m = re.search('foo(?=[a-z])(?P<ch>.)', 'foobar') 2>>> m.group('ch')
 3'b'
 4
 5>>> m = re.search('foo([a-z])(?P<ch>.)', 'foobar') 6>>> m.group('ch')
 7'a'
```

在第一次搜索中，在**行 1** 上，解析器如下进行:

1.  正则表达式的第一部分`foo`，匹配并使用搜索字符串`'foobar'`中的`'foo'`。
2.  下一部分`(?=[a-z])`，是一个匹配`'b'`的前瞻，但是解析器不会前进超过`'b'`。
3.  最后，`(?P<ch>.)`匹配下一个可用的单个字符，即`'b'`，并将其捕获到一个名为`ch`的组中。

`m.group('ch')`呼叫确认名为`ch`的群组包含`'b'`。

将其与第 5 行**上的搜索进行比较，后者不包含前瞻:**

1.  和第一个例子一样，正则表达式的第一部分`foo`，匹配并使用搜索字符串`'foobar'`中的`'foo'`。
2.  下一部分`([a-z])`，匹配并消耗`'b'`，解析器前进通过`'b'`。
3.  最后，`(?P<ch>.)`匹配下一个可用的单个字符，现在是`'a'`。

`m.group('ch')`确认，在这种情况下，名为`ch`的组包含`'a'`。

`(?!<lookahead_regex>)`

> 创建一个负的前瞻断言。

`(?!<lookahead_regex>)`断言正则表达式解析器当前位置后面的内容必须*不*匹配`<lookahead_regex>`。

以下是您之前看到的正面前瞻示例，以及它们的负面前瞻对应示例:

>>>

```py
 1>>> re.search('foo(?=[a-z])', 'foobar')
 2<_sre.SRE_Match object; span=(0, 3), match='foo'>
 3>>> print(re.search('foo(?![a-z])', 'foobar')) 4None
 5
 6>>> print(re.search('foo(?=[a-z])', 'foo123'))
 7None
 8>>> re.search('foo(?![a-z])', 'foo123') 9<_sre.SRE_Match object; span=(0, 3), match='foo'>
```

第 3 行和第 8 行上的否定前瞻断言规定`'foo'`后面的不应该是小写字母字符。这在**线 3** 上失败，但在**线 8** 上成功。这与相应的正向前瞻断言的情况相反。

与正向前瞻一样，与负向前瞻匹配的内容不是返回的 match 对象的一部分，也不会被使用。

`(?<=<lookbehind_regex>)`

> 创建正的后视断言。

`(?<=<lookbehind_regex>)`断言正则表达式解析器当前位置之前的内容必须匹配`<lookbehind_regex>`。

在下面的例子中，lookbehind 断言指定`'foo'`必须在`'bar'`之前:

>>>

```py
>>> re.search('(?<=foo)bar', 'foobar')
<_sre.SRE_Match object; span=(3, 6), match='bar'>
```

这里就是这种情况，所以匹配成功。与 lookahead 断言一样，搜索字符串中与 look ahead 匹配的部分不会成为最终匹配的一部分。

下一个示例无法匹配，因为 lookbehind 要求`'qux'`在`'bar'`之前:

>>>

```py
>>> print(re.search('(?<=qux)bar', 'foobar'))
None
```

对后视断言有一个限制，不适用于前视断言。lookbehind 断言中的`<lookbehind_regex>`必须指定固定长度的匹配。

例如，下面是不允许的，因为由`a+`匹配的字符串的长度是不确定的:

>>>

```py
>>> re.search('(?<=a+)def', 'aaadef') Traceback (most recent call last):
  File "<pyshell#72>", line 1, in <module>
    re.search('(?<=a+)def', 'aaadef')
  File "C:\Python36\lib\re.py", line 182, in search
    return _compile(pattern, flags).search(string)
  File "C:\Python36\lib\re.py", line 301, in _compile
    p = sre_compile.compile(pattern, flags)
  File "C:\Python36\lib\sre_compile.py", line 566, in compile
    code = _code(p, flags)
  File "C:\Python36\lib\sre_compile.py", line 551, in _code
    _compile(code, p.data, flags)
  File "C:\Python36\lib\sre_compile.py", line 160, in _compile
    raise error("look-behind requires fixed-width pattern")
sre_constants.error: look-behind requires fixed-width pattern
```

然而，这是可以的:

>>>

```py
>>> re.search('(?<=a{3})def', 'aaadef') <_sre.SRE_Match object; span=(3, 6), match='def'>
```

任何与`a{3}`匹配的都有固定的长度 3，所以`a{3}`在后视断言中是有效的。

`(?<!<lookbehind_regex>)`

> 创建负的后视断言。

`(?<!<lookbehind_regex>)`断言正则表达式解析器当前位置之前的内容必须*不*匹配`<lookbehind_regex>`:

>>>

```py
>>> print(re.search('(?<!foo)bar', 'foobar'))
None

>>> re.search('(?<!qux)bar', 'foobar')
<_sre.SRE_Match object; span=(3, 6), match='bar'>
```

与肯定的后视断言一样，`<lookbehind_regex>`必须指定固定长度的匹配。

[*Remove ads*](/account/join/)

### 其他元字符

还需要介绍几个元字符序列。这些是分散的元字符，显然不属于已经讨论过的任何类别。

`(?#...)`

> 指定注释。

正则表达式解析器忽略序列`(?#...)`中包含的任何内容:

>>>

```py
>>> re.search('bar(?#This is a comment) *baz', 'foo bar baz qux')
<_sre.SRE_Match object; span=(4, 11), match='bar baz'>
```

这允许您在 Python 中的正则表达式内指定文档，如果正则表达式特别长，这可能特别有用。

竖条或竖线(`|`)

> 指定要匹配的一组备选项。

一个表达式的形式`<regex`<sub>`1`</sub>`>|<regex`<sub>`2`</sub>`>|...|<regex`<sub>`n`</sub>`>`最多匹配一个指定的`<regex` <sub>`i`</sub> `>`表达式:

>>>

```py
>>> re.search('foo|bar|baz', 'bar')
<_sre.SRE_Match object; span=(0, 3), match='bar'>

>>> re.search('foo|bar|baz', 'baz')
<_sre.SRE_Match object; span=(0, 3), match='baz'>

>>> print(re.search('foo|bar|baz', 'quux'))
None
```

这里，`foo|bar|baz`将匹配`'foo'`、`'bar'`或`'baz'`中的任何一个。您可以使用`|`分隔任意数量的正则表达式。

交替是非贪婪的。正则表达式解析器从左到右查看由`|`分隔的表达式，并返回找到的第一个匹配。剩余的表达式不会被测试，即使其中一个会产生更长的匹配:

>>>

```py
 1>>> re.search('foo', 'foograult')
 2<_sre.SRE_Match object; span=(0, 3), match='foo'>
 3>>> re.search('grault', 'foograult')
 4<_sre.SRE_Match object; span=(3, 9), match='grault'>
 5
 6>>> re.search('foo|grault', 'foograult')
 7<_sre.SRE_Match object; span=(0, 3), match='foo'>
```

在这种情况下，在**线 6** 、`'foo|grault'`上指定的模式将在`'foo'`或`'grault'`上匹配。返回的匹配是`'foo'`，因为当从左到右扫描时，它首先出现，即使`'grault'`是一个更长的匹配。

您可以组合交替、分组和任何其他元字符来实现您需要的任何级别的复杂性。在下面的例子中，`(foo|bar|baz)+`表示一个或多个字符串`'foo'`、`'bar'`或`'baz'`的序列:

>>>

```py
>>> re.search('(foo|bar|baz)+', 'foofoofoo')
<_sre.SRE_Match object; span=(0, 9), match='foofoofoo'>
>>> re.search('(foo|bar|baz)+', 'bazbazbazbaz')
<_sre.SRE_Match object; span=(0, 12), match='bazbazbazbaz'>
>>> re.search('(foo|bar|baz)+', 'barbazfoo')
<_sre.SRE_Match object; span=(0, 9), match='barbazfoo'>
```

在下一个例子中，`([0-9]+|[a-f]+)`表示一个或多个十进制数字字符的序列，或者一个或多个字符的序列`'a-f'`:

>>>

```py
>>> re.search('([0-9]+|[a-f]+)', '456')
<_sre.SRE_Match object; span=(0, 3), match='456'>
>>> re.search('([0-9]+|[a-f]+)', 'ffda')
<_sre.SRE_Match object; span=(0, 4), match='ffda'>
```

有了`re`模块支持的所有元字符，实际上就没有限制了。

就这些了，伙计们！

这就完成了我们对 Python 的`re`模块所支持的正则表达式元字符的浏览。(实际上，并不完全是这样——在下面关于标志的讨论中，您将了解到更多的落伍者。)

有很多东西需要消化，但是一旦您熟悉了 Python 中的正则表达式语法，您可以执行的模式匹配的复杂性几乎是无限的。当您编写代码来处理文本数据时，这些工具非常方便。

如果你是正则表达式的新手，想要更多地练习使用它们，或者如果你正在开发一个使用正则表达式的应用程序，并且想要交互地测试它，那么请访问[正则表达式 101](https://regex101.com) 网站。真的很酷！

[*Remove ads*](/account/join/)

## 修改了与标志匹配的正则表达式

`re`模块中的大多数函数都有一个可选的`<flags>`参数。这包括你现在非常熟悉的功能，`re.search()`。

`re.search(<regex>, <string>, <flags>)`

> 应用指定的修饰符`<flags>`，扫描字符串中的正则表达式匹配。

标志修改正则表达式解析行为，允许您进一步优化模式匹配。

### 支持的正则表达式标志

下表简要总结了可用的标志。除了`re.DEBUG`之外的所有标志都有一个简短的单字母名称和一个较长的完整单词名称:

| 简称 | 长名字 | 影响 |
| --- | --- | --- |
| `re.I` | `re.IGNORECASE` | 使字母字符的匹配不区分大小写 |
| `re.M` | `re.MULTILINE` | 使字符串开头和字符串结尾的锚点与嵌入的换行符匹配 |
| `re.S` | `re.DOTALL` | 使点元字符匹配换行符 |
| `re.X` | `re.VERBOSE` | 允许在正则表达式中包含空白和注释 |
| `----` | `re.DEBUG` | 使正则表达式分析器向控制台显示调试信息 |
| `re.A` | `re.ASCII` | 用于字符分类的指定 ascii 编码 |
| `re.U` | `re.UNICODE` | 字符分类的指定 unicode 编码 |
| `re.L` | `re.LOCALE` | 基于当前区域设置指定字符分类的编码 |

以下部分更详细地描述了这些标志如何影响匹配行为。

`re.I`
T1】

> 使匹配不区分大小写。

当`IGNORECASE`生效时，字符匹配不区分大小写:

>>>

```py
 1>>> re.search('a+', 'aaaAAA')
 2<_sre.SRE_Match object; span=(0, 3), match='aaa'>
 3>>> re.search('A+', 'aaaAAA')
 4<_sre.SRE_Match object; span=(3, 6), match='AAA'>
 5
 6>>> re.search('a+', 'aaaAAA', re.I)
 7<_sre.SRE_Match object; span=(0, 6), match='aaaAAA'>
 8>>> re.search('A+', 'aaaAAA', re.IGNORECASE)
 9<_sre.SRE_Match object; span=(0, 6), match='aaaAAA'>
```

在**第 1 行**的搜索中，`a+`只匹配`'aaaAAA'`的前三个字符。同样，在第 3 行的**上，`A+`只匹配最后三个字符。但是在随后的搜索中，解析器忽略大小写，所以`a+`和`A+`匹配整个字符串。**

`IGNORECASE`影响包括字符类别的字母匹配:

>>>

```py
>>> re.search('[a-z]+', 'aBcDeF')
<_sre.SRE_Match object; span=(0, 1), match='a'>
>>> re.search('[a-z]+', 'aBcDeF', re.I)
<_sre.SRE_Match object; span=(0, 6), match='aBcDeF'>
```

当情况重要时，`[a-z]+`匹配的`'aBcDeF'`的最长部分就是最初的`'a'`。指定`re.I`使得搜索不区分大小写，所以`[a-z]+`匹配整个字符串。

`re.M`
T1】

> 使字符串开头和字符串结尾的锚点在嵌入的换行符处匹配。

默认情况下，`^`(字符串开头)和`$`(字符串结尾)锚点仅在搜索字符串的开头和结尾匹配:

>>>

```py
>>> s = 'foo\nbar\nbaz'

>>> re.search('^foo', s) <_sre.SRE_Match object; span=(0, 3), match='foo'> >>> print(re.search('^bar', s))
None
>>> print(re.search('^baz', s))
None

>>> print(re.search('foo$', s))
None
>>> print(re.search('bar$', s))
None
>>> re.search('baz$', s) <_sre.SRE_Match object; span=(8, 11), match='baz'>
```

在这种情况下，即使搜索字符串`'foo\nbar\nbaz'`包含嵌入的换行符，当锚定在字符串的开头时，只有`'foo'`匹配，当锚定在字符串的结尾时，只有`'baz'`匹配。

然而，如果一个字符串嵌入了换行符，你可以认为它是由多个内部行组成的。在这种情况下，如果设置了`MULTILINE`标志，`^`和`$`锚元字符也匹配内部行:

*   **`^`** 匹配字符串的开头或字符串中任何一行的开头(即紧跟在换行符之后)。
*   **`$`** 匹配字符串末尾或字符串中任何一行的末尾(紧接在换行符之前)。

以下是如上所示的相同搜索:

>>>

```py
>>> s = 'foo\nbar\nbaz'
>>> print(s)
foo
bar
baz

>>> re.search('^foo', s, re.MULTILINE)
<_sre.SRE_Match object; span=(0, 3), match='foo'>
>>> re.search('^bar', s, re.MULTILINE)
<_sre.SRE_Match object; span=(4, 7), match='bar'>
>>> re.search('^baz', s, re.MULTILINE)
<_sre.SRE_Match object; span=(8, 11), match='baz'>

>>> re.search('foo$', s, re.M)
<_sre.SRE_Match object; span=(0, 3), match='foo'>
>>> re.search('bar$', s, re.M)
<_sre.SRE_Match object; span=(4, 7), match='bar'>
>>> re.search('baz$', s, re.M)
<_sre.SRE_Match object; span=(8, 11), match='baz'>
```

在字符串`'foo\nbar\nbaz'`中，`'foo'`、`'bar'`和`'baz'`这三个都出现在字符串的开头或结尾，或者出现在字符串中一行的开头或结尾。随着`MULTILINE`标志的设置，当锚定`^`或`$`时，所有三个匹配。

**注意:**`MULTILINE`标志只以这种方式修改`^`和`$`锚。对`\A`和`\Z`主播没有任何影响:

>>>

```py
 1>>> s = 'foo\nbar\nbaz'
 2
 3>>> re.search('^bar', s, re.MULTILINE)
 4<_sre.SRE_Match object; span=(4, 7), match='bar'>
 5>>> re.search('bar$', s, re.MULTILINE)
 6<_sre.SRE_Match object; span=(4, 7), match='bar'>
 7
 8>>> print(re.search('\Abar', s, re.MULTILINE))
 9None
10>>> print(re.search('bar\Z', s, re.MULTILINE))
11None
```

在**第 3 行和第 5 行**上，`^`和`$`锚点指示必须在一行的开始和结束处找到`'bar'`。指定`MULTILINE`标志使这些匹配成功。

第 8 行和第 10 行上的例子使用了`\A`和`\Z`标志。您可以看到，即使`MULTILINE`标志有效，这些匹配也会失败。

`re.S`
T1】

> 使点(`.`)元字符匹配换行符。

请记住，默认情况下，点元字符匹配除换行符以外的任何字符。`DOTALL`旗解除了这一限制:

>>>

```py
 1>>> print(re.search('foo.bar', 'foo\nbar'))
 2None
 3>>> re.search('foo.bar', 'foo\nbar', re.DOTALL)
 4<_sre.SRE_Match object; span=(0, 7), match='foo\nbar'>
 5>>> re.search('foo.bar', 'foo\nbar', re.S)
 6<_sre.SRE_Match object; span=(0, 7), match='foo\nbar'>
```

在这个例子中，在**行 1** 上，点元字符与`'foo\nbar'`中的换行符不匹配。在**第 3 行和第 5 行**，`DOTALL`是有效的，所以点确实匹配换行符。注意，`DOTALL`国旗的简称是`re.S`，而不是你所想的`re.D`。

`re.X`
T1】

> 允许在正则表达式中包含空格和注释。

`VERBOSE`标志指定了一些特殊的行为:

*   regex 解析器忽略所有空格，除非它在字符类中或者用反斜杠转义。

*   如果正则表达式包含一个字符类中没有的或者用反斜杠转义的`#`字符，那么解析器会忽略它和它右边的所有字符。

这有什么用？它允许你用 Python 格式化一个正则表达式，使它更具可读性和自我文档化。

这里有一个例子，展示了如何使用它。假设您想要解析具有以下格式的电话号码:

*   可选的三位数区号，在括号中
*   可选空白
*   三位数前缀
*   分离器`'-'`或`'.'`
*   四位数行号

下面的正则表达式完成了这个任务:

>>>

```py
>>> regex = r'^(\(\d{3}\))?\s*\d{3}[-.]\d{4}$'

>>> re.search(regex, '414.9229')
<_sre.SRE_Match object; span=(0, 8), match='414.9229'>
>>> re.search(regex, '414-9229')
<_sre.SRE_Match object; span=(0, 8), match='414-9229'>
>>> re.search(regex, '(712)414-9229')
<_sre.SRE_Match object; span=(0, 13), match='(712)414-9229'>
>>> re.search(regex, '(712) 414-9229')
<_sre.SRE_Match object; span=(0, 14), match='(712) 414-9229'>
```

但是`r'^(\(\d{3}\))?\s*\d{3}[-.]\d{4}$'`是满眼不是吗？使用`VERBOSE`标志，您可以用 Python 编写相同的正则表达式，如下所示:

>>>

```py
>>> regex = r'''^               # Start of string
...             (\(\d{3}\))?    # Optional area code
...             \s*             # Optional whitespace
...             \d{3} # Three-digit prefix
...             [-.]            # Separator character
...             \d{4} # Four-digit line number
...             $               # Anchor at end of string
...             '''

>>> re.search(regex, '414.9229', re.VERBOSE)
<_sre.SRE_Match object; span=(0, 8), match='414.9229'>
>>> re.search(regex, '414-9229', re.VERBOSE)
<_sre.SRE_Match object; span=(0, 8), match='414-9229'>
>>> re.search(regex, '(712)414-9229', re.X)
<_sre.SRE_Match object; span=(0, 13), match='(712)414-9229'>
>>> re.search(regex, '(712) 414-9229', re.X)
<_sre.SRE_Match object; span=(0, 14), match='(712) 414-9229'>
```

`re.search()`调用与上面显示的相同，所以您可以看到这个正则表达式的工作方式与前面指定的相同。但是一看就没那么难懂了。

注意，[三重引号](https://realpython.com/lessons/triple-quoted-strings/)使得包含嵌入的换行符变得特别方便，这些换行符在`VERBOSE`模式中被视为被忽略的空白。

当使用`VERBOSE`标志时，要注意那些你希望有意义的空白。考虑这些例子:

>>>

```py
 1>>> re.search('foo bar', 'foo bar')
 2<_sre.SRE_Match object; span=(0, 7), match='foo bar'>
 3
 4>>> print(re.search('foo bar', 'foo bar', re.VERBOSE)) 5None
 6
 7>>> re.search('foo\ bar', 'foo bar', re.VERBOSE) 8<_sre.SRE_Match object; span=(0, 7), match='foo bar'>
 9>>> re.search('foo[ ]bar', 'foo bar', re.VERBOSE) 10<_sre.SRE_Match object; span=(0, 7), match='foo bar'>
```

至此您已经看到了一切，您可能想知道为什么第 4 行的**正则表达式`foo bar`与字符串`'foo bar'`不匹配。这是因为`VERBOSE`标志会导致解析器忽略空格字符。**

为了按照预期进行匹配，用反斜杠对空格字符进行转义，或者将其包含在一个字符类中，如第**行第 7 行和第 9** 行所示。

与`DOTALL`标志一样，注意`VERBOSE`标志有一个不直观的简称:`re.X`，而不是`re.V`。

`re.DEBUG`

> 显示调试信息。

`DEBUG`标志使 Python 中的 regex 解析器向控制台显示关于解析过程的调试信息:

>>>

```py
>>> re.search('foo.bar', 'fooxbar', re.DEBUG)
LITERAL 102
LITERAL 111
LITERAL 111
ANY None
LITERAL 98
LITERAL 97
LITERAL 114
<_sre.SRE_Match object; span=(0, 7), match='fooxbar'>
```

当解析器在调试输出中显示`LITERAL nnn`时，它显示的是正则表达式中文字字符的 ASCII 代码。在这种情况下，字面字符是`'f'`、`'o'`、`'o'`和`'b'`、`'a'`、`'r'`。

这里有一个更复杂的例子。这是前面关于`VERBOSE`标志的讨论中显示的电话号码正则表达式:

>>>

```py
>>> regex = r'^(\(\d{3}\))?\s*\d{3}[-.]\d{4}$'

>>> re.search(regex, '414.9229', re.DEBUG)
AT AT_BEGINNING
MAX_REPEAT 0 1
 SUBPATTERN 1 0 0
 LITERAL 40
 MAX_REPEAT 3 3
 IN
 CATEGORY CATEGORY_DIGIT
 LITERAL 41
MAX_REPEAT 0 MAXREPEAT
 IN
 CATEGORY CATEGORY_SPACE
MAX_REPEAT 3 3
 IN
 CATEGORY CATEGORY_DIGIT
IN
 LITERAL 45
 LITERAL 46
MAX_REPEAT 4 4
 IN
 CATEGORY CATEGORY_DIGIT
AT AT_END
<_sre.SRE_Match object; span=(0, 8), match='414.9229'>
```

这看起来像是很多你永远不需要的深奥信息，但它可能是有用的。参见下面的实际应用。

> #### 深潜:调试正则表达式解析
> 
> 正如您从上面所知道的，元字符序列`{m,n}`表示特定的重复次数。它匹配从`m`到`n`之前的任何重复:
> 
> >>>
> 
> ```py
> `>>> re.search('x[123]{2,4}y', 'x222y')
> <_sre.SRE_Match object; span=(0, 5), match='x222y'>` 
> ```
> 
> 您可以用`DEBUG`标志来验证这一点:
> 
> >>>
> 
> ```py
> `>>> re.search('x[123]{2,4}y', 'x222y', re.DEBUG)
> LITERAL 120
> MAX_REPEAT 2 4
>  IN
>  LITERAL 49
>  LITERAL 50
>  LITERAL 51
> LITERAL 121
> <_sre.SRE_Match object; span=(0, 5), match='x222y'>` 
> ```
> 
> `MAX_REPEAT 2 4`确认正则表达式解析器识别元字符序列`{2,4}`并将其解释为范围限定符。
> 
> 但是，如前所述，如果 Python 中正则表达式中的一对花括号包含除有效数字或数值范围之外的任何内容，那么它就失去了特殊的意义。
> 
> 您也可以验证这一点:
> 
> >>>
> 
> ```py
> `>>> re.search('x[123]{foo}y', 'x222y', re.DEBUG)
> LITERAL 120
> IN
>  LITERAL 49
>  LITERAL 50
>  LITERAL 51
> LITERAL 123 LITERAL 102 LITERAL 111 LITERAL 111 LITERAL 125 LITERAL 121` 
> ```
> 
> 您可以看到在调试输出中没有`MAX_REPEAT`标记。`LITERAL`标记表明解析器按字面意思处理`{foo}`,而不是作为量词元字符序列。`123`、`102`、`111`、`111`和`125`是字符串`'{foo}'`中字符的 ASCII 码。
> 
> 通过向您展示解析器如何解释您的正则表达式，`DEBUG`标志显示的信息可以帮助您排除故障。

奇怪的是，`re`模块没有定义单字母版本的`DEBUG`标志。如果您愿意，您可以定义自己的:

>>>

```py
>>> import re
>>> re.D
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 're' has no attribute 'D'

>>> re.D = re.DEBUG >>> re.search('foo', 'foo', re.D)
LITERAL 102
LITERAL 111
LITERAL 111
<_sre.SRE_Match object; span=(0, 3), match='foo'>
```

但是这可能更令人困惑，而不是更有帮助，因为您的代码的读者可能会将其误解为`DOTALL`标志的缩写。如果你真的做了这个作业，最好把它完整地记录下来。

`re.A`
`re.ASCII`
`re.U`
`re.UNICODE`
`re.L`

> 指定用于分析特殊正则表达式字符类的字符编码。

几个 regex 元字符序列(`\w`、`\W`、`\b`、`\B`、`\d`、`\D`、`\s`和`\S`)要求您将字符分配给某些类别，如单词、数字或空格。该组中的标志决定了用于将字符分配给这些类的编码方案。可能的编码是 ASCII、Unicode 或根据当前的区域设置。

在关于 Python 中的字符串和字符数据的教程中，在对 [`ord()`内置函数](https://realpython.com/python-strings/#built-in-string-functions)的讨论中，您已经简要介绍了字符编码和 Unicode。有关更深入的信息，请查看以下资源:

*   [Python 中的 Unicode &字符编码:无痛指南](https://realpython.com/python-encodings-guide)
*   [Python 的 Unicode 支持](https://docs.python.org/3/howto/unicode.html#python-s-unicode-support)

为什么字符编码在 Python 的正则表达式环境中如此重要？这里有一个简单的例子。

您之前已经了解到`\d`指定了一个数字字符。对`\d`元字符序列的描述表明它等同于字符类`[0-9]`。对于英语和西欧语言来说，情况确实如此，但对于世界上大多数语言来说，字符`'0'`到`'9'`并不代表所有甚至*任何一个*数字。

例如，这里有一个由三个[梵文数字字符](https://en.wikipedia.org/wiki/Devanagari#Numerals)组成的字符串:

>>>

```py
>>> s = '\u0967\u096a\u096c'
>>> s
'१४६'
```

对于 regex 解析器来说，要正确处理 Devanagari 脚本，数字元字符序列`\d`也必须匹配这些字符中的每一个。

Unicode 协会创建了 Unicode 来处理这个问题。Unicode 是一种字符编码标准，旨在代表世界上所有的书写系统。Python 3 中的所有字符串，包括正则表达式，默认都是 Unicode 的。

那么，回到上面列出的标志。这些标志通过指定使用的编码是 ASCII、Unicode 还是当前区域设置来帮助确定字符是否属于给定的类别:

*   **`re.U`** 和 **`re.UNICODE`** 指定 Unicode 编码。Unicode 是默认值，所以这些标志是多余的。支持它们主要是为了向后兼容。
*   **`re.A`** 和 **`re.ASCII`** 基于 ASCII 编码强制判定。如果你碰巧用英语操作，那么无论如何都会发生这种情况，所以这个标志不会影响是否找到匹配。
*   **`re.L`****`re.LOCALE`**根据当前地区确定。区域设置是一个过时的概念，不被认为是可靠的。除非在极少数情况下，你不太可能需要它。

使用默认的 Unicode 编码，regex 解析器应该能够处理任何语言。在下面的示例中，它正确地将字符串`'१४६'`中的每个字符识别为数字:

>>>

```py
>>> s = '\u0967\u096a\u096c'
>>> s
'१४६'
>>> re.search('\d+', s)
<_sre.SRE_Match object; span=(0, 3), match='१४६'>
```

下面是另一个例子，说明了字符编码如何影响 Python 中的正则表达式匹配。考虑这个字符串:

>>>

```py
>>> s = 'sch\u00f6n'
>>> s
'schön'
```

`'schön'`(德语中表示*漂亮的*或*漂亮的*)包含了`'ö'`字符，该字符具有 16 位十六进制 Unicode 值`00f6`。这个字符不能用传统的 7 位 ASCII 码表示。

如果您使用德语，那么您应该合理地期望正则表达式解析器将`'schön'`中的所有字符都视为单词字符。但是看看如果您使用`\w`字符类在`s`中搜索单词字符并强制使用 ASCII 编码会发生什么:

>>>

```py
>>> re.search('\w+', s, re.ASCII)
<_sre.SRE_Match object; span=(0, 3), match='sch'>
```

当您将编码限制为 ASCII 时，regex 解析器只将前三个字符识别为单词字符。比赛在`'ö'`停止。

另一方面，如果您指定了`re.UNICODE`或者允许编码默认为 Unicode，那么`'schön'`中的所有字符都符合单词字符的条件:

>>>

```py
>>> re.search('\w+', s, re.UNICODE)
<_sre.SRE_Match object; span=(0, 5), match='schön'>
>>> re.search('\w+', s)
<_sre.SRE_Match object; span=(0, 5), match='schön'>
```

`ASCII`和`LOCALE`标志可以在特殊情况下使用。但是一般来说，最好的策略是使用默认的 Unicode 编码。这应该可以正确处理任何世界语言。

### 在函数调用中组合`<flags>`个参数

定义了标志值，以便您可以使用[按位 OR](https://realpython.com/python-operators-expressions/#bitwise-operators) ( `|`)运算符将它们组合起来。这允许您在单个函数调用中指定几个标志:

>>>

```py
>>> re.search('^bar', 'FOO\nBAR\nBAZ', re.I|re.M)
<_sre.SRE_Match object; span=(4, 7), match='BAR'>
```

这个`re.search()`调用使用按位 OR 同时指定`IGNORECASE`和`MULTILINE`标志。

### 设置和清除正则表达式中的标志

除了能够向大多数`re`模块函数调用传递一个`<flags>`参数之外，您还可以在 Python 中修改 regex 中的标志值。有两个正则表达式元字符序列提供了这种能力。

`(?<flags>)`

> 为正则表达式的持续时间设置标志值。

在正则表达式中，元字符序列`(?<flags>)`为整个表达式设置指定的标志。

`<flags>`的值是集合`a`、`i`、`L`、`m`、`s`、`u`和`x`中的一个或多个字母。以下是它们与`re`模块标志的对应关系:

| 信 | 旗帜 |
| --- | --- |
| `a` | `re.A` `re.ASCII` |
| `i` | `re.I` `re.IGNORECASE` |
| `L` | `re.L` `re.LOCALE` |
| `m` | `re.M` `re.MULTILINE` |
| `s` | `re.S` `re.DOTALL` |
| `u` | `re.U` `re.UNICODE` |
| `x` | `re.X` `re.VERBOSE` |

`(?<flags>)`元字符序列作为一个整体匹配空字符串。它总是匹配成功，不消耗任何搜索字符串。

以下示例是设置`IGNORECASE`和`MULTILINE`标志的等效方式:

>>>

```py
>>> re.search('^bar', 'FOO\nBAR\nBAZ\n', re.I|re.M)
<_sre.SRE_Match object; span=(4, 7), match='BAR'>

>>> re.search('(?im)^bar', 'FOO\nBAR\nBAZ\n')
<_sre.SRE_Match object; span=(4, 7), match='BAR'>
```

请注意，`(?<flags>)`元字符序列为整个正则表达式设置了给定的标志，不管它放在表达式中的什么位置:

>>>

```py
>>> re.search('foo.bar(?s).baz', 'foo\nbar\nbaz')
<_sre.SRE_Match object; span=(0, 11), match='foo\nbar\nbaz'>

>>> re.search('foo.bar.baz(?s)', 'foo\nbar\nbaz')
<_sre.SRE_Match object; span=(0, 11), match='foo\nbar\nbaz'>
```

在上面的例子中，两个点元字符都匹配换行符，因为`DOTALL`标志是有效的。即使`(?s)`出现在表达式的中间或末尾，也是如此。

从 Python 3.7 开始，不赞成在正则表达式中除开头以外的任何地方指定`(?<flags>)`:

>>>

```py
>>> import sys
>>> sys.version
'3.8.0 (default, Oct 14 2019, 21:29:03) \n[GCC 7.4.0]'

>>> re.search('foo.bar.baz(?s)', 'foo\nbar\nbaz')
<stdin>:1: DeprecationWarning: Flags not at the start
 of the expression 'foo.bar.baz(?s)'
<re.Match object; span=(0, 11), match='foo\nbar\nbaz'>
```

它仍然产生适当的匹配，但是您将得到一个警告消息。

`(?<set_flags>-<remove_flags>:<regex>)`

> 设置或删除组持续时间的标志值。

`(?<set_flags>-<remove_flags>:<regex>)`定义与`<regex>`匹配的非捕获组。对于组中包含的`<regex>`，正则表达式解析器设置`<set_flags>`中指定的任何标志，并清除`<remove_flags>`中指定的任何标志。

`<set_flags>`和`<remove_flags>`的值通常是`i`、`m`、`s`或`x`。

在以下示例中，为指定的组设置了`IGNORECASE`标志:

>>>

```py
>>> re.search('(?i:foo)bar', 'FOObar')
<re.Match object; span=(0, 6), match='FOObar'>
```

这产生了一个匹配，因为`(?i:foo)`规定与`'FOO'`的匹配是不区分大小写的。

现在对比一下这个例子:

>>>

```py
>>> print(re.search('(?i:foo)bar', 'FOOBAR'))
None
```

和前面的例子一样，与`'FOO'`的匹配会成功，因为它不区分大小写。但是一旦在组之外，`IGNORECASE`就不再有效，所以与`'BAR'`的匹配是区分大小写的，并且会失败。

以下示例演示了如何关闭组的标志:

>>>

```py
>>> print(re.search('(?-i:foo)bar', 'FOOBAR', re.IGNORECASE))
None
```

同样，没有匹配。尽管`re.IGNORECASE`为整个调用启用了不区分大小写的匹配，但是元字符序列`(?-i:foo)`在该组的持续时间内关闭了`IGNORECASE`，因此与`'FOO'`的匹配失败。

从 Python 3.7 开始，您可以将`u`、`a`或`L`指定为`<set_flags>`，以覆盖指定组的默认编码:

>>>

```py
>>> s = 'sch\u00f6n'
>>> s
'schön'

>>> # Requires Python 3.7 or later
>>> re.search('(?a:\w+)', s)
<re.Match object; span=(0, 3), match='sch'>
>>> re.search('(?u:\w+)', s)
<re.Match object; span=(0, 5), match='schön'>
```

但是，您只能以这种方式设置编码。您不能删除它:

>>>

```py
>>> re.search('(?-a:\w+)', s)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3.8/re.py", line 199, in search
    return _compile(pattern, flags).search(string)
  File "/usr/lib/python3.8/re.py", line 302, in _compile
    p = sre_compile.compile(pattern, flags)
  File "/usr/lib/python3.8/sre_compile.py", line 764, in compile
    p = sre_parse.parse(p, flags)
  File "/usr/lib/python3.8/sre_parse.py", line 948, in parse
    p = _parse_sub(source, state, flags & SRE_FLAG_VERBOSE, 0)
  File "/usr/lib/python3.8/sre_parse.py", line 443, in _parse_sub
    itemsappend(_parse(source, state, verbose, nested + 1,
  File "/usr/lib/python3.8/sre_parse.py", line 805, in _parse
    flags = _parse_flags(source, state, char)
  File "/usr/lib/python3.8/sre_parse.py", line 904, in _parse_flags
    raise source.error(msg)
re.error: bad inline flags: cannot turn off flags 'a', 'u' and 'L' at
position 4
```

`u`、`a`和`L`是互斥的。每组只能出现一个。

## 结论

这就结束了你对正则表达式匹配和 Python 的`re`模块的介绍。恭喜你！你已经掌握了大量的材料。

**你现在知道如何:**

*   使用 **`re.search()`** 在 Python 中执行正则表达式匹配
*   使用正则表达式**元字符**创建复杂的模式匹配搜索
*   用**标志**调整正则表达式解析行为

但是你仍然只看到了模块中的一个函数:`re.search()`！`re`模块有更多有用的函数和对象可以添加到您的模式匹配工具包中。本系列的下一篇教程将向您介绍 Python 中的 regex 模块还能提供什么。

[« Functions in Python](https://realpython.com/defining-your-own-python-function/)[Regular Expressions: Regexes in Python (Part 1)](#)[Regular Expressions: Regexes in Python (Part 2) »](https://realpython.com/regex-python-part-2/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**正则表达式和用 Python 构建正则表达式**](/courses/building-regexes-python/)**************