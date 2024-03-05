# 正则表达式:Python 中的正则表达式(第 2 部分)

> 原文：<https://realpython.com/regex-python-part-2/>

在本系列的[之前的教程](https://realpython.com/regex-python)中，你已经涉及了很多内容。您了解了如何使用`re.search()`通过 Python 中的正则表达式执行模式匹配，并了解了许多正则表达式元字符和解析标志，您可以使用它们来微调您的模式匹配功能。

尽管如此，模块还能提供更多。

**在本教程中，您将:**

*   探索`re`模块提供的`re.search()`之外的更多功能
*   了解何时以及如何将 Python 中的正则表达式预编译成一个**正则表达式对象**
*   发现您可以用由`re`模块中的函数返回的**匹配对象**做的有用的事情

准备好了吗？让我们开始吧！

**免费奖励:** [从 Python 基础:Python 3 实用入门](https://realpython.com/bonus/python-basics-sample-free-chapter/)中获取一个示例章节，看看如何通过完整的课程(最新的 Python 3.9)从 Python 初学者过渡到中级。

## `re`模块功能

除了`re.search()`之外，`re`模块还包含其他几个函数来帮助您执行与 regex 相关的任务。

**注意:**您在之前的教程中已经看到了`re.search()`可以带一个可选的`<flags>`参数，它指定了修改解析行为的[标志](https://realpython.com/regex-python/#modifying-regular-expression-matching-with-flags)。除了`re.escape()`之外，下面显示的所有函数都以同样的方式支持`<flags>`参数。

您可以将`<flags>`指定为位置参数或关键字参数:

```py
re.search(<regex>, <string>, <flags>)
re.search(<regex>, <string>, flags=<flags>)
```

`<flags>`的缺省值总是`0`，这表示匹配行为没有特殊修改。记得在之前的教程中关于标志的[讨论中，`re.UNICODE`标志总是默认设置的。](https://realpython.com/regex-python/#supported-regular-expression-flags)

Python `re`模块中可用的正则表达式函数分为以下三类:

1.  搜索功能
2.  替代函数
3.  效用函数

以下部分将更详细地解释这些功能。

[*Remove ads*](/account/join/)

### 搜索功能

搜索函数在搜索字符串中扫描指定正则表达式的一个或多个匹配项:

| 功能 | 描述 |
| --- | --- |
| `re.search()` | 扫描字符串以查找正则表达式匹配 |
| `re.match()` | 在字符串的开头查找正则表达式匹配 |
| `re.fullmatch()` | 在整个字符串中查找正则表达式匹配 |
| `re.findall()` | 返回一个字符串中所有正则表达式匹配的列表 |
| `re.finditer()` | 返回一个迭代器，它从字符串中产生正则表达式匹配 |

从表中可以看出，这些功能彼此相似。但是每一个都以自己的方式调整搜索功能。

`re.search(<regex>, <string>, flags=0)`

> 扫描字符串中的正则表达式匹配。

如果你已经完成了本系列的[前一篇教程](https://realpython.com/regex-python/)，那么现在你应该已经很熟悉这个函数了。`re.search(<regex>, <string>)`在`<string>`中寻找`<regex>`匹配的任何位置:

>>>

```py
>>> re.search(r'(\d+)', 'foo123bar')
<_sre.SRE_Match object; span=(3, 6), match='123'>
>>> re.search(r'[a-z]+', '123FOO456', flags=re.IGNORECASE)
<_sre.SRE_Match object; span=(3, 6), match='FOO'>

>>> print(re.search(r'\d+', 'foo.bar'))
None
```

如果找到匹配，函数返回一个匹配对象，否则返回 [`None`](https://realpython.com/null-in-python/) 。

`re.match(<regex>, <string>, flags=0)`

> 在字符串的开头查找正则表达式匹配。

这与`re.search()`相同，除了如果`<regex>`在`<string>`中的任何地方匹配*则`re.search()`返回一个匹配，而`re.match()`只有在`<regex>`在`<string>`的*开始*处匹配时才返回一个匹配:*

>>>

```py
>>> re.search(r'\d+', '123foobar')
<_sre.SRE_Match object; span=(0, 3), match='123'>
>>> re.search(r'\d+', 'foo123bar')
<_sre.SRE_Match object; span=(3, 6), match='123'>

>>> re.match(r'\d+', '123foobar')
<_sre.SRE_Match object; span=(0, 3), match='123'>
>>> print(re.match(r'\d+', 'foo123bar'))
None
```

在上面的例子中，当数字在字符串的开头和中间时，`re.search()`匹配，但是只有当数字在开头时，`re.match()`才匹配。

请记住，在本系列的上一篇教程中，如果`<string>`包含嵌入的换行符，那么 [`MULTILINE`标志](https://realpython.com/regex-python/#modifying-regular-expression-matching-with-flags)会导致`re.search()`匹配位于`<string>`开头或`<string>`中包含的任何一行开头的脱字符(`^`)定位元字符:

>>>

```py
 1>>> s = 'foo\nbar\nbaz'
 2
 3>>> re.search('^foo', s)
 4<_sre.SRE_Match object; span=(0, 3), match='foo'>
 5>>> re.search('^bar', s, re.MULTILINE) 6<_sre.SRE_Match object; span=(4, 7), match='bar'>
```

`MULTILINE`标志不会以这种方式影响`re.match()`:

>>>

```py
 1>>> s = 'foo\nbar\nbaz'
 2
 3>>> re.match('^foo', s)
 4<_sre.SRE_Match object; span=(0, 3), match='foo'>
 5>>> print(re.match('^bar', s, re.MULTILINE)) 6None
```

即使设置了`MULTILINE`标志，`re.match()`也只会在`<string>`的开头匹配脱字符(`^`)锚，而不是在`<string>`中包含的行的开头。

请注意，虽然它说明了这一点，但上面例子中第 3 行的**符号(`^`)是多余的。使用`re.match()`，匹配基本上总是锚定在字符串的开头。**

`re.fullmatch(<regex>, <string>, flags=0)`

> 在整个字符串中查找正则表达式匹配。

这类似于`re.search()`和`re.match()`，但是只有当`<regex>`完全匹配`<string>`时，`re.fullmatch()`才返回一个匹配:

>>>

```py
 1>>> print(re.fullmatch(r'\d+', '123foo'))
 2None
 3>>> print(re.fullmatch(r'\d+', 'foo123'))
 4None
 5>>> print(re.fullmatch(r'\d+', 'foo123bar'))
 6None
 7>>> re.fullmatch(r'\d+', '123') 8<_sre.SRE_Match object; span=(0, 3), match='123'> 9
10>>> re.search(r'^\d+$', '123')
11<_sre.SRE_Match object; span=(0, 3), match='123'>
```

在**第 7 行**的调用中，搜索字符串`'123'`从头到尾全部由数字组成。所以这是唯一一种`re.fullmatch()`返回匹配的情况。

第 10 行**上的`re.search()`调用，其中`\d+`正则表达式被显式定位在搜索字符串的开头和结尾，在功能上是等效的。**

`re.findall(<regex>, <string>, flags=0)`

> 返回字符串中正则表达式的所有匹配项的列表。

`re.findall(<regex>, <string>)`返回`<string>`中`<regex>`所有非重叠匹配的列表。它从左到右扫描搜索字符串，并按找到的顺序返回所有匹配项:

>>>

```py
>>> re.findall(r'\w+', '...foo,,,,bar:%$baz//|')
['foo', 'bar', 'baz']
```

如果`<regex>`包含一个捕获组，那么返回列表只包含该组的内容，而不是整个匹配:

>>>

```py
>>> re.findall(r'#(\w+)#', '#foo#.#bar#.#baz#')
['foo', 'bar', 'baz']
```

在这种情况下，指定的正则表达式是`#(\w+)#`。匹配的字符串是`'#foo#'`、`'#bar#'`和`'#baz#'`。但是散列字符(`#`)不会出现在返回列表中，因为它们在分组括号之外。

如果`<regex>`包含不止一个捕获组，那么`re.findall()`返回包含捕获组的元组列表。每个元组的长度等于指定的组数:

>>>

```py
 1>>> re.findall(r'(\w+),(\w+)', 'foo,bar,baz,qux,quux,corge') 2[('foo', 'bar'), ('baz', 'qux'), ('quux', 'corge')]
 3
 4>>> re.findall(r'(\w+),(\w+),(\w+)', 'foo,bar,baz,qux,quux,corge') 5[('foo', 'bar', 'baz'), ('qux', 'quux', 'corge')]
```

在上面的例子中，**行 1** 上的正则表达式包含两个捕获组，所以`re.findall()`返回一个包含三个二元组的列表，每个二元组包含两个捕获的匹配。**第 4 行**包含三个组，所以返回值是两个三元组的列表。

`re.finditer(<regex>, <string>, flags=0)`

> 返回产生正则表达式匹配的迭代器。

`re.finditer(<regex>, <string>)`扫描`<string>`寻找`<regex>`的非重叠匹配，并返回一个迭代器，从它找到的任何匹配对象中产生匹配对象。它从左到右扫描搜索字符串，并按照找到匹配项的顺序返回匹配项:

>>>

```py
>>> it = re.finditer(r'\w+', '...foo,,,,bar:%$baz//|')
>>> next(it)
<_sre.SRE_Match object; span=(3, 6), match='foo'> >>> next(it)
<_sre.SRE_Match object; span=(10, 13), match='bar'> >>> next(it)
<_sre.SRE_Match object; span=(16, 19), match='baz'> >>> next(it)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration

>>> for i in re.finditer(r'\w+', '...foo,,,,bar:%$baz//|'):
...     print(i)
...
<_sre.SRE_Match object; span=(3, 6), match='foo'>
<_sre.SRE_Match object; span=(10, 13), match='bar'>
<_sre.SRE_Match object; span=(16, 19), match='baz'>
```

`re.findall()`和`re.finditer()`非常相似，但它们在两个方面有所不同:

1.  `re.findall()`返回一个列表，而`re.finditer()`返回一个迭代器。

2.  列表中`re.findall()`返回的条目是实际的匹配字符串，而`re.finditer()`返回的迭代器产生的条目是匹配对象。

任何你可以用一个完成的任务，你也可以用另一个来完成。你选择哪一个将视情况而定。正如您将在本教程后面看到的，可以从 match 对象中获得许多有用的信息。如果你需要这些信息，那么`re.finditer()`可能是更好的选择。

[*Remove ads*](/account/join/)

### 替代功能

替换函数替换搜索字符串中与指定正则表达式匹配的部分:

| 功能 | 描述 |
| --- | --- |
| `re.sub()` | 扫描字符串中的正则表达式匹配项，用指定的替换字符串替换字符串中的匹配部分，并返回结果 |
| `re.subn()` | 行为就像`re.sub()`一样，但也返回关于替换次数的信息 |

`re.sub()`和`re.subn()`用指定的替换创建一个新的字符串并返回它。原始字符串保持不变。(记住[字符串在 Python 中是不可变的](https://realpython.com/python-strings/#modifying-strings)，所以这些函数不可能修改原始字符串。)

`re.sub(<regex>, <repl>, <string>, count=0, flags=0)`

> 返回对搜索字符串执行替换后得到的新字符串。

`re.sub(<regex>, <repl>, <string>)`在`<string>`中找到`<regex>`最左边不重叠的出现，按照`<repl>`的指示替换每个匹配，并返回结果。`<string>`保持不变。

`<repl>`可以是字符串，也可以是函数，如下所述。

#### 字符串替换

如果`<repl>`是一个字符串，那么`re.sub()`将它插入到`<string>`中，代替任何匹配`<regex>`的序列:

>>>

```py
 1>>> s = 'foo.123.bar.789.baz'
 2
 3>>> re.sub(r'\d+', '#', s)
 4'foo.#.bar.#.baz'
 5>>> re.sub('[a-z]+', '(*)', s)
 6'(*).123.(*).789.(*)'
```

在**第 3 行**上，字符串`'#'`替换了`s`中的数字序列。在第 5 行的**上，字符串`'(*)'`替换了小写字母序列。在这两种情况下，`re.sub()`都会像往常一样返回修改后的字符串。**

`re.sub()`将`<repl>`中带编号的反向引用(`\<n>`)替换为相应捕获组的文本:

>>>

```py
>>> re.sub(r'(\w+),bar,baz,(\w+)',
...        r'\2,bar,baz,\1',
...        'foo,bar,baz,qux')
'qux,bar,baz,foo'
```

这里，捕获的组 1 和 2 包含`'foo'`和`'qux'`。在替换串`'\2,bar,baz,\1'`中，`'foo'`替换`\1`，`'qux'`替换`\2`。

还可以使用元字符序列`\g<name>`在替换字符串中引用用`(?P<name><regex>)`创建的命名反向引用:

>>>

```py
>>> re.sub(r'foo,(?P<w1>\w+),(?P<w2>\w+),qux',
...        r'foo,\g<w2>,\g<w1>,qux',
...        'foo,bar,baz,qux')
'foo,baz,bar,qux'
```

事实上，您也可以通过在尖括号内指定组号来引用编号为的*反向引用:*

>>>

```py
>>> re.sub(r'foo,(\w+),(\w+),qux',
...        r'foo,\g<2>,\g<1>,qux',
...        'foo,bar,baz,qux')
'foo,baz,bar,qux'
```

如果一个带编号的反向引用后面紧跟着一个文字数字字符，您可能需要使用这种技术来避免歧义。例如，假设您有一个类似于`'foo 123 bar'`的字符串，并且想要在数字序列的末尾添加一个`'0'`。你可以试试这个:

>>>

```py
>>> re.sub(r'(\d+)', r'\10', 'foo 123 bar') Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3.6/re.py", line 191, in sub
    return _compile(pattern, flags).sub(repl, string, count)
  File "/usr/lib/python3.6/re.py", line 326, in _subx
    template = _compile_repl(template, pattern)
  File "/usr/lib/python3.6/re.py", line 317, in _compile_repl
    return sre_parse.parse_template(repl, pattern)
  File "/usr/lib/python3.6/sre_parse.py", line 943, in parse_template
    addgroup(int(this[1:]), len(this) - 1)
  File "/usr/lib/python3.6/sre_parse.py", line 887, in addgroup
    raise s.error("invalid group reference %d" % index, pos)
sre_constants.error: invalid group reference 10 at position 1
```

唉，Python 中的 regex 解析器将`\10`解释为对第十个捕获组的反向引用，这在本例中并不存在。相反，您可以使用`\g<1>`来指代该组:

>>>

```py
>>> re.sub(r'(\d+)', r'\g<1>0', 'foo 123 bar')
'foo 1230 bar'
```

反向引用`\g<0>`是指整个匹配的文本。即使在`<regex>`中没有分组括号，这也是有效的:

>>>

```py
>>> re.sub(r'\d+', '/\g<0>/', 'foo 123 bar')
'foo /123/ bar'
```

如果`<regex>`指定零长度匹配，那么`re.sub()`将把`<repl>`替换到字符串中的每个字符位置:

>>>

```py
>>> re.sub('x*', '-', 'foo')
'-f-o-o-'
```

在上面的例子中，正则表达式`x*`匹配任何零长度序列，所以`re.sub()`在字符串中的每个字符位置插入替换字符串——在第一个字符之前，在每对字符之间，在最后一个字符之后。

如果`re.sub()`没有找到任何匹配，那么它总是不变地返回`<string>`。

#### 函数替换

如果您将`<repl>`指定为一个函数，那么`re.sub()`会为每个找到的匹配调用那个函数。它将每个相应的匹配对象作为参数传递给函数，以提供关于匹配的信息。然后，函数返回值变成替换字符串:

>>>

```py
>>> def f(match_obj):
...     s = match_obj.group(0)  # The matching string
...
...     # s.isdigit() returns True if all characters in s are digits
...     if s.isdigit():
...         return str(int(s) * 10)
...     else:
...         return s.upper()
...
>>> re.sub(r'\w+', f, 'foo.10.bar.20.baz.30')
'FOO.100.BAR.200.BAZ.300'
```

在这个例子中，`f()`在每个匹配中都被调用。结果，`re.sub()`将`<string>`的每个字母数字部分全部转换为大写，并将每个数字部分乘以`10`。

#### 限制替换数量

如果您为可选的`count`参数指定了一个正整数，那么`re.sub()`最多执行那么多的替换:

>>>

```py
>>> re.sub(r'\w+', 'xxx', 'foo.bar.baz.qux')
'xxx.xxx.xxx.xxx'
>>> re.sub(r'\w+', 'xxx', 'foo.bar.baz.qux', count=2)
'xxx.xxx.baz.qux'
```

和大多数`re`模块函数一样，`re.sub()`也接受可选的`<flags>`参数。

`re.subn(<regex>, <repl>, <string>, count=0, flags=0)`

> 返回对搜索字符串执行替换后得到的新字符串，并返回替换次数。

`re.subn()`与`re.sub()`相同，除了`re.subn()`返回一个由修改后的字符串和替换次数组成的二元组:

>>>

```py
>>> re.subn(r'\w+', 'xxx', 'foo.bar.baz.qux')
('xxx.xxx.xxx.xxx', 4)
>>> re.subn(r'\w+', 'xxx', 'foo.bar.baz.qux', count=2)
('xxx.xxx.baz.qux', 2)

>>> def f(match_obj):
...     m = match_obj.group(0)
...     if m.isdigit():
...         return str(int(m) * 10)
...     else:
...         return m.upper()
...
>>> re.subn(r'\w+', f, 'foo.10.bar.20.baz.30')
('FOO.100.BAR.200.BAZ.300', 6)
```

在所有其他方面，`re.subn()`的行为就像`re.sub()`一样。

[*Remove ads*](/account/join/)

### 实用功能

Python `re`模块中还有两个 regex 函数需要介绍:

| 功能 | 描述 |
| --- | --- |
| `re.split()` | 使用正则表达式作为分隔符将字符串分割成[子字符串](https://realpython.com/python-string-contains-substring/) |
| `re.escape()` | 转义正则表达式中的字符 |

这些函数涉及正则表达式匹配，但不属于上述任何一个类别。

`re.split(<regex>, <string>, maxsplit=0, flags=0)`

> 将一个字符串拆分成子字符串。

`re.split(<regex>, <string>)`使用`<regex>`作为分隔符将`<string>`分割成子字符串，并将子字符串作为列表返回。

以下示例将指定字符串拆分为由逗号(`,`)、分号(`;`)或斜线(`/`)字符分隔的子字符串，并由任意数量的空格包围:

>>>

```py
>>> re.split('\s*[,;/]\s*', 'foo,bar  ;  baz / qux')
['foo', 'bar', 'baz', 'qux']
```

如果`<regex>`包含捕获组，那么返回列表也包含匹配的分隔符字符串:

>>>

```py
>>> re.split('(\s*[,;/]\s*)', 'foo,bar  ;  baz / qux')
['foo', ',', 'bar', '  ;  ', 'baz', ' / ', 'qux']
```

这一次，返回列表不仅包含子字符串`'foo'`、`'bar'`、`'baz'`和`'qux'`，还包含几个分隔符字符串:

*   `','`
*   `' ; '`
*   `' / '`

如果您想将`<string>`分割成带分隔符的标记，以某种方式处理这些标记，然后使用最初分隔它们的相同分隔符将字符串拼凑在一起，这将非常有用:

>>>

```py
>>> string = 'foo,bar  ;  baz / qux'
>>> regex = r'(\s*[,;/]\s*)'
>>> a = re.split(regex, string)

>>> # List of tokens and delimiters
>>> a
['foo', ',', 'bar', '  ;  ', 'baz', ' / ', 'qux']

>>> # Enclose each token in <>'s
>>> for i, s in enumerate(a):
...
...     # This will be True for the tokens but not the delimiters
...     if not re.fullmatch(regex, s):
...         a[i] = f'<{s}>'
...

>>> # Put the tokens back together using the same delimiters
>>> ''.join(a)
'<foo>,<bar>  ;  <baz> / <qux>'
```

如果需要使用组，但不希望在返回列表中包含分隔符，则可以使用非捕获组:

>>>

```py
>>> string = 'foo,bar  ;  baz / qux'
>>> regex = r'(?:\s*[,;/]\s*)'
>>> re.split(regex, string)
['foo', 'bar', 'baz', 'qux']
```

如果可选的`maxsplit`参数存在并且大于零，那么`re.split()`最多执行那么多分割。返回列表中的最后一个元素是所有拆分发生后的剩余部分`<string>`:

>>>

```py
>>> s = 'foo, bar, baz, qux, quux, corge'

>>> re.split(r',\s*', s)
['foo', 'bar', 'baz', 'qux', 'quux', 'corge']
>>> re.split(r',\s*', s, maxsplit=3)
['foo', 'bar', 'baz', 'qux, quux, corge']
```

显式指定`maxsplit=0`相当于完全省略它。如果`maxsplit`是负的，那么`re.split()`不变地返回`<string>`(以防你在寻找一种什么都不做的复杂方法)。

如果`<regex>`包含捕获组，因此返回列表包含分隔符，并且`<regex>`匹配`<string>`的开始，那么`re.split()`将一个空字符串作为返回列表的第一个元素。类似地，如果`<regex>`匹配`<string>`的结尾，则返回列表中的最后一项是空字符串:

>>>

```py
>>> re.split('(/)', '/foo/bar/baz/')
['', '/', 'foo', '/', 'bar', '/', 'baz', '/', '']
```

在这种情况下，`<regex>`分隔符是一个单斜杠(`/`)字符。在某种意义上，在第一个分隔符的左边和最后一个分隔符的右边有一个空字符串。因此，`re.split()`将空字符串作为返回列表的第一个和最后一个元素是有意义的。

`re.escape(<regex>)`

> 转义正则表达式中的字符。

`re.escape(<regex>)`返回`<regex>`的副本，每个非单词字符(除字母、数字或下划线以外的任何字符)前面都有一个反斜杠。

如果您正在调用一个`re`模块函数，并且您传入的`<regex>`有许多特殊字符，您希望解析器照字面意思而不是作为元字符，这是很有用的。它可以省去手动输入所有反斜杠字符的麻烦:

>>>

```py
 1>>> print(re.match('foo^bar(baz)|qux', 'foo^bar(baz)|qux'))
 2None
 3>>> re.match('foo\^bar\(baz\)\|qux', 'foo^bar(baz)|qux')
 4<_sre.SRE_Match object; span=(0, 16), match='foo^bar(baz)|qux'>
 5
 6>>> re.escape('foo^bar(baz)|qux') == 'foo\^bar\(baz\)\|qux'
 7True
 8>>> re.match(re.escape('foo^bar(baz)|qux'), 'foo^bar(baz)|qux')
 9<_sre.SRE_Match object; span=(0, 16), match='foo^bar(baz)|qux'>
```

在这个例子中，**行 1** 上没有匹配，因为正则表达式`'foo^bar(baz)|qux'`包含表现为元字符的特殊字符。在**的第 3 行**，它们被用反斜杠显式转义，所以出现了匹配。**第 6 行和第 8 行**展示了您可以使用`re.escape()`获得相同的效果。

[*Remove ads*](/account/join/)

## Python 中编译的正则表达式对象

`re`模块支持将 Python 中的一个正则表达式预编译成一个**正则表达式对象**，这个对象可以在以后重复使用。

`re.compile(<regex>, flags=0)`

> 将正则表达式编译成正则表达式对象。

`re.compile(<regex>)`编译`<regex>`，返回对应的正则表达式对象。如果包含一个`<flags>`值，那么相应的标志将应用于对该对象执行的任何搜索。

有两种方法可以使用编译后的正则表达式对象。您可以将它指定为`re`模块函数的第一个参数来代替`<regex>`:

```py
re_obj = re.compile(<regex>, <flags>)
result = re.search(re_obj, <string>)
```

您也可以直接从正则表达式对象调用方法:

```py
re_obj = re.compile(<regex>, <flags>)
result = re_obj.search(<string>)
```

上面的两个例子都与此等价:

```py
result = re.search(<regex>, <string>, <flags>)
```

下面是您之前看到的一个示例，使用编译后的正则表达式对象进行了重新转换:

>>>

```py
>>> re.search(r'(\d+)', 'foo123bar')
<_sre.SRE_Match object; span=(3, 6), match='123'>

>>> re_obj = re.compile(r'(\d+)')
>>> re.search(re_obj, 'foo123bar') <_sre.SRE_Match object; span=(3, 6), match='123'>
>>> re_obj.search('foo123bar') <_sre.SRE_Match object; span=(3, 6), match='123'>
```

下面是另一个，它也使用了`IGNORECASE`标志:

>>>

```py
 1>>> r1 = re.search('ba[rz]', 'FOOBARBAZ', flags=re.I) 2
 3>>> re_obj = re.compile('ba[rz]', flags=re.I)
 4>>> r2 = re.search(re_obj, 'FOOBARBAZ') 5>>> r3 = re_obj.search('FOOBARBAZ') 6
 7>>> r1
 8<_sre.SRE_Match object; span=(3, 6), match='BAR'>
 9>>> r2
10<_sre.SRE_Match object; span=(3, 6), match='BAR'>
11>>> r3
12<_sre.SRE_Match object; span=(3, 6), match='BAR'>
```

在这个例子中，**行 1** 上的语句将 regex `ba[rz]`直接指定给`re.search()`作为第一个参数。在**第 4 行**上，`re.search()`的第一个参数是编译后的正则表达式对象`re_obj`。在**线 5** 上，在`re_obj`上直接调用`search()`。所有三种情况都产生相同的匹配。

### 为什么要编译正则表达式呢？

预编译有什么好处？有几个可能的优势。

如果您经常在 Python 代码中使用特定的正则表达式，那么预编译允许您将正则表达式的定义与其用法分开。这增强了模块性。考虑这个例子:

>>>

```py
>>> s1, s2, s3, s4 = 'foo.bar', 'foo123bar', 'baz99', 'qux & grault'

>>> import re
>>> re.search('\d+', s1)
>>> re.search('\d+', s2)
<_sre.SRE_Match object; span=(3, 6), match='123'>
>>> re.search('\d+', s3)
<_sre.SRE_Match object; span=(3, 5), match='99'>
>>> re.search('\d+', s4)
```

在这里，正则表达式`\d+`出现了几次。如果在维护这段代码的过程中，您决定需要一个不同的正则表达式，那么您需要在每个位置对它进行更改。在这个小例子中，这并不坏，因为它们的用途彼此接近。但是在一个更大的应用程序中，它们可能非常分散，很难跟踪。

以下内容更加模块化，更易于维护:

>>>

```py
>>> s1, s2, s3, s4 = 'foo.bar', 'foo123bar', 'baz99', 'qux & grault'
>>> re_obj = re.compile('\d+')

>>> re_obj.search(s1)
>>> re_obj.search(s2)
<_sre.SRE_Match object; span=(3, 6), match='123'>
>>> re_obj.search(s3)
<_sre.SRE_Match object; span=(3, 5), match='99'>
>>> re_obj.search(s4)
```

同样，通过使用变量赋值，无需预编译也可以实现类似的模块化:

>>>

```py
>>> s1, s2, s3, s4 = 'foo.bar', 'foo123bar', 'baz99', 'qux & grault'
>>> regex = '\d+'

>>> re.search(regex, s1)
>>> re.search(regex, s2)
<_sre.SRE_Match object; span=(3, 6), match='123'>
>>> re.search(regex, s3)
<_sre.SRE_Match object; span=(3, 5), match='99'>
>>> re.search(regex, s4)
```

理论上，您可能期望预编译也会导致更快的执行时间。假设您在同一个正则表达式上调用`re.search()`成千上万次。看起来，提前编译一次正则表达式要比在数千次使用中每次都重新编译更有效。

然而实际上，情况并非如此。事实是，`re`模块编译并且[缓存](https://en.wikipedia.org/wiki/Cache_(computing)#Software_caches)一个在函数调用中使用的正则表达式。如果随后在同一个 Python 代码中使用了同一个正则表达式，那么它不会被重新编译。而是从缓存中提取编译后的值。所以性能优势微乎其微。

总而言之，没有任何令人信服的理由用 Python 来编译正则表达式。像 Python 的大部分内容一样，它只是您工具箱中的一个工具，如果您觉得它可以提高代码的可读性或结构，您可以使用它。

[*Remove ads*](/account/join/)

### 正则表达式对象方法

编译后的正则表达式对象`re_obj`支持以下方法:

*   `re_obj.search(<string>[, <pos>[, <endpos>]])`
*   `re_obj.match(<string>[, <pos>[, <endpos>]])`
*   `re_obj.fullmatch(<string>[, <pos>[, <endpos>]])`
*   `re_obj.findall(<string>[, <pos>[, <endpos>]])`
*   `re_obj.finditer(<string>[, <pos>[, <endpos>]])`

除了它们也支持可选的`<pos>`和`<endpos>`参数之外，这些函数的行为方式都与您已经遇到的相应的`re`函数相同。如果这些都存在，那么搜索只适用于由`<pos>`和`<endpos>`指示的`<string>`部分，其作用与[切片标记](https://realpython.com/python-strings/#string-slicing)中的索引相同:

>>>

```py
 1>>> re_obj = re.compile(r'\d+')
 2>>> s = 'foo123barbaz'
 3
 4>>> re_obj.search(s)
 5<_sre.SRE_Match object; span=(3, 6), match='123'>
 6
 7>>> s[6:9]
 8'bar'
 9>>> print(re_obj.search(s, 6, 9))
10None
```

在上面的例子中，正则表达式是`\d+`，一个数字字符序列。第 4 行上的`.search()`调用搜索所有的`s`，所以有一个匹配。在**第 9 行**上，`<pos>`和`<endpos>`参数有效地将搜索限制在从字符 6 开始一直到但不包括字符 9 的子串(子串`'bar'`)，该子串不包含任何数字。

如果指定了`<pos>`但省略了`<endpos>`，那么搜索将应用于从`<pos>`到字符串末尾的子字符串。

注意，插入符号(`^`)和美元符号(`$`)等锚点仍然指整个字符串的开头和结尾，而不是由`<pos>`和`<endpos>`确定的子串:

>>>

```py
>>> re_obj = re.compile('^bar')
>>> s = 'foobarbaz'

>>> s[3:]
'barbaz'

>>> print(re_obj.search(s, 3))
None
```

这里，尽管`'bar'`出现在从字符 3 开始的子字符串的开头，但它不在整个字符串的开头，所以脱字符(`^`)定位符无法匹配。

以下方法也适用于编译后的正则表达式对象`re_obj`:

*   `re_obj.split(<string>, maxsplit=0)`
*   `re_obj.sub(<repl>, <string>, count=0)`
*   `re_obj.subn(<repl>, <string>, count=0)`

这些函数的行为类似于相应的`re`函数，但是它们不支持`<pos>`和`<endpos>`参数。

### 正则表达式对象属性

`re`模块为编译后的正则表达式对象定义了几个有用的属性:

| 属性 | 意义 |
| --- | --- |
| `re_obj.flags` | 任何对正则表达式有效的`<flags>` |
| `re_obj.groups` | 正则表达式中捕获组的数量 |
| `re_obj.groupindex` | 将由`(?P<name>)`结构定义的每个符号组名(如果有的话)映射到相应组号的字典 |
| `re_obj.pattern` | 产生该对象的`<regex>`模式 |

下面的代码演示了这些属性的一些用法:

>>>

```py
 1>>> re_obj = re.compile(r'(?m)(\w+),(\w+)', re.I)
 2>>> re_obj.flags 342
 4>>> re.I|re.M|re.UNICODE
 5<RegexFlag.UNICODE|MULTILINE|IGNORECASE: 42>
 6>>> re_obj.groups 72
 8>>> re_obj.pattern 9'(?m)(\\w+),(\\w+)'
10
11>>> re_obj = re.compile(r'(?P<w1>),(?P<w2>)')
12>>> re_obj.groupindex 13mappingproxy({'w1': 1, 'w2': 2})
14>>> re_obj.groupindex['w1']
151
16>>> re_obj.groupindex['w2']
172
```

注意，`.flags`包括任何指定为`re.compile()`参数的标志，任何在正则表达式中用`(?flags)`元字符序列指定的标志，以及任何默认有效的标志。在**行 1** 上定义的正则表达式对象中，定义了三个标志:

1.  **`re.I` :** 指定为`re.compile()`调用中的`<flags>`值
2.  **`re.M` :** 在正则表达式内指定为`(?m)`
3.  **`re.UNICODE` :** 默认启用

在**第 4 行**可以看到，`re_obj.flags`的值是这三个值的**逻辑或**，等于`42`。

在**第 11 行**定义的正则表达式对象的`.groupindex`属性的值在技术上是一个类型为`mappingproxy`的对象。实际上，它的功能就像一本[字典](https://realpython.com/courses/dictionaries-python/)。

[*Remove ads*](/account/join/)

## 匹配对象方法和属性

如您所见，当匹配成功时，`re`模块中的大多数函数和方法都返回一个**匹配对象**。因为匹配对象是真的 T4，所以你可以在条件中使用它:

>>>

```py
>>> m = re.search('bar', 'foo.bar.baz')
>>> m
<_sre.SRE_Match object; span=(4, 7), match='bar'>
>>> bool(m)
True

>>> if re.search('bar', 'foo.bar.baz'): ...     print('Found a match')
...
Found a match
```

但是 match 对象也包含相当多的关于匹配的有用信息。您已经看到了其中的一些——解释器在显示匹配对象时显示的`span=`和`match=`数据。使用 match 对象的方法和属性，您可以从 match 对象中获得更多信息。

### 匹配对象方法

下表总结了匹配对象`match`可用的方法:

| 方法 | 返回 |
| --- | --- |
| `match.group()` | 从`match`中指定的一个或多个捕获组 |
| `match.__getitem__()` | 来自`match`的一个捕获组 |
| `match.groups()` | 从`match`捕获的所有组 |
| `match.groupdict()` | 来自`match`的命名捕获组的字典 |
| `match.expand()` | 从`match`执行反向引用替换的结果 |
| `match.start()` | `match`的起始索引 |
| `match.end()` | `match`的结束索引 |
| `match.span()` | 作为元组的`match`的起始和结束索引 |

以下部分更详细地描述了这些方法。

`match.group([<group1>, ...])`

> 从匹配中返回指定的捕获组。

对于编号组，`match.group(n)`返回`n` <sup>`th`</sup> 组:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
>>> m.group(1)
'foo'
>>> m.group(3)
'baz'
```

**记住:**编号的捕获组是从 1 开始的，不是从 0 开始的。

如果使用`(?P<name><regex>)`捕获组，那么`match.group(<name>)`返回相应的命名组:

>>>

```py
>>> m = re.match(r'(?P<w1>\w+),(?P<w2>\w+),(?P<w3>\w+)', 'quux,corge,grault')
>>> m.group('w1')
'quux'
>>> m.group('w3')
'grault'
```

如果有多个参数，`.group()`将返回所有指定组的元组。给定的组可以出现多次，您可以按任何顺序指定任何捕获的组:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
>>> m.group(1, 3)
('foo', 'baz')
>>> m.group(3, 3, 1, 1, 2, 2)
('baz', 'baz', 'foo', 'foo', 'bar', 'bar')

>>> m = re.match(r'(?P<w1>\w+),(?P<w2>\w+),(?P<w3>\w+)', 'quux,corge,grault')
>>> m.group('w3', 'w1', 'w1', 'w2')
('grault', 'quux', 'quux', 'corge')
```

如果您指定了一个超出范围或不存在的组，那么`.group()`会引发一个`IndexError`异常:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
>>> m.group(4)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: no such group

>>> m = re.match(r'(?P<w1>\w+),(?P<w2>\w+),(?P<w3>\w+)', 'quux,corge,grault')
>>> m.group('foo')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: no such group
```

Python 中的正则表达式可能作为一个整体匹配，但包含一个不参与匹配的组。在这种情况下，`.group()`返回未参与组的`None`。考虑这个例子:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)?', 'foo,bar,')
>>> m
<_sre.SRE_Match object; span=(0, 8), match='foo,bar,'>
>>> m.group(1, 2)
('foo', 'bar')
```

从 match 对象可以看出，这个正则表达式是匹配的。前两个捕获的组分别包含`'foo'`和`'bar'`。

不过，第三组后面有一个问号(`?`)量词元字符，所以该组是可选的。如果在第二个逗号(`,`)后面有第三个单词字符序列，那么就会匹配，如果没有，那么也会匹配。

在这种情况下，没有。所以总的来说有匹配，但是第三组不参与其中。因此，`m.group(3)`仍然被定义并且是一个有效的引用，但是它返回`None`:

>>>

```py
>>> print(m.group(3))
None
```

也可能发生一个组多次参与整个比赛的情况。如果您为那个组号调用`.group()`,那么它只返回搜索字符串中最后一次匹配的部分。无法访问更早的匹配项:

>>>

```py
>>> m = re.match(r'(\w{3},)+', 'foo,bar,baz,qux')
>>> m
<_sre.SRE_Match object; span=(0, 12), match='foo,bar,baz,'>
>>> m.group(1)
'baz,'
```

在这个例子中，完全匹配是`'foo,bar,baz,'`，如所显示的匹配对象所示。`'foo,'`、`'bar,'`和`'baz,'`中的每一个都匹配组内的内容，但是`m.group(1)`只返回最后一个匹配项`'baz,'`。

如果您使用一个参数`0`或根本没有参数来调用`.group()`，那么它将返回整个匹配:

>>>

```py
 1>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
 2>>> m
 3<_sre.SRE_Match object; span=(0, 11), match='foo,bar,baz'> 4
 5>>> m.group(0)
 6'foo,bar,baz' 7>>> m.group()
 8'foo,bar,baz'
```

这是解释器在显示匹配对象时在`match=`之后显示的相同数据，正如你在上面的**行 3** 中看到的。

`match.__getitem__(<grp>)`

> 从匹配中返回捕获的组。

`match.__getitem__(<grp>)`与`match.group(<grp>)`相同，返回`<grp>`指定的单组:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
>>> m.group(2)
'bar'
>>> m.__getitem__(2)
'bar'
```

如果`.__getitem__()`只是简单地复制了`.group()`的功能，那么你为什么要使用它呢？你可能不会直接，但你可能会间接。请继续阅读，了解原因。

#### 魔法方法简介

`.__getitem__()`是 Python 中称为**魔法方法**的方法集合之一。当 Python 语句包含特定的对应语法元素时，解释器会调用这些特殊的方法。

**注:**魔法方法因为方法名的开头和结尾的分值下的**d**double**也被称为 **dunder 方法**。**

在本系列的后面，有几个关于面向对象编程的教程。在那里你会学到更多关于魔法的方法。

`.__getitem__()`对应的特定语法是用方括号索引。对于任何对象`obj`，无论何时使用表达式`obj[n]`，Python 都会在幕后悄悄地将其翻译成对`.__getitem__()`的调用。以下表达式实际上是等效的:

```py
obj[n]
obj.__getitem__(n)
```

语法`obj[n]`只有在`obj`所属的类或类型存在`.__getitem()__`方法时才有意义。Python 如何解释`obj[n]`将取决于该类的`.__getitem__()`的实现。

#### 返回匹配对象

从 Python 版本开始，`re`模块为匹配对象实现了`.__getitem__()`。实现是这样的，即`match.__getitem__(n)`与`match.group(n)`相同。

所有这些的结果是，您可以使用方括号索引语法从 match 对象访问捕获的组，而不是直接调用`.group()`:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
>>> m.group(2)
'bar'
>>> m.__getitem__(2)
'bar'
>>> m[2] 'bar'
```

这也适用于命名的捕获组:

>>>

```py
>>> m = re.match(
...     r'foo,(?P<w1>\w+),(?P<w2>\w+),qux',
...     'foo,bar,baz,qux')
>>> m.group('w2')
'baz'
>>> m['w2'] 'baz'
```

这可以通过显式调用`.group()`来实现，但这仍然是一个非常快捷的符号。

当一种编程语言提供了并非绝对必要的替代语法，但允许以更清晰、更易读的方式表达某种东西时，它被称为 [**句法糖**](https://en.wikipedia.org/wiki/Syntactic_sugar) 。对于匹配对象，`match[n]`是`match.group(n)`的语法糖。

**注意:**Python 中的许多对象都定义了一个`.__getitem()__`方法，允许使用方括号索引语法。但是，此功能仅适用于 Python 版或更高版本中的正则表达式匹配对象。

`match.groups(default=None)`

> 从匹配中返回所有捕获的组。

`match.groups()`返回所有捕获组的元组:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
>>> m.groups()
('foo', 'bar', 'baz')
```

正如您之前看到的，当 Python 中正则表达式中的一个组不参与整体匹配时，`.group()`会为该组返回`None`。默认情况下，`.groups()`也是如此。

在这种情况下，如果您希望`.groups()`返回其他内容，那么您可以使用`default`关键字参数:

>>>

```py
 1>>> m = re.search(r'(\w+),(\w+),(\w+)?', 'foo,bar,')
 2>>> m
 3<_sre.SRE_Match object; span=(0, 8), match='foo,bar,'>
 4>>> print(m.group(3))
 5None
 6
 7>>> m.groups()
 8('foo', 'bar', None) 9>>> m.groups(default='---')
10('foo', 'bar', '---')
```

这里，第三个`(\w+)`组不参与匹配，因为问号(`?`)元字符使其可选，并且字符串`'foo,bar,'`不包含第三个单词字符序列。默认情况下，`m.groups()`返回第三组的`None`，如**第 8 行**所示。在**第 10 行**上，您可以看到指定`default='---'`会导致它返回字符串`'---'`。

`.group()`没有对应的`default`关键字。对于不参与的组，它总是返回`None`。

`match.groupdict(default=None)`

> 返回命名捕获组的字典。

`match.groupdict()`返回用`(?P<name><regex>)`元字符序列捕获的所有命名组的字典。字典关键字是组名，字典值是相应的组值:

>>>

```py
>>> m = re.match(
...     r'foo,(?P<w1>\w+),(?P<w2>\w+),qux',
...     'foo,bar,baz,qux')
>>> m.groupdict()
{'w1': 'bar', 'w2': 'baz'}
>>> m.groupdict()['w2']
'baz'
```

与`.groups()`一样，对于`.groupdict()`，`default`参数决定了不参与组的返回值:

>>>

```py
>>> m = re.match(
...     r'foo,(?P<w1>\w+),(?P<w2>\w+)?,qux',
...     'foo,bar,,qux')
>>> m.groupdict()
{'w1': 'bar', 'w2': None}
>>> m.groupdict(default='---')
{'w1': 'bar', 'w2': '---'}
```

同样，由于问号(`?`)元字符，最后一组`(?P<w2>\w+)`不参与整体匹配。默认情况下，`m.groupdict()`为这个组返回`None`，但是您可以使用`default`参数来更改它。

`match.expand(<template>)`

> 从匹配中执行反向引用替换。

`match.expand(<template>)`返回通过对`<template>`执行反向引用替换得到的字符串，就像`re.sub()`会做的那样:

>>>

```py
 1>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
 2>>> m
 3<_sre.SRE_Match object; span=(0, 11), match='foo,bar,baz'>
 4>>> m.groups()
 5('foo', 'bar', 'baz')
 6
 7>>> m.expand(r'\2') 8'bar'
 9>>> m.expand(r'[\3] -> [\1]') 10'[baz] -> [foo]'
11
12>>> m = re.search(r'(?P<num>\d+)', 'foo123qux')
13>>> m
14<_sre.SRE_Match object; span=(3, 6), match='123'>
15>>> m.group(1)
16'123'
17
18>>> m.expand(r'--- \g<num> ---') 19'--- 123 ---'
```

这适用于数字反向引用，如上面的第 7**行和第 9** 行，也适用于命名反向引用，如第 18 行的**。**

`match.start([<grp>])`

`match.end([<grp>])`

> 返回匹配的开始和结束索引。

`match.start()`返回搜索字符串中匹配开始处的索引，而`match.end()`返回匹配结束后紧接着的索引:

>>>

```py
 1>>> s = 'foo123bar456baz'
 2>>> m = re.search('\d+', s)
 3>>> m
 4<_sre.SRE_Match object; span=(3, 6), match='123'> 5>>> m.start()
 63 7>>> m.end()
 86
```

当 Python 显示一个匹配对象时，这些是用关键字`span=`列出的值，如上面的**行 4** 所示。它们的行为类似于[字符串切片](https://realpython.com/python-strings/#string-slicing)值，所以如果您使用它们来切片原始搜索字符串，那么您应该得到匹配的子字符串:

>>>

```py
>>> m
<_sre.SRE_Match object; span=(3, 6), match='123'>
>>> s[m.start():m.end()]
'123'
```

`match.start(<grp>)`和`match.end(<grp>)`返回`<grp>`匹配的子串的起始和结束索引，可以是编号或命名的组:

>>>

```py
>>> s = 'foo123bar456baz'
>>> m = re.search(r'(\d+)\D*(?P<num>\d+)', s)

>>> m.group(1)
'123'
>>> m.start(1), m.end(1)
(3, 6)
>>> s[m.start(1):m.end(1)]
'123'

>>> m.group('num')
'456'
>>> m.start('num'), m.end('num')
(9, 12)
>>> s[m.start('num'):m.end('num')]
'456'
```

如果指定的组匹配一个空字符串，那么`.start()`和`.end()`相等:

>>>

```py
>>> m = re.search('foo(\d*)bar', 'foobar')
>>> m[1]
''
>>> m.start(1), m.end(1)
(3, 3)
```

如果你记得`.start()`和`.end()`的行为类似于切片索引，这是有意义的。任何开始和结束索引相等的字符串切片都将始终是空字符串。

当正则表达式包含不参与匹配的组时，会出现一种特殊情况:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)?', 'foo,bar,')
>>> print(m.group(3))
None
>>> m.start(3), m.end(3)
(-1, -1)
```

正如您之前看到的，在这种情况下，第三组不参与。`m.start(3)`和`m.end(3)`在这里没有实际意义，所以它们返回`-1`。

`match.span([<grp>])`

> 返回匹配的开始和结束索引。

`match.span()`以元组的形式返回匹配的起始和结束索引。如果您指定了`<grp>`，那么返回元组应用于给定的组:

>>>

```py
>>> s = 'foo123bar456baz'
>>> m = re.search(r'(\d+)\D*(?P<num>\d+)', s)
>>> m
<_sre.SRE_Match object; span=(3, 12), match='123bar456'>

>>> m[0]
'123bar456'
>>> m.span() (3, 12)

>>> m[1]
'123'
>>> m.span(1) (3, 6)

>>> m['num']
'456'
>>> m.span('num') (9, 12)
```

以下内容实际上是等效的:

*   `match.span(<grp>)`
*   `(match.start(<grp>), match.end(<grp>))`

`match.span()`只是提供了一种在一次方法调用中同时获得`match.start()`和`match.end()`的便捷方式。

[*Remove ads*](/account/join/)

### 匹配对象属性

像编译的正则表达式对象一样，match 对象也有几个有用的属性:

| 属性 | 意义 |
| --- | --- |
| `match.pos`
T1】 | 匹配的参数`<pos>`和`<endpos>`的有效值 |
| `match.lastindex` | 最后捕获的组的索引 |
| `match.lastgroup` | 最后捕获的组的名称 |
| `match.re` | 匹配项的已编译正则表达式对象 |
| `match.string` | 匹配的搜索字符串 |

以下部分提供了关于这些匹配对象属性的更多详细信息。

`match.pos`

`match.endpos`

> 包含用于搜索的`<pos>`和`<endpos>`的有效值。

请记住，一些方法在编译后的正则表达式上调用时，接受可选的`<pos>`和`<endpos>`参数，将搜索限制在指定搜索字符串的一部分。这些值可以从带有`.pos`和`.endpos`属性的匹配对象中访问:

>>>

```py
>>> re_obj = re.compile(r'\d+')
>>> m = re_obj.search('foo123bar', 2, 7)
>>> m
<_sre.SRE_Match object; span=(3, 6), match='123'>
>>> m.pos, m.endpos
(2, 7)
```

如果调用中没有包含`<pos>`和`<endpos>`参数，要么是因为它们被省略了，要么是因为正在讨论的函数不接受它们，那么`.pos`和`.endpos`属性实际上指示了字符串的开始和结束:

>>>

```py
 1>>> re_obj = re.compile(r'\d+')
 2>>> m = re_obj.search('foo123bar') 3>>> m
 4<_sre.SRE_Match object; span=(3, 6), match='123'>
 5>>> m.pos, m.endpos
 6(0, 9)
 7
 8>>> m = re.search(r'\d+', 'foo123bar') 9>>> m
10<_sre.SRE_Match object; span=(3, 6), match='123'>
11>>> m.pos, m.endpos
12(0, 9)
```

上面第 2 行的**调用可以接受`<pos>`和`<endpos>`参数，但是它们没有被指定。8 号线**上的`re.search()`呼叫根本带不走他们。无论哪种情况，`m.pos`和`m.endpos`都是`0`和`9`，搜索字符串`'foo123bar'`的起始和结束索引。****

`match.lastindex`

> 包含最后捕获的组的索引。

`match.lastindex`等于最后一个捕获组的整数索引:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
>>> m.lastindex
3
>>> m[m.lastindex]
'baz'
```

如果正则表达式包含潜在的未参与组，这允许您确定有多少组实际参与了匹配:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)?', 'foo,bar,baz') >>> m.groups()
('foo', 'bar', 'baz')
>>> m.lastindex, m[m.lastindex]
(3, 'baz')

>>> m = re.search(r'(\w+),(\w+),(\w+)?', 'foo,bar,') >>> m.groups()
('foo', 'bar', None)
>>> m.lastindex, m[m.lastindex]
(2, 'bar')
```

在第一个示例中，第三个组是可选的，因为问号(`?`)元字符，它确实参与了匹配。但是在第二个例子中没有。你能看出来，因为第一种情况下`m.lastindex`是`3`，第二种情况下`2`。

关于`.lastindex`，有一个微妙的问题需要注意。最后一个匹配的组并不总是语法上遇到的最后一个组。Python 文档给出了这个例子:

>>>

```py
>>> m = re.match('((a)(b))', 'ab')
>>> m.groups()
('ab', 'a', 'b')
>>> m.lastindex
1
>>> m[m.lastindex]
'ab'
```

最外面一组是`((a)(b))`，匹配`'ab'`。这是解析器遇到的第一个组，因此它成为组 1。但它也是最后一组匹配的，这就是为什么`m.lastindex`是`1`。

解析器识别的第二组和第三组是`(a)`和`(b)`。这是组`2`和`3`，但是他们在组`1`之前匹配。

`match.lastgroup`

> 包含最后捕获的组的名称。

如果最后捕获的组来自于`(?P<name><regex>)`元字符序列，那么`match.lastgroup`返回该组的名称:

>>>

```py
>>> s = 'foo123bar456baz'
>>> m = re.search(r'(?P<n1>\d+)\D*(?P<n2>\d+)', s)
>>> m.lastgroup
'n2'
```

如果最后捕获的组不是命名组，则`match.lastgroup`返回`None`:

>>>

```py
>>> s = 'foo123bar456baz'

>>> m = re.search(r'(\d+)\D*(\d+)', s)
>>> m.groups()
('123', '456')
>>> print(m.lastgroup)
None

>>> m = re.search(r'\d+\D*\d+', s)
>>> m.groups()
()
>>> print(m.lastgroup)
None
```

如上所示，这可能是因为最后捕获的组不是命名组，也可能是因为根本没有捕获的组。

`match.re`

> 包含匹配的正则表达式对象。

`match.re`包含产生匹配的正则表达式对象。如果您将正则表达式传递给`re.compile()`，您将得到相同的对象:

>>>

```py
 1>>> regex = r'(\w+),(\w+),(\w+)'
 2
 3>>> m1 = re.search(regex, 'foo,bar,baz')
 4>>> m1
 5<_sre.SRE_Match object; span=(0, 11), match='foo,bar,baz'>
 6>>> m1.re
 7re.compile('(\\w+),(\\w+),(\\w+)')
 8
 9>>> re_obj = re.compile(regex)
10>>> re_obj
11re.compile('(\\w+),(\\w+),(\\w+)')
12>>> re_obj is m1.re 13True
14
15>>> m2 = re_obj.search('qux,quux,corge')
16>>> m2
17<_sre.SRE_Match object; span=(0, 14), match='qux,quux,corge'>
18>>> m2.re
19re.compile('(\\w+),(\\w+),(\\w+)')
20>>> m2.re is re_obj is m1.re 21True
```

记住前面的内容,`re`模块在编译正则表达式后缓存它们，所以如果再次使用它们，不需要重新编译。出于这个原因，正如第 12 行**和第 20 行**的同一性比较所示，上面例子中所有不同的正则表达式对象都是完全相同的对象。

一旦您可以访问匹配的正则表达式对象，该对象的所有属性也是可用的:

>>>

```py
>>> m1.re.groups
3
>>> m1.re.pattern
'(\\w+),(\\w+),(\\w+)'
>>> m1.re.pattern == regex
True
>>> m1.re.flags
32
```

您还可以调用为编译后的正则表达式对象定义的任何[方法:](#regular-expression-object-methods)

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
>>> m.re
re.compile('(\\w+),(\\w+),(\\w+)')

>>> m.re.match('quux,corge,grault')
<_sre.SRE_Match object; span=(0, 17), match='quux,corge,grault'>
```

这里，`.match()`在`m.re`上被调用，使用相同的正则表达式在不同的搜索字符串上执行另一个搜索。

`match.string`

> 包含匹配的搜索字符串。

`match.string`包含作为匹配目标的搜索字符串:

>>>

```py
>>> m = re.search(r'(\w+),(\w+),(\w+)', 'foo,bar,baz')
>>> m.string
'foo,bar,baz'

>>> re_obj = re.compile(r'(\w+),(\w+),(\w+)')
>>> m = re_obj.search('foo,bar,baz')
>>> m.string
'foo,bar,baz'
```

从示例中可以看出，当 match 对象也是从一个编译过的正则表达式对象派生出来时，`.string`属性是可用的。

[*Remove ads*](/account/join/)

## 结论

您对 Python 的`re`模块的游览到此结束！

这个介绍性系列包含两篇关于 Python 中正则表达式处理的教程。如果你已经学完了[之前的教程](https://realpython.com/regex-python)和本教程，那么你现在应该知道如何:

*   充分利用`re`模块提供的所有功能
*   用 Python 预编译正则表达式
*   从匹配对象中提取信息

正则表达式非常通用和强大——实际上是一种独立的语言。您会发现它们在您的 Python 编码中是无价的。

**注意:**`re`模块很棒，它可能在大多数情况下都能很好地为你服务。然而，有一个替代的第三方 Python 模块叫做`regex`，它提供了更好的正则表达式匹配能力。你可以在 [`regex`项目页面](https://pypi.python.org/pypi/regex)了解更多。

在本系列的下一篇文章中，您将探索 Python 如何避免不同代码区域中标识符之间的冲突。正如您已经看到的，Python 中的每个函数都有自己的[名称空间](https://realpython.com/python-namespaces-scope/)，与其他函数截然不同。在下一篇教程中，您将学习如何在 Python 中实现名称空间，以及它们如何定义变量**范围**。

[« Regular Expressions: Regexes in Python (Part 1)](https://realpython.com/regex-python/)[Regular Expressions: Regexes in Python (Part 2)](#)[Namespaces and Scope in Python »](https://realpython.com/python-namespaces-scope/)********