# Python 正则表达式的零星内容

> 原文：<https://www.pythoncentral.io/the-odds-ends-of-python-regular-expressions/>

在本系列的前两部分中，我们研究了正则表达式的一些相当高级的用法。在这一部分中，我们后退一步，看看 Python 在 re 模块中提供的一些其他函数，然后我们讨论一些人们经常犯的错误(哈！)制造。

## 有用的 Python 正则表达式函数

Python 提供了几个函数，使得使用正则表达式操作字符串变得很容易。

*   通过一条语句，Python 可以返回与正则表达式匹配的所有子字符串的列表。

例如:

```py

>>> s = 'Hello world, this is a test!'

>>> print(re.findall(r'\S+', s))

['Hello', 'world,', 'this', 'is', 'a', 'test!']

```

`\S`表示*任何非空白字符*，所以正则表达式`\S+`表示*匹配一个或多个非空白字符*(例如一个单词)。

*   我们可以用另一个字符串替换每个匹配的子字符串。

例如:

```py

>>> print(re.sub( r'\S+' , 'WORD', s))

WORD WORD WORD WORD WORD WORD

```

对`re.sub`的调用用字符串“WORD”替换正则表达式(如单词)的所有匹配。

*   或者，如果您想遍历每个匹配的子字符串并自己处理它，`re.finditer`将遍历每个匹配，并在每次遍历时返回一个`MatchObject`。

例如:

```py

>>> for mo in re.finditer(r'\S+', s):

...    print('[%d:%d] = %s' % (mo.start(), mo.end(), mo.group()))

[0:5] = Hello

[6:12] = world,

[13:17] = this

[18:20] = is

[21:22] = a

[23:28] = test!

```

*   Python 也有一个函数，使用正则表达式作为分隔符，将字符串分割成多个部分。假设我们有一个字符串，它使用逗号和分号作为分隔符，到处都是空格。

例如:

```py

s = 'word1,word2 ,  word3;word4  ;  word'

```

分隔符的正则表达式是:`\s*[,;]\s*`。

或者用简单的英语说:

*   零个或多个空白字符。
*   逗号或分号。
*   零个或多个空白字符。

这就是它的作用:

```py

>>> s = 'word1,word2 ,  word3;word4  ;  word5'

>>> print(re.split(r'\s*[,;]\s*', s))

['word1', 'word2', 'word3', 'word4', 'word5']

```

每个单词都已被正确拆分，并删除了空格。

## 常见的 Python 正则表达式错误

### 搜索多行字符串时不使用 DOTALL 标志

在正则表达式中，特殊字符`.`表示*匹配任何字符*。

例如:

```py

>>> s = 'BEGIN hello world END'

>>> mo = re.search('BEGIN (.*) END', s)

>>> print(mo.group(1))

hello world

```

但是，如果被搜索的字符串由多行组成，**。**不匹配换行符(`\n`)。

```py

>>> s = '''BEGIN hello

...        world END'''

>>> mo = re.search('BEGIN (.*) END', s)

>>> print(mo)

None

```

我们的正则表达式说*找到单词 BEGIN，然后是一个或多个字符，然后是单词 END* ，所以发生的情况是 Python 找到了单词“BEGIN”，然后是一个或多个字符*直到换行符*，作为一个字符不匹配。然后，Python 查找单词“END ”,由于没有找到，正则表达式不匹配任何内容。

如果希望正则表达式匹配跨多行的子字符串，需要传入 **DOTALL** 标志:

```py

>>> mo = re.search('BEGIN (.*) END', s, re.DOTALL)

>>> print(mo.group())

BEGIN hello

world END

```

## 搜索多行字符串时不使用 MULTILINE 标志

在 UNIX 世界中，`^`和`$`被广泛理解为匹配一行的开始/结束，但是只有在设置了`MULTILINE`标志的情况下，Python 正则表达式才是这样。如果没有，它们将只匹配被搜索的整个字符串的开头/结尾。

```py

>>> s = '''hello

>>> ... world'''

>>> print(re.findall(r'^\S+$', s))

[]

```

为了获得我们期望的行为，传入`MULTILINE`(或简称为`M`标志:

```py

>>> print(re.findall(r'^\S+$', s, re.MULTILINE))

['hello', 'world']

```

## 不重复不贪婪

运算符`*`和`+`和`?`分别匹配 *0 个或更多的*、 *1 个或更多的*、 *0 个或 1 个*重复，默认情况下，它们是*贪婪的*(例如，它们试图匹配尽可能多的字符)。

一个典型的错误是试图使用这样的正则表达式来匹配 HTML 标签:`<.+&>`

这看起来很合理——匹配开始的`<`，然后一个或多个字符，然后结束的`>`——但是当我们在一些 HTML 上尝试时，会发生这样的情况:

```py

>>> s = '<head> <style> blah </style> </head>'

>>> mo = re.search('<.+>', s)

>>> print(mo.group())

<head> <style> blah </style> </style>

```

发生的事情是 Python 已经匹配了开始的`<`，然后一个或多个字符(head)，然后结束的`>`，但是它没有就此停止，而是尝试看看它是否可以做得更好，让`.`字符匹配*更多的*字符。事实上它可以，它可以匹配所有的东西，直到字符串最后的`>`，这就是为什么这个正则表达式最终匹配整个字符串。

解决这个问题的方法是在`.`操作符*后面加上一个`?`字符，使其成为非贪婪的*(例如，使其匹配尽可能少的字符)。

```py

>>> mo = re.search('<.+?>', s)

>>> print(mo.group())

<head>

```

现在，当 Python 到达第一个`>`(它关闭了初始的`标记)时，它会立即停止，而不是尝试看看是否能做得更好。`

 `## 不区分搜索的大小写

默认情况下，正则表达式区分大小写。例如:

```py

>>> s = 'Hello World!'

>>> mo = re.search('world', s)

>>> print(mo)

None

```

为了使搜索不区分大小写，传入`IGNORECASE`标志:

```py

>>> mo = re.search('world', s, re.IGNORECASE)

>>> print(mo.group())

World

```

## 不编译正则表达式

Python 做了大量工作来准备一个正则表达式，所以如果你要经常使用一个特定的正则表达式，首先编译它是值得的。

例如:

```py

>>> myRegex = re.compile('...')

>>> # This reads the file line-by-line

>>> for lineBuf in open(testFilename, 'r'):

... print(myRegex.findall(lineBuf))

```

现在 Python 只做一次准备工作，然后在每次循环中重用预编译的正则表达式，从而节省了大量时间。`