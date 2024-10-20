# Python 正则表达式简介

> 原文：<https://www.pythoncentral.io/introduction-to-python-regular-expressions/>

在 Python 中，在一个字符串中搜索另一个字符串相当容易:

```py

>>> str = 'Hello, world'

>>> print(str.find('wor'))

7

```

如果我们确切地知道*我们要找的是什么，这很好，但是如果我们要找的东西不是那么明确呢？例如，如果我们想搜索一年，那么我们知道这将是一个 4 位数的序列，但我们不知道这些数字将会是什么。这就是*正则表达式*的用武之地。它们允许我们基于我们所寻找的内容的一般描述来搜索子字符串，例如*搜索 4 个连续数字的序列*。*

在下面的例子中，我们导入了包含 Python 正则表达式功能的`re`模块，然后用我们的正则表达式(`\d\d\d\d`)和我们想要搜索的字符串调用`search`函数:

```py

>>> import re

>>> str = 'Today is 31 May 2012.'

>>> mo = re.search(r'\d\d\d\d', str)

>>> print(mo)

<_sre.SRE_Match object at 0x01D3A870>

>>> print(mo.group())

2012

>>> print('%s %s' % (mo.start(), mo.end())

16 20

```

在一个正则表达式中，`\d`表示*任意位*，所以`\d\d\d\d`表示*任意位，任意位，任意位*，或者说白了，*连续 4 位*。正则表达式大量使用反斜杠，反斜杠在 Python 中有特殊的含义，所以我们在字符串前面加了一个 *r* 使其成为一个*原始字符串*，它阻止 Python 以任何方式解释反斜杠。

如果`re.search`找到匹配我们正则表达式的东西，它返回一个*匹配对象*,其中保存了关于匹配内容的信息。在上面的例子中，我们打印出匹配的子字符串，以及它在被搜索的字符串中的开始和结束位置。

注意 Python 没有匹配日期(`31`)。它会匹配前两个字符，即与前两个`\d`匹配的 *3* 和 *1* ，但是下一个字符(一个空格)不会匹配第三个`\d`，所以 Python 会放弃并继续搜索字符串的剩余部分。

## 匹配一组字符

让我们试试另一个例子:

```py

>>> str = 'Today is 2012-MAY-31'

>>> mo = re.search(r'\d\d\d\d-[A-Z][A-Z][A-Z]-\d\d', str)

>>> print(mo.group())

2012-MAY-31

```

这一次，我们的正则表达式包含新元素`[A-Z]`。方括号表示*与其中一个字符*完全匹配。例如，`[abc]`表示 Python 将匹配一个 *a* 或 *b* 或 *c* ，但不匹配其他字母。由于我们想要匹配 *A* 和 *Z* 之间的任何字母，我们可以写出整个字母表(`[ABCDEFGHIJKLMNOPQRSTUVWXYZ]`)，但是谢天谢地，Python 允许我们使用连字符(`[A-Z]`)来缩短这个字母。所以，我们的正则表达式是`\d\d\d\d-[A-Z][A-Z][A-Z]-\d\d`，意思是:

*   寻找(或*匹配*)一个数字(4 次)。
*   匹配一个'-'字符。
*   匹配 A 和 Z 之间的一个字母(三次)。
*   匹配一个'-'字符。
*   匹配一个数字(2 次)。

如上例所示，Python 找到了嵌入在字符串中的日期。

不幸的是，我们的正则表达式目前只处理大写的月份名称:

```py

# The month uses lower-case letters

>>> str = 'Today is 2012-May-31'

>>> mo = re.search(r'\d\d\d\d-[A-Z][A-Z][A-Z]-\d\d', str)

>>> print(mo)

None

```

有两种方法可以解决这个问题。我们可以传入一个标志，表示搜索应该不区分大小写:

```py

>>> str = 'Today is 2012-May-31'

>>> mo = re.search(r'\d\d\d\d-[A-Z][A-Z][A-Z]-\d\d', str, re.IGNORECASE)

>>> print(mo.group())

2012-May-31

```

或者，我们可以扩展字符集来指定更多的字符:`[A-Za-z]`表示*大写 A 到大写 Z，或者小写 A 到小写 Z* 。

```py

>>> str = 'Today is 2012-May-31'

>>> mo = re.search(r'\d\d\d\d-[A-Za-z][A-Za-z][A-Za-z]-\d\d', str)

>>> print(mo.group())

2012-May-31

```

## 正则表达式的重复

上一个例子中的正则表达式开始变得有点笨拙，所以让我们看看如何简化它。

在正则表达式中，`{n}`(其中 n 是数字)表示*重复前面的元素 n 次*。所以我们可以重写这个正则表达式:

```py
\d\d\d\d-[A-Za-z][A-Za-z][A-Za-z]-\d\d
```

变成这样:

```py
\d{4}-[A-Za-z]{3}-\d{2}
```

这意味着:

*   匹配任意数字(4 次)。
*   匹配一个'-'字符。
*   匹配字母 A-Z 或 a-z (3 次)。
*   匹配一个'-'字符。
*   匹配任意数字(2 次)。

这就是它的作用:

```py

>>> str = 'Today is 2012-May-31'

>>> mo = re.search(r'\d{4}-[A-Za-z]{3}-\d{2}', str)

>>> print(mo.group())

2012-May-31

```

当指定应该匹配多少重复时，我们有很大的灵活性。

*   我们可以指定一个范围，例如`{2,4}`表示*匹配 2 - 4 次重复*。

```py

>>> str = 'abc12345def'

>>> mo = re.search(r'\d{2,4}', str)

>>> print(mo.group())

1234

```

*   我们可以忽略上限值，例如`{2,}`表示“匹配 2 次或更多次重复”。

```py

>>> str = "abc12345def"

>>> mo = re.search(r'\d{2,}', str)

>>> print(mo.group())

12345

```

# 常见重复的简写

有些类型的重复非常常见，它们有自己的语法。

*   `{1,}`表示*匹配前一个元素一次或多次*，但这也可以用特殊的`+`操作符来写(例如`\d+`)。

```py

>>> str = 'abc12345def'

>>> mo = re.search(r'\d+', str)

>>> print(mo.group())

12345

```

*   `{0,}`表示*匹配前一个元素零次或多次*，但这也可以用`*`操作符来写(如`\d*`)。

```py

>>> str = 'abc12345def'

>>> mo = re.search(r'\d*', str)

>>> print(mo.group())

```

呀，发生什么事了？！为什么这个什么都没打印出来？嗯，你必须小心*操作符，因为它会匹配*零*或更多的重复。在这种情况下，Python 看着被搜索字符串的第一个字符，对自己说*这是数字吗？没有。我匹配了零个或多个数字吗？是(零)，所以正则表达式已经匹配了。*如果我们看看`MatchObject`告诉我们的:

```py

>>> print('%s %s' % (mo.start(), mo.end()))

0 0

```

我们可以看到这正是所发生的，它在被搜索的字符串的最开始匹配了一个空的子字符串。让我们稍微改变一下我们的正则表达式:

```py

>>> str = 'abc12345def'

>>> mo = re.search(r'c\d*', str)

>>> print(mo.group())

c12345

```

现在我们的正则表达式说*匹配字母 c，然后零个或多个数字*，这就是 Python 随后找到的。

*   `{0,1}`表示*匹配前一个元素 0 或 1 次*，但这也可以用`?`操作符来写(如`\d?`)。

```py

>>> str = 'abc12345def'

>>> mo = re.search(r'c\d?', str)

# Note: the \d was matched 1 time

>>> print(mo.group())

c1

>>> mo = re.search(r'b\d?', str)

# Note: the \d was matched 0 times

>>> print(mo.group())

b

```

在下一篇文章中，我们将继续讨论正则表达式的更多高级用法。