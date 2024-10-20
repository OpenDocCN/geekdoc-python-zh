# Python Unicode:编码和解码字符串(在 Python 2.x 中)

> 原文：<https://www.pythoncentral.io/python-unicode-encode-decode-strings-python-2x/>

This article is on Unicode with Python 2.x If you want to learn about Unicode for Python 3.x, be sure to checkout our [Unicode for Python 3.x](https://www.pythoncentral.io/encoding-and-decoding-strings-in-python-3-x/ "Unicode for Python 3.x article") article. Also, if you're interested in checking if a Unicode string is a number, be sure to checkout our article on [how to check if a Unicode string is a number](https://www.pythoncentral.io/how-to-check-if-a-string-is-a-number-in-python-including-unicode/ "Check if a String is a Number in Python with str.isdigit (and Unicode)").

字符串是 Python 中最常用的数据类型之一，有时您可能希望(或不得不)使用包含或完全由标准 ASCII 集之外的字符组成的字符串(例如，带有重音符号或其他标记的字符)。

Python 2.x 提供了一种称为 *Unicode 字符串*的数据类型，用于使用字符串编码和解码方法处理 Unicode 数据。如果你想了解更多关于 Unicode 字符串的知识，一定要查看维基百科上关于 [Unicode](http://en.wikipedia.org/wiki/Unicode "Unicode") 的文章。

注意:当执行一个包含 Unicode 字符的 Python 脚本时，必须在脚本的顶部加上下面一行，告诉 Python 代码是 UTF-8/Unicode 格式的。

```py

# -*- coding: utf-8 -*-

```

## Python Unicode:概述

为了弄清楚“编码”和“解码”是怎么回事，让我们看一个示例字符串:

```py

>>> s = "Flügel"

```

我们可以看到我们的字符串 *s* 中有一个非 ASCII 字符，即“ü”或“umlaut-u”。假设我们处于标准的 Python 2.x 交互模式，让我们看看当我们引用字符串时会发生什么，以及当它被打印出来时会发生什么:

```py

>>> s

'Fl\xfcgel'

>>> print(s)

Flügel

```

打印给了我们赋给变量的值，但是显然在这个过程中发生了一些事情，把我们输入解释器的内容变成了看似不可理解的东西。非 ASCII 字符ü被一组幕后规则翻译成一个代码短语，即“\xfc”。换句话说，它是由*编码的*。

在这一点上， *s* 是一个 8 位的字符串，对我们来说基本上意味着它不是一个 Unicode 字符串。让我们看看如何用相同的数据生成一个 Unicode 字符串。最简单的方法是在文字字符串前面加上“u”前缀，将其标记为 Unicode 字符串:

```py

u = u"Flügel"

```

如果我们像对待`s`一样引用并打印`u`，我们会发现类似的东西:

```py

>>> u

u'Fl\xfcgel'

>>> print(u)

Flügel

```

我们可以看到，我们的“umlaut-u”的代码短语仍然是“\xfc ”,并且它打印出来是一样的——那么这是否意味着我们的 Unicode 字符串的编码方式与我们的 8 位字符串 *s* 相同呢？为了弄清楚这一点，让我们看看当我们在`u`和`s`上尝试时`encode`方法做了什么:

```py

>>> u.encode('latin_1')

'Fl\xfcgel'

>>> s.encode('latin_1')

Traceback (most recent call last):

   File "<pyshell#35>", line 1, in <module>

   s.encode('latin_1')

UnicodeDecodeError: 'ascii' codec can't decode byte 0xfc in position 2: ordinal not in range(128)

```

现在看起来编码 Unicode 字符串(用‘Latin-1’编码)返回了与字符串`s`相同的值，但是`encode`方法对字符串`s`不起作用。既然我们无法编码`s`，那解码呢？会给我们和`u`一样的价值吗？让我们来看看:

```py

>>> s.decode('latin-1')

u'Fl\xfcgel'

```

毕竟这正是它的作用。那么，`s`是一个 8 位字符串，而`u`是一个 Unicode 字符串，这有什么区别呢？他们的行为方式是一样的，不是吗？在我们的“umlaut-u”示例中，除了 Unicode 字符串前面的“u”之外，似乎没有什么不同。

区别在于，Unicode 字符串`u`使用的是 Unicode 标准为字符“umlaut-u”定义的代码短语，而 8 位字符串`s`使用的是“latin-1”编解码器(规则集)为“umlaut-u”定义的代码短语。

好吧，嗯...这很好，但是...他们还是一样的，对吗？那么，这有什么关系呢？

为了说明区别及其重要性，让我们考虑一个新的 8 位字符串:

```py

new_s = '\xe5\xad\x97'

```

与第一个不同，我们新的 8 位字符串是`only`代码短语——完全无法理解。

为什么我们不像最后一个 8 位字符串那样直接输入(或复制粘贴)字符呢？好吧，假设我们仍然使用标准的 Python 2.x console/IDLE，我们*不能*将这个值键入或粘贴到解释器中——因为如果我们这样做，它不会接受这个值。为什么？因为我们的`new_s`是一个亚洲文字字符的编码字符串(是的，只有一个字符)，并且交互模式/IDLE 反对这样的输入(如果您的系统安装了合适的输入键盘，您可以尝试一下并找出答案)。

现在的问题是，我们如何将这些代码短语转换成它们应该显示的字符？在第一个例子中，在`s`上使用 print 语句效果很好，所以对于`new_s`应该也是一样，对吗？让我们看看我们未知的亚洲文字是什么:

```py

>>>print new_s

å­—

```

啊哦...那不对。首先，那不是亚洲文字。其次，不止是一个人物。简单地引用`new_s`会给出我们分配给它的字符串，而`print`似乎不起作用。让我们看看 Unicode 字符串是否能帮到我们。

要创建新的 Unicode 字符串`new_u`，我们不能使用第一个例子中的方法——要这样做，我们必须输入带有“u”前缀的字符串的文字字符(我们还没有看到我们的字符，无论如何交互模式/IDLE 不会接受它作为输入)。

然而，我们确实通过*解码* `s`得到了`u`的值，所以同样地，我们应该能够通过解码`new_s`得到`new_u`的值。让我们像第一个例子那样尝试解码:

```py

>>> new_u = new_s.decode('latin_1')

>>> new_u

u'\xe5\xad\x97'

```

很好，现在我们已经使用与第一个示例相同的方法存储了解码后的`new_s`字符串值，让我们打印我们的 Unicode 字符串，看看我们的脚本字符是什么:

```py

>>> print(new_u)

å

```

哦...这不就是我们试图打印`new_s`字符串时得到的结果吗？？那么使用 Unicode 字符串真的没有什么不同吗？

没那么快——有一个细节被有意掩盖以证明一点:我们用来解码字符串的编码与第一个例子相同，即“latin-1”编解码器。然而，8 位字符串`new_s`不是用“拉丁语-1”编码的*，而是用“utf-8”编码的*

好吧，所以除非明确告诉你，否则你真的没有办法知道，但这仍然说明了这一点:*编码/编解码器/规则集在编码/解码字符串时会产生很大的不同。*

使用正确的编码，让我们看看会发生什么:

```py

>>> new_u = new_s.decode('utf_8')

>>> new_u

u'\u5b57'

>>> print(new_u)

字

```

终于！我们失踪已久的剧本角色找到了，看起来还不错。现在，尝试复制粘贴字符作为输入——你会发现它不起作用(我们仍然在谈论 Python 2.x 交互模式/IDLE)。

您可能还会注意到 *new_u* 的值有所不同，即它似乎只由一个代码短语组成(这次是以' \uXXXX '的形式)。这个 Unicode 标准为每个可能显示在屏幕上的字符或脚本字符的提供了一个唯一的代码短语。记住这一点，你也可以看出我们第一次试图解码 *new_s，*值是错误的(“u'\xe5\xad\x97 '”有 3 个代码短语，对于 Unicode 标准，这意味着 3 个独特的字符)。

好了，现在那些烦人的例子讲完了，让我们来回顾一下所有这些喧闹的要点:

1.  字符串是 Python 中最常见的数据类型之一，有时它们会包含非 ASCII 字符。
2.  当字符串包含非 ASCII 字符时，可以是 8 位字符串(*编码字符串*)，也可以是 Unicode 字符串(*解码字符串*)。
3.  要正确打印或显示一些字符串，需要对它们进行**解码** (Unicode 字符串)。
4.  编码/解码字符串时，编码/编解码器至关重要。

编码/编解码器就像你的字符串的 DNA——即使有相同的营养成分(输入),使用错误的 DNA(编解码器)会给你一个橘子，而你应该有一个苹果..