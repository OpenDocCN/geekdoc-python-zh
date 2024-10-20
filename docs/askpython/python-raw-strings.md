# 如何使用 Python 原始字符串？

> 原文：<https://www.askpython.com/python/string/python-raw-strings>

大家好！在本文中，我们将了解如何使用 Python **原始字符串**。这是 Python 的一个强大特性，利用它我们可以引入“原始字符串”，而不用转义任何字符。

Python 原始字符串是一个普通字符串，前缀是 **r** 或 **R** 。

这将反斜杠(“\”)等字符视为文字字符。这也意味着该字符不会被视为转义字符。

现在让我们看看使用原始字符串，使用一些说明性的例子！

* * *

## python 原始字符串

为了理解原始字符串的确切含义，让我们考虑下面的字符串，其序列为“\n”。

```py
s = "Hello\tfrom AskPython\nHi"
print(s)

```

现在，由于`s`是一个普通的字符串文字，序列“\t”和“\n”将被视为转义字符。

因此，如果我们打印该字符串，就会生成相应的转义序列(制表符和换行符)。

```py
Hello    from AskPython
Hi

```

现在，如果我们想让`s`成为一个原始字符串，会发生什么？

```py
# s is now a raw string
# Here, both backslashes will NOT be escaped.
s = r"Hello\tfrom AskPython\nHi"
print(s)

```

这里，两个反斜杠都不会被视为转义字符，所以 Python 不会打印制表符和换行符。

相反，它将简单地逐字打印“\t”和“\n”。

```py
Hello\tfrom AskPython\nHi

```

正如您所看到的，输出和输入是一样的，因为没有字符被转义！

现在，让我们看看另一个场景，原始字符串对我们非常有用，尤其是当 Python 字符串不起作用时。

考虑下面的字符串文字，具有序列“\x”。

```py
s = "Hello\xfrom AskPython"
print(s)

```

这里，序列“\x”不能使用标准的 unicode 编码进行解码。

```py
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 5-7: truncated \xXX escape

```

这意味着我们甚至不能把它放入字符串中。我们现在能做什么？

这就是原始字符串派上用场的地方。

我们可以很容易地把这个值传递给一个变量，只要把它看作一个原始的字符串文字！

```py
s = r"Hello\xfrom AskPython"
print(s)

```

现在，没有问题了，我们可以把这个原始的字符串文字作为一个普通的对象来传递！

```py
Hello\xfrom AskPython

```

**注意**:在某些情况下，如果您在控制台上打印 Python 原始字符串，您可能会得到如下结果:

```py
>>> r"Hello\xfrom AskPython"
'Hello\\xfrom AskPython'

```

这里，双反斜杠意味着它是一个普通的 Python 字符串，反斜杠被转义。由于`print()`函数打印常规的字符串文字，原始字符串被转换成这样一个字符串！

* * *

## 结论

在本文中，我们学习了如何使用 Python 原始字符串处理特殊字符而不转义它们。

* * *

## 参考

*   关于 Python 原始字符串的 JournalDev 文章

* * *