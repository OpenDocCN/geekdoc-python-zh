# Python 字符串是可打印的()

> 原文：<https://www.askpython.com/python/string/python-string-isprintable>

Python String**is printable()**函数可用于检查 Python 字符串是否可打印。

这到底是什么意思？让我们来了解一下！

* * *

## Python 字符串是可打印的()

Python 字符串中的某些字符不能直接打印到控制台(文件)。在 Unicode 字符数据库中，它们被归类为“其他”或“分隔符”。

注意，ASCII 空格字符( **0x20** )是一个例外，因为它显然可以被打印出来。

考虑可打印字符串的另一种方式是，它可以被成功解码为 Unicode 字符。任何不可打印的字符串都会引发这方面的错误。

它大致相当于以下方法:

```py
def isprintable(s, codec='utf8'):
    try: s.decode(codec)
    except UnicodeDecodeError: return False
    else: return True

```

如果我们能得到解码后的`utf-8`格式的字符串，那么它就可以被打印出来。否则，我们干脆返回`False`。

因此，让我们用一些例子来看看如何使用这个函数。

要使用这个函数，我们必须在 string 对象上调用 Python string isprintable()方法。

```py
ret = string.isprintable()

```

`ret`为布尔型，如果字符串可以打印，则为`True`。否则，就是`False`。

```py
>>> a = "Hello from AskPython"
>>> print(a.isprintable())
True

```

这里，我们的字符串中的所有字符都是可打印的，所以它返回`True`。

空字符串也是可打印的。

```py
>>> b = ""
>>> print(b.isprintable())
True

```

我们不能打印像`\n`或`\t`这样的转义字符，所以如果我们的字符串有任何转义序列，`isprintable()`将返回`False`。

```py
>>> c = "Hello from AskPython\n"
>>> print(c.isprintable())
False

>>> d = "Hello\tfrom\tAskPython"
>>> print(d.isprintable())
False

```

有些字符(像`\u0066` - > f)可以打印，有些字符(像`\u0009` - > \t)不能。

基本上，它们必须映射到有效的 unicode 字符。

```py
>>> e = "Hello \u0066rom AskPython"
>>> print(e.isprintable())
True
>>> print(e)
Hello from AskPython

>>> f = "Hello \u0066rom\u0009AskPython"
>>> print(f.isprintable())
False
>>> print(f)
Hello from      AskPython

```

为了结束 isprintable()方法的工作，让我们看看如何找出 unicode 数据库中所有不可打印的字符。

## 找出所有不可打印的字符

unicode 字符的总数是`2^16`，所以我们将创建一个遍历每个字符的循环，并检查它是否可以打印。

```py
count = 0

for ascii_val in range(2 ** 16):
    ch = chr(ascii_val)
    if not ch.isprintable():
        count += 1

print(f"Total Number of Non-Printable Unicode Characters = {count}")

```

**输出**

```py
Total Number of Non-Printable Unicode Characters = 10249

```

* * *

## 结论

在本文中，我们学习了如何使用 Python String isprintable()方法来检查字符串的字符是否可以被打印。

## 参考

*   字符串上的 JournalDev 文章是可打印的()
*   String 上的 [StackOverflow 问题【isprintable()](https://stackoverflow.com/questions/3636928/test-if-a-python-string-is-printable)

* * *