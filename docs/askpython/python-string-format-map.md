# 使用 Python 字符串 format_map()

> 原文：<https://www.askpython.com/python/string/python-string-format-map>

在本文中，我们将了解一下 [Python 字符串](https://www.askpython.com/python/string/python-string-functions) **format_map()** 方法。

该方法使用基于映射的替换返回字符串的格式化版本，使用花括号{}。

让我们用几个例子来正确理解这一点。

* * *

## Python 字符串格式 _map()基础知识

Python 字符串 format_map()函数从 Python 3.2 开始可用，因此请确保您使用的是 Python 的更新版本，而不是旧版本。

该字符串方法的基本语法如下:

```py
substituted_string = str.format_map(mapping)

```

在这里，`mapping`可以是任何映射，就像字典一样。映射可以被看作是{ `key` : `value` }的形式。

**Python String format_map()** 方法用`value`替换字符串中的所有`keys`。

这将返回一个新的字符串，包括所有替换(如果适用)。

为了更好地理解这一点，考虑下面的映射字典:

```py
dict_map = {"prog_lang": "Python", "site": "AskPython"}

```

现在，考虑一个格式字符串，在格式替换(花括号)下有字典的键。

```py
fmt_string = "Hello from {site}. {site} is a site where you can learn {prog_lang}"

```

现在，我们可以用“AskPython”替换所有出现的`{site}`，用`format_map()`替换所有出现的`{prog_lang}`。

```py
print(fmt_string.format_map(dict_map))

```

**输出**

```py
Hello from AskPython. AskPython is a site where you can learn Python

```

通过所有替换，我们得到了我们想要的输出！

现在，如果我们有一个在映射字典中不存在的额外格式会怎么样呢？

```py
dict_map = {"prog_lang": "Python", "site": "AskPython"}
fmt_string = "Hello from {site}. {site} is a site where you can learn {prog_lang}. What about {other_lang}?"
print(fmt_string.format_map(dict_map))

```

**输出**

```py
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    print(fmt_string.format_map(dict_map))
KeyError: 'other_lang'

```

我们得到一个`KeyError`异常。由于`{other_lang}`不属于映射字典，查找将失败！

* * *

## Python 字符串 format_map()与 format()的比较

您可能还记得， [format()](https://www.askpython.com/python/string/python-format-function) 方法也非常类似，通过对格式字符串进行适当的替换。

差异可总结如下:

*   format()方法**通过首先创建映射字典，然后执行替换，间接地**使用该方法的参数执行替换。
*   在 Python 字符串 format_map()的情况下，替换是使用映射字典*直接*完成的。
*   由于 **format_map()** 也不做新字典，所以比 format()略快。
*   **format_map()** 也可以使用 dictionary 子类进行映射，而 format()则不能。

为了说明最后一点，让我们创建一个类，它是`dict`的子类。

我们将尝试上述两种方法，并尝试使用`__missing__()` dunder 方法处理任何丢失的键。

```py
class MyClass(dict):
    def __missing__(self, key):
        return "#NOT_FOUND#"

fmt_string = "Hello from {site}. {site} is a site where you can learn {prog_lang}."

my_dict = MyClass(site="AskPython")

print(fmt_string.format_map(my_dict))

```

**输出**

```py
Hello from AskPython. AskPython is a site where you can learn #NOT_FOUND#.

```

这里发生了什么事？因为我们只在映射字典中添加了`{site: AskPython}`，所以缺少了`{prog_lang}`键。

因此，`__missing__()`方法返回“ **#NOT_FOUND#** ”。

如果我们使用 format()？

```py
fmt_string.format(my_dict)

```

**输出**

```py
Traceback (most recent call last):
  File "<pyshell#14>", line 1, in <module>
    fmt_string.format(my_dict)
KeyError: 'site'

```

嗯，`format()`没有为我们处理这个问题，它只是引发了一个`KeyError`异常。这是因为它将映射复制到一个新的 dictionary 对象。

因为映射是在一个新的 dictionary 对象上(没有使用我们的子类)，所以它没有 **__missing__** 方法！因此，它只能给出一个键错误！

* * *

## 结论

在本文中，我们学习了如何使用 Python String format_map()方法对格式字符串执行替换。我们还看到了与 **format()** 方法的快速比较。

## 参考

*   String format_map()上的 JournalDev 文章

* * *