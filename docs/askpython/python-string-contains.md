# Python 字符串包含()

> 原文：<https://www.askpython.com/python/string/python-string-contains>

在本文中，我们将看看一个 [Python 字符串](https://www.askpython.com/python/string/python-string-functions)是否包含另一个字符串。

怎么才能轻松做到？Python 有一个内置的字符串方法 **String。__ 包含 __()** ，我们可以轻松使用。

让我们看看如何使用这种方法。

* * *

## 字符串的语法。__ 包含 _ _()

该函数将接受两个字符串，如果一个字符串属于另一个字符串，则返回。因此这个方法的返回类型是一个**布尔值**，所以它将返回一个**真**，或者一个**假**。

至于我们如何调用这个方法，我们在一个 string 对象上使用它，检查这个 string 对象中是否有另一个字符串。

```py
ret = str_object.contains(another_string)

```

这将检查`str_object`是否包含字符串`another_string`，返回值被存储到`ret`。

现在让我们看一些例子来说明这一点。

* * *

## 使用字符串。__ 包含 _ _()

我们将检查一个 Python 字符串是否包含另一个字符串。

```py
my_str = "Hello from AskPython"

target = "AskPython"

if (my_str.__contains__(target)):
    print("String contains target!")
else:
    print("String does not contain target")

```

**输出**

```py
String contains target

```

由于“AskPython”是原字符串的子串，“Hello from AskPython”，所以返回值为`True`。

这个方法是区分大小写的，所以字符串“askpython”将**而不是**得到匹配。

```py
my_str = "Hello from AskPython"

target = "askpython"

if (my_str.__contains__(target)):
    print("String contains target!")
else:
    print("String does not contain target")

```

**输出**

```py
String does not contain target

```

## 使用 Python String contains()作为类方法

我们也可以将它用作`str`类的类方法，并使用两个参数而不是一个。

```py
ret = str.__contains__(str1, str2)

```

这类似于我们之前的用法，但是我们调用它作为 String 类的类方法。这将返回`True`是`str1`包含`str2`，否则为`False`。

```py
>>> print(str.__contains__("Hello from AskPython", "AskPython")
True

```

* * *

## 结论

在本文中，我们学习了如何使用**字符串。__contains__()** 方法检查一个 Python 字符串是否包含另一个字符串。

* * *

## 参考

*   关于 Python 字符串的 JournalDev 文章包含()方法

* * *