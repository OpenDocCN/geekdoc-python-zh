# 如何使用 Python ascii()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-ascii-function>

在本文中，我们将看看 Python **ascii()** 函数。

ascii()函数返回对象的字符串表示，但实际上只有 ASCII 字符。

剩余的非 ASCII 字符将用反斜杠(\)进行转义。例如，换行符(`\n`)是非 ASCII 字符。

我们现在将看一些例子来理解它到底是如何工作的！

* * *

## 使用 Python ascii()函数–一些示例

Python **ascii()** 函数采用单个参数，该参数可以是任何对象。所以所有类型的对象，像列表，字符串等等，都是有效的。这将返回一个字符串。

如果你在一个[列表](https://www.askpython.com/python/list/python-list)或者任何集合上使用它，这个函数将为集合的每个成员调用。

现在让我们来看看这个。

* * *

### 在原始数据类型上使用 Python ascii()

对于像`boolean`、`string`、`int`这样的基本数据类型，它们按照您的预期工作。

```py
i = 15
print(ascii(i))

b = True
print(ascii(b))

s = 'abc'
print(ascii(s))

s = 'Hello from\tAskPython\n'
print(ascii(s))

```

**输出**

```py
'15'
'True'
"'abc'"
"'Hello from\\tAskPython\\n'"

```

如您所见，对于非 ASCII 字符(\t，\n)，反斜杠本身需要进行转义。

### 在 Iterables/Collections 上使用 ascii()

如果你想在列表/元组/字典上使用它，你仍然可以！但是，这只是将它应用于集合/iterable 中的每个成员。

因此，如果一个列表有`n`个元素，我们将把函数应用到所有的`n`个元素上，并得到一个字符串列表。

```py
m = ["Hello from AskPython 22", "AskPythön", "Hi"]
print(ascii(m))

```

**输出**

```py
['Hello from AskPython 22', 'AskPyth\xf6n', 'Hi']

```

类似地，对于字典{ `key` : `value` }，它将应用于`key`和`value`。

```py
d = {'â':'å', '2':2, 'ç':'ć'}
print(ascii(d))

```

**输出**

```py
{'\xe2': '\xe5', '2': 2, '\xe7': '\u0107'}

```

对于元组，它类似于列表。所有元素都将被转换成类似 ASCII 字符的字符串表示形式。

```py
t = ("Hellö", 123, ["AskPython"])
print(ascii(t))

```

**输出**

```py
('Hell\xf6', 123, ['AskPython'])

```

### 与 repr()函数的比较

`repr()`函数也用于返回对象的字符串表示。但是不同之处在于`repr()`同样打印非 ascii 字符。

对于定制对象，`ascii()`函数在内部调用`__repr__()`函数，但是确保对非 ASCII 字符进行转义。

让我们通过使用一个类创建我们自己的对象来试验一下。

```py
class MyClass:
    def __init__(self, name):
        self.name = name

```

现在，让我们创建一个对象并尝试在其上调用`ascii()`和`repr()`，

```py
my_obj = MyClass("AskPythön")
print(ascii(my_obj))
print(repr(my_obj))

```

**输出**

```py
'<__main__.MyClass object at 0x7f6adcf30940>'
'<__main__.MyClass object at 0x7f6adcf30940>'

```

我们没有这个类的`repr()`函数，所以使用默认的`object`定义。这就是你在输出中看到`MyClass object`的原因。

要改变这一点，我们必须自己重载`__repr__()` dunder 方法。

```py
class MyClass:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name

```

现在当你调用`ascii()`或者`repr()`的时候，我们可以直接得到 name 属性！

```py
my_obj = MyClass("AskPythön")
print(ascii(my_obj))
print(repr(my_obj))

```

**输出**

```py
AskPyth\xf6n
AskPythön

```

现在，你可以清楚地看到不同之处！

* * *

## 结论

在本文中，我们学习了在 Python 中使用`ascii()`函数，并学习了在不同类型的对象上使用它。

## 参考

*   [Python 文档](https://docs.python.org/3.8/library/functions.html#ascii)关于 ascii()函数
*   关于 Python ascii()的 JournalDev 文章

* * *