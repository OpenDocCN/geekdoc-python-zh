# 自定义 Python 字符串:从 str 和 UserString 继承

> 原文：<https://realpython.com/inherit-python-str/>

Python **`str`** 类有许多有用的特性，当你在代码中处理**文本**或**字符串**时，这些特性可以帮到你。然而，在某些情况下，所有这些伟大的功能可能还不够。您可能需要创建**自定义的类似字符串的类**。在 Python 中要做到这一点，你可以直接从内置的`str`类继承，或者继承位于`collections`模块中的子类`UserString`。

**在本教程中，您将学习如何:**

*   通过继承**内置的`str`类**来创建定制的类似字符串的类
*   通过从 **`collections`模块**中子类化 **`UserString`** 来构建定制的类似字符串的类
*   决定何时使用`str`或`UserString`来创建**自定义的类似字符串的类**

同时，您将编写几个例子来帮助您在创建自定义字符串类时决定是使用`str`还是`UserString`。您的选择将主要取决于您的具体用例。

为了跟随本教程，如果您熟悉 Python 的内置 [`str`](https://realpython.com/python-strings/) 类及其标准特性，将会有所帮助。你还需要了解 Python 中的[面向对象编程](https://realpython.com/python3-object-oriented-programming/)和[继承](https://realpython.com/inheritance-composition-python/)的基础知识。

**示例代码:** [点击这里下载免费的示例代码](https://realpython.com/bonus/inherit-python-str-code/)，您将使用它来创建定制的类似字符串的类。

## 在 Python 中创建类似字符串的类

内置的 [`str`](https://realpython.com/python-strings/) 类允许你在 Python 中创建**字符串**。字符串是您将在许多情况下使用的字符序列，尤其是在处理文本数据时。有时，Python 的标准功能可能不足以满足您的需求。因此，您可能希望创建自定义的类似字符串的类来解决您的特定问题。

您通常会发现创建定制的类似字符串的类至少有两个原因:

1.  **通过添加新功能来扩展**常规字符串
2.  **修改**标准字符串的功能

您还可能面临需要同时扩展*和*修改字符串的标准功能的情况。

在 Python 中，您通常会使用以下技术之一来创建类似字符串的类。可以直接从 Python 内置的 [`str`](https://docs.python.org/3/library/stdtypes.html#str) 类继承或者从 [`collections`](https://realpython.com/python-collections-module/) 子类 [`UserString`](https://docs.python.org/3/library/collections.html#collections.UserString) 。

**注:**在[面向对象编程](https://realpython.com/python3-object-oriented-programming/)中，通常的做法是将动词**继承**和**子类**互换使用。

Python 字符串的一个相关特性是[不变性](https://docs.python.org/3/glossary.html#term-immutable)，这意味着您不能就地修改它们[。因此，当选择合适的技术来创建自己的定制的类似字符串的类时，您需要考虑您想要的特性是否会影响不变性。](https://en.wikipedia.org/wiki/In-place_algorithm)

例如，如果您需要修改现有 string 方法的当前行为，那么您可以子类化`str`。相比之下，如果你需要改变字符串的创建方式，那么从`str`继承将需要高级知识。你必须覆盖 [`.__new__()`](https://realpython.com/python-class-constructor/#object-creation-with-__new__) 方法。在后一种情况下，继承`UserString`可能会让你的生活更轻松，因为你不必碰`.__new__()`。

在接下来的部分中，您将了解每种技术的优缺点，这样您就可以决定哪种策略是解决特定问题的最佳策略。

[*Remove ads*](/account/join/)

## 从 Python 内置的`str`类继承而来

很长一段时间，直接继承用 [C](https://realpython.com/c-for-python-programmers/) 实现的 Python 类型是不可能的。Python 2.2 修复了这个问题。现在你可以[子类内置类型](https://docs.python.org/3/whatsnew/2.2.html#peps-252-and-253-type-and-class-changes)，包括`str`。当您需要创建自定义的类似字符串的类时，这个新特性非常方便。

通过直接从`str`继承，您可以扩展和修改这个内置类的标准行为。您还可以在新实例准备好之前，调整您的自定义字符串类的[实例化过程](https://realpython.com/python-class-constructor/#pythons-class-constructors-and-the-instantiation-process)来执行转换。

### 扩展字符串的标准行为

需要定制的类似字符串的类的一个例子是当您需要用新的行为扩展标准 Python 字符串时。例如，假设您需要一个类似字符串的类，它实现一个新方法来计算底层字符串中的字数。

在本例中，您的自定义字符串将使用空白字符作为其默认的单词分隔符。但是，它还应该允许您提供特定的分隔符。要编写满足这些需求的类，您可以这样做:

>>>

```py
>>> class WordCountString(str):
...     def words(self, separator=None):
...         return len(self.split(separator))
...
```

这个类直接继承自`str`。这意味着它提供了与其父类相同的接口。

在这个继承的接口之上，添加一个名为`.words()`的新方法。这个方法将一个`separator`字符作为参数传递给`.split()`。它的缺省值是`None`，它将[在连续的空格中分割](https://docs.python.org/3/library/stdtypes.html#str.split)。然后调用带有目标分隔符的`.split()`将底层字符串拆分成单词。最后，你用 [`len()`](https://realpython.com/len-python-function/) 函数来确定字数。

下面是如何在代码中使用该类:

>>>

```py
>>> sample_text = WordCountString(
...     """Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime
...     mollitia, molestiae quas vel sint commodi repudiandae consequuntur
...     voluptatum laborum numquam blanditiis harum quisquam eius sed odit
...     fugiat iusto fuga praesentium optio, eaque rerum! Provident similique
...     accusantium nemo autem. Veritatis obcaecati tenetur iure eius earum
...     ut molestias architecto voluptate aliquam nihil, eveniet aliquid
...     culpa officia aut! Impedit sit sunt quaerat, odit, tenetur error,
...     harum nesciunt ipsum debitis quas aliquid."""
... )

>>> sample_text.words()
68
```

酷！你的方法很有效。它将输入文本拆分成单词，然后返回单词计数。您可以修改这个方法如何定界和处理单词，但是当前的实现对于这个演示性的例子来说工作得很好。

在这个例子中，您没有修改 Python 的`str`的标准行为。您刚刚向自定义类添加了新的行为。然而，也可以通过覆盖它的任何默认方法来改变`str`的默认行为，这将在接下来进行探讨。

### 修改字符串的标准行为

为了学习如何在一个定制的类似字符串的类中修改`str`的标准行为，假设你需要一个字符串类，它总是[用大写字母打印](https://realpython.com/python-print/)。你可以通过覆盖`.__str__()` [特殊方法](https://docs.python.org/3/glossary.html#term-special-method)来实现，该方法负责字符串对象的打印。

这里有一个`UpperPrintString`类，它的行为符合您的需要:

>>>

```py
>>> class UpperPrintString(str):
...     def __str__(self):
...         return self.upper()
...
```

同样，这个类继承自`str`。`.__str__()`方法返回底层字符串`self`的副本，其中所有字母都是大写的。要变换字母，您使用 [`.upper()`](https://docs.python.org/3/library/stdtypes.html#str.upper) 方法。

要尝试您的自定义字符串类，请继续运行以下代码:

>>>

```py
>>> sample_string = UpperPrintString("Hello, Pythonista!")

>>> print(sample_string)
HELLO, PYTHONISTA!

>>> sample_string
'Hello, Pythonista!'
```

当您打印一个`UpperPrintString`的实例时，您会在屏幕上看到大写字母的字符串。请注意，原始字符串没有被修改或影响。您只更改了`str`的标准打印功能。

[*Remove ads*](/account/join/)

### 调整`str` 的实例化过程

在这一部分，您将做一些不同的事情。您将创建一个类似字符串的类，它在生成最终的字符串对象之前转换原始的输入字符串。例如，假设您需要一个类似字符串的类，它以小写形式存储所有字母。为此，您将尝试覆盖类初始值设定项`.__init__()`，并执行如下操作:

>>>

```py
>>> class LowerString(str):
...     def __init__(self, string):
...         super().__init__(string.lower())
...
```

在这个代码片段中，您提供了一个覆盖默认`str`初始化器的`.__init__()`方法。在这个`.__init__()`实现中，您使用 [`super()`](https://realpython.com/python-super/) 来访问父类的`.__init__()`方法。然后，在初始化当前字符串之前，调用输入字符串上的`.lower()`将它的所有字母转换成小写字母。

但是，上面的代码不起作用，您将在下面的示例中确认这一点:

>>>

```py
>>> sample_string = LowerString("Hello, Pythonista!")
Traceback (most recent call last):
    ...
TypeError: object.__init__() takes exactly one argument...
```

因为`str`对象是不可变的，你不能在`.__init__()`中改变它们的值。这是因为该值是在对象创建期间设置的，而不是在对象初始化期间设置的。在实例化过程中转换给定字符串值的唯一方法是覆盖 [`.__new__()`](https://realpython.com/python-class-constructor/#object-creation-with-__new__) 方法。

下面是如何做到这一点:

>>>

```py
>>> class LowerString(str):
...     def __new__(cls, string):
...         instance = super().__new__(cls, string.lower())
...         return instance
...

>>> sample_string = LowerString("Hello, Pythonista!")
>>> sample_string
'hello, pythonista!'
```

在这个例子中，您的`LowerString`类覆盖了超类的`.__new__()`方法来定制实例的创建方式。在这种情况下，您在创建新的`LowerString`对象之前转换输入字符串。现在，您的类按照您需要的方式工作。它接受一个字符串作为输入，并将其存储为小写字符串。

如果您需要在实例化时转换输入字符串，那么您必须覆盖`.__new__()`。这项技术需要 Python 的数据模型[和特殊方法的高级知识。](https://docs.python.org/3/reference/datamodel.html)

## 从`collections` 子类化`UserString`

第二个允许您创建定制的类似字符串的类的工具是来自`collections`模块的`UserString`类。这个类是内置`str`类型的包装器。当不能直接从内置的`str`类继承时，它被设计用来开发类似字符串的类。

直接子类化`str`的可能性意味着你可能不太需要`UserString`。然而，为了方便和向后兼容，这个类仍然可以在[标准库](https://docs.python.org/3/library/index.html)中找到。在实践中，这个类也有一些隐藏的有用的特性，您很快就会了解到。

`UserString`最相关的特性是它的`.data`属性，它允许您访问包装的字符串对象。该属性有助于创建定制字符串，尤其是在您希望的定制影响字符串可变性的情况下。

在接下来的两个小节中，您将重温前面小节中的例子，但是这次您将子类化`UserString`而不是`str`。首先，您将从扩展和修改 Python 字符串的标准行为开始。

### 扩展和修改字符串的标准行为

你可以通过继承`UserString`类来实现`WordCountString`和`UpperPrintString`，而不是继承内置的`str`类。这个新的实现只需要你改变超类。您不必更改类的原始内部实现。

以下是`WordCountString`和`UpperPrintString`的新版本:

>>>

```py
>>> from collections import UserString

>>> class WordCountString(UserString):
...     def words(self, separator=None):
...         return len(self.split(separator))
...

>>> class UpperPrintString(UserString):
...     def __str__(self):
...         return self.upper()
...
```

这些新实现与原始实现之间的唯一区别是，现在您是从`UserString`继承的。注意，从`UserString`继承需要你从`collections`模块[导入](https://realpython.com/python-import/)该类。

如果您使用与之前相同的示例来尝试这些类，那么您将会确认它们与基于`str`的等价类工作相同:

>>>

```py
>>> sample_text = WordCountString(
...     """Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime
...     mollitia, molestiae quas vel sint commodi repudiandae consequuntur
...     voluptatum laborum numquam blanditiis harum quisquam eius sed odit
...     fugiat iusto fuga praesentium optio, eaque rerum! Provident similique
...     accusantium nemo autem. Veritatis obcaecati tenetur iure eius earum
...     ut molestias architecto voluptate aliquam nihil, eveniet aliquid
...     culpa officia aut! Impedit sit sunt quaerat, odit, tenetur error,
...     harum nesciunt ipsum debitis quas aliquid."""
... )

>>> sample_text.words() 68

>>> sample_string = UpperPrintString("Hello, Pythonista!")
>>> print(sample_string) HELLO, PYTHONISTA!

>>> sample_string
'Hello, Pythonista!'
```

在这些例子中，`WordCountString`和`UpperPrintString`的新实现与旧的实现工作相同。那么，为什么要用`UserString`而不用`str`？到目前为止，没有明显的理由这样做。然而，当您需要修改字符串的创建方式时，`UserString`就派上了用场。

[*Remove ads*](/account/join/)

### 调整`UserString` 的实例化过程

你可以通过继承`UserString`来编写`LowerString`类。通过更改父类，您将能够在实例初始化器`.__init__()`中定制初始化过程，而无需覆盖实例创建者`.__new__()`。

这是你的新版本`LowerString`以及它在实践中是如何工作的:

>>>

```py
>>> from collections import UserString

>>> class LowerString(UserString):
...     def __init__(self, string):
...         super().__init__(string.lower())
...

>>> sample_string = LowerString("Hello, Pythonista!")
>>> sample_string
'hello, pythonista!'
```

在上面的例子中，通过使用`UserString`而不是`str`作为超类，可以对输入字符串进行转换。这种转换是可能的，因为`UserString`是一个包装类，在其`.data`属性中存储最终字符串，这是真正的不可变对象。

因为`UserString`是围绕`str`类的包装器，所以它提供了一种灵活而直接的方式来创建具有可变行为的定制字符串。通过从`str`继承来提供可变的行为是复杂的，因为类的自然不变性条件。

在下一节中，您将使用`UserString`创建一个类似字符串的类，模拟一个**可变**字符串数据类型。

## 在你的字符串类中模拟突变

作为为什么应该在 Python 工具包中使用`UserString`的最后一个例子，假设您需要一个可变的类似字符串的类。换句话说，您需要一个可以就地修改的类似字符串的类。

与[列表](https://realpython.com/python-lists-tuples/)和[字典](https://realpython.com/python-dicts/)不同，字符串不提供 [`.__setitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__setitem__) 特殊方法，因为它们是不可变的。您的自定义字符串将需要这个方法来允许您使用一个[赋值](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements)语句通过索引更新字符和[片段](https://docs.python.org/3/library/functions.html#slice)。

你的类字符串类也需要改变普通字符串方法的标准行为。为了使这个例子简短，您将只修改 [`.upper()`](https://docs.python.org/3/library/stdtypes.html#str.upper) 和 [`.lower()`](https://docs.python.org/3/library/stdtypes.html#str.lower) 方法。最后，您将提供一个 [`.sort()`](https://realpython.com/python-sort/) 方法来对字符串进行排序。

标准的字符串方法不会改变底层的字符串。它们返回带有所需转换的新字符串对象。在您的自定义字符串中，您需要方法就地执行它们的更改。

为了实现所有这些目标，启动您最喜欢的[代码编辑器](https://realpython.com/python-ides-code-editors-guide/)，创建一个名为`mutable_string.py`的文件，并编写以下代码:

```py
 1# mutable_string.py
 2
 3from collections import UserString
 4
 5class MutableString(UserString):
 6    def __setitem__(self, index, value):
 7        data_as_list = list(self.data)
 8        data_as_list[index] = value
 9        self.data = "".join(data_as_list)
10
11    def __delitem__(self, index):
12        data_as_list = list(self.data)
13        del data_as_list[index]
14        self.data = "".join(data_as_list)
15
16    def upper(self):
17        self.data = self.data.upper()
18
19    def lower(self):
20        self.data = self.data.lower()
21
22    def sort(self, key=None, reverse=False):
23        self.data = "".join(sorted(self.data, key=key, reverse=reverse))
```

下面是这段代码的逐行工作方式:

*   **三号线**从`collections`进口`UserString`。

*   **第 5 行**创建`MutableString`作为`UserString`的子类。

*   **第 6 行**定义`.__setitem__()`。每当您使用索引对序列运行赋值操作时，Python 都会调用这个特殊的方法，就像在`sequence[0] = value`中一样。这个`.__setitem__()`的实现将`.data`转换成一个列表，用`value`替换`index`处的项目，使用`.join()`构建最终的字符串，并将其值赋回`.data`。整个过程模拟了一个就地转化或突变。

*   **第 11 行**定义了 [`.__delitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__delitem__) ，这个特殊的方法允许你使用 [`del`](https://realpython.com/python-keywords/#the-del-keyword) 语句从你的可变字符串中按索引删除字符。它的实现类似于`.__setitem__()`。在第 13 行，您使用`del`从临时列表中删除条目。

*   **16 线**超越`UserString.upper()`并在`.data`上调用`str.upper()`。然后将结果存储回`.data`。同样，最后一个操作模拟了一个原位突变。

*   **第 19 行**使用与`.upper()`相同的技术覆盖`UserString.lower()`。

*   **第 22 行**定义了`.sort()`，它将内置的 [`sorted()`](https://docs.python.org/3/library/functions.html#sorted) 函数与 [`str.join()`](https://realpython.com/python-string-split-concatenate-join/) 方法结合起来，创建原始字符串的排序版本。注意，这个方法与`list.sort()`和内置的`sorted()`函数具有相同的签名。

就是这样！你的可变字符串准备好了！要尝试一下，请回到您的 Python shell 并运行以下代码:

>>>

```py
>>> from mutable_string import MutableString

>>> sample_string = MutableString("ABC def")
>>> sample_string
'ABC def'

>>> sample_string[4] = "x"
>>> sample_string[5] = "y"
>>> sample_string[6] = "z"
>>> sample_string
'ABC xyz'

>>> del sample_string[3]
>>> sample_string
'ABCxyz'

>>> sample_string.upper()
>>> sample_string
'ABCXYZ'

>>> sample_string.lower()
>>> sample_string
'abcxyz'

>>> sample_string.sort(reverse=True)
>>> sample_string
'zyxcba'
```

太好了！您的新可变的类字符串类如预期的那样工作。它允许您就地修改底层字符串，就像处理可变序列一样。注意，这个例子只包含了几个字符串方法。您可以尝试其他方法，继续为您的类提供新的可变性特性。

## 结论

你已经学会了用新的或修改过的行为创建**自定义的类似字符串的类**。您已经通过直接子类化内置的`str`类和从`UserString`继承完成了这一点，这是在 [`collections`](https://realpython.com/python-collections-module/) 模块中可用的一个方便的类。

在用 Python 创建自己的类似字符串的类时，继承`str`和子类化`UserString`都是合适的选择。

**在本教程中，您已经学会了如何:**

*   通过继承**内置的`str`类**来创建类似字符串的类
*   通过从 **`collections`模块**子类化 **`UserString`** 来构建类似字符串的类
*   决定什么时候子类化`str`或者`UserString`来创建你的**定制的类似字符串的类**

现在，您已经准备好编写定制的类似字符串的类，这将允许您充分利用 Python 中这种有价值且常见的数据类型的全部功能。

**示例代码:** [点击这里下载免费的示例代码](https://realpython.com/bonus/inherit-python-str-code/)，您将使用它来创建定制的类似字符串的类。***