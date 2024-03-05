# 自定义 Python 列表:从列表和用户列表继承

> 原文：<https://realpython.com/inherit-python-list/>

在您的 Python 编码冒险中的某个时刻，您可能需要创建**定制的类似列表的类**，具有修改的行为、新的功能，或者两者兼有。在 Python 中要做到这一点，你可以从一个[抽象基类](https://docs.python.org/3/library/collections.abc.html#module-collections.abc)继承，直接继承内置`list`类的子类，或者从`UserList`继承，后者位于`collections`模块中。

**在本教程中，您将学习如何:**

*   通过继承**内置的`list`类**来创建定制的类似列表的类
*   通过从 **`collections`模块**中子类化 **`UserList`** 来构建定制的列表类

您还将编写一些示例，帮助您决定在创建自定义列表类时使用哪个父类`list`或`UserList`。

为了充分利用本教程，您应该熟悉 Python 的内置 [`list`](https://realpython.com/python-lists-tuples/) 类及其标准特性。你还需要知道[面向对象编程](https://realpython.com/python3-object-oriented-programming/)的基础知识，理解[继承](https://realpython.com/inheritance-composition-python/)在 Python 中是如何工作的。

**免费下载:** [点击这里下载源代码](https://realpython.com/bonus/inherit-python-list-code/)，你将使用它来创建定制的列表类。

## 在 Python 中创建类似列表的类

内置的 [`list`](https://realpython.com/python-lists-tuples/) 类是 Python 中的基本数据类型。列表在很多情况下都很有用，并且有大量的实际用例。在某些用例中，Python `list`的标准功能可能不够，您可能需要创建定制的类似列表的类来解决手头的问题。

您通常会发现创建定制的类似列表的类至少有两个原因:

1.  **通过添加新功能来扩展**常规列表
2.  **修改**标准列表的功能

您还可能面临需要扩展*和*来修改列表的标准功能的情况。

根据您的具体需求和技能水平，您可以使用一些策略来创建您自己的定制列表类。您可以:

*   从适当的抽象基类继承，如 [`MutableSequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableSequence)
*   直接从 Python 内置的 [`list`](https://docs.python.org/3/library/stdtypes.html#list) 类继承
*   子类 [`UserList`](https://docs.python.org/3/library/collections.html#collections.UserList) 来自 [`collections`](https://realpython.com/python-collections-module/)

**注:**在[面向对象编程](https://realpython.com/python3-object-oriented-programming/)中，通常的做法是将动词**继承**和**子类**互换使用。

当您选择要使用的适当策略时，有一些注意事项。请继续阅读，了解更多详情。

[*Remove ads*](/account/join/)

## 从抽象基类构建类似列表的类

您可以通过继承适当的**抽象基类(ABC)** ，像 [`MutableSequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableSequence) ，来创建自己的列表类。除了 [`.__getitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__getitem__) 、 [`.__setitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__setitem__) 、 [`.__delitem__`](https://docs.python.org/3/reference/datamodel.html#object.__delitem__) 、 [`.__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 和`.insert()`之外，这个 ABC 提供了大多数`list`方法的通用实现。因此，当从这个类继承时，您必须自己实现这些方法。

为所有这些[特殊方法](https://docs.python.org/3/glossary.html#term-special-method)编写自己的实现是一项相当大的工作量。这很容易出错，并且需要 Python 及其[数据模型](https://docs.python.org/3/reference/datamodel.html)的高深知识。这也可能意味着性能问题，因为您将使用纯 Python 编写方法。

此外，假设您需要定制任何其他标准列表方法的功能，如 [`.append()`](https://realpython.com/python-append/) 或`.insert()`。在这种情况下，您必须覆盖默认实现，并提供一个满足您需求的合适实现。

这种创建类似列表的类的策略的主要优点是，如果您在自定义实现中遗漏了任何必需的方法，父 ABC 类会提醒您。

一般来说，只有当您需要一个与内置的`list`类完全不同的列表类时，您才应该采用这种策略。

在本教程中，您将通过继承内置的`list`类和标准库`collections`模块中的`UserList`类来创建类似列表的类。这些策略似乎是最快捷和最实用的。

## 从 Python 内置的`list`类继承而来

很长一段时间，直接继承用 [C](https://realpython.com/c-for-python-programmers/) 实现的 Python 类型是不可能的。Python 2.2 修复了这个问题。现在你可以[子类内置类型](https://docs.python.org/3/whatsnew/2.2.html#peps-252-and-253-type-and-class-changes)，包括`list`。这一变化给子类带来了一些技术优势，因为现在它们:

*   将在每个需要原始内置类型的地方工作
*   可以定义新的[实例](https://realpython.com/instance-class-and-static-methods-demystified/#instance-methods)、[静态](https://realpython.com/instance-class-and-static-methods-demystified/#static-methods)和[类](https://realpython.com/instance-class-and-static-methods-demystified/#class-methods)方法
*   可以将它们的[实例属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)存储在一个 [`.__slots__`](https://docs.python.org/3/reference/datamodel.html#object.__slots__) 类属性中，这实质上取代了 [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 属性

这个列表中的第一项可能是对需要 Python 内置类的 C 代码的要求。第二项允许您在标准列表行为的基础上添加新功能。最后，第三项将使您能够将子类的属性限制为那些在`.__slots__`中预定义的属性。

要开始创建定制的类似列表的类，假设您需要一个列表，它会自动将所有项目存储为字符串。假设您的定制列表将把[数字](https://realpython.com/python-numbers/)仅仅存储为字符串，您可以创建下面的`list`子类:

```py
# string_list.py

class StringList(list):
    def __init__(self, iterable):
        super().__init__(str(item) for item in iterable)

    def __setitem__(self, index, item):
        super().__setitem__(index, str(item))

    def insert(self, index, item):
        super().insert(index, str(item))

    def append(self, item):
        super().append(str(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(str(item) for item in other)
```

您的`StringList`类直接继承了`list`，这意味着它将继承标准 Python `list`的所有功能。因为您希望列表将项存储为字符串，所以需要修改所有在基础列表中添加或修改项的方法。这些方法包括以下内容:

*   **`.__init__`** 初始化所有类的新实例。
*   **`.__setitem__()`** 允许您使用项目的索引为现有项目分配一个新值，就像在`a_list[index] = item`中一样。
*   **`.insert()`** 允许你使用项目的索引在底层列表的给定位置插入一个新项目。
*   **`.append()`** 在底层列表的末尾增加一个新的单项。
*   **`.extend()`** 将一系列项目添加到列表的末尾。

您的`StringList`类从`list`继承的其他方法工作得很好，因为它们不添加或更新您的自定义列表中的项目。

**注意:**如果你想让你的`StringList`类支持**和加号运算符(`+`)的串联**，那么你还需要实现其他特殊的方法，比如 [`.__add__()`](https://docs.python.org/3/reference/datamodel.html#object.__add__) 、 [`.__radd__()`](https://docs.python.org/3/reference/datamodel.html#object.__radd__) 和 [`.__iadd__()`](https://docs.python.org/3/reference/datamodel.html#object.__iadd__) 。

要在代码中使用`StringList`,您可以这样做:

>>>

```py
>>> from string_list import StringList

>>> data = StringList([1, 2, 2, 4, 5])
>>> data
['1', '2', '2', '4', '5']

>>> data.append(6)
>>> data
['1', '2', '2', '4', '5', '6']

>>> data.insert(0, 0)
>>> data
['0', '1', '2', '2', '4', '5', '6']

>>> data.extend([7, 8, 9])
>>> data
['0', '1', '2', '2', '4', '5', '6', '7', '8', '9']

>>> data[3] = 3
>>> data
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
```

你的班级像预期的那样工作。它将所有输入值动态转换成字符串。那很酷，不是吗？当你创建一个新的`StringList`实例时，类的[初始化器](https://realpython.com/python-class-constructor/#object-initialization-with-__init__)会负责转换。

当您向类的实例追加、插入、扩展或分配新值时，支持每个操作的方法将负责字符串转换过程。这样，您的列表将始终将其项目存储为字符串对象。

[*Remove ads*](/account/join/)

## 从`collections` 子类化`UserList`

另一种创建定制列表类的方法是使用`collections`模块中的 [`UserList`](https://docs.python.org/3/library/collections.html#collections.UserList) 类。这个类是内置`list`类型的包装器。它是为在不可能直接从内置的`list`类继承时创建类似列表的对象而设计的。

尽管对这个类的需求已经被直接子类化内置的`list`类的可能性部分取代，但是为了方便和向后兼容，`UserList`仍然可以在[标准库](https://docs.python.org/3/library/index.html)中使用。

`UserList`的显著特点是它允许您访问它的`.data`属性，这可以方便您创建自定义列表，因为您不需要一直使用 [`super()`](https://realpython.com/python-super/) 。`.data`属性保存一个常规的 Python `list`，默认情况下为空。

下面是你如何通过继承`UserList`来重新实现你的`StringList`类:

```py
# string_list.py

from collections import UserList 
class StringList(UserList):
    def __init__(self, iterable):
        super().__init__(str(item) for item in iterable)

    def __setitem__(self, index, item):
 self.data[index] = str(item) 
    def insert(self, index, item):
 self.data.insert(index, str(item)) 
    def append(self, item):
 self.data.append(str(item)) 
    def extend(self, other):
        if isinstance(other, type(self)):
 self.data.extend(other)        else:
 self.data.extend(str(item) for item in other)
```

在这个例子中，访问`.data`属性允许您通过使用[委托](https://en.wikipedia.org/wiki/Delegation_pattern)以更直接的方式对类进行编码，这意味着`.data`中的列表负责处理所有请求。

现在你几乎不用使用`super()`这样的高级工具了。你只需要在类初始化器中调用这个函数，以防止在进一步的继承场景中出现问题。在其余的方法中，您只需利用保存常规 Python 列表的`.data`。使用列表是你可能已经掌握的技能。

**注意:**在上面的例子中，你可以重用[上一节](#inheriting-from-pythons-built-in-list-class)中`StringList`的内部实现，但是把父类从`list`改为`UserList`。您的代码将同样工作。然而，使用`.data`可以简化列表类的编码过程。

这个新版本和你的第一个版本`StringList`一样。继续运行以下代码进行试验:

>>>

```py
>>> from string_list import StringList

>>> data = StringList([1, 2, 2, 4, 5])
>>> data
['1', '2', '2', '4', '5']

>>> data.append(6)
>>> data
['1', '2', '2', '4', '5', '6']

>>> data.insert(0, 0)
>>> data
['0', '1', '2', '2', '4', '5', '6']

>>> data.extend([7, 8, 9])
>>> data
['0', '1', '2', '2', '4', '5', '6', '7', '8', '9']

>>> data[3] = 3
>>> data
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
```

正如您已经了解到的，暴露`.data`是`UserList`最相关的特性。这个属性可以简化你的类，因为你不需要一直使用`super()`。你可以利用`.data`并使用熟悉的`list`界面来处理这个属性。

## 编码列表类:实例

当您需要创建自定义的类似列表的类来添加或修改`list`的标准功能时，您已经知道如何使用`list`和`UserList`。

诚然，当您考虑创建一个类似列表的类时，从`list`继承可能比从`UserList`继承更自然，因为 Python 开发人员知道`list`。他们可能不知道`UserList`的存在。

您还知道这两个类的主要区别在于，当您从`UserList`继承时，您可以访问`.data`属性，这是一个常规列表，您可以通过标准的`list`接口对其进行操作。相比之下，从`list`继承需要关于 Python 数据模型的高级知识，包括像内置的`super()`函数和一些特殊方法这样的工具。

在接下来的部分中，您将使用这两个类编写一些实际的例子。写完这些例子后，当您需要在代码中定义定制的类似列表的类时，您可以更好地选择合适的工具。

### 只接受数字数据的列表

作为创建具有自定义行为的列表类的第一个例子，假设您需要一个只接受数字数据的列表。你的列表应该只存储[整数](https://realpython.com/python-numbers/#integers)、[浮点数](https://realpython.com/python-data-types/#floating-point-numbers)和[复数](https://realpython.com/python-complex-numbers/)。如果您试图存储任何其他数据类型的值，比如字符串，那么您的列表应该引发一个`TypeError`。

下面是一个具有所需功能的`NumberList`类的实现:

```py
# number_list.py

class NumberList(list):
    def __init__(self, iterable):
        super().__init__(self._validate_number(item) for item in iterable)

    def __setitem__(self, index, item):
        super().__setitem__(index, self._validate_number(item))

    def insert(self, index, item):
        super().insert(index, self._validate_number(item))

    def append(self, item):
        super().append(self._validate_number(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(self._validate_number(item) for item in other)

    def _validate_number(self, value):
        if isinstance(value, (int, float, complex)):
            return value
        raise TypeError(
            f"numeric value expected, got {type(value).__name__}"
        )
```

在这个例子中，您的`NumberList`类直接继承自`list`。这意味着您的类与内置的`list`类共享所有核心功能。您可以迭代`NumberList`的实例，使用它们的索引访问和更新它的条目，调用通用的`list`方法，等等。

现在，为了确保每个输入项都是一个数字，您需要在支持添加新项或更新列表中现有项的操作的所有方法中验证每个项。所需的方法与从 Python 内置的`list`类继承而来的[一节中的`StringList`示例相同。](#inheriting-from-pythons-built-in-list-class)

为了验证输入数据，您使用一个叫做`._validate_number()`的助手方法。该方法使用内置的 [`isinstance()`](https://docs.python.org/3/library/functions.html#isinstance) 函数来检查当前输入值是否是`int`、`float`或`complex`的实例，这些是 Python 中表示数值的内置类。

**注意:**在 Python 中检查一个值是否为数字的更通用的方法是使用 [`numbers`](https://docs.python.org/3/library/numbers.html#module-numbers) 模块中的 [`Number`](https://docs.python.org/3/library/numbers.html#numbers.Number) 。这将允许您验证 [`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction) 和 [`Decimal`](https://docs.python.org/3/library/decimal.html#decimal.Decimal) 对象。

如果输入值是数值数据类型的实例，那么您的帮助器函数将返回该值本身。否则，该函数会引发一个`TypeError` [异常](https://realpython.com/python-exceptions/)，并显示一条适当的错误消息。

要使用`NumberList`，请返回到您的[交互式](https://realpython.com/interacting-with-python/)会话并运行以下代码:

>>>

```py
>>> from number_list import NumberList

>>> numbers = NumberList([1.1, 2, 3j])
>>> numbers
[1.1, 2, 3j]

>>> numbers.append("4.2")
Traceback (most recent call last):
    ...
TypeError: numeric value expected, got str

>>> numbers.append(4.2)
>>> numbers
[1.1, 2, 3j, 4.2]

>>> numbers.insert(0, "0")
Traceback (most recent call last):
    ...
TypeError: numeric value expected, got str

>>> numbers.insert(0, 0)
>>> numbers
[0, 1.1, 2, 3j, 4.2]

>>> numbers.extend(["5.3", "6"])
Traceback (most recent call last):
    ...
TypeError: numeric value expected, got str

>>> numbers.extend([5.3, 6])
>>> numbers
[0, 1.1, 2, 3j, 4.2, 5.3, 6]
```

在这些例子中，在`numbers`中添加或修改数据的操作自动验证输入，以确保只接受数值。如果你给`numbers`加一个字符串值，那么你得到一个`TypeError`。

使用`UserList`的`NumberList`的另一个实现可以是这样的:

```py
# number_list.py

from collections import UserList 
class NumberList(UserList):
    def __init__(self, iterable):
        super().__init__(self._validate_number(item) for item in iterable)

    def __setitem__(self, index, item):
 self.data[index] = self._validate_number(item) 
    def insert(self, index, item):
 self.data.insert(index, self._validate_number(item)) 
    def append(self, item):
 self.data.append(self._validate_number(item)) 
    def extend(self, other):
        if isinstance(other, type(self)):
 self.data.extend(other)        else:
 self.data.extend(self._validate_number(item) for item in other) 
    def _validate_number(self, value):
        if isinstance(value, (int, float, complex)):
            return value
        raise TypeError(
            f"numeric value expected, got {type(value).__name__}"
        )
```

在这个新的`NumberList`实现中，您继承了`UserList`。同样，您的类将与常规的`list`共享所有核心功能。

在这个例子中，不是一直使用`super()`来访问父类中的方法和属性，而是直接使用`.data`属性。在某种程度上，与使用`super()`和其他高级工具如特殊方法相比，使用`.data`可以说简化了您的代码。

注意，你只在类初始化器`.__init__()`中使用`super()`。当您在 Python 中处理继承时，这是一个最佳实践。它允许您正确初始化父类中的属性，而不会破坏东西。

[*Remove ads*](/account/join/)

### 具有附加功能的列表

现在假设您需要一个类似列表的类，具有常规 Python `list`的所有标准功能。你的类还应该提供一些从 [JavaScript](https://realpython.com/python-vs-javascript/) 的[数组](https://developer.mozilla.org/es/docs/Web/JavaScript/Reference/Global_Objects/Array)数据类型中借用的额外功能。例如，您需要像下面这样的方法:

*   **`.join()`** 将列表中的所有项目串联成一个字符串。
*   **`.map(action)`** 通过对底层列表中的每个项目应用一个`action()` callable 来产生新的项目。
*   **`.filter(predicate)`** 在调用`predicate()`时会产生所有返回`True`的物品。
*   **`.for_each(func)`** 对底层列表中的每一项都调用`func()`来生成一些[副作用](https://en.wikipedia.org/wiki/Side_effect_(computer_science))。

这里有一个通过子类化`list`实现所有这些新特性的类:

```py
# custom_list.py

class CustomList(list):
    def join(self, separator=" "):
        return separator.join(str(item) for item in self)

    def map(self, action):
        return type(self)(action(item) for item in self)

    def filter(self, predicate):
        return type(self)(item for item in self if predicate(item))

    def for_each(self, func):
        for item in self:
            func(item)
```

`CustomList`中的`.join()`方法以一个分隔符作为参数，并使用它来连接当前列表对象中的项目，该列表对象由`self`表示。为此，您使用带有一个[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)的 [`str.join()`](https://realpython.com/python-string-split-concatenate-join/) 作为参数。这个生成器表达式使用`str()`将每一项转换成一个字符串对象。

`.map()`方法返回一个`CustomList`对象。为了构造这个对象，您使用一个生成器表达式，将`action()`应用到当前对象`self`中的每一项。请注意，该操作可以是任何可调用的操作，它将一个项作为参数并返回一个转换后的项。

`.filter()`方法也返回一个`CustomList`对象。要构建这个对象，您需要使用一个生成器表达式来生成`predicate()`返回`True`的项目。在这种情况下，`predicate()`必须是一个[布尔值函数](https://en.wikipedia.org/wiki/Boolean-valued_function)，它根据应用于输入项的特定条件返回`True`或`False`。

最后，`.for_each()`方法对底层列表中的每一项调用`func()`。这个调用没有返回任何东西，但是触发了一些副作用，您将在下面看到。

要在代码中使用该类，您可以执行如下操作:

>>>

```py
>>> from custom_list import CustomList

>>> words = CustomList(
...     [
...         "Hello,",
...         "Pythonista!",
...         "Welcome",
...         "to",
...         "Real",
...         "Python!"
...     ]
... )

>>> words.join()
'Hello, Pythonista! Welcome to Real Python!'

>>> words.map(str.upper)
['HELLO,', 'PYTHONISTA!', 'WELCOME', 'TO', 'REAL', 'PYTHON!']

>>> words.filter(lambda word: word.startswith("Py"))
['Pythonista!', 'Python!']

>>> words.for_each(print)
Hello,
Pythonista!
Welcome
to
Real
Python!
```

在这些例子中，首先在`words`上调用`.join()`。此方法返回一个唯一的字符串，该字符串是由基础列表中的所有项串联而成的。

对`.map()`的调用返回一个包含大写单词的`CustomList`对象。这种转换是将`str.upper()`应用于`words`中的所有项目的结果。这个方法与内置的 [`map()`](https://realpython.com/python-map-function/) 函数非常相似。主要的区别是，内置的`map()`函数返回一个迭代器，生成转换后的条目[和](https://en.wikipedia.org/wiki/Lazy_evaluation)，而不是返回一个列表。

`.filter()`方法将一个 [`lambda`](https://realpython.com/python-lambda/) 函数作为参数。在示例中，这个`lambda`函数使用 [`str.startswith()`](https://docs.python.org/3/library/stdtypes.html#str.startswith) 来选择以`"Py"`前缀开头的单词。注意，这个方法的工作方式类似于内置的 [`filter()`](https://realpython.com/python-filter-function/) 函数，它返回一个迭代器而不是一个列表。

最后，对`words`上的`.for_each()`的调用将每个单词打印到屏幕上，作为对底层列表中的每个项目调用 [`print()`](https://realpython.com/python-print/) 的副作用。注意，传递给`.for_each()`的函数应该将一个项目作为参数，但它不应该返回任何有成果的值。

你也可以通过继承`UserList`而不是`list`来实现`CustomList`。在这种情况下，您不需要更改内部实现，只需更改基类:

```py
# custom_list.py

from collections import UserList 
class CustomList(UserList):
    def join(self, separator=" "):
        return separator.join(str(item) for item in self)

    def map(self, action):
        return type(self)(action(item) for item in self)

    def filter(self, predicate):
        return type(self)(item for item in self if predicate(item))

    def for_each(self, func):
        for item in self:
            func(item)
```

请注意，在本例中，您只是更改了父类。没必要直接用`.data`。但是，如果你愿意，你可以使用它。这样做的好处是，您可以为阅读您代码的其他开发人员提供更多的上下文:

```py
# custom_list.py

from collections import UserList

class CustomList(UserList):
    def join(self, separator=" "):
 return separator.join(str(item) for item in self.data) 
    def map(self, action):
 return type(self)(action(item) for item in self.data) 
    def filter(self, predicate):
 return type(self)(item for item in self.data if predicate(item)) 
    def for_each(self, func):
 for item in self.data:            func(item)
```

在这个新版本的`CustomList()`中，唯一的变化是你用`self.data`替换了`self`，以表明你正在使用一个`UserList`子类。这一变化使您的代码更加清晰。

[*Remove ads*](/account/join/)

## 考虑性能:`list` vs `UserList`

至此，您已经学会了如何通过继承`list`或`UserList`来创建自己的列表类。您还知道这两个类之间唯一可见的区别是`UserList`公开了`.data`属性，这有助于编码过程。

在这一节中，当决定是使用`list`还是`UserList`来创建定制的类似列表的类时，您将考虑一个重要的方面。那是性能！

为了评估继承自`list`和`UserList`的类之间是否存在性能差异，您将使用`StringList`类。继续创建包含以下代码的新 Python 文件:

```py
# performance.py

from collections import UserList

class StringList_list(list):
    def __init__(self, iterable):
        super().__init__(str(item) for item in iterable)

    def __setitem__(self, index, item):
        super().__setitem__(index, str(item))

    def insert(self, index, item):
        super().insert(index, str(item))

    def append(self, item):
        super().append(str(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(str(item) for item in other)

class StringList_UserList(UserList):
    def __init__(self, iterable):
        super().__init__(str(item) for item in iterable)

    def __setitem__(self, index, item):
        self.data[index] = str(item)

    def insert(self, index, item):
        self.data.insert(index, str(item))

    def append(self, item):
        self.data.append(str(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            self.data.extend(other)
        else:
            self.data.extend(str(item) for item in other)
```

这两个类的工作原理是一样的。然而，它们在内部是不同的。`StringList_list`继承自`list`，其实现基于`super()`。相比之下，`StringList_UserList`继承自`UserList`，它的实现依赖于内部的`.data`属性。

要比较这两个类的性能，应该从计时标准列表操作开始，比如[实例化](https://realpython.com/python-class-constructor/#object-initialization-with-__init__)。然而，在这些例子中，两个初始化器是等价的，所以它们应该执行相同的操作。

测量新功能的执行时间也很有用。比如可以查看`.extend()`的执行时间。继续运行下面的代码:

>>>

```py
>>> import timeit
>>> from performance import StringList_list, StringList_UserList
>>> init_data = range(10000)

>>> extended_list = StringList_list(init_data)
>>> list_extend = min(
...     timeit.repeat(
...         stmt="extended_list.extend(init_data)",
...         number=5,
...         repeat=2,
...         globals=globals(),
...     )
... ) * 1e6

>>> extended_user_list = StringList_UserList(init_data)
>>> user_list_extend = min(
...     timeit.repeat(
...         stmt="extended_user_list.extend(init_data)",
...         number=5,
...         repeat=2,
...         globals=globals(),
...     )
... ) * 1e6

>>> f"StringList_list().extend() time: {list_extend:.2f} μs"
'StringList_list().extend() time: 4632.08 μs'

>>> f"StringList_UserList().extend() time: {user_list_extend:.2f} μs"
'StringList_UserList().extend() time: 4612.62 μs'
```

在这个性能测试中，您使用 [`timeit`](https://docs.python.org/3/library/timeit.html?highlight=timeit#module-timeit) 模块和 [`min()`](https://realpython.com/python-min-and-max/) 函数来测量一段代码的执行时间。目标代码包括使用一些样本数据在`StringList_list`和`StringList_UserList`的实例上对`.extend()`的调用。

在这个例子中，基于`list`的类和基于`UserList`的类之间的性能差异几乎不存在。

通常，当你创建一个定制的类似列表的类时，你会期望`list`的子类比`UserList`的子类执行得更好。为什么？因为`list`是用 C 写的，并且针对性能进行了优化，而`UserList`是用纯 Python 写的包装器类。

然而，在上面的例子中，看起来这个假设并不完全正确。因此，要决定哪个超类最适合您的特定用例，请确保运行性能测试。

撇开性能不谈，继承`list`可以说是 Python 中的自然方式，主要是因为`list`作为内置类直接供 Python 开发人员使用。此外，大多数 Python 开发人员将熟悉列表及其标准特性，这将允许他们更快地编写类似列表的类。

相比之下，`UserList`类位于`collections`模块中，这意味着如果想在代码中使用它，就必须导入它。另外，并不是所有的 Python 开发者都知道`UserList`的存在。然而，`UserList`仍然是一个有用的工具，因为它可以方便地访问`.data`属性，这有助于创建定制的类似列表的类。

## 结论

现在你已经学会了如何创建**定制列表类的类**和修改后的新行为。为此，您已经直接子类化了内置的`list`类。作为一种选择，你也继承了`UserList`类，它在 [`collections`](https://realpython.com/python-collections-module/) 模块中可用。

从`list`继承和子类化`UserList`都是解决在 Python 中创建自己的列表类问题的合适策略。

**在本教程中，您学习了如何:**

*   通过继承**内置`list`类**来创建类似列表的类
*   通过从 **`collections`模块**中子类化 **`UserList`** 来构建类似列表的类

现在，您可以更好地创建自己的自定义列表，从而充分利用 Python 中这种有用且常见的数据类型的全部功能。

**免费下载:** [点击这里下载源代码](https://realpython.com/bonus/inherit-python-list-code/)，你将使用它来创建定制的列表类。****