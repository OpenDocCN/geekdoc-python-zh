# 自定义 Python 字典:从 dict 和 UserDict 继承

> 原文：<https://realpython.com/inherit-python-dict/>

创建类似字典的类可能是您 Python 职业生涯中的一项需求。具体来说，您可能对使用修改的行为、新功能或两者来制作自定义词典感兴趣。在 Python 中，你可以通过继承一个[抽象基类](https://docs.python.org/3/library/collections.abc.html#module-collections.abc)，直接子类化内置的`dict`类，或者继承 [`UserDict`](https://docs.python.org/3/library/collections.html#collections.UserDict) 来做到这一点。

**在本教程中，您将学习如何:**

*   通过继承**内置的`dict`类**来创建类似字典的类
*   识别从`dict`继承时可能发生的**常见陷阱**
*   通过从`collections`模块中用**子类化`UserDict`** 来构建类似字典的类

此外，您将编写几个例子来帮助您理解使用`dict`和`UserDict`创建自定义字典类的优缺点。

为了充分利用本教程，您应该熟悉 Python 的内置 [`dict`](https://realpython.com/python-dicts/) 类及其标准功能和特性。你还需要知道[面向对象编程](https://realpython.com/python3-object-oriented-programming/)的基础知识，理解[继承](https://realpython.com/inheritance-composition-python/)在 Python 中是如何工作的。

**立即加入:** ，你将永远不会错过另一个 Python 教程、课程更新或帖子。

## 在 Python 中创建类似字典的类

内置的 [`dict`](https://realpython.com/python-dicts/) 类提供了一个有价值且通用的集合数据类型，Python **字典**。字典无处不在，包括您的代码和 Python 本身的代码。

有时，Python 字典的标准功能对于某些用例来说是不够的。在这些情况下，您可能需要创建一个自定义的类似字典的类。换句话说，您需要一个行为类似于常规字典的类，但是具有修改过的或新的功能。

您通常会发现创建定制的类似字典的类至少有两个原因:

1.  通过添加新功能扩展常规词典
2.  **修改**标准字典的功能

注意，您还可能面临需要扩展*和*来修改字典的标准功能的情况。

根据您的特定需求和技能水平，您可以从一些创建自定义词典的策略中进行选择。您可以:

*   从适当的抽象基类继承，如 [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)
*   直接从 Python 内置的 [`dict`](https://docs.python.org/3/library/stdtypes.html#dict) 类继承
*   子类 [`UserDict`](https://docs.python.org/3/library/collections.html#collections.UserDict) 来自 [`collections`](https://realpython.com/python-collections-module/)

当您选择要实施的适当策略时，有几个关键的考虑因素。请继续阅读，了解更多详情。

[*Remove ads*](/account/join/)

## 从抽象基类构建类似字典的类

这种创建类似字典的类的策略要求您从一个**抽象基类(ABC)** 继承，就像 [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping) 。该类提供了除 [`.__getitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__getitem__) 、 [`.__setitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__setitem__) 、 [`.__delitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__delitem__) 、 [`.__iter__()`](https://docs.python.org/3/reference/datamodel.html#object.__iter__) 和 [`.__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 之外的所有字典方法的具体泛型实现，这些方法需要您自己实现。

此外，假设您需要定制任何其他标准字典方法的功能。在这种情况下，您必须覆盖手头的方法，并提供一个合适的实现来满足您的需求。

这个过程意味着大量的工作。它也容易出错，并且需要 Python 及其[数据模型](https://docs.python.org/3/reference/datamodel.html)的高级知识。这也可能意味着性能问题，因为您将使用纯 Python 编写该类。

这种策略的主要优点是，如果您在自定义实现中遗漏了任何方法，父 ABC 都会提醒您。

出于这些原因，只有当您需要一个与内置字典完全不同的类似字典的类时，您才应该采用这种策略。

在本教程中，您将专注于通过继承内置的`dict`类和`UserDict`类来创建类似字典的类，这似乎是最快和最实用的策略。

## 从 Python 内置的`dict`类继承而来

很长一段时间，不可能在 [C](https://realpython.com/c-for-python-programmers/) 中实现 Python 类型的子类化。Python 2.2 修复了这个问题。现在你可以[直接子类化内置类型](https://docs.python.org/3/whatsnew/2.2.html#peps-252-and-253-type-and-class-changes)，包括`dict`。这一变化为子类带来了几个技术优势，因为现在它们:

*   将在每个需要原始内置类型的地方工作
*   可以定义新的[实例](https://realpython.com/instance-class-and-static-methods-demystified/#instance-methods)、[静态](https://realpython.com/instance-class-and-static-methods-demystified/#static-methods)和[类](https://realpython.com/instance-class-and-static-methods-demystified/#class-methods)方法
*   可以将它们的[实例属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)存储在一个 [`.__slots__`](https://docs.python.org/3/reference/datamodel.html#object.__slots__) 类属性中，这实质上取代了 [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 属性

这个列表中的第一项可能是对需要 Python 内置类的 C 代码的要求。第二项允许您在标准字典行为的基础上添加新功能。最后，第三项将使您能够将子类的属性限制为那些在`.__slots__`中预定义的属性。

尽管对内置类型进行子类化有几个优点，但它也有一些缺点。在字典的具体例子中，你会发现一些恼人的陷阱。例如，假设您想要创建一个类似字典的类，该类自动将其所有键存储为字符串，其中所有字母(如果存在的话)都是大写的。

为此，您可以创建一个覆盖`.__setitem__()`方法的`dict`的子类:

>>>

```py
>>> class UpperCaseDict(dict):
...     def __setitem__(self, key, value):
...         key = key.upper()
...         super().__setitem__(key, value)
...

>>> numbers = UpperCaseDict()
>>> numbers["one"] = 1
>>> numbers["two"] = 2
>>> numbers["three"] = 3

>>> numbers
{'ONE': 1, 'TWO': 2, 'THREE': 3}
```

酷！你的自定义词典似乎很好用。然而，在这个类中有一些隐藏的问题。如果您试图使用一些初始化数据创建一个`UpperCaseDict`的实例，那么您将会得到一个令人惊讶的错误行为:

>>>

```py
>>> numbers = UpperCaseDict({"one": 1, "two": 2, "three": 3})
>>> numbers
{'one': 1, 'two': 2, 'three': 3}
```

刚刚发生了什么？当你调用类的[构造函数](https://realpython.com/python-class-constructor/)时，为什么你的字典不把键转换成大写字母？看起来类的初始化器`.__init__()`没有隐式调用`.__setitem__()`来创建字典。因此，大写转换永远不会运行。

不幸的是，这个问题影响了其他的字典方法，像 [`.update()`](https://docs.python.org/3/library/stdtypes.html#dict.update) 和 [`.setdefault()`](https://docs.python.org/3/library/stdtypes.html#dict.setdefault) ，例如:

>>>

```py
>>> numbers = UpperCaseDict()
>>> numbers["one"] = 1
>>> numbers["two"] = 2
>>> numbers["three"] = 3

>>> numbers
{'ONE': 1, 'TWO': 2, 'THREE': 3}

>>> numbers.update({"four": 4})
>>> numbers
{'ONE': 1, 'TWO': 2, 'THREE': 3, 'four': 4}

>>> numbers.setdefault("five", 5)
5
>>> numbers
{'ONE': 1, 'TWO': 2, 'THREE': 3, 'four': 4, 'five': 5}
```

同样，在这些例子中，您的大写字母功能不能很好地工作。要解决这个问题，您必须提供所有受影响方法的自定义实现。例如，要解决初始化问题，您可以编写一个类似下面的`.__init__()`方法:

```py
# upper_dict.py

class UpperCaseDict(dict):
 def __init__(self, mapping=None, /, **kwargs):        if mapping is not None:
            mapping = {
                str(key).upper(): value for key, value in mapping.items()
            }
        else:
            mapping = {}
        if kwargs:
            mapping.update(
                {str(key).upper(): value for key, value in kwargs.items()}
            )
        super().__init__(mapping)

    def __setitem__(self, key, value):
        key = key.upper()
        super().__setitem__(key, value)
```

这里，`.__init__()`将密钥转换成大写字母，然后用结果数据初始化当前实例。

有了这个更新，自定义词典的初始化过程应该可以正常工作了。继续运行下面的代码来尝试一下:

>>>

```py
>>> from upper_dict import UpperCaseDict

>>> numbers = UpperCaseDict({"one": 1, "two": 2, "three": 3})
>>> numbers
{'ONE': 1, 'TWO': 2, 'THREE': 3}

>>> numbers.update({"four": 4})
>>> numbers
{'ONE': 1, 'TWO': 2, 'THREE': 3, 'four': 4}
```

提供您自己的`.__init__()`方法修复了初始化问题。然而，像`.update()`这样的其他方法继续不正确地工作，因为你可以从`"four"`键不是大写的得出结论。

为什么子类会有这样的行为？考虑到[开闭原则](https://en.wikipedia.org/wiki/Open–closed_principle)，内置类型[被设计和实现](https://www.youtube.com/watch?v=heJuQWNdwJI)。因此，它们可以扩展，但不能修改。允许修改这些类的核心特性可能会破坏它们的[不变量](https://en.wikipedia.org/wiki/Invariant_(mathematics)#Invariants_in_computer_science)。因此，Python 核心开发人员决定保护它们不被修改。

这就是为什么子类化内置的`dict`类会有点棘手，耗费人力，并且容易出错。幸运的是，你还有其他选择。来自`collections`模块的`UserDict`类就是其中之一。

[*Remove ads*](/account/join/)

## 从`collections` 子类化`UserDict`

从 Python 1.6 开始，该语言提供了作为标准库一部分的`UserDict`。这个类最初位于一个以类本身命名的模块中。在 Python 3 中，`UserDict`被移到了 [`collections`](https://realpython.com/python-collections-module/) 模块，这是一个更直观的地方，基于类的主要目的。

`UserDict`是在不可能直接从 Python 的`dict`继承的时候创建回来的。尽管对这个类的需求已经被直接子类化内置的`dict`类的可能性部分取代，但是为了方便和向后兼容，`UserDict`仍然可以在标准库中使用。

`UserDict`是一个常规`dict`对象的方便包装器。该类提供了与内置的`dict`数据类型相同的行为，并提供了通过 [`.data`](https://docs.python.org/3/library/collections.html#collections.UserDict.data) 实例属性访问底层字典的附加功能。这个特性有助于创建定制的类似字典的类，您将在本教程的后面了解到这一点。

`UserDict`是专门为*子类化*而不是直接实例化设计的，这意味着该类的主要目的是允许你通过[继承](https://realpython.com/inheritance-composition-python/)来创建类似字典的类。

还有其他隐藏的差异。要发现它们，回到最初的`UpperCaseDict`实现，并像下面的代码那样更新它:

>>>

```py
>>> from collections import UserDict 
>>> class UpperCaseDict(UserDict): ...     def __setitem__(self, key, value):
...         key = key.upper()
...         super().__setitem__(key, value)
...
```

这一次，不是从`dict`继承，而是从`UserDict`继承，它是从`collections`模块导入的。这个变化会对你的`UpperCaseDict`类的行为产生怎样的影响？看看下面的例子:

>>>

```py
>>> numbers = UpperCaseDict({"one": 1, "two": 2})

>>> numbers["three"] = 3
>>> numbers.update({"four": 4})
>>> numbers.setdefault("five", 5)
5

>>> numbers
{'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5}
```

现在`UpperCaseDict`一直正常工作。您不需要提供`.__init__()`、`.update()`或`.setdefault()`的定制实现。这个课程很有用！这是因为在`UserDict`中，所有更新现有密钥或添加新密钥的方法都始终依赖于您的`.__setitem__()`版本。

正如您之前了解到的，`UserDict`和`dict`之间最显著的区别是`.data`属性，它保存包装的字典。直接使用`.data`可以让你的代码更加简单，因为你不需要一直调用 [`super()`](https://realpython.com/python-super/) 来提供想要的功能。你只需访问`.data`就可以像使用任何普通词典一样使用它。

## 编码类似字典的类:实例

你已经知道`dict`的子类不会从`.update()`和`.__init__()`这样的方法中调用`.__setitem__()`。这个事实使得`dict`的子类的行为不同于典型的使用`.__setitem__()`方法的 Python 类。

要解决这个问题，您可以从`UserDict`继承，它从所有设置或更新底层字典中的值的操作中调用`.__setitem__()`。因为这个特性，`UserDict`可以让你的代码更安全、更紧凑。

诚然，当您考虑创建一个类似字典的类时，从`dict`继承比从`UserDict`继承更自然。这是因为所有的 Python 开发者都知道`dict`，但并不是所有的 Python 开发者都知道`UserDict`的存在。

从`dict`继承通常意味着某些问题可以通过使用`UserDict`来解决。然而，这些问题并不总是相关的。它们的相关性很大程度上取决于您希望如何定制字典的功能。

底线是`UserDict`并不总是正确的解决方案。一般来说，如果你想在不影响其核心结构的情况下扩展标准字典，那么从`dict`继承完全没问题。另一方面，如果你想通过覆盖它的特殊方法来改变核心字典行为，那么`UserDict`是你最好的选择。

无论如何，请记住`dict`是用 C 语言编写的，并且对性能进行了高度优化。与此同时，`UserDict`是用纯 Python 编写的，这在性能方面有很大的限制。

在决定是继承`dict`还是`UserDict`的时候，你要考虑几个因素。这些因素包括但不限于以下内容:

*   工作量
*   错误和缺陷的风险
*   易于使用和编码
*   表演

在下一节中，您将通过编写一些实际例子来体验列表中的前三个因素。稍后，在关于[性能](#considering-performance)的章节中，您将了解性能含义。

[*Remove ads*](/account/join/)

### 接受英式和美式拼法的字典

作为第一个例子，假设您需要一个存储美式英语关键字并允许美式英语或英式英语关键字查找的字典。要编写这个字典，您需要修改至少两个[特殊方法](https://docs.python.org/3/glossary.html#term-special-method)、`.__setitem__()`和`.__getitem__()`。

`.__setitem__()`方法将允许你总是用美式英语存储密钥。`.__getitem__()`方法将使得检索与给定键相关联的值成为可能，不管它是用美式英语还是英式英语拼写的。

因为您需要修改`dict`类的核心行为，所以使用`UserDict`来编写这个类是一个更好的选择。有了`UserDict`，你将不必提供`.__init__()`、`.update()`等等的定制实现。

当你子类化`UserDict`时，你有两种主要的方法来编码你的类。您可以依靠`.data`属性，这可能有助于编码，或者您可以依靠`super()`和特殊方法。

下面是依赖于`.data`的代码:

```py
# spelling_dict.py

from collections import UserDict

UK_TO_US = {"colour": "color", "flavour": "flavor", "behaviour": "behavior"}

class EnglishSpelledDict(UserDict):
    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            pass
        try:
            return self.data[UK_TO_US[key]]
        except KeyError:
            pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        try:
            key = UK_TO_US[key]
        except KeyError:
            pass
        self.data[key] = value
```

在本例中，首先定义一个常量`UK_TO_US`，其中包含作为键的英国单词和作为值的匹配美国单词。

然后你定义`EnglishSpelledDict`，继承自`UserDict`。`.__getitem__()`方法寻找当前的键。如果该键存在，则该方法返回它。如果该键不存在，那么该方法检查该键是否是用英式英语拼写的。如果是这样的话，那么这个键就被翻译成美式英语，并从底层字典中检索。

`.__setitem__()`方法试图在`UK_TO_US`字典中找到输入键。如果输入键存在于`UK_TO_US`，那么它会被翻译成美式英语。最后，该方法将输入`value`分配给目标`key`。

下面是您的`EnglishSpelledDict`类在实践中是如何工作的:

>>>

```py
>>> from spelling_dict import EnglishSpelledDict

>>> likes = EnglishSpelledDict({"color": "blue", "flavour": "vanilla"})

>>> likes
{'color': 'blue', 'flavor': 'vanilla'}

>>> likes["flavour"]
vanilla
>>> likes["flavor"]
vanilla

>>> likes["behaviour"] = "polite"
>>> likes
{'color': 'blue', 'flavor': 'vanilla', 'behavior': 'polite'}

>>> likes.get("colour")
'blue'
>>> likes.get("color")
'blue'

>>> likes.update({"behaviour": "gentle"})
>>> likes
{'color': 'blue', 'flavor': 'vanilla', 'behavior': 'gentle'}
```

通过对`UserDict`进行子类化，你可以避免编写大量代码。例如，您不必提供像`.get()`、`.update()`或`.setdefault()`这样的方法，因为它们的默认实现将自动依赖于您的`.__getitem__()`和`.__setitem__()`方法。

如果你要写的代码少了，那么你要做的工作也就少了。更重要的是，你会更安全，因为更少的代码通常意味着更低的错误风险。

这种实现的主要缺点是，如果有一天你决定更新`EnglishSpelledDict`并让它从`dict`继承，那么你将不得不重写大部分代码来抑制`.data`的使用。

下面的例子展示了如何使用`super()`和一些特殊的方法来提供与之前相同的功能。这一次，您的自定义字典与`dict`完全兼容，因此您可以随时更改父类:

```py
# spelling_dict.py

from collections import UserDict

UK_TO_US = {"colour": "color", "flavour": "flavor", "behaviour": "behavior"}

class EnglishSpelledDict(UserDict):
    def __getitem__(self, key):
        try:
 return super().__getitem__(key)        except KeyError:
            pass
        try:
 return super().__getitem__(UK_TO_US[key])        except KeyError:
            pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        try:
            key = UK_TO_US[key]
        except KeyError:
            pass
 super().__setitem__(key, value)
```

这个实现看起来与最初的略有不同，但工作原理相同。编码也可能更难，因为你不再使用`.data`了。相反，你使用的是`super()`、`.__getitem__()`和`.__setitem__()`。这段代码需要对 Python 的数据模型有一定的了解，这是一个复杂而高级的话题。

这个新实现的主要优点是你的类现在与`dict`兼容，所以如果你需要的话，你可以在任何时候改变这个超类。

**注意:**记住，如果您直接从`dict`继承，那么您需要重新实现`.__init__()`和其他方法，以便它们在将键添加到字典中时也将键翻译成美式拼写。

通过子类化`UserDict`来扩展标准字典功能通常比子类化`dict`更方便。主要原因是内置的`dict`有一些实现快捷方式和优化，最终迫使你覆盖那些如果你使用`UserDict`作为父类就可以继承的方法。

[*Remove ads*](/account/join/)

### 通过值访问键的字典

自定义词典的另一个常见需求是提供标准行为之外的附加功能。例如，假设您想要创建一个类似字典的类，该类提供检索映射到给定目标值的键的方法。

您需要一个方法来检索映射到目标值的第一个键。您还希望有一个方法返回那些映射到相等值的键的迭代器。

下面是这个自定义词典的一个可能的实现:

```py
# value_dict.py

class ValueDict(dict):
    def key_of(self, value):
        for k, v in self.items():
            if v == value:
                return k
        raise ValueError(value)

    def keys_of(self, value):
        for k, v in self.items():
            if v == value:
                yield k
```

这一次，不是从`UserDict`继承，而是从`dict`继承。为什么？在本例中，您添加的功能不会改变字典的核心特性。因此，从`dict`那里继承更合适。就性能而言，它也更加高效，您将在本教程的后面部分看到这一点。

`.key_of()`方法迭代底层字典中的键值对。[条件语句](https://realpython.com/python-conditional-statements/)检查与目标值匹配的值。`if`代码块返回第一个匹配值的关键字。如果缺少目标键，那么该方法会引发一个`ValueError`。

作为一个按需生成键的[生成器](https://realpython.com/introduction-to-python-generators/)方法，`.keys_of()`将只生成那些值与方法调用中作为参数提供的`value`相匹配的键。

下面是这本词典在实践中的用法:

>>>

```py
>>> from value_dict import ValueDict

>>> inventory = ValueDict()
>>> inventory["apple"] = 2
>>> inventory["banana"] = 3
>>> inventory.update({"orange": 2})

>>> inventory
{'apple': 2, 'banana': 3, 'orange': 2}

>>> inventory.key_of(2)
'apple'
>>> inventory.key_of(3)
'banana'

>>> list(inventory.keys_of(2))
['apple', 'orange']
```

酷！你的字典像预期的那样工作。它从 Python 的`dict`继承了核心字典的特性，并在此基础上实现了新的功能。

一般来说，你应该使用`UserDict`来创建一个类似字典的类，它的行为类似于内置的`dict`类，但是定制了它的一些核心功能，大部分是像`.__setitem__()`和`.__getitem__()`这样的特殊方法。

另一方面，如果您只需要一个类似字典的类，具有不影响或修改核心`dict`行为的扩展功能，那么您最好直接从 Python 中的`dict`继承。这种练习会更快、更自然、更有效。

### 具有附加功能的词典

作为如何实现具有附加特性的自定义字典的最后一个示例，假设您想要创建一个提供以下方法的字典:

| 方法 | 描述 |
| --- | --- |
| `.apply(action)` | 将可调用的`action`作为参数，并将其应用于基础字典中的所有值 |
| `.remove(key)` | 从底层字典中删除给定的`key` |
| `.is_empty()` | 根据字典是否为空返回`True`或`False` |

实现这三个方法，不需要修改内置`dict`类的核心行为。因此，子类化`dict`而不是`UserDict`似乎是一条可行之路。

下面是在`dict`之上实现所需方法的代码:

```py
# extended_dict.py

class ExtendedDict(dict):
    def apply(self, action):
        for key, value in self.items():
            self[key] = action(value)

    def remove(self, key):
        del self[key]

    def is_empty(self):
        return len(self) == 0
```

在这个例子中，`.apply()`将一个 callable 作为参数，并将其应用于底层字典中的每个值。然后将转换后的值重新分配给原始键。`.remove()`方法使用 [`del`](https://docs.python.org/3/reference/simple_stmts.html?highlight=del#the-del-statement) 语句从字典中删除目标键。最后，`.is_empty()`使用内置的 [`len()`](https://realpython.com/len-python-function/) 函数来查找字典是否为空。

下面是`ExtendedDict`的工作原理:

>>>

```py
>>> from extended_dict import ExtendedDict

>>> numbers = ExtendedDict({"one": 1, "two": 2, "three": 3})
>>> numbers
{'one': 1, 'two': 2, 'three': 3}

>>> numbers.apply(lambda x: x**2)
>>> numbers
{'one': 1, 'two': 4, 'three': 9}

>>> numbers.remove("two")
>>> numbers
{'one': 1, 'three': 9}

>>> numbers.is_empty()
False
```

在这些例子中，首先使用一个常规字典作为参数创建一个`ExtendedDict`的实例。然后在扩展字典上调用`.apply()`。该方法将一个 [`lambda`](https://realpython.com/python-lambda/) 函数作为参数，并将其应用于字典中的每个值，将目标值转换为它的平方。

然后，`.remove()`将一个现有的键作为参数，并从字典中删除相应的键-值对。最后，`.is_empty()`返回`False`，因为`numbers`不为空。如果底层字典为空，它将返回`True`。

[*Remove ads*](/account/join/)

## 考虑性能

从`UserDict`继承可能意味着性能成本，因为这个类是用纯 Python 编写的。另一方面，内置的`dict`类是用 C 编写的，并针对性能进行了高度优化。所以，如果你需要在性能关键的代码中使用自定义字典，那么确保对你的代码进行[计时](https://realpython.com/python-timer/)以发现潜在的性能问题。

为了检查当你从`UserDict`而不是`dict`继承时是否会出现性能问题，回到你的`ExtendedDict`类，将其代码复制到两个不同的类中，一个从`dict`继承，另一个从`UserDict`继承。

您的类应该是这样的:

```py
# extended_dicts.py

from collections import UserDict

class ExtendedDict_dict(dict):
    def apply(self, action):
        for key, value in self.items():
            self[key] = action(value)

    def remove(self, key):
        del self[key]

    def is_empty(self):
        return len(self) == 0

class ExtendedDict_UserDict(UserDict):
    def apply(self, action):
        for key, value in self.items():
            self[key] = action(value)

    def remove(self, key):
        del self[key]

    def is_empty(self):
        return len(self) == 0
```

这两个类的唯一区别是`ExtendedDict_dict`子类`dict`，而`ExtendedDict_UserDict`子类`UserDict`。

要检查它们的性能，可以从计时核心字典操作开始，比如类实例化。在 Python [交互式](https://realpython.com/interacting-with-python/)会话中运行以下代码:

>>>

```py
>>> import timeit
>>> from extended_dicts import ExtendedDict_dict
>>> from extended_dicts import ExtendedDict_UserDict

>>> init_data = dict(zip(range(1000), range(1000)))

>>> dict_initialization = min(
...     timeit.repeat(
...         stmt="ExtendedDict_dict(init_data)",
...         number=1000,
...         repeat=5,
...         globals=globals(),
...     )
... )

>>> user_dict_initialization = min(
...     timeit.repeat(
...         stmt="ExtendedDict_UserDict(init_data)",
...         number=1000,
...         repeat=5,
...         globals=globals(),
...     )
... )

>>> print(
...     f"UserDict is {user_dict_initialization / dict_initialization:.3f}",
...     "times slower than dict",
... )
UserDict is 35.877 times slower than dict
```

在这段代码中，您使用 [`timeit`](https://docs.python.org/3/library/timeit.html?highlight=timeit#module-timeit) 模块和 [`min()`](https://realpython.com/python-min-and-max/) 函数来测量一段代码的执行时间。在这个例子中，目标代码由实例化`ExtendedDict_dict`和`ExtendedDict_UserDict`组成。

一旦运行了这个时间测量代码，就可以比较两个初始化时间。在这个具体的例子中，基于`UserDict`的类的初始化比从`dict`派生的类慢。这一结果表明存在严重的性能差异。

测量新功能的执行时间可能也很有趣。比如可以查看`.apply()`的执行时间。要进行这项检查，请继续运行以下代码:

>>>

```py
>>> extended_dict = ExtendedDict_dict(init_data)
>>> dict_apply = min(
...     timeit.repeat(
...         stmt="extended_dict.apply(lambda x: x**2)",
...         number=5,
...         repeat=2,
...         globals=globals(),
...     )
... )

>>> extended_user_dict = ExtendedDict_UserDict(init_data)
>>> user_dict_apply = min(
...     timeit.repeat(
...         stmt="extended_user_dict.apply(lambda x: x**2)",
...         number=5,
...         repeat=2,
...         globals=globals(),
...     )
... )

>>> print(
...     f"UserDict is {user_dict_apply / dict_apply:.3f}",
...     "times slower than dict",
... )
UserDict is 1.704 times slower than dict
```

基于`UserDict`的类和基于`dict`的类这次的性能差别不是那么大，但是还是存在的。

通常，当你通过子类化`dict`来创建一个自定义字典时，你可以期望标准字典操作在这个类中比在基于`UserDict`的类中更有效。另一方面，新功能在两个类中可能有相似的执行时间。你怎么知道哪条路是最有效的呢？你必须对你的代码进行时间测量。

值得注意的是，如果您的目标是修改核心字典功能，那么`UserDict`可能是一个不错的选择，因为在这种情况下，您将主要用纯 Python 重写`dict`类。

## 结论

现在，您知道了如何用修改的行为和新功能创建定制的类似字典的类。您已经学会了通过直接子类化内置的`dict`类和从`collections`模块中可用的`UserDict`类继承来实现这一点。

**在本教程中，您学习了如何:**

*   通过继承**内置的`dict`类**来创建类似字典的类
*   识别继承 Python 内置`dict`类的**常见陷阱**
*   通过从`collections`模块中子类化 **`UserDict`** 来构建类似字典的类

您还编写了一些实例，帮助您理解在创建自定义字典类时使用`UserDict`和`dict`的利弊。

现在，您已经准备好创建自定义词典，并利用 Python 中这种有用的数据类型的全部功能来响应您的编码需求。

**立即加入:** ，你将永远不会错过另一个 Python 教程、课程更新或帖子。*****