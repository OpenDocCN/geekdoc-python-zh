# Python 中的 OrderedDict 与 Dict:适合工作的工具

> 原文：<https://realpython.com/python-ordereddict/>

有时你需要一个 Python [字典](https://realpython.com/python-dicts/)来记住条目的顺序。在过去，你只有一个工具来解决这个特定的问题:Python 的 [`OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict) 。它是一个字典子类，专门用来记住条目的顺序，这是由键的插入顺序定义的。

这在 Python 3.6 中有所改变。内置的`dict`类现在也保持其项目有序。因此，Python 社区中的许多人现在想知道`OrderedDict`是否仍然有用。仔细观察`OrderedDict`会发现这个职业仍然提供有价值的特性。

**在本教程中，您将学习如何:**

*   在你的代码中创建并使用 **`OrderedDict`对象**
*   确定`OrderedDict`和`dict`之间的**差异**
*   了解使用`OrderedDict` vs `dict`的**优点**和**缺点**

有了这些知识，当您想要保持项目的顺序时，您将能够选择最适合您需要的字典类。

在本教程结束时，您将看到一个使用`OrderedDict`实现基于字典的队列的示例，如果您使用常规的`dict`对象，这将更具挑战性。

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 在`OrderedDict`和`dict`之间选择

多年来，Python [字典](https://realpython.com/iterate-through-dictionary-python/#a-few-words-on-dictionaries)是无序的[数据结构](https://realpython.com/python-data-structures/)。Python 开发者已经习惯了这个事实，当他们需要保持数据有序时，他们依赖于[列表](https://realpython.com/python-lists-tuples/)或其他序列。随着时间的推移，开发人员发现需要一种新型的字典，一种可以保持条目有序的字典。

早在 2008 年， [PEP 372](https://www.python.org/dev/peps/pep-0372/) 就引入了给 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) 增加一个新字典类的想法。它的主要目标是记住由插入键的顺序定义的项目顺序。那就是`OrderedDict`的由来。

核心 Python 开发人员想要填补这个空白，提供一个能够保持插入键顺序的字典。这反过来又使得依赖于这一特性的特定算法的实现更加简单。

`OrderedDict`被添加到 [Python 3.1](https://docs.python.org/3/whatsnew/3.1.html) 的标准库中。它的 API 本质上和`dict`一样。然而，`OrderedDict`按照插入键的顺序遍历键和值。如果新条目覆盖了现有条目，则项目的顺序保持不变。如果一个条目被删除并重新插入，那么它将被移动到字典的末尾。

Python 3.6 引入了一个[对`dict`和](https://docs.python.org/3/whatsnew/3.6.html#new-dict-implementation)的新实现。这个新的实现在内存使用和迭代效率方面取得了巨大的成功。此外，新的实现提供了一个新的、有点出乎意料的特性:`dict`对象现在以它们被引入时的顺序保存它们的项目。最初，这个特性被认为是一个实现细节，文档建议不要依赖它。

**注意:**在本教程中，您将重点关注 [CPython](https://www.python.org/about/) 提供的`dict`和`OrderedDict`的实现。

用核心 Python 开发者和`OrderedDict`的合著者[雷蒙德·赫廷格](https://twitter.com/raymondh)的话说，这个类是专门为保持其项目有序而设计的，而`dict`的新实现被设计得紧凑并提供快速迭代:

> 目前的正规词典是基于我几年前提出的设计。该设计的主要目标是紧凑性和快速迭代密集的键和值数组。维持秩序是一个人工制品，而不是一个设计目标。这个设计可以维持秩序，但这不是它的专长。
> 
> 相比之下，我给了`collections.OrderedDict`一个不同的设计(后来由埃里克·斯诺用 C 语言编写)。主要目标是有效地维护秩序，即使是在严重的工作负载下，例如由`lru_cache`施加的负载，它经常改变秩序而不触及底层的`dict`。有意地，`OrderedDict`有一个优先排序能力的设计，以额外的内存开销和常数因子更差的插入时间为代价。
> 
> 我的目标仍然是让`collections.OrderedDict`有一个不同的设计，有不同于普通字典的性能特征。它有一些常规字典没有的特定于顺序的方法(比如从两端有效弹出的一个`move_to_end()`和一个`popitem()`)。`OrderedDict`需要擅长这些操作，因为这是它区别于常规字典的地方。([来源](https://mail.python.org/pipermail/python-dev/2017-December/151266.html))

在 [Python 3.7](https://realpython.com/python37-new-features/) 中，`dict`对象的项目排序特性被[宣布为](https://mail.python.org/pipermail/python-dev/2017-December/151283.html)Python 语言规范的正式部分。因此，从那时起，当开发人员需要一个保持条目有序的字典时，他们可以依赖`dict`。

此时，一个问题产生了:在`dict`的这个新实现之后，还需要`OrderedDict`吗？答案取决于您的具体用例，也取决于您希望在代码中有多明确。

在撰写本文时，`OrderedDict`的一些特性仍然使它有价值，并且不同于普通的`dict`:

1.  **意图信号:**如果你使用`OrderedDict`而不是`dict`，那么你的代码清楚地表明了条目在字典中的顺序是重要的。你清楚地表达了你的代码需要或者依赖于底层字典中的条目顺序。
2.  **控制条目的顺序:**如果您需要重新排列或重新排序字典中的条目，那么您可以使用 [`.move_to_end()`](https://docs.python.org/3/library/collections.html#collections.OrderedDict.move_to_end) 以及 [`.popitem()`](https://docs.python.org/3/library/collections.html#collections.OrderedDict.popitem) 的增强变体。
3.  **相等测试行为:**如果您的代码比较字典的相等性，并且条目的顺序在比较中很重要，那么`OrderedDict`是正确的选择。

至少还有一个在代码中继续使用`OrderedDict`的理由:**向后兼容性**。在运行 than 3.6 之前版本的环境中，依靠常规的`dict`对象来保持项目的顺序会破坏您的代码。

很难说`dict`会不会很快全面取代`OrderedDict`。如今，`OrderedDict`仍然提供有趣和有价值的特性，当你为一个给定的工作选择一个工具时，你可能想要考虑这些特性。

[*Remove ads*](/account/join/)

## Python 的`OrderedDict` 入门

Python 的`OrderedDict`是一个`dict`子类，它保留了**键-值对**，俗称**项**插入字典的顺序。当你迭代一个`OrderedDict`对象时，条目会按照原来的顺序被遍历。如果更新现有键的值，则顺序保持不变。如果您删除一个条目并重新插入，那么该条目将被添加到词典的末尾。

成为一个`dict`子类意味着它继承了常规字典提供的所有方法。`OrderedDict`还有其他特性，您将在本教程中了解到。然而，在本节中，您将学习在代码中创建和使用`OrderedDict`对象的基础知识。

### 创建`OrderedDict`个对象

与`dict`不同，`OrderedDict`不是内置类型，所以创建`OrderedDict`对象的第一步是[从`collections`导入](https://realpython.com/python-import/)类。有几种方法可以创建有序字典。它们中的大多数与你如何创建一个常规的`dict`对象是一样的。例如，您可以通过实例化不带参数的类来创建一个空的`OrderedDict`对象:

>>>

```py
>>> from collections import OrderedDict

>>> numbers = OrderedDict()

>>> numbers["one"] = 1
>>> numbers["two"] = 2
>>> numbers["three"] = 3

>>> numbers
OrderedDict([('one', 1), ('two', 2), ('three', 3)])
```

在这种情况下，首先从`collections`导入`OrderedDict`。然后通过实例化`OrderedDict`创建一个空的有序字典，而不向构造函数提供参数。

通过在方括号(`[]`)中提供一个键并为该键赋值，可以将键-值对添加到字典中。当您引用`numbers`时，您会得到一个键-值对的 iterable，它按照条目被插入字典的顺序保存条目。

您还可以将 iterable items 作为参数传递给`OrderedDict`的构造函数:

>>>

```py
>>> from collections import OrderedDict

>>> numbers = OrderedDict([("one", 1), ("two", 2), ("three", 3)])
>>> numbers
OrderedDict([('one', 1), ('two', 2), ('three', 3)])

>>> letters = OrderedDict({("a", 1), ("b", 2), ("c", 3)})
>>> letters
OrderedDict([('c', 3), ('a', 1), ('b', 2)])
```

当您使用一个[序列](https://docs.python.org/3/library/stdtypes.html?highlight=sequence#sequence-types-list-tuple-range)时，比如一个`list`或一个`tuple`，结果排序字典中的条目顺序与输入序列中条目的原始顺序相匹配。如果您使用一个`set`，就像上面的第二个例子，那么直到`OrderedDict`被创建之前，项目的最终顺序是未知的。

如果您使用一个常规字典作为一个`OrderedDict`对象的初始化器，并且您使用的是 Python 3.6 或更高版本，那么您会得到以下行为:

>>>

```py
Python 3.9.0 (default, Oct  5 2020, 17:52:02)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from collections import OrderedDict

>>> numbers = OrderedDict({"one": 1, "two": 2, "three": 3})
>>> numbers
OrderedDict([('one', 1), ('two', 2), ('three', 3)])
```

`OrderedDict`对象中项目的顺序与原始字典中的顺序相匹配。另一方面，如果您使用低于 3.6 的 Python 版本，那么项目的顺序是未知的:

>>>

```py
Python 3.5.10 (default, Jan 25 2021, 13:22:52)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from collections import OrderedDict

>>> numbers = OrderedDict({"one": 1, "two": 2, "three": 3})
>>> numbers
OrderedDict([('one', 1), ('three', 3), ('two', 2)])
```

因为 Python 3.5 中的字典不记得条目的顺序，所以在创建对象之前，您不知道结果有序字典中的顺序。从这一点上来说，秩序得到了维护。

您可以通过将[关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-arguments)传递给类构造函数来创建有序字典:

>>>

```py
>>> from collections import OrderedDict

>>> numbers = OrderedDict(one=1, two=2, three=3)
>>> numbers
OrderedDict([('one', 1), ('two', 2), ('three', 3)])
```

自从 [Python 3.6](https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep468) 以来，函数保留了调用中传递的关键字参数的顺序。因此，上面的`OrderedDict`中的项目顺序与您将关键字参数传递给构造函数的顺序相匹配。在早期的 Python 版本中，这个顺序是未知的。

最后，`OrderedDict`还提供了`.fromkeys()`，它从一个可迭代的键创建一个新字典，并将其所有值设置为一个公共值:

>>>

```py
>>> from collections import OrderedDict

>>> keys = ["one", "two", "three"]
>>> OrderedDict.fromkeys(keys, 0)
OrderedDict([('one', 0), ('two', 0), ('three', 0)])
```

在这种情况下，您使用一个键列表作为起点来创建一个有序字典。`.fromkeys()`的第二个参数为字典中的所有条目提供一个值。

[*Remove ads*](/account/join/)

### 管理`OrderedDict`中的项目

由于`OrderedDict`是一个[可变的](https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types)数据结构，你可以对它的实例执行**变异操作**。您可以插入新项目，更新和删除现有项目，等等。如果您在现有的有序字典中插入一个新项目，则该项目会被添加到字典的末尾:

>>>

```py
>>> from collections import OrderedDict

>>> numbers = OrderedDict(one=1, two=2, three=3)
>>> numbers
OrderedDict([('one', 1), ('two', 2), ('three', 3)])

>>> numbers["four"] = 4
>>> numbers
OrderedDict([('one', 1), ('two', 2), ('three', 3), ('four', 4)])
```

新添加的条目`('four', 4)`放在底层字典的末尾，因此现有条目的顺序保持不变，字典保持插入顺序。

如果从现有的有序字典中删除一个项目，然后再次插入该项目，则该项目的新实例将被放在字典的末尾:

>>>

```py
>>> from collections import OrderedDict
>>> numbers = OrderedDict(one=1, two=2, three=3)

>>> del numbers["one"]
>>> numbers
OrderedDict([('two', 2), ('three', 3)])

>>> numbers["one"] = 1
>>> numbers
OrderedDict([('two', 2), ('three', 3), ('one', 1)])
```

如果删除`('one', 1)`项并插入同一项的新实例，那么新项将被添加到底层字典的末尾。

如果您重新分配或更新一个`OrderedDict`对象中现有键值对的值，那么键会保持其位置，但会获得一个新值:

>>>

```py
>>> from collections import OrderedDict
>>> numbers = OrderedDict(one=1, two=2, three=3)

>>> numbers["one"] = 1.0
>>> numbers
OrderedDict([('one', 1.0), ('two', 2), ('three', 3)])

>>> numbers.update(two=2.0)
>>> numbers
OrderedDict([('one', 1.0), ('two', 2.0), ('three', 3)])
```

如果在有序字典中更新给定键的值，那么该键不会被移动，而是被赋予新的值。同样，如果您使用`.update()`来修改一个现有的键-值对的值，那么字典会记住键的位置，并将更新后的值赋给它。

### 迭代一个`OrderedDict`

就像普通的字典一样，你可以使用几种工具和技术通过一个对象`OrderedDict`来[迭代](https://realpython.com/iterate-through-dictionary-python/)。可以直接迭代键，也可以使用字典方法，比如[`.items()`](https://docs.python.org/3/library/stdtypes.html#dict.items)[`.keys()`](https://docs.python.org/3/library/stdtypes.html#dict.keys)[`.values()`](https://docs.python.org/3/library/stdtypes.html#dict.values):

>>>

```py
>>> from collections import OrderedDict
>>> numbers = OrderedDict(one=1, two=2, three=3)

>>> # Iterate over the keys directly
>>> for key in numbers:
...     print(key, "->", numbers[key])
...
one -> 1
two -> 2
three -> 3

>>> # Iterate over the items using .items()
>>> for key, value in numbers.items():
...     print(key, "->", value)
...
one -> 1
two -> 2
three -> 3

>>> # Iterate over the keys using .keys()
>>> for key in numbers.keys():
...     print(key, "->", numbers[key])
...
one -> 1
two -> 2
three -> 3

>>> # Iterate over the values using .values()
>>> for value in numbers.values():
...     print(value)
...
1
2
3
```

第一个 [`for`循环](https://realpython.com/python-for-loop/)直接迭代`numbers`的键。其他三个循环使用字典方法来迭代`numbers`的条目、键和值。

### 用`reversed()`和逆序迭代

从 [Python 3.5](https://docs.python.org/3/whatsnew/3.5.html#collections) 开始，`OrderedDict`提供的另一个重要特性是，它的项、键和值支持使用 [`reversed()`](https://docs.python.org/3/library/functions.html#reversed) 的反向迭代。这个[特性](https://docs.python.org/3/whatsnew/3.8.html#other-language-changes)被添加到了 [Python 3.8](https://realpython.com/python38-new-features/) 的常规字典中。因此，如果您的代码使用它，那么您的向后兼容性会受到普通字典的更多限制。

您可以将`reversed()`与`OrderedDict`对象的项目、键和值一起使用:

>>>

```py
>>> from collections import OrderedDict
>>> numbers = OrderedDict(one=1, two=2, three=3)

>>> # Iterate over the keys directly in reverse order
>>> for key in reversed(numbers):
...     print(key, "->", numbers[key])
...
three -> 3
two -> 2
one -> 1

>>> # Iterate over the items in reverse order
>>> for key, value in reversed(numbers.items()):
...     print(key, "->", value)
...
three -> 3
two -> 2
one -> 1

>>> # Iterate over the keys in reverse order
>>> for key in reversed(numbers.keys()):
...     print(key, "->", numbers[key])
...
three -> 3
two -> 2
one -> 1

>>> # Iterate over the values in reverse order
>>> for value in reversed(numbers.values()):
...     print(value)
...
3
2
1
```

本例中的每个循环都使用`reversed()`以逆序遍历有序字典中的不同元素。

常规词典也支持反向迭代。然而，如果您试图在低于 3.8 的 Python 版本中对常规的`dict`对象使用`reversed()`，那么您会得到一个`TypeError`:

>>>

```py
Python 3.7.9 (default, Jan 14 2021, 11:41:20)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> numbers = dict(one=1, two=2, three=3)

>>> for key in reversed(numbers):
...     print(key)
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'dict' object is not reversible
```

如果需要逆序遍历字典中的条目，那么`OrderedDict`是一个很好的盟友。使用常规字典极大地降低了向后兼容性，因为直到 Python 3.8，反向迭代才被添加到常规字典中。

[*Remove ads*](/account/join/)

## 探索 Python 的`OrderedDict` 的独特功能

从 Python 3.6 开始，常规字典按照插入底层字典的顺序保存条目。正如你到目前为止所看到的，这限制了`OrderedDict`的有用性。然而，`OrderedDict`提供了一些你在常规的`dict`对象中找不到的独特特性。

使用有序字典，您可以访问以下额外的和增强的方法:

*   [`.move_to_end()`](https://docs.python.org/3/library/collections.html#collections.OrderedDict.move_to_end) 是 Python 3.2 中添加的一个新方法[，它允许你将一个已有的条目移动到字典的末尾或开头。](https://docs.python.org/3/whatsnew/3.2.html#collections)

*   [`.popitem()`](https://docs.python.org/3/library/collections.html#collections.OrderedDict.popitem) 是其对应的 [`dict.popitem()`](https://docs.python.org/3/library/stdtypes.html#dict.popitem) 的增强变体，允许您从底层有序字典的末尾或开头移除和返回一个项目。

`OrderedDict`和`dict`在进行相等性测试时也表现不同。具体来说，当您比较有序字典时，条目的顺序很重要。正规词典就不是这样了。

最后，`OrderedDict`实例提供了一个名为 [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 的属性，这是你在常规字典实例中找不到的。此属性允许您向现有有序字典添加自定义可写属性。

### 用`.move_to_end()`和重新排序项目

`dict`和`OrderedDict`最显著的区别之一是后者有一个额外的方法叫做`.move_to_end()`。这种方法允许您将现有的条目移动到底层字典的末尾或开头，因此这是一个重新排序字典的好工具。

当您使用`.move_to_end()`时，您可以提供两个参数:

1.  **`key`** 持有标识您要移动的项目的键。如果`key`不存在，那么你得到一个 [`KeyError`](https://realpython.com/python-keyerror/) 。

2.  **`last`** 保存一个[布尔](https://realpython.com/python-boolean/)值，该值定义了您想要将手头的项目移动到词典的哪一端。它默认为`True`，这意味着该项目将被移动到词典的末尾或右侧。`False`表示该条目将被移到有序字典的前面或左侧。

下面是一个如何使用带有`key`参数的`.move_to_end()`并依赖于默认值`last`的例子:

>>>

```py
>>> from collections import OrderedDict
>>> numbers = OrderedDict(one=1, two=2, three=3)
>>> numbers
OrderedDict([('one', 1), ('two', 2), ('three', 3)])

>>> numbers.move_to_end("one")
>>> numbers
OrderedDict([('two', 2), ('three', 3), ('one', 1)])
```

当您用一个`key`作为参数调用`.move_to_end()`时，您将手头的键-值对移动到字典的末尾。这就是为什么`('one', 1)`现在处于最后的位置。请注意，其余项目仍保持原来的顺序。

如果您将`False`传递到`last`，那么您将该项目移动到开头:

>>>

```py
>>> numbers.move_to_end("one", last=False)
>>> numbers
OrderedDict([('one', 1), ('two', 2), ('three', 3)])
```

在这种情况下，您将`('one', 1)`移动到字典的开头。这提供了一个有趣而强大的特性。例如，使用`.move_to_end()`，您可以[按关键字对有序字典](https://realpython.com/sort-python-dictionary/)进行排序:

>>>

```py
>>> from collections import OrderedDict
>>> letters = OrderedDict(b=2, d=4, a=1, c=3)

>>> for key in sorted(letters):
...     letters.move_to_end(key)
...
>>> letters
OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
```

在本例中，首先创建一个有序字典`letters`。`for`循环遍历其排序后的键，并将每一项移动到字典的末尾。当循环结束时，有序字典的条目按键排序。

按值对字典排序将是一个有趣的练习，所以扩展下面的块并尝试一下吧！



按值对以下字典进行排序:

>>>

```py
>>> from collections import OrderedDict
>>> letters = OrderedDict(a=4, b=3, d=1, c=2)
```

作为实现解决方案的有用提示，考虑使用 [`lambda`函数](https://realpython.com/python-lambda/)。

您可以展开下面的方框，查看可能的解决方案。



您可以使用一个`lambda`函数来检索`letters`中每个键值对的值，并使用该函数作为`sorted()`的`key`参数:

>>>

```py
>>> for key, _ in sorted(letters.items(), key=lambda item: item[1]):
...     letters.move_to_end(key)
...
>>> letters
OrderedDict([('d', 1), ('c', 2), ('b', 3), ('a', 4)])
```

在这段代码中，您使用了一个`lambda`函数，该函数返回`letters`中每个键值对的值。对`sorted()`的调用使用这个`lambda`函数从输入 iterable，`letters.items()`的每个元素中提取一个**比较键**。然后你用`.move_to_end()`排序`letters`。

太好了！现在，您知道如何使用`.move_to_end()`对有序的字典进行重新排序。你已经准备好进入下一部分了。

[*Remove ads*](/account/join/)

### 移除带有`.popitem()`和的项目

`OrderedDict`另一个有趣的特点是它的增强版`.popitem()`。默认情况下，`.popitem()`按照 [LIFO](https://en.wikipedia.org/w/index.php?title=LIFO_(computing)&redirect=no) (后进先出)的顺序移除并返回一个项目。换句话说，它从有序字典的右端删除项目:

>>>

```py
>>> from collections import OrderedDict
>>> numbers = OrderedDict(one=1, two=2, three=3)

>>> numbers.popitem()
('three', 3)
>>> numbers.popitem()
('two', 2)
>>> numbers.popitem()
('one', 1)
>>> numbers.popitem()
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    numbers.popitem()
KeyError: 'dictionary is empty'
```

在这里，您使用`.popitem()`删除`numbers`中的所有项目。每次调用此方法都会从基础字典的末尾移除一项。如果你在一个空字典上调用`.popitem()`，那么你得到一个`KeyError`。到目前为止，`.popitem()`的行为和普通字典中的一样。

然而在`OrderedDict`中，`.popitem()`也接受一个名为`last`的布尔参数，默认为`True`。如果您将`last`设置为`False`，那么`.popitem()`将按照 [FIFO](https://en.wikipedia.org/wiki/FIFO_(computing_and_electronics)) (先进/先出)的顺序移除条目，这意味着它将从字典的开头移除条目:

>>>

```py
>>> from collections import OrderedDict
>>> numbers = OrderedDict(one=1, two=2, three=3)

>>> numbers.popitem(last=False)
('one', 1)
>>> numbers.popitem(last=False)
('two', 2)
>>> numbers.popitem(last=False)
('three', 3)
>>> numbers.popitem(last=False)
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    numbers.popitem(last=False)
KeyError: 'dictionary is empty'
```

当`last`设置为`True`时，您可以使用`.popitem()`从有序字典的开头移除和返回条目。在本例中，对`.popitem()`的最后一次调用引发了一个`KeyError`，因为底层字典已经为空。

### 测试字典之间的相等性

当您在布尔上下文中测试两个`OrderedDict`对象的相等性时，项目的顺序起着重要的作用。例如，如果您的有序字典包含相同的项目集，则测试结果取决于它们的顺序:

>>>

```py
>>> from collections import OrderedDict
>>> letters_0 = OrderedDict(a=1, b=2, c=3, d=4)
>>> letters_1 = OrderedDict(b=2, a=1, c=3, d=4)
>>> letters_2 = OrderedDict(a=1, b=2, c=3, d=4)

>>> letters_0 == letters_1
False

>>> letters_0 == letters_2
True
```

在这个例子中，`letters_1`与`letters_0`和`letters_2`相比，其条目的顺序略有不同，所以第一个测试返回`False`。在第二个测试中，`letters_0`和`letters_2`有相同的一组项目，它们的顺序相同，所以测试返回`True`。

如果你用普通字典尝试同样的例子，你会得到不同的结果:

>>>

```py
>>> letters_0 = dict(a=1, b=2, c=3, d=4)
>>> letters_1 = dict(b=2, a=1, c=3, d=4)
>>> letters_2 = dict(a=1, b=2, c=3, d=4)

>>> letters_0 == letters_1
True

>>> letters_0 == letters_2
True

>>> letters_0 == letters_1 == letters_2
True
```

在这里，当您测试两个常规字典的相等性时，如果两个字典有相同的条目集，您会得到`True`。在这种情况下，项目的顺序不会改变最终结果。

最后，`OrderedDict`对象和常规字典之间的相等测试不考虑条目的顺序:

>>>

```py
>>> from collections import OrderedDict
>>> letters_0 = OrderedDict(a=1, b=2, c=3, d=4)
>>> letters_1 = dict(b=2, a=1, c=3, d=4)

>>> letters_0 == letters_1
True
```

当您比较有序词典和常规词典时，条目的顺序并不重要。如果两个字典有相同的条目集，那么无论条目的顺序如何，它们都进行同等的比较。

### 向字典实例追加新属性

`OrderedDict`对象有一个`.__dict__`属性，你在常规字典对象中找不到。看一下下面的代码:

>>>

```py
>>> from collections import OrderedDict
>>> letters = OrderedDict(b=2, d=4, a=1, c=3)
>>> letters.__dict__
{}

>>> letters1 = dict(b=2, d=4, a=1, c=3)
>>> letters1.__dict__
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    letters1.__dict__
AttributeError: 'dict' object has no attribute '__dict__'
```

在第一个例子中，您访问有序字典`letters`上的`.__dict__`属性。Python 内部使用这个属性来存储可写的实例属性。第二个例子显示常规字典对象没有`.__dict__`属性。

您可以使用有序字典的`.__dict__`属性来存储动态创建的可写实例属性。有几种方法可以做到这一点。例如，您可以使用字典风格的赋值，就像在`ordered_dict.__dict__["attr"] = value`中一样。你也可以使用点符号，就像在`ordered_dict.attr = value`中一样。

下面是一个使用`.__dict__`将新函数附加到现有有序字典的例子:

>>>

```py
>>> from collections import OrderedDict
>>> letters = OrderedDict(b=2, d=4, a=1, c=3)

>>> letters.sorted_keys = lambda: sorted(letters.keys())
>>> vars(letters)
{'sorted_keys': <function <lambda> at 0x7fa1e2fe9160>}

>>> letters.sorted_keys()
['a', 'b', 'c', 'd']

>>> letters["e"] = 5
>>> letters.sorted_keys()
['a', 'b', 'c', 'd', 'e']
```

现在你有了一个`.sorted_keys()` [`lambda`函数](https://realpython.com/python-lambda/)附加到你的`letters`命令字典上。请注意，您可以通过直接使用**点符号**或使用 [`vars()`](https://realpython.com/python-scope-legb-rule/#vars) 来检查`.__dict__`的内容。

**注意:**这种动态属性被添加到给定类的特定实例中。在上面的例子中，那个实例是`letters`。这既不影响其他实例，也不影响类本身，所以您只能通过`letters`访问`.sorted_keys()`。

您可以使用这个动态添加的函数按照排序顺序遍历字典键，而不改变`letters`中的原始顺序:

>>>

```py
>>> for key in letters.sorted_keys():
...     print(key, "->", letters[key])
...
a -> 1
b -> 2
c -> 3
d -> 4
e -> 5

>>> letters
OrderedDict([('b', 2), ('d', 4), ('a', 1), ('c', 3), ('e', 5)])
```

这只是一个例子，说明了`OrderedDict`的这个特性有多有用。请注意，您不能用普通词典做类似的事情:

>>>

```py
>>> letters = dict(b=2, d=4, a=1, c=3)
>>> letters.sorted_keys = lambda: sorted(letters.keys())
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    letters.sorted_keys = lambda: sorted(letters.keys())
AttributeError: 'dict' object has no attribute 'sorted_keys'
```

如果您尝试向常规字典动态添加定制实例属性，那么您会得到一个`AttributeError`消息，告诉您底层字典手头没有该属性。这是因为常规字典没有一个`.__dict__`属性来保存新的实例属性。

[*Remove ads*](/account/join/)

## 用运算符合并和更新字典

[Python 3.9](https://realpython.com/python39-new-features/) 给字典空间增加了两个新的操作符。现在你有了**合并** ( `|`)和**更新** ( `|=`)字典操作符。这些操作符也处理`OrderedDict`实例:

>>>

```py
Python 3.9.0 (default, Oct  5 2020, 17:52:02)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from collections import OrderedDict

>>> physicists = OrderedDict(newton="1642-1726", einstein="1879-1955")
>>> biologists = OrderedDict(darwin="1809-1882", mendel="1822-1884")

>>> scientists = physicists | biologists
>>> scientists
OrderedDict([
 ('newton', '1642-1726'),
 ('einstein', '1879-1955'),
 ('darwin', '1809-1882'),
 ('mendel', '1822-1884')
])
```

顾名思义，merge 操作符将两个字典合并成一个包含两个初始字典的条目的新字典。如果表达式中的字典有公共键，那么最右边的字典的值将优先。

当您有一个字典并且想要更新它的一些值而不调用`.update()`时，update 操作符很方便:

>>>

```py
>>> physicists = OrderedDict(newton="1642-1726", einstein="1879-1955")

>>> physicists_1 = OrderedDict(newton="1642-1726/1727", hawking="1942-2018")
>>> physicists |= physicists_1
>>> physicists
OrderedDict([
 ('newton', '1642-1726/1727'),
 ('einstein', '1879-1955'),
 ('hawking', '1942-2018')
])
```

在这个例子中，您使用字典更新操作符来更新[牛顿的寿命](https://en.wikipedia.org/wiki/Isaac_Newton#cite_note-OSNS-3)信息。操作员就地更新字典。如果提供更新数据的字典有新的键，那么这些键将被添加到原始字典的末尾。

## 考虑性能

性能是编程中的一个重要课题。了解算法运行的速度或它使用的内存是人们普遍关心的问题。`OrderedDict`最初是用 Python 编写的[，然后用 C](https://github.com/python/cpython/blob/master/Lib/collections/__init__.py) 编写的[，以最大化其方法和操作的效率。这两个实现目前在标准库中都可用。然而，如果 C 实现由于某种原因不可用，Python 实现可以作为一种替代。](https://github.com/python/cpython/blob/226a012d1cd61f42ecd3056c554922f359a1a35d/Objects/odictobject.c)

`OrderedDict`的两个实现都涉及到使用一个[双向链表](https://realpython.com/linked-lists-python/#how-to-use-doubly-linked-lists)来捕获条目的顺序。尽管有些操作有线性时间，但`OrderedDict`中的链表实现被高度优化，以保持相应字典方法的快速时间。也就是说，有序字典上的操作是 [*O* (1)](https://realpython.com/sorting-algorithms-python/#measuring-efficiency-with-big-o-notation) ，但是与常规字典相比具有更大的常数因子。

总的来说，`OrderedDict`的性能比一般的字典要低。下面是一个测量两个字典类上几个操作的执行时间的例子:

```py
# time_testing.py

from collections import OrderedDict
from time import perf_counter

def average_time(dictionary):
    time_measurements = []
    for _ in range(1_000_000):
        start = perf_counter()
        dictionary["key"] = "value"
        "key" in dictionary
        "missing_key" in dictionary
        dictionary["key"]
        del dictionary["key"]
        end = perf_counter()
        time_measurements.append(end - start)
    return sum(time_measurements) / len(time_measurements) * int(1e9)

ordereddict_time = average_time(OrderedDict.fromkeys(range(1000)))
dict_time = average_time(dict.fromkeys(range(1000)))
gain = ordereddict_time / dict_time

print(f"OrderedDict: {ordereddict_time:.2f} ns")
print(f"dict: {dict_time:.2f} ns ({gain:.2f}x faster)")
```

在这个脚本中，您将计算在给定的字典上运行几个常见操作所需的`average_time()`。`for`循环使用 [`time.pref_counter()`](https://docs.python.org/3/library/time.html#time.perf_counter) 来衡量一组操作的执行时间。该函数返回运行所选操作集所需的平均时间(以纳秒为单位)。

**注意:**如果你有兴趣知道其他方法来计时你的代码，那么你可以看看 [Python 计时器函数:三种方法来监控你的代码](https://realpython.com/python-timer/)。

如果您从命令行[运行这个脚本](https://realpython.com/run-python-scripts/),那么您会得到类似如下的输出:

```py
$ python time_testing.py
OrderedDict: 272.93 ns
dict:        197.88 ns (1.38x faster)
```

正如您在输出中看到的，对`dict`对象的操作比对`OrderedDict`对象的操作快。

关于内存消耗，`OrderedDict`实例必须支付存储成本，因为它们的键列表是有序的。这里有一个脚本可以让您了解这种内存开销:

>>>

```py
>>> import sys
>>> from collections import OrderedDict

>>> ordereddict_memory = sys.getsizeof(OrderedDict.fromkeys(range(1000)))
>>> dict_memory = sys.getsizeof(dict.fromkeys(range(1000)))
>>> gain = 100 - dict_memory / ordereddict_memory * 100

>>> print(f"OrderedDict: {ordereddict_memory} bytes")
OrderedDict: 85408 bytes

>>> print(f"dict: {dict_memory} bytes ({gain:.2f}% lower)")
dict:        36960 bytes (56.73% lower)
```

在这个例子中，您使用 [`sys.getsizeof()`](https://docs.python.org/3/library/sys.html?highlight=sys#sys.getsizeof) 来测量两个字典对象的内存占用量(以字节为单位)。在输出中，您可以看到常规字典比其对应的`OrderedDict`占用更少的内存。

[*Remove ads*](/account/join/)

## 为工作选择正确的词典

到目前为止，你已经了解了`OrderedDict`和`dict`之间的细微差别。您已经了解到，尽管从 Python 3.6 开始，常规字典已经是有序的数据结构，但是使用`OrderedDict`仍然有一些价值，因为有一组有用的特性是`dict`中没有的。

下面总结了这两个类更相关的差异和特性，在您决定使用哪一个时应该加以考虑:

| 特征 | `OrderedDict` | `dict` |
| --- | --- | --- |
| 保持钥匙插入顺序 | 是(从 Python 3.1 开始) | 是(从 Python 3.6 开始) |
| 关于项目顺序的可读性和意图信号 | 高的 | 低的 |
| 对项目顺序的控制 | 高(`.move_to_end()`，增强型`.popitem()`) | 低(需要移除和重新插入项目) |
| 运营绩效 | 低的 | 高的 |
| 内存消耗 | 高的 | 低的 |
| 相等测试考虑项目的顺序 | 是 | 不 |
| 支持反向迭代 | 是(从 Python 3.5 开始) | 是(从 Python 3.8 开始) |
| 能够附加新的实例属性 | 是(`.__dict__`属性) | 不 |
| 支持合并(`&#124;`)和更新(`&#124;=`)字典操作符 | 是(从 Python 3.9 开始) | 是(从 Python 3.9 开始) |

这个表格总结了`OrderedDict`和`dict`之间的一些主要区别，当您需要选择一个字典类来解决一个问题或者实现一个特定的算法时，您应该考虑这些区别。一般来说，如果字典中条目的顺序对于代码的正确运行至关重要，那么你首先应该看一看`OrderedDict`。

## 构建基于字典的队列

您应该考虑使用`OrderedDict`对象而不是`dict`对象的一个用例是，当您需要实现基于字典的[队列](https://en.wikipedia.org/wiki/Queue_(abstract_data_type))时。队列是以 FIFO 方式管理其项目的常见且有用的数据结构。这意味着您在队列的末尾推入新的项目，而旧的项目从队列的开头弹出。

通常，队列实现一个操作来将一个项目添加到它们的末尾，这被称为**入队**操作。队列还实现了一个从其开始处移除项目的操作，这就是所谓的**出列**操作。

要创建基于字典的队列，启动您的[代码编辑器或 IDE](https://realpython.com/python-ides-code-editors-guide/) ，创建一个名为`queue.py`的新 Python 模块，并向其中添加以下代码:

```py
# queue.py

from collections import OrderedDict

class Queue:
    def __init__(self, initial_data=None, /, **kwargs):
        self.data = OrderedDict()
        if initial_data is not None:
            self.data.update(initial_data)
        if kwargs:
            self.data.update(kwargs)

    def enqueue(self, item):
        key, value = item
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value

    def dequeue(self):
        try:
            return self.data.popitem(last=False)
        except KeyError:
            print("Empty queue")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Queue({self.data.items()})"
```

在`Queue`中，首先初始化一个名为`.data`的[实例属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)。这个属性包含一个空的有序字典，您将使用它来存储数据。类初始化器采用第一个可选参数`initial_data`，允许您在实例化类时提供初始数据。初始化器还带有可选的关键字参数( [`kwargs`](https://realpython.com/python-kwargs-and-args/) )，允许您在构造函数中使用关键字参数。

然后编写`.enqueue()`，它允许您将键值对添加到队列中。在这种情况下，如果键已经存在，就使用`.move_to_end()`,对新键使用普通赋值。注意，为了让这个方法工作，您需要提供一个两项的`tuple`或`list`以及一个有效的键-值对。

`.dequeue()`实现使用`.popitem()`和设置为`False`的`last`从底层有序字典`.data`的开始移除和返回条目。在这种情况下，您使用一个 [`try` … `except`块](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions)来处理在空字典上调用`.popitem()`时发生的`KeyError`。

特殊方法 [`.__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 提供了检索内部有序字典`.data`长度所需的功能。最后，当您将数据结构打印到屏幕上时，特殊的方法 [`.__repr__()`](https://realpython.com/operator-function-overloading/#representing-your-objects-using-repr) 提供了队列的用户友好的**字符串表示**。

以下是一些如何使用`Queue`的例子:

>>>

```py
>>> from queue import Queue

>>> # Create an empty queue
>>> empty_queue = Queue()
>>> empty_queue
Queue(odict_items([]))

>>> # Create a queue with initial data
>>> numbers_queue = Queue([("one", 1), ("two", 2)])
>>> numbers_queue
Queue(odict_items([('one', 1), ('two', 2)]))

>>> # Create a queue with keyword arguments
>>> letters_queue = Queue(a=1, b=2, c=3)
>>> letters_queue
Queue(odict_items([('a', 1), ('b', 2), ('c', 3)]))

>>> # Add items
>>> numbers_queue.enqueue(("three", 3))
>>> numbers_queue
Queue(odict_items([('one', 1), ('two', 2), ('three', 3)]))

>>> # Remove items
>>> numbers_queue.dequeue()
('one', 1)
>>> numbers_queue.dequeue()
('two', 2)
>>> numbers_queue.dequeue()
('three', 3)
>>> numbers_queue.dequeue()
Empty queue
```

在这个代码示例中，首先使用不同的方法创建三个不同的`Queue`对象。然后使用`.enqueue()`在`numbers_queue`的末尾添加一个条目。最后，你多次调用`.dequeue()`来移除`numbers_queue`中的所有物品。请注意，对`.dequeue()`的最后一个调用将一条消息打印到屏幕上，通知您队列已经为空。

## 结论

多年来，Python 字典都是无序的数据结构。这揭示了对有序字典的需求，在项目的**顺序很重要的情况下，有序字典会有所帮助。所以 Python 开发者创造了 [`OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict) ，它是专门为保持其条目有序而设计的。**

Python 3.6 在常规词典中引入了一个新特性。现在他们还记得物品的顺序。有了这个补充，大多数 Python 程序员想知道他们是否还需要考虑使用`OrderedDict`。

**在本教程中，您学习了:**

*   如何在代码中创建和使用 **`OrderedDict`对象**
*   `OrderedDict`和`dict`之间的主要**差异**是什么
*   使用`OrderedDict` vs `dict`的**好处**和**坏处**是什么

现在，如果您的代码需要一个有序的字典，您可以更好地决定是使用`dict`还是`OrderedDict`。

在本教程中，您编写了一个如何实现基于字典的队列的示例，这是一个用例，表明`OrderedDict`在您的日常 Python 编码冒险中仍然有价值。******