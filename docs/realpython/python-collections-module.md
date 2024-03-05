# Python 的集合:专门化数据类型的自助餐

> 原文：<https://realpython.com/python-collections-module/>

Python 的 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) 模块提供了一组丰富的**专用容器数据类型**，这些数据类型经过精心设计，以 python 化且高效的方式处理特定的编程问题。该模块还提供了包装类，使得创建行为类似于内置类型`dict`、`list`和`str`的定制类更加安全。

学习`collections`中的数据类型和类将允许你用一套有价值的可靠而有效的工具来扩充你的编程工具包。

**在本教程中，您将学习如何:**

*   用`namedtuple`编写**可读**和**显式**代码
*   使用`deque`构建**高效队列和堆栈**
*   **用`Counter`快速计数**物体
*   用`defaultdict`处理**缺失的字典键**
*   用`OrderedDict`保证**插入顺序**
*   使用`ChainMap`将**多个字典**作为一个单元进行管理

为了更好地理解`collections`中的数据类型和类，你应该知道使用 Python 内置数据类型的基础知识，比如[列表、元组](https://realpython.com/python-lists-tuples/)和[字典](https://realpython.com/python-dicts/)。另外，文章的最后一部分需要一些关于 Python 中面向对象编程的基础知识。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## Python 的`collections` 入门

回到 [Python 2.4](https://docs.python.org/3/whatsnew/2.4.html#new-improved-and-deprecated-modules) ， [Raymond Hettinger](https://twitter.com/raymondh) 为[标准库](https://docs.python.org/3/library/index.html)贡献了一个名为 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) 的新模块。目标是提供各种专门的集合数据类型来解决特定的编程问题。

当时，`collections`只包含一个数据结构， **`deque`** ，专门设计为一个[双端队列](https://en.wikipedia.org/wiki/Double-ended_queue)，支持序列两端高效的**追加**和**弹出**操作。从这一点开始，标准库中的几个模块利用了`deque`来提高它们的类和结构的性能。一些突出的例子是 [`queue`](https://docs.python.org/3/library/queue.html#module-queue) 和 [`threading`](https://docs.python.org/3/library/threading.html#module-threading) 。

随着时间的推移，一些专门的容器数据类型填充了该模块:

| 数据类型 | Python 版本 | 描述 |
| --- | --- | --- |
| [T2`deque`](https://docs.python.org/3/library/collections.html#collections.deque) | [2.4](https://docs.python.org/3/whatsnew/2.4.html#new-improved-and-deprecated-modules) | 一个类似序列的集合，支持从序列的任意一端有效地添加和移除项 |
| [T2`defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict) | [2.5](https://docs.python.org/3/whatsnew/2.5.html#new-improved-and-removed-modules) | 字典子类，用于为缺失的键构造默认值，并自动将它们添加到字典中 |
| [T2`namedtuple()`](https://docs.python.org/3/library/collections.html#collections.namedtuple) | [2.6](https://docs.python.org/3/whatsnew/2.6.html#new-and-improved-modules) | 一个用于创建`tuple`子类的工厂函数，提供命名字段，允许通过名称访问项目，同时保持通过索引访问项目的能力 |
| [T2`OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict) | [2.7](https://docs.python.org/3/whatsnew/2.7.html#new-and-improved-modules) ， [3.1](https://docs.python.org/3/whatsnew/3.1.html#pep-372-ordered-dictionaries) | 字典子类，根据插入键的时间保持键-值对的顺序 |
| [T2`Counter`](https://docs.python.org/3/library/collections.html#collections.Counter) | [2.7](https://docs.python.org/3/whatsnew/2.7.html#new-and-improved-modules) ， [3.1](https://docs.python.org/3/whatsnew/3.1.html#pep-372-ordered-dictionaries) | 字典子类，支持对序列或可重复项中的唯一项进行方便的计数 |
| [T2`ChainMap`](https://docs.python.org/3/library/collections.html#collections.ChainMap) | [3.3](https://docs.python.org/3/whatsnew/3.3.html#collections) | 一个类似字典的类，允许将多个映射作为单个字典对象处理 |

除了这些专门的数据类型，`collections`还提供了三个基类来帮助创建定制列表、字典和[字符串](https://realpython.com/python-strings/):

| 班级 | 描述 |
| --- | --- |
| [T2`UserDict`](https://realpython.com/inherit-python-dict/) | 围绕字典对象的包装类，便于子类化`dict` |
| [T2`UserList`](https://realpython.com/inherit-python-list/) | 围绕列表对象的包装类，便于子类化`list` |
| [T2`UserString`](https://realpython.com/inherit-python-str/) | 一个围绕字符串对象的包装类，便于子类化`string` |

对这些包装类的需求部分被相应的标准内置数据类型的子类化能力所掩盖。但是，有时使用这些类比使用标准数据类型更安全，也更不容易出错。

有了对`collections`的简要介绍以及本模块中的数据结构和类可以解决的具体用例，是时候更仔细地研究它们了。在此之前，需要指出的是，本教程整体上是对`collections`的介绍。在接下来的大部分章节中，您会发现一个蓝色的警告框，它会引导您找到关于这个类或函数的专门文章。

[*Remove ads*](/account/join/)

## 提高代码可读性:`namedtuple()`

Python 的`namedtuple()`是一个[工厂函数](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming))，允许你用**命名字段**创建`tuple`子类。这些字段使用**点符号**让您直接访问给定命名元组中的值，就像在`obj.attr`中一样。

之所以需要这个特性，是因为使用索引来访问常规元组中的值很烦人，难以阅读，而且容易出错。如果您正在处理的元组有几个项，并且是在远离您使用它的地方构造的，这一点尤其正确。

**注:**查看[使用 namedtuple](https://realpython.com/python-namedtuple/) 编写 Python 和 Clean 代码，深入了解如何在 Python 中使用`namedtuple`。

在 Python 2.6 中，开发人员可以用点符号访问带有命名字段的 tuple 子类，这似乎是一个理想的特性。这就是`namedtuple()`的由来。如果与常规元组相比，用这个函数构建的元组子类在代码可读性方面是一大优势。

为了正确看待代码可读性问题，考虑一下 [`divmod()`](https://docs.python.org/3/library/functions.html#divmod) 。这个内置函数接受两个(非复杂的)[数字](https://realpython.com/python-numbers/)，并返回一个元组，该元组具有输入值的**整数除法**的**商**和**余数**:

>>>

```py
>>> divmod(12, 5)
(2, 2)
```

它工作得很好。然而，这个结果是否具有可读性？你能说出输出中每个数字的含义吗？幸运的是，Python 提供了一种改进方法。您可以使用`namedtuple`编写带有显式结果的自定义版本的`divmod()`:

>>>

```py
>>> from collections import namedtuple

>>> def custom_divmod(x, y):
...     DivMod = namedtuple("DivMod", "quotient remainder")
...     return DivMod(*divmod(x, y))
...

>>> result = custom_divmod(12, 5)
>>> result
DivMod(quotient=2, remainder=2)

>>> result.quotient
2
>>> result.remainder
2
```

现在你知道结果中每个值的含义了。您还可以使用点符号和描述性字段名称来访问每个独立的值。

要使用`namedtuple()`创建新的 tuple 子类，需要两个必需的参数:

1.  **`typename`** 是您正在创建的类的名称。它必须是一个带有[有效 Python 标识符](https://doc.python.org/3/reference/lexical_analysis.html#identifiers)的字符串。
2.  **`field_names`** 是字段名列表，您将使用它来访问结果元组中的项目。它可以是:
    *   一个[可迭代的](https://docs.python.org/3/glossary.html#term-iterable)字符串，比如`["field1", "field2", ..., "fieldN"]`
    *   由空格分隔的字段名组成的字符串，例如`"field1 field2 ... fieldN"`
    *   用逗号分隔字段名的字符串，如`"field1, field2, ..., fieldN"`

例如，以下是使用`namedtuple()`创建具有两个坐标(`x`和`y`)的样本 2D `Point`的不同方法:

>>>

```py
>>> from collections import namedtuple

>>> # Use a list of strings as field names
>>> Point = namedtuple("Point", ["x", "y"])
>>> point = Point(2, 4)
>>> point
Point(x=2, y=4)

>>> # Access the coordinates
>>> point.x
2
>>> point.y
4
>>> point[0]
2

>>> # Use a generator expression as field names
>>> Point = namedtuple("Point", (field for field in "xy"))
>>> Point(2, 4)
Point(x=2, y=4)

>>> # Use a string with comma-separated field names
>>> Point = namedtuple("Point", "x, y")
>>> Point(2, 4)
Point(x=2, y=4)

>>> # Use a string with space-separated field names
>>> Point = namedtuple("Point", "x y")
>>> Point(2, 4)
Point(x=2, y=4)
```

在这些例子中，首先使用字段名的`list`创建`Point`。然后你实例化`Point`来制作一个`point`对象。请注意，您可以通过字段名和索引来访问`x`和`y`。

剩下的例子展示了如何用一串逗号分隔的字段名、[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)和一串空格分隔的字段名创建一个等价的命名元组。

命名元组还提供了一系列很酷的特性，允许您定义字段的默认值，从给定的命名元组创建字典，替换给定字段的值，等等:

>>>

```py
>>> from collections import namedtuple

>>> # Define default values for fields
>>> Person = namedtuple("Person", "name job", defaults=["Python Developer"])
>>> person = Person("Jane")
>>> person
Person(name='Jane', job='Python Developer')

>>> # Create a dictionary from a named tuple
>>> person._asdict()
{'name': 'Jane', 'job': 'Python Developer'}

>>> # Replace the value of a field
>>> person = person._replace(job="Web Developer")
>>> person
Person(name='Jane', job='Web Developer')
```

这里，首先使用`namedtuple()`创建一个`Person`类。这一次，您使用一个名为`defaults`的可选参数，它接受元组字段的一系列默认值。注意`namedtuple()`将默认值应用于最右边的字段。

在第二个例子中，您使用 [`._asdict()`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._asdict) 从现有的命名元组创建一个字典。该方法返回一个使用字段名作为键的新字典。

最后，你用 [`._replace()`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._replace) 替换`job`的原始值。这个方法不更新 tuple [的位置](https://en.wikipedia.org/wiki/In-place_algorithm)，而是返回一个新命名的 tuple，其新值存储在相应的字段中。你知道为什么`._replace()`返回一个新的命名元组吗？

[*Remove ads*](/account/join/)

## 构建高效的队列和堆栈:`deque`

Python 的 **`deque`** 是`collections`中第一个数据结构。这种类似序列的数据类型是对[堆栈](https://realpython.com/how-to-implement-python-stack/)和[队列](https://realpython.com/queue-in-python/)的概括，旨在支持数据结构两端的高效内存和快速**追加**和**弹出**操作。

**注:**字`deque`读作“deck”，代表**d**double-**e**endd**que**UE。

在 Python 中，在`list`对象的开头或左侧进行追加和弹出操作效率很低，时间复杂度[*O*(*n*](https://wiki.python.org/moin/TimeComplexity))。如果处理大型列表，这些操作的开销会特别大，因为 Python 必须将所有项目移到右边，以便在列表的开头插入新项目。

另一方面，列表右侧的 append 和 pop 操作通常是高效的( *O* (1))，除非 Python 需要重新分配内存来增加底层列表以接受新项。

Python 的`deque`就是为了克服这个问题而产生的。在一个`deque`对象两侧的追加和弹出操作是稳定的和同样有效的，因为 deques 被实现为一个[双向链表](https://realpython.com/linked-lists-python/#how-to-use-doubly-linked-lists)。这就是为什么 deques 对于创建堆栈和队列特别有用。

以一个队列为例。它以**先进/先出** ( [先进先出](https://en.wikipedia.org/wiki/FIFO_(computing_and_electronics)))的方式管理项目。它就像一个管道，你在管道的一端推入新的项目，从另一端弹出旧的项目。将一个项目添加到队列的末尾被称为**入队**操作。从队列的前面或开始处移除一个项目称为**出列**。

**注:**查看 [Python 的 dequee:implementing Efficient queue and Stacks](https://realpython.com/python-deque/)以深入探究如何在 Python 代码中使用`deque`。

现在假设你正在为一个排队买电影票的人建模。你可以用一个`deque`来做。每次有新人来，你就让他们排队。当排在队伍前面的人拿到票时，你让他们出队。

下面是如何使用一个`deque`对象来模拟这个过程:

>>>

```py
>>> from collections import deque

>>> ticket_queue = deque()
>>> ticket_queue
deque([])

>>> # People arrive to the queue
>>> ticket_queue.append("Jane")
>>> ticket_queue.append("John")
>>> ticket_queue.append("Linda")

>>> ticket_queue
deque(['Jane', 'John', 'Linda'])

>>> # People bought their tickets
>>> ticket_queue.popleft()
'Jane'
>>> ticket_queue.popleft()
'John'
>>> ticket_queue.popleft()
'Linda'

>>> # No people on the queue
>>> ticket_queue.popleft()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: pop from an empty deque
```

在这里，首先创建一个空的`deque`对象来表示人的队列。要让一个人入队，可以使用 [`.append()`](https://docs.python.org/3/library/collections.html#collections.deque.append) ，它将项目添加到队列的右端。要让一个人出列，可以使用 [`.popleft()`](https://docs.python.org/3/library/collections.html#collections.deque.popleft) ，它移除并返回队列左端的项目。

**注意:**在 Python 标准库中，你会找到 [`queue`](https://docs.python.org/3/library/queue.html#module-queue) 。该模块实现了多生产者、多消费者队列，有助于在多线程之间安全地交换信息。

`deque`初始化器有两个可选参数:

1.  **`iterable`** 持有一个作为初始化器的 iterable。
2.  **`maxlen`** 保存一个指定`deque`最大长度的[整数](https://realpython.com/python-numbers/#integers)。

如果你不提供一个`iterable`，那么你会得到一个空的队列。如果您为 [`maxlen`](https://docs.python.org/3/library/collections.html#collections.deque.maxlen) 提供一个值，那么您的 deque 将只存储最多`maxlen`个项目。

拥有一个`maxlen`是一个方便的特性。例如，假设您需要在一个应用程序中实现一个最近文件的列表。在这种情况下，您可以执行以下操作:

>>>

```py
>>> from collections import deque

>>> recent_files = deque(["core.py", "README.md", "__init__.py"], maxlen=3)

>>> recent_files.appendleft("database.py")
>>> recent_files
deque(['database.py', 'core.py', 'README.md'], maxlen=3)

>>> recent_files.appendleft("requirements.txt")
>>> recent_files
deque(['requirements.txt', 'database.py', 'core.py'], maxlen=3)
```

一旦 dequeue 达到其最大大小(本例中为三个文件)，在 dequeue 的一端添加新文件会自动丢弃另一端的文件。如果您不为`maxlen`提供一个值，那么 deque 可以增长到任意数量的项目。

到目前为止，您已经学习了 deques 的基本知识，包括如何创建 deques 以及如何从给定的 deques 的两端追加和弹出项目。Deques 通过类似列表的界面提供了一些额外的特性。以下是其中的一些:

>>>

```py
>>> from collections import deque

>>> # Use different iterables to create deques
>>> deque((1, 2, 3, 4))
deque([1, 2, 3, 4])

>>> deque([1, 2, 3, 4])
deque([1, 2, 3, 4])

>>> deque("abcd")
deque(['a', 'b', 'c', 'd'])

>>> # Unlike lists, deque doesn't support .pop() with arbitrary indices
>>> deque("abcd").pop(2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: pop() takes no arguments (1 given)

>>> # Extend an existing deque
>>> numbers = deque([1, 2])
>>> numbers.extend([3, 4, 5])
>>> numbers
deque([1, 2, 3, 4, 5])

>>> numbers.extendleft([-1, -2, -3, -4, -5])
>>> numbers
deque([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])

>>> # Insert an item at a given position
>>> numbers.insert(5, 0)
>>> numbers
deque([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
```

在这些例子中，您首先使用不同类型的 iterables 创建 deques 来初始化它们。`deque`和`list`的一个区别是`deque.pop()`不支持弹出给定索引处的项目。

注意，`deque`为`.append()`、[、`.pop()`、](https://docs.python.org/3/library/collections.html#collections.deque.pop)[、`.extend()`、](https://docs.python.org/3/library/collections.html#collections.deque.extend)提供了姊妹方法，并带有后缀`left`来表示它们在底层 deque 的左端执行相应的操作。

Deques 也支持序列操作:

| 方法 | 描述 |
| --- | --- |
| [T2`.clear()`](https://docs.python.org/3/library/collections.html#collections.deque.clear) | 从队列中删除所有元素 |
| [T2`.copy()`](https://docs.python.org/3/library/collections.html#collections.deque.copy) | 创建一个 deque 的浅层副本 |
| [T2`.count(x)`](https://docs.python.org/3/library/collections.html#collections.deque.count) | 计算等于`x`的双队列元素的数量 |
| [T2`.remove(value)`](https://docs.python.org/3/library/collections.html#collections.deque.remove) | 删除第一次出现的`value` |

deques 的另一个有趣的特性是能够使用`.rotate()`旋转它们的元素:

>>>

```py
>>> from collections import deque

>>> ordinals = deque(["first", "second", "third"])
>>> ordinals.rotate()
>>> ordinals
deque(['third', 'first', 'second'])

>>> ordinals.rotate(2)
>>> ordinals
deque(['first', 'second', 'third'])

>>> ordinals.rotate(-2)
>>> ordinals
deque(['third', 'first', 'second'])

>>> ordinals.rotate(-1)
>>> ordinals
deque(['first', 'second', 'third'])
```

该方法向右旋转 deque `n`步骤。`n`的默认值为`1`。如果给`n`提供一个负值，那么旋转向左。

最后，您可以使用索引来访问 dequee 中的元素，但是您不能对 dequee 进行切片:

>>>

```py
>>> from collections import deque

>>> ordinals = deque(["first", "second", "third"])
>>> ordinals[1]
'second'

>>> ordinals[0:2]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: sequence index must be integer, not 'slice'
```

Deques 支持索引，但有趣的是，它们不支持切片。当您试图从现有的队列中检索一个切片时，您会得到一个`TypeError`。这是因为在链表上执行切片操作是低效的，所以该操作不可用。

[*Remove ads*](/account/join/)

## 处理丢失的按键:`defaultdict`

当你在 Python 中使用[字典](https://realpython.com/python-dicts/)时，你会面临的一个常见问题是如何处理丢失的键。如果您试图访问一个给定字典中不存在的键，那么您会得到一个`KeyError`:

>>>

```py
>>> favorites = {"pet": "dog", "color": "blue", "language": "Python"}

>>> favorites["fruit"]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'fruit'
```

有几种方法可以解决这个问题。比如可以用 [`.setdefault()`](https://docs.python.org/3/library/stdtypes.html#dict.setdefault) 。该方法将一个键作为参数。如果字典中存在该键，那么它将返回相应的值。否则，该方法插入该键，为其赋一个默认值，并返回该值:

>>>

```py
>>> favorites = {"pet": "dog", "color": "blue", "language": "Python"}

>>> favorites.setdefault("fruit", "apple")
'apple'

>>> favorites
{'pet': 'dog', 'color': 'blue', 'language': 'Python', 'fruit': 'apple'}

>>> favorites.setdefault("pet", "cat")
'dog'

>>> favorites
{'pet': 'dog', 'color': 'blue', 'language': 'Python', 'fruit': 'apple'}
```

在这个例子中，您使用`.setdefault()`为`fruit`生成一个默认值。由于这个键在`favorites`中不存在，`.setdefault()`创建了它并赋予它`apple`的值。如果你用一个存在的键调用`.setdefault()`，那么这个调用不会影响字典，你的键将保持原始值而不是默认值。

如果给定的键丢失，您也可以使用`.get()`返回一个合适的默认值:

>>>

```py
>>> favorites = {"pet": "dog", "color": "blue", "language": "Python"}

>>> favorites.get("fruit", "apple")
'apple'

>>> favorites
{'pet': 'dog', 'color': 'blue', 'language': 'Python'}
```

这里，`.get()`返回`apple`,因为底层字典中缺少该键。然而，`.get()`并没有为你创建新的密匙。

由于处理字典中丢失的键是一种常见的需求，Python 的`collections`也为此提供了一个工具。`defaultdict`类型是`dict`的子类，旨在帮助你解决丢失的键。

**注意:**查看[使用 Python defaultdict 类型处理丢失的键](https://realpython.com/python-defaultdict/)，深入了解如何使用 Python 的`defaultdict`。

`defaultdict`的构造函数将一个函数对象作为它的第一个参数。当您访问一个不存在的键时，`defaultdict`自动调用该函数，不带参数，为手边的键创建一个合适的默认值。

为了提供其功能，`defaultdict`将输入函数存储在 [`.default_factory`](https://docs.python.org/3/library/collections.html#collections.defaultdict.default_factory) 中，然后覆盖 [`.__missing__()`](https://docs.python.org/3/library/collections.html#collections.defaultdict.__missing__) 以在您访问任何丢失的键时自动调用该函数并生成默认值。

你可以使用任何可调用来初始化你的`defaultdict`对象。例如，使用 [`int()`](https://docs.python.org/3/library/functions.html#int) 您可以创建一个合适的**计数器**来计数不同的对象:

>>>

```py
>>> from collections import defaultdict

>>> counter = defaultdict(int)
>>> counter
defaultdict(<class 'int'>, {})
>>> counter["dogs"]
0
>>> counter
defaultdict(<class 'int'>, {'dogs': 0})

>>> counter["dogs"] += 1
>>> counter["dogs"] += 1
>>> counter["dogs"] += 1
>>> counter["cats"] += 1
>>> counter["cats"] += 1
>>> counter
defaultdict(<class 'int'>, {'dogs': 3, 'cats': 2})
```

在本例中，您创建了一个空的`defaultdict`，将`int()`作为它的第一个参数。当你访问一个不存在的键时，字典自动调用`int()`，它返回`0`作为当前键的默认值。这种`defaultdict`对象在 Python 中计数时非常有用。

`defaultdict`的另一个常见用例是将事物分组。在这种情况下，方便的工厂函数是`list()`:

>>>

```py
>>> from collections import defaultdict

>>> pets = [
...     ("dog", "Affenpinscher"),
...     ("dog", "Terrier"),
...     ("dog", "Boxer"),
...     ("cat", "Abyssinian"),
...     ("cat", "Birman"),
... ]

>>> group_pets = defaultdict(list)

>>> for pet, breed in pets:
...     group_pets[pet].append(breed)
...

>>> for pet, breeds in group_pets.items():
...     print(pet, "->", breeds)
...
dog -> ['Affenpinscher', 'Terrier', 'Boxer']
cat -> ['Abyssinian', 'Birman']
```

在这个例子中，您有关于宠物及其品种的原始数据，您需要按照宠物对它们进行分组。为此，在创建`defaultdict`实例时，使用`list()`作为`.default_factory`。这使您的字典能够自动创建一个空列表(`[]`)作为您访问的每个缺失键的默认值。然后你用这个列表来存储你的宠物的品种。

最后，你应该注意到由于`defaultdict`是`dict`的子类，它提供了相同的接口。这意味着你可以像使用普通字典一样使用你的`defaultdict`对象。

[*Remove ads*](/account/join/)

## 保持字典有序:`OrderedDict`

有时，您需要字典来记住键值对的插入顺序。多年来，Python 的常规[字典](https://realpython.com/iterate-through-dictionary-python/#a-few-words-on-dictionaries)是*无序的*数据结构[。所以，回到 2008 年，](https://realpython.com/python-data-structures/) [PEP 372](https://www.python.org/dev/peps/pep-0372/) 引入了给`collections`添加一个新字典类的想法。

新的类会根据钥匙插入的时间记住项目的顺序。这就是 [`OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict) 的由来。

`OrderedDict`在 [Python 3.1](https://docs.python.org/3/whatsnew/3.1.html) 中引入。其应用编程接口(API)与`dict`基本相同。然而，`OrderedDict`按照键被第一次插入字典的顺序遍历键和值。如果为现有键分配一个新值，则键-值对的顺序保持不变。如果一个条目被删除并重新插入，那么它将被移动到字典的末尾。

**注:**查看[Python 中的 OrderedDict vs dict:工作的正确工具](https://realpython.com/python-ordereddict/)以深入了解 Python 的`OrderedDict`以及为什么应该考虑使用它。

有几种方法可以创建`OrderedDict`对象。它们中的大多数与你如何创建一个普通的字典是一样的。例如，您可以通过实例化不带参数的类来创建一个空的有序字典，然后根据需要插入键值对:

>>>

```py
>>> from collections import OrderedDict

>>> life_stages = OrderedDict()

>>> life_stages["childhood"] = "0-9"
>>> life_stages["adolescence"] = "9-18"
>>> life_stages["adulthood"] = "18-65"
>>> life_stages["old"] = "+65"

>>> for stage, years in life_stages.items():
...     print(stage, "->", years)
...
childhood -> 0-9
adolescence -> 9-18
adulthood -> 18-65
old -> +65
```

在这个例子中，您通过实例化不带参数的`OrderedDict`来创建一个空的有序字典。接下来，像处理常规字典一样，将键值对添加到字典中。

当您[遍历字典](https://realpython.com/iterate-through-dictionary-python/)、`life_stages`时，您将获得键-值对，其顺序与您将它们插入字典的顺序相同。保证物品的顺序是`OrderedDict`解决的主要问题。

Python 3.6 引入了一个[的新实现`dict`](https://docs.python.org/3/whatsnew/3.6.html#new-dict-implementation) 。这种实现提供了一个意想不到的新特性:现在普通字典按照它们第一次插入的顺序保存它们的条目。

最初，这个特性被认为是一个实现细节，文档建议不要依赖它。然而，自从 [Python 3.7](https://realpython.com/python37-new-features/) ，[特性](https://mail.python.org/pipermail/python-dev/2017-December/151283.html)正式成为语言规范的一部分。那么，用`OrderedDict`有什么意义呢？

`OrderedDict`的一些特性仍然让它很有价值:

1.  **意图传达:**有了`OrderedDict`，你的代码会清楚的表明字典中条目的顺序很重要。你清楚地表达了你的代码需要或者依赖于底层字典中的条目顺序。
2.  **对条目顺序的控制:**使用`OrderedDict`，您可以访问 [`.move_to_end()`](https://docs.python.org/3/library/collections.html#collections.OrderedDict.move_to_end) ，这是一种允许您操纵字典中条目顺序的方法。您还将拥有一个增强的 [`.popitem()`](https://docs.python.org/3/library/collections.html#collections.OrderedDict.popitem) 变体，允许从底层字典的任意一端移除条目。
3.  **相等性测试行为:**使用`OrderedDict`，字典之间的相等性测试会考虑条目的顺序。因此，如果您有两个有序的字典，它们包含相同的条目组，但顺序不同，那么您的字典将被认为是不相等的。

使用`OrderedDict` : **向后兼容**至少还有一个原因。在运行 than 3.6 之前版本的环境中，依靠常规的`dict`对象来保持项目的顺序会破坏您的代码。

好了，现在是时候看看`OrderedDict`的一些很酷的功能了:

>>>

```py
>>> from collections import OrderedDict

>>> letters = OrderedDict(b=2, d=4, a=1, c=3)
>>> letters
OrderedDict([('b', 2), ('d', 4), ('a', 1), ('c', 3)])

>>> # Move b to the right end
>>> letters.move_to_end("b")
>>> letters
OrderedDict([('d', 4), ('a', 1), ('c', 3), ('b', 2)])

>>> # Move b to the left end
>>> letters.move_to_end("b", last=False)
>>> letters
OrderedDict([('b', 2), ('d', 4), ('a', 1), ('c', 3)])

>>> # Sort letters by key
>>> for key in sorted(letters):
...     letters.move_to_end(key)
...

>>> letters
OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
```

在这些例子中，您使用 [`.move_to_end()`](https://docs.python.org/3/library/collections.html#collections.OrderedDict.move_to_end) 来移动项目并重新排序`letters`。注意，`.move_to_end()`接受了一个名为`last`的可选参数，它允许您控制想要将条目移动到词典的哪一端。当您需要对词典中的条目进行排序或者需要以任何方式操纵它们的顺序时，这种方法非常方便。

`OrderedDict`和普通词典的另一个重要区别是它们如何比较相等性:

>>>

```py
>>> from collections import OrderedDict

>>> # Regular dictionaries compare the content only
>>> letters_0 = dict(a=1, b=2, c=3, d=4)
>>> letters_1 = dict(b=2, a=1, d=4, c=3)
>>> letters_0 == letters_1
True

>>> # Ordered dictionaries compare content and order
>>> letters_0 = OrderedDict(a=1, b=2, c=3, d=4)
>>> letters_1 = OrderedDict(b=2, a=1, d=4, c=3)
>>> letters_0 == letters_1
False

>>> letters_2 = OrderedDict(a=1, b=2, c=3, d=4)
>>> letters_0 == letters_2
True
```

这里，`letters_1`的项目顺序与`letters_0`不同。当你使用普通的字典时，这种差异并不重要，两种字典比较起来是一样的。另一方面，当你使用有序字典时，`letters_0`和`letters_1`并不相等。这是因为有序字典之间的相等测试考虑了内容以及条目的顺序。

[*Remove ads*](/account/join/)

## 一气呵成清点物体:`Counter`

对象计数是编程中常见的操作。假设你需要计算一个给定的条目在列表或 iterable 中出现了多少次。如果你的清单很短，那么计算清单上的项目会很简单快捷。如果你有一个很长的清单，那么计算清单会更有挑战性。

为了计数对象，你通常使用一个**计数器**，或者一个初始值为零的整数[变量](https://realpython.com/python-variables/)。然后递增计数器以反映给定对象出现的次数。

在 Python 中，你可以使用字典一次计算几个不同的对象。在这种情况下，键将存储单个对象，值将保存给定对象的重复次数，或对象的**计数**。

这里有一个例子，用一个普通的字典和一个 [`for`循环](https://realpython.com/python-for-loop/)来计算单词`"mississippi"`中的字母:

>>>

```py
>>> word = "mississippi"
>>> counter = {}

>>> for letter in word:
...     if letter not in counter:
...         counter[letter] = 0
...     counter[letter] += 1
...

>>> counter
{'m': 1, 'i': 4, 's': 4, 'p': 2}
```

循环遍历`word`中的字母。[条件语句](https://realpython.com/python-conditional-statements/)检查字母是否已经在字典中，并相应地将字母的计数初始化为零。最后一步是随着循环的进行增加字母的计数。

正如你已经知道的，`defaultdict` objects 在计数的时候很方便，因为你不需要检查键是否存在。字典保证任何丢失的键都有适当的默认值:

>>>

```py
>>> from collections import defaultdict

>>> counter = defaultdict(int)

>>> for letter in "mississippi":
...     counter[letter] += 1
...

>>> counter
defaultdict(<class 'int'>, {'m': 1, 'i': 4, 's': 4, 'p': 2})
```

在本例中，您创建了一个`defaultdict`对象，并使用`int()`对其进行初始化。使用`int()`作为工厂函数，底层默认字典会自动创建缺失的键，并方便地将其初始化为零。然后增加当前键的值来计算`"mississippi"`中字母的最终计数。

就像其他常见的编程问题一样，Python 也有一个处理计数问题的有效工具。在`collections`中，你会发现 [`Counter`](https://docs.python.org/3/library/collections.html#collections.Counter) ，这是一个专门为计数对象设计的`dict`子类。

以下是使用`Counter`编写`"mississippi"`示例的方法:

>>>

```py
>>> from collections import Counter

>>> Counter("mississippi")
Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})
```

哇！真快！一行代码就完成了。在这个例子中，`Counter`遍历`"mississippi"`，生成一个字典，将字母作为键，将它们的频率作为值。

**注:**查看 [Python 的计数器:计算对象的 Python 方式](https://realpython.com/python-counter/)深入了解`Counter`以及如何使用它高效地计算对象。

有几种不同的方法来实例化`Counter`。您可以使用[列表、元组](https://realpython.com/python-lists-tuples/)或任何具有重复对象的 iterables。唯一的限制是你的对象必须是可散列的 T4:

>>>

```py
>>> from collections import Counter

>>> Counter([1, 1, 2, 3, 3, 3, 4])
Counter({3: 3, 1: 2, 2: 1, 4: 1})

>>> Counter(([1], [1]))
Traceback (most recent call last):
  ...
TypeError: unhashable type: 'list'
```

整数是可散列的，所以`Counter`可以正常工作。另一方面，列表是不可散列的，所以`Counter`以一个`TypeError`失败。

被**哈希化**意味着你的对象必须有一个**哈希值**，在它们的生命周期中不会改变。这是一个要求，因为这些对象将作为字典键工作。在 Python 中，[不可变的](https://docs.python.org/3/glossary.html#term-immutable)对象也是可散列的。

**注:`Counter`中的**，经过高度优化的 [C 函数](https://github.com/python/cpython/blob/73b20ae2fb7a5c1374aa5c3719f64c53d29fa0d2/Modules/_collectionsmodule.c#L2307)提供计数功能。如果这个函数由于某种原因不可用，那么这个类使用一个等效的但是效率较低的 [Python 函数](https://github.com/python/cpython/blob/6f1e8ccffa5b1272a36a35405d3c4e4bbba0c082/Lib/collections/__init__.py#L503)。

由于`Counter`是`dict`的子类，所以它们的接口大多相同。但是，也有一些微妙的区别。第一个区别是`Counter`没有实现 [`.fromkeys()`](https://docs.python.org/3/library/collections.html#collections.Counter.fromkeys) 。这避免了不一致，比如`Counter.fromkeys("abbbc", 2)`，其中每个字母都有一个初始计数`2`，而不管它在输入 iterable 中的实际计数。

第二个区别是 [`.update()`](https://docs.python.org/3/library/collections.html#collections.Counter.update) 不会用新的计数替换现有对象(键)的计数(值)。它将两个计数相加:

>>>

```py
>>> from collections import Counter

>>> letters = Counter("mississippi")
>>> letters
Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})

>>> # Update the counts of m and i
>>> letters.update(m=3, i=4)
>>> letters
Counter({'i': 8, 'm': 4, 's': 4, 'p': 2})

>>> # Add a new key-count pair
>>> letters.update({"a": 2})
>>> letters
Counter({'i': 8, 'm': 4, 's': 4, 'p': 2, 'a': 2})

>>> # Update with another counter
>>> letters.update(Counter(["s", "s", "p"]))
>>> letters
Counter({'i': 8, 's': 6, 'm': 4, 'p': 3, 'a': 2})
```

在这里，您更新了`m`和`i`的计数。现在这些字母保存了它们初始计数的总和加上你通过`.update()`传递给它们的值。如果您使用一个不存在于原始计数器中的键，那么`.update()`会用相应的值创建一个新的键。最后，`.update()`接受可重复项、映射、关键字参数以及其他计数器。

**注意:**因为`Counter`是`dict`的一个子类，所以对于您可以在计数器的键和值中存储的对象没有限制。键可以存储任何可散列的对象，而值可以存储任何对象。但是，为了在逻辑上作为计数器工作，这些值应该是表示计数的整数。

`Counter`和`dict`的另一个区别是，访问丢失的键会返回`0`，而不是引发`KeyError`:

>>>

```py
>>> from collections import Counter

>>> letters = Counter("mississippi")
>>> letters["a"]
0
```

这种行为表明计数器中不存在的对象的计数为零。在这个例子中，字母`"a"`不在原始单词中，所以它的计数是`0`。

在 Python 中，`Counter`也可以用来模拟一个[多重集](https://en.wikipedia.org/wiki/Multiset)或**包**。多重集类似于[集](https://en.wikipedia.org/wiki/Set_(mathematics))，但是它们允许给定元素的多个实例。一个元素的实例数量被称为它的**多重性**。例如，您可以有一个类似{1，1，2，3，3，3，4，4}的多重集。

当您使用`Counter`来模拟多重集时，键代表元素，值代表它们各自的多重性:

>>>

```py
>>> from collections import Counter

>>> multiset = Counter({1, 1, 2, 3, 3, 3, 4, 4})
>>> multiset
Counter({1: 1, 2: 1, 3: 1, 4: 1})

>>> multiset.keys() == {1, 2, 3, 4}
True
```

在这里，`multiset`的键相当于一个 Python 集合。这些值包含集合中每个元素的多重性。

Python' `Counter`'提供了一些额外的特性，帮助您将它们作为多重集来使用。例如，您可以用元素及其多重性的映射来初始化您的计数器。您还可以对元素的多重性执行数学运算等等。

假设你在当地的宠物收容所工作。你有一定数量的宠物，你需要记录每天有多少宠物被收养，有多少宠物进出收容所。在这种情况下，可以使用`Counter`:

>>>

```py
>>> from collections import Counter

>>> inventory = Counter(dogs=23, cats=14, pythons=7)

>>> adopted = Counter(dogs=2, cats=5, pythons=1)
>>> inventory.subtract(adopted)
>>> inventory
Counter({'dogs': 21, 'cats': 9, 'pythons': 6})

>>> new_pets = {"dogs": 4, "cats": 1}
>>> inventory.update(new_pets)
>>> inventory
Counter({'dogs': 25, 'cats': 10, 'pythons': 6})

>>> inventory = inventory - Counter(dogs=2, cats=3, pythons=1)
>>> inventory
Counter({'dogs': 23, 'cats': 7, 'pythons': 5})

>>> new_pets = {"dogs": 4, "pythons": 2}
>>> inventory += new_pets
>>> inventory
Counter({'dogs': 27, 'cats': 7, 'pythons': 7})
```

太棒了！现在你可以用`Counter`记录你的宠物了。请注意，您可以使用`.subtract()`和`.update()`来加减计数或重数。您也可以使用加法(`+`)和减法(`-`)运算符。

在 Python 中，您可以将`Counter`对象作为多重集来做更多的事情，所以请大胆尝试吧！

[*Remove ads*](/account/join/)

## 将字典链接在一起:`ChainMap`

Python 的`ChainMap`将多个字典和其他映射组合在一起，创建一个单一对象，其工作方式非常类似于常规字典。换句话说，它接受几个映射，并使它们在逻辑上表现为一个映射。

`ChainMap`对象是**可更新的视图**，这意味着任何链接映射的变化都会影响到整个`ChainMap`对象。这是因为`ChainMap`没有将输入映射合并在一起。它保留了一个映射列表，并在该列表的顶部重新实现了公共字典操作。例如，关键字查找会连续搜索映射列表，直到找到该关键字。

**注意:**查看 [Python 的 ChainMap:有效管理多个上下文](https://realpython.com/python-chainmap/)，深入了解如何在 Python 代码中使用`ChainMap`。

当你使用`ChainMap`对象时，你可以有几个字典，或者是唯一的或者是重复的键。

无论哪种情况，`ChainMap`都允许您将所有的字典视为一个字典。如果您的字典中有唯一的键，您可以像使用单个字典一样访问和更新这些键。

如果您的字典中有重复的键，除了将字典作为一个字典管理之外，您还可以利用内部映射列表来定义某种类型的**访问优先级**。由于这个特性，`ChainMap`对象非常适合处理多种上下文。

例如，假设您正在开发一个[命令行界面(CLI)](https://en.wikipedia.org/wiki/Command-line_interface) 应用程序。该应用程序允许用户使用代理服务连接到互联网。设置优先级包括:

1.  命令行选项(`--proxy`、`-p`)
2.  用户主目录中的本地配置文件
3.  全局代理配置

如果用户在命令行提供代理，那么应用程序必须使用该代理。否则，应用程序应该使用下一个配置对象中提供的代理，依此类推。这是`ChainMap`最常见的用例之一。在这种情况下，您可以执行以下操作:

>>>

```py
>>> from collections import ChainMap

>>> cmd_proxy = {}  # The user doesn't provide a proxy
>>> local_proxy = {"proxy": "proxy.local.com"}
>>> global_proxy = {"proxy": "proxy.global.com"}

>>> config = ChainMap(cmd_proxy, local_proxy, global_proxy)
>>> config["proxy"]
'proxy.local.com'
```

`ChainMap`允许您为应用程序的代理配置定义适当的优先级。一个键查找搜索`cmd_proxy`，然后是`local_proxy`，最后是`global_proxy`，返回当前键的第一个实例。在这个例子中，用户没有在命令行提供代理，所以您的应用程序使用了`local_proxy`中的代理。

一般来说，`ChainMap`对象的行为类似于常规的`dict`对象。但是，它们还有一些附加功能。例如，它们有一个保存内部映射列表的 [`.maps`](https://docs.python.org/3/library/collections.html#collections.ChainMap.maps) 公共属性:

>>>

```py
>>> from collections import ChainMap

>>> numbers = {"one": 1, "two": 2}
>>> letters = {"a": "A", "b": "B"}

>>> alpha_nums = ChainMap(numbers, letters)
>>> alpha_nums.maps
[{'one': 1, 'two': 2}, {'a': 'A', 'b': 'B'}]
```

实例属性`.maps`允许您访问内部映射列表。该列表可更新。您可以手动添加和删除映射，遍历列表，等等。

另外，`ChainMap`提供了一个 [`.new_child()`](https://docs.python.org/3/library/collections.html#collections.ChainMap.new_child) 方法和一个 [`.parents`](https://docs.python.org/3/library/collections.html#collections.ChainMap.parents) 属性:

>>>

```py
>>> from collections import ChainMap

>>> dad = {"name": "John", "age": 35}
>>> mom = {"name": "Jane", "age": 31}
>>> family = ChainMap(mom, dad)
>>> family
ChainMap({'name': 'Jane', 'age': 31}, {'name': 'John', 'age': 35})

>>> son = {"name": "Mike", "age": 0}
>>> family = family.new_child(son)

>>> for person in family.maps:
...     print(person)
...
{'name': 'Mike', 'age': 0}
{'name': 'Jane', 'age': 31}
{'name': 'John', 'age': 35}

>>> family.parents
ChainMap({'name': 'Jane', 'age': 31}, {'name': 'John', 'age': 35})
```

使用`.new_child()`，您创建一个新的`ChainMap`对象，包含一个新的地图(`son`)，后跟当前实例中的所有地图。作为第一个参数传递的映射成为映射列表中的第一个映射。如果没有传递 map，那么这个方法使用一个空字典。

`parents`属性返回一个新的`ChainMap`对象，包含当前实例中除第一个以外的所有地图。当您需要在键查找中跳过第一个映射时，这很有用。

在`ChainMap`中要强调的最后一个特性是变异操作，比如更新键、添加新键、删除现有键、弹出键和清除字典，作用于内部映射列表中的第一个映射:

>>>

```py
>>> from collections import ChainMap

>>> numbers = {"one": 1, "two": 2}
>>> letters = {"a": "A", "b": "B"}

>>> alpha_nums = ChainMap(numbers, letters)
>>> alpha_nums
ChainMap({'one': 1, 'two': 2}, {'a': 'A', 'b': 'B'})

>>> # Add a new key-value pair
>>> alpha_nums["c"] = "C"
>>> alpha_nums
ChainMap({'one': 1, 'two': 2, 'c': 'C'}, {'a': 'A', 'b': 'B'})

>>> # Pop a key that exists in the first dictionary
>>> alpha_nums.pop("two")
2
>>> alpha_nums
ChainMap({'one': 1, 'c': 'C'}, {'a': 'A', 'b': 'B'})

>>> # Delete keys that don't exist in the first dict but do in others
>>> del alpha_nums["a"]
Traceback (most recent call last):
  ...
KeyError: "Key not found in the first mapping: 'a'"

>>> # Clear the dictionary
>>> alpha_nums.clear()
>>> alpha_nums
ChainMap({}, {'a': 'A', 'b': 'B'})
```

这些例子表明对一个`ChainMap`对象的变异操作只影响内部列表中的第一个映射。当您使用`ChainMap`时，这是一个需要考虑的重要细节。

棘手的是，乍一看，在给定的`ChainMap`中，任何现有的键值对都有可能发生变异。但是，您只能改变第一个映射中的键-值对，除非您使用`.maps`来直接访问和改变列表中的其他映射。

[*Remove ads*](/account/join/)

## 自定义内置:`UserString`、`UserList`和`UserDict`T3

有时您需要定制内置类型，如字符串、列表和字典，以添加和修改某些行为。从 [Python 2.2](https://docs.python.org/3/whatsnew/2.2.html#peps-252-and-253-type-and-class-changes) 开始，你可以通过直接子类化这些类型来实现。但是，这种方法可能会遇到一些问题，您马上就会看到。

Python 的`collections`提供了三个方便的包装类，模拟内置数据类型的行为:

1.  `UserString`
2.  `UserList`
3.  `UserDict`

通过常规和特殊方法的组合，您可以使用这些类来模拟和定制字符串、列表和字典的行为。

现在，开发人员经常问自己，当他们需要定制内置类型的行为时，是否有理由使用`UserString`、`UserList`和`UserDict`。答案是肯定的。

考虑到[的开闭原则](https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle)，内置类型[被设计和实现](https://www.youtube.com/watch?v=heJuQWNdwJI)。这意味着它们对扩展开放，但对修改关闭。允许修改这些类的核心特性可能会破坏它们的[不变量](https://en.wikipedia.org/wiki/Invariant_(mathematics)#Invariants_in_computer_science)。因此，Python 核心开发人员决定保护它们不被修改。

例如，假设您需要一个字典，当您插入键时，它会自动小写。您可以子类化`dict`并覆盖 [`.__setitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__setitem__) ，这样每当您插入一个键时，字典就会小写这个键名:

>>>

```py
>>> class LowerDict(dict):
...     def __setitem__(self, key, value):
...         key = key.lower()
...         super().__setitem__(key, value)
...

>>> ordinals = LowerDict({"FIRST": 1, "SECOND": 2})
>>> ordinals["THIRD"] = 3
>>> ordinals.update({"FOURTH": 4})

>>> ordinals
{'FIRST': 1, 'SECOND': 2, 'third': 3, 'FOURTH': 4}

>>> isinstance(ordinals, dict)
True
```

当您使用带有方括号(`[]`)的字典样式赋值来插入新键时，该字典可以正常工作。然而，当你将一个初始字典传递给[类构造函数](https://realpython.com/python-class-constructor/)或者当你使用 [`.update()`](https://docs.python.org/3/library/stdtypes.html#dict.update) 时，它不起作用。这意味着您需要覆盖[`.__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__)`.update()`，可能还有其他一些方法来让您的自定义词典正确工作。

现在看一下同样的字典，但是使用`UserDict`作为基类:

>>>

```py
>>> from collections import UserDict

>>> class LowerDict(UserDict):
...     def __setitem__(self, key, value):
...         key = key.lower()
...         super().__setitem__(key, value)
...

>>> ordinals = LowerDict({"FIRST": 1, "SECOND": 2})
>>> ordinals["THIRD"] = 3
>>> ordinals.update({"FOURTH": 4})

>>> ordinals
{'first': 1, 'second': 2, 'third': 3, 'fourth': 4}

>>> isinstance(ordinals, dict)
False
```

有用！您的自定义词典现在会在将所有新键插入词典之前将其转换为小写字母。注意，因为你不直接从`dict`继承，你的类不像上面的例子那样返回`dict`的实例。

`UserDict`在名为`.data`的实例属性中存储一个常规字典。然后，它围绕该字典实现它的所有方法。`UserList`和`UserString`工作方式相同，但是它们的`.data`属性分别拥有一个`list`和一个`str`对象。

如果您需要定制这些类中的任何一个，那么您只需要覆盖适当的方法并根据需要更改它们的功能。

一般来说，当您需要一个行为与底层包装内置类几乎相同的类，并且您想要定制其标准功能的某个部分时，您应该使用`UserDict`、`UserList`和`UserString`。

使用这些类而不是内置的等价类的另一个原因是访问底层的`.data`属性来直接操作它。

直接从内置类型继承的能力已经在很大程度上取代了`UserDict`、`UserList`和`UserString`的使用。然而，内置类型的内部实现使得在不重写大量代码的情况下很难安全地从它们继承。在大多数情况下，使用`collections`中合适的类更安全。这会让你避免一些问题和奇怪的行为。

## 结论

在 Python 的`collections`模块中，有几个**专门的容器数据类型**，可以用来处理常见的编程问题，比如计算对象数量、创建队列和堆栈、处理字典中丢失的键等等。

`collections`中的数据类型和类被设计成高效和 Pythonic 化的。它们对您的 Python 编程之旅非常有帮助，因此了解它们非常值得您花费时间和精力。

**在本教程中，您学习了如何:**

*   使用`namedtuple`编写**可读的**和**显式的**代码
*   使用`deque`构建**高效队列**和**堆栈**
*   **使用`Counter`有效地计数对象**
*   用`defaultdict`处理**缺失的字典键**
*   记住`OrderedDict`键的**插入顺序**
*   **用`ChainMap`在单个视图中链接多个字典**

您还了解了三个方便的包装器类:`UserDict`、`UserList`和`UserString`。当您需要创建模拟内置类型`dict`、`list`和`str`的行为的定制类时，这些类非常方便。*******