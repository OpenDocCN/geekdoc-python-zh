# 用 namedtuple 编写 Pythonic 式的干净代码

> 原文：<https://realpython.com/python-namedtuple/>

Python 的 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) 模块提供了一个名为 [`namedtuple()`](https://docs.python.org/3/library/collections.html#collections.namedtuple) 的[工厂函数](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming))，专门用来让你在处理元组的时候代码更加**python 化**。使用`namedtuple()`，您可以创建[不可变的](https://docs.python.org/3/glossary.html#term-immutable)序列类型，允许您使用描述性的字段名和**点符号**而不是模糊的整数索引来访问它们的值。

如果您有一些使用 Python 的经验，那么您应该知道编写 Python 代码是 Python 开发人员的核心技能。在本教程中，您将使用`namedtuple`提升该技能。

**在本教程中，您将学习如何:**

*   使用 **`namedtuple()`** 创建`namedtuple`类
*   识别并利用**的酷功能`namedtuple`的**
*   使用`namedtuple`实例编写**python 代码**
*   决定是使用`namedtuple`还是**类似的数据结构**
*   **子类** a `namedtuple`提供新特性

为了从本教程中获得最大收益，您需要对与编写 Python 可读代码相关的 Python 哲学有一个大致的了解。您还需要了解使用的基本知识:

*   [元组](https://realpython.com/python-lists-tuples/)
*   [字典](https://realpython.com/python-dicts/)
*   [类和面向对象编程](https://realpython.com/python3-object-oriented-programming/)
*   [数据类别](https://realpython.com/python-data-classes/)
*   [键入提示](https://realpython.com/python-type-checking/)

如果在开始本教程之前，您还没有掌握所有必需的知识，那也没关系！可以根据需要停下来复习一下以上资源。

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 使用`namedtuple`编写 Pythonic 代码

Python 的 [`namedtuple()`](https://docs.python.org/3/library/collections.html#collections.namedtuple) 是 [`collections`](https://docs.python.org/3/library/collections.html#module-collections) 中可用的一个[工厂函数](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming))。它允许你用**命名字段**创建`tuple`子类。您可以使用**点符号**和字段名来访问给定命名元组中的值，就像在`obj.attr`中一样。

Python 的`namedtuple`是为了提高代码可读性而创建的，它提供了一种使用描述性字段名称而不是整数索引来访问值的方法，在大多数情况下，整数索引不提供任何关于值是什么的上下文。这个特性也使得代码更干净，更易于维护。

相比之下，对常规元组中的值使用索引可能会令人讨厌、难以阅读并且容易出错。如果 tuple 有很多字段，并且是在远离使用它的地方构造的，这一点尤其正确。

**注意:**在本教程中，你会发现不同的术语用来指代 Python 的`namedtuple`，它的工厂函数，以及它的实例。

为了避免混淆，这里总结了在整个教程中如何使用每个术语:

| 学期 | 意义 |
| --- | --- |
| `namedtuple()` | 工厂功能 |
| `namedtuple`、`namedtuple`类 | `namedtuple()`返回的元组子类 |
| `namedtuple`实例，命名元组 | 特定`namedtuple`类的实例 |

你会发现这些术语在整个教程中都有相应的含义。

除了命名元组的这个主要特性之外，您会发现它们:

*   **是不可变的**数据结构吗
*   具有一致的[哈希](https://docs.python.org/3/library/functions.html#hash)值
*   可以作为**字典键**
*   可以存储在[组](https://realpython.com/python-sets/)中
*   根据类型和字段名创建一个有用的[文档串](https://realpython.com/documenting-python-code/)
*   提供一个有用的**字符串表示**，以`name=value`格式打印元组内容
*   支持**分度**
*   提供附加的方法和属性，如 [`._make()`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._make) ， [`_asdict()`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._asdict) ， [`._fields`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._fields) 等等
*   **与常规元组**向后兼容吗
*   有**个与常规元组相似的内存消耗**

一般来说，只要需要类似元组的对象，就可以使用`namedtuple`实例。命名元组的优势在于，它们提供了一种使用字段名和点符号来访问其值的方法。这将使您的代码更加 Pythonic 化。

通过对`namedtuple`及其一般特性的简要介绍，您可以更深入地在代码中创建和使用它们。

[*Remove ads*](/account/join/)

## 使用`namedtuple()` 创建类似元组的类

您使用一个`namedtuple()`来创建一个[不可变的](https://docs.python.org/3/glossary.html#term-immutable)和带有字段名称的类似元组的数据结构。在关于`namedtuple`的教程中，一个常见的例子是创建一个类来表示一个数学[点](https://en.wikipedia.org/wiki/Point_(geometry))。

根据问题的不同，您可能希望使用不可变的数据结构来表示给定点。以下是使用常规元组创建二维点的方法:

>>>

```py
>>> # Create a 2D point as a tuple
>>> point = (2, 4)
>>> point
(2, 4)

>>> # Access coordinate x
>>> point[0]
2
>>> # Access coordinate y
>>> point[1]
4

>>> # Try to update a coordinate value
>>> point[0] = 3
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
```

这里，您使用常规的`tuple`创建了一个不可变的二维`point`。这段代码是有效的:你有一个有两个坐标的`point`，你不能修改其中任何一个坐标。但是，这段代码可读吗？你能预先告诉我`0`和`1`指数是什么意思吗？为了避免这些歧义，你可以像这样使用一个`namedtuple`:

>>>

```py
>>> from collections import namedtuple

>>> # Create a namedtuple type, Point
>>> Point = namedtuple("Point", "x y")
>>> issubclass(Point, tuple)
True

>>> # Instantiate the new type
>>> point = Point(2, 4)
>>> point
Point(x=2, y=4)

>>> # Dot notation to access coordinates
>>> point.x
2
>>> point.y
4

>>> # Indexing to access coordinates
>>> point[0]
2
>>> point[1]
4

>>> # Named tuples are immutable
>>> point.x = 100
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: can't set attribute
```

现在您有了一个带有两个适当命名的字段的`point`，`x`和`y`。默认情况下，`point`提供了用户友好的描述性字符串表示(`Point(x=2, y=4)`)。它允许您使用点符号来访问坐标，这是方便的、可读的和明确的。您还可以使用索引来访问每个坐标的值。

**注意:**需要注意的是，虽然元组和命名元组是不可变的，但是它们存储的值不一定是不可变的。

创建保存可变值的元组或命名元组是完全合法的:

>>>

```py
>>> from collections import namedtuple

>>> Person = namedtuple("Person", "name children")
>>> john = Person("John Doe", ["Timmy", "Jimmy"])
>>> john
Person(name='John Doe', children=['Timmy', 'Jimmy'])
>>> id(john.children)
139695902374144

>>> john.children.append("Tina")
>>> john
Person(name='John Doe', children=['Timmy', 'Jimmy', 'Tina'])
>>> id(john.children)
139695902374144

>>> hash(john)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'
```

您可以创建包含可变对象的命名元组。您可以修改底层元组中的可变对象。然而，这并不意味着您正在修改元组本身。元组将继续保存相同的内存引用。

最后，具有可变值的元组或命名元组不是[可散列的](https://docs.python.org/3/glossary.html#term-hashable)，正如你在上面的例子中看到的。

最后，由于`namedtuple`类是`tuple`的子类，它们也是不可变的。所以如果你试图改变一个坐标的值，你会得到一个`AttributeError`。

### 向`namedtuple()` 提供所需的参数

正如您之前所了解的，`namedtuple()`是一个工厂函数，而不是一个典型的数据结构。要创建一个新的`namedtuple`，您需要向函数提供两个位置参数:

1.  **`typename`** 为`namedtuple()`返回的`namedtuple`提供类名。您需要将一个带有有效 Python 标识符的字符串传递给这个参数。
2.  **`field_names`** 提供了用于访问元组中的值的字段名称。您可以使用以下方式提供字段名称:
    *   一个[可迭代的](https://docs.python.org/3/glossary.html#term-iterable)字符串，比如`["field1", "field2", ..., "fieldN"]`
    *   每个字段名由空格分隔的字符串，例如`"field1 field2 ... fieldN"`
    *   每个字段名用逗号分隔的字符串，例如`"field1, field2, ..., fieldN"`

为了说明如何提供`field_names`，以下是创建点的不同方法:

>>>

```py
>>> from collections import namedtuple

>>> # A list of strings for the field names
>>> Point = namedtuple("Point", ["x", "y"])
>>> Point
<class '__main__.Point'>
>>> Point(2, 4)
Point(x=2, y=4)

>>> # A string with comma-separated field names
>>> Point = namedtuple("Point", "x, y")
>>> Point
<class '__main__.Point'>
>>> Point(4, 8)
Point(x=4, y=8)

>>> # A generator expression for the field names
>>> Point = namedtuple("Point", (field for field in "xy"))
>>> Point
<class '__main__.Point'>
>>> Point(8, 16)
Point(x=8, y=16)
```

在这些例子中，首先使用字段名的`list`创建`Point`。然后，使用带有逗号分隔的字段名的字符串。最后，使用一个生成器表达式。在这个例子中，最后一个选项可能看起来有些多余。然而，它旨在说明该过程的灵活性。

注意:如果你使用一个 iterable 来提供字段名，那么你应该使用一个类似序列的 iterable，因为字段的顺序对于产生可靠的结果很重要。

例如，使用`set`可以工作，但可能会产生意想不到的结果:

>>>

```py
>>> from collections import namedtuple

>>> Point = namedtuple("Point", {"x", "y"})
>>> Point(2, 4)
Point(y=2, x=4)
```

当您使用一个无序的 iterable 向一个`namedtuple`提供字段时，您可能会得到意想不到的结果。在上面的例子中，坐标名称被交换了，这可能不适合您的用例。

您可以使用任何有效的 Python 标识符作为字段名称，除了:

*   以下划线(`_`)开头的名称
*   Python [`keywords`](https://realpython.com/python-keywords/)

如果您提供的字段名违反了这些条件中的任何一个，那么您会得到一个`ValueError`:

>>>

```py
>>> from collections import namedtuple

>>> Point = namedtuple("Point", ["x", "_y"])
Traceback (most recent call last):
  ...
ValueError: Field names cannot start with an underscore: '_y'
```

在这个例子中，第二个字段名以和下划线开头，所以您得到一个`ValueError`告诉您字段名不能以那个字符开头。这是为了避免与`namedtuple`方法和属性的名称冲突。

在`typename`的例子中，当你看上面的例子时会产生一个问题:为什么我需要提供`typename`参数？答案是你需要一个由`namedtuple()`返回的类的名字。这类似于为现有类创建别名:

>>>

```py
>>> from collections import namedtuple

>>> Point1 = namedtuple("Point", "x y")
>>> Point1
<class '__main__.Point'>

>>> class Point:
...     def __init__(self, x, y):
...         self.x = x
...         self.y = y
...

>>> Point2 = Point
>>> Point2
<class '__main__.Point'>
```

在第一个例子中，您使用`namedtuple()`创建了`Point`。然后你把这个新类型分配给[全局](https://realpython.com/python-scope-legb-rule/#modules-the-global-scope) [变量](https://realpython.com/python-variables/) `Point1`。在第二个例子中，您创建了一个名为`Point`的常规 Python 类，然后将该类分配给`Point2`。在这两种情况下，类名都是`Point`。`Point1`和`Point2`是当前类的别名。

最后，您还可以使用关键字参数或提供现有字典来创建命名元组，如下所示:

>>>

```py
>>> from collections import namedtuple

>>> Point = namedtuple("Point", "x y")

>>> Point(x=2, y=4)
Point(x=2, y=4)

>>> Point(**{"x": 4, "y": 8})
Point(x=4, y=8)
```

在第一个例子中，您使用关键字参数来创建一个`Point`对象。在第二个例子中，您使用了一个字典，它的键与`Point`的字段相匹配。在这种情况下，你需要执行一个**字典解包**。

[*Remove ads*](/account/join/)

### 使用可选参数`namedtuple()`与

除了两个必需的参数外，`namedtuple()`工厂函数还接受以下可选参数:

*   `rename`
*   `defaults`
*   `module`

如果您将`rename`设置为`True`，那么所有无效的字段名将自动替换为位置名。

假设您的公司有一个用 Python 编写的旧数据库应用程序，用来管理与公司一起旅行的乘客的数据。要求您更新系统，您开始创建命名元组来存储从数据库中读取的数据。

应用程序提供了一个名为`get_column_names()`的函数，该函数返回一个包含列名的字符串列表，您认为可以使用该函数创建一个`namedtuple`类。您最终会得到以下代码:

```py
# passenger.py

from collections import namedtuple

from database import get_column_names

Passenger = namedtuple("Passenger", get_column_names())
```

然而，当您运行代码时，您会得到如下所示的[异常回溯](https://realpython.com/python-traceback/):

```py
Traceback (most recent call last):
  ...
ValueError: Type names and field names cannot be a keyword: 'class'
```

这告诉您,`class`列名不是您的`namedtuple`类的有效字段名称。为了防止这种情况，你决定使用`rename`:

```py
# passenger.py

# ...

Passenger = namedtuple("Passenger", get_column_names(), rename=True)
```

这导致`namedtuple()`自动用位置名称替换无效名称。现在假设您从数据库中检索一行并创建第一个`Passenger`实例，如下所示:

>>>

```py
>>> from passenger import Passenger
>>> from database import get_passenger_by_id

>>> Passenger(get_passenger_by_id("1234"))
Passenger(_0=1234, name='john', _2='Business', _3='John Doe')
```

在这种情况下，`get_passenger_by_id()`是您的假设应用程序中的另一个可用函数。它检索元组中给定乘客的数据。最终结果是您新创建的乘客有三个位置字段名称，只有`name`反映了原始的列名称。当您深入数据库时，您会发现“乘客”表包含以下列:

| 圆柱 | 商店 | 被替换了？ | 理由 |
| --- | --- | --- | --- |
| `_id` | 每位乘客的唯一标识符 | 是 | 它以下划线开头。 |
| `name` | 每位乘客的简称 | 不 | 这是一个有效的 Python 标识符。 |
| `class` | 乘客旅行的等级 | 是 | 这是一个 Python 关键字。 |
| `name` | 乘客的全名 | 是 | 重复了。 |

在基于控制之外的值创建命名元组的情况下，`rename`选项应该设置为`True`,这样无效字段就可以用有效的位置名重命名。

`namedtuple()`的第二个可选参数是`defaults`。该参数默认为 [`None`](https://realpython.com/null-in-python/) ，这意味着这些字段没有默认值。您可以将`defaults`设置为可迭代的值。在这种情况下，`namedtuple()`将`defaults` iterable 中的值分配给最右边的字段:

>>>

```py
>>> from collections import namedtuple

>>> Developer = namedtuple(
...     "Developer",
...     "name level language",
...     defaults=["Junior", "Python"]
... )

>>> Developer("John")
Developer(name='John', level='Junior', language='Python')
```

在这个例子中，`level`和`language`字段具有默认值。这使它们成为可选参数。因为您没有为`name`定义默认值，所以您需要在创建`namedtuple`实例时提供一个值。因此，没有默认值的参数是必需的。请注意，默认值应用于最右边的字段。

`namedtuple()`的最后一个参数是`module`。如果您为这个参数提供了一个有效的模块名，那么结果`namedtuple`的`.__module__`属性将被设置为这个值。此属性保存定义给定函数或可调用函数的模块的名称:

>>>

```py
>>> from collections import namedtuple

>>> Point = namedtuple("Point", "x y", module="custom")
>>> Point
<class 'custom.Point'>
>>> Point.__module__
'custom'
```

在这个例子中，当你在`Point`上访问`.__module__`时，你得到的结果是`'custom'`。这表明您的`Point`类是在您的`custom`模块中定义的。

在 [Python 3.6](https://docs.python.org/3/whatsnew/3.6.html#collections) 中将`module`参数添加到`namedtuple()`的[动机](https://bugs.python.org/issue17941)是为了使命名元组能够通过不同的 [Python 实现](https://docs.python.org/3/reference/introduction.html?highlight=ironpython#alternate-implementations)支持[酸洗](https://realpython.com/python-pickle-module/)。

[*Remove ads*](/account/join/)

## 探索`namedtuple`类的附加特性

除了继承自`tuple`的方法，如`.count()`和`.index()`，`namedtuple`类还提供了三个额外的方法和两个属性。为了防止与自定义字段的名称冲突，这些属性和方法的名称以下划线开头。在本节中，您将了解这些方法和属性以及它们是如何工作的。

### 从 Iterables 创建`namedtuple`实例

您可以使用 [`._make()`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._make) 来创建命名元组实例。该方法采用 iterable 值，并返回一个新的命名元组:

>>>

```py
>>> from collections import namedtuple

>>> Person = namedtuple("Person", "name age height")
>>> Person._make(["Jane", 25, 1.75])
Person(name='Jane', age=25, height=1.75)
```

这里，首先使用`namedtuple()`创建一个`Person`类。然后调用`._make()`，并在`namedtuple`中列出每个字段的值。注意，`._make()`是一个[类方法](https://realpython.com/instance-class-and-static-methods-demystified/)，它作为另一个[类构造函数](https://realpython.com/python-class-constructor/)工作，并返回一个新的命名元组实例。

最后，`._make()`期望一个单独的 iterable 作为参数，在上面的例子中是一个`list`。另一方面，`namedtuple`构造函数可以接受位置参数或关键字参数，正如您已经了解的那样。

### 将`namedtuple`个实例转换成字典

您可以使用 [`._asdict()`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._asdict) 将现有的命名元组实例转换为字典。该方法返回一个使用字段名作为键的新字典。结果字典的键与原始`namedtuple`中的字段顺序相同:

>>>

```py
>>> from collections import namedtuple

>>> Person = namedtuple("Person", "name age height")
>>> jane = Person("Jane", 25, 1.75)
>>> jane._asdict()
{'name': 'Jane', 'age': 25, 'height': 1.75}
```

当您在一个命名元组上调用`._asdict()`时，您会得到一个新的`dict`对象，它将字段名映射到它们在原始命名元组中对应的值。

自从 [Python 3.8](https://realpython.com/python38-new-features/) ，`._asdict()`回归了常规字典。在此之前，它返回了一个 [`OrderedDict`](https://realpython.com/python-ordereddict/) 对象:

>>>

```py
Python 3.7.9 (default, Jan 14 2021, 11:41:20)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from collections import namedtuple

>>> Person = namedtuple("Person", "name age height")
>>> jane = Person("Jane", 25, 1.75)
>>> jane._asdict()
OrderedDict([('name', 'Jane'), ('age', 25), ('height', 1.75)])
```

[Python 3.8 更新了`._asdict()`](https://docs.python.org/3/whatsnew/3.8.html#collections) 返回常规字典，因为在 Python 3.6 及以上版本中字典会记住它们的键的插入顺序。请注意，结果字典中键的顺序等同于原始命名元组中字段的顺序。

### 替换现有`namedtuple`实例中的字段

最后一个方法是 [`._replace()`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._replace) 。该方法采用形式为`field=value`的关键字参数，并返回一个新的`namedtuple`实例来更新所选字段的值:

>>>

```py
>>> from collections import namedtuple

>>> Person = namedtuple("Person", "name age height")
>>> jane = Person("Jane", 25, 1.75)

>>> # After Jane's birthday
>>> jane = jane._replace(age=26)
>>> jane
Person(name='Jane', age=26, height=1.75)
```

在本例中，您在 Jane 生日后更新她的年龄。尽管`._replace()`的名字可能意味着该方法修改了现有的命名元组，但实际上并不是这样。这是因为`namedtuple`实例是不可变的，所以`._replace()`不会就地更新`jane`。

### 探索附加的`namedtuple`属性

命名元组还有两个附加属性: [`._fields`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._fields) 和 [`._field_defaults`](https://docs.python.org/3/library/collections.html#collections.somenamedtuple._field_defaults) 。第一个属性包含一组列出字段名称的字符串。第二个属性包含一个字典，该字典将字段名映射到它们各自的默认值(如果有的话)。

对于`._fields`，您可以用它来自省您的`namedtuple`类和实例。您也可以从现有类别创建新类别:

>>>

```py
>>> from collections import namedtuple

>>> Person = namedtuple("Person", "name age height")

>>> ExtendedPerson = namedtuple(
...     "ExtendedPerson",
...     [*Person._fields, "weight"]
... )

>>> jane = ExtendedPerson("Jane", 26, 1.75, 67)
>>> jane
ExtendedPerson(name='Jane', age=26, height=1.75, weight=67)
>>> jane.weight
67
```

在本例中，您创建了一个名为`ExtendedPerson`的新`namedtuple`，它带有一个新字段`weight`。这种新型号扩展了你的旧型号`Person`。为此，您在`Person`访问`._fields`，并将其与一个附加字段`weight`一起解包到一个新列表中。

您还可以使用 Python 的 [`zip()`](https://realpython.com/python-zip-function/) 使用`._fields`迭代给定`namedtuple`实例中的字段和值:

>>>

```py
>>> from collections import namedtuple

>>> Person = namedtuple("Person", "name age height weight")
>>> jane = Person("Jane", 26, 1.75, 67)
>>> for field, value in zip(jane._fields, jane):
...     print(field, "->", value)
...
name -> Jane
age -> 26
height -> 1.75
weight -> 67
```

在这个例子中，`zip()`产生了形式为`(field, value)`的元组。这样，您可以访问底层命名元组中字段-值对的两个元素。另一种同时迭代字段和值的方法是使用`._asdict().items()`。来吧，试一试！

使用`._field_defaults`，您可以自省`namedtuple`类和实例，找出哪些字段提供默认值。拥有默认值使您的字段成为可选字段。例如，假设您的`Person`类应该包含一个额外的字段来保存这个人居住的国家。因为您主要与来自加拿大的人一起工作，所以您为`country`字段设置适当的默认值，如下所示:

>>>

```py
>>> from collections import namedtuple

>>> Person = namedtuple(
...     "Person",
...     "name age height weight country",
...     defaults=["Canada"]
... )

>>> Person._field_defaults
{'country': 'Canada'}
```

通过快速查询`._field_defaults`，您可以找出给定`namedtuple`中的哪些字段提供默认值。在这个例子中，您团队中的任何其他程序员都可以看到您的`Person`类提供了`"Canada"`作为`country`的便利默认值。

如果您的`namedtuple`没有提供默认值，那么`.field_defaults`保存一个空字典:

>>>

```py
>>> from collections import namedtuple

>>> Person = namedtuple("Person", "name age height weight country")
>>> Person._field_defaults
{}
```

如果你不给`namedtuple()`提供一个默认值列表，那么它依赖于`defaults`的默认值，也就是 [`None`](https://realpython.com/null-in-python/) 。在这种情况下，`._field_defaults`保存一个空字典。

[*Remove ads*](/account/join/)

## 用`namedtuple` 编写 Pythonic 代码

可以说，命名元组的基本用例是帮助您编写更多 Pythonic 代码。创建工厂函数是为了让你写可读的、明确的、干净的和可维护的代码。

在这一节中，您将编写一系列实用的示例，帮助您发现使用命名元组而不是常规元组的好机会，从而使您的代码更加 Pythonic 化。

### 使用字段名代替索引

假设您正在创建一个绘画应用程序，您需要根据用户的选择定义要使用的笔属性。您已经将笔的属性编码在一个元组中:

>>>

```py
>>> pen = (2, "Solid", True)

>>> if pen[0] == 2 and pen[1] == "Solid" and pen[2]:
...     print("Standard pen selected")
...
Standard pen selected
```

这行代码定义了一个包含三个值的元组。你能说出每个值的含义吗？也许你能猜到第二个值和线条样式有关，但是`2`和`True`是什么意思呢？

您可以添加一个很好的注释来为`pen`提供一些上下文，在这种情况下，您将会得到类似这样的结果:

>>>

```py
>>> # Tuple containing: line weight, line style, and beveled edges
>>> pen = (2, "Solid", True)
```

酷！现在您知道了元组中每个值的含义。然而，如果你或另一个程序员使用的`pen`与这个定义相去甚远呢？他们不得不回到定义中去，仅仅是为了记住每个值的含义。

这里有一个使用`namedtuple`的`pen`的替代实现:

>>>

```py
>>> from collections import namedtuple

>>> Pen = namedtuple("Pen", "width style beveled")
>>> pen = Pen(2, "Solid", True)

>>> if pen.width == 2 and pen.style == "Solid" and pen.beveled:
...     print("Standard pen selected")
...
Standard pen selected
```

现在你的代码清楚地表明`2`代表钢笔的宽度，`"Solid"`是线条样式，等等。任何阅读您的代码的人都可以看到并理解这一点。您的新实现`pen`有两行额外的代码。这是在可读性和可维护性方面产生巨大成功的少量工作。

### 从函数中返回多个命名值

可以使用命名元组的另一种情况是当您需要从给定的函数中返回多个值时。在这种情况下，使用命名元组可以使您的代码更具可读性，因为返回值还将为其内容提供一些上下文。

例如，Python 提供了一个名为 [`divmod()`](https://docs.python.org/3/library/functions.html#divmod) 的内置函数，该函数将两个数字作为参数，并返回一个元组，该元组具有从输入数字的整数除法得到的**商**和**余数**:

>>>

```py
>>> divmod(8, 4)
(2, 0)
```

为了记住每个数字的含义，你可能需要阅读`divmod()`的文档，因为数字本身并没有提供关于它们各自含义的太多信息。函数名也没多大帮助。

下面是一个函数，它使用一个`namedtuple`来阐明`divmod()`返回的每个数字的含义:

>>>

```py
>>> from collections import namedtuple

>>> def custom_divmod(a, b):
...     DivMod = namedtuple("DivMod", "quotient remainder")
...     return DivMod(*divmod(a, b))
...

>>> custom_divmod(8, 4)
DivMod(quotient=2, remainder=0)
```

在本例中，您为每个返回值添加了上下文，因此任何程序员在阅读您的代码时都可以立即理解每个数字的含义。

[*Remove ads*](/account/join/)

### 减少函数的参数数量

减少函数可以接受的参数数量被认为是最佳编程实践。这使得你的函数的签名更加简洁，并且优化了你的[测试](https://realpython.com/python-testing/)过程，因为减少了参数的数量和它们之间可能的组合。

同样，您应该考虑使用命名元组来处理这个用例。假设您正在编写一个管理客户信息的应用程序。该应用程序使用一个数据库来存储客户的数据。为了处理数据和更新数据库，您已经创建了几个函数。你的一个高层函数是`create_user()`，看起来是这样的:

```py
def create_user(db, username, client_name, plan):
    db.add_user(username)
    db.complete_user_profile(username, client_name, plan)
```

这个函数有四个参数。第一个参数`db`代表您正在使用的数据库。其余的论点与给定的客户密切相关。这是一个使用命名元组将参数数量减少到`create_user()`的好机会:

```py
User = namedtuple("User", "username client_name plan")
user = User("john", "John Doe", "Premium")

def create_user(db, user):
    db.add_user(user.username)
    db.complete_user_profile(
        user.username,
        user.client_name,
        user.plan
    )
```

现在`create_user()`只需要两个参数:`db`和`user`。在函数内部，使用方便的描述性字段名为`db.add_user()`和`db.complete_user_profile()`提供参数。你的高级功能`create_user()`，更侧重于`user`。测试也更容易，因为您只需要为每个测试提供两个参数。

### 从文件和数据库中读取表格数据

命名元组的一个非常常见的用例是使用它们来存储数据库记录。您可以使用列名作为字段名来定义`namedtuple`类，并将数据从数据库的行中检索到命名元组。你也可以对 [CSV 文件](https://realpython.com/python-csv/)做类似的事情。

例如，假设您有一个包含公司员工数据的 CSV 文件，您希望将该数据读入一个合适的数据结构，以便进一步处理。您的 CSV 文件如下所示:

```py
name,job,email
"Linda","Technical Lead","linda@example.com"
"Joe","Senior Web Developer","joe@example.com"
"Lara","Project Manager","lara@example.com"
"David","Data Analyst","david@example.com"
"Jane","Senior Python Developer","jane@example.com"
```

您正在考虑使用 Python 的 [`csv`模块](https://docs.python.org/3/library/csv.html#module-csv)及其 [`DictReader`](https://docs.python.org/3/library/csv.html#csv.DictReader) 来处理文件，但是您有一个额外的需求——您需要将数据存储到一个不可变的轻量级数据结构中。在这种情况下，`namedtuple`可能是个不错的选择:

>>>

```py
>>> import csv
>>> from collections import namedtuple

>>> with open("employees.csv", "r") as csv_file:
...     reader = csv.reader(csv_file)
...     Employee = namedtuple("Employee", next(reader), rename=True)
...     for row in reader:
...         employee = Employee(*row)
...         print(employee.name, employee.job, employee.email)
...
Linda Technical Lead linda@example.com
Joe Senior Web Developer joe@example.com
Lara Project Manager lara@example.com
David Data Analyst david@example.com
Jane Senior Python Developer jane@example.com
```

在这个例子中，首先在一个 [`with`语句](https://realpython.com/python-with-statement/)中打开`employees.csv`文件。然后使用 [`csv.reader()`](https://docs.python.org/3/library/csv.html?highlight=csv#csv.reader) 来获取 CSV 文件中各行的迭代器。使用`namedtuple()`，您创建了一个新的`Employee`类。对 [`next()`](https://docs.python.org/3/library/functions.html#next) 的调用从`reader`中检索第一行数据，其中包含 CSV 文件头。这个标题为您的`namedtuple`提供了字段名称。

**注意:**当您基于不受您控制的字段名创建`namedtuple`时，您应该将`.rename`设置为`True`。这样，您可以防止无效字段名称的问题，这在您处理数据库表和查询、CSV 文件或任何其他类型的表格数据时是一种常见的情况。

最后， [`for`循环](https://realpython.com/python-for-loop/)从 CSV 文件中的每个`row`创建一个`Employee`实例，然后[将](https://realpython.com/python-print/)雇员列表打印到屏幕上。

## 使用`namedtuple` vs 其他数据结构

到目前为止，您已经学习了如何创建命名元组，以使您的代码更具可读性、更显式和更具 Pythonic 性。您还编写了一些示例，帮助您发现在代码中使用命名元组的机会。

在这一节中，您将大致了解一下`namedtuple`类和其他 Python 数据结构(如字典、数据类和类型化的命名元组)之间的异同。您将比较命名元组与其他数据结构的以下特征:

*   可读性
*   易变性
*   内存使用
*   表演

这样，您就可以更好地为您的特定用例选择正确的数据结构。

[*Remove ads*](/account/join/)

### `namedtuple` vs 字典

[字典](https://realpython.com/python-dicts/)是 Python 中的基本数据结构。语言本身是围绕着字典建立的，所以它们无处不在。因为它们如此普遍和有用，你可能在你的代码中经常使用它们。但是字典和命名元组有多大区别呢？

就可读性而言，你大概可以说字典和命名元组一样可读。尽管它们没有提供通过点符号访问属性的方法，但是字典式的键查找非常易读和简单:

>>>

```py
>>> from collections import namedtuple

>>> jane = {"name": "Jane", "age": 25, "height": 1.75}
>>> jane["age"]
25

>>> # Equivalent named tuple
>>> Person = namedtuple("Person", "name age height")
>>> jane = Person("Jane", 25, 1.75)
>>> jane.age
25
```

在这两个例子中，您已经完全理解了代码及其意图。不过，命名元组定义需要两行额外的代码:一行用于[导入](https://realpython.com/python-import/)工厂函数，另一行用于定义`namedtuple`类`Person`。

这两种数据结构的一个很大的区别是字典是可变的，而命名元组是不可变的。这意味着您*可以*就地修改字典，但是您*不能*修改命名元组:

>>>

```py
>>> from collections import namedtuple

>>> jane = {"name": "Jane", "age": 25, "height": 1.75}
>>> jane["age"] = 26
>>> jane["age"]
26
>>> jane["weight"] = 67
>>> jane
{'name': 'Jane', 'age': 26, 'height': 1.75, 'weight': 67}

>>> # Equivalent named tuple
>>> Person = namedtuple("Person", "name age height")
>>> jane = Person("Jane", 25, 1.75)

>>> jane.age = 26
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: can't set attribute

>>> jane.weight = 67
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Person' object has no attribute 'weight'
```

您可以在字典中更新现有键的值，但不能在命名元组中做类似的事情。可以向现有字典添加新的键-值对，但不能向现有的命名元组添加字段-值对。

**注意:**在命名元组中，可以使用`._replace()`来更新给定字段的值，但是该方法创建并返回一个新的命名元组实例，而不是就地更新底层实例。

一般来说，如果您需要一个不可变的数据结构来正确地解决一个给定的问题，那么可以考虑使用一个命名元组来代替字典，这样就可以满足您的需求。

关于内存使用，命名元组是一种非常轻量级的数据结构。启动您的[代码编辑器或 IDE](https://realpython.com/python-ides-code-editors-guide/) 并创建以下脚本:

```py
# namedtuple_dict_memory.py

from collections import namedtuple
from pympler import asizeof

Point = namedtuple("Point", "x y z")
point = Point(1, 2, 3)

namedtuple_size = asizeof.asizeof(point)
dict_size = asizeof.asizeof(point._asdict())
gain = 100 - namedtuple_size / dict_size * 100

print(f"namedtuple: {namedtuple_size} bytes ({gain:.2f}% smaller)")
print(f"dict: {dict_size} bytes")
```

这个小脚本使用来自 [Pympler](https://pympler.readthedocs.io/en/latest/) 的`asizeof.asizeof()`来获取一个命名元组及其等价字典的内存占用。

**注意:** Pympler 是一个监控和分析 Python 对象内存行为的工具。

您可以照常使用 [`pip`](https://realpython.com/what-is-pip/) 从 [PyPI](https://realpython.com/pypi-publish-python-package/) 安装它:

```py
$ pip install pympler
```

在您运行这个命令之后，Pympler 将在您的 [Python 环境](https://realpython.com/effective-python-environment/)中可用，因此您可以运行上面的脚本。

如果您从命令行[运行脚本](https://realpython.com/run-python-scripts/)，那么您将得到以下输出:

```py
$ python namedtuple_dict_memory.py
namedtuple: 160 bytes (67.74% smaller)
dict:       496 bytes
```

该输出证实了命名元组比等效的字典消耗更少的内存。因此，如果内存消耗对您来说是一个限制，那么您应该考虑使用命名元组而不是字典。

**注意:**在比较命名元组和字典时，最终的内存消耗差异将取决于值的数量及其类型。不同的值，你会得到不同的结果。

最后，您需要了解命名元组和字典在操作性能方面有多么不同。为此，您将测试[成员资格](https://realpython.com/python-boolean/#the-in-operator)和属性访问操作。回到代码编辑器，创建以下脚本:

```py
# namedtuple_dict_time.py

from collections import namedtuple
from time import perf_counter

def average_time(structure, test_func):
    time_measurements = []
    for _ in range(1_000_000):
        start = perf_counter()
        test_func(structure)
        end = perf_counter()
        time_measurements.append(end - start)
    return sum(time_measurements) / len(time_measurements) * int(1e9)

def time_dict(dictionary):
    "x" in dictionary
    "missing_key" in dictionary
    2 in dictionary.values()
    "missing_value" in dictionary.values()
    dictionary["y"]

def time_namedtuple(named_tuple):
    "x" in named_tuple._fields
    "missing_field" in named_tuple._fields
    2 in named_tuple
    "missing_value" in named_tuple
    named_tuple.y

Point = namedtuple("Point", "x y z")
point = Point(x=1, y=2, z=3)

namedtuple_time = average_time(point, time_namedtuple)
dict_time = average_time(point._asdict(), time_dict)
gain = dict_time / namedtuple_time

print(f"namedtuple: {namedtuple_time:.2f} ns ({gain:.2f}x faster)")
print(f"dict: {dict_time:.2f} ns")
```

这个脚本对字典和命名元组的常见操作进行计时，比如成员测试和属性访问。在当前系统上运行该脚本会显示类似于以下内容的输出:

```py
$ namedtuple_dict_time.py
namedtuple: 527.26 ns (1.36x faster)
dict:       717.71 ns
```

该输出显示，对命名元组的操作比对字典的类似操作稍快。

[*Remove ads*](/account/join/)

### `namedtuple` vs 数据类

Python 3.7 带来了一个很酷的新特性:[数据类](https://realpython.com/python-data-classes/)。根据 [PEP 557](https://www.python.org/dev/peps/pep-0557/) 的说法，数据类类似于命名元组，但是它们是可变的:

> 数据类可以被认为是“带有默认值的可变命名元组”([来源](https://www.python.org/dev/peps/pep-0557/#abstract))

然而，更准确地说，数据类就像带有类型提示的可变命名元组。“默认值”部分根本没有区别，因为命名元组的字段也可以有默认值。所以，乍一看，主要的区别是可变性和类型提示。

要创建一个数据类，需要从 [`dataclasses`](https://docs.python.org/3/library/dataclasses.html#module-dataclasses) 导入 [`dataclass()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass) 装饰器。然后，您可以使用常规的类定义语法来定义数据类:

>>>

```py
>>> from dataclasses import dataclass

>>> @dataclass
... class Person:
...     name: str
...     age: int
...     height: float
...     weight: float
...     country: str = "Canada"
...

>>> jane = Person("Jane", 25, 1.75, 67)
>>> jane
Person(name='Jane', age=25, height=1.75, weight=67, country='Canada')
>>> jane.name
'Jane'
>>> jane.name = "Jane Doe"
>>> jane.name
'Jane Doe'
```

就可读性而言，数据类和命名元组之间没有显著差异。它们提供了相似的字符串表示，您可以使用点符号来访问它们的属性。

可变性——根据定义，数据类是可变的，因此您可以在需要时更改它们的属性值。然而，他们有一张王牌。您可以将`dataclass()`装饰器的`frozen`参数设置为`True`，并使它们不可变:

>>>

```py
>>> from dataclasses import dataclass

>>> @dataclass(frozen=True)
... class Person:
...     name: str
...     age: int
...     height: float
...     weight: float
...     country: str = "Canada"
...

>>> jane = Person("Jane", 25, 1.75, 67)
>>> jane.name = "Jane Doe"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 4, in __setattr__
dataclasses.FrozenInstanceError: cannot assign to field 'name'
```

如果您在对`dataclass()`的调用中将`frozen`设置为`True`，那么您就使数据类不可变。在这种情况下，当您尝试更新 Jane 的名字时，您会得到一个 [`FrozenInstanceError`](https://docs.python.org/3/library/dataclasses.html#dataclasses.FrozenInstanceError) 。

命名元组和数据类之间的另一个微妙区别是，后者在默认情况下是不可迭代的。坚持以 Jane 为例，尝试迭代她的数据:

>>>

```py
>>> for field in jane:
...     print(field)
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'Person' object is not iterable
```

如果您试图迭代一个基本的数据类，那么您会得到一个`TypeError`。这是普通班常见的。幸运的是，有办法解决这个问题。例如，您可以向`Person`添加一个 [`.__iter__()`](https://docs.python.org/3/library/stdtypes.html#container.__iter__) 特殊方法，如下所示:

>>>

```py
>>> from dataclasses import astuple, dataclass

>>> @dataclass
... class Person:
...     name: str
...     age: int
...     height: float
...     weight: float
...     country: str = "Canada"
...     def __iter__(self):
...         return iter(astuple(self))
...

>>> for field in Person("Jane", 25, 1.75, 67):
...     print(field)
...
Jane
25
1.75
67
Canada
```

这里，你先从`dataclasses`导入 [`astuple()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.astuple) 。这个函数将数据类转换成一个元组。然后将结果元组传递给 [`iter()`](https://docs.python.org/3/library/functions.html#iter) ，这样就可以构建并从`.__iter__()`返回一个[迭代器](https://docs.python.org/3/library/stdtypes.html#iterator-types)。有了这个添加，您就可以开始迭代 Jane 的数据了。

关于内存消耗，命名元组比数据类更轻量级。您可以通过创建并运行一个类似于上一节中看到的小脚本来确认这一点。要查看完整的脚本，请展开下面的框。



下面的脚本比较了`namedtuple`和它的等价数据类之间的内存使用情况:

```py
# namedtuple_dataclass_memory.py

from collections import namedtuple
from dataclasses import dataclass

from pympler import asizeof

PointNamedTuple = namedtuple("PointNamedTuple", "x y z")

@dataclass
class PointDataClass:
    x: int
    y: int
    z: int

namedtuple_memory = asizeof.asizeof(PointNamedTuple(x=1, y=2, z=3))
dataclass_memory = asizeof.asizeof(PointDataClass(x=1, y=2, z=3))
gain = 100 - namedtuple_memory / dataclass_memory * 100

print(f"namedtuple: {namedtuple_memory} bytes ({gain:.2f}% smaller)")
print(f"data class: {dataclass_memory} bytes")
```

在这个脚本中，您创建了一个命名元组和一个包含相似数据的数据类。然后比较它们的内存占用。

以下是运行脚本的结果:

```py
$ python namedtuple_dataclass_memory.py
namedtuple: 160 bytes (61.54% smaller)
data class: 416 bytes
```

与`namedtuple`类不同，数据类保留一个基于实例的 [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 来存储[可写的实例属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)。这有助于更大的内存占用。

接下来，您可以展开下面的部分来查看一个代码示例，该示例比较了`namedtuple`类和数据类在属性访问方面的性能。



以下脚本比较了命名元组及其等效数据类的属性访问性能:

```py
# namedtuple_dataclass_time.py

from collections import namedtuple
from dataclasses import dataclass
from time import perf_counter

def average_time(structure, test_func):
    time_measurements = []
    for _ in range(1_000_000):
        start = perf_counter()
        test_func(structure)
        end = perf_counter()
        time_measurements.append(end - start)
    return sum(time_measurements) / len(time_measurements) * int(1e9)

def time_structure(structure):
    structure.x
    structure.y
    structure.z

PointNamedTuple = namedtuple("PointNamedTuple", "x y z", defaults=[3])

@dataclass
class PointDataClass:
    x: int
    y: int
    z: int

namedtuple_time = average_time(PointNamedTuple(x=1, y=2, z=3), time_structure)
dataclass_time = average_time(PointDataClass(x=1, y=2, z=3), time_structure)
gain = dataclass_time / namedtuple_time

print(f"namedtuple: {namedtuple_time:.2f} ns ({gain:.2f}x faster)")
print(f"data class: {dataclass_time:.2f} ns")
```

这里，您对属性访问操作进行计时，因为这几乎是命名元组和数据类之间唯一常见的操作。您也可以对成员操作进行计时，但是您必须访问数据类的属性。

在性能方面，结果如下:

```py
$ python namedtuple_dataclass_time.py
namedtuple: 274.32 ns (1.08x faster)
data class: 295.37 ns
```

性能差异很小，所以在属性访问操作方面，可以说这两种数据结构具有相同的性能。

[*Remove ads*](/account/join/)

### `namedtuple`vs`typing.NamedTuple`T2】

Python 3.5 引入了一个名为 [`typing`](https://docs.python.org/3/library/typing.html#module-typing) 的[临时](https://docs.python.org/3/glossary.html#term-provisional-api)模块来支持函数类型注释或者[类型提示](https://docs.python.org/3/whatsnew/3.5.html#whatsnew-pep-484)。本模块提供 [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple) ，是`namedtuple`的打字版本。使用`NamedTuple`，您可以创建带有类型提示的`namedtuple`类。接下来是`Person`的例子，您可以创建一个等价的类型化命名元组，如下所示:

>>>

```py
>>> from typing import NamedTuple

>>> class Person(NamedTuple):
...     name: str
...     age: int
...     height: float
...     weight: float
...     country: str = "Canada"
...

>>> issubclass(Person, tuple)
True
>>> jane = Person("Jane", 25, 1.75, 67)
>>> jane.name
'Jane'
>>> jane.name = "Jane Doe"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: can't set attribute
```

使用`NamedTuple`，您可以通过点符号创建支持类型提示和属性访问的元组子类。由于生成的类是 tuple 子类，所以它也是不可变的。

在上面的例子中需要注意的一个微妙的细节是`NamedTuple`子类看起来比命名元组更像数据类。

当谈到内存消耗时，`namedtuple`和`NamedTuple`实例使用相同数量的内存。您可以展开下面的框来查看比较两者内存使用情况的脚本。



这里有一个脚本比较了一个`namedtuple`和它的对等`typing.NamedTuple`的内存使用情况:

```py
# typed_namedtuple_memory.py

from collections import namedtuple
from typing import NamedTuple

from pympler import asizeof

PointNamedTuple = namedtuple("PointNamedTuple", "x y z")

class PointTypedNamedTuple(NamedTuple):
    x: int
    y: int
    z: int

namedtuple_memory = asizeof.asizeof(PointNamedTuple(x=1, y=2, z=3))
typed_namedtuple_memory = asizeof.asizeof(
    PointTypedNamedTuple(x=1, y=2, z=3)
)

print(f"namedtuple: {namedtuple_memory} bytes")
print(f"typing.NamedTuple: {typed_namedtuple_memory} bytes")
```

在这个脚本中，您创建了一个命名元组和一个等效的类型化`NamedTuple`实例。然后比较两个实例的内存使用情况。

这一次，比较内存使用情况的脚本产生以下输出:

```py
$ python typed_namedtuple_memory.py
namedtuple:        160 bytes
typing.NamedTuple: 160 bytes
```

在这种情况下，两个实例消耗相同数量的内存，所以这次没有赢家。

由于`namedtuple`类和`NamedTuple`子类都是`tuple`的子类，所以它们有很多共同点。在这种情况下，您可以对字段和值的成员资格测试进行计时。您还可以使用点符号对属性访问进行计时。展开下面的方框，查看比较`namedtuple`和`NamedTuple`性能的脚本。



以下脚本比较了`namedtuple`和`typing.NamedTuple`的性能:

```py
# typed_namedtuple_time.py

from collections import namedtuple
from time import perf_counter
from typing import NamedTuple

def average_time(structure, test_func):
    time_measurements = []
    for _ in range(1_000_000):
        start = perf_counter()
        test_func(structure)
        end = perf_counter()
        time_measurements.append(end - start)
    return sum(time_measurements) / len(time_measurements) * int(1e9)

def time_structure(structure):
    "x" in structure._fields
    "missing_field" in structure._fields
    2 in structure
    "missing_value" in structure
    structure.y

PointNamedTuple = namedtuple("PointNamedTuple", "x y z")

class PointTypedNamedTuple(NamedTuple):
    x: int
    y: int
    z: int

namedtuple_time = average_time(PointNamedTuple(x=1, y=2, z=3), time_structure)
typed_namedtuple_time = average_time(
    PointTypedNamedTuple(x=1, y=2, z=3), time_structure
)

print(f"namedtuple: {namedtuple_time:.2f} ns")
print(f"typing.NamedTuple: {typed_namedtuple_time:.2f} ns")
```

在这个脚本中，首先创建一个命名元组，然后创建一个具有类似内容的类型化命名元组。然后比较两种数据结构上常见操作的性能。

结果如下:

```py
$ python typed_namedtuple_time.py
namedtuple:        503.34 ns
typing.NamedTuple: 509.91 ns
```

在这种情况下，可以说这两种数据结构在性能方面表现几乎相同。除此之外，使用`NamedTuple`创建命名元组可以使代码更加明确，因为您可以向字段添加类型信息。您还可以提供默认值，添加新功能，并为您的类型化命名元组编写[文档字符串](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)。

在本节中，您已经学习了很多关于`namedtuple`和其他类似的数据结构和类的知识。下面的表格总结了`namedtuple`与本节介绍的数据结构的比较:

|  | `dict` | 数据类 | `NamedTuple` |
| --- | --- | --- | --- |
| **可读性** | 类似的 | 平等的 | 平等的 |
| **不变性** | 不 | 默认为否，如果使用`@dataclass(frozen=True)`则为是 | 是 |
| **内存使用量** | 高等级的；级别较高的；较重要的 | 高等级的；级别较高的；较重要的 | 平等的 |
| **性能** | 慢的 | 类似的 | 类似的 |
| **可迭代性** | 是 | 默认为否，如果提供`.__iter__()`则为是 | 是 |

有了这个总结，您将能够选择最适合您当前需求的数据结构。此外，您应该考虑数据类和`NamedTuple`允许您添加类型提示，这是当前 Python 代码中非常需要的特性。

## 子类化`namedtuple`类

因为`namedtuple`类是常规的 Python 类，所以如果需要提供额外的功能、文档字符串、用户友好的字符串表示等等，可以对它们进行子类化。

例如，将一个人的年龄存储在一个对象中并不被认为是最佳实践。因此，您可能希望存储出生日期，并在需要时计算年龄:

>>>

```py
>>> from collections import namedtuple
>>> from datetime import date

>>> BasePerson = namedtuple(
...     "BasePerson",
...     "name birthdate country",
...     defaults=["Canada"]
... )

>>> class Person(BasePerson):
...     """A namedtuple subclass to hold a person's data."""
...     __slots__ = ()
...     def __repr__(self):
...         return f"Name: {self.name}, age: {self.age} years old."
...     @property
...     def age(self):
...         return (date.today() - self.birthdate).days // 365
...

>>> Person.__doc__
"A namedtuple subclass to hold a person's data."

>>> jane = Person("Jane", date(1996, 3, 5))
>>> jane.age
25
>>> jane
Name: Jane, age: 25 years old.
```

`Person`继承自`BasePerson`，是一个`namedtuple`类。在子类定义中，首先添加一个 docstring 来描述该类的功能。然后将 [`__slots__`](https://docs.python.org/3/reference/datamodel.html#object.__slots__) 设置为空元组，这样可以防止自动创建基于实例的`.__dict__`。这让你的`BasePerson`子类内存保持高效。

您还可以添加一个自定义的 [`.__repr__()`](https://docs.python.org/3/reference/datamodel.html#object.__repr__) 来为该类提供一个漂亮的字符串表示。最后，添加一个[属性](https://realpython.com/python-descriptors/#python-descriptors-in-properties)，使用 [`datetime`](https://realpython.com/python-datetime/) 计算这个人的年龄。

[*Remove ads*](/account/join/)

## 测量创作时间:`tuple` vs `namedtuple`

到目前为止，您已经根据几个特性将`namedtuple`类与其他数据结构进行了比较。在这一节中，您将大致了解常规元组和命名元组在**创建时间**方面的比较。

假设您有一个动态创建大量元组的应用程序。您决定使用命名元组来提高代码的 Pythonic 性和可维护性。一旦您更新了所有的[代码库](https://en.wikipedia.org/wiki/Codebase)以使用命名元组，您运行应用程序并注意到一些性能问题。经过一些测试后，您得出结论，这些问题可能与动态创建命名元组有关。

下面是一个脚本，它测量动态创建几个元组和命名元组所需的平均时间:

```py
# tuple_namedtuple_time.py

from collections import namedtuple
from time import perf_counter

def average_time(test_func):
    time_measurements = []
    for _ in range(1_000):
        start = perf_counter()
        test_func()
        end = perf_counter()
        time_measurements.append(end - start)
    return sum(time_measurements) / len(time_measurements) * int(1e9)

def time_tuple():
    tuple([1] * 1000)

fields = [f"a{n}" for n in range(1000)]
TestNamedTuple = namedtuple("TestNamedTuple", fields)

def time_namedtuple():
    TestNamedTuple(*([1] * 1000))

namedtuple_time = average_time(time_namedtuple)
tuple_time = average_time(time_tuple)
gain = namedtuple_time / tuple_time

print(f"tuple: {tuple_time:.2f} ns ({gain:.2f}x faster)")
print(f"namedtuple: {namedtuple_time:.2f} ns")
```

在这个脚本中，您将计算创建几个元组及其等价的命名元组所需的平均时间。如果您从命令行运行该脚本，那么您将得到类似如下的输出:

```py
$ python tuple_namedtuple_time.py
tuple:      7075.82 ns (3.36x faster)
namedtuple: 23773.67 ns
```

当您查看这个输出时，可以看到动态创建`tuple`对象比创建相似的命名元组要快得多。在某些情况下，例如使用大型数据库，创建命名元组所需的额外时间会严重影响应用程序的性能，因此如果您的代码动态创建了大量元组，请注意这一点。

## 结论

编写[Python](https://realpython.com/learning-paths/writing-pythonic-code/)代码是 Python 开发领域的一项热门技能。Python 代码是可读的、明确的、干净的、可维护的，并且利用了 Python 习惯用法和最佳实践。在本教程中，您了解了如何创建`namedtuple`类和实例，以及它们如何帮助您提高 Python 代码的质量。

**在本教程中，您学习了:**

*   如何创建和使用 **`namedtuple`** 类和实例
*   如何利用酷 **`namedtuple`功能**
*   何时使用`namedtuple`实例编写**python 代码**
*   何时使用一个`namedtuple`而不是一个类似的**数据结构**
*   如何向**子类的`namedtuple`** 添加新功能

有了这些知识，您可以大大提高现有和未来代码的质量。如果您经常使用元组，那么只要有意义，就考虑将它们转换成命名元组。这样做将使您的代码更具可读性和 Pythonic 性。*********