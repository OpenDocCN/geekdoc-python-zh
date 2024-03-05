# 用 Python 的 Enum 构建常量的枚举

> 原文：<https://realpython.com/python-enum/>

一些编程语言，比如 Java 和 C++，包括支持一种数据类型的语法，这种数据类型被称为**枚举**，或者简称为**枚举**。此数据类型允许您创建语义相关的常量集，您可以通过枚举本身访问这些常量。Python 没有针对枚举的专用语法。然而，Python [标准库](https://docs.python.org/3/library/index.html)有一个`enum`模块，通过`Enum`类支持枚举。

如果你来自一个有枚举的语言，并且你习惯于使用它们，或者如果你只是想学习如何在 Python 中使用枚举，那么本教程就是为你准备的。

**在本教程中，您将学习如何:**

*   使用 Python 的 **`Enum`** 类创建常量的**枚举**
*   在 Python 中使用枚举及其**成员**
*   使用**新功能**定制枚举类
*   编写**实用示例**来理解为什么要使用枚举

此外，您将探索位于`enum`中的其他特定枚举类型，包括`IntEnum`、`IntFlag`和`Flag`。他们会帮助你创建专门的枚举。

要跟随本教程，您应该熟悉 Python 中的[面向对象编程](https://realpython.com/python3-object-oriented-programming/)和[继承](https://realpython.com/inheritance-composition-python/)。

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/python-enum-code/)，您将使用它在 Python 中构建枚举。

## 了解 Python 中的枚举

几种编程语言，包括 [Java](https://realpython.com/java-vs-python/) 和 [C++](https://realpython.com/python-vs-cpp/) ，都有一个本地**枚举**或**枚举**数据类型作为它们语法的一部分。该数据类型允许您创建一组名为[的常量](https://realpython.com/python-constants/)，它们被视为包含枚举的**成员**。您可以通过枚举本身访问成员。

当您需要定义一组[不可变的](https://docs.python.org/3/glossary.html#term-immutable)和[离散的](https://en.wikipedia.org/wiki/Continuous_or_discrete_variable#Discrete_variable)相似或相关的常量值时，枚举就派上了用场，这些常量值在您的代码中可能有语义意义，也可能没有。

一周中的日子、一年中的月份和季节、地球的基本方向、程序的状态代码、HTTP 状态代码、交通灯的颜色以及 web 服务的定价计划都是编程中枚举的很好的例子。一般来说，只要有一个变量可以取一组有限的可能值中的一个，就可以使用枚举。

Python 的语法中没有枚举数据类型。好在 Python 3.4 在[标准库](https://docs.python.org/3/library/index.html)中增加了 [`enum`](https://docs.python.org/3/whatsnew/3.4.html#whatsnew-enum) 模块。这个模块提供了 [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum) 类，用于支持 Python 中的通用枚举。

PEP 435 引入了枚举，其定义如下:

> 枚举是绑定到唯一常数值的一组符号名。在枚举中，值可以通过标识进行比较，枚举本身可以迭代。([来源](https://peps.python.org/pep-0435/))

在添加到标准库中之前，您可以通过定义一系列相似或相关的常数来创建类似于枚举的东西。为此，Python 开发者经常使用以下习语:

>>>

```py
>>> RED, GREEN, YELLOW = range(3)

>>> RED
0

>>> GREEN
1
```

尽管这个习语可以工作，但是当你试图对大量相关的常量进行分组时，它的伸缩性并不好。另一个不便是第一个常量会有一个值`0`，在 Python 中是 falsy。在某些情况下，这可能是一个问题，尤其是那些涉及到[布尔](https://realpython.com/python-boolean/)测试的情况。

**注意:**如果您使用的是 Python 之前的版本，那么您可以通过安装 [enum34](https://pypi.org/project/enum34/) 库来创建枚举，这是标准库`enum`的一个反向移植。第三方库也是你的一个选择。

在大多数情况下，枚举可以帮助您避免上述习语的缺点。它们还将帮助您生成更有组织性、可读性和健壮性的代码。枚举有几个好处，其中一些与编码的简易性有关:

*   允许方便地**将相关常数**分组在一种[命名空间](https://realpython.com/python-namespaces-scope/)中
*   允许对枚举成员或枚举本身进行操作的自定义方法的额外行为
*   为枚举成员提供快速灵活的**访问**
*   启用对成员的**直接迭代**，包括它们的名称和值
*   在[ide 和编辑器](https://realpython.com/python-ides-code-editors-guide/)中促进**代码完成**
*   用静态检查器启用**类型**和**错误检查**
*   提供一个**可搜索**名称的中心
*   使用枚举成员时减少拼写错误

它们还通过提供以下好处使您的代码更加健壮:

*   确保**常量值**在代码执行过程中不会改变
*   通过区分几个枚举共享的相同值来保证类型安全
*   通过使用描述性名称代替神秘值或[幻数](https://en.wikipedia.org/wiki/Magic_number_(programming))来提高**可读性**和**可维护性**
*   通过利用可读的名称而不是没有明确含义的值来帮助**调试**
*   在整个代码中提供**单一的真理来源**和**一致性**

现在您已经了解了编程和 Python 中枚举的基础，您可以开始使用 Python 的`Enum`类创建自己的枚举类型。

[*Remove ads*](/account/join/)

## 用 Python 的`Enum` 创建枚举

Python 的`enum`模块提供了`Enum`类，允许您创建枚举类型。为了创建你自己的枚举，你可以子类化`Enum`或者使用它的函数 API。这两个选项都允许您将一组相关的常数定义为枚举成员。

在接下来的小节中，您将学习如何使用`Enum`类在代码中创建枚举。您还将学习如何为枚举设置自动生成的值，以及如何创建包含别名和唯一值的枚举。首先，您将学习如何通过子类化`Enum`来创建枚举。

### 通过子类化`Enum` 创建枚举

`enum`模块定义了一个具有[迭代](#iterating-through-enumerations)和[比较](#comparing-enumerations)能力的通用枚举类型。您可以使用此类型创建命名常量集，用于替换常见数据类型的文字，如数字和字符串。

何时应该使用枚举的一个经典示例是，当您需要创建一组表示一周中各天的枚举常量时。每一天都有一个符号名称和一个介于`1`和`7`之间的数值，包括 T0 和 T1。

下面是如何通过使用`Enum`作为你的**超类**或**父类**来创建这个枚举:

>>>

```py
>>> from enum import Enum

>>> class Day(Enum):
...     MONDAY = 1
...     TUESDAY = 2
...     WEDNESDAY = 3
...     THURSDAY = 4
...     FRIDAY = 5
...     SATURDAY = 6
...     SUNDAY = 7
...

>>> list(Day)
[
 <Day.MONDAY: 1>,
 <Day.TUESDAY: 2>,
 <Day.WEDNESDAY: 3>,
 <Day.THURSDAY: 4>,
 <Day.FRIDAY: 5>,
 <Day.SATURDAY: 6>,
 <Day.SUNDAY: 7>
]
```

您的`Day`类是`Enum`的子类。所以，你可以称`Day`为**枚举**，或者只是一个**枚举**。`Day.MONDAY`、`Day.TUESDAY`等为**枚举成员**，也称为**枚举成员**，或者简称为**成员**。每个成员必须有一个**值**，该值需要是常量。

因为枚举成员必须是常量，Python 不允许在运行时给枚举成员赋值:

>>>

```py
>>> Day.MONDAY = 0
Traceback (most recent call last):
    ...
AttributeError: Cannot reassign members.

>>> Day
<enum 'Day'>

>>> # Rebind Day
>>> Day = "Monday"
>>> Day
'Monday'
```

如果你试图改变一个枚举成员的值，那么你会得到一个`AttributeError`。与成员名称不同，包含枚举本身的名称不是常量，而是变量。因此，在程序执行的任何时候都有可能重新绑定这个名字，但是你应该避免这样做。

在上面的例子中，您已经重新分配了`Day`，它现在保存一个字符串而不是原来的枚举。这样做，您就失去了对枚举本身的引用。

通常，映射到成员的值是连续的整数。但是，它们可以是任何类型，包括用户定义的类型。在这个例子中，`Day.MONDAY`的值是`1`,`Day.TUESDAY`的值是`2`，以此类推。

**注意**:你可能注意到了`Day`的成员都是大写的。原因如下:

> 因为枚举是用来表示常量的，所以我们建议对枚举成员使用大写名称… ( [Source](https://docs.python.org/3/library/enum.html#module-enum) )

您可以将枚举视为常数的集合。像[列表、元组](https://realpython.com/python-lists-tuples/)或[字典](https://realpython.com/python-dicts/)一样，Python 枚举也是可迭代的。这就是为什么你可以用 [`list()`](https://docs.python.org/3/library/functions.html#func-list) 把一个枚举变成一个枚举成员的`list`。

Python 枚举的成员是容器枚举本身的实例:

>>>

```py
>>> from enum import Enum

>>> class Day(Enum):
...     MONDAY = 1
...     TUESDAY = 2
...     WEDNESDAY = 3
...     THURSDAY = 4
...     FRIDAY = 5
...     SATURDAY = 6
...     SUNDAY = 7
...

>>> type(Day.MONDAY)
<enum 'Day'>

>>> type(Day.TUESDAY)
<enum 'Day'>
```

你不应该混淆像`Day`这样的自定义枚举类和它的成员:`Day.MONDAY`、`Day.TUESDAY`等等。在这个例子中，`Day`枚举类型是枚举成员的中枢，这些成员恰好属于`Day`类型。

您也可以使用基于 [`range()`](https://realpython.com/python-range/) 的习语来构建枚举:

>>>

```py
>>> from enum import Enum

>>> class Season(Enum):
...     WINTER, SPRING, SUMMER, FALL = range(1, 5)
...

>>> list(Season)
[
 <Season.WINTER: 1>,
 <Season.SPRING: 2>,
 <Season.SUMMER: 3>,
 <Season.FALL: 4>
]
```

在这个例子中，你使用带有`start`和`stop`偏移量的`range()`。`start`偏移量允许您提供开始范围的数字，而`stop`偏移量定义范围停止生成数字的数字。

即使您使用`class`语法来创建枚举，它们也是不同于普通 Python 类的特殊类。与常规类不同，枚举:

*   不能被[实例化](https://realpython.com/python-class-constructor/#understanding-pythons-instantiation-process)
*   除非基本枚举没有成员，否则不能将[子类化为](https://realpython.com/inheritance-composition-python/#whats-inheritance)
*   为他们的成员提供一个可读的字符串表示
*   [是可迭代的](https://docs.python.org/3/glossary.html#term-iterable)，按顺序返回它们的成员
*   提供可用作[字典键](https://realpython.com/python-dicts/#dictionary-keys-vs-list-indices)的[可散列](https://realpython.com/python-hash-table/)成员
*   支持**方括号**语法、[调用](https://realpython.com/defining-your-own-python-function/#function-calls-and-definition)语法、**点符号**来访问成员
*   不允许成员[重新分配](https://docs.python.org/3/reference/simple_stmts.html?highlight=assignment#assignment-statements)

当您开始在 Python 中创建和使用自己的枚举时，您应该记住所有这些细微的差别。

通常，枚举的成员采用连续的整数值。然而，在 Python 中，成员的值可以是任何类型，包括用户定义的类型。例如，下面是一个学校成绩的枚举，它以降序使用不连续的数值:

>>>

```py
>>> from enum import Enum

>>> class Grade(Enum):
...     A = 90
...     B = 80
...     C = 70
...     D = 60
...     F = 0
...

>>> list(Grade)
[
 <Grade.A: 90>,
 <Grade.B: 80>,
 <Grade.C: 70>,
 <Grade.D: 60>,
 <Grade.F: 0>
]
```

这个例子表明 Python 枚举非常灵活，允许您为它们的成员使用任何有意义的值。您可以根据代码的意图设置成员值。

还可以为枚举成员使用字符串值。这里有一个你可以在网上商店使用的`Size`枚举的例子:

>>>

```py
>>> from enum import Enum

>>> class Size(Enum):
...     S = "small"
...     M = "medium"
...     L = "large"
...     XL = "extra large"
...

>>> list(Size)
[
 <Size.S: 'small'>,
 <Size.M: 'medium'>,
 <Size.L: 'large'>,
 <Size.XL: 'extra large'>
]
```

在本例中，与每个大小相关联的值包含一个描述，可以帮助您和其他开发人员理解代码的含义。

您还可以创建布尔值的[枚举。在这种情况下，您的枚举成员将只有两个值:](https://realpython.com/python-boolean/)

>>>

```py
>>> from enum import Enum

>>> class SwitchPosition(Enum):
...     ON = True
...     OFF = False
...

>>> list(SwitchPosition)
[<SwitchPosition.ON: True>, <SwitchPosition.OFF: False>]

>>> class UserResponse(Enum):
...     YES = True
...     NO = False
...

>>> list(UserResponse)
[<UserResponse.YES: True>, <UserResponse.NO: False>]
```

这两个例子展示了如何使用枚举向代码中添加额外的上下文。在第一个例子中，任何阅读您的代码的人都会知道代码模拟了一个具有两种可能状态的开关对象。这些附加信息极大地提高了代码的可读性。

您还可以定义具有异类值的枚举:

>>>

```py
>>> from enum import Enum

>>> class UserResponse(Enum):
...     YES = 1
...     NO = "No"
...

>>> UserResponse.NO
<UserResponse.NO: 'No'>

>>> UserResponse.YES
<UserResponse.YES: 1>
```

然而，从类型安全的角度来看，这种做法会使你的代码不一致。因此，不建议这样做。理想情况下，如果您有相同数据类型的值，这将会有所帮助，这与在枚举中将相似、相关的常数分组的想法是一致的。

最后，您还可以创建空枚举:

>>>

```py
>>> from enum import Enum

>>> class Empty(Enum):
...     pass
...

>>> list(Empty)
[]

>>> class Empty(Enum):
...     ...
...

>>> list(Empty)
[]

>>> class Empty(Enum):
...     """Empty enumeration for such and such purposes."""
...

>>> list(Empty)
[]
```

在这个例子中，`Empty`表示一个空的枚举，因为它没有定义任何成员常量。注意，您可以使用 [`pass`](https://realpython.com/python-pass/) 语句、 [`Ellipsis`](https://realpython.com/python-ellipsis/) 文字(`...`)或类级 [docstring](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings) 来创建空枚举。最后一种方法可以通过在 docstring 中提供额外的上下文来帮助您提高代码的可读性。

现在，你为什么需要定义一个空的枚举呢？当您需要构建一个枚举类的层次结构来通过[继承](https://realpython.com/inheritance-composition-python/)重用功能时，空枚举可以派上用场。

考虑下面的例子:

>>>

```py
>>> from enum import Enum
>>> import string

>>> class BaseTextEnum(Enum):
...     def as_list(self):
...         try:
...             return list(self.value)
...         except TypeError:
...             return [str(self.value)]
...

>>> class Alphabet(BaseTextEnum):
...     LOWERCASE = string.ascii_lowercase
...     UPPERCASE = string.ascii_uppercase
...

>>> Alphabet.LOWERCASE.as_list()
['a', 'b', 'c', 'd', ..., 'x', 'y', 'z']
```

在本例中，您将`BaseTextEnum`创建为一个没有成员的枚举。如果自定义枚举没有成员，你只能继承它的子类，所以`BaseTextEnum`符合条件。`Alphabet`类继承自你的空枚举，这意味着你可以访问`.as_list()`方法。此方法将给定成员的值转换为列表。

[*Remove ads*](/account/join/)

### 使用函数式 API 创建枚举

`Enum`类提供了一个[函数 API](https://docs.python.org/3/library/enum.html#functional-api) ，您可以用它来创建枚举，而不需要使用通常的类语法。你只需要调用带有适当参数的`Enum`，就像你调用[函数](https://realpython.com/defining-your-own-python-function/)或任何其他可调用函数一样。

这个功能性的 [API](https://en.wikipedia.org/wiki/API) 类似于 [`namedtuple()`](https://realpython.com/python-namedtuple/) 工厂函数的工作方式。在`Enum`的情况下，功能签名具有以下形式:

```py
Enum(
    value,
    names,
    *,
    module=None,
    qualname=None,
    type=None,
    start=1
)
```

从这个签名，你可以得出结论:`Enum`需要两个[位置](https://realpython.com/defining-your-own-python-function/#positional-arguments)参数，`value`和`names`。它还可以带多达四个[可选](https://realpython.com/python-optional-arguments/)和[仅关键字](https://realpython.com/defining-your-own-python-function/#keyword-only-arguments)参数。这些自变量是`module`、`qualname`、`type`和`start`。

下表总结了`Enum`签名中每个参数的内容和含义:

| 争吵 | 描述 | 需要 |
| --- | --- | --- |
| `value` | 保存具有新枚举类名称的字符串 | 是 |
| `names` | 为枚举成员提供名称 | 是 |
| `module` | 采用定义枚举类的模块的名称 | 不 |
| `qualname` | 保存定义枚举类的模块的位置 | 不 |
| `type` | 保存一个类作为第一个 mixin 类 | 不 |
| `start` | 从枚举值开始取起始值 | 不 |

要提供`names`参数，您可以使用以下对象:

*   包含用空格或逗号分隔的成员名称的字符串
*   可重复的成员名称
*   一个名值对的 iterable

当您需要[清理](https://realpython.com/python-pickle-module/)和取消清理您的枚举时，`module`和`qualname`参数起着重要的作用。如果没有设置`module`,那么 Python 将试图找到这个模块。如果失败，那么这个类将不可选择。类似地，如果没有设置`qualname`，那么 Python 会将其设置为[全局范围](https://realpython.com/python-scope-legb-rule/#modules-the-global-scope)，这可能会导致您的枚举在某些情况下无法取消选取。

当您想要为您的枚举提供一个 [mixin 类](https://en.wikipedia.org/wiki/Mixin)时，`type`参数是必需的。使用 mixin 类可以为您的自定义枚举提供新的功能，比如扩展的比较功能，您将在关于[将枚举与其他数据类型](#mixing-enumerations-with-other-types)混合的章节中了解到这一点。

最后，`start`参数提供了一种定制枚举初始值的方法。这个参数默认为`1`，而不是`0`。使用这个默认值的原因是`0`在布尔意义上是假的，但是枚举成员的计算结果是`True`。因此，从`0`开始似乎令人惊讶和困惑。

大多数情况下，在创建枚举时，您只需使用前两个参数`Enum`。下面是一个创建普通 [HTTP 方法](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol#Request_methods)枚举的例子:

>>>

```py
>>> from enum import Enum

>>> HTTPMethod = Enum(
...     "HTTPMethod", ["GET", "POST", "PUSH", "PATCH", "DELETE"]
... )

>>> list(HTTPMethod)
[
 <HTTPMethod.GET: 1>,
 <HTTPMethod.POST: 2>,
 <HTTPMethod.PUSH: 3>,
 <HTTPMethod.PATCH: 4>,
 <HTTPMethod.DELETE: 5>
]
```

这个对`Enum` [的调用返回](https://realpython.com/python-return-statement/)一个名为`HTTPMethod`的新枚举。要提供成员名称，可以使用字符串列表。每个字符串代表一个 HTTP 方法。注意，成员值被自动设置为从`1`开始的连续整数。您可以使用`start`参数更改这个初始值。

请注意，使用类语法定义上述枚举将产生相同的结果:

>>>

```py
>>> from enum import Enum

>>> class HTTPMethod(Enum):
...     GET = 1
...     POST = 2
...     PUSH = 3
...     PATCH = 4
...     DELETE = 5
...

>>> list(HTTPMethod)
[
 <HTTPMethod.GET: 1>,
 <HTTPMethod.POST: 2>,
 <HTTPMethod.PUSH: 3>,
 <HTTPMethod.PATCH: 4>,
 <HTTPMethod.DELETE: 5>
]
```

这里，您使用类语法来定义`HTTPMethod`枚举。这个例子完全等同于上一个例子，你可以从`list()`的输出中得出结论。

使用类语法还是函数式 API 来创建枚举是你自己的决定，主要取决于你的喜好和具体情况。但是，如果您想要动态创建枚举，那么函数式 API 可能是您唯一的选择。

请考虑以下示例，其中您使用用户提供的成员创建了一个枚举:

>>>

```py
>>> from enum import Enum

>>> names = []
>>> while True:
...     name = input("Member name: ")
...     if name in {"q", "Q"}:
...         break
...     names.append(name.upper())
...
Member name: YES
Member name: NO
Member name: q

>>> DynamicEnum = Enum("DynamicEnum", names)
>>> list(DynamicEnum)
[<DynamicEnum.YES: 1>, <DynamicEnum.NO: 2>]
```

这个例子有点极端，因为从用户的输入中创建任何对象都是一种非常冒险的做法，考虑到您无法预测用户将输入什么。然而，该示例旨在表明，当您需要动态创建枚举时，函数式 API 是一种可行的方法。

最后，如果您需要为您的枚举成员设置自定义值，那么您可以使用一个名值对的 iterable 作为您的`names`参数。在下面的示例中，您使用名称-值元组的列表来初始化所有枚举成员:

>>>

```py
>>> from enum import Enum

>>> HTTPStatusCode = Enum(
...     value="HTTPStatusCode",
...     names=[
...         ("OK", 200),
...         ("CREATED", 201),
...         ("BAD_REQUEST", 400),
...         ("NOT_FOUND", 404),
...         ("SERVER_ERROR", 500),
...     ],
... )

>>> list(HTTPStatusCode)
[
 <HTTPStatusCode.OK: 200>,
 <HTTPStatusCode.CREATED: 201>,
 <HTTPStatusCode.BAD_REQUEST: 400>,
 <HTTPStatusCode.NOT_FOUND: 404>,
 <HTTPStatusCode.SERVER_ERROR: 500>
]
```

像上面那样提供一个名称-值元组列表，可以为成员创建带有自定义值的`HTTPStatusCode`枚举。在这个例子中，如果您不想使用名称-值元组的列表，那么您也可以使用一个将名称映射到值的字典。

[*Remove ads*](/account/join/)

### 从自动值构建枚举

Python 的`enum`模块提供了一个叫做 [`auto()`](https://docs.python.org/3/library/enum.html#using-automatic-values) 的便利函数，允许你为你的枚举成员设置自动值。该函数的默认行为是为成员分配连续的整数值。

下面是`auto()`的工作原理:

>>>

```py
>>> from enum import auto, Enum

>>> class Day(Enum):
...     MONDAY = auto()
...     TUESDAY = auto()
...     WEDNESDAY = 3
...     THURSDAY = auto()
...     FRIDAY = auto()
...     SATURDAY = auto()
...     SUNDAY = 7
...

>>> list(Day)
[
 <Day.MONDAY: 1>,
 <Day.TUESDAY: 2>,
 <Day.WEDNESDAY: 3>,
 <Day.THURSDAY: 4>,
 <Day.FRIDAY: 5>,
 <Day.SATURDAY: 6>,
 <Day.SUNDAY: 7>
]
```

您需要为您需要的每个自动值调用一次`auto()`。您还可以将`auto()`与具体的值结合起来，就像本例中您对`Day.WEDNESDAY`和`Day.SUNDAY`所做的那样。

默认情况下，`auto()`从`1`开始为每个目标成员分配连续的整数。您可以通过覆盖 [`._generate_next_value_()`](https://docs.python.org/3/library/enum.html#using-automatic-values) 方法来调整这种默认行为，`auto()`使用该方法来生成自动值。

下面是一个如何做到这一点的示例:

>>>

```py
>>> from enum import Enum, auto

>>> class CardinalDirection(Enum):
...     def _generate_next_value_(name, start, count, last_values):
...         return name[0]
...     NORTH = auto()
...     SOUTH = auto()
...     EAST = auto()
...     WEST = auto()
...

>>> list(CardinalDirection)
[
 <CardinalDirection.NORTH: 'N'>,
 <CardinalDirection.SOUTH: 'S'>,
 <CardinalDirection.EAST: 'E'>,
 <CardinalDirection.WEST: 'W'>
]
```

在本例中，您创建了一个地球的[主方向](https://en.wikipedia.org/wiki/Cardinal_directions)的枚举，其中的值被自动设置为包含每个成员名字的第一个字符的字符串。请注意，在定义任何成员之前，您必须提供您的覆盖版本的`._generate_next_value_()`。这是因为成员将通过调用方法来构建。

### 使用别名和唯一值创建枚举

您可以创建两个或多个成员具有相同常数值的枚举。冗余成员被称为**别名**，在某些情况下非常有用。例如，假设您有一个包含一组操作系统(OS)的枚举，如以下代码所示:

>>>

```py
>>> from enum import Enum

>>> class OperatingSystem(Enum):
...     UBUNTU = "linux"
...     MACOS = "darwin"
...     WINDOWS = "win"
...     DEBIAN = "linux"
...

>>> # Aliases aren't listed
>>> list(OperatingSystem)
[
 <OperatingSystem.UBUNTU: 'linux'>,
 <OperatingSystem.MACOS: 'darwin'>,
 <OperatingSystem.WINDOWS: 'win'>
]

>>> # To access aliases, use __members__
>>> list(OperatingSystem.__members__.items())
[
 ('UBUNTU', <OperatingSystem.UBUNTU: 'linux'>),
 ('MACOS', <OperatingSystem.MACOS: 'darwin'>),
 ('WINDOWS', <OperatingSystem.WINDOWS: 'win'>),
 ('DEBIAN', <OperatingSystem.UBUNTU: 'linux'>)
]
```

Linux 发行版被认为是独立的操作系统。所以，Ubuntu 和 Debian 都是独立的系统，有不同的目标和目标受众。然而，它们共享一个叫做 Linux 的通用内核。

上面的枚举将操作系统映射到它们相应的内核。这种关系将`DEBIAN`变成了`UBUNTU`的别名，当您拥有与内核相关的代码以及特定于给定 Linux 发行版的代码时，这可能会很有用。

在上面的例子中需要注意的一个重要行为是，当你直接迭代枚举时，不考虑别名。如果您需要迭代所有成员，包括别名，那么您需要使用`.__members__`。在关于[遍历枚举](#iterating-through-enumerations)的章节中，您将了解到更多关于迭代和`.__members__`属性的知识。

您还可以选择在枚举中完全禁止别名。为此，您可以使用`enum`模块中的 [`@unique`](https://docs.python.org/3/library/enum.html#ensuring-unique-enumeration-values) [装饰器](https://realpython.com/primer-on-python-decorators/):

>>>

```py
>>> from enum import Enum, unique

>>> @unique
... class OperatingSystem(Enum):
...     UBUNTU = "linux"
...     MACOS = "darwin"
...     WINDOWS = "win"
...     DEBIAN = "linux"
...
Traceback (most recent call last):
    ...
ValueError: duplicate values in <enum 'OperatingSystem'>: DEBIAN -> UBUNTU
```

在这个例子中，你用`@unique`来修饰`OperatingSystem`。如果任何成员值是重复的，那么您将得到一个`ValueError`。这里，异常消息指出`DEBIAN`和`UBUNTU`共享相同的值，这是不允许的。

## 在 Python 中使用枚举

到目前为止，您已经了解了什么是枚举，何时使用它们，以及在代码中使用它们有什么好处。您还了解了如何使用`Enum`类作为超类或可调用类在 Python 中创建枚举。

现在是时候开始研究 Python 的枚举是如何工作的，以及如何在代码中使用它们了。

[*Remove ads*](/account/join/)

### 访问枚举成员

当在代码中使用枚举时，访问它们的成员是要执行的基本操作。在 Python 中，您将有三种不同的方法来访问枚举成员。

例如，假设您需要访问下面的`CardinalDirection`枚举的`NORTH`成员。在这种情况下，你可以这样做:

>>>

```py
>>> from enum import Enum

>>> class CardinalDirection(Enum):
...     NORTH = "N"
...     SOUTH = "S"
...     EAST = "E"
...     WEST = "W"
...

>>> # Dot notation
>>> CardinalDirection.NORTH <CardinalDirection.NORTH: 'N'>

>>> # Call notation
>>> CardinalDirection("N") <CardinalDirection.NORTH: 'N'>

>>> # Subscript notation
>>> CardinalDirection["NORTH"] <CardinalDirection.NORTH: 'N'>
```

本例中突出显示的第一行显示了如何使用**点符号**来访问一个枚举成员，这非常直观和易读。第二个突出显示的行通过**调用**枚举并以成员的值作为参数来访问目标成员。

**注意:**需要注意的是，用成员的值作为参数调用枚举会让你感觉像是在实例化该枚举。然而，正如您已经知道的，枚举不能被实例化:

>>>

```py
>>> week = Day()
Traceback (most recent call last):
    ...
TypeError: EnumMeta.__call__() missing 1 required positional argument: 'value'
```

试图创建一个现有枚举的实例是不允许的，所以如果你试图这样做，你会得到一个`TypeError`。因此，不能将实例化与通过枚举调用访问成员相混淆。

最后，突出显示的第三行显示了如何使用类似于**字典的符号**或**下标符号**来访问一个成员，并将该成员的名称作为目标键。

Python 的枚举为您访问成员提供了极大的灵活性。点符号可以说是 Python 代码中最常用的方法。然而，其他两种方法也有帮助。因此，使用满足您特定需求、惯例和风格的符号。

### 使用`.name`和`.value`属性

Python 枚举的成员是其包含类的实例。在 enum 类解析过程中，每个成员都会自动获得一个`.name`属性，该属性将成员的名称保存为一个字符串。成员还获得一个`.value`属性，该属性在类定义中存储分配给成员本身的值。

您可以像处理常规属性一样，使用点符号来访问`.name`和`.value`。考虑下面的例子，它模拟了一个信号量，通常被称为交通灯:

>>>

```py
>>> from enum import Enum

>>> class Semaphore(Enum):
...     RED = 1
...     YELLOW = 2
...     GREEN = 3
...

>>> Semaphore.RED.name
'RED'

>>> Semaphore.RED.value
1

>>> Semaphore.YELLOW.name
'YELLOW'
```

枚举成员的`.name`和`.value`属性分别让你直接访问成员的字符串名称和成员的值。当您遍历枚举时，这些属性会派上用场，这将在下一节中探讨。

### 遍历枚举

与常规类相比，Python 枚举的一个显著特征是枚举在默认情况下是可迭代的。因为它们是可迭代的，你可以在 [`for`循环](https://realpython.com/python-for-loop/)中使用它们，也可以和其他接受并处理可迭代的工具一起使用。

Python 的枚举支持按照定义顺序对成员进行直接迭代:

>>>

```py
>>> from enum import Enum

>>> class Flavor(Enum):
...     VANILLA = 1
...     CHOCOLATE = 2
...     MINT = 3
...

>>> for flavor in Flavor:
...     print(flavor)
...
Flavor.VANILLA
Flavor.CHOCOLATE
Flavor.MINT
```

在这个例子中，您使用一个`for`循环来迭代`Flavor`的成员。请注意，成员的产生顺序与它们在类定义中的定义顺序相同。

当你迭代一个枚举时，你可以访问`.name`和`.value`属性:

>>>

```py
>>> for flavor in Flavor:
...     print(flavor.name, "->", flavor.value)
...
VANILLA -> 1
CHOCOLATE -> 2
MINT -> 3
```

这种迭代技术看起来非常类似于[对字典](https://realpython.com/iterate-through-dictionary-python/)的迭代。因此，如果您熟悉字典迭代，那么使用这种技术遍历枚举将是一项简单的任务，有许多潜在的用例。

或者，枚举有一个名为`.__members__`的特殊属性，您也可以用它来迭代它们的成员。该属性包含一个将名称映射到成员的字典。遍历这个字典和直接遍历枚举的区别在于，字典允许您访问枚举的所有成员，包括您可能拥有的所有别名。

下面是一些使用`.__members__`遍历`Flavor`枚举的例子:

>>>

```py
>>> for name in Flavor.__members__:
...     print(name)
...
VANILLA
CHOCOLATE
MINT

>>> for name in Flavor.__members__.keys():
...     print(name)
...
VANILLA
CHOCOLATE
MINT

>>> for member in Flavor.__members__.values():
...     print(member)
...
Flavor.VANILLA
Flavor.CHOCOLATE
Flavor.MINT

>>> for name, member in Flavor.__members__.items():
...     print(name, "->", member)
...
VANILLA -> Flavor.VANILLA
CHOCOLATE -> Flavor.CHOCOLATE
MINT -> Flavor.MINT
```

您可以使用`.__members__`特殊属性对 Python 枚举的成员进行详细的编程访问。因为`.__members__`拥有一个常规字典，所以您可以使用适用于这个内置数据类型的所有迭代技术。这些技术包括使用字典方法，如 [`.key()`](https://realpython.com/iterate-through-dictionary-python/#iterating-through-keys) 、 [`.values()`](https://realpython.com/iterate-through-dictionary-python/#iterating-through-values) 和 [`.items()`](https://realpython.com/iterate-through-dictionary-python/#iterating-through-items) 。

[*Remove ads*](/account/join/)

### 在`if`和`match`语句中使用枚举

链式 [`if` … `elif`](https://realpython.com/python-conditional-statements/#the-else-and-elif-clauses) 语句和相对较新的 [`match` … `case`](https://realpython.com/python310-new-features/#structural-pattern-matching) 语句是可以使用枚举的常见且自然的地方。这两种结构都允许您根据特定条件采取不同的操作过程。

例如，假设您有一段处理交通控制应用程序中的信号量或交通灯的代码。您必须根据信号量的当前指示灯执行不同的操作。在这种情况下，您可以使用枚举来表示信号量及其指示灯。然后，您可以使用一系列`if` … `elif`语句来决定要运行的操作:

>>>

```py
>>> from enum import Enum

>>> class Semaphore(Enum):
...     RED = 1
...     YELLOW = 2
...     GREEN = 3
...

>>> def handle_semaphore(light):
...     if light is Semaphore.RED:
...         print("You must stop!")
...     elif light is Semaphore.YELLOW:
...         print("Light will change to red, be careful!")
...     elif light is Semaphore.GREEN:
...         print("You can continue!")
...

>>> handle_semaphore(Semaphore.GREEN)
You can continue!

>>> handle_semaphore(Semaphore.YELLOW)
Light will change to red, be careful!

>>> handle_semaphore(Semaphore.RED)
You must stop!
```

您的`handle_semaphore()`函数中的`if` … `elif`语句链检查当前灯光的值，以决定要采取的行动。注意对`handle_semaphore()`中的`print()`的调用只是占位符。在真正的代码中，你可以用更复杂的操作来代替它们。

如果您使用的是 [Python 3.10](https://realpython.com/python310-new-features/) 或更高版本，那么您可以快速将上面的`if` … `elif`语句链转换成等价的`match` … `case`语句:

>>>

```py
>>> from enum import Enum

>>> class Semaphore(Enum):
...     RED = 1
...     YELLOW = 2
...     GREEN = 3
...

>>> def handle_semaphore(light):
...     match light:
...         case Semaphore.RED:
...             print("You must stop!")
...         case Semaphore.YELLOW:
...             print("Light will change to red, be careful!")
...         case Semaphore.GREEN:
...             print("You can continue!")
...

>>> handle_semaphore(Semaphore.GREEN)
You can continue!

>>> handle_semaphore(Semaphore.YELLOW)
Light will change to red, be careful!

>>> handle_semaphore(Semaphore.RED)
You must stop!
```

这个新的`handle_semaphore()`实现等同于之前使用`if` … `elif`语句的实现。使用任何一种技术都是一个品味和风格的问题。这两种技术都工作得很好，并且在可读性方面不相上下。但是，请注意，如果您需要保证向后兼容低于 3.10 的 Python 版本，那么您必须使用链式`if` … `elif`语句。

最后，请注意，尽管枚举似乎可以很好地处理`if` … `elif`和`match` … `case`语句，但是您必须记住，这些语句不能很好地伸缩。如果您向目标枚举添加新成员，那么您需要更新处理函数来考虑这些新成员。

### 比较枚举数

能够在`if` … `elif`语句和`match` … `case`语句中使用枚举意味着枚举成员可以进行比较。默认情况下，枚举支持两种类型的比较运算符:

1.  **标识**，使用 [`is`](https://docs.python.org/3/reference/expressions.html#is) 和 [`is not`](https://docs.python.org/3/reference/expressions.html#is-not) 运算符
2.  **相等**，使用`==`和`!=`运算符

身份比较依赖于每个枚举成员是其枚举类的[单例](https://en.wikipedia.org/wiki/Singleton_pattern)实例这一事实。这个特性允许使用`is`和`is not`操作符对成员进行快速廉价的身份比较。

请考虑下面的示例，这些示例比较了枚举成员的不同组合:

>>>

```py
>>> from enum import Enum

>>> class AtlanticAveSemaphore(Enum):
...     RED = 1
...     YELLOW = 2
...     GREEN = 3
...     PEDESTRIAN_RED = 1
...     PEDESTRIAN_GREEN = 3
...

>>> red = AtlanticAveSemaphore.RED
>>> red is AtlanticAveSemaphore.RED
True
>>> red is not AtlanticAveSemaphore.RED
False

>>> yellow = AtlanticAveSemaphore.YELLOW
>>> yellow is red
False
>>> yellow is not red
True

>>> pedestrian_red = AtlanticAveSemaphore.PEDESTRIAN_RED
>>> red is pedestrian_red
True
```

每个枚举成员都有自己的标识，不同于其同级成员的标识。这条规则不适用于成员别名，因为它们只是对现有成员的引用，并且共享相同的标识。这就是为什么在最后一个例子中比较`red`和`pedestrian_red`会返回`True`。

**注意:**在 Python 中获取给定对象的标识，可以使用内置的 [`id()`](https://docs.python.org/3/library/functions.html#id) 函数，将对象作为参数。

不同枚举的成员之间的身份检查总是返回`False`:

>>>

```py
>>> class EighthAveSemaphore(Enum):
...     RED = 1
...     YELLOW = 2
...     GREEN = 3
...     PEDESTRIAN_RED = 1
...     PEDESTRIAN_GREEN = 3
...

>>> AtlanticAveSemaphore.RED is EighthAveSemaphore.RED
False

>>> AtlanticAveSemaphore.YELLOW is EighthAveSemaphore.YELLOW
False
```

产生这个错误结果的原因是不同枚举的成员是独立的实例，它们有自己的身份，所以对它们的任何身份检查都返回`False`。

相等运算符`==`和`!=`也在枚举成员之间起作用:

>>>

```py
>>> from enum import Enum

>>> class AtlanticAveSemaphore(Enum):
...     RED = 1
...     YELLOW = 2
...     GREEN = 3
...     PEDESTRIAN_RED = 1
...     PEDESTRIAN_GREEN = 3
...

>>> red = AtlanticAveSemaphore.RED
>>> red == AtlanticAveSemaphore.RED
True

>>> red != AtlanticAveSemaphore.RED
False

>>> yellow = AtlanticAveSemaphore.YELLOW
>>> yellow == red
False
>>> yellow != red
True

>>> pedestrian_red = AtlanticAveSemaphore.PEDESTRIAN_RED
>>> red == pedestrian_red
True
```

Python 的枚举通过分别委托给`is`和`is not`操作符来支持操作符`==`和`!=`。

正如您已经了解的，枚举成员总是有一个具体的值，可以是数字、字符串或任何其他对象。正因为如此，在枚举成员和公共对象之间运行相等比较可能很有诱惑力。

然而，这种比较并不像预期的那样工作，因为实际的比较是基于对象的身份:

>>>

```py
>>> from enum import Enum

>>> class Semaphore(Enum):
...     RED = 1
...     YELLOW = 2
...     GREEN = 3
...

>>> Semaphore.RED == 1
False

>>> Semaphore.YELLOW == 2
False

>>> Semaphore.GREEN != 3
True
```

即使每个示例中的成员值都等于整数，这些比较也会返回`False`。这是因为常规枚举成员按对象标识而不是按值进行比较。在上面的例子中，您将枚举成员与整数进行比较，这就像比较苹果和橙子一样。他们永远不会平等比较，因为他们有不同的身份。

**注意:** [稍后的](#building-integer-enumerations-intenum)，你会了解到`IntEnum`是可以和整数进行比较的特殊枚举。

最后，枚举的另一个与比较相关的特性是，您可以使用`in`和`not in`操作符对它们执行成员测试:

>>>

```py
>>> from enum import Enum

>>> class Semaphore(Enum):
...     RED = 1
...     YELLOW = 2
...     GREEN = 3
...

>>> Semaphore.RED in Semaphore
True

>>> Semaphore.GREEN not in Semaphore
False
```

Python 的枚举默认支持`in`和`not in`操作符。使用这些运算符，可以检查给定成员是否存在于给定的枚举中。

[*Remove ads*](/account/join/)

### 排序枚举

默认情况下，Python 的枚举不支持比较运算符，如`>`、`<`、`>=`和`<=`。这就是为什么不能直接使用内置的 [`sorted()`](https://realpython.com/python-sort/) 函数对枚举成员进行排序，如下例所示:

>>>

```py
>>> from enum import Enum

>>> class Season(Enum):
...     SPRING = 1
...     SUMMER = 2
...     AUTUMN = 3
...     WINTER = 4
...

>>> sorted(Season)
Traceback (most recent call last):
    ...
TypeError: '<' not supported between instances of 'Season' and 'Season'
```

当您使用枚举作为`sorted()`的参数时，您会得到一个`TypeError`，因为枚举不支持`<`操作符。然而，有一种方法可以通过使用`sorted()`调用中的`key`参数，成功地按照成员的名称和值对枚举进行排序。

以下是如何做到这一点:

>>>

```py
>>> sorted(Season, key=lambda season: season.value)
[
 <Season.SPRING: 1>,
 <Season.SUMMER: 2>,
 <Season.AUTUMN: 3>,
 <Season.WINTER: 4>
]

>>> sorted(Season, key=lambda season: season.name)
[
 <Season.AUTUMN: 3>,
 <Season.SPRING: 1>,
 <Season.SUMMER: 2>,
 <Season.WINTER: 4>
]
```

在第一个示例中，您使用了一个 [`lambda`](https://realpython.com/python-lambda/) 函数，该函数将一个枚举成员作为参数，并返回其`.value`属性。使用这种技术，您可以根据输入枚举的值对其进行排序。在第二个例子中，`lambda`函数接受一个枚举成员并返回它的`.name`属性。这样，您可以按成员名称对枚举进行排序。

## 用新行为扩展枚举

在前面的章节中，您已经学习了如何在 Python 代码中创建和使用枚举。到目前为止，您已经使用了默认枚举。这意味着您只使用了 Python 枚举的标准特性和行为。

有时，您可能需要为您的枚举提供自定义行为。为此，您可以向枚举中添加方法并实现所需的功能。也可以使用 mixin 类。在接下来的小节中，您将学习如何利用这两种技术来自定义您的枚举。

### 添加和调整成员方法

您可以像处理任何常规 Python 类一样，通过向枚举类添加新方法来为枚举提供新功能。枚举是具有特殊功能的类。像常规类一样，枚举可以有方法和特殊方法。

考虑下面的例子，改编自 [Python 文档](https://docs.python.org/3/library/enum.html#allowed-members-and-attributes-of-enumerations):

>>>

```py
>>> from enum import Enum

>>> class Mood(Enum):
...     FUNKY = 1
...     MAD = 2
...     HAPPY = 3
...
...     def describe_mood(self):
...         return self.name, self.value
...
...     def __str__(self):
...         return f"I feel {self.name}"
...
...     @classmethod
...     def favorite_mood(cls):
...         return cls.HAPPY
...

>>> Mood.HAPPY.describe_mood()
('HAPPY', 3)

>>> print(Mood.HAPPY)
I feel HAPPY

>>> Mood.favorite_mood()
<Mood.HAPPY: 3>
```

在这个例子中，您有一个包含三个成员的`Mood`枚举。像`.describe_mood()`这样的常规方法被绑定到包含它们的枚举的实例上，这些实例是枚举成员。因此，您必须在枚举成员上调用常规方法，而不是在枚举类本身上。

**注意:**记住 Python 的枚举是不能实例化的。枚举的成员是枚举的允许实例。因此，`self`参数代表当前成员。

类似地， [`.__str__()`](https://docs.python.org/3/reference/datamodel.html#object.__str__) 特殊方法对成员进行操作，提供每个成员的可打印表示。

最后，`.favorite_mood()`方法是一个[类方法](https://realpython.com/instance-class-and-static-methods-demystified/)，它对类或枚举本身进行操作。像这样的类方法提供了从类内部对所有枚举成员的访问。

当您需要实现[策略模式](https://en.wikipedia.org/wiki/Strategy_pattern)时，您也可以利用这种能力来包含额外的行为。例如，假设您需要一个类，该类允许您使用两种策略对数字列表进行升序和降序排序。在这种情况下，可以使用如下所示的枚举:

>>>

```py
>>> from enum import Enum

>>> class Sort(Enum):
...     ASCENDING = 1
...     DESCENDING = 2
...     def __call__(self, values):
...         return sorted(values, reverse=self is Sort.DESCENDING)
...

>>> numbers = [5, 2, 7, 6, 3, 9, 8, 4]

>>> Sort.ASCENDING(numbers)
[2, 3, 4, 5, 6, 7, 8, 9]

>>> Sort.DESCENDING(numbers)
[9, 8, 7, 6, 5, 4, 3, 2]
```

`Sort`的每个成员代表一种排序策略。 [`.__call__()`](https://docs.python.org/3/reference/datamodel.html#object.__call__) 方法使得`Sort`的成员是可调用的。在`.__call__()`中，您使用内置的 [`sorted()`](https://realpython.com/python-sort/) 函数根据被调用的成员对输入值进行升序或降序排序。

**注意:**上面的例子是一个使用 enum 来实现策略[设计模式](https://en.wikipedia.org/wiki/Software_design_pattern)的示范例子。实际上，没有必要为了包装`sorted()`函数而创建这个`Sort`枚举。相反，你可以直接使用`sorted()`和它的`reverse`参数，避免过度设计你的解决方案。

调用`Sort.ASCENDING`时，输入的数字按升序排序。相比之下，当您调用`Sort.DESCENDING`时，数字会按降序排列。就是这样！您已经使用枚举快速实现了策略设计模式。

[*Remove ads*](/account/join/)

### 将枚举与其他类型混合

Python 支持[多重继承](https://realpython.com/inheritance-composition-python/#inheriting-multiple-classes)作为其面向对象特性的一部分。这意味着在 Python 中，创建类层次结构时可以继承多个类。当您想同时重用几个类的功能时，多重继承就很方便了。

面向对象编程中的一个常见做法是使用所谓的[混合类](https://en.wikipedia.org/wiki/Mixin)。这些类提供了其他类可以使用的功能。在 Python 中，可以将 mixin 类添加到给定类的父类列表中，以自动获得 mixin 功能。

例如，假设您想要一个支持整数比较的枚举。在这种情况下，您可以在定义枚举时使用内置的`int`类型作为 mixin:

>>>

```py
>>> from enum import Enum

>>> class Size(int, Enum):
...     S = 1
...     M = 2
...     L = 3
...     XL = 4
...

>>> Size.S > Size.M
False
>>> Size.S < Size.M
True
>>> Size.L >= Size.M
True
>>> Size.L <= Size.M
False

>>> Size.L > 2
True
>>> Size.M < 1
False
```

在这个例子中，你的`Size`类继承自`int`和`Enum`。从`int`类型继承可以通过`>`、`<`、`>=`和`<=`比较操作符在成员之间进行直接比较。它还支持在`Size`成员和整数之间进行比较。

最后，请注意，当您将一个数据类型用作 mixin 时，成员的`.value`属性与成员本身并不相同，尽管它是等效的，并且会以同样的方式进行比较。这就是为什么你可以直接用整数来比较`Size`的成员。

**注意:**使用整数枚举成员值是一种非常常见的做法。这就是为什么`enum`模块提供了一个`IntEnum`来直接用整数值创建枚举。在名为[探索其他枚举类](#exploring-other-enumeration-classes)的章节中，您将了解到关于这个类的更多信息。

上面的例子表明，当您需要重用一个给定的功能时，用 mixin 类创建枚举通常会有很大的帮助。如果你决定在你的一些枚举中使用这种技术，那么你必须坚持下面的签名:

```py
class EnumName([mixin_type, ...], [data_type,] enum_type):
    # Members go here...
```

这个签名意味着您可以拥有一个或多个 mixin 类，最多一个数据类型类，以及父 enum 类。

考虑下面的例子:

>>>

```py
>>> from enum import Enum

>>> class MixinA:
...     def a(self):
...         print(f"MixinA: {self.value}")
...

>>> class MixinB:
...     def b(self):
...         print(f"MixinB: {self.value}")
...

>>> class ValidEnum(MixinA, MixinB, str, Enum):
...     MEMBER = "value"
...

>>> ValidEnum.MEMBER.a()  # Call .a() from MixinA
MixinA: value

>>> ValidEnum.MEMBER.b()  # Call .b() from MixinB
MixinB: value

>>> ValidEnum.MEMBER.upper()  # Call .upper() from str
'VALUE'

>>> class WrongMixinOrderEnum(Enum, MixinA, MixinB):
...     MEMBER = "value"
...
Traceback (most recent call last):
    ...
TypeError: new enumerations should be created as
 `EnumName([mixin_type, ...] [data_type,] enum_type)`

>>> class TooManyDataTypesEnum(int, str, Enum):
...     MEMBER = "value"
...
Traceback (most recent call last):
    ...
TypeError: 'TooManyDataTypesEnum': too many data types:
 {<class 'int'>, <class 'str'>}
```

`ValidEnum`类表明，在碱基序列中，您必须根据需要放置尽可能多的 mixin 类——但只能放置一种数据类型——在`Enum`之前。

`WrongMixinOrderEnum`显示，如果您将`Enum`放在最后一个位置之外的任何位置，那么您将得到一个`TypeError`,其中包含要使用的正确签名的信息。同时，`TooManyDataTypesEnum`确认你的 mixin 类列表必须最多有一个具体的数据类型，比如`int`或者`str`。

请记住，如果您在 mixin 类列表中使用具体的数据类型，那么成员值必须与该特定数据类型的类型相匹配。

## 探索其他枚举类

除了`Enum`之外，`enum`模块还提供了一些额外的类，允许您创建具有特定行为的枚举。您将拥有用于创建枚举常数的`IntEnum`类，该类也是`int`的子类，这意味着所有成员将拥有整数的所有特性。

你还会发现更多的专业类，比如`IntFlag`和`Flag`。这两个类都允许你创建常量的枚举集合，你可以使用[位操作符](https://realpython.com/python-bitwise-operators/)来组合它们。在下一节中，您将探索这些类以及它们如何在 Python 中工作。

### 构建整数枚举:`IntEnum`

整数枚举是如此常见，以至于`enum`模块导出了一个名为`IntEnum`的专用类，它是专门为涵盖这种用例而创建的。如果您需要您的枚举成员表现得像整数，那么您应该从`IntEnum`继承而不是从`Enum`继承。

子类化`IntEnum`相当于使用多重继承，将`int`作为 mixin 类:

>>>

```py
>>> from enum import IntEnum

>>> class Size(IntEnum):
...     S = 1
...     M = 2
...     L = 3
...     XL = 4
...

>>> Size.S > Size.M
False
>>> Size.S < Size.M
True
>>> Size.L >= Size.M
True
>>> Size.L <= Size.M
False

>>> Size.L > 2
True
>>> Size.M < 1
False
```

现在`Size`直接继承`IntEnum`而不是继承`int`和`Enum`。和以前版本的`Size`一样，这个新版本拥有完整的比较功能，并支持所有的比较操作符。也可以在整数运算中直接使用类成员。

`Size`将自动尝试将不同数据类型的任何值转换为整数。如果这种转换是不可能的，那么您将得到一个`ValueError`:

>>>

```py
>>> from enum import IntEnum

>>> class Size(IntEnum):
...     S = 1
...     M = 2
...     L = 3
...     XL = "4" ...

>>> list(Size)
[<Size.S: 1>, <Size.M: 2>, <Size.L: 3>, <Size.XL: 4>]

>>> class Size(IntEnum):
...     S = 1
...     M = 2
...     L = 3
...     XL = "4.o" ...
Traceback (most recent call last):
    ...
ValueError: invalid literal for int() with base 10: '4.o'
```

在第一个例子中，`Size`自动将字符串`"4"`转换为整数值。在第二个例子中，因为字符串`"4.o"`不包含有效的数值，所以得到一个`ValueError`，转换失败。

在当前稳定的 Python 版本 [3.10](https://realpython.com/python310-new-features/) 中，`enum`模块不包含`StrEnum`类。然而，这个类是枚举的另一个流行用例。因此，Python 3.11 将包含一个 [`StrEnum`](https://docs.python.org/3.11/library/enum.html#enum.StrEnum) 类型，直接支持常见的字符串操作。同时，您可以通过创建一个以`str`和`Enum`为父类的 mixin 类来模拟一个`StrEnum`类的行为。

[*Remove ads*](/account/join/)

### 创建整数标志:`IntFlag`和`Flag`

您可以使用 [`IntFlag`](https://docs.python.org/3/library/enum.html#enum.IntFlag) 作为应该支持位运算符的枚举的基类。对`IntFlag`子类的成员执行按位操作将返回一个对象，该对象也是底层枚举的成员。

下面是一个`Role`枚举的例子，它允许您在单个组合对象中管理不同的用户角色:

>>>

```py
>>> from enum import IntFlag

>>> class Role(IntFlag):
...     OWNER = 8
...     POWER_USER = 4
...     USER = 2
...     SUPERVISOR = 1
...     ADMIN = OWNER | POWER_USER | USER | SUPERVISOR
...

>>> john_roles = Role.USER | Role.SUPERVISOR
>>> john_roles
<Role.USER|SUPERVISOR: 3>

>>> type(john_roles)
<enum 'Role'>

>>> if Role.USER in john_roles:
...     print("John, you're a user")
...
John, you're a user

>>> if Role.SUPERVISOR in john_roles:
...     print("John, you're a supervisor")
...
John, you're a supervisor

>>> Role.OWNER in Role.ADMIN
True

>>> Role.SUPERVISOR in Role.ADMIN
True
```

在此代码片段中，您将创建一个枚举，该枚举保存给定应用程序中的一组用户角色。此枚举的成员保存整数值，您可以使用按位 OR 运算符(`|`)组合这些整数值。例如，名为约翰的用户同时拥有`USER`和`SUPERVISOR`两个角色。注意，存储在`john_roles`中的对象是您的`Role`枚举的成员。

**注意:**你应该记住基于`IntFlag`的枚举的单个成员，也称为**标志**，应该取 2 的幂(1，2，4，8，…)的值。然而，这并不是像`Role.ADMIN`这样的标志组合的必要条件。

在上面的例子中，您将`Role.ADMIN`定义为角色的组合。它的值是通过对枚举中以前角色的完整列表应用按位 OR 运算符而得到的。

`IntFlag`也支持整数运算，比如算术和比较运算。但是，这些类型的操作返回整数而不是成员对象:

>>>

```py
>>> Role.ADMIN + 1
16

>>> Role.ADMIN - 2
13

>>> Role.ADMIN / 3
5.0

>>> Role.ADMIN < 20
True
```

`IntFlag`成员也是`int`的子类。这就是为什么你可以在涉及整数的表达式中使用它们。在这些情况下，结果值将是一个整数，而不是枚举成员。

最后，您还可以在`enum`中找到`Flag`类。这个类的工作方式类似于`IntFlag`，并且有一些额外的限制:

>>>

```py
>>> from enum import Flag 
>>> class Role(Flag): ...     OWNER = 8
...     POWER_USER = 4
...     USER = 2
...     SUPERVISOR = 1
...     ADMIN = OWNER | POWER_USER | USER | SUPERVISOR
...

>>> john_roles = Role.USER | Role.SUPERVISOR
>>> john_roles
<Role.USER|SUPERVISOR: 3>

>>> type(john_roles)
<enum 'Role'>

>>> if Role.USER in john_roles:
...     print("John, you're a user")
...
John, you're a user

>>> if Role.SUPERVISOR in john_roles:
...     print("John, you're a supervisor")
...
John, you're a supervisor

>>> Role.OWNER in Role.ADMIN
True

>>> Role.SUPERVISOR in Role.ADMIN
True

>>> Role.ADMIN + 1 Traceback (most recent call last):
    ...
TypeError: unsupported operand type(s) for +: 'Role' and 'int'
```

`IntFlag`和`Flag`的主要区别在于后者不是从`int`继承的。因此，不支持整数运算。当您试图在整数运算中使用`Role`的成员时，您会得到一个`TypeError`。

就像`IntFlag`枚举的成员一样，`Flag`枚举的成员的值应该是 2 的幂。同样，这不适用于旗帜的组合，就像上面例子中的`Role.ADMIN`。

## 使用枚举:两个实际例子

Python 的枚举可以帮助你提高代码的可读性和组织性。您可以使用它们对相似的常数进行分组，然后在代码中使用这些常数将字符串、数字和其他值替换为可读且有意义的名称。

在接下来的部分中，您将编写几个处理常见枚举用例的实际例子。这些示例将帮助您决定何时您的代码可以从使用枚举中受益。

### 替换幻数

当您需要替换相关幻数集时，例如 HTTP 状态代码、计算机端口和退出代码，枚举非常有用。通过枚举，可以将这些数值常量分组，并为它们分配可读的描述性名称，以便以后在代码中使用和重用。

假设您有以下函数作为应用程序的一部分，该应用程序直接从 web 检索和处理 HTTP 内容:

>>>

```py
>>> from http.client import HTTPSConnection

>>> def process_response(response):
...     match response.getcode():
...         case 200:
...             print("Success!")
...         case 201:
...             print("Successfully created!")
...         case 400:
...             print("Bad request")
...         case 404:
...             print("Not Found")
...         case 500:
...             print("Internal server error")
...         case _:
...             print("Unexpected status")
...

>>> connection = HTTPSConnection("www.python.org")
>>> try:
...     connection.request("GET", "/")
...     response = connection.getresponse()
...     process_response(response)
... finally:
...     connection.close()
...
Success!
```

您的`process_response()`函数接受一个 HTTP `response`对象作为参数。然后它使用`.getcode()`方法从`response`获取状态码。`match` … `case`语句按顺序将当前状态代码与您的示例中作为幻数提供的一些标准状态代码进行比较。

如果出现匹配，则运行匹配的`case`中的代码块。如果不匹配，那么默认的`case`运行。注意，默认的`case`是使用下划线(`_`)作为匹配标准的。

其余代码连接到一个示例网页，执行一个`GET`请求，检索响应对象，并使用您的`process_response()`函数处理它。 [`finally`](https://realpython.com/python-exceptions/#cleaning-up-after-using-finally) 子句关闭活动连接以避免资源泄漏。

尽管这些代码可以工作，但是对于不熟悉 HTTP 状态代码及其相应含义的人来说，阅读和理解这些代码可能会很有挑战性。要解决这些问题并使您的代码更具可读性和可维护性，您可以使用枚举对 HTTP 状态代码进行分组，并为它们提供描述性的名称:

>>>

```py
>>> from enum import IntEnum
>>> from http.client import HTTPSConnection

>>> class HTTPStatusCode(IntEnum): ...     OK = 200
...     CREATED = 201
...     BAD_REQUEST = 400
...     NOT_FOUND = 404
...     SERVER_ERROR = 500
...

>>> def process_response(response):
...     match response.getcode():
...         case HTTPStatusCode.OK: ...             print("Success!")
...         case HTTPStatusCode.CREATED: ...             print("Successfully created!")
...         case HTTPStatusCode.BAD_REQUEST: ...             print("Bad request")
...         case HTTPStatusCode.NOT_FOUND: ...             print("Not Found")
...         case HTTPStatusCode.SERVER_ERROR: ...             print("Internal server error")
...         case _:
...             print("Unexpected status")
...

>>> connection = HTTPSConnection("www.python.org")
>>> try:
...     connection.request("GET", "/")
...     response = connection.getresponse()
...     process_response(response)
... finally:
...     connection.close()
...
Success!
```

这段代码向您的应用程序添加了一个名为`HTTPStatusCode`的新枚举。该枚举将目标 HTTP 状态代码分组，并给它们一个可读的名称。这也使它们严格保持不变，从而使你的应用程序更加可靠。

在`process_response()`中，您使用人类可读的描述性名称来提供上下文和内容信息。现在，任何阅读您的代码的人都会立即知道匹配标准是 HTTP 状态代码。他们还会很快发现每个目标代码的含义。

[*Remove ads*](/account/join/)

### 创建状态机

枚举的另一个有趣的用例是当你使用它们来重新创建一个给定系统的不同的可能状态时。如果您的系统在任何给定的时间都能处于有限状态中的一种，那么您的系统就像一个[状态机](https://en.wikipedia.org/wiki/Finite-state_machine)一样工作。当您需要实现这种常见的设计模式时，枚举非常有用。

作为如何使用 enum 实现状态机模式的例子，您创建了一个最小的[磁盘播放器](https://en.wikipedia.org/wiki/CD_player)模拟器。首先，创建一个包含以下内容的`disk_player.py`文件:

```py
# disk_player.py

from enum import Enum, auto

class State(Enum):
    EMPTY = auto()
    STOPPED = auto()
    PAUSED = auto()
    PLAYING = auto()
```

在这里，您定义了`State`类。这个类将你的磁盘播放器的所有可能状态分组:`EMPTY`、`STOPPED`、`PAUSED`和`PLAYING`。现在你可以编写`DiskPlayer`播放器类，看起来像这样:

```py
# disk_player.py
# ...

class DiskPlayer:
    def __init__(self):
        self.state = State.EMPTY

    def insert_disk(self):
        if self.state is State.EMPTY:
            self.state = State.STOPPED
        else:
            raise ValueError("disk already inserted")

    def eject_disk(self):
        if self.state is State.EMPTY:
            raise ValueError("no disk inserted")
        else:
            self.state = State.EMPTY

    def play(self):
        if self.state in {State.STOPPED, State.PAUSED}:
            self.state = State.PLAYING

    def pause(self):
        if self.state is State.PLAYING:
            self.state = State.PAUSED
        else:
            raise ValueError("can't pause when not playing")

    def stop(self):
        if self.state in {State.PLAYING, State.PAUSED}:
            self.state = State.STOPPED
        else:
            raise ValueError("can't stop when not playing or paused")
```

`DiskPlayer`类实现了您的播放器可以执行的所有可能的操作，包括插入和弹出磁盘、播放、暂停和停止播放器。注意`DiskPlayer`中的每个方法如何利用你的`State`枚举来检查和更新玩家的当前状态。

为了完成您的示例，您将使用传统的 [`if __name__ == "__main__":`](https://realpython.com/if-name-main-python/) 习语来包装几行代码，这些代码将允许您试用`DiskPlayer`类:

```py
# disk_player.py
# ...

if __name__ == "__main__":
    actions = [
        DiskPlayer.insert_disk,
        DiskPlayer.play,
        DiskPlayer.pause,
        DiskPlayer.stop,
        DiskPlayer.eject_disk,
        DiskPlayer.insert_disk,
        DiskPlayer.play,
        DiskPlayer.stop,
        DiskPlayer.eject_disk,
    ]
    player = DiskPlayer()
    for action in actions:
        action(player)
        print(player.state)
```

在这段代码中，您首先定义了一个`actions` [变量](https://realpython.com/python-variables/)，它保存了您将从`DiskPlayer`调用的方法序列，以便测试该类。然后创建一个磁盘播放器类的实例。最后，启动一个`for`循环来遍历动作列表，并通过`player`实例运行每个动作。

就是这样！您的磁盘播放器模拟器已准备好进行测试。要运行它，请在命令行执行以下命令:

```py
$ python disk_player.py
State.STOPPED
State.PLAYING
State.PAUSED
State.STOPPED
State.EMPTY
State.STOPPED
State.PLAYING
State.STOPPED
State.EMPTY
```

该命令的输出显示您的应用程序已经经历了所有可能的状态。当然，这个例子是最小的，没有考虑所有潜在的场景。这是一个演示性的例子，说明了如何使用枚举在代码中实现状态机模式。

## 结论

您现在知道如何在 Python 中创建和使用**枚举**。枚举，或简称为**枚举**，是许多编程语言中常见和流行的数据类型。使用枚举，您可以将相关常量集合分组，并通过枚举本身访问它们。

Python 不提供专用的枚举语法。然而，`enum`模块通过`Enum`类支持这种常见的数据类型。

**在本教程中，您已经学会了如何:**

*   使用 Python 的 **`Enum`** 类创建自己的**枚举**
*   使用枚举及其**成员**
*   为你的枚举类提供**新功能**
*   通过一些**实际例子**使用枚举

您还了解了其他有用的枚举类型，比如`IntEnum`、`IntFlag`和`Flag`。它们在`enum`中可用，将帮助你创建专门的枚举。

有了这些知识，现在就可以开始使用 Python 的枚举对语义相关的常量集进行分组、命名和处理了。枚举允许你更好地组织你的代码，使它更可读，更明确，更易维护。

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/python-enum-code/)，您将使用它在 Python 中构建枚举。*********