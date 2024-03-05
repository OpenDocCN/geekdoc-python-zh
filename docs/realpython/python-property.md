# Python 的 property():向类中添加托管属性

> 原文：<https://realpython.com/python-property/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**用 Python 的属性管理属性()**](/courses/property-python/)

使用 Python 的 [`property()`](https://docs.python.org/3/library/functions.html#property) ，可以在类中创建**托管属性**。当你需要修改它们的内部实现而不改变类的公共 API[时，你可以使用托管属性，也称为**属性**。提供稳定的 API 可以帮助您避免在用户依赖您的类和对象时破坏他们的代码。](https://en.wikipedia.org/wiki/API)

属性可以说是快速创建托管属性的最流行的方式，并且是最纯粹的 Pythonic 风格。

**在本教程中，您将学习如何:**

*   在您的类中创建**托管属性**或**属性**
*   执行**惰性属性评估**并提供**计算属性**
*   避免使用 **setter** 和 **getter** 方法，让你的类更加 Pythonic 化
*   创建**只读**、**读写**和**只写**属性
*   为你的类创建一致的和向后兼容的 API

您还将编写一些使用`property()`来验证输入数据、动态计算属性值、记录代码等等的实际例子。为了充分利用本教程，你应该知道 Python 中的[面向对象](https://realpython.com/python3-object-oriented-programming/)编程和[装饰者](https://realpython.com/primer-on-python-decorators/)的基础知识。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 管理类中的属性

当你用面向对象的编程语言定义一个类时，你可能会得到一些实例和类的 T2 属性。换句话说，根据语言的不同，您最终会得到可以通过实例、类甚至两者访问的变量。属性代表或保存给定对象的内部[状态](https://en.wikipedia.org/wiki/State_(computer_science))，您将经常需要访问和改变它。

通常，您至少有两种方法来管理属性。您可以直接访问和改变属性，也可以使用**方法**。方法是附加到给定类的函数。它们提供了对象可以用其内部数据和属性执行的行为和动作。

如果你向用户公开你的属性，那么它们就成为你的类的公共 API 的一部分。您的用户将直接在他们的代码中访问和修改它们。当您需要更改给定属性的内部实现时，问题就来了。

假设你正在上一门`Circle`课。最初的实现只有一个名为`.radius`的属性。您完成了对类的编码，并使它对您的最终用户可用。他们开始在他们的代码中使用`Circle`来创建许多令人敬畏的项目和应用程序。干得好！

现在假设你有一个重要的用户带着一个新的需求来找你。他们不希望`Circle`再存储半径。他们需要一个公共的`.diameter`属性。

此时，移除`.radius`开始使用`.diameter`可能会破坏一些最终用户的代码。你需要用一种方式来处理这种情况，而不是除掉`.radius`。

像 [Java](https://realpython.com/oop-in-python-vs-java/) 和 [C++](https://en.wikipedia.org/wiki/C%2B%2B) 这样的编程语言鼓励你永远不要暴露你的属性来避免这种问题。相反，您应该提供 [getter 和 setter](https://realpython.com/python-getter-setter/) 方法，也分别称为[访问器](https://en.wikipedia.org/wiki/Accessor_method)和[赋值器](https://en.wikipedia.org/wiki/Mutator_method)。这些方法提供了一种在不改变公共 API 的情况下改变属性内部实现的方法。

**注意:** Getter 和 setter 方法通常被认为是一种[反模式](https://en.wikipedia.org/wiki/Anti-pattern)和糟糕的面向对象设计的标志。这个命题背后的主要论点是，这些方法打破了[封装](https://en.wikipedia.org/wiki/Encapsulation_(computer_programming))。它们允许您访问和改变对象的组件。

最后，这些语言需要 getter 和 setter 方法，因为如果给定的需求发生变化，它们没有提供合适的方法来改变属性的内部实现。更改内部实现需要修改 API，这会破坏最终用户的代码。

[*Remove ads*](/account/join/)

### Python 中的 Getter 和 Setter 方法

从技术上讲，没有什么可以阻止你在 Python 中使用 getter 和 setter [方法](https://realpython.com/python3-object-oriented-programming/#instance-methods)。这种方法看起来是这样的:

```py
# point.py

class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def get_x(self):
        return self._x

    def set_x(self, value):
        self._x = value

    def get_y(self):
        return self._y

    def set_y(self, value):
        self._y = value
```

在本例中，您创建了具有两个非公共属性 `._x`和`._y`的`Point`，以保存手头点的[笛卡尔坐标](https://en.wikipedia.org/wiki/Cartesian_coordinate_system)。

**注意:** Python 没有[访问修饰符](https://en.wikipedia.org/wiki/Access_modifiers)的概念，比如`private`、`protected`、`public`，来限制对属性和方法的访问。在 Python 中，区别在于**公共**和**非公共**类成员。

如果您想表明给定的属性或方法是非公共的，那么您必须使用众所周知的 Python [约定](https://www.python.org/dev/peps/pep-0008/#method-names-and-instance-variables)，在名称前加上下划线(`_`)。这就是属性`._x`和`._y`命名的原因。

注意，这只是一个约定。它不会阻止你和其他程序员使用**点符号**访问属性，就像在`obj._attr`中一样。然而，违反这个惯例是不好的。

要访问和改变`._x`或`._y`的值，可以使用相应的 getter 和 setter 方法。继续将上面的`Point`定义保存在 Python [模块](https://realpython.com/python-modules-packages/)中，然后[将该类导入](https://realpython.com/python-import/)到您的[交互 shell](https://realpython.com/interacting-with-python/) 中。

以下是如何在代码中使用`Point`的方法:

>>>

```py
>>> from point import Point

>>> point = Point(12, 5)
>>> point.get_x()
12
>>> point.get_y()
5

>>> point.set_x(42)
>>> point.get_x()
42

>>> # Non-public attributes are still accessible
>>> point._x
42
>>> point._y
5
```

通过`.get_x()`和`.get_y()`，可以访问`._x`和`._y`的当前值。您可以使用 setter 方法在相应的托管属性中存储新值。从这段代码中，您可以确认 Python 没有限制对非公共属性的访问。你是否这样做取决于你自己。

### Pythonic 式的方法

尽管您刚才看到的例子使用了 Python 编码风格，但它看起来并不像 Python。在这个例子中，getter 和 setter 方法不会对`._x`和`._y`执行任何进一步的处理。你可以用更简洁的方式重写`Point`:

>>>

```py
>>> class Point:
...     def __init__(self, x, y):
...         self.x = x
...         self.y = y
...

>>> point = Point(12, 5)
>>> point.x
12
>>> point.y
5

>>> point.x = 42
>>> point.x
42
```

这部法典揭示了一个基本原则。在 Python 中，向最终用户公开属性是正常和常见的。你不需要总是用 getter 和 setter 方法来混淆你的类，这听起来很酷！然而，如何处理似乎涉及 API 变更的需求变更呢？

与 Java 和 C++不同，Python 提供了方便的工具，允许您在不更改公共 API 的情况下更改属性的底层实现。最流行的方法是将你的属性转化为属性。

**注意:**提供托管属性的另一种常见方法是使用[描述符](https://realpython.com/python-descriptors/)。然而，在本教程中，您将了解属性。

[属性](https://en.wikipedia.org/wiki/Property_(programming))表示普通属性(或字段)和方法之间的中间功能。换句话说，它们允许您创建行为类似于属性的方法。使用属性，您可以在需要时更改计算目标属性的方式。

例如，你可以把`.x`和`.y`都变成属性。通过这一更改，您可以继续将它们作为属性进行访问。您还将拥有一个包含`.x`和`.y`的底层方法，这将允许您修改它们的内部实现，并在您的用户访问和修改它们之前对它们执行操作。

**注意:**属性不是 Python 独有的。诸如 [JavaScript](https://realpython.com/python-vs-javascript/) 、 [C#](https://en.wikipedia.org/wiki/C_Sharp_(programming_language)) 、 [Kotlin](https://en.wikipedia.org/wiki/Kotlin_(programming_language)) 等语言也提供了创建属性作为类成员的工具和技术。

Python 属性的主要优势在于，它们允许您将属性作为公共 API 的一部分公开。如果您需要更改底层实现，那么您可以在任何时候毫不费力地将属性转换为属性。

在接下来的章节中，您将学习如何在 Python 中创建属性。

[*Remove ads*](/account/join/)

## Python 的`property()` 入门

Python 的 [`property()`](https://docs.python.org/3/library/functions.html#property) 是避免代码中正式的 getter 和 setter 方法的 python 方式。该功能允许您将[类属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)转换为**属性**或**管理属性**。由于`property()`是一个内置函数，你可以不用导入任何东西就可以使用它。此外，`property()`是在 C 语言中实现的[以确保最佳性能。](https://github.com/python/cpython/blob/main/Objects/descrobject.c#L1460)

**注意:**通常将`property()`称为内置函数。然而，`property`是一个被设计成作为函数而不是普通类工作的类。这就是为什么大多数 Python 开发者称之为函数。这也是为什么`property()`不遵循 Python 惯例为[命名类](https://www.python.org/dev/peps/pep-0008/#class-names)的原因。

本教程遵循调用`property()`函数而不是类的惯例。然而，在某些部分，你会看到它被称为一个类，以便于解释。

使用`property()`，您可以将 getter 和 setter 方法附加到给定的类属性上。这样，您可以处理该属性的内部实现，而无需在 API 中公开 getter 和 setter 方法。您还可以指定一种处理属性删除的方法，并为您的属性提供一个合适的 [docstring](https://realpython.com/documenting-python-code/) 。

以下是`property()`的完整签名:

```py
property(fget=None, fset=None, fdel=None, doc=None)
```

前两个参数接受将扮演 getter ( `fget`)和 setter ( `fset`)方法角色的函数对象。下面是每个参数的作用总结:

| 争吵 | 描述 |
| --- | --- |
| `fget` | 返回托管属性的值的函数 |
| `fset` | 允许您设置托管属性的值的函数 |
| `fdel` | 函数定义托管属性如何处理删除 |
| `doc` | 表示属性的 docstring 的字符串 |

`property()`的[返回值](https://realpython.com/python-return-statement/)就是被管理的属性本身。如果您访问托管属性，如在`obj.attr`中，那么 Python 会自动调用`fget()`。如果你给属性赋值，比如在`obj.attr = value`中，那么 Python 使用输入`value`作为参数调用`fset()`。最后，如果运行一个`del obj.attr`语句，那么 Python 会自动调用`fdel()`。

**注意:**`property()`的前三个参数取函数对象。您可以将函数对象视为不带调用括号的函数名。

您可以使用`doc`为您的属性提供一个合适的 docstring。您和您的程序员同事将能够使用 Python 的 [`help()`](https://docs.python.org/3/library/functions.html#help) 来读取该文档字符串。当您使用支持文档字符串访问的[代码编辑器和 ide](https://realpython.com/python-ides-code-editors-guide/)时，`doc`参数也很有用。

你可以使用`property()`作为[函数](https://realpython.com/defining-your-own-python-function/)或者[装饰器](https://realpython.com/primer-on-python-decorators/)来构建你的属性。在接下来的两节中，您将学习如何使用这两种方法。然而，您应该预先知道装饰器方法在 Python 社区中更受欢迎。

### 用`property()` 创建属性

您可以通过使用一组适当的参数调用`property()`并将其返回值赋给一个类属性来创建一个属性。`property()`的所有参数都是可选的。然而，你通常至少提供一个**设置函数**。

下面的例子展示了如何创建一个`Circle`类，它有一个方便的属性来管理它的半径:

```py
# circle.py

class Circle:
    def __init__(self, radius):
        self._radius = radius

    def _get_radius(self):
        print("Get radius")
        return self._radius

    def _set_radius(self, value):
        print("Set radius")
        self._radius = value

    def _del_radius(self):
        print("Delete radius")
        del self._radius

    radius = property(
        fget=_get_radius,
        fset=_set_radius,
        fdel=_del_radius,
        doc="The radius property."
    )
```

在这个代码片段中，您创建了`Circle`。类初始化器`.__init__()`将`radius`作为参数，并将其存储在一个名为`._radius`的非公共属性中。然后定义三个非公共方法:

1.  **`._get_radius()`** 返回`._radius`的当前值
2.  **`._set_radius()`** 以`value`为自变量，赋给`._radius`
3.  **`._del_radius()`** 删除实例属性`._radius`

一旦有了这三个方法，就可以创建一个名为`.radius`的类属性来存储 property 对象。为了初始化属性，您将三个方法作为参数传递给`property()`。还可以为您的属性传递一个合适的 docstring。

在这个例子中，您使用[关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-arguments)来提高代码可读性并防止混淆。这样，你就能确切地知道每个参数中使用了哪种方法。

为了尝试一下`Circle`,在您的 Python shell 中运行以下代码:

>>>

```py
>>> from circle import Circle

>>> circle = Circle(42.0)

>>> circle.radius
Get radius
42.0

>>> circle.radius = 100.0
Set radius
>>> circle.radius
Get radius
100.0

>>> del circle.radius
Delete radius
>>> circle.radius
Get radius
Traceback (most recent call last):
    ...
AttributeError: 'Circle' object has no attribute '_radius'

>>> help(circle)
Help on Circle in module __main__ object:

class Circle(builtins.object)
 ...
 |  radius
 |      The radius property.
```

`.radius`属性隐藏了非公共实例属性`._radius`，在本例中它现在是您的托管属性。可以直接访问并分配`.radius`。在内部，Python 会在需要时自动调用`._get_radius()`和`._set_radius()`。当执行`del circle.radius`时，Python 调用`._del_radius()`，删除底层`._radius`。



除了使用常规命名函数在属性中提供 getter 方法之外，还可以使用 [`lambda`](https://realpython.com/python-lambda/) 函数。

下面是`Circle`的一个版本，其中`.radius`属性使用一个`lambda`函数作为它的 getter 方法:

>>>

```py
>>> class Circle:
...     def __init__(self, radius):
...         self._radius = radius
...     radius = property(lambda self: self._radius)
...

>>> circle = Circle(42.0)
>>> circle.radius
42.0
```

如果 getter 方法的功能仅限于返回托管属性的当前值，那么使用`lambda`函数可能是一种方便的方法。

属性是管理**实例属性**的**类属性**。您可以将属性视为捆绑在一起的方法的集合。如果您仔细检查`.radius`，那么您可以发现您提供的原始方法作为`fget`、`fset`和`fdel`参数:

>>>

```py
>>> from circle import Circle

>>> Circle.radius.fget
<function Circle._get_radius at 0x7fba7e1d7d30>

>>> Circle.radius.fset
<function Circle._set_radius at 0x7fba7e1d78b0>

>>> Circle.radius.fdel
<function Circle._del_radius at 0x7fba7e1d7040>

>>> dir(Circle.radius)
[..., '__get__', ..., '__set__', ...]
```

您可以通过相应的`.fget`、`.fset`和`.fdel`来访问给定属性中的 getter、setter 和 deleter 方法。

属性也**覆盖描述符**。如果您使用 [`dir()`](https://realpython.com/python-scope-legb-rule/#dir) 来检查给定属性的内部成员，那么您会在列表中找到`.__set__()`和`.__get__()`。这些方法提供了[描述符协议](https://docs.python.org/3/howto/descriptor.html#descriptor-protocol)的默认实现。

**注:**如果你想更好地理解`property`作为一个类的内部实现，那么就去查阅一下文档中描述的[纯 Python `Property`类](https://docs.python.org/3/howto/descriptor.html#properties)。

例如，`.__set__()`的默认实现在您没有提供自定义 setter 方法时运行。在这种情况下，您会得到一个`AttributeError`,因为没有办法设置底层属性。

[*Remove ads*](/account/join/)

### 使用`property()`作为装饰器

Python 中到处都是装饰者。这些函数将另一个函数作为参数，并返回一个增加了功能的新函数。使用装饰器，您可以将预处理和后处理操作附加到现有的函数上。

当 [Python 2.2](https://docs.python.org/3/whatsnew/2.2.html#attribute-access) 引入`property()`时，装饰器语法不可用。定义属性的唯一方法是传递 getter、setter 和 deleter 方法，正如您之前所学的那样。装饰器语法是在 [Python 2.4](https://docs.python.org/3/whatsnew/2.4.html#pep-318-decorators-for-functions-and-methods) 中添加的，如今，使用`property()`作为装饰器是 Python 社区中最流行的做法。

装饰器语法包括在您想要装饰的函数的定义之前放置带有前导符号`@`的装饰器函数的名称:

```py
@decorator
def func(a):
    return a
```

在这个代码片段中，`@decorator`可以是一个旨在修饰`func()`的函数或类。此语法等效于以下内容:

```py
def func(a):
    return a

func = decorator(func)
```

最后一行代码重新分配名称`func`来保存调用`decorator(func)`的结果。请注意，这与您在上一节中创建属性时使用的语法相同。

Python 的`property()`也可以作为装饰器，所以您可以使用`@property`语法快速创建您的属性:

```py
 1# circle.py
 2
 3class Circle:
 4    def __init__(self, radius):
 5        self._radius = radius
 6
 7    @property
 8    def radius(self):
 9        """The radius property."""
10        print("Get radius")
11        return self._radius
12
13    @radius.setter
14    def radius(self, value):
15        print("Set radius")
16        self._radius = value
17
18    @radius.deleter
19    def radius(self):
20        print("Delete radius")
21        del self._radius
```

这段代码看起来与 getter 和 setter 方法非常不同。现在看起来更蟒蛇和干净。您不再需要使用诸如`._get_radius()`、`._set_radius()`和`._del_radius()`这样的方法名。现在您有了三个方法，它们具有相同的清晰的、描述性的类似属性的名称。这怎么可能呢？

创建属性的装饰方法需要使用底层托管属性的公共名称定义第一个方法，在本例中是`.radius`。这个方法应该实现 getter 逻辑。在上面的例子中，第 7 到 11 行实现了这个方法。

第 13 到 16 行定义了`.radius`的设置方法。在这种情况下，语法相当不同。你不用再次使用`@property`，而是使用`@radius.setter`。你为什么需要这么做？再看一下`dir()`输出:

>>>

```py
>>> dir(Circle.radius)
[..., 'deleter', ..., 'getter', 'setter']
```

除了`.fget`、`.fset`、`.fdel`等一堆特殊属性和方法，`property`还提供了`.deleter()`、`.getter()`、`.setter()`。这三个方法都返回一个新的属性。

当您用`@radius.setter`(第 13 行)修饰第二个`.radius()`方法时，您创建了一个新的属性，并重新分配类级名称`.radius`(第 8 行)来保存它。这个新属性包含与第 8 行的初始属性相同的一组方法，并添加了第 14 行提供的新 setter 方法。最后，装饰语法将新属性重新分配给`.radius`类级别的名称。

定义 deleter 方法的机制是类似的。这一次，您需要使用`@radius.deleter`装饰器。在这个过程的最后，您将获得一个具有 getter、setter 和 deleter 方法的完整属性。

最后，当您使用装饰器方法时，如何为您的属性提供合适的文档字符串？如果您再次检查`Circle`，您会注意到您已经通过在第 9 行向 getter 方法添加一个 docstring 完成了。

新的`Circle`实现与上一节中的示例工作相同:

>>>

```py
>>> from circle import Circle

>>> circle = Circle(42.0)

>>> circle.radius
Get radius
42.0

>>> circle.radius = 100.0
Set radius
>>> circle.radius
Get radius
100.0

>>> del circle.radius
Delete radius
>>> circle.radius
Get radius
Traceback (most recent call last):
    ...
AttributeError: 'Circle' object has no attribute '_radius'

>>> help(circle)
Help on Circle in module __main__ object:

class Circle(builtins.object)
 ...
 |  radius
 |      The radius property.
```

你不需要使用一对括号来调用`.radius()`作为一个方法。相反，您可以像访问常规属性一样访问`.radius`，这是属性的主要用途。它们允许您将方法视为属性，并且它们负责自动调用底层的方法集。

以下是在使用装饰器方法创建属性时需要记住的一些要点:

*   `@property`装饰者必须装饰 **getter 方法**。
*   docstring 必须放在 **getter 方法**中。
*   **setter 和 deleter 方法**必须分别用 getter 方法的名称加上`.setter`和`.deleter`来修饰。

到目前为止，您已经使用`property()`作为函数和装饰器创建了托管属性。如果您检查到目前为止的`Circle`实现，那么您会注意到它们的 getter 和 setter 方法并没有在您的属性之上添加任何真正的额外处理。

一般来说，您应该避免将不需要额外处理的属性变成属性。在这些情况下使用属性可以使您的代码:

*   不必要地冗长
*   令其他开发人员困惑
*   比基于常规属性的代码慢

除非你需要的不仅仅是简单的属性访问，否则不要写属性。它们浪费了 [CPU](https://en.wikipedia.org/wiki/Central_processing_unit) 的时间，更重要的是，它们浪费了*你的*时间。最后，您应该避免编写显式的 getter 和 setter 方法，然后将它们包装在一个属性中。相反，使用`@property`装饰器。这是目前最 Pythonic 化的方法。

[*Remove ads*](/account/join/)

## 提供只读属性

大概`property()`最基本的用例是在你的类中提供**只读属性**。假设您需要一个[不可变的](https://docs.python.org/3/glossary.html#term-immutable) `Point`类，它不允许用户改变其坐标、`x`和`y`的初始值。为了实现这个目标，你可以创建`Point`，如下例所示:

```py
# point.py

class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
```

这里，您将输入参数存储在属性`._x`和`._y`中。正如您已经了解到的，在名称中使用前导下划线(`_`)告诉其他开发人员它们是非公共属性，不应该使用点符号来访问，比如在`point._x`中。最后，定义两个 getter 方法并用`@property`修饰它们。

现在您有两个只读属性，`.x`和`.y`，作为您的坐标:

>>>

```py
>>> from point import Point

>>> point = Point(12, 5)

>>> # Read coordinates
>>> point.x
12
>>> point.y
5

>>> # Write coordinates
>>> point.x = 42
Traceback (most recent call last):
    ...
AttributeError: can't set attribute
```

这里，`point.x`和`point.y`是只读属性的基本示例。他们的行为依赖于`property`提供的底层描述符。正如您已经看到的，当您没有定义一个合适的 setter 方法时，默认的`.__set__()`实现会引发一个`AttributeError`。

您可以将`Point`的实现做得更深入一点，并提供显式的 setter 方法，该方法使用更详细、更具体的消息来引发自定义异常:

```py
# point.py

class WriteCoordinateError(Exception):
    pass

class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        raise WriteCoordinateError("x coordinate is read-only")

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        raise WriteCoordinateError("y coordinate is read-only")
```

在本例中，您定义了一个名为`WriteCoordinateError`的定制异常。这个异常允许您定制实现不可变`Point`类的方式。现在，这两种 setter 方法都用更明确的消息来引发您的自定义异常。来吧，给你的改进`Point`一个尝试！

## 创建读写属性

您还可以使用`property()`为托管属性提供**读写**能力。实际上，您只需要为您的属性提供适当的 getter 方法(“read”)和 setter 方法(“write”)，以便创建读写托管属性。

假设您希望您的`Circle`类有一个`.diameter`属性。然而，在类初始化器中获取半径和直径似乎是不必要的，因为你可以用一个计算另一个。这里有一个将`.radius`和`.diameter`作为读写属性进行管理的`Circle`:

```py
# circle.py

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)

    @property
    def diameter(self):
        return self.radius * 2

    @diameter.setter
    def diameter(self, value):
        self.radius = value / 2
```

这里，您创建了一个具有读写`.radius`的`Circle`类。在这种情况下，getter 方法只返回半径值。setter 方法转换半径的输入值，并将其分配给非公共的`._radius`，这是用于存储最终数据的变量。

在`Circle`及其`.radius`属性的新实现中，有一个微妙的细节需要注意。在这种情况下，类初始化器将输入值直接分配给`.radius`属性，而不是存储在专用的非公共属性中，比如`._radius`。

为什么？因为您需要确保作为半径提供的每个值，包括初始值，都经过 setter 方法并被转换为浮点数。

`Circle`还实现了一个`.diameter`属性作为属性。getter 方法使用半径计算直径。setter 方法做了一些奇怪的事情。它不是将输入直径`value`存储在专用属性中，而是计算半径并将结果写入`.radius`。

以下是您的`Circle`的工作方式:

>>>

```py
>>> from circle import Circle

>>> circle = Circle(42)
>>> circle.radius
42.0

>>> circle.diameter
84.0

>>> circle.diameter = 100
>>> circle.diameter
100.0

>>> circle.radius
50.0
```

在这些例子中，`.radius`和`.diameter`都作为普通属性工作，为你的`Circle`类提供一个干净的 Pythonic 式公共 API。

[*Remove ads*](/account/join/)

## 提供只写属性

您还可以通过调整如何实现属性的 getter 方法来创建**只写**属性。例如，您可以让 getter 方法在每次用户访问底层属性值时引发异常。

以下是使用只写属性处理密码的示例:

```py
# users.py

import hashlib
import os

class User:
    def __init__(self, name, password):
        self.name = name
        self.password = password

    @property
    def password(self):
        raise AttributeError("Password is write-only")

    @password.setter
    def password(self, plaintext):
        salt = os.urandom(32)
        self._hashed_password = hashlib.pbkdf2_hmac(
            "sha256", plaintext.encode("utf-8"), salt, 100_000
        )
```

`User`的初始化器将用户名和密码作为参数，分别存储在`.name`和`.password`中。您使用属性来管理您的类如何处理输入密码。每当用户试图检索当前密码时，getter 方法就会引发一个`AttributeError`。这将`.password`变成了只写属性:

>>>

```py
>>> from users import User

>>> john = User("John", "secret")

>>> john._hashed_password
b'b\xc7^ai\x9f3\xd2g ... \x89^-\x92\xbe\xe6'

>>> john.password
Traceback (most recent call last):
    ...
AttributeError: Password is write-only

>>> john.password = "supersecret"
>>> john._hashed_password
b'\xe9l$\x9f\xaf\x9d ... b\xe8\xc8\xfcaU\r_'
```

在这个例子中，您创建了一个带有初始密码的`john`实例。setter 方法将密码散列并存储在`._hashed_password`中。注意，当你试图直接访问`.password`时，你会得到一个`AttributeError`。最后，给`.password`分配一个新值会触发 setter 方法并创建一个新的散列密码。

在`.password`的 setter 方法中，你使用`os.urandom()`生成一个 32 字节的随机[字符串](https://realpython.com/python-strings/)作为你的散列函数的[盐](https://en.wikipedia.org/wiki/Salt_(cryptography))。要生成散列密码，可以使用 [`hashlib.pbkdf2_hmac()`](https://docs.python.org/3/library/hashlib.html#hashlib.pbkdf2_hmac) 。然后将得到的散列密码存储在非公共属性`._hashed_password`中。这样做可以确保您永远不会将明文密码保存在任何可检索的属性中。

## 将 Python 的`property()`付诸行动

到目前为止，您已经学习了如何使用 Python 的`property()`内置函数在您的类中创建托管属性。您将`property()`用作函数和装饰器，并了解了这两种方法之间的区别。您还学习了如何创建只读、读写和只写属性。

在接下来的部分中，您将编写几个例子来帮助您更好地理解`property()`的常见用例。

### 验证输入值

`property()`最常见的用例之一是构建托管属性，这些属性在存储输入数据或将其作为安全输入接受之前对其进行验证。[数据验证](https://en.wikipedia.org/wiki/Data_validation)是代码中的一个常见需求，它从用户或其他您认为不可信的信息源获取输入。

Python 的`property()`提供了一个快速可靠的工具来处理输入数据验证。例如，回想一下`Point`的例子，您可能要求`.x`和`.y`的值是有效的[数字](https://realpython.com/python-numbers/)。因为您的用户可以自由输入任何类型的数据，所以您需要确保您的点只接受数字。

这里有一个管理这个需求的`Point`的实现:

```py
# point.py

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        try:
            self._x = float(value)
            print("Validated!")
        except ValueError:
            raise ValueError('"x" must be a number') from None

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        try:
            self._y = float(value)
            print("Validated!")
        except ValueError:
            raise ValueError('"y" must be a number') from None
```

`.x`和`.y`的 setter 方法使用 [`try` … `except`](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions) 块，这些块使用 Python [EAFP](https://docs.python.org/3/glossary.html#term-eafp) 风格验证输入数据。如果对`float()`的调用成功，那么输入的数据是有效的，屏幕上会显示`Validated!`。如果`float()`引发了一个`ValueError`，那么用户将得到一个`ValueError`和一个更具体的消息。

**注意:**在上面的例子中，您使用语法 [`raise` … `from None`](https://stackoverflow.com/questions/24752395/python-raise-from-usage/) 来隐藏与引发异常的上下文相关的内部细节。从最终用户的角度来看，这些细节可能会令人困惑，使您的类看起来不完美。

查看文档中关于 [`raise`语句](https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement)的部分，了解关于这个主题的更多信息。

值得注意的是，在`.__init__()`中直接分配`.x`和`.y`属性确保了验证也发生在对象初始化期间。在使用`property()`进行数据验证时，不这样做是一个常见的错误。

下面是您的`Point`类现在的工作方式:

>>>

```py
>>> from point import Point

>>> point = Point(12, 5)
Validated!
Validated!
>>> point.x
12.0
>>> point.y
5.0

>>> point.x = 42
Validated!
>>> point.x
42.0

>>> point.y = 100.0
Validated!
>>> point.y
100.0

>>> point.x = "one"
Traceback (most recent call last):
     ...
ValueError: "x" must be a number

>>> point.y = "1o"
Traceback (most recent call last):
    ...
ValueError: "y" must be a number
```

如果给`.x`和`.y`赋值，使得`float()`可以转换成浮点数，那么验证就成功了，该值被接受。否则，你得到一个`ValueError`。

`Point`的实现揭示了`property()`的一个根本弱点。你发现了吗？

就是这样！您有遵循特定模式的重复代码。这种重复打破了 [DRY(不要重复自己)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)原则，所以你会想要[重构](https://realpython.com/python-refactoring/)这段代码来避免它。为此，您可以使用描述符抽象出重复的逻辑:

```py
# point.py

class Coordinate:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        try:
            instance.__dict__[self._name] = float(value)
            print("Validated!")
        except ValueError:
            raise ValueError(f'"{self._name}" must be a number') from None

class Point:
    x = Coordinate()
    y = Coordinate()

    def __init__(self, x, y):
        self.x = x
        self.y = y
```

现在你的代码更短了。通过将`Coordinate`定义为在一个地方管理数据验证的[描述符](https://realpython.com/python-descriptors/)，您成功地删除了重复代码。代码的工作方式就像您之前的实现一样。来吧，试一试！

一般来说，如果您发现自己在代码中到处复制和粘贴属性定义，或者发现了重复的代码，就像上面的例子一样，那么您应该考虑使用适当的描述符。

[*Remove ads*](/account/join/)

### 提供计算属性

如果您需要一个无论何时访问都可以动态构建其值的属性，那么`property()`就是合适的选择。这些类型的属性通常被称为**计算属性**。当你需要他们看起来像[渴望](https://en.wikipedia.org/wiki/Eager_evaluation)属性，但你希望他们[懒惰](https://en.wikipedia.org/wiki/Lazy_evaluation)时，他们很方便。

创建渴望属性的主要原因是为了在经常访问属性时优化计算成本。另一方面，如果你很少使用一个给定的属性，那么一个惰性属性可以把它的计算推迟到需要的时候，这可以让你的程序更有效率。

下面是一个如何使用`property()`在`Rectangle`类中创建计算属性`.area`的例子:

```py
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height
```

在这个例子中，`Rectangle`初始化器将`width`和`height`作为参数，并将它们存储在常规的实例属性中。只读属性`.area`在你每次访问它的时候计算并返回当前矩形的面积。

属性的另一个常见用例是为给定属性提供自动格式化的值:

```py
class Product:
    def __init__(self, name, price):
        self._name = name
        self._price = float(price)

    @property
    def price(self):
        return f"${self._price:,.2f}"
```

在本例中，`.price`是一个格式化并返回特定产品价格的属性。为了提供类似货币的格式，您使用一个带有适当格式选项的 [f 字符串](https://realpython.com/python-f-strings/)。

**注意:**这个例子使用浮点数来表示货币，这是不好的做法。相反，你应该使用来自[标准库](https://docs.python.org/3/library/index.html)的 [`decimal.Decimal`](https://docs.python.org/3/library/decimal.html#decimal.Decimal) 。

作为计算属性的最后一个例子，假设您有一个使用`.x`和`.y`作为笛卡尔坐标的`Point`类。您希望为您的点提供[极坐标](https://en.wikipedia.org/wiki/Polar_coordinate_system)，以便在一些计算中使用它们。极坐标系统使用到原点的距离和与水平坐标轴的角度来表示每个点。

这里有一个笛卡尔坐标`Point`类，它也提供计算极坐标:

```py
# point.py

import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def distance(self):
        return round(math.dist((0, 0), (self.x, self.y)))

    @property
    def angle(self):
        return round(math.degrees(math.atan(self.y / self.x)), 1)

    def as_cartesian(self):
        return self.x, self.y

    def as_polar(self):
        return self.distance, self.angle
```

这个例子展示了如何使用给定的`Point`对象的`.x`和`.y`笛卡尔坐标来计算其距离和角度。下面是这段代码在实践中的工作方式:

>>>

```py
>>> from point import Point

>>> point = Point(12, 5)

>>> point.x
12
>>> point.y
5

>>> point.distance
13
>>> point.angle
22.6

>>> point.as_cartesian()
(12, 5)
>>> point.as_polar()
(13, 22.6)
```

在提供计算属性或惰性属性时，`property()`是一个非常方便的工具。但是，如果您正在创建一个经常使用的属性，那么每次都计算它可能会非常昂贵和浪费。一个好的策略是一旦计算完成，T1 就缓存 T2。

### 缓存计算属性

有时，您有一个经常使用的给定计算属性。不断重复同样的计算可能是不必要的，也是昂贵的。要解决这个问题，您可以缓存计算出的值，并将其保存在一个非公共的专用属性中，以便进一步重用。

为了防止意外行为，您需要考虑输入数据的可变性。如果您有一个从常量输入值计算其值的属性，那么结果永远不会改变。在这种情况下，您可以只计算一次该值:

```py
# circle.py

from time import sleep

class Circle:
    def __init__(self, radius):
        self.radius = radius
        self._diameter = None

    @property
    def diameter(self):
        if self._diameter is None:
            sleep(0.5)  # Simulate a costly computation
            self._diameter = self.radius * 2
        return self._diameter
```

尽管`Circle`的这个实现正确地缓存了计算出的直径，但是它有一个缺点，如果你改变了`.radius`的值，那么`.diameter`将不会返回一个正确的值:

>>>

```py
>>> from circle import Circle

>>> circle = Circle(42.0)
>>> circle.radius
42.0

>>> circle.diameter  # With delay
84.0
>>> circle.diameter  # Without delay
84.0

>>> circle.radius = 100.0
>>> circle.diameter  # Wrong diameter
84.0
```

在这些示例中，您创建了一个半径等于`42.0`的圆。只有在第一次访问时,`.diameter`属性才会计算它的值。这就是为什么您在第一次执行中看到延迟，而在第二次执行中没有延迟。请注意，即使您更改了半径值，直径也保持不变。

如果计算属性的输入数据发生变化，则需要重新计算该属性:

```py
# circle.py

from time import sleep

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._diameter = None
        self._radius = value

    @property
    def diameter(self):
        if self._diameter is None:
            sleep(0.5)  # Simulate a costly computation
            self._diameter = self._radius * 2
        return self._diameter
```

每当您更改半径的值时，`.radius`属性的 setter 方法会将`._diameter`重置为 [`None`](https://realpython.com/null-in-python/) 。有了这个小小的更新，`.diameter`在`.radius`的每一次突变后，第一次访问它时会重新计算它的值:

>>>

```py
>>> from circle import Circle

>>> circle = Circle(42.0)

>>> circle.radius
42.0
>>> circle.diameter  # With delay
84.0
>>> circle.diameter  # Without delay
84.0

>>> circle.radius = 100.0
>>> circle.diameter  # With delay
200.0
>>> circle.diameter  # Without delay
200.0
```

酷！现在可以正常工作了！它会在您第一次访问它以及每次更改半径时计算直径。

创建缓存属性的另一个选项是使用标准库中的 [`functools.cached_property()`](https://docs.python.org/3/library/functools.html#functools.cached_property) 。这个函数就像一个装饰器，允许你将一个方法转换成一个缓存的属性。属性只计算其值一次，并在实例的生存期内将其作为普通属性进行缓存:

```py
# circle.py

from functools import cached_property
from time import sleep

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @cached_property
    def diameter(self):
        sleep(0.5)  # Simulate a costly computation
        return self.radius * 2
```

在这里，`.diameter`在您第一次访问它时计算并缓存它的值。这种实现适用于输入值不会发生变化的计算。它是这样工作的:

>>>

```py
>>> from circle import Circle

>>> circle = Circle(42.0)
>>> circle.diameter  # With delay
84.0
>>> circle.diameter  # Without delay
84.0

>>> circle.radius = 100
>>> circle.diameter  # Wrong diameter
84.0

>>> # Allow direct assignment
>>> circle.diameter = 200
>>> circle.diameter  # Cached value
200
```

当你访问`.diameter`时，你得到它的计算值。从现在开始，这个值保持不变。然而，与`property()`不同的是，`cached_property()`不会阻止属性突变，除非你提供一个合适的 setter 方法。这就是为什么您可以在最后几行中将直径更新为`200`。

如果您想创建一个不允许修改的缓存属性，那么您可以使用`property()`和 [`functools.cache()`](https://docs.python.org/3/library/functools.html#functools.cache) ，如下例所示:

```py
# circle.py

from functools import cache
from time import sleep

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    @cache
    def diameter(self):
        sleep(0.5) # Simulate a costly computation
        return self.radius * 2
```

这段代码将`@property`堆叠在`@cache`之上。两个装饰器的组合构建了一个防止突变的缓存属性:

>>>

```py
>>> from circle import Circle

>>> circle = Circle(42.0)

>>> circle.diameter  # With delay
84.0
>>> circle.diameter  # Without delay
84.0

>>> circle.radius = 100
>>> circle.diameter
84.0

>>> circle.diameter = 200
Traceback (most recent call last):
    ...
AttributeError: can't set attribute
```

在这些例子中，当你试图给`.diameter`赋值时，你会得到一个`AttributeError`，因为 setter 功能来自于`property`的内部描述符。

[*Remove ads*](/account/join/)

### 记录属性访问和突变

有时你需要跟踪你的代码做了什么，你的程序是如何运行的。在 Python 中这样做的一个方法是使用 [`logging`](https://realpython.com/python-logging/) 。该模块提供了记录代码所需的所有功能。它将允许您不断地观察代码，并生成关于它如何工作的有用信息。

如果您需要跟踪访问和变更给定属性的方式和时间，那么您也可以利用`property()`来实现:

```py
# circle.py

import logging

logging.basicConfig(
    format="%(asctime)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)

class Circle:
    def __init__(self, radius):
        self._msg = '"radius" was %s. Current value: %s'
        self.radius = radius

    @property
    def radius(self):
        """The radius property."""
        logging.info(self._msg % ("accessed", str(self._radius)))
        return self._radius

    @radius.setter
    def radius(self, value):
        try:
            self._radius = float(value)
            logging.info(self._msg % ("mutated", str(self._radius)))
        except ValueError:
            logging.info('validation error while mutating "radius"')
```

在这里，您首先导入`logging`并定义一个基本配置。然后用一个托管属性`.radius`实现`Circle`。每次您在代码中访问`.radius`时，getter 方法都会生成日志信息。setter 方法记录您在`.radius`上执行的每一个突变。它还记录了由于错误的输入数据而导致错误的情况。

下面是如何在代码中使用`Circle`:

>>>

```py
>>> from circle import Circle

>>> circle = Circle(42.0)

>>> circle.radius
14:48:59: "radius" was accessed. Current value: 42.0
42.0

>>> circle.radius = 100
14:49:15: "radius" was mutated. Current value: 100

>>> circle.radius
14:49:24: "radius" was accessed. Current value: 100
100

>>> circle.radius = "value"
15:04:51: validation error while mutating "radius"
```

记录来自属性访问和变异的有用数据可以帮助您调试代码。日志记录还可以帮助您识别有问题的数据输入的来源，分析代码的性能，发现使用模式等等。

### 管理属性删除

您还可以创建实现删除功能的属性。这可能是`property()`的一个罕见用例，但是在某些情况下，拥有一种删除属性的方法会很方便。

假设您正在实现自己的[树](https://en.wikipedia.org/wiki/Tree_(data_structure))数据类型。树是一种[抽象数据类型](https://en.wikipedia.org/wiki/Abstract_data_type)，它以层次结构存储元素。树组件通常被称为**节点**。除了根节点之外，树中的每个节点都有一个父节点。节点可以有零个或多个子节点。

现在假设您需要提供一种方法来删除或清除给定节点的子节点列表。下面的例子实现了一个使用`property()`来提供大部分功能的树节点，包括清除手边节点的子节点列表的能力:

```py
# tree.py

class TreeNode:
    def __init__(self, data):
        self._data = data
        self._children = []

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, value):
        if isinstance(value, list):
            self._children = value
        else:
            del self.children
            self._children.append(value)

    @children.deleter
    def children(self):
        self._children.clear()

    def __repr__(self):
        return f'{self.__class__.__name__}("{self._data}")'
```

在本例中，`TreeNode`表示自定义树数据类型中的一个节点。每个节点将它的孩子存储在一个 Python [列表](https://realpython.com/python-lists-tuples/)中。然后将`.children`实现为一个属性来管理底层的子列表。deleter 方法调用子列表上的`.clear()`将它们全部删除:

>>>

```py
>>> from tree import TreeNode

>>> root = TreeNode("root")
>>> child1 = TreeNode("child 1")
>>> child2 = TreeNode("child 2")

>>> root.children = [child1, child2]

>>> root.children
[TreeNode("child 1"), TreeNode("child 2")]

>>> del root.children
>>> root.children
[]
```

这里，首先创建一个`root`节点来开始填充树。然后创建两个新节点，并使用一个列表将它们分配给`.children`。 [`del`](https://realpython.com/python-keywords/#the-del-keyword) 语句触发`.children`的内部 deleter 方法，清空列表。

### 创建向后兼容的类 API

正如您已经知道的，属性将方法调用转化为直接的属性查找。这个特性允许您为您的类创建干净的 Pythonic 式 API。您可以公开您的属性，而不需要 getter 和 setter 方法。

如果您需要修改如何计算一个给定的公共属性，那么您可以将它转换成一个属性。属性使执行额外的处理成为可能，比如数据验证，而不必修改公共 API。

假设您正在创建一个会计应用程序，并且需要一个基类来管理货币。为此，您创建了一个`Currency`类，它公开了两个属性`.units`和`.cents`:

```py
class Currency:
    def __init__(self, units, cents):
        self.units = units
        self.cents = cents

    # Currency implementation...
```

这个类看起来干净而有 Pythonic 风格。现在假设您的需求发生了变化，您决定存储美分的总数，而不是单位和美分。从你的公共 API 中移除`.units`和`.cents`来使用类似`.total_cents`的东西会破坏不止一个客户的代码。

在这种情况下，`property()`可能是保持当前 API 不变的绝佳选择。以下是解决这个问题并避免破坏客户代码的方法:

```py
# currency.py

CENTS_PER_UNIT = 100

class Currency:
    def __init__(self, units, cents):
        self._total_cents = units * CENTS_PER_UNIT + cents

    @property
    def units(self):
        return self._total_cents // CENTS_PER_UNIT

    @units.setter
    def units(self, value):
        self._total_cents = self.cents + value * CENTS_PER_UNIT

    @property
    def cents(self):
        return self._total_cents % CENTS_PER_UNIT

    @cents.setter
    def cents(self, value):
        self._total_cents = self.units * CENTS_PER_UNIT + value

    # Currency implementation...
```

现在，您的类存储美分的总数，而不是独立的单位和美分。然而，您的用户仍然可以访问和修改他们代码中的`.units`和`.cents`,并得到和以前一样的结果。来吧，试一试！

当你写一些很多人将要构建的东西时，你需要保证对内部实现的修改不会影响最终用户使用你的类的方式。

[*Remove ads*](/account/join/)

## 覆盖子类中的属性

当您创建包含属性的 Python 类并在包或库中发布它们时，您应该预料到您的用户会用它们做许多不同的事情。其中之一可能是**对**进行子类化以定制它们的功能。在这些情况下，你的用户必须小心，并意识到一个微妙的陷阱。如果您部分覆盖了一个属性，那么您将失去未被覆盖的功能。

例如，假设您正在编写一个`Employee`类来管理公司内部会计系统中的员工信息。您已经有了一个名为`Person`的类，并且您想对它进行子类化以重用它的功能。

`Person`有一个作为属性实现的`.name`属性。`.name`的当前实现不满足以大写字母返回名称的要求。这就是你最终解决这个问题的方法:

```py
# persons.py

class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    # Person implementation...

class Employee(Person):
    @property
    def name(self):
        return super().name.upper()

    # Employee implementation...
```

在`Employee`中，您覆盖了`.name`以确保当您访问属性时，您获得大写的雇员姓名:

>>>

```py
>>> from persons import Employee, Person

>>> person = Person("John")
>>> person.name
'John'
>>> person.name = "John Doe"
>>> person.name
'John Doe'

>>> employee = Employee("John")
>>> employee.name
'JOHN'
```

太好了！`Employee`随心所欲！它使用大写字母返回名称。然而，随后的测试发现了一个意想不到的行为:

>>>

```py
>>> employee.name = "John Doe"
Traceback (most recent call last):
    ...
AttributeError: can't set attribute
```

发生了什么事？当你从一个父类中重写一个现有的属性时，你重写了那个属性的全部功能。在这个例子中，您只重新实现了 getter 方法。因此，`.name`失去了基类的其余功能。你不再有 setter 方法了。

这个想法是，如果你需要在子类中覆盖一个属性，那么你应该在你手头的属性的新版本中提供所有你需要的功能。

## 结论

属性是一种特殊类型的类成员，它提供了介于常规属性和方法之间的功能。属性允许您修改实例属性的实现，而无需更改该类的公共 API。能够保持 API 不变有助于避免破坏用户在旧版本类上编写的代码。

属性是在类中创建**托管属性**的[python 式](https://realpython.com/learning-paths/writing-pythonic-code/)方法。它们在现实编程中有几个用例，这使它们成为 Python 开发人员技能的重要补充。

**在本教程中，您学习了如何:**

*   用 Python 的`property()`创建**托管属性**
*   执行**惰性属性评估**并提供**计算属性**
*   避免 **setter** 和 **getter** 方法带有属性
*   创建**只读**、**读写**和**只写**属性
*   为你的类创建一致的和向后兼容的 API

您还编写了几个实际例子，带您了解最常见的`property()`用例。这些例子包括输入[数据验证](#validating-input-values)，计算属性，[记录](https://realpython.com/python-logging/)您的代码，等等。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**用 Python 的属性管理属性()**](/courses/property-python/)**********