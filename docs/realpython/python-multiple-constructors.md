# 在 Python 类中提供多个构造函数

> 原文：<https://realpython.com/python-multiple-constructors/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深你的理解: [**在你的 Python 类中提供多个构造器**](/courses/multiple-constructors-python/)

有时，您需要编写一个 Python 类，提供多种方法来构造对象。换句话说，你想要一个实现了**多个构造函数**的类。当您需要使用不同类型或数量的参数创建实例时，这种类型的类非常方便。拥有提供多个构造函数的工具将帮助您编写灵活的类，以适应不断变化的需求。

在 Python 中，有几种技术和工具可以用来构造类，包括通过可选参数模拟多个构造函数，通过类方法定制实例创建，以及使用 decorators 进行特殊调度。如果您想了解这些技术和工具，那么本教程就是为您准备的。

**在本教程中，您将学习如何:**

*   使用**可选参数**和**类型检查**模拟多个构造函数
*   使用内置的 **`@classmethod`** 装饰器编写多个构造函数
*   使用 **`@singledispatchmethod`** 装饰器重载你的类构造函数

您还将看到 Python 如何在内部**构造一个常规类的实例**，以及一些**标准库类**如何提供多个构造函数。

为了从本教程中获得最大收益，你应该具备[面向对象编程](https://realpython.com/python3-object-oriented-programming/)的基础知识，并了解如何用`@classmethod`定义[类方法](https://realpython.com/instance-class-and-static-methods-demystified/)。你也应该有在 Python 中使用[装饰器](https://realpython.com/primer-on-python-decorators/)的经验。

**免费奖励:** [点击此处获取免费的 Python OOP 备忘单](https://realpython.com/bonus/python-oop/)，它会为你指出最好的教程、视频和书籍，让你了解更多关于 Python 面向对象编程的知识。

## 在 Python 中实例化类

Python 用易于创建和使用的**类**支持[面向对象编程](https://realpython.com/python3-object-oriented-programming/)。Python 类提供了强大的特性，可以帮助你编写更好的软件。类就像是**对象**的蓝图，也称为**实例**。用同样的方式，你可以从一个蓝图中构建几个房子，你也可以从一个类中构建几个实例。

要在 Python 中定义一个类，需要使用 [`class`](https://realpython.com/python-keywords/#structure-keywords-def-class-with-as-pass-lambda) 关键字，后跟类名:

>>>

```py
>>> # Define a Person class
>>> class Person:
...     def __init__(self, name):
...         self.name = name
...
```

Python 有一套丰富的[特殊方法](https://docs.python.org/3/glossary.html#term-special-method)，你可以在你的类中使用。Python 隐式调用特殊方法来自动执行实例上的各种操作。有一些特殊的方法可以使你的对象可迭代，为你的对象提供合适的字符串表示，初始化实例属性，等等。

一个相当常见的特殊方法是 [`.__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__) 。这个方法提供了 Python 中所谓的**实例初始化器**。这个方法的工作是在实例化一个给定的类时，用适当的值初始化[实例属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)。

在`Person`中，`.__init__()`方法的第一个参数被称为`self`。此参数保存当前对象或实例，它在方法调用中隐式传递。Python 中的每个[实例方法](https://realpython.com/python3-object-oriented-programming/#instance-methods)都有这个参数。`.__init__()`的第二个参数叫做`name`，它将把人名作为一个[字符串](https://realpython.com/python-strings/)保存。

**注意:**使用 [`self`](https://www.python.org/dev/peps/pep-0008/#function-and-method-arguments) 来命名当前对象在 Python 中是一个相当强的约定，但不是必需的。然而，使用另一个名字会让你的 Python 开发伙伴感到惊讶。

一旦定义了一个类，就可以开始用**实例化**它。换句话说，您可以开始创建该类的对象。为此，您将使用熟悉的语法。只需使用一对括号(`()`)调用该类，这与您调用任何 Python [函数](https://realpython.com/defining-your-own-python-function/)的语法相同:

>>>

```py
>>> # Instantiating Person
>>> john = Person("John Doe")
>>> john.name
'John Doe'
```

在 Python 中，类名提供了其他语言如 [C++](https://realpython.com/python-vs-cpp/) 和 [Java](https://realpython.com/java-vs-python/) 调用**类构造函数**的内容。调用一个类，就像使用`Person`一样，触发 Python 的类**实例化过程**，该过程在内部分两步运行:

1.  **创建**目标类的一个新实例。
2.  **用合适的实例属性值初始化**实例。

继续上面的例子，作为参数传递给`Person`的值在内部传递给`.__init__()`，然后分配给实例属性`.name`。这样，您就用有效数据初始化了 person 实例`john`，您可以通过访问`.name`来确认这些数据。成功！`John Doe`的确是他的名字。

**注意:**当你调用这个类来创建一个新的实例时，你需要提供`.__init__()`需要的尽可能多的参数，这样这个方法就可以初始化所有需要初始值的实例属性。

现在您已经理解了对象初始化机制，您已经准备好学习 Python 在实例化过程的这一点之前做了什么。下面我们来挖掘另一种特殊的方法，叫做 [`.__new__()`](https://docs.python.org/3/reference/datamodel.html#object.__new__) 。这个方法负责在 Python 中创建新的实例。

**注意:**`.__new__()`特殊方法在 Python 中经常被称为**类构造函数**。然而，它的工作实际上是从类蓝图创建新的对象，所以你可以更准确地称它为**实例创建者**或**对象创建者**。

特殊方法`.__new__()`将底层类作为它的第一个参数，并返回一个新对象。这个对象通常是输入类的一个实例，但是在某些情况下，它可以是不同类的一个实例。

如果`.__new__()`返回的对象是当前类的一个实例，那么这个实例会立即传递给`.__init__()`进行初始化。这两个步骤在您调用类时运行。

Python 的 [`object`](https://docs.python.org/3/library/functions.html#object) 类提供了`.__new__()`和`.__init__()`的基础或默认实现。与`.__init__()`不同，您很少需要在自定义类中覆盖`.__new__()`。大多数时候，您可以放心地依赖它的默认实现。

总结一下到目前为止您所学到的内容，Python 的实例化过程从您用适当的参数调用一个类开始。然后，该过程分两步进行:用`.__new__()`方法创建对象，用`.__init__()`方法初始化对象。

既然您已经了解了 Python 的这种内部行为，那么您就可以开始在您的类中提供多个构造函数了。换句话说，您将提供多种方法来构造给定 Python 类的对象。

[*Remove ads*](/account/join/)

## 定义多个类构造函数

有时你想写一个类，允许你使用不同数据类型的参数甚至不同数量的参数来构造对象。实现这一点的一种方法是在手边的类中提供多个构造函数。每个构造函数都允许您使用一组不同的参数创建该类的实例。

一些编程语言，如 [C++](https://realpython.com/python-vs-cpp/) 、 [C#](https://en.wikipedia.org/wiki/C_Sharp_(programming_language)) 和 [Java](https://realpython.com/oop-in-python-vs-java/) ，支持所谓的[函数或方法重载](https://en.wikipedia.org/wiki/Function_overloading)。这个特性允许您提供多个类构造函数，因为它允许您创建多个具有相同名称和不同实现的函数或方法。

方法重载意味着根据你调用方法的方式，语言会选择合适的实现来运行。因此，您的方法可以根据调用的上下文执行不同的任务。

不幸的是，Python 不直接支持函数重载。Python 类将方法名保存在名为 [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 的内部[字典](https://realpython.com/python-dicts/)中，该字典保存类的[名称空间](https://realpython.com/python-namespaces-scope/)。像任何 Python 字典一样，`.__dict__`不能有重复的键，所以在给定的类中不能有多个同名的方法。如果您尝试这样做，那么 Python 将只记住手头方法的最后一个实现:

```py
# greet.py

class Greeter:
    def say_hello(self):
        print("Hello, World")

    def say_hello(self):
        print("Hello, Pythonista")
```

在这个例子中，您用两个方法创建了一个 Python 类`Greeter`。这两种方法名称相同，但实现略有不同。

要了解当两个方法同名时会发生什么，请将您的类保存到工作目录中的一个`greet.py`文件中，并在一个[交互会话](https://realpython.com/interacting-with-python/)中运行以下代码:

>>>

```py
>>> from greet import Greeter

>>> greeter = Greeter()

>>> greeter.say_hello()
Hello, Pythonista

>>> Greeter.__dict__
mappingproxy({..., 'say_hello': <function Greeter.say_hello at...>, ...})
```

在这个例子中，您在`greeter`上调用`.say_hello()`，它是`Greeter`类的一个实例。您在屏幕上看到的是`Hello, Pythonista`而不是`Hello, World`，这证实了该方法的第二个实现优先于第一个实现。

最后一行代码检查了`.__dict__`的内容，发现方法名`say_hello`在类名称空间中只出现了一次。这与 Python 中字典的工作方式是一致的。

Python 模块和交互式会话中的函数也会发生类似的情况。几个同名函数的最后一个实现优先于其余的实现:

>>>

```py
>>> def say_hello():
...     print("Hello, World")
...

>>> def say_hello():
...     print("Hello, Pythonista")
...

>>> say_hello()
Hello Pythonista
```

您在同一个解释器会话中定义了两个同名函数`say_hello()`。但是，第二个定义会覆盖第一个定义。当你调用函数时，你得到`Hello, Pythonista`，它确认最后一个函数定义生效。

一些编程语言用来提供多种方法调用方法或函数的另一种技术是[多重分派](https://en.wikipedia.org/wiki/Multiple_dispatch)。

使用这种技术，您可以编写同一个方法或函数的几种不同的实现，并根据调用中使用的参数的类型或其他特征来动态调度所需的实现。您可以使用来自[标准库](https://docs.python.org/3/library/index.html)的几个工具将这项技术引入到您的 Python 代码中。

Python 是一种相当灵活且功能丰富的语言，它提供了多种方法来实现多个构造函数，并使您的类更加灵活。

在下一节中，您将通过传递[可选参数](https://realpython.com/python-optional-arguments/)并通过检查参数类型来确定实例初始化器中的不同行为，从而模拟多个构造函数。

## 在你的类中模拟多个构造函数

在 Python 类中模拟多个构造函数的一个非常有用的技术是使用**默认参数值**为`.__init__()`提供**可选参数**。这样，您可以用不同的方式调用类构造函数，每次都能获得不同的行为。

另一个策略是检查`.__init__()`的参数的**数据类型**，根据您在调用中传递的具体数据类型提供不同的行为。这种技术允许您在一个类中模拟多个构造函数。

在本节中，您将学习如何通过为`.__init__()`方法的参数提供适当的默认值以及检查该方法参数的数据类型来模拟多种构造对象的方法。这两种方法都只需要一个`.__init__()`的实现。

[*Remove ads*](/account/join/)

### 使用`.__init__()`中的可选参数值

模拟多个构造函数的一种优雅而巧妙的方式是用可选参数实现一个`.__init__()`方法。您可以通过指定适当的[默认参数值](https://realpython.com/defining-your-own-python-function/#default-parameters)来做到这一点。

**注意:**你也可以在你的函数和方法中使用未定义数量的[位置参数](https://realpython.com/defining-your-own-python-function/#positional-arguments)或者未定义数量的[关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-arguments)来提供可选参数。查看[定义函数时使用 Python 可选参数](https://realpython.com/python-optional-arguments/)以获得关于这些选项的更多细节。

为此，假设您需要编写一个名为`CumulativePowerFactory`的[工厂](https://realpython.com/factory-method-python/)类。这个类将使用一串数字作为输入，创建计算特定幂的可调用对象。您还需要您的类来跟踪连续幂的总和。最后，您的类应该接受一个参数，该参数包含幂和的初始值。

继续在当前目录下创建一个`power.py`文件。然后输入下面的代码来实现`CumulativePowerFactory`:

```py
# power.py

class CumulativePowerFactory:
    def __init__(self, exponent=2, *, start=0):
        self._exponent = exponent
        self.total = start

    def __call__(self, base):
        power = base ** self._exponent
        self.total += power
        return power
```

`CumulativePowerFactory`的初始化器有两个可选参数，`exponent`和`start`。第一个参数包含您将用来计算一系列幂的指数。默认为`2`，这是计算能力时常用的值。

`exponent`后的星号或星号(`*`)表示`start`是一个[关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-only-arguments)。要将值传递给仅包含关键字的参数，需要显式使用参数的名称。换句话说，要将`arg`设置为`value`，需要显式键入`arg=value`。

`start`参数保存初始值以计算幂的累积和。它默认为`0`，这是在没有预先计算值来初始化累计功率和的情况下的合适值。

特殊方法 [`.__call__()`](https://docs.python.org/3/reference/datamodel.html#object.__call__) 将`CumulativePowerFactory`的实例转化为**可调用的**对象。换句话说，你可以像调用任何常规函数一样调用`CumulativePowerFactory`的实例。

在`.__call__()`里面，你首先计算`base`的幂提升到`exponent`。然后将结果值加到`.total`的当前值上。最后，你[返回](https://realpython.com/python-return-statement/)计算出的功率。

为了给`CumulativePowerFactory`一个尝试，在包含`power.py`的目录中打开一个 Python [交互会话](https://realpython.com/interacting-with-python/)并运行下面的代码:

>>>

```py
>>> from power import CumulativePowerFactory

>>> square = CumulativePowerFactory()
>>> square(21)
441
>>> square(42)
1764
>>> square.total
2205

>>> cube = CumulativePowerFactory(exponent=3)
>>> cube(21)
9261
>>> cube(42)
74088
>>> cube.total
83349

>>> initialized_cube = CumulativePowerFactory(3, start=2205)
>>> initialized_cube(21)
9261
>>> initialized_cube(42)
74088
>>> initialized_cube.total
85554
```

这些例子展示了`CumulativePowerFactory`如何模拟多个构造函数。例如，第一个构造函数没有参数。它允许您创建计算`2`次方的类实例，这是`exponent`参数的默认值。`.total`实例属性保存你所计算的幂的累积和。

第二个例子展示了一个构造函数，它将`exponent`作为一个参数，并返回一个计算多维数据集的可调用实例。在这种情况下，`.total`的工作方式与第一个例子相同。

第三个例子展示了`CumulativePowerFactory`似乎有另一个构造函数，允许您通过提供`exponent`和`start`参数来创建实例。现在`.total`以`2205`的值开始，它初始化幂的和。

在类中实现`.__init__()`时使用可选参数是创建模拟多个构造函数的类的一种简洁而巧妙的技术。

### 检查`.__init__()`中的参数类型

模拟多个构造函数的另一种方法是编写一个`.__init__()`方法，它根据参数类型表现不同。在 Python 中检查一个[变量](https://realpython.com/python-variables/)的类型，一般依靠内置的 [`isinstance()`](https://docs.python.org/dev/library/functions.html#isinstance) 函数。如果一个对象是一个给定类的实例，这个函数返回`True`，否则返回`False`:

>>>

```py
>>> isinstance(42, int)
True

>>> isinstance(42, float)
False

>>> isinstance(42, (list, int))
True

>>> isinstance(42, list | int)  # Python >= 3.10
True
```

`isinstance()`的第一个参数是您想要类型检查的对象。第二个参数是引用的类或数据类型。您还可以将一个类型的[元组](https://realpython.com/python-lists-tuples/)传递给该参数。如果您运行的是 [Python 3.10](https://realpython.com/python310-new-features/) 或更高版本，那么您也可以使用新的[联合语法](https://www.python.org/dev/peps/pep-0604/)和管道符号(`|`)。

现在假设您想继续处理您的`Person`类，并且您需要该类也接受这个人的出生日期。您的代码将出生日期表示为一个 [`date`](https://docs.python.org/3/library/datetime.html#date-objects) 对象，但是为了方便起见，您的用户也可以选择以给定格式的字符串形式提供出生日期。在这种情况下，您可以执行如下操作:

>>>

```py
>>> from datetime import date

>>> class Person:
...     def __init__(self, name, birth_date):
...         self.name = name
...         if isinstance(birth_date, date):
...             self.birth_date = birth_date
...         elif isinstance(birth_date, str):
...             self.birth_date = date.fromisoformat(birth_date)
...

>>> jane = Person("Jane Doe", "2000-11-29")
>>> jane.birth_date
datetime.date(2000, 11, 29)

>>> john = Person("John Doe", date(1998, 5, 15))
>>> john.birth_date
datetime.date(1998, 5, 15)
```

在`.__init__()`中，首先定义通常的`.name`属性。[条件语句](https://realpython.com/python-conditional-statements/)的`if`子句检查所提供的出生日期是否是一个`date`对象。如果是这样，那么您定义`.birth_date`来存储手头的数据。

`elif`子句检查`birth_date`参数是否属于`str`类型。如果是这样，那么将`.birth_date`设置为从提供的字符串构建的`date`对象。注意，`birth_date`参数应该是一个带有日期的字符串，格式为 [ISO](https://en.wikipedia.org/wiki/ISO_8601) ， *YYYY-MM-DD* 。

就是这样！现在您有了一个`.__init__()`方法，它模拟了一个具有多个构造函数的类。一个构造函数接受`date`类型的参数。另一个构造函数接受字符串类型的参数。

**注意:**如果您运行的是 [Python 3.10](https://realpython.com/python310-new-features/) 或更高版本，那么您也可以使用[结构化模式匹配语法](https://realpython.com/python310-new-features/#structural-pattern-matching)来实现本节中的技术。

上面例子中的技术有一个缺点，就是扩展性不好。如果您有多个参数可以接受不同数据类型的值，那么您的实现很快就会变成一场噩梦。因此，这种技术被认为是 Python 中的反模式。

**注意:** [PEP 443](https://www.python.org/dev/peps/pep-0443/) 声明“……目前 Python 代码的一种常见反模式是检查接收到的参数的类型，以便决定如何处理对象。”根据同一份文件，这种编码模式是“脆弱的，不可扩展的”。

PEP 443 因此引入了[单分派](https://docs.python.org/3/glossary.html#term-single-dispatch) [通用函数](https://docs.python.org/3/glossary.html#term-generic-function)来帮助你尽可能避免使用这种编码反模式。您将在章节[中了解更多关于这个特性的内容，该章节提供了带有`@singledispatchmethod`](#providing-multiple-constructors-with-singledispatchmethod) 的多个类构造器。

例如，如果用户为`birth_date`输入一个 [Unix 时间](https://en.wikipedia.org/wiki/Unix_time)值，会发生什么？查看以下代码片段:

>>>

```py
>>> linda = Person("Linda Smith", 1011222000)

>>> linda.birth_date
Traceback (most recent call last):
    ...
AttributeError: 'Person' object has no attribute 'birth_date'
```

当您访问`.birth_date`时，您会得到一个 [`AttributeError`](https://realpython.com/python-traceback/#attributeerror) ，因为您的条件语句没有考虑不同日期格式的分支。

要解决这个问题，您可以继续添加`elif`子句来涵盖用户可以传递的所有可能的日期格式。您还可以添加一个`else`子句来捕捉**不支持的日期格式**:

>>>

```py
>>> from datetime import date

>>> class Person:
...     def __init__(self, name, birth_date):
...         self.name = name
...         if isinstance(birth_date, date):
...             self.birth_date = birth_date
...         elif isinstance(birth_date, str):
...             self.birth_date = date.fromisoformat(birth_date)
...         else:
...             raise ValueError(f"unsupported date format: {birth_date}")
...

>>> linda = Person("Linda Smith", 1011222000)
Traceback (most recent call last):
    ...
ValueError: unsupported date format: 1011222000
```

在这个例子中，如果`birth_date`的值不是一个`date`对象或者包含有效 ISO 日期的字符串，那么`else`子句就会运行。这样，例外情况就不会悄无声息地通过。

[*Remove ads*](/account/join/)

## 用 Python 中的`@classmethod`提供多个构造函数

在 Python 中提供多个构造函数的一个强大技术是使用 [`@classmethod`](https://realpython.com/instance-class-and-static-methods-demystified/#class-methods) 。这个装饰器允许你把一个常规方法变成一个**类方法**。

与常规方法不同，类方法不将当前实例`self`作为参数。相反，它们接受类本身，通常作为`cls`参数传入。使用`cls`来命名这个参数是 Python 社区中一个流行的约定。

下面是定义类方法的基本语法:

>>>

```py
>>> class DemoClass:
...     @classmethod
...     def class_method(cls):
...         print(f"A class method from {cls.__name__}!")
...

>>> DemoClass.class_method()
A class method from DemoClass!

>>> demo = DemoClass()
>>> demo.class_method()
A class method from DemoClass!
```

`DemoClass`使用 Python 内置的`@classmethod`装饰器定义一个类方法。`.class_method()`的第一个论点持有类本身。通过这个参数，您可以从类内部访问该类。在这个例子中，您访问了`.__name__`属性，它将底层类的名称存储为一个[字符串](https://realpython.com/python-strings/)。

值得注意的是，您可以使用类或手边的类的具体实例来访问类方法。无论您如何调用`.class_method()`，它都会接收`DemoClass`作为它的第一个参数。您可以使用类方法作为构造函数的最终原因是，您不需要实例来调用类方法。

使用`@classmethod`可以向给定的类中添加任意多的显式构造函数。这是实现多个构造函数的一种流行的 Pythonic 方式。你也可以在 Python 中将这种类型的构造函数称为**替代构造函数**，正如[雷蒙德·赫廷格](https://twitter.com/raymondh)在他的 PyCon 演讲 [Python 的类开发工具包](https://www.youtube.com/watch?v=HTLu2DFOdTg&list=PLRVdut2KPAguz3xcd22i_o_onnmDKj3MA)中所做的那样。

现在，如何使用类方法来定制 Python 的实例化过程呢？您将控制两个步骤:对象创建和初始化，而不是微调`.__init__()`和对象初始化。通过下面的例子，你将学会如何做到这一点。

### 从直径构造一个圆

要用`@classmethod`创建你的第一个类构造器，假设你正在编写一个几何相关的应用程序，需要一个`Circle`类。最初，您按如下方式定义您的类:

```py
# circle.py

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

    def __repr__(self):
        return f"{self.__class__.__name__}(radius={self.radius})"
```

`Circle`的初始化器将一个半径值作为参数，并将其存储在一个名为`.radius`的实例属性中。然后这个类使用 Python 的 [`math`](https://realpython.com/python-math-module/) 模块实现计算圆的面积和周长的方法。特殊方法 [`.__repr__()`](https://docs.python.org/dev/reference/datamodel.html#object.__repr__) 为您的类返回一个合适的**字符串表示**。

继续在您的工作目录中创建`circle.py`文件。然后打开 Python 解释器，运行下面的代码来测试`Circle`:

>>>

```py
>>> from circle import Circle

>>> circle = Circle(42)
>>> circle
Circle(radius=42)

>>> circle.area()
5541.769440932395
>>> circle.perimeter()
263.89378290154264
```

酷！你的类工作正常！现在说你也想用直径实例化`Circle`。您可以做一些类似于`Circle(diameter / 2)`的事情，但这并不十分 Pythonic 化或直观。最好有一个替代的构造函数，直接使用它们的直径来创建圆。

继续将下面的类方法添加到`.__init__()`之后的`Circle`:

```py
# circle.py

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @classmethod
 def from_diameter(cls, diameter):        return cls(radius=diameter / 2)

    # ...
```

在这里，您将`.from_diameter()`定义为一个类方法。它的第一个参数接收对包含它的类`Circle`的引用。

第二个参数包含您想要创建的特定圆的直径。在该方法中，首先使用输入值`diameter`计算半径。然后通过调用半径为`diameter`参数的`cls`来实例化`Circle`。

这样，您可以完全控制使用直径作为参数来创建和初始化`Circle`的实例。

**注意:**在上面的例子中，你似乎可以通过调用`Circle`本身而不是`cls`来达到相同的结果。然而，如果你的类是子类，这可能会导致错误。当这些子类用`.from_diameter()`初始化时，它们将调用`Circle`而不是自己。

对`cls`参数的调用自动运行 Python 实例化一个类所需的对象创建和初始化步骤。最后，`.from_diameter()`将新实例返回给调用者。

**注意:**Python 社区中一个流行的惯例是使用`from`介词来命名作为类方法创建的构造函数。

以下是如何使用全新的构造函数通过直径来创建圆:

>>>

```py
>>> from circle import Circle

>>> Circle.from_diameter(84)
Circle(radius=42.0)

>>> circle.area()
5541.769440932395
>>> circle.perimeter()
263.89378290154264
```

对`Circle`上的`.from_diameter()`的调用返回该类的一个新实例。为了构造该实例，该方法使用直径而不是半径。注意，`Circle`的其余功能和以前一样。

像上面例子中那样使用`@classmethod`是在类中提供显式多个构造函数的最常见方式。使用这种技术，您可以选择为您提供的每个备选构造函数选择正确的名称，这可以使您的代码更具可读性和可维护性。

[*Remove ads*](/account/join/)

### 从笛卡尔坐标构建一个极点

对于一个使用类方法提供多个构造函数的更详细的例子，假设您有一个在数学相关的应用程序中表示一个[极坐标](https://en.wikipedia.org/wiki/Polar_coordinate_system)点的类。你需要一种方法使你的类更加灵活，这样你也可以使用[笛卡尔](https://en.wikipedia.org/wiki/Cartesian_coordinate_system)坐标构造新的实例。

下面是如何编写一个构造函数来满足这一要求:

```py
# point.py

import math

class PolarPoint:
    def __init__(self, distance, angle):
        self.distance = distance
        self.angle = angle

    @classmethod
    def from_cartesian(cls, x, y):
        distance = math.dist((0, 0), (x, y))
        angle = math.degrees(math.atan2(y, x))
        return cls(distance, angle)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(distance={self.distance:.1f}, angle={self.angle:.1f})"
        )
```

在这个例子中，`.from_cartesian()`接受两个参数，分别代表给定点的`x`和`y`笛卡尔坐标。然后该方法计算所需的`distance`和`angle`来构造相应的`PolarPoint`对象。最后，`.from_cartesian()`返回该类的一个新实例。

下面是该类使用两种坐标系的工作方式:

>>>

```py
>>> from point import PolarPoint

>>> # With polar coordinates
>>> PolarPoint(13, 22.6)
PolarPoint(distance=13.0, angle=22.6)

>>> # With cartesian coordinates
>>> PolarPoint.from_cartesian(x=12, y=5)
PolarPoint(distance=13.0, angle=22.6)
```

在这些例子中，您使用标准的实例化过程和您的替代构造函数`.from_cartesian()`，使用概念上不同的初始化参数来创建`PolarPoint`实例。

## 探索现有 Python 类中的多个构造函数

使用`@classmethod`装饰器在一个类中提供多个构造函数是 Python 中相当流行的技术。有几个内置和标准库类的例子使用这种技术来提供多个可选的构造函数。

在本节中，您将了解这些类中最著名的三个例子: [`dict`](https://realpython.com/python-dicts/) 、 [`datetime.date`](https://realpython.com/python-datetime/) 和 [`pathlib.Path`](https://realpython.com/python-pathlib/#creating-paths) 。

### 从关键字构建字典

字典是 Python 中的一种基本数据类型。它们存在于每一段 Python 代码中，无论是显式的还是隐式的。它们也是语言本身的基石，因为 [CPython](https://realpython.com/cpython-source-code-guide/) 实现的重要部分依赖于它们。

在 Python 中，有几种方法可以[定义字典](https://realpython.com/python-dicts/#defining-a-dictionary)实例。您可以使用字典文字，它由花括号(`{}`)中的键值对组成。例如，您也可以使用关键字参数或双项元组序列显式调用`dict()`。

这个流行的类还实现了一个名为 [`.fromkeys()`](https://docs.python.org/dev/library/stdtypes.html?highlight=strip#dict.fromkeys) 的替代构造函数。这个类方法有一个`iterable`键和一个可选的`value`。`value`参数默认为 [`None`](https://realpython.com/null-in-python/) ，并作为结果字典中所有键的值。

现在，`.fromkeys()`如何在您的代码中发挥作用？假设您正在经营一家动物收容所，您需要构建一个小应用程序来跟踪目前有多少动物生活在您的收容所中。你的应用程序使用字典来存储动物的库存。

因为您已经知道您能够在庇护所中安置哪些物种，所以您可以动态地创建初始库存字典，如下面的代码片段所示:

>>>

```py
>>> allowed_animals = ["dog", "cat", "python", "turtle"]

>>> animal_inventory = dict.fromkeys(allowed_animals, 0)

>>> animal_inventory
{'dog': 0, 'cat': 0, 'python': 0, 'turtle': 0}
```

在这个例子中，您使用`.fromkeys()`构建了一个初始字典，它从`allowed_animals`获取键。通过将这个值作为第二个参数提供给`.fromkeys()`，将每只动物的初始库存设置为`0`。

正如您已经了解到的，`value`默认为`None`，在某些情况下，这可能是您的字典的键的合适的初始值。然而，在上面的例子中，`0`是一个方便的值，因为您正在处理每个物种的个体数量。

**注意:**大多数情况下， [`collections`](https://realpython.com/python-collections-module/) 模块中的 [`Counter`](https://realpython.com/python-counter/) 类是一个更适合处理库存问题的工具，就像上面的例子一样。然而，`Counter`并没有提供一个合适的`.fromkeys()`实现来防止类似`Counter.fromkeys("mississippi", 0)`的歧义。

标准库中的其他[映射](https://docs.python.org/3/glossary.html#term-mapping)也有一个名为`.fromkeys()`的构造函数。[`OrderedDict`](https://realpython.com/python-ordereddict/)[`defaultdict`](https://realpython.com/python-defaultdict/)[`UserDict`](https://docs.python.org/3/library/collections.html?highlight=collections#collections.UserDict)就是这种情况。例如，`UserDict`的[源代码](https://github.com/python/cpython/blob/992565f7f72fd8250b788795f76eedcff5636a64/Lib/collections/__init__.py#L1170)提供了`.fromkeys()`的如下实现:

```py
@classmethod
def fromkeys(cls, iterable, value=None):
    d = cls()
    for key in iterable:
        d[key] = value
    return d
```

这里，`.fromkeys()`将一个`iterable`和一个`value`作为自变量。该方法通过调用`cls`创建一个新的字典。然后它遍历`iterable`中的键并将每个值设置为`value`，默认为`None`，和往常一样。最后，该方法返回新创建的字典。

[*Remove ads*](/account/join/)

### 创建`datetime.date`个对象

标准库中的`datetime.date`类是另一个利用多个构造函数的类。这个类提供了几个可选的构造函数，比如 [`.today()`](https://docs.python.org/dev/library/datetime.html#datetime.date.today) 、 [`.fromtimestamp()`](https://docs.python.org/dev/library/datetime.html#datetime.date.fromtimestamp) 、 [`.fromordinal()`](https://docs.python.org/dev/library/datetime.html#datetime.date.fromordinal) 、。它们都允许您使用概念上不同的参数来构造`datetime.date`对象。

下面是一些如何使用这些构造函数来创建`datetime.date`对象的例子:

>>>

```py
>>> from datetime import date
>>> from time import time

>>> # Standard constructor
>>> date(2022, 1, 13)
datetime.date(2022, 1, 13)

>>> date.today()
datetime.date(2022, 1, 13)

>>> date.fromtimestamp(1642110000)
datetime.date(2022, 1, 13)
>>> date.fromtimestamp(time())
datetime.date(2022, 1, 13)

>>> date.fromordinal(738168)
datetime.date(2022, 1, 13)

>>> date.fromisoformat("2022-01-13")
datetime.date(2022, 1, 13)
```

第一个示例使用标准类构造函数作为引用。第二个例子展示了如何使用`.today()`从当天的日期构建一个`date`对象。

其余的例子展示了`datetime.date`如何使用几个类方法来提供多个构造函数。这种构造函数的多样性使得实例化过程非常灵活和强大，涵盖了广泛的用例。它还通过描述性的方法名称提高了代码的可读性。

### 寻找回家的路

Python 的标准库中的`pathlib`模块提供了方便和现代的工具，用于优雅地处理代码中的系统路径。如果你从未使用过这个模块，那么看看 [Python 3 的 pathlib 模块:驯服文件系统](https://realpython.com/python-pathlib/)。

`pathlib`中最方便的工具是它的`Path`类。这个类允许你以跨平台的方式处理你的系统路径。`Path`是另一个提供多个构造函数的标准类库。例如，您会发现`Path.home()`，它从您的主目录创建一个路径对象:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

***>>>

```py
>>> from pathlib import Path

>>> Path.home()
WindowsPath('C:/Users/username')
```

>>>

```py
>>> from pathlib import Path

>>> Path.home()
PosixPath('/home/username')
```

`.home()`构造函数返回一个代表用户主目录的新路径对象。当您在 Python 应用程序和项目中处理[配置文件](https://en.wikipedia.org/wiki/Configuration_file)时，这个可选的构造函数会很有用。

最后，`Path`还提供了一个名为 [`.cwd()`](https://docs.python.org/dev/library/pathlib.html?highlight=pathlib#pathlib.Path.cwd) 的构造函数。该方法从当前工作目录创建一个 path 对象。来吧，试一试！

## 为多个构造函数提供`@singledispatchmethod`

您将学习的最后一项技术被称为[单分派](https://docs.python.org/3/glossary.html#term-single-dispatch) [通用函数](https://docs.python.org/3/glossary.html#term-generic-function)。使用这种技术，您可以向类中添加多个构造函数，并根据它们的第一个参数的类型有选择地运行它们。

单分派通用函数由对不同数据类型实现相同操作的多个函数组成。底层的**分派算法**根据单个参数的类型决定运行哪个实现。这就是**一词**的由来。

从 Python 3.8 开始，您可以使用 [`@singledispatch`](https://docs.python.org/dev/library/functools.html#functools.singledispatch) 或 [`@singledispatchmethod`](https://docs.python.org/3/library/functools.html#functools.singledispatchmethod) 装饰器分别将一个函数或一个方法转化为一个单分派泛型函数。 [PEP 443](https://www.python.org/dev/peps/pep-0443/) 说明你可以在 [`functools`](https://docs.python.org/3/library/functools.html#module-functools) 模块中找到这些装饰者。

在常规函数中，Python 根据函数第一个参数的类型选择要调度的实现。在方法中，目标参数是紧跟在`self`之后的第一个参数。

### 单一分派方法的演示示例

要将单分派方法技术应用于给定的类，您需要定义一个基本方法实现并用`@singledispatchmethod`修饰它。然后，您可以编写替代实现，并使用基方法的名称加上`.register`来修饰它们。

下面的示例展示了基本语法:

```py
# demo.py

from functools import singledispatchmethod

class DemoClass:
    @singledispatchmethod
    def generic_method(self, arg):
        print(f"Do something with argument of type: {type(arg).__name__}")

    @generic_method.register
    def _(self, arg: int):
        print("Implementation for an int argument...")

    @generic_method.register(str)
    def _(self, arg):
        print("Implementation for a str argument...")
```

在`DemoClass`中，首先定义一个名为`generic_method()`的基础方法，并用`@singledispatchmethod`修饰它。然后定义两个可选的`generic_method()`实现，并用`@generic_method.register`修饰它们。

在本例中，您使用一个下划线(`_`)作为占位符名称来命名替代实现。在实际代码中，您应该使用描述性名称，前提是它们不同于基方法名称`generic_method()`。当使用描述性名称时，考虑添加一个前导下划线来将替代方法标记为**非公共**，并防止最终用户直接调用。

你可以使用[类型注释](https://realpython.com/python-type-checking/#annotations)来定义目标参数的类型。您还可以显式地将目标参数的类型传递给`.register()`装饰器。如果您需要定义一个方法来处理几个类型，那么您可以堆叠对`.register()`的多个调用，每个调用都有所需的类型。

您的类是这样工作的:

>>>

```py
>>> from demo import DemoClass

>>> demo = DemoClass()

>>> demo.generic_method(42)
Implementation for an int argument...

>>> demo.generic_method("Hello, World!")
Implementation for a str argument...

>>> demo.generic_method([1, 2, 3])
Do something with argument of type: list
```

如果使用一个整数作为参数调用`.generic_method()`，那么 Python 将运行对应于`int`类型的实现。同样，当您使用字符串调用方法时，Python 会调度字符串实现。最后，如果您使用未注册的数据类型(比如列表)调用`.generic_method()`，那么 Python 将运行该方法的基本实现。

您还可以使用这种技术来重载`.__init__()`，这将允许您为此方法提供多个实现，因此，您的类将有多个构造函数。

[*Remove ads*](/account/join/)

### 单一分派方法的真实示例

作为使用`@singledispatchmethod`的一个更现实的例子，假设您需要继续向您的`Person`类添加特性。这一次，您需要提供一种方法，根据一个人的出生日期来计算他的大概年龄。为了给`Person`添加这个特性，您可以使用一个助手类来处理与出生日期和年龄相关的所有信息。

继续在您的工作目录中创建一个名为`person.py`的文件。然后向其中添加以下代码:

```py
 1# person.py
 2
 3from datetime import date
 4from functools import singledispatchmethod
 5
 6class BirthInfo:
 7    @singledispatchmethod
 8    def __init__(self, birth_date):
 9        raise ValueError(f"unsupported date format: {birth_date}")
10
11    @__init__.register(date)
12    def _from_date(self, birth_date):
13        self.date = birth_date
14
15    @__init__.register(str)
16    def _from_isoformat(self, birth_date):
17        self.date = date.fromisoformat(birth_date)
18
19    @__init__.register(int)
20    @__init__.register(float)
21    def _from_timestamp(self, birth_date):
22        self.date = date.fromtimestamp(birth_date)
23
24    def age(self):
25        return date.today().year - self.date.year
```

下面是这段代码的工作原理:

*   **第 3 行**从`datetime`导入`date`,这样你以后可以将任何输入的日期转换成一个`date`对象。

*   **第 4 行**导入`@singledispatchmethod`来定义重载方法。

*   **第 6 行**将`BirthInfo`定义为一个普通的 Python 类。

*   **第 7 到 9 行**使用`@singledispatchmethod`将类初始化器定义为单分派泛型方法。这是该方法的基本实现，它为不支持的日期格式引发一个`ValueError`。

*   **第 11 到 13 行**注册了直接处理`date`对象的`.__init__()`的实现。

*   **第 15 到 17 行**定义了`.__init__()`的实现，它处理以 ISO 格式的字符串形式出现的日期。

*   **第 19 行到第 22 行**注册了一个实现，它处理从[纪元](https://docs.python.org/dev/library/time.html#epoch)开始以秒为单位的 Unix 时间日期。这一次，您通过将`.register`装饰器与 [`int`](https://realpython.com/python-numbers/#integers) 和 [`float`](https://realpython.com/python-numbers/#floating-point-numbers) 类型堆叠起来，注册了重载方法的两个实例。

*   **第 24 到 25 行**提供了一种计算给定人员年龄的常规方法。注意，`age()`的实现并不完全准确，因为它在计算年龄时没有考虑一年中的月和日。`age()`方法只是丰富示例的一个额外特性。

现在你可以在你的`Person`类中使用[组合](https://realpython.com/inheritance-composition-python/#whats-composition)来利用新的`BirthInfo`类。继续用下面的代码更新`Person`:

```py
# person.py
# ...

class Person:
    def __init__(self, name, birth_date):
        self.name = name
 self._birth_info = BirthInfo(birth_date) 
    @property
 def age(self):        return self._birth_info.age()

    @property
 def birth_date(self):        return self._birth_info.date
```

在这次更新中，`Person`有一个新的非公共属性叫做`._birth_info`，它是`BirthInfo`的一个实例。这个实例用输入参数`birth_date`初始化。`BirthInfo`的重载初始化器会根据用户的出生日期初始化`._birth_info`。

然后，您将`age()`定义为一个[属性](https://realpython.com/python-property/)，以提供一个计算属性，返回这个人当前的大概年龄。对`Person`的最后一个添加是`birth_date()`属性，它将这个人的出生日期作为一个`date`对象返回。

要试用您的`Person`和`BirthInfo`类，请打开一个交互式会话并运行以下代码:

>>>

```py
>>> from person import Person

>>> john = Person("John Doe", date(1998, 5, 15))
>>> john.age
24
>>> john.birth_date
datetime.date(1998, 5, 15)

>>> jane = Person("Jane Doe", "2000-11-29")
>>> jane.age
22
>>> jane.birth_date
datetime.date(2000, 11, 29)

>>> linda = Person("Linda Smith", 1011222000)
>>> linda.age
20
>>> linda.birth_date
datetime.date(2002, 1, 17)

>>> david = Person("David Smith", {"year": 2000, "month": 7, "day": 25})
Traceback (most recent call last):
    ...
ValueError: unsupported date format: {'year': 2000, 'month': 7, 'day': 25}
```

您可以使用不同的日期格式实例化`Person`。`BirthDate`的内部实例自动将输入的日期转换成 date 对象。如果用不支持的日期格式(比如字典)实例化`Person`，那么就会得到一个`ValueError`。

注意，`BirthDate.__init__()`负责为您处理输入的出生日期。不需要使用显式的替代构造函数来处理不同类型的输入。您可以使用标准构造函数实例化该类。

单分派方法技术的主要限制是它依赖于单个参数，即`self`之后的第一个参数。如果您需要使用多个参数来分派适当的实现，那么可以查看一些现有的第三方库，例如 [multipledispatch](https://pypi.org/project/multipledispatch/) 和 [multimethod](https://pypi.org/project/multimethod/) 。

## 结论

用**多个构造函数**编写 Python 类可以让你的代码更加通用灵活，覆盖广泛的用例。多个构造函数是一个强大的功能，它允许您根据需要使用不同类型的参数、不同数量的参数或两者来构建基础类的实例。

**在本教程中，您学习了如何:**

*   使用**可选参数**和**类型检查**模拟多个构造函数
*   使用内置的 **`@classmethod`** 装饰器编写多个构造函数
*   使用 **`@singledispatchmethod`** 装饰器重载你的类构造函数

您还了解了 Python 如何在内部**构造给定类的实例**，以及一些**标准库类**如何提供多个构造函数。

有了这些知识，现在您可以用多个构造函数来丰富您的类，用几种方法来处理 Python 中的实例化过程。

**免费奖励:** [点击此处获取免费的 Python OOP 备忘单](https://realpython.com/bonus/python-oop/)，它会为你指出最好的教程、视频和书籍，让你了解更多关于 Python 面向对象编程的知识。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深你的理解: [**在你的 Python 类中提供多个构造器**](/courses/multiple-constructors-python/)***********