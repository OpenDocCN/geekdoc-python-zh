# Python 类构造器:控制对象实例化

> 原文：<https://realpython.com/python-class-constructor/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。一起看书面教程加深理解: [**使用 Python 类构造函数**](/courses/using-python-class-constructors/)

类构造函数是 Python 中面向对象编程的基础部分。它们允许您创建并正确初始化给定类的对象，使这些对象随时可用。类构造函数在内部触发 Python 的实例化过程，该过程贯穿两个主要步骤:**实例创建**和**实例初始化**。

如果您想更深入地了解 Python 内部如何构造对象，并学习如何定制这个过程，那么本教程就是为您准备的。

**在本教程中，您将:**

*   了解 Python 内部的**实例化过程**
*   自定义对象初始化使用 **`.__init__()`**
*   通过覆盖 **`.__new__()`** 微调对象创建

有了这些知识，您将能够在自定义 Python 类中调整对象的创建和初始化，这将使您能够在更高级的层次上控制实例化过程。

为了更好地理解本教程中的例子和概念，你应该熟悉 Python 中的[面向对象编程](https://realpython.com/python3-object-oriented-programming/)和[特殊方法](https://docs.python.org/3/glossary.html#term-special-method)。

**免费奖励:** [点击此处获取免费的 Python OOP 备忘单](https://realpython.com/bonus/python-oop/)，它会为你指出最好的教程、视频和书籍，让你了解更多关于 Python 面向对象编程的知识。

## Python 的类构造器和实例化过程

像许多其他编程语言一样，Python 支持[面向对象编程](https://realpython.com/python3-object-oriented-programming/)。在 Python 面向对象功能的核心，您会发现 [`class`](https://realpython.com/python-keywords/#structure-keywords-def-class-with-as-pass-lambda) 关键字，它允许您定义自定义类，这些类可以具有用于存储数据的[属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)和用于提供行为的[方法](https://realpython.com/python3-object-oriented-programming/#instance-methods)。

一旦你有了一个要处理的类，那么你就可以开始创建这个类的新的**实例**或**对象**，这是一种在你的代码中重用功能的有效方法。

创建和初始化给定类的对象是面向对象编程的基本步骤。这一步通常被称为**对象构造**或**实例化**。负责运行这个实例化过程的工具通常被称为**类构造器**。

[*Remove ads*](/account/join/)

### 了解 Python 的类构造函数

在 Python 中，要构造一个给定类的对象，只需要调用带有适当参数的类，就像调用任何[函数](https://realpython.com/defining-your-own-python-function/)一样:

>>>

```py
>>> class SomeClass:
...     pass
...

>>> # Call the class to construct an object
>>> SomeClass()
<__main__.SomeClass object at 0x7fecf442a140>
```

在这个例子中，您使用关键字`class`定义了`SomeClass`。此类当前为空，因为它没有属性或方法。相反，类的主体只包含一个 [`pass`](https://realpython.com/python-pass/) 语句作为占位符语句，它什么也不做。

然后通过调用带有一对括号的类来创建一个新的`SomeClass`实例。在这个例子中，您不需要在调用中传递任何参数，因为您的类还不接受参数。

在 Python 中，当您像上面的例子那样调用一个类时，您调用的是类构造函数，它通过触发 Python 的内部实例化过程来创建、初始化并返回一个新的对象。

最后要注意的一点是，调用一个类不同于调用一个类的*实例*。这是两个不同且不相关的话题。要使一个类的实例可调用，需要实现一个 [`.__call__()`](https://docs.python.org/3/reference/datamodel.html#object.__call__) 的特殊方法，这个方法与 Python 的实例化过程无关。

### 了解 Python 的实例化过程

每当调用 Python 类来创建新实例时，就会触发 Python 的**实例化过程**。该过程通过两个独立的步骤运行，可以描述如下:

1.  **创建目标类的新实例**
2.  **用适当的初始[状态](https://en.wikipedia.org/wiki/State_(computer_science))初始化新实例**

运行第一步，Python 类有一个特殊的方法叫做 [`.__new__()`](https://docs.python.org/3/reference/datamodel.html#object.__new__) ，负责创建并返回一个新的空对象。然后另一个特殊的方法， [`.__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__) ，接受结果对象，以及类构造函数的参数。

`.__init__()`方法将新对象作为其第一个参数`self`。然后，它使用类构造函数传递给它的参数将任何必需的实例属性设置为有效状态。

简而言之，Python 的实例化过程从调用类构造函数开始，它触发**实例创建器**、`.__new__()`，创建一个新的空对象。这个过程继续使用**实例初始化器**、`.__init__()`，它接受构造函数的参数来初始化新创建的对象。

为了探究 Python 的实例化过程在内部是如何工作的，出于演示的目的，考虑下面这个实现了定制版本的`.__new__()`和`.__init__()`两种方法的`Point`类的例子:

```py
 1# point.py
 2
 3class Point:
 4    def __new__(cls, *args, **kwargs):
 5        print("1\. Create a new instance of Point.")
 6        return super().__new__(cls)
 7
 8    def __init__(self, x, y):
 9        print("2\. Initialize the new instance of Point.")
10        self.x = x
11        self.y = y
12
13    def __repr__(self) -> str:
14        return f"{type(self).__name__}(x={self.x}, y={self.y})"
```

下面是这段代码的详细内容:

*   **第 3 行**使用`class`关键字后跟类名来定义`Point`类。

*   **第 4 行**定义了`.__new__()`方法，该方法将类作为其第一个参数。注意，使用`cls`作为这个参数的名称是 Python 中的一个强约定，就像使用`self`来命名当前实例一样。该方法还采用了 [`*args`和`**kwargs`](https://realpython.com/python-kwargs-and-args/) ，允许向底层实例传递未定义数量的初始化参数。

*   **第 5 行** [在`.__new__()`运行对象创建步骤时打印一条消息给](https://realpython.com/python-print/)。

*   第 6 行用`cls`作为参数调用父类的`.__new__()`方法来创建一个新的`Point`实例。在这个例子中， [`object`](https://docs.python.org/3/library/functions.html#object) 是父类，调用 [`super()`](https://realpython.com/python-super/) 可以访问它。然后返回实例。这个实例将是`.__init__()`的第一个参数。

*   **第 8 行**定义了`.__init__()`，负责初始化步骤。该方法采用名为`self`的第一个参数，它保存了对当前实例的引用。该方法还需要两个额外的参数，`x`和`y`。这些参数保存实例属性`.x`和`.y`的初始值。您需要在对`Point()`的调用中为这些参数传递合适的值，稍后您将了解到这一点。

*   **第 9 行**在`.__init__()`运行对象初始化步骤时打印一条信息。

*   **10 线和 11 线**分别初始化`.x`和`.y`。为此，他们使用提供的输入参数`x`和`y`。

*   **第 13 行和第 14 行**实现了 [`.__repr__()`](https://docs.python.org/3/reference/datamodel.html#object.__repr__) 特殊方法，它为您的`Point`类提供了一个合适的字符串表示。

有了`Point`,您可以发现实例化过程在实践中是如何工作的。将你的代码保存到一个名为`point.py`的文件中，然后[在命令行窗口中启动你的 Python 解释器](https://realpython.com/interacting-with-python/#starting-the-interpreter)。然后运行以下代码:

>>>

```py
>>> from point import Point

>>> point = Point(21, 42)
1\. Create a new instance of Point.
2\. Initialize the new instance of Point.

>>> point
Point(x=21, y=42)
```

调用`Point()`类构造函数创建、初始化并返回该类的一个新实例。然后这个实例被分配给`point` [变量](https://realpython.com/python-variables/)。

在本例中，对构造函数的调用还让您了解 Python 内部运行来构造实例的步骤。首先，Python 调用了`.__new__()`，然后调用了`.__init__()`，产生了一个新的完全初始化的`Point`实例，正如您在示例末尾所确认的。

要继续学习 Python 中的类实例化，您可以尝试手动运行这两个步骤:

>>>

```py
>>> from point import Point

>>> point = Point.__new__(Point)
1\. Create a new instance of Point.

>>> # The point object is not initialized
>>> point.x
Traceback (most recent call last):
    ...
AttributeError: 'Point' object has no attribute 'x'
>>> point.y
Traceback (most recent call last):
    ...
AttributeError: 'Point' object has no attribute 'y'

>>> point.__init__(21, 42)
2\. Initialize the new instance of Point.

>>> # Now point is properly initialized
>>> point
Point(x=21, y=42)
```

在这个例子中，您首先在您的`Point`类上调用`.__new__()`，将类本身作为第一个参数传递给方法。这个调用只运行实例化过程的第一步，创建一个新的空对象。注意，以这种方式创建实例绕过了对`.__init__()`的调用。

**注意:**上面的代码片段旨在作为实例化过程如何在内部工作的示范示例。这不是你在真实代码中通常会做的事情。

一旦有了新的对象，就可以通过使用一组合适的参数调用`.__init__()`来初始化它。在这个调用之后，您的`Point`对象被正确地初始化，它的所有属性都被设置好了。

关于`.__new__()`需要注意的一个微妙而重要的细节是，它也可以返回一个不同于实现方法本身的类的实例。当这种情况发生时，Python 不会调用当前类中的`.__init__()`,因为没有办法明确知道如何初始化不同类的对象。

考虑下面的例子，其中`B`类的`.__new__()`方法返回了`A`类的一个实例:

```py
# ab_classes.py

class A:
    def __init__(self, a_value):
        print("Initialize the new instance of A.")
        self.a_value = a_value

class B:
    def __new__(cls, *args, **kwargs):
        return A(42)

    def __init__(self, b_value):
        print("Initialize the new instance of B.")
        self.b_value = b_value
```

因为`B.__new__()`返回不同类的实例，所以 Python 不运行`B.__init__()`。为了确认这种行为，将代码保存到一个名为`ab_classes.py`的文件中，然后在一个交互式 Python 会话中运行以下代码:

>>>

```py
>>> from ab_classes import B

>>> b = B(21)
Initialize the new instance of A.

>>> b.b_value
Traceback (most recent call last):
    ...
AttributeError: 'A' object has no attribute 'b_value'

>>> isinstance(b, B)
False
>>> isinstance(b, A)
True

>>> b.a_value
42
```

对`B()`类构造函数的调用运行`B.__new__()`，它返回一个`A`的实例，而不是`B`。这就是为什么`B.__init__()`从来不跑。注意`b`没有`.b_value`属性。相比之下，`b`确实有一个值为`42`的`.a_value`属性。

现在您已经知道了 Python 内部创建给定类的实例所采取的步骤，您可以更深入地研究一下`.__init__()`、`.__new__()`的其他特征，以及它们运行的步骤。

[*Remove ads*](/account/join/)

## 对象初始化用`.__init__()`

在 Python 中，`.__init__()`方法可能是您在自定义类中覆盖的最常见的特殊方法。几乎所有的类都需要一个定制的`.__init__()`实现。重写此方法将允许您正确初始化对象。

这个初始化步骤的目的是让您的新对象处于有效状态，以便您可以在代码中立即开始使用它们。在这一节中，您将学习编写自己的`.__init__()`方法的基础，以及它们如何帮助您定制您的类。

### 提供自定义对象初始化器

您可以编写的最基本的`.__init__()`实现只是负责将输入参数分配给匹配的实例属性。例如，假设您正在编写一个需要`.width`和`.height`属性的`Rectangle`类。在这种情况下，您可以这样做:

>>>

```py
>>> class Rectangle:
...     def __init__(self, width, height):
...         self.width = width
...         self.height = height
...

>>> rectangle = Rectangle(21, 42)
>>> rectangle.width
21
>>> rectangle.height
42
```

正如您之前所学的，`.__init__()`在 Python 中运行对象实例化过程的第二步。它的第一个参数`self`，保存了调用`.__new__()`得到的新实例。`.__init__()`的其余参数通常用于初始化实例属性。在上面的例子中，您使用`.__init__()`的`width`和`height`参数初始化了矩形的`.width`和`.height`。

重要的是要注意，不计算`self`，`.__init__()`的参数与您在调用类构造函数时传递的参数相同。所以，在某种程度上，`.__init__()` [签名](https://en.wikipedia.org/wiki/Type_signature)定义了类构造函数的签名。

此外，请记住`.__init__()`不能[显式返回](https://realpython.com/python-return-statement/#explicit-return-statements)与 [`None`](https://realpython.com/null-in-python/) 不同的任何内容，否则会得到一个 [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError) 异常:

>>>

```py
>>> class Rectangle:
...     def __init__(self, width, height):
...         self.width = width
...         self.height = height
...         return 42
...

>>> rectangle = Rectangle(21, 42)
Traceback (most recent call last):
    ...
TypeError: __init__() should return None, not 'int'
```

在这个例子中，`.__init__()`方法试图返回一个整数[数](https://realpython.com/python-numbers/)，最终在运行时引发一个`TypeError`异常。

上面例子中的错误消息说`.__init__()`应该返回`None`。但是，不需要显式返回`None`，因为没有显式`return`语句的方法和函数在 Python 中只是隐式返回 [`None`](https://realpython.com/null-in-python/) 。

使用上面的`.__init__()`实现，您可以确保当您使用适当的参数调用类构造函数时，`.width`和`.height`被初始化为有效状态。这样，您的矩形将在构建过程完成后立即可用。

在`.__init__()`中，您还可以对输入参数运行任何转换，以正确初始化实例属性。例如，如果您的用户将直接使用`Rectangle`，那么您可能希望验证提供的`width`和`height`，并在初始化相应的属性之前确保它们是正确的:

>>>

```py
>>> class Rectangle:
...     def __init__(self, width, height):
...         if not (isinstance(width, (int, float)) and width > 0):
...             raise ValueError(f"positive width expected, got {width}")
...         self.width = width
...         if not (isinstance(height, (int, float)) and height > 0):
...             raise ValueError(f"positive height expected, got {height}")
...         self.height = height
...

>>> rectangle = Rectangle(-21, 42)
Traceback (most recent call last):
    ...
ValueError: positive width expected, got -21
```

在这个更新的`.__init__()`实现中，在初始化相应的`.width`和`.height`属性之前，确保输入的`width`和`height`参数是正数。如果其中一个验证失败，那么您会得到一个`ValueError`。

**注意:**处理属性验证的一个更复杂的技术是将属性转化为**属性**。要了解关于属性的更多信息，请查看 [Python 的 property():向您的类添加托管属性](https://realpython.com/python-property/)。

现在假设您正在使用[继承](https://realpython.com/inheritance-composition-python/)来创建一个定制的类层次结构，并在代码中重用一些功能。如果你的子类提供了一个`.__init__()`方法，那么这个方法必须用适当的参数显式调用基类的`.__init__()`方法，以确保实例的正确初始化。为此，您应该使用内置的`super()`函数，如下例所示:

>>>

```py
>>> class Person:
...     def __init__(self, name, birth_date):
...         self.name = name
...         self.birth_date = birth_date
...

>>> class Employee(Person):
...     def __init__(self, name, birth_date, position):
...         super().__init__(name, birth_date) ...         self.position = position
...

>>> john = Employee("John Doe", "2001-02-07", "Python Developer")

>>> john.name
'John Doe'
>>> john.birth_date
'2001-02-07'
>>> john.position
'Python Developer'
```

`Employee`的`.__init__()`方法中的第一行用`name`和`birth_date`作为参数调用`super().__init__()`。这个调用确保了父类`Person`中`.name`和`.birth_date`的初始化。这种技术允许你用新的属性和功能扩展基类。

总结这一节，您应该知道`.__init__()`的基本实现来自内置的`object`类。当您没有在类中提供显式的`.__init__()`方法时，这个实现会被自动调用。

[*Remove ads*](/account/join/)

### 构建灵活的对象初始化器

通过调整`.__init__()`方法，可以使对象的初始化步骤灵活多样。为此，最流行的技术之一是使用[可选参数](https://realpython.com/python-optional-arguments/)。这种技术允许您编写这样的类，其中构造函数在实例化时接受不同的输入参数集。在给定的时间使用哪些参数将取决于您的特定需求和上下文。

举个简单的例子，看看下面的`Greeter`类:

```py
# greet.py

class Greeter:
    def __init__(self, name, formal=False):
        self.name = name
        self.formal = formal

    def greet(self):
        if self.formal:
            print(f"Good morning, {self.name}!")
        else:
            print(f"Hello, {self.name}!")
```

在这个例子中，`.__init__()`采用一个名为`name`的常规参数。它还需要一个名为`formal`的[可选参数](https://realpython.com/python-multiple-constructors/#using-optional-argument-values-in-__init__)，默认为`False`。因为`formal`有一个缺省值，你可以依靠这个值或者通过提供你自己的值来构造对象。

该类的最终行为将取决于`formal`的值。如果这个参数是`False`，那么当你打电话给`.greet()`时，你会得到一个非正式的问候。否则，你会得到更正式的问候。

要试用`Greeter`，请将代码保存到一个`greet.py`文件中。然后在工作目录中打开一个交互式会话，并运行以下代码:

>>>

```py
>>> from greet import Greeter

>>> informal_greeter = Greeter("Pythonista")
>>> informal_greeter.greet()
Hello, Pythonista!

>>> formal_greeter = Greeter("Pythonista", formal=True)
>>> formal_greeter.greet()
Good morning, Pythonista!
```

在第一个例子中，通过向参数`name`传递一个值并依赖默认值`formal`来创建一个`informal_greeter`对象。当你在`informal_greeter`对象上调用`.greet()`时，你会在屏幕上得到一个非正式的问候。

在第二个例子中，您使用一个`name`和一个`formal`参数来实例化`Greeter`。因为`formal`是`True`，所以叫`.greet()`的结果是正式的问候。

尽管这只是一个玩具示例，但它展示了默认参数值是如何成为一个强大的 Python 特性，可以用来为类编写灵活的初始化器。这些初始化器将允许你根据你的需要使用不同的参数集合来实例化你的类。

好吧！现在你已经知道了`.__init__()`和对象初始化步骤的基础，是时候换个方式开始深入`.__new__()`和对象创建步骤了。

## 用`.__new__()`创建对象

当编写 Python 类时，通常不需要提供自己的`.__new__()`特殊方法的实现。大多数时候，内置`object`类的基本实现足以构建当前类的空对象。

然而，这种方法有一些有趣的用例。例如，可以使用`.__new__()`创建[不可变](https://docs.python.org/3/glossary.html#term-immutable)类型的子类，如 [`int`](https://realpython.com/python-numbers/#integers) 、 [`float`](https://realpython.com/python-numbers/#floating-point-numbers) 、 [`tuple`](https://realpython.com/python-lists-tuples/) 、 [`str`](https://realpython.com/python-strings/) 。

在接下来的小节中，您将学习如何在您的类中编写自定义的`.__new__()`实现。为此，您将编写几个示例，让您知道何时需要重写该方法。

### 提供自定义对象创建者

通常，只有当您需要在较低的级别控制新实例的创建时，您才需要编写一个定制的`.__new__()`实现。现在，如果您需要这个方法的自定义实现，那么您应该遵循几个步骤:

1.  **通过使用适当的参数调用`super().__new__()`来创建一个新的实例**。
2.  **根据您的具体需求定制新实例**。
3.  **返回新实例**继续实例化过程。

通过这三个简洁的步骤，您将能够在 Python 实例化过程中定制实例创建步骤。以下是如何将这些步骤转化为 Python 代码的示例:

```py
class SomeClass:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        # Customize your instance here...
        return instance
```

这个例子提供了一种`.__new__()`的模板实现。像往常一样，`.__new__()`将当前类作为一个参数，这个参数通常被称为`cls`。

请注意，您正在使用 [`*args`和`**kwargs`](https://realpython.com/python-kwargs-and-args/) 通过接受任意数量的参数来使方法更加灵活和可维护。你应该总是用`*args`和`**kwargs`来定义`.__new__()`，除非你有很好的理由遵循不同的模式。

在第一行`.__new__()`中，您调用父类的`.__new__()`方法来创建一个新实例并为其分配内存。要访问父类的`.__new__()`方法，可以使用`super()`函数。这一连串的调用将您带到了`object.__new__()`，它是所有 Python 类的`.__new__()`的基本实现。

**注意:**内置的`object`类是所有 Python 类的默认基类。

下一步是定制新创建的实例。您可以做任何需要做的事情来定制手头的实例。最后，在第三步中，您需要返回新的实例，以继续初始化步骤的实例化过程。

需要注意的是，`object.__new__()`本身只接受一个*单个*参数，即要实例化的类。如果你用更多的参数调用`object.__new__()`，那么你会得到一个`TypeError`:

>>>

```py
>>> class SomeClass:
...     def __new__(cls, *args, **kwargs):
...         return super().__new__(cls, *args, **kwargs)
...     def __init__(self, value):
...         self.value = value
...

>>> SomeClass(42)
Traceback (most recent call last):
    ...
TypeError: object.__new__() takes exactly one argument (the type to instantiate)
```

在这个例子中，您将`*args`和`**kwargs`作为对`super().__new__()`的调用中的附加参数。底层的`object.__new__()`只接受类作为参数，所以当你实例化类时，你得到一个`TypeError`。

然而，如果您的类没有覆盖`.__new__()`,`object.__new__()`仍然接受并传递额外的参数给`.__init__()`，如下面的`SomeClass`的变体所示:

>>>

```py
>>> class SomeClass:
...     def __init__(self, value):
...         self.value = value
...

>>> some_obj = SomeClass(42)
>>> some_obj
<__main__.SomeClass object at 0x7f67db8d0ac0>
>>> some_obj.value
42
```

在`SomeClass`的这个实现中，您没有覆盖`.__new__()`。然后对象创建被委托给`object.__new__()`，它现在接受`value`并将其传递给`SomeClass.__init__()`来完成实例化。现在您可以创建新的完全初始化的`SomeClass`实例，就像示例中的`some_obj`一样。

酷！现在您已经了解了编写自己的`.__new__()`实现的基础，您已经准备好深入一些实际的例子，这些例子展示了 Python 编程中这种方法的一些最常见的用例。

[*Remove ads*](/account/join/)

### 不可变内置类型的子类化

首先，您将从`.__new__()`的一个用例开始，它由不可变内置类型的子类化组成。举个例子，假设你需要写一个`Distance`类作为 Python 的`float`类型的子类。您的类将有一个附加属性来存储用于测量距离的单位。

这里是解决这个问题的第一种方法，使用`.__init__()`方法:

>>>

```py
>>> class Distance(float):
...     def __init__(self, value, unit):
...         super().__init__(value)
...         self.unit = unit
...

>>> in_miles = Distance(42.0, "Miles")
Traceback (most recent call last):
    ...
TypeError: float expected at most 1 argument, got 2
```

当你继承一个不可变的内置数据类型时，你会得到一个错误。问题的一部分在于，该值是在创建时设置的，在初始化时更改它已经太晚了。另外，`float.__new__()`是在幕后调用的，它不像`object.__new__()`那样处理额外的参数。这就是在您的示例中引起错误的原因。

要解决这个问题，您可以在创建时用`.__new__()`初始化对象，而不是覆盖`.__init__()`。下面是你在实践中如何做到这一点:

>>>

```py
>>> class Distance(float):
...     def __new__(cls, value, unit):
...         instance = super().__new__(cls, value)
...         instance.unit = unit
...         return instance
...

>>> in_miles = Distance(42.0, "Miles")
>>> in_miles
42.0
>>> in_miles.unit
'Miles'
>>> in_miles + 42.0
84.0

>>> dir(in_miles)
['__abs__', '__add__', ..., 'real', 'unit']
```

在本例中，`.__new__()`运行您在上一节中学到的三个步骤。首先，该方法通过调用`super().__new__()`来创建当前类`cls`的一个新实例。这一次，调用回滚到`float.__new__()`，它创建一个新的实例，并使用`value`作为参数初始化它。然后该方法通过添加一个`.unit`属性来定制新的实例。最后，新的实例被返回。

**注意:**上面例子中的`Distance`类没有提供一个合适的单位转换机制。这意味着类似于`Distance(10, "km") + Distance(20, "miles")`的东西在添加值之前不会尝试转换单位。如果你对转换单位感兴趣，那么在 [PyPI](https://pypi.org/project/Pint) 上查看[品脱](https://pint.readthedocs.io/)项目。

就是这样！现在您的`Distance`类按预期工作，允许您使用一个实例属性来存储测量距离的单位。与存储在给定的`Distance`实例中的浮点值不同，`.unit`属性是可变的，因此您可以随时更改它的值。最后，注意对 [`dir()`](https://realpython.com/python-scope-legb-rule/#dir) 函数的调用揭示了您的类是如何从`float`继承特性和方法的。

### 返回不同类的实例

返回一个不同类的对象是一个需求，这会增加对定制实现`.__new__()`的需求。但是，您应该小心，因为在这种情况下，Python 会完全跳过初始化步骤。因此，在代码中使用新创建的对象之前，您有责任将它置于有效状态。

看看下面的例子，其中的`Pet`类使用`.__new__()`返回随机选择的类的实例:

```py
# pets.py

from random import choice

class Pet:
    def __new__(cls):
        other = choice([Dog, Cat, Python])
        instance = super().__new__(other)
        print(f"I'm a {type(instance).__name__}!")
        return instance

    def __init__(self):
        print("Never runs!")

class Dog:
    def communicate(self):
        print("woof! woof!")

class Cat:
    def communicate(self):
        print("meow! meow!")

class Python:
    def communicate(self):
        print("hiss! hiss!")
```

在这个例子中，`Pet`提供了一个`.__new__()`方法，通过从现有类的列表中随机选择一个类来创建一个新的实例。

下面是你如何使用这个`Pet`类作为宠物对象的工厂:

>>>

```py
>>> from pets import Pet

>>> pet = Pet()
I'm a Dog!
>>> pet.communicate()
woof! woof!
>>> isinstance(pet, Pet)
False
>>> isinstance(pet, Dog)
True

>>> another_pet = Pet()
I'm a Python!
>>> another_pet.communicate()
hiss! hiss!
```

每次实例化`Pet`，都会从不同的类中获得一个随机对象。这个结果是可能的，因为对`.__new__()`可以返回的对象没有限制。以这种方式使用`.__new__()`将一个类转换成一个灵活而强大的对象工厂，而不局限于它自身的实例。

最后，注意`Pet`的`.__init__()`方法是如何从不运行的。这是因为`Pet.__new__()`总是返回不同类的对象，而不是`Pet`本身。

### 在你的类中只允许一个实例

有时您需要实现一个只允许创建单个实例的类。这种类型的类通常被称为[单例](https://en.wikipedia.org/wiki/Singleton_pattern)类。在这种情况下，`.__new__()`方法就派上了用场，因为它可以帮助您限制给定类可以拥有的实例数量。

**注意:**大多数有经验的 Python 开发者会说，你不需要在 Python 中实现单例[设计模式](https://en.wikipedia.org/wiki/Software_design_pattern)，除非你已经有了一个工作类，并且需要在其上添加模式的功能。

其他时候，您可以使用模块级的常量[来获得相同的单例功能，而不必编写相对复杂的类。](https://realpython.com/python-constants/)

这里有一个用一个`.__new__()`方法编码一个`Singleton`类的例子，这个方法允许一次只创建一个实例。为此，`.__new__()`检查在类属性上缓存的先前实例的存在:

>>>

```py
>>> class Singleton(object):
...     _instance = None
...     def __new__(cls, *args, **kwargs):
...         if cls._instance is None:
...             cls._instance = super().__new__(cls)
...         return cls._instance
...

>>> first = Singleton()
>>> second = Singleton()
>>> first is second
True
```

本例中的`Singleton`类有一个名为`._instance`的[类属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)，默认为`None`，并作为[缓存](https://en.wikipedia.org/wiki/Cache_(computing))工作。`.__new__()`方法通过测试条件`cls._instance is None`来检查先前的实例是否不存在。

**注意:**在上面的例子中，`Singleton`没有提供`.__init__()`的实现。如果你需要这样一个带有`.__init__()`方法的类，那么记住这个方法将在你每次调用`Singleton()`构造函数时运行。这种行为会导致奇怪的初始化效果和错误。

如果这个条件为真，那么`if`代码块创建一个`Singleton`的新实例，并将其存储到`cls._instance`。最后，该方法将新的或现有的实例返回给调用方。

然后实例化`Singleton`两次，试图构造两个不同的对象，`first`和`second`。如果您用 [`is`](https://realpython.com/python-is-identity-vs-equality/) 操作符来比较这些对象的身份，那么您会注意到这两个对象是同一个对象。名字`first`和`second`只是引用了同一个`Singleton`对象。

[*Remove ads*](/account/join/)

### 部分模拟`collections.namedtuple`

作为如何在代码中利用`.__new__()`的最后一个例子，您可以运用您的 Python 技能，编写一个部分模拟 [`collections.namedtuple()`](https://realpython.com/python-namedtuple/) 的工厂函数。`namedtuple()`函数允许您创建`tuple`的子类，具有访问元组中项目的命名字段的附加特性。

下面的代码实现了一个`named_tuple_factory()`函数，它通过覆盖一个名为`NamedTuple`的嵌套类的`.__new__()`方法来部分模拟这个功能:

```py
 1# named_tuple.py
 2
 3from operator import itemgetter
 4
 5def named_tuple_factory(type_name, *fields):
 6    num_fields = len(fields)
 7
 8    class NamedTuple(tuple):
 9        __slots__ = ()
10
11        def __new__(cls, *args):
12            if len(args) != num_fields:
13                raise TypeError(
14                    f"{type_name} expected exactly {num_fields} arguments,"
15                    f" got {len(args)}"
16                )
17            cls.__name__ = type_name
18            for index, field in enumerate(fields):
19                setattr(cls, field, property(itemgetter(index)))
20            return super().__new__(cls, args)
21
22        def __repr__(self):
23            return f"""{type_name}({", ".join(repr(arg) for arg in self)})"""
24
25    return NamedTuple
```

下面是这个工厂函数的逐行工作方式:

*   **线 3** 从 [`operators`](https://docs.python.org/3/library/operator.html#module-operator) 模块导入`itemgetter()`。此函数允许您使用项目在包含序列中的索引来检索项目。

*   **第 5 行**定义`named_tuple_factory()`。这个函数接受名为`type_name`的第一个参数，它将保存您想要创建的 tuple 子类的名称。`*fields`参数允许您将未定义数量的字段名作为[字符串](https://realpython.com/python-strings/)传递。

*   第 6 行定义了一个本地[变量](https://realpython.com/python-variables/)来保存用户提供的命名字段的数量。

*   **第 8 行**定义了一个名为`NamedTuple`的嵌套类，它继承自内置的`tuple`类。

*   **第 9 行**提供了一个 [`.__slots__`](https://docs.python.org/3/glossary.html#term-__slots__) 类属性。该属性定义了一个保存实例属性的元组。这个元组通过替代实例的字典 [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 来节省内存，否则它将扮演类似的角色。

*   **第 11 行**以`cls`作为第一个参数实现`.__new__()`。该实现还采用`*args`参数来接受未定义数量的字段值。

*   **第 12 行到第 16 行**定义了一个[条件](https://realpython.com/python-conditional-statements/)语句，该语句检查要存储在最终元组中的条目数量是否与命名字段的数量不同。如果是这种情况，那么 conditional 将引发一个带有错误消息的`TypeError`。

*   **第 17 行**将当前类的`.__name__`属性设置为`type_name`提供的值。

*   **第 18 行和第 19 行**定义了一个 [`for`循环](https://realpython.com/python-for-loop/)，该循环将每个命名字段转化为一个属性，该属性使用`itemgetter()`返回目标`index`处的项目。循环使用内置的 [`setattr()`](https://docs.python.org/3/library/functions.html#setattr) 函数来执行这个动作。注意，内置的 [`enumerate()`](https://realpython.com/python-enumerate/) 函数提供了合适的`index`值。

*   **第 20 行**照常通过调用`super().__new__()`返回当前类的一个新实例。

*   **第 22 行和第 23 行**为 tuple 子类定义了一个`.__repr__()`方法。

*   第 25 行返回新创建的`NamedTuple`类。

为了测试您的`named_tuple_factory()`，在包含`named_tuple.py`文件的目录中启动一个交互式会话，并运行以下代码:

>>>

```py
>>> from named_tuple import named_tuple_factory

>>> Point = named_tuple_factory("Point", "x", "y")

>>> point = Point(21, 42)
>>> point
Point(21, 42)
>>> point.x
21
>>> point.y
42
>>> point[0]
21
>>> point[1]
42

>>> point.x = 84
Traceback (most recent call last):
    ...
AttributeError: can't set attribute

>>> dir(point)
['__add__', '__class__', ..., 'count', 'index', 'x', 'y']
```

在这段代码中，您通过调用`named_tuple_factory()`创建了一个新的`Point`类。该调用中的第一个参数表示结果类对象将使用的名称。第二个和第三个参数是结果类中可用的命名字段。

然后，通过调用类构造函数为`.x`和`.y`字段创建一个`Point`对象。要访问每个命名字段的值，可以使用点符号。您还可以使用索引来检索值，因为您的类是 tuple 子类。

因为在 Python 中元组是不可变的数据类型，所以不能在的位置给点的坐标[赋值。如果你尝试这样做，你会得到一个`AttributeError`。](https://en.wikipedia.org/wiki/In-place_algorithm)

最后，用您的`point`实例作为参数调用`dir()`,会发现您的对象继承了 Python 中常规元组的所有属性和方法。

## 结论

现在您知道 Python 类构造函数如何允许您实例化类，因此您可以在代码中创建具体的、随时可用的对象。在 Python 中，类构造器在内部触发实例化或构造过程，这个过程经过**实例创建**和**实例初始化**。这些步骤由`.__new__()`和`.__init__()`特殊方法驱动。

通过学习 Python 的类构造函数、实例化过程以及`.__new__()`和`.__init__()`方法，您现在可以管理您的定制类如何构造新的实例。

**在本教程中，您学习了:**

*   Python 的**实例化过程**如何在内部工作
*   你自己的 **`.__init__()`** 方法如何帮助你定制对象初始化
*   如何覆盖 **`.__new__()`** 方法来创建自定义对象

现在，您已经准备好利用这些知识来微调您的类构造函数，并在使用 Python 进行面向对象编程的过程中完全控制实例的创建和初始化。

**免费奖励:** [点击此处获取免费的 Python OOP 备忘单](https://realpython.com/bonus/python-oop/)，它会为你指出最好的教程、视频和书籍，让你了解更多关于 Python 面向对象编程的知识。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。一起看书面教程加深理解: [**使用 Python 类构造函数**](/courses/using-python-class-constructors/)*******