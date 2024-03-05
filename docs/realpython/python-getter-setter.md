# Getters 和 Setters:在 Python 中管理属性

> 原文：<https://realpython.com/python-getter-setter/>

如果你来自像 Java 或 T2c++这样的语言，那么你可能习惯于为类中的每个属性编写 T4 getter 和 setter 方法。这些方法允许你访问和改变私有属性，同时保持**封装**。在 Python 中，通常会将属性作为公共 API 的一部分公开，并在需要具有函数行为的属性时使用**属性**。

尽管属性是 Pythonic 式的方法，但它们也有一些实际的缺点。因此，您会发现有些情况下 getters 和 setters 比属性更好。

**在本教程中，您将:**

*   在你的类中编写 **getter** 和 **setter** 方法
*   用**属性**替换 getter 和 setter 方法
*   探索其他工具来取代 Python 中的 getter 和 setter 方法
*   决定什么时候 **setter** 和 **getter** 方法可以成为作业的**正确工具**

为了充分利用本教程，您应该熟悉 Python [面向对象](https://realpython.com/python3-object-oriented-programming/)编程。如果你有 Python [属性](https://realpython.com/python-property/)和[描述符](https://realpython.com/python-descriptors/)的基础知识，那将是一个加分项。

**源代码:** [点击这里获取免费的源代码](https://realpython.com/bonus/python-getter-setter-code/)，它向您展示了如何以及何时使用 Python 中的 getters、setters 和 properties。

## 了解 Getter 和 Setter 方法

当你在面向对象编程(OOP)中定义一个类时，你可能会以一些实例和类[属性](https://realpython.com/python3-object-oriented-programming/#class-and-instance-attributes)结束。这些属性只是可以通过实例、类或两者来访问的[变量](https://realpython.com/python-variables/)。

属性保存对象的内部[状态](https://en.wikipedia.org/wiki/State_(computer_science))。在许多情况下，您需要访问和改变这个状态，这涉及到访问和改变属性。通常，至少有两种方法可以访问和改变属性。您可以:

1.  直接访问并变异属性
***   使用**方法**来访问和改变属性*

*如果您向您的用户公开一个类的属性，那么这些属性会自动成为该类的公共 [API](https://en.wikipedia.org/wiki/API) 的一部分。它们将是**公共属性**，这意味着你的用户将直接访问和改变他们代码中的属性。

如果您需要更改属性本身的内部实现，拥有一个属于类 API 的属性将会成为一个问题。这个问题的一个明显的例子是当你想把一个**存储的**属性变成一个**计算的**属性。存储属性将通过检索和存储数据来立即响应访问和突变操作，而计算属性将在这些操作之前运行计算。

常规属性的问题是它们不能有内部实现*,因为它们只是变量。因此，更改属性的内部实现需要将属性转换成方法，这可能会破坏用户的代码。为什么？因为如果他们希望代码继续工作，就必须在整个代码库中将属性访问和变异操作更改为方法调用。

为了处理这种问题，一些编程语言，如 Java 和 C++，要求你提供操作类属性的方法。这些方法通常被称为 **getter** 和 **setter** 方法。您还可以找到被称为[访问器](https://en.wikipedia.org/wiki/Accessor_method)和[赋值器](https://en.wikipedia.org/wiki/Mutator_method)的方法。

[*Remove ads*](/account/join/)

### 什么是 Getter 和 Setter 方法？

Getter 和 setter 方法在许多面向对象编程语言中非常流行。所以，很可能你已经听说过他们了。作为一个粗略的定义，你可以说 getters 和 setters 是:

*   **Getter:** 一个允许你*访问*一个给定类中的属性的方法
*   **Setter:** 一个方法，允许你*设置*或者*改变*一个类中属性的值

在 OOP 中，getter 和 setter 模式表明，只有当你确定没有人需要将行为附加到公共属性时，才应该使用公共属性。如果一个属性可能改变它的内部实现，那么你应该使用 getter 和 setter 方法。

实现 getter 和 setter 模式需要:

1.  使您的属性成为非公共的
2.  为每个属性编写 getter 和 setter 方法

例如，假设您需要编写一个具有文本和字体属性的`Label`类。如果您要使用 getter 和 setter 方法来管理这些属性，那么您应该编写如下代码所示的类:

```py
# label.py

class Label:
    def __init__(self, text, font):
        self._text = text
        self._font = font

    def get_text(self):
        return self._text

    def set_text(self, value):
        self._text = value

    def get_font(self):
        return self._font

    def set_font(self, value):
        self._font = value
```

在这个例子中，`Label`的构造函数有两个参数，`text`和`font`。这些参数分别存储在`._text`和`._font`非公共实例属性中。

然后为这两个属性定义 getter 和 setter 方法。通常，getter 方法返回目标属性的值，而 setter 方法获取一个新值并将其赋给底层属性。

**注意:** Python 没有[访问修饰符](https://en.wikipedia.org/wiki/Access_modifiers)的概念，比如`private`、`protected`和`public`，来限制对类中属性和方法的访问。在 Python 中，区别在于**公共**和**非公共**类成员。

如果你想表明一个给定的属性或方法是非公共的，那么你应该使用 Python [的约定](https://www.python.org/dev/peps/pep-0008/#method-names-and-instance-variables)，在名字前加一个下划线(`_`)。

注意，这只是一个约定。它不会阻止你和其他程序员使用**点符号**访问属性，就像在`obj._attr`中一样。然而，违反这个惯例是不好的。

您可以像下面的例子一样使用您的`Label`类:

>>>

```py
>>> from label import Label

>>> label = Label("Fruits", "JetBrains Mono NL")
>>> label.get_text()
'Fruits'

>>> label.set_text("Vegetables")

>>> label.get_text()
'Vegetables'

>>> label.get_font()
'JetBrains Mono NL'
```

对公共访问隐藏它的属性，而公开 getter 和 setter 方法。您可以在需要访问或变更类的属性时使用这些方法，正如您已经知道的，这些属性是非公共的，因此不是类 API 的一部分。

### Getter 和 Setter 方法从何而来？

为了理解 getter 和 setter 方法的来源，回到`Label`的例子，假设您想自动以大写字母存储标签的文本。不幸的是，您不能简单地将这种行为添加到像`.text`这样的常规属性中。您只能通过方法添加行为，但是将公共属性转换成方法会在您的 API 中引入一个**突破性的变化**。

那么，你能做什么？嗯，在 Python 中，你最有可能使用一个[属性](https://realpython.com/python-property/)，你很快就会知道。然而，像 [Java](https://realpython.com/oop-in-python-vs-java/) 和 [C++](https://en.wikipedia.org/wiki/C%2B%2B) 这样的编程语言不支持类似属性的构造，或者它们的属性不太像 Python 属性。

这就是为什么这些语言鼓励你*永远不要将你的属性作为你的公共 API*的一部分。相反，你必须*提供 getter 和 setter 方法*，这提供了一种快速的方法来改变你的属性的内部实现，而不改变你的公共 API。

[封装](https://en.wikipedia.org/wiki/Encapsulation_(computer_programming))是另一个与 getter 和 setter 方法起源相关的基本话题。本质上，这一原则指的是将数据与操作该数据的方法捆绑在一起。这样，访问和变异操作将只通过方法来完成。

该原则还与限制对对象属性的直接访问有关，这将防止暴露实现细节或违反状态不变性。

为了给`Label`提供 Java 或 C++中新需要的功能，必须从一开始就使用 getter 和 setter 方法。如何应用 getter 和 setter 模式来解决 Python 中的问题？

考虑以下版本的`Label`:

```py
# label.py

class Label:
    def __init__(self, text, font):
 self.set_text(text)        self.font = font

    def get_text(self):
        return self._text

    def set_text(self, value):
 self._text = value.upper()  # Attached behavior
```

在这个更新版本的`Label`中，您为标签的文本提供了 getter 和 setter 方法。保存文本的属性是**非公共属性**，因为它的名字前面有一个下划线`._text`。setter 方法执行输入转换，将文本转换成大写字母。

现在，您可以像下面的代码片段一样使用您的`Label`类:

>>>

```py
>>> from label import Label

>>> label = Label("Fruits", "JetBrains Mono NL")
>>> label.get_text()
'FRUITS'

>>> label.set_text("Vegetables")
>>> label.get_text()
'VEGETABLES'
```

酷！您已成功将所需行为添加到标签的文本属性中。现在你的 setter 方法有了一个真正的目标，而不仅仅是给 target 属性赋一个新值。它的目标是向`._text`属性添加额外的行为。

尽管 getter 和 setter 模式在其他编程语言中很常见，但在 Python 中却不是这样。

向类中添加 getter 和 setter 方法可以显著增加代码行数。Getters 和 setters 也遵循一种重复而枯燥的模式，需要额外的时间来完成。这种模式容易出错，而且很乏味。您还会发现，从所有这些额外代码中获得的即时功能通常是零。

所有这些听起来像是 Python 开发人员不想在他们的代码中做的事情。在 Python 中，您可能会编写类似于以下代码片段的`Label`类:

>>>

```py
>>> class Label:
...     def __init__(self, text, font):
...         self.text = text
...         self.font = font
...
```

这里，`.text,`和`.font`是公共属性，作为类的 API 的一部分公开。这意味着您的用户可以随时更改他们的价值:

>>>

```py
>>> label = Label("Fruits", "JetBrains Mono NL")
>>> label.text
'Fruits'

>>> # Later...
>>> label.text = "Vegetables"
>>> label.text
'Vegetables'
```

暴露像`.text`和`.font`这样的属性是 Python 中的常见做法。因此，您的用户将在他们的代码中直接访问和改变这种属性。

像上面的例子一样，公开属性是 Python 中的一种常见做法。在这些情况下，切换到 getters 和 setters 将会引入突破性的变化。那么，如何处理需要在属性中添加行为的情况呢？Pythonic 式的方法是用属性替换属性。

[*Remove ads*](/account/join/)

## 使用属性代替 Getters 和 setter:Python 方式

将行为附加到属性的 Pythonic 方式是将属性本身变成一个**属性**。属性将获取、设置、删除和记录基础数据的方法打包在一起。因此，属性是具有附加行为的特殊属性。

您可以像使用常规属性一样使用属性。当您访问属性时，会自动调用其附加的 getter 方法。同样，当您变更属性时，会调用它的 setter 方法。这种行为提供了将功能附加到属性的方法，而不会在代码的 API 中引入重大更改。

作为属性如何帮助您将行为附加到属性的示例，假设您需要一个`Employee`类作为员工管理系统的一部分。您从以下基本实现开始:

```py
# employee.py

class Employee:
    def __init__(self, name, birth_date):
        self.name = name
        self.birth_date = birth_date

    # Implementation...
```

这个类的[构造函数](https://realpython.com/python-class-constructor/)有两个参数，手边雇员的姓名和出生日期。这些属性直接存储在两个实例属性中，`.name`和`.birth_date`。

您可以立即开始使用该类:

>>>

```py
>>> from employee import Employee

>>> john = Employee("John", "2001-02-07")

>>> john.name
'John'
>>> john.birth_date
'2001-02-07'

>>> john.name = "John Doe"
>>> john.name
'John Doe'
```

`Employee`允许您创建实例，以便直接访问相关的姓名和出生日期。注意，您也可以通过使用直接赋值来改变属性。

随着项目的发展，您会有新的需求。您需要用大写字母存储雇员的姓名，并将出生日期转换成一个 [`date`](https://realpython.com/python-datetime/) 对象。为了满足这些需求而不破坏您的 API，使用`.name`和`.birth_date`的 getter 和 setter 方法，您可以使用属性:

```py
# employee.py

from datetime import date

class Employee:
    def __init__(self, name, birth_date):
        self.name = name
        self.birth_date = birth_date

    @property
 def name(self):        return self._name

    @name.setter
 def name(self, value):        self._name = value.upper()

    @property
 def birth_date(self):        return self._birth_date

    @birth_date.setter
 def birth_date(self, value):        self._birth_date = date.fromisoformat(value)
```

在这个增强版的`Employee`中，使用`@property`装饰器将`.name`和`.birth_date`变成属性。现在每个属性都有一个 getter 和一个 setter 方法，以属性本身命名。注意，`.name`的 setter 把输入的名字变成了大写字母。同样，`.birth_date`的 setter 自动为你将输入的日期转换成一个`date`对象。

如前所述，属性的一个简洁的特性是，您可以将它们用作常规属性:

>>>

```py
>>> from employee import Employee

>>> john = Employee("John", "2001-02-07")

>>> john.name
'JOHN'

>>> john.birth_date
datetime.date(2001, 2, 7)

>>> john.name = "John Doe"
>>> john.name
'JOHN DOE'
```

酷！您已经向`.name`和`.birth_date`属性添加了行为，而没有影响您的类的 API。有了属性，您就能够像引用常规属性一样引用这些属性。在幕后，Python 负责为您运行适当的方法。

您必须避免通过在 API 中引入更改来破坏用户的代码。Python 的`@property` decorator 是实现这一点的 python 方式。在 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 中，属性被正式推荐为处理需要功能行为的属性的正确方法:

> 对于简单的公共数据属性，最好只公开属性名，不要使用复杂的访问器/赋值器方法。请记住，如果您发现一个简单的数据属性需要增加功能行为，Python 为未来的增强提供了一个简单的途径。在这种情况下，使用属性将功能实现隐藏在简单的数据属性访问语法后面。([来源](https://peps.python.org/pep-0008/#designing-for-inheritance))

Python 的属性有很多潜在的用例。例如，您可以使用属性以优雅和简单的方式创建[只读](https://realpython.com/python-property/#providing-read-only-attributes)、[读写](https://realpython.com/python-property/#creating-read-write-attributes)和[只写](https://realpython.com/python-property/#providing-write-only-attributes)属性。属性允许您删除和记录基础属性等等。更重要的是，属性允许您使常规属性的行为像带有附加行为的托管属性一样，而不改变您使用它们的方式。

由于属性的原因，Python 开发人员倾向于使用一些准则来设计他们的类的 API:

*   在适当的时候使用**公共属性**，即使您预期该属性在将来需要功能行为。
*   **避免**为你的属性定义 **setter** 和 **getter** 方法。如果需要，您可以随时将它们转换为属性。
*   当你需要**将行为**附加到属性上并在你的代码中将它们作为常规属性使用时，使用**属性**。
*   避免属性中的副作用，因为没有人会期望像赋值这样的操作会产生任何副作用。

Python 的属性很酷！正因为如此，人们倾向于过度使用它们。通常，只有在需要在特定属性之上添加额外处理时，才应该使用属性。把你所有的属性都变成属性会浪费你的时间。这也可能意味着性能和可维护性问题。

[*Remove ads*](/account/join/)

## 用更高级的工具替换 Getters 和 Setters】

到目前为止，您已经学习了如何创建基本的 getter 和 setter 方法来管理类的属性。您还了解了属性是解决向现有属性添加功能行为问题的 Pythonic 方法。

在接下来的几节中，您将了解到可以用来替换 Python 中 getter 和 setter 方法的其他工具和技术。

### Python 的描述符

[描述符](https://realpython.com/python-descriptors/)是 Python 的一个高级特性，允许你在类中创建带有附加行为的属性。要创建一个描述符，你需要使用**描述符协议**，尤其是`.__get__()`和`.__set__()`T6】的特殊方法。

描述符非常类似于属性。事实上，属性是一种特殊类型的描述符。然而，常规描述符比属性更强大，可以通过不同的类重用。

为了说明如何使用描述符创建具有功能行为的属性，假设您需要继续开发您的`Employee`类。这一次，您需要一个属性来存储雇员开始为公司工作的日期:

```py
# employee.py

from datetime import date

class Employee:
 def __init__(self, name, birth_date, start_date):        self.name = name
        self.birth_date = birth_date
 self.start_date = start_date 
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value.upper()

    @property
    def birth_date(self):
        return self._birth_date

    @birth_date.setter
    def birth_date(self, value):
        self._birth_date = date.fromisoformat(value)

 @property def start_date(self): return self._start_date 
 @start_date.setter def start_date(self, value): self._start_date = date.fromisoformat(value)
```

在本次更新中，您向`Employee`添加了另一个属性。这个新属性将允许您管理每个员工的开始日期。同样，setter 方法将日期从字符串转换成一个`date`对象。

这个类按预期工作。然而，它开始看起来重复和无聊。所以，你决定[重构](https://realpython.com/python-refactoring/)这个类。您注意到您在两个与日期相关的属性中执行相同的操作，并且您想到使用一个描述符来打包重复的功能:

```py
# employee.py

from datetime import date

class Date:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        instance.__dict__[self._name] = date.fromisoformat(value)

class Employee:
 birth_date = Date() start_date = Date() 
    def __init__(self, name, birth_date, start_date):
        self.name = name
        self.birth_date = birth_date
        self.start_date = start_date

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value.upper()
```

这段代码比之前的版本更简洁，重复性更少。在这次更新中，您将创建一个`Date`描述符来管理与日期相关的属性。描述符有一个自动存储属性名的`.__set_name__()`方法。它还有`.__get__()`和`.__set__()`方法，分别作为属性的 getter 和 setter。

本节中的`Employee`的两个实现工作方式类似。来吧，给他们一个尝试！

一般来说，如果您发现自己的类中有相似的属性定义，那么您应该考虑使用描述符。

### `.__setattr__()`和`.__getattr__()`方法

另一种替代 Python 中传统 getter 和 setter 方法的方法是使用 [`.__setattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__setattr__) 和 [`.__getattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattr__) 特殊方法来管理属性。考虑下面的例子，它定义了一个`Point`类。该类自动将输入坐标转换为浮点数:

```py
# point.py

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getattr__(self, name: str):
        return self.__dict__[f"_{name}"]

    def __setattr__(self, name, value):
        self.__dict__[f"_{name}"] = float(value)
```

`Point`的[初始化器](https://realpython.com/python-class-constructor/#object-initialization-with-__init__)取两个坐标，`x`和`y`。`.__getattr__()`方法返回由`name`表示的坐标。为此，该方法使用实例名称空间字典， [`.__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) 。请注意，属性的最终名称将在您在`name`中传递的任何内容之前有一个下划线。每当您使用点符号访问`Point`的属性时，Python 会自动调用`.__getattr__()`。

`.__setattr__()`方法添加或更新属性。在这个例子中，`.__setattr__()`对每个坐标进行操作，并使用内置的`float()`函数将其转换为浮点数。同样，每当您对包含类的任何属性运行赋值操作时，Python 都会调用`.__setattr__()`。

下面是这个类在实践中的工作方式:

>>>

```py
>>> from point import Point

>>> point = Point(21, 42)

>>> point.x
21.0
>>> point.y
42.0

>>> point.x = 84
>>> point.x
84.0

>>> dir(point)
['__class__', '__delattr__', ..., '_x', '_y']
```

您的`Point`类自动将坐标值转换成浮点数。您可以访问坐标、`x`和`y`，就像访问任何其他常规属性一样。然而，访问和变异操作分别通过`.__getattr__()`和`.__setattr__()`。

请注意，`Point`允许您将坐标作为公共属性来访问。但是，它将它们存储为非公共属性。您可以使用内置的`dir()`函数来确认这一行为。

本节中的例子有点奇特，您可能不会在代码中使用类似的东西。但是，您在示例中使用的工具允许您对属性访问和变异执行验证或转换，就像 getter 和 setter 方法一样。

在某种意义上，`.__getattr__()`和`.__setattr__()`是 getter 和 setter 模式的一种通用实现。在幕后，这些方法充当 getters 和 setters，支持 Python 中的常规属性访问和变异。

[*Remove ads*](/account/join/)

## 决定是否在 Python 中使用 Getters 和 Setters 或 Properties

在现实世界的编码中，您会发现 getter 和 setter 方法优于属性的一些用例，尽管属性通常是 Pythonic 的方式。

例如，getter 和 setter 方法可能更适合处理您需要:

*   **在属性访问或变异上运行代价高昂的转换**
*   带**额外参数**和**标志**
*   使用[继承](https://realpython.com/inheritance-composition-python/)
*   引发与属性访问和变异相关的[异常](https://realpython.com/python-exceptions/)
*   促进**异构**开发**团队**的整合

在接下来的部分中，您将深入这些用例，以及为什么 getter 和 setter 方法比属性更适合处理这些用例。

### 避免属性背后的缓慢方法

您应该避免将缓慢的操作隐藏在 Python 属性之后。API 的用户希望属性访问和变异像常规变量访问和变异一样执行。换句话说，用户将期望这些操作在*瞬间*发生，并且没有*副作用*。

离这个期望太远会让你的 API 使用起来奇怪和不愉快，违反了[最小惊奇原则](https://en.wikipedia.org/wiki/Principle_of_least_astonishment)。

此外，如果你的用户在一个[循环](https://realpython.com/python-for-loop/)中反复访问和改变你的属性，那么他们的代码会涉及太多的开销，这可能会产生巨大的*意想不到的*性能问题。

相比之下，传统的 getter 和 setter 方法使得*显式地*通过方法调用来访问或改变给定的属性。事实上，您的用户会意识到调用一个方法可能需要时间，并且他们的代码的性能可能会因此而有很大的差异。

在 API 中明确这些事实有助于减少用户在代码中访问和改变属性时的惊讶。

简而言之，如果你打算使用一个属性来管理一个属性，那么就要确保属性背后的方法是快速的，并且不会产生副作用。相比之下，如果您处理的是慢速访问器和赋值器方法，那么与属性相比，您更喜欢传统的 getters 和 setters 方法。

### 接受额外的参数和标志

与 Python 属性不同，传统的 getter 和 setter 方法允许更灵活的属性访问和变异。例如，假设您有一个带有`.birth_date`属性的`Person`类。这个属性在人的一生中应该是不变的。因此，您决定该属性将是只读的。

然而，由于人为错误的存在，您将面临有人在输入给定人员的出生日期时出错的情况。您可以通过提供一个带`force`标志的 setter 方法来解决这个问题，如下例所示:

```py
# person.py

class Person:
    def __init__(self, name, birth_date):
        self.name = name
        self._birth_date = birth_date

    def get_birth_date(self):
        return self._birth_date

    def set_birth_date(self, value, force=False):
        if force:
            self._birth_date = value
        else:
            raise AttributeError("can't set birth_date")
```

在这个例子中，您为`.birth_date`属性提供了传统的 getter 和 setter 方法。setter 方法带有一个名为`force`的额外参数，它允许您强制修改一个人的出生日期。

**注意:**传统的 setter 方法通常不会接受一个以上的参数。对于一些开发人员来说，上面的例子可能看起来很奇怪，甚至不正确。然而，它的目的是展示一种在某些情况下有用的技术。

这个类是这样工作的:

>>>

```py
>>> from person import Person

>>> jane = Person("Jane Doe", "2000-11-29")
>>> jane.name
'Jane Doe'

>>> jane.get_birth_date()
'2000-11-29'

>>> jane.set_birth_date("2000-10-29") Traceback (most recent call last):
    ...
AttributeError: can't set birth_date

>>> jane.set_birth_date("2000-10-29", force=True) >>> jane.get_birth_date()
'2000-10-29'
```

当您试图使用`.set_birth_date()`修改 Jane 的出生日期，而没有将`force`设置为`True`时，您会得到一个`AttributeError`，表示该属性无法设置。相反，如果您将`force`设置为`True`，那么您将能够更新 Jane 的出生日期，以纠正输入日期时出现的任何错误。

需要注意的是，Python 属性不接受 setter 方法中的额外参数。它们只是接受要设置或更新的值。

[*Remove ads*](/account/join/)

### 使用继承:Getter 和 setter vs . Properties

Python 属性的一个问题是它们在继承场景中表现不佳。例如，假设您需要扩展或修改子类中属性的 getter 方法。实际上，没有安全的方法可以做到这一点。你不能只覆盖 getter 方法，并期望属性的其余功能保持与父类中的相同。

出现此问题是因为 getter 和 setter 方法隐藏在属性内部。它们不是独立遗传的，而是作为一个整体。因此，当您重写从父类继承的属性的 getter 方法时，您重写了整个属性，包括它的 setter 方法和它的其余内部组件。

例如，考虑以下类层次结构:

```py
# person.py

class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

class Employee(Person):
    @property
    def name(self):
        return super().name.upper()
```

在这个例子中，您覆盖了`Employee`中`.name`属性的 getter 方法。这样，您就隐式地覆盖了整个`.name`属性，包括它的 setter 功能:

>>>

```py
>>> from person import Employee

>>> jane = Employee("Jane")

>>> jane.name
'JANE'

>>> jane.name = "Jane Doe"
Traceback (most recent call last):
    ...
AttributeError: can't set attribute 'name'
```

现在`.name`是一个只读属性，因为父类的 setter 方法没有被继承，而是被一个全新的属性覆盖。你不想那样，是吗？你如何解决这个继承问题？

如果您使用传统的 getter 和 setter 方法，那么这个问题就不会发生:

```py
# person.py

class Person:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def set_name(self, value):
        self._name = value

class Employee(Person):
    def get_name(self):
        return super().get_name().upper()
```

这个版本的`Person`提供了独立的 getter 和 setter 方法。`Employee`子类`Person`，覆盖 name 属性的 getter 方法。这个事实并不影响 setter 方法，该方法`Employee`成功地从其父类`Person`继承而来。

下面是这个新版本的`Employee`的工作原理:

>>>

```py
>>> from person import Employee

>>> jane = Employee("Jane")

>>> jane.get_name()
'JANE'

>>> jane.set_name("Jane Doe")
>>> jane.get_name()
'JANE DOE'
```

现在`Employee`已经完全可以使用了。被重写的 getter 方法按预期工作。setter 方法也可以工作，因为它是从`Person`成功继承的。

### 在属性访问或突变时引发异常

大多数时候，你不会想到像`obj.attribute = value`这样的赋值语句会引发异常。相比之下，您可以期望方法在响应错误时引发异常。在这方面，传统的 getter 和 setter 方法比属性更显式。

例如，`site.url = "123"`看起来不像是可以引发异常的东西。它看起来应该像一个常规的属性赋值。另一方面，`site.set_url("123")`看起来确实像是可以引发异常的东西，也许是一个`ValueError`，因为输入值不是一个网站的有效的 [URL](https://en.wikipedia.org/wiki/URL) 。在这个例子中，setter 方法更加明确。它清楚地表达了代码可能的行为。

根据经验，除非使用属性来提供只读属性，否则应避免在 Python 属性中引发异常。如果您需要在属性访问或变异时引发异常，那么您应该考虑使用 getter 和 setter 方法，而不是属性。

在这些情况下，使用 getters 和 setters 将减少用户的惊讶，并使您的代码更符合常见的实践和期望。

### 促进团队整合和项目迁移

在许多成熟的编程语言中，提供 getter 和 setter 方法是常见的做法。如果你和一个来自其他语言背景的开发团队一起开发一个 Python 项目，那么很可能 getter 和 setter 模式对他们来说比 Python 属性更熟悉。

在这种类型的异构团队中，使用 getters 和 setters 可以促进新开发人员融入团队。

使用 getter 和 setter 模式也可以提高 API 的一致性。它允许您提供基于方法调用的 API，而不是将方法调用与直接属性访问和变异相结合的 API。

通常，当 Python 项目增长时，您可能需要将项目从 Python 迁移到另一种语言。新语言可能没有属性，或者它们的行为可能不像 Python 属性那样。在这些情况下，从一开始就使用传统的 getters 和 setters 会使将来的迁移不那么痛苦。

在上述所有情况下，您应该考虑使用传统的 getter 和 setter 方法，而不是 Python 中的属性。

[*Remove ads*](/account/join/)

## 结论

现在你知道什么是 **getter** 和 **setter** 方法，以及它们来自哪里。这些方法允许访问和改变属性，同时避免 API 的改变。然而，由于属性的存在，它们在 Python 中并不流行。属性允许您向属性中添加行为，同时避免 API 中的破坏性更改。

尽管属性是替代传统的 getter 和 setter 的 Pythonic 方式，但是属性也有一些实际的缺点，可以用 getter 和 setter 来克服。

**在本教程中，您已经学会了如何:**

*   用 Python 写 **getter** 和 **setter** 方法
*   使用 Python **属性**替换 getter 和 setter 方法
*   使用 Python **工具**，比如描述符，来替换 getters 和 setters
*   决定什么时候 **setter** 和 **getter** 方法可以成为作业的**正确工具**

有了这些知识，您现在可以决定何时在 Python 类中使用 getter 和 setter 方法或属性。

**源代码:** [点击这里获取免费的源代码](https://realpython.com/bonus/python-getter-setter-code/)，它向您展示了如何以及何时使用 Python 中的 getters、setters 和 properties。*********