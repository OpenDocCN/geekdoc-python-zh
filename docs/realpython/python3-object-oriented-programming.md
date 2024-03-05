# Python 3 中的面向对象编程(OOP)

> 原文：<https://realpython.com/python3-object-oriented-programming/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 中的面向对象编程(OOP)介绍**](/courses/intro-object-oriented-programming-oop-python/)

**面向对象编程** (OOP)是一种通过将相关属性和行为捆绑到单独的**对象**中来构建程序的方法。在本教程中，您将学习 Python 中面向对象编程的基础知识。

从概念上讲，对象就像系统的组件。把一个程序想象成某种工厂装配线。在装配线的每一步，一个系统组件处理一些材料，最终将原材料转化为成品。

对象包含数据(如装配线上每个步骤的原材料或预处理材料)和行为(如每个装配线组件执行的操作)。

**在本教程中，您将学习如何:**

*   创建一个**类**，这就像是创建一个对象的蓝图
*   使用类来**创建新对象**
*   具有**类继承**的模型系统

**注:**本教程改编自 [*Python 基础知识:Python 实用入门 3*](https://realpython.com/products/python-basics-book/) 中“面向对象编程(OOP)”一章。

这本书使用 Python 内置的 [IDLE](https://realpython.com/python-idle/) 编辑器来创建和编辑 Python 文件，并与 Python shell 进行交互，因此在整个教程中你会偶尔看到对 IDLE 的引用。但是，从您选择的编辑器和环境中运行示例代码应该没有问题。

**免费奖励:** [点击此处获取免费的 Python OOP 备忘单](https://realpython.com/bonus/python-oop/)，它会为你指出最好的教程、视频和书籍，让你了解更多关于 Python 面向对象编程的知识。

## Python 中的面向对象编程是什么？

面向对象编程是一种[编程范式](http://en.wikipedia.org/wiki/Programming_paradigm)，它提供了一种结构化程序的方法，从而将属性和行为捆绑到单独的**对象**中。

例如，一个对象可以代表一个具有**属性**如姓名、年龄和地址以及**行为**如走路、说话、呼吸和跑步的人。或者它可以代表一封[电子邮件](https://realpython.com/python-send-email/)，具有收件人列表、主题和正文等属性，以及添加附件和发送等行为。

换句话说，面向对象编程是一种对具体的、真实世界的事物建模的方法，如汽车，以及事物之间的关系，如公司和雇员、学生和教师等等。OOP 将现实世界中的实体建模为软件对象，这些对象有一些与之相关的数据，并且可以执行某些功能。

另一个常见的编程范例是**过程化编程**，它像菜谱一样构建程序，以函数和代码块的形式提供一组步骤，这些步骤按顺序流动以完成任务。

关键的一点是，在 Python 中，对象是面向对象编程的核心，不仅像在过程编程中一样表示数据，而且在程序的整体结构中也是如此。

[*Remove ads*](/account/join/)

## 在 Python 中定义一个类

原始的[数据结构](https://realpython.com/courses/python-data-types/)——像数字、[字符串](https://realpython.com/python-strings/)和列表——被设计用来表示简单的信息，比如一个苹果的价格、一首诗的名字或者你最喜欢的颜色。如果你想表现更复杂的东西呢？

例如，假设您想要跟踪某个组织中的员工。您需要存储每个员工的一些基本信息，比如他们的姓名、年龄、职位以及他们开始工作的年份。

一种方法是将每个雇员表示为一个[列表](https://realpython.com/python-lists-tuples/):

```py
kirk = ["James Kirk", 34, "Captain", 2265]
spock = ["Spock", 35, "Science Officer", 2254]
mccoy = ["Leonard McCoy", "Chief Medical Officer", 2266]
```

这种方法有许多问题。

首先，它会使较大的代码文件更难管理。如果在声明`kirk`列表的地方引用几行之外的`kirk[0]`，你会记得索引为`0`的元素是雇员的名字吗？

第二，如果不是每个雇员在列表中有相同数量的元素，它可能会引入错误。在上面的`mccoy`列表中，缺少年龄，所以`mccoy[1]`将返回`"Chief Medical Officer"`而不是麦考伊博士的年龄。

让这类代码更易于管理和维护的一个好方法是使用**类**。

### 类与实例

类用于创建用户定义的数据结构。类定义了名为**方法**的函数，这些方法标识了从类创建的对象可以对其数据执行的行为和动作。

在本教程中，您将创建一个`Dog`类来存储一些关于单只狗的特征和行为的信息。

一个类是应该如何定义的蓝图。它实际上不包含任何数据。`Dog`类指定名字和年龄是定义狗的必要条件，但它不包含任何特定狗的名字或年龄。

类是蓝图，而**实例**是从类构建的包含真实数据的对象。`Dog`类的实例不再是蓝图。这是一只真正的狗，它有一个名字，像四岁的迈尔斯。

换句话说，一个类就像一个表格或问卷。实例就像一个已经填写了信息的表单。就像许多人可以用他们自己的独特信息填写同一个表单一样，许多实例可以从单个类中创建。

### 如何定义一个类

所有的类定义都以关键字`class`开始，后面是类名和冒号。缩进到类定义下面的任何代码都被认为是类体的一部分。

下面是一个`Dog`类的例子:

```py
class Dog:
    pass
```

`Dog`类的主体由一条语句组成:关键字`pass`。`pass`通常被用作占位符，表示代码最终的去向。它允许您在 Python 不抛出错误的情况下运行这段代码。

**注意:** Python 类名按照惯例是用 CapitalizedWords 符号写的。例如，一个特定品种的狗(如杰克罗素梗)的类可以写成`JackRussellTerrier`。

`Dog`类现在不是很有趣，所以让我们通过定义所有`Dog`对象应该具有的一些属性来稍微修饰一下它。有许多属性可供我们选择，包括名字、年龄、毛色和品种。为了简单起见，我们只使用姓名和年龄。

所有`Dog`对象必须具有的属性在一个叫做`.__init__()`的方法中定义。每次创建一个新的`Dog`对象，`.__init__()`通过分配对象的属性值来设置对象的初始**状态**。也就是说，`.__init__()`初始化该类的每个新实例。

你可以给`.__init__()`任意数量的参数，但是第一个参数总是一个叫做`self`的[变量](https://realpython.com/python-variables/)。当一个新的类实例被创建时，该实例被自动传递给`.__init__()`中的`self`参数，这样就可以在对象上定义新的**属性**。

让我们用一个创建`.name`和`.age`属性的`.__init__()`方法来更新`Dog`类:

```py
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

注意，`.__init__()`方法的签名缩进了四个空格。该方法的主体缩进八个空格。这个缩进非常重要。它告诉 Python,`.__init__()`方法属于`Dog`类。

在`.__init__()`的主体中，有两条语句使用了`self`变量:

1.  **`self.name = name`** 创建一个名为`name`的属性，并赋予它`name`参数的值。
2.  **`self.age = age`** 创建一个名为`age`的属性，并赋予它`age`参数的值。

在`.__init__()`中创建的属性称为**实例属性**。实例属性的值特定于类的特定实例。所有的`Dog`对象都有名称和年龄，但是`name`和`age`属性的值会根据`Dog`实例的不同而不同。

另一方面，**类属性**是对所有类实例具有相同值的属性。您可以通过在`.__init__()`之外给[变量](https://realpython.com/python-variables/)赋值来定义一个类属性。

例如，下面的`Dog`类有一个名为`species`的类属性，其值为`"Canis familiaris"`:

```py
class Dog:
    # Class attribute
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age
```

类属性直接定义在类名的第一行下面，缩进四个空格。它们必须总是被赋予一个初始值。当创建类的实例时，会自动创建类属性并将其赋给初始值。

使用类属性为每个类实例定义应该具有相同值的属性。对于因实例而异的属性，请使用实例属性。

现在我们有了一个`Dog`类，让我们创建一些狗吧！

[*Remove ads*](/account/join/)

## 用 Python 实例化一个对象

打开 IDLE 的交互窗口，键入以下内容:

>>>

```py
>>> class Dog:
...     pass
```

这创建了一个没有属性或方法的新的`Dog`类。

从一个类创建一个新对象叫做**实例化**一个对象。您可以通过键入类名，后跟左括号和右括号来实例化一个新的`Dog`对象:

>>>

```py
>>> Dog()
<__main__.Dog object at 0x106702d30>
```

您现在在`0x106702d30`有了一个新的`Dog`对象。这个看起来很有趣的字母和数字串是一个**内存地址**，它指示了`Dog`对象在你的计算机内存中的存储位置。请注意，您在屏幕上看到的地址会有所不同。

现在实例化第二个`Dog`对象:

>>>

```py
>>> Dog()
<__main__.Dog object at 0x0004ccc90>
```

新的`Dog`实例位于不同的内存地址。这是因为它是一个全新的实例，与您实例化的第一个`Dog`对象完全不同。

要从另一个角度看这个问题，请键入以下内容:

>>>

```py
>>> a = Dog()
>>> b = Dog()
>>> a == b
False
```

在这段代码中，您创建了两个新的`Dog`对象，并将它们分配给变量`a`和`b`。当您使用`==`运算符比较`a`和`b`时，结果是`False`。尽管`a`和`b`都是`Dog`类的实例，但它们在内存中代表两个不同的对象。

### 类别和实例属性

现在创建一个新的`Dog`类，它有一个名为`.species`的类属性和两个名为`.name`和`.age`的实例属性:

>>>

```py
>>> class Dog:
...     species = "Canis familiaris"
...     def __init__(self, name, age):
...         self.name = name
...         self.age = age
```

要实例化这个`Dog`类的对象，您需要为`name`和`age`提供值。如果没有，Python 就会抛出一个`TypeError`:

>>>

```py
>>> Dog()
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    Dog()
TypeError: __init__() missing 2 required positional arguments: 'name' and 'age'
```

要将参数传递给`name`和`age`参数，请将值放入类名后的括号中:

>>>

```py
>>> buddy = Dog("Buddy", 9)
>>> miles = Dog("Miles", 4)
```

这创建了两个新的`Dog`实例——一个是九岁的狗 Buddy，另一个是四岁的狗 Miles。

`Dog`类的`.__init__()`方法有三个参数，那么为什么在示例中只有两个参数传递给它呢？

当实例化一个`Dog`对象时，Python 会创建一个新的实例，并将其传递给`.__init__()`的第一个参数。这实质上移除了`self`参数，因此您只需要担心`name`和`age`参数。

在您创建了`Dog`实例之后，您可以使用**点符号**来访问它们的实例属性:

>>>

```py
>>> buddy.name
'Buddy'
>>> buddy.age
9

>>> miles.name
'Miles'
>>> miles.age
4
```

您可以用同样的方式访问类属性:

>>>

```py
>>> buddy.species
'Canis familiaris'
```

使用类来组织数据的一个最大的优点是实例保证具有您期望的属性。所有的`Dog`实例都有`.species`、`.name`和`.age`属性，所以您可以放心地使用这些属性，因为它们总是会返回值。

虽然属性被保证存在，但是它们的值*可以被动态地改变:*

>>>

```py
>>> buddy.age = 10
>>> buddy.age
10

>>> miles.species = "Felis silvestris"
>>> miles.species
'Felis silvestris'
```

在这个例子中，您将`buddy`对象的`.age`属性更改为`10`。然后你将`miles`对象的`.species`属性改为`"Felis silvestris"`，这是一种猫。这使得迈尔斯成为一只非常奇怪的狗，但它是一条有效的蟒蛇！

这里的关键是定制对象在默认情况下是可变的。如果一个对象可以动态改变，那么它就是可变的。例如，列表和[字典](https://realpython.com/python-dicts/)是可变的，但是字符串和元组是[不可变的](https://realpython.com/courses/immutability-python/)。

[*Remove ads*](/account/join/)

### 实例方法

**实例方法**是定义在类内部的函数，只能从该类的实例中调用。就像`.__init__()`一样，实例方法的第一个参数总是`self`。

在空闲状态下打开一个新的编辑器窗口，键入下面的`Dog`类:

```py
class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Instance method
    def description(self):
        return f"{self.name} is {self.age} years old"

    # Another instance method
    def speak(self, sound):
        return f"{self.name} says {sound}"
```

这个`Dog`类有两个实例方法:

1.  **`.description()`** 返回显示狗的名字和年龄的字符串。
2.  **`.speak()`** 有一个名为`sound`的参数，返回一个包含狗的名字和狗发出的声音的字符串。

将修改后的`Dog`类保存到名为`dog.py`的文件中，按 `F5` 运行程序。然后打开交互式窗口并键入以下内容，查看实例方法的运行情况:

>>>

```py
>>> miles = Dog("Miles", 4)

>>> miles.description()
'Miles is 4 years old'

>>> miles.speak("Woof Woof")
'Miles says Woof Woof'

>>> miles.speak("Bow Wow")
'Miles says Bow Wow'
```

在上面的`Dog`类中，`.description()`返回一个包含关于`Dog`实例`miles`信息的字符串。在编写自己的类时，最好有一个方法返回一个字符串，该字符串包含关于类实例的有用信息。然而，`.description()`并不是最[蟒](https://realpython.com/learning-paths/writing-pythonic-code/)的做法。

当您创建一个`list`对象时，您可以使用`print()`来显示一个类似于列表的字符串:

>>>

```py
>>> names = ["Fletcher", "David", "Dan"]
>>> print(names)
['Fletcher', 'David', 'Dan']
```

让我们看看当你`print()`这个`miles`对象时会发生什么:

>>>

```py
>>> print(miles)
<__main__.Dog object at 0x00aeff70>
```

当你`print(miles)`时，你会得到一个看起来很神秘的消息，告诉你`miles`是一个位于内存地址`0x00aeff70`的`Dog`对象。这条消息没什么帮助。您可以通过定义一个名为`.__str__()`的特殊实例方法来改变打印的内容。

在编辑器窗口中，将`Dog`类的`.description()`方法的名称改为`.__str__()`:

```py
class Dog:
    # Leave other parts of Dog class as-is

    # Replace .description() with __str__()
    def __str__(self):
        return f"{self.name} is {self.age} years old"
```

保存文件，按 `F5` 。现在，当你`print(miles)`时，你会得到一个更友好的输出:

>>>

```py
>>> miles = Dog("Miles", 4)
>>> print(miles)
'Miles is 4 years old'
```

像`.__init__()`和`.__str__()`这样的方法被称为 **dunder 方法**，因为它们以双下划线开始和结束。在 Python 中，有许多 dunder 方法可以用来定制类。虽然对于一本初级 Python 书籍来说，这是一个过于高级的主题，但是理解 dunder 方法是掌握 Python 中面向对象编程的重要部分。

在下一节中，您将看到如何更进一步，从其他类创建类。

[*Remove ads*](/account/join/)

### 检查你的理解能力

展开下面的方框，检查您的理解程度:



创建一个具有两个实例属性的`Car`类:

1.  `.color`，它将汽车颜色的名称存储为一个字符串
2.  `.mileage`，以整数形式存储汽车的里程数

然后实例化两个`Car`对象——一辆行驶 20，000 英里的蓝色汽车和一辆行驶 30，000 英里的红色汽车——并打印出它们的颜色和里程。您的输出应该如下所示:

```py
The blue car has 20,000 miles.
The red car has 30,000 miles.
```

您可以展开下面的方框查看解决方案:



首先，创建一个具有`.color`和`.mileage`实例属性的`Car`类:

```py
class Car:
    def __init__(self, color, mileage):
        self.color = color
        self.mileage = mileage
```

`.__init__()`的`color`和`mileage`参数被分配给`self.color`和`self.mileage`，这就创建了两个实例属性。

现在您可以创建两个`Car`实例:

```py
blue_car = Car(color="blue", mileage=20_000)
red_car = Car(color="red", mileage=30_000)
```

通过将值`"blue"`传递给`color`参数并将`20_000`传递给`mileage`参数来创建`blue_car`实例。类似地，`red_car`用值`"red"`和`30_000`创建。

要打印每个`Car`对象的颜色和里程，您可以在包含两个对象的`tuple`上循环:

```py
for car in (blue_car, red_car):
    print(f"The {car.color} car has {car.mileage:,} miles")
```

上述`for`循环中的 [f 字符串](https://realpython.com/python-f-strings/)将`.color`和`.mileage`属性插入到字符串中，并使用`:,` [格式说明符](https://realpython.com/python-formatted-output/#f-string-formatting)打印以千为单位分组并以逗号分隔的里程。

最终输出如下所示:

```py
The blue car has 20,000 miles.
The red car has 30,000 miles.
```

当你准备好了，你可以进入下一部分。

## 从 Python 中的其他类继承

继承是一个类继承另一个类的属性和方法的过程。新形成的类称为**子类**，子类派生的类称为**父类**。

**注:**本教程改编自 [*Python 基础知识:Python 实用入门 3*](https://realpython.com/products/python-basics-book/) 中“面向对象编程(OOP)”一章。如果你喜欢你正在阅读的东西，那么一定要看看这本书的其余部分。

子类可以重写或扩展父类的属性和方法。换句话说，子类继承父类的所有属性和方法，但也可以指定自己独有的属性和方法。

虽然这个类比并不完美，但是你可以把对象继承想象成类似于基因继承。

你可能从你母亲那里遗传了你的发色。这是你与生俱来的属性。假设你决定把头发染成紫色。假设你的母亲没有紫色的头发，你只是**覆盖了**你从你母亲那里继承的头发颜色属性。

从某种意义上说，你也从父母那里继承了你的语言。如果你的父母说英语，那么你也会说英语。现在想象你决定学习第二语言，比如德语。在这种情况下，您已经**扩展了**您的属性，因为您添加了一个您的父母没有的属性。

### 狗狗公园的例子

假设你在一个狗狗公园。公园里有很多不同品种的狗，都在从事各种狗的行为。

假设现在您想用 Python 类来建模 dog park。您在上一节中编写的`Dog`类可以通过名字和年龄来区分狗，但不能通过品种来区分。

您可以通过添加一个`.breed`属性来修改编辑器窗口中的`Dog`类:

```py
class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age, breed):
        self.name = name
        self.age = age
        self.breed = breed
```

这里省略了前面定义的实例方法，因为它们对于本次讨论并不重要。

按 `F5` 保存文件。现在，您可以通过在交互式窗口中实例化一群不同的狗来模拟狗公园:

>>>

```py
>>> miles = Dog("Miles", 4, "Jack Russell Terrier")
>>> buddy = Dog("Buddy", 9, "Dachshund")
>>> jack = Dog("Jack", 3, "Bulldog")
>>> jim = Dog("Jim", 5, "Bulldog")
```

每种狗的行为都略有不同。例如，牛头犬低沉的叫声听起来像汪汪叫，但是腊肠犬的叫声更高，听起来更像 T2 吠声。

仅使用`Dog`类，每次在`Dog`实例上调用`.speak()`的`sound`参数时，必须提供一个字符串:

>>>

```py
>>> buddy.speak("Yap")
'Buddy says Yap'

>>> jim.speak("Woof")
'Jim says Woof'

>>> jack.speak("Woof")
'Jack says Woof'
```

向每个对`.speak()`的调用传递一个字符串是重复且不方便的。此外，代表每个`Dog`实例发出的声音的字符串应该由它的`.breed`属性决定，但是这里您必须在每次调用它时手动将正确的字符串传递给`.speak()`。

您可以通过为每种狗创建一个子类来简化使用`Dog`类的体验。这允许您扩展每个子类继承的功能，包括为`.speak()`指定一个默认参数。

[*Remove ads*](/account/join/)

### 父类 vs 子类

让我们为上面提到的三个品种分别创建一个子类:杰克罗素梗、腊肠犬和牛头犬。

作为参考，下面是`Dog`类的完整定义:

```py
class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name} is {self.age} years old"

    def speak(self, sound):
        return f"{self.name} says {sound}"
```

记住，要创建一个子类，你要创建一个有自己名字的新类，然后把父类的名字放在括号里。将以下内容添加到`dog.py`文件中，以创建`Dog`类的三个新子类:

```py
class JackRussellTerrier(Dog):
    pass

class Dachshund(Dog):
    pass

class Bulldog(Dog):
    pass
```

按 `F5` 保存并运行文件。定义子类后，现在可以在交互窗口中实例化一些特定品种的狗:

>>>

```py
>>> miles = JackRussellTerrier("Miles", 4)
>>> buddy = Dachshund("Buddy", 9)
>>> jack = Bulldog("Jack", 3)
>>> jim = Bulldog("Jim", 5)
```

子类的实例继承父类的所有属性和方法:

>>>

```py
>>> miles.species
'Canis familiaris'

>>> buddy.name
'Buddy'

>>> print(jack)
Jack is 3 years old

>>> jim.speak("Woof")
'Jim says Woof'
```

要确定给定对象属于哪个类，可以使用内置的`type()`:

>>>

```py
>>> type(miles)
<class '__main__.JackRussellTerrier'>
```

如果你想确定`miles`是否也是`Dog`类的一个实例呢？你可以通过内置的`isinstance()`来实现:

>>>

```py
>>> isinstance(miles, Dog)
True
```

注意`isinstance()`有两个参数，一个对象和一个类。在上面的例子中，`isinstance()`检查`miles`是否是`Dog`类的实例，并返回`True`。

`miles`、`buddy`、`jack`和`jim`对象都是`Dog`实例，但是`miles`不是`Bulldog`实例，`jack`也不是`Dachshund`实例:

>>>

```py
>>> isinstance(miles, Bulldog)
False

>>> isinstance(jack, Dachshund)
False
```

更一般地说，从子类创建的所有对象都是父类的实例，尽管它们可能不是其他子类的实例。

现在你已经为一些不同品种的狗创建了子类，让我们给每个品种赋予它自己的声音。

### 扩展父类的功能

由于不同品种的狗的叫声略有不同，所以您希望为它们各自的`.speak()`方法的`sound`参数提供一个默认值。为此，你需要在每个品种的类定义中覆盖`.speak()`。

要重写在父类上定义的方法，需要在子类上定义一个同名的方法。下面是`JackRussellTerrier`类的情况:

```py
class JackRussellTerrier(Dog):
    def speak(self, sound="Arf"):
        return f"{self.name} says {sound}"
```

现在`.speak()`被定义在`JackRussellTerrier`类上，`sound`的默认参数被设置为`"Arf"`。

用新的`JackRussellTerrier`类更新`dog.py`并按 `F5` 保存并运行文件。现在，您可以在一个`JackRussellTerrier`实例上调用`.speak()`，而无需向`sound`传递参数:

>>>

```py
>>> miles = JackRussellTerrier("Miles", 4)
>>> miles.speak()
'Miles says Arf'
```

有时狗会发出不同的叫声，所以如果迈尔斯生气了，你仍然可以用不同的声音呼叫`.speak()`:

>>>

```py
>>> miles.speak("Grrr")
'Miles says Grrr'
```

关于类继承要记住的一点是，对父类的更改会自动传播到子类。只要被更改的属性或方法没有在子类中被重写，就会发生这种情况。

例如，在编辑器窗口中，更改由`Dog`类中的`.speak()`返回的字符串:

```py
class Dog:
    # Leave other attributes and methods as they are

    # Change the string returned by .speak()
    def speak(self, sound):
        return f"{self.name} barks: {sound}"
```

保存文件并按 `F5` 。现在，当您创建一个名为`jim`的新的`Bulldog`实例时，`jim.speak()`返回新的字符串:

>>>

```py
>>> jim = Bulldog("Jim", 5)
>>> jim.speak("Woof")
'Jim barks: Woof'
```

然而，在一个`JackRussellTerrier`实例上调用`.speak()`不会显示新的输出样式:

>>>

```py
>>> miles = JackRussellTerrier("Miles", 4)
>>> miles.speak()
'Miles says Arf'
```

有时完全重写父类的方法是有意义的。但是在这个实例中，我们不希望`JackRussellTerrier`类丢失任何可能对`Dog.speak()`的输出字符串格式进行的更改。

为此，您仍然需要在子类`JackRussellTerrier`上定义一个`.speak()`方法。但是不需要显式定义输出字符串，您需要使用传递给`JackRussellTerrier.speak()`的相同参数调用子类`.speak()`的内的`Dog`类的`.speak()` *。*

您可以使用 [`super()`](https://realpython.com/python-super/) 从子类的方法内部访问父类:

```py
class JackRussellTerrier(Dog):
    def speak(self, sound="Arf"):
        return super().speak(sound)
```

当您在`JackRussellTerrier`中调用`super().speak(sound)`时，Python 会在父类`Dog`中搜索一个`.speak()`方法，并用变量`sound`调用它。

用新的`JackRussellTerrier`类更新`dog.py`。保存文件并按下 `F5` ，这样您就可以在交互窗口中测试它了:

>>>

```py
>>> miles = JackRussellTerrier("Miles", 4)
>>> miles.speak()
'Miles barks: Arf'
```

现在，当您调用`miles.speak()`时，您将看到输出反映了`Dog`类中的新格式。

**注意:**在上面的例子中，**类的层次结构**非常简单。`JackRussellTerrier`类只有一个父类`Dog`。在现实世界的例子中，类的层次结构会变得非常复杂。

不仅仅是在父类中搜索方法或属性。它遍历整个类层次结构，寻找匹配的方法或属性。如果不小心，`super()`可能会有惊人的结果。

[*Remove ads*](/account/join/)

### 检查你的理解能力

展开下面的方框，检查您的理解程度:



创建一个继承自`Dog`类的`GoldenRetriever`类。给`GoldenRetriever.speak()`的`sound`参数一个默认值`"Bark"`。为您的父类`Dog`使用以下代码:

```py
class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name} is {self.age} years old"

    def speak(self, sound):
        return f"{self.name} says {sound}"
```

您可以展开下面的方框查看解决方案:



创建一个名为`GoldenRetriever`的类，它继承了`Dog`类并覆盖了`.speak()`方法:

```py
class GoldenRetriever(Dog):
    def speak(self, sound="Bark"):
        return super().speak(sound)
```

`GoldenRetriever.speak()`中的`sound`参数被赋予默认值`"Bark"`。然后用`super()`调用父类的`.speak()`方法，传递给`sound`的参数与`GoldenRetriever`类的`.speak()`方法相同。

## 结论

在本教程中，您学习了 Python 中的面向对象编程(OOP)。大多数现代编程语言，如 [Java](https://go.java/) 、 [C#](https://docs.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/) 、 [C++](https://www.cplusplus.com/info/description/) ，都遵循 OOP 原则，因此无论你的编程生涯走向何方，你在这里学到的知识都将适用。

**在本教程中，您学习了如何:**

*   定义一个**类**，它是一种对象的蓝图
*   从一个类中实例化一个**对象**
*   使用**属性**和**方法**来定义对象的**属性**和**行为**
*   使用**继承**从**父类**创建**子类**
*   使用 **`super()`** 引用父类上的方法
*   使用 **`isinstance()`** 检查一个对象是否从另一个类继承

如果你喜欢在这个例子中从 [*Python 基础知识:Python 3*](https://realpython.com/products/python-basics-book/) 实用介绍中学到的东西，那么一定要看看[这本书的其余部分](https://realpython.com/products/python-basics-book/)。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 中的面向对象编程(OOP)介绍**](/courses/intro-object-oriented-programming-oop-python/)********