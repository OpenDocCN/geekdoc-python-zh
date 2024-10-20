# 了解 Python 中的 __init__()方法

> 原文：<https://www.askpython.com/python/oops/init-method>

在本文中，我们讨论 OOPs 的概念——[Python 构造函数](https://www.askpython.com/python/oops/python-class-constructor-init-function),并详细解释如何使用 __init__()方法初始化对象。

## 什么是构造函数？

在我们进入构造函数的概念之前，这里有一个关于类和对象的快速介绍:

*“在 OOP 中，object 是指开发者创建的抽象数据类型。对象由其参数(状态值)和行为(方法)定义/表征。类是构建特定类型对象的蓝图或一组指令。*

构造函数可以非常简单地理解为在对象初始化/创建期间调用的特殊方法。构造函数通常被定义为类定义中的一个函数，它接受状态参数并使用这些用户定义的参数创建一个对象。

在 python 中，构造函数方法是 __init__()，编写为:

```py
def __init__(self, object_parameters...):
    # Initialize the object

```

该函数将 self 和对象参数作为输入。顾名思义，对象参数是定义对象的状态变量。

“self”是 python 中的保留关键字，表示类的实例。“self”关键字允许轻松访问该类的方法和参数，以便在该类的其他方法中使用。

例如，可以使用 self.var 访问对象状态变量“var”

当我们看到创建类的例子时，这个想法会变得更加清晰。

## 创建一个简单的类:笛卡尔点

在这个例子中，我们为 2D 笛卡尔点的类创建了一个类。这个类有两个状态变量——x 和 y，它们定义了点的位置。

```py
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def coords(self):
        print("The point is at: ({}, {})".format(self.x, self.y))

```

__init__()方法将 self 和两个状态变量作为输入。然后，我们通过将对象的状态变量设置为用户定义的值来初始化对象。

对象的状态变量(这里是 x 和 y)可以分别使用 self.x 和 self.y 来访问。

这里的另一个方法是 coords()，它打印点的当前位置。注意我们如何使用 self.x 和 self.y 访问这些点。

```py
# Create object `P` of class Point with value 3 and 4
P = Point1(3, 4)

# Print the coordinates of point `P`
P.coords()

```

![Image 23](img/c4a287ac4bc3edcf9567f51077d2e986.png)

## __init__()方法中的默认变量

像 python 中的任何其他函数一样，__init__()方法允许您使用默认值。当一个类接受大量输入参数时，这个特性特别有用。这里我们构造了另一个类 Point1，如果没有指定数据点，它将 Point 初始化为(0，0)。

```py
class Point1:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def coords(self):
        print("The point is at: ({}, {})".format(self.x, self.y))

# Create object of class Point1 with default parameters
Q = Point1()
Q.coords()

```

![Image 22](img/502fdb2d67d4f52103843bbf985f13c8.png)

## 结论

我们已经看到了如何为 python 类定义构造函数。虽然是最可行和最简单的方法，但事实证明这不是唯一的方法。Python 允许更复杂的类初始化，这在继承/扩展其他类时尤其重要。我们将在以后的文章中看到更多这样的例子。