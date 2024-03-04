# Python 中的类

> 原文：<https://www.pythonforbeginners.com/basics/classes-in-python>

如果你是 python 编程的初学者，你必须了解像整数、浮点数、字符串和复数这样的基本数据类型。此外，您可能已经了解了内置数据结构，如 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)、列表、元组和集合。在本文中，我们将学习使用 python 中的类创建复杂的数据类型来存储真实世界对象的信息。

## Python 中有哪些类？

正如我们所知，python 是一种面向对象的编程语言，我们可以用 python 创建对象来表示现实世界中的对象。

python 中的对象是标识现实世界对象的属性集合。

例如，如果我们试图用 python 将一个长方体描述为一个对象，我们将指定长方体的长度、宽度和高度的值，并将定义它的属性，如表面积、重量和体积。

为了定义长方体对象的属性，我们使用类。因此，类是 python 中对象的蓝图，类决定了对象的属性和功能。

在我们的长方体例子中，类将是一个构造，它定义了对象的长度、宽度、高度、表面积、重量和体积。定义长方体属性的类将被定义如下。

```py
class Cuboid:
    #code for defining constructor
    #codes to define methods
```

## 如何在 Python 中使用类创建对象？

类只是任何对象的蓝图，它们不能在程序中使用。为了创建由类定义的对象，我们使用类的构造函数来实例化对象。因此，对象也被称为类的实例。

类的构造函数是使用关键字 __init__()定义的特殊方法。构造函数定义对象的属性，并按如下方式初始化它们。

```py
class Cuboid:
    def __init__(self):
        self.length=0
        self.breadth=0
        self.height=0
        self.weight=0
```

我们还可以创建一个构造函数，它将属性值作为输入参数，然后像下面这样初始化它们。

```py
class Cuboid:
    def __init__(self, length, breadth, height, weight):
        self.length = length
        self.breadth = breadth
        self.height = height
        self.weight = weight
```

## 类别属性

对象的属性对于一个类的任何实例都是私有的。与此不同，类属性是类本身的属性，它们由类的每个实例共享。

类属性是在所有方法和构造函数之外的类头下面声明的，如下所示。

```py
class Cuboid:
    name = "Cuboid"

    def __init__(self, length, breadth, height, weight):
        self.length = length
        self.breadth = breadth
        self.height = height
        self.weight = weight 
```

在上面的代码中，我们为长方体类定义了一个类属性“name”。

## 在类中定义方法

为了定义对象的属性和功能，我们在类中定义方法。方法是在类中定义的函数，用来执行特定的任务并给出一些输出。

例如，从长方体的长度、宽度、高度和重量确定长方体的表面积、体积和密度的方法可以定义如下。

```py
class Cuboid:
    name = "Cuboid"

    def __init__(self, length, breadth, height, weight):
        self.length = length
        self.breadth = breadth
        self.height = height
        self.weight = weight

    def volume(self):
        x = self.length
        y = self.breadth
        z = self.height
        v = x * y * z
        print("The volume is:", v)

    def density(self):
        x = self.length
        y = self.breadth
        z = self.height
        v = x * y * z
        d = self.weight / v
        print("Density is:", d)

    def surface_area(self):
        x = self.length
        y = self.breadth
        z = self.height
        s = 2 * (x * y + y * z + x * z)
        print("The surface area is:", s)
```

定义了一个类的所有属性和方法后，我们可以在 python 程序中使用它来实例化一个对象，并如下使用它们。

```py
class Cuboid:
    name = "Cuboid"

    def __init__(self, length, breadth, height, weight):
        self.length = length
        self.breadth = breadth
        self.height = height
        self.weight = weight

    def volume(self):
        x = self.length
        y = self.breadth
        z = self.height
        v = x * y * z
        print("The volume is:", v)

    def density(self):
        x = self.length
        y = self.breadth
        z = self.height
        v = x * y * z
        d = self.weight / v
        print("Density is:", d)

    def surface_area(self):
        x = self.length
        y = self.breadth
        z = self.height
        s = 2 * (x * y + y * z + x * z)
        print("The surface area is:", s)

myCuboid = Cuboid(1, 2, 4,4.5)
myCuboid.density()
myCuboid.surface_area()
myCuboid.volume()
```

输出:

```py
Density is: 0.5625
The surface area is: 28
The volume is: 8
```

## 结论

在本文中，我们研究了 python 中的类的概念。我们还看到了如何使用长方体的例子在类中实现构造函数和方法。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。