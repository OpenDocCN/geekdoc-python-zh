# Python 中的数据类

> 原文：<https://www.pythonforbeginners.com/basics/dataclass-in-python>

在用 python 编程时，您可能已经使用了类来创建不同的对象。python 中的类对于在我们的程序中描述真实世界的物体非常有帮助。在本文中，我们将讨论一个名为 Dataclass 的装饰器，用它我们可以修改一个类的属性。我们还将讨论在用 python 编程时数据类的重要性。

## 什么是数据类？

Dataclass 是在 dataclasses 模块中定义的装饰器。它是在 python 3.7 中引入的。dataclass decorator 可以用来实现定义只有数据和极少功能的对象的类。

使用 dataclass decorator 定义的类有非常具体的用途和属性，我们将在下面的部分中讨论。让我们首先讨论如何在 python 中使用 dataclass 实现一个类。

## Python 中如何使用 dataclass？

dataclass 装饰器已在 dataclasses 模块中定义。您可以首先使用 PIP 安装 dataclasses 模块，如下所示。

```py
pip3 install --upgrade dataclasses 
```

安装 dataclasses 模块后，可以使用 import 语句导入 dataclass 装饰器，如下所示。

```py
from dataclasses import dataclass
```

现在让我们用 dataclass decorator 定义一个类。

```py
from dataclasses import dataclass

@dataclass
class Person:
    Name: str
    Country: str
    Age: int

candidate = Person("Joe Biden", "USA", 78)
print("The candidate is:",candidate) 
```

输出:

```py
The candidate is: Person(Name='Joe Biden', Country='USA', Age=78) 
```

您可能会注意到，我们在上面的代码中指定了类属性的数据类型。此外，在使用 dataclass 装饰器时，我们不需要实现 __init__()构造函数。装饰器本身为我们实现了 __init__()方法。

## 在 Python 中使用 dataclass 的好处

我们可以使用 dataclass decorator 定义类来表示对象。

如果我们想使用 print 语句打印对象的属性，当我们在没有使用 dataclass decorator 的情况下实现了一个类时，我们必须使用 __repr__()方法。否则，输出如下。

```py
class Person:
    def __init__(self, name, country, age):
        self.Name = name
        self.Country = country
        self.Age = age

candidate = Person("Joe Biden", "USA", 78)
print("The candidate is:", candidate) 
```

输出:

```py
The candidate is: <__main__.Person object at 0x7fb7289a8070> 
```

要打印类属性，我们必须实现 __repr__()方法，如下所示。

```py
class Person:
    def __init__(self, name, country, age):
        self.Name = name
        self.Country = country
        self.Age = age

    def __repr__(self):
        return "Name: {}, Country: {}, Age: {}".format(self.Name, self.Country, self.Age)

candidate = Person("Joe Biden", "USA", 78)
print("The candidate is:", candidate) 
```

输出:

```py
The candidate is: Name: Joe Biden, Country: USA, Age: 78
```

但是，当我们使用 dataclass decorator 时，所有的类属性都被打印出来，而没有实现 __repr__()方法。这可以在下面的例子中观察到。

```py
from dataclasses import dataclass

@dataclass
class Person:
    Name: str
    Country: str
    Age: int

candidate = Person("Joe Biden", "USA", 78)
print("The candidate is:",candidate)
```

输出:

```py
The candidate is: Person(Name='Joe Biden', Country='USA', Age=78)
```

简单类和带有 dataclass decorator 的类之间的另一个主要区别是比较类实例的方式。

例如，当我们创建一个类并使用==操作符比较它的实例时，python 解释器检查对象的标识或内存位置，只有当两个实例引用相同的内存位置时，它们才被认为是相等的。这可以在下面的程序中观察到。

```py
class Person:
    def __init__(self, name, country, age):
        self.Name = name
        self.Country = country
        self.Age = age

    def __repr__(self):
        return "Name: {}, Country: {}, Age: {}".format(self.Name, self.Country, self.Age)

candidate1 = Person("Joe Biden", "USA", 78)
candidate2 = Person("Joe Biden", "USA", 78)

print("Candidate 1 is:", candidate1)
print("Candidate 2 is:", candidate2)

print("Both the candidates are same?", candidate1 == candidate2) 
```

输出:

```py
Candidate 1 is: Name: Joe Biden, Country: USA, Age: 78
Candidate 2 is: Name: Joe Biden, Country: USA, Age: 78
Both the candidates are same? False
```

在这里，您可以看到候选 1 和候选 2 被认为是不同的，因为它们是不同的对象，并且引用不同的内存位置。

相反，当我们使用 dataclass decorator 定义一个类时，比较操作符的工作方式非常不同。当我们使用==操作符比较类的两个实例时，比较的是对象的类属性中的值，而不是内存位置。如果两个实例中相应属性的值相等，则称对象相等。您可以在下面的程序中观察到这一点。

```py
from dataclasses import dataclass

@dataclass
class Person:
    Name: str
    Country: str
    Age: int

candidate1 = Person("Joe Biden", "USA", 78)
candidate2 = Person("Joe Biden", "USA", 78)
print("The candidate 1 is:", candidate1)
print("The candidate 2 is:", candidate2)
print("Both the candidates are same?", candidate1 == candidate2) 
```

输出:

```py
The candidate 1 is: Person(Name='Joe Biden', Country='USA', Age=78)
The candidate 2 is: Person(Name='Joe Biden', Country='USA', Age=78)
Both the candidates are same? True
```

这里，两个候选对象被认为是相等的，因为对象的属性是相等的。因此，当我们使用 dataclass 装饰器实现类时，我们可以很容易地比较对象内部的数据。

您可以看到 dataclass 装饰器为我们提供了比较对象的更好方法。否则，我们将不得不定义方法来比较对象。这可能导致在时间和空间方面的高成本执行。

## 结论

在本文中，我们讨论了 python 中的 dataclass 装饰器。我们还实现了它，并看到了它的一些特殊属性，这些属性使它成为我们程序中有用的构造。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)