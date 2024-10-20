# Python 中的封装

> 原文：<https://www.askpython.com/python/oops/encapsulation-in-python>

当使用像 Python 这样的面向对象编程语言时，Python 中的封装是需要理解的 4 个重要概念之一。另外三个是[继承](https://www.askpython.com/python/oops/inheritance-in-python)、[多态](https://www.askpython.com/python/oops/polymorphism-in-python)和抽象。

## 什么是封装？

当使用类和处理敏感数据时，提供对程序中使用的所有变量的全局访问不是一个好的选择。封装为我们提供了一种访问所需变量的方法，而无需为程序提供对这些变量的完全访问。

通过使用专门为此目的定义的方法，可以更新、修改或删除变量中的数据。使用这种编程方法的好处是改善了对输入数据的控制和更好的安全性。

## Python 中的封装是什么？

封装的概念在所有面向对象的编程语言中都是相同的。当这些概念应用于特定的语言时，差异就显现出来了。

与 Java 等为变量和方法提供访问修饰符(公共或私有)的语言相比，Python 提供了对所有变量和方法的全局访问。

查看下面的演示，了解如何轻松访问变量。

```py
class Person:
    def __init__(self, name, age=0):
        self.name = name
        self.age = age

    def display(self):
        print(self.name)
        print(self.age)

person = Person('Dev', 30)
#accessing using class method
person.display()
#accessing directly from outside
print(person.name)
print(person.age)

```

**输出**

```py
Dev
30
Dev
30

```

由于 Python 中没有访问修饰符，我们将使用一些不同的方法来控制 Python 程序中变量的访问。

* * *

## 控制访问的方法

Python 提供了多种方法来限制程序中变量和方法的访问。让我们详细讨论一下这些方法。

## 使用单下划线

标识私有变量的一个常见 Python 编程约定是在它前面加一个下划线。现在，这对于编译器来说并没有什么影响。该变量仍然可以像往常一样访问。但是作为程序员已经学会的惯例，它告诉其他程序员变量或方法只能在类的范围内使用。

请参见下面的示例:

```py
class Person:
    def __init__(self, name, age=0):
        self.name = name
        self._age = age

    def display(self):
        print(self.name)
        print(self._age)

person = Person('Dev', 30)
#accessing using class method
person.display()
#accessing directly from outside
print(person.name)
print(person._age)

```

**输出**

```py
Dev
30
Dev
30

```

很明显变量 access 是不变的。但是我们能做些什么来真正使它成为私有的吗？让我们进一步看看。

* * *

## 使用双下划线

如果你想让类成员，也就是方法和变量私有，那么你应该在它们前面加双下划线。但是 Python 为 private 修饰符提供了某种支持。这种机制被称为**名为**。这样，仍然可以从外部访问类成员。

#### 名字叫莽林

在 python 中，任何带有 __Var 的标识符都会被 Python 解释器重写为 _Classname__Var，而类名仍然是当前的类名。这种改名的机制在 Python 中被称为**名字篡改**。

在下面的例子中，在 person 类中，年龄变量被改变了，它的前缀是双下划线。

```py
class Person:
    def __init__(self, name, age=0):
        self.name = name
        self.__age = age

    def display(self):
        print(self.name)
        print(self.__age)

person = Person('Dev', 30)
#accessing using class method
person.display()
#accessing directly from outside
print('Trying to access variables from outside the class ')
print(person.name)
print(person.__age)

```

**输出**

```py
Dev
30
Trying to access variables from outside the class
Dev
Traceback (most recent call last):
  File "Person.py", line 16, in <module>
    print(person.__age)
AttributeError: 'Person' object has no attribute '__age'

```

您可以观察到，仍然可以使用方法访问变量，这是类的一部分。但是你不能从外部直接访问年龄，因为它是一个私有变量。

* * *

## 使用 Getter 和 Setter 方法访问私有变量

如果您想要访问和更改私有变量，应该使用访问器(getter)方法和赋值器(setter)方法，因为它们是类的一部分。

```py
class Person:
    def __init__(self, name, age=0):
        self.name = name
        self.__age = age

    def display(self):
        print(self.name)
        print(self.__age)

    def getAge(self):
        print(self.__age)

    def setAge(self, age):
        self.__age = age

person = Person('Dev', 30)
#accessing using class method
person.display()
#changing age using setter
person.setAge(35)
person.getAge()

```

**输出**

```py
Dev
30
35

```

* * *

## Python 中封装的好处

封装不仅确保了更好的数据流，还保护了来自外部数据源的数据。封装的概念使得代码能够自给自足。它在实现层面非常有帮助，因为它优先考虑“如何”类型的问题，把复杂性抛在后面。您应该将数据隐藏在单元中，以便于封装并保护数据。

## Python 中封装的需求是什么

下面的原因说明了为什么开发人员觉得封装很方便，为什么面向对象的概念超越了许多编程语言。

*   封装有助于在每个应用程序中实现定义良好的交互。
*   面向对象的概念侧重于 Python 中代码的可重用性。(干——不要重复)。
*   可以安全地维护应用程序。
*   它通过适当的代码组织确保了代码的灵活性。
*   它促进了用户的流畅体验，而没有暴露任何后端复杂性。
*   它提高了代码的可读性。一部分代码的任何变化都不会影响到另一部分。
*   封装可确保数据保护，避免意外访问数据。可以用上面讨论的方法访问受保护的数据。

Python 中的封装是，数据隐藏在对象定义之外。它使开发人员能够开发用户友好的体验。这也有助于保护数据免遭破坏，因为代码是高度安全的，不能被外部来源访问。

引用: [Python 类和私有变量](https://docs.python.org/3.4/tutorial/classes.html)