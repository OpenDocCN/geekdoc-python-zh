# Python 201:什么是描述符？

> 原文：<https://www.blog.pythonlibrary.org/2016/06/10/python-201-what-are-descriptors/>

描述符早在 2.2 版本中就被引入 Python 了。它们为开发人员提供了向对象添加托管属性的能力。创建描述符需要的方法有 **__get__** 、 **__set__** 和 **__delete__** 。如果您定义了这些方法中的任何一个，那么您就创建了一个描述符。

描述符背后的思想是从对象的字典中获取、设置或删除属性。当您访问一个类属性时，这将启动查找链。如果查找的值是一个定义了描述符方法的对象，那么描述符方法将被调用。

描述符增强了 Python 内部的许多魔力。它们使得属性、方法甚至**超级**函数工作。它们还用于实现 Python 2.2 中引入的新样式类。

* * *

### 描述符协议

创建描述符的协议非常简单。您只需定义以下一项或多项:

*   __get__(self，obj，type=None)，返回值
*   __set__(self，obj，value)，返回 None
*   __delete__(self，obj)，返回 None

一旦您定义了至少一个，您就创建了一个描述符。如果您同时定义了 __get__ 和 __set__，您就创建了一个数据描述符。只定义了 __get__()的描述符称为非数据描述符，通常用于方法。描述符类型有这种区别的原因是，如果一个实例的字典碰巧有一个数据描述符，那么在查找过程中描述符将优先。如果实例的字典有一个条目与非数据描述符匹配，那么字典自己的条目将优先于描述符。

如果同时定义了 __get__ 和 __set__，也可以创建一个只读描述符，但是在调用 __set__ 方法时会引发一个 **AttributeError** 。

* * *

### 调用描述符

调用描述符最常见的方法是在访问属性时自动调用描述符。典型的例子是 **my_obj.attribute_name** 。这将使你的对象在**我的对象**中查找**属性名称**。如果你的**属性名**恰好定义了 __get__()，那么**属性名。__get__(my_obj)** 将被调用。这完全取决于你的实例是一个对象还是一个类。

这背后的神奇之处在于被称为 __getattribute__ 的神奇方法，它会把 **my_obj.a** 变成这个:**类型(my_obj)。__dict__['a']。__get__(a，type(a))** 。你可以在这里阅读 Python 文档中关于实现的所有内容:【https://docs.python.org/3/howto/descriptor.html. 

根据所述文档，关于调用描述符，有几点需要记住:

*   描述符是通过 __getattribute__ 方法的默认实现调用的
*   如果您覆盖 __getattribute__，这将阻止描述符被自动调用
*   对象。__getattribute__()并键入。__getattribute__()不要以同样的方式调用 __get__ ()
*   数据描述符总是会覆盖实例字典
*   非数据描述符可以被实例字典覆盖。

关于所有这些如何工作的更多信息可以在 Python 的[数据模型](https://docs.python.org/3/reference/datamodel.html#object.__getattribute__)、Python 源代码和吉多·范·罗苏姆的文档[“在 Python 中统一类型和类”](https://www.python.org/download/releases/2.2.3/descrintro/#cooperation)中找到。

* * *

### 描述符示例

此时，您可能会对如何使用描述符感到困惑。当我学习一个新概念时，如果我有几个例子来演示它是如何工作的，我总是觉得很有帮助。所以在这一节中，我们将看一些例子，这样你将知道如何在你自己的代码中使用描述符！

让我们从编写一个非常简单的数据描述符开始，然后在一个类中使用它。这个例子基于 Python 文档中的一个例子:

```py
class MyDescriptor():
    """
    A simple demo descriptor
    """
    def __init__(self, initial_value=None, name='my_var'):
        self.var_name = name
        self.value = initial_value

    def __get__(self, obj, objtype):
        print('Getting', self.var_name)
        return self.value

    def __set__(self, obj, value):
        msg = 'Setting {name} to {value}'
        print(msg.format(name=self.var_name, value=value))
        self.value = value

class MyClass():
    desc = MyDescriptor(initial_value='Mike', name='desc')
    normal = 10

if __name__ == '__main__':
    c = MyClass()
    print(c.desc)
    print(c.normal)
    c.desc = 100
    print(c.desc)

```

这里我们创建了一个类并定义了三个神奇的方法:

*   __init__ -我们的构造函数，它接受一个值和变量的名称
*   __get__ -打印当前变量名并返回值
*   __set__ -打印出变量的名称和我们刚刚赋值的值，并自己设置值

然后我们创建一个类，它创建描述符的一个实例作为类属性，还创建一个普通的类属性。然后，我们通过创建一个普通类的实例并访问我们的类属性来运行一些“测试”。以下是输出:

```py
Getting desc
Mike
10
Setting desc to 100
Getting desc
100

```

如你所见，当我们访问 **c.desc** 时，它打印出我们的“Getting”消息，我们打印出它返回的内容，即“Mike”。接下来，我们打印出常规类属性的值。最后，我们更改描述符变量的值，这导致我们的“设置”消息被打印出来。我们还要仔细检查当前值，以确保它确实被设置了，这就是为什么您会看到最后一条“获取”消息。

Python 使用描述符来构建属性、绑定/未绑定方法和类方法。如果您在 Python 的文档中查找 property 类，您会发现它非常接近描述符协议:

```py
property(fget=None, fset=None, fdel=None, doc=None)

```

它清楚地显示了 property 类有一个 getter、setter 和一个 deleting 方法。

让我们看另一个例子，在这里我们使用描述符来进行验证:

```py
from weakref import WeakKeyDictionary

class Drinker:
    def __init__(self):
        self.req_age = 21
        self.age = WeakKeyDictionary()

    def __get__(self, instance_obj, objtype):
        return self.age.get(instance_obj, self.req_age)

    def __set__(self, instance, new_age):
        if new_age < 21:
            msg = '{name} is too young to legally imbibe'
            raise Exception(msg.format(name=instance.name))
        self.age[instance] = new_age
        print('{name} can legally drink in the USA'.format(
            name=instance.name))

    def __delete__(self, instance):
        del self.age[instance]

class Person:
    drinker_age = Drinker()

    def __init__(self, name, age):
        self.name = name
        self.drinker_age = age

p = Person('Miguel', 30)
p = Person('Niki', 13)

```

我们再次创建一个描述符类。在这种情况下，我们使用 Python 的 **weakref** 库的 **WeakKeyDictionary** ，这是一个简洁的类，它创建了一个弱映射键的字典。这意味着当字典中没有对某个键的强引用时，该键及其值将被丢弃。我们在这个例子中使用它来防止我们的 Person 实例无限期地徘徊。

无论如何，我们最关心的描述符部分在我们的 __set__ 方法中。在这里，我们检查实例的**年龄**参数是否大于 21，如果你想喝酒精饮料，这是你在美国需要达到的年龄。如果你的年龄较低，那么它将引发一个异常。否则它会打印出这个人的名字和一条信息。为了测试我们的描述符，我们创建了两个实例，一个大于 21 岁，另一个小于 21 岁。如果运行此代码，您应该会看到以下输出:

```py
Miguel can legally drink in the USA
Traceback (most recent call last):
  File "desc_validator.py", line 32, in 
    p = Person('Niki', 13)
  File "desc_validator.py", line 28, in __init__
    self.drinker_age = age
  File "desc_validator.py", line 14, in __set__
    raise Exception(msg.format(name=instance.name))
Exception: Niki is too young to legally imbibe

```

这显然是按照预期的方式工作的，但它是如何工作的并不明显。这样做的原因是当我们设置 **drinker_age** 时，Python 注意到它是一个描述符。Python 知道 **drinker_age** 是一个描述符，因为我们在创建它作为一个类属性时就这样定义了它:

```py
drinker_age = Drinker()

```

因此，当我们去设置它时，我们实际上调用了描述符的 __set__ 方法，该方法传入了实例和我们试图设置的年龄。如果年龄小于 21 岁，那么我们用一个自定义消息引发一个异常。否则，我们会打印出一条消息，说明您已经足够大了。

回到这一切是如何工作的，如果我们试图打印出 drinker_age，Python 将执行 Person.drinker_age。__get__。因为 drinker_age 是一个描述符，所以它的 __get__ 才是真正被调用的。如果你想设置饮酒者年龄，你可以这样做:

```py
p.drinker_age = 32

```

Python 会调用 **Person.drinker_age。__set__** 由于该方法也在我们的描述符中实现，因此描述符方法是被调用的方法。一旦您跟踪几次代码执行，您将很快看到这一切是如何工作的。

要记住的主要事情是描述符链接到类而不是实例。

* * *

### 包装材料

描述符非常重要，因为它们出现在 Python 源代码的所有地方。如果你了解它们是如何工作的，它们也会对你非常有用。然而，它们的用例非常有限，您可能不会经常使用它们。希望本文能帮助您了解描述符的用处，以及您自己何时可能需要使用它。

* * *

### 相关阅读

*   描述符[操作方法](https://docs.python.org/3/howto/descriptor.html)
*   伊恩在[描述符](http://www.ianbicking.org/blog/2008/10/decorators-and-descriptors.html)上斗嘴
*   Ned Batchelder 关于描述符的文章
*   IBM on Python [描述符](http://www.ibm.com/developerworks/library/os-pythondescriptors/)
*   Python 描述符[(第 1 部分，共 2 部分)](http://martyalchin.com/2007/nov/23/python-descriptors-part-1-of-2/s)
*   类和对象 II: [描述符](http://intermediatepythonista.com/classes-and-objects-ii-descriptors)
*   Python 描述符使[变得简单](https://www.smallsurething.com/python-descriptors-made-simple/)