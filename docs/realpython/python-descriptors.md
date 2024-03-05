# Python 描述符:简介

> 原文：<https://realpython.com/python-descriptors/>

描述符是 Python 的一个特殊特性，它赋予了隐藏在语言背后的许多魔力。如果您曾经认为 Python 描述符是一个很少有实际应用的高级主题，那么本教程是帮助您理解这一强大特性的完美工具。您将会理解为什么 Python 描述符是如此有趣的主题，以及您可以将它们应用到什么样的用例中。

本教程结束时，你会知道:

*   什么是 **Python 描述符**
*   在 Python 的**内部**中使用它们
*   如何**实现**你自己的描述符
*   **何时使用** Python 描述符

本教程面向中高级 Python 开发人员，因为它涉及到 Python 的内部原理。然而，如果你还没有达到这个水平，那就继续读下去吧！您将找到关于 Python 和查找链的有用信息。

**免费奖励:** ，它向您展示了三种高级装饰模式和技术，您可以用它们来编写更简洁、更 Python 化的程序。

## 什么是 Python 描述符？

描述符是 Python 对象，它实现了**描述符协议**的一种方法，这种方法使您能够创建在作为其他对象的属性被访问时具有特殊行为的对象。在这里，您可以看到描述符协议的正确定义:

```py
__get__(self, obj, type=None) -> object
__set__(self, obj, value) -> None
__delete__(self, obj) -> None
__set_name__(self, owner, name)
```

如果你的描述符只实现了`.__get__()`，那么它就是一个**非数据描述符**。如果它实现了`.__set__()`或`.__delete__()`，那么它就是一个**数据描述符**。请注意，这种差异不仅仅是名称上的，也是行为上的差异。这是因为数据描述符在查找过程中具有优先权，稍后您将会看到这一点。

看一下下面的例子，它定义了一个描述符，当控制台被访问时，该描述符在控制台上记录一些内容:

```py
# descriptors.py
class Verbose_attribute():
    def __get__(self, obj, type=None) -> object:
        print("accessing the attribute to get the value")
        return 42
    def __set__(self, obj, value) -> None:
        print("accessing the attribute to set the value")
        raise AttributeError("Cannot change the value")

class Foo():
    attribute1 = Verbose_attribute()

my_foo_object = Foo()
x = my_foo_object.attribute1
print(x)
```

在上面的例子中，`Verbose_attribute()`实现了描述符协议。一旦它被实例化为`Foo`的一个属性，它就可以被认为是一个描述符。

作为描述符，当使用点符号访问它时，它有**绑定行为**。在这种情况下，每次访问描述符以获取或设置值时，描述符都会在控制台上记录一条消息:

*   当它被访问到`.__get__()`值时，它总是返回值`42`。
*   当它被访问到`.__set__()`一个特定值时，它引发一个`AttributeError` [异常](https://realpython.com/courses/introduction-python-exceptions/)，这是[推荐的方式](https://docs.python.org/3/howto/descriptor.html#descriptor-protocol)来实现**只读**描述符。

现在，运行上面的示例，您将看到描述符在返回常量值之前记录对控制台的访问:

```py
$ python descriptors.py
accessing the attribute to get the value
42
```

这里，当您试图访问`attribute1`时，描述符会将此访问记录到控制台，如`.__get__()`中所定义的。

[*Remove ads*](/account/join/)

## 描述符如何在 Python 内部工作

如果您有作为面向对象 Python 开发人员的经验，那么您可能会认为前一个例子的方法有点矫枉过正。使用属性也可以达到同样的效果。虽然这是真的，但您可能会惊讶地发现 Python 中的属性只是…描述符！稍后您将会看到，属性并不是利用 Python 描述符的唯一特性。

### 属性中的 Python 描述符

如果您想在不显式使用 Python 描述符的情况下获得与上一个示例相同的结果，那么最直接的方法是使用一个**属性**。下面的示例使用了一个属性，该属性在控制台被访问时记录消息:

```py
# property_decorator.py
class Foo():
    @property
    def attribute1(self) -> object:
        print("accessing the attribute to get the value")
        return 42

    @attribute1.setter
    def attribute1(self, value) -> None:
        print("accessing the attribute to set the value")
        raise AttributeError("Cannot change the value")

my_foo_object = Foo()
x = my_foo_object.attribute1
print(x)
```

上面的例子使用了[装饰器](https://realpython.com/primer-on-python-decorators/)来定义一个带有附加的 [getter 和 setter](https://realpython.com/python-getter-setter/) 方法的属性。但是你可能知道，装饰者只是语法糖。事实上，前面的例子可以写成如下:

```py
# property_function.py
class Foo():
    def getter(self) -> object:
        print("accessing the attribute to get the value")
        return 42

    def setter(self, value) -> None:
        print("accessing the attribute to set the value")
        raise AttributeError("Cannot change the value")

    attribute1 = property(getter, setter)

my_foo_object = Foo()
x = my_foo_object.attribute1
print(x)
```

现在你可以看到这个属性已经被使用 [`property()`](https://realpython.com/python-property/) 创建了。该函数的签名如下:

```py
property(fget=None, fset=None, fdel=None, doc=None) -> object
```

`property()`返回一个实现描述符协议的`property`对象。它使用参数`fget`、`fset`和`fdel`来实际实现协议的三种方法。

### 方法和函数中的 Python 描述符

如果你曾经用 Python 写了一个[面向对象的程序](https://realpython.com/python3-object-oriented-programming/)，那么你肯定使用过**方法**。这些是为对象实例保留第一个参数的常规函数。当您使用点符号访问一个方法时，您正在调用相应的函数并将对象实例作为第一个参数传递。

将您的`obj.method(*args)`调用转换成`method(obj, *args)`的神奇之处在于`function`对象的`.__get__()`实现，事实上，它是一个**非数据描述符**。特别是，`function`对象实现了`.__get__()`，这样当你用点符号访问它时，它会返回一个绑定方法。接下来的`(*args)`通过传递所有需要的额外参数来调用函数。

为了了解它是如何工作的，请看一下这个来自官方文档的纯 Python 例子:

```py
import types

class Function(object):
    ...
    def __get__(self, obj, objtype=None):
        "Simulate func_descr_get() in Objects/funcobject.c"
        if obj is None:
            return self
        return types.MethodType(self, obj)
```

在上面的例子中，当用点符号访问函数时，调用`.__get__()`并返回一个绑定方法。

这适用于常规实例方法，就像它适用于类方法或静态方法一样。所以，如果你用`obj.method(*args)`调用一个静态方法，那么它会自动转换成`method(*args)`。类似地，如果你用`obj.method(type(obj), *args)`调用一个类方法，那么它会自动转换成`method(type(obj), *args)`。

**注:**要了解更多关于`*args`的信息，请查看 [Python args 和 kwargs:去神秘化](https://realpython.com/python-kwargs-and-args/)。

在[官方文档](https://docs.python.org/3/howto/descriptor.html#functions-and-methods)中，你可以找到一些例子，说明如果用纯 Python 而不是实际的 [C](https://realpython.com/build-python-c-extension-module/) 实现来编写[静态方法和类方法](https://realpython.com/instance-class-and-static-methods-demystified/)将如何实现。例如，一个可能的静态方法实现可能是这样的:

```py
class StaticMethod(object):
    "Emulate PyStaticMethod_Type() in Objects/funcobject.c"
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, objtype=None):
        return self.f
```

同样，这也可能是一个类方法实现:

```py
class ClassMethod(object):
    "Emulate PyClassMethod_Type() in Objects/funcobject.c"
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        def newfunc(*args):
            return self.f(klass, *args)
        return newfunc
```

注意，在 Python 中，类方法只是一个静态方法，它将类引用作为参数列表的第一个参数。

[*Remove ads*](/account/join/)

## 如何使用查找链访问属性

为了更好地理解 Python 描述符和 Python 内部机制，您需要理解当访问属性时 Python 中会发生什么。在 Python 中，每个对象都有一个内置的`__dict__`属性。这是一个包含对象本身定义的所有属性的字典。要了解这一点，请考虑以下示例:

```py
class Vehicle():
    can_fly = False
    number_of_weels = 0

class Car(Vehicle):
    number_of_weels = 4

    def __init__(self, color):
        self.color = color

my_car = Car("red")
print(my_car.__dict__)
print(type(my_car).__dict__)
```

这段代码创建了一个新对象，并打印了对象和类的`__dict__`属性的内容。现在，运行脚本并分析输出，查看设置的`__dict__`属性:

```py
{'color': 'red'}
{'__module__': '__main__', 'number_of_weels': 4, '__init__': <function Car.__init__ at 0x10fdeaea0>, '__doc__': None}
```

`__dict__`属性按预期设置。注意，在 Python 中，**一切都是对象**。一个类实际上也是一个对象，所以它也有一个包含该类所有属性和方法的`__dict__`属性。

那么，当你在 Python 中访问一个属性时，到底发生了什么呢？让我们用前一个例子的修改版本做一些测试。考虑以下代码:

```py
# lookup.py
class Vehicle(object):
    can_fly = False
    number_of_weels = 0

class Car(Vehicle):
    number_of_weels = 4

    def __init__(self, color):
        self.color = color

my_car = Car("red")

print(my_car.color)
print(my_car.number_of_weels)
print(my_car.can_fly)
```

在这个例子中，您创建了一个继承自`Vehicle`类的`Car`类的实例。然后，您可以访问一些属性。如果您运行这个示例，那么您可以看到您得到了所有您期望的值:

```py
$ python lookup.py
red
4
False
```

这里，当您访问实例`my_car`的属性`color`时，您实际上是在访问对象`my_car`的`__dict__`属性的单个值。当你访问对象`my_car`的属性`number_of_wheels`时，你实际上是在访问类`Car`的属性`__dict__`的单个值。最后，当您访问`can_fly`属性时，您实际上是通过使用`Vehicle`类的`__dict__`属性来访问它。

这意味着可以像这样重写上面的例子:

```py
# lookup2.py
class Vehicle():
    can_fly = False
    number_of_weels = 0

class Car(Vehicle):
    number_of_weels = 4

    def __init__(self, color):
        self.color = color

my_car = Car("red")

print(my_car.__dict__['color'])
print(type(my_car).__dict__['number_of_weels'])
print(type(my_car).__base__.__dict__['can_fly'])
```

当您测试这个新示例时，您应该会得到相同的结果:

```py
$ python lookup2.py
red
4
False
```

那么，当你用点符号访问一个对象的属性时会发生什么呢？解释器如何知道你真正需要的是什么？这里有一个叫做**查找链**的概念:

*   首先，您将获得从以您正在寻找的属性命名的**数据描述符**的`__get__`方法返回的结果。

*   如果失败了，那么您将获得您的对象的`__dict__`的值作为以您正在寻找的属性命名的键。

*   如果失败，那么您将获得从以您正在寻找的属性命名的**非数据描述符**的`__get__`方法返回的结果。

*   如果失败，那么您将获得您的对象类型的`__dict__`的值作为以您正在寻找的属性命名的键。

*   如果失败，那么您将获得您的对象父类型的`__dict__`的值，用于以您正在寻找的属性命名的键。

*   如果失败，那么对对象的[方法解析顺序](https://data-flair.training/blogs/python-multiple-inheritance/)中的所有父类型重复上一步。

*   如果其他的都失败了，那么你会得到一个`AttributeError`异常。

现在你可以明白为什么知道描述符是数据描述符还是 T2 非数据描述符很重要了吧？它们在查找链的不同层次上，稍后您会看到这种行为上的差异非常方便。

## 如何正确使用 Python 描述符

如果您想在代码中使用 Python 描述符，那么您只需要实现**描述符协议**。该协议最重要的方法是`.__get__()`和`.__set__()`，它们具有以下签名:

```py
__get__(self, obj, type=None) -> object
__set__(self, obj, value) -> None
```

当您实现协议时，请记住以下几点:

*   **`self`** 是你正在写的描述符的实例。
*   **`obj`** 是您的描述符附加到的对象的实例。
*   **`type`** 是描述符附加到的对象的类型。

在`.__set__()`中，你没有`type` [变量](https://realpython.com/python-variables/)，因为你只能在对象上调用`.__set__()`。相比之下，您可以在对象和类上调用`.__get__()`。

另一件需要知道的重要事情是 Python 描述符在每个类中只被实例化一次。这意味着包含描述符**的类的每个实例都共享那个描述符实例**。这可能是您没有预料到的，并且会导致一个典型的陷阱，就像这样:

```py
# descriptors2.py
class OneDigitNumericValue():
    def __init__(self):
        self.value = 0
    def __get__(self, obj, type=None) -> object:
        return self.value
    def __set__(self, obj, value) -> None:
        if value > 9 or value < 0 or int(value) != value:
            raise AttributeError("The value is invalid")
        self.value = value

class Foo():
    number = OneDigitNumericValue()

my_foo_object = Foo()
my_second_foo_object = Foo()

my_foo_object.number = 3
print(my_foo_object.number)
print(my_second_foo_object.number)

my_third_foo_object = Foo()
print(my_third_foo_object.number)
```

这里有一个类`Foo`，它定义了一个属性`number`，这是一个描述符。该描述符接受一个单位数字值，并将其存储在描述符本身的属性中。然而，这种方法行不通，因为`Foo`的每个实例共享同一个描述符实例。您实际上创建的只是一个新的类级属性。

尝试运行代码并检查输出:

```py
$ python descriptors2.py
3
3
3
```

您可以看到，`Foo`的所有实例都具有相同的属性`number`值，即使最后一个实例是在设置了`my_foo_object.number`属性之后创建的。

那么，如何解决这个问题呢？您可能认为使用字典来保存它所附加的所有对象的描述符的所有值是个好主意。这似乎是一个很好的解决方案，因为`.__get__()`和`.__set__()`都有`obj`属性，这是你所附加的对象的实例。您可以将该值用作字典的键。

不幸的是，这种解决方案有一个很大的缺点，您可以在下面的示例中看到:

```py
# descriptors3.py
class OneDigitNumericValue():
    def __init__(self):
        self.value = {}

    def __get__(self, obj, type=None) -> object:
        try:
            return self.value[obj]
        except:
            return 0

    def __set__(self, obj, value) -> None:
        if value > 9 or value < 0 or int(value) != value:
            raise AttributeError("The value is invalid")
        self.value[obj] = value

class Foo():
    number = OneDigitNumericValue()

my_foo_object = Foo()
my_second_foo_object = Foo()

my_foo_object.number = 3
print(my_foo_object.number)
print(my_second_foo_object.number)

my_third_foo_object = Foo()
print(my_third_foo_object.number)
```

在这个例子中，您使用一个字典来存储描述符中所有对象的`number`属性值。当您运行这段代码时，您会看到它运行良好，并且行为符合预期:

```py
$ python descriptors3.py
3
0
0
```

不幸的是，这里的缺点是描述符保留了对所有者对象的强引用。这意味着如果你销毁了对象，内存就不会被释放，因为[垃圾收集器](https://realpython.com/python-memory-management/#garbage-collection)一直在描述符中寻找对该对象的引用！

您可能认为这里的解决方案是使用弱引用。虽然可能如此，但您必须处理这样一个事实，即不是所有的东西都可以作为弱引用，而且当您的对象被收集时，它们会从您的字典中消失。

这里最好的解决方案是简单地*而不是*将值存储在描述符本身中，而是将它们存储在描述符所附加的*对象*中。接下来尝试这种方法:

```py
# descriptors4.py
class OneDigitNumericValue():
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name) or 0

    def __set__(self, obj, value) -> None:
        obj.__dict__[self.name] = value

class Foo():
    number = OneDigitNumericValue("number")

my_foo_object = Foo()
my_second_foo_object = Foo()

my_foo_object.number = 3
print(my_foo_object.number)
print(my_second_foo_object.number)

my_third_foo_object = Foo()
print(my_third_foo_object.number)
```

在本例中，当您为对象的`number`属性设置一个值时，描述符会使用与描述符本身相同的名称，将该值存储在它所附加到的对象的`__dict__`属性中。

这里唯一的问题是，当实例化描述符时，必须将名称指定为参数:

```py
number = OneDigitNumericValue("number")
```

直接写`number = OneDigitNumericValue()`不是更好吗？有可能，但是如果你运行的 Python 版本低于 3.6，那么你需要用[元类](https://realpython.com/python-metaclasses/)和[装饰器](https://realpython.com/primer-on-python-decorators/)来增加一点魔力。然而，如果您使用的是 [Python 3.6](https://docs.python.org/3/whatsnew/3.6.html) 或更高版本，那么描述符协议会有一个新方法`.__set_name__()`为您完成所有这些神奇的事情，正如在 [PEP 487](https://www.python.org/dev/peps/pep-0487/) 中所提议的:

```py
__set_name__(self, owner, name)
```

使用这个新方法，无论何时实例化一个描述符，这个方法都会被调用，并且自动设置参数`name`。

现在，尝试为 Python 3.6 及更高版本重写前面的示例:

```py
# descriptors5.py
class OneDigitNumericValue():
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name) or 0

    def __set__(self, obj, value) -> None:
        obj.__dict__[self.name] = value

class Foo():
    number = OneDigitNumericValue()

my_foo_object = Foo()
my_second_foo_object = Foo()

my_foo_object.number = 3
print(my_foo_object.number)
print(my_second_foo_object.number)

my_third_foo_object = Foo()
print(my_third_foo_object.number)
```

现在，`.__init__()`已经移除，`.__set_name__()`已经实现。这样就可以创建描述符，而不需要指定用于存储值的内部属性的名称。您的代码现在看起来也更好更干净了！

再运行一次这个例子，确保一切正常:

```py
$ python descriptors5.py
3
0
0
```

如果您使用 Python 3.6 或更高版本，这个示例应该运行起来没有问题。

[*Remove ads*](/account/join/)

## 为什么要使用 Python 描述符？

现在您知道了什么是 Python 描述符，以及 Python 本身如何使用它们来增强它的一些特性，比如方法和属性。您还了解了如何创建 Python 描述符，同时避免一些常见的陷阱。现在一切都应该很清楚了，但你可能仍然想知道为什么要使用它们。

根据我的经验，我认识很多高级 Python 开发人员，他们以前从未使用过这个特性，也不需要它。这很正常，因为没有多少用例需要 Python 描述符。然而，这并不意味着 Python 描述符只是高级用户的学术话题。仍然有一些好的用例可以证明学习如何使用它们的代价是值得的。

### 惰性属性

第一个也是最直接的例子是**惰性属性**。这些属性的初始值直到第一次被访问时才会被加载。然后，它们加载初始值，并缓存该值以供以后重用。

考虑下面的例子。您有一个包含方法`meaning_of_life()`的类`DeepThought`,该方法在花费大量时间集中精力后返回值:

```py
# slow_properties.py
import time

class DeepThought:
    def meaning_of_life(self):
        time.sleep(3)
        return 42

my_deep_thought_instance = DeepThought()
print(my_deep_thought_instance.meaning_of_life())
print(my_deep_thought_instance.meaning_of_life())
print(my_deep_thought_instance.meaning_of_life())
```

如果您运行这段代码并尝试访问该方法三次，那么您每三秒钟就会得到一个答案，这是该方法中的[睡眠](https://realpython.com/python-sleep/)时间的长度。

现在，一个惰性属性可以在这个方法第一次执行时只计算一次。然后，它将缓存结果值，这样，如果您再次需要它，您可以立即获得它。您可以通过使用 Python 描述符来实现这一点:

```py
# lazy_properties.py
import time

class LazyProperty:
    def __init__(self, function):
        self.function = function
        self.name = function.__name__

    def __get__(self, obj, type=None) -> object:
        obj.__dict__[self.name] = self.function(obj)
        return obj.__dict__[self.name]

class DeepThought:
    @LazyProperty
    def meaning_of_life(self):
        time.sleep(3)
        return 42

my_deep_thought_instance = DeepThought()
print(my_deep_thought_instance.meaning_of_life)
print(my_deep_thought_instance.meaning_of_life)
print(my_deep_thought_instance.meaning_of_life)
```

花点时间研究这段代码，了解它是如何工作的。你能在这里看到 Python 描述符的威力吗？在这个例子中，当您使用`@LazyProperty`描述符时，您正在实例化一个描述符并传递给它`.meaning_of_life()`。该描述符将方法及其名称存储为实例变量。

由于它是非数据描述符，当您第一次访问`meaning_of_life`属性的值时，`.__get__()`会被自动调用并在`my_deep_thought_instance`对象上执行`.meaning_of_life()`。结果值存储在对象本身的`__dict__`属性中。当您再次访问`meaning_of_life`属性时，Python 将使用**查找链**在`__dict__`属性中查找该属性的值，该值将被立即返回。

请注意，这是可行的，因为在本例中，您只使用了描述符协议的一种方法`.__get__()`。您还实现了一个非数据描述符。如果您实现了数据描述符，那么这个技巧就不会奏效。按照查找链，它将优先于存储在`__dict__`中的值。要对此进行测试，请运行以下代码:

```py
# wrong_lazy_properties.py
import time

class LazyProperty:
    def __init__(self, function):
        self.function = function
        self.name = function.__name__

    def __get__(self, obj, type=None) -> object:
        obj.__dict__[self.name] = self.function(obj)
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        pass

class DeepThought:
    @LazyProperty
    def meaning_of_life(self):
        time.sleep(3)
        return 42

my_deep_thought_instance = DeepThought()
print(my_deep_thought_instance.meaning_of_life)
print(my_deep_thought_instance.meaning_of_life)
print(my_deep_thought_instance.meaning_of_life)
```

在这个例子中，你可以看到仅仅实现了`.__set__()`、*，即使它根本没有做任何事情*，也创建了一个数据描述符。现在，懒惰属性的把戏停止工作。

### D.R.Y .代码

描述符的另一个典型用例是编写可重用的代码，并使您的代码成为可重用的代码。

考虑一个例子，其中有五个不同的属性具有相同的行为。只有当每个属性是偶数时，才能将其设置为特定值。否则，它的值被设置为 0:

```py
# properties.py
class Values:
    def __init__(self):
        self._value1 = 0
        self._value2 = 0
        self._value3 = 0
        self._value4 = 0
        self._value5 = 0

    @property
    def value1(self):
        return self._value1

    @value1.setter
    def value1(self, value):
        self._value1 = value if value % 2 == 0 else 0

    @property
    def value2(self):
        return self._value2

    @value2.setter
    def value2(self, value):
        self._value2 = value if value % 2 == 0 else 0

    @property
    def value3(self):
        return self._value3

    @value3.setter
    def value3(self, value):
        self._value3 = value if value % 2 == 0 else 0

    @property
    def value4(self):
        return self._value4

    @value4.setter
    def value4(self, value):
        self._value4 = value if value % 2 == 0 else 0

    @property
    def value5(self):
        return self._value5

    @value5.setter
    def value5(self, value):
        self._value5 = value if value % 2 == 0 else 0

my_values = Values()
my_values.value1 = 1
my_values.value2 = 4
print(my_values.value1)
print(my_values.value2)
```

如您所见，这里有许多重复的代码。可以使用 Python 描述符在所有属性之间共享行为。您可以创建一个`EvenNumber`描述符，并将其用于所有属性，如下所示:

```py
# properties2.py
class EvenNumber:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name) or 0

    def __set__(self, obj, value) -> None:
        obj.__dict__[self.name] = (value if value % 2 == 0 else 0)

class Values:
    value1 = EvenNumber()
    value2 = EvenNumber()
    value3 = EvenNumber()
    value4 = EvenNumber()
    value5 = EvenNumber()

my_values = Values()
my_values.value1 = 1
my_values.value2 = 4
print(my_values.value1)
print(my_values.value2)
```

这段代码现在看起来好多了！重复的部分消失了，逻辑现在在一个地方实现了，所以如果你需要改变它，你可以很容易地做到。

[*Remove ads*](/account/join/)

## 结论

既然您已经知道 Python 如何使用描述符来增强它的一些优秀特性，那么您将会是一个更有意识的开发人员，理解为什么一些 Python 特性以它们的方式实现。

**你已经学会:**

*   什么是 Python 描述符以及何时使用它们
*   Python 内部使用描述符的地方
*   如何实现自己的描述符

此外，您现在已经知道了 Python 描述符特别有用的一些特定用例。例如，当您有一个必须在许多属性之间共享的公共行为，甚至是不同类的属性时，描述符就很有用。

如果您有任何问题，请在下面留下评论或通过 [Twitter](https://www.twitter.com/mastro35) 联系我！如果你想更深入地研究 Python 描述符，那么请查看官方的 Python 描述符指南[。](https://docs.python.org/3/howto/descriptor.html)****