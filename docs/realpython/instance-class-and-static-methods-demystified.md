# Python 的实例、类和静态方法不再神秘

> 原文：<https://realpython.com/instance-class-and-static-methods-demystified/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解:[**Python 中的 OOP 方法类型:@ class Method vs @ static Method vs Instance Methods**](/courses/python-method-types/)

在本教程中，我将帮助揭开 *[类方法](https://docs.python.org/3/library/functions.html#classmethod)* 、 *[静态方法](https://docs.python.org/3/library/functions.html#staticmethod)* 和常规*实例方法*背后的秘密。

如果你对它们的区别有了直观的理解，你就能写出面向对象的 Python，更清楚地表达它的意图，从长远来看也更容易维护。

**免费奖励:** [点击此处获取免费的 Python OOP 备忘单](https://realpython.com/bonus/python-oop/)，它会为你指出最好的教程、视频和书籍，让你了解更多关于 Python 面向对象编程的知识。

## 实例、类和静态方法——概述

让我们从编写一个(Python 3)类开始，该类包含所有三种方法类型的简单示例:

```py
class MyClass:
    def method(self):
        return 'instance method called', self

    @classmethod
    def classmethod(cls):
        return 'class method called', cls

    @staticmethod
    def staticmethod():
        return 'static method called'
```

> **注意:**对于 Python 2 用户:`@staticmethod`和`@classmethod`装饰器从 Python 2.4 开始可用，这个例子将照常工作。不要使用普通的`class MyClass:`声明，你可以选择用`class MyClass(object):`语法声明一个继承自`object`的新型类。除此之外你可以走了。

[*Remove ads*](/account/join/)

### 实例方法

`MyClass`上的第一个方法叫做`method`，是一个常规的*实例方法*。这是基本的、没有多余装饰的方法类型，您将在大多数时间使用。您可以看到该方法有一个参数`self`，当调用该方法时，它指向`MyClass`的一个实例(当然，实例方法可以接受不止一个参数)。

通过`self`参数，实例方法可以自由访问同一个对象上的属性和其他方法。在修改对象的状态时，这给了它们很大的权力。

实例方法不仅可以修改对象状态，还可以通过`self.__class__`属性访问类本身。这意味着实例方法也可以修改类状态。

### 类方法

让我们将它与第二种方法`MyClass.classmethod`进行比较。我用一个 [`@classmethod`](https://docs.python.org/3/library/functions.html#classmethod) 装饰器标记了这个方法，以将其标记为一个*类方法*。

类方法不是接受一个`self`参数，而是接受一个`cls`参数，该参数在方法被调用时指向类，而不是对象实例。

因为类方法只能访问这个`cls`参数，所以它不能修改对象实例状态。这需要访问`self`。但是，类方法仍然可以修改应用于该类所有实例的类状态。

### 静态方法

第三种方法，`MyClass.staticmethod`被标记为 [`@staticmethod`](https://docs.python.org/3/library/functions.html#staticmethod) 装饰者将其标记为*静态方法*。

这种类型的方法既不接受`self`也不接受`cls`参数(当然，它可以接受任意数量的其他参数)。

因此，静态方法既不能修改对象状态，也不能修改类状态。静态方法在它们可以访问的数据方面受到限制——它们主要是一种给你的方法命名空间的方式。

## 让我们看看他们的行动吧！

我知道这个讨论到目前为止还只是理论性的。我相信，对这些方法类型在实践中的不同之处有一个直观的理解是很重要的。我们现在来看一些具体的例子。

让我们看看当我们调用这些方法时，它们是如何工作的。我们将首先创建该类的一个实例，然后调用它的三个不同的方法。

`MyClass`是以这样的方式建立的，每个方法的实现返回一个包含信息的元组，以便我们跟踪发生了什么——以及该方法可以访问类或对象的哪些部分。

下面是当我们调用一个**实例方法**时发生的情况:

>>>

```py
>>> obj = MyClass()
>>> obj.method()
('instance method called', <MyClass instance at 0x10205d190>)
```

这证实了`method`(实例方法)可以通过`self`参数访问对象实例(打印为`<MyClass instance>`)。

当调用该方法时，Python 用实例对象`obj`替换`self`参数。我们可以忽略点调用语法的语法糖(`obj.method()`)并手动传递实例对象*以获得相同的结果:*

*>>>

```py
>>> MyClass.method(obj)
('instance method called', <MyClass instance at 0x10205d190>)
```

如果不先创建实例就试图调用该方法，您能猜到会发生什么吗？

顺便说一下，实例方法也可以通过`self.__class__`属性访问*类本身*。这使得实例方法在访问限制方面非常强大——它们可以修改对象实例的状态*和类本身的状态*。

接下来让我们试试**类方法**:

>>>

```py
>>> obj.classmethod()
('class method called', <class MyClass at 0x101a2f4c8>)
```

调用`classmethod()`向我们展示了它不能访问`<MyClass instance>`对象，只能访问代表类本身的`<class MyClass>`对象(Python 中的一切都是对象，甚至是类本身)。

注意当我们调用`MyClass.classmethod()`时，Python 如何自动将类作为第一个参数传递给函数。通过*点语法*调用 Python 中的方法会触发这种行为。实例方法上的`self`参数以同样的方式工作。

请注意，命名这些参数`self`和`cls`只是一种约定。你可以很容易地将它们命名为`the_object`和`the_class`，并得到相同的结果。重要的是它们在方法的参数列表中位于第一位。

现在是调用静态方法的时候了:

>>>

```py
>>> obj.staticmethod()
'static method called'
```

你看到我们是如何在对象上调用`staticmethod()`并成功完成的吗？当一些开发人员得知可以在对象实例上调用静态方法时，他们感到很惊讶。

在幕后，Python 只是通过在使用点语法调用静态方法时不传入`self`或`cls`参数来实施访问限制。

这证实了静态方法既不能访问对象实例状态，也不能访问类状态。它们像常规函数一样工作，但是属于类(和每个实例)的名称空间。

现在，让我们看看当我们试图在类本身上调用这些方法时会发生什么——而不事先创建对象实例:

>>>

```py
>>> MyClass.classmethod()
('class method called', <class MyClass at 0x101a2f4c8>)

>>> MyClass.staticmethod()
'static method called'

>>> MyClass.method()
TypeError: unbound method method() must
 be called with MyClass instance as first
 argument (got nothing instead)
```

我们能够很好地调用`classmethod()`和`staticmethod()`，但是尝试调用实例方法`method()`失败，出现了`TypeError`。

这是意料之中的——这一次我们没有创建对象实例，而是尝试直接在类蓝图本身上调用实例函数。这意味着 Python 无法填充`self`参数，因此调用失败。

这应该会使这三种方法类型之间的区别更加清晰。但我不会就此罢休。在接下来的两节中，我将通过两个稍微现实一点的例子来说明何时使用这些特殊的方法类型。

我将围绕这个基本类给出我的例子:

```py
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    def __repr__(self):
        return f'Pizza({self.ingredients!r})'
```

>>>

```py
>>> Pizza(['cheese', 'tomatoes'])
Pizza(['cheese', 'tomatoes'])
```

> **注意:**这个代码示例和本教程后面的代码示例使用 [Python 3.6 f-strings](https://dbader.org/blog/python-string-formatting) 来构造由`__repr__`返回的字符串。在 Python 2 和 Python 3.6 之前的版本中，您可以使用不同的字符串格式表达式，例如:
> 
> ```py
> `def __repr__(self):
>     return 'Pizza(%r)' % self.ingredients` 
> ```

[*Remove ads*](/account/join/)

## 美味披萨工厂`@classmethod`

如果你在现实生活中接触过比萨，你会知道有许多美味的变化:

```py
Pizza(['mozzarella', 'tomatoes'])
Pizza(['mozzarella', 'tomatoes', 'ham', 'mushrooms'])
Pizza(['mozzarella'] * 4)
```

几个世纪前，意大利人就想出了他们的比萨饼分类，所以这些美味的比萨饼都有自己的名字。我们应该好好利用这一点，给我们的`Pizza`类的用户一个更好的界面来创建他们渴望的披萨对象。

一个漂亮而干净的方法是使用类方法作为我们可以创建的不同种类的披萨的工厂函数:

```py
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    def __repr__(self):
        return f'Pizza({self.ingredients!r})'

    @classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomatoes'])

    @classmethod
    def prosciutto(cls):
        return cls(['mozzarella', 'tomatoes', 'ham'])
```

注意我是如何在`margherita`和`prosciutto`工厂方法中使用`cls`参数，而不是直接调用`Pizza`构造函数的。

这是一个你可以用来遵循[不要重复自己(干)](https://en.wikipedia.org/wiki/Don't_repeat_yourself)原则的窍门。如果我们决定在某个时候重命名这个类，我们就不需要记住更新所有 classmethod 工厂函数中的构造函数名。

现在，我们能用这些工厂方法做什么呢？让我们尝试一下:

>>>

```py
>>> Pizza.margherita()
Pizza(['mozzarella', 'tomatoes'])

>>> Pizza.prosciutto()
Pizza(['mozzarella', 'tomatoes', 'ham'])
```

如您所见，我们可以使用工厂函数来创建新的`Pizza`对象，这些对象按照我们想要的方式进行配置。它们都在内部使用相同的`__init__`构造函数，只是提供了一个记住所有不同成分的捷径。

看待类方法这种用法的另一种方式是，它们允许您为类定义可选的构造函数。

Python 只允许每个类有一个`__init__`方法。使用类方法可以根据需要添加尽可能多的可选构造函数。这可以使您的类的接口自文档化(在一定程度上)并简化它们的使用。

## 何时使用静态方法

在这里想出一个好的例子有点困难。但是告诉你，我会继续把披萨比喻的越来越薄…(好吃！)

这是我想到的:

```py
import math

class Pizza:
    def __init__(self, radius, ingredients):
        self.radius = radius
        self.ingredients = ingredients

    def __repr__(self):
        return (f'Pizza({self.radius!r}, '
                f'{self.ingredients!r})')

    def area(self):
        return self.circle_area(self.radius)

    @staticmethod
    def circle_area(r):
        return r ** 2 * math.pi
```

我在这里改变了什么？首先，我修改了构造函数和`__repr__`来接受一个额外的`radius`参数。

我还添加了一个计算并返回比萨饼面积的`area()`实例方法(这也是一个很好的`@property`的候选方法——但是，嘿，这只是一个玩具示例)。

我没有使用众所周知的圆形面积公式直接在`area()`中计算面积，而是将其分解到一个单独的`circle_area()`静态方法中。

我们来试试吧！

>>>

```py
>>> p = Pizza(4, ['mozzarella', 'tomatoes'])
>>> p
Pizza(4, ['mozzarella', 'tomatoes'])
>>> p.area()
50.26548245743669
>>> Pizza.circle_area(4)
50.26548245743669
```

当然，这是一个有点简单的例子，但是它可以很好地帮助解释静态方法提供的一些好处。

正如我们所了解的，静态方法不能访问类或实例状态，因为它们没有带`cls`或`self`参数。这是一个很大的限制——但这也是一个很好的信号，表明一个特定的方法独立于它周围的一切。

在上面的例子中，很明显`circle_area()`不能以任何方式修改类或类实例。(当然，你可以用一个全局变量[和](https://realpython.com/python-variables/)来解决这个问题，但这不是这里的重点。)

这为什么有用呢？

将一个方法标记为静态方法不仅仅是暗示一个方法不会修改类或实例状态 Python 运行时也强制实施了这一限制。

像这样的技术可以让你清楚地交流你的类架构的各个部分，这样新的开发工作就可以自然地在这些设定的界限内进行。当然，无视这些限制是很容易的。但在实践中，它们通常有助于避免违背原始设计的意外修改。

换句话说，使用静态方法和类方法是传达开发人员意图的方式，同时充分实施该意图以避免大多数会破坏设计的疏忽错误和 bug。

谨慎应用，当有意义时，以这种方式编写一些方法可以提供维护好处，并减少其他开发人员错误使用您的类的可能性。

静态方法在编写测试代码时也有好处。

因为`circle_area()`方法完全独立于类的其余部分，所以更容易测试。

在单元测试中测试方法之前，我们不必担心设置一个完整的类实例。我们可以像测试一个常规函数一样开始工作。同样，这使得将来的维护更加容易。

[*Remove ads*](/account/join/)

## 关键要点

*   实例方法需要一个类实例，可以通过`self`访问实例。
*   类方法不需要类实例。他们不能访问实例(`self`)，但是他们可以通过`cls`访问类本身。
*   静态方法不能访问`cls`或`self`。它们像常规函数一样工作，但是属于类的命名空间。
*   静态方法和类方法相互交流，并(在一定程度上)强化开发人员对类设计的意图。这有利于维护。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解:[**Python 中的 OOP 方法类型:@ class Method vs @ static Method vs Instance Methods**](/courses/python-method-types/)******