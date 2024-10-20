# Python 201 -超级

> 原文：<https://www.blog.pythonlibrary.org/2016/05/25/python-201-super/>

过去我曾经简单地写过关于 super 的文章，但是我决定再写一些关于这个特殊 Python 函数的更有趣的东西。

早在 Python 2.2 中就引入了**超级**内置函数。超级函数将返回一个代理对象，该对象将方法调用委托给类型的父类或兄弟类。如果这一点还不清楚的话，它允许你做的是访问在类中被覆盖的继承方法。超级函数有两个用例。第一种是在单一继承中，super 可以用来引用父类，而不需要显式地命名它们。这可以使您的代码在将来更易于维护。这类似于你在其他编程语言中发现的行为，比如 Dylan 的*next-method*。

第二个用例是在动态执行环境中，super 支持协作式多重继承。这实际上是一个非常独特的用例，可能只适用于 Python，因为它在只支持单一继承的语言和静态编译的语言中都找不到。

即使在核心开发者中，super 也有相当多的争议。原始文档很混乱，使用 super 也很棘手。有些人甚至将 super 标记为有害的，尽管那篇文章似乎更适用于 super 的 Python 2 实现，而不是 Python 3 版本。我们将从如何在 Python 2 和 3 中调用 super 开始这一章。然后我们将学习方法解析顺序。

* * *

### Python 2 对 Python 3

让我们从一个常规的类定义开始。然后我们将使用 Python 2 添加 super，看看它是如何变化的。

```py

class MyParentClass(object):
    def __init__(self):
        pass

class SubClass(MyParentClass):
    def __init__(self):
        MyParentClass.__init__(self)

```

这对于单一继承来说是一个非常标准的设置。我们有一个基类和子类。基类的另一个名字是父类，甚至是超类。无论如何，在子类中我们也需要初始化父类。Python 的核心开发者觉得把这种东西做得更抽象更便携会是个好主意，于是就加了超级函数。在 Python 2 中，子类如下所示:

```py

class SubClass(MyParentClass):
    def __init__(self):
        super(SubClass, self).__init__()

```

**Python 3** 把这个简化了一点。让我们来看看:

```py

class MyParentClass():
    def __init__(self):
        pass

class SubClass(MyParentClass):
    def __init__(self):
        super().__init__()

```

您将注意到的第一个变化是父类不再需要显式地基于**对象**基类。第二个变化是对**超级**的调用。我们不再需要向它传递任何东西，但是 super 会隐式地做正确的事情。大多数类实际上都传递了参数，所以让我们看看在这种情况下超级签名是如何变化的:

```py

class MyParentClass():
    def __init__(self, x, y):
        pass

class SubClass(MyParentClass):
    def __init__(self, x, y):
        super().__init__(x, y)

```

这里我们只需要调用 super 的 **__init__** 方法并传递参数。还是很好很直接。

* * *

### 方法解析顺序(MRO)

方法解析顺序(MRO)只是派生该类的类型列表。所以如果你有一个类继承了另外两个类，你可能会认为 MRO 就是它自己和它继承的两个父类。然而，父类也继承了 Python 的基类:**object**。让我们来看一个例子，它将使这一点更加清楚:

```py

class X:
    def __init__(self):
        print('X')
        super().__init__()

class Y:
    def __init__(self):
        print('Y')
        super().__init__()

class Z(X, Y):
    pass

z = Z()
print(Z.__mro__)

```

这里我们创建了 3 个类。前两个只是打印出类名，最后一个继承了前两个。然后我们实例化这个类，并打印出它的 MRO:

```py

X
Y
(, <class>, <class>, <class>) 
```

如您所见，当您实例化它时，每个父类都打印出它的名称。然后我们得到方法解析顺序，是 ZXY 和 object。另一个很好的例子是，当您在基类中创建一个类变量，然后覆盖它时，会发生什么:

```py

class Base:
    var = 5
    def __init__(self):
        pass

class X(Base):
    def __init__(self):
        print('X')
        super().__init__()

class Y(Base):
    var = 10
    def __init__(self):
        print('Y')
        super().__init__()

class Z(X, Y):
    pass

z = Z()
print(Z.__mro__)
print(super(Z, z).var)

```

所以在这个例子中，我们创建了一个基类，class 变量设置为 5。然后我们创建基类的子类:X 和 Y。Y 覆盖基类的 class 变量并将其设置为 10。最后，我们创建继承自 X 和 y 的类 Z。当我们在类 Z 上调用 super 时，哪个类变量会被打印出来？尝试运行此代码，您将获得以下结果:

```py

X
Y
(, <class>, <class>, <class>, <class>)
10 
```

让我们稍微分析一下。类 Z 继承自 X 和 y。所以当我们问它它的 **var** 是什么时，MRO 会查看 X 是否被定义。它不在那里，所以它移动到 Y，Y 有它，所以这就是返回的内容。尝试向 X 添加一个类变量，您会看到它覆盖了 Y，因为它在方法解析顺序中是第一个。

有一个叫 Michele Simionato 的家伙创建了一个精彩的文档，详细描述了 Python 的方法解析顺序。你可以在这里查看:[https://www.python.org/download/releases/2.3/mro/](https://www.python.org/download/releases/2.3/mro/)。这是一个很长的阅读，你可能需要重读几遍，但它很好地解释了 MRO。顺便说一句，你可能注意到这篇文章被标记为针对 Python 2.3，但它仍然适用于 Python 3，即使现在对 super 的调用有点不同。

Python 3 中对 super 方法进行了小幅更新。在 Python 3 中，super 可以计算出它是从哪个类调用的，以及包含它的方法的第一个参数。甚至在第一个参数不叫 **self** 的时候也可以！这是你在 Python 3 中调用 **super()** 时看到的。在 Python 2 中，您需要调用 **super(ClassName，self)** ，但是在 Python 3 中这被简化了。

因为这个事实，超级知道如何解释 MRO，并且它将这个信息存储在下面的魔法属性中: **__thisclass__** 和 **__self_class__** 。让我们看一个例子:

```py

class Base():
    def __init__(self):
        s = super()
        print(s.__thisclass__)
        print(s.__self_class__)
        s.__init__()

class SubClass(Base):
    pass

sub = SubClass()

```

在这里，我们创建一个基类，并将超级调用赋给一个变量，这样我们就可以找出那些神奇的属性包含了什么。然后我们把它们打印出来并初始化。最后，我们创建基类的一个子类，并实例化该子类。输出到 stdout 的结果是:

```py

 <class>
```

这很酷，但是可能不太方便，除非你在做元类或混合类。

* * *

### 包扎

你会在互联网上看到很多有趣的超级例子。他们中的大部分人一开始会有点迷惑，但是你会发现他们，或者认为这真的很酷，或者想知道你为什么要这么做。就我个人而言，在我的大部分工作中，我并不需要 super，但它对向前兼容很有用。这意味着您希望您的代码在未来工作时尽可能少地进行更改。所以今天你可能从单一继承开始，但是一两年后，你可能会添加另一个基类。如果你正确地使用 super，那么这将是很容易添加的。我也看到了使用 super 进行依赖注入的论点，但是我没有看到后一种用例的任何好的、具体的例子。不过，这是一件值得记住的好事。

超级功能可能非常方便，也可能非常令人困惑，或者两者兼而有之。明智地使用它，它会很好地为你服务。

* * *

### 进一步阅读

*   关于 Python Super 需要知道的事情[【第 1 页，共 3 页】](http://www.artima.com/weblogs/viewpost.jsp?thread=236275)
*   [超算超](http://rhettinger.wordpress.com/2011/05/26/super-considered-super/)
*   超级上的 Python [文档](https://docs.python.org/3/library/functions.html#super)
*   Python 的 Super 很俏皮，[但是不能用](https://fuhm.net/super-harmful/)
*   Python 201: [什么是超级？](https://www.blog.pythonlibrary.org/2014/01/21/python-201-what-is-super/)
*   MRO 在 Python 中做了什么？
*   Python [方法解析顺序](https://www.python.org/download/releases/2.3/mro/)