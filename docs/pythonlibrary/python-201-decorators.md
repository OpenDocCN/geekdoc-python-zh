# Python 201:装饰者

> 原文：<https://www.blog.pythonlibrary.org/2014/03/13/python-201-decorators/>

Python decorators 真的很酷，但是一开始可能有点难以理解。Python 中的装饰器是一个接受另一个函数作为参数的函数。装饰者通常会修改或增强它接受的函数，并返回修改后的函数。这意味着当你调用一个修饰函数时，你将得到一个与基本定义相比可能有一点不同的函数，它可能有额外的特性。但是让我们倒回去一点。我们也许应该回顾一下装饰器的基本构件，即函数。

* * *

### 简单的功能

函数是一个代码块，以 Python 关键字 **def** 开头，后跟函数的实际名称。一个函数可以接受零个或多个参数、关键字参数或两者的混合。一个函数总是会返回一些东西。如果你不指定函数应该返回什么，它将返回 **None** 。下面是一个非常简单的函数，它只返回一个字符串:

```py

#----------------------------------------------------------------------
def a_function():
    """A pretty useless function"""
    return "1+1"

#----------------------------------------------------------------------
if __name__ == "__main__":
    value = a_function()
    print(value)

```

我们调用函数并打印返回值。让我们创建另一个函数:

```py

#----------------------------------------------------------------------
def another_function(func):
    """
    A function that accepts another function
    """
    def other_func():
        val = "The result of %s is %s" % (func(),
                                          eval(func())
                                          )
        return val
    return other_func

```

该函数接受一个参数，并且该参数必须是函数或可调用的。事实上，它实际上只应该使用前面定义的函数来调用。你会注意到这个函数内部有一个嵌套函数，我们称之为 **other_func** 。它将接受传递给它的函数的结果，对它进行求值，并创建一个字符串来告诉我们它做了什么，然后返回这个字符串。让我们看看完整版的代码:

```py

#----------------------------------------------------------------------
def another_function(func):
    """
    A function that accepts another function
    """

    def other_func():
        val = "The result of %s is %s" % (func(),
                                          eval(func())
                                          )
        return val
    return other_func

#----------------------------------------------------------------------
def a_function():
    """A pretty useless function"""
    return "1+1"

#----------------------------------------------------------------------
if __name__ == "__main__":
    value = a_function()
    print(value)
    decorator = another_function(a_function)
    print decorator()

```

这就是装修工的工作方式。我们创建一个函数，然后将它传递给第二个函数。第二个函数是装饰函数。装饰器将修改或增强传递给它的函数，并返回修改内容。如果您运行此代码，您应该会看到以下内容作为 stdout 的输出:

```py

1+1
The result of 1+1 is 2

```

让我们稍微修改一下代码，将 **another_function** 变成一个装饰器:

```py

#----------------------------------------------------------------------
def another_function(func):
    """
    A function that accepts another function
    """

    def other_func():
        val = "The result of %s is %s" % (func(),
                                          eval(func())
                                          )
        return val
    return other_func

#----------------------------------------------------------------------
@another_function
def a_function():
    """A pretty useless function"""
    return "1+1"

#----------------------------------------------------------------------
if __name__ == "__main__":
    value = a_function()
    print(value)

```

您会注意到，在 Python 中，decorator 以符号 **@** 开始，后面是函数名，我们将用它来“装饰”我们的正则表达式。要应用装饰器，只需将它放在函数定义之前的行上。现在当我们调用 **a_function** 时，它将被修饰，我们将得到如下结果:

```py

The result of 1+1 is 2

```

让我们创建一个真正有用的装饰器。

* * *

### 创建日志装饰器

有时你会想要创建一个函数正在做什么的日志。大多数时候，您可能会在函数本身中进行日志记录。有时，您可能希望在函数级别进行，以了解程序的流程，或者满足一些业务规则，如审计。这里有一个小装饰器，我们可以用它来记录任何函数的名称和它返回的内容:

```py

import logging

#----------------------------------------------------------------------
def log(func):
    """
    Log what function is called
    """
    def wrap_log(*args, **kwargs):
        name = func.__name__
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # add file handler
        fh = logging.FileHandler("%s.log" % name)
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.info("Running function: %s" % name)
        result = func(*args, **kwargs)
        logger.info("Result: %s" % result)
        return func
    return wrap_log

#----------------------------------------------------------------------
@log
def double_function(a):
    """
    Double the input parameter
    """
    return a*2

#----------------------------------------------------------------------
if __name__ == "__main__":
    value = double_function(2)

```

这个小脚本有一个接受函数作为唯一参数的 **log** 函数。它将创建一个 logger 对象和一个基于函数名的日志文件名。然后，日志函数将记录调用了什么函数以及函数返回了什么(如果有的话)。

* * *

### 内置装饰器

Python 附带了几个内置的装饰器。三大因素是:

*   @classmethod
*   @静态方法
*   @属性

Python 的标准库的各个部分也有 decorators。一个例子就是**functools . wrapps**。不过，我们将把我们的范围限制在以上三个方面。

* * *

### @classmethod 和@staticmethod

我自己从来没有用过这些，所以我做了一些调查。调用 **@classmethod** decorator 时，可以使用类的实例，或者直接由类本身作为第一个参数。根据 Python [文档](http://docs.python.org/2/library/functions.html#classmethod) : *它既可以在类上调用(比如 C.f())，也可以在实例上调用(比如 C())。f())。该实例被忽略，但其类除外。如果为派生类调用类方法，派生类对象将作为隐含的第一个参数传递。我在研究中发现,@classmethod 装饰器的主要用例是作为初始化的替代构造函数或帮助方法。*

**@staticmethod** decorator 只是一个类内部的函数。无论有没有实例化该类，都可以调用它。一个典型的用例是当你有一个函数，你认为它和一个类有联系。这在很大程度上是一种风格选择。

看看这两个装饰器如何工作的代码示例可能会有所帮助:

```py

########################################################################
class DecoratorTest(object):
    """
    Test regular method vs @classmethod vs @staticmethod
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        pass

    #----------------------------------------------------------------------
    def doubler(self, x):
        """"""
        print("running doubler")
        return x*2

    #----------------------------------------------------------------------
    @classmethod
    def class_tripler(klass, x):
        """"""
        print("running tripler: %s" % klass)
        return x*3

    #----------------------------------------------------------------------
    @staticmethod
    def static_quad(x):
        """"""
        print("running quad")
        return x*4

#----------------------------------------------------------------------
if __name__ == "__main__":
    decor = DecoratorTest()
    print(decor.doubler(5))
    print(decor.class_tripler(3))
    print(DecoratorTest.class_tripler(3))
    print(DecoratorTest.static_quad(2))
    print(decor.static_quad(3))

    print(decor.doubler)
    print(decor.class_tripler)
    print(decor.static_quad)

```

这个例子演示了你可以用同样的方式调用一个常规方法和两个修饰方法。您会注意到，可以从类或从类的实例中直接调用@classmethod 和@staticmethod 修饰函数。如果你试图用这个类(比如 DecoratorTest.doubler(2))调用一个常规函数，你会收到一个 **TypeError** 。您还会注意到，最后一条 print 语句显示 decor.static_quad 返回一个常规函数，而不是一个绑定方法。

* * *

### Python 的属性

我今年已经写过一次关于 **@property** decorator 的文章，所以我将在这里转载那篇[文章](https://www.blog.pythonlibrary.org/2014/01/20/python-201-properties/)的一个微小变化。

Python 有一个被称为**属性**的简洁概念，它可以做几件有用的事情。我们将研究如何做到以下几点:

*   将类方法转换为只读属性
*   将设置器和获取器重新实现到属性中

使用属性的一个最简单的方法是将它用作方法的装饰。这允许您将类方法转换为类属性。当我需要对值进行某种组合时，我发现这很有用。其他人发现它对于编写他们希望作为方法访问的转换方法很有用。让我们看一个简单的例子:

```py

########################################################################
class Person(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, first_name, last_name):
        """Constructor"""
        self.first_name = first_name
        self.last_name = last_name

    #----------------------------------------------------------------------
    @property
    def full_name(self):
        """
        Return the full name
        """
        return "%s %s" % (self.first_name, self.last_name)

```

在上面的代码中，我们创建了两个类属性: **self.first_name** 和 **self.last_name** 。接下来，我们创建一个 **full_name** 方法，它附带了一个 **@property** decorator。这允许我们在解释器会话中执行以下操作:

```py

>>> person = Person("Mike", "Driscoll")
>>> person.full_name
'Mike Driscoll'
>>> person.first_name
'Mike'
>>> person.full_name = "Jackalope"
Traceback (most recent call last):
  File "", line 1, in <fragment>AttributeError: can't set attribute
```

如您所见，因为我们将方法转换成了属性，所以可以使用普通的点符号来访问它。但是，如果我们试图将属性设置为不同的值，我们将会引发一个 **AttributeError** 。更改**全名**属性的唯一方法是间接更改:

```py

>>> person.first_name = "Dan"
>>> person.full_name
'Dan Driscoll'

```

这是一种限制，所以让我们看看另一个例子，我们可以创建一个属性，让**允许我们设置它。**

### 用 Python 属性替换 Setters 和 Getters

让我们假设我们有一些遗留代码，是某个不太了解 Python 的人写的。如果您和我一样，您以前已经见过这种代码:

```py

from decimal import Decimal

########################################################################
class Fees(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self._fee = None

    #----------------------------------------------------------------------
    def get_fee(self):
        """
        Return the current fee
        """
        return self._fee

    #----------------------------------------------------------------------
    def set_fee(self, value):
        """
        Set the fee
        """
        if isinstance(value, str):
            self._fee = Decimal(value)
        elif isinstance(value, Decimal):
            self._fee = value

```

要使用这个类，我们必须使用如下定义的 setters 和 getters:

```py

>>> f = Fees()
>>> f.set_fee("1")
>>> f.get_fee()
Decimal('1')

```

如果您想在不中断依赖这段代码的所有应用程序的情况下，将属性的普通点标记访问添加到这段代码中，您可以通过添加一个属性非常简单地更改它:

```py

from decimal import Decimal

########################################################################
class Fees(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self._fee = None

    #----------------------------------------------------------------------
    def get_fee(self):
        """
        Return the current fee
        """
        return self._fee

    #----------------------------------------------------------------------
    def set_fee(self, value):
        """
        Set the fee
        """
        if isinstance(value, str):
            self._fee = Decimal(value)
        elif isinstance(value, Decimal):
            self._fee = value

    fee = property(get_fee, set_fee)

```

我们在这段代码的末尾添加了一行。现在我们可以这样做:

```py

>>> f = Fees()
>>> f.set_fee("1")
>>> f.fee
Decimal('1')
>>> f.fee = "2"
>>> f.get_fee()
Decimal('2')

```

如您所见，当我们以这种方式使用**属性**时，它允许 fee 属性在不破坏遗留代码的情况下自己设置和获取值。让我们使用属性装饰器重写这段代码，看看我们是否能让它允许设置。

```py

from decimal import Decimal

########################################################################
class Fees(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self._fee = None

    #----------------------------------------------------------------------
    @property
    def fee(self):
        """
        The fee property - the getter
        """
        return self._fee

    #----------------------------------------------------------------------
    @fee.setter
    def fee(self, value):
        """
        The setter of the fee property
        """
        if isinstance(value, str):
            self._fee = Decimal(value)
        elif isinstance(value, Decimal):
            self._fee = value

#----------------------------------------------------------------------
if __name__ == "__main__":
    f = Fees()

```

上面的代码演示了如何为**费用**属性创建一个“setter”。你可以通过用一个叫做 **@fee.setter** 的装饰器来装饰第二个方法，也叫做 **fee** 来做到这一点。当您执行以下操作时，将调用 setter:

```py

>>> f = Fees()
>>> f.fee = "1"

```

如果你看看**属性**的签名，它有 fget、fset、fdel 和 doc 作为“参数”。如果您想捕捉针对属性的 **del** 命令，您可以使用 **@fee.deleter** 创建另一个使用相同名称的修饰方法来对应删除函数。

* * *

### 包扎

至此，您应该知道如何创建自己的装饰器，以及如何使用 Python 的一些内置装饰器。我们看了@classmethod、@property 和@staticmethod。我很想知道我的读者如何使用内置装饰器，以及他们如何使用自己的定制装饰器。

* * *

### 进一步阅读

*   关于 [@classmethod](http://docs.python.org/2/library/functions.html#classmethod) 的 Python 文档
*   关于 [@staticmethod](http://docs.python.org/2/library/functions.html#staticmethod) 的 Python 文档
*   [通过 12 个简单的步骤了解 Python Decorators！](http://simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/)
*   Python 元编程:[一个简短的装饰解释](http://dietbuddha.blogspot.com/2014/01/python-metaprogramming-decorators.html)
*   stack overflow:[Python 中的@staticmethod 和@classmethod 有什么区别？](http://stackoverflow.com/q/136097/393194)
*   [Python 中的 getter 和 setter](http://eli.thegreenplace.net/2009/02/06/getters-and-setters-in-python/)
*   关于遗产的官方 Python [文档](http://docs.python.org/release/2.6/library/functions.html#property)
*   关于在 [StackOverflow](https://stackoverflow.com/questions/16025462/what-is-the-right-way-to-put-a-docstring-on-python-property) 上向 Python 属性添加文档字符串的讨论