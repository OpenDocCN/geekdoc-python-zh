# Python 201:属性

> 原文：<https://www.blog.pythonlibrary.org/2014/01/20/python-201-properties/>

Python 有一个被称为**属性**的简洁概念，它可以做几件有用的事情。在本文中，我们将探讨如何做到以下几点:

*   将类方法转换为只读属性
*   将设置器和获取器重新实现到属性中

在本文中，您将学习如何以几种不同的方式使用内置类**属性**。希望在文章结束时，您会看到它是多么有用。

### 入门指南

使用属性的一个最简单的方法是将它用作方法的装饰。这允许您将类方法转换为类属性。当我需要对值进行某种组合时，我发现这很有用。其他人发现它对于编写他们希望作为方法访问的转换方法很有用。让我们看一个简单的例子:

```py
class Person(object):

    def __init__(self, first_name, last_name):
        """Constructor"""
        self.first_name = first_name
        self.last_name = last_name

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
  File "", line 1, in 
AttributeError: can't set attribute

```

如您所见，因为我们将方法转换成了属性，所以可以使用普通的点符号来访问它。但是，如果我们试图将属性设置为不同的值，我们将会引发一个 **AttributeError** 。更改**全名**属性的唯一方法是间接更改:

```py
>>> person.first_name = "Dan"
>>> person.full_name
'Dan Driscoll'

```

这是一种限制，所以让我们看看另一个例子，我们可以创建一个属性，让**允许我们设置它。**

### 用 Python 属性替换 Setters 和 Getters

让我们假设我们有一些遗留代码，是某个不太了解 Python 的人写的。如果你像我一样，你以前已经见过这种代码:

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

### 包扎

现在您知道了如何在自己的类中使用 Python 属性。希望您能找到在自己的代码中使用它们的更有用的方法。

### 附加阅读

*   [Python 中的 getter 和 setter](http://eli.thegreenplace.net/2009/02/06/getters-and-setters-in-python/)
*   关于遗产的官方 Python [文档](http://docs.python.org/release/2.6/library/functions.html#property)
*   关于在 [StackOverflow](https://stackoverflow.com/questions/16025462/what-is-the-right-way-to-put-a-docstring-on-python-property) 上向 Python 属性添加文档字符串的讨论