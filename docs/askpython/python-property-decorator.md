# 如何使用 Python 属性装饰器？

> 原文：<https://www.askpython.com/python/built-in-methods/python-property-decorator>

又见面了！在本文中，我们将看看 Python property decorator。

Python 有一个非常有用的特性，叫做 decorators，它只是函数包装器的语法糖。这里，我们将关注属性装饰器，这是一种特殊类型的装饰器。

这个主题可能会让你有点困惑，所以我们将使用说明性的例子一步一步地介绍它。我们开始吧！

* * *

## Python 属性装饰器基于什么？

属性装饰器基于内置的`property()`函数。这个函数返回一个特殊的`property`对象。

您可以在您的 Python 解释器中调用它并查看一下:

```py
>>> property()
<property object at 0x000002BBD7302BD8>

```

这个`property`对象有一些额外的方法，用于获取和设置对象的值。它也有删除它的方法。

方法列表如下:

*   `property().getter`
*   `property().setter`
*   `property().deleter`

但这还不止于此！这些方法也可以用在其他对象上，它们本身也可以充当装饰者！

所以，举个例子，我们可以用`property().getter(obj)`，它会给我们另一个属性对象！

所以，需要注意的是属性装饰器将使用这个函数，它有一些特殊的方法来读写对象。但是这对我们有什么帮助呢？

现在让我们来看看。

* * *

## 使用 Python 属性装饰器

要使用属性装饰器，我们需要将它包装在任何函数/方法周围。

这里有一个简单的例子:

```py
@property
def fun(a, b):
    return a + b

```

这与以下内容相同:

```py
def fun(a, b):
    return a + b

fun = property(fun)

```

所以在这里，我们用`fun()`包装`property()`，这正是装饰者所做的！

现在让我们举一个简单的例子，在一个类方法上使用属性装饰器。

考虑下面的类，没有任何修饰的方法:

```py
class Student():
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.full_id = self.name + " - " + str(self.id)

    def get_name(self):
        return self.name

s = Student("Amit", 10)
print(s.name)
print(s.full_id)

# Change only the name
s.name = "Rahul"
print(s.name)
print(s.full_id)

```

**输出**

```py
Amit
Amit - 10
Rahul
Amit - 10

```

这里，正如你所看到的，当我们只改变对象的`name`属性时，它对`full_id`属性的引用仍然没有更新！

为了确保每当`name`或`id`更新时`full_id`属性也会更新，一个解决方案是将`full_id`变成一个方法。

```py
class Student():
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def get_name(self):
        return self.name

    # Convert full_id into a method
    def full_id(self):
        return self.name + " - " + str(self.id)

s = Student("Amit", 10)
print(s.name)
# Method call
print(s.full_id())

s.name = "Rahul"
print(s.name)
# Method call
print(s.full_id())

```

**输出**

```py
Amit
Amit - 10
Rahul
Rahul - 10

```

这里，我们已经通过将`full_id`转换成方法`full_id()`解决了我们的问题。

然而，这并不是解决这个问题的最好方法，因为您可能需要将所有这样的属性转换成一个方法，并将这些属性转换成方法调用。这不方便！

为了减少我们的痛苦，我们可以使用`@property`装饰器来代替！

想法是把`full_id()`做成一个方法，但是用`@property`把它封装起来。这样，我们将能够更新`full_id`，而不必将其视为函数调用。

我们可以直接这样做:`s.full_id`。注意这里没有方法调用。这是因为属性装饰器。

让我们现在就来试试吧！

```py
class Student():
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def get_name(self):
        return self.name

    @property
    def full_id(self):
        return self.name + " - " + str(self.id)

s = Student("Amit", 10)
print(s.name)
# No more method calls!
print(s.full_id)

s.name = "Rahul"
print(s.name)
# No more method calls!
print(s.full_id)

```

**输出**

```py
Amit
Amit - 10
Rahul
Rahul - 10

```

事实上，这种方法现在奏效了！现在，我们不需要使用括号调用`full_id`。

虽然它仍然是一个方法，但属性装饰器屏蔽了它，并将其视为[类](https://www.askpython.com/python/oops/python-classes-objects)的属性！现在名字没意义了吧！？

## 通过 setter 使用属性

在上面的例子中，这种方法是可行的，因为我们没有直接显式地修改`full_id`属性。默认情况下，使用`@property`会使该属性只读。

这意味着您不能显式更改该属性。

```py
s.full_id = "Kishore"
print(s.full_id)

```

**输出**

```py
---> 21 s.full_id = "Kishore"
     22 print(s.full_id)

AttributeError: can't set attribute

```

显然，我们没有权限，因为该属性是只读的！

要让属性可写，还记得我们讲过的`property().setter`方法吗，也是 decorator？

原来我们可以使用`@full_id.setter`添加另一个`full_id`属性，使其可写。`@full_id.setter`将继承原来`full_id`的所有财产，所以我们可以直接添加！

但是，我们不能在 setter 属性中直接使用`full_id`属性。请注意，这将导致无限递归下降！

```py
class Student():
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def get_name(self):
        return self.name

    @property
    def full_id(self):
        return self.name + " - " + str(self.id)

    @full_id.setter
    def full_id(self, value):
        # Infinite recursion depth!
        # Notice that you're calling the setter property of full_id() again and again
        self.full_id = value

```

为了避免这种情况，我们将在我们的类中添加一个隐藏属性`_full_id`。我们将修改`@property`装饰器来返回这个属性。

更新后的代码将如下所示:

```py
class Student():
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self._full_id = self.name + " - " + str(self.id)

    def get_name(self):
        return self.name

    @property
    def full_id(self):
        return self._full_id

    @full_id.setter
    def full_id(self, value):
        self._full_id = value

s = Student("Amit", 10)
print(s.name)
print(s.full_id)

s.name = "Rahul"
print(s.name)
print(s.full_id)

s.full_id = "Kishore - 20"
print(s.name)
print(s.id)
print(s.full_id)

```

**输出**

```py
Amit
Amit - 10
Rahul
Amit - 10
Rahul
10
Kishore - 20

```

我们已经成功地让`full_id`属性拥有了 getter 和 setter 属性！

* * *

## 结论

希望这能让您更好地理解属性装饰器，以及为什么您可能需要使用类属性，比如`_full_id`。

这些隐藏的类属性(像`_full_id`)让我们很容易使用外部的`full_id`属性！

这正是现代开源项目中大量使用属性的原因。

它们让最终用户非常容易，也让开发人员很容易将隐藏属性与非隐藏属性分离开来！

## 参考

*   [StackOverflow 关于物业装修的问题](https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work-in-python)

* * *