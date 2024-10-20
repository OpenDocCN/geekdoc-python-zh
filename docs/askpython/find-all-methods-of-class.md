# 如何在 Python 中找到给定类的所有方法？

> 原文：<https://www.askpython.com/python/examples/find-all-methods-of-class>

大家好！在今天的文章中，我们将看看如何找到一个给定类的所有方法。

往往直接列出一个类的所有方法是非常方便的，这样我们就可以基于某些方法进行一些预处理。

我们开始吧！我们将向您展示实现这一点的一些方法，您可以使用以下任何一种方法。

* * *

## 定义我们的模板类

让我们首先定义一个虚拟类，从中我们可以验证我们的输出。

考虑下面的类，它有一些方法:

```py
class MyClass(object):
    def __init__(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = a

    def add(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state + a
        return self.state

    def subtract(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state - a
        return self.state

    def multiply(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state * a
        return self.state

    def divide(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state / a
        return self.state

```

该类存储了一个浮点属性`state`，我们可以使用算术运算来操作它。

现在让我们看看列出自定义类的方法的一些方式。

* * *

## 方法 1–使用 dir()函数列出类中的方法

要列出这个类的方法，一种方法是使用 Python 中的 [dir()函数](https://www.askpython.com/python/built-in-methods/python-dir-method)。

`dir()`函数将返回该类的所有函数和属性。

让我们看看如果我们为`MyClass`尝试会发生什么。

```py
print(dir(MyClass))

```

**输出**

```py
['__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 'add',
 'divide',
 'multiply',
 'subtract']

```

好了，我们可以看到已经列出了我们的`add`、`divide`、`subtract`和`multiply`方法！但是，其他的方法呢？

嗯，这些方法(以双下划线开头的方法)被称为 **dunder 方法**。

这些通常由包装函数调用。例如，`dict()`函数调用`__dict__()`方法。

### 从输出中过滤数据方法

通常，我们不需要带双下划线前缀的方法，所以我们可以使用下面的代码片段来过滤它们:

```py
method_list = [method for method in dir(MyClass) if method.startswith('__') is False]
print(method_list)

```

**输出**

```py
['add', 'divide', 'multiply', 'subtract']

```

哇！我们现在只得到我们想要的算术方法！

然而，我们目前的解决方案有一个问题。

还记得`dir()`调用一个类的方法和属性吗？

### 处理类的属性

如果我们在一个类中有属性，它也会列出来。考虑下面的例子。

```py
class MyClass(object):

    # MyClass property
    property1 = [1, 2, 3]

    def __init__(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = a

    def add(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state + a
        return self.state

    def subtract(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state - a
        return self.state

    def multiply(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state * a
        return self.state

    def divide(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state / a
        return self.state

    @staticmethod
    def global_method(a, b):
        return a + b

    @classmethod
    def myclass_method(cls):
        return cls

method_list = [method for method in dir(MyClass) if method.startswith('_') is False]
print(method_list)

```

现在，你认为输出会是什么？

**输出**

```py
['add', 'divide', 'global_method', 'multiply', 'myclass_method', 'property1', 'subtract']

```

这也给了我们`property1`，这不是我们想要的。

我们需要再做一个过滤器来区分方法和属性。

但这真的很简单。主要区别是任何属性对象都是**不**可调用的，而方法是可以调用的！

在 Python 中，我们可以使用布尔函数`callable(attribute)`来检查属性是否可以被调用。

现在让我们将它包含到我们的旧代码中。

```py
method_list = [attribute for attribute in dir(MyClass) if callable(getattr(MyClass, attribute)) and attribute.startswith('__') is False]
print(method_list)

```

让我们在不理解列表的情况下进行分解:

```py
method_list = []

# attribute is a string representing the attribute name
for attribute in dir(MyClass):
    # Get the attribute value
    attribute_value = getattr(MyClass, attribute)
    # Check that it is callable
    if callable(attribute_value):
        # Filter all dunder (__ prefix) methods
        if attribute.startswith('__') == False:
            method_list.append(attribute)

print(method_list)

```

我们还将`method`改为`attribute`，这样就消除了误导的意图！

现在让我们来测试一下。

**输出**

```py
['add', 'divide', 'global_method', 'multiply', 'myclass_method', 'subtract']

```

事实上，我们确实得到了方法列表，而没有属性！

## 方法 2–使用 optparse。OptionParser

现在，如果你不太习惯使用`dir()`，这是你可以使用的另一种方法。

我们可以使用`inspect`模块来列出方法。

也就是说，我们可以使用`inspect.getmembers(instance, predicate=inspect.ismethod)`来获得方法列表。

这将自动为您完成工作，您只需要处理输出。让我们看一个例子。

```py
import inspect

class MyClass(object):

    # MyClass property
    property1 = [1, 2, 3]

    def __init__(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = a

    def add(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state + a
        return self.state

    def subtract(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state - a
        return self.state

    def multiply(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state * a
        return self.state

    def divide(self, a):
        assert isinstance(a, float) or isinstance(a, int)
        self.state = self.state / a
        return self.state

    @staticmethod
    def global_method(a, b):
        return a + b

    @classmethod
    def myclass_method(cls):
        return cls

# Create our instance
instance = MyClass(100)

# Get the list of functions
method_list = inspect.getmembers(MyClass, predicate=inspect.ismethod)

print(method_list)

```

**输出**

```py
[('__init__',
  <bound method MyClass.__init__ of <__main__.MyClass object at 0x000001E55E36F390>>),
 ('add',
  <bound method MyClass.add of <__main__.MyClass object at 0x000001E55E36F390>>),
 ('divide',
  <bound method MyClass.divide of <__main__.MyClass object at 0x000001E55E36F390>>),
 ('multiply',
  <bound method MyClass.multiply of <__main__.MyClass object at 0x000001E55E36F390>>),
 ('myclass_method',
  <bound method MyClass.myclass_method of <class '__main__.MyClass'>>),
 ('subtract',
  <bound method MyClass.subtract of <__main__.MyClass object at 0x000001E55E36F390>>)]

```

我们可以得到每个元组的第一个元素，从而得到方法名。

* * *

## 使用检查模块的注意事项

注意，我们得到了一个元组列表。元组的第一个元素是函数的名称，第二个元素表示方法对象本身。

虽然这似乎是一个好的解决方案，但您可能会注意到一些事情。

*   对于`dir()`，我们直接使用了类名本身。但是在这里，我们需要传递一个实例。
*   列表中也没有显示 staticmethods。根据您的使用情况，您可能需要/可能不需要它。

鉴于以上几点，我建议你保持简单，使用`dir()`功能！

* * *

## 结论

在本文中，我们看到了如何在 Python 中列出给定类的所有方法。

## 参考

*   关于列出一个类的所有方法

* * *