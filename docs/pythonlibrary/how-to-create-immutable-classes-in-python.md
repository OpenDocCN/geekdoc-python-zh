# 如何用 Python 创建“不可变”的类

> 原文：<https://www.blog.pythonlibrary.org/2014/01/17/how-to-create-immutable-classes-in-python/>

我最近读了很多关于 Python 的神奇方法，最近还读了一些创建不可变类的方法。不可变类不允许程序员向实例添加属性(即猴子补丁)。如果我们实际上先看一个正常的类，会更容易理解一点。我们将从一个猴子补丁的例子开始，然后看一种使类“不可变”的方法。

### 猴子修补 Python 类

首先，我们需要创建一个可以玩的类。这里有一个简单的类，它不做任何事情:

```py

########################################################################
class Mutable(object):
    """
    A mutable class
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        pass

```

现在让我们创建这个类的一个实例，看看是否可以添加一个属性:

```py

>>> mut_obj = Mutable()
>>> mut_obj.monkey = "tamarin"
>>> mut_obj.monkey
'tamarin'

```

这个类允许我们在运行时向它添加属性。现在我们知道了如何做一些简单的猴子补丁，让我们尝试阻止这种行为。

### 创建不可变的类

我读到的一个关于不可变类的例子提到，你可以通过用**_ _ 插槽 __** 替换一个类的 **__dict__** 来创建一个不可变类。让我们看看这是什么样子:

```py

########################################################################
class Immutable(object):
    """
    An immutable class
    """
    __slots__ = ["one", "two", "three"]

    #----------------------------------------------------------------------
    def __init__(self, one, two, three):
        """Constructor"""
        super(Immutable, self).__setattr__("one", one)
        super(Immutable, self).__setattr__("two", two)
        super(Immutable, self).__setattr__("three", three)

    #----------------------------------------------------------------------
    def __setattr__(self, name, value):
        """"""
        msg = "'%s' has no attribute %s" % (self.__class__,
                                            name)
        raise AttributeError(msg)

```

现在我们只需要创建这个类的一个实例，看看我们是否可以用猴子来修补它:

```py

>>> i = Immutable(1, 2, 3)
>>> i.four = 4
Traceback (most recent call last):
  File "", line 1, in <fragment>AttributeError: 'Immutable' object has no attribute 'four'
```

在这种情况下，该类不允许我们对实例进行猴子修补。相反，我们收到一个 AttibuteError。让我们尝试更改其中一个属性:

```py

>>> i = Immutable(1, 2, 3)
>>> i.one = 2
Traceback (most recent call last):
  File "c:\Users\mdriscoll\Desktop\rep-fonts\immutable\immute_slots.py", line 1, in ########################################################################
  File "c:\Users\mdriscoll\Desktop\rep-fonts\immutable\immute_slots.py", line 33, in __setattr__
    raise AttributeError(msg)
AttributeError: '<class>' has no attribute one
```

这是因为我们已经覆盖了 **__setattr__** 方法。如果你想的话，你可以重写这个方法，什么都不做。这将阻止追溯的发生，但也防止值被更改。如果您喜欢明确说明正在发生的事情，那么提出一个错误可能是一种方法。

如果你读了一些关于插槽的书，你会很快发现不鼓励以这种方式使用插槽。为什么？因为槽主要是作为内存优化而创建的(它减少了属性访问时间)。

您可以通过以下链接了解更多关于插槽的信息:

*   关于插槽的 Python [文档](http://docs.python.org/2/reference/datamodel.html?highlight=__slots__#slots)
*   stack overflow:[python _ _ slots _ _](http://stackoverflow.com/q/472000/393194)
*   什么是插槽？

https://stackoverflow.com/questions/472000/python-slots