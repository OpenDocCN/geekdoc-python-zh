# Python 类介绍(第 2 部分，共 2 部分)

> 原文：<https://www.pythoncentral.io/introduction-to-python-classes-part-2-of-2/>

在本系列的第一部分中，我们看了在 Python 中使用类的基础知识。现在我们来看看一些更高级的主题。

## **Python 类继承**

Python 类支持*继承*，这让我们获得一个类定义并扩展它。让我们创建一个继承(或*从[第一部分](https://www.pythoncentral.io/introduction-to-python-classes/)中的例子派生出*)的新类:

```py

class Foo:

def __init__(self, val):

self.val = val

def printVal(self):

print(self.val)
类 derived Foo(Foo):
def negate val(self):
self . val =-self . val

```

这定义了一个名为`DerivedFoo`的类，它拥有`Foo`类所拥有的一切，还添加了一个名为`negateVal`的新方法。这就是它的作用:

```py

>>> obj = DerivedFoo(42)

>>> obj.printVal()

42

>>> obj.negateVal()

>>> obj.printVal()

-42

```

当我们重新定义(或*覆盖*)一个已经在基类中定义的方法时，继承变得非常有用:

```py

class DerivedFoo2(Foo):

def printVal(self):

print('My value is %s' % self.val)

```

我们可以按如下方式测试该类:

```py

>>> obj2 = DerivedFoo2(42)

>>> obj2.printVal()

My value is 42

```

派生类重新定义了`printVal`方法来做一些不同的事情，每当调用`printVal`时都会使用这个新版本。这让我们可以改变类的行为，这通常是我们想要的(因为如果我们想要原始的行为，我们只需要使用原始的类)。注意，该方法的新版本调用旧版本，并且调用以基类的名称为前缀(否则 Python 会认为您调用的是新版本)。

Python 提供了几个函数来帮助您确定一个对象是什么类:

*   检查一个对象是否是指定类的一个实例，或者是一个派生类。

例如以下内容:

```py

>>> print(isinstance(obj, Foo))

True

>>> print(isinstance(obj, DerivedFoo))

True

>>> print(isinstance(obj, DerivedFoo2))

False

```

*   检查一个类是否派生自另一个类

例如以下内容:

```py

>>> print(issubclass(DerivedFoo, Foo))

True

>>> print(issubclass(int, Foo))

False

```

### **Python 类迭代器和生成器**

Python 的`for`语句将循环遍历任何可*迭代的*，包括内置的数据类型，如数组和字典。例如:

```py

>>> arr = [1,2,3]

>>> for x in arr:

...     print(x)

1

2

3

```

当我们定义自己的类时，我们可以使它们可迭代，这将允许它们在 for 循环中工作。我们通过定义一个返回一个*迭代器*(一个跟踪我们在循环中的位置的对象)的`__iter__`方法和一个返回下一个可用值的`__next__`方法来实现这一点。请注意，Python 3.x 和 Python 2.x 的`next`方法的语法是不同的。对于 Python 3.x，您必须使用`__next__`方法，而对于 Python 2.x，您必须使用`next`方法。

这里有一个简单的例子，可以让你在一个数据结构上向后迭代。下面是类的定义:

*   [Python 3.x](#custom-tab-0-python-3-x)
*   [Python 2.x](#custom-tab-0-python-2-x)

*   [Python 3.x](#)

[python]
class Backwards:
def __init__(self, val):
self.val = val
self.pos = len(val)

def __iter__(self):
回归自我

def __next__(self):
#如果 self.pos < = 0:
提升 StopIteration，我们就完成了

self.pos = self.pos - 1
return self.val[self.pos]
[/python]

*   [Python 2.x](#)

[python]
class Backwards:
def __init__(self, val):
self.val = val
self.pos = len(val)

def __iter__(self):
回归自我

def next(self):
#我们完成了
如果 self.pos < = 0:
提高 StopIteration

self.pos = self.pos - 1
return self.val[self.pos]
[/python]

这是一个迭代类的例子:

```py

>>> for x in Backwards([1,2,3]):

...     print(x)

3

2

1

```

该类跟踪两件事，被迭代的数据结构和下一个要返回的值。`__iter__`方法只是返回对对象本身的引用，因为这是用来管理循环的。当 Python 遍历对象时，它会重复调用`next`方法来获取下一个值，直到没有剩余值时抛出`StopIteration`异常。

这是一个非常简单的例子，但是它的大部分是锅炉板代码(获取每一项并跟踪我们在循环中的进度)，每次我们想要创建一个 iterable 类时都是一样的。然而，Python 又一次拯救了我们，它给我们提供了一种方法，使用*生成器*来消除所有这些重复的管理代码。

一个*生成器*是一种特殊的函数，它返回一个 iterable 对象，这个对象自动地记住它在一个循环中的位置。这是同一个例子，这次使用了一个发生器。

该功能可定义如下(注意:使用`yield`关键字):

```py

def backwards(val):

for n in range(len(val), 0, -1):

yield val[n-1]

```

我们可以这样使用发电机:

```py

>>> for x in backwards([1,2,3]):

...     print(x)

3

2

1

```

如果你以前从未见过这种事情，你可能真的很难理解它，但是最简单的方法就是像这样阅读`backwards`函数:

*   在传入的值上向后循环。
*   在每一遍中，`yield`下一个值，即暂时停止执行循环，并将下一个值返回给调用者。它做它想做的任何事情，然后当它再次调用我们时，我们从我们停止的地方继续循环。

### **作为对象的 Python 类**

一个类描述了该类的实例看起来像什么，也就是说，它们将有什么方法和成员变量。在内部，Python 在自己的对象中跟踪每个类的定义，我们可以修改这个对象。这意味着我们可以动态地改变类的定义，甚至在运行时创建一个全新的类！

让我们从一个简单的类定义开始:

```py

class Foo:

def __init__(self, val):

self.val = val

```

让我们来看看用法:

```py

>>> obj = Foo(42)

>>> obj.printVal()

AttributeError: Foo instance has no attribute 'printVal'

```

哎呀！我们得到一个错误，因为这个类没有一个`printVal`方法。

好了，再加一个:-)。我们可以这样定义它:

```py

def printVal(self):

print(self.val)

```

我们可以将函数添加到类中，如下所示:

```py

>>> Foo.printVal = printVal

>>> obj.printVal()

42

```

我们定义了一个名为 **printVal** 的方法，它是独立的(也就是说，它是在类之外定义的)，但是它看起来像一个类方法(也就是说，它有一个 self 参数)。然后，我们将它添加到类定义(`Foo.printVal = printVal`)中，这使得它变得可用，就好像它是原始类定义的一部分一样。

如果我们想删除它，我们可以使用普通的`del`语句:

```py

>>> del Foo.printVal

>>> obj.printVal()

AttributeError: Foo instance has no attribute 'printVal'

```

为了在运行时创建一个全新的类，我们使用了`type`方法:

```py

>>> obj = MyNewClass()

NameError: name 'MyNewClass' is not defined

>>> MyNewClass = type('MyNewClass', (object,), dict())

>>> obj = MyNewClass()

>>> print(obj)

<__main__.MyNewClass object at 0x01D79DCC>

```

`type`调用的第二个参数是我们想要从中派生的类的列表，而第三个参数是组成类定义的方法和成员变量的字典(您可以在这里定义它们，或者如上所述动态添加它们)。

要理解 Python 中的生成器和 yield 关键字，请查看文章 [Python 生成器和 yield 关键字](https://www.pythoncentral.io/python-generators-and-yield-keyword/ "Python Generators and the Yield Keyword")。