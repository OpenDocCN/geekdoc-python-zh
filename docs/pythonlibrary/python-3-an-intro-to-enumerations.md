# Python 3:枚举介绍

> 原文：<https://www.blog.pythonlibrary.org/2018/03/20/python-3-an-intro-to-enumerations/>

Python 在 3.4 版本的标准库中增加了 **enum** 模块。Python [文档](https://docs.python.org/3.6/library/enum.html)描述了这样一个枚举:

> 枚举是绑定到唯一常数值的一组符号名称(成员)。在枚举中，成员可以通过标识进行比较，并且枚举本身可以被迭代。

让我们看看如何创建一个枚举对象:

```py

>>> from enum import Enum
>>> class AnimalEnum(Enum):
        HORSE = 1
        COW = 2
        CHICKEN = 3
        DOG = 4
>>> print(AnimalEnum.CHICKEN)
AnimalEnum.CHICKEN
>>> print(repr(AnimalEnum.CHICKEN))

```

这里我们创建一个名为 **AnimalEnum** 的枚举类。在类内部，我们创建名为*枚举成员*的类属性，它们是常量。当您试图打印出一个枚举成员时，您将得到相同的字符串。但是如果您打印出枚举成员的 *repr* ，您将获得枚举成员及其值。

如果你试图修改一个枚举成员，Python 会抛出一个 **AttributeError** :

```py

>>> AnimalEnum.CHICKEN = 5
Traceback (most recent call last):
  Python Shell, prompt 5, line 1
  File "C:\Users\mike\AppData\Local\Programs\PYTHON\PYTHON36-32\Lib\enum.py", line 361, in __setattr__
    raise AttributeError('Cannot reassign members.')
builtins.AttributeError: Cannot reassign members.

```

枚举成员具有一些属性，您可以使用这些属性来获取它们的名称和值:

```py

>>> AnimalEnum.CHICKEN.name
'CHICKEN'
>>> AnimalEnum.CHICKEN.value
3

```

枚举也支持迭代。所以你可以做一些有趣的事情，比如:

```py

>>> for animal in AnimalEnum:
        print('Name: {}  Value: {}'.format(animal, animal.value))

Name: AnimalEnum.HORSE  Value: 1
Name: AnimalEnum.COW  Value: 2
Name: AnimalEnum.CHICKEN  Value: 3
Name: AnimalEnum.DOG  Value: 4

```

Python 的枚举不允许创建同名的枚举成员:

```py

>>> class Shapes(Enum):
...     CIRCLE = 1
...     SQUARE = 2
...     SQUARE = 3
... 
Traceback (most recent call last):
  Python Shell, prompt 13, line 1
  Python Shell, prompt 13, line 4
  File "C:\Users\mike\AppData\Local\Programs\PYTHON\PYTHON36-32\Lib\enum.py", line 92, in __setitem__
    raise TypeError('Attempted to reuse key: %r' % key)
builtins.TypeError: Attempted to reuse key: 'SQUARE'

```

如您所见，当您试图重用一个枚举成员名时，它将引发一个 **TypeError** 。

您也可以像这样创建一个枚举:

```py

>>> AnimalEnum = Enum('Animal', 'HORSE COW CHICKEN DOG')
>>> AnimalEnum
 >>> AnimalEnum.CHICKEN
 <animal.chicken:></animal.chicken:> 
```

我个人认为这真的很棒！

* * *

### 枚举成员访问

有趣的是，有多种方法可以访问枚举成员。例如，如果您不知道哪个枚举是哪个，您可以直接调用该枚举并向其传递一个值:

```py

>>> AnimalEnum(2)

```

如果您碰巧传入了一个无效值，那么 Python 会抛出一个 **ValueError**

```py

>>> AnimalEnum(8)
Traceback (most recent call last):
  Python Shell, prompt 11, line 1
  File "C:\Users\mike\AppData\Local\Programs\PYTHON\PYTHON36-32\Lib\enum.py", line 291, in __call__
    return cls.__new__(cls, value)
  File "C:\Users\mike\AppData\Local\Programs\PYTHON\PYTHON36-32\Lib\enum.py", line 533, in __new__
    return cls._missing_(value)
  File "C:\Users\mike\AppData\Local\Programs\PYTHON\PYTHON36-32\Lib\enum.py", line 546, in _missing_
    raise ValueError("%r is not a valid %s" % (value, cls.__name__))
builtins.ValueError: 8 is not a valid AnimalEnum

```

您也可以通过名称访问枚举:

```py

>>> AnimalEnum['CHICKEN']

```

* * *

### 枚举细节

enum 模块还有一些其他有趣的东西可以导入。例如，您可以为您的枚举创建自动值:

```py

>>> from enum import auto, Enum
>>> class Shapes(Enum):
        CIRCLE = auto()
        SQUARE = auto()
        OVAL = auto() 
>>> Shapes.CIRCLE

```

您还可以导入一个方便的枚举装饰器来确保您的枚举成员是唯一的:

```py

>>> @unique
    class Shapes(Enum):
        CIRCLE = 1
        SQUARE = 2
        TRIANGLE = 1

Traceback (most recent call last):
  Python Shell, prompt 18, line 2
  File "C:\Users\mike\AppData\Local\Programs\PYTHON\PYTHON36-32\Lib\enum.py", line 830, in unique
    (enumeration, alias_details))
builtins.ValueError: duplicate values found in : TRIANGLE -> CIRCLE 
```

这里我们创建一个枚举，它有两个成员试图映射到同一个值。因为我们添加了 **@unique** 装饰器，所以如果枚举成员中有任何重复值，就会引发 ValueError。如果您没有应用 **@unique** 装饰器，那么您可以拥有具有相同值的枚举成员。

* * *

### 包扎

虽然我不认为 enum 模块对 Python 来说是真正必要的，但它是一个很好的工具。该文档有更多的例子，并演示了其他类型的枚举，因此绝对值得一读。

* * *

### 进一步阅读

*   关于[枚举模块](https://docs.python.org/3.4/library/enum.html#module-enum) 的 Python 3 文档
*   如何用 Python 表示一个枚举？