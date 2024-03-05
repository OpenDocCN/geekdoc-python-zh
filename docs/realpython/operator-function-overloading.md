# 自定义 Python 类中的运算符和函数重载

> 原文：<https://realpython.com/operator-function-overloading/>

如果你在 Python 中对一个`str`对象使用过`+`或`*`操作符，你一定会注意到它与`int`或`float`对象相比的不同行为:

>>>

```py
>>> # Adds the two numbers
>>> 1 + 2
3

>>> # Concatenates the two strings
>>> 'Real' + 'Python'
'RealPython'

>>> # Gives the product
>>> 3 * 2
6

>>> # Repeats the string
>>> 'Python' * 3
'PythonPythonPython'
```

您可能想知道同一个内置操作符或函数如何为不同类的对象显示不同的行为。这分别称为运算符重载或函数重载。本文将帮助您理解这种机制，以便您可以在自己的 Python 类中做同样的事情，并使您的对象更加 Python 化。

您将了解以下内容:

*   在 Python 中处理操作符和内置的 API
*   `len()`和其他内置背后的“秘密”
*   如何使您的类能够使用运算符
*   如何让你的类与 Python 的内置函数兼容

**免费奖励:** [点击此处获取免费的 Python OOP 备忘单](https://realpython.com/bonus/python-oop/)，它会为你指出最好的教程、视频和书籍，让你了解更多关于 Python 面向对象编程的知识。

另外，您还将看到一个示例类，它的对象将与这些操作符和函数兼容。我们开始吧！

## Python 数据模型

假设您有一个代表在线订单的类，它有一个购物车(一个`list`)和一个客户(一个`str`或代表客户的另一个类的实例)。

**注意:**如果你需要复习 Python 中的 OOP，可以看看这篇关于真正 Python 的教程:[Python 3 中的面向对象编程(OOP)](https://realpython.com/python3-object-oriented-programming/)

在这种情况下，想要获得购物车列表的长度是很自然的。一些 Python 新手可能会决定在他们的类中实现一个名为`get_cart_len()`的方法来完成这项工作。但是您可以配置内置的 [`len()`](https://realpython.com/len-python-function/) ，当给定我们的对象时，它返回购物车列表的长度。

在另一种情况下，我们可能想在购物车中添加一些东西。同样，不熟悉 Python 的人会考虑实现一个名为`append_to_cart()`的方法，该方法获取一个商品并将其添加到购物车列表中。但是您可以配置`+`操作符，让它向购物车添加一个新商品。

Python 使用特殊的方法完成所有这些工作。这些特殊的方法有一个命名约定，其中名称以两个下划线开始，后跟一个标识符，以另一对下划线结束。

本质上，每个内置函数或操作符都有一个与之对应的特殊方法。比如有`__len__(),`对应`len()`，有`__add__()`对应`+`操作员。

默认情况下，大多数内置和操作符都不会处理你的类的对象。您必须在类定义中添加相应的特殊方法，以使您的对象与内置运算符和运算符兼容。

当您这样做时，与其关联的函数或运算符的行为会根据方法中定义的行为而改变。

这正是[数据模型](https://docs.python.org/3/reference/datamodel.html)(Python 文档的第 3 节)帮助您完成的。它列出了所有可用的特殊方法，并为您提供了重载内置函数和运算符的方法，以便您可以在自己的对象上使用它们。

让我们看看这意味着什么。

**有趣的事实:**由于这些方法使用的命名惯例，它们也被称为*邓德方法*，这是评分方法下*T5】ddouble**的简写。有时它们也被称为*特殊方法*或*魔法方法*。不过，我们更喜欢 *dunder 方法*！***

[*Remove ads*](/account/join/)

## 像`len()`和`[]` 这样的内部操作

Python 中的每个类都为内置函数和方法定义了自己的行为。当你把某个类的一个实例传递给一个内置函数，或者在实例上使用一个操作符，实际上相当于调用一个带有相关参数的特殊方法。

如果有一个内置函数`func()`，并且该函数对应的特殊方法是`__func__()`，Python 将对该函数的调用解释为`obj.__func__()`，其中`obj`是对象。在操作符的例子中，如果你有一个操作符`opr`，并且对应的特殊方法是`__opr__()`，Python 将类似`obj1 <opr> obj2`的东西解释为`obj1.__opr__(obj2)`。

因此，当你在一个对象上调用`len()`时，Python 将调用作为`obj.__len__()`来处理。当您在 iterable 上使用`[]`操作符来获取索引处的值时，Python 将其作为`itr.__getitem__(index)`来处理，其中`itr`是 iterable 对象，`index`是您想要获取的索引。

因此，当您在自己的类中定义这些特殊方法时，您会覆盖与它们相关联的函数或操作符的行为，因为在幕后，Python 正在调用您的方法。让我们更好地理解这一点:

>>>

```py
>>> a = 'Real Python'
>>> b = ['Real', 'Python']
>>> len(a)
11
>>> a.__len__()
11
>>> b[0]
'Real'
>>> b.__getitem__(0)
'Real'
```

如您所见，当您使用该函数或其相应的特殊方法时，会得到相同的结果。事实上，当您使用`dir()`获得一个`str`对象的属性和方法列表时，除了在`str`对象上可用的常用方法之外，您还会在列表中看到这些特殊的方法:

>>>

```py
>>> dir(a)
['__add__',
 '__class__',
 '__contains__',
 '__delattr__',
 '__dir__',
 ...,
 '__iter__',
 '__le__',
 '__len__',
 '__lt__',
 ...,
 'swapcase',
 'title',
 'translate',
 'upper',
 'zfill']
```

如果一个内置函数或操作符的行为不是由特殊方法在类中定义的，那么你将得到一个`TypeError`。

那么，如何在你的类中使用特殊的方法呢？

## 重载内置函数

数据模型中定义的许多特殊方法可以用来改变函数的行为，例如`len`、`abs`、`hash`、`divmod`等等。为此，您只需要在您的类中定义相应的特殊方法。让我们看几个例子:

### 使用`len()` 给你的对象一个长度

要改变`len()`的行为，您需要在您的类中定义`__len__()`特殊方法。每当你将你的类的一个对象传递给`len()`，你对`__len__()`的自定义定义将被用来获得结果。让我们为我们在开头谈到的订单类实现`len()`:

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
...     def __len__(self):
...         return len(self.cart)
...
>>> order = Order(['banana', 'apple', 'mango'], 'Real Python')
>>> len(order)
3
```

如您所见，您现在可以使用`len()`直接获得购物车的长度。此外，说“订单长度”比调用类似于`order.get_cart_len()`的东西更直观。你的召唤既有 Pythonic 式的，也更直观。当您没有定义`__len__()`方法，但仍然在您的对象上调用`len()`时，您会得到一个`TypeError`:

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
>>> order = Order(['banana', 'apple', 'mango'], 'Real Python')
>>> len(order)  # Calling len when no __len__
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: object of type 'Order' has no len()
```

但是，当重载`len()`时，您应该记住 Python 要求函数返回一个整数。如果您的方法返回的不是整数，那么您将得到一个`TypeError`。这很可能是为了与以下事实保持一致:通常使用`len()`来获得序列的长度，该长度只能是整数:

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
...     def __len__(self):
...         return float(len(self.cart))  # Return type changed to float
...
>>> order = Order(['banana', 'apple', 'mango'], 'Real Python')
>>> len(order)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'float' object cannot be interpreted as an integer
```

### 使用`abs()` 使您的对象工作

通过在类中定义`__abs__()`特殊方法，您可以为类的实例指定内置的[的行为。对`abs()`的返回值没有限制，当你的类定义中没有这个特殊方法时，你会得到一个`TypeError`。](https://realpython.com/python-absolute-value/#using-the-built-in-abs-function-with-numbers)

在二维空间中表示一个向量的类中，`abs()`可以用来得到向量的长度。让我们来看看它的实际应用:

>>>

```py
>>> class Vector:
...     def __init__(self, x_comp, y_comp):
...         self.x_comp = x_comp
...         self.y_comp = y_comp
...
...     def __abs__(self):
...         return (self.x_comp ** 2 + self.y_comp ** 2) ** 0.5
...
>>> vector = Vector(3, 4)
>>> abs(vector)
5.0
```

说“向量的绝对值”比调用类似`vector.get_mag()`的东西更直观。

[*Remove ads*](/account/join/)

### 使用`str()` 漂亮地打印你的对象

内置的`str()`用于将一个类的实例转换成一个`str`对象，或者更恰当地说，用于获得一个用户友好的对象字符串表示，它可以被普通用户而不是程序员读取。通过在你的类中定义`__str__()`方法，你可以定义当你的对象被传递给`str()`时应该显示的字符串格式。此外，`__str__()`是 Python 在对象上调用 [`print()`](https://realpython.com/python-print/) 时使用的方法。

让我们在`Vector`类中实现它，将`Vector`对象格式化为`xi+yj`。负 y 分量将使用[格式小型语言](https://docs.python.org/3/library/string.html#format-specification-mini-language)进行处理:

>>>

```py
>>> class Vector:
...     def __init__(self, x_comp, y_comp):
...         self.x_comp = x_comp
...         self.y_comp = y_comp
...
...     def __str__(self):
...         # By default, sign of +ve number is not displayed
...         # Using `+`, sign is always displayed
...         return f'{self.x_comp}i{self.y_comp:+}j'
...
>>> vector = Vector(3, 4)
>>> str(vector)
'3i+4j'
>>> print(vector)
3i+4j
```

需要`__str__()`返回一个`str`对象，如果返回类型是非字符串，我们得到一个`TypeError`。

### 使用`repr()` 表示你的对象

内置的`repr()`用于获得对象的可解析字符串表示。如果一个对象是可解析的，这意味着当`repr`与 [`eval()`](https://realpython.com/python-eval-function/) 等函数结合使用时，Python 应该能够从表示中重新创建对象。要定义`repr()`的行为，可以使用`__repr__()`的特殊方法。

这也是 Python 用来在 REPL 会话中显示对象的方法。如果没有定义`__repr__()`方法，您将得到类似于`<__main__.Vector object at 0x...>`试图在 REPL 会话中查看对象的结果。让我们在`Vector`课堂上看看它的作用:

>>>

```py
>>> class Vector:
...     def __init__(self, x_comp, y_comp):
...         self.x_comp = x_comp
...         self.y_comp = y_comp
...
...     def __repr__(self):
...         return f'Vector({self.x_comp}, {self.y_comp})'
...

>>> vector = Vector(3, 4)
>>> repr(vector)
'Vector(3, 4)'

>>> b = eval(repr(vector))
>>> type(b), b.x_comp, b.y_comp
(__main__.Vector, 3, 4)

>>> vector  # Looking at object; __repr__ used
'Vector(3, 4)'
```

**注意:**在没有定义`__str__()`方法的情况下，Python 使用`__repr__()`方法打印对象，以及在调用`str()`时表示对象。如果两种方法都没有，默认为`<__main__.Vector ...>`。但是`__repr__()`是用于在交互会话中显示对象的唯一方法。课堂上没有它会产生`<__main__.Vector ...>`。

此外，虽然`__str__()`和`__repr__()`之间的这种区别是推荐的行为，但许多流行的库忽略了这种区别，并交替使用这两种方法。

这里有一篇关于`__repr__()`和`__str__()`的推荐文章，作者是我们自己的丹·巴德: [Python 字符串转换 101:为什么每个类都需要一个“repr”](https://dbader.org/blog/python-repr-vs-str)。

### 使用`bool()` 使你的对象真假

内置的`bool()`可以用来获取一个对象的真值。要定义它的行为，可以使用`__bool__()`(Python 2 . x 中的`__nonzero__()`)特殊方法。

这里定义的行为将决定一个实例在所有需要获得真值的上下文中的真值，比如在`if`语句中。

例如，对于上面定义的`Order`类，如果购物车列表的长度不为零，则可以认为实例是真的。这可用于检查是否应处理订单:

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
...     def __bool__(self):
...         return len(self.cart) > 0
...
>>> order1 = Order(['banana', 'apple', 'mango'], 'Real Python')
>>> order2 = Order([], 'Python')

>>> bool(order1)
True
>>> bool(order2)
False

>>> for order in [order1, order2]:
...     if order:
...         print(f"{order.customer}'s order is processing...")
...     else:
...         print(f"Empty order for customer {order.customer}")
Real Python's order is processing...
Empty order for customer Python
```

**注意:**当`__bool__()`特殊方法没有在类中实现时，`__len__()`返回的值作为真值，非零值表示`True`，零值表示`False`。如果这两种方法都没有实现，那么该类的所有实例都被认为是`True`。

还有许多重载内置函数的特殊方法。您可以在[文档](https://docs.python.org/3/reference/datamodel.html#basic-customization)中找到它们。讨论了其中一些之后，让我们转到操作符。

## 重载内置运算符

改变操作符的行为就像改变函数的行为一样简单。您在您的类中定义它们对应的特殊方法，操作符根据这些方法中定义的行为工作。

这些与上述特殊方法的不同之处在于，它们需要接受定义中除了`self`之外的另一个参数，通常称为`other`。我们来看几个例子。

[*Remove ads*](/account/join/)

### 使用`+` 添加您的对象

与`+`操作符相对应的特殊方法是`__add__()`方法。添加自定义的`__add__()`定义会改变操作者的行为。建议`__add__()`返回类的新实例，而不是修改调用实例本身。在 Python 中，您会经常看到这种行为:

>>>

```py
>>> a = 'Real'
>>> a + 'Python'  # Gives new str instance
'RealPython'
>>> a  # Values unchanged
'Real'
>>> a = a + 'Python'  # Creates new instance and assigns a to it
>>> a
'RealPython'
```

您可以在上面看到，在一个`str`对象上使用`+`操作符实际上会返回一个新的`str`实例，保持调用实例(`a`)的值不变。要改变它，我们需要显式地将新实例分配给`a`。

让我们使用操作符在`Order`类中实现向购物车添加新商品的功能。我们将遵循推荐的做法，让操作者返回一个新的`Order`实例，该实例包含我们需要的更改，而不是直接对我们的实例进行更改:

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
...     def __add__(self, other):
...         new_cart = self.cart.copy()
...         new_cart.append(other)
...         return Order(new_cart, self.customer)
...
>>> order = Order(['banana', 'apple'], 'Real Python')

>>> (order + 'orange').cart  # New Order instance
['banana', 'apple', 'orange']
>>> order.cart  # Original instance unchanged
['banana', 'apple']

>>> order = order + 'mango'  # Changing the original instance
>>> order.cart
['banana', 'apple', 'mango']
```

类似地，您有`__sub__()`、`__mul__()`和其他定义`-`、`*`等行为的特殊方法。这些方法还应该返回该类的一个新实例。

### 快捷键:`+=`操作符

`+=`操作符是表达式`obj1 = obj1 + obj2`的快捷方式。与之相对应的特殊方法是`__iadd__()`。`__iadd__()`方法应该直接对`self`参数进行修改并返回结果，结果可能是也可能不是`self`。这种行为与`__add__()`截然不同，因为后者创建了一个新对象并返回它，正如你在上面看到的。

粗略地说，任何在两个对象上使用的`+=`都相当于这个:

```py
>>> result = obj1 + obj2
>>> obj1 = result
```

这里，`result`是`__iadd__()`返回的值。第二个赋值由 Python 自动处理，这意味着您不需要像在`obj1 = obj1 + obj2`的情况下那样显式地将`obj1`赋值给结果。

让我们为`Order`类实现这一点，这样就可以使用`+=`将新商品添加到购物车中:

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
...     def __iadd__(self, other):
...         self.cart.append(other)
...         return self
...
>>> order = Order(['banana', 'apple'], 'Real Python')
>>> order += 'mango'
>>> order.cart
['banana', 'apple', 'mango']
```

可以看出，任何更改都是直接对`self`进行的，然后返回。当你返回一个随机值，比如一个字符串或者一个整数，会发生什么？

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
...     def __iadd__(self, other):
...         self.cart.append(other)
...         return 'Hey, I am string!'
...
>>> order = Order(['banana', 'apple'], 'Real Python')
>>> order += 'mango'
>>> order
'Hey, I am string!'
```

即使相关商品被添加到购物车中，`order`的值也变成了`__iadd__()`返回的值。Python 隐式地为您处理了这个任务。如果您在实现中忘记返回某些内容，这可能会导致令人惊讶的行为:

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
...     def __iadd__(self, other):
...         self.cart.append(other)
...
>>> order = Order(['banana', 'apple'], 'Real Python')
>>> order += 'mango'
>>> order  # No output
>>> type(order)
NoneType
```

因为所有 Python 函数(或方法)都隐式返回 [`None`](https://realpython.com/null-in-python/) ，`order`被重新分配给`None`，当`order`被检查时，REPL 会话不显示任何输出。看`order`的类型，你看到现在是`NoneType`。因此，一定要确保在`__iadd__()`的实现中返回一些东西，并且是操作的结果，而不是其他。

类似于`__iadd__()`，你有`__isub__()`、`__imul__()`、`__idiv__()`和其他特殊的方法来定义`-=`、`*=`、`/=`和其他类似的行为。

**注意:**当`__iadd__()`或者它的朋友从你的类定义中消失了，但是你仍然在你的对象上使用它们的操作符，Python 使用`__add__()`和它的朋友来获得操作的结果，并把它分配给调用实例。一般来说，只要`__add__()`和它的朋友正常工作(返回某个操作的结果)，在你的类中不实现`__iadd__()`和它的朋友是安全的。

Python [文档](https://docs.python.org/3.6/reference/datamodel.html?highlight=data%20model#object.__iadd__)对这些方法有很好的解释。另外，看一下[这个](https://docs.python.org/3.6/faq/programming.html#why-does-a-tuple-i-item-raise-an-exception-when-the-addition-works)示例，它展示了在使用[不可变](https://realpython.com/courses/immutability-python/)类型时`+=`和其他类型所涉及的注意事项。

[*Remove ads*](/account/join/)

### 使用`[]` 对对象进行索引和切片

`[]`操作符被称为索引操作符，在 Python 中用于各种上下文，例如获取序列中索引处的值，获取与字典中的键相关联的值，或者通过切片获取序列的一部分。您可以使用`__getitem__()`特殊方法改变它的行为。

让我们配置我们的`Order`类，这样我们就可以直接使用对象并从购物车中获得一个商品:

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
...     def __getitem__(self, key):
...         return self.cart[key]
...
>>> order = Order(['banana', 'apple'], 'Real Python')
>>> order[0]
'banana'
>>> order[-1]
'apple'
```

你会注意到，在上面,`__getitem__()`的参数名不是`index`,而是`key`。这是因为参数主要有三种形式:**一个整数值**，在这种情况下它要么是一个索引要么是一个字典键，**一个字符串值**，在这种情况下它是一个字典键， [**一个切片对象**](https://docs.python.org/3/library/functions.html#slice) ，在这种情况下它将对类使用的序列进行切片。虽然还有其他可能性，但这些是最常遇到的。

由于我们的内部[数据结构](https://realpython.com/python-data-structures/)是一个列表，我们可以使用`[]`操作符对列表进行切片，在这种情况下，`key`参数将是一个切片对象。这是在你的类中有一个`__getitem__()`定义的最大优势之一。只要使用支持切片的数据结构(列表、元组、字符串等)，就可以配置对象来直接对结构进行切片:

>>>

```py
>>> order[1:]
['apple']
>>> order[::-1]
['apple', 'banana']
```

**注意:**有一个类似的`__setitem__()`特殊方法，用来定义`obj[x] = y`的行为。这个方法除了`self`之外还有两个参数，一般称为`key`和`value`，可以用来改变`key`到`value`的值。

### 反向运算符:使你的类在数学上正确

虽然定义`__add__()`、`__sub__()`、`__mul__()`和类似的特殊方法允许您在类实例是左侧操作数时使用运算符，但如果类实例是右侧操作数，运算符将不起作用:

>>>

```py
>>> class Mock:
...     def __init__(self, num):
...         self.num = num
...     def __add__(self, other):
...         return Mock(self.num + other)
...
>>> mock = Mock(5)
>>> mock = mock + 6
>>> mock.num
11

>>> mock = 6 + Mock(5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported operand type(s) for +: 'int' and 'Mock'
```

如果你的类表示一个数学实体，比如一个向量，一个坐标，或者一个复数，应用操作符应该在两种情况下都有效，因为它是一个有效的数学运算。

此外，如果操作符只在实例是左操作数时起作用，那么在许多情况下，我们违反了交换性的基本原则。因此，为了帮助您使您的类在数学上正确，Python 为您提供了**反向特殊方法**，如`__radd__()`、`__rsub__()`、`__rmul__()`等等。

这些函数处理诸如`x + obj`、`x - obj`和`x * obj`之类的调用，其中`x`不是相关类的实例。就像`__add__()`和其他方法一样，这些反向特殊方法应该返回一个带有操作变化的类的新实例，而不是修改调用实例本身。

让我们在`Order`类中配置`__radd__()`,这样它将在购物车的前面添加一些东西。这可用于根据订单优先级组织购物车的情况:

>>>

```py
>>> class Order:
...     def __init__(self, cart, customer):
...         self.cart = list(cart)
...         self.customer = customer
...
...     def __add__(self, other):
...         new_cart = self.cart.copy()
...         new_cart.append(other)
...         return Order(new_cart, self.customer)
...
...     def __radd__(self, other):
...         new_cart = self.cart.copy()
...         new_cart.insert(0, other)
...         return Order(new_cart, self.customer)
...
>>> order = Order(['banana', 'apple'], 'Real Python')

>>> order = order + 'orange'
>>> order.cart
['banana', 'apple', 'orange']

>>> order = 'mango' + order
>>> order.cart
['mango', 'banana', 'apple', 'orange']
```

## 完整的例子

为了把所有这些要点都讲清楚，最好看一个同时实现这些操作符的示例类。

让我们重新发明轮子，实现我们自己的类来表示复数，`CustomComplex`。我们类的对象将支持各种内置函数和运算符，使它们的行为与内置复数类非常相似:

```py
from math import hypot, atan, sin, cos

class CustomComplex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
```

构造函数只处理一种调用，`CustomComplex(a, b)`。它接受位置参数，表示复数的实部和虚部。

让我们在类中定义两个方法，`conjugate()`和`argz()`，它们将分别给出复数的复共轭和自变量:

```py
def conjugate(self):
    return self.__class__(self.real, -self.imag)

def argz(self):
    return atan(self.imag / self.real)
```

**注意:** `__class__`不是一个特殊的方法，而是一个默认存在的类属性。它有一个对类的引用。通过在这里使用它，我们可以获得它，然后以通常的方式调用构造函数。换句话说，这相当于`CustomComplex(real, imag)`。这样做是为了避免在某一天类名改变时重构代码。

接下来，我们配置`abs()`来返回一个复数的模数:

```py
def __abs__(self):
    return hypot(self.real, self.imag)
```

我们将遵循推荐的`__repr__()`和`__str__()`之间的区别，将第一个用于可解析的字符串表示，将第二个用于“漂亮”的表示。

`__repr__()`方法将简单地返回字符串中的`CustomComplex(a, b)`，这样我们就可以调用`eval()`来重新创建对象，而`__str__()`方法将返回括号中的复数，如`(a+bj)`:

```py
def __repr__(self):
    return f"{self.__class__.__name__}({self.real}, {self.imag})"

def __str__(self):
    return f"({self.real}{self.imag:+}j)"
```

数学上，可以把任意两个复数相加，或者把一个实数加到一个复数上。让我们配置`+`操作符，使其对两种情况都有效。

该方法将检查右侧运算符的类型。如果是`int`或`float`，它将只增加实数部分(因为任何实数`a`都等同于`a+0j`)，而如果是另一个复数，它将改变两个部分:

```py
def __add__(self, other):
    if isinstance(other, float) or isinstance(other, int):
        real_part = self.real + other
        imag_part = self.imag

    if isinstance(other, CustomComplex):
        real_part = self.real + other.real
        imag_part = self.imag + other.imag

    return self.__class__(real_part, imag_part)
```

类似地，我们为`-`和`*`定义行为:

```py
def __sub__(self, other):
    if isinstance(other, float) or isinstance(other, int):
        real_part = self.real - other
        imag_part = self.imag

    if isinstance(other, CustomComplex):
        real_part = self.real - other.real
        imag_part = self.imag - other.imag

    return self.__class__(real_part, imag_part)

def __mul__(self, other):
    if isinstance(other, int) or isinstance(other, float):
        real_part = self.real * other
        imag_part = self.imag * other

    if isinstance(other, CustomComplex):
        real_part = (self.real * other.real) - (self.imag * other.imag)
        imag_part = (self.real * other.imag) + (self.imag * other.real)

    return self.__class__(real_part, imag_part)
```

由于加法和乘法都是可交换的，我们可以通过分别调用`__radd__()`和`__rmul__()`中的`__add__()`和`__mul__()`来定义它们的反向运算符。另一方面，需要定义`__rsub__()`的行为，因为减法是不可交换的:

```py
def __radd__(self, other):
    return self.__add__(other)

def __rmul__(self, other):
    return self.__mul__(other)

def __rsub__(self, other):
    # x - y != y - x
    if isinstance(other, float) or isinstance(other, int):
        real_part = other - self.real
        imag_part = -self.imag

    return self.__class__(real_part, imag_part)
```

**注意:**您可能已经注意到，我们没有添加一个构造来处理这里的`CustomComplex`实例。这是因为，在这种情况下，两个操作数都是我们类的实例，并且`__rsub__()`不会负责处理操作。相反，`__sub__()`将被称为。这是一个微妙但重要的细节。

现在，我们来处理两个操作符，`==`和`!=`。它们使用的特殊方法分别是`__eq__()`和`__ne__()`。如果两个复数对应的实部和虚部相等，则称这两个复数相等。当其中任何一个不相等时，就说它们不相等:

```py
def __eq__(self, other):
    # Note: generally, floats should not be compared directly
    # due to floating-point precision
    return (self.real == other.real) and (self.imag == other.imag)

def __ne__(self, other):
    return (self.real != other.real) or (self.imag != other.imag)
```

**注:** [浮点指南](http://floating-point-gui.de/errors/comparison/)是一篇关于比较浮点和浮点精度的文章。它强调了直接比较浮点数所涉及的注意事项，这正是我们正在做的事情。

也可以用一个简单的[公式](http://tutorial.math.lamar.edu/Extras/ComplexPrimer/Roots.aspx)将一个复数提升到任意次方。我们使用`__pow__()`特殊方法为内置的`pow()`和`**`操作符配置行为:

```py
def __pow__(self, other):
    r_raised = abs(self) ** other
    argz_multiplied = self.argz() * other

    real_part = round(r_raised * cos(argz_multiplied))
    imag_part = round(r_raised * sin(argz_multiplied))

    return self.__class__(real_part, imag_part)
```

**注意:**仔细看看方法的定义。我们调用`abs()`来获得复数的模。所以，一旦你为你的类中的一个特定的函数或者操作符定义了特殊的方法，它就可以在同一个类的其他方法中使用。

让我们创建这个类的两个实例，一个具有正虚部，一个具有负虚部:

>>>

```py
>>> a = CustomComplex(1, 2)
>>> b = CustomComplex(3, -4)
```

字符串表示:

>>>

```py
>>> a
CustomComplex(1, 2)
>>> b
CustomComplex(3, -4)
>>> print(a)
(1+2j)
>>> print(b)
(3-4j)
```

使用`eval()`和`repr()`重新创建对象:

>>>

```py
>>> b_copy = eval(repr(b))
>>> type(b_copy), b_copy.real, b_copy.imag
(__main__.CustomComplex, 3, -4)
```

加法、减法和乘法:

>>>

```py
>>> a + b
CustomComplex(4, -2)
>>> a - b
CustomComplex(-2, 6)
>>> a + 5
CustomComplex(6, 2)
>>> 3 - a
CustomComplex(2, -2)
>>> a * 6
CustomComplex(6, 12)
>>> a * (-6)
CustomComplex(-6, -12)
```

平等和不平等检查:

>>>

```py
>>> a == CustomComplex(1, 2)
True
>>> a ==  b
False
>>> a != b
True
>>> a != CustomComplex(1, 2)
False
```

最后，对一个复数求幂:

>>>

```py
>>> a ** 2
CustomComplex(-3, 4)
>>> b ** 5
CustomComplex(-237, 3116)
```

正如您所看到的，我们的自定义类的对象的行为和外观都像内置类的对象，并且非常 Pythonic 化。下面嵌入了该类的完整示例代码。



```py
from math import hypot, atan, sin, cos

class CustomComplex():
    """
 A class to represent a complex number, a+bj.
 Attributes:
 real - int, representing the real part
 imag - int, representing the imaginary part

 Implements the following:

 * Addition with a complex number or a real number using `+`
 * Multiplication with a complex number or a real number using `*`
 * Subtraction of a complex number or a real number using `-`
 * Calculation of absolute value using `abs`
 * Raise complex number to a power using `**`
 * Nice string representation using `__repr__`
 * Nice user-end viewing using `__str__`

 Notes:
 * The constructor has been intentionally kept simple
 * It is configured to support one kind of call:
 CustomComplex(a, b)
 * Error handling was avoided to keep things simple
 """

    def __init__(self, real, imag):
        """
 Initializes a complex number, setting real and imag part
 Arguments:
 real: Number, real part of the complex number
 imag: Number, imaginary part of the complex number
 """
        self.real = real
        self.imag = imag

    def conjugate(self):
        """
 Returns the complex conjugate of a complex number
 Return:
 CustomComplex instance
 """
        return CustomComplex(self.real, -self.imag)

    def argz(self):
        """
 Returns the argument of a complex number
 The argument is given by:
 atan(imag_part/real_part)
 Return:
 float
 """
        return atan(self.imag / self.real)

    def __abs__(self):
        """
 Returns the modulus of a complex number
 Return:
 float
 """
        return hypot(self.real, self.imag)

    def __repr__(self):
        """
 Returns str representation of an instance of the 
 class. Can be used with eval() to get another 
 instance of the class
 Return:
 str
 """
        return f"CustomComplex({self.real}, {self.imag})"

    def __str__(self):
        """
 Returns user-friendly str representation of an instance 
 of the class
 Return:
 str
 """
        return f"({self.real}{self.imag:+}j)"

    def __add__(self, other):
        """
 Returns the addition of a complex number with
 int, float or another complex number
 Return:
 CustomComplex instance
 """
        if isinstance(other, float) or isinstance(other, int):
            real_part = self.real + other
            imag_part = self.imag

        if isinstance(other, CustomComplex):
            real_part = self.real + other.real
            imag_part = self.imag + other.imag

        return CustomComplex(real_part, imag_part)

    def __sub__(self, other):
        """
 Returns the subtration from a complex number of
 int, float or another complex number
 Return:
 CustomComplex instance
 """
        if isinstance(other, float) or isinstance(other, int):
            real_part = self.real - other
            imag_part = self.imag

        if isinstance(other, CustomComplex):
            real_part = self.real - other.real
            imag_part = self.imag - other.imag

        return CustomComplex(real_part, imag_part)

    def __mul__(self, other):
        """
 Returns the multiplication of a complex number with
 int, float or another complex number
 Return:
 CustomComplex instance
 """
        if isinstance(other, int) or isinstance(other, float):
            real_part = self.real * other
            imag_part = self.imag * other

        if isinstance(other, CustomComplex):
            real_part = (self.real * other.real) - (self.imag * other.imag)
            imag_part = (self.real * other.imag) + (self.imag * other.real)

        return CustomComplex(real_part, imag_part)

    def __radd__(self, other):
        """
 Same as __add__; allows 1 + CustomComplex('x+yj')
 x + y == y + x
 """
        pass

    def __rmul__(self, other):
        """
 Same as __mul__; allows 2 * CustomComplex('x+yj')
 x * y == y * x
 """
        pass

    def __rsub__(self, other):
        """
 Returns the subtraction of a complex number from
 int or float
 x - y != y - x
 Subtration of another complex number is not handled by __rsub__
 Instead, __sub__ handles it since both sides are instances of
 this class
 Return:
 CustomComplex instance
 """
        if isinstance(other, float) or isinstance(other, int):
            real_part = other - self.real
            imag_part = -self.imag

        return CustomComplex(real_part, imag_part)

    def __eq__(self, other):
        """
 Checks equality of two complex numbers
 Two complex numbers are equal when:
 * Their real parts are equal AND
 * Their imaginary parts are equal
 Return:
 bool
 """
        # note: comparing floats directly is not a good idea in general
        # due to floating-point precision
        return (self.real == other.real) and (self.imag == other.imag)

    def __ne__(self, other):
        """
 Checks inequality of two complex numbers
 Two complex numbers are unequal when:
 * Their real parts are unequal OR
 * Their imaginary parts are unequal
 Return:
 bool
 """
        return (self.real != other.real) or (self.imag != other.imag)

    def __pow__(self, other):
        """
 Raises a complex number to a power
 Formula:
 z**n = (r**n)*[cos(n*agrz) + sin(n*argz)j], where
 z = complex number
 n = power
 r = absolute value of z
 argz = argument of z
 Return:
 CustomComplex instance
 """
        r_raised = abs(self) ** other
        argz_multiplied = self.argz() * other

        real_part = round(r_raised * cos(argz_multiplied))
        imag_part = round(r_raised * sin(argz_multiplied))

        return CustomComplex(real_part, imag_part)
```

[*Remove ads*](/account/join/)

## 回顾和资源

在本教程中，您了解了 Python 数据模型以及如何使用该数据模型来构建 Python 类。您了解了如何改变内置函数的行为，比如`len()`、`abs()`、`str()`、`bool()`等等。您还了解了如何改变内置操作符的行为，如`+`、`-`、`*`、`**`等等。

**免费奖励:** [点击此处获取免费的 Python OOP 备忘单](https://realpython.com/bonus/python-oop/)，它会为你指出最好的教程、视频和书籍，让你了解更多关于 Python 面向对象编程的知识。

阅读完本文后，您就可以自信地创建利用 Python 最佳惯用特性的类，并使您的对象具有 Python 语言的特性！

有关数据模型、函数和运算符重载的更多信息，请参考以下资源:

*   [第 3.3 节，Python 文档中数据模型部分的特殊方法名](https://docs.python.org/3/reference/datamodel.html#special-method-names)
*   卢西亚诺·拉马尔霍的《流畅的蟒蛇》
*   [Python 的把戏:本书](https://realpython.com/products/python-tricks-book/)*****