# Python 中 __str__ 和 __repr__ 有什么区别

> 原文：<https://www.pythoncentral.io/what-is-the-difference-between-str-and-repr-in-python/>

## Python 中 __str__ 和 __repr__ 的用途

在我们深入讨论之前，让我们来看看 Python 的官方文档中关于这两个函数的内容:

> `object.__repr__(self)`:由`repr()`内置函数和字符串转换(反引号)调用，计算对象的“正式”字符串表示。
> `object.__str__(self)`:由`str()`内置函数和 print 语句调用，计算对象的“非正式”字符串表示。
> 
> 引用自 [Python 的数据模型](http://docs.python.org/2/reference/datamodel.html)

从官方文档中，我们知道`__repr__`和`__str__`都是用来“表示”一个对象的。`__repr__`应该是“正式”代表，而`__str__`是“非正式”代表。

那么，Python 默认的任何对象的`__repr__`和`__str__`实现是什么样子的呢？

例如，假设我们有一个`int` `x`和一个`str` `y`，我们想知道这两个对象的`__repr__`和`__str__`的返回值:

```py

>>> x = 1

>>> repr(x)

'1'

>>> str(x)

'1'

>>> y = 'a string'

>>> repr(y)

"'a string'"

>>> str(y)

'a string'

```

虽然`int x`的`repr()`和`str()`的返回值是相同的，但是您应该注意到`str y`的返回值之间的差异。认识到`str`对象的默认实现`__repr__`可以作为`eval`的参数调用，返回值将是有效的`str`对象，这一点很重要:

```py

>>> repr(y)

"'a string'"

>>> y2 = eval(repr(y))

>>> y == y2

True

```

而`__str__`的返回值甚至不是一个可以被 eval 执行的有效语句:

```py

>>> str(y)

'a string'

>>> eval(str(y))

Traceback (most recent call last):

File "<stdin>", line 1, in <module>

File "<string>", line 1

a string

^

SyntaxError: unexpected EOF while parsing

```

因此，如果可能的话，对象的“正式”表示应该可以被 **eval()** 调用并返回相同的对象。如果不可能的话，比如对象的成员引用自己，导致无限循环引用，那么`__repr__`应该是明确的，包含尽可能多的信息。

```py

>>> class ClassA(object):

...   def __init__(self, b=None):

...     self.b = b

...

...   def __repr__(self):

...     return '%s(%r)' % (self.__class__, self.b)

...

>>>

>>> class ClassB(object):

...   def __init__(self, a=None):

...     self.a = a

...

...   def __repr__(self):

...     return "%s(%r)" % (self.__class__, self.a)

...

>>> a = ClassA()

>>> b = ClassB(a=a)

>>> a.b = b

>>> repr(b)

RuntimeError: maximum recursion depth exceeded while calling a Python object

```

你可以用不同的方式定义`ClassB.__repr__`,而不是完全遵循`__repr__`对`ClassB`的要求，这将导致无限递归问题，其中`a.__repr__`调用`b.__repr__`,而`b.__repr__`调用`a.__repr__`,后者调用`b.__repr__,`,如此循环往复。尽可能多地显示对象信息的方法与有效的 eval-constrained`__repr__`一样好。

```py

>>> class ClassB(object):

...   def __init__(self, a=None):

...     self.a = a

...

...   def __repr__(self):

...     return '%s(a=a)' % (self.__class__)

...
> > > a = class a()
>>>b = class b(a = a)
>>>a . b = b
>>>repr(a)
<class ' _ _ main _ _。主要的，主要的。class '>(a = a))"
>>>repr(b)
"<class ' _ _ main _ _。>(a = a)

```

因为`__repr__`是对象的官方表示，所以您总是希望调用`"repr(an_object)"`来获得关于对象的最全面的信息。然而，有时`__str__`也是有用的。因为`__repr__`可能太复杂而无法检查所讨论的对象是否复杂(想象一个对象有十几个属性)，`__str__`有助于快速概述复杂的对象。例如，假设您想要检查一个冗长日志文件中间的`datetime`对象，以找出用户照片的`datetime`不正确的原因:

```py

>>> from datetime import datetime

>>> now = datetime.now()

>>> repr(now)

'datetime.datetime(2013, 2, 5, 4, 43, 11, 673075)'

>>> str(now)

'2013-02-05 04:43:11.673075'

```

现在的`__str__`表示看起来比从`__repr__`生成的正式表示更清晰、更易读。有时候，能够快速理解对象中存储的内容对于理解复杂程序的“大”图景是很有价值的。

## Python 中 __str__ 和 __repr__ 之间的问题

需要记住的一个重要问题是，容器的`__str__`使用包含的对象的`__repr__`。

```py

>>> from datetime import datetime

>>> from decimal import Decimal

>>> print((Decimal('42'), datetime.now()))

(Decimal('42'), datetime.datetime(2013, 2, 5, 4, 53, 32, 646185))

>>> str((Decimal('42'), datetime.now()))

"(Decimal('42'), datetime.datetime(2013, 2, 5, 4, 57, 2, 459596))"

```

因为 Python 更喜欢明确性而不是可读性，所以元组的`__str__`调用调用包含的对象的`__repr__`，即对象的“正式”表示。虽然正式表示比非正式表示更难阅读，但它是明确的，并且对错误更健壮。

## Python 中 __str__ 和 __repr__ 之间的提示和建议

*   为你实现的每个类实现`__repr__`。不应该有任何借口。
*   对于你认为可读性对不模糊性更重要的类，实现`__str__`。