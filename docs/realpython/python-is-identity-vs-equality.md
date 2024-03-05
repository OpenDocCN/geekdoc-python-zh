# Python！=' Is Not 'is not ':在 Python 中比较对象

> 原文：<https://realpython.com/python-is-identity-vs-equality/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**比较 Python 对象的正确方式:“is”vs = = "**](/courses/python-is-identity-vs-equality/)

Python 标识操作符(`is`)和等式操作符(`==`)之间有细微的区别。当您使用 Python 的`is`操作符比较数字时，您的代码可以运行得很好，直到它突然[不](https://medium.com/peloton-engineering/the-dangers-of-using-is-in-python-f42941124027)。你可能在哪里听说过 Python `is`操作符比`==`操作符快，或者你可能觉得它看起来更像[Python](https://realpython.com/tutorials/best-practices/)。然而，重要的是要记住，这些操作符的行为并不完全相同。

`==`操作符比较两个对象的值或**相等性**，而 Python `is`操作符检查两个[变量](https://realpython.com/python-variables/)是否指向内存中的同一个对象。在绝大多数情况下，这意味着你应该使用等号运算符`==`和`!=`，除非你比较 [`None`](https://realpython.com/null-in-python/) 。

在本教程中，您将学习:

*   **对象相等和对象相同**有什么区别
*   何时使用等式和等式运算符来比较对象
*   这些 **Python 操作者**在幕后做什么
*   为什么使用`is`和`is not`比较值会导致**意外行为**
*   如何编写一个**自定义`__eq__()`类方法**来定义等式运算符行为

**Python 中途站:**本教程是一个**快速**和**实用**的方法来找到你需要的信息，所以你会很快回到你的项目！

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 用 Python 比较身份是和不是运算符

Python 的`is`和`is not`操作符比较两个对象的**身份**。在 [CPython](https://realpython.com/cpython-source-code-guide/) 中，这是他们的内存地址。Python 中的一切都是一个[对象](https://realpython.com/courses/intro-object-oriented-programming-oop-python/)，每个对象都存储在一个特定的[内存位置](https://realpython.com/python-memory-management/)。Python 的`is`和`is not`操作符检查两个变量是否指向内存中的同一个对象。

注意:记住，具有相同值的对象通常存储在不同的内存地址。

您可以使用`id()`来检查对象的身份:

>>>

```py
>>> help(id)
Help on built-in function id in module builtins:

id(obj, /)
 Return the identity of an object.

 This is guaranteed to be unique among simultaneously existing objects.
 (CPython uses the object's memory address.)

>>> id(id)
2570892442576
```

最后一行显示了存储内置函数`id`本身的内存地址。

在一些常见的情况下，具有相同值的对象默认具有相同的 id。例如，数字-5 到 256 在 CPython 中被**拘留**。每个数字都存储在内存中一个单一的固定位置，这为常用的整数节省了内存。

你可以用`sys.intern()`到[实习生](https://docs.python.org/3.7/library/sys.html?highlight=sys.intern#sys.intern)的琴弦来演奏。此函数允许您比较它们的内存地址，而不是逐字符比较字符串:

>>>

```py
>>> from sys import intern
>>> a = 'hello world'
>>> b = 'hello world'
>>> a is b
False
>>> id(a)
1603648396784
>>> id(b)
1603648426160

>>> a = intern(a)
>>> b = intern(b)
>>> a is b
True
>>> id(a)
1603648396784
>>> id(b)
1603648396784
```

变量`a`和`b`最初指向内存中两个不同的对象，如它们不同的 id 所示。当你对它们进行实习时，你要确保`a`和`b`指向内存中的同一个对象。任何带有值`'hello world'`的新[字符串](https://realpython.com/python-strings/)现在都将在新的内存位置创建，但是当你实习这个新字符串时，你要确保它与你实习的第一个`'hello world'`指向相同的内存地址。

**注意:**即使一个对象的内存地址在任何给定的时间都是唯一的，它也会在相同代码的运行之间发生变化，并且依赖于 CPython 的版本和运行它的机器。

其他默认被拘留的对象有`None`、`True`、`False`和[简单字符串](https://github.com/satwikkansal/wtfpython#-strings-can-be-tricky-sometimes-)。请记住，大多数时候，具有相同值的不同对象将存储在不同的内存地址。**这意味着您不应该使用 Python `is`操作符来比较值。**

[*Remove ads*](/account/join/)

### 当只有一些整数被保留时

在幕后，Python 实习生使用常用值(例如，整数-5 到 256)的对象来[节省内存](https://docs.python.org/3/c-api/long.html#c.PyLongObject)。下面的代码向您展示了为什么只有一些整数有固定的内存地址:

>>>

```py
>>> a = 256
>>> b = 256
>>> a is b
True
>>> id(a)
1638894624
>>> id(b)
1638894624

>>> a = 257
>>> b = 257
>>> a is b
False

>>> id(a)
2570926051952
>>> id(b)
2570926051984
```

最初，`a`和`b`指向内存中同一个被拘留的对象，但是当它们的值超出**公共整数**(范围从-5 到 256)的范围时，它们被存储在不同的内存地址。

### 当多个变量指向同一个对象时

当您使用赋值操作符(`=`)使一个变量等于另一个变量时，您使这些变量指向内存中的同一个对象。这可能会导致[可变](https://realpython.com/courses/immutability-python/)对象的意外行为:

>>>

```py
>>> a = [1, 2, 3]
>>> b = a
>>> a
[1, 2, 3]
>>> b
[1, 2, 3]

>>> a.append(4)
>>> a
[1, 2, 3, 4]
>>> b
[1, 2, 3, 4]

>>> id(a)
2570926056520
>>> id(b)
2570926056520
```

刚刚发生了什么？您向`a`添加了一个新元素，但是现在`b`也包含了这个元素！嗯，在`b = a`的那一行，你设置`b`指向和`a`相同的内存地址，这样两个变量现在指向同一个对象。

如果你定义这些[列表](https://realpython.com/courses/lists-tuples-python/)彼此独立，那么它们被存储在不同的内存地址并独立运行:

>>>

```py
>>> a = [1, 2, 3]
>>> b = [1, 2, 3]
>>> a is b
False
>>> id(a)
2356388925576
>>> id(b)
2356388952648
```

因为`a`和`b`现在指的是内存中不同的对象，改变一个不会影响另一个。

## 用 Python == and 比较等式！=运算符

回想一下，具有**相同值**的对象通常存储在**单独的内存地址**。如果你想检查两个对象是否有相同的值，使用相等操作符`==`和`!=`，不管它们存储在内存的什么地方。绝大多数情况下，这是你想做的。

### 当对象副本相等但不相同时

在下面的例子中，您将`b`设置为`a`的副本(T1 是一个可变对象，比如一个[列表](https://realpython.com/python-lists-tuples/)或者一个[字典](https://realpython.com/python-dicts/))。这两个变量将具有相同的值，但每个变量将存储在不同的内存地址:

>>>

```py
>>> a = [1, 2, 3]
>>> b = a.copy()
>>> a
[1, 2, 3]
>>> b
[1, 2, 3]

>>> a == b
True
>>> a is b
False

>>> id(a)
2570926058312
>>> id(b)
2570926057736
```

`a`和`b`现在存储在不同的内存地址，所以`a is b`不再返回`True`。然而，`a == b`返回`True`，因为两个对象具有相同的值。

### 等式比较是如何工作的

等式运算符`==`的魔力发生在`==`符号左边的对象的`__eq__()`类方法中。

**注意:**除非右边的对象是左边对象的**子类**，否则就是这种情况。更多信息，请查看官方[文档](https://docs.python.org/3/reference/datamodel.html#object.__eq__)。

这是一个[神奇的类方法](https://realpython.com/operator-function-overloading/)，每当这个类的一个实例与另一个对象进行比较时，就会调用这个方法。如果这个方法没有实现，那么`==`默认比较两个对象的内存地址。

作为一个练习，创建一个从`str`继承的`SillyString`类，并实现`__eq__()`来比较这个字符串的长度是否与另一个对象的长度相同:

```py
class SillyString(str):
    # This method gets called when using == on the object
    def __eq__(self, other):
        print(f'comparing {self} to {other}')
        # Return True if self and other have the same length
        return len(self) == len(other)
```

现在，SillyString `'hello world'`应该等于 string `'world hello'`，甚至等于任何其他具有相同长度的对象:

>>>

```py
>>> # Compare two strings
>>> 'hello world' == 'world hello'
False

>>> # Compare a string with a SillyString
>>> 'hello world' == SillyString('world hello')
comparing world hello to hello world
True

>>> # Compare a SillyString with a list
>>> SillyString('hello world') == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
comparing hello world to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
True
```

当然，对于一个行为类似于字符串的对象来说，这是愚蠢的行为，但是它确实说明了当您使用`==`比较两个对象时会发生什么。除非实现了特定的`__ne__()`类方法，否则`!=`操作符会给出相反的响应。

上面的例子也清楚地向您展示了为什么使用 Python `is`操作符来与`None`进行比较，而不是使用`==`操作符是一个好的实践。它不仅因为比较内存地址而更快，而且因为不依赖于任何`__eq__()`类方法的逻辑而更安全。

[*Remove ads*](/account/join/)

## 比较 Python 比较运算符

作为一个经验法则，你应该总是使用等式运算符`==`和`!=`，除了当你和`None`比较的时候:

*   **使用 Python 的`==`和`!=`操作符来比较对象的相等性**。这里，你通常比较两个对象的值。如果你想比较两个对象是否有相同的内容，并且不关心它们在内存中的存储位置，这就是你需要的。

*   **当你想要比较对象身份**时，使用 Python `is`和`is not`操作符。这里，您比较的是两个变量是否指向内存中的同一个对象。这些运算符的主要用例是当您与`None`进行比较时。通过内存地址与`None`进行比较比使用类方法更快也更安全。

具有相同值的变量通常存储在不同的内存地址。这意味着您应该使用`==`和`!=`来比较它们的值，只有当您想要检查两个变量是否指向同一个内存地址时，才使用 Python 的`is`和`is not`操作符。

## 结论

在本教程中，您已经了解到`==`和`!=` **比较两个对象**的值，而 Python `is`和`is not`操作符比较两个变量**是否引用内存**中的同一个对象。如果您记住这一区别，那么您应该能够防止代码中的意外行为。

如果你想了解更多关于**对象实习**和 Python `is`操作符的精彩世界，那么看看[为什么在 Python](https://lerner.co.il/2015/06/16/why-you-should-almost-never-use-is-in-python/) 中几乎不应该使用“is”。您还可以看看如何使用`sys.intern()`来优化内存使用和字符串的比较时间，尽管 Python 可能已经在幕后自动为您处理了这一点。

既然您已经了解了**等式和等式操作符**的功能，那么您可以尝试编写自己的`__eq__()`类方法，这些方法定义了在使用`==`操作符时如何比较这个类的实例。去应用这些 Python 比较运算符的新知识吧！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**比较 Python 对象的正确方式:“is”vs = = "**](/courses/python-is-identity-vs-equality/)****