# Python 中的 Null:理解 Python 的 NoneType 对象

> 原文：<https://realpython.com/null-in-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 的 None: Null in Python**](/courses/python-none/)

如果你有使用其他编程语言的经验，如 [C](https://realpython.com/build-python-c-extension-module/) 或 [Java](https://realpython.com/oop-in-python-vs-java/) ，那么你可能听说过 **`null`** 的概念。许多语言用这个来表示不指向任何东西的[指针](https://realpython.com/pointers-in-python/)，表示[变量](https://realpython.com/python-variables/)何时为空，或者标记尚未提供的默认参数。在那些语言中,`null`通常被定义为`0`,但是在 Python 中的`null`是不同的。

Python 使用[关键字](https://realpython.com/python-keywords/) `None`来定义`null`对象和变量。虽然在其他语言中,`None`确实服务于与`null`相同的一些目的，但它完全是另一种野兽。与 Python 中的`null`一样，`None`没有被定义为`0`或其他任何值。在 Python 中，`None`是对象，是一等公民！

在本教程中，您将学习:

*   什么是 **`None`** 以及如何测试
*   何时以及为何使用`None`作为**默认参数**
*   在你的**回溯**中`None`和`NoneType`是什么意思
*   如何在**型式检验**中使用`None`
*   Python 中的 **`null`是如何工作的**

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 理解 Python 中的空值

`None`是函数中没有`return`语句时函数返回的值:

>>>

```py
>>> def has_no_return():
...     pass
>>> has_no_return()
>>> print(has_no_return())
None
```

当你调用`has_no_return()`时，你看不到任何输出。然而，当您打印对它的调用时，您将看到它返回的隐藏的`None`。

事实上，`None`如此频繁地作为返回值出现，以至于 Python [REPL](https://realpython.com/interacting-with-python/) 不会打印`None`，除非你明确地告诉它:

>>>

```py
>>> None
>>> print(None)
None
```

`None`本身没有输出，但是打印它会将`None`显示到控制台。

有趣的是， [`print()`](https://realpython.com/python-print/) 本身没有返回值。如果你试图打印一个对 [`print()`](https://realpython.com/courses/python-print/) 的调用，那么你会得到`None`:

>>>

```py
>>> print(print("Hello, World!"))
Hello, World!
None
```

这看起来可能很奇怪，但是`print(print("..."))`向你展示了内在`print()`返回的`None`。

`None`也常用作**缺失或默认参数**的信号。例如，`None`在 [`list.sort`](https://docs.python.org/3/library/stdtypes.html#list.sort "list.sort") 的文档中出现两次:

>>>

```py
>>> help(list.sort)
Help on method_descriptor:

sort(...)
 L.sort(key=None, reverse=False) -> None -- stable sort *IN PLACE*
```

这里，`None`是`key`参数的默认值，也是返回值的[类型提示](https://realpython.com/courses/python-type-checking/)。`help`的确切产量可能因平台而异。当您在您的解释器中运行这个命令时，您可能会得到不同的输出，但是它将是相似的。

[*Remove ads*](/account/join/)

## 使用 Python 的空对象`None`

通常，你会使用`None`作为比较的一部分。一个例子是当你需要检查某个结果或参数是否为`None`时。从 [`re.match`](https://docs.python.org/3/library/re.html#re.match "re.match") 中取你得到的结果。你的[正则表达式](https://realpython.com/regex-python/)匹配给定的字符串了吗？您将看到两种结果之一:

1.  **返回一个`Match`对象:**你的正则表达式找到一个匹配。
2.  **返回一个`None`对象:**你的正则表达式没有找到匹配。

在下面的代码块中，您正在测试模式`"Goodbye"`是否匹配一个[字符串](https://realpython.com/python-strings/):

>>>

```py
>>> import re
>>> match = re.match(r"Goodbye", "Hello, World!")
>>> if match is None:
...     print("It doesn't match.")
It doesn't match.
```

这里，您使用`is None`来测试模式是否匹配字符串`"Hello, World!"`。这个代码块演示了一个重要的规则，当您检查`None`时要记住:

*   **使用了身份运算符`is`和`is not`吗？**
*   **不要**使用等式运算符`==`和`!=`。

当您比较用户定义的对象时，等式操作符可能会被愚弄，这些对象被**覆盖**:

>>>

```py
>>> class BrokenComparison:
...     def __eq__(self, other):
...         return True
>>> b = BrokenComparison()
>>> b == None  # Equality operator
True
>>> b is None  # Identity operator
False
```

这里，等式运算符`==`返回错误的答案。另一方面，[身份操作符`is`](https://realpython.com/python-is-identity-vs-equality/) 不会被愚弄，因为你不能覆盖它。

**注意:**关于如何与`None`进行比较的更多信息，请查看[该做的和不该做的:Python 编程建议](https://realpython.com/lessons/dos-and-donts-python-programming-recommendations/)。

`None`是[福尔西](https://realpython.com/lessons/if-statements/)，意思是`not None`是`True`。如果您只想知道结果是否为假，那么如下测试就足够了:

>>>

```py
>>> some_result = None
>>> if some_result:
...     print("Got a result!")
... else:
...     print("No result.")
...
No result.
```

输出没有告诉你`some_result`就是`None`，只告诉你它是假的。如果你必须知道你是否有一个`None`对象，那么使用`is`和`is not`。

以下对象也是假的:

*   清空[列表](https://realpython.com/courses/lists-tuples-python/)
*   清空[字典](https://realpython.com/python-dicts/)
*   清空[组](https://realpython.com/python-sets/)
*   空[字符串](https://realpython.com/courses/python-strings/)
*   `0`
*   `False`

关于比较、真值和假值的更多信息，你可以阅读如何使用 [Python `or`操作符](https://realpython.com/python-or-operator/)，如何使用 [Python `and`操作符](https://realpython.com/python-and-operator/)，以及如何使用 [Python `not`操作符](https://realpython.com/python-not-operator/)。

## 在 Python 中声明空变量

在一些语言中，变量来源于**声明**。它们不需要被赋予初始值。在这些语言中，某些类型变量的初始默认值可能是`null`。然而，在 Python 中，变量来源于**赋值语句**。看一下下面的代码块:

>>>

```py
>>> print(bar)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'bar' is not defined
>>> bar = None
>>> print(bar)
None
```

这里，你可以看到一个值为`None`的变量不同于一个未定义的变量。Python 中的所有变量都是通过赋值产生的。如果你给一个变量赋值`None`，它在 Python 中只会以`null`开始。

[*Remove ads*](/account/join/)

## 使用`None`作为默认参数

通常，您会使用`None`作为可选参数的**默认值**。这里有一个很好的理由使用`None`而不是可变类型，比如 list。想象一个这样的函数:

```py
def bad_function(new_elem, starter_list=[]):
    starter_list.append(new_elem)
    return starter_list
```

`bad_function()`包含令人讨厌的惊喜。当您使用现有列表调用它时，它工作得很好:

>>>

```py
>>> my_list = ['a', 'b', 'c']
>>> bad_function('d', my_list)
['a', 'b', 'c', 'd']
```

在这里，您将`'d'`添加到列表的末尾，没有任何问题。

但是，如果您多次调用这个函数而没有使用`starter_list`参数，那么您将开始看到不正确的行为:

>>>

```py
>>> bad_function('a')
['a']
>>> bad_function('b')
['a', 'b']
>>> bad_function('c')
['a', 'b', 'c']
```

在定义函数时，`starter_list`的默认值只计算一次，所以每次没有传递现有列表时，代码都会重用它。

构建这个函数的正确方法是使用`None`作为默认值，然后测试它并根据需要实例化一个新的列表:

>>>

```py
 1>>> def good_function(new_elem, starter_list=None):
 2...     if starter_list is None: 3...         starter_list = [] 4...     starter_list.append(new_elem)
 5...     return starter_list
 6...
 7>>> good_function('e', my_list)
 8['a', 'b', 'c', 'd', 'e']
 9>>> good_function('a')
10['a']
11>>> good_function('b')
12['b']
13>>> good_function('c')
14['c']
```

`good_function()`通过每次调用创建一个新的列表，而不是传递一个现有的列表，按照您想要的方式进行操作。它之所以有效，是因为您的代码每次调用带有默认参数的函数时都会执行第 2 行和第 3 行。

## 在 Python 中使用`None`作为空值

当`None`是有效的输入对象时，你会怎么做？例如，如果`good_function()`可以向列表中添加元素，也可以不添加，而`None`是要添加的有效元素，那会怎么样呢？在这种情况下，您可以定义一个专门用作默认的类，同时与`None`相区别:

>>>

```py
>>> class DontAppend: pass
...
>>> def good_function(new_elem=DontAppend, starter_list=None):
...     if starter_list is None:
...         starter_list = []
...     if new_elem is not DontAppend:
...         starter_list.append(new_elem)
...     return starter_list
...
>>> good_function(starter_list=my_list)
['a', 'b', 'c', 'd', 'e']
>>> good_function(None, my_list)
['a', 'b', 'c', 'd', 'e', None]
```

在这里，类`DontAppend`作为不追加的信号，所以你不需要`None`来做这个。这让你可以在需要的时候添加`None`。

当`None`也可能是返回值时，您可以使用这种技术。例如，如果在字典中找不到关键字，默认情况下， [`dict.get`](https://realpython.com/python-dicts/#dgetltkeygt-ltdefaultgt) 会返回`None`。如果`None`在您的字典中是一个有效值，那么您可以这样调用`dict.get`:

>>>

```py
>>> class KeyNotFound: pass
...
>>> my_dict = {'a':3, 'b':None}
>>> for key in ['a', 'b', 'c']:
...     value = my_dict.get(key, KeyNotFound)
...     if value is not KeyNotFound:
...         print(f"{key}->{value}")
...
a->3
b->None
```

这里您已经定义了一个定制类`KeyNotFound`。现在，当一个键不在字典中时，你可以返回`KeyNotFound`，而不是返回`None`。这使得您可以返回`None`，而这是字典中的实际值。

## 回溯中的解密`None`

当`NoneType`出现在你的[回溯](https://realpython.com/python-traceback/)中，说明你没想到会是`None`的东西实际上是`None`，你试图用一种你不能用`None`的方式使用它。几乎总是，这是因为你试图在它上面调用一个方法。

例如，您在上面的`my_list`中多次调用了 [`append()`](https://realpython.com/python-append/) ，但是如果`my_list`不知何故变成了列表之外的任何东西，那么`append()`就会失败:

>>>

```py
>>> my_list.append('f')
>>> my_list
['a', 'b', 'c', 'd', 'e', None, 'f']
>>> my_list = None
>>> my_list.append('g')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'NoneType' object has no attribute 'append'
```

这里，您的代码引发了非常常见的`AttributeError`，因为底层对象`my_list`不再是一个列表。您已经将它设置为`None`，它不知道如何`append()`，因此代码抛出一个异常。

当您在代码中看到类似这样的[回溯](https://realpython.com/courses/python-traceback/)时，首先查找引发错误的属性。在这里，是`append()`。从那里，您将看到您试图调用它的对象。在这种情况下，它是`my_list`，您可以从回溯上方的代码中看出这一点。最后，弄清楚这个对象是如何变成`None`的，并采取必要的步骤来修改代码。

[*Remove ads*](/account/join/)

## 检查 Python 中的空值

在 Python 中，有两种[类型检查](https://realpython.com/python-type-checking/)情况需要关注`null`。第一种情况是你在[返回`None`T5 的时候:](https://realpython.com/python-type-checking/#functions-without-return-values)

>>>

```py
>>> def returns_None() -> None:
...     pass
```

这种情况类似于根本没有`return`语句时，默认情况下返回`None`。

第二种情况更具挑战性。它是您获取或返回一个值的地方，这个值可能是`None`，但也可能是其他(单个)类型。这种情况就像你对上面的`re.match`所做的，它返回一个`Match`对象或者`None`。

对于参数，过程是相似的:

```py
from typing import Any, List, Optional
def good_function(new_elem:Any, starter_list:Optional[List]=None) -> List:
    pass
```

从上面修改`good_function()`，从`typing`导入`Optional`，返回一个`Optional[Match]`。

## 在引擎盖下看一看

在许多其他语言中，`null`只是`0`的同义词，但 Python 中的`null`是一个成熟的**对象**:

>>>

```py
>>> type(None)
<class 'NoneType'>
```

这一行显示`None`是一个对象，它的类型是`NoneType`。

`None`本身作为 Python 中的`null`内置于语言中:

>>>

```py
>>> dir(__builtins__)
['ArithmeticError', ..., 'None', ..., 'zip']
```

在这里，你可以看到`__builtins__`列表中的`None`，它是解释器为 [`builtins`](https://docs.python.org/3/library/builtins.html) 模块保留的字典。

`None`是一个关键词，就像`True`和`False`一样。但是正因为如此，你不能像你可以直接从`__builtins__`到达`None`，比如说`ArithmeticError`。不过，你可以用一个 [`getattr()`](https://docs.python.org/3/library/functions.html#getattr) 的招数得到它:

>>>

```py
>>> __builtins__.ArithmeticError
<class 'ArithmeticError'>
>>> __builtins__.None
  File "<stdin>", line 1
    __builtins__.None
                    ^
SyntaxError: invalid syntax
>>> print(getattr(__builtins__, 'None'))
None
```

当你使用`getattr()`时，你可以从`__builtins__`中获取实际的`None`，这是你简单地用`__builtins__.None`索取无法做到的。

尽管 Python 在许多错误消息中输出了单词`NoneType`，但是`NoneType`在 Python 中并不是一个标识符。它不在`builtins`。只有用`type(None)`才能到达。

`None`是**的独生子**。也就是说，`NoneType`类只给你同一个`None`实例。您的 Python 程序中只有一个`None`:

>>>

```py
>>> my_None = type(None)()  # Create a new instance
>>> print(my_None)
None
>>> my_None is None
True
```

即使您试图创建一个新实例，您仍然会得到现有的`None`。

你可以用 [`id()`](https://docs.python.org/3.8/library/functions.html#id "id") 证明`None`和`my_None`是同一个对象:

>>>

```py
>>> id(None)
4465912088
>>> id(my_None)
4465912088
```

这里，`id`为`None`和`my_None`输出相同的整数值意味着它们实际上是同一个对象。

**注意:**由`id`产生的实际值会因系统而异，甚至会因程序执行而异。在最流行的 Python 运行时 [CPython](https://realpython.com/cpython-source-code-guide/) 下，`id()`通过报告对象的内存地址来完成工作。住在同一个内存地址的两个对象是同一个对象。

如果你试图赋值给`None`，那么你会得到一个 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) :

>>>

```py
>>> None = 5
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
SyntaxError: can't assign to keyword
>>> None.age = 5
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'NoneType' object has no attribute 'age'
>>> setattr(None, 'age', 5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'NoneType' object has no attribute 'age'
>>> setattr(type(None), 'age', 5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can't set attributes of built-in/extension type 'NoneType'
```

上面所有的例子都表明你不能修改`None`或者`NoneType`。它们是真正的常数。

您也不能子类化`NoneType`:

>>>

```py
>>> class MyNoneType(type(None)):
...     pass
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: type 'NoneType' is not an acceptable base type
```

这个回溯表明解释器不会让你创建一个继承自`type(None)`的新类。

[*Remove ads*](/account/join/)

## 结论

`None`是 Python 工具箱中一个强大的工具。与`True`和`False`一样，`None`是一个不可变的关键字。作为 Python 中的`null`,您可以用它来标记缺失的值和结果，甚至是默认参数，这是比可变类型更好的选择。

**现在你可以:**

*   用`is`和`is not`测试`None`
*   选择`None`何时是代码中的有效值
*   使用`None`及其替代参数作为默认参数
*   破译回溯中的`None`和`NoneType`
*   在类型提示中使用`None`和`Optional`

Python 中的`null`怎么用？在下面的评论区留下你的评论吧！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 的 None: Null in Python**](/courses/python-none/)******