# Python 的 reduce():从函数式到 python 式

> 原文：<https://realpython.com/python-reduce-function/>

Python 的 [`reduce()`](https://docs.python.org/3/library/functools.html#functools.reduce) 是一个函数，实现了一种叫做 [**折叠**](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) 或者**还原**的数学技术。当您需要将一个函数应用于一个 iterable 并将它简化为一个累积值时,`reduce()`非常有用。Python 的`reduce()`在具有**函数式编程**背景的开发者中很受欢迎，但是 Python 还能提供更多。

在本教程中，你将会了解到`reduce()`是如何工作的，以及如何有效地使用它。您还将介绍一些替代的 Python 工具，它们可能比 T1 更加[Python 化](https://realpython.com/learning-paths/writing-pythonic-code/)、可读和高效。

在本教程中，您将学习:

*   Python 的 **`reduce()`** 是如何工作的
*   更常见的缩减用例有哪些
*   如何用`reduce()`来**解决**这些用例
*   有哪些替代的 Python 工具可用于解决这些相同的用例

有了这些知识，您将能够决定在解决 Python 中的归约或折叠问题时使用哪些工具。

为了更好地理解 Python 的`reduce()`，了解一下如何使用 [Python iterables](https://docs.python.org/3/glossary.html#term-iterable) ，尤其是如何使用 [`for`循环](https://realpython.com/courses/python-for-loop/)来循环它们，将会很有帮助。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 探索 Python 中的函数式编程

[函数式编程](https://realpython.com/python-functional-programming/)是一种基于将问题分解成一组独立函数的编程范式。理想情况下，每个函数只接受一组输入参数并产生一个输出。

在函数式编程中，对于给定的输入，函数没有任何影响输出的内部状态。这意味着任何时候你用相同的输入参数调用一个函数，你将得到相同的结果或输出。

在函数式程序中，输入数据流经一组函数。每个函数对其输入进行操作，并产生一些输出。函数式编程尽量避免可变的数据类型和状态变化。它处理函数间流动的数据。

函数式编程的其他核心特性包括:

*   使用[](https://realpython.com/python-thinking-recursively/)**递归而不是循环或其他结构作为主要的流量控制结构**
***   关注[列表](https://realpython.com/python-lists-tuples/)或数组处理*   关注于*要计算什么*而不是*如何*计算它*   使用[纯函数](https://en.wikipedia.org/wiki/Pure_function)，避免**副作用***   [高阶函数](http://en.wikipedia.org/wiki/Higher-order_function)的使用*

*这个列表中有几个重要的概念。以下是对其中一些的近距离观察:

*   **递归**是一种技术，在这种技术中，函数直接或间接地调用自己，以便进行循环。它允许程序在长度未知或不可预测的数据结构上循环。

*   **纯函数**是完全没有副作用的函数。换句话说，它们是不更新或修改程序中任何全局变量、对象或数据结构的函数。这些函数产生仅取决于输入的输出，这更接近于数学函数的概念。

*   **高阶函数**是通过将函数作为参数、返回函数或两者兼而有之来操作其他函数的函数，就像[Python decorator](https://realpython.com/primer-on-python-decorators/)一样。

由于 Python 是一种多范例编程语言，它提供了一些支持函数式编程风格的工具:

*   功能为[一级对象](https://realpython.com/primer-on-python-decorators/#first-class-objects)
*   [递归](https://realpython.com/python-recursion/)功能
*   匿名函数有 [`lambda`](https://realpython.com/python-lambda/)
*   [迭代器](https://docs.python.org/3/glossary.html#term-iterator)和[生成器](https://realpython.com/introduction-to-python-generators/)
*   标准模块，如 [`functools`](https://docs.python.org/3/library/functools.html#module-functools) 和 [`itertools`](https://realpython.com/python-itertools/)
*   工具有[`map()`](https://docs.python.org/3/library/functions.html#map)[`filter()`](https://docs.python.org/3/library/functions.html#filter)[`reduce()`](https://docs.python.org/3/library/functools.html#functools.reduce)[`sum()`](https://docs.python.org/3/library/functions.html#sum)[`len()`](https://realpython.com/len-python-function/)[`any()`](https://realpython.com/any-python/)[`all()`](https://realpython.com/python-all/)[`min()``max()`](https://realpython.com/python-min-and-max/)等等

尽管 Python [没有受到函数式编程语言](http://python-history.blogspot.com/2009/04/origins-of-pythons-functional-features.html)的很大影响，但早在 1993 年，就有对上面列出的一些函数式编程特性的明确需求。

作为回应，一些功能工具被添加到语言中。据[吉多·范·罗苏姆](https://es.wikipedia.org/wiki/Guido_van_Rossum)称，它们是由一名社区成员提供的:

> Python 获得了`lambda`、`reduce()`、`filter()`和`map()`，感谢(我相信)一个 Lisp 黑客错过了它们并提交了工作补丁。([来源](http://www.artima.com/weblogs/viewpost.jsp?thread=98196))

多年来，一些新特性，如[列表理解](https://realpython.com/list-comprehension-python/)、[生成器表达式](https://realpython.com/introduction-to-python-generators/)，以及内置函数如`sum()`、`min()`、`max()`、`all()`和`any()`，被视为[python 式](https://realpython.com/learning-paths/writing-pythonic-code/)对[、`map()`、](https://realpython.com/python-map-function/)[、`filter()`、`reduce()`的替代。Guido](https://realpython.com/python-filter-function/) [计划从 Python 3 的语言中删除](http://www.artima.com/weblogs/viewpost.jsp?thread=98196)`map()``filter()``reduce()`，甚至`lambda`。

幸运的是，这个移除并没有生效，主要是因为 Python 社区不想放过这么受欢迎的特性。它们仍然存在，并在具有强大函数式编程背景的开发人员中广泛使用。

在本教程中，你将讲述如何使用 Python 的`reduce()`来处理可重复项，并在不使用 [`for`循环](https://realpython.com/python-for-loop/)的情况下将它们减少到一个累积值。您还将了解到一些 Python 工具，您可以用它们来代替`reduce()`，使您的代码更加 Python 化、可读和高效。

[*Remove ads*](/account/join/)

## Python 的`reduce()` 入门

Python 的 **`reduce()`** 实现了一种俗称 [**折叠**](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) 或**还原**的数学技巧。当您将一系列项目缩减为单个累积值时，您正在进行折叠或缩减。Python 的`reduce()`操作任何[可迭代的](https://docs.python.org/3/glossary.html#term-iterable)——不仅仅是列表——并执行以下步骤:

1.  **将**一个函数(或可调用函数)应用于 iterable 中的前两项，并生成部分结果。
2.  **使用**该部分结果，连同 iterable 中的第三项，生成另一个部分结果。
3.  **重复**该过程，直到 iterable 用尽，然后返回一个累积值。

Python 的`reduce()`背后的思想是获取一个现有的函数，将其累积应用于 iterable 中的所有项，并生成一个最终值。一般来说，Python 的`reduce()`在处理可重复项时很方便，无需编写显式的`for`循环。由于`reduce()`是用 C 写的，它的内部循环可以比显式的 Python `for`循环更快。

Python 的`reduce()`原本是内置函数(在 [Python 2.x](https://docs.python.org/2/library/functions.html#reduce) 中依然如此)，但在 [Python 3.0](https://docs.python.org/3/whatsnew/3.0.html#builtins) 中被移到了`functools.reduce()`。这个决定是基于一些可能的性能和可读性问题。

将`reduce()`移至 [`functools`](https://docs.python.org/3/library/functools.html#module-functools) 的另一个原因是引入了内置函数，如`sum()`、`any()`、`all()`、`max()`、`min()`和`len()`，这些函数为`reduce()`提供了更高效、更易读和更 Pythonic 化的处理常见用例的方式。在本教程的后面，您将学习如何使用它们来代替`reduce()`。

在 Python 3.x 中，如果你需要使用`reduce()`，那么你首先必须使用一个 [`import`语句](https://realpython.com/courses/python-imports-101/)将函数导入到你的[当前作用域](https://realpython.com/python-scope-legb-rule/)中，方法如下:

1.  **`import functools`** 然后像`functools.reduce()`一样使用[全限定名](https://docs.python.org/3/glossary.html#term-qualified-name)。
2.  **`from functools import reduce`** 然后直接调用`reduce()`。

根据`reduce()`的[文档](https://docs.python.org/3/library/functools.html#functools.reduce)，该函数具有以下特征:

```py
functools.reduce(function, iterable[, initializer])
```

Python 文档还指出`reduce()`大致相当于以下 Python 函数:

```py
def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value
```

像这个 Python 函数一样，`reduce()`的工作原理是在一个从左到右的循环中将一个双参数函数应用到`iterable`的项上，最终将`iterable`减少到一个累积的`value`。

Python 的`reduce()`还接受第三个可选参数`initializer`，它为计算或归约提供一个种子值。

在接下来的两节中，您将深入了解 Python 的`reduce()`是如何工作的，以及每个参数背后的含义。

### 所需参数:`function`和`iterable`

Python 的`reduce()`的第一个参数是一个两个参数的函数，方便地称为`function`。该函数将应用于 iterable 中的项目，以累计计算最终值。

尽管[官方文档](https://docs.python.org/3/library/functools.html#functools.reduce)将`reduce()`的第一个参数称为“两个参数的函数”，但是只要可调用对象接受两个参数，您就可以将任何 Python 可调用对象传递给`reduce()`。可调用对象包括[类](https://realpython.com/python3-object-oriented-programming/)，实现一个叫做 [`__call__()`](https://docs.python.org/3/reference/datamodel.html#object.__call__) 的特殊方法的实例，实例方法，[类方法，静态方法](https://realpython.com/courses/staticmethod-vs-classmethod-python/)，以及函数。

**注意:**关于 Python 可调用对象的更多细节，您可以查看 Python [文档](https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy)并向下滚动到“可调用类型”

第二个必需参数`iterable`，顾名思义，将接受任何 Python iterable。这包括[列表、元组](https://realpython.com/python-lists-tuples/)、[、`range`对象](https://realpython.com/courses/python-range-function/)、生成器、迭代器、[集合](https://realpython.com/courses/sets-python/)、[字典](https://realpython.com/courses/dictionaries-python/)键和值，以及任何其他可以迭代的 Python 对象。

**注意**:如果你给 Python 的`reduce()`传递一个迭代器，那么这个函数需要穷尽迭代器才能得到最终值。所以，手头的迭代器不会保持[懒惰](https://en.wikipedia.org/wiki/Lazy_evaluation)。

为了理解`reduce()`是如何工作的，你将编写一个函数来计算两个[数字](https://realpython.com/python-numbers/)和[的和，并将等价的数学运算打印到屏幕上。代码如下:](https://realpython.com/python-print/)

>>>

```py
>>> def my_add(a, b):
...     result = a + b
...     print(f"{a} + {b} = {result}")
...     return result
```

该函数计算`a`和`b`的和，使用 [f 串](https://realpython.com/courses/python-3-f-strings-improved-string-formatting-syntax/)打印一条操作消息，并返回计算结果。它是这样工作的:

>>>

```py
>>> my_add(5, 5)
5 + 5 = 10
10
```

`my_add()`是一个双参数函数，所以可以将它和一个 iterable 一起传递给 Python 的`reduce()`,以计算 iterable 中项的累积和。查看以下使用数字列表的代码:

>>>

```py
>>> from functools import reduce

>>> numbers = [0, 1, 2, 3, 4]

>>> reduce(my_add, numbers)
0 + 1 = 1
1 + 2 = 3
3 + 3 = 6
6 + 4 = 10
10
```

当您调用`reduce()`，传递`my_add()`和`numbers`作为参数时，您会得到一个输出，显示`reduce()`执行的所有操作，以得出`10`的最终结果。在这种情况下，操作等同于`((((0 + 1) + 2) + 3) + 4) = 10`。

上例中对`reduce()`的调用将`my_add()`应用于`numbers` ( `0`和`1`)中的前两项，并得到结果`1`。然后`reduce()`使用`1`和`numbers`中的下一项(即`2`)作为参数调用`my_add()`，得到结果`3`。重复该过程，直到`numbers`用完所有项目，并且`reduce()`返回`10`的最终结果。

[*Remove ads*](/account/join/)

### `initializer`可选参数:

Python 的`reduce()`的第三个参数叫做`initializer`，是可选的。如果你给`initializer`提供一个值，那么`reduce()`会把它作为第一个参数提供给`function`的第一个调用。

这意味着对`function`的第一次调用将使用`initializer`的值和`iterable`的第一项来执行它的第一次部分计算。之后，`reduce()`继续处理`iterable`的后续项目。

下面是一个使用`my_add()`并将`initializer`设置为`100`的例子:

>>>

```py
>>> from functools import reduce

>>> numbers = [0, 1, 2, 3, 4]

>>> reduce(my_add, numbers, 100)
100 + 0 = 100
100 + 1 = 101
101 + 2 = 103
103 + 3 = 106
106 + 4 = 110
110
```

因为您为`initializer`提供了一个值`100`，Python 的`reduce()`在第一次调用中使用这个值作为`my_add()`的第一个参数。注意，在第一次迭代中，`my_add()`使用`100`和`0`，即`numbers`的第一项，来执行计算`100 + 0 = 100`。

另一点需要注意的是，如果你给`initializer`提供一个值，那么`reduce()`将比没有`initializer`时多执行一次迭代。

如果您计划使用`reduce()`来处理可能为空的 iterables，那么最好为`initializer`提供一个值。当`iterable`为空时，Python 的`reduce()`将使用该值作为其默认返回值。如果你不提供一个`initializer`值，那么`reduce()`将引发一个`TypeError`。看一下下面的例子:

>>>

```py
>>> from functools import reduce

>>> # Using an initializer value >>> reduce(my_add, [], 0)  # Use 0 as return value
0

>>> # Using no initializer value >>> reduce(my_add, [])  # Raise a TypeError with an empty iterable
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: reduce() of empty sequence with no initial value
```

如果你用空的`iterable`调用`reduce()`，那么函数将返回提供给`initializer`的值。如果你不提供一个`initializer`，那么`reduce()`在处理空的可重复项时会抛出一个`TypeError`。

**注意:**要深入了解什么是 Python 回溯，请查看[了解 Python 回溯](https://realpython.com/python-traceback/)。

既然您已经熟悉了`reduce()`的工作方式，那么您就可以学习如何将它应用于一些常见的编程问题。

## 用 Python 的`reduce()` 减少迭代次数

到目前为止，您已经学习了 Python 的`reduce()`是如何工作的，以及如何使用一个[用户定义函数](https://realpython.com/defining-your-own-python-function/)来使用它减少迭代次数。您还了解了`reduce()`的每个参数的含义以及它们是如何工作的。

在这一节中，您将看到`reduce()`的一些常见用例，以及如何使用函数解决它们。您还将了解到一些替代的 Python 工具，您可以使用它们来代替`reduce()`来使您的代码更加 Python 化、高效和可读。

### 对数值求和

Python 的`reduce()`的`"Hello, World!"`就是 **sum 用例**。它包括计算一系列数字的累积和。假设你有一个类似`[1, 2, 3, 4]`的数字列表。其总和将为`1 + 2 + 3 + 4 = 10`。这里有一个如何使用 Python `for`循环解决这个问题的简单例子:

>>>

```py
>>> numbers = [1, 2, 3, 4]
>>> total = 0
>>> for num in numbers:
...     total += num
...
>>> total
10
```

`for`循环迭代`numbers`中的每个值，并在`total`中累加它们。最终结果是所有值的总和，在本例中是`10`。像本例中的`total`一样使用的[变量](https://realpython.com/courses/variables-python/)有时被称为**累加器**。

这可以说是 Python 的`reduce()`最常见的用例。要用`reduce()`实现这个操作，您有几种选择。其中一些包括使用具有以下功能之一的`reduce()`:

*   一个[用户自定义函数](https://realpython.com/defining-your-own-python-function/)
*   一个 [`lambda`功能](https://realpython.com/courses/python-lambda-functions/)
*   一个函数叫做 [`operator.add()`](https://docs.python.org/3/library/operator.html#operator.add)

要使用用户定义的函数，您需要编写一个将两个数相加的函数。然后你就可以用`reduce()`来使用那个功能了。对于这个例子，您可以如下重写`my_add()`:

>>>

```py
>>> def my_add(a, b):
...     return a + b
...

>>> my_add(1, 2)
3
```

`my_add()`将两个数`a`和`b`相加，并返回结果。有了`my_add()`,您可以使用`reduce()`来计算 Python iterable 中值的总和。方法如下:

>>>

```py
>>> from functools import reduce

>>> numbers = [1, 2, 3, 4]

>>> reduce(my_add, numbers)
10
```

对`reduce()`的调用将`my_add()`应用于`numbers`中的项目，以计算它们的累积和。最后的结果是`10`，不出所料。

您也可以通过使用`lambda`函数来执行相同的计算。在这种情况下，您需要一个`lambda`函数，它接受两个数字作为参数并返回它们的和。看一下下面的例子:

>>>

```py
>>> from functools import reduce

>>> numbers = [1, 2, 3, 4]

>>> reduce(lambda a, b: a + b, numbers)
10
```

`lambda`函数接受两个参数并返回它们的和。`reduce()`在循环中应用`lambda`函数来计算`numbers`中各项的累计和。

同样，您可以利用名为 [`operator`](https://docs.python.org/3/library/operator.html#module-operator) 的 Python 模块。这个模块导出了一堆对应于 Python 内部操作符的函数。对于手头的问题，您可以将`operator.add()`与 Python 的`reduce()`一起使用。看看下面的例子:

>>>

```py
>>> from operator import add
>>> from functools import reduce

>>> add(1, 2)
3

>>> numbers = [1, 2, 3, 4]

>>> reduce(add, numbers)
10
```

在这个例子中，`add()`接受两个参数并返回它们的和。所以，你可以使用`add()`和`reduce()`来计算`numbers`所有项目的总和。由于`add()`是用 C 语言编写的，并且针对效率进行了优化，所以当使用`reduce()`来解决 sum 用例时，它可能是您的最佳选择。注意使用`operator.add()`也比使用`lambda`函数更具可读性。

sum 用例在编程中如此常见，以至于 Python 从[版本 2.3](https://docs.python.org/3/whatsnew/2.3.html#other-language-changes) 开始就包含了一个专用的内置函数`sum()`来解决这个问题。`sum()`被声明为`sum(iterable[, start])`。

`start`是`sum()`的可选参数，默认为`0`。该函数将`start`的值从左到右加到`iterable`的项目上，并返回总数。看一下下面的例子:

>>>

```py
>>> numbers = [1, 2, 3, 4]

>>> sum(numbers)
10
```

由于`sum()`是内置函数，所以不需要导入任何东西。你随时都可以得到它。使用`sum()`是解决 sum 用例的最巧妙的方法。它干净、易读、简洁。它遵循一个核心 Python 原则:

> 简单比复杂好。([来源](https://www.python.org/dev/peps/pep-0020))

与使用`reduce()`或`for`循环相比，`sum()`的加入在可读性和性能方面是一个巨大的胜利。

**注意:**关于比较`reduce()`与其他 Python reduction 工具的性能的更多细节，请查看[性能是关键](#performance-is-key)一节。

如果您正在处理 sum 用例，那么良好的实践推荐使用`sum()`。

[*Remove ads*](/account/join/)

### 数值相乘

Python 的`reduce()`的**乘积用例**与 sum 用例颇为相似，但这次的运算是乘法。换句话说，您需要计算 iterable 中所有值的乘积。

例如，假设您有一个列表`[1, 2, 3, 4]`。它的产品将是`1 * 2 * 3 * 4 = 24`。您可以使用 Python `for`循环来计算。看看下面的例子:

>>>

```py
>>> numbers = [1, 2, 3, 4]
>>> product = 1
>>> for num in numbers:
...     product *= num
...
>>> product
24
```

循环迭代`numbers`中的项目，将每个项目乘以前一次迭代的结果。在这种情况下，累加器`product`的起始值应该是`1`而不是`0`。因为任何数乘以零都是零，所以起始值`0`将总是使你的乘积等于`0`。

这个计算也是 Python 的`reduce()`的一个非常流行的用例。同样，您将涉及解决问题的三种方法。您将把`reduce()`用于:

1.  用户定义的函数
2.  一个`lambda`功能
3.  一个函数叫做 [`operator.mul()`](https://docs.python.org/3/library/operator.html#operator.mul)

对于选项 1，您需要编写一个自定义函数，它接受两个参数并返回它们的乘积。然后您将使用这个函数和`reduce()`来计算 iterable 中各项的乘积。看一下下面的代码:

>>>

```py
>>> from functools import reduce

>>> def my_prod(a, b):
...     return a * b
...

>>> my_prod(1, 2)
2

>>> numbers = [1, 2, 3, 4]

>>> reduce(my_prod, numbers)
24
```

函数`my_prod()`将两个数`a`和`b`相乘。对`reduce()`的调用遍历`numbers`的项，并通过将`my_prod()`应用于连续的项来计算它们的乘积。最终结果是`numbers`中所有项目的乘积，在本例中是`24`。

如果您喜欢使用一个`lambda`函数来解决这个用例，那么您需要一个接受两个参数并返回它们的乘积的函数。这里有一个例子:

>>>

```py
>>> from functools import reduce

>>> numbers = [1, 2, 3, 4]

>>> reduce(lambda a, b: a * b, numbers)
24
```

当`reduce()`遍历`numbers`时，匿名函数通过将连续的项目相乘来变魔术。同样，结果是`numbers`中所有项目的乘积。

您还可以使用`operator.mul()`来处理产品用例。`operator.mul()`取两个数，返回两个数相乘的结果。这是解决当前问题的正确功能。看看下面的例子:

>>>

```py
>>> from operator import mul
>>> from functools import reduce

>>> mul(2, 2)
4

>>> numbers = [1, 2, 3, 4]

>>> reduce(mul, numbers)
24
```

由于`mul()`是高度优化的，如果你使用这个函数，而不是用户定义的函数或`lambda`函数，你的代码会执行得更好。请注意，这个解决方案的可读性也更好。

最后，如果您使用的是 [Python 3.8](https://realpython.com/courses/cool-new-features-python-38/) ，那么您就可以获得这个用例的更 Python 化、可读性更强的解决方案。Python 3.8 增加了一个名为 [`prod()`](https://docs.python.org/3/library/math.html#math.prod) 的新函数，它驻留在 [Python `math`模块](https://realpython.com/python-math-module/)中。这个函数类似于`sum()`，但是返回一个`start`值乘以一个`iterable`数的乘积。

对于`math.prod()`，参数`start`是可选的，默认为`1`。它是这样工作的:

>>>

```py
>>> from math import prod

>>> numbers = [1, 2, 3, 4]

>>> prod(numbers)
24
```

与使用`reduce()`相比，这在可读性和效率方面也是一大胜利。所以，如果你使用的是 [Python 3.8](https://realpython.com/python38-new-features/) ，并且产品缩减是你代码中的常见操作，那么你使用`math.prod()`会比使用 Python 的`reduce()`更好。

[*Remove ads*](/account/join/)

### 寻找最小值和最大值

**在 iterable 中寻找最小和最大**值的问题也是一个归约问题，您可以使用 Python 的`reduce()`来解决。这个想法是比较 iterable 中的项目，找出最小值或最大值。

假设你有一个数字列表`[3, 5, 2, 4, 7, 1]`。在这个列表中，最小值是`1`，最大值是`7`。要找到这些值，可以使用 Python `for`循环。查看以下代码:

>>>

```py
>>> numbers = [3, 5, 2, 4, 7, 1]

>>> # Minimum >>> min_value, *rest = numbers
>>> for num in rest:
...     if num < min_value:
...         min_value = num
...
>>> min_value
1

>>> # Maximum >>> max_value, *rest = numbers
>>> for num in rest:
...     if num > max_value:
...         max_value = num
...
>>> max_value
7
```

两个循环都迭代`rest`中的项目，并根据连续比较的结果更新`min_value`或`max_value`的值。注意最初，`min_value`和`max_value`持有数字`3`，这是`numbers`中的第一个值。变量`rest`保存`numbers`中的剩余值。换句话说，`rest = [5, 2, 4, 7, 1]`。

**注意:**在上面的例子中，你使用 Python [iterable 解包操作符(`*` )](https://www.python.org/dev/peps/pep-0448) 到**解包**或将`numbers`中的值展开成两个变量。在第一种情况下，净效果是`min_value`获得`numbers`中的第一个值，即`3`，而`rest`将剩余的值收集到一个列表中。

查看以下示例中的详细信息:

>>>

```py
>>> numbers = [3, 5, 2, 4, 7, 1]

>>> min_value, *rest = numbers
>>> min_value
3
>>> rest
[5, 2, 4, 7, 1]

>>> max_value, *rest = numbers
>>> max_value
3
>>> rest
[5, 2, 4, 7, 1]
```

Python iterable 解包操作符(`*`)在您需要将一个序列或 iterable 解包成几个变量时非常有用。

为了更好地理解 Python 中的解包操作，你可以查看一下 [PEP 3132 扩展的可迭代解包](https://www.python.org/dev/peps/pep-3132)和 [PEP 448 附加解包一般化](https://www.python.org/dev/peps/pep-0448)。

现在，考虑如何使用 Python 的`reduce()`找到 iterable 中的最小值和最大值。同样，您可以根据需要使用用户定义的函数或`lambda`函数。

下面的代码实现了一个使用两个不同的用户定义函数的解决方案。第一个函数将接受两个参数，`a`和`b`，并返回它们的最小值。第二个函数将使用类似的过程，但它将返回最大值。

下面是一些函数，以及如何将它们与 Python 的`reduce()`一起使用来查找 iterable 中的最小值和最大值:

>>>

```py
>>> from functools import reduce

>>> # Minimum >>> def my_min_func(a, b):
...     return a if a < b else b
...

>>> # Maximum >>> def my_max_func(a, b):
...     return a if a > b else b
...

>>> numbers = [3, 5, 2, 4, 7, 1]

>>> reduce(my_min_func, numbers)
1

>>> reduce(my_max_func, numbers)
7
```

当你用`my_min_func()`和`my_max_func()`运行`reduce()`时，你分别得到`numbers`中的最小值和最大值。`reduce()`遍历`numbers`的条目，进行累积对比较，最终返回最小值或最大值。

**注意:**为了实现`my_min_func()`和`my_max_func()`，您使用了一个 Python 条件表达式，或者三元运算符，作为一个`return`值。要深入了解什么是条件表达式以及它们如何工作，请查看 Python 中的[条件语句(if/elif/else)](https://realpython.com/python-conditional-statements/#conditional-expressions-pythons-ternary-operator) 。

你也可以使用一个`lambda`函数来解决最小值和最大值问题。看看下面的例子:

>>>

```py
>>> from functools import reduce

>>> numbers = [3, 5, 2, 4, 7, 1]

>>> # Minimum >>> reduce(lambda a, b: a if a < b else b, numbers)
1

>>> # Maximum >>> reduce(lambda a, b: a if a > b else b, numbers)
7
```

这一次，您使用两个`lambda`函数来确定`a`是小于还是大于`b`。在这种情况下，Python 的`reduce()`将`lambda`函数应用于`numbers`中的每个值，并与之前的计算结果进行比较。在这个过程的最后，你得到最小值或最大值。

最小值和最大值问题在编程中非常常见，因此 Python 添加了内置函数来执行这些缩减。这些函数被方便地称为 [`min()`](https://docs.python.org/3/library/functions.html#min) 和 [`max()`](https://docs.python.org/3/library/functions.html#max) ，你不需要导入任何东西就能使用它们。它们是这样工作的:

>>>

```py
>>> numbers = [3, 5, 2, 4, 7, 1]

>>> min(numbers)
1

>>> max(numbers)
7
```

当您使用`min()`和`max()`来查找 iterable 中的最小值和最大值时，您的代码比使用 Python 的`reduce()`更具可读性。此外，由于`min()`和`max()`是高度优化的 C 函数，你也可以说你的代码会更有效率。

所以，在用 Python 解决这个问题时，最好使用`min()`和`max()`而不是`reduce()`。

### 检查所有值是否为真

Python 的`reduce()`的**全真用例**涉及到发现一个 iterable 中的所有项是否都为真。要解决这个问题，您可以将`reduce()`与用户定义的函数或`lambda`函数一起使用。

首先编写一个`for`循环，看看 iterable 中的所有项是否都为真。代码如下:

>>>

```py
>>> def check_all_true(iterable):
...     for item in iterable:
...         if not item:
...             return False
...     return True
...

>>> check_all_true([1, 1, 1, 1, 1])
True

>>> check_all_true([1, 1, 1, 1, 0])
False

>>> check_all_true([])
True
```

如果`iterable`中的所有值都为真，那么`check_all_true()`返回`True`。否则返回`False`。它也返回空的可重复项`True`。`check_all_true()`执行一个**短路评估**。这意味着函数一发现假值就返回，而不处理`iterable`中的其余项目。

为了使用 Python 的`reduce()`解决这个问题，您需要编写一个函数，它接受两个参数，如果两个参数都为真，则返回`True`。如果一个或两个参数都为假，那么函数将返回`False`。代码如下:

>>>

```py
>>> def both_true(a, b):
...     return bool(a and b)
...

>>> both_true(1, 1)
True

>>> both_true(1, 0)
False

>>> both_true(0, 0)
False
```

这个函数有两个参数，`a`和`b`。然后使用 [`and`操作符](https://docs.python.org/3/reference/expressions.html#and)来测试两个参数是否都为真。如果两个参数都为真，返回值将是`True`。否则，它将成为`False`。

在 Python 中，以下对象[被认为是假的](https://docs.python.org/3/library/stdtypes.html#truth-value-testing):

*   [常数](https://realpython.com/python-constants/)像 [`None`](https://realpython.com/null-in-python/) 和`False`
*   具有零值的数字类型，如`0`、`0.0`、`0j`、[、`Decimal(0)`、](https://docs.python.org/3/library/decimal.html#decimal.Decimal)[、`Fraction(0, 1)`、](https://docs.python.org/3/library/fractions.html#fractions.Fraction)
*   空序列和集合，如`""`、`()`、`[]`、`{}`、[、](https://realpython.com/python-sets/)和`range(0)`
*   实现返回值为`False`的 [`__bool__()`](https://docs.python.org/3/reference/datamodel.html#object.__bool__) 或返回值为`0`的 [`__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 的对象

任何其他对象都将被视为真。

你需要使用 [`bool()`](https://docs.python.org/3/library/functions.html#bool) 将`and`的返回值转换成`True`或者`False`。如果你不使用`bool()`，那么你的函数不会像预期的那样运行，因为`and`返回表达式中的一个对象，而不是`True`或`False`。看看下面的例子:

>>>

```py
>>> a = 0
>>> b = 1
>>> a and b
0

>>> a = 1
>>> b = 2
>>> a and b
2
```

如果表达式中的第一个值为 false，则返回该值。否则，它将返回表达式中的最后一个值，而不考虑其真值。这就是为什么在这种情况下需要使用`bool()`的原因。`bool()`返回对布尔表达式或对象求值后得到的[布尔值](https://realpython.com/python-boolean/) ( `True`或`False`)。使用`bool()`查看示例:

>>>

```py
>>> a = 0
>>> b = 1
>>> bool(a and b)
False

>>> a = 1
>>> b = 2
>>> bool(a and b)
True
```

在对表达式或手边的对象求值后，`bool()`将总是返回`True`或`False`。

**注意:**为了更好的理解 Python 中的操作符和表达式，可以查阅 Python 中的[操作符和表达式](https://realpython.com/python-operators-expressions/)。

你可以通过`both_true()`到`reduce()`来检查一个 iterable 的所有项是否为真。这是如何工作的:

>>>

```py
>>> from functools import reduce

>>> reduce(both_true, [1, 1, 1, 1, 1])
True

>>> reduce(both_true, [1, 1, 1, 1, 0])
False

>>> reduce(both_true, [], True)
True
```

如果您将`both_true()`作为参数传递给`reduce()`，那么如果 iterable 中的所有项都为真，您将得到`True`。否则你会得到`False`。

在第三个例子中，您将`True`传递给`reduce()`的`initializer`，以获得与`check_all_true()`相同的行为，并避免出现`TypeError`。

你也可以使用一个`lambda`函数来解决`reduce()`的全真用例。以下是一些例子:

>>>

```py
>>> from functools import reduce

>>> reduce(lambda a, b: bool(a and b), [0, 0, 1, 0, 0])
False

>>> reduce(lambda a, b: bool(a and b), [1, 1, 1, 2, 1])
True

>>> reduce(lambda a, b: bool(a and b), [], True)
True
```

这个`lambda`函数与`both_true()`非常相似，使用相同的表达式作为返回值。如果两个参数都为真，则返回`True`。否则返回`False`。

请注意，与`check_all_true()`不同，当您使用`reduce()`来解决全真用例时，没有短路评估，因为`reduce()`直到遍历整个可迭代对象后才返回。这会给代码增加额外的处理时间。

例如，假设您有一个列表`lst = [1, 0, 2, 0, 0, 1]`，您需要检查`lst`中的所有项目是否都为真。在这种情况下，`check_all_true()`将在它的循环处理完第一对条目(`1`和`0`)后立即结束，因为`0`为假。你不需要继续迭代，因为你手头已经有了问题的答案。

另一方面，`reduce()`解决方案直到处理完`lst`中的所有项目才会结束。那是五次迭代之后。现在想象一下，如果您正在处理一个大的可迭代对象，这会对您的代码性能产生什么影响！

幸运的是，Python 提供了正确的工具，以一种 Python 式的、可读的、高效的方式解决所有真实的问题:内置函数 [`all()`](https://realpython.com/python-all/) 。

您可以使用`all(iterable)`来检查`iterable`中的所有项目是否都为真。以下是`all()`的工作方式:

>>>

```py
>>> all([1, 1, 1, 1, 1])
True

>>> all([1, 1, 1, 0, 1])
False

>>> all([])
True
```

循环遍历 iterable 中的项目，检查每个项目的真值。如果`all()`发现一个错误的条目，那么它返回`False`。否则返回`True`。如果你用一个空的 iterable 调用`all()`，那么你会得到`True`，因为在一个空的 iterable 中没有 false 项。

`all()`是一个针对性能优化的 C 函数。该功能也通过短路评估来实现。所以，如果你正在处理 Python 中的全真问题，那么你应该考虑使用`all()`而不是`reduce()`。

[*Remove ads*](/account/join/)

### 检查是否有值为真

Python 的`reduce()`的另一个常见用例是**任意真实用例**。这一次，您需要确定 iterable 中是否至少有一项为真。要解决这个问题，您需要编写一个函数，它接受一个 iterable，如果 iterable 中的任何一项为真，则返回`True`，否则返回`False`。看看这个函数的如下实现:

>>>

```py
>>> def check_any_true(iterable):
...     for item in iterable:
...         if item:
...             return True
...     return False
...

>>> check_any_true([0, 0, 0, 1, 0])
True

>>> check_any_true([0, 0, 0, 0, 0])
False

>>> check_any_true([])
False
```

如果`iterable`中至少有一项为真，那么`check_any_true()`返回`True`。只有当*所有*项为假或者 iterable 为空时，它才返回`False`。这个函数还实现了一个短路评估，因为它一找到真值(如果有的话)就返回。

为了使用 Python 的`reduce()`解决这个问题，您需要编写一个函数，它接受两个参数，如果其中至少有一个为真，则返回`True`。如果两者都为假，那么函数应该返回`False`。

下面是这个函数的一个可能的实现:

>>>

```py
>>> def any_true(a, b):
...     return bool(a or b)
...

>>> any_true(1, 0)
True

>>> any_true(0, 1)
True

>>> any_true(0, 0)
False
```

如果至少有一个参数为真，则`any_true()`返回`True`。如果两个参数都为假，那么`any_true()`返回`False`。与上面部分中的`both_true()`一样，`any_true()`使用`bool()`将表达式`a or b`的结果转换为`True`或`False`。

[Python `or`操作符](https://realpython.com/python-or-operator/)的工作方式与`and`略有不同。它返回表达式中的第一个真对象或最后一个对象。看看下面的例子:

>>>

```py
>>> a = 1
>>> b = 2
>>> a or b
1

>>> a = 0
>>> b = 1
>>> a or b
1

>>> a = 0
>>> b = []
>>> a or b
[]
```

Python `or`操作符返回第一个真对象，或者，如果两个都为假，则返回最后一个对象。因此，您还需要使用`bool()`从`any_true()`获得一致的返回值。

一旦你有了这个功能，你就可以继续减少。看看下面对`reduce()`的调用:

>>>

```py
>>> from functools import reduce

>>> reduce(any_true, [0, 0, 0, 1, 0])
True

>>> reduce(any_true, [0, 0, 0, 0, 0])
False

>>> reduce(any_true, [], False)
False
```

您已经使用 Python 的`reduce()`解决了这个问题。请注意，在第三个例子中，您将`False`传递给`reduce()`的初始化器，以重现原始`check_any_true()`的行为，同时避免出现`TypeError`。

**注:**和上一节的例子一样，`reduce()`的这些例子不做短路评价。这意味着它们会影响代码的性能。

您还可以使用带有`reduce()`的`lambda`函数来解决任何真实的用例。你可以这样做:

>>>

```py
>>> from functools import reduce

>>> reduce(lambda a, b: bool(a or b), [0, 0, 1, 1, 0])
True

>>> reduce(lambda a, b: bool(a or b), [0, 0, 0, 0, 0])
False

>>> reduce(lambda a, b: bool(a or b), [], False)
False
```

这个`lambda`功能和`any_true()`挺像的。如果两个参数中有一个为真，它将返回`True`。如果两个参数都为假，那么它返回`False`。

尽管这种解决方案只需要一行代码，但它仍然会使您的代码不可读，或者至少难以理解。同样，Python 提供了一个不使用`reduce()`就能高效解决任意真问题的工具:内置函数 [`any()`](https://realpython.com/any-python/) 。

`any(iterable)`循环遍历`iterable`中的项目，测试每个项目的真值，直到找到一个真项目。该函数一找到真值就返回`True`。如果`any()`没有找到真值，那么它返回`False`。这里有一个例子:

>>>

```py
>>> any([0, 0, 0, 0, 0])
False

>>> any([0, 0, 0, 1, 0])
True

>>> any([])
False
```

同样，您不需要导入`any()`来在代码中使用它。`any()`按预期工作。如果 iterable 中的所有项都是假的，它将返回`False`。否则返回`True`。注意，如果你用一个空的 iterable 调用`any()`，那么你会得到`False`，因为在一个空的 iterable 中没有 true 项。

与`all()`一样，`any()`是一个针对性能优化的 C 函数。它也是通过短路评估实现的。所以，如果你正在处理 Python 中的任意真问题，那么考虑使用`any()`而不是`reduce()`。

[*Remove ads*](/account/join/)

## 比较`reduce()`和`accumulate()`T2

一个名为 [`accumulate()`](https://docs.python.org/3/library/itertools.html#itertools.accumulate) 的 Python 函数驻留在 [`itertools`](https://realpython.com/python-itertools/) 中，行为类似于`reduce()`。`accumulate(iterable[, func])`接受一个必需的参数`iterable`，它可以是任何 Python iterable。可选的第二个参数`func`需要是一个函数(或一个可调用对象)，它接受两个参数并返回一个值。

返回一个迭代器。这个迭代器中的每一项都将是`func`执行的计算的累积结果。默认计算是总和。如果你不给`accumulate()`提供一个函数，那么结果迭代器中的每一项都将是`iterable`中前面的项加上手边的项的累加和。

看看下面的例子:

>>>

```py
>>> from itertools import accumulate
>>> from operator import add
>>> from functools import reduce

>>> numbers = [1, 2, 3, 4]

>>> list(accumulate(numbers))
[1, 3, 6, 10]

>>> reduce(add, numbers)
10
```

注意，结果迭代器中的最后一个值与`reduce()`返回的值相同。这是这两个函数的主要相似之处。

**注意:**由于`accumulate()`返回一个迭代器，所以需要调用`list()`来消耗迭代器，得到一个 list 对象作为输出。

另一方面，如果您为`accumulate()`的`func`参数提供一个双参数函数(或可调用函数),那么结果迭代器中的项将是由`func`执行的计算的累积结果。这里有一个使用`operator.mul()`的例子:

>>>

```py
>>> from itertools import accumulate
>>> from operator import mul
>>> from functools import reduce

>>> numbers = [1, 2, 3, 4]

>>> list(accumulate(numbers, mul))
[1, 2, 6, 24]

>>> reduce(mul, numbers)
24
```

在这个例子中，您可以再次看到`accumulate()`返回值中的最后一项等于`reduce()`返回的值。

## 考虑性能和可读性

Python 的`reduce()`可能会有非常糟糕的性能，因为它通过多次调用函数来工作。这可能会使您的代码运行缓慢且效率低下。当使用复杂的用户定义函数或`lambda`函数时，使用`reduce()`也会损害代码的可读性。

在本教程中，您已经了解到 Python 提供了一系列工具，可以优雅地替代`reduce()`，至少对于它的主要用例是这样。以下是到目前为止你阅读的主要收获:

1.  **尽可能使用专用函数**来解决 Python 的`reduce()`用例。诸如`sum()`、`all()`、`any()`、`max()`、`min()`、`len()`、`math.prod()`等函数会让你的代码更快，可读性更好，可维护性更强，并且[python 化](https://realpython.com/learning-paths/writing-pythonic-code/)。

2.  **使用`reduce()`时避免复杂的用户自定义函数**。这些类型的函数会使你的代码难以阅读和理解。您可以使用一个显式的、可读的`for`循环来代替。

3.  **使用`reduce()`时避免复杂的`lambda`功能**。它们还会让你的代码变得不可读和混乱。

第二点和第三点是圭多本人关切的问题，他说:

> 所以现在`reduce()`。这实际上是我最讨厌的一个，因为除了几个涉及`+`或`*`的例子，几乎每次我看到一个带有重要函数参数的`reduce()`调用，我都需要拿起笔和纸来画出实际输入到那个函数中的内容，然后我才明白`reduce()`应该做什么。因此，在我看来，`reduce()`的适用性仅限于关联操作符，在其他情况下，最好显式写出累加循环。([来源](http://www.artima.com/weblogs/viewpost.jsp?thread=98196))

接下来的两节将帮助您在代码中实现这个一般建议。他们还提供了一些额外的建议，帮助你在真正需要使用 Python 的`reduce()`时有效地使用它。

### 性能是关键

如果您打算使用`reduce()`来解决您在本教程中所涉及的用例，那么您的代码将会比使用专用内置函数的代码慢得多。在下面的例子中，您将使用 [`timeit.timeit()`](https://docs.python.org/3/library/timeit.html#timeit.timeit) 来快速测量少量 Python 代码的执行时间，并了解它们的总体性能。

`timeit()`需要几个参数，但是对于这些例子，你只需要使用下面的:

*   **`stmt`** 持有你需要时间的陈述。
*   **`setup`** 需要额外的语句进行常规设置，就像 [`import`语句](https://realpython.com/absolute-vs-relative-python-imports/)一样。
*   **`globals`** 拥有一个字典，其中包含运行`stmt`所需的[全局名称空间](https://realpython.com/python-namespaces-scope/)。

看一下下面的例子，这些例子对使用不同工具的`reduce()`和使用 Python 的`sum()`的**和用例**进行了计时:

>>>

```py
>>> from functools import reduce
>>> from timeit import timeit

>>> # Using a user-defined function >>> def add(a, b):
...     return a + b
...
>>> use_add = "functools.reduce(add, range(100))"
>>> timeit(use_add, "import functools", globals={"add": add})
13.443158069014316

>>> # Using a lambda expression >>> use_lambda = "functools.reduce(lambda x, y: x + y, range(100))"
>>> timeit(use_lambda, "import functools")
11.998800784000196

>>> # Using operator.add() >>> use_operator_add = "functools.reduce(operator.add, range(100))"
>>> timeit(use_operator_add, "import functools, operator")
5.183870767941698

>>> # Using sum() >>> timeit("sum(range(100))", globals={"sum": sum})
1.1643308430211619
```

即使你会得到不同的数字，取决于你的硬件，你可能会得到最好的时间测量使用`sum()`。这个内置函数也是 sum 问题可读性最强、最 Pythonic 化的解决方案。

**注意:**关于如何为你的代码计时的更详细的方法，请查看 [Python 计时器函数:监控你的代码的三种方法](https://realpython.com/python-timer/)。

第二个最好的选择是将`reduce()`和`operator.add()`一起使用。`operator`中的函数是用 C 语言编写的，并且针对性能进行了高度优化。因此，它们应该比用户定义的函数、`lambda`函数或`for`循环执行得更好。

[*Remove ads*](/account/join/)

### 可读性计数

当使用 Python 的`reduce()`时，代码可读性也是一个重要的关注点。尽管`reduce()`通常会比 Python `for`循环执行得更好，正如 Guido 自己所说，干净的[Python 循环](https://realpython.com/courses/how-to-write-pythonic-loops/)通常比使用`reduce()`更容易理解。

Python 3.0 指南中的[新特性强调了这一观点，它说:](https://docs.python.org/3/whatsnew/3.0.html)

> 如果真的需要就用`functools.reduce()`；然而，99%的情况下，显式的`for`循环更具可读性。([来源](https://docs.python.org/3/whatsnew/3.0.html)

为了更好地理解可读性的重要性，假设您开始学习 Python，并试图解决一个关于计算 iterable 中所有偶数之和的练习。

如果你已经知道 Python 的`reduce()`并且在过去做过一些函数式编程，那么你可能会想到下面的解决方案:

>>>

```py
>>> from functools import reduce

>>> def sum_even(it):
...     return reduce(lambda x, y: x + y if not y % 2 else x, it, 0)
...

>>> sum_even([1, 2, 3, 4])
6
```

在这个函数中，使用`reduce()`对 iterable 中的偶数进行累加求和。`lambda`函数接受两个参数`x`和`y`，如果它们是偶数，则返回它们的和。否则，它返回`x`，其中保存了前一次求和的结果。

此外，您将`initializer`设置为`0`，因为否则您的 sum 将有一个初始值`1`(`iterable`中的第一个值)，它不是一个偶数，会在您的函数中引入一个 bug。

该函数按照您的预期工作，并且您对结果感到满意。但是，您将继续深入研究 Python，了解`sum()`和[生成器表达式](https://realpython.com/introduction-to-python-generators/)。您决定使用这些新工具重新设计您的函数，现在您的函数如下所示:

>>>

```py
>>> def sum_even(iterable):
...     return sum(num for num in iterable if not num % 2)
...

>>> sum_even([1, 2, 3, 4])
6
```

当你看到这些代码时，你会感到非常自豪，你应该这样做。你做得很好！这是一个漂亮的 Python 函数，读起来几乎像普通英语。它也是高效的和 Pythonic 式的。你怎么想呢?

## 结论

Python 的`reduce()`允许你使用 Python 调用和`lambda`函数对 iterables 执行归约操作。`reduce()`将函数应用于 iterable 中的项目，并将它们简化为单个累积值。

**在本教程中，您已经学习了:**

*   什么是**还原**，或者**折叠**，以及它什么时候可能有用
*   如何使用 Python 的 **`reduce()`** 解决常见的数字相加或相乘等归约问题
*   哪些**python 工具**可以用来有效地替换代码中的`reduce()`

有了这些知识，在解决 Python 中的归约问题时，您将能够决定哪些工具最适合您的编码需求。

这些年来，`reduce()`已经被更多的 Pythonic 工具所取代，比如`sum()`、`min()`、`max()`、`any()`等等。但是，`reduce()`还在，还在函数式程序员中流行。如果你对使用`reduce()`或它的任何 Python 替代品有任何问题或想法，那么一定要在下面的评论中发表。*********