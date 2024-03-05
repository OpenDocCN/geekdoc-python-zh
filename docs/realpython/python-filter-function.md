# Python 的 filter():从 Iterables 中提取值

> 原文：<https://realpython.com/python-filter-function/>

Python 的 [`filter()`](https://docs.python.org/3/library/functions.html#filter) 是一个内置函数，允许你处理一个 iterable，提取那些满足给定条件的项。这个过程通常被称为**滤波**操作。使用`filter()`，您可以将一个**过滤函数**应用到一个 iterable，并生成一个新的 iterable，其中包含满足当前条件的条目。在 Python 中，`filter()`是你可以用来进行[函数式编程](https://realpython.com/python-functional-programming/)的工具之一。

**在本教程中，您将学习如何:**

*   在你的代码中使用 Python 的 **`filter()`**
*   从你的迭代中提取**需要的值**
*   将`filter()`与其他**功能工具**结合
*   **用更多的**蟒**工具替换** `filter()`

有了这些知识，您将能够在代码中有效地使用`filter()`。或者，您可以选择使用[列表理解](https://realpython.com/list-comprehension-python/)或[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)来编写更多的[python 式](https://realpython.com/learning-paths/writing-pythonic-code/)和可读代码。

为了更好地理解`filter()`，对[可迭代](https://docs.python.org/3/glossary.html#term-iterable)、[、`for`循环](https://realpython.com/python-for-loop/)、[函数](https://realpython.com/defining-your-own-python-function/)和[、`lambda`函数](https://realpython.com/python-lambda/)有所了解会有所帮助。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 中的函数式编码

**函数式编程**是一种范式，提倡使用函数来执行程序中的几乎每一项任务。纯函数式风格依赖于不修改输入参数和不改变程序状态的函数。他们只是采用一组特定的参数，而[每次都返回](https://realpython.com/python-return-statement/)相同的结果。这类函数被称为[纯函数](https://en.wikipedia.org/wiki/Pure_function)。

在函数式编程中，函数通常对数据数组进行操作、转换，并产生具有附加功能的新数组。函数式编程中有三种基本操作:

1.  [映射](https://en.wikipedia.org/wiki/Map_(higher-order_function))将一个转换函数应用于一个可迭代对象，并生成一个新的可迭代的已转换项目。
2.  [过滤](https://en.wikipedia.org/wiki/Filter_(higher-order_function))将一个[谓词或布尔值函数](https://en.wikipedia.org/wiki/Boolean-valued_function)应用于一个可迭代对象，并生成一个新的可迭代对象，其中包含满足[布尔](https://realpython.com/python-boolean/)条件的项目。
3.  [归约](https://en.wikipedia.org/wiki/Fold_(higher-order_function))将归约函数应用于 iterable，并返回单个累积值。

Python [并没有受到函数式语言](https://web.archive.org/web/20161104183819/http://python-history.blogspot.com.br/2009/04/origins-of-pythons-functional-features.html)的严重影响，而是受到了[命令式语言](https://en.wikipedia.org/wiki/Imperative_programming)的影响。但是，它提供了几个允许您使用函数样式的特性:

*   [匿名函数](https://realpython.com/python-lambda/)
*   一个 [`map()`](https://realpython.com/python-map-function/) 功能
*   一个 [`filter()`](https://docs.python.org/3/library/functions.html#filter) 功能
*   一个 [`reduce()`](https://realpython.com/python-reduce-function/) 功能

Python 中的函数是[一级对象](https://realpython.com/primer-on-python-decorators/#first-class-objects)，这意味着你可以像对待任何其他对象一样传递它们。您还可以将它们用作其他函数的参数和返回值。接受其他函数作为参数或者返回函数(或者两者都接受)的函数被称为[高阶函数](https://en.wikipedia.org/wiki/Higher-order_function)，这也是函数式编程中的一个理想特性。

在本教程中，您将了解到`filter()`。这个内置函数是 Python 中比较流行的函数工具之一。

[*Remove ads*](/account/join/)

## 理解过滤问题

假设您需要处理一个由[数字](https://realpython.com/python-numbers/)组成的列表，并返回一个只包含那些大于`0`的数字的新列表。解决这个问题的一个快速方法是像这样使用一个`for`循环:

>>>

```py
>>> numbers = [-2, -1, 0, 1, 2]

>>> def extract_positive(numbers):
...     positive_numbers = []
...     for number in numbers:
...         if number > 0:  # Filtering condition ...             positive_numbers.append(number)
...     return positive_numbers
...

>>> extract_positive(numbers)
[1, 2]
```

`extract_positive()`中的循环遍历`numbers`并将每个大于`0`的数字存储在`positive_numbers`中。[条件语句](https://realpython.com/python-conditional-statements/) *过滤掉*负数和`0`。这种功能被称为**过滤**。

过滤操作包括用一个[谓词](https://en.wikipedia.org/wiki/Predicate_(mathematical_logic))函数测试 iterable 中的每个值，并只保留那些函数产生真结果的值。过滤操作在编程中相当常见，所以大多数编程语言都提供了处理这些操作的工具。在下一节中，您将了解 Python 过滤 iterables 的方法。

## Python 的`filter()` 入门

Python 提供了一个方便的内置函数`filter()`，它抽象出了过滤操作背后的逻辑。这是它的签名:

```py
filter(function, iterable)
```

第一个参数`function`必须是单参数函数。通常，您为该参数提供一个谓词(布尔值)函数。换句话说，你提供了一个根据特定条件返回`True`或`False`的函数。

这个`function`起到了**决策函数**的作用，也称为**过滤函数**，因为它提供了从输入可迭代中过滤掉不需要的值并在结果可迭代中保留那些您想要的值的标准。请注意，术语**不需要的值**指的是当`filter()`使用`function`处理它们时评估为假的那些值。

**注意:**`filter()`的第一个参数是一个**函数对象**，这意味着你需要传递一个函数，而不需要用一对括号来调用它。

第二个参数`iterable`，可以保存任何 Python iterable，比如一个[列表，元组](https://realpython.com/python-lists-tuples/)，或者[集合](https://realpython.com/python-sets/)。它还可以保存生成器和迭代器对象。关于`filter()`重要的一点是它只接受一个`iterable`。

为了执行过滤过程，`filter()`在一个循环中将`function`应用于`iterable`的每一项。结果是一个迭代器，它产生`iterable`的值，其中`function`返回一个真值。该过程不会修改原始的输入 iterable。

由于`filter()`是用 [C](https://github.com/python/cpython/blob/master/Python/bltinmodule.c) 编写的，并且经过了高度优化，其内部隐式循环在执行时间方面比常规的`for`循环更高效。这种效率可以说是在 Python 中使用函数的最重要的优势。

在循环中使用`filter()`的第二个优点是，它返回一个`filter`对象，这是一个根据需要产生值的迭代器，促进了一种[惰性求值](https://en.wikipedia.org/wiki/Lazy_evaluation)策略。返回迭代器使得`filter()`比等价的`for`循环更有内存效率。

**注意:**在 Python 2.x 中， [`filter()`](https://docs.python.org/2/library/functions.html#filter) 返回`list`对象。这种行为在 [Python 3.x](https://docs.python.org/3/whatsnew/3.0.html#views-and-iterators-instead-of-lists) 中有所改变。现在，该函数返回一个`filter`对象，这是一个迭代器，根据需要生成条目。众所周知，Python 迭代器是内存高效的。

在关于正数的例子中，可以使用`filter()`和一个方便的谓词函数来提取所需的数字。为了编写谓词，您可以使用一个`lambda`或一个用户定义的函数:

>>>

```py
>>> numbers = [-2, -1, 0, 1, 2]

>>> # Using a lambda function
>>> positive_numbers = filter(lambda n: n > 0, numbers)
>>> positive_numbers
<filter object at 0x7f3632683610>
>>> list(positive_numbers)
[1, 2]

>>> # Using a user-defined function
>>> def is_positive(n):
...     return n > 0
...
>>> list(filter(is_positive, numbers))
[1, 2]
```

在第一个例子中，您使用了一个提供过滤功能的`lambda`函数。对`filter()`的调用将`lambda`函数应用于`numbers`中的每个值，并过滤掉负数和`0`。由于`filter()`返回一个迭代器，所以需要调用`list()`来消耗迭代器并创建最终列表。

**注意:**因为`filter()`是一个内置函数，你不需要[导入](https://realpython.com/python-import/)任何东西来在你的代码中使用它。

在第二个示例中，您编写`is_positive()`来接受一个数字作为参数，如果该数字大于`0`，则返回`True`。否则返回`False`。对`filter()`的调用将`is_positive()`应用于`numbers`中的每个值，过滤掉负数。这个解决方案比它的对等方案更具可读性。

实际上，`filter()`并不局限于上面例子中的布尔函数。您可以使用其他类型的函数，并且`filter()`将评估它们的返回值的真实性:

>>>

```py
>>> def identity(x):
...     return x
...

>>> identity(42)
42

>>> objects = [0, 1, [], 4, 5, "", None, 8]
>>> list(filter(identity, objects))
[1, 4, 5, 8]
```

在这个例子中，过滤函数`identity()`没有显式返回`True`或`False`，而是采用了相同的参数。由于`0`、`[]`、`""`、[、`None`、](https://realpython.com/null-in-python/)为假，`filter()`使用它们的**真值**将其过滤掉。最终的列表只包含那些在 Python 中为真的值。

**注意:** Python 遵循一套规则来确定一个对象的真值。

例如，下列[对象是假的](https://docs.python.org/3/library/stdtypes.html#truth-value-testing):

*   像 [`None`](https://realpython.com/null-in-python/) 和`False`这样的常数
*   带有零值的数字类型，如`0`、`0.0`、`0j`、[、`Decimal(0)`、](https://docs.python.org/3/library/decimal.html#decimal.Decimal)[、`Fraction(0, 1)`、](https://docs.python.org/3/library/fractions.html#fractions.Fraction)
*   空序列和集合，如`""`、`()`、`[]`、`{}`、[、](https://realpython.com/python-sets/)、`range(0)`
*   实现返回值为`False`的 [`__bool__()`](https://docs.python.org/3/reference/datamodel.html#object.__bool__) 或返回值为`0`的 [`__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 的对象

任何其他物体都将被认为是真实的。

最后，如果您将`None`传递给`function`，那么`filter()`使用**标识函数**并产生`iterable`中所有评估为`True`的元素:

>>>

```py
>>> objects = [0, 1, [], 4, 5, "", None, 8]

>>> list(filter(None, objects))
[1, 4, 5, 8]
```

在这种情况下，`filter()`使用您之前看到的 Python 规则测试输入 iterable 中的每一项。然后它产生那些评估为`True`的项目。

到目前为止，你已经学习了`filter()`的基础知识以及它是如何工作的。在接下来的小节中，您将学习如何使用`filter()`来处理可重复项，并在没有循环的情况下丢弃不需要的值。

[*Remove ads*](/account/join/)

## 用`filter()` 过滤可重复项

`filter()`的工作是对输入 iterable 中的每个值应用一个决策函数，并返回一个新的 iterable，其中包含那些通过测试的项。以下部分提供了一些实用的例子，这样您就可以开始使用`filter()`了。

### 提取偶数

作为第一个例子，假设您需要处理一个整数列表，并构建一个包含偶数的新列表。解决这个问题的第一个方法可能是使用如下的`for`循环:

>>>

```py
>>> numbers = [1, 3, 10, 45, 6, 50]

>>> def extract_even(numbers):
...     even_numbers = []
...     for number in numbers:
...         if number % 2 == 0:  # Filtering condition ...             even_numbers.append(number)
...     return even_numbers
...

>>> extract_even(numbers)
[10, 6, 50]
```

这里，`extract_even()`接受一个整数的 iterable 并返回一个只包含偶数的列表。条件语句扮演着过滤器的角色，它测试每个数字，以确定它是否是偶数。

当您遇到这样的代码时，您可以将过滤逻辑提取到一个小的谓词函数中，并与`filter()`一起使用。这样，您可以在不使用显式循环的情况下执行相同的计算:

>>>

```py
>>> numbers = [1, 3, 10, 45, 6, 50]

>>> def is_even(number):
...     return number % 2 == 0  # Filtering condition ...

>>> list(filter(is_even, numbers))
[10, 6, 50]
```

这里，`is_even()`接受一个整数，如果是偶数，则返回`True`，否则返回`False`。对`filter()`的调用做了艰苦的工作，过滤掉奇数。结果，你得到一个偶数的列表。这段代码比它的对等`for`循环更短更有效。

### 寻找质数

另一个有趣的例子可能是提取给定区间内所有的[素数](https://en.wikipedia.org/wiki/Prime_number)。为此，您可以首先编写一个谓词函数，该函数将一个整数作为参数，如果该数字是质数，则返回`True`，否则返回`False`。你可以这样做:

>>>

```py
>>> import math

>>> def is_prime(n):
...     if n <= 1:
...         return False
...     for i in range(2, int(math.sqrt(n)) + 1):
...         if n % i == 0:
...             return False
...     return True
...

>>> is_prime(5)
True
>>> is_prime(12)
False
```

过滤逻辑现在在`is_prime()`中。该函数遍历`2`和`n`的[平方根](https://realpython.com/python-square-root-function/)之间的整数。在循环内部，[条件语句](https://realpython.com/python-conditional-statements/)检查当前数字是否能被区间中的任何其他数字整除。如果是，那么函数返回`False`,因为这个数不是质数。否则，它返回`True`来表示输入的数字是质数。

有了`is_prime()`并经过测试，您可以使用`filter()`从一个区间中提取素数，如下所示:

>>>

```py
>>> list(filter(is_prime, range(1, 51)))
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
```

对`filter()`的调用提取了在`1`和`50`之间的所有质数。`is_prime()`中使用的算法来自维基百科关于[素性测试](https://en.wikipedia.org/wiki/Primality_test#Simple_methods)的文章。如果您需要更有效的方法，可以查看那篇文章。

### 去除样本中的异常值

当你试图[描述和总结一个样本数据](https://realpython.com/python-statistics/)时，你可能会从寻找它的平均值开始。平均值是一种非常流行的[集中趋势](https://en.wikipedia.org/wiki/Central_tendency)度量，并且通常是分析数据集的第一种方法。它能让你快速了解数据的中心，或**位置**。

在某些情况下，对于给定的样本，平均值不是一个足够好的集中趋势度量。[异常值](https://realpython.com/python-statistics/#outliers)是影响平均值准确度的因素之一。异常值是与样本或总体中的其他观察值显著不同的[数据点](https://en.wikipedia.org/wiki/Data_point)。除此之外，它们在统计学中没有唯一的数学定义。

然而，在[正态分布的](https://en.wikipedia.org/wiki/Normal_distribution)样本中，异常值通常被定义为距离样本均值超过两个[标准偏差](https://en.wikipedia.org/wiki/Standard_deviation)的数据点。

现在，假设您有一个正态分布的样本，其中有一些影响平均准确度的异常值。您已经研究了异常值，并且知道它们是不正确的数据点。以下是如何使用 [`statistics`](https://docs.python.org/3/library/statistics.html#module-statistics) 模块中的几个函数和`filter()`来清理数据:

>>>

```py
>>> import statistics as st
>>> sample = [10, 8, 10, 8, 2, 7, 9, 3, 34, 9, 5, 9, 25]

>>> # The mean before removing outliers
>>> mean = st.mean(sample)
>>> mean
10.692307692307692

>>> stdev = st.stdev(sample)
>>> low = mean - 2 * stdev
>>> high = mean + 2 * stdev

>>> clean_sample = list(filter(lambda x: low <= x <= high, sample)) >>> clean_sample
[10, 8, 10, 8, 2, 7, 9, 3, 9, 5, 9, 25]

>>> # The mean after removing outliers
>>> st.mean(clean_sample)
8.75
```

在突出显示的行中，如果给定的数据点位于平均值和两个标准偏差之间，`lambda`函数返回`True`。否则，它返回`False`。用此功能过滤`sample`时，`34`被排除。在这种清理之后，样本的平均值具有显著不同的值。

[*Remove ads*](/account/join/)

### 验证 Python 标识符

您也可以对包含非数字数据的 iterables 使用`filter()`。例如，假设您需要处理一列[字符串](https://realpython.com/python-strings/)，并提取那些有效的 Python [标识符](https://docs.python.org/3/reference/lexical_analysis.html#identifiers)。在做了一些研究之后，您发现 Python 的 [`str`](https://docs.python.org/3/library/stdtypes.html#str) 提供了一个名为 [`.isidentifier()`](https://docs.python.org/3/library/stdtypes.html#str.isidentifier) 的方法，可以帮助您完成验证。

下面是如何使用`filter()`和`str.isidentifier()`来快速验证标识符:

>>>

```py
>>> words = ["variable", "file#", "header", "_non_public", "123Class"]

>>> list(filter(str.isidentifier, words))
['variable', 'header', '_non_public']
```

在这种情况下，`filter()`在`words`中的每个字符串上运行`.isidentifier()`。如果字符串是有效的 Python 标识符，那么它将包含在最终结果中。否则，该单词将被过滤掉。注意，在调用`filter()`时，需要使用`str`来访问`.isidentifier()`。

**注意:**除了`.isidentifier()`，`str`提供了一套丰富的`.is*()` [方法](https://docs.python.org/3/library/stdtypes.html#string-methods)，可以用来过滤字符串的可重复项。

最后，一个有趣的练习可能是进一步举例，检查标识符是否也是一个[关键字](https://realpython.com/python-keywords/)。来吧，试一试！提示:您可以使用 [`keyword`](https://docs.python.org/3/library/keyword.html#module-keyword) 模块中的 [`.kwlist`](https://docs.python.org/3/library/keyword.html#keyword.kwlist) 。

### 寻找回文单词

当您熟悉 Python 字符串时，经常出现的一个练习是在字符串列表中找到[回文单词](https://en.wikipedia.org/wiki/Palindrome)。回文单词向后读和向前读是一样的。典型的例子是“夫人”和“赛车”

要解决这个问题，首先要编写一个谓词函数，该函数接受一个字符串，并检查它在两个方向(向前和向后)上的读数是否相同。下面是一个可能的实现:

>>>

```py
>>> def is_palindrome(word):
...     reversed_word = "".join(reversed(word))
...     return word.lower() == reversed_word.lower()
...

>>> is_palindrome("Racecar")
True
>>> is_palindrome("Python")
False
```

在`is_palindrome()`中，你首先将原来的`word`反转，存储在`reversed_word`中。然后返回两个词相等的比较结果。在这种情况下，您使用`.lower()`来防止与大小写相关的差异。如果你用一个回文单词调用这个函数，那么你会得到`True`。否则，你得到`False`。

您已经有了一个可以识别回文单词的谓词函数。以下是你如何使用`filter()`来完成这项艰巨的工作:

>>>

```py
>>> words = ("filter", "Ana", "hello", "world", "madam", "racecar")

>>> list(filter(is_palindrome, words))
['Ana', 'madam', 'racecar']
```

酷！你的`filter()`和`is_palindrome()`组合工作正常。它同样简洁、易读、高效。干得好！

## 将`filter()`与其他功能工具结合

到目前为止，您已经学习了如何使用`filter()`在 iterables 上运行不同的过滤操作。在实践中，您可以将`filter()`与其他功能工具结合起来，在不使用显式循环的情况下对可迭代对象执行许多不同的任务。在接下来的两节中，你将学习使用`filter()`以及 [`map()`](https://realpython.com/python-map-function/) 和 [`reduce()`](https://realpython.com/python-reduce-function/) 的基础知识。

### 偶数的平方:`filter()`和`map()`

有时你需要获取一个 iterable，用一个**转换函数**处理它的每一个项，并用结果项生成一个新的 iterable。那样的话，可以用`map()`。该函数具有以下签名:

```py
map(function, iterable[, iterable1, ..., iterableN])
```

论点是这样的:

1.  **`function`** 掌握着转换功能。这个函数应该接受和传递给`map()`的 iterables 一样多的参数。
2.  **`iterable`** 掌握着一条巨蟒。注意，您可以向`map()`提供几个 iterables，但这是可选的。

`map()`将`function`应用于`iterable`中的每一项，将其转换为具有附加功能的不同值。然后`map()`按需产生每个转换的项目。

为了说明如何使用`filter()`和`map()`，假设您需要计算给定列表中所有偶数的平方值。在这种情况下，您可以使用`filter()`提取偶数，然后使用`map()`计算平方值:

>>>

```py
>>> numbers = [1, 3, 10, 45, 6, 50]

>>> def is_even(number):
...     return number % 2 == 0
...

>>> even_numbers = list(filter(is_even, numbers))
>>> even_numbers
[10, 6, 50]

>>> list(map(lambda n: n ** 2, even_numbers))
[100, 36, 2500]

>>> list(map(lambda n: n ** 2, filter(is_even, numbers)))
[100, 36, 2500]
```

首先，您使用`filter()`和`is_even()`得到偶数，就像您到目前为止所做的一样。然后用一个接受一个数字并返回其平方值的`lambda`函数调用`map()`。对`map()`的调用将`lambda`函数应用于`even_numbers`中的每个数字，因此您得到一个平方偶数的列表。最后一个例子展示了如何在一个表达式中组合`filter()`和`map()`。

[*Remove ads*](/account/join/)

### 偶数之和:`filter()`和`reduce()`

Python 中的另一个函数式编程工具是`reduce()`。与`filter()`和`map()`仍然是内置函数不同，`reduce()`被移到了 [`functools`](https://docs.python.org/3/library/functools.html#module-functools) 模块。当您需要将一个函数应用于一个 iterable 并将它简化为一个累积值时，这个函数非常有用。这种操作通常被称为[缩小或折叠](https://en.wikipedia.org/wiki/Fold_(higher-order_function))。

`reduce()`的签名是这样的:

```py
reduce(function, iterable, initial)
```

这些论点的意思是:

1.  **`function`** 保存任何接受两个参数并返回一个值的 Python 可调用函数。
2.  **`iterable`** 容纳任何可迭代的 Python。
3.  **`initial`** 保存一个值，作为第一次部分计算或归约的起点。这是一个可选参数。

对`reduce()`的调用通过将`function`应用于`iterable`中的前两项开始。这样，它计算第一个累积结果，称为**累加器**。然后`reduce()`使用累加器和`iterable`中的第三项计算下一个累加结果。该过程继续，直到函数返回单个值。

如果向`initial`提供一个值，那么`reduce()`使用`initial`和`iterable`的第一项运行第一个部分计算。

下面是一个结合`filter()`和`reduce()`来累计计算列表中所有偶数的总和的例子:

>>>

```py
>>> from functools import reduce
>>> numbers = [1, 3, 10, 45, 6, 50]

>>> def is_even(number):
...     return number % 2 == 0
...

>>> even_numbers = list(filter(is_even, numbers))
>>> reduce(lambda a, b: a + b, even_numbers)
66

>>> reduce(lambda a, b: a + b, filter(is_even, numbers))
66
```

这里，对`reduce()`的第一次调用计算了`filter()`提供的所有偶数的总和。为此，`reduce()`使用了一个`lambda`函数，一次将两个数字相加。

最后一个例子展示了如何链接`filter()`和`reduce()`来产生与之前相同的结果。

## 用`filterfalse()` 过滤可重复项

在 [`itertools`](https://realpython.com/python-itertools/) 中，你会发现一个叫做 [`filterfalse()`](https://docs.python.org/3/library/itertools.html#itertools.filterfalse) 的函数，它执行`filter()`的逆运算。它将 iterable 作为参数，并返回一个新的迭代器，该迭代器产生决策函数返回 false 结果的项目。如果你使用`None`作为`filterfalse()`的第一个参数，那么你会得到错误的条目。

拥有`filterfalse()`功能的意义在于促进**代码重用**。如果您已经有了一个决策函数，那么您可以使用它和`filterfalse()`来获得被拒绝的项目。这使您不必编写逆决策函数。

在接下来的部分中，您将编写一些示例，展示如何利用`filterfalse()`来重用现有的决策函数并继续进行一些过滤。

### 提取奇数

您已经编写了一个名为`is_even()`的谓词函数来检查一个数字是否是偶数。有了这个函数和`filterfalse()`的帮助，您可以构建一个迭代器，它可以产生奇数，而不必编写一个`is_odd()`函数:

>>>

```py
>>> from itertools import filterfalse
>>> numbers = [1, 3, 10, 45, 6, 50]

>>> def is_even(number):
...     return number % 2 == 0
...

>>> list(filterfalse(is_even, numbers))
[1, 3, 45]
```

在这个例子中，`filterfalse()`返回一个迭代器，它从输入迭代器中产生奇数。注意，对`filterfalse()`的调用是简单易懂的。

[*Remove ads*](/account/join/)

### 过滤掉 NaN 值

有时，当您使用[浮点运算](https://en.wikipedia.org/wiki/Floating-point_arithmetic)时，您可能会遇到 [NaN(不是一个数字)](https://en.wikipedia.org/wiki/NaN)值的问题。例如，假设您正在计算包含 NaN 值的数据样本的平均值。如果您使用 Python 的`statistics`模块进行计算，那么您会得到以下结果:

>>>

```py
>>> import statistics as st

>>> sample = [10.1, 8.3, 10.4, 8.8, float("nan"), 7.2, float("nan")]
>>> st.mean(sample)
nan
```

在这个例子中，对`mean()`的调用返回`nan`，这不是您能得到的最有价值的值。NaN 值可以有不同的来源。它们可能是由于无效输入、损坏的数据等原因造成的。您应该在应用程序中找到正确的策略来处理它们。一种替代方法是将它们从数据中删除。

[`math`模块](https://realpython.com/python-math-module/)提供了一个方便的函数 [`isnan()`](https://docs.python.org/3/library/math.html#math.isnan) 可以帮你解决这个问题。该函数将数字`x`作为参数，如果`x`是 NaN，则返回`True`，否则返回`False`。您可以使用该函数在`filterfalse()`调用中提供过滤标准:

>>>

```py
>>> import math
>>> import statistics as st
>>> from itertools import filterfalse

>>> sample = [10.1, 8.3, 10.4, 8.8, float("nan"), 7.2, float("nan")]

>>> st.mean(filterfalse(math.isnan, sample))
8.96
```

将`math.isnan()`与`filterfalse()`一起使用允许您从平均值计算中排除所有 NaN 值。注意，过滤之后，对`mean()`的调用返回一个值，该值提供了对样本数据的更好描述。

## Pythonic 风格编码

尽管`map()`、`filter()`和`reduce()`在 Python 生态系统中已经存在很长时间了，但是**列表理解**和**生成器表达式**已经成为 Python 几乎所有用例中强大的竞争对手。

这些函数提供的功能几乎总是使用生成器表达式或列表理解来更明确地表达。在接下来的两节中，您将学习如何用列表理解或生成器表达式替换对`filter()`的调用。这种替换将使您的代码更加 Pythonic 化。

### 用列表理解替换`filter()`

您可以使用以下模式将对`filter()`的调用快速替换为等价的列表理解:

```py
# Generating a list with filter()
list(filter(function, iterable))

# Generating a list with a list comprehension
[item for item in iterable if function(item)]
```

在这两种情况下，最终目的都是创建一个列表对象。列表理解方法比其等价的`filter()`构造更明确。快速阅读理解可以揭示迭代以及`if`子句中的过滤功能。

使用 list comprehensions 而不是`filter()`可能是当今大多数 Python 开发人员的做法。然而，与`filter()`相比，列表理解有一些缺点。最显著的一个就是懒评的缺失。此外，当开发人员开始阅读使用`filter()`的代码时，他们立即知道代码正在执行过滤操作。然而，这在使用列表理解的代码中并不明显。

在将`filter()`构造转化为列表理解时，需要注意的一个细节是，如果将`None`传递给`filter()`的第一个参数，那么等价的列表理解如下所示:

```py
# Generating a list with filter() and None
list(filter(None, iterable))

# Equivalent list comprehension
[item for item in iterable if item]
```

在这种情况下，list comprehension 中的`if`子句测试`item`的真值。这个测试遵循您已经看到的关于真值的标准 Python 规则。

下面是一个用列表理解替换`filter()`来构建偶数列表的例子:

>>>

```py
>>> numbers = [1, 3, 10, 45, 6, 50]

>>> # Filtering function
>>> def is_even(x):
...     return x % 2 == 0
...

>>> # Use filter()
>>> list(filter(is_even, numbers))
[10, 6, 50]

>>> # Use a list comprehension
>>> [number for number in numbers if is_even(number)]
[10, 6, 50]
```

在这个例子中，您可以看到列表理解变体更加明确。它读起来几乎像简单的英语。列表理解解决方案还避免了必须调用`list()`来构建最终列表。

[*Remove ads*](/account/join/)

### 用生成器表达式替换`filter()`

对`filter()`的自然替换是一个**生成器表达式**。这是因为`filter()`返回一个迭代器，它像生成器表达式一样按需生成条目。众所周知，Python 迭代器是内存高效的。这就是为什么`filter()`现在返回一个迭代器而不是一个列表。

下面是如何使用生成器表达式来编写上一节中的示例:

>>>

```py
>>> numbers = [1, 3, 10, 45, 6, 50]

>>> # Filtering function
>>> def is_even(x):
...     return x % 2 == 0
...

>>> # Use filter()
>>> even_numbers = filter(is_even, numbers)
>>> even_numbers
<filter object at 0x7f58691de4c0>
>>> list(even_numbers)
[10, 6, 50]

>>> # Use a generator expression
>>> even_numbers = (number for number in numbers if is_even(number))
>>> even_numbers
<generator object <genexpr> at 0x7f586ade04a0>
>>> list(even_numbers)
[10, 6, 50]
```

就内存消耗而言，生成器表达式与调用`filter()`一样有效。这两个工具都返回按需生成项目的迭代器。使用任何一种都可能是品味、方便或风格的问题。所以，你说了算！

## 结论

Python 的`filter()`允许你对 iterables 执行**过滤**操作。这种操作包括将一个**布尔函数**应用于 iterable 中的项目，并只保留那些函数返回真结果的值。通常，您可以使用`filter()`来处理现有的 iterables，并生成包含您当前需要的值的新 iterables。

**在本教程中，您学习了如何:**

*   用 Python 创作的 **`filter()`**
*   使用`filter()`到**过程变量**并保持您需要的值
*   将`filter()`与 **`map()`** 和 **`reduce()`** 结合起来处理不同的问题
*   将`filter()`替换为**列表理解**和**生成器表达式**

有了这些新的知识，你现在可以在你的代码中使用`filter()`，给它一个[函数风格](https://realpython.com/learning-paths/functional-programming/)。您也可以切换到更 Pythonic 化的风格，用[列表理解](https://realpython.com/list-comprehension-python/)或[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)替换`filter()`。******