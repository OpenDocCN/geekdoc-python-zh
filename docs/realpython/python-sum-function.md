# Python 的 sum():对值求和的 python 方式

> 原文：<https://realpython.com/python-sum-function/>

Python 的内置函数`sum()`是一种高效且[的 python 式](https://realpython.com/learning-paths/writing-pythonic-code/)方法来对一系列数值求和。将几个数字相加是许多计算中常见的中间步骤，因此对于 Python 程序员来说,`sum()`是一个非常方便的工具。

作为一个额外的有趣用例，您可以使用`sum()`连接[列表和元组](https://realpython.com/python-lists-tuples/)，这在您需要简化列表列表时会很方便。

**在本教程中，您将学习如何:**

*   使用**通用技术和工具**手工计算数值总和
*   使用 **Python 的`sum()`** 高效地将几个数值相加
*   **用`sum()`连接列表和元组**
*   用`sum()`处理常见的**求和问题**
*   为`sum()`中的**参数**使用合适的值
*   在`sum()`和**之间选择替代工具**来求和并连接对象

这些知识将帮助您使用`sum()`或其他替代和专门的工具有效地处理和解决代码中的求和问题。

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 理解求和问题

将数值相加是编程中相当常见的问题。例如，假设您有一个数字列表[1，2，3，4，5]，并希望将它们相加以计算它们的总和。使用标准算术，您将做类似这样的事情:

1 + 2 + 3 + 4 + 5 = 15

就数学而言，这个表达式非常简单。它会引导您完成一系列简短的加法运算，直到您找到所有数字的总和。

手动进行这种特殊的计算是可能的，但是想象一下在其他一些情况下这可能是不可能的。如果您有一个特别长的数字列表，手动添加可能效率低下且容易出错。如果你甚至不知道列表中有多少项，会发生什么？最后，设想一个场景，您需要添加的项目数量动态或不可预测地变化。

在这种情况下，无论你有一长串或短串的[数字](https://realpython.com/python-numbers/)，Python 对于解决**求和问题**都非常有用。

如果你想通过从头开始创建自己的解决方案来对数字求和，那么你可以尝试使用一个 [`for`循环](https://realpython.com/python-for-loop/):

>>>

```py
>>> numbers = [1, 2, 3, 4, 5]
>>> total = 0

>>> for number in numbers:
...     total += number
...

>>> total
15
```

这里，首先创建`total`，并将其初始化为`0`。这个[变量](https://realpython.com/python-variables/)作为一个[累加器](https://en.wikipedia.org/wiki/Accumulator_(computing))工作，你在其中存储中间结果直到你得到最终结果。循环遍历`numbers`，并通过使用[增加赋值](https://docs.python.org/3/reference/simple_stmts.html#augmented-assignment-statements)累加每个连续值来更新`total`。

你也可以在一个[函数](https://realpython.com/defining-your-own-python-function/)中包装`for`循环。这样，您可以为不同的列表重用代码:

>>>

```py
>>> def sum_numbers(numbers):
...     total = 0
...     for number in numbers:
...         total += number
...     return total
...

>>> sum_numbers([1, 2, 3, 4, 5])
15

>>> sum_numbers([])
0
```

在`sum_numbers()`中，您将一个[可迭代](https://realpython.com/python-for-loop/#iterables)——具体来说，是一个数值列表——作为一个参数，然后[返回](https://realpython.com/python-return-statement/)输入列表中值的总和。如果输入列表为空，那么函数返回`0`。这个`for`循环就是你之前看到的那个。

也可以用[递归](https://realpython.com/python-recursion/)代替迭代。递归是一种[函数式编程](https://realpython.com/python-functional-programming/)技术，在这种技术中，函数在其自己的定义中被调用。换句话说，递归函数在循环中调用自己:

>>>

```py
>>> def sum_numbers(numbers):
...     if len(numbers) == 0:
...         return 0
...     return numbers[0] + sum_numbers(numbers[1:])
...

>>> sum_numbers([1, 2, 3, 4, 5])
15
```

当你定义一个递归函数时，你有陷入无限循环的风险。为了防止这种情况，您需要定义一个停止递归的**基本用例**和一个调用函数并开始隐式循环的**递归用例**。

在上面的例子中，基本情况意味着零长度列表的总和是`0`。递归情况意味着总和是第一个值`numbers[0]`，加上其余值的总和`numbers[1:]`。因为递归情况在每次迭代中使用较短的序列，所以当`numbers`是一个零长度列表时，您可能会遇到基本情况。作为最终结果，您得到了输入列表中所有条目的总和，`numbers`。

**注意:**在这个例子中，如果你不检查一个空的输入列表(你的基本情况)，那么`sum_numbers()`将永远不会进入一个无限的递归循环。当您的`numbers`列表长度达到`0`时，代码试图从空列表中访问一个项目，这将引发一个`IndexError`并中断循环。

使用这种实现，你永远不会从这个函数中得到一个和。你每次都会得到一个`IndexError`。

Python 中对一列数字求和的另一种选择是从 [`functools`](https://docs.python.org/3/library/functools.html#module-functools) 中使用 [`reduce()`](https://realpython.com/python-reduce-function/) 。要获得一系列数字的总和，您可以将 [`operator.add`](https://docs.python.org/3/library/operator.html#operator.add) 或适当的 [`lambda`函数](https://realpython.com/python-lambda/)作为第一个参数传递给`reduce()`:

>>>

```py
>>> from functools import reduce
>>> from operator import add

>>> reduce(add, [1, 2, 3, 4, 5])
15

>>> reduce(add, [])
Traceback (most recent call last):
    ...
TypeError: reduce() of empty sequence with no initial value

>>> reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])
15
```

你可以调用`reduce()`加一个还原，或者[折叠](https://en.wikipedia.org/wiki/Fold_(higher-order_function))，`function`加一个`iterable`作为参数。然后`reduce()`使用输入函数处理`iterable`并返回一个累积值。

在第一个例子中，归约函数是`add()`，它取两个数并将它们相加。最终结果是输入`iterable`中数字的总和。作为一个缺点，`reduce()`用空`iterable`调用时会引出一个 [`TypeError`](https://realpython.com/python-traceback/#typeerror) 。

在第二个例子中，reduction 函数是一个返回两个数相加的`lambda`函数。

由于像这样的求和在编程中很常见，所以每次需要对一些数字求和时都要编写一个新函数，这是一项重复的工作。此外，使用`reduce()`并不是最容易理解的解决方案。

Python 提供了一个专用的内置函数来解决这个问题。该功能被方便地称为 [`sum()`](https://docs.python.org/3/library/functions.html#sum) 。因为它是一个内置函数，你可以直接在你的代码中使用它，而不需要[导入](https://realpython.com/python-import/)任何东西。

[*Remove ads*](/account/join/)

## Python 的`sum()` 入门

可读性是 Python 哲学背后最重要的原则之一。当对一列值求和时，想象你要求一个循环做什么。您希望它循环遍历一些数字，将它们累积在一个中间变量中，并返回最终的和。然而，你或许可以想象一个不需要循环的可读性更好的求和版本。你想让 Python 取一些数字，然后把它们加起来。

现在想想`reduce()`是如何求和的。使用`reduce()`可能比基于循环的解决方案可读性更差，也更不直接。

这就是为什么 [Python 2.3](https://docs.python.org/3/whatsnew/2.3.html) 添加了`sum()`作为内置函数，为求和问题提供 Python 式的解决方案。[亚历克斯·马尔泰利](https://en.wikipedia.org/wiki/Alex_Martelli)贡献了这个函数，它现在是对一系列值求和的首选语法:

>>>

```py
>>> sum([1, 2, 3, 4, 5])
15

>>> sum([])
0
```

哇！很整洁，不是吗？它读起来像简单的英语，清楚地传达了你在输入列表上执行的动作。使用`sum()`比使用`for`循环或`reduce()`调用更具可读性。与`reduce()`不同，`sum()`不会在你提供一个空的 iterable 时抛出`TypeError`。相反，它可以理解地返回`0`。

您可以使用以下两个参数调用`sum()`:

1.  **`iterable`** 是必选参数，可以容纳任何 Python iterable。iterable 通常包含数值，但也可以包含[列表或元组](https://realpython.com/python-lists-tuples/)。
2.  **`start`** 是可选参数，可以保存初始值。然后将该值添加到最终结果中。默认为`0`。

在内部，`sum()`从左到右将`start`加上`iterable`中的值相加。输入`iterable`中的值通常是数字，但是您也可以使用列表和元组。可选参数`start`可以接受一个数字、列表或元组，这取决于传递给`iterable`的内容。它不能带一个[字符串](https://realpython.com/python-strings/)。

在接下来的两节中，您将学习在代码中使用`sum()`的基本知识。

### `iterable`所需参数:

接受任何 Python iterable 作为它的第一个参数使得`sum()`通用、可重用并且[多态](https://en.wikipedia.org/wiki/Polymorphism_(computer_science))。由于这个特性，您可以将`sum()`与列表、元组、[集、](https://realpython.com/python-sets/) [`range`](https://realpython.com/python-range/) 对象和[字典](https://realpython.com/python-dicts/)一起使用:

>>>

```py
>>> # Use a list
>>> sum([1, 2, 3, 4, 5])
15

>>> # Use a tuple
>>> sum((1, 2, 3, 4, 5))
15

>>> # Use a set
>>> sum({1, 2, 3, 4, 5})
15

>>> # Use a range
>>> sum(range(1, 6))
15

>>> # Use a dictionary
>>> sum({1: "one", 2: "two", 3: "three"})
6
>>> sum({1: "one", 2: "two", 3: "three"}.keys())
6
```

在所有这些例子中，`sum()`计算输入 iterable 中所有值的算术和，而不考虑它们的类型。在两个字典示例中，对`sum()`的两个调用都返回输入字典的键的总和。第一个例子默认情况下对键求和，第二个例子对键求和是因为输入字典上的 [`.keys()`](https://docs.python.org/3/library/stdtypes.html#dict.keys) 调用。

如果您的字典在其值中存储了数字，并且您想要对这些值求和而不是对键求和，那么您可以使用 [`.values()`](https://docs.python.org/3/library/stdtypes.html#dict.values) 来实现这一点，就像在`.keys()`示例中一样。

你也可以使用`sum()`和[列表理解](https://realpython.com/list-comprehension-python/)作为论元。下面是一个计算一系列值的平方和的示例:

>>>

```py
>>> sum([x ** 2 for x in range(1, 6)])
55
```

[Python 2.4](https://docs.python.org/3/whatsnew/2.4.html#what-s-new-in-python-2-4) 在语言中增加了[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)。同样，当您使用生成器表达式作为参数时，`sum()`会按预期工作:

>>>

```py
>>> sum(x ** 2 for x in range(1, 6))
55
```

这个例子展示了处理求和问题的一个最重要的技巧。它在一行代码中提供了一个优雅、易读、高效的解决方案。

[*Remove ads*](/account/join/)

### `start`可选参数:

第二个也是可选的参数`start`，允许您提供一个值来初始化求和过程。当您需要按顺序处理累积值时，此参数很方便:

>>>

```py
>>> sum([1, 2, 3, 4, 5], 100)  # Positional argument
115

>>> sum([1, 2, 3, 4, 5], start=100)  # Keyword argument
115
```

这里，您提供一个初始值`100`到`start`。实际效果是`sum()`将这个值添加到输入 iterable 中值的累积和中。注意，您可以提供`start`作为[位置参数](https://realpython.com/defining-your-own-python-function/#positional-arguments)或[关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-arguments)。后一种选择更加清晰易读。

如果你不给`start`提供一个值，那么它默认为`0`。默认值`0`确保了返回输入值总和的预期行为。

## 对数值求和

`sum()`的主要目的是提供一种将数值相加的 Pythonic 方式。到目前为止，您已经看到了如何使用函数对整数求和。此外，您可以将`sum()`与任何其他数字 Python 类型一起使用，例如 [`float`](https://realpython.com/python-numbers/#floating-point-numbers) 、 [`complex`](https://realpython.com/python-complex-numbers/) 、 [`decimal.Decimal`](https://docs.python.org/3/library/decimal.html#decimal.Decimal) 和 [`fractions.Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction) 。

下面是几个对不同数值类型的值使用`sum()`的例子:

>>>

```py
>>> from decimal import Decimal
>>> from fractions import Fraction

>>> # Sum floating-point numbers
>>> sum([10.2, 12.5, 11.8])
34.5
>>> sum([10.2, 12.5, 11.8, float("inf")])
inf
>>> sum([10.2, 12.5, 11.8, float("nan")])
nan

>>> # Sum complex numbers
>>> sum([3 + 2j, 5 + 6j])
(8+8j)

>>> # Sum Decimal numbers
>>> sum([Decimal("10.2"), Decimal("12.5"), Decimal("11.8")])
Decimal('34.5')

>>> # Sum Fraction numbers
>>> sum([Fraction(51, 5), Fraction(25, 2), Fraction(59, 5)])
Fraction(69, 2)
```

在这里，您首先将`sum()`与**浮点数**一起使用。当您在调用`float("inf")`和`float("nan")`中使用特殊符号`inf`和`nan`时，值得注意函数的行为。第一个符号代表一个**无穷大的**值，所以`sum()`返回`inf`。第二个符号代表 [NaN(非数字)](https://en.wikipedia.org/wiki/NaN)值。由于不能将数字和非数字相加，结果得到`nan`。

其他例子对`complex`、`Decimal`和`Fraction`数的可迭代项求和。在所有情况下，`sum()`使用适当的数值类型返回结果的累积和。

## 串联序列

尽管`sum()`主要用于操作数值，但是您也可以使用该函数来连接序列，比如列表和元组。为此，您需要为`start`提供一个合适的值:

>>>

```py
>>> num_lists = [[1, 2, 3], [4, 5, 6]]
>>> sum(num_lists, start=[])
[1, 2, 3, 4, 5, 6]

>>> # Equivalent concatenation
>>> [1, 2, 3] + [4, 5, 6]
[1, 2, 3, 4, 5, 6]

>>> num_tuples = ((1, 2, 3), (4, 5, 6))
>>> sum(num_tuples, start=())
(1, 2, 3, 4, 5, 6)

>>> # Equivalent concatenation
>>> (1, 2, 3) + (4, 5, 6)
(1, 2, 3, 4, 5, 6)
```

在这些例子中，您使用`sum()`来连接列表和元组。这是一个有趣的特性，您可以使用它来扁平化一个列表列表或一个元组。这些例子工作的关键要求是为`start`选择一个合适的值。例如，如果你想连接列表，那么`start`需要保存一个列表。

在上面的例子中，`sum()`在内部执行连接操作，所以它只处理那些支持连接的序列类型，字符串除外:

>>>

```py
>>> num_strs = ["123", "456"]
>>> sum(num_strs, "0")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: sum() can't sum strings [use ''.join(seq) instead]
```

当您试图使用`sum()`来连接字符串时，您会得到一个`TypeError`。正如异常消息所建议的，在 Python 中应该使用 [`str.join()`](https://docs.python.org/3/library/stdtypes.html#str.join) 来连接字符串。稍后当您到达使用`sum()` 的替代物的[部分时，您将看到使用这种方法的例子。](#using-alternatives-to-sum)

## 用 Python 的`sum()`练习

到目前为止，您已经学习了使用`sum()`的基本知识。您已经学习了如何使用这个函数将数值相加，以及连接序列，如列表和元组。

在这一节中，您将看到更多关于何时以及如何在代码中使用`sum()`的例子。通过这些实际的例子，您将了解到，当您执行计算时，这个内置函数是非常方便的，因为您需要将计算一系列数字的和作为中间步骤。

您还将了解到`sum()`在处理列表和元组时会很有帮助。您将看到的一个特殊例子是当您需要展平一系列列表时。

[*Remove ads*](/account/join/)

### 计算累计金额

您要编写的第一个例子是关于如何利用`start`参数对数值的累积列表求和。

假设你正在开发一个系统来管理一个给定产品在几个不同销售点的销售。每天，您都会从每个销售点获得一份售出单位报告。您需要系统地计算累计金额，以了解整个公司在一周内销售了多少台设备。要解决这个问题，可以使用`sum()`:

>>>

```py
>>> cumulative_sales = 0

>>> monday = [50, 27, 42]
>>> cumulative_sales = sum(monday, start=cumulative_sales)
>>> cumulative_sales
119

>>> tuesday = [12, 32, 15]
>>> cumulative_sales = sum(tuesday, start=cumulative_sales)
>>> cumulative_sales
178

>>> wednesday = [20, 24, 42]
>>> cumulative_sales = sum(wednesday, start=cumulative_sales)
>>> cumulative_sales
264
 ...
```

通过使用`start`，您可以设置一个初始值来初始化总和，这允许您将连续的单位添加到先前计算的小计中。在这个周末，你会得到公司的总销售量。

### 计算样本的平均值

`sum()`的另一个实际使用案例是在做进一步计算之前，将其作为中间计算。例如，假设您需要计算一个数值样本的算术平均值[和](https://en.wikipedia.org/wiki/Arithmetic_mean)。算术平均值，也称为**平均值**，是样本中数值的总和除以数值的个数，即[个数据点](https://en.wikipedia.org/wiki/Unit_of_observation#Data_point)。

如果你有样本[2，3，4，2，3，6，4，2]并且你想手工计算算术平均值，那么你可以解这个运算:

(2 + 3 + 4 + 2 + 3 + 6 + 4 + 2) / 8 = 3.25

如果你想通过使用 Python 来加速这个过程，你可以把它分成两部分。计算的第一部分，也就是把数字加在一起，是`sum()`的任务。操作的下一部分是除以 8，使用样本中的数字计数。要计算你的[除数](https://en.wikipedia.org/wiki/Divisor)，可以用 [`len()`](https://realpython.com/len-python-function/) :

>>>

```py
>>> data_points = [2, 3, 4, 2, 3, 6, 4, 2]

>>> sum(data_points) / len(data_points)
3.25
```

这里，对`sum()`的调用计算样本中数据点的总和。接下来，您使用`len()`来获得数据点的数量。最后，执行所需的除法来计算样本的算术平均值。

实际上，您可能希望将此代码转换为具有一些附加功能的函数，例如描述性名称和空样本检查:

>>>

```py
>>> # Python >= 3.8

>>> def average(data_points):
...     if (num_points := len(data_points)) == 0:
...         raise ValueError("average requires at least one data point")
...     return sum(data_points) / num_points
...

>>> average([2, 3, 4, 2, 3, 6, 4, 2])
3.25

>>> average([])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in average
ValueError: average requires at least one data point
```

在`average()`中，首先检查输入样本是否有数据点。如果没有，那么您将引发一个带有描述性消息的`ValueError`。在这个例子中，您使用 [walrus 操作符](https://realpython.com/python38-new-features/#the-walrus-in-the-room-assignment-expressions)将数据点的数量存储在变量`num_points`中，这样您就不需要再次调用`len()`。[返回语句](https://realpython.com/python-return-statement/)计算样本的算术平均值，并将其发送回调用代码。

**注:**计算一个数据样本的均值是统计和数据分析中的常见操作。Python 标准库提供了一个名为 [`statistics`](https://docs.python.org/3/library/statistics.html#module-statistics) 的便捷模块来处理这类计算。

在`statistics`模块中，你会发现一个名为 [`mean()`](https://docs.python.org/3/library/statistics.html#statistics.mean) 的函数:

>>>

```py
>>> from statistics import mean

>>> mean([2, 3, 4, 2, 3, 6, 4, 2])
3.25

>>> mean([])
Traceback (most recent call last):
    ...
statistics.StatisticsError: mean requires at least one data point
```

`statistics.mean()`函数的行为与您之前编写的`average()`函数非常相似。当你用一个数值样本调用`mean()`时，你将得到输入数据的算术平均值。当您将一个空列表传递给`mean()`时，您将获得一个`statistics.StatisticsError`。

请注意，当您使用适当的样本调用`average()`时，您将获得期望的平均值。如果你用一个空样本调用`average()`，那么你会得到一个预期的`ValueError`。

### 求两个序列的点积

使用`sum()`可以解决的另一个问题是寻找两个等长数值序列的[点积](https://en.wikipedia.org/wiki/Dot_product)。点积是输入序列中每对值的[乘积](https://en.wikipedia.org/wiki/Product_(mathematics))的代数和。例如，如果您有序列(1，2，3)和(4，5，6)，那么您可以使用加法和乘法手动计算它们的点积:

1 × 4 + 2 × 5 + 3 × 6 = 32

要从输入序列中提取连续的值对，可以使用 [`zip()`](https://realpython.com/python-zip-function/) 。然后，您可以使用生成器表达式将每对值相乘。最后，`sum()`可以对乘积求和:

>>>

```py
>>> x_vector = (1, 2, 3)
>>> y_vector = (4, 5, 6)

>>> sum(x * y for x, y in zip(x_vector, y_vector))
32
```

使用`zip()`，您可以用每个输入序列的值生成一个元组列表。生成器表达式在每个元组上循环，同时将先前由`zip()`排列的连续值对相乘。最后一步是使用`sum()`将产品加在一起。

上面例子中的代码是有效的。然而，点积是为长度相等的序列定义的，那么如果提供不同长度的序列会发生什么呢？在这种情况下，`zip()`会忽略最长序列中的额外值，从而导致不正确的结果。

为了处理这种可能性，您可以将对`sum()`的调用包装在一个自定义函数中，并提供对输入序列长度的适当检查:

>>>

```py
>>> def dot_product(x_vector, y_vector):
...     if len(x_vector) != len(y_vector):
...         raise ValueError("Vectors must have equal sizes")
...     return sum(x * y for x, y in zip(x_vector, y_vector))
...

>>> dot_product((1, 2, 3), (4, 5, 6))
32

>>> dot_product((1, 2, 3, 4), (5, 6, 3))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in dot_product
ValueError: Vectors must have equal sizes
```

这里，`dot_product()`以两个序列为自变量，返回它们对应的点积。如果输入序列具有不同的长度，那么该函数产生一个`ValueError`。

将功能嵌入到自定义函数中允许您重用代码。它还让您有机会描述性地命名该函数，以便用户只需阅读其名称就能知道该函数的用途。

[*Remove ads*](/account/join/)

### 展平列表列表

平整列表列表是 Python 中的一项常见任务。假设您有一个列表列表，需要将它展平为一个包含原始嵌套列表中所有项目的列表。在 Python 中，你可以使用几种[方法中的任何一种来展平列表。例如，您可以使用一个`for`循环，如以下代码所示:](https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists)

>>>

```py
>>> def flatten_list(a_list):
...     flat = []
...     for sublist in a_list:
...         flat += sublist
...     return flat
...

>>> matrix = [
...     [1, 2, 3],
...     [4, 5, 6],
...     [7, 8, 9],
... ]

>>> flatten_list(matrix)
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

在`flatten_list()`内部，循环遍历包含在`a_list`中的所有嵌套列表。然后，它使用一个增强的赋值操作(`+=`)在`flat`中将它们连接起来。因此，您将获得一个包含原始嵌套列表中所有项目的平面列表。

但是坚持住！在本教程中，你已经学会了如何使用`sum()`来连接序列。你能像上面的例子一样使用这个特性来展平列表吗？是啊！方法如下:

>>>

```py
>>> matrix = [
...     [1, 2, 3],
...     [4, 5, 6],
...     [7, 8, 9],
... ]

>>> sum(matrix, [])
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

真快！单行代码和`matrix`现在是一个平面列表。然而，使用`sum()`似乎不是最快的解决方案。

任何包含串联的解决方案的一个重要缺点是，在幕后，每个中间步骤都会创建一个新列表。就内存使用而言，这可能是相当浪费的。最终返回的列表只是在每一轮连接中创建的所有列表中最近创建的列表。相反，使用列表理解可以确保您只创建和返回一个列表:

>>>

```py
>>> def flatten_list(a_list):
...     return [item for sublist in a_list for item in sublist]
...

>>> matrix = [
...     [1, 2, 3],
...     [4, 5, 6],
...     [7, 8, 9],
... ]

>>> flatten_list(matrix)
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

这个新版本的`flatten_list()`在内存使用方面效率更高，浪费更少。然而，[嵌套的理解](https://realpython.com/list-comprehension-python/#watch-out-for-nested-comprehensions)可能难以阅读和理解。

使用 [`.append()`](https://realpython.com/python-append/) 可能是展平列表列表的可读性最强的方法:

>>>

```py
>>> def flatten_list(a_list):
...     flat = []
...     for sublist in a_list:
...         for item in sublist:
...             flat.append(item)
...     return flat
...

>>> matrix = [
...     [1, 2, 3],
...     [4, 5, 6],
...     [7, 8, 9],
... ]

>>> flatten_list(matrix)
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

在这个版本的`flatten_list()`中，阅读您代码的人可以看到函数在`a_list`中的每个`sublist`上迭代。在第一个`for`循环中，它遍历`sublist`中的每个`item`，最终用`.append()`填充新的`flat`列表。就像前面的理解一样，这个解决方案在这个过程中只创建一个列表。这种解决方案的一个优点是可读性很强。

## 使用`sum()`的替代品

正如您已经了解到的，`sum()`通常有助于处理数值。然而，在处理浮点数时，Python 提供了一个替代工具。在 [`math`](https://realpython.com/python-math-module/) 中，您会发现一个名为 [`fsum()`](https://docs.python.org/3/library/math.html#math.fsum) 的函数，它可以帮助您提高浮点计算的总体精度。

在一个任务中，您可能希望连接或链接几个可重复项，以便可以将它们作为一个整体使用。对于这种场景，可以看看 [`itertools`](https://realpython.com/python-itertools/) 模块的功能 [`chain()`](https://docs.python.org/3/library/itertools.html#itertools.chain) 。

您可能还需要一个任务来连接字符串列表。在本教程中，您已经了解到无法使用`sum()`来连接字符串。这个函数不是为字符串连接而构建的。最 Pythonic 化的替代就是使用 [`str.join()`](https://docs.python.org/3/library/stdtypes.html#str.join) 。

### 浮点数求和:`math.fsum()`

如果您的代码经常用`sum()`对浮点数求和，那么您应该考虑使用`math.fsum()`来代替。这个函数比`sum()`更仔细地执行浮点计算，这提高了计算的精度。

根据其[文档](https://docs.python.org/3/library/math.html#math.fsum)，`fsum()`“通过跟踪多个中间部分和来避免精度损失。”文档提供了以下示例:

>>>

```py
>>> from math import fsum

>>> sum([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])
0.9999999999999999

>>> fsum([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])
1.0
```

用`fsum()`，你得到一个更精确的结果。然而，你应该注意到`fsum()`并没有解决浮点运算中的[表示](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Internal_representation)错误。以下示例揭示了这一限制:

>>>

```py
>>> from math import fsum

>>> sum([0.1, 0.2])
0.30000000000000004

>>> fsum([0.1, 0.2])
0.30000000000000004
```

在这些示例中，两个函数返回相同的结果。这是因为无法用二进制浮点精确表示值`0.1`和`0.2`:

>>>

```py
>>> f"{0.1:.28f}"
'0.1000000000000000055511151231'

>>> f"{0.2:.28f}"
'0.2000000000000000111022302463'
```

然而，与`sum()`不同的是，当您将非常大的数字和非常小的数字相加时，`fsum()`可以帮助您减少浮点误差传播:

>>>

```py
>>> from math import fsum

>>> sum([1e-16, 1, 1e16])
1e+16
>>> fsum([1e-16, 1, 1e16])
1.0000000000000002e+16

>>> sum([1, 1, 1e100, -1e100] * 10_000)
0.0
>>> fsum([1, 1, 1e100, -1e100] * 10_000)
20000.0
```

哇！第二个例子相当令人惊讶，完全击败了`sum()`。有了`sum()`，你得到的结果是`0.0`。这与你用`fsum()`得到的`20000.0`的正确结果相差甚远。

[*Remove ads*](/account/join/)

### 用`itertools.chain()` 串联可重复项

如果您正在寻找一个方便的工具来连接或链接一系列可重复项，那么可以考虑使用来自`itertools`的`chain()`。这个函数可以接受多个可迭代对象，并构建一个[迭代器](https://docs.python.org/3/glossary.html#term-iterator)，从第一个产生项目，从第二个产生项目，以此类推，直到它穷尽所有的输入可迭代对象:

>>>

```py
>>> from itertools import chain

>>> numbers = chain([1, 2, 3], [4, 5, 6], [7, 8, 9])
>>> numbers
<itertools.chain object at 0x7f0d0f160a30>
>>> next(numbers)
1
>>> next(numbers)
2

>>> list(chain([1, 2, 3], [4, 5, 6], [7, 8, 9]))
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

当您调用`chain()`时，您从输入 iterables 中获得一个 iterator。在本例中，您使用 [`next()`](https://docs.python.org/3/library/functions.html#next) 从`numbers`访问连续的项目。如果您想使用列表，那么您可以使用`list()`来使用迭代器并返回一个常规的 Python 列表。

`chain()`也是在 Python 中展平列表列表的一个好选项:

>>>

```py
>>> from itertools import chain

>>> matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

>>> list(chain(*matrix))
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

要用`chain()`展平一个列表列表，需要使用**可迭代解包操作符** ( `*`)。该操作符解包所有输入的可迭代对象，以便`chain()`可以处理它们并生成相应的迭代器。最后一步是调用`list()`来构建所需的平面列表。

### 用`str.join()`和连接字符串

正如你已经看到的，`sum()`没有[连接或加入](https://realpython.com/python-string-split-concatenate-join/)字符串。如果您需要这样做，那么 Python 中首选且最快的工具是`str.join()`。此方法将一系列字符串作为参数，并返回一个新的串联字符串:

>>>

```py
>>> greeting = ["Hello,", "welcome to", "Real Python!"]

>>> " ".join(greeting)
'Hello, welcome to Real Python!'
```

使用`.join()`是连接字符串的最有效的方式。这里，您使用一个字符串列表作为参数，并从输入中构建一个单独的字符串。注意`.join()`在连接过程中使用调用方法的字符串作为分隔符。在这个例子中，您在由一个空格字符(`" "`)组成的字符串上调用`.join()`，所以来自`greeting`的原始字符串在您的最终字符串中由空格分隔。

## 结论

现在可以使用 Python 的内置函数 [`sum()`](https://docs.python.org/3/library/functions.html#sum) 将多个数值相加在一起。这个函数提供了一种有效的、可读的、Pythonic 式的方法来解决代码中的**求和问题**。如果您正在处理需要对数值求和的数学计算，那么`sum()`可以成为您的救命稻草。

**在本教程中，您学习了如何:**

*   使用**通用技术和工具**对数值求和
*   使用 **Python 的`sum()`** 有效地添加几个数值
*   **使用`sum()`连接序列**
*   用`sum()`处理常见的**求和问题**
*   为`sum()`中的的 **`iterable`和`start`参数使用合适的值**
*   在`sum()`和**之间选择替代工具**来求和并连接对象

有了这些知识，您现在能够以一种 Pythonic 式的、可读的、高效的方式将多个数值相加。*****