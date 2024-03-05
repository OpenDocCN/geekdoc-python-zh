# Python return 语句:用法和最佳实践

> 原文：<https://realpython.com/python-return-statement/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**有效使用 Python 返回语句**](/courses/effective-python-return-statement/)

Python [`return`语句](https://en.wikipedia.org/wiki/Return_statement)是[函数](https://realpython.com/defining-your-own-python-function/)和[方法](https://realpython.com/python3-object-oriented-programming/#instance-methods)的关键组件。您可以使用`return`语句让您的函数将 Python 对象发送回调用者代码。这些对象称为函数的**返回值**。您可以使用它们在您的程序中执行进一步的计算。

如果你想编写具有[python 式](https://realpython.com/learning-paths/writing-pythonic-code/)和健壮的定制函数，有效地使用`return`语句是一项核心技能。

在本教程中，您将学习:

*   如何在函数中使用 **Python `return`语句**
*   如何从函数中返回**单值**或**多值**
*   使用`return`语句时，要遵循哪些**最佳实践**

有了这些知识，您将能够用 Python 编写更具可读性、可维护性和简洁的函数。如果你对 Python 函数完全陌生，那么在深入本教程之前，你可以查看定义你自己的 Python 函数的[。](https://realpython.com/defining-your-own-python-function/)

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 函数入门

大多数编程语言都允许您为执行具体计算的代码块指定一个名称。这些命名的代码块可以快速重用，因为您可以使用它们的名称从代码中的不同位置调用它们。

程序员将这些命名的代码块称为 [**子程序**](https://en.wikipedia.org/wiki/Subroutine) 、**例程**、**过程**或**函数**，这取决于它们使用的语言。在某些语言中，例程或过程和函数之间有明显的区别。

有时这种差别如此之大，以至于您需要使用一个特定的关键字来定义一个过程或子例程，而使用另一个关键字来定义一个函数。例如， [Visual Basic](https://en.wikipedia.org/wiki/Visual_Basic) 编程语言使用`Sub`和`Function`来区分这两者。

一般来说，**过程**是一个命名的代码块，它执行一组动作而不计算最终值或结果。另一方面，**函数**是一个命名的代码块，它执行一些动作，目的是计算最终值或结果，然后将结果发送回调用者代码。过程和函数都可以作用于一组**输入值**，通常称为**参数**。

在 Python 中，这类命名代码块被称为[函数](https://realpython.com/defining-your-own-python-function/)，因为它们总是向调用者发回一个值。Python 文档对函数的定义如下:

> 向调用者返回一些值的一系列语句。也可以向其传递零个或多个可能在主体执行中使用的[参数](https://docs.python.org/3/glossary.html#term-argument)。([来源](https://docs.python.org/3/glossary.html#term-function)

尽管官方文档声明函数“向调用者返回一些值”，但是您很快就会看到函数可以向调用者代码返回任何 Python 对象。

一般来说，一个函数**带**个参数(如果有的话)，**执行**一些操作，**返回**一个值(或者对象)。函数返回给调用者的值通常被称为函数的**返回值**。所有 Python 函数都有一个返回值，或者是显式的，或者是隐式的。在本教程的后面，您将讨论显式返回值和隐式返回值之间的区别。

要编写一个 Python 函数，您需要一个以 [`def`关键字](https://realpython.com/defining-your-own-python-function/)开头的**头**，后面是函数名、一个可选的逗号分隔参数列表(在一对必需的括号内)和一个最后的冒号。

函数的第二个组成部分是它的**代码块**，或**体**。Python 使用[缩进](https://docs.python.org/3/reference/lexical_analysis.html#indentation)而不是括号、`begin`和`end`关键字等来定义代码块。因此，要在 Python 中定义函数，可以使用以下语法:

```py
def function_name(arg1, arg2,..., argN):
    # Function's code goes here...
    pass
```

当您编写 Python 函数时，您需要定义一个带有关键字`def`的头、函数名和括号中的参数列表。注意，参数列表是可选的，但是括号在语法上是必需的。然后您需要定义函数的代码块，它将开始向右缩进一级。

在上面的例子中，你使用了一个 [`pass`语句](https://realpython.com/python-pass/)。当您需要在代码中使用占位符语句来确保语法正确，但不需要执行任何操作时，这种语句非常有用。`pass`语句也被称为**空操作**，因为它们不执行任何操作。

**注意:**定义函数及其参数的完整语法超出了本教程的范围。关于这个主题的深入资源，请查看[定义自己的 Python 函数](https://realpython.com/defining-your-own-python-function/)。

要使用一个函数，你需要调用它。函数调用由函数名和括号中的函数参数组成:

```py
function_name(arg1, arg2, ..., argN)
```

只有在函数需要时，才需要将参数传递给函数调用。另一方面，圆括号在函数调用中总是必需的。如果你忘记了它们，那么你就不能调用这个函数，而是作为一个函数对象来引用它。

为了让你的函数返回值，你需要使用 [Python `return`语句](https://docs.python.org/3/reference/simple_stmts.html#grammar-token-return_stmt)。这就是从现在开始你要报道的内容。

[*Remove ads*](/account/join/)

## 理解 Python `return`语句

**Python `return`语句**是一个特殊的语句，您可以在函数或[方法](https://realpython.com/python3-object-oriented-programming/#instance-methods)内部使用它来将函数的结果发送回调用者。一条`return`语句由 [`return`关键字](https://realpython.com/python-keywords/#returning-keywords-return-yield)和可选的**返回值**组成。

Python 函数的返回值可以是任何 Python 对象。Python 中的一切都是对象。所以，你的函数可以返回数值( [`int`](https://docs.python.org/3/library/functions.html#int) ， [`float`](https://docs.python.org/3/library/functions.html#float) ， [`complex`](https://docs.python.org/3/library/functions.html#complex) 值)，对象的集合和序列( [`list`，`tuple`](https://realpython.com/courses/lists-tuples-python/) ， [`dictionary`](https://realpython.com/courses/lists-tuples-python/) 或 [`set`](https://realpython.com/courses/sets-python/) 对象)，用户定义的对象，类，函数，甚至[模块或包](https://realpython.com/courses/python-modules-packages/)。

您可以省略函数的返回值，使用不带返回值的空`return`。您也可以省略整个`return`语句。在这两种情况下，返回值都将是 [`None`](https://realpython.com/null-in-python/) 。

在接下来的两节中，您将了解到`return`语句是如何工作的，以及如何使用它将函数的结果返回给调用者代码。

### 显式`return`语句

一个**显式`return`语句**立即终止一个函数的执行，并将返回值发送回调用者代码。要向 Python 函数添加显式的`return`语句，您需要使用`return`，后跟一个可选的返回值:

>>>

```py
>>> def return_42():
...     return 42  # An explicit return statement
...

>>> return_42()  # The caller code gets 42
42
```

定义`return_42()`时，在函数代码块的末尾添加一个显式的`return`语句(`return 42`)。`42`是`return_42()`的显式返回值。这意味着任何时候你调用`return_42()`，这个函数都会把`42`发送回调用者。

**注意:**您可以使用显式的`return`语句，不管有没有返回值。如果你构建一个没有指定返回值的`return`语句，那么你将隐式返回`None`。

如果用显式的`return`语句定义一个函数，该语句有一个显式的返回值，那么可以在任何表达式中使用该返回值:

>>>

```py
>>> num = return_42()
>>> num
42

>>> return_42() * 2
84

>>> return_42() + 5
47
```

由于`return_42()`返回一个数值，您可以在数学表达式或任何其他类型的表达式中使用该值，在这些表达式中，该值具有逻辑或连贯的含义。这就是调用方代码利用函数返回值的方式。

请注意，您只能在函数或方法定义中使用`return`语句。如果你在别的地方使用它，那么你会得到一个 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) :

>>>

```py
>>> return 42
  File "<stdin>", line 1
SyntaxError: 'return' outside function
```

当您在函数或方法之外使用`return`时，您会得到一个`SyntaxError`，告诉您该语句不能在函数之外使用。

**注意:** [常规方法、类方法、静态方法](https://realpython.com/instance-class-and-static-methods-demystified/)只是 Python 类上下文中的函数。因此，您将涉及的所有`return`陈述概念也适用于它们。

您可以使用任何 Python 对象作为返回值。由于 Python 中的一切都是对象，所以可以返回[字符串](https://realpython.com/python-strings/)，列表，元组，字典，函数，[类，实例](https://realpython.com/python3-object-oriented-programming/#classes-vs-instances)，用户定义的对象，甚至模块或包。

例如，假设您需要编写一个函数，它接受一个整数列表并返回一个只包含原始列表中偶数的列表。这里有一种编码这个函数的方法:

>>>

```py
>>> def get_even(numbers):
...     even_nums = [num for num in numbers if not num % 2]
...     return even_nums
...

>>> get_even([1, 2, 3, 4, 5, 6])
[2, 4, 6]
```

`get_even()`使用一个[列表理解](https://realpython.com/courses/using-list-comprehensions-effectively/)来创建一个过滤掉原始`numbers`中奇数的列表。然后，该函数返回结果列表，其中只包含偶数。

一个常见的做法是使用一个[表达式](https://realpython.com/python-operators-expressions/)的结果作为一个`return`语句的返回值。为了应用这个想法，您可以将`get_even()`重写如下:

>>>

```py
>>> def get_even(numbers):
...     return [num for num in numbers if not num % 2]
...

>>> get_even([1, 2, 3, 4, 5, 6])
[2, 4, 6]
```

列表理解得到评估，然后函数返回结果列表。注意，你只能在一个`return`语句中使用[表达式](https://docs.python.org/3/glossary.html#term-expression)。表达式不同于[语句](https://docs.python.org/3/glossary.html#term-statement)，就像[条件句](https://realpython.com/courses/python-conditional-statements/)或[循环](https://realpython.com/courses/how-to-write-pythonic-loops/)。

**注意:**尽管`list`理解是使用`for`和(可选)`if`关键字构建的，但它们被认为是表达式而不是陈述。这就是为什么您可以在`return`语句中使用它们。

再举一个例子，假设您需要计算一个数值样本的平均值。为此，您需要将值的总和除以值的个数。下面是一个使用内置函数 [`sum()`](https://docs.python.org/3/library/functions.html#sum) 和 [`len()`](https://realpython.com/len-python-function/) 的例子:

>>>

```py
>>> def mean(sample):
...     return sum(sample) / len(sample)
...

>>> mean([1, 2, 3, 4])
2.5
```

在`mean()`中，你不用一个局部[变量](https://realpython.com/python-variables/)来存储计算的结果。相反，您可以直接使用表达式作为返回值。Python 首先对表达式`sum(sample) / len(sample)`求值，然后返回求值结果，在本例中是值`2.5`。

[*Remove ads*](/account/join/)

### 隐式`return`语句

Python 函数总是有返回值。Python 中没有过程或例程的概念。因此，如果您没有在`return`语句中显式地使用返回值，或者如果您完全省略了`return`语句，那么 Python 将隐式地为您返回一个默认值。默认返回值将永远是`None`。

假设您正在编写一个将`1`加到一个数字`x`的函数，但是您忘记了提供一个`return`语句。在这种情况下，您将得到一个使用`None`作为返回值的**隐式`return`语句**:

>>>

```py
>>> def add_one(x):
...     # No return statement at all
...     result = x + 1
...

>>> value = add_one(5)
>>> value

>>> print(value)
None
```

如果你不提供一个带有显式返回值的显式`return`语句，那么 Python 将使用`None`作为返回值提供一个隐式`return`语句。在上面的例子中，`add_one()`将`1`加到`x`上，并将值存储在`result`中，但它不返回`result`。所以才会得到`value = None`而不是`value = 6`。要解决这个问题，你需要或者`return result`或者直接`return x + 1`。

返回`None`的函数的一个例子是 [`print()`](https://realpython.com/python-print/) 。这个函数的目标是将对象打印到文本流文件，这通常是标准输出(您的屏幕)。所以，这个函数不需要显式的`return`语句，因为它不返回任何有用或有意义的东西:

>>>

```py
>>> return_value = print("Hello, World")
Hello, World

>>> print(return_value)
None
```

对`print()`的调用将`Hello, World`打印到屏幕上。由于这是`print()`的目的，函数不需要返回任何有用的东西，所以你得到`None`作为返回值。

**注意:**[Python 解释器](https://realpython.com/lessons/overview-python-interpreter/)不显示`None`。因此，要在[交互会话](https://realpython.com/run-python-scripts/#how-to-run-python-code-interactively)中显示`None`的返回值，您需要显式使用`print()`。

不管你的函数有多长多复杂，任何没有显式`return`语句的函数，或者有`return`语句但没有返回值的函数，都将返回`None`。

## 返回 vs 打印

如果您正在一个交互式会话中工作，那么您可能会认为打印一个值和返回一个值是等效的操作。考虑以下两个函数及其输出:

>>>

```py
>>> def print_greeting():
...     print("Hello, World")
...

>>> print_greeting()
Hello, World

>>> def return_greeting():
...     return "Hello, World"
...

>>> return_greeting()
'Hello, World'
```

这两个功能似乎做同样的事情。在这两种情况下，你都会看到`Hello, World`印在你的屏幕上。只有一个细微的明显区别——第二个例子中的单引号。但是看看如果返回另一种数据类型会发生什么，比如说一个`int`对象:

>>>

```py
>>> def print_42():
...     print(42)
...

>>> print_42()
42

>>> def return_42():
...     return 42
...

>>> return_42()
42
```

现在看不出有什么不同了。在这两种情况下，您都可以在屏幕上看到`42`。如果您刚刚开始使用 Python，这种行为可能会令人困惑。您可能认为返回和打印一个值是等效的操作。

现在，假设您对 Python 越来越深入，并且开始编写您的第一个脚本。您打开一个文本编辑器，键入以下代码:

```py
 1def add(a, b):
 2    result = a + b
 3    return result
 4
 5add(2, 2)
```

`add()`取两个数，相加，并返回结果。在**第 5 行**，你调用`add()`对`2`加`2`求和。由于您仍在学习返回和打印值之间的区别，您可能希望您的脚本将`4`打印到屏幕上。然而，事实并非如此，您的屏幕上什么也看不到。

自己试试吧。将您的脚本保存到一个名为`adding.py`和[的文件中，从您的命令行](https://realpython.com/run-python-scripts/)运行它，如下所示:

```py
$ python3 adding.py
```

如果你从命令行运行`adding.py`，那么你不会在屏幕上看到任何结果。这是因为当您运行脚本时，您在脚本中调用的函数的返回值不会像在交互式会话中那样打印到屏幕上。

如果您希望您的脚本在屏幕上显示调用`add()`的结果，那么您需要显式调用`print()`。查看`adding.py`的以下更新:

```py
 1def add(a, b):
 2    result = a + b
 3    return result
 4
 5print(add(2, 2))
```

现在，当你运行`adding.py`时，你会在屏幕上看到数字`4`。

因此，如果你在一个交互式会话中工作，那么 Python 会直接在你的屏幕上显示任何函数调用的结果。但是如果你正在写一个脚本，你想看到一个函数的返回值，那么你需要显式地使用`print()`。

[*Remove ads*](/account/join/)

## 返回多个值

您可以使用一个`return`语句从一个函数中返回多个值。为此，您只需提供几个用逗号分隔的返回值。

例如，假设您需要编写一个函数，该函数获取数字数据的样本并返回统计度量的摘要。要编写该函数，可以使用 Python 标准模块 [`statistics`](https://docs.python.org/3/library/statistics.html) ，它提供了几个用于计算数值数据的数理统计的函数。

下面是您的函数的一个可能的实现:

```py
import statistics as st

def describe(sample):
    return st.mean(sample), st.median(sample), st.mode(sample)
```

在`describe()`中，通过同时返回样本的平均值、中值和众数，利用 Python 在单个`return`语句中返回多个值的能力。注意，要返回多个值，您只需要按照您希望它们返回的顺序将它们写在一个逗号分隔的列表中。

一旦你编写了`describe()`，你就可以利用一个强大的 Python 特性，即 [iterable unpacking](https://realpython.com/lessons/tuple-assignment-packing-unpacking/) ，将三个度量值解包成三个独立的[变量](https://realpython.com/courses/variables-python/)，或者你可以将所有内容存储在一个变量中:

>>>

```py
>>> sample = [10, 2, 4, 7, 9, 3, 9, 8, 6, 7]
>>> mean, median, mode = describe(sample)

>>> mean
6.5

>>> median
7.0

>>> mode
7

>>> desc = describe(sample)
>>> desc
(6.5, 7.0, 7)

>>> type(desc)
<class 'tuple'>
```

在这里，您将`describe()`的三个返回值解包到变量`mean`、`median`和`mode`中。注意，在最后一个例子中，您将所有的值存储在一个变量`desc`中，这个变量是一个 Python `tuple`。

**注意:**你可以通过给一个变量分配几个逗号分隔的值来构建一个 Python `tuple`。没有必要使用括号来创建一个`tuple`。这就是为什么多个返回值被打包在一个`tuple`中。

内置函数 [`divmod()`](https://docs.python.org/3/library/functions.html#divmod) 也是返回多个值的函数的一个例子。该函数将两个(非复数)数字作为参数，并返回两个数字、两个输入值的商和除法的余数:

>>>

```py
>>> divmod(15, 3)
(5, 0)

>>> divmod(8, 3)
(2, 2)
```

对`divmod()`的调用返回一个元组，该元组包含作为参数提供的两个非复数的商和余数。这是一个有多个返回值的函数的例子。

## 使用 Python `return`语句:最佳实践

到目前为止，您已经了解了 Python `return`语句的基本工作原理。您现在知道如何编写向调用者返回一个或多个值的函数。此外，您已经了解到，如果您没有为给定的函数添加带有显式返回值的显式`return`语句，那么 Python 会为您添加。该值将是`None`。

在这一节中，您将会看到几个例子，这些例子将会引导您通过一组良好的编程实践来有效地使用`return`语句。这些实践将帮助您用 Python 编写可读性更强、可维护性更好、更健壮、更高效的函数。

### 显式返回`None`

一些程序员依赖 Python 添加到任何没有显式函数的函数中的隐式`return`语句。这可能会让来自其他编程语言的开发人员感到困惑，在其他编程语言中，没有返回值的函数被称为**过程**。

在某些情况下，您可以向函数中添加一个显式的`return None`。然而，在其他情况下，您可以依赖 Python 的默认行为:

*   如果你的函数执行动作，但是没有一个清晰而有用的`return`值，那么你可以省略返回`None`，因为这样做是多余的和令人困惑的。你也可以使用一个没有返回值的空的`return`来表明你从函数中返回的意图。

*   如果你的函数有多个`return`语句，并且返回`None`是一个有效的选项，那么你应该考虑明确使用`return None`，而不是依赖 Python 的默认行为。

这些实践可以通过明确传达您的意图来提高代码的可读性和可维护性。

当需要返回`None`时，您可以使用三种可能的方法之一:

1.  省略`return`语句，依赖返回`None`的默认行为。
2.  使用不带返回值的空`return`，它也返回`None`。
3.  显式返回`None`。

这在实践中是如何工作的:

>>>

```py
>>> def omit_return_stmt():
...     # Omit the return statement
...     pass
...
>>> print(omit_return_stmt())
None

>>> def bare_return():
...     # Use a bare return
...     return
...
>>> print(bare_return())
None

>>> def return_none_explicitly():
...     # Return None explicitly
...     return None
...
>>> print(return_none_explicitly())
None
```

是否显式返回`None`是个人决定。然而，您应该考虑到，在某些情况下，显式的`return None`可以避免可维护性问题。对于来自其他编程语言的开发人员来说尤其如此，因为他们的行为不像 Python 那样。

[*Remove ads*](/account/join/)

### 记住返回值

编写自定义函数时，您可能会不小心忘记从函数中返回值。在这种情况下，Python 会为你返回`None`。这可能会导致一些细微的错误，对于一个初学 Python 的开发者来说，理解和调试是很困难的。

您可以通过在函数头之后立即编写`return`语句来避免这个问题。然后，您可以第二次编写函数体。这里有一个模板，您可以在编写 Python 函数时使用:

```py
def template_func(args):
    result = 0  # Initialize the return value
    # Your code goes here...
    return result  # Explicitly return the result
```

如果你习惯了这样开始你的函数，那么你就有可能不再错过`return`语句。使用这种方法，您可以编写函数体，测试它，并在知道函数可以工作后重命名变量。

这种做法可以提高您的生产率，并使您的功能不容易出错。也可以为你节省很多[调试](https://realpython.com/python-debugging-pdb/)的时间。

### 避免复杂的表达式

正如您之前看到的，在 Python 函数中使用表达式的结果作为返回值是一种常见的做法。如果您使用的表达式变得太复杂，那么这种做法会导致函数难以理解、调试和维护。

例如，如果您正在进行一项复杂的计算，那么使用带有有意义名称的临时变量**来增量计算最终结果将更具可读性。**

考虑以下计算数字数据样本的方差的函数:

>>>

```py
>>> def variance(data, ddof=0):
...     mean = sum(data) / len(data)
...     return sum((x - mean) ** 2 for x in data) / (len(data) - ddof)
...

>>> variance([3, 4, 7, 5, 6, 2, 9, 4, 1, 3])
5.24
```

你在这里使用的表达很复杂，很难理解。调试起来也很困难，因为您要在一个表达式中执行多个操作。要解决这个特殊的问题，您可以利用增量开发方法来提高函数的可读性。

看看下面这个`variance()`的替代实现:

>>>

```py
>>> def variance(data, ddof=0):
...     n = len(data)
...     mean = sum(data) / n
...     total_square_dev = sum((x - mean) ** 2 for x in data)
...     return total_square_dev / (n - ddof)
...

>>> variance([3, 4, 7, 5, 6, 2, 9, 4, 1, 3])
5.24
```

在`variance()`的第二个实现中，您通过几个步骤计算方差。每一步都由一个具有有意义名称的临时变量表示。

在调试代码时，像`n`、`mean`和`total_square_dev`这样的临时变量通常很有帮助。例如，如果其中一个出错，那么您可以调用`print()`来了解在`return`语句运行之前发生了什么。

一般来说，你应该避免在你的`return`语句中使用复杂的表达式。相反，您可以将代码分成多个步骤，并在每个步骤中使用临时变量。使用临时变量可以使代码更容易调试、理解和维护。

### 返回值 vs 修改全局

没有带有有意义返回值的显式`return`语句的函数通常会执行有[副作用](https://realpython.com/defining-your-own-python-function/#side-effects)的动作。一个**副作用**可以是，例如，打印一些东西到屏幕上，修改一个[全局变量](https://realpython.com/python-scope-legb-rule/#modules-the-global-scope)，更新一个对象的状态，[写一些文本到一个文件](https://realpython.com/read-write-files-python/#reading-and-writing-opened-files)，等等。

修改全局变量通常被认为是一种糟糕的编程实践。就像带有复杂表达式的程序一样，修改全局变量的程序可能很难调试、理解和维护。

当你修改一个全局变量时，你可能会影响到所有的函数、类、对象以及依赖于这个全局变量的程序的其他部分。

要理解一个修改全局变量的程序，你需要知道程序中可以看到、访问和改变这些变量的所有部分。因此，良好的实践建议编写**自包含函数**，它接受一些参数并返回一个(或多个)有用的值，而不会对全局变量产生任何副作用。

此外，使用显式`return`语句返回有意义值的函数比修改或更新全局变量的函数更容易[测试](https://realpython.com/pytest-python-testing/)。

以下示例显示了一个更改全局变量的函数。该函数使用了 [`global`语句](https://realpython.com/python-scope-legb-rule/#the-global-statement)，这在 Python 中也被认为是一种糟糕的编程实践:

>>>

```py
>>> counter = 0

>>> def increment():
...     global counter
...     counter += 1
...

>>> increment()
>>> counter
1
```

在这个例子中，首先创建一个全局变量`counter`，初始值为`0`。在`increment()`中，你使用一个`global`语句告诉函数你想要修改一个全局变量。最后一条语句将`counter`增加`1`。

调用`increment()`的结果将取决于`counter`的初始值。`counter`不同的初始值会产生不同的结果，所以函数的结果不能由函数本身控制。

为了避免这种行为，您可以编写一个自包含的`increment()`，它接受参数并返回一个仅依赖于输入参数的一致值:

>>>

```py
>>> counter = 0

>>> def increment(var):
...     return var + 1
...

>>> increment(counter)
1

>>> counter
0

>>> # Explicitly assign a new value to counter
>>> counter = increment(counter)
>>> counter
1
```

现在调用`increment()`的结果只取决于输入参数而不是`counter`的初始值。这使得函数更加健壮，也更容易测试。

**注意:**为了更好地理解如何测试你的 Python 代码，请查看[用 PyTest 进行测试驱动开发](https://realpython.com/courses/test-driven-development-pytest/)。

此外，当您需要更新`counter`时，您可以通过调用`increment()`来显式地完成。通过这种方式，您可以在整个代码中更好地控制`counter`所发生的事情。

一般来说，避免使用修改全局变量的函数是一个好习惯。如果可能的话，试着编写带有显式`return`语句的**自包含函数**，该语句返回一致且有意义的值。

[*Remove ads*](/account/join/)

### 将`return`与条件句一起使用

Python 函数并不局限于单个`return`语句。如果一个给定的函数有不止一个`return`语句，那么遇到的第一个语句将决定函数执行的结束以及它的返回值。

用多个`return`语句编写函数的一种常见方式是使用[条件语句](https://realpython.com/courses/python-conditional-statements/)，它允许你根据评估某些条件的结果提供不同的`return`语句。

假设您需要编写一个函数，它接受一个数字并返回它的[绝对值](https://realpython.com/python-absolute-value)。如果数字大于`0`，那么你将返回相同的数字。如果这个数字小于`0`，那么你将返回它的相反值，或者非负值。

下面是这个函数的一个可能的实现:

>>>

```py
>>> def my_abs(number):
...     if number > 0:
...         return number
...     elif number < 0:
...         return -number
...

>>> my_abs(-15)
15

>>> my_abs(15)
15
```

`my_abs()`有两个显式的`return`语句，每个都包装在自己的 [`if`语句](https://realpython.com/python-conditional-statements/)中。它还有一个隐式的`return`语句。如果`number`恰好是`0`，那么这两个条件都不成立，函数结束时没有命中任何显式的`return`语句。当这种情况发生时，你自动得到`None`。

看看下面使用`0`作为参数对`my_abs()`的调用:

>>>

```py
>>> print(my_abs(0))
None
```

当你使用`0`作为参数调用`my_abs()`时，你得到的结果是`None`。这是因为执行流到达了函数的末尾，而没有到达任何显式的`return`语句。可惜的是，`0`的绝对值是`0`，不是`None`。

要解决这个问题，您可以在新的`elif`子句或最终的`else`子句中添加第三个`return`语句:

>>>

```py
>>> def my_abs(number):
...     if number > 0:
...         return number
...     elif number < 0:
...         return -number
...     else:
...         return 0
...

>>> my_abs(0)
0

>>> my_abs(-15)
15

>>> my_abs(15)
15
```

现在，`my_abs()`检查每一个可能的条件，`number > 0`，`number < 0`和`number == 0`。这个例子的目的是说明当你使用条件语句来提供多个`return`语句时，你需要确保每个可能的选项都有自己的`return`语句。否则，你的函数会有一个隐藏的 bug。

最后，您可以使用一条`if`语句以更简洁、高效和[的方式实现`my_abs()`:](https://realpython.com/learning-paths/writing-pythonic-code/)

>>>

```py
>>> def my_abs(number):
...     if number < 0:
...         return -number
...     return number
...

>>> my_abs(0)
0

>>> my_abs(-15)
15

>>> my_abs(15)
15
```

在这种情况下，如果`number < 0`，您的函数将命中第一个`return`语句。在所有其他情况下，无论是`number > 0`还是`number == 0`，它都命中第二个`return`语句。有了这个新的实现，您的函数看起来好多了。它可读性更强，更简洁，也更高效。

**注意:**有一个方便的内置 Python 函数叫做 [`abs()`](https://docs.python.org/3/library/functions.html#abs) 用于计算一个数的绝对值。上面例子中的函数仅仅是为了说明正在讨论的问题。

如果您使用`if`语句来提供几个`return`语句，那么您不需要一个`else`子句来涵盖最后一个条件。只需在函数代码块的末尾和缩进的第一级添加一个`return`语句。

### 返回`True`或`False`

组合使用`if`和`return`语句的另一个常见用例是当你编写一个[谓词](https://en.wikipedia.org/wiki/Predicate_(mathematical_logic))或[布尔值](https://en.wikipedia.org/wiki/Boolean-valued_function)函数时。这种函数根据给定的条件返回`True`或`False`。

例如，假设您需要编写一个函数，它接受两个整数`a`和`b`，如果`a`能被`b`整除，则返回`True`。否则，函数应该返回`False`。下面是一个可能的实现:

>>>

```py
>>> def is_divisible(a, b):
...     if not a % b:
...         return True
...     return False
...

>>> is_divisible(4, 2)
True

>>> is_divisible(7, 4)
False
```

如果`a`除以`b`的余数等于`0`，则`is_divisible()`返回`True`。否则返回`False`。注意，在 Python 中，一个`0`值是 [falsy](https://realpython.com/python-data-types/#boolean-type-boolean-context-and-truthiness) ，所以需要使用 [`not`运算符](https://realpython.com/python-operators-expressions/#logical-operators)对条件的真值求反。

有时，您会编写包含如下运算符的谓词函数:

*   [比较运算符](https://realpython.com/python-operators-expressions/#comparison-operators)`==``!=``>``>=``<`和`<=`
*   [隶属操作员](https://docs.python.org/3/reference/expressions.html#membership-test-operations) `in`
*   [身份符](https://realpython.com/python-operators-expressions/#identity-operators) `is`
*   [布尔运算符](https://docs.python.org/3/reference/expressions.html#not) `not`

在这些情况下，你可以在你的`return`语句中直接使用一个[布尔表达式](https://realpython.com/python-boolean/)。这是可能的，因为这些操作符要么返回`True`要么返回`False`。按照这个想法，这里有一个`is_divisible()`的新实现:

>>>

```py
>>> def is_divisible(a, b):
...     return not a % b
...

>>> is_divisible(4, 2)
True

>>> is_divisible(7, 4)
False
```

如果`a`能被`b`整除，那么`a % b`返回`0`，在 Python 中是 falsy。所以，要返回`True`，你需要使用`not`操作符。

**注意:** Python 遵循一套规则来确定一个对象的真值。

例如，以下物体[被认为是假的](https://docs.python.org/3/library/stdtypes.html#truth-value-testing):

*   [常数](https://realpython.com/python-constants/)像 [`None`](https://realpython.com/null-in-python/) 和`False`
*   具有零值的数字类型，如`0`、`0.0`、`0j`、[、`Decimal(0)`、](https://docs.python.org/3/library/decimal.html#decimal.Decimal)[、`Fraction(0, 1)`、](https://docs.python.org/3/library/fractions.html#fractions.Fraction)
*   空序列和集合，如`""`、`()`、`[]`、`{}`、[、](https://realpython.com/python-sets/)和`range(0)`
*   实现返回值为`False`的 [`__bool__()`](https://docs.python.org/3/reference/datamodel.html#object.__bool__) 或返回值为`0`的 [`__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 的对象

任何其他物体都将被认为是真实的。

另一方面，如果您试图以之前看到的方式使用包含布尔运算符的条件，如 [`or`](https://realpython.com/python-or-operator/) 和 [`and`](https://realpython.com/python-keywords/#operator-keywords-and-or-not-in-is) ，那么您的谓词函数将无法正常工作。这是因为这些操作符的行为不同。它们返回条件中的一个操作数，而不是`True`或`False`:

>>>

```py
>>> 0 and 1
0
>>> 1 and 2
2

>>> 1 or 2
1
>>> 0 or 1
1
```

一般情况下，`and`返回第一个假操作数或最后一个操作数。另一方面，`or`返回第一个真操作数或最后一个操作数。因此，要编写一个包含这些操作符之一的谓词，您需要使用一个显式的`if`语句或者调用内置函数 [`bool()`](https://docs.python.org/3/library/functions.html#bool) 。

假设您想要编写一个谓词函数，它接受两个值，如果两个值都为真，则返回`True`，否则返回`False`。这是实现该功能的第一种方法:

>>>

```py
>>> def both_true(a, b):
...     return a and b
...
>>> both_true(1, 2)
2
```

由于`and`返回操作数，而不是`True`或`False`，你的函数不能正常工作。解决这个问题至少有三种可能性:

1.  一个明确的 [`if`陈述](https://realpython.com/lessons/if-statements/)
2.  一个[条件表达式(三元运算符)](https://realpython.com/python-conditional-statements/#conditional-expressions-pythons-ternary-operator)
3.  内置的 Python 函数`bool()`

如果你使用第一种方法，那么你可以写`both_true()`如下:

>>>

```py
>>> def both_true(a, b):
...     if a and b:
...         return True
...     return False
...

>>> both_true(1, 2)
True

>>> both_true(1, 0)
False
```

`if`语句检查`a`和`b`是否都是真的。如果是，那么`both_true()`返回`True`。否则返回`False`。

另一方面，如果您使用 Python 条件表达式或三元运算符，那么您可以按如下方式编写谓词函数:

>>>

```py
>>> def both_true(a, b):
...     return True if a and b else False
...
>>> both_true(1, 2)
True

>>> both_true(1, 0)
False
```

这里，您使用一个条件表达式为`both_true()`提供一个返回值。如果`a`和`b`都为真，则条件表达式被评估为`True`。否则，最后的结果就是`False`。

最后，如果您使用`bool()`，那么您可以将`both_true()`编码如下:

>>>

```py
>>> def both_true(a, b):
...     return bool(a and b)
...

>>> both_true(1, 2)
True

>>> both_true(1, 0)
False
```

如果`a`和`b`为真，则`bool()`返回`True`，否则`False`返回。用什么方法解决这个问题取决于你。然而，第二种解决方案似乎更具可读性。你怎么想呢?

[*Remove ads*](/account/join/)

### 短路回路

循环中的`return`语句执行某种**短路**。它中断循环执行并使函数立即返回。为了更好地理解这种行为，您可以编写一个模拟 [`any()`](https://realpython.com/any-python/) 的函数。这个内置函数接受一个 iterable，如果至少有一个 iterable 项为 true，则返回`True`。

为了模拟`any()`，您可以编写如下函数:

>>>

```py
>>> def my_any(iterable):
...     for item in iterable:
...         if item:
...             # Short-circuit
...             return True
...     return False

>>> my_any([0, 0, 1, 0, 0])
True

>>> my_any([0, 0, 0, 0, 0])
False
```

如果`iterable`中的任意`item`为真，那么执行流程进入`if`块。`return`语句中断循环并立即返回，返回值为`True`。如果`iterable`中没有值为真，则`my_any()`返回`False`。

该功能实现了**短路评估**。例如，假设您传递了一个包含一百万项的 iterable。如果 iterable 中的第一项恰好为真，那么循环只运行一次，而不是一百万次。这可以在运行代码时节省大量处理时间。

需要注意的是，要在循环中使用`return`语句，您需要将该语句包装在`if`语句中。否则，循环将总是在第一次迭代中中断。

### 识别死代码

一旦一个函数命中一个`return`语句，它就终止而不执行任何后续代码。因此，出现在函数的`return`语句之后的代码通常被称为**死代码**。Python 解释器在运行函数时完全忽略死代码。因此，在函数中包含这样的代码是无用的，也是令人困惑的。

考虑下面的函数，它在其`return`语句后添加了代码:

>>>

```py
>>> def dead_code():
...     return 42
...     # Dead code
...     print("Hello, World")
...

>>> dead_code()
42
```

本例中的语句`print("Hello, World")`永远不会执行，因为该语句出现在函数的`return`语句之后。识别死代码并删除它是一个很好的实践，可以用来编写更好的函数。

值得注意的是，如果你使用条件语句来提供多个`return`语句，那么你可以在一个`return`语句之后有代码，只要它在`if`语句之外，就不会死:

>>>

```py
>>> def no_dead_code(condition):
...     if condition:
...         return 42
...     print("Hello, World")
...

>>> no_dead_code(True)
42
>>> no_dead_code(False)
Hello, World
```

尽管对`print()`的调用是在`return`语句之后，但它不是死代码。当`condition`被评估为`False`时，运行`print()`调用，并且将`Hello, World`打印到您的屏幕上。

### 返回多个命名对象

当你在编写一个在单个`return`语句中返回多个值的函数时，你可以考虑使用一个 [`collections.namedtuple`](https://realpython.com/python-namedtuple/) 对象来使你的函数更具可读性。`namedtuple`是一个集合类，它返回`tuple`的一个子类，该子类有字段或属性。您可以使用[点符号](https://docs.python.org/3/reference/expressions.html#attribute-references)或[索引操作](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations)来访问这些属性。

`namedtuple`的初始化器有几个参数。然而，要开始在代码中使用`namedtuple`，您只需要知道前两个:

1.  **`typename`** 保存您正在创建的类似元组的类的名称。它需要是一个字符串。
2.  **`field_names`** 保存元组类的字段或属性的名称。它可以是一系列字符串，如`["x", "y"]`，也可以是单个字符串，每个名称用空格或逗号分隔，如`"x y"`或`"x, y"`。

当你需要返回多个值时，使用一个`namedtuple`可以让你的函数更加易读，而不需要太多的努力。考虑以下使用`namedtuple`作为返回值的`describe()`的更新:

```py
import statistics as st
from collections import namedtuple

def describe(sample):
    Desc = namedtuple("Desc", ["mean", "median", "mode"])
    return Desc(
        st.mean(sample),
        st.median(sample),
        st.mode(sample),
    )
```

在`describe()`中，你创建了一个叫做`Desc`的`namedtuple`。该对象可以具有命名属性，您可以使用点标记法或索引操作来访问这些属性。在这个例子中，这些属性是`"mean"`、`"median"`和`"mode"`。

您可以创建一个`Desc`对象，并将其用作返回值。为此，您需要像处理任何 Python 类一样实例化`Desc`。注意，您需要为每个命名属性提供一个具体的值，就像您在`return`语句中所做的那样。

下面是`describe()`现在的工作方式:

>>>

```py
>>> sample = [10, 2, 4, 7, 9, 3, 9, 8, 6, 7]
>>> stat_desc = describe(sample)

>>> stat_desc
Desc(mean=5.7, median=6.0, mode=6)

>>> # Get the mean by its attribute name
>>> stat_desc.mean
5.7

>>> # Get the median by its index
>>> stat_desc[1]
6.0

>>> # Unpack the values into three variables
>>> mean, median, mode = describe(sample)

>>> mean
5.7

>>> mode
6
```

当您用数字数据的样本调用`describe()`时，您会得到一个包含样本的平均值、中值和众数的`namedtuple`对象。请注意，您可以通过使用点符号或索引操作来访问元组的每个元素。

最后，还可以使用 iterable 解包操作将每个值存储在它自己的独立变量中。

[*Remove ads*](/account/join/)

## 返回函数:闭包

在 Python 中，函数是[一级对象](https://realpython.com/primer-on-python-decorators/#first-class-objects)。一级对象是可以赋给变量、作为参数传递给函数或在函数中用作返回值的对象。因此，您可以在任何`return`语句中使用函数对象作为返回值。

以一个函数作为自变量，返回一个函数作为结果，或者两者都是的函数是一个[高阶函数](http://en.wikipedia.org/wiki/Higher-order_function)。一个[闭包工厂函数](https://realpython.com/inner-functions-what-are-they-good-for/#closures-and-factory-functions)是 Python 中高阶函数的一个常见例子。这种函数接受一些参数并返回一个[内部函数](https://realpython.com/inner-functions-what-are-they-good-for/)。内部函数通常被称为**闭包**。

闭包携带关于其封闭执行范围的信息。这提供了一种在函数调用之间保留状态信息的方法。当您需要基于[懒惰或延迟评估](https://en.wikipedia.org/wiki/Lazy_evaluation)的概念编写代码时，闭包工厂函数非常有用。

假设您需要编写一个 helper 函数，它接受一个数字并返回该数字乘以给定因子的结果。您可以编写如下函数:

```py
def by_factor(factor, number):
    return factor * number
```

`by_factor()`以`factor`和`number`为自变量，返回它们的乘积。因为`factor`很少在你的应用程序中改变，你会发现在每个函数调用中提供相同的因子很烦人。因此，您需要一种方法来在对`by_factor()`的调用之间保留`factor`的状态或值，并仅在需要时更改它。要在两次调用之间保留当前的值`factor`，可以使用闭包。

下面的`by_factor()`实现使用闭包在调用之间保留`factor`的值:

>>>

```py
>>> def by_factor(factor):
...     def multiply(number):
...         return factor * number
...     return multiply
...

>>> double = by_factor(2)
>>> double(3)
6
>>> double(4)
8

>>> triple = by_factor(3)
>>> triple(3)
9
>>> triple(4)
12
```

在`by_factor()`内部，您定义了一个名为`multiply()`的内部函数，并在不调用它的情况下返回它。您返回的函数对象是一个闭包，它保留了关于`factor`状态的信息。换句话说，它在两次调用之间记住了`factor`的值。这就是为什么`double`记得`factor`等于`2`而`triple`记得`factor`等于`3`。

请注意，您可以自由地重用`double`和`triple`，因为它们不会忘记各自的状态信息。

你也可以使用一个 [`lambda`函数](https://realpython.com/python-lambda/)来创建闭包。有时候使用一个`lambda`函数可以让你的闭包工厂更加简洁。这里有一个使用`lambda`函数的`by_factor()`的替代实现:

>>>

```py
>>> def by_factor(factor):
...     return lambda number: factor * number
...

>>> double = by_factor(2)
>>> double(3)
6
>>> double(4)
8
```

这个实现就像最初的例子一样工作。在这种情况下，使用`lambda`函数提供了一种快速简洁的方式来编码`by_factor()`。

## 接受和返回函数:装饰者

使用`return`语句返回函数对象的另一种方式是编写[装饰函数](https://realpython.com/primer-on-python-decorators/)。一个**装饰函数**接受一个函数对象作为参数并返回一个函数对象。装饰器以某种方式处理被装饰的函数，并返回它或者用另一个函数或可调用对象替换它。

当您需要在不修改现有函数的情况下向其添加额外的逻辑时，Decorators 非常有用。例如，您可以编写一个装饰器来记录函数调用，验证函数的参数，测量给定函数的执行时间，等等。

以下示例显示了一个装饰函数，您可以使用它来了解给定 Python 函数的执行时间:

>>>

```py
>>> import time

>>> def my_timer(func):
...     def _timer(*args, **kwargs):
...         start = time.time()
...         result = func(*args, **kwargs)
...         end = time.time()
...         print(f"Execution time: {end - start}")
...         return result
...     return _timer
...

>>> @my_timer
... def delayed_mean(sample):
...     time.sleep(1)
...     return sum(sample) / len(sample)
...

>>> delayed_mean([10, 2, 4, 7, 9, 3, 9, 8, 6, 7])
Execution time: 1.0011096000671387
6.5
```

`delayed_mean()`头上面的语法`@my_timer`等价于表达式`delayed_mean = my_timer(delayed_mean)`。在这种情况下，你可以说`my_timer()`在装修`delayed_mean()`。

一旦你[导入](https://realpython.com/absolute-vs-relative-python-imports/)或者运行一个模块或者脚本，Python 就会运行装饰函数。所以，当你调用`delayed_mean()`时，你实际上是在调用`my_timer()`的返回值，也就是函数对象`_timer`。对修饰的`delayed_mean()`的调用将返回样本的平均值，还将测量原始`delayed_mean()`的执行时间。

在这种情况下，您使用 [`time()`](https://docs.python.org/3/library/time.html#time.time) 来测量装饰器内部的执行时间。`time()`驻留在一个名为 [`time`](https://docs.python.org/3/library/time.html) 的模块中，该模块提供了一组与时间相关的函数。`time()`以浮点数形式返回自[纪元](https://docs.python.org/3/library/time.html#epoch)以来的时间(秒)。调用`delayed_mean()`前后的时间差将让您了解函数的执行时间。

**注意:**在`delayed_mean()`中，您使用函数 [`time.sleep()`](https://docs.python.org/3/library/time.html#time.sleep) ，它将调用代码的执行暂停给定的秒数。为了更好地理解如何使用`sleep()`，请查看 [Python sleep():如何向代码](https://realpython.com/python-sleep/)添加时间延迟。

Python 中其他常见的装饰器例子有 [`classmethod()`、`staticmethod()`](https://realpython.com/courses/staticmethod-vs-classmethod-python/) 、 [`property()`、T6】。如果你想更深入地了解 Python decorator，那么看看 Python decorator](https://docs.python.org/3/library/functions.html#property)的[初级读本。你也可以看看](https://realpython.com/primer-on-python-decorators/) [Python Decorators 101](https://realpython.com/courses/python-decorators-101/) 。

[*Remove ads*](/account/join/)

## 返回用户定义的对象:工厂模式

Python `return`语句也可以返回[用户定义的对象](https://realpython.com/python3-object-oriented-programming/#how-to-define-a-class-in-python)。换句话说，您可以使用自己的自定义对象作为函数中的返回值。这种能力的一个常见用例是[工厂模式](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming))。

工厂模式定义了一个接口，用于动态创建对象，以响应您在编写程序时无法预测的情况。您可以使用一个函数实现用户定义对象的工厂，该函数接受一些初始化参数，并根据具体的输入返回不同的对象。

假设您正在编写一个绘画应用程序。您需要根据用户的选择动态创建不同的形状。你的程序将会有正方形、圆形、长方形等等。要动态创建这些形状，首先需要创建将要使用的形状类:

```py
class Circle:
    def __init__(self, radius):
        self.radius = radius
    # Class implementation...

class Square:
    def __init__(self, side):
        self.side = side
    # Class implementation...
```

一旦您为每个形状创建了一个类，您就可以编写一个函数，该函数将形状的名称作为一个字符串以及一个可选的[参数(`*args`)和关键词参数(`**kwargs` )](https://realpython.com/python-kwargs-and-args/) 列表，来动态创建和初始化形状:

```py
def shape_factory(shape_name, *args, **kwargs):
    shapes = {"circle": Circle, "square": Square}
    return shapes[shape_name](*args, **kwargs)
```

该函数创建具体形状的实例，并将其返回给调用者。现在，您可以使用`shape_factory()`来创建不同形状的对象，以满足用户的需求:

>>>

```py
>>> circle = shape_factory("circle", radius=20)
>>> type(circle)
<class '__main__.Circle'>
>>> circle.radius
20

>>> square = shape_factory("square", side=10)
>>> type(square)
<class '__main__.Square'>
>>> square.side
10
```

如果您以所需形状的名称作为字符串调用`shape_factory()`，那么您将获得一个与您刚刚传递给工厂的`shape_name`相匹配的形状的新实例。

## 使用`try` … `finally`模块中的`return`

当您在带有 [`finally`](https://realpython.com/python-exceptions/#cleaning-up-after-using-finally) 子句的 [`try`语句](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions)内使用`return`语句时，该`finally`子句总是在`return`语句之前执行。这确保了`finally`子句中的代码将一直运行。看看下面的例子:

>>>

```py
>>> def func(value):
...     try:
...         return float(value)
...     except ValueError:
...         return str(value)
...     finally:
...         print("Run this before returning")
...

>>> func(9)
Run this before returning
9.0

>>> func("one")
Run this before returning
'one'
```

当您调用`func()`时，您会将`value`转换为浮点数或字符串对象。在此之前，您的函数运行`finally`子句并在屏幕上打印一条消息。您添加到`finally`子句的任何代码都将在函数运行其`return`语句之前执行。

## 在发生器功能中使用`return`

一个主体中带有 [`yield`语句](https://realpython.com/introduction-to-python-generators/#understanding-the-python-yield-statement)的 Python 函数就是一个 [**生成器函数**](https://realpython.com/introduction-to-python-generators/) 。当你调用一个生成器函数时，它返回一个[生成器迭代器](https://docs.python.org/3/glossary.html#term-generator-iterator)。所以，你可以说一个生成器函数是一个**生成器工厂**。

您可以在生成器函数中使用一个`return`语句来表示生成器已经完成。`return`语句将使发电机发出 [`StopIteration`](https://docs.python.org/3/library/exceptions.html#StopIteration) 。返回值将作为参数传递给`StopIteration`的初始化器，并赋给它的`.value`属性。

这里有一个生成器，它按需生成`1`和`2`，然后返回`3`:

>>>

```py
>>> def gen():
...     yield 1
...     yield 2
...     return 3
...

>>> g = gen()
>>> g
<generator object gen at 0x7f4ff4853c10>

>>> next(g)
1
>>> next(g)
2

>>> next(g)
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    next(g)
StopIteration: 3
```

`gen()`根据需要返回产生`1`和`2`的生成器对象。要从生成器对象中检索每个数字，可以使用 [`next()`](https://docs.python.org/3/library/functions.html#next) ，这是一个内置函数，用于从 Python 生成器中检索下一个项目。

对`next()`的前两个调用分别检索`1`和`2`。第三次调用，发电机耗尽，你得到一个`StopIteration`。注意，生成器函数(`3`)的返回值变成了`StopIteration`对象的`.value`属性。

[*Remove ads*](/account/join/)

## 结论

Python `return`语句允许您将任何 Python 对象从您的[自定义函数](https://realpython.com/defining-your-own-python-function/)发送回调用者代码。该语句是任何 Python 函数或方法的基础部分。如果您掌握了如何使用它，那么您就可以编写健壮的函数了。

**在本教程中，您已经学会了如何:**

*   在你的**函数**中有效地使用 **Python `return`语句**
*   从你的函数中返回**单值**或**多值**给调用者代码
*   使用`return`语句时，应用**最佳实践**

此外，您还学习了一些关于`return`语句的更高级的用例，比如如何编写一个[闭包工厂函数](https://realpython.com/inner-functions-what-are-they-good-for/#closures-and-factory-functions)和一个[装饰函数](https://realpython.com/primer-on-python-decorators)。有了这些知识，你将能够用 Python 编写更多的[Python 化的](https://realpython.com/learning-paths/writing-pythonic-code/)、健壮的、可维护的函数。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**有效使用 Python 返回语句**](/courses/effective-python-return-statement/)***********