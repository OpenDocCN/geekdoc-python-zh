# Python 中什么时候用省略号？

> 原文：<https://realpython.com/python-ellipsis/>

在英语写作中，你可以用省略号来表示你漏掉了什么。本质上，您使用三个点(`...`)来替换内容。但是省略号不仅仅存在于散文中——你可能也在 Python 源代码中看到过三个点。

省略号文字(`...`)计算为 Python 的 **`Ellipsis`** 。因为[`Ellipsis`T8 是一个内置的](https://docs.python.org/3/library/constants.html#Ellipsis)[常量](https://realpython.com/python-constants/)，你可以使用`Ellipsis`或者`...`而不用[导入](https://realpython.com/python-import/)它:

>>>

```py
>>> ...
Ellipsis

>>> Ellipsis
Ellipsis

>>> ... is Ellipsis
True
```

虽然三个点作为 Python 语法看起来很奇怪，但是在某些情况下使用`...`会很方便。**但是在 Python 中什么时候应该使用`Ellipsis`？**

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/python-ellipsis-code/)，你将使用它来掌握省略号文字。

## 简而言之:在 Python 中使用省略号作为占位符

虽然您可以互换使用`...`和`Ellipsis`，但您通常会在代码中选择`...`。类似于在英语中使用三个点来省略内容，可以使用 Python 中的省略号作为未写代码的占位符:

```py
# ellipsis_example.py

def do_nothing():
    ...

do_nothing()
```

当您运行`ellipsis_example.py`并执行`do_nothing()`时，Python 会毫无怨言地运行:

```py
$ python ellipsis_example.py
$
```

当您在 Python 中执行一个函数体中只包含`...`的函数时，不会出现错误。这意味着您可以使用省略号作为占位符，类似于 [`pass`关键字](https://realpython.com/python-pass/)。

使用三个点创建最小的视觉混乱。所以，当你在线分享你的代码时，替换不相关的代码是很方便的。

省略代码的常见情况是使用**存根**的时候。您可以将存根视为实函数的替身。当您只需要一个函数签名但不想执行函数体中的代码时，存根就可以派上用场了。例如，在开发应用程序时，您可能希望阻止外部请求。

假设你有一个[烧瓶项目](https://realpython.com/flask-blueprint/)，你在`custom_stats.count_visitor()`创建了自己的访客计数器。`count_visitor()`功能连接到跟踪访问者数量的数据库。为了在调试模式下测试应用程序时不把自己算进去，可以创建一个`count_visitor()`的存根:

```py
 1# app.py
 2
 3from flask import Flask
 4
 5from custom_stats import count_visitor
 6
 7app = Flask(__name__)
 8
 9if app.debug:
10    def count_visitor(): ... 11
12@app.route("/")
13def home():
14    count_visitor()
15    return "Hello, world!"
```

因为在这种情况下`count_visitor()`的内容无关紧要，所以在函数体中使用省略号是个好主意。当您在调试模式下运行 Flask 应用程序时，Python 调用`count_visitor()`没有错误或不必要的副作用:

```py
$ flask --app app --debug run
 * Serving Flask app 'app'
 * Debug mode: on
```

如果您在调试模式下运行 Flask 应用程序，那么第 14 行中的`count_visitor()`引用第 10 行中的存根。在`count_visitor()`的函数体中使用`...`可以帮助你测试你的应用程序而不会有副作用。

上面的例子显示了如何在较小的范围内使用存根。对于更大的项目，存根经常用在[单元测试](https://realpython.com/python-testing/)中，因为它们有助于以一种隔离的方式测试你的部分代码。

此外，如果您熟悉 Python 中的[类型检查，那么关于省略号和存根的讨论可能会让您想起一些事情。](https://realpython.com/python-type-checking/)

进行类型检查最常用的工具是 **mypy** 。为了确定标准库和第三方库定义的类型， [mypy](http://mypy-lang.org/index.html) 使用**存根文件**:

> 存根文件是一个包含 Python 模块公共接口框架的文件，包括类、变量、函数——最重要的是它们的类型。([来源](https://mypy.readthedocs.io/en/stable/getting_started.html#stubs-intro))

您可以访问 Python 的[类型化存储库](https://github.com/python/typeshed)，并探索这个存储库包含的存根文件中`...`的用法。当你深入到静态类型的主题时，你可能会发现`Ellipsis`常量的另一个用例。在下一节中，您将学习何时在类型提示中使用`...`。

[*Remove ads*](/account/join/)

## 类型提示中的省略号是什么意思？

在上一节中，您了解了可以使用`Ellipsis`作为存根文件的占位符，包括在类型检查时。但是你也可以在**类型提示**中使用`...`。在本节中，您将学习如何使用`...`来:

1.  指定同质类型的可变长度元组
2.  替换可调用函数的参数列表

类型提示是一种很好的方式，可以明确您在代码中期望的数据类型。但有时，您希望在不完全限制用户如何使用对象的情况下使用类型提示。例如，您可能希望指定一个只包含整数的元组，但是整数的数量可以是任意的。这时省略号就派上用场了:

```py
 1# tuple_example.py
 2
 3numbers: tuple[int, ...] 4
 5# Allowed:
 6numbers = ()
 7numbers = (1,)
 8numbers = (4, 5, 6, 99)
 9
10# Not allowed:
11numbers = (1, "a")
12numbers = [1, 3]
```

在第 3 行，您定义了一个类型为[元组](https://mypy.readthedocs.io/en/stable/kinds_of_types.html#tuple-types)的变量`numbers`。`numbers`变量必须是只包含整数的元组。总量不重要。

第 6、7 和 8 行中的变量定义是有效的，因为它们符合类型提示。不允许使用`numbers`的其他定义:

*   **第 11 行**不包含同质项目。
*   **第 12 行**不是元组，是列表。

如果您安装了 mypy，那么您可以使用 mypy 运行代码来列出这两个错误:

```py
$ mypy tuple_example.py
tuple_example.py:11: error: Incompatible types in assignment
 (expression has type "Tuple[int, str]", variable has type "Tuple[int, ...]")
tuple_example.py:12: error: Incompatible types in assignment
 (expression has type "List[int]", variable has type "Tuple[int, ...]")
```

在 tuple 类型提示中使用`...`意味着您希望 tuple 中的所有项都是相同的类型。

另一方面，当您对[可调用类型](https://mypy.readthedocs.io/en/stable/kinds_of_types.html#callable-types-and-lambdas)使用省略号文字时，您实际上解除了对如何调用**可调用类型**的一些限制，可能是在参数的数量或类型方面:

```py
 1from typing import Callable
 2
 3def add_one(i: int) -> int:
 4    return i + 1
 5
 6def multiply_with(x: int, y: int) -> int:
 7    return x * y
 8
 9def as_pixels(i: int) -> str:
10    return f"{i}px"
11
12def calculate(i: int, action: Callable[..., int], *args: int) -> int: 13    return action(i, *args)
14
15# Works:
16calculate(1, add_one)
17calculate(1, multiply_with, 3)
18
19# Doesn't work:
20calculate(1, 3)
21calculate(1, as_pixels)
```

在第 12 行，您定义了一个可调用的参数，`action`。这个可调用函数可以接受任意数量和类型的参数，但必须返回一个整数。有了`*args: int`，你还允许可变数量的[可选参数](https://realpython.com/python-optional-arguments/)，只要它们是整数。在第 13 行的`calculate()`函数体中，用整数`i`和任何其他传入的参数调用`action`。

当你定义一个[可调用类型](https://mypy.readthedocs.io/en/latest/kinds_of_types.html#callable-types-and-lambdas)时，你必须让 Python 知道你允许什么类型作为输入，以及你期望这个可调用类型返回什么类型。通过使用`Callable[..., int]`，您说您不介意这个可调用函数接受多少和哪些类型的参数。然而，您已经指定它必须返回一个整数。

您在第 16 行和第 17 行中作为参数传递给`calculate()`的函数符合您设置的规则。`add_one()`和`multiply_with()`都是返回整数的可调用函数。

第 20 行的代码无效，因为`3`是不可调用的。可调用函数必须是你可以调用的东西，因此得名。

虽然`as_pixels()`是可调用的，但是它在第 21 行的用法也是无效的。在第 10 行，你正在创建一个 [f 弦](https://realpython.com/python-formatted-output/#the-python-formatted-string-literal-f-string)。您得到一个字符串作为返回值，这不是您期望的整数类型。

在上面的例子中，您已经研究了如何在元组和可调用的类型提示中使用省略号文字:

| 类型 | `Ellipsis`用法 |
| --- | --- |
| `tuple` | 用统一类型定义未知长度的数据元组 |
| `Callable` | 代表可调用的参数列表，移除限制 |

接下来，您将学习如何在 NumPy 中使用`Ellipsis`。

[*Remove ads*](/account/join/)

## 在 NumPy 中如何使用省略号进行切片？

如果你以前和 **NumPy** 一起工作过，那么你可能会遇到`Ellipsis`的另一种用法。在 [NumPy](https://realpython.com/numpy-tutorial/) 中，您可以用省略号文本分割多维数组。

从一个没有利用 NumPy 中的`Ellipsis`的例子开始:

>>>

```py
>>> import numpy as np

>>> dimensions = 3
>>> items_per_dimension = 2
>>> max_items = items_per_dimension**dimensions
>>> axes = np.repeat(items_per_dimension, dimensions)
>>> arr = np.arange(max_items).reshape(axes)
>>> arr
array([[[0, 1],
 [2, 3]],

 [[4, 5],
 [6, 7]]])
```

在本例中，您将通过组合 NumPy 中的 [`.arange()`](https://realpython.com/how-to-use-numpy-arange/) 和 [`.reshape()`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html) 来创建一个[三维数组](https://realpython.com/numpy-array-programming/)。

如果您想指定最后一个维度的第一项，那么您可以借助冒号(`:`)用[分割 NumPy 数组](https://note.nkmk.me/en/python-numpy-ndarray-slice/):

>>>

```py
>>> arr[:, :, 0]
array([[0, 2],
 [4, 6]])
```

因为`arr`有三个维度，所以需要指定三个切片。但是想象一下，如果你增加更多的维度，语法会变得多么烦人！更糟糕的是，你无法判断一个数组有多少个维度:

>>>

```py
>>> import numpy as np

>>> dimensions = np.random.randint(1,10)
>>> items_per_dimension = 2
>>> max_items = items_per_dimension**dimensions
>>> axes = np.repeat(items_per_dimension, dimensions)
>>> arr = np.arange(max_items).reshape(axes)
```

在本例中，您正在创建一个最多可以有十个维度的数组。你可以使用 [NumPy 的`.ndim()`T4 来找出`arr`有多少个维度。但是在这种情况下，使用`...`是更好的方法:](https://note.nkmk.me/en/python-numpy-ndarray-ndim-shape-size/)

>>>

```py
>>> arr[..., 0]
array([[[[ 0,  2],
 [ 4,  6]],

 [[ 8, 10],
 [12, 14]]],

 [[[16, 18],
 [20, 22]],

 [[24, 26],
 [28, 30]]]])
```

这里，`arr`有五个维度。因为维度的数量是随机的，所以您的输出可能看起来不同。尽管如此，用`...`来指定你的多维数组还是可以的。

NumPy 提供了更多的选项来使用`Ellipsis`来指定一个元素或者数组的范围。查看 [NumPy: `Ellipsis` ( `...` ) for `ndarray`](https://note.nkmk.me/en/python-numpy-ellipsis/) ，发现这三个小点的更多用例。

## Python 中的三个点永远是省略号吗？

一旦你学习了 Python 的`Ellipsis`，你可能会更加注意 Python 世界中每个省略号的出现。然而，你可能会在 Python 中看到三个点，*不*代表`Ellipsis`常量。

在 Python 交互 shell 中，三个点表示**二级提示**:

>>>

```py
>>> def hello_world():
...     print("Hello, world!")
...
```

例如，当您在 [Python 解释器](https://docs.python.org/3/tutorial/interpreter.html)中[定义函数](https://realpython.com/defining-your-own-python-function/)时，或者当您创建 [`for`循环](https://realpython.com/python-for-loop/)时，提示会持续多行。

在上面的例子中，这三个点不是省略号，而是函数体的二级提示。

在 Python 中，你还在其他地方发现过三个点吗？请在下面的评论中与真正的 Python 社区分享您的发现！

[*Remove ads*](/account/join/)

## 结论

省略号文字(`...`)计算为`Ellipsis`常量。最常见的是，你可以使用`...`作为占位符，例如当你创建函数的存根时。

在类型提示中，这三个点可以在你需要灵活性的时候派上用场。您可以指定一个同质类型的可变长度元组，并用省略号文本替换可调用类型的参数列表。

如果您使用 NumPy，那么您可以使用`...`通过用`Ellipsis`对象替换可变长度维度来简化切片语法。有了这三个点提供的整洁的语法，您可以使您的代码更具可读性。

根据经验，你可以记住你通常使用 Python 的`Ellipsis`来省略代码。有些人甚至会说省略号可以让不完整的代码看起来很可爱(T2)

**源代码:** [点击这里下载免费的源代码](https://realpython.com/bonus/python-ellipsis-code/)，你将使用它来掌握省略号文字。***