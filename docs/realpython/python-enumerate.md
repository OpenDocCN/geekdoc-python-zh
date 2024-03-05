# Python enumerate():用计数器简化循环

> 原文：<https://realpython.com/python-enumerate/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**用 Python enumerate()循环**](/courses/looping-with-python-enumerate/)

在 Python 中， [`for`循环](https://realpython.com/python-for-loop/)通常被写成对可迭代对象的循环。这意味着您不需要计数变量来访问 iterable 中的项。但是，有时您确实希望有一个在每次循环迭代中都发生变化的变量。您可以使用 Python 的 **`enumerate()`** 同时从 iterable 中获取计数器和值，而不是自己创建和递增变量！

**在本教程中，您将了解如何:**

*   使用 **`enumerate()`** 在循环中获得一个计数器
*   将`enumerate()`应用到**显示项目计数**
*   将`enumerate()`与**条件语句**一起使用
*   实现自己的**等价函数**到`enumerate()`
*   **解包`enumerate()`返回的值**

我们开始吧！

**免费下载:** [从 CPython Internals:您的 Python 3 解释器指南](https://realpython.com/bonus/cpython-internals-sample/)获得一个示例章节，向您展示如何解锁 Python 语言的内部工作机制，从源代码编译 Python 解释器，并参与 CPython 的开发。

## 用 Python 中的`for`循环迭代

Python 中的`for`循环使用了**基于集合的迭代**。这意味着 Python 会在每次迭代时将 iterable 中的下一项分配给循环变量，如下例所示:

>>>

```py
>>> values = ["a", "b", "c"]

>>> for value in values:
...     print(value)
...
a
b
c
```

在这个例子中，`values`是一个带有三个[字符串](https://realpython.com/python-strings/)、`"a"`、`"b"`和`"c"`的[列表](https://realpython.com/python-lists-tuples/)。在 Python 中，列表是一种可迭代对象。在`for`循环中，循环变量是`value`。在循环的每次迭代中，`value`被设置为从`values`开始的下一项。

接下来，你[将](https://realpython.com/python-print/)打印到屏幕上。基于集合的迭代的优点是它有助于避免其他编程语言中常见的[逐个错误](https://en.wikipedia.org/wiki/Off-by-one_error)。

现在想象一下，除了值本身之外，您还想在每次迭代时将列表中项的索引打印到屏幕上。完成这项任务的一种方法是创建一个变量来存储索引，并在每次迭代中更新它:

>>>

```py
>>> index = 0

>>> for value in values:
...     print(index, value)
...     index += 1
...
0 a
1 b
2 c
```

在这个例子中，`index`是一个整数，记录你在列表中的位置。在循环的每次迭代中，你打印出`index`和`value`。循环的最后一步是将存储在`index`中的数字更新 1。当你忘记在每次迭代中更新`index`时，会出现一个常见的错误:

>>>

```py
>>> index = 0

>>> for value in values:
...     print(index, value)
...
0 a
0 b
0 c
```

在这个例子中，`index`在每次迭代中都停留在`0`上，因为没有代码在循环结束时更新它的值。特别是对于长的或复杂的循环，这种错误是出了名的难以追踪。

解决这个问题的另一种常见方法是使用 [`range()`](https://docs.python.org/3/library/stdtypes.html#range) 结合 [`len()`](https://realpython.com/len-python-function/) 来自动创建索引。这样，您不需要记住更新索引:

>>>

```py
>>> for index in range(len(values)):
...     value = values[index]
...     print(index, value)
...
0 a
1 b
2 c
```

本例中，`len(values)`返回`values`的长度，也就是`3`。然后 [`range()`](https://realpython.com/python-range/) 创建一个迭代器，从默认的起始值`0`开始运行，直到到达`len(values)`减 1。在这种情况下，`index`成为你的循环变量。在循环中，您将`value`设置为等于当前值`index`的`values`中的项目。最后，你打印出`index`和`value`。

在这个例子中，一个可能发生的常见错误是在每次迭代开始时忘记更新`value`。这类似于之前忘记更新索引的 bug。这是这个循环不被认为是[python 式](https://realpython.com/courses/how-to-write-pythonic-loops/)的一个原因。

这个例子也有一些限制，因为`values`必须允许使用整数索引来访问它的项目。允许这种访问的可重复项在 Python 中被称为**序列**。

**技术细节:**根据 [Python 文档](https://docs.python.org/3/glossary.html#term-iterable)，一个 **iterable** 是任何可以一次返回一个成员的对象。根据定义，iterables 支持[迭代器协议](https://docs.python.org/3/library/stdtypes.html#typeiter)，该协议指定了在[迭代器](https://docs.python.org/3/glossary.html#term-iterator)中使用对象时如何返回对象成员。Python 有两种常用的可迭代类型:

1.  [序列](https://docs.python.org/3/glossary.html#term-sequence)
2.  [发电机](https://docs.python.org/3/glossary.html#term-generator)

任何 iterable 都可以在`for`循环中使用，但是只有序列可以被[整数](https://realpython.com/python-numbers/#integers)索引访问。试图通过索引从[生成器](http://www.dabeaz.com/generators/Generators.pdf)或[迭代器](https://dbader.org/blog/python-iterators)中访问项目将引发`TypeError`:

>>>

```py
>>> enum = enumerate(values)
>>> enum[0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'enumerate' object is not subscriptable
```

在这个例子中，你把`enumerate()`的返回值赋给 [`enum`](https://realpython.com/python-enum/) 。`enumerate()`是一个迭代器，所以试图通过索引访问它的值会引发一个`TypeError`。

幸运的是，Python 的 [`enumerate()`](https://docs.python.org/3/library/functions.html#enumerate) 让你避免了所有这些问题。这是一个**内置的**函数，这意味着自从 2003 年在 Python 2.3 中添加了[以来，它在 Python 的每个版本中都可用。](https://www.python.org/dev/peps/pep-0279/)

[*Remove ads*](/account/join/)

## 使用 Python 的`enumerate()`

你可以在一个循环中使用 [`enumerate()`](https://realpython.com/courses/how-to-write-pythonic-loops/) ，就像你使用原始的 iterable 对象一样。不是将 iterable 直接放在`for`循环中的`in`之后，而是放在`enumerate()`的括号内。您还必须稍微更改循环变量，如下例所示:

>>>

```py
>>> for count, value in enumerate(values):
...     print(count, value)
...
0 a
1 b
2 c
```

当您使用`enumerate()`时，该函数会返回给*两个*循环变量:

1.  当前迭代的**计数**
2.  当前迭代中项的**值**

就像普通的`for`循环一样，循环变量可以被命名为您想要的任何名称。在这个例子中使用了`count`和`value`，但是它们可以被命名为`i`和`v`或者任何其他有效的 Python 名称。

使用`enumerate()`，您不需要记得从 iterable 中访问项，也不需要记得在循环结束时推进索引。Python 的魔力会自动为您处理所有事情！

**技术细节:**使用逗号分隔的两个循环变量`count`和`value`是[参数解包](https://realpython.com/defining-your-own-python-function/#argument-tuple-unpacking)的一个例子。本文稍后将进一步讨论这个强大的 Python 特性。

Python 的`enumerate()`有一个额外的参数，可以用来控制计数的起始值。默认情况下，起始值是`0`，因为 Python 序列类型的索引是从零开始的。换句话说，当你想检索一个列表的第一个元素时，你可以使用 index `0`:

>>>

```py
>>> print(values[0])
a
```

在这个例子中可以看到，用索引`0`访问`values`给出了第一个元素`a`。然而，很多时候您可能不希望从`enumerate()`开始计数，而是从`0`开始计数。例如，您可能希望为用户输出一个自然计数。在这种情况下，您可以使用`enumerate()`的`start`参数来更改起始计数:

>>>

```py
>>> for count, value in enumerate(values, start=1):
...     print(count, value)
...
1 a
2 b
3 c
```

在这个例子中，您传递了`start=1`，它在第一次循环迭代中以值`1`开始`count`。将它与前面的例子进行比较，在前面的例子中，`start`的默认值是`0`，看看您是否能发现不同之处。

## 用 Python `enumerate()`练习

每当你需要在循环中使用计数和一个项目时，你都应该使用`enumerate()`。请记住，`enumerate()`会在每次迭代时将计数递增 1。然而，这只是稍微限制了您的灵活性。因为 count 是一个标准的 Python 整数，所以可以以多种方式使用它。在接下来的几节中，您将看到`enumerate()`的一些用法。

### 可迭代项目的自然计数

在上一节中，您看到了如何使用`enumerate()`和`start`来为用户创建一个自然计数。`enumerate()`在 Python 代码库中也是这样使用的。您可以在一个脚本中看到一个例子，它读取 reST 文件并在出现格式问题时告诉用户。

**注意:** reST，也称为[重构文本](https://en.wikipedia.org/wiki/ReStructuredText)，是 Python 用于文档的文本文件的标准格式。在 Python 类和函数中，您会经常看到 reST 格式的字符串作为[文档字符串](https://realpython.com/documenting-python-code/)出现。读取源代码文件并告诉用户格式问题的脚本被称为**linter**，因为它们在代码中寻找隐喻的 [lint](https://en.wikipedia.org/wiki/Lint_(software)) 。

这个例子是由 [`rstlint.py`](https://github.com/python/cpython/blob/2d6097d027e0dd3debbabc702aa9c98d94ba32a3/Doc/tools/rstlint.py#L96-L104) 略加修改而来。不要太担心这个函数如何检查问题。重点是展示`enumerate()`的真实使用情况:

```py
 1def check_whitespace(lines):
 2    """Check for whitespace and line length issues."""
 3    for lno, line in enumerate(lines): 4        if "\r" in line:
 5            yield lno+1, "\\r in line" 6        if "\t" in line:
 7            yield lno+1, "OMG TABS!!!1" 8        if line[:-1].rstrip(" \t") != line[:-1]:
 9            yield lno+1, "trailing whitespace"
```

`check_whitespace()`接受一个参数`lines`，它是应该被评估的文件的行。在`check_whitespace()`的第三行，`enumerate()`在`lines`上方循环使用。这将返回行号，缩写为`lno`和`line`。因为没有使用`start`，所以`lno`是文件中行的从零开始的计数器。`check_whitespace()`然后对错位字符进行多次检查:

1.  回车(`\r`)
2.  制表符(`\t`)
3.  行尾有空格或制表符吗

当这些项目之一出现时，`check_whitespace()` [产生](https://realpython.com/introduction-to-python-generators/#understanding-the-python-yield-statement)当前行号和对用户有用的消息。计数变量`lno`中添加了`1`，因此它返回计数行号，而不是从零开始的索引。当`rstlint.py`的用户阅读消息时，他们会知道去哪一行和修复什么。

[*Remove ads*](/account/join/)

### 跳过项目的条件语句

使用条件语句处理项目可能是一种非常强大的技术。有时，您可能只需要在循环的第一次迭代中执行操作，如下例所示:

>>>

```py
>>> users = ["Test User", "Real User 1", "Real User 2"]
>>> for index, user in enumerate(users):
...     if index == 0:
...         print("Extra verbose output for:", user)
...     print(user)
...
Extra verbose output for: Test User
Real User 1
Real User 2
```

在这个例子中，您使用一个列表作为用户的模拟数据库。第一个用户是您的测试用户，因此您想要打印关于该用户的额外诊断信息。因为您已经设置了系统，测试用户是第一个，所以您可以使用循环的第一个索引值来打印额外的详细输出。

您还可以将数学运算与计数或索引的条件结合起来。例如，您可能需要从 iterable 中返回项，但前提是它们有偶数索引。您可以通过使用`enumerate()`来完成此操作:

>>>

```py
>>> def even_items(iterable):
...     """Return items from ``iterable`` when their index is even."""
...     values = []
...     for index, value in enumerate(iterable, start=1):
...         if not index % 2:
...             values.append(value)
...     return values
...
```

`even_items()`接受一个名为`iterable`的参数，它应该是某种 Python 可以循环的对象类型。首先，`values`被初始化为一个空列表。然后用`enumerate()`在`iterable`上创建一个`for`循环，并设置`start=1`。

在`for`循环中，检查`index`除以`2`的余数是否为零。如果是，那么您[将](https://realpython.com/python-append/)项添加到`values`中。最后，你[返回](https://realpython.com/python-return-statement/) `values`。

你可以通过使用一个[列表理解](https://realpython.com/list-comprehension-python/)在一行中做同样的事情，而不用初始化空列表，从而使代码更加[python 化](https://realpython.com/python-pep8/):

>>>

```py
>>> def even_items(iterable):
...     return [v for i, v in enumerate(iterable, start=1) if not i % 2]
...
```

在这个示例代码中，`even_items()`使用一个列表理解而不是一个`for`循环来从列表中提取索引为偶数的每个项目。

您可以通过从从`1`到`10`的整数范围中获取偶数索引项来验证`even_items()`是否按预期工作。结果将是`[2, 4, 6, 8, 10]`:

>>>

```py
>>> seq = list(range(1, 11))

>>> print(seq)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

>>> even_items(seq)
[2, 4, 6, 8, 10]
```

正如所料，`even_items()`从`seq`返回偶数索引的项目。当你处理整数时，这不是获得偶数的最有效的方法。然而，现在您已经验证了`even_items()`工作正常，您可以获得 ASCII 字母表的偶数索引字母:

>>>

```py
>>> alphabet = "abcdefghijklmnopqrstuvwxyz"

>>> even_items(alphabet)
['b', 'd', 'f', 'h', 'j', 'l', 'n', 'p', 'r', 't', 'v', 'x', 'z']
```

`alphabet`是包含 ASCII 字母表中所有 26 个小写字母的字符串。调用`even_items()`并传递`alphabet`会返回字母表中交替字母的列表。

Python 字符串是序列，可以在循环中使用，也可以在整数索引和切片中使用。所以在字符串的情况下，您可以使用方括号来更有效地实现与`even_items()`相同的功能:

>>>

```py
>>> list(alphabet[1::2])
['b', 'd', 'f', 'h', 'j', 'l', 'n', 'p', 'r', 't', 'v', 'x', 'z']
```

在这里使用[字符串切片](https://realpython.com/courses/python-strings/)，给出起始索引`1`，它对应于第二个元素。第一个冒号后没有结束索引，所以 Python 转到了字符串的末尾。然后添加第二个冒号，后跟一个`2`,这样 Python 将接受所有其他元素。

然而，正如您之前看到的，生成器和迭代器不能被索引或切片，所以您仍然会发现`enumerate()`很有用。继续前面的例子，您可以创建一个[生成器函数](https://realpython.com/introduction-to-python-generators/)，它按需生成字母表中的字母:

>>>

```py
>>> def alphabet():
...     alpha = "abcdefghijklmnopqrstuvwxyz"
...     for a in alpha:
...         yield a

>>> alphabet[1::2]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'function' object is not subscriptable

>>> even_items(alphabet())
['b', 'd', 'f', 'h', 'j', 'l', 'n', 'p', 'r', 't', 'v', 'x', 'z']
```

在这个例子中，您定义了`alphabet()`，一个生成器函数，当在一个循环中使用这个函数时，它会一个接一个地生成字母表中的字母。Python 函数，无论是生成器还是常规函数，都不能通过方括号索引来访问。你在第二行试试这个，它引发了一个`TypeError`。

不过，您可以在循环中使用生成器函数，在最后一行通过将`alphabet()`传递给`even_items()`来这样做。可以看到结果和前面两个例子是一样的。

[*Remove ads*](/account/join/)

## 了解 Python `enumerate()`

在前几节中，您已经看到了何时以及如何使用`enumerate()`的例子。现在您已经掌握了`enumerate()`的实际方面，您可以学习更多关于函数内部如何工作的内容。

为了更好地理解`enumerate()`是如何工作的，您可以用 Python 实现自己的版本。你版本的`enumerate()`有两个要求。它应该:

1.  接受 iterable 和起始计数值作为参数
2.  从 iterable 发回一个包含当前计数值和相关项的元组

编写满足这些规范的函数的一种方法在 [Python 文档](https://docs.python.org/3/library/functions.html#enumerate)中给出:

>>>

```py
>>> def my_enumerate(sequence, start=0):
...     n = start
...     for elem in sequence:
...         yield n, elem
...         n += 1
...
```

`my_enumerate()`有两个论点:`sequence`和`start`。`start`的默认值为`0`。在函数定义中，您将`n`初始化为`start`的值，并在`sequence`上运行`for`循环。

对于`sequence`中的每个`elem`，你`yield`控制回调用位置，发回`n`和`elem`的当前值。最后，增加`n`为下一次迭代做准备。你可以在这里看到`my_enumerate()`的动作:

>>>

```py
>>> seasons = ["Spring", "Summer", "Fall", "Winter"]

>>> my_enumerate(seasons)
<generator object my_enumerate at 0x7f48d7a9ca50>

>>> list(my_enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]

>>> list(my_enumerate(seasons, start=1))
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

首先，创建一个四季列表。接下来，您将看到用`seasons`作为`sequence`调用`my_enumerate()`会创建一个生成器对象。这是因为您使用了`yield`关键字将值发送回调用者。

最后，从`my_enumerate()`开始创建两个列表，其中一个列表的起始值保留为默认值`0`，另一个列表的`start`更改为`1`。在这两种情况下，最终都会得到一个元组列表，其中每个元组的第一个元素是计数，第二个元素是来自`seasons`的值。

虽然您可以只用几行 Python 代码实现一个与`enumerate()`等价的函数，但是`enumerate()` [的实际代码是用 C](https://github.com/python/cpython/blob/c8ba47b5518f83b5766fefe6f68557b5033e1d70/Objects/enumobject.c) 编写的。这意味着它是超级快速和高效的。

## 用`enumerate()` 解包参数

当您在一个`for`循环中使用`enumerate()`时，您告诉 Python 使用两个变量，一个用于计数，一个用于值本身。你可以通过使用一个叫做**参数解包**的 Python 概念来做到这一点。

参数解包的思想是一个元组可以根据序列的长度分成几个变量。例如，您可以将两个元素的元组解包为两个变量:

>>>

```py
>>> tuple_2 = (10, "a")
>>> first_elem, second_elem = tuple_2
>>> first_elem
10
>>> second_elem
'a'
```

首先，创建一个包含两个元素的元组，`10`和`"a"`。然后将该元组解包为`first_elem`和`second_elem`，每个元组被赋予一个来自该元组的值。

当您调用`enumerate()`并传递一系列值时，Python 返回一个**迭代器**。当您向迭代器请求下一个值时，它会产生一个包含两个元素的元组。元组的第一个元素是计数，第二个元素是您传递的序列中的值:

>>>

```py
>>> values = ["a", "b"]
>>> enum_instance = enumerate(values)
>>> enum_instance
<enumerate at 0x7fe75d728180>
>>> next(enum_instance)
(0, 'a')
>>> next(enum_instance)
(1, 'b')
>>> next(enum_instance)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

在本例中，您创建了一个名为`values`的列表，其中包含两个元素`"a"`和`"b"`。然后将`values`传递给`enumerate()`，并将返回值赋给`enum_instance`。当您打印`enum_instance`时，您可以看到它是一个具有特定内存地址的`enumerate()`的实例。

然后你用 Python 内置的 [`next()`](https://docs.python.org/3/library/functions.html#next) 从`enum_instance`中获取下一个值。`enum_instance`返回的第一个值是一个计数为`0`的元组，来自`values`的第一个元素是`"a"`。

在`enum_instance`上再次调用`next()`产生另一个元组，这一次使用计数`1`和来自`values`、`"b"`的第二个元素。最后，再次调用`next()`会引发`StopIteration`，因为`enum_instance`不再返回任何值。

当在`for`循环中使用 iterable 时，Python 会在每次迭代开始时自动调用`next()`，直到引发`StopIteration`为止。Python 将从 iterable 中检索到的值赋给循环变量。

如果 iterable 返回一个元组，那么可以使用参数解包将元组的元素分配给多个变量。这就是你在本教程前面通过使用两个循环变量所做的。

另一次你可能会看到用`for`循环解包参数是用内置的 [`zip()`](https://docs.python.org/3/library/functions.html#zip) ，它允许你同时迭代两个或更多的序列。在每次迭代中， [`zip()`](https://realpython.com/python-zip-function/) 返回一个元组，该元组收集所有传递的序列中的元素:

>>>

```py
>>> first = ["a", "b", "c"]
>>> second = ["d", "e", "f"]
>>> third = ["g", "h", "i"]
>>> for one, two, three in zip(first, second, third):
...     print(one, two, three)
...
a d g
b e h
c f i
```

通过使用`zip()`，你可以同时迭代`first`、`second`和`third`。在`for`循环中，从`first`到`one`，从`second`到`two`，从`third`到`three`分配元素。然后打印这三个值。

您可以通过使用**嵌套**参数解包来组合`zip()`和`enumerate()`:

>>>

```py
>>> for count, (one, two, three) in enumerate(zip(first, second, third)):
...     print(count, one, two, three)
...
0 a d g
1 b e h
2 c f i
```

在本例的`for`循环中，您将`zip()`嵌套在`enumerate()`中。这意味着每次`for`循环迭代时，`enumerate()`产生一个元组，第一个值作为计数，第二个值作为另一个元组，包含从参数到`zip()`的元素。为了解开嵌套结构，您需要添加括号来捕获来自`zip()`的嵌套元素元组中的元素。

还有其他方法可以模仿`enumerate()`结合 [`zip()`](https://realpython.com/python-zip-function/) 的行为。一种方法使用 [`itertools.count()`](https://realpython.com/python-itertools/#evens-and-odds) ，默认情况下返回从零开始的连续整数。你可以把前面的例子改成使用 [`itertools.count()`](https://docs.python.org/3/library/itertools.html#itertools.count) :

>>>

```py
>>> import itertools
>>> for count, one, two, three in zip(itertools.count(), first, second, third):
...     print(count, one, two, three)
...
0 a d g
1 b e h
2 c f i
```

在这个例子中使用`itertools.count()`允许您使用一个单独的`zip()`调用来生成计数以及循环变量，而不需要对嵌套参数进行解包。

[*Remove ads*](/account/join/)

## 结论

Python 的`enumerate()`允许您在需要来自 iterable 的计数和值时编写 python 式的`for`循环。`enumerate()`的一大优点是它返回一个带有计数器和值的元组，因此您不必自己递增计数器。它还为您提供了更改计数器初始值的选项。

**在本教程中，您学习了如何:**

*   在你的`for`循环中使用 Python 的 **`enumerate()`**
*   在几个**真实世界的例子中应用`enumerate()`**
*   使用**参数解包**从`enumerate()`获取值
*   实现自己的**等价函数**到`enumerate()`

您还看到了在一些真实世界的代码中使用的`enumerate()`，包括在 [CPython](https://realpython.com/cpython-source-code-guide/) 代码库中。现在，您拥有了简化循环并使 Python 代码变得时尚的超能力！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**用 Python enumerate()循环**](/courses/looping-with-python-enumerate/)******