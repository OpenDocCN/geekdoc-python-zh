# Python 中的基本输入、输出和字符串格式

> 原文：<https://realpython.com/python-input-output/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 中的阅读输入和写作输出**](/courses/reading-input-writing-output-python/)

对于一个有用的程序，它通常需要通过从用户那里获得输入数据并向用户显示结果数据来与外界进行通信。在本教程中，您将了解 Python 的输入和输出。

输入可以由用户直接通过键盘输入，也可以来自外部资源，如文件或数据库。输出可以直接显示到控制台或 IDE，通过图形用户界面(GUI)显示到屏幕，或者再次显示到外部源。

在本介绍性系列的[之前的教程](https://realpython.com/python-for-loop/)中，您将:

*   比较了编程语言用来实现确定迭代的不同范例
*   了解了 iterables 和 iterators，这两个概念构成了 Python 中明确迭代的基础
*   将它们联系在一起，学习 Python 的 [for loops](https://realpython.com/python-for-loop/)

**本教程结束时，你将知道如何:**

*   通过内置功能 **`input()`** 从键盘上接受用户输入
*   用内置函数 **`print()`** 显示输出到控制台
*   使用 **Python f-strings** 格式化字符串数据

事不宜迟，我们开始吧！

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 从键盘读取输入

程序经常需要从用户那里获取数据，通常是通过键盘输入的方式。在 Python 中实现这一点的一种方法是使用 [`input()`](https://docs.python.org/3/library/functions.html#input) :

> `input([<prompt>])`
> 
> 从键盘上读取一行。([文档](https://docs.python.org/3/library/functions.html#input)

`input()`功能暂停程序执行，以允许用户从键盘键入一行输入。一旦用户按下 `Enter` 键，所有键入的字符被读取并作为[字符串](https://realpython.com/python-strings/)返回:

>>>

```py
>>> user_input = input()
foo bar baz
>>> user_input
'foo bar baz'
```

请注意，您的返回字符串不包括用户按下 `Enter` 键时生成的换行符。

如果包含可选的`<prompt>`参数，那么`input()`会将其显示为一个提示，以便用户知道应该输入什么:

>>>

```py
>>> name = input("What is your name? ")
What is your name? Winston Smith
>>> name
'Winston Smith'
```

`input()`总是返回一个字符串。如果你想要一个数字类型，那么你需要用内置的`int()`、`float()`或`complex()`函数将字符串转换成合适的类型:

>>>

```py
 1>>> number = input("Enter a number: ")
 2Enter a number: 50
 3>>> print(number + 100) 4Traceback (most recent call last):
 5  File "<stdin>", line 1, in <module>
 6TypeError: must be str, not int
 7
 8>>> number = int(input("Enter a number: ")) 9Enter a number: 50
10>>> print(number + 100) 11150
```

在上面的例子中，第 3 行的表达式`number + 100`是无效的，因为`number`是一个字符串，而`100`是一个整数。为了避免出现这种错误，第 8 行在收集用户输入后立即将`number`转换成一个整数。这样，第 10 行的计算`number + 100`有两个整数要相加。正因为如此，对 [`print()`](https://realpython.com/python-print/) 的调用成功。

**Python 版本注意:**如果您发现自己正在使用 Python 2.x 代码，您可能会发现 Python 版本 2 和 3 的输入函数略有不同。

Python 2 中的`raw_input()`从键盘读取输入并返回。如上所述，Python 2 中的`raw_input()`的行为就像 Python 3 中的`input()`。

但是 Python 2 也有一个函数叫做`input()`。在 Python 2 中，`input()`从键盘读取输入，*将其作为 Python 表达式*进行解析和求值，并返回结果值。

Python 3 没有提供一个函数来完成 Python 2 的`input()`所做的事情。您可以用表达式`eval(input())`模仿 Python 3 中的效果。但是，这是一个安全风险，因为它允许用户运行任意的、潜在的恶意代码。

有关`eval()`及其潜在安全风险的更多信息，请查看 [Python eval():动态评估表达式](https://realpython.com/python-eval-function/)。

使用`input()`，你可以从你的用户那里收集数据。但是如果你想向他们展示你的程序计算出的任何结果呢？接下来，您将学习如何在控制台中向用户显示输出。

[*Remove ads*](/account/join/)

## 将输出写入控制台

除了从用户那里获取数据，程序通常还需要将数据返回给用户。用 Python 中的 [`print()`](https://docs.python.org/3/library/functions.html#print) 可以将程序数据显示到控制台。

要向控制台显示对象，请将它们作为逗号分隔的参数列表传递给`print()`。

> `print(<obj>, ..., <obj>)`
> 
> 向控制台显示每个`<obj>`的字符串表示。([文档](https://docs.python.org/3/library/functions.html#print))

默认情况下，`print()`用一个空格分隔对象，并在输出的末尾附加一个新行:

>>>

```py
>>> first_name = "Winston"
>>> last_name = "Smith"

>>> print("Name:", first_name, last_name)
Name: Winston Smith
```

您可以指定任何类型的对象作为`print()`的参数。如果一个对象不是一个字符串，那么`print()`在显示它之前将它转换成一个合适的字符串表示:

>>>

```py
>>> example_list = [1, 2, 3]
>>> type(example_list)
<class 'list'>

>>> example_int = -12
>>> type(example_int)
<class 'int'>

>>> example_dict = {"foo": 1, "bar": 2}
>>> type(example_dict)
<class 'dict'>

>>> type(len)
<class 'builtin_function_or_method'>

>>> print(example_list, example_int, example_dict, len)
[1, 2, 3] -12 {'foo': 1, 'bar': 2} <built-in function len>
```

如你所见，甚至像[列表](https://realpython.com/python-lists-tuples/)、[字典](https://realpython.com/python-dicts/)和[函数](https://realpython.com/defining-your-own-python-function/)这样的复杂类型也可以用`print()`显示到控制台。

## 具有高级功能的打印

`print()`接受一些额外的参数，对输出的格式提供适度的控制。每一个都是一种特殊类型的论点，叫做**关键词论点**。在这个介绍性系列的后面，您将会遇到一个关于[函数和参数传递](https://realpython.com/defining-your-own-python-function/)的教程，这样您就可以了解更多关于关键字参数的知识。

不过，现在你需要知道的是:

*   关键字参数的形式为`<keyword>=<value>`。
*   传递给`print()`的任何关键字参数必须出现在末尾，在要显示的对象列表之后。

在下面几节中，您将看到这些关键字参数如何影响由`print()`产生的控制台输出。

### 分离打印值

添加关键字参数`sep=<str>`会导致 Python 通过<str>而不是默认的单个空格:</str>来分隔对象

>>>

```py
>>> print("foo", 42, "bar")
foo 42 bar

>>> print("foo", 42, "bar", sep="/")
foo/42/bar

>>> print("foo", 42, "bar", sep="...")
foo...42...bar

>>> d = {"foo": 1, "bar": 2, "baz": 3}
>>> for k, v in d.items():
...     print(k, v, sep=" -> ")
...
foo -> 1
bar -> 2
baz -> 3
```

要将对象挤在一起，中间没有任何空间，请指定一个空字符串(`""`)作为分隔符:

>>>

```py
>>> print("foo", 42, "bar", sep="")
foo42bar
```

您可以使用`sep`关键字指定任意字符串作为分隔符。

[*Remove ads*](/account/join/)

### 控制换行符

关键字参数`end=<str>`导致输出由`<str>`终止，而不是由默认换行符终止:

>>>

```py
>>> if True:
...     print("foo", end="/")
...     print(42, end="/")
...     print("bar")
...
foo/42/bar
```

例如，如果您在一个循环中显示值，您可以使用`end`使值显示在一行上，而不是单独的行上:

>>>

```py
>>> for number in range(10):
...     print(number)
...
0
1
2
3
4
5
6
7
8
9

>>> for number in range(10):
...     print(number, end=(" " if number < 9 else "\n"))
...
0 1 2 3 4 5 6 7 8 9
```

您可以使用`end`关键字将任何字符串指定为输出终止符。

### 将输出发送到流

`print()`接受两个额外的关键字参数，这两个参数都会影响函数处理输出流的方式:

1.  **`file=<stream>` :** 默认情况下，`print()`将其输出发送到一个名为`sys.stdout`的默认流，这个流通常相当于控制台。`file=<stream>`参数使`print()`将输出发送到由`<stream>`指定的替代流。

2.  **`flush=True` :** 通常，`print()`缓冲其输出，只间歇地写入输出流。`flush=True`指定 Python 在每次调用`print()`时强制刷新输出流。

为了完整起见，这里给出了这两个关键字参数。在学习旅程的这个阶段，您可能不需要太关心输出流。

## 使用格式化字符串

虽然您可以深入了解 [Python `print()`函数](https://realpython.com/python-print/)，但它提供的控制台输出格式充其量只是初步的。您可以选择如何分离打印的对象，并指定打印行末尾的内容。大概就是这样。

在许多情况下，您需要更精确地控制要显示的数据的外观。Python 提供了几种格式化输出字符串数据的方法。在本节中，您将看到一个使用 [Python f-strings 格式化字符串](https://realpython.com/python-f-strings/#f-strings-a-new-and-improved-way-to-format-strings-in-python)的例子。

**注意:****f-string 语法**是字符串格式化的现代方法之一。要进行深入讨论，您可能需要查看这些教程:

*   [Python 字符串格式化技巧&最佳实践](https://realpython.com/python-string-formatting/)
*   [Python 3 的 f-Strings:改进的字符串格式化语法(指南)](https://realpython.com/python-f-strings/)

在 Python 中的[格式化字符串输出的教程中，您还将更详细地了解字符串格式化的两种方法，f-strings 和`str.format()`，该教程在本介绍性系列教程的后面。](https://realpython.com/python-formatted-output/)

在本节中，您将使用 f 字符串来格式化您的输出。假设您编写了一些要求用户输入姓名和年龄的代码:

>>>

```py
>>> name = input("What is your name? ")
What is your name? Winston

>>> age = int(input("How old are you? "))
How old are you? 24

>>> print(name)
Winston

>>> print(age)
24
```

您已经成功地从您的用户那里收集了数据，并且您还可以将其显示回他们的控制台。要创建格式良好的输出消息，可以使用 f 字符串语法:

>>>

```py
>>> f"Hello, {name}. You are {age}."
Hello, Winston. You are 24.
```

string 允许你把变量名放在花括号(`{}`)中，把它们的值注入到你正在构建的字符串中。你所需要做的就是在字符串的开头加上字母`f`或者`F`。

接下来，假设您想告诉您的用户 50 年后他们的年龄。Python f-strings 允许您在没有太多开销的情况下做到这一点！您可以在花括号之间添加任何 **Python 表达式**，Python 将首先计算它的值，然后将其注入到您的 f 字符串中:

>>>

```py
>>> f"Hello, {name}. In 50 years, you'll be {age + 50}."
Hello, Winston. In 50 years, you'll be 74.
```

您已经将`50`添加到从用户处收集的`age`的值中，并在前面使用`int()`将其转换为整数。整个计算发生在 f 弦的第二对花括号里。相当酷！

**注意:**如果你想了解更多关于使用这种方便的字符串格式化技术，那么你可以更深入地阅读关于 [Python 3 的 f-Strings](https://realpython.com/python-f-strings/) 的指南。

Python f-strings 可以说是在 Python 中格式化字符串的最方便的方式。如果你只想学习一种方法，最好坚持使用 Python 的 f 字符串。然而，这种语法从 Python 3.6 开始才可用，所以如果您需要使用 Python 的旧版本，那么您将不得不使用不同的语法，例如`str.format()`方法或字符串模运算符。

[*Remove ads*](/account/join/)

## Python 输入和输出:结论

在本教程中，您学习了 Python 中的输入和输出，以及 Python 程序如何与用户通信。您还研究了一些参数，您可以使用这些参数向输入提示添加消息，或者定制 Python 如何向用户显示输出。

**你已经学会了如何:**

*   通过内置功能 **`input()`** 从键盘上接受用户输入
*   用内置函数 **`print()`** 显示输出到控制台
*   使用 **Python f-strings** 格式化字符串数据

在这个介绍性系列的下一篇教程中，您将学习另一种字符串格式化技术，并且将更深入地使用 f 字符串。

[« Python "for" Loops (Definite Iteration)](https://realpython.com/python-for-loop/)[Basic Input and Output in Python](#)[Python String Formatting Techniques »](https://realpython.com/python-formatted-output/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 中的阅读输入和写作输出**](/courses/reading-input-writing-output-python/)*****