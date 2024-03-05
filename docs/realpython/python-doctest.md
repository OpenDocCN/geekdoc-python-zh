# Python 的 doctest:一次记录并测试你的代码

> 原文:# t0]https://realython . com/python-doctest/

你有兴趣为你的代码编写同时作为**文档**和**测试用例**的用例吗？如果你的答案是*是*，那么 Python 的`doctest`模块就适合你。这个模块提供了一个测试框架，没有太陡的学习曲线。它允许您将代码示例用于两个目的:记录和测试您的代码。

除了允许你使用你的代码文档来测试代码本身，`doctest`还将帮助你在任何时候保持你的代码和它的文档的完美同步。

**在本教程中，您将:**

*   在代码的文档和文档字符串中编写 **`doctest`测试**
*   了解 **`doctest`如何在**内部工作
*   探索`doctest`的**限制**和**安全影响**
*   使用`doctest`进行**测试驱动开发**
*   使用不同的**策略和工具**运行你的`doctest`测试

您不必安装任何第三方库或学习复杂的 API 来遵循本教程。你只需要知道 Python 编程的基础，以及如何使用 Python [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) 或者[交互 shell](https://realpython.com/interacting-with-python/) 。

**示例代码:** [点击这里下载免费的示例代码](https://realpython.com/bonus/python-doctest-code/)，您将使用 Python 的`doctest`同时记录和测试您的代码。

## 用例子和测试记录你的代码

几乎所有有经验的程序员都会告诉你，[记录](https://realpython.com/documenting-python-code/)你的代码是一种最佳实践。有些人会说代码和它的文档同样重要和必要。其他人会告诉你文档甚至比代码本身更重要。

在 Python 中，你会发现许多记录项目、应用程序，甚至模块和脚本的方法。大型项目通常需要专门的外部文档。但是在小项目中，使用**显式名称**、[注释](https://realpython.com/python-comments-guide/)和**文档字符串**可能就足够了:

```py
 1"""This module implements functions to process iterables."""
 2
 3def find_value(value, iterable):
 4    """Return True if value is in iterable, False otherwise."""
 5    # Be explicit by using iteration instead of membership
 6    for item in iterable:
 7        if value == item:  # Find the target value by equality
 8            return True
 9    return False
```

像`find_value()`这样明确的名字可以帮助你清楚地表达给定对象的内容和目的。这样的名称可以提高代码的可读性和可维护性。

注释，比如第 5 行和第 7 行的注释，是您在代码的不同位置插入的文本片段，用来阐明代码做了什么以及为什么做。请注意，Python 注释以一个`#`符号开始，可以占据自己的行，也可以是现有行的一部分。

注释有几个缺点:

*   它们被解释器或编译器忽略，这使得它们在运行时不可访问。
*   当代码发展而注释保持不变时，他们通常会变得过时。

文档字符串，或简称为 [docstrings](https://realpython.com/python-project-documentation-with-mkdocs/#step-3-write-and-format-your-docstrings) ，是一个简洁的 Python 特性，可以帮助您在编写代码的过程中编写文档。与注释相比，文档字符串的优势在于解释器不会忽略它们。它们是你代码的活的一部分。

因为文档字符串是代码的活动部分，所以可以在运行时访问它们。为此，您可以在您的[包、模块](https://realpython.com/python-modules-packages/)、[类](https://realpython.com/python3-object-oriented-programming/#define-a-class-in-python)、方法和[函数](https://realpython.com/defining-your-own-python-function/)上使用`.__doc__`特殊属性。

像 [MkDocs](https://realpython.com/python-project-documentation-with-mkdocs/) 和 [Sphinx](https://www.sphinx-doc.org/en/master/) 这样的工具可以利用 docstrings 自动生成项目文档。

在 Python 中，可以将文档字符串添加到包、模块、类、方法和函数中。如果你想学习如何写好 docstrings，那么 [PEP 257](https://peps.python.org/pep-0257/) 提出了一系列你可以遵循的约定和建议。

当您编写 docstrings 时，通常的做法是为您的代码嵌入**用法示例**。这些示例通常模拟 REPL 会话。

在文档字符串中嵌入代码示例提供了一种记录代码的有效方法，以及在编写代码时测试代码的快速方法。是的，如果你以正确的方式编写代码并使用正确的工具运行它们，你的代码例子可以作为测试用例。

在代码中嵌入类似 REPL 的代码示例有助于:

*   保持文档与代码的当前状态同步
*   表达您代码的**预期用途**
*   在你写代码的时候测试你的代码

这些好处听起来不错！现在，如何运行嵌入到文档和文档字符串中的代码示例呢？可以从[标准库](https://docs.python.org/3/library/index.html)中使用 Python 的 [`doctest`](https://docs.python.org/3/library/doctest.html#module-doctest) 模块。

[*Remove ads*](/account/join/)

## 了解 Python 的`doctest`模块

在本节中，您将了解 Python 的 [`doctest`](https://docs.python.org/3/library/doctest.html#module-doctest) 模块。该模块是标准库的一部分，因此您不必安装任何第三方库就可以在日常编码中使用它。除此之外，您将学习什么是`doctest`以及何时使用这个简洁的 Python 工具。为了揭开序幕，你将从了解什么是`doctest`开始。

### 什么是`doctest`以及它是如何工作的

`doctest`模块是一个轻量级的**测试框架**，它提供了快速简单的[测试自动化](https://en.wikipedia.org/wiki/Test_automation)。它可以从您的项目文档和代码的文档字符串中读取测试用例。这个框架是 Python 解释器附带的，并且遵循电池包含的原则。

您可以在代码或命令行中使用`doctest`。为了找到并运行您的测试用例，`doctest`遵循几个步骤:

1.  在文档和文档字符串中搜索看起来像 Python [交互会话](https://realpython.com/interacting-with-python/)的文本
2.  解析这些文本片段，以区分可执行代码和预期结果
3.  像常规 Python 代码一样运行可执行代码
4.  将执行结果与预期结果进行比较

框架在你的文档以及包、模块、函数、类和方法的文档字符串中搜索测试用例。它不会在您[导入](https://realpython.com/python-import/)的任何对象中搜索测试用例。

一般来说，`doctest`将所有以主(`>>>`)或次(`...` ) REPL 提示符开始的文本行解释为可执行的 Python 代码。紧跟在任何一个提示后面的行被理解为代码的预期输出或结果。

### 什么`doctest`对有用

`doctest`框架非常适合于[验收测试](https://en.wikipedia.org/wiki/Acceptance_testing)在[集成](https://en.wikipedia.org/wiki/Integration_testing)和系统测试层面的快速自动化。验收测试是您运行来确定是否满足给定项目的[规范](https://en.wikipedia.org/wiki/Specification)的那些测试，而集成测试旨在保证项目的不同组件作为一个组正确工作。

您的`doctest`测试可以存在于您项目的文档和您代码的文档字符串中。例如，包含`doctest`测试的**包级 docstring** 是进行集成测试的一种非常好的快速方法。在这个级别，您可以测试整个包及其模块、类、函数等的集成。

一组高级的`doctest`测试是预先定义程序规格的一个很好的方法。同时，低级别的[单元测试](https://en.wikipedia.org/wiki/Unit_testing)让你设计你的程序的单个构建模块。然后，只要你愿意，就可以让你的计算机随时对照测试来检查代码。

在类、方法和函数级别，测试是在你编写代码时测试你的代码的强大工具。您可以在编写代码本身的同时，逐渐将测试用例添加到您的文档字符串中。这种实践将允许你生成更加可靠和健壮的代码，特别是如果你坚持测试驱动开发的原则的话。

总之，您可以将`doctest`用于以下目的:

*   编写**快速有效的测试用例**在编写代码时检查代码
*   在你的项目、包和模块上运行**验收**、**回归**和**集成**测试用例
*   检查你的**文档串**是否是最新的**并且与目标代码**同步****
***   验证您的项目的**文档**是否是最新的***   为您的项目、包和模块编写**实践教程***   说明如何**使用项目的 API**以及预期的输入和输出必须是什么***

***在您的文档和文档字符串中进行`doctest`测试是您的客户或团队成员在评估代码的特性、规范和质量时运行这些测试的一种极好的方式。

## 用 Python 编写自己的`doctest`测试

现在您已经知道了什么是`doctest`以及您可以用它来做什么，您将学习如何使用`doctest`来测试您的代码。不需要特殊的设置，因为`doctest`是 Python 标准库的一部分。

在接下来的小节中，您将学习如何检查函数、方法和其他可调用函数的[返回值](https://realpython.com/python-return-statement/)。类似地，您将理解如何检查给定代码的打印输出。

您还将学习如何为必须引发[异常](https://realpython.com/python-exceptions/)的代码创建测试用例，以及如何在执行测试用例之前运行准备步骤。最后，您将回顾一下关于`doctest`测试语法的一些细节。

[*Remove ads*](/account/join/)

### 创建用于检查返回和打印值的`doctest`测试

代码测试的第一个也可能是最常见的用例是检查函数、方法和其他可调用函数的**返回值**。您可以通过`doctest`测试来做到这一点。例如，假设您有一个名为`add()`的函数，它将两个[数](https://realpython.com/python-numbers/)作为参数，并返回它们的算术和:

```py
# calculations.py

def add(a, b):
    return float(a + b)
```

这个函数将两个数相加。记录您的代码是一个很好的实践，所以您可以向这个函数添加一个 docstring。您的 docstring 可能如下所示:

```py
# calculations.py

def add(a, b):
    """Compute and return the sum of two numbers.

 Usage examples:
 >>> add(4.0, 2.0) 6.0
 >>> add(4, 2) 6.0
 """
    return float(a + b)
```

这个 docstring 包括两个如何使用`add()`的例子。每个示例都包含一个初始行，以 Python 的主要交互提示`>>>`开始。这一行包括一个带有两个数字参数的对`add()`的调用。然后，该示例的第二行包含了**预期输出**，它与函数的预期返回值相匹配。

在这两个例子中，预期的输出是一个[浮点数](https://realpython.com/python-numbers/#floating-point-numbers)，这是必需的，因为函数总是返回这种类型的数字。

您可以使用`doctest`运行这些测试。继续运行以下命令:

```py
$ python -m doctest calculations.py
```

该命令不会向您的屏幕发出任何输出。显示没有输出意味着`doctest`运行了所有的测试用例，没有发现任何失败的测试。

如果您想让`doctest`详细描述运行测试的过程，那么使用`-v`开关:

```py
$ python -m doctest -v calculations.py
Trying:
 add(4.0, 2.0) Expecting:
 6.0 ok
Trying:
 add(4, 2) Expecting:
 6.0 ok
1 items had no tests:
 calculations
1 items passed all tests:
 2 tests in calculations.add
2 tests in 2 items.
2 passed and 0 failed. Test passed.
```

使用`-v`选项运行`doctest`会产生描述测试运行过程的详细输出。前两行突出显示了实际的测试及其相应的预期输出。每个测试的预期输出之后的一行显示单词`ok`，意味着目标测试成功通过。在本例中，两个测试都通过了，正如您可以在最后一个突出显示的行中确认的那样。

`doctest`测试的另一个常见用例是检查给定代码的**打印输出**。继续创建一个名为`printed_output.py`的新文件，并向其中添加以下代码:

```py
# printed_output.py

def greet(name="World"):
    """Print a greeting to the screen.

 Usage examples:
 >>> greet("Pythonista")
 Hello, Pythonista!
 >>> greet()
 Hello, World!
 """
    print(f"Hello, {name}!")
```

这个函数将一个名字作为参数，[将问候打印到屏幕上](https://realpython.com/python-print/)。您可以像往常一样从命令行使用`doctest`在这个函数的 docstring 中运行测试:

```py
$ python -m doctest -v printed_output.py
Trying:
 greet("Pythonista") Expecting:
 Hello, Pythonista! ok
Trying:
 greet() Expecting:
 Hello, World! ok
1 items had no tests:
 printed_output
1 items passed all tests:
 2 tests in printed_output.greet
2 tests in 2 items.
2 passed and 0 failed.
Test passed.
```

这些测试按预期工作，因为 Python REPL 在屏幕上显示返回和打印的值。这种行为允许`doctest`匹配测试用例中返回和打印的值。

使用`doctest`测试一个函数在屏幕上显示的内容非常简单。对于其他测试框架，进行这种测试可能会稍微复杂一些。您将需要处理[标准输出流](https://realpython.com/python-subprocess/#the-standard-io-streams)，这可能需要高级 Python 知识。

### 了解`doctest`如何匹配预期和实际测试输出

在实践中，`doctest`在匹配预期输出和实际结果时非常严格。例如，使用整数而不是浮点数会破坏`add()`函数的测试用例。

其他微小的细节，如使用空格或制表符，用双引号将返回的[字符串](https://realpython.com/python-strings/)括起来，或者插入空行，也会导致测试中断。考虑以下玩具测试案例作为上述问题的示例:

```py
# failing_tests.py

"""Sample failing tests.

The output must be an integer
>>> 5 + 7
12.0

The output must not contain quotes
>>> print("Hello, World!")
'Hello, World!'

The output must not use double quotes
>>> "Hello," + " World!"
"Hello, World!"

The output must not contain leading or trailing spaces
>>> print("Hello, World!")
 Hello, World!

The output must not be a blank line
>>> print()

"""
```

当您从命令行使用`doctest`运行这些测试时，您会得到冗长的输出。这里有一个细目分类:

```py
$ python -m doctest -v failing_tests.py
Trying:
 5 + 7
Expecting:
 12.0
**********************************************************************
File ".../failing_tests.py", line 6, in broken_tests
Failed example:
 5 + 7 Expected:
 12.0 Got:
 12
```

在第一段输出中，预期的结果是一个浮点数。但是，`5 + 7`会返回一个整数值`12`。因此，`doctest`将测试标记为失败。`Expected:`和`Got:`标题给你关于检测到的问题的提示。

下一条输出如下所示:

```py
Trying:
 print("Hello, World!")
Expecting:
 'Hello, World!'
**********************************************************************
File ".../failing_tests.py", line 10, in broken_tests
Failed example:
 print("Hello, World!")
Expected:
 'Hello, World!' Got:
 Hello, World!
```

在本例中，预期的输出使用单引号。然而，`print()`函数在输出时没有加引号，导致测试失败。

该命令的输出继续如下:

```py
Trying:
 "Hello," + "World!"
Expecting:
 "Hello, World!"
**********************************************************************
File ".../failing_tests.py", line 14, in broken_tests
Failed example:
 "Hello," + " World!"
Expected:
 "Hello, World!" Got:
 'Hello, World!'
```

这段输出显示了另一个失败的测试。在这个例子中，问题是 Python 在交互式部分显示字符串时使用单引号而不是双引号。同样，这种微小的差异会使您的测试失败。

接下来，您将获得以下输出:

```py
Trying:
 print("Hello, World!")
Expecting:
 Hello, World!
**********************************************************************
File ".../failing_tests.py", line 18, in broken_tests
Failed example:
 print("Hello, World!")
Expected:
 Hello, World! Got:
 Hello, World!
```

在本例中，测试失败是因为预期的输出包含前导空格。然而，实际输出没有前导空格。

最终的输出如下所示:

```py
Trying:
 print()
Expecting nothing
**********************************************************************
File ".../failing_tests.py", line 22, in broken_tests
Failed example:
 print()
Expected nothing Got:
 <BLANKLINE> **********************************************************************
1 items had failures:
 5 of   5 in broken_tests
5 tests in 1 items.
0 passed and 5 failed. ***Test Failed*** 5 failures.
```

在常规的 REPL 会话中，不带参数调用`print()`会显示一个空行。在`doctest`测试中，空行意味着您刚刚执行的代码没有发出任何输出。这就是为什么`doctest`的输出说什么都没预料到，却得到了`<BLANKLINE>`。在关于`doctest`的[限制](#exploring-some-limitations-of-doctest)一节中，您将了解到更多关于这个`<BLANKLINE>`占位符标签的信息。

**注意:**空行包括仅包含被[视为空白](https://docs.python.org/3/library/string.html#string.whitespace)的字符的行，例如空格和制表符。在名为[处理空白字符和其他字符](#dealing-with-whitespaces-and-other-characters)的章节中，你会学到更多关于空白字符的知识。

总而言之，您必须保证实际测试输出和预期输出之间的完美匹配。因此，确保每个测试用例之后的那一行与您需要代码返回或打印的内容完全匹配。

[*Remove ads*](/account/join/)

### 编写`doctest`测试来捕捉异常

除了测试成功的返回值之外，您通常还需要测试那些在响应错误或其他问题时会引发异常的代码。

在捕捉返回值和异常时，`doctest`模块遵循几乎相同的规则。它搜索看起来像 Python 异常报告或[回溯](https://realpython.com/python-traceback/)的文本，并检查代码引发的任何异常。

例如，假设您已经将下面的`divide()`函数添加到您的`calculations.py`文件中:

```py
# calculations.py
# ...

def divide(a, b):
    return float(a / b)
```

该函数将两个数字作为参数，并将它们的商作为浮点数返回。当`b`的值不是`0`时，该函数按预期工作，但是它为`b == 0`引发了一个异常:

>>>

```py
>>> from calculations import divide

>>> divide(84, 2)
42.0

>>> divide(15, 3)
5.0

>>> divide(42, -2)
-21.0

>>> divide(42, 0)
Traceback (most recent call last):
    ...
ZeroDivisionError: division by zero
```

前三个例子表明，当除数`b`不同于`0`时，`divide()`工作良好。然而，当`b`为`0`时，该功能以`ZeroDivisionError`中断。这个异常表明该操作不被允许。

如何使用`doctest`测试来测试这个异常呢？查看以下代码中的 docstring，尤其是最后一个测试用例:

```py
# calculations.py
# ...

def divide(a, b):
    """Compute and return the quotient of two numbers.

 Usage examples:
 >>> divide(84, 2)
 42.0
 >>> divide(15, 3)
 5.0
 >>> divide(42, -2)
 -21.0

 >>> divide(42, 0)
 Traceback (most recent call last): ZeroDivisionError: division by zero """
    return float(a / b)
```

前三个测试工作正常。所以，把注意力集中在最后一个测试上，尤其是高亮显示的行。第一个突出显示的行包含一个所有异常回溯通用的头。第二个突出显示的行包含实际的异常及其特定消息。这两行是`doctest`成功检查预期异常的唯一要求。

在处理异常回溯时，`doctest`完全忽略回溯体，因为它可能会发生意外变化。实际上，`doctest`只关注第一行，即`Traceback (most recent call last):`，以及最后一行。正如您已经知道的，第一行是所有异常回溯所共有的，而最后一行显示了关于引发的异常的信息。

因为`doctest` *完全*忽略了回溯体，所以你可以在 docstrings 中对它做任何你想做的事情。通常情况下，只有当回溯体对文档有重要价值时，才需要包含它。根据您的选择，您可以:

1.  完全移除追溯体
2.  用省略号(`...`)替换回溯正文的部分内容
3.  用省略号完全替换回溯正文
4.  用任何自定义文本或解释替换追溯正文
5.  包括完整的回溯主体

无论如何，回溯体只对阅读你的文档的人有意义。这个列表中的第二个、第四个和最后一个选项只有在回溯增加了代码文档的价值时才有用。

如果在最后一个测试用例中包含完整的回溯，下面是`divide()`的 docstring 的样子:

```py
# calculations.py
# ...

def divide(a, b):
    """Compute and return the quotient of two numbers.

 Usage examples:
 >>> divide(84, 2)
 42.0
 >>> divide(15, 3)
 5.0
 >>> divide(42, -2)
 -21.0

 >>> divide(42, 0)
 Traceback (most recent call last): File "<stdin>", line 1, in <module> divide(42, 0) File "<stdin>", line 2, in divide return float(a / b) ZeroDivisionError: division by zero """
    return float(a / b)
```

回溯正文显示了导致异常的文件和行的信息。它还显示了直到失败代码行的整个堆栈跟踪。有时，这些信息在记录代码时会很有用。

在上面的例子中，注意如果你包含了完整的回溯体，那么你必须保持体的原始缩进。否则，测试将失败。现在继续在命令行上用`doctest`运行您的测试。记得使用`-v`开关来获得详细的输出。

### 构建更精细的测试

通常，您需要测试依赖于代码中其他对象的功能。例如，您可能需要测试给定类的方法。为此，您需要首先实例化该类。

`doctest`模块能够运行创建和导入对象、调用函数、分配变量、计算表达式等代码。在运行实际的测试用例之前，您可以利用这种能力来执行各种准备步骤。

例如，假设您正在编写一个[队列](https://realpython.com/queue-in-python/)数据结构，并决定使用来自 [`collections`](https://realpython.com/python-collections-module/) 模块的 [`deque`](https://realpython.com/python-deque/) 数据类型来有效地实现它。经过几分钟的编码，您最终得到了以下代码:

```py
# queue.py

from collections import deque

class Queue:
    def __init__(self):
        self._elements = deque()

    def enqueue(self, element):
        self._elements.append(element)

    def dequeue(self):
        return self._elements.popleft()

    def __repr__(self):
        return f"{type(self).__name__}({list(self._elements)})"
```

您的`Queue`类只实现了两个基本的队列操作，**入队**和**出列**。入队允许您将项目或元素添加到队列的末尾，而出队允许您从队列的开头移除和返回项目。

`Queue`还实现了一个 [`.__repr__()`](https://docs.python.org/3/reference/datamodel.html#object.__repr__) 方法，该方法提供了类的**字符串表示**。这个方法将在编写和运行您的`doctest`测试中扮演重要的角色，稍后您将对此进行探索。

现在假设你想编写`doctest`测试来保证`.enqueue()`和`.dequeue()`方法工作正常。为此，首先需要创建一个`Queue`的实例，并用一些样本数据填充它:

```py
 1# queue.py
 2
 3from collections import deque
 4
 5class Queue:
 6    def __init__(self):
 7        self._elements = deque()
 8
 9    def enqueue(self, element):
10        """Add items to the right end of the queue.
11
12 >>> numbers = Queue()
13 >>> numbers
14 Queue([])
15
16 >>> for number in range(1, 4):
17 ...     numbers.enqueue(number)
18
19 >>> numbers
20 Queue([1, 2, 3])
21 """
22        self._elements.append(element)
23
24    def dequeue(self):
25        """Remove and return an item from the left end of the queue.
26
27 >>> numbers = Queue()
28 >>> for number in range(1, 4):
29 ...     numbers.enqueue(number)
30 >>> numbers
31 Queue([1, 2, 3])
32
33 >>> numbers.dequeue()
34 1
35 >>> numbers.dequeue()
36 2
37 >>> numbers.dequeue()
38 3
39 >>> numbers
40 Queue([])
41 """
42        return self._elements.popleft()
43
44    def __repr__(self):
45        return f"{type(self).__name__}({list(self._elements)})"
```

在`enqueue()`的 docstring 中，首先运行一些*设置*步骤。第 12 行创建了一个`Queue`的实例，而第 13 行和第 14 行检查该实例是否已经成功创建并且当前为空。注意您是如何使用定制的字符串表示`Queue`来表达这个准备步骤的输出的。

第 16 行和第 17 行运行一个 [`for`循环](https://realpython.com/python-for-loop/)，该循环使用`.enqueue()`用一些样本数据填充`Queue`实例。在这种情况下，`.enqueue()`不返回任何东西，所以不必检查任何返回值。最后，第 19 行和第 20 行运行实际测试，确认`Queue`实例现在包含了预期顺序的样本数据。

在`.dequeue()`中，第 27 到 31 行创建了一个新的`Queue`实例，用一些样本数据填充它，并检查数据是否被成功添加。同样，这些是在测试`.dequeue()`方法本身之前需要运行的设置步骤。

真正的测试出现在第 33 到 41 行。在这些行中，您调用了三次`.dequeue()`。每个调用都有自己的输出线路。最后，第 39 行和第 40 行验证调用`.dequeue()`的结果是`Queue`的实例完全为空。

在上面的例子中需要强调的一点是，`doctest`在一个专用的上下文或[范围](https://realpython.com/python-scope-legb-rule/)中运行单独的文档字符串。因此，在一个 docstring 中声明的名称不能在另一个 docstring 中使用。因此，`.enqueue()`中定义的`numbers`对象在`.dequeue()`中是不可访问的。在测试后一种方法之前，您需要在`.dequeue()`中创建一个新的`Queue`实例。

在[理解`doctest`作用域机制](#understanding-the-doctest-scoping-mechanism)一节中，您将深入了解`doctest`如何管理您的测试用例的执行范围。

[*Remove ads*](/account/join/)

### 处理空白和其他字符

关于空格和反斜杠等字符，规则有点复杂。预期的输出不能由空行或仅包含空白字符的行组成。这样的行被解释为预期输出的结尾。

如果您的预期输出包含空行，那么您必须使用`<BLANKLINE>`占位符标记来替换它们:

```py
# greet.py

def greet(name="World"):
    """Print a greeting.

 Usage examples:
 >>> greet("Pythonista")
 Hello, Pythonista!
 <BLANKLINE> How have you been?
 """
    print(f"Hello, {name}!")
    print()
    print("How have you been?")
```

`greet()`的预期输出包含一个空行。为了让您的`doctest`测试通过，您必须在每个预期的空白行上使用`<BLANKLINE>`标签，就像您在上面突出显示的行中所做的那样。

当制表符出现在测试输出中时，匹配起来也很复杂。预期输出中的制表符会自动转换为空格。相比之下，实际输出中的制表符不会被修改。

这种行为会使您的测试失败，因为预期的和实际的输出不匹配。如果您的代码输出包含制表符，那么您可以使用 [`NORMALIZE_WHITESPACE`](https://docs.python.org/3/library/doctest.html#doctest.NORMALIZE_WHITESPACE) 选项或[指令](https://docs.python.org/3/library/doctest.html#doctest-directives)使`doctest`测试通过。关于如何处理输出中的制表符的例子，请查看您的`doctest`测试部分中的[嵌入指令。](#embedding-directives-in-your-doctest-tests)

在你的`doctest`测试中，反斜杠也需要特别注意。由于[显式行连接](https://docs.python.org/3/reference/lexical_analysis.html#explicit-line-joining)或其他原因而使用反斜杠的测试必须使用**原始字符串**，也称为 **r 字符串**，它将准确地保存您键入的反斜杠:

```py
# greet.py

def greet(name="World"):
    r"""Print a greeting.

 Usage examples:
 >>> greet("Pythonista")
 /== Hello, Pythonista! ==\ \== How have you been? ==/ """
    print(f"/== Hello, {name}! ==\\")
    print("\\== How have you been? ==/")
```

在这个例子中，您使用一个原始字符串来编写这个新版本的`greet()`的 docstring。注意 docstring 中的前导`r`。请注意，在实际代码中，您将反斜杠(`\\`)加倍以对其进行转义，但是在 docstring 中您不需要将它加倍。

如果您不想使用原始字符串作为转义反斜杠的方式，那么您可以使用通过将反斜杠加倍来转义反斜杠的常规字符串。按照这个建议，您也可以编写上面的测试用例，如下例所示:

```py
# greet.py

def greet(name="World"):
    """Print a greeting.

 Usage examples:
 >>> greet("Pythonista")
 /== Hello, Pythonista! ==\\ \\== How have you been? ==/ """
    print(f"/== Hello, {name}! ==\\")
    print("\\== How have you been? ==/")
```

在这个新版本的测试用例中，您在您的`doctest`测试的预期输出中使用了双倍的反斜杠字符来对它们进行转义。

### 总结`doctest`测试语法

正如您已经知道的，`doctest`通过寻找模仿 Python 交互式会话的文本片段来识别测试。根据这个规则，以`>>>`提示符开始的行被解释为[简单语句](https://docs.python.org/3/reference/simple_stmts.html#simple-statements)、[复合语句](https://docs.python.org/3/reference/compound_stmts.html#compound-statements)头，或者表达式。类似地，以`...`提示符开始的行在复合语句中被解释为延续行。

任何不以`>>>`或`...`开头的行，直到下一个`>>>`提示符或空白行，都代表您期望的代码输出。输出必须像在 Python 交互式会话中一样，包括返回值和打印输出。空白行和`>>>`提示用作测试分隔符或终止符。

如果在以`>>>`或`...`开头的行之间没有任何输出行，那么`doctest`假设该语句预期没有输出，当您调用返回 [`None`](https://realpython.com/null-in-python/) 的函数或有赋值语句时就是这种情况。

`doctest`模块忽略任何不符合`doctest`测试语法的东西。这种行为允许您在测试之间包含解释性文本、图表或任何您需要的东西。您可以在下面的示例中利用这一特性:

```py
# calculations.py
# ...

def divide(a, b):
    """Compute and return the quotient of two numbers.

 Usage examples:
 >>> divide(84, 2)
 42.0
 >>> divide(15, 3)
 5.0
 >>> divide(42, -2)
 -21.0

 The test below checks if the function catches zero divisions: >>> divide(42, 0)
 Traceback (most recent call last):
 ZeroDivisionError: division by zero
 """
    return float(a / b)
```

在这次对`divide()`的更新中，您在最终测试的上方添加了解释文本。注意，如果解释文本在两个测试之间，那么在解释本身之前需要一个空行。这个空行将告诉`doctest`先前测试的输出已经完成。

下面是对`doctest`测试语法的总结:

*   测试在 **`>>>`提示**之后开始，并继续 **`...`提示**，就像在 Python 交互会话中一样。
*   **预期输出**必须在测试后立即占用生产线。
*   发送到**标准输出流**的输出被捕获。
*   发送到**标准错误流**的输出没有被捕获。
*   只要预期的输出处于相同的缩进水平，测试开始的**列**并不重要。

标准输入流和 T2 输出流的概念超出了本教程的范围。要深入了解这些概念，请查看[子进程模块:用 Python 包装程序](https://realpython.com/python-subprocess/)中的[标准 I/O 流](https://realpython.com/python-subprocess/#the-standard-io-streams)部分。

[*Remove ads*](/account/join/)

### 了解失败测试的输出

到目前为止，你已经成功地运行了大多数`doctest`测试。然而，在现实世界中，在让代码工作之前，您可能会面临许多失败的测试。在本节中，您将学习如何解释和理解失败的`doctest`测试产生的输出。

当测试失败时，`doctest`显示失败的测试和失败的原因。您将在测试报告的末尾有一行总结成功和失败的测试。例如，考虑下面的例子，在您的原始`failing_tests.py`文件的稍微修改版本中测试失败:

```py
# failing_tests.py

"""Sample failing tests.

The output must be an integer
>>> 5 + 7
12.0

The output must not contain quotes
>>> print("Hello, World!")
'Hello, World!'

The output must not use double quotes
>>> "Hello," + "World!"
"Hello, World!"

The output must not contain leading or trailing spaces
>>> print("Hello, World!")
 Hello, World!

The traceback doesn't include the correct exception message
>>> raise ValueError("incorrect value")
Traceback (most recent call last):
ValueError: invalid value
"""
```

该文件包含一系列失败的测试。测试失败的原因各不相同。每次测试前的注释都强调了失败的根本原因。如果您从命令行使用`doctest`运行这些测试，那么您将得到冗长的输出。为了更好地理解输出，您可以将其分成小块:

```py
$ python -m doctest failing_tests.py
**********************************************************************
File "failing_tests.py", line 2, in failing_tests.py
Failed example:
 5 + 7 Expected:
 12.0 Got:
 12
```

在第一段输出中，测试失败了，因为您使用了一个浮点数作为预期的输出。然而，实际输出是一个整数。您可以通过检查紧接在 **`Failed example:`** 标题之后的一行来快速发现失败的测试。

同样，您可以通过检查 **`Expected:`** 标题下的行找到预期输出，实际输出显示在 **`Got:`** 标题下。通过比较预期输出和实际输出，您可能会找到失败的原因。

所有失败的测试都有相似的输出格式。您会发现`Expected:`和`Got:`标题会引导您找到导致测试失败的问题:

```py
**********************************************************************
File "failing_tests.py", line 6, in failing_tests.py
Failed example:
 print("Hello, World!")
Expected:
 'Hello, World!' Got:
 Hello, World! **********************************************************************
File "failing_tests.py", line 10, in failing_tests.py
Failed example:
 "Hello," + " World!"
Expected:
 "Hello, World!" Got:
 'Hello, World!' **********************************************************************
File "failing_tests.py", line 14, in failing_tests.py
Failed example:
 print("Hello, World!")
Expected:
 Hello, World! Got:
 Hello, World!
```

预期输出和实际输出之间的差异可能非常微妙，比如没有引号，使用双引号而不是单引号，甚至意外地插入前导或尾随空格。

当测试检查引发的异常时，由于异常回溯，输出可能会变得混乱。然而，仔细的检查通常会引导您找到故障的原因:

```py
**********************************************************************
File "failing_tests.py", line 18, in failing_tests.py
Failed example:
 raise ValueError("incorrect value")
Expected:
 Traceback (most recent call last):
 ValueError: invalid value Got:
 Traceback (most recent call last):
 ...
 raise ValueError("incorrect value")
 ValueError: incorrect value
```

在本例中，预期异常显示的消息与实际异常中的消息略有不同。当您更新了代码但忘记更新相应的`doctest`测试时，可能会发生类似的事情。

输出的最后部分显示了失败测试的摘要:

```py
**********************************************************************
1 items had failures:
 5 of   5 in broken_tests.txt
***Test Failed*** 5 failures.
```

在这个例子中，所有五个测试都失败了，您可以从阅读最后一行得出结论。最后一行有如下的一般格式:`***Test Failed*** N failures.`这里，`N`表示代码中失败测试的数量。

## 在您的项目中提供`doctest`测试

使用`doctest`，您可以从您的文档、您的专用测试文件以及您的代码文件中的文档字符串中执行测试用例。

在本节中，您将使用一个名为`calculations.py`的模块作为示例项目。然后您将学习如何使用`doctest`来运行这个小项目的以下部分的测试:

*   `README.md`文件
*   专用测试文件
*   文档字符串

在下面的可折叠部分，您将找到`calculations.py`文件的完整源代码:



```py
# calculations.py

"""Provide several sample math calculations.

This module allows the user to make mathematical calculations.

Module-level tests:
>>> add(2, 4)
6.0
>>> subtract(5, 3)
2.0
>>> multiply(2.0, 4.0)
8.0
>>> divide(4.0, 2)
2.0
"""

def add(a, b):
    """Compute and return the sum of two numbers.

 Tests for add():
 >>> add(4.0, 2.0)
 6.0
 >>> add(4, 2)
 6.0
 """
    return float(a + b)

def subtract(a, b):
    """Calculate the difference of two numbers.

 Tests for subtract():
 >>> subtract(4.0, 2.0)
 2.0
 >>> subtract(4, 2)
 2.0
 """
    return float(a - b)

def multiply(a, b):
    """Compute and return the product of two numbers.

 Tests for multiply():
 >>> multiply(4.0, 2.0)
 8.0
 >>> multiply(4, 2)
 8.0
 """
    return float(a * b)

def divide(a, b):
    """Compute and return the quotient of two numbers.

 Tests for divide():
 >>> divide(4.0, 2.0)
 2.0
 >>> divide(4, 2)
 2.0
 >>> divide(4, 0)
 Traceback (most recent call last):
 ZeroDivisionError: division by zero
 """
    return float(a / b)
```

将上述代码保存在一个名为`calculations.py`的文件中。将这个文件放在一个有正确名称的目录中。

[*Remove ads*](/account/join/)

### 在您的项目文档中包含`doctest`测试

为了开始这个小项目的`doctest`测试，您将从在包含`calculations.py`的同一个目录中创建`README.md`开始。这个`README.md`文件将使用 Markdown 语言为您的`calculations.py`文件提供最少的文档:

```py
<!-- README.md -->

# Functions to Perform Arithmetic Calculations

The `calculations.py` Python module provides basic arithmetic
operations, including addition, subtraction, multiplication, and division.

Here are a few examples of how to use the functions in `calculations.py`:

```
python
>>> import calculations

>>> calculations.add(2, 2)
4.0

>>> calculations.subtract(2, 2)
0.0

>>> calculations.multiply(2, 2)
4.0

>>> calculations.divide(2, 2)
1.0

```py

These examples show how to use the `calculations.py` module in your code.
```

这个 Markdown 文件包含了对您的`calculations.py`文件的最小描述和一些用 Python 代码块包装的使用示例。注意，第一行代码导入了模块本身。

另一个重要的细节是，您在最终测试之后和结束三个反勾号(`"\x60\x60\x60"`)之前包含了一个空行。您需要这个空行来表示您的`doctest`测试已经完成，否则三个反勾号将被视为预期的输出。

您可以像往常一样使用`doctest`模块运行上述 Markdown 文件中的测试:

```py
$ python -m doctest -v README.md
Trying:
 import calculations
Expecting nothing
ok
Trying:
 calculations.add(2, 2)
Expecting:
 4.0
ok
Trying:
 calculations.subtract(2, 2)
Expecting:
 0.0
ok
Trying:
 calculations.multiply(2, 2)
Expecting:
 4.0
ok
Trying:
 calculations.divide(2, 2)
Expecting:
 1.0
ok
1 items passed all tests:
 5 tests in README.md
5 tests in 1 items.
5 passed and 0 failed.
Test passed.
```

正如您可以从上面的输出中确认的，您的`README.md`文件中的所有`doctest`测试都成功运行了，并且都通过了。

### 向您的项目添加专用测试文件

在项目中提供`doctest`测试的另一种方式是使用一个专用的测试文件。为此，您可以使用纯文本文件。例如，您可以使用包含以下内容的名为`test_calculations.txt`的文件:

```py
>>> import calculations

>>> calculations.add(2, 2)
4.0

>>> calculations.subtract(2, 2)
0.0

>>> calculations.multiply(2, 2)
4.0

>>> calculations.divide(2, 2)
1.0
```

这个 TXT 文件是一个带有一些`doctest`测试的专用测试文件。同样，您可以从命令行使用`doctest`运行这些样本测试用例:

```py
$ python -m doctest -v test_calculations.txt
Trying:
 import calculations
Expecting nothing
ok
Trying:
 calculations.add(2, 2)
Expecting:
 4.0
ok
Trying:
 calculations.subtract(2, 2)
Expecting:
 0.0
ok
Trying:
 calculations.multiply(2, 2)
Expecting:
 4.0
ok
Trying:
 calculations.divide(2, 2)
Expecting:
 1.0
ok
1 items passed all tests:
 5 tests in test_calculations.txt
5 tests in 1 items.
5 passed and 0 failed.
Test passed.
```

您所有的`doctest`测试都成功运行并通过。如果您不想让过多的`doctest`测试使您的文档变得杂乱，那么您可以使用`doctest`运行的专用测试文件是一个不错的选择。

### 在代码的文档字符串中嵌入`doctest`测试

最后，也可能是最常见的，提供`doctest`测试的方法是通过项目的文档字符串。使用文档字符串，您可以进行不同级别的测试:

*   包裹
*   组件
*   类和方法
*   功能

您可以在您的包的 [`__init__.py`](https://realpython.com/python-modules-packages/#package-initialization) 文件的 docstring 中编写包级`doctest`测试。其他测试将存在于它们各自容器对象的文档字符串中。例如，您的`calculations.py`文件有一个包含`doctest`测试的模块级 docstring:

```py
# calculations.py

"""Provide several sample math calculations.

This module allows the user to make mathematical calculations.

Module-level tests:
>>> add(2, 4)
6.0
>>> subtract(5, 3)
2.0
>>> multiply(2.0, 4.0)
8.0
>>> divide(4.0, 2)
2.0
"""

# ...
```

同样，在所有您在`calculations.py`中定义的函数中，您有包含`doctest`测试的函数级文档字符串。看看他们！

如果您返回到定义了`Queue`类的`queue.py`文件，那么您可以添加类级别的`doctest`测试，如下面的代码片段所示:

```py
# queue.py

from collections import deque

class Queue:
    """Implement a Queue data type.

 >>> Queue()
 Queue([])

 >>> numbers = Queue()
 >>> numbers
 Queue([])

 >>> for number in range(1, 4):
 ...     numbers.enqueue(number)
 >>> numbers
 Queue([1, 2, 3])
 """

    # ...
```

上述 docstring 中的`doctest`测试检查了`Queue`类是如何工作的。这个例子只添加了对`.enqueue()`方法的测试。您能为`.dequeue()`方法添加测试吗？那将是很好的锻炼！

您可以从命令行运行项目的 docstrings 中的所有`doctest`测试，就像您到目前为止所做的那样。但是在接下来的部分中，您将更深入地研究运行您的`doctest`测试的不同方法。

[*Remove ads*](/account/join/)

## 了解`doctest`范围界定机制

`doctest`的一个重要方面是它在专用的上下文或范围中运行单个的文档字符串。当你对一个给定的模块运行`doctest`时，`doctest`会创建该模块的[全局作用域](https://realpython.com/python-scope-legb-rule/#modules-the-global-scope)的一个浅层副本。然后`doctest`创建一个[局部作用域](https://realpython.com/python-scope-legb-rule/#functions-the-local-scope),其中定义了将首先执行的文档字符串中的变量。

一旦测试运行，`doctest`清理它的本地范围，丢弃任何本地名字。因此，在一个 docstring 中声明的本地名称不能在下一个 docstring 中使用。每个 docstring 都将在一个定制的局部作用域中运行，但是`doctest`全局作用域对于模块中的所有 docstring 都是通用的。

考虑下面的例子:

```py
# context.py

total = 100

def decrement_by(number):
    """Decrement the global total variable by a given number.

 >>> local_total = decrement_by(50)
 >>> local_total
 50

 Changes to total don't affect the code's global scope
 >>> total
 100
 """
    global total
    total -= number
    return total

def increment_by(number):
    """Increment the global total variable by a given number.

 The initial value of total's shallow copy is 50
 >>> increment_by(10)
 60

 The local_total variable is not defined in this test
 >>> local_total
 Traceback (most recent call last):
 NameError: name 'local_total' is not defined
 """
    global total
    total += number
    return total
```

如果用`doctest`运行这个文件，那么所有的测试都会通过。在`decrement_by()`中，第一个测试定义了一个局部变量`local_total`，它以值`50`结束。这个值是从`total`的全局浅拷贝中减去`number`的结果。第二个测试显示`total`保持了它的初始值`100`，确认了`doctest`测试不影响代码的全局范围，只影响它的浅层拷贝。

通过创建模块全局范围的浅层副本，`doctest`确保运行测试不会改变实际模块的全局范围。然而，对你的全局作用域的浅层副本中的变量的改变会传播到其他的`doctest`测试。这就是为什么`increment_by()`中的第一个测试返回`60`而不是`110`。

`increment_by()`中的第二个测试确认在测试运行后局部范围被清理。因此，在 docstring 中定义的局部变量对其他 docstring 不可用。清理局部范围可以防止测试间的依赖，这样给定测试用例的痕迹就不会导致其他测试用例通过或失败。

当您使用一个专用的测试文件来提供`doctest`测试时，来自这个文件的所有测试都在相同的执行范围内运行。这样，给定测试的执行会影响后面测试的结果。这种行为不是有益的。测试需要相互独立。否则，知道哪个测试失败了并不能给你明确的线索，告诉你代码中哪里出了问题。

在这种情况下，您可以通过将每个测试放在它自己的文件中，为每个测试提供它们自己的执行范围。这种实践将解决范围问题，但是会给测试运行任务增加额外的工作量。

赋予每个测试自己的执行范围的另一种方法是在函数中定义每个测试，如下所示:

```py
>>> def test_add():
...     import calculations
...     return calculations.add(2, 4)
>>> test_add()
6.0
```

在这个例子中，共享作用域中唯一的对象是`test_add()`函数。`calculations`模块将不可用。

`doctest`作用域机制主要是为了保证您的`doctest`测试的安全和独立执行。

## `doctest` 的一些局限性探究

与其他测试框架相比，`doctest`最大的限制可能是缺少与 pytest 中的[夹具](https://docs.pytest.org/en/6.2.x/fixture.html#fixture)或 [`unittest`](https://realpython.com/python-testing/#unittest) 中的[设置](https://docs.python.org/3/library/unittest.html#unittest.TestCase.setUp)和[拆卸](https://docs.python.org/3/library/unittest.html#unittest.TestCase.tearDown)机制相当的功能。如果您需要安装和拆卸代码，那么您必须在每个受影响的 docstring 中编写它。或者，您可以使用 [`unittest` API](https://docs.python.org/3/library/doctest.html?highlight=doctest#unittest-api) ，它提供了一些设置和拆卸选项。

`doctest`的另一个限制是，它严格地将测试的预期输出与测试的实际输出进行比较。`doctest`模块需要精确匹配。如果只有一个字符不匹配，那么测试失败。这种行为使得正确测试一些 Python 对象变得困难。

作为这种严格匹配的一个例子，假设您正在测试一个返回集合的函数。在 Python 中，集合不会以任何特定的顺序存储它们的元素，因此由于元素的随机顺序，您的测试在大多数情况下都会失败。

考虑下面这个实现了一个`User`类的例子:

```py
# user.py

class User:
    def __init__(self, name, favorite_colors):
        self.name = name
        self._favorite_colors = set(favorite_colors)

    @property
    def favorite_colors(self):
        """Return the user's favorite colors.

 Usage examples:
 >>> john = User("John", {"#797EF6", "#4ADEDE", "#1AA7EC"})
 >>> john.favorite_colors
 {'#797EF6', '#4ADEDE', '#1AA7EC'}
 """
        return self._favorite_colors
```

这个`User`类带`name`和一系列喜欢的颜色。类初始化器将输入的颜色转换成一个`set`对象。`favorite_colors()` [属性](https://realpython.com/python-property/)返回用户喜欢的颜色。因为集合以随机的顺序存储它们的元素，你的`doctest`测试在大多数时候都会失败:

```py
$ python -m doctest -v user.py
Trying:
 john = User("John", {"#797EF6", "#4ADEDE", "#1AA7EC"})
Expecting nothing
ok
Trying:
 john.favorite_colors
Expecting:
 {'#797EF6', '#4ADEDE', '#1AA7EC'}
**********************************************************************
File ".../user.py", line ?, in user.User.favorite_colors
Failed example:
 john.favorite_colors
Expected:
 {'#797EF6', '#4ADEDE', '#1AA7EC'} Got:
 {'#797EF6', '#1AA7EC', '#4ADEDE'} 3 items had no tests:
 user
 user.User
 user.User.__init__
**********************************************************************
1 items had failures:
 1 of   2 in user.User.favorite_colors
2 tests in 4 items.
1 passed and 1 failed.
***Test Failed*** 1 failures.
```

第一个测试是`User`实例化，它没有任何预期的输出，因为结果被赋给了一个变量。第二个测试对照函数的实际输出检查预期输出。输出是不同的，因为集合是无序的集合，这使得测试失败。

要解决这个问题，您可以在您的`doctest`测试中使用内置的 [`sorted()`](https://realpython.com/python-sort/) 函数:

```py
# user.py

class User:
    def __init__(self, name, favorite_colors):
        self.name = name
        self._favorite_colors = set(favorite_colors)

    @property
    def favorite_colors(self):
        """Return the user's favorite colors.

 Usage examples:
 >>> john = User("John", {"#797EF6", "#4ADEDE", "#1AA7EC"})
 >>> sorted(john.favorite_colors) ['#1AA7EC', '#4ADEDE', '#797EF6'] """
        return self._favorite_colors
```

现在，第二个`doctest`测试被包装在对`sorted()`的调用中，该调用返回一个排序颜色的[列表](https://realpython.com/python-lists-tuples/)。请注意，您还必须更新预期的输出，以包含排序后的颜色列表。现在测试将成功通过。去试试吧！

缺乏[参数化](https://realpython.com/pytest-python-testing/#parametrization-combining-tests)能力是`doctest`的另一个限制。参数化包括为给定的测试提供输入参数和预期输出的多种组合。测试框架必须对每个组合运行目标测试，并检查是否所有的组合都通过了测试。

参数化允许你用一个测试函数快速创建多个测试用例，这将增加你的[测试覆盖率](https://en.wikipedia.org/wiki/Code_coverage)，并提高你的生产力。即使`doctest`不直接支持参数化，你也可以用一些方便的技术来模拟该特征:

```py
# even_numbers.py

def get_even_numbers(numbers):
    """Return the even numbers in a list.

 >>> args = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
 >>> expected = [[2, 4], [6, 8], [10, 12]]

 >>> for arg, expected in zip(args, expected):
 ...     get_even_numbers(arg) == expected
 True
 True
 True
 """
    return [number for number in numbers if number % 2 == 0]
```

在这个例子中，首先创建两个列表，包含输入参数和`get_even_numbers()`的预期输出。`for`循环使用内置的 [`zip()`](https://realpython.com/python-zip-function/) 函数并行遍历两个列表。在循环内部，您运行一个测试，将`get_even_numbers()`的实际输出与相应的预期输出进行比较。

`doctest`的另一个具有挑战性的用例是当对象依赖默认字符串表示`object.__repr__()`时测试对象的创建。Python 对象的默认字符串表示通常包括对象的内存地址，每次运行时都会有所不同，这会导致测试失败。

继续`User`的例子，假设您想要将下面的测试添加到类初始化器中:

```py
# user.py

class User:
    def __init__(self, name, favorite_colors):
        """Initialize instances of User.

 Usage examples:
 >>> User("John", {"#797EF6", "#4ADEDE", "#1AA7EC"})
 <user.User object at 0x103283970>
 """
        self.name = name
        self._favorite_colors = set(favorite_colors)

# ...
```

当实例化`User`时，显示默认的字符串表示。在这种情况下，输出包括随执行而变化的存储器地址。这种变化会使您的`doctest`测试失败，因为内存地址永远不会匹配:

```py
$ python -m doctest -v user.py
Trying:
 User("John", {"#797EF6", "#4ADEDE", "#1AA7EC"})
Expecting:
 <user.User object at 0x103283970>
**********************************************************************
File ".../user.py", line 40, in user.User.__init__
Failed example:
 User("John", {"#797EF6", "#4ADEDE", "#1AA7EC"})
Expected:
 <user.User object at 0x103283970> Got:
 <user.User object at 0x10534b070> 2 items had no tests:
 user
 user.User
**********************************************************************
1 items had failures:
 1 of   1 in user.User.__init__
1 tests in 3 items.
0 passed and 1 failed.
***Test Failed*** 1 failures.
```

这种测试总是失败，因为每次运行代码时，`User`实例都会占用不同的内存地址。作为解决这个问题的方法，您可以使用`doctest`的 [`ELLIPSIS`](https://docs.python.org/3/library/doctest.html#doctest.ELLIPSIS) **指令**:

```py
# user.py

class User:
    def __init__(self, name, favorite_colors):
        """Initialize instances of User.

 Usage examples:
 >>> User("John", {"#797EF6", "#4ADEDE", "#1AA7EC"}) # doctest: +ELLIPSIS <user.User object at 0x...>
 """
        self.name = name
        self._favorite_colors = set(favorite_colors)

# ...
```

您已经在突出显示的行的末尾添加了注释。这个注释在测试中启用了`ELLIPSIS`指令。现在，您可以在预期的输出中用省略号替换内存地址。如果您现在运行测试，那么它会通过，因为`doctest`将省略号理解为测试输出的可变部分的替换。

**注意:**`doctest`模块定义了一些其他的指令，您可以在不同的情况下使用。您将在[控制`doctest`的行为:标志和指令](#controlling-the-behavior-of-doctest-flags-and-directives)一节中了解更多关于它们的信息。

当您希望输出包含对象的标识时，也会出现类似的问题，如下例所示:

>>>

```py
>>> id(1.0)
4402192272
```

您将无法使用`doctest`来测试这样的代码。在这个例子中，您不能使用`ELLIPSIS`指令，因为您必须用省略号替换完整的输出，`doctest`会将这三个点解释为继续提示。因此，看起来测试没有输出。

考虑下面的演示示例:

```py
# identity.py

def get_id(obj):
    """Return the identity of an object.

 >>> get_id(1)  # doctest: +ELLIPSIS
 ...
 """
    return id(obj)
```

这个函数只是一个例子，说明即使使用了`ELLIPSIS`指令，一个对象的身份也会使你的测试失败。如果您用`doctest`运行这个测试，那么您将得到一个失败:

```py
$ python -m doctest -v identity.py
Trying:
 get_id(1)  # doctest: +ELLIPSIS
Expecting nothing
**********************************************************************
File ".../identity.py", line 4, in identity.get_id
Failed example:
 get_id(1)  # doctest: +ELLIPSIS
Expected nothing Got:
 4340007152 1 items had no tests:
 identity
**********************************************************************
1 items had failures:
 1 of   1 in identity.get_id
1 tests in 2 items.
0 passed and 1 failed.
***Test Failed*** 1 failures.
```

正如您可以从这个输出中突出显示的行中确认的那样，`doctest`期望输出什么都不是，但是收到了一个实际的对象标识。因此，测试失败。

使用`doctest`时要记住的最后一个主题是，在一个 docstring 中有许多测试会使代码难以阅读和理解，因为长的 docstring 会增加函数签名和函数体之间的距离。

幸运的是，现在这不是一个大问题，因为大多数代码编辑器允许您折叠文档字符串并专注于代码。或者，您可以将测试移动到模块级的 docstring 或专用的测试文件中。

[*Remove ads*](/account/join/)

## 使用`doctest` 时考虑安全性

在当今的信息技术行业中，安全性是一项普遍而重要的要求。从外部来源运行代码，包括以字符串或文档字符串形式出现的代码，总是隐含着安全风险。

`doctest`模块在内部使用 [`exec()`](https://realpython.com/python-exec/) 来执行嵌入在文档字符串和文档文件中的任何测试，这可以从模块的源代码中得到证实:

```py
# doctest.py

class DocTestRunner:
    # ...

    def __run(self, test, compileflags, out):
        # ...
        try:
 # Don't blink!  This is where the user's code gets run.            exec(
                compile(example.source, filename, "single", compileflags, True),
                test.globs
            )
            self.debugger.set_continue() # ==== Example Finished ====
            exception = None
        except KeyboardInterrupt:
        # ...
```

正如突出显示的行所指出的，用户代码在对`exec()`的调用中运行。这个内置函数在 Python 社区中是众所周知的，因为它是一个相当危险的工具，允许执行任意代码。

`doctest`模块也不能幸免于与`exec()`相关的潜在[安全问题](https://realpython.com/python-exec/#uncovering-and-minimizing-the-security-risks-behind-exec)。所以，如果你曾经用`doctest`测试进入外部代码库，那么避免运行测试，直到你仔细通读它们并确保它们在你的计算机上运行是安全的。

## 使用`doctest`进行测试驱动开发

在实践中，您可以使用两种不同的方法来编写和运行使用`doctest`的测试。**第一种方法**包括以下步骤:

1.  写你的代码。
2.  在 Python REPL 中运行代码。
3.  将相关的 REPL 片段复制到您的文档字符串或文档中。
4.  使用`doctest`运行测试。

这种方法的主要缺点是您在步骤 1 中编写的实现可能有问题。这也违背了[测试驱动开发(TDD)](https://realpython.com/python-hash-table/#take-a-crash-course-in-test-driven-development) 的理念，因为你是在写完代码之后再写测试。

相比之下，**第二种方法**包括在编写通过测试的代码之前编写`doctest`测试。在这种情况下，步骤如下:

1.  使用`doctest`语法在文档字符串或文档中编写测试。
2.  编写通过测试的代码。
3.  使用`doctest`运行测试。

这种方法保留了 TDD 的精神，即您应该在编写代码之前编写测试。

举个例子，假设你正在参加一个[面试](https://realpython.com/python-coding-interview-tips/)，面试官要求你实现 [FizzBuzz](https://en.wikipedia.org/wiki/Fizz_buzz) 算法。您需要编写一个函数，它接受一个数字列表，并将任何可被 3 整除的数字替换为单词`"fizz"`，将任何可被 5 整除的数字替换为单词`"buzz"`。如果一个数能被 3 和 5 整除，那么你必须用`"fizz buzz"`字符串替换它。

您希望使用 TDD 技术来编写这个函数，以确保可靠性。因此，您决定使用`doctest`测试作为快速解决方案。首先，编写一个测试来检查能被 3 整除的数字:

```py
# fizzbuzz.py

# Replace numbers that are divisible by 3 with "fizz"
def fizzbuzz(numbers):
    """Implement the Fizz buzz game.

 >>> fizzbuzz([3, 6, 9, 12])
 ['fizz', 'fizz', 'fizz', 'fizz']
 """
```

该函数还没有实现。它只有一个`doctest`测试，用于检查当输入数字被 3 整除时，函数是否如预期的那样工作。现在您可以运行测试来检查它是否通过:

```py
$ python -m doctest -v fizzbuzz.py
Trying:
 fizzbuzz([3, 6, 9, 12])
Expecting:
 ['fizz', 'fizz', 'fizz', 'fizz']
**********************************************************************
File ".../fizzbuzz.py", line 5, in fizzbuzz.fizzbuzz
Failed example:
 fizzbuzz([3, 6, 9, 12])
Expected:
 ['fizz', 'fizz', 'fizz', 'fizz']
Got nothing
1 items had no tests:
 fizzbuzz
**********************************************************************
1 items had failures:
 1 of   1 in fizzbuzz.fizzbuzz
1 tests in 2 items.
0 passed and 1 failed.
***Test Failed*** 1 failures.
```

这个输出告诉您有一个失败的测试，这符合您的函数还没有任何代码的事实。现在您需要编写代码来通过测试:

```py
# fizzbuzz.py

# Replace numbers that are divisible by 3 with "fizz"
def fizzbuzz(numbers):
    """Implement the Fizz buzz game.

 >>> fizzbuzz([3, 6, 9, 12])
 ['fizz', 'fizz', 'fizz', 'fizz']
 """
    result = []
    for number in numbers:
        if number % 3 == 0:
            result.append("fizz")
        else:
            result.append(number)
    return result
```

现在，您的函数对输入的数字进行迭代。在循环中，您在一个[条件语句](https://realpython.com/python-conditional-statements/)中使用模运算符(`%`)来检查当前数字是否能被 3 整除。如果检查成功，那么您[将`"fizz"`字符串追加](https://realpython.com/python-append/)到`result`，它最初保存一个空的`list`对象。否则，您将追加数字本身。

如果您现在用`doctest`运行测试，那么您将得到以下输出:

```py
python -m doctest -v fizzbuzz.py
Trying:
 fizzbuzz([3, 6, 9, 12])
Expecting:
 ['fizz', 'fizz', 'fizz', 'fizz']
ok
1 items had no tests:
 fizz
1 items passed all tests:
 1 tests in fizz.fizzbuzz
1 tests in 2 items.
1 passed and 0 failed.
Test passed.
```

酷！你已经通过了测试。现在您需要测试能被 5 整除的数字。以下是更新后的`doctest`测试以及通过测试的代码:

```py
# fizzbuzz.py

# Replace numbers that are divisible by 5 with "buzz"
def fizzbuzz(numbers):
    """Implement the Fizz buzz game.

 >>> fizzbuzz([3, 6, 9, 12])
 ['fizz', 'fizz', 'fizz', 'fizz']

 >>> fizzbuzz([5, 10, 20, 25]) ['buzz', 'buzz', 'buzz', 'buzz'] """
    result = []
    for number in numbers:
        if number % 3 == 0:
            result.append("fizz")
 elif number % 5 == 0: result.append("buzz")        else:
            result.append(number)
    return result
```

前两行突出显示的内容提供了被 5 整除的数字的测试和预期输出。第二对突出显示的行实现了运行检查的代码，并用所需的字符串`"buzz"`替换数字。继续运行测试以确保代码通过。

最后一步是检查能被 3 和 5 整除的数字。通过检查能被 15 整除的数字，你可以一步完成。下面是`doctest`测试和所需的代码更新:

```py
# fizzbuzz.py

# Replace numbers that are divisible by 3 and 5 with "fizz buzz"
def fizzbuzz(numbers):
    """Implement the Fizz buzz game.

 >>> fizzbuzz([3, 6, 9, 12])
 ['fizz', 'fizz', 'fizz', 'fizz']

 >>> fizzbuzz([5, 10, 20, 25])
 ['buzz', 'buzz', 'buzz', 'buzz']

 >>> fizzbuzz([15, 30, 45]) ['fizz buzz', 'fizz buzz', 'fizz buzz'] 
 >>> fizzbuzz([3, 6, 5, 2, 15, 30]) ['fizz', 'fizz', 'buzz', 2, 'fizz buzz', 'fizz buzz'] """
    result = []
    for number in numbers:
 if number % 15 == 0: result.append("fizz buzz") elif number % 3 == 0:            result.append("fizz")
        elif number % 5 == 0:
            result.append("buzz")
        else:
            result.append(number)
    return result
```

在对`fizzbuzz()`函数的最后一次更新中，您添加了`doctest`测试来检查能被 3 和 5 整除的数字。您还将添加一个最终测试，用不同的数字来检查函数。

在函数体中，在链式`if` … `elif`语句的开头添加一个新分支。这个新分支检查可被 3 和 5 整除的数字，用`"fizz buzz"`字符串替换它们。请注意，您需要将该检查放在链的开始，因为否则，该函数将不能很好地工作。

[*Remove ads*](/account/join/)

## 运行您的 Python `doctest`测试

到目前为止，您已经运行了许多`doctest`测试。要运行它们，您可以使用命令行和带有`-v`选项的`doctest`命令来生成详细输出。然而，这并不是运行您的`doctest`测试的唯一方式。

在接下来的小节中，您将学习如何从 Python 代码内部运行`doctest`测试。您还将了解关于从命令行或终端运行`doctest`的更多细节。

### 从您的代码运行`doctest`

Python 的`doctest`模块导出了两个函数，当您需要从 Python 代码而不是命令行运行`doctest`测试时，这两个函数会派上用场。这些功能如下:

| 功能 | 描述 |
| --- | --- |
| [T2`testfile()`](https://docs.python.org/3/library/doctest.html#doctest.testfile) | 从专用的测试文件运行`doctest`测试 |
| [T2`testmod()`](https://docs.python.org/3/library/doctest.html#doctest.testmod) | 从 Python 模块运行`doctest`测试 |

以您的`test_calculations.txt`为起点，您可以使用 Python 代码中的`testfile()`来运行这个文件中的测试。为此，您只需要两行代码:

```py
# run_file_tests.py

import doctest

doctest.testfile("test_calculations.txt", verbose=True)
```

第一行导入`doctest`，而第二行使用您的测试文件作为参数调用`testfile()`。在上面的例子中，您使用了`verbose`参数，这使得函数产生详细的输出，就像从命令行运行`doctest`时使用的`-v`选项一样。如果不将`verbose`设置为`True`，那么`testfile()`将不会显示任何输出，除非测试失败。

测试文件的内容被视为包含您的`doctest`测试的单个 docstring。该文件不必是 Python 程序或模块。

`testfile()`函数带有几个其他的[可选参数](https://realpython.com/python-optional-arguments/)，允许您在运行测试的过程中定制进一步的细节。您必须使用[关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-arguments)来提供函数的任何可选参数。查看[函数的文档](https://docs.python.org/3/library/doctest.html#doctest.testfile),了解关于其参数及其各自含义的更多信息。

如果您需要从您的代码库运行常规 Python 模块中的`doctest`测试，那么您可以使用`testmod()`函数。您可以通过两种不同的方式使用该函数。第一种方法是在目标模块中附加以下代码片段:

```py
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
```

当文件[作为脚本](https://realpython.com/run-python-scripts/)运行时， [name-main](https://realpython.com/if-name-main-python/) 习语允许您执行代码，但当它作为模块导入时则不允许。在这个条件中，首先导入`doctest`，然后在`verbose`设置为`True`的情况下调用`testmod()`。如果你将模块作为脚本运行，那么`doctest`将运行它在模块中发现的所有测试。

`testmod()`的所有参数都是可选的。为了提供它们，除了第一个参数之外，您需要对所有参数使用关键字参数，第一个参数可以选择保存一个模块对象。

用`testmod()`运行`doctest`测试的第二种方法是创建一个专用的测试运行程序文件。例如，如果您想在`calculation.py`中运行测试而不修改模块本身，那么您可以创建一个包含以下内容的`run_module_tests.py`文件:

```py
# run_module_tests.py

import doctest

import calculations

doctest.testmod(calculations, verbose=True)
```

这一次，您需要导入目标模块`calculations`，并将模块对象作为第一个参数传递给`testmod()`。这个调用将使`doctest`运行`calculations.py`中定义的所有测试。继续使用下面的命令尝试一下:

```py
$ python run_module_tests.py
```

运行这个命令后，您将得到典型的`doctest`输出，其中包含了关于您的`calculations`模块中测试的所有细节。关于命令的输出，重要的是要记住，如果不将`verbose`设置为`True`，那么除非测试失败，否则不会得到任何输出。在下一节中，您将了解更多关于失败测试的输出。

除了目标模块和`verbose`标志之外，`testmod()`还有几个其他参数，允许您调整测试执行的不同方面。查看函数的[文档](https://docs.python.org/3/library/doctest.html#doctest.testmod)以获得关于当前参数的更多细节。

最后，本节中的函数旨在使`doctest`易于使用。然而，它们给你有限的定制能力。如果您需要用`doctest`对测试代码的过程进行更细粒度的控制，那么您可以使用该模块的高级 [API](https://docs.python.org/3/library/doctest.html#advanced-api) 。

[*Remove ads*](/account/join/)

### 从命令行执行`doctest`

您已经知道了使用`doctest`命令从命令行运行`doctest`测试的基础。使用该命令最简单的方法是将目标文件或模块作为参数。例如，您可以通过执行以下命令来运行您的`calculations.py`文件中的所有测试:

```py
$ python -m doctest calculations.py
```

这个命令运行测试，但是不发出任何输出，除非您有一些失败的测试。这就是为什么在迄今为止运行的几乎所有示例中都使用了`-v`开关。

正如您已经了解到的，`-v`或`--verbose`开关让`doctest`发布一份它已经运行的所有测试的详细报告，并在报告的末尾附上一份摘要。除了这个命令行选项，`doctest`还接受以下选项:

| [计]选项 | 描述 |
| --- | --- |
| `-h`，`--help` | 显示`doctest`的命令行帮助 |
| `-o`，`--option` | 指定在运行测试时使用的一个或多个`doctest`选项标志或指令 |
| `-f`，`--fail-fast` | 在第一次失败后停止运行您的`doctest`测试 |

在大多数情况下，您可能会从命令行运行`doctest`。在上表中，您会发现最复杂的选项是`-o`或`--option`，因为有一个很长的标志列表可供您使用。您将在命令行部分的[使用标志中了解更多关于这些标志的信息。](#using-flags-at-the-command-line)

## 控制`doctest`的行为:标志和指令

`doctest`模块提供了一系列命名的[常量](https://realpython.com/python-constants/)，当您使用`-o`或`--option`开关从命令行运行`doctest`时，可以将它们用作标志。当您向您的`doctest`测试添加指令时，您也可以使用这些常量。

使用这组常量作为命令行标志或指令将允许您控制`doctest`的各种行为，包括:

*   接受`1`的`True`
*   拒绝空行
*   规范化空白
*   用省略号(`...`)缩写输出
*   忽略异常细节，如异常消息
*   跳过给定的测试
*   第一次失败测试后结束

此列表不包括所有当前选项。您可以查看[文档](https://docs.python.org/3/library/doctest.html#option-flags)以获得常量及其含义的完整列表。

在下一节中，您将从学习如何从命令行使用这个简洁的`doctest`特性开始。

### 在命令行使用标志

当您使用`-o`或`--option`开关从命令行运行`doctest`时，您可以使用标志常量。例如，假设您有一个名为`options.txt`的测试文件，其内容如下:

```py
>>> 5 < 7
1
```

在这个测试中，您使用`1`作为预期输出，而不是使用`True`。这个测试会通过，因为`doctest`允许分别用`1`和`0`替换`True`和`False`。这个特性与 Python [布尔](https://realpython.com/python-boolean/)值可以用整数表示[这一事实有关。因此，如果您用`doctest`运行这个文件，那么测试将会通过。](https://realpython.com/python-boolean/#python-booleans-as-numbers)

历史上，`doctest`让布尔值被`1`和`0`代替，以方便向 [Python 2.3](https://docs.python.org/3/whatsnew/2.3.html) 的过渡，后者引入了专用的[布尔类型](https://peps.python.org/pep-0285/)。但是，这种行为在某些情况下可能不完全正确。幸运的是， [`DONT_ACCEPT_TRUE_FOR_1`](https://docs.python.org/3/library/doctest.html#doctest.DONT_ACCEPT_TRUE_FOR_1) 标志将使这个测试失败:

```py
$ python -m doctest -o DONT_ACCEPT_TRUE_FOR_1 options.txt
**********************************************************************
File "options.txt", line 3, in options.txt
Failed example:
 5 < 7
Expected:
 1 Got:
 True **********************************************************************
1 items had failures:
 1 of   1 in options.txt
***Test Failed*** 1 failures.
```

通过运行带有`DONT_ACCEPT_TRUE_FOR_1`标志的`doctest`命令，您可以让测试严格检查布尔值、`True`或`False`，如果是整数则失败。要修复测试，您必须将预期输出从`1`更新到`True`。之后，您可以再次运行测试，它会通过。

现在假设您有一个具有大量输出的测试，您需要一种方法来简化预期的输出。在这种情况下，`doctest`允许您使用省略号。继续将以下测试添加到您的`options.txt`图块的末尾:

```py
>>> print("Hello, Pythonista! Welcome to Real Python!")
Hello, ... Python!
```

如果您使用`doctest`运行这个文件，那么第二个测试将会失败，因为预期的输出与实际的输出不匹配。为了避免这种失败，您可以使用`ELLIPSIS`标志运行`doctest`:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
PS> python -m doctest `
> -o DONT_ACCEPT_TRUE_FOR_1 `
> -o ELLIPSIS options.txt
```

```py
$ python -m doctest \
    -o DONT_ACCEPT_TRUE_FOR_1 \
 -o ELLIPSIS options.txt
```

这个命令不会为您的第二个测试发出任何输出，因为您使用了 [`ELLIPSIS`](https://docs.python.org/3/library/doctest.html#doctest.ELLIPSIS) 标志。这个标志让`doctest`知道`...`字符替换了部分预期输出。

注意，要在`doctest`的一次运行中传递多个标志，每次都需要使用`-o`开关。遵循这种模式，您可以根据需要使用任意多的标志，使您的测试更加健壮、严格或灵活。

处理像制表符这样的空白字符是一项相当具有挑战性的任务，因为`doctest`会自动用常规空格替换它们，从而使您的测试失败。考虑向您的`options.txt`文件添加一个新的测试:

```py
>>> print("\tHello, World!")
    Hello, World!
```

即使您在测试及其预期输出中使用制表符，该测试也会失败，因为`doctest`在预期输出中用空格内部替换制表符:

*   [*视窗*](#windows-2)
**   [**Linux + macOS**](#linux-macos-2)*

```py
PS> python -m doctest `
> -o DONT_ACCEPT_TRUE_FOR_1 `
> -o ELLIPSIS options.txt
**********************************************************************
File "options.txt", line 9, in options.txt
Failed example:
 print("\tHello, World!")
Expected:
 Hello, World!
Got:
 Hello, World!
**********************************************************************
1 items had failures:
 1 of   3 in options.txt
***Test Failed*** 1 failures.
```

```py
$ python -m doctest \
    -o DONT_ACCEPT_TRUE_FOR_1 \
    -o ELLIPSIS options.txt
**********************************************************************
File "options.txt", line 9, in options.txt
Failed example:
 print("\tHello, World!")
Expected:
 Hello, World!
Got:
 Hello, World!
**********************************************************************
1 items had failures:
 1 of   3 in options.txt
***Test Failed*** 1 failures.
```

如果您曾经有一个测试发出了一个像这样的特殊空白字符，那么您可以在下面的命令中使用 [`NORMALIZE_WHITESPACE`](https://docs.python.org/3/library/doctest.html#doctest.NORMALIZE_WHITESPACE) 标志:

*   [*视窗*](#windows-3)
**   [**Linux + macOS**](#linux-macos-3)*

```py
PS> python -m doctest `
> -o DONT_ACCEPT_TRUE_FOR_1 `
> -o ELLIPSIS `
> -o NORMALIZE_WHITESPACE options.txt
```

```py
$ python -m doctest \
    -o DONT_ACCEPT_TRUE_FOR_1 \
    -o ELLIPSIS \
 -o NORMALIZE_WHITESPACE options.txt
```

现在您的输出将是干净的，因为`doctest`已经为您规范化了制表符。

### 在您的`doctest`测试中嵌入指令

一个`doctest` [指令](https://docs.python.org/3/library/doctest.html#doctest-directives)由一个行内注释组成，该注释以`# doctest:`开始，然后包括一个逗号分隔的标志常量列表。指令*启用*或*禁用*给定的`doctest`功能。要启用该功能，请在旗帜名称前写一个加号(`+`)。要禁用某个功能，可以写一个减号(`–`)来代替。

当您从命令行使用`doctest`时，指令的工作方式类似于标志。然而，指令允许您进行更细粒度的控制，因为它们在您的`doctest`测试中的特定行上工作。

例如，您可以向您的`options.txt`添加一些指令，这样您就不需要在运行`doctest`时传递多个命令行标志:

```py
>>> 5 < 7  # doctest: +DONT_ACCEPT_TRUE_FOR_1 True

>>> print(
...    "Hello, Pythonista! Welcome to Real Python!"
... )  # doctest: +ELLIPSIS Hello, ... Python!

>>> print("\tHello, World!")  # doctest: +NORMALIZE_WHITESPACE
    Hello, World!
```

在这段代码中，突出显示的行在测试旁边插入内联指令。第一个指令允许强制使用布尔值。第二个指令允许您使用省略号来缩写测试的预期输出。final 指令对预期和实际输出中的空白字符进行规范化。

现在，您可以运行`options.txt`文件，而无需向`doctest`命令传递任何标志:

```py
$ python -m doctest options.txt
```

这个命令不会发出任何输出，因为`doctest`指令已经处理了测试的所有需求。

在`doctest`中，标志和指令非常相似。主要的区别在于，标志旨在从命令行使用，而指令必须在测试本身中使用。在某种意义上，标志比指令更动态。当在测试策略中使用标志、指令或两者时，您总能找到一个好的平衡点。

## 使用`unittest`和 pytest 运行`doctest`测试

`doctest`模块提供了一种非常方便的方式来将测试用例添加到项目的文档中。然而，`doctest`并不能替代成熟的测试框架，比如标准库 [`unittest`](https://docs.python.org/3/library/unittest.html?highlight=unittest#module-unittest) 或者第三方 [pytest](https://realpython.com/pytest-python-testing/) 。在具有大量复杂代码库的大型项目中尤其如此。对于这类项目，`doctest`可能不够用。

举例来说，假设你正在开始一个新项目，为少数客户提供一个创新的 [web 服务](https://realpython.com/api-integration-in-python/#rest-apis-and-web-services)。在这一点上，您认为使用`doctest`来自动化您的测试过程是可以的，因为这个项目的规模和范围都很小。因此，您在文档和 docstrings 中嵌入了一堆`doctest`测试，每个人都很高兴。

在没有任何警告的情况下，您的项目开始变得越来越大，越来越复杂。您现在为越来越多的用户提供服务，他们不断要求新的功能和错误修复。现在，您的项目需要提供更可靠的服务。

由于这种新情况，你已经注意到`doctest`测试不够灵活和强大，不足以确保可靠性。你需要一个全功能的测试框架，包括夹具、安装和拆卸机制、参数化等等。

在这种情况下，您认为如果您决定使用`unittest`或 pytest，那么您将不得不重写所有旧的`doctest`测试。好消息是你不必这么做。`unittest`和 pytest 都可以运行`doctest`测试。这样，您的旧测试将自动加入到您的测试用例库中。

### 使用`unittest`运行`doctest`测试

如果你想用`unittest`运行`doctest`测试，那么你可以使用`doctest` API。API 允许你将`doctest`测试转换成 [`unittest`测试套件](https://docs.python.org/3/library/unittest.html#unittest.TestSuite)。为此，您将有两个主要函数:

| 功能 | 描述 |
| --- | --- |
| [T2`DocFileSuite()`](https://docs.python.org/3/library/doctest.html#doctest.DocFileSuite) | 将一个或多个文本文件中的`doctest`测试转换成一个`unittest`测试套件 |
| [T2`DocTestSuite()`](https://docs.python.org/3/library/doctest.html#doctest.DocTestSuite) | 将模块中的`doctest`测试转换成`unittest`测试套件 |

要将您的`doctest`测试与`unittest`发现机制集成，您必须向您的`unittest`样板代码添加一个`load_tests()`函数。举个例子，回到你的`test_calculations.txt`文件:

```py
>>> import calculations

>>> calculations.add(2, 2)
4.0

>>> calculations.subtract(2, 2)
0.0

>>> calculations.multiply(2, 2)
4.0

>>> calculations.divide(2, 2)
1.0
```

正如您已经知道的，这个文件包含了对您的`calculations.py`文件的`doctest`测试。现在假设您需要将`test_calculations.txt`中的`doctest`测试集成到您的`unittest`基础设施中。在这种情况下，您可以执行如下操作:

```py
# test_calculations.py

import doctest
import unittest

def load_tests(loader, tests, ignore):
 tests.addTests(doctest.DocFileSuite("test_calculations.txt"))    return tests

# Your unittest tests goes here...

if __name__ == "__main__":
    unittest.main()
```

`unittest`会自动调用`load_tests()`函数，框架会在您的代码中发现测试。突出显示的线条很神奇。它加载在`test_calculations.txt`中定义的`doctest`测试，并将它们转换成`unittest`测试套件。

一旦将这个函数添加到您的`unittest`基础设施中，您就可以使用以下命令运行该套件:

```py
$ python test_calculations.py
.
---------------------------------------------------------------
Ran 1 test in 0.004s

OK
```

酷！您的`doctest`测试成功运行。从这个输出中，您可以得出结论，`unittest`将测试文件的内容解释为单个测试，这与`doctest`将测试文件解释为单个 docstring 的事实是一致的。

在上面的例子中，所有的测试都通过了。如果您曾经有过失败的测试，那么您将得到模拟失败测试的常规`doctest`输出的输出。

如果您的`doctest`测试存在于您代码的文档字符串中，那么您可以使用下面的`load_tests()`变体将它们集成到您的`unittest`套件中:

```py
# test_calculations.py

import doctest
import unittest

import calculations 
def load_tests(loader, tests, ignore):
 tests.addTests(doctest.DocTestSuite(calculations))    return tests

# Your unittest goes here...

if __name__ == "__main__":
    unittest.main()
```

您不是从专用的测试文件中加载`doctest`测试，而是使用`DocTestSuite()`函数从`calculations.py`模块中读取它们。如果您现在运行上面的文件，那么您将得到以下输出:

```py
$ python test_calculations.py
.....
---------------------------------------------------------------
Ran 5 tests in 0.004s

OK
```

这一次，输出反映了五个测试。原因是您的`calculations.py`文件包含一个模块级 docstring 和四个带有`doctest`测试的函数级 docstring。每个独立的文档字符串被解释为一个单独的测试。

最后，您还可以将来自一个或多个文本文件的测试和来自`load_tests()`函数中的一个模块的测试结合起来:

```py
import doctest
import unittest

import calculations

def load_tests(loader, tests, ignore):
 tests.addTests(doctest.DocFileSuite("test_calculations.txt")) tests.addTests(doctest.DocTestSuite(calculations))    return tests

if __name__ == "__main__":
    unittest.main()
```

这个版本的`load_tests()`从`test_calculations.txt`和`calculations.py`模块运行`doctest`测试。继续从命令行运行上面的脚本。您的输出将反映六个通过的测试，包括来自`calculations.py`的五个测试和来自`test_calculations.txt`的一个测试。记住像`test_calculations.txt`这样的专用测试文件被解释为一个单独的测试。

### 使用 pytest 运行`doctest`测试

如果你决定使用 pytest 第三方库来自动化你的项目测试，那么你也可以[集成你的`doctest`测试](https://doc.pytest.org/en/latest/how-to/doctest.html)。在这种情况下，您可以使用 pytest 的`--doctest-glob`命令行选项，如下例所示:

```py
$ pytest --doctest-glob="test_calculations.txt"
```

运行此命令时，您会得到如下输出:

```py
===================== test session starts =====================
platform darwin -- Python 3.10.3, pytest-7.1.1, pluggy-1.0.0
rootdir: .../python-doctest/examples
collected 1 item

test_calculations.txt .                                  [100%]

===================== 1 passed in 0.02s =======================
```

就像`unittest`一样，pytest 将您的专用测试文件解释为单个测试。`--doctest-glob`选项接受并匹配允许您运行多个文件的模式。一个有用的模式可能是`"test*.txt"`。

您也可以直接从代码的文档字符串中执行`doctest`测试。为此，您可以使用`--doctest-modules`命令行选项。这个命令行选项将扫描您的工作目录下的所有模块，加载并运行它找到的任何`doctest`测试。

如果您想使这种集成永久化，那么您可以将以下参数添加到项目根目录下的 pytest 配置文件中:

```py
; pytest.ini

[pytest]
addopts = --doctest-modules
```

从现在开始，每当您在项目目录上运行 pytest 时，所有的`doctest`测试都会被找到并执行。

## 结论

现在你知道如何编写同时作为**文档**和**测试用例**的代码示例。为了将您的示例作为测试用例运行，您使用了 Python 标准库中的`doctest`模块。这个模块用一个低学习曲线的快速测试框架武装了你，允许你立即开始自动化你的测试过程。

**在本教程中，您学习了如何:**

*   将 **`doctest`测试**添加到您的文档和文档字符串中
*   使用 Python 的 **`doctest`** 模块
*   解决`doctest`中的**限制**和**安全隐患**
*   使用`doctest`和**测试驱动开发**方法
*   使用不同的**策略和工具**执行`doctest`测试

使用`doctest`测试，您将能够快速自动化您的测试。您还将保证您的代码及其文档始终保持同步。

**示例代码:** [点击这里下载免费的示例代码](https://realpython.com/bonus/python-doctest-code/)，您将使用 Python 的`doctest`同时记录和测试您的代码。***********************