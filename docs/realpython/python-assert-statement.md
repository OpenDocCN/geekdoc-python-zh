# Python 的断言:像专家一样调试和测试你的代码

> 原文：<https://realpython.com/python-assert-statement/>

Python 的`assert`语句允许你在代码中编写[健全性检查](https://en.wikipedia.org/wiki/Sanity_check)。这些检查被称为**断言**，当你开发代码时，你可以用它们来测试某些假设是否成立。如果您的任何断言为假，那么您的代码中就有一个 bug。

在开发过程中，断言是**记录**、**调试**、**测试**代码的便利工具。一旦您在断言的帮助下调试和测试了您的代码，那么您就可以关闭它们来为生产优化代码。断言将帮助您使您的代码更加高效、健壮和可靠。

在本教程中，您将学习:

*   什么是断言以及何时使用它们
*   Python 的 **`assert`语句**如何工作
*   `assert`如何帮助你**记录**，**调试**，以及**测试**你的代码
*   如何禁用断言以提高生产中的性能
*   使用`assert`语句时，你可能会面临哪些**常见陷阱**

为了从本教程中获得最大收益，您应该已经了解了[表达式和运算符](https://realpython.com/python-operators-expressions/)、[函数](https://realpython.com/defining-your-own-python-function/)、[条件语句](https://realpython.com/python-conditional-statements/)和[异常](https://realpython.com/python-exceptions/)。对[编写](https://realpython.com/documenting-python-code/)、[调试](https://realpython.com/python-debugging-pdb/)和[测试](https://realpython.com/python-testing/) Python 代码有基本了解者优先。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 了解 Python 中的断言

Python 实现了一个名为[断言](https://en.wikipedia.org/wiki/Assertion_(software_development))的特性，这在应用程序和项目的开发过程中非常有用。在其他几种语言中你也会发现这个特性，比如 [C](https://realpython.com/c-for-python-programmers/) 和 [Java](https://realpython.com/java-vs-python/) ，它对于[记录](https://realpython.com/documenting-python-code/)、[调试](https://realpython.com/python-debugging-pdb/)和[测试](https://realpython.com/python-testing/)你的代码来说很方便。

如果你正在寻找一个工具来加强你的调试和测试过程，那么断言就是你要找的。在这一节中，您将学习断言的基础知识，包括它们是什么，它们有什么好处，以及什么时候不应该在代码中使用它们。

[*Remove ads*](/account/join/)

### 什么是断言？

在 Python 中，断言是可以用来在开发过程中设置**健全性检查**的[语句](https://docs.python.org/3/glossary.html#term-statement)。断言允许您通过检查某些特定条件是否为真来测试代码的正确性，这在您调试代码时会很方便。

断言条件应该总是真的，除非你的程序有 bug。如果结果证明条件为假，那么断言将引发一个异常并终止程序的执行。

使用断言，您可以设置检查来确保代码中的[不变量](https://en.wikipedia.org/wiki/Invariant_(mathematics)#Invariants_in_computer_science)保持不变。通过这样做，你可以检查像[前置条件](https://en.wikipedia.org/wiki/Precondition)和[后置条件](https://en.wikipedia.org/wiki/Postcondition)这样的假设。例如，您可以按照如下方式测试条件:*此参数不是 [`None`](https://realpython.com/null-in-python/)* 或*此返回值是一个[字符串](https://realpython.com/python-strings/)* 。当你开发一个程序时，这种检查可以帮助你尽可能快地发现错误。

### 断言有什么好处？

断言主要是为了调试。它们将帮助您确保在添加特性和修复代码中的其他错误时，不会引入新的错误。然而，在您的开发过程中，它们可以有其他有趣的用例。这些用例包括记录和测试你的代码。

断言的主要作用是当程序中出现错误时触发警报。在这个上下文中，断言意味着*确保这个条件保持为真。否则，抛出一个错误。*

实际上，您可以在开发时使用断言来检查程序中的前置条件和后置条件。例如，程序员经常在函数的开头放置断言来检查输入是否有效(前提条件)。程序员还在函数的返回值之前放置断言，以检查输出是否有效(后置条件)。

断言清楚地表明，您想要检查给定的条件是否为真并且保持为真。在 Python 中，它们还可以包含一个可选的消息来明确描述即将发生的错误或问题。这就是为什么它们也是记录代码的有效工具。在这种情况下，他们的主要优势是采取具体行动的能力，而不是像[评论](https://realpython.com/python-comments-guide/)和[文档串](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)那样被动。

最后，断言对于在代码中编写[测试用例](https://en.wikipedia.org/wiki/Test_case)也是非常理想的。您可以编写简明扼要的测试用例，因为断言提供了一种快速检查给定条件是否满足的方法，它定义了测试是否通过。

在本教程的后面，您将了解更多关于断言的这些常见用例。现在你将学习什么时候*不应该*使用断言。

### 什么时候不使用断言？

一般来说，您不应该对**数据处理**或**数据验证**使用断言，因为您可以在生产代码中禁用断言，这最终会删除所有基于断言的处理和验证代码。使用断言进行数据处理和验证是一个常见的陷阱，您将在本教程后面的[理解`assert`](#understanding-common-pitfalls-of-assert) 的常见陷阱中了解到。

此外，断言不是一个**错误处理**工具。断言的最终目的不是处理生产中的错误，而是在开发过程中通知您，以便您可以修复它们。在这方面，您不应该使用常规的 [`try` … `except`](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions) 语句来编写捕捉断言错误的代码。

## 理解 Python 的`assert`语句

现在你知道了什么是断言，它们有什么用，以及什么时候不应该在代码中使用它们。是时候学习编写自己的断言的基础知识了。首先，注意 Python 将断言实现为带有关键字的`assert` [语句，而不是作为](https://realpython.com/python-keywords/)[函数](https://realpython.com/defining-your-own-python-function/)。这种行为可能是混乱和问题的常见来源，您将在本教程的后面部分了解到。

在本节中，您将学习使用 [`assert`](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement) 语句在代码中引入断言的基础知识。您将学习`assert`语句的语法。最重要的是，您将理解这个语句在 Python 中是如何工作的。最后，您还将学习 [`AssertionError`](https://docs.python.org/3/library/exceptions.html#AssertionError) 异常的基础知识。

### `assert`语句的语法

一条`assert`语句由`assert`关键字、要测试的表达式或条件以及一条可选消息组成。这个条件应该总是真的。如果断言条件为真，那么什么都不会发生，程序继续正常执行。另一方面，如果条件变为假，那么`assert`通过引发`AssertionError`来暂停程序。

在 Python 中，`assert`是一个[简单语句](https://docs.python.org/3/reference/simple_stmts.html#simple-statements)，语法如下:

```py
assert expression[, assertion_message]
```

这里，`expression`可以是任何有效的 Python [表达式](https://realpython.com/python-operators-expressions/)或对象，然后对其进行[真值](https://realpython.com/python-boolean/#python-boolean-testing)测试。如果`expression`为假，那么语句抛出一个`AssertionError`。`assertion_message`参数是可选的，但鼓励使用。它可以保存描述语句应该捕获的问题的字符串。

下面是这种说法在实践中的工作方式:

>>>

```py
>>> number = 42
>>> assert number > 0

>>> number = -42
>>> assert number > 0
Traceback (most recent call last):
    ...
AssertionError
```

对于真表达式，断言成功，并且什么也没有发生。在这种情况下，您的程序会继续正常执行。相反，falsy 表达式会使断言失败，引发一个`AssertionError`并中断程序的执行。

为了让其他开发人员明白您的`assert`语句，您应该添加一条描述性的断言消息:

>>>

```py
>>> number = 42
>>> assert number > 0, f"number greater than 0 expected, got: {number}"

>>> number = -42
>>> assert number > 0, f"number greater than 0 expected, got: {number}"
Traceback (most recent call last):
    ...
AssertionError: number greater than 0 expected, got: -42
```

该断言中的消息清楚地说明了哪个条件应该为真，以及是什么导致该条件失败。注意到`assert`的`assertion_message`参数是可选的。然而，它可以帮助你更好地理解测试中的情况，并找出你所面临的问题。

因此，无论何时使用`assert`，对`AssertionError`异常的[回溯](https://realpython.com/python-traceback/)使用描述性断言消息是一个好主意。

关于`assert`语法的重要一点是，这个语句*不需要一对括号来对表达式和可选消息进行分组。在 Python 中，`assert`是语句而不是函数。使用一对括号会导致意想不到的行为。*

例如，像下面这样的断言会引出一个 [`SyntaxWarning`](https://docs.python.org/3/library/exceptions.html#SyntaxWarning) :

>>>

```py
>>> number = 42

>>> assert(number > 0, f"number greater than 0 expected, got: {number}")
<stdin>:1: SyntaxWarning: assertion is always true, perhaps remove parentheses?
```

这个警告与 Python 中非空的[元组](https://realpython.com/python-lists-tuples/)总是为真有关。在本例中，括号将断言表达式和消息转换成一个两项元组，其值始终为 true。

幸运的是，Python 的最新版本抛出了一个`SyntaxWarning`来警告您这种误导性的语法。然而，在该语言的旧版本中，像上面这样的`assert`语句总是会成功。

当您使用超过一行的长表达式或消息时，这个问题经常出现。在这些情况下，括号是格式化代码的自然方式，您可能会得到如下结果:

```py
number = 42

assert (
    number > 0 and isinstance(number, int),
    f"number greater than 0 expected, got: {number}"
)
```

使用一对括号将一个长行分成多行是 Python 代码中常见的格式化实践。然而，在`assert`语句的上下文中，括号将断言表达式和消息变成了两项元组。

在实践中，如果您想将一个长断言拆分成几行，那么您可以使用反斜杠字符(`\`)来表示[显式行连接](https://docs.python.org/3/reference/lexical_analysis.html#explicit-line-joining):

```py
number = 42

assert number > 0 and isinstance(number, int), \
    f"number greater than 0 expected, got: {number}"
```

该断言第一行末尾的反斜杠将断言的两个[物理行](https://docs.python.org/3/reference/lexical_analysis.html#physical-lines)连接成一个[逻辑行](https://docs.python.org/3/reference/lexical_analysis.html#logical-lines)。通过这种方式，您可以拥有合适的[行长度](https://realpython.com/python-pep8/#maximum-line-length-and-line-breaking)，而不会在代码中出现警告或逻辑错误的风险。

**注意:** [PEP 679](https://www.python.org/dev/peps/pep-0679/) 创建于 2022 年 1 月 7 日，提议允许在断言表达式和消息周围使用括号。如果 PEP 得到批准和实现，那么偶然元组的问题在未来不会影响 Python 代码。

这个括号相关的问题有一个极端的例子。如果您只在括号中提供断言表达式，那么`assert`将会工作得很好:

>>>

```py
>>> number = 42
>>> assert(number > 0)

>>> number = -42
>>> assert(number > 0)
Traceback (most recent call last):
    ...
AssertionError
```

为什么会这样？要创建一个单项式元组，您需要在项本身后面放置一个逗号。在上面的代码中，括号本身不会创建元组。这就是解释器忽略括号的原因，`assert`按预期工作。

尽管括号在上面示例中描述的场景中似乎可以工作，但这不是推荐的做法。你可能会搬起石头砸自己的脚。

[*Remove ads*](/account/join/)

### `AssertionError`异常

如果一个`assert`语句的条件评估为假，那么`assert`引发一个 [`AssertionError`](https://docs.python.org/3/library/exceptions.html#AssertionError) 。如果您提供可选的断言消息，那么这个消息在内部被用作`AssertionError`类的参数。无论哪种方式，引发的异常都会中断程序的执行。

大多数时候，你不会在代码中显式地引发`AssertionError`异常。`assert`语句负责在断言条件失败时引发这个异常。此外，你不应该试图通过编写代码来捕捉`AssertionError`异常来处理错误，这一点你将在本教程的后面学到。

最后，`AssertionError`是一个继承自 [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception) 类的内置异常，被认为是一个**具体异常**，应该被抛出而不是子类化。

就是这样！现在你知道了`assert`语句的基础。您已经学习了语句的语法、`assert`在实践中如何工作，以及`AssertionError`异常的主要特征是什么。是时候向前迈进，探索一些用 Python 编写断言的有效而通用的方法了。

## 探索常见的断言格式

在编写`assert`语句时，您会发现 Python 代码中常见的几种断言格式。了解这些格式将允许您编写更好的断言。

以下示例展示了一些常见的断言格式，从比较对象的断言开始:

>>>

```py
>>> # Comparison assertions
>>> assert 3 > 2
>>> assert 3 == 2
Traceback (most recent call last):
    ...
AssertionError

>>> assert 3 > 2 and 5 < 10
>>> assert 3 == 2 or 5 > 10
Traceback (most recent call last):
    ...
AssertionError
```

**比较断言**旨在测试使用[比较运算符](https://realpython.com/python-operators-expressions/#comparison-operators)比较两个或更多对象的条件。这些断言还可以包括基于[布尔](https://realpython.com/python-boolean/)操作符的复合表达式。

另一种常见的断言格式与[成员资格](https://docs.python.org/3/reference/expressions.html#membership-test-operations)测试相关:

>>>

```py
>>> # Membership assertions
>>> numbers = [1, 2, 3, 4, 5]
>>> assert 4 in numbers
>>> assert 10 in numbers
Traceback (most recent call last):
    ...
AssertionError
```

**成员断言**允许你检查一个给定的条目是否存在于一个特定的集合中，比如一个[列表](https://realpython.com/python-lists-tuples/)，元组[，集合](https://realpython.com/python-sets/)，[字典](https://realpython.com/python-dicts/)等等。这些断言使用成员操作符 [`in`](https://docs.python.org/3/reference/expressions.html#in) 和 [`not in`](https://docs.python.org/3/reference/expressions.html#not-in) 来执行所需的检查。

以下示例中的断言格式与对象的[身份](https://realpython.com/python-is-identity-vs-equality/)相关:

>>>

```py
>>> # Identity assertions
>>> x = 1
>>> y = x
>>> null = None

>>> assert x is y
>>> assert x is not y
Traceback (most recent call last):
    ...
AssertionError

>>> assert null is None
>>> assert null is not None
Traceback (most recent call last):
    ...
AssertionError
```

**身份断言**提供了一种测试对象身份的方法。在这种情况下，断言表达式使用了恒等运算符， [`is`](https://docs.python.org/3/reference/expressions.html#is) 和 [`is not`](https://docs.python.org/3/reference/expressions.html#is-not) 。

最后，您将学习如何在断言的上下文中检查对象的数据类型:

>>>

```py
>>> # Type check assertions
>>> number = 42
>>> assert isinstance(number, int)

>>> number = 42.0
>>> assert isinstance(number, int)
Traceback (most recent call last):
    ...
AssertionError
```

**类型检查断言**通常涉及使用内置的 [`isinstance()`](https://docs.python.org/3/library/functions.html#isinstance) 函数来确保给定的对象是某个或某些类的实例。

尽管这些是 Python 代码中最常见的断言格式，但是还有许多其他的可能性。例如，您可以使用内置的 [`all()`](https://realpython.com/python-all/) 和 [`any()`](https://realpython.com/any-python/) 函数来编写检查 iterable 中项的真值的断言:

>>>

```py
>>> assert all([True, True, True])
>>> assert all([True, False, True])
Traceback (most recent call last):
    ...
AssertionError

>>> assert any([False, True, False])
>>> assert any([False, False, False])
Traceback (most recent call last):
    ...
AssertionError
```

`all()`断言检查输入 iterable 中的所有项是否为真，而`any()`示例检查输入 iterable 中的任何项是否为真。

你的想象力是编写有用断言的唯一限制。您可以使用谓词或[布尔值函数](https://en.wikipedia.org/wiki/Boolean-valued_function)、常规 Python 对象、比较表达式、布尔表达式或通用 Python 表达式来编写断言。您的断言将取决于您在给定时刻需要检查的具体条件。

现在您知道了一些可以在代码中使用的最常见的断言格式。是时候了解断言的具体用例了。在下一节中，您将学习如何使用断言来记录、调试和测试您的代码。

[*Remove ads*](/account/join/)

## 用断言记录你的代码

`assert`语句是记录代码的有效方法。例如，如果你想声明一个特定的`condition`在你的代码中应该总是为真，那么`assert condition`可能比注释或者文档字符串更好更有效，你马上就会知道。

为了理解为什么断言是一个方便的文档工具，假设您有一个函数，它接受一个服务器名和一组端口号。该函数将遍历试图连接到目标服务器的端口号。为了让您的功能正常工作，端口元组不应该为空:

```py
def get_response(server, ports=(443, 80)):
    # The ports argument expects a non-empty tuple
    for port in ports:
        if server.connect(port):
            return server.get()
    return None
```

如果有人不小心用空元组调用了`get_response()`，那么`for`循环永远不会运行，即使服务器可用，函数也会返回`None`。为了提醒程序员注意这个错误的调用，您可以使用注释，就像您在上面的例子中所做的那样。然而，使用`assert`语句可能更有效:

```py
def get_response(server, ports=(443, 80)):
    assert len(ports) > 0, f"ports expected a non-empty tuple, got {ports}"
    for port in ports:
        if server.connect(port):
            return server.get()
    return None
```

与注释相比，`assert`语句的优势在于，当条件不为真时，`assert`会立即引发一个`AssertionError`。之后，您的代码停止运行，因此它避免了异常行为，并直接将您指向特定的问题。

因此，在上述情况下使用断言是记录您的意图并避免由于意外错误或恶意行为者而难以发现的错误的有效而强大的方法。

## 用断言调试你的代码

从本质上来说，`assert`语句是一个调试助手，用于测试在代码正常执行期间应该保持正确的条件。对于作为调试工具的断言，您应该编写它们，以便失败表明您的代码中有 bug。

在本节中，您将学习如何使用`assert`语句来帮助您在开发时调试代码。

### 使用断言进行调试的示例

在开发过程中，通常会使用断言来调试代码。这个想法是为了确保特定的条件是真实的，并保持真实。如果一个断言的条件变为假，那么您立即知道您有一个 bug。

例如，假设您有下面的`Circle`类:

```py
# circle.py

import math

class Circle:
    def __init__(self, radius):
        if radius < 0:
            raise ValueError("positive radius expected")
        self.radius = radius

    def area(self):
        assert self.radius >= 0, "positive radius expected"
        return math.pi * self.radius ** 2
```

该类的初始化器 [`.__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__) ，将`radius`作为参数，并确保输入值是一个正数。此检查可防止圆的半径为负。

方法计算圆的面积。然而，在此之前，该方法使用一个`assert`语句来保证`.radius`保持为正数。你为什么要加这张支票？好吧，假设你在一个团队中工作，你的一个同事需要将下面的方法添加到`Circle`:

```py
class Circle:
    # ...

 def correct_radius(self, correction_coefficient):        self.radius *= correction_coefficient
```

该方法采用一个校正系数，并将其应用于`.radius`的当前值。然而，该方法不验证系数，引入了一个微妙的错误。你能发现它吗？假设用户无意中提供了一个负的修正系数:

>>>

```py
>>> from circle import Circle

>>> tire = Circle(42)
>>> tire.area()
5541.769440932395

>>> tire.correct_radius(-1.02)
>>> tire.radius
-42.84

>>> tire.area()
Traceback (most recent call last):
    ...
AssertionError: positive radius expected
```

对`.area()`的第一次调用工作正常，因为初始半径是正的。但是对`.area()`的第二次调用用一个`AssertionError`破坏了你的代码。为什么？发生这种情况是因为对`.correct_radius()`的调用将半径变成了负数，这暴露了一个错误:该函数没有正确检查有效输入。

在本例中，您的`assert`语句在半径可能取无效值的情况下充当看门狗。`AssertionError`立即指出了具体的问题:`.radius`意外地变成了负数。您必须弄清楚这种意外的变化是如何发生的，然后在投入生产之前修复您的代码。

[*Remove ads*](/account/join/)

### 关于使用断言进行调试的几点考虑

开发人员经常使用`assert`语句来陈述前提条件，就像你在上面的例子中所做的一样，其中`.area()`在进行任何计算之前检查有效的`.radius`。开发人员也使用断言来陈述后置条件。例如，在将值返回给调用者之前，您可以检查函数的**返回值**是否有效。

一般来说，你用`assert`语句检查的条件应该是真的，除非你或你团队中的另一个开发人员在代码中引入了一个 bug。换句话说，这些条件永远不应该是假的。他们的目的是在有人引入 bug 时快速标记。在这方面，断言是代码中的早期警报。这些警报在开发过程中很有用。

如果这些条件中的一个失败了，那么程序将崩溃并显示一个`AssertionError`，告诉你哪个条件没有成功。这种行为将帮助您更快地跟踪和修复错误。

为了正确地使用断言作为调试工具，您不应该使用`try` … `except`块来捕获和处理`AssertionError`异常。如果一个断言失败了，那么你的程序就会崩溃，因为一个假设为真的条件变成了假的。您不应该通过用`try` … `except`块捕捉异常来改变这种预期的行为。

断言的正确用法是通知开发人员程序中不可恢复的错误。断言不应该发出预期错误的信号，比如`FileNotFoundError`，用户可以采取纠正措施并重试。

断言的目标应该是揭露程序员的错误，而不是用户的错误。断言在开发过程中是有用的，而不是在生产过程中。当您发布代码时，它应该(大部分)没有错误，并且不应该要求断言正确工作。

最后，一旦您的代码准备好生产，您不必显式地删除断言。您可以禁用它们，您将在下一节了解到这一点。

## 禁用生产中的性能断言

现在假设你已经到了开发周期的末尾。您的代码已经过广泛的审查和测试。您的所有断言都通过了，并且您的代码已经准备好发布新版本了。此时，您可以通过禁用您在开发过程中添加的断言来优化用于生产的代码。为什么要这样优化代码呢？

断言在开发过程中非常有用，但是在生产中，它们会影响代码的性能。例如，一个包含许多始终运行的断言的代码库可能比没有断言的相同代码要慢。断言需要时间来运行，并且它们消耗内存，所以在生产中禁用它们是明智的。

现在，如何才能真正禁用您的断言呢？好吧，你有两个选择:

1.  使用`-O`或`-OO`选项运行 Python。
2.  将`PYTHONOPTIMIZE`环境变量设置为适当的值。

在这一节中，您将学习如何使用这两种技术来禁用您的断言。在此之前，您将了解内置的`__debug__`常量，这是 Python 用来禁用断言的内部机制。

### 理解`__debug__`内置常数

Python 有一个内置常数叫做 [`__debug__`](https://docs.python.org/3/library/constants.html#debug__) 。这个常数与`assert`语句密切相关。Python 的`__debug__`是一个布尔常量，默认为`True`。它是一个常量，因为一旦 Python 解释器运行，就不能更改它的值:

>>>

```py
>>> import builtins
>>> "__debug__" in dir(builtins)
True

>>> __debug__
True

>>> __debug__ = False
  File "<stdin>", line 1
SyntaxError: cannot assign to __debug__
```

在这个代码片段中，首先确认`__debug__`是一个内置的 Python，它总是对您可用。`True`是`__debug__`的默认值，一旦 Python 解释器运行，就没有办法改变这个值。

`__debug__`的值取决于 Python 运行的模式，普通模式还是优化模式:

| 方式 | `__debug__`的值 |
| --- | --- |
| 正常(或调试) | `True` |
| 最佳化的 | `False` |

普通模式通常是您在开发过程中使用的模式，而优化模式是您应该在生产中使用的模式。现在，`__debug__`和断言有什么关系？在 Python 中，`assert`语句相当于以下条件语句:

```py
if __debug__:
    if not expression:
        raise AssertionError(assertion_message)

# Equivalent to
assert expression, assertion_message
```

如果`__debug__`为真，那么运行外层`if`语句下的代码。内部的`if`语句检查`expression`是否为真，只有当表达式为[而非](https://realpython.com/python-not-operator/)真时，才会引发一个`AssertionError`。这是默认或正常的 Python 模式，在这种模式下，您的所有断言都被启用，因为`__debug__`是`True`。

另一方面，如果`__debug__`是`False`，那么外层`if`语句下的代码不会运行，这意味着您的断言将被禁用。在这种情况下，Python 运行在优化模式下。

正常或调试模式允许您在开发和测试代码时使用断言。一旦您当前的开发周期完成，您就可以切换到优化模式并禁用断言，以使您的代码为生产做好准备。

要激活优化模式并禁用您的断言，您可以使用 [`–O`](https://docs.python.org/3/using/cmdline.html#cmdoption-O) 或 [`-OO`](https://docs.python.org/3/using/cmdline.html#cmdoption-OO) 选项启动 Python 解释器，或者将系统变量 [`PYTHONOPTIMIZE`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONOPTIMIZE) 设置为适当的值。在接下来的几节中，您将学习如何进行这两种操作。

[*Remove ads*](/account/join/)

### 使用`-O`或`-OO`选项运行 Python

您可以通过将`__debug__`常量设置为`False`来禁用所有的`assert`语句。为了完成这项任务，您可以使用 Python 的`-O`或`-OO`命令行选项，在优化模式下运行解释器。

`-O`选项在内部将`__debug__`设置为`False`。这一更改删除了`assert`语句和您在条件目标`__debug__`下显式引入的任何代码。`-OO`选项的作用与`-O`相同，同样会丢弃文档字符串。

使用`-O`或`-OO`命令行选项运行 Python 会使编译后的[字节码](https://docs.python.org/3/glossary.html#term-bytecode)变小。此外，如果您有几个断言或`if __debug__:`条件，那么这些命令行选项也可以让您的代码更快。

现在，这种优化对您的断言有什么影响呢？它使他们失去能力。例如，在包含`circle.py`文件的目录中打开命令行或终端，用`python -O`命令运行一个交互式会话。在那里，运行下面的代码:

>>>

```py
>>> # Running Python in optimized mode
>>> __debug__
False

>>> from circle import Circle

>>> # Normal use of Circle
>>> ring = Circle(42)
>>> ring.correct_radius(1.02)
>>> ring.radius
42.84
>>> ring.area()
5765.656926346065

>>> # Invalid use of Circle
>>> ring = Circle(42)
>>> ring.correct_radius(-1.02)
>>> ring.radius
-42.84
>>> ring.area()
5765.656926346065
```

因为`-O`选项通过将`__debug__`设置为`False`来禁用您的断言，所以您的`Circle`类现在接受负半径，如最后一个示例所示。这种行为是完全错误的，因为你不能有一个半径为负的圆。另外，使用错误的半径作为输入来计算圆的面积。

在优化模式下禁用断言的可能性是为什么不能使用`assert`语句来验证输入数据，而是作为调试和测试过程的辅助手段的主要原因。

**注意:**断言通常在生产代码中被关闭，以避免它们可能导致的任何开销或副作用。

对于`Circle`类的一个 Pythonic 解决方案是使用 [`@property`](https://realpython.com/python-property/) 装饰器将`.radius`属性转换为**托管属性**。这样，每次属性改变时，您都执行`.radius`验证:

```py
# circle.py

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("positive radius expected")
        self._radius = value

    def area(self):
        return math.pi * self.radius ** 2

    def correct_radius(self, correction_coefficient):
        self.radius *= correction_coefficient
```

现在，`.radius`是一个托管属性，使用`@property`装饰器提供 setter 和 getter 方法。您已经将验证代码从`.__init__()`移到了 setter 方法中，每当类改变`.radius`的值时都会调用该方法。

现在，如果您在优化模式下运行代码，更新后的`Circle`将按预期工作:

>>>

```py
>>> # Running Python in optimized mode
>>> __debug__
False

>>> from circle import Circle

>>> # Normal use of Circle
>>> ring = Circle(42)
>>> ring.correct_radius(1.02)
>>> ring.radius
42.84
>>> ring.area()
5765.656926346065

>>> # Invalid use of Circle
>>> ring = Circle(42)
>>> ring.correct_radius(-1.02)
Traceback (most recent call last):
    ...
ValueError: positive radius expected
```

`Circle`总是在赋值前验证`.radius`的值，你的类工作正常，为`.radius`的负值产生一个`ValueError`。就是这样！您已经用一个优雅的解决方案修复了这个 bug。

在优化模式下运行 Python 的一个有趣的副作用是，显式`if __debug__:`条件下的代码也被禁用。考虑以下脚本:

```py
# demo.py

print(f"{__debug__ = }")

if __debug__:
    print("Running in Normal mode!")
else:
    print("Running in Optimized mode!")
```

这个脚本显式地检查`if` … `else`语句中`__debug__`的值。只有当`__debug__`为`True`时，`if`代码块中的代码才会运行。相反，如果`__debug__`是`False`，那么`else`块中的代码将运行。

现在尝试在正常和优化模式下运行脚本来检查它在每种模式下的行为:

```py
$ python demo.py
__debug__ = True
Running in Normal mode!

$ python -O demo.py
__debug__ = False
Running in Optimized mode!
```

当您在正常模式下执行脚本时，`if __debug__:`条件下的代码会运行，因为在这种模式下`__debug__`是`True`。另一方面，当您使用`-O`选项在优化模式下执行脚本时，`__debug__`变为`False`，并且运行`else`块下的代码。

Python 的`-O`命令行选项从最终编译的字节码中删除断言。Python 的`-OO`选项执行与`-O`相同的优化，除了从字节码中移除文档字符串。

因为两个选项都将`__debug__`设置为`False`，所以任何显式`if __debug__:`条件下的代码也会停止工作。这种行为提供了一种强大的机制，可以在 Python 项目的开发阶段引入仅用于调试的代码。

现在您知道了使用 Python 的`-O`和`-OO`选项在生产代码中禁用断言的基本知识。然而，每次需要运行生产代码时都使用这些选项运行 Python 似乎是重复的，并且可能容易出错。为了自动化这个过程，您可以使用`PYTHONOPTIMIZE`环境变量。

[*Remove ads*](/account/join/)

### 设置`PYTHONOPTIMIZE`环境变量

您还可以通过将`PYTHONOPTIMIZE`环境变量设置为适当的值，在禁用断言的优化模式下运行 Python。例如，将该变量设置为非空字符串相当于使用`-O`选项运行 Python。

要尝试`PYTHONOPTIMIZE`,启动您的命令行并运行以下命令:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
C:\> set PYTHONOPTIMIZE="1"
```

```py
$ export PYTHONOPTIMIZE="1"
```

一旦将`PYTHONOPTIMIZE`设置为非空字符串，就可以用基本的`python`命令启动 Python 解释器。该命令将自动在优化模式下运行 Python。

现在继续从包含您的`circle.py`文件的目录中运行下面的代码:

>>>

```py
>>> from circle import Circle

>>> # Normal use of Circle
>>> ring = Circle(42)
>>> ring.correct_radius(1.02)
>>> ring.radius
42.84

>>> # Invalid use of Circle
>>> ring = Circle(42)
>>> ring.correct_radius(-1.02)
>>> ring.radius
-42.84
```

同样，您的断言是关闭的，`Circle`类接受负半径值。您再次在优化模式下运行 Python。

另一种可能是将`PYTHONOPTIMIZE`设置为一个整数值`n`，这相当于使用`-O`选项`n`次运行 Python。换句话说，你正在使用`n` **级的优化**:

*   [*视窗*](#windows-2)
**   [**Linux + macOS**](#linux-macos-2)*

```py
C:\> set PYTHONOPTIMIZE=1  # Equivalent to python -O

C:\> set PYTHONOPTIMIZE=2  # Equivalent to python -OO
```

```py
$ export PYTHONOPTIMIZE=1  # Equivalent to python -O

$ export PYTHONOPTIMIZE=2  # Equivalent to python -OO
```

可以用任意整数来设置`PYTHONOPTIMIZE`。然而，Python 只实现了两个级别的优化。使用大于`2`的数字对编译后的字节码没有实际影响。此外，将`PYTHONOPTIMIZE`设置为`0`将导致解释器以正常模式运行。

### 在优化模式下运行 Python

当您运行 Python 时，解释器会将任何导入的模块动态编译成字节码。编译后的字节码将位于一个名为`__pycache__/`的目录中，该目录位于包含提供导入代码的模块的目录中。

在`__pycache__/`中，您会发现一个`.pyc`文件，该文件以您的原始模块加上解释器的名称和版本命名。`.pyc`文件的名称还将包括用于编译代码的优化级别。

例如，当您从`circle.py`导入代码时， [Python 3.10](https://realpython.com/python310-new-features/) 解释器会根据优化级别生成以下文件:

| 文件名 | 命令 | `PYTHONOPTIMIZE` |
| --- | --- | --- |
| `circle.cpython-310.pyc` | `python circle.py` | `0` |
| `circle.cpython-310.opt-1.pyc` | `python -O circle.py` | `1` |
| `circle.cpython-310.opt-2.pyc` | `python -OO circle.py` | `2` |

该表中每个文件的名称包括原始模块的名称(`circle`)、生成代码的解释器(`cpython-310`)和优化级别(`opt-x`)。该表还总结了`PYTHONOPTIMIZE`变量的相应命令和值。 [PEP 488](https://www.python.org/dev/peps/pep-0488/) 提供了更多关于`.pyc`文件命名格式的上下文。

在第一级优化中运行 Python 的主要结果是解释器将`__debug__`设置为`False`，并从最终编译的字节码中删除断言。这些优化使得代码比在正常模式下运行的相同代码更小，并且可能更快。

第二级优化的作用与第一级相同。它还从编译后的代码中删除了所有的文档字符串，从而产生了更小的编译后的字节码。

[*Remove ads*](/account/join/)

## 用断言测试你的代码

测试是开发过程中断言有用的另一个领域。测试归结为将观察值与期望值进行比较，以检查它们是否相等。这种检查非常适合断言。

断言必须检查通常应该为真的条件，除非您的代码中有 bug。这个想法是测试背后的另一个重要概念。

[`pytest`](https://realpython.com/pytest-python-testing/) 第三方库是 Python 中流行的测试框架。在其核心，你会发现`assert`语句，你可以用它在`pytest`中编写大多数测试用例。

这里有几个使用`assert`语句编写测试用例的例子。下面的例子利用了一些提供测试材料的内置函数:

```py
# test_samples.py

def test_sum():
    assert sum([1, 2, 3]) == 6

def test_len():
    assert len([1, 2, 3]) > 0

def test_reversed():
    assert list(reversed([1, 2, 3])) == [3, 2, 1]

def test_membership():
    assert 3 in [1, 2, 3]

def test_isinstance():
    assert isinstance([1, 2, 3], list)

def test_all():
    assert all([True, True, True])

def test_any():
    assert any([False, True, False])

def test_always_fail():
    assert pow(10, 2) == 42
```

所有这些测试用例都使用了`assert`语句。它们中的大多数都是使用您以前学过的断言格式编写的。它们都展示了如何用`pytest`编写真实世界的测试用例来检查代码的不同部分。

现在，为什么`pytest`更喜欢测试用例中的普通`assert`语句，而不是定制的 [API](https://en.wikipedia.org/wiki/API) ，这也是其他测试框架更喜欢的？这一选择背后有几个显著的优势:

*   `assert`语句允许`pytest`降低入门门槛，并在一定程度上拉平学习曲线，因为它的用户可以利用他们已经知道的 Python 语法。
*   `pytest`的用户不需要从库中导入任何东西来开始编写测试用例。如果他们的测试用例变得复杂，需要更高级的特性，他们只需要开始导入东西。

这些优势使得使用`pytest`对于初学者和来自其他使用定制 API 的测试框架的人来说是一种愉快的体验。

例如，标准库 [`unittest`](https://docs.python.org/3/library/unittest.html) 模块提供了一个由一系列 [`.assert*()`方法](https://docs.python.org/3/library/unittest.html#assert-methods)组成的 API，其工作方式非常类似于`assert`语句。对于刚开始使用框架的开发人员来说，这种 API 可能很难学习和记忆。

您可以使用`pytest`来运行上面所有的测试用例。首先，您需要通过发出`python -m pip install pytest`命令来安装库。然后你可以从命令行执行`pytest test_samples.py`。后一个命令将显示类似如下的输出:

```py
========================== test session starts =========================
platform linux -- Python 3.10.0, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
rootdir: /home/user/python-assert
collected 8 items 
test_samples.py .......F                                           [100%] 
========================== FAILURES =====================================
__________________________ test_always_fail _____________________________

    def test_always_fail():
>       assert pow(10, 2) == 42
E       assert 100 == 42 E        +  where 100 = pow(10, 2) 
test_samples.py:25: AssertionError
========================== short test summary info ======================
FAILED test_samples.py::test_always_fail - assert 100 == 42
========================== 1 failed, 7 passed in 0.21s ==================
```

输出中第一个突出显示的行告诉您`pytest`发现并运行了八个测试用例。第二个突出显示的行显示八个测试中有七个成功通过。这就是为什么你会得到七个绿点和一个`F`。

**注意:**为了避免`pytest`的问题，您必须在正常模式下运行您的 Python 解释器。请记住，优化模式禁用断言。因此，确保您不是在优化模式下运行 Python。

您可以通过运行以下命令来检查您的`PYTHOPTIMIZE`环境变量的当前值:

*   [*视窗*](#windows-3)
**   [**Linux + macOS**](#linux-macos-3)*

```py
C:\> echo %PYTHONOPTIMIZE%
```

```py
$ echo $PYTHONOPTIMIZE
```

如果`PYTHONOPTIMIZE`被设置，则该命令的输出将显示其当前值。***  ***值得注意的一个显著特点是`pytest`与`assert`语句很好地集成在一起。该库可以显示错误报告，其中包含关于失败断言的详细信息以及它们失败的原因。例如，查看上面输出中以`E`字母开头的行。它们显示错误消息。

这些行清楚地揭示了失败的根本原因。在这个例子中，`pow(10, 2)`返回的是`100`而不是`42`，这是故意错误的。您可以使用 [`pytest.raises()`](https://docs.pytest.org/en/latest/how-to/assert.html#assertions-about-expected-exceptions) 来处理[预计会失败](https://realpython.com/python-testing/#handling-expected-failures)的代码。

## 了解`assert` 的常见陷阱

尽管断言是如此伟大和有用的工具，它们也有一些缺点。像任何其他工具一样，断言可能会被误用。您已经了解到，在开发过程中，您应该主要将断言用于调试和测试代码。相比之下，您不应该依赖断言在生产代码中提供功能，这是断言陷阱的主要驱动因素之一。

特别是，如果您在以下方面使用断言，您可能会陷入困境:

*   处理和验证数据
*   处理错误
*   运行有副作用的操作

断言的另一个常见问题是，在生产中保持它们的启用会对代码的性能产生负面影响。

最后，Python 默认启用断言，这可能会让来自其他语言的开发人员感到困惑。在接下来的部分中，您将了解所有这些可能的断言陷阱。您还将学习如何在自己的 Python 代码中避免它们。

[*Remove ads*](/account/join/)

### 使用`assert`进行数据处理和验证

您不应该使用`assert`语句来验证用户的输入或来自外部来源的任何其他输入数据。这是因为生产代码通常会禁用断言，这将删除所有的验证。

例如，假设您正在用 Python 构建一个在线商店，您需要添加接受折扣券的功能。您最终编写了以下函数:

```py
# store.py

# Code under development
def price_with_discount(product, discount):
    assert 0 < discount < 1, "discount expects a value between 0 and 1"
    new_price = int(product["price"] * (1 - discount))
    return new_price
```

注意到`price_with_discount()`第一行的`assert`语句了吗？它可以保证折扣价不会等于或低于零美元。这一断言还确保了新价格不会高于产品的原价。

现在考虑一双鞋打八五折的例子:

>>>

```py
>>> from store import price_with_discount

>>> shoes = {"name": "Fancy Shoes", "price": 14900}

>>> # 25% off -> $111.75
>>> price_with_discount(shoes, 0.25)
11175
```

好吧，`price_with_discount()`工作得很好！它将产品作为一个[字典](https://realpython.com/python-dicts/)，将预期折扣应用于当前价格，并返回新价格。现在，尝试应用一些无效折扣:

>>>

```py
>>> # 200% off
>>> price_with_discount(shoes, 2.0)
Traceback (most recent call last):
    ...
AssertionError: discount expects a value between 0 and 1

>>> # 100% off
>>> price_with_discount(shoes, 1)
Traceback (most recent call last):
    ...
AssertionError: discount expects a value between 0 and 1
```

应用无效折扣会引发一个指出违反条件的`AssertionError`。如果你在开发和测试你的网上商店时遇到过这个错误，那么通过查看[回溯](https://realpython.com/python-traceback/)应该不难发现发生了什么。

如果最终用户可以用禁用的断言在生产代码中直接调用`price_with_discount()`,那么上面例子的真正问题就来了。在这种情况下，该函数不会检查`discount`的输入值，可能会接受错误的值并破坏折扣功能的正确性。

一般来说，您可以在开发过程中编写`assert`语句来处理、验证或检验数据。然而，如果这些操作在生产代码中仍然有效，那么一定要用一个`if`语句或者一个`try` … `except`块来替换它们。

这里有一个新版本的`price_with_discount()`,它使用了条件而不是断言:

```py
# store.py

# Code in production
def price_with_discount(product, discount):
    if 0 < discount < 1:
        new_price = int(product["price"] * (1 - discount))
        return new_price
    raise ValueError("discount expects a value between 0 and 1")
```

在这个新的`price_with_discount()`实现中，您用一个显式条件语句替换了`assert`语句。现在，只有当输入值在`0`和`1`之间时，该函数才会应用折扣。否则，就会出现一个`ValueError`，发出问题信号。

现在，您可以将对该函数的任何调用封装在一个`try` … `except`块中，该块捕获`ValueError`，并向用户发送一条信息性消息，以便他们可以相应地采取行动。

这个例子的寓意是，您不应该依赖于`assert`语句进行数据处理或数据验证，因为这个语句在生产代码中通常是关闭的。

### 用`assert`和处理错误

断言的另一个重要缺陷是，有时开发人员将断言用作一种快速的错误处理方式。因此，如果产品代码删除了断言，那么重要的错误检查也会从代码中删除。因此，请记住，断言不能代替良好的错误处理。

下面是一个使用断言进行错误处理的例子:

```py
# Bad practice
def square(x):
    assert x >= 0, "only positive numbers are allowed"
    return x ** 2

try:
    square(-2)
except AssertionError as error:
    print(error)
```

如果在生产环境中使用禁用的断言执行这段代码，那么`square()`将永远不会运行`assert`语句并引发`AssertionError`。在这种情况下，`try` … `except`块是多余的，不起作用。

你能做些什么来修正这个例子呢？尝试更新`square()`以使用`if`语句和`ValueError`:

```py
# Best practice
def square(x):
    if x < 0:
        raise ValueError("only positive numbers are allowed")
    return x ** 2

try:
    square(-2)
except ValueError as error:
    print(error)
```

现在`square()`通过使用一个显式的`if`语句来处理这种情况，该语句不能在产品代码中被禁用。您的`try` … `except`块现在处理一个`ValueError`，这在本例中是一个更合适的异常。

永远不要在你的代码中捕获`AssertionError`异常，因为那会压制失败的断言，这是误用断言的明显标志。相反，捕捉与您正在处理的错误明显相关的具体异常，并让您的断言失败。

除非有 bug，否则只使用断言来检查在程序的正常执行过程中不应该发生的错误。请记住，断言可以被禁用。

[*Remove ads*](/account/join/)

### 对带有副作用的表达式运行`assert`

当您使用该语句检查具有某种副作用的操作、函数或表达式时，`assert`语句会出现另一个微妙的陷阱。换句话说，这些操作修改了操作的[范围](https://realpython.com/python-scope-legb-rule/)之外的对象的[状态](https://en.wikipedia.org/wiki/State_(computer_science))。

在这些情况下，副作用会在每次代码运行断言时发生，这可能会悄悄地改变程序的全局状态和行为。

考虑下面的玩具例子，其中一个函数修改了一个全局变量的值作为副作用:

>>>

```py
>>> sample = [42, 27, 40, 38]

>>> def popped(sample, index=-1):
...     item = sample.pop(index)
...     return item
...

>>> assert sample[-1] == popped(sample)
>>> assert sample[1] == popped(sample, 1)

>>> sample
[42, 40]
```

在这个例子中，`popped()`在数据的输入`sample`中给定的`index`处返回`item`，其副作用是也删除了所述的`item`。

使用断言来确保您的函数返回正确的项似乎是合适的。然而，这将导致函数的内部副作用在每个断言中运行，修改`sample`的原始内容。

为了防止类似上面例子中的意外行为，请使用不会产生副作用的断言表达式。例如，您可以使用[纯函数](https://en.wikipedia.org/wiki/Pure_function)，它只接受输入参数并返回相应的输出，而不修改来自其他作用域和[名称空间](https://realpython.com/python-namespaces-scope/)的对象的状态。

### 用`assert` 影响性能

生产中过多的断言会影响代码的性能。当断言的条件涉及太多逻辑时，这个问题就变得很关键，比如长复合条件、长时间运行的谓词函数，以及隐含着高成本实例化过程的[类](https://realpython.com/python3-object-oriented-programming/#define-a-class-in-python)。

断言可以从两个主要方面影响代码的性能。他们将:

1.  花费**时间**执行
2.  使用额外的**内存**

检查`None`值的`assert`语句可能相对便宜。然而，更复杂的断言，尤其是那些运行大量代码的断言，会明显降低代码的速度。断言也消耗内存来存储它们自己的代码和任何需要的数据。

为了避免生产代码中的性能问题，您应该使用 Python 的`-O`或`-OO`命令行选项，或者根据您的需要设置`PYTHONOPTIMIZE`环境变量。这两种策略都会通过生成无断言编译的字节码来优化代码，这样运行起来更快，占用的内存也更少。

此外，为了防止开发过程中的性能问题，您的断言应该相当简明扼要。

### 默认启用`assert`条语句

在 Python 中，默认情况下启用断言。当解释器在正常模式下运行时，`__debug__`变量是`True`，您的断言被启用。这种行为是有意义的，因为您通常在正常模式下开发、调试和测试代码。

如果您想要禁用您的断言，那么您需要显式地这样做。您可以使用`-o`或`-OO`选项运行 Python 解释器，或者将`PYTHONOPTIMIZE`环境变量设置为适当的值。

相比之下，其他编程语言默认禁用断言。例如，如果您从 Java 进入 Python，您可能会认为您的断言不会运行，除非您显式地打开它们。对于 Python 初学者来说，这种假设可能是常见的困惑来源，所以请记住这一点。

## 结论

现在你知道了如何使用 Python 的`assert`语句在整个代码中设置[健全性检查](https://en.wikipedia.org/wiki/Sanity_check)，并确保某些条件为真并保持不变。当这些条件中的任何一个失败时，你就清楚地知道发生了什么。这样，您可以快速调试和修复代码。

在开发阶段，当您需要**记录**、**调试**、**测试**您的代码时，`assert`语句非常方便。在本教程中，您学习了如何在代码中使用断言，以及它们如何使您的调试和测试过程更加高效和简单。

**在本教程中，您学习了:**

*   什么是断言以及何时使用它们
*   Python 的 **`assert`语句**如何工作
*   `assert`对于**记录**、**调试**、**测试**代码是多么方便
*   如何禁用断言以提高生产中的性能
*   使用`assert`语句时，你会面临哪些**常见陷阱**

有了这些关于`assert`语句的知识，您现在可以编写健壮、可靠且错误较少的代码，这将使您成为更高水平的开发人员。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。******************