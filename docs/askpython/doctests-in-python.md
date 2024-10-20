# Python 中的文档测试——概述

> 原文：<https://www.askpython.com/python-modules/doctests-in-python>

让我们了解一下 Python 中一种特殊的测试和调试机制。Python 中的文档测试是函数的测试用例，它们可以用来验证函数是否按预期工作。

## Python 中的 docstrings 是什么？

在我们继续进行文档测试之前，我们需要了解一下[文档字符串](https://www.askpython.com/python/python-docstring)。

*   文档字符串是包含在三重引号中的可选字符串，它是在声明函数时首先写入的。
*   文档字符串用于描述一个函数。我们可以写一个函数做什么，它是如何工作的，它接受的参数数量，它返回的对象类型，等等。

所有这些都向程序员描述了函数的用途，程序员可以使用`__doc__`属性访问函数的 docstring。

让我们举一个打印数字阶乘的函数的例子..

```py
def factorial(num):
    """
    A function that returns the factorial of a given number.
    No. of arguments: 1, Integer
    Returns: Integer
    """
    res = 1
    for i in range(1, num+1):
        res *= i
    print(res)

```

正如你所看到的，在声明函数之后，在做任何事情之前，我们写了一个用三重引号括起来的字符串来描述函数。
这将使该字符串成为该函数的文档，访问属性`__doc__`将返回该字符串。让我们现在做那件事。

```py
print(factorial.__doc__)

```

**输出:**

```py
 A function that returns the factorial of a given number.
    No. of arguments: 1, Integer
    Returns: Integer 
```

现在我们清楚了什么是 docstring，我们可以继续进行 doctests 了。

## Python 中的 doctests 是什么？

正如我们前面讨论的，Python 中的 doctests 是在 docstring 内部编写的测试用例。在我们的例子中，5 的阶乘是 120，因此调用`factorial(5)`将打印`120`，同样，调用`factorial(0)`将打印`1`。

这些可以是我们可以验证函数的测试用例，为此，我们在 docstring 中使用如下语法描述它们:

```py
def factorial(num):
    """
    A function that returns the factorial of a given number.
    No. of arguments: 1, Integer
    Returns: Integer

    >>> factorial(5)
    120

    >>> factorial(0)
    1
    """
    res = 1
    for i in range(1, num+1):
        res *= i
    print(res)

```

如果您还记得 Python shell，我们将所有代码都写在 shell 中的三个尖括号(`>>>`)之后，当我们按 enter 键时，代码会立即执行。

因此，如果我们通过 Python shell 调用`factorial(5)`,它看起来就像我们在上面的 docstring 中编写的一样。

在 docstring 中指定这一点告诉 Python，上面几行是在 shell 中运行`factorial(5)`后的预期输出。

类似地，下面我们写了`factorial(0)`的确切预期输出。

注意，doctests 对空格和制表符很敏感，所以我们需要准确地写出我们想要的结果。

我们还可以指定函数由于错误输入而可能返回的异常和错误。

现在我们已经在函数中编写了一些 doctests，让我们使用它们并检查函数是否正常工作。

### Python 中成功的文档测试

```py
import doctest
doctest.testmod(name='factorial', verbose=True)

```

这就是我们在 Python 中使用 doctests 的方式。我们导入一个名为`doctest`的模块，并使用它的`testmod`函数，如图所示。

**输出将如下所示:**

```py
Trying:
    factorial(5)
Expecting:
    120
ok
Trying:
    factorial(0)
Expecting:
    1
ok
1 items had no tests:
    factorial
1 items passed all tests:
   2 tests in factorial.factorial
2 tests in 2 items.
2 passed and 0 failed.
Test passed.
TestResults(failed=0, attempted=2)

```

如您所见，它将运行每个测试用例，并检查实际输出是否与预期输出相匹配。最后，它将打印测试结果，程序员将能够分析该函数的执行情况。

如果任何一个测试用例失败，它将在预期的输出之后打印出准确的输出，并指定最后失败的测试用例的数量。

### Python 中失败的文档测试

让我们用 Python 做一些我们知道会失败的文档测试:

```py
def factorial(num):
    """
    A function that returns the factorial of a given number.
    No. of arguments: 1, Integer
    Returns: Integer

    >>> factorial(5)
    120

    >>> factorial(0)
    1

    >>> factorial(2)
    Two
    """
    res = 1
    for i in range(1, num+1):
        res *= i
    print(res)

import doctest
doctest.testmod(name='factorial', verbose=True)

```

在第三个 doctest 中，发送`2`永远不会打印`Two`，所以让我们看看输出:

```py
Trying:
    factorial(5)
Expecting:
    120
ok
Trying:
    factorial(0)
Expecting:
    1
ok
Trying:
    factorial(2)
Expecting:
    Two
**********************************************************************
File "__main__", line 13, in factorial.factorial
Failed example:
    factorial(2)
Expected:
    Two
Got:
    2
1 items had no tests:
    factorial
**********************************************************************
1 items had failures:
   1 of   3 in factorial.factorial
3 tests in 2 items.
2 passed and 1 failed.
***Test Failed*** 1 failures.
TestResults(failed=1, attempted=3)

```

对于第三个测试用例，它失败了，模块准确地显示了它是如何失败的，最后，我们看到尝试了三个测试用例，一个失败了。

## Python 中 Doctests 的使用？

Python 中的 Doctests 是为了在创建函数时考虑预期的输出而使用的。

如果你需要一个函数在调用时准确地输出一些内容，那么你可以在 doctest 中指定它，最后，doctest 模块将允许你一次运行所有的测试用例，你将能够看到这个函数是如何执行的。

提到的测试用例应该正是你所期望的，如果其中任何一个失败了，那就表明函数中有一个 bug 应该被纠正。

成品的文件测试必须总是成功的。

虽然我们不能编写每一个测试用例，但是在一个大项目中，编写那些可能由于意外输入而失败的测试用例是一个好主意，比如 0，9999999，-1，或者“banana”。

## 结论

在本教程中，我们学习了 Python 中的 doctests 是什么，如何编写，如何使用，以及何时使用。

我们讨论了 doctests 如何成为程序员的测试机制，以及它如何使编写测试用例变得容易。

我希望你学到了一些东西，并在另一个教程中看到你。