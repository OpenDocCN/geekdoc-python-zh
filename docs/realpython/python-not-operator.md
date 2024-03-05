# 在 Python 中使用“非”布尔运算符

> 原文：<https://realpython.com/python-not-operator/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和编写的教程一起看，加深你的理解: [**使用 Python not 运算符**](/courses/using-not-operator/)

Python 的 **`not`** 运算符允许你反转布尔表达式和对象的**真值**。您可以在布尔上下文中使用这个操作符，比如`if`语句和`while`循环。它也可以在非布尔上下文中工作，这允许您反转变量的真值。

有效地使用`not`操作符将帮助你写出精确的负布尔表达式来控制程序中的执行流程。

在本教程中，您将学习:

*   Python 的 **`not`** 运算符是如何工作的
*   如何在**布尔**和**非布尔**上下文中使用`not`运算符
*   如何使用 **`operator.not_()`** 函数执行逻辑否定
*   如何以及何时避免代码中不必要的**负逻辑**

您还将编写一些实际的例子，让您更好地理解`not`操作符的一些主要用例，以及围绕它的使用的最佳实践。为了从本教程中获得最大收益，您应该对[布尔](https://realpython.com/python-boolean/)逻辑、[条件语句](https://realpython.com/python-conditional-statements/)和[循环](https://realpython.com/python-while-loop/)有所了解。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 在 Python 中使用布尔逻辑

[乔治·布尔](https://en.wikipedia.org/wiki/George_Boole)将现在所知的[布尔代数](https://en.wikipedia.org/wiki/Boolean_algebra)放在一起，它依赖于**真**和**假**值。它还定义了一组布尔运算: [`AND`](https://en.wikipedia.org/wiki/Logical_conjunction) ， [`OR`](https://en.wikipedia.org/wiki/Logical_disjunction) ， [`NOT`](https://en.wikipedia.org/wiki/Negation) 。这些布尔值和运算符在编程中很有帮助，因为它们可以帮助您决定程序中的操作过程。

在 Python 中，[布尔类型](https://realpython.com/python-boolean/)、[、`bool`、](https://docs.python.org/3/library/functions.html#bool)，是[、`int`、](https://docs.python.org/3/library/functions.html#int)的子类:

>>>

```py
>>> issubclass(bool, int)
True
>>> help(bool)
Help on class bool in module builtins:

class bool(int)
 bool(x) -> bool
 ...
```

这个类型有两个可能的值，`True`和`False`，是 Python 中的[内置常量](https://docs.python.org/3/library/constants.html#built-in-constants)，必须大写。在内部，Python 将它们实现为整数:

>>>

```py
>>> type(True)
<class 'bool'>
>>> type(False)
<class 'bool'>

>>> isinstance(True, int)
True
>>> isinstance(False, int)
True

>>> int(True)
1
>>> int(False)
0
```

Python 内部将其布尔值实现为`True`的`1`和`False`的`0`。继续在您的[交互式](https://realpython.com/interacting-with-python/) shell 中执行`True + True`，看看会发生什么。

Python 提供了三种布尔或逻辑运算符:

| 操作员 | 逻辑运算 |
| --- | --- |
| [T2`and`](https://realpython.com/python-and-operator/) | 结合 |
| [T2`or`](https://realpython.com/python-or-operator/) | 分离 |
| [T2`not`](https://realpython.com/python-keywords/#the-not-keyword) | 否认 |

使用这些操作符，您可以通过将[布尔表达式](https://realpython.com/python-operators-expressions/#logical-expressions-involving-boolean-operands)相互连接、将对象相互连接，甚至将布尔表达式与对象连接来构建表达式。Python 使用英语单词作为布尔运算符。这些单词是该语言的**关键词**，所以你不能在不导致[语法错误](https://realpython.com/invalid-syntax-python/)的情况下将它们用作[标识符](https://docs.python.org/3/reference/lexical_analysis.html#identifiers)。

在本教程中，你将学习 Python 的`not`操作符，它实现了逻辑`NOT`操作或**否定**。

[*Remove ads*](/account/join/)

## Python 的`not`操作符入门

`not`运算符是在 Python 中实现求反的布尔或逻辑运算符。它是[一元](https://en.wikipedia.org/wiki/Unary_operation)，这意味着它只需要一个**操作数**。操作数可以是一个**布尔表达式**或者任何 Python **对象**。甚至用户定义的对象也可以。`not`的任务是反转其操作数的**真值**。

如果你将`not`应用于一个计算结果为`True`的操作数，那么你将得到`False`。如果你将`not`应用于一个假操作数，那么你会得到`True`:

>>>

```py
>>> not True
False

>>> not False
True
```

`not`运算符对其操作数的真值求反。真操作数返回`False`。假操作数返回`True`。这两种说法揭示了通常所说的`not`的**真值表**:

| `operand` | `not operand` |
| --- | --- |
| `True` | `False` |
| `False` | `True` |

使用`not`，你可以否定任何布尔表达式或对象的真值。这一功能在以下几种情况下很有价值:

*   在`if`语句和`while`循环的背景下检查**未满足的条件**
*   **反转一个对象或表达式的真值**
*   检查**值是否不在给定的容器**中
*   检查**对象的身份**

在本教程中，您将找到涵盖所有这些用例的示例。首先，您将从学习`not`操作符如何处理布尔表达式以及常见的 Python 对象开始。

布尔表达式总是返回布尔值。在 Python 中，这种表达式返回`True`或`False`。假设您想检查一个给定的数字[变量](https://realpython.com/python-variables/)是否大于另一个变量:

>>>

```py
>>> x = 2
>>> y = 5

>>> x > y
False

>>> not x > y
True
```

表达式`x > y`总是返回`False`，所以你可以说它是一个布尔表达式。如果你把`not`放在这个表达式前面，那么你会得到相反的结果，`True`。

**注:** Python 按照严格的顺序对运算符求值，俗称[运算符优先级](https://docs.python.org/3/reference/expressions.html#operator-precedence)。

例如，Python 首先计算数学和比较运算符。然后它计算逻辑运算符，包括`not`:

>>>

```py
>>> not True == False
True

>>> False == not True
File "<stdin>", line 1
 False == not True
 ^
SyntaxError: invalid syntax

>>> False == (not True)
True
```

在第一个例子中，Python 对表达式`True == False`求值，然后通过对`not`求值来否定结果。

在第二个例子中，Python 首先计算等式运算符(`==`)并引发一个`SyntaxError`，因为没有办法比较`False`和`not`。您可以用括号(`()`)将表达式`not True`括起来来解决这个问题。这个快速更新告诉 Python 首先计算带括号的表达式。

在逻辑运算符中，`not`的优先级高于具有相同优先级的`and`运算符和`or`运算符。

您还可以将`not`用于常见的 Python 对象，例如[数字](https://realpython.com/python-numbers/)、[字符串](https://realpython.com/python-strings/)、[列表、元组](https://realpython.com/python-lists-tuples/)、[字典](https://realpython.com/python-dicts/)、[集合](https://realpython.com/python-sets/)，用户定义的对象等等:

>>>

```py
>>> # Use "not" with numeric values
>>> not 0
True
>>> not 42
False
>>> not 0.0
True
>>> not 42.0
False
>>> not complex(0, 0)
True
>>> not complex(42, 1)
False

>>> # Use "not" with strings
>>> not ""
True
>>> not "Hello"
False

>>> # Use "not" with other data types
>>> not []
True
>>> not [1, 2, 3]
False
>>> not {}
True
>>> not {"one": 1, "two": 2}
False
```

在每个示例中，`not`对其操作数的真值求反。为了确定一个给定的对象是真还是假，Python 使用 [`bool()`](https://docs.python.org/3/library/functions.html#bool) ，根据手头对象的真值返回`True`或`False`。

这个内置函数在内部使用以下规则来计算其输入的真实值:

> 默认情况下，除非对象的类定义了返回`False`的 [`__bool__()`](https://docs.python.org/3/reference/datamodel.html#object.__bool__) 方法或返回零的 [`__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 方法，否则对象被视为真。以下是大多数被认为是假的内置对象:
> 
> *   定义为假的常数:`None`和`False`。
> *   任意数值类型的零:`0`、`0.0`、`0j`、`Decimal(0)`、`Fraction(0, 1)`
> *   空序列和集合:`''`、`()`、`[]`、`{}`、`set()`、`range(0)`
> 
> ([来源](https://docs.python.org/3/library/stdtypes.html#truth-value-testing))

一旦`not`知道了其操作数的真值，就返回相反的布尔值。如果对象评估为`True`，那么`not`返回`False`。否则，它返回`True`。

**注意:**总是返回`True`或`False`是`not`与另外两个布尔运算符`and`运算符和`or`运算符的重要区别。

`and`操作符和`or`操作符返回表达式中的一个操作数，而`not`操作符总是返回一个布尔值:

>>>

```py
>>> 0 and 42
0
>>> True and False
False
>>> True and 42 > 27
True

>>> 0 or 42
42
>>> True or False
True
>>> False or 42 < 27
False

>>> not 0
True
>>> not 42
False
>>> not True
False
```

使用`and`操作符和`or`操作符，当这些值中的一个显式地来自对操作数的求值时，可以从表达式中得到`True`或`False`。否则，您会得到表达式中的一个操作数。另一方面，`not`的行为有所不同，不管它采用什么操作数，都返回`True`或`False`。

为了表现得像`and`操作符和`or`操作符一样，`not`操作符必须创建并返回新的对象，这通常是不明确的，也不总是直截了当的。例如，如果像`not "Hello"`这样的表达式返回一个空字符串(`""`)该怎么办？像`not ""`这样的表达式会返回什么？这就是为什么`not`运算符总是返回`True`或`False`的原因。

现在您已经知道了`not`在 Python 中是如何工作的，您可以深入到这个逻辑操作符的更具体的用例中。在下一节中，您将学习在**布尔上下文**中使用`not`。

[*Remove ads*](/account/join/)

## 在布尔上下文中使用`not`运算符

像其他两个逻辑操作符一样，`not`操作符在布尔上下文中特别有用。在 Python 中，有两个定义布尔上下文的语句:

1.  [`if`语句](https://realpython.com/python-conditional-statements/)让你执行**条件执行**，根据一些初始条件采取不同的行动过程。
2.  [`while`循环](https://realpython.com/python-while-loop/)让您执行**条件迭代**并在给定条件为真时运行重复任务。

这两个结构是你所谓的[控制流](https://en.wikipedia.org/wiki/Control_flow)语句的一部分。它们帮助你决定程序的执行路径。在使用`not`操作符的情况下，您可以使用它来选择当给定的条件不满足时要采取的动作。

### `if`报表

您可以在`if`语句中使用`not`操作符来检查给定的条件是否不满足。要做一个`if`语句测试某件事是否没有发生，您可以将`not`操作符放在手边的条件前面。因为`not`操作符返回否定的结果，所以“真”变成了`False`，反之亦然。

带有`not`逻辑运算符的`if`语句的语法是:

```py
if not condition:
    # Do something...
```

在这个例子中，`condition`可以是一个布尔表达式或者任何有意义的 Python 对象。例如，`condition`可以是包含字符串、列表、字典、集合甚至用户自定义对象的变量。

如果`condition`评估为假，那么`not`返回`True`并且`if`代码块运行。如果`condition`评估为真，那么`not`返回`False`并且`if`代码块不执行。

一种常见的情况是使用一个[谓词](https://en.wikipedia.org/wiki/Predicate_(mathematical_logic))或[布尔值函数](https://en.wikipedia.org/wiki/Boolean-valued_function)作为`condition`。假设你想在做任何进一步的处理之前检查一个给定的数是否是质数。在这种情况下，您可以编写一个`is_prime()`函数:

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

>>> # Work with prime numbers only
>>> number = 3
>>> if is_prime(number):
...     print(f"{number} is prime")
...
3 is prime
```

在本例中，`is_prime()`将一个整数作为参数，如果该数是质数，则返回`True`。否则，它返回`False`。

您也可以在否定条件语句中使用该函数来处理那些您只想使用[合数](https://en.wikipedia.org/wiki/Composite_number)的情况:

>>>

```py
>>> # Work with composite numbers only
>>> number = 8
>>> if not is_prime(number):
...     print(f"{number} is composite")
...
8 is composite
```

因为也有可能您只需要处理合数，所以您可以像在第二个例子中所做的那样，通过将`is_prime()`与`not`操作符结合起来重用它。

编程中的另一个常见情况是找出一个数字是否在特定的数值区间内。在 Python 中，要确定一个数字`x`是否在给定的区间内，可以使用`and`操作符，也可以适当地链接比较操作符:

>>>

```py
>>> x = 30

>>> # Use the "and" operator
>>> if x >= 20 and x < 40:
...     print(f"{x} is inside")
...
30 is inside

>>> # Chain comparison operators
>>> if 20 <= x < 40:
...     print(f"{x} is inside")
...
30 is inside
```

在第一个例子中，您使用`and`操作符创建一个复合布尔表达式，检查`x`是否在`20`和`40`之间。第二个例子进行了同样的检查，但是使用了链式操作符，这是 Python 中的最佳实践。

**注意:**在大多数编程语言中，表达式`20 <= x < 40`没有意义。它将从评估`20 <= x`开始，这是真的。下一步是将真实结果与`40`进行比较，这没有多大意义，因此表达式失败。在 Python 中，会发生一些不同的事情。

Python 在内部将这种类型的表达式重写为等价的`and`表达式，比如`x >= 20 and x < 40`。然后，它执行实际的评估。这就是为什么你在上面的例子中得到正确的结果。

您可能还需要检查某个数字是否超出了目标区间。为此，你可以使用`or`操作符:

>>>

```py
>>> x = 50

>>> if x < 20 or x >= 40:
...     print(f"{x} is outside")
...
50 is outside
```

这个`or`表达式允许你检查`x`是否在`20`到`40`的区间之外。但是，如果您已经有了一个成功检查数字是否在给定区间内的工作表达式，那么您可以重用该表达式来检查相反的情况:

>>>

```py
>>> x = 50

>>> # Reuse the chained logic
>>> if not (20 <= x < 40):
...     print(f"{x} is outside")
50 is outside
```

在本例中，您将重用最初编码的表达式来确定一个数字是否在目标区间内。在表达式前有`not`，你检查`x`是否在`20`到`40`的区间之外。

[*Remove ads*](/account/join/)

### `while`循环

第二个可以使用`not`操作符的布尔上下文是在`while`循环中。这些循环在满足给定条件时迭代，或者直到您通过使用 [`break`](https://realpython.com/python-keywords/#iteration-keywords-for-while-break-continue-else) 、使用 [`return`](https://realpython.com/python-return-statement/) 或引发[异常](https://realpython.com/python-exceptions/)跳出循环。在`while`循环中使用`not`允许你在给定条件不满足时进行迭代。

假设您想编写一个小的 Python 游戏来猜测 1 到 10 之间的一个随机数。作为第一步，您决定使用 [`input()`](https://docs.python.org/3/library/functions.html#input) 来捕获用户名。因为名字是游戏其余部分工作的要求，所以你需要确保你得到它。为此，您可以使用一个`while`循环来询问用户名，直到用户提供一个有效的用户名。

启动你的[代码编辑器或 IDE](https://realpython.com/python-ides-code-editors-guide/) ，为你的游戏创建一个新的`guess.py`文件。然后添加以下代码:

```py
 1# guess.py
 2
 3from random import randint
 4
 5secret = randint(1, 10)
 6
 7print("Welcome!")
 8
 9name = ""
10while not name:
11    name = input("Enter your name: ").strip()
```

在`guess.py`中，你先从 [`random`](https://realpython.com/python-random/) 中导入 [`randint()`](https://docs.python.org/3/library/random.html#random.randint) 。此函数允许您在给定范围内生成随机整数。在这种情况下，您正在生成从`1`到`10`的数字，两者都包括在内。然后向用户打印一条欢迎消息。

第 10 行的`while`循环迭代，直到用户提供一个有效的名称。如果用户只按下 `Enter` 而没有提供名字，那么`input()`返回一个空字符串(`""`)，循环再次运行，因为`not ""`返回`True`。

现在，您可以通过编写提供猜测功能的代码来继续您的游戏。您可以自己完成，或者您可以展开下面的框来查看一个可能的实现。



游戏的第二部分应该允许用户输入 1 到 10 之间的数字作为他们的猜测。游戏应该将用户的输入与当前的秘密数字进行比较，并相应地采取行动。下面是一个可能的实现:

```py
while True:
    user_input = input("Guess a number between 1 and 10: ")
    if not user_input.isdigit():
        user_input = input("Please enter a valid number: ")
    guess = int(user_input)
    if guess == secret:
        print(f"Congrats {name}! You win!")
        break
    elif guess > secret:
        print("The secret number is lower than that...")
    else:
        print("The secret number is greater than that...")
```

您使用一个无限的`while`循环来接受用户的输入，直到他们猜出`secret`的数字。在每次迭代中，您检查输入是否匹配`secret`，并根据结果向用户提供线索。来吧，试一试！

作为练习，您可以在用户输掉游戏之前限制尝试次数。在这种情况下，尝试三次可能是个不错的选择。

你对这个小游戏的体验如何？要了解更多关于 Python 游戏编程的知识，请查看[PyGame:Python 游戏编程入门](https://realpython.com/pygame-a-primer/)。

现在您已经知道如何在布尔上下文中使用`not`，是时候学习在非布尔上下文中使用`not`了。这就是你在下一节要做的。

## 在非布尔上下文中使用`not`运算符

因为`not`操作符也可以将常规对象作为操作数，所以您也可以在非布尔上下文中使用它。换句话说，您可以在`if`语句或`while`循环之外使用它。可以说，`not`操作符在非布尔上下文中最常见的用例是反转给定变量的真值。

假设您需要在一个循环中交替执行两个不同的操作。在这种情况下，您可以使用标志变量在每次迭代中切换操作:

>>>

```py
>>> toggle = False

>>> for _ in range(4):
...     print(f"toggle is {toggle}")
...     if toggle:
...         # Do something...
...         toggle = False ...     else:
...         # Do something else...
...         toggle = True ...
toggle is False
toggle is True
toggle is False
toggle is True
```

每次这个循环运行时，您都要检查`toggle`的真值，以决定采取哪种行动。在每个代码块的末尾，您更改`toggle`的值，这样您就可以在下一次迭代中运行替代操作。更改`toggle`的值需要您重复两次类似的逻辑，这可能容易出错。

您可以使用`not`操作符来克服这个缺点，使您的代码更干净、更安全:

>>>

```py
>>> toggle = False

>>> for _ in range(4):
...     print(f"toggle is {toggle}")
...     if toggle:
...         pass  # Do something...
...     else:
...         pass  # Do something else...
...     toggle = not toggle ...
toggle is False
toggle is True
toggle is False
toggle is True
```

现在突出显示的行使用`not`操作符在`True`和`False`之间交替`toggle`的值。与您之前编写的示例相比，这段代码更简洁、重复性更低、更不容易出错。

## 使用基于函数的`not`操作符

与`and`操作符和`or`操作符不同，`not`操作符在 [`operator`](https://docs.python.org/3/library/operator.html#module-operator) 中有一个等价的基于函数的实现。这个功能叫做 [`not_()`](https://docs.python.org/3/library/operator.html#operator.not_) 。它将一个对象作为参数，并返回与等效的`not obj`表达式相同的结果:

>>>

```py
>>> from operator import not_

>>> # Use not_() with numeric values
>>> not_(0)
True
>>> not_(42)
False
>>> not_(0.0)
True
>>> not_(42.0)
False
>>> not_(complex(0, 0))
True
>>> not_(complex(42, 1))
False

>>> # Use not_() with strings
>>> not_("")
True
>>> not_("Hello")
False

>>> # Use not_() with other data types
>>> not_([])
True
>>> not_([1, 2, 3])
False
>>> not_({})
True
>>> not_({"one": 1, "two": 2})
False
```

要使用`not_()`，首先需要从`operator`导入。然后，您可以将该函数与任何 Python 对象或表达式一起用作参数。结果与使用等效的`not`表达式是一样的。

**注:** Python 还有 [`and_()`](https://docs.python.org/3/library/operator.html#operator.and_) 和 [`or_()`](https://docs.python.org/3/library/operator.html#operator.or_) 功能。然而，它们反映了相应的[位操作符](https://realpython.com/python-bitwise-operators/)，而不是布尔操作符。

`and_()`和`or_()`函数也适用于布尔参数:

>>>

```py
>>> from operator import and_, or_

>>> and_(False, False)
False
>>> and_(False, True)
False
>>> and_(True, False)
False
>>> and_(True, True)
True

>>> or_(False, False)
False
>>> or_(False, True)
True
>>> or_(True, False)
True
>>> or_(True, True)
True
```

在这些例子中，你使用`and_()`和`or_()`以及`True`和`False`作为参数。注意，表达式的结果分别匹配`and`和`not`操作符的真值表。

当您使用[高阶函数](http://en.wikipedia.org/wiki/Higher-order_function)，例如 [`map()`](https://realpython.com/python-map-function/) 、 [`filter()`](https://realpython.com/python-filter-function/) 等时，使用`not_()`函数代替`not`运算符会很方便。下面是一个使用`not_()`函数和 [`sorted()`](https://realpython.com/python-sort/) 对雇员列表进行排序的例子，方法是将空的雇员姓名放在列表的末尾:

>>>

```py
>>> from operator import not_

>>> employees = ["John", "", "", "Jane", "Bob", "", "Linda", ""]

>>> sorted(employees, key=not_)
['John', 'Jane', 'Bob', 'Linda', '', '', '', '']
```

在这个例子中，您有一个名为`employees`的初始列表，它包含一串名字。其中一些名称是空字符串。对`sorted()`的调用使用`not_()`作为`key`函数来创建一个新的对雇员进行排序的列表，将空的名字移动到列表的末尾。

[*Remove ads*](/account/join/)

## 使用 Python 的`not`操作符:最佳实践

当您使用`not`操作符时，您应该考虑遵循一些最佳实践，这些实践可以使您的代码更具可读性、更干净、更有 Pythonic 风格。在本节中，您将了解到在[成员资格](https://docs.python.org/3/reference/expressions.html#membership-test-operations)和[身份](https://docs.python.org/3/reference/expressions.html#is-not)测试的上下文中使用`not`操作符的一些最佳实践。

您还将了解负逻辑如何影响代码的可读性。最后，您将了解一些方便的技术，可以帮助您避免不必要的负面逻辑，这是一种编程最佳实践。

### 会员资格测试

当您确定特定的对象是否存在于给定的容器数据类型(如列表、元组、集合或字典)中时，成员资格测试通常很有用。要在 Python 中执行这种测试，可以使用 [`in`](https://realpython.com/python-keywords/#operator-keywords-and-or-not-in-is) 操作符:

>>>

```py
>>> numbers = [1, 2, 3, 4]

>>> 3 in numbers
True

>>> 5 in numbers
False
```

如果左边的对象在表达式右边的容器中，`in`操作符返回`True`。否则，它返回`False`。

有时你可能需要检查一个对象在给定的容器中是否是*而不是*。你怎么能这样做？这个问题的答案是`not`运算符。

在 Python 中，有两种不同的语法来检查对象是否不在给定的容器中。Python 社区认为第一种语法不好，因为它很难读懂。第二个语法读起来像普通英语:

>>>

```py
>>> # Bad practice
>>> not "c" in ["a", "b", "c"]
False

>>> # Best practice
>>> "c" not in ["a", "b", "c"]
False
```

第一个例子有效。然而，前导的`not`使得阅读您代码的人很难确定操作符是在处理`"c"`还是整个表达式`"c" in ["a", "b", "c"]`。这个细节使得表达难以阅读和理解。

第二个例子要清楚得多。Python 文档将第二个示例中的语法称为 [`not in`](https://docs.python.org/3/reference/expressions.html#not-in) 运算符。第一种语法可能是初学 Python 的人的常见做法。

现在是时候回顾一下检查一个数字是在数值区间内还是在数值区间外的例子了。如果您只处理整数，那么`not in`操作符提供了一种更易读的方式来执行这种检查:

>>>

```py
>>> x = 30

>>> # Between 20 and 40
>>> x in range(20, 41)
True

>>> # Outside 20 and 40
>>> x not in range(20, 41)
False
```

第一个例子检查`x`是否在`20`到`40`的范围或区间内。注意，您使用`41`作为 [`range()`](https://realpython.com/python-range/) 的第二个参数，将`40`包含在检查中。

当您处理整数时，这个关于在哪里使用`not`操作符的小技巧会对代码的可读性产生很大的影响。

### 检查物体的身份

用 Python 编码的另一个常见需求是检查对象的[身份](https://realpython.com/python-is-identity-vs-equality/)。您可以使用 [`id()`](https://docs.python.org/3/library/functions.html#id) 来确定对象的身份。这个内置函数将一个对象作为参数，并返回一个唯一标识当前对象的整数。这个数字代表对象的身份。

检查身份的实用方法是使用 [`is`](https://docs.python.org/3/reference/expressions.html#is) 操作符，这在一些条件语句中非常有用。例如，`is`操作符最常见的用例之一是测试给定对象*是否为* [`None`](https://realpython.com/null-in-python/) :

>>>

```py
>>> obj = None
>>> obj is None
True
```

当左操作数与右操作数相同时，`is`运算符返回`True`。否则，它返回`False`。

在这种情况下，问题是:如何检查两个对象是否具有相同的身份？同样，您可以使用两种不同的语法:

>>>

```py
>>> obj = None

>>> # Bad practice
>>> not obj is None
False

>>> # Best practice
>>> obj is not None
False
```

在这两个例子中，您检查`obj`是否与`None`对象具有相同的标识。第一个语法有些难读，而且不符合 Pythonic 语言。`is not`的语法更加清晰明了。Python 文档将这种语法称为 [`is not`](https://docs.python.org/3/reference/expressions.html#is-not) 操作符，并将其作为最佳实践推广使用。

[*Remove ads*](/account/join/)

### 避免不必要的负逻辑

`not`操作符使您能够颠倒给定条件或对象的含义或逻辑。在编程中，这种特性被称为**否定逻辑**或[否定](https://en.wikipedia.org/wiki/Negation)。

正确使用否定逻辑可能很棘手，因为这种逻辑很难思考和理解，更不用说解释了。一般来说，否定逻辑意味着比肯定逻辑更高的认知负荷。因此，只要有可能，你应该使用积极的提法。

下面是一个使用负条件返回输入数字绝对值的`custom_abs()`函数的例子:

>>>

```py
>>> def custom_abs(number):
...     if not number < 0:
...         return number
...     return -number
...

>>> custom_abs(42)
42

>>> custom_abs(-42)
42
```

这个函数接受一个数字作为参数，并返回它的绝对值。您可以通过使用积极的逻辑实现相同的结果，只需进行最小的更改:

>>>

```py
>>> def custom_abs(number):
...     if number < 0:
...         return -number
...     return number
...

>>> custom_abs(42)
42

>>> custom_abs(-42)
42
```

就是这样！你的`custom_abs()`现在使用正逻辑。更直白易懂。为了得到这个结果，您删除了`not`并移动了负号(`-`)来修改低于`0`的输入`number`。

**注意:** Python 提供了一个名为 [`abs()`](https://realpython.com/python-absolute-value/#using-the-built-in-abs-function-with-numbers) 的内置函数，返回一个数值输入的绝对值。`custom_abs()`的目的是方便话题的呈现。

您可以找到许多类似的例子，其中更改比较运算符可以删除不必要的否定逻辑。假设你想检查一个变量`x`是否等于给定值*而不是*。您可以使用两种不同的方法:

>>>

```py
>>> x = 27

>>> # Use negative logic
>>> if not x == 42:
...     print("not 42")
...
not 42

>>> # Use positive logic
>>> if x != 42:
...     print("not 42")
...
not 42
```

在本例中，通过将比较运算符从等于(`==`)改为不同(`!=`)来删除`not`运算符。在许多情况下，您可以通过使用适当的关系或相等运算符以不同的方式表达条件来避免负逻辑。

然而，有时负逻辑可以节省您的时间，并使您的代码更加简洁。假设您需要一个条件语句来初始化一个给定的文件，而这个文件在文件系统中并不存在。在这种情况下，您可以使用`not`来检查文件是否不存在:

```py
from pathlib import Path

file = Path("/some/path/config.ini")

if not file.exists():
    # Initialize the file here...
```

`not`操作符允许您反转在`file`上调用 [`.exists()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.exists) 的结果。如果`.exists()`返回`False`，那么你需要初始化文件。然而，如果条件为假，则`if`代码块不会运行。这就是为什么你需要`not`操作符来反转`.exists()`的结果。

**注意:**上面的例子使用了[标准库](https://docs.python.org/3/library/index.html)中的 [`pathlib`](https://docs.python.org/3/library/pathlib.html#module-pathlib) 来处理文件路径。要更深入地了解这个很酷的库，请查看 [Python 3 的 pathlib 模块:驯服文件系统](https://realpython.com/python-pathlib/)。

现在想想如何把这个否定条件变成肯定条件。到目前为止，如果文件存在，您不需要执行任何操作，因此您可以考虑使用一个 [`pass`语句](https://realpython.com/python-pass/)和一个附加的`else`子句来处理文件初始化:

```py
if file.exists():
    pass # YAGNI
else:
    # Initialize the file here...
```

尽管这段代码有效，但它违反了[“你不需要它”(YAGNI)](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it) 原则。这是消除消极逻辑的一次特别坚决的尝试。

这个例子背后的想法是要表明，有时使用否定逻辑是正确的做法。因此，您应该考虑您的具体问题并选择适当的解决方案。一个好的经验法则是尽可能避免消极逻辑，而不是不惜一切代价去避免它。

最后，你要特别注意避免**双重否定**。假设您有一个名为`NON_NUMERIC`的常量，它保存了 Python 无法转换成数字的字符，比如字母和标点符号。从语义上说，这个常数本身意味着否定。

现在假设您需要检查一个给定的字符是否是一个数值。既然已经有了`NON_NUMERIC`，可以想到用`not`来检查条件:

```py
if char not in NON_NUMERIC:
    number = float(char)
    # Do further computations...
```

这段代码看起来很奇怪，在您的程序员生涯中，您可能永远不会做这样的事情。然而，做一些类似的事情有时很诱人，比如上面的例子。

这个例子使用了双重否定。它依赖`NON_NUMERIC`，也依赖`not`，很难消化理解。如果你曾经遇到过这样的一段代码，那么花一分钟试着积极地写它，或者至少，试着去掉一层否定。

[*Remove ads*](/account/join/)

## 结论

Python 的 **`not`** 是将布尔表达式和对象的**真值**反转的逻辑运算符。当您需要检查条件语句和`while`循环中未满足的条件时，这很方便。

您可以使用`not`操作符来帮助您决定程序中的操作过程。您还可以使用它来反转代码中布尔变量的值。

**在本教程中，您学习了如何:**

*   使用 Python 的 **`not`** 操作符工作
*   在**布尔**和**非布尔上下文**中使用`not`运算符
*   使用 **`operator.not_()`** 在函数式中执行逻辑否定
*   尽可能避免代码中不必要的负逻辑

为此，您编写了一些实用的例子来帮助您理解`not`操作符的一些主要用例，因此您现在可以更好地准备在自己的代码中使用它。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和编写的教程一起看，加深你的理解: [**使用 Python not 运算符**](/courses/using-not-operator/)********