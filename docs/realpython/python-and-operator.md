# 在 Python 中使用“与”布尔运算符

> 原文：<https://realpython.com/python-and-operator/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 Python 和运算符**](/courses/and-operator-python/)

Python 有三个[布尔](https://realpython.com/python-boolean/)运算符，或者说**逻辑运算符** : `and`、`or`和`not`。在决定程序将遵循的执行路径之前，您可以使用它们来检查是否满足某些条件。在本教程中，您将了解到`and`操作符以及如何在您的代码中使用它。

**在本教程中，您将学习如何:**

*   理解 Python 的 **`and`运算符**背后的逻辑
*   构建并理解使用`and`操作符的**布尔**和**非布尔表达式**
*   在**布尔上下文**中使用`and`操作符来决定程序的**动作过程**
*   在**非布尔上下文**中使用`and`操作符使你的代码更加简洁

您还将编写一些实际的例子，帮助您理解如何使用`and`操作符以[python 式](https://realpython.com/learning-paths/writing-pythonic-code/)的方式处理不同的问题。即使你不使用`and`的所有特性，了解它们也会让你写出更好更准确的代码。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 在 Python 中使用布尔逻辑

早在 1854 年，[乔治·布尔](https://en.wikipedia.org/wiki/George_Boole)撰写了[思想法则](https://en.wikipedia.org/wiki/The_Laws_of_Thought)，其中包含了所谓的[布尔代数](https://en.wikipedia.org/wiki/Boolean_algebra)。这个代数依赖于两个值:**真**和**假**。它还定义了一组布尔运算，也称为逻辑运算，由通用运算符 [`AND`](https://en.wikipedia.org/wiki/Logical_conjunction) 、 [`OR`](https://en.wikipedia.org/wiki/Logical_disjunction) 和 [`NOT`](https://en.wikipedia.org/wiki/Negation) 表示。

这些布尔值和操作符在编程中非常有用。例如，您可以用运算符构造任意复杂的[布尔表达式](https://en.wikipedia.org/wiki/Boolean_expression)，并确定它们的结果[真值](https://en.wikipedia.org/wiki/Truth_value)为真或假。你可以使用布尔表达式的**真值**来决定你的程序的行动过程。

在 Python 中，[布尔类型](https://realpython.com/python-boolean/) `bool`是 [`int`](https://docs.python.org/3/library/functions.html#int) 的子类，可以取值`True`或`False`:

>>>

```py
>>> issubclass(bool, int)
True
>>> help(bool)
Help on class bool in module builtins:

class bool(int)
 ...

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

正如您在这段代码中看到的，Python 将`bool`实现为`int`的子类，有两个可能的值，`True`和`False`。这些值是 Python 中的[内置常量](https://docs.python.org/3/library/constants.html#built-in-constants)。它们在内部被实现为整数[数字](https://realpython.com/python-numbers/)，其中`True`的值为`1`，而`False`的值为`0`。注意`True`和`False`都必须大写。

除了`bool`类型，Python 还提供了三个布尔运算符，或者逻辑运算符，允许您将布尔表达式和对象组合成更复杂的表达式。这些运算符如下:

| 操作员 | 逻辑运算 |
| --- | --- |
| [T2`and`](https://realpython.com/python-keywords/#the-and-keyword) | 结合 |
| [T2`or`](https://realpython.com/python-or-operator/) | 分离 |
| [T2`not`](https://realpython.com/python-not-operator/) | 否认 |

使用这些运算符，您可以连接几个布尔表达式和对象来构建您自己的表达式。与其他语言不同，Python 使用英语单词来表示布尔运算符。这些单词是该语言的**关键词**，所以不能作为标识符使用。

在本教程中，您将学习 Python 的`and`操作符。该运算符执行逻辑`AND`运算。您将了解它是如何工作的，以及如何在布尔或非布尔上下文中使用它。

[*Remove ads*](/account/join/)

## Python 的`and`操作符入门

Python 的`and`操作符接受两个**操作数**，它们可以是布尔表达式、对象或组合。有了这些操作数，`and`操作符可以构建更复杂的表达式。一个`and`表达式中的操作数通常被称为**条件**。如果两个条件都为真，那么`and`表达式返回真结果。否则，它将返回错误结果:

>>>

```py
>>> True and True
True

>>> False and False
False

>>> True and False
False

>>> False and True
False
```

这些例子表明，只有当表达式中的两个操作数都为真时，`and`表达式才返回`True`。由于`and`操作符需要两个操作数来构建一个表达式，所以它是一个**二元操作符**。

上面的快速示例显示了所谓的`and`运算符真值表:

| `operand1` | `operand2` | `operand1 and operand2` |
| --- | --- | --- |
| 真实的 | 真实的 | 真实的 |
| 真实的 | 错误的 | 错误的 |
| 错误的 | 错误的 | 错误的 |
| 错误的 | 真实的 | 错误的 |

这个表格总结了像`operand1 and operand2`这样的布尔表达式的结果**真值**。表达式的结果取决于其操作数的真值。如果两个都是真的，那就是真的。否则，它就是假的。这是`and`操作符背后的一般逻辑。然而，这个操作符在 Python 中能做的不止这些。

在接下来的小节中，您将学习如何使用`and`来构建您自己的带有不同类型操作数的表达式。

### 使用 Python 的`and`运算符和布尔表达式

您通常会使用逻辑运算符来构建**复合布尔表达式**，它是[变量](https://realpython.com/python-variables/)和值的组合，结果产生一个布尔值。换句话说，布尔表达式返回`True`或`False`。

比较和相等测试是这种表达式的常见示例:

>>>

```py
>>> 5 == 3 + 2
True
>>> 5 > 3
True
>>> 5 < 3
False
>>> 5 != 3
True

>>> [5, 3] == [5, 3]
True

>>> "hi" == "hello"
False
```

所有这些表达式都返回`True`或`False`，这意味着它们是布尔表达式。您可以使用`and`关键字将它们组合起来，创建复合表达式，一次测试两个或更多的子表达式:

>>>

```py
>>> 5 > 3 and 5 == 3 + 2
True

>>> 5 < 3 and 5 == 5
False

>>> 5 == 5 and 5 != 5
False

>>> 5 < 3 and 5 != 5
False
```

这里，当你组合两个`True`表达式时，你得到的结果是`True`。任何其他组合返回`False`。从这些例子中，您可以得出结论，使用`and`操作符创建复合布尔表达式的语法如下:

```py
expression1 and expression2
```

如果两个子表达式`expression1`和`expression2`的值都是`True`，那么复合表达式就是`True`。如果至少有一个子表达式的计算结果为`False`，那么结果为`False`。

在构建复合表达式时，可以使用的`and`操作符的数量没有限制。这意味着您可以使用几个`and`操作符在一个表达式中组合两个以上的子表达式:

>>>

```py
>>> 5 > 3 and 5 == 3 + 2 and 5 != 3
True

>>> 5 < 3 and 5 == 3 and 5 != 3
False
```

同样，如果所有子表达式的计算结果都是`True`，那么就得到`True`。否则，你会得到`False`。特别是当表达式变长时，您应该记住 Python 是从左到右顺序计算表达式的。

[*Remove ads*](/account/join/)

### 短路评估

Python 的逻辑运算符，比如`and`、`or`，用的是一种叫做[短路求值](https://en.wikipedia.org/wiki/Short-circuit_evaluation)，或者**懒求值**的东西。换句话说，Python 只在需要的时候计算右边的操作数。

为了确定一个`and`表达式的最终结果，Python 从评估左操作数开始。如果是假的，那么整个表达式都是假的。在这种情况下，不需要计算右边的操作数。Python 已经知道最终结果了。

左操作数为假会自动使整个表达式为假。对剩余的操作数求值是对 CPU 时间的浪费。Python 通过简化计算来防止这种情况。

相比之下，`and`运算符仅在第一个操作数为真时才计算右边的操作数。在这种情况下，最终结果取决于右操作数的真值。如果为真，那么整个表达式为真。否则，表达式为假。

要演示短路功能，请看以下示例:

>>>

```py
>>> def true_func():
...     print("Running true_func()")
...     return True
...

>>> def false_func():
...     print("Running false_func()")
...     return False
...

>>> true_func() and false_func()  # Case 1
Running true_func()
Running false_func()
False

>>> false_func() and true_func()  # Case 2
Running false_func()
False

>>> false_func() and false_func()  # Case 3
Running false_func()
False

>>> true_func() and true_func()  # Case 4
Running true_func()
Running true_func()
True
```

下面是这段代码的工作原理:

*   **案例 1** : Python 对`true_func()`求值，返回`True`。为了确定最终结果，Python 对`false_func()`求值并得到`False`。您可以通过查看两个函数的输出来确认这一点。
*   **案例二** : Python 对`false_func()`求值，返回`False`。Python 已经知道最后的结果是`False`，所以不评价`true_func()`。
*   **案例三** : Python 运行`false_func()`，结果得到`False`。它不需要对重复的函数进行第二次求值。
*   **案例四** : Python 对`true_func()`求值，结果得到`True`。然后，它再次计算该函数。因为两个操作数的计算结果都是`True`，所以最终结果是`True`。

Python 从左到右处理布尔表达式。当它不再需要评估任何进一步的操作数或子表达式来确定最终结果时，它停止。总结一下这个概念，你应该记住如果一个`and`表达式中的左操作数为假，那么右操作数就不会被求值。

短路计算会对代码的性能产生重大影响。为了利用这一点，在构建`and`表达式时，请考虑以下提示:

*   将耗时的表达式放在关键字`and`的右边。这样，如果短路规则生效，代价高昂的表达式就不会运行。
*   将更有可能为假的表达式放在关键字`and`的左边。这样，Python 更有可能通过只计算左操作数来确定整个表达式是否为假。

有时，您可能希望避免特定布尔表达式中的惰性求值。你可以通过使用[位操作符](https://realpython.com/python-bitwise-operators/) ( `&`、`|`、`~`)来做到这一点。这些运算符也适用于布尔表达式，但是它们会急切地对操作数[求值](https://en.wikipedia.org/wiki/Eager_evaluation):

>>>

```py
>>> def true_func():
...     print("Running true_func()")
...     return True
...

>>> def false_func():
...     print("Running false_func()")
...     return False
...

>>> # Use logical and
>>> false_func() and true_func()
Running false_func()
False

>>> # Use bitwise and
>>> false_func() & true_func()
Running false_func()
Running true_func()
False
```

在第一个表达式中，`and`操作符像预期的那样缓慢地工作。它计算第一个函数，因为结果是假的，所以它不计算第二个函数。然而，在第二个表达式中，按位 AND 运算符(`&`)急切地调用两个函数，即使第一个函数返回`False`。注意，在这两种情况下，最终结果都是`False`。

尽管这一招很管用，但通常不被鼓励。您应该使用按位运算符来处理位，使用布尔运算符来处理布尔值和表达式。要更深入地了解按位运算符，请查看 Python 中的[按位运算符。](https://realpython.com/python-bitwise-operators/)

### 对公共对象使用 Python 的`and`操作符

您可以使用`and`操作符在一个表达式中组合两个 Python 对象。在那种情况下，Python 内部使用 [`bool()`](https://docs.python.org/3/library/functions.html#bool) 来确定操作数的真值。因此，您得到的是一个特定的对象，而不是一个布尔值。如果一个给定的操作数显式地求值为`True`或`False`，你只能得到`True`或`False`:

>>>

```py
>>> 2 and 3
3

>>> 5 and 0.0
0.0

>>> [] and 3
[]

>>> 0 and {}
0

>>> False and ""
False
```

在这些例子中，`and`表达式如果计算结果为`False`，则返回左边的操作数。否则，它返回右边的操作数。为了产生这些结果，`and`操作符使用 Python 的内部规则来确定对象的真值。Python 文档这样陈述这些规则:

> 默认情况下，除非对象的类定义了返回`False`的 [`__bool__()`](https://docs.python.org/3/reference/datamodel.html#object.__bool__) 方法或返回零的 [`__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 方法，否则对象被视为真。以下是大多数被认为是假的内置对象:
> 
> *   定义为假的常数:`None`和`False`。
> *   任意数值类型的零:`0`、`0.0`、`0j`、`Decimal(0)`、`Fraction(0, 1)`
> *   空序列和集合:`''`、`()`、`[]`、`{}`、`set()`、`range(0)`
> 
> ([来源](https://docs.python.org/3/library/stdtypes.html#truth-value-testing))

记住这些规则，再看看上面的代码。在第一个例子中，整数`2`为真(非零)，所以`and`返回右操作数`3`。在第二个例子中，`5`为真，所以`and`返回右操作数，即使它的计算结果为`False`。

下一个例子使用一个空列表(`[]`)作为左操作数。由于空列表的计算结果为 false，`and`表达式返回空列表。得到`True`或`False`的唯一情况是在表达式中显式使用布尔对象。

**注意:**如果你需要从一个涉及普通对象的`and`表达式中得到`True`或者`False`而不是布尔表达式，那么你可以使用`bool()`。这个内置函数根据您作为参数提供的特定对象的真值显式返回`True`或`False`。

下面是当您将`and`操作符用于普通 Python 对象而不是布尔表达式时，您可以如何总结它的行为。请注意，Python 使用每个对象的真值来确定最终结果:

| `object1` | `object2` | `object1 and object2` |
| --- | --- | --- |
| 错误的 | 错误的 | `object1` |
| 错误的 | 真实的 | `object1` |
| 真实的 | 真实的 | `object2` |
| 真实的 | 错误的 | `object2` |

一般来说，如果一个`and`表达式中的操作数是对象而不是布尔表达式，那么操作符返回左边的对象，如果它的计算结果是`False`。否则，它返回右边的对象，即使它的值是`False`。

[*Remove ads*](/account/join/)

### 混合布尔表达式和对象

您还可以在一个`and`表达式中组合布尔表达式和常见的 Python 对象。在这种情况下，`and`表达式仍然返回左操作数(如果它为假),否则它返回右操作数。返回值可以是`True`、`False`或常规对象，这取决于表达式的哪一部分提供了该结果:

>>>

```py
>>> 2 < 4 and 2
2
>>> 2 and 2 < 4
True

>>> 2 < 4 and []
[]
>>> [] and 2 < 4
[]

>>> 5 > 10 and {}
False
>>> {} and 5 > 10
{}

>>> 5 > 10 and 4
False
>>> 4 and 5 > 10
False
```

这些例子使用了布尔表达式和公共对象的组合。在每一对例子中，你可以看到你可以得到一个非布尔对象或者一个布尔值，`True`或者`False`。结果将取决于表达式的哪一部分提供最终结果。

下表总结了组合布尔表达式和常见 Python 对象时`and`运算符的行为:

| `expression` | `object` | `expression and object` |
| --- | --- | --- |
| `True` | 真实的 | `object` |
| `True` | 错误的 | `object` |
| `False` | 错误的 | `False` |
| `False` | 真实的 | `False` |

为了找出返回的内容，Python 对左边的布尔表达式进行求值，以获得其布尔值(`True`或`False`)。然后 Python 使用其内部规则来确定右边对象的真值。

作为测试您理解程度的一个练习，您可以尝试通过将第三列中操作数的顺序换成`object and expression`来重写该表。尝试预测每行将返回什么。

### 组合 Python 逻辑运算符

正如您在本教程前面看到的，Python 提供了两个额外的逻辑操作符:`or`操作符和`not`操作符。您可以将它们与`and`操作符一起使用来创建更复杂的复合表达式。如果你想用多个逻辑运算符做出准确清晰的表达式，那么你需要考虑每个运算符的[优先级](https://docs.python.org/3/reference/expressions.html#operator-precedence)。换句话说，您需要考虑 Python 执行它们的顺序。

与其他运算符相比，Python 的逻辑运算符优先级较低。然而，有时使用一对括号(`()`)来确保一致和可读的结果是有益的:

>>>

```py
>>> 5 or 2 and 2 > 1
5

>>> (5 or 3) and 2 > 1
True
```

这些例子在一个复合表达式中结合了`or`操作符和`and`操作符。就像`and`操作符一样，`or`操作符使用短路评估。然而，与`and`不同的是，`or`操作符一旦找到真操作数就会停止。你可以在第一个例子中看到这一点。因为`5`为真，所以`or`子表达式立即返回`5`，而不计算表达式的其余部分。

相比之下，如果将`or`子表达式放在一对括号中，那么它将作为单个真操作数工作，并且`2 > 1`也会被求值。最后的结果是`True`。

要点是，如果你在一个表达式中使用多个逻辑操作符，那么你应该考虑使用括号来使你的意图清晰。这个技巧也将帮助你得到正确的逻辑结果。

## 在布尔上下文中使用 Python 的`and`操作符

像 Python 的所有布尔操作符一样，`and`操作符在**布尔上下文**中特别有用。布尔上下文是您可以找到布尔运算符的大多数真实用例的地方。

Python 中有两种主要结构定义布尔上下文:

1.  [`if`语句](https://realpython.com/python-conditional-statements/)让你执行**条件执行**并根据一些初始条件的结果采取不同的行动。
2.  [`while`循环](https://realpython.com/python-while-loop/)让您执行**条件迭代**并在给定条件为真时运行重复任务。

这两个结构是你所谓的[控制流](https://en.wikipedia.org/wiki/Control_flow)语句的一部分。它们帮助你决定程序的执行路径。

您可以使用 Python 的`and`操作符在`if`语句和`while`循环中构造复合布尔表达式。

[*Remove ads*](/account/join/)

### `if`报表

布尔表达式通常被称为**条件**，因为它们通常意味着满足给定需求的需要。它们在条件语句中非常有用。在 Python 中，这种类型的语句以 [`if`关键字](https://realpython.com/python-keywords/#control-flow-keywords-if-elif-else)开始，并以一个条件继续。条件语句还可以包括`elif`和`else`子句。

Python 条件语句遵循英语语法中条件句的逻辑。如果条件为真，则执行`if`代码块。否则，执行跳转到不同的代码块:

>>>

```py
>>> a = -8

>>> if a < 0:
...     print("a is negative")
... elif a > 0:
...     print("a is positive")
... else:
...     print("a is equal to 0")
...
a is negative
```

因为`a`保持负数，所以条件`a < 0`为真。`if`代码块运行，屏幕上显示出`a is negative` [消息。但是，如果将`a`的值改为正数，那么`elif`块运行，Python 打印`a is positive`。最后，如果您将`a`设置为零，那么`else`代码块就会执行。继续玩`a`看看会发生什么！](https://realpython.com/python-print/)

现在，假设您想确保在运行某段代码之前满足两个条件，也就是说这两个条件都为真。为了验证这一点，假设您需要获得运行您的脚本的用户的年龄，处理该信息，并向用户显示他们当前的生活阶段。

启动您最喜欢的[代码编辑器或 IDE](https://realpython.com/python-ides-code-editors-guide/) 并创建以下脚本:

```py
# age.py

age = int(input("Enter your age: "))

if age >= 0 and age <= 9:
    print("You are a child!")
elif age > 9 and age <= 18:
    print("You are an adolescent!")
elif age > 18 and age <= 65:
    print("You are an adult!")
elif age > 65:
    print("Golden ages!")
```

这里你用 [`input()`](https://realpython.com/python-input-output/#reading-input-from-the-keyboard) 得到用户的年龄，然后[用](https://realpython.com/convert-python-string-to-int/) [`int()`](https://docs.python.org/3/library/functions.html#int) 把转换成整数。`if`子句检查`age`是否大于或等于`0`。在同一子句中，它检查`age`是否小于或等于`9`。为此，您需要构建一个`and`复合布尔表达式。

三个`elif`子句检查其他间隔，以确定与用户年龄相关联的生命阶段。

如果您从命令行[运行这个脚本](https://realpython.com/run-python-scripts/),那么您会得到如下结果:

```py
$ python age.py
Enter your age: 25
You are an adult!
```

根据您在命令行中输入的年龄，脚本会采取不同的操作。在这个具体的例子中，您提供了 25 岁的年龄，并在屏幕上显示了消息`You are an adult!`。

### `while`循环

`while`循环是第二个可以使用`and`表达式来控制程序执行流程的结构。通过在`while`语句头中使用`and`操作符，可以测试几个条件，只要所有条件都满足，就重复循环的代码块。

假设你正在为一个制造商制作一个控制系统的原型。该系统有一个关键机制，应该在 500 psi 或更低的压力下工作。如果压力超过 500 磅/平方英寸，而保持在 700 磅/平方英寸以下，那么系统必须运行一系列给定的标准安全动作。对于大于 700 psi 的压力，系统必须运行一套全新的安全措施。

为了解决这个问题，您可以使用一个带有`and`表达式的`while`循环。这里有一个脚本模拟了一个可能的解决方案:

```py
 1# pressure.py
 2
 3from time import sleep
 4from random import randint
 5
 6def control_pressure():
 7    pressure = measure_pressure()
 8    while True:
 9        if pressure <= 500:
10            break
11
12        while pressure > 500 and pressure <= 700:
13            run_standard_safeties()
14            pressure = measure_pressure()
15
16        while pressure > 700:
17            run_critical_safeties()
18            pressure = measure_pressure()
19
20    print("Wow! The system is safe...")
21
22def measure_pressure():
23    pressure = randint(490, 800)
24    print(f"psi={pressure}", end="; ")
25    return pressure
26
27def run_standard_safeties():
28    print("Running standard safeties...")
29    sleep(0.2)
30
31def run_critical_safeties():
32    print("Running critical safeties...")
33    sleep(0.7)
34
35if __name__ == "__main__":
36    control_pressure()
```

在`control_pressure()`中，您在第 8 行创建了一个无限的`while`循环。如果系统稳定且压力低于 500 psi，条件语句将跳出循环，程序结束。

在第 12 行，当系统压力保持在 500 psi 和 700 psi 之间时，第一个嵌套的`while`循环运行标准安全动作。在每次迭代中，循环获得新的压力测量值，以在下一次迭代中再次测试条件。如果压力超过 700 磅/平方英寸，那么管线 16 上的第二个回路运行临界安全动作。

**注意:**上例中`control_pressure()`的实现旨在展示`and`操作符如何在`while`循环的上下文中工作。

然而，这并不是您可以编写的最有效的实现。您可以重构`control_pressure()`来使用单个循环，而不使用`and`:

```py
def control_pressure():
    while True:
        pressure = measure_pressure()
        if pressure > 700:
            run_critical_safeties()
        elif 500 < pressure <= 700:
            run_standard_safeties()
        elif pressure <= 500:
            break
    print("Wow! The system is safe...")
```

在这个可替换的实现中，不使用`and`，而是使用链式表达式`500 < pressure <= 700`，它和`pressure > 500 and pressure <= 700`做的一样，但是更干净、更 Pythonic 化。另一个好处是你只需要调用`measure_pressure()`一次，这样效率会更高。

为了[运行这个脚本](https://realpython.com/run-python-scripts/)，打开您的命令行并输入以下命令:

```py
$ python pressure.py
psi=756; Running critical safeties...
psi=574; Running standard safeties...
psi=723; Running critical safeties...
psi=552; Running standard safeties...
psi=500; Wow! The system is safe...
```

您屏幕上的输出应该与这个示例输出略有不同，但是您仍然可以了解应用程序是如何工作的。

[*Remove ads*](/account/join/)

## 在非布尔上下文中使用 Python 的`and`操作符

事实上，`and`可以返回除了`True`和`False`之外的对象，这是一个有趣的特性。例如，这个特性允许您对**条件执行**使用`and`操作符。假设您需要更新一个`flag`变量，如果给定列表中的第一项等于某个期望值。对于这种情况，您可以使用条件语句:

>>>

```py
>>> a_list = ["expected value", "other value"]
>>> flag = False

>>> if len(a_list) > 0 and a_list[0] == "expected value":
...     flag = True
...

>>> flag
True
```

这里，条件检查列表是否至少有一项。如果是，它检查列表中的第一项是否等于`"expected value"`字符串。如果两个检查都通过，则`flag`变为`True`。您可以利用`and`操作符来简化这段代码:

>>>

```py
>>> a_list = ["expected value", "other value"]
>>> flag = False

>>> flag = len(a_list) > 0 and a_list[0] == "expected value" 
>>> flag
True
```

在这个例子中，突出显示的行完成了所有的工作。它检查这两个条件并一次完成相应的赋值。这个表达式从上一个例子中使用的`if`语句中去掉了`and`操作符，这意味着您不再在布尔上下文中工作。

上例中的代码比您之前看到的等价条件语句更简洁，但是可读性较差。为了正确理解这个表达式，您需要知道`and`操作符在内部是如何工作的。

## 将 Python 的`and`操作符投入使用

到目前为止，您已经学习了如何使用 Python 的`and`操作符来创建复合布尔表达式和非布尔表达式。您还学习了如何在布尔上下文中使用这个逻辑运算符，比如`if`语句和`while`循环。

在这一节中，您将构建几个实际的例子来帮助您决定何时使用`and`操作符。通过这些例子，您将了解如何利用`and`来编写更好、更 Pythonic 化的代码。

### 展平嵌套的`if`语句

Python 的[禅的一个原则是“扁平比嵌套好”例如，虽然有两层嵌套的`if`语句的代码是正常的，完全没问题，但是当你有两层以上的嵌套时，你的代码看起来就变得混乱和复杂了。](https://www.python.org/dev/peps/pep-0020/#the-zen-of-python)

假设你需要测试一个给定的数字是否为正。然后，一旦你确认它是正数，你需要检查这个数字是否低于给定的正值。如果是，您可以使用手头的数字进行特定的计算:

>>>

```py
>>> number = 7

>>> if number > 0:
...     if number < 10:
...         # Do some calculation with number...
...         print("Calculation done!")
...
Calculation done!
```

酷！这两个嵌套的`if`语句解决了你的问题。你先检查数字是否为正，然后再检查它是否低于`10`。在这个小例子中，对`print()`的调用是特定计算的占位符，只有当两个条件都为真时才运行。

尽管代码可以工作，但是最好通过移除嵌套的`if`来使它更加 Pythonic 化。你怎么能这样做？嗯，您可以使用`and`操作符将两个条件组合成一个复合条件:

>>>

```py
>>> number = 7

>>> if number > 0 and number < 10:
...     # Do some calculation with number...
...     print("Calculation done!")
...
Calculation done!
```

像`and`操作符这样的逻辑操作符通常通过移除嵌套的条件语句来提供改进代码的有效方法。尽可能利用它们。

在这个具体的例子中，您使用`and`创建一个复合表达式，检查一个数字是否在给定的范围或区间内。Python 通过链接表达式提供了一种更好的方式来执行这种检查。比如你可以把上面的条件写成`0 < number < 10`。这是下一节的主题。

### 检查数值范围

仔细查看下一节中的例子，您可以得出结论，Python 的`and`操作符是一个方便的工具，用于检查特定数值是否在给定的区间或范围内。例如，以下表达式检查数字`x`是否在`0`和`10`之间，包括两者:

>>>

```py
>>> x = 5
>>> x >= 0 and x <= 10
True

>>> x = 20
>>> x >= 0 and x <= 10
False
```

在第一个表达式中，`and`操作符首先检查`x`是否大于或等于`0`。由于条件为真，`and`操作员检查`x`是否低于或等于`10`。最终结果为真，因为第二个条件也为真。这意味着该数字在期望的区间内。

在第二个示例中，第一个条件为真，但第二个条件为假。一般结果为 false，这意味着该数字不在目标区间内。

您可以将此逻辑包含在函数中，并使其可重用:

>>>

```py
>>> def is_between(number, start=0, end=10):
...     return number >= start and number <= end
...

>>> is_between(5)
True
>>> is_between(20)
False

>>> is_between(20, 10, 40)
True
```

在这个例子中，`is_between()`将`number`作为参数。还需要`start`和`end`，它们定义了目标区间。注意，这些参数有[默认参数值](https://docs.python.org/dev/tutorial/controlflow.html#default-argument-values)，这意味着它们是[可选参数](https://realpython.com/python-optional-arguments/)。

您的`is_between()`函数返回评估一个`and`表达式的结果，该表达式检查`number`是否在`start`和`end`之间，包括这两个值。

**注:**无意中写出总是返回`False`的`and`表达式是常见错误。假设您想要编写一个表达式，从给定的计算中排除在`0`和`10`之间的值。

为了达到这个结果，您从两个布尔表达式开始:

1.  `number < 0`
2.  `number > 10`

以这两个表达式为起点，您可以考虑使用`and`将它们组合成一个复合表达式。然而，没有一个数同时小于`0`和大于`10`，所以你最终得到一个总是假的条件:

>>>

```py
>>> for number in range(-100, 100):
...     included = number < 0 and number > 10
...     print(f"Is {number} included?", included)
...
Is -100 included? False
Is -99 included? False

...

Is 0 included? False
Is 1 included? False

...

Is 98 included? False
Is 99 included? False
```

在这种情况下，`and`是处理手头问题的错误逻辑运算符。您应该使用`or`操作符。来吧，试一试！

尽管使用`and`操作符允许您优雅地检查一个数字是否在给定的区间内，但是有一种更 Pythonic 化的技术可以处理同样的问题。在数学中，你可以写 0 < x < 10 来表示 x 在 0 和 10 之间。

在大多数编程语言中，这个表达式没有意义。然而，在 Python 中，这个表达式非常有用:

>>>

```py
>>> x = 5
>>> 0 < x < 10
True

>>> x = 20
>>> 0 < x < 10
False
```

在不同的编程语言中，这个表达式将从计算`0 < x`开始，这是正确的。下一步是将真正的布尔值与`10`进行比较，这没有多大意义，所以表达式失败。在 Python 中，会发生一些不同的事情。

Python 在内部将这种类型的表达式重写为等价的`and`表达式，比如`x > 0 and x < 10`。然后，它执行实际的评估。这就是为什么你在上面的例子中得到正确的结果。

就像您可以用多个`and`操作符链接几个子表达式一样，您也可以在不显式使用任何`and`操作符的情况下链接它们:

>>>

```py
>>> x = 5
>>> y = 15

>>> 0 < x < 10 < y < 20
True

>>> # Equivalent and expression
>>> 0 < x and x < 10 and 10 < y and y < 20
True
```

您还可以使用这个 Python 技巧来检查几个值是否相等:

>>>

```py
>>> x = 10
>>> y = 10
>>> z = 10

>>> x == y == z
True

>>> # Equivalent and expression
>>> x == y and y == z
True
```

链式比较表达式是一个很好的特性，可以用多种方式编写。但是，你要小心。在某些情况下，最终的表达式可能很难阅读和理解，特别是对于来自没有这个特性的语言的程序员来说。

[*Remove ads*](/account/join/)

### 有条件地链接函数调用

如果你曾经在 [Unix 系统](https://en.wikipedia.org/wiki/Unix)上使用过 [Bash](https://en.wikipedia.org/wiki/Bash_(Unix_shell)) ，那么你可能知道`command1 && command2`构造。这是一种方便的技术，允许您在一个链中运行几个命令。当且仅当前一个命令成功时，每个命令才会运行:

```py
$ cd /not_a_dir && echo "Success"
bash: cd: /not_a_dir: No such file or directory

$ cd /home && echo "Success"
Success
```

这些例子使用 Bash 的短路和操作符(`&&`)使`echo`命令的执行依赖于`cd`命令的成功。

由于 Python 的`and`也实现了惰性求值的思想，所以可以用它来模拟这个 Bash 技巧。例如，您可以在一个单独的`and`表达式中链接一系列函数调用，如下所示:

```py
func1() and func2() and func3() ... and funcN()
```

在这种情况下，Python 调用`func1()`。如果函数的返回值评估为真值，那么 Python 调用`func2()`，以此类推。如果其中一个函数返回 false 值，那么 Python 不会调用其余的函数。

下面是一个使用一些 [`pathlib`](https://realpython.com/python-pathlib/) 函数来操作文本文件的例子:

>>>

```py
>>> from pathlib import Path
>>> file = Path("hello.txt")
>>> file.touch()

>>> # Use a regular if statement
>>> if file.exists():
...     file.write_text("Hello!")
...     file.read_text()
...
6
'Hello!'

>>> # Use an and expression
>>> file.exists() and file.write_text("Hello!") and file.read_text()
'Hello!'
```

不错！在一行代码中，您可以有条件地运行三个函数，而不需要一个`if`语句。在这个具体的例子中，唯一可见的区别是 [`.write_text()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.write_text) 返回它写入文件的字节数。交互式 shell 会自动将该值显示在屏幕上。请记住，当您将代码作为脚本运行时，这种差异是不可见的。

## 结论

Python 的`and`操作符允许你构造复合布尔表达式，你可以用它来决定你的程序的动作过程。您可以使用`and`操作符来解决布尔或非布尔上下文中的几个问题。学习如何正确使用`and`操作符可以帮助你编写更多的[python 式](https://realpython.com/learning-paths/writing-pythonic-code/)代码。

**在本教程中，您学习了如何:**

*   使用 Python 的 **`and`操作符**
*   用 Python 的`and`操作符构建**布尔**和**非布尔表达式**
*   在布尔上下文中使用`and`操作符来决定程序的**动作过程**
*   在**非布尔上下文**中使用`and`操作符使你的代码更加简洁

浏览本教程中的实际例子可以帮助您大致了解如何使用`and`操作符在 Python 代码中做出决策。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 Python 和运算符**](/courses/and-operator-python/)********