# 在 Python 中使用“或”布尔运算符

> 原文：<https://realpython.com/python-or-operator/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 Python 或者操作符**](/courses/using-python-or-operator/)

Python 中有三种布尔运算符:`and`、`or`和`not`。有了它们，你可以测试条件并决定你的程序将采取哪条执行路径。在本教程中，您将学习 Python `or`操作符以及如何使用它。

**本教程结束时，您将学会:**

*   Python `or`操作符的工作原理

*   如何在布尔和非布尔上下文中使用 Python `or`操作符

*   在 Python 中使用`or`可以解决什么样的编程问题

*   当别人使用 Python `or`操作符的一些特殊特性时，如何阅读和更好地理解他们的代码

通过构建一些实际的例子，您将学习如何使用 Python `or`操作符。即使您没有真正使用 Python `or`操作符提供的所有可能性，掌握它将允许您编写更好的代码。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 布尔逻辑

[乔治·布尔](https://en.wikipedia.org/wiki/George_Boole)(1815–1864)开发了现在所谓的 [**布尔代数**](https://en.wikipedia.org/wiki/Boolean_algebra) ，这是计算机硬件和编程语言背后的数字逻辑的基础。

布尔代数围绕表达式和对象的**真值**(无论它们是**真**还是**假**)构建，并基于布尔运算`AND`、`OR`和`NOT`。这些操作是通过逻辑 or [布尔运算符](https://realpython.com/python-boolean/)实现的，这些运算符允许您创建**布尔表达式**，这些表达式的计算结果为真或假。

在布尔逻辑的帮助下，您可以评估条件，并根据这些条件的真值决定您的程序将执行什么操作。这是编程的一个重要基础，为您提供了决定程序执行流程的工具。

让我们来看看 Python 中与布尔逻辑相关的一些基本概念:

*   **布尔型**是一种值类型，可以是`True`或`False`。在 Python 中，布尔类型是`bool`，是`int`的一个子类型。

*   **布尔值**是 Python 中的值`True`或`False`(带大写 *T* 和 *F* )。

*   一个**布尔变量**是一个[变量](https://realpython.com/python-variables/)，可以是`True`也可以是`False`。布尔变量常用作`flags`，表示特定条件是否存在。

*   一个**布尔表达式**是返回`True`或`False`的表达式。

*   **布尔上下文**可以是`if`条件和`while`循环，其中 Python 期望表达式评估为布尔值。几乎可以在布尔上下文中使用任何表达式或对象，Python 会尝试确定其真值。

*   **操作数**是表达式(布尔或非布尔)中涉及的子表达式或对象，由运算符连接。

*   **布尔或逻辑运算符**有`AND`(逻辑`AND`或合取)、`OR`(逻辑`OR`或析取)、`NOT`(逻辑`NOT`或否定)。关键字 **`and`** 、 **`or`** 、 **`not`** 是这些操作的 Python 运算符。

现在您对布尔逻辑有了更好的了解，让我们继续一些更具体的 Python 主题。

[*Remove ads*](/account/join/)

## Python 布尔运算符

Python 有三个布尔运算符，它们以普通英语单词的形式输出:

1.  [T2`and`](https://realpython.com/python-and-operator/)
2.  [T2`or`](https://realpython.com/python-keywords/#the-or-keyword)
3.  [T2`not`](https://realpython.com/python-not-operator/)

这些运算符连接布尔表达式(和对象)以创建复合布尔表达式。

Python 布尔操作符总是接受两个布尔表达式或两个对象或它们的组合，所以它们被认为是**二元操作符**。

在本教程中，您将学习 Python `or`操作符，它是在 Python 中实现逻辑`OR`操作的操作符。你会学到它是如何工作的，以及如何使用它。

## Python `or`操作符如何工作

使用布尔`OR`操作符，您可以将两个布尔表达式连接成一个复合表达式。至少有一个子表达式必须为真，复合表达式才能被认为是真的，哪个都没关系。如果两个子表达式都为假，则表达式为假。

这是`OR`操作符背后的一般逻辑。然而，Python `or`操作符完成了所有这些工作以及更多工作，您将在接下来的章节中看到。

### 将`or`与布尔表达式一起使用

您将需要两个子表达式来创建一个使用 Python `or`操作符作为连接器的布尔表达式。带有`or`的布尔表达式的基本语法如下:

```py
# Syntax for Boolean expression with or in Python
exp1 or exp2
```

如果至少有一个子表达式(`exp1`或`exp2`)的计算结果为`True`，则该表达式被认为是`True`。如果两个子表达式的计算结果都是`False`，那么表达式就是`False`。这个定义被称为**或**，因为它既允许两种可能性，也允许两种可能性。

下面是 Python `or`操作符行为的总结:

| `exp1`的结果 | `exp2`的结果 | `exp1 or exp2`的结果 |
| --- | --- | --- |
| `True` | `True` | `True` |
| `True` | `False` | `True` |
| `False` | `True` | `True` |
| `False` | `False` | `False` |

***表一。*** *逻辑 Python `or`运算符:真值表*

此表总结了类似于`exp1 or exp2`的布尔表达式的结果真值，取决于其子表达式的真值。

让我们通过编写一些实际例子来说明**表 1** 中所示的结果真值:

>>>

```py
>>> exp1 = 1 == 2
>>> exp1
False
>>> exp2 = 7 > 3
>>> exp2
True
>>> exp1 or exp2  # Return True, because exp2 is True
True
>>> exp2 or exp1  # Also returns True
True
>>> exp3 = 3 < 1
>>> exp1 or exp3  # Return False, because both are False
False
```

在前面的例子中，每当一个子表达式被求值为`True`，全局结果就是`True`。另一方面，如果两个子表达式都被求值为`False`，那么全局结果也是`False`。

[*Remove ads*](/account/join/)

### 将`or`用于公共对象

一般来说，涉及`OR`运算的表达式的操作数应该具有如**表 1** 所示的布尔值，并返回一个真值作为结果。对于对象，Python 对此并不严格，它在内部实现了一组规则来决定一个对象是真还是假:

> 默认情况下，除非对象的类定义了返回`False`的 [`__bool__()`](https://docs.python.org/3/reference/datamodel.html#object.__bool__) 方法或返回零的 [`__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__) 方法，否则对象被视为真。以下是大多数被认为是假的内置对象:
> 
> *   定义为假的常数:`None`和`False`。
> *   任意数值类型的零:`0`、`0.0`、`0j`、`Decimal(0)`、`Fraction(0, 1)`
> *   空序列和集合:`''`、`()`、`[]`、`{}`、`set()`、`range(0)`
> 
> ([来源](https://docs.python.org/3/library/stdtypes.html#truth-value-testing))

如果`or`操作中涉及的操作数是对象而不是布尔表达式，那么 Python `or`操作符返回 true 或 false 对象，而不是您所期望的值`True`或`False`。这个对象的真值是根据你之前看到的规则确定的。

这意味着 Python 不会将`or`操作的结果强制转换为`bool`对象。如果在 Python 中使用`or`测试两个对象，那么操作符将返回表达式中第一个求值为真的对象或最后一个对象，而不管其真值如何:

>>>

```py
>>> 2 or 3
2
>>> 5 or 0.0
5
>>> [] or 3
3
>>> 0 or {}
{}
```

在前两个例子中，第一个操作数(`2`和`5`)为真(非零)，所以 Python `or`操作符总是返回第一个。

在最后两个示例中，左操作数为 false(空对象)。Python `or`操作符计算两个操作数，并返回右边的对象，该对象可能计算为 true 或 false。

**注意:**如果你真的需要从一个包含对象的布尔表达式中获得值`True`或`False`，那么你可以使用`bool(obj)`，这是一个内置函数，根据`obj`的真值返回`True`或`False`。

您可以将前面代码中显示的行为总结如下:

| 左侧对象 | 右对象 | `x or y`的结果 |
| --- | --- | --- |
| `x` | `y` | `x`，如果评估为真，否则`y`。 |

***表二。*** *Python `or`测试对象时操作符的行为而不是布尔表达式*

简而言之，Python `or`操作符返回表达式中第一个计算结果为 true 的对象或最后一个对象，而不考虑其真值。

您可以通过在单个表达式中链接几个操作来概括这种行为，如下所示:

```py
a or b or c or d
```

在这个例子中，Python `or`操作符返回它找到的第一个或最后一个真操作数。这是记住`or`在 Python 中如何工作的经验法则。

### 混合布尔表达式和对象

您还可以在一个`or`操作中组合布尔表达式和常见的 Python 对象。在这种情况下，Python `or`操作符仍将返回第一个真操作数或最后一个操作数，但返回值可能是`True`或`False`或您正在测试的对象:

| **表达式的结果** | **对象结果** | **`exp or obj`的结果** |
| --- | --- | --- |
| `True` | `True` | `True` |
| `True` | `False` | `True` |
| `False` | `False` | `obj` |
| `False` | `True` | `obj` |

***表三。*** *Python `or`运算符测试对象和布尔表达式时的行为*

让我们通过一些例子来看看这是如何工作的:

>>>

```py
>>> 2 < 4 or 2  # Case 1
True
>>> 2 < 4 or []  # Case 2
True
>>> 5 > 10 or []  # Case 3
[]
>>> 5 > 10 or 4  # Case 4
4
```

在**案例 1** 和**案例 2** 中，子表达式`2 < 4`被求值为`True`，返回值为`True`。另一方面，在**案例 3** 和**案例 4** 中，子表达式`5 > 10`被求值为`False`，所以最后一个操作数被返回，你得到的是一个空列表(`[]`)和一个整数(`4`)，而不是`True`或`False`。

作为练习，您可以通过颠倒第三列中表达式的顺序来扩展**表 3** ，也就是说，使用`obj or exp`并尝试预测结果。

[*Remove ads*](/account/join/)

### 短路评估

Python 有时可以在评估所有相关的子表达式和对象之前确定布尔表达式的真值。例如，Python `or`操作符一旦发现被认为是真的东西，就停止计算操作数。例如，下面的表达式总是`True`:

>>>

```py
>>> True or 4 < 3
True
```

如果`or`表达式中的第一个操作数的值为真，不管第二个操作数的值是多少(`4 < 3`是`False`)，那么该表达式都被认为是真的，第二个操作数永远不会被计算。这被称为**短路(懒惰)评估**。

让我们考虑另一个例子:

>>>

```py
>>> def true_func():
...     print('Running true_func()')
...     return True
...
>>> def false_func():
...     print('Running false_func()')
...     return False
...
>>> true_func() or false_func()  # Case 1
Running true_func()
True
>>> false_func() or true_func()  # Case 2
Running false_func()
Running true_func()
True
>>> false_func() or false_func()  # Case 3
Running false_func()
Running false_func()
False
>>> true_func() or true_func()  # Case 4
Running true_func()
True
```

在**案例 1** 中，Python 评估了`true_func()`。因为它返回`True`，所以不计算下一个操作数(`false_func()`)。请注意，短语`Running false_func()`从未被打印出来。最后，整个表情被认为是`True`。

**案例 2** 对两个函数求值，因为第一个操作数(`false_func()`)是`False`。然后运算符返回第二个结果，也就是`true_func()`返回的值，也就是`True`。

**案例 3** 评估两个函数，因为两个函数都返回`False`。操作返回最后一个函数的返回值，即`False`，表达式被认为是`False`。

在**案例 4** 中，Python 只对第一个函数求值，是`True`，表达式是`True`。

在**短路(惰性)评估**中，如果表达式的值可以仅由第一个操作数确定，则不评估布尔表达式的第二个操作数。Python(像其他语言一样)为了提高性能而绕过了第二次计算，因为计算第二个操作数会不必要地浪费 CPU 时间。

最后，当谈到使用 Python `or`操作符时的性能时，请考虑以下几点:

*   Python `or`操作符右边的表达式可能会调用执行实质性或重要工作的函数，或者具有在短路规则生效时不会发生的副作用。

*   更有可能为真的条件可能是最左边的条件。这种方法可以减少程序的执行时间，因为这样 Python 就可以通过计算第一个操作数来确定条件是否为真。

### 章节摘要

您已经学习了 Python `or`操作符是如何工作的，并且已经看到了它的一些主要特性和行为。现在，您已经了解了足够的知识，可以通过学习如何使用运算符来解决现实世界中的问题。

在此之前，让我们回顾一下 Python 中关于`or`的一些要点:

*   它满足布尔`OR`操作符应该遵循的一般规则。如果一个或两个布尔子表达式为真，则结果为真。否则，如果两个子表达式都为假，则结果为假。

*   当它测试 Python 对象时，它返回对象而不是`True`或`False`值。这意味着如果表达式`x or y`的值为真，它将返回`x`，否则将返回`y`(不考虑其真值)。

*   它遵循一组预定义的 Python 内部规则来确定对象的真值。

*   一旦发现被认为是真的东西，它就停止计算操作数。这就叫短路或者懒评。

现在是时候借助一些例子来学习在哪里以及如何使用这个操作符了。

## 布尔上下文

在这一节中，您将看到一些如何使用 Python `or`操作符的实际例子，并学习如何利用它有些不寻常的行为来编写更好的 Python 代码。

在两种主要情况下，您可以说您正在 Python 中的布尔上下文中工作:

1.  **[`if`语句](https://realpython.com/python-conditional-statements/) :** 条件执行
2.  **[`while`循环](https://realpython.com/python-while-loop/) :** 条件重复

使用一个`if`语句，你可以根据某些条件的真值来决定程序的执行路径。

另一方面，`while`循环允许你重复一段代码，只要给定的条件保持为真。

这两个结构是你所谓的**控制流语句**的一部分。它们帮助你决定程序的执行路径。

您可以使用 Python `or`操作符来构建适用于`if`语句和`while`循环的布尔表达式，您将在接下来的两节中看到。

[*Remove ads*](/account/join/)

### `if`报表

假设您想在选择特定的执行路径之前确保两个条件中的一个(或两个)为真。在这种情况下，您可以使用 Python `or`操作符连接一个表达式中的条件，并在`if`语句中使用该表达式。

假设您需要得到用户的确认，以便根据用户的回答运行一些操作:

>>>

```py
>>> def answer():
...     ans = input('Do you...? (yes/no): ')
...     if ans.lower() == 'yes' or ans.lower() == 'y':
...         print(f'Positive answer: {ans}')
...     elif ans.lower() == 'no' or ans.lower() == 'n':
...         print(f'Negative answer: {ans}')
...
>>> answer()
Do you...? (yes/no): y
Positive answer: y
>>> answer()
Do you...? (yes/no): n
Negative answer: n
```

这里，您获得用户的[输入](https://realpython.com/python-input-output/)，并将其分配给`ans`。然后，`if`语句开始从左到右检查条件。如果它们中至少有一个被评估为真，那么它执行`if`代码块。`elif`语句也是如此。

在对`answer()`的第一次调用中，用户的输入是`y`，满足第一个条件，执行`if`代码块。在第二次调用中，用户的输入(`n`)满足了第二个条件，因此`elif`代码块运行。如果用户输入不满足任何条件，则不执行任何代码块。

另一个例子是当你试图确定一个数字是否超出范围时。在这种情况下，也可以使用 Python `or`操作符。以下代码测试`x`是否在`20`到`40`的范围之外:

>>>

```py
>>> def my_range(x):
...     if x < 20 or x > 40:
...         print('Outside')
...     else:
...         print('Inside')
...
>>> my_range(25)
Inside
>>> my_range(18)
Outside
```

当你用`x=25`调用`my_range()`时，`if`语句测试`25 < 20`，也就是`False`。然后测试`x > 40`，也是`False`。最终结果是`False`，所以执行了`else`块。

另一方面，`18 < 20`被评估为`True`。然后 Python `or`操作符进行短路评估，条件被认为是`True`。执行主块，值超出范围。

### `while`循环

`while`循环是布尔上下文的另一个例子，你可以使用 Python `or`操作符。通过在循环头中使用`or`,您可以测试几个条件并运行主体，直到所有条件都评估为假。

假设您需要测量一些工业设备的工作温度，直到温度达到 100°F 至 140°F。为此，您可以使用`while`回路:

```py
from time import sleep

temp = measure_temp()  # Initial temperature measurement

while temp < 100 or temp > 140:
    print('Temperature outside the recommended range')
    print('New Temperature measure in 30 seconds')
    sleep(30)
    print('Measuring Temperature...')
    temp = measure_temp()
    print(f'The new Temperature is {temp} ºF')
```

这是一个几乎是伪代码的玩具例子，但它说明了这个想法。这里，`while`循环运行，直到`temp`在 100°F 和 140°F 之间。如果温度值超出范围，则循环体运行，您将再次测量温度。一旦`measure_temp()`返回值介于 100 华氏度和 140 华氏度之间，循环结束。[使用`sleep(30)`](https://realpython.com/python-time-module/#suspending-execution) 每 30 秒测量一次温度。

**注意:**在前面的代码示例中，您使用 Python 的 f-strings 进行字符串格式化，如果您想更深入地了解 f-strings，那么您可以看看 [Python 3 的 f-Strings:一种改进的字符串格式化语法(指南)](https://realpython.com/python-f-strings/)。

## 非布尔上下文

您可以在布尔上下文之外利用 Python `or`操作符的特殊特性。经验法则仍然是布尔表达式的结果是第一个真操作数或者是行中的最后一个。

请注意，逻辑运算符(`or`)在赋值运算符(`=`)之前进行计算，因此您可以像处理普通表达式一样将布尔表达式的结果赋给变量:

>>>

```py
>>> a = 1
>>> b = 2
>>> var1 = a or b
>>> var1
1
>>> a = None
>>> b = 2
>>> var2 = a or b
>>> var2
2
>>> a = []
>>> b = {}
>>> var3 = a or b
>>> var3
{}
```

在这里，`or`操作符按预期工作，如果两个操作数的值都为假，则返回第一个真操作数或最后一个操作数。

您可以利用 Python 中`or`的这种有点特殊的行为来实现一些常见编程问题的 Python 解决方案。让我们看一些真实世界的例子。

[*Remove ads*](/account/join/)

### 变量的默认值

使用 Python `or`操作符的一种常见方式是根据其真值从一组对象中选择一个对象。您可以通过使用赋值语句来实现这一点:

>>>

```py
>>> x = a or b or None
```

在这里，你将表达式中的第一个真实对象赋值给`x`。如果所有对象(本例中的`a`和`b`都是假对象，那么 Python `or`操作符返回最后一个操作数 [`None`](https://realpython.com/null-in-python/) 。这是因为`or`操作符根据操作数的真值返回其中一个操作数。

您还可以使用此功能为变量分配默认值。以下示例在`a`为真时将`x`设置为`a`，否则设置为`default`:

>>>

```py
>>> x = a or default
```

在前面的代码中，只有当`a`的值为真时，才将`a`赋值给`x`。否则，`x`被分配给`default`。

### 默认`return`值

您可以在调用时操纵一些内置函数的`return`值。像 [`max()`和`min()`](https://realpython.com/python-min-and-max/) 这样的函数，它们将一个 iterable 作为参数并返回一个值，可能是这种黑客攻击的完美候选。

如果你给`max()`或`min()`提供一个空的 iterable，那么你将得到一个`ValueError`。然而，您可以通过使用 Python `or`操作符来修改这种行为。让我们来看看下面的代码:

>>>

```py
>>> lst = []  # Empty list to test max() and min()
>>> max(lst)
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    max(lst)
ValueError: max() arg is an empty sequence
>>> min(lst)
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    min(lst)
ValueError: min() arg is an empty sequence
>>> # Use Python or operator to modify this behavior
>>> max(lst or [0])  # Return 0
0
>>> min(lst or [0])  # Return 0
0
```

`max()`和`min()`的默认行为是，如果用空的 iterable 调用它们，就会引发一个`ValueError`。但是，通过使用 Python `or`操作符，您可以为这些函数提供一个默认的`return`值，并覆盖它们的默认行为。

**注意:**在前面的代码示例中，您看到了 Python 如何在出现问题时引发异常。如果你想了解更多关于 Python 中异常的知识，那么你可以看看[Python 异常介绍](https://realpython.com/courses/introduction-python-exceptions/)。

### 可变默认参数

初级 Python 程序员面临的一个常见问题是试图使用可变对象作为函数的默认参数。

默认参数的可变值可以在调用之间保持状态。这往往是意想不到的。发生这种情况是因为默认的参数值只被评估和保存一次，也就是说，当运行`def`语句时，而不是每次调用结果函数时。这就是为什么在函数内部改变可变缺省值时要小心的原因。

考虑下面的例子:

>>>

```py
>>> def mutable_default(lst=[]):  # Try to use a mutable value as default
...     lst.append(1)  # Change same object each time
...     print(lst)
...
>>> mutable_default(lst=[3, 2])  # Default not used
[3, 2, 1]
>>> mutable_default()  # Default used
[1]
>>> mutable_default()  # Default grows on each call
[1, 1]
>>> mutable_default()
[1, 1, 1]
```

这里，对`mutable_default()`的每次调用都将`1`追加到`lst`的末尾，因为`lst`保存了对同一个对象的引用(默认为`[]`)。并不是每次函数被调用时，你都会得到一个新的`list`。

如果这不是您想要的行为，那么传统的(也是最安全的)解决方案是将默认值移到函数体中:

>>>

```py
>>> def mutable_default(lst=None):  # Use None as formal default
...     if lst is None:
...         lst = []  # Default used? Then lst gets a new empty list.
...     lst.append(1)
...     print(lst)
...
>>> mutable_default(lst=[3, 2])  # Default not used
[3, 2, 1]
>>> mutable_default()  # Default used
[1]
>>> mutable_default()
[1]
```

使用这种实现，您可以确保每次不带参数调用`mutable_default()`时`lst`被设置为空`list`，依赖于`lst`的默认值。

本例中的`if`语句几乎可以被赋值语句`lst = lst or []`代替。这样，如果没有参数传入函数，那么`lst`将默认为`None`，Python `or`操作符将返回右边的空列表:

>>>

```py
>>> def mutable_default(lst=None):  # Use None as formal default
...     lst = lst or []  # Default used? Then lst gets an empty list.
...     lst.append(1)
...     print(lst)
...
>>> mutable_default(lst=[3, 2])  # Default not used
[3, 2, 1]
>>> mutable_default()  # Default used
[1]
>>> mutable_default()
[1]
```

然而，这并不完全相同。例如，如果传入一个空的`list`，那么`or`操作将导致函数修改并打印一个新创建的`list`，而不是像`if`版本那样修改并打印最初传入的`list`。

如果您非常确定您将只使用非空的`list`对象，那么您可以使用这种方法。否则，坚持使用`if`版本。

[*Remove ads*](/account/join/)

### 零除法

在处理数字计算时，零除法可能是一个常见的问题。为了避免这个问题，很可能你会通过使用一个`if`语句来检查分母是否等于`0`。

让我们来看一个例子:

>>>

```py
>>> def divide(a, b):
...     if not b == 0:
...         return a / b
...
>>> divide(15, 3)
5.0
>>> divide(0, 3)
0.0
>>> divide(15, 0)
```

这里，您测试了分母(`b`)是否不等于`0`，然后您返回了除法运算的结果。如果`b == 0`被评估为`True`，那么`divide()`隐式返回`None`。让我们看看如何获得类似的结果，但是这次使用 Python `or`操作符:

>>>

```py
>>> def divide(a, b):
...     return b == 0 or a / b
...
>>> divide(15, 3)
5.0
>>> divide(0, 3)
0.0
>>> divide(15, 0)
True
```

在这种情况下，Python `or`操作符计算第一个子表达式(`b == 0`)。只有当这个子表达式是`False`时，才计算第二个子表达式(`a / b`)，最终结果将是`a`和`b`的除法。

与前一个例子的不同之处在于，如果`b == 0`被求值为`True`，那么`divide()`返回`True`，而不是隐式的`None`。

### `lambda`中的多个表达式

Python 提供了 [`lambda`表达式](https://realpython.com/python-lambda/)，允许你创建简单的匿名函数。表达式`lambda parameters: expression`产生一个函数对象。如果您想定义简单的回调函数和按键函数，这种函数可能会很有用。

编写`lambda`函数最常见的模式是使用单个`expression`作为返回值。然而，您可以改变这一点，让`lambda`通过使用 Python `or`操作符来执行几个表达式:

>>>

```py
>>> lambda_func = lambda hello, world: print(hello, end=' ') or print(world)
>>> lambda_func('Hello', 'World!')
Hello World!
```

在这个例子中，您已经强制`lambda`运行两个表达式(`print(hello, end=' ')`和`print(world)`)。但是这个代码是如何工作的呢？这里`lambda`运行一个布尔表达式，其中执行两个函数。

当`or`对第一个函数求值时，它接收`None`，这是 [`print()`](https://realpython.com/python-print/) 的隐式返回值。由于`None`被认为是假的，`or`继续评估它的第二个操作数，并最终返回它作为布尔表达式的结果。

在这种情况下，布尔表达式返回的值也是`lambda`返回的值:

>>>

```py
>>> result = lambda_func('Hello', 'World!')
Hello World!
>>> print(result)
None
```

这里，`result`保存对`lambda`返回的值的引用，该值与布尔表达式返回的值相同。

## 结论

您现在已经知道 Python `or`操作符是如何工作的，以及如何使用它来解决 Python 中的一些常见编程问题。

现在您已经了解了 Python `or`操作符的基础，您将能够:

*   在布尔和非布尔上下文中使用 Python `or`操作符

*   有效使用 Python `or`操作符解决几种编程问题

*   利用 Python 中的`or`的一些特殊特性，编写更好、更 Python 化的代码

*   当别人使用 Python `or`操作符时，阅读并更好地理解他们的代码

此外，您还学习了一点布尔逻辑，以及它在 Python 中的一些主要概念。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 Python 或者操作符**](/courses/using-python-or-operator/)********