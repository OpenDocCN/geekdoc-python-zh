# Python 关键词:简介

> 原文：<https://realpython.com/python-keywords/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**探索 Python 中的关键词**](/courses/exploring-keywords-in-python/)

每种编程语言都有特殊的保留字，或**关键字**，它们有特定的含义和使用限制。Python 也不例外。Python 关键字是任何 Python 程序的基本构件。

在本文中，您将找到对所有 Python 关键字的基本介绍，以及有助于了解每个关键字的更多信息的其他资源。

**本文结束时，你将能够:**

*   **识别** Python 关键字
*   **了解**每个关键词的用途
*   **使用`keyword`模块以编程方式使用关键字来处理**

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 关键词

Python 关键字是特殊的保留字，具有特定的含义和用途，除了这些特定的用途之外，不能用于任何其他用途。这些关键字总是可用的——您永远不必将它们导入到您的代码中。

Python 关键字不同于 Python 的[内置函数和类型](https://docs.python.org/3/library/functions.html)。内置函数和类型也总是可用的，但是它们在使用时没有关键字那么严格。

一个你*不能*用 Python 关键字做的事情的例子是给它们赋值。如果你尝试，那么你会得到一个`SyntaxError`。如果你试图给一个内置函数或类型赋值，你不会得到一个`SyntaxError`，但这仍然不是一个好主意。关于关键字可能被误用的更深入的解释，请查看 Python 中的[无效语法:语法错误的常见原因](https://realpython.com/invalid-syntax-python/#misspelling-missing-or-misusing-python-keywords)。

截至 Python 3.8，Python 中有[三十五个关键字](https://docs.python.org/3.8/reference/lexical_analysis.html#keywords)。下面是本文其余部分相关章节的链接:

| [T2`False`](#the-true-and-false-keywords) | [T2`await`](#the-await-keyword) | [T2`else`](#the-else-keyword) | [T2`import`](#the-import-keyword) | [T2`pass`](#the-pass-keyword) |
| [T2`None`](#the-none-keyword) | [T2`break`](#the-break-keyword) | [T2`except`](#the-except-keyword) | [T2`in`](#the-in-keyword) | [T2`raise`](#the-raise-keyword) |
| [T2`True`](#the-true-and-false-keywords) | [T2`class`](#the-class-keyword) | [T2`finally`](#the-finally-keyword) | [T2`is`](#the-is-keyword) | [T2`return`](#the-return-keyword) |
| [T2`and`](#the-and-keyword) | [T2`continue`](#the-continue-keyword) | [T2`for`](#the-for-keyword) | [T2`lambda`](#the-lambda-keyword) | [T2`try`](#the-try-keyword) |
| [T2`as`](#the-as-keyword) | [T2`def`](#the-def-keyword) | [T2`from`](#the-from-keyword) | [T2`nonlocal`](#the-nonlocal-keyword) | [T2`while`](#the-while-keyword) |
| [T2`assert`](#the-assert-keyword) | [T2`del`](#the-del-keyword) | [T2`global`](#the-global-keyword) | [T2`not`](#the-not-keyword) | [T2`with`](#the-with-keyword) |
| [T2`async`](#the-async-keyword) | [T2`elif`](#the-elif-keyword) | [T2`if`](#the-if-keyword) | [T2`or`](#the-or-keyword) | [T2`yield`](#the-yield-keyword) |

你可以使用这些链接跳转到你想阅读的关键词，或者你可以继续阅读一个导游。

**注意:**两个关键词除了它们最初的用例之外，还有额外的用途。`else`关键字也是与循环一起使用的[，以及与`try`和`except`T10 一起使用的](#the-else-keyword-used-with-loops)[。`as`关键字也与`with`关键字](#the-else-keyword-used-with-try-and-except)一起使用[。](#the-as-keyword-used-with-with)

[*Remove ads*](/account/join/)

## 如何识别 Python 关键词

随着时间的推移，Python 关键字列表已经发生了变化。例如，直到 Python 3.7 才添加了关键字`await`和`async`。另外，`print`和`exec`在 Python 2.7 中都是关键字，但在 Python 3+中已经变成了内置函数，不再出现在关键字列表中。

在下面几节中，您将学习几种方法来知道或找出哪些单词是 Python 中的关键字。

### 使用带有语法高亮显示的 IDE

外面有很多好的 Python IDEs。它们都会突出显示关键字，以区别于代码中的其他单词。这将帮助您在编程时快速识别 Python 关键字，从而避免错误地使用它们。

### 使用 REPL 中的代码来检查关键字

在 [Python REPL](https://realpython.com/interacting-with-python/#using-the-python-interpreter-interactively) 中，有多种方法可以识别有效的 Python 关键字并了解更多。

**注意:**本文中的代码示例使用 [Python 3.8](https://realpython.com/python38-new-features/) ，除非另有说明。

您可以使用`help()`获得可用关键字列表:

>>>

```py
>>> help("keywords")

Here is a list of the Python keywords.  Enter any keyword to get more help.

False               class               from                or
None                continue            global              pass
True                def                 if                  raise
and                 del                 import              return
as                  elif                in                  try
assert              else                is                  while
async               except              lambda              with
await               finally             nonlocal            yield
break               for                 not
```

接下来，如上面的输出所示，您可以通过传入您需要更多信息的特定关键字来再次使用`help()`。例如，您可以使用`pass`关键字来实现这一点:

>>>

```py
>>> help("pass")
The "pass" statement
********************

 pass_stmt ::= "pass"

"pass" is a null operation — when it is executed, nothing happens. It
is useful as a placeholder when a statement is required syntactically,
but no code needs to be executed, for example:

 def f(arg): pass    # a function that does nothing (yet)

 class C: pass       # a class with no methods (yet)
```

Python 还提供了一个`keyword`模块，用于以编程方式处理 Python 关键字。Python 中的`keyword`模块为处理关键字提供了两个有用的成员:

1.  **`kwlist`** 为您正在运行的 Python 版本提供了所有 Python 关键字的列表。
2.  **`iskeyword()`** 提供了一种简便的方法来确定一个字符串是否也是一个关键字。

要获得您正在运行的 Python 版本中所有关键字的列表，并快速确定定义了多少个关键字，请使用`keyword.kwlist`:

>>>

```py
>>> import keyword
>>> keyword.kwlist
['False', 'None', 'True', 'and', 'as', 'assert', 'async', ...
>>> len(keyword.kwlist)
35
```

如果您需要更多地了解某个关键字，或者需要以编程的方式使用关键字，那么 Python 为您提供了这种文档和工具。

### 找一个`SyntaxError`

最后，另一个表明你正在使用的单词实际上是一个关键字的指标是，当你试图给它赋值，用它命名一个函数，或者用它做其他不允许的事情时，你是否得到了一个`SyntaxError`。这个有点难发现，但这是 Python 让你知道你在错误地使用关键字的一种方式。

## Python 关键字及其用法

以下部分根据 Python 关键字的用法对其进行分组。例如，第一组是所有用作值的关键字，第二组是用作运算符的关键字。这些分组将帮助您更好地理解如何使用关键字，并提供一种很好的方式来组织 Python 关键字的长列表。

以下章节中使用的一些术语可能对您来说是新的。这里对它们进行了定义，您应该在继续之前了解它们的含义:

*   **真值**是指一个值的[布尔](https://realpython.com/python-boolean/)求值。值的真值表示该值是**真值**还是**假值**。

*   **真值**表示在布尔上下文中评估为真的任何值。要确定一个值是否为真，将其作为参数传递给`bool()`。如果它返回`True`，那么这个值就是 the。真值的例子有非空字符串、任何不是`0`的数字、非空列表等等。

*   **Falsy** 表示在布尔上下文中评估为假的任何值。要确定一个值是否为 falsy，将其作为参数传递给`bool()`。如果它返回`False`，那么值就是 falsy。虚假值的例子有`""`、`0`、`[]`、`{}`和`set()`。

有关这些术语和概念的更多信息，请查看 Python 中的[运算符和表达式。](https://realpython.com/python-operators-expressions/#evaluation-of-non-boolean-values-in-boolean-context)

[*Remove ads*](/account/join/)

### 数值关键词:`True`、`False`、`None`、

有三个 Python 关键字用作值。这些值是[单值](https://python-patterns.guide/gang-of-four/singleton/)值，可以反复使用，并且总是引用完全相同的对象。您很可能会经常看到和使用这些值。

#### `True`和`False`关键词

**`True`** 关键字在 Python 代码中用作布尔真值。Python 关键字 **`False`** 类似于`True`关键字，但是具有相反的布尔值 false。在其他编程语言中，你会看到这些关键字被写成小写(`true`和`false`)，但是在 Python 中它们总是被写成大写。

Python 关键字`True`和`False`可以分配给[变量](https://realpython.com/python-variables/)并直接进行比较:

>>>

```py
>>> x = True
>>> x is True
True

>>> y = False
>>> y is False
True
```

Python 中的大多数值在传递给`bool()`时将计算为`True`。Python 中只有几个值在传递给`bool()` : `0`、`""`、`[]`和`{}`时会计算为`False`。向`bool()`传递一个值表示该值的真值，或者等价的布尔值。通过将值传递给`bool()`，您可以将值的真实性与`True`或`False`进行比较:

>>>

```py
>>> x = "this is a truthy value"
>>> x is True
False
>>> bool(x) is True
True

>>> y = ""  # This is falsy
>>> y is False
False
>>> bool(y) is False
True
```

请注意，使用 **`is`** 将真值直接与`True`或`False`进行比较不起作用。只有当你想知道一个值实际上是*`True`还是`False`时，你才应该直接将这个值与`True`或`False`进行比较。

当编写基于值的真实性的条件语句时，您应该*而不是*直接与`True`或`False`进行比较。您可以依靠 Python 来为您进行条件的真实性检查:

>>>

```py
>>> x = "this is a truthy value"
>>> if x is True:  # Don't do this
...     print("x is True")
...
>>> if x:  # Do this
...     print("x is truthy")
...
x is truthy
```

在 Python 中，一般不需要将值转换成显式的`True`或`False`。Python 将隐式地为您确定值的真实性。

#### `None`关键词

Python 关键字 **`None`** 表示没有值。在其他编程语言中，`None`被表示为`null`、`nil`、`none`、`undef`或`undefined`。

`None`也是一个函数返回的默认值，如果它没有 [`return`语句](https://realpython.com/python-return-statement/):

>>>

```py
>>> def func():
...     print("hello")
...
>>> x = func()
hello
>>> print(x)
None
```

要更深入地了解这个非常重要和有用的 Python 关键字，请查看 Python: Understanding Python 的 NoneType 对象中的 [Null。](https://realpython.com/null-in-python/)

### 操作员关键词:`and`、`or`、`not`、`in`、`is`、

几个 Python 关键字被用作运算符。在其他编程语言中，这些操作符使用类似于`&`、`|`和`!`的符号。这些的 Python 运算符都是关键字:

| 数学运算符 | 其他语言 | Python 关键字 |
| --- | --- | --- |
| 还有，∧ | `&&` | `and` |
| 或者，∨ | `&#124;&#124;` | `or` |
| 不是， | `!` | `not` |
| 包含，&in; |  | `in` |
| 身份 | `===` | `is` |

Python 代码是为了可读性而设计的。这就是为什么在其他编程语言中使用符号的许多操作符在 Python 中是关键字。

#### `and`关键词

Python 关键字 **`and`** 用于确定左右操作数是真还是假。如果两个操作数都是真的，那么结果将是真的。如果一个是假的，那么结果将是假的:

```py
<expr1> and <expr2>
```

注意，`and`语句的结果不一定是`True`或`False`。这是因为`and`的古怪行为。与其将操作数评估为布尔值，`and`不如简单地返回`<expr1>`，如果为 falsy，否则返回`<expr2>`。一条`and`语句的结果可以传递给`bool()`以获得显式的`True`或`False`值，或者它们可以在一条条件`if`语句中使用。

如果您想定义一个与`and`表达式做同样事情的表达式，但是不使用`and`关键字，那么您可以使用 Python 三元运算符:

```py
left if not left else right
```

上述语句将产生与`left and right`相同的结果。

因为`and`如果为 falsy 则返回第一个操作数，否则返回最后一个操作数，所以也可以在赋值中使用`and`:

```py
x = y and z
```

如果 y 是 falsy，那么这将导致`x`被赋予`y`的值。否则，`x`将被赋予`z`的值。然而，这导致了混乱的代码。一个更详细、更清晰的替代方案是:

```py
x = y if not y else z
```

这段代码比较长，但是它更清楚地表明了您想要完成的任务。

#### `or`关键词

Python 的 **`or`** 关键字用于判断是否至少有一个操作数是真的。如果第一个操作数为真，则`or`语句返回该操作数，否则返回第二个操作数:

```py
<expr1> or <expr2>
```

就像关键字`and`一样，`or`不会将其操作数转换为布尔值。相反，它依赖于他们的真实性来决定结果。

如果您想在不使用`or`的情况下编写类似于`or`的表达式，那么您可以使用三元表达式:

```py
left if left else right
```

该表达式将产生与`left or right`相同的结果。为了利用这种行为，您有时也会看到在赋值中使用`or`。通常不鼓励这种做法，而支持更明确的赋值。

为了更深入地了解`or`，你可以阅读关于[如何使用 Python `or`操作符](https://realpython.com/python-or-operator/)。

#### `not`关键词

Python 的 **`not`** 关键字用于获取变量的相反布尔值:

>>>

```py
>>> val = ""  # Truthiness value is `False`
>>> not val
True

>>> val = 5  # Truthiness value is `True`
>>> not val
False
```

`not`关键字用于条件语句或其他布尔表达式中，以*翻转*布尔含义或结果。与**`and`****`or`**不同， **`not`** 会确定显式布尔值，`True`或`False`，然后返回相反的。

如果您想在不使用`not`的情况下获得相同的行为，那么您可以使用以下三元表达式:

```py
True if bool(<expr>) is False else False
```

该语句将返回与`not <expr>`相同的结果。

#### `in`关键词

Python 的 **`in`** 关键字是一个强大的遏制检查，或者说**隶属运算符**。给定要查找的元素和要搜索的容器或序列，`in`将返回`True`或`False`，指示是否在容器中找到了该元素:

```py
<element> in <container>
```

使用`in`关键字的一个很好的例子是检查字符串中的特定字母:

>>>

```py
>>> name = "Chad"
>>> "c" in name
False
>>> "C" in name
True
```

`in`关键字适用于所有类型的容器:列表、字典、集合、字符串以及任何定义了`__contains__()`或者可以被迭代的东西。

#### `is`关键词

Python 的 **`is`** 关键字是一个身份检查。这与检查相等性的`==`操作符不同。有时两个事物可以被认为是相等的，但在内存中不是完全相同的对象。`is`关键字确定两个对象是否是完全相同的对象:

```py
<obj1> is <obj2>
```

如果`<obj1>`和`<obj2>`在内存中是完全相同的对象，它将返回`True`，否则它将返回`False`。

大多数时候你会看到`is`用来检查一个对象是否是`None`。由于`None`是单例的，只能存在`None`的一个实例，所以所有的`None`值都是内存中完全相同的对象。

如果这些概念对你来说是新的，那么你可以通过查看 [Python 来获得更深入的解释！=' Is Not 'is not ':在 Python 中比较对象](https://realpython.com/python-is-identity-vs-equality/)。为了更深入地了解`is`是如何工作的，请查看 Python 中的[操作符和表达式。](https://realpython.com/python-operators-expressions/#identity-operators)

[*Remove ads*](/account/join/)

### 控制流关键字:`if`、`elif`、`else`、

三个 Python 关键字用于控制流:`if`、`elif`和`else`。这些 Python 关键字允许您使用条件逻辑，并在特定条件下执行代码。这些关键字非常常见——它们几乎会出现在你用 Python 看到或编写的每个程序中。

#### `if`关键词

**`if`** 关键字用于开始一个[条件语句](https://realpython.com/python-conditional-statements/)。一个`if`语句允许你写一个代码块，只有当`if`后面的表达式是真的时才被执行。

`if`语句的语法以行首的关键字`if`开始，后面是一个有效的表达式，将对其真值进行评估:

```py
if <expr>:
    <statements>
```

语句是大多数程序的重要组成部分。有关`if`语句的更多信息，请查看 Python 中的[条件语句。](https://realpython.com/python-conditional-statements/#introduction-to-the-if-statement)

`if`关键字的另一个用途是作为 Python 的[三元运算符](https://realpython.com/python-conditional-statements/#conditional-expressions-pythons-ternary-operator)的一部分:

```py
<var> = <expr1> if <expr2> else <expr3>
```

这是下面的一行`if...else`语句:

```py
if <expr2>:
    <var> = <expr1>
else:
    <var> = <expr3>
```

如果您的表达式是不复杂的语句，那么使用三元表达式提供了一个很好的方法来稍微简化您的代码。一旦条件变得有点复杂，依靠标准的`if`语句通常会更好。

#### `elif`关键词

**`elif`** 语句的外观和功能与`if`语句相似，但有两个主要区别:

1.  使用`elif`仅在一个`if`语句或另一个`elif`之后有效。
2.  您可以根据需要使用任意多的`elif`语句。

在其他编程语言中，`elif`要么是`else if`(两个独立的单词)，要么是`elseif`(两个单词混合在一起)。当你看到 Python 中的`elif`时，想想`else if`:

```py
if <expr1>:
    <statements>
elif <expr2>:
    <statements>
elif <expr3>:
    <statements>
```

Python 没有 [`switch`语句](https://en.wikipedia.org/wiki/Switch_statement)。获得其他编程语言用`switch`语句提供的相同功能的一种方法是使用`if`和`elif`。关于在 Python 中再现`switch`语句的其他方法，请查看 Python 中的[仿真 switch/case 语句。](https://realpython.com/courses/emulating-switch-case-python/)

#### `else`关键词

**`else`** 语句，结合 Python 关键字`if`和`elif`，表示只有当其他条件块`if`和`elif`都为假时才应该执行的代码块:

```py
if <expr>:
    <statements>
else:
    <statements>
```

请注意，`else`语句没有采用条件表达式。对于 Python 程序员来说，了解 [`elif`和`else`关键词](https://realpython.com/python-conditional-statements/#the-else-and-elif-clauses)及其正确用法至关重要。它们和`if`一起构成了任何 Python 程序中最常用的组件。

[*Remove ads*](/account/join/)

### 迭代关键词:`for`、`while`、`break`、`continue`、`else`、

循环和迭代是非常重要的编程概念。几个 Python 关键字用于创建和处理循环。这些，就像上面用于条件的 Python 关键字一样，将会在你遇到的每个 Python 程序中使用和看到。理解它们以及它们的正确用法将有助于您提高 Python 程序员的水平。

#### `for`关键词

Python 中最常见的循环是`for`循环。它由前面解释的 Python 关键字 **`for`** 和 **`in`** 组合而成。`for`循环的基本语法如下:

```py
for <element> in <container>:
    <statements>
```

一个常见的例子是循环播放数字 1 到 5，并将它们打印到屏幕上:

>>>

```py
>>> for num in range(1, 6):
...     print(num)
...
1
2
3
4
5
```

在其他编程语言中，`for`循环的语法看起来会有些不同。您经常需要指定变量、继续的条件以及增加变量的方式(`for (int i = 0; i < 5; i++)`)。

在 Python 中，`for`循环就像其他编程语言中的 [for-each 循环](https://en.wikipedia.org/wiki/Foreach_loop)。给定要迭代的对象，它将每次迭代的值赋给变量:

>>>

```py
>>> people = ["Kevin", "Creed", "Jim"]
>>> for person in people:
...     print(f"{person} was in The Office.")
...
Kevin was in The Office.
Creed was in The Office.
Jim was in The Office.
```

在这个例子中，我们从人名列表(容器)开始。`for`循环从行首的`for`关键字开始，接着是为列表中的每个元素赋值的变量，然后是`in`关键字，最后是容器(`people`)。

Python 的`for`循环是任何 Python 程序的另一个主要成分。要了解更多关于`for`循环的信息，请查看[Python“for”循环(有限迭代)](https://realpython.com/python-for-loop/)。

#### `while`关键词

Python 的 [`while`循环](https://realpython.com/python-while-loop/)使用关键字 **`while`** ，工作方式类似于其他编程语言中的`while`循环。只要跟在`while`关键字后面的条件是真的，跟在`while`语句后面的代码块就会不断重复执行:

```py
while <expr>:
    <statements>
```

**注意:**对于下面的无限循环示例，如果您决定在自己的机器上尝试，请准备好使用 `Ctrl` + `C` 来停止该进程。

在 Python 中指定无限循环最简单的方法是使用`while`关键字和一个总是真实的表达式:

>>>

```py
>>> while True:
...     print("working...")
...
```

关于无限循环的更多例子，请查看 Python 中的[套接字编程(指南)](https://realpython.com/python-sockets/)。要了解更多关于`while`循环的信息，请查看[Python“while”循环(无限迭代)](https://realpython.com/python-while-loop/)。

#### `break`关键词

如果你需要提前退出一个循环，那么你可以使用 **`break`** 关键字。这个关键字在`for`和`while`循环中都有效:

```py
for <element> in <container>:
    if <expr>:
        break
```

使用`break`关键字的一个例子是，如果你对一列数字中的整数求和，当总数超过给定值时，你想退出:

>>>

```py
>>> nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> total_sum = 0
>>> for num in nums:
...     total_sum += num
...     if total_sum > 10:
...         break
...
>>> total_sum
15
```

Python 关键字`break`和`continue`在处理循环时都是有用的工具。要深入讨论它们的用法，请查看[Python“while”循环(无限迭代)](https://realpython.com/python-while-loop/#the-python-break-and-continue-statements)。如果您想探索`break`关键字的另一个用例，那么您可以学习[如何在 Python](https://realpython.com/python-do-while/) 中模拟 do-while 循环。

#### `continue`关键词

Python 还有一个 **`continue`** 关键字，用于当你想跳到下一个循环迭代时。与大多数其他编程语言一样，`continue`关键字允许您停止执行当前的循环迭代，并继续下一次迭代:

```py
for <element> in <container>:
    if <expr>:
        continue
```

`continue`关键字也适用于`while`循环。如果在一个循环中到达了`continue`关键字，那么当前的迭代停止，并开始循环的下一次迭代。

#### 与循环一起使用的`else`关键字

除了将`else`关键字用于条件`if`语句之外，您还可以将它用作循环的一部分。当与循环一起使用时， **`else`** 关键字指定如果循环正常退出时应该运行的代码，这意味着没有提前调用`break`来退出循环。

将`else`与`for`循环一起使用的语法如下所示:

```py
for <element> in <container>:
    <statements>
else:
    <statements>
```

这非常类似于在`if`语句中使用`else`。使用带有`while`循环的`else`看起来很相似:

```py
while <expr>:
    <statements>
else:
    <statements>
```

Python 标准文档中有一节是关于使用`break`和`else`的[,还有一个`for`循环](https://docs.python.org/3.3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops),你真应该去看看。它用一个很好的例子来说明`else`块的用处。

它显示的任务是在数字 2 到 9 之间循环寻找质数。有一种方法可以做到这一点，那就是使用带有**标志变量**的标准`for`循环:

>>>

```py
>>> for n in range(2, 10):
...     prime = True
...     for x in range(2, n):
...         if n % x == 0:
...             prime = False
...             print(f"{n} is not prime")
...             break
...     if prime:
...         print(f"{n} is prime!")
...
2 is prime!
3 is prime!
4 is not prime
5 is prime!
6 is not prime
7 is prime!
8 is not prime
9 is not prime
```

您可以使用`prime`标志来指示循环是如何退出的。如果它正常退出，那么`prime`标志保持`True`。如果用`break`退出，那么`prime`标志将被设置为`False`。一旦在内部的`for`循环之外，你可以检查这个标志来确定`prime`是否是`True`，如果是，打印出这个数字是质数。

`else`块提供了更简单的语法。如果您发现自己必须在一个循环中设置一个标志，那么可以将下一个示例视为一种潜在的简化代码的方法:

>>>

```py
>>> for n in range(2, 10):
...     for x in range(2, n):
...         if n % x == 0:
...             print(f"{n} is not prime")
...             break
...     else:
...         print(f"{n} is prime!")
...
2 is prime!
3 is prime!
4 is not prime
5 is prime!
6 is not prime
7 is prime!
8 is not prime
9 is not prime
```

在这个例子中，使用`else`块唯一需要做的事情就是删除`prime`标志，并用`else`块替换最后的`if`语句。这最终会产生与前面示例相同的结果，只是代码更清晰。

有时候在循环中使用`else`关键字看起来有点奇怪，但是一旦你明白它可以让你避免在循环中使用标志，它就会成为一个强大的工具。

[*Remove ads*](/account/join/)

### 结构关键词:`def`、`class`、`with`、`as`、`pass`、`lambda`、

为了[定义函数](https://realpython.com/defining-your-own-python-function/)和类或者使用[上下文管理器](https://dbader.org/blog/python-context-managers-and-with-statement)，您将需要使用本节中的一个 Python 关键字。它们是 Python 语言的重要组成部分，了解何时使用它们将有助于您成为更好的 Python 程序员。

#### `def`关键词

Python 的关键字 **`def`** 用于定义一个类的函数或方法。这相当于 JavaScript 和 PHP 中的`function`。用`def`定义函数的基本语法如下:

```py
def <function>(<params>):
    <body>
```

在任何 Python 程序中，函数和方法都是非常有用的结构。要了解更多关于定义它们的细节，请查看[定义自己的 Python 函数](https://realpython.com/defining-your-own-python-function/#function-calls-and-definition)。

#### `class`关键词

要在 Python 中定义一个类，可以使用`class`关键字。用 **`class`** 定义类的一般语法如下:

```py
class MyClass(<extends>):
    <body>
```

类是面向对象编程中的强大工具，您应该了解它们以及如何定义它们。要了解更多，请查看 Python 3 中的[面向对象编程(OOP)。](https://realpython.com/python3-object-oriented-programming/#how-to-define-a-class-in-python)

#### `with`关键词

在 Python 中，上下文管理器是一个非常有用的结构。每个上下文管理器在您指定的语句之前和之后执行特定的代码。要使用一个，你使用 **`with`** 关键字:

```py
with <context manager> as <var>:
    <statements>
```

使用`with`为您提供了一种方法来定义要在上下文管理器的[范围](https://realpython.com/python-scope-legb-rule/)内执行的代码。最基本的例子是当你在 Python 中使用[文件 I/O](https://realpython.com/read-write-files-python/) 时。

如果你想[打开一个文件](https://realpython.com/working-with-files-in-python/)，对该文件做些什么，然后确保该文件被正确关闭，那么你可以使用上下文管理器。考虑这个例子，其中`names.txt`包含一个名字列表，每行一个名字:

>>>

```py
>>> with open("names.txt") as input_file:
...    for name in input_file:
...        print(name.strip())
...
Jim
Pam
Cece
Philip
```

由`open()`提供并由 **`with`** 关键字启动的文件 I/O 上下文管理器打开文件进行读取，将打开的文件指针分配给`input_file`，然后执行您在`with`块中指定的任何代码。然后，在块被执行后，文件指针关闭。即使`with`块中的代码引发了异常，文件指针仍然会关闭。

关于使用`with`和上下文管理器的一个很好的例子，请查看 [Python 定时器函数:监控代码的三种方法](https://realpython.com/python-timer/#understanding-context-managers-in-python)。

#### `as`关键字与`with` 一起使用

如果你想访问传递给`with`的表达式或上下文管理器的结果，你需要用 **`as`** 给它起别名。您可能还见过用于别名导入和异常的`as`,这没有什么不同。别名在`with`块中可用:

```py
with <expr> as <alias>:
    <statements>
```

大多数时候，你会看到这两个 Python 关键字，`with`和`as`一起使用。

#### `pass`关键词

由于 Python 没有块指示符来指定块的结束，所以使用了 **`pass`** 关键字来指定该块故意留空。这相当于**不操作**，或者**不操作**。以下是使用`pass`指定块为空白的几个例子:

```py
def my_function():
    pass

class MyClass:
    pass

if True:
    pass
```

关于`pass`的更多信息，请查看[pass 语句:如何在 Python](https://realpython.com/python-pass/) 中不做任何事情。

#### `lambda`关键词

**`lambda`** 关键字用于定义一个没有名字，只有一条语句，返回结果的函数。用`lambda`定义的函数称为**λ函数**:

```py
lambda <args>: <statement>
```

一个计算参数的`lambda`函数的基本例子是这样的:

```py
p10 = lambda x: x**10
```

这相当于用`def`定义一个函数:

```py
def p10(x):
    return x**10
```

`lambda`函数的一个常见用途是为另一个函数指定不同的行为。例如，假设您想按整数值对字符串列表进行排序。 [`sorted()`](https://realpython.com/python-sort/) 的默认行为是将字符串按字母顺序排序。但是使用`sorted()`，你可以指定列表应该按照哪个键排序。

lambda 函数提供了一种很好的方式来实现这一点:

>>>

```py
>>> ids = ["id1", "id2", "id30", "id3", "id20", "id10"]
>>> sorted(ids)
['id1', 'id10', 'id2', 'id20', 'id3', 'id30']

>>> sorted(ids, key=lambda x: int(x[2:]))
['id1', 'id2', 'id3', 'id10', 'id20', 'id30']
```

此示例在将字符串转换为整数后，不是根据字母顺序，而是根据最后一个字符的数字顺序对列表进行排序。如果没有`lambda`，你将不得不定义一个函数，给它一个名字，然后把它传递给`sorted()`。`lambda`使这段代码更干净。

作为比较，这是上面的例子在没有使用`lambda`的情况下看起来的样子:

>>>

```py
>>> def sort_by_int(x):
...     return int(x[2:])
...
>>> ids = ["id1", "id2", "id30", "id3", "id20", "id10"]
>>> sorted(ids, key=sort_by_int)
['id1', 'id2', 'id3', 'id10', 'id20', 'id30']
```

这段代码产生与`lambda`示例相同的结果，但是您需要在使用它之前定义函数。

关于`lambda`的更多信息，请查看[如何使用 Python Lambda 函数](https://realpython.com/python-lambda/)。

[*Remove ads*](/account/join/)

### 返回关键词:`return`、`yield`、

有两个 Python 关键字用于指定从函数或方法返回什么:`return`和`yield`。理解何时何地使用`return`对于成为一名更好的 Python 程序员至关重要。`yield`关键字是 Python 的一个更高级的特性，但它也是理解的一个有用工具。

#### `return`关键词

Python 的 **`return`** 关键字只作为用`def`定义的函数的一部分有效。当 Python 遇到这个关键字时，它将在该点退出函数，并返回在`return`关键字之后的任何结果:

```py
def <function>():
    return <expr>
```

如果没有给定表达式，默认情况下，`return`将返回`None`:

>>>

```py
>>> def return_none():
...     return
...
>>> return_none()
>>> r = return_none()
>>> print(r)
None
```

但是，大多数情况下，您希望返回表达式或特定值的结果:

>>>

```py
>>> def plus_1(num):
...    return num + 1
...
>>> plus_1(9)
10
>>> r = plus_1(9)
>>> print(r)
10
```

你甚至可以在一个函数中多次使用`return`关键字。这允许您在函数中有多个出口点。当您希望有多个 return 语句时，一个经典的例子是下面的计算[阶乘](https://en.wikipedia.org/wiki/Factorial)的[递归](https://realpython.com/python-thinking-recursively/)解决方案:

```py
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)
```

在上面的阶乘函数中，有两种情况下你会想从函数中返回。第一种是基本情况，当数字为`1`时，第二种是常规情况，当您想要将当前数字乘以下一个数字的阶乘值时。

要了解更多关于关键字`return`的信息，请查看[定义你自己的 Python 函数](https://realpython.com/defining-your-own-python-function/#the-return-statement)。

#### `yield`关键词

Python 的 **`yield`** 关键字有点像`return`关键字，因为它指定了从函数返回什么。然而，当一个函数有一个`yield`语句时，返回的是一个**生成器**。然后可以将生成器传递给 Python 的内置`next()`来获取函数返回的下一个值。

当你用`yield`语句调用一个函数时，Python 会执行这个函数，直到它到达第一个`yield`关键字，然后返回一个生成器。这些被称为生成器函数:

```py
def <function>():
    yield <expr>
```

最简单的例子是返回相同值集的生成器函数:

>>>

```py
>>> def family():
...     yield "Pam"
...     yield "Jim"
...     yield "Cece"
...     yield "Philip"
...
>>> names = family()
>>> names
<generator object family at 0x7f47a43577d8>
>>> next(names)
'Pam'
>>> next(names)
'Jim'
>>> next(names)
'Cece'
>>> next(names)
'Philip'
>>> next(names)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

一旦`StopIteration`异常被引发，生成器就完成返回值。为了再次浏览这些名字，您需要再次调用`family()`并获得一个新的生成器。大多数情况下，一个生成器函数会作为一个`for`循环的一部分被调用，它会为您执行`next()`调用。

关于`yield`关键字和使用生成器和生成器函数的更多信息，请查看[如何在 Python 中使用生成器和屈服](https://realpython.com/introduction-to-python-generators/)和 [Python 生成器 101](https://realpython.com/courses/python-generators/) 。

[*Remove ads*](/account/join/)

### 导入关键词:`import`、`from`、`as`、

对于那些不像 Python 关键字和内置的工具，还不能用于您的 Python 程序，您需要将它们导入到您的程序中。Python 的标准库中有许多有用的模块，只需要导入即可。在 [PyPI](https://pypi.org/) 中还有许多其他有用的库和工具，一旦你把它们安装到你的环境中，你就需要把它们导入到你的程序中。

以下是用于将模块导入程序的三个 Python 关键字的简要描述。有关这些关键字的更多信息，请查看 [Python 模块和包——简介](https://realpython.com/python-modules-packages/#the-import-statement)和 [Python 导入:高级技术和技巧](https://realpython.com/python-import/)。

#### `import`关键词

Python 的`import`关键字用于导入或包含在 Python 程序中使用的模块。基本用法语法如下所示:

```py
import <module>
```

该语句运行后，`<module>`将可供您的程序使用。

例如，如果你想使用标准库中`collections`模块中的 [`Counter`](https://realpython.com/python-counter/) 类，那么你可以使用下面的代码:

>>>

```py
>>> import collections
>>> collections.Counter()
Counter()
```

以这种方式导入`collections`使得整个`collections`模块，包括`Counter`类，对您的程序可用。通过使用模块名，您可以访问该模块中所有可用的工具。要访问`Counter`，您需要从模块`collections.Counter`中引用它。

#### `from`关键词

**`from`** 关键字与`import`一起使用，从模块中导入特定的内容:

```py
from <module> import <thing>
```

这将把`<module>`中的`<thing>`导入到你的程序中。这两个 Python 关键字，`from`和`import`，一起使用。

如果您想使用标准库中`collections`模块中的`Counter`，那么您可以专门导入它:

>>>

```py
>>> from collections import Counter
>>> Counter()
Counter()
```

像这样导入`Counter`使得`Counter`类可用，但是来自`collections`模块的其他任何东西都不可用。`Counter`现已可用，无需从`collections`模块引用。

#### `as`关键词

**`as`** 关键字用于**别名**一个导入的模块或工具。它与 Python 关键字`import`和`from`一起使用，以更改正在导入的东西的名称:

```py
import <module> as <alias>
from <module> import <thing> as <alias>
```

对于名字很长或者有一个常用导入别名的模块，`as`有助于创建别名。

如果您想从 collections 模块导入`Counter`类，但将其命名为不同的名称，您可以通过使用`as`来为其起别名:

>>>

```py
>>> from collections import Counter as C
>>> C()
Counter()
```

现在`Counter`可以在你的程序中使用了，但是它被`C`引用了。`as`进口别名更常见的用法是用于 [NumPy](https://realpython.com/numpy-array-programming/) 或 [Pandas](https://realpython.com/learning-paths/pandas-data-science/) 包装。这些通常使用标准别名导入:

```py
import numpy as np
import pandas as pd
```

这是从一个模块中导入所有内容的一个更好的选择，它允许您缩短正在导入的模块的名称。

[*Remove ads*](/account/join/)

### 异常处理关键字:`try`、`except`、`raise`、`finally`、`else`、`assert`、

任何 Python 程序最常见的一个方面就是异常的引发和捕获。因为这是所有 Python 代码的一个基本方面，所以有几个 Python 关键字可以帮助您使代码的这一部分清晰简洁。

下面几节将介绍这些 Python 关键字及其基本用法。关于这些关键词的更深入的教程，请查看 [Python 异常:简介](https://realpython.com/python-exceptions/)。

#### `try`关键词

任何异常处理块都以 Python 的 **`try`** 关键字开始。这在大多数其他具有异常处理的编程语言中是相同的。

`try`块中的代码可能会引发异常。其他几个 Python 关键字与`try`相关联，用于定义如果出现不同的异常或在不同的情况下应该做什么。这些是`except`、`else`和`finally`:

```py
try:
    <statements>
<except|else|finally>:
    <statements>
```

一个`try`块是无效的，除非它在整个`try`语句中至少有一个用于异常处理的 Python 关键字。

如果您想要计算并返回每加仑汽油的英里数(`mpg`)，给定行驶的英里数和使用的加仑数，那么您可以编写如下函数:

```py
def mpg(miles, gallons):
    return miles / gallons
```

您可能看到的第一个问题是，如果将`gallons`参数作为`0`传入，您的代码可能会引发一个`ZeroDivisionError`。`try`关键字允许您修改上面的代码来适当地处理这种情况:

```py
def mpg(miles, gallons):
    try:
        mpg = miles / gallons
    except ZeroDivisionError:
        mpg = None
    return mpg
```

现在如果`gallons = 0`，那么`mpg()`不会引发异常，而是返回`None`。这可能更好，或者您可能决定要引发不同类型的异常或以不同的方式处理这种情况。您将在下面看到该示例的扩展版本，以说明用于异常处理的其他关键字。

#### `except`关键词

Python 的 **`except`** 关键字与`try`一起使用，定义当出现特定异常时做什么。你可以用一个`try`拥有一个或多个`except`区块。基本用法如下:

```py
try:
    <statements>
except <exception>:
    <statements>
```

以前面的`mpg()`为例，如果有人传递了不能与`/`操作符一起工作的类型，您也可以做一些特定的事情。在前面的例子中已经定义了`mpg()`，现在试着用字符串而不是数字来调用它:

>>>

```py
>>> mpg("lots", "many")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in mpg
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

您也可以修改`mpg()`并使用多个`except`块来处理这种情况:

```py
def mpg(miles, gallons):
    try:
        mpg = miles / gallons
    except ZeroDivisionError:
        mpg = None
    except TypeError as ex:
        print("you need to provide numbers")
        raise ex
    return mpg
```

在这里，您修改`mpg()`以仅在将有用的提醒打印到屏幕上之后引发`TypeError`异常。

请注意，`except`关键字也可以与`as`关键字结合使用。这与`as`的其他用法效果相同，给引发的异常一个别名，这样您就可以在`except`块中使用它。

尽管在语法上是允许的，但是尽量不要使用`except`语句作为隐式的捕获。更好的做法是总是明确地捕捉*的某个东西*，即使它只是`Exception`:

```py
try:
    1 / 0
except:  # Don't do this
    pass

try:
    1 / 0
except Exception:  # This is better
    pass

try:
    1 / 0
except ZeroDivisionError:  # This is best
    pass
```

如果你真的想捕捉大范围的异常，那么指定父类`Exception`。这是一个更明确的总括，它不会捕捉你可能不想捕捉的异常，如`RuntimeError`或`KeyboardInterrupt`。

#### `raise`关键词

**`raise`** 关键字引发了一个异常。如果您发现您需要引发一个异常，那么您可以使用`raise`,后跟要引发的异常:

```py
raise <exception>
```

在前面的`mpg()`示例中，您使用了`raise`。当您捕捉到`TypeError`时，您会在屏幕上显示一条消息后再次引发该异常。

#### `finally`关键词

Python 的 **`finally`** 关键字有助于指定无论在`try`、`except`或`else`块中发生什么都应该运行的代码。要使用`finally`，将其作为`try`块的一部分，并指定无论如何都要运行的语句:

```py
try:
    <statements>
finally:
    <statements>
```

使用前面的例子，指定无论发生什么情况，您都想知道函数是用什么参数调用的，这可能是有帮助的。您可以修改`mpg()`来包含一个`finally`块来实现这个功能:

```py
def mpg(miles, gallons):
    try:
        mpg = miles / gallons
    except ZeroDivisionError:
        mpg = None
    except TypeError as ex:
        print("you need to provide numbers")
        raise ex
    finally:
        print(f"mpg({miles}, {gallons})")
    return mpg
```

现在，无论如何调用`mpg()`或者结果是什么，都要打印用户提供的参数:

>>>

```py
>>> mpg(10, 1)
mpg(10, 1)
10.0

>>> mpg("lots", "many")
you need to provide numbers
mpg(lots, many)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 8, in mpg
  File "<stdin>", line 3, in mpg
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

关键字`finally`可能是异常处理代码中非常有用的一部分。

#### `else`关键字与`try`和`except`T3 一起使用

您已经了解到 **`else`** 关键字可以与`if`关键字和 Python 中的循环一起使用，但是它还有一个用途。它可以与`try`和`except` Python 关键字结合使用。只有当您同时使用至少一个`except`模块时，您才能以这种方式使用`else`:

```py
try:
    <statements>
except <exception>:
    <statements>
else:
    <statements>
```

在这种情况下，`else`块中的代码仅在`try`块中出现异常*而非*时执行。换句话说，如果`try`块成功执行了所有代码，那么`else`块代码将被执行。

在`mpg()`的例子中，假设您想要确保无论传入什么数字组合，结果`mpg`总是作为`float`返回。你可以这样做的方法之一是使用一个`else`块。如果`mpg`的`try`块计算成功，那么在返回之前将结果转换成`else`块中的`float`:

```py
def mpg(miles, gallons):
    try:
        mpg = miles / gallons
    except ZeroDivisionError:
        mpg = None
    except TypeError as ex:
        print("you need to provide numbers")
        raise ex
    else:
        mpg = float(mpg) if mpg is not None else mpg
    finally:
        print(f"mpg({miles}, {gallons})")
    return mpg
```

现在，调用`mpg()`的结果，如果成功，将总是一个`float`。

关于使用`else`块作为`try`和`except`块的一部分的更多信息，请查看 [Python 异常:简介](https://realpython.com/python-exceptions/#the-else-clause)。

#### `assert`关键词

Python 中的 **`assert`** 关键字用于指定一个 [`assert`语句](https://realpython.com/python-assert-statement/)，或者一个关于表达式的断言。如果表达式(`<expr>`)为真，一个`assert`语句将导致一个 no-op，如果表达式为假，它将引发一个`AssertionError`。要定义断言，使用`assert`后跟一个表达式:

```py
assert <expr>
```

一般来说，`assert`语句会被用来确定某件需要为真的事情。但是，您不应该依赖它们，因为根据 Python 程序的执行方式，它们可以被忽略。

[*Remove ads*](/account/join/)

### 异步编程关键词:`async`、`await`、

异步编程是一个复杂的话题。定义了两个 Python 关键字来帮助提高异步代码的可读性和整洁性:`async`和`await`。

下面几节将介绍这两个异步关键字及其基本语法，但它们不会深入异步编程。要了解更多关于异步编程的知识，请查看[Python 中的异步 IO:完整演练](https://realpython.com/async-io-python/)和[Python 中的异步特性入门](https://realpython.com/python-async-features/)。

#### `async`关键词

**`async`** 关键字与`def`一起使用，定义一个异步函数，或[协程](https://realpython.com/async-io-python/#the-asyncawait-syntax-and-native-coroutines)。语法就像定义一个函数，只是在开头添加了`async`:

```py
async def <function>(<params>):
    <statements>
```

您可以通过在函数的常规定义前添加关键字`async`来使函数异步。

#### `await`关键词

Python 的 **`await`** 关键字在异步函数中用于指定函数中的一个点，在这个点上，控制权被交还给事件循环，以供其他函数运行。您可以通过将`await`关键字放在任何`async`函数的调用之前来使用它:

```py
await <some async function call>
# OR
<var> = await <some async function call>
```

使用`await`时，可以调用异步函数并忽略结果，也可以在函数最终返回时将结果存储在一个变量中。

### 变量处理关键字:`del`、`global`、`nonlocal`、

三个 Python 关键字用于处理变量。`del`关键字比`global`和`nonlocal`关键字更常用。但是知道并理解这三个关键词仍然是有帮助的，这样你就可以确定何时以及如何使用它们。

#### `del`关键词

**`del`** 在 Python 中用于取消设置变量或名称。您可以在变量名上使用它，但更常见的用途是从[列表](https://realpython.com/python-lists-tuples/#python-lists)或[字典](https://realpython.com/python-dicts/)中移除索引。要取消设置一个变量，使用`del`,后跟您想要取消设置的变量:

```py
del <variable>
```

让我们假设您想要清理一个从 API 响应中得到的字典，方法是扔掉您知道不会使用的键。您可以使用关键字`del`来实现:

>>>

```py
>>> del response["headers"]
>>> del response["errors"]
```

这将从字典`response`中删除`"headers"`和`"errors"`键。

#### `global`关键词

如果你需要修改一个没有在函数中定义但是在**全局范围**中定义的变量，那么你需要使用 **`global`** 关键字。这是通过在函数中指定需要从全局范围将哪些变量拉入函数来实现的:

```py
global <variable>
```

一个基本的例子是用函数调用增加一个全局变量。你可以用关键字`global`来实现:

>>>

```py
>>> x = 0
>>> def inc():
...     global x
...     x += 1
...
>>> inc()
>>> x
1
>>> inc()
>>> x
2
```

这通常不被认为是好的做法，但是它确实有它的用处。要了解更多关于`global`关键字的信息，请查看[Python Scope&LEGB 规则:解析代码中的名称](https://realpython.com/python-scope-legb-rule/#the-global-statement)。

#### `nonlocal`关键词

**`nonlocal`** 关键字与`global`相似，它允许您修改不同范围的变量。有了`global`，你从中提取的范围就是全局范围。对于`nonlocal`，您从中提取的作用域是**父作用域**。语法类似于`global`:

```py
nonlocal <variable>
```

这个关键字不常使用，但有时会很方便。关于作用域和`nonlocal`关键字的更多信息，请查看 [Python 作用域&LEGB 规则:解析代码中的名称](https://realpython.com/python-scope-legb-rule/#the-nonlocal-statement)。

[*Remove ads*](/account/join/)

## 弃用的 Python 关键字

有时，Python 关键字会成为一个内置函数。`print`和`exec`都是这种情况。在 2.7 版本中，这些曾经是 Python 关键字，但后来被改为内置函数。

### 前`print`关键词

当 **`print`** 是一个关键字时，打印到屏幕上的语法如下:

```py
print "Hello, World"
```

请注意，它看起来像许多其他关键字语句，关键字后跟参数。

现在`print`已经不是关键字了，打印是用内置的`print()`完成的。要将某些内容打印到屏幕上，现在可以使用以下语法:

```py
print("Hello, World")
```

关于打印的更多信息，请查看 Python print()函数的指南。

### 前`exec`关键词

在 Python 2.7 中， **`exec`** 关键字将 Python 代码作为字符串执行。这是使用以下语法完成的:

```py
exec "<statements>"
```

您可以在 Python 3+中获得相同的行为，只是使用了内置的`exec()`。例如，如果您想在 Python 代码中执行`"x = 12 * 7"`，那么您可以执行以下操作:

>>>

```py
>>> exec("x = 12 * 7")
>>> x == 84
True
```

关于`exec()`及其用途的更多信息，请查看[如何运行您的 Python 脚本](https://realpython.com/run-python-scripts/#hacking-exec)和 [Python 的 exec():执行动态生成的代码](https://realpython.com/python-exec/)。

## 结论

Python 关键字是任何 Python 程序的基本构件。理解它们的正确用法是提高您的 Python 技能和知识的关键。

在整篇文章中，您看到了一些巩固您对 Python 关键字的理解并帮助您编写更高效和可读性更好的代码的东西。

**在这篇文章中，你已经了解到:**

*   3.8 版本中的 **Python 关键字**及其基本用法
*   几个**资源**帮助你加深对许多关键词的理解
*   如何使用 Python 的 **`keyword`模块**以编程的方式处理关键字

如果你理解了这些关键词中的大部分，并能自如地使用它们，那么你可能会有兴趣了解更多关于 [Python 的语法](https://realpython.com/cpython-source-code-guide/#grammar)以及使用这些关键词的[语句](https://docs.python.org/3/reference/compound_stmts.html)是如何被指定和构造的。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**探索 Python 中的关键词**](/courses/exploring-keywords-in-python/)*************