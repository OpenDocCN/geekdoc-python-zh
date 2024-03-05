# Python 布尔值:用真值优化代码

> 原文：<https://realpython.com/python-boolean/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python 布尔:利用真值**](/courses/booleans-leveraging-truth/)

**Python 布尔**类型是 Python 的[内置数据类型](https://realpython.com/python-data-types/)之一。它用来表示一个表达式的真值。例如，表达式`1 <= 2`是`True`，而表达式`0 == 1`是`False`。理解 Python 布尔值的行为对于用 Python 很好地编程是很重要的。

**在本教程中，您将学习如何:**

*   用**布尔运算符**操作布尔值
*   将布尔值转换为其他类型
*   将其他类型转换为 **Python 布尔值**
*   使用 Python 布尔值编写**高效可读的** Python 代码

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 布尔类型

Python 布尔类型只有两个可能的值:

1.  `True`
2.  `False`

其他值都不会将`bool`作为其类型。您可以通过内置的`type()`检查`True`和`False`的类型:

>>>

```py
>>> type(False)
<class 'bool'>
>>> type(True)
<class 'bool'>
```

`False`和`True`的`type()`都是`bool`。

类型`bool`是中内置的**，这意味着它在 Python 中总是可用的，不需要导入。然而，名字本身并不是语言中的关键字。虽然以下被认为是不好的风格，但是可以指定名称为`bool`:**

>>>

```py
>>> bool
<class 'bool'>
>>> bool = "this is not a type"
>>> bool
'this is not a type'
```

尽管技术上可行，但为了避免混淆，强烈建议您不要给`bool`分配不同的值。

[*Remove ads*](/account/join/)

### Python 布尔作为关键字

内置名称不是关键字。就 Python 语言而言，它们是常规的[变量](https://realpython.com/python-variables/)。如果您给它们赋值，那么您将覆盖内置值。

相比之下，`True`和`False`这两个名字是*而不是*内置的。它们是**关键词**。与其他许多 [Python 关键字](https://realpython.com/python-keywords/)不同，`True`和`False`都是 Python **表达式**。因为它们是表达式，所以它们可以用在任何其他表达式可以用的地方，比如`1 + 1`。

可以给变量赋值，但是不能给`True`赋值:

>>>

```py
>>> a_true_alias = True
>>> a_true_alias
True
>>> True = 5
  File "<stdin>", line 1
SyntaxError: cannot assign to True
```

因为`True`是一个关键字，所以不能给它赋值。同样的规则也适用于`False`:

>>>

```py
>>> False = 5
  File "<stdin>", line 1
SyntaxError: cannot assign to False
```

不能赋值给`False`，因为它是 Python 中的一个关键字。这样，`True`和`False`的行为就像其他数值常量一样。例如，您可以将`1.5`传递给函数或将其赋给变量。然而，不可能给`1.5`赋值。语句`1.5 = 5`不是有效的 Python。`1.5 = 5`和`False = 5`都是无效的 Python 代码，解析时会抛出 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) 。

### Python 布尔值作为数字

在 Python 中，布尔值被认为是一种**数字**类型。这意味着它们实际上是[号](https://realpython.com/python-numbers/)。换句话说，您可以对布尔值应用算术运算，也可以将它们与数字进行比较:

>>>

```py
>>> True == 1
True
>>> False == 0
True
>>> True + (False / True)
1.0
```

布尔值的数字性质没有太多用途，但是有一种技术可能会对您有所帮助。因为`True`等于`1`，`False`等于`0`，所以将布尔值相加是一种快速计算`True`值个数的方法。当您需要计算满足某个条件的项目数量时，这很方便。

例如，如果你想分析一首[经典儿童诗](https://www.poetryfoundation.org/poems/42916/jabberwocky)中的一节，看看有多少行包含单词`"the"`，那么`True`等于`1`和`False`等于`0`这个事实就非常方便了:

>>>

```py
>>> lines="""\
... He took his vorpal sword in hand;
...       Long time the manxome foe he sought—
... So rested he by the Tumtum tree
...       And stood awhile in thought.
... """.splitlines()
>>> sum("the" in line.lower() for line in lines) / len(lines)
0.5
```

像这样对[生成器表达式](https://realpython.com/lessons/map-function-vs-generator-expressions/)中的所有值求和，可以让你知道`True`在生成器中出现了多少次。以不区分大小写的方式，`True`在生成器中的次数等于包含单词`"the"`的行数。将这个数字除以总行数，得到匹配行数与总行数的比率。

要了解为什么会这样，您可以将上面的代码分成几个小部分:

>>>

```py
>>> lines = """\
... He took his vorpal sword in hand;
...       Long time the manxome foe he sought—
... So rested he by the Tumtum tree
...       And stood awhile in thought.
... """
>>> line_list = lines.splitlines()
>>> "the" in line_list[0]
False
>>> "the" in line_list[1]
True
>>> 0 + False + True # Equivalent to 0 + 0 + 1
1
>>> ["the" in line for line in line_list]
[False, True, True, False]
>>> False + True + True + False
2
>>> len(line_list)
4
>>> 2/4
0.5
```

`line_list`变量保存一个行列表。第一行没有单词`"the"`，所以`"the" in line_list[0]`是`False`。在第二行中，`"the"`确实出现了，所以`"the" in line_list[1]`就是`True`。既然布尔是数字，就可以把它们加到数字上，`0 + False + True`给出`1`。

由于`["the" in line for line in line_list]`是四个布尔的列表，所以可以把它们加在一起。当你加上`False + True + True + False`，你得到`2`。现在，如果你把结果除以列表的长度`4`，你得到`0.5`。单词`"the"`出现在所选内容的一半行中。这是利用布尔是数字这一事实的一种有用方式。

## 布尔运算符

布尔运算符是那些接受**布尔输入**并返回**布尔结果**的运算符。

**注意:**稍后，您将看到这些操作符可以被赋予其他输入，并且不总是返回布尔结果。现在，所有的例子都将使用布尔输入和结果。在[真实性](#python-boolean-testing)一节中，你会看到这是如何推广到其他值的。

因为 Python 布尔值只有两个可能的选项，即`True`或`False`，所以可以完全根据运算符分配给每个可能的输入组合的结果来指定运算符。这些规格被称为**真值表**，因为它们显示在一个表格中。

稍后您会看到，在某些情况下，知道一个操作符的输入就足以确定它的值。在这些情况下，其他输入是*而不是*被评估。这被称为**短路评估**。

短路评估的重要性取决于具体案例。在某些情况下，它可能对您的程序没有什么影响。在其他情况下，例如当计算不影响结果的表达式时，它提供了显著的性能优势。在最极端的情况下，代码的正确性可能取决于短路评估。

[*Remove ads*](/account/join/)

### 没有输入的运算符

您可以将`True`和`False`视为没有输入的布尔运算符。其中一个操作符总是返回`True`，另一个总是返回`False`。

将 Python 布尔值视为运算符有时很有用。例如，这种方法有助于提醒你它们不是变量。出于同样的原因，你不能分配给`+`，也不可能分配给`True`或`False`。

只有两个 Python 布尔值存在。没有输入的布尔运算符总是返回相同的值。正因为如此，`True`和`False`是仅有的两个不接受输入的布尔运算符。

### `not`布尔运算符

唯一有一个自变量的布尔运算符是 [`not`](https://realpython.com/python-not-operator/) 。它接受一个参数并返回相反的结果:`True`的`False`和`False`的`True`。这是一个真值表:

| `A` | `not A` |
| --- | --- |
| `True` | `False` |
| `False` | `True` |

该表说明了`not`返回自变量的相反真值。因为`not`只有一个参数，所以它不会短路。它在返回结果之前评估其参数:

>>>

```py
>>> not True
False
>>> not False
True
>>> def print_and_true():
...     print("I got called")
...     return True
...
>>> not print_and_true()
I got called
False
```

最后一行显示`not`在返回`False`之前评估其输入。

您可能想知道为什么没有其他接受单个参数的布尔运算符。为了理解其中的原因，您可以查看一个表，该表显示了所有理论上可能的布尔运算符，这些运算符只接受一个参数:

| `A` | `not A` | 身份 | 是 | 不 |
| --- | --- | --- | --- | --- |
| `True` | `False` | `True` | `True` | `False` |
| `False` | `True` | `False` | `True` | `False` |

一个参数只有四种可能的运算符。除了`not`之外，其余三个操作符都有一些古怪的名字，因为它们实际上并不存在:

*   **`Identity`** :因为这个操作符只是返回它的输入，所以你可以把它从你的代码中删除而不会有任何影响。

*   **`Yes`** :这是一个短路运算符，因为它不依赖于它的自变量。你可以用`True`代替它，得到同样的结果。

*   **`No`** :这是另一个短路运算符，因为它不依赖于它的自变量。你可以用`False`代替它，得到同样的结果。

只有一个参数的其他可能的操作符都没有用。

### `and`布尔运算符

[`and`](https://realpython.com/python-and-operator/) 运算符有两个参数。除非两个输入都是`True`，否则计算结果为`False`。您可以用下面的真值表定义`and`的行为:

| `A` | `B` | `A and B` |
| --- | --- | --- |
| `True` | `True` | `True` |
| `False` | `True` | `False` |
| `True` | `False` | `False` |
| `False` | `False` | `False` |

此表很冗长。但是，它说明了与上面描述相同的行为。如果`A`是`False`，那么`B`的值无关紧要。因此，如果第一个输入是`False`，则`and`会短路。换句话说，如果第一个输入是`False`，那么第二个输入不会被计算。

下面的代码有第二个输入，它有一个[副作用](https://realpython.com/defining-your-own-python-function/#side-effects)，打印，为了提供一个具体的例子:

>>>

```py
>>> def print_and_return(x):
...     print(f"I am returning {x}")
...     return x
...
>>> True and print_and_return(True)
I am returning True
True
>>> True and print_and_return(False)
I am returning False
False
>>> False and print_and_return(True)
False
>>> False and print_and_return(False)
False
```

在最后两种情况下，不打印任何内容。这个函数没有被调用，因为调用它不需要确定`and`操作符的值。当表情有副作用时，意识到短路是很重要的。在最后两个例子中，短路评估防止了印刷副作用的发生。

这种行为至关重要的一个例子是在可能引发异常的代码中:

>>>

```py
>>> def inverse_and_true(n):
...     1 // n
...     return True
...
>>> inverse_and_true(5)
True
>>> inverse_and_true(0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in inverse_and_true
ZeroDivisionError: integer division or modulo by zero
>>> False and inverse_and_true(0)
False
```

函数`inverse_and_true()`被公认是愚蠢的，许多 [linters](https://realpython.com/python-code-quality/) 会警告表达式`1 // n`是无用的。当给定`0`作为参数时，它确实达到了整齐失败的目的，因为除以`0`是无效的。然而，最后一行没有引发异常。由于短路计算，函数没有被调用，没有发生`0`的除法运算，也没有引发异常。

相反，`True and inverse_and_true(0)`会引发一个异常。在这种情况下，`and`的结果需要第二个输入的值。一旦评估了第二个输入，就会调用`inverse_and_true(0)`，它会除以`0`，并引发一个异常。

[*Remove ads*](/account/join/)

### `or`布尔运算符

[`or`操作符](https://realpython.com/python-or-operator/)的值是`True`，除非其两个输入都是`False`。`or`运算符也可以由以下真值表定义:

| `A` | `B` | `A or B` |
| --- | --- | --- |
| `True` | `True` | `True` |
| `False` | `True` | `True` |
| `True` | `False` | `True` |
| `False` | `False` | `False` |

这个表很冗长，但是它和上面的解释有相同的意思。

非正式使用时，单词*或*可能有两种意思:

*   [**独占*或***](https://en.wikipedia.org/wiki/Exclusive_or) 就是*或*在短语“你可以申请延期或按时提交作业”中的用法在这种情况下，你不能既申请延期又按时提交作业。

*   [**包括*或***](https://en.wikipedia.org/wiki/Logical_disjunction) 有时用连词*和/或*表示。例如，“如果你在这项任务中表现出色，那么你可以加薪和/或升职”意味着你可能会加薪和升职。

当 Python 解释关键字`or`时，它使用包含的*或*来完成。如果两个输入都是`True`，那么`or`的结果就是`True`。

因为使用了包含的*或*，Python 中的`or`运算符也使用了短路求值。如果第一个参数是`True`，那么结果就是`True`，不需要对第二个参数求值。以下示例演示了`or`的短路评估:

>>>

```py
>>> def print_and_true():
...     print("print_and_true called")
...     return True
...
>>> True or print_and_true()
True
>>> False or print_and_true()
print_and_true called
True
```

第二个输入不会被`or`评估，除非第一个输入是`False`。在实践中，`or`的短路评估比`and`的少得多。然而，在阅读代码时记住这种行为是很重要的。

### 其他布尔运算符

布尔逻辑的数学理论决定了除了`not`、`and`、`or`之外不需要其他运算符。两个输入上的所有其他运算符都可以根据这三个运算符来指定。三个或三个以上输入的所有运算符都可以用两个输入的运算符来表示。

事实上，即使同时拥有`or`和`and`也是多余的。`and`算子可以用`not`和`or`来定义，`or`算子可以用`not`和`and`来定义。然而，`and`和`or`太有用了，所有编程语言都有这两个。

有十六种可能的双输入布尔运算符。除了`and`和`or`，实际中很少需要。正因为如此，`True`、`False`、`not`、`and`和`or`是唯一内置的 Python 布尔运算符。

## 比较运算符

Python 的一些操作符检查两个对象之间的关系是否成立。由于这种关系要么成立，要么不成立，这些被称为**比较操作符**的操作符总是返回布尔值。

比较运算符是布尔值最常见的来源。

### 平等与不平等

最常见的比较运算符是**相等运算符(`==` )** 和**不等运算符(`!=` )** 。如果不使用这些操作符中的至少一个，几乎不可能编写出任何有意义的 Python 代码。

等式运算符(`==`)是 Python 代码中使用最多的运算符之一。您经常需要将一个未知结果与一个已知结果进行比较，或者将两个未知结果进行比较。一些函数返回的值需要与一个[标记](https://en.wikipedia.org/wiki/Sentinel_value)进行比较，以查看是否检测到某种边缘条件。有时你需要比较两个函数的结果。

等号运算符通常用于比较数字:

>>>

```py
>>> 1 == 1
True
>>> 1 == 1.0
True
>>> 1 == 2
False
```

你可能以前用过[等式操作符](https://realpython.com/python-is-identity-vs-equality/)。它们是 Python 中最常见的一些操作符。对于所有内置的 Python 对象，以及大多数第三方类，它们返回一个**布尔**值:`True`或`False`。

**注意:**Python 语言并不强制`==`和`!=`返回布尔值。像 [NumPy](https://realpython.com/numpy-array-programming/) 和 [pandas](https://realpython.com/pandas-python-explore-dataset/) 这样的库返回其他值。

受欢迎程度仅次于等式运算符的是**不等式**运算符(`!=`)。如果参数不相等，则返回`True`，如果相等，则返回`False`。这些例子同样范围广泛。许多[单元测试](https://realpython.com/python-testing/)检查该值不等于特定的无效值。在尝试替代方法之前，web 客户端可能会检查错误代码是否为`404 Not Found`。

以下是使用 Python 不等式运算符的两个示例:

>>>

```py
>>> 1 != 2
True
>>> 1 != (1 + 0.0)
False
```

关于 Python 不等式操作符最令人惊讶的事情可能是它首先存在的事实。毕竟，你可以用`not (1 == 2)`达到和`1 != 2`一样的效果。Python 通常避免额外的语法，尤其是额外的核心操作符，因为用其他方法很容易实现。

然而，不平等是如此频繁地使用，它被认为是值得有一个专门的运营商。在 Python 的旧版本中，在`1.x`系列中，实际上有*两种*不同的语法。

作为一个愚人节玩笑，Python 仍然支持不平等的另一种语法和正确的`__future__`导入:

>>>

```py
>>> from __future__ import barry_as_FLUFL
>>> 1 <> 2
True
```

这个*不应该在任何真正使用的代码中使用*。不过，它可能会在您的下一个 Python 知识之夜派上用场。

[*Remove ads*](/account/join/)

### 订单比较

另一组测试操作符是**顺序**比较操作符。有四种顺序比较运算符，可以按两种性质进行分类:

*   **方向**:小于还是大于？
*   **严格**:是否允许平等？

因为这两个选择是独立的，所以得到了`2 * 2 == 4`顺序比较运算符。下表列出了所有四种类型:

|  | 不到 | 大于 |
| --- | --- | --- |
| 严格的 | `<` | `>` |
| 不严格 | `<=` | `>=` |
|  |  |  |

方向有两个选项，严格也有两个选项。这导致总共四个顺序比较运算符。

没有为所有对象定义顺序比较运算符。一些对象没有有意义的顺序。尽管[列表和元组](https://realpython.com/python-lists-tuples/)按字典顺序**，[字典](https://realpython.com/courses/dictionaries-python/)没有有意义的顺序:**

**>>>

```py
>>> {1: 3} < {2: 4}
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: '<' not supported between instances of 'dict' and 'dict'
```

字典应该如何排序并不明显。按照 Python 的[禅，面对歧义，Python 拒绝猜测。](https://www.python.org/dev/peps/pep-0020/)

虽然[字符串](https://realpython.com/python-strings/)和[整数](https://realpython.com/python-numbers/#integers)是分开排序的，但是不支持类型间比较:

>>>

```py
>>> 1 <= "1"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: '<=' not supported between instances of 'int' and 'str'
```

同样，由于没有明显的方法来定义顺序，Python 拒绝比较它们。这类似于加法运算符(`+`)。虽然可以将字符串和整数相加，但是将字符串和整数相加会引发异常。

当顺序比较运算符*被定义*时，通常它们返回一个布尔值。

**注意** : Python 并不强制比较运算符返回布尔值。虽然所有内置 Python 对象和大多数第三方对象在比较时都返回布尔值，但也有例外。

例如，NumPy 数组或 [pandas 数据帧](https://realpython.com/pandas-dataframe/)之间的比较运算符返回数组和数据帧。在本教程的后面，您将看到更多关于 NumPy 和布尔值的交互。

在 Python 中比较数字是检查边界条件的一种常用方法。注意`<`不允许相等，而`<=`允许:

>>>

```py
>>> 1 <= 1
True
>>> 1 < 1
False
>>> 2 > 3
False
>>> 2 >= 2
True
```

程序员经常使用比较运算符，却没有意识到它们会返回一个 Python 布尔值。

### `is`操作员

[`is`操作符](https://realpython.com/lessons/is-operator/)检查**对象标识**。换句话说，只有当`x`和`y`对同一个对象求值时，`x is y`才会对`True`求值。`is`操作符有一个相反的操作符`is not`。

`is`和`is not`的典型用法是比较列表的同一性:

>>>

```py
>>> x = []
>>> y = []
>>> x is x
True
>>> x is not x
False
>>> x is y
False
>>> x is not y
True
```

即使`x == y`，它们也不是同一个对象。`is not`操作符总是返回与`is`相反的结果。除了可读性，表达式`x is not y`和表达式`not (x is y)`没有区别。

记住，上面的例子显示了仅用于列表的`is`操作符。`is`操作符对[不可变](https://realpython.com/courses/immutability-python/)对象(如数字和字符串)的行为是[更复杂](https://realpython.com/python-is-identity-vs-equality/#when-only-some-integers-are-interned)。

[*Remove ads*](/account/join/)

### `in`操作员

`in`操作员检查**的成员资格**。对象可以定义它所认为的成员。大多数序列(如列表)都将其元素视为成员:

>>>

```py
>>> small_even = [2, 4]
>>> 1 in small_even
False
>>> 2 in small_even
True
>>> 10 in small_even
False
```

由于`2`是列表的一个元素，`2 in small_even`返回`True`。由于`1`和`10`不在列表中，其他表达式返回`False`。在所有情况下，`in`操作符都返回一个布尔值。

由于字符串是字符序列，您可能希望它们也检查成员资格。换句话说，作为字符串成员的字符将为`in`返回`True`，而不是字符串成员的字符将返回`False`:

>>>

```py
>>> "e" in "hello beautiful world"
True
>>> "x" in "hello beautiful world"
False
```

因为`"e"`是字符串的第二个元素，所以第一个例子返回`True`。由于`x`没有出现在字符串中，第二个例子返回`False`。但是，与单个字符一样，子字符串也被视为字符串的成员:

>>>

```py
>>> "beautiful" in "hello beautiful world"
True
>>> "belle" in "hello beautiful world"
False
```

因为`"beautiful"`是一个子串，所以`in`操作符返回`True`。因为`"belle"`不是子串，所以`in`操作符返回`False`。尽管事实上`"belle"`中的每个字母都是字符串的一员。

与运算符`is`和`==`一样，`in`运算符也有一个对立面`not in`。你可以使用`not in`来确认一个元素不是一个对象的成员。

### 链接比较运算符

比较运算符可以形成**链**。通过用比较运算符分隔表达式以形成一个更大的表达式，可以创建比较运算符链:

>>>

```py
>>> 1 < 2 < 3
True
```

表达式`1 < 2 < 3`是一个比较运算符链。它包含由比较运算符分隔的表达式。结果是`True`，因为链的两个部分都是`True`。你可以拆开链条，看看它是如何工作的:

>>>

```py
>>> 1 < 2 and 2 < 3
True
```

由于`1 < 2`返回`True`，`2 < 3`返回`True`，`and`返回`True`。比较链相当于在它的所有链接上使用`and`。在这种情况下，由于`True and True`返回`True`，所以整个链的结果是`True`。这意味着如果任何一个环节是`False`，那么整个链就是`False`:

>>>

```py
>>> 1 < 3 < 2
False
```

这个比较链返回`False`，因为不是所有的链接都是`True`。因为比较链是一个隐式的`and`运算符，如果甚至一个环节是`False`，那么整个链就是`False`。你可以拆开链条，看看它是如何工作的:

>>>

```py
>>> 1 < 3 and 3 < 2
False
```

在这种情况下，链的各部分计算为以下布尔值:

*   `1 < 3`是`True`
*   `3 < 2`是`False`

这意味着结果一个是`True`，一个是`False`。由于`True and False`等于`False`，整个链条的价值就是`False`。

只要类型可以比较，就可以在比较链中混合使用类型和操作:

>>>

```py
>>> 1 < 2 < 1
False
>>> 1 == 1.0 < 0.5
False
>>> 1 == 1.0 == True
True
>>> 1 < 3 > 2
True
>>> 1 < 2 < 3 < 4 < 5
True
```

运营商不一定都是一样的。甚至类型也不必完全相同。在上面的例子中，有三种数值类型:

1.  `int`
2.  `float`
3.  `bool`

这是三种不同的数值类型，但是您可以毫无问题地比较不同数值类型的对象。

#### 短路链评估

如果链使用隐式`and`，那么链也必须短路。这很重要，因为即使在没有定义顺序比较的情况下，链也有可能返回`False`:

>>>

```py
>>> 2 < "2"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: '<' not supported between instances of 'int' and 'str'
>>> 3 < 2 < "2"
False
```

尽管 Python 不能对整数和字符串数字进行顺序比较，但是`3 < 2 < "2"`的计算结果是`False`，因为它不计算第二次比较。在这种情况下，短路评估防止了另一个副作用:引发异常。

比较链的短路评估可以防止其他异常:

>>>

```py
>>> 3 < 2 < (1//0)
False
```

用`1`除以`0`会得到一个`ZeroDivisionError`。但是因为短路求值，Python 并不对无效除法求值。这意味着 Python 不仅跳过了对比较的评估，还跳过了对比较的输入的评估。

理解比较链的另一个重要方面是，当 Python *对链中的元素求值时，它只求值一次:*

>>>

```py
>>> def foo():
...     print("I'm foo")
...     return 1
...
>>> 0 < foo() < 2
I'm foo
True
>>> (0 < foo()) and (foo() < 2)
I'm foo
I'm foo
True
```

因为中间的元素只计算一次，所以重构`x < y < z`到`(x < y) and (y < z)`并不总是安全的。尽管链在短路评估中表现得像`and`，但它只评估所有值一次，包括中间值。

链对于**范围检查**特别有用，它确认一个值落在给定的范围内。例如，在包含工作小时数的每日发票中，您可以执行以下操作:

>>>

```py
>>> hours_worked = 5
>>> 1 <= hours_worked <= 25
True
```

如果工作了`0`个小时，那么就没有理由发送发票。算上夏令时，一天的最大小时数是`25`。上述范围检查确认一天的工作小时数在允许的范围内。

#### 混合运算符和链接

到目前为止，我们所有的例子都涉及到`==`、`!=`和顺序比较。但是，您可以链接 Python 的所有比较操作符。这可能会导致令人惊讶的行为:

>>>

```py
>>> a = 0
>>> a is a < 1
True
>>> (a is a) < 1
False
>>> a is (a < 1)
False
```

因为`a is a < 1`是比较链，所以求值为`True`。你可以把链条分成几部分:

*   表达式`a is a`是`True`，因为它是针对自身评估的任何值。
*   表达式`a < 1`是`True`，因为`0`小于`1`。

因为两个部分都是`True`，所以链计算为`True`。

然而，习惯于 Python 中其他操作符的人可能会认为，像其他包含多个操作符的表达式(如`1 + 2 * 3`)一样，Python 在表达式中插入了括号。然而，插入括号的两种方式都不会计算出`True`。

如果你分解表达式，你可以看到为什么两者的值都是`False`。如果分解第一个表达式，会得到以下结果:

>>>

```py
>>> a = 0
>>> a is a
True
>>> True == 1
True
>>> (a is a) < 1
False
```

你可以在上面看到，`a is a`返回`True`，就像对任何值一样。这意味着`(a is a) < 1`和`True < 1`是一样的。布尔是数值类型，`True`等于`1`。所以`True < 1`和`1 < 1`一样。由于这是一个**严格**不等式，而`1 == 1`则返回 False。

第二种表达方式有所不同:

>>>

```py
>>> a = 0
False
>>> a < 1
True
>>> 0 is True
<stdin>:1: SyntaxWarning: "is" with a literal. Did you mean "=="?
False
```

由于`0`小于`1`，`a < 1`返回`True`。既然是`0 != True`，那么就不可能是`0 is True`的情况。

**注意**:不要对上面的`SyntaxWarning`掉以轻心。在数字上使用`is`可能会[混淆](https://realpython.com/python-is-identity-vs-equality/#when-only-some-integers-are-interned)。但是，具体到你知道数字*不*相等的情况，你可以知道`is`也会返回`False`。虽然这个例子是正确的，但它不是一个好的 Python 编码风格的例子。

从中得出的最重要的教训是，用`is`链接比较通常不是一个好主意。它会让读者困惑，可能也没有必要。

像`is`一样，`in`操作符和它的反义词`not in`在链接时经常会产生令人惊讶的结果:

>>>

```py
>>> "b" in "aba" in "cabad" < "cabae"
True
```

为了尽量避免混淆，这个例子用不同的操作符将比较链接起来，并使用带有字符串的`in`来检查子字符串。同样，这不是一个编写良好的代码的例子！然而，能够阅读这个例子并理解它为什么返回`True`是很重要的。

最后，你可以用`not in`链接`is not`:

>>>

```py
>>> greeting = "hello"
>>> quality = "good"
>>> end_greeting = "farewell"
>>> greeting is not quality not in end_greeting
True
```

注意两个运算符中`not`的顺序不一样！负算子是`is not`和`not in`。这符合英语中的常规用法，但在修改代码时很容易出错。

[*Remove ads*](/account/join/)

## Python 布尔测试

Python 布尔值最常见的用法是在 [`if`语句](https://realpython.com/python-conditional-statements/)中。如果值为`True`，将执行该语句:

>>>

```py
>>> 1 == 1
True
>>> if 1 == 1:
...     print("yep")
...
yep
>>> 1 == 2
False
>>> if 1 == 2:
...     print("yep")
...
```

只有当表达式计算结果为`True`时，才会调用`print()`。然而，在 Python 中你可以给`if`任何值。`if`认为`True`的值称为 [**真值**](https://realpython.com/python-operators-expressions/#evaluation-of-non-boolean-values-in-boolean-context) ，`if`认为`False`的值称为 [**假值**](https://realpython.com/python-operators-expressions/#evaluation-of-non-boolean-values-in-boolean-context) 。

`if`通过内部调用内置的`bool()`来决定哪些值为真，哪些值为假。您已经遇到了作为 Python 布尔类型的`bool()`。当被调用时，它将对象转换为布尔值。

### `None`为布尔值

单例对象`None`总是错误的:

>>>

```py
>>> bool(None)
False
```

这在检查标记值的`if`语句中通常很有用。然而，用`is None`显式检查身份通常更好。有时`None`可以与短路评估结合使用，以便有一个默认设置。

例如，您可以使用`or`将`None`替换为空列表:

>>>

```py
>>> def add_num_and_len(num, things=None):
...     return num + len(things or [])
...
>>> add_num_and_len(5, [1, 2, 3])
8
>>> add_num_and_len(6)
6
```

在本例中，如果`things`为非空列表，则不会创建列表，因为`or`会在对`[]`求值之前短路。

### 布尔值形式的数字

对于数字来说，`bool(x)`相当于`x != 0`。这意味着唯一虚假的整数是`0`:

>>>

```py
>>> bool(3), bool(-5), bool(0)
(True, True, False)
```

所有非零整数都是真的。对于[浮点数](https://realpython.com/python-numbers/#floating-point-numbers)也是如此，包括像[无穷大](https://realpython.com/python-math-module/#infinity)和[非一数(NaN)](https://realpython.com/python-math-module/#not-a-number-nan) 这样的特殊浮点数:

>>>

```py
>>> import math
>>> [bool(x) for x in [0, 1.2, 0.5, math.inf, math.nan]]
[False, True, True, True, True]
```

因为无穷大和 NaN 不等于`0`，所以它们是真的。

浮点数上的相等和不相等比较是微妙的操作。由于执行`bool(x)`等同于`x != 0`，这可能会导致浮点数的惊人结果:

>>>

```py
>>> bool(0.1 + 0.2 + (-0.2) + (-0.1))
True
>>> 0.1 + 0.2 + (-0.2) + (-0.1)
2.7755575615628914e-17
```

浮点数计算可能不精确。正因为如此，`bool()`对浮点数的结果可能会令人惊讶。

Python 在标准库中有更多的数字类型，它们遵循相同的规则。对于非内置数值类型，`bool(x)`也等同于`x != 0`。`fractions`模块在标准库中。像其他数字类型一样，唯一的假分数是`0/1`:

>>>

```py
>>> import fractions
>>> bool(fractions.Fraction("1/2")), bool(fractions.Fraction("0/1"))
(True, False)
```

与整数和浮点数一样，分数只有在等于`0`时才是假的。

`decimal`模块也在标准库中。类似地，只有当小数等于`0`时，它们才是假的:

>>>

```py
>>> import decimal, math
>>> with decimal.localcontext(decimal.Context(prec=3)) as ctx:
...     bool(ctx.create_decimal(math.pi) - ctx.create_decimal(22)/7)
...
False
>>> with decimal.localcontext(decimal.Context(prec=4)) as ctx:
...     bool(ctx.create_decimal(math.pi) - ctx.create_decimal(22)/7)
...
True
```

数字`22 / 7`是圆周率小数点后两位的近似值。这个事实在公元前三世纪由阿基米德讨论过。用这个精度计算`22 / 7`和圆周率之差，结果是 falsy。当以更高的精度计算差值时，差值不等于`0`，真值也不等于。

[*Remove ads*](/account/join/)

### 布尔值序列

一般来说，当`len()`的结果为`0`时，拥有 [`len()`](https://realpython.com/len-python-function/) 的对象将为假。不管它们是列表、元组、集合、字符串还是字节字符串:

>>>

```py
>>> bool([1]), bool([])
(True, False)
>>> bool((1,2)), bool(())
(True, False)
>>> bool({1,2,3}), bool(set())
(True, False)
>>> bool({1: 2}), bool({})
(True, False)
>>> bool("hello"), bool("")
(True, False)
>>> bool(b"xyz"), bool(b"")
(True, False)
```

所有具有长度的内置 Python 对象都遵循这一规则。稍后，对于非内置对象，您将看到该规则的一些例外。

### 其他类型为布尔值

除非类型有一个`len()`或者明确定义它们是真还是假，否则它们总是真的。对于内置类型和用户定义类型来说都是如此。特别是，函数总是真实的:

>>>

```py
>>> def func():
...     pass
...
>>> bool(func)
True
```

方法也总是真理。如果在调用函数或方法时缺少括号，您可能会遇到这种情况:

>>>

```py
>>> import datetime
>>> def before_noon():
...     return datetime.datetime.now().hour < 12
...
>>> def greet():
...     if before_noon:
...             print("Good morning!")
...     else:
...             print("Good evening!")
...
>>> greet()
Good morning!
>>> datetime.datetime.now().hour
20
```

这可能是由于忘记了括号或者误导性的文档没有提到您需要调用该函数。如果你期望一个 Python 布尔值，但是有一个函数返回一个布尔值，那么它总是真的。

默认情况下，用户定义的类型总是真实的:

>>>

```py
>>> class Dummy:
...     pass
...
>>> bool(Dummy())
True
```

创建一个空类会使该类的每个对象都变得真实。除非定义了特殊的方法，否则所有的对象都是真的。如果你想创建你的类 falsy 的一些实例，你可以定义`.__bool__()`:

>>>

```py
>>> class BoolLike:
...     am_i_truthy = False
...     def __bool__(self):
...             return self.am_i_truthy
...
>>> x = BoolLike()
>>> bool(x)
False
>>> x.am_i_truthy = True
>>> bool(x)
True
```

你也可以使用`.__bool__()`让一个物体既不真实也不虚假:

>>>

```py
>>> class ExcludedMiddle:
...     def __bool__(self):
...             raise ValueError("neither")
...
>>> x = ExcludedMiddle()
>>> bool(x)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in __bool__
ValueError: neither

>>> if x:
...     print("x is truthy")
... else:
...     print("x is falsy")
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in __bool__
ValueError: neither
```

`if`语句也使用了`.__bool__()`。它这样做是为了评估对象是真还是假，从而确定执行哪个分支。

如果在一个类上定义了`__len__`方法，那么它的实例就有一个`len()`。在这种情况下，当实例的长度为`0`时，实例的布尔值将为 falsy:

>>>

```py
>>> class DummyContainer:
...     my_length = 0
...     def __len__(self):
...         return self.my_length
...
>>> x = DummyContainer()
>>> bool(x)
False
>>> x.my_length = 5
>>> bool(x)
True
```

在这个例子中，`len(x)`在赋值前返回`0`，赋值后返回`5`。然而，反之则不然。定义`.__bool__()`不会给实例一个长度:

>>>

```py
>>> class AlwaysTrue:
...     def __bool__(self):
...         return True
...
>>> class AlwaysFalse:
...     def __bool__(self):
...         return False
...
>>> bool(AlwaysTrue()), bool(AlwaysFalse())
(True, False)

>>> len(AlwaysTrue())
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: object of type 'AlwaysTrue' has no len()

>>> len(AlwaysFalse())
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: object of type 'AlwaysFalse' has no len()
```

定义`.__bool__()`不会使任何一个类的实例有一个`len()`。当`.__bool__()`和`.__len__()`都被定义时，`.__bool__()`优先:

>>>

```py
>>> class BooleanContainer:
...     def __len__(self):
...         return 100
...     def __bool__(self):
...         return False
...
>>> x=BooleanContainer()
>>> len(x)
100
>>> bool(x)
False
```

即使`x`有`100`的长度，还是 falsy。

[*Remove ads*](/account/join/)

### 示例:NumPy 数组

上面的例子看起来像是只有当你用 Python 编写一个类来演示边界情况时才会发生的事情。然而，使用 [PyPI](https://realpython.com/what-is-pip/) : [NumPy](https://numpy.org/) 上最流行的库之一也可能得到类似的结果。

[数组](https://realpython.com/numpy-array-programming/)和数字一样，是假是真取决于它们和`0`相比如何:

>>>

```py
>>> from numpy import array
>>> x = array([0])
>>> len(x)
1
>>> bool(x)
False
```

即使`x`的长度为`1`，它仍然是 falsy，因为它的值是`0`。

当数组有多个元素时，有些元素可能是假的，有些可能是真的。在这些情况下，NumPy 将引发一个异常:

>>>

```py
>>> from numpy import array
>>> import textwrap
>>> y=array([0, 1])
>>> try:
...     bool(y)
... except ValueError as exc:
...     print("\n".join(textwrap.wrap(str(exc))))
...
The truth value of an array with more than one element is ambiguous.
Use a.any() or a.all()
```

这个异常非常冗长，为了便于阅读，代码使用文本处理来换行。

一个更有趣的例子是空数组。你可能想知道这些是否像其他序列一样是假的或真的，因为它们不等于`0`。正如你在上面看到的，这不是唯一的两个可能的答案。数组也可以拒绝布尔值。

有趣的是，这些选项中没有一个是完全正确的:

>>>

```py
>>> bool(array([]))
<stdin>:1: DeprecationWarning: The truth value of an empty array is ambiguous.
Returning False, but in future this will result in an error.
Use `array.size > 0` to check that an array is not empty.
False
```

虽然空数组目前是错误的，但是依赖这种行为是危险的。在一些未来的 NumPy 版本中，这将引发一个异常。

### 运算符和函数

Python 中还有一些地方进行布尔测试。其中之一就是布尔运算符。

操作符`and`、`or`和`not`接受任何支持布尔测试的值。在`not`的情况下，它将总是返回一个布尔值:

>>>

```py
>>> not 1
False
>>> not 0
True
```

`not`的真值表仍然是正确的，但是现在它接受了输入的真实性。

在`and`和`or`的情况下，除了短路评估之外，它们还返回停止评估时的值:

>>>

```py
>>> 1 and 2
2
>>> 0 and 1
0
>>> 1 or 2
1
>>> 0 or 2
2
```

真值表仍然是正确的，但它们现在定义了结果的真实性，这取决于输入的真实性。例如，当您想给值设置默认值时，这很方便。

假设您有一个名为`summarize()`的函数，如果文本太长，它会获取开头和结尾，并在中间添加一个省略号(`...`)。这在一些无法容纳全文的报告中可能很有用。但是，一些数据集缺少由`None`表示的值。

由于`summarize()`假设输入是一个字符串，它将在`None`失败:

>>>

```py
>>> def summarize(long_text):
...     if len(long_text) <= 4:
...         return long_text
...     return long_text[:2] +"..." + long_text[-2:]
...
>>> summarize("hello world")
'he...ld'
>>> summarize("hi")
'hi'
>>> summarize("")
''
>>> summarize(None)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in summarize
TypeError: object of type 'NoneType' has no len()

>>> for a in ["hello world", "hi", "", None]:
...     print("-->", summarize(a or ""))
...
--> he...ld
--> hi
-->
-->
```

这个例子利用了`None`的虚假性和`or`不仅短路而且还返回最后一个待评估值的事实。打印报告的代码将`or ""`添加到`summarize()`的参数中。添加`or ""`有助于您避免仅仅一个小的代码更改就出现错误。

内置函数 [`all()`](https://realpython.com/python-all/) 和 [`any()`](https://realpython.com/any-python/) 计算真值和短路，但不返回最后一个要计算的值。`all()`检查其所有论点是否真实:

>>>

```py
>>> all([1, 2, 3])
True
>>> all([0, 1, 2])
False
>>> all(x / (x - 1) for x in [0, 1])
False
```

在最后一行，`all()`没有为`1`评估`x / (x - 1)`。既然`1 - 1`是`0`，这就多了一个`ZeroDivisionError`。

检查它的任何参数是否正确:

>>>

```py
>>> any([1, 0, 0])
True
>>> any([False, 0, 0.0])
False
>>> any(1 / x for x in [1, 0])
True
```

在最后一行，`any()`没有为`0`评估`1 / x`。

[*Remove ads*](/account/join/)

## 结论

Python Boolean 是一种常用的数据类型，有许多有用的应用。您可以使用布尔运算符，如`not`、`and`、`or`、`in`、`is`、`==`和`!=`来比较值，并检查成员资格、身份或相等性。您还可以使用带有`if`语句的布尔测试，根据表达式的真实性来控制程序的流程。

**在本教程中，您已经学会了如何:**

*   用**布尔运算符**操作布尔值
*   将布尔值转换为其他类型
*   将其他类型转换为 **Python 布尔值**
*   使用布尔值编写**高效可读的** Python 代码

您现在知道了短路求值是如何工作的，并且认识到了布尔值和`if`语句之间的联系。这些知识将有助于您理解现有的代码，并避免可能导致您自己的程序出错的常见陷阱。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python 布尔:利用真值**](/courses/booleans-leveraging-truth/)*************