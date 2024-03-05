# 如何用 PEP 8 写出漂亮的 Python 代码

> 原文：<https://realpython.com/python-pep8/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和编写的教程一起看，加深理解: [**用 PEP 8**](/courses/writing-beautiful-python-code-pep-8/) 编写漂亮的 Pythonic 代码

PEP8，有时拼写为 PEP 8 或 PEP-8，是一个提供如何编写 Python 代码的指南和最佳实践的文档。它是由吉多·范·罗苏姆、巴里·华沙和尼克·科格兰在 2001 年写的。PEP 8 的主要目的是提高 Python 代码的可读性和一致性。

PEP 代表 Python 增强提议，有好几个。PEP 是一个文档，它描述了为 Python 提出的新特性，并为社区记录了 Python 的一些方面，如设计和风格。

本教程概述了 PEP 8 中的主要指导方针。它的目标是初级到中级程序员，因此我没有涉及一些最高级的主题。你可以通过阅读完整的 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 文档来了解这些。

**本教程结束时，你将能够**:

*   写符合 PEP 8 的 Python 代码
*   理解人教版 8 中的指导原则背后的原因
*   设置您的开发环境，以便您可以开始编写符合 PEP 8 的 Python 代码

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 为什么我们需要 PEP 8

> "可读性很重要。"
> 
> —*Python 之禅*

PEP 8 的存在是为了提高 Python 代码的可读性。但是为什么可读性如此重要呢？为什么编写可读代码是 Python 语言的指导原则之一？

正如吉多·范·罗苏姆所说，“代码被阅读的次数比它被编写的次数多得多。”您可能要花几分钟，或者一整天，编写一段代码来处理用户认证。一旦你写了，你就不会再写了。但是你一定要再读一遍。这段代码可能仍然是您正在进行的项目的一部分。每次你回到那个文件，你都必须记住代码做了什么，为什么要写它，所以可读性很重要。

如果你是 Python 的新手，在你写完一段代码后的几天或几周内，很难记住这段代码做了什么。如果你遵循 PEP 8，你可以确定你已经很好地命名了你的[变量](https://realpython.com/python-variables/)。您将知道您已经添加了足够多的空白，因此在代码中遵循逻辑步骤更加容易。你也会很好地注释你的代码。所有这些都将意味着你的代码可读性更强，更容易理解。作为初学者，遵循 PEP 8 的规则可以使学习 Python 成为一项更加愉快的任务。

如果你在找一份开发工作，遵循 PEP 8 尤其重要。编写清晰易读的代码显示了专业性。这会告诉雇主你知道如何很好地组织你的代码。

如果你有更多编写 Python 代码的经验，那么你可能需要与其他人合作。在这里编写可读的代码是至关重要的。其他人，可能从未见过你或见过你的编码风格，将不得不阅读和理解你的代码。拥有你所遵循和认可的指导方针将会让其他人更容易阅读你的代码。

[*Remove ads*](/account/join/)

## 命名惯例

> “显性比隐性好。”
> 
> —*Python 之禅*

当你写 Python 代码时，你必须命名很多东西:变量、函数、类、包等等。选择明智的名字会节省你以后的时间和精力。你将能够从名字中猜出某个变量、函数或类代表什么。您还将避免使用不合适的名称，这可能会导致难以调试的错误。

**注意**:切勿使用`l`、`O`或`I`单字母名称，因为根据字体不同，这些名称可能会被误认为`1`和`0`:

```py
O = 2  # This may look like you're trying to reassign 2 to zero
```

### 命名风格

下表概述了 Python 代码中的一些常见命名样式以及何时应该使用它们:

| 类型 | 命名约定 | 例子 |
| --- | --- | --- |
| 功能 | 使用小写单词。用下划线分隔单词以提高可读性。 | `function`，`my_function` |
| 可变的 | 使用小写的单个字母、单词或多个单词。用下划线分隔单词以提高可读性。 | `x`、`var`、`my_variable` |
| 班级 | 每个单词以大写字母开头。不要用下划线分隔单词。这种风格被称为[骆驼案或者帕斯卡案](https://en.wikipedia.org/wiki/Camel_case)。 | `Model`，`MyClass` |
| 方法 | 使用小写单词。用下划线分隔单词以提高可读性。 | `class_method`，`method` |
| 常数 | 使用大写的单个字母、单词或多个单词。用下划线分隔单词以提高可读性。 | `CONSTANT`、`MY_CONSTANT`、`MY_LONG_CONSTANT` |
| 组件 | 使用一个或多个短的小写单词。用下划线分隔单词以提高可读性。 | `module.py`，`my_module.py` |
| 包裹 | 使用一个或多个短的小写单词。不要用下划线分隔单词。 | `package`，`mypackage` |

这些是一些常见的命名约定以及如何使用它们的示例。但是，为了编写可读的代码，您仍然必须小心选择字母和单词。除了在代码中选择正确的命名风格，您还必须仔细选择名称。以下是如何尽可能有效地做到这一点的几点建议。

### 如何选择名字

为变量、函数、类等等选择名称可能很有挑战性。在编写代码时，您应该在命名选择上多加考虑，因为这将使您的代码更具可读性。在 Python 中命名对象的最佳方式是使用描述性名称，以便清楚地表明对象代表什么。

在命名变量时，您可能会倾向于选择简单的单字母小写名称，如`x`。但是，除非你使用`x`作为数学函数的自变量，否则不清楚`x`代表什么。假设您将一个人的名字存储为一个[字符串](https://realpython.com/python-strings/)，并且您想使用字符串切片对他们的名字进行不同的格式化。您可能会得到这样的结果:

>>>

```py
>>> # Not recommended
>>> x = 'John Smith'
>>> y, z = x.split()
>>> print(z, y, sep=', ')
'Smith, John'
```

这是可行的，但是你必须记住`x`、`y`和`z`代表什么。也可能让合作者感到困惑。更清晰的名称选择应该是这样的:

>>>

```py
>>> # Recommended
>>> name = 'John Smith'
>>> first_name, last_name = name.split()
>>> print(last_name, first_name, sep=', ')
'Smith, John'
```

同样，为了减少打字量，在选择名字时使用缩写会很有诱惑力。在下面的例子中，我定义了一个函数`db()`,它接受一个参数`x`,并将其加倍:

```py
# Not recommended
def db(x):
    return x * 2
```

乍一看，这似乎是一个明智的选择。很容易成为 double 的缩写。但是想象一下几天后回到这段代码。你可能已经忘记了你想用这个函数实现什么，这使得猜测你如何缩写它变得很困难。

下面的例子就清楚多了。如果您在编写这段代码几天后再来看这段代码，您仍然能够阅读并理解这个函数的用途:

```py
# Recommended
def multiply_by_two(x):
    return x * 2
```

同样的理念也适用于 Python 中的所有其他数据类型和对象。尽可能使用最简洁但具有描述性的名称。

[*Remove ads*](/account/join/)

## 代码布局

> “漂亮总比难看好。”
> 
> —*Python 之禅*

你如何布局你的代码对它的可读性有很大的影响。在这一节中，您将学习如何添加垂直空格来提高代码的可读性。您还将学习如何处理 PEP 8 中推荐的 79 个字符的行限制。

### 空白行

垂直空白，或空白行，可以大大提高代码的可读性。堆积在一起的代码可能会让人不知所措，难以阅读。类似地，代码中太多的空行会使代码看起来非常稀疏，读者可能需要进行不必要的滚动。下面是关于如何使用垂直空格的三个关键指导原则。

用两个空行包围顶级函数和类。顶级函数和类应该是完全独立的，并处理独立的功能。在它们周围留出额外的垂直空间是有意义的，这样就能清楚地看出它们是分开的:

```py
class MyFirstClass:
    pass

class MySecondClass:
    pass

def top_level_function():
    return None
```

用一个空行包围类中的方法定义。在一个类中，所有的函数都是相互关联的。最好在它们之间只留一行:

```py
class MyClass:
    def first_method(self):
        return None

    def second_method(self):
        return None
```

**函数内部尽量少用空行，以显示清晰的步骤。**有时候，一个复杂的函数要在 [`return`语句](https://realpython.com/python-return-statement/)之前完成几个步骤。为了帮助读者理解函数内部的逻辑，在每个步骤之间留一个空行会很有帮助。

在下面的例子中，有一个函数计算一个[列表](https://realpython.com/python-lists-tuples/)的方差。这是一个分两步的问题，所以我在每一步之间留了一个空行。在`return`语句前还有一个空行。这有助于读者清楚地看到返回的内容:

```py
def calculate_variance(number_list):
    sum_list = 0
    for number in number_list:
        sum_list = sum_list + number
    mean = sum_list / len(number_list)

    sum_squares = 0
    for number in number_list:
        sum_squares = sum_squares + number**2
    mean_squares = sum_squares / len(number_list)

    return mean_squares - mean**2
```

如果你小心使用垂直空格，它可以大大提高你的代码的可读性。它有助于读者直观地理解您的代码是如何分成几个部分的，以及这些部分是如何相互关联的。

### 最大线路长度和断线

PEP 8 建议行数应限制在 79 个字符以内。这是因为它允许您一个接一个地打开多个文件，同时避免换行。

当然，将语句控制在 79 个字符以内并不总是可能的。PEP 8 概述了允许语句跨几行运行的方法。

如果代码包含在圆括号、中括号或大括号中，Python 将假定行连续:

```py
def function(arg_one, arg_two,
             arg_three, arg_four):
    return arg_one
```

如果不能使用隐式延续，那么可以使用反斜杠来换行:

```py
from mypkg import example1, \
    example2, example3
```

但是，如果您可以使用隐式延续，那么您应该这样做。

如果需要在二元操作符周围换行，比如`+`和`*`，它应该在操作符之前换行。这条规则源于数学。数学家们一致认为，在二进制运算符之前中断可以提高可读性。比较下面两个例子。

以下是在二元运算符前中断的示例:

```py
# Recommended
total = (first_variable
         + second_variable
         - third_variable)
```

您可以立即看到哪个变量被增加或减少，因为运算符就在被运算的变量旁边。

现在，让我们来看一个二元运算符后的中断示例:

```py
# Not Recommended
total = (first_variable +
         second_variable -
         third_variable)
```

在这里，很难看出哪个变量在增加，哪个变量在减少。

在二进制操作符之前中断会产生更可读的代码，所以 PEP 8 鼓励这样做。二元运算符后*持续*中断的代码仍然符合 PEP 8。但是，我们鼓励您在二元运算符前中断。

[*Remove ads*](/account/join/)

## 缩进

> "应该有一种——最好只有一种——显而易见的方法来做这件事。"
> 
> —*Python 之禅*

缩进或前导空格在 Python 中非常重要。Python 中代码行的缩进级别决定了语句如何分组。

考虑下面的例子:

```py
x = 3
if x > 5:
    print('x is larger than 5')
```

缩进的 [`print`语句](https://realpython.com/python-print/)让 Python 知道只有当`if`语句返回`True`时才应该执行。同样的缩进也适用于告诉 Python 在调用函数时执行什么代码，或者什么代码属于给定的类。

PEP 8 规定的关键缩进规则如下:

*   使用 4 个连续空格表示缩进。
*   比制表符更喜欢空格。

### 制表符与空格

如上所述，缩进代码时应该使用空格而不是制表符。当您按下 `Tab` 键时，您可以调整文本编辑器中的设置以输出 4 个空格而不是一个制表符。

如果您正在使用 Python 2，并且已经混合使用了制表符和空格来缩进代码，那么在尝试运行它时，您不会看到错误。为了帮助您检查一致性，您可以在从命令行运行 Python 2 代码时添加一个`-t`标志。当您使用制表符和空格不一致时，解释器将发出警告:

```py
$ python2 -t code.py
code.py: inconsistent use of tabs and spaces in indentation
```

相反，如果您使用`-tt`标志，解释器将发出错误而不是警告，您的代码将不会运行。使用这种方法的好处是解释器告诉你不一致的地方在哪里:

```py
$ python2 -tt code.py
 File "code.py", line 3
 print(i, j)
 ^
TabError: inconsistent use of tabs and spaces in indentation
```

Python 3 不允许混合使用制表符和空格。因此，如果您使用的是 Python 3，则会自动发出这些错误:

```py
$ python3 code.py
 File "code.py", line 3
 print(i, j)
 ^
TabError: inconsistent use of tabs and spaces in indentation
```

您可以使用制表符或空格来表示缩进，从而编写 Python 代码。但是，如果您正在使用 Python 3，您必须与您的选择保持一致。否则，您的代码将不会运行。PEP 8 建议您始终使用 4 个连续空格来表示缩进。

### 换行后的缩进

当您使用行继续符将行保持在 79 个字符以下时，使用缩进来提高可读性是很有用的。它允许读者区分两行代码和跨越两行的一行代码。您可以使用两种缩进样式。

第一个是将缩进的块与开始分隔符对齐:

```py
def function(arg_one, arg_two,
             arg_three, arg_four):
    return arg_one
```

有时，您会发现只需要 4 个空格来对齐开始分隔符。这通常出现在跨越多行的`if`语句中，因为`if`、空格和左括号组成了 4 个字符。在这种情况下，很难确定`if`语句中的嵌套代码块从哪里开始:

```py
x = 5
if (x > 3 and
    x < 10):
    print(x)
```

在这种情况下，PEP 8 提供了两种替代方法来帮助提高可读性:

*   在最终条件后添加注释。由于大多数编辑器中的语法突出显示，这将把条件从嵌套代码中分离出来:

    ```py
    x = 5
    if (x > 3 and
        x < 10):
        # Both conditions satisfied
        print(x)` 
    ```

*   在行延续上添加额外的缩进:

    ```py
    x = 5
    if (x > 3 and
            x < 10):
        print(x)` 
    ```

换行符后的另一种缩进样式是**悬挂缩进**。这是一个印刷术语，意思是段落或语句中除第一行以外的每一行都缩进。您可以使用悬挂缩进来直观地表示一行代码的延续。这里有一个例子:

```py
var = function(
    arg_one, arg_two,
    arg_three, arg_four)
```

注意:当你使用悬挂缩进时，第一行不能有任何参数。以下示例不符合 PEP 8:

```py
# Not Recommended
var = function(arg_one, arg_two,
    arg_three, arg_four)
```

当使用悬挂缩进时，添加额外的缩进来区分连续行和函数中包含的代码。下面的示例很难阅读，因为函数内部的代码与后续行的缩进级别相同:

```py
# Not Recommended
def function(
    arg_one, arg_two,
    arg_three, arg_four):
    return arg_one
```

相反，最好在行继续符上使用双缩进。这有助于区分函数参数和函数体，从而提高可读性:

```py
def function(
        arg_one, arg_two,
        arg_three, arg_four):
    return arg_one
```

当您编写符合 PEP 8 的代码时，79 个字符的行限制迫使您在代码中添加换行符。为了提高可读性，您应该缩进一个续行，以表明它是一个续行。有两种方法可以做到这一点。第一个是将缩进的块与开始分隔符对齐。第二种是使用悬挂式缩进。您可以自由选择在换行后使用哪种缩进方法。

[*Remove ads*](/account/join/)

### 右大括号放在哪里

换行允许您在圆括号、方括号或大括号内换行。很容易忘记右大括号，但是把它放在一个合理的地方是很重要的。否则，会使读者感到困惑。PEP 8 为隐含行延续中的右大括号位置提供了两个选项:

*   将右大括号与前一行的第一个非空白字符对齐:

    ```py
    list_of_numbers = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
        ]` 
    ```

*   将右大括号与开始构造的行的第一个字符对齐:

    ```py
    list_of_numbers = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    ]` 
    ```

你可以自由选择使用哪个选项。但是，一如既往，一致性是关键，所以尝试坚持以上方法之一。

## 评论

> "如果实现很难解释，这是一个坏主意."
> 
> —*Python 之禅*

您应该在编写代码时使用注释来记录代码。重要的是将你的代码文档化，这样你和任何合作者都能理解它。当您或其他人阅读注释时，他们应该能够容易地理解该注释所适用的代码，以及它如何与您的代码的其余部分相适应。

向代码中添加注释时，需要记住以下要点:

*   将注释和文档字符串的行长度限制为 72 个字符。
*   使用完整的句子，以大写字母开头。
*   如果您更改代码，请确保更新注释。

### 块注释

使用块注释来记录一小部分代码。当您必须编写几行代码来执行单个操作(例如从文件导入数据或更新数据库条目)时，它们非常有用。它们很重要，因为它们帮助其他人理解给定代码块的用途和功能。

PEP 8 为编写块注释提供了以下规则:

*   将块注释缩进到与它们描述的代码相同的级别。
*   每行以一个`#`开头，后跟一个空格。
*   用包含单个`#`的行分隔段落。

以下是解释`for`循环功能的块注释。请注意，该句子会换行以保留 79 个字符的行限制:

```py
for i in range(0, 10):
    # Loop over i ten times and print out the value of i, followed by a
    # new line character
    print(i, '\n')
```

有时，如果代码非常专业，那么有必要在块注释中使用多个段落:

```py
def quadratic(a, b, c, x):
    # Calculate the solution to a quadratic equation using the quadratic
    # formula.
    #
    # There are always two solutions to a quadratic equation, x_1 and x_2.
    x_1 = (- b+(b**2-4*a*c)**(1/2)) / (2*a)
    x_2 = (- b-(b**2-4*a*c)**(1/2)) / (2*a)
    return x_1, x_2
```

如果你对什么类型的注释合适有疑问，那么块注释通常是个不错的选择。在你的代码中尽可能多地使用它们，但是如果你对你的代码做了修改，一定要更新它们！

### 行内注释

行内注释解释一段代码中的一条语句。它们有助于提醒您，或者向他人解释，为什么某一行代码是必需的。以下是 PEP 8 对他们的评价:

*   谨慎使用行内注释。
*   将行内注释写在它们所引用的语句所在的同一行。
*   用两个或更多空格将行内注释与语句分隔开。
*   像块注释一样，用一个`#`和一个空格开始行内注释。
*   不要用它们来解释显而易见的事情。

下面是一个行内注释的示例:

```py
x = 5  # This is an inline comment
```

有时，行内注释似乎是必要的，但是您可以使用更好的命名约定来代替。这里有一个例子:

```py
x = 'John Smith'  # Student Name
```

这里，行内注释给出了额外的信息。然而，使用`x`作为人名的变量名是不好的做法。如果重命名变量，则不需要行内注释:

```py
student_name = 'John Smith'
```

最后，像这样的行内注释是不好的做法，因为它们陈述了明显而混乱的代码:

```py
empty_list = []  # Initialize empty list

x = 5
x = x * 5  # Multiply x by 5
```

内联注释比块注释更具体，在不必要的时候很容易添加它们，这会导致混乱。你可以只使用块注释，所以，除非你确定你需要行内注释，如果你坚持块注释，你的代码更有可能是 PEP 8 兼容的。

[*Remove ads*](/account/join/)

### 文档字符串

文档字符串，或称[文档字符串](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)，是用双引号(`"""`)或单引号(`'''`)括起来的字符串，出现在任何函数、类、方法或模块的第一行。您可以使用它们来解释和记录特定的代码块。有一个完整的 PEP， [PEP 257](https://www.python.org/dev/peps/pep-0257/) ，涵盖了 docstrings，但是您将在这一节中得到一个摘要。

适用于文档字符串的最重要的规则如下:

*   用三个双引号将文档字符串括起来，如`"""This is a docstring"""`所示。

*   为所有公共模块、函数、类和方法编写它们。

*   将结束多行文档字符串的`"""`单独放在一行中:

    ```py
    def quadratic(a, b, c, x):
        """Solve quadratic equation via the quadratic formula.

     A quadratic equation has the following form:
     ax**2 + bx + c = 0

     There always two solutions to a quadratic equation: x_1 & x_2.
     """
        x_1 = (- b+(b**2-4*a*c)**(1/2)) / (2*a)
        x_2 = (- b-(b**2-4*a*c)**(1/2)) / (2*a)

        return x_1, x_2` 
    ```

*   对于单行文档字符串，保持`"""`在同一行:

    ```py
    def quadratic(a, b, c, x):
        """Use the quadratic formula"""
        x_1 = (- b+(b**2-4*a*c)**(1/2)) / (2*a)
        x_2 = (- b-(b**2-4*a*c)**(1/2)) / (2*a)

        return x_1, x_2` 
    ```

关于记录 Python 代码的更详细的文章，请参见 James Mertz 的[记录 Python 代码:完整指南](https://realpython.com/documenting-python-code/#docstrings-background)。

## 表达式和语句中的空格

> “疏比密好。”
> 
> —*Python 之禅*

如果使用得当，空格在表达式和语句中非常有用。如果没有足够的空白，那么代码可能很难阅读，因为它们都被捆绑在一起。如果有太多的空白，那么就很难在一个语句中直观地组合相关的术语。

### 二元运算符周围的空格

用一个空格将下列二元运算符括起来:

*   赋值运算符(`=`、`+=`、`-=`等等)

*   比较(`==`、`!=`、`>`、`<`)。`>=`、`<=`)和(`is`、`is not`、`in`、`not in`)

*   布尔型(`and`、`not`、`or`)

**注意**:当`=`用于给函数参数赋值默认值时，不要用空格将其括起来。

```py
# Recommended
def function(default_parameter=5):
    # ...

# Not recommended
def function(default_parameter = 5):
    # ...
```

当一个语句中有多个操作符时，在每个操作符前后添加一个空格看起来会令人困惑。相反，最好只在优先级最低的操作符周围添加空格，尤其是在执行数学运算时。这里有几个例子:

```py
# Recommended
y = x**2 + 5
z = (x+y) * (x-y)

# Not Recommended
y = x ** 2 + 5
z = (x + y) * (x - y)
```

您也可以将此应用于有多个条件的`if`语句:

```py
# Not recommended
if x > 5 and x % 2 == 0:
    print('x is larger than 5 and divisible by 2!')
```

在上面的例子中，`and`操作符的优先级最低。因此，将`if`语句表达如下可能更清楚:

```py
# Recommended
if x>5 and x%2==0:
    print('x is larger than 5 and divisible by 2!')
```

您可以自由选择哪个更清晰，但要注意的是，您必须在操作符的两边使用相同数量的空白。

以下情况是不可接受的:

```py
# Definitely do not do this!
if x >5 and x% 2== 0:
    print('x is larger than 5 and divisible by 2!')
```

在切片中，冒号充当二元运算符。因此，上一节概述的规则适用，两边应该有相同数量的空白。以下列表切片示例是有效的:

```py
list[3:4]

# Treat the colon as the operator with lowest priority
list[x+1 : x+2]

# In an extended slice, both colons must be
# surrounded by the same amount of whitespace
list[3:4:5]
list[x+1 : x+2 : x+3]

# The space is omitted if a slice parameter is omitted
list[x+1 : x+2 :]
```

总之，大多数操作符都应该用空格括起来。但是，这条规则有一些注意事项，比如在函数参数中，或者在一个语句中组合多个运算符时。

[*Remove ads*](/account/join/)

### 何时避免添加空格

在某些情况下，添加空白会使代码更难阅读。过多的空白会使代码过于稀疏，难以理解。PEP 8 非常清晰地列举了不适合使用空格的例子。

避免添加空格的最重要的地方是在行尾。这被称为**尾随空白**。它是不可见的，会产生难以跟踪的错误。

下面列出了一些应该避免添加空格的情况:

*   紧接在圆括号、方括号或大括号内:

    ```py
    # Recommended
    my_list = [1, 2, 3]

    # Not recommended
    my_list = [ 1, 2, 3, ]` 
    ```

*   在逗号、分号或冒号之前:

    ```py
    x = 5
    y = 6

    # Recommended
    print(x, y)

    # Not recommended
    print(x , y)` 
    ```

*   在开始函数调用的参数列表的左括号之前:

    ```py
    def double(x):
        return x * 2

    # Recommended
    double(3)

    # Not recommended
    double (3)` 
    ```

*   在开始索引或切片的左括号之前:

    ```py
    # Recommended
    list[3]

    # Not recommended
    list [3]` 
    ```

*   在结尾逗号和右括号之间:

    ```py
    # Recommended
    tuple = (1,)

    # Not recommended
    tuple = (1, )` 
    ```

*   要对齐赋值运算符:

    ```py
    # Recommended
    var1 = 5
    var2 = 6
    some_long_var = 7

    # Not recommended
    var1          = 5
    var2          = 6
    some_long_var = 7` 
    ```

确保代码中没有尾随空格。在其他情况下，PEP 8 不鼓励添加额外的空白，比如在括号内，逗号和冒号前。你也不应该为了对齐操作符而添加额外的空格。

## 编程建议

> “简单比复杂好。”
> 
> —*Python 之禅*

您经常会发现，在 Python(以及任何其他编程语言)中，有几种方法可以执行类似的操作。在本节中，您将看到 PEP 8 提供的一些建议，以消除歧义并保持一致性。

**不要使用等价运算符将布尔值与`True`或`False`进行比较。**你经常需要检查一个[布尔值](https://realpython.com/python-boolean/)是真还是假。这样做时，使用如下语句会很直观:

```py
# Not recommended
my_bool = 6 > 5
if my_bool == True:
    return '6 is bigger than 5'
```

这里不需要使用等价运算符`==`。`bool`只能取值`True`或`False`。写下以下内容就足够了:

```py
# Recommended
if my_bool:
    return '6 is bigger than 5'
```

这种用布尔值执行`if`语句的方式需要的代码更少，也更简单，所以 PEP 8 鼓励它。

**利用空序列在`if`语句中为假的事实。如果你想检查一个列表是否为空，你可能想检查列表的长度。如果列表为空，那么它的长度为`0`，当在`if`语句中使用时，相当于`False`。这里有一个例子:**

```py
# Not recommended
my_list = []
if not len(my_list):
    print('List is empty!')
```

然而，在 Python 中，任何空列表、字符串或元组都是 [falsy](https://docs.python.org/3/library/stdtypes.html#truth-value-testing) 。因此，我们可以提出一个更简单的替代方案:

```py
# Recommended
my_list = []
if not my_list:
    print('List is empty!')
```

虽然两个例子都会打印出`List is empty!`，但是第二个选项更简单，所以 PEP 8 鼓励它。

**在`if`语句中使用`is not`而不是`not ... is`。**如果你试图检查一个变量是否有一个定义的值，有两个选项。第一种是用`x is not None`对`if`语句求值，如下例所示:

```py
# Recommended
if x is not None:
    return 'x exists!'
```

第二种选择是对`x is None`进行评估，然后根据`not`的结果生成`if`声明:

```py
# Not recommended
if not x is None:
    return 'x exists!'
```

虽然两个选项都会被正确评估，但第一个更简单，所以 PEP 8 鼓励它。

**当你指`if x is not None:`的时候不要用`if x:`。有时，你可能有一个函数，它的参数默认为`None`。在检查此类参数`arg`是否被赋予了不同的值时，一个常见的错误是使用以下内容:**

```py
# Not Recommended
if arg:
    # Do something with arg...
```

这段代码检查`arg`是否正确。相反，您希望检查`arg`是否为`not None`，因此最好使用以下代码:

```py
# Recommended
if arg is not None:
    # Do something with arg...
```

这里犯的错误是假设`not None`和 truthy 是等价的。你可以设置`arg = []`。正如我们在上面看到的，空列表在 Python 中被评估为 falsy。因此，即使参数`arg`被赋值，条件也不满足，因此`if`语句体中的代码不会被执行。

**用`.startswith()`和`.endswith()`代替切片。**如果你试图检查一个字符串`word`是否以单词`cat`为前缀或后缀，使用[列表切片](https://realpython.com/python-strings/#string-slicing)似乎是明智的。然而，列表切片容易出错，您必须在前缀或后缀中硬编码字符数。对于不太熟悉 Python 列表切片的人来说，也不清楚您想要实现什么:

```py
# Not recommended
if word[:3] == 'cat':
    print('The word starts with "cat"')
```

然而，这不如使用`.startswith()`更具可读性:

```py
# Recommended
if word.startswith('cat'):
    print('The word starts with "cat"')
```

同样，当你检查后缀时，同样的原则也适用。下面的例子概述了如何检查一个字符串是否以`jpg`结尾:

```py
# Not recommended
if file_name[-3:] == 'jpg':
    print('The file is a JPEG')
```

虽然结果是正确的，但符号有点笨拙，难以阅读。相反，您可以使用`.endswith()`,如下例所示:

```py
# Recommended
if file_name.endswith('jpg'):
    print('The file is a JPEG')
```

与大多数这些编程建议一样，目标是可读性和简单性。在 Python 中，有许多不同的方法来执行相同的操作，因此关于选择哪种方法的指南很有帮助。

[*Remove ads*](/account/join/)

## 何时忽略 PEP 8

这个问题的简短答案是永远不会。如果你严格遵循 PEP 8，你可以保证你会有干净的、专业的、可读的代码。这将有利于你以及合作者和潜在雇主。

然而，PEP 8 中的一些准则在以下情况下不方便:

*   如果遵循 PEP 8 会破坏与现有软件的兼容性
*   如果你正在做的代码与 PEP 8 不一致
*   如果代码需要与旧版本的 Python 保持兼容

## 帮助确保您的代码遵循 PEP 8 的提示和技巧

要确保你的代码符合 PEP 8，需要记住很多东西。当你开发代码时，记住所有这些规则可能是一项艰巨的任务。更新过去的项目以符合 PEP 8 特别耗时。幸运的是，有工具可以帮助加速这个过程。有两类工具可以用来加强 PEP 8 的兼容性:linters 和 autoformatters。

### 棉绒

Linters 是分析代码和标记错误的程序。他们提供了如何修复错误的建议。当作为文本编辑器的扩展安装时，Linters 特别有用，因为它们会在您书写时标记错误和文体问题。在这一节中，您将看到 linters 如何工作的概述，最后是到文本编辑器扩展的链接。

Python 代码的最佳例子如下:

*   **[`pycodestyle`](https://pypi.org/project/pycodestyle/)** 是一个根据 PEP 8 中的一些样式约定来检查你的 Python 代码的工具。

    使用`pip`安装`pycodestyle`:

    ```py
    $ pip install pycodestyle` 
    ```

    您可以使用以下命令从终端运行`pycodestyle`:

    ```py
    $ pycodestyle code.py
    code.py:1:17: E231 missing whitespace after ','
    code.py:2:21: E231 missing whitespace after ','
    code.py:6:19: E711 comparison to None should be 'if cond is None:'` 
    ```

*   **[`flake8`](https://pypi.org/project/flake8/)** 是一个结合了调试器`pyflakes`和`pycodestyle`的工具。

    使用`pip`安装`flake8`:

    ```py
    $ pip install flake8` 
    ```

    使用以下命令从终端运行`flake8`:

    ```py
    $ flake8 code.py
    code.py:1:17: E231 missing whitespace after ','
    code.py:2:21: E231 missing whitespace after ','
    code.py:3:17: E999 SyntaxError: invalid syntax
    code.py:6:19: E711 comparison to None should be 'if cond is None:'` 
    ```

    还显示了一个输出示例。

**注意**:输出的多余一行表示语法错误。

这些也可以作为对 [Atom](https://atom.io/packages/linter-flake8) 、 [Sublime Text](https://github.com/SublimeLinter/SublimeLinter-flake8) 、 [Visual Studio Code](https://code.visualstudio.com/docs/python/linting#_flake8) 和 [VIM](https://github.com/nvie/vim-flake8) 的扩展。您还可以找到关于为 Python 开发设置 [Sublime Text](https://realpython.com/setting-up-sublime-text-3-for-full-stack-python-development/) 和 [VIM](https://realpython.com/vim-and-python-a-match-made-in-heaven/#code-folding) 的指南，以及在 *Real Python* 上对一些流行的文本编辑器的[概述。](https://realpython.com/python-ides-code-editors-guide/)

### 自动套用格式器

自动套用格式程序是自动重构代码以符合 PEP 8 的程序。曾经这样的程序是 [`black`](https://pypi.org/project/black/) ，它自动套用符合 PEP 8 中*大部分*规则的代码。一个很大的不同是，它将行长度限制为 88 个字符，而不是 79 个字符。但是，您可以通过添加命令行标志来覆盖它，如下例所示。

使用`pip`安装`black`。它需要 Python 3.6+才能运行:

```py
$ pip install black
```

它可以通过命令行运行，就像 linters 一样。假设您从一个名为`code.py`的文件中的以下不符合 PEP 8 的代码开始:

```py
for i in range(0,3):
    for j in range(0,3):
        if (i==2):
            print(i,j)
```

然后，您可以通过命令行运行以下命令:

```
$ black code.py
reformatted code.py
All done! ✨ 🍰 ✨
```py

`code.py`会自动重新格式化成这样:

```
for i in range(0, 3):
    for j in range(0, 3):
        if i == 2:
            print(i, j)
```py

如果你想改变行长度限制，那么你可以使用`--line-length`标志:

```
$ black --line-length=79 code.py
reformatted code.py
All done! ✨ 🍰 ✨
```

另外两个自动套用格式器， [`autopep8`](https://pypi.org/project/autopep8/) 和 [`yapf`](https://pypi.org/project/yapf/) ，执行与`black`类似的操作。

另一个*真正的 Python* 教程，Alexander van Tol 的 [Python 代码质量:工具&最佳实践](https://realpython.com/python-code-quality/)，给出了如何使用这些工具的完整解释。

[*Remove ads*](/account/join/)

## 结论

现在，您知道了如何使用 PEP 8 中的指导方针编写高质量、可读的 Python 代码。虽然这些指导方针可能看起来很迂腐，但是遵循它们确实可以改进您的代码，尤其是当涉及到与潜在的雇主或合作者共享您的代码时。

在本教程中，您学习了:

*   PEP 8 是什么，为什么存在
*   为什么你应该写 PEP 8 兼容的代码
*   如何编写符合 PEP 8 的代码

除此之外，您还看到了如何使用 linters 和 autoformatters 根据 PEP 8 指南检查您的代码。

如果你想了解更多关于 PEP 8 的信息，那么你可以阅读[完整的文档](https://www.python.org/dev/peps/pep-0008/)，或者访问【pep8.org】的，它包含相同的信息，但是格式很好。在这些文档中，你会发现 PEP 8 指南的其余部分在本教程中没有涉及。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和编写的教程一起看，加深理解: [**用 PEP 8**](/courses/writing-beautiful-python-code-pep-8/) 编写漂亮的 Pythonic 代码**********