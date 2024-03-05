# Python 代码质量:工具和最佳实践

> 原文：<https://realpython.com/python-code-quality/>

在本文中，我们将识别高质量的 Python 代码，并向您展示如何提高您自己代码的质量。

我们将分析和比较您可以用来将代码提升到下一个级别的工具。无论您使用 Python 已经有一段时间了，还是刚刚开始，您都可以从这里讨论的实践和工具中受益。

## 什么是代码质量？

你当然想要高质量的代码，谁不想呢？但是为了提高代码质量，我们必须定义它是什么。

快速的谷歌搜索会产生许多定义代码质量的结果。事实证明，这个词对人们来说有很多不同的含义。

定义代码质量的一种方法是着眼于光谱的一端:高质量的代码。希望您能同意以下高质量的代码标识符:

*   它做它应该做的事情。
*   它不包含缺陷或问题。
*   它易于阅读、维护和扩展。

这三个标识符虽然简单，但似乎得到了普遍认同。为了进一步扩展这些想法，让我们深入了解为什么每一个在软件领域都很重要。

[*Remove ads*](/account/join/)

## 为什么代码质量很重要？

为了确定为什么高质量的代码是重要的，让我们重温一下这些标识符。我们将看到当代码不满足它们时会发生什么。

### 它没有做它应该做的事情

满足需求是任何产品、软件等的基础。我们制作软件来做一些事情。如果最后，它没有做到…嗯，它肯定不是高质量的。如果达不到基本要求，甚至很难称之为低质量。

### 是否包含缺陷和问题

如果你正在使用的东西有问题或者给你带来问题，你可能不会称之为高质量。事实上，如果它足够糟糕，你可能会完全停止使用它。

为了不用软件做例子，假设你的吸尘器在普通地毯上效果很好。它能清理所有的灰尘和猫毛。一个灾难性的夜晚，猫打翻了一株植物，把泥土洒得到处都是。当你试图用吸尘器清理这堆脏东西时，它坏了，把脏东西弄得到处都是。

虽然真空吸尘器在某些情况下工作，但它不能有效地处理偶尔的额外负载。因此，你不会称之为高品质的吸尘器。

这是我们希望在代码中避免的问题。如果事情在边缘情况下破裂，缺陷导致不必要的行为，我们就没有高质量的产品。

### 很难读取、维护或扩展

想象一下:一个客户请求一个新特性。写原始代码的人走了。替换它们的人现在必须理解已经存在的代码。那个人就是你。

如果代码很容易理解，你就能更快地分析问题并提出解决方案。如果代码复杂且令人费解，您可能会花费更长的时间，并可能做出一些错误的假设。

如果能在不破坏原有功能的情况下轻松添加新功能，那也不错。如果代码不容易扩展，你的新特性可能会破坏其他东西。

没有人*希望*处于必须阅读、维护或扩展低质量代码的位置。这对每个人来说意味着更多的头痛和更多的工作。

你不得不处理低质量的代码已经够糟糕了，但是不要让别人处于同样的情况。您可以提高自己编写的代码的质量。

如果你和一个开发团队一起工作，你可以开始实施一些方法来确保更好的整体代码质量。当然，前提是你有他们的支持。你可能需要赢得一些人的支持(请随意发送这篇文章给他们😃).

## 如何提高 Python 代码质量

在我们追求高质量代码的过程中，有一些事情需要考虑。首先，这个旅程不是一个纯粹客观的旅程。对于高质量的代码是什么样子，有一些强烈的感觉。

虽然每个人都有希望在上面提到的标识符上达成一致，但是他们实现的方式是一条主观的道路。当您谈到实现可读性、可维护性和可扩展性时，通常会出现一些最固执己见的话题。

所以请记住，虽然本文将试图保持客观，但当涉及到代码时，有一个非常固执己见的世界。

所以，让我们从最固执己见的话题开始:代码风格。

[*Remove ads*](/account/join/)

### 风格指南

啊，是的。古老的问题:[空格还是制表符](https://blog.codinghorror.com/death-to-the-space-infidels/)？

不管你个人对如何表示空白有什么看法，可以有把握地假设你至少想要代码的一致性。

风格指南的目的是定义一种一致的方式来编写代码。通常这都是修饰性的，意味着它不会改变代码的逻辑结果。尽管如此，一些文体选择确实避免了常见的逻辑错误。

风格指南有助于实现使代码易于阅读、维护和扩展的目标。

就 Python 而言，有一个广为接受的标准。它部分是由 Python 编程语言本身的作者编写的。

[PEP 8](http://pep8.org/) 提供了 Python 代码的编码约定。Python 代码遵循这种风格指南是相当常见的。这是一个很好的起点，因为它已经定义好了。

一个姐妹 Python 增强提案， [PEP 257](https://www.python.org/dev/peps/pep-0257/) 描述了 Python 文档字符串的约定，这些字符串旨在[记录](https://realpython.com/documenting-python-code/)模块、类、函数和方法。额外的好处是，如果 docstrings 是一致的，有工具能够直接从代码生成文档。

这些指南所做的就是*定义*一种样式代码的方式。但是你如何执行它呢？那么代码中的缺陷和问题是什么呢，你如何发现它们呢？这就是棉绒的由来。

### 棉绒

#### 什么是棉绒？

首先，我们来说说 lint。那些微小的，恼人的小瑕疵不知何故布满了你的衣服。没有那些线头，衣服看起来和感觉都好多了。你的代码没有什么不同。小错误、风格不一致和危险的逻辑不会让你的代码感觉很棒。

但是我们都会犯错。你不能指望自己总能及时抓住他们。输入错误的[变量](https://realpython.com/python-variables/)名称，忘记了右括号，Python 中不正确的跳转，用错误数量的参数调用函数，等等。Linters 有助于识别这些问题区域。

此外，[大多数编辑器和 ide](https://realpython.com/python-ides-code-editors-guide/)能够在你输入的时候在后台运行 linters。这就产生了一个能够在运行代码之前突出显示、强调或识别代码中问题区域的环境。这就像是高级的代码拼写检查。它用弯弯曲曲的红线强调问题，就像你最喜欢的文字处理器一样。

Linters 分析代码以检测各种类别的 lint。这些类别可以大致定义如下:

1.  逻辑 Lint
    *   代码错误
    *   具有潜在意外结果的代码
    *   危险的代码模式
2.  文体线头
    *   不符合规定惯例的代码

还有一些代码分析工具可以提供对代码的其他洞察。虽然根据定义可能不是 linters，但是这些工具通常与 linters 一起使用。他们也希望提高代码的质量。

最后，还有一些工具可以自动将代码格式化为某种规格。这些自动化工具确保了我们劣等的人类头脑不会搞乱惯例。

#### Python 有哪些 Linter 选项？

在深入研究您的选择之前，重要的是要认识到一些“棉绒”只是多个棉绒很好地包装在一起。这些组合棉绒的一些流行例子如下:

**Flake8** :能够检测逻辑和风格 lint。它将 pycodestyle 的样式和复杂性检查添加到 PyFlakes 的逻辑 lint 检测中。它结合了以下棉绒:

*   PyFlakes
*   pycodestyle(以前为 pep8)
*   麦凯布

**Pylama** :由大量 linters 等工具组成的代码审计工具，用于分析代码。它结合了以下内容:

*   pycodestyle(以前为 pep8)
*   pydocstyle(原 pep257)
*   PyFlakes
*   麦凯布
*   Pylint
*   氡
*   jslint

以下是一些独立的棉绒分类和简要说明:

| 棉绒 | 种类 | 描述 |
| --- | --- | --- |
| [皮林特](https://www.pylint.org/) | 逻辑和风格 | 检查错误，尝试执行编码标准，寻找代码味道 |
| [PyFlakes](https://github.com/PyCQA/pyflakes) | 逻辑学的 | 分析程序并检测各种错误 |
| [pycodestyle](https://github.com/PyCQA/pycodestyle) | 体裁上的 | 对照 PEP 8 中的一些样式约定进行检查 |
| [pydocstyle](https://github.com/PyCQA/pydocstyle) | 体裁上的 | 检查是否符合 Python 文档字符串约定 |
| [土匪](https://github.com/PyCQA/bandit) | 逻辑学的 | 分析代码以发现常见的安全问题 |
| [MyPy](http://mypy-lang.org/) | 逻辑学的 | 检查可选的强制静态类型 |

这里有一些代码分析和格式化工具:

| 工具 | 种类 | 描述 |
| --- | --- | --- |
| [麦凯布](https://github.com/PyCQA/mccabe) | 分析的 | 检查[麦凯布复杂度](https://en.wikipedia.org/wiki/Cyclomatic_complexity) |
| [氡](http://radon.readthedocs.io/en/latest/) | 分析的 | 分析代码的各种度量(代码行数、复杂性等) |
| [黑色](https://github.com/ambv/black) | 格式程序 | 毫不妥协地格式化 Python 代码 |
| [Isort](https://github.com/timothycrosley/isort) | 格式程序 | 通过按字母顺序排序并分成几个部分来格式化导入 |

#### 比较 Python 短绒

让我们更好地了解不同的棉绒能够捕捉什么以及输出是什么样的。为此，我用默认设置在一些不同的 linters 上运行了相同的代码。

下面是我在 linters 中运行的代码。它包含各种逻辑和风格问题:



```py
 1"""
 2code_with_lint.py
 3Example Code with lots of lint!
 4"""
 5import io
 6from math import *
 7
 8
 9from time import time
10
11some_global_var = 'GLOBAL VAR NAMES SHOULD BE IN ALL_CAPS_WITH_UNDERSCOES'
12
13def multiply(x, y):
14    """
15 This returns the result of a multiplation of the inputs
16 """
17    some_global_var = 'this is actually a local variable...'
18    result = x* y
19    return result
20    if result == 777:
21        print("jackpot!")
22
23def is_sum_lucky(x, y):
24    """This returns a string describing whether or not the sum of input is lucky
25 This function first makes sure the inputs are valid and then calculates the
26 sum. Then, it will determine a message to return based on whether or not
27 that sum should be considered "lucky"
28 """
29    if x != None:
30        if y is not None:
31            result = x+y;
32            if result == 7:
33                return 'a lucky number!'
34            else:
35                return( 'an unlucky number!')
36
37            return ('just a normal number')
38
39class SomeClass:
40
41    def __init__(self, some_arg,  some_other_arg, verbose = False):
42        self.some_other_arg  =  some_other_arg
43        self.some_arg        =  some_arg
44        list_comprehension = [((100/value)*pi) for value in some_arg if value != 0]
45        time = time()
46        from datetime import datetime
47        date_and_time = datetime.now()
48        return
```

下面的比较显示了我在分析上述文件时使用的 linters 及其运行时。我应该指出，这些并不完全可比，因为它们服务于不同的目的。例如，PyFlakes 不像 Pylint 那样识别风格错误。

| 棉绒 | 命令 | 时间 |
| --- | --- | --- |
| [皮林特](https://www.pylint.org/) | pylint code_with_lint.py | 1.16 秒 |
| [PyFlakes](https://github.com/PyCQA/pyflakes) | pyflakes code_with_lint.py | 0.15 秒 |
| [pycodestyle](https://github.com/PyCQA/pycodestyle) | pycodestyle code_with_lint.py | 0.14 秒 |
| [pydocstyle](https://github.com/PyCQA/pydocstyle) | pydocstyle code_with_lint.py | 0.21 秒 |

有关每个的输出，请参见下面的部分。

##### Pylint

皮林特是最古老的棉绒之一(大约 2006 年)，现在仍然维护得很好。有些人可能会称这个软件久经沙场。它已经存在了足够长的时间，贡献者已经修复了大多数主要的 bug，核心特性也已经开发得很好了。

对 Pylint 的常见抱怨是它很慢，默认情况下过于冗长，并且需要大量的配置才能让它按照您想要的方式工作。除了速度慢之外，其他的抱怨有点像一把双刃剑。啰嗦可以是因为彻底。大量的配置意味着对你的偏好有很大的适应性。

事不宜迟，对上面填充了 lint 的代码运行 Pylint 后的输出:

```py
No config file found, using default configuration
************* Module code_with_lint
W: 23, 0: Unnecessary semicolon (unnecessary-semicolon)
C: 27, 0: Unnecessary parens after 'return' keyword (superfluous-parens)
C: 27, 0: No space allowed after bracket
                return( 'an unlucky number!')
                      ^ (bad-whitespace)
C: 29, 0: Unnecessary parens after 'return' keyword (superfluous-parens)
C: 33, 0: Exactly one space required after comma
    def __init__(self, some_arg,  some_other_arg, verbose = False):
                               ^ (bad-whitespace)
C: 33, 0: No space allowed around keyword argument assignment
    def __init__(self, some_arg,  some_other_arg, verbose = False):
                                                          ^ (bad-whitespace)
C: 34, 0: Exactly one space required around assignment
        self.some_other_arg  =  some_other_arg
                             ^ (bad-whitespace)
C: 35, 0: Exactly one space required around assignment
        self.some_arg        =  some_arg
                             ^ (bad-whitespace)
C: 40, 0: Final newline missing (missing-final-newline)
W:  6, 0: Redefining built-in 'pow' (redefined-builtin)
W:  6, 0: Wildcard import math (wildcard-import)
C: 11, 0: Constant name "some_global_var" doesn't conform to UPPER_CASE naming style (invalid-name)
C: 13, 0: Argument name "x" doesn't conform to snake_case naming style (invalid-name)
C: 13, 0: Argument name "y" doesn't conform to snake_case naming style (invalid-name)
C: 13, 0: Missing function docstring (missing-docstring)
W: 14, 4: Redefining name 'some_global_var' from outer scope (line 11) (redefined-outer-name)
W: 17, 4: Unreachable code (unreachable)
W: 14, 4: Unused variable 'some_global_var' (unused-variable)
...
R: 24,12: Unnecessary "else" after "return" (no-else-return)
R: 20, 0: Either all return statements in a function should return an expression, or none of them should. (inconsistent-return-statements)
C: 31, 0: Missing class docstring (missing-docstring)
W: 37, 8: Redefining name 'time' from outer scope (line 9) (redefined-outer-name)
E: 37,15: Using variable 'time' before assignment (used-before-assignment)
W: 33,50: Unused argument 'verbose' (unused-argument)
W: 36, 8: Unused variable 'list_comprehension' (unused-variable)
W: 39, 8: Unused variable 'date_and_time' (unused-variable)
R: 31, 0: Too few public methods (0/2) (too-few-public-methods)
W:  5, 0: Unused import io (unused-import)
W:  6, 0: Unused import acos from wildcard import (unused-wildcard-import)
...
W:  9, 0: Unused time imported from time (unused-import)
```

请注意，我用省略号对类似的行进行了压缩。这很难理解，但是在这段代码中有很多琐碎的东西。

注意，Pylint 在每个问题区域前面加上了一个`R`、`C`、`W`、`E`或`F`，意思是:

*   “良好实践”度量违规的因子
*   违反编码标准的规定
*   注意文体问题或小的编程问题
*   重要编程问题的错误(即最有可能的错误)
*   [F]防止进一步处理的错误

以上列表直接来自 Pylint 的[用户指南](http://pylint.pycqa.org/en/latest/user_guide/output.html)。

##### PyFlakes

Pyflakes“做出一个简单的承诺:它永远不会抱怨风格，它会非常非常努力地尝试永远不会发出误报”。这意味着 Pyflakes 不会告诉您缺少文档字符串或不符合命名风格的参数名称。它主要关注逻辑代码问题和潜在的错误。

这里的好处是速度。PyFlakes 的运行时间是 Pylint 的一小部分。

对上面填充了 lint 的代码运行后的输出:

```py
code_with_lint.py:5: 'io' imported but unused
code_with_lint.py:6: 'from math import *' used; unable to detect undefined names
code_with_lint.py:14: local variable 'some_global_var' is assigned to but never used
code_with_lint.py:36: 'pi' may be undefined, or defined from star imports: math
code_with_lint.py:36: local variable 'list_comprehension' is assigned to but never used
code_with_lint.py:37: local variable 'time' (defined in enclosing scope on line 9) referenced before assignment
code_with_lint.py:37: local variable 'time' is assigned to but never used
code_with_lint.py:39: local variable 'date_and_time' is assigned to but never used
```

这里的缺点是解析这个输出可能有点困难。各种问题和错误没有按类型进行标记或组织。取决于你如何使用它，这可能根本不是问题。

##### pycodestyle(原 pep8)

用于检查 [PEP8](http://pep8.org/) 的一些样式约定。不检查命名约定，也不检查文档字符串。它捕捉到的错误和警告被分类在[这个表](https://pycodestyle.readthedocs.io/en/latest/intro.html#error-codes)中。

对上面填充了 lint 的代码运行后的输出:

```py
code_with_lint.py:13:1: E302 expected 2 blank lines, found 1
code_with_lint.py:15:15: E225 missing whitespace around operator
code_with_lint.py:20:1: E302 expected 2 blank lines, found 1
code_with_lint.py:21:10: E711 comparison to None should be 'if cond is not None:'
code_with_lint.py:23:25: E703 statement ends with a semicolon
code_with_lint.py:27:24: E201 whitespace after '('
code_with_lint.py:31:1: E302 expected 2 blank lines, found 1
code_with_lint.py:33:58: E251 unexpected spaces around keyword / parameter equals
code_with_lint.py:33:60: E251 unexpected spaces around keyword / parameter equals
code_with_lint.py:34:28: E221 multiple spaces before operator
code_with_lint.py:34:31: E222 multiple spaces after operator
code_with_lint.py:35:22: E221 multiple spaces before operator
code_with_lint.py:35:31: E222 multiple spaces after operator
code_with_lint.py:36:80: E501 line too long (83 > 79 characters)
code_with_lint.py:40:15: W292 no newline at end of file
```

这个输出的好处是 lint 是按类别标记的。如果您不在乎遵守特定的约定，也可以选择忽略某些错误。

##### pydocstyle(原 pep257)

与 pycodestyle 非常相似，除了它不是根据 PEP8 代码样式约定进行检查，而是根据来自 [PEP257](https://www.python.org/dev/peps/pep-0257/) 的约定检查 docstrings。

对上面填充了 lint 的代码运行后的输出:

```py
code_with_lint.py:1 at module level:
        D200: One-line docstring should fit on one line with quotes (found 3)
code_with_lint.py:1 at module level:
        D400: First line should end with a period (not '!')
code_with_lint.py:13 in public function `multiply`:
        D103: Missing docstring in public function
code_with_lint.py:20 in public function `is_sum_lucky`:
        D103: Missing docstring in public function
code_with_lint.py:31 in public class `SomeClass`:
        D101: Missing docstring in public class
code_with_lint.py:33 in public method `__init__`:
        D107: Missing docstring in __init__
```

同样，像 pycodestyle 一样，pydocstyle 对它发现的各种错误进行标记和分类。该列表与 pycodestyle 中的任何内容都不冲突，因为所有错误都以 docstring 的`D`为前缀。这些错误的列表可以在[这里](http://www.pydocstyle.org/en/latest/error_codes.html)找到。

##### 无绒毛代码

您可以根据 linter 的输出来调整之前填充了 lint 的代码，最终会得到如下结果:



```py
 1"""Example Code with less lint."""
 2
 3from math import pi
 4from time import time
 5from datetime import datetime
 6
 7SOME_GLOBAL_VAR = 'GLOBAL VAR NAMES SHOULD BE IN ALL_CAPS_WITH_UNDERSCOES'
 8
 9
10def multiply(first_value, second_value):
11    """Return the result of a multiplation of the inputs."""
12    result = first_value * second_value
13
14    if result == 777:
15        print("jackpot!")
16
17    return result
18
19
20def is_sum_lucky(first_value, second_value):
21    """
22 Return a string describing whether or not the sum of input is lucky.
23
24 This function first makes sure the inputs are valid and then calculates the
25 sum. Then, it will determine a message to return based on whether or not
26 that sum should be considered "lucky".
27 """
28    if first_value is not None and second_value is not None:
29        result = first_value + second_value
30        if result == 7:
31            message = 'a lucky number!'
32        else:
33            message = 'an unlucky number!'
34    else:
35        message = 'an unknown number! Could not calculate sum...'
36
37    return message
38
39
40class SomeClass:
41    """Is a class docstring."""
42
43    def __init__(self, some_arg, some_other_arg):
44        """Initialize an instance of SomeClass."""
45        self.some_other_arg = some_other_arg
46        self.some_arg = some_arg
47        list_comprehension = [
48            ((100/value)*pi)
49            for value in some_arg
50            if value != 0
51        ]
52        current_time = time()
53        date_and_time = datetime.now()
54        print(f'created SomeClass instance at unix time: {current_time}')
55        print(f'datetime: {date_and_time}')
56        print(f'some calculated values: {list_comprehension}')
57
58    def some_public_method(self):
59        """Is a method docstring."""
60        pass
61
62    def some_other_public_method(self):
63        """Is a method docstring."""
64        pass
```

根据上面的棉绒，该代码是不起毛的。虽然逻辑本身基本上是无意义的，但您可以看到，至少一致性得到了加强。

在上面的例子中，我们在编写完所有代码后运行了 linters。然而，这并不是检查代码质量的唯一方法。

[*Remove ads*](/account/join/)

## 我什么时候可以检查我的代码质量？

您可以检查代码的质量:

*   当你写的时候
*   当它被检入时
*   当你进行测试的时候

让 linters 经常运行你的代码是很有用的。如果没有自动化和一致性，大型团队或项目很容易忽略目标，并开始创建质量较低的代码。当然，这是慢慢发生的。一些写得不好的逻辑，或者一些代码的格式与邻近的代码不匹配。随着时间的推移，所有的棉绒堆积起来。最终，你可能会陷入一些有问题的、难以阅读的、难以修复的、维护起来很痛苦的东西。

为了避免这种情况，经常检查代码质量！

### 正如你写的

您可以在编写代码时使用 linters，但是配置您的环境这样做可能需要一些额外的工作。这通常是一个为你的 IDE 或编辑器选择插件的问题。事实上，大多数 ide 已经内置了 linters。

以下是为各种编辑提供的关于 Python 林挺的一些一般信息:

*   [崇高的文字](https://realpython.com/setting-up-sublime-text-3-for-full-stack-python-development/)
*   [VS 代码](https://code.visualstudio.com/docs/python/linting)
*   [Atom](https://atom.io/packages/search?q=python+linter)
*   [Vim](https://realpython.com/vim-and-python-a-match-made-in-heaven/#syntax-checkinghighlighting)
*   [Emacs](https://realpython.com/emacs-the-best-python-editor/#additional-python-features)

### 在您签入代码之前

如果您正在使用 Git，可以设置 Git 挂钩在提交之前运行您的 linters。其他版本控制系统也有类似的方法，在系统中的某个操作之前或之后运行脚本。您可以使用这些方法来阻止任何不符合质量标准的新代码。

虽然这看起来有些极端，但是强制每一位代码通过 lint 筛选是确保持续质量的重要一步。在代码的前门自动进行筛选可能是避免代码中充满棉绒的最好方法。

### 运行测试时

你也可以将棉绒直接放入任何你可以用来持续集成的系统中。如果代码不符合质量标准，linters 可以被设置为构建失败。

同样，这似乎是一个极端的步骤，尤其是在现有代码中已经有很多 linter 错误的情况下。为了解决这个问题，一些持续集成系统将允许您选择只有在新代码增加了已经存在的 linter 错误的数量时才使构建失败。这样，您就可以开始提高质量，而无需对现有的代码库进行整体重写。

## 结论

高质量的代码做它应该做的事情而不会中断。它易于阅读、维护和扩展。它运行起来没有任何问题或缺陷，而且写得便于下一个人一起工作。

希望不言而喻，你应该努力拥有这样高质量的代码。幸运的是，有一些方法和工具可以帮助提高代码质量。

风格指南将为您的代码带来一致性。 [PEP8](http://pep8.org/) 是 Python 的一个伟大起点。Linters 将帮助您识别问题区域和不一致之处。您可以在整个开发过程中使用 linters，甚至可以自动标记 lint 填充的代码，以免发展太快。

让 linters 抱怨风格也避免了在代码评审期间讨论风格的需要。有些人可能会发现从这些工具而不是团队成员那里更容易得到坦诚的反馈。此外，一些团队成员可能不想在代码评审期间“挑剔”风格。Linters 避免政治，节省时间，并抱怨任何不一致。

此外，本文中提到的所有 linters 都有各种命令行选项和配置，允许您根据自己的喜好定制工具。你可以想多严格就多严格，也可以想多宽松就多宽松，这是要认识到的一件重要的事情。

提高代码质量是一个过程。您可以采取措施改进它，而不完全禁止所有不一致的代码。意识是伟大的第一步。只需要一个人，比如你，首先意识到高质量的代码有多重要。***