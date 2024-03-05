# Python 实践问题:解析 CSV 文件

> 原文：<https://realpython.com/python-interview-problem-parsing-csv-files/>

你是一名开发人员，在即将到来的面试之前，你在寻找一些使用逗号分隔值(CSV)文件的练习吗？本教程将引导您完成一系列 Python CSV 实践问题，帮助您做好准备。

本教程面向中级 Python 开发人员。它假设一个[Python 的基础知识](https://realpython.com/products/python-basics-book/)和处理 [CSV 文件](https://realpython.com/python-csv/)。和[其他练习题教程](https://realpython.com/python-practice-problems/)一样，这里列出的每个问题都有问题描述。您将首先看到问题陈述，然后有机会开发您自己的解决方案。

**在本教程中，您将探索:**

*   **编写使用 CSV 文件的代码**
*   用 pytest 做**测试驱动开发**
*   **讨论您的解决方案**和可能的改进
*   内置 CSV 模块和**熊猫**之间的权衡

通过单击下面的链接，您可以获得本教程中遇到的每个问题的单元测试失败的框架代码:

**获取源代码:** [单击此处获取源代码，您将在本教程中使用](https://realpython.com/bonus/interview-parsing-csv-code/)来练习解析 CSV 文件。

## Python CSV 解析:足球比分

你的第一个问题是关于英超联赛的排名。解决这个不需要什么专门的足球知识，Python 就行！

当你解决问题时，试着为每一点功能编写更多的单元测试，然后*编写功能以通过测试。这就是所谓的[测试驱动开发](https://realpython.com/courses/test-driven-development-pytest/)，这是一个展示你的编码和测试能力的好方法！*

[*Remove ads*](/account/join/)

### 问题描述

对于这一轮的问题，坚持标准库`csv`模块。稍后你会用[熊猫](https://pandas.pydata.org/)再拍一次。这是你的第一个问题:

> **找出最小目标差值**
> 
> 编写一个程序，在命令行上输入文件名并处理 CSV 文件的内容。内容将是英格兰超级联赛赛季末的足球排名。你的程序应该确定那个赛季哪个队的净胜球最少。
> 
> CSV 文件的第一行是列标题，随后的每一行显示一个团队的数据:
> 
> ```py
> `Team,Games,Wins,Losses,Draws,Goals For,Goals Against
> Arsenal,38,26,9,3,79,36` 
> ```
> 
> 标有`Goals For`和`Goals Against`的栏包含该赛季各队的总进球数。(所以阿森纳进了 79 个球，对他们进了 36 个球。)
> 
> 写个程序读取文件，然后打印出`Goals For`和`Goals Against`相差最小的队伍名称。用 pytest 创建单元测试来测试你的程序。

框架代码中提供了一个单元测试，用于测试您稍后将看到的问题陈述。您可以在编写解决方案时添加更多内容。还有两个 [pytest 夹具](https://realpython.com/pytest-python-testing/#fixtures-managing-state-and-dependencies)给定:

```py
# test_football_v1.py
import pytest
import football_v1 as fb

@pytest.fixture
def mock_csv_data():
    return [
        "Team,Games,Wins,Losses,Draws,Goals For,Goals Against",
        "Liverpool FC, 38, 32, 3, 3, 85, 33",
        "Norwich City FC, 38, 5, 27, 6, 26, 75",
    ]

@pytest.fixture
def mock_csv_file(tmp_path, mock_csv_data):
    datafile = tmp_path / "football.csv"
    datafile.write_text("\n".join(mock_csv_data))
    return str(datafile)
```

第一个 fixture 提供了一个由[字符串](https://realpython.com/python-strings/)组成的列表，这些字符串[模仿](https://realpython.com/python-mock-library/)真实的 CSV 数据，第二个 fixture 提供了一个由测试数据支持的文件名。字符串列表中的每个字符串代表测试文件中的一行。

**注意:**此处的解决方案将有一组非详尽的测试，仅证明基本功能。对于一个真实的系统，你可能想要一个更完整的[测试套件](https://en.wikipedia.org/wiki/Test_suite)，可能利用[参数化](https://docs.pytest.org/en/stable/parametrize.html)。

请记住，所提供的装置只是一个开始。在设计解决方案的每个部分时，添加使用它们的单元测试！

### 问题解决方案

这里讨论一下 *Real Python* 团队达成的解决方案，以及团队是如何达成的。

**注意:**记住，在您准备好查看每个 Python 练习问题的答案之前，不要打开下面折叠的部分！



乐谱解析的怎么样了？你准备好看到真正的 Python 团队给出的答案了吗？

在解决这个问题的过程中，该团队通过编写并多次重写代码，提出了几个解决方案。在面试中，你通常只有一次机会。在实时编码的情况下，您可以使用一种技术来解决这个问题，那就是花一点时间来讨论您现在可以使用的其他实现选项。

#### 解决方案 1

您将研究这个问题的两种不同的解决方案。您将看到的第一个解决方案运行良好，但仍有改进的空间。您将在这里使用测试驱动开发(TDD)模型，因此您不会首先查看完整的解决方案，而只是查看解决方案的整体计划。

将解决方案分成几个部分允许您在编写代码之前为每个部分编写单元测试。这是该解决方案的大致轮廓:

1.  在生成器中读取并解析 CSV 文件的每一行。
2.  计算给定线的队名和分数差。
3.  求最小分数差。

让我们从第一部分开始，一次一行地读取和解析文件。您将首先为该操作构建测试。

##### 读取并解析

给定问题的描述，您提前知道列是什么，所以您不需要输出中的第一行标签。您还知道每一行数据都有七个字段，因此您可以测试您的解析函数[是否返回](https://realpython.com/python-return-statement/)一个行列表，每一行都有七个条目:

```py
# test_football_v1.py
import pytest
import football_v1 as fb

# ...

def test_parse_next_line(mock_csv_data):
    all_lines = [line for line in fb.parse_next_line(mock_csv_data)]
    assert len(all_lines) == 2
    for line in all_lines:
        assert len(line) == 7
```

您可以看到这个测试使用了您的第一个 pytest fixture，它提供了一个 CSV 行列表。这个测试利用了 CSV 模块可以解析一个[列表对象](https://realpython.com/python-lists-tuples/)或者一个[文件对象](https://realpython.com/working-with-files-in-python/)的事实。这对于您的测试来说非常方便，因为您还不必担心管理文件对象。

测试使用一个[列表理解](https://realpython.com/list-comprehension-python/)来读取从`parse_next_line()`开始的所有行，这将是一个[生成器](https://realpython.com/introduction-to-python-generators/)。然后，它断言列表中的几个属性:

*   列表中有两个条目。
*   每个条目本身是一个包含七个项目的列表。

现在您有了一个测试，您可以运行它来确认它是否运行以及它是否如预期的那样失败:

```py
$ pytest test_football_v1.py
============================= test session starts ==============================
platform linux -- Python 3.7.1, pytest-6.2.1, py-1.10.0, pluggy-0.13.1
rootdir: /home/jima/coding/realPython/articles/jima-csv
collected 1 item

test_football_v1.py F                                                   [100%]

=================================== FAILURES ===================================
_______________________________ test_parse_next_line ___________________________

mock_csv_data = ['Team,Games,Wins,Losses,Draws,Goals For,Goals Against', ....

 def test_parse_next_line(mock_csv_data ):
>       all_lines = [line for line in fb.parse_next_line(mock_csv_data)]
E       AttributeError: module 'football_v1' has no attribute 'parse_next_line'

test_football_csv.py:30: AttributeError
=========================== short test summary info ============================
FAILED test_football_v1.py::test_parse_next_line - AttributeError: module 'fo...
============================== 1 failed in 0.02s ===============================
```

测试失败是因为`parse_next_line()`是未定义的，考虑到您还没有编写它，这是有意义的。当你知道测试会失败时运行测试会给你信心，当测试最终通过时，你所做的改变就是修复它们的原因。

**注意:**上面的 pytest 输出假设您有一个名为`football_v1.py`的文件，但是它不包含函数`parse_next_line()`。如果你没有这个文件，你可能会得到一个错误提示`ModuleNotFoundError: No module named 'football_v1'`。

接下来你要写缺失的`parse_next_line()`。这个函数将是一个生成器，返回文件中每一行的解析版本。您需要添加一些代码来跳过标题:

```py
# football_v1.py
import csv

def parse_next_line(csv_file):
    for line in csv.DictReader(csv_file):
        yield line
```

该函数首先创建一个`csv.DictReader()`，它是 CSV 文件的迭代器。`DictReader`使用标题行作为它创建的[字典](https://realpython.com/python-dicts/)的关键字。文件的每一行都用这些键和相应的值构建了一个字典。这个字典是用来创建您的生成器的。

现在用您的单元测试来尝试一下:

```py
$ pytest test_football_v1.py
============================= test session starts ==============================
platform linux -- Python 3.7.1, pytest-6.2.1, py-1.10.0, pluggy-0.13.1
rootdir: /home/jima/coding/realPython/articles/jima-csv
collected 1 item

test_football_v1.py .                                                    [100%]

============================== 1 passed in 0.01s ===============================
```

太棒了。您的第一个功能块正在工作。您知道您添加的代码是使测试通过的原因。现在你可以进入下一步，计算给定线的分数差。

##### 计算微分

该函数将获取由`parse_next_line()`解析的值列表，并计算分数差`Goals For - Goals Against`。这就是那些具有少量代表性数据的测试装置将会有所帮助的地方。您可以手动计算测试数据中两条线的差异，得到利物浦足球俱乐部的差异为 52，诺里奇城足球俱乐部的差异为 49。

这个测试将使用您刚刚完成的生成器函数从测试数据中提取每一行:

```py
# test_football_v1.py
import pytest
import football_v1 as fb

# ...

def test_get_score_difference(mock_csv_data):
    reader = fb.parse_next_line(mock_csv_data)
    assert fb.get_name_and_diff(next(reader)) == ("Liverpool FC", 52)
    assert fb.get_name_and_diff(next(reader)) == ("Norwich City FC", 49)
```

首先创建刚刚测试过的生成器，然后使用`next()`遍历两行测试数据。 [`assert`语句](https://realpython.com/python-assert-statement/)测试每个手工计算的值是否正确。

和以前一样，一旦有了测试，就可以运行它以确保它失败:

```py
$ pytest test_football_v1.py
============================= test session starts ==============================
platform linux -- Python 3.7.1, pytest-6.2.1, py-1.10.0, pluggy-0.13.1
rootdir: /home/jima/coding/realPython/articles/jima-csv
collected 2 items

test_football_v1.py .F                                                   [100%]

=================================== FAILURES ===================================
__________________________ test_get_score_difference ___________________________

mock_csv_data = ['Team,Games,Wins,Losses,Draws,Goals For,Goals Against', ...

 def test_get_score_difference(mock_csv_data):
 reader = fb.parse_next_line(mock_csv_data)
>       team, diff = fb.get_name_and_diff(next(reader))
E       AttributeError: module 'football_v1' has no attribute 'get_name_and ...

test_football_v1.py:38: AttributeError
=========================== short test summary info ============================
FAILED test_football_v1.py::test_get_score_difference - AttributeError: modul...
========================= 1 failed, 1 passed in 0.03s ==========================
```

现在测试已经就绪，看看`get_name_and_diff()`的实现。由于`DictReader`为您将 CSV 值放入字典中，您可以从每个字典中检索团队名称并计算目标差异:

```py
# football_v1.py
def get_name_and_diff(team_stats):
    diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"])
    return team_stats["Team"], diff
```

您可以把它写成一行程序，但是把它分成几个清晰的字段可能会提高可读性。它还可以使调试这段代码变得更容易。如果你在面试中现场编码，这些都是很好的提点。表明你对可读性有所考虑会有所不同。

现在您已经实现了这个功能，您可以重新运行您的测试:

```py
$ pytest test_football_v1.py
============================= test session starts ==============================
platform linux -- Python 3.7.1, pytest-6.2.1, py-1.10.0, pluggy-0.13.1
rootdir: /home/jima/coding/realPython/articles/jima-csv
collected 2 items

test_football_v1.py .F                                                   [100%]

=================================== FAILURES ===================================
__________________________ test_get_score_difference ___________________________

mock_csv_data = ['Team,Games,Wins,Losses,Draws,Goals For,Goals Against', ...

 def test_get_score_difference(mock_csv_data):
 reader = fb.parse_next_line(mock_csv_data)
 assert fb.get_name_and_diff(next(reader)) == ("Liverpool FC", 52)
>       assert fb.get_name_and_diff(next(reader)) == ("Norwich City FC", 49)
E       AssertionError: assert ('Norwich City FC', -49) == ('Norwich City FC'...
E         At index 1 diff: -49 != 49
E         Use -v to get the full diff

test_football_v1.py:40: AssertionError
=========================== short test summary info ============================
FAILED test_football_v1.py::test_get_score_difference - AssertionError: asser...
========================= 1 failed, 1 passed in 0.07s ==========================
```

哎呦！这是不对的。该函数返回的差值不应为负。还好你写了测试！

您可以通过在返回值上使用 [`abs()`](https://docs.python.org/3.8/library/functions.html#abs) 来更正:

```py
# football_v1.py
def get_name_and_diff(team_stats):
    diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"])
 return team_stats["Team"], abs(diff)
```

你可以在函数的最后一行看到它现在调用了`abs(diff)`，所以你不会得到负数的结果。现在用您的测试来尝试这个版本，看看它是否通过:

```py
$ pytest test_football_v1.py
============================= test session starts ==============================
platform linux -- Python 3.7.1, pytest-6.2.1, py-1.10.0, pluggy-0.13.1
rootdir: /home/jima/coding/realPython/articles/jima-csv
collected 2 items

test_football_v1.py ..                                                   [100%]

============================== 2 passed in 0.01s ===============================
```

那好多了。如果你想找到净胜球差距最小的球队，你就需要差距的绝对值。

##### 查找最小值

对于您的最后一块拼图，您需要一个函数，它使用您的生成器获取 CSV 文件的每一行，并使用您的函数返回每一行的球队名称和得分差异，然后找到这些差异的最小值。对此的测试是框架代码中给出的总体测试:

```py
# test_football_v1.py
import pytest
import football_v1 as fb

# ...

def test_get_min_score(mock_csv_file):
    assert fb.get_min_score_difference(mock_csv_file) == (
        "Norwich City FC",
        49,
    )
```

您再次使用提供的 pytest fixtures，但是这一次您使用`mock_csv_file` fixture 来获取一个文件的文件名，该文件包含您到目前为止一直在使用的相同的测试数据集。该测试调用您的最终函数，并断言您手动计算的正确答案:诺维奇城队以 49 球的比分差距最小。

至此，您已经看到在被测试的函数实现之前测试失败了，所以您可以跳过这一步，直接跳到您的解决方案:

```py
# football_v1.py
def get_min_score_difference(filename):
    with open(filename, "r", newline="") as csv_file:
        min_diff = 10000
        min_team = None
        for line in parse_next_line(csv_file):
            team, diff = get_name_and_diff(line)
            if diff < min_diff:
                min_diff = diff
                min_team = team
    return min_team, min_diff
```

该函数使用[上下文管理器](https://realpython.com/courses/python-context-managers-and-with-statement/)打开给定的 CSV 文件进行读取。然后它设置`min_diff`和`min_team`变量，您将使用它们来跟踪您在遍历列表时找到的最小值。你在`10000`开始最小差异，这对于足球比分似乎是安全的。

然后，该函数遍历每一行，获取团队名称和差异，并找到差异的最小值。

当您针对测试运行此代码时，它会通过:

```py
 $ pytest test_football_v1.py
============================= test session starts ==============================
platform linux -- Python 3.7.1, pytest-6.2.1, py-1.10.0, pluggy-0.13.1
rootdir: /home/jima/coding/realPython/articles/jima-csv
collected 3 items

test_football_v1.py ...                                                  [100%]

============================== 3 passed in 0.03s ===============================
```

恭喜你！您已经找到了所述问题的解决方案！

一旦你做到了这一点，尤其是在面试的情况下，是时候检查你的解决方案，看看你是否能找出让代码更可读、更健壮或更 T2 的变化。这是您将在下一部分中执行的操作。

#### 解决方案 2:重构解决方案 1

从整体上看一下你对这个问题的第一个解决方案:

```py
# football_v1.py
import csv

def parse_next_line(csv_file):
    for line in csv.DictReader(csv_file):
        yield line

def get_name_and_diff(team_stats):
    diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"])
    return team_stats["Team"], abs(diff)

def get_min_score_difference(filename):
    with open(filename, "r", newline="") as csv_file:
        min_diff = 10000
        min_team = None
        for line in parse_next_line(csv_file):
            team, diff = get_name_and_diff(line)
            if diff < min_diff:
                min_diff = diff
                min_team = team
    return min_team, min_diff
```

从整体上看这段代码，有一些事情需要注意。其中之一是`get_name_and_diff()`并没有做那么多。它只从字典中取出三个字段并减去。第一个函数`parse_next_line()`也相当短，似乎可以将这两个函数结合起来，让生成器只返回球队名称和分数差。

您可以将这两个函数重构为一个名为`get_next_name_and_diff()`的新函数。如果你跟随本教程，现在是一个很好的时机将`football_v1.py`复制到`football_v2.py`并对测试文件做类似的操作。坚持您的 TDD 过程，您将重用您的第一个解决方案的测试:

```py
# test_football_v2.py
import pytest
import football_v2 as fb

 # ...

def test_get_min_score(mock_csv_file):
    assert fb.get_min_score_difference(mock_csv_file) == (
        "Norwich City FC",
        49,
    )

def test_get_score_difference(mock_csv_data):
 reader = fb.get_next_name_and_diff(mock_csv_data) assert next(reader) == ("Liverpool FC", 52) assert next(reader) == ("Norwich City FC", 49) with pytest.raises(StopIteration): next(reader)
```

第一个测试`test_get_min_score()`保持不变，因为它测试的是最高级别的功能，这是不变的。

其他两个测试函数合并成一个函数，将返回的项目数和返回值的测试合并成一个测试。它借助 Python 内置的`next()`直接使用从`get_next_name_and_diff()`返回的生成器。

下面是将这两个非测试函数放在一起时的样子:

```py
# football_v2.py
import csv

def get_next_name_and_diff(csv_file):
    for team_stats in csv.DictReader(csv_file):
        diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"])
        yield team_stats["Team"], abs(diff)
```

这个函数看起来确实像前面的函数挤在一起。它使用`csv.DictReader()`,而不是产生从每一行创建的字典，只产生团队名称和计算的差异。

虽然就可读性而言，这并不是一个巨大的改进，但它将允许您在剩余的函数中做一些其他的简化。

剩下的功能`get_min_score_difference()`，也有一定的改进空间。手动遍历列表以找到最小值是标准库提供的功能。幸运的是，这是顶级功能，所以您的测试不需要更改。

如上所述，您可以使用 [`min()`](https://realpython.com/python-min-and-max/) 从标准库中找到列表或 iterable 中的最小项。“或可迭代”部分很重要。您的`get_next_name_and_diff()`生成器符合可迭代条件，因此`min()`将运行生成器并找到最小结果。

一个问题是`get_next_name_and_diff()`产生了`(team_name, score_differential)`个元组，并且您想要最小化差值。为了方便这个用例，`min()`有一个关键字参数，`key`。您可以提供一个函数，或者在您的情况下提供一个 [`lambda`](https://realpython.com/python-lambda/) ，来指示它将使用哪些值来搜索最小值:

```py
# football_v2.py
def get_min_score_difference(filename):
    with open(filename, "r", newline="") as csv_data:
 return min(get_next_name_and_diff(csv_data), key=lambda item: item[1])
```

这种变化将代码压缩成一个更小、更 Pythonic 化的函数。用于`key`的λ允许`min()`找到分数差的最小值。对新代码运行 pytest 表明，它仍然解决了上述问题:

```py
$ pytest test_football_v2.py
============================= test session starts ==============================
platform linux -- Python 3.7.1, pytest-6.2.1, py-1.10.0, pluggy-0.13.1
rootdir: /home/jima/coding/realPython/articles/jima-csv
collected 3 items

test_football_v2.py ...                                                  [100%]

============================== 3 passed in 0.01s ===============================
```

以这种方式花时间检查和重构代码在日常编码中是一个很好的实践，但在面试环境中可能实用，也可能不实用。即使你觉得在面试中没有时间或精力来完全重构你的解决方案，花一点时间向面试官展示你的想法也是值得的。

当你在面试时，花一分钟指出，“这些功能很小——我可以合并它们，”或者，“如果我推动这个显式循环，那么我可以使用`min()`功能”，这将向面试官展示你*知道*这些事情。没有人在第一次尝试时就能得出最优解。

面试中另一个值得讨论的话题是边角案例。解决方案能处理坏的数据线吗？像这样的主题有助于很好的测试，并且可以在早期发现很多问题。有时候在面试中讨论这些问题就足够了，有时候回去重构你的测试和代码来处理这些问题是值得的。

你可能还想讨论问题的定义。特别是这个问题有一个不明确的规范。如果两个队有相同的差距，解决方案应该是什么？您在这里看到的解决方案选择了第一个，但也有可能返回全部，或者最后一个，或者其他一些决定。

这种类型的模糊性在实际项目中很常见，因此认识到这一点并将其作为一个主题提出来可能表明您正在思考超越代码解决方案的问题。

既然您已经使用 Python `csv`模块解决了一个问题，那么就用一个类似的问题再试一次。

## Python CSV 解析:天气数据

你的第二个问题看起来和第一个很相似。使用类似的结构来解决它可能是个好主意。一旦你完成了这个问题的解决方案，你将会读到一些重构重用代码的想法，所以在工作中要记住这一点。

### 问题描述

这个问题涉及到解析 CSV 文件中的天气数据:

> **最高平均温度**
> 
> 编写一个程序，在命令行上输入文件名并处理 CSV 文件的内容。内容将是一个月的天气数据，每行一天。
> 
> 您的程序应该确定哪一天的平均温度最高，其中平均温度是当天最高温度和最低温度的平均值。这通常不是计算平均温度的方法，但在这个演示中是可行的。
> 
> CSV 文件的第一行是列标题:
> 
> ```py
> `Day,MaxT,MinT,AvDP,1HrP TPcn,PDir,AvSp,Dir,MxS,SkyC,MxR,Mn,R AvSLP
> 1,88,59,74,53.8,0,280,9.6,270,17,1.6,93,23,1004.5` 
> ```
> 
> 日期、最高温度和最低温度是前三列。
> 
> 用 pytest 编写单元测试来测试你的程序。

与足球比分问题一样，框架代码中提供了测试问题陈述的单元测试:

```py
# test_weather_v1.py
import pytest
import weather_v1 as wthr

@pytest.fixture
def mock_csv_data():
    return [
        "Day,MxT,MnT,AvT,AvDP,1HrP TPcn,PDir,AvSp,Dir,MxS,SkyC,MxR,Mn,R AvSLP",
        "1,88,59,74,53.8,0,280,9.6,270,17,1.6,93,23,1004.5",
        "2,79,63,71,46.5,0,330,8.7,340,23,3.3,70,28,1004.5",
    ]

@pytest.fixture
def mock_csv_file(tmp_path, mock_csv_data):
    datafile = tmp_path / "weather.csv"
    datafile.write_text("\n".join(mock_csv_data))
    return str(datafile)
```

再次注意，给出了两个装置。第一个提供模拟真实 CSV 数据的字符串列表，第二个提供由测试数据支持的文件名。字符串列表中的每个字符串代表测试文件中的一行。

请记住，所提供的装置只是一个开始。在设计解决方案的每个部分时添加测试！

[*Remove ads*](/account/join/)

### 问题解决方案

这里讨论一下真正的 Python 团队达成了什么。

**注意:**记住，在你准备好查看这个 Python 练习题的答案之前，不要打开下面折叠的部分！



您将在这里看到的解决方案与前面的解决方案非常相似。您看到了上面略有不同的一组测试数据。这两个测试函数基本上与足球解决方案相同:

```py
# test_weather_v1.py
import pytest
import weather_v1 as wthr

 # ...

def test_get_max_avg(mock_csv_file):
    assert wthr.get_max_avg(mock_csv_file) == (1, 73.5)

def test_get_next_day_and_avg(mock_csv_data):
    reader = wthr.get_next_day_and_avg(mock_csv_data)
    assert next(reader) == (1, 73.5)
    assert next(reader) == (2, 71)
    with pytest.raises(StopIteration):
        next(reader)
```

虽然这些测试是好的，但是当你更多地思考问题并在你的解决方案中发现 bug 时，添加新的测试也是好的。这里有一些新的测试，涵盖了你在上一个问题结束时想到的一些极限情况:

```py
# test_weather_v1.py
import pytest
import weather_v1 as wthr

 # ...

def test_no_lines():
    no_data = []
    for _ in wthr.get_next_day_and_avg(no_data):
        assert False

def test_trailing_blank_lines(mock_csv_data):
    mock_csv_data.append("")
    all_lines = [x for x in wthr.get_next_day_and_avg(mock_csv_data)]
    assert len(all_lines) == 2
    for line in all_lines:
        assert len(line) == 2

def test_mid_blank_lines(mock_csv_data):
    mock_csv_data.insert(1, "")
    all_lines = [x for x in wthr.get_next_day_and_avg(mock_csv_data)]
    assert len(all_lines) == 2
    for line in all_lines:
        assert len(line) == 2
```

这些测试包括传入空文件的情况，以及 CSV 文件中间或结尾有空行的情况。文件的第一行有坏数据的情况更有挑战性。如果第一行不包含标签，数据是否仍然满足问题的要求？真正的 Python 解决方案假定这是无效的，并且不对其进行测试。

对于这个问题，代码本身不需要做太大的改动。和以前一样，如果你正在你的机器上处理这些解决方案，现在是复制`football_v2.py`到`weather_v1.py`的好时机。

如果您从足球解决方案开始，那么生成器函数被重命名为`get_next_day_and_avg()`，调用它的函数现在是`get_max_avg()`:

```py
# weather_v1.py
import csv

def get_next_day_and_avg(csv_file):
    for day_stats in csv.DictReader(csv_file):
        day_number = int(day_stats["Day"])
        avg = (int(day_stats["MxT"]) + int(day_stats["MnT"])) / 2
        yield day_number, avg

def get_max_avg(filename):
    with open(filename, "r", newline="") as csv_file:
        return max(get_next_day_and_avg(csv_file), key=lambda item: item[1])
```

在这种情况下，你稍微改变一下`get_next_day_and_avg()`。您现在得到的是一个代表天数并计算平均温度的整数，而不是团队名称和分数差。

调用`get_next_day_and_avg()`的函数已经改为使用 [`max()`](https://realpython.com/python-min-and-max/) 而不是`min()`，但仍然保持相同的结构。

针对这段代码运行新的测试显示了使用标准库中的工具的优势:

```py
$ pytest test_weather_v1.py
============================= test session starts ==============================
platform linux -- Python 3.7.1, pytest-6.2.1, py-1.10.0, pluggy-0.13.1
rootdir: /home/jima/coding/realPython/articles/jima-csv
collected 5 items

test_weather_v1.py .....                                                 [100%]

============================== 5 passed in 0.05s ===============================
```

新函数通过了您添加的新空行测试。那个人会帮你处理那些案子。您的测试运行没有错误，您有一个伟大的解决方案！

在面试中，讨论你的解决方案的性能可能是好的。对于这里的框架代码提供的小数据文件，速度和内存使用方面的性能并不重要。但是如果天气数据是上个世纪的每日报告呢？这个解决方案会遇到[内存](https://realpython.com/python-memory-management/)问题吗？有没有办法通过重新设计解决方案来解决这些问题？

到目前为止，这两种解决方案具有相似的结构。在下一节中，您将看到重构这些解决方案，以及如何在它们之间共享代码。

## Python CSV 解析:重构

到目前为止，您看到的两个问题非常相似，解决它们的程序也非常相似。一个有趣的面试问题可能是要求你[重构](https://realpython.com/python-refactoring/)这两个解决方案，找到一种共享代码的方法，使它们更易于维护。

### 问题描述

这个问题和前面两个有点不同。对于本节，从前面的问题中提取解决方案，并对它们进行重构，以重用常见的代码和结构。在现实世界中，这些解决方案足够小，以至于这里的重构工作可能不值得，但它确实是一个很好的思考练习。

### 问题解决方案

这是真正的 Python 团队完成的重构。

**注意:**记住，在你准备好查看这个 Python 练习题的答案之前，不要打开下面折叠的部分！



从查看这两个问题的解决方案代码开始。不算测试，足球解决方案有两个函数长:

```py
# football_v2.py
import csv

def get_next_name_and_diff(csv_file):
    for team_stats in csv.DictReader(csv_file):
        diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"])
        yield team_stats["Team"], abs(diff)

def get_min_score_difference(filename):
    with open(filename, "r", newline="") as csv_data:
        return min(get_next_name_and_diff(csv_data), key=lambda item: item[1])
```

类似地，平均温度解由两个函数组成。相似的结构指出了需要重构的领域:

```py
# weather_v1.py
import csv

def get_next_day_and_avg(csv_file):
    for day_stats in csv.DictReader(csv_file):
        day_number = int(day_stats["Day"])
        avg = (int(day_stats["MxT"]) + int(day_stats["MnT"])) / 2
        yield day_number, avg

def get_max_avg(filename):
    with open(filename, "r", newline="") as csv_file:
        return max(get_next_day_and_avg(csv_file), key=lambda item: item[1])
```

在比较代码时，有时使用`diff`工具来比较每个代码的文本是很有用的。不过，您可能需要从文件中删除额外的代码来获得准确的图片。在这种情况下，[文件字符串](https://realpython.com/documenting-python-code/)被删除。当你`diff`这两个解决方案时，你会发现它们非常相似:

```py
--- football_v2.py   2021-02-09 19:22:05.653628190 -0700 +++ weather_v1.py 2021-02-09 19:22:16.769811115 -0700 @@ -1,9 +1,10 @@ -def get_next_name_and_diff(csv_file): -    for team_stats in csv.DictReader(csv_file): -        diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"]) -        yield team_stats["Team"], abs(diff) +def get_next_day_and_avg(csv_file): +    for day_stats in csv.DictReader(csv_file): +        day_number = int(day_stats["Day"]) +        avg = (int(day_stats["MxT"]) + int(day_stats["MnT"])) / 2 +        yield day_number, avg -def get_min_score_difference(filename): -    with open(filename, "r", newline="") as csv_data: -        return min(get_next_name_and_diff(csv_data), key=lambda item: item[1]) +def get_max_avg(filename): +    with open(filename, "r", newline="") as csv_file: +        return max(get_next_day_and_avg(csv_file), key=lambda item: item[1])
```

除了函数和变量的名称，还有两个主要区别:

1.  足球解得出`Goals For`和`Goals Against`的差值，而天气解得出`MxT`和`MnT`的平均值。
2.  足球解决方案找到结果的`min()`，而天气解决方案使用`max()`。

第二个区别可能不值得讨论，所以让我们从第一个开始。

这两个发生器功能在结构上是相同的。不同的部分通常可以描述为“获取一行数据并从中返回两个值”，这听起来像一个函数定义。

如果你重新编写足球解决方案来实现这个功能，它会让程序变得更长:

```py
# football_v3.py
import csv

def get_name_and_diff(team_stats):
    diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"])
    return team_stats["Team"], abs(diff)

def get_next_name_and_diff(csv_file):
    for team_stats in csv.DictReader(csv_file):
        yield get_name_and_diff(team_stats)
```

虽然这段代码比较长，但它提出了一些有趣的观点，值得在采访中讨论。有时候当你重构时，让代码更易读会导致代码更长。这里的情况可能不是这样，因为很难说将这个函数分离出来会使代码更具可读性。

然而，还有另外一点。有时为了重构代码，您必须降低代码的可读性或简洁性，以使公共部分可见。这绝对是你要去的地方。

最后，这是一个讨论[单一责任原则](https://en.wikipedia.org/wiki/Single-responsibility_principle)的机会。在高层次上，单一责任原则声明您希望代码的每一部分，一个类，一个方法，或者一个函数，只做一件事情或者只有一个责任。在上面的重构中，您将从每行数据中提取值的职责从负责迭代`csv.DictReader()`的函数中抽出。

如果你回头看看你在上面足球问题的解决方案 1 和解决方案 2 之间所做的重构，你会看到最初的重构将`parse_next_line()`和`get_name_and_diff()`合并成了一个函数。在这个重构中，你把它们拉了回来！乍一看，这似乎是矛盾的，因此值得更仔细地研究。

在第一次重构中，合并两个功能很容易被称为违反单一责任原则。在这种情况下，在拥有两个只能一起工作的小函数和将它们合并成一个仍然很小的函数之间有一个可读性权衡。在这种情况下，合并它们似乎使代码更具可读性，尽管这是主观的。

在这种情况下，您出于不同的原因将这两个功能分开。这里的分裂不是最终目标，而是通往你目标的一步。通过将功能一分为二，您能够在两个解决方案之间隔离和共享公共代码。

对于这样一个小例子，这种分割可能是不合理的。然而，正如您将在下面看到的，它允许您有更多的机会共享代码。这种技术将一个功能块从一个函数中提取出来，放入一个独立的函数中，通常被称为[提取方法](https://refactoring.guru/extract-method)技术。一些[ide 和代码编辑器](https://realpython.com/python-ides-code-editors-guide/)提供工具来帮助你完成这个操作。

此时，您还没有获得任何东西，下一步将使代码稍微复杂一些。你将把`get_name_and_diff()`传递给生成器。乍一看，这似乎违反直觉，但它将允许您重用生成器结构:

```py
# football_v4.py
import csv

def get_name_and_diff(team_stats):
    diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"])
    return team_stats["Team"], abs(diff)

def get_next_name_and_diff(csv_file, func):
    for team_stats in csv.DictReader(csv_file):
 yield func(team_stats) 
def get_min_score_difference(filename):
    with open(filename, "r", newline="") as csv_data:
 return min( get_next_name_and_diff(csv_data, get_name_and_diff), key=lambda item: item[1], )
```

这看起来像是一种浪费，但是有时候重构是一个将解决方案分解成小块以隔离不同部分的过程。尝试对天气解决方案进行同样的更改:

```py
# weather_v2.py
import csv

def get_day_and_avg(day_stats):
    day_number = int(day_stats["Day"])
    avg = (int(day_stats["MxT"]) + int(day_stats["MnT"])) / 2
    return day_number, avg

def get_next_day_and_avg(csv_file, func):
    for day_stats in csv.DictReader(csv_file):
        yield func(day_stats)

def get_max_avg(filename):
    with open(filename, "r", newline="") as csv_file:
        return max(
            get_next_day_and_avg(csv_file, get_day_and_avg),
            key=lambda item: item[1],
        )
```

这使得两个解决方案看起来更加相似，更重要的是，突出了两者之间的不同之处。现在，两种解决方案之间的差异主要包含在传入的函数中:

```py
--- football_v4.py   2021-02-20 16:05:53.775322250 -0700 +++ weather_v2.py 2021-02-20 16:06:04.771459061 -0700 @@ -1,19 +1,20 @@ import csv -def get_name_and_diff(team_stats): -    diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"]) -    return team_stats["Team"], abs(diff) +def get_day_and_avg(day_stats): +    day_number = int(day_stats["Day"]) +    avg = (int(day_stats["MxT"]) + int(day_stats["MnT"])) / 2 +    return day_number, avg -def get_next_name_and_diff(csv_file, func): -    for team_stats in csv.DictReader(csv_file): -        yield func(team_stats) +def get_next_day_and_avg(csv_file, func): +    for day_stats in csv.DictReader(csv_file): +        yield func(day_stats) -def get_min_score_difference(filename): -    with open(filename, "r", newline="") as csv_data: -        return min( -            get_next_name_and_diff(csv_data, get_name_and_diff), +def get_max_avg(filename): +    with open(filename, "r", newline="") as csv_file: +        return max( +            get_next_day_and_avg(csv_file, get_day_and_avg), key=lambda item: item[1], )
```

一旦到了这一步，您可以将生成器函数重命名为更通用的名称。您还可以将它移动到自己的模块中，这样您就可以在两个解决方案中重用该代码:

```py
# csv_parser.py
import csv

def get_next_result(csv_file, func):
    for stats in csv.DictReader(csv_file):
        yield func(stats)
```

现在您可以重构每个解决方案来使用这个公共代码。这是足球解决方案的重构版本:

```py
# football_final.py
import csv_reader

def get_name_and_diff(team_stats):
    diff = int(team_stats["Goals For"]) - int(team_stats["Goals Against"])
    return team_stats["Team"], abs(diff)

def get_min_score_difference(filename):
    with open(filename, "r", newline="") as csv_data:
        return min(
            csv_reader.get_next_result(csv_data, get_name_and_diff),
            key=lambda item: item[1],
        )
```

天气解决方案的最终版本虽然相似，但在问题需要的地方有所不同:

```py
# weather_final.py
import csv_parser

def get_name_and_avg(day_stats):
    day_number = int(day_stats["Day"])
    avg = (int(day_stats["MxT"]) + int(day_stats["MnT"])) / 2
    return day_number, avg

def get_max_avg(filename):
    with open(filename, "r", newline="") as csv_file:
        return max(
            csv_parser.get_next_result(csv_file, get_name_and_avg),
            key=lambda item: item[1],
        )
```

您编写的单元测试可以被拆分，这样它们可以分别测试每个模块。

虽然这种特殊的重构导致了更少的代码，但是思考一下——并且在面试的情况下，讨论一下——这是否是一个好主意是有好处的。对于这一组特殊的解决方案，它可能不是。这里共享的代码大约有十行，而这些行只使用了两次。此外，这两个问题总体上相当不相关，这使得组合解决方案有点不太明智。

然而，如果你必须做四十个符合这个模型的操作，那么这种类型的重构可能是有益的。或者，如果你分享的生成器函数很复杂，很难得到正确的结果，那么它也将是一个更大的胜利。

这些都是面试时讨论的好话题。然而，对于像这样的问题集，您可能想讨论处理 CSV 文件时最常用的包:pandas。你现在会看到的。

## Python CSV 解析:熊猫

到目前为止，您在解决方案中使用了标准库中的`csv.DictReader`类，这对于这些相对较小的问题来说效果很好。

对于更大的问题， [pandas](https://realpython.com/pandas-read-write-files/) 包可以以极好的速度提供很好的结果。你的最后一个挑战是用熊猫重写上面的足球程序。

### 问题描述

这是本教程的最后一个问题。对于这个问题，你将使用熊猫重写足球问题的解决方案。pandas 解决方案看起来可能与只使用标准库的解决方案不同。

### 问题解决方案

这里讨论了团队达成的解决方案以及他们是如何达成的。

**注意:**记住，在您准备好查看每个 Python 练习问题的答案之前，不要打开下面折叠的部分！



这个 pandas 解决方案的结构不同于标准库解决方案。不使用生成器，而是使用 pandas 来解析文件并创建一个[数据帧](https://realpython.com/pandas-dataframe/)。

由于这种差异，您的测试看起来相似，但略有不同:

```py
# test_football_pandas.py
import pytest
import football_pandas as fb

@pytest.fixture
def mock_csv_file(tmp_path):
    mock_csv_data = [
        "Team,Games,Wins,Losses,Draws,Goals For,Goals Against",
        "Liverpool FC, 38, 32, 3, 3, 85, 33",
        "Norwich City FC, 38, 5, 27, 6, 26, 75",
    ]
    datafile = tmp_path / "football.csv"
    datafile.write_text("\n".join(mock_csv_data))
    return str(datafile)

def test_read_data(mock_csv_file):
    df = fb.read_data(mock_csv_file)
    rows, cols = df.shape
    assert rows == 2
    # The dataframe df has all seven of the cols in the original dataset plus
    # the goal_difference col added in read_data().
    assert cols == 8

def test_score_difference(mock_csv_file):
    df = fb.read_data(mock_csv_file)
    assert df.team_name[0] == "Liverpool FC"
    assert df.goal_difference[0] == 52
    assert df.team_name[1] == "Norwich City FC"
    assert df.goal_difference[1] == 49

def test_get_min_diff(mock_csv_file):
    df = fb.read_data(mock_csv_file)
    diff = fb.get_min_difference(df)
    assert diff == 49

def test_get_team_name(mock_csv_file):
    df = fb.read_data(mock_csv_file)
    assert fb.get_team(df, 49) == "Norwich City FC"
    assert fb.get_team(df, 52) == "Liverpool FC"

def test_get_min_score(mock_csv_file):
    assert fb.get_min_score_difference(mock_csv_file) == (
        "Norwich City FC",
        49,
    )
```

这些测试包括三个动作:

1.  读取文件并创建数据帧
2.  求最小微分
3.  找到与最小值相对应的队名

这些测试与第一个问题中的测试非常相似，所以与其详细检查测试，不如关注解决方案代码，看看它是如何工作的。您将从一个名为`read_data()`的函数开始创建数据帧:

```py
 1# football_pandas.py
 2import pandas as pd
 3
 4def read_data(csv_file):
 5    return (
 6        pd.read_csv(csv_file)
 7        .rename(
 8            columns={
 9                "Team": "team_name",
10                "Goals For": "goals",
11                "Goals Against": "goals_allowed",
12            }
13        )
14        .assign(goal_difference=lambda df: abs(df.goals - df.goals_allowed))
15    )
```

哇！这是一行函数的一堆代码。像这样将方法调用链接在一起被称为使用[流畅接口](https://en.wikipedia.org/wiki/Fluent_interface)，这在处理 pandas 时相当常见。一个[数据帧](https://pandas.pydata.org/docs/reference/frame.html)上的每个方法返回一个`DataFrame`对象，所以你可以将方法调用链接在一起。

理解这样的代码的关键是，如果它跨越多行，从左到右、从上到下地理解它。

在这种情况下，从第 6 行的 [`pd.read_csv()`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) 开始，它读取 CSV 文件并返回初始的`DataFrame`对象。

第 7 行的下一步是在返回的数据帧上调用 [`.rename()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html#pandas.DataFrame.rename) 。这将把数据帧的列重命名为将作为属性工作的名称。你关心的三个栏目改名为`team_name`、`goals`、`goals_allowed`。一会儿你会看到如何访问它们。

从`.rename()`返回的值是一个新的 DataFrame，在第 14 行，您调用它的 [`.assign()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html) 来添加一个新列。该列将被称为`goal_difference`，并且您提供一个 lambda 函数来为每一行计算它。同样，`.assign()`返回它被调用的`DataFrame`对象，该对象用于该函数的返回值。

**注意:** pandas 为您将在这个解决方案中使用的每个列名提供了属性。这产生了良好的、可读的结果。然而，它确实有一个潜在的陷阱。

如果属性名与 pandas 中的 DataFrame 方法冲突，命名冲突可能会导致意外的行为。如果您有疑问，您可以随时使用 [`.loc[]`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html) 来访问列值。

您的解决方案中的下一个函数展示了一些神奇熊猫可以提供的功能。利用 pandas 将整个列作为一个对象进行寻址并在其上调用方法的能力。在这个实例中，您调用 [`.min()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.min.html) 来查找该列的最小值:

```py
# football_pandas.py
def get_min_difference(parsed_data):
    return parsed_data.goal_difference.min()
```

熊猫提供了几个类似于`.min()`的功能，可以让你[快速有效地操纵行和列](https://realpython.com/pandas-python-explore-dataset/)。

你的解决方案的下一部分是找到与最小分数差相对应的队名。`get_team()`再次使用流畅的编程风格将单个数据帧上的多个调用链接在一起:

```py
# football_pandas.py
def get_team(parsed_data, min_score_difference):
    return (
        parsed_data.query(f"goal_difference == {min_score_difference}")
        .reset_index()
        .loc[0, "team_name"]
    )
```

在这个函数中，您调用 [`.query()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) ，指定您想要的行中的`goal_difference`列等于您之前找到的最小值。从`.query()`返回的值是一个新的 DataFrame，具有相同的列，但只有那些匹配查询的行。

由于 pandas 管理查询索引的一些内部机制，需要下一个调用 [`.reset_index()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html) ，以便于访问这个新数据帧的第一行。一旦索引被重置，您调用`.loc[]`来获取行`0`和`team_name`列，这将从第一行返回匹配最小分数差的球队名称。

最后，您需要一个函数将所有这些放在一起，并返回球队名称和最小差异。和这个问题的其他解决方案一样，这个函数叫做`get_min_score_difference()`:

```py
# football_pandas.py
def get_min_score_difference(csv_file):
    df = read_data(csv_file)
    min_diff = get_min_difference(df)
    team = get_team(df, min_diff)
    return team, min_diff
```

这个函数使用前面的三个函数将团队名称和最小差异放在一起。

这就完成了你的熊猫版足球节目。它看起来不同于其他两种解决方案:

```py
# football_pandas.py
import pandas as pd

def read_data(csv_file):
    return (
        pd.read_csv(csv_file)
        .rename(
            columns={
                "Team": "team_name",
                "Goals For": "goals",
                "Goals Against": "goals_allowed",
            }
        )
        .assign(goal_difference=lambda df: abs(df.goals - df.goals_allowed))
    )

def get_min_difference(parsed_data):
    return parsed_data.goal_difference.min()

def get_team(parsed_data, min_score_difference):
    return (
        parsed_data.query(f"goal_difference == {min_score_difference}")
        .reset_index()
        .loc[0, "team_name"]
    )

def get_min_score_difference(csv_file):
    df = read_data(csv_file)
    min_diff = get_min_difference(df)
    team = get_team(df, min_diff)
    return team, min_diff
```

既然你已经看到了一个基于熊猫的解决方案，思考一下这个解决方案比你看到的其他解决方案更好或更差是一个好主意。这种类型的讨论可以在面试中提出来。

这里的 pandas 解决方案比标准库版本稍长，但是如果目标是这样的话，当然可以缩短。对于像这样的小问题来说，熊猫可能有点小题大做了。然而，对于更大、更复杂的问题，花费额外的时间和复杂性引入 pandas 可以节省大量的编码工作，并且比直接使用 CSV 库更快地提供解决方案。

这里要讨论的另一个角度是，你正在进行的项目是否有或被允许有外部依赖性。在一些项目中，引入熊猫这样的额外项目可能需要大量的政治或技术工作。在这种情况下，标准库解决方案会更好。

[*Remove ads*](/account/join/)

## 结论

这一套 Python CSV 解析练习题到此结束！您已经练习了如何将 Python 技能应用于 CSV 文件，并且还花了一些时间来思考可以在面试中讨论的折衷方案。然后，您查看了重构解决方案，既从单个问题的角度，也从两个解决方案中重构公共代码的角度。

**除了解决这些问题，你还学了:**

*   **用`csv.DictReader()`类编写代码**
*   **利用熊猫**解决 CSV 问题
*   **在面试中讨论**你的解决方案
*   谈论**设计决策**和权衡

现在，您已经准备好面对 Python CSV 解析问题，并在采访中讨论它了！如果您有任何问题或者对其他 Python 实践问题有任何建议，请随时在下面的评论区联系我们。祝你面试好运！

请记住，您可以通过单击下面的链接下载这些问题的框架代码:

**获取源代码:** [单击此处获取源代码，您将在本教程中使用](https://realpython.com/bonus/interview-parsing-csv-code/)来练习解析 CSV 文件。***