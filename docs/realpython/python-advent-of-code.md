# 代码的出现:用 Python 解决你的难题

> 原文：<https://realpython.com/python-advent-of-code/>

《降临代码》是一个在线的降临日历，在那里你可以找到从 12 月 1 日到 25 日每天提供的新的编程难题。虽然你可以在任何时候解谜，但当新的谜题解开时，那种兴奋真的很特别。您可以参与任何编程语言的“代码降临”——包括 Python！

在本教程的帮助下，你将准备好开始解谜并获得你的第一颗金星。

在本教程中，您将学习:

*   什么是在线降临节日历
*   解决谜题如何提高你的编程技能
*   你如何**参与**代码的出现
*   当解决降临代码难题时，你如何组织你的代码和测试
*   如何在解谜时使用测试驱动开发

代码难题的来临被设计为对解决问题感兴趣的任何人接近。不需要厚重的计算机科学背景也能参与。相反,《代码的出现》是学习新技能和测试 Python 新特性的绝佳舞台。

**源代码:** [点击此处下载免费源代码](https://realpython.com/bonus/python-advent-of-code-code/)，向您展示如何用 Python 解决代码难题。

## 编程中的困惑？

玩拼图似乎是浪费你的编程时间。毕竟，看起来你并没有真正产生任何有用的东西，也没有推进你当前的项目。

然而，花些时间练习编程难题有几个好处:

*   编程难题通常比你的常规工作任务更具体、更有内容。它们为你提供了一个机会来练习逻辑思维，解决比你日常工作中通常需要处理的问题更简单的问题。

*   你可以经常用几个类似的谜题来挑战自己。这允许你**建立程序性记忆**，就像肌肉记忆一样，并获得构建某种代码的经验。

*   谜题的设计往往着眼于解决问题。它们允许你**学习和应用经过试验和测试的算法**,并且是任何程序员工具箱的重要组成部分。

*   对于一些难题的解决方案，如果算法效率低下，即使是最伟大的超级计算机也会太慢。您可以**分析您的解决方案的性能**,并获得经验来帮助您理解什么时候简单的方法足够快，什么时候需要更优化的程序。

*   大多数编程语言非常适合解决编程难题。这给了你一个很好的机会**比较不同任务的不同编程语言**。谜题也是了解一门新的编程语言或者尝试你最喜欢的语言的一些[最新特性](https://realpython.com/python311-new-features/)的好方法。

最重要的是，用一个编程难题挑战自己通常是非常有趣的！当你把这些都加起来时，留出一些时间玩拼图会很有收获。

[*Remove ads*](/account/join/)

## 探索在线解决编程难题的选项

幸运的是，有许多网站可以让你找到编程难题并尝试解决它们。这些网站呈现的问题类型、你如何提交你的解决方案以及这些网站能提供什么样的反馈和社区通常都有所不同。因此，你应该花些时间四处看看，找到那些最吸引你的。

在本教程中，你将了解代码的[出现，包括你可以在那里找到什么样的谜题，以及你可以使用哪些工具和技巧来解决它们。但是，您也可以从其他地方开始解决编程难题:](https://adventofcode.com/about)

*   exercisem 拥有多种不同编程语言的学习轨迹。每条学习路线都提供了编码挑战、关于不同编程概念的小教程，以及为您提供解决方案反馈的导师。

*   [欧拉项目](https://projecteuler.net/)由来已久。该网站提供了数百个谜题，通常是数学问题。你可以用任何编程语言解决问题，一旦你解决了一个难题，你就可以进入一个社区线程，在那里你可以和其他人讨论你的解决方案。

*   代码大战提供了大量的编码挑战，他们称之为[卡塔斯](https://en.wikipedia.org/wiki/Kata)。您可以用许多不同的编程语言通过它们内置的编辑器和自动化测试来解决难题。之后，你可以将你的解决方案与其他人的进行比较，并在论坛中讨论策略。

*   如果你在找工作，HackerRank 有很棒的功能。他们提供许多不同技能的认证，包括解决问题和 Python 编程，以及一个工作板，让你在求职申请中展示你的解谜技能。

还有许多其他网站可以让你练习解谜技巧。在本教程的其余部分，您将重点关注代码时代提供了什么。

## 为《代码降临》做准备:圣诞节的 25 个新鲜谜题

代码时代到了！它是由 [Eric Wastl](https://realpython.com/interview-eric-wastl/) 在 2015 年创办的。从那以后，每年 12 月都会出版一个新的**降临节日历**，里面有 25 个新的编程难题。这些年来，谜题变得越来越受欢迎。[超过 235，000 人已经解决了至少一个 2021 年的谜题。](https://adventofcode.com/2021/stats)

**注:**传统上，[降临节日历](https://en.wikipedia.org/wiki/Advent_calendar)是用来计算等待圣诞节时[降临节](https://en.wikipedia.org/wiki/Advent)的日子的日历。多年来，降临节日历变得越来越商业化，已经失去了一些与基督教的联系。

大多数降临节日历开始于 12 月 1 日，结束于 12 月 24 日，平安夜，或 12 月 25 日，圣诞节。现在有各种各样的降临节日历，包括[乐高日历](https://www.lego.com/en-us/search?q=advent)、[茶叶日历](https://perchs.dk/vare/a-c-perchs-julekalender-2022-peach/?lang=en)和[化妆品日历](https://www.elfcosmetics.com/shine-bright-24-day-advent-calendar/70898.html)。

在传统的降临节日历中，你每天打开一扇门来展示里面的东西。《代码降临》模拟了这一点，从 12 月 1 日到 12 月 25 日，每天给你一个新的谜题。对于你解决的每一个难题，你将获得属于你的金色星星。

在这一节中，您将更加熟悉《代码的来临》,并初步了解您的第一个难题。[稍后](#solving-advent-of-code-with-python)，你将看到如何解决这些难题的细节，并练习自己解决一些难题。

### 代码拼图的出现

代码降临节是一个在线降临节日历，从 12 月 1 日到 12 月 25 日每天发布一个新的谜题。每个谜题在美国东部时间[午夜开始发售。代码难题的出现有几个典型特征:](https://www.timeanddate.com/time/zones/et)

*   每个谜题都由两部分组成，但是第二部分直到你完成第一部分才会显示出来。
*   每完成一个部分，你将获得一个金星(⭐)。这意味着，如果你在一年内解决了所有的谜题，你每天可以获得两颗星星和五十颗星星。
*   这个难题对每个人来说都是一样的，但是你需要根据你从 Code site 的出现中获得的个性化输入来解决它。这意味着你对一个谜题的回答会和别人的不一样，即使你用同样的代码来计算。

你可以参加[全球竞赛](https://adventofcode.com/leaderboard)成为第一个解决每个难题的人。然而，这里通常挤满了高技能、有竞争力的程序员。如果你把《代码的来临》作为自己的练习，或者如果你向你的朋友和同事发起一场小型的友好比赛，它可能会让[变得更有趣](https://jeroenheijmans.github.io/advent-of-code-surveys/)。

为了感受一下代码拼图是如何出现的，请考虑一下 2020 年的第一天拼图:

> 在你离开之前，会计的精灵们只需要你搞定你的**费用报告**(你的拼图输入)；显然，有些事情不太对劲。
> 
> 具体来说，他们需要你**找到加起来等于`2020`** 的两个条目，然后将这两个数字相乘。

每年，都有一个非常愚蠢的背景故事将谜题结合在一起。2020 年的故事描述了你在连续几年拯救圣诞节后，试图去度一个当之无愧的假期。这个故事通常对谜题没有影响，但是继续下去还是很有趣的。

在故事的情节元素之间，你会发现谜题本身。在本例中，您在难题输入中寻找两个总计为 2，020 的条目。在描述问题的解释之后，您通常会找到一个示例，显示您需要进行的计算:

> 例如，假设您的费用报告包含以下内容:
> 
> ```py
> `1721
> 979
> 366
> 299
> 675
> 1456` 
> ```
> 
> 在这个列表中，总计为`2020`的两个条目是`1721`和`299`。将它们相乘产生`1721 * 299 = 514579`，所以正确答案是 **`514579`** 。

这个例子显示了这个特定数字列表的答案。如果你准备开始解决这个难题，你现在应该开始考虑如何在任何有效的数字列表中找到这两个条目。然而，在深入这个难题之前，您将探索如何使用代码站点的出现。

[*Remove ads*](/account/join/)

### 如何参与代码降临

您已经看到了一个代码难题出现的例子。接下来，您将了解如何提交您的答案。你从来没有提交任何代码来解决难题。您只需提交答案，答案通常是一个数字或一个文本字符串。

一般来说，你会按照一系列的步骤来解决网站上的一个难题:

1.  **在[网站](https://adventofcode.com/auth/login)登陆**。您可以通过使用来自 GitHub、Google、Twitter 或 Reddit 等其他服务的凭据来实现这一点。

2.  **阅读**谜题文本，特别注意给出的例子。您应该确保了解示例数据的解决方案。

3.  **下载**您对谜题的个性化输入。你需要这个输入来找到你对这个问题的唯一答案。

4.  **编写您的解决方案**。这是有趣的部分，在本教程的剩余部分中，您将得到大量的练习。

5.  **在谜题页面上输入您对谜题的答案**。如果你的答案是正确的，那么你将获得一颗金星，谜题的第二部分开始了。

6.  **对拼图的第二部分重复**步骤 2 和 4。这第二部分与第一部分相似，但是它通常增加了一个转折，要求您修改代码。

7.  **在拼图页面上输入您的第二个答案**,赢取您的第二颗星并完成拼图。

记住，你不需要提交任何代码，只需要你的谜题答案。这意味着任何编程语言都可以解决代码难题。许多人利用《代码的来临》来练习和学习一种新的编程语言。《降临代码》的创作者 Eric Wastl 在 2019 年做了一次[演讲](https://www.youtube.com/watch?v=gibVyxpi-qA)，他谈到了参与者的不同背景和动机，以及其他一些事情。

**注意:**有一个[排行榜](https://adventofcode.com/leaderboard)用于代码的出现。一般来说，你应该**忽略这个排行榜**！它只显示谁在谜题出现后提交了前 100 个答案。要想有机会加入排行榜，你需要大量的准备、奉献和有竞争力的编程经验。

相反，你应该看看**私人排行榜**。这些在你登录后就变成了[可用](https://adventofcode.com/leaderboard/private)，它们给你一个邀请你的朋友和同事到一个更轻松的社区的机会。你可以选择根据解决谜题的**或者简单地根据人们解决的**谜题数量**来给你的私人排行榜打分。**

您还可以将您在私人排行榜中的名字链接到您的 [GitHub](https://github.com/) 帐户，这样您就可以与朋友分享您的解决方案。登录后，您可以通过点击代码网站出现菜单中的*设置*进行设置。

Advent of Code 是完全免费使用的，但是仍然有一些不同的方式可以支持这个项目:

*   **在你的社交媒体上分享关于代码出现的信息**,让大家知道。
*   **通过参加 [r/adventofcode](https://www.reddit.com/r/adventofcode/) 子编辑或其他论坛来帮助他人**。
*   **邀请您的朋友**参加《代码降临》，在[私人排行榜](https://adventofcode.com/leaderboard/private)上分享您的成果。
*   [捐赠](https://adventofcode.com/support)给《代码降临》。如果你这样做了，那么在网站上你的名字旁边会有一个 **AoC++** 徽章。

在接下来的章节中，您将看到一些关于如何准备用 Python 解决代码问题的建议。还有一个[很棒的列表](https://github.com/Bogdanp/awesome-advent-of-code)你可以查看与《代码的来临》相关的许多不同资源的链接，包括其他几个人的解决方案。

## 用 Python 解决代码的出现

代码的出现已经成为世界各地许多编码人员的年度亮点。2021 年，[超过 235，000](https://adventofcode.com/2021/stats) 人提交了他们的解决方案。自 2015 年代码问世以来，程序员已经收集了[超过一千万颗星星](https://twitter.com/ericwastl/status/1474765035071315968)。[许多参与者](https://github.com/search?q=advent+of+code)用 Python 来解谜。

好了，现在轮到你了！前往[降临代码网站](https://adventofcode.com/)，看看最新的谜题。然后，回到本教程来获得一些提示，并帮助开始用 Python 解决降临代码难题。

### 一个谜题的剖析

在这一节中，您将探索代码难题出现的典型剖析。此外，您将了解一些可以用来与之交互的工具。

每次出现的代码难题被分成两部分。当你开始拼图时，你只能看到第一部分。一旦你提交了第一部分的正确答案，第二部分就会解锁。这通常是对您在第一部分中解决的问题的一种扭曲。有时，你会发现有必要[重构](https://realpython.com/python-refactoring/)第一部分的解决方案，而其他时候，你可以基于已经完成的工作快速解决第二部分。

两个部分总是使用相同的难题输入。您可以从当天的谜题页面下载您的谜题输入。你会在谜题描述后找到一个链接。

**注:**如前所述，你的谜题输入是个性化的。这意味着如果你和其他人讨论解决方案，他们的最终答案可能会和你的不同。

为了提交你的谜题解决方案，你需要做的一切——除了实际解决谜题——你都可以从代码的出现网站上做。你应该用它来提交你的第一个解决方案，这样你就可以熟悉流程了。

稍后，您可以使用几个工具来组织代码设置并更有效地工作。例如，您可以使用 [`advent-of-code-data`](https://pypi.org/project/advent-of-code-data/) 包下载数据。是一个可以用 [`pip`](https://realpython.com/what-is-pip/) 安装的 Python 包:

```py
$ python -m pip install advent-of-code-data
```

你可以使用`advent-of-code-data`通过它的`aocd`工具在命令行上下载一个特定的谜题输入集。另一个有趣的可能性是在 Python 代码中自动下载和缓存您的个性化谜题输入:

>>>

```py
>>> from aocd.models import Puzzle
>>> puzzle = Puzzle(year=2020, day=1)

>>> # Personal input data. Your data will be different.
>>> puzzle.input_data[:20]
'1753\n1858\n1860\n1978\n'
```

在使用`advent-of-code-data`下载您的个性化数据之前，您需要[在环境变量或文件中设置您的会话 ID](https://github.com/wimglenn/advent-of-code-wim/issues/1) 。你会在[文档](https://github.com/wimglenn/advent-of-code-data#quickstart)中找到对此的解释。如果你感兴趣，那么你也可以使用`advent-of-code-data`或`aocd`来提交你的解决方案并回顾你之前的回答。

作为谜题文本的一部分，您还会发现一个或几个示例，这些示例通常基于比您的个性化输入数据更小的数据进行计算。您应该仔细阅读这些示例，并确保在开始编码之前您理解了要求您做的事情。

您可以使用示例为您的代码设置[测试](https://realpython.com/python-testing/)。一种方法是对示例数据手动运行您的解决方案，并确认您得到了预期的答案。或者，你可以使用类似 [`pytest`](https://realpython.com/pytest-python-testing/) 的工具来自动化这个过程。

**注意:** [测试驱动开发(TDD)](https://realpython.com/python-hash-table/) 是你在实现代码之前编写测试的过程。因为《代码的出现》为你提供了小例子的预期答案，它给你一个很好的机会去尝试你自己的测试驱动开发。

当你试着自己解决一些难题时，你会学到更多关于 TDD 的知识。

你可以用普通的 Python 和标准库解决所有的代码难题。但是，在您整理解决方案时，有几个软件包可以帮助您:

*   [`advent-of-code-data`](https://pypi.org/project/advent-of-code-data/) 可以下载您输入的数据并提交您的解决方案。
*   [`advent-of-code-ocr`](https://pypi.org/project/advent-of-code-ocr/) 可以将一些谜题的 ASCII 艺术解答转换成字符串。
*   [`pytest`](https://realpython.com/pytest-python-testing/) 可以自动检查你对例题数据的解答。
*   [`parse`](https://realpython.com/python-packages/#parse-for-matching-strings) 可以用比[正则表达式](https://realpython.com/regex-python/)更简单的语法解析字符串。
*   [`numpy`](https://realpython.com/numpy-tutorial/) 可以有效地计算带有数组的数字。
*   [`colorama`](https://pypi.org/project/colorama/) 可以在终端中动画显示您的解决方案。
*   [`rich`](https://rich.readthedocs.io/) 可以让你的终端输出更具视觉吸引力。

如果你创建一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)并安装这些包，那么你将拥有一个非常坚实的工具箱来迎接代码冒险的到来。[稍后](#solution-strategies)，你会看到如何使用`parse`、`numpy`和`colorama`来解谜的例子。

[*Remove ads*](/account/join/)

### 解决方案的结构

在上一节中，您已经熟悉了如何阅读和理解代码难题的降临。在本节中，您将了解如何解决这些问题。在解决代码难题之前，您不需要做大量的设置。

你想过如何解决你之前看到的谜题吗？回想一下，您正在查找列表中两个数字的乘积，其总和为 2，020。在继续之前，想一想——也许可以编码一下——如何找到下面列表中哪两个条目的总数是 2，020:

>>>

```py
>>> numbers = [1721, 979, 366, 299, 675, 1456]
```

以下脚本显示了解决 2020 年第一天[谜题的第一部分的一种方法:](https://adventofcode.com/2020/day/1)

>>>

```py
 1>>> for num1 in numbers:
 2...     for num2 in numbers:
 3...         if num1 < num2 and num1 + num2 == 2020: 4...             print(num1 * num2)
 5...
 6514579
```

嵌套的`for`循环从列表中查找两个数字的所有组合。第 3 行的测试实际上比它需要的要稍微复杂一些:您只需要测试这些数字的总和是 2，020。然而，通过添加条件，即`num1`应该小于`num2`，可以避免两次找到解。

在这个例子中，一个解看起来像`num1 = 1721`和`num2 = 299`，但是因为你可以以任何顺序添加数字，这意味着`num1 = 299`和`num2 = 1721`也形成了一个解。通过额外的检查，只报告后一种组合。

一旦你有了这个解决方案，你就可以将你的个性化输入数据复制到`numbers`列表中，并计算出你的谜题答案。

**注:**有比尝试所有可能性更高效的方法来计算这个答案。然而，从基本方法开始通常是个好主意。[引用乔·阿姆斯特朗](https://en.wikipedia.org/wiki/Joe_Armstrong_(programmer))的话说:

> 让它工作，然后让它漂亮，然后如果你真的，真的有必要，让它快。90%的时候，如果你把它做得漂亮，它已经很快了。所以真的，把它做漂亮就好！([来源](https://henrikwarne.com/2021/04/16/more-good-programming-quotes-part-5/))
> 
> — *乔·阿姆斯特朗*

现在你已经看到了这个难题的解决方案，你能把它变漂亮吗？

随着您处理更多的难题，您可能会开始觉得将数据复制到代码中并将其重写为有效的 Python 变得令人厌倦。类似地，在代码中添加一些函数会给你带来更多的灵活性。例如，您可以使用它们向代码中添加测试。

Python 有许多解析字符串的强大功能。从长远来看，最好让输入数据保持下载时的样子，让 Python 将它们解析成可用的数据结构。事实上，将代码分成两个功能通常是有益的。一个函数将解析字符串输入，另一个函数将解决这个难题。基于这些原则，您可以重写您的代码:

```py
 1# aoc202001.py
 2
 3import pathlib
 4import sys
 5
 6def parse(puzzle_input):
 7    """Parse input."""
 8    return [int(line) for line in puzzle_input.split()]
 9
10def part1(numbers):
11    """Solve part 1."""
12    for num1 in numbers:
13        for num2 in numbers:
14            if num1 < num2 and num1 + num2 == 2020:
15                return num1 * num2
16
17if __name__ == "__main__":
18    for path in sys.argv[1:]:
19        print(f"\n{path}:")
20        puzzle_input = pathlib.Path(path).read_text().strip()
21
22        numbers = parse(puzzle_input)
23        print(part1(numbers))
```

在第 12 到 15 行，你会发现你之前的解决方案。首先，您已经将它包装在一个函数中。这使得以后向您的代码中添加自动测试变得更加容易。您还添加了一个`parse()`函数，可以将多行字符串转换成一系列数字。

在第 20 行，使用 [`pathlib`](https://realpython.com/python-pathlib/) 将文件内容作为文本读取，并去掉末尾的任何空白行。循环通过 [`sys.argv`](https://realpython.com/python-command-line-arguments/#the-sysargv-array) 给你所有在命令行输入的文件名。

这些变化使您在处理解决方案时更加灵活。假设您已经将示例数据存储在名为`example.txt`的文件中，并将您的个性化输入数据存储在名为`input.txt`的文件中。然后，通过在命令行上提供它们的名称，您可以在其中任何一个甚至两个服务器上运行您的解决方案:

```py
$ python aoc202001.py example.txt input.txt
example.txt:
514579

input.txt:
744475
```

`514579`确实是使用示例输入数据时的问题答案。请记住，您的个性化输入数据的解决方案将与上面显示的不同。

现在是时候给代码网站的出现一个旋转了！前往 [2020 年降临代码日历](https://adventofcode.com/2020/)并找到第一天的谜题。如果你还没有，下载你的输入数据并计算你的解谜方案。然后，在网站上输入您的解决方案，点击*提交*。

恭喜你！你刚刚赢得了你的第一颗星！

[*Remove ads*](/account/join/)

### 一个起始模板

正如你在上面看到的，代码谜题的出现遵循一个固定的结构。因此，为自己创建一个模板是有意义的，当您开始编写解决方案时，可以将它作为一个起点。在这样的模板中，你到底想要多少结构是个人喜好的问题。首先，您将探索一个基于您在上一节中看到的原则的模板示例:

```py
 1# aoc_template.py
 2
 3import pathlib
 4import sys
 5
 6def parse(puzzle_input):
 7    """Parse input."""
 8
 9def part1(data):
10    """Solve part 1."""
11
12def part2(data):
13    """Solve part 2."""
14
15def solve(puzzle_input):
16    """Solve the puzzle for the given input."""
17    data = parse(puzzle_input)
18    solution1 = part1(data)
19    solution2 = part2(data)
20
21    return solution1, solution2
22
23if __name__ == "__main__":
24    for path in sys.argv[1:]:
25        print(f"{path}:")
26        puzzle_input = pathlib.Path(path).read_text().strip()
27        solutions = solve(puzzle_input)
28        print("\n".join(str(solution) for solution in solutions))
```

该模板具有单独的功能，用于解析输入以及解决谜题的两个部分。15 到 28 行根本不需要碰。它们负责从输入文件中读取文本，调用`parse()`、`part1()`和`part2()`，然后向控制台报告解决方案。

您可以创建一个类似的模板来测试您的解决方案。

**注意:**正如您之前所了解的，示例数据对于创建测试非常有用，因为它们代表了具有相应解决方案的已知数据。

下面的模板使用`pytest`作为测试运行器。它是为几个不同的测试准备的，测试每一个功能`parse()`、`part1()`和`part2()`:

```py
 1# test_aoc_template.py
 2
 3import pathlib
 4import pytest
 5import aoc_template as aoc
 6
 7PUZZLE_DIR = pathlib.Path(__file__).parent
 8
 9@pytest.fixture
10def example1():
11    puzzle_input = (PUZZLE_DIR / "example1.txt").read_text().strip()
12    return aoc.parse(puzzle_input)
13
14@pytest.fixture
15def example2():
16    puzzle_input = (PUZZLE_DIR / "example2.txt").read_text().strip()
17    return aoc.parse(puzzle_input)
18
19@pytest.mark.skip(reason="Not implemented")
20def test_parse_example1(example1):
21    """Test that input is parsed properly."""
22    assert example1 == ...
23
24@pytest.mark.skip(reason="Not implemented")
25def test_part1_example1(example1):
26    """Test part 1 on example input."""
27    assert aoc.part1(example1) == ...
28
29@pytest.mark.skip(reason="Not implemented")
30def test_part2_example1(example1):
31    """Test part 2 on example input."""
32    assert aoc.part2(example1) == ...
33
34@pytest.mark.skip(reason="Not implemented")
35def test_part2_example2(example2):
36    """Test part 2 on example input."""
37    assert aoc.part2(example2) == ...
```

稍后您将看到如何使用这个模板的示例[。在那之前，有几件事你应该注意:](#part-1-solution-using-templates)

*   如第 1 行所示，您应该用前缀`test_`来命名您的`pytest`文件。
*   类似地，每个测试都在一个以前缀`test_`命名的函数中实现。您可以在第 20、25、30 和 35 行看到这样的例子。
*   您应该更改第 5 行的 import 来导入您的解决方案代码。
*   该模板假设示例数据存储在名为`example1.txt`和`example2.txt`的文件中。
*   当您准备开始测试时，您应该删除第 19、24、29 和 34 行上的跳过标记。
*   根据示例数据和相应的解决方案，您需要填写第 22、27、32 和 37 行上的省略号(`...`)。

例如，如果您要将此模板改编为前一节中第一部分 2020 年 1 月 1 日谜题的重写解决方案，那么您需要创建一个文件`example1.txt`，包含以下内容:

```py
1721
979
366
299
675
1456
```

接下来，您将删除前两个测试的跳过标记，并按如下方式实现它们:

```py
# test_aoc202001.py

def test_parse_example1(example1):
    """Test that input is parsed properly."""
    assert example1 == [1721, 979, 366, 299, 675, 1456]

def test_part1_example1(example1):
    """Test part 1 on example input."""
    assert aoc.part1(example1) == 514579
```

最后，您需要确保您正在导入您的解决方案。如果您使用了文件名`aoc202001.py`，那么您应该将第 5 行改为导入`aoc202001`:

```py
 1# test_aoc202001.py
 2
 3import pathlib
 4import pytest
 5import aoc202001 as aoc 6
 7# ...
```

然后运行`pytest`来检查您的解决方案。如果您正确地实现了您的解决方案，那么您会看到类似这样的内容:

```py
$ pytest
====================== test session starts =====================
collected 4 items

test_aoc202001.py ..ss                                     [100%]
================= 2 passed, 2 skipped in 0.02s =================
```

注意`ss`前面的两个点(`..`)。它们代表两个通过的测试。如果测试失败了，你会看到`F`而不是每个点，以及对错误的详细解释。

像 [Cookiecutter](https://cookiecutter.readthedocs.io/) 和 [Copier](https://copier.readthedocs.io/) 这样的工具使得使用这样的模板更加容易。如果你安装了复印机，那么你可以使用一个[模板](https://github.com/gahjelle/template-aoc-python)，类似于你在这里看到的，通过运行以下命令:

```py
$ copier gh:gahjelle/template-aoc-python advent_of_code
```

这将在您计算机上的`advent_of_code`目录的子目录中为一个特定的谜题设置模板。

[*Remove ads*](/account/join/)

### 解决策略

代码难题的出现是非常多样的。随着时间的推移，你会解决许多不同的问题，并发现许多不同的解决策略。

其中一些策略非常通用，可以应用于任何难题。如果你发现自己被困在了一个难题上，这里有一些你可以尝试摆脱困境的方法:

*   **重读描述**。代码难题的出现通常被很好地指定，但是它们中的一些可能是相当信息密集的。确保你没有遗漏谜题的重要部分。
*   主动使用**示例数据**。确保您理解这些结果是如何实现的，并检查您的代码是否能够重现这些示例。
*   有些谜题可能会有点复杂。**将问题分解成更小的步骤**，分别实现和测试每一步。
*   如果您的代码适用于示例数据，但不适用于您的个性化输入数据，那么您可以基于您能够手动计算的数字来构建**额外的测试用例**,以查看您的代码是否覆盖了所有的极限情况。
*   如果你仍然被困住了，那就联系一些致力于《代码降临》的**论坛**上的**你的朋友**和其他解谜者，询问他们是如何解谜的。

随着你做越来越多的谜题，你会开始认识到一些反复出现的一般类型的谜题。

一些谜题涉及文本和密码。Python 有几个操作文本字符串的强大工具，包括许多[字符串方法](https://realpython.com/python-strings/)。为了读取和解析字符串，了解一下[正则表达式](https://docs.python.org/library/re.html)的基础知识是很有帮助的。但是，您也可以经常使用第三方的 [`parse`](https://realpython.com/python-packages/#parse-for-matching-strings) 库。

例如，假设您有一个字符串`"shiny gold bags contain 2 dark red bags."`，并且想要从中解析[相关信息](https://adventofcode.com/2020/day/7)。您可以使用`parse`及其模式语法:

>>>

```py
>>> import parse
>>> PATTERN = parse.compile( ...     "{outer_color} bags contain {num:d}  {inner_color} bags." ... ) 
>>> match = PATTERN.search("shiny gold bags contain 2 dark red bags.") >>> match.named
{'outer_color': 'shiny gold', 'num': 2, 'inner_color': 'dark red'}
```

在后台，`parse`构建一个正则表达式，但是您使用一个更简单的语法，类似于 [f 字符串](https://realpython.com/python-f-strings/)使用的语法。

在其中一些文本问题中，你被明确要求使用**代码和解析器**，通常构建一个小的定制[汇编语言](https://en.wikipedia.org/wiki/Assembly_language)。解析完代码后，经常需要运行给定的程序。实际上，这意味着你要构建一个小型的[状态机](https://en.wikipedia.org/wiki/Finite-state_machine)，它可以跟踪它的当前状态，包括它的内存内容。

您可以使用[类](https://realpython.com/python3-object-oriented-programming/)将状态和行为放在一起。在 Python 中，[数据类](https://realpython.com/python-data-classes/)对于快速建立状态机非常有用。在以下示例中，您实现了一个可以处理两条不同指令的小型状态机:

```py
 1# aoc_state_machine.py
 2
 3from dataclasses import dataclass
 4
 5@dataclass
 6class StateMachine:
 7    memory: dict[str, int]
 8    program: list[str]
 9
10    def run(self):
11        """Run the program."""
12        current_line = 0
13        while current_line < len(self.program):
14            instruction = self.program[current_line]
15
16            # Set a register to a value
17            if instruction.startswith("set "):
18                register, value = instruction[4], int(instruction[6:])
19                self.memory[register] = value
20
21            # Increase the value in a register by 1
22            elif instruction.startswith("inc "):
23                register = instruction[4]
24                self.memory[register] += 1
25
26            # Move the line pointer
27            current_line += 1
```

两条指令`set`和`inc`在`.run()`内被解析和处理。请注意，第 7 行和第 8 行的[类型提示](https://realpython.com/python-type-checking/)使用了一个[更新的语法](https://realpython.com/python39-new-features/#type-hint-lists-and-dictionaries-directly)，该语法只适用于 [Python 3.9](https://realpython.com/python39-new-features/) 和更高版本。如果你使用的是旧版本的 Python，那么你可以从`typing`导入`Dict`和`List`。

要运行你的状态机，你首先要用一个初始内存初始化它，然后把程序加载到机器中。接下来，你调用`.run()`。程序完成后，您可以检查`.memory`以查看机器的新状态:

>>>

```py
>>> from aoc_state_machine import StateMachine
>>> state_machine = StateMachine(
...     memory={"g": 0}, program=["set g 45", "inc g"]
... )
>>> state_machine.run()
>>> state_machine.memory
{'g': 46}
```

这个程序首先将`g`设置为`45`的值，然后增加它，保持它的最终值`46`。

一些有趣的谜题涉及**网格和迷宫**。如果你的网格有固定的大小，那么你可以使用 [NumPy](https://realpython.com/numpy-tutorial/) 来获得它的有效表示。迷宫通常有助于形象化。您可以使用 [Colorama](https://pypi.org/project/colorama/) 在您的终端中直接绘图:

```py
# aoc_grid.py

import numpy as np
from colorama import Cursor

grid = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 2, 1],
        [1, 1, 1, 1, 1],
    ]
)

num_rows, num_cols = grid.shape
for row in range(num_rows):
    for col in range(num_cols):
        symbol = " *o"[grid[row, col]]
        print(f"{Cursor.POS(col + 1, row + 2)}{symbol}")
```

这个脚本展示了一个使用 NumPy 数组存储网格的例子，然后使用 Colorama 中的`Cursor.POS`将光标定位在终端中以打印出网格。当您运行这个脚本时，您将看到如下输出:

```py
$ python aoc_grid.py
*****
*   *
*** *
*  o*
*****
```

在代码运行时可视化代码可能会很有趣，也会给你一些好的见解。当你调试时，不太明白发生了什么，它也是一个无价的帮助。

到目前为止，在本教程中，您已经获得了一些关于如何使用降临代码谜题的一般提示。在接下来的部分中，你将会得到更明确的答案，并解答早年的三个谜题。

[*Remove ads*](/account/join/)

## 练习降临代码:2019 年第 1 天

你将尝试自己解决的第一个谜题是 2019 年第一天的，名为**火箭方程的暴政**。这是一个典型的第一天难题，因为解决方案并不复杂。这是一个很好的练习，可以让你习惯如何使用 Advent Code，并检查你的环境是否设置正确。

### 第一部分:谜题描述

在 2019 年的故事线中，你正在营救被困在太阳系边缘的圣诞老人。在第一个谜题中，你正在准备发射火箭:

> 精灵们很快把你装进飞船，准备发射。
> 
> 在第一次去/不去投票中，每个 Elf 都去，直到燃料计数器上升。他们还没有确定所需的燃料量。
> 
> 发射给定的**模块**所需的燃料基于其**质量**。具体来说，要找到一个模块所需的燃料，取其质量，除以 3，四舍五入，然后减去 2。

示例数据如下所示:

> *   For the mass of `12`, divide by 3 and round down to get `4`, and subtract 2 to get `2`.
> *   For an object with mass `14`, divide by 3 and round down to get `4`, so the required fuel is also `2`.
> *   For the mass of `1969`, the required fuel is `654`.
> *   For an object with mass `100756`, the required fuel is `33583`.

你需要计算你的宇宙飞船的总燃料需求:

> 燃料计数器-Upper 需要知道总的燃料需求。要找到它，单独计算每个模块的质量所需的燃料(您的难题输入)，然后将所有的燃料值加在一起。
> 
> 你飞船上所有模块的燃料需求总和是多少？

现在是时候尝试自己解决这个难题了！下载您的个性化输入数据并在代码发布时检查您的解决方案可能是最有趣的事情，这样您就可以获得奖励。但是，如果您还没有准备好登录《降临代码》,请根据上面提供的示例数据来解决这个难题。

### 第 1 部分:解决方案

完成拼图并获得星星后，您可以展开折叠块来查看拼图解决方案的讨论:



这个解决方案的讨论比解谜所需的要复杂一些。我们的目标是在第一个解决方案中探索一些额外的细节，以便为下一个谜题做更好的准备。

本节分为两部分:

1.  一个简短的关于整数除法的讨论以及它是如何帮助我们的。
2.  这个难题的简单解决方案。

然后，在下一节中，您将看到另一个解决方案，它使用了您之前看到的解决方案和测试的模板。

要返回到当前的谜题，请再次查看要求您执行的计算:

> [要]找到一个模块所需的燃料，取其质量，除以 3，四舍五入，然后减去 2。

您可以一个接一个地执行这些步骤:

>>>

```py
>>> mass = 14
>>> mass / 3
4.666666666666667

>>> int(mass / 3)
4

>>> int(mass / 3) - 2
2
```

对于正数，可以用`int()`到[向下舍入](https://realpython.com/python-rounding/)。如果你的数字可能是负数，那么你应该用`math.floor()`来代替。

Python 和许多其他编程语言都支持一步完成除法和舍入。这被称为[整数除法](https://realpython.com/python-numbers/#integer-division)，由**整数除法运算符** ( `//`)完成。然后，您可以重写之前的计算:

>>>

```py
>>> mass = 14
>>> mass // 3
4

>>> mass // 3 - 2
2
```

使用`mass // 3`除以三并一步向下舍入。现在，您可以计算每个质量的燃料，并将它们相加，以解决这个难题:

>>>

```py
>>> masses = [12, 14, 1969, 100756]
>>> total_fuel = 0

>>> for mass in masses:
...     total_fuel += mass // 3 - 2
...
>>> total_fuel
34241
```

四个示例模块总共需要`34241`个燃料单元。在谜题描述中，它们分别被列为需要`2`、`2`、`654`和`33583`燃料单元。把这些加起来，你得到`34241`，这证实了你的计算。您可以用您的个性化输入数据替换`masses`列表中的数字，以获得您自己的谜题答案。

如果你熟悉[理解](https://realpython.com/list-comprehension-python/)和[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)，那么你可以使用 [`sum()`](https://realpython.com/python-sum-function/) 来缩短你的代码:

>>>

```py
>>> masses = [12, 14, 1969, 100756]
>>> sum(mass // 3 - 2 for mass in masses)
34241
```

有了`sum()`，你就不需要手动把每个燃料需求加起来。相反，您可以用一行代码解决当前的难题。

你现在已经解决了谜题的第一部分。然而，在进入谜题的第二部分之前，下一部分将展示在解决这个问题时，如何使用之前看到的模板。

### 第 1 部分:使用模板的解决方案

展开下面的折叠块，查看 2019 年第一天代码拼图第一部分的另一个解决方案——这次使用您之前看到的[模板](#a-starting-template)来组织您的代码并简化测试:



如果你要做几个降临代码谜题，那么把你的解决方案组织到文件夹中是个好主意。这允许你将所有与拼图相关的文件放在一起。保持整洁的一个好方法是为代码出现的每一年建立一个文件夹，并且在每年的文件夹中为每一天建立文件夹。

对于这个谜题，你可以这样设置:

```py
advent_of_code/
│
└── 2019/
    └── 01_the_tyranny_of_the_rocket_equation/
        ├── aoc201901.py
        ├── input.txt
        ├── example1.txt
        └── test_aoc201901.py
```

您将您的个性化输入数据存储在`input.txt`中，而`example1.txt`包含来自谜题描述的示例数据:

```py
12
14
1969
100756
```

然后，您可以使用这些数据来设置您的第一个测试。从前面的测试模板开始，填写解析输入和解决第一部分的测试:

```py
 1# test_aoc201901.py
 2
 3import pathlib
 4import pytest
 5import aoc201901 as aoc 6
 7PUZZLE_DIR = pathlib.Path(__file__).parent
 8
 9@pytest.fixture
10def example1():
11    puzzle_input = (PUZZLE_DIR / "example1.txt").read_text().strip()
12    return aoc.parse(puzzle_input)
13
14@pytest.fixture
15def example2():
16    puzzle_input = (PUZZLE_DIR / "example2.txt").read_text().strip()
17    return aoc.parse(puzzle_input)
18
19def test_parse_example1(example1):
20    """Test that input is parsed properly."""
21    assert example1 == [12, 14, 1969, 100756] 22
23def test_part1_example1(example1):
24    """Test part 1 on example input."""
25    assert aoc.part1(example1) == 2 + 2 + 654 + 33583 26
27@pytest.mark.skip(reason="Not implemented")
28def test_part2_example1(example1):
29    """Test part 2 on example input."""
30    assert aoc.part2(example1) == ...
31
32@pytest.mark.skip(reason="Not implemented")
33def test_part2_example2(example2):
34    """Test part 2 on example input."""
35    assert aoc.part2(example2) == ...
```

您希望解析器读取文本文件并将每一行转换成列表中的一个数字。您在第 21 行指定这个值作为`test_parse_example1()`中的期望值。`test_part1_example1()`的期望值是文中提到的四种燃油需求的总和。

最后，根据解决方案模板添加`aoc201901.py`:

```py
 1# aoc201901.py
 2
 3import pathlib
 4import sys
 5
 6def parse(puzzle_input):
 7    """Parse input."""
 8
 9def part1(data):
10    """Solve part 1."""
11
12def part2(data):
13    """Solve part 2."""
14
15def solve(puzzle_input):
16    """Solve the puzzle for the given input."""
17    data = parse(puzzle_input)
18    solution1 = part1(data)
19    solution2 = part2(data)
20
21    return solution1, solution2
22
23if __name__ == "__main__":
24    for path in sys.argv[1:]:
25        print(f"{path}:")
26        puzzle_input = pathlib.Path(path).read_text().strip()
27        solutions = solve(puzzle_input)
28        print("\n".join(str(solution) for solution in solutions))
```

在您开始将您的解决方案添加到模板之前，花一分钟运行`pytest`来确认测试确实失败了。在很多细节之间，你应该得到这样的东西:

```py
$ pytest
test_aoc201901.py FFss                                       [100%]

===================== short test summary info =====================
FAILED test_parse_example1 - assert None == [12, 14, 1969, 100756]
FAILED test_part1_example1 - assert None == (((2 + 2) + 654) + 33583)
================== 2 failed, 2 skipped in 0.09s ===================
```

请注意，正如所料，您有两个测试失败了。这种工作方式被称为[测试驱动开发(TDD)](https://en.wikipedia.org/wiki/Test-driven_development) 。您首先编写您的测试，并确保它们失败。之后，您实现必要的代码来使它们通过。对于这个谜题来说，这似乎有些矫枉过正，但对于更具挑战性的问题来说，这可能是一个非常有用的习惯。

是时候将您的解决方案添加到`aoc201901.py`中了。首先，解析输入数据。它们作为由[换行符](https://realpython.com/python-print/#calling-print) ( `\n`)分隔的数字文本串被传递给`parse()`，并且应该被转换成一个整数列表:

```py
# aoc201901.py

# ...

def parse(puzzle_input):
    """Parse input."""
    return [int(line) for line in puzzle_input.split("\n")]
```

[列表理解](https://realpython.com/python-coding-interview-tips/#use-list-comprehensions-instead-of-map-and-filter)将这些行组装成一个列表，并将它们转换成整数。再次运行`pytest`并确认您的第一个测试`test_parse_example1()`不再失败。

接下来，将您的解决方案添加到拼图中:

```py
# aoc201901.py

# ...

def part1(module_masses):
    """Solve part 1."""
    return sum(mass // 3 - 2 for mass in module_masses)
```

正如上一节所讨论的，您正在通过使用`sum()`来解决第一部分。您还可以将通用参数`data`的名称改为更具体的名称。由于数据描述了每个火箭模块的质量，你称这个参数为`module_masses`。

通过再次运行`pytest`确认您的解决方案是正确的:

```py
$ pytest
test_aoc201901.py ..ss                                       [100%]

================== 2 passed, 2 skipped in 0.01s ===================
```

测试通过后，您可以通过对`input.txt`运行程序来解决个性化输入数据的难题:

```py
$ python aoc201901.py input.txt
input.txt:
3550236
None
```

你自己的答案会和这里显示的不一样，`3550236`。底部的`None`输出表示第二部分的解决方案，您还没有实现它。现在可能是看第二部分的好时机了！

你现在可以进入拼图的第二部分了。你准备好扭转了吗？

### 第二部分:谜题描述

每次出现的代码难题都由两部分组成，其中第二部分只有在您解决第一部分后才会显示。第二部分总是与第一部分相关，并将使用相同的输入数据。然而，你可能经常需要重新思考你的方法来解决前半部分的难题，以便解释后半部分。

展开下面的折叠块，查看 2019 年第一天代码拼图的第二部分:



你让火箭起飞的任务还在继续:

> 在第二次“去/不去”投票中，负责火箭方程式复核的 Elf 停止发射程序。显然，你忘了把额外的燃料包括在你刚刚添加的燃料中。
> 
> 燃料本身就像一个模块一样需要燃料——取其质量，除以 3，四舍五入，然后减去 2。然而，那个燃料**也需要燃料**，而**那个燃料**也需要燃料，以此类推。任何需要**负燃料**的质量应被视为需要**零燃料**；剩余的质量，如果有的话，则由**许愿真的很难**处理，它没有质量，不在这个计算范围内。

当然，给你的飞船添加所有的燃料会使它更重。考虑到增加的重量，你需要添加更多的燃料，但是燃料也是需要考虑的。要了解这在实践中是如何工作的，请看下面的例子:

> 所以，对于每个模块的质量，计算它的燃料并加到总数中。然后，将刚刚计算的燃油量作为输入质量，重复该过程，直到燃油需求为零或负值。例如:
> 
> *   质量为`14`的模块需要`2`燃料。这种燃料不需要更多的燃料(2 除以 3 并向下舍入为`0`，这将要求负燃料)，因此所需的总燃料仍然只是`2`。
> *   首先，一个质量为`1969`的模块需要`654`燃料。然后，这种燃料需要`216`更多的燃料(`654 / 3 - 2`)。`216`然后需要`70`更多的燃料，这需要`21`燃料，这需要`5`燃料，这不需要更多的燃料。因此，一个质量为`1969`的模块所需的总燃料是`654 + 216 + 70 + 21 + 5 = 966`。
> *   一个质量为`100756`的模块所需的燃料及其燃料为:`33583 + 11192 + 3728 + 1240 + 411 + 135 + 43 + 12 + 2 = 50346`。

示例仍然使用与第一部分相同的数字。质量为`12`的模块所需的燃料没有规定，但是你可以通过使用与质量为`14`的模块相同的计算方法来计算出它将是`2`。你需要回答的问题是一样的:

> 考虑到添加燃料的质量，你们飞船上所有模块的燃料需求总量是多少？(分别计算每个模块的燃料需求，然后在最后将它们全部相加。)

试着解决这个问题。你能获得第二颗星吗？

在下一节中，您将看到第二部分的一个可能的解决方案。但是，先试着自己解决这个难题。如果你需要一个开始的提示，然后展开下面的框:



像这部分谜题中的重复计算通常很适合递归。

你做得怎么样？你的火箭准备好发射了吗？

[*Remove ads*](/account/join/)

### 第 2 部分:解决方案

本节将展示如何解决第二部分，继续使用您在上面看到的[模板:](#part-1-solution-using-templates)



继续测试驱动的开发工作流，从向测试文件添加新的例子开始。示例使用了与第一部分相同的数字，因此您可以使用相同的`example1.txt`文件。因此，您可以从您的测试代码中移除`example2()`夹具和`test_part2_example2()`测试。接下来，移除跳过标记并执行`test_part2_example1()`:

```py
# test_aoc201901.py

# ...

def test_part2_example1(example1):
    """Test part 2 on example input."""
    assert aoc.part2(example1) == 2 + 2 + 966 + 50346
```

像以前一样，运行`pytest`来确认您的测试失败了。

**注意:** `pytest`有一个很好的选项`-k`，您可以使用它来运行您的测试的一个子集。使用`-k`，您可以过滤测试名称。例如，为了只运行与第二部分相关的测试，您可以使用`pytest -k part2`。这也是使用一致的和描述性的测试名称的一个很好的激励。

接下来，是实际执行的时候了。因为你被要求重复计算燃料，你可能想要达到[递归](https://realpython.com/python-recursion/)。

一个**递归函数**是一个调用自身的函数。当实现一个递归函数时，你应该注意包含一个停止条件:什么时候函数应该停止调用自己？在这个例子中，停止条件在谜题描述中被非常明确地提到。当燃油变为零或负值时，您应该停止。

随着您的解决方案变得越来越复杂，使用助手函数是个好主意。例如，您可以添加一个函数来计算一个模块所需的所有燃料。助手函数的一个好处是你可以独立于难题解决方案来测试它们。

在您的`aoc201901.py`解决方案文件中添加以下新函数:

```py
# aoc201901.py

# ...

def all_fuel(mass):
    """Calculate fuel while taking mass of the fuel into account.

 ## Example:

 >>> all_fuel(1969)
 966
 """
```

您已经在 docstring 中添加了一个 [doctest](https://realpython.com/python-doctest/) 。您可以通过添加`--doctest-modules`标志来告诉`pytest`运行文档测试:

```py
$ pytest --doctest-modules
aoc201901.py F                                               [ 20%]
test_aoc201901.py ..F                                        [100%]
___________________ [doctest] aoc201901.all_fuel __________________
023 Calculate fuel while taking mass of the fuel into account.
024
025     ## Example:
026
027     >>> all_fuel(1969)
Expected:
 966
Got nothing
```

作为来自`pytest`的输出的一部分，您将看到一个提示，说明`all_fuel()` doctest 失败了。添加 doctests 是确保您的助手函数如您所愿的一个好方法。注意，这个测试不依赖于任何输入文件。相反，您可以直接检查上面给出的一个例子。

接下来，执行燃料计算:

```py
 1# aoc201901.py
 2
 3# ...
 4
 5def all_fuel(mass):
 6    """Calculate fuel while taking mass of the fuel into account.
 7
 8 ## Example:
 9
10 >>> all_fuel(1969)
11 966
12 """
13    fuel = mass // 3 - 2
14    if fuel <= 0:
15        return 0
16    else:
17        return fuel + all_fuel(mass=fuel)
```

第 14 行实现停止条件，而第 17 行执行递归调用。您可以运行测试来检查计算是否按预期工作。

在继续解决整个难题之前，请注意，您可以使用 [walrus 操作符](https://realpython.com/python-walrus-operator/) ( `:=`)来更简洁地编写函数:

```py
# aoc201901.py

# ...

def all_fuel(mass):
    """Calculate fuel while taking mass of the fuel into account.

 ## Example:

 >>> all_fuel(1969)
 966
 """
    return 0 if (fuel := mass // 3 - 2) < 0 else fuel + all_fuel(fuel)
```

虽然代码更短，但也更密集。你是否觉得最终结果更具可读性，这是一个品味和经验的问题。

为了完成这个难题，您还需要实现`part2()`。您的`all_fuel()`函数计算每个模块所需的燃料，所以剩下的就是将所有模块的燃料加在一起:

```py
# aoc201901.py

# ...

def part2(module_masses):
    """Solve part 2."""
    return sum(all_fuel(mass) for mass in module_masses)
```

`part2()`的实现最终与`part1()`非常相似。你只需要改变每个质量的燃料计算。

最后，运行`pytest`来确认一切正常。然后根据您的输入运行您的程序，以获得最终的谜题答案:

```py
$ python aoc201901.py input.txt
input.txt:
3550236
5322455
```

回到 Code 网站问世，输入自己的答案，会和上面的不一样。你的第二颗星星在等着你！

在完全离开这个难题之前，请注意，不使用递归也可以解决第二部分。你可以使用[循环](https://realpython.com/python-while-loop/)来做同样的计算。这里有一个可能的实现:

```py
# aoc201901.py

# ...

def part2(module_masses):
    """Solve part 2."""
    total_fuel = 0
    for mass in module_masses:
        while (mass := mass // 3 - 2) > 0:
            total_fuel += mass

    return total_fuel
```

对于每个质量，`while`循环计算所有需要的燃料，并将其添加到运行的总燃料计数中。

用编程难题挑战自己的一个有趣的事情是，它们给了你一个很好的机会来尝试不同的问题解决方案并进行比较。

恭喜你！你现在已经解决了整个代码难题。你准备好迎接更具挑战性的挑战了吗？

## 练习代码降临:2020 年第 5 天

你将尝试解决的第二个谜题是 2020 年第五天的[谜题，叫做**二进制登机**。这个谜题比前一个更具挑战性，但是最终的解决方案不需要很多代码。首先看看第一部分的拼图描述。](https://adventofcode.com/2020/day/5)

### 第一部分:谜题描述

2020 年，你正努力去你应得的度假胜地。第五天，你正要登机，这时麻烦来了:

> 你登上飞机，却发现了一个新问题:你的登机牌掉了！你不确定哪个座位是你的，所有的空乘人员都忙于应付突然通过护照检查的人群。
> 
> 你写一个快速的程序，用你手机的摄像头扫描附近所有的登机牌(你的字谜输入)；也许你可以通过排除法找到你的位置。
> 
> 这家航空公司使用**二进制空间分割**为乘客提供座位，而不是区域或分组。可以像`FBFBBFFRLR`一样指定座位，其中`F`表示“前面”，`B`表示“后面”，`L`表示“左边”，`R`表示“右边”。
> 
> 前 7 个字符将是`F`或`B`；这些精确地指定了飞机上 128 行**中的一行**(编号为`0`到`127`)。每个字母告诉你给定的座位在哪个半个区域。
> 
> 从整个行列表开始；第一个字母表示座位是在**前面** ( `0`到`63`)还是在**后面** ( `64`到`127`)。下一个字母表示该座位位于该区域的哪一半，以此类推，直到只剩下一行。
> 
> 例如，只考虑`FBFBBFFRLR`的前七个字符:
> 
> *   首先考虑整个范围，从第`0`行到第`127`行。
> *   `F`表示取**下半部**，保留行`0`到`63`。
> *   `B`表示取**上半部**，保留行`32`到`63`。
> *   `F`表示取**下半部**，保留行`32`到`47`。
> *   `B`表示取**上半部**，保留行`40`到`47`。
> *   `B`保持从`44`到`47`的行。
> *   `F`保持从`44`到`45`的行。
> *   最后的`F`保持两者中较低的，**排`44`** 。
> 
> 最后三个字符将是`L`或`R`；这些精确地指定了飞机上 8 列座位中的一列(编号为`0`到`7`)。再次进行与上述相同的过程，这次只有三个步骤。`L`表示保留**下半部**，而`R`表示保留**上半部**。
> 
> 例如，只考虑`FBFBBFFRLR`的最后 3 个字符:
> 
> *   首先考虑整个范围，从列`0`到`7`。
> *   `R`表示取**上半部**，保留列`4`至`7`。
> *   `L`表示取**下半部**，保留`4`至`5`列。
> *   最后的`R`保持两者的上位，**列`5`** 。
> 
> 于是，解码`FBFBBFFRLR`发现是在**行`44`，列`5`** 的座位。
> 
> 每个座位都有一个唯一的**座位 ID** :将行乘以 8，然后添加列。在本例中，座位的 ID 为`44 * 8 + 5 =` **`357`** 。
> 
> 以下是其他一些登机牌:
> 
> *   `BFFFBBFRRR`:行`70`，列`7`，座位号`567`。
> *   `FFFBBBFRRR`:行`14`，列`7`，座位号`119`。
> *   `BBFFBBFRLL`:行`102`，列`4`，座位号`820`。
> 
> 作为一个理智的检查，看看你的登机牌清单。登机牌上最高的座位号是多少？

这个谜题描述里有很多信息！然而，它最关心的是[二进制空间划分](https://en.wikipedia.org/wiki/Binary_space_partitioning)如何为这家特定的航空公司工作。

现在，试着自己解决这个难题吧！请记住，如果从正确的角度考虑，从登机牌规格到座位 ID 的转换并不像一开始看起来那么复杂。如果你发现你正在努力完成这一部分，那么请展开下面的方框，查看如何开始的提示:



登机牌规格是基于[二进制](https://realpython.com/python-bitwise-operators/#binary-system-in-five-minutes)，只是伪装了不同的字符。你能把登机牌翻译成二进制数字吗？

当你完成了你的解决方案，看看下一部分，看看关于这个难题的讨论。

### 第 1 部分:解决方案

既然您已经亲自尝试过了，那么您可以继续并展开下面的模块，看看您可以解决这个难题的一种方法:



您可以根据文本中的描述实现座位 id 的计算。以下函数采取与示例相同的步骤:

```py
# aoc202005.py

# ...

def decode(string):
    """Decode a boarding pass string into a number."""
    start, end = 0, 2 ** len(string)
    for char in string:
        if char in {"F", "L"}:
            end -= (end - start) // 2
        elif char in {"B", "R"}:
            start += (end - start) // 2

    return start
```

您可以通过`start`和`end`限制可能的行或列的范围。`start`在范围内，`end`不在。这使得数学更容易，因为它在整个计算过程中保持了`end - start`的差可以被 2 整除。降低每个`F`或`L`的上限，增加每个`B`或`R`的下限`start`。您可以检查该函数是否给出与示例相同的结果:

>>>

```py
>>> decode("FBFBBFF")
44

>>> decode("RLR")
5

>>> decode("FBFBBFFRLR")
357
```

使用`decode()`，您可以计算登机牌的行、列和座位 ID。然而，Python 已经有内置工具来为您执行相同的计算。

这个谜题的名字，二进制寄宿，以及提到二进制空间分割，意在让你开始思考(或阅读)T2 双星系统。二进制是由`0`和`1`两位数字组成的数字系统，而不是传统的十位数。

拼图中，登机牌规格真的是二进制数。不同的是，他们用`F`或`L`代替`0`，用`B`和`R`代替`1`。比如`FBFBBFFRLR`可以翻译成二进制数 0101100101 <sub>2</sub> 。您可以使用 Python 将其转换为常规的十进制数:

>>>

```py
>>> int("0101100101", base=2)
357
```

你认识那个答案吗？`357`确实是`FBFBBFFRLR`的座位 ID。换句话说，为了计算座位 id，你需要将`F`、`L`、`B`、`R`翻译成它们各自的二进制数字。有几种方法可以做到这一点，但是 Python 的标准库中的 [`str.translate()`](https://docs.python.org/3/library/stdtypes.html#str.translate) 可能是最方便的。它是这样工作的:

>>>

```py
>>> mapping = str.maketrans({"F": "0", "L": "0", "B": "1", "R": "1"})
>>> "FBFBBFFRLR".translate(mapping)
'0101100101'
```

`.translate()`方法使用类似`70`的字符代码，而不是类似`"F"`的字符串。不过，您可以使用方便的功能 [`str.maketrans()`](https://docs.python.org/3/library/stdtypes.html#str.maketrans) 来设置基于字符串的翻译。现在，您可以使用这些工具通过三个步骤来解决这个难题:

1.  将登机牌规格转换为二进制数。
2.  计算二进制数的十进制值以获得座位 id。
3.  找到最大的座位号。

设置新拼图的模板，其中`input.txt`包含您的个性化拼图输入:

```py
advent_of_code/
│
└── 2020/
    └── 05_binary_boarding/
        ├── aoc202005.py
        ├── input.txt
        ├── example1.txt
        └── test_aoc202005.py
```

您可以像往常一样将工作示例添加到`example1.txt`中:

```py
FBFBBFFRLR
BFFFBBFRRR
FFFBBBFRRR
BBFFBBFRLL
```

接下来，你要准备第一部分的测试。在这样做之前，您应该考虑如何解析难题输入。

一种选择是将输入文件解析成字符串列表。但是，您也可以将从登机牌规格到座位 ID 的转换视为解析过程的一部分。需要考虑的一个因素是，您是否认为稍后需要原始的登机牌字符串，也就是在第二部分。

您决定抓住这个机会，并立即解析座位 id。如果在第二部分中需要登机牌字符串，那么您可以随时返回并重构代码。将以下测试添加到测试文件中:

```py
# test_aoc202005.py

# ...

def test_parse_example1(example1):
    """Test that input is parsed properly."""
    assert example1 == [357, 567, 119, 820]

def test_part1_example1(example1):
    """Test part 1 on example input."""
    assert aoc.part1(example1) == 820
```

像往常一样，运行`pytest`来确认您的测试失败了。那么是时候开始实施你的解决方案了。从解析开始:

```py
# aoc202005.py

# ...

BP2BINARY = str.maketrans({"F": "0", "B": "1", "L": "0", "R": "1"})

def parse(puzzle_input):
    """Parse input."""
    return [
        int(bp.translate(BP2BINARY), base=2)
        for bp in puzzle_input.split("\n")
    ]
```

您设置了登机牌字符串和二进制数字之间的转换表。然后使用`.translate()`将输入的每个登机牌转换成二进制数字，使用`int()`将二进制数字转换成座位 ID。

查找最高座位 ID 现在很简单:

```py
# aoc202005.py

# ...

def part1(seat_ids):
    """Solve part 1."""
    return max(seat_ids)
```

Python 内置的 [`max()`](https://realpython.com/python-min-and-max/) 找到一个列表中的最高值。现在，您可以运行您的测试来确认您的解决方案是否有效，然后根据您的个性化输入运行您的代码来得到您对这个难题的答案。

是时候进入拼图的第二部分了。你能登机吗？

### 第二部分:谜题描述

当你准备好拼图的第二部分时，展开下面的部分:



与第一部分相比，第二部分的描述非常简短:

> **丁！**“系好安全带”的指示灯已经亮起。该去找你的座位了。
> 
> 这是一个完全满员的航班，所以你的座位应该是你的名单中唯一缺少的登机牌。然而，有一个问题:飞机最前面和最后面的一些座位在这架飞机上不存在，所以它们也会从你的列表中消失。
> 
> 不过，你的座位不在最前面或最后面；您的 id 为+1 和-1 的座位将出现在您的列表中。
> 
> 你的座位号是多少？

你能找到你的座位吗？

慢慢来，努力解决第二部分的问题。

[*Remove ads*](/account/join/)

### 第 2 部分:解决方案

当您准备好将您的解决方案与另一个进行比较时，请打开下面的盒子:



在谜题的第二部分，你要在数字列表中寻找一个缺失的数字。

有几种方法可以解决这个问题。例如，您可以对所有数字进行排序，并比较排序列表中的连续项目。另一种选择是使用 Python 强大的[集合](https://realpython.com/python-sets/)。您可以首先创建完整的有效座位 id。然后，您可以计算这个完整集合与您列表上的座位 id 集合之间的集合差。

但是，在开始实现之前，您应该为它添加一个测试。在这种情况下，示例数据实际上不适合用于测试。他们有许多座位 id 丢失，而不是像字谜文本指定的那样只有一个。您最好手动创建一个小测试。有一种方法可以做到:

```py
# test_aoc202005.py

# ...

def test_part2():
    """Test part 2 on example input."""
    seat_ids = [3, 9, 4, 8, 5, 10, 7, 11]
    assert aoc.part2(seat_ids) == 6
```

列表`[3, 9, 4, 8, 5, 10, 7, 11]`包含从 3 到 11 的所有座位 id，6 除外。这个小例子满足了这个难题的条件。因此，您的解决方案应该能够找出丢失的座位 ID。

在这个实现中，您将使用`set()`方法:

```py
 1# aoc202005.py
 2
 3# ...
 4
 5def part2(seat_ids):
 6    """Solve part 2."""
 7    all_ids = set(range(min(seat_ids), max(seat_ids) + 1))
 8    return (all_ids - set(seat_ids)).pop()
```

在第 7 行，您创建了所有有效的座位 id。这些是数据集中最小座位 ID 和最大座位 ID 之间的数字，包括这两个数字。为了找到您的座位 ID，您将您的座位 ID 列表转换为一个集合，将其与所有 ID 的集合进行比较，并弹出剩余的一个座位 ID。

太好了，你又完成了一个谜题！为了使事情圆满，请尝试 2021 年的一个谜题。

## 练习代码降临:2021 年第 5 天

作为第三个密码难题出现的例子，你将仔细观察 2021 年第五天。这个谜题叫做**热液冒险**，将带你进行一次深海探险。解决方案会比前两个谜题更复杂一些。看看拼图的描述。

### 第一部分:谜题描述

2021 年的故事线始于精灵们不小心将圣诞老人雪橇的钥匙掉进了海里。为了拯救圣诞节，你最终在一艘潜水艇里搜寻他们。第五天，你会遇到海底的一片热液喷口。

事实证明，这些通风口对你的潜艇有害，你需要绘制出该区域的地图，以避开最危险的区域:

> 它们倾向于形成**线**；潜水艇会很有帮助地列出附近的喷口线(你的拼图输入)供你查看。例如:
> 
> ```py
> `0,9 -> 5,9
> 8,0 -> 0,8
> 9,4 -> 3,4
> 2,2 -> 2,1
> 7,0 -> 7,4
> 6,4 -> 2,0
> 0,9 -> 2,9
> 3,4 -> 1,4
> 0,0 -> 8,8
> 5,5 -> 8,2` 
> ```
> 
> 每一排通风口以格式`x1,y1 -> x2,y2`给出一个线段，其中`x1`、`y1`是线段一端的坐标，`x2`、`y2`是另一端的坐标。这些线段包括两端的点。换句话说:
> 
> *   类似于`1,1 -> 1,3`的条目覆盖了点`1,1`、`1,2`和`1,3`。
> *   类似于`9,7 -> 7,7`的条目覆盖了点`9,7`、`8,7`和`7,7`。
> 
> 目前，**只考虑水平线和垂直线**:或者`x1 = x2`或者`y1 = y2`的线。

该示例显示了难题输入如何描述给定坐标处的线。你的工作是找到这些线重叠的地方:

> 为了避开最危险的区域，你需要确定**至少两条线重叠的点的数量**。在上例中，这是[……]共 **`5`** 分。
> 
> 只考虑水平线和垂直线。至少有两条线在多少点上重叠？

和上一个谜题一样，谜题文本中有很多信息。这些信息主要是关于你应该如何解释你的字谜输入。

**注:**在[全拼图描述](https://adventofcode.com/2021/day/5)中还有一些附加信息。特别是，有一个图表显示了网格上绘制的所有线条。

试着自己解决这个难题。完成后，继续下一节，看看一个可能的解决方案。

### 第 1 部分:输入解析

有许多方法可以解决这个难题。展开下面的块，开始处理输入数据:



你的任务是计算两条线或多条线覆盖了多少个点。最直接的方法可能如下:

1.  将有问题的每条线转换成组成该线的点集。
2.  计算每个点在所有线条中出现的次数。
3.  计算出现两次或更多次的点数。

在开始编码之前，你应该考虑如何表示点和线。这可能是使用专用的`Point`和`Line`类的一个很好的用例，在[数据类](https://realpython.com/python-data-classes/)的帮助下实现。

然而，在这个解决方案中，您将选择一个基本的表示，对每个点使用一个 2 元组整数，对每条线使用一个 4 元组整数。例如，`(0, 9)`代表点`0,9`，`(0, 9, 5, 9)`代表线`0,9 -> 5,9`。

如果可以简化计算，从简单的数据结构开始，并准备好转向更复杂的解决方案通常是好的。您的第一个任务是解析输入数据。设置好模板后，您应该添加一些示例数据。

您可以使用给定的示例数据，但是从创建一个更简单的数据集开始可能会更容易。将以下内容添加到`example1.txt`:

```py
2,0 -> 0,2
0,2 -> 2,2
0,0 -> 0,2
0,0 -> 2,2
```

这些数据代表四条线:两条对角线、一条水平线和一条垂直线。为了完整起见，您也可以将谜题描述中给出的示例数据添加到`example2.txt`中。接下来，您将手工拼写出您想要如何在您的测试文件中表示这四行:

```py
# test_aoc202105.py

# ...

def test_parse_example1(example1):
    """Test that input is parsed properly."""
    assert example1 == [
        (2, 0, 0, 2),
        (0, 2, 2, 2),
        (0, 0, 0, 2),
        (0, 0, 2, 2),
    ]
```

像往常一样，您应该运行`pytest`来确认您的测试失败。有几种方法可以解析输入，因为您希望从每行中提取四个数字。例如，您可以使用一个[正则表达式](https://realpython.com/regex-python/)。在这里，您将重复使用 string `.split()`方法:

```py
 1# aoc202105.py
 2
 3# ...
 4
 5def parse(puzzle_input):
 6    """Parse input."""
 7    return [
 8        tuple(
 9            int(xy)
10            for points in line.split(" -> ")
11            for xy in points.split(",")
12        )
13        for line in puzzle_input.split("\n")
14    ]
```

这当然是一个拗口的问题。为了理解解析是如何工作的，从第 13 行开始。这就建立了一个主循环，它通过在新行上拆分谜题输入来查看每一行。

接下来，将第 8 到 12 行的元组理解应用到每一行。它首先拆分箭头符号(`->`)上的每一行，然后拆分逗号(`,`)上的每一对结果数字。最后，用`int()`将每个数字从字符串转换成整数。

运行您的测试来确认`parse()`如预期的那样解析您的输入。

即使您的代码可以工作，您也可能希望避免大量嵌套的理解。例如，您可以将其重写如下:

```py
# aoc202105.py

# ...

def parse(puzzle_input):
    """Parse input."""
    lines = []
    for line in puzzle_input.split("\n"):
        point1, point2 = line.split(" -> ")
        x1, y1 = point1.split(",")
        x2, y2 = point2.split(",")
        lines.append((int(x1), int(y1), int(x2), int(y2)))
    return lines
```

在这个版本中，您将显式地构建行列表。对于每一行，首先将字符串分成两个点，然后将每个点分成单独的 x 和 y 坐标。

一旦你用一个你能处理的结构表示了数据，那么你就可以继续解决这个难题了。

### 第 1 部分:解决方案

你将继续拼图的第一部分。下面的解决方案利用了 [Python 3.10](https://realpython.com/python310-new-features/) 中引入的[结构模式匹配](https://realpython.com/python310-new-features/#structural-pattern-matching)特性。展开折叠部分以阅读详细信息:



这个难题的主要挑战是将每条线从它当前的表示转换成一个单独点的列表。接下来你会解决这个问题。首先添加一个函数的签名，该函数可以将一条线转换成一系列点，包括一个记录预期输出的 doctest:

```py
# aoc202105.py

# ...

def points(line):
    """List all points making up a line.

 ## Examples:

 >>> points((0, 3, 3, 3))  # Horizontal line
 [(0, 3), (1, 3), (2, 3), (3, 3)]
 >>> points((3, 4, 3, 0))  # Vertical line
 [(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)]
 """
```

您希望该函数返回一个点列表，您可以在以后对其进行计数。现在，你需要考虑水平线和垂直线。您已经为这两种情况添加了测试。

谜题描述提示你如何识别水平线和垂直线，因为其中一个坐标是恒定的。你可以用一个`if`测试来找到这些。但是，您也可以利用这个机会练习使用 Python 3.10 中引入的`match` … `case`语句:

```py
 1# aoc202105.py
 2
 3# ...
 4
 5def points(line):
 6    """List all points making up a line.
 7
 8 ## Examples:
 9
10 >>> points((0, 3, 3, 3))  # Horizontal line
11 [(0, 3), (1, 3), (2, 3), (3, 3)]
12 >>> points((3, 4, 3, 0))  # Vertical line
13 [(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)]
14 """
15    match line:
16        case (x1, y1, x2, y2) if x1 == x2:
17            return [(x1, y) for y in range(y1, y2 + 1)]
18        case (x1, y1, x2, y2) if y1 == y2:
19            return [(x, y1) for x in range(x1, x2 + 1)]
```

这个`match` … `case`结构非常有表现力，但是如果你以前没有使用过它，可能会觉得有点神奇。

每个`case`都试图匹配`line`的结构。所以在第 16 行，你要寻找一个 4 元组。此外，您将 4 元组的值分别解包到变量`x1`、`y1`、`x2`和`y2`中。最后，通过要求`x1`和`x2`必须相等来保证匹配。实际上，这代表一条垂直线。

类似地，第 18 行的`case`语句挑选出水平线。对于每一行，使用 [`range()`](https://realpython.com/python-range/) 列出每一个点，注意要包括端点。

现在，做你的测试。如果您包括文档测试，那么您会注意到有些地方不太对劲:

```py
$ pytest --doctest-modules
___________________ [doctest] aoc202105.points ___________________
List all points making up a line

 ## Examples:

 >>> points((0, 3, 3, 3))  # Horizontal line
 [(0, 3), (1, 3), (2, 3), (3, 3)]
 >>> points((3, 4, 3, 0))  # Vertical line
Expected:
 [(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)]
Got:
 []
```

竖线示例返回一个空列表。随着您的研究，您意识到这个例子调用了`range(4, 1)`，这是一个空的范围，因为`1`小于`4`，并且您正在使用默认的步骤`1`。为了解决这个问题，你可以引入一个更复杂的`range()`表达式。

为了避免在`points()`中放入更多的逻辑，您决定创建一个新的助手函数来处理必要的`range()`逻辑:

```py
# aoc202105.py

# ...

def coords(start, stop):
    """List coordinates between start and stop, inclusive."""
    step = 1 if start <= stop else -1
    return range(start, stop + step, step)
```

如果`start`大于`stop`，那么你要确保使用一个`-1`的步长。您现在可以更新`points()`以使用新功能:

```py
# aoc202105.py

# ...

def points(line):
    """List all points making up a line.

 ## Examples:

 >>> points((0, 3, 3, 3))  # Horizontal line
 [(0, 3), (1, 3), (2, 3), (3, 3)]
 >>> points((3, 4, 3, 0))  # Vertical line
 [(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)]
 """
    match line:
        case (x1, y1, x2, y2) if x1 == x2:
 return [(x1, y) for y in coords(y1, y2)]        case (x1, y1, x2, y2) if y1 == y2:
 return [(x, y1) for x in coords(x1, x2)]
```

通过用`coords()`替换`range()`，你应该能够处理所有的水平线和垂直线。运行您的测试以确认您的代码现在工作正常。

现在，您可以将线转换为单独的点。计划的下一步是计算每个点是多少条线的一部分。您可以遍历所有点并显式计数，但是 Python 的标准库中有许多强大的工具。在这种情况下，您可以使用`collections`模块中的 [`Counter`](https://realpython.com/python-counter/) :

```py
 1# aoc202105.py
 2
 3import collections
 4
 5# ...
 6
 7def count_overlaps(lines):
 8    """Count overlapping points between a list of lines.
 9
10 ## Example:
11
12 >>> count_overlaps(
13 ...     [(3, 3, 3, 5), (3, 3, 6, 3), (6, 6, 6, 3), (4, 5, 6, 5)]
14 ... )
15 3
16 """
17    overlaps = collections.Counter(
18        point for line in lines for point in points(line)
19    )
20    return sum(num_points >= 2 for num_points in overlaps.values())
```

在第 16 行，你循环每一行中的每一点，并将所有点传递给`Counter`。产生的计数器本质上是一个字典，其值指示每个键出现的次数。

要找到两条线或多条线重叠的点的数量，您可以查看您的计数器中有多少个点被看到两次或更多次。

您几乎已经完成了第 1 部分。你只需要用`count_overlaps()`连接谜题输入，并确保你按照要求去做——只“考虑水平线和垂直线”

您可以通过使用更多的理解来过滤所有的行:

```py
# aoc202105.py

# ...

def part1(lines):
    """Solve part 1."""
    vertical = [(x1, y1, x2, y2) for x1, y1, x2, y2 in lines if x1 == x2]
    horizontal = [(x1, y1, x2, y2) for x1, y1, x2, y2 in lines if y1 == y2]
    return count_overlaps(vertical + horizontal)
```

你只通过那些坐标不变的线。在您的个人输入上运行您的代码以计算您的解决方案，并提交它以获得您的下一颗星。

唷！你已经完成了拼图的第一部分。是时候看看第 2 部分为您准备了什么。

[*Remove ads*](/account/join/)

### 第二部分:谜题描述

展开以下部分，阅读 2021 年第 5 天谜题的第二部分:



你可能已经怀疑你不能永远忽略那些对角线:

> 不幸的是，只考虑水平线和垂直线并不能给你全貌；你还需要考虑**对角线**。
> 
> 由于热液喷口绘图系统的限制，你的列表中的线只能是水平的、垂直的或正好 45 度的对角线。换句话说:
> 
> *   类似于`1,1 -> 3,3`的条目覆盖了点`1,1`、`2,2`和`3,3`。
> *   类似于`9,7 -> 7,9`的条目覆盖了点`9,7`、`8,8`和`7,9`。
> 
> 你仍然需要确定**至少两条线重叠的点的数量**。在上例中，这是[……]现在共有的 **`12`** 点。
> 
> 考虑所有的线。至少有两条线在多少点上重叠？

您可能可以重用第 1 部分中所做的大量工作。但是，你怎么能把那些对角线考虑进去呢？

摆弄第二部分，试着自己解决。一旦你完成了，看看下一部分可能的解决方案。

### 第 2 部分:解决方案

当您准备好查看解决方案并与您自己的解决方案进行比较时，请点击以显示以下解决方案:



第二部分的变化是你需要把对角线考虑进去。这个问题仍然要求你计算两条线或多条线的点数。换句话说，您仍然可以在第二部分中使用`count_overlaps()`，但是您需要扩展`points()`以便它可以处理对角线。

幸运的是，所有对角线都正好是 45 度。这样做的实际结果是，这些线中的点的坐标仍然具有连续的整数坐标。

例如，`5,5 -> 8,2`涵盖了`5,5`、`6,4`、`7,3`、`8,2`等穴位。注意，x 坐标是`5`、`6`、`7`和`8`，而 y 坐标是`5`、`4`、`3`和`2`。您可以在网格上手动绘制直线，如下所示:

```py
 123456789 x
1 .........
2 .......#.
3 ......#..
4 .....#...
5 ....#....
6 .........
y
```

上面的数字代表 x 坐标，而左边的数字代表 y 坐标。点(`.`)代表网格，线的点用散列符号(`#`)标注。

为了完成您的解决方案，您需要对当前代码进行两处调整:

1.  更改`points()`使其也能转换对角线。
2.  使用完整的行列表调用`count_overlaps()`。

先从适应`points()`开始。第一个很好的改变是更新 doctest 以包含一个对角线的例子:

```py
# aoc202105.py

# ...

def points(line):
    """List all points making up a line.

 ## Examples:

 >>> points((0, 3, 3, 3))  # Horizontal line
 [(0, 3), (1, 3), (2, 3), (3, 3)]
 >>> points((3, 4, 3, 0))  # Vertical line
 [(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)]
 >>> points((1, 2, 3, 4))  # Diagonal line [(1, 2), (2, 3), (3, 4)] """
    # ...
```

你已经添加了线`1,2 -> 3,4`，它覆盖了点`1,2`、`2,3`和`3,4`。运行您的测试来确认对角线还没有被处理。

您需要在您的`match` … `case`陈述中添加一个新案例。`case`报表从上到下一次检查一个。如果您在现有语句下面添加新代码，那么您将知道您正在处理对角线。因此，你不需要警卫:

```py
# aoc202105.py

# ...

def points(line):
    """List all points making up a line.

 ## Examples:

 >>> points((0, 3, 3, 3))  # Horizontal line
 [(0, 3), (1, 3), (2, 3), (3, 3)]
 >>> points((3, 4, 3, 0))  # Vertical line
 [(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)]
 >>> points((1, 2, 3, 4))  # Diagonal line
 [(1, 2), (2, 3), (3, 4)]
 """
    match line:
        case (x1, y1, x2, y2) if x1 == x2:
            return [(x1, y) for y in coords(y1, y2)]
        case (x1, y1, x2, y2) if y1 == y2:
            return [(x, y1) for x in coords(x1, x2)]
 case (x1, y1, x2, y2): return [(x, y) for x, y in zip(coords(x1, x2), coords(y1, y2))]
```

第三个`case`语句通过同时改变 x 和 y 坐标来处理对角线。在这里，你也从创建`coords()`中获得了一些回报，因为直接使用`range()`绘制对角线要比水平和垂直线条复杂得多。

现在，您可以将对角线转换为单独的点，剩下的唯一任务是计算重叠的数量。由于`count_overlaps()`委托给了`points()`，它现在也可以处理对角线了。您可以用一行代码实现第二部分的解决方案:

```py
# aoc202105.py

# ...

def part2(lines):
    """Solve part 2."""
    return count_overlaps(lines)
```

您应该运行您的测试，以确保一切按预期运行。然后计算你对第二部分的答案，并在《代码的来临》网站上提交。

恭喜你！到目前为止，你已经解决了至少三个降临密码难题。幸运的是，还有上百个[等着你](https://adventofcode.com/events)！

## 结论

代码的来临是有趣的编程难题的一个伟大的资源！你可以用它来练习你的解决问题的能力，挑战你的朋友来一场有趣的比赛和共同的学习经历。在下一集的真实 Python 播客中，你可以听到更多关于代码降临的内容:[用 Python 解决代码降临难题](https://realpython.com/podcasts/rpp/89/)。

如果你还没有这样做，那就去代码网站试试一些新的谜题。

**在本教程中，您已经学习了:**

*   解决谜题如何提高你的编程技能
*   你如何**参与**代码的出现
*   你如何解决不同种类的谜题
*   当解决降临代码难题时，你如何组织你的代码和测试
*   如何在解谜时使用测试驱动开发

Real Python 拥有一个私人排行榜和一个关于代码问世的社区论坛。成为[真正的 Python 成员](https://realpython.com/join/)，加入 [`#advent-of-code`](https://realpython.com/community/) Slack 频道即可访问。

**源代码:** [点击此处下载免费源代码](https://realpython.com/bonus/python-advent-of-code-code/)，向您展示如何用 Python 解决代码难题。*********