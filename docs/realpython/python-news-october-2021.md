# Python 新闻:2021 年 10 月有什么新消息

> 原文：<https://realpython.com/python-news-october-2021/>

作为全球志愿者所做的伟大工作的高潮， **Python 3.10** 的发布主宰了 2021 年 10 月**Python 社区的新闻周期**。在这个版本推出新特性的同时，Python 在 **TIOBE 编程社区指数**中被评为本月最佳编程语言。

通过参与 **Python 开发者调查**和回答 PyCon US 2022 **提案征集**，您还有一些支持社区的新机会。

让我们深入了解过去一个月最大的 **Python 新闻**！

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## Python 3.10 版本

Python 的新版本现在每年发布。我们可以期待核心开发者在每年十月和我们其他人**分享一个可爱的糖果袋。随着 Python 3.10 于 10 月 4 日推出测试版，每个人都有一些令人兴奋的东西可以期待。**

Python 的每个版本都有一个发布经理，他负责协调所有的变更，并构建和准备用于分发的文件。Python 3.10 和 3.11 的发布经理是 [Pablo Galindo Salgado](https://twitter.com/pyblogsal) 。在 Python 的第一次尝试中，他构建并在 YouTube 上发布了 Python **。**

[*Remove ads*](/account/join/)

### Python 3.10 亮点

新版本包括对语言的许多改进。我们最喜欢的是改进的**错误消息**，简化的**类型联合的语法**，以及**结构模式匹配**。

[改进的错误消息](https://realpython.com/python310-new-features/#better-error-messages)将使您的生活更加轻松，无论您是新的 Python 开发人员还是有经验的开发人员。特别是，当你的代码[不是有效的 Python](https://realpython.com/invalid-syntax-python/) 时，你得到的反馈在 Python 3.10 中比在以前的版本中更有针对性和可操作性。例如，考虑下面的代码，其中第一行末尾没有右括号:

```py
news = ["errors", "types", "patterns"
print(", ".join(news))
```

在 Python 3.9 和更早版本中，如果您尝试运行此代码，将会看到以下内容:

```py
 File "errors.py", line 2
    print(", ".join(news))
        ^
SyntaxError: invalid syntax
```

这个解释不是很有见地。更糟糕的是，报告的行号是错误的。实际的错误发生在第 1 行，而不是错误消息所说的第 2 行。Python 3.9 中引入的[新解析器](https://realpython.com/python39-new-features/#a-more-powerful-python-parser)，允许更好的反馈:

```py
 File "errors.py", line 1
    news = ["errors", "types", "patterns"
           ^
SyntaxError: '[' was never closed
```

行号没错，附带的解释切中要害。这将允许您直接进入，修复错误，并继续编码！

类型联合的简化语法允许你使用类型提示，通常不需要任何额外的导入。你可以使用[类型提示](https://realpython.com/python-type-checking/)来注释你的代码，从你的编辑器中获得更多的支持，并且更早地发现错误。

[`typing`](https://docs.python.org/3/library/typing.html) 模块是向 Python 添加静态类型的核心。然而，在最近几个版本中，越来越多的工具已经从`typing`转移到内置功能。在 Python 3.10 中，允许使用管道操作符(`|`)来指定类型联合，而不是从`typing`导入`Union`。以下代码片段显示了新语法的示例:

```py
def mean(numbers: list[float | int]) -> float | None:
    return sum(numbers) / len(numbers) if numbers else None
```

`number`的注释指定它应该是一个由`float`和`int`对象组成的列表。以前，你可能会把它写成`List[Union[float, int]]`。类似地，返回值的注释`float | None`是类型联合的一个特例，也可以写成`Optional[float]`。新的语法意味着你可以注释很多代码，甚至不需要导入`typing`。

[结构模式匹配](https://realpython.com/python310-new-features/#structural-pattern-matching)是一种处理数据结构的强大方法，你可能从函数式语言如 Elixir、Scala 和 Haskell 中了解到这一点。我们在三月和八月的新闻简报中预览了这个功能。

当您需要操作[列表](https://realpython.com/python-lists-tuples/)、[字典](https://realpython.com/python-dicts/)、[数据类](https://realpython.com/python-data-classes/)或其他结构时，结构模式匹配处于最佳状态。下面的例子实现了一个[递归函数](https://realpython.com/python-recursion/)，它对一组数字求和。它让您快速了解新语法:

```py
def sum(numbers, accumulator=0):
    match numbers:
        case []:
            return accumulator
        case [head, *tail]:
            return sum(tail, accumulator + head)
```

这段代码使用`accumulator`来跟踪运行总数。您将`numbers`匹配到两个不同的案例。

在第一种情况下，`numbers`是一个空列表。因为你不需要在你的 sum 上增加更多，所以你可以返回`accumulator`。第二种情况说明了当列表中至少有一个元素时该怎么做:命名第一个元素为`head`，命名列表的其余元素为`tail`。您将`head`添加到您的运行总数中，然后递归调用`sum()`获得剩余的元素。

您可以使用`if`语句实现相同的算法。然而，新的语法打开了一个更加功能化的思考 Python 代码的方式，这可能是一个有趣的探索前进的途径。

在我们的[专用教程](https://realpython.com/python310-new-features/)中，深入了解这些改进的细节，以及 Python 3.10 中的所有其他新特性。

[*Remove ads*](/account/join/)

### YouTube 上的现场 Python 3.10 发布会

通常，新 Python 版本的实际发布是在闭门造车的情况下进行的。虽然[提前宣布了](https://www.python.org/dev/peps/pep-0619/)，但是下载[新版本](https://www.python.org/downloads/release/python-3100/)的链接往往会突然出现。

今年不一样了！发布经理 Pablo Galindo Salgado 和来自 [Python Discord](https://www.pythondiscord.com/) 的[Leon sandy](https://twitter.com/lemonsaurus_rex)邀请所有人参加在 [YouTube](https://www.youtube.com/watch?v=AHT2l3hcIJg) 上直播的发布会。尽管互联网经历了糟糕的一天，但直播效果很好，我们都可以看到 Pablo 运行他的神奇脚本，让 Python 在全世界可用。

除了 Pablo 和 Leon，其他几位核心贡献者也参加了聚会:

*   [ukasz Langa](https://twitter.com/llanga)展示了[对打字系统](https://www.youtube.com/watch?v=AHT2l3hcIJg&t=22m15s)的更新。
*   [Brandt Bucher](https://github.com/brandtbucher) 推出[结构模式匹配](https://www.youtube.com/watch?v=AHT2l3hcIJg&t=43m53s)。
*   [Carol Willing](https://twitter.com/WillingCarol) 主持了一场关于 [Python 社区](https://www.youtube.com/watch?v=AHT2l3hcIJg&t=67m50s)的讨论。
*   [Irit Katriel](https://github.com/iritkatriel) 向[展示了如何为 CPython 的发展做出](https://www.youtube.com/watch?v=AHT2l3hcIJg&t=122m53s)贡献。

该流仍然可用。如果您有兴趣获得一个独特的外观，并了解发布新版本的 Python 需要什么，请查看它。

## Python 在 TIOBE 的第一名

TIOBE 编程社区指数是编程语言受欢迎程度的指标。它基于搜索引擎的[结果，已经被追踪了 20 多年。](https://www.tiobe.com/tiobe-index/programming-languages-definition/)

在 10 月份的排名中，Python 首次登上榜首。事实上，这是第一次一种不叫 Java 或 C 的语言登上了索引的榜首。

虽然这只是一个指数，但结果证实 Python 是一种非常受欢迎的编程语言，仍然有很多人对它感兴趣，在线上有很多可供开发人员使用的资源。

## 2021 年 Python 开发者调查

一年一度的 [Python 开发者调查](https://surveys.jetbrains.com/s3/c1-python-developers-survey-2021)已经开始。这项调查对于理解社区如何使用 Python 语言和支持它的生态系统非常重要。来自[早些年](https://www.jetbrains.com/lp/python-developers-survey-2020/)的结果给了我们很多启示。这些结果对于社区的许多部分规划如何使用他们有限的资源是重要的输入。

如果您有时间贡献您的答案，您可以通过打开[调查](https://surveys.jetbrains.com/s3/c1-python-developers-survey-2021)来完成。问题相当多样，但是你可以计划在大约十到十五分钟内完成。今年，有几个新问题将有助于[驻地开发商](https://realpython.com/python-news-july-2021/#cpython-has-a-full-time-developer-in-residence)和[包装项目经理](https://realpython.com/python-news-august-2021/#python-has-a-packaging-project-manager)的工作。

## PyCon US 2022:征集提案

PyCon US 2022 的准备工作进展顺利。会议将于明年 4 月 27 日(T3)至 5 月 5 日(T5)在 T2 盐湖城举行。和往常一样，会议将包括两天的指导研讨会，三天的演讲和其他演示，以及四天的 sprints，在这里你可以和社区中的其他 Python 程序员一起工作。

如果你想参加 PyCon，看看[征集提案](https://pycon.blogspot.com/2021/10/pycon-us-2022-call-for-proposals-is-open.html)。提交提案的截止日期是 2021 年 12 月 20 日。您可以参与四种类型的演示:

1.  会谈通常持续 30 分钟，在主要会议期间举行，从 4 月 29 日到 5 月 1 日。
2.  Charlas 是用西班牙语进行的演讲。4 月 29 日星期五将会有一场查尔斯的演唱会。
3.  **教程**是在会议的前两天(4 月 27 日和 4 月 28 日)进行的三小时研讨会。
4.  在 4 月 29 日至 5 月 1 日的主要会议期间，在会议厅展示海报。

PyCon 鼓励任何人提交提案，不管你的经验水平如何。前往 PyCon 的[提交页面](https://us.pycon.org/2022/speaking/speaking/)了解更多信息。

## 浏览器中的 Visual Studio 代码

Visual Studio 代码编辑器是许多 Python 开发者的最爱。作为[10 月发布](https://code.visualstudio.com/updates/v1_62)的一部分，编辑器可以在完全运行于你的浏览器的零安装版本中获得。

您可以通过导航到 [vscode.dev](https://vscode.dev/) 来打开 web 的 VS 代码。一旦你到了那里，你就可以在支持的浏览器上打开文件，甚至是目录，然后开始工作。还有一些不支持的功能，包括[终端](https://realpython.com/advanced-visual-studio-code-python/#setting-up-your-terminal)和[调试器](https://realpython.com/advanced-visual-studio-code-python/#debugging-your-python-scripts-in-visual-studio-code)。尽管如此，你将会有一个很好的编辑体验，你已经习惯了桌面版本的大部分功能和扩展。

网络编辑器让你可以即时访问存储在 [GitHub](https://github.com/) 或 [Azure](https://dev.azure.com/) 中的代码。您可以导航到一个存储库，然后在 URL 前面添加`vscode.dev`,在编辑器中打开它。例如，您可以通过输入`vscode.dev/github.com/realpython/reader`作为您的 URL 在`github.com/realpython/reader`打开存储库。这类似于——但不完全相同——当你在 GitHub 中查看一个库时，按下 `.` 来启动一个编辑器。

[*Remove ads*](/account/join/)

## Python 的下一步是什么？

随着 Python 语言新版本的发布，10 月对 Python 来说永远是一个激动人心的月份。在*真实的 Python* ，我们期待着深入挖掘 Python 3.10，我们迫不及待地想看看在**11 月**会有什么新东西等着我们。

来自**10 月**的 **Python 新闻**你最喜欢的片段是什么？我们错过了什么值得注意的吗？请在评论中告诉我们，我们可能会在下个月的 Python 新闻综述中介绍您。

快乐的蟒蛇！***