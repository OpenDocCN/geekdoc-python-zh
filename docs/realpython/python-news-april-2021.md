# Python 新闻:2021 年 4 月有什么新消息

> 原文：<https://realpython.com/python-news-april-2021/>

如果你和 Python 开发人员相处的时间足够长，你最终会听到有人谈论 Python 社区有多棒。如果你想了解 2021 年 4 月**的 **Python 社区**中发生的事情，那么你来对了地方来获取你的**新闻**！*

*从改善用户体验的更好的错误消息到社区驱动的推迟 CPython 变更的努力，2021 年 4 月是一个充满故事的月份，这些故事提醒我们 Python 因其社区而变得更好。

让我们深入了解过去一个月最大的 Python 新闻！

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 由于有远见的赞助商，PSF 正在招人

2021 年 2 月，谷歌成为 **Python 软件基金会(PSF)** 第一个有远见的赞助商。此后不久，彭博工程公司也成为了一个有远见的赞助商。

**有远见的赞助商**是最高级别的赞助商，提供大量资金支持 PSF 计划。来自谷歌和彭博的赞助正在帮助 PSF 雇佣两名新的全职员工。

[*Remove ads*](/account/join/)

### CPython 常驻开发人员

多亏了谷歌的赞助基金，PSF 已经宣布计划雇佣一名常驻开发者。根据 [PSF 的公告](https://pyfound.blogspot.com/2021/04/the-psf-is-hiring-developer-in.html)，常驻开发者将“解决积压问题，进行分析研究以了解项目的志愿者时间和资金，调查项目优先级及其未来任务，并开始处理这些优先级。”

该全职职位的资助期限为一年，截止日期为**2021 年 5 月 16 日开始接受简历[提交。](https://mail.python.org/archives/list/python-committers@python.org/thread/QRKY4T7UCFQH4ZUPJT5IXSSIPXNLTCGK/)**然而，这个职位似乎只对现有的核心开发人员开放。

雇佣一名全职员工来支持 CPython 开发对于 PSF 和 Python 社区来说是一个巨大的进步。这个决定是受 Django Fellowship Program 的启发，该项目雇佣有偿承包商来处理行政和社区管理任务。

### Python 打包项目经理

彭博的捐款资助了一个 Python 包装项目经理的职位。根据 [PSF 的公告](https://pyfound.blogspot.com/2021/04/the-psf-is-hiring-python-packaging.html)，项目经理将“监督改进和增加的功能，这将使所有 Python 用户受益，同时引导 PyPI 发展成为可持续的服务。”

申请截止日期为**2021 年 5 月 18 日**。想了解更多关于这个职位的信息，请查看 [Python 职位公告栏](https://www.python.org/jobs/5317/)。

你可以阅读更多关于彭博支持 Python 的决定，以及他们为什么对 Python 打包生态系统特别感兴趣的博客文章[通过“左移”支持 Python 社区](https://www.techatbloomberg.com/blog/supporting-the-python-community-by-shifting-left/)

## Python 3.10 将改进错误消息

2021 年 4 月 9 日，Python 核心开发者 [Pablo Galindo](https://twitter.com/pyblogsal) ，他也是 Python 3.10 和 3.11 的发布经理，在推特上向 Python 教育者提出了一个问题:

> Python 教育者和用户:最近我一直致力于改进 CPython 中的语法错误消息。什么错误(目前只有*个语法错误*😅)你或你的学生感到困惑的信息？你认为我们应该改进哪些？🤔(请努力帮助更多的人🙏).([来源](https://twitter.com/pyblogsal/status/1380516575485263873)

这条推文得到了很多关注，包括要求改进关于[赋值(`=`)和比较(`==`)操作符](https://realpython.com/python-operators-expressions/)、缩进错误和缺少冒号的错误消息。在某些情况下，加林多指出他已经改进了人们提到的错误！

例如，在一个标题为 [bpo-42997:改善缺失的错误消息:在套件](https://github.com/python/cpython/pull/24292)之前，Galindo 改善了缺失冒号的错误消息。在 Python 3.10 中，如果您忘记在定义函数的[后键入冒号(`:`，您将会看到这个新的和改进的消息:](https://realpython.com/defining-your-own-python-function/)

>>>

```py
>>> # Python 3.10a7
>>> def f()
  File "<stdin>", line 1
    def f()
          ^
SyntaxError: expected ':'
```

对比一下 [Python 3.9](https://realpython.com/python39-new-features/) 中的错误消息:

>>>

```py
>>> # Python 3.9.4
>>> def f()
  File "<stdin>", line 1
    def f()
           ^
SyntaxError: invalid syntax
```

这是一个很小的变化，但是指出解释器需要一个冒号，而不仅仅是告诉用户他们的[语法是无效的](https://realpython.com/invalid-syntax-python/)更有帮助，而且不仅仅是对初学者。有其他语言经验但不熟悉 Python 语法的开发人员也会喜欢更友好的消息传递。

标题为 [bpo-43797:改进无效比较的语法错误](https://github.com/python/cpython/pull/25317)的 PR 中介绍了改进的无效比较错误消息。在 Python 3.10 中，如果您不小心在`if`语句中键入了赋值运算符而不是比较运算符，您将会看到以下消息:

>>>

```py
>>> # Python 3.10a7
>>> a = 1
>>> b = 2
>>> if a = b:
  File "<stdin>", line 1
    if a = b:
       ^^^^^
SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='?
```

这是对 Python 3.9 中现有消息的重大改进:

>>>

```py
>>> # Python 3.9.4
>>> a = 1
>>> b = 2
>>> if a = b:
  File "<stdin>", line 1
    if a = b:
         ^
SyntaxError: invalid syntax
```

除了更好的`SyntaxError`消息外， [`AttributeError`](https://twitter.com/pyblogsal/status/1382290422970654720) ， [`NameError`](https://twitter.com/pyblogsal/status/1382339369743319040) ， [`IndentationError`](https://twitter.com/pyblogsal/status/1384881060396285955) 消息也得到了改进。

Galindo 在 4 月 14 日的推文中分享了一个例子，展示了如果你输入了错误的属性名称，Python 3.10 将如何建议现有的属性:

>>>

```py
>>> # Python 3.10a7
>>> import collections
>>> collections.namedtoplo
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'collections' has no attribute 'namedtoplo'. Did you mean: 'namedtuple'?
```

一些用户对搜索现有姓名的费用表示关切。以下是加林多的回应:

> 它只发生在显示未被捕获的异常时，所以它发生在解释器将要结束时，所以它不会影响运行时。即使在这种情况下，它也有许多限制(如字符串的长度或候选项的数量)来保持成本最小。([来源](https://twitter.com/pyblogsal/status/1382382009620779010))

总的来说，改进的错误消息对 Python 的用户体验是一个很大的改进。有关错误消息改进的完整列表，请查看 Python 文档中的[Python 3.10 新特性](https://docs.python.org/3.10/whatsnew/3.10.html#better-error-messages)页面。

[*Remove ads*](/account/join/)

## PEP 563、PEP 649 和 Python 类型注释的未来

[PEP 484](https://www.python.org/dev/peps/pep-0484) 早在 2014 年就引入了类型提示。类型提示允许您在函数参数、类属性和变量上指定类型，稍后可以使用像 [Mypy](https://github.com/python/mypy) 这样的工具[静态类型检查](https://realpython.com/python-type-checking/)。

例如，下面的代码定义了函数`add()`，它将两个[整数](https://realpython.com/python-numbers/#integers)、`x`和`y`相加:

```py
def add(x: int, y: int) -> int:
    return x + y
```

类型提示指定两个参数`x`和`y`应该是类型`int`并且函数返回一个`int`值。

### 当前如何评估类型提示

目前，类型提示[必须是有效的 Python 表达式](https://www.python.org/dev/peps/pep-0484/#acceptable-type-hints)，因为它们在**函数定义时间**被求值。一旦类型提示被评估，它们就作为[字符串](https://realpython.com/python-strings/)存储在对象的`.__annotations__`属性中:

>>>

```py
>>> def add(x: int, y: int) -> int:
...     return x + y
...
>>> add.__annotations__
{'x': 'int', 'y': 'int', 'return': 'int'}
```

您通常可以使用`typing`模块中的`get_type_hints()`来检索实际的类型对象:

>>>

```py
>>> import typing
>>> typing.get_type_hints(add)
{'x': <class 'int'>, 'y': <class 'int'>, 'return': <class 'int'>}
```

为实现运行时类型检查的工具提供一些基本支持。Python 作为一种[动态类型语言](https://en.wikipedia.org/wiki/Type_system#DYNAMIC)，如果没有第三方工具，可能永远不会支持真正的运行时类型检查。

在函数定义时计算类型提示有一些缺点。例如，必须定义类型的名称。否则，Python 会抛出一个`NameError`:

>>>

```py
>>> def add(x: Number, y: Number) -> Number:
...     return x + y
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'Number' is not defined
```

您可以通过使用类似于 [`typing.Union`](https://docs.python.org/3/library/typing.html#typing.Union) 的东西定义一个[类型别名](https://realpython.com/python-type-checking/#type-aliases)来创建一个允许`int`和`float`值的`Number`别名，从而避免这种情况:

>>>

```py
>>> from typing import Union
>>> Number = Union[int, float]
>>> def add(x: Number, y: Number) -> Number:
...     return x + y
...
>>> # No NameError is raised!
```

但是，有一种情况，当您需要从类本身的方法中[返回](https://realpython.com/python-return-statement/)一个类的实例时，您不能在使用类型别名之前定义它:

```py
class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    @classmethod
    def origin(cls) -> Point:  # Point type is still undefined
        return cls(0, 0)
```

在名字被定义之前使用它被称为**前向引用**。为了避免这种情况，PEP 484 要求使用字符串作为类型名:

```py
class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

 @classmethod    def origin(cls) -> "Point":
        return cls(0, 0)
```

对字符串的求值被延迟，直到模块完全加载。

在函数定义时计算类型注释的另一个缺点是，它会在导入类型化模块时增加计算开销。

[*Remove ads*](/account/join/)

### PEP 563 建议改进什么类型注释

2017 年 9 月， [PEP 563](https://www.python.org/dev/peps/pep-0563/) 提出从在函数定义时评估注释改为在内置`__annotations__`字典中以字符串形式保存注释。这有效地解决了前向引用问题，并消除了导入类型化模块的计算开销。

从 Python 3.7 开始，您可以通过从`__future__`导入`annotations`来[访问这个新行为](https://realpython.com/python37-new-features/#typing-enhancements)。这允许你使用一个`Number`类型的提示，即使它还没有被定义:

>>>

```py
>>> from __future__ import annotations
>>> def add(x: Number, y: Number) -> Number:
...     return x + y
...
>>> # No NameError is raised!
>>> add.__annotations__
{'x': 'Number', 'y': 'Number', 'return': 'Number'}
```

为了评估类型提示，您需要调用`get_type_hints()`。换句话说，类型提示的评估被推迟到调用`get_type_hints()`或其他函数，比如 [`eval()`](https://realpython.com/python-eval-function/) 。

最初，计划是逐步引入这一新行为，并最终将其作为 Python 4 中的默认行为。然而，该决定是在 2020 年 Python 语言峰会后做出的，将于 2021 年 10 月发布。

关于这种新行为将如何影响 Python 用户，PEP 563 做出如下声明:

> 静态类型检查器看不到行为上的差异，而运行时使用注释的工具将不得不执行延迟的评估。([来源](https://www.python.org/dev/peps/pep-0563/#implementation))

换句话说，工具不能再期望为它们评估类型提示，并且需要更新它们的代码来根据需要显式地评估注释。

随着 PEP 563 被接受并计划改变默认行为，像 [FastAPI](https://github.com/tiangolo/fastapi) 和 [pydantic](https://github.com/samuelcolvin/pydantic) 这样在运行时使用注释的项目开始支持 PEP。

### 为什么项目难以实施 PEP 563

2021 年 4 月 15 日，pydantic 的所有者和核心贡献者 [Samuel Colvin](https://twitter.com/samuel_colvin) 在 pydantic 的 GitHub 知识库上为[撰写了一期](https://github.com/samuelcolvin/pydantic/issues/2678)文章，解释了“试图评估那些字符串以获得真正的注释对象是多么困难，也许不可能总是正确的。”

Colvin 列出了 22 个 pydantic 问题，强调了维护人员在尝试实现 PEP 563 时遇到的困难。他这样解释实施如此艰难的原因:

> 原因很复杂，但基本上`typing.get_type_hints()`并不总是有效，我们介绍的试图修复它的众多方法也是如此。即使`typing.get_type_hints()`没有错误，它仍然会比当前的语义慢很多。([来源](https://github.com/samuelcolvin/pydantic/issues/2678)

在 bugs.python.org 搜索[“get _ type _ hints”](https://bugs.python.org/issue?%40columns=id%2Cactivity%2Ctitle%2Ccreator%2Cassignee%2Cstatus%2Ctype&%40sort=-activity&%40filter=status&%40action=searchid&ignore=file%3Acontent&%40search_text=get_type_hints&submit=search&status=-1%2C1%2C2%2C3)发现了许多未解决的问题，似乎证实了科尔文的说法，即`typing.get_type_hints()`并不是在每种情况下都能正确工作。

作为一个解决方案，Colvin 指出了 [PEP 649](https://www.python.org/dev/peps/pep-0649/) ，它是由 Python 核心开发者 Larry Hastings 在 2021 年 1 月编写的。PEP 649 提出了注释的延期评估，而不是 PEP 563 的延期评估。

简而言之，PEP 649 推迟类型提示的评估，直到访问了`.__annotations__`属性。这解决了 PEP 563 提出的两个问题，同时也解决了 pydantic 和 FastAPI 等项目遇到的问题。

Colvin 在 pydantic 的 GitHub 知识库上发表了他对 PEP 649 的支持，在 4 月 15 日的一条消息中也表达了他对 PEP 649 的支持[python-dev 邮件列表上。](https://mail.python.org/mailman3/lists/python-dev.python.org/)

### Python 用户和核心开发者如何达成友好的解决方案

在 Colvin 表达了他对 python-dev 的担忧之后，核心开发人员开始与他讨论如何解决这个问题。

当一个用户请求指导委员会接受 PEP 649 并避免破坏 pydantic 时，核心开发人员 [Paul Ganssle](https://twitter.com/pganssle) 回应指出这些不是唯一的选择，并建议在 Python 3.11 发布之前保持 PEP 563 可选:

> 我应该指出，“接受 PEP 649”和“打破 pydantic”不是这里唯一的选择。将打破 pydantic 的是 PEP 563 弃用期的结束，而不是实现 PEP 649 的失败。
> 
> 其他可行的选择包括:
> 
> *   在我们就这个问题的解决方案达成一致之前，不要加入 PEP 563。
> *   永远离开 PEP 563 选择加入。
> *   弃用 PEP 563，回到原状。
> 
> …假设这是一个真正的问题(部分基于 attrs 花了多长时间才获得对 PEP 563 的支持…如果 PEP 563 也在其他几个地方悄悄破坏工作，我不会感到惊讶)，我的投票是让 PEP 563 至少在 3.11 之前选择加入，而不是试图匆忙完成对 PEP 649 的讨论和实施。([来源](https://mail.python.org/archives/list/python-dev@python.org/message/GT2HQDH2HOZFSOTA5LHTFL5OV46UPKTB/))

就连 Pablo Galindo 也对 Colvin 最初的 pydantic 问题发表了意见，他表示希望 Colvin 能够早点通知核心团队，同时也确认团队正在认真对待 Colvin 的反馈:

> 作为 Python 3.10 的发布经理，我很难过这里提到的第一个问题…可以追溯到 2018 年，但我们听到了所有这些问题，以及它们如何影响 pydantic 严重危险地接近 beta 冻结。…
> 
> 无论如何，对我们来说，确保我们所有的用户群都被考虑在内是一件非常严肃的事情，所以你可以确信我们在讨论整体问题时会考虑到这一点。([来源](https://github.com/samuelcolvin/pydantic/issues/2678#issuecomment-820835053)

Python 核心开发者和指导委员会成员 Carol Willing 也在 Colvin 的问题上发帖，以证实他的担忧，并向所有人保证可以达成解决方案:

> 让我先声明，我是 pydantic 和 FastAPI 的一个非常满意的用户，我非常感谢维护者和他们周围的社区所做的工作和贡献。…
> 
> 我很乐观，我们可以找到 pydantic / FastAPI 和 Python 的双赢。我认为，如果我们不试图过早地将解决办法两极分化为“要么全有，要么全无”或“接受或拒绝 649”，这是可能的。为了实现这一点，我们需要通过“什么是可能的”来看待这个问题，权衡利弊，并朝着“好但可能不理想”的解决方案努力。([来源](https://github.com/samuelcolvin/pydantic/issues/2678#issuecomment-820866646))

最后，在 4 月 20 日，就在 Colvin 提醒 Python 核心开发者 pydantic 面临的问题的五天后，指导委员会[宣布](https://mail.python.org/archives/list/python-dev@python.org/thread/CLVXXPQ2T2LQ5MP2Y53VVQFCXYWQJHKZ/)将推迟 PEP 563 的采用，直到 Python 3.11:

> 指导委员会已经仔细考虑了这个问题，以及许多建议的替代方案和解决方案，我们已经决定，在这一点上，我们不能冒险破坏 PEP 563 的兼容性。我们需要回滚使字符串化注释成为默认注释的更改，至少在 3.10 中是这样。(巴勃罗已经在做这个了。)
> 
> 明确一点，我们并不是还原 PEP 563 本身。未来的导入将继续像 Python 3.7 以来那样工作。在 Python 3.11 之前，我们不会将 PEP 563 基于字符串的注释作为默认注释。这将让我们有时间找到一个适合所有人的解决方案。([来源](https://mail.python.org/archives/list/python-dev@python.org/thread/CLVXXPQ2T2LQ5MP2Y53VVQFCXYWQJHKZ/))

指导委员会的决定受到了 Colvin 的欢迎，并受到了 pydantic 和 FastAPI 用户的欢迎。这一决定也赢得了 Python 开发者[吉多·范·罗苏姆](https://twitter.com/gvanrossum)的赞扬，他对指导委员会表示赞赏:

> 你有所罗门的智慧。回滚使 PEP 563 成为默认行为的代码是 3.10 唯一明智的解决方案。([来源](https://mail.python.org/archives/list/python-dev@python.org/message/4BFSYEKU3P7FNRKVDD7RZXTNEEA6PRXU/))

最终，pydantic 维护人员避免了一个严重的头痛问题，Python 核心开发人员也是如此。正如 Galindo 所指出的，重要的是，在实现 PEP 时遇到问题的维护人员要尽快联系核心开发人员，以避免混乱的局面，并确保及时满足需求。

看起来科尔文已经把这个反馈记在心里了。他在听到指导委员会的决定后对加林多的答复中说:

> 这对我来说是一次非常积极的经历，我对与 python-dev 社区的交流更加积极。
> 
> 在未来，我会参与到发布过程中，把它当成我的一部分(非常小的一部分)，而不是发生在我身上的事情。([来源](https://github.com/samuelcolvin/pydantic/issues/2678#issuecomment-823590454))

项目维护人员是 Python 发布过程的一部分，而不是受制于匿名开发人员突发奇想的无助的旁观者，这种想法对每个人来说都是很重要的。

作为 Python 用户，我们的反馈有助于帮助 Python 核心开发者和指导委员会做出决策。即使我们可能不同意做出的每一个决定，科尔文的经历证明了指导委员会听取我们的意见。

[*Remove ads*](/account/join/)

## Python 的下一步是什么？

4 月，Python 出现了一些激动人心的新发展。在*真实 Python* 展会上，我们对 Python 的未来感到兴奋，迫不及待地想看看在**5 月**会有什么新的东西等着我们。

来自**的 **Python 新闻**4 月**你最喜欢的一条是什么？我们错过了什么值得注意的吗？请在评论中告诉我们，我们可能会在下个月的 Python 新闻综述中介绍您。

快乐的蟒蛇！******