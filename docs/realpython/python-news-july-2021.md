# Python 新闻:2021 年 7 月有什么新消息

> 原文：<https://realpython.com/python-news-july-2021/>

【2021 年 7 月对于 Python 社区来说，这是激动人心的一个月！Python 软件基金会雇佣了有史以来第一个 **CPython 常驻开发人员**——一个致力于 CPython 开发的全职带薪职位。来自 CPython 开发团队的其他消息中，[回溯](https://realpython.com/python-traceback/)和**错误消息**得到了一些急需的关注。

让我们深入了解过去一个月最大的 **Python 新闻**！

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## CPython 有一个全职的常驻开发人员

在我们六月的新闻综述中，我们特别报道了 Python 软件基金会宣布他们正在招聘一名常驻 CPython 开发人员。7 月，PSF 的计划取得了成果，聘用了 ukasz Langa。

作为 CPython 核心开发人员和 Python 社区的活跃成员，ukasz 可能为真正的 Python 读者所熟悉。在 [Real Python 播客](https://realpython.com/podcasts/rpp/7/)的第 7 集，茹卡斯加入主持人[克里斯·贝利](https://realpython.com/team/cbailey/)谈论[黑色](https://black.readthedocs.io/en/stable/)代码格式化程序的起源，他作为 Python 3.8 和 3.9 的 Python 发布经理的经历，以及他如何将 Python 与[他对音乐的兴趣](https://www.youtube.com/watch?v=02CLD-42VdI)融合在一起。

作为第一个 CPython 常驻开发人员，ukasz 负责:

*   处理拉取请求和问题积压
*   进行分析研究以了解 CPython 的志愿者时间和资金
*   调查项目优先级及其未来任务
*   处理项目优先级

在关于他的新角色的声明中，他描述了他对 PSF 招聘公告的反应:

> 当 PSF 第一次宣布常驻开发者职位时，我立刻对 Python 充满了难以置信的希望。我认为对于这个项目来说，这是一个具有变革潜力的角色。简而言之，我认为常驻开发人员(DIR)的使命是加速其他所有人的开发体验。这不仅包括核心开发团队，而且最重要的是提交 pull 请求和在 tracker 上创建问题的驱动贡献者。([来源](https://lukasz.langa.pl/a072a74b-19d7-41ff-a294-e6b1319fdb6e/))

在他的个人网站上，ukasz 每周在一系列周报告中记录他的工作。在工作的第一周，他完成了 14 个问题和 54 个拉动式请求(PRs)，审查了 9 个 PRs，并编写了 6 个自己的 PRs。

“不过不要对这些数字太过兴奋，”祖卡斯在他的第一份周报中写道。“按照 CPython 的开发方式，许多变化都是从`main`分支开始的，然后再移植到【Python】3.10，通常还会移植到 3.9。所以有些变化是三倍的。”

每周报告提供的透明度令人耳目一新，并提供了一个独特的角色幕后的外观。未来的申请人将会有一个极好的资源来帮助他们了解这份工作需要什么，什么地方做得好，哪里可以改进。

祖卡斯在 7 月写了两份周报:

*   【2021 年每周报告，7 月 12 日至 18 日
*   【2021 年每周报告，7 月 19 日至 25 日

这个系列一直持续到八月。每份报告都包括一个*亮点*部分，展示了一周内 ukasz 应对的几个最有趣的挑战。这一节尤其值得一读，因为它深入探讨了语言特性和错误修复。

祝贺 ukasz 成为第一位常驻 CPython 开发人员！真正的 Python 很高兴看到他在任期内取得的成就，并为 PSF 成功创造并填补了这一角色而欣喜若狂。

[*Remove ads*](/account/join/)

## Python 3.11 获得增强的错误报告

Python 3.10 的发布指日可待，正如我们在 5 月报道的那样，新的解释器将对错误消息进行一些改进。Python 3.11 继续致力于改进错误。

对，没错！尽管 Python 3.10 要到 10 月才会发布[，但是 Python 3.11 的工作已经在进行了！](https://www.python.org/dev/peps/pep-0619/)

### 回溯中的细粒度错误位置

Python 3.10 和 3.11 的发布经理 Pablo Galindo 于 2021 年 7 月 16 日在[的推特](https://twitter.com/pyblogsal/status/1416034899639750659)上分享了他和他的团队已经完成了 [PEP 657](https://www.python.org/dev/peps/pep-0657/) 的实现。PEP 增加了对“回溯中细粒度错误位置”的支持，对于新的和有经验的 Python 开发人员来说，这是一次重要的用户体验升级。

**有趣的事实:** A **PEP** 是一个 **Python 增强提议**，并且是主要的方法，在其中提议的 Python 语言特性被记录并在整个核心 Python 开发团队中共享。你可以通过阅读 [PEP 1](https://www.python.org/dev/peps/pep-0001) 来了解更多关于 PEP 的信息——第一个 PEP！

为了说明新的错误位置报告有多细粒度，考虑下面的代码片段，它将值`1`赋给名为`x`的嵌套字典中的键`"d"`:

```py
x["a"]["b"]["c"]["d"] = 1
```

在 Python 3.10 之前的任何 Python 3 版本中，如果键`"a"`、`"b"`或`"c"`的任何一个值是`None`，那么执行上面的代码片段会引发一个`TypeError`，告诉您不能下标 a `NoneType`:

```py
Traceback (most recent call last):
  File "test.py", line 2, in <module>
    x['a']['b']['c']['d'] = 1
TypeError: 'NoneType' object is not subscriptable
```

这个误差*是准确的*，但不是*有用的*。哪个值是`None`？是在`x["a"]`、`x["b"]`还是`x["c"]`的值？找到错误的确切位置需要进行更多的调试，并且可能会很昂贵和耗时。

在 Python 3.11 中，相同的代码片段会产生一个回溯，其中包含一些有用的注释，指向`None`值的确切位置:

```py
Traceback (most recent call last):
  File "test.py", line 2, in <module>
    x['a']['b']['c']['d'] = 1
    ~~~~~~~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
```

脱字符指向`NoneType`的确切位置。元凶是`x["c"]`！不再猜测，不再调试。错误消息本身为您提供了查找错误原因所需的所有信息。

Python 社区热烈欢迎这一变化。在撰写本文时，Pablo 的推文已经获得了超过 4000 个赞，评论中充满了 Python 开发人员表达他们的感谢。

一些开发人员设法找到了目前不支持的边缘案例——至少在当前的实现中不支持。例如， [Will McGugan](https://twitter.com/willmcgugan) 想知道新的位置报告对于亚洲字符和表情符号是否会像预期的那样工作。[这条推特帖子](https://twitter.com/willmcgugan/status/1416129912063373324)证实了缺乏支持。

这种改变也是有代价的。正如在 PEP 中提到的[，实现需要“向每个](https://www.python.org/dev/peps/pep-0657/#rationale)[字节码](https://realpython.com/cpython-source-code-guide/#what-does-a-compiler-do)指令添加新数据”

**有趣的事实:**你经常听到 Python 被称为一种**解释语言**，但这并不是 100%准确。事实上，Python 代码被**编译**成一种叫做**字节码**的低级语言。CPython 解释的是编译器生成的字节码指令，而不是您的 Python 代码。

要了解更多关于 CPython 如何工作的信息，请点击 *Real Python* 查看[您的 CPython 源代码指南](https://realpython.com/cpython-source-code-guide)。

添加字节码指令的最终结果是标准库的`.pyc`文件增加了 22%。这听起来是一个显著的增长，但它仅相当于大约 6MB，负责 PEP 的团队认为:

> 这是一个非常容易接受的数字，因为开销的数量级非常小，尤其是考虑到现代计算机的存储大小和内存容量…
> 
> 我们知道这些信息的额外成本对于一些用户来说可能是不可接受的，所以我们提出了一种选择退出机制，它将导致生成的代码对象没有额外的信息，同时也允许 pyc [sic]文件不包括额外的信息。([来源](https://www.python.org/dev/peps/pep-0657/#rationale))

[退出机制](https://www.python.org/dev/peps/pep-0657/#opt-out-mechanism)由一个新的`PYTHONDEBUGRANGES`环境变量和一个新的命令行选项组成。

你可以阅读 [PEP 657](https://www.python.org/dev/peps/pep-0657) 了解更多关于新错误位置报告的信息。你可以在[Python 3.11 新特性](https://docs.python.org/3.11/whatsnew/3.11.html)文档中找到更多关于这个特性的例子。

[*Remove ads*](/account/join/)

### 改进了循环导入的错误消息

CPython 常驻开发人员 ukasz Langa 在他 7 月 19-26 日的每周报告中称，Python 3.11 中增加了一个改进的循环导入错误消息。

考虑以下封装结构:

```py
a
├── b
│   ├── c.py
│   └── __init__.py
└── __init__.py
```

在`a/b/__init__.py`中有下面一行代码:

```py
import a.b.c
```

`c.py`文件包含这行代码:

```py
import a.b
```

这导致了一种情况，即包`b`依赖于模块`c`，即使模块`c`也依赖于包`b`。在 Python 3.10 之前的 Python 版本中，此结构会生成一条含糊的错误消息:

```py
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/private/tmp/cpymain/a/b/__init__.py", line 1, in <module>
    import a.b.c
  File "/private/tmp/cpymain/a/b/c.py", line 3, in <module>
    a.b
AttributeError: module 'a' has no attribute 'b'
```

像这样的消息让无数 Python 开发人员感到沮丧！

多亏了来自 CPython 开发者 Filipe lains 的 pull 请求，新的错误信息更加清晰:

```py
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/private/tmp/cpymain/a/b/__init__.py", line 1, in <module>
    import a.b.c
    ^^^^^^^^^^^^
  File "/private/tmp/cpymain/a/b/c.py", line 3, in <module>
    a.b
    ^^^
AttributeError: cannot access submodule 'b' of module 'a'
(most likely due to a circular import)
```

有没有其他的插入语有可能节省这么多的头部抨击？

在撰写本文时，还不清楚这一变化是否已经被移植到 Python 3.10。您可能需要再等待一个发布周期才能看到新的错误消息。

## Python 的下一步是什么？

7 月，Python 出现了一些令人兴奋的发展。在*真实 Python* 展会上，我们对 Python 的未来感到兴奋，迫不及待地想看看在**8 月**会有什么新东西等着我们。

你最喜欢的来自**7 月**的 **Python 新闻**是什么？我们错过了什么值得注意的吗？请在评论中告诉我们，我们可能会在下个月的 Python 新闻综述中介绍您。

快乐的蟒蛇！**