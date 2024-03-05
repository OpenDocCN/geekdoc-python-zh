# Python 新闻:2021 年 3 月有什么新消息

> 原文：<https://realpython.com/python-news-march-2021/>

Python 在很多方面都是一种动态语言:它不仅不是像 C 或 C++那样的静态语言，而且还在不断发展。如果你想了解 2021 年 3 月在**的**巨蟒**世界里发生的事情，那么你来对地方了，来获取你的**新闻**！**

2021 年 3 月标志着 Python 语言的核心发生了显著的变化，增加了[结构模式匹配](https://www.python.org/dev/peps/pep-0636/)，现在可以在 Python 3.10.0 的最新 [alpha 版本中测试。](https://pythoninsider.blogspot.com/2021/03/python-3100a6-is-now-available-for.html)

除了语言本身的变化，对于 Python 来说，三月是一个充满激动人心的历史性时刻的月份。这种语言[庆祝了它的 30 岁生日](https://pyfound.blogspot.com/2021/03/happy-anniversary-to-python-and-python.html)，并成为第一批[登陆另一个星球](https://twitter.com/thepsf/status/1362516507918483458)的开源技术之一。

让我们深入了解过去一个月最大的 Python 新闻！

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

### Python 岁了

虽然 Python 的实际出生日期是[1991 年 2 月 20 日](https://python-history.blogspot.com/2009/01/brief-timeline-of-python.html)，也就是[0 . 9 . 0 版](https://github.com/smontanaro/python-0.9.1)发布的时间，但是三月是一个值得庆祝的好月份。今年 3 月是 [Python 软件基金会](https://www.python.org/psf/)成立于 2001 年 3 月 6 日的[20 周年](https://pyfound.blogspot.com/2021/03/happy-anniversary-to-python-and-python.html)。

在过去的三十年里，Python 已经发生了很大的变化，无论是作为一门语言还是作为一个组织。从 Python 2 到 Python 3 的过渡[花了十年才完成](https://www.python.org/doc/sunset-python-2/#:~:text=The%20sunset%20date%20has%20now,when%20we%20released%20Python%202.7.)。决策的组织模式也发生了变化:该语言的创造者吉多·范·罗苏姆曾是掌舵人，但 2018 年成立了一个五人指导委员会，以规划 Python 的未来。

生日快乐，巨蟒！为🥂的更多岁月干杯

[*Remove ads*](/account/join/)

### 结构模式匹配来到 Python 3.10.0

Python 3.10.0 是 Python 的下一个次要版本，预计将于 2021 年 10 月 4 日发布。此次更新将为核心语法带来一大新增:**结构模式匹配**，这是在 [PEP 634](https://www.python.org/dev/peps/pep-0634/) 中提出的。你可以说结构模式匹配给 Python 增加了一种 [`switch`语句](https://en.wikipedia.org/wiki/Switch_statement)，但这并不完全准确。模式匹配做得更多。

举个例子，来自 [PEP 635](https://www.python.org/dev/peps/pep-0635/) 。假设您需要检查对象`x`是否是一个包含套接字连接的主机和端口信息的元组，以及可选的模式，比如 HTTP 或 HTTPS。您可以使用一个 [`if` … `elif` … `else`](https://realpython.com/python-conditional-statements/) 块来编写这样的代码:

```py
if isinstance(x, tuple) and len(x) == 2:
    host, port = x
    mode = "http"
elif isinstance(x, tuple) and len(x) == 3:
    host, port, mode = x
else:
    # Etc…
```

Python 新的结构模式匹配允许你使用一个 **`match`** 语句来更清晰地写这个:

```py
match x:
    case host, port:
        mode = "http"
    case host, port, mode:
        pass
    # Etc…
```

`match`语句检查对象的*形状*是否匹配其中一种情况，并将数据从对象绑定到`case`表达式中的变量名。

不是每个人都对模式匹配感到兴奋，这个特性受到了核心开发团队和[更广泛的社区](https://twitter.com/brandon_rhodes/status/1360032460613050368)的批评。在接受公告中，指导委员会承认这些问题，同时也表示支持该提案:

> 我们承认模式匹配对 Python 来说是一个很大的改变，在整个社区达成共识几乎是不可能的。不同的人对语义和语法的不同方面有所保留或关注(指导委员会也是如此)。尽管如此，经过深思熟虑，……我们相信，PEP 634 等规范中规定的模式匹配将是对 Python 语言的一大补充。([来源](https://lwn.net/Articles/845480/))

尽管意见不一，模式匹配*是*即将到来的下一个 Python 版本。你可以通过阅读 [PEP 636](https://www.python.org/dev/peps/pep-0636/) 中的教程来了解更多关于模式匹配的工作原理。

### 巨蟒登陆火星

2 月 18 日，[毅力](https://www.nasa.gov/perseverance)号火星车经过 7 个月的旅程在火星着陆。(从技术上来说，这是一个 2 月的新闻项目，但它太酷了，我们不得不将其纳入本月！)

坚持不懈带来了大量的新仪器和科学实验，这将为科学家们提供迄今为止对火星的最佳观察。毅力依赖于一系列开源软件和现成的硬件，这使它成为迄今为止最容易实现的火星漫游项目。

Python 是依靠毅力生存的开源技术之一。它在火星车上被用来处理着陆过程中拍摄的图像和视频。

毅力号携带的最令人兴奋的实验之一是别出心裁的火星直升机，这是一架小型无人机，用于在稀薄的火星大气中测试飞行。Python 是[飞控软件](https://nasa.github.io/fprime/#f-system-requirements)的**开发需求**之一，称为**F’**。

### 2020 年 Python 开发者调查结果在

由 [JetBrains](https://www.jetbrains.com/) 和 Python 软件基金会进行的 [2020 Python 开发者调查](https://www.jetbrains.com/lp/python-developers-survey-2020/)的结果出来了，与去年的调查相比，他们显示了一些有趣的变化。

2020 年，94%的受访者报告主要使用 Python 3，高于 2019 年的 90%和 2017 年的 75%。有趣的是，Python 2 仍然在计算机图形和游戏开发领域的大多数受访者中使用。

Flask 和 [Django](https://realpython.com/tutorials/django/) 继续主导 web 框架，分别拥有 46%和 43%的采用率。新人 [FastAPI](https://realpython.com/fastapi-python-web-apis/) 以 12%的采用率成为第三大最受欢迎的 web 框架——考虑到 2020 年是该框架首次出现在选项列表中，这是一个令人难以置信的壮举。

在“你目前的 Python 开发使用的主要编辑器是什么？”这个问题上，Visual Studio Code 获得了超过 5%的回答份额这使得微软的 IDE 占据了 29%的份额，进一步缩小了 Visual Studio 代码和 [PyCharm](https://realpython.com/pycharm-guide/) 之间的差距，后者仍以 33%的份额高居榜首。

查看[调查结果](https://www.jetbrains.com/lp/python-developers-survey-2020/)，了解更多关于 Python 及其生态系统的统计数据。

[*Remove ads*](/account/join/)

### Django 3.2 的新特性

Django 3.2 将于 2021 年 4 月发布，随之而来的是一系列令人印象深刻的新特性。

一个主要的更新增加了对**函数索引**的支持，它允许你索引表达式和[数据库](https://realpython.com/tutorials/databases/)函数，比如索引小写文本或涉及一个或多个数据库列的数学公式。

函数索引是在`Model`类的`Meta.indexes`选项中创建的。这里有一个改编自[官方发布说明](https://docs.djangoproject.com/en/3.2/releases/3.2/#functional-indexes)的例子:

```py
from django.db import models
from django.db.models import F, Index, Value

class MyModel(models.Model):
    height = models.IntegerField()
    weight = models.IntegerField()

    class Meta:
        indexes = [
 Index( F("height") / (F("weight") + Value(5)), name="calc_idx", ),        ]
```

这将创建一个名为`calc_idx`的函数索引，该索引对一个表达式进行索引，该表达式将`height`字段除以`weight`字段，然后加上`5`。

Django 3.2 中的另一个与索引相关的变化是支持覆盖索引的 PostgreSQL。一个**覆盖索引**允许您在一个索引中存储多个列。这使得只包含索引字段的查询能够得到满足，而不需要额外的表查找。换句话说，您的查询可以快得多！

另一个值得注意的变化是添加了管理站点装饰者，简化了自定义显示 T2 和动作功能 T4 的创建。

要获得 Django 3.2 中新特性的完整列表，请查看官方发布说明。 *Real Python* 贡献者 [Haki Benita](https://realpython.com/team/hbenita/) 也有一篇有用的[概述文章](https://hakibenita.com/django-32-exciting-features)，通过更多的上下文和几个例子带你了解一些即将推出的特性。

### PEP 621 到达最终状态

早在 2016 年， [PEP 518](https://www.python.org/dev/peps/pep-0518/) 引入了 [`pyproject.toml`文件](https://snarky.ca/what-the-heck-is-pyproject-toml/)作为指定项目构建需求的标准化地方。以前，您只能在 [`setup.py`文件](https://realpython.com/pypi-publish-python-package/#configuring-your-package)中指定元数据。这导致了一些问题，因为执行`setup.py`和读取构建依赖关系需要安装一些构建依赖关系。

在过去的几年里越来越受欢迎，现在不仅仅用于存储构建需求。像 [`black`自动套用格式器](https://github.com/psf/black)这样的项目使用`pyproject.toml`来[存储包配置](https://github.com/psf/black/blob/master/pyproject.toml)。

[PEP 621](https://www.python.org/dev/peps/pep-0621/#abstract) 于 2020 年 11 月被临时接受，并于 2021 年 3 月 1 日被标记为最终版本，它规定了如何在`pyproject.toml`文件中写入项目的核心元数据。从表面上看，这似乎是一个不太重要的 PEP，但它代表了一个远离`setup.py`文件的持续运动，并指向 Python 打包生态系统的改进。

### PyPI 现在是 GitHub 秘密扫描集成商

**Python 包索引**，或 [PyPI](https://realpython.com/pypi-publish-python-package/) ，是*下载所有组成 Python 丰富生态系统的包的地方。在`pypi.org`网站和`files.pythonhosted.org`网站之间，PyPI 每月产生超过[20*Pb*的流量](https://twitter.com/EWDurbin/status/1375748779450765316?s=20)。超过 20，000 太字节！*

有这么多的人和组织依赖 PyPI，保证索引的安全是最重要的。这个月，PyPI 成为了官方的 [GitHub 秘密扫描集成商](https://github.blog/changelog/2021-03-22-the-python-package-index-is-now-a-github-secret-scanning-integrator/)。GitHub 现在[会检查每一个提交给公共库](https://docs.github.com/en/code-security/secret-security/about-secret-scanning)的泄露的 PyPI API 令牌，如果发现任何漏洞，会禁用库并通知其所有者。

### Python 的下一步是什么？

Python 继续以越来越大的势头发展。随着越来越多的用户转向这种语言来完成越来越多的任务，Python 及其生态系统将会继续发展，这是很自然的。在*真实 Python* 展会上，我们对 Python 的未来感到兴奋，迫不及待地想看看在**4 月**会有什么新的东西等着我们。

来自**三月**的 **Python 新闻**你最喜欢的片段是什么？我们错过了什么值得注意的吗？请在评论中告诉我们，我们可能会在下个月的 Python 新闻综述中介绍您。

快乐的蟒蛇！**