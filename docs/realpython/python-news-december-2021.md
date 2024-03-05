# Python 新闻:2021 年 12 月有什么新消息

> 原文：<https://realpython.com/python-news-december-2021/>

2021 年 12 月的**年，第四届 Python **指导委员会**选举产生，一如既往地由新老成员组成。Python 生命周期的发布周期一直在旋转，随着新的版本 **Python 3.10** 和即将发布的 **Python 3.11** 的发布。与此同时，流行的 **Python 3.6** 也到了生命周期的尽头，将不再被支持。**

在这一连串的活动中，来自各地的开发人员通过解决一年一度的**降临代码**谜题在假期中获得了一些乐趣。

让我们深入了解过去一个月最大的 **Python 新闻**！

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## Python 指导委员会选举

[吉多·范·罗苏姆](https://twitter.com/gvanrossum)是 Python 的创造者。很长一段时间，他也是语言的[**【BDFL】**](https://en.wikipedia.org/wiki/Benevolent_dictator_for_life)仁慈的终身独裁者，负责监督所有被实施的变革。

2018 年夏天，Guido [辞去了](https://mail.python.org/pipermail/python-committers/2018-July/005664.html)BDFL 的职务，并要求社区提出一种新的语言治理模式。在一些[讨论](https://www.python.org/dev/peps/pep-8000/)之后，社区决定选举一个[指导委员会](https://www.python.org/dev/peps/pep-0013/)来指导 Python 的开发。

新的指导委员会定期选举，或多或少与 Python 的每个版本相一致。因此，这些术语通常由将在该术语期间发布的相应 Python 版本来标记。Python 3.11 任期的最近一次选举于 12 月上半月举行，最终结果于 12 月 17 日公布。

自第一届指导委员会于 2019 年 1 月选举产生以来，以下成员已任职:

*   巴里华沙(3.8，3.9，3.10)
*   布雷特·卡农 (3.8，3.9，3.10)
*   **卡罗尔·威林** (3.8，3.9，3.10)
*   **古多·凡·rossum**(3.8)
*   尼克·科格兰 (3.8)
*   巴勃罗·加林多·萨尔加多 (3.10)
*   托马斯伍特斯 (3.9，3.10)
*   维克多·斯坦纳 (3.9)

对于 Python 3.11 任期选举，有十个合格的候选人。投票期从 2021 年 12 月 1 日持续至 15 日。总共有 67 名 Python 核心开发人员投了票。

获得[最多票数](https://www.python.org/dev/peps/pep-8103/#results)的五个人，也就是新指导委员会的成员是:

*   布雷特·卡农
*   格雷戈里·史密斯
*   巴勃罗·加林多·萨尔加多
*   **彼得·维多利亚**
*   托马斯·伍特斯

布雷特、[巴勃罗](https://twitter.com/pyblogsal)和[托马斯](https://twitter.com/yhg1s)是理事会的返回成员，而[格雷戈里](https://twitter.com/gpshead)和[彼得](https://twitter.com/EnCuKou)将担任他们的第一个任期。

指导委员会在让每个人都有发言权的同时，还承担着指导机构群体的重要工作。去年春天围绕[推迟评估注解](https://realpython.com/python37-new-features/#typing-enhancements)的[讨论](https://realpython.com/python-news-april-2021/#pep-563-pep-649-and-the-future-of-python-type-annotations)是一个很好的例子，说明了理事会如何通过寻求共识来发挥领导作用。

我们在 *Real Python* 感谢指导委员会为语言和社区所做的工作，我们祝愿委员会在新的任期一切顺利。

[*Remove ads*](/account/join/)

## 新的 Python 版本

Python 的最新版本， [Python 3.10](https://realpython.com/python310-new-features/) ，是[在 2021 年 10 月](https://realpython.com/python-news-october-2021/#the-python-310-release)发布的。它的第一个维护版本 Python 3.10.1 于 12 月 6 日发布。像往常一样，这个小版本包括[许多小的](https://docs.python.org/3.10/whatsnew/changelog.html#python-3-10-1-final) [错误修复](https://realpython.com/python-bugfix-version/)以及对文档和测试的更新。

Python 的下一个版本 Python 3.11 将于 2022 年 10 月发布。然而，核心开发团队已经为新特性和改进工作了好几个月。如果您有兴趣预览即将推出的产品，可以下载并安装最新的预发布版本。

即使 Python 3.11 的正式发布已经过去了好几个月，所谓的 [alpha 版本](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)已经可以供你使用了。12 月 8 日， [Python 3.11.0a3](https://www.python.org/downloads/release/python-3110a3/) ，第三个 alpha 版本公开。您可以尝试体验一下[改进的错误报告](https://realpython.com/python-news-july-2021/#python-311-gets-enhanced-error-reporting)和[更快的执行速度](https://github.com/faster-cpython)。

就其本质而言，alpha 版本是不稳定的，可能包含许多错误。因此，在生产或其他重要的脚本中，您永远不应该依赖 Python alpha 版本。但是，如果您很想尝试新的特性和改进，可以考虑尝试一下新的预发布版本。

## Python 3.6 的生命周期结束

在发布周期的另一边，你会发现 Python 3.6，它在 2021 年 12 月 23 日达到了它的[寿命终止](https://endoflife.date/python)日期。在 Python 3.6 中，你得到了许多流行的[新特性](https://docs.python.org/3/whatsnew/3.6.html)，包括如下:

*   [数字文字中的下划线](https://realpython.com/python-numbers/#integers)
*   [f 弦](https://realpython.com/python-f-strings/)
*   [变量注释](https://realpython.com/python-type-checking/#variable-annotations)
*   使用[保证元素排序的更高效的字典](https://realpython.com/python-ordereddict/)
*   对 Python 的[异步特性](https://realpython.com/python-async-features/)的许多改进

然而，Python 3.6 现在已经超过五年了。Python 3.6 不会有任何新的维护版本，即使发现了严重的安全问题。虽然您的 Python 3.6 仍将继续工作，但您应该确保升级任何仍运行 Python 3.6 或更旧版本的重要系统。

由于 f 字符串、类型注释和异步等特性的流行，Python 3.6 多年来一直是许多库支持的最低版本。然而，像 [NumPy](https://pypi.org/project/numpy/1.22.0/) 和 [Django](https://pypi.org/project/Django/4.0/) 这样的流行库已经将 [Python 3.8](https://realpython.com/python38-new-features/) 列为他们最新版本的最低要求。获得对依赖项的适当支持是保持 Python 版本合理更新的另一个原因。

PyPI Stats 是一个了解不同 Python 包和 Python 版本使用的好网站。您可以在他们的[网页](https://pypistats.org/packages/__all__)上或通过 [`pypistats`](https://github.com/hugovk/pypistats) 命令行工具访问统计数据:

```py
$ pypistats python_minor __all__ -m 2021-12
| category | percent |      downloads |
| :------- | ------: | -------------: |
| 3.7      |  41.15% |  5,002,371,969 |
| 3.8      |  21.56% |  2,621,179,853 |
| 3.6      |  14.98% |  1,821,281,479 |
| 3.9      |   7.93% |    964,495,785 |
| 2.7      |   7.13% |    866,934,431 |
| null     |   4.03% |    490,065,646 |
| 3.10     |   1.46% |    177,027,749 |
| 3.5      |   1.23% |    149,063,639 |
| 3.4      |   0.51% |     62,069,015 |
| 3.11     |   0.00% |        536,407 |
| 3.3      |   0.00% |         24,849 |
| 3.2      |   0.00% |          4,131 |
| 2.6      |   0.00% |          3,683 |
| 2.8      |   0.00% |            123 |
| 3.1      |   0.00% |             73 |
| 4.11     |   0.00% |             26 |
| Total    |         | 12,155,058,858 |

Date range: 2021-12-01 - 2021-12-31
```

这份概览显示，2021 年 12 月 PyPI 上大约 15%的下载是针对 Python 3.6 的。总的来说，几乎四分之一的下载是针对现在已经过时的 Python 版本的，包括 Python 2。参见[It ' s Time to Stop use Python 3.6](https://pythonspeed.com/articles/stop-using-python-3.6/)作者 [Itamar Turner-Trauring](https://realpython.com/podcasts/rpp/24/) 关于让您的 Python 保持最新的更深入讨论。

## 代码的出现

[Advent of Code](https://realpython.com/python-advent-of-code/) 是一个在线降临节日历，从 12 月 1 日到 25 日，每天都会发布新的编程谜题。上个月，来自世界各地的程序员连续第七年聚集在一起进行友好比赛。[Code 2021](https://adventofcode.com/2021/)的问世成为迄今为止最受欢迎的版本，有超过 20 万的[参与者。](https://twitter.com/ericwastl/status/1470928270854078465)

每年，一个精彩的[故事](https://www.webtoons.com/en/challenge/advent-of-code-2021/intro/viewer?title_no=713188&episode_no=1)伴随着谜题。今年，你需要通过找回掉在海底的圣诞老人的雪橇钥匙来拯救圣诞节。在你寻找钥匙的过程中，你要和一只[巨型乌贼](https://adventofcode.com/2021/day/4)玩宾果游戏，惊叹[灯笼鱼](https://adventofcode.com/2021/day/6)的产卵能力，帮助一群[片脚类动物](https://adventofcode.com/2021/day/23)找到它们的家洞穴，并解决许多其他令人兴奋的谜题。

即使你错过了每个谜题解开时所有令人兴奋的事情，你仍然可以回去解决所有这些谜题，以及追溯到 2015 年的早期版本。[杰西·范·艾尔特伦](https://www.linkedin.com/in/jessevanelteren/)做了一个[分析](https://jvanelteren.github.io/blog/2022/01/02/analysis_aoc_stats.html)将《代码 2021》的出现与过去几年进行比较。

类似地，[耶鲁安·海曼斯](https://twitter.com/jeroenheijmans)做了一个[非官方调查](https://jeroenheijmans.github.io/advent-of-code-surveys/)代码参与者的降临。调查表明，Python 是解决降临代码难题最受欢迎的语言，超过 40%的受访者使用它。

在 *Real Python* ，我们为我们的[社区](https://realpython.com/community/)举办了一场[私人排行榜](https://realpython.com/python-advent-of-code/#how-to-participate-in-advent-of-code)比赛，非常有趣。我们要感谢《降临代码》的创作者 [Eric Wastl](https://realpython.com/interview-eric-wastl/) ，感谢他多年来为该项目付出的所有努力，并祝贺他在 2021 年 12 月达到了 [500，000 总用户](https://twitter.com/ericwastl/status/1469040666118438919)和 [10，000，000 总星级](https://twitter.com/ericwastl/status/1474765035071315968)的里程碑。

如果你想自己尝试这些谜题，看看我们的指南。[竞赛程序员手册](https://github.com/pllk/cphb/)是解决这类难题的综合资源，包含许多有用技术和算法的信息。

[*Remove ads*](/account/join/)

## Python 的下一步是什么？

2021 年对于 Python 来说是很棒的一年，Python 3.10 的发布是其中的亮点之一。在*真实的 Python* ，我们期待着 2022 年和 Python 3.11 的进一步发展，由新的指导委员会指导，由核心开发团队和来自世界各地的其他志愿者实施。

十二月份你最喜欢的 Python 新闻是什么？我们错过了什么值得注意的吗？请在评论中告诉我们，我们可能会在下个月的 Python 新闻综述中介绍您。

快乐的蟒蛇！**