# Python 3 简介

> 原文：<https://realpython.com/python-introduction/>

Python 是一种高级解释脚本语言，由荷兰国家数学和计算机科学研究所的吉多·范·罗苏姆在 20 世纪 80 年代后期开发。最初的版本于 1991 年在 alt.sources [新闻组](https://en.wikipedia.org/wiki/Usenet)上发布，1.0 版本于 1994 年发布。

Python 2.0 发布于 2000 年，2.x 版本是直到 2008 年 12 月的主流版本。当时，开发团队决定发布 3.0 版本，其中包含一些相对较小但重要的更改，这些更改与 2.x 版本不向后兼容。Python 2 和 3 非常相似，Python 3 的一些特性被反向移植到了 Python 2。但总的来说，他们仍然不太兼容。

Python 2 和 3 都继续得到维护和开发，并定期发布更新。在撰写本文时，最新的可用版本是 2.7.15 和 3.6.5。然而，官方已经为 Python 2 确定了 2020 年 1 月 1 日的[生命周期结束日期，在此之后，将不再保留该日期。如果你是 Python 的新手，建议你关注 Python 3，就像本教程一样。](https://pythonclock.org)

Python 仍然由研究所的核心开发团队维护，Guido 仍然负责，他被 Python 社区授予了 BDFL(仁慈的终身独裁者)的称号。顺便说一句，Python 这个名字不是来源于蛇，而是来源于英国喜剧团[蒙蒂·派森的飞行马戏团](https://en.wikipedia.org/wiki/Monty_Python%27s_Flying_Circus)，圭多是这个马戏团的粉丝，大概现在仍然是。在 Python 文档中很容易找到对 Monty Python 草图和电影的引用。

**免费 PDF 下载:** [Python 3 备忘单](https://realpython.com/bonus/python-cheat-sheet-short/)

## 为什么选择 Python？

如果你要写程序，有几十种常用语言可供选择。为什么选择 Python？下面是一些使 Python 成为吸引人的选择的特性。

[*Remove ads*](/account/join/)

### Python 流行

Python 在过去几年里越来越受欢迎。2018 年[栈溢出开发者调查](https://insights.stackoverflow.com/survey/2018)将 Python 列为今年第七大最受欢迎和第一大最受欢迎的技术。[全球的世界级软件开发公司每天都在使用 Python。](https://realpython.com/world-class-companies-using-python/)

根据 Dice 的[研究，根据编程语言指数](https://insights.dice.com/2016/02/01/whats-hot-and-not-in-tech-skills/)的[受欢迎程度，Python 也是最热门的技能之一，是世界上最受欢迎的编程语言。](https://pypl.github.io/PYPL.html)

由于 Python 作为编程语言的流行和广泛使用，Python 开发者受到追捧，待遇优厚。如果你想深入了解 [Python 薪资统计和工作机会，你可以点击这里](https://dbader.org/blog/why-learn-python)。

### Python 被解读

许多语言都是编译的，这意味着你创建的源代码在运行之前需要被翻译成机器代码，即你的计算机处理器的语言。用解释语言编写的程序被直接传递给直接运行它们的解释器。

这使得开发周期更快，因为您只需输入代码并运行它，没有中间的编译步骤。

解释语言的一个潜在缺点是执行速度。编译成计算机处理器的本地语言的程序往往比解释程序运行得更快。对于一些计算特别密集的应用程序，如图形处理或密集的数字处理，这可能是限制性的。

然而，在实践中，对于大多数程序来说，执行速度的差异是以毫秒或者最多以秒来衡量的，并且对于人类用户来说是不可察觉的。对于大多数应用程序来说，用解释型语言编码的便利性通常是值得的。

**延伸阅读:**参见[这个维基百科页面](https://en.wikipedia.org/wiki/Interpreted_language)阅读更多关于解释语言和编译语言之间的差异。

### Python 是免费的

Python 解释器是在 OSI 批准的开源许可下开发的，可以自由安装、使用和分发，甚至用于商业目的。

该解释器的一个版本几乎适用于任何平台，包括各种风格的 Unix、Windows、macOS、智能手机和平板电脑，以及你可能听说过的任何其他平台。一个版本甚至为剩下的六个使用 OS/2 的人而存在。

### Python 是可移植的

因为 Python 代码是被解释的，而不是被编译成本机指令，所以为一个平台编写的代码可以在任何其他安装了 Python 解释器的平台上运行。(任何解释语言都是如此，不仅仅是 Python。)

### Python 很简单

就编程语言而言，Python 相对来说比较整洁，开发人员有意让它保持这样。

从语言中关键字或保留字的数量可以粗略估计语言的复杂性。这些词是编译器或解释器保留的特殊含义，因为它们指定了语言的特定内置功能。

Python 3 有 33 个[关键词](https://realpython.com/python-keywords/)，Python 2 有 31 个。相比之下，C++有 62 个，Java 有 53 个，Visual Basic 有超过 120 个，尽管后面这些例子可能会因实现或方言而有所不同。

Python 代码结构简单干净，易学易读。事实上，正如您将看到的，语言定义强制实施易于阅读的代码结构。

[*Remove ads*](/account/join/)

### 但事情没那么简单

尽管语法简单，Python 支持大多数在高级语言中预期的结构，包括复杂的动态数据类型、结构化和函数式编程，以及[面向对象编程](https://realpython.com/python3-object-oriented-programming/)。

此外，还有一个非常丰富的类库和函数库，它提供的功能远远超出了语言内置的功能，比如数据库操作或 GUI 编程。

Python 完成了许多编程语言没有完成的事情:语言本身设计简单，但是就你可以用它完成的事情而言，它是非常通用的。

## 结论

本节概述了 **Python** 编程语言，包括:

*   Python 的发展简史
*   您可能选择 Python 作为您的语言选择的一些原因

Python 是一个很好的选择，无论你是一个希望学习基础知识的初级程序员，还是一个设计大型应用程序的有经验的程序员，或者介于两者之间。Python 的基础很容易掌握，但是它的功能非常强大。

继续下一节，了解如何在您的计算机上获取和安装 Python。

[Introduction to Python](#)[Installing Python »](https://realpython.com/installing-python/)**