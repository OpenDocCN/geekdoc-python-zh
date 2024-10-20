# python vs Julia——比较

> 原文：<https://www.askpython.com/python/python-vs-julia>

在本文中，我们将比较 Python 和 Julia。由麻省理工学院开发的 Julia 编程已经成为从长远来看可能取代 Python 的顶级语言之一。尽管 Julia 的开发者坚信 Julia 和 Python 是携手并进的，但我们试图探究为什么 Julia 可能成为 Python 的潜在替代品。我们将探讨它们的特点和缺点。

## python vs Julia–快速概述

Python 和 Julia 都是开源语言，它们是动态类型的，并且有一个与我们的自然语言高度相似的语法。目前稳定版的 Julia 是 1.5.4，python 是 3.9.2。

|  | 计算机编程语言 | 朱莉娅 |
| 开发人 | Python 软件基金会 | 用它 |
| /
编译解释 | 解释 | 编辑 |
| 速度 | 慢的 | 快的 |
| 范例 | 面向对象、流行和功能 | 功能的 |
| 类型系统 | 动态类型化 | 动态类型化 |
| 图书馆支持 | 丰富成熟的图书馆支持 | 积极发展的图书馆 |
| 使用
语言的公司 | 谷歌、脸书、Spotify、Quora、
网飞、Reddit 等。 | 亚马逊，苹果，迪斯尼，
福特，谷歌，美国宇航局等。 |
| 发展 | 成熟(3.9.2 版) | 积极开发(v1.5.4) |

Comparison table of Python and Julia

### 速度

使这篇文章相关的一个非常重要的因素是 Julia 的速度。以下是展示 Julia 速度的基准:

![](img/be0bd4d6fdd9f1f863f9f58d58bd7f8a.png)

这个速度的主要原因是基于 LLVM 的 Julia JIT 编译器。编译器进行了许多高级抽象和优化，以使 Julia 这么快。Julia 解决了两个程序问题，而且大部分的 Julia 和它的库都是用 Julia 本身写的。另一方面，Python 是解释型的，速度较慢，这使得它不适合大型计算。

*python 中有一些库，比如 Numba 和 Jax，它们允许*使用 *JIT 编译器进行快速计算，但这些都是非常特定于应用的*。

### 范式

Julia 支持函数式编程，对类型层次结构有现成的支持。Python 允许我们更加灵活地解决我们的程序。Python 支持函数式、面向对象和面向过程的编程。

### 代码可重用性

Julia 最重要的因素之一是代码的可重用性。代码重用也是面向对象编程的主要特征之一，但事实证明 Julia 的类型系统和多重调度对代码重用更有效。

### 图书馆支持

Python 有巨大的库支持。你想做的每件事都可以在图书馆里找到。从制造不和谐的机器人到近似的样条插值，一切都是可用的和开源的。Python 已经有 30 多年的历史了，所以这些库大部分都很成熟。python 中很少流行的库/框架有 [SciPy](https://www.askpython.com/python-modules/python-scipy) 、 [Django](https://www.askpython.com/django/django-forms) 、TensorFlow、 [Pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 等。

朱莉娅还提供巨大的图书馆支持，主要倾向于科学研究。这些库正在被快速开发，并且每天都有新的库被开发出来。他们中的大多数还没有达到 1.0 版本，所以这意味着你可能会遇到一些错误。但是这些图书馆做他们最擅长的事情，有些对 Julia 来说是非常独特的。一些非常流行的 Julia 库有 Flux、Pluto、微分方程、JuMP 等。

### 社区

Python 是非常流行的语言(2021 年前 3)。它拥有大量的社区支持，来自各种背景的人们想出各种方法来帮助和维护社区。Python 编程语言国际社区每年都会召开几次会议(PyCons)。PyCons 托管在世界各地，大部分是由当地 Python 社区的志愿者组织的。在这样的社区活动中，你可以看到从软件开发人员到研究人员到学生的各种人。

朱莉娅也是一个非常包容的社区，人们来自各种背景。Julia 仍然在爬受欢迎的阶梯，所以你不能指望 python 有这么大的社区，但肯定是一个支持者。

### 朱莉娅对其他语言的支持

Julia 允许用户调用用 C、python、R 等语言编写的代码。朱莉娅会直接打电话给你。这意味着你不需要把所有的代码都转换成 Julia，而是通过使用 Julia 库来调用它们。

### 结论

Python 和 Julia 各有优缺点。朱莉娅还很年轻，潜力巨大。相比之下，Python 是一种疯狂的流行语言，如果你遇到任何困难，你一定会找到以前解决过这个问题的人！选择永远是你的！如果你喜欢探索新的编程语言，Julia 可以成为你探索的对象。