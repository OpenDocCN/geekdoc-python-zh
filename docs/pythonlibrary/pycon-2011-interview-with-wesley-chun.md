# PyCon 2011:专访卫斯理·陈

> 原文：<https://www.blog.pythonlibrary.org/2011/02/21/pycon-2011-interview-with-wesley-chun/>

随着 PyCon 的临近，blogger 社区受邀采访即将参加活动的演讲者。我选择了 Wesley Chun，他是《核心 Python 编程》的作者，也是《与 Django 一起进行 Python Web 开发》[的合著者。在这次采访中，我问了 Wesley 关于他的演讲,](http://amzn.to/hai9UI)[在 Google App Engine 上运行 Django Apps](http://us.pycon.org/2011/schedule/presentations/237/)以及 PyCon 的总体情况。让我们看看他有什么要说的:

你希望与会者从这次演讲中学到什么？

我希望所有与会者带着更大的乐观情绪离开这个演讲，他们可以带着他们的 Django 应用程序，在 Google App Engine 上进行很少或没有修改的情况下运行它们，利用他们需要的可伸缩性，这是靠自己很难实现的。

这个演讲的一部分是伪营销，以提高 Django-nonrel 的知名度，这是如何让 Django 应用程序在 App Engine 上运行的基础。自从 App Engine 在 2008 年首次亮相以来的几年里，已经有几个工具，称为助手和补丁，来帮助你在 App Engine 上运行 Django 应用程序。不幸的是，这些旧系统要求你修改应用程序的数据模型，以便让它们在 App Engine 上运行。Django-nonrel 则不是这样，当用户希望在任何 NoSQL 或非关系数据库上运行 Django 应用程序时，它应该成为用户应该使用的主要工具。

除了 Django-nonrel，开发人员还需要相应的 NoSQL 适配器代码，djangoappengine(用于 Google App Engine 的数据存储)，Django-mongodb-engine(用于 mongodb)。他们(和其他人)正在开发其他 NoSQL 数据库的适配器，但比这更令人兴奋的是 NoSQL 的加入！

是什么让你决定谈论这个话题的？

我想做这个演讲有很多原因...我之前已经提到了意识。另一件事是，人们如此习惯于助手和补丁，以至于他们没有意识到还有更好的工具。

另一个重要原因是供应商锁定的概念，这种现象是指您无法轻松地将应用程序和/或数据迁移到不同的平台，因为您被当前的供应商“锁定”了。人们抱怨你不能在其他地方运行 App Engine 应用程序，但这不是真的。除了谷歌的原始版本，你还可以选择不同的后端...其中两个后端是 AppScale 和 TyphoonAE。类似地，如果你创建了一个 Django 应用程序，并通过传统的主机运行它，Django-nonrel 可以帮助你把它移植到 App Engine，只需要很少的移植。类似地，如果你写了一个 Django 应用程序，并使用 Django-nonrel 在 App Engine 上运行它，把它转移到传统的主机上应该不需要太多的工作。

**3)在 Google App Engine 上使用 Django 有什么利弊？**

最明显的优点是可扩展性。这是一件既困难又昂贵的事情。为什么不利用谷歌的聪明人，他们在核心基础设施中构建了可伸缩性来帮助他们...，嗯，谷歌！有了 Django-nonrel，如果谷歌不适合你，你可以带着你的应用去别的地方运行！这里没有供应商锁定。

一个缺点是，如果你习惯于传统的关系数据库模型，App Engine 的数据存储仍然难以理解。您还不能完全执行原始 SQL 或连接。Google 确实给了你一个被称为 GQL 的简化的 SQL 语法，但是它并不是完整的图片。此外，为了换取它的好处，你必须放弃对你的应用程序的一些控制，让谷歌托管它。您不能使用任何 C/C++扩展，也不能访问您的日志或其他原始文件。

今年在 PyCon 上你最期待什么？

我期待着与我在过去十年参加 Python 和开源会议时遇到的那些熟悉而友好的面孔建立联系。因为我们都在不同的地理位置，这是你可以指望见到一年没见的人并叙叙旧的唯一时间，无论是在展厅还是在有趣的走廊谈话中。

我也很高兴学习 Python 世界中的新事物。它似乎每年都在增长，所以很难跟上社区的最新发展。我也期待着重复我去年的 Python 3 演讲，部分原因是它每年都变得越来越重要，作为我演讲研究的一部分，我将找出哪些项目已经转移到 Python 3。

**5)往年 PyCon 你最喜欢的部分是什么？**

Python 生态圈最棒的部分是有一个伟大的编程语言作为后盾，但是下一个最好的部分是人...Python 社区本身。PyCon 是与社区互动的最佳场所。会议是非凡的，因为会谈，能够达到所有技能水平(从初级到高级)，持续两天的高超教程，引人注目的会议会谈，令人惊叹的闪电会谈和开放空间会议，当然，还有走廊对话，更不用说来自这种会议的疯狂迷因和黑客，如 oh war-[http://pycon.ohwar.com](http://pycon.ohwar.com)。

想想看:你可能会在今年的 PyCon 上见到你最喜欢的 Python 作者，同时了解 Django 和 Google App Engine。如果这还不能让你满意，Wesley 也在做一个关于 Python 3 的演讲。你还在等什么？登陆 [PyCon 网站](http://us.pycon.org/2011/home/)并报名吧！

*注:本文交叉发布到 [PyCon 博客](http://us.pycon.org/2011/blog/2011/02/14/pycon-2011-interview-wesley-chun/)*