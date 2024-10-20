# 本周 PyDev:Gaetan Delannay

> 原文：<https://www.blog.pythonlibrary.org/2020/04/27/pydev-of-the-week-gaetan-delannay/>

本周我们欢迎 Gaetan Delannay 成为我们的本周 PyDev！盖坦是 Appy、Python 网络框架和 T2 企业家的创造者。

让我们花些时间更好地了解盖坦吧！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

对我来说，谈论我自己是很不寻常的，但是让我们试一试。大约 11 年前，我创办了一家名为 GeezTeem 的个人公司。我 90%的时间都在用 Python 编程，为比利时的公众和非营利部门开发和维护大约十个软件产品。作为一个偏执狂，所有这些东西都是用 [Appy](http://appyframework.org) 制作的，这是我自己的(GPL)框架，用于用 Python 构建 web 应用程序。

被开源哲学所说服，我的大部分代码都是开源的，在 GPL 许可下发布。在用自己的翅膀飞行之前，我已经在各种公司、公共实体或研究中心工作了 10 年，作为开发人员、测试人员、研究人员、质量工程师或软件架构师，在 3D 图形、需求工程、质子治疗控制软件或管理信息系统等各种领域学习和试验软件工程的所有方面。

起点是比利时那慕尔大学的软件工程硕士学位。除了编码，写作(法语)，打网球和作曲是我最喜欢的活动。在以 [Null ID](http://nullid.org/controleK.html) 的名字发布了我的第一张专辑后，有人告诉我，我可能受到了精神影响。


**你为什么开始使用 Python？**

20 多年前，在我的第一份工作中，一位新任命的质量经理(我的老板)向我解释了他对软件质量的创新方法:用 Python 开发工具和脚本，以促进软件开发项目。这就是我学习 Python 的方法。我开始尝试这种语言，从小脚本开始，最终创建了一个复杂的工具，生成存根和框架，用于互连用 C++编写的软件前端和用 Ada 编写的后端。

你还知道哪些编程语言，你最喜欢哪一种？

我有 C，C++，Ada，Java，Python 和 web 技术(HTML / CSS / Javascript)的经验。Python 显然是我的最爱。对我来说，它是最优雅、简洁和强大的编程语言。它允许我独自构建大型、复杂但实用的面向对象软件产品。

你现在在做什么项目？

作为 GeezTeem，我所有的项目都是基于网络的管理系统。几乎有 40 个公共行政部门(包括一个议会和两个政府)使用 HubSessions 来准备、讨论和发布他们的官方决定。https://be-coag.be/是血友病患者的在线工具。PlanetKids 允许父母为他们的孩子注册参加由比利时两个城市的所有协会提议的夏季活动。Plat-com 是一个合作平台，由几十个从事儿童保育工作的协会使用。

哪些 Python 库是你最喜欢的(核心或第三方)？

多年来，我一直被 Zope 对象数据库 ZODB T1 所吸引，它对开发者来说是完全透明的。pyUNO，允许在服务器模式下控制 LibreOffice，也是必须的。在核心 Python 库中，我最近发现了 pathlib，它极大地提高了基于路径的操作的可读性。

Appy 项目是如何产生的？

15 年前，我从事一个大型 Java 项目，部署了几台 J2EE 链式服务器。我被要求使用 XSL-FO 转换开发 PDF 导出。我花了将近 10 个人工日来制作第一个！一场噩梦。我开始考虑使用 Python 和 LibreOffice 以更有效的方式完成这些任务。POD (Python OpenDocument)，Appy 最著名的部分诞生了，最初是在火车上手写在纸上的。Appy.pod 让我将 10 个人工日减少到 10 个...分钟。后来，Appy 成长起来，成为构建 web 应用程序的完整框架。最初基于 Zope 和 Plone，当前版本(0.9)仍然对 Zope 有依赖性。Appy 1.0 正在积极开发中，将在几周内准备就绪。它已经成为一个独立的、全功能的 web 框架(重新)用 Python 3 编写，使用 ZODB 作为数据库(独立于 Zope 打包)。

**您在这个产品包中克服了哪些挑战？**

构建应用服务器的想法是一个巨大而疯狂的任务。我花了几年时间来开发它，主要是在晚上和周末，但现在，两年以来，它的部分资金来自我的客户。Appy 1 是我关于软件开发的知识和经验的总结和综合。我很高兴能在不久的将来出版它！

你还有什么想说的吗？

Appy 是在 GPL 下发布的，因此是一个污染性的开源许可证。我已经收到了几个在 LGPL 发布 Appy.pod 或者获得商业许可的请求。这两种解决方案都不符合我的开源愿景，但我不想阻止人们使用 Appy.pod。最后，我找到了解决这一困境的方法，并开发了一个商业版的 Appy.pod...功能受限。如果你阅读 Appy 代码，你会发现一个叫做 CommercialError 的异常，它在 Appy 的商业版本中被调用，每次用户试图使用一些高级功能时:

```py
class CommercialError(AppyError):
    '''Raised when some functionality is called from the commercial 
       version but is available only in the free, open source version.'''
    MSG = 'This feature is not available in the commercial version. It is only available in the free, open source (GPL) version of Appy.'
    def __init__(self): AppyError.__init__(self, self.MSG)
```

在向第一家提出要求的公司提出这一建议后，我再也没有收到他们的任何消息。他们可能认为我疯了。但是从那以后，最近有两家公司，一家在法国，一家在英国，买了我的商业许可证。

盖坦，谢谢你接受采访！