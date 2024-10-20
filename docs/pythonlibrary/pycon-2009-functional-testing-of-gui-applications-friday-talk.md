# PyCon 2009-GUI 应用程序的功能测试(周五演讲)

> 原文：<https://www.blog.pythonlibrary.org/2009/03/28/pycon-2009-functional-testing-of-gui-applications-friday-talk/>

上周五(3 月 27 日)去 PyCon 的最后一个讲座，我去了 Michael Foord 的、[、*桌面应用*、](http://us.pycon.org/2009/conference/schedule/event/34/)的功能测试。他在演示中使用了他的书 *[中的 IronPython 示例。他的主要话题是关于测试图形用户界面及其固有的问题。Foord 给出了很多测试 GUI 的好理由，比如确保新代码不会破坏功能，它在重构时非常有用，单个测试充当应用程序的规范，它让你知道一个功能何时完成，测试可以推动开发。](http://www.ironpythoninaction.com/)*

使用 GUI 框架的一个大问题是，当你测试它们时，你不能阻塞主循环。这可能是一种痛苦。Foord 的解决方案是使用工具包的定时器对象来拉进并运行测试。他还提到在您的应用程序中创建钩子，允许您自己检测它。他的[幻灯片](http://www.voidspace.org.uk/python/articles/testing/)列出了可以帮助 GUI 测试的各种包，比如 WATSUP 和 guitest，以及其他几个包。我不确定他的网站上是否有完整的列表，但是给他写封短信，他可能会给你。

最后，我发现这是我周五参加的内容更丰富的讲座之一。它给了我一些如何在我自己的应用程序中实现测试的想法。希望这些会有结果。

![](img/7569ea9bbf93bba5131ec3d15c9bf4a0.png)