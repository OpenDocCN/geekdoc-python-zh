# 组装 wxPython 应用程序

> 原文：<https://www.blog.pythonlibrary.org/2008/06/09/putting-together-a-wxpython-application/>

大约一周前，我写道，我正在开发一个示例应用程序，我将在这里发布。当我在做的时候，我意识到我需要找到一种简单、有条理和通用的方法来分解它。因此，我决定就我的应用程序创建一系列“如何做”的文章，并将它们发布在这里。然后，我将发布另一篇文章，将所有的片段放在一起。

该应用程序包括以下部分:

*   wx。箱式筛分机
*   wx。对话类
*   wx。菜单，wx。StatusBar 和 wx。工具栏
*   wx。AboutBox

*注意:我已经有一个 [BoxSizer 教程](https://www.blog.pythonlibrary.org/?p=22)完成。*

我认为这涵盖了 wxPython 的主要部分。我还将使用标准 Python 2.5 库的 email、urllib 和 smtplib 模块以及一些特定于 win32 的模块。我想你会发现这套文章很有教育意义。请务必让我知道你的想法。