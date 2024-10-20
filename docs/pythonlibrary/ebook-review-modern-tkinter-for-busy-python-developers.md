# 电子书评论:忙碌的 Python 开发者的现代 Tkinter

> 原文：<https://www.blog.pythonlibrary.org/2012/05/12/ebook-review-modern-tkinter-for-busy-python-developers/>

我最近从[亚马逊](http://www.amazon.com/gp/product/B0071QDNLO/ref=as_li_ss_tl?ie=UTF8&tag=thmovsthpy-20&linkCode=as2&camp=1789&creative=390957&creativeASIN=B0071QDNLO)购买了马克·罗斯曼为忙碌的 Python 开发者编写的 **Modern Tkinter，昨天刚刚完成。我认为它很新，但我现在找不到它的发布日期。不管怎样，我们继续复习吧！**

### 快速回顾

*   **为什么我买了这本书:**我买这本书是因为我一直打算深入研究其他 Python GUI 工具包，而且自从约翰·格雷森的 [Python 和 Tkinter 编程](http://www.amazon.com/gp/product/1884777813/ref=as_li_ss_tl?ie=UTF8&tag=thmovsthpy-20&linkCode=as2&camp=1789&creative=390957&creativeASIN=1884777813)之后，我就再也没有看过新的 Tkinter 书了
*   我完成它的原因:它有一个非常好的写作风格，尽管部件章节开始变得拖沓
*   **我想把它给:**任何想让他们的 Tkinter 应用程序看起来更本地或者想了解一点 Tkinter 新主题系统的人。

### 图书格式

据我所知，这是另一个亚马逊 Kindle 或其他任何接受 mobi 的 mobi 书。根据亚马逊的说法，它可以打印出大约 147 页，大小不到一兆。

### 书籍内容

第 1 章和第 2 章是介绍性的，给出了项目的一些背景信息。第 3 章只是关于安装 Tkinter，我不是很懂。然而，这本书对 Tkinter 的新 ttk 部分以及它如何只在 Python 2.7 和 3.x 中可用进行了大做文章，需要注意的是，在提到 2.7 一次后，作者的行为就像 ttk 只在 3.x 中可用一样，这是混乱和错误的。对于 2.7 之前的版本，你实际上可以[下载](http://pypi.python.org/pypi/pyttk)它，但默认情况下它是 2.7 的(反正在 Windows 上)。

不管怎样，第 4 章和第 5 章介绍了 Tkinter 的概念。第 6 章和第 8 章是与第 7 章相关的小部件，讨论网格几何管理器。第 9 章介绍菜单；10 浏览窗口和对话框；11 是组织性的(笔记本、窗格式窗户等)；12 是关于字体、颜色和图像。第 13-15 章涵盖了大部件:分别是画布、文本和树部件。最后一章，也就是第 16 章，讲述了应用程序的主题化。

### 回顾

正如我已经提到的，这本书是以一种引人入胜的方式写的。我听说你可以让 Tkinter 看起来更好，但是 Tk 8.5+中的新东西(包含在 Python 2.7+中)让让你的应用程序看起来更好变得容易了。ttk 小部件的主题化方式使它们看起来是本地的或接近本地的，由于新的主题化功能，听起来你实际上可以很容易地对其进行主题化。

我注意到的不好的地方是在文本中有几个 PYTHONTODO 语句的例子。我想作者是想在这些地方多写些东西，只是忘记删除了。第 6 章中有一部分是关于组合框的，听起来你可以将数据和列表中的项目联系起来。作者声明他将在后面的 Listbox 部分中讨论它，但结果是 Tkinter 根本没有提供这样做的方法。你必须想出自己的方法，尽管作者描述了一些想法，但他并没有展示它们。我认为它可能是类似 wxPython 内置的做这种事情的方式，我在这里写了[和](https://www.blog.pythonlibrary.org/2010/12/16/wxpython-storing-object-in-combobox-or-listbox-widgets/)，但事实并非如此。哦好吧。

你应该设置 Tkinter 设置的一些方法不清楚，写得很奇怪，通常是这样的: *"step？额？”*。我不太确定问号是不是必须的，但我猜不是。通常当提到设置时，没有例子来说明它们是如何工作的。在阅读了所有这些内容并看到了一些配置 Tkinter 小部件或实例化它们的奇怪方式后，我仍然认为这非常不直观和不一致。

另一方面，我认为这本书在很短的时间内包含了很多有用的信息。我感到很受鼓舞，再次尝试 Tkinter，看看它能做什么，也因为我想写一些关于它的文章。我的裁决？如果你想学习关于 ttk 的新东西，这本书很有意义。据我所知，除了官方文档之外，市场上没有任何东西包含这方面的信息。请注意，很少有完整的例子，主题一章实际上从来没有展示如何创建自己的主题，它只是给你足够的信息来做这件事。因此，如果你是一名新的 Python GUI 开发人员，并且想使用 Tkinter，那么这本书可能适合你。

**更新(2012 年 5 月 31 日):这篇文章被重新格式化并转载于[I-Programmer](http://www.i-programmer.info/bookreviews/62-python/4289-modern-tkinter-for-busy-python-developers.html)**