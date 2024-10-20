# py OWA-2009 年 10 月会议总结

> 原文：<https://www.blog.pythonlibrary.org/2009/10/02/pyowa-october-2009-meeting-wrap-up/>

昨晚我们举行了十月派瓦会议，我认为这很有趣。我们请到了一家名为 Priority5 的当地技术公司的高管，他告诉我们他是如何开始使用 Python 的，以及他们如何在他目前的雇主那里使用 Python。我们还讨论了 Optparse、ConfigParser 和 ConfigObj。

这位 [Priority5](http://www.priority5.com/) 的代表使用了一个看起来像 Touchsmart 的惠普触摸屏(类似于[这个](http://nexus404.com/Blog/2008/06/26/hp-touchsmart-iq504-22inch-touchscreen-core-2-duo-pc-now-available-touchsmart-iq504-pc/))来演示他公司最酷的项目之一。不幸的是，这是一个用于国防的产品，所以我们不能记录演示。总之，他说这个产品(它看起来像一个地球仪，你可以用它缩小到街道的高度)是用 C++做底层，用 Python 做高层。

GUI 是用 [pyQT](http://www.riverbankcomputing.co.uk/news) 创建的，而互联网连接是用 [Twisted](http://twistedmatrix.com/trac/) 完成的。他们使用 [SqlAlchemy](http://www.sqlalchemy.org) 作为他们的 ORM。他告诉我们他们如何使用 pySerial 对其中一个触摸屏上的串行端口进行逆向工程。他还提到了他的组织如何通过贡献补丁来帮助 pyQT & Twisted，以及他们如何帮助开发 [Py++](http://language-binding.net/pyplusplus/pyplusplus.html) 。他们还使用 [SCons](http://www.scons.org/) 进行构建，使用 [Trac](http://trac.edgewall.org/) 进行 bug 跟踪(最后一个可能是他在以前的组织中使用的)...在那一点上他不是很清楚)。

看着他在华盛顿 DC 的街道上拉近镜头，看着摄像机的实时画面，真是有趣。当演讲者开始阐述该系统如何在总统就职典礼或其他重要活动中使用，或者只是听听他们的程序如何模拟工作时，演讲也非常有趣。

接下来的演示是关于 Python 的标准库加上第三方包，即 [Optparse](http://docs.python.org/library/optparse.html) 、 [ConfigParser](http://docs.python.org/library/configparser.html) 和 [ConfigObj](http://www.voidspace.org.uk/python/configobj.html) 。虽然这次谈话本身很好，但与第一次相比就相形见绌了。我们应该颠倒会谈的顺序。无论如何，他使用了爱荷华州的地理信息系统库来说明这个例子。

虽然我们只有 6 个人参加这个会议，但我想我们都有所收获。希望下次会有更多。说到这里，下一次会议将于 2009 年 11 月 2 日星期一下午 7-9 点在马歇尔郡治安官办公室举行。详情请访问 Pyowa 官方网站了解！