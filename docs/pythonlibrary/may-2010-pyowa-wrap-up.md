# 2010 年 5 月 Pyowa 总结

> 原文：<https://www.blog.pythonlibrary.org/2010/05/07/may-2010-pyowa-wrap-up/>

昨晚，我们举行了 2010 年 5 月的 Pyowa 会议。这是爱荷华州唯一的 Python 用户组，我们欢迎任何用 Python 编程的人(或对学习 Python 感兴趣的人)加入我们的组。在这次会议上，我们做了三个很好的报告。第一个是由吉姆，他的主题是网络抓取。他使用 [Mechanize](http://wwwsearch.sourceforge.net/mechanize/) 和 [lxml](http://codespeak.net/lxml/) 的组合来抓取埃姆斯市的网站，以便在他自己的一个网站上存档。

Mechanize 允许 Jim 模拟浏览器并浏览网站。它可以填写表格，用你提供的凭证登录，等等。然后，他使用 lxml 解析他想要的页面，如果他需要下载什么，他只需结合 wget 使用 *os.system* 。也提到了[美汤](http://www.crummy.com/software/BeautifulSoup/)库，但是 Jim 没有使用它。我们的另一位成员说，他们的组织确实使用了一段时间美丽的汤，并对结果感到满意。

我们接下来的两个演示是由一个叫凯文的人做的。他谈到了 [Mercurial](http://mercurial.selenic.com/) 分布式版本控制系统和 [Trac](http://trac.edgewall.org/) ，一个基于网络的问题跟踪器。Kevin 向我们展示了如何建立一个 Mercurial 存储库，添加文件，分支，更新，合并等等。他使用 [virtualenv](http://pypi.python.org/pypi/virtualenv) 完成了所有这些，这是一种分离项目的简便方法。完成 Mercurial 讲座后，Kevin 向我们展示了如何使用他的 Mercurial 存储库设置 Trac、添加票证、在 Mercurial 中提交票证修复程序，以及 Trac 中包含的各种管理工具。Kevin 还强调了一些他喜欢的 Trac 和 Mercurial 插件。

如果你愿意参加我们的下一次会议，它将在同一个地点举行，6 月 3 日，也就是星期四，在爱荷华州的埃姆斯公共图书馆。如果您想分享您使用 Python 或其众多项目中的一个的经验，那将非常好！请给我发电子邮件，地址是 python library . org 的 mike，这样我们可以给你安排时间。观看我们的[网站](http://www.pyowa.org)获取最新信息。