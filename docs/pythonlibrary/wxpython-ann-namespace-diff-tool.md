# wxPython: ANN:名称空间差异工具

> 原文：<https://www.blog.pythonlibrary.org/2011/11/10/wxpython-ann-namespace-diff-tool/>

昨晚，Andrea Gavana 向全世界发布了他的新命名空间比较工具(NDT)。我得到了他的许可，可以在这里为所有不关注 wxPython 邮件列表的人转载他的声明。我认为这听起来是一个非常酷的工具。你应该去看看，看看你有什么想法。下面是[公告](https://groups.google.com/forum/#!topic/wxpython-users/oK8dfnLQ7Rc) :

*描述
= = = = = = = = = =*

“名称空间差异工具”( NDT)是一个图形用户界面，可以用来发现库的不同版本之间的差异，甚至是同一库的不同迭代/子版本之间的差异。

该工具可用于确定缺少什么并且仍然需要实现
，或者新版本中有什么新内容，哪些项目没有
文档字符串等等。

罗宾·邓恩对最初想法的完整描述:

[http://SVN . wxwidgets . org/view VC/wx/wxPython/Phoenix/trunk/todo . txt？六](http://svn.wxwidgets.org/viewvc/wx/wxPython/Phoenix/trunk/TODO.txt?view=markup)...

:警告:由于 GUI 中的大多数小部件都是所有者绘制或自定义的，
在其他
平台上，界面本身很可能会看起来很混乱(Mac，我在跟你说话)。请尝试创建一个补丁来修复
在这个意义上的任何可能的问题。

:注意:请参考 TODOs 部分，了解仍然需要
实现的事项列表。

需求
============

为了运行无损检测，需要安装以下软件包:

- Python 2。x(其中 5<= X <= 7); - wxPython >= 2 . 8 . 10；
- SQLAlchemy > = 0.6.4。

更多关于如何使用它的详细说明，待办事项，我测试 NDT 的
库/包列表，截图和下载链接可以在这里找到:

[http://xoomer.virgilio.it/infinity77/main/NDT.html](http://xoomer.virgilio.it/infinity77/main/NDT.html)

如果你偶然发现了一个 bug(这很有可能)，请务必让我知道。但最重要的是，请努力为这个 bug 创建一个
补丁。

根据[线程](https://groups.google.com/forum/#!topic/wxpython-users/oK8dfnLQ7Rc)的说法，已经发现并修复了一些 bug。