# 第十六章 接下来学习什么？

**目录表**

*   图形软件
*   GUI 工具概括
*   探索更多内容
*   概括

如果你已经完全读完了这本书并且也实践着编写了很多程序，那么你一定已经能够非常熟练自如地使用 Python 了。你可能也已经编写了一些 Python 程序来尝试练习各种 Python 技能和特性。如果你还没有那样做的话，那么你一定要快点去实践。现在的问题是“接下来学习什么？”。

我会建议你先解决这样一个问题：创建你自己的命令行 地址簿 程序。在这个程序中，你可以添加、修改、删除和搜索你的联系人（朋友、家人和同事等等）以及它们的信息（诸如电子邮件地址和/或电话号码）。这些详细信息应该被保存下来以便以后提取。

思考一下我们到目前为止所学的各种东西的话，你会觉得这个问题其实相当简单。如果你仍然希望知道该从何处入手的话，那么这里也有一个提示。

**提示（其实你不应该阅读这个提示）** 创建一个类来表示一个人的信息。使用字典储存每个人的对象，把他们的名字作为键。使用 cPickle 模块永久地把这些对象储存在你的硬盘上。使用字典内建的方法添加、删除和修改人员信息。

一旦你完成了这个程序，你就可以说是一个 Python 程序员了。现在，请立即寄一封信给我感谢我为你提供了这本优秀的教材吧。是否告知我，如你所愿，但是我确实希望你能够告诉我。

这里有一些继续你的 Python 之路的方法：

使用 Python 的**GUI**库——你需要使用这些库来用 Python 语言创建你自己的图形程序。使用 GUI 库和它们的 Python 绑定，你可以创建你自己的 IrfanView、Kuickshow 软件或者任何别的类似的东西。绑定让你能够使用 Python 语言编写程序，而使用的库本身是用 C、C++或者别的语言编写的。

有许多可供选择的使用 Python 的 GUI：

*   **PyQt** 这是 Qt 工具包的 Python 绑定。Qt 工具包是构建 KDE 的基石。Qt，特别是配合 Qt Designer 和出色的 Qt 文档之后，它极其易用并且功能非常强大。你可以在 Linux 下免费使用它，但是如果你在 Windows 下使用它需要付费。使用 PyQt，你可以在 Linux/Unix 上开发免费的（GPL 约定的）软件，而开发具产权的软件则需要付费。一个很好的 PyQt 资源是[《使用 Python 语言的 GUI 编程：Qt 版》](http://www.opendocs.org/pyqt/)请查阅[官方主页](http://www.riverbankcomputing.co.uk/pyqt/index.php)以获取更多详情。

*   **PyGTK** 这是 GTK+工具包的 Python 绑定。GTK+工具包是构建 GNOME 的基石。GTK+在使用上有很多怪癖的地方，不过一旦你习惯了，你可以非常快速地开发 GUI 应用程序。Glade 图形界面设计器是必不可少的，而文档还有待改善。GTK+在 Linux 上工作得很好，而它的 Windows 接口还不完整。你可以使用 GTK+开发免费和具有产权的软件。请查阅[官方主页](http://www.pygtk.org/)以获取更多详情。

*   **wxPython** 这是 wxWidgets 工具包的 Python 绑定。wxPython 有与它相关的学习方法。它的可移植性极佳，可以在 Linux、Windows、Mac 甚至嵌入式平台上运行。有很多 wxPython 的 IDE，其中包括 GUI 设计器以及如[SPE（Santi's Python Editor）](http://spe.pycs.net)和[wxGlade](http://wxglade.sourceforge.net)那样的 GUI 开发器。你可以使用 wxPython 开发免费和具有产权的软件。请查阅[官方主页](http://www.wxpython.org/)以获取更多详情。

*   **TkInter** 这是现存最老的 GUI 工具包之一。如果你使用过 IDLE，它就是一个 TkInter 程序。在[PythonWare.org](http://www.pythonware.com/library/tkinter/introduction/index.htm)上的 TkInter 文档是十分透彻的。TkInter 具备可移植性，可以在 Linux/Unix 和 Windows 下工作。重要的是，TkInter 是标准 Python 发行版的一部分。

*   要获取更多选择，请参阅[Python.org 上的 GUI 编程 wiki 页](http://www.python.org/cgi-bin/moinmoin/GuiProgramming)。

不幸的是，并没有单一的标准 Python GUI 工具。我建议你根据你的情况在上述工具中选择一个。首要考虑的因素是你是否愿意为 GUI 工具付费。其次考虑的是你是想让你的程序运行在 Linux 下、Windows 下还是两者都要。第三个考虑因素根据你是 Linux 下的 KDE 用户还是 GNOME 用户而定。

未来的章节 我打算为本书编写一或两个关于 GUI 编程的章节。我可能会选择 wxPython 作为工具包。如果你想要表达你对这个主题的意见，请加入[byte-of-python 邮件列表](http://lists.ibiblio.org/mailman/listinfo/byte-of-python)。在这个邮件列表中，读者会与我讨论如何改进本书。

# 探索更多内容

*   **Python 标准库**是一个丰富的库，在大多数时候，你可以在这个库中找到你所需的东西。这被称为 Python 的“功能齐全”理念。我强烈建议你在开始开发大型 Python 程序之前浏览一下[Python 标准文档](http://docs.python.org)。

*   [Python.org](http://www.python.org/)——Python 编程语言的官方主页。你可以在上面找到 Python 语言和解释器的最新版本。另外还有各种邮件列表活跃地讨论 Python 的各方面内容。

*   **comp.lang.python**是讨论 Python 语言的世界性新闻组。你可以把你的疑惑和询问贴在这个新闻组上。可以使用[Google 群](http://groups.google.com/groups?hl=en&lr=&ie=UTF-8&group=comp.lang.python)在线访问这个新闻组，或加入作为新闻组镜像的[邮件列表](http://mail.python.org/mailman/listinfo/python-list)。

*   [《Python 实用大全》](http://aspn.activestate.com/ASPN/Python/Cookbook/)是一个极有价值的秘诀和技巧集合，它帮助你解决某些使用 Python 的问题。这是每个 Python 用户必读的一本书。

*   [《迷人的 Python》](http://gnosis.cx/publish/tech_index_cp.html)是 David Mertz 编著的一系列优秀的 Python 相关文章。

*   [《深入理解 Python》](http://www.diveintopython.org/)是给有经验的 Python 程序员的一本很优秀的书。如果你已经完整地阅读了本书，那么我强烈建议你接下来阅读《深入理解 Python》。它覆盖了包括 XML 处理、单元测试和功能性编程在内的广泛的主题。

*   [Jython](http://www.jython.org/)是用 Java 语言实现的 Python 解释器。这意味着你可以用 Python 语言编写程序而同时使用 Java 库！Jython 是一个稳定成熟的软件。如果你也是一个 Java 程序员，我强烈建议你尝试一下 Jython。

*   [IronPython](http://www.ironpython.com/)是用 C#语言实现的 Python 解释器，可以运行在.NET、Mono 和 DotGNU 平台上。这意味着你可以用 Python 语言编写程序而使用.NET 库以及其他由这三种平台提供的库！IronPython 还只是一个前期 alpha 测试软件，现在还只适合用来进行试验。Jim Hugunin，IronPython 的开发者，已经加入了微软公司，将在将来全力开发一个完整版本的 IronPython。

*   [Lython](http://www.caddr.com/code/lython/)是 Python 语言的 Lisp 前段。它类似于普通的 Lisp 语言，会被直接编译为 Python 字节码，这意味着它能与我们普通的 Python 代码协同工作。

*   另外还有很多很多的 Python 资源。其中比较有趣的有[Daily Python-URL!](http://www.pythonware.com/daily/)，它使你保持与 Python 的最新进展同步。另外还有[Vaults of Parnassus](http://www.vex.net/parnassus/)、[ONLamp.com Python DevCenter](http://www.onlamp.com/python/)、[dirtSimple.org](http://dirtsimple.org/)、[Python Notes](http://pythonnotes.blogspot.com/)等等。

# 概括

现在，我们已经来到了本书的末尾，但是就如那句名言，这只是 开始的结束 ！你现在是一个满怀渴望的 Python 用户，毫无疑问你准备用 Python 解决许多问题。你可以使你的计算机自动地完成许多先前无法想象的工作或者编写你自己的游戏，以及更多别的什么东西。所以，请出发吧！