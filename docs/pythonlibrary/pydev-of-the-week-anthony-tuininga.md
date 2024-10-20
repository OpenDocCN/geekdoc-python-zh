# 本周 PyDev:安东尼·图伊宁加

> 原文：<https://www.blog.pythonlibrary.org/2017/12/11/pydev-of-the-week-anthony-tuininga/>

本周，我们欢迎安东尼·图宁加成为我们的本周 PyDev！Anthony 是 cx 套件中 cx_Freeze 库的创建者之一。你可以感受一下他目前在 Github 上的工作。让我们花些时间来更好地了解安东尼吧！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

I grew up in a small town in the central interior of British Columbia, Canada. In spite of it being a small town, my school managed to acquire a personal computer shortly after they were made available. I was fascinated and quickly became the school guru. That experience convinced me that computers and especially programming were in my future. I moved to Edmonton, Alberta, Canada in order to attend university and ended up staying there permanently. Instead of only taking computing science courses I ended up combining them with engineering and received a computer engineering degree. After university I first worked for a small consulting firm, then for a large consulting firm and am now working for the software company, Oracle, in the database group. Besides working with computers I enjoy reading and both cross-country and downhill skiing.**Why did you start using Python?**

在 20 世纪 90 年代末，我开发了一个 C++库和一套工具来管理 Oracle 数据库对象。这些工作得相当好，但它们需要相当多的时间来开发和维护。我发现了 Python 及其 C API，并对最终成为 cx_Oracle Python 模块的东西做了一些实验。几天之内，我已经用 Python 和 cx_Oracle 重写了 C++库和一些工具，向自己证明 Python 是一个很好的选择。尽管被解释并且理论上比 C++慢，但我用 Python 编写的工具实际上更快，主要是因为与 C++相比，我可以用 Python 使用更高级的数据操作技术。我用 Python 完成了重写，并继续扩展我对 Python 的使用，直到我工作的公司的旗舰产品广泛使用它。令人欣慰的是，我工作的公司看到了开源模式的好处，我能够将我在那里开发的库和工具开源。其中包括 cx_PyGenLib、cx_PyOracleLib、cx_Oracle、cx_OracleTools、cx_OracleDBATools、cx_bsdiff、cx_Logging、ceODBC 和 cx_Freeze。

你还知道哪些编程语言，你最喜欢哪一种？

由于我喜欢尝试语言，所以我对相当多的语言了解有限，但在我的职业生涯中，我经常使用的语言是 C、C++、SQL、PL/SQL、HTML、JavaScript 和 Python。其中，Python 是我的最爱。我最近开始尝试使用 Go，作为 C/C++的替代品，它给我带来了新鲜空气。时间会证明我是否能很好地利用它，特别是因为我目前的工作需要大量使用 C 语言，而且这种情况不太可能很快改变。

你现在在做什么项目？

During work hours I am working on a C wrapper for the Oracle Call Interface API called ODPI-C ([https://github.com/oracle/<wbr>odpi](https://github.com/oracle/odpi)), cx_Oracle ([https://github.com/oracle/<wbr>python-cx_Oracle](https://github.com/oracle/python-cx_Oracle)) and node-oracledb ([https://github.com/oracle/<wbr>node-oracledb](https://github.com/oracle/node-oracledb)). Outside of work hours I still do a bit of work on cx_Freeze ([https://github.com/anthony-<wbr>tuininga/cx_Freeze](https://github.com/anthony-tuininga/cx_Freeze)).

哪些 Python 库是你最喜欢的(核心或第三方)？

The modules I have found to be the most useful in my work have been reportlab (cross platform tool for creating PDFs programmatically), xlsxwriter (cross platform tool for creating Excel documents without requiring Excel itself) and wxPython (cross platform GUI toolkit). I have also recently been making use of the virtues of the venv module (earlier known as virtualenv) and have found it to be excellent for testing.**What was your motivation for creating the cx_Freeze package?**As mentioned earlier I had built a number of tools for managing Oracle database objects. I wanted to distribute these to others without requiring them to install Python itself. I first experimented with the freeze tool that comes with Python itself and found that it worked but wasn't easy to use or create executables. I discovered py2exe but it was only developed for Windows and we had Linux machines on which we wanted to run these tools. So I built cx_Freeze and it worked well enough that I was able to easily distribute my tools and later full applications on both Windows and Linux and (with some help from the community) macOS. My current job doesn't require this capability so I have not been able to spend as much time on it as I did before.

**在维护这个项目的过程中，你学到的最重要的三件事是什么？**

These lessons have been learned not just with cx_Freeze but also with cx_Oracle, the other well-used module I originally developed. First, code you write that works well for you will break when other people get their hands on it! Everyone thinks differently and makes mistakes differently and that becomes obvious very quickly. Second, although well-written code is the most important aspect of a project (keep in mind lesson #1), documentation, samples and test cases are nearly as important and take almost as much time to do well, and without them others will find your project considerably more difficult to use. Finally, even though additional people bring additional and possibly conflicting ideas, the project is considerably stronger and useful the more contributors there are.

你还有什么想说的吗？

I can't think of anything right now!**Thanks for doing the interview!**![](img/80fa6adcfc3f64d015030e2a410150ff.png)