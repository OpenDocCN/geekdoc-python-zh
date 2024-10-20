# 本周 PyDev:卢西亚诺·拉马尔霍

> 原文：<https://www.blog.pythonlibrary.org/2014/11/03/pydev-of-the-week-luciano-ramalho/>

本周的 PyDev 是卢西亚诺·拉马尔霍，他是即将出版的《流畅的 Python》一书的作者。他很友好，花了几分钟时间与我交谈。让我们看看他要说什么。

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

*我在 1995 年成为巴西第一家大型新闻门户网站 Brasil Online (BOL)的 CTO，没有学位。我用计算器(TI-58)自学编程，当时微型计算机在巴西还很少见而且非常昂贵(20 世纪 70 年代末)。我开始学了一些不同的专业(数学、经济学、哲学)，但随着计算机变得越来越普遍，我的程序员生涯开始起步，我多次从大学辍学。后来，在第一次网络浪潮中，我成立了一家公司，专门为新闻机构构建定制的 CMS。该公司是网络泡沫破裂的受害者，然后我终于回到了大学并获得了图书馆和信息科学学位，因为我在构建大型门户网站时对信息架构产生了兴趣，而 LIS 是信息架构的基础所在。*

作为业余爱好，我帮助运行 Garoa Hacker Clube，一个巴西的黑客空间。我也是棋盘游戏的收藏家和发明者，也是易经的学生。我不太喜欢骑自行车，但我喜欢在我的城市圣保罗骑自行车——我住的西区很好，大部分地方都很平坦。

**你为什么开始使用 Python？**

在 BOL，我们尝试了几种服务器端 web 编程的语言，但是在 Java-EE 出现之前(那是 1995-1996 年)，我们主要使用 Perl，一点 PHP/FI，甚至一些 Java。我对这些选择都不满意，所以当我离开去创建自己的公司时，我做了大量的研究并发现了 Python。那是 1998 年。到那年年底，我已经用 Python 部署了我的第一个新闻门户，并且我被深深地吸引住了。我用 Zope 作为框架。

你还知道哪些编程语言，你最喜欢哪一种？

我一直喜欢尝试新的语言，但是 Python 非常适合我的大脑，以至于在我开始使用它之后，我已经不像以前那样四处看看了。在 Python 之前，Perl 是我最喜欢的语言，在那之前，我是 Turbo Pascal 的超级粉丝。我花了很多时间玩 Smalltalk，我喜欢它，但我从来没有用它建立过任何真实的东西。在我开始使用 Python 之后，我学习最多的语言是 Ruby 和 LISP (Scheme 和 Common Lisp)。我也试着了解 Java。最近我对 Clojure、Go 和 Elixir 非常感兴趣，但是在我开始写我的书《流利的 Python》之后，我又 100%地专注于 Python 了。在我 2015 年完成这本书之后，我计划认真地钻研后三者之一。

你现在在做什么项目？

*现在我 100%的时间都在研究流畅的 Python([http://shop.oreilly.com/product/0636920032519.do](http://shop.oreilly.com/product/0636920032519.do))。这是我的第一本书，我非常喜欢写它。与我在 O'Reilly 的编辑 Meghan Blanchette 和全明星技术评论团队一起工作是一种特权。我正在使用 Atlas，O'Reilly 的图书出版平台，它基于 AsciiDoc、git、DocBook 和许多其他部分，集成得非常好。我在这里写了关于这个过程的博客:[http://www.python.pro.br/ramblog/with-oreilly/](http://www.python.pro.br/ramblog/with-oreilly/)*

在我意识到我需要 100%的奉献来写作之前，我正在开发 Pingo ( [http://pingo.io](http://pingo.io) )，这是一个用于多设备 GPIO 编程的 Python 库，支持 Raspberry Pi、pcDuino、UDOO 和 Arduinos(通过 USB 远程控制)。我写完书就回去看，但不是抛弃。Pingo 的另一位核心开发人员卢卡斯·维多正在业余时间从事这项工作。

哪些 Python 库是你最喜欢的(核心或第三方)？

 *我爱请求，烧瓶，金字塔和 lxml。它们使用起来都很愉快，Pythonic 化且健壮——是我们开发 Pingo 的灵感来源。Tkinter 有一个老而笨拙的名声，但它非常强大，最近已经可以构建好看的 GUI 了。看一看 Tkinter 中的 Canvas 对象:它就像是一个乐高积木，可以做面向对象的 ui，用于图表或基于矢量的绘图。很厉害，水平很高。最后，在专业地使用了 Twisted 和 Tornado 之后，我希望有它们最好特性的混合，我认为 async.io 是 Python 历史上一个非常重要阶段的开始:用生成器适当地支持现代异步编程。*

是什么让你决定写一本关于 Python 的书？

我从 1998 年开始使用 Python，从 1999 年开始教授它，所以写一本关于它的书的想法已经伴随我很久了。我现在这样做是因为我设法把所有必要的部分放在一起:与我欣赏的出版商的交易，能够专注于写作的财政资源，一套涵盖语言许多方面的精心打造的例子。这些例子是我在 Python.pro.br 工作的一部分，这是我与伦佐·努西泰利共同拥有的一家培训公司。自从我们在 2012 年创建 Python.pro.br 以来，我建立了大量的例子、解释和图表，并要求以书籍的形式组织起来。

如果你想看我用来教 Python 的图表、解释和例子，这里有一些例子:

*   [Python 元组:不变但可能变化](http://radar.oreilly.com/2014/10/python-tuples-immutable-but-potentially-changing.html)(为 O'Reilly Radar 博客发布的一篇文章)
*   [带描述符的封装](https://speakerdeck.com/ramalho/python-encapsulation-with-descriptors)(在 2013 年美国 PyCon 和 2014 年 OSCON 上展示的演讲幻灯片)
*   [迭代&生成器:Python 之道](https://speakerdeck.com/ramalho/iterators-and-generators-the-python-way)(2013 年美国皮肯大会和 2013 年 OSCON 大会上的演讲幻灯片)

 *当然，我的教诲的最终集锦是书😉*

*   [流畅的 Python:清晰、简洁、有效的编程](http://shop.oreilly.com/product/0636920032519.do)

**谁是目标受众？**

我正在为那些对 Python 有所了解，但又想充分利用 Python 3 作为一种现代的、一致的、优雅的和非常高效的面向对象的语言的人编写流利的 Python。我一边写一边想:

*   *随着 Python 成为主流，程序员们被扔进了 Python 项目的深水区*
*   那些肤浅地学习 Python 的程序员——也许是在一些学术领域——但是想要专业地使用它
*   *花了大量时间在支持 Python 2 的遗留代码库上的程序员，他们想学习 Python 3 的现代习惯用法(其中许多在 Python 2.7 中也有)*

你还有什么想说的吗？

作为一名 FLOSS 倡导者，我参加过围绕不同语言、数据库、平台、框架等组织的社区活动。我很荣幸成为 Python 社区的一员，在这里我们享受着友好、专业、团结与和谐的完美结合。

### 前一周的 PyDevs

*   沃纳·布鲁欣
*   浸信会二次方阵
*   本·劳什