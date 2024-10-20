# 本周皮德夫:瑞安·米切尔

> 原文：<https://www.blog.pythonlibrary.org/2015/10/19/pydev-of-the-week-ryan-mitchell/>

本周，我们欢迎瑞安·米切尔([@组装师](https://twitter.com/Kludgist))成为我们本周的 PyDev。Ryan 是用 Python 编写的[网页抓取和用 Java 编写的](http://www.amazon.com/gp/product/1491910291/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=1491910291&linkCode=as2&tag=thmovsthpy-20&linkId=UQHISFFOLKVHPEPZ)[即时网页抓取的作者。](http://www.amazon.com/gp/product/1849696888/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=1849696888&linkCode=as2&tag=thmovsthpy-20&linkId=2FYUTALJFWRHZGE5)让我们花些时间来更好地了解 Ryan。

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我是奥林工程学院的毕业生，明年将获得哈佛大学扩展研究学院的软件工程硕士学位(在过去的三年里，我一半时间都在从事这项工作)。凭借“普通工程”的学士学位，我在 UX 和应用程序设计、企业家精神、it、生物工程、软件架构设计和编程之间徘徊。不过，我想我已经非常坚定地选择了软件工程。我目前是南波士顿一家名为 LinkeDrive 的初创公司的 SE，我们从长途卡车引擎中获取大数据。这太棒了。

就爱好而言，我是波士顿科学博物馆科技工作室的志愿者，每周日下午。我热爱教学，这是参与社区活动的好方法，同时还能看到你想看的所有 IMAX 和天文馆展览！我也是一名变装女王，也是永久放纵修女会(一个由“变装修女”组成的非营利性组织，主要为当地事业筹款)的成员——这是一个更长的故事了！
**你为什么开始使用 Python？**

说实话，那是因为这是我当时上的一门大学课程的必修课！我高中时在 Sun 和微软实习过，所以显然没有接触过太多 Python。在大学期间，我忙于工程课程，没有太多时间去学习任何不需要的东西。但是一旦我有机会学习它，我真的爱上了这种语言，并且它很快成为我非 web 项目的首选。

你还知道哪些编程语言，你最喜欢哪一种？

在过去的 12 年里，我几乎涉猎了每一种现代编程语言(当教授说我们可以用任何语言编写 FORTRAN 95 时，我甚至开玩笑地学习了它来做一个项目——从那以后他改变了政策！).和许多人一样，我从 BASIC 开始，然后转向 C、Java、C#、Perl，然后开始用 PHP 和 JavaScript 做网站(但我们在生活中都犯过错误)。在大学里，我涉猎了很多学术语言，比如 MATLAB/C 代码，当然还有 Python。

我尽量不偏爱语言，但我的日常工作主要是用 Java，我是这种语言的爱好者。如果我正在处理一个涉及大量数学的机器学习项目，我可能会使用 Python。如果有很多复杂的业务逻辑，我可能会使用 Java，但这主要是对每种语言的公认惯例和流行库的偏见，而不是语言本身的基本属性。

你现在在做什么项目？

当然，除了我的日常工作之外，我还会在网上搜集大量信息来宣传和支持这本书，为第二版寻找灵感，写博客，为 O ' Reilly 制作一个视频系列(10 月份拍摄，可能很快就会发布)。

另外，我还在做一个超级机密的项目，可能要到下一次 DEF CON 才会发布(或者更早，如果这个话题不被接受的话，但还是祈祷吧！)

哪些 Python 库是你最喜欢的(核心或第三方)？

*最喜欢的核心库:urllib -我写网页抓取器，所以不得不说！*

第三方:BeautifulSoup 是一个显而易见的选择。我知道它与所有的 HTML 解析器竞争激烈，当然还有 Python 的核心 HTMLParser，但是我发现 BeautifulSoup 与其他一些库相比，速度非常快，重量轻，灵活，易于使用。绝对是我解析 HTML 的首选。

最近我喜欢上了 Python 图像库——我经常使用它来自动解决验证码问题，甚至进行随机的批量图像处理任务(例如，调整图像文件夹的大小),这对于手工操作来说是很痛苦的。

当然，还有 SciPy、NumPy 和 NLTK。我唯一的抱怨是，我对机器学习知之甚少，无法最大限度地使用它们，但是它们非常容易上手和运行，即使你有一个相对琐碎的任务要做，所以我真的建议你看看其中的一个或全部，如果你还没有！我报名参加了一个非常硬核的机器学习课程，将于 9 月开始，因此我对此非常兴奋，并对这些库有了更多的了解。

是什么让你决定写一本关于 Python 的书？

嗯，这本书其实不是关于 Python 的，而是关于 web 抓取的！几年前，我在 Packt 出版社写了一本小一点的书——用 Java 进行即时网络抓取。网络抓取是我非常喜欢的一门学科，也是我喜欢教学的一门学科。是 Packt 建议我用 Java 写这本书的。那时，我已经有几年没有使用 Java 了，所以这种语言肯定不是我的首选。因为工作的原因，我当时正在用 Python 编写 web scrapers，所以说实话，我需要为此做很多研究。

去年，我就写一本更长的网络信息搜集书籍的问题找到了 O ' Reilly，并告诉他们我可以用 Python 或者 Java 来写。他们当然会说 Python——讽刺的是，我现在已经换了工作，所以我白天都在写 Java。我从来没有用我当时日常工作所用的语言写过一本书，这使得当你决定是否要在一行的末尾添加一个分号时，真的很困惑！

你还有什么想说的吗？

我很荣幸成为本周的 PyDev！任何人都可以随时联系我[@组装师](https://twitter.com/kludgist)。我喜欢在 Twitter 上讨论 Python 和网络抓取。另外，我在[http://pythonscraping.com/blog](http://pythonscraping.com/blog)有一个初出茅庐的博客，在那里我写关于随机网络抓取/Python 的想法，并欢迎对使用 Python 的[网络抓取](http://www.amazon.com/gp/product/1491910291/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=1491910291&linkCode=as2&tag=thmovsthpy-20&linkId=UQHISFFOLKVHPEPZ)第二版的反馈！谢谢！

**谢谢！**

### 一周的最后 10 个 PyDevs

*   [卡罗尔心甘情愿](https://www.blog.pythonlibrary.org/2015/10/12/pydev-of-the-week-carol-willing/)
*   迈克尔·福格曼
*   [特雷西·奥斯本](https://www.blog.pythonlibrary.org/2015/09/28/pydev-of-the-week-tracy-osborn/)
*   [特里匈奴](https://www.blog.pythonlibrary.org/2015/09/21/pydev-of-the-week-trey-hunner/)
*   克里斯托弗·克拉克
*   马修·努祖姆
*   [肯尼斯·洛夫](https://www.blog.pythonlibrary.org/2015/08/31/pydev-of-the-week-kenneth-love/)
*   梅拉妮·克拉奇菲尔德
*   莱西·威廉姆斯·亨舍尔
*   [公羊轴](https://www.blog.pythonlibrary.org/2015/08/10/pydev-of-the-week-ram-rachum/)