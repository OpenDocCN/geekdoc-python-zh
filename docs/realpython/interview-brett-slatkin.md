# 与 Brett Slatkin 的 Python 社区访谈

> 原文：<https://realpython.com/interview-brett-slatkin/>

今天我要和布雷特·斯莱特金 对话，他是谷歌的首席软件工程师，也是 Python 编程书籍 [*有效 Python*](https://realpython.com/asins/0134853989/) 的作者。加入我们，讨论 Brett 在 Google 使用 Python 的经历，重构，以及他在撰写他的书的第二版时所面临的挑战。事不宜迟，让我们开始吧！

**瑞奇:** *欢迎来到*真正的巨蟒，*布雷特。我很高兴你能和我一起参加这次面试。让我们像对待所有客人一样开始吧。你是如何开始编程的，你是什么时候开始使用 Python 的？*

![Brett Slatkin](img/12c10c170987644e157135f7b176f43d.png)

布雷特:谢谢你邀请我！我的编程之路漫长而多变。Python 是我教育结束时一个令人惊讶的转折。这是我从未预料到的，但却是一种祝福。一旦我学会了这门语言，我就爱上了它。

在我成长的过程中，我们家有电脑，这是一种巨大的特权。我总是喜欢把东西拆开来看看它们是如何工作的，我不记得有哪一次我不使用电脑并试图弄清楚它们。我已故的祖父最喜欢的故事之一是关于他有一次印刷问题，我帮他解决了。(当时我 3 岁。)

五年级的时候，有一天，我们全班去了一个计算机实验室，在一个[苹果 IIgs](https://en.wikipedia.org/wiki/Apple_IIGS) 上用 [Logo](https://en.wikipedia.org/wiki/Logo_(programming_language)) 编程。幸运的是，从那以后，我妈妈鼓励我对编程感兴趣，给我买书，让我参加书呆子夏令营。我用[乐高技术控制中心](http://lukazi.blogspot.com/2014/07/lego-legos-first-programmable-product.html)工具包给小机器编程。这些就是今天的乐高头脑风暴的前身。

我还使用 Borland Delphi 构建了简单的 GUI 应用程序，比如猜数字游戏。然后，我当地网吧的工作人员给我看了 Unix、Linux 和 C，让我大吃一惊。那时候我就知道我长大了要当程序员。

高中期间，我在当地的社区大学免费上课，学习了 [C](https://realpython.com/build-python-c-extension-module/) ，C++，x86 汇编，[游戏开发](https://realpython.com/pygame-a-primer/)。我读了 [Scott Meyers 的](https://en.wikipedia.org/wiki/Scott_Meyers)第二版*有效 C++* ，成为 C++及其所有神秘特性的超级粉丝。有一年夏天，我在一家网络泡沫初创公司实习，期间我用 ASP 和 VBScript 编写 web 应用程序。我对网络编程最感兴趣，并试图构建自己版本的 [DikuMUD](https://en.wikipedia.org/wiki/DikuMUD) 和[热线](https://en.wikipedia.org/wiki/Hotline_Communications)。

在大学里，我主修计算机工程，因为我想了解计算机如何在晶体管层面上工作。我的大部分课程是电气工程，在那里我使用了编程工具，如 [SPICE](https://en.wikipedia.org/wiki/SPICE) 、 [VHDL](https://en.wikipedia.org/wiki/VHDL) 和 [MATLAB](https://realpython.com/matlab-vs-python/) 。计算机科学课，学了 LISP，写了很多 [Java](https://realpython.com/oop-in-python-vs-java/) 。我是各种课程的助教，非常喜欢教别人。

最后，我第一次接触 Python 是在 2003 年，当时我正在研究 BitTorrent T2 客户端是如何工作的。(那时候是开源的。)我认为代码很难看，因为所有的`__dunder__`特殊方法，我忽略了我所有项目的语言！

2005 年大学毕业后，我去了谷歌，因为它似乎是学习更多网络编程知识的最佳地方。但是在公司的早期，你不能选择你所从事的工作作为新员工。你刚刚被分配到一个团队。

在我的第一天，我的老板给了我一本书 *Python 概要*，并告诉我去修复一个大约 25KLOC 的代码库。我原以为在工作中我会为网络系统编写 C++，但是我最终不得不做一些完全不同的事情。更糟糕的是，我只能靠自己了，因为原来的程序员已经因为精疲力竭而去强制休假了！

幸运的是，亚历克斯·马尔泰利是我加入的团队中的一位同事。他是《简而言之的 T2 Python》一书的作者，他帮助我学习了这门语言。我能够重写我继承的大部分代码库，修复它的基本问题，并将其扩展到 Google 的整个机器生产舰队。信不信由你，近 15 年后的今天，这个代码仍在使用。

与我使用 C++和 Java 的经验相比，我从使用 Python 中获得的生产力是惊人的。我对编程和语言的看法已经完全改变了。我现在对 Python 很感兴趣！

你已经在谷歌接触过使用 Python，你现在是谷歌的首席软件工程师，但是在你的任期内，你推出了谷歌的第一个云产品，App Engine。现在你是谷歌调查和其他项目的技术负责人。Python 在帮助你开发这些方面发挥了多大的作用，Python 在 Google 的未来是什么样的？

Brett: 毫无疑问，我在谷歌的职业生涯要归功于 Python，但事情远不止如此。作为背景，你要知道创建 App Engine 的五位工程师最初使用的是 JavaScript，类似于网景公司 1995 年的 [Livewire](https://developer.mozilla.org/en-US/docs/Archive/Web/Server-Side_JavaScript) 服务器。然而，这是在 Chrome 发布和 [V8](https://en.wikipedia.org/wiki/V8_(JavaScript_engine)) 大幅性能改进导致 NodeJS 之前。

创始人担心服务器端 JavaScript 会让系统显得过于小众。他们想提供[灯](https://en.wikipedia.org/wiki/LAMP_(software_bundle))栈，当时很流行。LAMP 中的“P”代表 Perl、PHP 或 Python。Perl 通常会过时，PHP 也不是谷歌内部使用的语言。但是 Python 是，感谢[彼得·诺维格](https://norvig.com/)，所以它是自然的选择。

2006 年末，当我用 Python 实现了 [dev_appserver](https://cloud.google.com/appengine/docs/standard/python3/testing-and-deploying-your-app#local-dev-server) 时，我开始在 App Engine 团队中做一个 20%的项目。在此之前，没有办法在部署应用程序之前在本地运行它。我对该产品的潜力及其对 Python 的使用感到非常兴奋，以至于不久后我就转而全职从事这项工作。

当时也在谷歌工作的吉多·范·罗苏姆很快加入了我们。在 2008 年的发布会上，我非常荣幸地做了[现场演示](https://youtu.be/tcbpTQXNwac)，在那里我用 Python 从头开始编写了一个 web 应用程序，并在几分钟内完成了部署。这些年来，我和 Guido 在各种项目上合作，其中最有趣的是我广泛测试的[ndb](https://cloud.google.com/appengine/docs/standard/python/ndb/async)([asyncio](https://docs.python.org/3/library/asyncio.html)的前身)。

我最喜欢的是在 App Engine 环境中用 Python 编写完整的应用程序，以推动产品的极限。我经常用我的演示程序打破系统，碰到我们没有意识到的新障碍。这将导致我们改进基础设施，以确保下一个需要扩展的应用程序，如 [Snapchat](https://www.businessinsider.com/snapchat-is-built-on-googles-cloud-2014-1) ，不会遇到同样的问题。

我最喜欢的演示应用是我和 T2 一起开发的 PubSubHubbub 协议和 hub。该协议将 RSS 源转化为实时流。它给开放网络带来了 Twitter 等服务的活力。

在其鼎盛时期，我们的 hub 应用程序每秒钟处理数百万个提要和数千个请求，所有这些都使用 Python on App Engine。从那时起，这个中心就被整合到了谷歌的标准网络爬行基础设施中。PubSubHubbub(现在叫做 [WebSub](https://www.w3.org/TR/websub/) )成为组成 [OStatus](https://en.wikipedia.org/wiki/OStatus) 的更大规范组的一部分，帮助 [Mastadon](https://blog.joinmastodon.org/2017/09/mastodon-and-the-w3c/) 起步。

[谷歌调查](https://research.google/pubs/pub46243/)作为 App Engine 上的另一个 Python 演示应用开始。我在短短几周内就完成了整个系统的端到端原型。我和我的联合创始人有机会向拉里·佩奇展示它，他批准了这个项目。由于 Python 提供的杠杆作用，我们在一年后推出了一个精益团队。

我们的服务现在每秒处理 50 多万个请求，影响了谷歌的大部分用户和收入。代码库已经从大约 10KLOC 增长到超过 1MLOC，其中大部分是 Python。我们不得不将一些服务迁移到 Go、C++和 Java，因为我们遇到了各种 CPU 成本和延迟限制，但 Python 仍然是我们所做一切的核心。

Python 已经在谷歌找到了自己的位置，主要是作为数据科学的工具[、](https://colab.research.google.com/notebooks/welcome.ipynb)[机器学习的工具](https://developers.google.com/machine-learning/crash-course)和 [DevOps / SRE 的工具](https://landing.google.com/sre/sre-book/chapters/release-engineering/)。现在，整个公司的工程师都在努力放弃 Python 2，将整个 monorepo 迁移到 Python 3，以保持与 Python 社区和开源包的一致。

**Ricky:** *除了你刚刚谈到的，你还是广受欢迎的 Python 编程书籍* [有效 Python](https://realpython.com/asins/0134853989/)*的作者，我们之前在[最佳 Python 书籍](https://realpython.com/best-python-books/)中评论过这些书籍。你最近出版了第二版，有实质性的更新。更新包括什么，自第一版以来有什么变化？*

布雷特:感谢你认为我的书是最好的书之一！第一版的反响非常好。它甚至被翻译成了八种语言！在[官方网站](https://effectivepython.com)上有很多关于这本新书的信息，包括所有项目标题的完整目录、各种样本和示例代码。

第二版 *Effective Python* 的长度几乎是原版的两倍，除了提供 30 多条新建议外，还对所有建议进行了重大修改。这本书的第一版是在 2014 年写的，当时考虑的是 Python 2.7 和 Python 3.4。在过去的 5 年里，Python 3.5 到 [Python 3.8](https://realpython.com/courses/cool-new-features-python-38/) 增加了太多东西，所以有很多东西可以写！总的风格是一样的，它仍然包括多色语法突出。

新书涵盖了 walrus 操作符、 [f-strings](https://realpython.com/courses/python-3-f-strings-improved-string-formatting-syntax/) 、字典、类[装饰器](https://realpython.com/courses/python-decorators-101/)的最佳实践，以及函数的所有重要特性(包括[异步](https://realpython.com/python-async-features/)执行)。我将详细介绍标准库中的解包归纳、定制排序、测试工具和算法性能。我还解释了如何利用更高级的工具，比如[打字](https://realpython.com/python-type-checking/)模块和`memoryview`类型。

在这个新版本中，我对元类的建议完全改变了，这要感谢语言中添加了`__init_subclass__`和`__set_name__`。我对使用`send()`和`throw()`的高级发生器的建议现在与第一版相反。(现在我说避开他们。)我将 [asyncio](https://realpython.com/courses/python-3-concurrency-asyncio-module/) 的优势与线程和进程进行了比较，这在以前是完全没有的。我提供了关于如何将同步代码转换为异步代码，以及如何将异步代码与线程混合的指导。

任何读过这本书的人，请随时给我发送问题或反馈！

**瑞奇:** *在 PyCon 2016 上，你做了一个题为[重构 Python:为什么以及如何重构你的代码](https://youtu.be/D_6ybDcU5gc)的演讲，这个演讲也在我们的[十大必看 PyCon 演讲](https://realpython.com/must-watch-pycon-talks/)之列。我很想知道你演讲的动机。是什么关于重构(或者说缺乏重构)激发了你做一个关于重构的演讲？人们似乎很纠结于重构吗？他们会陷入同样的陷阱吗？*

Brett: 谢谢你把我的演讲列入你的十大清单！

当我做这个演讲的时候，我已经在同一个代码库工作了大约六年。那个时代的任何代码都需要不断地重构和更新，以防止它腐烂和变得不可用。

当依赖性以向后不兼容的方式改变时，当底层基础设施转变为具有不同的性能特征时，或者当产品决策打破了大量先前的假设时，我所说的 rot 就会发生。因此，重构是一种持续的需要，它经常出现在我的脑海中。这次经历让我对重构技巧的重要性有了新的认识。

人们经常谈论测试、工具和框架如何使他们成为更有效的程序员。但是我相信重构是一项被低估的基本技能，每个程序员都需要集中精力去提高。为什么？在我看来，最好的代码是不存在的、已经被删除或从未被写过的代码。你能拥有的最好的程序是一个空的`.py`文件。问题是你如何从你在一个项目中的位置更接近那个理想？通过重构。

我想帮助我团队中的其他程序员学习如何更好地重构，但是马丁·福勒的《重构》这本书自 1999 年以来就没有更新过。所以我在考虑为这本书写一个 Python 专用版本，使它现代化，成为我的同事们可以阅读的东西。然而，当我得知 Fowler 正在开发第二版的*重构*时，我就把它搁置了，该版本已经发布，并且使用 JavaScript 作为实现语言。

不幸的是，它仍然达不到我所需要的。我希望看到一本专门针对 Python 的重构书，或者一组可行的例子，充分利用这种语言所提供的一切。

**里基:** *现在我的最后几个问题。你在业余时间还做些什么？除了 Python 和编程，你还有什么其他的爱好和兴趣？*

布雷特:我们家里有两个很小的孩子，所以我和妻子花很多时间陪他们。唱歌和演奏音乐(主要是钢琴，但有时吉他、尤克里里琴、木琴和口琴)是享受我们在一起时光的最好方式。我对所有这些乐器都很糟糕，也不是一个伟大的歌手，但我们仍然有很多美好的时光。

我喜欢每天都在外面，要么在旧金山的山上散步，要么去跑步(通常推着婴儿车)。我一有机会就喜欢冲浪。我喜欢阅读，尤其是长篇新闻，但事实上，主要是 reddit 和 T2 龙虾。我想成为一名更好的厨师，在懒惰和美味之间找到正确的平衡。我还试图自学如何使用贝叶斯方法建立统计模型。如果说这些年来我学到了什么，那就是没有什么是确定的。

* * *

谢谢你，Brett，这周和我一起！你可以在 [Twitter](https://twitter.com/haxor) 或者 [GitHub](https://github.com/bslatkin) 上找到 Brett，在[effectivepython.com](https://effectivepython.com)了解更多关于第二版有效 Python 的信息。

如果你想让我在未来采访谁，请在下面的评论中联系我，或者在 Twitter 上给我发消息。编码快乐！