# 本周 PyDev:罗伯特·斯莫尔希尔

> 原文：<https://www.blog.pythonlibrary.org/2022/12/19/pydev-of-the-week-robert-smallshire/>

本周，我们欢迎罗伯特·斯莫尔夏尔([@罗布·斯莫尔夏尔](https://twitter.com/robsmallshire))成为我们本周的 PyDev！罗伯特是[六十北](https://sixty-north.com/)咨询公司的创始人。罗伯特还为 Pluralsight 制作了令人惊叹的 [Python 视频](https://www.pluralsight.com/authors/robert-smallshire)。罗伯特也有几本他创作的 Python 书籍。

让我们花几分钟时间来更好地了解 Robert！

## 你能告诉我们一些关于你自己的情况吗(爱好、教育等):

在 20 世纪 80 年代早期的家庭计算革命中，我在大约 9 岁的时候开始接触计算，并在那个时候开始编程。我现在 48 岁了，退一步想一想我已经修修补补了将近 40 年，这很奇怪。从小到大，在我接受正规教育的过程中，我从未觉得编程会成为我的职业。我只有一个计算机方面的小资格证书(计算机研究 GCSE——16 岁时获得的英国资格证书),毕业后，我的研究使我进入了自然科学领域，最终获得了地质学博士学位。在我的学校和大学期间，我一直在断断续续地进行业余编程，包括编写一些软件来帮助我父亲的工程测量业务，但在我读博士期间需要更多新颖的软件，所以我必须创建它。我把时间分成两部分，一部分花在漫长的夏季露营，在法国和瑞士的阿尔卑斯山进行野外工作，另一部分花在英国潮湿的冬天，躲在实验室或家里用 C++编写分析和建模软件来处理结果。

当我还在写博士论文的时候，我开始为一家大学附属公司工作，从事能源和采矿方面的商业 R&D。这也是野外工作和案头工作的公平分配，在犹他州摩押附近的沙漠中进行了数月的旅行。白天，我们在野外工作，或者从一架轻型飞机上工作，我们已经说服飞行员卸下舱门，以便更好地进行低空航空摄影。晚上回到我们的基地，我们会编写 Perl 脚本来处理我们的数据。偶尔在休息日，我们会去拱门和峡谷国家公园远足。这是一项高强度的脑力和体力工作，在我工作生涯的初期，我非常幸运能与如此聪明和积极的人们一起工作；我从他们身上学到了很多。

尽管有做实地工作的特权，但我很快意识到，能够编程是一种超能力，而且当你天生对手头的问题感到好奇并有动力去解决时，这种能力会加倍。对地质学有很好理解并有编程天赋的人似乎相对较少，意识到这一点，我决定申请一家公司的工作，这家公司提供我在 R&D 公司遇到的一些分析软件。

这将我带到了苏格兰的格拉斯哥，这是我第一次真正以“开发人员”的身份工作。我们开发了一个大型复杂的图形密集型 C++系统，在我加入的时候，这个系统运行在昂贵的硅图形和 Sun 工作站上。该软件允许我们建立地球内部的 3D 模型，并通过地质时间的第四维度来回运行它们，通过算法剥离沉积层，折叠和展开弯曲的岩石，以及断裂和不断裂的岩石。这几乎是不可思议的！再一次，我有幸与一些非常聪明、积极、精力充沛的人一起工作并向他们学习，包括我未来的妻子。几年后，我领导了这家公司的开发团队，我很高兴地向大家报告，我在这个时候(2002 年的某个时候)开始设计的一个新系统在二十年后的今天仍在开发和销售。五年后，经历了能源市场的起伏带来的一些困难时期，我和我的合伙人决定寻找其他选择。不管怎样，她经常往返于英国和挪威之间，在搬到加拿大的想法失败后，我最终在挪威奥斯陆找到了一份工作，并在几周内搬了家。几个月后，我的搭档也加入了。

在挪威，我再次从事石油和天然气行业的模拟软件工作，但我的新雇主的一切都比我迄今为止所经历的要大一个数量级，包括 250 万行 C++的单片代码库，这需要一夜之间从头编译。仅仅几个星期后，我觉得自己犯了一个职业生涯的巨大错误，如果我没有穿越北海，我可能会像到达一样迅速离开。不过，还有另一个选择。正如马丁·福勒所说，“你要么改变你的公司，要么改变你的公司。”

我决定留在这里足够长的时间，看看我是否能有所作为，几个月后，我发现事情正朝着我的方向发展。一些其他的新鲜血液被带进了已经变得愚笨的软件开发文化中，我们一起开始扭转局面。其中一个人叫 Austin Bingham，他和我有很多共同点，都是挪威移民，都有过使用 Python 的正面经历，都非常认真地对待软件工程和设计，而不仅仅是“编码”。在七年多的时间里，我升到了首席软件架构师这个令人兴奋的高度，我向你保证，这听起来比实际情况更宏伟。但尽管如此，在一个营业额达数亿美元的企业中，我仍然是负责我们产品内部设计和编程的最高级别人员。我的一个关键决定是通过将 CPython 解释器嵌入到我们的核心产品中来引入 Python 形式的脚本 API。

在我七年任期快结束时，公司被出售，并被一系列私募股权公司转让，很明显，金融工程比软件工程更有价值——而且很可能更有价值。不久之后，该公司在一次复杂的交易中再次被一家美国大型企业集团收购，这家企业集团似乎有点惊讶地发现，他们已经收购了一家软件公司以及他们实际上认为他们正在收购的业务部门。现在，作为 135 000 名员工中的一员——一台巨大机器中的一个小齿轮——我决定是时候继续前进了。

我想继续前进的另一个原因是我想离开石油和天然气行业。原因有两个:首先，由于我和妻子都在能源行业工作，我们这个不断壮大的家庭特别容易受到该行业臭名昭著的盛衰周期的影响。其次，我担心我的工作对气候的负面影响。在地球系统科学方面受过一些训练，并认识到软件可以极大地增强人类能力，很明显，像我这样的人可能会对气候产生巨大的负面影响。世界上很大一部分油田都是用软件建模和模拟的，我至少名义上对这些软件负有技术责任，我的设计和代码也是其中的一部分。

在斯塔万格和奥斯陆的酒吧与我的同事奥斯汀·宾汉姆(Austin Bingham)进行了几次会面后，我们决定在 2013 年建立自己的公司，提供软件培训和咨询服务。

我们的新公司 Sixty North 自然会专注于我们熟知的软件技术——尤其是 Python——以及服务于具有重要科学或工程内容的问题领域，但也包括那些我们可以利用软件架构技能来管理大型系统复杂性的领域。我们的经验是，许多科学家和工程师有能力编写小型代码，但缺乏设计系统的知识、技能和经验，因此他们可以相对优雅地成长——这在今天可能比以往任何时候都更真实。

我们经营 Sixty North 已经十年了，主要是作为一家生活方式公司，而不是追求永久的增长。事实证明，我们非常擅长我们所做的事情，并且能够在我们的头上保持一个屋顶，并与少数人一起维持一个企业。

职业生涯讲了很多，那我在外面做什么呢？我试图培养一些爱好，让我远离屏幕，让我保持活跃。我 20 多岁的大部分时间都在爬山、下洞穴和骑自行车，但在 30 多岁时，我的身体活动水平显著下降，当时我和妻子忙于——更不用说不堪重负——忙于应付繁忙的职业生涯、商务旅行和抚养孩子。在我四十多岁的时候，我又开始骑自行车了，现在我尽量在夏天的大部分时间里骑自行车，在冬天尽可能多的滑雪。我很幸运生活在世界上一个非常美丽的地方。

## 你为什么开始使用 Python？

我第一次认真使用 Python 是在 2001 年，当时我在 [SCons 构建工具](https://scons.org)中遇到了它。当时，我正在为苏格兰的一家公司开发用 C++编写的图形密集型商业地球科学模拟软件。代码有一个可怕的构建系统，用 make 递归实现(参见递归 Make 被认为是有害的),这很难理解，也不可靠。那时候，我们庞大的 C++代码库必须在 Irix、Solaris 和 Windows 上连夜构建，因此构建系统中的错误和失误代价高昂。在评估了一些可供选择的 make 之后，我们偶然发现了 SCons，并进入了 Python 的世界。我们使用的是 Python 1.5.2，因为这是我们可以在我们需要支持的所有系统上构建的最新版本，带有我们拥有的编译器。当时 Perl 是我的首选脚本语言，但是在接下来的一两年里，我对 Perl 的使用几乎完全被 Python 取代了，尽管我所有的“严肃”编程仍然是用 C++完成的。

我觉得更高级的语言会让我们的团队比在 C++中更有效率，以至于我花了很多精力在我们的大型 C++应用程序中嵌入 Perl 解释器。回想起来，我们选择了一个合理的软件架构——C++和嵌入式脚本解释器，类似于现代的网络浏览器——但是在 Perl 中，我们犯了一个错误，选择了一种可读性和可维护性都不如 c++的语言！

大约在这个时候，我正在试验 Java，遇到了用于 JVM 的 Jython——Python。我对这种组合感到非常兴奋，因为它承诺将快速的编译语言(Java)与高级语言(Python)结合起来，这两种语言都将避免 C++中与内存管理相关的许多臭名昭著的陷阱。特别是，Java 提供了 Swing GUI 工具包、Java 2D 和 Java 3D 图形 API，可以通过在 Jython 解释器上执行的 Python 代码很好地实现这些功能。我记得曾向一位同事热情介绍过 Samuele Pedroni 和 Noel Rappin 的《Jython Essentials 》( 2002)——这是一本比当时大多数直接的 Python 书籍更好的 Python 介绍——并在 JVM 上用 Jython 构建了有趣的原型应用程序，这些应用程序可移植到我们使用的所有操作系统上，并且避免了乏味的编译-链接-运行循环。

可悲的是，Jython 从未真正达到逃逸速度，尽管它同时拥有 Python 和 Java 标准库，但它提供了许多常规 CPython 目前仍然缺乏的东西，特别是在 GUI 和图形工具包方面。从那以后，我将 Python 引入了其他基于 C++的公司，也是通过 SCons 向量，后来在 Austin Bingham 的帮助下，通过将 Python 解释器嵌入 C++应用程序。

## 你还知道哪些编程语言，你最喜欢哪一种？

我已经提到了 Perl、C++和 Java，但我是在 20 世纪 80 年代中期用 BBC BASIC 学习编程的，随后通过 COMAL(晦涩难懂！)，6502 和 ARM 汇编语言，Pascal，C，C++，Delphi (Object Pascal)。我还用 C#和 F#开发了相当重要的代码库，甚至在专业环境中开发了一些 Haskell。大部分或者可能是大部分现在已经被遗忘了，但是我现在经常使用的语言是 Python(每天)、JavaScript(很多天)和 Java(偶尔大量使用)，这三种语言的组合反映了我在工作中使用的语言。我仍然喜欢探索新的和旧的语言(但对我来说是新的)。最近，我尝试了 Julia 编程语言，并且正在为我正在设计和制造的一台家酿 8 位计算机的老式 6809 微处理器编写一个汇编程序(用 Python)。如果我需要再一次用 C++的性能来开发新项目，我会非常努力地学习 Rust。如果我需要做更多的 JavaScript(很可能)，我可以看到自己想要进入 TypeScript。

我看到许多程序员在他们的职业生涯中等待下一种神奇的编程语言来解决他们所有的问题。我也经历过类似的情绪——特别是第一次体验 Lisp 时，或者是在。NET 公共语言运行时——但是我觉得我现在已经过了那个阶段至少十年了，回头看我自己还是很天真的。我们目前有一些优秀的编程语言和生态系统，而不是闪亮的新语言，通过使用我们已经拥有的语言会很容易获得收益。关键是要聪明而勤奋地使用它们，认真对待系统和软件架构。如果你知道一种高性能/低水平的语言，如 C++，知道用于 web 的 JavaScript，知道一种通用的低摩擦语言，如 Python，你几乎可以实现任何事情。

在所有这些语言中，Python 是不断吸引我的语言，也是我首先接触的语言，除非设计约束迫使我转向另一个方向。Python 使得从最初的想法到有用的解决方案的时间特别短。也就是说，理解 Python 什么时候不合适真的很重要，在这方面，我已经为一些代价高昂的错误负责。

## 你现在在做什么项目？

大约十年前，当奥斯汀·宾汉姆(Austin Bingham)和我创办咨询和培训公司 Sixty North 时，我们——在瑞典哥德堡的一次软件会议上偶然相遇——开始为 Pluralsight 制作在线培训课程材料。任何进行过预录培训的人都知道，设计培训材料、制作好的示例、手动
捕捉高质量的演示视频、录制清晰的音频，以及编辑所有这些来制作高质量的产品是一项巨大的工作。在我们课程的第一次迭代中，我们做的一切都和大多数人现在做的一样，用视频捕捉我们“现场”开发代码，进行无数次重拍和大量编辑，将代码片段粘贴到 Keynote 幻灯片中并手动注释，等等。

当需要更新我们的课程以跟上 Python 的发展、PyCharm 等工具的最新版本、更高分辨率的输出以及更严格和时尚的图形设计要求时，我认为公平地说，数百小时的手工返工的
前景并没有立即让我们充满喜悦。

相反，我们认为至少在原则上，我们可以从机器可读的课程描述中产生所有的材料(演示视频、幻灯片、图表)。我们可以自动同步视频和音频画外音，然后，当新版本的 Python 或 PyCharm 发布时，或者当需要以不同的视频分辨率提供课程时，我们可以对配置文件或演示示例代码进行一些更新，并重新呈现整个课程。

当然，对于这种需求的“原则上的”和“实践中的”解决方案之间的区别在于，在构建工具来做到这一点，以及向系统描述我们所有的视频课程方面，需要做大量的工作。不用说，我们在 Pluralsight 上发布了大约 25 个小时的 Python 视频培训材料，这些材料可以完全自动地呈现，而且非常重要的是，可以廉价地修改。

At the time of writing, we're exploring our options for bringing this technology to a wider audience, and removing some of the rough edges from the now substantial video course production system my colleagues and I at Sixty North have built.

## Which Python libraries are your favorite (core or 3rd party)?

Many of the Python packages we make are designed to have a clear Python API and on top of that a CLI. I've found click to be an excellent library for specifying command-line interfaces. For testing, I regularly turn to David MacIver’s excellent property-based testing library, Hypothesis.

## What are the top 3 things you learned while writing a book or video course?

1\. Having to teach a topic in an excellent way to learn it well.
2\. Finding good teaching examples which exhibit Goldilocks “just right” complexity requires long walks or bike rides, but also a grounding in experience to understand and demonstrate their relevance.
3\. Most books have a hopeless financial return compared to invested effort, but are good for gaining credibility. I wouldn’t advise anybody to write a technical book for the money from sales, but to instead write one to support other aspects of a broader consulting or training business. For example. our Python Craftsman series, The Python Apprentice, The Python Journeyman, and The Python Master, are derived directly from our work on our classroom and Pluralsight training materials and they mutually support each other.

## Is there anything else you’d like to say?

This open ended question made me contemplate the things which have transformed my ability to build quality software. Along with using a language you love – or at least learning to love the language you use – I would add the following advice:

First of all, taking testing seriously has had a big impact on my work. I sometimes, though by no means always, practice Test-Driven Development (TDD). Even when I’m not using TDD the tests are rarely far behind and are usually written contemporaneously with the production code. The effort in arranging for code to be testable will be repaid many times over, not just in terms of correctness, but for other desirable qualities of the system.

Secondly, taking architecture and design seriously has been transformational for me. Largely this is about following a handful of maxims: “Do one thing and do it well”, “Separate concerns”, “An architecturally significant decision is one which is costly to change later”, “Instead of asking objects for their state, tell objects what to do, and give them what they need to do it”, “Prefer pure functions”, and so on.

Many of these boil down to keeping control of coupling and cohesion, and it’s hard to overstate the importance of these for sustained success.

The third, least technical, and most enjoyable practice that has made a huge impression on me in recent years is daily pair or ensemble programming. This really took off for us during the Covid-19 pandemic to the extent that the majority of code we write at Sixty North has at least two people involved at the point of creation, and I feel our code, happiness and team spirit is much better for it. I wouldn’t like to go back to long stretches of working alone, to asynchronous code review, or to modes of working based around pull-requests.

Finally, I’d like to thank you for giving me opportunity to tell a bit of my story.

**Thanks for doing the interview, Robert!**