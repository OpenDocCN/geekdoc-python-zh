# 8 家使用 Python 的世界级软件公司

> 原文：<https://realpython.com/world-class-companies-using-python/>

目前有 500 多种编程语言，而且每天都在增加。不可否认的是，这些方法中的大多数是重叠的，而且很多从来都不是为了在理论或实验室环境之外使用。但是对于日常编码和商业中使用的编程语言，你必须做出选择。你应该学习什么语言，你为什么要花时间学习它们？

由于这是一个专门研究 Python 的网站，我们已经告诉你为什么 Python 是一门很好的学习语言。你可能知道 Python 可能是树莓派最喜欢的语言(因为大多数都预装了它)。知道了这一点，你就知道你可以用一个 Pi 工具包和一点点聪明才智做什么惊人的事情。**虽然很容易看出如何修补 Python，但您可能想知道这如何转化为实际的业务和现实世界的应用程序。**

我们现在要做的是告诉你你所知道的使用 Python 的八家顶级公司。通过这种方式，您可以看到 Python 开发人员在现实世界中有多么大的机会。

## 工业光魔

工业光魔(ILM)是特效工作室，由乔治·卢卡斯于 1975 年创建，为《星球大战》创造特效。从那时起，他们就成了 FX 的代名词，因为他们在电影和商业中的工作赢得了多个奖项。

在早期，ILM 专注于实际效果，但很快意识到计算机生成的效果是 FX 的未来。他们的 CGI 部门成立于 1979 年，他们的第一个效果是《星际迷航 II:可汗之怒》中创世纪项目的爆炸场景。

最初，ILM 的 CGI 工作室运行在 Unix shell 上，但这只是处理相对较少的工作量。因为工作室预见了 CGI 的未来，他们开始寻找一个可以处理他们在未来看到的积极向上扩展的系统。

ILM 选择了 Python 1.4 而不是 Perl 和 Tcl，选择使用 Python 是因为它可以更快地集成到他们现有的基础设施中。因为 [Python 与 C 和 C++](https://www.davincicoders.com/codingblog/2017/2/10/love-movies-learn-to-code-python-and-you-might-work-for-ilm) 的简单互操作性，ILM 很容易将 Python 导入他们专有的照明软件。这让他们可以将 Python 放在更多的地方，用它来包装软件组件并扩展他们的标准图形应用程序。

工作室已经在他们工作的多个其他方面使用了 Python。开发人员使用 Python 来跟踪和审计管道功能，维护为每部电影制作的每张图像的数据库。随着越来越多的 ILM 程序由 Python 控制，它创建了一个更简单的统一工具集，允许更有效的生产管道。对于现实世界的例子，只要看看 ILM 使用的一种高清文件格式 [OpenEXR](http://www.openexr.com/index.html) 就知道了。作为软件包的一部分， [PyIlmBase](https://github.com/openexr/openexr/tree/develop/PyIlmBase) 被包含在内(尽管它有一个 Boost 依赖项)。

尽管有许多评论，ILM 仍然认为 Python 是满足其需求的最佳解决方案。开源代码与支持更改的能力相结合，确保 Python 将在很长一段时间内继续满足 ILM 的需求。

[*Remove ads*](/account/join/)

## 谷歌

谷歌几乎从一开始就是 Python 的支持者。一开始，Google 的创始人[做出了“Python 在我们能做的地方，C++在我们必须做的地方”的决定。这意味着 C++被用在内存控制是必要的，并且需要低延迟的地方。在其他方面，Python 支持易于维护和相对快速的交付。](https://stackoverflow.com/questions/2560310/heavy-usage-of-python-at-google/2561008#2561008)

即使用 Perl 或 Bash 为 Google 编写了其他脚本，这些脚本也经常被重新编码到 Python 中。原因是因为易于部署以及 Python 的维护非常简单。事实上，根据《在 Plex》的作者 Steven Levy 所说，谷歌的第一个网络爬行蜘蛛最初是用 Java 1.0 的[版本编写的，因为太难了，所以他们把它改写成了 Python。](https://realpython.com/oop-in-python-vs-java/)

Python 现在是官方的 Google 服务器端语言之一——c++、Java 和 Go 是另外三种——允许部署到生产中。如果你不确定 Python 对谷歌有多重要，Python 的 BDFL[吉多·范·罗苏姆](https://en.wikipedia.org/wiki/Guido_van_Rossum)从 2005 年到 2012 年在谷歌工作。

最重要的是，彼得·诺维格说:

> “Python 从一开始就是 Google 的重要组成部分，并且随着系统的成长和发展而保持不变。如今，数十名谷歌工程师使用 Python，我们正在寻找更多掌握这种语言的人。”

## 脸书

脸书的制作工程师非常热衷于 Python，这使它成为社交媒体巨头中第三大最受欢迎的语言(仅次于 C++和他们专有的 PHP 方言 Hack)。平均而言，脸书有超过 5，000 项公用事业和服务承诺，管理基础架构、二进制分发、硬件映像和运营自动化。

Python 库的易用性意味着生产工程师不必编写或维护太多代码，让他们可以专注于实时改进。这也确保了脸书的基础设施能够高效地扩展。

根据脸书 2016 年的一篇帖子，Python 目前负责基础设施管理中的多种服务。其中包括使用 TORconfig 处理网络交换机设置和映像，使用 FBOSS 处理白盒交换机 CLI，使用 Dapper 调度和执行维护工作。

脸书发布了许多为 Py3 编写的开源 Python 项目，包括一个[脸书 Ads API](https://github.com/facebook/facebook-python-ads-sdk) 和一个 [Python 异步 IRCbot 框架](https://github.com/facebook/pyaib)。脸书目前正在将其基础设施和装卸机从 2 升级到 3.4，AsyncIO 正在帮助他们的工程师进行升级。

## Instagram

2016 年，Instagram 工程团队夸口说，他们正在[运行完全用 Python](https://engineering.instagram.com/web-service-efficiency-at-instagram-with-python-4976d078e366) 编写的 Django web 框架的全球最大部署。这可能在今天仍然适用。Instagram 的软件工程师闵妮这样描述他们对 Python 的生产使用:

> “我们最初选择使用 Python 是因为它简单实用的名声，这与我们‘先做简单的事情’的理念非常吻合。”"

从那时起，Instagram 的工程团队投入了时间和资源，以保持他们的 Python 部署在大规模( [~8 亿月活跃用户](https://www.statista.com/statistics/253577/number-of-monthly-active-instagram-users/))下的可行性，他们正在运营:

> “通过我们为 Instagram 的网络服务构建效率框架的工作，我们有信心继续使用 Python 扩展我们的服务基础设施。我们也开始在 Python 语言本身上投入更多，并开始探索将我们的 Python 从版本 2 迁移到版本 3。”

2017 年，Instagram 将其大部分 Python 代码库[从 Python 2.7 迁移到 Python 3](https://thenewstack.io/instagram-makes-smooth-move-python-3/) 。您可以观看郭美孜和丁辉的 [PyCon 2017 主题演讲](https://www.youtube.com/watch?v=66XoCk79kjM)，了解他们在大规模代码迁移方面的经验:

[https://www.youtube.com/embed/66XoCk79kjM?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com](https://www.youtube.com/embed/66XoCk79kjM?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com)

## Spotify

这家音乐流媒体巨头是 Python 的大力支持者，主要将这种语言用于数据分析和后端服务。在后端，有大量的服务都通过 0MQ 或 [ZeroMQ](http://zguide.zeromq.org/page:all) 进行通信，这是一个用 Python 和 C++(以及其他语言)编写的开源网络库和框架。

之所以用 Python 编写服务，是因为 Spotify 喜欢用 Python 编写和编码时开发管道的速度。Spotify 架构的最新更新一直在使用 [gevent](http://www.gevent.org/) ，它提供了一个带有高级同步 API 的快速事件循环。

为了向用户提供建议和推荐，Spotify 依赖于大量的分析。为了解释这些，Spotify 使用了与 Hadoop 同步的 Python 模块 [Luigi](https://github.com/spotify/luigi) 。这个开源模块处理库如何协同工作，并快速合并错误日志，以便进行故障排除和重新部署。

总的来说，Spotify 使用超过 6000 个单独的 Python 进程，这些进程在 Hadoop 集群的节点上协同工作。

[*Remove ads*](/account/join/)

## Quora

这个庞大的众包问答平台想了很久很久，想用什么语言来实现他们的想法。Quora 的创始人之一 Charlie Cheever ，将他们的选择缩小到 Python、C#、Java 和 Scala。他们使用 Python 的最大问题是缺乏类型检查，而且相对较慢。

据 Adam D'Angelo 说，[他们决定不使用 C#](https://www.quora.com/Why-did-Quora-choose-Python-for-its-development-What-technological-challenges-did-the-founders-face-before-they-decided-to-go-with-Python-rather-than-PHP/answer/Adam-DAngelo?srid=vt0q) 因为这是微软的专有语言，他们不想受制于任何未来的变化。此外，任何开源代码充其量只能得到二等支持。

Java 比 Python 更难写，而且它不像 Python 那样适合非 Java 程序。当时，Java 还处于起步阶段，所以他们担心未来的支持以及这种语言是否会继续发展。

取而代之的是，Quora 的创始人效仿 Google，选择使用 Python，因为它易于编写且可读性强，并为性能关键部分实现了 C++。他们通过编写单元测试来完成同样的事情，从而避开了 Python 缺乏类型检查的缺陷。

使用 Python 的另一个关键考虑是当时存在几个好的框架，包括 Django 和 Pylons。此外，因为他们知道 Quora 将涉及服务器/客户端交互，不一定是全页面加载，让 Python 和 JS 如此好地合作是一个巨大的优势。

## 网飞

网飞使用 Python 的方式与 Spotify 非常相似，依靠 Python 语言在服务器端进行数据分析。然而，它并不止于此。网飞允许他们的软件工程师选择用什么语言来编码，并且已经注意到 Python 应用程序的数量有了很大的增长。

接受调查时，网飞的工程师提到了标准库、非常活跃的开发社区和丰富多样的第三方库，它们可以解决几乎任何给定的问题。此外，由于 Python 非常容易开发，它已经成为网飞许多其他服务的关键。

使用 Python 的主要场所之一是中央警报网关。这个 RESTful web 应用程序处理来自任何地方的警报，然后将它们发送给需要看到它们的人或团体。此外，该应用程序有能力抑制已经处理过的重复警报，并在某些情况下，执行自动解决方案，如重新启动进程或终止看起来开始不稳定的事情。考虑到警报的绝对数量，这个应用程序对网飞来说是一个巨大的胜利。智能地处理它们意味着开发人员和工程师不会被多余的调用淹没。

Python 在网飞的另一个应用领域是用于跟踪安全变化和历史的 monkey 应用程序。这些猴子用于跟踪和警告任何组中 EC2 安全相关策略的任何变化，跟踪这些环境中的任何变化。它们还用于确保跟踪网飞多个域附带的数十个 SSL 证书。从追踪数据来看，自 2012 年以来，网飞的意外到期率从四分之一降至零。

## 收存箱

这个基于云的存储系统在其桌面客户端使用 Python。如果你对 Dropbox 在 Python 上的投入有任何疑问，那就想想 2012 年，他们成功说服了 Python 的创造者、仁慈的终身独裁者吉多·范·罗苏姆离开谷歌，加入 Dropbox。

Rossum 加入 Dropbox 的条件是[他将成为一名工程师](https://medium.com/dropbox-makers/guido-van-rossum-on-finding-his-way-e018e8b5f6b1)，而不是一名领导，甚至不是一名经理。在第一年，他帮助 Dropbox 社区中的其他用户实现了共享数据存储的能力。

虽然 Dropbox 的许多库和内部都是专有的，而不是开源的，但该公司已经发布了用 python 编写的非常有效的 API，让你可以看到他们的工程师是如何思考的。当你阅读[对 Dropbox 工程师](https://talkpython.fm/episodes/transcript/30/python-community-and-python-at-dropbox)的采访，了解到他们的服务器端代码有很大一部分是 Python 时，你也可以从字里行间体会到这一点。

有趣的是，虽然客户端程序是用 Python 编写的，但是它们利用 Mac 和 Windows 机器上的各种库来提供统一的体验。这是因为 Python 不是预装在 Windows 上的，根据您的 Mac，您的 Python 版本会有所不同。

## Reddit " t0 "号

该网站在 2017 年每月有 5.42 亿访客，使其成为美国第四大访问量网站和世界第七大访问量网站。2015 年有 7315 万次投稿，825.4 亿次浏览量。在这一切的背后，形成软件主干的是 Python。

Reddit 最初是用 Lisp 编码的，但是在 2005 年 12 月，也就是它上线 6 个月后，这个网站被重新编码成 Python。这种变化的主要原因是 Python 拥有更广泛的代码库，在开发上更加灵活。最初运行该网站的 web 框架 web.py 现在是一个开源项目。

在 2009 年的一次采访中，Steve Huffman 和 Alexis Ohanian 在 Pycon 上被问到为什么 Reddit 仍然使用 Python 作为它的框架。[根据霍夫曼](https://brainsik.net/2009/why-reddit-uses-python/)的说法，第一个原因与变化的原因相同:

> “什么都有一个图书馆。我们一直在学习这些技术和架构。因此，当我不理解连接池时，我可以找到一个库，直到我自己更好地理解它并编写我们自己的库。我不理解 web 框架，所以我们将使用其他人的，直到我们做出自己的……Python 有一个这样的强大支撑。”

Reddit 坚持使用 Python 的第二个原因是贯穿所有使用 Python 开发的公司的一条共同主线。根据 Huffman 的说法，是代码的可读性:

> “当我们雇佣新员工时……我认为我们还没有雇佣懂 Python 的员工。我只是说，‘你写的所有东西都需要用 Python 来写。’这样我就可以读了。这太棒了，因为我可以从房间的另一边看到他们的屏幕，不管他们的代码是好是坏。因为好的 Python 代码有非常明显的结构。
> 
> 这让我的生活轻松多了。[……]它极具表现力、可读性和可写性。这让生活变得更加顺畅”

**更新:**是的，现在有 9 家世界级的公司在生产中使用 Python。最初我们没有单独统计 Instagram，因为该公司为脸书所有。但鉴于 Instagram 团队令人印象深刻的运营规模，我们认为给他们一个单独的要点是有意义的。

[*Remove ads*](/account/join/)

## 还有人吗？

在这篇文章中，我们关注了八家在生产中使用 Python 的世界级成功软件公司。但他们不是唯一的。截至 2018 年，Python 的采用达到了一个新的高峰，并继续攀升。

我们有没有漏掉名单上的人？在下面留下评论，让我们知道你最喜欢的 Python 商店！***