# 本周的 PyDev:a . Jesse Jiryu Davis

> 原文：<https://www.blog.pythonlibrary.org/2015/02/09/pydev-of-the-week-a-jesse-jiryu-davis/>

本周我们邀请了 a . Jesse Jiryu Davis([@ jessejiryudavis](https://twitter.com/jessejiryudavis))加入我们，成为我们的本周 PyDev。我是通过 Jesse 在他的[博客](http://emptysqua.re/)上发表的关于 Python 的文章来了解他的。他还受雇于 [MongoDB](http://www.mongodb.org/) 。Jesse 还是几个 Python 相关项目的贡献者，比如 [pymongo](https://github.com/ajdavis) 。让我们花些时间了解一下我们的蟒蛇同伴吧！

[![a-jesse-jiryu-davis](img/67222d2478b488a7a9e2c6a15fc9884d.png)](https://www.blog.pythonlibrary.org/wp-content/uploads/2014/12/a-jesse-jiryu-davis.jpg)

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我在奥柏林学院获得了计算机科学学士学位，这在当时并不是个好主意。我在网络泡沫时期上学。奥柏林的计算机科学教授被软件公司挖走了，或者被愿意支付足够的薪水让他们留在学术界的大大学挖走了。奥柏林是一个理想主义的地方，它不会给计算机科学教授比同等级别的人类学教授多两倍的薪水。这是一个很好的原则，它让我们系损失了很多教员。数学和物理的教授来了，本科生也互相教授。我教的是中级 C++。我没有教得很好。

所以我接受了不完整的教育，在网络泡沫破裂后我就毕业了，但我仍然设法在奥斯汀的一家小型飞行数据分析公司找到了一份好工作。几年后，我离开去南加州的横须贺禅山中心学习禅一年，然后来到纽约。

**你为什么开始使用 Python？**

当我来到纽约时，我有两年用 C++做 3D 图形的经验。不知何故，我在 Wireless Generation(现在是 Amplify Education)做了三年 Python 和 SQL。他们在 CGI 脚本中使用无栈 Python 2.3，所以这就是我第一次学习 Python 的方式。

无堆栈 Python 可以在生成器暂停时对其进行 pickle 处理，所以这就是我们如何处理多消息协议的:当 CGI 脚本等待客户端发送下一条消息时，它将其状态保存为 NFS 共享磁盘上的 pickle 生成器，并自行终止。当下一条消息到达时，无论哪个 web 前端接收到它，都会找到保存的文件，解开生成器，并恢复它，直到生成器生成回复消息。

这些天来，与我一起工作的大多数人都离开了 Amplify，去创建 Mortar Data 和 DataDog，而那些留在 Amplify 的人已经用现代 Python 和 MongoDB 更新了技术堆栈！

我离开后，在纽约的创业现场做自由职业者，MongoDB 是我在这里遇到的最令人兴奋的东西。所以我以 30 号员工的身份加入了。

你还知道哪些编程语言，你最喜欢哪一种？

我对任何东西的了解都不及 Python 的一半。我要改变这一切。明年我会专注于我的 C 技能。

你现在在做什么项目？

我和我的老板 Bernie Hackett 一起开发 PyMongo，Python 的 MongoDB 驱动程序。PyMongo 是顶级的 Python 包之一，它被用于所有平台上的所有 Python 版本。这是一项重大的责任，我们必须小心行事。

马达是我的一个有趣的副业。它类似于 PyMongo，但它是异步的。它与 Tornado、异步 Python 框架一起工作，我目前正在添加对 asyncio 的支持。最终它甚至会扭曲，所以这将是一个 Python 异步帽子戏法。

我还对另一个项目很感兴趣，Monary。这是一个专门为 NumPy 开发的 MongoDB 驱动程序，由 David Beach 编写。单线程每秒可以读取一两百万行。在 MongoDB，Inc .我们认为 Monary 真的很酷，所以我一直在领导我们对它的贡献。

MongoDB 有 C、C++、C#、Java、Node、Perl、PHP、Python、Ruby 和 Scala 的驱动程序。让所有这些司机行为一致是一项巨大的努力。我写了一些我们的规范，定义了所有的驱动程序应该如何运行。例如，我写了一个“服务器发现和监控规范”,它定义了我们的驱动程序用来与 MongoDB 服务器集群对话的分布式系统策略。我在 YAML 定义了单元测试，证明我们的驱动程序符合相同的规范，尽管有不同的实现。

哪些 Python 库是你最喜欢的(核心或第三方)？

*龙卷风和阿辛西奥。上次我接近 Twisted 失败了，但我会再试一次。Yappi 是一个比 cProfile 好得多的分析器，但它鲜为人知。*

你还有什么想说的吗？

对于那些以不使用 ide 为荣的人，我有一条信息要告诉他们:你需要使用 PyCharm，并且要善于使用它。当我看到你在 vim 中到处乱转，试图通过按名称搜索来找到函数，试图用“打印”语句来调试时，你看起来很可怜。你认为自己很有效率，但是不使用 PyCharm 浪费了太多时间。你就像一个不知道自己舞跳得有多差的人。学好 PyCharm，用它进行编辑、调试、版本控制。您将直接跳到函数定义，您将在几秒钟内解决合并冲突，您将立即诊断错误。你会感谢我的。

感谢您的宝贵时间！

### 一周的最后 10 个 PyDevs

*   伊夫林·德米罗夫
*   [Andrea Gavana](https://www.blog.pythonlibrary.org/2015/01/26/pydev-of-the-week-andrea-gavana/)
*   蒂姆·戈登
*   道格·赫尔曼
*   玛格丽塔·迪利奥博士
*   [马里亚诺·莱因加特](https://www.blog.pythonlibrary.org/2014/12/29/pydev-of-the-week-mariano-reingart/)
*   巴斯卡尔·乔德里
*   蒂姆·罗伯茨
*   汤姆·克里斯蒂
*   史蒂夫·霍尔登