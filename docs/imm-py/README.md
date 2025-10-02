# Python 进阶

《Python 进阶》是《Intermediate Python》的中文译本, 谨以此献给进击的 Python 和 Python 程序员们!

### 快速阅读传送门

*   可以直接使用 Github 快速阅读任一章节：[进入目录](https://github.com/eastlakeside/interpy-zh/blob/master/SUMMARY.md)
*   也可以使用 Gitbook 更完整顺序地阅读：[进入 Gitbook](https://eastlakeside.gitbooks.io/interpy-zh/content/)
*   还可以下载 pdf/epub/mobi 到本地或 kindle 上阅读：[进入下载](https://github.com/eastlakeside/interpy-zh/releases)

# 前言

Python，作为一个"老练"、"小清新"的开发语言，已受到广大才男俊女的喜爱。我们也从最基础的 Python 粉，经过时间的吹残慢慢的变成了 Python 老鬼。

IntermediatePython 这本书具有如下几个优点：

1.  简单
2.  易读
3.  易译

这些都不是重点，重点是：**它是一本开脑洞的书**。无论你是 Python 初学者，还是 Python 高手，它显现给你的永远是 Pyhton 里最美好的事物。

> 世上语言千万种 美好事物藏其中

译者在翻译过程中，慢慢发现，本书作者的行文方式有着科普作家的风范，--那就是能将晦涩难懂的技术用比较清晰简洁的方式进行呈现，深入浅出的风格在每个章节的讨论中都得到了体现：

*   每个章节都非常精简，5 分钟就能看完，用最简洁的例子精辟地展现了原理
*   每个章节都会通过疑问，来引导读者主动思考答案
*   每个章节都引导读者做延伸阅读，让有兴趣的读者能进一步举一反三
*   每个章节都是独立的，你可以挑选任意的章节开始阅读，而不受影响

总之，这本书非常方便随时选取一个章节进行阅读，而且每次阅读一个章节，你都可能会有一些新的发现。

## 原书作者

感谢英文原著作者 @yasoob《[Intermediate Python](https://github.com/yasoob/intermediatePython)》，有了他才有了这里的一切

## 译者

老高 @spawnris
刘宇 @liuyu
明源 @muxueqz
大牙 @suqi
蒋委员长 @jiedo

## 欢迎建议指正或直接贡献代码

[`github.com/eastlakeside/interpy-zh/issues`](https://github.com/eastlakeside/interpy-zh/issues)

### 微信交流群

> ![微信群](img/dd9b1836)

### 微信打赏支持：

> ![wechat_donate](img/donate.png)

# 序

这是一本[Intermediate Python](https://github.com/yasoob/intermediatePython) 的中文译本, 谨以此献给进击的 Python 和 Python 程序员们!

这是一次团队建设、一次尝鲜、一次对自我的提升。相信每个有为青年，心里想藏着一个小宇宙：**我想要做一件有意思的事**。$$什么是有意思的事？$$ **别闹**

Python，作为一个"老练"、"小清新"的开发语言，已受到广大才男俊女的喜爱。我们也从最基础的 Python 粉，经过时间的摧残慢慢的变成了 Python 老鬼。因此一开始 @大牙 提出要翻译点什么的时候，我还是挺兴奋的，团队一起协作，不单可以磨练自己，也能加强团队之间的协作。为此在经过短暂的讨论后，翻译的内容就定为：《Intermediate Python》。

IntermediatePython 这本书具有如下几个优点：

1.  简单
2.  易读
3.  易译

这些都不是重点，重点是：**它是一本开脑洞的书**。无论你是 Python 初学者，还是 Python 高手，它显现给你的永远是 Pyhton 里最美好的事物。

> 世上语言千万种 美好事物藏其中

翻译的过程很顺利，语言很易懂，因此各位读者欢迎捐赠，或加入微信群讨论。

# 译后感

# 译者后记

## 译者大牙感言

在翻译过程中，慢慢发现，本书作者的行文方式有着科普作家的风范，--那就是能将晦涩难懂的技术用比较清晰简洁的方式进行呈现，深入浅出的风格在每个章节的讨论中都得到了体现：

*   每个章节都非常精简，5 分钟就能看完，用最简洁的例子精辟地展现了原理
*   每个章节都会通过疑问，来引导读者主动思考答案
*   每个章节都引导读者做延伸阅读，让有兴趣的读者能进一步举一反三
*   每个章节都是独立的，你可以挑选任意的章节开始阅读，而不受影响

总之，这本书非常方便随时选取一个章节进行阅读，而且每次阅读一个章节，你都可能会有一些新的发现。

## 本译作已开源，欢迎 PullRequest

*   gitbook: [`eastlakeside.gitbooks.io/interpy-zh/`](https://eastlakeside.gitbooks.io/interpy-zh/)
*   github: [`github.com/eastlakeside/interpy-zh`](https://github.com/eastlakeside/interpy-zh)
*   gitbook 与 github 已绑定，会互相同步

# 原作者前言

# 关于原作者

我是 Muhammad Yasoob Ullah Khalid.

我已经广泛使用 Python 编程 3 年多了. 同时参与了很多开源项目. 并定期在[我的博客](http://pythontips.com/)里写一些关于 Python 有趣的话题.

2014 年我在柏林举办的欧洲最大的 Python 会议**EuroPython**上做过精彩的演讲.

> 译者注：分享的主题为：《Session: Web Scraping in Python 101》 地址：[`ep2014.europython.eu/en/schedule/sessions/20/`](https://ep2014.europython.eu/en/schedule/sessions/20/)

如果你能给我有意思的工作机会, 请联系我哦.

> 译者注：嗯，硬广，你来中国么，HOHO

# 作者前言

Hello 大家好! 我非常自豪地宣布我自己创作的书完成啦.
经过很多辛苦工作和决心, 终于将不可能变成了可能, "Intermediate Python"终于杀青.
ps: 它还将持续更新 :)

Python 是一门奇妙的语言, 还有一个强大而友爱的程序员社区.
然而, 在你理解消化掉 Python 的基础后, 关于下一步学习什么的资料比较缺乏. 而我正是要通过本书来解决这一问题. 我会给你一些可以进一步探索的有趣的话题的信息.

本书讨论的这些话题将会打开你的脑洞, 将你引导至 Python 语言的一些美好的地方. 我最开始学习 Python 时, 渴望见到 Python 最优雅的地方, 而本书正是这些渴望的结果.

无论你是个初学者, 中级或者甚至高级程序员, 你都会在这本书里有所收获.

请注意本书不是一个指导手册, 也不会教你 Python. 因为书中的话题并没有进行基础解释, 而只提供了展开讨论前所需的最少信息.

好啦，你肯定也和我一样兴奋, 那让我们开始吧!

# 开源公告

注意: 这本书是开源的, 也是一个持续进展中的工作. 如果你发现 typo, 或者想添加更多内容进来, 或者可以改进的任意地方(我知道你会发现很多), 那么请慷慨地提交一个 pull request, 我会无比高兴地合并进来. :)

另外, 我决定将这本书免费发布! 我相信它会帮助到那些需要帮助的人. 祝你们好运!

这里是免费阅读链接:

*   [Html](http://book.pythontips.com/)
*   [PDF](http://readthedocs.org/projects/intermediatepythongithubio/downloads/pdf/latest/)
*   [GitHub](https://github.com/IntermediatePython/intermediatePython)

# 广告

注意: 你也可以现在为我捐助, 如果你想买[Gumroad](https://gumroad.com/l/intermediate_python) 提供的高大上版本.

你也可以加入我的[邮件列表](http://eepurl.com/bwjcej), 这样你可以保持同步获取到重大更新或者我未来其他项目!

最后而且也很重要的是, 如果你读了这本书, 并且发现它很有帮助, 那么一个私人邮件和一个 tweet 分享, 对我来说会很有意义.