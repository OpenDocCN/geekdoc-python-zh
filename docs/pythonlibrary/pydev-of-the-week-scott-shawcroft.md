# 本周 PyDev:斯科特·肖克罗夫特

> 原文：<https://www.blog.pythonlibrary.org/2019/07/01/pydev-of-the-week-scott-shawcroft/>

本周我们欢迎 Scott Shawcroft ( [@tannewt](https://twitter.com/tannewt) )成为我们的本周 PyDev！Scott 是 [CircuitPython](https://learn.adafruit.com/welcome-to-circuitpython/what-is-circuitpython) 的首席开发者，这是一种为微控制器设计的 Python 编程语言的变体。如果你想知道斯科特还在做什么，他的[网站](http://tannewt.org/)是个不错的起点。让我们花一些时间来更好地了解 Scott！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我叫 Scott，2009 年毕业于华盛顿大学计算机工程专业。后来，我加入了谷歌的地图团队，从事地图的渲染和设计工作。2015 年离开，去做自己的事。我为赛车四轴飞行器设计了一个模块化的飞行控制器系统，同时学习了硬件。我的爱好包括跑步、攀岩、视频游戏和旧货淘复古电子产品(这样我就可以把 CircuitPython 放进去了。)

**你为什么开始使用 Python？**

大约在 2004 年，我开始使用 Python 制作我的第一个桌面应用程序( [Denu](http://denu.sourceforge.net/) )。我第一次学习编程是用 PHP 和网站。我想转向桌面编程，还记得我站在书店里，在 Perl 和 Python 书籍之间做选择。我因为某些原因挑了 Python，一直没有回头。

你还知道哪些编程语言，你最喜欢哪一种？

我说过，想要动态 HTML 之后，先学了 PHP。(这是在 CSS 和 Javascript 真正成为一个东西之前。)从那以后我就再也没有真正接触过 PHP。

在学校里，我们主要学习 Java，少量学习其他语言。在我讲授计算机编程入门课程的同时，我还教授了一个用 Python 讲授这门课程的选修课。买了新的 MacBook Pro 后，我用 Python 对苹果多点触控板进行了逆向工程，并在 2008 年用 C 语言为它实现了一个守护进程。(这是我的 Linux 内核[声称成名](https://github.com/torvalds/linux/blob/6f0d349d922ba44e4348a17a78ea51b7135965b1/drivers/input/mouse/bcm5974.c#L7)。)

我在谷歌做 Javascript，在 GMail 上实习。一旦我在谷歌开始全职工作，我就在服务器上做 C++。对于我的嵌入式工作，我主要使用 C 语言(甚至在 CircuitPython 中)。

选择一个最喜欢的有点难。Python 总是脚本编写、原型制作和教学的良好开端。当你想管理你自己的内存时，最新版本的 C 和 C++也非常好。

你现在在做什么项目？

我的核心项目是 CircuitPython。它是 Python 的重新实现，目标是使编程和构建变得容易。它基于 MicroPython，为我们在 CircuitPython 中所做的改进奠定了基础。我的日常工作是扩展和完善 CircuitPython 的底层 C 代码。

我在业余时间设计和破解硬件，让 CircuitPython 在新设备上运行。例如，我设计了一个运行 CircuitPython 的 GameBoy 推车，使编写 GameBoy 程序变得更加容易。我还有一个烤面包机和钢琴键盘等着 CircuitPython 的大脑。之前，我用 CircuitPython 制作了一个定制的机械电脑键盘。我喜欢黑客设备使用 CircuitPython，因为这是攻击代码最简单的方法。

哪些 Python 库是你最喜欢的(核心或第三方)？

使用硬件时 Struct 可能是我的最爱。它对于与外部传感器接口非常有用。

我最喜欢的第三方库是 requests，因为它使得编写像 GitHub 这样的 REST APIs 很容易。

**你是如何成为 CircuitPython 的首席开发人员的？**

在某种程度上，这是偶然的。我在为我的房子制作传感器时发现了 Adafruit。几年后，在制作飞行控制器时，我成了 Adafruit“展示与讲述”节目的常客。一旦我的硬件业务不可持续，我就开始寻找软件工作。我问 Adafruit 他们是否有什么，他们提出付钱让我把 MicroPython 移植到他们的 M0 主板上(后来变成了 CircuitPython)。进展非常顺利，我继续努力。

所以，在某种程度上，我成为了领导者。随着项目的发展，我已经设定了项目的愿景，并开始着手其他一些项目。我们将看到 CircuitPython 如何随着它的继续发展而发展。

CircuitPython 的未来让你兴奋的是什么？

我很高兴看到人们用它做出各种不同的东西。CircuitPython 以一种前所未有的方式将 Python 的易用性与硬件的有形性结合在一起。

随着我们扩大设备支持，我们将会看到更多使用 CircuitPython 构建的项目。当我们添加移动工作流支持时，我们将看到我们的受众范围扩大，包括那些主要使用移动设备而不是笔记本电脑或台式机的人。

你能描述一下让 Python 在嵌入式系统上工作的一些挑战吗？

在嵌入式系统上开发 Python 的最大挑战是有限的 RAM。随着一个项目发展到使用许多库，ram 的占用也在增长。一旦 RAM 已满或碎片化，代码就无法继续。幸运的是，廉价的微控制器空间仍然是摩尔定律的领地。我移植 MicroPython 的原始微控制器 SAMD21 有 32 千字节的 ram。我们最新的 SAMD51 有 192 千字节到 256 千字节。空间很大。😉

你还有什么想说的吗？

我想鼓励软件人员尝试用硬件来构建一些东西。拿着自己编程的实物并与之互动是一件非常有趣的事情。

微控制器编程为“全堆栈”带来了全新的含义。通过跳过 Windows 或 Linux 等完整操作系统包含的许多层，在微控制器上运行的代码更接近于“金属”。如果没有这些层，就更容易理解 CPU 和内存的机制，因为它只是您运行的代码。

斯科特，谢谢你接受采访！