# 本周 PyDev:马西莫·迪皮耶罗

> 原文：<https://www.blog.pythonlibrary.org/2016/03/21/pydev-of-the-week-massimo-dipierro/>

本周，我们欢迎马西莫·迪皮耶罗( [@mdipierro](https://twitter.com/mdipierro) )成为我们本周的 PyDev！马西莫是 [web2py](http://web2py.com) 的发明者和首席开发者，但他也为许多其他项目做出了贡献，你可以在他的 [github 简介](https://github.com/mdipierro)上看到这些。他还是以下书籍的作者:[Python 中的注释算法:在物理、生物和金融中的应用](http://www.amazon.com/gp/product/0991160401/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=0991160401&linkCode=as2&tag=thmovsthpy-20&linkId=TOXGCPE3UX4VT572)和 [web2py 完整参考手册](http://www.amazon.com/gp/product/0578120216/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=0578120216&linkCode=as2&tag=thmovsthpy-20&linkId=4LC3FCN6GQK5R4VN)。让我们花些时间去更好地了解他！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我今年 44 岁，出生在意大利，从小就喜欢数学和科学。我拥有比萨大学的物理学学士学位和英国南安普顿大学的理论物理学博士学位。这意味着我能读懂谢尔登在《生活大爆炸》中写的大部分公式。虽然，当 Sheldon 在研究弦理论时，我在研究晶格量子色动力学，一种 QCD 的计算方法，这种模型描述了夸克如何结合在一起形成宇宙中的大多数物质。这就是我来到美国的原因:我来到这里，在费米国家加速器实验室做博士后研究员。最终，我意识到我在计算机领域的学术生涯比在物理领域有更好的机会，于是我接受了 T2 德保罗大学计算机学院的一份工作。我大部分时间都在编程、做研究和咨询。我喜欢水肺潜水、航海和滑雪，但是我不能做我想做的那么多。

**你为什么开始使用 Python？**

大约 10 年前，我开始使用 Python。当我还是一名物理学家的时候，我以一个用于晶格量子色动力学计算的 C++库而闻名，这个库被称为费米量子色动力学库。我为它创建了一个 Python 包装器。

你还知道哪些编程语言，你最喜欢哪一种？
 *我知道很多编程语言。我最了解和最喜欢的是 C、C++、Python 和 JavaScript。我学的第一门语言是 COBOL。我父亲教我的。第二个是基本的。第三个是帕斯卡。第四个是 Clipper，我卖出的第一个程序是 Clipper。我在正式课程中学到的唯一语言是 Scheme。我从名著《计算机程序的结构和解释》中学到了这一点，并编写了符号微分代码。在我的博士论文中，我使用高性能的 FORTRAN 和 C++编写了一个 Cray T3E，并使用了 MPI。在 DePaul，我教过 C/C++、Java、Python、JavaScript 和 OpenCL。Java 是我最不喜欢的语言。JavaScript 是我第二不喜欢的，但随着时间的推移，我已经学会欣赏它，我发现自己最近比 Python 使用得更多。*

你现在在做什么项目？

我正在做许多项目，有些我不能谈论。大多数项目都与 Python 语言有关，尤其是 web2py。我为 web2py 用户提供支持，有时帮助他们启动新项目。其他时候，我帮助他们修复旧项目。我还在不断开发新的 Python 模块，以拓展 web2py 的功能，探索它的未来。一个花费了我大量时间但与 web2py 无关的项目是卡米奥。我是 http://camio.com T2 公司的高级工程师，我们已经建立了一个平台，可以将任何带摄像头的设备变成安全摄像头。图像和视频被上传并存储在云中，由神经网络进行分析，并使用自然语言进行搜索。大部分代码都是用 Python 构建的，包括机器学习。

哪些 Python 库是你最喜欢的(核心或第三方)？

我最喜欢的图书馆是 pickle 和 ast。Pickle 是一个核心库，允许序列化和反序列化 Python 中的几乎任何数据结构。这意味着它是微不足道的保存和恢复任何程序的状态在任何时候，毫不费力。在编译语言中没有等价的东西。Ast 是另一个核心库。它允许处理抽象语法树，即代码的符号表示。例如，这使得 Python 代码能够实时转换成其他语言，如 C/JavaScript/OpenCL，如我的项目[https://github.com/mdipierro/ocl](https://github.com/mdipierro/ocl)，Cython 和 Numba。

你还有什么想说的吗？

我喜欢 Python 有很多原因。一个是简单的语法。另一个是它的数字库(numpy，scipy，sympy，matplotlib，scikit learn)的强大功能。另一个原因是，它是一种解释性语言，因此，它可以做神奇的事情。例如，它可以用 __code__ 自省，可以反编译并重新编译成其他语言。

如果你正在阅读这篇文章，可能是因为你知道我是 web2py 的首席开发者。你可能听说过。你所听到的可能与它不太相符。在 web2py 中，我们试图做一些独特的事情，在许多方面，我们采用了一种与其他 Pythonic web 框架非常不同的方法。一个独特的特征是它的哲学:在 web2py 中,“不要重复自己”胜过“显式比隐式好”,因此一切都有默认值。这意味着代码非常紧凑，但并不意味着代码不能定制。另一个特性完全包含了电池:我们为 web2py 提供了一个 sqlite、一个数据库抽象层、一个支持 SSL 的多线程 web 服务器、基于 web 的 IDE、用于记录错误的标签系统、缓存库(redis、memcache)、登录方法(ldap、pam、openid 等)。)、表单小部件、验证器、支付系统等。Web2py 在许多方面都是第一个，它基于 WSGI，具有自动数据库迁移，具有强制 CRSF 保护，并且自 2007 年以来就有了基于 Web 的 IDE。我们仍然向后兼容 2007 版本。然而，我们已经取得了很大进展。例如，我们对基于 Tornado 的 websockets 和异步后台任务队列的支持已经有 5 年了。底线是:也许你应该再看一眼 web2py，也许你可以帮助我们把它做得更好。

感谢您接受采访！