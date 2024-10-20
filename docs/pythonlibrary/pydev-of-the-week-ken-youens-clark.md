# 本周 PyDev:肯·尤恩斯·克拉克

> 原文：<https://www.blog.pythonlibrary.org/2021/01/11/pydev-of-the-week-ken-youens-clark/>

本周我们欢迎 Ken Youens-Clark([@ kycl 4 rk](https://twitter.com/kycl4rk))成为我们的本周 PyDev！他是来自曼宁的[小 Python 项目](https://amzn.to/3ngOe3L)的作者。他已经在 T4 的 YouTube 上为他的每一章做了视频演讲。

![Ken Youens-Clark](img/45a71949696ff30e9cb5b595602780af.png)

让我们花些时间来更好地了解肯！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我是打鼓长大的，最初在大学里学了几年音乐专业。我在大学里换了几次专业，最后拿到了英国文学学士学位，辅修了音乐。大学毕业后，我开始在乐队中演奏电贝司和竖式贝司，在过去的几年里，我最喜欢弹钢琴、小提琴、鼓，有时也弹博德兰鼓。我喜欢烹饪，尤其是为我的家人烘焙，我和我的妻子喜欢看“伟大的英国烘焙秀”和阅读烹饪书籍来寻找灵感。

我从来没有学过计算机，也是边工作边学编程。我小时候有一台 TRS-80，但我可能从未在上面写过超过一百行的 BASIC 语言。我拿工资写的第一门语言是 Windows 3.1 上的 Visual Basic。90 年代末，我曾是一名桌面 Windows 程序员，后来当我转而从事 Unix 和 Perl 工作时，我迷上了“互联网”这个东西。这让我在一个基因组学实验室找到了一份网络开发员的工作，这份工作变成了我在生物信息学领域的职业。

我于 2015 年开始在亚利桑那大学(UA)攻读硕士学位，20 年后完成本科学位，2019 年毕业。

**你为什么开始使用 Python？**

在 UA 的 Bonnie Hurwitz 博士的实验室工作时，我有幸帮助她向生物学家和工程师教授初级编程技能。从 2015 年开始，我们使用 Perl，因为这是我们最喜欢的语言，它在生物信息学中被广泛使用。

几年后，很明显 Python 会是更好的选择。Python 的语法很简单，它在科学计算领域已经超越了 Perl，并且有更多的工作等待着受过训练的 Python 程序员。

2017 年左右，我开始将我所有的培训材料转换为 Python，并全职转向 Python 编码，以便精通。这种变化真的让我在机器学习的冒险中受益匪浅，我发现它是我从命令行程序到 web 后端的大部分日常开发的首选语言。

你还知道哪些编程语言，你最喜欢哪一种？

我的第一门语言是 Visual Basic 和 Delphi，我已经完全忘记了。我仍然可以破解 Perl，而且我经常写 bash，而且写得相当好。我使用 Elm，一种纯函数式语言，它是 Haskell 的子集，用于动态 web 前端，我也非常喜欢在 Rust 中工作，我不得不说这可能是我现在最喜欢的。

你现在在做什么项目？

我目前正在为 O'Reilly 写一本新书，名为《用 Python 再现生物信息学》。我希望它能在 2020 年底前提前发布，它应该会在 2021 年出版。所有的代码和测试都可以在 https://github.com/kyclark/biofx_python 找到。

几个月前，我在亚利桑那州图森市的关键路径研究所开始了一个新的职位，目前我正在做 clinicaltrials.gov 网站的内部镜像。该项目涉及到后端的 Python、前端的 Elm 以及一些有趣的关系数据库(如 Postgres)和非关系数据库(如 MongoDB)。

哪些 Python 库是你最喜欢的(核心或第三方)？

对于命令行编程，我非常依赖“argparse”来处理参数。在编写代码时，我使用“yapf”(另一个 Python 格式化程序)来格式化代码，使用“ylint”和“flake8”来检查格式和样式，使用“mypy”来检查类型注释，使用“pytest”进行测试。

没有正则表达式我无法工作，所以我经常使用“re”模块。我经常连接 SQLite、MySQL、PostgreSQL 和 MongoDB 等数据库，所以这些模块对我来说非常重要。我认为“FastAPI”模块对于编写后端 API 是必不可少的。

从风格上来说，我真的很喜欢使用“itertools”和“functools”来编写更像纯函数式语言的 Python。

你是如何成为曼宁的作者的？

在将我的教学材料从 Perl 转移到 Python 并教授了几年之后，我决定我有足够的材料来出版一些新奇的东西，特别是因为似乎没有人尝试向初学者教授测试驱动开发。最初，我带着一个非常混乱的想法去找奥赖利，想写一本生物信息学的书，将游戏和谜题与生物学的例子结合起来。

我被告知要把这些分开，非生物信息学的材料变成了[微型 Python 项目](https://www.manning.com/books/tiny-python-projects)。

作为一名作者，你学到的最重要的三件事是什么？

我真的很喜欢课堂教学，我也知道了我在写作中遗漏了多少，因为我会在课堂上即兴发挥，用我的手和黑板。我已经学会了创建大量的图表，并且不需要任何知识。

我还学会了利用课堂时间和学生一起编写实例，而不是讲课。我发现这对学生来说更有吸引力。他们实际上学会了如何从头开始编写程序，只需添加一两行代码，然后运行并测试程序。

这导致了我如何试图围绕一个程序来写我的书的每一章，使用它作为一个工具来教授一些概念，如正则表达式和文件处理。

我试图教大家如何从一个模板开始每个程序，并一步一步地修改它以满足给定的测试套件。我认为这导致了章节的集中和独立，让读者在每一章的结尾都有一种写了真正的程序的感觉。最后，我学到了很多关于包容性写作的知识。

我认为作为一个中年、异性恋、白人男性的生活给我留下了很多无意识的偏见，有些编辑用一些不欢迎我的语言来称呼我。我认为可以让文章变得轻松的小笑话可能会很糟糕。我甚至开始把“只是”或“简单”这样的简单词汇视为可怕的精英主义，比如“只是写算法”或“答案是简单地做 x。”

学习任何东西都具有挑战性，所以我尽量避免任何让编程看起来容易或轻松的语言。

你还有什么想说的吗？

我一直很喜欢 Perl 社区的友情，我发现 Python 也同样友好和支持。在 Perl 中，我非常依赖 CPAN，并设法贡献了一些我自己的模块，我发现 Python 的 pypi 也是一个同样了不起的资源。

我非常喜欢 Python，但我也认识到这种语言会让你犯严重的错误，所以我强调在“mypy”和大量测试中使用类型注释的必要性。这一直是我的课堂教学和我的两本书的中心焦点。

肯，谢谢你接受采访！