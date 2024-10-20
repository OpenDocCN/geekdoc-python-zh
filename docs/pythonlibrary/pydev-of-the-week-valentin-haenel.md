# 本周 PyDev:瓦伦丁·海内尔

> 原文：<https://www.blog.pythonlibrary.org/2019/06/10/pydev-of-the-week-valentin-haenel/>

本周我们欢迎瓦伦丁·海内尔([@ ESC _ _](https://twitter.com/esc___))成为我们本周的 PyDev！瓦伦丁是 [Numba](http://numba.pydata.org/) 和其他几个软件包的核心开发者，你可以在他的[网站](http://haenel.co/)或 [Github](https://github.com/esc) 上看到。他还在欧洲的各种会议上做了几次演讲。让我们花些时间来更好地了解瓦伦汀吧！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我在爱丁堡大学获得了计算机科学学士学位，在柏林的伯恩斯坦中心获得了计算神经科学硕士学位。如今，我倾向于更传统的计算机科学主题，如压缩算法和编译器。在业余时间，我会和我可爱的妻子格洛丽亚在一起，放风筝，在柏林玩长板。我在 Github 上做 Python 和开源大概有 10 年了。

**你为什么开始使用 Python？**

我第一次开始使用 Python 是在我的硕士项目中。Python 过去和现在都是计算神经科学中非常流行的，既用于对 EEG 和 fMRI 等传感器数据进行机器学习，也用于模拟神经模型和神经元网络。我以前一直在使用 Java，花了一些时间来适应动态(鸭)打字风格。作为学术工作的一部分，我接触了早期的科学堆栈，当时主要由 Numpy、Scipy、Matplotlib 和命令行 IPython shell 组成。那时我最早的一些 Python 作品仍然存在。我做了一个项目，用一种特定的模型来模拟尖峰神经元:

[https://github.com/esc/molif](https://github.com/esc/molif)——这是我的第一个 github 回购。

也是从那时起，我的第一个包进入了 Debian，一个到特定类型的硬件光度计的 Python 接口。事实上，我刚刚在这台 Ubuntu 机器上检查过(2019 年 3 月)，该包仍然可用:

```py

$ apt search pyoptical
Sorting... Done
Full Text Search... Done
python-pyoptical/bionic,bionic 0.4-1.1 all
python interface to the CRS 'OptiCAL' photometer
```

:)

你还知道哪些编程语言，你最喜欢哪一种？

我知道一点 C，shell，go 和 Java，但是 Python 是我最喜欢的。我的一个朋友正在从事一个名为“@”的秘密编程语言项目，其目标是...良好的...仅限运行时——非常有趣。

你现在在做什么项目？

我现在正在为 Anaconda Inc .开发 [Numba](https://numba.pydata.org/) ，除此之外，我还在开发 Blosc(http://blosc.org/)，包括 [python-blosc](http://python-blosc.blosc.org/) 和 [Bloscpack](https://github.com/blosc/bloscpack) 。除此之外，还有几个规模较小但有点受欢迎的项目，是我自己运行的，即 [wiki2beamer](https://wiki2beamer.github.io/) 、 [git-big-picture](https://github.com/esc/git-big-picture) 、 [conda-zsh-complation](https://github.com/esc/conda-zsh-completion) 和 [yadoma](https://github.com/esc/yadoma) 。最近，我对时间追踪越来越感兴趣，并开始使用和贡献[freld](https://faereld.readthedocs.io/en/latest/)。

哪些 Python 库是你最喜欢的(核心或第三方)？

我一直对制作命令行界面感兴趣。为了这个任务，我查阅了许多库，如 getopt、optparse、argparse、bup/options.py、miniparser、opster、blargs、plac、begins 和 click(我忘记了吗？！).然而，有一个图书馆是我经常回来的，也是我最推荐的，那就是 docopt:http://docopt.org/。关于把你的命令行界面设计成一个程序概要，然后从中获得一个完全成熟的解析器，有一些东西要说。对我个人来说，这是构造命令行参数解析器最快、最自然、最直观、最方便的方法。如果你还没有意识到，你一定要去看看！

你是如何和 Numba 扯上关系的？

我在 Anaconda Inc .看到一个软件工程职位的空缺，主要在 Numba 上做开源工作。在编译器上做底层工作是我的拿手好戏，也是我很久以来一直想做的事情。我申请了，他们给了我一份工作，剩下的就是历史了。

你能解释一下为什么你会使用 Numba 而不是 PyPy 或 Cython 吗？

Cython 是 Python 的超集，它有额外的语法，允许静态类型，然后编译代码以 C-speed 运行，也就是“cythonize”代码。该代码不能再作为常规 Python 代码运行。Numba 的侵入性比这小得多，但有类似的目标。它提供了`@ jit` decorator，允许 Numba 使用 LLVM 编译器基础设施执行即时(jit)类型推理和编译。重要的是，它在 Python 字节码上这样做，不需要注释任何类型，代码可以像普通 Python 一样继续运行(一旦注释掉`@ jit` decorator。)这样做的好处是，您可以将可移植的数字代码作为纯 Python 发布，只有 Numba 作为依赖项，这将大大减少您的打包和分发开销。Cython 和 Numba 传统上都用于科学领域。这是因为它们与现有的生态系统、原生库(Cython 甚至可以与 C++接口，而 Numba 不能)交互良好，并且被设计为对 Numpy 非常敏感。所以这些是你在那个领域工作时会用到的:例如机器学习和广义上的任何科学算法和模拟。另一方面，PyPy 传统上对整个科学堆栈没有很好的支持。现在(2019 年初)稍微好一点，因为 Numpy 和 Pandas 都可以编译，并且已经做了很多工作来使
c-extensions 在 PyPy 中工作。

无论如何，PyPy 的主要目标集中在超越 CPython(Python 解释器的 C 实现)作为 Python 程序的基础，它正在缓慢但肯定地实现。

因此，总而言之:PyPy 是 Python 语言的未来，但它还没有为数据密集型应用做好准备。如果你今天想拥有尽可能高的计算效率，那么 Numba 和 Cython 都是不错的选择。Numba 非常容易试用——只需修饰一下您的瓶颈——众所周知，它可以将代码加速一到两个数量级。

对于想开始帮助开源项目的新人，你有什么建议？

去给自己找个痒处；找到一个你最喜欢的语言的项目，你觉得对你个人有用，然后改进它。然后，把你的改变反馈回来。很有可能，如果它对你有用，对其他人也会有用。此外，因为它对你个人有用，你可能会继续为它做出贡献，因为你最终会在其中获得既得利益。所以很明显，个人工具是寻找这种工具的一个很好的类别。找到对你日常有用的东西，并为此做出贡献。此外，不要害怕公开你的代码，如果你的贡献被拒绝也不要气馁，你才刚刚开始你的旅程，所以继续前进。祝你好运！

你还有什么想说的吗？

非常感谢所有开源/自由软件开发者和贡献者。我很自豪能成为这个奇妙的、鼓舞人心的社区的一员。

瓦伦丁，谢谢你接受采访！