# 本周 PyDev:达明·阿维拉

> 原文：<https://www.blog.pythonlibrary.org/2017/12/18/pydev-of-the-week-damian-avila/>

本周我们欢迎达明·阿维拉成为我们的本周 PyDev！Damian 为 Anaconda 工作，这是一个 Python 和 R 的开源发行版，主要关注数据科学。他也是 Jupyter/IPython 幻灯片扩展《T2 崛起》的作者。你可以在 [Github](https://github.com/damianavila) 上或者通过查看他的[网站](http://www.damian.oquanta.info/)来感受一下达明在做什么。让我们花些时间来更好地了解我们的同胞 Pythonista！

[![](img/1b97aa2356faed1349a7c11a107beb57.png)](https://www.blog.pythonlibrary.org/wp-content/uploads/2017/12/damian_avila.jpg)

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

当然可以！

如今，我是一名软件开发人员，也是 Anaconda 公司的团队领导。在我之前的生活中，我作为生物化学家从阿根廷布宜诺斯艾利斯的联合国大学毕业。

事实上，我目前与我的伴侣丹妮拉和我的猫 Bituinas 住在阿根廷的科尔多瓦，但每隔一周我都会去布宜诺斯艾利斯(700 公里外)，我 5 岁的儿子 Facundo 和他的母亲住在一起，我会和他呆几天。

我的主要兴趣是数据科学、金融、数据可视化和 Jupyter/IPython 生态系统。在开源领域，我为几个项目做出了有意义的贡献，现在是热门项目的核心开发者，比如 [Jupyter](http://jupyter.org/) 、 [Nikola](https://getnikola.com/) 和 [Bokeh](https://bokeh.pydata.org/en/latest/) 。

我也开始了自己的项目，成为最受欢迎的项目。

我在几个国家和国际会议上发表过演讲、教程和海报，我还撰写和教授过关于科学 Python 生态系统的教程。我是 Python Argentina、Scientific Python Argentina 和 PyData Argentina 社区的成员，也是 Jupyter 指导委员会的成员。

有趣的事实是，我练习合气道已经好几年了，现在我正试图回到常规练习中来。

**你为什么开始使用 Python？**

这个回答起来很有趣。几年前，大约 2011 年，我把玩金融模型作为一种爱好(奇怪的爱好，不是吗？).我发现自己需要创建比我使用的终端用户软件提供的模型更复杂的模型(IIRC，我使用的是一个叫 EasyReg 的软件)。然后，我告诉自己，你应该开始学习如何制作自己的模型了！！

我发现了一门关于“科学家的数值方法”的有趣课程，我们有 3 节 4 小时的入门课，内容是关于 3 种编程语言:Fortran、C++和...Python。我发现 Python 很容易上手(别忘了我是一名生物化学家，研究的是免疫系统的细胞，没有计算机，全是实验性的东西，没有编程的东西)。因此，我从零开始尝试用 Python 重新创建我的金融模型，并在 2011 年底参加了我的第一次阿根廷 PyCon，发表了一篇关于我的模型的演讲。

你还知道哪些编程语言，你最喜欢哪一种？

*   我一直在用其他语言编程，比如 JavaScript，CoffeeScript，还有一点 TypeScript。
*   我经常使用 bash。
*   我玩了很多 HTML 和 CSS。
*   我学过一点 Fortran 和 C++。

我最喜欢的还是 Python，我开始觉得 TypeScript 很有趣。

你现在在做什么项目？

在我的工作中，我领导着一个团队在 Anaconda Enterprise Notebooks(Anaconda 企业笔记本)中工作，这是 Anaconda 平台 v4 的一个组件。在我的工作之外，在我的空闲时间，我通常为 Jupyter 生态系统做贡献。我也在 [RISE](https://github.com/damianavila/RISE) 中工作过，这是我几年前写的一个流行的笔记本扩展，被很多人使用(可能比我知道的多，这很好！！).

哪些 Python 库是你最喜欢的(核心或第三方)？

我在这里有点偏见，但我相信 Jupyter 生态系统库是我最喜欢的，Jupyter 笔记本应用程序是首选。它完全是破坏性的，在多个领域都是有用的。我相信它演变成[木星实验室](https://youtu.be/w7jq4XgwLJQ)会是一种“革命性的”。

**你创造 RISE 套餐的动机是什么？**

我认为“展示”你的想法是很重要的，即使笔记本本身是一个可以实现这一目标的文件，它也不太适合在讲座或课堂上分享信息。幻灯片演示的概念在这种情况下有很多优势，我想把这些优势带到 Jupyter 的体验中。我的第一个想法是使用 [nbconvert](http://nbconvert.readthedocs.io/en/latest/usage.html#convert-revealjs) 从笔记本文档生成一个静态演示，这是几年前在那个包中实现的。

但是，我也看到了拥有笔记本体验带给您的交互性的需求，这就是我开始 RISE 的原因，它本质上是笔记本的一个“slidy”视图，您可以在幻灯片中立即执行代码和实验内容，我认为这在当时非常强大，可以分享知识和见解。

在维护这个项目的过程中，你学到的最重要的三件事是什么？

1)人是伟大的，我总是从我的工作中得到赞赏，看到许多人认为它有用，我感到非常有价值。

2)你是免费给你时间，你应该关心你的用户，但是你的用户也应该关心你。不要让项目维护耗尽你的精力。如果有必要的话，抽出一些时间。

3)当你和人们互动时，试着给他们提供食物。如果你给他们背景、反馈并讨论他们的想法，你会有一个新的贡献者。

你还有什么想说的吗？

作为 Jupyter 社区的一员，我想鼓励你帮助我们！有多种方法可以帮助这个项目(不仅仅是编码！)，所以 [ping 我们](http://jupyter.org/community.html)如果你有一些兴趣和时间。

最后，正如你从我之前的回答中看到的，我经历过，你永远不知道生活会把你引向何方。谦虚学习...一切都会水到渠成。

感谢您接受采访！