# 本周 PyDev:杰里米·图鲁普

> 原文：<https://www.blog.pythonlibrary.org/2022/04/04/pydev-of-the-week-jeremy-tuloup/>

本周，我们欢迎杰里米·图鲁普( [@jtpio](https://twitter.com/jtpio) )成为我们本周的 PyDev！Jeremy 是 Jupyter 项目的核心开发人员。你可以在杰瑞米的 [GitHub 简介](https://github.com/jtpio)上看到他的其他贡献。杰瑞米偶尔也会在他的[网站](https://jtp.io/)上发布文章。

让我们花些时间来更好地了解杰里米吧！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

大家好，我是 Jeremy，QuantStack 的技术总监和开源开发人员。

我拥有 INSA 里昂大学的计算机科学和工程硕士学位。

在加入 QuantStack 之前，我在视频游戏行业工作。首先是在瑞典的 Ubisoft Massive，开发休闲和 AAA 游戏，主要是游戏性和游戏引擎开发。然后在柏林 Wooga 做手机游戏，重点是数据工具。

作为一种爱好，我真的很喜欢长跑，在山里徒步旅行几天或几周。这是一个伟大的方式断开一点和充电电池！

**你为什么开始使用 Python？**

在我第一次实习期间，我主要是用 Java 编程语言为 Android 开发移动应用程序。

另外，我对探索几个不同的计算机科学主题很感兴趣。我觉得我需要一种能让我更好地表达自己的语言。Python 一开始看起来足够简单，非常平易近人。所以我决定学习它，事实证明这是一个伟大的选择！

你还知道哪些编程语言，你最喜欢哪一种？

在前端开发方面做了很多工作，我也使用 TypeScript。类型系统极大地帮助了对代码的推理和快速捕捉错误。此外，这是对 JavaScript 的一个很好的增强，使处理大型代码库变得更简单、更易于管理。

你现在在做什么项目？

作为一名开源开发者，我的工作主要集中在 Jupyter 项目上。我帮助维护几个子项目，如 JupyterLab、Jupyter Notebook 和 Voila Dashboards。

最近我一直在关注基于 JupyterLab 组件的笔记本 v7 过渡。笔记本 v7 将是流行的 Jupyter 笔记本的下一个主要版本。笔记本是 Jupyter 生态系统的主要支柱之一，参与这个项目真的很令人兴奋！

哪些 Python 库是你最喜欢的(核心或第三方)？

这可能不是一个真正的库，但我真的想为 Pyodide 项目大声疾呼。Pyodide 更像是一个 Python 发行版，而不是一个库。它是编译到 WebAssembly 的 CPython，包括一些流行的数据科学包，如 numpy 和 pandas。

由于编译成 WebAssembly，Pyodide 可以在浏览器中运行，所以你让 Python 在浏览器中运行！这就是 JupyterLite(见下文)用来提供在浏览器中运行的交互式数据科学环境。虽然在 CPython 和包装故事(利用 conda forge 基础设施)的上游还有一些工作要做，但这已经是一个很好的开始，为未来打下了良好的基础。

你是如何成为核心 Jupyter 开发者的？

我是 Jupyter 的长期用户。我开始使用 Jupyter 来跟踪一些针对在线文章和博客帖子的个人阅读习惯。我想知道在给定的时间内我可以处理多少文章，并试图找到一些模式。

Then I started to use Jupyter more at work for generating reports on how our game was performing while working at Ubisoft Massive. At my previous job we also had a Data Science team using JupyterHub and internal extensions to perform analysis on game performances.

Progressively I started submitting bug fixes to the upstream projects such as JupyterLab and Voila, and progressively learned more about the various projects and their codebases. Over time and after a couple of months of contributions, I started to get commit rights on a couple of projects, making me feel part of the Jupyter community even more.

Later I joined QuantStack and started to work full-time on these open source projects, making much bigger contributions and helping with maintenance and releases.

**What parts of Jupyter's core do you find most interesting and why?**

The strength of Jupyter is the coherent set of standards and protocols around it.

I find very interesting the fact that we can innovate very fast while still building around that protocol. This is for example the case with the relatively new JupyterLite project, which runs a full Jupyter distribution in the browser including a Python kernel backed by Pyodide. This project is a great example of reusing existing work and components, and interfacing them in a different way to produce something useful and new.

**Thanks for doing the interview, Jeremy!**