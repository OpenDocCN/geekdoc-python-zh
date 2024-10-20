# 本周 PyDev:亨利·施雷纳三世

> 原文：<https://www.blog.pythonlibrary.org/2022/02/28/pydev-of-the-week-henry-schreiner-iii/>

本周，我们欢迎亨利·施雷纳成为我们的本周人物！Henry 是与 scikit、PyPA 等相关的多个项目的开源维护者/核心贡献者。你可以通过查看亨利的 [GitHub 简介](https://github.com/henryiii)来感受一下他在做什么。亨利还创建了一个在线免费的“[提升你的 Python 水平](https://henryiii.github.io/level-up-your-python/notebooks/0%20Intro.html)”课程。

![Henry Schreiner](img/5e0f01f10776c161d5279879d7ca15eb.png)

让我们花些时间更好地了解亨利吧！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我是一名计算物理学家和研究软件工程师，在普林斯顿大学工作。我与 [IRIS-HEP](https://iris-hep.org/) 合作，这是一个高能物理(HEP)可持续软件的多大学研究所。我们正在开发支持 HEP 数据重现性分析的软件。我在几个领域工作，专注于分析系统的底层软件栈和打包；当我不这样做的时候，我会使用直方图相关的工具。我还参与了创新算法的开发、教学和推广工作。

业余时间做很多 OSS 的贡献，所以其他爱好大部分都是可以和家人一起做的事情。我已经开始向我 5 岁的孩子介绍过山车；我喜欢电脑建模和动画，以及特效，但没有太多的时间去做了。

**你为什么开始使用 Python？**

大概是上大学的时候，开始用[搅拌机](https://www.blender.org/)；这是我对开源社区的介绍，并很快成为我最喜欢的程序。它有一个内置的 Python 解释器，因为它我真的很想学习使用 Python。这让我真的很想学 Python。在 2008 年西北大学本科生(REU)的研究经历中，我有机会在几个集群上工作——提交工作是一件痛苦的事。我用 Python 来监控和提交作业；这使我能够多线程化，比旧的 bash 脚本做更多的事情；我可以在别人手动操作之前捡起空节点。我离开的时候，每个人都想要我的剧本。

在 UT Austin 攻读高能物理学博士学位期间，我重写了一个大型 Matlab 分析代码库，然后慢慢地将它转移到 Python 上。新代码可以在任何地方的任何机器上运行，包括在伯利兹的丛林中，在那里我没有 Matlab 许可证。在我的毕业工作接近尾声时，我获得了 Plumbum 发布经理的职位，这是我第一次作为一个持续的开源项目的维护者。在欧洲粒子物理研究所开始做博士后后，我发现了一个不断增长的 Python 分析师社区，所以从那以后我就一直参与 Python——经常坐在 C++ / Python 的边界上或者使用 Python。我帮助启动了 Scikit-HEP，这是一个针对 HEP 的 Python 包集合。

 **你还知道哪些编程语言，你最喜欢哪一种？**

我是从 C++开始的；我写了 [CLI11](https://github.com/CLIUtils/CLI11) ，一个流行的 C++命令行解析器，我是 [pybind11](https://pybind11.readthedocs.io/) 的维护者。我喜欢这种语言每三年一次的变化，但也对在 Python 中使用它的困难感到沮丧——由于失去了许多 linux 所基于的 CentOS LTS 版本，我们在工具链中的 C++标准支持方面倒退了，而不是前进了。

我懂一点 C 语言，但我不擅长，而且打算一直这样下去；我有点太喜欢面向对象编程了。我也非常喜欢 CMake，从技术上来说，它也是一门语言。我非常喜欢红宝石；我用它来做 Jekyll 和 Homebrew 这就像“没有训练轮的 Python”，我喜欢它让你做的事情——成为一名伟大的厨师更容易使用锋利的刀，即使它们很危险。我也写过很多 Matlab，但是好几年没用过了。我知道一些 Lua，主要是为了 LuaLaTeX，但也是为了一些研究工作——设计一种嵌入应用程序的微小语言是一个非常酷的想法——很像 Blender 使用 Python 的方式。

由于社区、范围和支持，Python 是我的最爱。如果我要选择下一门语言，我会在 Rust 和 Haskell 之间左右为难——但是现在，我可能会选择 Rust。它正在变成一种编写 Python 扩展的伟大语言。

**What projects are you working on now?**

For work, I work on [boost-histogram](https://github.com/scikit-hep/boost-histogram) / [hist](https://github.com/scikit-hep/hist), [vec<wbr>tor](https://github.com/scikit-hep/vector), [awkward-array](https://awkward-array.org/), [particle](https://github.com/scikit-hep/particle), [<wbr>DecayLanguage](https://github.com/scikit-hep/decaylanguage), [Scikit-HEP/<wbr>cookie](https://github.com/scikit-hep/cookie) and other packages in [Scikit-HEP](https://scikit-hep.org/). We have 30-40 packages at this point, and I help with at least the packing on many of them. I also work on training materials, like [Modern CMake](https://cliutils.gitlab.io/modern-cmake), [Level Up Your Python](https://henryiii.github.io/level-up-your-python), and several minicourses, and the [Scikit-HEP developer](https://scikit-hep.org/developer) pages. As a mix of work and free time, I work on [cibuildwheel](https://cibuildwheel.readthedocs.io/), [pybind11](https://pybind11.readthedocs.io/), [bui<wbr>ld](https://pypa-build.readthedocs.io/), [scikit-build](https://github.com/scikit-build/scikit-build), and [GooFit](https://github.com/GooFit/GooFit). In my free time, I work on [CLI11](https://github.com/CLIUtils/CLI11) and [plumbum](https://plumbum.readthedocs.io/en/latest). I also blog occasionally on [iscinumpy.dev](https://iscinumpy.dev/). I also contribute to various OSS projects.

**Which Python libraries are your favorite (core or 3rd party)?**

Many of my favorite projects I ended up becoming a maintainer on, so I'll just focus on ones I am not officially part of.

**[Pipx](https://pypa.github.io/pipx)** is a fantastic tool that now lives alongside pip in the Python Packaging Authority (PyPA). A lot of time is spent trying to teach new Python users to work with virtual environments, and version conflicts are being more common (due to over use of pre-emptive capping, a pet peeve of mine); but pipx skips all that for applications - you can just use pipx instead of pip and then version conflicts and the slow pip update solves just go away. I *really* like `pipx run`, which will download and run an application in one step, even on CI; GitHub Actions & Azure provides it as a supported package manager, even without `actions/setup-python` - perfect for easy composite shell actions (like cibuildwheel's)! `pipx run` even caches the environment and reuses it if it's less than a week old, so I no longer have to think about what's installed or what's out-of-date locally, I just use `pipx run` to access all of PyPI anywhere (that I have pipx, which is everywhere). (I'm a homebrew macOS user, so `pipx install` - or any install doesn't work well with the automatic Python upgrades, but pipx run works beautifully.)

I used to dislike tox - it had a weird language, bad defaults, ugly output, and didn't tell a user how to run commands themselves if they wanted to set up things themselves. While Tox4 is likely better, I've really loved **[Nox](https://nox.thea.codes/)**. It (intentionally) looks like pytest, it doesn't hide or assume anything, it works for much more than packaging - it's almost like a script runner with venv (and conda/mamba) support, with pretty printouts.

Getting away from the common theme of packaging above, I also love pretty-printing and color, so I'll have to call out the Textualize libraries, **[Rich](https://rich.readthedocs.io/)** / **[Textual](https://github.com/Textualize/textual)**; they are beautiful.

For the standard library, I love **contextlib**; context managers are fantastic, and a bit underused, and it has some really nice newer additions too.

**How did you end up working on so many Python packages?**

I got involved with Scikit-HEP at the beginning, and there we quickly collected older packages that were in need of maintenance. Working on a large number of packages at the same time helps you appreciate using common, shared tools for the job, rather than writing your own. It also forces you to appreciate packaging. Many of the packages I work on are used heavily by the code I started with.

Besides, show anyone that you can help them with packaging and they will usually take you on in a heartbeat. 🙂

**Of the Python packages, you have worked on or created, which is your favorite and why?**

Each package is for a different use, it's hard to choose a favorite. I have a reason to like and be involved in all of them. Probably my favorite project was the most different from what I normally do - the [Princeton Open Ventilation Monitor](https://github.com/Princeton-Penn-Vents/princeton-penn-flowmeter) project. In early 2020, a group of physicists, engineers, and scientists got together and developed a device to monitor airflow in ventilator systems, initially working with our local hospitals. I developed both the backend software, the graphical interface, and the on-device interface too, while Jim Pivarski (of [Awkward-Array](https://awkward-array.org/)) developed the breath analysis code. It was an incredibly intense month for all of us, but in the end we had a great device and a really powerful multi-device software system (which is now all open source with open access designs). It was really fun to work on something that was not a library; I got to design for Python 3.7 instead of 2.7+ (3.6+ today), and I worked with things I wouldn't normally get to, like PyQT, line displays and rotary controls, and lots of threading. This is also where I properly learned to use static typing & MyPy, which was critical in writing code for hardware that wasn't even built yet.

I have other exciting things planned that might take that "favorite" title. I'm hoping to get the chance to [rewrite scikit-build](https://iscinumpy.dev/post/scikit-build-proposal). I'm planning on using [rich](https://rich.readthedocs.io/), [textual](https://github.com/Textualize/textual), and [plotext](https://github.com/piccolomo/plotext) to make a HEP data browser in the terminal - which would also be an "app".

**Is there anything else you’d like to say?**

Don't undervalue consistency, readability, and static analysis, which makes code easier to read and maintain with less effort, and often helps keep bugs down. *Reading* code *that is not yours* is incredibly important skill, as is packaging, so you can use code others wrote without rewriting it yourself. Tools like pre-commit, mypy, and nox really help code be more accessible. If you make choices that seem to help one specific case, that is almost never worth the loss in consistency which helps others easily digest your code and either use it or even contribute to it. Providing a noxfile can really help "fly-by" contributors!

It's okay to abandon a project (azure-wheel-helpers, in my case) when you find a library (cibuildwheel) that is better than yours, and instead contribute to that. By helping them, you can help a larger audience, and avoid duplicating work.

I'd highly recommend reading [scikit-hep.org/<wbr>developer](https://scikit-hep.org/developer) (with an accompanying [cookiecutter](https://github.com/scikit-hep/cookie)!) if you are developing code, even if you are not developing in HEP or even scientific fields. I also contribute to [packaging.python.org](http://packaging.python.org/), but I'm a lot more free to be opinionated there and recommend specific workflows and tools.

**Thanks for doing the interview, Henry!**