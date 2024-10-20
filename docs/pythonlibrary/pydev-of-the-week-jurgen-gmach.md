# 本周 PyDev:于尔根·格马奇

> 原文：<https://www.blog.pythonlibrary.org/2022/06/20/pydev-of-the-week-jurgen-gmach/>

本周，我们欢迎尤尔根·Gmach([@ jug MAC 00](https://twitter.com/jugmac00))成为我们本周的 PyDev！于尔根是 [tox 自动化项目](https://tox.wiki/en/latest/)的维护者。你可以在他的[网站](https://jugmac00.github.io/)上看到于尔根还在忙些什么。你也可以在 [GitHub 或者](https://github.com/jugmac00) [Launchpad](https://launchpad.net/~jugmac00) 上查看 Jürgen 的代码。

![Jürgen Gmach](img/3305d3651d44e032e4f612ae6e2c1d21.png)

让我们花些时间来更好地了解于尔根吧！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

嗨，我是尤尔根。我是 Canonical 的软件工程师。我住在德国南部，就在美丽的多瑙河和巴伐利亚森林之间。

我从小就对计算机感兴趣，起初在我的 C64 Commodore 上玩电脑游戏，后来用 Basic 编写简单的应用程序。

学校里一位非常热情的老师激起了我对经济学的兴趣，所以我决定学习经济学。

在我学习的几年里，我越来越被这个叫做互联网的新事物所吸引。

我用 HTML 创建了网站，最引人注目的是一个相当成功的在线台球社区，后来我把它移植到 PHPNuke，为此我必须学习 PHP 以及如何编写补丁。

有一次，我决定我需要跟随我的心，所以我开始在一家当地公司做软件工程师。

闲暇时，我喜欢外出。根据天气和季节的不同，我喜欢徒步旅行、骑自行车、游泳或采蘑菇，有时独自一人，但大多数时候是和我可爱的家人在一起。

**你为什么开始使用 Python？**

我的第一份工程工作是受雇开发一个基于 Python 和 Zope 的大型内部网应用程序。所以我不得不在工作中学习 Python。

有一个关于这个技术堆栈的小背景故事。我当时的同事首先尝试用 Ruby 创建自己的应用服务器，但他的尝试总是失败，所以他一度选择了 Zope 和 Python。

从那以后 Python 就一直在我的生活中。

我永远欠我同事的情。

你还知道哪些编程语言，你最喜欢哪一种？

如前所述，我开始用 Basic 编程，我在大学时学习了 Bash 和 Pascal，在 CSS 和 JavaScript 出现之前我用 HTML 创建了静态网站，用 Perl 创建了动态网站，我用 PHP 创建了小型和大型网站，我用 Python、Rust、Bash 和 Go 创建了命令行应用程序，我编写并维护了相当多的 JavaScript，我使用 Java 或 C 对项目进行了修复，我调试了 Lua 和 Sieve 脚本，但我肯定最熟悉 Python，也是我最喜欢的。

你现在在做什么项目？

我于 2021 年 10 月加入 Canonical，从事 Launchpad 项目，该项目由许多部分组成，最著名的是一个类似于 GitHub 的代码托管系统，以及一个构建场，在那里可以为 Ubuntu 和其他系统构建所有优秀的包。

我的团队目前正在从零开始构建 CI 系统，这是一个超级有趣的任务。虽然我对所有涉及的系统都有贡献，但大部分时间，我都在 CI 运行程序上工作。最棒的是——这些都是开源的。

我还花一些业余时间从事多个开源项目。

那就是 tox，任务自动化工具，Zope 基金会的近 300 个项目，Morepath web 框架，Flask-Reuploaded，我把它们分出来，这样就不会被维护了。我也做了很多路过的贡献。

哪些 Python 库是你最喜欢的(核心或第三方)？

我当然不想在没有 [tox](https://github.com/tox-dev/tox) 的情况下维护 300 个 Zope 存储库，它提供了测试、运行 linters 和构建文档的标准接口。

说到 linters，我从来不会没有[预提交](https://github.com/pre-commit/pre-commit)和[片 8](https://github.com/PyCQA/flake8) ，以及更多取决于项目的内容。

当我需要创建一个命令行应用程序时， [argparse](https://docs.python.org/3/library/argparse.html) 是我的首选。我特别喜欢它的多功能性和它附带的标准库。

all-repos 是一个奇妙的利基应用程序和库，当我需要用一个命令更新几十个，或者在 Zope 的情况下，甚至几百个库时，我会使用它。我在 [PyConUS](https://youtu.be/5zEn3Jta2Dg?t=2040) 做了一个简短的介绍。

你是如何参与毒理项目的？

哦，这个很有趣。我甚至在“[测试 tox 4 预发布的规模](https://jugmac00.github.io/blog/testing-the-tox-4-pre-release-at-scale/)”中写了博客。

简而言之:
为了能够只用几个人维护 300 个 Zope 项目，我们需要统一的接口，所以我们使用 tox 进行测试。只需克隆它并运行“tox”——无需设置虚拟环境，无需阅读文档，无需修改测试路径。

由于 tox 的核心维护者 Bernát Gabor 在 Twitter 上宣布，他计划发布 tox 4，这将是一个完全的重写，我认为对所有 300 个项目运行 tox 4 alpha 是一个好主意。为此，我用全回购来做毒理分析。我发现并报告了几个边缘案例，有一次我试图自己修复其中一些——在 Bernát 的帮助下，效果相当不错。

因为我非常喜欢与 tox 一起工作，所以我不仅贡献代码，还在 [StackOverflow](https://stackoverflow.com/tags/tox) 上回答问题，并对新的 bug 报告进行分类。

一天，出乎意料的，Bernát 让我成为我最喜欢的开源项目的维护者！！！疯了！

作为一个开源包的维护者，你学到的前三件事是什么？

You cannot help everybody. Let's take tox as an example. It is not a big project, but with more than 5 million downloads per month, and thousands of users, two things regularly happen. Users ask questions that even I as a maintainer cannot answer, as maybe the original poster uses a different IDE, a different operating system, or some very specific software... but it is ok to not know everything.

Also, with so many users you will be asked to implement things that are only helpful for very few users on the one hand, and on the other hand, will make maintenance harder in the long term. So you need to learn to say "no".

Also, don't be the single maintainer of a project. It is much more fun with some fellow maintainers. You can split the workload, learn how others maintain a project, and most importantly, you can have a break whenever you want. Life happens! I am currently building a house for my family so I cannot spend too much time on my projects - but this is ok!

And finally, and maybe most important. Let loose. When you no longer feel joy in maintaining your project, pass it on to your fellow maintainers, or look for new maintainers. That is ok. It is also ok to declare your project as unmaintained. You do not owe anybody anything, except yourself.

I think the above things are not only valid for open-source projects, but also for work, and possibly also for life in general.

**Is there anything else you’d like to say?**

If I could give my younger self three tips, these would be:

Take down notes of things you learn, write a developer journal or even a public blog. That way you reinforce what you learn and you can always look it up later on.

Go to conferences!!! You will pick up so many new things, and most importantly, you will meet and get to know so many great people from all over the world.

Shut down your computer. Go outside and have some fresh air and some fun!

**Thanks for doing the interview, Jürgen!**