# 与 Kenneth Reitz 的 Python 社区访谈

> 原文：<https://realpython.com/interview-kenneth-reitz/>

本周，我很兴奋能采访多产的肯尼斯·雷兹！

肯尼斯是极受欢迎的 [`requests`](http://docs.python-requests.org/en/master/) 和 [`pipenv`](https://pipenv.readthedocs.io/en/latest/) 库的作者。加入我们，一起讨论他的最新项目和他迄今为止编写的最具挑战性的代码。

**Ricky:** *从头说起吧……你是怎么进入编程的，什么时候开始用 Python 的？*

![Kenneth Reitz](img/6cfefb1267cfd0c31e551bc48f896c50.png)

肯尼斯:我很小的时候就开始编程了。我爸是程序员，我 9 岁就自学了 BASIC 和 C(在这个帮助下)。我在大学开始使用 Python，当时我上了第一门 CS 课程。不久后，我辍学并学习了许多其他编程语言，但我总是不断回到 Python。

**瑞奇:** *祝贺你在[数字海洋](https://realpython.com/digital-ocean)的新工作。你是开发商关系团队的资深成员。你如何看待你在 Heroku 的前一份工作中的角色转变，我们对数字海洋在 Python 领域的发展有何期待？*

肯尼斯:谢谢！我真的很享受这个新角色，也很享受为整个开发社区服务的机会，而不仅仅是 Python 社区。然而，我的最新作品， [Responder](http://python-responder.org/en/latest/) ，是一个数字海洋项目，所以在 Python 领域我们有更多的期待空间😊

当然，你最出名的是编写了非常流行的`requests`库和新的`pipenv`库。Python.org 现在推荐使用`pipenv`进行依赖管理。社区收到了怎样的`pipenv`？你有没有看到来自社区的很多阻力，开发者更喜欢坚持`venv`或者更老的依赖管理方法？

**Kenneth:** 社区反响很好，甚至像 GitHub 这样的公司也在使用它的安全漏洞扫描标准。除了 reddit 上的一些仇恨之外，我根本没有看到来自社区的太多抵制。我花了一段时间才意识到 [/r/python](https://www.reddit.com/r/python) 与其说代表了 python 社区，不如说代表了使用 Python 的 redditors。

里奇: *现在在你的请求库上达到 3 亿次下载很酷，但作为一名吉他手，我更兴奋的是你的最新项目[皮瑟里](https://github.com/kennethreitz/pytheory)。你能告诉我们一些关于它和你对这个项目未来的计划吗？*

PyTheory 是一个非常有趣的库，它试图将所有已知的音乐系统封装到一个库中。目前，有一个系统:西方。它可以以编程方式呈现西方体系的所有不同音阶，并告诉您音符的音高(以十进制或符号表示法)。此外，还有指板和和弦图表，因此您可以为吉他指定自定调音，并使用它生成和弦图表。很抽象。

绝对是我写过的最有挑战性的东西。

**里基:** *所以在最近放弃你的 Mac 电脑，转而使用微软的 [VS Code](https://code.visualstudio.com/) 进行 Python 开发之后，作为一名 Windows 用户，你感到高兴和自豪吗？对于那些从 Windows 95 年开始就没用过 Windows 的读者来说，他们错过了什么？*

肯尼斯:我喜欢苹果电脑，比起 Windows 我更喜欢它。我只是目前对我的设置感到厌倦，并决定通过运行 Windows 来挑战自己。不过，我很开心，也很有收获。感觉就像在家一样。

Windows 已经不是过去的样子了。它现在是一个真正坚实的操作系统。我在我的 iMac Pro 上运行它，它像做梦一样嗡嗡作响。

里基: *我知道你是个热衷于摄影的人。你从事这项工作多久了，你拍过的最喜欢的照片是什么？除了 Python，你还有其他爱好和兴趣吗？*

肯尼斯:我认真地投入摄影已经有 10 年了。我拍过的最喜欢的照片大概是[这张](https://500px.com/photo/54603002/seasonal-harmonies-by-kenneth-reitz)。这是在我患了几个星期的偏头痛之后，用胶片相机拍摄的，那是我第一次能够在户外行走。

**里基:** *最后，有什么智慧的临别赠言吗？你有什么想分享和/或宣传的吗？*

**肯尼斯:**响应者！我的新 Python web 服务框架。它是 ASGI，看起来很熟悉，速度很快，而且比 Flask 更容易使用！非常适合构建 API。[看看吧！](http://python-responder.org/en/latest/)

* * *

谢谢你，肯尼斯，这周和我在一起。在 Twitter 上实时观看 Responder 的开发真是太棒了。你可以关注它的发展或者在这里提出一个问题。

一如既往，如果你想让我在未来采访某人，请在下面的评论中联系我，或者在 Twitter 上给我发消息。