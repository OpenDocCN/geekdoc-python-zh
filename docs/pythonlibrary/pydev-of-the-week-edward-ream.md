# 本周 PyDev:Edward Ream

> 原文：<https://www.blog.pythonlibrary.org/2018/04/09/pydev-of-the-week-edward-ream/>

本周我们欢迎 Edward Ream 成为我们的本周 PyDev！Edward 是 Leo(**L**E**E**ditor with**O**ut lines)文本编辑器/大纲视图的创建者。他从 2001 年开始使用 Python。你可以看到爱德华在 Github 上做了什么。让我们花些时间去更好地了解他吧！

Driscoll:你能告诉我们一些关于你自己的情况吗(爱好、教育等)

**令:** â€‹Piano，行走，科学，文学，历史。我每周都读科学和自然。我鼓励你的读者也这样做。从阅读文章摘要和每篇文章的摘要开始。*它们是为非专业人士编写的。所有的科学家都是他们直接研究领域之外的外行人！*

我跟随迈克尔·布库斯-博米尔学习钢琴和理论已经很多年了。他对音乐几乎了如指掌。他是一位伟大的爵士乐和古典音乐演奏家。也是一个密友。

我经常做[大笑瑜伽](https://en.wikipedia.org/wiki/Laughter_yoga)。这无疑是我做过的最具变革性的事情，包括[霍夫曼过程](https://www.hoffmaninstitute.org/the-process/)和[里程碑式的教育论坛](http://www.landmarkworldwide.com/the-landmark-forum)。我发现笑有助于我与人交往，缓解社交场合的焦虑。

笑声对创造力有巨大的帮助。我清楚地记得一段“火腿与克莱德”在与 [J .克雷格·文特尔](http://www.jcvi.org/cms/home/)讨论[他们的作品](https://www.nature.com/news/minimal-cell-raises-stakes-in-race-to-harness-synthetic-life-1.19633)时大笑的视频。唉，找不到链接了。哦，对了，哈姆是诺贝尔奖获得者。

我经常散步来解决问题。诀窍是做白日梦。当任何相关的想法出现时，我会把它记在一个 89 美分的笔记本上。这个想法是忘记我刚刚写下的东西，这样新的想法就会出现。试图记住事情会扼杀进一步的白日梦。

丽贝卡是我的缪斯。她是一名园丁兼织布工，但当我和她谈论利奥时，她会问一些很棒的问题，甚至是特别技术性的问题。想想那有多棒！

在与她交谈中，我萌生了用《更多的局外人》作为(后来的)利奥原型的想法。很可能没有她这一切都不会发生。

Driscoll:你为什么开始使用 Python？

令:因为苹果违背了将 YellowBox 移植到 Windows 的承诺。所以目标 c 是不可能的。

我在 2001 年初开始学习 Python。那年我去了 DC 郊外的 Pycon，结果被大雪困住了。走在首都附近，看不到车水马龙，感觉很奇怪。

Driscoll:你还知道哪些编程语言，你最喜欢哪一种？

到目前为止，Python 是我最喜欢的。令人恼火的是，网络本可以基于 Python。

就在今年，我意识到这与 javascript 语言无关。是工具的问题。所以我在学习 node.js 和 vue.js 的世界，由 Leo 的 devs 驱动。

德里斯科尔:你现在在做什么项目？

**令:**狮子座是我的生命。[此页面](https://github.com/leo-editor/leo-editor/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+label%3Asummary)包含我的计划的摘要项目。â€‹At:现在克里斯·乔治和我正在制作利奥的主题机械装置。长期计划:

1\. [Better integration with the jupyter ecosystem](https://github.com/leo-editor/leo-editor/issues/797).2\. Integrating [Joe Orr's great vue.js](https://github.com/kaleguy/leovue#leo-vue) stuff.
Be sure to scroll to the bottom of the page.3\. Keeping up with vscode, atom, etc. 😉
 ****Driscoll:你最喜欢哪些 Python 库(核心或第三方)？**

**Ream:** [pyflakes](https://pypi.python.org/pypi/pyflakes) ，目前为止。Python 开发者如果不使用它就是在浪费时间。我研究了它的每一行代码。这是天才的作品。

pyflakes 基本上捕捉到了我最初所有的错误。它是如此之快，以至于我可以配置 Leo 在我保存一个. py 文件时运行 pyflakes。我配置 Leo 在 pyflakes 检测到错误时发出一个对话框😉

Driscoll:你是如何进入 Leo 项目的？

**令:**利奥是在我努力理解[唐纳德·克努特的 CWEB 体系](https://www-cs-faculty.stanford.edu/~knuth/cweb.html)中成长起来的。这被认为是理解程序的一大进步，但是我不能理解它！第一个大顿悟是:“网是伪装的轮廓”。

我在 1995 年发明了 Leo 的前身，从那以后就一直在研究它。狮子座的[历史页面包含了详细信息。后来，网络协作把我从缺乏想法和精力中解救了出来。现在，我们都被提高狮子座的想法淹没了。
 **德里斯科尔:你能给我们概述一下什么是狮子座吗？**](http://leoeditor.com/history.html#beginnings)

**Ream:** 我从一开始就纠结于这个问题。[利奥的维基百科页面](https://en.wikipedia.org/wiki/Leo_(text_editor))是一个干巴巴的总结。[狮子座的序言](http://leoeditor.com/preface.html)是另一个。但这些可能对你或你的读者没有太大帮助。

我可以通过列举对 Leo 有贡献的伟大程序员来回答“侧面”: Terry Brown、Brian Harry、Bernhard Mulder、Vitalije Milosevitch、Ville M. Vainio 等等。当我对利奥一世心存疑虑时，请记住它们🙂

你必须用狮子座才能真正理解它。我们狮子座人谈论狮子座世界。许多人确实明白这一点:

“当我第一次打开 Leo 时，是出于好奇。但是用过了，就再也不回去了。他们得把利奥从我冰冷、死气沉沉的手指中撬出来！”特拉弗斯·霍夫

你会在[页面中发现人们对狮子座](http://leoeditor.com/testimonials.html)的评价。

但还是。所有这些都是间接的。正如 Vitalije 最近指出的，真正的 Aha 是 *Leo 将大纲结构与代码本身*彻底连接起来。

对于 Leo 来说，代码不仅仅是文本。它生活在一个“超级复制轮廓”中，即一个有向无环图。只有这样的数据结构允许对克隆进行简单的定义。

**从静态组织中克隆自由轮廓**。数据，包括部分程序，可以驻留在同一大纲中任意多个位置。我一直用这个。我收集我正在工作的方法的复制品，以便我能清楚地看到它们。这消除了大量的搜索。Leo 的克隆查找命令以惊人的方式利用了克隆，但是我不想让你的读者不知所措😉

**所有代码/脚本都可以轻松访问它们所嵌入的大纲**。Leo 的 execute-script 命令定义了三个常量:c、g 和 p。c 是 Leo 的 *live* 代码的*所有*的网关。g 是一个 python 模块，包含许多有用的实用程序。最重要的是，p 是大纲中当前选择的节点。p.h 是它的标题。p.b 是它的正文。p.u 是一组可任意扩展的 u(用户定义的属性)。

Leonine 脚本是可以自动访问 c、g 和 p 的脚本。为了让您对这类脚本的能力有所了解，下面是第二个“hello world”示例。把这个放到*任意* Leo 节点，做 Ctrl-B(执行脚本):

```py
for p in c.all_positions():
    print(p.level()*' ' + p.h)
```

它打印大纲中的所有标题，适当缩进。

这就是*整个*脚本。注意那些不叫的狗！

但这仅仅是开始。通过 c 上易于使用的方法，c 常量允许脚本修改它们所在的大纲。包括我哥在内的一些用户用这个把 Leo 变成数据库。

I'm going on and on, so I'll leave it to interested readers to learn about @button nodes, @test nodes, and all the rest.Neither atom nor vscode are likely to have these features.  Emacs org mode comes closest, but in org mode everything is still plain text.

Driscoll:你从开源工作中学到了什么？

我最重要的目标是激发他人的潜能。我很少不得不与持有错误观点的人打交道，我总是用这样的想法来纠正这些错误，即那个人以后可能会提出伟大的想法。

我通过偶尔改变我的想法来训练人们不要太尊重我的想法😉我希望人们，甚至是新手，反驳我的观点。现在，几乎所有新的事情都是由于用户的紧急请求而发生的😉

我不是最伟大的程序员。但是，我是一个好的项目经理。我之所以这么说，是因为 Leo 的开发人员创造了我做梦也没想到的东西:

*   没有伯恩哈德·穆德，就不会有@clean 和[穆德/令更新算法](http://leoeditor.com/appendices.html#the-mulder-ream-update-algorithm)。
*   @button 如果没有' e '就不会发生。
*   利奥的迷你缓冲器是从布莱恩·哈利的原型发展而来的。
*   没有维尼奥镇，就不会有利奥的伊普森桥。

**项目第一**。我们从不贬低他人，尽管我们可能会深入讨论技术问题。嘘，你可能会认为这只是常识，但显然不是。

我们有一个原则，我们从不争论偏好。如果有两种合法的、重要的、不相容的方式让狮子座工作，我们就创造了一个狮子座的环境。这消除了无用的争论。

我们从未遇到过巨魔的问题，也许是因为所有对它的访问都是有节制的。我是一个以证据为基础的人，没有时间做任何事或任何人。

One last thing. Last year I realized that Leo is my entire professional life. Progress has increased significantly since I went "all in".

德里斯科尔:你还有什么想说的吗？

**令:**工作流程至上。我立即为所有东西创建 github 问题。目标是不需要记住*任何事情*。我继续寻找方法让 Leo 加快我的工作流程。

If your readers have not already done so, I highly recommend that they read [Getting Things Done](https://www.amazon.com/Getting-Things-Done-Stress-Free-Productivity/dp/0142000280). Read the entire book, not some summary. Much of what I am about to say stems from this book.Recently I have relearned the fundamental lesson that procrastination saps energy.  I am intensely focused on doing one thing at a time.  This means, for example, that even the smallest mental nits must be cleared.  Some can be filed in my Leo outlines. Others are better put in new github issues.I've been programming for 40+ years. It's important to respect my natural body rhythms. I take regular breaks in the afternoon, after making sure that I am not trying to remember *anything.* This is the only way to survive long term.**Driscoll: Thanks for doing the interview!****