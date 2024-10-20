# 本周 PyDev:彼得·鲍姆加特纳

> 原文：<https://www.blog.pythonlibrary.org/2022/04/11/pydev-of-the-week-peter-baumgartner/>

本周我们欢迎彼得·鲍姆加特纳( [@pmbaumgartner](https://twitter.com/pmbaumgartner) )成为我们本周的 PyDev！彼得也是[的 Python 博客作者](https://www.peterbaumgartner.com/blog/)，他写了一些关于 Python 和数据科学的文章。彼得还收集了一些有趣的 Jupyter 笔记本，你可以从中学习。你可以在 GitHub[上看到 Peter 正在做的项目。](https://github.com/pmbaumgartner/)

让我们花一些时间来更好地了解 Peter！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我目前是[爆](https://explosion.ai/)的机器学习工程师。在此之前，我在一家名为 [RTI International](https://www.rti.org/) 的非营利研究机构工作，当我开始使用 python 研究数据科学时，我在德勤工作。我在北卡罗来纳州立大学[高级分析研究所](https://analytics.ncsu.edu/)获得了分析硕士学位。在那之前，我是一名高中数学老师。

我的主要爱好是跑步和用绘图仪创作艺术。每个周末，我都试着参加由 [Parkrun](https://www.parkrun.us/) 举办的当地定时 5 公里跑，这是一个非常酷的组织，我鼓励任何能力的跑步者参加。笔式绘图仪基本上是一个你编程绘制的机器人——我去年买了一个，并把我的大部分工作发布到 [twitter](https://twitter.com/search?q=from%3A%40pmbaumgartner%20%23plottertwitter&src=typed_query&f=live) 。

这些天来，我没有太多的时间从事业余爱好，因为我的大部分空闲时间都花在了帮助抚养我现在 1 岁的儿子克拉克上。他是一个非常有趣、喜欢冒险的孩子，现在他可以移动了，正在享受探索世界的乐趣。

**你为什么开始使用 Python？**

完成硕士学位后，我在当地一家营销公司做一些合同工作。我的硕士项目用 SAS 编程语言教会了我们一切，但是 SAS 很贵，而且在我看来用它编程很痛苦，所以对于这个合同工作，我们决定使用 python。这是一次真正的尝试，因为我必须学习 python，并为客户提供有用的分析。最后，它成功了，因为它让我意识到编程其实可以很有趣，而不总是一场斗争。从那时起，我就一直是 python 的主要用户。

你还知道哪些编程语言，你最喜欢哪一种？

我学的第一门语言是 Visual Basic——我在高中时上过一门计算机科学课，这门课真的让我大开眼界，让我了解到你可以在编程中做一些很酷的事情。在大学里，我也学习了一门使用 C++的计算机科学课程，但是我完全不记得那些知识了。我在读硕士期间也学了一点 SAS 和 R。

专业做编程以来，我学的最多的语言 Julia，是我最喜欢的非 python 语言。我喜欢它的一点是，从 python 开始学习很容易，我认为这非常重要。我以前尝试过学习 Rust，但是语法和概念太不同了，对我来说太难了。与 Julia 一起，它让我认识到思考和解决问题有不同的方式，我可以从概念上把我学到的东西应用到我如何开发 python 程序上。这也迫使我增加了一些我从未学过的计算机科学基础知识。

总的来说，我鼓励每个人学习第二种编程语言，但可能是一种在语法上接近他们母语的语言。在用 python 编程了大约 5 年之后，学习另一种语言对我来说真的很有帮助，仅仅是为了接触一种编程语言如何工作的替代方式。

你现在在做什么项目？

在 Explosion，我们刚刚推出了名为 [spaCy Tailored Pipelines](https://explosion.ai/spacy-tailored-pipelines) 的咨询服务，这将带来一些非常有趣的应用自然语言处理项目。除此之外，我花了很多时间回顾人们如何使用我们的产品，并通过更新文档、添加示例或创建新的开源库来完善我们的工具。例如，我最近开发的一个[组件](https://github.com/pmbaumgartner/corpus_statistics)简单地计算了人们通过空间管道传递文本时看到的标记。我开始着手这项工作，因为我注意到很多人都在请求这项功能，他们对这项功能在 spaCy 中应该如何工作感到困惑。另一个例子是从 HTML 解析文本的[组件](https://github.com/pmbaumgartner/spacy-html-tokenizer)。人们经常会从抓取的网页中获取数据，并希望对其进行自然语言处理，但如果他们只是从 HTML 中获取原始文本，他们就会忽略文档的结构，这可能会对下游产生一些负面影响。

哪些 Python 库是你最喜欢的(核心或第三方)？

我认为这是最难回答的问题，因为有太多好的问题了。

核心:

*   collections - I use `Counter` and `defaultdict` all the time.
*   itertools - `chain` is awesome, `groupby` is great for data, and I've used `combinations` for work and plotter art
*   pathlib - So much of my work for applied projects deals with paths that I'd be totally lost without pathlib.
*   tempfile - Sometimes I work with libraries that have APIs that require persisting to disk when I'd rather pass a buffer. `tempfile` makes it super easy to work with these in a clean way.

第三方:有我几乎每天都用的:`pandas`、`spaCy`、`umap-learn`、`altair`、`tqdm`、`pytest`、`black`、`numpy`都很神奇。

Then there are some libraries that I love and use in specific circumstances, like `typer`, `rich`, `questionary` for CLI tools. `poetry` for packaging. `streamlit` for making simple apps. `numba` for faster array operations. `sentence-transformers` for NLP when sentences are involved. `loguru` for logging. `shapely` and `vsketch` for anything with my plotter.

**How did you decide to write a Python blog?**

I've had a blog for a long time but until recently I was publishing on it less than I was happy with. Recently I've been trying to reframe my writing process by recognizing that blog posts don't have to be perfect. I read a lot about good writing, and read a lot of good technical writing, and often times that puts so many constraints in my head when writing something that I never end up finishing anything. The practices for good writing would be useful if I was writing something more formal, like a book, but for my personal blog I give myself permission to not think about that stuff too much.

**Where do you get your ideas from when it comes to writing articles?**

Almost all of my articles are documenting things that I've recently learned. I try to think of writing a blog post as steps in the [Feynman technique](https://fs.blog/feynman-technique/) of learning. I used to be a teacher, so I also try and be cognizant of the [Curse of Knowledge](https://digitallearning.arizona.edu/news/curse-knowledge) and write things down *as* I'm learning them, rather than after I'm done learning, then reorganize those original thoughts in the way that I think about something after I've learned it.

**Is there anything else you’d like to say?**

Support the developers and organizations that help make the python ecosystem great!

 **Thanks for doing the interview, Peter!**