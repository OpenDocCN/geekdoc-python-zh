# Python 新闻:2021 年 5 月有什么新消息

> 原文：<https://realpython.com/python-news-may-2021/>

如果你想了解 2021 年 5 月**的 **Python** 世界中发生的事情，那么你来对地方了，可以得到你的**新闻**！*

*五月是重大事件发生的一个月。托盘项目 T1 是流行框架如 T2 烧瓶 T3 和 T4 点击 T5 的所在地，发布了所有六个核心项目的新的主要版本。Python 软件基金会(PSF) 主办了 PyCon US 2021，这是一次虚拟会议，提供了真实的现场体验。

让我们深入了解过去一个月最大的 Python 新闻！

**免费下载:** [从 CPython Internals:您的 Python 3 解释器指南](https://realpython.com/bonus/cpython-internals-sample/)获得一个示例章节，向您展示如何解锁 Python 语言的内部工作机制，从源代码编译 Python 解释器，并参与 CPython 的开发。

## 微软成为 PSF 第三个有远见的赞助商

在上个月的新闻综述中，我们报道了谷歌和彭博工程公司是如何成为 PSF 的首批两个有远见的赞助商的。四月底，PSF 也[宣布](https://pyfound.blogspot.com/2021/04/welcoming-microsoft-as-visionary-sponsor.html)微软增加了对远见者的支持。

微软正在向 Python 打包工作组提供财政支持:

> 作为我们对 PSF 的 15 万美元财政赞助的一部分，我们将把我们的资金集中到包装工作组，以帮助进一步改善 PyPI 和包装生态系统的开发成本。由于最近披露的安全漏洞，可信供应链对我们和 Python 社区来说是一个关键问题，我们很高兴能为长期改进做出贡献。([来源](https://devblogs.microsoft.com/python/supporting-the-python-community/#development-of-python-and-related-projects))

除了有远见的赞助商身份，微软还有五名 Python 核心开发人员兼职为 Python 做贡献:布雷特·坎农、史蒂夫·道尔、吉多·范·罗苏姆、埃里克·斯诺和巴里·华沙。

关于微软对 Python 和 PSF 支持的更多信息，请查看其官方声明。

查看 Steve Dower 的[账户](https://medium.com/microsoft-open-source-stories/python-at-microsoft-flying-under-the-radar-eabbdebe4fb0),了解微软对 Python 的立场在这些年是如何变化的。你也可以在[的真实 Python 播客](https://realpython.com/podcasts/rpp/47/)上听 Brett Cannon 分享他在微软使用 Python 的经历。

[*Remove ads*](/account/join/)

## 托盘发布所有核心项目的新的主要版本

经过 Pallets 团队及其众多开源贡献者两年的辛勤工作，终于发布了所有六个核心项目的新的主要版本:

*   [2.0 号烧瓶](https://flask.palletsprojects.com/en/2.0.x/changes#version-2-0-0)
*   [工具 2.0](https://werkzeug.palletsprojects.com/en/2.0.x/changes/#version-2-0-0)
*   [金贾 3.0](https://jinja.palletsprojects.com/en/3.0.x/changes/#version-3-0-0)
*   [点击 8.0](https://click.palletsprojects.com/en/8.0.x/changes/#version-8-0)
*   [危险 2.0](https://itsdangerous.palletsprojects.com/en/2.0.x/changes/#version-2-0-0)
*   [MarkupSafe 2.0](https://markupsafe.palletsprojects.com/en/2.0.x/changes/#version-2-0-0)

所有六个项目都放弃了对 Python 2 和 Python 3.5 的支持，使得 Python 3.6 成为支持的最低版本。删除了以前不赞成使用的代码，并添加了一些新的不赞成使用的代码。

影响所有六个项目的一些主要变化包括:

*   将默认分支重命名为`main`
*   添加全面的类型注释，使[类型检查](https://realpython.com/python-type-checking/)用户代码更加有用，并提供与[ide](https://realpython.com/python-ides-code-editors-guide/)的更好集成
*   使用像[预提交](https://pre-commit.com/)、[黑色](https://black.readthedocs.io/en/stable/)和[薄片 8](https://flake8.pycqa.org/en/latest/) 这样的工具，在所有的代码库和新的拉取请求中实施一致的风格

除了上面列出的大范围变化，单个项目还有几个吸引人的新特性。

### Flask 获得原生`asyncio`支持

根据 2020 年 Python 开发者调查，Flask 是最流行的 Python web 框架。Flask 2.0 的原生支持肯定会让该框架的支持者高兴。

您可以将从路由到错误处理程序到请求前和请求后功能的所有东西都放入协程中，这意味着您可以使用`async def`和`await`来定义视图:

```py
@app.route("/get-data")
async def get_data():
    data = await async_db_query(...)
    return jsonify(data)
```

在这个取自 [Flask docs](https://flask.palletsprojects.com/en/2.0.x/async-await/) 的示例代码片段中，定义了一个名为`get_data()`的异步视图。它进行异步数据库查询，然后以 JSON 格式返回数据。

Flask 的支持并非没有警告。Flask 仍然是一个 [Web 服务器网关接口(WSGI)](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface) 应用程序，并且和任何其他 WSGI 框架一样受到相同的限制。Flask 的文档描述了这些限制:

> 异步函数需要事件循环才能运行。Flask 作为一个 WSGI 应用程序，使用一个 worker 来处理一个请求/响应周期。当一个请求进入一个异步视图时，Flask 将在一个线程中启动一个事件循环，在那里运行视图函数，然后返回结果。
> 
> 即使对于异步视图，每个请求仍然会占用一个工作线程。好处是您可以在一个视图中运行异步代码，例如进行多个并发数据库查询、对外部 API 的 HTTP 请求等。但是，您的应用程序一次可以处理的请求数量将保持不变。([来源](https://flask.palletsprojects.com/en/2.0.x/async-await/#performance))

如果你是异步编程的新手，看看 Python 中的[异步 IO:一个完整的演练](https://realpython.com/async-io-python/)。你也可以从 Flask 2.0 中的文章 [Async 中获得 Flask 新的`asyncio`支持，这篇文章是 PyCoder 的每周](https://testdriven.io/blog/flask-async/)时事通讯中的特色。

除了原生的`asyncio`支持，Flask 2.0 还为常见的 [HTTP 方法](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)增加了一些新的路由装饰器。例如，在 Flask 1.x 中，您使用`@app.route()`视图装饰器声明了一个支持`POST`方法的路由:

```py
@app.route("/submit-form", methods=["POST"]) def submit_form():
    return handle_form_data(request.form)
```

在 Flask 2.0 中，您可以使用`@app.post()`视图装饰器来缩短这段代码:

```py
@app.post("/submit-form") def submit_form():
    return handle_form_data(request.form)
```

这是一个很小的变化，但是可读性有了很大的提高！

你可以在官方 [changelog](https://flask.palletsprojects.com/en/2.0.x/changes/#version-2-0-0) 中找到 Flask 2.0 的所有改动。

[*Remove ads*](/account/join/)

### Jinja 获得改进的异步环境

在 Jinja 2.x 中的支持需要一个补丁系统以及[一些](https://jinja.palletsprojects.com/en/2.11.x/api/#async-support)开发者需要记住的警告。原因之一是 Jinja 2.x 支持 Python 2.7 和 Python 3.5。

现在所有的托盘项目都只支持 Python 3.6+，修补系统[被移除](https://github.com/pallets/jinja/pull/1392)，为使用 Jinja 3.0 的项目提供更自然的`asyncio`体验。

你可以在官方 [changelog](https://flask.palletsprojects.com/en/2.0.x/changes/#version-2-0-0) 中找到 Jinja 3.0 的所有改动。

### 点击得到一个检修过的外壳标签完成系统

为应用程序构建优秀的[命令行界面](https://realpython.com/command-line-interfaces-python-argparse/) (CLI)可能是一件苦差事。 [Click](https://click.palletsprojects.com/en/latest/) 项目通过其友好的 API 帮助减轻了这一负担。

shell 用户期望从 CLI 获得的特性之一是**制表符补全**，当用户键入几个字符并按下 `Tab` 时，它会提示命令名、选项名和选项值。

Click 一直支持 shell tab 补全，但是实现起来很混乱，正如 Pallets 维护者 [David Lord](https://twitter.com/davidism) 在 2020 年 3 月的一期 GitHub 上指出的:

> 我一直在尝试审查[一个 pull 请求],它增加了基于类型的完成，这让我意识到完成是多么的混乱，无论是在 Click 中还是在 shells 如何实现和记录它们的系统中。…
> 
> 我们不得不重新实现 shell 应该做的事情，比如转义特殊字符、添加空格(目录除外)、排序等等。如果用户想要提供他们自己的完成，他们也必须记住这样做。
> 
> 我们没有理由只返回完成。我们已经支持返回描述，大概我们可以扩展更多。如果 Click 可以向完成脚本指示它应该使用 Bash 或 ZSH 提供的其他函数，这不是很酷吗？([来源](https://github.com/pallets/click/issues/1484#issue-574225486))

到 2020 年 10 月，Click 的 shell tab 补全系统已经全面检修，内置了对 [Bash](https://www.gnu.org/software/bash/) 、 [Zsh](https://www.zsh.org/) 和 [fish](https://fishshell.com/) 的支持。该系统是可扩展的。您可以添加对其他 shells 的支持，并且可以在多个级别上定制完成建议。

新的完成系统现在在 Click 8.0 中可用，对于希望在用户最喜欢的 shell 中为用户提供友好的 CLI 体验的项目来说，这是一个巨大的胜利。

你可以在官方的[变更日志](https://click.palletsprojects.com/en/8.0.x/changes/#version-8-0)上找到 Click 8.0 以上的完整变更列表。

## PyCon US 2021 连接世界各地的 Pythonistas】

对于美国的皮托尼斯塔来说，晚春总是令人兴奋的时刻。PyCon US 是致力于 Python 的最大年度大会，传统上在四月或五月举行。

今年的 PyCon US 与以往的会议略有不同。最初定于在宾夕法尼亚州匹兹堡举行的 [PyCon US 2021](https://us.pycon.org/2021/) 因新冠肺炎疫情而转变为仅在线活动。

### 感觉像真的一样的虚拟会议

PyCon US 2020 也是虚拟的，但最后一刻过渡到网上会议让组织者几乎没有时间准备一次真正的 PyCon 体验。今年，PSF 有充足的时间来计划，它提供了一个令人难以置信的参与活动，真实地反映了过去 PyCon US 会议的精神。

虽然谈话是预先录制的，但视频是按时间表播放的，而不是按需提供的。每个讲座都有一个与之相关的聊天室，演讲者可以与参与者互动并回答问题。

大会还设有一个虚拟展厅，将 Pythonistas 与 Python 世界的各种组织联系起来，包括微软、谷歌、彭博、[和更多的](https://us.pycon.org/2021/sponsorship/sponsors/)。

然而，PyCon 2021 最吸引人的部分是执行良好的开放空间和休息区。[开放空间](https://realpython.com/pycon-guide/#open-spaces)是类似 meetup 的小型活动，允许与会者围绕共同的兴趣进行会面和互动。Python 作者、业余无线电爱好者、社区组织者等等都有开放的空间。

会议休息区包括虚拟桌子，让有限数量的人参加视频会议。任何人都可以抢一把空椅子，参与到谈话中，即使谈话已经开始了。休息室给了 PyCon 一种真正独特的氛围，具有你在面对面会议中所期望的所有自发性，有效地实现了虚拟的[走廊轨道](https://ericmjl.github.io/blog/2016/6/3/the-pycon-ers-guide-to-the-hallway-track/)，这是 PyCon US 的标志之一。

将 PyCon US 搬到网上使得全球更多的 Pythonistas 可以参加会议。Python 爱好者不再有旅行和住宿费用的负担，只需支付入场费，就可以在自己舒适的家中加入 PyCon。

如果你错过了 2021 年的 PyCon US，你很快就可以在 YouTube 上观看这场演讲。在撰写本文时，这些视频仍在后期制作中，但应该会在未来几周内推出。

[*Remove ads*](/account/join/)

### Python 的未来集中在性能上

PyCon US 的目标之一是将 Python 核心开发者和 Python 用户聚集在一起，讨论该语言的现状和未来愿景。每年的 [Python 语言峰会](https://us.pycon.org/2021/summits/language/)都会聚集 Python 实现的维护者，比如 [CPython](https://realpython.com/cpython-source-code-guide/) 、[pypypy](https://realpython.com/pypy-faster-python/)和 [Jython](https://www.jython.org/) ，分享信息，解决问题。

今年的语言峰会有几个激动人心的演讲。 [Dino Viehland](https://twitter.com/DinoViehland) 谈到了 Instagram 在其内部以性能为导向的项目 [Cinder](https://github.com/facebookincubator/cinder) 中对 CPython 的[改进，包括**对异步 I/O** 的几项增强。](https://pyfound.blogspot.com/2021/05/the-2021-python-language-summit-cpython.html)

Python 的创造者[吉多·范·罗苏姆](https://twitter.com/gvanrossum)提出了让 CPython 更快的计划。Van Rossum 的目标是**在 Python 3.11** 之前将 CPython 的速度翻倍。提升的性能将主要惠及运行 CPU 密集型纯 Python 代码或使用 Python 内置工具和网站的用户。

今年 Python 语言峰会的另一个令人兴奋的特点是，PSF 给了真正的 Python 自己的 T2 一个机会，在一系列博客文章中报道峰会的演讲和讨论。你可以在 [PSF 博客](https://pyfound.blogspot.com/2021/05/the-2021-python-language-summit.html)上找到她所有关于语言峰会的文章。

## Python 的下一步是什么？

五月对于 Python 来说是一个多事之秋。在 *Real Python* 展会上，我们对 Python 的未来感到兴奋，迫不及待地想看看在**6 月**会有什么新东西等着我们。

你最喜欢的 **Python 新闻**来自**5 月**的哪一条？我们错过了什么值得注意的吗？请在评论中告诉我们，我们可能会在下个月的 Python 新闻综述中介绍您。

快乐的蟒蛇！*****