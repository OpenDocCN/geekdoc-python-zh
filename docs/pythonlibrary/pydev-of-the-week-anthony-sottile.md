# 本周 PyDev:安东尼·索特瓦

> 原文：<https://www.blog.pythonlibrary.org/2018/10/29/pydev-of-the-week-anthony-sottile/>

本周，我们欢迎安东尼·索特瓦( [@codewithanthony](https://twitter.com/codewithanthony) )成为我们本周的 PyDev！安东尼是 [tox](https://github.com/tox-dev/tox) 和 [pytest](https://github.com/pytest-dev/pytest) 软件包的维护者之一。他也是“死蛇”PPA 团队的成员，该团队为某些 EOL Linux 发行版反向移植 Python。虽然你可以在安东尼的[网站](http://anthonysottile.com/index.htm)上发现一些关于他的信息，但你可能会从他的 [Github 简介](https://github.com/asottile)中了解更多。

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

从小我就一直对电脑很着迷。我最早写的一些程序就是为了简化(阅读:*作弊*)作业。一个包含使用 visual basic 的基本二次公式求解 [gui 界面的 word 文档被复制到了相当多的软盘上。我最终转向了 web 开发，因为这是一种更容易获得的发布机制。](https://www.youtube.com/watch?v=hkDD03yeLnU)

我就读于密歇根大学。)原本是学生物化学的。我想通过医学和研究来改变世界，但两年后，我决定转向我更强烈的激情。经过激烈的竞争，我终于在两年内完成了四年的课程，并获得了计算机科学学位！

我的大部分个人时间都花在了骑自行车上(我在一个花哨的电子表格中仔细记录了这一点——今年记录了 4600 英里)。我的其他一些爱好是徒步旅行、烹饪、跑步、滑雪比赛、拉小提琴和写一点诗(这是一个约会网站吗？).当然，我花了很大一部分时间构建开源软件并为之做出贡献🙂

我最大的成就是成为一名口袋妖怪大师——不仅完成了一个活生生的口袋妖怪，还通过合法地捕捉每一个可能的[闪亮的](https://bulbapedia.bulbagarden.net/wiki/Shiny_Pok%C3%A9mon)口袋妖怪，忍受了极度的乏味。从那以后一切都在走下坡路。

**你为什么开始使用 Python？**

我第一次接触 python 是通过在 Yelp 的就业。尽管我被聘为 JavaScript 前端开发人员，但由于好奇心的驱使，我很快就钻研了全栈开发。最终，我建立了一个 web 基础设施和开发工具团队。一路走来，我的补全主义天性将我带到了语言的许多角落，包括元编程、打包、python 3 移植、RPython (pypy)、C 扩展等等！作为我目前选择的毒药，我决定我最好知道它是如何工作的！

你还知道哪些编程语言，你最喜欢哪一种？

在某个时候，我会认为自己是 JavaScript、C#、Java、C++和 C(以及其他许多语言)的专家。我从编程语言中得到的一个价值是拥有静态保证的能力(通常以类型检查器的形式)。因此，我对 python 的渐进类型化方法在类型检查领域的改进感到兴奋！我用过的最喜欢的语言(就语法和特性而言)是 C#(尽管这可能只是因为 Visual Studio 有多好)。

这里的荣誉奖是 go。虽然我认为还有一些地方可以做得更好(标签、打包、泛型)，但 go 有一个我希望每种语言都具备的杀手级特性:作为一等公民重写代码。“go fmt”不仅证明了这一原则的成功(格式化代码只有一种方式！)但是编写读取和操作代码的工具是很容易的，并且受到鼓励。

你现在在做什么项目？

我热衷的项目是[pre-commit](https://pre-commit.com)——一个用于管理 git 钩子(主要是 linters 和 fixers)的多语言框架。在维护框架的同时，我还构建了一些[我的](https://github.com/asottile/add-trailing-comma)和[自己的](https://github.com/asottile/pyupgrade) [修复程序](https://github.com/asottile/reorder_python_imports)。最近，我还花了一些时间帮助维护 [tox](https://github.com/tox-dev/tox) 和 [pytest](https://github.com/pytest-dev/pytest) 。我的一个较新的项目 [all-repos](https://github.com/asottile/all-repos) ，是一个大规模管理微升的工具，使搜索和应用跨库的大规模变化变得容易。

哪些 Python 库是你最喜欢的(核心或第三方)？

对我来说，这实际上是一个非常有趣的问题——我不想把科学排除在外，于是求助于 [all-repos](https://github.com/asottile/all-repos) 来尝试寻找答案！好的，所以[结果](https://i.fluffy.cc/WvD3B3VqqvtcxNvJlxqsnzvnhP2g8t6l.html)并不完全是最有趣的(当然你导入了很多‘OS’！)但是让我们深入了解一些异常值和我的最爱。列表上的第一个是[pytest](https://github.com/pytest-dev/pytest)——当然是第一个，我认为测试非常重要，还有什么比 pytest 更好的工具可以使用呢！在标准库中，我最喜欢的是“argparse ”,因为我倾向于编写大量的命令行实用程序，所以它被大量使用。在 favorites 部分的一些荣誉提名是“sqlite3 ”(非常适合原型设计和令人惊讶的性能)、“collections ”(适合“namedtuple ”)、“contextlib ”(适合“contextmanager ”)和“typing ”(这是我最近开始接触的)。我用得最多的是“ast”和“tokenize”——我真的很喜欢静态分析，并且倾向于为它编写一堆工具。

你是如何成为“死蛇”团队的一员的(顺便说一句，我真的很感激)？

也许我最喜欢开源的部分是，如果你提供帮助，人们通常会接受。我最初在 Yelp 工作时，在将各种包移植到生命周期结束的 ubuntu lucid 上时，开始从事 debian 打包工作。就在 python 3.5 发布之前，lucid 到达了其支持周期和 launchpad 的末尾(理应如此！)禁用 PPA 上传为 lucid(包括死蛇)。开发人员(包括我自己)想使用一些新特性，比如“subprocess.run”、“async ”/“await”以及解包泛化。当时，升级到尚未停产的发行版还需要 2 到 3 年的时间。我成功地移植了 3.5，并在这个过程中学到了很多东西(撤销多重架构、调整依赖关系、修补测试等等)。).当 3.6 发布时，deadsnakes ppa 主页上显示了一条消息，表示支持将停止。我主动提出帮助维护，稍加指导就有了 3.6 的工作资源，剩下的就是历史了！

作为团队的一员，你面临过哪些挑战？

老实说，一旦你学会了工具(或者说[自动化工具](https://github.com/deadsnakes/runbooks)), dead snakes 的大部分职责和维护都是非常简单明了的。大多数易维护性来自于三个相对高质量的上游:debian、ubuntu，当然还有 cpython。大部分的维护工作来自于新版本(一个新的 LTS 会导致所有包的重新构建和两年发行版变化的调整！).

安东尼，谢谢你接受采访！