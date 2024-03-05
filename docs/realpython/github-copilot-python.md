# GitHub Copilot:以思维的速度与 Python 一起飞行

> 原文：<https://realpython.com/github-copilot-python/>

GitHub Copilot 是一项激动人心的新技术，它承诺为你的代码编辑器提供一个由**人工智能**驱动的虚拟助手，当它向公众发布时，引起了相当大的争议。Python 是该工具特别支持的语言之一。看完这篇教程，你就知道 GitHub Copilot 到底是一个风险，一个噱头，还是软件工程中真正的游戏规则改变者。

**在本教程中，您将学习如何:**

*   在你的代码编辑器中安装 **GitHub Copilot 扩展**
*   将你对任务的自然语言描述转换成工作代码
*   在多个可选的**智能代码完成**建议之间进行选择
*   探索不熟悉的框架和编程语言
*   教 GitHub Copilot 如何使用您的**自定义 API**
*   使用虚拟对程序员实时练习测试驱动开发

要继续本教程，你需要有一个个人 [GitHub](https://github.com/) 账户和一个代码编辑器，比如 [Visual Studio Code](https://realpython.com/python-development-visual-studio-code/) 或者一个集成开发环境，比如 [PyCharm](https://realpython.com/pycharm-guide/) 。

**免费下载:** [点击这里下载免费的键盘快捷键备忘单](https://realpython.com/bonus/github-copilot-python-pdf/)让 GitHub Copilot 的编码更快。

## With GitHub Copilot 入门

[GitHub Copilot](https://github.com/features/copilot) 是第一款基于 [OpenAI Codex](https://openai.com/blog/openai-codex/) 系统的**商业产品**，可以实时将[自然语言](https://realpython.com/nltk-nlp-python/)翻译成十几种编程语言的代码。OpenAI Codex 本身就是 [GPT-3](https://en.wikipedia.org/wiki/GPT-3) 深度学习语言模型的后代。Codex 中的[神经网络](https://realpython.com/python-ai-neural-network/)在文本和 GitHub 上托管的数以亿计的公共代码库中进行了训练。

**注意:**你可以通过收听真正的 Python 播客第 121 集[来了解更多关于 GPT-3 的信息，该播客由数据科学家朱迪·波切尔主讲。](https://realpython.com/podcasts/rpp/121/#t=989)

GitHub Copilot 理解一些编程语言和许多人类语言，这意味着您不仅限于英语。例如，如果你的母语是西班牙语，那么你可以用母语与 GitHub Copilot 交谈。

最初，[产品仅作为技术预览版](https://github.blog/2021-06-29-introducing-github-copilot-ai-pair-programmer/)提供给特定人群。最近[改变了](https://github.blog/2022-06-21-github-copilot-is-generally-available-to-all-developers/)，今天，任何人都可以在他们的代码编辑器中体验到**人工智能**不可思议的力量。如果你想试驾一下，你需要订阅 GitHub Copilot。

[*Remove ads*](/account/join/)

### 订阅 GitHub Copilot

要启用 GitHub Copilot，请转到 GitHub 个人资料中的[计费设置](https://github.com/settings/billing)并向下滚动，直到看到相关部分。不幸的是，这项服务对大多数人来说并不是免费的。在撰写本文时，这项服务每月收费 10 美元，如果提前支付，每年收费 100 美元。你可以享受一个**60 天的试用期**，无需支付任何费用，但必须提供你的账单信息。

**注意:**请务必在到期前取消未付费的订阅计划，以免产生不必要的费用！

**学生**和**开源维护者**可以免费订阅 GitHub Copilot。如果你是一个幸运的人，那么在启用该服务后，你会看到以下信息:

![GitHub Copilot Billing Status](img/029bfbc0d43602a2d1738ea9e10cc8ba.png)

<figcaption class="figure-caption text-center">GitHub Copilot Billing Status</figcaption>

GitHub 将根据学术注册证明，例如您学校 ID 的照片或[中的电子邮件地址，每年验证您的状态一次。edu 域，或者你在流行的开源软件库中的活动。](https://en.wikipedia.org/wiki/.edu)

有关设置和管理 GitHub 订阅的详细说明，请遵循[官方文档](https://docs.github.com/en/billing/managing-billing-for-github-copilot/managing-your-github-copilot-subscription)中的步骤。接下来，您将学习如何为 Visual Studio 代码安装 GitHub Copilot 扩展。如果你更喜欢使用 GitHub Copilot 和 PyCharm，那么直接跳到[了解如何使用](#install-a-pycharm-plugin)。

### 安装 Visual Studio 代码扩展

因为[微软拥有 GitHub](https://news.microsoft.com/announcement/microsoft-acquires-github/) ，所以毫不奇怪他们的 Visual Studio 代码编辑器是第一个获得 GitHub Copilot 支持的工具。在 Visual Studio 代码中安装扩展有几种方法，但最快的方法可能是使用 `Ctrl` + `P` 或 `Cmd` + `P` 调出*快速打开*面板，然后键入以下命令:

```py
ext install GitHub.copilot
```

当你按下 `Enter` 确认后，它会安装扩展并提示你重新加载编辑器。

或者，您可以在位于窗口左侧的*活动栏*中找到*扩展*图标，并尝试在 [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot) 上搜索 GitHub Copilot 扩展:

[![GitHub Copilot Extension for Visual Studio Code](img/237af5308d46957394d6ce7ac6579667.png)](https://files.realpython.com/media/vscode_ext.f3c9e44514bd.png)

<figcaption class="figure-caption text-center">GitHub Copilot Extension for Visual Studio Code</figcaption>

您也可以使用相应的键盘快捷键直接在 Visual Studio 代码中显示*扩展*视图。

安装完成后，Visual Studio 代码将要求您登录到 GitHub，以访问您的 GitHub 配置文件，您的新扩展需要:

[![Authorize GitHub for VS Code](img/d85625e613985f7b40e973484c53e950.png)](https://files.realpython.com/media/vscode_auth.9cdc89c0c76a.png)

<figcaption class="figure-caption text-center">Authorize GitHub for VS Code</figcaption>

Visual Studio 代码需要知道您是谁才能验证您的 GitHub Copilot 订阅状态。然而，授予对 GitHub 配置文件的访问权限也将允许编辑读取您的*私有*库。如果你改变主意，那么你可以在任何时候撤销这个授权，方法是进入你的 GitHub 配置文件设置，在[授权 OAuth 应用](https://github.com/settings/applications)中找到 **GitHub for VS 代码**。

**注意:**如果你在任何地方遇到困难，请查看官方的[Visual Studio 代码](https://docs.github.com/en/copilot/getting-started-with-github-copilot/getting-started-with-github-copilot-in-visual-studio-code)指南中的 GitHub Copilot 入门。

为了在 Visual Studio 代码中更高效地使用 GitHub Copilot，以下是值得记住的最常见的键盘快捷键:

| 行动 | Windows / Linux | 马科斯 |
| --- | --- | --- |
| 触发内嵌建议 | `Alt` + `\` | `Option` + `\` |
| 见下一条建议 | `Alt` + `]` | `Option` + `]` |
| 参见前面的建议 | `Alt` + `[` | `Option` + `[` |
| 接受建议 | T2`Tab` | T2`Tab` |
| 驳回内嵌建议 | T2`Esc` | T2`Esc` |
| 在新标签中显示所有建议 | `Ctrl` + `Enter` | `Ctrl` + `Enter` |

如果您在使用默认快捷键时遇到问题，那么尝试在 Visual Studio 代码中定义您自己的[键绑定](https://realpython.com/advanced-visual-studio-code-python/#keyboard-shortcuts)。如果您使用的是非美式键盘布局，这可能会特别有帮助。

有时 GitHub Copilot 的建议可能会妨碍您。如果是这种情况，那么您可以通过单击编辑器窗口右下角的扩展图标来全局禁用它们，或者针对特定的编程语言禁用它们:

[![GitHub Copilot Icon in Visual Studio Code](img/560e0675a6143a3af9ce055d1c8f361b.png)](https://files.realpython.com/media/vscode-copilot-icon.eebd6dbe7408.png)

<figcaption class="figure-caption text-center">GitHub Copilot Icon in Visual Studio Code</figcaption>

就是这样！您已经准备好开始在 Visual Studio 代码中使用 GitHub Copilot 扩展。但是如果你更喜欢使用 GitHub Copilot 和 PyCharm，请继续阅读了解如何使用。

[*Remove ads*](/account/join/)

### 安装一个 PyCharm 插件

PyCharm 是由 [JetBrains](https://www.jetbrains.com/) 提供的众多流行集成开发环境之一，共享一个通用的 GitHub Copilot 插件。你可以通过打开 IDE 中的*设置*，并从选项列表中选择*插件*来安装该插件。然后，在 *Marketplace* 选项卡上，搜索 GitHub Copilot 插件并点击其旁边的*安装*按钮:

[![GitHub Copilot Plugin for PyCharm](img/f9e19684bd8c4aee9dcc93a9e19bc21c.png)](https://files.realpython.com/media/screenshot-pycharm-install-plugin.5a9481511399.png)

<figcaption class="figure-caption text-center">GitHub Copilot Plugin for PyCharm</figcaption>

安装插件后，系统会提示您重启 IDE。当你这样做时，你必须通过从 PyCharm 菜单中选择*工具*来登录 GitHub，然后选择 *GitHub 副驾驶*，然后*登录 GitHub* :

[![GitHub Login Menu in PyCharm](img/047250197387e95b7b6ac09ca1f27b49.png)](https://files.realpython.com/media/screenshot-pycharm-login-copilot.2e38ef21eca9.png)

<figcaption class="figure-caption text-center">GitHub Login Menu in PyCharm</figcaption>

这将生成一个伪随机的**设备代码**，你必须将其复制并粘贴到你的网络浏览器中的[设备激活](https://github.com/login/device)页面，登录 GitHub 后你将被带到该页面:

[![Device Activation on GitHub](img/6a60b943352d3306314955f97698fc8f.png)](https://files.realpython.com/media/scrrenshot-github-device-activation.862da44141ca.png)

<figcaption class="figure-caption text-center">Device Activation on GitHub</figcaption>

与 Visual Studio 代码一样，您需要授权 PyCharm 的插件来验证您的身份和相应的 GitHub Copilot 订阅状态。然而，PyCharm 使用的是 **GitHub API** 而不是 OAuth 令牌，所以授权过程看起来有点不同:

[![Authorize GitHub for PyCharm](img/0b3da99b34dd3a4b4c6aa2290de6f60b.png)](https://files.realpython.com/media/screenshot_pycharm_auth.32ababed68d8.png)

<figcaption class="figure-caption text-center">Authorize GitHub for PyCharm</figcaption>

请注意，授权访问您的 GitHub 配置文件将允许插件检索您的配置文件信息，如您的电子邮件地址，并读取您的*私有*存储库。如果你改变主意，那么你可以在任何时候撤销这个授权，方法是进入你的 GitHub 个人资料设置，在[授权的 GitHub 应用](https://github.com/settings/apps/authorizations)中找到 **GitHub Copilot 插件**。

**注:**如果你在某一点上遇到困难，请查看 JetBrains IDE 中官方的[GitHub Copilot 入门指南。](https://docs.github.com/en/copilot/getting-started-with-github-copilot/getting-started-with-github-copilot-in-a-jetbrains-ide)

为了让在 PyCharm 中使用 GitHub Copilot 的工作更有效率，以下是值得记住的最常见的键盘快捷键:

| 行动 | Windows / Linux | 马科斯 |
| --- | --- | --- |
| 触发内嵌建议 | `Alt` + `\` | `Option` + `\` |
| 见下一条建议 | `Alt` + `]` | `Option` + `]` |
| 参见前面的建议 | `Alt` + `[` | `Option` + `[` |
| 接受建议 | T2`Tab` | T2`Tab` |
| 驳回内嵌建议 | T2`Esc` | T2`Esc` |
| 在新标签中显示所有建议 | `Alt` + `Enter` | `Alt` + `Enter` |

有时 GitHub Copilot 自动完成可能会妨碍您。如果是这样的话，你可以通过点击编辑器窗口右下角的插件图标来禁用它们，或者针对特定的编程语言禁用它们:

![GitHub Copilot Icon in PyCharm](img/049a389114ee5dab19f8f2f79fe01c11.png)

<figcaption class="figure-caption text-center">GitHub Copilot Icon in PyCharm</figcaption>

就是这样！您可以开始使用 PyCharm 中的 GitHub Copilot 插件了。

### 将控制权交给 GitHub 副驾驶

现在是时候确保 GitHub Copilot 在 Visual Studio 代码或 PyCharm 中正常工作了。为了检查 GitHub Copilot 是否在您的 **Visual Studio 代码**编辑器中正常工作，创建一个新的文本文件，选择 Python 作为底层编程语言，并开始编写一个示例[函数签名](https://en.wikipedia.org/wiki/Type_signature)，例如`hello()`:

[https://player.vimeo.com/video/731460961?background=1](https://player.vimeo.com/video/731460961?background=1)

<figcaption class="figure-caption text-center">GitHub Copilot in Visual Studio Code</figcaption>

只要您在第一行末尾键入冒号(`:`)来引入新的代码块，GitHub Copilot 就会为您填充建议的函数体。直到你点击 `Tab` 接受或者点击 `Esc` 拒绝，它才会以灰色字体显示。在这种情况下，建议的代码调用 [`print()`](https://realpython.com/python-print/) 函数在屏幕上显示`Hello World`文本。虽然这并不壮观，但它证实了 GitHub Copilot 确实工作正常。

注意:你注意到扩展在你的函数体中填充得有多快了吗？GitHub Copilot 背后的工程团队投入了大量精力来确保给定建议的**低延迟**，以获得更好的开发者体验。

在 **PyCharm** 中使用 GitHub Copilot 实际上和在其他代码编辑器中一样。要验证插件安装是否成功，请尝试不同的示例。开始编写一个函数签名，其名称可能表明您希望将两个数字相加，例如`add(a, b)`:

[https://player.vimeo.com/video/731470794?background=1](https://player.vimeo.com/video/731470794?background=1)

<figcaption class="figure-caption text-center">GitHub Copilot in PyCharm</figcaption>

果然 GitHub Copilot 给出了一个非常明智的建议，返回`a`和`b`之和。注意从函数返回值和在屏幕上打印结果之间的区别。你聪明的虚拟助手可以从函数的名字和参数中推断出意图。

**注:**公平地说，GitHub Copilot 没有什么内在的魔力。它接受了大量高质量数据的训练，允许它根据迄今为止在你的文件或项目中看到的内容来决定**最有可能的输出**。因为这个工具不能理解你的代码，它并不总是能得到正确的建议。

在本教程的其余部分，您将探索 GitHub Copilot 在日常软件工程任务中的几个实际用例。您将学习如何通过获得根据您的编程需求定制的即时代码建议，将您的**生产率**提升到一个全新的水平。

[*Remove ads*](/account/join/)

## 从自然语言合成 Python 代码

因为 GitHub Copilot 接受了自然语言(T0)和不同(T2)编程语言(T3)样本的训练，它似乎理解(T4)两个(T5)领域。因此，用简单的英语或其他自然语言向 GitHub Copilot 解释一个抽象的问题，并期望它用期望的编程语言生成相应的代码是完全可能的。

底层的机器学习模型也能够做相反的事情——即，用自然语言解释一段代码，甚至将一种编程语言翻译成另一种语言。想象一下这对初学者和有创造力的人有多大的帮助，他们对自己想要完成的事情有一个愿景，但是还没有掌握计算机编程。

你将会看到人类和计算机语言之间的翻译在实践中是怎样的。

### 使用 Python 注释描述问题

尽管编程界的有影响力的人物，比如罗伯特·c·马丁认为 T2 的代码注释是反模式的，但是注释有时可以帮助 T4 解释为什么某段代码看起来是这样的。你通常为你未来的自己或者你在同一个代码库工作的队友写评论。

当您将 GitHub Copilot 添加到组合中时，它将成为另一个能够阅读您的代码注释的目标受众。考虑下面 Python 中的**单行注释**，它描述了经典的 [`Hello, World!`](https://en.wikipedia.org/wiki/%22Hello,_World!%22_program) 程序:

```py
# Print "Hello, World!"
```

在您的代码编辑器中输入该注释后，您会注意到 GitHub Copilot 并没有自动获取它。当您选择通过评论与它交流时，您必须打开 GitHub Copilot 侧面板或选项卡才能看到建议。或者，您可以开始键入一些代码，让它自动完成。无论哪种方式，编写上面的注释都应该为您提供以下 Python 代码:

```py
print("Hello, World!")
```

这与您通过编写`hello()`函数存根验证 Visual Studio 代码扩展时得到的建议几乎相同。然而，这一次，您得到的输出略有不同。GitHub Copilot 理解您希望将引用的评论片段视为文字文本，而不是指令。

显然，这对 GitHub Copilot 来说太容易了。通过请求更具体的输出来提高标准怎么样？例如，您可能想用西班牙语反向打印`Hello, World!`:

```py
# Print "Hello, World!" backward in Spanish.
```

在编辑器中刷新 GitHub Copilot 面板后，您会看到新的建议。每次触发 GitHub Copilot 时，它们的数量和质量可能会有所不同。你能从这个评论中得到的最好的答案是:

```py
print("¡Hola, mundo!"[::-1])
```

现在，那令人印象深刻！GitHub Copilot 不仅能生成正确的代码，还能生成有经验的 Pythonic 专家会自己编写的 Pythonic 代码。当你加入更多的评论时，这些建议会变得更加有趣。

### 添加更多注释，增加问题复杂性

用一行注释来描述一个问题是可以的，但是你只能在里面装这么多内容。幸运的是，可以将**多个连续的评论**组合成一个符合逻辑且有凝聚力的故事，GitHub Copilot 会将其视为一个整体。最好在每行上放一个完整的句子，不要换行，但是您可以选择在句子中间包含一个明确的反斜杠(`\`)来标记换行:

```py
# Ask the user to provide a line of text.
# Scan the text for the following mildly offensive words: \
# arse, bloody, damn, dummy.
# If you find any, then replace its letters with asterisks \
# except for the first letter in each offensive word.
# Print the resulting text.

def main():
```

确保使用**语法正确的语言**，并注意你的**标点**以获得准确的结果。在这种情况下，您还可以在注释后添加一个**函数签名**，这为 GitHub Copilot 提供了额外的线索。你将得到的一个建议看起来相当不错:

```py
def main():
    text = input("Enter a line of text: ")
    offensive_words = ["arse", "bloody", "damn", "dummy"]
    for word in offensive_words:
        if word in text:
            text = text.replace(word, word[0] + "*" * (len(word) - 1))
    print(text)
```

如果你正在寻找上面函数中所发生的事情的解释，只需要看看你评论中的文字描述就可以了。建议的代码非常接近您在那里描述的内容。

**注意:** GitHub Copilot 生成各种风格和约定的代码。为了实现一致性和消除不必要的干扰，本教程中所有的代码建议都被重新格式化，以符合 [PEP 8](https://realpython.com/python-pep8/) 风格指南。请记住，在大多数编辑器中，只需按一下按钮，您就可以自己重新格式化代码。

当您调用`main()`函数时，您将能够评估生成的代码执行得有多好:

```py
Enter a line of text: She loves you, dummy.
She loves you, d****.
```

是不是很神奇？你给 GitHub Copilot 一个任务的自然语言描述，它就为你找到了正确的解决方案。

注意:记住，你将得到的建议可能与本教程中给出的不同。有时在得到想要的结果之前需要反复试验，所以如果你没有马上得到满意的结果，试着稍微调整一下你的评论。

关于 GitHub Copilot 需要注意的重要一点是，它实施了许多过滤器来阻止真正的攻击性词语、淫秽内容和敏感信息，如个人数据或秘密 API 密钥。它会尽力不给你包含这些元素的建议。你可以通过引诱 GitHub Copilot 泄露某人的个人信息或秘密来试验这些过滤器:

| 代码片段 | 建议完成 |
| --- | --- |
| `offensive_words = [` | 没有人 |
| `# My phone number is` | `# My phone number is +1 (###) ###-####` |
| `GITHUB_API_KEY =` | `GITHUB_API_KEY = '<GITHUB_API_KEY>'` |

在大多数情况下，它在识别敏感信息和提供随机或匿名输出方面做得很好。然而，过滤机制并不完美，因此理论上它可能会从训练数据集中泄露某人的实际数据。根据官方网站，这种情况发生的可能性很小:

> GitHub Copilot 建议的绝大部分代码都是以前没见过的。我们最新的内部研究表明，大约 1%的情况下，建议可能包含一些长度超过 150 个字符的代码片段，这些代码片段与训练集相匹配。([来源](https://github.com/features/copilot))

稍后您将[更详细地探索这种可能性](#raises-intellectual-property-rights-concerns),但是现在，是时候让 GitHub Copilot 为您解决一个编程难题了。

[*Remove ads*](/account/join/)

### 解答一道编程竞赛谜题

为了测试 GitHub Copilot 的真正能力，您可以复制并粘贴一个**编码挑战**的文本描述片段，看看它会如何应对。例如，为什么不从 2021 年[降临日的第一天开始抓取](https://realpython.com/python-advent-of-code/)[声纳扫描](https://adventofcode.com/2021/day/1)谜题的第一部分代码日历:

```py
# For example, suppose you had the following report:
#
# 199
# 200
# 208
# 210
# 200
# 207
# 240
# 269
# 260
# 263
#
# (...)
#
# In this example, there are 7 measurements that are larger \
# than the previous measurement.
#
# How many measurements are larger than the previous measurement?

def solve(measurements):
```

问题的完整描述稍长，为了节省空间，在上面的代码块中进行了缩写。简而言之，您的目标是确定深度测量相对于前一次测量增加的次数。GitHub Copilot 很好地解决了这个具体问题，考虑到有多少人处理了完全相同的任务并在 GitHub 上分享了他们的解决方案:

```py
def solve(measurements):
    count = 0
    for i in range(1, len(measurements)):
        if measurements[i] > measurements[i - 1]:
            count += 1
    return count
```

这是一个短循环，从测量列表上的第二个项目开始，将当前读数与前一个读数进行比较。如果当前读数更大，那么它增加计数器，函数在最后返回。这种解决方案工作正常，并且相当高效和可读，但是不要觉得有义务以文字形式使用 GitHub Copilot 的建议。这不是全有或全无！

你可以像编辑自己的代码一样编辑 GitHub Copilot 生成的代码。有时，您可能不喜欢它的格式、它应用的命名约定，或者您更愿意重写的特定逻辑片段。在其他场合，看到一个建议可能会激发你想出一个更聪明的替代方案。在这种情况下，您可以使用这个一行程序解决方案，例如:

```py
def solve(measurements):
    return sum(1 for x, y in zip(measurements, measurements[1:]) if y - x > 0)
```

对于每对相邻的测量值，如果它们的差是正的，则取用[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)产生的一序列的和。根据您的 Python 知识水平，您可能更喜欢这个或建议的代码片段。

现在您知道了如何使用一个或多个 Python 注释从自然语言生成代码。然而，作为一名软件工程师，您可能会更加欣赏上下文代码建议。

## 接收更多智能代码完成建议

你可以把 GitHub Copilot 看作是类固醇上的智能代码完成机制，它在深层次上理解你的项目的上下文，为你提供最合适的建议。当你与 GitHub Copilot 一起工作足够长的时间，它有时可能会给你一种毛骨悚然的感觉，它可以读取你的想法。在本节中，您将看到几个演示这种行为的例子。

### 让 GitHub Copilot 读取您的想法

说你要[求二次](https://en.wikipedia.org/wiki/Root-finding_algorithms)[多项式](https://en.wikipedia.org/wiki/Polynomial)的根，也叫[二次函数](https://en.wikipedia.org/wiki/Quadratic_function)，三个系数:𝑎、𝑏、𝑐.您可以使用以下代数形式来表示这样的函数:

![Quadratic Formula](img/36b7dadd38655b85cb807342ca0143b9.png)

例如，具体函数 [𝑥 <sup>2</sup> + 2𝑥 - 3](https://www.wolframalpha.com/input?i=x%5E2+%2B+2x+-+3) 有这些系数值:𝑎 = 1，𝑏 = 2，𝑐 = -3，你可以用它们来计算所谓的[判别式](https://en.wikipedia.org/wiki/Discriminant)，通常用希腊字母 delta 表示:

![Discriminant of a Polynomial](img/86af2dfba5b79810afe8c9c0a06b6b14.png)

在这种情况下，当您用上述公式中的相应值替换字母𝑎、𝑏和𝑐时，函数的判别式等于十六([δ= 16](https://www.wolframalpha.com/input?i=discriminant%28x%5E2+%2B+2x+-+3%29))。根据判别式的符号，您的二次函数可以有两个、一个或根本没有根:

![Root Formulas](img/d48e260dc0b73c28f4e0e1bcace8b72f.png)

由于之前计算的判别式是一个正数，16，您知道您的二次函数正好有两个根，当您应用上面的公式时，它们的值为𝑥 <sub>1</sub> = -3 和𝑥 <sub>2</sub> = 1。将它们中的任何一个作为𝑥变量的值插入到二次函数中，都会使函数返回零。该函数的根是绘制时相应的[抛物线](https://en.wikipedia.org/wiki/Parabola)与横轴相交的点。

现在，您可以实现一个 Python 函数，根据这种多项式的三个系数(𝑎、𝑏和𝑐.)来求其根您将希望通过计算判别式和评估根的正确数量来遵循相同的算法。首先写一个描述性的函数签名，它可能暗示你的意图是什么:

```py
def find_roots(a, b, c):
```

过一会儿，GitHub Copilot 就会开始给你建议。如果你幸运的话，其中一个会惊人的准确:

```py
def find_roots(a, b, c):
    d = b**2 - 4 * a * c
    if d < 0:
        return None
    elif d == 0:
        return -b / (2 * a)
    else:
        return (-b + d**0.5) / (2 * a), (-b - d**0.5) / (2 * a)
```

该函数使用正确的公式计算判别式，并将结果存储在辅助变量中。根据判别式的符号，它返回 [`None`](https://realpython.com/null-in-python/) 、单个根值或由两个根值组成的[元组](https://realpython.com/python-lists-tuples/#python-tuples)，所有这些都计算正确。注意，建议的解决方案使用了**取幂运算符** ( `**`)来计算判别式的平方根。这避免了从 Python 的 [`math`](https://realpython.com/python-math-module/) 模块中导入`sqrt()`函数。

**注意:**你看到的*的建议可能看起来不同，因为人工智能并不完全是确定性的，因为它会不断学习新事物。在其他时候，您甚至可能得到相同输入的替代结果。当您的项目中有额外的源代码时尤其如此，GitHub Copilot 将从中提取上下文。*

好吧，但是生成的函数只能找到[实数](https://en.wikipedia.org/wiki/Real_number)根，如果它们存在的话。如果您想显示所有在[复数](https://en.wikipedia.org/wiki/Complex_number)域中的根呢？在下一节中，您将发现如何给 GitHub Copilot 这样的提示。

[*Remove ads*](/account/join/)

### 提供背景以获得更好的建议

代数的[基本定理](https://en.wikipedia.org/wiki/Fundamental_theorem_of_algebra)陈述了具有复系数的𝑛多项式恰好具有𝑛复数[根](https://en.wikipedia.org/wiki/Zero_of_a_function)。换句话说，作为二次多项式的二次函数，*总是*恰好有两个复数根，即使在实数域中没有复数根。

考虑函数 [𝑥 <sup>2</sup> + 1](https://www.wolframalpha.com/input?i=x%5E2+%2B+1) ，其行列式为负，表示无根。你可以通过画出函数的抛物线来确认没有根，抛物线不穿过横轴。然而，同一个函数有两个复杂的根，𝑥 <sub>1</sub> = -𝑖和𝑥 <sub>2</sub> = 𝑖，其中𝑖是平方后计算结果为-1 的[虚数单位](https://en.wikipedia.org/wiki/Imaginary_unit):𝑖<sup>2</sup>=-1。

你怎么能要求 GitHub Copilot 改变实现，从而得到复杂的根而不是真正的根呢？

您需要通过给 GitHub Copilot 一些可借鉴的上下文来为问题添加约束。例如，您可以[导入您希望使用的模块](https://realpython.com/python-modules-packages/#the-import-statement),或者编写一个用自然语言描述预期结果的 [Python docstring](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings) :

```py
import cmath

def find_roots(a, b, c):
    """Return a tuple of complex numbers."""
```

这里，您导入了`cmath`模块，它是`math`模块的复杂对应物。它包含了`cmath.sqrt()`函数，可以计算负数的平方根，而`math.sqrt()`在这种情况下会产生一个错误:

>>>

```py
>>> import cmath
>>> cmath.sqrt(-1)
1j

>>> import math
>>> math.sqrt(-1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: math domain error
```

复数域中`-1`的平方根产生虚数单位，Python 将其称为`1j`。你可以阅读更多关于[在 Python](https://realpython.com/python-complex-numbers/) 中使用复数的内容，了解为什么它使用字母`j`而不是`i`来表示虚数单位。

您的 docstring 指示函数应该返回的预期数据类型。在某些情况下，你可能需要用更具体的词语来阐明你的期望。例如，写“一个既有复数*又有复数*的元组”将意味着一个正好由两个元素组成的元组。另一方面，单词*对*而不是*元组*会不太明确。

**注意:**除了文档字符串，GitHub Copilot 还能理解你的 Python 代码中的[类型提示](https://realpython.com/python-type-checking/)。

添加了这两个小线索后，GitHub Copilot 现在将为完全相同的函数签名生成不同的实现:

```py
import cmath

def find_roots(a, b, c):
    """Return a tuple of complex numbers."""
    d = (b**2) - (4 * a * c)
    x1 = (-b + cmath.sqrt(d)) / (2 * a)
    x2 = (-b - cmath.sqrt(d)) / (2 * a)
    return x1, x2
```

它像以前一样计算判别式，但不再检查其符号。相反，该函数通过利用`cmath.sqrt()`函数来计算两个复杂的根，正如您所希望的那样。您可以在 Python REPL 中测试您的新函数，以检查该函数是否正确计算了两个复数根:

>>>

```py
>>> import cmath

>>> def find_roots(a, b, c):
...     """Return a tuple of complex numbers."""
...     d = (b**2) - (4 * a * c)
...     x1 = (-b + cmath.sqrt(d)) / (2 * a)
...     x2 = (-b - cmath.sqrt(d)) / (2 * a)
...     return x1, x2
...
>>> find_roots(1, 0, 1)  # Function f(x) = x² + 1
(1j, -1j)

>>> 1j**2 + 1
0j

>>> (-1j)**2 + 1
0j
```

这段代码非常棒！函数`x² + 1`返回两个复杂根`1j`和`-1j`的`0`。

即使这没有让你感到特别兴奋，你可能会对 GitHub Copilot 的创造力印象深刻，它可以节省你很多时间。接下来，您将使用 GitHub Copilot 生成一个类的主体。

### 受益于 GitHub Copilot 的创造力

有多少次，你在设计一种新的数据类型时，却被正确的属性或它们的实现所困扰？有了 GitHub Copilot，你可以高枕无忧，因为它会发明新的属性，[方法](https://realpython.com/instance-class-and-static-methods-demystified/)，以及[属性](https://realpython.com/python-property/)，按下 `Tab` 。

假设您想使用 Python 的[数据类](https://realpython.com/python-data-classes/)定义一个`Person`类。首先给新的数据类型起一个有意义的名字，并引入第一个属性，称为`.first_name`:

[https://player.vimeo.com/video/731808891?background=1](https://player.vimeo.com/video/731808891?background=1)

<figcaption class="figure-caption text-center">GitHub Copilot Suggesting Class Attributes</figcaption>

GitHub Copilot 通过建议下一个最可能的属性`.last_name`，然后是`.age`，立即从那里开始。然而，你知道一个人的年龄会随着时间的推移而变化，所以你记录了他的出生日期。GitHub Copilot 的下一个逻辑建议是基于当前日期计算一个人年龄的方法。当您定义一个新的属性时，它通过连接名字和姓氏来完整地完成它的主体。

最后，这是您在代码编辑器中只需几次击键就能得到的结果:

```py
from dataclasses import dataclass
from datetime import date

@dataclass
class Person:
    first_name: str
    last_name: str
    birth_date: date

    def age(self):
        return (date.today() - self.birth_date).days // 365

    @property
    def full_name(self):
        return f"{self.first_name}  {self.last_name}"
```

这是前所未有的巨大的时间节省和生产力提升的 T2。即使你知道该键入什么，GitHub Copilot 也能让你以思维的速度编码，猜测你可能在想什么，然后提供一个合适的建议，你只需点击一个按钮或按键就能接受。

[*Remove ads*](/account/join/)

### 成为一名拥有私人翻译的编程语言者

在一个文件中混合多种语言并不少见，比如 [HTML](https://developer.mozilla.org/en-US/docs/Web/HTML) 、 [CSS](https://developer.mozilla.org/en-US/docs/Web/CSS) 、 [JavaScript](https://realpython.com/python-vs-javascript/) 、 [Django 模板语言](https://realpython.com/django-templates-tags-filters/)和 Python。幸运的是，GitHub Copilot 知道十几种编程语言，甚至更多的框架和几种人类语言。它可以根据上下文在它们之间随意切换，完全不会打断你的流程。

例如，您可能希望定义一个 Python 变量来存储一个通过匹配名称检索用户的 SQL 查询**。只要在变量名中使用正确的单词，您应该会得到一个合理的建议，比如这个:**

[https://player.vimeo.com/video/731837821?background=1](https://player.vimeo.com/video/731837821?background=1)

<figcaption class="figure-caption text-center">GitHub Copilot Suggesting a SQL Query</figcaption>

请注意，您是如何在各个阶段中分别获得各个行的，因为您使用三重引号(`"""`)定义了一个多行字符串文字。上面视频中描绘的结果如下:

```py
query_users_by_first_or_last_name = """
SELECT * FROM users
WHERE first_name LIKE %s OR last_name LIKE %s """
```

乍一看，这个查询看起来不错，尽管 GitHub Copilot 对您的表名和要搜索的两列做了一些假设。不过，令人欣慰的是，它生成了一个[预准备语句](https://en.wikipedia.org/wiki/Prepared_statement)，而不是一个带有 SQL 查询的普通字符串，从而帮助[阻止了一次 SQL 注入攻击](https://realpython.com/prevent-python-sql-injection/)。

好了，现在你对 GitHub Copilot 已经很熟悉了。但是请耐心等待，因为还有很多内容要介绍！

## 与虚拟伙伴练习结对编程

甚至在访问 GitHub Copilot 官方网站之前，你会很快在网络搜索结果中注意到它被宣传为一个*人工智能配对程序员*。简而言之，[结对编程](https://en.wikipedia.org/wiki/Pair_programming)是一种流行的[敏捷](https://en.wikipedia.org/wiki/Agile_software_development)技术，涉及到两个工程师一起工作在同一个功能上。从表面上看，生产这样的软件会花费更多的钱，但是从长远来看，它保证了更少的昂贵的修复错误。

结对编程的好处包括:

*   生成代码的质量更高
*   整个团队对代码库有更好的整体理解
*   知识和最佳实践的**分享**

很难超越坐在一个真实的人旁边的优势，他可以对你的代码给出诚实的反馈，发现潜在的问题，并把你引向正确的方向。结对编程的许多好处只有在团队协作时才能看到。另一方面，人工智能可能会给你糟糕的建议，导致代码质量下降。最终，要不要听由你自己决定！

也就是说，GitHub Copilot 在提高您的工作效率方面非常出色，您将会发现这一点。

### 生成用于测试的样本数据夹具

在日常开发过程中，您经常需要考虑测试、文档或默认设置的样本数据。如果你一片空白，那么 GitHub Copilot 将会帮助你。例如，回想一下您之前定义的`Person`类:

```py
from dataclasses import dataclass
from datetime import date

@dataclass
class Person:
    first_name: str
    last_name: str
    birth_date: date

    def age(self):
        return (date.today() - self.birth_date).days // 365

    @property
    def full_name(self):
        return f"{self.first_name}  {self.last_name}"
```

现在，假设您需要创建这个类的几个实例来表示一家公司的雇员。当您仍在编辑同一个文件或从另一个模块导入了`Person`类时，您可以使用 GitHub Copilot 来填充员工列表。使用一个有表现力的变量名声明一个空的 Python 列表，并在左方括号(`[`)后点击 `Enter` 来触发建议:

```py
employees = [
```

当您在每个建议行后不断点击 `Tab` 时，您可能会得到以下员工列表:

```py
employees = [
    Person("John", "Smith", date(1970, 1, 1)),
    Person("Jane", "Doe", date(1980, 1, 1)),
    Person("Joe", "Doe", date(1990, 1, 1)),
    Person("Jack", "Doe", date(2000, 1, 1)),
    Person("Jill", "Doe", date(2010, 1, 1)),
    Person("Jana", "Doe", date(2020, 1, 1)),
]
```

GitHub Copilot 成功地将您的`Person`类用于员工列表元素。稍后，您可以将该列表用作一组需要它的[测试用例](https://en.wikipedia.org/wiki/Test_case)的通用[测试夹具](https://en.wikipedia.org/wiki/Test_fixture)。例如，您可以将列表包装在测试框架将调用的函数中。GitHub Copilot 通过建议测试和被测代码，在[测试](https://realpython.com/python-testing/)过程中提供了很好的帮助。

[*Remove ads*](/account/join/)

### 希望测试用例神奇地出现

给你的测试函数起一个长的描述性的名字是一个很好的习惯，因为它们会在测试运行报告中显示出来。当其中一个失败时，一个名副其实的测试函数会立即将您的注意力吸引到失败的地方。[行为驱动开发的倡导者](https://en.wikipedia.org/wiki/Behavior-driven_development)建议使用以单词*应该*开头的整句话来关注被测试的行为，这使得测试看起来像一个业务需求规范。

这种有时长得离谱的函数名的额外好处是 GitHub Copilot 可以使用它们来帮助您生成测试用例实现。例如，在`Person`类的测试模块中定义以下函数签名:

```py
def test_should_not_be_able_to_set_full_name():
```

大多数测试框架会自动发现测试用例，如果它们遵循标准的命名约定，比如当你在测试函数名前面加上`test_`的时候。使用 GitHub Copilot，为上述功能触发的建议可能如下所示:

```py
def test_should_not_be_able_to_set_full_name():
    person = Person("John", "Doe", date(1980, 1, 1))
    with pytest.raises(AttributeError):
        person.full_name = "Jane Doe"
```

奇怪的是，GitHub Copilot 更喜欢外部的 [pytest](https://realpython.com/pytest-python-testing/) 库，你必须手动安装和导入，而不是标准库中的内置`unittest`模块。

**注意:**这个选择可能会告诉你这两个工具的流行程度，事实上，pytest 可以说是 Python 生态系统中最广泛和最通用的测试框架之一。

虽然自动实现现有代码的测试用例有时可能是有帮助的，但是使用测试驱动的开发，以自顶向下的方式翻转步骤和开发软件可能更令人满意。在这种方法中，您首先编写您的测试用例作为代码的高级规范，这还不存在。一旦你有了一个自动化测试用例，然后你写一些代码使它通过。

### 测试驱动开发

如果你以前没有练习过 **TDD** ，那么看看[用 TDD 在 Python 中构建哈希表](https://realpython.com/python-hash-table/)教程，这是一个关于测试驱动开发的实践性的、循序渐进的速成课程。

简而言之，您可以将该过程总结为三个步骤:

1.  写一个你将要满足的失败测试案例
2.  实现最少的代码来让你的测试用例通过
3.  可选地，当所有测试用例仍然通过时，重构代码

然后，冲洗，重复！只要你足够自律以保持这种永无休止的循环，你就会编写具有高[测试覆盖率](https://en.wikipedia.org/wiki/Code_coverage)和文档化的**可测试代码**。同时，您将避免编写永远不需要的代码，从而降低整体维护成本。也就是说，测试驱动的开发不是一个银弹，所以它在研发项目中可能不实用。

好吧，复制上面提到的教程中的`HashTable`实现怎么样，使用 GitHub Copilot 作为你的虚拟对程序员？在同一文件夹中创建两个相邻的空 Python 文件:

```py
src/
├── hashtable.py
└── test_hashtable.py
```

第一个，在下面视频的左边，将包含被测试的代码。另一个，描绘在右边，将是你驱动实现的测试用例的家。接下来，编写第一个测试用例来验证一个新的`HashTable`类的实例化:

[https://player.vimeo.com/video/732011037?background=1](https://player.vimeo.com/video/732011037?background=1)

<figcaption class="figure-caption text-center">GitHub Copilot Helping With Test-Driven Development</figcaption>

注意，为了遵循测试驱动开发，您应该在实现相应的代码之前和之后运行每个测试用例，以确保您测试的是正确的东西。除此之外，你应该*只*实现最基本的功能来满足你的测试。但是 GitHub Copilot 领先一步，尝试预测您未来可能需要的代码。这不是一个严格的真正的测试驱动开发方法。

概括地说，这是您刚刚编写的两个测试用例，用于检查您是否可以创建一个具有或不具有初始容量的哈希表:

```py
# test_hashtable.py

from hashtable import HashTable

def test_should_create_hashtable():
    assert HashTable() is not None

def test_should_create_hashtable_with_capacity():
    assert HashTable(capacity=10) is not None
```

基于它们，GitHub Copilot 目前已经生成了以下哈希表实现:

```py
# hashtable.py

class HashTable:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.buckets = [None] * capacity
        self.size = 0
```

initializer 方法将默认容量设置为 10，以防有人在没有提供任何参数的情况下创建哈希表，这是由您的第一个测试用例决定的。容量随后存储在实例属性中。注意，GitHub Copilot 正确地识别了哈希表的大小和容量之间的差异。它还假设了基于[的经典散列表实现，通过创建空桶来分离链接](https://realpython.com/python-hash-table/#isolate-collided-keys-with-separate-chaining)。

虽然 GitHub Copilot 在监视你的背后方面可能不如人类程序员，但它作为虚拟助理为你的问题提供答案，确实做得非常好。这就像有人在 Google 或 Stack Overflow 中搜索您的独特问题的解决方案，内置在您需要的代码编辑器中！

[*Remove ads*](/account/join/)

## 消除堆栈溢出，实现即时的环境感知解决方案

互联网上充斥着关于过度使用像 Stack Overflow 或 Google 这样的网站来寻找程序员问题答案的笑话。比如曾经有过[对热门](https://dev.to/rly)[奥赖利](https://www.oreilly.com/)书籍封面的模仿，比如著名的*从栈溢出*复制粘贴:

[![Copying and Pasting from Stack Overflow](img/46a9054000f9fcac8bd735424087e24f.png)](https://files.realpython.com/media/so.48f6a9886871.png)

<figcaption class="figure-caption text-center">Copying and Pasting from Stack Overflow (Image: [@ThePracticalDev](https://twitter.com/ThePracticalDev/status/705825638851149824))</figcaption>

这并不奇怪，因为大多数开发人员在日常工作中经常使用这些工具，现在没有它们将很难生活。

GitHub Copilot 可以减少你对那些工具的依赖。它可以立即交付定制的**代码片段**，像手套一样适合你的项目环境，甚至不同于你在网上找到的最佳答案。此外，它不会强迫您离开代码编辑器，因此您可以呆在该区域并加速您的进度。请继续阅读，体验一下随手可得的 Python 社区的集体知识。

### 再也不要想样板代码了

Python 有一个相对富于表现力的语法，它允许你编写简洁易读的代码。然而，偶尔的[样板代码](https://en.wikipedia.org/wiki/Boilerplate_code)是不可避免的，即使是在你最喜欢的编程语言中。有时，没有办法编写一段看起来冗长的代码，但看起来并没有做很多有用的工作。

比如[从一个 URL 地址获取数据](https://realpython.com/python-requests/)，序列化到 [JSON](https://realpython.com/python-json/) ，将结果保存到一个[文件](https://realpython.com/working-with-files-in-python/)涉及到很多无聊的细节，贡献不了多少价值。如果您能够专注于对您最重要的事情，也就是说，在让 GitHub Copilot 将各个步骤映射到代码片段的同时，整理出一个更高层次问题的解决方案，这不是很酷吗？

嗯，原来可以！和以前一样，首先编写一个描述性的函数签名，剩下的交给 GitHub Copilot:

```py
def save_json(url, filename):
```

响应上述签名的建议函数体可能如下:

```py
def save_json(url, filename):
    import json
    import requests
    import os

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(json.dumps(requests.get(url).json()))
    else:
        print("File already exists")
```

它依赖于外部的[请求](https://pypi.org/project/requests/)库，您可能需要安装这个库，因为它是用 Python 发出 HTTP 请求的[事实上的标准](https://en.wikipedia.org/wiki/De_facto_standard)。该函数利用[上下文管理器](https://realpython.com/python-with-statement/)在将 JSON 转储到文件后[关闭文件](https://realpython.com/why-close-file-python/)。这已经足够好了，但是您可能需要添加一些修饰性的调整:

```py
import json
import os
import requests

def save_json(url, filename):
    if not os.path.exists(filename):
        with open(filename, mode="w", encoding="utf-8") as file:
            file.write(json.dumps(requests.get(url).json()))
    else:
        print("File already exists")
```

根据 [PEP 8](https://realpython.com/python-pep8/) 风格指南，你通常应该在函数之外定义你的导入语句，外部库模块应该在标准库模块之后。此外，[建议](https://realpython.com/python310-new-features/#default-text-encodings)在使用 Python 处理文件时，明确说明[字符编码](https://realpython.com/python-encodings-guide/)，比如 UTF-8，以避免不同操作系统之间的差异，这些差异可能会导致不兼容的默认设置。

**注意:**你可以使用[黑](https://pypi.org/project/black/)或者类似的工具来保持生成代码的格式与你项目的其余部分一致。

使用 GitHub Copilot 可能比堆栈溢出稍微安全一些。有了堆栈溢出，您可能会发现自己盲目地逐字复制和粘贴别人的代码，而没有完全理解它或考虑它在您的情况下是否有意义。接受 GitHub Copilot 的建议可能同样鲁莽，如果不是更鲁莽的话，但至少它给了你一个微调的、上下文特定的拼图，更有可能工作。

GitHub Copilot 的另一个亮点是让您不必深究您想要使用的库或 API 的文档。

### 总是将 API 文档放在手边

假设您想编写一个小的 Python 函数来获取 GitHub 用户的公共存储库列表。在传统方法中，你首先在网上搜索 *github api* ，然后登陆 [GitHub REST API](https://docs.github.com/en/rest) 文档页面。然后，您可能会被可供选择的 REST APIs 的数量，以及它们所有的指南、快速入门和参考文档所淹没。

幸运的是，你有 GitHub Copilot，它已经被训练使用**众所周知的 API**，所以你可以给它一个关于调用哪个 API 的最小提示。创建一个名为`github_api_client`的新 Python 模块，并在其中键入以下代码:

```py
# github_api_client.py

import os

GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")

def get_repository_names(username):
```

稍后，您将在一个终端会话中运行这个脚本，您的令牌存储在一个[环境变量](https://en.wikipedia.org/wiki/Environment_variable)中。习惯上通过环境变量读取**密钥**和**配置**数据，所以你利用 Python 的`os`模块读取你的个人 **GitHub API 令牌**，你可能需要它来访问 API。

**注意:**虽然你不需要令牌来获取某人的公共存储库，但是作为一个匿名客户端，你将被[限制在每小时六十个 API 请求](https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting)。为了突破这个限制，你需要使用你的[个人访问令牌](https://github.com/settings/tokens)来验证你自己。现在这样做是有意义的，因为大多数 API 端点无论如何都需要认证。

其中一个建议的结果是开箱即用的:

```py
# github_api_client.py

import os

GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")

def get_repository_names(username):
    import requests

    url = f"https://api.github.com/users/{username}/repos"
    headers = {"Authorization": f"token {GITHUB_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return [repo["name"] for repo in response.json()]
```

在测试这个功能之前，记得在 GitHub 概要文件上生成一个新的[个人访问令牌](https://github.com/settings/tokens)，并在终端中设置相应的环境变量:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
PS> $env:GITHUB_API_TOKEN=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
```

```py
$ export GITHUB_API_TOKEN=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
```

然后，当您仍然在定义环境变量的同一个终端会话中时，以一种[交互模式](https://docs.python.org/3/using/cmdline.html#cmdoption-i)运行带有您的代码的源文件，以便 Python 读取您可以调用的生成函数:

>>>

```py
$ python -i github_api_client.py
>>> for name in get_repository_names("gvanrossum"):
...     print(name)
...
500lines
asyncio
ballot-box
cpython
ctok
exceptiongroup
guidos_time_machine
gvanrossum.github.io
http-get-perf
minithesis
mirror-cwi-stdwin
mypy
mypy-dummy
path-pep
patma
pep550
peps
Pyjion
pythonlabs
pythonlabs-com-azure
pytype
pyxl3
```

在这种情况下，你得到的是由 Python 的创造者[吉多·范·罗苏姆](https://github.com/gvanrossum)制作的一个相当短的公共库列表。

好吧，使用一个众所周知的 API 并不是特别困难，但是利用一个 GitHub Copilot 以前没有见过的自定义 API 怎么样呢？好吧，接下来你就知道了。

[*Remove ads*](/account/join/)

### 教 GitHub 副驾驶说你自己的方言

假设您有以下 Python 模块，它通过在三个主要平台之一上包装一个[文本到语音(TTS)](https://en.wikipedia.org/wiki/Speech_synthesis) 命令来定义一个用于语音合成的**定制 API** :

```py
# custom_api.py

import platform
import subprocess

class TextToSpeechAPIClient:
    def __init__(self, command=None):
        self.command = command or get_default_command()

    def say(self, text):
        subprocess.call(self.command(text))

def get_default_command():
    match platform.system():
        case "Darwin":
            return lambda text: ["say", text]
        case "Linux":
            return lambda text: ["spd-say", text]
        case "Windows":
            return lambda text: \
            "PowerShell -Command \"Add-Type –AssemblyName System.Speech; " \
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer)." \
            f"Speak('{text}');\""
```

`TextToSpeechAPIClient`类为与您的操作系统相关的语音合成命令提供一个可选参数。下面定义的助手函数利用 Python 3.10 中引入的[匹配语句](https://realpython.com/python310-new-features/#structural-pattern-matching)来确定正确的命令。

现在，创建另一个 Python 模块，并告诉 GitHub Copilot 通过在评论中许愿来学习如何使用您的新 API:

```py
# main.py

# Say "Hello, World!" using the custom text-to-speech API client.

if __name__ == "__main__":
```

文件末尾的[执行条件](https://realpython.com/python-main-function/#use-if-__name__-__main__-to-control-the-execution-of-your-code)通过传达您想要编写一个 Python 脚本，为 GitHub Copilot 提供了必要的上下文。否则，你可能会得到不太有用的建议。然而，有了这个重要的检查，其中一个建议看起来将是这样的:

```py
# main.py

# Say "Hello, World!" using the custom text-to-speech API client.

if __name__ == "__main__":
    import custom_api

    client = custom_api.TextToSpeechAPIClient()
    client.say("Hello, World!")
```

太棒了。GitHub Copilot 在另一个 Python 模块中找到了您的自定义 API，在需要的地方导入它，并完全按照它应该的那样使用`TextToSpeechAPIClient`类。虽然它在本地模块上做得很好，但它也可以为第三方模块提供代码片段。

### 导航一个不熟悉的框架或库

用 GitHub Copilot 探索 Python 中的新库是一种令人愉快的体验。也许你正在重新访问一个你的知识已经有点生疏的框架，你正在努力回忆从哪里开始。但是，即使你非常熟悉一个给定的工具，GitHub Copilot 有时也会给你带来惊喜，它会建议你更有效的解决方案或你不知道的 API 部分。

假设您想要使用 [Flask](https://realpython.com/python-web-applications/) 框架公开一个 [REST API](https://realpython.com/api-integration-in-python/) 端点，以提供从基于文件的 [SQLite](https://realpython.com/python-sqlite-sqlalchemy/) 数据库查询的 JSON 序列化图书列表。现在，创建一个新的 Python 源文件，导入必要的模块，并将本段中的前一句改写成 Python 注释:

```py
import sqlite3
import flask

# Expose a REST API endpoint using the Flask framework \
# to serve a JSON-serialized list of books queried from \
# a file-based SQLite database.
```

等待建议到达，然后选择一个看起来最好的。有时，当您开始键入预期的代码并通过在每一行点击 `Tab` 逐行请求建议时，您会获得更好的结果。最终，您可能会得到这样的结果:

```py
import sqlite3
import flask

# Expose a REST API endpoint using the Flask framework \
# to serve a JSON-serialized list of books queried from \
# a file-based SQLite database.

app = flask.Flask(__name__)

@app.route("/books")
def get_books():
    conn = sqlite3.connect("books.db")
    c = conn.cursor()
    c.execute("SELECT * FROM books")
    books = c.fetchall()
    conn.close()
    return flask.jsonify(books)

if __name__ == "__main__":
    app.run(debug=True)
```

在本例中，GitHub Copilot 搭建了 Flask 应用程序的典型结构，您可能已经忘记了，它定义了您所请求的端点。为了测试它是否工作，在 Flask 应用程序所在的同一文件夹中打开您的终端，并创建一个名为`books.db`的新的基于文件的数据库，其中包含一些样本书籍。

要输入 SQL 查询，您可以在批处理模式下运行`sqlite3`命令，并用[文件尾(EOF)](https://en.wikipedia.org/wiki/End-of-file) 字符终止它:

```py
$  sqlite3  books.db  -batch CREATE  TABLE  books( id  INTEGER  PRIMARY  KEY  AUTOINCREMENT, isbn  TEXT, author  TEXT, title  TEXT ); INSERT  INTO  books(isbn,  author,  title)  VALUES ('978-0132350884',  'Robert C. Martin',  'Clean Code'), ('978-1449340377',  'David Beazley',  'Python Cookbook'), ('978-0131872486',  'Bruce Eckel',  'Thinking in Java'), ('978-1775093329',  'David Amos',  'Python Basics');
```

在 Windows 上，您通常可以使用 `Ctrl` + `Z` 将 EOF 字符发送到[标准输入流](https://en.wikipedia.org/wiki/Standard_streams#Standard_input_(stdin))，而在 Linux 和 macOS 上，您将使用 `Ctrl` + `D` 组合键。

接下来，通过运行脚本，在默认网络接口和端口号上启动 Flask 应用程序，然后将 web 浏览器导航到`/books`端点。或者，如果你使用的是 macOS 或 Linux，你可以直接在终端中使用类似于 [cURL](https://en.wikipedia.org/wiki/CURL) 的命令来获取书籍:

```py
$  curl  http://127.0.0.1:5000/books [ [ 1, "978-0132350884", "Robert C. Martin", "Clean Code" ], [ 2, "978-1449340377", "David Beazley", "Python Cookbook" ], [ 3, "978-0131872486", "Bruce Eckel", "Thinking in Java" ], [ 4, "978-1775093329", "David Amos", "Python Basics" ] ]
```

非常引人注目的是，GitHub Copilot 生成的代码是活的！想想吧。您仅仅提供了所需程序的自然语言描述，并导入了两个模块来给出使用哪个库的上下文。与此同时，人工智能将你的描述变成了一个工作的网络应用。

也就是说，GitHub Copilot 也有其黑暗的一面，您将在下一节中了解到。

[*Remove ads*](/account/join/)

## 考虑反对使用 GitHub Copilot 的理由

当你在网上搜索关于 GitHub Copilot 的信息时，你会发现赞扬和兴奋的话语，以及相当多的批评。有些是正确的，而有些则源于对技术目的的误解。最终，GitHub Copilot 对不同的人意味着不同的东西。读完这一节，你会意识到你的人工智能虚拟助手的最大缺点。

### 绝对不是自动驾驶！

名为 *copilot* 是 GitHub 描述这款产品的一个聪明的选择。它避免误导人们认为它可以控制和取代你作为程序员。就像航空业的[副驾驶](https://en.wikipedia.org/wiki/First_officer_(aviation))，通常被称为副驾驶，GitHub 副驾驶协助你，但不为你工作。这一措辞与[高级驾驶辅助系统](https://en.wikipedia.org/wiki/Advanced_driver-assistance_system)形成鲜明对比，如今一些汽车制造商直截了当地称之为*自动驾驶系统*。

你必须记住 GitHub Copilot 产生的代码并不总是理想的。事实上，它有时可能是次优的或不安全的，并且可能遵循糟糕的编程实践。

当代码包含语法错误或者完全没有意义时，问题就不大了，因为这很容易被发现。然而，乍一看似乎合理的代码仍然可能是不正确的，并且包含逻辑错误。因此，在没有仔细检查和测试 GitHub Copilot 生成的代码之前，你应该*永远不要*相信它！

换句话说，GitHub Copilot 是一个非常有价值的工具，但是你绝对不应该仅仅依赖它的建议，比如为一个核电站构建关键任务软件。

### 存在潜在的安全风险

因为 GitHub Copilot 背后的机器学习模型是在公共存储库上训练的，所以它容易受到[数据中毒](https://en.wikipedia.org/wiki/Adversarial_machine_learning#Data_poisoning)的影响。不良行为者可能故意用恶意代码污染训练数据集，这些恶意代码可能会欺骗模型在您的代码编辑器中建议类似的模式。由于训练数据中的大量代码，这也可能是偶然发生的。

再次声明，使用 GitHub Copilot 风险自担！

### 引发知识产权问题

GitHub Copilot 可能侵犯了知识产权，这是目前最严重的问题。因为它基于以前看到的东西合成代码，所以它可能会从私人储存库、有版权的公共储存库，甚至是拥有非许可许可证的开源储存库泄露**专有算法**。

**注意:**要使用 GitHub Copilot，您必须接受其[条款和条件](https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features#github-copilot)，其中明确声明您同意共享位于您的存储库中的代码和数据，包括您的*私人*存储库，以改进服务。

Flask 的创造者[阿明·罗纳彻](https://twitter.com/mitsuhiko)，最近在[的推特](https://twitter.com/mitsuhiko/status/1410886329924194309)上展示了 GitHub Copilot 如何不折不扣地引用[雷神之锤](https://en.wikipedia.org/wiki/Quake_III_Arena)对[快速平方根倒数](https://en.wikipedia.org/wiki/Fast_inverse_square_root#Overview_of_the_code)算法的实现。虽然生成的代码属于[约翰·卡马克](https://en.wikipedia.org/wiki/John_Carmack)，GitHub Copilot 建议在它的基础上使用完全随机的许可证:

[https://player.vimeo.com/video/732243025?background=1](https://player.vimeo.com/video/732243025?background=1)

<figcaption class="figure-caption text-center">Armin Ronacher's Tweet</figcaption>

上例中生成的代码甚至包括带有诅咒词的原始注释。GitHub 声称已经安装了过滤器，理论上应该可以屏蔽这些词。他们还应该删除敏感信息，如 API 密钥或电子邮件地址，这些信息可能会意外提交给 GitHub。然而，你永远无法确定。

### 鼓励作弊

如果你想掌握一门新的编程语言、框架、库或 API，GitHub Copilot 会是一个很好的学习工具。与此同时，不难想象一些学生滥用 GitHub Copilot 作弊，用简单的英语复制并粘贴任务描述，并在几秒钟内完成任务。回想一下你在本教程开始时做的代码拼图的[出现。](#solve-a-programming-contest-puzzle)

此外，没有作弊意图的学生可能更愿意远离 GitHub Copilot，因为这可能会使他们变得懒惰，并阻碍他们独立思考。像所有强大的工具一样，你可以出于正确或错误的目的使用 GitHub Copilot。

### 需要订阅计划

对 GitHub Copilot 最常见的抱怨之一是，它需要[付费订阅](https://docs.github.com/en/billing/managing-billing-for-github-copilot/about-billing-for-github-copilot)，这限制了潜在用户的数量。许多人似乎对被收费购买一个被认为是基于开源代码的工具感到不安。

更糟糕的是，无论你在世界的哪个角落，固定费用都是固定的。它没有考虑购买力平价，这使得一些国家的用户比其他国家的用户负担更重。如果你在一个购买力很高的国家从事工程工作，或者如果你的公司批量购买订阅，那么这个价格可能看起来是合理的。否则，价格可能会非常高。

另一方面，GitHub 至少提供了试用期，并让经过验证的学生和开源维护者可以免费使用这项服务，这很公平。最终，尽管价格可能会让一些人望而却步，但它通常是物有所值的，尤其是考虑到如此庞大的神经网络基础设施的维护成本。毕竟，它需要一个分布式集群来运行并实现低延迟。

如果你想更深入地了解更多关于底层机器学习模型的知识，那么请阅读 OpenAI Codex 上的原始[论文](https://arxiv.org/abs/2107.03374)。

## 结论

GitHub Copilot 是一款革命性的编程辅助工具，可以提高你作为软件工程师的速度和效率。它通过生成**样板代码**来节省您的时间，让您不必钻研**文档**。因为它了解你项目的**背景**，你得到的**即时建议**是量身定制的，通常以你想要的方式工作。

**在本教程中，您学习了如何:**

*   在你的代码编辑器中安装 **GitHub Copilot 扩展**
*   将任务的自然语言描述转换成工作代码
*   在多个可选的**智能代码完成**建议之间进行选择
*   探索不熟悉的框架和编程语言
*   教 GitHub Copilot 如何使用您的**自定义 API**
*   使用虚拟对程序员实时练习测试驱动开发

你对 GitHub Copilot 有什么看法？值这个价吗？你是否足够着迷于在未来的项目中使用它？在下面给我们留下评论吧！

**免费下载:** [点击这里下载免费的键盘快捷键备忘单](https://realpython.com/bonus/github-copilot-python-pdf/)让 GitHub Copilot 的编码更快。*************