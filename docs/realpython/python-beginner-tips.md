# 学习 Python 编程的 11 个初学者技巧

> 原文：<https://realpython.com/python-beginner-tips/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**学习 Python 的 11 个初学者小技巧**](/courses/python-beginner-tips/)

你们决定踏上[学习 Python](https://realpython.com/products/python-basics-book/) 的旅程，我们太激动了！我们从读者那里收到的最常见的问题之一是*“学习 Python 的最好方法是什么？”*

我相信学习任何编程语言的第一步是确保你理解如何学习。学习如何学习可以说是计算机编程中最关键的技能。

为什么知道如何学习如此重要？答案很简单:随着语言的发展，库被创建，工具被升级。知道如何学习对于跟上这些变化并成为一名成功的程序员至关重要。

在本文中，我们将提供几个学习策略，帮助您开始成为 rockstar Python 程序员的旅程！

**免费下载:** [从《Python 基础:Python 3 实用入门》中获取一个示例章节](https://realpython.com/bonus/python-basics-sample-download/)，看看如何通过 Python 3.8 的最新完整课程从初级到中级学习 Python。

## 让它粘在一起

这里有一些提示，可以帮助你作为一个初学程序员真正坚持你正在学习的新概念:

[*Remove ads*](/account/join/)

### 技巧 1:每天编码

当你学习一门新语言时，连贯性是非常重要的。我们建议每天对代码做出承诺。可能很难相信，但是肌肉记忆在编程中起着很大的作用。致力于每天编码将真正有助于开发肌肉记忆。虽然一开始看起来令人畏惧，但是可以考虑从每天 25 分钟开始，然后一点一点地努力。

查看 Python 指南的[第一步，了解关于设置的信息以及帮助您入门的练习。](https://realpython.com/learn/python-first-steps/)

### 技巧 2:把它写出来

作为一名新程序员，当你在你的旅程中前进时，你可能想知道你是否应该做笔记。是的，你应该！事实上，研究表明，手写笔记最有利于长期记忆。这对那些致力于成为全职开发人员的人来说尤其有益，因为许多[面试](https://realpython.com/python-coding-interview-tips/)都会涉及到在白板上写代码。

一旦你开始做小项目和程序，手写也可以帮助你在使用计算机之前规划你的代码。如果您写出您将需要哪些函数和类，以及它们将如何交互，您可以节省大量时间。

### 技巧 3:去互动！

无论你是在学习基本的 Python 数据结构(字符串、列表、字典等。)第一次，或者正在调试应用程序，交互式 Python shell 将是您最好的学习工具之一。我们在这个网站上也经常使用它！

要使用交互式 Python shell(有时也称为[“Python REPL”](https://realpython.com/interacting-with-python/))，首先要确保你的电脑上安装了 Python。我们有一个循序渐进的教程来帮助你做到这一点。要激活交互式 Python shell，只需打开您的终端并根据您的安装运行`python`或`python3`。你可以在这里找到更具体的方向[。](https://realpython.com/learn/python-first-steps/)

现在您已经知道了如何启动 shell，下面是几个在学习时如何使用 shell 的示例:

***学习使用 dir():*** 可以对元素执行哪些操作

>>>

```py
>>> my_string = 'I am a string'
>>> dir(my_string)
['__add__', ..., 'upper', 'zfill']  # Truncated for readability
```

从`dir()`返回的元素是您可以应用于该元素的所有方法(即动作)。例如:

>>>

```py
>>> my_string.upper()
>>> 'I AM A STRING'
```

注意，我们调用了`upper()`方法。你能看出它是干什么的吗？它使字符串中的所有字母都大写！在本教程的[“操纵字符串”中了解更多关于这些内置方法的信息。](https://realpython.com/learn/python-first-steps/)

***学习一个元素的类型:***

>>>

```py
>>> type(my_string)
>>> str
```

***使用内置帮助系统获取完整文档:***

>>>

```py
>>> help(str)
```

***导入库，玩玩:***

>>>

```py
>>> from datetime import datetime
>>> dir(datetime)
['__add__', ..., 'weekday', 'year']  # Truncated for readability
>>> datetime.now()
datetime.datetime(2018, 3, 14, 23, 44, 50, 851904)
```

***运行 shell 命令:***

>>>

```py
>>> import os
>>> os.system('ls')
python_hw1.py python_hw2.py README.txt
```

[*Remove ads*](/account/join/)

### 技巧 4:休息一下

当你在学习的时候，重要的是离开并吸收概念。番茄工作法被广泛使用，并且有所帮助:你工作 25 分钟，休息一会儿，然后重复这个过程。休息对于有效的学习至关重要，尤其是当你吸收大量新信息的时候。

调试时，中断尤其重要。如果你碰到了一个 bug，却不知道到底是哪里出了问题，那就休息一下。离开你的电脑，去散步，或者和朋友聊天。

在编程中，你的代码必须完全遵循语言和逻辑的规则，所以即使少了一个引号也会破坏一切。新鲜的眼睛有很大的不同。

### 技巧 5:成为一名虫子赏金猎人

说到遇到 bug，一旦你开始编写复杂的程序，你不可避免地会在代码中遇到 bug。这发生在我们每个人身上！不要让错误挫败你。相反，自豪地拥抱这些时刻，把自己想象成一个虫子赏金猎人。

调试时，重要的是要有一种方法来帮助您找到问题出在哪里。按照代码执行的顺序仔细检查代码，确保每个部分都能正常工作，这是一个很好的方法。

一旦你知道哪里出了问题，将下面一行代码插入你的脚本`import pdb; pdb.set_trace()`并运行它。这是 [Python 调试器](https://realpython.com/python-debugging-pdb/)，将把你带入交互模式。调试器也可以用`python -m pdb <my_file.py>`从命令行运行。

## 让它协作起来

一旦事情开始停滞不前，通过合作加快你的学习。这里有一些策略可以帮助你从与他人的合作中获得最大收益。

### 提示 6:让你周围都是正在学习的人

虽然编码看起来像是一项单独的活动，但实际上当你们一起工作时效果最好。当你学习用 Python 编程时，你周围的人也在学习是非常重要的。这将允许你分享你一路上学到的技巧和诀窍。

不认识也不用担心。有很多方法可以认识其他热衷于学习 Python 的人！寻找当地的活动或聚会，或者加入 [PythonistaCafe](https://www.pythonistacafe.com/) ，这是一个面向像您一样的 Python 爱好者的点对点学习社区！

### 技巧 7:教导

据说学东西最好的方法是教它。在学习 Python 的时候确实如此。有很多方法可以做到这一点:与其他 Python 爱好者一起写白板，写博客解释新学到的概念，录制视频解释你学到的东西，或者只是在电脑前自言自语。这些策略中的每一个都将巩固你的理解，并暴露你理解中的任何差距。

### 技巧 8:配对程序

[结对编程](https://en.wikipedia.org/wiki/Pair_programming)是一种涉及两个开发人员在一个工作站完成一项任务的技术。这两个开发人员在“驾驶员”和“导航员”之间转换“驱动者”编写代码，而“导航者”帮助指导问题的解决，并在编写代码时对其进行审查。经常切换，以获得双方的利益。

结对编程有很多好处:它不仅让你有机会让别人审查你的代码，还能让你看到其他人是如何思考问题的。接触多种想法和思维方式将有助于你在回到自己编码的时候解决问题。

### 技巧 9:问“好”的问题

人们总说没有烂问题这一说，但说到编程，就有可能把一个问题问烂。当你向一个对你要解决的问题知之甚少或一无所知的人寻求帮助时，最好按照这个缩写提出好的问题:

*   给出你想做的事情的背景，清楚地描述问题。
*   **O** :概述你已经尝试解决问题的事情。
*   对可能的问题给出你最好的猜测。这有助于帮助你的人不仅知道你在想什么，而且知道你自己也做了一些思考。
*   演示正在发生的事情。包括代码、一条[回溯](https://realpython.com/python-traceback/)错误消息，以及对导致错误的执行步骤的解释。这样，提供帮助的人就不必试图重现问题。

好的问题可以节省很多时间。跳过这些步骤中的任何一步都会导致反复的对话，从而引发冲突。作为初学者，你要确保你问了好的问题，这样你就可以练习交流你的思维过程，这样帮助你的人就会很乐意继续帮助你。

[*Remove ads*](/account/join/)

## 做某事

大多数(如果不是全部的话)Python 开发人员会告诉你，为了学习 Python，你必须边做边学。做练习只能带你走这么远:你通过构建学到最多。

### 技巧 10:建造一些东西，任何东西

对于初学者来说，有许多小练习可以真正帮助你对 Python 变得自信，并开发我们上面谈到的肌肉记忆。一旦你牢固掌握了基本的数据结构(字符串、列表、字典、集合)、[面向对象编程](https://realpython.com/python3-object-oriented-programming/)和编写类，是时候开始构建了！

你建造什么并不重要，重要的是你如何建造它。建筑之旅确实是教会你最多的东西。你只能从阅读真正的 Python 文章和课程中学到这么多。你的大部分学习将来自于使用 Python 来构建一些东西。你要解决的问题会教会你很多。

有很多关于初级 Python 项目的列表。以下是让你开始的一些想法:

*   猜数字游戏
*   简单的计算器应用
*   [掷骰模拟器](https://realpython.com/python-dice-roll/)
*   [比特币价格通知服务](https://realpython.com/python-bitcoin-ifttt/)

如果你发现很难想出 Python 实践项目，请观看视频。它列出了一个策略，当你感到停滞不前时，你可以用它来产生数以千计的项目想法。

### 提示#11:为开源做贡献

在开源模式中，软件源代码是公开的，任何人都可以合作。有许多 Python 库是开源项目并接受贡献。此外，许多公司发布开源项目。这意味着你可以使用这些公司的工程师编写和生产的代码。

[参与开源 Python 项目](https://dbader.org/blog/python-open-source-contributing)是创造极有价值的学习体验的好方法。假设您决定提交一个 bugfix 请求:您提交一个[“拉请求”](https://help.github.com/articles/about-pull-requests/)来将您的补丁添加到代码中。

接下来，项目经理将审查你的工作，提供意见和建议。这将使您能够学习 Python 编程的最佳实践，并练习与其他开发人员交流。

要获得更多帮助您进入开源世界的技巧和策略，请查看下面嵌入的视频:

[https://www.youtube.com/embed/jTTf4oLkvaM?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com](https://www.youtube.com/embed/jTTf4oLkvaM?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com)

## 向前去学习吧！

现在您已经有了这些学习策略，您已经准备好开始您的 Python 之旅了！在这里找到真正的 Python 初学者学习路线图！我们还提供初级水平的 [Python 课程](https://realpython.com/products/real-python-course/)，使用有趣的例子帮助你学习编程和 web 开发。

编码快乐！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**学习 Python 的 11 个初学者小技巧**](/courses/python-beginner-tips/)*****