# 最邪恶的 Python 反模式

> 原文：<https://realpython.com/the-most-diabolical-python-antipattern/>

以下是《强大的 Python[的作者 Aaron Maxwell 的客座博文。](https://powerfulpython.com)

* * *

有很多方法可以写出糟糕的代码。但是在 Python 中，有一个特别重要。

我们筋疲力尽，但仍欢欣鼓舞。在另外两个工程师尝试了*三天*来修复一个神秘的 [Unicode](https://realpython.com/python-encodings-guide/) bug，但都徒劳无功后，我终于在仅仅一天的工作后隔离了原因。十分钟后，我们找到了候选人。

悲剧在于，我们本可以跳过七天，直接进入十分钟。但是我已经超越了我自己…

这是笑点。下面这段代码是 Python 开发人员可以编写的最具自我破坏性的东西之一:

```py
try:
    do_something()
except:
    pass
```

有相当于同一事物的变体——例如说`except Exception:`或`except Exception as e:`。它们都造成了同样的巨大危害:无声无息地隐藏错误条件，否则这些错误条件可以被快速检测和调度。

为什么我认为这是当今 Python 世界中最邪恶的反模式？

*   人们这样做是因为他们预计那里会发生特定的错误。然而，捕捉`Exception`隐藏了*所有的*错误……甚至那些完全出乎意料的错误。
*   当 bug 最终被发现时——因为它已经出现在产品中，这太频繁了——您可能很少或根本不知道它在代码库中的什么地方出了问题。甚至在 try 块中发现错误都要花费你一段令人沮丧的时间。
*   一旦您意识到错误发生在那里，由于缺少关键信息，您的故障诊断就会受到很大的阻碍。什么是错误/异常类？涉及什么调用或数据结构？错误源自哪一行代码，在哪个文件中？
*   你正在丢弃堆栈跟踪——一个*字面上无价的*信息体，它可以决定在几天内还是在*分钟内*排除一个错误。是的，几分钟。稍后将详细介绍。
*   最糟糕的是，这很容易伤害在代码库工作的工程师的士气、快乐甚至自尊。当错误抬头时，故障排除人员可以花几个小时去了解根本原因。他们认为自己是个糟糕的程序员，因为要花很长时间才能搞清楚。他们不是；通过静默捕捉异常而产生的错误本质上很难识别、排查和修复。

在我近十年用 Python 编写应用程序的经历中，无论是个人还是作为团队的一员，这种模式都是开发人员生产力和应用程序可靠性的最大消耗……特别是从长期来看。如果你认为你有更坏的人选，我很想听听。

## 为什么我们要这样对待自己？

当然，没有人*故意*写代码来给你的开发伙伴增加压力，破坏应用程序的可靠性。我们这样做是因为在正常操作中，try 块中的代码有时会以某种特定的方式失败。乐观地说，[尝试然后捕捉异常](https://realpython.com/python-exceptions/)是处理这种情况的一种优秀且完全 Pythonic 化的方式。

不知不觉中，捕捉异常，然后静静地继续，似乎并不是一个可怕的想法。但是一旦你保存了这个文件，你就已经设置了你的代码来产生最糟糕的错误:

*   在开发过程中可以逃避检测，并被推出到实际生产系统中的错误。
*   在您意识到 bug 一直在发生之前，bug 可以在产品代码中存在几分钟、几小时、几天或几周。
*   难以排除的错误。
*   即使知道被抑制的异常是在哪里出现的，也很难修复的错误。

注意，我并不是说永远不要捕捉异常。有*有*好的理由来捕捉异常，然后继续——只是没有*默默地*。一个很好的例子是一个任务关键的过程，你只是不想永远下去。这里，一个聪明的模式是注入捕获异常的 try 子句，记录严重性为`logging.ERROR`或更高的[全栈跟踪](https://realpython.com/python-traceback/)，并继续。

[*Remove ads*](/account/join/)

## 解决方案

那么，如果我们不想捕捉异常，我们该怎么做呢？有两个选择。

在大多数情况下，最好的选择是捕捉一个更具体的异常。大概是这样的:

```py
try:
    do_something()
# Catch some very specific exception - KeyError, ValueError, etc.
except ValueError:
    pass
```

这是你应该尝试的第一件事。这需要对调用的代码有一点了解，所以您知道它可能会引发什么类型的错误。当你第一次写代码时，这比清理别人的代码更容易做好。

如果某个代码路径必须广泛地捕捉所有异常——例如，某个长期运行的持久化进程的顶层循环——那么每个被捕捉的异常*必须*将**全栈跟踪**连同时间戳一起写入日志或文件。如果你使用的是 [Python 的`logging`模块](https://realpython.com/python-logging/)，这非常简单——每个 logger 对象都有一个名为 exception 的方法，接受一个消息字符串。如果在 except 块中调用它，被捕获的异常将自动被完整记录，包括跟踪。

```py
import logging

def get_number():
    return int('foo')
try:
    x = get_number()
except Exception as ex:
    logging.exception('Caught an error')
```

日志将包含错误消息，后面是跨几行的格式化的[堆栈跟踪](https://realpython.com/courses/python-traceback/):

```py
ERROR:root:Caught an error
Traceback (most recent call last):
  File "example-logging-exception.py", line 8, in <module>
    x = get_number()
  File "example-logging-exception.py", line 5, in get_number
    return int('foo')
ValueError: invalid literal for int() with base 10: 'foo'
```

非常容易。

如果您的应用程序以其他方式记录日志——不使用`logging`模块——会怎么样？假设您不想重构您的应用程序来这样做，您可以只获取和格式化与异常相关联的回溯。这在 Python 3 中是最简单的:

```py
# The Python 3 version. It's a little less work.
import traceback

def log_traceback(ex):
    tb_lines = traceback.format_exception(ex.__class__, ex, ex.__traceback__)
    tb_text = ''.join(tb_lines)
    # I'll let you implement the ExceptionLogger class,
    # and the timestamping.
    exception_logger.log(tb_text)

try:
    x = get_number()
except Exception as ex:
    log_traceback(ex)
```

在 Python 2 中，您必须做稍微多一点的工作，因为异常对象没有附加它们的回溯。您可以通过调用 except 块中的`sys.exc_info()`来实现这一点:

```py
import sys
import traceback

def log_traceback(ex, ex_traceback):
    tb_lines = traceback.format_exception(ex.__class__, ex, ex_traceback)
    tb_text = ''.join(tb_lines)
    exception_logger.log(tb_text)

try:
    x = get_number()
except Exception as ex:
    # Here, I don't really care about the first two values.
    # I just want the traceback.
    _, _, ex_traceback = sys.exc_info()
    log_traceback(ex, ex_traceback)
```

事实证明，您可以定义一个既适用于 Python 2 又适用于 Python 3 的回溯记录函数:

```py
import traceback

def log_traceback(ex, ex_traceback=None):
    if ex_traceback is None:
        ex_traceback = ex.__traceback__
    tb_lines = [ line.rstrip('\n') for line in
                 traceback.format_exception(ex.__class__, ex, ex_traceback)]
    exception_logger.log(tb_lines)
```

## 你现在能做什么

“好吧，亚伦，你说服了我。我为我过去所做的一切哭泣和悲伤。我该如何赎罪？”我很高兴你问了。这里有一些你可以从今天开始的练习。

### 在你的编码指南中明确禁止它

如果您的团队进行代码审查，您可能有一个编码指南文档。如果没有，很容易开始-这可以像创建一个新的维基页面一样简单，你的第一个条目可以是这个。只需添加以下两条准则:

*   如果某个代码路径必须广泛地捕捉所有异常——例如，某个长期运行的持久化进程的顶层循环——那么每个这样捕捉到的异常*必须*将**完整堆栈跟踪**连同时间戳一起写入日志或文件。不仅仅是异常类型和消息，还有完整的堆栈跟踪。
*   对于所有其他 except 子句——实际上应该是绝大多数——捕获的异常类型必须尽可能具体。类似于 [KeyError](https://realpython.com/python-keyerror/) ，或者 ConnectionTimeout 等。

[*Remove ads*](/account/join/)

### 为除条款之外的现有天桥创建车票

以上将有助于防止新问题进入你的代码库。现有的过宽抓地力如何？简单:在你的 bug 跟踪系统中制作一个标签或问题来修复它。这是一个简单的行动步骤，大大增加了问题被解决和不被遗忘的机会。说真的，你现在就可以做这件事。

我建议您为每个存储库或应用程序创建一个标签，通过代码进行审计，找到每个捕获异常的地方。(你可以通过搜索“except:”和“except Exception”的代码库来找到它们。)对于每一次出现，要么将其转换为捕捉一个非常具体的异常类型；或者，如果不清楚应该是什么，可以修改 except 块来记录完整的堆栈跟踪。

或者，审计开发人员可以为任何特定的 try/except 块创建额外的票证。如果您觉得异常类可以做得更具体，但是对代码的这一部分不够了解，无法确定，那么这是一件好事。在这种情况下，您需要编写代码来记录完整的堆栈跟踪；创建一个单独的票证以进行进一步调查；然后分配给更清楚的人。如果你发现自己花了超过五分钟思考一个特定的 try/except 块，我建议你这样做，然后继续下一个。

### 教育你的团队成员

你定期举行工程会议吗？每周、每两周还是每月？在下一次培训中，花五分钟时间解释这个反模式、它对团队生产力的影响以及简单的解决方案。

更好的办法是，事先去找你的技术主管或工程经理，告诉他们这件事。这将更容易说服他们，因为他们至少和你一样关心团队的生产力。把这篇文章的链接发给他们。见鬼，如果你不得不，我会帮忙的-让他们和我通电话，我会说服他们。

你可以在你的社区中接触到更多的人。你会去当地的 Python 聚会吗？他们会进行闪电式会谈吗，或者你能在下一次会议上协商五到十五分钟的发言时间吗？通过传播这一崇高的事业来服务你的工程师同事。

## 为什么要记录完整的堆栈跟踪？

以上几次，我反复强调记录完整的堆栈跟踪，而不仅仅是异常对象的消息。如果这看起来像是更多的工作，那是因为它可能是:跟踪中的换行符会扰乱日志系统的格式，您可能不得不修改 traceback 模块，等等。仅仅记录消息本身还不够吗？

不，不是的。精心制作的异常消息只告诉您 except 子句在哪里——在哪个文件中，在哪一行代码中。它通常不会缩小范围，但是让我们假设最好的情况。只记录消息比什么都不记录要好，但不幸的是，它不能告诉您任何关于错误来源的信息。一般来说，它可以在一个完全不同的文件或模块中，而且通常不太容易猜到。

除此之外，团队开发的实际应用程序往往有多个代码路径可以调用异常引发块。也许只有当 Foo 类的方法 bar 被调用时才会出现错误，但函数 bar()不会被调用。只记录消息不会帮助您区分这两者。

我所知道的最好的战争故事是在一个大约 50 人的中型工程团队中工作时得到的。我是一个相对较新的人，四个多月来，我被一个 unicode bug 困扰着，这个 bug 经常会吵醒那些随叫随到的人。异常被捕获，消息被记录，但是没有记录其他信息。另外两名高级工程师各自研究了几天，然后放弃了，说他们搞不明白。

这些也是令人生畏的聪明工程师。最后，出于无奈，他们试着传给我。利用他们大量的笔记，我立即着手重现这个问题，得到一个堆栈跟踪。六个小时后，我终于明白了。一旦我有了那个该死的堆栈跟踪，你能猜到我花了多长时间来修复吗？

十分钟。没错。一旦我们有了堆栈跟踪，修复就很明显了。如果我们从一开始就记录堆栈跟踪，就可以节省工程师一周的时间。还记得上面我说的堆栈跟踪可以在几天内解决一个 bug 和几分钟内解决一个 bug 的区别吗？我没开玩笑。

(有趣的是，从中也有好的一面。正是这样的经历让我开始[写更多关于 Python](http://migrateup.com/python-newsletter/realpython0a/) 的东西，以及我们作为工程师如何更有效地使用这种语言。)**