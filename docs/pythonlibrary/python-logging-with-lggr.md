# 使用 lggr 的 Python 日志记录

> 原文：<https://www.blog.pythonlibrary.org/2012/08/27/python-logging-with-lggr/>

有些人抱怨我的上一篇日志文章，怀疑它是否有必要，因为 docs 和 Doug Hellman 已经写了关于这个主题的文章。我有时想知道为什么我也写这些话题，但通常当我写的时候，我会有很多读者。在这种情况下，我在一周内就那一篇文章获得了近 10，000 次点击，所以我想肯定还有空间让我写这样的主题。我收到了一些关于替代日志库的评论。其中之一是彼得·唐斯的《lggr》。我们将很快地看一下这个项目，看看它是如何发展的。文档在这一点上非常肤浅，但是让我们看看我们能做些什么。

### 入门指南

如您所料，您需要从 Github 下载或签出该项目。一旦有了它，您就可以使用通常的命令来安装它:

 `python setup.py install` 

当然，你也可以在下载后使用 easy_install 或 pip 来安装它，我认为甚至有一种方法可以直接使用 pip 和 github URL。

### 编写简单的 Lggr

现在让我们深入研究一些代码，看看这个包是如何工作的。这里有一个非常简单的例子:

```py

import lggr

log = lggr.Lggr()

# add handler to write to stdout
log.add(log.ERROR, lggr.Printer())

# add handler to write to file
f = open("sample.log", "a")
log.add(log.INFO, lggr.Printer(f))

log.info("This is an informational message")

try:
    print (1/0)
except ZeroDivisionError:
    log.error("ERROR: You can't divide by zero!")
log.close()

```

这段代码将创建两个处理程序。一个将错误消息输出到 stdout，另一个将信息性消息输出到文件。然后我们给每个人写一条信息。遗憾的是，lggr 似乎不提供 Python 的标准日志模块在记录错误消息时提供的回溯信息。在日志模块中，它实际上将是一个完整的回溯。另一方面，lggr 的代码非常简洁，易于理解。lggr 包还提供了以下记录器(或者作者喜欢称之为协程):

*   StderrPrinter -写入 stderr
*   SocketWriter(主机，端口)-写入网络套接字
*   Emailer(收件人)-发送电子邮件
*   GMailer(收件人、gmail _ 用户名、Gmail _ 密码、subject= "可选")也发送电子邮件，但它是从 Gmail 发送的

如果你在公司网络上，Emailer 可能无法工作，因为那个端口可能被阻塞了。至少，当我尝试的时候，这似乎是我的问题。你可能更幸运，这取决于你的公司屏蔽了多少。无论如何，lggr 包还提供了二十几个可以记录的日志变量，比如 threadname、codecontext、stack_info、filename 等等。我个人可以看到这将是非常方便的。

### 包扎

我没有看到任何关于旋转文件处理程序的东西，但除此之外，这个项目似乎是一个非常容易使用的日志包。如果您正在寻找比 Python 的日志模块学习曲线稍低的东西，那么这个包可能就是您想要的。