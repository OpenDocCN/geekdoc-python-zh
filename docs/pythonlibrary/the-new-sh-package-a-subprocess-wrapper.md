# 新的 sh 包——子流程包装器

> 原文：<https://www.blog.pythonlibrary.org/2014/08/07/the-new-sh-package-a-subprocess-wrapper/>

前几天偶然看到一篇[文章](http://blog.endpoint.com/2014/07/python-subprocess-wrapping-with-sh.html)讲的是 [pbs](https://pypi.python.org/pypi/pbs) 包的一个分叉叫 [sh](http://amoffat.github.io/sh/) 。这些包是 Python 的子流程模块的包装器。基本上，sh 允许您直接从 Python 导入和使用 shell 命令。本文将通过几个例子向您展示如何使用这个有趣的小库。

*注意，在撰写本文时，sh 包只支持 Linux 和 Mac。如果您需要 Windows 支持，那么您应该尝试 pbs 项目。*

* * *

### 入门指南

要开始使用 sh 包，您需要安装它。最简单的方法是使用 pip:

```py

pip install sh

```

现在您已经安装了它，我们准备开始学习了！

* * *

### 使用 sh

要使用 sh，您只需导入想要使用的命令。让我们在 Python 的解释器中用几个简单的例子来尝试一下:

```py

>>> from sh import ls
>>> ls("/home/mdriscoll")
Downloads   Music      Public    
Desktop    err.log     nohup.out  Pictures  Templates
Documents  keyed.kdbx  PDF	  Settings  Videos

>>> import sh
>>> sh.firefox("https://www.blog.pythonlibrary.org/")

>>> sh.ping("www.yahoo.com", c=4)
PING ds-eu-fp3.wa1.b.yahoo.com (46.228.47.115) 56(84) bytes of data.
64 bytes from ir1.fp.vip.ir2.yahoo.com (46.228.47.115): icmp_seq=1 ttl=50 time=144 ms
64 bytes from ir1.fp.vip.ir2.yahoo.com (46.228.47.115): icmp_seq=2 ttl=50 time=121 ms
64 bytes from ir1.fp.vip.ir2.yahoo.com (46.228.47.115): icmp_seq=3 ttl=50 time=119 ms
64 bytes from ir1.fp.vip.ir2.yahoo.com (46.228.47.115): icmp_seq=4 ttl=50 time=122 ms

--- ds-eu-fp3.wa1.b.yahoo.com ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3004ms
rtt min/avg/max/mdev = 119.726/126.862/144.177/10.036 ms

```

上面的例子演示了几个不同的概念。首先，您可以从 sh 包中导入命令名。在本例中，我们导入了 **ls** 命令，并对我的主文件夹运行它。接下来，我们导入 sh 模块，并使用它打开 Firefox 浏览器到特定的 web 页面。最后我们打电话给平。您会注意到，如果命令接受命令行参数，您不会将它们包含在传递给命令的字符串中。相反，您将它们变成 Python 风格的参数。在这种情况下，“-c 4”变成了“c=4”，这告诉 ping 命令只 ping 4 次。

如果您想运行一个长时间运行的流程，sh 项目支持通过 **_bg=True** 参数将其放在后台。

还可以通过一些特殊的关键字参数来重定向 stdout 和 stderr:**_ out**和 **_err** 。您需要向这些参数传递一个文件或类似文件的对象，以使它们正常工作。

* * *

### 包扎

该项目的文档有更多的信息和额外的例子，值得你花时间细读。它告诉你如何完成子命令，获得退出代码，管道，子命令等等。