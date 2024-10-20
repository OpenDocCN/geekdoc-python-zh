# 对 sh 包的简单介绍

> 原文：<https://www.blog.pythonlibrary.org/2016/01/20/a-brief-intro-to-the-sh-package/>

前几天偶然看到一个有趣的项目叫做 [sh](https://pypi.python.org/pypi/sh) ，我相信指的就是 shell(或者终端)。它曾经是 [pbs 项目](https://pypi.python.org/pypi/pbs)，但是他们为了我还没有弄清楚的原因而重新命名了它。无论如何，sh 包是一个包装子进程的包装器，允许开发人员更简单地调用可执行文件。基本上，它会将你的系统程序映射到 Python 函数。注意，sh 只支持 linux 和 mac，而 pbs 也支持 Windows。

让我们看几个例子。

```py

>>> from sh import ls
>>> ls

>>> ls('/home')
user_one  user_two

```

在上面的代码中，我简单地从 sh 导入了 ls“命令”。然后我在我的 home 文件夹上调用它，它会吐出那里存在哪些用户文件夹。让我们试试运行 Linux 的 **which** 命令。

```py

>>> import sh
>>> sh.which('python')
'/usr/bin/python'

```

这次我们只是导入 sh 并使用 **sh.which** 调用 which 命令。在这种情况下，我们传入我们想知道位置的程序名。换句话说，它的工作方式与常规的 which 程序相同。

* * *

### 多个参数

如果需要向命令传递多个参数，该怎么办？让我们来看看 **ping** 命令来了解一下吧！

```py

>>> sh.ping('-c', '4', 'www.google.com')
PING www.google.com (74.125.225.17) 56(84) bytes of data.
64 bytes from ord08s12-in-f17.1e100.net (74.125.225.17): icmp_seq=1 ttl=55 time=16.3 ms
64 bytes from ord08s12-in-f17.1e100.net (74.125.225.17): icmp_seq=2 ttl=55 time=15.1 ms
64 bytes from ord08s12-in-f17.1e100.net (74.125.225.17): icmp_seq=3 ttl=55 time=21.3 ms
64 bytes from ord08s12-in-f17.1e100.net (74.125.225.17): icmp_seq=4 ttl=55 time=23.8 ms

--- www.google.com ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3002ms
rtt min/avg/max/mdev = 15.121/19.178/23.869/3.581 ms

```

在这里，我们调用 ping 并告诉它我们只希望计数为 4。如果我们不这样做，它基本上会一直运行，直到我们告诉它停止，这就把 Python 挂在了我的机器上。

实际上，您也可以使用关键字参数作为传递给被调用程序的参数。这里有一个等价的例子:

```py

>>> sh.ping('www.google.com', c='4')

```

这有点违背直觉，因为关键字参数在 URL 后面，而在前面的例子中正好相反。然而，这更像是一个 Python 构造，是 sh 包有意为之的。在 Python 中，你毕竟不能让一个关键字参数后跟一个常规参数。

* * *

### 包扎

当我偶然发现这个项目时，我认为这是一个非常好的想法。仅仅使用 Python 的子流程模块真的很难吗？不尽然，但这种方式更有趣，有些人甚至会称 sh 更“Pythonic 化”。不管怎样，我认为值得你花时间去看看。还有一些其他的功能没有在这里介绍，比如“烘焙”,所以如果你想了解更多关于它的其他特性，查看这个项目的文档是个好主意。

* * *

### 附加阅读

*   sh [PyPI 页面](https://pypi.python.org/pypi/sh)
*   sh [github 页面](https://amoffat.github.io/sh/)
*   Python 初学者- [如何在 Python 中使用 sh](http://www.pythonforbeginners.com/modules-in-python/how-to-use-sh-in-python)