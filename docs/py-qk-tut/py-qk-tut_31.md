## Python 标准库的学习准备

[`www.cnblogs.com/vamei/archive/2012/07/23/2605345.html`](http://www.cnblogs.com/vamei/archive/2012/07/23/2605345.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

Python 标准库是 Python 强大的动力所在，我们已经在[前文](http://www.cnblogs.com/vamei/archive/2012/07/18/2597212.html)中有所介绍。由于标准库所涉及的应用很广，所以需要学习一定的背景知识。

1\. 计算机硬件原理

这一部份需要了解内存，CPU，磁盘存储以及 IO 的功能和性能，了解计算机工作的流程，了解指令的概念。这些内容基础而重要。

Python 标准库的一部份是为了提高系统的性能(比如 mmap)，所以有必要了解基本的计算机各个组成部分的性能。

2\. 操作系统（特别是 Linux 系统）

 在了解操作系统时，下面是重点：

1) 操作系统的进程管理，比如什么是 UID, PID, daemon

2) 进程之间的信号通信，比如使用 kill 传递信号的方式

学习进程相关的内容，是为了方便于学习 os 包，thread 包，multiprocessing 包，signal 包

3) 文件管理，文件的几种类型。

4) 文件读写(IO)接口

5) 文件的权限以及其它的文件信息(meta data)

6) 常用系统命令以及应用，比如 ls, mv, rm, mkdir, chmod, zip, tar..., 

学习文件相关的内容，，是为了学习 os 包, shutil 包中文件管理相关的部分。学习文件接口对于文本输入输出的理解很重要，也会影响到对于 socket 包, select 包概念的理解。此外，python 中的归档(archive)和压缩(compress)功能也和操作系统中的类似。

7) Linux shell，比如说 file name matching，对于理解 glob 包等有帮助。如果你对 Linux 的正则表达(regular expression)有了解的话，python 的正则表达的学习会变得比较容易。学习 Linux 命令行中的参数传递对于理解 python 标准库中解析命令行的包也是有用的。

3\. 网络相关

Python 的一大应用是在网络方面。但 Python 和标准库的知识并不够用，往往需要补充更多的知识。一些基本的网络知识可以大大降低学习曲线的陡度。

1) TCP/IP 的基础的分层架构

这方面的内容太广博了，所以可以有选择地了解骨干知识。

2）常用的应用层协议，比如 http, 以及邮件相关的协议，特别是它们的工作过程。

3）根据需要，了解 html/css/javascript/jQuery/frame 等

如果想利用 python 建服务器，比如在 google app engine 上，这些知识是需要的。

4\. 数据结构

如果需要学习和应用标准库中的新的数据对象，需要一些数据结构的知识，比如队列，树等。如果已经了解了这些数据结构，这些部分的学习就没有任何难度了。

5\. 数据库

当使用 python 中数据库相关的包时(比如 sqlite3)，需要对数据库，特别是关系型数据库，需要有一个基本概念。

6\. 加密和文本编码相关

这一部分没有经验，不敢乱说。

最后的，也是最重要的，就是 Python 基本的对象概念和动态类型概念。这些你可以参照[快速教程](http://www.cnblogs.com/vamei/tag/Python%E6%95%99%E7%A8%8B/)，并尝试阅读更多的资料和源码，来加深对概念的理解。Python 标准库学习的基本难度其实在于这些背景知识。一个了解这些背景知识或者其它语言的库的人，应该可以在很短的时间内掌握 Python 基础库。