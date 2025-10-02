### Python 标准库 09 当前进程信息 (部分 os 包)

#### [www.cnblogs.com](http://www.cnblogs.com/vamei/archive/2012/10/12/2721016.html)

 作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

我们在[Linux 的概念与体系](http://www.cnblogs.com/vamei/archive/2012/10/10/2718229.html)，多次提及进程的重要性。Python 的 os 包中有查询和修改进程信息的函数。学习 Python 的这些工具也有助于理解 Linux 体系。

### 进程信息 

os 包中相关函数如下：

uname() 返回操作系统相关信息。类似于 Linux 上的 uname 命令。

umask() 设置该进程创建文件时的权限 mask。类似于 Linux 上的 umask 命令，见[Linux 文件管理背景知识](http://www.cnblogs.com/vamei/archive/2012/09/09/2676792.html)

get*() 查询 (*由以下代替)

    uid, euid, resuid, gid, egid, resgid ：权限相关，其中 resuid 主要用来返回 saved UID。相关介绍见[Linux 用户与“最小权限”原则](http://www.cnblogs.com/vamei/archive/2012/10/07/2713593.html)

    pid, pgid, ppid, sid                 ：进程相关。相关介绍见[Linux 进程关系](http://www.cnblogs.com/vamei/archive/2012/10/07/2713023.html)

put*() 设置 (*由以下代替)

    euid, egid： 用于更改 euid，egid。

    uid, gid  ： 改变进程的 uid, gid。只有 super user 才有权改变进程 uid 和 gid (意味着要以$sudo python 的方式运行 Python)。

    pgid, sid ： 改变进程所在的进程组(process group)和会话(session)。

getenviron()：获得进程的环境变量

setenviron()：更改进程的环境变量

例 1，进程的 real UID 和 real GID

```py
import os print(os.getuid()) print(os.getgid())

```

将上面的程序保存为 py_id.py 文件，分别用$python py_id.py 和$sudo python py_id.py 看一下运行结果

### saved UID 和 saved GID

我们希望 saved UID 和 saved GID 如我们在[Linux 用户与“最小权限”原则](http://www.cnblogs.com/vamei/archive/2012/10/07/2713593.html)中描述的那样工作，但这很难。原因在于，当我们写一个 Python 脚本后，我们实际运行的是 python 这个解释器，而不是 Python 脚本文件。对比 C，C 语言直接运行由 C 语言编译成的执行文件。我们必须更改 python 解释器本身的权限来运用 saved UID 机制，然而这么做又是异常危险的。

比如说，我们的 python 执行文件为/usr/bin/python (你可以通过$which python 获知)

我们先看一下

$ls -l /usr/bin/python

的结果:

-rwxr-xr-x root root

我们修改权限以设置 set UID 和 set GID 位 (参考[Linux 用户与“最小权限”原则](http://www.cnblogs.com/vamei/archive/2012/10/07/2713593.html)) 

$sudo chmod 6755 /usr/bin/python

/usr/bin/python 的权限成为:

-rwsr-sr-x root root

随后，我们运行文件下面 test.py 文件，这个文件可以是由普通用户 vamei 所有:

```py
import os print(os.getresuid())

```

我们得到结果:

(1000, 0, 0)

上面分别是 UID，EUID，saved UID。我们只用执行一个由普通用户拥有的 python 脚本，就可以得到 super user 的权限！所以，这样做是极度危险的，我们相当于交出了系统的保护系统。想像一下 Python 强大的功能，别人现在可以用这些强大的功能作为攻击你的武器了！使用下面命令来恢复到从前:

$sudo chmod 0755 /usr/bin/python

关于脚本文件的 saved UID/GID，更加详细的讨论见

[`www.faqs.org/faqs/unix-faq/faq/part4/section-7.html`](http://www.faqs.org/faqs/unix-faq/faq/part4/section-7.html)

### 总结

get*, set*

umask(), uname()