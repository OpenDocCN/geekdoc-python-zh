# Python 中的资源模块——初学者入门

> 原文：<https://www.askpython.com/python-modules/resource-module>

**在开始阅读本文之前，需要注意的是`resource`模块是一个特定于 UNIX 的包，不能在 POSIX(即 Windows 操作系统)中工作。**

## 资源模块简介

在使用系统监控和资源的过程中，我们发现自己想知道是否有更好的方法来监控系统信息，而不是手动浏览控制面板中的所有系统日志。

沿着形成一个与这个概念相关的想法的轨迹再走一点，我们会稍微理解这可能是可能的，并且以脚本的形式完全可行。

嗯，再往这个思路看一点，这是个很棒的主意！

使用`python-crontab`、`sched`和 [dateutil](https://www.askpython.com/python-modules/dateutil-module) 模块自动执行脚本，将提供每天特定时间间隔的自动更新日志，因此，您不必在特定时间手动接收信息。

但是，在尝试将其自动化之前，我们需要一些可以首先为您提供这些信息的东西，这就是`resource`模块发挥作用的地方。

用于提供关于系统资源的基本信息，以及控制系统资源的功能，`resource`模块正是我们要找的。

所以，让我们切入正题，开始学习这个模块吧！

## 使用 Python 中的资源模块

作为 Python 标准库的一部分，`resource`模块不需要单独安装，这意味着在安装了 Python 的全新服务器或客户机上使用该模块应该会自动完成，不会有任何问题。

然而，据报道，python 的某些版本似乎确实面临资源模块的任何问题，因此，建议使用 [pip 命令](https://www.askpython.com/python-modules/python-pip)安装资源模块。

```py
pip install python-resources

```

现在我们已经完成了，我们仍然需要使用它的组件来检索所需的信息，所以，让我们开始导入吧！

### 1.0 建立生态系统

在我们开始使用由`resource`模块提供的功能之前，我们需要首先导入该模块。

```py
# Importing functions from the resource module
from resource import *
import time

```

既然我们已经导入了模块，现在我们可以开始检索系统资源的信息了。

### 1.1 底层参数用法

模块的功能主要取决于提供给返回所需信息的函数的参数。

这些参数的几个例子是，

*   `resource.RUSAGE_SELF`–调用进程消耗的资源。
*   `resource.RUSAGE_CHILDREN`–子进程消耗的资源。
*   `resource.RUSAGE_BOTH`–当前进程和子进程消耗的资源。
*   `resource.RUSAGE_THREAD`–当前线程消耗的资源。

所有这些 RUSAGE_*符号都被传递给`getrusage()`函数，以指定请求哪个进程信息。

### 1.2 演示

```py
# Function used to retrieve information regarding
## Resources consumed by the current process or it's children
### Non CPU bound task
time.sleep(3)

# Calling for the resources consumed by the current process.
print(getrusage(RUSAGE_SELF))

### CPU bound task
for i in range(10**8):
    _ = 1+1

# Calling for the resources consumed by the current process.
print(getrusage(RUSAGE_SELF))

# Output
# resource.struct_rusage(ru_utime=0.016, ru_stime=0.004, ru_maxrss=5216, ru_ixrss=0, ru_idrss=0, ru_isrss=0, ru_minflt=732, ru_majflt=1, ru_nswap=0, ru_inblock=80, ru_oublock=0, ru_msgsnd=0, ru_msgrcv=0, ru_nsignals=0, ru_nvcsw=6, ru_nivcsw=9)

# resource.struct_rusage(ru_utime=14.176, ru_stime=0.02, ru_maxrss=5140, ru_ixrss=0, ru_idrss=0, ru_isrss=0, ru_minflt=730, ru_majflt=0, ru_nswap=0, ru_inblock=0, ru_oublock=0, ru_msgsnd=0, ru_msgrcv=0, ru_nsignals=0, ru_nvcsw=1, ru_nivcsw=177)

```

我们接收到的输出是资源类对象的形式，在对象的结构中包含了所有必需的信息。

结构中的每个输出值都是浮点值，分别表示在用户和系统模式下执行所花费的时间。

如果您希望了解每个参数的更多信息，或者希望了解整个模块，您可能有兴趣访问[文档](https://docs.python.org/3/library/resource.html#resource.getrusage)页面的`getrusage()`部分。

### 1.3 向前迈进

使用这个模块应该已经为您提供了一个关于可以由`resource`模块检索的资源的概念。

扩展这个模块并在脚本中实现它将会监视系统进程并定期检查资源消耗。

如果您希望使用这样的想法，明智的做法是查看各种其他模块，如用于系统进程的`psutil`、`sys`和`os`模块。

为了安排自动检查，您可能希望使用`dateutil`、`sched`和`python-crontab`模块。

## 结论

此模块的用例主要涉及创建脚本，这些脚本往往会监控系统的功能和过程。

如果您希望使用系统进程、测试和监控，如前一节所述，您应该研究的文章是 [psutil](https://www.askpython.com/python-modules/psutil-module) 、 [sys](https://www.askpython.com/python-modules/python-sys-module) 、 [os](https://www.askpython.com/python-modules/python-os-module-10-must-know-functions) 和 [dateutil](https://www.askpython.com/python-modules/dateutil-module) 模块。

## 参考

*   [官方资源文档](https://docs.python.org/3/library/resource.html)
*   [StackOverflow:资源模块安装](https://stackoverflow.com/questions/49232580/how-to-import-resource-module)
*   [StackOverflow : Windows OS 资源模块](https://stackoverflow.com/questions/37710848/importerror-no-module-named-resource)