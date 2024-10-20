# 使用 sched 模块在 Python 中进行调度

> 原文：<https://www.askpython.com/python-modules/sched-module>

先说 Python 中的 sched 模块。在使用 Python 中的 [`datetime`模块时，你一定碰到过一个你希望自己能在 Python 中拥有并使用的特性，那就是*调度*。](https://www.askpython.com/python-modules/python-datetime-module)

**事件调度**，一个有目的的调度任务，可用于根据时间或日期的输入来提醒或执行事件。

过去也考虑过调度，因此，我们现在有了可以使用的`sched`模块。

在本文中，我们将探索该模块的各种用途和用例，但是，为了简单起见，我们将使用`time`模块。

如果您在使用自己版本的`sched`模块时发现任何令人困惑的地方，您可能想看看我们关于使用[日期时间](https://www.askpython.com/python-modules/python-datetime-module)模块的文章，快速回顾一下日期时间对象。

## 在 Python 中安装 sched 模块

听到这个消息您可能会感到惊讶，但是，这个模块不需要安装或包管理器，因为它默认出现在 Python 的标准库中。

甚至在文档中也是如此！如果您希望访问它，以便更清楚地了解这些论点和关键词，您可以在本文底部的参考资料中找到链接。

## 如何使用 sched 模块？

使用 sched 模块的先决条件是对 datetime/time 对象有基本的了解。

如果你以前使用过`datetime`模块或者只使用过`time`模块，你可能会很高兴知道`sched`模块是 datetime 的扩展，很像另一个模块 [dateutil](https://www.askpython.com/python-modules/dateutil-module) 。

### 1.0 调度程序–导入 sched 模块

作为一个整体，`sched`模块只包含一个类，如果你想亲自验证一下，这里有[源代码](https://github.com/python/cpython/blob/3.9/Lib/sched.py)。

那么，这对我们意味着什么呢？

简而言之，只有一个类，因此，我们将只创建一个对象，它可以利用 scheduler 类的所有特性。

这个类被称为`scheduler`。我们可以马上开始，但是在开始之前，我们想先导入模块来使用它。

```py
import sched, time

```

### 1.1 如何创建调度程序对象

创建 scheduler 对象非常简单，在导入`sched`模块后，您只需要写一行代码就可以使用它。

```py
# Intializing s as the scheduler object
s = sched.scheduler(time.time, time.sleep)

```

这一行代码为您提供了使用时间的`time`模块的功能，甚至提供了延迟，支持多线程操作。

这实际上创建了一个变量`s`，它是作为 *sched* 模块的类`scheduler`的对象创建的。

### 1.2 使用调度程序对象

接下来，我们将使用所提供的功能打印出一组时间对象，以及操作本身执行的时间。

在这个小小的片段中，我们正在处理`sched`模块的核心，创建和输入事件。

就像我们使用线程一样，在`sched`模块中，我们使用 *run* 方法来运行所有计划运行的任务。

```py
# Creating a method to print time
def print_time(a="default"):
    print("From print_time", time.time(), a)

# Method to print a few times pre-decided
def print_some_times():
    print("This is the current time : ", time.time())

    # default command to print time
    s.enter(10, 1, print_time)

    # passing an argument to be printed after the time
    s.enter(10, 1, print_time, argument=('positional',))

    # passing a keyword argument to print after the time
    s.enter(10, 1, print_time, kwargs={'a': 'keyword'})

    # runs the scheduler object
    s.run()
    print("Time at which the program comes to an end: ", time.time())

# Output
# This is the current time :  1609002547.484134
# From print_time 1609002557.4923606 default
# From print_time 1609002557.4923606 positional
# From print_time 1609002557.4923606 keyword
# Time at which the program comes to an end :  1609002557.4923606

```

需要注意的是调度程序对象使用的`run`方法。这是一个运行所有预定事件的函数，并且还将根据 *delayfunc* 参数提供的时间*等待*。

这更深入地研究了并发和[多线程](https://www.askpython.com/python-modules/multithreading-in-python)的概念，其中有*启动*、*运行*、*等待*和*通知*的概念，如果你感兴趣的话，这本书非常有趣。

除此之外，您可能还注意到了一些参数，这些参数是为了展示 print 语句之间的区别而添加的。

### 1.3 附加功能

确实存在一些我们在这个例子中没有必要研究的函数，包括**取消**、**清空**和**队列**函数。

*   **cancel** 功能用于从队列中删除特定的事件。
*   **empty** 函数用于返回关于队列状态的布尔响应，以及队列是否为空。
*   **队列**功能按照事件运行的顺序为我们提供了一个可用/即将到来的事件列表。每个事件都有一个由事件细节组成的名为元组的[。](https://www.askpython.com/python/python-namedtuple)

## 结论

正如您所看到的，Python 提供的这个标准模块的门道是相当明显的，并且有可能帮助您开发更多好的特性，作为您代码的补充或框架！

如果您希望将来使用这个模块，那么在编写代码时，请不要犹豫打开这篇文章作为参考。

浏览文档可能是一项令人望而生畏的任务，这就是为什么我们试图用用户友好的文章来帮助你浏览它们。

查看我们关于我们已经介绍过的不同模块的其他文章， [datetime](https://www.askpython.com/python-modules/python-datetime-module) ， [dateutil](https://www.askpython.com/python-modules/dateutil-module) ， [psutil](https://www.askpython.com/python-modules/psutil-module) ，以及我们一直最喜欢的数据科学工具 [pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) ！

## 参考

*   [官方计划文档](https://docs.python.org/3/library/sched.html)
*   [Python 中的线程](https://docs.python.org/3/library/threading.html)
*   [Python 和时间](https://www.askpython.com/python-modules/python-time-module)