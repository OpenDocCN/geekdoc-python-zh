# 什么是 Python 全局解释器锁(GIL)？

> 原文：<https://realpython.com/python-gil/>

Python 全局解释器锁或 [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) ，简单来说，就是一个互斥体(或锁)，只允许一个[线程](https://realpython.com/intro-to-python-threading/)持有 Python 解释器的控制权。

这意味着在任何时间点都只能有一个线程处于执行状态。执行单线程程序的开发人员看不到 GIL 的影响，但它可能会成为 CPU 受限和多线程代码的性能瓶颈。

由于 GIL 一次只允许一个线程执行，即使在具有一个以上 CPU 内核的多线程架构中，GIL 也获得了 Python“臭名昭著”特性的名声。

在本文中，您将了解 GIL 如何影响您的 Python 程序的性能，以及如何减轻它可能对您的代码造成的影响。

## GIL 为 Python 解决了什么问题？

Python 使用引用计数进行[内存管理](https://realpython.com/python-memory-management/)。这意味着用 Python 创建的对象有一个引用计数变量，它跟踪指向该对象的引用的数量。当这个计数达到零时，对象占用的内存被释放。

让我们看一个简短的代码示例来演示引用计数是如何工作的:

>>>

```py
>>> import sys
>>> a = []
>>> b = a
>>> sys.getrefcount(a)
3
```

在上面的例子中，空列表对象`[]`的引用计数是 3。列表对象被`a`、`b`引用，参数传递给`sys.getrefcount()`。

回到 GIL:

问题是这个引用计数变量需要防止两个线程同时增加或减少其值的竞争情况。如果发生这种情况，可能会导致永远不会释放的内存泄漏，或者更糟糕的是，在对该对象的引用仍然存在的情况下错误地释放内存。这可能会导致 Python 程序崩溃或其他“奇怪”的错误。

通过将*锁*添加到跨线程共享的所有数据结构中，可以保证引用计数变量的安全，这样它们就不会被不一致地修改。

但是为每个对象或对象组添加一个锁意味着将存在多个锁，这可能导致另一个问题——死锁(死锁只有在有多个锁的情况下才会发生)。另一个副作用是反复获取和释放锁会导致性能下降。

GIL 是解释器本身的一个锁，它增加了一条规则，即任何 Python 字节码的执行都需要获取解释器锁。这可以防止死锁(因为只有一个锁)，并且不会引入太多的性能开销。但是它有效地使任何受 CPU 限制的 Python 程序成为单线程的。

GIL 虽然被 Ruby 等其他语言的解释器使用，但并不是这个问题的唯一解决方案。一些语言通过使用引用计数之外的方法(如垃圾收集)来避免线程安全内存管理的 GIL 需求。

另一方面，这意味着这些语言通常必须通过添加其他性能提升功能(如 JIT 编译器)来弥补 GIL 的单线程性能优势的损失。

[*Remove ads*](/account/join/)

## 为什么选择 GIL 作为解决方案？

那么，为什么在 Python 中使用了一种看起来如此阻碍的方法呢？这是 Python 开发者的一个错误决定吗？

用 Larry Hastings 的话来说，GIL 的设计决策是让 Python 像今天这样流行的原因之一。

自从操作系统没有线程概念的时候，Python 就出现了。Python 被设计成易于使用，以使开发更快，越来越多的开发人员开始使用它。

许多扩展是为现有的 C 库编写的，这些库的特性是 Python 所需要的。为了防止不一致的变化，这些 C 扩展需要 GIL 提供的线程安全内存管理。

GIL 易于实现，并且很容易添加到 Python 中。它提高了单线程程序的性能，因为只需要管理一个锁。

非线程安全的 c 库变得更容易集成。这些 C 扩展成为 Python 容易被不同社区采用的原因之一。

如你所见，GIL 是一个实用的解决方案，解决了早期 Python 开发人员面临的一个难题。

## 对多线程 Python 程序的影响

当你看一个典型的 Python 程序或者任何计算机程序时，在性能上受 CPU 限制的程序和受 I/O 限制的程序之间是有区别的。

CPU 受限程序是那些将 CPU 推向极限的程序。这包括进行数学计算的程序，如矩阵乘法、搜索、图像处理等。

I/O 绑定程序是那些花费时间等待来自用户、文件、数据库、网络等的输入/输出的程序。I/O 绑定的程序有时不得不等待相当长的时间，直到它们从源获得它们需要的东西，这是由于在输入/输出准备好之前，源可能需要做它自己的处理，例如，用户考虑在输入提示中输入什么，或者在它自己的进程中运行数据库查询。

让我们来看一个简单的执行倒计时的 CPU 绑定程序:

```py
# single_threaded.py
import time
from threading import Thread

COUNT = 50000000

def countdown(n):
    while n>0:
        n -= 1

start = time.time()
countdown(COUNT)
end = time.time()

print('Time taken in seconds -', end - start)
```

在我的 4 核系统上运行这段代码会产生以下输出:

```py
$ python single_threaded.py
Time taken in seconds - 6.20024037361145
```

现在我对代码做了一点修改，使用两个并行线程进行相同的倒计时:

```py
# multi_threaded.py
import time
from threading import Thread

COUNT = 50000000

def countdown(n):
    while n>0:
        n -= 1

t1 = Thread(target=countdown, args=(COUNT//2,))
t2 = Thread(target=countdown, args=(COUNT//2,))

start = time.time()
t1.start()
t2.start()
t1.join()
t2.join()
end = time.time()

print('Time taken in seconds -', end - start)
```

当我再次运行时:

```py
$ python multi_threaded.py
Time taken in seconds - 6.924342632293701
```

正如你所看到的，两个版本花了几乎相同的时间来完成。在多线程版本中，GIL 阻止 CPU 绑定的线程并行执行。

GIL 对 I/O 绑定的多线程程序的性能没有太大影响，因为锁是在线程等待 I/O 时共享的

但是线程完全受限于 CPU 的程序，例如，使用线程来部分处理图像的程序，不仅会由于锁而变成单线程，而且与被编写为完全单线程的情况相比，执行时间也会增加，如以上示例所示。

这种增加是锁增加的获取和释放开销的结果。

[*Remove ads*](/account/join/)

## 为什么 GIL 还没有被移除？

Python 的开发者收到了很多关于这方面的抱怨，但是像 Python 这样流行的语言不可能带来像移除 GIL 那样重大的改变而不引起向后不兼容的问题。

GIL 显然可以被移除，开发者和研究人员在过去已经多次这样做了，但所有这些尝试都破坏了现有的 C 扩展，这些扩展严重依赖于 GIL 提供的解决方案。

当然，GIL 还能解决其他问题，但其中一些会降低单线程和多线程 I/O 绑定程序的性能，还有一些太难了。毕竟，你不希望现有的 Python 程序在新版本出来后运行得更慢，对吗？

Python 的创始人和 BDFL 吉多·范·罗苏姆在 2007 年 9 月的文章[“移除 GIL 并不容易”](https://www.artima.com/weblogs/viewpost.jsp?thread=214235)中给了社区一个答案:

> “只有当单线程程序(以及多线程但 I/O 受限的程序)*的性能不下降*时，我才会欢迎 Py3k *中的一组补丁。”*

从那以后的任何尝试都没有满足这个条件。

## 为什么在 Python 3 中没有去掉？

Python 3 确实有机会从零开始开发许多特性，在这个过程中，打破了一些现有的 C 扩展，然后需要更新和移植更改才能与 Python 3 一起工作。这就是为什么早期版本的 Python 3 被社区采用的速度较慢的原因。

但是为什么 GIL 没有和他一起被移走呢？

移除 GIL 会使 Python 3 在单线程性能上比 Python 2 慢，你可以想象这会导致什么。你不能否认 GIL 的单线程性能优势。所以结果是 Python 3 仍然有 GIL。

但是 Python 3 确实给现有的 GIL 带来了重大改进—

我们讨论了 GIL 对“只受 CPU 限制”和“只受 I/O 限制”的多线程程序的影响，但是对于一些线程受 I/O 限制而另一些线程受 CPU 限制的程序呢？

在这样的程序中，Python 的 GIL 通过不给 I/O 绑定线程从 CPU 绑定线程获取 GIL 的机会来饿死 I/O 绑定线程。

这是因为 Python 中内置的一种机制，该机制强制线程在连续使用固定间隔后释放 GIL **，如果没有其他人获得 GIL，同一线程可以继续使用。**

>>>

```py
>>> import sys
>>> # The interval is set to 100 instructions:
>>> sys.getcheckinterval()
100
```

这种机制的问题是，在大多数情况下，CPU 绑定的线程会在其他线程获得 GIL 之前重新获得它。这是由大卫·比兹利研究的，可视化可以在这里找到。

Antoine Pitrou 在 2009 年的 Python 3.2 中修复了这个问题，他添加了一个机制来查看被丢弃的其他线程的 GIL 获取请求的数量，并且不允许当前线程在其他线程有机会运行之前重新获取 GIL。

## 如何应对 Python 的 GIL

如果 GIL 给你带来了麻烦，你可以尝试以下几种方法:

**多处理 vs 多线程:**最流行的方法是使用多处理方法，用多个进程代替线程。每个 Python 进程都有自己的 Python 解释器和内存空间，因此 GIL 不会成为问题。Python 有一个 [`multiprocessing`](https://docs.python.org/2/library/multiprocessing.html) 模块，让我们可以像这样轻松地创建流程:

```py
from multiprocessing import Pool
import time

COUNT = 50000000
def countdown(n):
    while n>0:
        n -= 1

if __name__ == '__main__':
    pool = Pool(processes=2)
    start = time.time()
    r1 = pool.apply_async(countdown, [COUNT//2])
    r2 = pool.apply_async(countdown, [COUNT//2])
    pool.close()
    pool.join()
    end = time.time()
    print('Time taken in seconds -', end - start)
```

在我的系统上运行这个命令会得到以下输出:

```py
$ python multiprocess.py
Time taken in seconds - 4.060242414474487
```

与多线程版本相比，性能有了相当大的提高，对吗？

时间没有下降到我们上面看到的一半，因为流程管理有自己的开销。多个进程比多个线程更重，所以请记住，这可能会成为伸缩瓶颈。

**替代的 Python 解释器:** Python 有多个解释器实现。分别用 [C](https://realpython.com/c-for-python-programmers/) 、 [Java](https://realpython.com/oop-in-python-vs-java/) 、C#和 Python 编写的 CPython、Jython、IronPython 和 [PyPy](https://realpython.com/pypy-faster-python/) 是最受欢迎的。GIL 仅存在于原始的 Python 实现 CPython 中。如果您的程序及其库可用于其他实现之一，那么您也可以尝试它们。

**耐心等待:**尽管许多 Python 用户利用了 GIL 的单线程性能优势。多线程程序员不必担心，因为 Python 社区中一些最聪明的人正在努力将 GIL 从 CPython 中移除。一种这样的尝试被称为[直肠切除术](https://github.com/larryhastings/gilectomy)。

蟒蛇 GIL 经常被认为是一个神秘而困难的话题。但是请记住，作为一个 Pythonista，如果您正在编写 C 扩展或者如果您在程序中使用 CPU 绑定的多线程，您通常只会受到它的影响。

在这种情况下，本文将为您提供理解什么是 GIL 以及如何在自己的项目中处理它所需的一切。如果你想了解 GIL 的底层内部运作，我建议你观看大卫·比兹利的[理解 Python GIL](https://youtu.be/Obt-vMVdM8s) 演讲。**