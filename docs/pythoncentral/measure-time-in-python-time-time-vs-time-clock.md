# 在 Python 中测量时间–Time . Time()与 time.clock()

> 原文：<https://www.pythoncentral.io/measure-time-in-python-time-time-vs-time-clock/>

在我们深入 Python 中测量时间的区别之前，一个先决条件是理解计算世界中各种类型的时间。第一种类型的时间称为 CPU 或执行时间，它衡量 CPU 在执行一个程序上花费的时间。第二种类型的时间被称为挂钟时间，它衡量的是在计算机中执行一个程序的总时间。挂钟时间也称为运行时间。与 CPU 时间相比，挂钟时间通常更长，因为执行被测程序的 CPU 可能同时也在执行其他程序的指令。

另一个重要的概念是所谓的系统时间，它是由系统时钟测量的。系统时间代表计算机系统对时间流逝的概念。应该记住，操作系统可以修改系统时钟，从而修改系统时间。

Python 的`time`模块提供了各种与时间相关的函数。由于大多数时候函数都用相同的名称调用特定于平台的 C 库函数，所以这些函数的语义是平台相关的。

## **time . time vs . time . clock**

时间测量的两个有用功能是`time.time`和`time.clock`。`time.time`以秒为单位返回自纪元以来的时间，即时间开始的点。对于任何操作系统，您都可以运行 time.gmtime(0)来找出给定系统上的纪元。对于 Unix，纪元是 1970 年 1 月 1 日。对于 Windows，纪元是 1601 年 1 月 1 日。`time.time`通常用于在 Windows 上对程序进行基准测试。虽然`time.time`在 Unix 和 Windows 上表现相同，但是`time.clock`有不同的含义。在 Unix 上，`time.clock`以秒为单位返回当前处理器时间，即目前为止执行当前线程所花费的 CPU 时间。在 Windows 上，它根据 Win32 函数`QueryPerformanceCounter`返回从第一次调用该函数以来经过的挂钟时间，以秒表示。`time.time`和`time.clock`的另一个不同之处在于，如果系统时钟在两次调用之间被调慢，那么`time.time`可能会返回一个低于前一次调用的值，而`time.clock`总是返回非递减值。

下面是一个在 Unix 机器上运行`time.time`和`time.clock`的例子:

#在基于 Unix 的 OS 上
【python】
>>>导入时间
> > >打印(time.time()、time . clock())
1359147652.31 0.021184
>>>time . sleep(1)
>>>打印(time.time()、time . clock())
135914752.31

`time.time()`显示挂钟时间过去了大约一秒，而`time.clock()`显示花费在当前进程上的 CPU 时间少于 1 微秒。`time.clock()`比`time.time()`精度高很多。

在 Windows 下运行相同的程序会得到完全不同的结果:

**在 Windows 上**

```py

>>> import time

>>> print(time.time(), time.clock())

1359147763.02 4.95873078841e-06

>>> time.sleep(1)

>>> print(time.time(), time.clock())

1359147764.04 1.01088769662

```

`time.time()`和`time.clock()`都显示挂钟时间过去了大约一秒。与 Unix 不同，`time.clock()`不返回 CPU 时间，而是返回比`time.time()`精度更高的挂钟时间。

给定`time.time()`和`time.clock()`的平台相关行为，我们应该使用哪一个来衡量程序的“精确”性能？嗯，看情况。如果程序预期运行在一个几乎为程序投入了足够多资源的系统中，例如，一个运行基于 Python 的 web 应用程序的专用 web 服务器，那么使用`time.clock()`测量程序是有意义的，因为 web 应用程序可能是运行在服务器上的主要程序。如果期望程序在同时运行许多其他程序的系统中运行，那么使用`time.time()`测量程序是有意义的。大多数情况下，我们应该使用基于挂钟的计时器来测量程序的性能，因为它经常反映生产环境。

### **time it 模块**

Python 的`timeit`模块提供了一种简单的计时方式，而不是处理`time.time()`和`time.clock()`在不同平台上的不同行为，这往往容易出错。除了从代码中直接调用它，您还可以从命令行调用它。

例如:

**在基于 Unix 的操作系统上**

```py

% python -m timeit -n 10000 '[v for v in range(10000)]'

10000 loops, best of 3: 365 usec per loop

% python -m timeit -n 10000 'map(lambda x: x^2, range(1000))'

10000 loops, best of 3: 145 usec per loop

```

#在 Windows 上

```py

C:\Python27>python.exe -m timeit -n 10000 "[v for v in range(10000)]"

10000 loops, best of 3: 299 usec per loop

C:\Python27>python.exe -m timeit -n 10000 "map(lambda x: x^2, range(1000))"

10000 loops, best of 3: 109 usec per loop

```

闲置中

```py

>>> import timeit

>>> total_time = timeit.timeit('[v for v in range(10000)]', number=10000)

>>> print(total_time)

3.60528302192688 # total wall-clock time to execute the statement 10000 times

>>> print(total_time / 10000)

0.00036052830219268796 # average time per loop

>>> total_time = timeit.timeit('[v for v in range(10000)]', number=10000)

>>> print(total_time)

3.786295175552368 # total wall-lock time to execute the statement 10000 times

>>> print(total_time / 10000)

0.0003786295175552368 # average time per loop

```

`timeit`用的是哪个定时器？根据`timeit`的源代码，它使用了最好的定时器:

```py

import sys
如果 sys.platform == 'win32': 
 #在 Windows 上，最佳计时器是 time . clock
default _ timer = time . clock
else:
#在大多数其他平台上，最佳计时器是 time . time
default _ timer = time . time

```

`timeit`的另一个重要机制是它在执行过程中禁用垃圾收集器，如下面的代码所示:

```py

import gc
g cold = GC . isenabled()
GC . disable()
try:
timing = self . inner(it，self.timer) 
最后:
 if gcold: 
 gc.enable() 

```

如果应该启用垃圾收集来更准确地测量程序的性能，例如，当程序分配和取消分配大量对象时，那么您应该在设置期间启用它:

```py

>>> timeit.timeit("[v for v in range(10000)]", setup="gc.enable()", number=10000)

3.6051759719848633

```

除了非常特殊的情况，你应该总是使用模块`timeit`来测试一个程序。此外，值得记住的是，测量程序的性能总是依赖于上下文，因为没有程序是在具有无限计算资源的系统中执行的，并且从多个循环中测量的平均时间总是优于在一次执行中测量的时间。