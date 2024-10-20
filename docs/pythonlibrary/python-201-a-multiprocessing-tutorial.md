# Python 201:多重处理教程

> 原文：<https://www.blog.pythonlibrary.org/2016/08/02/python-201-a-multiprocessing-tutorial/>

多处理模块是在 2.6 版本中添加到 Python 中的。它最初是由杰西·诺勒和理查德·奥德克尔克在 PEP 371 中定义的。多处理模块允许您以与使用线程模块生成线程几乎相同的方式生成进程。这里的想法是，因为您现在正在生成进程，所以您可以避免全局解释器锁(GIL ),并充分利用一台机器上的多个处理器。

多处理包还包括一些根本不在线程模块中的 API。例如，有一个简单的 Pool 类，您可以使用它来并行执行一个跨多个输入的函数。我们将在后面的部分中讨论 Pool。我们将从多处理模块的**进程**类开始。

* * *

### 多重处理入门

**进程**类与线程模块的线程类非常相似。让我们尝试创建一系列调用同一个函数的流程，看看它是如何工作的:

```py

import os

from multiprocessing import Process

def doubler(number):
    """
    A doubling function that can be used by a process
    """
    result = number * 2
    proc = os.getpid()
    print('{0} doubled to {1} by process id: {2}'.format(
        number, result, proc))

if __name__ == '__main__':
    numbers = [5, 10, 15, 20, 25]
    procs = []

    for index, number in enumerate(numbers):
        proc = Process(target=doubler, args=(number,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

```

对于这个例子，我们导入 Process 并创建一个 **doubler** 函数。在函数内部，我们将传入的数字加倍。我们还使用 Python 的 **os** 模块来获取当前进程的 ID(或 pid)。这将告诉我们哪个进程正在调用该函数。然后在底部的代码块中，我们创建一系列进程并启动它们。最后一个循环只是在每个进程上调用 **join()** 方法，告诉 Python 等待进程终止。如果需要停止一个进程，可以调用它的 **terminate()** 方法。

运行此代码时，您应该会看到类似于以下内容的输出:

```py

5 doubled to 10 by process id: 10468
10 doubled to 20 by process id: 10469
15 doubled to 30 by process id: 10470
20 doubled to 40 by process id: 10471
25 doubled to 50 by process id: 10472

```

不过，有时为您的过程取一个更容易理解的名字会更好。幸运的是，Process 类确实允许您访问相同的进程。让我们来看看:

```py

import os

from multiprocessing import Process, current_process

def doubler(number):
    """
    A doubling function that can be used by a process
    """
    result = number * 2
    proc_name = current_process().name
    print('{0} doubled to {1} by: {2}'.format(
        number, result, proc_name))

if __name__ == '__main__':
    numbers = [5, 10, 15, 20, 25]
    procs = []
    proc = Process(target=doubler, args=(5,))

    for index, number in enumerate(numbers):
        proc = Process(target=doubler, args=(number,))
        procs.append(proc)
        proc.start()

    proc = Process(target=doubler, name='Test', args=(2,))
    proc.start()
    procs.append(proc)

    for proc in procs:
        proc.join()

```

这一次，我们导入了一些额外的东西: **current_process。**current _ process 和线程模块的 **current_thread** 基本是一回事。我们用它来获取调用我们函数的线程的名字。您会注意到，对于前五个过程，我们没有设置名称。然后对于第六个，我们将流程名设置为“Test”。让我们看看我们得到了什么输出:

```py

5 doubled to 10 by: Process-2
10 doubled to 20 by: Process-3
15 doubled to 30 by: Process-4
20 doubled to 40 by: Process-5
25 doubled to 50 by: Process-6
2 doubled to 4 by: Test

```

输出表明，默认情况下，多处理模块为每个进程分配一个数字作为其名称的一部分。当然，当我们指定一个名字时，一个数字不会被加进去。

* * *

### 锁

多处理模块支持锁的方式与线程模块非常相似。你所需要做的就是导入**锁**，获取它，做一些事情，然后释放它。让我们来看看:

```py

from multiprocessing import Process, Lock

def printer(item, lock):
    """
    Prints out the item that was passed in
    """
    lock.acquire()
    try:
        print(item)
    finally:
        lock.release()

if __name__ == '__main__':
    lock = Lock()
    items = ['tango', 'foxtrot', 10]
    for item in items:
        p = Process(target=printer, args=(item, lock))
        p.start()

```

在这里，我们创建了一个简单的打印函数，打印您传递给它的任何内容。为了防止线程相互干扰，我们使用了一个锁对象。这段代码将遍历我们的三项列表，并为每一项创建一个流程。每个进程都将调用我们的函数，并向它传递 iterable 中的一项。因为我们使用了锁，所以队列中的下一个进程将等待锁被释放，然后才能继续。

* * *

### 记录

记录进程与记录线程略有不同。这是因为 Python 的日志包不使用进程共享锁，所以可能会导致来自不同进程的消息混淆。让我们尝试将基本日志添加到前面的示例中。代码如下:

```py

import logging
import multiprocessing

from multiprocessing import Process, Lock

def printer(item, lock):
    """
    Prints out the item that was passed in
    """
    lock.acquire()
    try:
        print(item)
    finally:
        lock.release()

if __name__ == '__main__':
    lock = Lock()
    items = ['tango', 'foxtrot', 10]
    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    for item in items:
        p = Process(target=printer, args=(item, lock))
        p.start()

```

记录日志的最简单方法是将其全部发送到 stderr。我们可以通过调用 **log_to_stderr()** 函数来实现。然后我们调用 **get_logger** 函数来访问一个日志记录器，并将其日志记录级别设置为 INFO。代码的其余部分是相同的。我要注意，我在这里没有调用 **join()** 方法。相反，父线程(即您的脚本)在退出时会隐式调用 **join()** 。

当您这样做时，您应该得到如下输出:

```py

[INFO/Process-1] child process calling self.run()
tango
[INFO/Process-1] process shutting down
[INFO/Process-1] process exiting with exitcode 0
[INFO/Process-2] child process calling self.run()
[INFO/MainProcess] process shutting down
foxtrot
[INFO/Process-2] process shutting down
[INFO/Process-3] child process calling self.run()
[INFO/Process-2] process exiting with exitcode 0
10
[INFO/MainProcess] calling join() for process Process-3
[INFO/Process-3] process shutting down
[INFO/Process-3] process exiting with exitcode 0
[INFO/MainProcess] calling join() for process Process-2

```

现在，如果您想将日志保存到磁盘，那么就有点棘手了。你可以在 Python 的[日志食谱](https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes)中读到这个话题。

* * *

### 台球课

Pool 类用于表示一个工作进程池。它有允许你卸载任务到工作进程的方法。让我们看一个非常简单的例子:

```py

from multiprocessing import Pool

def doubler(number):
    return number * 2

if __name__ == '__main__':
    numbers = [5, 10, 20]
    pool = Pool(processes=3)
    print(pool.map(doubler, numbers))

```

基本上，这里发生的事情是，我们创建一个 Pool 实例，并告诉它创建三个工作进程。然后我们使用 **map** 方法将一个函数和一个 iterable 映射到每个流程。最后我们打印结果，在这个例子中实际上是一个列表:**【10，20，40】**。

您还可以通过使用 **apply_async** 方法在池中获得流程的结果:

```py

from multiprocessing import Pool

def doubler(number):
    return number * 2

if __name__ == '__main__':
    pool = Pool(processes=3)
    result = pool.apply_async(doubler, (25,))
    print(result.get(timeout=1))

```

这让我们可以询问过程的结果。这就是 **get** 函数的意义所在。它试图得到我们的结果。您会注意到，我们还设置了一个超时，以防我们调用的函数发生问题。我们毕竟不希望它无限期地阻塞。

* * *

### 过程通信

谈到进程间的通信，多处理模块有两种主要方法:队列和管道。队列实现实际上是线程和进程安全的。让我们来看一个相当简单的例子，它基于我的一篇线程文章中的队列代码:

```py

from multiprocessing import Process, Queue

sentinel = -1

def creator(data, q):
    """
    Creates data to be consumed and waits for the consumer
    to finish processing
    """
    print('Creating data and putting it on the queue')
    for item in data:

        q.put(item)

def my_consumer(q):
    """
    Consumes some data and works on it

    In this case, all it does is double the input
    """
    while True:
        data = q.get()
        print('data found to be processed: {}'.format(data))
        processed = data * 2
        print(processed)

        if data is sentinel:
            break

if __name__ == '__main__':
    q = Queue()
    data = [5, 10, 13, -1]
    process_one = Process(target=creator, args=(data, q))
    process_two = Process(target=my_consumer, args=(q,))
    process_one.start()
    process_two.start()

    q.close()
    q.join_thread()

    process_one.join()
    process_two.join()

```

这里我们只需要导入队列和流程。然后我们有两个函数，一个用来创建数据并将其添加到队列中，另一个用来消费和处理数据。向队列添加数据是通过使用队列的 **put()** 方法完成的，而从队列获取数据是通过 **get** 方法完成的。最后一段代码只是创建队列对象和几个进程，然后运行它们。您会注意到，我们在流程对象上调用 **join()** ，而不是队列本身。

* * *

### 包扎

我们这里有很多材料。您已经学习了如何使用多处理模块来定位常规函数、使用队列在进程间通信、命名线程等等。Python 文档中还有很多本文没有涉及到的内容，所以一定要深入研究。同时，您现在知道了如何使用 Python 来利用计算机的所有处理能力！

* * *

### 相关阅读

*   关于[多处理模块](https://docs.python.org/3/library/multiprocessing.html)的 Python 文档
*   本周 Python 模块:[多重处理](https://pymotw.com/2/multiprocessing/)
*   Python 并发性- [将队列移植到多处理](https://www.blog.pythonlibrary.org/2012/08/03/python-concurrency-porting-from-a-queue-to-multiprocessing/)