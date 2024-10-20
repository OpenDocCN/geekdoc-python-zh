# Python 201:线程教程

> 原文：<https://www.blog.pythonlibrary.org/2016/07/28/python-201-a-tutorial-on-threads/>

在 Python 1.5.2 中首次引入了**线程**模块，作为低级**线程**模块的增强。线程模块使得处理线程变得更加容易，并且允许程序一次运行多个操作。

请注意，Python 中的线程最适合 I/O 操作，例如从互联网下载资源或读取计算机上的文件和目录。如果你需要做一些 CPU 密集型的事情，那么你会想看看 Python 的**多重处理**模块。原因是 Python 有全局解释器锁(GIL ),基本上让所有线程都在一个主线程中运行。因此，当您使用线程运行多个 CPU 密集型操作时，您可能会发现它实际上运行得更慢。所以我们将关注线程做得最好的地方:I/O 操作！

* * *

### 线程简介

线程可以让你运行一段长时间运行的代码，就像它是一个独立的程序一样。这有点像调用**子进程**，只不过你调用的是一个函数或类，而不是一个单独的程序。我总是觉得看一个具体的例子很有帮助。让我们来看看一些非常简单的东西:

```py

import threading

def doubler(number):
    """
    A function that can be used by a thread
    """
    print(threading.currentThread().getName() + '\n')
    print(number * 2)
    print()

if __name__ == '__main__':
    for i in range(5):
        my_thread = threading.Thread(target=doubler, args=(i,))
        my_thread.start()

```

这里我们导入线程模块并创建一个名为 **doubler** 的常规函数。我们的函数取一个值，然后将它加倍。它还打印出调用该函数的线程的名称，并在末尾打印一个空行。然后在最后一段代码中，我们创建了五个线程，并依次启动每个线程。你会注意到，当我们实例化一个线程时，我们将它的**目标**设置为我们的 doubler 函数，并且我们还将一个参数传递给该函数。 **args** 参数看起来有点奇怪的原因是，我们需要将一个序列传递给 doubler 函数，它只需要一个参数，所以我们需要在末尾加一个逗号来创建一个序列 1。

注意，如果您想等待线程终止，您需要调用它的 **join()** 方法。

当您运行此代码时，您应该得到以下输出:

```py

Thread-1

0

Thread-2

2

Thread-3

4

Thread-4

6

Thread-5

8

```

当然，您通常不希望将输出打印到 stdout。当你这样做的时候，这可能会变成一团乱麻。相反，你应该使用 Python 的**日志**模块。它是线程安全的，工作非常出色。让我们修改上面的例子来使用日志模块，并命名我们的线程，同时我们将它:

```py

import logging
import threading

def get_logger():
    logger = logging.getLogger("threading_example")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("threading.log")
    fmt = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger

def doubler(number, logger):
    """
    A function that can be used by a thread
    """
    logger.debug('doubler function executing')
    result = number * 2
    logger.debug('doubler function ended with: {}'.format(
        result))

if __name__ == '__main__':
    logger = get_logger()
    thread_names = ['Mike', 'George', 'Wanda', 'Dingbat', 'Nina']
    for i in range(5):
        my_thread = threading.Thread(
            target=doubler, name=thread_names[i], args=(i,logger))
        my_thread.start()

```

这段代码中最大的变化是添加了 **get_logger** 函数。这段代码将创建一个设置为调试级别的记录器。它会将日志保存到当前工作目录(即脚本运行的位置)，然后我们为记录的每一行设置格式。格式包括时间戳、线程名、日志级别和记录的消息。

在 doubler 函数中，我们将 **print** 语句改为 logging 语句。您会注意到，当我们创建线程时，我们将日志记录器传递给了 doubler 函数。我们这样做的原因是，如果您在每个线程中实例化日志记录对象，您最终会得到多个日志记录单例，并且您的日志中会有许多重复的行。

最后，我们通过创建一个名称列表来命名线程，然后使用 **name** 参数将每个线程设置为一个特定的名称。当您运行这段代码时，您应该得到一个包含以下内容的日志文件:

```py

2016-07-24 20:39:50,055 - Mike - DEBUG - doubler function executing
2016-07-24 20:39:50,055 - Mike - DEBUG - doubler function ended with: 0
2016-07-24 20:39:50,055 - George - DEBUG - doubler function executing
2016-07-24 20:39:50,056 - George - DEBUG - doubler function ended with: 2
2016-07-24 20:39:50,056 - Wanda - DEBUG - doubler function executing
2016-07-24 20:39:50,056 - Wanda - DEBUG - doubler function ended with: 4
2016-07-24 20:39:50,056 - Dingbat - DEBUG - doubler function executing
2016-07-24 20:39:50,057 - Dingbat - DEBUG - doubler function ended with: 6
2016-07-24 20:39:50,057 - Nina - DEBUG - doubler function executing
2016-07-24 20:39:50,057 - Nina - DEBUG - doubler function ended with: 8

```

该输出是不言自明的，所以让我们继续。我想在这一部分再谈一个话题。即子类化**线程。螺纹**。让我们以最后一个例子为例，我们将创建自己的自定义子类，而不是直接调用 Thread。以下是更新后的代码:

```py

import logging
import threading

class MyThread(threading.Thread):

    def __init__(self, number, logger):
        threading.Thread.__init__(self)
        self.number = number
        self.logger = logger

    def run(self):
        """
        Run the thread
        """
        logger.debug('Calling doubler')
        doubler(self.number, self.logger)

def get_logger():
    logger = logging.getLogger("threading_example")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("threading_class.log")
    fmt = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger

def doubler(number, logger):
    """
    A function that can be used by a thread
    """
    logger.debug('doubler function executing')
    result = number * 2
    logger.debug('doubler function ended with: {}'.format(
        result))

if __name__ == '__main__':
    logger = get_logger()
    thread_names = ['Mike', 'George', 'Wanda', 'Dingbat', 'Nina']
    for i in range(5):
        thread = MyThread(i, logger)
        thread.setName(thread_names[i])
        thread.start()

```

在这个例子中，我们只是子类化了**线程。螺纹**。我们像以前一样传入想要加倍的数字和日志对象。但是这一次，我们通过调用线程对象上的 **setName** 来设置线程的名称。我们仍然需要在每个线程上调用 **start** ，但是你会注意到我们不需要在子类中定义它。当你调用 **start** 时，它会通过调用 **run** 方法来运行你的线程。在我们的类中，我们调用 doubler 函数来进行处理。除了我添加了一行额外的输出之外，输出几乎是一样的。继续运行它，看看你会得到什么。

* * *

### 锁和同步

当您有多个线程时，您可能会发现自己需要考虑如何避免冲突。我这么说的意思是，你可能会有一个用例，其中不止一个线程需要同时访问相同的资源。如果你不考虑这些问题并相应地计划，那么你最终会遇到一些问题，这些问题总是发生在最坏的时候，通常发生在生产中。

解决方法是使用锁。Python 的线程模块提供了一个锁，可以由单个线程持有，也可以根本没有线程持有。如果一个线程试图获取一个已经锁定的资源的锁，那么这个线程基本上会暂停，直到锁被释放。让我们看一个相当典型的例子，一些代码没有任何锁定功能，但应该添加它:

```py

import threading

total = 0

def update_total(amount):
    """
    Updates the total by the given amount
    """
    global total
    total += amount
    print (total)

if __name__ == '__main__':
    for i in range(10):
        my_thread = threading.Thread(
            target=update_total, args=(5,))
        my_thread.start()

```

让这个例子更加有趣的是添加一个长度可变的 **time.sleep** 调用。不管怎样，这里的问题是一个线程可能会调用 **update_total** ，在它完成更新之前，另一个线程可能会调用它并试图更新它。根据操作的顺序，该值可能只被添加一次。

让我们给函数添加一个锁。有两种方法可以做到这一点。第一种方法是使用 **try/finally** ，因为我们希望确保锁总是被释放。这里有一个例子:

```py

import threading

total = 0
lock = threading.Lock()

def update_total(amount):
    """
    Updates the total by the given amount
    """
    global total
    lock.acquire()
    try:
        total += amount
    finally:
        lock.release()
    print (total)

if __name__ == '__main__':
    for i in range(10):
        my_thread = threading.Thread(
            target=update_total, args=(5,))
        my_thread.start()

```

在这里，我们只是在做其他事情之前获取锁。然后我们尝试更新总数，最后，我们释放锁并打印当前总数。实际上，我们可以使用 Python 的 **with** 语句来消除大量样板文件:

```py

import threading

total = 0
lock = threading.Lock()

def update_total(amount):
    """
    Updates the total by the given amount
    """
    global total
    with lock:
        total += amount
    print (total)

if __name__ == '__main__':
    for i in range(10):
        my_thread = threading.Thread(
            target=update_total, args=(5,))
        my_thread.start()

```

如您所见，我们不再需要 **try/finally** ，因为由 **with** 语句提供的上下文管理器为我们完成了所有这些工作。

当然，你也会发现自己在需要多线程访问多个函数的地方编写代码。当您第一次开始编写并发代码时，您可能会这样做:

```py

import threading

total = 0
lock = threading.Lock()

def do_something():
    lock.acquire()

    try:
        print('Lock acquired in the do_something function')
    finally:
        lock.release()
        print('Lock released in the do_something function')

    return "Done doing something"

def do_something_else():
    lock.acquire()

    try:
        print('Lock acquired in the do_something_else function')
    finally:
        lock.release()
        print('Lock released in the do_something_else function')

    return "Finished something else"

if __name__ == '__main__':
    result_one = do_something()
    result_two = do_something_else()

```

这在这种情况下工作正常，但是假设您有多个线程调用这两个函数。当一个线程在函数上运行时，另一个线程也可能在修改数据，结果会不正确。问题是，你可能甚至没有立即注意到结果是错误的。有什么解决办法？让我们试着弄清楚。

常见的第一个想法是在两个函数调用周围添加一个锁。让我们尝试修改上面的示例，如下所示:

```py

import threading

total = 0
lock = threading.RLock()

def do_something():

    with lock:
        print('Lock acquired in the do_something function')
    print('Lock released in the do_something function')

    return "Done doing something"

def do_something_else():
    with lock:
        print('Lock acquired in the do_something_else function')
    print('Lock released in the do_something_else function')

    return "Finished something else"

def main():
    with lock:
        result_one = do_something()
        result_two = do_something_else()

    print (result_one)
    print (result_two)

if __name__ == '__main__':
    main()

```

当你真正去运行这段代码时，你会发现它只是挂起。原因是我们刚刚告诉线程模块获取锁。所以当我们调用第一个函数时，它发现锁已经被持有并阻塞。它将继续阻塞，直到锁被释放，这是永远不会发生的。

这里真正的解决方案是使用一个**重入锁**。Python 的线程模块通过 **RLock** 函数提供了一个。只需换线**锁=穿线。Lock()** 至 **lock =线程。RLock()** 并尝试重新运行代码。您的代码现在应该可以工作了！

如果您想用实际的线程来尝试上面的代码，那么我们可以用下面的代码替换对 **main** 的调用:

```py

if __name__ == '__main__':
    for i in range(10):
        my_thread = threading.Thread(
            target=main)
        my_thread.start()

```

这将在每个线程中运行 **main** 函数，该函数将依次调用其他两个函数。您最终也会得到 10 组输出。

* * *

### 倍

线程模块有一个名为 **Timer** 的简洁类，您可以用它来表示在指定时间后应该发生的动作。他们实际上启动了自己的定制线程，并使用与普通线程相同的 **start()** 方法启动。您也可以使用**取消**方法来停止计时器。需要注意的是，你甚至可以在定时器开始前就取消它。

有一天，我遇到了一个用例，我需要与一个已经启动的子流程进行通信，但是我需要它超时。虽然有很多不同的方法来解决这个问题，但我最喜欢的解决方案是使用线程模块的 Timer 类。

对于这个例子，我们将看看如何使用 **ping** 命令。在 Linux 中，ping 命令将一直运行，直到您杀死它。所以 Timer 类在 Linux 领域变得特别方便。这里有一个例子:

```py

import subprocess

from threading import Timer

kill = lambda process: process.kill()
cmd = ['ping', 'www.google.com']
ping = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

my_timer = Timer(5, kill, [ping])

try:
    my_timer.start()
    stdout, stderr = ping.communicate()
finally:
    my_timer.cancel()

print (str(stdout))

```

这里我们只是设置了一个 lambda，我们可以用它来终止进程。然后，我们开始 ping 作业并创建一个计时器对象。您会注意到第一个参数是以秒为单位的等待时间，然后是要调用的函数和要传递给该函数的参数。在这种情况下，我们的函数是一个 lambda，我们传递给它一个参数列表，这个列表碰巧只有一个元素。如果您运行这个代码，它应该运行大约 5 秒钟，然后打印出 ping 的结果。

* * *

### 其他线程组件

线程模块还包括对其他项目的支持。例如，您可以创建一个**信号量**，它是计算机科学中最古老的同步原语之一。基本上，一个信号量管理一个内部计数器，每当你对它调用**获取**时，这个计数器就会递减，当你调用**释放**时，**就会递增**。计数器被设计成不能低于零。所以如果你碰巧在它为零的时候调用了 acquire，那么它就会阻塞。

另一个有用的工具是**事件**。它将允许您使用信号在线程之间进行通信。在下一节中，我们将看一个使用事件的例子。

最后在 Python 3.2 中，添加了 **Barrier** 对象。Barrier 是一个原语，它基本上管理一个线程池，其中的线程必须相互等待。为了通过障碍，线程需要调用 **wait()** 方法，该方法将一直阻塞，直到所有线程都进行了调用。然后它将同时释放所有线程。

* * *

### 线程通信

在一些用例中，您会希望让线程相互通信。正如我们前面提到的，您可以使用 create a**Event**来实现这个目的。但是更常见的方法是使用一个**队列**。对于我们的例子，我们将实际上使用两者！让我们看看这是什么样子:

```py

import threading

from queue import Queue

def creator(data, q):
    """
    Creates data to be consumed and waits for the consumer
    to finish processing
    """
    print('Creating data and putting it on the queue')
    for item in data:
        evt = threading.Event()
        q.put((item, evt))

        print('Waiting for data to be doubled')
        evt.wait()

def my_consumer(q):
    """
    Consumes some data and works on it

    In this case, all it does is double the input
    """
    while True:
        data, evt = q.get()
        print('data found to be processed: {}'.format(data))
        processed = data * 2
        print(processed)
        evt.set()
        q.task_done()

if __name__ == '__main__':
    q = Queue()
    data = [5, 10, 13, -1]
    thread_one = threading.Thread(target=creator, args=(data, q))
    thread_two = threading.Thread(target=my_consumer, args=(q,))
    thread_one.start()
    thread_two.start()

    q.join()

```

让我们把它分解一下。首先，我们有一个 creator(也称为 producer)函数，用于创建我们想要处理(或消费)的数据。然后，我们使用另一个函数来处理我们称为 **my_consumer** 的数据。creator 函数将使用队列的 **put** 方法将数据放入队列，消费者将不断检查更多数据，并在数据可用时进行处理。队列处理所有锁的获取和释放，所以您不必这样做。

在这个例子中，我们创建了一个想要双精度的值的列表。然后我们创建两个线程，一个用于创建者/生产者，一个用于消费者。您会注意到，我们向每个线程传递一个队列对象，这是处理锁的神奇之处。队列将让第一个线程向第二个线程提供数据。当第一个将一些数据放入队列时，它还会传入一个事件，然后等待该事件完成。然后在消费者中，数据被处理，当它完成时，它调用事件的 **set** 方法，该方法告诉第一个线程第二个线程已经完成处理，它可以继续。

代码调用的最后一行是队列对象的 **join** 方法，它告诉队列等待线程完成。当第一个线程用完了要放入队列的项目时，它就结束了。

* * *

### 包扎

我们在这里讨论了很多材料。您已经了解了以下内容:

*   线程基础
*   锁定的工作原理
*   什么是事件以及如何使用它们
*   如何使用计时器
*   使用队列/事件的线程间通信

现在你已经知道了线程是如何使用的，它们有什么用处，我希望你能在自己的代码中找到许多好的用法。

* * *

### 相关阅读

*   关于[线程模块](https://docs.python.org/3/library/threading.html)的 Python 文档
*   伊莱·本德斯基- [Python 线程:通信和停止](http://eli.thegreenplace.net/2011/12/27/python-threads-communication-and-stopping)