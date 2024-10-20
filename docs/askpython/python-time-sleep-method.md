# 使用 Python time.sleep()方法

> 原文：<https://www.askpython.com/python-modules/python-time-sleep-method>

在本教程中，我们将学习 Python time.sleep()方法。在我们之前的教程中，我们查看了[时间模块](https://www.askpython.com/python-modules/python-time-module)，这是用于执行各种计时任务的默认实用程序库。

Python time.sleep()方法用于在给定的时间内暂停当前程序/线程的执行。

当前程序/线程在这段时间内基本上什么都不做，所以在从当前状态恢复之前，它会“休眠”一段时间。

让我们来看看如何使用这个函数。

* * *

## Python time.sleep()用法

这个函数是`time`模块的一部分，因此我们使用像 time.sleep()这样的点符号来调用它。我们必须首先导入时间模块。

```py
import time

```

现在，为了暂停程序的执行，我们需要指定秒数作为参数。

```py
import time

num_seconds = 5

print('Going to sleep for', str(num_seconds), 'seconds') 
time.sleep(num_seconds)
print('Woke up after', str(num_seconds), 'seconds')

```

**输出**

```py
Going to sleep for 5 seconds
Woke up after 5 seconds

```

如果您在您的机器上尝试这样做，您的程序将在两次输出之间暂停 5 秒钟，因为它在这段时间处于休眠状态。

我们也可以将秒数指定为浮点数，这样就可以休眠`0.001`秒(1 毫秒)甚至`0.0000001`秒(1 微秒)。

这将使延迟在浮点和时钟精度范围内尽可能精确。

```py
import time

num_millis = 2

print('Going to sleep for', str(num_millis), 'milliseconds') 
time.sleep(num_millis / 1000)
print('Woke up after', str(num_millis), 'milliseconds')

```

**输出**

```py
Going to sleep for 2 milliseconds
Woke up after 2 milliseconds

```

为了测量睡眠的准确时间，我们可以使用`time.time()`方法来启动计时器。计时器的开始值和结束值之差就是我们的执行时间。

让我们在上面的程序中测试一下我们的实际睡眠时间。

```py
import time

num_millis = 2

print('Going to sleep for', str(num_millis), 'milliseconds')
# Start timer

start_time = time.time() 
time.sleep(num_millis / 1000)
# End timer
end_time = time.time()

print('Woke up after', str(end_time - start_time), 'seconds')

```

**输出**

```py
Going to sleep for 2 milliseconds
Woke up after 0.0020711421966552734 seconds

```

这里，正如你所看到的，时间并不是 2 毫秒。大概是`2.071`毫秒，比它略大。

这是由于操作系统在分配资源、进程调度等方面的一些延迟，这可能会导致轻微的延迟。

这种延迟的程度会有所不同，因为您不知道操作系统在特定时刻的确切状态。

## time.sleep 的可变时间延迟()

如果您出于某种原因想要不同的延迟量，我们可以将一个变量传递给`time.sleep()`。

```py
import time

delays = [1, 1.5, 2]

for delay in delays:
    print('Sleeping for', delay, 'seconds')
    time.sleep(delay)

```

**输出**

```py
Sleeping for 1 seconds
Sleeping for 1.5 seconds
Sleeping for 2 seconds

```

既然我们已经介绍了如何在程序中使用`time.sleep()`，我们也可以对线程做同样的事情。

* * *

## 在线程上使用 Python time.sleep()

这在多线程环境中是一个有用的功能，因为多个线程可能需要等待特定资源被释放。

下面的代码片段展示了我们如何使用 Python `time.sleep()`让多线程等待并打印输出。

```py
import time
from threading import Thread

class Worker(Thread):
    # Entry point after thread.start() is invoked
    def run(self):
        for i in range(4):
            print('Worker Thread', i)
            time.sleep(i + 1)

class Waiter(Thread):
    def run(self):
        for i in range(10, 15):
            print('Waiter thread', i)
            time.sleep(i - 9)

print('Starting Worker Thread....')
Worker().start()
print('Starting Waiter Thread....')
Waiter().start()
print('Main thread finished!')

```

**输出**

```py
Starting Worker Thread....
Worker Thread 0
Starting Waiter Thread....
Waiter thread 10
Main thread finished!
Worker Thread 1
Waiter thread 11
Worker Thread 2
Waiter thread 12
Worker Thread 3
Waiter thread 13
Waiter thread 14

```

这里，主线程(程序)的执行独立于两个线程的执行。因此，我们的主程序首先完成，然后是`Worker`和`Waiter`线程。

* * *

## 结论

在本文中，我们学习了以各种方式使用 Python `time.sleep()`函数。

## 参考

*   关于 time.sleep()的 JournalDev 文章
*   [Python 文档](https://docs.python.org/3/library/time.html)

* * *