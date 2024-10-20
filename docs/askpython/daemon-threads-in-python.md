# Python 中的守护线程——它们是什么，如何创建它们？

> 原文：<https://www.askpython.com/python-modules/daemon-threads-in-python>

大家好！在今天的帖子中，我们将看看如何在 Python 中使用守护线程。在我们开始讨论主题之前，让我们先看看什么是守护线程！

* * *

## 守护线程

*守护线程*是一种可以在后台独立运行的线程。这些类型的线程独立于主线程执行。所以这些被称为非阻塞线程。

在 Python 中什么时候可能需要守护线程？

假设您需要一个长时间运行的任务来尝试读取日志文件。当此任务在日志中检测到错误消息时，必须提醒用户。

我们可以为这个任务分配一个守护线程，它可以一直监视我们的日志文件，而我们的主程序做它通常的工作！

关于守护线程最好的部分是，一旦主程序完成，它们将自动停止执行！

如果你需要一个短任务，一个守护线程会在它返回后停止执行。然而，由于这个性质，守护线程被广泛用于长期运行的后台任务。

现在，让我们看一个例子，展示我们如何在 Python 中使用它们！

* * *

## 在 Python 中使用守护线程——动手实现

Python 中的这些例子将使用 Python 中的[线程模块，它是标准库的一部分。](https://www.askpython.com/python-modules/multithreading-in-python)

```py
import threading

```

为了说明守护线程的强大功能，让我们首先创建两个线程，A 和 b。

我们将让**线程 A** 执行一个简短的计算，而**线程 B** 试图监控一个共享资源。

如果这个资源被设置为`True`，我们将让线程 B 提醒用户这个状态。

```py
import threading
import time

# Set the resource to False initially
shared_resource = False 
 # A lock for the shared resource
lock = threading.Lock()

def perform_computation():

    # Thread A will call this function and manipulate the resource
    print(f'Thread {threading.currentThread().name} - performing some computation....')
    shared_resource = True
    print(f'Thread {threading.currentThread().name} - set shared_resource to True!')
    print(f'Thread {threading.currentThread().name} - Finished!')
    time.sleep(1)

def monitor_resource():
    # Thread B will monitor the shared resource
    while shared_resource == False:
        time.sleep(1)
    print(f'Thread {threading.currentThread().name} - Detected shared_resource = False')
    time.sleep(1)
    print(f'Thread {threading.currentThread().name} - Finished!')

if __name__ == '__main__':
    a = threading.Thread(target=perform_computation, name='A')
    b = threading.Thread(target=monitor_resource, name='B')

    # Now start both threads
    a.start()
    b.start()

```

这里，**线程 A** 会将`shared_resource`设置为`True`，**线程 B** 会等待这个资源为真。

**输出**

```py
Thread A - performing some computation....
Thread A - set shared_resource to True!
Thread A - Finished!
Thread B - Detected shared_resource = False
Thread B - Finished!

```

请注意，这两个线程都是普通线程。现在让我们让线程 B 成为一个守护线程。让我们看看现在会发生什么。

为此，我们可以将其设置为`threading.Thread(daemon=True)`方法中的一个参数。

```py
import threading
import time

shared_resource = False # Set the resource to False initially
lock = threading.Lock() # A lock for the shared resource

def perform_computation():
    # Thread A will call this function and manipulate the resource
    print(f'Thread {threading.currentThread().name} - performing some computation....')
    shared_resource = True
    print(f'Thread {threading.currentThread().name} - set shared_resource to True!')
    print(f'Thread {threading.currentThread().name} - Finished!')
    time.sleep(1)

def monitor_resource():
    # Thread B will monitor the shared resource
    while shared_resource == False:
        time.sleep(1)
    print(f'Daemon Thread {threading.currentThread().name} - Detected shared_resource = False')
    time.sleep(1)
    print(f'Daemon Thread {threading.currentThread().name} - Finished!')

if __name__ == '__main__':
    a = threading.Thread(target=perform_computation, name='A')
    b = threading.Thread(target=monitor_resource, name='B', daemon=True) # Make thread B as a daemon thread

    # Now start both threads
    a.start()
    b.start()

```

**输出**

```py
Thread A - performing some computation....
Thread A - set shared_resource to True!
Thread A - Finished!
Daemon Thread B - Detected shared_resource = False

```

这里，注意守护线程没有完成。这是因为它会自动被主线程杀死。

守护线程的非阻塞特性使得它对很多 Python 应用程序非常有用。

* * *

## 结论

在本文中，我们了解了如何在 Python 应用程序中使用守护线程

* * *

## 参考

*   Python 线程模块[文档](https://docs.python.org/3/library/threading.html)
*   关于 Python 中守护线程的 JournalDev 文章

* * *