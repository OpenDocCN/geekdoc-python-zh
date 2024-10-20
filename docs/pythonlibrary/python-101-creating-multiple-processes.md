# Python 101 -创建多个流程

> 原文：<https://www.blog.pythonlibrary.org/2020/07/15/python-101-creating-multiple-processes/>

现在大部分 CPU 厂商都在打造多核 CPU。连手机都是多核的！由于全局解释器锁，Python 线程不能使用这些内核。从 Python 2.6 开始，添加了`multiprocessing`模块，让您可以充分利用机器上的所有内核。

在本文中，您将了解以下主题:

*   使用流程的优点
*   使用过程的缺点
*   使用`multiprocessing`创建流程
*   子类化`Process`
*   创建进程池

本文并不是对多重处理的全面概述。一般来说，多处理和并发的主题更适合写在自己的书里。如果需要，您可以点击此处查看`multiprocessing`模块的文档:

*   [https://docs.python.org/2/library/multiprocessing.html](https://docs.python.org/2/library/multiprocessing.html)

现在，让我们开始吧！

### 使用流程的优点

使用流程有几个好处:

*   进程使用单独的内存空间
*   与线程相比，代码可以更加直接
*   使用多个 CPUs 内核
*   避免全局解释器锁(GIL)
*   子进程可以被杀死(不像线程)
*   `multiprocessing`模块有一个类似于`threading.Thread`的界面
*   适用于 CPU 密集型处理(加密、二分搜索法、矩阵乘法)

现在让我们来看看流程的一些缺点！

### 使用过程的缺点

使用流程也有一些缺点:

*   进程间通信更加复杂
*   内存占用大于线程

现在让我们学习如何用 Python 创建一个流程！

### 使用`multiprocessing`创建流程

`multiprocessing`模块被设计成模仿`threading.Thread`类的工作方式。

下面是一个使用`multiprocessing`模块的例子:

```py
import multiprocessing
import random
import time

def worker(name: str) -> None:
    print(f'Started worker {name}')
    worker_time = random.choice(range(1, 5))
    time.sleep(worker_time)
    print(f'{name} worker finished in {worker_time} seconds')

if __name__ == '__main__':
    processes = []
    for i in range(5):
        process = multiprocessing.Process(target=worker, 
                                          args=(f'computer_{i}',))
        processes.append(process)
        process.start()

    for proc in processes:
        proc.join()
```

第一步是导入`multiprocessing`模块。另外两个导入分别用于`random`和`time`模块。

然后你有愚蠢的`worker()`函数假装做一些工作。它接受一个`name`并且不返回任何东西。在`worker()`函数中，它将打印出工作者的`name`，然后它将使用`time.sleep()`来模拟做一些长时间运行的过程。最后，它会打印出它已经完成。

代码片段的最后一部分是创建 5 个工作进程的地方。你使用`multiprocessing.Process()`，它的工作方式和`threading.Thread()`差不多。你告诉`Process`使用什么目标函数，传递什么参数给它。主要区别在于，这次您创建了一个`list`流程。对于每个流程，您调用它的`start()`方法来启动流程。

最后，循环遍历进程列表并调用它的`join()`方法，该方法告诉 Python 等待进程终止。

运行此代码时，您将看到类似于以下内容的输出:

```py
Started worker computer_2
computer_2 worker finished in 2 seconds
Started worker computer_1
computer_1 worker finished in 3 seconds
Started worker computer_3
computer_3 worker finished in 3 seconds
Started worker computer_0
computer_0 worker finished in 4 seconds
Started worker computer_4
computer_4 worker finished in 4 seconds
```

每次运行脚本时，由于使用了`random`模块，输出会有一些不同。试试看，自己看吧！

### 子类化`Process`

来自`multiprocessing`模块的`Process`类也可以被子类化。它的工作方式与`threading.Thread`类非常相似。

让我们来看看:

```py
# worker_thread_subclass.py

import random
import multiprocessing
import time

class WorkerProcess(multiprocessing.Process):

    def __init__(self, name):
        multiprocessing.Process.__init__(self)
        self.name = name

    def run(self):
        """
        Run the thread
        """
        worker(self.name)

def worker(name: str) -> None:
    print(f'Started worker {name}')
    worker_time = random.choice(range(1, 5))
    time.sleep(worker_time)
    print(f'{name} worker finished in {worker_time} seconds')

if __name__ == '__main__':
    processes = []
    for i in range(5):
        process = WorkerProcess(name=f'computer_{i}')
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
```

这里您子类化了`multiprocess.Process()`并覆盖了它的`run()`方法。

接下来，在代码末尾的循环中创建进程，并将其添加到进程列表中。然后，为了让进程正常工作，您需要遍历进程列表，并对每个进程调用`join()`。这与上一节中的流程示例完全一样。

这个类的输出应该与上一节的输出非常相似。

### 创建进程池

如果您有许多进程要运行，有时您会希望限制可以同时运行的进程的数量。例如，假设您需要运行 20 个进程，但您的处理器只有 4 个内核。您可以使用`multiprocessing`模块创建一个进程池，将一次运行的进程数量限制为 4 个。

你可以这样做:

```py
import random
import time

from multiprocessing import Pool

def worker(name: str) -> None:
    print(f'Started worker {name}')
    worker_time = random.choice(range(1, 5))
    time.sleep(worker_time)
    print(f'{name} worker finished in {worker_time} seconds')

if __name__ == '__main__':
    process_names = [f'computer_{i}' for i in range(15)]
    pool = Pool(processes=5)
    pool.map(worker, process_names)
    pool.terminate()
```

在这个例子中，您有相同的`worker()`函数。代码的真正内容在最后，您使用列表理解创建了 15 个进程名。然后创建一个`Pool`，并将一次运行的进程总数设置为 5。要使用`pool`，您需要调用`map()`方法，并将您希望调用的函数以及要传递给该函数的参数传递给它。

Python 现在将一次运行 5 个(或更少)进程，直到所有进程都完成。您需要在最后调用池中的`terminate()`,否则您将看到如下消息:

```py
/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/multiprocessing/resource_tracker.py:216: 
UserWarning: resource_tracker: There appear to be 6 leaked semaphore objects to clean up at shutdown
```

现在你知道如何用 Python 创建一个进程`Pool`！

### 包扎

您现在已经了解了使用`multiprocessing`模块的基本知识。您已经了解了以下内容:

*   使用流程的优点
*   使用过程的缺点
*   使用`multiprocessing`创建流程
*   子类化`Process`
*   创建进程池

除了这里介绍的内容，还有很多其他的内容。您可以学习如何使用 Python 的`Queue`模块从流程中获取输出。还有进程间通信的话题。还有更多。然而，目标是学习如何创建流程，而不是学习`multiprocessing`模块的每一个细微差别。并发是一个很大的主题，需要比本文更深入的讨论。