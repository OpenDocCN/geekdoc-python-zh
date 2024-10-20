# Python 101 -创建多线程

> 原文：<https://www.blog.pythonlibrary.org/2022/04/26/python-101-creating-multiple-threads/>

并发是编程中的一个大话题。并发的概念是一次运行多段代码。Python 有几个内置于其标准库中的不同解决方案。您可以使用线程或进程。在这一章中，你将学习使用线程。

当您运行自己的代码时，您使用的是单线程。如果想在后台运行别的东西，可以使用 Python 的`threading`模块。

在本文中，您将了解以下内容:

*   Pros of Using Threads
*   Cons of Using Threads
*   Creating Threads
*   Subclassing `Thread`
*   Writing Multiple Files with Threads

**注意** : *这一章并不打算全面地介绍线程。但是您将学到足够的知识来开始在您的应用程序中使用线程。*

让我们从回顾使用线程的利与弊开始吧！

## 使用线程的优点

线程在以下方面很有用:

*   They have a small memory footprint, which means they are lightweight to use
*   Memory is shared between threads - which makes it easy to share state across threads
*   Allows you to easily make responsive user interfaces
*   Great option for I/O bound applications (such as reading and writing files, databases, etc)

现在让我们来看看缺点！

## 使用线程的缺点

线程在以下方面**不**有用:

*   Poor option for CPU bound code due to the **Global Interpreter Lock (GIL)** - see below
*   They are not interruptible / able to be killed
*   Code with threads is harder to understand and write correctly
*   Easy to create race conditions

全局解释器锁是一个保护 Python 对象的互斥锁。这意味着它防止多个线程同时执行 Python 字节码。所以当你使用线程时，它们不会在你机器上的所有 CPU 上运行。

线程非常适合运行 I/O 繁重的应用程序、图像处理和 NumPy 的数字处理，因为它们不使用 GIL 做任何事情。如果您需要跨多个 CPU 运行并发进程，请使用`multiprocessing`模块。你将在下一章学习`multiprocessing`模块。

当你有一个计算机程序，它依赖于某个事件发生的顺序来正确执行时，就会发生竞争情况。如果您的线程没有按顺序执行，那么下一个线程可能无法工作，您的应用程序可能会崩溃或以意想不到的方式运行。

## 创建线程

如果你所做的只是谈论线程，那么线程是令人困惑的。熟悉如何编写实际代码总是好的。对于这一章，您将使用下面使用`_thread`模块的`threading`模块。

`threading`模块的完整文档可在此处找到:

*   [https://docs.python.org/3/library/threading.html](https://docs.python.org/3/library/threading.html)

让我们写一个简单的例子，展示如何创建多线程。将以下代码放入名为`worker_threads.py`的文件中:

```py
# worker_threads.py

import random
import threading
import time

def worker(name: str) -> None:
    print(f'Started worker {name}')
    worker_time = random.choice(range(1, 5))
    time.sleep(worker_time)
    print(f'{name} worker finished in {worker_time} seconds')

if __name__ == '__main__':
    for i in range(5):
        thread = threading.Thread(
                target=worker,
                args=(f'computer_{i}',),
                )
        thread.start()
```

前三个导入让您可以访问`random`、`threading`和`time`模块。你可以用`random`生成伪随机数，或者从一个序列中随机选择。`threading`模块是用来创建线程的，而`time`模块可以用于许多与时间相关的事情。

在这段代码中，您使用`time`等待一段随机的时间来模拟您的“工人”代码工作。

接下来，创建一个`worker()`函数，它接收工人的`name`。当这个函数被调用时，它将打印出哪个工人已经开始工作。然后它会在 1 到 5 之间选择一个随机数。您使用这个数字来模拟员工使用`time.sleep()`工作的时间。最后，您打印出一条消息，告诉您一个工人已经完成了工作，以及这项工作用了多长时间。

最后一个代码块创建了 5 个工作线程。要创建一个线程，您需要将您的`worker()`函数作为线程要调用的`target`函数来传递。传递给`thread`的另一个参数是一个参数元组，`thread`将把它传递给目标函数。然后你调用`thread.start()`来开始运行那个线程。

当函数停止执行时，Python 会删除你的线程。

尝试运行代码，您将看到输出如下所示:

```py
Started worker computer_0
Started worker computer_1
Started worker computer_2
Started worker computer_3
Started worker computer_4
computer_0 worker finished in 1 seconds
computer_3 worker finished in 1 seconds
computer_4 worker finished in 3 seconds
computer_2 worker finished in 3 seconds
computer_1 worker finished in 4 seconds
```

你的输出会与上面的不同，因为工人`sleep()`的时间是随机的。事实上，如果您多次运行该代码，每次调用该脚本可能会有不同的结果。

`threading.Thread` is a class. Here is its full definition:

```py
threading.Thread(
    group=None, target=None, name=None,
    args=(), kwargs={},
    *,
    daemon=None,
    )
```

You could have named the threads when you created the thread rather than inside of the `worker()` function. The `args` and `kwargs` are for the target function. You can also tell Python to make the thread into a `daemon`. "Daemon threads" have no claim on the Python interpreter, which has two main consequences: 1) if only daemon threads are left, Python will shut down, and 2) when Python shuts down, daemon threads are abruptly stopped with no notification. The `group` parameter should be left alone as it was added for future extension when a `ThreadGroup` is added to the Python language.

## Subclassing `Thread`

The `Thread` class from the `threading` module can also be subclassed. This allows you more fine-grained control over your thread's creation, execution and eventual deletion. You will encounter subclassed threads often.

Let's rewrite the previous example using a subclass of `Thread`. Put the following code into a file named `worker_thread_subclass.py`.

```py
# worker_thread_subclass.py

import random
import threading
import time

class WorkerThread(threading.Thread):

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.id = id(self)

    def run(self):
        """
        Run the thread
        """
        worker(self.name, self.id)

def worker(name: str, instance_id: int) -> None:
    print(f'Started worker {name} - {instance_id}')
    worker_time = random.choice(range(1, 5))
    time.sleep(worker_time)
    print(f'{name} - {instance_id} worker finished in '
          f'{worker_time} seconds')

if __name__ == '__main__':
    for i in range(5):
        thread = WorkerThread(name=f'computer_{i}')
        thread.start()
```

In this example, you create the `WorkerThread` class. The constructor of the class, `__init__()`, accepts a single argument, the `name` to be given to thread. This is stored off in an instance attribute, `self.name`. Then you override the `run()` method.

The `run()` method is already defined in the `Thread` class. It controls how the thread will run. It will call or invoke the function that you passed into the class when you created it. When you create your own `run()` method in your subclass, it is known as **overriding** the original. This allows you to add custom behavior such as logging to your thread that isn't there if you were to use the base class's `run()` method.

You call the `worker()` function in the `run()` method of your `WorkerThread`. The `worker()` function itself has a minor change in that it now accepts the `instance_id` argument which represents the class instance's unique id. You also need to update the `print()` functions so that they print out the `instance_id`.

The other change you need to do is in the `__main__` conditional statement where you call `WorkerThread` and pass in the name rather than calling `threading.Thread()` directly as you did in the previous section.

When you call `start()` in the last line of the code snippet, it will call `run()` for you itself. The `start()` method is a method that is a part of the `threading.Thread` class and you did not override it in your code.

The output when you run this code should be similar to the original version of the code, except that now you are also including the instance id in the output. Give it a try and see for yourself!

## Writing Multiple Files with Threads

There are several common use cases for using threads. One of those use cases is writing multiple files at once. It's always nice to see how you would approach a real-world problem, so that's what you will be doing here.

To get started, you can create a file named `writing_thread.py`. Then add the following code to your file:

```py
# writing_thread.py

import random
import time
from threading import Thread

class WritingThread(Thread):

    def __init__(self, 
                 filename: str, 
                 number_of_lines: int,
                 work_time: int = 1) -> None:
        Thread.__init__(self)
        self.filename = filename
        self.number_of_lines = number_of_lines
        self.work_time = work_time

    def run(self) -> None:
        """
        Run the thread
        """
        print(f'Writing {self.number_of_lines} lines of text to '
              f'{self.filename}')
        with open(self.filename, 'w') as f:
            for line in range(self.number_of_lines):
                text = f'This is line {line+1}\n'
                f.write(text)
                time.sleep(self.work_time)
        print(f'Finished writing {self.filename}')

if __name__ == '__main__':
    files = [f'test{x}.txt' for x in range(1, 6)]
    for filename in files:
        work_time = random.choice(range(1, 3))
        number_of_lines = random.choice(range(5, 20))
        thread = WritingThread(filename, number_of_lines, work_time)
        thread.start()
```

Let's break this down a little and go over each part of the code individually:

```py
import random
import time
from threading import Thread

class WritingThread(Thread):

    def __init__(self, 
                 filename: str, 
                 number_of_lines: int,
                 work_time: int = 1) -> None:
        Thread.__init__(self)
        self.filename = filename
        self.number_of_lines = number_of_lines
        self.work_time = work_time
```

Here you created the `WritingThread` class. It accepts a `filename`, a `number_of_lines` and a `work_time`. This allows you to create a text file with a specific number of lines. The `work_time` is for sleeping between writing each line to simulate writing a large or small file.

Let's look at what goes in `run()`:

```py
def run(self) -> None:
    """
    Run the thread
    """
    print(f'Writing {self.number_of_lines} lines of text to '
          f'{self.filename}')
    with open(self.filename, 'w') as f:
        for line in range(self.number_of_lines):
            text = f'This is line {line+1}\n'
            f.write(text)
            time.sleep(self.work_time)
    print(f'Finished writing {self.filename}')
```

This code is where all the magic happens. You print out how many lines of text you will be writing to a file. Then you do the deed and create the file and add the text. During the process, you `sleep()` to add some artificial time to writing the files to disk.

The last piece of code to look at is as follows:

```py
if __name__ == '__main__':
    files = [f'test{x}.txt' for x in range(1, 6)]
    for filename in files:
        work_time = random.choice(range(1, 3))
        number_of_lines = random.choice(range(5, 20))
        thread = WritingThread(filename, number_of_lines, work_time)
        thread.start()
```

In this final code snippet, you use a list comprehension to create 5 file names. Then you loop over the files and create them. You use Python's `random` module to choose a random `work_time` amount and a random `number_of_lines` to write to the file. Finally you create the `WritingThread` and `start()` it.

When you run this code, you will see something like this get output:

```py
Writing 5 lines of text to test1.txt
Writing 18 lines of text to test2.txt
Writing 7 lines of text to test3.txt
Writing 11 lines of text to test4.txt
Writing 11 lines of text to test5.txt
Finished writing test1.txt
Finished writing test3.txt
Finished writing test4.txtFinished writing test5.txt

Finished writing test2.txt
```

You may notice some odd output like the line a couple of lines from the bottom. This happened because multiple threads happened to write to stdout at once.

You can use this code along with Python's `urllib.request` to create an application for downloading files from the Internet. Try that project out on your own.

## Wrapping Up

You have learned the basics of threading in Python. In this chapter, you learned about the following:

*   Pros of Using Threads
*   Cons of Using Threads
*   Creating Threads
*   Subclassing `Thread`
*   Writing Multiple Files with Threads

There is a lot more to threads and concurrency than what is covered here. You didn't learn about thread communication, thread pools, or locks for example. However you do know the basics of creating threads and you will be able to use them successfully. In the next chapter, you will continue to learn about concurrency in Python through discovering how `multiprocessing` works in Python!

## Related Articles

*   Python 101 - [Creating Multiple Processes](https://www.blog.pythonlibrary.org/2020/07/15/python-101-creating-multiple-processes/)

*   Python 201: [A Tutorial on Threads](https://www.blog.pythonlibrary.org/2016/07/28/python-201-a-tutorial-on-threads/)

This article is based on a chapter from **Python 101: 2nd Edition**. You can purchase Python 101 on [Amazon](https://amzn.to/2Zo1ARG) or [Leanpub](https://leanpub.com/py101).