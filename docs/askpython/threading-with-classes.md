# Python 中的类线程——简要指南

> 原文：<https://www.askpython.com/python/oops/threading-with-classes>

本教程将向你解释如何利用类在 Python 中构建一个线程。但是首先，让我们定义一个线程。

* * *

## 什么是线程？

线程是一个并行执行流。这意味着您的代码将同时进行两件事情。

一个**线程**是当一个进程被调度执行时，用于执行的最小处理单元。

* * *

## Python 中线程化的优势

*   多个线程可以在具有多个 CPU 的计算机系统上同时运行。因此，额外的应用程序可能会并发运行，从而加快进程的速度。
*   在单个和多个 CPU 的情况下，输入都是有响应的。
*   线程有局部变量。
*   当一个全局变量在一个线程中被更新时，它也会影响其他线程，这意味着全局变量内存是在所有线程中共享的。

* * *

## 开始一个新线程

现在你知道什么是线程了，让我们看看如何构建一个线程。它与 Windows 和 Linux 都兼容。

```py
thread.start_new_thread ( func, args[, kwargs] )

```

## 使用类实现线程

现在，看下面的代码来理解一个线程是如何使用一个类形成的。在这种情况下，类名是 c1。在类 c1 中，创建了两个对象 obj 和 obj1。

线程从 **Obj.start()** 开始。

```py
import threading

class c1(threading.Thread) :
    def run(self) :
        for _ in range (2) :
            print(threading.currentThread().getName())
obj= c1(name='Hello')
obj1= c1(name='Bye')
obj.start()
obj1.start()

```

代码的输出如下所示:

```py
Hello
Hello
Bye
Bye

```

* * *

## 结论

恭喜你！您刚刚学习了如何使用 Python 编程语言构建线程。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Python 中的同步——Python 中的同步线程](https://www.askpython.com/python/examples/synchronization-in-python)
2.  Python 中的守护线程——它们是什么，如何创建它们？
3.  [Python 中的多线程:一个简单的参考](https://www.askpython.com/python-modules/multithreading-in-python)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *