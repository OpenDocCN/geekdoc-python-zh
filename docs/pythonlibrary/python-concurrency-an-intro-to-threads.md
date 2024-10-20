# Python 并发性:线程介绍

> 原文：<https://www.blog.pythonlibrary.org/2014/02/24/python-concurrency-an-intro-to-threads/>

Python 有许多不同的并发结构，比如线程、队列和多处理。线程模块曾经是实现并发的主要方式。几年前，多处理模块被添加到 Python 标准库套件中。不过，本文将主要关注线程模块。

* * *

### 入门指南

我们将从一个简单的例子开始，这个例子演示了线程是如何工作的。我们将子类化 **Thread** 类，并将其名称输出到 stdout。让我们开始编码吧！

```py

import random
import time

from threading import Thread

########################################################################
class MyThread(Thread):
    """
    A threading example
    """

    #----------------------------------------------------------------------
    def __init__(self, name):
        """Initialize the thread"""
        Thread.__init__(self)
        self.name = name
        self.start()

    #----------------------------------------------------------------------
    def run(self):
        """Run the thread"""
        amount = random.randint(3, 15)
        time.sleep(amount)
        msg = "%s has finished!" % self.name
        print(msg)

#----------------------------------------------------------------------
def create_threads():
    """
    Create a group of threads
    """
    for i in range(5):
        name = "Thread #%s" % (i+1)
        my_thread = MyThread(name=name)

if __name__ == "__main__":
    create_threads()

```

在上面的代码中，我们导入 Python 的随机模块，时间模块，并从线程模块导入线程类。接下来，我们子类化 Thread 并覆盖它的 **__init__** 方法，以接受一个我们标记为“name”的参数。要启动一个线程，你必须调用它的 **start()** 方法，所以我们在 init 结束时这样做。当你启动一个线程时，它会自动调用它的 **run** 方法。我们覆盖了它的 run 方法，让它选择一个随机的睡眠时间。这里的 **random.randint** 示例将使 Python 从 3-15 中随机选择一个数字。然后我们让线程休眠我们随机选择的秒数来模拟它实际做的事情。最后，我们打印出线程的名称，让用户知道线程已经完成。

**create_threads** 函数将创建 5 个线程，给每个线程一个唯一的名字。如果您运行这段代码，您应该会看到类似这样的内容:

```py

Thread #2 has finished!
Thread #1 has finished!
Thread #3 has finished!
Thread #4 has finished!
Thread #5 has finished!

```

输出的顺序每次都会不同。尝试运行该代码几次，以查看顺序的变化。

* * *

### 编写线程下载程序

除了作为解释线程如何工作的工具之外，前面的例子没有什么用处。所以在这个例子中，我们将创建一个可以从互联网下载文件的线程类。美国国税局有大量的 PDF 表单，供其公民用于纳税。我们将使用这个免费资源进行演示。代码如下:

```py

# Use this version for Python 2
import os
import urllib2

from threading import Thread

########################################################################
class DownloadThread(Thread):
    """
    A threading example that can download a file
    """

    #----------------------------------------------------------------------
    def __init__(self, url, name):
        """Initialize the thread"""
        Thread.__init__(self)
        self.name = name
        self.url = url

    #----------------------------------------------------------------------
    def run(self):
        """Run the thread"""
        handle = urllib2.urlopen(self.url)
        fname = os.path.basename(self.url)
        with open(fname, "wb") as f_handler:
            while True:
                chunk = handle.read(1024)
                if not chunk:
                    break
                f_handler.write(chunk)
        msg = "%s has finished downloading %s!" % (self.name,
                                                   self.url)
        print(msg)

#----------------------------------------------------------------------
def main(urls):
    """
    Run the program
    """
    for item, url in enumerate(urls):
        name = "Thread %s" % (item+1)
        thread = DownloadThread(url, name)
        thread.start()

if __name__ == "__main__":
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]
    main(urls)

```

这基本上是对第一个脚本的完全重写。在这个示例中，我们导入了 os 和 urllib2 模块以及线程模块。我们将使用 urllib2 在 thread 类中进行实际的下载。os 模块用于提取我们正在下载的文件的名称，因此我们可以使用它在我们的机器上创建一个同名的文件。在 DownloadThread 类中，我们设置了 __init__ 来接受线程的 url 和名称。在 run 方法中，我们打开 url，提取文件名，然后使用该文件名在磁盘上命名/创建文件。然后，我们使用一个 **while** 循环一次下载一千字节的文件，并将其写入磁盘。文件保存完成后，我们打印出线程的名称和下载完成的 url。

**更新:**

Python 3 的代码版本略有不同。你必须导入 **urllib** 并使用 **urllib.request.urlopen** 而不是 **urllib2.urlopen** 。下面是 Python 3 版本:

```py

# Use this version for Python 3
import os
import urllib.request

from threading import Thread

########################################################################
class DownloadThread(Thread):
    """
    A threading example that can download a file
    """

    #----------------------------------------------------------------------
    def __init__(self, url, name):
        """Initialize the thread"""
        Thread.__init__(self)
        self.name = name
        self.url = url

    #----------------------------------------------------------------------
    def run(self):
        """Run the thread"""
        handle = urllib.request.urlopen(self.url)
        fname = os.path.basename(self.url)
        with open(fname, "wb") as f_handler:
            while True:
                chunk = handle.read(1024)
                if not chunk:
                    break
                f_handler.write(chunk)
        msg = "%s has finished downloading %s!" % (self.name,
                                                   self.url)
        print(msg)

#----------------------------------------------------------------------
def main(urls):
    """
    Run the program
    """
    for item, url in enumerate(urls):
        name = "Thread %s" % (item+1)
        thread = DownloadThread(url, name)
        thread.start()

if __name__ == "__main__":
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]
    main(urls)

```

* * *

### 包扎

现在你知道了如何在理论和实践中使用线程。当你创建一个用户界面并希望保持界面可用时，线程尤其有用。如果没有线程，当您下载大文件或对数据库进行大查询时，用户界面会变得没有响应，并且看起来会挂起。为了防止这种情况发生，你需要在线程中执行长时间运行的过程，然后在完成后将信息反馈给你的接口。

* * *

### 相关阅读

*   Python 文档- [第 16.2 节:线程](http://docs.python.org/2/library/threading.html)
*   Python 并发:[一个队列的例子](https://www.blog.pythonlibrary.org/2012/08/01/python-concurrency-an-example-of-a-queue/)
*   Python 并发:[从队列移植到多处理](https://www.blog.pythonlibrary.org/2012/08/03/python-concurrency-porting-from-a-queue-to-multiprocessing/)