# Python 并发性:队列的一个例子

> 原文：<https://www.blog.pythonlibrary.org/2012/08/01/python-concurrency-an-example-of-a-queue/>

Python 内置了很多很酷的并发工具，比如线程、队列、信号量和多处理。在本文中，我们将花一些时间学习如何使用队列。如果您直接使用队列，那么它可以用于先进先出或后进后出的类似堆栈的实现。如果你想看实际操作，请看本文末尾的 Hellman 文章。我们将混合线程并创建一个简单的文件下载器脚本来演示队列在我们需要并发的情况下是如何工作的。

### 创建下载应用程序

这段代码大致基于 Hellman 的文章和 IBM 的文章，因为它们都展示了如何以各种方式下载 URL。这个实现实际上是下载文件。在我们的例子中，我们将使用美国国税局的税务表格。让我们假设我们是一个小企业主，我们需要为我们的员工下载一堆这样的表格。下面是一些符合我们需求的代码:

```py

import os
import Queue
import threading
import urllib2

########################################################################
class Downloader(threading.Thread):
    """Threaded File Downloader"""

    #----------------------------------------------------------------------
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    #----------------------------------------------------------------------
    def run(self):
        while True:
            # gets the url from the queue
            url = self.queue.get()

            # download the file
            self.download_file(url)

            # send a signal to the queue that the job is done
            self.queue.task_done()

    #----------------------------------------------------------------------
    def download_file(self, url):
        """"""
        handle = urllib2.urlopen(url)
        fname = os.path.basename(url)
        with open(fname, "wb") as f:
            while True:
                chunk = handle.read(1024)
                if not chunk: break
                f.write(chunk)

#----------------------------------------------------------------------
def main(urls):
    """
    Run the program
    """
    queue = Queue.Queue()

    # create a thread pool and give them a queue
    for i in range(5):
        t = Downloader(queue)
        t.setDaemon(True)
        t.start()

    # give the queue some data
    for url in urls:
        queue.put(url)

    # wait for the queue to finish
    queue.join()

if __name__ == "__main__":
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]
    main(urls)

```

让我们把它分解一下。首先，我们需要看一下 **main** 函数定义，看看这一切是如何流动的。这里我们看到它接受一个 URL 列表。然后，main 函数创建一个队列实例，并将其传递给 5 个守护线程。守护进程线程和非守护进程线程的主要区别在于，你必须跟踪非守护进程线程并自己关闭它们，而对于守护进程线程，你基本上只需设置它们并忘记它们，当你的应用程序关闭时，它们也会关闭。接下来，我们用传入的 URL 加载队列(使用它的 put 方法)。最后，我们通过 join 方法告诉队列等待线程进行处理。在下载类中，我们有一行“self.queue.get()”，它会一直阻塞，直到队列有东西要返回。这意味着线程只是无所事事地等待拾取某些东西。这也意味着线程要从队列中“获取”某些东西，它必须调用队列的“get”方法。因此，当我们在队列中添加或放置项目时，线程池将拾取或“获取”项目并处理它们。这也被称为“德清”。一旦处理完队列中的所有项目，脚本就结束并退出。在我的机器上，它可以在不到一秒的时间内下载完所有 5 个文档。

### 进一步阅读

*   [Python 实用线程编程](http://www.ibm.com/developerworks/aix/library/au-threadingpython/)
*   Doug Hellman 的 PyMOTW:Queue "[一个线程安全的 FIFO 实现](http://www.doughellmann.com/PyMOTW/Queue/)
*   wxPython wiki: [长时间运行的任务](http://wiki.wxpython.org/LongRunningTasks)
*   [Python 线程同步](http://www.laurentluce.com/posts/python-threads-synchronization-locks-rlocks-semaphores-conditions-events-and-queues/):锁、RLocks、信号量、条件、事件和队列