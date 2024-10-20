# Python 并发性:从队列移植到多处理

> 原文：<https://www.blog.pythonlibrary.org/2012/08/03/python-concurrency-porting-from-a-queue-to-multiprocessing/>

本周早些时候，我写了一篇关于 Python 队列的简单的[帖子](https://www.blog.pythonlibrary.org/2012/08/01/python-concurrency-an-example-of-a-queue/)，并演示了如何将它们与线程池一起使用，以从美国国税局的网站下载一组 pdf。今天，我决定尝试将代码“移植”到 Python 的多处理模块上。正如我的一位读者所指出的，由于 Python 中的全局解释器锁(GIL ), Python 的队列和线程只能在一个内核上运行。多处理模块(以及 Stackless 和其他几个项目)可以在多核和 GIL 上运行(如果你好奇，请参见[文档](http://docs.python.org/library/multiprocessing.html))。不管怎样，我们开始吧。

### 创建多处理下载应用程序

从队列切换到使用多处理模块非常简单。为了方便起见，我们还将使用请求库而不是 urllib 来下载文件。让我们看看代码:

```py
import multiprocessing
import os
import requests

########################################################################
class MultiProcDownloader(object):
    """
    Downloads urls with Python's multiprocessing module
    """

    #----------------------------------------------------------------------
    def __init__(self, urls):
        """ Initialize class with list of urls """
        self.urls = urls

    #----------------------------------------------------------------------
    def run(self):
        """
        Download the urls and waits for the processes to finish
        """
        jobs = []
        for url in self.urls:
            process = multiprocessing.Process(target=self.worker, args=(url,))
            jobs.append(process)
            process.start()
        for job in jobs:
            job.join()

    #----------------------------------------------------------------------
    def worker(self, url):
        """
        The target method that the process uses tp download the specified url
        """
        fname = os.path.basename(url)
        msg = "Starting download of %s" % fname
        print msg, multiprocessing.current_process().name
        r = requests.get(url)
        with open(fname, "wb") as f:
            f.write(r.content)

#----------------------------------------------------------------------
if __name__ == "__main__":
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]
    downloader = MultiProcDownloader(urls)
    downloader.run()

```

您应该将类似这样的内容输出到 stdout:

 `Starting download of f1040a.pdf Process-2
Starting download of f1040.pdf Process-1
Starting download of f1040es.pdf Process-4
Starting download of f1040sb.pdf Process-5
Starting download of f1040ez.pdf Process-3` 

让我们把这段代码分解一下。很快，你会注意到你没有像使用 threading.Thread 那样子类化多处理模块，相反，我们只是创建了一个只接受 URL 列表的泛型类。在我们实例化该类之后，我们调用它的 **run** 方法，该方法将遍历 URL 并为每个 URL 创建一个进程。它还会将每个进程添加到一个作业列表中。我们这样做的原因是因为我们想要调用每个进程的 **join** 方法，正如您所料，该方法会等待进程完成。如果您愿意，您可以向 join 方法传递一个数字，这个数字基本上是一个超时值，它将导致 join 返回进程是否实际完成。如果不这样做，那么 join 将无限期阻塞。

 `如果一个进程挂起或者你厌倦了等待它，你可以调用它的 **terminate** 方法来杀死它。根据[文档](http://docs.python.org/library/multiprocessing.html#exchanging-objects-between-processes)，在多处理模块中有一个队列类，你可以以与普通队列相似的方式使用它，因为它几乎是原始队列的克隆。如果你想更深入地挖掘这个很酷的模块的所有可能性，我建议你看看下面的一些链接。

### 额外资源

*   多重处理[文档](docs.python.org/library/multiprocessing.html)
*   Doug Hellman 的 PyMOTW [多重处理文章](http://www.doughellmann.com/PyMOTW/multiprocessing/basics.html)
*   IBM 关于多处理的[文章](http://www.ibm.com/developerworks/aix/library/au-multiprocessing/)
*   [与 Python 的多重处理共享计数器](http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/)