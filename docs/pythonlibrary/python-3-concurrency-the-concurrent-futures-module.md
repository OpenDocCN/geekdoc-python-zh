# Python 3 并发性 concurrent.futures 模块

> 原文：<https://www.blog.pythonlibrary.org/2016/08/03/python-3-concurrency-the-concurrent-futures-module/>

Python 3.2 中增加了 **concurrent.futures** 模块。根据 Python 文档，它*为开发者提供了异步执行调用的高级接口。*基本上是 concurrent.futures 是 Python 线程和多处理模块之上的一个抽象层，简化了它们的使用。然而，应该注意的是，虽然抽象层简化了这些模块的使用，但它也消除了它们的许多灵活性，所以如果您需要做一些定制，那么这可能不是您的最佳模块。

Concurrent.futures 包含一个名为 **Executor** 的抽象类。但是它不能直接使用，所以你需要使用它的两个子类之一: **ThreadPoolExecutor** 或者 **ProcessPoolExecutor** 。正如您可能已经猜到的，这两个子类分别映射到 Python 的线程和多处理 API。这两个子类都将提供一个池，您可以将线程或进程放入其中。

术语**未来**在计算机科学中有着特殊的含义。它指的是在使用并发编程技术时可以用于同步的构造。**未来**实际上是一种在进程或线程完成处理之前描述其结果的方式。我喜欢把它们看作一个悬而未决的结果。

* * *

### 创建池

当您使用 concurrent.futures 模块时，创建一个工人池是非常容易的。让我们从重写我的 **asyncio** [文章](https://www.blog.pythonlibrary.org/2016/07/26/python-3-an-intro-to-asyncio/)中的下载代码开始，这样它现在使用 concurrent.futures 模块。这是我的版本:

```py

import os
import urllib.request

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

def downloader(url):
    """
    Downloads the specified URL and saves it to disk
    """
    req = urllib.request.urlopen(url)
    filename = os.path.basename(url)
    ext = os.path.splitext(url)[1]
    if not ext:
        raise RuntimeError('URL does not contain an extension')

    with open(filename, 'wb') as file_handle:
        while True:
            chunk = req.read(1024)
            if not chunk:
                break
            file_handle.write(chunk)
    msg = 'Finished downloading {filename}'.format(filename=filename)
    return msg

def main(urls):
    """
    Create a thread pool and download specified urls
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(downloader, url) for url in urls]
        for future in as_completed(futures):
            print(future.result())

if __name__ == '__main__':
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]
    main(urls)

```

首先，我们进口我们需要的东西。然后我们创建我们的**下载器**函数。我继续并稍微更新了它，这样它就可以检查 URL 的末尾是否有扩展名。如果没有，那么我们将引发一个**运行时错误**。接下来，我们创建一个 **main** 函数，在这里线程池被实例化。实际上，您可以将 Python 的 **with** 语句与 ThreadPoolExecutor 和 ProcessPoolExecutor 一起使用，这非常方便。

无论如何，我们设置我们的池，使它有五个工人。然后我们使用一个列表理解来创建一组期货(或工作),最后我们调用 **as_complete** 函数。这个方便的函数是一个迭代器，当它们完成时产生未来。当它们完成时，我们打印出结果，这是一个从我们的 downloader 函数返回的字符串。

如果我们使用的函数计算量非常大，那么我们可以很容易地将 ThreadPoolExecutor 替换为 ProcessPoolExecutor，并且只需要修改一行代码。

我们可以通过使用 concurrent.future 的 **map** 方法来稍微清理一下这段代码。让我们稍微重写一下我们的池代码来利用这一点:

```py

import os
import urllib.request

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

def downloader(url):
    """
    Downloads the specified URL and saves it to disk
    """
    req = urllib.request.urlopen(url)
    filename = os.path.basename(url)
    ext = os.path.splitext(url)[1]
    if not ext:
        raise RuntimeError('URL does not contain an extension')

    with open(filename, 'wb') as file_handle:
        while True:
            chunk = req.read(1024)
            if not chunk:
                break
            file_handle.write(chunk)
    msg = 'Finished downloading {filename}'.format(filename=filename)
    return msg

def main(urls):
    """
    Create a thread pool and download specified urls
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        return executor.map(downloader, urls, timeout=60)

if __name__ == '__main__':
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]
    results = main(urls)
    for result in results:
        print(result)

```

这里的主要区别在于 **main** 函数，它减少了两行代码。 **map** 方法就像 Python 的 map 一样，它接受一个函数和一个 iterable，然后为 iterable 中的每一项调用函数。您还可以为每个线程添加一个超时，这样如果其中一个线程挂起，它就会被停止。最后，从 Python 3.5 开始，他们增加了一个 **chunksize** 参数，当你有一个非常大的 iterable 时，这个参数可以在使用线程池时帮助提高性能。但是，如果您碰巧正在使用进程池，chunksize 将不会有任何影响。

* * *

### 僵局

concurrent.futures 模块的一个缺陷是，当与一个 **Future** 关联的调用者也在等待另一个 Future 的结果时，您可能会意外地创建死锁。这听起来有点令人困惑，所以让我们看一个例子:

```py

from concurrent.futures import ThreadPoolExecutor

def wait_forever():
    """
    This function will wait forever if there's only one
    thread assigned to the pool
    """
    my_future = executor.submit(zip, [1, 2, 3], [4, 5, 6])
    result = my_future.result()
    print(result)

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(wait_forever)

```

这里我们导入 ThreadPoolExecutor 类并创建它的一个实例。请注意，我们将其最大工作线程数设置为一个线程。然后我们提交我们的函数， **wait_forever** 。在我们的函数内部，我们向线程池提交另一个作业，该作业应该将两个列表压缩在一起，获取该操作的结果并将其打印出来。然而，我们刚刚制造了一个僵局！原因是我们有一个未来等待另一个未来结束。基本上，我们希望一个挂起的操作等待另一个不能很好工作的挂起的操作。

让我们稍微重写一下代码，让它工作起来:

```py

from concurrent.futures import ThreadPoolExecutor

def wait_forever():
    """
    This function will wait forever if there's only one
    thread assigned to the pool
    """
    my_future = executor.submit(zip, [1, 2, 3], [4, 5, 6])

    return my_future

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=3)
    fut = executor.submit(wait_forever)

    result = fut.result()
    print(list(result.result()))

```

在这种情况下，我们只是从函数返回内部未来，然后要求其结果。在我们返回的未来上调用 **result** 的结果实际上是另一个未来。如果我们在这个嵌套的 future 上调用 **result** 方法，我们会得到一个 **zip** 对象，所以为了找出实际的结果是什么，我们用 Python 的 **list** 函数包装 zip 并打印出来。

* * *

### 包扎

现在您有了另一个好用的并发工具。您可以根据需要轻松创建线程或进程池。如果您需要运行受网络或 I/O 限制的进程，您可以使用线程池类。如果您有一个计算量很大的任务，那么您会希望使用进程池类来代替。只是要小心不正确地调用期货，否则可能会出现死锁。

* * *

### 相关阅读

*   关于 [concurrent.futures 库](https://docs.python.org/3/library/concurrent.futures.html)的 Python 3 文档
*   Python 冒险: [concurrent.futures](https://pythonadventures.wordpress.com/tag/threadpoolexecutor/)
*   python:concurrent . futures 模块的快速介绍
*   Eli Bendersky: Python - [用 concurrent.futures 并行化 CPU 绑定的任务](http://eli.thegreenplace.net/2013/01/16/python-paralellizing-cpu-bound-tasks-with-concurrent-futures)