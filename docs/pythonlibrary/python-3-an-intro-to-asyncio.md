# python 3——asyncio 简介

> 原文：<https://www.blog.pythonlibrary.org/2016/07/26/python-3-an-intro-to-asyncio/>

在 Python 3.4 版本中， **asyncio** 模块作为临时包被添加到 Python 中。这意味着 asyncio 有可能接收到向后不兼容的更改，甚至可能在 Python 的未来版本中被删除。根据文档 asyncio " *提供了使用协程编写单线程并发代码、通过套接字和其他资源多路复用 I/O 访问、运行网络客户端和服务器以及其他相关原语的基础设施。本章并不打算涵盖 asyncio 的所有功能，但是您将学习如何使用该模块以及它为什么有用。*

如果您在旧版本的 Python 中需要类似 asyncio 的东西，那么您可能想看看 Twisted 或 gevent。

* * *

### 定义

asyncio 模块提供了一个围绕*事件循环*的框架。事件循环基本上是等待某件事情发生，然后对事件进行操作。它负责处理诸如 I/O 和系统事件之类的事情。Asyncio 实际上有几个可用的循环实现。该模块将默认为对于它所运行的操作系统来说最有可能是最有效的模块；但是，如果您愿意，也可以显式选择事件循环。一个事件循环基本上就是说“当事件 A 发生时，用函数 B 反应”。

想象一个服务器，它在等待某人到来并请求一个资源，比如一个网页。如果网站不是很受欢迎，服务器会闲置很长时间。但是当它成功时，服务器需要做出反应。这种反应被称为事件处理。当用户加载网页时，服务器将检查并调用一个或多个事件处理程序。一旦这些事件处理程序完成，它们需要将控制权交还给事件循环。为了在 Python 中做到这一点，asyncio 使用了*协程*。

协程是一个特殊的函数，它可以放弃对调用者的控制而不丢失它的状态。协程是消费者，也是生成器的扩展。与线程相比，它们的一大优势是执行时不会占用太多内存。请注意，当您调用一个协程函数时，它实际上并不执行。相反，它将返回一个协程对象，您可以将该对象传递给事件循环，以便立即或稍后执行它。

使用 asyncio 模块时，您可能会遇到的另一个术语是 *future* 。一个*未来*基本上是一个表示尚未完成的工作结果的对象。您的事件循环可以观察未来的对象，并等待它们完成。当一个未来结束时，它被设置为完成。Asyncio 还支持锁和信号量。

最后一条我要提的信息是*任务*。任务是协程的包装器和未来的子类。您甚至可以使用事件循环来安排任务。

* * *

### 异步和等待

Python 3.5 中添加了 **async** 和**wait**关键字，以定义一个**原生协程**，并使它们与基于生成器的协程相比成为一个独特的类型。如果你想深入了解 async 和 await，你可以看看 PEP 492。

在 Python 3.4 中，您将创建如下所示的协程:

```py

# Python 3.4 coroutine example
import asyncio

@asyncio.coroutine
def my_coro():
    yield from func()

```

这个装饰器在 Python 3.5 中仍然有效，但是 **types** 模块收到了一个以**协程**函数形式的更新，它现在会告诉你你正在交互的是否是一个本地协程。从 Python 3.5 开始，您可以使用**异步定义**来从语法上定义一个协程函数。所以上面的函数看起来会像这样:

```py

import asyncio

async def my_coro():
    await func()

```

当您以这种方式定义协程时，您不能在协程函数中使用 **yield** 。相反，它必须包含一个用于将值返回给调用者的**返回**或**等待**语句。注意 **await** 关键字只能在**异步定义**函数中使用。

**async / await** 关键字可以被认为是用于异步编程的 API。asyncio 模块只是一个框架，恰好使用 **async / await** 进行异步编程。实际上有一个名为 [curio](https://github.com/dabeaz/curio) 的项目证明了这个概念，因为它是一个事件循环的独立实现，在幕后使用了 **async / await** 。

* * *

### 一个糟糕的协同例子

虽然有大量的背景信息来了解所有这些是如何工作的肯定是有帮助的，但是有时您只是想看一些例子，这样您就可以对语法以及如何将这些东西放在一起有一个感觉。记住这一点，让我们从一个简单的例子开始！

你想完成的一个相当常见的任务是从某个地方下载一个文件，不管是内部资源还是互联网上的文件。通常你会想要下载多个文件。因此，让我们创建一对能够做到这一点的协程:

```py

import asyncio
import os
import urllib.request

async def download_coroutine(url):
    """
    A coroutine to download the specified url
    """
    request = urllib.request.urlopen(url)
    filename = os.path.basename(url)

    with open(filename, 'wb') as file_handle:
        while True:
            chunk = request.read(1024)
            if not chunk:
                break
            file_handle.write(chunk)
    msg = 'Finished downloading {filename}'.format(filename=filename)
    return msg

async def main(urls):
    """
    Creates a group of coroutines and waits for them to finish
    """
    coroutines = [download_coroutine(url) for url in urls]
    completed, pending = await asyncio.wait(coroutines)
    for item in completed:
        print(item.result())

if __name__ == '__main__':
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]

    event_loop = asyncio.get_event_loop()
    try:
        event_loop.run_until_complete(main(urls))
    finally:
        event_loop.close()

```

在这段代码中，我们导入我们需要的模块，然后使用 **async** 语法创建我们的第一个协程。这个协程被称为**download _ 协程**，它使用 Python 的 **urllib** 来下载传递给它的任何 URL。完成后，它将返回一条这样的消息。

另一个协程是我们的主协程。它基本上接受一个或多个 URL 的列表，并将它们排队。我们使用 asyncio 的 **wait** 函数来等待协程完成。当然，要真正启动协同程序，需要将它们添加到事件循环中。我们在得到一个事件循环的最后这样做，然后调用它的 **run_until_complete** 方法。你会注意到我们将**主**协程传递给了事件循环。这将开始运行主协程，主协程将第二个协程排队并让它运行。这就是所谓的链式协同程序。

这个例子的问题是，它实际上根本不是一个协程。原因是 **download_coroutine** 函数不是异步的。这里的问题是，urllib 不是异步的，而且，我也没有使用来自的**等待**或**产出。更好的方法是使用 **aiohttp** 包。接下来让我们来看看！**

* * *

### 一个更好的协同例子

aiohttp 包是为创建异步 http 客户端和服务器而设计的。您可以像这样用 pip 安装它:

```py

pip install aiohttp

```

安装完成后，让我们更新代码以使用 aiohttp，这样我们就可以下载文件了:

```py

import aiohttp
import asyncio
import async_timeout
import os

async def download_coroutine(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            filename = os.path.basename(url)
            with open(filename, 'wb') as f_handle:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f_handle.write(chunk)
            return await response.release()

async def main(loop):
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]

    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = [download_coroutine(session, url) for url in urls]
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))

```

您会注意到我们在这里导入了几个新项目: **aiohttp** 和 **async_timeout** 。后者实际上是 aiohttp 的依赖项之一，允许我们创建超时上下文管理器。让我们从代码的底部开始，一步步向上。在底部的条件语句中，我们开始异步事件循环，并调用我们的 main 函数。在 main 函数中，我们创建了一个 **ClientSession** 对象，并将其传递给我们的 download 协程函数，用于我们想要下载的每个 URL。在 **download_coroutine** 中，我们创建了一个 **async_timeout.timeout()** 上下文管理器，它基本上创建了一个 X 秒的计时器。当秒数用完时，上下文管理器结束或超时。在这种情况下，超时时间为 10 秒。接下来，我们调用会话的 **get()** 方法，该方法为我们提供了一个响应对象。现在我们到了有点不可思议的部分。当您使用响应对象的**内容**属性时，它返回一个 **aiohttp 的实例。StreamReader** 允许我们下载任何大小的文件。当我们读取文件时，我们把它写到本地磁盘上。最后我们调用响应的 **release()** 方法，这将完成响应处理。

根据 aiohttp 的文档，因为响应对象是在上下文管理器中创建的，所以它在技术上隐式地调用 release()。但是在 Python 中，显式通常更好，文档中有一个注释，我们不应该依赖于正在消失的连接，所以我认为在这种情况下最好是释放它。

这里仍有一部分被阻塞，这是实际写入磁盘的代码部分。当我们写文件的时候，我们仍然在阻塞。还有另一个名为 [aiofiles](https://github.com/Tinche/aiofiles) 的库，我们可以用它来尝试使文件写入也是异步的，但是我将把更新留给读者。

* * *

### 安排通话

您还可以使用 asyncio 事件循环调度对常规函数的调用。我们要看的第一个方法是 **call_soon** 。 **call_soon** 方法基本上会尽可能快地调用您的回调或事件处理程序。它作为一个 FIFO 队列工作，所以如果一些回调需要一段时间运行，那么其他的回调将被延迟，直到前面的回调完成。让我们看一个例子:

```py

import asyncio
import functools

def event_handler(loop, stop=False):
    print('Event handler called')
    if stop:
        print('stopping the loop')
        loop.stop()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.call_soon(functools.partial(event_handler, loop))
        print('starting event loop')
        loop.call_soon(functools.partial(event_handler, loop, stop=True))

        loop.run_forever()
    finally:
        print('closing event loop')
        loop.close() 

```

asyncio 的大多数函数不接受关键字，所以如果我们需要将关键字传递给事件处理程序，我们将需要 **functools** 模块。每当调用我们的常规函数时，它都会将一些文本输出到 stdout。如果您碰巧将它的**停止**参数设置为**真**，它也会停止事件循环。

第一次调用它时，我们不停止循环。第二次调用时，我们停止了循环。我们想要停止循环的原因是我们已经告诉它 **run_forever** ，这将使事件循环进入无限循环。一旦循环停止，我们就可以关闭它。如果运行此代码，您应该会看到以下输出:

```py

starting event loop
Event handler called
Event handler called
stopping the loop
closing event loop

```

有一个相关的函数叫做 **call_soon_threadsafe** 。顾名思义，它的工作方式与 **call_soon** 相同，但是它是线程安全的。

如果你真的想把一个呼叫延迟到将来的某个时间，你可以使用 **call_later** 功能。在这种情况下，我们可以将 call_soon 签名更改为:

```py

loop.call_later(1, event_handler, loop)

```

这将延迟调用我们的事件处理程序一秒钟，然后它将调用它并将循环作为它的第一个参数传入。

如果你想在未来安排一个特定的时间，那么你需要获取循环的时间而不是计算机的时间。你可以这样做:

```py

current_time = loop.time()

```

一旦你有了它，你就可以使用 **call_at** 函数并传递你想要它调用你的事件处理器的时间。假设我们想在五分钟后调用事件处理程序。你可以这样做:

```py

loop.call_at(current_time + 300, event_handler, loop)

```

在本例中，我们使用获取的当前时间，并在其上附加 300 秒或 5 分钟。通过这样做，我们将调用事件处理程序的时间延迟了五分钟！相当整洁！

* * *

### 任务

任务是未来的子类，是协程的包装器。它们使您能够跟踪它们完成处理的时间。因为它们是未来的一种类型，所以其他协程可以等待一个任务，而你也可以在任务处理完成时获取它的结果。让我们看一个简单的例子:

```py

import asyncio

async def my_task(seconds):
    """
    A task to do for a number of seconds
    """
    print('This task is taking {} seconds to complete'.format(
        seconds))
    await asyncio.sleep(seconds)
    return 'task finished'

if __name__ == '__main__':
    my_event_loop = asyncio.get_event_loop()
    try:
        print('task creation started')
        task_obj = my_event_loop.create_task(my_task(seconds=2))
        my_event_loop.run_until_complete(task_obj)
    finally:
        my_event_loop.close()

    print("The task's result was: {}".format(task_obj.result()))

```

这里我们创建一个异步函数，它接受函数运行所需的秒数。这模拟了一个长时间运行的过程。然后我们创建我们的事件循环，然后通过调用事件循环对象的 **create_task** 函数创建一个任务对象。 **create_task** 函数接受我们想要变成任务的函数。然后我们告诉事件循环运行，直到任务完成。在最后，我们得到任务的结果，因为它已经完成。

通过使用它们的 **cancel** 方法，任务也可以很容易地被取消。当你想结束一个任务的时候就调用它。如果一个任务在等待另一个操作时被取消，该任务将引发一个**取消错误**。

* * *

### 包扎

至此，您应该已经了解了足够多的知识，可以开始自己使用 asyncio 库了。asyncio 库非常强大，允许你做很多非常酷和有趣的任务。Python 文档是开始学习 asyncio 库的好地方。

**更新**:这篇文章最近在这里被翻译成[俄文](http://itscreen.tk/blog/25-python-3-intro-v-asyncio/)。

* * *

### 相关阅读

*   Python 的 asyncio [文档](https://docs.python.org/3/library/asyncio.html)
*   本周 Python 模块: [asyncio](https://pymotw.com/3/asyncio/index.html)
*   Brett Cannon - [在 Python 3.5 中 async / await 到底是如何工作的？](http://www.snarky.ca/how-the-heck-does-async-await-work-in-python-3-5)
*   StackAbuse - Python 异步等待[教程](http://stackabuse.com/python-async-await-tutorial/)
*   [中型](https://medium.com/@greut/a-slack-bot-with-pythons-3-5-asyncio-ad766d8b5d8f#.fez8o6emy) -带有 Python 3.5 的 asyncio 的 slack 机器人
*   Math U Code - [用 Python 3.4 的 Asyncio 和 Node.js 了解异步 IO](http://sahandsaba.com/understanding-asyncio-node-js-python-3-4.html)
*   Dobbs 博士-[Python 3.4 中新的 asyncio 模块:事件循环](http://www.drdobbs.com/open-source/the-new-asyncio-module-in-python-34-even/240168401)
*   有效的 Python Item 40: [考虑协程并发运行许多函数](http://www.informit.com/articles/article.aspx?p=2320938)
*   PEP 492 - [带有异步和等待语法的协同程序](https://www.python.org/dev/peps/pep-0492)