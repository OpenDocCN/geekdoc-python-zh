# aiohttp 简介

> 原文：<https://www.blog.pythonlibrary.org/2016/11/09/an-intro-to-aiohttp/>

Python 3.5 增加了一些新的语法，允许开发人员更容易地创建异步应用程序和包。一个这样的包是 aiohttp，它是 asyncio 的 http 客户端/服务器。基本上，它允许你编写异步客户端和服务器。aiohttp 包还支持服务器 WebSockets 和客户端 WebSockets。您可以使用 pip 安装 aiohttp:

```py

pip install aiohttp

```

现在我们已经安装了 aiohttp，让我们来看看他们的一个例子！

* * *

### 获取网页

aiohtpp 的文档中有一个有趣的例子,展示了如何获取网页的 HTML。让我们来看看它是如何工作的:

```py

import aiohttp
import asyncio
import async_timeout

async def fetch(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()

async def main(loop):
    async with aiohttp.ClientSession(loop=loop) as session:
        html = await fetch(session, 'https://www.blog.pythonlibrary.org')
        print(html)

loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))

```

在这里，我们只需导入 aiohttp、Python 的 asyncio 和 async_timeout，这使我们能够使协程超时。我们在代码底部创建事件循环，并调用 main()函数。它将创建一个 ClientSession 对象，我们将这个对象和要获取的 URL 一起传递给 fetch()函数。最后，在 fetch()函数中，我们使用 our timeout 并尝试获取 URL 的 HTML。如果一切正常，没有超时，您将会看到大量文本涌入 stdout。

* * *

### 用 aiohttp 下载文件

开发人员要做的一个相当常见的任务是使用线程或进程下载文件。我们也可以使用协程下载文件！让我们来看看如何实现:

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
        for url in urls:
            await download_coroutine(session, url)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))

```

您会注意到我们在这里导入了几个新项目: **aiohttp** 和 **async_timeout** 。后者实际上是 aiohttp 的依赖项之一，允许我们创建超时上下文管理器。让我们从代码的底部开始，一步步向上。在底部的条件语句中，我们开始异步事件循环，并调用我们的 main 函数。在 main 函数中，我们创建了一个 **ClientSession** 对象，并将其传递给我们的 download 协程函数，用于我们想要下载的每个 URL。在 **download_coroutine** 中，我们创建了一个 **async_timeout.timeout()** 上下文管理器，它基本上创建了一个 X 秒的计时器。当秒数用完时，上下文管理器结束或超时。在这种情况下，超时时间为 10 秒。接下来，我们调用会话的 **get()** 方法，该方法为我们提供了一个响应对象。现在我们到了有点不可思议的部分。当您使用响应对象的**内容**属性时，它返回一个 **aiohttp 的实例。StreamReader** 允许我们下载任何大小的文件。当我们读取文件时，我们把它写到本地磁盘上。最后我们调用响应的 **release()** 方法，这将完成响应处理。

根据 aiohttp 的文档，因为响应对象是在上下文管理器中创建的，所以它在技术上隐式地调用 release()。但是在 Python 中，显式通常更好，文档中有一个注释，我们不应该依赖于正在消失的连接，所以我认为在这种情况下最好是释放它。

这里仍有一部分被阻塞，这是实际写入磁盘的代码部分。当我们写文件的时候，我们仍然在阻塞。还有另一个名为 [aiofiles](https://github.com/Tinche/aiofiles) 的库，我们可以用它来尝试使文件写异步，我们接下来会看到它。

*注意:上面的部分来自我以前的[文章](https://www.blog.pythonlibrary.org/2016/07/26/python-3-an-intro-to-asyncio/)。*

* * *

### 使用 aiofiles 进行异步写入

你将需要安装 [aiofiles](https://github.com/Tinche/aiofiles) 来完成这项工作。让我们弄清楚这一点:

```py

pip install aiofiles

```

现在我们有了所有需要的项目，我们可以更新我们的代码了！注意，这段代码只在 Python 3.6 或更高版本中有效。

```py

import aiofiles
import aiohttp
import asyncio
import async_timeout
import os

async def download_coroutine(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            filename = os.path.basename(url)
            async with aiofiles.open(filename, 'wb') as fd:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    await fd.write(chunk)
            return await response.release()

async def main(loop, url):
    async with aiohttp.ClientSession(loop=loop) as session:
        await download_coroutine(session, url)

if __name__ == '__main__':
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
            asyncio.gather(
                *(main(loop, url) for url in urls)
                )
                )

```

唯一的变化是为 **aiofiles** 添加一个导入，然后改变我们打开文件的方式。你会注意到现在是

```py

async with aiofiles.open(filename, 'wb') as fd:

```

我们使用 await 来编写代码:

```py

await fd.write(chunk)

```

除此之外，代码是相同的。这里提到的一些可移植性问题[你应该知道。](https://github.com/python/asyncio/wiki/ThirdParty#filesystem)

* * *

### 包扎

现在你应该对如何使用 aiohttp 和 aiofiles 有了一些基本的了解。这两个项目的文档都值得一看，因为本教程实际上只是触及了这些库的皮毛。

* * *

### 相关阅读

*   介绍 [asyncio](https://www.blog.pythonlibrary.org/2016/07/26/python-3-an-intro-to-asyncio/)
*   aiohttp 官方[稳定发布](https://www.reddit.com/r/Python/comments/53ohlv/aiohttp_10_the_first_officially_stable_release/)
*   aiohttp [文档](http://aiohttp.readthedocs.io/en/stable/)
*   aiohttp [Github](https://github.com/KeepSafe/aiohttp)
*   aiofiles [Github](https://github.com/Tinche/aiofiles)