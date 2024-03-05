# Python sleep():如何在代码中添加时间延迟

> 原文：<https://realpython.com/python-sleep/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 sleep()编写一个 Python 正常运行时间 Bot**](/courses/python-sleep-uptime-bot/)

你曾经需要让你的 Python 程序等待什么吗？大多数时候，您会希望代码尽可能快地执行。但是有时候让你的代码休眠一会儿实际上是对你最有利的。

例如，您可以使用 Python `sleep()`调用来模拟程序中的延迟。也许您需要等待文件上传或下载，或者等待图形加载或绘制到屏幕上。您甚至可能需要在调用 web [API](https://realpython.com/python-api/) 或查询数据库之间暂停。在你的程序中添加 Python **`sleep()`** 调用可以帮助你解决这些问题，甚至更多！

**在本教程中，您将学习如何使用**添加 Python `sleep()`调用

*   `time.sleep()`
*   装修工
*   线
*   Async IO
*   图形用户界面

本文面向希望增长 Python 知识的中级开发人员。如果这听起来像你，那么让我们开始吧！

**免费奖励:** ，它向您展示了三种高级装饰模式和技术，您可以用它们来编写更干净、更 Python 化的程序。

## 添加 Python `sleep()`调用`time.sleep()`

Python 内置了让程序休眠的支持。 [`time`模块](https://realpython.com/python-time-module/)有一个函数 [`sleep()`](https://docs.python.org/3/library/time.html#time.sleep) ，你可以用它来暂停执行你指定的任意秒数的调用线程。

这里有一个如何使用`time.sleep()`的例子:

>>>

```py
>>> import time
>>> time.sleep(3) # Sleep for 3 seconds
```

如果在控制台中运行这段代码，那么在 REPL 中输入新语句之前，应该会有一段延迟。

**注:**在 Python 3.5 中，核心开发者对`time.sleep()`的行为做了些许改动。新的 Python `sleep()`系统调用将至少持续您指定的秒数，即使睡眠被信号中断。但是，如果信号本身引发了异常，这就不适用。

你可以通过使用 Python 的 [`timeit`](https://docs.python.org/3/library/timeit.html) 模块来测试睡眠持续多长时间:

```py
$ python3 -m timeit -n 3 "import time; time.sleep(3)"
3 loops, best of 5: 3 sec per loop
```

在这里，您运行带有`-n`参数的`timeit`模块，它告诉`timeit`运行后面的语句的次数。您可以看到`timeit`运行了该语句 3 次，最佳运行时间是 3 秒，这是所期望的。

默认情况下，`timeit`运行代码的次数是一百万次。如果您使用默认的`-n`运行上面的代码，那么每次迭代 3 秒，您的终端将会挂起大约 34 天！`timeit`模块有几个其他的命令行选项，你可以在它的[文档](https://docs.python.org/3/library/timeit.html#timeit-command-line-interface)中查看。

让我们创造一些更真实的东西。系统管理员需要知道他们的一个网站何时关闭。您希望能够定期检查网站的状态代码，但不能经常查询 web 服务器，否则会影响性能。进行这种检查的一种方法是使用 Python `sleep()`系统调用:

```py
import time
import urllib.request
import urllib.error

def uptime_bot(url):
    while True:
        try:
            conn = urllib.request.urlopen(url)
        except urllib.error.HTTPError as e:
            # Email admin / log
            print(f'HTTPError: {e.code} for {url}')
        except urllib.error.URLError as e:
            # Email admin / log
            print(f'URLError: {e.code} for {url}')
        else:
            # Website is up
            print(f'{url} is up')
 time.sleep(60) 
if __name__ == '__main__':
    url = 'http://www.google.com/py'
    uptime_bot(url)
```

这里您创建了`uptime_bot()`，它将一个 URL 作为它的参数。然后，该函数尝试用`urllib`打开该 URL。如果有一个`HTTPError`或`URLError`，那么程序会捕捉到它并打印出错误。(在实际环境中，您可能会记录错误，并向网站管理员或系统管理员发送电子邮件。)

如果没有错误发生，那么您的代码将显示一切正常。不管发生什么，你的程序都会休眠 60 秒。这意味着你每分钟只能访问一次网站。此示例中使用的 URL 是错误的，因此它将每分钟向您的控制台输出一次以下内容:

```py
HTTPError: 404 for http://www.google.com/py
```

继续更新代码，使用一个已知良好的网址，如 [`http://www.google.com`](http://www.google.com) 。然后，您可以重新运行它，以查看它是否成功工作。您也可以尝试更新代码来发送电子邮件或记录错误。关于如何做到这一点的更多信息，请查看使用 Python 发送电子邮件的和登录 Python 的。

[*Remove ads*](/account/join/)

## 用 decorator添加 Python `sleep()`调用

有时，您需要重试一个失败的功能。这种情况的一个常见用例是当您因为服务器繁忙而需要重试文件下载时。您通常不希望过于频繁地向服务器发出请求，所以在每个请求之间添加一个 Python `sleep()`调用是可取的。

我亲身经历的另一个用例是，我需要在自动化测试期间检查用户界面的状态。用户界面的加载速度可能比平时快或慢，这取决于我运行测试的计算机。这会改变我的程序正在验证的时候屏幕上的内容。

在这种情况下，我可以告诉程序休眠一会儿，然后在一两秒钟后重新检查。这可能意味着通过测试和失败测试之间的差别。

在这两种情况下，您都可以使用一个**装饰器**来添加一个 Python `sleep()`系统调用。如果你不熟悉装饰者，或者如果你想重温他们，那么看看 Python 装饰者的[初级读本。让我们看一个例子:](https://realpython.com/primer-on-python-decorators/)

```py
import time
import urllib.request
import urllib.error

def sleep(timeout, retry=3):
    def the_real_decorator(function):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < retry:
                try:
                    value = function(*args, **kwargs)
                    if value is None:
                        return
                except:
                    print(f'Sleeping for {timeout} seconds')
                    time.sleep(timeout)
                    retries += 1
        return wrapper
    return the_real_decorator
```

是你的装潢师。它接受一个`timeout`值和它应该接受的次数`retry`，默认为 3。在`sleep()`内部是另一个函数`the_real_decorator()`，它接受修饰函数。

最后，最里面的函数`wrapper()`接受您传递给修饰函数的参数和关键字参数。这就是奇迹发生的地方！您使用一个 [`while`](https://realpython.com/courses/mastering-while-loops/) 循环来重试调用该函数。如果有异常，那么您调用`time.sleep()`，递增`retries`计数器，并尝试再次运行该函数。

现在重写`uptime_bot()`来使用你的新装饰器:

```py
@sleep(3) def uptime_bot(url):
    try:
        conn = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        # Email admin / log
        print(f'HTTPError: {e.code} for {url}')
        # Re-raise the exception for the decorator
        raise urllib.error.HTTPError
    except urllib.error.URLError as e:
        # Email admin / log
        print(f'URLError: {e.code} for {url}')
        # Re-raise the exception for the decorator
        raise urllib.error.URLError
    else:
        # Website is up
        print(f'{url} is up')

if __name__ == '__main__':
    url = 'http://www.google.com/py'
    uptime_bot(url)
```

这里，你用 3 秒的`sleep()`来修饰`uptime_bot()`。您还删除了原来的 [`while`循环](https://realpython.com/python-while-loop/)，以及旧的调用`sleep(60)`。装潢师现在负责这个。

您所做的另一个更改是在异常处理块中添加了一个`raise`。这是为了让装修工正常工作。您可以编写装饰器来处理这些错误，但是由于这些异常只适用于`urllib`，您最好保持装饰器的原样。这样，它将与更广泛的功能一起工作。

**注意:**如果你想温习一下 Python 中的异常处理，那么请查看 [Python 异常:简介](https://realpython.com/python-exceptions/)。

你可以对你的室内设计师做一些改进。如果它用尽了重试次数，仍然失败，那么您可以让它重新引发上一个错误。装饰者还会在最后一次失败后等待 3 秒钟，这可能是您不希望发生的事情。请随意尝试这些作为练习！

## 用线程添加 Python `sleep()`调用

有时候，你可能想给一个**线程**添加一个 Python `sleep()`调用。也许您正在针对生产环境中有数百万条记录的数据库运行迁移脚本。您不想造成任何停机，但是也不想等待过长的时间来完成迁移，所以您决定使用线程。

**注意:**线程是 Python 中进行[并发](https://realpython.com/python-concurrency/)的一种方法。您可以同时运行多个线程来增加应用程序的吞吐量。如果你不熟悉 Python 中的线程，那么请查看[Python 中的线程介绍](https://realpython.com/intro-to-python-threading/)。

为了防止客户注意到任何类型的速度下降，每个线程都需要运行一小段时间，然后休眠。有两种方法可以做到这一点:

1.  像以前一样使用`time.sleep()`。
2.  使用`threading`模块中的`Event.wait()`。

我们先来看一下`time.sleep()`。

[*Remove ads*](/account/join/)

### 使用`time.sleep()`

Python [日志食谱](https://docs.python.org/3/howto/logging-cookbook.html#logging-from-multiple-threads)展示了一个使用`time.sleep()`的好例子。Python 的`logging`模块是线程安全的，所以在这个练习中，它比 [`print()`语句](https://realpython.com/python-print/)更有用一些。以下代码基于此示例:

```py
import logging
import threading
import time

def worker(arg):
    while not arg["stop"]:
        logging.debug("worker thread checking in")
 time.sleep(1) 
def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(relativeCreated)6d  %(threadName)s  %(message)s"
    )
    info = {"stop": False}
    thread = threading.Thread(target=worker, args=(info,))
    thread_two = threading.Thread(target=worker, args=(info,))
    thread.start()
    thread_two.start()

    while True:
        try:
            logging.debug("Checking in from main thread")
 time.sleep(0.75)        except KeyboardInterrupt:
            info["stop"] = True
            logging.debug('Stopping')
            break
    thread.join()
    thread_two.join()

if __name__ == "__main__":
    main()
```

这里，您使用 Python 的`threading`模块来创建两个线程。您还创建了一个日志记录对象，它将把`threadName`记录到 stdout 中。接下来，启动两个线程，并不时地从主线程启动一个循环来记录日志。你用`KeyboardInterrupt`来捕捉用户按下 `Ctrl` + `C` 。

尝试在您的终端上运行上面的代码。您应该会看到类似如下的输出:

```py
 0 Thread-1 worker thread checking in
 1 Thread-2 worker thread checking in
 1 MainThread Checking in from main thread
752 MainThread Checking in from main thread
1001 Thread-1 worker thread checking in
1001 Thread-2 worker thread checking in
1502 MainThread Checking in from main thread
2003 Thread-1 worker thread checking in
2003 Thread-2 worker thread checking in
2253 MainThread Checking in from main thread
3005 Thread-1 worker thread checking in
3005 MainThread Checking in from main thread
3005 Thread-2 worker thread checking in
```

当每个线程运行然后休眠时，日志输出被打印到控制台。既然您已经尝试了一个示例，您将能够在自己的代码中使用这些概念。

### 使用`Event.wait()`

`threading`模块提供了一个`Event()`，你可以像`time.sleep()`一样使用它。然而，`Event()`有一个额外的好处，那就是响应速度更快。这样做的原因是，当事件被设置时，程序将立即跳出循环。使用`time.sleep()`，您的代码将需要等待 Python `sleep()`调用完成，然后线程才能退出。

这里你想使用`wait()`的原因是因为`wait()`是**非阻塞**，而`time.sleep()`是**阻塞**。这意味着当您使用`time.sleep()`时，您将阻止主线程继续运行，同时等待`sleep()`调用结束。`wait()`解决了这个问题。你可以在 Python 的[线程文档](https://docs.python.org/3/library/threading.html#event-objects)中了解更多关于这一切是如何工作的。

下面是如何用`Event.wait()`添加一个 Python `sleep()`调用:

```py
import logging
import threading

def worker(event):
    while not event.isSet():
        logging.debug("worker thread checking in")
 event.wait(1) 
def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(relativeCreated)6d  %(threadName)s  %(message)s"
    )
 event = threading.Event() 
    thread = threading.Thread(target=worker, args=(event,))
    thread_two = threading.Thread(target=worker, args=(event,))
    thread.start()
    thread_two.start()

    while not event.isSet():
        try:
            logging.debug("Checking in from main thread")
 event.wait(0.75)        except KeyboardInterrupt:
            event.set()
            break

if __name__ == "__main__":
    main()
```

在这个例子中，您创建了`threading.Event()`并将其传递给`worker()`。(回想一下，在前面的例子中，您传递了一个字典。)接下来，你设置你的循环来检查`event`是否被设置。如果不是，那么您的代码会打印一条消息，并在再次检查之前等待一段时间。要设置事件，可以按 `Ctrl` + `C` 。一旦事件被设置，`worker()`将返回，循环将中断，程序结束。

**注意:**如果你想了解更多关于字典的知识，那么就来看看 Python 中的[字典](https://realpython.com/courses/dictionaries-python/)。

仔细看看上面的代码块。你如何给每个工作线程分配不同的睡眠时间？你能搞清楚吗？请自行解决这个练习！

## 使用异步 IO 添加 Python `sleep()`调用

在 3.4 版本中，Python 增加了异步功能，从那以后，这个特性集一直在积极地扩展。**异步编程**是一种并行编程，允许你一次运行多个任务。当一个任务完成时，它会通知主线程。

`asyncio`是一个模块，允许您异步添加 Python `sleep()`调用。如果你不熟悉 Python 异步编程的实现，那么看看 Python 中的[异步 IO:完整演练](https://realpython.com/async-io-python/)和 [Python 并发&并行编程](https://realpython.com/learning-paths/python-concurrency-parallel-programming/)。

这里有一个来自 Python 自己的[文档](https://docs.python.org/3/library/asyncio.html)的例子:

```py
import asyncio

async def main():
    print('Hello ...')
 await asyncio.sleep(1)    print('... World!')

# Python 3.7+
asyncio.run(main())
```

在这个例子中，您运行`main()`，让它在两次 [`print()`](https://realpython.com/courses/python-print/) 调用之间休眠一秒钟。

这里有一个来自`asyncio`文档的[协程和任务](https://docs.python.org/3/library/asyncio-task.html)部分的更有说服力的例子:

```py
import asyncio
import time

async def output(sleep, text):
 await asyncio.sleep(sleep)    print(text)

async def main():
    print(f"Started: {time.strftime('%X')}")
 await output(1, 'First') await output(2, 'Second') await output(3, 'Third')    print(f"Ended: {time.strftime('%X')}")

# Python 3.7+
asyncio.run(main())
```

在这段代码中，您创建了一个名为`output()`的工作器，它接收到`sleep`的秒数和要打印出来的`text`。然后，使用 Python 的`await`关键字等待`output()`代码运行。这里需要`await`,因为`output()`已经被标记为`async`函数，所以你不能像调用普通函数一样调用它。

当您运行这段代码时，您的程序将执行`await` 3 次。代码将等待 1 秒、2 秒和 3 秒，总等待时间为 6 秒。您也可以重写代码，使任务并行运行:

```py
import asyncio
import time

async def output(text, sleep):
    while sleep > 0:
        await asyncio.sleep(1)
        print(f'{text} counter: {sleep} seconds')
        sleep -= 1

async def main():
    task_1 = asyncio.create_task(output('First', 1))
    task_2 = asyncio.create_task(output('Second', 2))
    task_3 = asyncio.create_task(output('Third', 3))
    print(f"Started: {time.strftime('%X')}")
    await task_1
    await task_2
    await task_3                                 
    print(f"Ended: {time.strftime('%X')}")

if __name__ == '__main__':
    asyncio.run(main())
```

现在你正在使用**任务**的概念，你可以用`create_task()`来完成它。当您在`asyncio`中使用任务时，Python 将异步运行任务。因此，当您运行上面的代码时，它应该在总共 3 秒内完成，而不是 6 秒。

[*Remove ads*](/account/join/)

## 使用 GUI添加 Python `sleep()`调用

命令行应用程序并不是唯一需要添加 Python `sleep()`调用的地方。当你创建一个**图形用户界面(GUI)** 时，你偶尔会需要添加延迟。例如，您可能创建一个 FTP 应用程序来下载数百万个文件，但是您需要在批处理之间添加一个`sleep()`调用，这样您就不会使服务器陷入困境。

GUI 代码将在一个名为**事件循环**的主线程中运行所有的处理和绘图。如果您在 GUI 代码中使用`time.sleep()`，那么您将阻塞它的事件循环。从用户的角度来看，应用程序可能会冻结。当应用程序使用这种方法休眠时，用户将无法与应用程序进行交互。(在 Windows 上，您甚至可能会收到一个关于您的应用程序现在如何无响应的警告。)

幸运的是，除了`time.sleep()`还有其他方法可以使用。在接下来的几节中，您将学习如何在 [Tkinter](https://realpython.com/python-gui-tkinter/) 和 wxPython 中添加 Python `sleep()`调用。

### 睡在 Tkinter

[`tkinter`](https://docs.python.org/3/library/tk.html) 是 Python 标准库的一部分。如果您在 Linux 或 Mac 上使用的是预装版本的 Python，则可能无法使用它。如果你得到了一个`ImportError`，那么你需要考虑如何将它添加到你的系统中。但是如果你[自己安装 Python](https://realpython.com/installing-python/)，那么`tkinter`应该已经可以用了。

首先，我们来看一个使用`time.sleep()`的例子。运行这段代码，看看当你以错误的方式添加 Python `sleep()`调用时会发生什么:

```py
import tkinter
import time

class MyApp:
    def __init__(self, parent):
        self.root = parent
        self.root.geometry("400x400")
        self.frame = tkinter.Frame(parent)
        self.frame.pack()
        b = tkinter.Button(text="click me", command=self.delayed)
        b.pack()

    def delayed(self):
 time.sleep(3) 
if __name__ == "__main__":
    root = tkinter.Tk()
    app = MyApp(root)
    root.mainloop()
```

运行完代码后，按下 GUI 中的按钮。在等待`sleep()`完成的过程中，该按钮将持续三秒钟。如果应用程序有其他按钮，那么你将无法点击它们。您也不能在应用程序睡眠时关闭它，因为它不能响应 close 事件。

为了让`tkinter`正常睡眠，你需要使用`after()`:

```py
import tkinter

class MyApp:
    def __init__(self, parent):
        self.root = parent
        self.root.geometry("400x400")
        self.frame = tkinter.Frame(parent)
        self.frame.pack()
 self.root.after(3000, self.delayed) 
    def delayed(self):
        print('I was delayed')

if __name__ == "__main__":
    root = tkinter.Tk()
    app = MyApp(root)
    root.mainloop()
```

这里您创建了一个 400 像素宽 400 像素高的应用程序。它上面没有小部件。它只会显示一个框架。然后，调用`self.root.after()`，其中`self.root`是对`Tk()`对象的引用。`after()`需要两个参数:

1.  休眠的毫秒数
2.  睡眠结束时要调用的方法

在这种情况下，您的应用程序将在 3 秒钟后将一个字符串打印到 stdout。你可以把`after()`看作是`time.sleep()`的`tkinter`版本，但是它也增加了在睡眠结束后调用函数的能力。

您可以使用该功能来改善用户体验。通过添加 Python `sleep()`调用，您可以让应用程序看起来加载得更快，然后在启动后启动一些运行时间更长的进程。这样，用户就不必等待应用程序打开。

### 睡在 wxPython

wxPython 和 Tkinter 之间有两个主要区别:

1.  wxPython 有更多的小部件。
2.  wxPython 的目标是在所有平台上都具有原生的外观和感觉。

Python 中没有包含 wxPython 框架，所以您需要自己安装它。如果你不熟悉 wxPython，那么看看[如何用 wxPython](https://realpython.com/python-gui-with-wxpython/) 构建一个 Python GUI 应用程序。

在 wxPython 中，可以使用`wx.CallLater()`添加一个 Python `sleep()`调用:

```py
import wx

class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='Hello World')
 wx.CallLater(4000, self.delayed)        self.Show()

    def delayed(self):
        print('I was delayed')

if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
```

这里，你直接子类化`wx.Frame`，然后调用`wx.CallLater()`。该函数采用与 Tkinter 的`after()`相同的参数:

1.  休眠的毫秒数
2.  睡眠结束时要调用的方法

运行这段代码时，您应该会看到一个没有任何小部件的空白小窗口。4 秒钟后，您将看到字符串`'I was delayed'`被打印到 stdout。

使用`wx.CallLater()`的好处之一是它是线程安全的。您可以在线程中使用该方法来调用主 wxPython 应用程序中的函数。

[*Remove ads*](/account/join/)

## 结论

通过本教程，您已经获得了一项可添加到 Python 工具箱中的有价值的新技术！您知道如何添加延迟来调整应用程序的速度，并防止它们耗尽系统资源。您甚至可以使用 Python `sleep()`调用来帮助您的 GUI 代码更有效地重绘。这将为您的客户带来更好的用户体验！

概括地说，您已经学习了如何使用以下工具添加 Python `sleep()`调用:

*   `time.sleep()`
*   装修工
*   线
*   `asyncio`
*   Tkinter
*   wxPython

现在，您可以利用您所学到的知识，开始让您的代码休眠了！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 sleep()编写一个 Python 正常运行时间 Bot**](/courses/python-sleep-uptime-bot/)******