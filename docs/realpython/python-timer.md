# Python 计时器函数:监控代码的三种方式

> 原文：<https://realpython.com/python-timer/>

虽然许多开发人员[认为](https://www.python.org/doc/essays/blurb/) Python 是一种有效的编程语言，但纯 Python 程序可能比编译语言(如 [C](https://realpython.com/c-for-python-programmers/) 、Rust 和 [Java](https://realpython.com/oop-in-python-vs-java/) )中的程序运行得更慢。在本教程中，你将学习如何使用一个 **Python 定时器**来监控你的程序运行的速度。

在本教程中，你将学习如何使用:

*   **`time.perf_counter()`** 用 Python 来度量时间
*   **类**保持状态
*   **上下文管理器**处理代码块
*   **装饰者**定制一个功能

您还将获得关于类、上下文管理器和装饰器如何工作的背景知识。当您探索每个概念的示例时，您会受到启发，在代码中使用其中的一个或几个来计时代码执行，以及在其他应用程序中。每种方法都有其优点，您将根据具体情况学习使用哪种方法。另外，您将拥有一个工作的 Python 计时器，可以用来监控您的程序！

**Decorators Q &文字记录:** [点击此处获取我们 Python decorators Q &的 25 页聊天记录，这是真实 Python 社区 Slack 中的一个会话](https://realpython.com/bonus/decorators-qa-2019/)，我们在这里讨论了常见的 decorator 问题。

## Python 定时器

首先，您将看到一些在整个教程中使用的示例代码。稍后，您将向该代码添加一个 **Python 计时器**来监控其性能。您还将学习一些最简单的方法来测量这个例子的运行时间。

[*Remove ads*](/account/join/)

### Python 定时器功能

如果你查看 Python 内置的 [`time`](https://docs.python.org/3/library/time.html) 模块，你会注意到几个可以测量时间的函数:

*   [T2`monotonic()`](https://docs.python.org/3/library/time.html#time.monotonic)
*   [T2`perf_counter()`](https://docs.python.org/3/library/time.html#time.perf_counter)
*   [T2`process_time()`](https://docs.python.org/3/library/time.html#time.process_time)
*   [T2`time()`](https://docs.python.org/3/library/time.html#time.time)

[Python 3.7](https://realpython.com/python37-new-features/) 引入了几个新的函数，像 [`thread_time()`](https://docs.python.org/3/library/time.html#time.thread_time) ，以及上面所有函数的**纳秒**版本，以`_ns`后缀命名。比如 [`perf_counter_ns()`](https://docs.python.org/3/library/time.html#time.perf_counter_ns) 就是`perf_counter()`的纳秒版本。稍后您将了解更多关于这些函数的内容。现在，请注意文档中对`perf_counter()`的描述:

> 返回性能计数器的值(以秒为单位),即具有最高可用分辨率来测量短时间的时钟。([来源](https://docs.python.org/3/library/time.html#time.perf_counter))

首先，您将使用`perf_counter()`创建一个 Python 定时器。[稍后](#other-python-timer-functions)，您将把它与其他 Python 计时器函数进行比较，并了解为什么`perf_counter()`通常是最佳选择。

### 示例:下载教程

为了更好地比较向代码中添加 Python 计时器的不同方法，在本教程中，您将对同一代码示例应用不同的 Python 计时器函数。如果您已经有了想要度量的代码，那么您可以自由地跟随示例。

本教程中您将使用的示例是一个简短的函数，它使用 [`realpython-reader`](https://pypi.org/project/realpython-reader/) 包下载 Real Python 上的最新教程。要了解更多关于真正的 Python Reader 及其工作原理，请查看[如何将开源 Python 包发布到 PyPI](https://realpython.com/pypi-publish-python-package/) 。您可以使用 [`pip`](https://realpython.com/what-is-pip/) 在您的系统上安装`realpython-reader`:

```py
$ python -m pip install realpython-reader
```

然后，你可以[导入](https://realpython.com/python-import/)这个包作为`reader`。

您将把这个例子存储在一个名为`latest_tutorial.py`的文件中。代码由一个函数组成，该函数下载并打印 Real Python 的最新教程:

```py
 1# latest_tutorial.py
 2
 3from reader import feed
 4
 5def main():
 6    """Download and print the latest tutorial from Real Python"""
 7    tutorial = feed.get_article(0)
 8    print(tutorial)
 9
10if __name__ == "__main__":
11    main()
```

处理大部分艰难的工作:

*   **三号线**从`realpython-reader`进口`feed`。该模块包含从[真实 Python 提要](https://realpython.com/contact/#rss-atom-feed)下载教程的功能。
*   **第 7 行**从 Real Python 下载最新教程。数字`0`是一个偏移量，其中`0`表示最近的教程，`1`是以前的教程，依此类推。
*   **第 8 行**将教程打印到控制台。
*   运行脚本时，第 11 行调用`main()`。

当您运行此示例时，您的输出通常如下所示:

```py
$ python latest_tutorial.py
# Python Timer Functions: Three Ways to Monitor Your Code

While many developers recognize Python as an effective programming language,
pure Python programs may run more slowly than their counterparts in compiled
languages like C, Rust, and Java. In this tutorial, you'll learn how to use
a Python timer to monitor how quickly your programs are running.

[ ... ]

## Read the full article at https://realpython.com/python-timer/ »

* * *
```

根据您的网络，代码可能需要一段时间运行，因此您可能希望使用 Python 计时器来监控脚本的性能。

### 你的第一个 Python 定时器

现在，您将使用`time.perf_counter()`向示例添加一个基本的 Python 计时器。同样，这是一个**性能计数器**，非常适合为你的代码计时。

`perf_counter()`以秒为单位测量从某个未指定的时刻开始的时间，这意味着对函数的单次调用的返回值是没有用的。然而，当您查看对`perf_counter()`的两次调用之间的差异时，您可以计算出两次调用之间经过了多少秒:

>>>

```py
>>> import time
>>> time.perf_counter()
32311.48899951

>>> time.perf_counter()  # A few seconds later
32315.261320793
```

在这个例子中，你给`perf_counter()`打了两个电话，几乎相隔 4 秒。您可以通过计算两个输出之间的差异来确认这一点:32315.26 - 32311.49 = 3.77。

现在，您可以将 Python 计时器添加到示例代码中:

```py
 1# latest_tutorial.py
 2
 3import time 4from reader import feed
 5
 6def main():
 7    """Print the latest tutorial from Real Python"""
 8    tic = time.perf_counter() 9    tutorial = feed.get_article(0)
10    toc = time.perf_counter() 11    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds") 12
13    print(tutorial)
14
15if __name__ == "__main__":
16    main()
```

注意，在下载教程之前和之后都要调用`perf_counter()`。然后，通过计算两次调用之间的差异，打印下载教程所用的时间。

**注意:**在第 11 行，字符串前的`f`表示这是一个 **f 字符串**，这是一种格式化文本字符串的便捷方式。`:0.4f`是一个格式说明符，表示数字`toc - tic`应该打印为一个有四位小数的十进制数。

有关 f 字符串的更多信息，请查看 Python 3 的 f 字符串:一种改进的字符串格式化语法。

现在，当您运行该示例时，您将看到教程开始之前所用的时间:

```py
$ python latest_tutorial.py
Downloaded the tutorial in 0.6721 seconds # Python Timer Functions: Three Ways to Monitor Your Code

[ ... ]
```

就是这样！您已经讲述了为自己的 Python 代码计时的基础知识。在本教程的其余部分，您将了解如何将 Python 计时器封装到一个类、一个上下文管理器和一个装饰器中，以使它更加一致和方便使用。

[*Remove ads*](/account/join/)

## 一个 Python 定时器类

回头看看您是如何将 Python 计时器添加到上面的示例中的。注意，在下载教程之前，您至少需要一个变量(`tic`)来存储 Python 定时器的状态。稍微研究了一下代码之后，您可能还会注意到，添加这三个突出显示的行只是为了计时！现在，您将创建一个与您手动调用`perf_counter()`相同的类，但是以一种更加可读和一致的方式。

在本教程中，您将创建并更新`Timer`，这个类可以用来以几种不同的方式为您的代码计时。带有一些额外特性的最终代码也可以在 [PyPI](https://pypi.org/project/codetiming) 上以`codetiming`的名字获得。您可以像这样在您的系统上安装它:

```py
$ python -m pip install codetiming
```

你可以在本教程后面的[部分找到更多关于`codetiming`的信息，Python 计时器代码](#the-python-timer-code)。

### 理解 Python 中的类

**类**是[面向对象编程](https://realpython.com/python3-object-oriented-programming/)的主要构件。一个**类**本质上是一个模板，你可以用它来创建**对象**。虽然 Python 并不强迫你以面向对象的方式编程，但类在这种语言中无处不在。为了快速证明，研究 [`time`模块](https://realpython.com/python-time-module/):

>>>

```py
>>> import time
>>> type(time)
<class 'module'>

>>> time.__class__
<class 'module'>
```

`type()`返回对象的类型。这里你可以看到模块实际上是从一个`module`类创建的对象。您可以使用特殊属性`.__class__`来访问定义对象的类。事实上，Python 中的几乎所有东西都是一个类:

>>>

```py
>>> type(3)
<class 'int'>

>>> type(None)
<class 'NoneType'>

>>> type(print)
<class 'builtin_function_or_method'>

>>> type(type)
<class 'type'>
```

在 Python 中，当您需要对需要跟踪特定状态的东西建模时，类非常有用。一般来说，一个类是称为**属性**的属性和称为**方法**的行为的集合。关于类和面向对象编程的更多背景知识，请查看 Python 3 中的[面向对象编程(OOP)或](https://realpython.com/python3-object-oriented-programming/)[官方文档](https://docs.python.org/3/tutorial/classes.html)。

### 创建 Python 定时器类

类有利于跟踪状态。在一个`Timer`类中，你想要记录一个计时器何时开始计时，以及从那时起已经过了多长时间。对于`Timer`的第一个实现，您将添加一个`._start_time`属性，以及`.start()`和`.stop()`方法。将以下代码添加到名为`timer.py`的文件中:

```py
 1# timer.py
 2
 3import time
 4
 5class TimerError(Exception):
 6    """A custom exception used to report errors in use of Timer class"""
 7
 8class Timer:
 9    def __init__(self):
10        self._start_time = None
11
12    def start(self):
13        """Start a new timer"""
14        if self._start_time is not None:
15            raise TimerError(f"Timer is running. Use .stop() to stop it")
16
17        self._start_time = time.perf_counter()
18
19    def stop(self):
20        """Stop the timer, and report the elapsed time"""
21        if self._start_time is None:
22            raise TimerError(f"Timer is not running. Use .start() to start it")
23
24        elapsed_time = time.perf_counter() - self._start_time
25        self._start_time = None
26        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
```

这里发生了一些不同的事情，所以花点时间一步一步地浏览代码。

在第 5 行，您定义了一个`TimerError`类。`(Exception)`符号意味着`TimerError` **从另一个名为`Exception`的类继承了**。Python 使用这个内置的类进行错误处理。您不需要给`TimerError`添加任何属性或方法，但是拥有一个自定义错误会让您更加灵活地处理`Timer`内部的问题。更多信息，请查看 [Python 异常:简介](https://realpython.com/python-exceptions/)。

`Timer`本身的定义从第 8 行开始。当您第一次创建或**实例化**一个来自类的对象时，您的代码调用特殊方法`.__init__()`。在`Timer`的第一个版本中，您只初始化了`._start_time`属性，它将用于跟踪您的 Python 定时器的状态。当计时器不运行时，它的值为`None`。一旦定时器开始运行，`._start_time`会跟踪定时器的启动时间。

**注意:**`._start_time`的[下划线](https://namingconvention.org/python/underscore.html) ( `_`)前缀是 Python 约定。它表明`._start_time`是一个内部属性，用户不应该操纵`Timer`类。

当您调用`.start()`来启动一个新的 Python 计时器时，您首先检查计时器是否已经运行。然后你将当前值存储在`._start_time`中。

另一方面，当您调用`.stop()`时，您首先检查 Python 计时器是否正在运行。如果是的话，那么你可以计算出`perf_counter()`的当前值和你存储在`._start_time`中的值之间的差值。最后，您重置`._start_time`以便计时器可以重新启动，并打印经过的时间。

下面是使用`Timer`的方法:

>>>

```py
>>> from timer import Timer
>>> t = Timer()
>>> t.start()

>>> t.stop()  # A few seconds later
Elapsed time: 3.8191 seconds
```

将这个与之前的[例子](#your-first-python-timer)进行比较，在那里你直接使用了`perf_counter()`。代码的结构相当相似，但是现在代码更清晰了，这是使用类的好处之一。通过仔细选择您的类、方法和属性名，您可以使您的代码非常具有描述性！

[*Remove ads*](/account/join/)

### 使用 Python 定时器类

现在将`Timer`应用到`latest_tutorial.py`。您只需要对之前的代码做一些修改:

```py
# latest_tutorial.py

from timer import Timer from reader import feed

def main():
    """Print the latest tutorial from Real Python"""
 t = Timer() t.start()    tutorial = feed.get_article(0)
 t.stop() 
    print(tutorial)

if __name__ == "__main__":
    main()
```

请注意，该代码与您之前使用的代码非常相似。除了使代码更具可读性之外，`Timer`还负责将经过的时间打印到控制台，这使得记录花费的时间更加一致。当您运行代码时，您将得到几乎相同的输出:

```py
$ python latest_tutorial.py
Elapsed time: 0.6462 seconds # Python Timer Functions: Three Ways to Monitor Your Code

[ ... ]
```

打印从`Timer`开始经过的时间可能是一致的，但是这种方式似乎不是很灵活。在下一节中，您将看到如何定制您的类。

### 增加更多便利性和灵活性

到目前为止，您已经了解了当您想要封装状态并确保代码中行为一致时，类是合适的。在本节中，您将为 Python 计时器增加更多的便利性和灵活性:

*   在报告花费的时间时，使用**适应性文本**和格式
*   将**灵活的日志**应用到屏幕、日志文件或程序的其他部分
*   创建一个 Python 计时器，它可以在几次调用中累计 T1
*   构建一个 Python 定时器的**信息表示**

首先，看看如何定制用于报告花费时间的文本。在前面的代码中，文本`f"Elapsed time: {elapsed_time:0.4f} seconds"`被硬编码为`.stop()`。您可以使用**实例变量**为类增加灵活性，实例变量的值通常作为参数传递给`.__init__()`，并存储为`self`属性。为了方便起见，您还可以提供合理的默认值。

要添加`.text`作为一个`Timer`实例变量，您可以在`timer.py`中这样做:

```py
# timer.py

def __init__(self, text="Elapsed time: {:0.4f} seconds"):
    self._start_time = None
 self.text = text
```

请注意，默认文本`"Elapsed time: {:0.4f} seconds"`是作为常规字符串给出的，而不是 f 字符串。你不能在这里使用 f-string，因为 f-string 立即计算，当你实例化`Timer`时，你的代码还没有计算运行时间。

**注意:**如果你想用一个 f-string 来指定`.text`，那么你需要用双花括号来转义实际运行时间将替换的花括号。

一个例子就是`f"Finished {task} in {{:0.4f}} seconds"`。如果`task`的值是`"reading"`，那么这个 f 字符串将被评估为`"Finished reading in {:0.4f} seconds"`。

在`.stop()`中，您使用`.text`作为模板，使用`.format()`来填充模板:

```py
# timer.py

def stop(self):
    """Stop the timer, and report the elapsed time"""
    if self._start_time is None:
        raise TimerError(f"Timer is not running. Use .start() to start it")

    elapsed_time = time.perf_counter() - self._start_time
    self._start_time = None
 print(self.text.format(elapsed_time))
```

更新到`timer.py`后，您可以按如下方式更改文本:

>>>

```py
>>> from timer import Timer
>>> t = Timer(text="You waited {:.1f} seconds")
>>> t.start()

>>> t.stop()  # A few seconds later
You waited 4.1 seconds
```

接下来，假设您不只是想在控制台上打印一条消息。也许您想保存您的时间测量值，以便可以将它们存储在数据库中。您可以通过从`.stop()`返回`elapsed_time`的值来实现这一点。然后，调用代码可以选择忽略该返回值或保存它供以后处理。

也许你想将`Timer`集成到你的[日志程序](https://realpython.com/python-logging/)中。为了支持来自`Timer`的日志或其他输出，您需要更改对`print()`的调用，以便用户可以提供他们自己的日志功能。这可以类似于您之前定制文本的方式来完成:

```py
 1# timer.py
 2
 3# ...
 4
 5class Timer:
 6    def __init__(
 7        self,
 8        text="Elapsed time: {:0.4f} seconds",
 9        logger=print 10    ):
11        self._start_time = None
12        self.text = text
13        self.logger = logger 14
15    # Other methods are unchanged
16
17    def stop(self):
18        """Stop the timer, and report the elapsed time"""
19        if self._start_time is None:
20            raise TimerError(f"Timer is not running. Use .start() to start it")
21
22        elapsed_time = time.perf_counter() - self._start_time
23        self._start_time = None
24
25        if self.logger: 26            self.logger(self.text.format(elapsed_time)) 27
28        return elapsed_time
```

您没有直接使用`print()`，而是在第 13 行创建了另一个实例变量`self.logger`，它应该引用一个以字符串作为参数的函数。除了`print()`，你还可以在[文件对象](https://docs.python.org/3/glossary.html#term-file-object)上使用类似 [`logging.info()`](https://docs.python.org/3/library/logging.html#logging.info) 或者`.write()`的函数。还要注意第 25 行的`if`测试，它允许您通过`logger=None`完全关闭打印。

下面是两个展示新功能的例子:

>>>

```py
>>> from timer import Timer
>>> import logging
>>> t = Timer(logger=logging.warning)
>>> t.start()

>>> t.stop()  # A few seconds later
WARNING:root:Elapsed time: 3.1610 seconds
3.1609658249999484

>>> t = Timer(logger=None)
>>> t.start()

>>> value = t.stop()  # A few seconds later
>>> value
4.710851433001153
```

当您在交互式 shell 中运行这些示例时，Python 会自动打印返回值。

您将添加的第三个改进是累积**时间测量值**的能力。例如，当你在一个循环中调用一个慢速函数时，你可能想这样做。您将使用一个[字典](https://realpython.com/python-dicts/)以命名计时器的形式添加更多的功能，该字典跟踪您代码中的每个 Python 计时器。

假设您正在将`latest_tutorial.py`扩展为一个`latest_tutorials.py`脚本，该脚本下载并打印来自 Real Python 的十个最新教程。以下是一种可能的实现方式:

```py
# latest_tutorials.py

from timer import Timer
from reader import feed

def main():
    """Print the 10 latest tutorials from Real Python"""
    t = Timer(text="Downloaded 10 tutorials in {:0.2f} seconds")
    t.start()
    for tutorial_num in range(10):
        tutorial = feed.get_article(tutorial_num)
        print(tutorial)
    t.stop()

if __name__ == "__main__":
    main()
```

代码循环遍历从 0 到 9 的数字，并将它们用作`feed.get_article()`的偏移参数。当您运行该脚本时，您会将大量信息打印到您的控制台:

```py
$ python latest_tutorials.py
# Python Timer Functions: Three Ways to Monitor Your Code

[ ... The text of the tutorials ... ]
Downloaded 10 tutorials in 0.67 seconds
```

这段代码的一个微妙问题是，您不仅要测量下载教程所花费的时间，还要测量 Python 将教程打印到屏幕上所花费的时间。这可能没那么重要，因为与下载时间相比，打印时间可以忽略不计。尽管如此，在这种情况下，有一种方法可以精确地确定你所追求的是什么，这将是一件好事。

**注意:**下载十个教程所花的时间和下载一个教程所花的时间差不多。这不是你代码中的错误！相反，`reader`在第一次调用`get_article()`时缓存真正的 Python 提要，并在以后的调用中重用这些信息。

有几种方法可以在不改变当前`Timer.`实现的情况下解决这个问题。然而，支持这个用例将会非常有用，你只需要几行代码就可以做到。

首先，您将引入一个名为`.timers`的字典作为`Timer`上的**类变量**，这意味着`Timer`的所有实例将共享它。您可以通过在任何方法之外定义它来实现它:

```py
class Timer:
    timers = {}
```

可以直接在类上或通过类的实例来访问类变量:

>>>

```py
>>> from timer import Timer
>>> Timer.timers
{}

>>> t = Timer()
>>> t.timers
{}

>>> Timer.timers is t.timers
True
```

在这两种情况下，代码都返回相同的空类字典。

接下来，您将向 Python 计时器添加可选名称。您可以将该名称用于两个不同的目的:

1.  **在代码中查找**经过的时间
2.  **累积同名的**个计时器

要向 Python 计时器添加名称，需要对`timer.py`再做两处修改。首先，`Timer`应该接受`name`作为参数。第二，当定时器停止时，经过的时间应该加到`.timers`:

```py
 1# timer.py
 2
 3# ...
 4
 5class Timer:
 6    timers = {} 7
 8    def __init__(
 9        self,
10        name=None, 11        text="Elapsed time: {:0.4f} seconds",
12        logger=print,
13    ):
14        self._start_time = None
15        self.name = name 16        self.text = text
17        self.logger = logger
18
19        # Add new named timers to dictionary of timers 20        if name: 21            self.timers.setdefault(name, 0) 22
23    # Other methods are unchanged
24
25    def stop(self):
26        """Stop the timer, and report the elapsed time"""
27        if self._start_time is None:
28            raise TimerError(f"Timer is not running. Use .start() to start it")
29
30        elapsed_time = time.perf_counter() - self._start_time
31        self._start_time = None
32
33        if self.logger:
34            self.logger(self.text.format(elapsed_time))
35        if self.name: 36            self.timers[self.name] += elapsed_time 37
38        return elapsed_time
```

注意，在向`.timers`添加新的 Python 定时器时，使用了`.setdefault()`。这是一个很棒的[特性](https://realpython.com/python-coding-interview-tips/#define-default-values-in-dictionaries-with-get-and-setdefault)，它只在`name`还没有在字典中定义的情况下设置值。如果`name`已经在`.timers`中使用，则该值保持不变。这允许您累积几个计时器:

>>>

```py
>>> from timer import Timer
>>> t = Timer("accumulate")
>>> t.start()

>>> t.stop()  # A few seconds later
Elapsed time: 3.7036 seconds
3.703554293999332

>>> t.start()

>>> t.stop()  # A few seconds later
Elapsed time: 2.3449 seconds
2.3448921170001995

>>> Timer.timers
{'accumulate': 6.0484464109995315}
```

您现在可以重新访问`latest_tutorials.py`，并确保只计算下载教程所花费的时间:

```py
# latest_tutorials.py

from timer import Timer
from reader import feed

def main():
    """Print the 10 latest tutorials from Real Python"""
 t = Timer("download", logger=None)    for tutorial_num in range(10):
 t.start()        tutorial = feed.get_article(tutorial_num)
 t.stop()        print(tutorial)

 download_time = Timer.timers["download"] print(f"Downloaded 10 tutorials in {download_time:0.2f} seconds") 
if __name__ == "__main__":
    main()
```

重新运行该脚本将给出与前面类似的输出，尽管现在您只是对教程的实际下载进行计时:

```py
$ python latest_tutorials.py
# Python Timer Functions: Three Ways to Monitor Your Code

[ ... The text of the tutorials ... ]
Downloaded 10 tutorials in 0.65 seconds
```

你将对`Timer`做的最后一个改进是，当你交互地使用它时，它会提供更多的信息。尝试以下方法:

>>>

```py
>>> from timer import Timer
>>> t = Timer()
>>> t
<timer.Timer object at 0x7f0578804320>
```

最后一行是 Python 表示对象的默认方式。虽然您可以从中收集一些信息，但通常不是很有用。相反，最好能看到类似于`Timer`的名字，或者它将如何报告时间的信息。

在 Python 3.7 中，[数据类](https://realpython.com/python-data-classes/)被添加到标准库中。这些为您的类提供了一些便利，包括更丰富的表示字符串。

**注意:**数据类仅包含在 Python 3.7 及更高版本中。然而，Python 3.6 的 PyPI 上有一个[反向端口](https://pypi.org/project/dataclasses/)。

您可以使用`pip`来安装它:

```py
$ python -m pip install dataclasses
```

更多信息请参见 Python 3.7+(指南)中的[数据类。](https://realpython.com/python-data-classes/)

使用`@dataclass`装饰器将 Python 定时器转换成数据类。在本教程的后面，你会学到更多关于装饰师[的知识。现在，你可以把这看作是告诉 Python`Timer`是一个数据类的符号:](#a-python-timer-decorator)

```py
 1# timer.py
 2
 3import time
 4from dataclasses import dataclass, field
 5from typing import Any, ClassVar
 6
 7# ...
 8
 9@dataclass
10class Timer:
11    timers: ClassVar = {}
12    name: Any = None
13    text: Any = "Elapsed time: {:0.4f} seconds"
14    logger: Any = print
15    _start_time: Any = field(default=None, init=False, repr=False)
16
17    def __post_init__(self):
18        """Initialization: add timer to dict of timers"""
19        if self.name:
20            self.timers.setdefault(self.name, 0)
21
22    # The rest of the code is unchanged
```

这段代码取代了您之前的`.__init__()`方法。请注意数据类如何使用看起来类似于您在前面看到的用于定义所有变量的类变量语法的语法。事实上，`.__init__()`是根据类定义中的[注释变量](https://realpython.com/python-type-checking/#variable-annotations)自动为数据类创建的。

要使用数据类，您需要对变量进行注释。您可以使用该注释将[类型提示](https://realpython.com/python-type-checking/)添加到代码中。如果你不想使用类型提示，那么你可以用`Any`来注释所有的变量，就像上面所做的一样。您将很快学会如何向数据类添加实际的类型提示。

以下是关于`Timer`数据类的一些注意事项:

*   **第 9 行:**`@dataclass`装饰器将`Timer`定义为一个数据类。

*   **第 11 行:**数据类需要特殊的`ClassVar`注释来指定`.timers`是一个类变量。

*   **第 12 到 14 行:** `.name`、`.text`、`.logger`将被定义为`Timer`上的属性，其值可以在创建`Timer`实例时指定。它们都有给定的默认值。

*   **第 15 行:**回想一下`._start_time`是一个特殊的属性，用于跟踪 Python 定时器的状态，但是它应该对用户隐藏。利用`dataclasses.field()`，你说`._start_time`应该从`.__init__()`和`Timer`的表象中去掉。

*   **第 17 到 20 行:**除了设置实例属性，您还可以使用特殊的`.__post_init__()`方法进行任何需要的初始化。在这里，您使用它将命名计时器添加到`.timers`。

新的`Timer`数据类的工作方式与之前的常规类一样，只是它现在有了一个很好的表示:

>>>

```py
>>> from timer import Timer
>>> t = Timer()
>>> t
Timer(name=None, text='Elapsed time: {:0.4f} seconds',
 logger=<built-in function print>)

>>> t.start()

>>> t.stop()  # A few seconds later
Elapsed time: 6.7197 seconds
6.719705373998295
```

现在你有了一个非常简洁的版本`Timer`,它是一致的、灵活的、方便的、信息丰富的！您也可以将本节中所做的许多改进应用到项目中的其他类型的类中。

在结束这一部分之前，重新看看目前的完整源代码。您会注意到在代码中添加了[类型提示](https://realpython.com/python-type-checking/)，以获得额外的文档:

```py
# timer.py

from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer:
    timers: ClassVar[Dict[str, float]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time
```

使用类创建 Python 计时器有几个好处:

*   可读性:如果你仔细选择类名和方法名，你的代码读起来会更自然。
*   一致性:如果你将属性和行为封装到属性和方法中，你的代码会更容易使用。
*   **灵活性:**如果您使用带有默认值的属性，而不是硬编码的值，您的代码将是可重用的。

这个类非常灵活，您几乎可以在任何想要监控代码运行时间的情况下使用它。然而，在接下来的部分中，您将学习如何使用上下文管理器和装饰器，这对于定时代码块和函数来说更加方便。

[*Remove ads*](/account/join/)

## Python 定时器上下文管理器

您的 Python `Timer`类已经取得了很大的进步！与你创建的第一个[Python 定时器](#your-first-python-timer)相比，你的代码已经变得相当强大了。然而，仍然有一些样板代码是使用您的`Timer`所必需的:

1.  首先，实例化该类。
2.  在你想要计时的代码块之前调用`.start()`。
3.  代码块后调用`.stop()`。

幸运的是，Python 有一个在代码块前后调用函数的独特构造:上下文管理器**。在本节中，您将了解什么是[上下文管理器和 Python 的`with`语句](https://realpython.com/python-with-statement/)，以及如何创建自己的上下文管理器。然后您将扩展`Timer`,这样它也可以作为上下文管理器工作。最后，您将看到使用`Timer`作为上下文管理器如何简化您的代码。*

*### 理解 Python 中的上下文管理器

上下文管理器成为 Python 的一部分已经有很长时间了。它们是由 PEP 343 在 2005 年提出的，并在 Python 2.5 中首次实现。您可以通过使用 **`with`** [关键字](https://realpython.com/python-keywords/)来识别代码中的上下文管理器:

```py
with EXPRESSION as VARIABLE:
    BLOCK
```

`EXPRESSION`是返回上下文管理器的 Python 表达式。上下文管理器可选地绑定到名称`VARIABLE`。最后，`BLOCK`是任何常规的 Python 代码块。上下文管理器将保证你的程序在`BLOCK`之前调用一些代码，在`BLOCK`执行之后调用另一些代码。后者会发生，即使`BLOCK`引发异常。

上下文管理器最常见的用途可能是处理不同的资源，比如文件、锁和数据库连接。在您使用完资源后，上下文管理器将用于释放和清理资源。下面的例子通过打印包含冒号的行揭示了`timer.py`的基本结构。更重要的是，它展示了在 Python 中[打开文件的通用习语:](https://realpython.com/working-with-files-in-python/)

>>>

```py
>>> with open("timer.py") as fp:
...     print("".join(ln for ln in fp if ":" in ln))
...
class TimerError(Exception):
class Timer:
 timers: ClassVar[Dict[str, float]] = {}
 name: Optional[str] = None
 text: str = "Elapsed time: {:0.4f} seconds"
 logger: Optional[Callable[[str], None]] = print
 _start_time: Optional[float] = field(default=None, init=False, repr=False)
 def __post_init__(self) -> None:
 if self.name is not None:
 def start(self) -> None:
 if self._start_time is not None:
 def stop(self) -> float:
 if self._start_time is None:
 if self.logger:
 if self.name:
```

请注意，文件指针`fp`从未被显式关闭，因为您使用了`open()`作为上下文管理器。您可以确认`fp`已经自动关闭:

>>>

```py
>>> fp.closed
True
```

在本例中，`open("timer.py")`是一个返回上下文管理器的表达式。该上下文管理器被绑定到名称`fp`。上下文管理器在`print()`执行期间有效。这一行代码块在`fp`的上下文中执行。

`fp`是上下文管理器是什么意思？从技术上讲，这意味着`fp`实现了**上下文管理器协议**。Python 语言下有许多不同的[协议](https://www.python.org/dev/peps/pep-0544/)。您可以将协议视为一个契约，它规定了您的代码必须实现哪些特定的方法。

[上下文管理器协议](https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers)由两种方法组成:

1.  **进入与上下文管理器相关的上下文时，调用`.__enter__()`** 。
2.  **退出与上下文管理器相关的上下文时，调用`.__exit__()`** 。

换句话说，要自己创建一个上下文管理器，需要编写一个实现`.__enter__()`和`.__exit__()`的类。不多不少。试试*你好，世界！*上下文管理器示例:

```py
# greeter.py

class Greeter:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(f"Hello {self.name}")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"See you later, {self.name}")
```

`Greeter`是上下文管理器，因为它实现了上下文管理器协议。你可以这样使用它:

>>>

```py
>>> from greeter import Greeter
>>> with Greeter("Akshay"):
...     print("Doing stuff ...")
...
Hello Akshay
Doing stuff ...
See you later, Akshay
```

首先，注意如何在你做事情之前调用`.__enter__()`,而在之后调用`.__exit__()`。在这个简化的例子中，您没有引用上下文管理器。在这种情况下，您不需要给上下文管理器起一个带有`as`的名字。

接下来，注意`.__enter__()`如何返回`self`。`.__enter__()`的返回值被`as`绑定。在创建上下文管理器时，通常希望从`.__enter__()`返回`self`。您可以按如下方式使用返回值:

>>>

```py
>>> from greeter import Greeter
>>> with Greeter("Bethan") as grt:
...     print(f"{grt.name} is doing stuff ...")
...
Hello Bethan
Bethan is doing stuff ...
See you later, Bethan
```

最后，`.__exit__()`带三个参数:`exc_type`、`exc_value`和`exc_tb`。这些用于上下文管理器中的错误处理，它们反映了`sys.exc_info()` 的[返回值。](https://docs.python.org/3/library/sys.html#sys.exc_info)

如果在执行代码块时发生了异常，那么您的代码会调用带有异常类型、异常实例和[回溯](https://realpython.com/python-traceback/)对象的`.__exit__()`。通常，您可以在上下文管理器中忽略这些，在这种情况下，会在重新引发异常之前调用`.__exit__()`:

>>>

```py
>>> from greeter import Greeter
>>> with Greeter("Rascal") as grt:
...     print(f"{grt.age} does not exist")
...
Hello Rascal
See you later, Rascal Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
AttributeError: 'Greeter' object has no attribute 'age'
```

您可以看到`"See you later, Rascal"`被打印出来，尽管代码中有一个错误。

现在您知道了什么是上下文管理器，以及如何创建自己的上下文管理器。如果你想更深入，那么在标准库中查看 [`contextlib`](https://docs.python.org/3/library/contextlib.html) 。它包括定义新的上下文管理器的方便方法，以及现成的上下文管理器，您可以使用它们来[关闭对象](https://docs.python.org/3/library/contextlib.html#contextlib.closing)、[抑制错误](https://docs.python.org/3/library/contextlib.html#contextlib.suppress)，甚至[什么都不做](https://docs.python.org/3/library/contextlib.html#contextlib.nullcontext)！更多信息，请查看[上下文管理器和 Python 的`with`语句](https://realpython.com/python-with-statement/)。

[*Remove ads*](/account/join/)

### 创建 Python 定时器上下文管理器

您已经看到了上下文管理器一般是如何工作的，但是它们如何帮助处理计时代码呢？如果您可以在代码块之前和之后运行某些函数，那么您就可以简化 Python 计时器的工作方式。到目前为止，在为代码计时时，您需要显式调用`.start()`和`.stop()`，但是上下文管理器可以自动完成这项工作。

同样，对于作为上下文管理器工作的`Timer`,它需要遵守上下文管理器协议。换句话说，它必须实现`.__enter__()`和`.__exit__()`来启动和停止 Python 定时器。所有必要的功能都已经可用，所以不需要编写太多新代码。只需将以下方法添加到您的`Timer`类中:

```py
# timer.py

# ...

@dataclass
class Timer:
    # The rest of the code is unchanged

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()
```

`Timer`现在是上下文管理器。实现的重要部分是当进入上下文时，`.__enter__()`调用`.start()`启动 Python 定时器，当代码离开上下文时，`.__exit__()`使用`.stop()`停止 Python 定时器。尝试一下:

>>>

```py
>>> from timer import Timer
>>> import time
>>> with Timer():
...     time.sleep(0.7)
...
Elapsed time: 0.7012 seconds
```

您还应该注意两个更微妙的细节:

1.  **`.__enter__()`** 返回`self`，`Timer`实例，允许用户使用`as`将`Timer`实例绑定到变量。例如，`with Timer() as t:`将创建指向`Timer`对象的变量`t`。

2.  **`.__exit__()`** 期望三个参数带有关于在上下文执行期间发生的任何异常的信息。在您的代码中，这些参数被打包到一个名为`exc_info`的元组中，然后被忽略，这意味着`Timer`不会尝试任何异常处理。

`.__exit__()`在这种情况下不做任何错误处理。尽管如此，上下文管理器的一个重要特性是，无论上下文如何退出，它们都保证调用`.__exit__()`。在以下示例中，您通过除以零故意制造了一个误差:

>>>

```py
>>> from timer import Timer
>>> with Timer():
...     for num in range(-3, 3):
...         print(f"1 / {num} = {1 / num:.3f}")
...
1 / -3 = -0.333
1 / -2 = -0.500
1 / -1 = -1.000
Elapsed time: 0.0001 seconds Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
ZeroDivisionError: division by zero
```

请注意，`Timer`打印出运行时间，即使代码崩溃。可以检查和抑制`.__exit__()`中的错误。更多信息参见[文档](https://docs.python.org/3/reference/datamodel.html#object.__exit__)。

### 使用 Python 定时器上下文管理器

现在您将学习如何使用`Timer`上下文管理器来为真正的 Python 教程下载计时。回想一下您之前是如何使用`Timer`的:

```py
# latest_tutorial.py

from timer import Timer
from reader import feed

def main():
    """Print the latest tutorial from Real Python"""
    t = Timer()
    t.start()
    tutorial = feed.get_article(0)
    t.stop()

    print(tutorial)

if __name__ == "__main__":
    main()
```

您正在为呼叫`feed.get_article()`计时。您可以使用上下文管理器使代码更短、更简单、更易读:

```py
# latest_tutorial.py

from timer import Timer
from reader import feed

def main():
    """Print the latest tutorial from Real Python"""
 with Timer():        tutorial = feed.get_article(0)

    print(tutorial)

if __name__ == "__main__":
    main()
```

这段代码实际上和上面的代码做的一样。主要的区别在于，您没有定义无关变量`t`，这使得您的[名称空间](https://realpython.com/python-namespaces-scope/)更加清晰。

运行该脚本应该会得到一个熟悉的结果:

```py
$ python latest_tutorial.py
Elapsed time: 0.71 seconds # Python Timer Functions: Three Ways to Monitor Your Code

[ ... ]
```

将上下文管理器功能添加到 Python 计时器类中有几个好处:

*   **省力:**你只需要一行额外的代码来计时一段代码的执行。
*   **可读性:**调用上下文管理器是可读的，您可以更清楚地可视化您正在计时的代码块。

使用`Timer`作为上下文管理器几乎和直接使用`.start()`和`.stop()`一样灵活，而且样板代码更少。在下一节中，您将学习如何使用`Timer`作为装饰器。这将使监控完整函数的运行时变得更加容易。

[*Remove ads*](/account/join/)

## 一个 Python 定时器装饰器

你的`Timer`课现在很全能。然而，有一个用例您可以进一步简化它。假设您想要跟踪代码库中一个给定函数所花费的时间。使用上下文管理器，您有两种不同的选择:

1.  **每次调用函数时使用`Timer`:**

    ```py
    with Timer("some_name"):
        do_something()` 
    ```

    如果你在很多地方调用`do_something()`，那么这将变得很繁琐，很难维护。

2.  **将函数中的代码包装在上下文管理器中:**

    ```py
    def do_something():
        with Timer("some_name"):
            ...` 
    ```

    只需要在一个地方添加`Timer`，但是这给`do_something()`的整个定义增加了一级缩进。

更好的解决方案是使用`Timer`作为**装饰器**。装饰器是用来修改函数和类的行为的强大构造。在这一节中，您将了解装饰器是如何工作的，如何将`Timer`扩展为装饰器，以及这将如何简化计时功能。关于装饰者的更深入的解释，请参见【Python 装饰者入门[。](https://realpython.com/primer-on-python-decorators/)

### 理解 Python 中的装饰者

一个**装饰器**是一个包装另一个函数来修改其行为的函数。这种技术是可行的，因为函数是 Python 中的[一级对象](https://realpython.com/primer-on-python-decorators/#first-class-objects)。换句话说，函数可以赋给变量，也可以用作其他函数的参数，就像任何其他对象一样。这为您提供了很大的灵活性，并且是 Python 几个最强大特性的基础。

作为第一个例子，您将创建一个什么都不做的装饰器:

```py
def turn_off(func):
    return lambda *args, **kwargs: None
```

首先，注意`turn_off()`只是一个常规函数。使它成为装饰器的是，它将一个函数作为唯一的参数，并返回一个函数。您可以使用`turn_off()`来修改其他功能，如下所示:

>>>

```py
>>> print("Hello")
Hello
  >>> print = turn_off(print)
>>> print("Hush")
>>> # Nothing is printed
```

第`print = turn_off(print)` **行用`turn_off()`修饰符修饰**打印语句。实际上，它用由`turn_off()`返回的`lambda *args, **kwargs: None`代替了`print()`。 [lambda](https://realpython.com/python-lambda/) 语句表示一个除了返回`None`之外什么也不做的匿名函数。

要定义更多有趣的装饰器，你需要了解[内部函数](https://realpython.com/inner-functions-what-are-they-good-for/)。一个**内部函数**是定义在另一个函数内部的函数。内部函数的一个常见用途是创建函数工厂:

```py
def create_multiplier(factor):
    def multiplier(num):
        return factor * num
    return multiplier
```

`multiplier()`是一个内部函数，定义在`create_multiplier()`内部。请注意，您可以访问`multiplier()`内的`factor`，而`create_multiplier()`外的`multiplier()`没有定义:

>>>

```py
>>> multiplier
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'multiplier' is not defined
```

相反，您可以使用`create_multiplier()`来创建新的乘数函数，每个函数都基于不同的因子:

>>>

```py
>>> double = create_multiplier(factor=2)
>>> double(3)
6

>>> quadruple = create_multiplier(factor=4)
>>> quadruple(7)
28
```

同样，您可以使用内部函数来创建装饰器。记住，装饰器是一个返回函数的函数:

```py
 1def triple(func):
 2    def wrapper_triple(*args, **kwargs):
 3        print(f"Tripled {func.__name__!r}")
 4        value = func(*args, **kwargs)
 5        return value * 3
 6    return wrapper_triple
```

`triple()`是一个装饰器，因为它是一个期望函数`func()`作为其唯一参数并返回另一个函数`wrapper_triple()`的函数。注意`triple()`本身的结构:

*   **第 1 行**开始定义`triple()`，并期望一个函数作为参数。
*   **第 2 到 5 行**定义了内部函数`wrapper_triple()`。
*   **第 6 行**返回`wrapper_triple()`。

这种模式普遍用于定义装饰者。有趣的部分发生在内部函数中:

*   **第 2 行**开始定义`wrapper_triple()`。这个函数将取代`triple()`修饰的函数。参数是 [`*args`和`**kwargs`](https://realpython.com/python-kwargs-and-args/) ，它们收集您传递给函数的任何位置和关键字参数。这给了你在任何函数上使用`triple()`的灵活性。
*   **第 3 行**打印出被修饰函数的名称，并注意到`triple()`已经被应用于它。
*   **第 4 行**调用`func()`，`triple()`修饰过的功能。它将传递给`wrapper_triple()`的所有参数。
*   **第 5 行**将`func()`的返回值三倍并返回。

试试吧！`knock()`是返回单词 [`Penny`](https://en.wikipedia.org/wiki/Penny_%28The_Big_Bang_Theory%29) 的函数。看看如果增加三倍会发生什么:

>>>

```py
>>> def knock():
...     return "Penny! "
...
>>> knock = triple(knock)
>>> result = knock()
Tripled 'knock'

>>> result
'Penny! Penny! Penny! '
```

将一个文本字符串乘以一个数字是一种重复形式，所以`Penny`重复三次。装饰发生在`knock = triple(knock)`。

一直重复`knock`感觉有点笨拙。相反， [PEP 318](https://www.python.org/dev/peps/pep-0318/) 引入了一个更方便的语法来应用装饰器。下面的`knock()`定义与上面的定义相同:

>>>

```py
>>> @triple
... def knock():
...     return "Penny! "
...
>>> result = knock()
Tripled 'knock'

>>> result
'Penny! Penny! Penny! '
```

符号用来应用装饰符。在这种情况下，`@triple`意味着`triple()`被应用于紧随其后定义的函数。

标准库中定义的少数装饰者之一是`@functools.wraps`。在定义自己的装饰者时，这一条非常有用。因为装饰者有效地用一个函数替换了另一个函数，所以他们给你的函数制造了一个微妙的问题:

>>>

```py
>>> knock
<function triple.<locals>.wrapper_triple at 0x7fa3bfe5dd90>
```

`@triple`修饰`knock()`，然后被`wrapper_triple()`内部函数替换，正如上面的输出所证实的。这也将替换名称、[文档字符串](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)和其他元数据。通常，这不会有太大的效果，但会使自省变得困难。

有时，修饰函数必须有正确的元数据。`@functools.wraps`解决了这个问题:

```py
import functools 
def triple(func):
 @functools.wraps(func)    def wrapper_triple(*args, **kwargs):
        print(f"Tripled {func.__name__!r}")
        value = func(*args, **kwargs)
        return value * 3
    return wrapper_triple
```

使用这个新的`@triple`定义，元数据被保留:

>>>

```py
>>> @triple
... def knock():
...     return "Penny! "
...
>>> knock
<function knock at 0x7fa3bfe5df28>
```

请注意，`knock()`现在保持其正确的名称，即使在装饰之后。在定义装饰器时使用`@functools.wraps`是一种好的形式。您可以为大多数装饰者使用的蓝图如下:

```py
import functools

def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator
```

要查看更多关于如何定义 decorator 的例子，请查看 Python Decorators 入门教程中列出的例子。

[*Remove ads*](/account/join/)

### 创建 Python 计时器装饰器

在这一节中，您将学习如何扩展您的 Python 计时器，以便您也可以将它用作装饰器。然而，作为第一个练习，您将从头开始创建一个 Python 计时器装饰器。

基于上面的蓝图，您只需要决定在调用修饰函数之前和之后做什么。这类似于在进入和退出上下文管理器时要做什么的考虑。您希望在调用修饰函数之前启动 Python 计时器，并在调用完成后停止 Python 计时器。您可以定义一个 [`@timer`装饰者](https://realpython.com/primer-on-python-decorators/#timing-functions)，如下所示:

```py
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer
```

注意`wrapper_timer()`与您为计时 Python 代码建立的早期模式有多么相似。您可以如下应用`@timer`:

>>>

```py
>>> @timer
... def latest_tutorial():
...     tutorial = feed.get_article(0)
...     print(tutorial)
...
>>> latest_tutorial()
# Python Timer Functions: Three Ways to Monitor Your Code

[ ... ]
Elapsed time: 0.5414 seconds
```

回想一下，您还可以将装饰器应用于先前定义的函数:

>>>

```py
>>> feed.get_article = timer(feed.get_article)
```

因为`@`在定义函数时适用，所以在这些情况下需要使用更基本的形式。使用装饰器的一个好处是你只需要应用一次，它每次都会计算函数的时间:

>>>

```py
>>> tutorial = feed.get_article(0)
Elapsed time: 0.5512 seconds
```

做这项工作。然而，在某种意义上，你又回到了起点，因为`@timer`没有`Timer`的任何灵活性或便利性。你也能让你的`Timer`类表现得像一个装饰者吗？

到目前为止，您已经将 decorators 用作应用于其他函数的函数，但这并不完全正确。装饰者必须是**可召唤者**。Python 中有很多[可调用类型](https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy)。您可以通过在自己的类中定义特殊的`.__call__()`方法来使自己的对象可调用。以下函数和类的行为类似:

>>>

```py
>>> def square(num):
...     return num ** 2
...
>>> square(4)
16

>>> class Squarer:
...     def __call__(self, num):
...         return num ** 2
...
>>> square = Squarer()
>>> square(4)
16
```

这里，`square`是一个可调用的实例，可以平方数字，就像第一个例子中的`square()`函数一样。

这为您提供了一种向现有的`Timer`类添加装饰功能的方法:

```py
# timer.py

import functools

# ...

@dataclass
class Timer:

    # The rest of the code is unchanged

    def __call__(self, func):
        """Support using Timer as a decorator"""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper_timer
```

`.__call__()`利用`Timer`已经是一个上下文管理器的事实来利用您已经在那里定义的便利。确定你也在`timer.py`上方[导入](https://realpython.com/absolute-vs-relative-python-imports/) `functools`。

您现在可以使用`Timer`作为装饰器:

>>>

```py
>>> @Timer(text="Downloaded the tutorial in {:.2f} seconds")
... def latest_tutorial():
...     tutorial = feed.get_article(0)
...     print(tutorial)
...
>>> latest_tutorial()
# Python Timer Functions: Three Ways to Monitor Your Code

[ ... ]
Downloaded the tutorial in 0.72 seconds
```

在结束这一部分之前，要知道有一种更直接的方法可以将 Python 计时器变成装饰器。您已经看到了上下文管理器和装饰器之间的一些相似之处。它们通常都用于在执行给定代码之前和之后做一些事情。

基于这些相似性，标准库中定义了一个 [mixin 类](https://realpython.com/inheritance-composition-python/#mixing-features-with-mixin-classes)，称为 [`ContextDecorator`](https://docs.python.org/3/library/contextlib.html#contextlib.ContextDecorator) 。您可以简单地通过继承`ContextDecorator`来为您的上下文管理器类添加装饰功能:

```py
from contextlib import ContextDecorator

# ...

@dataclass
class Timer(ContextDecorator):
    # Implementation of Timer is unchanged
```

当你以这种方式使用`ContextDecorator`时，没有必要自己实现`.__call__()`，所以你可以安全地从`Timer`类中删除它。

[*Remove ads*](/account/join/)

### 使用 Python 计时器装饰器

接下来，您将最后一次重做`latest_tutorial.py`示例，使用 Python 定时器作为装饰器:

```py
 1# latest_tutorial.py
 2
 3from timer import Timer
 4from reader import feed
 5
 6@Timer() 7def main():
 8    """Print the latest tutorial from Real Python"""
 9    tutorial = feed.get_article(0)
10    print(tutorial)
11
12if __name__ == "__main__":
13    main()
```

如果您将这个实现与没有任何计时的[原始实现](#example-download-tutorials)进行比较，那么您会注意到唯一的区别是第 3 行`Timer`的导入和第 6 行`@Timer()`的应用。使用 decorators 的一个显著优点是它们通常很容易应用，正如你在这里看到的。

然而，装饰器仍然适用于整个函数。这意味着除了下载时间之外，您的代码还考虑了打印教程所需的时间。最后一次运行脚本:

```py
$ python latest_tutorial.py
# Python Timer Functions: Three Ways to Monitor Your Code

[ ... ]
Elapsed time: 0.69 seconds
```

运行时间输出的位置是一个信号，表明您的代码也在考虑打印所花费的时间。正如您在这里看到的，您的代码在教程的之后打印了经过的时间*。*

当您使用`Timer`作为装饰器时，您会看到与使用上下文管理器类似的优势:

*   你只需要一行额外的代码来计时一个函数的执行。
*   **可读性:**当您添加装饰器时，您可以更清楚地注意到您的代码将为函数计时。
*   **一致性:**你只需要在定义函数的时候添加装饰器。每次调用时，您的代码都会持续计时。

然而，装饰器不像上下文管理器那样灵活。您只能将它们应用于完整的功能。可以在已经定义的函数中添加装饰器，但是这有点笨拙，也不太常见。

## Python 定时器代码

您可以展开下面的代码块来查看 Python 计时器的最终源代码:



```py
# timer.py

import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[Dict[str, float]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()
```

GitHub 上的 [`codetiming`库](https://github.com/realpython/codetiming)中也有该代码。

您可以将代码保存到一个名为`timer.py`的文件中，然后导入到您的程序中，这样您就可以自己使用代码了:

>>>

```py
>>> from timer import Timer
```

`Timer`在 [PyPI](https://pypi.org/project/codetiming) 上也有，所以更简单的选择是使用 [`pip`](https://realpython.com/what-is-pip/) 安装它:

```py
$ python -m pip install codetiming
```

注意 PyPI 上的包名是`codetiming`。在安装软件包和导入`Timer`时，您都需要使用这个名称:

>>>

```py
>>> from codetiming import Timer
```

除了名字和[的一些额外特性](https://github.com/realpython/codetiming/releases)，`codetiming.Timer`的工作方式与`timer.Timer`完全一样。总而言之，你可以用三种不同的方式使用`Timer`:

1.  作为**类**:

    ```py
    t = Timer(name="class")
    t.start()
    # Do something
    t.stop()` 
    ```

2.  作为**上下文管理器**:

    ```py
    with Timer(name="context manager"):
        # Do something` 
    ```

3.  作为一名**装饰师**:

    ```py
    @Timer(name="decorator")
    def stuff():
        # Do something` 
    ```

这种 Python 计时器主要用于监控代码在单个关键代码块或函数上花费的时间。在下一节中，如果您想要优化代码，您将得到一个备选方案的快速概述。

[*Remove ads*](/account/join/)

## 其他 Python 定时器函数

使用 Python 为代码计时有很多选择。在本教程中，您已经学习了如何创建一个灵活方便的类，您可以用几种不同的方式来使用它。在 PyPI 上快速[搜索](https://pypi.org/search/?q=timer)显示已经有许多项目提供 Python 定时器解决方案。

在本节中，您将首先了解标准库中用于测量时间的不同函数，包括为什么`perf_counter()`更好。然后，您将探索优化代码的替代方法，而`Timer`并不适合。

### 使用替代的 Python 定时器函数

在本教程中，您一直在使用`perf_counter()`来进行实际的时间测量，但是 Python 的`time`库附带了其他几个也可以测量时间的函数。以下是一些备选方案:

*   [T2`time()`](https://docs.python.org/3/library/time.html#time.time)
*   [T2`perf_counter_ns()`](https://docs.python.org/3/library/time.html#time.perf_counter_ns)
*   [T2`monotonic()`](https://docs.python.org/3/library/time.html#time.monotonic)
*   [T2`process_time()`](https://docs.python.org/3/library/time.html#time.process_time)

拥有多个函数的一个原因是 Python 将时间表示为一个`float`。浮点数本质上是不准确的。您可能以前见过这样的结果:

>>>

```py
>>> 0.1 + 0.1 + 0.1
0.30000000000000004

>>> 0.1 + 0.1 + 0.1 == 0.3
False
```

Python 的`float`遵循浮点运算的 IEEE 754 标准，试图用 64 位表示所有浮点数。因为浮点数有无限多种，你不可能用有限的位数全部表示出来。

IEEE 754 规定了一个系统，在这个系统中，您可以表示的数字密度是变化的。你越接近一，你能代表的数字就越多。对于更大的数字，你可以表达的数字之间有更多的**空间**。当你用一个`float`来表示时间时，这会产生一些后果。

考虑一下`time()`。这个函数的主要目的是表示现在的实际时间。这是从给定时间点开始的秒数，称为[时期](https://realpython.com/python-time-module/#the-epoch)。`time()`返回的数字相当大，这意味着可用的数字较少，分辨率受到影响。具体来说，`time()`无法测量**纳秒**的差异:

>>>

```py
>>> import time
>>> t = time.time()
>>> t
1564342757.0654016

>>> t + 1e-9
1564342757.0654016

>>> t == t + 1e-9
True
```

一纳秒是十亿分之一秒。注意，给`t`加一纳秒不会影响结果。另一方面，`perf_counter()`使用某个未定义的时间点作为其历元，允许其使用较小的数字，从而获得更好的分辨率:

>>>

```py
>>> import time
>>> p = time.perf_counter()
>>> p
11370.015653846

>>> p + 1e-9
11370.015653847

>>> p == p + 1e-9
False
```

这里，您会注意到在`p`上增加一纳秒实际上会影响结果。有关如何使用`time()`的更多信息，请参见[Python 时间模块](https://realpython.com/python-time-module/)的初学者指南。

用`float`表示时间的挑战是众所周知的，所以 Python 3.7 引入了一个新的选项。每个`time`测量函数现在都有一个相应的`_ns`函数，它返回纳秒数作为`int`，而不是秒数作为`float`。例如，`time()`现在有了一个纳秒级的对应物，叫做`time_ns()`:

>>>

```py
>>> import time
>>> time.time_ns()
1564342792866601283
```

在 Python 中整数是无限的，所以这允许`time_ns()`永远给出纳秒级的分辨率。类似地，`perf_counter_ns()`是`perf_counter()`的纳秒变体:

>>>

```py
>>> import time
>>> time.perf_counter()
13580.153084446

>>> time.perf_counter_ns()
13580765666638
```

因为`perf_counter()`已经提供了纳秒级的分辨率，所以使用`perf_counter_ns()`的优势更少。

**注意:** `perf_counter_ns()`仅在 Python 3.7 及更高版本中可用。在本教程中，你已经在你的`Timer`类中使用了`perf_counter()`。这样，您也可以在旧版本的 Python 上使用`Timer`。

有关`time`中`_ns`函数的更多信息，请查看 Python 3.7 中的[新功能。](https://realpython.com/python37-new-features/#timing-precision)

`time`中有两个函数不测量睡眠花费的时间。这些是`process_time()`和`thread_time()`，在某些设置中很有用。然而，对于`Timer`，您通常想要测量花费的全部时间。上面列表中的最后一个函数是`monotonic()`。这个名字暗示这个函数是一个单调计时器，是一个永远不能向后移动的 Python 计时器。

所有这些功能都是单调的，只有`time()`除外，如果调整系统时间，它可以倒退。在某些系统上，`monotonic()`与`perf_counter()`的功能相同，可以互换使用。然而，情况并非总是如此。您可以使用`time.get_clock_info()`获得关于 Python 定时器函数的更多信息:

>>>

```py
>>> import time
>>> time.get_clock_info("monotonic")
namespace(adjustable=False, implementation='clock_gettime(CLOCK_MONOTONIC)',
 monotonic=True, resolution=1e-09)

>>> time.get_clock_info("perf_counter")
namespace(adjustable=False, implementation='clock_gettime(CLOCK_MONOTONIC)',
 monotonic=True, resolution=1e-09)
```

在您的系统上，结果可能有所不同。

PEP 418 描述了引入这些功能背后的一些基本原理。它包括以下简短描述:

> *   `time.monotonic()`: Timeout and scheduling, not affected by system clock update.
> *   `time.perf_counter()`: benchmark test, the most accurate short-period clock.
> *   `time.process_time()`: profiling, CPU time of the process ( [source](https://www.python.org/dev/peps/pep-0418/#rationale) )

可以看出，`perf_counter()`通常是 Python 计时器的最佳选择。

[*Remove ads*](/account/join/)

### 用`timeit` 估算运行时间

假设您试图从代码中挤出最后一点性能，并且您想知道将列表转换为集合的最有效方法。您希望使用`set()`和设置的文字`{...}`进行比较。为此，您可以使用 Python 计时器:

>>>

```py
>>> from timer import Timer
>>> numbers = [7, 6, 1, 4, 1, 8, 0, 6]
>>> with Timer(text="{:.8f}"):
...     set(numbers)
...
{0, 1, 4, 6, 7, 8}
0.00007373 
>>> with Timer(text="{:.8f}"):
...     {*numbers}
...
{0, 1, 4, 6, 7, 8}
0.00006204
```

这个测试似乎表明 set literal 可能会稍微快一些。然而，这些结果是相当不确定的，如果您重新运行代码，您可能会得到非常不同的结果。那是因为你只试了一次代码。例如，您可能很不走运，在您的计算机正忙于其他任务时运行该脚本。

更好的方法是使用 [`timeit`](https://docs.python.org/3/library/timeit.html) 标准库。它旨在精确测量小代码片段的执行时间。虽然您可以作为常规函数从 Python 导入和调用`timeit.timeit()`，但是使用[命令行接口](https://realpython.com/python-command-line-arguments/)通常更方便。您可以对两个变量计时，如下所示:

```py
$ python -m timeit --setup "nums = [7, 6, 1, 4, 1, 8, 0, 6]" "set(nums)"
2000000 loops, best of 5: 163 nsec per loop

$ python -m timeit --setup "nums = [7, 6, 1, 4, 1, 8, 0, 6]" "{*nums}"
2000000 loops, best of 5: 121 nsec per loop
```

`timeit`多次自动调用您的代码，以消除噪声测量。来自`timeit`的结果证实了 set literal 比`set()`快。

**注意:**在可以下载文件或访问数据库的代码上使用`timeit`时要小心。由于`timeit`会自动调用你的程序几次，你可能会无意中向服务器发送垃圾请求！

最后， [IPython 交互外壳](https://ipython.org/)和 [Jupyter 笔记本](https://realpython.com/jupyter-notebook-introduction/)通过`%timeit`魔法命令对此功能提供了额外的支持:

>>>

```py
In [1]: numbers = [7, 6, 1, 4, 1, 8, 0, 6]

In [2]: %timeit set(numbers)
171 ns ± 0.748 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

In [3]: %timeit {*numbers}
147 ns ± 2.62 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```

同样，测量表明使用 set 文字更快。在 Jupyter 笔记本中，您还可以使用`%%timeit` cell-magic 来测量运行整个电池的时间。

### 用评测器寻找代码中的瓶颈

`timeit`非常适合对特定的代码片段进行基准测试。然而，用它来检查程序的所有部分并定位哪个部分花费的时间最多是非常麻烦的。相反，你可以使用一个**分析器**。

[`cProfile`](https://docs.python.org/3/library/profile.html) 是一个可以随时从标准库中访问的剖析器。您可以通过多种方式使用它，尽管通常最直接的方式是将其用作命令行工具:

```py
$ python -m cProfile -o latest_tutorial.prof latest_tutorial.py
```

该命令在剖析打开的情况下运行`latest_tutorial.py`。按照`-o`选项的指定，将`cProfile`的输出保存在`latest_tutorial.prof`中。输出数据是二进制格式，需要专门的程序来理解它。同样，Python 在标准库中有一个选项！在您的`.prof`文件上运行 [`pstats`](https://docs.python.org/3/library/profile.html#pstats.Stats) 模块会打开一个交互式概要统计浏览器:

```py
$ python -m pstats latest_tutorial.prof
Welcome to the profile statistics browser.
latest_tutorial.prof% help 
Documented commands (type help <topic>):
========================================
EOF  add  callees  callers  help  quit  read  reverse  sort  stats  strip
```

要使用`pstats`，您可以在提示符下键入命令。这里您可以看到集成的帮助系统。通常你会使用`sort`和`stats`命令。要获得更清晰的输出，`strip`可能很有用:

```py
latest_tutorial.prof% strip latest_tutorial.prof% sort cumtime latest_tutorial.prof% stats 10
 1393801 function calls (1389027 primitive calls) in 0.586 seconds

 Ordered by: cumulative time
 List reduced from 1443 to 10 due to restriction <10>

 ncalls tottime percall cumtime percall filename:lineno(function)
 144/1   0.001   0.000   0.586   0.586 {built-in method builtins.exec}
 1   0.000   0.000   0.586   0.586 latest_tutorial.py:3(<module>)
 1   0.000   0.000   0.521   0.521 contextlib.py:71(inner)
 1   0.000   0.000   0.521   0.521 latest_tutorial.py:6(read_latest_tutorial)
 1   0.000   0.000   0.521   0.521 feed.py:28(get_article)
 1   0.000   0.000   0.469   0.469 feed.py:15(_feed)
 1   0.000   0.000   0.469   0.469 feedparser.py:3817(parse)
 1   0.000   0.000   0.271   0.271 expatreader.py:103(parse)
 1   0.000   0.000   0.271   0.271 xmlreader.py:115(parse)
 13   0.000   0.000   0.270   0.021 expatreader.py:206(feed)
```

该输出显示总运行时间为 0.586 秒。它还列出了代码花费时间最多的十个函数。这里你已经按累计时间(`cumtime`)排序，这意味着当给定的函数调用另一个函数时，你的代码计算时间。

您可以看到您的代码几乎所有的时间都花在了`latest_tutorial`模块中，尤其是在`read_latest_tutorial()`中。虽然这可能是对您已经知道的东西的有用确认，但是发现您的代码实际花费时间的地方通常更有意思。

总时间(`tottime`)列表示代码在一个函数中花费的时间，不包括在子函数中的时间。您可以看到，上面的函数都没有真正花时间来做这件事。为了找到代码花费时间最多的地方，发出另一个`sort`命令:

```py
latest_tutorial.prof% sort tottime latest_tutorial.prof% stats 10
 1393801 function calls (1389027 primitive calls) in 0.586 seconds

 Ordered by: internal time
 List reduced from 1443 to 10 due to restriction <10>

 ncalls tottime percall cumtime percall filename:lineno(function)
 59   0.091   0.002   0.091   0.002 {method 'read' of '_ssl._SSLSocket'}
 114215   0.070   0.000   0.099   0.000 feedparser.py:308(__getitem__)
 113341   0.046   0.000   0.173   0.000 feedparser.py:756(handle_data)
 1   0.033   0.033   0.033   0.033 {method 'do_handshake' of '_ssl._SSLSocket'}
 1   0.029   0.029   0.029   0.029 {method 'connect' of '_socket.socket'}
 13   0.026   0.002   0.270   0.021 {method 'Parse' of 'pyexpat.xmlparser'}
 113806   0.024   0.000   0.123   0.000 feedparser.py:373(get)
 3455   0.023   0.000   0.024   0.000 {method 'sub' of 're.Pattern'}
 113341   0.019   0.000   0.193   0.000 feedparser.py:2033(characters)
 236   0.017   0.000   0.017   0.000 {method 'translate' of 'str'}
```

你现在可以看到，`latest_tutorial.py`实际上大部分时间都在使用套接字或者处理 [`feedparser`](https://pypi.org/project/feedparser/) 内部的数据。后者是用于解析教程提要的真正 Python 阅读器的依赖项之一。

您可以使用`pstats`来了解您的代码在哪里花费了大部分时间，然后尝试优化您发现的任何**瓶颈**。您还可以使用该工具来更好地理解代码的结构。例如，命令`callees`和`callers`将显示给定的函数调用了哪些函数，以及哪些函数被调用了。

您还可以研究某些功能。通过使用短语`timer`过滤结果，检查`Timer`导致了多少开销:

```py
latest_tutorial.prof% stats timer
 1393801 function calls (1389027 primitive calls) in 0.586 seconds

 Ordered by: internal time
 List reduced from 1443 to 8 due to restriction <'timer'>

 ncalls tottime percall cumtime percall filename:lineno(function)
 1   0.000   0.000   0.000   0.000 timer.py:13(Timer)
 1   0.000   0.000   0.000   0.000 timer.py:35(stop)
 1   0.000   0.000   0.003   0.003 timer.py:3(<module>)
 1   0.000   0.000   0.000   0.000 timer.py:28(start)
 1   0.000   0.000   0.000   0.000 timer.py:9(TimerError)
 1   0.000   0.000   0.000   0.000 timer.py:23(__post_init__)
 1   0.000   0.000   0.000   0.000 timer.py:57(__exit__)
 1   0.000   0.000   0.000   0.000 timer.py:52(__enter__)
```

幸运的是，`Timer`只会产生最小的开销。完成调查后，使用`quit`离开`pstats`浏览器。

对于一个更强大的界面到配置文件数据，检查出 [KCacheGrind](https://kcachegrind.github.io/) 。它使用自己的数据格式，但是您可以使用 [`pyprof2calltree`](https://pypi.org/project/pyprof2calltree/) 转换来自`cProfile`的数据:

```py
$ pyprof2calltree -k -i latest_tutorial.prof
```

该命令将转换`latest_tutorial.prof`并打开 KCacheGrind 来分析数据。

最后一个选项是 [`line_profiler`](https://pypi.org/project/line-profiler/) 。`cProfile`可以告诉你你的代码在哪个函数中花费的时间最多，但是它不能让你知道在那个函数中哪个行是最慢的。这就是`line_profiler`可以帮助你的地方。

**注意:**您还可以分析代码的内存消耗。这超出了本教程的范围。但是，如果您需要监控程序的内存消耗，您可以查看 [`memory-profiler`](https://pypi.org/project/memory-profiler/) 。

请注意，行分析需要时间，并且会给运行时增加相当多的开销。正常的工作流程是首先使用`cProfile`来确定要调查哪些函数，然后对这些函数运行`line_profiler`。`line_profiler`不是标准库的一部分，所以你应该首先按照[安装说明](https://github.com/pyutils/line_profiler#installation)来设置它。

在运行分析器之前，您需要告诉它要分析哪些函数。您可以通过在源代码中添加一个`@profile`装饰器来做到这一点。例如，为了对`Timer.stop()`进行概要分析，您可以在`timer.py`中添加以下内容:

```py
@profile def stop(self) -> float:
    # The rest of the code is unchanged
```

请注意，您没有在任何地方导入`profile`。相反，当您运行探查器时，它会自动添加到全局命名空间中。不过，在完成分析后，您需要删除这一行。否则你会得到一个`NameError`。

接下来，使用`kernprof`运行分析器，它是`line_profiler`包的一部分:

```py
$ kernprof -l latest_tutorial.py
```

该命令自动将 profiler 数据保存在一个名为`latest_tutorial.py.lprof`的文件中。您可以使用`line_profiler`查看这些结果:

```py
$ python -m line_profiler latest_tutorial.py.lprof
Timer unit: 1e-06 s

Total time: 1.6e-05 s
File: /home/realpython/timer.py
Function: stop at line 35

# Hits Time PrHit %Time Line Contents
=====================================
35                      @profile
36                      def stop(self) -> float:
37                          """Stop the timer, and report the elapsed time"""
38  1   1.0   1.0   6.2     if self._start_time is None:
39                              raise TimerError(f"Timer is not running. ...")
40
41                          # Calculate elapsed time
42  1   2.0   2.0  12.5     elapsed_time = time.perf_counter() - self._start_time
43  1   0.0   0.0   0.0     self._start_time = None
44
45                          # Report elapsed time
46  1   0.0   0.0   0.0     if self.logger:
47  1  11.0  11.0  68.8         self.logger(self.text.format(elapsed_time))
48  1   1.0   1.0   6.2     if self.name:
49  1   1.0   1.0   6.2         self.timers[self.name] += elapsed_time
50
51  1   0.0   0.0   0.0     return elapsed_time
```

首先，注意这个报告中的时间单位是微秒(`1e-06 s`)。通常，最容易查看的数字是`%Time`，它告诉您代码在每一行中花费在函数中的时间占总时间的百分比。在这个例子中，您可以看到您的代码在第 47 行花费了几乎 70%的时间，这是格式化和打印计时器结果的行。

## 结论

在本教程中，您已经尝试了几种不同的方法来将 Python 计时器添加到代码中:

*   您使用了一个**类**来保存状态并添加一个用户友好的界面。类非常灵活，直接使用`Timer`可以完全控制如何以及何时调用计时器。

*   您使用了一个**上下文管理器**来为代码块添加特性，并且如果必要的话，在之后进行清理。上下文管理器使用起来很简单，添加`with Timer()`可以帮助你在视觉上更清楚地区分你的代码。

*   您使用了一个**装饰器**来为函数添加行为。Decorators 简洁而引人注目，使用`@Timer()`是监控代码运行时的一种快捷方式。

您还了解了在对代码进行基准测试时为什么应该选择`time.perf_counter()`而不是`time.time()`,以及在优化代码时有哪些其他选择。

现在您可以在自己的代码中添加 Python 计时器函数了！在日志中记录程序运行的速度有助于监控脚本。对于类、上下文管理器和装饰器一起很好地发挥作用的其他用例，你有什么想法吗？在下面留下评论吧！

## 资源

要更深入地了解 Python 计时器函数，请查看以下资源:

*   **[`codetiming`](https://pypi.org/project/codetiming/)** 是 PyPI 上可用的 Python 定时器。
*   **[`time.perf_counter()`](https://docs.python.org/3/library/time.html#time.perf_counter)** 是用于精确计时的性能计数器。
*   **[`timeit`](https://docs.python.org/3/library/timeit.html)** 是一个比较代码片段运行时的工具。
*   **[`cProfile`](https://docs.python.org/3/library/profile.html)** 是一个在脚本和程序中寻找瓶颈的剖析器。
*   **[`pstats`](https://docs.python.org/3/library/profile.html#pstats.Stats)** 是一个查看分析器数据的命令行工具。
*   **[KCachegrind](https://kcachegrind.github.io/)** 是查看 profiler 数据的 GUI。
*   **[`line_profiler`](https://pypi.org/project/line-profiler/)** 是一个用于测量单独代码行的分析器。
*   **[`memory-profiler`](https://pypi.org/project/memory-profiler/)** 是一个用于监控内存使用情况的分析器。************