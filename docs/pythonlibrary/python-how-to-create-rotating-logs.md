# Python:如何创建旋转日志

> 原文：<https://www.blog.pythonlibrary.org/2014/02/11/python-how-to-create-rotating-logs/>

Python 的日志模块有很多选项。在本文中，我们将研究日志模块创建循环日志的能力。Python 支持两种类型的旋转日志:

*   基于大小旋转日志(**旋转文件处理器**
*   根据某个时间间隔( **TimedRotatingFileHandler** )轮换日志

让我们花一些时间来了解这两种类型的记录器是如何实现和使用的。

* * *

### 旋转文件处理程序

日志模块中的 RotatingFileHandler 类允许开发人员创建一个日志处理程序对象，使他们能够根据日志的大小来旋转日志。您可以使用 **maxBytes** 参数来告诉它何时旋转日志。这意味着当日志达到一定的字节数时，它会被“翻转”。当文件大小即将超出时，会出现这种情况。处理程序将关闭该文件，并自动打开一个新文件。如果您为 **backupCount** 参数传入一个数字，那么它会将“. 1”、“. 2”等附加到日志文件的末尾。让我们看一个简单的例子:

```py

import logging
import time

from logging.handlers import RotatingFileHandler

#----------------------------------------------------------------------
def create_rotating_log(path):
    """
    Creates a rotating log
    """
    logger = logging.getLogger("Rotating Log")
    logger.setLevel(logging.INFO)

    # add a rotating handler
    handler = RotatingFileHandler(path, maxBytes=20,
                                  backupCount=5)
    logger.addHandler(handler)

    for i in range(10):
        logger.info("This is test log line %s" % i)
        time.sleep(1.5)

#----------------------------------------------------------------------
if __name__ == "__main__":
    log_file = "test.log"
    create_rotating_log(log_file)

```

这段代码基于 [Python 日志记录指南](http://docs.python.org/2/howto/logging-cookbook.html#using-file-rotation)中的一个例子。这里我们创建一个日志级别为 INFO 的循环日志。然后，我们设置处理程序，每当日志文件的长度为 20 个字节时就旋转日志。是的，这是一个低得荒谬的数字，但它使演示发生了什么更容易。接下来，我们创建一个循环，它将在日志文件中创建 10 行，在每次调用 log 之间有一个休眠。如果您运行这段代码，您应该得到六个文件:原始的 test.log 和 5 个备份日志。

现在让我们看看如何使用一个 **TimedRotatingFileHandler** 。

* * *

### TimedRotatingFileHandler

TimedRotatingFileHandler 允许开发人员根据过去的时间创建一个循环日志。您可以将其设置为在以下时间条件下旋转日志:

*   秒
*   分钟(米)
*   小时
*   日(d)
*   w0-w6(工作日，0 =星期一)
*   午夜

要设置其中一个条件，只需在第二个参数 **when** 中传递它。您还需要设置间隔参数。让我们来看一个例子:

```py

import logging
import time

from logging.handlers import TimedRotatingFileHandler

#----------------------------------------------------------------------
def create_timed_rotating_log(path):
    """"""
    logger = logging.getLogger("Rotating Log")
    logger.setLevel(logging.INFO)

    handler = TimedRotatingFileHandler(path,
                                       when="m",
                                       interval=1,
                                       backupCount=5)
    logger.addHandler(handler)

    for i in range(6):
        logger.info("This is a test!")
        time.sleep(75)

#----------------------------------------------------------------------
if __name__ == "__main__":
    log_file = "timed_test.log"
    create_timed_rotating_log(log_file)

```

此示例将每分钟轮换一次日志，备份计数为 5。更实际的轮换可能是在小时上，所以您应该将间隔设置为 60，或者将时间设置为“h”。当这段代码运行时，它也将创建 6 个文件，但是它将使用 strftime 格式 **%Y-%m-%d_%H-%M-%S** 附加一个时间戳，而不是将整数附加到日志文件名上。

* * *

### 包扎

现在你知道如何使用 Python 强大的旋转日志了。希望你能把它集成到你自己的应用程序或未来的程序中。

* * *

### 相关阅读

*   关于[旋转文件处理器](http://docs.python.org/2/library/logging.handlers.html#rotatingfilehandler)的 Python 文档
*   关于 [TimedRotatingFileHandler](http://docs.python.org/2/library/logging.handlers.html#timedrotatingfilehandler) 的 Python 文档
*   Python 日志记录指南- [使用文件旋转](http://docs.python.org/2/howto/logging-cookbook.html#using-file-rotation)
*   Python 101: [日志介绍](https://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/)
*   Python 日志:[如何记录多个位置](https://www.blog.pythonlibrary.org/2013/07/18/python-logging-how-to-log-to-multiple-locations/)