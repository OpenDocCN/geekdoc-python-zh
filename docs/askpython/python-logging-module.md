# Python 日志模块

> 原文：<https://www.askpython.com/python-modules/python-logging-module>

Python **Logging** 模块用于实现一个灵活的事件驱动日志系统，为应用存储日志事件或消息提供了一种便捷的方式。

* * *

## Python 日志模块–记录器

**Logger** 对象是这个模块的对象，我们可以操纵它来完成所有需要的日志记录。

要实例化 Logger 对象，我们必须始终指定:

```py
log_object = logging.getLogger(name)

```

对同名的`getLogger(name)`的多次调用总是引用同一个对象。

现在我们有了 logger 对象，我们可以对它使用多个函数。

* * *

## 将消息写入控制台

每当需要报告事件时，我们发出 logger 对象的内容，以便主运行程序得到状态变化的通知。

为此，我们对要发出的消息的严重程度进行了划分，称为`LEVEL`。

| 水平 | 当使用它时 |
| 调试 | 用于调试目的的详细信息 |
| 信息 | 确认一切正常 |
| 警告 | 意外发生的迹象 |
| 错误 | 更严重的问题是，当软件不能执行某些功能时 |
| 批评的 | 最严重的严重错误 |

这用于写入相应的日志文件或控制台。默认级别为`WARNING`，表示只跟踪该级别及以上的事件(即默认跟踪`WARNING`、`ERROR`、`CRITICAL`)

这使得程序员可以根据所选的严重程度控制如何显示这些状态消息。

格式:`logging.info(message)`将消息显示到控制台/文件上。

以下示例说明了这种方法

```py
import logging
# This message will be printed to the console
logging.warning('Warning message')

# Will not be printed to the console
logging.info('Works as expected')

```

输出

```py
WARNING:root:Warning message

```

* * *

## 记录到文件中

我们使用`logging.basicConfig()`来创建一个日志文件处理程序。

格式:`logging.basicConfig(filename, level)`

```py
import logging

# Create the logfile
logging.basicConfig(filename='sample.log', level=logging.DEBUG)

logging.debug('Debug message')
logging.info('info message')
logging.warning('Warning message')

```

输出

```py
[email protected] # cat sample.log
DEBUG:root:Debug message
INFO:root:info message
WARNING:root:Warning message

```

**注意**:对`basicConfig()`的调用必须在对`debug()`、`info()`等的任何调用之前。

还有另一个参数`filemode`，用于`basicConfig()`函数，它指定了日志文件的模式。

下面的例子使`sample.log`具有只写模式，这意味着写入其中的任何消息都将覆盖文件的先前内容。

```py
logging.basicConfig(filename='sample.log', filemode='w', level=logging.DEBUG)

```

* * *

## 从多个模块记录日志

因为日志文件对象和处理程序在多个模块中提供了相同的上下文，所以我们可以在其他模块中直接使用它们。

下面显示了一个示例

```py
# main.py
import logging
import sample_module

def main():
    logging.basicConfig(filename='application.log', level=logging.INFO)
    logging.info('main module started')
    sample_module.sample_function()
    logging.info('completed')

main()

```

```py
# sample_module.py
import logging

def sample_function():
    logging.info('Sample module doing work')

```

在这里，同一个日志对象可以由多个模块共享，因此非常适合模块化代码。

* * *

## 消息的格式

默认情况下，输出消息有一个包含节点名称和消息级别的消息头。要更改显示消息的格式，必须指定合适的格式。

```py
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.debug('sample message')

```

输出

```py
DEBUG:sample message

```

### 在邮件中显示日期和时间

添加`%(asctime)s`格式说明符来表示消息中的时间。

```py
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.warning('Sample message')

```

输出

```py
12/26/2019 12:50:38 PM Sample message

```

* * *

## 记录器对象名称

默认情况下，日志消息的第一部分包含所使用的 logger 对象的级别和名称。(例如:`DEBUG:ROOT:sample message`)

通常，如果没有指定`name`参数，它默认为根节点的名称`ROOT`。

否则，最好使用`__name__`变量，因为它是 Python 包名称空间中模块的名称。

```py
import logging

logger = logging.getLogger(__name__)

```

## 修改消息的级别

Logger 对象为我们提供了一种修改显示消息的阈值级别的方法。`setLevel(level)`方法用于设置记录器对象的级别。

格式:`logger.setLevel(level)`

```py
import logging

logger = logging.getLogger(__name__)

# Set the log level as DEBUG
logger.setLevel(logging.DEBUG)

# The DEBUG level message is displayed
logger.debug('Debug message')

```

输出

```py
No handlers could be found for logger "__main__"

```

这不是我们所期望的。为什么没有显示消息，什么是处理程序？

* * *

## 日志处理程序

日志处理程序是一个完成日志/控制台写入工作的组件。logger 对象调用日志处理程序来显示消息的内容。

与记录器的情况不同，处理程序从不直接实例化。有不同类型的处理程序，每一种都有自己的实例化方法。

## 处理程序的类型

日志模块中有各种各样的处理程序，但我们主要关注 3 个最常用的处理程序，即:

*   StreamHandler
*   文件处理程序
*   NullHandler

### StreamHandler

StreamHandler 用于将日志输出发送到流，如`stdout`、`stderr`，或任何支持`write()`和`flush()`方法的类似文件的对象，如管道、FIFOs 等。

我们可以使用`StreamHandler()`来初始化一个 StreamHandler 对象，它可以在控制台上显示来自我们的 Logger 对象的消息。

前面的代码片段现在可以通过调用`StreamHandler()`和`handler.setLevel()`来完成。

```py
import logging

# Instantiate the logger object
logger = logging.getLogger(name='hi')

# Set the level of the logger
logger.setLevel(logging.DEBUG)

# Initialise the handler object for writing
handler = logging.StreamHandler()

# The handler also needs to have a level
handler.setLevel(logging.DEBUG)

# Add the handler to the logger object
logger.addHandler(handler)

# Now, the message is ready to be printed by the handler
logger.debug('sample message')

```

输出

```py
sample message

```

### 文件处理程序

为了记录到文件中，我们可以使用 FileHandler 对象。它也类似于 StreamHandler 对象，但是这里引用了一个文件描述符，以便对文件进行日志记录。

在实例化日志处理程序时，可以修改上面的代码片段。通过将类型更改为 FileHandler，可以将消息记录到文件中。

```py
handler_name = logging.FileHandler(filename='sample.log', mode='a')

```

### NullHandler

这个处理程序本质上不写任何东西(相当于将输出管道化到`/dev/null`)，因此被认为是一个无操作处理程序，对库开发人员很有用。

* * *

## 结论

我们学习了如何使用日志模块 API 根据消息的严重程度将消息记录到控制台和文件中。我们还学习了如何使用格式说明符来指定消息的显示方式，以及如何使用日志处理程序来控制和修改记录的消息的级别。

* * *

## 参考

日志模块的 Python 官方文档:[https://docs . python . org/3/how to/Logging . html # Logging-basic-tutorial](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial)