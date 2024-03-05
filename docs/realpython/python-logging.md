# 在 Python 中登录

> 原文：<https://realpython.com/python-logging/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**登录 Python**](/courses/logging-python/)

日志是程序员工具箱中非常有用的工具。它可以帮助您更好地理解程序的流程，并发现您在开发时可能没有想到的场景。

日志为开发人员提供了一双额外的眼睛，时刻关注着应用程序正在经历的流程。它们可以存储信息，比如哪个用户或 IP 访问了应用程序。如果发生错误，它们可以告诉您程序在到达发生错误的代码行之前的状态，从而提供比堆栈跟踪更多的信息。

通过从正确的位置记录有用的数据，您不仅可以轻松地调试错误，还可以使用这些数据来分析应用程序的性能，以规划扩展或查看使用模式来规划营销。

Python 提供了一个日志系统作为其标准库的一部分，因此您可以快速地将日志添加到您的应用程序中。在本文中，您将了解为什么使用这个模块是向您的应用程序添加日志记录的最佳方式，以及如何快速入门，并且您将了解一些可用的高级特性。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 日志模块

Python 中的[日志模块](https://realpython.com/python-logging-source-code/)是一个随时可用的强大模块，旨在满足初学者以及企业团队的需求。大多数第三方 Python 库都使用它，因此您可以将日志消息与这些库中的日志消息集成在一起，为您的应用程序生成一个同构日志。

向 Python 程序添加日志记录就像这样简单:

```py
import logging
```

导入日志模块后，您可以使用一个叫做“日志记录器”的东西来记录您想要查看的消息。默认情况下，有 5 个标准级别来表示事件的严重性。每个都有相应的方法，可用于记录该严重级别的事件。定义的级别(按严重性递增顺序排列)如下:

*   调试
*   信息
*   警告
*   错误
*   批评的

日志模块为您提供了一个默认的日志记录器，允许您无需做太多配置就可以开始使用。每个级别对应的方法都可以调用，如下例所示:

```py
import logging

logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')
```

上述程序的输出如下所示:

```py
WARNING:root:This is a warning message
ERROR:root:This is an error message
CRITICAL:root:This is a critical message
```

输出显示了每条消息前的严重性级别以及`root`，这是日志记录模块为其默认日志记录器指定的名称。(日志程序将在后面的章节中详细讨论。)这种格式显示由冒号(`:`)分隔的级别、名称和消息，是默认的输出格式，可以配置为包括时间戳、行号和其他细节。

请注意，`debug()`和`info()`消息没有被记录。这是因为，默认情况下，日志模块记录严重级别为`WARNING`或更高的消息。如果您愿意，可以通过配置日志模块来记录所有级别的事件来改变这种情况。您还可以通过更改配置来定义自己的严重性级别，但通常不建议这样做，因为这会与您可能正在使用的一些第三方库的日志混淆。

[*Remove ads*](/account/join/)

## 基本配置

您可以使用`basicConfig(**` *`kwargs`* `)`的方法来配置日志记录:

> “您会注意到日志模块打破了 PEP8 styleguide，使用了`camelCase`命名约定。这是因为它采用了 Log4j，一个 Java 中的日志记录实用程序。这是软件包中的一个已知问题，但在决定将其添加到标准库中时，它已经被用户采用，更改它以满足 PEP8 要求会导致向后兼容性问题。”[(来源)](https://wiki.python.org/moin/LoggingPackage)

`basicConfig()`的一些常用参数如下:

*   `level`:root logger 将被设置为指定的严重级别。
*   `filename`:指定文件。
*   `filemode`:如果给定了`filename`，则以此模式打开文件。默认是`a`，意思是追加。
*   `format`:这是日志消息的格式。

通过使用`level`参数，您可以设置想要记录的日志消息的级别。这可以通过传递类中可用的一个常数来实现，这将允许记录该级别或更高级别的所有日志调用。这里有一个例子:

```py
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug('This will get logged')
```

```py
DEBUG:root:This will get logged
```

所有在`DEBUG`级别或以上的事件都将被记录。

类似地，对于记录到文件而不是控制台，可以使用`filename`和`filemode`，并且您可以使用`format`来决定消息的格式。下面的示例显示了所有三种方法的用法:

```py
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This will get logged to a file')
```

```py
root - ERROR - This will get logged to a file
```

该消息看起来像这样，但是将被写入一个名为`app.log`的文件，而不是控制台。filemode 设置为`w`，这意味着每次调用`basicConfig()`时，日志文件以“写模式”打开，程序每次运行都会重写文件。filemode 的默认配置是`a`，它是 append。

您可以通过使用更多的参数来定制根日志记录器，这些参数可以在[这里](https://docs.python.org/3/library/logging.html#logging.basicConfig)找到。

需要注意的是，调用`basicConfig()`来配置根日志记录器只有在之前没有配置根日志记录器的情况下才有效。**基本上这个函数只能调用一次。**

`debug()`、`info()`、`warning()`、`error()`、`critical()`如果之前没有调用过`basicConfig()`，也会自动调用`basicConfig()`而不带参数。这意味着在第一次调用上述函数之一后，您不能再配置根日志记录器，因为它们会在内部调用`basicConfig()`函数。

`basicConfig()`中的默认设置是设置记录器以下列格式写入控制台:

```py
ERROR:root:This is an error message
```

## 格式化输出

虽然您可以将程序中任何可以表示为字符串的变量作为消息传递给日志，但是有一些基本元素已经是`LogRecord`的一部分，可以很容易地添加到输出格式中。如果您想记录进程 ID 以及级别和消息，您可以这样做:

```py
import logging

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
logging.warning('This is a Warning')
```

```py
18472-WARNING-This is a Warning
```

`format`可以任意排列带有`LogRecord`属性的字符串。可用属性的完整列表可以在[这里](https://docs.python.org/3/library/logging.html#logrecord-attributes)找到。

这是另一个您可以添加日期和时间信息的示例:

```py
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Admin logged in')
```

```py
2018-07-11 20:12:06,288 - Admin logged in
```

`%(asctime)s`添加`LogRecord`的创建时间。可以使用`datefmt`属性更改格式，该属性使用与 datetime 模块中的格式化函数相同的格式化语言，例如`time.strftime()`:

```py
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.warning('Admin logged out')
```

```py
12-Jul-18 20:53:19 - Admin logged out
```

你可以在这里找到指南[。](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)

[*Remove ads*](/account/join/)

### 记录变量数据

在大多数情况下，您会希望在日志中包含来自应用程序的动态信息。您已经看到，日志记录方法将一个字符串作为参数，在单独的一行中用可变数据格式化一个字符串并将其传递给 log 方法似乎是很自然的。但这实际上可以通过使用消息的格式字符串并附加变量数据作为参数来直接完成。这里有一个例子:

```py
import logging

name = 'John'

logging.error('%s raised an error', name)
```

```py
ERROR:root:John raised an error
```

传递给该方法的参数将作为变量数据包含在消息中。

虽然您可以使用任何格式样式，但 Python 3.6 中引入的 [f-strings](https://realpython.com/python-f-strings/) 是一种格式化字符串的绝佳方式，因为它们有助于保持格式简短易读:

```py
import logging

name = 'John'

logging.error(f'{name} raised an error')
```

```py
ERROR:root:John raised an error
```

### 捕获堆栈跟踪

日志模块还允许您捕获应用程序中的完整堆栈跟踪。如果将`exc_info`参数作为`True`传递，则可以捕获[异常信息](https://realpython.com/python-exceptions/)，日志记录函数调用如下:

```py
import logging

a = 5
b = 0

try:
  c = a / b
except Exception as e:
  logging.error("Exception occurred", exc_info=True)
```

```py
ERROR:root:Exception occurred
Traceback (most recent call last):
 File "exceptions.py", line 6, in <module>
 c = a / b
ZeroDivisionError: division by zero
[Finished in 0.2s]
```

如果`exc_info`没有设置为`True`，上述程序的输出不会告诉我们任何关于异常的信息，在现实世界中，这可能不像`ZeroDivisionError`那么简单。想象一下，试图在一个复杂的代码库中调试一个错误，而日志只显示以下内容:

```py
ERROR:root:Exception occurred
```

这里有一个小提示:如果您从一个异常处理程序进行日志记录，请使用`logging.exception()`方法，该方法记录一条级别为`ERROR`的消息，并将异常信息添加到消息中。更简单的说，叫`logging.exception()`就像叫`logging.error(exc_info=True)`。但是因为这个方法总是转储异常信息，所以应该只从异常处理程序中调用它。看一下这个例子:

```py
import logging

a = 5
b = 0
try:
  c = a / b
except Exception as e:
  logging.exception("Exception occurred")
```

```py
ERROR:root:Exception occurred
Traceback (most recent call last):
 File "exceptions.py", line 6, in <module>
 c = a / b
ZeroDivisionError: division by zero
[Finished in 0.2s]
```

使用`logging.exception()`将显示`ERROR`级别的日志。如果您不希望这样，您可以调用从`debug()`到`critical()`的任何其他日志记录方法，并将`exc_info`参数作为`True`传递。

## 类别和功能

到目前为止，我们已经看到了名为`root`的默认记录器，每当像这样直接调用它的函数时，它都会被日志模块使用:`logging.debug()`。你可以(也应该)通过[创建一个`Logger`类的对象](https://realpython.com/python3-object-oriented-programming/)来定义你自己的日志记录器，尤其是当你的应用程序有多个模块的时候。让我们看看模块中的一些类和函数。

日志模块中定义的最常用的类如下:

*   **`Logger` :** 这个类的对象将在应用程序代码中直接调用函数。

*   **`LogRecord` :** 记录器自动创建`LogRecord`对象，这些对象包含与正在记录的事件相关的所有信息，比如记录器的名称、函数、行号、消息等等。

*   **`Handler` :** 处理程序将`LogRecord`发送到所需的输出目的地，如控制台或文件。`Handler`是`StreamHandler`、`FileHandler`、`SMTPHandler`、`HTTPHandler`等子类的基础。这些子类将日志输出发送到相应的目的地，比如`sys.stdout`或磁盘文件。

*   **`Formatter` :** 这是通过指定一个列出输出应该包含的属性的字符串格式来指定输出格式的地方。

其中，我们主要处理的是`Logger`类的对象，它们是使用模块级函数`logging.getLogger(name)`实例化的。用同一个`name`多次调用`getLogger()`将返回一个对同一个`Logger`对象的引用，这样我们就不用把日志对象传递到每个需要的地方。这里有一个例子:

```py
import logging

logger = logging.getLogger('example_logger')
logger.warning('This is a warning')
```

```py
This is a warning
```

这将创建一个名为`example_logger`的定制日志记录器，但是与根日志记录器不同，定制日志记录器的名称不是默认输出格式的一部分，必须添加到配置中。将其配置为显示记录器名称的格式将会产生如下输出:

```py
WARNING:example_logger:This is a warning
```

同样，与根日志记录器不同，定制日志记录器不能使用`basicConfig()`进行配置。您必须使用处理程序和格式化程序来配置它:

> “建议我们使用模块级记录器，通过将名称参数`__name__`传递给`getLogger()`来创建一个记录器对象，因为记录器本身的名称会告诉我们从哪里记录事件。`__name__`是 Python 中一个特殊的内置变量，它计算当前模块的名称[(来源)](https://docs.python.org/3/library/logging.html#logger-objects)

[*Remove ads*](/account/join/)

## 使用处理程序

当您想要配置自己的日志记录程序，并在生成日志时将日志发送到多个地方时，处理程序就会出现。处理程序将日志消息发送到已配置的目的地，如标准输出流或文件，或者通过 HTTP 或 SMTP 发送到您的电子邮件。

您创建的记录器可以有多个处理程序，这意味着您可以将其设置为保存到日志文件中，也可以通过电子邮件发送。

与记录器一样，您也可以在处理程序中设置严重级别。如果您希望为同一个日志程序设置多个处理程序，但希望每个处理程序具有不同的严重级别，这将非常有用。例如，您可能希望将级别为`WARNING`及以上的日志记录到控制台，但是级别为`ERROR`及以上的所有内容也应该保存到文件中。这里有一个程序可以做到这一点:

```py
# logging_example.py

import logging

# Create a custom logger
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.warning('This is a warning')
logger.error('This is an error')
```

```py
__main__ - WARNING - This is a warning
__main__ - ERROR - This is an error
```

这里，`logger.warning()`创建了一个保存所有事件信息的`LogRecord`，并将其传递给它拥有的所有处理程序:`c_handler`和`f_handler`。

`c_handler`是一个带`WARNING`电平的`StreamHandler`，从`LogRecord`获取信息，生成指定格式的输出，并打印到控制台。`f_handler`是一个带有级别`ERROR`的`FileHandler`，它忽略这个`LogRecord`，因为它的级别是`WARNING`。

当`logger.error()`被调用时，`c_handler`的行为与之前完全一样，`f_handler`得到一个`ERROR`级别的`LogRecord`，因此它继续生成一个类似于`c_handler`的输出，但它不是将输出打印到控制台，而是以如下格式将其写入指定文件:

```py
2018-08-03 16:12:21,723 - __main__ - ERROR - This is an error
```

对应于`__name__`变量的记录器的名称被记录为`__main__`，这是 Python 分配给开始执行的模块的名称。如果这个文件是由其他模块导入的，那么`__name__`变量将对应于它的名字 *logging_example* 。下面是它的样子:

```py
# run.py

import logging_example
```

```py
logging_example - WARNING - This is a warning
logging_example - ERROR - This is an error
```

## 其他配置方法

您可以使用模块和类函数，或者通过创建一个配置文件或一个[字典](https://realpython.com/python-dicts/)并分别使用`fileConfig()`或`dictConfig()`加载，如上所示配置日志记录。如果您想在正在运行的应用程序中更改日志记录配置，这些选项非常有用。

下面是一个文件配置示例:

```py
[loggers] keys=root,sampleLogger [handlers] keys=consoleHandler [formatters] keys=sampleFormatter [logger_root] level=DEBUG handlers=consoleHandler [logger_sampleLogger] level=DEBUG handlers=consoleHandler qualname=sampleLogger propagate=0 [handler_consoleHandler] class=StreamHandler level=DEBUG formatter=sampleFormatter args=(sys.stdout,) [formatter_sampleFormatter] format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

在上面的文件中，有两个记录器、一个处理程序和一个格式化程序。在定义它们的名称后，通过在它们的名称前添加单词 logger、handler 和 formatter 来配置它们，用下划线分隔。

要加载这个配置文件，您必须使用`fileConfig()`:

```py
import logging
import logging.config

logging.config.fileConfig(fname='file.conf', disable_existing_loggers=False)

# Get the logger specified in the file
logger = logging.getLogger(__name__)

logger.debug('This is a debug message')
```

```py
2018-07-13 13:57:45,467 - __main__ - DEBUG - This is a debug message
```

配置文件的路径作为参数传递给`fileConfig()`方法，而`disable_existing_loggers`参数用于保持或禁用调用函数时出现的记录器。如未提及，默认为`True`。

下面是字典方法的 [YAML](https://realpython.com/python-yaml/) 格式的相同配置:

```py
version:  1 formatters: simple: format:  '%(asctime)s  -  %(name)s  -  %(levelname)s  -  %(message)s' handlers: console: class:  logging.StreamHandler level:  DEBUG formatter:  simple stream:  ext://sys.stdout loggers: sampleLogger: level:  DEBUG handlers:  [console] propagate:  no root: level:  DEBUG handlers:  [console]
```

下面的例子展示了如何从`yaml`文件加载配置:

```py
import logging
import logging.config
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

logger.debug('This is a debug message')
```

```py
2018-07-13 14:05:03,766 - __main__ - DEBUG - This is a debug message
```

[*Remove ads*](/account/join/)

## 保持冷静，阅读日志

日志模块被认为是非常灵活的。它的设计非常实用，并且应该适合您的开箱即用的用例。您可以将基本的日志记录添加到一个小项目中，或者如果您正在处理一个大项目，您甚至可以创建自己的定制日志级别、处理程序类等等。

如果您还没有在应用程序中使用日志记录，现在是开始使用的好时机。如果做得正确，日志记录肯定会消除开发过程中的许多摩擦，并帮助您找到将应用程序提升到下一个级别的机会。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**登录 Python**](/courses/logging-python/)******