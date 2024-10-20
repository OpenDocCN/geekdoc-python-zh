# Python 101:日志介绍

> 原文：<https://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/>

Python 在其标准库中提供了一个非常强大的日志库。很多程序员使用 **print** 语句进行调试(包括我自己)，但是你也可以使用日志来做这件事。使用日志记录实际上更干净，因为您不必遍历所有代码来删除 print 语句。在本教程中，我们将讨论以下主题:

*   创建简单的记录器
*   如何从多个模块登录
*   日志格式
*   日志配置

在本教程结束时，您应该能够自信地为您的应用程序创建自己的日志。我们开始吧！

### 创建简单的记录器

使用日志模块创建日志既简单又直接。最简单的方法是看一段代码，然后解释它，所以这里有一些代码供你阅读:

```py
import logging

# add filemode="w" to overwrite
logging.basicConfig(filename="sample.log", level=logging.INFO)

logging.debug("This is a debug message")
logging.info("Informational message")
logging.error("An error has happened!")

```

如您所料，要访问日志模块，您必须首先导入它。创建日志最简单的方法是使用日志模块的 basicConfig 函数，并向它传递一些关键字参数。它接受以下内容:文件名、文件模式、格式、日期、级别和流。在我们的示例中，我们传递给它一个文件名和日志级别，我们将日志级别设置为 INFO。日志记录有五个级别(按升序排列):调试、信息、警告、错误和严重。默认情况下，如果您多次运行此代码，它将追加到日志中(如果它已经存在)。如果您想让日志记录器覆盖日志，那么就像代码注释中提到的那样传入一个 **filemode="w"** 。说到运行代码，如果您运行了一次，应该会得到以下结果:

```py
INFO:root:Informational message
ERROR:root:An error has happened!

```

请注意，调试消息不在输出中。这是因为我们将级别设置为 INFO，所以我们的日志记录程序只会记录 INFO、警告、错误或关键消息。**根**部分仅仅意味着这个日志消息来自根日志记录器或主日志记录器。我们将在下一节中探讨如何改变这一点，以便更好地描述。如果不使用 **basicConfig** ，那么日志模块将输出到 console / stdout。

日志模块还可以将一些异常记录到文件中，或者记录到您配置的任何地方。这里有一个例子:

```py
import logging

logging.basicConfig(filename="sample.log", level=logging.INFO)
log = logging.getLogger("ex")

try:
    raise RuntimeError
except Exception, err:
    log.exception("Error!")

```

这将把整个回溯记录到文件中，这在调试时非常方便。

### 如何从多个模块登录(以及格式化！)

您编写的代码越多，您就越有可能创建一组共同工作的定制模块。如果您希望它们都登录到同一个位置，那么您来对地方了。我们将看一下简单的方法，然后展示一个更复杂的方法，它也更具可定制性。这里有一个简单的方法:

```py
import logging
import otherMod

#----------------------------------------------------------------------
def main():
    """
    The main entry point of the application
    """
    logging.basicConfig(filename="mySnake.log", level=logging.INFO)
    logging.info("Program started")
    result = otherMod.add(7, 8)
    logging.info("Done!")

if __name__ == "__main__":
    main()

```

在这里，我们导入日志记录和我们自己创建的一个模块(“otherMod”)。然后我们像以前一样创建日志文件。另一个模块如下所示:

```py
# otherMod.py
import logging

#----------------------------------------------------------------------
def add(x, y):
    """"""
    logging.info("added %s and %s to get %s" % (x, y, x+y))
    return x+y

```

如果您运行主代码，您应该得到一个包含以下内容的日志:

```py
INFO:root:Program started
INFO:root:added 7 and 8 to get 15
INFO:root:Done!

```

你看这样做有什么问题吗？您真的不能很容易地判断日志消息来自哪里。写这个日志的模块越多，只会越混乱。所以我们需要解决这个问题。这让我们想到了创建记录器的复杂方式。让我们来看一个不同的实现:

```py
import logging
import otherMod2

#----------------------------------------------------------------------
def main():
    """
    The main entry point of the application
    """
    logger = logging.getLogger("exampleApp")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    fh = logging.FileHandler("new_snake.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)

    logger.info("Program started")
    result = otherMod2.add(7, 8)
    logger.info("Done!")

if __name__ == "__main__":
    main()

```

在这里，我们创建一个名为“exampleApp”的记录器实例。我们设置它的日志级别，创建一个日志文件处理程序对象和一个日志格式化程序对象。文件处理程序必须将 formatter 对象设置为其格式化程序，然后文件处理程序必须添加到 logger 实例中。**主**中的其余代码大部分都是一样的。请注意，它不是“logging.info ”,而是“logger.info”或任何您称之为 logger 变量的东西。以下是更新后的 **otherMod2** 代码:

```py
# otherMod2.py
import logging

module_logger = logging.getLogger("exampleApp.otherMod2")

#----------------------------------------------------------------------
def add(x, y):
    """"""
    logger = logging.getLogger("exampleApp.otherMod2.add")
    logger.info("added %s and %s to get %s" % (x, y, x+y))
    return x+y

```

注意，这里我们定义了两个记录器。在这种情况下，我们不用**模块记录器**做任何事情，但是我们使用另一个。如果运行主脚本，您应该会在文件中看到以下输出:

```py
2012-08-02 15:37:40,592 - exampleApp - INFO - Program started
2012-08-02 15:37:40,592 - exampleApp.otherMod2.add - INFO - added 7 and 8 to get 15
2012-08-02 15:37:40,592 - exampleApp - INFO - Done!

```

你会注意到，我们不再有任何参考根已被删除。相反，它使用我们的 Formatter 对象，该对象表示我们应该获得人类可读的时间、记录器名称、日志记录级别，然后是消息。这些实际上被称为**日志记录**属性。关于日志记录属性的完整列表，请参见[文档](http://docs.python.org/library/logging.html#logrecord-attributes)，因为这里列出的太多了。

### 为工作和娱乐配置日志

日志模块可以通过 3 种不同的方式进行配置。您可以像我们在本文前面所做的那样，使用方法(记录器、格式化器、处理程序)来配置它；您可以使用一个配置文件，并将其传递给 file config()；或者您可以创建一个配置信息字典，并将其传递给 dictConfig()函数。让我们先创建一个配置文件，然后看看如何用 Python 执行它。下面是一个配置文件示例:

```py
[loggers]
keys=root,exampleApp

[handlers]
keys=fileHandler, consoleHandler

[formatters]
keys=myFormatter

[logger_root]
level=CRITICAL
handlers=consoleHandler

[logger_exampleApp]
level=INFO
handlers=fileHandler
qualname=exampleApp

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=myFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
formatter=myFormatter
args=("config.log",)

[formatter_myFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=

```

您会注意到我们指定了两个记录器:root 和 exampleApp。无论什么原因，都需要“root”。如果不包含它，Python 会从 config.py 的 **_install_loggers** 函数中抛出一个 ValueError，该函数是日志模块的一部分。如果您将 root 的处理程序设置为 **fileHandler** ，那么您将导致日志输出加倍，因此为了防止这种情况发生，我们将它发送到控制台。仔细研究这个例子。前三个部分中的每个键都需要一个部分。现在让我们看看如何在代码中加载它:

```py
# log_with_config.py
import logging
import logging.config
import otherMod2

#----------------------------------------------------------------------
def main():
    """
    Based on http://docs.python.org/howto/logging.html#configuring-logging
    """
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger("exampleApp")

    logger.info("Program started")
    result = otherMod2.add(7, 8)
    logger.info("Done!")

if __name__ == "__main__":
    main()

```

如您所见，您需要做的就是将配置文件路径传递给**logging . config . file config**。您还会注意到，我们不再需要所有的设置代码，因为它们都在配置文件中。我们也可以直接导入 **otherMod2** 模块，不做任何改动。无论如何，如果您运行以上内容，您的日志文件中应该会出现以下内容:

```py
2012-08-02 18:23:33,338 - exampleApp - INFO - Program started
2012-08-02 18:23:33,338 - exampleApp.otherMod2.add - INFO - added 7 and 8 to get 15
2012-08-02 18:23:33,338 - exampleApp - INFO - Done!

```

您可能已经猜到了，它与另一个示例非常相似。现在我们将转到另一个配置方法。直到 Python 2.7 才添加了字典配置方法(dictConfig ),所以请确保您拥有该方法或更好的方法，否则您将无法跟上。没有很好地记录这是如何工作的。事实上，出于某种原因，文档中的示例显示了 YAML。无论如何，这里有一些工作代码供您查看:

```py
# log_with_config.py
import logging
import logging.config
import otherMod2

#----------------------------------------------------------------------
def main():
    """
    Based on http://docs.python.org/howto/logging.html#configuring-logging
    """
    dictLogConfig = {
        "version":1,
        "handlers":{
                    "fileHandler":{
                        "class":"logging.FileHandler",
                        "formatter":"myFormatter",
                        "filename":"config2.log"
                        }
                    },        
        "loggers":{
            "exampleApp":{
                "handlers":["fileHandler"],
                "level":"INFO",
                }
            },

        "formatters":{
            "myFormatter":{
                "format":"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            }
        }

    logging.config.dictConfig(dictLogConfig)

    logger = logging.getLogger("exampleApp")

    logger.info("Program started")
    result = otherMod2.add(7, 8)
    logger.info("Done!")

if __name__ == "__main__":
    main()

```

如果您运行这段代码，您将得到与前一个方法相同的输出。注意，当您使用字典配置时，您不需要“根”记录器。

### 包扎

至此，您应该知道如何开始使用记录器，以及如何以几种不同的方式配置它们。您还应该已经了解了如何使用 Formatter 对象修改输出。如果你真的想了解输出，我建议你查看下面的一些链接。

### 附加阅读

*   记录模块[文档](http://docs.python.org/library/logging.html)
*   记录[如何操作](http://docs.python.org/howto/logging.html)
*   日志[食谱](http://docs.python.org/howto/logging-cookbook.html)
*   [日志 _ 树](http://rhodesmill.org/brandon/2012/logging_tree/)包
*   水管工杰克的 [Python 日志 101](http://plumberjack.blogspot.com/2009/09/python-logging-101.html)
*   停止使用 print 进行调试:[Python 日志模块的 5 分钟快速入门指南](http://inventwithpython.com/blog/2012/04/06/stop-using-print-for-debugging-a-5-minute-quickstart-guide-to-pythons-logging-module/)
*   Hellman 的 [PyMOTW 日志页面](http://www.doughellmann.com/PyMOTW/logging/)

### 源代码

*   [logging_tut.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/08/logging_tut.zip)