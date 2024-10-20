# Python:如何创建异常日志装饰器

> 原文：<https://www.blog.pythonlibrary.org/2016/06/09/python-how-to-create-an-exception-logging-decorator/>

前几天，我决定创建一个装饰器来捕捉异常并记录它们。我在 [Github](https://gist.github.com/diosmosis/1148066) 上找到了一个相当复杂的例子，我用它来思考如何完成这项任务，并得出了以下结论:

```py

# exception_decor.py

import functools
import logging

def create_logger():
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("example_logger")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    fh = logging.FileHandler("/path/to/test.log")

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)
    return logger

def exception(function):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        logger = create_logger()
        try:
            return function(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__
            logger.exception(err)

            # re-raise the exception
            raise
    return wrapper

```

在这段代码中，我们有两个函数。第一个创建一个日志对象并返回它。第二个函数是我们的装饰函数。这里，我们将传入的函数包装在一个 try/except 中，并使用我们的记录器记录发生的任何异常。您会注意到，我还记录了发生异常的函数名。

现在我们只需要测试一下这个装饰器。为此，您可以创建一个新的 Python 脚本，并向其中添加以下代码。确保将它保存在与上面代码相同的位置。

```py

from exception_decor import exception

@exception
def zero_divide():
    1 / 0

if __name__ == '__main__':
    zero_divide()

```

当您从命令行运行此代码时，您应该得到一个包含以下内容的日志文件:

```py

2016-06-09 08:26:50,874 - example_logger - ERROR - There was an exception in  zero_divide
Traceback (most recent call last):
  File "/home/mike/exception_decor.py", line 29, in wrapper
    return function(*args, **kwargs)
  File "/home/mike/test_exceptions.py", line 5, in zero_divide
    1 / 0
ZeroDivisionError: integer division or modulo by zero

```

我认为这是一个方便的代码，我希望你会发现它也很有用！

**UPDATE** :一位敏锐的读者指出，将这个脚本一般化是一个好主意，这样您就可以向装饰者传递一个 logger 对象。让我们来看看它是如何工作的！

### 将一个记录器传递给我们的装饰者

首先，让我们将我们的日志代码分离到它自己的模块中。姑且称之为 **exception_logger.py** 。下面是放入该文件的代码:

```py

# exception_logger.py

import logging

def create_logger():
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("example_logger")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    fh = logging.FileHandler(r"/path/to/test.log")

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)
    return logger

logger = create_logger()

```

接下来，我们需要修改装饰器代码，这样我们就可以接受一个日志记录器作为参数。务必将其保存为 **exception_decor.py**

```py

# exception_decor.py

import functools

def exception(logger):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur

    @param logger: The logging object
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                # log the exception
                err = "There was an exception in  "
                err += func.__name__
                logger.exception(err)

            # re-raise the exception
            raise
        return wrapper
    return decorator

```

你会注意到这里有多层嵌套函数。请务必仔细研究，以了解发生了什么。最后我们需要修改我们的测试脚本:

```py

from exception_decor import exception
from exception_logger import logger

@exception(logger)
def zero_divide():
    1 / 0

if __name__ == '__main__':
    zero_divide()

```

这里我们导入了我们的装饰器和记录器。然后我们装饰我们的函数，并把 logger 对象传递给 decorator。如果运行这段代码，您应该会看到与第一个示例中生成的文件相同的文件。玩得开心！