# Python 日志记录:如何记录到多个位置

> 原文：<https://www.blog.pythonlibrary.org/2013/07/18/python-logging-how-to-log-to-multiple-locations/>

今天我决定弄清楚如何让 Python 同时记录到文件和控制台。大多数时候，我只是想记录一个文件，但偶尔我也想在控制台上看到一些东西，以帮助调试。我在 Python 文档中找到了这个[古老的例子](http://docs.python.org/release/2.5.2/lib/multiple-destinations.html)，并最终用它模拟了下面的脚本:

```py

import logging

#----------------------------------------------------------------------
def log(path, multipleLocs=False):
    """
    Log to multiple locations if multipleLocs is True
    """
    fmt_str = '%(asctime)s - %(name)s - %(message)s'
    formatter = logging.Formatter(fmt_str)

    logging.basicConfig(filename=path, level=logging.INFO,
                        format=fmt_str)

    if multipleLocs:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        logging.getLogger("").addHandler(console)

    logging.info("This is an informational message")
    try:
        1 / 0
    except ZeroDivisionError:
        logging.exception("You can't do that!")

    logging.critical("THIS IS A SHOW STOPPER!!!")

if __name__ == "__main__":
    log("sample.log") # log only to file
    log("sample2.log", multipleLocs=True) # log to file AND console!

```

正如您所看到的，当您将 True 传递给第二个参数时，该脚本将创建一个 **StreamHandler** ()的实例，然后您可以通过以下调用配置该实例并将其添加到当前记录器中:

```py

logging.getLogger("").addHandler(console)

```

这在 Linux 上工作得很好，但是在 Windows 7 上没有创建 **sample2.log** ,所以我必须修改 if 语句如下:

```py

if multipleLocs:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    fhandler = logging.FileHandler(path)
    fhandler.setFormatter(formatter)

    logging.getLogger("").addHandler(console)
    logging.getLogger("").addHandler(fhandler)

```

现在，我应该注意到这导致了一个相当奇怪的错误，因为 Python 以某种方式跟踪我在调用我的日志函数时写入的文件名，因此当我告诉它写入 sample2.log 时，它会写入 sample 2 . log 和原始的 sample.log。下面是一个正确工作的更新示例:

```py

import logging
import os

#----------------------------------------------------------------------
def log(path, multipleLocs=False):
    """
    Log to multiple locations if multipleLocs is True
    """
    fname = os.path.splitext(path)[0]
    logger = logging.getLogger("Test_logger_%s" % fname)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if multipleLocs:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    logger.info("This is an informational message")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("You can't do that!")

    logger.critical("THIS IS A SHOW STOPPER!!!")

if __name__ == "__main__":
    log("sample.log") # log only to file
    log("sample2.log", multipleLocs=True) # log to file AND console!

```

您会注意到，这一次我们将日志记录器的名称基于日志的文件名。日志模块非常灵活，玩起来很有趣。我希望你和我一样对此感兴趣。