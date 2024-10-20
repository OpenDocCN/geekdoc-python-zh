# Python 101:异常处理

> 原文：<https://www.blog.pythonlibrary.org/2012/09/12/python-101-exception-handling/>

Python 提供了强大的异常处理功能，并将其融入到语言中。异常处理是每个程序员都需要学习的东西。它允许程序员继续他们的程序，或者在异常发生后优雅地终止应用程序。Python 使用 try/except/finally 约定。我们将花一些时间了解标准异常，如何创建自定义异常，以及如何在调试时获取异常信息。

### 基本异常处理

首先，必须说明的是，通常不建议使用裸露的异常！你会看到它们被使用，我自己也不时地使用它们。一只光秃秃的除了看起来像这样:

```py

try:
    print monkey
except:
    print "This is an error message!"

```

如您所见，这将捕获所有异常，但是您并不真正了解异常的任何信息，这使得处理异常非常棘手。因为您不知道捕获了什么异常，所以您也不能打印出与用户非常相关的消息。另一方面，有时候我认为使用一个简单的 except 更容易。我的一位读者就此与我联系，他说数据库交互是一个很好的例子。我注意到，在处理数据库时，我似乎也遇到了大量的异常，所以我同意这一点。我认为当你在一行中遇到 3 或 4 个以上的异常处理程序时，最好使用一个空的 except，如果你需要回溯，有很多方法可以得到它(见最后一节)。

让我们来看几个常见的异常处理的例子。

```py

try:
    import spam
except ImportError:
    print "Spam for eating, not importing!"

try:
    print python
except NameError, e:
    print e

try:
    1 / 0
except ZeroDivisionError:
    print "You can't divide by zero!"

```

第一个例子捕获了一个 ImportError，它只在您导入 Python 找不到的内容时发生。如果您试图在没有实际安装 SQLAlchemy 的情况下导入它，或者拼错了模块或包名，您就会看到这种情况。捕捉 ImportErrors 的一个很好的用途是当您想要导入一个替代模块时。取 **md5** 模块。Python 2.5 中的[已被弃用](http://docs.python.org/library/md5.html)，因此如果您正在编写支持 2.5-3.x 的代码，那么您可能希望尝试导入 md5 模块，然后在导入失败时退回到较新的(也是推荐的)hashlib 模块。

下一个异常是 NameError。当一个变量还没有被定义时，你会得到这个。您还会注意到，我们在异常部分添加了一个逗号“e”。这让我们不仅可以捕捉到错误，还可以打印出它的信息部分，在本例中是:**name‘python’没有定义**。

第三个也是最后一个异常是一个如何捕捉零除法异常的例子。是的，用户仍然试图被零除，不管你告诉他们多少次不要这样做。所有这些异常，我们已经查看了我们的子类 **Exception** ，所以如果你要写一个**Exception**子句，你实际上是在写一个捕捉所有异常的 bare Exception。

现在，如果您想捕获多个错误，但不是所有错误，您会怎么做？还是不管发生了什么都做点什么？让我们来了解一下！

```py

try:
    5 + "python"
except TypeError:
    print "You can't add strings and integers!"
except WindowsError:
    print "A Windows error has occurred. Good luck figuring it out!"
finally:
    print "The finally section ALWAYS runs!"

```

在上面的代码中，你会看到在一个 **try** 下有两个 **except** 语句。您可以根据需要添加任意多的例外，并对每个例外做不同的处理。在 Python 2.5 及更新版本中，你实际上可以这样把异常串在一起: **except TypeError，WindowsError:** 或 **except (TypeError，WindowsError)** 。您还会注意到，我们有一个可选的 **finally** 子句，无论是否有异常，该子句都会运行。它有利于[清理动作](http://docs.python.org/tutorial/errors.html#defining-clean-up-actions)，比如关闭文件或数据库连接。你也可以做一个 try/except/else/finally 或者 try/except/else，但是 **else** 在实践中有点混乱，老实说，我从来没有见过它被使用。基本上，else 子句只在 except 没有执行时执行。如果您不想将一堆代码放在 **try** 部分，这可能会引发其他错误，那么这是非常方便的。更多信息参见[文档](http://docs.python.org/tutorial/errors.html#handling-exceptions)。

### 获取整个堆栈回溯

如果您想获得异常的完整回溯，该怎么办？Python 为此提供了一个模块。从逻辑上来说，这叫做**回溯**。这里有一个快速和肮脏的小例子:

```py

import traceback

try:
    with open("C:/path/to/py.log") as f:
        for line in f:
            print line
except IOError, exception:
    print exception
    print 
    print traceback.print_exc()

```

上面的代码将打印出异常文本，打印一个空行，然后使用回溯模块的 **print_exc** 方法打印整个回溯。traceback 模块中有许多其他方法，允许您格式化输出或获取堆栈跟踪的各个部分，而不是完整的部分。你应该查看一下[文档](http://docs.python.org/library/traceback.html)以获得更多的细节和例子。

还有另一种方法可以不使用回溯模块，至少不直接获得整个回溯。你可以使用 Python 的**日志**模块。这里有一个简单的例子:

```py

import logging

logging.basicConfig(filename="sample.log", level=logging.INFO)
log = logging.getLogger("ex")

try:
    raise RuntimeError
except RuntimeError, err:
    log.exception("RuntimeError!")

```

这将在运行脚本的同一目录下创建一个日志文件，包含以下内容:

```py

ERROR:ex:RuntimeError!
Traceback (most recent call last):
  File "C:\Users\mdriscoll\Desktop\exceptionLogger.py", line 7, in raise RuntimeError
RuntimeError 
```

如您所见，它将记录正在记录的级别(错误)、记录器名称(ex)和我们传递给它的消息(RuntimeError！)以及完整的追溯。您可能会发现这比使用回溯模块更方便。

### 创建自定义异常

当您编写复杂的程序时，您可能会发现需要创建自己的异常。幸运的是，Python 使得编写新异常的过程变得非常容易。这里有一个非常简单的例子:

```py

########################################################################
class ExampleException(Exception):
    pass

try:
    raise ExampleException("There are no droids here")
except ExampleException, e:
    print e

```

我们在这里所做的就是用一个空白的主体子类化异常类型。然后，我们通过在 try/except 子句中引发错误来测试它，并打印出我们传递给它的自定义错误消息。实际上，您可以通过引发任何异常来做到这一点: **raise NameError("这是一个不正确的名称！")**。与此相关，参见 pydanny 的[文章](http://pydanny.com/attaching-custom-exceptions-to-functions-and-classes.html)中关于将自定义异常附加到函数和类的内容。它非常好，展示了一些巧妙的技巧，使得使用您的自定义异常更加容易，而不必导入它们太多。

### 包扎

现在你应该知道如何成功地捕捉异常，获取回溯，甚至创建你自己的定制异常。您有工具让您的脚本继续运行，即使发生了不好的事情！当然，如果停电了，他们也不会帮忙，但你可以覆盖大多数情况。享受针对用户和他们使你的程序崩溃的各种方法强化你的代码的乐趣。

### 附加阅读

*   官方[异常教程](http://docs.python.org/py3k/tutorial/errors.html)
*   Python 的回溯模块[文档](http://docs.python.org/library/traceback.html)
*   StackOverflow: [在一行中捕获多个异常，除了](http://stackoverflow.com/questions/6470428/catch-multiple-exceptions-in-one-line-except-block)