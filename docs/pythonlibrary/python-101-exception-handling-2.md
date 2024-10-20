# Python 101 -异常处理

> 原文：<https://www.blog.pythonlibrary.org/2020/06/17/python-101-exception-handling-2/>

开发软件是一项艰苦的工作。为了让你的软件变得更好，你的应用程序需要保持工作，即使发生了意想不到的事情。例如，假设您的应用程序需要从互联网上下载信息。如果使用您的应用程序的人失去了互联网连接，会发生什么？

另一个常见的问题是，如果用户输入了无效的输入，该怎么办。或者试图打开应用程序不支持的文件。

所有这些情况都可以使用 Python 内置的异常处理功能来处理，这些功能通常被称为`try`和`except`语句。

在本文中，您将了解到:

*   常见例外
*   处理异常
*   引发异常
*   检查异常对象
*   使用`finally`语句
*   使用`else`语句

让我们从了解一些最常见的异常开始。

### 最常见的例外

Python 支持许多不同的异常。当你第一次开始使用这种语言时，你可能会看到下面的一些:

*   所有其他异常所基于的基础异常
*   `AttributeError` -当属性引用或赋值失败时引发。
*   `ImportError` -当导入语句找不到模块定义或当...导入找不到要导入的名称。
*   `ModuleNotFoundError`-import error 的一个子类，当模块无法定位时，由 import 引发
*   `IndexError` -当序列下标超出范围时引发。
*   `KeyError` -在现有键集中找不到映射(字典)键时引发。
*   `KeyboardInterrupt` -当用户点击中断键时触发(通常为`Control-C`或`Delete`)。
*   `NameError` -找不到本地或全局名称时引发。
*   `OSError` -当函数返回与系统相关的错误时引发。
*   `RuntimeError` -当检测到不属于任何其他类别的错误时引发。
*   `SyntaxError` -当解析器遇到语法错误时引发。
*   `TypeError` -当一个操作或函数被应用到一个不合适类型的对象时引发。关联的值是一个字符串，给出关于类型不匹配的详细信息。
*   `ValueError` -当内置操作或函数接收到类型正确但值不正确的参数，并且这种情况没有通过更精确的异常(如 IndexError)来描述时引发。
*   `ZeroDivisionError`—当除法或模运算的第二个参数为零时引发。

有关内置异常的完整列表，您可以在此处查看 Python 文档:

*   [https://docs.python.org/3/library/exceptions.html](https://docs.python.org/3/library/exceptions.html)。

现在让我们来看看当一个异常发生时，实际上如何处理它。

### 处理异常

Python 提供了一种特殊的语法，可以用来捕捉异常。它被称为`try/except`语句。

这是您将用于捕捉异常的基本形式:

```py
try:
    # Code that may raise an exception goes here
except ImportError:
    # Code that is executed when an exception occurs
```

您将您认为可能有问题的代码放在`try`块中。这可能是打开文件的代码，也可能是从用户那里获得输入的代码。第二个模块被称为`except`模块。这段代码只有在出现`ImportError`时才会被执行。

当你在没有指定异常类型的情况下编写`except`时，它被称为**裸异常**。不建议使用这些方法:

```py
try:
    with open('example.txt') as file_handler:
        for line in file_handler:
            print(line)
except:
    print('An error occurred')
```

创建一个空的异常是不好的做法，因为你不知道你正在捕捉什么类型的异常。这使得找出你做错了什么变得更加困难。如果您将异常类型缩小到您期望的类型，那么意外的类型实际上会使您的应用程序崩溃，并显示有用的消息。

此时，您可以决定是否要捕捉其他条件。

假设您想要捕获多个异常。有一种方法可以做到:

```py
try:
    with open('example.txt') as file_handler:
        for line in file_handler:
            print(line)
    import something
except OSError:
    print('An error occurred')
except ImportError:
    print('Unknown import!')
```

这个异常处理程序将捕获两种类型的异常:`OSError`和`ImportError`。如果发生另一种类型的异常，这个处理程序将不会捕捉到它，您的代码将会停止。

通过这样做，您可以将上面的代码重写得简单一点:

```py
try:
    with open('example.txt') as file_handler:
        for line in file_handler:
            print(line)
    import something
except (OSError, ImportError):
    print('An error occurred')
```

当然，通过创建异常元组，这将混淆哪个异常已经发生。换句话说，这段代码使得知道发生了哪个异常变得更加困难。

### 引发异常

在你捕捉到一个异常后你会怎么做？你有几个选择。您可以像前面的例子一样打印出一条消息。您还可以将消息记录到日志文件中。或者，如果您知道该异常需要停止应用程序的执行，您可以重新引发该异常。

引发异常是强制异常发生的过程。你在特殊情况下提出例外。例如，如果应用程序进入不良状态，您可能会引发一个异常。在已经处理了异常之后，您还会倾向于引发异常。

您可以使用 Python 的内置`raise`语句来引发异常:

```py
try:
    raise ImportError
except ImportError:
    print('Caught an ImportError')
```

当您引发异常时，您可以让它打印出一条自定义消息:

```py
>>> raise Exception('Something bad happened!')
Traceback (most recent call last):
  Python Shell, prompt 1, line 1
builtins.Exception: Something bad happened!
```

如果不提供消息，则异常如下所示:

```py
>>> raise Exception
Traceback (most recent call last):
  Python Shell, prompt 2, line 1
builtins.Exception:
```

现在让我们来了解一下异常对象！

### 检查异常对象

当异常发生时，Python 会创建一个异常对象。您可以通过使用`as`语句将异常对象赋给一个变量来检查它:

```py
>>> try:
...     raise ImportError('Bad import')
... except ImportError as error:
...     print(type(error))
...     print(error.args)
...     print(error)
... 
<class 'ImportError'>
('Bad import',)
Bad import
```

在本例中，您将`ImportError`对象分配给了`error`。现在您可以使用 Python 的`type()`函数来了解它是哪种异常。这将允许您解决本文前面提到的问题，当您有一个异常元组，但您不能立即知道您捕获了哪个异常。

如果您想更深入地调试异常，您应该查找 Python 的`traceback`模块。

### 使用`finally`语句

除了`try`和`except`，还有更多关于`try/except`的陈述。您也可以向它添加一个`finally`语句。`finally`语句是一个代码块，即使在`try`语句中出现异常，它也会一直运行。

您可以使用`finally`语句进行清理。例如，您可能需要关闭数据库连接或文件句柄。为此，您可以将代码包装在一个`try/except/finally`语句中。

让我们看一个人为的例子:

```py
>>> try:
...     1 / 0
... except ZeroDivisionError:
...     print('You can not divide by zero!')
... finally:
...     print('Cleaning up')
... 
You can not divide by zero!
Cleaning up
```

这个例子演示了如何处理`ZeroDivisionError`异常以及添加清理代码。

但是您也可以完全跳过`except`语句，而是创建一个`try/finally`:

```py
>>> try:
...     1/0
... finally:
...     print('Cleaning up')
... 
Cleaning upTraceback (most recent call last):
  Python Shell, prompt 6, line 2
builtins.ZeroDivisionError: division by zero
```

这次您不处理`ZeroDivisionError`异常，但是`finally`语句的代码块仍然运行。

### 使用`else`语句

还有一个语句可以用于 Python 的异常处理，那就是`else`语句。当没有异常时，可以使用`else`语句来执行代码。

这里有一个例子:

```py
>>> try:
...     print('This is the try block')
... except IOError:
...     print('An IOError has occurred')
... else:
...     print('This is the else block')
... 
This is the try block
This is the else block
```

在这段代码中，没有发生异常，所以`try`块和`else`块都运行。

让我们试着提高一个`IOError`，看看会发生什么:

```py
>>> try:
...     raise IOError
...     print('This is the try block')
... except IOError:
...     print('An IOError has occurred')
... else:
...     print('This is the else block')
... 
An IOError has occurred
```

由于出现异常，只有`try`和`except`模块运行。注意，`try`块在`raise`语句处停止运行。它根本没有到达`print()`函数。一旦出现异常，下面的所有代码都会被跳过，直接进入异常处理代码。

### 包扎

现在您知道了使用 Python 内置异常处理的基本知识。在本文中，您了解了以下主题:

*   常见例外
*   处理异常
*   引发异常
*   检查异常对象
*   使用`finally`语句
*   使用`else`语句

学习如何有效地捕捉异常需要练习。一旦你学会了如何捕捉异常，你将能够强化你的代码，使它以一种更好的方式工作，即使发生了意想不到的事情。

### 相关阅读

*   Python 101: [条件语句](https://www.blog.pythonlibrary.org/2020/04/29/python-101-conditional-statements/)
*   Python 101: [学习循环](https://www.blog.pythonlibrary.org/2020/05/27/python-101-learning-about-loops/)
*   Python 101: [了解集合](https://www.blog.pythonlibrary.org/2020/04/28/python-101-learning-about-sets/)