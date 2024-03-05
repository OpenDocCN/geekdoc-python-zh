# 上下文管理器和 Python 的 with 语句

> 原文：<https://realpython.com/python-with-statement/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**上下文管理器和 Python 的 with 语句**](/courses/with-statement-python/)

Python 中的`with`语句对于正确管理程序中的外部资源是一个非常有用的工具。它允许您利用现有的**上下文管理器**来自动处理安装和拆卸阶段，无论您何时处理外部资源或需要这些阶段的操作。

此外，**上下文管理协议**允许你创建自己的上下文管理器，这样你就可以定制处理系统资源的方式。那么，`with`语句有什么用呢？

在本教程中，您将学习:

*   **Python `with`语句**是做什么的以及如何使用
*   什么是**上下文管理协议**
*   如何实现自己的**上下文管理器**

有了这些知识，你就可以编写更有表现力的代码，避免程序中的资源泄露和 T2。`with`语句通过抽象它们的功能并允许它们被分解和重用，帮助您实现一些常见的资源管理模式。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 用 Python 管理资源

你在编程中会遇到的一个常见问题是如何正确管理**外部资源**，比如[文件](https://realpython.com/working-with-files-in-python/)、[锁](https://en.wikipedia.org/wiki/Lock_(computer_science))，以及网络连接。有时，一个程序会永远保留那些资源，即使你不再需要它们。这种问题被称为[内存泄漏](https://en.wikipedia.org/wiki/Memory_leak)，因为每次创建和打开给定资源的新实例而不关闭现有实例时，可用内存都会减少。

合理管理资源通常是一个棘手的问题。它需要一个**安装**阶段和一个**拆卸**阶段。后一个阶段需要你执行一些清理动作，比如[关闭一个文件](https://realpython.com/why-close-file-python/)，释放一个锁，或者关闭一个网络连接。如果您忘记执行这些清理操作，那么您的应用程序将保持资源活动。这可能会危及宝贵的系统资源，如内存和网络带宽。

例如，当开发人员使用数据库时，一个常见的问题是程序不断创建新的连接而不释放或重用它们。在这种情况下，数据库[后端](https://en.wikipedia.org/wiki/Back-end_database)可以停止接受新的连接。这可能需要管理员登录并手动终止这些过时的连接，以使数据库再次可用。

另一个常见的问题出现在开发人员处理文件的时候。[将文本写入文件](https://realpython.com/read-write-files-python/)通常是一个缓冲操作。这意味着在文件上调用`.write()`不会立即导致将文本写入物理文件，而是写入临时缓冲区。有时，当缓冲区未满，开发人员忘记调用`.close()`时，部分数据可能会永远丢失。

另一种可能是你的应用程序遇到错误或[异常](https://realpython.com/python-exceptions/)，导致控制流绕过负责释放手头资源的代码。下面是一个使用 [`open()`](https://docs.python.org/3/library/functions.html#open) 将一些文本写入文件的例子:

```py
file = open("hello.txt", "w")
file.write("Hello, World!")
file.close()
```

如果在调用`.write()`的过程中出现异常，这个实现不能保证文件会被关闭。在这种情况下，代码永远不会调用`.close()`，因此您的程序可能会泄漏一个文件描述符。

在 Python 中，可以使用两种通用方法来处理资源管理。您可以将代码包装在:

1.  一个 [`try` … `finally`](https://realpython.com/python-exceptions/#cleaning-up-after-using-finally) 构念
2.  一个 [`with`](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement) 构造

第一种方法非常通用，允许您提供安装和拆卸代码来管理任何类型的资源。但是，有点啰嗦。此外，如果您忘记了任何清理操作怎么办？

第二种方法提供了一种简单的方式来提供和重用安装和拆卸代码。在这种情况下，您将受到限制，即`with`语句仅适用于[上下文管理器](https://docs.python.org/3/glossary.html#term-context-manager)。在接下来的两节中，您将学习如何在代码中使用这两种方法。

[*Remove ads*](/account/join/)

### `try`……`finally`接近

处理文件可能是编程中资源管理最常见的例子。在 Python 中，可以使用一个`try` … `finally`语句来正确地处理打开和关闭文件:

```py
# Safely open the file
file = open("hello.txt", "w")

try:
    file.write("Hello, World!")
finally:
    # Make sure to close the file after using it
    file.close()
```

在这个例子中，您需要安全地打开文件`hello.txt`，这可以通过在`try` … `except`语句中包装对`open()`的调用来实现。稍后，当你试图写`file`时，`finally`子句将保证`file`被正确关闭，即使在`try`子句中调用`.write()`的过程中出现异常。当您在 Python 中管理外部资源时，可以使用这种模式来处理安装和拆卸逻辑。

上例中的`try`块可能会引发异常，比如`AttributeError`或`NameError`。您可以像这样在`except`子句中处理这些异常:

```py
# Safely open the file
file = open("hello.txt", "w")

try:
    file.write("Hello, World!")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")
finally:
    # Make sure to close the file after using it
    file.close()
```

在本例中，您捕获了在写入文件时可能发生的任何潜在异常。在现实生活中，您应该使用特定的[异常类型](https://docs.python.org/3/library/exceptions.html#built-in-exceptions)而不是通用的 [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception) 来防止未知错误无声无息地传递。

### `with`语句接近

Python **`with`语句**创建了一个**运行时上下文**，允许你在[上下文管理器](https://docs.python.org/3/library/stdtypes.html#context-manager-types)的控制下运行一组语句。 [PEP 343](https://www.python.org/dev/peps/pep-0343/) 增加了`with`语句，以便能够分解出`try` … `finally`语句的标准用例。

与传统的`try` … `finally`结构相比，`with`语句可以让你的代码更加清晰、安全和可重用。[标准库中的很多类](https://docs.python.org/3/library/index.html)都支持`with`语句。一个经典的例子是 [`open()`](https://docs.python.org/3/library/functions.html#open) ，它允许你使用`with`来处理[文件对象](https://docs.python.org/3/glossary.html#term-file-object)。

要编写一个`with`语句，需要使用以下通用语法:

```py
with expression as target_var:
    do_something(target_var)
```

上下文管理器对象是在`with`之后评估`expression`的结果。换句话说，`expression`必须返回一个实现**上下文管理协议**的对象。该协议包括两种特殊方法:

1.  [`.__enter__()`](https://docs.python.org/3/library/stdtypes.html#contextmanager.__enter__) 被`with`语句调用进入运行时上下文。
2.  [`.__exit__()`](https://docs.python.org/3/library/stdtypes.html#contextmanager.__exit__) 在执行离开`with`代码块时被调用。

`as`说明符是可选的。如果您提供一个带有`as`的`target_var`，那么在上下文管理器对象上调用`.__enter__()`的[返回值](https://realpython.com/python-return-statement/)将被绑定到该变量。

**注意:**一些上下文管理器从`.__enter__()`返回`None`，因为它们没有有用的对象返回给调用者。在这些情况下，指定一个`target_var`没有意义。

下面是 Python 遇到`with`语句时该语句的处理方式:

1.  调用`expression`来获取上下文管理器。
2.  存储上下文管理器的`.__enter__()`和`.__exit__()`方法以备后用。
3.  在上下文管理器上调用`.__enter__()`,并将其返回值绑定到`target_var`(如果提供的话)。
4.  执行`with`代码块。
5.  当`with`代码块完成时，调用上下文管理器上的`.__exit__()`。

在这种情况下，`.__enter__()`，通常提供设置代码。`with`语句是一个[复合语句](https://docs.python.org/3/reference/compound_stmts.html#compound-statements)，它启动一个代码块，类似于[条件语句](https://realpython.com/python-conditional-statements/)或 [`for`循环](https://realpython.com/python-for-loop/)。在这个代码块中，可以运行几条语句。通常，如果适用的话，您可以使用`with`代码块来操作`target_var`。

一旦`with`代码块完成，`.__exit__()`就会被调用。这个方法通常提供拆卸逻辑或清理代码，比如在打开的文件对象上调用`.close()`。这就是为什么`with`语句如此有用。它使得正确获取和释放资源变得轻而易举。

下面是如何使用`with`语句打开`hello.txt`文件进行写入的方法:

```py
with open("hello.txt", mode="w") as file:
    file.write("Hello, World!")
```

当您运行这个`with`语句时，`open()`返回一个 [`io.TextIOBase`](https://docs.python.org/3/library/io.html#io.TextIOBase) 对象。这个对象也是一个上下文管理器，所以`with`语句调用`.__enter__()`，并将其返回值赋给`file`。然后，您可以在`with`代码块中操作该文件。当块结束时，`.__exit__()`会自动被调用并为您关闭文件，即使在`with`块中出现异常。

这个`with`构造比它的`try` … `finally`替代要短，但是也不那么通用，正如你已经看到的。您只能对支持上下文管理协议的对象使用`with`语句，而`try` … `finally`允许您对任意对象执行清理操作，而无需支持上下文管理协议。

在 Python 3.1 和更高版本中，`with`语句[支持多个上下文管理器](https://docs.python.org/3/whatsnew/3.1.html#other-language-changes)。您可以提供任意数量的上下文管理器，用逗号分隔:

```py
with A() as a, B() as b:
    pass
```

这类似于嵌套的`with`语句，但是没有嵌套。当您需要一次打开两个文件(第一个用于读取，第二个用于写入)时，这可能很有用:

```py
with open("input.txt") as in_file, open("output.txt", "w") as out_file:
    # Read content from input.txt
    # Transform the content
    # Write the transformed content to output.txt
    pass
```

在这个例子中，您可以添加代码来读取和转换`input.txt`的内容。然后你在同一个代码块里把最终结果写到`output.txt`里。

然而，在一个`with`中使用多个上下文管理器有一个缺点。如果您使用这个特性，那么您可能会突破您的行长度限制。要解决这个问题，您需要使用反斜杠(`\`)来延续行，因此您可能会得到一个难看的最终结果。

`with`语句可以使处理系统资源的代码更具可读性、可重用性和简洁，更不用说更安全了。它有助于避免 bug 和泄漏，因为它让你在使用资源后几乎不可能忘记清理、关闭和释放资源。

使用`with`允许您抽象出大部分资源处理逻辑。不需要每次都写一个带有安装和拆卸代码的明确的`try` … `finally`语句，`with`会为您处理这些并避免重复。

[*Remove ads*](/account/join/)

## 使用 Python `with`语句

只要 Python 开发人员将`with`语句整合到他们的编码实践中，该工具已经被证明有几个有价值的用例。Python 标准库中越来越多的对象现在提供了对上下文管理协议的支持，因此您可以在`with`语句中使用它们。

在本节中，您将编写一些示例，展示如何在标准库中和第三方库中的几个类中使用`with`语句。

### 使用文件

到目前为止，您已经使用了`open()`来提供上下文管理器，并在`with`结构中操作文件。通常推荐使用`with`语句打开文件，因为它确保打开的[文件描述符](https://en.wikipedia.org/wiki/File_descriptor)在执行流离开`with`代码块后自动关闭。

正如您之前看到的，使用`with`打开文件的最常见方式是通过内置的`open()`:

```py
with open("hello.txt", mode="w") as file:
    file.write("Hello, World!")
```

在这种情况下，由于上下文管理器在离开`with`代码块后关闭文件，一个常见的错误可能如下:

>>>

```py
>>> file = open("hello.txt", mode="w")

>>> with file:
...     file.write("Hello, World!")
...
13

>>> with file:
...     file.write("Welcome to Real Python!")
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: I/O operation on closed file.
```

第一个`with`成功将`"Hello, World!"`写入`hello.txt`。注意，`.write()`返回写入文件的字节数，`13`。然而，当你试图运行第二个`with`时，你会得到一个`ValueError`，因为你的`file`已经关闭。

使用`with`语句打开和管理文件的另一种方法是使用 [`pathlib.Path.open()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.open) :

>>>

```py
>>> import pathlib

>>> file_path = pathlib.Path("hello.txt")

>>> with file_path.open("w") as file:
...     file.write("Hello, World!")
...
13
```

[`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) 是一个表示你电脑中物理文件的具体路径的类。在指向物理文件的`Path`对象上调用`.open()`会像`open()`一样打开它。因此，`Path.open()`的工作方式类似于`open()`，但是文件路径是由您调用方法的`Path`对象自动提供的。

由于 [`pathlib`](https://realpython.com/python-pathlib/) 提供了一种优雅、简单且[Python 化的](https://realpython.com/learning-paths/writing-pythonic-code/)方式来操作文件系统路径，您应该考虑在您的`with`语句中使用`Path.open()`作为 Python 中的最佳实践。

最后，每当你加载一个外部文件时，你的程序应该检查可能的问题，比如一个丢失的文件，读写访问，等等。以下是您在处理文件时应该考虑使用的一般模式:

```py
import pathlib
import logging

file_path = pathlib.Path("hello.txt")

try:
    with file_path.open(mode="w") as file:
        file.write("Hello, World!")
except OSError as error:
    logging.error("Writing to file %s failed due to: %s", file_path, error)
```

在这个例子中，您将`with`语句包装在一个 [`try` … `except`语句](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions)中。如果在执行`with`的过程中出现了 [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) ，那么您可以使用 [`logging`](https://realpython.com/python-logging/) 用一条用户友好的描述性消息来记录错误。

### 遍历目录

[`os`](https://docs.python.org/3/library/os.html#module-os) 模块提供了一个名为 [`scandir()`](https://docs.python.org/3/library/os.html#os.scandir) 的函数，该函数返回给定目录中条目对应的 [`os.DirEntry`](https://docs.python.org/3/library/os.html#os.DirEntry) 对象的迭代器。这个函数是专门设计来在遍历目录结构时提供最佳性能的。

以给定目录的路径作为参数调用`scandir()`会返回一个支持上下文管理协议的迭代器:

>>>

```py
>>> import os

>>> with os.scandir(".") as entries:
...     for entry in entries:
...         print(entry.name, "->", entry.stat().st_size, "bytes")
...
Documents -> 4096 bytes
Videos -> 12288 bytes
Desktop -> 4096 bytes
DevSpace -> 4096 bytes
.profile -> 807 bytes
Templates -> 4096 bytes
Pictures -> 12288 bytes
Public -> 4096 bytes
Downloads -> 4096 bytes
```

在本例中，您编写了一个将`os.scandir()`作为上下文管理器供应商的`with`语句。然后你遍历所选目录中的条目(`"."`)，然后[在屏幕上打印出它们的名称和大小](https://realpython.com/python-print/)。在这种情况下，`.__exit__()`调用 [`scandir.close()`](https://docs.python.org/3/library/os.html#os.scandir.close) 关闭迭代器，释放获取的资源。请注意，如果在您的机器上运行此命令，您将得到不同的输出，这取决于您当前目录的内容。

[*Remove ads*](/account/join/)

### 执行高精度计算

与内置的[浮点数](https://realpython.com/python-numbers/#floating-point-numbers)不同， [`decimal`](https://docs.python.org/3/library/decimal.html#module-decimal) 模块提供了一种调整精度的方法，以便在涉及 [`Decimal`](https://docs.python.org/3/library/decimal.html#decimal.Decimal) 数字的给定计算中使用。精度默认为`28`位，但是您可以更改它以满足您的问题要求。使用`decimal`的 [`localcontext()`](https://docs.python.org/3/library/decimal.html#decimal.localcontext) 进行自定义精度计算的快速方法是:

>>>

```py
>>> from decimal import Decimal, localcontext

>>> with localcontext() as ctx:
...     ctx.prec = 42
...     Decimal("1") / Decimal("42")
...
Decimal('0.0238095238095238095238095238095238095238095')

>>> Decimal("1") / Decimal("42")
Decimal('0.02380952380952380952380952381')
```

这里，`localcontext()`提供了一个上下文管理器，它创建一个本地十进制上下文，并允许您使用自定义精度执行计算。在`with`代码块中，您需要将`.prec`设置为您想要使用的新精度，即上面示例中的`42`位置。当`with`代码块结束时，精度被重置回默认值，`28`位。

### 处理多线程程序中的锁

在 Python 标准库中有效使用`with`语句的另一个好例子是 [`threading.Lock`](https://docs.python.org/3/library/threading.html?highlight=threading#threading.Locks) 。这个类提供了一个原语锁，以防止多个线程在一个[多线程](https://realpython.com/intro-to-python-threading/)应用程序中同时修改一个共享资源。

您可以在一个`with`语句中使用一个`Lock`对象作为上下文管理器来自动获取和释放一个给定的锁。例如，假设您需要保护一个银行账户的余额:

```py
import threading

balance_lock = threading.Lock()

# Use the try ... finally pattern
balance_lock.acquire()
try:
    # Update the account balance here ...
finally:
    balance_lock.release()

# Use the with pattern
with balance_lock:
    # Update the account balance here ...
```

第二个例子中的`with`语句在执行流进入和离开语句时自动获取和释放一个锁。这样，您可以专注于代码中真正重要的东西，而忘记那些重复的操作。

在这个例子中，`with`语句中的锁创建了一个被称为[临界区](https://en.wikipedia.org/wiki/Critical_section)的受保护区域，它防止对账户余额的并发访问。

### 使用 pytest 测试异常

到目前为止，您已经使用 Python 标准库中可用的上下文管理器编写了几个示例。然而，一些第三方库包括支持上下文管理协议的对象。

假设你正在用 [pytest](https://realpython.com/pytest-python-testing/) 测试你的代码。您的一些函数和代码块在某些情况下会引发异常，您希望测试这些情况。为此，您可以使用 [`pytest.raises()`](https://docs.pytest.org/en/stable/reference.html#pytest.raises) 。此函数允许您断言代码块或函数调用会引发给定的异常。

因为`pytest.raises()`提供了一个上下文管理器，所以您可以像这样在`with`语句中使用它:

>>>

```py
>>> import pytest

>>> 1 / 0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ZeroDivisionError: division by zero

>>> with pytest.raises(ZeroDivisionError):
...     1 / 0
...

>>> favorites = {"fruit": "apple", "pet": "dog"}
>>> favorites["car"]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'car'

>>> with pytest.raises(KeyError):
...     favorites["car"]
...
```

在第一个例子中，您使用`pytest.raises()`来捕捉表达式`1 / 0`引发的 [`ZeroDivisionError`](https://docs.python.org/3/library/exceptions.html#ZeroDivisionError) 。第二个示例使用函数来捕获当您访问给定字典中不存在的键时引发的 [`KeyError`](https://realpython.com/python-keyerror/) 。

如果您的函数或代码块没有引发预期的异常，那么`pytest.raises()`会引发一个失败异常:

>>>

```py
>>> import pytest

>>> with pytest.raises(ZeroDivisionError):
...     4 / 2
...
2.0
Traceback (most recent call last):
  ...
Failed: DID NOT RAISE <class 'ZeroDivisionError'>
```

`pytest.raises()`的另一个很酷的特性是，您可以指定一个目标变量来检查引发的异常。例如，如果您想要验证错误消息，那么您可以这样做:

>>>

```py
>>> with pytest.raises(ZeroDivisionError) as exc:
...     1 / 0
...
>>> assert str(exc.value) == "division by zero"
```

您可以使用所有这些`pytest.raises()`特性来捕获您从[函数](https://realpython.com/defining-your-own-python-function/)和代码块中引发的异常。这是一个很酷很有用的工具，您可以将它整合到您当前的测试策略中。

[*Remove ads*](/account/join/)

## 总结`with`陈述的优点

为了总结到目前为止您所学到的内容，这里列出了在代码中使用 Python `with`语句的一系列好处:

*   使**资源管理**比其等价的`try` … `finally`语句更安全
*   封装了**上下文管理器**中`try` … `finally`语句的标准用法
*   允许重用自动管理给定操作的**设置**和**拆卸**阶段的代码
*   帮助避免**资源泄漏**

一致地使用`with`语句可以提高代码的总体质量，并通过防止资源泄漏问题使代码更加安全。

## 使用`async with`语句

`with`语句也有异步版本， [`async with`](https://docs.python.org/3/reference/compound_stmts.html?highlight=async#the-async-with-statement) 。您可以使用它来编写依赖于异步代码的上下文管理器。在这种代码中很容易看到`async with`，因为许多 [IO 操作](https://en.wikipedia.org/wiki/Input/output)都涉及安装和拆卸阶段。

例如，假设您需要编写一个异步函数来检查给定的站点是否在线。为此，您可以像这样使用 [`aiohttp`](https://docs.aiohttp.org/en/stable/index.html) 、 [`asyncio`](https://realpython.com/async-io-python/) 和`async with`:

```py
 1# site_checker_v0.py
 2
 3import aiohttp
 4import asyncio
 5
 6async def check(url):
 7    async with aiohttp.ClientSession() as session: 8        async with session.get(url) as response: 9            print(f"{url}: status -> {response.status}")
10            html = await response.text()
11            print(f"{url}: type -> {html[:17].strip()}")
12
13async def main():
14    await asyncio.gather(
15        check("https://realpython.com"),
16        check("https://pycoders.com"),
17    )
18
19asyncio.run(main())
```

下面是这个脚本的作用:

*   **第 3 行** [导入](https://realpython.com/python-import/) `aiohttp`，为`asyncio`和 Python 提供异步 HTTP 客户端和服务器。注意，`aiohttp`是一个第三方包，可以通过在命令行运行`python -m pip install aiohttp`来安装。
*   **第 4 行**导入`asyncio`，它允许你使用`async`和`await`语法编写[并发](https://realpython.com/python-concurrency/)代码。
*   **第 6 行**使用`async` [关键字](https://realpython.com/python-keywords/)将`check()`定义为异步函数。

在`check()`中，您定义了两个嵌套的`async with`语句:

*   **第 7 行**定义了一个外部`async with`，它实例化`aiohttp.ClientSession()`以获得一个上下文管理器。它将返回的对象存储在`session`中。
*   **第 8 行**定义了一个内部`async with`语句，使用`url`作为参数调用`session`上的`.get()`。这将创建第二个上下文管理器并返回一个`response`。
*   **第 9 行**打印手头`url`的响应[状态码](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes)。
*   **10 号线**在`response`上运行对`.text()`的唤醒调用，并将结果存储在`html`中。
*   **第 11 行**打印站点`url`及其文件类型， [`doctype`](https://en.wikipedia.org/wiki/Document_type_declaration) 。
*   **第 13 行**定义了脚本的 [`main()`](https://realpython.com/python-main-function/) 函数，也是一个[协程](https://docs.python.org/3/library/asyncio-task.html#coroutine)。
*   **第 14 行**从`asyncio`调用 [`gather()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather) 。该函数按顺序同时运行[个可应用对象](https://docs.python.org/3/library/asyncio-task.html#asyncio-awaitables)。在这个例子中，`gather()`用不同的 [URL](https://en.wikipedia.org/wiki/URL) 运行`check()`的两个实例。
*   **19 线**运行`main()`使用 [`asyncio.run()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.run) 。该函数创建一个新的`asyncio` [事件循环](https://docs.python.org/3/library/asyncio-eventloop.html)，并在操作结束时关闭它。

如果您从命令行运行这个脚本,那么您将得到类似如下的输出:

```py
$ python site_checker_v0.py
https://realpython.com: status -> 200
https://pycoders.com: status -> 200
https://pycoders.com: type -> <!doctype html>
https://realpython.com: type -> <!doctype html>
```

酷！您的脚本正常工作，并且您确认两个站点当前都可用。您还可以从每个站点的主页检索有关文档类型的信息。

**注意:**由于并发任务调度和网络延迟的不确定性，您的输出可能会略有不同。特别是，各行可以以不同的顺序出现。

`async with`语句的工作方式类似于常规的`with`语句，但是它需要一个**异步上下文管理器**。换句话说，它需要一个能够在其进入和退出方法中暂停执行的上下文管理器。异步上下文管理器实现特殊的方法 [`.__aenter__()`](https://docs.python.org/3/reference/datamodel.html#object.__aenter__) 和 [`.__aexit__()`](https://docs.python.org/3/reference/datamodel.html#object.__aexit__) ，它们对应于常规上下文管理器中的`.__enter__()`和`.__exit__()`。

`async with ctx_mgr`构造隐式地在进入上下文时使用`await ctx_mgr.__aenter__()`,在退出上下文时使用`await ctx_mgr.__aexit__()`。这无缝地实现了`async`上下文管理器行为。

## 创建自定义上下文管理器

您已经使用过标准库和第三方库中的上下文管理器。`open()`、`threading.Lock`、`decimal.localcontext()`或其他人没有什么特别或神奇的。它们只是返回实现上下文管理协议的对象。

您可以通过在基于**类的**上下文管理器中实现`.__enter__()`和`.__exit__()`特殊方法来提供相同的功能。您还可以使用标准库中的 [`contextlib.contextmanager`](https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager) 装饰器和适当编码的[生成器](https://realpython.com/introduction-to-python-generators/)函数创建定制的基于**函数的**上下文管理器。

一般来说，上下文管理器和`with`语句并不局限于资源管理。它们允许您提供和重用常见的安装和拆卸代码。换句话说，使用上下文管理器，您可以执行任何一对需要在另一个操作或过程的之前*和*之后*完成的操作，例如:*

*   打开和关闭
*   锁定并释放
*   更改和重置
*   创建和删除
*   进入和退出
*   开始和停止
*   安装和拆卸

您可以提供代码来安全地管理上下文管理器中的任何一对操作。然后，您可以在整个代码的`with`语句中重用该上下文管理器。这可以防止错误并减少重复的样板代码。它也让你的[API](https://realpython.com/python-api/)更安全、更干净、更用户友好。

在接下来的两节中，您将学习创建基于类和基于函数的上下文管理器的基础知识。

[*Remove ads*](/account/join/)

## 编码基于类的上下文管理器

要实现上下文管理协议并创建基于**类的**上下文管理器，您需要将`.__enter__()`和`__exit__()`特殊方法添加到您的类中。下表总结了这些方法的工作原理、它们采用的参数以及可以放入其中的逻辑:

| 方法 | 描述 |
| --- | --- |
| `.__enter__(self)` | 该方法处理设置逻辑，并在进入新的`with`上下文时被调用。它的返回值被绑定到`with`目标变量。 |
| `.__exit__(self, exc_type, exc_value, exc_tb)` | 该方法处理拆卸逻辑，并在执行流离开`with`上下文时被调用。如果发生异常，那么`exc_type`、`exc_value`和`exc_tb`分别保存异常类型、值和回溯信息。 |

当`with`语句执行时，它调用上下文管理器对象上的`.__enter__()`,表示您正在进入一个新的运行时上下文。如果您提供一个带有`as`说明符的目标变量，那么`.__enter__()`的返回值将被赋给该变量。

当执行流离开上下文时，`.__exit__()`被调用。如果`with`代码块中没有出现异常，那么`.__exit__()`的最后三个参数被设置为 [`None`](https://realpython.com/null-in-python/) 。否则，它们保存与当前异常相关联的类型、值和[回溯](https://realpython.com/python-traceback/)。

如果`.__exit__()`方法返回`True`，那么`with`块中发生的任何异常都会被吞掉，并在`with`之后的下一条语句处继续执行。如果`.__exit__()`返回`False`，那么异常被传播到上下文之外。当方法不显式返回任何内容时，这也是默认行为。您可以利用这个特性在上下文管理器中封装异常处理。

### 编写一个基于类的上下文管理器示例

这里有一个基于类的上下文管理器示例，它实现了两种方法，`.__enter__()`和`.__exit__()`。它还展示了 Python 如何在一个`with`构造中调用它们:

>>>

```py
>>> class HelloContextManager:
...     def __enter__(self):
...         print("Entering the context...")
...         return "Hello, World!"
...     def __exit__(self, exc_type, exc_value, exc_tb):
...         print("Leaving the context...")
...         print(exc_type, exc_value, exc_tb, sep="\n")
...

>>> with HelloContextManager() as hello:
...     print(hello)
...
Entering the context...
Hello, World!
Leaving the context...
None
None
None
```

`HelloContextManager`实现了`.__enter__()`和`.__exit__()`。在`.__enter__()`中，您首先打印一条消息，表示执行流正在进入一个新的上下文。然后你返回`"Hello, World!"`字符串。在`.__exit__()`中，您打印一条消息，表示执行流正在离开上下文。您还可以打印它的三个参数的内容。

当`with`语句运行时，Python 会创建一个新的`HelloContextManager`实例，并调用它的`.__enter__()`方法。你知道这一点是因为屏幕上印着`Entering the context...`。

**注意:**使用上下文管理器的一个常见错误是忘记调用传递给`with`语句的对象。

在这种情况下，语句无法获得所需的上下文管理器，您会得到这样一个`AttributeError`:

>>>

```py
>>> with HelloContextManager as hello:
...     print(hello)
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: __enter__
```

异常消息没有说太多，在这种情况下，您可能会感到困惑。所以，一定要调用`with`语句中的对象来提供相应的上下文管理器。

然后 Python 运行`with`代码块，将`hello`打印到屏幕上。注意，`hello`保存着`.__enter__()`的返回值。

当执行流退出`with`代码块时，Python 调用`.__exit__()`。你知道这一点是因为你把`Leaving the context...`印在了你的屏幕上。输出的最后一行确认了`.__exit__()`的三个参数被设置为`None`。

**注:**当你不记得`.__exit__()`的确切签名并且不需要访问它的参数时，一个常用的技巧是使用 [`*args`和`**kwargs`](https://realpython.com/python-kwargs-and-args/) ，就像在`def __exit__(self, *args, **kwargs):`中一样。

现在，如果在执行`with`块的过程中出现异常，会发生什么？继续编写下面的`with`语句:

>>>

```py
>>> with HelloContextManager() as hello:
...     print(hello)
...     hello[100]
...
Entering the context...
Hello, World!
Leaving the context...
<class 'IndexError'>
string index out of range
<traceback object at 0x7f0cebcdd080>
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
IndexError: string index out of range
```

在这种情况下，您尝试在[字符串](https://realpython.com/python-strings/) `"Hello, World!"`中检索索引`100`处的值。这引发了一个`IndexError`，并且`.__exit__()`的参数设置如下:

*   **`exc_type`** 是例外类，`IndexError`。
*   **`exc_value`** 是例外的实例。
*   **`exc_tb`** 是追溯对象。

当您想要在上下文管理器中封装异常处理时，这种行为非常有用。

[*Remove ads*](/account/join/)

### 在上下文管理器中处理异常

作为在上下文管理器中封装异常处理的一个例子，假设您希望在使用`HelloContextManager`时`IndexError`是最常见的异常。您可能希望在上下文管理器中处理该异常，这样就不必在每个`with`代码块中重复异常处理代码。在这种情况下，您可以这样做:

```py
# exc_handling.py

class HelloContextManager:
    def __enter__(self):
        print("Entering the context...")
        return "Hello, World!"

    def __exit__(self, exc_type, exc_value, exc_tb):
        print("Leaving the context...")
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in your with block: {exc_type}")
            print(f"Exception message: {exc_value}")
            return True

with HelloContextManager() as hello:
    print(hello)
    hello[100]

print("Continue normally from here...")
```

在`.__exit__()`中，你检查`exc_value`是否是`IndexError`的一个实例。如果是这样，那么您打印几条信息性消息，最后用`True`返回。返回一个[真值](https://realpython.com/python-operators-expressions/#evaluation-of-non-boolean-values-in-boolean-context)使得在`with`代码块之后，可以吞下异常并继续正常执行。

在这个例子中，如果没有`IndexError`发生，那么该方法返回`None`并且异常传播出去。然而，如果你想更明确，那么你可以从`if`块外面返回`False`。

如果您从命令行运行`exc_handling.py`，那么您会得到以下输出:

```py
$ python exc_handling.py
Entering the context...
Hello, World!
Leaving the context...
An exception occurred in your with block: <class 'IndexError'>
Exception message: string index out of range
Continue normally from here...
```

`HelloContextManager`现在能够处理发生在`with`代码块中的`IndexError`异常。因为当一个`IndexError`发生时你返回`True`，执行流程在下一行继续，就在退出`with`代码块之后。

### 打开文件进行写入:第一版

既然您已经知道了如何实现上下文管理协议，那么您可以通过编写一个实际的例子来了解这一点。下面是如何利用`open()`来创建一个打开文件进行写入的上下文管理器:

```py
# writable.py

class WritableFile:
    def __init__(self, file_path):
        self.file_path = file_path

    def __enter__(self):
        self.file_obj = open(self.file_path, mode="w")
        return self.file_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_obj:
            self.file_obj.close()
```

`WritableFile`实现了上下文管理协议，支持`with`语句，就像最初的`open()`一样，但是它总是使用`"w"`模式打开文件进行写入。以下是如何使用新的上下文管理器:

>>>

```py
>>> from writable import WritableFile

>>> with WritableFile("hello.txt") as file:
...    file.write("Hello, World!")
...
```

运行这段代码后，您的`hello.txt`文件包含了`"Hello, World!"`字符串。作为一个练习，你可以写一个补充的上下文管理器来打开文件进行阅读，但是使用`pathlib`功能。去试试吧！

### 重定向标准输出

当您编写自己的上下文管理器时，需要考虑一个微妙的细节，即有时您没有从`.__enter__()`返回的有用对象，因此无法分配给`with`目标变量。在那些情况下，你可以显式返回 [`None`](https://realpython.com/null-in-python/) 或者你可以只依赖 Python 的[隐式返回](https://realpython.com/python-return-statement/#implicit-return-statements)值，也就是`None`。

例如，假设您需要将标准输出 [`sys.stdout`](https://docs.python.org/3/library/sys.html#sys.stdout) 临时重定向到磁盘上的给定文件。为此，您可以创建一个上下文管理器，如下所示:

```py
# redirect.py

import sys

class RedirectedStdout:
    def __init__(self, new_output):
        self.new_output = new_output

    def __enter__(self):
        self.saved_output = sys.stdout
        sys.stdout = self.new_output

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.saved_output
```

这个上下文管理器通过它的构造函数获取一个文件对象。在`.__enter__()`中，您将标准输出`sys.stdout`重新分配给一个实例属性，以避免丢失对它的引用。然后重新分配标准输出，使其指向磁盘上的文件。在`.__exit__()`中，你只是将标准输出恢复到它的原始值。

要使用`RedirectedStdout`，您可以这样做:

>>>

```py
>>> from redirect import RedirectedStdout

>>> with open("hello.txt", "w") as file:
...     with RedirectedStdout(file):
...         print("Hello, World!")
...     print("Back to the standard output...")
...
Back to the standard output...
```

本例中的外层`with`语句提供了一个文件对象，您将使用它作为新的输出，`hello.txt`。内部的`with`临时将标准输出重定向到`hello.txt`，因此对`print()`的第一次调用直接写入该文件，而不是在屏幕上打印`"Hello, World!"`。注意，当你离开内部的`with`代码块时，标准输出会回到它的初始值。

`RedirectedStdout`是一个上下文管理器的简单例子，它没有从`.__enter__()`返回有用的值。然而，如果您只是重定向`print()`输出，您可以获得相同的功能，而不需要编写上下文管理器。你只需要像这样给`print()`提供一个`file`参数:

>>>

```py
>>> with open("hello.txt", "w") as file:
...     print("Hello, World!", file=file)
...
```

在这个例子中，`print()`将您的`hello.txt`文件作为一个参数。这使得`print()`直接写入你磁盘上的物理文件，而不是将`"Hello, World!"`打印到你的屏幕上。

[*Remove ads*](/account/join/)

### 测量执行时间

就像其他类一样，上下文管理器可以封装一些内部的[状态](https://en.wikipedia.org/wiki/State_(computer_science))。下面的例子展示了如何创建一个**有状态**上下文管理器来测量给定代码块或函数的执行时间:

```py
# timing.py

from time import perf_counter

class Timer:
    def __enter__(self):
        self.start = perf_counter()
        self.end = 0.0
        return lambda: self.end - self.start

    def __exit__(self, *args):
        self.end = perf_counter()
```

当你在一个`with`语句中使用`Timer`时，就会调用`.__enter__()`。该方法使用 [`time.perf_counter()`](https://docs.python.org/3/library/time.html#time.perf_counter) 获取`with`代码块开头的时间，并存储在`.start`中。它还初始化`.end`并返回计算时间增量的 [`lambda`函数](https://realpython.com/python-lambda/)。在这种情况下，`.start`保持初始状态或时间测量。

**注意:**要深入了解如何为代码计时，请查看 [Python 计时器函数:三种监控代码的方法](https://realpython.com/python-timer/)。

一旦`with`块结束，`.__exit__()`就会被调用。该方法获取块结束时的时间，并更新`.end`的值，以便`lambda`函数可以计算运行`with`代码块所需的时间。

下面是如何在代码中使用这个上下文管理器:

>>>

```py
>>> from time import sleep
>>> from timing import Timer

>>> with Timer() as timer:
...     # Time-consuming code goes here...
...     sleep(0.5)
...

>>> timer()
0.5005456680000862
```

使用`Timer`，你可以测量任何一段代码的执行时间。在这个例子中，`timer`保存了计算时间增量的`lambda`函数的一个实例，所以您需要调用`timer()`来获得最终结果。

## 创建基于功能的上下文管理器

Python 的[生成器函数](https://realpython.com/introduction-to-python-generators/)和 [`contextlib.contextmanager`](https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager) 装饰器提供了实现上下文管理协议的另一种便捷方式。如果您用`@contextmanager`修饰一个适当编码的生成器函数，那么您会得到一个基于**函数的**上下文管理器，它自动提供所需的方法`.__enter__()`和`.__exit__()`。这可以为您节省一些样板代码，让您的生活更加愉快。

使用`@contextmanager`和生成器函数创建上下文管理器的一般模式如下:

>>>

```py
>>> from contextlib import contextmanager

>>> @contextmanager
... def hello_context_manager():
...     print("Entering the context...")
...     yield "Hello, World!"
...     print("Leaving the context...")
...

>>> with hello_context_manager() as hello:
...     print(hello)
...
Entering the context...
Hello, World!
Leaving the context...
```

在本例中，您可以在`hello_context_manager()`中识别两个可见部分。在`yield`语句之前，有 setup 部分。在那里，您可以放置获取托管资源的代码。当执行流进入上下文时，`yield`之前的一切都开始运行。

在`yield`语句之后，有一个 teardown 部分，您可以在其中释放资源并进行清理。`yield`之后的代码运行在`with`块的末尾。`yield`语句本身提供了将被分配给`with`目标变量的对象。

这种实现和使用上下文管理协议的实现实际上是等效的。根据你觉得哪一个更有可读性，你可能会更喜欢其中一个。基于函数的实现的缺点是它需要理解高级 Python 主题，比如[装饰器](https://realpython.com/primer-on-python-decorators/)和生成器。

`@contextmanager`装饰器减少了创建上下文管理器所需的样板文件。不用用`.__enter__()`和`.__exit__()`方法写整个类，你只需要用一个`yield`实现一个生成器函数，它产生你想要`.__enter__()`返回的任何东西。

### 打开文件进行写入:第二版

您可以使用`@contextmanager`来重新实现您的`WritableFile`上下文管理器。下面是用这种技术重写后的样子:

>>>

```py
>>> from contextlib import contextmanager

>>> @contextmanager
... def writable_file(file_path):
...     file = open(file_path, mode="w")
...     try:
...         yield file
...     finally:
...         file.close()
...

>>> with writable_file("hello.txt") as file:
...     file.write("Hello, World!")
...
```

在这种情况下，`writable_file()`是一个打开`file`进行写入的生成器函数。然后它暂时挂起自己的执行，**让出**资源，这样`with`可以将它绑定到它的目标变量。当执行流程离开`with`代码块时，函数继续执行并正确关闭`file`。

[*Remove ads*](/account/join/)

### 嘲笑时间

作为如何使用`@contextmanager`创建定制上下文管理器的最后一个例子，假设您正在测试一段使用时间测量的代码。代码使用 [`time.time()`](https://docs.python.org/3/library/time.html#time.time) 来获得当前的时间测量值并做一些进一步的计算。由于时间度量不同，您决定模仿`time.time()`，这样您就可以测试您的代码。

这里有一个基于函数的上下文管理器可以帮你做到这一点:

>>>

```py
>>> from contextlib import contextmanager
>>> from time import time

>>> @contextmanager
... def mock_time():
...     global time
...     saved_time = time
...     time = lambda: 42
...     yield
...     time = saved_time
...

>>> with mock_time():
...     print(f"Mocked time: {time()}")
...
Mocked time: 42

>>> # Back to normal time
>>> time()
1616075222.4410584
```

在`mock_time()`中，你使用一个 [`global`语句](https://realpython.com/python-scope-legb-rule/#the-global-statement)来表示你将要修改全局名`time`。然后，您将原始的`time()`函数对象保存在`saved_time`中，这样您可以在以后安全地恢复它。下一步是使用一个总是返回相同值`42`的`lambda`函数来[猴子补丁](https://en.wikipedia.org/wiki/Monkey_patch) `time()`。

裸露的`yield`语句指定这个上下文管理器没有有用的对象发送回`with`目标变量供以后使用。在`yield`之后，您将全局`time`重置为其原始内容。

当执行进入`with`块时，任何对`time()`的调用都返回`42`。一旦离开`with`代码块，对`time()`的调用将返回预期的当前时间。就是这样！现在您可以测试与时间相关的代码了。

## 用上下文管理器编写好的 APIs】

上下文管理器非常灵活，如果您创造性地使用`with`语句，那么您可以为您的类、[模块和包](https://realpython.com/python-modules-packages/)定义方便的 API。

例如，如果您想要管理的资源是某种报告生成器应用程序中的**文本缩进级别**该怎么办？在这种情况下，您可以编写如下代码:

```py
with Indenter() as indent:
    indent.print("hi!")
    with indent:
        indent.print("hello")
        with indent:
            indent.print("bonjour")
    indent.print("hey")
```

这读起来几乎像是一种用于缩进文本的[领域特定语言(DSL)](https://en.wikipedia.org/wiki/Domain-specific_language) 。另外，请注意这段代码如何多次进入和离开同一个上下文管理器，以便在不同的缩进级别之间切换。运行此代码片段会产生以下输出，并打印出格式整齐的文本:

```py
hi!
    hello
        bonjour
hey
```

如何实现上下文管理器来支持这一功能？这可能是一个很好的练习，让你了解上下文管理器是如何工作的。因此，在您检查下面的实现之前，您可能需要一些时间，尝试自己解决这个问题，作为一个学习练习。

准备好了吗？下面是如何使用上下文管理器类实现此功能:

```py
class Indenter:
    def __init__(self):
        self.level = -1

    def __enter__(self):
        self.level += 1
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.level -= 1

    def print(self, text):
        print("    " * self.level + text)
```

在这里，每当执行流进入上下文时，`.__enter__()`将`.level`增加`1`。该方法还返回当前实例`self`。在`.__exit__()`中，你减少`.level`，这样每次退出上下文，打印的文本都后退一级。

这个例子中的关键点是从`.__enter__()`返回`self`允许您在几个嵌套的`with`语句中重用同一个上下文管理器。这将在每次进入和离开给定的上下文时改变文本的缩进级别。

此时，对您来说，一个很好的练习是编写这个上下文管理器的基于函数的版本。来吧，试一试！

## 创建异步上下文管理器

要创建异步上下文管理器，您需要定义`.__aenter__()`和`.__aexit__()`方法。下面的脚本是您之前看到的原始脚本`site_checker_v0.py`的重新实现，但是这次您提供了一个定制的异步上下文管理器来包装会话创建和关闭功能:

```py
# site_checker_v1.py

import aiohttp
import asyncio

class AsyncSession:
    def __init__(self, url):
        self._url = url

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        response = await self.session.get(self._url)
        return response

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        await self.session.close()

async def check(url):
    async with AsyncSession(url) as response:
        print(f"{url}: status -> {response.status}")
        html = await response.text()
        print(f"{url}: type -> {html[:17].strip()}")

async def main():
    await asyncio.gather(
        check("https://realpython.com"),
        check("https://pycoders.com"),
    )

asyncio.run(main())
```

此脚本的工作方式与其先前的版本`site_checker_v0.py`相似。主要区别在于，在本例中，您提取了原始外部`async with`语句的逻辑，并将其封装在`AsyncSession`中。

在`.__aenter__()`中，您创建一个`aiohttp.ClientSession()`，等待`.get()`响应，最后返回响应本身。在`.__aexit__()`中，您关闭会话，这对应于这个特定情况下的拆卸逻辑。注意`.__aenter__()`和`.__aexit__()`必须返回一个合适的对象。换句话说，您必须用`async def`来定义它们，这将返回一个根据定义可调用的协程对象。

如果您从命令行运行该脚本，那么您会得到类似如下的输出:

```py
$ python site_checker_v1.py
https://realpython.com: status -> 200
https://pycoders.com: status -> 200
https://realpython.com: type -> <!doctype html>
https://pycoders.com: type -> <!doctype html>
```

太好了！您的脚本就像它的第一个版本一样工作。它同时向两个站点发送 [`GET`请求](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol#Request_methods)，并处理相应的响应。

最后，在编写异步上下文管理器时，通常的做法是实现四种特殊的方法:

1.  `.__aenter__()`
2.  `.__aexit__()`
3.  `.__enter__()`
4.  `.__exit__()`

这使得您的上下文管理器可以与两种版本的`with`一起使用。

[*Remove ads*](/account/join/)

## 结论

Python **`with`语句**是管理程序中外部资源的强大工具。然而，它的用例并不局限于资源管理。您可以使用`with`语句以及现有的和定制的上下文管理器来处理给定流程或操作的设置和拆除阶段。

底层的**上下文管理协议**允许您创建定制的上下文管理器，并分解设置和拆卸逻辑，以便您可以在代码中重用它们。

**在本教程中，您学习了:**

*   **Python `with`语句**是做什么的以及如何使用
*   什么是**上下文管理协议**
*   如何实现自己的**上下文管理器**

有了这些知识，你就能写出安全、简洁、有表现力的代码。您还可以避免程序中的资源泄漏。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**上下文管理器和 Python 的 with 语句**](/courses/with-statement-python/)***********