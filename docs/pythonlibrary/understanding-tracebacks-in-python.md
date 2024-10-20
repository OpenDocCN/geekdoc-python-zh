# 理解 Python 中的回溯

> 原文：<https://www.blog.pythonlibrary.org/2018/07/24/understanding-tracebacks-in-python/>

当你第一次开始学习如何编程时，你首先想知道的事情之一是错误信息意味着什么。在 Python 中，错误消息通常被称为**回溯**。以下是一些常见的回溯错误:

*   句法误差
*   ImportError or ModuleNotFoundError
*   属性错误
*   NameError

当你得到一个错误时，通常建议你回溯(即回溯)。所以从回溯的底部开始，倒着读。

让我们看几个 Python 中回溯的简单例子。

* * *

### 句法误差

一个非常常见的错误(或异常)是语法错误。当程序员在写出代码时出错时，就会发生语法错误。例如，他们可能会忘记关闭左括号，或者偶然在字符串周围使用引号。让我们来看一个我在空闲时运行的例子:

```py

>>> print('This is a test)

SyntaxError: EOL while scanning string literal

```

在这里，我们试图打印出一个字符串，我们收到一个语法错误。它告诉我们，错误与它没有找到行尾(EOL)有关。在这种情况下，我们没有用单引号结束字符串。

让我们看看另一个会引发语法错误的例子:

```py

def func
    return 1

```

当您从命令行运行此代码时，您将收到以下消息:

```py

File "syn.py", line 1
  def func
           ^
SyntaxError: invalid syntax

```

这里 SyntaxError 说我们使用了“无效语法”。然后 Python 使用一个箭头(^)来指出我们弄乱语法的确切位置，这很有帮助。最后，我们了解到我们需要查看的代码行在“第 1 行”。利用所有这些事实，我们可以很快发现我们忘记了在函数定义的末尾添加一对括号和一个冒号。

* * *

### 导入错误

我看到的另一个常见错误是 **ImportError** ，即使是有经验的开发人员也是如此。每当 Python 找不到您试图导入的模块时，您就会看到这个错误。这里有一个例子:

```py

>>> import some
Traceback (most recent call last):
  File "", line 1, in <module>ImportError: No module named some
```

这里我们了解到 Python 无法找到“some”模块。注意，在 Python 3 中，您可能会得到一个 **ModuleNotFoundError** 错误，而不是 ImportError。ModuleNotFoundError 只是 ImportError 的一个子类，实际上意思相同。不管您最终看到哪个异常，您看到这个错误的原因是因为 Python 找不到模块或包。这实际上意味着模块要么安装不正确，要么根本没有安装。大多数时候，你只需要弄清楚这个模块是哪个包的一部分，然后用 pip 或 conda 安装它。

* * *

### 属性错误

这个 **AttributeError** 真的很容易被意外击中，尤其是如果你的 IDE 中没有代码完成的话。当您尝试调用一个不存在的属性时，会出现此错误:

```py

>>> my_string = 'Python'  
>>> my_string.up()    
Traceback (most recent call last):
  File "", line 1, in <module>my_string.up()
AttributeError: 'str' object has no attribute 'up'
```

在这里，我试图使用一个不存在的字符串方法“up ”,而我应该调用“upper”。基本上，这个问题的解决方案是阅读手册或检查数据类型，并确保调用手边对象的正确属性。

* * *

### NameError

当找不到本地或全局名称时，出现**名称错误**。如果你是编程新手，这种解释似乎很模糊。这是什么意思？在这种情况下，这意味着你试图与一个变量或者一个还没有定义的对象进行交互。让我们假设您打开一个 Python 解释器，并键入以下内容:

```py

>>> print(var)

Traceback (most recent call last):
  File "", line 1, in <module>print(var)
NameError: name 'var' is not defined
```

这里你发现**‘var’并没有定义**。这很容易解决，因为我们需要做的就是将“var”设置为某个值。让我们来看看:

```py

>>> var = 'Python'

>>> print(var)

Python

```

看到这有多简单了吗？

* * *

### 包扎

在 Python 中你会看到很多错误，知道如何诊断这些错误的原因对于调试非常有用。很快，它将成为你的第二天性，你将能够只看一眼追溯，就知道到底发生了什么。Python 中还有许多其他内置异常，记录在他们的网站上，我鼓励你熟悉它们，这样你就知道它们的意思了。大多数时候，这应该是显而易见的。

* * *

### 相关阅读

*   关于[错误和异常](https://docs.python.org/3/tutorial/errors.html)的 Python 文档
*   [Python 中的内置异常](https://docs.python.org/3/library/exceptions.html)
*   回溯[模块](https://docs.python.org/2/library/traceback.html)
*   [在 Python 中处理异常](https://wiki.python.org/moin/HandlingExceptions)