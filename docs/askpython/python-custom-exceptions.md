# Python 自定义异常

> 原文：<https://www.askpython.com/python/python-custom-exceptions>

每当遇到错误时，就会引发一个异常，它表示程序出了问题。默认情况下，语言为我们定义了很多例外，比如传递错误类型时的`TypeError`。在本文中，我们将看看如何在 Python 中创建我们自己的定制异常。

但是在我们看一下如何实现定制异常之前，让我们看看如何在 Python 中引发不同类型的异常。

* * *

## 引发异常

Python 允许程序员使用`raise`关键字手动引发异常。

格式:`raise ExceptionName`

根据传递给函数的输入，below 函数会引发不同的异常。

```py
def exception_raiser(string):
    if isinstance(string, int):
        raise ValueError
    elif isinstance(string, str):
        raise IndexError
    else:
        raise TypeError

```

**输出**:

```py
>>> exception_raiser(123)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in exception_raiser
ValueError
>>> exception_raiser('abc')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 5, in exception_raiser
IndexError
>>> exception_raiser([123, 456])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 7, in exception_raiser
TypeError

```

正如您所观察到的，程序员可以根据输入选择不同类型的异常。这也为错误处理提供了很好的灵活性，因为我们可以主动预测为什么会出现异常。

* * *

## 定义自定义例外

类似地，Python 也允许我们定义自己的定制异常。使用`raise`关键字，我们可以完全控制这个异常能做什么，以及何时引发它。让我们看看如何定义和实现一些自定义异常。

### 1.创建自定义异常类

我们可以创建一个定制的异常类来定义新的异常。再次强调，使用类背后的想法是因为 Python 将所有东西都视为一个类。所以一个异常也可以是一个类并不奇怪！

所有的异常都继承了父类`Exception`,我们在创建类的时候也会继承它。

我们将创建一个名为`MyException`的类，只有当传递给它的输入是一个列表并且列表中的元素数量是奇数时，它才会引发异常。

```py
class MyException(Exception):
	pass

def list_check(lst):
    if len(lst) % 2 != 0:
        raise MyException

# MyException will not be raised
list_check([1, 2, 3, 4])

# MyException will be raised
list_check([1, 3, 5])    

```

**输出**:

```py
[email protected]:~# python3 exceptions.py
Traceback (most recent call last):
  File "exceptions.py", line 12, in <module>
    list_check([1, 3, 5])
  File "exceptions.py", line 6, in list_check
    raise MyException
__main__.MyException

```

### 2.添加自定义消息和错误

我们可以添加我们自己的错误消息，并将它们打印到控制台，用于我们的自定义异常。这涉及到传递我们的`MyException`类中的另外两个参数，即`message`和`error`参数。

让我们修改我们的原始代码，为我们的异常考虑一个定制的 ***消息*** 和 ***错误*** 。

```py
class MyException(Exception):
    def __init__(self, message, errors):
        # Call Exception.__init__(message)
        # to use the same Message header as the parent class
        super().__init__(message)
        self.errors = errors
        # Display the errors
        print('Printing Errors:')
        print(errors)

def list_check(lst):
    if len(lst) % 2 != 0:
        raise MyException('Custom Message', 'Custom Error')

# MyException will not be raised
list_check([1, 2, 3, 4])

# MyException will be raised
list_check([1, 3, 5])

```

**输出**:

```py
Printing Errors:
Custom Error
Traceback (most recent call last):
  File "exceptions.py", line 17, in <module>
    list_check([1, 3, 5])
  File "exceptions.py", line 11, in list_check
    raise MyException('Custom Message', 'Custom Error')
__main__.MyException: Custom Message

```

因此，我们已经成功地实现了我们自己的定制异常，包括添加用于调试目的的定制错误消息！如果您正在构建一个库/API，而另一个程序员想知道在定制异常出现时到底哪里出错了，这将非常有用。

* * *

## 结论

在本文中，我们学习了如何使用`raise`关键字引发异常，还使用一个类构建了我们自己的异常，并向我们的异常添加了错误消息。

## 参考

*   关于自定义异常的 JournalDev 文章
*   [Python 中的异常处理](https://www.askpython.com/python/python-exception-handling)

* * *