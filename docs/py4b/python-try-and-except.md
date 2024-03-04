# 在 Python 中尝试 and Except

> 原文：<https://www.pythonforbeginners.com/error-handling/python-try-and-except>

之前我写过 Python 中的错误和异常。这篇文章将讲述如何处理这些问题。异常处理允许我们在出现异常时继续(或终止)程序。

### 错误处理

Python 中的错误处理是通过使用异常来完成的，这些异常在 try 块中被捕获，在 except 块中被处理。

### 试着除了

如果遇到错误，try 块代码将停止执行，并向下转移到 except 块。

除了在 try 块之后使用 except 块之外，还可以使用 finally 块。

无论是否发生异常，finally 块中的代码都将被执行。

### 引发异常

您可以使用 raise exception [，value]语句在自己的程序中引发异常。

引发异常会中断当前的代码执行，并返回异常，直到它被处理。

### 例子

一个 try 块如下所示

```py
try:
    print "Hello World"
except:
    print "This is an error message!"

```

### 异常错误

一些常见的异常错误有:

**io error-**如果文件无法打开。

**ImportError–**如果 python 找不到模块

**value error–**当内置操作或函数收到类型正确但值不合适的参数时引发

**keyboard interrupt—**当用户点击中断键(通常是 Control-C 或 Delete)时引发

**e ofError–**当一个内置函数(input()或 raw_input())在没有读取任何数据的情况下遇到文件结束条件(EOF)时引发

### 例子

让我们看一些使用异常的例子。

```py
except IOError:
    print('An error occured trying to read the file.')

except ValueError:
    print('Non-numeric data found in the file.')

except ImportError:
    print "NO module found"

except EOFError:
    print('Why did you do an EOF on me?')

except KeyboardInterrupt:
    print('You cancelled the operation.')

except:
    print('An error occured.')

```

Python 中有许多内置的异常。