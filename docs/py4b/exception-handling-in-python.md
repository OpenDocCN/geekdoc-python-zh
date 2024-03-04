# Python 中的异常处理

> 原文：<https://www.pythonforbeginners.com/error-handling/exception-handling-in-python>

## 概观

在本帖中，我们将介绍 Python 如何处理异常错误。

## 什么是例外？

异常是在程序执行过程中发生的错误。当发生
错误时，Python 会生成一个可以处理异常，这避免了您的
程序崩溃。

## 为什么要使用异常？

异常在很多方面都便于[处理程序中的错误和特殊情况](https://www.pythonforbeginners.com/error-handling/how-to-handle-errors-and-exceptions-in-python)
。当你认为你的代码会产生错误时，那么
你可以使用异常处理。

## 引发异常

您可以使用 raise exception
语句在自己的程序中引发异常。

引发异常会中断当前的代码执行，并返回异常
,直到它被处理。

## 异常错误

下面是 Python 中一些常见的[异常错误](https://www.pythonforbeginners.com/error-handling/python-errors-and-exceptions):
io error
如果文件打不开。

如果 python 找不到模块，则导入错误

当内置操作或函数接收到类型正确但值不合适的参数时，引发 ValueError

当用户点击中断键(通常是 Control-C 或 Delete)时，引发键盘中断

当其中一个内置函数(input()或 raw_input())在没有读取任何数据的情况下遇到
文件结束条件(e of)时，引发 EOFError

## 异常错误示例

现在，当我们知道一些异常错误意味着什么时，让我们看一些
例子:

尝试使用尽可能少的 Try 块，并尝试通过它们抛出的异常类型来区分失败
条件。

## 设置异常处理块

要在 Python 中使用异常处理，首先需要有一个除了
之外的总括子句。

单词“try”和“except”是 Python 关键字，用于捕捉异常。

**try-exception【异常名称】**(见上例) **blocks**

try 子句中的代码将逐个语句地执行。

如果出现异常，将跳过 try 块的剩余部分，并执行
except 子句。

```py
try:
some statements here
except:
exception handling
Let's see a short example on how to do this:

try:
print 1/0
```

```py
except ZeroDivisionError:
print "You can't divide by zero, you're silly."
```

## 它是如何工作的？

错误处理是通过使用在 try
块中捕获并在 except 块中处理的异常来完成的。如果遇到错误，try 块
代码执行停止，并向下转移到 except 块。

除了在 try 块后使用 except 块，还可以使用
finally 块。

无论异常
是否发生，finally 块中的代码都将被执行。

## 代码示例

让我们写一些代码来看看在你的
程序中不使用错误处理时会发生什么。

这个程序会要求用户输入一个 1 到 10 之间的数字，然后打印出
这个数字。

```py
number = int(raw_input("Enter a number between 1 - 10"))
print "you entered number", number
```

只要用户输入一个数字，这个程序就会非常有趣，但是如果用户输入其他东西(比如一个字符串)会发生什么呢？

```py
Enter a number between 1 - 10
hello
```

你可以看到，当我们输入一个字符串时，程序抛出了一个错误。

```py
 Traceback (most recent call last):
File "enter_number.py", line 1, in
number = int(raw_input("Enter a number between 1 - 10
"))
ValueError: invalid literal for int() with base 10: 'hello'
```

ValueError 是一种异常类型。让我们看看如何使用异常处理来修复前面的程序

```py
import sys
print "Lets fix the previous code with exception handling"

try:
number = int(raw_input("Enter a number between 1 - 10"))

except ValueError:
print "Err.. numbers only"
sys.exit()

print "you entered number", number
```

如果我们现在运行程序，输入一个字符串(而不是一个数字)，我们可以看到
得到一个不同的输出。

```py
Lets fix the previous code with exception handling
Enter a number between 1 - 10
hello
Err.. numbers only
```

```py
Try … except … else 子句try，except 语句中的 else 子句必须跟在所有 except 子句之后，如果 try 子句没有引发
异常，那么
对于必须执行的代码非常有用。
试试:
data = something _ that _ can _ go _ errorIOError: 
异常错误处理else:
doing _ different _ exception _ handling
else 子句中的异常不被前面的 except 子句处理。确保 else 子句在 finally 块之前运行。Try … finally 子句finally 子句是可选的。其目的是定义在任何情况下都必须执行的清理行动
尝试:
抬起键盘中断最后:
打印“再见，世界！”
……
再见，世界！
键盘中断
无论
异常是否发生，finally 子句总是在离开 try 语句之前执行。
记住，如果你不在 except 行上指定一个异常类型，它将
捕获所有异常，这是一个坏主意，因为这意味着你的程序将忽略
意外的错误以及 except 块实际准备
处理的错误。更多阅读[http://en.wikibooks.org/wiki/Python_Programming/Exceptions](https://en.wikibooks.org/wiki/Python_Programming/Exceptions)[http://www.linuxjournal.com/article/5821](https://www.linuxjournal.com/article/5821)[http://docs.python.org/2/library/exceptions.html](https://docs.python.org/2/library/exceptions.html)[http://docs.python.org/2/tutorial/errors.html](https://docs.python.org/2/tutorial/errors.html)[http://stackoverflow.com/questions/855759/python-try-else](https://stackoverflow.com/questions/855759/python-try-else)    

```