# Python 中的错误和异常

> 原文：<https://www.pythonforbeginners.com/error-handling/python-errors-and-exceptions>

## 错误和异常

在 Python 中，有两种错误:语法错误和异常。这篇文章将描述这些错误是什么。即将发布的帖子将展示我们如何处理这些错误。

## 句法误差

先说语法错误，(也叫解析错误)。

解析器重复出错的行，并显示一个“箭头”,指向行中检测到错误的最早点。

错误是由箭头前的标记引起的(或者至少是在箭头前的标记处检测到的):在本例中，错误是在关键字 print 处检测到的，因为它前面缺少一个冒号(':')。

文件名和行号被打印出来，这样您就知道在输入来自脚本的情况下应该在哪里查找。

#### 例子

语法错误的示例

```py
>>> while True print 'Hello world'
  File "", line 1, in ?
    while True print 'Hello world'
                   ^
SyntaxError: invalid syntax 
```

## 例外

Python 中的另一种错误是例外。

即使语句或表达式在语法上是正确的，当试图执行它时，也可能会导致错误。

在执行过程中检测到的错误称为异常。

异常有不同的类型，类型作为消息的一部分打印出来。

示例中的类型是 ZeroDivisionError、NameError 和 TypeError。

#### 异常错误

异常错误的示例。

```py
>>> 10 * (1/0)
Traceback (most recent call last):
  File "", line 1, in ?
ZeroDivisionError: integer division or modulo by zero
>>> 4 + spam*3
Traceback (most recent call last):
  File "", line 1, in ?
NameError: name 'spam' is not defined
>>> '2' + 2
Traceback (most recent call last):
  File "", line 1, in ?
TypeError: cannot concatenate 'str' and 'int' objects 
```