# Python 中的异常:不同类型的异常以及如何在 Python 中处理它们

> 原文：<https://www.askpython.com/python/examples/exceptions-in-python>

每当您编写更大的代码片段和构建更复杂的应用程序时，Python 中的异常就会变得司空见惯。当一个人无法解决这些问题时，他们会变得很烦人。

什么时候会出现错误？

*   给出错误的输入
*   模块/库/资源不可访问
*   超越记忆或者时间
*   程序员犯的任何语法错误

* * *

## Python 中的不同异常

一个**异常**被定义为一个程序中中断程序流程和停止代码执行的条件。Python 提供了一种惊人的方式来**处理这些异常**，使得代码运行没有任何错误和中断。

异常可以属于内置错误/异常，也可以具有自定义异常。一些常见的内置异常如下:

1.  零除法错误
2.  NameError
3.  内建 Error
4.  io 错误
5.  埃费罗尔

## 在 Python 中创建测试异常

让我们看看 Python 解释器中异常的一些例子。让我们看看下面给出的代码的输出。

```py
a = int(input("Enter numerator: "))
b = int(input("Enter denominator: "))
print("a/b results in : ")
print(a/b)

```

分子是整数，分母为 0 时的输出如下所示。

```py
Enter numerator: 2
Enter denominator: 0
a/b results in : 
Traceback (most recent call last):
  File "C:/Users/Hp/Desktop/test.py", line 4, in <module>
    print(a/b)
ZeroDivisionError: division by zero

```

* * *

## 用 Try 避免异常..除...之外..

为了避免出现错误并停止程序流程，我们使用了 [**try-except** 语句](https://www.askpython.com/python/python-exception-handling)。整个代码逻辑放在 try 块中，except 块处理发生异常/错误的情况。

其语法如下所述:

```py
try:    
    #block of code     

except <Name of Exception>:    
    #block of code    

#Rest of the code

```

* * *

## 在 Python 中处理 ZeroDivisionError 异常

让我们来看看我们之前提到的代码，在 try-except 块的帮助下，显示了 **ZeroDivisionError** 。看看下面提到的代码。

```py
try:
    a = int(input("Enter numerator: "))
    b = int(input("Enter denominator: "))
    print(a/b)
except ZeroDivisionError:
    print("Denominator is zero")

```

对于与前面相同的输入，该代码的输出如下所示。

```py
Enter numerator: 2
Enter denominator: 0
Denominator is zero

```

* * *

## 结论

现在，您已经了解了异常处理，我希望您清楚异常处理的基本概念。

您可以自己尝试各种例外。编码快乐！感谢您的阅读！😇

* * *