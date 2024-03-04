# Python 中的异常处理:编写健壮的 Python 程序

> 原文：<https://www.pythonforbeginners.com/error-handling/exception-handling-in-python-increasing-robustness-of-your-python-program>

在用 python 编写程序时，可能会出现这样的情况:程序进入一种不希望的状态，称为异常，然后退出执行。这可能导致已完成的工作丢失，甚至可能导致内存泄漏。在本文中，我们将看到如何处理这些异常，以便程序可以使用 python 中的异常处理以正常方式继续执行。我们还将看到在 python 中实现异常处理的不同方法。

## Python 中有哪些异常？

异常是程序中不希望出现的事件/错误，它会中断程序中语句的执行流程，并在程序本身没有处理异常时停止程序的执行。python 中有一些预定义的异常。我们还可以通过创建继承 python 中异常类的类来定义异常，然后在程序执行期间使用 raise 关键字创建异常，从而声明用户定义的异常。

在下面的程序中，我们创建了一个字典，并尝试使用 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中的键来访问值。这里，在第二个 print 语句中出现了一个错误，因为`"c"`在字典中不是一个键。遇到错误时，程序将停止执行。

```py
#create a dictionary
myDict={"a":1,"b":2}
#this will print the value
print(myDict["a"])
#this will generate a KeyError exception and program will exit.
print(myDict["c"])
```

输出:

```py
1
Traceback (most recent call last):
  File "<string>", line 6, in <module>
KeyError: 'c'
```

在上面的代码中，我们可以看到在打印 1 之后，程序在第二条语句处退出，并通知`KeyError` 已经发生。我们可以处理这个错误，并使用 python `try` 和`except` 块生成定制输出。

## python 中如何处理异常？

为了处理程序可能产生的异常，我们使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 并最终在代码中阻塞来执行语句。

*   `try` 块包含要执行的可能产生错误/异常的代码。
*   `except` 块具有处理 try 块中生成的错误/异常的代码。
*   finally 块中的`code` 总是执行，无论 try 块是否产生异常，finally 块都会被执行。

我们编写必须在 try 块中执行的代码。在 except 块中，我们编写代码来处理 try 块生成的异常。在 finally 块中，我们实现那些最终必须执行的代码部分。无论是否生成异常，finally block 总是在 try 和 except block 之后执行。

在下面的代码中，我们使用 try 和 except 块实现了上一个示例中使用的程序，以便在遇到错误时程序正常终止。

```py
try:
    #create a dictionary
    myDict={"a":1,"b":2}
    #this will print the value
    print(myDict["a"])
    #this will generate a KeyError exception and program will exit from try  block
    print(myDict["c"])
except:
    print("Error Occurred in Program. Terminating.")
```

输出

```py
1
Error Occurred in Program. Terminating.
```

在上面的例子中，我们可以看到在 try 块中执行第一个 print 语句后，程序并没有终止。遇到错误后，它执行 except 块中的语句，然后终止。这里我们必须记住，异常发生点之后的 try 块中的语句将不会被执行。

## python 中如何处理特定的异常？

为了不同地处理每个异常，我们可以向异常块提供参数。当生成与参数类型相同的异常时，将执行特定块中的代码。

在下面的代码中，我们将专门处理`KeyError` 异常，其余的异常将由普通的 except 块处理。

```py
 try:
    #create a dictionary
    myDict={"a":1,"b":2}
    #this will print the value
    print(myDict["a"])
    #this will generate a NameError exception and program will exit from try block
    print(a)
except(KeyError):
    print("Key is not present in the dictionary. proceeding ahead")
except: 
    print("Error occured. proceeding ahead")
try:
    #create a dictionary
    myDict={"a":1,"b":2}
    #this will print the value
    print(myDict["a"])
    #this will generate a NameError exception and program will exit from try block
    print(myDict["c"])
except(KeyError):
    print("Key is not present in the dictionary. Terminating the program")
except: 
    print("Error occured. Terminating")
```

输出:

```py
1
Error occured. proceeding ahead
1
Key is not present in the dictionary. Terminating the program
```

在上面的程序中，我们可以看到`KeyError`已经通过将它作为参数传递给 except 块进行了特殊处理，其他异常在 python 中的异常处理过程中正常处理。

## Python 中异常处理什么时候用 Finally 块？

最后，当程序中的一些语句需要执行时，不管程序中是否产生异常，都会使用块。在完成[文件处理](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)或使用网络连接的程序中，程序必须在终止前终止连接或关闭文件。我们将 finally 块放在 try 和 except 块之后。

```py
try:
    #create a dictionary
    myDict={"a":1,"b":2}
    #this will print the value
    print(myDict["a"])
    #this will generate a NameError exception and program will exit from try block
    print(myDict["c"])
except(KeyError):
    print("Key is not present in the dictionary. proceeding ahead")
finally:
    print("This is the compulsory part kept in finally block and will always be executed.")
```

输出:

```py
1
Key is not present in the dictionary. proceeding ahead
This is the compulsory part kept in finally block and will always be executed.
```

在上面的代码中，我们可以看到 try 块引发了一个异常，这个异常由 except 块处理，然后最后执行 finally 块。

## Python 中异常处理何时使用 else 块？

当我们需要在成功执行 try 块中的语句后执行某些代码语句时，我们也可以将 else 块与 python try except 块一起使用。在 try 和 except 块之后写入 Else 块。这里，我们必须记住，else 块中生成的错误/异常不是由 except 块中的语句处理的。

```py
 try:
    #create a dictionary
    myDict={"a":1,"b":2}
    #this will print the value
    print(myDict["a"])
except:
    print("I am in except block and will get executed when an exception occurs in try block")
else:
    print("I am in else block and will get executed every time after try block is executed successfully.") 
```

输出:

```py
1
I am in else block and will get executed every time after try block is executed successfully.
```

在上面的程序中，我们可以看到，当 try 块成功执行时，else 块中的代码已经执行。如果 try 块引发异常，那么只有 except 块会被执行。如果 try 块生成异常，那么 else 块中的代码将不会被执行。

## 如何在 Python 中生成用户定义的异常？

我们还可以通过使用 python 中的异常处理对一些值进行约束。为了生成一个用户定义的异常，我们在满足特定条件时使用“raise”关键字。然后由代码的 except 块处理该异常。

为了创建一个用户定义的异常，我们创建一个具有期望异常名称的类，它应该继承异常类。之后，我们可以根据实现约束的需要在代码中的任何地方引发异常。

```py
#create an exception class
class SmallNumberException (Exception):
    pass

try:
    #create a dictionary
    myDict={"a":1,"b":2}
    #this will raise SmallNumberException
    if(myDict["a"]<10):
        raise SmallNumberException
except(SmallNumberException):
    print("The Number is smaller than 10")
```

输出:

```py
The Number is smaller than 10 
```

在上面的代码中，我们可以看到已经创建了一个用户定义的异常，它继承了 exception 类，并在一个条件语句之后被引发，以检查数字是否小于 10。我们可以在任何地方使用用户定义的异常来为程序中的变量值添加约束。

## 结论

在本文中，我们学习了 python 中的异常和异常处理。我们还研究了如何在异常处理期间实现 try、except、finally 和 else 块。此外，我们还研究了如何创建自定义的用户定义的错误和异常，以实现对变量的约束。