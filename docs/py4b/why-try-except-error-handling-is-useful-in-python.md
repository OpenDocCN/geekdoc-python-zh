# 为什么 try-except 错误处理在 Python 中很有用

> 原文：<https://www.pythonforbeginners.com/error-handling/why-try-except-error-handling-is-useful-in-python>

异常和错误是任何程序中都不希望发生的事件，它们可能导致程序执行过早终止。在 python 中，我们使用 try-except 错误处理来处理带有 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 和 finally 块的异常。在本文中，我们将通过不同的例子来研究在 python 中使用异常处理的一些原因。所以让我们开始吧！

## 我们可以处理程序员无法预测的错误

在现实世界的应用中，程序中使用的变量有许多用例及约束。程序员不可能每次都检查所有的约束，他们可能会遗漏一些情况，这可能会导致程序在执行时出错。在这些情况下，我们可以使用 try-except 错误处理来处理意外错误，并在需要时恢复程序。

例如，假设用户在执行过程中按下`ctrl+c`或`del`键来中断程序，那么程序会因为出现`KeyBoardInterrupt`错误而突然终止。我们可以如下使用 try-except 错误处理来处理错误，并在终止程序之前显示自定义消息，或者我们可以执行其他语句来将程序的状态保存在文件系统中。这样，如果用户不小心按下中断程序的键，我们可以避免程序写入文件的数据丢失。

```py
try:
    dividend=int(input())
    divisor=int(input())
    print("Dividend is:",end=" ")
    print(dividend)
    print("Divisor is:",end=" ")
    print(divisor)
    quotient=dividend/divisor
    print("Quotient is:",end=" ")
    print(quotient)
except (KeyboardInterrupt):
    print("Operation has been cancelled by the user")
```

## 使用 try-except 错误处理，我们可以处理运行时异常

即使程序处理了所有的约束并且程序在语法上是正确的，在程序执行期间也可能会出现不依赖于程序中实现的逻辑的错误。例如，如果在一个 [python 读文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)操作中没有找到一个文件，程序可能会出错，并可能提前终止执行。在这种情况下，程序应该表现得更健壮，并以平稳的方式处理这种情况。这可以使用 python 中的 try-except 错误处理来完成。。其他运行时错误，如`ZeroDivisionError`、`KeyError`、`NameError`、`IndexError`等，也可以使用有效的除尝试错误处理来处理。

例如，下面的程序被编写为将一个数除以另一个数，但是当除数变为零时，它将导致`ZeroDivisionError`，并且程序将终止给出如下输出。

```py
dividend=10
divisor=0
print("Dividend is:",end=" ")
print(dividend)
print("Divisor is:",end=" ")
print(divisor)
quotient=dividend/divisor
print("Quotient is:",end=" ")
print(quotient)
```

输出:

```py
Dividend is: 10
Divisor is: 0
Traceback (most recent call last):

  File "<ipython-input-17-f6b48848354a>", line 1, in <module>
    runfile('/home/aditya1117/untitled0.py', wdir='/home/aditya1117')

  File "/usr/lib/python3/dist-packages/spyder_kernels/customize/spydercustomize.py", line 827, in runfile
    execfile(filename, namespace)

  File "/usr/lib/python3/dist-packages/spyder_kernels/customize/spydercustomize.py", line 110, in execfile
    exec(compile(f.read(), filename, 'exec'), namespace)

  File "/home/aditya1117/untitled0.py", line 9, in <module>
    quotient=dividend/divisor

ZeroDivisionError: division by zero
```

我们可以通过在 try-except 块中包含除法运算来避免程序过早终止，这样当`ZeroDivisionError`发生时，它将自动由 except 块中的代码处理。这可以在下面的程序中看到。

```py
dividend=10
divisor=0
print("Dividend is:",end=" ")
print(dividend)
print("Divisor is:",end=" ")
print(divisor)
try:
    quotient=dividend/divisor
    print("Quotient is:",end=" ")
    print(quotient)
except (ZeroDivisionError):
    print("Divisor Cannot be zero")
```

输出:

```py
Dividend is: 10
Divisor is: 0
Divisor Cannot be zero
```

## 我们可以使用 try-except 错误处理将错误处理代码与业务逻辑分开。

通过使用 try-except 错误处理，我们可以轻松地将处理错误的代码与实现逻辑的代码分开。使用 try except 错误处理使代码更具可读性，因为业务逻辑和错误处理代码是完全分离的。

例如，假设我们想确定一个人的出生年份，我们必须勾选年龄不能为负。我们将使用如下 if-else 条件语句来实现这一点。

```py
 age= -10
print("Age is:")
print(age)
if age<0:
    print("Input Correct age.")
else:
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
```

输出:

```py
Age is:
-10
Input Correct age.
```

在上面的程序中，我们在同一个代码块中检查并处理年龄为负的情况。我们可以使用 try-except 错误处理将条件检查分成一个块，将错误处理分成另一个块，如下所示。

```py
try:
    age= -10
    print("Age is:")
    print(age)
    if age<0:
        raise ValueError
    yearOfBirth= 2021-age
    print("Year of Birth is:")
    print(yearOfBirth)
except ValueError:
    print("Input Correct age.")
```

输出:

```py
Age is:
-10
Input Correct age.
```

在这里，我们可以看到我们已经检查了 try 块中年龄的非负数，当年龄为负数时`ValueError`被引发。`ValueError`然后由 except 块中的代码处理打印消息。因此，我们将条件检查和错误处理代码分离到不同的块中，这将提高源代码的可读性。

## 我们可以使用 try-except 错误处理在调用堆栈中向上传播错误。

在 python 中，如果一个函数被另一个函数调用，它可以将错误传播给调用函数，就像被调用函数返回任何值一样。这种机制可用于将错误传播到函数堆栈中的任意数量的函数，因此它使我们能够自由地在一个地方实现所有的异常处理代码，而不是在每个函数中实现它。我们可以在主函数本身中使用 try-except 错误处理来处理所有错误，并在被调用的函数中引发异常或错误。每当在任何被调用的函数中出现异常时，它将被传播到主函数，在主函数中应该用适当的错误处理代码来处理它。

例如，在下面的代码中，`ValueError`是在函数`a()`中引发的，但是处理`ValueError`的代码是在函数`b()`中编写的。当函数`b()`调用函数 `a()`时，`a()`将错误传播给函数`b()`，然后处理错误并打印消息。

```py
def a():
    print("PythonForBeginners in function a")
    raise ValueError
def b():
    try:
        print("PythonForBeginners in function b")
        a()
    except Exception as e:
        print("Caught error from called function in function b")
#call function b
b()
```

输出:

```py
PythonForBeginners in function b
PythonForBeginners in function a
Caught error from called function in function b
```

在上面的过程中，每当任何函数中出现异常时，python 解释器都会向后搜索函数调用堆栈，以检查特定错误的错误处理代码是否已在任何调用函数中实现，以防错误未在函数本身中得到处理。如果当前函数在任何阶段都没有正确的错误处理代码，错误将传播到当前函数的调用函数，直到找到正确的错误处理代码或到达主函数。

## 结论

在本文中，我们通过不同的例子了解了如何在 python 中使用 try-except 错误处理来执行不同的任务。我们还简要了解了错误是如何在函数调用堆栈中传播的，以及它对于编写模块化源代码是如何有用的。请继续关注更多内容丰富的文章。