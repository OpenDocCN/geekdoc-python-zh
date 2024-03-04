# 何时使用 try/catch 而不是 if/else

> 原文：<https://www.pythonforbeginners.com/control-flow-2/when-to-use-try-catch-instead-of-if-else>

在编程时，我们必须处理施加在变量上的许多约束，以便程序能够以正确的方式执行。为了对变量施加约束，我们使用 if else 块和 try catch 块。在本文中，我们将看到这两种构造是如何工作的，并且我们将研究可以使用 if-else 块和可以使用 try-except 块的条件。

## if-else 的工作原理

在 python 或任何其他编程语言中，If-else 块用于根据条件语句控制程序中语句的执行。如果 If 块中提到的条件为真，则执行 if 块中编写的语句。如果条件评估为 False，则执行程序的 else 块中编写的语句。

If-else 块主要用于根据我们预先知道的条件对程序进行流程控制。

例如，假设我们在程序中执行除法运算。在这种情况下，除数中的变量不能为零。如果除数变为零，程序将出错，并出现`ZeroDivisionError`。

```py
def dividewithoutcondition(dividend,divisor):
    print("Dividend is:",end=" ")
    print(dividend)
    print("Divisor is:",end=" ")
    print(divisor)
    quotient=dividend/divisor
    print("Quotient is:",end=" ")
    print(quotient)

#Execute the function 
dividewithoutcondition(10,2)
dividewithoutcondition(10,0)
```

输出:

```py
Dividend is: 10
Divisor is: 2
Quotient is: 5.0
Dividend is: 10
Divisor is: 0
Traceback (most recent call last):

  File "<ipython-input-2-f6b48848354a>", line 1, in <module>
    runfile('/home/aditya1117/untitled0.py', wdir='/home/aditya1117')

  File "/usr/lib/python3/dist-packages/spyder_kernels/customize/spydercustomize.py", line 827, in runfile
    execfile(filename, namespace)

  File "/usr/lib/python3/dist-packages/spyder_kernels/customize/spydercustomize.py", line 110, in execfile
    exec(compile(f.read(), filename, 'exec'), namespace)

  File "/home/aditya1117/untitled0.py", line 22, in <module>
    dividewithoutcondition(10,0)

  File "/home/aditya1117/untitled0.py", line 6, in dividewithoutcondition
    quotient=dividend/divisor

ZeroDivisionError: division by zero
```

在上面的实现中，我们可以看到，当给定 0 作为除数时，出现错误，程序突然结束。为了避免错误，使程序按预期执行，我们可以按以下方式使用 if- else 块。

```py
 def dividewithifelse(dividend,divisor):
    print("Dividend is:",end=" ")
    print(dividend)
    print("Divisor is:",end=" ")
    print(divisor)
    if divisor==0:
        print("Divisor cannot be Zero")
    else:
        quotient=dividend/divisor
        print("Quotient is:",end=" ")
        print(quotient)
#Execute the function 
dividewithifelse(10,2)
dividewithifelse(10,0) 
```

输出:

```py
Dividend is: 10
Divisor is: 2
Quotient is: 5.0
Dividend is: 10
Divisor is: 0
Divisor cannot be Zero
```

在上面的程序中，我们首先检查除数是否为零。在第一种情况下，即除数不为零时，程序正常工作。在第二次除法运算中，当除数为零时，程序打印出除数不能为零，并正常退出，不像没有任何条件检查的程序那样会出错。因此，可以使用 else 块在程序中实现抢先检查，这样程序就不会出错。

## 试捕作业

[Python try-catch](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块主要用于 Python 中的错误或异常处理。每当在 try 块中执行语句时发生错误并引发异常，catch 块就会处理该异常。try 块中的语句正常执行，直到出现错误。一旦出现错误，控制就转到 catch 块，并在那里处理错误。一般来说，如果我们不知道程序执行过程中会出现什么错误或异常，我们会使用 try-except 块。

如果我们想用 try-catch 块实现除数不能为零的约束来实现同一个程序除数，我们可以这样做。在 python 中，try-catch 块被称为 try-except 块。

```py
 def dividewithtryexcept(dividend,divisor):
    print("Dividend is:",end=" ")
    print(dividend)
    print("Divisor is:",end=" ")
    print(divisor)
    try:
        quotient=dividend/divisor
        print("Quotient is:",end=" ")
        print(quotient)
    except(ZeroDivisionError):
        print("Divisor cannot be Zero")
```

输出:

```py
Dividend is: 10
Divisor is: 2
Quotient is: 5.0
Dividend is: 10
Divisor is: 0
Divisor cannot be Zero
```

这里，产生的输出与使用 if-else 块实现函数时的输出完全相同。但是这两种结构的工作方式完全不同。if-else 块优先工作并阻止错误发生，而 try-except 块在错误发生后处理错误。因此，在 try-except 块中，系统的使用比 if-else 块多。

## 尝试捕捉优于 if-else 的优势

1.  Try-catch 块可用于处理系统生成的错误，以及通过手动引发异常来实现条件语句，而 if else 块只能实现条件语句，不能处理系统生成的错误。
2.  单个 try 块可以检查任意多的条件，然后抛出错误，这些错误将由 catch 块处理。而在使用 if-else 块时，我们必须为每个条件显式实现 if 块。
3.  如果 else 块给源代码带来了很多复杂性，但是 try except 块很简单，因此使用 try-catch 块可以确保源代码的可读性。

## if-else 优于 try-catch

1.  If else 语句以先发制人的方式运行，并通过检查条件来防止错误发生，而 try-catch 块允许错误发生，因为控制返回到系统，然后系统将控制转移回 catch 块。这样，if else 块比 try-catch 语句更有效。
2.  If-else 块完全由应用程序处理，但是当使用 try-catch 从 try 块引发异常时，控制被转移到系统，然后系统再次将控制转移回应用程序，异常由 catch 块处理。由于这个原因，try-except 块在执行时比 if-else 块开销更大。

## 什么时候使用 if-else 块？

当程序员预先知道程序执行时可能出现的情况时，必须使用 If-else 块而不是 try-catch 块。在这种情况下，使用 if else 语句而不是 try-except 块将使程序更加高效。

## 什么时候使用 try-catch？

在许多情况下，程序员可能无法预测和检查所有的条件。这可能会导致程序执行出错。在这种情况下，为了处理未修复的错误，程序员应该使用 try-except 块，而不是 if-else 语句。

每当执行像 [python 读取文件](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)或写入文件这样的文件操作时，可能会发生文件不在文件系统中或者在写入文件时关闭文件之前发生错误的情况。如果发生这种情况，程序可能会出错，之前所做的工作可能会丢失。为了避免这些事情，除了在执行文件操作时必须使用块。

## 结论

记住以上几点，可以得出结论，如果我们正在编写一个源代码，其中我们已经知道每个约束和变量可以取的值，那么应该使用 if-else 语句来实现这些约束。当我们不知道程序执行过程中会遇到什么情况时，我们应该使用 try-except 或 try-catch 块。此外，我们应该尽量避免使用 try-catch 或 try-except 块进行流控制。只有 if-else 块应该用于流控制。此外，在执行文件处理操作时，我们必须使用 try-except 块。请继续关注更多内容丰富的文章。