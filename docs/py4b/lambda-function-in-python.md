# Python 中的 Lambda 函数

> 原文：<https://www.pythonforbeginners.com/basics/lambda-function-in-python>

在编程时，我们可能会面临几种需要反复使用同一个数学语句的情况。在这种情况下，多次使用同一个语句会降低源代码的可读性。在本文中，我们将讨论如何使用 lambda 函数代替重复语句来消除 python 代码中的冗余。

## Python 中的函数是什么？

python 中的[函数是一组用于执行原子任务的语句。函数可以接受零个或多个参数，也可以返回值或对象。](https://www.pythonforbeginners.com/basics/python-functions-cheat-sheet)

例如，下面定义的 square()函数将一个数字作为输入参数，并将它的平方作为输出值返回。

```py
def square(n):
    return n ** 2
```

您可能会注意到，我们已经使用 def 语句定义了函数。相反，lambda 函数是使用 lambda 语句定义的。

## Python 中的 Lambda 函数是什么？

lambda 函数类似于 python 中的函数。但是它有一些限制。

python 中的一个函数可以有多个语句，while loop、if-else 语句和其他编程构造来执行任何任务。另一方面，一个 lambda 函数只能有一条语句。

我们用 python 中的 lambda 关键字定义了一个 lambda 函数。声明 lambda 函数的语法如下。

`myFunction = lambda [arguments]: expression`

这里，

*   “myFunction”是将使用此语句创建的 lambda 函数的名称。
*   “lambda”是用于定义 lambda 函数的关键字。
*   “Arguments”是 lambda 函数的参数。这里，我们将“参数”放在方括号[]中，因为 lambda 函数可能有也可能没有输入参数。
*   “表达式”一般是数学表达式或者类似 print()的函数调用。

## 如何使用 Lambda 函数？

在 python 中，我们可以用 lambda 函数代替单行数学语句。例如，我们可以创建一个 lambda 函数，它接受一个数字并按如下方式计算其平方。

```py
square = lambda n: n ** 2
print("Square of 4 is: ", square(4))
```

输出:

```py
Square of 4 is:  16
```

使用 lambda 函数的另一种方法是减少代码中的冗余。例如，假设您需要计算四个数字的平方，并按如下方式一个接一个地打印它们的平方。

```py
def square(n):
    return n ** 2

print("Square of {} is:{} ".format(4, square(4)))
print("Square of {} is:{} ".format(3, square(3)))
print("Square of {} is:{} ".format(5, square(5)))
print("Square of {} is:{}".format(10, square(10)))
```

输出:

```py
Square of 4 is:16 
Square of 3 is:9 
Square of 5 is:25 
Square of 10 is:100
```

在这里，您可以观察到打印语句导致代码冗余。在这种情况下，我们可以创建一个 lambda 函数，它接受传递给 print 函数中 format()方法的输入参数。然后，我们可以使用这个 lambda 函数来删除代码中的冗余，如下所示。

```py
def square(n):
    return n ** 2

print_square = lambda x, y: print("Square of {} is:{} ".format(x, y))
print_square(4, square(4))
print_square(3, square(3))
print_square(5, square(5))
print_square(10, square(10))
```

输出:

```py
Square of 4 is:16 
Square of 3 is:9 
Square of 5 is:25 
Square of 10 is:100 
```

## 结论

在本文中，我们讨论了 python 中的 lambda 函数。我们还看了一些可以在 python 中使用 lambda 函数的情况。我建议你阅读这篇关于 python 中的[闭包的文章。](https://www.pythonforbeginners.com/basics/closures-in-python)