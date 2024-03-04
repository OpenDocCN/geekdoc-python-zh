# Python 中的 TypeError

> 原文：<https://www.pythonforbeginners.com/basics/typeerror-in-python>

用 Python 编程的时候有没有试过用字符串除整数？如果是，您可能会得到类似“TypeError:不支持/:“int”和“str”的操作数类型”的错误消息。在本文中，我们将讨论 Python 中的这个 TypeError 异常。我们还将研究发生 TypeError 异常的不同情况，以及如何避免它们。

## Python 中什么是 TypeError？

TypeError 是 Python 编程语言中的一个[异常，当操作中对象的数据类型不合适时就会发生。例如，如果试图用字符串除一个整数，则整数和字符串对象的数据类型将不兼容。因此，Python 解释器将引发一个 TypeError 异常，如下例所示。](https://www.pythonforbeginners.com/error-handling/exception-handling-in-python)

```py
myInt = 100
myStr = "10"
myResult = myInt / myStr 
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    myResult = myInt / myStr
TypeError: unsupported operand type(s) for /: 'int' and 'str'
```

让我们再举一个例子，假设我们想要连接两个列表。我们可以使用+操作符来实现，如下所示。

```py
list1 = [1, 2, 3]
list2 = [4, 5, 6]
myResult = list1 + list2
print("First list is:", list1)
print("second list is:", list2)
print("Resultant list is:", myResult)
```

输出:

```py
First list is: [1, 2, 3]
second list is: [4, 5, 6]
Resultant list is: [1, 2, 3, 4, 5, 6]
```

现在假设我们传递一个元组来代替第二个列表。这里，列表和元组数据类型在串联运算符中是不兼容的。因此，python 解释器将引发如下所示的 TypeError 异常。

```py
list1 = [1, 2, 3]
list2 = (4, 5, 6)
myResult = list1 + list2
print("First list is:", list1)
print("second list is:", list2)
print("Resultant list is:", myResult)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    myResult = list1 + list2
TypeError: can only concatenate list (not "tuple") to list 
```

看看这些例子，我们可以说，如果一个操作中不同对象的数据类型不兼容，因而不合适，那么 TypeError 就是 python 解释器引发的一个异常。

现在让我们来看一些可能发生 TypeError 异常的情况。

## Python 中什么时候会出现 TypeError 异常？

异常迫使程序提前终止。同样，没有人希望他们的程序中出现异常。但是，我们无法控制用户如何将输入传递给程序。可能有各种可能发生 TypeError 异常的情况。

让我们来看看其中的一些。

### 使用内置函数时可能会出现 TypeError 异常

所有内置函数都接受特定类型的输入参数。例如，集合中的 add()方法只接受不可变的对象，如整数、字符串、元组、浮点数等作为输入参数。如果我们试图给一个像 list 这样的可变对象作为 add()方法的输入，它将引发 TypeError，并显示一条消息"*type error:unhashable type:' list '*"如下所示。

```py
mySet = {1, 2, 3}
myList = [4, 5, 6]
mySet.add(myList) 
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    mySet.add(myList)
TypeError: unhashable type: 'list'
```

### 在两种不兼容的数据类型之间执行操作时，可能会出现 TypeError 异常

我们知道，在 Python 中，数学运算或位运算只针对某些数据类型定义。例如，我们可以将一个整数加到一个整数或浮点数上。另一方面，我们不能将字符串对象添加到整数中。向 string 对象添加整数将导致 TypeError，并显示消息" *TypeError:不支持+: 'int '和' str'* "的操作数类型，如下所示。

```py
myInt = 100
myStr = "200"
myResult = myInt + myStr
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    myResult = myInt + myStr
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

同样，所有的数学运算只允许在特定的数据类型之间进行。如果试图对具有不兼容数据类型的对象执行数学运算，将会发生 TypeError。

如果我们谈论按位运算，我们可以对整数执行按位运算，但不能对字符串执行。例如，我们可以将一个整数右移两位，如下所示。

```py
myInt = 100
myResult = myInt >> 2
print("The given Integer is:", myInt)
print("Result is:", myResult)
```

输出:

```py
The given Integer is: 100
Result is: 25
```

另一方面，如果我们试图对一个字符串执行右移操作，它将引发 TypeError，并显示消息" *TypeError:不支持的操作数类型用于> > : 'str '和' int'* ",如下所示。

```py
myStr = "100"
myResult = myStr >> 2
print("The given String is:", myStr)
print("Result is:", myResult)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 2, in <module>
    myResult = myStr >> 2
TypeError: unsupported operand type(s) for >>: 'str' and 'int'
```

因此，您可以看到，对不兼容的数据类型执行数学或位运算会导致程序中出现 TypeError 异常。

### 调用不可调用的对象时可能会出现 TypeError 异常

在 python 中，函数、方法以及所有在类定义中实现了 __call__()方法的对象都是可调用的。我们可以像调用函数或方法一样调用任何可调用的对象。

另一方面，如果我们调用一个不可调用的对象，比如 integer，它将引发一个 TypeError 异常，并显示消息"*type error:' int ' object is not callable*",如下所示。

```py
myInt = 100
myInt()
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 2, in <module>
    myInt()
TypeError: 'int' object is not callable
```

## 如何避免 Python 中的 TypeError 异常？

程序中的错误是不可避免的。但是，您总是可以最小化错误的发生。要最大限度地减少 TypeError 异常，可以使用以下准则。

1.  每当您试图使用内置方法或函数时，请务必阅读其文档。这将帮助您理解函数的输入和输出。了解输入和输出将有助于避免程序中的类型错误异常。
2.  在执行数学或按位运算时，可以事先检查操作数的数据类型。这将帮助您避免对不兼容的数据类型执行数学或位运算。因此，您将能够避免 TypeError 异常。
3.  给程序中的变量、函数、类和方法起适当的名字。这将帮助您避免调用不可调用的对象。因此，您将能够避免 TypeError 异常。

## 结论

在本文中，我们讨论了 TypeError 异常、其原因以及如何避免它们。您还可以使用 [python try-except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块来处理这些异常。但是，我会建议您避免异常，而不是在异常发生后处理它。

要了解更多关于 python 编程的知识，您可以阅读这篇关于 Python 中的[字符串操作的文章。你可能也会喜欢这篇关于](https://www.pythonforbeginners.com/basics/string-manipulation-in-python) [Python IndexError](https://www.pythonforbeginners.com/basics/indexerror-in-python) 的文章。

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！