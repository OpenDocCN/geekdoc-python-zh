# Python 中的闭包

> 原文：<https://www.pythonforbeginners.com/basics/closures-in-python>

你可能听说过 Python 中的 decorators。装饰器是使用闭包实现的。在本文中，我们将研究 python 中的闭包。为了更好地理解闭包，我们将首先研究嵌套函数和自由变量，因为它们是理解 python 中闭包的先决条件。

## 什么是嵌套函数？

嵌套函数是定义在另一个函数内部的函数。通过作用域规则，嵌套函数可以访问其封闭函数中定义的所有变量。嵌套函数可以使用其封闭函数中的任何非局部变量，但我们不能修改非局部变量，因为默认情况下它处于只读模式。这可以从下面的例子中理解。

```py
def enclosing_function():
   myVar = 1117
   print("It is enclosing function which encloses a nested function.")

   def nested_function(val):
       print("I am in nested function and I can access myVar from my enclosing function's scope.")
       print("The value of myVar is:", myVar)
       temp = myVar + val
       print("Value after adding {} to {} is {}.".format(val, myVar, temp))

   nested_function(10)

# Execution
enclosing_function()
```

输出:

```py
It is enclosing function which encloses a nested function.
I am in nested function and I can access myVar from my enclosing function's scope.
The value of myVar is: 1117
Value after adding 10 to 1117 is 1127.
```

在上面的代码中，我们定义了一个名为 **enclosing_function 的封闭函数。**在其中，我们定义了一个名为 **nested_function** 的嵌套函数。在这里，您可以看到我们可以访问在**嵌套函数**中的**封闭函数**内声明的变量。

要修改嵌套函数中的非局部变量，可以使用 nonlocal 关键字声明非局部变量。

## 什么是自由变量？

在程序中，我们只能在声明变量的范围内访问变量。换句话说，如果我们在函数或代码块的范围内声明一个变量，它只能在该范围内被访问。如果我们试图访问范围之外的变量，将会发生 NameError 异常。

在某些情况下，比如嵌套函数，我们可以在定义变量的范围之外访问变量。这种可以在声明范围之外访问的变量称为自由变量。在上面的例子中， **myVar** 是一个自由变量，因为我们可以在 **nested_function** 中访问它。

现在我们已经理解了嵌套函数和自由变量的概念，让我们看看 Python 中的闭包是什么。

## Python 中的闭包是什么？

闭包是一种我们可以将数据附加到代码上的技术。为了在 Python 中实现闭包，我们使用自由变量和嵌套函数。在封闭函数中定义了嵌套函数后，我们使用 return 语句返回嵌套函数。这里，嵌套函数应该使用自由变量来创建闭包。如果嵌套函数使用自由变量，则嵌套函数称为闭包函数。

闭包函数允许你使用一个变量，即使定义它的作用域已经从内存中移除。通过使用闭包函数，我们可以访问在封闭函数中定义的自由变量，即使封闭函数的作用域已经被破坏。因此，封闭函数中的数据被附加到闭包函数中的代码上。

## Python 中闭包存在的条件

对于 Python 中存在的闭包，应该满足某些条件。这些条件如下。

1.  在封闭函数中应该定义一个嵌套函数。
2.  嵌套函数应该从封闭函数中访问变量。
3.  封闭函数应该返回嵌套函数。

## Python 中的闭包示例

学习了 Python 中闭包背后的概念之后，让我们看一个例子来更好地理解这个概念。下面是一个 python 程序，用于创建向输入中添加随机数的函数。

```py
import random

def random_addition():
   values = [123, 1123, 11123, 111123, 1111123]

   def nested_function(val):
       x = random.choice(values)
       temp = x + val
       print("Value after adding {} to {} is {}.".format(val, x, temp))

   return nested_function

# Execution
random_function = random_addition()
random_function(10)
random_function(23)
random_function(24) 
```

输出:

```py
Value after adding 10 to 123 is 133.
Value after adding 23 to 1123 is 1146.
Value after adding 24 to 111123 is 111147. 
```

在上面的代码中，我们定义了一个名为 **random_addition** 的封闭函数。在**随机加法**中，我们有一个数字列表。在 **random_addition** 中，我们定义了一个 **nested_function** ，它接受一个数字作为输入，并从 **random_addition** 中定义的数字列表中添加一个随机数到输入值中，并打印输出。

如果你仔细观察代码，你会发现函数 **random_addition** 在程序中只执行了一次，它返回的嵌套函数被赋给了 **random_function** 变量。一旦**随机加法**的执行完成，其作用域将从内存中清除。但是，我们可以执行 **random_function** 任意次，每次我们访问 **random_addition** 函数**中定义的值。**即使存储器中不存在 **random_addition** 的范围，也会发生这种情况。这意味着数据已经附加到代码上，而代码是 Python 中闭包的主要功能。

## 结论

在本文中，我们讨论了 Python 中的嵌套函数、自由变量和闭包。如果你想避免使用全局变量，闭包是一个很好的工具。闭包也用于在 python 中实现 decorators。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)