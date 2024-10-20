# 函数式编程-简介

> 原文：<https://www.askpython.com/python/examples/functional-programming-introduction>

在本教程中，我们将学习函数式编程的基础知识，并通过一些例子了解如何用 Python 实现函数式编程。我们还将看看函数式编程的优缺点。

## 什么是函数式编程？

函数式编程是另一种编程范例，就像过程式编程和面向对象编程一样。

在函数式编程中，我们不是编写语句来产生输出，而是应用一系列函数来获得输出。

它可用于最终结果和中间值或开始值彼此没有物理相关性的情况。

为此，我们将问题分解为简单的函数，并使用一系列单一用途的函数来执行复杂的任务。

## 如何用 python 实现函数式编程？

为了在 Python 中实现函数式编程，我们将问题分解为纯函数，然后以声明的方式在序列中应用这些函数以产生输出。

***纯函数是指函数的输出不应该依赖于程序的全局变量和状态，也不应该产生任何副作用**。*

也就是说，函数式编程中使用的函数的输出应该只依赖于输入。

在本文中，我们将使用`map()`、`filter()`和`reduce()`方法将过程程序转换成函数程序。

### 1.Pyhton 中的 map()函数

[`map()`函数](https://www.askpython.com/python/built-in-methods/map-method-in-python)将一个函数作为它的第一个参数，将一个可迭代对象作为它的第二个参数，或者其后任意数量的可迭代对象。然后，在对输入 iterables 的每个元素应用函数后，它返回一个 map 对象。然后我们可以将 map 对象转换成我们想要的 iterable 类型。

```py
#define the function that increments a number by 1 
def increment_by_one(number):
    return number+1

#define a list
numbers= [1,3,45,67,34,78,23,56,98,104,123]
print("input list to increment values by 1 is:")
print(numbers)

#procedural way to get a list containing incremented elements by 1
result=[]
for num in numbers:
    result.append(increment_by_one(num))
print("Result obtained through procedural way is:")
print(result)

#functional way to obtain a list containing incremented elements by 1
resultbyfunc=map(increment_by_one,numbers)
print("Result obtained through functional way is:")
print(list(resultbyfunc))

```

输出:

```py
input list to increment values by 1 is:
[1, 3, 45, 67, 34, 78, 23, 56, 98, 104, 123]
Result obtained through procedural way is:
[2, 4, 46, 68, 35, 79, 24, 57, 99, 105, 124]
Result obtained through functional way is:
[2, 4, 46, 68, 35, 79, 24, 57, 99, 105, 124]

```

### 2.python 中的 filter()函数

[`filter()`函数](https://www.askpython.com/python/built-in-methods/python-filter-function)在 iterable 上应用一个函数，测试 iterable 输入的每个元素的条件，并返回 true 或 false。

它将一个函数作为它的第一个参数，其他参数是输入函数必须应用的可迭代变量。After execution filter()还返回一个迭代器，该迭代器只迭代那些在传递给输入函数时返回 true 的输入 iterables 元素。

```py
#define the function that returns true when passed an even number as input
def check_if_even(number):
    if number%2==0:
        return True
    else:
        return False

#define a list
numbers= [1,3,45,67,34,78,23,56,98,104,123]
print("input list to filter even numbers is:")
print(numbers)

#procedural way to get a list containing even numbers from input list
result=[]
for num in numbers:
    if check_if_even(num)==True:
        result.append(num)
print("Result obtained through procedural way is:")
print(result)

#functional way to obtain a list containing even numbers from input list
resultbyfunc=filter(check_if_even,numbers)
print("Result obtained through functional way is:")
print(list(resultbyfunc))

```

输出:

```py
input list to filter even numbers is:
[1, 3, 45, 67, 34, 78, 23, 56, 98, 104, 123]
Result obtained through procedural way is:
[34, 78, 56, 98, 104]
Result obtained through functional way is:
[34, 78, 56, 98, 104]

```

### 3.Python 中的 reduce()函数

`reduce()`方法用于生成累积值，如 iterable 中所有元素的总和。它在`functools`模块中定义。

我们可以传递一个函数，它接受两个参数并返回一个累积输出作为 reduce()的第一个参数，返回一个 iterable 作为第二个参数。

`reduce()`将 input 函数从左到右应用于输入 iterable 的项，并将 iterable 缩减为单个累积值并返回值。

下面是一个使用过程方法和`reduce()` 函数来计算列表元素总和的例子。

```py
#import reduce function
from functools import reduce
#define the function that returns the sum of two numbers when passed as input

def add(num1,num2):
    return num1+num2

#define a list
numbers= [1,3,45,67,34,78,23,56,98,104,123]
print("input list to find sum of elements is:")
print(numbers)

#procedural way to get the sum of numbers from input list
result=0
for num in numbers:
    result=result+num
print("Result obtained through procedural way is:")
print(result)

#functional way to obtain the sum of numbers from input list
resultbyfunc=reduce(add,numbers)
print("Result obtained through functional way is:")
print(resultbyfunc)

```

输出:

```py
input list to find sum of elements is:
[1, 3, 45, 67, 34, 78, 23, 56, 98, 104, 123]
Result obtained through procedural way is:
632
Result obtained through functional way is:
632

```

现在我们将通过一个例子来理解如何使用函数式编程。

## 将过程程序转换成函数程序

假设给我们一个数字列表，我们必须**找出列表中能被 5 整除的偶数的平方和**。

我们将使用程序性和功能性范例来实现问题的解决方案，并尝试了解程序之间的差异。

以下是实现上述问题解决方案的程序方法。

```py
#define a function that returns square of a number 
def square(num):
    return num*num

#define a function that checks if a number is even
def is_even(num):
    if num%2==0:
        return True
    else:
        return False

#define a function that checks divisibility by 5
def is_divisible_by_five(num):
    if num%5==0:
        return True
    else:
        return False

#define a list
numbers= [1,20,45,67,34,78,80,23,56,98,104,50,60,90,123]
print("input list to find the solution is:")
print(numbers)

#procedural way to find the solution
#extract elements which are dvisible by 5 and are even
temp=[]
for num in numbers:
    if is_even(num) and is_divisible_by_five(num):
        temp.append(num)

#calculate square of elements in temp
sqtemp=[]
for num in temp:
    sqtemp.append(square(num))

#find sum of squared elements
result=0
for num in sqtemp:
    result=result+num

print("Result obtained through procedural way is:")
print(result)

```

输出

```py
input list to find the solution is:
[1, 20, 45, 67, 34, 78, 80, 23, 56, 98, 104, 50, 60, 90, 123]
Result obtained through procedural way is:
21000

```

现在，我们将以下面的方式在函数范例中实现上面的代码。

```py
#import reduce function
from functools import reduce

#define the function that returns sum of two numbers when passed as input
def add(num1,num2):
    return num1+num2

#define a function that returns square of a number 
def square(num):
    return num*num

#define a function that checks if a number is even
def is_even(num):
    if num%2==0:
        return True
    else:
        return False
#define a function that checks divisibility by 5
def is_divisible_by_five(num):
    if num%5==0:
        return True
    else:
        return False

#define a list
numbers= [1,20,45,67,34,78,80,23,56,98,104,50,60,90,123]
print("input list to find the solution is:")
print(numbers)

#functional way to find the solution
#filter numbers divisible by 5
temp1=filter(is_divisible_by_five,numbers)

#filter even numbers
temp2=filter(is_even,temp1)

#find square of numbers
temp3=map(square,temp2)

#find sum of squares
result=reduce(add,temp3)
print("Result obtained through functional way is:")
print(result)

```

输出:

```py
input list to find the solution is:
[1, 20, 45, 67, 34, 78, 80, 23, 56, 98, 104, 50, 60, 90, 123]
Result obtained through functional way is:
21000

```

## 过程编程和函数编程的区别

*   在过程式编程中，我们使用一系列指令，这些指令使用条件运算符和循环来实现我们的示例，而我们只是通过向函数传递数据来进行函数调用，并将返回值传递给另一个函数来获得结果。在主逻辑的实现中没有使用条件运算符。
*   在函数式编程中，我们使用纯函数，它们执行非常简单的操作，正如我们在示例中所做的那样，但是过程式程序中的函数可能非常复杂，并且可能有副作用。
*   由于过程性程序涉及条件，它们很难调试，而函数性程序是声明性的，每个函数都有固定的工作，没有副作用，这使得它们很容易调试。

## 函数式编程的优势

从上面的例子可以看出，下面是函数式编程的优点:

*   当我们在函数式编程中使用纯函数时，调试变得很容易。
*   纯函数的可重用性高，在一次调用中只完成一次操作，所以使用纯函数增加了程序的模块化。
*   函数式程序的可读性很高，因为程序是声明性的，没有条件语句。

## 什么时候应该使用函数式编程？

函数式编程最适合做数学计算。如果你正在解决复杂的数学程序，这些程序可以分成核心步骤，函数式编程是这种情况下最好的选择。

## 什么时候不应该使用函数式编程？

*   如果你是编程初学者，就不应该使用函数式编程。我们的大脑被训练去理解序列，最初，甚至理解程序性的程序都很困难。
*   如果你正在做一个大项目，避免使用函数式编程，因为在编码阶段函数式程序的维护是困难的。
*   在函数式编程中，代码的可重用性是一项非常棘手的任务，因此您需要非常擅长它，以节省您的时间和精力。

## 结论

在本教程中，我们已经了解了什么是函数式编程，以及如何用 python 实现它。我们还看到了过程式编程和函数式编程之间的区别，函数式编程的优势，以及对于给定的任务我们是否应该使用函数式编程。