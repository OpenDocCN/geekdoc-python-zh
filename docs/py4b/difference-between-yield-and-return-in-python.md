# Python 中收益和回报的区别

> 原文：<https://www.pythonforbeginners.com/basics/difference-between-yield-and-return-in-python>

在用 python 编程时，您可能使用过 yield 和 return 语句。在本文中，我们将讨论回报率和收益率关键字的理论概念。我们还将看看 python 中 yield 和 return 语句的不同之处。

## Python 中的 yield 和 return 是什么？

Yield 和 return 是 python 中的关键词。它们在函数中用于在程序中将值从一个函数传递到另一个函数。

## return 关键字

return 语句在函数中用于将对象返回给调用方函数。我们可以返回单个值，如数字、字符串或容器对象，如 python 字典、元组或列表。

例如，sumOfNums()函数在下面的源代码中向调用者返回一个数字。

```py
def sumOfNums(num1, num2):
    result = num2 + num1
    return result

output = sumOfNums(10, 20)
print("Sum of 10 and 20 is:", output) 
```

输出:

```py
Sum of 10 and 20 is: 30
```

类似地，我们可以使用 return 语句返回容器对象，如下例所示。这里，函数“square”将一列数字作为输入，并返回输入列表中元素的平方列表。

```py
def square(list1):
    newList = list()
    for i in list1:
        newList.append(i * i)
    return newList

input_list = [1, 2, 3, 4, 5, 6]
print("input list is:", input_list)
output = square(input_list)
print("Output list is:", output) 
```

输出:

```py
input list is: [1, 2, 3, 4, 5, 6]
Output list is: [1, 4, 9, 16, 25, 36]
```

一个函数中可以有多个 return 语句。但是，一旦在程序中执行了 return 语句，return 语句之后写的语句就永远不会被执行。

## yield 关键字

yield 语句也在函数中用于向调用函数返回值。但是收益声明以不同的方式工作。当 yield 语句在函数中执行时，它向调用者返回一个生成器对象。可以使用 next()函数或 for 循环访问 generator 对象中的值，如下所示。

```py
def square(list1):
    newList = list()
    for i in list1:
        newList.append(i * i)
    yield newList

input_list = [1, 2, 3, 4, 5, 6]
print("input list is:", input_list)
output = square(input_list)
print("Output from the generator is:", output)
print("Elements in the generator are:",next(output)) 
```

输出:

```py
input list is: [1, 2, 3, 4, 5, 6]
Output from the generator is: <generator object square at 0x7fa59b674a50>
Elements in the generator are: [1, 4, 9, 16, 25, 36]
```

一个函数可以有多个 yield 语句。当执行第一条 yield 语句时，它会暂停函数的执行，并向调用函数返回一个生成器。当我们使用 next()函数在生成器上执行下一个操作时，该函数再次恢复并执行，直到下一个 yield 语句。这个过程可以持续到函数的最后一个语句。您可以通过下面的例子来理解这一点。

```py
def square(list1):
    yield list1[0]**2
    yield list1[1] ** 2
    yield list1[2] ** 2
    yield list1[3] ** 2
    yield list1[4] ** 2
    yield list1[5] ** 2

input_list = [1, 2, 3, 4, 5, 6]
print("input list is:", input_list)
output = square(input_list)
print("Output from the generator is:", output)
print("Elements in the generator are:")
for i in output:
    print(i) 
```

输出:

```py
input list is: [1, 2, 3, 4, 5, 6]
Output from the generator is: <generator object square at 0x7fa421848a50>
Elements in the generator are:
1
4
9
16
25
36
```

您应该记住，当在执行完最后一个 yield 语句后将生成器传递给 next()函数时，它会导致 StopIteration 错误。可以通过在除了块之外的 [python try 中使用 next()函数来避免。](https://www.pythonforbeginners.com/error-handling/python-try-and-except)

## 收益和回报之间的差异

python 中 yield 和 return 语句的工作方式有两个主要区别。

*   Return 语句停止函数的执行。而 yield 语句只暂停函数的执行。
*   程序中写在 return 语句之后的语句是不可达的，并且永远不会被执行。另一方面，在 yield 语句之后编写的语句在函数恢复执行时执行。

## 结论

在本文中，我们学习了 Python 中的 yield 和 return 语句。我们还研究了收益和回报语句之间的差异。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)