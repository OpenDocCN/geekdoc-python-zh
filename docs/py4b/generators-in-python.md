# Python 中的生成器

> 原文：<https://www.pythonforbeginners.com/basics/generators-in-python>

你了解 python 中的[函数吗？如果您的回答是肯定的，那么让我带您了解一下 python 中生成器函数和生成器的有趣概念。在这篇文章中，我们将看看如何在程序中定义和使用生成器。我们还会用一些例子来看看生成器和函数有什么不同。](https://www.pythonforbeginners.com/basics/python-functions-cheat-sheet)

## 什么是函数？

在 python 中，函数是完成某些特定工作的代码块。例如，一个函数可以将两个数相加，一个函数可以从你的计算机中删除一个文件，一个函数可以做任何你想让它做的特定任务。

函数也使用 return 语句返回所需的输出值。下面给出了一个函数的例子。它将两个数字作为输入，将它们相乘，并使用 return 语句返回输出值。

```py
def multiplication(num1, num2):
    product = num1 * num2
    return product

result = multiplication(10, 15)
print("Product of 10 and 15 is:", result) 
```

输出:

```py
Product of 10 and 15 is: 150
```

## 什么是生成器函数？

生成器函数类似于 python 中的函数，但它向调用者提供类似迭代器的生成器作为输出，而不是对象或值。此外，我们在生成器函数中使用 yield 语句而不是 return 语句。yield 语句在生成器函数执行时暂停它的执行，并将输出值返回给调用者。生成器函数可以有一个或多个 yield 语句，但不能有 return 语句。

我们可以用类似于 python 中函数的方式定义一个生成器函数，但是我们不能使用 return 语句。相反，我们使用收益表。下面是一个生成器函数的示例，它将从 1 到 10 的数字返回给调用者。

```py
def num_generator():
    for i in range(1, 11):
        yield i

gen = num_generator()
print("Values obtained from generator function are:")
for element in gen:
    print(element) 
```

输出:

```py
Values obtained from generator function are:
1
2
3
4
5
6
7
8
9
10
```

## Python 中的生成器是什么？

python 中的生成器是一种迭代器，用于使用 next()函数执行生成器函数。为了执行一个生成器函数，我们将它赋给生成器变量。然后我们使用 next()方法来执行生成器函数。

next()函数将生成器作为输入，并执行生成器函数，直到下一个 yield 语句。之后，暂停执行生成器功能。为了继续执行，我们再次调用 next()函数，将生成器作为输入。同样，生成器函数执行到下一个 yield 语句。这个过程可以继续，直到生成器函数的执行完成。从下面的例子可以理解这个过程。

```py
def num_generator():
    yield 1
    yield 2
    yield 3
    yield 4

gen = num_generator()
for i in range(4):
    print("Accessing element from generator.")
    element = next(gen)
    print(element) 
```

输出:

```py
Accessing element from generator.
1
Accessing element from generator.
2
Accessing element from generator.
3
Accessing element from generator.
4

Process finished with exit code 0 
```

在上面的输出中，您可以看到每次调用 next()函数时，都会打印下一个 yield 语句中的元素。它显示了每次调用 next()函数时，生成器函数都会继续执行。

如果在生成器函数完成执行后，我们试图用生成器作为输入调用 next()函数，next()函数将引发 StopIteration 异常。因此，建议在除了块之外的 [python try 中使用 next()函数。此外，我们还可以使用 for 循环](https://www.pythonforbeginners.com/error-handling/python-try-and-except)遍历 [python 中的生成器。它将产生与使用 next()函数执行程序时相同的结果。](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)

```py
def num_generator():
    yield 1
    yield 2
    yield 3
    yield 4

gen = num_generator()
for i in gen:
    print("Accessing element from generator.")
    print(i)
```

输出:

```py
Accessing element from generator.
1
Accessing element from generator.
2
Accessing element from generator.
3
Accessing element from generator.
4

Process finished with exit code 0 
```

## Python 中的生成器示例

因为我们已经讨论了 Python 中的生成器和生成器函数，所以让我们实现一个程序来更好地理解上面的概念。在下面的程序中，我们实现了一个生成器函数，它将一个列表作为输入，并计算列表中元素的平方。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def square_generator(input_list):
    for element in input_list:
        print("Returning the square of next element:",element)
        yield element*element

print("The input list is:",myList)
gen = square_generator(myList)
for i in range(10):
    print("Accessing square of next element from generator.")
    square = next(gen)
    print(square)
```

输出:

```py
The input list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Accessing square of next element from generator.
Returning the square of next element: 1
1
Accessing square of next element from generator.
Returning the square of next element: 2
4
Accessing square of next element from generator.
Returning the square of next element: 3
9
Accessing square of next element from generator.
Returning the square of next element: 4
16
Accessing square of next element from generator.
Returning the square of next element: 5
25
Accessing square of next element from generator.
Returning the square of next element: 6
36
Accessing square of next element from generator.
Returning the square of next element: 7
49
Accessing square of next element from generator.
Returning the square of next element: 8
64
Accessing square of next element from generator.
Returning the square of next element: 9
81
Accessing square of next element from generator.
Returning the square of next element: 10
100 
```

在上面的例子中，您可以看到，每当使用生成器作为输入执行 next()函数时，它都会执行一次循环，直到 yield 语句。一旦 yield 语句被执行，生成器函数的执行就会暂停，直到我们再次执行 next()函数。

## 函数和生成器函数的主要区别

函数和生成器函数的主要区别如下。

*   函数有一个 return 语句，而生成器函数有一个 yield 语句。
*   函数在执行第一条 return 语句后停止执行。然而，生成器函数只是在 yield 语句执行后暂停执行。
*   函数返回一个值或一个容器对象，而生成器函数返回一个生成器对象。

## 结论

在本文中，我们讨论了 Python 中的生成器函数和生成器。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)