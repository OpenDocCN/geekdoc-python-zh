# Python 中的生成器理解

> 原文：<https://www.pythonforbeginners.com/basics/generator-comprehension-in-python>

您可能已经使用列表理解从不同的序列和容器对象创建列表。在本文中，我们将讨论如何理解用 Python 创建生成器。我们还将讨论生成器理解的例子，以及如何用它来代替列表理解。

## 什么是生成器理解？

生成器理解是初始化生成器以访问容器对象或数据序列中的元素的一种方式。

通常，我们通过使用 yield 语句实现一个生成器函数来创建一个生成器对象。例如，假设您想要创建一个生成器，它给出给定列表中元素的平方作为输出。我们可以使用 yield 语句创建这样一个生成器，如下所示。

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

我们可以使用 generator comprehension 从单个语句中的任何序列创建一个生成器，而不是实现 generator 函数来创建一个生成器。因此，让我们讨论生成器理解的语法和实现。

## 生成器理解语法

生成器理解的语法几乎与列表理解相同。

集合理解的语法是:*生成器* = **(** *表达式* **for** *元素***in***iterable***if***条件* **)**

*   ***可迭代*** 可以是任何可迭代对象，如列表、集合、元组或字典，我们必须从这些对象创建一个新的生成器来访问其元素。
*   ***元素*** 是我们正在为其创建生成器的可迭代对象的元素。
*   ***表达式*** 包含从 ***元素*** *中导出的值或任何数学表达式。*
*   ***条件*** 是在生成器中排除或包含一个 ***元素*** 所需的条件表达式。条件语句是可选的，如果必须访问 iterable 的所有元素，可以省略“if condition”。
*   ***生成器*** 是 Python 中使用 generator comprehension 新创建的生成器的名称。

让我们使用一个简单的程序来理解这个语法，使用 generator comprehension 从现有的列表中创建生成器。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("The given list is:",myList)
mygen = (element**2 for element in myList)
print("Elements obtained from the generator are:")
for ele in mygen:
    print(ele) 
```

输出:

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("The given list is:",myList)
mygen = (element**2 for element in myList)
print("Elements obtained from the generator are:")
for ele in mygen:
    print(ele) 
```

在上面的程序中，我们得到了一个由 10 个数字组成的列表，我们创建了一个生成器，它给出给定列表元素的平方作为输出。在语句***mygen =(myList 中的元素* * 2)***中，生成器 comprehension 已经被用来创建名为 ***mygen*** 的 ***生成器*，它给出 ***myList*** 中元素的平方作为输出。**

让我们看一个使用条件语句的生成器理解的例子。假设您想要创建一个生成器，它只输出列表中偶数的平方。这可以使用生成器理解来实现，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("The given list is:", myList)
mygen = (element ** 2 for element in myList if element % 2 == 0)
print("Elements obtained from the generator are:")
for ele in mygen:
    print(ele) 
```

输出:

```py
The given list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Elements obtained from the generator are:
4
16
36
64
100 
```

在上面的程序中，我们得到了一个由 10 个数字组成的列表，我们创建了一个生成器，它给出给定集合中偶数元素的平方作为输出。在语句 ***mygen =(如果元素% 2 == 0，则 myList 中的元素为元素* * 2)***中，generator comprehension 用于创建生成器 ***mygen*** ，该生成器给出 ***myList*** 中那些偶数元素的平方作为输出。

## 发电机理解的好处

在 Python 中使用生成器理解而不是生成器函数来创建生成器给了我们很多好处。

*   生成器理解使我们能够使用更少的代码行实现相同的功能。
*   与列表理解或集合理解不同，生成器理解不初始化任何对象。因此，您可以使用生成器理解而不是列表理解或集合理解来减少程序的内存需求。
*   生成器理解也使代码更具可读性，这有助于源代码的调试和维护。

## 结论

在本文中，我们讨论了 Python 中的生成器理解。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)