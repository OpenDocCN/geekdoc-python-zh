# Python 中的迭代器

> 原文：<https://www.pythonforbeginners.com/basics/iterator-in-python>

你一定在编程时使用过不同的数据结构，比如 python 字典、列表、元组和集合。我们经常需要按顺序访问这些数据结构的元素。

为了顺序地迭代这些数据结构，我们通常使用 for 循环和元素索引。在本文中，我们将尝试理解如何在不使用 for 循环或元素索引的情况下访问列表或元组的元素。

因此，让我们深入了解 Python 中迭代器和可迭代对象的概念。

## Python 中什么是 iterable？

像列表或集合这样的容器对象可以包含许多元素。如果我们可以一次访问一个容器对象的成员元素，那么这个容器对象就叫做 iterable。在我们的程序中，我们使用不同的可重复项，如链表、元组、集合或字典。

使用 for 循环可以一次访问一个 iterable 的元素，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Elements of the list are:")
for element in myList:
    print(element) 
```

输出:

```py
Elements of the list are:
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

在循环的[中，我们从头到尾访问所有元素。但是，请考虑这样一种情况，只有在特定事件发生时，我们才需要访问 iterable 的下一个元素。](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)

例如，让我们假设一个场景，其中我们有一个 100 个元素的列表。我们不断地要求用户输入一个数字，只要他们输入一个偶数，我们就打印列表的下一个元素。

现在，使用 for 循环无法做到这一点，因为我们无法预测用户何时会输入一个偶数。输入中没有顺序或次序会阻止我们在程序中使用 for 循环来迭代列表。在这种情况下，迭代器是访问可迭代元素的便利工具。那么，让我们学习什么是迭代器，以及如何用 python 创建迭代器来访问 iterable 的元素。

## Python 中的迭代器是什么？

迭代器是可以被迭代的对象。换句话说，我们可以使用迭代器访问可迭代对象中的所有元素。

在 python 中，我们可以使用 iter()方法为任何容器对象创建迭代器。iter()方法接受一个 iterable 对象作为输入，并返回同一对象的迭代器。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
myIter = iter(myList)
print("list is:", myList)
print("Iterator for the list is:", myIter) 
```

输出:

```py
list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Iterator for the list is: <list_iterator object at 0x7f4c21734070>
```

在输出中，您可以看到通过向 iter()函数传递一个列表创建了一个 list_iterator 对象。

## 如何在 Python 中遍历一个迭代器？

遍历迭代器最简单的方法是使用 for 循环。我们可以使用 for 循环访问迭代器中的每个元素，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
myIter = iter(myList)
print("list is:", myList)
print("Elements in the iterator are:")
for element in myIter:
    print(element)
```

输出:

```py
list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Elements in the iterator are:
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

如前所述，如果我们没有遍历迭代器的顺序，for 循环将不起作用。对于这一点，我们可以用两种方法。

访问迭代器元素的第一种方法是使用 __next__()方法。当在迭代器上调用 __next__()方法时，它返回前一个遍历元素旁边的元素。它总是保留关于上次返回的元素的信息，并且每当它被调用时，它只返回尚未被遍历的下一个元素。我们可以通过下面的例子来理解这一点。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
myIter = iter(myList)
print("list is:", myList)
print("Elements in the iterator are:")
try:
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
    print(myIter.__next__())
except StopIteration as e:
    print("All elements in the iterator already traversed. Raised exception", e)
```

输出:

```py
list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Elements in the iterator are:
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
All elements in the iterator already traversed. Raised exception 
```

访问迭代器元素的另一种方法是使用 next()函数。next()函数将迭代器作为输入，返回尚未遍历的下一个元素，就像 __next__()方法一样。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
myIter = iter(myList)
print("list is:", myList)
print("Elements in the iterator are:")
try:
    print(next(myIter))
    print(next(myIter))
    print(next(myIter))
    print(next(myIter))
    print(next(myIter))
    print(next(myIter))
    print(next(myIter))
    print(next(myIter))
    print(next(myIter))
    print(next(myIter))
    print(next(myIter))
except StopIteration as e:
    print("All elements in the iterator already traversed. Raised exception", e) 
```

输出:

```py
list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Elements in the iterator are:
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
All elements in the iterator already traversed. Raised exception 
```

从上面两个例子可以观察到 __next__()方法和 next()函数的功能几乎是相似的。此外，当迭代器的所有元素都已被遍历时，next()函数和 __next__()方法都会引发 StopIteration 错误。因此，建议使用除了块之外的 [python try 来使用异常处理。](https://www.pythonforbeginners.com/error-handling/python-try-and-except)

## 迭代器的用例示例

现在让我们仔细看看上面讨论的情况，只有当用户输入一个偶数时，我们才需要访问列表中的元素。

现在，我们有了 next()函数，无论何时调用该函数，它都可以从迭代器中访问元素。我们可以用它来访问列表中的元素。为此，我们将为列表创建一个迭代器，然后每当用户输入一个偶数时，我们将使用 next()函数访问列表中的元素。我们可以为此实现一个 python 函数，如下所示。

```py
myList = range(0, 101)
myIter = iter(myList)
while True:
    try:
        user_input = int(input("Input an even number to get an output, 0 to exit:"))
        if user_input == 0:
            print("Good Bye")
            break
        elif user_input % 2 == 0:
            print("Very good. Here is an output for you.")
            print(next(myIter))
        else:
            print("Input an even number.")
            continue
    except StopIteration as e:
        print("All the output has been exhausted.") 
```

输出:

```py
Input an even number to get an output, 0 to exit:1
Input an even number.
Input an even number to get an output, 0 to exit:2
Very good. Here is an output for you.
0
Input an even number to get an output, 0 to exit:4
Very good. Here is an output for you.
1
Input an even number to get an output, 0 to exit:3
Input an even number.
Input an even number to get an output, 0 to exit:6
Very good. Here is an output for you.
2
Input an even number to get an output, 0 to exit:0
Good Bye 
```

## 结论

在本文中，我们讨论了如何使用 Python 中的迭代器从不同的可迭代对象中访问元素。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)