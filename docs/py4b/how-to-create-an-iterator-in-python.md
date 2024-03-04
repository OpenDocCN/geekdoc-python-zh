# 如何在 Python 中创建迭代器

> 原文：<https://www.pythonforbeginners.com/basics/how-to-create-an-iterator-in-python>

迭代器用于以顺序方式访问可迭代对象的元素。我们可以为任何容器对象创建一个迭代器，比如一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)、列表、元组或集合。在本文中，我们将讨论如何在 python 中创建迭代器。我们还将学习如何通过重写内置方法来创建自定义迭代器。

## 使用内置方法创建迭代器

我们可以使用 iter()函数和 __next__()方法创建一个迭代器。iter()函数接受一个容器对象，比如 list、tuple 或 set，并返回一个迭代器，通过它我们可以访问容器对象的元素。

要使用 iter()函数为任何容器对象创建迭代器，我们只需将对象传递给 iter()函数。该函数创建一个迭代器并返回对它的引用。我们可以使用 iter()函数创建一个迭代器，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
myIter = iter(myList)
print("The list is:", myList)
print("The iterator is:", myIter) 
```

输出:

```py
The list is: [1, 2, 3, 4, 5, 6, 7]
The iterator is: <list_iterator object at 0x7f73fed18070>
```

在输出中，您可以看到在执行 iter()函数时创建了一个 list_iterator 对象。

## 如何从迭代器中访问元素？

要使用序列中的迭代器访问容器对象的元素，我们可以使用如下的 for 循环。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
myIter = iter(myList)
print("The list is:", myList)
print("The elements in the iterator are:")
for i in myIter:
    print(i) 
```

输出:

```py
The list is: [1, 2, 3, 4, 5, 6, 7]
The elements in the iterator are:
1
2
3
4
5
6
7
```

如果需要逐个访问元素，可以使用 next()函数或 __next()方法。

为了使用 next()函数遍历迭代器，我们将迭代器作为输入参数传递给函数。它返回迭代器中的下一个元素。next()函数还会记住迭代器上次遍历的索引。当再次调用它时，它返回尚未遍历的下一个元素。这可以在下面的例子中观察到。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
myIter = iter(myList)
print("The list is:", myList)
print("The elements in the iterator are:")
element = next(myIter)
print(element)
element = next(myIter)
print(element)
element = next(myIter)
print(element)
element = next(myIter)
print(element)
element = next(myIter)
print(element)
element = next(myIter)
print(element)
element = next(myIter)
print(element)
```

输出:

```py
The list is: [1, 2, 3, 4, 5, 6, 7]
The elements in the iterator are:
1
2
3
4
5
6
7

Process finished with exit code 0 
```

__next__()方法的工作方式与 next()函数类似。每当在迭代器上调用 __next__()方法时，它都会返回尚未遍历的下一个元素。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
myIter = iter(myList)
print("The list is:", myList)
print("The elements in the iterator are:")
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element) 
```

输出:

```py
The list is: [1, 2, 3, 4, 5, 6, 7]
The elements in the iterator are:
1
2
3
4
5
6
7

Process finished with exit code 0 
```

当没有元素可供遍历，并且我们使用 next()函数或 __next__()方法时，它会引发 StopIteration 异常。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
myIter = iter(myList)
print("The list is:", myList)
print("The elements in the iterator are:")
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element) 
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/webscraping.py", line 19, in <module>
    element = myIter.__next__()
StopIteration
```

建议在 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块中使用这些函数来避免异常。

## 如何在 Python 中创建自定义迭代器

要创建自定义迭代器，我们可以覆盖设置为 default 的 __iter__()和 __next__()方法。我们将通过下面的例子来理解如何创建一个自定义迭代器。

假设您想要创建一个迭代器，当我们使用迭代器遍历对象时，它返回列表或元组中每个元素的平方。为此，我们将重写 __iter__()方法和 __next__()方法。

迭代器的构造函数将接受列表或元组以及其中的元素总数。然后它将初始化迭代器类。因为我们需要跟踪被遍历的最后一个元素，我们将初始化一个索引字段，并将其设置为 0。

```py
class SquareIterator:
    def __init__(self, data, noOfElements):
        self.data = data
        self.noOfElements = noOfElements
        self.count = 0 
```

方法的作用是初始化一个迭代器。__iter__()方法实现如下。

```py
class SquareIterator:
    def __init__(self, data, noOfElements):
        self.data = data
        self.noOfElements = noOfElements
        self.count = 0

    def __iter__(self):
        return self 
```

在重写 __iter__()方法之后，我们将重写 __next__()方法。__next__()方法用于遍历尚未遍历的下一个元素。这里，我们需要返回下一个元素的平方。

因为我们已经在 __iter__()方法中将索引初始化为-1，所以我们将首先递增索引。之后，我们将返回指定索引处元素的平方。每次调用 __next__()方法时，索引都会递增，指定索引处的元素的平方将作为输出给出。当遍历的元素数量等于列表或元组中元素的总数时，我们将引发 StopIteration 异常。它将停止迭代。__next__()方法可以定义如下。

```py
class SquareIterator:
    def __init__(self, data, noOfElements):
        self.data = data
        self.noOfElements = noOfElements
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.noOfElements:
            square = self.data[self.count] ** 2
            self.count = self.count + 1
            return square
        else:
            raise StopIteration 
```

创建迭代器的整个程序如下。

```py
class SquareIterator:
    def __init__(self, data, noOfElements):
        self.data = data
        self.noOfElements = noOfElements
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.noOfElements:
            square = self.data[self.count] ** 2
            self.count = self.count + 1
            return square
        else:
            raise StopIteration

myList = [1, 2, 3, 4, 5, 6, 7]
myIter = SquareIterator(myList, 7)
print("The list is:", myList)
print("The elements in the iterator are:")
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element)
element = myIter.__next__()
print(element) 
```

输出:

```py
The list is: [1, 2, 3, 4, 5, 6, 7]
The elements in the iterator are:
1
4
9
16
25
36
49

Process finished with exit code 0 
```

## 结论

在本文中，我们研究了在 Python 中创建迭代器的两种方法。在遍历容器对象时，可以使用自定义迭代器对容器对象的元素执行不同的操作。建议实现一些程序来创建迭代器，以便更好地理解这个主题。