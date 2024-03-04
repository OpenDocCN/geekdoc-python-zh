# 在 Python 中删除列表的所有元素

> 原文：<https://www.pythonforbeginners.com/basics/delete-all-elements-of-a-list-in-python>

在 python 中，我们几乎在每个程序中都使用列表。我们已经讨论了在 python 中比较两个列表和反转列表的方法。在本文中，我们将讨论在 python 中删除列表中所有元素的不同方法。

## 使用 clear()方法删除 Python 中列表的所有元素

pop()方法用于删除列表中的最后一个元素。当在列表上调用时，pop()方法返回列表的最后一个元素，并将其从列表中删除。我们可以使用 while 循环和 pop()方法来删除列表中的所有元素。

为此，我们可以在 while 循环中继续调用列表上的 pop()方法，直到列表变空。一旦列表变空，while 循环将停止执行，我们将得到一个没有元素的列表。您可以在下面的程序中观察到这一点。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("The original list is:")
print(myList)
# deleting elements using the pop() method
while myList:
    myList.pop()
print("List after deleting all the elements:",myList)
```

输出:

```py
The original list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9]
List after deleting all the elements: []
```

我们可以使用 clear()方法删除列表中的所有元素，而不是多次使用 pop()方法。当在列表上调用 clear()方法时，会删除列表中的所有元素。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("The original list is:")
print(myList)
# deleting elements using the clear() method
myList.clear()
print("List after deleting all the elements:",myList)
```

输出:

```py
The original list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9]
List after deleting all the elements: []
```

## 使用 del 语句删除 Python 中列表的所有元素

del 语句用于删除 python 中的对象。我们还可以使用 del 语句删除列表中的元素。为此，我们将创建一个包含所有元素的列表片段。之后，我们将删除切片。由于切片包含对原始列表中元素的引用，原始列表中的所有元素都将被删除，我们将得到一个空列表。你可以这样做。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("The original list is:")
print(myList)
# deleting elements using the del method
del myList[:]
print("List after deleting all the elements:", myList)
```

输出:

```py
The original list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9]
List after deleting all the elements: []
```

## 使用*运算符删除 Python 中列表的所有元素

这是 Python 中最少使用的从列表中移除元素的方法之一。你可能知道，将一个列表乘以任意数 N 会将列表中的元素重复 N 次。同样的，当我们将一个列表乘以 0 时，列表中的所有元素都会被删除。因此，我们可以将给定的列表乘以 0，以删除它的所有元素，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("The original list is:")
print(myList)
# deleting elements using *
myList = myList * 0
print("List after deleting all the elements:", myList) 
```

输出:

```py
The original list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9]
List after deleting all the elements: []
```

## 结论

在本文中，我们讨论了用 python 删除列表中所有元素的四种不同方法。要了解更多关于 python 中的列表，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于如何用 python 获得列表的最后一个元素的文章。