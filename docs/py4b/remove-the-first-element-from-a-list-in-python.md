# 在 Python 中从列表中移除第一个元素

> 原文：<https://www.pythonforbeginners.com/basics/remove-the-first-element-from-a-list-in-python>

列表是 python 中最常用的数据结构之一。在 python 中，我们有不同的方法来对列表执行操作。在这篇文章中，我们将看看使用不同的方法从列表中删除第一个元素的不同方法，如切片、pop()方法、remove()方法和 del 操作。

## 使用切片从列表中移除第一个元素

在 python 中，切片是创建字符串或列表的一部分的操作。通过切片，我们可以访问任何字符串、元组或列表的不同部分。为了对名为 myList 的列表执行切片，我们使用语法`myList[start, end, difference]`，其中“start”和“end”分别是切片列表在原始列表中开始和结束的索引。“差异”用于选择序列中的元素。从索引=“开始+n *差异”的列表中选择元素，其中 n 是整数，并且索引应该小于“结束”。

为了使用切片来移除列表的第一个元素，我们将从第二个元素开始取出子列表，并留下第一个元素。因此，第一个元素将从列表中删除。这可以如下进行。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("Original List is:", myList)
myList = myList[1::1]
print("List after modification is:", myList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
List after modification is: [2, 3, 4, 5, 6, 7]
```

## 使用 pop()方法从列表中移除第一个元素

pop()方法用于从指定的索引中移除列表中的任何元素。它将元素的索引作为可选的输入参数，并在从列表中删除元素后，返回指定索引处的元素。如果没有传递输入参数，它将在删除后返回列表的最后一个元素。

要使用 pop()方法删除列表的第一个元素，我们将调用列表上的 pop()方法，并将传递第一个元素的索引，即 0 作为输入。从列表中删除后，它将返回列表的第一个元素，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("Original List is:", myList)
myList.pop(0)
print("List after modification is:", myList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
List after modification is: [2, 3, 4, 5, 6, 7]
```

## 使用 remove()方法

remove()方法用于删除列表中第一个出现的元素。调用 remove()方法时，它将从列表中删除的元素作为输入，并从列表中删除该元素。当元素不在列表中时，它会引发 ValueError。为了处理 ValueError，您可以使用除了块之外的 [python try 来使用异常处理。](https://www.pythonforbeginners.com/error-handling/python-try-and-except)

要使用 remove()方法删除列表的第一个元素，我们将使用其索引(即 0)来访问列表的第一个元素。然后我们将元素作为参数传递给 remove()方法。这将删除列表的第一个元素，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("Original List is:", myList)
element = myList[0]
myList.remove(element)
print("List after modification is:", myList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
List after modification is: [2, 3, 4, 5, 6, 7]
```

## 使用 del 关键字从列表中删除第一个元素

python 中的 del 关键字用于删除对象。在 python 中，每个变量都指向一个对象。我们知道列表的第一个元素指向内存中的一个对象，我们也可以使用 del 关键字删除它，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("Original List is:", myList)
del myList[0]
print("List after modification is:", myList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
List after modification is: [2, 3, 4, 5, 6, 7]
```

## 结论

在本文中，我们看到了用 python 从列表中删除第一个元素的不同方法。我们使用了切片、pop()方法、remove()方法和 del 关键字来删除列表中的第一个元素。要了解更多关于列表的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。请继续关注更多内容丰富的文章。