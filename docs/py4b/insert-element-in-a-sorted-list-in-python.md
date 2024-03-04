# 在 Python 中的排序列表中插入元素

> 原文：<https://www.pythonforbeginners.com/basics/insert-element-in-a-sorted-list-in-python>

通常，我们在 python 中把元素添加到列表的末尾。然而，如果给我们一个排序的列表，并要求我们在插入新元素时保持元素的顺序，这可能会成为一个乏味的任务。在本文中，我们将讨论用 python 在排序列表中插入元素的不同方法。

## 如何在排序列表中插入元素？

如果给我们一个排序列表，并要求我们在插入新元素时保持元素的顺序，我们首先需要找到可以插入新元素的位置。之后，我们可以使用切片或`insert()`方法将元素插入到列表中。

### 使用切片

为了使用[切片](https://www.pythonforbeginners.com/dictionary/python-slicing)在排序列表中插入一个新元素，我们将首先找到元素将被插入的位置。为此，我们将找到列表中的元素大于要插入的元素的索引。然后，我们将列表分成两部分，一部分包含小于要插入的元素的所有元素，另一部分包含大于或等于要插入的元素的所有元素。

创建切片后，我们将创建一个列表，其中要插入的元素是它唯一的元素。此后，我们将连接切片。这样，我们可以创建一个包含新元素的排序列表。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 3, 5, 6, 7, 8, 9, 10]
print("Original list is:", myList)
element = 4
print("The element to be inserted is:", element)
l = len(myList)
index = 0
for i in range(l):
    if myList[i] > element:
        index = i
        break
myList = myList[:index] + [element] + myList[index:]
print("The updated list is:", myList)
```

输出:

```py
Original list is: [1, 2, 3, 5, 6, 7, 8, 9, 10]
The element to be inserted is: 4
The updated list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### 使用 insert()方法

在找到大于待插入元素的元素索引后，我们可以使用`insert()`方法在排序列表中插入元素。当在列表上调用`insert()`方法时，该方法将索引作为其第一个输入参数，将要插入的元素作为第二个输入参数。执行后，元素被插入到列表中。

在找到比要插入的元素大的元素后，我们将使用如下所示的`insert()`方法在该元素之前插入该元素。

```py
myList = [1, 2, 3, 5, 6, 7, 8, 9, 10]
print("Original list is:", myList)
element = 4
print("The element to be inserted is:", element)
l = len(myList)
index = 0
for i in range(l):
    if myList[i] > element:
        index = i
        break
myList.insert(index, element)
print("The updated list is:", myList)
```

输出:

```py
Original list is: [1, 2, 3, 5, 6, 7, 8, 9, 10]
The element to be inserted is: 4
The updated list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

建议阅读:[机器学习中的回归与实例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)

## 使用等分模块在排序列表中插入元素

`bisect`模块为我们提供了`insort()`函数，通过它我们可以将一个元素插入到一个排序列表中。 `insort()`方法将排序后的列表作为第一个输入参数，将要插入的元素作为第二个输入参数。执行后，元素被插入到列表中。您可以在下面的示例中观察到这一点。

```py
import bisect

myList = [1, 2, 3, 5, 6, 7, 8, 9, 10]
print("Original list is:", myList)
element = 4
print("The element to be inserted is:", element)
bisect.insort(myList, element)
print("The updated list is:", myList)
```

输出:

```py
Original list is: [1, 2, 3, 5, 6, 7, 8, 9, 10]
The element to be inserted is: 4
The updated list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

## 结论

在本文中，我们讨论了用 python 在排序列表中插入元素的不同方法。要了解更多关于列表的知识，你可以阅读这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于用 python 理解[字典的文章。](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)