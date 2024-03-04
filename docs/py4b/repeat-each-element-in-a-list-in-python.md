# 在 Python 中重复列表中的每个元素

> 原文：<https://www.pythonforbeginners.com/lists/repeat-each-element-in-a-list-in-python>

在 python 中，我们为各种任务使用列表。我们已经讨论了列表上的各种操作，比如[计算列表中元素的频率](https://www.pythonforbeginners.com/lists/count-the-frequency-of-elements-in-a-list)或者[反转列表](https://www.pythonforbeginners.com/lists/how-to-reverse-a-list-in-python)。在本文中，我们将讨论三种在 python 中使列表中的每个元素重复 k 次的方法。

## Python 中如何将一个列表中的每个元素重复 k 次？

为了在 python 中重复列表中的元素，我们将一个给定列表的元素重复插入到一个新列表中。为了让列表中的每个元素重复 k 次，我们将把现有 k 次的每个元素插入到新列表中。

例如，如果我们有一个列表`myList=[1,2,3,4,5]`，我们必须重复列表的元素两次，输出列表将变成`[1,1,2,2,3,3,4,4,5, 5]`。

为了执行这个操作，我们可以用[来循环](https://www.pythonforbeginners.com/basics/loops)或者用`append()`方法来列表理解。或者，我们可以使用 `itertools.chain.from_iterable()`和`itertools.repeat()`方法。我们将在接下来的章节中讨论每一种方法。

## 使用 For 循环和 append()方法将列表中的每个元素重复 k 次

方法用于将元素添加到列表的末尾。当在列表上调用时，`append()`方法获取一个元素并将其添加到列表中，如下所示。

```py
myList = [1, 2, 3, 4, 5]
print("The original list is:", myList)
element = 6
myList.append(element)
print("List after appending {} is: {}".format(element, myList))
```

输出:

```py
The original list is: [1, 2, 3, 4, 5]
List after appending 6 is: [1, 2, 3, 4, 5, 6]
```

为了让列表中的每个元素重复 k 次，我们将首先创建一个名为`newList`的空列表。之后，我们将遍历输入列表的每个元素。在遍历过程中，我们将使用 for 循环、`range()`方法和`append()`方法将现有列表的每个元素添加到新创建的列表中 k 次。执行 for 循环后，输入列表的每个元素将在新列表中重复 k 次，如下例所示。

```py
myList = [1, 2, 3, 4, 5]
print("The original list is:", myList)
k = 2
newList = []
for element in myList:
    for i in range(k):
        newList.append(element)
print("The output list is:", newList)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5]
The output list is: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
```

## 使用列表理解将列表中的每个元素重复 k 次

除了使用 for 循环，我们可以使用 list comprehension 将列表中的每个元素重复 k 次，如下所示。

```py
myList = [1, 2, 3, 4, 5]
print("The original list is:", myList)
k = 2
newList = [element for element in myList for i in range(k)]
print("The output list is:", newList)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5]
The output list is: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
```

## 使用 itertools.chain.from_iterable()和 itertools.repeat()方法

我们可以使用`itertools.chain.from_iterable()`和`itertools.repeat()`方法将列表中的每个元素重复 k 次，而不是显式地将元素添加到新列表中来重复它们。

`itertools.repeat()`方法将一个值和该值必须重复的次数作为输入参数，并创建一个迭代器。例如，我们可以创建一个迭代器，它重复任意给定值五次，如下所示。

```py
import itertools

element = 5
k = 10
newList = list(itertools.repeat(element, k))
print("The output list is:", newList)
```

输出:

```py
The output list is: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5] 
```

`itertools.chain.from_iterable()`获取可迭代对象的列表，并使用作为参数传递的可迭代对象的所有元素创建一个迭代器，如下所示。

```py
import itertools

myList1 = [1, 2, 3, 4, 5]
myList2 = [6, 7, 8, 9, 10]
newList = list(itertools.chain.from_iterable([myList1, myList2]))
print("The output list is:", newList)
```

输出:

```py
The output list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

为了将列表中的每个元素重复 k 次，我们将首先使用`itertools.repeat()`方法和`list()`构造函数为现有列表中的每个元素创建一个列表。这里，每个迭代器将有 k 次不同的元素。之后，我们将使用`itertools.chain.from_iterable()` 方法从使用`itertools.repeat()`方法创建的列表中创建输出列表。这样，我们将得到一个所有元素重复 k 次的列表。

```py
import itertools

myList = [1, 2, 3, 4, 5]
print("The original list is:", myList)
k = 2
listOfLists = [list(itertools.repeat(element, k)) for element in myList]
newList = list(itertools.chain.from_iterable(listOfLists))
print("The output list is:", newList)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5]
The output list is: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
```

## 结论

在本文中，我们讨论了 python 中列表中每个元素重复 k 次的三种方法。要阅读更多关于 python 中列表的内容，你可以阅读这篇关于如何用 python 从列表中删除最后一个元素的文章。