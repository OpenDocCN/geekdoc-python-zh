# Python 中列表中最大的元素

> 原文：<https://www.pythonforbeginners.com/basics/largest-element-in-a-list-in-python>

我们经常使用列表来存储数字。在本文中，我们将讨论在 python 中查找列表中最大元素的不同方法。

## 使用 Python 中的 sort()方法的列表中最大的元素

如果列表被排序，最大的元素位于列表的末尾，我们可以使用语法`list_name[-1]`来访问它。如果列表按降序排序，列表中最大的元素位于第一个位置，即索引 0。我们可以使用语法`list_name[0]`来访问它。

当一个列表没有排序时，找到最大的数字是一个稍微不同的任务。在这种情况下，我们可以首先对列表进行排序，然后访问排序列表的最后一个元素。或者，我们可以检查列表中的每个元素，然后找到列表中最大的元素。

为了通过排序列表找到最大的元素，我们可以使用`sort()` 方法。当在列表上调用`sort()`方法时，该方法按升序对列表进行排序。排序后，我们可以从 index -1 中得到列表的最大元素，如下所示。

```py
myList = [1, 23, 12, 45, 67, 344, 26]
print("The given list is:")
print(myList)
myList.sort()
print("The maximum element in the list is:", myList[-1])
```

输出:

```py
The given list is:
[1, 23, 12, 45, 67, 344, 26]
The maximum element in the list is: 344
```

## 使用 sorted()方法查找列表中最大的元素

如果不允许对原始列表进行排序，可以使用`sorted()`功能对列表进行排序。`sorted()`函数将一个列表作为输入参数，并返回一个排序列表。获得排序后的列表后，我们可以在索引-1 处找到列表的最大元素，如下所示。

```py
myList = [1, 23, 12, 45, 67, 344, 26]
print("The given list is:")
print(myList)
newList = sorted(myList)
print("The maximum element in the list is:", newList[-1])
```

输出:

```py
The given list is:
[1, 23, 12, 45, 67, 344, 26]
The maximum element in the list is: 344
```

## Python 中使用临时变量的列表中最大的元素

对一个列表进行排序需要`O(n*log(n))`时间，其中 n 是列表中元素的数量。对于较大的列表，在我们获得最大的元素之前，可能要花很长时间对列表进行排序。使用另一种方法，我们可以在`O(n)`时间内找到列表中最大的元素。

在这种方法中，我们将创建一个变量`myVar`，并用列表的第一个元素初始化它。现在，我们将考虑`myVar`具有最大的元素。之后，我们将比较`myVar`和列表中的每个元素。如果发现任何元素大于`myVar`，我们将用当前值更新`myVar`中的值。遍历整个列表后，我们将在`myVar`变量中获得列表的最大元素。您可以在下面的示例中观察到这一点。

```py
myList = [1, 23, 12, 45, 67, 344, 26]
print("The given list is:")
print(myList)
myVar = myList[0]
for element in myList:
    if element > myVar:
        myVar = element
print("The maximum element in the list is:", myVar)
```

输出:

```py
The given list is:
[1, 23, 12, 45, 67, 344, 26]
The maximum element in the list is: 344
```

## 使用 Python 中的 max()函数的列表中最大的元素

代替上面的方法，你可以直接使用`max()`函数来查找列表中最大的元素。`max()`函数将列表作为输入参数，并返回列表中最大的元素，如下所示。

```py
myList = [1, 23, 12, 45, 67, 344, 26]
print("The given list is:")
print(myList)
myVar = max(myList)
print("The maximum element in the list is:", myVar)
```

输出:

```py
The given list is:
[1, 23, 12, 45, 67, 344, 26]
The maximum element in the list is: 344
```

## 结论

在本文中，我们讨论了在 python 中查找列表中最大元素的各种方法。要了解更多关于 python 中的列表，可以阅读这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 python 中的[集合理解](https://www.pythonforbeginners.com/basics/set-comprehension-in-python)的文章。