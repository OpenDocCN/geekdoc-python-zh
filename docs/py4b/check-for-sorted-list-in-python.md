# 在 Python 中检查排序列表

> 原文：<https://www.pythonforbeginners.com/lists/check-for-sorted-list-in-python>

列表是 python 中最常用的数据结构之一。在本文中，我们将讨论在 python 中检查排序列表的不同方法。

## 如何在 Python 中检查排序列表？

排序列表的所有元素都是按非降序排列的。换句话说，如果列表是排序列表，则索引 `i` 处的元素总是小于或等于索引`i+1`处的元素。例如，考虑下面的列表。

`myList1=[1,2,3,4,5,6,7]`

`myList2=[1,2,3,3,4,5,5]`

`myList3=[1,2,3,4,2,5,6]`

在这里，如果我们从左向右移动，`myList1`将所有元素按升序排列。因此，这是一个排序列表。`myList2`的元素不是按递增顺序排列的，但是如果我们从左向右移动，所有元素都是按非递减顺序排列的。于是，`myList2`也是一个排序列表。但是`myList3`中的元素既不是升序也不是非降序。因此，它不是一个排序列表。

## 在 Python 中检查排序列表

要检查一个排序列表，我们只需要遍历列表，检查所有的元素是否按非降序排列。为此，我们将使用一个变量`isSorted`和一个 For 循环。首先，假设列表已经排序，我们将初始化变量`isSorted`为`True`。之后，我们将遍历列表，检查从索引 0 到末尾，索引`“i”`处的所有元素是否都小于或等于索引`“i+1”`处的元素。如果我们发现索引`“i”`处的任何元素大于索引`“i+1”`处的元素，我们将把值`False`赋给`isSorted`变量，表示列表没有被排序。然后我们将使用 break 语句结束循环，因为我们已经发现列表没有排序。

如果我们找不到任何比它右边的元素大的元素，变量`isSorted`将保持`True`,并表示列表已排序。您可以在下面的示例中观察到这一点。

```py
def checkSortedList(newList):
    isSorted = True
    l = len(newList)
    for i in range(l - 1):
        if newList[i] > newList[i + 1]:
            isSorted = False
    return isSorted

myList1 = [1, 2, 3, 4, 5, 6, 7]
myList2 = [1, 2, 3, 3, 4, 5, 5]
myList3 = [1, 2, 3, 4, 2, 5, 6]
print("The list {} is sorted: {} ".format(myList1, checkSortedList(myList1)))
print("The list {} is sorted: {} ".format(myList2, checkSortedList(myList2)))
print("The list {} is sorted: {} ".format(myList3, checkSortedList(myList3)))
```

输出:

```py
The list [1, 2, 3, 4, 5, 6, 7] is sorted: True 
The list [1, 2, 3, 3, 4, 5, 5] is sorted: True 
The list [1, 2, 3, 4, 2, 5, 6] is sorted: False
```

## 结论

在本文中，我们已经讨论了如何检查排序列表。要了解更多关于列表的知识，你可以阅读这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于用 python 理解[字典的文章。](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)