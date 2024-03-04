# Python 中列表的联合

> 原文：<https://www.pythonforbeginners.com/basics/union-of-lists-in-python>

在 python 中，我们几乎在每个程序中都使用列表。有时，我们可能需要合并两个列表。需要合并的列表可能有一些共同的元素。为了避免在结果列表中包含重复的元素，我们可以执行给定列表的并集。在本文中，我们将讨论列表上的联合操作意味着什么，以及如何在 python 中执行两个列表的联合。

## 如何在 Python 中执行列表的并集？

要在 python 中执行两个列表的联合，我们只需创建一个输出列表，它应该包含来自两个输入列表的元素。例如，如果我们有`list1=[1,2,3,4,5,6]`和`list2=[2,4,6,8,10,12]`,`list1`和`list2`的联合将是 `[1,2,3,4,5,6,8,10,12]`。您可以观察到输出列表的每个元素要么属于`list1`要么属于`list2`。换句话说，list1 和 list2 中的每个元素都出现在输出列表中。

我们可以使用各种方法在 python 中执行列表的联合。让我们逐一讨论。

## Python 中使用 For 循环的列表联合

为了使用 for 循环执行列表的联合，我们将首先创建一个名为`newList`的空列表来存储输出列表的值。之后，我们将使用`extend()`方法将第一个输入列表的所有元素添加到`newList`。现在，我们必须将第二个输入列表中还没有的元素添加到`newList`中。为此，我们将遍历第二个列表的每个元素，并检查它是否存在于`newList`中。如果元素还没有出现在`newList`中，我们将使用`append()`方法将元素附加到`newList`中。遍历完第二个输入列表后，我们将得到包含两个输入列表的并集的列表作为`newList`。您可以在下面的示例中观察到这一点。

```py
list1 = [1, 2, 3, 4, 5, 6]
print("First list is:", list1)
list2 = [2, 4, 6, 8, 10, 12]
print("Second list is:", list2)
newList = []
newList.extend(list1)
for element in list2:
    if element not in newList:
        newList.append(element)
print("Union of the lists is:", newList)
```

输出:

```py
First list is: [1, 2, 3, 4, 5, 6]
Second list is: [2, 4, 6, 8, 10, 12]
Union of the lists is: [1, 2, 3, 4, 5, 6, 8, 10, 12]
```

如果允许您修改输入列表，您可以优化上面的方法。为了优化程序，您可以简单地检查第二个输入列表中的元素是否出现在第一个输入列表中，或者不使用 for 循环。如果元素不存在，我们可以将该元素追加到第一个输入列表中。在执行 for 循环后，我们将获得第一个输入列表中两个列表的并集，如下所示。

```py
list1 = [1, 2, 3, 4, 5, 6]
print("First list is:", list1)
list2 = [2, 4, 6, 8, 10, 12]
print("Second list is:", list2)
for element in list2:
    if element not in list1:
        list1.append(element)
print("Union of the lists is:", list1)
```

输出:

```py
First list is: [1, 2, 3, 4, 5, 6]
Second list is: [2, 4, 6, 8, 10, 12]
Union of the lists is: [1, 2, 3, 4, 5, 6, 8, 10, 12]
```

或者，您可以将列表的并集存储在第二个输入列表中，如下所示。

```py
list1 = [1, 2, 3, 4, 5, 6]
print("First list is:", list1)
list2 = [2, 4, 6, 8, 10, 12]
print("Second list is:", list2)
for element in list1:
    if element not in list2:
        list2.append(element)
print("Union of the lists is:", list2)
```

输出:

```py
First list is: [1, 2, 3, 4, 5, 6]
Second list is: [2, 4, 6, 8, 10, 12]
Union of the lists is: [2, 4, 6, 8, 10, 12, 1, 3, 5]
```

## Python 中使用集合的列表联合

像并集和交集这样的运算最初是为集合定义的。我们也可以用集合来寻找 python 中两个列表的并集。要使用集合对列表执行联合操作，可以将输入列表转换为集合。之后，您可以使用`union()`方法执行集合并集操作。最后，您可以将输出集转换回列表，如下所示。

```py
list1 = [1, 2, 3, 4, 5, 6]
print("First list is:", list1)
list2 = [2, 4, 6, 8, 10, 12]
print("Second list is:", list2)
set1 = set(list1)
set2 = set(list2)
newList = list(set1.union(set2))
print("Union of the lists is:", newList)
```

输出:

```py
First list is: [1, 2, 3, 4, 5, 6]
Second list is: [2, 4, 6, 8, 10, 12]
Union of the lists is: [1, 2, 3, 4, 5, 6, 8, 10, 12]
```

## 结论

在本文中，我们讨论了如何在 python 中执行列表的联合。想要了解更多关于列表的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于如何用 python 反转列表的文章[。](https://www.pythonforbeginners.com/lists/how-to-reverse-a-list-in-python)