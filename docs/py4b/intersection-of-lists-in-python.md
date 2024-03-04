# Python 中列表的交集

> 原文：<https://www.pythonforbeginners.com/basics/intersection-of-lists-in-python>

列表是 python 中最常用的数据结构之一。有时，我们可能需要找到任意两个给定列表之间的公共元素。在本文中，我们将讨论如何在 python 中对两个列表执行交集操作，以找到它们之间的公共元素。

## 如何在 python 中执行列表的交集？

要在 python 中执行两个列表的交集，我们只需创建一个输出列表，它应该包含两个输入列表中都存在的元素。例如，如果我们有`list1=[1,2,3,4,5,6]`和`list2=[2,4,6,8,10,12]`,`list1`和`list2`的交集将是`[2,4,6]`。您可以观察到输出列表的每个元素都属于`list1`和`list2`。换句话说，输出列表只包含那些同时出现在两个输入列表中的元素。

我们可以使用各种方法在 python 中执行列表的交集。让我们逐一讨论。

## python 中使用 For 循环的两个列表的交集

为了执行两个列表的交集，我们将首先创建一个名为`newList`的空列表来存储输出列表的元素。之后，我们将遍历第一个输入列表，并检查它的元素是否出现在第二个输入列表中。如果一个元素同时出现在两个列表中，我们将把这个元素附加到`newList`中。在执行 for 循环之后，我们将在`newList`中获得两个输入列表中的所有元素。

```py
list1 = [1, 2, 3, 4, 5, 6]
print("First list is:", list1)
list2 = [2, 4, 6, 8, 10, 12]
print("Second list is:", list2)
newList = []
for element in list1:
    if element in list2:
        newList.append(element)
print("Intersection of the lists is:", newList)
```

输出:

```py
First list is: [1, 2, 3, 4, 5, 6]
Second list is: [2, 4, 6, 8, 10, 12]
Intersection of the lists is: [2, 4, 6]
```

或者，您可以遍历第二个输入列表，检查它的元素是否出现在第一个输入列表中。然后，您可以根据元素的存在向`newList`添加元素，如下所示。

```py
list1 = [1, 2, 3, 4, 5, 6]
print("First list is:", list1)
list2 = [2, 4, 6, 8, 10, 12]
print("Second list is:", list2)
newList = []
for element in list2:
    if element in list1:
        newList.append(element)
print("Intersection of the lists is:", newList)
```

输出:

```py
First list is: [1, 2, 3, 4, 5, 6]
Second list is: [2, 4, 6, 8, 10, 12]
Intersection of the lists is: [2, 4, 6]
```

我建议您使用 for 循环遍历较小的列表。这将使您的代码更加高效。

## python 中使用集合的两个列表的交集

像并集和交集这样的运算最初是为集合定义的。我们也可以用集合来寻找两个列表的交集。要使用集合对列表执行交集操作，可以将输入列表转换为集合。之后，您可以使用`intersection()`方法执行设置交集操作。最后，您可以将输出集转换回列表，如下所示。

```py
list1 = [1, 2, 3, 4, 5, 6]
print("First list is:", list1)
list2 = [2, 4, 6, 8, 10, 12]
print("Second list is:", list2)
set1 = set(list1)
set2 = set(list2)
newList = list(set1.intersection(set2))
print("Intersection of the lists is:", newList)
```

输出:

```py
First list is: [1, 2, 3, 4, 5, 6]
Second list is: [2, 4, 6, 8, 10, 12]
Intersection of the lists is: [2, 4, 6]
```

## 结论

在本文中，我们讨论了如何在 python 中执行列表的交集。想要了解更多关于列表的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于如何用 python 反转列表的文章[。](https://www.pythonforbeginners.com/lists/how-to-reverse-a-list-in-python)