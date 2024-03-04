# 检查列表是否有重复的元素

> 原文：<https://www.pythonforbeginners.com/basics/check-if-a-list-has-duplicate-elements>

列表是 Python 中最常用的数据结构。在编程时，您可能会遇到这样的情况:您需要一个只包含唯一元素的列表，或者您希望检查一个列表中是否有重复的元素。在本文中，我们将研究不同的方法来检查一个列表中是否有重复的元素。

## 使用集合检查列表是否有重复的元素

我们知道 Python 中的集合只包含唯一的元素。我们可以使用集合的这个属性来检查一个列表是否有重复的元素。

为此，我们将从列表的元素中创建一个集合。之后，我们将检查列表和集合的大小。如果两个对象的大小相等，它将确认列表中没有重复的元素。如果集合的大小大于列表，这意味着列表包含重复的元素。我们可以从下面的例子中理解这一点。

```py
def check_duplicate(l):
    mySet = set(l)
    if len(mySet) == len(l):
        print("List has no duplicate elements.")
    else:
        print("The list contains duplicate elements")

list1 = [1, 2, 3, 4, 5, 6, 7]
print("List1 is:", list1)
check_duplicate(list1)
list2 = [1, 2, 1, 2, 4, 6, 7]
print("List2 is:", list2)
check_duplicate(list2) 
```

输出:

```py
List1 is: [1, 2, 3, 4, 5, 6, 7]
List has no duplicate elements.
List2 is: [1, 2, 1, 2, 4, 6, 7]
The list contains duplicate elements
```

在上面的方法中，我们需要从列表的所有元素中创建一个集合。之后，我们还要检查集合和列表的大小。这些操作非常昂贵。

我们可以只搜索第一个重复的元素，而不是使用这种方法。为此，我们将从列表的第一个元素开始，并不断将它们添加到集合中。在将元素添加到集合之前，我们将检查该元素是否已经存在于集合中。如果是，则列表包含重复的元素。如果我们能够将列表中的每个元素添加到集合中，则列表中不包含任何重复的元素。这可以从下面的例子中理解。

```py
def check_duplicate(l):
    visited = set()
    has_duplicate = False
    for element in l:
        if element in visited:
            print("The list contains duplicate elements.")
            has_duplicate = True
            break
        else:
            visited.add(element)
    if not has_duplicate:
        print("List has no duplicate elements.")

list1 = [1, 2, 3, 4, 5, 6, 7]
print("List1 is:", list1)
check_duplicate(list1)
list2 = [1, 2, 1, 2, 4, 6, 7]
print("List2 is:", list2)
check_duplicate(list2) 
```

输出:

```py
List1 is: [1, 2, 3, 4, 5, 6, 7]
List has no duplicate elements.
List2 is: [1, 2, 1, 2, 4, 6, 7]
The list contains duplicate elements.
```

## 使用 count()方法检查列表中是否有重复的元素

为了检查一个列表是否只有唯一的元素，我们还可以计算列表中不同元素的出现次数。为此，我们将使用 count()方法。在列表上调用 count()方法时，该方法将元素作为输入参数，并返回元素在列表中出现的次数。

为了检查列表是否包含重复的元素，我们将统计每个元素的频率。同时，我们还将维护一个已访问元素的列表，这样我们就不必计算已访问元素的出现次数。一旦发现任何元素的计数大于 1，就证明列表中有重复的元素。我们可以这样实现。

```py
def check_duplicate(l):
    visited = set()
    has_duplicate = False
    for element in l:
        if element in visited:
            pass
        elif l.count(element) == 1:
            visited.add(element)
        elif l.count(element) > 1:
            has_duplicate = True
            print("The list contains duplicate elements.")
            break
    if not has_duplicate:
        print("List has no duplicate elements.")

list1 = [1, 2, 3, 4, 5, 6, 7, 8]
print("List1 is:", list1)
check_duplicate(list1)
list2 = [1, 2, 1, 2, 4, 6, 7, 8]
print("List2 is:", list2)
check_duplicate(list2)
```

输出:

```py
List1 is: [1, 2, 3, 4, 5, 6, 7, 8]
List has no duplicate elements.
List2 is: [1, 2, 1, 2, 4, 6, 7, 8]
The list contains duplicate elements.
```

## 使用 counter()方法检查列表中是否有重复的元素

我们还可以使用 counter()方法来检查一个列表是否只有唯一的元素。counter()方法。counter()方法将 iterable 对象作为输入，并返回一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)，其中键由 iterable 对象的元素组成，与键相关联的值是元素的频率。在使用 counter()方法获得列表中每个元素的频率后，我们可以检查任何元素的频率是否大于 1。如果是，则列表包含重复的元素。否则不会。

```py
from collections import Counter

def check_duplicate(l):
    counter = Counter(l)
    has_duplicate = False
    frequencies = counter.values()
    for i in frequencies:
        if i > 1:
            has_duplicate = True
            print("The list contains duplicate elements.")
            break
    if not has_duplicate:
        print("List has no duplicate elements.")

list1 = [1, 2, 3, 4, 5, 6, 7, 8]
print("List1 is:", list1)
check_duplicate(list1)
list2 = [1, 2, 1, 2, 4, 6, 7, 8]
print("List2 is:", list2)
check_duplicate(list2) 
```

输出:

```py
List1 is: [1, 2, 3, 4, 5, 6, 7, 8]
List has no duplicate elements.
List2 is: [1, 2, 1, 2, 4, 6, 7, 8]
The list contains duplicate elements.
```

## 结论

在本文中，我们讨论了检查列表是否只有唯一元素的四种方法。我们使用了 sets、count()和 counter()方法来实现我们的方法。要了解更多关于列表的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。