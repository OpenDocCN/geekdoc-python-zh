# Python 中的 Frozenset

> 原文：<https://www.pythonforbeginners.com/basics/frozenset-in-python>

使用 python 编程时，您可能在程序中使用过集合、列表和字典。在本文中，我们将研究 Python 中另一个名为 frozenset 的容器对象。我们将讨论如何创建一个 frozenset，以及如何访问它的元素。

## Python 中的 frozenset 是什么？

您可能使用过 python 中的集合。Frozensets 是具有集合的所有属性但不可变的容器对象。frozenset 与集合的关系类似于元组与列表的关系。冷冻集的主要特性如下。

*   冷冻集包含独特的元素。
*   它们是不可变的，不能在 frozenset 中添加、修改或删除任何元素。
*   我们只能在创建 frozenset 对象的过程中向 frozenset 添加元素。

现在让我们讨论如何创建一个冷冻集并访问它的元素。

## 如何用 Python 创建一个 frozenset？

我们可以使用 frozenset()构造函数创建一个 frozenset。frozenset()构造函数接受一个容器对象作为输入，并用容器对象的元素创建一个 frozenset。例如，我们可以创建一个包含列表元素的 frozenset，如下所示。

```py
myList = [1, 2, 3, 4, 5]
print("The given list is:")
print(myList)
myFrozenset = frozenset(myList)
print("The output frozenset is:")
print(myFrozenset) 
```

输出:

```py
The given list is:
[1, 2, 3, 4, 5]
The output frozenset is:
frozenset({1, 2, 3, 4, 5})
```

类似地，我们可以使用集合的元素创建一个 frozenset，如下所示。

```py
mySet = {1, 2, 3, 4, 5}
print("The given set is:")
print(mySet)
myFrozenset = frozenset(mySet)
print("The output frozenset is:")
print(myFrozenset) 
```

输出:

```py
The given set is:
{1, 2, 3, 4, 5}
The output frozenset is:
frozenset({1, 2, 3, 4, 5})
```

当没有输入给 frozenset()构造函数时，它创建一个空的 frozenset。

```py
myFrozenset = frozenset()
print("The output frozenset is:")
print(myFrozenset)
```

输出:

```py
The output frozenset is:
frozenset()
```

当我们将一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)作为输入传递给 frozenset()构造函数时，它会创建一个字典键的 frozenset。这可以在下面的例子中观察到。

```py
myDict = {1:1, 2:4, 3:9, 4:16, 5:25}
print("The given dictionary is:")
print(myDict)
myFrozenset = frozenset(myDict)
print("The output frozenset is:")
print(myFrozenset) 
```

输出:

```py
The given dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The output frozenset is:
frozenset({1, 2, 3, 4, 5})
```

## 从冷冻集中访问元素

与其他容器对象类似，我们可以使用迭代器访问 frozenset 的元素，如下所示。

```py
mySet = {1, 2, 3, 4, 5}
print("The given set is:")
print(mySet)
myFrozenset = frozenset(mySet)
print("The elements of the frozenset are:")
iterator=iter(myFrozenset)
for i in iterator:
    print(i) 
```

输出:

```py
The given set is:
{1, 2, 3, 4, 5}
The elements of the frozenset are:
1
2
3
4
5 
```

我们还可以使用 for 循环遍历 frozenset 的元素，如下所示。

```py
mySet = {1, 2, 3, 4, 5}
print("The given set is:")
print(mySet)
myFrozenset = frozenset(mySet)
print("The elements of the frozenset are:")
for i in myFrozenset:
    print(i) 
```

输出:

```py
The given set is:
{1, 2, 3, 4, 5}
The elements of the frozenset are:
1
2
3
4
5
```

## 向冷冻集添加元素

我们不能向 frozenset 添加元素，因为它们是不可变的。同样，我们不能修改或删除 frozenset 中的元素。

## Python 中 set 和 frozenset 的区别

python 中的 frozenset 可以被认为是不可变的集合。set 和 frozenset 的主要区别在于我们不能修改 frozenset 中的元素。集合和冷冻集的其他性质几乎相同。

## 结论

在本文中，我们讨论了如何用 python 创建 frozenset，以及它的属性是什么。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)