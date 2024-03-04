# 检查 Python 中的超集

> 原文：<https://www.pythonforbeginners.com/basics/check-for-superset-in-python>

在 python 中，我们使用集合来存储唯一的不可变对象。在本文中，我们将讨论什么是集合的超集。我们还将讨论在 Python 中检查超集的方法。

## 什么是超集？

集合的超集是包含给定集合的所有元素的另一个集合。换句话说，如果我们有一个集合 A 和集合 B，并且集合 B 的每个元素都属于集合 A，那么集合 A 被称为集合 B 的超集。

让我们考虑一个例子，其中给我们三个集合 A、B 和 C，如下所示。

`A={1,2,3,4,5,6,7,8}`

`B={2,4,6,8}`

`C={0,1,2,3,4}`

在这里，您可以观察到集合 B 中的所有元素都出现在集合 A 中。因此，集合 A 是集合 B 的超集。另一方面，集合 C 中的所有元素都不属于集合 A。因此，集合 A 不是集合 C 的超集。

您可以观察到超集总是比原始集拥有更多或相等的元素。现在，让我们描述一个在 python 中检查超集的逐步算法。

建议阅读:[Python 中的聊天应用](https://codinginfinite.com/python-chat-application-tutorial-source-code/)

## 如何在 Python 中检查超集？

假设给了我们两个集合 A 和 B。现在，我们必须检查集合 B 是否是集合 A 的超集。为此，我们将遍历集合 A 中的所有元素，并检查它们是否出现在集合 B 中。如果集合 A 中存在一个不属于集合 B 的元素，我们就说集合 B 不是集合 A 的超集。否则，集合 B 就是集合 A 的超集。

为了在 Python 中实现这种方法，我们将使用一个 for 循环和一个标志变量`isSuperset`。我们将初始化`isSuperset`变量为 True，表示集合 B 是集合 A 的超集。现在我们将使用 for 循环遍历集合 A。在遍历集合 A 中的元素时，我们将检查该元素是否出现在集合 B 中。

如果我们在 A 中发现了集合 B 中没有的元素，我们将把`False`赋值给`isSuperset`,表明集合 B 不是集合 A 的超集。

如果我们在集合 A 中没有发现任何不属于集合 B 的元素，`isSuperset`变量将包含值`True`,表明集合 B 是集合 A 的超集。

```py
def checkSuperset(set1, set2):
    isSuperset = True
    for element in set2:
        if element not in set1:
            isSuperset = False
            break
    return isSuperset

A = {1, 2, 3, 4, 5, 6, 7, 8}
B = {2, 4, 6, 8}
C = {0, 1, 2, 3, 4}
print("Set {} is: {}".format("A", A))
print("Set {} is: {}".format("B", B))
print("Set {} is: {}".format("C", C))
print("Set A is superset of B :", checkSuperset(A, B))
print("Set A is superset of C :", checkSuperset(A, C))
print("Set B is superset of C :", checkSuperset(B, C))
```

输出:

```py
Set A is: {1, 2, 3, 4, 5, 6, 7, 8}
Set B is: {8, 2, 4, 6}
Set C is: {0, 1, 2, 3, 4}
Set A is superset of B : True
Set A is superset of C : False
Set B is superset of C : False
```

## 使用 issuperset()方法检查超集

我们还可以使用 `issuperset()`方法来检查 python 中的超集。当在集合 A 上调用`issuperset()`方法时，该方法接受集合 B 作为输入参数，如果集合 A 是 B 的超集，则返回`True`，否则返回`False`。

您可以使用`issuperset()`方法来检查 python 中的超集，如下所示。

```py
A = {1, 2, 3, 4, 5, 6, 7, 8}
B = {2, 4, 6, 8}
C = {0, 1, 2, 3, 4}
print("Set {} is: {}".format("A", A))
print("Set {} is: {}".format("B", B))
print("Set {} is: {}".format("C", C))
print("Set A is superset of B :", A.issuperset(B))
print("Set A is superset of C :", A.issuperset(C))
print("Set B is superset of C :", B.issuperset(C))
```

输出:

```py
Set A is: {1, 2, 3, 4, 5, 6, 7, 8}
Set B is: {8, 2, 4, 6}
Set C is: {0, 1, 2, 3, 4}
Set A is superset of B : True
Set A is superset of C : False
Set B is superset of C : False
```

## 结论

在本文中，我们讨论了在 python 中检查超集的两种方法。要了解更多关于集合的知识，你可以阅读这篇关于 python 中的集合理解的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。