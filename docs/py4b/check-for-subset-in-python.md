# 在 Python 中检查子集

> 原文：<https://www.pythonforbeginners.com/basics/check-for-subset-in-python>

python 中的[集合是包含唯一不可变对象的数据结构。在本文中，我们将讨论什么是集合的子集，以及如何在 python 中检查子集。](https://www.pythonforbeginners.com/basics/set-operations-in-python)

## 什么是子集？

集合的子集是包含给定集合的部分或全部元素的另一个集合。换句话说，如果我们有一个集合 A 和集合 B，并且集合 B 的每个元素都属于集合 A，那么称集合 B 是集合 A 的子集。

让我们考虑一个例子，其中给我们三个集合 A、B 和 C，如下所示。

`A={1,2,3,4,5,6,,7,8}`

`B={2,4,6,8}`

`C={0,1,2,3,4}`

在这里，您可以观察到集合 B 中的所有元素都出现在集合 a 中。因此，集合 B 是集合 a 的子集。另一方面，集合 C 中的所有元素都不属于集合 a。因此，集合 C 不是集合 a 的子集。

您可以观察到，一个子集的元素总是少于或等于原始集合的元素。空集也被认为是任何给定集合的子集。现在，让我们描述一个在 python 中检查子集的逐步算法。

## 如何在 Python 中检查子集？

假设给了我们两个集合 A 和 B。现在，我们必须检查集合 B 是否是集合 A 的子集。为此，我们将遍历集合 B 中的所有元素，并检查它们是否出现在集合 A 中。如果集合 B 中存在一个不属于集合 A 的元素，我们就说集合 B 不是集合 A 的子集。否则，集合 B 就是集合 A 的子集。

为了在 Python 中实现这种方法，我们将使用一个 for 循环和一个标志变量`isSubset`。我们将把`isSubset`变量初始化为 True，表示集合 B 是集合 A 的子集。我们这样做是为了确保空的集合 B 也被认为是集合 A 的子集。在遍历集合 B 中的元素时，我们将检查该元素是否出现在集合 A 中。

如果我们找到了集合 A 中不存在的元素，我们将把`False`赋值给`isSubset`，表明集合 B 不是集合 A 的子集。

如果我们在集合 B 中没有发现任何不属于集合 A 的元素，`isSubset`变量将包含值`True`,表明集合 B 是集合 A 的子集。

```py
def checkSubset(set1, set2):
    isSubset = True
    for element in set1:
        if element not in set2:
            isSubset = False
            break
    return isSubset

A = {1, 2, 3, 4, 5, 6, 7, 8}
B = {2, 4, 6, 8}
C = {0, 1, 2, 3, 4}
print("Set {} is: {}".format("A", A))
print("Set {} is: {}".format("B", B))
print("Set {} is: {}".format("C", C))
print("Set B is subset of A :", checkSubset(B, A))
print("Set C is subset of A :", checkSubset(C, A))
print("Set B is subset of C :", checkSubset(B, C))
```

输出:

```py
Set A is: {1, 2, 3, 4, 5, 6, 7, 8}
Set B is: {8, 2, 4, 6}
Set C is: {0, 1, 2, 3, 4}
Set B is subset of A : True
Set C is subset of A : False
Set B is subset of C : False
```

建议阅读:[Python 中的聊天应用](https://codinginfinite.com/python-chat-application-tutorial-source-code/)

## 使用 issubset()方法检查子集

我们还可以使用 `issubset()`方法来检查 python 中的子集。当对集合 A 调用 `issubset()`方法时，该方法接受集合 B 作为输入参数，如果集合 A 是 B 的子集，则返回`True`，否则返回`False`。

您可以使用 `issubset()`方法来检查 python 中的子集，如下所示。

```py
A = {1, 2, 3, 4, 5, 6, 7, 8}
B = {2, 4, 6, 8}
C = {0, 1, 2, 3, 4}
print("Set {} is: {}".format("A", A))
print("Set {} is: {}".format("B", B))
print("Set {} is: {}".format("C", C))
print("Set B is subset of A :", B.issubset(A))
print("Set C is subset of A :", C.issubset(A))
print("Set B is subset of C :", B.issubset(C))
```

输出:

```py
Set A is: {1, 2, 3, 4, 5, 6, 7, 8}
Set B is: {8, 2, 4, 6}
Set C is: {0, 1, 2, 3, 4}
Set B is subset of A : True
Set C is subset of A : False
Set B is subset of C : False
```

## 结论

在本文中，我们讨论了在 python 中检查子集的方法。要了解更多关于集合的知识，你可以阅读这篇关于 python 中的集合理解的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。