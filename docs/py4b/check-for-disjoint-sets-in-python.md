# 检查 Python 中不相交的集合

> 原文：<https://www.pythonforbeginners.com/basics/check-for-disjoint-sets-in-python>

在 python 中，集合是用于存储唯一不可变对象的容器对象。在本文中，我们将讨论 python 中的不相交集合。我们还将讨论 python 中检查不相交集合的不同方法。

## 什么是不相交集合？

如果两个集合没有任何公共元素，则称它们是不相交的。如果两个给定的集合之间存在任何公共元素，那么它们就不是不相交的集合。

假设我们已经设置了 A、B 和 C，如下所示。

```py
A = {1, 2, 3, 4, 5, 6, 7, 8}
B = {2, 4, 6, 8, 10, 12}
C = {10, 20, 30, 40, 50}
```

在这里，您可以观察到集合 A 和集合 B 有一些共同的元素，即 2、4、6 和 8。因此，它们不是不相交的集合。另一方面，集合 A 和集合 C 没有公共元素。因此，集合 A 和集合 C 将被称为不相交集合。

## Python 中如何检查不相交集合？

要检查不相交的集合，我们只需检查给定集合中是否存在任何公共元素。如果两个集合中有公共元素，那么这两个集合就不是不相交的集合。否则，它们将被视为不相交的集合。

为了实现这个逻辑，我们将声明一个变量`isDisjoint`并将其初始化为`True`，假设两个集合都是不相交的集合。之后，我们将使用 for 循环遍历其中一个输入集。在遍历时，我们将检查集合中的每个元素是否存在于另一个集合中。如果我们在第一个集合中找到任何属于第二个集合的元素，我们将把值`False`赋给变量`isDisjoint`，表示这些集合不是不相交的集合。

如果输入集之间没有公共元素，那么在执行 for 循环后，`isDisjoint`变量将保持为`True`。因此，表示这些集合是不相交的集合。

```py
def checkDisjoint(set1, set2):
    isDisjoint = True
    for element in set1:
        if element in set2:
            isDisjoint = False
            break
    return isDisjoint

A = {1, 2, 3, 4, 5, 6, 7, 8}
B = {2, 4, 6, 8, 10, 12}
C = {10, 20, 30, 40, 50}
print("Set {} is: {}".format("A", A))
print("Set {} is: {}".format("B", B))
print("Set {} is: {}".format("C", C))
print("Set A and B are disjoint:", checkDisjoint(A, B))
print("Set A and C are disjoint:", checkDisjoint(A, C))
print("Set B and C are disjoint:", checkDisjoint(B, C))
```

输出:

```py
Set A is: {1, 2, 3, 4, 5, 6, 7, 8}
Set B is: {2, 4, 6, 8, 10, 12}
Set C is: {40, 10, 50, 20, 30}
Set A and B are disjoint: False
Set A and C are disjoint: True
Set B and C are disjoint: False
```

建议阅读:[Python 中的聊天应用](https://codinginfinite.com/python-chat-application-tutorial-source-code/)

## 使用 isdisjoint()方法检查不相交的集合

除了上面讨论的方法，我们可以使用`isdisjoint()`方法来检查 python 中不相交的集合。当在一个集合上调用时，`isdisjoint()` 方法将另一个集合作为输入参数。执行后，如果集合是不相交的集合，则返回`True`。否则返回`False`。您可以在下面的示例中观察到这一点。

```py
A = {1, 2, 3, 4, 5, 6, 7, 8}
B = {2, 4, 6, 8, 10, 12}
C = {10, 20, 30, 40, 50}
print("Set {} is: {}".format("A", A))
print("Set {} is: {}".format("B", B))
print("Set {} is: {}".format("C", C))
print("Set A and B are disjoint:", A.isdisjoint(B))
print("Set A and C are disjoint:", A.isdisjoint(C))
print("Set B and C are disjoint:", B.isdisjoint(C))
```

输出:

```py
Set A is: {1, 2, 3, 4, 5, 6, 7, 8}
Set B is: {2, 4, 6, 8, 10, 12}
Set C is: {40, 10, 50, 20, 30}
Set A and B are disjoint: False
Set A and C are disjoint: True
Set B and C are disjoint: False
```

## 结论

在本文中，我们讨论了在 python 中检查不相交集合的两种方法。要了解更多关于集合的知识，你可以阅读这篇关于 python 中[集合理解](https://www.pythonforbeginners.com/basics/set-comprehension-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。