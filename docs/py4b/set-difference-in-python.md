# 在 Python 中设置差异

> 原文：<https://www.pythonforbeginners.com/basics/set-difference-in-python>

集合用于存储唯一的对象。有时，我们可能需要在一个集合中找到在另一个集合中不存在的元素。为此，我们使用集合差运算。在本文中，我们将讨论什么是集合差。我们还将讨论在 python 中寻找集合差异的方法。

## 集合差是什么？

当我们给定两个集合 A 和 b 时。集合差(A-B)是由属于 A 但不在集合 b 中的所有元素组成的集合

同样，集合差(B-A)是由属于 B 但不在集合 A 中的所有元素组成的集合。

考虑以下集合。

`A={1,2,3,4,5,6,7}`

`B={5,6,7,8,9,10,11}`

这里，[集合 A-B 将包含元素](https://www.pythonforbeginners.com/basics/how-to-add-an-element-to-a-set-in-python) 1、2、3 和 4，因为这些元素存在于集合 A 中但不属于集合 B。类似地，集合`B-A`将包含元素 8、9、10、11，因为这些元素存在于集合 B 中但不属于集合 A。

现在让我们讨论在 python 中寻找集合差异的方法。

## 如何在 Python 中求集合差？

给定集合 A 和 B，如果我们想找到集合差 A-B，我们将首先创建一个名为`output_set`的空集。之后，我们将使用 for 循环遍历集合 A。在遍历时，我们将[检查每个元素](https://www.pythonforbeginners.com/basics/check-if-a-list-has-duplicate-elements)是否出现在集合 B 中。如果集合 A 中的任何元素不属于集合 B，我们将使用`add()`方法将该元素添加到`output_set`中。

执行 for 循环后，我们将在`output_set`中获得设定的差值 A-B。您可以在下面的示例中观察到这一点。

```py
A = {1, 2, 3, 4, 5, 6, 7}
B = {5, 6, 7, 8, 9, 10, 11}
output_set = set()
for element in A:
    if element not in B:
        output_set.add(element)
print("The set A is:", A)
print("The set B is:", B)
print("The set A-B is:", output_set)
```

输出:

```py
The set A is: {1, 2, 3, 4, 5, 6, 7}
The set B is: {5, 6, 7, 8, 9, 10, 11}
The set A-B is: {1, 2, 3, 4}
```

如果我们想找到集合差 B-A，我们将使用 for 循环遍历集合 B。在遍历时，我们将检查每个元素是否出现在集合 A 中。如果集合 B 中的任何元素不属于集合 A，我们将使用`add()` 方法将该元素添加到`output_set`中。

在执行 for 循环后，我们将在`output_set`中得到设定的差值 B-A。您可以在下面的示例中观察到这一点。

```py
A = {1, 2, 3, 4, 5, 6, 7}
B = {5, 6, 7, 8, 9, 10, 11}
output_set = set()
for element in B:
    if element not in A:
        output_set.add(element)
print("The set A is:", A)
print("The set B is:", B)
print("The set B-A is:", output_set)
```

输出:

```py
The set A is: {1, 2, 3, 4, 5, 6, 7}
The set B is: {5, 6, 7, 8, 9, 10, 11}
The set B-A is: {8, 9, 10, 11}
```

## 使用 Python 中的 Difference()方法查找集合差异

Python 为我们提供了`difference()`方法来寻找集合差异。在集合 A 上调用`difference()` 方法时，将集合 B 作为输入参数，计算集合差，并返回包含集合中元素的集合(A-B)。您可以在下面的示例中观察到这一点。

```py
A = {1, 2, 3, 4, 5, 6, 7}
B = {5, 6, 7, 8, 9, 10, 11}
output_set = A.difference(B)
print("The set A is:", A)
print("The set B is:", B)
print("The set A-B is:", output_set)
```

输出:

```py
The set A is: {1, 2, 3, 4, 5, 6, 7}
The set B is: {5, 6, 7, 8, 9, 10, 11}
The set A-B is: {1, 2, 3, 4}
```

## 结论

在本文中，我们讨论了如何在 python 中找到集合差。要了解更多关于集合的知识，你可以阅读这篇关于 python 中[集合理解](https://www.pythonforbeginners.com/basics/set-comprehension-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。