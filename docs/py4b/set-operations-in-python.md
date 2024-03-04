# Python 中的集合运算

> 原文：<https://www.pythonforbeginners.com/basics/set-operations-in-python>

集合是包含唯一元素的容器对象。在本文中，我们将研究各种集合运算，如并集、交集和集合差。我们还将在 python 中实现 set 操作。

## 联合操作

如果我们给定两个集合 A 和 B，这两个集合的并集被计算为包含集合 A 和集合 B 中的元素的集合。我们使用符号**∨**表示集合并集运算。

如果我们有一个集合 C 是 A 和 B 的并集，我们可以写 C = A**∩**B .**例如，假设我们有集合 A={1，2，3，4，5 }，集合 B = {2，4，6，8}，那么集合 C = A**∩**B 将包含元素{1，2，3，4，5，6，8 }。你可以观察到 C 中的每个元素要么属于 A，要么属于 B，要么同时属于 A 和 B。**

**我们可以使用 union()方法在 python 中实现 set union 操作。在集合 A 上调用 union()方法时，它将另一个集合 B 作为输入参数，并返回由 A 和 B 的并集构成的集合。**

```py
`A = {1, 2, 3, 4, 5}
B = {2, 4, 6, 8}
print("Set A is:", A)
print("Set B is:", B)
C = A.union(B)
print("Union of A and B is:", C)` 
```

**输出:**

```py
`Set A is: {1, 2, 3, 4, 5}
Set B is: {8, 2, 4, 6}
Union of A and B is: {1, 2, 3, 4, 5, 6, 8}`
```

**我们可以同时执行两个以上集合的联合。在这种情况下，由 union 运算创建的集合包含参与 union 运算的每个集合中的元素。例如，如果我们设置了 A= {1，2，3，4，5}，B= {2，4，6，8，10}，C={ 7，8，9，10}，那么结果集 D = A**∩**B**∩**C 将包含元素{1，2，3，4，5，6，7，8，9，10 }。**

**我们可以在 python 中执行两个以上集合的并集，方法是在 union()方法上调用其他集合作为输入，如下所示。**

```py
`A = {1, 2, 3, 4, 5}
B = {2, 4, 6, 8}
C = {7, 8, 9, 10}
print("Set A is:", A)
print("Set B is:", B)
print("Set C is:", C)
D = A.union(B, C)
print("Union of A, B and C is:", D)` 
```

**输出:**

```py
`Set A is: {1, 2, 3, 4, 5}
Set B is: {8, 2, 4, 6}
Set C is: {8, 9, 10, 7}
Union of A, B and C is: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}`
```

## **交集运算**

**如果我们给定两个集合 A 和 B，则这两个集合的交集被计算为包含集合 A 和集合 B 中都存在的那些元素的集合。我们使用 **∩** 符号来表示集合交集运算。**

**如果我们有一个集合 C 是 A 和 B 的交集，我们可以写 C= A **∩** B. 例如，假设我们有集合 A={1，2，3，4，5，6 }，集合 B = {2，4，6，8}，那么集合 C= A **∩** B 将包含元素{2，4，6 }。你可以观察到 C 中的每个元素同时属于 A 或者和 b。**

**我们可以使用 intersection()方法在 python 中实现设置交集操作。在集合 A 上调用 intersection()方法时，它将另一个集合 B 作为输入参数，并返回由 A 和 B 的交集形成的集合。**

```py
`A = {1, 2, 3, 4, 5, 6}
B = {2, 4, 6, 8}
print("Set A is:", A)
print("Set B is:", B)
C = A.intersection(B)
print("Intersection of A and B is:", C)` 
```

**输出:**

```py
`Set A is: {1, 2, 3, 4, 5, 6}
Set B is: {8, 2, 4, 6}
Intersection of A and B is: {2, 4, 6}`
```

**我们可以同时执行两个以上集合的交集。在这种情况下，由交集运算创建的集合包含参与交集运算的每个集合中存在的元素。例如，如果我们设置 A= {1，2，3，4，5，6}，设置 B= {2，4，6，8，10}，设置 C={2，4，7，8，9，10}，那么结果集 D= A **∩** B **∩** C 将包含元素{ 2，4}。**

**在 python 中，当对单个集合调用 intersection()方法时，我们可以通过将其他集合作为输入来执行两个以上集合的交集，如下所示。**

```py
`A = {1, 2, 3, 4, 5, 6}
B = {2, 4, 6, 8, 10}
C = {2, 4, 7, 8, 9, 10}
print("Set A is:", A)
print("Set B is:", B)
print("Set C is:", C)
D = A.intersection(B, C)
print("Intersection of A, B and C is:", D)` 
```

**输出:**

```py
`Set A is: {1, 2, 3, 4, 5, 6}
Set B is: {2, 4, 6, 8, 10}
Set C is: {2, 4, 7, 8, 9, 10}
Intersection of A, B and C is: {2, 4}`
```

## **集合差分运算**

**如果给我们两个集合 A 和 B，则集合 A 和集合 B 的差被计算为包含那些存在于集合 A 中但不存在于集合 B 中的元素的集合。我们使用符号**–**表示集合差运算。**

**如果我们有一个集合 C 是 A 和 B 的差，我们可以写成 C = A**–**B .**例如，假设我们设 A={1，2，3，4，5，6 }，集合 B = {2，4，6，8}，那么集合 C = A**–**B 将包含元素{1，3，5 }。你可以观察到，C 中的每个元素都属于 A，但不属于 b。****

****我们可以使用 difference()方法在 python 中实现 set difference 操作。在集合 A 上调用 difference()方法时，该方法将另一个集合 B 作为输入参数，并返回由 A 和 B 的差形成的集合。****

```py
**`A = {1, 2, 3, 4, 5, 6}
B = {2, 4, 6, 8}
print("Set A is:", A)
print("Set B is:", B)
C = A.difference(B)
print("Difference of A and B is:", C)`** 
```

****输出:****

```py
**`Set A is: {1, 2, 3, 4, 5, 6}
Set B is: {8, 2, 4, 6}
Difference of A and B is: {1, 3, 5}`**
```

## ****结论****

****在本文中，我们讨论了不同的集合运算。我们还使用集合在 python 中实现了它们。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)****