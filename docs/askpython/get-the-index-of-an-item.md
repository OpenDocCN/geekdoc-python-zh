# 获取 Python 列表中项目的索引——3 种简单的方法

> 原文：<https://www.askpython.com/python/list/get-the-index-of-an-item>

读者朋友们，你们好！希望你们都过得好。在本文中，我们将重点关注获取 Python 列表中元素索引的**不同技术。**

那么，让我们开始吧。

* * *

## 什么是 Python 列表？

一个 [Python 列表](https://www.askpython.com/python/list/python-list)是一个数据结构，它有效地扮演了一个[数组](https://www.askpython.com/python/array/python-array-examples)的角色。此外,“列表”以动态方式存储元素，并可用于存储不同类型的元素，这与数组不同。

因此，列表可以被认为是数组数据结构的更好的替代物，并且可以保存完全不同的元素。

* * *

## 如何获取 Python 列表中某项的索引？

理解了 Python List 的工作原理之后，现在让我们开始用不同的方法来获取列表中一个条目的索引。

### 方法一:列表理解

[Python List Comprehension](https://www.askpython.com/python/list/python-list-comprehension) 可用于利用列表中特定元素的所有出现的索引列表。

**语法:**

```py
[expression for element in iterator if condition]

```

使用`List comprehension`，我们可以得到一个项目在列表中所有出现的索引值，即**位置。**

**举例:**

```py
lst = [10,20,30,10,50,10,45,10] 

print ("List : " ,lst) 

res = [x for x in range(len(lst)) if lst[x] == 10] 

print ("Indices at which element 10 is present: " + str(res)) 

```

**输出:**

```py
List :  [10, 20, 30, 10, 50, 10, 45, 10]
Indices at which element 10 is present: [0, 3, 5, 7]

```

* * *

### 方法 2:使用 index()方法

Python 内置的 index()方法可以用来获取列表中特定元素的索引值。

**语法:**

```py
index(element,start,end)

```

`start` 和`end` 参数是可选的，代表执行搜索的位置范围。

与其他方法不同，index()方法只返回列表中特定项目第一次出现的索引值。

如果所提到的元素不在列表中，则引发`ValueError exception`。

**举例:**

```py
lst = [10,20,30,10,50,10,45,10] 

print ("List : " ,lst) 

print("Index at which element 10 is present :",lst.index(10)) 

```

**输出:**

```py
List :  [10, 20, 30, 10, 50, 10, 45, 10]
Index at which element 10 is present : 0

```

* * *

### 方法 3:使用 enumerate()函数

[Python enumerate()方法](https://www.askpython.com/python/built-in-methods/python-enumerate-method)也可以用于**返回特定元素**在列表中所有出现的索引位置。

**举例:**

```py
lst = [10,20,30,10,50,10,45,10] 

print ("List : " ,lst) 

res = [x for x, z in enumerate(lst) if z == 10] 

print ("Indices at which element 10 is present: " + str(res)) 

```

这里，`enumerate() method`设置了一个计数器，它在每次成功搜索该特定项目后递增，并返回它的索引值。

**输出:**

```py
List :  [10, 20, 30, 10, 50, 10, 45, 10]
Indices at which element 10 is present: [0, 3, 5, 7]

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。在那之前，学习愉快！！

* * *

## 参考

*   [如何获取 Python 列表中某项的索引— StackOverflow](https://stackoverflow.com/questions/176918/finding-the-index-of-an-item-in-a-list)