# 使用第一个元素对元组列表进行排序

> 原文：<https://www.askpython.com/python/tuple/sort-a-list-of-tuples>

嘿皮托尼斯！今天我们将学习一些关于基本数据结构**列表**和**元组**的高级概念。在本文中，主要关注的是如何使用第一个元素对元组列表进行排序。所以让我们开始吧。

## Python 中的列表是什么？

[列表](https://www.askpython.com/python/list/python-list)是一种基本的数据结构，在序列中保存不同数据类型的元素。序列不需要遵循顺序。python 中的列表跳过了编程时可能会出现问题的以下几点。

1.  固定大小/非动态。
2.  无法保存不同数据类型的对象/元素。
3.  运行时需要一个循环来显示整个数组。
4.  更长更复杂的代码操作。

相反，列表具有以下特征:

1.  我们可以添加、删除、编辑列表中任何地方的元素。
2.  **排序很简单**–使用内置的 sort()方法，我们可以对元素进行升序和降序排序。但是，为了排序，元素应该是相似的数据类型。
3.  **动态**–当我们在列表中添加一个特定的元素时，Python 会为每次添加的下一个元素保留额外的内存空间。这使得它充满活力。
4.  **易于访问**–通过调用列表变量，整个列表显示在屏幕上。

**代码:**

```py
list_1 = [34, 4, 5, -4, 5]  # list of integers
list_2 = ["one", "two", "three"]  # list of strings
list_3 = [45.7, 4.9, 0.0, -8]  # list of floating point numbers
list_4 = ['a', 'b', 'c', 'd'] # list of characters
list_5 = ["sam", 45, 90.0, 'a']    # a list of elements of different data types

```

当我们在一行中打印这些元素时，我们得到以下输出

**输出:**

```py
>>> [34, 4, 5, -4, 5]
>>> ["one", "two", "three"]
>>> [45.7, 4.9, 0.0, -8]
>>> ['a', 'b', 'c', 'd'] 
>>> ["sam", 45, 90.0, 'a']

```

这是操纵数据最有利可图的方式之一。有一些内置的方法，其中某些操作如搜索、排序等是可能的。

## Python 中的元组是什么？

[Python 中的元组](https://www.askpython.com/python/tuple/python-tuple)是一种**列表式的** **数据结构**。只是语法和一些操作不同。元组只是帮助我们将不同数据类型的特定元素封装到一个封闭的**不可编辑的**容器中。

以下是元组的特征:

1.  不可变:一旦声明，我们就不能编辑它。我们既不能添加也不能从中删除项目。
2.  **固定大小:**我们不能对其进行任何大小更改。这是由于元组的不变性。
3.  **可以保存不同数据类型的元素。**

**语法:**

有两种方法可以声明一个元组:

```py
tup_1 = 1, 2, 3, 4
print(tup_1)
tup_2 = (1, 2, 3, 4)
print(tup_2)

```

**输出:**

```py
>>> (1, 2, 3, 4)
>>> (1, 2, 3, 4)

```

当我们声明一个元组时，添加括号是一个基本的例程。但是，如果我们只是将多个值赋给一个变量，那么默认情况下，Python 解释器会将它视为一个元组。

### 解包元组

当我们使用 iterables 解包元组时。有两种技术。

*   **使用简单变量**

```py
tup_1 = 1, 2, 3, 4
a, b, c, d = tup_1

print("Unpacking values from a tuple.....")
print("Value of a is: ", a)
print("Value of b is: ", b)
print("Value of c is: ", c)
print("Value of d is: ", d)

```

**输出:**

```py
>>> Unpacking values from a tuple.....
>>> Value of a is: 1
>>> Value of b is: 2
>>> Value of c is: 3
>>> Value of d is: 4

```

*   **使用通配符**

```py
a, b, *c, d = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
print("Unpacking values from a tuple.....")
print("Value of a is: ", a)
print("Value of a is: ", b)
print("Value of a is: ", c)
print("Value of a is: ", d)

```

**输出:**

```py
>>> Unpacking values from a tuple.....
>>> Value of a is: 11
>>> Value of b is: 22
>>> Value of c is: (33, 44, 55,  66, 77, 88, 99)
>>> Value of d is: 110

```

这里变量 **c** 的值是一个**元组**。这是因为**通配符“*”**帮助我们将一组值赋给一个变量。

## 在 Python 中轻松排序元组列表

**众所周知，在 Python 中 tuple 是一种不可变类型的数据结构。因此，为了排序的目的，我们需要对它进行一些显式的操作。**

#### 做一个元组列表，然后排序。

**代码:**

```py
tup_1 = (11, -5, -56, 9, 4)
tup_2 = (3, 43, -1, 90.0)
tup_3 = (4.5, 3.0, 9.0, 23.0)

tupleOfTuples = (tup_1, tup_2, tup_3)
print("Printing a tuple of tuples...")
print(tupleOfTuples )

```

**输出:**

```py
>>> Printing a tuple of tuples...
>>> ((11, -5, -56, 9, 4), (3, 43, -1, 90.0), (4.5, 3.0, 9.0, 23.0))

```

Python 提供了创建**嵌套元组(或元组的元组)的灵活性。**所以，我们创建一个，即——**tupleOfTuples，**然后在屏幕上打印出来。然后让我们用一个内置的函数将它重组成一个列表，然后我们将实现排序。

```py
listOfTuples = list(tupleOfTuples)
print(listOfTuples)

listOfTuples.sort()
print("Sorted list of tuples: ", listOfTuples)

```

**输出:**

```py
>>> Raw list of tuples....
>>> [(11, -5, -56, 9, 4), (3, 43, -1, 90.0), (4.5, 3.0, 9.0, 23.0)]
>>> Sorted list of tuples...
>>> [(3, 43, -1, 90.0), (4.5, 3.0, 9.0, 23.0), (11, -5, -56, 9, 4)]

```

这样，我们创建了一个元组列表，并用每个元组的第一个元素对列表进行排序。

#### 转换为字典并执行排序

Python 中的字典是无序的键值对的集合。当我们处理复杂的数据时，这使得它变得简单易用。

**代码**

```py
nums = {4.0:5, 6.0:4, 90:3, 34:5}
a = nums.items()

for k,v in nums.items():
    print(k, ":", v)

print(a)

```

**输出:**

```py
4.0 : 5
6.0 : 4
90 : 3
34 : 5

dict_items([(4.0, 5), (6.0, 4), (90, 3), (34, 5)])

```

现在我们已经得到了元组形式的条目列表，让我们把它做成一个原始列表，然后对它进行排序。

```py
a = list(a)
print("Items list:", a)
print("Sorting the list of items...")
a.sort()
print("Returning a sorted list...")
print(a)

```

**输出:**

```py
Items list: [(4.0, 5), (6.0, 4), (90, 3), (34, 5)]
Sorting the list of items...
Returning a sorted lis... 
[(4.0, 5), (6.0, 4), (34, 5), (90, 3)]

```

让我们反复检查我们是对还是错…

**代码:**

```py
for i in a:
    print(i)

```

这里创建一个 for 循环，遍历列表 a 的每个元素。如果每个元素的第一个元素大于另一个元素的前一个元素，那么我们可以说元组列表是排序的。

```py
(4.0, 5)
(6.0, 4)
(34, 5)
(90, 3)

```

### 对元组列表进行排序的完整代码

**代码 1:**

```py
tup_1 = (11, -5, -56, 9, 4)
tup_2 = (3, 43, -1, 90.0)
tup_3 = (4.5, 3.0, 9.0, 23.0)

tupleOfTuples = (tup_1, tup_2, tup_3)
print("Printing a tuple of tuples...")
print(tupleOfTuples )

listOfTuples = list(tupleOfTuples)
print(listOfTuples)

listOfTuples.sort()
print("Sorted list of tuples: ", listOfTuples)

```

**代码 2:**

```py
nums = {4.0:5, 6.0:4, 90:3, 34:5}
a = nums.items()

for k,v in nums.items():
    print(k, ":", v)

print(a)

a = list(a)
print("Items list:", a)
print("Sorting the list of items...")
a.sort()
print("Returning a sorted list...")
print(a)

for i in a:
    print(i)

```

## 结论

这样，我们看到了如何通过两种方式使用第一个元素对元组列表进行排序。本文还介绍了如何声明列表和元组的基础知识。所有这些都增加了对 python 中这些数据结构的基本理解。所有最好的和快乐的编码。