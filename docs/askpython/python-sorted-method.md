# 使用 Python sorted()方法

> 原文：<https://www.askpython.com/python/built-in-methods/python-sorted-method>

## 介绍

在本教程中，我们将讨论 Python sorted()方法。

任何编程语言中的排序都是对任何可迭代对象广泛执行的操作。这种排序可以是任何顺序，比如升序或降序。Python 在进行这种类型的排序操作时提供了广泛的选项。其中之一就是`sorted()`法。

现在让我们进入正题，看一些例子。

## 了解 Python sorted()方法

Python `sorted()`方法按照升序或降序对传递的 **iterable** 进行排序(按照指定的方式)，并以列表的形式返回值(已排序)。下面给出了使用该方法的**语法**。

```py
sorted(iterable, *, key=None, reverse=False)

```

这里，

*   `Iterable`可以是列表、元组或集合。这个 iterable 由方法排序并返回，
*   `key`–默认设置为无。它确定对值进行排序所基于的参数，
*   `reverse`–它接受布尔值，即真或假。如果传递的值是`True`，则 iterable 以逆序或降序排序。然而，对于`False`或默认条件，排序按升序进行。

## Python 中 sorted()方法的工作原理

因此，对于作为参数传递的不同值，我们实际上可以按升序或降序对任何可迭代对象进行排序。同样在**的基础上自定义**或**内置**函数。

让我们看看如何在 Python 中使用`sorted()`方法。以各种方式对可迭代对象进行排序。

### 使用 Python sorted()按升序排序

下面的代码演示了如何使用 Python `sorted()`方法对任何 iterable 进行升序排序(普通排序)。

```py
#initialisation of variables
list1= [2,7,6,24,73,23,57]
tup1= ('d','c','a','b')

print(sorted(list1)) #sorted list1
print(tuple(sorted(tup1))) #sorted list is type casted to tuple

```

**输出**:

```py
[2, 6, 7, 23, 24, 57, 73]
('a', 'b', 'c', 'd')

```

这里，

*   首先，我们初始化两个可迭代的对象。一个**列表**，另一个 **[元组](https://www.askpython.com/python/tuple/python-tuple)** ，
*   然后我们直接将它们传递给没有其他参数的`sorted()`方法(默认情况下，key 是 **none** ，reverse 被设置为 **false** )，
*   在打印结果时，注意我们已经对元组的情况进行了类型转换。这样做是因为 Python `sorted()`方法以列表的形式返回排序的 iterable。

从上面的输出中我们可以看到，列表和元组都按照我们想要的方式进行了排序。

### 使用 Python sorted()进行降序排序

现在让我们看看如何使用 Python 的`sorted()`方法以降序或**反转**的方式进行排序。

```py
#initialisation of variables
list1 = [2,7,6,24,73,23,57]
tup1 = ('d','c','a','b')

print(sorted(list1, reverse= True)) #sorted list1 in reversed order
print(tuple(sorted(tup1, reverse= True))) #reversed sorted list is type casted to tuple

```

**输出**:

```py
[73, 57, 24, 23, 7, 6, 2]
('d', 'c', 'b', 'a')

```

类似于我们前面的例子，我们初始化并传递一个列表和一个元组给各自的`sorted()`方法。这里唯一的变化是，这次我们将该方法的**反向**参数设置为`True`。这导致结果列表和元组以相反的方式排序。

### 使用 sorted()中的键进行自定义排序

在本节中，我们将重点关注`sorted()`方法的**键**参数。正如我们前面看到的，我们可以将任何用户定义的或内置的函数传递给`sorted()`方法，作为确定基于哪些值进行排序的关键。

仔细看下面的例子，它根据对应列表项 tuple 的第三个( **3rd** )元素对一个元组列表进行排序。

```py
#initialisation of variables
list1 = [(9,8,7), (6,5,4), (3,2,1)]

def f(list):
    return list[2]
print("Sorting o the basis of key function: ", sorted(list1, key=f))

```

**输出**:

```py
Sorting o the basis of key function:  [(3, 2, 1), (6, 5, 4), (9, 8, 7)]

```

这里，`f()`是一个用户自定义函数，返回传递的元组的第**第 3 个**元素。为`sorted()`方法设置`key`参数确保列表的排序 **list1** 基于每个元组元素的第 3 个元素进行。

如果键没有被传递，对于默认的 **none** 值，`sorted()`将按照 list1 中每个元组的第**1**元素对列表进行排序。

## 结论

因此，在本教程中，我们学习了使用 Python `sorted()`方法进行排序，以及它的各种用途。为了更好地理解，我们建议你自己练习这些代码。如有任何问题，欢迎使用下面的评论。

## 参考

*   Python sorted()方法——日志开发帖子，
*   [排序](https://docs.python.org/3/howto/sorting.html)–Python 文档。