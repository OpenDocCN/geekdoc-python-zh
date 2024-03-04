# 获取 Python 中列表的最后一个元素

> 原文：<https://www.pythonforbeginners.com/basics/get-the-last-element-of-a-list-in-python>

列表是 python 程序中最常用的数据结构之一。在本文中，我们将研究用 python 获取列表最后一个元素的不同方法。为此，我们将使用索引、pop()方法、切片和反向迭代器等方法。

## 使用 Python 中的索引获取列表的最后一个元素

python 中的索引是一种从列表中访问元素的方式。在 python 中，我们可以使用正索引，也可以使用负索引。正索引从零开始，对应于列表的第一个元素，列表的最后一个元素由索引“listLen-1”标识，其中“listLen”是列表的长度。

或者，负索引从-1 开始，对应于列表的最后一个元素。一直到"-listLen "，其中 listLen 是列表的长度。索引“-listLen”对应于列表中的第一个元素。

要获得一个列表的最后一个元素，我们可以首先使用 len()函数找到列表的长度。然后，我们可以访问索引“listLen-1”处的列表的最后一个元素，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("Given List is:", myList)
listLen = len(myList)
lastElement = myList[listLen - 1]
print("Last element of the list is:", lastElement)
```

输出:

```py
Given List is: [1, 2, 3, 4, 5, 6, 7]
Last element of the list is: 7
```

或者，我们可以使用负索引来访问列表的最后一个元素。列表的最后一个元素在索引-1 处，可以如下访问。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("Given List is:", myList)
lastElement = myList[- 1]
print("Last element of the list is:", lastElement)
```

输出:

```py
Given List is: [1, 2, 3, 4, 5, 6, 7]
Last element of the list is: 7
```

我们可以看到，在使用负索引时，我们不必计算列表的长度。

## 使用 pop()方法

pop()方法用于从指定的索引中移除列表中的任何元素。它将元素的索引作为可选的输入参数，并在从列表中删除元素后，返回指定索引处的元素。如果没有传递输入参数，它将在删除后返回列表的最后一个元素。

我们可以使用 pop()方法获取列表的最后一个元素，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("Given List is:", myList)
lastElement = myList.pop()
print("Last element of the list is:", lastElement)
```

输出:

```py
Given List is: [1, 2, 3, 4, 5, 6, 7]
Last element of the list is: 7
```

请记住，pop()方法还会删除被访问的元素。所以，只有当你还想删除列表的最后一个元素时，才使用这个方法。

## 使用 Python 中的切片获取列表的最后一个元素

在 python 中，切片是创建字符串或列表的一部分的操作。通过切片，我们可以访问任何字符串、元组或列表的不同部分。为了对名为的列表执行切片，我们使用语法 listName[start，end，interval]，其中“start”和“end”分别是切片列表在原始列表中开始和结束的索引。“间隔”用于选择序列中的元素。从列表中的索引处选择元素，这些索引距离起始索引是“间隔”的整数倍。

为了使用 python 中的[切片来访问列表的最后一个元素，我们可以以这样的方式对列表进行切片，使得它只包含最后一个元素。然后我们可以如下访问该元素。](https://www.pythonforbeginners.com/dictionary/python-slicing)

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("Given List is:", myList)
lastElement = myList[-1:][0]
print("Last element of the list is:", lastElement)
```

输出:

```py
Given List is: [1, 2, 3, 4, 5, 6, 7]
Last element of the list is: 7
```

## 使用反向迭代器获取列表的最后一个元素

我们可以使用反向迭代器来获取列表的最后一个元素。要创建反向迭代器，我们可以使用 reversed()方法。reversed()方法接受任何 iterable 对象作为输入，并返回 iterable 的反向迭代器。

为了获得列表的最后一个元素，我们将首先使用 reversed()方法创建列表的反向迭代器。然后我们将访问反向迭代器的第一个元素，这将是原始列表的最后一个元素。这可以如下进行。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("Given List is:", myList)
reverseIter = reversed(myList)
lastElement = next(reverseIter)
print("Last element of the list is:", lastElement) 
```

输出:

```py
Given List is: [1, 2, 3, 4, 5, 6, 7]
Last element of the list is: 7
```

## 使用 itemgetter 获取列表的最后一个元素

我们可以创建一个 itemgetter 对象来访问列表的最后一个元素。itemgetter()方法是在 python 的运算符模块中定义的。itemgetter()方法将索引作为输入，创建一个可调用的对象。callable 对象将 iterable 作为输入，并提取指定索引处的元素。

要访问列表的最后一个元素，我们可以调用 itemgetter()方法，输入索引为-1，然后我们可以访问列表的最后一个元素，如下所示。

```py
import operator
myList = [1, 2, 3, 4, 5, 6, 7]
print("Given List is:", myList)
lastElement = operator.itemgetter(-1)(myList)
print("Last element of the list is:", lastElement)
```

输出:

```py
Given List is: [1, 2, 3, 4, 5, 6, 7]
Last element of the list is: 7
```

## 结论

在本文中，我们看到了在 python 中获取列表最后一个元素的不同方法。要阅读更多关于列表的内容，请阅读这篇关于 python 中的列表理解的文章。请继续关注更多内容丰富的文章。