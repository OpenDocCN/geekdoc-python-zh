# Python 中的索引——初学者完全指南

> 原文：<https://www.askpython.com/python/list/indexing-in-python>

Python 中的索引是什么？–尽管看起来很简单，但是正确解释 Python 中的索引工作方式可能会有点棘手。因此，请坐下来阅读我们的文章，以便更深入地理解 Python 中的索引。

## 先决条件-什么是可迭代的？

在开始索引之前，我们先了解一下 [iterables](https://www.askpython.com/python/built-in-methods/python-iterator) 是什么，它们的主要功能是什么。索引背后非常需要可重复项的知识。

**那么什么是可迭代的？**

它是 Python 中一种特殊类型的[对象，您可以对其进行迭代。这意味着您可以遍历对象中包含的所有不同元素或实体。使用循环](https://www.askpython.com/python/oops/python-classes-objects)的[可以轻松实现。](https://www.askpython.com/python/python-for-loop)

在这种情况下，所有这些可迭代项所携带的是两个叫做 __iter__()或 __getitem__()的特殊方法，它们实现了 ***序列语义*** 。

```py
#Lists are iterable items in Python. 
randomItems = [4, 6, 2, 56, 23, 623, 453]

#Each individual element inside a list can be accessed using a for loop
for item sin randomItems: 
    print(item)

```

除了列表，字符串和元组在 Python 中也是可迭代的。这是一个如何迭代字符串的例子。

```py
title = "Lose Yourself" 

#Looping through each character in the string
for char in title: 
    print(char)

```

输出:

```py
L
o
s
e

Y
o
u
r
s
e
l
f

```

现在我们对 Python 中的可迭代对象有了一些了解。这与索引有什么关系？

## Python 中的索引是什么？

Python 中的索引是一种通过位置引用 iterable 中单个项的方法。换句话说，您可以在 iterable 中直接访问您选择的元素，并根据您的需要进行各种操作。

在我们讨论 Python 中的索引示例之前，有一件重要的事情需要注意:

在 Python 中，对象是“零索引”的，这意味着位置计数从零开始。许多其他编程语言遵循相同的模式。事实上，你们中的许多人应该已经对它很熟悉了，因为它在互联网上在模因文化中很流行。

如果一个列表中有 5 个元素。则第一个元素(即最左边的元素)保持“第零”位置，随后是第一、第二、第三和第四位置的元素。

```py
fruits = ["apple", "grape", "orange", "guava", "banana"]

#Printing out the indexes of Apples and Banana
print("Index of Apple: ", fruits.index("apple"))
print("Index of Banana: ", fruits.index("banana"))

```

输出:

```py
Index of Apple: 0
Index of Banana: 4

```

当对列表调用 index()方法并将项目名称作为参数传递时，可以显示列表中特定项目的索引。

在下一节中，我们最终将学习如何在 iterable 对象上使用 index()方法。

### 什么是 Python 索引运算符？

Python 索引运算符由左右方括号表示:[]。但是，语法要求您在括号内输入一个数字。

### Python 索引运算符语法

```py
ObjectName[n] #Where n is just an integer number that represents the position of the element we want to access. 

```

## 在 Python 中使用索引的步骤

下面，我们将找出在 Python 中使用索引的例子。

### 1.索引字符串

```py
greetings = "Hello, World!"

print(greetings[0]) #Prints the 0-th element in our string

print(greetings[5]) #Prints the 5-th element in our string

print(greetings[12]) #Prints the 12-th element in our string

```

输出:

```py
H
,
!

```

我们可以清楚地看到我们的 print 函数是如何访问 string 对象中的不同元素来获得我们想要的特定字符的。

### 2.Python 中的负索引

我们最近学习了如何在列表和字符串中使用索引来获取我们感兴趣的特定项目。尽管在我们之前的所有例子中，我们在索引操作符(方括号)中使用了正整数，但这并不是必须的。

通常，如果我们对列表的最后几个元素感兴趣，或者我们只想从相反的一端索引列表，我们可以使用负整数。这种从另一端转位的过程称为负转位。

**注意:在负索引中，最后一个元素用-1 而不是-0 表示。**

```py
letters = ['a', 's', 'd', 'f']

#We want to print the last element of the list
print(letters[-1]) #Notice we didn't use -0 

#To print the 2nd last element from an iterable
print(letters[-2])

```

输出:

```py
f
d

```

## 结论

希望您喜欢我们的文章，并学会了如何在自己的代码中使用索引。快乐编码。