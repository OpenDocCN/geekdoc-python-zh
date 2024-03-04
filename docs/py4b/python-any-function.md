# Python any()函数

> 原文：<https://www.pythonforbeginners.com/basics/python-any-function>

在 python 中，我们通常使用比较运算符和逻辑运算符来检查不同数量元素的条件。如果您必须检查元素列表中的条件，该怎么办？在本文中，我们将讨论 python 中的 any()函数。我们还将看到如何对不同的可迭代对象使用`any()`函数。

## Python 中的 any()是什么？

any()函数用于[检查在一个 iterable 对象中是否存在一个元素](https://www.pythonforbeginners.com/basics/check-if-a-list-has-duplicate-elements)，该元素的值是否为 True。`any()`函数将列表、元组、集合、字典或字符串等可迭代对象作为其输入参数。执行后，如果 iterable 中至少有一个元素的值为`True`，它将返回`True`。否则返回`False`。您可以在下面的示例中观察到这一点。

```py
myList1 = [1, 2, 3, 4]
myList2 = [False, False, False]
myList3 = []
print("The list is:", myList1)
output = any(myList1)
print("List contains one or more  elements that evaluate to True:", output)
print("The list is:", myList2)
output = any(myList2)
print("List contains one or more  elements that evaluate to True:", output)
print("The list is:", myList3)
output = any(myList3)
print("List contains one or more  elements that evaluate to True:", output)
```

输出:

```py
The list is: [1, 2, 3, 4]
List contains one or more  elements that evaluate to True: True
The list is: [False, False, False]
List contains one or more  elements that evaluate to True: False
The list is: []
List contains one or more  elements that evaluate to True: False
```

你可以把`any()`函数的工作理解为`or`操作符的一个应用。对于具有元素`element1, element2, element3,.... elementN`的可迭代对象，使用`any()`相当于执行语句 `element1 OR element2 OR element3 OR ….., OR elementN`。

## Python 中带有可迭代对象的 any()函数

当我们将一个列表作为输入参数传递给 any()函数时，如果列表的[元素中至少有一个元素的值为 True，它将返回 True。](https://www.pythonforbeginners.com/basics/get-the-last-element-of-a-list-in-python)

```py
myList1 = [1, 2, 3, 4]
myList2 = [False, False, False]
myList3 = []
print("The list is:", myList1)
output = any(myList1)
print("List contains one or more  elements that evaluate to True:", output)
print("The list is:", myList2)
output = any(myList2)
print("List contains one or more  elements that evaluate to True:", output)
print("The list is:", myList3)
output = any(myList3)
print("List contains one or more  elements that evaluate to True:", output)
```

输出:

```py
The list is: [1, 2, 3, 4]
List contains one or more  elements that evaluate to True: True
The list is: [False, False, False]
List contains one or more  elements that evaluate to True: False
The list is: []
List contains one or more  elements that evaluate to True: False
```

当我们将一个空列表传递给`any()` 函数时，它返回 False。

当我们将任何非空字符串作为输入参数传递给`any()`函数时，它返回`True`。

```py
myStr1="PythonForBeginners"
myStr2=""
print("The string is:", myStr1)
output = any(myStr1)
print("The string contains one or more  elements:", output)
print("The string is:", myStr2)
output = any(myStr2)
print("The string contains one or more  elements:", output)
```

输出:

```py
The string is: PythonForBeginners
The string contains one or more  elements: True
The string is: 
The string contains one or more  elements: False 
```

对于空字符串，`any()`函数返回`False`。

与列表类似，当我们将一个集合作为输入参数传递给 any()函数时，如果集合的[元素中至少有一个元素的值为 True，则返回 True。](https://www.pythonforbeginners.com/basics/how-to-add-an-element-to-a-set-in-python)

```py
mySet1 = {1, 2, 3, 4}
mySet2 = {False}
mySet3 = set()
print("The Set is:", mySet1)
output = any(mySet1)
print("Set contains one or more  elements that evaluate to True:", output)
print("The Set is:", mySet2)
output = any(mySet2)
print("Set contains one or more  elements that evaluate to True:", output)
print("The Set is:", mySet3)
output = any(mySet3)
print("Set contains one or more  elements that evaluate to True:", output)
```

输出:

```py
The Set is: {1, 2, 3, 4}
Set contains one or more  elements that evaluate to True: True
The Set is: {False}
Set contains one or more  elements that evaluate to True: False
The Set is: set()
Set contains one or more  elements that evaluate to True: False
```

当我们将一个空集传递给`any()`函数时，它返回`False`。

## Python 中带字典的 any()函数

当我们将字典作为输入参数传递给`any()`函数时，如果 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中至少有一个键的值为`True`，它将返回`True`。否则返回`False`。

```py
myDict1 = {1: 1, 2: 2, 3: 3, 4: 4}
myDict2 = {False: 1}
myDict3 = {}
print("The Dictionary is:", myDict1)
output = any(myDict1)
print("Dictionary contains one or more  keys that evaluate to True:", output)
print("The Dictionary is:", myDict2)
output = any(myDict2)
print("Dictionary contains one or more  keys that evaluate to True:", output)
print("The Dictionary is:", myDict3)
output = any(myDict3)
print("Dictionary contains one or more  keys that evaluate to True:", output)
```

输出:

```py
The Dictionary is: {1: 1, 2: 2, 3: 3, 4: 4}
Dictionary contains one or more  keys that evaluate to True: True
The Dictionary is: {False: 1}
Dictionary contains one or more  keys that evaluate to True: False
The Dictionary is: {}
Dictionary contains one or more  keys that evaluate to True: False
```

当我们将一个空字典传递给`any()`函数时，它返回`False`。

## 结论

在本文中，我们讨论了 python 中的 any()函数。我们还对不同的可迭代对象使用了`any()`函数，并观察了函数的输出。要了解更多关于 python 编程的知识，你可以在[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)上阅读这篇文章。