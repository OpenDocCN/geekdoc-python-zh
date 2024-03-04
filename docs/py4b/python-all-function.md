# Python all()函数

> 原文：<https://www.pythonforbeginners.com/basics/python-all-function>

在 python 中，我们通常使用比较运算符和逻辑运算符来检查不同数量元素的条件。如果您必须检查元素列表中的条件，该怎么办？在本文中，我们将讨论 python 中的`all()` 函数。我们还将看到如何对不同的可迭代对象使用`all()`函数。

## Python 中的`all()`函数是什么？

`all()`函数用于检查一个 iterable 对象中的所有元素是否都等于`True`。`all()`函数将列表、元组、集合、字典或字符串等可迭代对象作为其输入参数。执行后，如果 iterable 的所有元素的值都是`True`，它将返回`True`。否则返回`False`。您可以在下面的示例中观察到这一点。

```py
myList1 = [1, 2, 3, 4]
myList2 = [1, True, False]
myList3 = []
print("The list is:", myList1)
output = all(myList1)
print("All the elements of the list evaluate to True:", output)
print("The list is:", myList2)
output = all(myList2)
print("All the elements of the list evaluate to True:", output)
print("The list is:", myList3)
output = all(myList3)
print("All the elements of the list evaluate to True:", output)
```

输出:

```py
The list is: [1, 2, 3, 4]
All the [elements of the list](https://www.pythonforbeginners.com/basics/find-the-index-of-an-element-in-a-list) evaluate to True: True
The list is: [1, True, False]
All the elements of the list evaluate to True: False
The list is: []
All the [elements of the list](https://www.pythonforbeginners.com/basics/find-the-index-of-an-element-in-a-list) evaluate to True: True
```

你可以把`all()`函数的工作理解为`and`操作符的一个应用。对于具有元素`element1, element2, element3,.... elementN`的可迭代对象，使用`all()`函数相当于执行语句 `element1 AND element2 AND element3 AND ….., AND elementN`。

## 带有 iterable 对象的 all()函数

当我们将一个列表作为输入参数传递给 all()函数时，如果列表的所有[元素的值都为 True，那么它将返回 True。](https://www.pythonforbeginners.com/basics/find-the-index-of-an-element-in-a-list)

```py
myList1 = [1, 2, 3, 4]
myList2 = [1, True, False]
myList3 = []
print("The list is:", myList1)
output = all(myList1)
print("All the elements of the list evaluate to True:", output)
print("The list is:", myList2)
output = all(myList2)
print("All the elements of the list evaluate to True:", output)
print("The list is:", myList3)
output = all(myList3)
print("All the elements of the list evaluate to True:", output)
```

输出:

```py
The list is: [1, 2, 3, 4]
All the elements of the list evaluate to True: True
The list is: [1, True, False]
All the elements of the list evaluate to True: False
The list is: []
All the elements of the list evaluate to True: True
```

当我们将一个空列表传递给`all()` 函数时，它返回`True`。但是，如果列表中有一个元素的值为`False`，那么`all()`函数将返回`False`。

当我们将任何字符串作为输入参数传递给`all()` 函数时，它返回`True`。

```py
myStr1 = "PythonForBeginners"
myStr2 = ""
print("The string is:", myStr1)
output = all(myStr1)
print("The output is:", output)
print("The string is:", myStr2)
output = all(myStr2)
print("The output is:", output)
```

输出:

```py
The string is: PythonForBeginners
The output is: True
The string is: 
The output is: True
```

对于空字符串，`all()`函数返回`True`。

类似于列表，当我们将一个集合作为输入参数传递给`all()`函数时，如果该集合的所有元素的值都为`True`，它将返回`True`。

```py
mySet1 = {1, 2, 3, 4}
mySet2 = {1, 2, True, False}
mySet3 = set()
print("The Set is:", mySet1)
output = all(mySet1)
print("All the elements of the set evaluate to True:", output)
print("The Set is:", mySet2)
output = all(mySet2)
print("All the elements of the set evaluate to True:", output)
print("The Set is:", mySet3)
output = all(mySet3)
print("All the elements of the set evaluate to True:", output)
```

输出:

```py
The Set is: {1, 2, 3, 4}
All the elements of the set evaluate to True: True
The Set is: {False, 1, 2}
All the elements of the set evaluate to True: False
The Set is: set()
All the elements of the set evaluate to True: True
```

当我们将一个空集传递给`all()`函数时，它返回`True`。

## Python 中带字典的 all()函数

当我们将一个字典作为输入参数传递给`all()` 函数时，如果 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)的所有键都评估为`True`，它将返回`True`。否则返回`False`。

```py
myDict1 = {1: 1, 2: 2, 3: 3, True: 4}
myDict2 = {False: 1, 1: 2, True: False}
myDict3 = {}
print("The Dictionary is:", myDict1)
output = all(myDict1)
print("All the keys of the dictionary evaluate to True:", output)
print("The Dictionary is:", myDict2)
output = all(myDict2)
print("All the keys of the dictionary evaluate to True:", output)
print("The Dictionary is:", myDict3)
output = all(myDict3)
print("All the keys of the dictionary evaluate to True:", output)
```

输出:

```py
The Dictionary is: {1: 4, 2: 2, 3: 3}
All the keys of the dictionary evaluate to True: True
The Dictionary is: {False: 1, 1: False}
All the keys of the dictionary evaluate to True: False
The Dictionary is: {}
All the keys of the dictionary evaluate to True: True
```

当我们将一个空字典传递给`all()`函数时，它返回 True。

## 结论

在本文中，我们讨论了 python 中的 all()函数。我们还对不同的可迭代对象使用了`all()`函数，并观察了函数的输出。要了解更多关于 python 编程的知识，你可以在[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)上阅读这篇文章。