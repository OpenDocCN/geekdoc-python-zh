# 在 Python 中从集合中移除元素

> 原文：<https://www.pythonforbeginners.com/basics/remove-elements-from-a-set-in-python>

我们在 python 中使用集合来存储唯一的不可变对象。在本文中，我们将讨论如何在 python 中从集合中移除元素。为此，我们将讨论使用不同方法的四种途径。

## 使用 pop()方法从集合中移除元素

我们可以使用 pop()方法从集合中移除元素。当在集合上调用时，`pop()`方法不接受输入参数。执行后，它从集合中移除一个[随机元素](https://www.pythonforbeginners.com/basics/select-random-element-from-a-list-in-python)，并返回该元素。您可以在下面的示例中观察到这一点。

```py
mySet = {1, 2, 3, 4, 5, 6}
print("The input set is:", mySet)
element = mySet.pop()
print("The output set is:", mySet)
print("The popped element is:", element)
```

输出:

```py
The input set is: {1, 2, 3, 4, 5, 6}
The output set is: {2, 3, 4, 5, 6}
The popped element is: 1
```

`pop()`方法只适用于非空集合。如果我们在空集上调用`pop()`方法，它将引发`KeyError`异常，如下所示。

```py
mySet = set()
print("The input set is:", mySet)
element = mySet.pop()
print("The output set is:", mySet)
print("The popped element is:", element)
```

输出:

```py
The input set is: set()
/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (1.26.7) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 3, in <module>
    element = mySet.pop()
KeyError: 'pop from an empty set'
```

方法从集合中删除一个随机元素。如果我们必须删除一个特定的元素，我们可以使用下面讨论的方法。

## 使用 Remove()方法从集合中移除元素

要从集合中删除指定的元素，我们可以使用`remove()` 方法。在集合上调用 `remove()`方法时，该方法将一个元素作为输入参数，并从集合中删除该元素，如下所示。

```py
mySet = {1, 2, 3, 4, 5, 6}
print("The input set is:", mySet)
mySet.remove(3)
print("The output set is:", mySet)
```

输出:

```py
The input set is: {1, 2, 3, 4, 5, 6}
The output set is: {1, 2, 4, 5, 6}
```

如果作为输入传递给`remove()`方法的元素不在集合中，程序将运行到一个`KeyError`异常，如下所示。

```py
mySet = {1, 2, 3, 4, 5, 6}
print("The input set is:", mySet)
mySet.remove(7)
print("The output set is:", mySet)
```

输出:

```py
The input set is: {1, 2, 3, 4, 5, 6}
/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (1.26.7) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 3, in <module>
    mySet.remove(7)
KeyError: 7
```

类似地，如果我们试图使用`remove()`方法从空集合中移除一个元素，它将引发`KeyError`异常。

```py
mySet = set()
print("The input set is:", mySet)
mySet.remove(7)
print("The output set is:", mySet)
```

输出:

```py
The input set is: set()
/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (1.26.7) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 3, in <module>
    mySet.remove(7)
KeyError: 7
```

## 使用 discard()方法从集合中移除元素

当我们试图使用`remove()`方法从集合中删除一个元素，而该元素不在集合中时，程序将运行到一个`KeyError`异常。您可以使用`discard()` 方法而不是`remove()`方法来避免这种情况。当在集合上调用`discard()`方法时，该方法接受一个元素作为输入参数，并从集合中删除该元素，就像在`remove()`方法中一样。

```py
mySet = {1, 2, 3, 4, 5, 6}
print("The input set is:", mySet)
mySet.discard(3)
print("The output set is:", mySet)
```

输出:

```py
The input set is: {1, 2, 3, 4, 5, 6}
The output set is: {1, 2, 4, 5, 6}
```

然而，如果我们尝试用删除集合中不存在的元素，discard()方法不会引发[异常。该设置保持不变，程序不会遇到`KeyError`异常。](https://www.pythonforbeginners.com/error-handling/python-try-and-except)

```py
mySet = {1, 2, 3, 4, 5, 6}
print("The input set is:", mySet)
mySet.discard(7)
print("The output set is:", mySet)
```

输出:

```py
The input set is: {1, 2, 3, 4, 5, 6}
The output set is: {1, 2, 3, 4, 5, 6}
```

## 使用 clear()方法

`clear()` 方法用于一次从任何给定的集合中删除所有元素。当在集合上调用时，`clear()`方法不接受输入参数，并从集合中删除所有元素。您可以在下面的示例中观察到这一点。

```py
mySet = {1, 2, 3, 4, 5, 6}
print("The input set is:", mySet)
mySet.clear()
print("The output set is:", mySet)
```

输出:

```py
The input set is: {1, 2, 3, 4, 5, 6}
The output set is: set()
```

## 结论

在本文中，我们讨论了在 python 中从集合中删除元素的四种方法。要了解更多关于集合的知识，你可以阅读这篇关于 Python 中[集合理解](https://www.pythonforbeginners.com/basics/set-comprehension-in-python)的文章。你可能也会喜欢这篇关于用 Python 阅读[列表的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)