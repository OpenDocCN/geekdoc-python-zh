# Python 中的词典理解

> 原文：<https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python>

在 python 中使用字典时，可能会出现这样的情况:我们需要根据某个条件通过包含字典中的某些项来从另一个字典创建一个新的字典，或者我们希望创建一个新的字典，其中包含符合特定条件的键和值。我们可以通过使用 for 循环逐个手工检查字典条目，并将它们添加到新字典中来实现。在本教程中，我们将看到如何使用 python 中一个名为 dictionary comprehension 的简洁结构来达到同样的效果。

## 什么是词典理解？

就像[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)是一种从其他列表创建列表的方法，字典理解是一种从另一个字典或任何其他可迭代对象创建 python 字典的方法。我们可以根据条件语句用原始字典中的相同元素创建新字典，或者根据需要转换字典中的条目。在使用字典理解创建新字典时，我们可以修改字典中条目的键或值或者键和值。

在使用字典理解创建字典时，我们必须从任何字典或任何其他 iterable 中获取一个键值对。字典理解的语法如下。

```py
myDict={key:value for var in iterable}
```

当使用 list 或 tuple 的元素作为键和从这些元素派生的值从 list 或 tuple 创建字典时，我们可以使用下面的语法。

```py
myDict={key:value for var in list_name}
myDict={key:value for var in tuple_name}
```

当从另一个字典创建一个字典时，我们可以使用下面的语法。

```py
myDict={key:value for (key,value) in dict_name.items()}
```

## 如何利用字典理解？

我们可以通过使用列表中的元素来创建字典。假设我们想要创建一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)，用列表中的元素作为键和值，那么我们可以这样做。

```py
myList=[1,2,3,4]
myDict={x:x for x in myList}
print("List is:")
print(myList)
print("Created dictionary is:")
print(myDict)
```

输出:

```py
List is:
[1, 2, 3, 4]
Created dictionary is:
{1: 1, 2: 2, 3: 3, 4: 4}
```

在输出中，我们可以看到创建的字典具有与给定列表中的元素相同的键和值。

我们还可以在创建新字典时，在将列表元素添加到字典之前修改它们。假设我们想创建一个字典，把列表中的元素作为键，它们的平方作为值，我们可以这样做。

```py
myList=[1,2,3,4]
myDict={x:x**2 for x in myList}
print("List is:")
print(myList)
print("Created dictionary is:")
print(myDict)
```

输出:

```py
List is:
[1, 2, 3, 4]
Created dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
```

在输出中，我们可以看到创建的字典将列表中的元素作为键，将它们的方块作为值。

在创建字典时，我们还可以根据条件有选择地使用列表中的条目。假设我们想创建一个字典，它的列表元素是偶数键，它们的平方是值，我们可以这样做。

```py
myList=[1,2,3,4]
myDict={x:x**2 for x in myList if x%2==0}
print("List is:")
print(myList)
print("Created dictionary is:")
print(myDict)
```

输出:

```py
List is:
[1, 2, 3, 4]
Created dictionary is:
{2: 4, 4: 16}
```

在输出中，我们可以看到，在创建的字典中，只有那些偶数的列表元素被用作键，它们的平方是字典的键的相应值。

我们也可以通过使用另一个字典中的条目来创建一个字典。为此，我们可以使用`items()`方法从原始字典中提取键值对。`items()` 方法给出了一个元组列表作为`(key,value)`对，我们可以用它来创建一个新的字典。

例如，我们可以使用字典理解创建一个与给定字典完全一样的新字典，如下所示。

```py
myDict={1:1,2:2,3:3,4:4}
print("Original Dictionary is")
print(myDict)
newDict={key:value for (key,value) in myDict.items()}
print("New dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is
{1: 1, 2: 2, 3: 3, 4: 4}
New dictionary is:
{1: 1, 2: 2, 3: 3, 4: 4}
```

在输出中，我们可以看到新字典包含与原始字典相同的键值对。

我们还可以修改原始字典中的字典条目，然后将它们包含到新字典中。例如，下面的程序在将条目添加到新字典之前修改了它们的值。

```py
 myDict={1:1,2:2,3:3,4:4}
print("Original Dictionary is")
print(myDict)
newDict={key:value**2 for (key,value) in myDict.items()}
print("New dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is
{1: 1, 2: 2, 3: 3, 4: 4}
New dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
```

在输出中，我们可以看到新字典的键与原始字典的键相同，但是新字典中项的值是原始字典中项的值的平方。

我们还可以在创建新字典时使用条件从旧字典中有选择地选择条目。

例如，下面的程序过滤掉键是奇数的项，并且只考虑那些要添加到键是偶数的新字典中的项。在将这些条目添加到新字典之前，它进一步平方这些条目的值。

```py
myDict={1:1,2:2,3:3,4:4}
print("Original Dictionary is")
print(myDict)
newDict={key:value**2 for (key,value) in myDict.items() if key%2==0}
print("New dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is
{1: 1, 2: 2, 3: 3, 4: 4}
New dictionary is:
{2: 4, 4: 16}
```

在输出中，我们可以看到新字典中的键值对只有偶数作为它们的键，它们的平方作为各自的值。

## 结论

在本文中，我们已经了解了什么是词典理解，以及如何使用它们来创建新词典。字典理解可用于在创建字典时替换 for 循环，并在不使用复杂条件时使源代码更加简洁并增加代码的可读性。如果我们在字典理解中使用复杂的条件语句，可能会降低源代码的可读性。代码的执行也可能会变慢，程序将需要更多的内存空间。因此，在处理复杂的条件语句来创建字典时，最好使用 for 循环而不是字典理解。请继续关注更多内容丰富的文章。