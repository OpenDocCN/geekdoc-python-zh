# 在 Python 中查找列表中最大值的索引

> 原文：<https://www.pythonforbeginners.com/basics/find-the-index-of-max-value-in-a-list-in-python>

python 中的列表是最常用的数据结构之一。在本文中，我们将讨论在 python 中查找列表中最大值的索引的不同方法。

## 在 Python 中使用 for 循环查找列表中最大值的索引

要使用 for 循环查找列表中最大值的索引，我们将使用以下过程。

*   首先，假设第一个元素是列表的最大元素，我们将把变量`max_index`初始化为 0。
*   之后，我们将使用`len()`函数找到列表的长度。`len()`函数将一个列表作为其输入参数，并返回列表的长度。
*   一旦我们得到了列表的长度，我们将使用`range()`函数和一个 for 循环来遍历列表。迭代时，在列表的每个索引处，我们将检查当前索引处的元素是否大于索引处的元素`max_index`。
*   如果我们找到一个索引，其中的元素大于索引`max_index`处的元素，我们将把当前索引赋给变量`max_index`。

在整个列表迭代之后，我们将获得变量`max_index`中最大元素的索引。

您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 21, 12]
print("The list is:", myList)
max_index = 0
list_len = len(myList)
for index in range(list_len):
    if myList[index] > myList[max_index]:
        max_index = index
print("Index of the maximum element is:", max_index)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 21, 12]
Index of the maximum element is: 7
```

在上面的例子中，如果最大值元素出现多次，我们得到最大值元素最左边的索引。

为了获得最大元素的最右边的索引，在比较元素时，可以使用大于或等于`(>=)`操作符，而不是大于`(>)`操作符，如下所示。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
max_index = 0
list_len = len(myList)
for index in range(list_len):
    if myList[index] >= myList[max_index]:
        max_index = index
print("Index of the maximum element is:", max_index)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Index of the maximum element is: 9
```

## 使用 Max()函数和 Index()方法查找列表中最大值的索引

要编写更 python 化的代码，可以使用`max()`函数和`index()`方法在 python 中查找列表中最大值的索引。

### max()函数

`max()` 函数接受一个容器对象，如列表、元组或集合，作为它的输入参数。执行后，它返回容器对象的最大元素。

例如，如果我们给一个列表作为`max()`函数的输入参数，它将返回列表的最大元素。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
max_val = max(myList)
print("The maximum value is:", max_val)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
The maximum value is: 55
```

### index()方法

在列表上调用`index()` 方法时，该方法将一个元素作为其输入参数，并从列表的开始处返回该元素第一次出现的索引。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
index = myList.index(23)
print("Index of 23 is:", index)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Index of 23 is: 2
```

当输入元素不在列表中时，`index()`方法引发一个`ValueError`异常，表明给定元素不在列表中，如下所示。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
index = myList.index(112)
print("Index of 112 is:", index)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    index = myList.index(112)
ValueError: 112 is not in list 
```

为了在 python 中找到列表中最大值的索引，我们将首先使用`max()`函数找到列表中的最大元素。之后，我们将调用列表上的`index()`方法，将最大元素作为其输入参数。在执行了`index()`方法后，我们将得到列表中最大元素的索引。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
max_val = max(myList)
index = myList.index(max_val)
print("Index of maximum value is:", index)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Index of maximum value is: 7
```

这里，最大值 55 出现在两个索引处。然而，`index()`方法只返回元素最左边出现的索引。

## 使用 Max()函数和 For 循环查找多次出现的最大值的索引

在使用`max()`函数和`index()`方法的方法中，我们只能在 python 中找到列表中最大值的第一个索引。为了在最大值多次出现的情况下找到列表中最大值的索引，我们可以使用下面的方法。

*   首先，我们将使用`max()`函数找到列表中的最大值。
*   然后，我们将创建一个名为`list_indices`的列表来存储列表中最大值的索引。
*   之后，我们将使用`len()` 函数找到输入列表的长度。 `len()`函数将列表作为其输入参数，并返回列表的长度。我们将把长度存储在变量`list_len`中。
*   获得列表长度后，我们将使用`range()`函数创建一个从 0 到`list_len`的数字序列。`range()`函数将`list_len`作为其输入参数，并返回一个包含从 0 到`list_len-1`的数字序列。
*   现在，我们将使用循环的[来迭代数字序列。迭代时，我们将检查列表中与序列中当前编号相等的索引处的元素是否等于最大值。](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)
*   如果我们在序列中找到一个索引，在这个索引处有列表中的最大值。我们将使用`append()`方法将索引存储在`list_indices`中。当在列表上调用`append()`方法时，它将把索引值作为输入参数，并把它添加到列表中。

在执行 for 循环后，我们将获得`list_indices`中最大值的所有索引。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
max_val = max(myList)
list_indices = []
list_len = len(myList)
sequence = range(list_len)
for index in sequence:
    if myList[index] == max_val:
        list_indices.append(index)
print("Indices of maximum element are:", list_indices)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Indices of maximum element are: [7, 9]
```

## 在多次出现的情况下，使用 Max()函数和列表理解来查找最大值的索引

不使用 for 循环，您可以使用[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)以及`range()`函数和`max()`函数从给定的输入列表中获取最大值的索引列表，如下所示。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
max_val = max(myList)
list_len = len(myList)
sequence = range(list_len)
list_indices = [index for index in sequence if myList[index] == max_val]
print("Indices of maximum element are:", list_indices)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Indices of maximum element are: [7, 9]
```

## 使用 Max()和 enumerate()函数查找多次出现的最大值的索引

`enumerate()`函数用于给容器对象的元素分配一个计数器。它接受一个类似 list 的容器对象，并返回一个元组列表。在元组列表中，每个元组包含一个数字，该数字将元素的索引表示为其第一个元素，并将列表中的相应值表示为其第二个输入参数。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
enumerated_list = list(enumerate(myList))
print("Enumerated list is:", enumerated_list)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Enumerated list is: [(0, 1), (1, 2), (2, 23), (3, 32), (4, 12), (5, 44), (6, 34), (7, 55), (8, 46), (9, 55), (10, 21), (11, 12)]
```

为了使用`enumerate()`函数找到 max 元素的索引，我们将使用以下步骤。

*   首先，我们将使用`max()`函数找到输入列表中的最大元素。
*   然后，我们将创建一个名为`max_indices`的空列表来存储列表中 max 元素的索引。
*   创建列表后，我们将使用`enumerate()`函数获得元组的枚举列表。
*   一旦我们得到了枚举列表，我们将使用 for 循环遍历元组。
*   迭代时，我们将检查元组中的当前值是否等于最大值。如果是，我们将把与元组中的元素相关联的索引存储在`max_indices`中。为此，我们将调用`max_indices`上的`append()`方法，将当前索引作为其输入参数。
*   如果元组中的当前值不等于最大值，我们将移动到下一个元组。

在执行 for 循环后，我们将获得`max_indices`列表中 max 元素的所有索引。您可以在下面的代码中观察到这一点。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
enumerated_list = list(enumerate(myList))
max_indices = []
max_element = max(myList)
for element_tuple in enumerated_list:
    index = element_tuple[0]
    element = element_tuple[1]
    if element == max_element:
        max_indices.append(index)
print("Indices of maximum element are:", max_indices) 
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Indices of maximum element are: [7, 9]
```

除了在上面的例子中使用 for 循环，您可以使用 list comprehension 和`enumerate()`函数来查找列表中最大元素的索引，如下所示。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
enumerated_list = list(enumerate(myList))
max_element = max(myList)
max_indices = [index for (index, element) in enumerated_list if element == max_element]
print("Indices of maximum element are:", max_indices)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Indices of maximum element are: [7, 9]
```

## 使用 Numpy 模块在 Python 中查找列表中最大值的索引

我们还可以使用 numpy 模块在 python 中查找列表中最大值的索引。numpy 模块为我们提供了`argmax()`方法来查找列表中最大值的索引。当在 numpy 数组上调用时，`argmax()`方法返回最大元素的索引。

为了在 python 中找到列表中的最大值，我们将首先使用`array()`构造函数将输入列表转换成一个 numpy 数组。`array()`构造函数将一个类似 list 的容器对象作为它的输入参数。执行后，它返回一个 numpy 数组，包含与输入容器对象中相同数量的元素。

从输入列表创建 numpy 数组后，我们将使用`argmax()` 函数来查找列表中最大值的索引，如下所示。

```py
import numpy

myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
print("The list is:", myList)
array = numpy.array(myList)
max_index = array.argmax()
print("Index of maximum element is:", max_index)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 55, 21, 12]
Index of maximum element is: 7
```

如果最大值出现多次，`argmax()`方法将从列表左侧返回第一次出现的最大值的索引。你可以在上面的例子中观察到这一点。

## 结论

在本文中，我们讨论了在 python 中查找列表中最大值的索引的不同方法。如果只需要找到最大值第一次出现的索引，可以使用带有`max()` 函数和`index()`方法的方法。如果您需要获得最大值出现的所有指标，您应该使用使用`max()`函数和`enumerate()`函数的方法。

我希望你喜欢阅读这篇文章。要了解更多关于 python 编程的知识，您可以阅读这篇关于如何在 Python 中[删除列表中所有出现的字符的文章。您可能也喜欢这篇关于如何](https://www.pythonforbeginners.com/basics/remove-all-occurrences-of-a-character-in-a-list-or-string-in-python)[检查 python 字符串是否包含数字](https://www.pythonforbeginners.com/strings/check-if-a-python-string-contains-a-number)的文章。

请继续关注更多内容丰富的文章。

快乐学习！