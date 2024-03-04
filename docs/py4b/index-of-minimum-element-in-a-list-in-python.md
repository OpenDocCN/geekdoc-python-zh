# Python 中列表中最小元素的索引

> 原文：<https://www.pythonforbeginners.com/basics/index-of-minimum-element-in-a-list-in-python>

当我们需要随机访问时，我们在 python 程序中使用列表来存储不同类型的对象。在本文中，我们将讨论在 python 中查找列表中最小元素的索引的不同方法。

## 使用 for 循环的列表中最小元素的索引

我们在 python 中使用 for 循环来迭代类似 list 的容器对象的元素。为了使用 python 中的 for 循环找到列表中最小元素的索引，我们可以使用 `len()` 函数和`range()`函数。

### Python 中的 len()函数

python 中的`len()`函数用于查找列表或元组等集合对象的长度。它将类似 list 的容器对象作为其输入参数，并在执行后返回集合对象的长度。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 23, 32, 12, 44, 34, 55, 46, 21, 12]
print("The list is:", myList)
list_len = len(myList)
print("Length of the list is:", list_len)
```

输出:

```py
The list is: [1, 2, 23, 32, 12, 44, 34, 55, 46, 21, 12]
Length of the list is: 11
```

这里，我们将一个包含 11 个元素的列表传递给了`len()`函数。执行后，它返回相同的值。

### Python 中的 range()函数

`range()`函数用于在 python 中生成一系列数字。在最简单的情况下，`range()`函数将一个正数 N 作为输入参数，并返回一个包含从 0 到 N-1 的数字序列。您可以在下面的示例中观察到这一点。

```py
sequence = range(11)
print("The sequence is:", sequence)
```

输出:

```py
The sequence is: range(0, 11)
```

为了使用 for 循环、 `len()`函数和`range()`函数在 python 中找到列表中最小元素的索引，我们将使用以下步骤。

*   首先，我们将使用`len()`函数计算输入列表的长度。我们将把该值存储在变量`list_len`中。
*   计算完列表的长度后，我们将使用`range()`函数创建一个从 0 到 `list_len-1`的数字序列。
*   现在，我们将定义一个变量`min_index`并给它赋值 0，假设列表的最小元素出现在索引 0 处。
*   创建序列后，我们将使用 for 循环遍历序列中的数字。迭代时，我们将访问输入列表中每个索引处的元素。
*   在每个索引处，我们将检查当前索引处的元素是否小于列表中`min_index`索引处的元素。如果当前元素小于`min_index`处的元素，我们将把`min_index`更新为当前索引。

在执行 for 循环后，您将在`min_index`变量中获得列表中最小元素的索引。您可以在下面的示例中观察到这一点。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
min_index = 0
list_len = len(myList)
for index in range(list_len):
    if myList[index] < myList[min_index]:
        min_index = index
print("Index of the minimum element is:", min_index)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Index of the minimum element is: 3
```

在上面的方法中，如果列表中多次出现最小元素，您将获得该元素最左边的索引。要获得最小元素所在的最右边的索引，可以在比较列表元素时使用小于或等于运算符，而不是小于运算符。您可以在下面的示例中观察到这一点。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
min_index = 0
list_len = len(myList)
for index in range(list_len):
    if myList[index] <= myList[min_index]:
        min_index = index
print("Index of the minimum element is:", min_index)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Index of the minimum element is: 11
```

## 使用 min()函数和 Index()方法对列表中的最小元素进行索引

在 python 中，我们可以使用`min()`函数和 `index()`方法来查找列表中最小元素的索引，而不是使用 for 循环来迭代整个列表。

### Python 中的 min()函数

`min()`函数用于查找容器对象中的最小元素，如列表、元组或集合。`min()` 函数将一个类似 list 的集合对象作为它的输入参数。执行后，它返回列表中的最小元素。您可以在下面的示例中观察到这一点。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
min_val = min(myList)
print("The minimum value is:", min_val)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
The minimum value is: 1
```

### Python 中的 index()方法

方法的作用是找到一个元素在列表中的位置。当在列表上调用时，`index()`方法将一个元素作为它的输入参数。执行后，它返回该元素第一次出现的索引。例如，我们可以在列表中找到元素 23 的索引，如下所示。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
index = myList.index(23)
print("Index of 23 is:", index)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Index of 23 is: 2
```

如上所述，如果单个元素出现多次，`index()`方法只返回该元素最左边的索引。如果我们将元素 1 作为输入参数传递给`index()` 方法，那么输出将是 3，尽管元素 1 也出现在索引 11 处。

如果输入参数中给出的元素不在列表中，那么`index()`方法会抛出一个`ValueError`异常，表示该元素不在列表中。您可以在下面的示例中观察到这一点。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
index = myList.index(105)
print("Index of 105 is:", index)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 3, in <module>
    index = myList.index(105)
ValueError: 105 is not in list
```

这里，我们将 105 作为输入参数传递给了`index()`方法。但是，105 不在列表中。因此，程序运行到`ValueError`异常，显示 105 不在列表中。

为了使用`min()`函数和`index()` 函数找到列表中最小元素的索引，我们将使用以下步骤。

首先，我们将使用`min()`函数找到列表中的最小元素。我们将把该值存储在变量`min_val`中。

找到列表中的最小值后，我们将调用列表中的`index()`方法，并将`min_val`作为其输入参数。执行后，`index()`方法将返回列表中最小元素的索引。您可以在下面的示例中观察到这一点。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
min_val = min(myList)
index = myList.index(min_val)
print("Index of minimum value is:", index)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Index of minimum value is: 3
```

在元素多次出现的情况下，这种方法只能找到列表中最小元素的最左边的索引。如果想获得最小元素的最右边的索引，可以使用前面讨论的 for 循环和`len()` 函数。

## 使用 Numpy 模块的列表中最小元素的索引

模块被设计成以一种有效的方式处理数字和数组。我们还可以使用`numpy`模块在 python 中找到列表中最小元素的索引。

`numpy`模块为我们提供了`argmin()`方法来查找 NumPy 数组中最小元素的索引。当在一个`numpy`数组上调用`argmin()`方法时，它返回数组中最小元素的索引。

为了使用`argmin()`方法获得列表中最小元素的索引，我们将首先把列表转换成一个 numpy 数组。为此，我们将使用`array()`函数。

`array()`函数将一个列表作为它的输入参数，并返回一个 numpy 数组。您可以在下面的示例中观察到这一点。

```py
import numpy

myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
array = numpy.array(myList)
print("The array is:", array)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
The array is: [11  2 23  1 32 12 44 34 55 46 21  1 12]
```

将列表转换成 numpy 数组后，我们将调用数组上的 `argmin()`方法。执行后，`argmin()`方法将返回最小元素的索引。

```py
import numpy

myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
array = numpy.array(myList)
min_index = array.argmin()
print("Index of minimum element is:", min_index)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Index of minimum element is: 3
```

当最小元素出现多次时，`argmin()`函数将返回最小元素最左边的索引。在上面的例子中，您可以看到`argmin()`方法返回 3，即使最小元素 1 也出现在索引 11 处。

## 多次出现时列表中最小元素的索引

有时，我们可能需要找到列表中最小元素的所有出现。在接下来的部分中，我们将讨论在多次出现的情况下找到列表中最小元素的索引的不同方法。

### 使用 min()函数和 For 循环的列表中最小元素的索引

当列表中多次出现最小元素时，我们可以使用`min()`函数和 for 循环来获取最小元素的索引。为此，我们将使用以下步骤。

*   首先，我们将创建一个名为`min_indices`的空列表来存储最小元素的索引。
*   然后，我们将使用`len()`函数找到输入列表的长度。我们将把长度存储在变量`list_len`中。
*   在获得列表的长度后，我们将使用`range()`函数创建一个从 0 到 list_len-1 的数字序列。
*   接下来，我们将使用`min()` 函数找到列表中的最小元素。
*   找到最小元素后，我们将使用 for 循环遍历数字序列。迭代时，我们将检查列表中当前索引处的元素是否等于最小元素。
    *   如果我们找到一个等于最小元素的元素，我们将使用`append()`方法将其索引添加到`min_indices`列表中。当在`min_indices`列表上调用`append()`方法时，该方法将把`index`作为其输入参数。执行后，它会将`index`作为一个元素添加到`min_indices`列表中。
    *   如果当前索引处的数字不等于最小元素，我们将移动到下一个元素。

在执行 for 循环之后，我们将获得列表中最小元素的索引。您可以在下面的示例中观察到这一点。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
min_val = min(myList)
list_indices = []
list_len = len(myList)
sequence = range(list_len)
for index in sequence:
    if myList[index] == min_val:
        list_indices.append(index)
print("Indices of minimum element are:", list_indices)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Indices of minimum element are: [3, 11]
```

### 使用 min()函数和列表理解的列表中最小元素的索引

[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)用于通过对元素施加一些条件，使用现有列表的元素创建新列表。列表理解的一般语法如下。

```py
new_list=[f(element) for element in existing_list if condition]
```

这里，

*   `new_list`是执行语句后创建的列表。
*   `existing_list`是输入列表。
*   `element`表示现有列表中的一个元素。这个变量用于迭代`existing_list`。
*   `condition`是对元素或`f(element)`施加的条件。

为了使用 python 中的列表理解找到列表中最小元素的索引，我们将使用以下步骤。

*   首先，我们将使用`len()` 函数找到输入列表的长度。我们将把长度存储在变量`list_len`中。
*   在获得列表的长度后，我们将使用`range()` 函数创建一个从 0 到 list_len-1 的数字序列。
*   接下来，我们将使用`min()`函数找到列表中的最小元素。
*   在找到最小元素后，我们将使用列表理解来获得最小元素的索引列表。
*   在列表理解中，
    *   我们将使用数字序列来代替`existing_list`。
    *   `element`将代表数字序列的元素。
    *   `f(element)` 将等于`element`。
    *   代替`condition`，我们将使用输入列表中 index 元素的值应该等于 minimum 元素的条件。

执行该语句后，我们将得到包含列表中最小元素索引的`new_list`,如下所示。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
min_val = min(myList)
list_len = len(myList)
sequence = range(list_len)
new_list = [index for index in sequence if myList[index] == min_val]
print("Indices of minimum element are:", new_list)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Indices of minimum element are: [3, 11]
```

### 使用 min()函数和 enumerate()函数的列表中最小元素的索引

不使用 `len()` 函数和`range()`函数，我们可以使用`enumerate()`函数和`min()`函数来查找列表中最小元素的索引。

#### Python 中的 enumerate()函数

`enumerate()`函数用于枚举容器对象的元素，如列表或元组。

`enumerate()`函数将一个类似 list 的容器对象作为它的输入参数。执行后，它返回一个包含元组的枚举列表。列表中的每个元组包含两个元素。元组的第一个元素是分配给元素的索引。第二个元素是原始列表中的相应元素，作为输入给`enumerate()`函数。您可以在下面的示例中观察到这一点。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
enumerated_list = list(enumerate(myList))
print("Enumerated list is:", enumerated_list)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Enumerated list is: [(0, 11), (1, 2), (2, 23), (3, 1), (4, 32), (5, 12), (6, 44), (7, 34), (8, 55), (9, 46), (10, 21), (11, 1), (12, 12)]
```

为了使用`enumerate()`函数找到列表中最小元素的索引，我们将使用以下步骤。

*   首先，我们将创建一个名为`min_indices`的空列表来存储最小元素的索引。
*   之后，我们将使用`min()`函数找到输入列表中的最小元素。
*   然后，我们将使用`enumerate()`函数从输入列表中创建枚举列表。
*   获得枚举列表后，我们将使用 for 循环遍历枚举列表中的元组。
*   在迭代元组时，我们将检查当前元组中的元素是否等于最小元素。
*   如果当前元组中的元素等于最小元素，我们将使用`append()`方法将当前索引附加到`min_indices`。否则，我们将移动到枚举列表中的下一个元组。

在执行 for 循环之后，我们将获得`min_indices`列表中最小元素的索引。您可以在下面的示例中观察到这一点。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
enumerated_list = list(enumerate(myList))
min_indices = []
min_element = min(myList)
for element_tuple in enumerated_list:
    index = element_tuple[0]
    element = element_tuple[1]
    if element == min_element:
        min_indices.append(index)
print("Indices of minimum element are:", min_indices)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Indices of minimum element are: [3, 11]
```

我们可以使用列表理解来查找列表中最小元素的索引，而不是使用 for 循环来迭代枚举列表中的元组，如下所示。

```py
myList = [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
print("The list is:", myList)
enumerated_list = list(enumerate(myList))
min_element = min(myList)
min_indices = [index for (index, element) in enumerated_list if element == min_element]
print("Indices of minimum element are:", min_indices)
```

输出:

```py
The list is: [11, 2, 23, 1, 32, 12, 44, 34, 55, 46, 21, 1, 12]
Indices of minimum element are: [3, 11]
```

## 结论

在本文中，我们讨论了在 python 中查找列表中最小元素的索引的不同方法。我们还讨论了在最小元素多次出现的情况下，查找列表中最小元素的所有索引。

要在 python 中找到列表中最小元素的索引，我建议您使用带有 `index()`方法的`min()`函数。在多次出现的情况下，这种方法给出最小元素的最左边的索引。

如果需要在 python 中找到列表中最小元素的最右边的索引，可以使用 for 循环和小于或等于运算符。

在多次出现的情况下，如果需要获得 python 中一个列表中最小元素的所有索引，可以使用带有 for 循环的方法或者带有`enumerate()`函数的方法。两者的执行效率几乎相同。

我希望你喜欢阅读这篇文章。要了解更多关于 python 编程的知识，您可以阅读这篇关于如何在 Python 中[删除列表中所有出现的字符的文章。您可能也喜欢这篇关于如何](https://www.pythonforbeginners.com/basics/remove-all-occurrences-of-a-character-in-a-list-or-string-in-python)[检查 python 字符串是否包含数字](https://www.pythonforbeginners.com/strings/check-if-a-python-string-contains-a-number)的文章。