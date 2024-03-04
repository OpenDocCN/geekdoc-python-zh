# 用 Python 复制列表

> 原文：<https://www.pythonforbeginners.com/basics/copy-a-list-in-python>

在用 python 编程时，我们有时需要在多个地方存储相同的数据。这可能是因为我们需要保存原始数据。在本文中，我们将讨论在 python 中复制列表的不同方法。

## 用 Python 复制列表

当我们需要在 python 中复制一个整数时，我们只需将一个变量赋给另一个变量，如下所示。

```py
num1 = 10
print("The first number is:", num1)
num2 = num1
print("The copied number is:", num2)
```

输出:

```py
The first number is: 10
The copied number is: 10
```

这里，我们创建了一个值为 10 的变量`num1`。然后，我们将`num1`赋给另一个变量`num2`。赋值后，即使我们改变了原始变量，复制变量中的值仍然不受影响。您可以在下面的示例中观察到这一点。

```py
num1 = 10
print("The first number is:", num1)
num2 = num1
print("The copied number is:", num2)
num1 = 15
print("The first number after modification:", num1)
print("The copied number is:", num2)
```

输出:

```py
The first number is: 10
The copied number is: 10
The first number after modification: 15
The copied number is: 10
```

在上面的例子中，你可以看到在修改了`num1`之后`num2`中的值并没有被修改。

现在，让我们使用赋值操作复制一个列表。

```py
list1 = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", list1)
list2 = list1
print("The copied list is:", list2)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The copied list is: [1, 2, 3, 4, 5, 6, 7]
```

在这种情况下，当我们更改原始列表时，复制的列表也会被修改。

```py
list1 = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", list1)
list2 = list1
print("The copied list is:", list2)
list1.append(23)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list2)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The copied list is: [1, 2, 3, 4, 5, 6, 7]
The original list after modification is: [1, 2, 3, 4, 5, 6, 7, 23]
The copied list after modification is: [1, 2, 3, 4, 5, 6, 7, 23]
```

为什么会这样？

通过赋值复制适用于整数，因为整数是不可变的对象。当一个整数被赋给另一个整数时，两者指的是同一个对象。一旦我们修改了任何整数变量，就会创建一个新的 python 对象，而原始的 python 对象不受影响。您可以在下面的示例中观察到这一点。

```py
num1 = 10
print("The id of first number is:", id(num1))
num2 = num1
print("The id of copied number is:", id(num2))
num1 = 15
print("The id of first number after modification:", id(num1))
print("The id of copied number after modification is:", id(num2)) 
```

输出:

```py
The id of first number is: 9789248
The id of copied number is: 9789248
The id of first number after modification: 9789408
The id of copied number after modification is: 9789248 
```

在这里，您可以看到在使用赋值操作符将`num1`复制到`num2`之后，`num1`和`num2`的 id 是相同的。然而，当我们修改`num1`时，`num1`的 id 会发生变化。

列表是可变的对象。当我们修改一个列表时，原始的列表对象也被修改。因此，在修改列表变量的过程中不会创建 python 对象，这种变化会反映在复制的和原始的列表变量中。您可以在下面的示例中观察到这一点。

```py
list1 = [1, 2, 3, 4, 5, 6, 7]
print("The id of original list is:", id(list1))
list2 = list1
print("The id of  copied list is:", id(list2))
list1.append(23)
print("The id of  original list after modification is:", id(list1))
print("The id of copied list after modification is:", id(list2))
```

输出:

```py
The id of original list is: 139879630222784
The id of  copied list is: 139879630222784
The id of  original list after modification is: 139879630222784
The id of copied list after modification is: 139879630222784
```

在这里，您可以看到，即使在`list1`中修改后，两个列表的 id 仍然相同。因此，即使在修改之后，也可以确认原始列表和复制列表引用相同的对象。

由于赋值操作不适用于在 python 中复制列表，我们将讨论在 python 中复制列表的不同方法。在此之前，我们先来讨论一下 python 中的`id()` 函数。

### Python 中的 id()函数

每个 python 对象都有一个唯一的标识符。我们可以使用`id()`函数来获取与任何 python 对象相关联的标识符，如上面的例子所示。

如果两个变量作为输入传递给`id()`函数，给出相同的输出，那么这两个变量指的是同一个对象。

如果`id()`函数的输出对于不同的变量是不同的，那么它们指的是不同的对象。

在接下来的部分中，我们将使用`id()`函数来检查一个列表是否被复制。如果列表复制成功，两个列表的标识符将会不同。在这种情况下，当我们修改一个列表时，它不会影响另一个列表。

如果引用列表的两个变量的标识符相同，则这两个变量引用同一个列表。在这种情况下，在与变量关联的列表中所做的任何更改都将反映在与另一个变量关联的列表中。

有了这个背景，现在让我们来讨论在 python 中复制列表的不同方法。

### 使用 Python 中的 List()构造函数复制列表

`list()`构造函数用于从任何可迭代对象创建一个新的列表，比如列表、元组或集合。它将一个 iterable 对象作为其输入参数，并返回一个包含输入 iterable 对象元素的新列表。

为了使用 python 中的`list()`构造函数复制一个列表，我们将把原始列表作为输入参数传递给 list()构造函数。

执行后， `list()`构造函数将返回一个新的列表，其中包含原始列表中的元素。新列表和原始列表将具有不同的标识符，这些标识符可以使用`id()`方法获得。因此，对其中一个列表的任何更改都不会影响另一个列表。您可以在下面的示例中观察到这一点。

```py
list1 = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = list(list1)
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1.append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy) 
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The id of original list is: 140137673798976
The copied list is: [1, 2, 3, 4, 5, 6, 7]
The id of  copied list is: 140137673851328
The original list after modification is: [1, 2, 3, 4, 5, 6, 7, 10]
The copied list after modification is: [1, 2, 3, 4, 5, 6, 7]
```

在上面的例子中，你可以看到原始列表和复制列表的 id 是不同的。因此，对原始列表进行更改不会影响复制的列表。

### 使用 Python 中的 append()方法复制列表

方法用于向列表中添加一个新元素。当在列表上调用时，它将一个元素作为其输入参数，并将其添加到列表的最后一个位置。

为了使用 python 中的 `append()` 方法复制一个列表，我们将首先创建一个名为`list_copy`的空列表。之后，我们将使用 for 循环遍历原始列表。在迭代过程中，我们将使用`append()`方法将原始列表的每个元素添加到`list_copy`中。

执行 for 循环后，我们将在`list_copy`变量中获得复制的列表。您可以在下面的示例中观察到这一点。

```py
list1 = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = []
for element in list1:
    list_copy.append(element)
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1.append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy) 
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The id of original list is: 140171559405056
The copied list is: [1, 2, 3, 4, 5, 6, 7]
The id of  copied list is: 140171560708032
The original list after modification is: [1, 2, 3, 4, 5, 6, 7, 10]
The copied list after modification is: [1, 2, 3, 4, 5, 6, 7]
```

您可以观察到新列表和原始列表具有不同的标识符，这些标识符是我们使用`id()`方法获得的。因此，对其中一个列表的任何更改都不会影响到另一个列表。

### 使用 Python 中的 extend()方法复制列表

方法用来一次添加多个元素到一个列表中。当在列表上调用时， `extend()` 方法将 iterable 对象作为其输入参数。执行后，它将输入 iterable 对象的所有元素追加到列表中。

为了使用`extend()`方法复制一个列表，我们将首先创建一个名为`list_copy`的空列表。之后，我们将调用`list_copy`上的`extend()` 方法，将原始列表作为其输入参数。执行后，我们将在`list_copy`变量中得到复制的列表。

新列表和原始列表将具有不同的标识符，这些标识符可以使用`id()` 方法获得。因此，对其中一个列表的任何更改都不会影响另一个列表。您可以在下面的示例中观察到这一点。

```py
list1 = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = []
list_copy.extend(list1)
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1.append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy) 
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The id of original list is: 139960369243648
The copied list is: [1, 2, 3, 4, 5, 6, 7]
The id of  copied list is: 139960370546624
The original list after modification is: [1, 2, 3, 4, 5, 6, 7, 10]
The copied list after modification is: [1, 2, 3, 4, 5, 6, 7]
```

### 在 Python 中使用切片复制列表

[python 中的切片](https://www.pythonforbeginners.com/dictionary/python-slicing)用于创建列表一部分的副本。切片的一般语法如下。

```py
new_list=original_list[start_index:end_index]
```

这里，

*   `new_list`是执行语句后创建的列表。
*   `original_list`是给定的输入列表。
*   `start_index`是必须包含在`new_list`中的最左边元素的索引。如果我们让`start_index`为空，默认值为 0，列表从第一个元素开始复制。
*   `end_index`是必须包含在`new_list`中的最右边元素的索引。如果我们让`end_index`为空，默认值就是列表的长度。因此，列表被复制到最后一个元素。

您可以使用切片来复制列表，如下例所示。

```py
list1 = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = list1[:]
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1.append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The id of original list is: 139834922264064
The copied list is: [1, 2, 3, 4, 5, 6, 7]
The id of  copied list is: 139834923567040
The original list after modification is: [1, 2, 3, 4, 5, 6, 7, 10]
The copied list after modification is: [1, 2, 3, 4, 5, 6, 7]
```

您可以观察到新列表和原始列表具有不同的标识符，这些标识符是我们使用`id()` 方法获得的。因此，对其中一个列表的任何更改都不会影响另一个列表。

### 使用 Python 中的列表理解复制列表

L [ist comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 用于通过对元素施加一些条件，使用现有列表的元素来创建新列表。列表理解的一般语法如下。

```py
new_list=[element for element in existing_list if condition]
```

这里，

*   `new_list`是执行语句后创建的列表。
*   `existing_list`是输入列表。
*   `element`代表现有列表的元素。这个变量用于迭代`existing_list`。
*   `condition`是对元素施加的条件。

这里，我们需要使用列表理解来复制一个给定的列表。因此，我们将不使用任何条件。下面是使用 python 中的 list comprehension 在 python 中复制列表的代码。

```py
list1 = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = [element for element in list1]
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1.append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy) 
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The id of original list is: 139720431945088
The copied list is: [1, 2, 3, 4, 5, 6, 7]
The id of  copied list is: 139720433248128
```

您可以观察到新列表和原始列表具有不同的标识符，这些标识符是我们使用`id()` 方法获得的。因此，对其中一个列表所做的任何更改都不会影响另一个列表。

### 使用 Python 中的 Copy()方法复制列表

Python 还为我们提供了在 python 中复制列表的`copy()`方法。在列表上调用`copy()`方法时，会返回原始列表的副本。新列表和原始列表将具有不同的标识符，这些标识符可以使用`id()` 方法获得。因此，对其中一个列表的任何更改都不会影响另一个列表。您可以在下面的示例中观察到这一点。

```py
list1 = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = list1.copy()
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1.append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy) 
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The id of original list is: 140450078000576
The copied list is: [1, 2, 3, 4, 5, 6, 7]
The id of  copied list is: 140450079303616
The original list after modification is: [1, 2, 3, 4, 5, 6, 7, 10]
The copied list after modification is: [1, 2, 3, 4, 5, 6, 7]
```

## 在 Python 中复制列表列表

在 python 中复制列表的列表与复制列表不同。例如，让我们使用`copy()`方法复制 python 中的列表列表。

```py
list1 = [[1, 2, 3], [4, 5, 6,], [7,8,9]]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = list1.copy()
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
```

输出:

```py
The original list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of original list is: 139772961010560
The copied list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of  copied list is: 139772961063040
```

这里，您可以看到原始列表和复制列表的 id 是不同的。这意味着两个列表是不同的 python 对象。尽管如此，当我们对原始列表进行任何修改时，它都会反映在修改后的列表中。您可以在下面的示例中观察到这一点。

```py
list1 = [[1, 2, 3], [4, 5, 6, ], [7, 8, 9]]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = list1.copy()
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1[2].append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy)
```

输出:

```py
The original list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of original list is: 139948423344896
The copied list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of  copied list is: 139948423397504
The original list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
The copied list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
```

这种情况是没有根据的。这是由于内存中列表的存储模式造成的。列表列表包含对内部列表的引用。

当我们使用`copy()`方法复制一个列表列表时，列表列表以及内部列表的引用也被复制。因此，不会复制内部列表，只会复制对内部列表的引用。因此，当我们对原始列表中的任何内部列表进行更改时，它也会反映在复制的列表中。

为了避免这种情况，我们可以迭代地复制列表列表的元素。

### 使用 Python 中的 append()方法复制列表列表

要使用 python 中的`append()` 方法复制列表列表，我们将使用以下步骤。

*   首先，我们将创建一个名为`list_copy`的空列表。
*   之后，我们将使用 for 循环遍历列表列表。对于列表列表中的每个内部列表，我们将执行以下任务。
    *   首先，我们将创建一个名为`temp`的空列表。之后，我们将使用另一个 for 循环迭代内部列表的元素。
    *   迭代时，我们将使用`append()`方法将当前内部列表的元素追加到`temp`中。当在`temp`上被调用时，`append()`方法将从内部列表中获取一个元素作为输入参数，并将它附加到`temp`。
*   遍历完当前内部列表的每个元素后，我们将把`temp`追加到`list_copy`。之后，我们将移动到下一个内部列表，并重复前面的步骤。

一旦我们遍历了输入列表的所有内部列表，我们将在`copy_list`变量中获得列表的复制列表。您可以在下面的示例中观察到这一点。

```py
list1 = [[1, 2, 3], [4, 5, 6, ], [7, 8, 9]]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = []
for inner_list in list1:
    temp = []
    for element in inner_list:
        temp.append(element)
    list_copy.append(temp)
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1[2].append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy)
```

输出:

```py
The original list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of original list is: 139893771608960
The copied list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of  copied list is: 139893771661952
The original list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
The copied list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

迭代复制列表元素后，您可以看到原始列表中的修改不会影响复制的列表。

### 使用 Python 中的 extend()方法复制列表列表

除了使用`append()`方法，我们还可以使用`extend()`方法来复制 python 中的列表列表。为此，我们将使用以下步骤。

*   首先，我们将创建一个名为`list_copy`的空列表。
*   之后，我们将使用 for 循环遍历列表列表。对于列表列表中的每个内部列表，我们将执行以下任务。
*   首先，我们将创建一个名为`temp`的空列表。之后，我们将调用`temp`上的`extend()` 方法，并将内部列表作为其输入参数。
*   然后，我们将把 temp 追加到`list_copy`中。之后，我们将进入下一个内部列表。

一旦我们遍历了输入列表的所有内部列表，我们将在`copy_list`变量中获得列表的复制列表。您可以在下面的示例中观察到这一点。

```py
list1 = [[1, 2, 3], [4, 5, 6, ], [7, 8, 9]]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = []
for inner_list in list1:
    temp = []
    temp.extend(inner_list)
    list_copy.append(temp)
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1[2].append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy)
```

输出:

```py
The original list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of original list is: 140175921128448
The copied list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of  copied list is: 140175921181312
The original list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
The copied list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### 使用 Python 中的 Copy()方法复制列表列表

我们还可以使用`copy()`方法在 python 中复制一个列表列表。为此，我们将使用以下步骤。

*   首先，我们将创建一个名为`list_copy`的空列表。
*   之后，我们将使用 for 循环遍历列表列表。对于列表列表中的每个内部列表，我们将执行以下任务。
*   我们将调用内部列表上的`copy()`方法。我们将把 `copy()`方法的输出分配给一个变量`temp`。
*   然后，我们将把 temp 追加到`list_copy`中。之后，我们将进入下一个内部列表。

一旦我们遍历了输入列表的所有内部列表，我们将在`copy_list`变量中获得列表的复制列表。您可以在下面的示例中观察到这一点。

```py
list1 = [[1, 2, 3], [4, 5, 6, ], [7, 8, 9]]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = []
for inner_list in list1:
    temp = inner_list.copy()
    list_copy.append(temp)
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1[2].append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy)
```

输出:

```py
The original list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of original list is: 140468123341760
The copied list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of  copied list is: 140468123394560
The original list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
The copied list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### 使用 Python 中的复制模块复制列表列表

`copy`模块为我们提供了`deepcopy()`方法，通过它我们可以复制嵌套的对象。`deepcopy()`方法将 python 对象作为输入参数，递归地复制输入对象的所有元素。

我们可以使用下面的例子所示的`deepcopy()`方法在 python 中复制一个列表列表。

```py
import copy
list1 = [[1, 2, 3], [4, 5, 6, ], [7, 8, 9]]
print("The original list is:", list1)
print("The id of original list is:", id(list1))
list_copy = copy.deepcopy(list1)
print("The copied list is:", list_copy)
print("The id of  copied list is:", id(list_copy))
list1[2].append(10)
print("The original list after modification is:", list1)
print("The copied list after modification is:", list_copy)
```

输出:

```py
The original list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of original list is: 139677987171264
The copied list is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
The id of  copied list is: 139677987171776
The original list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
The copied list after modification is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

您可以观察到新列表和原始列表具有不同的标识符，这些标识符是我们使用 `id()`方法获得的。因此，对一个列表的任何更改都不会影响另一个列表。

## 结论

在本文中，我们讨论了用 python 复制列表的不同方法。我们还研究了用 python 复制列表列表的不同方法。

要了解更多关于 python 编程的知识，你可以阅读这篇关于如何在 python 中[检查排序列表的文章。您可能也会喜欢这篇关于如何用 python](https://www.pythonforbeginners.com/lists/check-for-sorted-list-in-python) 将字典[保存到文件中的文章。](https://www.pythonforbeginners.com/dictionary/how-to-save-dictionary-to-file-in-python)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！