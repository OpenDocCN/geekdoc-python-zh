# Python 中的列表列表

> 原文：<https://www.pythonforbeginners.com/basics/list-of-lists-in-python>

当我们需要顺序访问数据时，python 中使用列表来存储数据。在本文中，我们将讨论如何用 python 创建一个列表列表。我们还将实现一些程序来执行各种操作，比如在 python 中对列表进行排序、遍历和反转。

## Python 中的列表列表是什么？

python 中的[列表列表是包含列表作为其元素的列表。下面是一个列表的例子。](https://www.pythonforbeginners.com/basics/intersection-of-lists-in-python)

```py
myList=[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

这里，`myList`包含五个列表作为其元素。因此，它是一个列表列表。

## 用 Python 创建一个列表列表

要在 python 中创建一个列表列表，可以使用方括号来存储所有内部列表。例如，如果您有 5 个列表，并且您想从给定的列表中创建一个列表，您可以将它们放在方括号中，如下面的 python 代码所示。

```py
list1 = [1, 2, 3, 4, 5]
print("The first list is:", list1)
list2 = [12, 13, 23]
print("The second list is:", list2)
list3 = [10, 20, 30]
print("The third list is:", list3)
list4 = [11, 22, 33]
print("The fourth list is:", list4)
list5 = [12, 24, 36]
print("The fifth list is:", list5)
myList = [list1, list2, list3, list4, list5]
print("The list of lists is:")
print(myList)
```

输出:

```py
The first list is: [1, 2, 3, 4, 5]
The second list is: [12, 13, 23]
The third list is: [10, 20, 30]
The fourth list is: [11, 22, 33]
The fifth list is: [12, 24, 36]
The list of lists is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

在上面的例子中，您可以观察到我们已经使用给定的列表创建了一个列表列表。

### Python 中使用 append()方法的列表列表

我们还可以使用 python 中的`append()`方法创建一个列表列表。在列表上调用`append()` 方法时，该方法将一个对象作为输入，并将其附加到列表的末尾。为了使用`append()`方法创建一个列表列表，我们将首先创建一个新的空列表。为此，您可以使用方括号符号或`list()`构造函数。`list()`构造函数在没有输入参数的情况下执行时，返回一个空列表。

创建空列表后，我们可以使用`append()`方法将所有给定的列表添加到创建的列表中，以创建 python 中的列表列表，如下面的代码片段所示。

```py
list1 = [1, 2, 3, 4, 5]
print("The first list is:", list1)
list2 = [12, 13, 23]
print("The second list is:", list2)
list3 = [10, 20, 30]
print("The third list is:", list3)
list4 = [11, 22, 33]
print("The fourth list is:", list4)
list5 = [12, 24, 36]
print("The fifth list is:", list5)
myList = []
myList.append(list1)
myList.append(list2)
myList.append(list3)
myList.append(list4)
myList.append(list5)
print("The list of lists is:")
print(myList)
```

输出:

```py
The first list is: [1, 2, 3, 4, 5]
The second list is: [12, 13, 23]
The third list is: [10, 20, 30]
The fourth list is: [11, 22, 33]
The fifth list is: [12, 24, 36]
The list of lists is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]] 
```

如果你想创建一个类似于二维数组的列表，只使用整数数据类型，你可以使用嵌套的 for 循环和`append()`方法来创建一个列表列表。

在这种方法中，我们将首先创建一个新的列表，比如说`myList`。之后，我们将使用嵌套的 for 循环将其他列表追加到`myList`中。在嵌套循环的外部 for 循环中，我们将创建另一个空列表，比如说`tempList`。在内部 for 循环中，我们将使用`append()`方法将数字附加到`tempList`。

在将数字附加到`tempList`之后，我们将得到一个整数列表。之后，我们将进入外部 for 循环，并将`tempList`追加到`myList`。这样，我们可以创建一个列表列表。

例如，假设我们必须创建一个 3×3 的数字数组。为此，我们将使用`range()`函数和 [for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)来创建 python 中的列表列表，如下所示。

```py
myList = []
for i in range(3):
    tempList = []
    for j in range(3):
        element = i + j
        tempList.append(element)
    myList.append(tempList)
print("The list of lists is:")
print(myList)
```

输出:

```py
The list of lists is:
[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
```

### 使用 Python 中的列表理解创建列表列表

不使用 for 循环，你可以使用[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)和`range()`函数以简洁的方式创建一个列表列表，如下例所示。

```py
myList = [[i+j for i in range(3)] for j in range(3)]
print("The list of lists is:")
print(myList)
```

输出:

```py
The list of lists is:
[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
```

## 在 Python 中访问列表列表中的元素

我们可以使用列表索引来访问列表的内容。在平面列表或一维列表中，我们可以使用元素的索引直接访问列表元素。例如，如果我们想使用正值作为列表元素的索引，我们可以使用索引 0 访问列表的第一项，如下所示。

```py
myList = [1, 2, 3, 4, 5]
print("The list is:")
print(myList)
print("The first item of the list is:")
print(myList[0])
```

输出:

```py
The list is:
[1, 2, 3, 4, 5]
The first item of the list is:
1
```

类似地，如果我们使用负值作为列表索引，我们可以使用 index -1 访问列表的最后一个元素，如下所示。

```py
myList = [1, 2, 3, 4, 5]
print("The list is:")
print(myList)
print("The last item of the list is:")
print(myList[-1])
```

输出:

```py
The list is:
[1, 2, 3, 4, 5]
The last item of the list is:
5 
```

如果你想从一个列表的列表中访问内部列表，你可以像上面的例子一样使用列表索引。

如果使用正数作为列表索引，可以使用索引 0 访问列表中的第一个内部列表，如下所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The list of lists is:")
print(myList)
print("The first item of the nested list is:")
print(myList[0])
```

输出:

```py
The list of lists is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The first item of the nested list is:
[1, 2, 3, 4, 5]
```

类似地，如果您使用负数作为列表索引，您可以从列表列表中访问最后一个内部列表，如下所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The list of lists is:")
print(myList)
print("The last item of the nested list is:")
print(myList[-1])
```

输出:

```py
The list of lists is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The last item of the nested list is:
[12, 24, 36]
```

要访问内部列表的元素，需要在列表名称后使用双方括号。这里，第一个方括号表示内部列表的索引，第二个方括号表示内部列表中元素的索引。

例如，您可以使用方括号从列表列表中访问第二个列表的第三个元素，如下所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The list of lists is:")
print(myList)
print("The third element of the second inner list is:")
print(myList[1][2])
```

输出:

```py
The list of lists is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The third element of the second inner list is:
23
```

## 在 Python 中遍历列表列表

若要遍历列表列表的元素，我们可以将用于循环。要打印内部列表，我们可以简单地遍历列表列表，如下所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
for inner_list in myList:
    print(inner_list)
```

输出:

```py
[1, 2, 3, 4, 5]
[12, 13, 23]
[10, 20, 30]
[11, 22, 33]
[12, 24, 36]
```

我们也可以打印列表的元素，而不是在遍历列表时打印整个列表。为此，除了前面示例中显示的 For 循环之外，我们还将使用另一个 for 循环。在内部 for 循环中，我们将遍历内部列表并打印它们的元素，如下所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
for inner_list in myList:
    for element in inner_list:
        print(element, end=",")
    print("")
```

输出:

```py
1,2,3,4,5,
12,13,23,
10,20,30,
11,22,33,
12,24,36,
```

## 在 Python 中从列表列表中删除元素

要从一个列表中删除一个内部列表，我们可以使用不同的 list-objects 方法。

### 使用 pop()方法从列表中删除一个元素

我们可以使用`pop()`方法删除列表中的最后一项。当在列表列表上调用`pop()`方法时，删除最后一个元素并返回最后一个位置的列表。我们可以通过下面一个简单的例子来理解这一点。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
myList.pop()
print("The modified list is:")
print(myList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The modified list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33]]
```

要删除任何其他内部列表，我们需要知道它的索引。例如，我们可以使用`pop()`方法删除列表的第二个元素。为此，我们将调用列表上的`pop()` 方法，并将第二个列表的索引(即 1)传递给 `pop()` 方法。执行后，`pop()`方法将从列表列表中删除第二个内部列表，并返回它，如下例所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
myList.pop(1)
print("The modified list is:")
print(myList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The modified list is:
[[1, 2, 3, 4, 5], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

### 使用 Remove()方法从列表列表中移除元素

如果我们知道必须删除的元素，我们也可以使用`remove()`方法删除一个内部列表。当在列表上调用时，`remove()`方法将待删除的元素作为其输入参数。执行后，它删除作为输入参数传递的元素的第一个匹配项。要删除任何内部列表，我们可以使用如下所示的`remove()`方法。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
myList.remove([1, 2, 3, 4, 5])
print("The modified list is:")
print(myList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The modified list is:
[[12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

## Python 中的扁平化列表

有时，我们需要将一系列列表扁平化来创建一个一维列表。为了使列表变平，我们可以使用 for 循环和`append()`方法。在这种方法中，我们将首先创建一个空列表，比如说`outputList`。

在创建了`outputList`之后，我们将使用一个嵌套的 for 循环来遍历列表的列表。在外部 for 循环中，我们将选择一个内部列表。之后，我们将在内部 for 循环中遍历内部列表的元素。在内部 for 循环中，我们将调用`outputList`上的`append()`方法，并将内部 for 循环的元素作为输入参数传递给`append()`方法。

执行 for 循环后，我们将获得一个从列表列表创建的平面列表，如下面的代码所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = []
for inner_list in myList:
    for element in inner_list:
        outputList.append(element)
print("The flattened list is:")
print(outputList) 
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The flattened list is:
[1, 2, 3, 4, 5, 12, 13, 23, 10, 20, 30, 11, 22, 33, 12, 24, 36]
```

除了使用 for 循环，您还可以使用 list comprehension 来展平列表列表，如下所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = [x for l in myList for x in l]
print("The flattened list is:")
print(outputList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The flattened list is:
[1, 2, 3, 4, 5, 12, 13, 23, 10, 20, 30, 11, 22, 33, 12, 24, 36]
```

## Python 中列表的反向列表

我们可以用两种方法来反转列表的列表。一种方法是只颠倒内部列表的顺序，而保持内部列表中元素的顺序不变。另一种方法是颠倒内部列表中元素的顺序。

### 在 Python 中颠倒列表列表中内部列表的顺序

为了简化内部列表的逆序，我们将首先创建一个空列表，比如说`outputList`。之后，我们将按照相反的顺序遍历列表。在遍历时，我们将把内部列表追加到`outputList`。这样，在 for 循环执行之后，我们将得到列表`outputList`的反向列表。您可以在下面的示例中观察到这一点。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = []
listlen = len(myList)
for i in range(listlen):
    outputList.append(myList[listlen - 1 - i])
print("The reversed list is:")
print(outputList) 
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The reversed list is:
[[12, 24, 36], [11, 22, 33], [10, 20, 30], [12, 13, 23], [1, 2, 3, 4, 5]]
```

不使用 for 循环，可以使用`reverse()`方法来反转列表的列表。在列表中调用`reverse()`方法时，会颠倒列表中元素的顺序。当我们在列表列表上调用`reverse()` 方法时，它将反转内部列表的顺序，如下例所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
myList.reverse()
print("The reversed list is:")
print(myList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The reversed list is:
[[12, 24, 36], [11, 22, 33], [10, 20, 30], [12, 13, 23], [1, 2, 3, 4, 5]]
```

在上面的方法中，原始列表被修改。然而，在前一个例子中情况并非如此。因此，您可以根据是否必须修改原始列表来选择一种方法。

### 在 Python 中颠倒列表列表中内部列表元素的顺序

除了颠倒内部列表的顺序之外，还可以颠倒内部列表中元素的顺序。为此，我们将首先创建一个空列表，比如说`outputList`。之后，我们将使用 for 循环以逆序遍历列表列表。在 for 循环中，我们将创建一个空列表，比如说`tempList`。之后，我们将使用另一个 for 循环以相反的顺序遍历内部列表的元素。在遍历内部列表的元素时，我们会将元素追加到`tempList`中。在内部循环之外，我们将把`tempList`附加到`outputList`上。

执行 for 循环后，我们将得到一个列表，其中所有元素的顺序都相反，如下面的代码所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = []
listlen = len(myList)
for i in range(listlen):
    tempList = []
    currList = myList[listlen - 1 - i]
    innerLen = len(currList)
    for j in range(innerLen):
        tempList.append(currList[innerLen - 1 - j])
    outputList.append(tempList)

print("The reversed list is:")
print(outputList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The reversed list is:
[[36, 24, 12], [33, 22, 11], [30, 20, 10], [23, 13, 12], [5, 4, 3, 2, 1]]
```

您可以使用如下所示的`reverse()` 方法，而不是使用 for 循环来反转内部列表的元素。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = []
listlen = len(myList)
for i in range(listlen):
    myList[listlen - 1 - i].reverse()
    outputList.append(myList[listlen - 1 - i])

print("The reversed list is:")
print(outputList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The reversed list is:
[[36, 24, 12], [33, 22, 11], [30, 20, 10], [23, 13, 12], [5, 4, 3, 2, 1]]
```

这里，我们首先反转了 for 循环中的内部列表。之后，我们将其追加到了`outputList`。通过这种方式，我们获得了列表的列表，其中内部列表以及内部列表的元素以与原始列表相反的顺序出现。

## Python 中列表的排序列表

为了在 python 中对列表进行排序，我们可以使用 sort()方法或 sorted 函数。

### 使用 Sort()方法对 Python 中的列表进行排序

当在列表上调用`sort()`方法时，该方法按照升序对列表的元素进行排序。当我们在列表列表上调用`sort()`方法时，它根据内部列表的第一个元素对内部列表进行排序。

换句话说，在所有内部列表的第一个元素中，第一个元素最小的内部列表被分配列表列表中的第一个位置。类似地，在所有内部列表的第一个元素中，第一个元素最大的内部列表被分配到最后一个位置。

同样，如果两个内部列表在第一个位置有相同的元素，它们的位置根据第二个元素决定。如果内部列表的第二个元素也是相同的，列表的位置将根据第三个元素决定，依此类推。您可以在下面的示例中观察到这一点。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
myList.sort()
print("The sorted list is:")
print(myList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The sorted list is:
[[1, 2, 3, 4, 5], [10, 20, 30], [11, 22, 33], [12, 13, 23], [12, 24, 36]]
```

您还可以更改`sort()`方法的行为。为此，您可以使用`sort()`方法的参数`key`。'`key`'方法将一个操作符或一个函数作为输入参数。例如，如果要根据内部列表的第三个元素对列表进行排序，可以传递一个使用内部列表第三个元素的运算符，如下所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
myList.sort(key=lambda x: x[2])
print("The sorted list is:")
print(myList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The sorted list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

如果你想根据内部列表的最后一个元素对列表进行排序，你可以这样做。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
myList.sort(key=lambda x: x[-1])
print("The sorted list is:")
print(myList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The sorted list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

类似地，如果您想根据内部列表的长度对列表进行排序，您可以将`len()`函数传递给`sort()`方法的参数“`key`”。执行后，`sort()`方法将使用内部列表的长度对列表进行排序。您可以在下面的示例中观察到这一点。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
myList.sort(key=len)
print("The sorted list is:")
print(myList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The sorted list is:
[[12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36], [1, 2, 3, 4, 5]]
```

### 使用 sorted()函数对 Python 中的列表进行排序

如果不允许修改列表的原始列表，可以使用 `sorted()` 功能对列表进行排序。`sorted()`函数的工作方式类似于`sort()` 方法。但是，它不是对原始列表进行排序，而是返回一个排序后的列表。

要对列表进行排序，可以将列表传递给`sorted()` 函数。执行后，`sorted()` 函数将返回排序后的列表，如下例所示。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = sorted(myList)
print("The sorted list is:")
print(outputList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The sorted list is:
[[1, 2, 3, 4, 5], [10, 20, 30], [11, 22, 33], [12, 13, 23], [12, 24, 36]]
```

您还可以使用`key`参数来对使用`sorted()`函数的列表进行排序。例如，您可以使用如下所示的`sorted()` 函数根据内部列表的第三个元素对列表进行排序。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = sorted(myList, key=lambda x: x[2])
print("The sorted list is:")
print(outputList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The sorted list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

如果你想根据内部列表的最后一个元素对列表进行排序，你可以这样做。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = sorted(myList, key=lambda x: x[-1])
print("The sorted list is:")
print(outputList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The sorted list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

类似地，如果您想根据内部列表的长度对列表进行排序，您可以将`len()` 函数传递给`sorted()`函数的参数“`key`”。执行后，`sorted()`函数将使用内部列表的长度返回列表的排序列表。您可以在下面的示例中观察到这一点。

```py
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = sorted(myList, key=len)
print("The sorted list is:")
print(outputList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The sorted list is:
[[12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36], [1, 2, 3, 4, 5]]
```

## 在 Python 中连接两个列表

如果给了你两个列表，你想连接这两个列表，你可以使用+操作符，如下所示。

```py
list1 = [[1, 2, 3, 4, 5], [12, 13, 23]]
list2 = [[10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The first list is:")
print(list1)
print("The second list is:")
print(list2)
print("The concatenated list is:")
myList = list1 + list2
print(myList)
```

输出:

```py
The first list is:
[[1, 2, 3, 4, 5], [12, 13, 23]]
The second list is:
[[10, 20, 30], [11, 22, 33], [12, 24, 36]]
The concatenated list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]] 
```

这里，两个列表的内部元素被连接成一个列表。

## 在 Python 中复制列表列表

要在 python 中复制列表的列表，我们可以使用`copy`模块中提供的`copy()`和`deepcopy()` 方法。

### Python 中列表的浅拷贝列表

`copy()`方法将嵌套列表作为输入参数。执行后，它返回一个类似于原始列表的列表列表。您可以在下面的示例中观察到这一点。

```py
import copy
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = copy.copy(myList)
print("The copied list is:")
print(outputList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The copied list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

上例中讨论的操作称为浅层复制。这里，复制列表和原始列表中的内部元素指向同一个内存位置。因此，每当我们在复制的列表中进行更改时，它都会反映在原始列表中。同样，如果我们在原始列表中做了更改，它会反映在复制的列表中。为了避免这种情况，可以使用`deepcopy()` 方法。

### Python 中列表的深层拷贝列表

`deepcopy()` 方法将一个嵌套列表作为它的输入参数。执行后，它在不同的位置创建嵌套列表所有元素的副本，然后返回复制的列表。因此，每当我们在复制的列表中进行更改时，它不会反映在原始列表中。同样，如果我们在原始列表中做了更改，它不会反映在复制的列表中。您可以在下面的示例中观察到这一点。

```py
import copy
myList = [[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
print("The original list is:")
print(myList)
outputList = copy.deepcopy(myList)
print("The copied list is:")
print(outputList)
```

输出:

```py
The original list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
The copied list is:
[[1, 2, 3, 4, 5], [12, 13, 23], [10, 20, 30], [11, 22, 33], [12, 24, 36]]
```

## 结论

在本文中，我们讨论了 python 中的 list 列表。我们已经讨论了如何对列表的列表执行各种操作。我们还讨论了浅拷贝和深拷贝如何处理列表列表。此外，我们已经讨论了如何在 python 中对列表进行排序、反转、展平和遍历，要了解更多关于 python 编程语言的知识，您可以阅读这篇关于 python 中的[字典理解的文章](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)。您可能也会喜欢这篇关于 python 中的[文件处理的文章。](https://www.pythonforbeginners.com/filehandling/file-handling-in-python)