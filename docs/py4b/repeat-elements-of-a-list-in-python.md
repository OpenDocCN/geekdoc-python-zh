# Python 中列表的重复元素

> 原文：<https://www.pythonforbeginners.com/lists/repeat-elements-of-a-list-in-python>

python 中的列表是最常用的数据结构之一。我们已经讨论了列表上的各种操作，比如[计算列表中元素的频率](https://www.pythonforbeginners.com/lists/count-the-frequency-of-elements-in-a-list)或者[反转列表](https://www.pythonforbeginners.com/lists/how-to-reverse-a-list-in-python)。在本文中，我们将研究在 python 中重复列表元素的不同方法。

## 如何在 Python 中重复列表元素

为了在 python 中重复列表中的元素，我们将列表中现有的元素插入到同一个列表中。这样，列表中的元素就会重复出现。
例如，如果我们有一个列表`myList=[1,2,3,4,5]`，我们必须重复列表的元素两次，输出列表将变成`[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]`。
在 python 中，我们可以使用 append()方法、extend()方法或*操作符来重复列表中的元素。我们将在下面的章节中详细讨论所有这些方法。

## 使用 append()方法重复列表中的元素

append()方法用于将元素追加到列表的最后。在列表上调用时，append()方法获取一个元素并将其添加到列表中，如下所示。

```py
myList = [1, 2, 3, 4, 5]
print("The given list is:", myList)
number = 10
print("The input number to append to list is:", number)
myList.append(number)
print("The output list is:", myList)
```

输出:

```py
The given list is: [1, 2, 3, 4, 5]
The input number to append to list is: 10
The output list is: [1, 2, 3, 4, 5, 10]
```

为了将列表中的元素重复 n 次，我们首先将现有的列表复制到一个临时列表中。之后，我们将使用 for 循环、range()方法和 append()方法将临时列表的元素添加到原始列表中 n 次。执行 for 循环后，列表中的元素将重复 n 次，如下例所示。

```py
myList = [1, 2, 3, 4, 5]
print("The given list is:", myList)
tempList = list(myList)
count = 2
print("Number of Times to repeat the elements:",count)
for i in range(count):
    for element in tempList:
        myList.append(element)
print("The output list is:", myList)
```

输出:

```py
The given list is: [1, 2, 3, 4, 5]
Number of Times to repeat the elements: 2
The output list is: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
```

## 使用 extend()方法重复列表中的元素

我们可以使用 extend()方法一次添加列表中的所有元素，而不是使用 append()方法逐个添加元素。在列表上调用 extend()方法时，它接受列表作为输入参数，并将输入列表添加到现有列表中，如下所示。

```py
myList = [1, 2, 3, 4, 5]
print("The given list is:", myList)
newList = [6, 7]
print("The input list to append elements from is:", newList)
myList.extend(newList)
print("The output list is:", myList)
```

输出:

```py
The given list is: [1, 2, 3, 4, 5]
The input list to append elements from is: [6, 7]
The output list is: [1, 2, 3, 4, 5, 6, 7]
```

要使用 extend()方法重复一个列表的元素，我们将把现有列表的元素复制到一个临时列表中。之后，我们将使用带有 for 循环的 extend()方法来重复列表中的元素，如下所示。

```py
myList = [1, 2, 3, 4, 5]
print("The given list is:", myList)
tempList = list(myList)
count = 2
print("Number of Times to repeat the elements:",count)
for i in range(count):
    myList.extend(tempList)
print("The output list is:", myList)
```

输出:

```py
The given list is: [1, 2, 3, 4, 5]
Number of Times to repeat the elements: 2
The output list is: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
```

## 使用*运算符

运算符也可以用来重复列表中的元素。当我们使用*运算符将一个列表与任意数字相乘时，它会重复给定列表中的元素。这里，我们只需要记住，要重复元素 n 次，我们必须将列表乘以(n+1)。

```py
myList = [1, 2, 3, 4, 5]
print("The given list is:", myList)
count = 2
print("Number of Times to repeat the elements:", count)
myList = myList * 3
print("The output list is:", myList)
```

输出:

```py
The given list is: [1, 2, 3, 4, 5]
Number of Times to repeat the elements: 2
The output list is: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
```

## 结论

在本文中，我们讨论了在 python 中重复列表元素的三种方法。在所有这些方法中，使用*操作符的方法是最有效和最容易实现的。因此，我建议您使用这种方法来重复 python 中的列表元素。要了解更多关于 python 中的列表，你可以阅读这篇关于 python 中的列表理解的文章。