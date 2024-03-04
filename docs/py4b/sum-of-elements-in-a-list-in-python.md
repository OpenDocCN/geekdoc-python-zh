# Python 中列表中元素的总和

> 原文：<https://www.pythonforbeginners.com/basics/sum-of-elements-in-a-list-in-python>

Python 列表是最常用的数据结构之一。我们经常需要对列表执行不同的操作。在本文中，我们将讨论在 python 中查找列表中元素之和的不同方法。

## 使用 For 循环查找列表中元素的总和

查找列表中元素总和的第一种方法是遍历列表，并使用循环的[将每个元素相加。为此，我们将首先使用 len()方法计算列表的长度。之后，我们将声明一个变量`sumOfElements`为 0。之后，我们将使用 range()函数创建一个从 0 到`(length of the list-1)`的数字序列。使用这个序列中的数字，我们可以访问给定列表的元素，并将它们添加到`sumOfElements`中，如下所示。](https://www.pythonforbeginners.com/basics/loops)

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("The given list is:")
print(myList)
list_length=len(myList)
sumOfElements=0
for i in range(list_length):
    sumOfElements=sumOfElements+myList[i]

print("Sum of all the elements in the list is:", sumOfElements) 
```

输出:

```py
The given list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9]
Sum of all the elements in the list is: 45
```

或者，我们可以使用 for 循环直接遍历列表。这里，我们将直接访问列表中的每个元素，并将它们添加到`sumOfElements`中，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("The given list is:")
print(myList)
sumOfElements = 0
for element in myList:
    sumOfElements = sumOfElements + element

print("Sum of all the elements in the list is:", sumOfElements) 
```

输出:

```py
The given list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9]
Sum of all the elements in the list is: 45
```

## 使用 While 循环查找列表中元素的总和

我们还可以使用一个 [while 循环](https://www.pythonforbeginners.com/loops/python-while-loop)来查找列表中元素的总和。为此，我们将首先使用 len()方法计算列表的长度。之后，我们将初始化名为 count 和`sumOfElements`的变量。我们将把两个元素都初始化为 0。

在 while 循环中，我们将使用 count 变量访问列表中的每个元素，并将它们添加到`sumOfElements`。之后，我们会将计数值增加 1。我们将继续这个过程，直到计数等于列表的长度。

你可以用 python 写一个程序来求一个列表中元素的和，如下。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("The given list is:")
print(myList)
list_length = len(myList)
sumOfElements = 0
count = 0
while count < list_length:
    sumOfElements = sumOfElements + myList[count]
    count = count + 1

print("Sum of all the elements in the list is:", sumOfElements)
```

输出:

```py
The given list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9]
Sum of all the elements in the list is: 45
```

## 使用 Sum()函数对列表中的元素求和

Python 还为我们提供了一个内置的 sum()函数来计算任何集合对象中元素的总和。sum()函数接受一个 iterable 对象，如 list、tuple 或 set，并返回该对象中元素的总和。

可以使用 sum()函数计算列表元素的总和，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("The given list is:")
print(myList)
sumOfElements = sum(myList)
print("Sum of all the elements in the list is:", sumOfElements)
```

输出:

```py
The given list is:
[1, 2, 3, 4, 5, 6, 7, 8, 9]
Sum of all the elements in the list is: 45
```

## 结论

在本文中，我们讨论了用 python 查找列表中元素总和的不同方法。要阅读更多关于 python 中的列表，你可以阅读这篇关于如何在 python 中比较两个列表的文章。你可能也会喜欢[列表中的这篇文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)