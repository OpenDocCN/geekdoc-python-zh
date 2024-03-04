# 如何在 Python 中反转列表

> 原文：<https://www.pythonforbeginners.com/lists/how-to-reverse-a-list-in-python>

[查看帖子](https://www.pythonforbeginners.com/lists/how-to-reverse-a-list-in-python)

python 中的列表是最常用的数据结构。我们经常需要反转一个列表，从列表中删除一个元素，向列表中添加或插入一个元素，或者在 python 中合并两个列表。在这篇文章中，我们将看看在 python 中反转列表的不同方法。

## 使用循环反转列表

反转列表最简单的方法是以相反的顺序遍历列表。在迭代时，我们可以将每个遍历的元素添加到一个新列表中，这将创建一个反向列表。为此，首先我们将计算列表的长度，然后我们将以相反的顺序遍历列表，我们将使用 append()方法将每个遍历的元素追加到新列表中，如下所示。

```py
myList=[1,2,3,4,5,6,7]
print("Original List is:",myList)
list_len=len(myList)
newList=[]
for i in range(list_len):
    newList.append(myList[list_len-1-i])
print("reversed list is:",newList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
reversed list is: [7, 6, 5, 4, 3, 2, 1]
```

虽然我们已经用上面的方法创建了一个反向列表，但是在迭代之前计算列表的长度是有开销的。

我们也可以使用 while 循环来反转列表。为此，在每次迭代中，我们将使用 pop()方法从原始列表中弹出一个元素，直到列表变空，如下所示。

```py
myList=[1,2,3,4,5,6,7]
print("Original List is:",myList)
newList=[]
while myList: #empty list points to None and will break the loop
    temp=myList.pop()
    newList.append(temp)

print("reversed list is:",newList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
reversed list is: [7, 6, 5, 4, 3, 2, 1]
```

这种反转列表的方式会删除原始列表中的所有元素。因此，只有当程序中不再需要原始列表时，才应该使用这种方式来反转 python 中的列表。

## 使用切片反转列表

切片是一种操作，借助它我们可以访问、更新或删除列表中的元素。我们也可以使用切片来反转列表，如下所示。

```py
myList=[1,2,3,4,5,6,7]
print("Original List is:",myList)
newList=myList[::-1]
print("reversed list is:",newList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
reversed list is: [7, 6, 5, 4, 3, 2, 1]
```

## 使用 reversed()方法

我们可以使用 reversed()方法来反转列表。reversed()方法将一个类似迭代器的列表作为输入，并返回一个反向迭代器。之后，我们可以使用 for 循环从迭代器创建反向列表，如下所示。

```py
myList=[1,2,3,4,5,6,7]
print("Original List is:",myList)
newList=[]
reverse_iterator=reversed(myList)
for i in reverse_iterator:
    newList.append(i)
print("reversed list is:",newList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
reversed list is: [7, 6, 5, 4, 3, 2, 1]
```

我们也可以使用[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)来代替 for 循环，从反向迭代器创建一个列表，如下所示。

```py
myList=[1,2,3,4,5,6,7]
print("Original List is:",myList)
reverse_iterator=reversed(myList)
newList= [i for i in reverse_iterator]
print("reversed list is:",newList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
reversed list is: [7, 6, 5, 4, 3, 2, 1]
```

我们也可以使用 list()构造函数直接将反向迭代器转换成列表，如下所示。

```py
myList=[1,2,3,4,5,6,7]
print("Original List is:",myList)
reverse_iterator=reversed(myList)
newList=list(reverse_iterator)
print("reversed list is:",newList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
reversed list is: [7, 6, 5, 4, 3, 2, 1]
```

## 使用 Reverse()方法反转列表

在上面描述的所有方法中，我们创建了一个新的列表，以逆序存储原始列表的值。现在，我们将了解如何在适当的位置反转列表，即如何反转将作为输入提供给我们的同一列表。

要以这种方式反转列表，我们可以使用 reverse()方法。在列表上调用 reverse()方法时，将按如下方式反转元素的顺序。

```py
myList=[1,2,3,4,5,6,7]
print("Original List is:",myList)
myList.reverse()
print("reversed list is:",myList)
```

输出:

```py
Original List is: [1, 2, 3, 4, 5, 6, 7]
reversed list is: [7, 6, 5, 4, 3, 2, 1]
```

## 结论

在本文中，我们看到了如何使用 for 循环、切片、列表理解、reverse()方法和 reversed()方法在 [python 中反转列表。请继续关注更多内容丰富的文章。](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)