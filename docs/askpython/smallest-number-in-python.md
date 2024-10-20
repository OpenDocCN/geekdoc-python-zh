# 在 Python 中寻找最小数字的 3 种简单方法

> 原文：<https://www.askpython.com/python/examples/smallest-number-in-python>

你好。这篇文章是为希望理解用 Python 寻找最小数的基本代码的初学者而写的。让我们开始吧。

## 如何在 Python 中求最小的数？

我们的目标是在 Python 中找到列表中给定的所有数字中最小的数字。

说如果列表是:[32，54，67，21]

输出应该是:21

在本文中，我们将了解 3 种不同的方法来做到这一点。

### 1.使用 Python min()

[Min()](https://www.askpython.com/python/built-in-methods/python-min-method) 是 python 中的内置函数，它以一个列表作为参数，返回列表中最小的数字。下面给出一个例子

```py
#declaring a list
list1 = [-1, 65, 49, 13, -27] 
print ("list = ", list1)

#finding smallest number
s_num = min (list1)
print ("The smallest number in the given list is ", s_num)

```

**输出:**

```py
list = [-1, 65, 49, 13, -27]
The smallest number in the given list is  -27

```

这是求最小数的最简单的方法之一。您需要做的就是将列表作为参数传递给 min()。

### 2.使用 Python 排序()

[Sort()](https://www.askpython.com/python/list/python-sort-list) 是 python 中的另一个内置方法，它不返回列表中最小的 号。相反，它按升序对列表进行排序。

所以通过对列表排序，我们可以使用索引来访问列表的第一个元素，这将是列表中最小的数字。让我们看看代码:

```py
#declaring a list
list1 = [17, 53, 46, 8, 71]
print ("list = ", list1)

#sorting the list
list1.sort ()

#printing smallest number
print ("The smallest number in the given list is ", list1[0])

```

**输出:**

```py
list =  [17, 53, 46, 8, 71]
The smallest number in the given list is 8

```

### 3.使用“for”循环

```py
ls1 = []
total_ele = int (input (" How many elements you want to enter? "))

#getting list from the user
for i in range (total_ele):
  n =int (input ("Enter a number:"))
  ls1.append(n)
print (ls1)
min = ls1[0]

#finding smallest number
for i in range (len (ls1)):
  if ls1[i] < min:
    min = ls1[i]
print ("The smallest element is ", min)

```

在上面的代码中，我们使用两个 [**用于**循环](https://www.askpython.com/python/python-loops-in-python)，一个用于从用户处获取列表元素，另一个用于从列表中找到最小的数字。

从用户那里获得元素后，我们将列表的第一个元素(索引为 0)定义为最小的数(min)。然后使用 for 循环，我们将列表中的每个元素与最小的**进行比较，如果任何元素小于最小**的**，它将成为新的最小**的**。**

这就是我们如何从用户给定的列表中得到最小的数字。

**上述代码的输出为:**

```py
How many elements you want to enter? 4
Enter a number: 15
Enter a number: 47
Enter a number: 23
Enter a number: 6
[15, 47, 23, 6]
The smallest number is  6

```

## 结论

这是一些在 python 中从给定列表中寻找最小数字的方法。希望你明白这一点！如果有任何问题，请随时提问。谢谢大家！🙂