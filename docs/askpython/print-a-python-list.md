# 打印 Python 列表的 3 种简单方法

> 原文：<https://www.askpython.com/python/list/print-a-python-list>

读者朋友们，你们好！在本文中，我们将关注打印 Python 列表的不同方法。所以，让我们开始吧！

* * *

## 首先，什么是 Python 列表？

Python 为我们提供了各种数据结构来存储和处理数据。名单就是其中之一。

[Python List](https://www.askpython.com/python/list/python-list) 是存储可变数据值序列的数据结构。此外，列表可以被认为是元素的有序集合，即它们遵循元素的顺序。

现在让我们关注一些打印列表元素的技术。

* * *

## 1.使用 map()函数打印 Python 列表

Python [map()](https://www.askpython.com/python/built-in-methods/map-method-in-python) 函数可以和 [join()](https://www.askpython.com/python/string/python-string-join-method) 函数联合使用，轻松打印 Python 列表。

**语法:**

```py
''.join(map(str,list))

```

**举例:**

```py
lst = [10,20,30,'John',50,'Joe']
print("Elements of the List:\n")
print('\n'.join(map(str, lst))) 

```

**说明:**

*   首先，我们应用 map 方法，将列表中的值转换为字符串，也就是说，我们将它们映射为字符串格式。
*   然后，我们应用 join 方法来组装元素，并包含一个新行来分隔元素。

**输出:**

```py
Elements of the List:

10
20
30
John
50
Joe

```

* * *

## 2.使用' * '符号打印 Python 列表

接下来，我们将使用 **Python '* '符号**来打印一个列表。

**语法:**

```py
*list

```

我们可以通过包含 **sep** 值来定制输出。下面，我们将分隔值设置为一个**换行符**。

**举例:**

```py
lst = [10,20,30,'John',50,'Joe'] 
print("Elements of the List:\n")
print(*lst, sep = "\n") 

```

**输出:**

```py
Elements of the List:

10
20
30
John
50
Joe

```

* * *

## 3.简单的方法——使用 for 循环

作为初学者，天真的方法总是最好的入门方法！

在这个方法中，我们使用循环的[遍历列表的每个元素，然后在这里打印一个 Python 列表。](https://www.askpython.com/python/python-for-loop)

**语法:**

```py
for element in list:
    print(element)

```

**举例:**

```py
lst = [10,20,30,'John',50,'Joe'] 
print("Elements of the List:\n")
for x in lst:
    print(x)

```

**输出:**

```py
Elements of the List:

10
20
30
John
50
Joe

```

* * *

## 结论

如上所述，这些是打印 Python 列表的不同方式。当然，我们可以用更多的方法来做同样的事情。但是我相信，方法 1 和 2(上面讨论过的)仍然很突出。

这个话题到此结束。更多与 Python 相关的话题，敬请关注，在此之前，请继续与我们一起学习和成长！！🙂