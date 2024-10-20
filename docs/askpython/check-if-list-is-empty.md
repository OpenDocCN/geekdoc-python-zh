# 检查列表是否为空——3 个简单的方法

> 原文：<https://www.askpython.com/python/list/check-if-list-is-empty>

嘿伙计们！希望你们都过得好。在这篇文章中，我们将关注检查列表是否为空的不同技术。

在进入之前，让我们看一下 Python 列表。

* * *

## 什么是 Python 列表？

[Python List](https://www.askpython.com/python/list/python-list) 是一种将数据动态存储到其中的数据结构。在 Python 中，它服务于[数组](https://www.askpython.com/python/array/python-array-examples)的目的。此外，列表可以存储不同种类的元素，即不同数据类型的元素。

现在，已经理解了列表的工作原理，让我们来理解检查列表是否为空的不同方法。

* * *

## 技巧 1:使用 len()函数

`Python len() function` 可用于检查列表是否为空。如果 len()函数返回零，则该列表为空。

**举例:**

```py
lst = [] ## empty list

length = len(lst)

if length == 0:
    print("List is empty -- ",length)
else:
    print("List isn't empty -- ",length)

```

**输出:**

```py
List is empty --  0

```

* * *

## 技巧 2:使用条件语句

Python 条件 [if 语句](https://www.askpython.com/python/python-if-else-elif-statement)可用于检查列表是否为空，如下所示

**语法:**

```py
if not list:
   #empty
else:

```

**举例:**

```py
lst = [] ## empty list

if not lst:
    print("List is empty.")
else:
    print("List isn't empty.")

```

在上面的例子中，我们使用 if 语句来验证列表中是否存在任何元素。

**输出:**

```py
List is empty.

```

* * *

## 技巧 3:直接比较

我们可以通过直接将列表与空列表进行比较来检查是否存在空列表，即如下所示的[ ]

**语法:**

```py
if list == []:
  #empty
else:

```

**举例:**

```py
lst = list() ## empty list

if lst == []:
    print("List is empty.")
else:
    print("List isn't empty.")

```

这里，我们将指定的列表与空列表进行了比较，以检查给定的列表是否为空。

**输出:**

```py
List is empty.

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，请随时在下面评论。

在那之前，学习愉快！！

* * *

## 参考

*   [检查空 Python 列表的方法— StackOverFlow](https://stackoverflow.com/questions/53513/how-do-i-check-if-a-list-is-empty)