# 什么是 Python reversed()函数？

> 原文：<https://www.askpython.com/python/built-in-methods/python-reversed-function>

你好，希望你一切都好！在本文中，我们将了解内置函数的工作原理— **Python reversed()函数**。

* * *

## Python reversed()函数的工作原理

Python 为我们提供了大量的内置函数来处理和操作数据。

Python reversed()函数就是这样一个函数。

`Python reversed() function`以相反的顺序处理输入数据值。reversed()函数处理列表、字符串等。并通过以逆序处理给定数据元素的序列来返回迭代器。

因此，可以说 Python reversed()函数可以用于任何数据结构的数据值的反向筛选。

理解了 reversed()函数的工作原理后，现在让我们来关注一下它的语法。

* * *

## Python 反向()函数的语法

如上所述，Python reversed()函数以相反的顺序遍历数据值。

**语法:**

```py
reversed(data)

```

reversed()函数返回一个反向迭代器，即它返回一个迭代器，该迭代器以相反的顺序获取并表示数据值。

现在，让我们在下面的小节中理解 Python reversed()函数的实现。

* * *

## 通过实例实现 reversed()函数

在下面的例子中，我们已经创建了一个数字数据值的列表，并将其传递给 reversed()函数。

**例 1:**

```py
lst = [10, 15, 43, 56]
rev = reversed(lst)
print(rev)
print(list(rev))

```

reversed()函数返回反向迭代器，如我们尝试执行— `print(rev)`时的输出所示。

此外，为了访问 reversed()函数操作的数据值，我们使用 list()函数通过反向迭代器打印数据值。

**输出:**

```py
<list_reverseiterator object at 0x00000272A6901390>
[56, 43, 15, 10]

```

在这个例子中，我们将字符串值传递给了列表，然后将它传递给了 reversed()函数。

**例 2:**

```py
lst = ['Python','Java','C++','ML']
rev = reversed(lst)
print(list(rev))

```

**输出:**

```py
['ML', 'C++', 'Java', 'Python']

```

**例 3:**

```py
tuple_data = ('Python','Java','C++','ML',100,21)
rev = reversed(tuple_data)
print(tuple(rev))

```

现在，我们已经创建了一个数据值的[元组](https://www.askpython.com/python/tuple/python-tuple)，并将其传递给 reversed()函数来反转数据值。

**输出:**

```py
(21, 100, 'ML', 'C++', 'Java', 'Python')

```

**例 4:**

```py
data = list(range(1,5))
print("Original Data: ",data)
rev = list(reversed(data))
print("Reversed Data: ",rev)

```

Python reversed()函数可以与 [range()函数](https://www.askpython.com/python/built-in-methods/python-range-method)一起使用，以逆序返回数据值序列的迭代器。

**输出:**

```py
Original Data:  [1, 2, 3, 4]
Reversed Data:  [4, 3, 2, 1]

```

* * *

## 结论

到此为止，我们已经结束了这个话题。如果你有任何疑问，欢迎在下面评论。请继续关注我们，获取更多此类帖子！

* * *

## 参考

*   Python reversed()函数— JournalDev