# 如何在 Python 中从列表中移除元素？

> 原文：<https://www.askpython.com/python/list/remove-elements-from-a-list>

在本文中，我们将介绍 Python 中从列表中移除元素的所有方法。

[Python 列表](https://www.askpython.com/python/list/python-list)是日常编程中使用的最基本的数据结构。我们会遇到需要从列表中移除元素的情况，在本文中，我们将讨论这一点。

## 1.根据值从列表中删除元素

Python 是一种著名的编程语言的原因之一是存在大量的内置函数。这些内置的函数非常方便，因此使得 Python 的编写非常方便。

### 移除()函数

Python 有一个内置的函数，`remove()`可以帮助我们根据值删除元素。

```py
# List of integers
lis = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# Remove element with value = 1
lis.remove(1)

# Printing the list
print(lis)

# Remove element with value = 9
lis.remove(9)

# Printing the list
print(lis)

```

**输出:**

```py
[3, 4, 1, 5, 9, 2, 6, 5]
[3, 4, 1, 5, 2, 6, 5]

```

这里需要注意的关键点是:

*   `remove()`函数接受一个参数——要删除的值。
*   如果给定值出现多次，则删除第一个。
*   移除一个元素不会在该位置留下空白，它只是将后面的元素向左移动。
*   如果列表中没有这样的元素，那么脚本会产生一个错误。

* * *

### remove()函数的无错误用法

有一种简单的方法可以在删除一个元素时绕过错误，以防程序员不知道它在列表中。我们将使用条件中的[来实现这一点。](https://www.askpython.com/python/python-if-else-elif-statement)

```py
# List of integers
lis = [1, 4, 2, 6, 1, 9, 10]

# Value to be removed
val = 6

# Check if the list contains the value 
if val in lis:

	# Remove the value from the list
	lis.remove(val)

# Printing the list
print(lis)

```

**输出:**

```py
[3, 1, 4, 1, 5, 9, 2, 5]

```

在上面的代码片段中，我们在移除之前首先检查列表中的值是否存在。

* * *

### 删除列表中出现的所有值

正如我们前面提到的，remove()函数只删除第一次出现的值。为了取出所述值的所有实例，我们将使用 [while 循环](https://www.askpython.com/python/python-while-loop)。

```py
# List of integers
lis = [1, 4, 2, 6, 1, 9, 10]

# Value to be removed
val = 1

# Run until the list containes the value 
while val in lis:

	# Remove the value from the list
	lis.remove(val)

# Printing the list
print(lis)

```

**输出:**

```py
[3, 4, 5, 9, 2, 6, 5]

```

以上总结了`remove()`功能的用法。

* * *

## 2.基于索引删除元素

有几种方法可以根据索引删除元素。让我们快速浏览一下每一条。

### 的关键字

是 Python 中一个强大的工具，用来移除整个对象。它还可以用来从给定的列表中移除元素。

```py
# List of integers
lis = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# Removing element from the start (index = 0)
del lis[0]

# Printing the list
print(lis)

# Removing element from the last (index = -1)
del lis[-1]

# Printing the list
print(lis)

```

**输出:**

```py
[1, 4, 1, 5, 9, 2, 6, 5]
[1, 4, 1, 5, 9, 2, 6]

```

从上述脚本中得出的一些观察结果如下:

*   `del`不是方法。这是一个删除放在它后面的项的语句。
*   从特定索引中移除元素会将下一个值移动到该特定索引，如果它不是最后一个索引的话。
*   提供大于(或等于)列表长度的索引将引发*“超出范围”*错误。

* * *

### pop()函数

顾名思义，`pop()`函数从指定的索引中弹出一个元素。

```py
# List of integers
lis = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# Removing the fourth element (index = 3)
lis.pop(3)

# Printing the list
print(lis)

# Removing the second last element (index = -2)
lis.pop(-2)

# Printing the list
print(lis)

```

**输出:**

```py
[3, 1, 4, 5, 9, 2, 6, 5]
[3, 1, 4, 5, 9, 2, 5]

```

我们在这里了解到的`pop()`方法是:

*   它有一个参数——列表的索引
*   它根据给定的索引从列表中移除元素。下列元素向左移动。
*   它支持反向索引。
*   如果索引不在列表中，它将引发一个*“超出范围”*错误。

我们有一篇关于使用 [pop()方法](https://www.askpython.com/python/list/python-list-pop)的完整文章。

* * *

## 3.从列表中移除一定范围的元素

Python 提供了从列表中删除一系列元素的功能。这可以通过`del`语句来完成。

```py
# List of integers
lis = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# Removing first and second element
del lis[0:2]

# Printing the list
print(lis)

# Removing last two elements
del lis[-2:]

# Printing the list
print(lis)

```

**输出:**

```py
[4, 1, 5, 9, 2, 6, 5]
[4, 1, 5, 9, 2]

```

让我们试着理解这个过程:

*   要从一个序列的列表中删除多个元素，我们需要向`del`语句提供一系列元素。
*   一系列元素采用开始索引和/或结束索引，由冒号`':'`分隔。
*   要删除的值包括起始索引，但不包括结束索引处的值。
*   如果缺少结束索引，则范围包括列表末尾之前的所有元素。

* * *

## 4.从列表中删除所有元素

Python 提供了一种在一行中清空整个列表的方法。

```py
# List of integers
lis = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# Removing all elements
lis.clear()

# Printing the list
print(lis)

```

**输出:**

```py
[ ]

```

如果该函数应用于空列表，它不会引发任何错误。

* * *

## 结论

使用从列表中移除元素的方式取决于您的判断，是通过值还是索引。不同的环境需要不同的方法，因此 Python 提供了各种从 Python 列表中删除元素的方法。

我们希望读者阅读这篇文章没有困难。如有任何疑问，欢迎在下方评论。