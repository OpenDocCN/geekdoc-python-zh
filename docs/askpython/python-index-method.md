# Python index()方法

> 原文：<https://www.askpython.com/python/list/python-index-method>

## 介绍

所以今天在本教程中，我们将讨论 Python index()方法。

为字符串、列表和元组类定义了`index()`方法。对于[字符串](https://www.askpython.com/python/string)，该方法返回给定子字符串出现的最小索引。而对于[列表](https://www.askpython.com/python/list/python-list)和[元组](https://www.askpython.com/python/tuple/python-tuple)，该方法返回给定对象所在位置的最小索引。

## 使用 Python index()方法

从上面的定义可以清楚地看出，Python index()方法只是为 iterables 定义的。因此，它不适用于字典或集合，因为它们不遵循索引顺序。

对于一个**可迭代**，使用 Python `index()`方法的语法如下所示。

```py
iterable.index(sub, start, end)

```

这里，

*   **iterable** 可以是任何对象，如列表、字符串或元组，
*   **sub** 是 iterable 中要找到其最小索引的子字符串或项目，
*   **start** 是搜索开始的起始索引。如果没有指定，它默认设置为 0，
*   **end** 是搜索将进行到的最后一个索引。如果没有指定，它的值被认为等于 iterable 的长度。

**请注意**:如果在 start to end index 范围内没有找到 object sub，该方法会引发一个`ValueError`。

## Python 中 index()方法的示例

现在我们知道了对任何**可迭代**使用`index()`方法的语法，让我们通过一些例子来尝试使用它。

### 1.列表索引()

`index()`方法是**列表**类的成员函数。并且广泛用于搜索列表中的值。

```py
# initialisation of variables
list1 = [9, 2, 7, 6, 8, 2, 3, 5, 1]

#index() with list
print("Value 2 first found at index: ", list1.index(2))
print("Value 2 first found at index(within range 4-7) : ", list1.index(2,4,7))

```

**输出**:

```py
Value 2 first found at index:  1
Value 2 first found at index(within range 4-7) :  5

```

在上面的例子中，我们首先初始化了一个列表`list1`。接下来，我们试图得到值 **2** 出现的最小索引。

当我们试图寻找没有指定开始和结束的值 2 时，程序返回一个索引 **1** 。因此，很明显，在整个列表中，`index()`方法返回的是存在 **2** 的最小索引。

接下来对于一个指定的范围( **4-7** ，该方法给我们一个值 **5** 。这是 **2** 在列表中第二次出现的索引。但在 4-7 的范围内，5 号指数是最小的。

注意:`index()`方法对元组也以同样的方式工作。

### 2.字符串 Python 索引()

到了**字符串**，成员函数 Python `index()`返回指定子字符串开始处的最小索引。

让我们看一个例子。

```py
# initialisation of string
str1 = "Python Python"

#index() with string
print("sub-string 'Py' first found at index: ", str1.index('Py'))
print("sub-string 'Py' first found at index(within range 5-10) : ", str1.index('Py',5,10))

```

**输出**:

```py
sub-string 'Py' first found at index:  0
sub-string 'Py' first found at index(within range 5-10) :  7

```

在这里，对于未指定范围的第一次搜索，Python `index()`方法在搜索子字符串“ **Py** 时返回 0。正如我们所见，这是字符串`string1`中出现“ **Py** 的最小索引。

当我们指定一个范围(此处为 **5-10** )时，该方法相应地从第 5 到第 10 个索引搜索“Py”。从输出中可以清楚地看到，该函数在**第 7 个**位置找到了子字符串的开始。

## 结论

对于任何 iterable，值得注意的是，如果在给定的 iterable 中没有找到传递的 **sub(object)** ，则引发`ValueError`。

所以在本教程中，我们学习了 Python 中的`index()`方法的工作和使用。关于这个话题的任何进一步的问题，请在下面随意评论。

## 参考

*   [Python 列表](https://www.askpython.com/python/list/python-list)–要知道的 15 件事，
*   [Python 字符串](https://www.askpython.com/python/string)–教程，
*   Python 字符串索引()–日志开发帖子，
*   [Python 中的数组索引](https://stackoverflow.com/questions/15726618/array-indexing-in-python)——堆栈溢出问题。