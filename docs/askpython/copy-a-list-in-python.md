# 用 Python 复制列表的方法

> 原文：<https://www.askpython.com/python/list/copy-a-list-in-python>

在本文中，我们将了解在 Python 中复制列表的各种技术。

[Python List](https://www.askpython.com/python/list/python-list) 是存储和操作数据值的数据结构。

* * *

## 技巧 1:用 Python 复制列表的 extend()方法

Python 内置的 extend()方法可以用来将一个列表中的元素复制到另一个列表中。

`extend() method`基本上以一个 **iterable 作为参数** say list、 [tuple](https://www.askpython.com/python/tuple/python-tuple) 、 [dict](https://www.askpython.com/python/dictionary) 等，它迭代或遍历 iterable 的元素，并以一个元素接一个元素的方式将元素添加到新列表中。

**语法:**

```py
list.iterable(iterable)

```

**举例:**

```py
list1 = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
copy_list = []
copy_list.extend(list1)
print("Input List:", list1) 
print("Copied List:", copy_list) 

```

**输出:**

```py
Input List: [10, 20, 30, 40, 50, 60, 70, 80, 90]
Copied List: [10, 20, 30, 40, 50, 60, 70, 80, 90]

```

* * *

## 技巧 2:用 Python 复制列表的切片操作符

Python `slicing operator`被认为是复制 Python 列表元素最有效的方法。

**语法:**

```py
[start:stop:steps]

```

*   **开始:**决定切片的开始。
*   **停止:**该参数决定 iterable 切片的结束
*   **步骤:**确定要跳过的元素数量或必须执行切片的间隔。

在上面，为了复制列表，我们使用了以下格式的切片:

```py
[:]

```

这仅仅意味着列表的切片将在**开始索引处开始，即索引 0** ，并将在**步长值= 1** 的**最后元素**处结束。

**举例:**

```py
list1 = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
copy_list = []
copy_list = list1[:]
print("Input List:", list1) 
print("Copied List:", copy_list) 

```

输出:

```py
Input List: [10, 20, 30, 40, 50, 60, 70, 80, 90]
Copied List: [10, 20, 30, 40, 50, 60, 70, 80, 90]

```

* * *

## 技巧 3:列表理解用 Python 复制列表

Python 列表理解技术对于在 Python 中复制列表很有用。这只是用一行代码创建语句的另一种方法

**语法:**

```py
[element for element in list]

```

**举例**:

```py
list1 = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
copy_list = []
copy_list = [item for item in list1]
print("Input List:", list1) 
print("Copied List:", copy_list) 

```

在上面的代码片段中，我们使用了列表理解，其中“item”充当指针元素，遍历列表“list1 ”,并以逐个元素的方式复制数据值。

**输出:**

```py
Input List: [10, 20, 30, 40, 50, 60, 70, 80, 90]
Copied List: [10, 20, 30, 40, 50, 60, 70, 80, 90]

```

* * *

## 技巧 4:list()方法复制一个列表

Python `list() method`基本上接受一个 iterable 作为参数，并将序列作为列表返回，即**将 iterable 转换为列表**。

**语法:**

```py
list([iterable])

```

在下面这段代码中，我们将 list-list1 传递给 list()方法，以便用 list-list1 的所有元素创建一个新列表，从而达到复制列表的目的。

**举例:**

```py
list1 = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
copy_list = []
copy_list = list(list1)
print("Input List:", list1) 
print("Copied List:", copy_list) 

```

**输出:**

```py
Input List: [10, 20, 30, 40, 50, 60, 70, 80, 90]
Copied List: [10, 20, 30, 40, 50, 60, 70, 80, 90]

```

* * *

## 技巧 5: Python copy()方法复制列表

Python 内置的`copy() method`可以用来将一个列表的数据项复制到另一个列表中。copy()方法**通过遍历列表**将一个列表中的元素逐个元素地复制到另一个列表中。

**语法:**

```py
list.copy()

```

**举例:**

```py
list1 = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
copy_list = []
copy_list = list1.copy()
print("Input List:", list1) 
print("Copied List:", copy_list) 

```

**输出:**

```py
Input List: [10, 20, 30, 40, 50, 60, 70, 80, 90]
Copied List: [10, 20, 30, 40, 50, 60, 70, 80, 90]

```

* * *

## 技巧 6:复制 Python 列表的 append()方法

Python 内置的`append() method`可以很容易地用于将一个列表的元素复制到另一个列表中。

顾名思义，append()方法**追加，即将列表元素附加到所需列表**的末尾。

但是因为我们使用的是一个空列表，在这种情况下，我们可以使用这个方法在 Python 中复制一个列表。

**语法:**

```py
list.append(value or element)

```

**举例:**

```py
list1 = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
copy_list = []
for ele in list1: copy_list.append(ele) 
print("Input List:", list1) 
print("Copied List:", copy_list) 

```

**输出:**

```py
Input List: [10, 20, 30, 40, 50, 60, 70, 80, 90]
Copied List: [10, 20, 30, 40, 50, 60, 70, 80, 90]

```

* * *

## 结论

因此，我们揭示了用 Python 复制列表的不同方法。

但是，读者们，这并不是学习的结束，我强烈建议每个人参考上面的例子，并尝试实际执行。

* * *

## 参考

*   Python 列表