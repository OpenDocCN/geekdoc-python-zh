# Python 元组列表的 5 个示例

> 原文：<https://www.askpython.com/python/list/python-list-of-tuples>

嘿，读者们！在本文中，我们将关注元组的 Python 列表。

* * *

## 什么是 Python 列表和元组？

[Python List](https://www.askpython.com/python/list/python-list) 是维护可变数据元素有序集合的数据结构。

```py
list-name = [ item1, item2, ....., itemN]

```

列表中的元素包含在**方括号[]** 中。

[Python Tuple](https://www.askpython.com/python/tuple/python-tuple) 是一种不可变的数据结构，其元素被括在**括号()**中。

```py
tuple-name = (item1, item2, ...., itemN)

```

* * *

### Python 元组列表

我们可以创建元组的**列表，即元组的元素可以包含在列表中，从而以与 Python 列表相似的方式遵循特征。因为 Python 元组使用较少的空间，所以创建元组列表在各个方面都更有用。**

**举例:**

```py
LT_data = [(1,2,3),('S','P','Q')]
print("List of Tuples:\n",LT_data)

```

**输出:**

```py
List of Tuples:
 [(1, 2, 3), ('S', 'P', 'Q')]

```

* * *

### 使用 zip()函数的 Python 元组列表

[Python zip()函数](https://www.askpython.com/python/built-in-methods/python-zip-function)可用于映射所有列表，使用以下命令创建元组列表:

```py
list(zip(list))

```

`zip() function`根据传递给它的值返回元组的 iterable。此外，`list() function`将创建这些元组的列表，作为 zip()函数的输出。

**举例:**

```py
lst1 = [10,20,30]
lst2 = [50,"Python","JournalDev"]
lst_tuple = list(zip(lst1,lst2))
print(lst_tuple)

```

**输出:**

```py
[(10, 50), (20, 'Python'), (30, 'JournalDev')]

```

* * *

### 定制的元素分组，同时形成元组列表

在形成元组列表时，我们可以根据列表/元组中的元素数量来提供定制的元素分组。

```py
[element for element in zip(*[iter(list)]*number)]

```

列表理解和`zip() function`一起用于将元组转换成列表并创建元组列表。`Python iter() function`用于一次迭代一个对象的元素。“**号**将指定要加入单个元组以形成列表的元素的数量。

**例 1:**

```py
lst = [50,"Python","JournalDev",100]
lst_tuple = [x for x in zip(*[iter(lst)])]
print(lst_tuple)

```

在上面的例子中，我们已经使用 [iter()方法](https://www.askpython.com/python/python-iter-function)形成了一个元组列表，其中一个元素包含在一个元组中。

**输出:**

```py
[(50,), ('Python',), ('JournalDev',), (100,)]

```

**例 2:**

```py
lst = [50,"Python","JournalDev",100]
lst_tuple = [x for x in zip(*[iter(lst)]*2)]
print(lst_tuple)

```

在这个例子中，两个元素包含在一个元组中，以形成一个元组列表。

**输出:**

```py
[(50, 'Python'), ('JournalDev', 100)]

```

* * *

### 使用 map()函数的 Python 元组列表

[Python map 函数](https://www.askpython.com/python/built-in-methods/map-method-in-python)可用于创建元组列表。`map() function`将函数映射并应用到传递给该函数的 iterable。

```py
map(function, iterable)

```

**举例:**

```py
lst = [[50],["Python"],["JournalDev"],[100]]
lst_tuple =list(map(tuple, lst))
print(lst_tuple)

```

在这个例子中，我们使用 map()函数将输入列表映射到 tuple 函数。此后，list()函数用于创建映射元组值的列表。

**输出:**

```py
[(50,), ('Python',), ('JournalDev',), (100,)]

```

* * *

### 使用列表理解和 tuple()方法的 Python 元组列表

Python tuple()方法与 List Comprehension 一起可用于形成元组列表。

`tuple() function`帮助从传递给它的元素集中创建元组。

**举例:**

```py
lst = [[50],["Python"],["JournalDev"],[100]]
lst_tuple =[tuple(ele) for ele in lst]
print(lst_tuple)

```

**输出:**

```py
[(50,), ('Python',), ('JournalDev',), (100,)]

```

* * *

## 结论

到此，我们已经到了文章的结尾。我希望你们都喜欢学习 Python 元组列表这个有趣的概念。

如果你有任何疑问，欢迎在下面评论。

* * *

## 参考

*   [创建元组列表— StackOverflow](https://stackoverflow.com/questions/7313157/python-create-list-of-tuples-from-lists/7313188)