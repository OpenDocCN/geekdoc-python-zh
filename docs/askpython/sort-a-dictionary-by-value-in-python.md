# Python 中按值对字典排序的 3 种方法

> 原文：<https://www.askpython.com/python/dictionary/sort-a-dictionary-by-value-in-python>

嘿，伙计们！在本文中，我们将揭示用 Python 按值对字典进行排序的步骤。

* * *

## 技巧 1:在 Python 中使用 sorted()函数按值对字典进行排序

在 Python 中，`sorted() method`和 Python `operator.itemgetter() method`可以用来按值对字典进行排序。

Python 运算符模块具有各种内置函数来处理数据和值。operator.itemgetter(m)方法将输入对象视为 iterable，并从 iterable 中获取所有“m”值。

**语法:Python operator . item getter(m)**

```py
operator.itemgetter(value)

```

[Python sorted](https://www.askpython.com/python/built-in-methods/python-sorted-method) ()方法将 dict 按升序/降序排序。

**语法:Python sorted()方法**

```py
sorted(iterable, key)

```

*   `iterable`:表示要排序的元素。
*   `key`(可选):决定元素排序顺序的函数。

**举例:**

```py
from operator import itemgetter
inp_dict = { 'a':3,'ab':2,'abc':1,'abcd':0 }
print("Dictionary: ", inp_dict)
sort_dict= dict(sorted(inp_dict.items(), key=operator.itemgetter(1))) 
print("Sorted Dictionary by value: ", sort_dict)

```

在上面的函数中，我们设置了参数 key = operator.itemgetter(1 ),因为这里的“1”表示字典的值,“0”表示字典的键。因此，我们已经按照升序对字典进行了排序。

**输出:**

```py
Dictionary:  {'a': 3, 'ab': 2, 'abc': 1, 'abcd': 0}
Sorted Dictionary by value:  {'abcd': 0, 'abc': 1, 'ab': 2, 'a': 3}

```

* * *

## 技术 2: Python lambda 函数和 sorted()方法

Python sorted()方法可以与 lambda 函数结合使用，按照 Python 中的值以预定义的顺序对字典进行排序。

Python lambda 函数创建匿名函数，即没有名称的函数。它有助于优化代码。

**语法:Python lambda 函数**

```py
lambda arguments: expression

```

**举例:**

```py
inp_dict = { 'a':3,'ab':2,'abc':1,'abcd':0 }
print("Dictionary: ", inp_dict)
sort_dict= dict(sorted(inp_dict.items(), key=lambda item: item[1])) 
print("Sorted Dictionary by value: ", sort_dict)

```

在上面的代码片段中，我们创建了一个 lambda 函数，并通过按值遍历字典(即 item[1])将字典的值作为参数传递给它。

**输出:**

```py
Dictionary:  {'a': 3, 'ab': 2, 'abc': 1, 'abcd': 0}
Sorted Dictionary by value:  {'abcd': 0, 'abc': 1, 'ab': 2, 'a': 3}

```

* * *

## 技巧 3: Python sorted()方法和 dict.items()

Python sorted()函数可用于在 Python 中按值对字典进行排序，方法是通过 dict.items()将值传递给方法。

`dict.items() method`从字典中获取键/值。

**语法:**

```py
dict.items()

```

**举例:**

```py
inp_dict = { 'a':3,'ab':2,'abc':1,'abcd':0 }
print("Dictionary: ", inp_dict)
sort_dict= dict(sorted((value, key) for (key,value) in inp_dict.items())) 
print("Sorted Dictionary by value: ", sort_dict)

```

在上面的代码片段中，我们将(value，key)对传递给了排序函数，并使用 dict.items()方法获取了 dict 值。在这种情况下，我们将接收排序后的(值，键)对作为输出。

**输出:**

```py
Dictionary:  {'a': 3, 'ab': 2, 'abc': 1, 'abcd': 0}
Sorted Dictionary by value:  {0: 'abcd', 1: 'abc', 2: 'ab', 3: 'a'}

```

* * *

## 结论

因此，在本文中，我们已经了解了 Python 中按值对字典进行排序的不同方式。

* * *

## 参考

*   Python 按值排序字典— JournalDev