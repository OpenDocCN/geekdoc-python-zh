# Python Dict items()方法的 5 个例子

> 原文：<https://www.askpython.com/python/dictionary/python-dict-items-method>

嘿，伙计们！在本文中，我们将关注于 **Python Dict items()方法**来提取和表示字典值。

* * *

## Python Dict items()方法是如何工作的？

[Python 字典](https://www.askpython.com/python/dictionary)是以**键值对**格式存储数据元素的数据结构。此外，每个唯一键都与唯一/冗余值相关联。

为了迭代和显示字典元素，Python 的 dict items()方法提供了这个功能。

`dict.items() method`将字典所占用的数据元素显示为键-值对的[列表](https://www.askpython.com/python/list/python-list)。

**语法:**

```py
dict.items()

```

dict.items()方法**不接受任何参数**并且**返回一个包含键值的列表作为字典的元组对**。

* * *

### 使用 Python 字典项()返回(键，值)对的列表

在本例中，我们创建了一个字典，并使用 dict.items()方法返回输入字典中所有键值对的列表。

```py
inp_dict = { 'a':3,'ab':2,'abc':1,'abcd':0 }
print("Dictionary: ", inp_dict.items())

```

**输出:**

```py
Dictionary:  dict_items([('a', 3), ('ab', 2), ('abc', 1), ('abcd', 0)])

```

* * *

### 使用 Python 字典项返回字符串项

在下面的示例中，输入字典包含字符串值形式的键值对。

因此，可以说`dict.items() method` **不受键或值**类型的影响，并返回 dict 中作为元组对出现的所有元素的列表。

```py
lang_dict = { 
    'A':'Python',
    'B':'C++',
    'C':'Kotlin'}
print("Dictionary: ", lang_dict.items())

```

**输出:**

```py
Dictionary:  dict_items([('A', 'Python'), ('B', 'C++'), ('C', 'Kotlin')])

```

* * *

### 使用字典项()返回空字典

dict.items()方法不会为空字典抛出任何类型的错误或异常。因此，**如果字典为空，dict.items()方法返回一个空列表**。

```py
empty_dict = {}
print("Dictionary: ", empty_dict.items())

```

**输出:**

```py
Dictionary:  dict_items([])

```

* * *

### 更新值后打印出字典项目

在下面的例子中，我们创建了一个输入字典，并使用 items()函数显示字典值。

此外，我们已经使用 items()方法更新了字典键-值对并再次显示了字典。因此，可以说 dict.items()方法很好地配合了 dictionary 的更新，并以列表格式显示更新后的值。

如果 dict 得到更新，这些变化会自动反映在使用 dict.input()方法显示的列表中。

```py
lang_dict = { 
    'A':'Python',
    'B':'C++',
    'C':'Kotlin'}
print("Dictionary before updation: ", lang_dict.items())
lang_dict['B']='Ruby'
print("Dictionary after updation: ", lang_dict.items())

```

**输出:**

```py
Dictionary before updation:  dict_items([('A', 'Python'), ('B', 'C++'), ('C', 'Kotlin')])
Dictionary after updation:  dict_items([('A', 'Python'), ('B', 'Ruby'), ('C', 'Kotlin')])

```

* * *

### 不使用 Dict items()方法打印出字典元素

我们可以通过调用如下所示的输入字典对象，直接以键值形式显示字典的元素:

```py
input_dict={}
print(input_dict)

```

上述代码行将返回一个空括号“{ 0 }”。

在下面的例子中，我们使用了 dictionary 对象和 dict.items()方法来显示 dict 键值对。

```py
lang_dict = { 
    'A':'Python',
    'B':'C++',
    'C':'Kotlin'}
print("Dictionary: ", lang_dict)
print("Dictionary using items() method:", lang_dict.items())

```

**输出:**

```py
Dictionary:  {'A': 'Python', 'B': 'C++', 'C': 'Kotlin'}
Dictionary using items() method: dict_items([('A', 'Python'), ('B', 'C++'), ('C', 'Kotlin')])

```

* * *

## 摘要

*   Python dict.items()方法用于以元组对列表的形式显示 dict 的数据元素。
*   如果 dict 为空，则 dict.items()方法返回一个空列表。
*   当我们更新字典时，dict.items()方法会将这些更改很好地合并到输出列表中。

* * *

## 结论

因此，在本文中，我们已经理解了 **Python Dictionary items()方法**在各种例子中的工作。

* * *

## 参考

*   [Python Dict items()方法—文档](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)