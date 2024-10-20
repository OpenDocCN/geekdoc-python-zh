# 如何更新一个 Python 字典？

> 原文：<https://www.askpython.com/python/dictionary/how-to-update-a-python-dictionary>

嘿，伙计们！在本文中，我们将揭示更新 Python 字典的过程。

* * *

## 更新 Python 字典的步骤入门

[Python Dictionary](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial) 是一种数据结构，它将数据元素保存在一个**键-值对**中，并且基本上作为一个**无序的元素集合**。为了更新相关键的值，Python Dict 内置了一个方法— `dict.update() method`来更新 Python 字典。

dict.update()方法用于更新与输入字典中的键相关联的值。

**语法:**

```py
input_dict.update(dict)

```

该函数不返回任何值，而是用新关联的键值更新同一个输入字典。

**举例:**

```py
dict = {"Python":100,"Java":150}
up_dict = {"Python":500}
print("Dictionary before updation:",dict)
dict.update(up_dict)
print("Dictionary after updation:",dict)

```

**输出:**

```py
Dictionary before updation: {'Python': 100, 'Java': 150}
Dictionary after updation: {'Python': 500, 'Java': 150}

```

* * *

### 用 Iterable 更新 Python 字典

除了更新字典的键值之外，我们还可以使用来自其他可迭代对象的值来追加和更新 Python 字典。

**语法:**

```py
dict.update(iterable)

```

**举例:**

```py
dict = {"Python":100,"Java":150}
print("Dictionary before updation:",dict)
dict.update(C = 35,Fortran = 40)
print("Dictionary after updation:",dict)

```

在上面的例子中，我们已经用传递给 update()函数的值更新了输入 dict。因此，输入 dict 会被追加，并用传递给函数的值进行更新。

**输出:**

```py
Dictionary before updation: {'Python': 100, 'Java': 150}
Dictionary after updation: {'Python': 100, 'Java': 150, 'C': 35, 'Fortran': 40}

```

* * *

## 更新嵌套 Python 字典

嵌套字典是字典中的字典。Python 嵌套字典可以使用以下语法使用相应的键值进行更新:

**语法:**

```py
dict[outer-key][inner-key]='new-value'

```

**举例:**

```py
dict = { 'stud1_info':{'name':'Safa','Roll-num':25},'stud2_info':{'name':'Ayush','Roll-num':24}}
print("Dictionary before updation:",dict)
dict['stud2_info']['Roll-num']=78
dict['stud1_info']['name']='Riya'
print("Dictionary after updation:",dict)

```

在上面的示例中，我们已经将内部键的值:“Roll-num”外部键的值:“stud2_info”更新为 78，并将内部键的值:“name”外部键的值:“stud1_info”更新为“Riya”。

**输出:**

```py
Dictionary before updation: {'stud1_info': {'name': 'Safa', 'Roll-num': 25}, 'stud2_info': {'name': 'Ayush', 'Roll-num': 24}}
Dictionary after updation: {'stud1_info': {'name': 'Riya', 'Roll-num': 25}, 'stud2_info': {'name': 'Ayush', 'Roll-num': 78}}

```

* * *

## 结论

因此，在本文中，我们已经了解了更新 Python 字典和嵌套字典的值的方法。

我强烈建议读者阅读 [Python 字典教程](https://www.askpython.com/python/dictionary)以深入理解字典概念。

* * *

## 参考

*   Python 字典— JournalDev