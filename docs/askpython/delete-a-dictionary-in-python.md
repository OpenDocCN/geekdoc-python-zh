# 在 Python 中删除字典的方法

> 原文：<https://www.askpython.com/python/dictionary/delete-a-dictionary-in-python>

想删除 Python 中的字典？Python 字典是 Python 中使用的一种数据结构，它接受键值对形式的元素。在本文中，我们将了解从 Python 字典中删除键值对的不同**方法。**

* * *

## Python 中删除字典的 dict.clear()

Python 内置了`clear() method`来删除 Python 中的字典。clear()方法删除 dict 中存在的所有键值对，并返回一个空的 dict。

**语法:**

```py
dict.clear()

```

**举例:**

```py
inp_dict = {"A":"Python","B":"Java","C":"Fortan","D":"Javascript"}
print("Elements of the dict before performing the deletion operation:\n")
print(str(inp_dict))
inp_dict.clear()
print("\nElements of the dict after performing the deletion operation:")
print(str(inp_dict))

```

**输出:**

```py
Elements of the dict before performing the deletion operation:

{'B': 'Java', 'D': 'Javascript', 'C': 'Fortan', 'A': 'Python'}

Elements of the dict after performing the deletion operation:
{}

```

* * *

## 从 Python 字典中删除键值对的技术

以下技术可用于从字典中删除键值对:

*   **Python pop()函数**
*   **Python del 关键字**
*   **内置 Python popitem()方法**
*   **字典理解连同 Python items()方法**

* * *

### 1.使用 pop()方法

Python pop()方法可以用来从字典中删除一个键和一个与之关联的值，即一个键-值对。

**语法:**

```py
dict.pop(key)

```

`pop()`方法基本上接受一个要从字典中删除的**键**。它**从 dict 中删除关键字以及与关键字**相关的值，并返回更新后的 dict。

**举例:**

```py
inp_dict = {"A":"Python","B":"Java","C":"Fortan","D":"Javascript"}
print("Elements of the dict before performing the deletion operation:\n")

print(str(inp_dict))

pop_item = inp_dict.pop("A")

print("\nThe deleted element:")
print(str(pop_item))

print("\nElements of the dict after performing the deletion operation:\n")
print(str(inp_dict))

```

在上面的代码片段中，我们将键“A”作为参数传递给了 pop()方法。因此，它删除了与“A”相关联的键值对。

**输出:**

```py
Elements of the dict before performing the deletion operation:

{'A': 'Python', 'B': 'Java', 'C': 'Fortan', 'D': 'Javascript'}

The deleted element:
Python

Elements of the dict after performing the deletion operation:

{'B': 'Java', 'C': 'Fortan', 'D': 'Javascript'}
​

```

* * *

### 2.Python del 关键字

Python `del`实际上是一个**关键字**，基本用于**删除对象**。众所周知，Python 将所有东西都视为一个对象，这就是为什么我们可以很容易地使用 del 通过逐个删除元素来删除 Python 中的字典。

Python del 关键字也可以用来从输入字典值中删除一个键值对。

**语法:**

```py
del dict[key]

```

**例 1:**

```py
inp_dict = {"A":"Python","B":"Java","C":"Fortan","D":"Javascript"}
print("Elements of the dict before performing the deletion operation:\n")
print(str(inp_dict))

del inp_dict["D"]

print("\nElements of the dict after performing the deletion operation:\n")
print(str(inp_dict))

```

**输出:**

```py
Elements of the dict before performing the deletion operation:

{'A': 'Python', 'B': 'Java', 'C': 'Fortan', 'D': 'Javascript'}

Elements of the dict after performing the deletion operation:

{'A': 'Python', 'B': 'Java', 'C': 'Fortan'}

```

**例 2:** **从嵌套字典中删除键值对**

**语法:**

我们可以从**嵌套的 Python 字典**中删除键值对。del 关键字的语法如下:

```py
del dict[outer-dict-key-name][key-name-associated-with-the-value]

```

**举例:**

```py
inp_dict = {"Python":{"A":"Set","B":"Dict","C":"Tuple","D":"List"},
             "1":"Java","2":"Fortan"}
print("Elements of the dict before performing the deletion operation:\n")
print(str(inp_dict))

del inp_dict["Python"]["C"]

print("\nElements of the dict after performing the deletion operation:\n")
print(str(inp_dict))

```

因此，在这里，我们删除了与外部键“`Python`”相关联的键-值对“`C:Tuple`”。

**输出:**

```py
Elements of the dict before performing the deletion operation:

{'Python': {'A': 'Set', 'B': 'Dict', 'C': 'Tuple', 'D': 'List'}, '1': 'Java', '2': 'Fortan'}

Elements of the dict after performing the deletion operation:

{'Python': {'A': 'Set', 'B': 'Dict', 'D': 'List'}, '1': 'Java', '2': 'Fortan'}

```

* * *

### 3.Python popitem()方法

Python `popitem()`函数可以用来从 Python 字典中删除随机或任意的键值对。popitem()函数不接受**任何参数**，并返回从字典中删除的键值对。

**语法:**

```py
dict.popitem()

```

**举例:**

```py
inp_dict = {"A":"Python","B":"Java","C":"Fortan","D":"Javascript"}
print("Elements of the dict before performing the deletion operation:\n")
print(str(inp_dict))
pop_item = inp_dict.popitem()
print("\nThe deleted element:")
print(str(pop_item))
print("\nElements of the dict after performing the deletion operation:\n")
print(str(inp_dict))

```

**输出:**

```py
Elements of the dict before performing the deletion operation:

{'A': 'Python', 'B': 'Java', 'C': 'Fortan', 'D': 'Javascript'}

The deleted element:
('D', 'Javascript')

Elements of the dict after performing the deletion operation:

{'A': 'Python', 'B': 'Java', 'C': 'Fortan'}

```

* * *

### 4.Python 字典理解和 items()方法

Python items()方法和 Dict Comprehension 一起可以用来删除 Python 中的字典。

Python `items() method`基本上不带参数，返回一个包含特定字典中所有键值对列表的对象。

**语法:**

```py
dict.items()

```

Python `Dict Comprehension`可用于通过接受来自特定 iterable 的键值对来创建字典。

**语法:**

```py
{key: value for key, value in iterable}

```

在从字典中删除键-值对的上下文中，items()方法可用于将键-值对的列表作为 iterable 提供给字典理解。

`if statement`用于遇到前面提到的键值。如果遇到上述键值，它将返回一个新的 dict，其中包含除了与要删除的键相关联的键值对之外的所有键值对。

**举例:**

```py
inp_dict = {"A":"Set","B":"Dict","C":"Tuple","D":"List",
             "1":"Java","2":"Fortan"}
print("Elements of the dict before performing the deletion operation:\n")
print(str(inp_dict))

res_dict = {key:value for key, value in inp_dict.items() if key != "1"} 

print("\nElements of the dict after performing the deletion operation:\n")
print(str(res_dict))

```

**输出:**

```py
Elements of the dict before performing the deletion operation:

{'A': 'Set', 'B': 'Dict', 'C': 'Tuple', 'D': 'List', '1': 'Java', '2': 'Fortan'}

Elements of the dict after performing the deletion operation:

{'A': 'Set', 'B': 'Dict', 'C': 'Tuple', 'D': 'List', '2': 'Fortan'}

```

* * *

## 通过在迭代时指定元素来删除 Python 中的字典

在处理 Python 字典时，我们可能会遇到这样的情况:我们希望在字典的迭代过程中删除一个键值对。

为此，我们可以创建输入字典的**列表，并使用`for loop`遍历字典的[列表](https://www.askpython.com/python/list/python-list)。**

**语法:**

```py
list(dict)

```

最后，通过使用一个`if statement`，我们将检查循环是否遇到要删除的键。一旦遇到键，就可以使用 **del 关键字**删除键-值对。

**举例:**

```py
inp_dict = {"A":"Set","B":"Dict","C":"Tuple","D":"List",
             "1":"Java","2":"Fortan"}
print("Elements of the dict before performing the deletion operation:\n")
print(str(inp_dict))

for KEY in list(inp_dict): 
    if KEY =="B":  
        del inp_dict[KEY] 

print("\nElements of the dict after performing the deletion operation:\n")
print(str(inp_dict))

```

**输出:**

```py
Elements of the dict before performing the deletion operation:

{'A': 'Set', 'B': 'Dict', 'C': 'Tuple', 'D': 'List', '1': 'Java', '2': 'Fortan'}

Elements of the dict after performing the deletion operation:

{'A': 'Set', 'C': 'Tuple', 'D': 'List', '1': 'Java', '2': 'Fortan'}

```

* * *

## 结论

因此，我们揭示了从 Python 字典中删除键值对的不同技术。

* * *

## 参考

*   Python 字典 JournalDev
*   [Python 字典-官方文档](https://docs.python.org/3/tutorial/datastructures.html)