# 检查 Python 字典中是否存在键的 4 个简单技巧

> 原文：<https://www.askpython.com/python/dictionary/check-if-key-exists-in-a-python-dictionary>

在本文中，我们将重点讨论检查 Python 字典中是否存在键的 4 种方法。**Python 字典**基本上是一种数据结构，其中数据项存储在键值对中。

* * *

## 技巧 1:“in”操作符检查 Python 字典中是否存在键

Python `in operator`和`if statement`可以用来检查特定的键是否存在于输入 Python 字典中。

Python in operator 主要检查特定的元素或值是否包含在特定的序列中，如列表、元组、字典等。

**语法:**

```py
for/if value in iterable:

```

**举例:**

```py
inp_dict = {'Python': "A", 'Java':"B", 'Ruby':"C", 'Kotlin':"D"} 

search_key = 'Ruby'

if search_key in inp_dict: 
		print("The key is present.\n") 

else: 
		print("The key does not exist in the dictionary.") 

```

在上面的例子中，我们使用了 if 语句和 Python `in`操作符来检查关键字‘Ruby’是否出现在字典中。

**输出:**

```py
The key is present.

```

* * *

## 技巧 2: Python keys()方法

Python 内置的`keys() method`可以用来检查现有字典中是否存在一个键。

**语法:**

```py
dict.keys()

```

keys()方法不接受**参数**并返回一个**对象，该对象表示特定输入字典中存在的所有键**的列表。

因此，为了检查 dict 中是否存在特定的键，我们使用`Python if statement`和 keys()方法来比较 search_key 和 keys()方法返回的键列表。如果键存在，它将跟随 If 部分的语句，否则它将跳转到`else` 部分的语句。

**举例:**

```py
inp_dict = {'Python': "A", 'Java':"B", 'Ruby':"C", 'Kotlin':"D"} 

search_key = 'Ruby'

if search_key in inp_dict.keys(): 
		print("The key is present.\n") 

else: 
		print("The key does not exist in the dictionary.") 

```

**输出:**

```py
The key is present.

```

**例 2:**

```py
inp_dict = {'Python': "A", 'Java':"B", 'Ruby':"C", 'Kotlin':"D"} 

search_key = 'Cpp'

if search_key in inp_dict.keys(): 
		print("The key is present.\n") 

else: 
		print("The key does not exist in the dictionary.") 

```

**输出:**

```py
The key does not exist in the dictionary.

```

* * *

## 技术 3: get()方法检查 Python 字典中是否存在键

Python `get() method`可以用来检查特定的键是否出现在字典的键-值对中。

如果键碰巧出现在字典中，get()方法实际上**返回与键**相关联的值，否则它返回“ **None** ”。

**语法:**

```py
dict.get(key, default=None)

```

我们把要搜索的关键字作为参数传递给 get()方法，如果 get()函数没有返回`None`，也就是说，如果关键字存在于 dict 中，我们就打印它。

**例 1:**

```py
inp_dict = {'Python': "A", 'Java':"B", 'Ruby':"C", 'Kotlin':"D"} 

if inp_dict.get('Python')!=None: 
		print("The key is present.\n") 

else: 
		print("The key does not exist in the dictionary.") 

```

**输出:**

```py
The key is present.

```

* * *

## 技巧 4: Python has_key()方法

**注意:****Python 3 版及以上已经省略了 has_keys()方法。**

Python `has_key() method`检查特定的键在 dict 中是否可用，并返回 True，否则返回 false。

**语法:**

```py
dict.has_keys()

```

**举例:**

```py
inp_dict = {'Python': "A", 'Java':"B", 'Ruby':"C", 'Kotlin':"D"} 

search_key = 'Kotlin'

if inp_dict.has_key(search_key): 
		print("The key is present.\n") 

else: 
		print("The key does not exist in the dictionary.") 

```

* * *

## 结论

因此，在本文中，我们揭示并理解了检查 Python 字典中是否存在 key 的各种技术。

我推荐所有的读者通过下面的帖子来更详细地了解 Python 字典。

*   [Python 字典](https://www.askpython.com/python/dictionary)

* * *

## 参考

*   Python 词典
*   Python if 语句