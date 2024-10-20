# Python 添加到字典

> 原文：<https://www.askpython.com/python/dictionary/python-add-to-dictionary>

[Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)基本上包含了**键值**对形式的元素。

这是一个*无序的项目集合*。

**创建字典:**

```py
cities = {"Pune": "Maharashtra", "Ahemdabad": "Gujarat"}
print(cities)
#type(cities)

```

**输出:**

```py
{'Pune': 'Maharashtra', 'Ahemdabad': 'Gujarat'}
```

* * *

## 如何在 Python 中添加字典

*   **通过使用 update()方法**
*   **通过使用 _setitem_()方法**
*   **通过使用下标符号**
*   **通过使用“*”运算符**

* * *

### 1.通过使用 update()方法

update()方法使用户能够向 dict 添加多个键值对。

```py
info = {'name':'Safa', 'age':21} 
print("Current Dict is: ", info) 

info.update({'Address':'Pune'}) 
print("Updated Information is: ", info) 

```

**输出:**

```py
Current Dict is:  {'name': 'Safa', 'age': 21}
Updated Information is:  {'name': 'Safa', 'age': 21, 'Address': 'Pune'}
```

* * *

### 2.通过使用 _setitem_()方法

Python Dictionary 的 _setitem_()方法用于向 dict 添加一个键值对。

```py
info = {'name':'Safa', 'age':'21'} 

info.__setitem__('Address', 'Pune') 
print(info) 

```

**输出:**

```py
{'age': '21', 'name': 'Safa', 'Address': 'Pune'}
```

* * *

### 3.通过使用下标符号

下标符号有助于向字典添加一个新的键值对。如果该键不存在，则会创建一个新键，并为其分配上述值。

**语法:**

```py
dict[new-key]=[new-value]
```

```py
info = {'name':'Safa', 'age':'21'} 

info['Address'] = 'Pune'

print(info) 

```

O **输出:T1**

```py
{'name': 'Safa', 'age': '21', 'Address': 'Pune'}
```

* * *

### 4.通过使用“**”运算符

“**”操作符基本上将键值对添加到新字典中，并将其与旧字典合并。

```py
info = {'name':'Safa', 'age':'21'} #old dict

#adding item to the new dict(result) and merging with old dict(info)  
result = {**info, **{'Address': 'Pune'}}  

print(result) 

```

**输出:**

```py
{'name': 'Safa', 'age': '21', 'Address': 'Pune'}
```

* * *

## 向嵌套 Python 字典添加键

```py
info = {'TEST' : {'name' : 'Safa', 'age' : 21}} 

print("The Input dictionary: " + str(info)) 

info['TEST']['Address'] = 'Pune'

print("Dictionary after adding key to nested dict: " + str(info)) 

```

**输出:**

```py
The Input dictionary: {'TEST': {'name': 'Safa', 'age': 21}}
Dictionary after adding key to nested dict: {'TEST': {'name': 'Safa', 'age': 21, 'Address': 'Pune'}}
```

* * *

## 向 Python 字典添加多个键值对

```py
info = {'TEST' : {'name' : 'Safa', 'age' : 21}} 

info.update([ ('Address', 'Pune') , ('zip_code',411027 )])
print(info)

```

**输出:**

```py
{'TEST': {'name': 'Safa', 'age': 21}, 'Address': 'Pune', 'zip_code': 411027}
```

* * *

## 将一部词典加入另一部词典

```py
info = {'TEST' : {'name' : 'Safa', 'age' : 21}} 

info1 = { 'SET' : {'number' : 452756345, 'Address' : 'Pune'}}
#Adding elements of info1 to info
info.update(info1)
print(info)          

```

**输出:**

```py
{'TEST': {'name': 'Safa', 'age': 21}, 'SET': {'number': 452756345, 'Address': 'Pune'}}
```

* * *

## 结论

因此，在本文中，我们已经理解并实现了向 Python 字典添加键值对的可能方法。

* * *

## 参考

*   Python 添加到字典
*   [Python 字典文档](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)