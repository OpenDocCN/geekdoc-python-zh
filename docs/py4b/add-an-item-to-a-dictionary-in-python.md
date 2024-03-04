# 用 Python 向字典中添加项目

> 原文：<https://www.pythonforbeginners.com/dictionary/add-an-item-to-a-dictionary-in-python>

python 中的字典是一种以键值对形式存储数据的数据结构。键值对也称为项目。每个字典中的键值对由冒号“:”分隔，字典中的每个条目由逗号“，”分隔。在本文中，我们将研究用 python 向字典添加条目的不同方法。

## 使用下标符号将项目添加到词典中

如果我们有一个名为`myDict`的字典和一个值为`myKey`和`myValue`的键值对，那么我们可以使用语法`myDict [ myKey ] = myValue`将键值对添加到字典中。这可以如下进行。

```py
myDict = {"name": "PythonForBeginners", "acronym": "PFB"}
print("Original Dictionary is:", myDict)
myDict["niche"] = "programming"
print("Modified Dictionary is:", myDict)
```

输出:

```py
Original Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB'}
Modified Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
```

在上面的例子中，我们已经添加了一个新的关键字“利基”,它具有与之相关的值“编程”。

请记住，如果要添加到字典中的条目的键已经存在，则该键的值将被新值覆盖。这可以从下面的例子中看出。

```py
myDict = {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
print("Original Dictionary is:", myDict)
myDict["niche"] = "python programming"
print("Modified Dictionary is:", myDict)
```

输出:

```py
Original Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
Modified Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'python programming'}
```

在上面的例子中，关键字“niche”已经存在于字典中，其值为“programming”。当我们尝试添加以“niche”作为键，以“python 编程”作为关联值的键-值对时，与“niche”关联的值被更新为新值。

## 使用 Python 中的 update()方法将项目添加到字典中

我们可以使用 update()方法向字典中添加一个条目。在字典上调用 update()方法时，它将字典或具有键值对的 iterable 对象作为输入，并将条目添加到字典中。

我们可以给一个新的字典作为 update()方法的输入，并将条目添加到给定的字典中，如下所示。

```py
myDict = {"name": "PythonForBeginners", "acronym": "PFB"}
print("Original Dictionary is:", myDict)
myDict.update({'niche': 'programming'})
print("Modified Dictionary is:", myDict)
```

输出:

```py
Original Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB'}
Modified Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
```

我们可以给出一个包含键-值对的元组列表作为 update()方法的输入，并向给定的字典添加条目，如下所示。

```py
myDict = {"name": "PythonForBeginners", "acronym": "PFB"}
print("Original Dictionary is:", myDict)
items = [("niche", "programming")]
myDict.update(items)
print("Modified Dictionary is:", myDict)
```

输出:

```py
Original Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB'}
Modified Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
```

我们还可以将键值对作为关键字参数传递给 update()方法，以便向字典中添加元素。这里，键将被用作关键字参数，值将被分配为关键字参数的输入，如下所示。

```py
myDict = {"name": "PythonForBeginners", "acronym": "PFB"}
print("Original Dictionary is:", myDict)
myDict.update(niche="programming")
print("Modified Dictionary is:", myDict)
```

输出:

```py
Original Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB'}
Modified Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
```

## 使用 Python 中的**运算符将项目添加到字典中

双星号(**)运算符用于向函数传递可变长度的关键字参数。我们还可以使用**操作符将一个键值对添加到另一个字典中。当我们将**操作符应用于字典时，它会反序列化字典并将其转换为键值对的集合。这个键值对的集合可以再次转换成字典。

要将一个条目添加到字典中，我们将首先创建一个只包含该条目的字典。然后，我们将使用**操作符合并新字典和必须添加条目的字典，如下所示。

```py
myDict = {"name": "PythonForBeginners", "acronym": "PFB"}
print("Original Dictionary is:", myDict)
newDict = {'niche': 'programming'}
myDict = {**myDict, **newDict}
print("Modified Dictionary is:", myDict)
```

输出:

```py
Original Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB'}
Modified Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
```

## 使用 __setitem__()方法将项目添加到词典中

我们还可以使用 __setitem__()方法向字典中添加一个条目。在字典上调用 __setitem__()方法时，它将新的键和值分别作为第一个和第二个参数，并将键-值对添加到字典中，如下所示。

```py
myDict = {"name": "PythonForBeginners", "acronym": "PFB"}
print("Original Dictionary is:", myDict)
myDict.__setitem__('niche', 'programming')
print("Modified Dictionary is:", myDict)
```

输出:

```py
Original Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB'}
Modified Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
```

如果字典中已经存在该键，则与它关联的值将被新值覆盖。这可以从下面的例子中看出。

```py
myDict = {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
print("Original Dictionary is:", myDict)
myDict.__setitem__('niche', 'python programming')
print("Modified Dictionary is:", myDict)
```

输出:

```py
Original Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'programming'}
Modified Dictionary is: {'name': 'PythonForBeginners', 'acronym': 'PFB', 'niche': 'python programming'}
```

## 结论

在本文中，我们看到了用 python 向字典添加条目的不同方法。要阅读更多关于 python 中字典的内容，你可以阅读这篇关于[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章或者这篇关于如何[在 python 中合并两个字典](https://www.pythonforbeginners.com/dictionary/merge-dictionaries-in-python)的文章。