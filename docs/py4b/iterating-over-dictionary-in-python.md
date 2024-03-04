# 在 Python 中迭代字典

> 原文：<https://www.pythonforbeginners.com/dictionary/iterating-over-dictionary-in-python>

字典是 python 中最常用的数据结构之一。它包含键值对形式的数据。在使用字典处理数据时，我们可能需要迭代字典中的条目来更改值或读取字典中的值。在本文中，我们将看到 python 中遍历字典的各种方法。

## 使用 for 循环迭代字典

当我们在 python 中使用 for 循环遍历列表或元组时，我们也可以使用 for 循环遍历 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)。

当我们试图使用 for 循环迭代一个字典时，它会隐式调用`__iter__()`方法。`__iter__()`方法返回一个迭代器，在这个迭代器的帮助下，我们可以遍历整个字典。正如我们所知，python 中的字典是使用键索引的，由`__iter__()`方法返回的迭代器遍历 python 字典中的键。

因此，使用 for 循环，我们可以迭代并访问字典的所有键，如下所示。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB","about":"Python Tutorials Website"}
print("The dictionary is:")
print(myDict)
print("The keys in the dictionary are:")
for x in myDict:
    print(x)
```

输出:

```py
The dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB', 'about': 'Python Tutorials Website'}
The keys in the dictionary are:
name
acronym
about
```

在输出中，我们可以看到所有的键都被打印出来了。在 [for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)中，迭代器`x`遍历字典中的所有键，然后打印出来。

使用 for 循环获得字典的键后，我们还可以如下迭代字典中的值。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB","about":"Python Tutorials Website"}
print("The dictionary is:")
print(myDict)
print("The values in the dictionary are:")
for x in myDict:
    print(myDict[x])
```

输出:

```py
The dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB', 'about': 'Python Tutorials Website'}
The values in the dictionary are:
PythonForBeginners
PFB
Python Tutorials Website
```

在上面的代码中，我们简单地获得了一个迭代器，它遍历中的键，然后我们使用语法`dict_name[key_name]`访问与键相关的值，然后输出这些值。

## 迭代字典的关键字

我们可以使用`keys()`方法来迭代字典的键。当在字典上调用时,`keys()` 方法返回字典中的键列表，然后我们可以遍历列表来访问字典中的键，如下所示。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB","about":"Python Tutorials Website"}
print("The dictionary is:")
print(myDict)
print("The keys in the dictionary are:")
keyList=myDict.keys()
for x in keyList:
    print(x)
```

输出:

```py
The dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB', 'about': 'Python Tutorials Website'}
The keys in the dictionary are:
name
acronym
about
```

一旦我们在列表中有了键，我们还可以访问与字典的键相关的值，如下所示。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB","about":"Python Tutorials Website"}
print("The dictionary is:")
print(myDict)
print("The values in the dictionary are:")
keyList=myDict.keys()
for x in keyList:
    print(myDict[x])
```

输出:

```py
The dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB', 'about': 'Python Tutorials Website'}
The values in the dictionary are:
PythonForBeginners
PFB
Python Tutorials Website
```

## 在 python 中迭代字典的值

如果我们只想访问字典中的值，我们可以借助于`values()`方法来实现。在字典上调用 `values()`方法时，会返回字典中所有值的列表。我们可以使用`values()`方法和 for 循环访问字典中的值，如下所示。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB","about":"Python Tutorials Website"}
print("The dictionary is:")
print(myDict)
print("The values in the dictionary are:")
valueList=myDict.values()
for x in valueList:
    print(x)
```

输出:

```py
The dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB', 'about': 'Python Tutorials Website'}
The values in the dictionary are:
PythonForBeginners
PFB
Python Tutorials Website
```

在输出中，我们可以看到字典中的所有值都被逐个打印出来。

## 用 python 迭代字典中的条目

我们可以使用`items()`方法迭代并访问键值对。在字典上调用`items()`方法时，会返回一个元组列表，这些元组具有成对的键和值。每个元组在其`0th`索引上有一个键，并且与该键相关联的值出现在元组的`1st`索引上。我们可以使用如下所示的`items()`方法来访问键值对。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB","about":"Python Tutorials Website"}
print("The dictionary is:")
print(myDict)
print("The items in the dictionary are:")
itemsList=myDict.items()
for x in itemsList:
    print(x)
```

输出:

```py
The dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB', 'about': 'Python Tutorials Website'}
The items in the dictionary are:
('name', 'PythonForBeginners')
('acronym', 'PFB')
('about', 'Python Tutorials Website')
```

除了使用`items()`方法迭代字典中的条目，我们还可以在 for 循环中使用两个迭代器迭代字典的键和值，如下所示。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB","about":"Python Tutorials Website"}
print("The dictionary is:")
print(myDict)
itemList=myDict.items()
print("The key value pairs in the dictionary are:")
for x,y in itemList:
    print(x,end=":")
    print(y)
```

输出:

```py
The dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB', 'about': 'Python Tutorials Website'}
The key value pairs in the dictionary are:
name:PythonForBeginners
acronym:PFB
about:Python Tutorials Website
```

在上面的程序中，第一个迭代器迭代字典中的关键字，第二个迭代器迭代与这些关键字相关联的各个值，这些关键字以元组的形式存在，包含由`items()`方法返回的列表中的关键字值对。

## 结论

在本文中，我们看到了迭代字典中数据的各种方法。我们已经看到了如何使用 for 循环和内置方法如`keys()`、`values()`和`items()`来访问字典的键和值。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，以使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。