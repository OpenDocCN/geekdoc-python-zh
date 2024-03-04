# 从字典中的值获取键

> 原文：<https://www.pythonforbeginners.com/dictionary/get-key-from-value-in-dictionary>

在 python 中，我们可以通过简单地使用语法 dict_name[key_name]来使用键获取字典中的值。但是当我们有值时，没有任何方法可以提取与值相关的键。在这篇文章中，我们将看到在字典中从给定值中获取密钥的方法。

## 通过在字典中搜索项目从值中获取键

这是获取值的键的最简单的方法。在这个方法中，我们将检查每个键-值对，以找到与当前值相关联的键。对于这个任务，我们将为一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)使用 items()方法。items()方法返回包含键值对的元组列表。我们将搜索每个元组，以找到与我们的值相关联的键，如下所示。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB"}
print("Dictionary is:")
print(myDict)
dict_items=myDict.items()
print("Given value is:")
myValue="PFB"
print(myValue)
print("Associated Key is:")
for key,value in dict_items:
    if value==myValue:
        print(key)
```

输出:

```py
Dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB'}
Given value is:
PFB
Associated Key is:
acronym
```

在上面的程序中，我们使用 myDict.items()在 myDict 中创建了一个条目列表，然后我们检查列表中的每个条目，以找到我们的值的键。

## 使用 python 列表从值中获取键

我们可以创建一个单独的键和值的列表，然后使用 index()方法从给定的值中检索键。对于此任务，我们将首先使用 keys()方法创建字典中的键列表，然后使用 values()方法创建字典中的值列表。现在我们将使用 index()方法从值列表中获取给定值的索引。因为我们知道关键字列表与值列表中的值具有相同的关键字顺序，所以值列表中的值的索引将与关键字列表中的关联关键字的索引相同。因此，在值列表中找到值的索引后，我们可以在相同索引的键列表中找到键。这可以如下进行。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB"}
print("Dictionary is:")
print(myDict)
dict_keys=list(myDict.keys())
dict_values=list(myDict.values())
print("Given value is:")
myValue="PFB"
print(myValue)
val_index=dict_values.index(myValue)
print("Associated key is:")
myKey=dict_keys[val_index]
print(myKey)
```

输出:

```py
Dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB'}
Given value is:
PFB
Associated key is:
acronym
```

## 使用列表理解从值中获取键

不使用 index()方法，我们可以使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 来获取与给定值相关联的键。为了找到这个键，我们将使用如下的列表理解来创建一个键的列表，这些键的关联值等于给定值。

```py
 myDict={"name":"PythonForBeginners","acronym":"PFB"}
print("Dictionary is:")
print(myDict)
dict_items=myDict.items()
print("Given value is:")
myValue="PFB"
print(myValue)
print("Associated key is:")
myKey=[key for key,value in dict_items if value==myValue]
print(myKey)
```

输出:

```py
Dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB'}
Given value is:
PFB
Associated key is:
['acronym']
```

## 结论

在本文中，我们看到了使用 list comprehension、items()方法和 list index()方法从 python 字典中获取键的三种方法。请继续关注更多内容丰富的文章。