# 检查 Python 中的字典中是否存在某个键

> 原文：<https://www.pythonforbeginners.com/basics/check-if-a-key-exists-in-a-dictionary-in-python>

我们使用 python 字典来存储键值对。有时，我们需要检查字典中是否存在某个键。在本 python 教程中，我们将通过工作示例讨论不同的方法来检查一个键是否存在于 python 中的给定字典中。

## 使用 get()方法检查字典中是否存在一个键

### get()方法

当在 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)上调用`get()`方法时，该方法将一个键作为输入参数，并返回字典中与该键相关联的值。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "name"
print("The input key is:", key)
value = myDict.get(key)
print("The associated value is:", value)
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input key is: name
The associated value is: Python For Beginners
```

如果字典中没有这个键， `get()`方法返回如下所示的`None`。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "Aditya"
print("The input key is:", key)
value = myDict.get(key)
print("The associated value is:", value)
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input key is: Aditya
The associated value is: None
```

在这里，您可以看到关键字'`Aditya`'在字典中不存在。因此，`get()`方法返回值`None`。

为了使用 `get()`方法检查给定的特定键是否存在于字典中，我们将调用字典上的`get()` 方法，并将该键作为输入参数传递。如果`get()`方法返回`None`，我们会说这个键在字典中不存在。否则，我们会说关键字存在于字典中。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "Aditya"
print("The input key is:", key)
value = myDict.get(key)
if value is None:
    print("The key doesn't exist in the dictionary.")
else:
    print("The key exists in the dictionary.")
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input key is: Aditya
The key doesn't exist in the dictionary.
```

上面的方法只有在字典中没有键具有与之相关联的值`None`时才有效。如果一个键的值是`None`,程序会给出错误的结果。例如，看看下面的源代码。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog","Aditya":None}
print("The dictionary is:")
print(myDict)
key = "Aditya"
print("The input key is:", key)
value = myDict.get(key)
if value is None:
    print("The key doesn't exist in the dictionary.")
else:
    print("The key exists in the dictionary.")
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'Aditya': None}
The input key is: Aditya
The key doesn't exist in the dictionary.
```

在这里，您可以观察到关键字'`Aditya`'出现在字典中。但是，程序给出了一个错误的结果，说字典中没有这个键。这是因为关键字'`Aditya`'在字典中的关联值是`None`。

### 使用 get()方法检查字典中是否存在多个键

给定一个键列表，如果我们需要检查一个字典中是否存在多个键，我们将遍历这个键列表。迭代时，我们将调用字典上的 `get()` 方法，将当前键作为其输入参数。

在 for 循环中，如果`get()` 方法返回`None`，我们会说这个键在字典中不存在。否则，我们会说关键字存在于字典中。

您可以在下面的示例中观察这个过程。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = ["Aditya","name","url"]
print("The input keys are:", keys)
for key in keys:
    value = myDict.get(key)
    if value is None:
        print("The key '{}' doesn't exist in the dictionary.".format(key))
    else:
        print("The key '{}' exists in the dictionary.".format(key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input keys are: ['Aditya', 'name', 'url']
The key 'Aditya' doesn't exist in the dictionary.
The key 'name' exists in the dictionary.
The key 'url' exists in the dictionary.
```

### 使用 get()方法检查字典列表中是否存在一个键

为了检查字典列表中是否存在某个键，我们将使用 for 循环来遍历字典列表。迭代时，我们将调用每个字典上的`get()`方法，将键作为输入参数。

如果任何一个字典返回一个不同于`None`的值，我们就说这个键存在于字典列表中。一旦找到密钥，我们将使用 break 语句来结束循环。

如果`get()`方法为一个字典返回`None`，我们将使用 continue 语句移动到下一个字典。

如果`get()`方法为所有的字典返回 None，我们会说这个键不在字典列表中。您可以在下面的示例中观察到这一点。

```py
listOfDicts = [{1: 1, 2: 4, 3: 9},
               {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB",
                "type": "python blog"},
               {"person": "Aditya", "Country": "India"}
               ]
key = "Aditya"
print("The key is:", key)
keyFound = False
for dictionary in listOfDicts:
    value = dictionary.get(key)
    if value is not None:
        print("The key '{}' is present in the list of dictionaries.".format(key))
        keyFound = True
        break
    else:
        continue
if not keyFound:
    print("The key '{}' is not present in the list of dictionaries.".format(key))
```

输出:

```py
The key is: Aditya
The key 'Aditya' is not present in the list of dictionaries.
```

## 使用 for 循环检查字典中是否存在某个键

使用`get()`方法具有很高的时间复杂度，因为我们需要检索与每个键相关联的值。我们可以通过直接检查字典中是否存在键或者不使用 for 循环来避免这种情况，如下所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "url"
print("The input key is:", key)
keyFound = False
for keys in myDict:
    if keys == key:
        print("The key '{}' is present in the dictionary.".format(key))
        keyFound = True
        break
    else:
        continue
if not keyFound:
    print("The key '{}' is not present in the dictionary.".format(key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input key is: url
The key 'url' is present in the dictionary.
```

这里，字典作为包含键的容器对象工作。我们遍历迭代器，逐个检查字典中是否存在所需的键。

### 使用 for 循环检查字典中是否存在多个键

为了检查字典中是否存在多个键，我们将遍历键列表。在迭代输入键时，我们将迭代字典来检查每个输入键是否都存在于字典中。

如果在字典中找到一个键，我们就说这个键存在于字典中。否则不会。

一旦在字典中找到一个键，我们将使用 break 语句从内部 for 循环中出来。

在检查字典中是否存在某个键之后，我们将移动到下一个键，并重复相同的过程。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
input_keys = ["Aditya", "name", "url"]
print("The input keys are:", input_keys)
for key in input_keys:
    keyFound = False
    for keys in myDict:
        if keys == key:
            print("The key '{}' is present in the dictionary.".format(key))
            keyFound = True
            break
        else:
            continue
    if not keyFound:
        print("The key '{}' is not present in the dictionary.".format(key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input keys are: ['Aditya', 'name', 'url']
The key 'Aditya' is not present in the dictionary.
The key 'name' is present in the dictionary.
The key 'url' is present in the dictionary.
```

### 使用 for 循环检查字典列表中是否存在某个键

为了检查一个键是否存在于字典列表中，我们将首先使用循环的[遍历字典列表中的每个字典。在迭代时，我们将使用另一个 for 循环来检查输入键是否存在于字典中。一旦我们发现该键存在于字典中，我们将打印出该键存在于字典列表中。之后，我们将使用 break 语句跳出 for 循环。](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)

如果字典列表中的字典都没有输入键，我们将打印出字典中不存在该键。您可以在下面的示例中观察到这一点。

```py
listOfDicts = [{1: 1, 2: 4, 3: 9},
               {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB",
                "type": "python blog"},
               {"person": "Aditya", "Country": "India"}
               ]
key = "name"
print("The key is:", key)
keyFound = False
for dictionary in listOfDicts:
    for keys in dictionary:
        if keys == key:
            print("The key '{}' is present in the list of dictionaries.".format(key))
            keyFound = True
            break
        else:
            continue
if not keyFound:
    print("The key '{}' is not present in the list of dictionaries.".format(key))
```

输出:

```py
The key is: name
The key 'name' is present in the list of dictionaries.
```

## 使用成员运算符检查字典中是否存在某个键

成员操作符`(in operator)` 用于检查容器对象中是否存在元素。

要检查字典中是否存在某个键，我们可以使用如下所示的 in 操作符。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "Aditya"
print("The input key is:", key)
if key in myDict:
    print("The key '{}' is present in the dictionary.".format(key))
else:
    print("The key '{}' is not present in the dictionary.".format(key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input key is: Aditya
The key 'Aditya' is not present in the dictionary.
```

### 使用成员运算符检查字典中是否存在多个键

为了检查字典中是否存在多个键，我们将使用 for 循环和成员操作符。我们将使用 for 循环遍历键列表。迭代时，我们将使用成员操作符检查字典中是否存在一个键。如果我们发现一个键存在于字典中，我们会说这个键存在于字典中。否则不会。

您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
input_keys = ["Aditya", "name", "url"]
print("The input keys are:", input_keys)
for key in input_keys:
    if key in myDict:
        print("The key '{}' is present in the dictionary.".format(key))
    else:
        print("The key '{}' is not present in the dictionary.".format(key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input keys are: ['Aditya', 'name', 'url']
The key 'Aditya' is not present in the dictionary.
The key 'name' is present in the dictionary.
The key 'url' is present in the dictionary.
```

### 使用成员操作符检查一个键是否存在于 Python 中的字典列表中

给定一个字典列表，如果我们需要检查一个键是否存在于字典中，我们将使用下面的过程。

我们将使用 for 循环遍历字典列表。迭代时，我们将使用成员操作符检查每个字典中是否存在该键。如果我们在现在的字典里找不到答案，我们就要去下一本字典。一旦我们发现这个键存在于字典中，我们将打印出这个键存在于字典中。之后，我们将使用 break 语句跳出循环。

如果我们在任何一本词典中都找不到关键字，我们将在最后打印出来。您可以在下面的示例中观察到这一点。

```py
listOfDicts = [{1: 1, 2: 4, 3: 9},
               {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB",
                "type": "python blog"},
               {"person": "Aditya", "Country": "India"}
               ]
key = "name"
print("The key is:", key)
keyFound = False
for dictionary in listOfDicts:
    if key in dictionary:
        print("The key '{}' is present in the list of dictionaries.".format(key))
        keyFound = True
        break
    else:
        continue
if not keyFound:
    print("The key '{}' is not present in the list of dictionaries.".format(key))
```

输出:

```py
The key is: name
The key 'name' is present in the list of dictionaries.
```

## 使用 keys()方法检查字典中是否存在某个键

### keys()方法

在字典上调用`keys()`方法时，它返回包含字典中所有键的`dict_keys`对象的副本，如下面的 python 脚本所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys=myDict.keys()
print("The keys are:", keys)
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The keys are: dict_keys(['name', 'url', 'acronym', 'type']) 
```

为了使用`keys()`方法检查特定的键是否存在于字典中，我们将首先获得`dict_keys`对象。之后，我们将使用 for 循环遍历`dict_keys`对象。迭代时，我们将检查当前键是否是我们正在搜索的键。如果是，我们会说这个键在字典里。否则不会。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.keys()
print("The keys are:", keys)
input_key = "url"
keyFound = False
for key in keys:
    if key == input_key:
        print("The key '{}' exists in the dictionary.".format(input_key))
        keyFound = True
        break
    else:
        continue
if not keyFound:
    print("The key '{}' does not exist in the dictionary.".format(input_key)) 
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The keys are: dict_keys(['name', 'url', 'acronym', 'type'])
The key 'url' exists in the dictionary.
```

除了使用 for 循环，我们还可以使用 membership 操作符来检查字典中是否存在该键，如下所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.keys()
print("The keys are:", keys)
input_key = "url"
if input_key in keys:
    print("The key '{}' exists in the dictionary.".format(input_key))
else:
    print("The key '{}' doesn't exist in the dictionary.".format(input_key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The keys are: dict_keys(['name', 'url', 'acronym', 'type'])
The key 'url' exists in the dictionary.
```

### 使用 Keys()方法检查字典中是否存在多个键

为了检查字典中是否存在多个键，我们将遍历输入键的列表。对于每个键，我们将使用成员操作符来检查该键是否存在于由`keys()`方法返回的`dict_key`对象中。如果是，我们将打印出该键存在于字典中。否则，我们将打印出该键不存在。最后，我们将转到下一个键。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.keys()
print("The keys are:", keys)
input_keys = ["Aditya", "name", "url"]
print("The input keys are:", input_keys)
for input_key in input_keys:
    if input_key in keys:
        print("The key '{}' exists in the dictionary.".format(input_key))
    else:
        print("The key '{}' doesn't exist in the dictionary.".format(input_key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The keys are: dict_keys(['name', 'url', 'acronym', 'type'])
The input keys are: ['Aditya', 'name', 'url']
The key 'Aditya' doesn't exist in the dictionary.
The key 'name' exists in the dictionary.
The key 'url' exists in the dictionary. 
```

### 使用 keys()方法检查字典列表中是否存在某个键

为了使用`keys()`方法检查一个键是否存在于字典列表中，我们将使用下面的过程。

*   我们将使用 for 循环遍历字典列表。
*   在对字典进行迭代时，我们将首先使用`keys()`方法获得字典的键列表。之后，我们将使用成员操作符来检查输入键是否出现在键列表中。
*   如果这个键出现在列表中，我们将打印出来。之后，我们将使用 break 语句跳出 for 循环。
*   如果我们在当前的键列表中没有找到该键，我们将使用 continue 语句移动到字典列表中的下一个字典。
*   在遍历了所有的字典之后，如果我们没有找到这个键，我们将打印出这个键在字典中不存在。

您可以在下面的示例中观察整个过程。

```py
listOfDicts = [{1: 1, 2: 4, 3: 9},
               {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB",
                "type": "python blog"},
               {"person": "Aditya", "Country": "India"}
               ]
key = "name"
print("The key is:", key)
keyFound = False
for dictionary in listOfDicts:
    keys = dictionary.keys()
    if key in keys:
        print("The key '{}' is present in the list of dictionaries.".format(key))
        keyFound = True
        break
    else:
        continue
if not keyFound:
    print("The key '{}' is not present in the list of dictionaries.".format(key))
```

输出:

```py
The key is: name
The key 'name' is present in the list of dictionaries.
```

## 使用 viewkeys()方法检查字典中是否存在某个键

在 python 版本中，我们可以使用`viewkeys()`方法代替 keys 方法来检查字典中是否存在某个键。

### viewkeys()方法

在 python 字典上调用`viewkeys()`方法时，会返回包含字典键的`dict_key`对象的视图，如下所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys=myDict.viewkeys()
print("The keys are:", keys)
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
('The keys are:', dict_keys(['url', 'acronym', 'type', 'name'])) 
```

为了使用`viewkeys()`方法检查指定的键是否存在于字典中，我们将首先使用`viewkeys()` 方法获得`dict_keys`对象。之后，我们将使用 for 循环遍历`dict_keys`对象。

迭代时，我们将检查当前键是否是我们正在搜索的键。如果是，我们会说这个键在字典里。否则不会。您可以在下面的代码中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.viewkeys()
print("The keys are:", keys)
input_key = "url"
keyFound = False
for key in keys:
    if key == input_key:
        print("The key '{}' exists in the dictionary.".format(input_key))
        keyFound = True
        break
    else:
        continue
if not keyFound:
    print("The key '{}' does not exist in the dictionary.".format(input_key))
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
('The keys are:', dict_keys(['url', 'acronym', 'type', 'name']))
The key 'url' exists in the dictionary. 
```

除了使用 for 循环，我们还可以使用 membership 操作符来检查字典中是否存在该键，如下所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.viewkeys()
print("The keys are:", keys)
input_key = "url"
if input_key in keys:
    print("The key '{}' exists in the dictionary.".format(input_key))
else:
    print("The key '{}' doesn't exist in the dictionary.".format(input_key))
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
('The keys are:', dict_keys(['url', 'acronym', 'type', 'name']))
The key 'url' exists in the dictionary.
```

### 使用 viewkeys()方法检查字典中是否存在多个键

为了使用`viewkeys()`方法检查字典中是否存在多个键，我们将遍历输入键的列表。

对于每个键，我们将使用成员操作符来检查该键是否存在于由`viewkeys()`方法返回的`dict_keys`对象中。如果是，我们将打印出该键存在于字典中。否则，我们将打印出该键不存在。最后，我们将转到下一个键。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.viewkeys()
print("The keys are:", keys)
input_keys = ["Aditya", "name", "url"]
print("The input keys are:", input_keys)
for input_key in input_keys:
    if input_key in keys:
        print("The key '{}' exists in the dictionary.".format(input_key))
    else:
        print("The key '{}' doesn't exist in the dictionary.".format(input_key))
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
('The keys are:', dict_keys(['url', 'acronym', 'type', 'name']))
('The input keys are:', ['Aditya', 'name', 'url'])
The key 'Aditya' doesn't exist in the dictionary.
The key 'name' exists in the dictionary.
The key 'url' exists in the dictionary.
```

### 使用 viewkeys()方法检查字典列表中是否存在某个键

为了使用`viewkeys()`方法检查一个键是否存在于字典列表中，我们将使用下面的过程。

*   我们将使用 for 循环遍历字典列表。
*   在对字典进行迭代时，我们将首先通过调用字典上的`viewkeys()`方法来获得`dict_keys`对象。之后，我们将使用成员操作符来检查输入键是否出现在`dict_keys`对象中。
*   如果键存在于`dict_keys`对象中，我们将打印出来。之后，我们将使用 break 语句跳出 for 循环。
*   如果我们在当前的`dict_keys`对象中没有找到关键字，我们将使用 continue 语句移动到字典列表中的下一个字典。
*   在遍历了所有的字典之后，如果我们没有找到这个键，我们将打印出这个键在字典列表中不存在。

您可以在下面的示例中观察整个过程。

```py
listOfDicts = [{1: 1, 2: 4, 3: 9},
               {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB",
                "type": "python blog"},
               {"person": "Aditya", "Country": "India"}
               ]
key = "name"
print("The key is:", key)
keyFound = False
for dictionary in listOfDicts:
    keys = dictionary.viewkeys()
    if key in keys:
        print("The key '{}' is present in the list of dictionaries.".format(key))
        keyFound = True
        break
    else:
        continue
if not keyFound:
    print("The key '{}' is not present in the list of dictionaries.".format(key))
```

输出:

```py
('The key is:', 'name')
The key 'name' is present in the list of dictionaries.
```

## 使用 iterkeys()方法检查字典中是否存在一个键

在 python 版本中，我们也可以使用`iterkeys()`方法代替 `viewkeys()`方法来检查字典中是否存在一个键。

### iterkeys()方法

当在 python 字典上调用`iterkeys()` 方法时，它返回一个迭代器，该迭代器迭代字典的键。

为了使用`iterkeys()`方法检查一个键是否存在于字典中，我们将首先通过调用字典上的 `iterkeys()`方法获得迭代器。之后，我们将使用 for 循环和迭代器遍历这些键。

迭代时，我们将检查当前键是否是我们正在搜索的键。如果是，我们会说这个键在字典里。否则不会。您可以在下面的 python 程序中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
iterator = myDict.iterkeys()
input_key = "url"
keyFound = False
for key in iterator:
    if key == input_key:
        print("The key '{}' exists in the dictionary.".format(input_key))
        keyFound = True
        break
    else:
        continue
if not keyFound:
    print("The key '{}' does not exist in the dictionary.".format(input_key))
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
The key 'url' exists in the dictionary. 
```

不使用 for 循环，您可以使用成员操作符和 `iterkeys()`方法来检查字典中是否存在一个键，如下所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
iterator = myDict.iterkeys()
input_key = "url"
if input_key in iterator:
    print("The key '{}' exists in the dictionary.".format(input_key))
else:
    print("The key '{}' doesn't exist in the dictionary.".format(input_key)) 
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
The key 'url' exists in the dictionary.
```

### 使用 iterkeys()方法检查字典中是否存在多个键

为了使用`iterkeys()`方法检查字典中是否存在多个键，我们将首先使用`iterkeys()`方法获得字典键的迭代器。之后，我们将遍历输入键列表。

对于每个键，我们将使用成员操作符来检查该键是否存在于由`iterkeys()`方法返回的迭代器对象中。如果是，我们将打印出该键存在于字典中。否则，我们将打印出该键不存在。最后，我们将转到下一个键。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
iterator = myDict.iterkeys()
input_keys = ["Aditya", "name", "url"]
print("The input keys are:", input_keys)
for input_key in input_keys:
    if input_key in iterator:
        print("The key '{}' exists in the dictionary.".format(input_key))
    else:
        print("The key '{}' doesn't exist in the dictionary.".format(input_key))
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
('The input keys are:', ['Aditya', 'name', 'url'])
The key 'Aditya' doesn't exist in the dictionary.
The key 'name' doesn't exist in the dictionary.
The key 'url' doesn't exist in the dictionary. 
```

### 使用 iterkeys()方法检查字典列表中是否存在某个键

为了使用`iterkeys()`方法检查一个键是否存在于字典列表中，我们将使用下面的过程。

*   我们将使用 for 循环遍历字典列表。
*   在对字典进行迭代时，我们将首先通过调用字典上的`iterkeys()`方法获得字典键的迭代器。之后，我们将使用成员操作符来检查迭代器中是否存在输入键。
*   如果键出现在迭代器中，我们将打印出来。之后，我们将使用 break 语句跳出 for 循环。
*   如果我们在当前迭代器中没有找到键，我们将使用 python continue 语句移动到字典列表中的下一个字典。
*   在遍历了所有的字典之后，如果我们没有找到这个键，我们将打印出这个键在字典中不存在。

您可以在下面的示例中观察整个过程。

```py
listOfDicts = [{1: 1, 2: 4, 3: 9},
               {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB",
                "type": "python blog"},
               {"person": "Aditya", "Country": "India"}
               ]
key = "name"
print("The key is:", key)
keyFound = False
for dictionary in listOfDicts:
    iterator = dictionary.iterkeys()
    if key in iterator:
        print("The key '{}' is present in the list of dictionaries.".format(key))
        keyFound = True
        break
    else:
        continue
if not keyFound:
    print("The key '{}' is not present in the list of dictionaries.".format(key))
```

输出:

```py
('The key is:', 'name')
The key 'name' is present in the list of dictionaries.
```

## 结论

在本文中，我们讨论了检查一个键是否存在于 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中的不同方法。如果您正在使用 python 3.x，那么在所有方法中，您应该使用使用成员运算符的方法来检查字典中是否存在某个键。如果您正在使用 python 2.x，您可以使用使用`iterkeys()` 方法的方法来检查给定的键是否存在于字典中。这两种方法对于各自版本的 python 来说是最快的。

我希望你喜欢阅读这篇文章。要了解更多关于 python 编程的知识，您可以阅读这篇关于如何在 Python 中[删除列表中所有出现的字符的文章。您可能也喜欢这篇关于如何](https://www.pythonforbeginners.com/basics/remove-all-occurrences-of-a-character-in-a-list-or-string-in-python)[检查 python 字符串是否包含数字](https://www.pythonforbeginners.com/strings/check-if-a-python-string-contains-a-number)的文章。

请继续关注更多内容丰富的文章。

快乐学习！