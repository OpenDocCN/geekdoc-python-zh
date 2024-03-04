# 检查 Python 中的字典中是否存在值

> 原文：<https://www.pythonforbeginners.com/basics/check-if-value-exists-in-a-dictionary-in-python>

我们使用字典来存储和操作 python 程序中的键值对。有时，我们需要检查一个值是否存在于字典中。在本 python 教程中，我们将讨论检查一个值是否存在于 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中的不同方法。在这里，当检查值时，我们可能有可用的键。我们将讨论在这两种情况下如何检查一个值是否存在于字典中。

## 当我们有可用的键时，检查字典中是否存在一个值

当我们有了字典的键时，我们可以使用下标操作符或`get()`方法来检查字典中是否存在给定的值。让我们逐一讨论每种方法。

### 使用下标运算符检查字典中是否存在某个值

#### 下标运算符

当我们有一个键，我们想检查一个值是否存在于字典中，我们可以使用下标操作符。为此，我们可以使用方括号来检索与使用以下语法的键相关联的值。

```py
value=dict[key_val]
```

这里，`dict`是字典的名称，`key_val`是键。执行后，上述语句返回与字典中的键相关联的值。您可以在下面的代码示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "name"
value = myDict[key]
print("The value associated with the key \"{}\" is \"{}\".".format(key, value)) 
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The value associated with the key "name" is "Python For Beginners".
```

这里，我们首先创建了一个字典名`myDict`，带有关键字`name`、`url`、`acronym`和`type`。之后，我们使用下标操作符检索了与键'`name`'相关联的值。在这里，程序正常工作。

但是，可能会出现字典中不存在提供给下标操作符的键的情况。在这种情况下，程序会遇到如下所示的`KeyError`异常。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "class"
value = myDict[key]
print("The value associated with the key \"{}\" is \"{}\".".format(key, value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/webscraping.py", line 5, in <module>
    value = myDict[key]
KeyError: 'class'
```

这里，关键字'`class`'在字典中不存在。因此，程序运行到`[KeyError](https://www.pythonforbeginners.com/basics/python-keyerror)`异常。

在这种情况下，程序会突然终止，程序执行期间所做的任何工作都会丢失。在这些情况下，您可以使用 [python try-except](https://www.pythonforbeginners.com/error-handling/how-to-best-use-try-except-in-python) 块来处理异常，如下所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "class"
try:
    value = myDict[key]
    print("The value associated with the key \"{}\" is \"{}\".".format(key, value))
except KeyError:
    print("The key '{}' is not present in the dictionary.".format(key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The key 'class' is not present in the dictionary.
```

在上面的例子中，`KeyError`异常是在 try 块中引发的。在 except 块中，我们捕获异常，并通过为用户打印适当的消息来正常终止程序。

#### 当我们只有一个输入键时

为了使用下标符号检查一个值是否存在于字典中，我们将获得与字典中的键名称相关联的值。之后，我们将检查获得的值是否等于给定的值，我们正在检查该值是否存在。

如果两个值都匹配，我们就说输入值对存在于字典中。否则，不会。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "name"
input_value = "Python For Beginners"
print("Input key is:", key)
print("Input value is:", input_value)
try:
    value = myDict[key]
    if value == input_value:
        print("'{}' is present in the dictionary".format(input_value))
    else:
        print("'{}' is not present in the dictionary".format(input_value))
except KeyError:
    print("The key '{}' is not present in the dictionary.".format(key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
Input key is: name
Input value is: Python For Beginners
'Python For Beginners' is present in the dictionary
```

我们已经得到了上述代码中的关键字'`name`'。除此之外，我们还有值`'Python For Beginners`，我们必须检查它是否存在。由于我们只有一个键，在这种情况下，我们刚刚获得了与给定键相关联的值。之后，我们将获得的值与给定的输入值进行比较，以检查输入值是否存在于字典中。

#### 当我们有多个输入键时

现在，我们有多个键，我们需要检查字典中是否存在一个值。这里，我们需要对每个键执行上面讨论的整个操作。

在这种方法中，我们将使用 for 循环遍历作为输入给出的键列表。对于列表中出现的每个键，我们将检索相关联的值，并将其与输入值进行比较。如果两个值都匹配，我们就说输入值存在于字典中。同时，我们将跳出 for 循环。

如果没有一个键有与输入值相等的关联值，我们就说这个值不在字典中。您可以在下面的示例中观察整个过程。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = ["name", 'type']
input_value = "Python For Beginners"
print("Input keys are:", keys)
print("Input value is:", input_value)
valueFound = False
for key in keys:

    try:
        value = myDict[key]
        if value == input_value:
            print("'{}' is present in the dictionary".format(input_value))
            valueFound = True
            break
        else:
            continue

    except KeyError:
        print("The key '{}' is not present in the dictionary.".format(key))
if not valueFound:
    print("'{}' is not present in the dictionary".format(input_value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
Input keys are: ['name', 'type']
Input value is: Python For Beginners
'Python For Beginners' is present in the dictionary
```

在使用下标操作符时，如果字典中不存在这个键，程序就会遇到`KeyError`异常。处理`KeyError`异常在时间和内存方面代价很高。因此，我们可以通过使用`keys()`方法或使用 `get()`方法检查一个键的存在来避免异常。

### 使用 keys()方法检查字典中是否存在某个值

在字典上调用 `keys()`方法时，会返回一个包含字典键的`dict_keys`对象。您可以在下面的结果中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.keys()
print("The keys in the dictionary are:")
print(keys)
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The keys in the dictionary are:
dict_keys(['name', 'url', 'acronym', 'type'])
```

#### 当我们只有一个输入键时

为了使用`keys()`方法检查一个值是否存在于字典中，我们将首先获得字典的键列表。

之后，我们将检查特定键的存在，以确定该键对于现有字典是有效的键。如果输入键出现在键列表中，我们将继续检查字典中是否存在给定值。

对于有效的键，为了检查字典中是否存在该值，我们将获取与字典的键相关联的值。之后，我们将检查获得的值是否等于给定的值，我们正在检查该值是否存在。如果两个值都匹配，我们就说这个值存在于字典中。否则不会。

您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.keys()
print("The keys in the dictionary are:")
print(keys)
key="name"
input_value="Python For Beginners"
if key in keys:
    value = myDict[key]
    if value == input_value:
        print("'{}' is present in the dictionary.".format(input_value))
    else:
        print("'{}' is not present in the dictionary.".format(input_value))
else:
    print("The key '{}' is not present in the dictionary.".format(key))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The keys in the dictionary are:
dict_keys(['name', 'url', 'acronym', 'type'])
'Python For Beginners' is present in the dictionary.
```

#### 当我们有多个输入键时

当我们有不止一个键时，我们可以使用 for 循环来遍历键列表。我们将对每个给定的键重复整个过程。如果没有一个键具有与给定值相同的关联值，我们就说该值不在字典中。你可以在下面的例子中观察整个过程。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.keys()
print("The keys in the dictionary are:")
print(keys)
input_keys = ['Aditya',"name", 'type']
input_value = "Python For Beginners"
valueFound = False
for key in input_keys:
    if key in keys:
        value = myDict[key]
        if value == input_value:
            print("'{}' is present in the dictionary.".format(input_value))
            valueFound = True
            break
        else:
            continue
    else:
        print("The key '{}' is not present in the dictionary.".format(key))

if not valueFound:
    print("'{}' is not present in the dictionary.".format(input_value)) 
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The keys in the dictionary are:
dict_keys(['name', 'url', 'acronym', 'type'])
The key 'Aditya' is not present in the dictionary.
'Python For Beginners' is present in the dictionary. 
```

### 使用 get()方法检查字典中是否存在某个值

我们可以使用`get()`方法来检查字典中是否存在一个值，而不是使用`keys()` 方法来检查一个键的存在，然后使用下标操作符来获取值。

在字典上调用时，`get()`方法接受一个键作为输入参数。如果字典中存在该键，它将返回与该键相关联的值，如下所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "name"
print("THe key is '{}'.".format(key))
value = myDict.get(key)
print("THe value is '{}'.".format(value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
THe key is 'name'.
THe value is 'Python For Beginners'.
```

如果给定的键在字典中不存在，`get()`方法返回默认值`None`。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "website"
print("THe key is '{}'.".format(key))
value = myDict.get(key)
print("THe value is '{}'.".format(value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
THe key is 'website'.
THe value is 'None'.
```

#### 当我们只有一个输入键时

为了使用 get 函数检查字典中是否存在某个值，我们将获取与给定键相关联的值。之后，我们将检查获得的值是否等于给定值。如果是，我们会说该值存在于字典中。否则，不会。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
key = "name"
input_value = "Python For Beginners"
print("The key is '{}'.".format(key))
print("The input value is '{}'.".format(input_value))
value = myDict.get(key)
if value is None:
    print("The key '{}' is not present in the dictionary.".format(key))
elif value == input_value:
    print("The value '{}' is present in the dictionary.".format(input_value))
else:
    print("The value '{}' is not present in the dictionary.".format(input_value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The key is 'name'.
The input value is 'Python For Beginners'.
The value 'Python For Beginners' is present in the dictionary.
```

#### 当我们有多个输入键时

如果给了我们多个键，我们可以使用 for 循环来遍历键列表。迭代时，我们可以检查每个键的输入值是否存在。如果给定的键都没有与给定值相等的关联值，我们就说该值不在字典中。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = ['Aditya', "name", 'url']
input_value = "Python For Beginners"
print("The input keys are '{}'.".format(keys))
print("The input value is '{}'.".format(input_value))
valueFound=False
for key in keys:
    value = myDict.get(key)
    if value is None:
        print("The key '{}' is not present in the dictionary.".format(key))
    elif value == input_value:
        print("The value '{}' is present in the dictionary.".format(input_value))
        valueFound=True
        break
if not valueFound:
    print("The value '{}' is not present in the dictionary.".format(input_value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input keys are '['Aditya', 'name', 'url']'.
The input value is 'Python For Beginners'.
The key 'Aditya' is not present in the dictionary.
The value 'Python For Beginners' is present in the dictionary.
```

到目前为止，我们已经讨论了不同的场景来检查一个值是否存在于一个字典中，当我们被给定一些字典的键时。

现在让我们讨论当没有给定键，而只给定了一个我们必须检查其存在的值时，检查一个值是否存在于字典中的不同方法。

## 当键不可用时，检查字典中是否存在值

### 使用 keys()方法检查字典中是否存在某个值

为了使用 `keys()`方法检查一个值是否存在于字典中，我们将首先通过在字典上执行`keys()`方法来获得键的列表。

之后，我们将使用下标操作符来获取与键列表中的每个键相关联的字典值。如果任何获得的值等于我们正在检查其存在的值，我们将说该值存在于字典中。否则不会。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
input_value = "Python For Beginners"
print("The input value is '{}'.".format(input_value))
keys = myDict.keys()
isPresent = False
for key in keys:
    value = myDict[key]
    if value == input_value:
        print("'{}' is present in the dictionary.".format(input_value))
        isPresent = True
        break
    else:
        continue
if not isPresent:
    print("'{}' is not present in the dictionary.".format(input_value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input value is 'Python For Beginners'.
'Python For Beginners' is present in the dictionary.
```

#### 使用 keys()方法检查字典中是否存在多个值

如果我们需要检查字典中是否存在多个值，我们将首先使用如下所示的`get()`方法和`keys()`方法获得一个值列表。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.keys()
print("The keys of the dictionary are:")
print(keys)
values = []
for key in keys:
    value = myDict.get(key)
    values.append(value)
print("The obtained values are '{}'.".format(values)) 
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The keys of the dictionary are:
dict_keys(['name', 'url', 'acronym', 'type'])
The obtained values are '['Python For Beginners', 'pythonforbeginners.com', 'PFB', 'python blog']'. 
```

这里，我们首先使用`keys()`方法获得了字典的键。之后，我们创建了一个空列表来存储字典的值。然后，我们使用`get()`方法获得与字典中每个键相关联的值，并将其存储在列表中。

在获得字典中的值列表后，我们将检查每个输入值是否都在其中。为此，我们可以使用带有成员操作符的 For 循环，如下所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
keys = myDict.keys()
print("The keys of the dictionary are:")
print(keys)
input_values = ["Python For Beginners", "PFB", 'Aditya']
print("The input values are:", input_values)
values = []
for key in keys:
    value = myDict.get(key)
    values.append(value)
print("The obtained values are '{}'.".format(values))
for value in input_values:
    if value in values:
        print("The value '{}' is present in the dictionary.".format(value))
    else:
        print("The value '{}' is not present in the dictionary.".format(value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The keys of the dictionary are:
dict_keys(['name', 'url', 'acronym', 'type'])
The input values are: ['Python For Beginners', 'PFB', 'Aditya']
The obtained values are '['Python For Beginners', 'pythonforbeginners.com', 'PFB', 'python blog']'.
The value 'Python For Beginners' is present in the dictionary.
The value 'PFB' is present in the dictionary.
The value 'Aditya' is not present in the dictionary. 
```

### 使用 values()方法检查字典中是否存在某个值

在字典上调用`values()`方法时，它返回包含字典值的`dict_values`对象的副本。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.values()
print("The values in the dictionary are:")
print(values)
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The values in the dictionary are:
dict_values(['Python For Beginners', 'pythonforbeginners.com', 'PFB', 'python blog'])
```

为了使用`values()`方法检查一个值是否存在于字典中，我们将首先通过调用字典上的`values()`方法获得包含字典值的`dict_values`对象。

之后，我们将遍历值列表，检查作为用户输入给出的值是否出现在值列表中。如果是，我们会说该值存在于字典中。否则不会。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.values()
print("The values in the dictionary are:")
print(values)
input_value = "Python For Beginners"
print("The input value is '{}'.".format(input_value))
isPresent = False
for value in values:
    if value == input_value:
        print("'{}' is present in the dictionary.".format(input_value))
        isPresent = True
        break
    else:
        continue
if not isPresent:
    print("'{}' is not present in the dictionary.".format(input_value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The values in the dictionary are:
dict_values(['Python For Beginners', 'pythonforbeginners.com', 'PFB', 'python blog'])
The input value is 'Python For Beginners'.
'Python For Beginners' is present in the dictionary.
```

我们可以使用成员操作符“`in`”来检查给定值是否出现在值列表中，而不是使用 for 循环来遍历值列表。`in`操作符的语法如下。

```py
element in container_object
```

`in`操作符是一个二元操作符，它将一个元素作为第一个操作数，将一个容器对象或迭代器作为第二个操作数。执行后，如果元素存在于容器对象或迭代器中，则返回`True`。否则，它返回`False`。

为了检查字典中是否存在给定的值，我们将使用成员操作符检查该值是否存在于由`values()`方法返回的值列表中，如下例所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.values()
print("The values in the dictionary are:")
print(values)
input_value = "Python For Beginners"
print("The input value is '{}'.".format(input_value))
if input_value in values:
    print("'{}' is present in the dictionary.".format(input_value))
else:
    print("'{}' is not present in the dictionary.".format(input_value)) 
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The values in the dictionary are:
dict_values(['Python For Beginners', 'pythonforbeginners.com', 'PFB', 'python blog'])
The input value is 'Python For Beginners'.
'Python For Beginners' is present in the dictionary.
```

#### 使用 Values()方法检查字典中是否存在多个值

如果给我们多个值来检查它们的存在，我们将使用一个带有成员操作符的 for 循环和`values()`方法来检查键的存在。这里，我们将使用 for 循环迭代输入值列表。迭代时，我们将检查当前值是否出现在使用`values()`方法获得的值列表中。如果是，我们将打印该值存在于字典中。否则不会。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
input_values = ["Python For Beginners", "PFB", 'Aditya']
print("The input values are:", input_values)
values = myDict.values()
for value in input_values:
    if value in values:
        print("The value '{}' is present in the dictionary.".format(value))
    else:
        print("The value '{}' is not present in the dictionary.".format(value))
```

输出:

```py
The dictionary is:
{'name': 'Python For Beginners', 'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog'}
The input values are: ['Python For Beginners', 'PFB', 'Aditya']
The value 'Python For Beginners' is present in the dictionary.
The value 'PFB' is present in the dictionary.
The value 'Aditya' is not present in the dictionary.
```

### 使用 viewvalues()方法检查字典中是否存在某个值

如果您使用的是 Python 版，而不是使用 `values()`方法，那么您可以使用`viewvalues()` 方法来检查字典中是否存在一个值。

在字典上调用`viewvalues()`方法时，它返回包含字典中值的新视图的`dict_values`对象的视图，如下所示。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.viewvalues()
print("The values in the dictionary are:")
print(values)
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
The values in the dictionary are:
dict_values(['pythonforbeginners.com', 'PFB', 'python blog', 'Python For Beginners'])
```

获得`dict_values`对象后，我们可以检查输入值是否存在于字典中。

为了使用`viewvalues()`方法检查一个值是否存在于字典中，我们将首先通过调用字典上的`viewvalues()`方法获得`dict_values`对象。

之后，我们将遍历`dict_values`对象，检查作为用户输入给出的值是否出现在值列表中。如果是，我们会说该值存在于字典中。否则不会。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.viewvalues()
print("The values in the dictionary are:")
print(values)
input_value = "Python For Beginners"
print("The input value is '{}'.".format(input_value))
isPresent = False
for value in values:
    if value == input_value:
        print("'{}' is present in the dictionary.".format(input_value))
        isPresent = True
        break
    else:
        continue
if not isPresent:
    print("'{}' is not present in the dictionary.".format(input_value))
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
The values in the dictionary are:
dict_values(['pythonforbeginners.com', 'PFB', 'python blog', 'Python For Beginners'])
The input value is 'Python For Beginners'.
'Python For Beginners' is present in the dictionary.
```

除了使用 for 循环，我们还可以使用如下所示的成员测试来检查输入值在`dict_values`对象中的存在。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.viewvalues()
print("The values in the dictionary are:")
print(values)
input_value = "Python For Beginners"
print("The input value is '{}'.".format(input_value))
if input_value in values:
    print("'{}' is present in the dictionary.".format(input_value))
else:
    print("'{}' is not present in the dictionary.".format(input_value)) 
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
The values in the dictionary are:
dict_values(['pythonforbeginners.com', 'PFB', 'python blog', 'Python For Beginners'])
The input value is 'Python For Beginners'.
'Python For Beginners' is present in the dictionary.
```

#### 使用 viewvalues()方法检查字典中是否存在多个值

如果给我们多个值来检查它们的存在，我们将使用一个带有成员操作符的 for 循环和`viewvalues()`方法来检查键的存在。这里，我们将使用 for 循环迭代输入值列表。迭代时，我们将检查当前值是否出现在使用`viewvalues()`方法获得的值列表中。如果是，我们将打印该值存在于字典中。否则不会。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.viewvalues()
print("The values in the dictionary are:")
print(values)
input_values = ["Python For Beginners",'PFB','Aditya']
print("The input values are '{}'.".format(input_values))
for input_value in input_values:
    if input_value in values:
        print("'{}' is present in the dictionary.".format(input_value))
    else:
        print("'{}' is not present in the dictionary.".format(input_value))
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
The values in the dictionary are:
dict_values(['pythonforbeginners.com', 'PFB', 'python blog', 'Python For Beginners'])
The input values are '['Python For Beginners', 'PFB', 'Aditya']'.
'Python For Beginners' is present in the dictionary.
'PFB' is present in the dictionary.
'Aditya' is not present in the dictionary.
```

### 使用 itervalues()方法检查字典中是否存在某个值

在 python 2 中，我们还可以使用 itervalues()方法来检查一个值是否存在于字典中。

在字典上调用`itervalues()` 方法时，它返回一个迭代器，我们可以用它迭代字典中的值。

为了使用`itervalues()`方法检查一个值是否存在于字典中，我们将首先通过在字典上调用 `itervalues()`方法来获得它返回的迭代器。之后，我们可以使用 for 循环遍历迭代器，并检查输入值是否出现在迭代器中。如果是，我们会说该值存在于字典中。否则不会。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.itervalues()
input_value = "Python For Beginners"
print("The input value is '{}'.".format(input_value))
isPresent = False
for value in values:
    if value == input_value:
        print("'{}' is present in the dictionary.".format(input_value))
        isPresent = True
        break
    else:
        continue
if not isPresent:
    print("'{}' is not present in the dictionary.".format(input_value))
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
The input value is 'Python For Beginners'.
'Python For Beginners' is present in the dictionary.
```

除了使用 for 循环，我们还可以使用如下所示的成员测试来检查输入值在`dict_values`对象中的存在。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.itervalues()
input_value = "Python For Beginners"
print("The input value is '{}'.".format(input_value))
if input_value in values:
    print("'{}' is present in the dictionary.".format(input_value))
else:
    print("'{}' is not present in the dictionary.".format(input_value)) 
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
The input value is 'Python For Beginners'.
'Python For Beginners' is present in the dictionary. 
```

#### 使用 itervalues()方法检查字典中是否存在多个值

如果给我们多个值来检查它们的存在，我们将使用一个带有成员操作符的 for 循环和`itervalues()` 方法来检查键的存在。

这里，我们将使用 for 循环迭代输入值列表。迭代时，我们将检查当前值是否出现在使用`itervalues()` 方法获得的值的迭代器中。如果是，我们将打印该值存在于字典中。否则不会。您可以在下面的示例中观察到这一点。

```py
myDict = {"name": "Python For Beginners", "url": "pythonforbeginners.com", "acronym": "PFB", "type": "python blog"}
print("The dictionary is:")
print(myDict)
values = myDict.itervalues()
input_values = ["Python For Beginners",'PFB','Aditya']
print("The input values are '{}'.".format(input_values))
for input_value in input_values:
    if input_value in values:
        print("'{}' is present in the dictionary.".format(input_value))
    else:
        print("'{}' is not present in the dictionary.".format(input_value))
```

输出:

```py
The dictionary is:
{'url': 'pythonforbeginners.com', 'acronym': 'PFB', 'type': 'python blog', 'name': 'Python For Beginners'}
The input values are '['Python For Beginners', 'PFB', 'Aditya']'.
'Python For Beginners' is present in the dictionary.
'PFB' is not present in the dictionary.
'Aditya' is not present in the dictionary.
```

## 结论

在本文中，我们讨论了检查字典中是否存在某个值的各种方法。如果您有字典的键，您可以使用使用`get()`方法的方法来检查一个值是否存在于字典中。如果没有密钥，应该使用 python 3.x 中的`values()`方法。对于 python 3.x 版，应该使用带有`itervalues()`方法的方法，因为它是所有方法中最快的。

我希望你喜欢阅读这篇文章。要了解更多关于 python 编程的知识，您可以阅读这篇关于如何在 Python 中[删除列表中所有出现的字符的文章。您可能也喜欢这篇关于如何](https://www.pythonforbeginners.com/basics/remove-all-occurrences-of-a-character-in-a-list-or-string-in-python)[检查 python 字符串是否包含数字](https://www.pythonforbeginners.com/strings/check-if-a-python-string-contains-a-number)的文章。

请继续关注更多内容丰富的文章。

快乐学习！