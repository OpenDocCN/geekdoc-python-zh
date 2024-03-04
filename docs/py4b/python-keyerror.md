# Python KeyError

> 原文：<https://www.pythonforbeginners.com/basics/python-keyerror>

在 python 中使用字典时，您可能会遇到 KeyError。在本文中，我们将讨论什么是 KeyError，它是如何发生的，以及如何在使用 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)时避免 KeyError。

## 什么是 Python KeyError？

简而言之，当我们试图用字典中不存在的键访问字典中的值时，KeyError 是 python 程序引发的异常。

例如，看看下面的程序。这里，字典 **myDict** 具有键 1、2 和 3，值 1、4 和 9 与这些键相关联。当我们试图用键 4 访问字典时，程序会引发一个 KeyError。这是因为 4 在字典中不是一个键。

```py
myDict = {1: 1, 2: 4, 3: 9}
print("The dictionary is:", myDict)
print(myDict[4]) 
```

输出:

```py
The dictionary is: {1: 1, 2: 4, 3: 9}
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string.py", line 3, in <module>
    print(myDict[4])
KeyError: 4
```

## 如何避免 Python KeyError 异常？

有多种方法可以避免 KeyError 异常。让我们逐一讨论。

### 使用 If else 语句

使用 if else 语句，我们可以在访问值之前检查给定的键是否存在于字典的键中。这有助于我们避免 KeyError 异常。

```py
myDict = {1: 1, 2: 4, 3: 9}
print("The dictionary is:", myDict)
key = 4
if key in myDict.keys():
    print(myDict[key])
else:
    print("{} not a key of dictionary".format(key)) 
```

输出:

```py
The dictionary is: {1: 1, 2: 4, 3: 9}
4 not a key of dictionary
```

这里的缺点是，每次我们都必须检查给定的键是否存在于字典的键中。如果我们使用 get()方法从字典中访问值，就可以避免这些额外的工作。

### 使用 get()方法

当在字典上调用 get()方法时，它将给定的键和一个可选值作为输入。如果给定的键存在于字典中，它会将相关的值作为输出提供给该键，如下所示。

```py
myDict = {1: 1, 2: 4, 3: 9}
print("The dictionary is:", myDict)
key = 3
print("Key is:",key)
print("Value associated to the key is:",myDict.get(key)) 
```

输出:

```py
The dictionary is: {1: 1, 2: 4, 3: 9}
Key is: 3
Value associated to the key is: 9
```

当给定的键不在字典中时，如果没有传递可选值，它将返回 None。

```py
myDict = {1: 1, 2: 4, 3: 9}
print("The dictionary is:", myDict)
key = 4
print("Key is:",key)
print("Value associated to the key is:",myDict.get(key)) 
```

输出:

```py
 The dictionary is: {1: 1, 2: 4, 3: 9}
Key is: 4
Value associated to the key is: None
```

如果我们将一个可选值作为输入传递给 get()方法，当给定的键不在字典中时，它将返回该值。

```py
myDict = {1: 1, 2: 4, 3: 9}
print("The dictionary is:", myDict)
key = 4
print("Key is:",key)
print("Value associated to the key is:",myDict.get(key,16)) 
```

输出:

```py
The dictionary is: {1: 1, 2: 4, 3: 9}
Key is: 4
Value associated to the key is: 16
```

### 使用 Try Except

我们可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块来处理 KeyError 异常。为此，我们将执行代码，使用 try 块中的给定键来访问值，并将处理 except 块中的异常，如下所示。

```py
myDict = {1: 1, 2: 4, 3: 9}
print("The dictionary is:", myDict)
key=4
print("Key is:",key)
try:
    val=myDict[key]
    print("Value associated to the key is:",val)
except KeyError:
    print("Key not present in Dictionary") 
```

输出:

```py
The dictionary is: {1: 1, 2: 4, 3: 9}
Key is: 4
Key not present in Dictionary
```

## 结论

在本文中，我们讨论了键错误以及处理它们的方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)