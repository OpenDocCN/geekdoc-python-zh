# Python:更多词典

> 原文：<https://www.pythonforbeginners.com/dictionary/python-more-dictionarys>

### 什么是字典？

字典是具有“键”和“值”的条目的集合。

字典是可变的。您不必重新分配字典来对其进行
更改。

它们就像列表一样，除了没有指定的索引号，
而是由你自己编索引:

### 示例 1

```py
testList = ["first", "second", "third"]
testDict = {0:"first", 1:"second", 2:"third"} 
```

Python 中的字典是用{}括起来的，要创建一个字典，必须提供一个键/值。

字典中的每个键必须是唯一的。

冒号放在键和值之间(键:值)

每个键:值对由逗号分隔

### 示例 2

```py
>> phonenumbers = {'Jack':'555-555', 'Jill':'555-556'} 

phonebook = {}
phonebook["Jack"] = "555-555"
phonebook["Jill"] = "555-556"

print phonebook
{'Jill': '555-556', 'Jack': '555-555'} 
```

字典只有一种工作方式，要从字典中获取一个值，必须输入键。

您不能提供值并获取密钥。

### 示例 3

```py
phonebook = {}
phonebook["Jack"] = "555-555"
phonebook["Jill"] = "555-556"

print phonebook['Jill']
555-556 
```

### 键/值用法

```py
 To add a key / value pair in a dictionary
>>phonebook["Matt"] = "555-557"

To change a key / value pair:
>>phonebook["Jack"] = '555-558'

To remove a key / value pair, use del
 >>del phonebook["Jill"]

To see if a key exists, use has_key() method
>>phonebook.has_key("Matt")

To copy whole dictionary, use the copy() method
phonebook2 = phonebook.copy() 
```

在存储查找结果时，我通常使用字典。