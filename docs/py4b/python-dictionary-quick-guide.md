# Python 字典快速指南

> 原文：<https://www.pythonforbeginners.com/dictionary/python-dictionary-quick-guide>

如题，这是一本 Python 字典快速指南。

请查看[字典教程](https://www.pythonforbeginners.com/basics/dictionary)了解更多关于字典的文章。

```py
 # key/value pairs declaration
 dict = {
 ‘key1′:’value1′,
 ‘key2′:’value2′,
 ‘key3′:’value3′
 }

#Get all keys
 dict.keys()

#Get all values
 dict.values()

#Modifying
 dict['key2'] = ‘value8′

#Accessing
 print dict['key1']

# prints ‘value2′
 print dict['key2']

# empty declaration + assignment of key-value pair
 emptyDict = {}
 emptyDict['key4']=’value4′

# looping through dictionaries (keys and values)
 for key in dict:
 print dict[key]

# sorting keys and accessing their value in order
 keys = dict.keys()
 keys.sort()
 for key in keys:
 print dict[key]

# looping their values directory (not in order)
 for value in dict.values():
 print value

# getting both the keys and values at once
 for key,value in dict.items():
 print “%s=%s” % (key,value)

# deleting an entry
 del dict['key2']

# delete all entries in a dictionary
 dict.clear()

# size of the dictionary
 len(dict) 
```