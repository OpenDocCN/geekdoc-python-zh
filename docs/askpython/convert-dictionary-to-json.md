# 如何用 Python 把 Dictionary 转换成 JSON？

> 原文：<https://www.askpython.com/python/dictionary/convert-dictionary-to-json>

在本文中，让我们学习如何将 python 字典转换成 JSON。我们先来了解一下 JSON 是什么。JSON 代表 javascript 对象符号。它通常用于在 web 客户端和 web 服务器之间交换信息。JSON 的结构类似于 python 中的字典。条件是，JSON 键必须总是带双引号的字符串。而且，对应于键的值可以是任何数据类型，比如字符串、整数、嵌套 JSON 等。JSON 的返回类型是“字符串”对象类型。

***也读:[如何用 Python 把 JSON 转换成字典？](https://www.askpython.com/python/dictionary/convert-json-to-a-dictionary)***

示例:

```py
import json

a =  '{ "One":"A", "Two":"B", "Three":"C"}'

```

python 中的 Dictionary 是一种内置的数据类型，用于将数据存储在与值格式相关联的键中。存储在字典中的数据是无序的、唯一的对(键总是唯一的，值可以重复)，并且是可变的。dictionary 的返回类型是“dict”对象类型。

示例:

```py
#Dictionary in python is built-in datatype so 
#no need to import anything.

dict1 = { 'One' : 1, 'Two' : 2, 'C': 3}

```

## 将 Dict 转换为 JSON

Python 确实有一个名为“json”的默认模块，可以帮助我们将不同的数据形式转换成 JSON。我们今天使用的函数是 json。dumps()方法允许我们将 python 对象(在本例中是 dictionary)转换成等价的 JSON 对象。

*   首先，我们导入 **[json 模块](https://www.askpython.com/python-modules/python-json-module)**
*   为必须转换成 JSON 字符串的字典分配一个变量名。
*   使用 json.dumps( *变量*)进行转换

***注意**:不要混淆 json.dumps 和 json.dump. `json.dumps()`是一种可以将 Python 对象转换成 json 字符串的方法，而`json.dump()`是一种用于将 json 写入/转储到文件中的方法。*

**JSON 的语法**

```py
json.dumps(dict,intend)
```

*   **dict**–我们需要转换的 [**python 字典**](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)
*   intent–缩进的数量(代码行开头的空格)

```py
import json

dict1 ={ 
  "Name": "Adam", 
  "Roll No": "1", 
  "Class": "Python"
} 

json_object = json.dumps(dict1, indent = 3) 
print(json_object)

```

**输出:**

```py
{
   "Name": "Adam",
   "Roll No": "1",
   "Class": "Python"
}
```

## 使用 sort_keys 属性将字典转换为 JSON

使用前面讨论的 *dumps()* 方法中的 *sort_key* 属性，以排序的方式返回一个 JSON 对象。如果属性设置为 TRUE，那么字典将被排序并转换为 JSON 对象。如果它被设置为 FALSE，那么字典已经转换了它的方式，没有排序。

```py
import json

dict1 ={ 
  "Adam": 1,
  "Olive" : 4, 
  "Malcom": 3,
   "Anh": 2, 
} 

json_object = json.dumps(dict1, indent = 3, sort_keys = True) 
print(json_object)

```

**输出:**

```py
{
   "Adam": 1,
   "Anh": 2,
   "Malcom": 3,
   "Olive": 4
} 
```

## 将嵌套的 dict 转换成 JSON

在 dict 中声明的 dict 称为嵌套 dict。dumps()方法也可以将这种嵌套的 dict 转换成 json。

```py
dict1 ={ 
  "Adam": {"Age" : 32, "Height" : 6.2},
  "Malcom" : {"Age" : 26, "Height" : 5.8},
}

json_object = json.dumps(dict1, indent = 3, sort_keys = True) 
print(json_object)

```

**输出:**

```py
{
   "Adam": {
      "Age": 32,
      "Height": 6.2
   },
   "Malcom": {
      "Age": 26,
      "Height": 5.8
   }
}
```

## 摘要

在本文中，我们讨论了如何将字典数据结构转换成 JSON 以便进一步处理。我们使用 json 模块将字典序列化为 JSON。

### 参考

*   [https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html)
*   [https://docs.python.org/3/tutorial/datastructures.html](https://docs.python.org/3/tutorial/datastructures.html)