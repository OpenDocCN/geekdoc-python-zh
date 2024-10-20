# 将 JSON 序列化和反序列化为 Python 中的对象

> 原文：<https://www.askpython.com/python/examples/serialize-deserialize-json>

读者朋友们，你们好！在本文中，我们将关注 Python 中 JSON 到对象的**序列化和反序列化的概念。**

所以，让我们开始吧！！🙂

在处理数据和 API 时，我们会遇到字典或 JSON 格式的数据。有时，我们需要一些函数来实现它们之间的相互转换。我们将了解一些序列化和反序列化数据的方法。

***也读作: [Python JSON 模块](https://www.askpython.com/python-modules/python-json-module)***

* * *

## Python 中 JSON 数据的序列化

序列化是将原始数据的数据类型转换成 JSON 格式的过程。因此，我们的意思是说，原始数据通常是一个[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)现在将遵循 Javascript 对象符号格式。

同样，Python 为我们提供了下面的函数来轻松地将我们的数据公式化为 JSON——

1.  **json.dump()函数**
2.  **json.dumps()函数**

### json.dump()函数

在 json.dump()函数中，它接受原始数据作为输入，将数据转换成 json 格式，然后存储到 JSON 文件中。

**语法**:

```py
json.dump(data, file-object)

```

*   数据:需要转换成 JSON 格式的实际数据。
*   file-object:它是指向存储转换后的数据的 JSON 文件的对象。如果文件不存在，则在对象指向的位置创建一个新文件。

**举例**:

```py
import json

data= {
    "details": {
        "name": "YZ",
        "subject": "Engineering",
        "City": "Pune"
    }
}

with open( "info.json" , "w" ) as x:
    json.dump( data, x )

```

### json.dumps()函数

与 dump()函数不同，json.dumps()函数确实将原始数据转换为 json 格式，但将其存储为字符串，而不是指向文件对象。

**语法**:

```py
json.dumps(data)

```

**举例**:

```py
import json

data= {
    "details": {
        "name": "YZ",
        "subject": "Engineering",
        "City": "Pune"
    }
}
res = json.dumps(data)
print(res)

```

**输出—**

```py
{"details": {"name": "YZ","subject": "Engineering","City": "Pune"}}

```

* * *

## JSON 数据的反序列化

理解了反序列化之后，现在让我们颠倒一下这个过程。

也就是说，通过反序列化，我们可以很容易地将 JSON 数据转换成默认/本地数据类型，通常是一个字典。

同样，Python 为我们提供了以下函数来实现反序列化的概念

1.  **json.load()函数**
2.  **json.loads()函数**

* * *

### json.load()函数

这里，load()函数使我们能够将 JSON 数据转换成本地字典格式。

**语法**:

```py
json.load(data)

```

**举例**:

在这个例子中，我们首先使用 [open()函数](https://www.askpython.com/python/built-in-methods/python-open-method)加载 JSON 文件。之后，我们将引用 JSON 文件的对象传递给 load()函数，并将其反序列化为[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)的形式。

```py
import json

data = open('info.json',)

op = json.load(data)

print(op)
print("Datatype after de-serialization : " + str(type(op)))

```

**输出**:

```py
{"details": {"name": "YZ","subject": "Engineering","City": "Pune"}}
Datatype after de-serialization : <class 'dict'>

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！🙂