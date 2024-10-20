# Python json 模块

> 原文：<https://www.askpython.com/python-modules/python-json-module>

在深入 Python JSON 模块之前，让我们了解一下 JSON 是什么。JSON(JavaScript 对象符号)是一种标准化的格式，允许在互联网上交换数据。

由于这已经成为通过互联网进行信息交换的标准，任何 Python 应用程序使用这种格式发送和接收数据都是有意义的。

Python 内置的 **json** 模块是将 Python 对象转换成 json 对象的接口。

在本教程中，我们来看看 json 模块中一些最常用的方法。

* * *

## JSON 对象的格式

在进入模块细节之前，让我们理解 JSON 对象是由什么组成的。

这实际上非常类似于 Python 字典，其中有一组 **{Key: value}** 对。唯一的小区别是 JSON 对象有一个左花括号和右花括号。

下面是一个简单的 JSON 对象的例子

```py
{
    "name": "John",
    "age": 42,
    "married": True,
    "qualifications": ["High School Diploma", "Bachelors"]
}

```

JSON 对象可以由各种属性组成，包括字符串、整数甚至列表。

现在我们知道了 JSON 对象是由什么组成的，让我们看看 Python **json** 模块的方法。

* * *

## 导入 Python json 模块

Python 已经准备好了 **json** 模块，所以不需要使用 pip 进行安装。

要导入这个模块，只需输入

```py
import json

```

## json . dumps()–构造一个 JSON 对象

我们可以使用`json.dumps()`方法将 Python 对象编码成 JSON 对象。

你可以把`dumps()`想象成将 Python 对象序列化成 Python JSON 对象并返回一个字符串。如果您希望通过互联网传输数据，这是必需的。

下表列出了不同 Python 对象的编码数据。

| 计算机编程语言 | JSON |
| --- | --- |
| **格言** | 目标 |
| **列表**，**元组** | 排列 |
| **str** | 线 |
| **int** ， **float** ，**int**–&**float**派生枚举 | 数字 |
| **真** | 真实的 |
| **假** | 错误的 |
| **无** | 空 |

它将任何可以序列化的 Python 对象作为参数，并返回一个字符串。

格式:

```py
json_object = json.dumps(serializable_object)

```

这里，`serializable_object`是一个 Python 对象，如列表、字符串等，可以序列化。它不能是函数/lambda 等。

```py
import json

python_object = ['Hello', 'from', 'AskPython', 42]

json_object = json.dumps(python_object)

print(type(json_object), json_object)

```

**输出**

```py
<class 'str'> ["Hello", "from", "AskPython", 42]

```

如果对象不可序列化，该方法将引发一个`TypeError`。

```py
>>> import json
>>> a = lambda x : x * 2
>>> a(2)
4
>>> json.dumps(a)
Traceback (most recent call last):
    raise TypeError(f'Object of type {o.__class__.__name__}
TypeError: Object of type function is not JSON serializable

```

### 字典的分类关键字

如果我们将一个 [Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)传递给`json.dumps()`，我们可以指定另一个参数`sort_keys`，这将使 Python json 对象拥有排序的键。

```py
import json

dict_obj = {1:"one", 20: "twenty", 5:"five"}

json_obj = json.dumps(dict_obj, sort_keys = True)

print(json_obj)

```

输出

```py
{"1": "one", "5": "five", "20": "twenty"}

```

我们的输出确实有排序的键。

**注**:由于数字被编码成 JSON，所以被转换成字符串。使用适当的方法，它将被正确地反序列化回整数。

### 漂亮地打印 Python JSON 对象

我们可以使用`json.dumps()`的`indent`参数来指定缩进级别。通常，`indent = 4`会让输出看起来很不错。

```py
import json

dict_obj = {1:"one", 20: "twenty", 5:"five"}

json_obj = json.dumps(dict_obj, sort_keys = True, indent = 4)

print(json_obj)

```

**输出**

```py
{
    "1": "one",
    "5": "five",
    "20": "twenty"
}

```

### JSON . Dump()–转储到文件中

我们也可以将一个对象转储到一个文件中，如果你希望以后使用它，使用另一种方法`json.dump()`。

**格式**:

```py
json.dump(data, file_object)

```

`json.dump()`方法接收数据并将其写入一个文件对象。

因此您可以打开一个新文件，并使用`json.dump()`写入该文件对象

```py
import json

python_object = ['Hello', 'from', 'AskPython', 42]

with open("sample.json", "w") as wf:
    json.dump(python_object, wf)

```

**输出**

```py
[email protected] $ cat sample.json
["Hello", "from", "AskPython", 42]

```

如您所见，Python 对象确实被转储到文件中。

现在，让我们把第一个例子中显示的 JSON 对象存储到一个文件中。

```py
import json

json_object = {
    "name": "John",
    "age": 42,
    "married": True,
    "qualifications": ["High School Diploma", "Bachelors"]
}

with open("sample.json", "w") as wf:
    json.dump(json_object, wf)

```

**输出**

```py
[email protected] $ cat sample.json
{"name": "John", "age": 42, "married": true, "qualifications": ["High School Diploma", "Bachelors"]}

```

* * *

## 反序列化 JSON 对象

类似于将 Python 对象编码成 JSON 对象，我们也可以反过来，将 JSON 对象转换成 Python 对象。这被称为**反序列化**。

我们可以使用方法`json.loads()`和`json.load()`来做到这一点，类似于`json.dumps()`和`json.dump()`。

### json.loads()

这将使用`json.dumps()`编码的 json 对象转换回 Python 对象。

```py
import json

python_object = ['Hello', 'from', 'AskPython', 42]

encoded_object = json.dumps(python_object)

decoded_object = json.loads(encoded_object)

print(type(decoded_object), decoded_object)

```

**输出**

```py
<class 'list'> ['Hello', 'from', 'AskPython', 42]

```

我们已经成功地获得了我们的旧列表对象！

### JSON . load()–从文件反序列化

这执行了与`json.dump()`相反的操作，将 json 对象从文件转换回 Python 对象。

让我们拿我们的`sample.json`文件，用这个方法取回数据。

```py
import json

with open("sample.json", "r") as rf:
    decoded_data = json.load(rf)

print(decoded_data)

```

**输出**

```py
{'name': 'John', 'age': 42, 'married': True, 'qualifications': ['High School Diploma', 'Bachelors']}

```

事实上，我们再次获得了我们存储在文件中的旧的 JSON 对象！

现在我们已经介绍了这个模块最常用的方法，让我们进入下一步:创建我们自己的 JSON 编码器！

* * *

## 创建我们自己的 JSON 编码器

`json`模块使用一个名为`json.JSONEncoder`的编码器，它使用上表中的规则对 Python 对象进行编码。

然而，它并不编码所有的 Python 对象，根据我们面临的问题，我们可能需要编写自己的 JSON 编码器，以一种特殊的方式编码这些对象。

为此，我们必须编写自定义的编码器类。姑且称之为`MyEncoder`。这个[必须扩展`json.JSONEncoder`类的](https://www.askpython.com/python/oops/inheritance-in-python)，以增加它现有的特性。

在这个演示中，我们将使用 numpy 数组，并将它们转换成 Python JSON 对象。现在 json 模块默认情况下不能处理 numpy 数组，所以如果你试图在没有扩展类的情况下转换 numpy 数组，你会得到一个 TypeError:

```py
TypeError: Object of type ndarray is not JSON serializable

```

让我们编写这个类，通过在我们的`default()`处理程序方法中将 numpy 数组转换成 Python 列表，将它序列化并编码成 json 对象。

```py
import json
import numpy as np

class MyEncoder(json.JSONEncoder):
    # Handles the default behavior of
    # the encoder when it parses an object 'obj'
    def default(self, obj):
        # If the object is a numpy array
        if isinstance(obj, np.ndarray):
            # Convert to Python List
            return obj.tolist()
        else:
            # Let the base class Encoder handle the object
            return json.JSONEncoder.default(self, obj)

# Numpy array of floats
a = np.arange(1, 10, 0.5)
print(type(a), a)

# Pass our encoder to json.dumps()
b = json.dumps(a, cls=MyEncoder)
print(b)

```

最后，我们通过将类名传递给`json.dumps()`的`cls`参数对其进行编码。

因此，编码调用将是:

```py
json_object = json.dumps(python_object, cls=MyEncoder)

```

**输出**

```py
<class 'numpy.ndarray'> [1\.  1.5 2\.  2.5 3\.  3.5 4\.  4.5 5\.  5.5 6\.  6.5 7\.  7.5 8\.  8.5 9\.  9.5]
[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]

```

事实上，我们的自定义编码器现在可以将 numpy 数组转换成 JSON 对象！我们现在已经完成了第一个复合编码器。

您可以扩展这个功能，为您的特定用例编写不同的编码器！

* * *

## 结论

在本文中，我们学习了如何使用 Python 的`json`模块来进行各种涉及 JSON 对象的操作。

* * *

## 参考

*   JSON 模块上的官方 Python 文档

* * *