# python 101:JSON 使用简介

> 原文：<https://www.blog.pythonlibrary.org/2020/09/15/python-101-an-intro-to-working-with-json/>

JavaScript Object Notation，通常被称为 JSON，是一种受 JavaScript object literal 语法启发的轻量级数据交换格式。JSON 对于人类来说很容易读写。也便于计算机解析生成。JSON 用于存储和交换数据，与 XML 的使用方式非常相似。

Python 有一个名为`json`的内置库，可以用来创建、编辑和解析 JSON。您可以在此阅读关于该图书馆的所有信息:

*   [https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html)

了解 JSON 的样子可能会有所帮助。这里有一个来自 https://json.org 的 JSON 的例子:

```py
{"menu": {
  "id": "file",
  "value": "File",
  "popup": {
    "menuitem": [
      {"value": "New", "onclick": "CreateNewDoc()"},
      {"value": "Open", "onclick": "OpenDoc()"},
      {"value": "Close", "onclick": "CloseDoc()"}
    ]
  }
}}
```

从 Python 的角度来看，这个 JSON 是一个嵌套的 Python 字典。你会发现 JSON 总是被翻译成某种原生 Python 数据类型。在本文中，您将了解以下内容:

*   编码 JSON 字符串
*   解码 JSON 字符串
*   将 JSON 保存到磁盘
*   从磁盘加载 JSON
*   使用`json.tool`验证 JSON

JSON 是一种非常流行的格式，经常在 web 应用程序中使用。您会发现，知道如何使用 Python 与 JSON 交互在您自己的工作中很有用。

我们开始吧！

### 编码 JSON 字符串

Python 的`json`模块使用`dumps()`将对象序列化为字符串。`dumps()`中的“s”代表“弦”。通过在一些代码中使用`json`模块，可以更容易地看出这是如何工作的:

```py
>>> import json
>>> j = {"menu": {
...   "id": "file",
...   "value": "File",
...   "popup": {
...     "menuitem": [
...       {"value": "New", "onclick": "CreateNewDoc()"},
...       {"value": "Open", "onclick": "OpenDoc()"},
...       {"value": "Close", "onclick": "CloseDoc()"}
...     ]
...   }
... }}
>>> json.dumps(j)
'{"menu": {"id": "file", "value": "File", "popup": {"menuitem": [{"value": "New", '
'"onclick": "CreateNewDoc()"}, {"value": "Open", "onclick": "OpenDoc()"}, '
'{"value": "Close", "onclick": "CloseDoc()"}]}}}'
```

这里使用了`json.dumps()`，它将 Python 字典转换成 JSON 字符串。该示例的输出被修改为换行打印。否则字符串将全部在一行上。

现在您已经准备好学习如何将一个对象写入磁盘了！

### 将 JSON 保存到磁盘

Python 的`json`模块使用`dump()`函数将对象序列化或编码为 JSON 格式的流，成为类似文件的对象。Python 中的类文件对象是指使用 Python 的`io`模块创建的文件处理程序或对象。

继续创建一个名为`create_json_file.py`的文件，并向其中添加以下代码:

```py
# create_json_file.py

import json

def create_json_file(path, obj):
    with open(path, 'w') as fh:
        json.dump(obj, fh)

if __name__ == '__main__':
    j = {"menu": {
        "id": "file",
        "value": "File",
        "popup": {
          "menuitem": [
            {"value": "New", "onclick": "CreateNewDoc()"},
            {"value": "Open", "onclick": "OpenDoc()"},
            {"value": "Close", "onclick": "CloseDoc()"}
          ]
        }
      }}
    create_json_file('test.json', j)
```

在本例中，您使用了`json.dump()`，它用于写入文件或类似文件的对象。它将写入文件处理程序`fh`。

现在您可以学习如何解码 JSON 字符串了！

### 解码 JSON 字符串

JSON 字符串的解码或反序列化是通过`loads()`方法完成的。`loads()`是`dumps()`的配套功能。下面是它的用法示例:

```py
>>> import json
>>> j_str = """{"menu": {
...   "id": "file",
...   "value": "File",
...   "popup": {
...     "menuitem": [
...       {"value": "New", "onclick": "CreateNewDoc()"},
...       {"value": "Open", "onclick": "OpenDoc()"},
...       {"value": "Close", "onclick": "CloseDoc()"}
...     ]
...   }
... }}
... """
>>> j_obj = json.loads(j_str)
>>> type(j_obj)
<class 'dict'>
```

这里，您将前面的 JSON 代码重新创建为 Python 多行字符串。然后使用`json.loads()`加载 JSON 字符串，将它转换成 Python 对象。在这种情况下，它将 JSON 转换为 Python 字典。

现在您已经准备好学习如何从文件加载 JSON 了！

### 从磁盘加载 JSON

使用`json.load()`从文件加载 JSON。这里有一个例子:

```py
# load_json_file.py

import json

def load_json_file(path):
    with open(path) as fh:
        j_obj = json.load(fh)
    print(type(j_obj))

if __name__ == '__main__':
    load_json_file('example.json')
```

在这段代码中，您像以前看到的那样打开传入的文件。然后您将文件处理程序`fh`传递给`json.load()`，这将把 JSON 转换成 Python 对象。

还可以使用 Python 的`json`模块来验证 JSON。接下来你会知道如何去做。

### 使用`json.tool`验证 JSON

Python 的`json`模块提供了一个工具，您可以在命令行上运行该工具来检查 JSON 是否具有正确的语法。这里有几个例子:

```py
$ echo '{1.2:3.4}' | python -m json.tool
Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
$ echo '{"1.2":3.4}' | python -m json.tool
{
    "1.2": 3.4
}
```

第一个调用将字符串`'{1.2:3.4}'`传递给`json.tool`，这告诉您 JSON 代码有问题。第二个示例向您展示了如何解决这个问题。当固定字符串被传递给`json.tool`时，它会“漂亮地打印”出 JSON，而不是发出一个错误。

### 包扎

在处理 web APIs 和 web 框架时，经常使用 JSON 格式。Python 语言为您提供了一个很好的工具，可以用来将 JSON 转换成 Python 对象，然后在`json`库中再转换回来。

在本章中，您学习了以下内容:

*   编码 JSON 字符串
*   解码 JSON 字符串
*   将 JSON 保存到磁盘
*   从磁盘加载 JSON
*   使用`json.tool`验证 JSON

现在，您有了另一个可以使用 Python 的有用工具。只要稍加练习，您很快就能使用 JSON 了！