# python Pretty print JSON——完整指南

> 原文：<https://www.askpython.com/python/examples/pretty-print-json-in-python>

在本文中，我们将学习如何打印一个 JSON 文件。你可能会被“Python Pretty Print JSON”这个标题搞糊涂。让我们也理解这一点。在开始之前，我们首先需要理解一个 JSON 文件。

## 了解 JSON 文件格式

JSON 是一种语法或格式，以文本形式存储数据并在网络上传输或交换。机器和人类一样容易读写。许多 API 和数据库都使用这种格式。JSON 代表 JavaScript 对象符号。其编写的编程代码的文件扩展名是`.json`。让我们来看看 JSON 文件的一些关键特性。

*   JSON 提供了一种易于使用的方法。
*   它描述了一种最小化的数据形式，使用非常少的空间，因此速度非常快。
*   它是开源的，可以免费使用。
*   它给出清晰、兼容、易读的结果。
*   JSON 没有任何帮助他独立运行的依赖项。
*   将数据写入 JSON 文件的语法与 Python 字典完全相同。

让我们看看下面的 JSON 文件。

```py
{
  "name": "mine",
  "version": "1.0.0",
  "description": "hey ",
  "main": "2.js",
  "dependencies": {
    "express": "^4.17.1",
    "mongoose": "^5.12.9",
    "nodemon": "^2.0.7",
    "pug": "^3.0.2"
  }}

```

在上面的例子中，你可以看到一个 JSON 文件很可能看起来像一个 [**python 字典**](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial) ，包含一些键*(名称、版本、描述、主、依赖项)*和值 *(mine，1.0.0，hey，2.js，{"express": "^4.17.1 "，" mongose ":" ^5.12.9″,"nodemon": " ^2.0.7 "，* "pug": "^3.0.2" })。

它也是结构化的。如果我们使用 ms_doc 或在记事本中打开它，我们可以看到同样的结果。但是使用 python 的打印函数，它将按如下非结构化方式打印。我们需要以一种结构化的格式 **打印相同的内容，这是为了美观或者以一种更漂亮的方式**。

```py
{{
  "name": "mine", "version": "1.0.0","description": "hey ","main": "2.js","dependencies": {"express": "^4.17.1","mongoose": "^5.12.9","nodemon": "^2.0.7","pug": "^3.0.2"
  }}

```

## 正在读取 JSON 文件

我们的第一步将是 [**读取 JSON 文件**](https://www.askpython.com/python/examples/read-a-json-file-in-python) 。让我们也看看我们的代码片段。

```py
import json
with open(file="/content/package.json", mode='r') as our_file:
    output = json.load(our_file)
    print(output)

```

在上面的代码片段中，我们使用

*   打开 JSON 文件的关键字。
*   以只读模式打开文件的`mode='r'`参数。
*   `json.load()`方法将我们的文件加载到名为 output 的对象文件中。

在打印相同的内容后，我们得到如下所示的 JSON 文件。

```py
{'name': 'mine', 'version': '1.0.0', 'description': 'hey ', 'main': '2.js', 'dependencies': {'express': '^4.17.1', 'mongoose': '^5.12.9', 'nodemon': '^2.0.7', 'pug': '^3.0.2'}, 'devDependencies': {}, 'scripts': {'test': 'echo "Error: no test specified" && exit 1'}, 'author': 'brand', 'license': 'ISC'}

```

您可以看到，通过正常打印，我们得到的输出应该不是格式化的、漂亮的或结构化的。我们希望它印刷精美，结构良好。我们使用关键字 Pretty 只是为了同样的目的。(漂亮就是漂亮的意思。我们需要以一种漂亮的方式打印 JSON 文件。)

我们可以通过一些特殊的方法或者使用一些特殊的库来得到我们期望的结果。让我们拥有它。

### 使用 json.dumps()方法打印 JSON

```py
json.dump( object_name, indent = "your value")

```

上面描述了 [**json.dumps()方法的语法**](https://www.askpython.com/python/examples/serialize-deserialize-json) `.` 我们需要传递加载 json 文件的对象名，并传递指定我们想要用来缩进 json 的空格数的 indent 值。我们只需要添加这一行就可以打印出更漂亮的 JSON 文件。让我们快速看看下面。

```py
with open(file="/content/package.json", mode='r') as our_file:
    output = json.load(our_file)
    pretty_output = json.dumps(output, indent=4)
    print(pretty_output)

```

我们可以得到如下打印的 JSON 文件。

```py
{
    "name": "mine",
    "version": "1.0.0",
    "description": "hey ",
    "main": "2.js",
    "dependencies": {
        "express": "^4.17.1",
        "mongoose": "^5.12.9",
        "nodemon": "^2.0.7",
        "pug": "^3.0.2"
    },
    "devDependencies": {},
    "scripts": {
        "test": "echo \"Error: no test specified\" && exit 1"
    },
    "author": "brand",
    "license": "ISC"
}

```

### 打印 json 使用

`pprint module`提供了 Python 中的 [**pprint()方法**](https://www.askpython.com/python-modules/pprint-module) ，用于以一种可读且漂亮的方式打印数据。它是标准库的一部分，强烈推荐与 API 请求和数据库一起用于代码调试。让我们看看如何在我们的代码片段中使用相同的代码。

```py
import json
from pprint import pprint

with open("/content/package.json", 'r') as our_file:
    output = json.load(our_file)
    pprint(output)

```

在上面的代码片段中，我们已经像以前一样加载了 JSON 文件，使用了`json.load()`方法，然后**将序列化的对象`our_file`传递给了`pprint()`方法**，这也允许我们打印结构化的漂亮输出。

```py
{'author': 'brand',
 'dependencies': {'express': '^4.17.1',
                  'mongoose': '^5.12.9',
                  'nodemon': '^2.0.7',
                  'pug': '^3.0.2'},
 'description': 'hey ',
 'devDependencies': {},
 'license': 'ISC',
 'main': '2.js',
 'name': 'mine',
 'scripts': {'test': 'echo "Error: no test specified" && exit 1'},
 'version': '1.0.0'}

```

## 摘要

今天，我们讨论了使用 python 的漂亮打印。我们学习了如何以更漂亮的方式打印 JSON 文件。我们也用了两种方法，得到了相同的结果。希望你也喜欢它。我们必须带着一些更有趣的话题再次访问。