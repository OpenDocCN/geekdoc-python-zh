# 如何用 Python 读取 JSON 文件

> 原文：<https://www.askpython.com/python/examples/read-a-json-file-in-python>

在本文中，我们将看看如何用 Python 读取 JSON 文件。

通常，您可能希望轻松地从相关的 json 文件中读取和解析 json 数据。让我们找出一些方法，通过这些方法我们可以很容易地读取和提取出这些数据！

* * *

## 方法 1:使用 json.load()读取 Python 中的 json 文件

[json 模块](https://www.askpython.com/python-modules/python-json-module)是 Python3 中的内置模块，使用`json.load()`为我们提供了 json 文件处理能力。

使用这种方法，我们可以在直接读取 Python 中的 JSON 文件后构造一个 Python 对象。

假设`sample.json`是一个 JSON 文件，内容如下:

```py
{
"name": "AskPython",
"type": "website",
"language": "Python"
}

```

我们可以使用下面的程序将 json 对象加载到 Python 对象中。我们现在可以使用字典的 **{key: value}** 对映射轻松访问它！

```py
import json

with open("sample.json", "r") as rf:
    decoded_data = json.load(rf)

print(decoded_data)
# Check is the json object was loaded correctly
try:    
    print(decoded_data["name"])
except KeyError:
    print("Oops! JSON Data not loaded correctly using json.loads()")

```

**输出**

```py
{'name': 'AskPython', 'type': 'website', 'language': 'Python'}
AskPython

```

事实上，我们能够从文件中正确加载 JSON 对象！

* * *

### 方法 2:对大型 json 文件使用 ijson

如果您的 JSON 文件足够大，以至于将全部内容放入内存的成本很高，那么更好的方法是使用`ijson`将文件内容转换成**流**。

流是对象的集合(就像 JSON 对象一样),只有在需要时才会加载到内存中。这意味着我们的数据加载器“懒惰地”加载数据，也就是说，只在需要的时候加载。

在处理大文件时，这可以降低内存需求。流的内容存储在一个临时缓冲区中，这使得处理千兆字节的 JSON 文件成为可能！

要安装`ijson`，请使用 pip！

```py
pip install ijson

```

现在，为了进行实验，我们将使用一个稍微小一点的 JSON 文件，因为下载千兆字节的数据非常耗时！

我将在[这个](https://pomber.github.io/covid19/timeseries.json)链接上使用 COVID timeseries JSON 文件。下载该文件，并将其重命名为`covid_timeseries.json`。文件大小必须约为 2 MB。

```py
import ijson

for prefix, type_of_object, value in ijson.parse(open("covid_timeseries.json")):
    print(prefix, type_of_object, value)

```

**样本输出(几行)**

```py
Yemen.item.date string 2020-4-13
Yemen.item map_key confirmed
Yemen.item.confirmed number 1
Yemen.item map_key deaths
Yemen.item.deaths number 0
Yemen.item map_key recovered
Yemen.item.recovered number 0
Yemen.item end_map None
Yemen.item start_map None
Yemen.item map_key date
Yemen.item.date string 2020-4-14
Yemen.item map_key confirmed
Yemen.item.confirmed number 1
Yemen.item map_key deaths
Yemen.item.deaths number 0
Yemen.item map_key recovered
Yemen.item.recovered number 0
Yemen.item end_map None
Yemen end_array None

```

这将打印巨大的 JSON 文件的内容，但是您可以保留一个计数器变量以避免打印整个文件。

虽然`ijson`可能很慢，但它似乎在较低的内存范围内运行。如果您正在处理大文件，可以尝试此模块。

* * *

## 结论

在本文中，我们学习了如何用 Python 读取 JSON 文件。我们还简要介绍了如何使用`ijson`处理大量数据。

* * *

## 参考

*   关于处理大型 json 文件的 StackOverflow 问题

* * *