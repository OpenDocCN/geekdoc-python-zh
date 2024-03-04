# 用 Python 编码 JSON

> 原文：<https://www.pythonforbeginners.com/json/parsing-json-python>

Python 预装了 JSON 编码器和解码器，使得在应用程序中使用 JSON 变得非常简单

对 JSON 进行编码的最简单的方法是使用字典。这个基本字典保存各种数据类型的随机值。

```py
 data = {
    a: 0,
    b: 9.6,
    c: "Hello World",
    d: {
        a: 4
    }
} 

```

然后，我们使用 json.dumps()将字典转换成 json 对象。

```py
 import json

data = {
    a: 0,
    b: 9.6,
    c: "Hello World",
    d: {
        a: 4
    }
}

json_data = json.dumps(data)
print(json_data) 

```

这将打印出来

```py
 {"c": "Hello World", "b": 9.6, "d": {"e": [89, 90]}, "a": 0} 

```

请注意默认情况下键是如何排序的，您必须像这样将 sort_keys=True 参数添加到 json.dumps()中。

```py
 import json

data = {
    a: 0,
    b: 9.6,
    c: "Hello World",
    d: {
        a: 4
    }
}

json_data = json.dumps(data, sort_keys=True)
print(json_data) 

```

然后输出排序后的关键字。

```py
 {"a": 0, "b": 9.6, "c": "Hello World", "d": {"e": [89, 90]}} 

```