# 用 Python 解析 JSON

> 原文：<https://www.pythonforbeginners.com/json/parsingjson>

## 概观

对 HTTP API 的请求通常只是带有一些查询参数的 URL。

## API 响应

我们从 API 得到的响应是数据，数据可以有各种格式，最流行的是 XML 和 JSON。

许多 HTTP APIs 支持多种响应格式，因此开发人员可以选择他们更容易解析的格式。

## 入门指南

首先，让我们创建一个简单的数据结构并放入一些数据。

首先，我们将 json 模块导入到程序中。

```py
import json

# Create a data structure
data = [ { 'Hola':'Hello', 'Hoi':"Hello", 'noun':"hello" } ]
```

要将数据打印到屏幕上，非常简单:

```py
print 'DATA:', (data)
```

当我们如上打印数据时，我们将看到以下输出:

```py
DATA: [{'noun': 'hello', 'Hola': 'Hello', 'Hoi': 'Hello'}]
```

## JSON 函数

当您在 Python 中使用 JSON 时，我们可以利用不同函数

## Json 转储

json.dumps 函数采用 Python 数据结构，并将其作为 json 字符串返回。

```py
json_encoded = json.dumps(data)

# print to screen

print json_encoded

OUTPUT:

[{"noun": "hello", "Hola": "Hello", "Hoi": "Hello"}]
```

## Json 加载

json.loads()函数接受一个 json 字符串，并将其作为 Python 数据
结构返回。

```py
decoded_data = json.loads(json_encoded)

# print to screen

print decoded_data

OUTPUT:

[{u'noun': u'hello', u'Hola': u'Hello', u'Hoi': u'Hello’}]
```