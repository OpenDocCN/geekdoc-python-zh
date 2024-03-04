# JSON 是什么

> 原文：<https://www.pythonforbeginners.com/json/what-is-json>

### JSON 是什么？

JSON (JavaScript Object Notation)是一种紧凑的、基于文本的计算机数据交换格式。json 的官方互联网媒体类型是 application/json，JSON 文件扩展名是。json

### JSON 建立在两种结构之上:

名称/值对的集合
值的有序列表。

JSON 采用这些形式:对象、数组、值、字符串、数字

#### 目标

名称/值对的无序集合。以{开头，以}结尾。每个名称后接:(冒号)名称/值对由，(逗号)分隔。

#### 排列

值的有序集合。开始于[结束于]。值由(逗号)分隔。

#### 价值

可以是双引号中的字符串、数字、true 或 false 或 null，也可以是对象或数组。

#### 线

零个或多个 Unicode 字符的序列，用双引号括起来，使用反斜杠转义。

#### 数字

整数、长整型、浮点型

以下示例显示了描述一个人的对象的 JSON 表示:

```py
{
    "firstName": "John",
    "lastName": "Smith",
    "age": 25,
    "address": {
        "streetAddress": "21 2nd Street",
        "city": "New York",
        "state": "NY",
        "postalCode": "10021"
    },
    "phoneNumber": [
        {
            "type": "home",
            "number": "212 555-1234"
        },
        {
            "type": "fax",
            "number": "646 555-4567"
        }
    ]
}

```

JSON 数据结构直接映射到 Python 数据类型，因此这是一个强大的工具，可以直接访问数据，而不必编写任何 XML 解析代码。JSON 曾经像字典一样加载到 Python 中。