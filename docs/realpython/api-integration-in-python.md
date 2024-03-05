# Python 和 REST APIs:与 Web 服务交互

> 原文：<https://realpython.com/api-integration-in-python/>

网上有数量惊人的数据。许多**网络服务**，如 YouTube 和 GitHub，通过**应用编程接口(API)** 让第三方应用程序可以访问它们的数据。构建 API 最流行的方式之一是 **REST** 架构风格。Python 提供了一些很棒的工具，不仅可以从 REST API 获取数据，还可以构建自己的 Python REST APIs。

在本教程中，您将学习:

*   什么是 **REST** 架构
*   REST APIs 如何提供对 web 数据的访问
*   如何使用 **`requests`** 库消费 REST APIs 中的数据
*   构建 REST API 需要采取什么步骤
*   一些流行的 **Python 工具**用于构建 REST APIs

通过使用 Python 和 REST APIs，您可以检索、解析、更新和操作您感兴趣的任何 web 服务提供的数据。

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

## REST 架构

**REST** 代表[**re**presentation**s**state**t**transfer](https://en.wikipedia.org/wiki/Representational_state_transfer)是一种软件架构风格，它定义了网络上[客户机和服务器](https://en.wikipedia.org/wiki/Client%E2%80%93server_model)通信的模式。REST 为软件架构提供了一组约束，以提高系统的性能、可伸缩性、简单性和可靠性。

REST 定义了以下架构约束:

*   **无状态:**服务器不会在来自客户端的请求之间维护任何状态。
*   **客户机-服务器:**客户机和服务器必须相互解耦，允许各自独立开发。
*   **可缓存:**从服务器检索的数据应该可以被客户端或服务器缓存。
*   **统一接口:**服务器将提供统一的接口来访问资源，而不需要定义它们的表示。
*   **分层系统:**客户端可以通过[代理](https://en.wikipedia.org/wiki/Proxy_server)或[负载均衡器](https://en.wikipedia.org/wiki/Load_balancing_(computing))等其他层间接访问服务器上的资源。
*   **按需编码(可选):**服务器可能会将自己可以运行的代码转移到客户端，比如针对单页面应用的 [JavaScript](https://realpython.com/python-vs-javascript/) 。

注意，REST 不是一个规范，而是一套关于如何构建网络连接软件系统的指南。

[*Remove ads*](/account/join/)

## REST APIs 和 Web 服务

REST web 服务是任何遵守 REST 架构约束的 web 服务。这些 web 服务通过 API 向外界公开它们的数据。REST APIs 通过公共 web URLs 提供对 web 服务数据的访问。

例如，下面是 GitHub 的 REST API 的 URL 之一:

```py
https://api.github.com/users/<username>
```

这个 URL 允许您访问特定 GitHub 用户的信息。您通过向特定的 URL 发送一个 [HTTP 请求](https://realpython.com/python-https/#what-is-http)并处理响应来访问来自 REST API 的数据。

### HTTP 方法

REST APIs 监听 [HTTP 方法](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)，如`GET`、`POST`和`DELETE`，以了解在 web 服务的资源上执行哪些操作。**资源**是 web 服务中可用的任何数据，可以通过对 REST API 的 **HTTP 请求**来访问和操作。HTTP 方法告诉 API 在资源上执行哪个操作。

虽然有许多 HTTP 方法，但下面列出的五种方法是 REST APIs 中最常用的:

| HTTP 方法 | 描述 |
| --- | --- |
| `GET` | 检索现有资源。 |
| `POST` | 创建新资源。 |
| `PUT` | 更新现有资源。 |
| `PATCH` | 部分更新现有资源。 |
| `DELETE` | 删除资源。 |

REST API 客户端应用程序可以使用这五种 HTTP 方法来管理 web 服务中资源的状态。

### 状态代码

一旦 REST API 接收并处理一个 HTTP 请求，它将返回一个 **HTTP 响应**。这个响应中包含一个 **HTTP 状态代码**。此代码提供了有关请求结果的信息。向 API 发送请求的应用程序可以检查状态代码，并根据结果执行操作。这些操作可能包括处理错误或向用户显示成功消息。

下面是 REST APIs 返回的最常见的状态代码列表:

| 密码 | 意义 | 描述 |
| --- | --- | --- |
| `200` | 好 | 请求的操作成功。 |
| `201` | 创造 | 创建了新资源。 |
| `202` | 可接受的 | 请求已收到，但尚未进行修改。 |
| `204` | 没有内容 | 请求成功，但响应没有内容。 |
| `400` | 错误的请求 | 请求格式不正确。 |
| `401` | 未经授权的 | 客户端无权执行请求的操作。 |
| `404` | 未发现 | 找不到请求的资源。 |
| `415` | 不支持的媒体类型 | 服务器不支持请求数据格式。 |
| `422` | 不可处理实体 | 请求数据格式正确，但包含无效或丢失的数据。 |
| `500` | 内部服务器错误 | 处理请求时，服务器出现错误。 |

这十个状态代码仅代表可用的 [HTTP 状态代码](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes)的一小部分。状态代码根据结果的类别进行编号:

| 代码范围 | 种类 |
| --- | --- |
| `2xx` | 成功的手术 |
| `3xx` | 重寄 |
| `4xx` | 客户端错误 |
| `5xx` | 服务器错误 |

当使用 REST APIs 时，HTTP 状态代码很方便，因为您经常需要根据请求的结果执行不同的逻辑。

### API 端点

REST API 公开了一组公共 URL，客户端应用程序使用这些 URL 来访问 web 服务的资源。在 API 的上下文中，这些 URL 被称为**端点**。

为了帮助澄清这一点，请看下表。在这个表格中，您将看到一个假想的 [CRM](https://en.wikipedia.org/wiki/Customer_relationship_management) 系统的 API 端点。这些端点用于代表系统中潜在`customers`的客户资源:

| HTTP 方法 | API 端点 | 描述 |
| --- | --- | --- |
| `GET` | `/customers` | 获取客户名单。 |
| `GET` | `/customers/<customer_id>` | 获得单个客户。 |
| `POST` | `/customers` | 创建新客户。 |
| `PUT` | `/customers/<customer_id>` | 更新客户。 |
| `PATCH` | `/customers/<customer_id>` | 部分更新客户。 |
| `DELETE` | `/customers/<customer_id>` | 删除客户。 |

上面的每个端点基于 HTTP 方法执行不同的操作。

**注意:**为简洁起见，省略了端点的基本 URL。实际上，您需要完整的 URL 路径来访问 API 端点:

```py
https://api.example.com/customers
```

这是您用来访问此端点的完整 URL。基本 URL 是除了`/customers`之外的所有内容。

你会注意到一些端点的末尾有`<customer_id>`。这个符号意味着您需要在 URL 后面添加一个数字`customer_id`来告诉 REST API 您想要使用哪个`customer`。

上面列出的端点仅代表系统中的一种资源。生产就绪的 REST APIs 通常有数十甚至数百个不同的端点来管理 web 服务中的资源。

[*Remove ads*](/account/join/)

## REST 和 Python:消费 API

为了编写与 REST APIs 交互的代码，大多数 Python 开发人员求助于 [`requests`](https://realpython.com/python-requests/) 来发送 HTTP 请求。这个库抽象出了进行 HTTP 请求的复杂性。这是少数几个值得作为标准库的一部分来对待的项目之一。

要开始使用`requests`，需要先安装。您可以使用 [`pip`](https://realpython.com/what-is-pip/) 来安装它:

```py
$ python -m pip install requests
```

现在您已经安装了`requests`，您可以开始发送 HTTP 请求了。

### 获取

`GET`是使用 REST APIs 时最常用的 HTTP 方法之一。该方法允许您从给定的 API 中检索资源。`GET`是一个**只读的**操作，所以您不应该使用它来修改现有的资源。

为了测试本节中的`GET`和其他方法，您将使用一个名为 [JSONPlaceholder](https://jsonplaceholder.typicode.com/) 的服务。这个免费服务提供了虚假的 API 端点，这些端点发送回`requests`可以处理的响应。

为了进行测试，启动 [Python REPL](https://realpython.com/interacting-with-python/) 并运行以下命令向 JSONPlaceholder 端点发送一个`GET`请求:

>>>

```py
>>> import requests
>>> api_url = "https://jsonplaceholder.typicode.com/todos/1"
>>> response = requests.get(api_url)
>>> response.json()
{'userId': 1, 'id': 1, 'title': 'delectus aut autem', 'completed': False}
```

这段代码调用`requests.get()`向`/todos/1`发送一个`GET`请求，后者用 ID 为`1`的`todo`项进行响应。然后可以在`response`对象上调用 [`.json()`](https://docs.python-requests.org/en/master/user/quickstart/#json-response-content) 来查看从 API 返回的数据。

响应数据被格式化为 [JSON](https://www.json.org/json-en.html) ，一个类似于 [Python 字典](https://realpython.com/python-dicts/)的键值存储。这是一种非常流行的数据格式，也是大多数 REST APIs 事实上的交换格式。

除了从 API 查看 JSON 数据，您还可以查看关于`response`的其他内容:

>>>

```py
>>> response.status_code
200

>>> response.headers["Content-Type"]
'application/json; charset=utf-8'
```

在这里，您访问`response.status_code`来查看 HTTP 状态代码。您还可以使用`response.headers`查看响应的 [HTTP 头](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers)。这个字典包含关于响应的元数据，比如响应的`Content-Type`。

### 帖子

现在，看看如何使用 REST API 的`requests`到`POST`数据来创建新资源。您将再次使用 JSONPlaceholder，但是这次您将在请求中包含 JSON 数据。这是您将发送的数据:

```py
{ "userId":  1, "title":  "Buy milk", "completed":  false }
```

该 JSON 包含一个新的`todo`项目的信息。回到 Python REPL，运行下面的代码来创建新的`todo`:

>>>

```py
>>> import requests
>>> api_url = "https://jsonplaceholder.typicode.com/todos"
>>> todo = {"userId": 1, "title": "Buy milk", "completed": False}
>>> response = requests.post(api_url, json=todo)
>>> response.json()
{'userId': 1, 'title': 'Buy milk', 'completed': False, 'id': 201}

>>> response.status_code
201
```

这里，您调用`requests.post()`在系统中创建新的`todo`。

首先，创建一个包含您的`todo`数据的字典。然后你将这个字典传递给`requests.post()`的`json`关键字参数。当您这样做时，`requests.post()`自动将请求的 HTTP 头`Content-Type`设置为`application/json`。它还将`todo`序列化为一个 JSON 字符串，并将其附加到请求体中。

如果您不使用`json`关键字参数来提供 JSON 数据，那么您需要相应地设置`Content-Type`并手动序列化 JSON。下面是前面代码的等效版本:

>>>

```py
>>> import requests
>>> import json >>> api_url = "https://jsonplaceholder.typicode.com/todos"
>>> todo = {"userId": 1, "title": "Buy milk", "completed": False}
>>> headers =  {"Content-Type":"application/json"} >>> response = requests.post(api_url, data=json.dumps(todo), headers=headers) >>> response.json()
{'userId': 1, 'title': 'Buy milk', 'completed': False, 'id': 201}

>>> response.status_code
201
```

在这段代码中，您添加了一个包含设置为`application/json`的单个标题`Content-Type`的`headers`字典。这告诉 REST API 您正在发送带有请求的 JSON 数据。

然后调用`requests.post()`，但不是将`todo`传递给`json`参数，而是首先调用`json.dumps(todo)`来序列化它。在它被序列化之后，你把它传递给`data`关键字参数。`data`参数告诉`requests`请求中包含什么数据。您还可以将`headers`字典传递给`requests.post()`来手动设置 HTTP 头。

当您像这样调用`requests.post()`时，它与前面的代码具有相同的效果，但是您可以对请求进行更多的控制。

**注:** [`json.dumps()`](https://docs.python.org/3/library/json.html#json.dumps) 来自于标准库中的 [`json`](https://docs.python.org/3/library/json.html) 包。这个包提供了在 Python 中使用 [JSON 的有用方法。](https://realpython.com/python-json/)

一旦 API 响应，您就调用`response.json()`来查看 JSON。JSON 包括为新的`todo`生成的`id`。`201`状态代码告诉您一个新的资源已经创建。

[*Remove ads*](/account/join/)

### 放

除了`GET`和`POST`，`requests`还提供了对所有其他 HTTP 方法的支持，这些方法可以和 REST API 一起使用。下面的代码发送一个`PUT`请求，用新数据更新现有的`todo`。通过`PUT`请求发送的任何数据将完全替换`todo`的现有值。

您将使用与用于`GET`和`POST`相同的 JSONPlaceholder 端点，但是这次您将把`10`附加到 URL 的末尾。这告诉 REST API 您想要更新哪个`todo`:

>>>

```py
>>> import requests
>>> api_url = "https://jsonplaceholder.typicode.com/todos/10"
>>> response = requests.get(api_url)
>>> response.json()
{'userId': 1, 'id': 10, 'title': 'illo est ... aut', 'completed': True}

>>> todo = {"userId": 1, "title": "Wash car", "completed": True} >>> response = requests.put(api_url, json=todo) >>> response.json() {'userId': 1, 'title': 'Wash car', 'completed': True, 'id': 10}

>>> response.status_code
200
```

在这里，首先调用`requests.get()`来查看现有`todo`的内容。接下来，用新的 JSON 数据调用`requests.put()`来替换现有的待办事项值。调用`response.json()`时可以看到新的值。成功的`PUT`请求将总是返回`200`而不是`201`,因为您不是在创建一个新的资源，而是在更新一个现有的资源。

### 补丁

接下来，您将使用`requests.patch()`来修改现有`todo`上特定字段的值。`PATCH`与`PUT`的不同之处在于，它不会完全取代现有的资源。它只修改与请求一起发送的 JSON 中设置的值。

您将使用上一个示例中的相同的`todo`来测试`requests.patch()`。以下是当前值:

```py
{'userId': 1, 'title': 'Wash car', 'completed': True, 'id': 10}
```

现在您可以用新值更新`title`:

>>>

```py
>>> import requests
>>> api_url = "https://jsonplaceholder.typicode.com/todos/10"
>>> todo = {"title": "Mow lawn"} >>> response = requests.patch(api_url, json=todo) >>> response.json()
{'userId': 1, 'id': 10, 'title': 'Mow lawn', 'completed': True}

>>> response.status_code
200
```

当你调用`response.json()`时，你可以看到`title`被更新为`Mow lawn`。

### 删除

最后但同样重要的是，如果您想完全删除一个资源，那么您可以使用`DELETE`。下面是删除一个`todo`的代码:

>>>

```py
>>> import requests
>>> api_url = "https://jsonplaceholder.typicode.com/todos/10"
>>> response = requests.delete(api_url) >>> response.json()
{}

>>> response.status_code
200
```

您用一个 API URL 调用`requests.delete()`，该 URL 包含您想要删除的`todo`的 ID。这将向 REST API 发送一个`DELETE`请求，然后 REST API 移除匹配的资源。删除资源后，API 发送回一个空的 JSON 对象，表明资源已经被删除。

`requests`库是使用 REST APIs 的一个很棒的工具，也是 Python 工具箱中不可或缺的一部分。在下一节中，您将改变思路，考虑如何构建 REST API。

## REST 和 Python:构建 API

REST API 设计是一个巨大的话题，有很多层。如同技术领域的大多数事情一样，对于构建 API 的最佳方法有各种各样的观点。在本节中，您将看到一些在构建 API 时推荐遵循的步骤。

### 身份资源

构建 REST API 的第一步是识别 API 将管理的资源。通常将这些资源描述为复数名词，如`customers`、`events`或`transactions`。当您在 web 服务中识别不同的资源时，您将构建一个名词列表，描述用户可以在 API 中管理的不同数据。

当您这样做时，请确保考虑任何嵌套的资源。例如，`customers`可能有`sales`，或者`events`可能包含`guests`。当您定义 API 端点时，建立这些资源层次结构将会有所帮助。

[*Remove ads*](/account/join/)

### 定义您的端点

一旦您确定了 web 服务中的资源，您将希望使用这些资源来定义 API 端点。下面是一些您可能在支付处理服务的 API 中找到的`transactions`资源的端点示例:

| HTTP 方法 | API 端点 | 描述 |
| --- | --- | --- |
| `GET` | `/transactions` | 获取交易列表。 |
| `GET` | `/transactions/<transaction_id>` | 获得单笔交易。 |
| `POST` | `/transactions` | 创建新的交易记录。 |
| `PUT` | `/transactions/<transaction_id>` | 更新交易记录。 |
| `PATCH` | `/transactions/<transaction_id>` | 部分更新交易记录。 |
| `DELETE` | `/transactions/<transaction_id>` | 删除交易记录。 |

这六个端点涵盖了您需要在 web 服务中创建、读取、更新和删除`transactions`的所有操作。基于用户可以使用 API 执行的操作，web 服务中的每个资源都有一个类似的端点列表。

**注意:**端点不应该包含动词。相反，您应该选择适当的 HTTP 方法来传达端点的操作。例如，下面的端点包含一个不需要的动词:

```py
GET /getTransactions
```

这里，`get`在不需要的时候包含在端点中。HTTP 方法`GET`已经通过指示动作为端点提供了语义。您可以从端点中删除`get`:

```py
GET /transactions
```

这个端点只包含一个复数名词，HTTP 方法`GET`传递动作。

现在来看一个嵌套资源端点的例子。在这里，您将看到嵌套在`events`资源下的`guests`的端点:

| HTTP 方法 | API 端点 | 描述 |
| --- | --- | --- |
| `GET` | `/events/<event_id>/guests` | 弄一份宾客名单。 |
| `GET` | `/events/<event_id>/guests/<guest_id>` | 找一个单独的客人。 |
| `POST` | `/events/<event_id>/guests` | 创建新客人。 |
| `PUT` | `/events/<event_id>/guests/<guest_id>` | 更新客人。 |
| `PATCH` | `/events/<event_id>/guests/<guest_id>` | 部分更新客人。 |
| `DELETE` | `/events/<event_id>/guests/<guest_id>` | 删除客人。 |

有了这些端点，您可以管理系统中特定事件的`guests`。

这不是为嵌套资源定义端点的唯一方式。有些人更喜欢使用[查询字符串](https://en.wikipedia.org/wiki/Query_string)来访问嵌套资源。查询字符串允许您在 HTTP 请求中发送附加参数。在下面的端点中，您添加了一个查询字符串来获取特定`event_id`的`guests`:

```py
GET /guests?event_id=23
```

这个端点将过滤掉任何不引用给定`event_id`的`guests`。与 API 设计中的许多事情一样，您需要决定哪种方法最适合您的 web 服务。

**注意:【REST API 不太可能在 web 服务的整个生命周期中保持不变。资源会发生变化，您需要更新您的端点来反映这些变化。这就是 **API 版本**的用武之地。API 版本控制允许您修改 API，而不用担心破坏现有的集成。**

有很多种版本控制策略。选择正确的选项取决于 API 的需求。下面是一些最流行的 API 版本控制选项:

*   [URI 版本](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design#uri-versioning)
*   [HTTP 报头版本](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design#header-versioning)
*   [查询字符串版本](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design#query-string-versioning)
*   [媒体类型版本](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design#media-type-versioning)

无论您选择什么策略，对 API 进行版本控制都是重要的一步，以确保它能够适应不断变化的需求，同时支持现有用户。

既然您已经介绍了端点，那么在下一节中，您将会看到在 REST API 中格式化数据的一些选项。

### 选择您的数据交换格式

格式化 web 服务数据的两个流行选项是 [XML](https://en.wikipedia.org/wiki/XML) 和 JSON。传统上，XML 和[SOAP](https://en.wikipedia.org/wiki/SOAP)API 非常受欢迎，但是 JSON 和 REST APIs 更受欢迎。为了比较这两者，请看一个格式化为 XML 和 JSON 的示例`book`资源。

这是 XML 格式的书:

```py
<?xml version="1.0" encoding="UTF-8" ?>
<book>
    <title>Python Basics</title>
    <page_count>635</page_count>
    <pub_date>2021-03-16</pub_date>
    <authors>
        <author>
            <name>David Amos</name>
        </author>
        <author>
            <name>Joanna Jablonski</name>
        </author>
        <author>
            <name>Dan Bader</name>
        </author>
        <author>
            <name>Fletcher Heisler</name>
        </author>
    </authors>
    <isbn13>978-1775093329</isbn13>
    <genre>Education</genre>
</book>
```

XML 使用一系列的**元素**来编码数据。每个元素都有一个开始和结束标记，数据在它们之间。元素可以嵌套在其他元素中。你可以在上面看到，几个`<author>`标签嵌套在`<authors>`中。

现在，看看 JSON 中的同一个`book`:

```py
{ "title":  "Python Basics", "page_count":  635, "pub_date":  "2021-03-16", "authors":  [ {"name":  "David Amos"}, {"name":  "Joanna Jablonski"}, {"name":  "Dan Bader"}, {"name":  "Fletcher Heisler"} ], "isbn13":  "978-1775093329", "genre":  "Education" }
```

JSON 以类似于 Python 字典的键值对存储数据。像 XML 一样，JSON 支持任何级别的嵌套数据，因此您可以对复杂数据建模。

JSON 和 XML 本质上都没有谁更好，但是 REST API 开发人员更喜欢 JSON。当您将 REST API 与前端框架如 [React](https://reactjs.org/) 或 [Vue](https://vuejs.org/) 配对时尤其如此。

[*Remove ads*](/account/join/)

### 设计成功响应

一旦选择了数据格式，下一步就是决定如何响应 HTTP 请求。来自 REST API 的所有响应应该具有相似的格式，并包含正确的 HTTP 状态代码。

在这一节中，您将看到一个管理库存`cars`的假想 API 的一些示例 HTTP 响应。这些例子将让你知道应该如何格式化你的 API 响应。为了清楚起见，我们将查看原始的 HTTP 请求和响应，而不是使用像`requests`这样的 HTTP 库。

首先，看一下对`/cars`的`GET`请求，它返回一个`cars`列表:

```py
GET /cars HTTP/1.1
Host: api.example.com
```

这个 HTTP 请求由四部分组成:

1.  **`GET`** 是 HTTP 方法类型。
2.  **`/cars`** 是 API 端点。
3.  **`HTTP/1.1`** 是 HTTP 版本。
4.  **`Host: api.example.com`** 是 API 主机。

这四个部分就是你向`/cars`发送一个`GET`请求所需要的全部。现在来看看回应。这个 API 使用 JSON 作为数据交换格式:

```py
HTTP/1.1 200 OK
Content-Type: application/json
...

[ { "id":  1, "make":  "GMC", "model":  "1500 Club Coupe", "year":  1998, "vin":  "1D7RV1GTXAS806941", "color":  "Red" }, { "id":  2, "make":  "Lamborghini", "model":"Gallardo", "year":2006, "vin":"JN1BY1PR0FM736887", "color":"Mauve" }, { "id":  3, "make":  "Chevrolet", "model":"Monte Carlo", "year":1996, "vin":"1G4HP54K714224234", "color":"Violet" } ]
```

API 返回一个包含一列`cars`的响应。您知道响应是成功的，因为有了`200 OK`状态代码。该响应还有一个设置为`application/json`的`Content-Type`报头。这告诉用户将响应解析为 JSON。

**注意:**当你使用一个真正的 API 时，你会看到比这更多的 HTTP 头。这些头文件在不同的 API 之间是不同的，所以在这些例子中它们被排除了。

务必在回复中设置正确的`Content-Type`标题。如果你发送 JSON，那么设置`Content-Type`为`application/json`。如果是 XML，那么将其设置为`application/xml`。这个头告诉用户应该如何解析数据。

您还需要在回复中包含适当的状态代码。对于任何成功的`GET`请求，您应该返回`200 OK`。这告诉用户他们的请求按预期得到了处理。

看看另一个`GET`请求，这次是针对一辆车:

```py
GET /cars/1 HTTP/1.1
Host: api.example.com
```

这个 HTTP 请求查询汽车`1`的 API。以下是回应:

```py
HTTP/1.1 200 OK
Content-Type: application/json

{ "id":  1, "make":  "GMC", "model":  "1500 Club Coupe", "year":  1998, "vin":  "1D7RV1GTXAS806941", "color":  "Red" },
```

这个响应包含一个带有汽车数据的 JSON 对象。既然是单个对象，就不需要用列表包装。与上一个响应一样，这也有一个`200 OK`状态代码。

**注意:**`GET`请求不应该修改现有的资源。如果请求包含数据，那么这个数据应该被忽略，API 应该返回没有改变的资源。

接下来，查看添加新车的`POST`请求:

```py
POST /cars HTTP/1.1
Host: api.example.com
Content-Type: application/json

{ "make":  "Nissan", "model":  "240SX", "year":  1994, "vin":  "1N6AD0CU5AC961553", "color":  "Violet" }
```

这个`POST`请求在请求中包含新车的 JSON。它将`Content-Type`头设置为`application/json`，这样 API 就知道请求的内容类型。API 将从 JSON 创建一辆新车。

以下是回应:

```py
HTTP/1.1 201 Created
Content-Type: application/json

{ "id":  4, "make":  "Nissan", "model":  "240SX", "year":  1994, "vin":  "1N6AD0CU5AC961553", "color":  "Violet" }
```

这个响应有一个`201 Created`状态代码，告诉用户一个新的资源已经创建。确保对所有成功的`POST`请求使用`201 Created`而不是`200 OK`。

这个响应还包括一个由 API 生成的带有`id`的新车副本。在响应中发回一个`id`非常重要，这样用户就可以再次修改资源。

**注意:**当用户用`POST`创建资源或者用`PUT`或`PATCH`修改资源时，一定要发送回一份副本，这一点很重要。这样，用户可以看到他们所做的更改。

现在来看看一个`PUT`请求:

```py
PUT /cars/4 HTTP/1.1
Host: api.example.com
Content-Type: application/json

{ "make":  "Buick", "model":  "Lucerne", "year":  2006, "vin":  "4T1BF3EK8AU335094", "color":"Maroon" }
```

这个请求使用前一个请求中的`id`用所有新数据更新汽车。提醒一下，`PUT`用新数据更新资源上的所有域。以下是回应:

```py
HTTP/1.1 200 OK
Content-Type: application/json

{ "id":  4, "make":  "Buick",  "model":  "Lucerne",  "year":  2006,  "vin":  "4T1BF3EK8AU335094",  "color":"Maroon"  }
```

该响应包括带有新数据的`car`的副本。同样，您总是希望为一个`PUT`请求发送回完整的资源。这同样适用于`PATCH`的请求:

```py
PATCH /cars/4 HTTP/1.1
Host: api.example.com
Content-Type: application/json

{ "vin":  "VNKKTUD32FA050307", "color":  "Green" }
```

请求只更新资源的一部分。在上面的请求中，`vin`和`color`字段将被更新为新值。以下是回应:

```py
HTTP/1.1 200 OK
Content-Type: application/json

{ "id":  4, "make":  "Buick", "model":  "Lucerne", "year":  2006, "vin":  "VNKKTUD32FA050307",  "color":  "Green"  }
```

该响应包含`car`的完整副本。如您所见，只有`vin`和`color`字段被更新。

最后，看看当 REST API 收到一个`DELETE`请求时应该如何响应。这里有一个删除`car`的`DELETE`请求:

```py
DELETE /cars/4 HTTP/1.1
```

这个`DELETE`请求告诉 API 移除 ID 为`4`的`car`。以下是回应:

```py
HTTP/1.1 204 No Content
```

该响应仅包括状态代码`204 No Content`。此状态代码告诉用户操作成功，但是响应中没有返回任何内容。这是有意义的，因为`car`已经被删除了。没有理由在响应中发送它的副本。

当一切按计划进行时，上面的响应工作得很好，但是如果请求有问题会发生什么呢？在下一节中，您将看到当错误发生时，您的 REST API 应该如何响应。

[*Remove ads*](/account/join/)

### 设计错误响应

对 REST API 的请求总有可能失败。定义错误响应是一个好主意。这些响应应该包括发生了什么错误的描述以及相应的状态代码。在这一节中，您将看到几个例子。

首先，看一下对 API 中不存在的资源的请求:

```py
GET /motorcycles HTTP/1.1
Host: api.example.com
```

这里，用户向`/motorcycles`发送一个`GET`请求，这个请求并不存在。API 发回以下响应:

```py
HTTP/1.1 404 Not Found
Content-Type: application/json
...

{ "error":  "The requested resource was not found." }
```

该响应包括一个`404 Not Found`状态代码。除此之外，响应还包含一个带有描述性错误消息的 JSON 对象。提供一个描述性的错误信息给用户更多的错误上下文。

现在来看看用户发送无效请求时的错误响应:

```py
POST /cars HTTP/1.1
Host: api.example.com
Content-Type: application/json

{ "make":  "Nissan", "year":  1994, "color":  "Violet"
```

这个`POST`请求包含 JSON，但是格式不正确。它的结尾缺少了一个右花括号(`}`)。API 将无法处理这些数据。错误响应告诉用户有关问题的信息:

```py
HTTP/1.1 400 Bad Request
Content-Type: application/json

{ "error":  "This request was not properly formatted. Please send again." }
```

这个响应包括一个描述性的错误消息和`400 Bad Request`状态代码，告诉用户他们需要修复这个请求。

即使格式正确，请求也可能在其他几个方面出错。在下一个例子中，用户发送了一个`POST`请求，但是包含了一个不支持的媒体类型:

```py
POST /cars HTTP/1.1
Host: api.example.com
Content-Type: application/xml

<?xml version="1.0" encoding="UTF-8" ?>
<car>
    <make>Nissan</make>
    <model>240SX</model>
    <year>1994</year>
    <vin>1N6AD0CU5AC961553</vin>
    <color>Violet</color>
</car>
```

在这个请求中，用户发送 XML，但是 API 只支持 JSON。API 的响应如下:

```py
HTTP/1.1 415 Unsupported Media Type
Content-Type: application/json

{ "error":  "The application/xml mediatype is not supported." }
```

这个响应包含了`415 Unsupported Media Type`状态代码，表明`POST`请求包含了 API 不支持的数据格式。这个错误代码对于格式错误的数据是有意义的，但是对于格式正确但仍然无效的数据呢？

在下一个例子中，用户发送了一个`POST`请求，但是包含了与其他数据的字段不匹配的`car`数据:

```py
POST /cars HTTP/1.1
Host: api.example.com
Content-Type: application/json

{ "make":  "Nissan", "model":  "240SX", "topSpeed":  120  "warrantyLength":  10  }
```

在这个请求中，用户向 JSON 添加了`topSpeed`和`warrantyLength`字段。API 不支持这些字段，因此它会响应一条错误消息:

```py
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/json

{ "error":  "Request had invalid or missing data." }
```

该响应包括`422 Unprocessable Entity`状态代码。此状态代码表示请求没有任何问题，但是数据无效。REST API 需要验证传入的数据。如果用户随请求发送数据，那么 API 应该验证数据并通知用户任何错误。

响应请求，不管是成功的还是错误的，都是 REST API 最重要的工作之一。如果你的 API 是直观的，并提供准确的响应，那么用户围绕你的 web 服务构建应用程序就更容易了。幸运的是，一些优秀的 Python web 框架抽象出了处理 HTTP 请求和返回响应的复杂性。在下一节中，您将看到三个流行的选项。

[*Remove ads*](/account/join/)

## REST 和 Python:行业工具

在这一节中，您将看到用 Python 构建 REST APIs 的三个流行框架。每个框架都有优点和缺点，所以您必须评估哪个最适合您的需求。为此，在接下来的章节中，您将会看到每个框架中的 REST API。所有的例子都是针对一个类似的管理国家集合的 API。

每个国家将有以下字段:

*   **`name`** 是国家的名称。
*   **`capital`** 是这个国家的首都。
*   **`area`** 是以平方公里为单位的国家的面积。

字段`name`、`capital`和`area`存储世界上某个特定国家的数据。

大多数时候，从 REST API 发送的数据来自数据库。[连接数据库](https://realpython.com/flask-connexion-rest-api-part-2/)超出了本教程的范围。对于以下示例，您将在 Python 列表中存储数据。例外情况是 Django REST 框架示例，它运行 Django 创建的 SQLite 数据库。

**注意:**建议您为每个示例创建单独的文件夹，以分离源文件。您还会希望使用[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)来隔离依赖性。

为了保持一致性，您将使用`countries`作为所有三个框架的主要端点。您还将使用 JSON 作为所有三个框架的数据格式。

现在您已经了解了 API 的背景，您可以继续下一部分，在这里您将看到 **Flask** 中的 REST API。

### 烧瓶

[Flask](https://realpython.com/introduction-to-flask-part-1-setting-up-a-static-site/) 是用于构建 web 应用和 REST APIs 的 Python 微框架。Flask 为您的应用程序提供了坚实的基础，同时留给您许多设计选择。Flask 的主要工作是处理 HTTP 请求，并将它们路由到应用程序中适当的函数。

**注意:**本节中的代码使用了新的 [Flask 2](https://palletsprojects.com/blog/flask-2-0-released/) 语法。如果你运行的是老版本的 Flask，那么使用 [`@app.route("/countries")`](https://flask.palletsprojects.com/en/2.0.x/api/#flask.Flask.route) 而不是 [`@app.get("/countries")`](https://flask.palletsprojects.com/en/2.0.x/api/#flask.Flask.get) 和 [`@app.post("/countries")`](https://flask.palletsprojects.com/en/2.0.x/api/#flask.Flask.post) 。

为了在旧版本的 [Flask](https://flask.palletsprojects.com/en/1.1.x/) 中处理`POST`请求，您还需要将`methods`参数添加到`@app.route()`:

```py
@app.route("/countries", methods=["POST"])
```

这个路由处理对 Flask 1 中的`/countries`的`POST`请求。

以下是 REST API 的 Flask 应用程序示例:

```py
# app.py
from flask import Flask, request, jsonify

app = Flask(__name__)

countries = [
    {"id": 1, "name": "Thailand", "capital": "Bangkok", "area": 513120},
    {"id": 2, "name": "Australia", "capital": "Canberra", "area": 7617930},
    {"id": 3, "name": "Egypt", "capital": "Cairo", "area": 1010408},
]

def _find_next_id():
    return max(country["id"] for country in countries) + 1

@app.get("/countries")
def get_countries():
    return jsonify(countries)

@app.post("/countries")
def add_country():
    if request.is_json:
        country = request.get_json()
        country["id"] = _find_next_id()
        countries.append(country)
        return country, 201
    return {"error": "Request must be JSON"}, 415
```

这个应用程序定义了 API 端点`/countries`来管理国家列表。它处理两种不同的请求:

1.  **`GET /countries`** 返回`countries`的列表。
2.  **`POST /countries`** 向列表中添加一个新的`country`。

**注意:**这个 Flask 应用程序只包含处理两种类型的 API 端点请求的函数，`/countries`。在一个完整的 REST API 中，您可能希望扩展它，以包含所有必需操作的函数。

您可以通过安装带有`pip`的`flask`来试用这个应用程序:

```py
$ python -m pip install flask
```

一旦安装了`flask`，将代码保存在一个名为`app.py`的文件中。要运行这个 Flask 应用程序，首先需要将一个名为`FLASK_APP`的环境变量设置为`app.py`。这告诉 Flask 哪个文件包含您的应用程序。

在包含`app.py`的文件夹中运行以下命令:

```py
$ export FLASK_APP=app.py
```

这会将当前 shell 中的`FLASK_APP`设置为`app.py`。也可以将`FLASK_ENV`设置为`development`，将 Flask 置于**调试模式**:

```py
$ export FLASK_ENV=development
```

除了提供有用的错误消息，调试模式还会在所有代码更改后触发应用程序的重新加载。如果没有调试模式，您必须在每次更改后重启服务器。

**注意:**以上命令可以在 macOS 或 Linux 上运行。如果您在 Windows 上运行它，那么您需要在命令提示符下像这样设置`FLASK_APP`和`FLASK_ENV`:

```py
C:\> set FLASK_APP=app.py
C:\> set FLASK_ENV=development
```

现在`FLASK_APP`和`FLASK_ENV`被设置在 Windows 外壳内部。

准备好所有的环境变量后，您现在可以通过调用`flask run`来启动 Flask 开发服务器:

```py
$ flask run
* Serving Flask app "app.py" (lazy loading)
* Environment: development
* Debug mode: on
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

这将启动运行该应用程序的服务器。打开你的浏览器，进入`http://127.0.0.1:5000/countries`，你会看到如下的回应:

```py
[ { "area":  513120, "capital":  "Bangkok", "id":  1, "name":  "Thailand" }, { "area":  7617930, "capital":  "Canberra", "id":  2, "name":  "Australia" }, { "area":  1010408, "capital":  "Cairo", "id":  3, "name":  "Egypt" } ]
```

这个 JSON 响应包含在`app.py`开头定义的三个`countries`。看看下面的代码，看看这是如何工作的:

```py
@app.get("/countries")
def get_countries():
    return jsonify(countries)
```

这段代码使用`@app.get()`，一个 Flask route [decorator](https://realpython.com/primer-on-python-decorators/) ，将`GET`请求连接到应用程序中的一个函数。当您访问`/countries`时，Flask 调用修饰函数来处理 HTTP 请求，然后[返回](https://realpython.com/python-return-statement/)一个响应。

上面的代码中，`get_countries()`取`countries`，是一个 [Python 列表](https://realpython.com/python-lists-tuples/)，用`jsonify()`转换成 JSON。这个 JSON 在响应中返回。

**注意:**大多数时候，你可以直接从 Flask 函数返回一个 Python 字典。Flask 会自动将任何 Python 字典转换成 JSON。您可以通过下面的函数看到这一点:

```py
@app.get("/country")
def get_country():
    return countries[1]
```

在这段代码中，您从`countries`返回第二个字典。Flask 会把这个字典转换成 JSON。当您请求`/country`时，您将看到以下内容:

```py
{ "area":  7617930, "capital":  "Canberra", "id":  2, "name":  "Australia" }
```

这是你从`get_country()`返回的字典的 JSON 版本。

在`get_countries()`中，您需要使用`jsonify()`,因为您返回的是一个字典列表，而不仅仅是一个字典。Flask 不会自动将列表转换成 JSON。

现在来看看`add_country()`。该函数处理对`/countries`的`POST`请求，并允许您向列表中添加一个新的国家。它使用 Flask [`request`](https://flask.palletsprojects.com/en/1.1.x/quickstart/#the-request-object) 对象来获取关于当前 HTTP 请求的信息:

```py
@app.post("/countries")
def add_country():
    if request.is_json:
        country = request.get_json()
        country["id"] = _find_next_id()
        countries.append(country)
        return country, 201
    return {"error": "Request must be JSON"}, 415
```

该函数执行以下操作:

1.  使用 [`request.is_json`](https://flask.palletsprojects.com/en/1.1.x/api/?highlight=is_json#flask.Request.is_json) 检查请求是否为 JSON
2.  使用`request.get_json()`创建新的`country`实例
3.  找到下一个`id`并将其设置在`country`上
4.  将新的`country`附加到`countries`
5.  在响应中返回`country`以及`201 Created`状态代码
6.  如果请求不是 JSON，则返回错误消息和`415 Unsupported Media Type`状态代码

`add_country()`也调用`_find_next_id()`来确定新`country`的`id`:

```py
def _find_next_id():
    return max(country["id"] for country in countries) + 1
```

这个辅助函数使用一个[生成器表达式](https://realpython.com/introduction-to-python-generators/)来选择所有的国家 id，然后对它们调用 [`max()`](https://realpython.com/python-min-and-max/) 来获得最大值。它将这个值递增`1`以获得下一个要使用的 ID。

您可以使用命令行工具 [curl](https://curl.se/) 在 shell 中尝试这个端点，它允许您从命令行发送 HTTP 请求。在这里，您将向`countries`列表中添加一个新的`country`:

```py
$ curl -i http://127.0.0.1:5000/countries \
-X POST \
-H 'Content-Type: application/json' \
-d '{"name":"Germany", "capital": "Berlin", "area": 357022}'

HTTP/1.0 201 CREATED
Content-Type: application/json
...

{
 "area": 357022,
 "capital": "Berlin",
 "id": 4,
 "name": "Germany"
}
```

这个 curl 命令有一些选项，了解这些选项很有帮助:

*   **`-X`** 为请求设置 HTTP 方法。
*   **`-H`** 给请求添加一个 HTTP 头。
*   **`-d`** 定义了请求数据。

设置好这些选项后，curl 在一个`POST`请求中发送 JSON 数据，其中`Content-Type`头设置为`application/json`。REST API 返回`201 CREATED`以及您添加的新`country`的 JSON。

**注意:**在这个例子中，`add_country()`不包含任何确认请求中的 JSON 与`countries`的格式匹配的验证。如果您想在 flask 中验证 json 的格式，请查看 [flask-expects-json](https://pypi.org/project/flask-expects-json/) 。

您可以使用 curl 向`/countries`发送一个`GET`请求，以确认新的`country`已被添加。如果您没有在 curl 命令中使用`-X`，那么默认情况下它会发送一个`GET`请求:

```py
$ curl -i http://127.0.0.1:5000/countries

HTTP/1.0 200 OK
Content-Type: application/json
...

[
 {
 "area": 513120,
 "capital": "Bangkok",
 "id": 1,
 "name": "Thailand"
 },
 {
 "area": 7617930,
 "capital": "Canberra",
 "id": 2,
 "name": "Australia"
 },
 {
 "area": 1010408,
 "capital": "Cairo",
 "id": 3,
 "name": "Egypt"
 },
 {
 "area": 357022,
 "capital": "Berlin",
 "id": 4,
 "name": "Germany"
 }
]
```

这将返回系统中国家的完整列表，最新的国家在底部。

这只是 Flask 功能的一个示例。这个应用程序可以扩展到包括所有其他 HTTP 方法的端点。Flask 还有一个庞大的扩展生态系统，为 REST APIs 提供额外的功能，比如[数据库集成](https://realpython.com/flask-connexion-rest-api/)、[认证](https://realpython.com/flask-google-login/)和后台处理。

[*Remove ads*](/account/join/)

### Django REST 框架

构建 REST APIs 的另一个流行选项是 [Django REST framework](https://realpython.com/django-rest-framework-quick-start/) 。Django REST framework 是一个 [Django](https://realpython.com/get-started-with-django-1/) 插件，它在现有 Django 项目的基础上增加了 REST API 功能。

要使用 Django REST 框架，您需要一个 Django 项目。如果您已经有了一个，那么您可以将本节中的模式应用到您的项目中。否则，继续下去，您将构建一个 Django 项目并添加到 Django REST 框架中。

首先，用`pip`安装`Django`和`djangorestframework`:

```py
$ python -m pip install Django djangorestframework
```

这将安装`Django`和`djangorestframework`。你现在可以使用`django-admin`工具来创建一个新的 Django 项目。运行以下命令启动您的项目:

```py
$ django-admin startproject countryapi
```

该命令在当前目录下创建一个名为`countryapi`的新文件夹。这个文件夹中是运行 Django 项目所需的所有文件。接下来，您将在您的项目中创建一个新的 **Django 应用程序**。Django 将项目的功能分解成应用程序。每个应用程序管理项目的不同部分。

**注意:**在本教程中，您将只接触到 Django 的皮毛。如果你有兴趣了解更多，请查看可用的 [Django 教程](https://realpython.com/tutorials/django/)。

要创建应用程序，请将目录更改为`countryapi`并运行以下命令:

```py
$ python manage.py startapp countries
```

这将在您的项目中创建一个新的`countries`文件夹。这个文件夹中是这个应用程序的基本文件。

既然您已经创建了一个应用程序，那么您需要将它告诉 Django。在您刚刚创建的`countries`文件夹旁边是另一个名为`countryapi`的文件夹。此文件夹包含项目的配置和设置。

**注意:**这个文件夹与 Django 在运行`django-admin startproject countryapi`时创建的根文件夹同名。

打开`countryapi`文件夹中的`settings.py`文件。在`INSTALLED_APPS`中添加以下几行，告诉 Django 关于`countries`应用程序和 Django REST 框架的信息:

```py
# countryapi/settings.py
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
 "rest_framework", "countries", ]
```

您已经为`countries`应用程序和`rest_framework`添加了一行。

您可能想知道为什么需要将`rest_framework`添加到应用程序列表中。您需要添加它，因为 Django REST 框架只是另一个 Django 应用程序。Django 插件是打包分发的 Django 应用程序，任何人都可以使用。

下一步是创建 Django 模型来定义数据的字段。在`countries`应用程序内部，用以下代码更新`models.py`:

```py
# countries/models.py
from django.db import models

class Country(models.Model):
 name = models.CharField(max_length=100) capital = models.CharField(max_length=100) area = models.IntegerField(help_text="(in square kilometers)")
```

这段代码定义了一个`Country`模型。Django 将使用这个模型为国家数据创建数据库表和列。

运行以下命令，让 Django 根据这个模型更新数据库:

```py
$ python manage.py makemigrations
Migrations for 'countries':
 countries/migrations/0001_initial.py
 - Create model Country

$ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, contenttypes, countries, sessions
Running migrations:
 Applying contenttypes.0001_initial... OK
 Applying auth.0001_initial... OK
 ...
```

这些命令使用 [Django 迁移](https://realpython.com/django-migrations-a-primer/)在数据库中创建一个新表。

这个表开始是空的，但是最好有一些初始数据，这样就可以测试 Django REST 框架。为此，您将使用一个 [Django fixture](https://realpython.com/django-pytest-fixtures/) 在数据库中加载一些数据。

将以下 JSON 数据复制并保存到一个名为`countries.json`的文件中，并保存在`countries`目录下:

```py
[ { "model":  "countries.country", "pk":  1, "fields":  { "name":  "Thailand", "capital":  "Bangkok", "area":  513120 } }, { "model":  "countries.country", "pk":  2, "fields":  { "name":  "Australia", "capital":  "Canberra", "area":  7617930 } }, { "model":  "countries.country", "pk":  3, "fields":  { "name":  "Egypt", "capital":  "Cairo", "area":  1010408 } } ]
```

这个 JSON 包含三个国家的数据库条目。调用以下命令将该数据加载到数据库中:

```py
$ python manage.py loaddata countries.json
Installed 3 object(s) from 1 fixture(s)
```

这将向数据库中添加三行。

至此，您的 Django 应用程序已经设置完毕，并填充了一些数据。您现在可以开始向项目中添加 Django REST 框架了。

Django REST 框架采用现有的 Django 模型，并将其转换为 JSON 用于 REST API。它通过**模型序列化器**来实现这一点。模型序列化器告诉 Django REST 框架如何将模型实例转换成 JSON，以及应该包含哪些数据。

您将从上面为`Country`模型创建您的序列化程序。首先在`countries`应用程序中创建一个名为`serializers.py`的文件。完成之后，将下面的代码添加到`serializers.py`:

```py
# countries/serializers.py
from rest_framework import serializers
from .models import Country

class CountrySerializer(serializers.ModelSerializer):
    class Meta:
        model = Country
        fields = ["id", "name", "capital", "area"]
```

这个序列化器`CountrySerializer`继承了`serializers.ModelSerializer`，根据`Country`的模型字段自动生成 JSON 内容。除非指定，否则`ModelSerializer`子类将包含 JSON 中 Django 模型的所有字段。您可以通过将`fields`设置为您希望包含的数据列表来修改此行为。

就像 Django 一样，Django REST 框架使用[视图](https://docs.djangoproject.com/en/dev/topics/http/views/)从数据库中查询数据并显示给用户。不用从头开始编写 REST API 视图，你可以子类化 Django REST 框架的 [`ModelViewSet`](https://www.django-rest-framework.org/api-guide/viewsets/#modelviewset) 类，该类拥有常见 REST API 操作的默认视图。

**注意:**Django REST 框架文档将这些视图称为[动作](https://www.django-rest-framework.org/api-guide/viewsets/#viewset-actions)。

下面是`ModelViewSet`提供的动作及其等效 HTTP 方法的列表:

| HTTP 方法 | 行动 | 描述 |
| --- | --- | --- |
| `GET` | `.list()` | 获取国家列表。 |
| `GET` | `.retrieve()` | 得到一个国家。 |
| `POST` | `.create()` | 创建一个新的国家。 |
| `PUT` | `.update()` | 更新一个国家。 |
| `PATCH` | `.partial_update()` | 部分更新一个国家。 |
| `DELETE` | `.destroy()` | 删除一个国家。 |

如您所见，这些动作映射到 REST API 中的标准 HTTP 方法。你可以[在你的子类中覆盖这些动作](https://www.django-rest-framework.org/api-guide/viewsets/#example)或者[根据你的 API 的需求添加额外的动作](https://www.django-rest-framework.org/api-guide/viewsets/#marking-extra-actions-for-routing)。

下面是名为`CountryViewSet`的`ModelViewSet`子类的代码。这个类将生成管理`Country`数据所需的视图。将以下代码添加到`countries`应用程序内的`views.py`:

```py
# countries/views.py
from rest_framework import viewsets

from .models import Country
from .serializers import CountrySerializer

class CountryViewSet(viewsets.ModelViewSet):
    serializer_class = CountrySerializer
    queryset = Country.objects.all()
```

在这个类中，`serializer_class`被设置为`CountrySerializer`，`queryset`被设置为`Country.objects.all()`。这告诉 Django REST framework 要使用哪个序列化程序，以及如何在数据库中查询这个特定的视图集。

一旦创建了视图，就需要将它们映射到适当的 URL 或端点。为此，Django REST 框架提供了一个`DefaultRouter`，它将自动为一个`ModelViewSet`生成 URL。

在`countries`应用程序中创建一个`urls.py`文件，并将以下代码添加到该文件中:

```py
# countries/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import CountryViewSet

router = DefaultRouter()
router.register(r"countries", CountryViewSet)

urlpatterns = [
    path("", include(router.urls))
]
```

这段代码创建了一个`DefaultRouter`，并在`countries` URL 下注册了`CountryViewSet`。这将把`CountryViewSet`的所有 URL 放在`/countries/`下。

**注意:** Django REST 框架自动在`DefaultRouter`生成的任何端点的末尾追加一个正斜杠(`/`)。您可以禁用此行为，如下所示:

```py
router = DefaultRouter(trailing_slash=False)
```

这将禁用端点末尾的正斜杠。

最后，您需要更新项目的基本`urls.py`文件，以包含项目中所有的`countries`URL。用下面的代码更新`countryapi`文件夹中的`urls.py`文件:

```py
# countryapi/urls.py
from django.contrib import admin
from django.urls import path, include 
urlpatterns = [
    path("admin/", admin.site.urls),
 path("", include("countries.urls")), ]
```

这将所有的 URL 放在`/countries/`下。现在您已经准备好尝试 Django 支持的 REST API 了。在根目录`countryapi`中运行以下命令来启动 Django 开发服务器:

```py
$ python manage.py runserver
```

开发服务器现在正在运行。继续向`/countries/`发送`GET`请求，以获得 Django 项目中所有国家的列表:

```py
$ curl -i http://127.0.0.1:8000/countries/ -w '\n'

HTTP/1.1 200 OK
...

[
 {
 "id": 1,
 "name":"Thailand",
 "capital":"Bangkok",
 "area":513120
 },
 {
 "id": 2,
 "name":"Australia",
 "capital":"Canberra",
 "area":7617930
 },
 {
 "id": 3,
 "name":"Egypt",
 "capital":"Cairo",
 "area":1010408
 }
]
```

Django REST 框架发回一个 JSON 响应，其中包含您之前添加的三个国家。上面的回答是为了可读性而格式化的，所以你的回答看起来会有所不同。

您在`countries/urls.py`中创建的 [`DefaultRouter`](https://www.django-rest-framework.org/api-guide/routers/#defaultrouter) 为所有标准 API 端点的请求提供了 URL:

*   `GET /countries/`
*   `GET /countries/<country_id>/`
*   `POST /countries/`
*   `PUT /countries/<country_id>/`
*   `PATCH /countries/<country_id>/`
*   `DELETE /countries/<country_id>/`

您可以在下面多尝试几个端点。向`/countries/`发送一个`POST`请求，在 Django 项目中创建一个新的`Country`:

```py
$ curl -i http://127.0.0.1:8000/countries/ \
-X POST \
-H 'Content-Type: application/json' \
-d '{"name":"Germany", "capital": "Berlin", "area": 357022}' \
-w '\n'

HTTP/1.1 201 Created
...

{
 "id":4,
 "name":"Germany",
 "capital":"Berlin",
 "area":357022
}
```

这将使用您在请求中发送的 JSON 创建一个新的`Country`。Django REST 框架返回一个`201 Created`状态代码和新的`Country`。

**注意:**默认情况下，响应末尾不包含新行。这意味着 JSON 可能会在您的命令提示符下运行。上面的 curl 命令包含了`-w '\n'`来在 JSON 后面添加一个换行符，以解决这个问题。

您可以通过向已有`id`的`GET /countries/<country_id>/`发送请求来查看已有的`Country`。运行以下命令获得第一个`Country`:

```py
$ curl -i http://127.0.0.1:8000/countries/1/ -w '\n'

HTTP/1.1 200 OK
...

{
 "id":1,
 "name":"Thailand",
 "capital":"Bangkok",
 "area":513120
}
```

响应包含第一个`Country`的信息。这些例子只涵盖了`GET`和`POST`请求。您可以自行尝试`PUT`、`PATCH`和`DELETE`请求，看看如何从 REST API 中完全管理您的模型。

正如您所看到的，Django REST 框架是构建 REST APIs 的一个很好的选择，尤其是如果您已经有了一个 Django 项目，并且想要添加一个 API。

[*Remove ads*](/account/join/)

### FastAPI

FastAPI 是一个针对构建 API 而优化的 Python web 框架。它使用了 [Python 类型提示](https://realpython.com/python-type-checking/)，并且内置了对[异步操作](https://realpython.com/async-io-python/)的支持。FastAPI 构建在 [Starlette](https://www.starlette.io/) 和 [Pydantic](https://pydantic-docs.helpmanual.io/) 之上，性能非常好。

下面是一个用 FastAPI 构建的 REST API 的例子:

```py
# app.py
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

def _find_next_id():
    return max(country.country_id for country in countries) + 1

class Country(BaseModel):
    country_id: int = Field(default_factory=_find_next_id, alias="id")
    name: str
    capital: str
    area: int

countries = [
    Country(id=1, name="Thailand", capital="Bangkok", area=513120),
    Country(id=2, name="Australia", capital="Canberra", area=7617930),
    Country(id=3, name="Egypt", capital="Cairo", area=1010408),
]

@app.get("/countries")
async def get_countries():
    return countries

@app.post("/countries", status_code=201)
async def add_country(country: Country):
    countries.append(country)
    return country
```

这个应用程序使用 FastAPI 的特性为您在其他示例中看到的相同的`country`数据构建一个 REST API。

您可以通过安装带有`pip`的`fastapi`来尝试此应用程序:

```py
$ python -m pip install fastapi
```

您还需要安装`uvicorn[standard]`，一个可以运行 FastAPI 应用程序的服务器:

```py
$ python -m pip install uvicorn[standard]
```

如果你已经安装了`fastapi`和`uvicorn`，那么将上面的代码保存在一个名为`app.py`的文件中。运行以下命令启动开发服务器:

```py
$ uvicorn app:app --reload
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

服务器现在正在运行。打开浏览器，进入`http://127.0.0.1:8000/countries`。您将看到 FastAPI 以如下方式响应:

```py
[ { "id":  1, "name":"Thailand", "capital":"Bangkok", "area":513120 }, { "id":  2, "name":"Australia", "capital":"Canberra", "area":7617930 }, { "id":  3, "name":"Egypt", "capital":"Cairo", "area":1010408 } ]
```

FastAPI 用一个包含一列`countries`的 JSON 数组来响应。您也可以通过向`/countries`发送`POST`请求来添加新的国家:

```py
$ curl -i http://127.0.0.1:8000/countries \
-X POST \
-H 'Content-Type: application/json' \
-d '{"name":"Germany", "capital": "Berlin", "area": 357022}' \
-w '\n'

HTTP/1.1 201 Created
content-type: application/json
...

{"id":4,"name":"Germany","capital":"Berlin","area": 357022}
```

您添加了一个新国家。您可以通过`GET /countries`确认这一点:

```py
$ curl -i http://127.0.0.1:8000/countries -w '\n'

HTTP/1.1 200 OK
content-type: application/json
...

[
 {
 "id":1,
 "name":"Thailand",
 "capital":"Bangkok",
 "area":513120,
 },
 {
 "id":2,
 "name":"Australia",
 "capital":"Canberra",
 "area":7617930
 },
 {
 "id":3,
 "name":"Egypt",
 "capital":"Cairo",
 "area":1010408
 },
 {
 "id":4,
 "name": "Germany",
 "capital": "Berlin",
 "area": 357022
 }
]
```

FastAPI 返回一个 JSON 列表，其中包括您刚刚添加的新国家。

您会注意到 FastAPI 应用程序看起来类似于 Flask 应用程序。像 Flask 一样，FastAPI 也有一个集中的特性集。它并不试图处理 web 应用程序开发的所有方面。它旨在构建具有现代 Python 特性的 API。

如果你靠近`app.py`的顶部，你会看到一个叫做`Country`的类，它扩展了`BaseModel`。`Country`类描述了 REST API 中的数据结构:

```py
class Country(BaseModel):
    country_id: int = Field(default_factory=_find_next_id, alias="id")
    name: str
    capital: str
    area: int
```

这是一个 [Pydantic 模型](https://pydantic-docs.helpmanual.io/usage/models/)的例子。Pydantic 模型在 FastAPI 中提供了一些有用的特性。它们使用 Python 类型注释来强制类中每个字段的数据类型。这允许 FastAPI 为 API 端点自动生成具有正确数据类型的 JSON。它还允许 FastAPI 验证传入的 JSON。

强调第一行很有帮助，因为这一行有很多内容:

```py
country_id: int = Field(default_factory=_find_next_id, alias="id")
```

在这一行中，您可以看到`country_id`，它为`Country`的 ID 存储了一个[整数](https://realpython.com/python-numbers/#integers)。它使用 Pydantic 的 [`Field`函数](https://pydantic-docs.helpmanual.io/usage/schema/#field-customisation)来修改`country_id`的行为。在这个例子中，您将关键字参数`default_factory`和`alias`传递给`Field`。

第一个参数`default_factory`被设置为`_find_next_id()`。该参数指定每当创建新的`Country`时运行的函数。返回值将被赋给`country_id`。

第二个参数`alias`被设置为`id`。这告诉 FastAPI 输出键`"id"`而不是 JSON 中的`"country_id"`:

```py
{ "id":1, "name":"Thailand", "capital":"Bangkok", "area":513120, },
```

这个`alias`也意味着当你创建一个新的`Country`时，你可以使用`id`。您可以在`countries`列表中看到:

```py
countries = [
    Country(id=1, name="Thailand", capital="Bangkok", area=513120),
    Country(id=2, name="Australia", capital="Canberra", area=7617930),
    Country(id=3, name="Egypt", capital="Cairo", area=1010408),
]
```

这个列表包含 API 中初始国家的三个`Country`实例。Pydantic 模型提供了一些很棒的特性，并允许 FastAPI 轻松处理 JSON 数据。

现在看看这个应用程序中的两个 API 函数。第一个函数`get_countries()`，返回一个`countries`列表，用于对`/countries`的`GET`请求:

```py
@app.get("/countries")
async def get_countries():
    return countries
```

FastAPI 将根据 Pydantic 模型中的字段自动创建 JSON，并根据 Python 类型提示设置正确的 JSON 数据类型。

当您向`/countries`发出`POST`请求时，Pydantic 模型也提供了一个好处。您可以在下面的第二个 API 函数中看到，参数`country`有一个`Country`注释:

```py
@app.post("/countries", status_code=201)
async def add_country(country: Country):
    countries.append(country)
    return country
```

这个类型注释告诉 FastAPI 根据`Country`验证传入的 JSON。如果不匹配，那么 FastAPI 将返回一个错误。您可以通过用 JSON 发出一个与 Pydantic 模型不匹配的请求来尝试一下:

```py
$ curl -i http://127.0.0.1:8000/countries \
-X POST \
-H 'Content-Type: application/json' \
-d '{"name":"Germany", "capital": "Berlin"}' \
-w '\n'

HTTP/1.1 422 Unprocessable Entity
content-type: application/json
...

{
 "detail": [
 {
 "loc":["body","area"],
 "msg":"field required",
 "type":"value_error.missing"
 }
 ]
}
```

这个请求中的 JSON 缺少一个值`area`，所以 FastAPI 返回一个响应，其中包含状态代码`422 Unprocessable Entity`以及关于错误的详细信息。Pydantic 模型使这种验证成为可能。

这个例子只是触及了 FastAPI 的皮毛。凭借其高性能和现代化的特性，如`async`函数和自动文档，FastAPI 值得考虑作为您的下一个 REST API。

[*Remove ads*](/account/join/)

## 结论

REST APIs 无处不在。了解如何利用 Python 来消费和构建 API 可以让您处理 web 服务提供的大量数据。

**在本教程中，你已经学会了如何**:

*   识别 **REST 架构**风格
*   使用 HTTP 方法和状态代码
*   使用 **`requests`** 从外部 API 获取和使用数据
*   为 REST API 定义**端点**、**数据**和**响应**
*   开始使用 Python 工具构建一个 **REST API**

使用您的新 Python REST API 技能，您不仅能够与 web 服务交互，还能够为您的应用程序构建 REST API。这些工具为各种有趣的、数据驱动的应用和服务打开了大门。**********