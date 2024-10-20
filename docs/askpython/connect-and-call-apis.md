# Python 中如何连接和调用 API？

> 原文：<https://www.askpython.com/python/examples/connect-and-call-apis>

读者朋友们，你们好！在本文中，我们将讨论在 Python 中连接和调用 API 的不同方式。所以，让我们开始吧！

* * *

## 什么是 API？

API 是`Application programming Interface`的首字母缩写。它可以理解为规则的组合，使我们能够通过我们的系统访问 web 上的外部服务。

因此，API 确定并设置了某些格式，我们可以通过这些格式访问模型中的服务和数据。从 Python 等编程语言的角度来看，API 被认为是 web 上可用的数据源，可以通过编程语言的特定库来访问。

* * *

## 对 API 的请求类型

在使用 API 时，下面列出了一些我们用来对 API 执行某些操作的常用指令或命令

1.  **GET 命令**:它使用户能够以特定的格式(通常是 JSON)从 API 获取数据到他们的系统上。
2.  **POST 命令**:该命令使我们能够向 API 添加数据，即向 web 上的服务添加数据。
3.  **删除命令**:可以从 web 上的 API 服务中删除某些信息。
4.  **PUT 命令**:使用 PUT 命令，我们可以更新 web 上 API 服务中已有的数据或信息。

* * *

## API 的状态/响应代码

在连接到一个 API 时，它返回某些响应代码，这些代码决定了我们与 web 上的 API 的连接状态。让我们看看一些状态代码——

1.  **200** : **OK** 。这意味着我们与 web 上的 API 有了一个健康的**连接。**
2.  **204** :表示我们可以**成功连接到 API，但是没有从服务返回任何数据**。
3.  **401** : **认证失败**！
4.  **403**:**API 服务禁止访问**。
5.  **404** :在服务器 /web 上没有找到请求的 **API 服务。**
6.  **500** : **内部服务器出现错误**。

* * *

## 使用 Python 连接和调用 API 的步骤

现在让我们讨论使用 Python 作为脚本语言来建立与 API 的健康连接的步骤。

### 示例 1:连接到 web 上的 URL

在本例中，我们将遵循以下步骤来建立到 web 上的 URL 的健康连接。

#### 1.导入必要的库

为了连接到 API 并对其执行操作，我们需要将`Python requests library`导入到环境中。

```py
import requests

```

#### 2.执行操作以连接到 API

这里，我们使用了 **GET 命令**连接到 API，如下所示

```py
response_API = requests.get('https://www.askpython.com/')

```

我们已经将需要连接的 url 传递到了`get()`函数中。

#### 3.打印响应代码

`status_code`变量使我们能够查看我们与 API 的连接状态。

```py
response_API.status_code

```

你可以在下面找到完整的代码！

```py
import requests
response_API = requests.get('https://www.askpython.com/')
print(response_API.status_code)

```

**输出:**

```py
200

```

* * *

### **示例** 2:连接到 GMAIL API

在这个例子中，我们将从[这个](https://developers.google.com/gmail/api/reference/rest)链接形成一个到开源 GMAIL API 的健康连接。

看看下面这段代码吧！

**举例:**

```py
import requests
response_API = requests.get('https://gmail.googleapis.com/$discovery/rest?version=v1')
print(response_API.status_code)

```

**输出:**

```py
200

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 相关的帖子，敬请关注 [Python @ AskPython](https://www.askpython.com/python) ，在此之前，祝你学习愉快！！🙂