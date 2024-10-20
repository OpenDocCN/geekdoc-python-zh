# 用 Python 从 API 中提取数据——详细指南！

> 原文：<https://www.askpython.com/python/examples/pull-data-from-an-api>

读者朋友们，你们好！在本文中，我们将关注如何从 Python 中的 API 中提取数据。

所以，让我们开始吧！

* * *

## 使用 Python 从 API 提取数据的步骤

现在让我们来关注一下从 API 中提取特定数据需要遵循的步骤。

您可以查看关于 **[连接到 API](https://www.askpython.com/python/examples/connect-and-call-apis)** 的文章，以了解更多关于 API 和响应状态代码等的信息。

让我们开始吧！

* * *

### 示例 1:从开源 COVID API 中提取数据

在这个例子中，我们将连接到一个开源的 COVID API，只是为了以定制的方式提取和解析 json 信息。

* * *

#### 1.连接到 API

首先，我们需要连接到一个 API 并建立一个安全连接，如下所示

在本文中，我们使用了 [COVID19-India API](https://data.covid19india.org/) 从 state-wise 列表中获取案例数据。

```py
import requests
import json
response_API = requests.get('https://api.covid19india.org/state_district_wise.json')
#print(response_API.status_code)

```

当我们从 API 中提取数据时，我们使用了`get()`函数从 API 中获取信息。

* * *

#### 2.从 API 获取数据

在与 API 建立了健康的连接之后，下一个任务是从 API 中提取数据。看下面的代码！

```py
data = response_API.text

```

`requests.get(api_path).text`帮助我们从提到的 API 中提取数据。

* * *

#### 3.将数据解析成 JSON 格式

提取完数据后，现在是将数据转换和解码成正确的 JSON 格式的时候了，如下所示

```py
json.loads(data)

```

[json.loads()函数](https://www.askpython.com/python/examples/read-a-json-file-in-python)将数据解析成 **JSON** 格式。

* * *

#### 4.提取数据并打印出来

JSON 格式包含的数据是类似于 [Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)的键值格式。因此，我们可以使用如下所示的关键值提取并打印数据

```py
parse_json['Andaman and Nicobar Islands']['districtData']['South Andaman']['active']

```

* * *

#### 你可以在下面找到完整的代码！

```py
import requests
import json
response_API = requests.get('https://api.covid19india.org/state_district_wise.json')
#print(response_API.status_code)
data = response_API.text
parse_json = json.loads(data)
active_case = parse_json['Andaman and Nicobar Islands']['districtData']['South Andaman']['active']
print("Active cases in South Andaman:", active_case)

```

**输出:**

```py
Active cases in South Andaman: 19

```

* * *

### 示例 2:从开源 GMAIL API 中提取数据

现在，让我们连接并从 [GMAIL API](https://developers.google.com/gmail/api/reference/rest) 中提取数据。这个 API 代表了我们可以从 API 中获取的一般结构和信息。

所以，让我们开始吧！

看看下面的代码吧！

**举例:**

```py
import requests
import json
response_API = requests.get('https://gmail.googleapis.com/$discovery/rest?version=v1')
#print(response_API.status_code)
data = response_API.text
parse_json = json.loads(data)
info = parse_json['description']
print("Info about API:\n", info)
key = parse_json['parameters']['key']['description']
print("\nDescription about the key:\n",key)

```

**输出:**

```py
Info about API:
 The Gmail API lets you view and manage Gmail mailbox data like threads, messages, and labels.

Description about the key:
 API key. Your API key identifies your project and provides you with API access, quota, and reports. Required unless you provide an OAuth 2.0 token.

```

**说明:**

*   首先，我们使用`get()`函数连接到通用的 GMAIL API。
*   在与 API 形成健康的连接后，我们使用`response_object.text` 从 API 获取数据
*   现在，我们使用`json.loads()`函数将数据解析成 JSON 格式。
*   最后，我们从 JSON 对象中提取数据，比如 API 的描述和键的描述。
*   您可以通过访问示例中提到的 API 链接来交叉检查这些值。

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，请随时在下面评论。

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！！🙂