# 如何在 Python 中使用 Reddit API

> 原文：<https://www.pythonforbeginners.com/api/how-to-use-reddit-api-in-python>

## Reddit API–概述

在早先的一篇文章“[如何用 Python](https://www.pythonforbeginners.com/python-on-the-web/how-to-access-various-web-services-in-python "how_to_access_web_services") 访问各种 Web 服务”中，我们描述了
如何通过 API 访问 YouTube、Vimeo 和 Twitter 等服务。

注意，有几个 [Reddit 包装器](https://github.com/reddit/reddit/wiki/API-Wrappers "api_wrappers_reddit")可以用来与 Reddit 交互。

包装器是一个 API 客户端，通常用于通过调用 API 本身将 API 包装成易于使用的函数。

这导致它的用户不太关心代码
实际上是如何工作的。

如果你不使用包装器，你将不得不直接访问 Reddits API，
这正是我们在本文中要做的。

## 入门指南

既然我们要关注 Reddit 的 API，那就让我们去看看他们的 [API
文档](https://www.reddit.com/dev/api "reddit_api")。我建议你熟悉文档，并且
额外注意概述和关于“模式哈希”、“T4”、“全名”和“类型前缀”的章节。

API 的结果将作为 XML 或 JSON 返回。在这篇文章中，我们将使用 JSON 格式。

关于 JSON 结构的更多信息
，请参考上面的帖子或者[官方文档](https://docs.python.org/2/library/json.html "json_off_docs")。

## API 文档

在 API 文档中，您可以看到有大量的事情要做。

在这篇文章中，我们选择从我们自己的 Reddit 帐户中提取信息。

我们需要的信息是: **GET /user/username/where[。json |。xml ]**

获取/用户/用户名/where[。json |。xml ]

？/用户/用户名/概述
？/用户/用户名/提交的
？/用户/用户名/评论
？/用户/用户名/喜欢的
？/用户/用户名/不喜欢的
？/用户/用户名/隐藏
？/用户/用户名/已保存

## 查看 JSON 输出

例如，如果我们想使用“评论”，URL 将是:
**【http://www.reddit.com/user/spilcm/comments/.json】**

您可以看到，我们已经用自己的输入替换了“用户名”和“位置”。

要查看数据响应，您可以发出一个 curl 请求，如下所示:

```py
curl http://www.reddit.com/user/spilcm/comments/.json 
```

…或者只需将 URL 粘贴到您的浏览器中。

可以看到响应是 JSON。这可能很难在
浏览器中看到，除非你安装了 JSONView 插件。

这些扩展可用于[火狐](https://addons.mozilla.org/en-us/firefox/addon/jsonview/ "jsonview_ff")和 [Chrome](https://chrome.google.com/webstore/detail/jsonview/chklaanhfefbnpoihckbnefhakgolnmc "jsonview_chrome") 。

## 开始编码

现在我们有了 URL，让我们开始做一些编码。

打开你最喜欢的空闲/编辑器，导入我们需要的模块。

导入模块。pprint 和 json 模块是可选的。

```py
from pprint import pprint

import requests

import json 
```

## 进行 API 调用

现在是时候对 Reddit 进行 API 调用了。

```py
r = requests.get(r'http://www.reddit.com/user/spilcm/comments/.json') 
```

现在，我们有一个名为“r”的响应对象。我们可以从这个物体中获得我们需要的所有信息。

## JSON 响应内容

Requests 模块带有一个内置的 JSON 解码器，我们可以用它来处理 JSON 数据。

正如你在上面的图片中看到的，我们得到的输出并不是我们真正想要显示的。

问题是，我们如何从中提取有用的数据？

如果我们只想查看“r”对象中的键:

```py
r = requests.get(r'http://www.reddit.com/user/spilcm/comments/.json')

data = r.json()

print data.keys() 
```

这将为我们提供以下输出:

[u'kind '，u'data']

这些钥匙对我们来说非常重要。

现在是获取我们感兴趣的数据的时候了。

获取 JSON 提要并将输出复制/粘贴到 JSON 编辑器中，以便更容易地查看数据。

一种简单的方法是将 JSON 结果粘贴到在线 JSON 编辑器中。

我使用[http://jsoneditoronline.org/](http://jsoneditoronline.org/ "json_editor_online")，但是任何 JSON 编辑器都应该做这项工作。

让我们来看一个例子:

```py
r = requests.get(r'http://www.reddit.com/user/spilcm/comments/.json')
r.text
```

从图中可以看出，我们得到了与之前
打印密钥时相同的密钥(种类、数据)。

## 将 JSON 转换成字典

让我们将 JSON 数据转换成 Python 字典。

你可以这样做:

```py
r.json()

#OR

json.loads(r.text) 
```

现在，当我们有了一个 Python 字典，我们开始使用它来获得我们想要的结果。

## 导航以查找有用的数据

沿着你的路往下走，直到你找到你想要的。

```py
r = requests.get(r'http://www.reddit.com/user/spilcm/comments/.json')

r.text

data = r.json()

print data['data']['children'][0] 
```

结果存储在变量“数据”中。

为了访问我们的 JSON 数据，我们简单地使用括号符号，就像这样:
data['key']。

记住数组是从零开始索引的。

我们可以使用 for 循环遍历字典，而不是打印每一个条目。

```py
for child in data['data']['children']:

    print child['data']['id'], "
", child['data']['author'],child['data']['body']

    print 
```

我们可以像这样访问任何我们想要的东西，只要查找你感兴趣的数据。

## 完整的剧本

正如您在我们完整的脚本中看到的，我们只需要导入一个模块:
(请求)

```py
import requests

r = requests.get(r'http://www.reddit.com/user/spilcm/comments/.json')

r.text

data = r.json()

for child in data['data']['children']:
    print child['data']['id'], "
", child['data']['author'],child['data']['body']
    print 
```

当您运行该脚本时，您应该会看到类似如下的内容:

##### 更多阅读

[http://docs.python-requests.org/en/latest/](http://docs.python-requests.org/en/latest/ "requests_")