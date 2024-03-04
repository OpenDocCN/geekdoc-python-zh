# 如何用 Python 访问各种 Web 服务

> 原文：<https://www.pythonforbeginners.com/python-on-the-web/how-to-access-various-web-services-in-python>

## 概观

学习 Python 的一个非常好的方法是尝试使用各种 Web 服务
API。

#### 如何访问 Youtube、Vimeo、Twitter 等网络服务？

为了回答这个问题，我们首先要了解一些关于 API、
JSON、数据结构等等的知识。

## 入门指南

对于那些关注我们的人来说，希望你已经获得了一些基本的 Python 知识。对于还没有看过的人，我建议你开始阅读我们网站顶部的
页面，或者点击下面你想让
了解更多的链接。

[Python 教程](https://www.pythonforbeginners.com/python-tutorial/)

[基础知识(概述)](https://www.pythonforbeginners.com/basics/python-quick-guide "basics_python")

[字典](/dictionary/ "dictionary_python")

[功能](https://www.pythonforbeginners.com/basics/python-functions-cheat-sheet "functions")

[列表](https://www.pythonforbeginners.com/basics/python-lists-cheat-sheet "lists_python")

[循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python "loops_python")

[模块](https://www.pythonforbeginners.com/modules-in-python/python-modules "modules_python")
 [琴弦](https://www.pythonforbeginners.com/basics/strings "strings_python")

## API:应用编程接口

API 是一种协议，旨在被软件组件用作相互通信的接口。API 是一组编程指令和
标准，用于访问基于 web 的软件应用程序(如上所述)。

借助 API，应用程序可以在没有任何用户知识或
干预的情况下相互对话。
通常，像谷歌、Vimeo 和 Twitter 这样的公司向公众发布它的 API
，这样开发者就可以开发基于它的服务的产品。

重要的是要知道 API 是软件到软件的接口，而不是用户接口。

## API 密钥

互联网上的许多服务(如 Twitter、脸书..)要求你
有一个“API 密匙”。

应用编程接口密钥(API key)是由调用 API 的
计算机程序传入的代码，用于向网站标识调用程序、其开发者、
或其用户。

API 密钥用于跟踪和控制 API 的使用方式，例如
防止恶意使用或滥用 API。

API 密钥通常既作为唯一的标识符，又作为用于
认证的秘密令牌，并且通常具有一组对与之相关联的 API
的访问权限。
当我们与一个 API 交互时，我们经常以一种叫做 JSON 的形式得到响应。

## Json

让我们快速而不深入地了解一下 JSON 是什么。JSON (JavaScript Object Notation)是一种紧凑的、基于文本的格式，用于计算机交换数据。

它建立在两个结构之上:
——名称/值对的集合

–有序的值列表。JSON 采用这些形式:对象、数组、值、字符串、数字

对象
–无序的名称/值对集合。
–以{开始，以}结束。
–每个名称后跟:(冒号)
–名称/值对由(逗号)分隔。
数组
–有序的数值集合。
–以【开始，以】结束。
–值由(逗号)分隔。
值
–可以是双引号中的字符串，数字，或真或假或空，
或者是对象或数组。
字符串
–零个或多个 Unicode 字符的序列，用双引号
括起来，使用反斜杠转义。
数字
–整数、长整型、浮点型

## 访问 Web 服务

Python 为我们提供了与 json 交互的 [json](https://docs.python.org/2/library/json.html "json_python") 和 [simplejson](https://pypi.python.org/pypi/simplejson "simplejson") 模块。
这个时候，我们应该知道 API 是什么，它的作用是什么。另外，我们现在
知道了 JSON 的基础知识。

为了开始访问 web 服务，我们首先需要找到一个 URL 来调用 API。

在我们获得 URL 之前，我真的建议你阅读提供的文档
(如果有的话)。

文档描述了如何使用 API，并包含了关于我们如何与之交互的重要信息。
我们需要的 URL 通常可以在公司网站
上找到，与 API 文档在同一个地方。

举个例子:

YouTube
[http://g data . YouTube . com/feeds/API/standard feeds/most _ popular？v=2 & alt=json](https://gdata.youtube.com/feeds/api/standardfeeds/most_popular?v=2&alt=json "youtube_jsonc")

http://vimeo.com/api/v2/video/video_id.output

Reddit
[http://www.reddit.com/user/spilcm/comments/.json](https://www.reddit.com/user/spilcm/comments/.json "reddit_json")

请不要认为这些会过时，因此，请确认您拥有最新的
版本。

当你有了一个 URL 并且你已经阅读了提供的文档，我们从
导入我们需要的模块开始。

## 我需要什么模块？

我在使用 JSON 时通常使用的模块有:
–[请求](http://docs.python-requests.org/en/latest/ "requests_module")
–[JSON](https://docs.python.org/2/library/json.html "json_module")(或[simple JSON](https://pypi.python.org/pypi/simplejson "simplejson_module"))
–[pprint](https://docs.python.org/2/library/pprint.html "pprint_module")

我曾经使用 urllib2 模块来打开 URL，但是自从 Kenneth Reitz
给了我们请求模块，我就让这个模块来完成我的大部分 HTTP 任务。

## 处理数据

一旦您知道您需要哪个 URL 并导入了必要的模块，我们就可以使用请求模块来获取 JSON 提要。

r = requests . get(" http://www . Reddit . com/user/spil cm/about/。json")
r.text

您可以将输出复制并粘贴到 JSON 编辑器中，以便更容易地查看数据。

我使用[http://jsoneditoronline.org/](http://jsoneditoronline.org/ "jsoneditor")，但是任何 JSON 编辑器都应该做这项工作。

下一步是将 JSON 输出转换成 Python 字典。

## 转换数据

这将获取 JSON 字符串并使其成为一个字典:
json.loads(r.text)

注意:还可以使用
json.dumps()将 python 对象序列化为 JSON。

然而，这不是我们现在想做的。

## 遍历结果

我们知道有一个 python 字典，我们可以开始使用它来获得我们想要的结果。

一种常见的方法是循环结果，得到你感兴趣的数据。

这有时可能是棘手的部分，你需要仔细观察
结构是如何呈现的。
同样，使用 Json 编辑器会更容易。

## 使用 YouTube API

此时，我们应该有足够的知识和信息来创建一个程序

该程序将显示 YouTube 上最受欢迎的视频。

```py
#Import the modules
import requests
import json

# Get the feed
r = requests.get("http://gdata.youtube.com/feeds/api/standardfeeds/top_rated?v=2&alt=jsonc")
r.text

# Convert it to a Python dictionary
data = json.loads(r.text)

# Loop through the result.
for item in data['data']['items']:

    print "Video Title: %s" % (item['title'])

    print "Video Category: %s" % (item['category'])

    print "Video ID: %s" % (item['id'])

    print "Video Rating: %f" % (item['rating'])

    print "Embed URL: %s" % (item['player']['default'])

    print
```

看看我们如何遍历结果来获得我们想要的键和值。

## YouTube、Vimeo 和 Twitter 示例

[如何在 Python 中使用 YouTube API](https://www.pythonforbeginners.com/api/using-the-youtube-api "youtube_pfb")
 [如何在 Python 中使用 Vimeo API](https://www.pythonforbeginners.com/api/how-to-use-the-vimeo-api-in-python "vimeo_api_pfb")

[如何使用 Python 中的 Twitter API](https://www.pythonforbeginners.com/code-snippets-source-code/tweet-search-with-python "twitter_script_pfb")

[解析 JSON](https://www.pythonforbeginners.com/json/parsingjson "parsing_json_pfb")

## 各种 Web 服务的 API 文档

YouTube
[https://developers . Google . com/YouTube/2.0/developers _ guide _ JSON](https://developers.google.com/youtube/2.0/developers_guide_json "youtube_api_doc")

http://developer.vimeo.com/apis/

推特
[https://dev.twitter.com/docs/api/1.1/overview](https://dev.twitter.com/docs/api/1.1/overview "twitter_api_doc")

Reddit
[http://www.reddit.com/dev/api](https://www.reddit.com/dev/api "reddit_api")