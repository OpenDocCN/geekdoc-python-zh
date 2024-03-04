# Python API 和 JSON

> 原文：<https://www.pythonforbeginners.com/json/python-api-and-json>

## 什么是 API？

应用编程接口(API)是一种协议，旨在被软件组件用作相互通信的接口。它基本上是一组用于访问基于 Web 的软件应用程序或 Web 工具的编程指令和标准。一家软件公司(如亚马逊、谷歌等)向公众发布其 API，以便其他软件开发商可以设计由其服务驱动的产品。关于 API 更详细的解释，请阅读来自 howstuffworks.com 的这篇优秀文章。

## 使用 JSON 与 API 交互

重要的是要知道 API 是软件到软件的接口，而不是用户接口。有了 API，应用程序可以在没有任何用户知识或干预的情况下相互交流。当我们想要与 Python 中的 API 交互时(比如访问 web 服务)，我们以一种叫做 JSON 的形式得到响应。为了与 json 交互，我们可以使用 JSON 和 simplejson 模块。JSON (JavaScript Object Notation)是一种紧凑的、基于文本的计算机数据交换格式，曾经像字典一样加载到 Python 中。JSON 数据结构直接映射到 Python 数据类型，这使得它成为直接访问数据的强大工具，而无需编写任何 XML 解析代码。

## 我该怎么做？

让我们展示如何通过使用 Twittes API 来实现这一点。你要做的第一件事，就是找到一个 URL 来调用 API。下一步是导入我们需要的模块。

```py
import json
import urllib2

# open the url and the screen name 
# (The screen name is the screen name of the user for whom to return results for)
url = "http://api.twitter.com/1/statuses/user_timeline.json?screen_name=python"

# this takes a python object and dumps it to a string which is a JSON
# representation of that object
data = json.load(urllib2.urlopen(url))

# print the result
print data
```

## 更多示例

[使用 Youtube API](https://www.pythonforbeginners.com/api/using-the-youtube-api "youtube")

[使用 Vimeo API](https://www.pythonforbeginners.com/api/how-to-use-the-vimeo-api-in-python "vimeo")

[使用 Twitter API](https://www.pythonforbeginners.com/code-snippets-source-code/tweet-search-with-python "twitter")

[使用美味的 API](http://www.michael-noll.com/projects/delicious-python-api/ "delicious")

[LastFM API](https://www.last.fm/api "lastfm")

[亚马逊 API](https://aws.amazon.com/code/Python?browse=1 "Amazon")

[谷歌应用编程接口](https://developers.google.com/maps/ "google")

##### 来源

[http://money . how stuff works . com/business-communication s/how-to-leverage-an-API-for-conferencing 1 . htm](http://money.howstuffworks.com/business-communications/how-to-leverage-an-api-for-conferencing1.htm "howto")