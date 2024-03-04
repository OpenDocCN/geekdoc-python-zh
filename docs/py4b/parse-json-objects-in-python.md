# 用 Python 解析 JSON 对象

> 原文：<https://www.pythonforbeginners.com/json/parse-json-objects-in-python>

## 概观

在这篇文章中，我们将解释如何用 Python 解析 JSON 对象。

当您想从各种 web 服务访问一个 API
并以 JSON 给出响应时，知道如何解析 JSON 对象是很有用的。

## 入门指南

你要做的第一件事，就是找到一个 URL 来调用 API。

在我的例子中，我将使用 Twitter API。

从导入程序所需的模块开始。

```py
import json
import urllib2 
```

打开 URL 和屏幕名称。

```py
url = "http://api.twitter.com/1/statuses/user_timeline.json?screen_name=wordpress" 
```

打印出结果

```py
print data 
```

## 使用 Twitter API 解析数据

这是一个非常简单的程序，只是让你知道它是如何工作的。

```py
#Importing modules
import json
import urllib2

# Open the URL and the screen name
url = "http://api.twitter.com/1/statuses/user_timeline.json?screen_name=wordpress"

# This takes a python object and dumps it to a string which is a JSON representation of that object
data = json.load(urllib2.urlopen(url))

#print the result
print data 
```

如果你有兴趣看另一个如何在 Python 中使用 JSON 的例子，请
看看[“IMDB 爬虫”脚本](https://www.pythonforbeginners.com/code-snippets-source-code/imdb-crawler)。

要使用 Twitter API，请参阅 Twitter 上的官方文档。
[https://dev.twitter.com/docs](https://dev.twitter.com/docs "twitter-docs")