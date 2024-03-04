# 在 Python 中使用 YouTube API

> 原文：<https://www.pythonforbeginners.com/api/using-the-youtube-api>

## 概观

在本帖中，我们将探讨如何在 Python 中使用 YouTube API。这个程序将展示我们如何使用 API 从 YouTube 中检索提要。这个特定的脚本将显示 YouTube 上目前最受欢迎的视频。

## 标准源

一些最受欢迎的 YouTube 源有:最近最受欢迎观看次数最多排名最高讨论次数最多最受欢迎链接次数最多特色最多回复次数

## 入门指南

为了从 YouTube 提要中获取我们想要的数据，我们首先要导入必要的模块。

```py
import requests
import json 
```

我们通过打印出程序所做的事情让它看起来更“漂亮”。然后，我们通过使用请求模块来获取提要。我以前使用 urllib2 模块来打开 URL，但是自从 Kenneth Reitz 给了我们请求模块之后，我就让这个模块来完成我的大部分 HTTP 任务。

```py
r = requests.get("http://gdata.youtube.com/feeds/api/standardfeeds/top_rated?v=2&alt=jsonc")
r.text 
```

在我们获得提要并将其保存到变量“r”之后，我们将它转换成一个 Python 字典。

```py
 data = json.loads(r.text) 
```

现在，我们有一个[使用 for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)来遍历数据

```py
for item in data['data']['items']: 
```

这有时可能是棘手的部分，你需要仔细观察这个结构是如何呈现给你的。使用一个 [JSON 编辑器](http://jsoneditoronline.org/ "jsoneditor")会使它变得更容易。

## 使用 YouTube API 获取数据

这个脚本将显示 YouTube 上最受欢迎的视频。

```py
# Import the modules
import requests
import json

# Make it a bit prettier..
print "-" * 30
print "This will show the Most Popular Videos on YouTube"
print "-" * 30

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

使用[字符串格式](https://www.pythonforbeginners.com/basics/strings-formatting)使得查看代码更加容易。

```py
 # Sample Output ------------------------------ This will show the Most Popular Videos on YouTube ------------------------------ Video Title: PSY - GANGNAM STYLE (?????) M/V Video Category: Music Video ID: 9bZkp7q19f0 Video Rating: 4.614460 Embed URL: http://www.youtube.com/watch?v=9bZkp7q19f0&feature=youtube_gdata_player Video Title: PSY - GENTLEMAN M/V Video Category: Music Video ID: ASO_zypdnsQ Video Rating: 4.372500 Embed URL: http://www.youtube.com/watch?v=ASO_zypdnsQ&feature=youtube_gdata_player Video Title: MACKLEMORE & RYAN LEWIS - THRIFT SHOP FEAT. WANZ (OFFICIAL VIDEO) Video Category: Music Video ID: QK8mJJJvaes Video Rating: 4.857624 Embed URL: http://www.youtube.com/watch?v=QK8mJJJvaes&feature=youtube_gdata_player
```

## 如何使用其他 YouTube 源

要使用另一个 YouTube 标准提要，只需替换 URL 中的提要:http://gdata.youtube.com/feeds/api/standardfeeds/**most _ responded**？v=2 & alt=jsonc 您可能需要修改脚本来获得预期的结果。