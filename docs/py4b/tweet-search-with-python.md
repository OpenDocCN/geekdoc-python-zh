# 使用 Python 进行 Tweet 搜索

> 原文：<https://www.pythonforbeginners.com/python-on-the-web/tweet-search-with-python>

## 概观

Twitter 的 API 是基于 REST 的，将返回 XML 或 JSON 格式的结果，以及 RSS 和 ATOM 提要格式的结果。任何客户端都可以访问公共时间轴，但所有其他 Twitter 方法都需要认证。

## 关于这个剧本

该计划是有据可查的，应该是直截了当的。打开一个文本编辑器，复制并粘贴下面的代码。

将文件另存为:“tweet_search.py”并退出编辑器。

## 入门指南

让我们来看看下面这个叫做 tweet_search.py 的程序

```py
#!/usr/bin/python

import json
import sys
import urllib2
import os

usage = """
Usage: ./tweet_search.py 'keyword'
e.g ./tweet_search.py pythonforbeginners

Use "+" to replace whitespace"
e.g ./tweet_search.py "python+for+beginners"
"""

# Check that the user puts in an argument, else print the usage variable, then quit.
if len(sys.argv)!=2:
    print (usage)
    sys.exit(0)

# The screen name in Twitter, is the screen name of the user for whom to return results for. 

# Set the screen name to the second argument
screen = sys.argv[1]

# Open the twitter search URL the result will be shown in json format
url = urllib2.urlopen("http://search.twitter.com/search.json?q="+screen)

#convert the data and load it into json
data = json.load(url)

#to print out how many tweets there are
print len(data), "tweets"

# Start parse the tweets from the result

# Get only text
for tweet in data["results"]:
    print tweet["text"]

# Get the status and print out the contents
for status in data['results']:
    print "(%s) %s" % (status["created_at"], status["text"]) 
```

## 它是如何工作的？

让我们分解这个脚本，看看它做了什么。

该脚本从导入我们将需要的模块开始

第 3-6 行

```py
import json
import sys
import urllib2
import os 
```

我们创建一个用法变量来解释如何使用这个脚本。

`**Line 8-14** usage = """ Usage: ./tweet_search.py 'keyword' e.g ./tweet_search.py pythonforbeginners Use "+" to replace whitespace" e.g ./tweet_search.py "python+for+beginners" """`

在第 16 行，我们检查用户是否输入了参数，否则打印用法变量，然后退出。

```py
 if len(sys.argv)!=2:
    print (usage)
    sys.exit(0) 
```

第 21-24 行将 Twitter 屏幕名称设置为第二个参数。

```py
 screen = sys.argv[1] 
```

第 27 行打开 twitter 搜索 URL，结果将以 json 格式显示。

```py
 url = urllib2.urlopen("http://search.twitter.com/search.json?q="+screen) 
```

第 30 行转换数据并将其加载到 json 中

```py
 data = json.load(url) 
```

在第 33 行，我们打印出了推文的数量

```py
 print len(data), "tweets" 
```

从第 38 行开始，我们开始从结果中解析 tweets

```py
for tweet in data["results"]:
    print tweet["text"] 
```

我们在这个脚本中做的最后一件事是获取状态并打印出内容(第 42 行)

```py
 for status in data['results']:
    print "(%s) %s" % (status["created_at"], status["text"]) 
```

一行一行地检查脚本，看看它做了什么。一定要看着它，试着去理解它。