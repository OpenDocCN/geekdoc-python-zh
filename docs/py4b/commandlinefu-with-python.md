# 使用 Python 的 CommandLineFu

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/commandlinefu-with-python>

## 概观

练习 Python 编码的最好方法之一是研究一些代码，并亲自尝试。通过做大量的代码练习，您将会对它真正做的事情有更好的理解。

换句话说，边做边学。为了帮助你提高你的 Python 编码技能，我创建了一个程序，它使用了 CommandLineFu.com 的 API

## CommandLineFu API

当您想要使用基于 Web 的服务时，常见的第一步是查看它们是否有 API。

幸运的是，[Commandlinefu.com](http://www.Commandlinefu.com "commandlinefudotcom")提供了一个，可以在这里找到[:](https://www.commandlinefu.com/site/api "commandlinefu-api")

“commandlinefu.com 的内容有多种不同的格式，你可以随心所欲。任何包含命令列表的页面(例如按标签、函数或用户列出的列表)都可以以您选择的格式返回，只需简单地更改请求 URL。

他们提供的示例 URL 是:

*http://www . command line fu . com/commands/'命令集'/'格式'/*

其中:command-set 是指定要返回哪组命令的 URL 组件。

可能的值有:

*   浏览/按投票排序
*   标记的/163/grep
*   匹配/ssh/c3No

格式是下列之一:

*   纯文本
*   json
*   简易资讯聚合

我更喜欢使用 json 格式，这也是我在下面的程序中使用的格式。

## 创建“命令行搜索工具”

我想我们已经有了所有需要的 API 信息，所以让我们开始吧。该计划是有据可查的，应该是直截了当的。

打开一个文本编辑器，复制并粘贴下面的代码。将文件另存为:“commandlinefu.py”并退出编辑器。

```py
#!/usr/bin/env python27
import urllib2
import base64
import json
import os
import sys
import re

os.system("clear")
print "-" * 80
print "Command Line Search Tool"
print "-" * 80

def Banner(text):
    print "=" * 70
    print text
    print "=" * 70
    sys.stdout.flush()

def sortByVotes():
    Banner('Sort By Votes')
    url = "http://www.commandlinefu.com/commands/browse/sort-by-votes/json"
    request = urllib2.Request(url)
    response = json.load(urllib2.urlopen(request))
    #print json.dumps(response,indent=2)
    for c in response:
        print "-" * 60
        print c['command']

def sortByVotesToday():
    Banner('Printing All commands the last day (Sort By Votes) ')
    url = "http://www.commandlinefu.com/commands/browse/last-day/sort-by-votes/json"
    request = urllib2.Request(url)
    response = json.load(urllib2.urlopen(request))
    for c in response:
        print "-" * 60
        print c['command']

def sortByVotesWeek():
    Banner('Printing All commands the last week (Sort By Votes) ')
    url = "http://www.commandlinefu.com/commands/browse/last-week/sort-by-votes/json"
    request = urllib2.Request(url)
    response = json.load(urllib2.urlopen(request))
    for c in response:
        print "-" * 60
        print c['command']

def sortByVotesMonth():
    Banner('Printing: All commands from the last months (Sorted By Votes) ')
    url = "http://www.commandlinefu.com/commands/browse/last-month/sort-by-votes/json"
    request = urllib2.Request(url)
    response = json.load(urllib2.urlopen(request))
    for c in response:
        print "-" * 60
        print c['command']

def sortByMatch():
    #import base64
    Banner("Sort By Match")
    match = raw_input("Please enter a search command: ")
    bestmatch = re.compile(r' ')
    search = bestmatch.sub('+', match)
    b64_encoded = base64.b64encode(search)
    url = "http://www.commandlinefu.com/commands/matching/" + search + "/" + b64_encoded + "/json"
    request = urllib2.Request(url)
    response = json.load(urllib2.urlopen(request))
    for c in response:
        print "-" * 60
  print c['command']

print """
1\. Sort By Votes (All time)
2\. Sort By Votes (Today)
3\. Sort by Votes (Week)
4\. Sort by Votes (Month)
5\. Search for a command

Press enter to quit
"""

while True:
  answer = raw_input("What would you like to do? ")

 if answer == "":
    sys.exit()

  elif answer == "1":
   sortByVotes()

  elif answer == "2":
   print sortByVotesToday()

  elif answer == "3":
   print sortByVotesWeek()

  elif answer == "4":
   print sortByVotesMonth()

  elif answer == "5":
   print sortByMatch()

  else:
   print "Not a valid choice" 
```

当你运行这个程序时，你会看到一个菜单，你可以从中做出选择。

```py
 --------------------------------------------------------------------------------
Command Line Search Tool
--------------------------------------------------------------------------------

1\. Sort By Votes (All time)
2\. Sort By Votes (Today)
3\. Sort by Votes (Week)
4\. Sort by Votes (Month)
5\. Search for a command

Press enter to quit

What would you like to do?
...
... 
```