# Python 命令行 IMDB Scraper

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/imdb-crawler>

## 概观

这个脚本将要求输入电影名称和年份，然后向 IMDB 查询。

## 命令行 IMDB 刮刀

第一步是导入必要的模块。

```py
#!/usr/bin/env python27

#Importing the modules

from BeautifulSoup import BeautifulSoup
import sys
import urllib2
import re
import json

#Ask for movie title
title = raw_input("Please enter a movie title: ")

#Ask for which year
year = raw_input("which year? ")

#Search for spaces in the title string
raw_string = re.compile(r' ')

#Replace spaces with a plus sign
searchstring = raw_string.sub('+', title)

#Prints the search string
print searchstring

#The actual query
url = "http://www.imdbapi.com/?t=" + searchstring + "&y="+year

request = urllib2.Request(url)

response = json.load(urllib2.urlopen(request))

print json.dumps(response,indent=2) 
```

好好享受吧！