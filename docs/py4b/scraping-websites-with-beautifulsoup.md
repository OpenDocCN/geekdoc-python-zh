# 用 Python 抓取网站

> 原文：<https://www.pythonforbeginners.com/beautifulsoup/scraping-websites-with-beautifulsoup>

## 什么是 BeautifulSoup？

BeautifulSoup 是来自 Crummy 的第三方 Python 库。

该库是为像屏幕抓取这样的快速周转项目而设计的

## 我能做什么

Beautiful Soup 解析你给它的任何东西，并为你遍历树。

你可以用它来查找一个网站的所有链接

查找所有 URL 与“foo.com”匹配的链接

找到有粗体文本的表格标题，然后给我那个文本。

找到每一个有 href 属性的“a”元素。

## 我需要什么？

您需要首先安装 BeautifulSoup 模块，然后将
模块导入到您的脚本中。

可以用 pip install beautifulsoup4 或者 easy_install beautifulsoup4 安装。

在最近版本的 Debian 和 Ubuntu 中，它也可以作为 python-beautifulsoup4 包使用。

Beautiful Soup 4 在 Python 2 (2.6+)和 Python 3 上都可以工作。

## 美丽的例子

在开始之前，我们必须导入两个模块=> BeutifulSoup 和 urllib2

Urlib2 用来打开我们想要的 URL。

因为 BeautifulSoup 没有为您获取网页，所以您必须使用 urllib2 模块来完成这项工作。

```py
#import the library used to query a website
import urllib2 
```

## 搜索并查找所有 html 标签

我们将使用 soup.findAll 方法来搜索 soup 对象，以
匹配页面中的文本和 html 标签。

```py
from BeautifulSoup import BeautifulSoup

import urllib2 
url = urllib2.urlopen("http://www.python.org")

content = url.read()

soup = BeautifulSoup(content)

links = soup.findAll("a") 
```

这将打印出 python.org 所有带有“a”标签的元素。

这是定义超链接的标签，用于从一个页面
链接到另一个页面

## 在 Reddit 上查找所有链接

使用 Python 内置的 urllib2 模块获取 Reddit 网页的 HTML。

一旦我们有了页面的实际 HTML，我们就创建一个新的 BeautifulSoup
类来利用它的简单 API。

```py
from BeautifulSoup import BeautifulSoup

import urllib2

pageFile = urllib2.urlopen("http://www.reddit.com")

pageHtml = pageFile.read()

pageFile.close()

soup = BeautifulSoup("".join(pageHtml))

#sAll = soup.findAll("li")

sAll = soup.findAll("a")

for href in sAll:
    print href 
```

## 网站废了《赫芬顿邮报》

这是我在 newthinktank.com 看到的另一个例子

```py
from urllib import urlopen

from BeautifulSoup import BeautifulSoup

import re

# Copy all of the content from the provided web page
webpage = urlopen('http://feeds.huffingtonpost.com/huffingtonpost/LatestNews').read()

# Grab everything that lies between the title tags using a REGEX
patFinderTitle = re.compile('')

# Grab the link to the original article using a REGEX
patFinderLink = re.compile('')

# Store all of the titles and links found in 2 lists
findPatTitle = re.findall(patFinderTitle,webpage)

findPatLink = re.findall(patFinderLink,webpage)

# Create an iterator that will cycle through the first 16 articles and skip a few
listIterator = []

listIterator[:] = range(2,16)

soup2 = BeautifulSoup(webpage)

#print soup2.findAll("title")

titleSoup = soup2.findAll("title")

linkSoup = soup2.findAll("link")

for i in listIterator:
    print titleSoup[i]
    print linkSoup[i]
    print "
" 
```

##### 更多阅读

[http://www.crummy.com/software/BeautifulSoup/](http://www.crummy.com/software/BeautifulSoup/ "crummy") http://www . new thinktank . com/2010/11/python-2-7-tutorial-pt-13-website-scraping/
[http://kochi-coders.com/?p=122](http://kochi-coders.com/?p=122 "BeautifulSoup")