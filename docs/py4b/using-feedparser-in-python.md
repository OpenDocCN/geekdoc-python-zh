# 在 Python 中使用 Feedparser

> 原文：<https://www.pythonforbeginners.com/feedparser/using-feedparser-in-python>

## 概观

在本帖中，我们将看看如何使用 Python 下载和解析整合的
提要。

我们将使用的 Python 模块是“Feedparser”。

完整的文档可以在[这里](https://pythonhosted.org/feedparser/ "feedparser_pythonhosted")找到。

## 什么是 RSS？

RSS 代表 Rich Site Summary，使用标准的 web feed 格式发布经常更新的信息:博客条目、新闻标题、音频、视频。

一个 RSS 文档(称为“feed”、“web feed”或“channel”)包括完整或
摘要文本，以及元数据，如出版日期和作者姓名。来源

## 什么是 Feedparser？

Feedparser 是一个 Python 库，可以解析所有已知格式的提要，包括
Atom、RSS 和 RDF。它可以在 Python 2.4 一直运行到 3.3。来源

## RSS 元素

在我们安装 feedparser 模块并开始编码之前，让我们看一看一些可用的 RSS 元素。

RSS 提要中最常用的元素是“标题”、“链接”、“描述”、“T0”、“发布日期”和“条目 ID”。

不太常用的元素是“图像”、“类别”、“附件”和“云”。

## 安装 Feedparser

要在你的电脑上安装 feedparser，打开你的终端，使用
" **pip"** (一个安装和管理 Python 包的工具)进行安装

sudo pip 安装 feedparser

要验证是否安装了 feedparser，可以运行一个“pip list”。

当然，您也可以进入交互模式，并在那里导入 feedparser
模块。

如果您看到如下输出，您可以确定它已经安装。

```py
>>> import feedparser
>>> 
```

既然我们已经安装了 feedparser 模块，我们就可以开始使用它了。

## 获取 RSS 源

你可以使用任何你想要的 RSS 源。因为我喜欢读 Reddit 的文章，所以我用 T2 的文章作为例子。

reddit 由许多子 Reddit 组成，我现在对
特别感兴趣的是“Python”子 Reddit。

获取 RSS 提要的方法是只需查找该子 reddit 的 URL 并
添加一个“rss”给它。

我们需要的 python 子 reddit 的 RSS 提要是:
http://www.reddit.com/r/python/.rss

## 使用 Feedparser

您从导入 feedparser 模块开始您的程序。

```py
import feedparser 
```

创建提要。放入你想要的 RSS 源。

```py
d = feedparser.parse('http://www.reddit.com/r/python/.rss') 
```

频道元素在 d.feed 中可用(还记得上面的“RSS 元素”吗)

这些项目在 d.entries 中可用，这是一个列表。

您访问列表中的项目的顺序与它们在
原始提要中出现的顺序相同，因此第一个项目在 d.entries[0]中可用。

打印源的标题

```py
print d['feed']['title']

>>> Python 
```

解析相对链接

```py
print d['feed']['link']

>>> http://www.reddit.com/r/Python/ 
```

解析转义的 HTML

```py
print d.feed.subtitle

>>> news about the dynamic, interpreted, interactive, object-oriented, extensible
programming language Python 
```

查看条目数量

```py
print len(d['entries'])

>>> 25 
```

提要中的每个条目都是一个字典。使用[0]打印第一个条目。

```py
print d['entries'][0]['title'] 

>>> Functional Python made easy with a new library: Funcy 
```

打印第一个条目及其链接

```py
print d.entries[0]['link'] 

>>> http://www.reddit.com/r/Python/comments/1oej74/functional_python_made_easy_with_a_new_
library/ 
```

使用 for 循环打印所有文章及其链接。

```py
for post in d.entries:
    print post.title + ": " + post.link + "
"

>>>
Functional Python made easy with a new library: Funcy: http://www.reddit.com/r/Python/
comments/1oej74/functional_python_made_easy_with_a_new_
library/

Python Packages Open Sourced: http://www.reddit.com/r/Python/comments/1od7nn/
python_packages_open_sourced/

PyEDA 0.15.0 Released: http://www.reddit.com/r/Python/comments/1oet5m/
pyeda_0150_released/

PyMongo 2.6.3 Released: http://www.reddit.com/r/Python/comments/1ocryg/
pymongo_263_released/
.....
.......
........ 
```

报告馈送类型和版本

```py
print d.version      

>>> rss20 
```

对所有 HTTP 头的完全访问权限

```py
print d.headers          	

>>> 
{'content-length': '5393', 'content-encoding': 'gzip', 'vary': 'accept-encoding', 'server':
"'; DROP TABLE servertypes; --", 'connection': 'close', 'date': 'Mon, 14 Oct 2013 09:13:34
GMT', 'content-type': 'text/xml; charset=UTF-8'} 
```

只需从头部获取内容类型

```py
print d.headers.get('content-type')

>>> text/xml; charset=UTF-8 
```

使用 feedparser 是解析 RSS 提要的一种简单而有趣的方式。

#### 来源

[http://www.slideshare.net/LindseySmith1/feedparser](http://www.slideshare.net/LindseySmith1/feedparser "slidehare")
http://code.google.com/p/feedparser/
