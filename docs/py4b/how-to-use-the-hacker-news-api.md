# 如何使用黑客新闻 API

> 原文：<https://www.pythonforbeginners.com/api/how-to-use-the-hacker-news-api>

## 黑客新闻 API–概述

今天我将浏览“黑客新闻的非官方 Python API”，可以在这里找到

## 什么是黑客新闻？

Hacker News 是一个面向程序员和企业家的社交新闻网站，提供与计算机科学和企业家精神相关的内容。[ [来源](https://en.wikipedia.org/wiki/Hacker_News "hackernews")

## Python 包装器

在[如何在 Python 中使用 Reddit API](https://www.pythonforbeginners.com/api/how-to-use-reddit-api-in-python "reddit_api")中，我们描述了如何直接访问 Reddit API。另一种方法是使用一个 [Reddit 包装器](https://github.com/praw-dev/praw "praw")。包装器是一个 API 客户端，通常用于通过调用 API 本身将 API 包装成易于使用的函数。当使用包装器时，您不必关心幕后发生了什么，这对初学者来说有时会更容易。可以把它想象成 python 和 web 服务之间的接口。

## 入门指南

让我们开始使用 [pip](http://www.pip-installer.org/en/latest/ "pip") 工具安装它。

```py
pip search HackerNews
HackerNews                - Python API for Hacker News.

pip install HackerNews 
```

已成功安装 HackerNews 清理… pip 显示 HackerNews —名称:HackerNews 版本:1.3.3 位置:/usr/local/lib/python 2.7/dist-packages 需要:BeautifulSoup4

## API 文档

建议您阅读文档，该文档可在这个 [Github](https://github.com/thekarangoel/HackerNewsAPI "github_api") 页面上获得。该 API 包含几个类(HN 和故事)。这些类为您提供了方法。HN 类可用的方法有:get_top_stories()从 HN 主页返回一个故事对象列表 get_newest_stories()从 HN 最新页面返回一个故事对象列表 Story 类可用的方法有:Print _ Story()–打印一个故事的细节

## 运行程序

API 的作者在他的 Github 页面上提供了一个例子。该示例打印 Hacker news 的热门文章和新文章。打开您最喜欢的编辑器，复制并粘贴以下代码。

```py
#!/usr/bin/env python

from hn import HN

hn = HN()

# print top 10 stories from homepage
for story in hn.get_top_stories()[:10]:
    story.print_story()
    print '*' * 50
    print ''

# print 10 latest stories
for story in hn.get_newest_stories()[:10]:
    story.print_story()
    print '*' * 50
    print ''

# print all self posts from the homepage
for story in hn.get_top_stories():
    if story.is_self_post:
        story.print_story()
        print '*' * 50
        print '' 
```

将其另存为 test_bot.py 并运行。该程序将循环浏览黑客新闻上的每个故事(帖子),并给出 10 个最新的故事。对于每个帖子，它将显示在故事详细信息排名下看到的信息——页面上故事的排名 Story _ id——故事的 id 标题——故事的标题是 _ self _ post——真实的自我/工作故事链接——它指向的 url(无自我帖子)域——链接的域(无自我帖子) 点数–故事提交者的点数/因果关系–提交故事提交者的用户 _ 链接–上面的用户资料链接发布时间–发布时间之前的数量 _ 评论–评论数量 _ 链接–使用 python 包装的 API 到评论页面的链接很好也很简单，但请尝试理解幕后发生的事情。 理解代码中发生了什么是很重要的，一旦你学会了，你就可以使用包装器了。尝试获取关于 Json】和 API 的知识，了解它们大多数是如何工作的。

## 更多阅读

[HackerNewsAPI](https://github.com/thekarangoel/HackerNewsAPI "hackernews_api")

[Python API 列表](https://www.pythonforbeginners.com/api/how-to-use-the-hacker-news-api)