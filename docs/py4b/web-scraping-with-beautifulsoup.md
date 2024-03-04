# 美声网刮

> 原文：<https://www.pythonforbeginners.com/python-on-the-web/web-scraping-with-beautifulsoup>

## 网页抓取

"网络抓取(网络采集或网络数据提取)是一种从网站提取信息的计算机软件技术."

在 Python 中 HTML 解析很容易，尤其是在 BeautifulSoup 库的帮助下。在这篇文章中，我们将抓取一个网站(我们自己的)来提取所有的 URL。

## 入门指南

首先，确保您已经安装了必要的模块。在下面的例子中，我们在安装了 Python 2.7 的系统上使用了 [Beautiful Soup 4](http://www.crummy.com/software/BeautifulSoup/ "bs4") 和[请求](http://docs.python-requests.org/en/latest/ "requests")。使用 [pip](http://www.pip-installer.org/en/latest/ "pip_install") 可以安装 BeautifulSoup 和请求:

 `$ pip install requests`

`$ pip install beautifulsoup4` 

## 什么是美汤？

在他们[网站](http://www.crummy.com/software/BeautifulSoup/ "beautifulsoup")的顶部，你可以读到:“你没有写那可怕的一页。你只是想从中获取一些数据。美丽的汤是来帮忙的。自 2004 年以来，它已经为程序员节省了数小时或数天的快速屏幕抓取项目工作。

## 美丽的汤特色:

Beautiful Soup 为导航、搜索和修改解析树提供了一些简单的方法和 Pythonic 式的习惯用法:一个用于解析文档和提取所需内容的工具包。编写应用程序不需要太多代码。

Beautiful Soup 自动将传入文档转换为 Unicode，将传出文档转换为 UTF-8。您不必考虑编码，除非文档没有指定编码，而且 Beautiful Soup 不能自动检测编码。然后，您只需指定原始编码。

Beautiful Soup 位于 lxml 和 html5lib 等流行的 Python 解析器之上，允许您尝试不同的解析策略或以速度换取灵活性。

## 从任何网站提取 URL

现在，当我们知道 BS4 是什么，并且我们已经在我们的机器上安装了它，让我们看看我们可以用它做什么。

```py
 from bs4 import BeautifulSoup

import requests

url = raw_input("Enter a website to extract the URL's from: ")

r  = requests.get("http://" +url)

data = r.text

soup = BeautifulSoup(data)

for link in soup.find_all('a'):
    print(link.get('href')) 
```

当我们运行这个程序时，它会要求我们提供一个网站来提取网址

```py
 Enter a website to extract the URL's from: www.pythonforbeginners.com

> [通过实例学习 Python](https://www.pythonforbeginners.com/)

[https://www.pythonforbeginners.com/embed/#?secret=0SfA46x7bA](https://www.pythonforbeginners.com/embed/#?secret=0SfA46x7bA)

https://www.pythonforbeginners.com/python-overview-start-here/ 

https://www.pythonforbeginners.com/dictionary/ 

https://www.pythonforbeginners.com/python-functions-cheat-sheet/

> [列表](https://www.pythonforbeginners.com/basics/python-lists-cheat-sheet/)

[https://www.pythonforbeginners.com/basics/python-lists-cheat-sheet/embed/#?secret=VYLwzbHO5d](https://www.pythonforbeginners.com/basics/python-lists-cheat-sheet/embed/#?secret=VYLwzbHO5d) 

https://www.pythonforbeginners.com/loops/ 

https://www.pythonforbeginners.com/python-modules/ 

https://www.pythonforbeginners.com/strings/ 

https://www.pythonforbeginners.com/sitemap/ 

https://www.pythonforbeginners.com/feed/

> [通过实例学习 Python](https://www.pythonforbeginners.com/)

[https://www.pythonforbeginners.com/embed/#?secret=0SfA46x7bA](https://www.pythonforbeginners.com/embed/#?secret=0SfA46x7bA) 

.... .... ....

推荐大家阅读我们的介绍文章:[美人汤 4 Python](https://www.pythonforbeginners.com/beautifulsoup/beautifulsoup-4-python) 获取更多关于美人汤的知识和了解。

##### 更多阅读

[http://www.crummy.com/software/BeautifulSoup/](http://www.crummy.com/software/BeautifulSoup/ "beautifulsoup")

[http://docs.python-requests.org/en/latest/index.html](http://docs.python-requests.org/en/latest/index.html "requests")

