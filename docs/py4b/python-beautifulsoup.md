# 在 Python 中使用 urllib2 和 BeautifulSoup

> 原文：<https://www.pythonforbeginners.com/beautifulsoup/python-beautifulsoup>

在昨天的帖子里，我给了 BeautifulSoup 一个[介绍。](https://www.pythonforbeginners.com/beautifulsoup/python-beautifulsoup-basic)

因为 BeautifulSoup 没有为您获取网页，所以您必须使用 urllib2 模块来完成这项工作。

### 美丽的例子

请查看代码中的注释，看看它做了什么

```py
#import the library used to query a website
import urllib2

#specify the url you want to query
url = "http://www.python.org"

#Query the website and return the html to the variable 'page'
page = urllib2.urlopen(url)

#import the Beautiful soup functions to parse the data returned from the website
from BeautifulSoup import BeautifulSoup

#Parse the html in the 'page' variable, and store it in Beautiful Soup format
soup = BeautifulSoup(page)

#to print the soup.head is the head tag and soup.head.title is the title tag
print soup.head
print soup.head.title

#to print the length of the page, use the len function
print len(page)

#create a new variable to store the data you want to find.
tags = soup.findAll('a')

#to print all the links
print tags

#to get all titles and print the contents of each title
titles = soup.findAll('span', attrs = { 'class' : 'titletext' })
for title in allTitles:
    print title.contents 
```

在接下来的文章中，我会写更多关于这个令人敬畏的模块。