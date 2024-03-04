# BeautifulSoup 简介

> 原文：<https://www.pythonforbeginners.com/beautifulsoup/python-beautifulsoup-basic>

### 什么是 BeautifulSoup？

```py
 BeautifulSoup is a Python library from [www.crummy.com](http://www.crummy.com/software/BeautifulSoup/ "crummy") 
```

### 它能做什么

```py
 On their website they write "Beautiful Soup parses anything you give it, and does the tree traversal stuff for you. 

**You can tell it to:**

"Find all the links"

"Find all the links of class externalLink"

"Find all the links whose urls match "foo.com"

"Find the table heading that's got bold text, then give me that text."" 
```

### 美丽的例子

```py
 In this example, we will try and find a link (a tag) in a webpage. 

Before we start, we have to import two modules. (BeutifulSoup and urllib2). 

Urlib2 is used to open the URL we want. 

We will use the soup.findAll method to search through the soup object to match fortext and html tags within the page. 
```

```py
from BeautifulSoup import BeautifulSoup
import urllib2

url = urllib2.urlopen("http://www.python.org")
content = url.read()
soup = BeautifulSoup(content)
links = soup.findAll("a")

```

##### 输出

```py
 That will print out all the elements in python.org with an "a" tag. 

(The "a" tag defines a hyperlink, which is used to link from one page to another.) 
```

### 美丽组图示例 2

```py
 To make it a bit more useful, we can specify the URL's we want to return. 
```

```py
from BeautifulSoup import BeautifulSoup
import urllib2
import re

url = urllib2.urlopen("http://www.python.org")
content = url.read()
soup = BeautifulSoup(content)
for a in soup.findAll('a',href=True):
    if re.findall('python', a['href']):
        print "Found the URL:", a['href']

```

##### 进一步阅读

```py
 I recommend that you head over to [http://www.crummy.com](http://www.crummy.com/software/BeautifulSoup/ "Crummy") to read more about what you can do with this awesome module. 
```