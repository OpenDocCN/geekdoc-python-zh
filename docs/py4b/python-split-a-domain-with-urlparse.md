# 使用 urlparse 分割域

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-split-a-domain-with-urlparse>

## 分割一个域

```py
 This is a simple script to split the domain name from a URL. It does that by usingPythons urlparse module. 
```

```py
import urlparse
url = "http://python.org"
domain = urlparse.urlsplit(url)[1].split(':')[0]
print "The domain name of the url is: ", domain

```