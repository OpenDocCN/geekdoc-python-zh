# URL lib 2–用户代理

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-modules-urllib2-user-agent>

## 概观

This post will show how to add headers to a HTTP request. By default urllib2 identifies itself as Python-urllib/2.7 : GET / HTTP/1.1″ 200 151 “-” “Python-urllib/2.7” That can sometimes be confusing for certain sites. With the user_agent header in Python, it’s possible to alter that and specify any identity you like. The example below, use the Mozilla 5.10 as a User Agent, and that is also what will show up in the web server log file.

```py
import urllib2

req = urllib2.Request('http://192.168.1.2/')

req.add_header('User-agent', 'Mozilla 5.10')

res = urllib2.urlopen(req)

html = res.read() 
```

This is what will show up in the log file.

```py
 "GET / HTTP/1.1" 200 151 "-" "Mozilla 5.10" 
```