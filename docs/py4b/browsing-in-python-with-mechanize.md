# 使用机械化在 Python 中浏览

> 原文：<https://www.pythonforbeginners.com/python-on-the-web/browsing-in-python-with-mechanize>

## 使用 Mechanize 浏览

Python 中的 mechanize 模块类似于 perl WWW:Mechanize。

它给你一个类似浏览器的对象来与网页交互。

这是一个如何在程序中使用它的例子。

```py
import mechanize
br = mechanize.Browser()
br.open("http://www.example.com/") 
```

跟随第二个链接，元素文本匹配[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)

```py
response1 = br.follow_link(text_regex=r"cheeses*shop", nr=1)
assert br.viewing_html()
print br.title()
print response1.geturl()
print response1.info()  # headers
print response1.read()  # body 
```

要从网站获取响应代码，您可以

```py
from mechanize import Browser
browser = Browser()
response = browser.open('http://www.google.com')
print response.code 
```

从网站获取所有表单

```py
import mechanize
br = mechanize.Browser()
br.open("http://www.google.com/")
for f in br.forms():
    print f 
```

我在[http://stockrt.github.com](https://stockrt.github.com/p/emulating-a-browser-in-python-with-mechanize/ "emulating")发现这个帖子，非常准确地描述了
如何使用 mechanize 在 Python 中模拟一个浏览器。

用 Python 浏览([由 Drew Stephens](http://dinomite.net/blog/2007/web-browsing-with-python/ "mechanize1") 编写)

```py
#!/usr/bin/python
import re
from mechanize import Browser
br = Browser() 
```

忽略 robots.txt

```py
br.set_handle_robots( False ) 
```

谷歌需要一个不是机器人的用户代理

```py
br.addheaders = [('User-agent', 'Firefox')] 
```

检索 Google 主页，保存响应

```py
br.open( "http://google.com" ) 
```

选择搜索框并搜索“foo”

```py
br.select_form( 'f' )
br.form[ 'q' ] = 'foo' 
```

获取搜索结果

```py
br.submit() 
```

找到 foofighters.com 的链接；我们为什么要搜索？

```py
resp = None

for link in br.links():
    siteMatch = re.compile( 'www.foofighters.com' ).search( link.url )

    if siteMatch:
        resp = br.follow_link( link )
        break 
```

打印网站

```py
content = resp.get_data()
print content 
```

上面的脚本被分割开来，以便于阅读