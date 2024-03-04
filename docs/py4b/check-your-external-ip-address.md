# 检查您的外部 IP 地址

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/check-your-external-ip-address>

## 概观

这是一个简单的 Python 脚本，用于检查您拥有哪个外部 IP 地址。首先，我们导入 urllib 和 re 模块。

## 检查您的 IP 地址

我们将用来检查 IP 地址的 URL 是:http://checkip.dyndns.org

```py
import urllib
import re

print "we will try to open this url, in order to get IP Address"

url = "http://checkip.dyndns.org"

print url

request = urllib.urlopen(url).read()

theIP = re.findall(r"d{1,3}.d{1,3}.d{1,3}.d{1,3}", request)

print "your IP Address is: ",  theIP 
```

快乐脚本