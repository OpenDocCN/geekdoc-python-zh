# 获取 IP 地址的地理位置

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/script-get-the-geo-location-of-an-ip-address>

又到了脚本的时间了，这个脚本将根据用户的输入地理定位一个 IP 地址。对于这个脚本，我们将使用一组 Python 模块来完成这个任务。首先，我们检查用户是否输入了足够的参数，如果没有，他们的“用法”变量将被打印出来，告诉用户如何使用它。

我们使用 [geody](http://www.geody.com/ "geody") web 服务来查找 IP 的地理位置。

```py
import re
import sys
import urllib2
import BeautifulSoup

usage = "Run the script: ./geolocate.py IPAddress"

if len(sys.argv)!=2:
    print(usage)
    sys.exit(0)

if len(sys.argv) > 1:
    ipaddr = sys.argv[1]

geody = "http://www.geody.com/geoip.php?ip=" + ipaddr
html_page = urllib2.urlopen(geody).read()
soup = BeautifulSoup.BeautifulSoup(html_page)

# Filter paragraph containing geolocation info.
paragraph = soup('p')[3]

# Remove html tags using regex.
geo_txt = re.sub(r'<.*?>', '', str(paragraph))
print geo_txt[32:].strip() 
```

这个脚本是从 snipplr.com 的这个[帖子](http://snipplr.com/view/55465/geoipy-a-simple-ip-geolocation-python-script/ "geoip")上复制的