# 从 Apache 日志中获取热门页面

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/getting-most-popular-pages-your-apache-logfile>

Apache 日志文件可能非常大，很难阅读。这里有一种从 Apache 日志文件中获取最常访问的页面(或文件)列表的方法。

在这个例子中，我们只想知道 GET 请求的 URL。我们将使用 Python 集合中的奇妙计数器

```py
 import collections

logfile = open("yourlogfile.log", "r")

clean_log=[]

for line in logfile:
    try:
        # copy the URLS to an empty list.
        # We get the part between GET and HTTP
        clean_log.append(line[line.index("GET")+4:line.index("HTTP")])
    except:
        pass

counter = collections.Counter(clean_log)

# get the Top 50 most popular URLs
for count in counter.most_common(50):
    print(str(count[1]) + "	" + str(count[0]))

logfile.close() 
```