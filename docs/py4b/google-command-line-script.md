# Google 命令行脚本

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/google-command-line-script>

## 概观

今天的帖子将展示如何用 Python(2.7 . x 版)制作 Google 命令行脚本

注意:从 2010 年 11 月 1 日起，谷歌网络搜索 API 已被正式否决。它将按照我们的弃用政策继续工作，但您每天可以提出的请求数量将受到限制。因此，我们鼓励您转向新的自定义搜索 API。”" "

为了向 Web 搜索 API 发出请求，我们必须导入我们需要的模块。

```py
urllib2
Loads the URL response

urllib
To make use of urlencode

json
Google returns JSON

```

接下来，我们指定我们请求的 URL:[http://ajax.googleapis.com/ajax/services/search/web?v=1.0&](https://ajax.googleapis.com/ajax/services/search/web?v=1.0& "ajax.googleapis.com")

为了使它更具交互性，我们将要求用户输入并将结果保存到一个名为“query”的变量中。

```py
query = raw_input("What do you want to search for ? >> ")

```

通过加载 URL 响应来创建响应对象，包括我们上面要求的查询。

```py
response = urllib2.urlopen (url + query ).read()

```

#处理 JSON 字符串。data = json.loads(响应)

从这一点上，我们可以玩弄的结果

##### GoogleSearch.py

让我们看看完整的剧本

```py
import urllib2
import urllib
import json

url = "http://ajax.googleapis.com/ajax/services/search/web?v=1.0&"

query = raw_input("What do you want to search for ? >> ")

query = urllib.urlencode( {'q' : query } )

response = urllib2.urlopen (url + query ).read()

data = json.loads ( response )

results = data [ 'responseData' ] [ 'results' ]

for result in results:
    title = result['title']
    url = result['url']
    print ( title + '; ' + url )

```

打开文本编辑器，复制并粘贴上面的代码。

将文件另存为 GoogleSearch.py 并退出编辑器。

运行脚本:$ python searchGoogle.py

```py
What do you want to search for ? >> python for beginners
BeginnersGuide - Python Wiki; http://wiki.python.org/moin/BeginnersGuide
Python For Beginners; http://www.python.org/about/gettingstarted/
Python For Beginners; https://www.pythonforbeginners.com/

```