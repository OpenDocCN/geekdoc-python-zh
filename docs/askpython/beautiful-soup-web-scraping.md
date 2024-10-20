# 蟒蛇美丽的汤，方便网页刮

> 原文：<https://www.askpython.com/python/beautiful-soup-web-scraping>

读者朋友们，你们好！在本文中，我们将详细关注用于 Web 抓取的 **Python Beautiful Soup 模块**。

所以，让我们开始吧！🙂

***也可阅读:[如何使用 Python Scrapy 刮出 Google 搜索结果](https://www.askpython.com/python-modules/scrape-google-search-results-python-scrapy)***

* * *

## 使用漂亮的汤进行网络刮擦–清晰概述

如今，随着数据科学和机器学习在 IT 行业占据主导地位，数据变得越来越重要。

当我们想到一个特定的领域或主题时，有许多方法可以获取数据并对其进行分析。当谈到获取数据进行分析时，我们从各种网站收集数据进行分析，并从中调查出可能性。

类似地，这些概念产生了网络抓取的概念。

通过网络抓取，我们可以在网页上冲浪和搜索数据，从网页上收集必要的数据，然后轻松地以定制的格式保存。这就是我们称之为从网上抓取数据的原因。

了解了抓取之后，现在让我们继续用 Python 把 Beautiful Soup 作为 Web 抓取的一个模块。

* * *

## 用于网页抓取的 Python 美汤模块

网络抓取的概念并不像听起来那么简单。

首先，当我们希望从网站上抓取数据时，我们需要编写一个脚本，向主服务器请求数据。

接下来，通过定制脚本，我们可以将数据从网页下载到我们的工作站上。

最后，我们可以根据 HTML 标签定制我们希望收集的信息，这样就可以从网站上下载特定的信息。

Python 为我们提供了漂亮的 Soup 模块，它由各种函数组成，可以轻松地从网页中抓取数据。有了漂亮的 Soup 模块，我们可以轻松抓取和抓取 HTML、XML、网页、文档等。

* * *

## 用美汤刮谷歌搜索结果

首先，当在服务器上搜索单词 **science** 时，我们将使用 Beautiful Soup 模块来抓取网页的结果。

最初，我们需要在 python 环境中加载 BeautifulSoup 模块。

```py
from bs4 import BeautifulSoup
import requests

```

现在，我们将提供需要搜索的网页的 URL。此外，我们将单词 **science** 附加到 URL，以便获得与数据科学相关的帖子的网络链接。

此外，我们设置了**用户代理头**,让服务器识别我们希望下载抓取数据的系统和浏览器。

```py
A = ("Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
       "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
       )

```

现在，我们需要向 URL 添加一个 GET 请求，以便从搜索结果中下载 HTML 内容。

```py
requests.get(url, header)

```

此外，我们从下载的 HTML 内容中定制并获取所有的 **Header 3** 值。

**举例**:

```py
import requests
from bs4 import BeautifulSoup
import random

text = 'science'
url = 'https://google.com/search?q=' + text
A1 = ("Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
       "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
       )

Agent1 = A1[random.randrange(len(A1))]

headers = {'user-agent': Agent1}
requ = requests.get(url, headers=headers)

soup_obj = BeautifulSoup(requ.text, 'lxml')
for x in soup_obj.find_all('h3'):
    print(x.text)
    print('#######')

```

**输出**:

```py
Science
#######
American Association for the Advancement of Science (Nonprofit organization)
#######
Science (Peer-reviewed journal)
#######
Science | AAAS
#######
Science
#######
Science - Wikipedia
#######
ScienceDirect.com | Science, health and medical journals
#######
science | Definition, Disciplines, & Facts | Britannica
#######
Science News | The latest news from all areas of science
#######
Science - Home | Facebook
#######
Science Magazine - YouTube
#######
Department Of Science & Technology 
#######

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂