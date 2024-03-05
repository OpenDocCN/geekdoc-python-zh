# 用 Scrapy 和 MongoDB 进行网页抓取和抓取

> 原文：<https://realpython.com/web-scraping-and-crawling-with-scrapy-and-mongodb/>

上次我们实现了一个基本的网络抓取器，它从 StackOverflow 下载最新的问题，并将结果存储在 [MongoDB](https://realpython.com/introduction-to-mongodb-and-python/) 中。**在这篇文章中，我们将扩展我们的抓取器，使它能够抓取每个页面底部的分页链接，并从每个页面抓取问题(问题标题和 URL)。**

**免费奖励:** ，向您展示如何从 Python 访问 MongoDB。

**更新:**

1.  09/06/2015 -更新到最新版本的 [Scrapy](http://doc.scrapy.org/en/1.0/) (v1.0.3)和 [PyMongo](http://api.mongodb.org/python/3.0.3/) (v3.0.3) -干杯！

> 在您开始任何刮擦工作之前，请查看网站的使用条款政策，并遵守 robots.txt 文件。此外，要遵守道德规范，不要在短时间内让大量请求涌入网站。对待你刮下的任何地方，就像是你自己的一样。

* * *

这是 Real Python 和 Gyö rgy 的合作作品，Gyö rgy 是一名 Python 爱好者和软件开发人员，目前在一家大数据公司工作，同时在寻找一份新工作。可以在 twitter 上问他问题- [@kissgyorgy](https://twitter.com/kissgyorgy) 。

## 开始使用

有两种可能的方法从我们停止的地方继续。

第一种方法是扩展我们现有的蜘蛛，用一个 xpath 表达式从`parse_item`方法的响应中提取每个下一页链接，只使用一个对同一个`parse_item`方法进行回调的`yield`对象。这样 scrapy 会自动向我们指定的链接发出新的请求。你可以在 [Scrapy 文档](http://doc.scrapy.org/en/1.0/topics/spiders.html#spiders)中找到更多关于这种方法的信息。

另一个更简单的选择是利用不同类型的蜘蛛——即`CrawlSpider` ( [链接](http://doc.scrapy.org/en/1.0/topics/spiders.html#crawlspider))。这是基本`Spider`的扩展版本，专为我们的用例而设计。

[*Remove ads*](/account/join/)

## 爬虫

我们将使用上一个教程中相同的 Scrapy 项目，所以如果需要的话，可以从 [repo](https://github.com/realpython/stack-spider/releases/tag/v1) 中获取代码。

### 创建样板文件

在“stack”目录中，首先由[从`crawl`模板生成](http://doc.scrapy.org/en/1.0/topics/commands.html#std:command-genspider)蜘蛛样板文件:

```py
$ scrapy genspider stack_crawler stackoverflow.com -t crawl
Created spider 'stack_crawler' using template 'crawl' in module:
 stack.spiders.stack_crawler
```

Scrapy 项目现在应该是这样的:

```py
├── scrapy.cfg
└── stack
    ├── __init__.py
    ├── items.py
    ├── pipelines.py
    ├── settings.py
    └── spiders
        ├── __init__.py
        ├── stack_crawler.py
        └── stack_spider.py
```

并且 *stack_crawler.py* 文件应该是这样的:

```py
# -*- coding: utf-8 -*-
import scrapy
from scrapy.contrib.linkextractors import LinkExtractor
from scrapy.contrib.spiders import CrawlSpider, Rule

from stack.items import StackItem

class StackCrawlerSpider(CrawlSpider):
    name = 'stack_crawler'
    allowed_domains = ['stackoverflow.com']
    start_urls = ['http://www.stackoverflow.com/']

    rules = (
        Rule(LinkExtractor(allow=r'Items/'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        i = StackItem()
        #i['domain_id'] = response.xpath('//input[@id="sid"]/@value').extract()
        #i['name'] = response.xpath('//div[@id="name"]').extract()
        #i['description'] = response.xpath('//div[@id="description"]').extract()
        return i
```

我们只需要对这个样板文件做一些更新…

### 更新`start_urls`列表

首先，将第一页的问题添加到`start_urls`列表中:

```py
start_urls = [
    'http://stackoverflow.com/questions?pagesize=50&sort=newest'
]
```

### 更新`rules`列表

接下来，我们需要通过给`rules`属性添加一个[正则表达式](https://realpython.com/regex-python/)来告诉蜘蛛它可以在哪里找到下一个页面链接:

```py
rules = [
    Rule(LinkExtractor(allow=r'questions\?page=[0-9]&sort=newest'),
         callback='parse_item', follow=True)
]
```

Scrapy 现在将根据这些链接自动请求新页面，并将响应传递给`parse_item`方法以提取问题和标题。

> 如果你密切关注，这个正则表达式将抓取限制在前 9 页，因为对于这个演示，我们不想抓取所有的 176，234 页！

### 更新`parse_item`方法

现在我们只需要写如何用 xpath 解析页面，我们在上一个教程中已经完成了——所以只需要复制它:

```py
def parse_item(self, response):
    questions = response.xpath('//div[@class="summary"]/h3')

    for question in questions:
        item = StackItem()
        item['url'] = question.xpath(
            'a[@class="question-hyperlink"]/@href').extract()[0]
        item['title'] = question.xpath(
            'a[@class="question-hyperlink"]/text()').extract()[0]
        yield item
```

对蜘蛛来说就是这样，但是不要现在就开始。

[*Remove ads*](/account/join/)

### 添加下载延迟

我们需要通过在 *settings.py* 中设置[下载延迟](http://doc.scrapy.org/en/1.0/topics/settings.html#std:setting-DOWNLOAD_DELAY)来善待 StackOverflow(以及任何网站，就此而言):

```py
DOWNLOAD_DELAY = 5
```

这告诉 Scrapy 在每次新请求之间至少等待 5 秒钟。你本质上是在限制自己的速度。如果你不这样做，StackOverflow 会限制你的流量；如果你继续抓取网站而不设置速率限制，你的 IP 地址可能会被禁止。所以，善待你刮到的任何地方，就像是你自己的一样。

现在只剩下一件事要做——存储数据。

## MongoDB

上次我们只下载了 50 个问题，但由于我们这次获取了更多的数据，我们希望避免向数据库中添加重复的问题。我们可以通过使用 MongoDB [upsert](http://docs.mongodb.org/v3.0/reference/method/db.collection.update/#upsert-option) 来实现，这意味着如果问题标题已经在数据库中，我们就更新它，否则就插入。

修改我们之前定义的`MongoDBPipeline`:

```py
class MongoDBPipeline(object):

    def __init__(self):
        connection = pymongo.MongoClient(
            settings['MONGODB_SERVER'],
            settings['MONGODB_PORT']
        )
        db = connection[settings['MONGODB_DB']]
        self.collection = db[settings['MONGODB_COLLECTION']]

    def process_item(self, item, spider):
        for data in item:
            if not data:
                raise DropItem("Missing data!")
        self.collection.update({'url': item['url']}, dict(item), upsert=True)
        log.msg("Question added to MongoDB database!",
                level=log.DEBUG, spider=spider)
        return item
```

> 为了简单起见，我们没有优化查询，也没有处理索引，因为这不是一个生产环境。

## 测试

启动蜘蛛！

```py
$ scrapy crawl stack_crawler
```

现在坐好，看着你的数据库充满数据！

```py
$ mongo
MongoDB shell version: 3.0.4
> use stackoverflow
switched to db stackoverflow
> db.questions.count()
447
>
```

## 结论

您可以从 [Github 资源库](https://github.com/realpython/stack-spider/releases/tag/v2)下载完整的源代码。带着问题在下面评论。干杯！

**免费奖励:** ，向您展示如何从 Python 访问 MongoDB。

> 寻找更多的网页抓取？一定要去看看[真正的 Python 课程](https://realpython.com/courses/)。想雇一个专业的网页抓取者吗？查看[围棋](http://www.goscrape.com/)。**