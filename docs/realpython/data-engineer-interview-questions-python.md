# 用 Python 编写数据工程师面试问题

> 原文：<https://realpython.com/data-engineer-interview-questions-python/>

参加面试可能是一个既费时又累人的过程，技术性的面试可能压力更大！本教程旨在为你在[数据工程师](https://realpython.com/python-data-engineer/)面试中会遇到的一些常见问题做好准备。您将学习如何回答关于数据库、Python 和 [SQL](https://realpython.com/python-sql-libraries/) 的问题。

**本教程结束时，你将能够:**

*   了解常见的数据工程师面试问题
*   区分关系数据库和非关系数据库
*   使用 Python 建立[数据库](https://realpython.com/tutorials/databases/)
*   使用 Python 查询数据

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 成为数据工程师

数据工程的角色可以是广泛而多样的。你需要掌握多种技术和概念的应用知识。数据工程师思维灵活。因此，他们可以精通多个主题，如数据库、软件开发、 [DevOps](https://realpython.com/learning-paths/python-devops/) 和[大数据](https://realpython.com/pyspark-intro/)。

[*Remove ads*](/account/join/)

### 数据工程师是做什么的？

鉴于其不同的技能组合，数据工程角色可以跨越许多不同的工作描述。一个数据工程师可以负责[数据库](https://realpython.com/tutorials/databases/)设计、模式设计，并创建多个数据库解决方案。这项工作还可能涉及数据库管理员。

作为一名**数据工程师**，你可以充当数据库和[数据科学](https://realpython.com/tutorials/data-science/)团队之间的桥梁。在这种情况下，您还将负责数据清理和准备。如果涉及到大数据，那么您的工作就是为这些数据提供高效的解决方案。这项工作可能与 DevOps 角色重叠。

您还需要为报告和分析进行有效的数据查询。您可能需要与多个数据库交互或编写存储过程。对于像高流量网站或服务这样的许多解决方案，可能存在不止一个数据库。在这些情况下，数据工程师负责建立数据库，维护它们，并在它们之间传输数据。

### Python 如何帮助数据工程师？

Python 被认为是编程语言的瑞士军刀。它在数据科学、后端系统和服务器端脚本中特别有用。这是因为 Python 具有强大的类型、简单的语法和丰富的第三方库可供使用。 [Pandas](https://realpython.com/pandas-python-explore-dataset/) 、SciPy、 [Tensorflow](https://realpython.com/numpy-tensorflow-performance/) 、 [SQLAlchemy](https://realpython.com/flask-connexion-rest-api/) 和 [NumPy](https://realpython.com/numpy-array-programming/) 是一些在不同行业的生产中使用最广泛的库。

最重要的是，Python 减少了开发时间，这意味着公司的开支更少。对于一个数据工程师来说，大多数代码的执行是受数据库限制的，而不是受 CPU 限制的。正因为如此，利用 Python 的简单性是有意义的，即使与 C#和 Java 等编译语言相比，它的性能会降低。

## 回答数据工程师面试问题

现在你知道你的角色可能包括什么，是时候学习如何回答一些数据工程师面试问题了！虽然有很多内容需要介绍，但是在整个教程中，您将会看到一些实用的 Python 示例来指导您。

## 关于关系数据库的问题

数据库是系统中最重要的组件之一。没有他们，就没有国家和历史。虽然您可能没有将数据库设计视为优先事项，但要知道它会对页面加载速度产生重大影响。在过去的几年中，一些大公司引入了一些新的工具和技术:

*   NoSQL
*   缓存数据库
*   图形数据库
*   SQL 数据库中的 NoSQL 支持

发明这些和其他技术是为了尝试提高数据库处理请求的速度。你可能需要在你的数据工程师面试中谈论这些概念，所以让我们复习一些问题！

### Q1:关系数据库与非关系数据库

**关系数据库**是以表格形式存储数据的数据库。每个表都有一个**模式**，这是一个记录需要拥有的列和类型。每个模式必须至少有一个唯一标识该记录的主键。换句话说，数据库中没有重复的行。此外，每个表可以使用外键与其他表相关联。

关系数据库的一个重要方面是模式的改变必须应用于所有记录。这有时会在迁移过程中导致中断和大麻烦。非关系数据库以不同的方式处理事情。它们本质上是无模式的，这意味着记录可以用不同的模式和不同的嵌套结构保存。记录仍然可以有主键，但是模式的改变是在逐个条目的基础上进行的。

您需要根据正在执行的功能类型执行速度比较测试。您可以选择`INSERT`、`UPDATE`、`DELETE`或其他功能。模式设计、索引、聚合的数量和记录的数量也会影响这个分析，所以您需要进行彻底的测试。稍后您将了解更多关于如何做到这一点的信息。

数据库在**可扩展性**上也有所不同。非关系数据库的分布可能不那么令人头疼。这是因为相关记录的集合可以很容易地存储在特定的节点上。另一方面，关系数据库需要更多的思考，通常使用主从系统。

### 一个 SQLite 例子

既然您已经回答了什么是关系数据库，那么是时候深入研究一些 Python 了！SQLite 是一个方便的数据库，您可以在本地机器上使用。数据库是一个单一的文件，这使得它非常适合于原型设计。首先，导入所需的 Python 库并创建一个新的数据库:

```py
import sqlite3

db = sqlite3.connect(':memory:')  # Using an in-memory database
cur = db.cursor()
```

现在，您已经连接到一个内存数据库，并准备好了光标对象。

接下来，您将创建以下三个表:

1.  **客户:**该表将包含一个主键以及客户的名和姓。
2.  **Items:** 该表将包含主键、项目名称和项目价格。
3.  **购买的商品**:该表将包含订单号、日期和价格。它还将连接到 Items 和 Customer 表中的主键。

现在，您已经对表格的外观有了一个概念，可以继续创建它们了:

```py
cur.execute('''CREATE TABLE IF NOT EXISTS Customer (
 id integer PRIMARY KEY,
 firstname varchar(255),
 lastname varchar(255) )''')
cur.execute('''CREATE TABLE IF NOT EXISTS Item (
 id integer PRIMARY KEY,
 title varchar(255),
 price decimal )''')
cur.execute('''CREATE TABLE IF NOT EXISTS BoughtItem (
 ordernumber integer PRIMARY KEY,
 customerid integer,
 itemid integer,
 price decimal,
 CONSTRAINT customerid
 FOREIGN KEY (customerid) REFERENCES Customer(id),
 CONSTRAINT itemid
 FOREIGN KEY (itemid) REFERENCES Item(id) )''')
```

您已经向`cur.execute()`传递了一个查询来创建您的三个表。

最后一步是用数据填充表:

```py
cur.execute('''INSERT INTO Customer(firstname, lastname)
 VALUES ('Bob', 'Adams'),
 ('Amy', 'Smith'),
 ('Rob', 'Bennet');''')
cur.execute('''INSERT INTO Item(title, price)
 VALUES ('USB', 10.2),
 ('Mouse', 12.23),
 ('Monitor', 199.99);''')
cur.execute('''INSERT INTO BoughtItem(customerid, itemid, price)
 VALUES (1, 1, 10.2),
 (1, 2, 12.23),
 (1, 3, 199.99),
 (2, 3, 180.00),
 (3, 2, 11.23);''') # Discounted price
```

现在每个表中都有一些记录，您可以使用这些数据来回答更多的数据工程师面试问题。

[*Remove ads*](/account/join/)

### Q2: SQL 聚合函数

**聚合函数**是对结果集执行数学运算的函数。一些例子包括`AVG`、`COUNT`、`MIN`、`MAX`和`SUM`。通常，您需要`GROUP BY`和`HAVING`子句来补充这些聚合。一个有用的聚合函数是`AVG`，您可以使用它来计算给定结果集的平均值:

>>>

```py
>>> cur.execute('''SELECT itemid, AVG(price) FROM BoughtItem GROUP BY itemid''')
>>> print(cur.fetchall())
[(1, 10.2), (2, 11.73), (3, 189.995)]
```

这里，您已经检索了数据库中购买的每件商品的平均价格。您可以看到，`itemid`为`1`的商品平均价格为 10.20 美元。

为了使上面的输出更容易理解，您可以显示项目名称来代替`itemid`:

>>>

```py
>>> cur.execute('''SELECT item.title, AVG(boughtitem.price) FROM BoughtItem as boughtitem
...             INNER JOIN Item as item on (item.id = boughtitem.itemid)
...             GROUP BY boughtitem.itemid''')
...
>>> print(cur.fetchall())
[('USB', 10.2), ('Mouse', 11.73), ('Monitor', 189.995)]
```

现在，您更容易看到平均价格为 10.20 美元的商品是`USB`。

另一个有用的聚合是`SUM`。您可以使用此功能显示每位客户的总消费金额:

>>>

```py
>>> cur.execute('''SELECT customer.firstname, SUM(boughtitem.price) FROM BoughtItem as boughtitem
...             INNER JOIN Customer as customer on (customer.id = boughtitem.customerid)
...             GROUP BY customer.firstname''')
...
>>> print(cur.fetchall())
[('Amy', 180), ('Bob', 222.42000000000002), ('Rob', 11.23)]
```

平均而言，名为 Amy 的客户花费了大约 180 美元，而 Rob 仅花费了 11.23 美元！

如果你的面试官喜欢数据库，那么你可能想温习一下嵌套查询、连接类型和关系数据库执行查询的步骤。

### Q3:加速 SQL 查询

速度取决于各种因素，但主要受以下各项的影响:

*   连接
*   聚集
*   遍历
*   记录

连接数量越多，复杂性就越高，表中的遍历次数也就越多。对涉及几个表的几千条记录执行多重连接是非常昂贵的，因为数据库还需要缓存中间结果！此时，您可能会开始考虑如何增加内存大小。

速度还受到数据库中是否存在索引的影响。索引非常重要，它允许您快速搜索整个表，并为查询中指定的某个列找到匹配项。

索引以更长的插入时间和一些存储为代价对记录进行排序。可以组合多个列来创建一个索引。例如，列`date`和`price`可以合并，因为您的查询依赖于这两个条件。

### Q4:调试 SQL 查询

大多数数据库都包含一个`EXPLAIN QUERY PLAN`，它描述了数据库执行查询的步骤。对于 SQLite，您可以通过在`SELECT`语句前添加`EXPLAIN QUERY PLAN`来启用该功能:

>>>

```py
>>> cur.execute('''EXPLAIN QUERY PLAN SELECT customer.firstname, item.title, 
...                item.price, boughtitem.price FROM BoughtItem as boughtitem
...                INNER JOIN Customer as customer on (customer.id = boughtitem.customerid)
...                INNER JOIN Item as item on (item.id = boughtitem.itemid)''')
...
>>> print(cur.fetchall())
[(4, 0, 0, 'SCAN TABLE BoughtItem AS boughtitem'), 
(6, 0, 0, 'SEARCH TABLE Customer AS customer USING INTEGER PRIMARY KEY (rowid=?)'), 
(9, 0, 0, 'SEARCH TABLE Item AS item USING INTEGER PRIMARY KEY (rowid=?)')]
```

该查询尝试列出所有购买商品的名字、商品标题、原价和购买价格。

下面是查询计划本身的样子:

```py
SCAN  TABLE  BoughtItem  AS  boughtitem SEARCH  TABLE  Customer  AS  customer  USING  INTEGER  PRIMARY  KEY  (rowid=?) SEARCH  TABLE  Item  AS  item  USING  INTEGER  PRIMARY  KEY  (rowid=?)
```

请注意，Python 代码中的 fetch 语句只返回解释，而不返回结果。那是因为`EXPLAIN QUERY PLAN`并不打算在生产中使用。

[*Remove ads*](/account/join/)

## 关于非关系数据库的问题

在上一节中，您展示了关系数据库和非关系数据库之间的区别，并在 Python 中使用了 SQLite。现在你要专注于 NoSQL。你的目标是突出它的优势、差异和用例。

### 一个 MongoDB 例子

您将使用与之前相同的数据，但是这次您的数据库将是 [MongoDB](https://realpython.com/introduction-to-mongodb-and-python/) 。这个 NoSQL 数据库是基于文档的，伸缩性很好。首先，您需要安装所需的 Python 库:

```py
$ pip install pymongo
```

您可能还想安装 [MongoDB Compass 社区](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-windows/#install-mdb-edition)。它包括一个非常适合可视化数据库的本地 [IDE](https://realpython.com/python-ides-code-editors-guide/) 。有了它，您可以看到创建的记录，创建触发器，并充当数据库的可视化管理员。

**注意:**要运行本节中的代码，您需要一个正在运行的数据库服务器。要了解更多关于如何设置它的信息，请查看[MongoDB 和 Python 简介](https://realpython.com/introduction-to-mongodb-and-python/)。

以下是创建数据库并插入一些数据的方法:

```py
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")

# Note: This database is not created until it is populated by some data
db = client["example_database"]

customers = db["customers"]
items = db["items"]

customers_data = [{ "firstname": "Bob", "lastname": "Adams" },
                  { "firstname": "Amy", "lastname": "Smith" },
                  { "firstname": "Rob", "lastname": "Bennet" },]
items_data = [{ "title": "USB", "price": 10.2 },
              { "title": "Mouse", "price": 12.23 },
              { "title": "Monitor", "price": 199.99 },]

customers.insert_many(customers_data)
items.insert_many(items_data)
```

您可能已经注意到，MongoDB 将数据记录存储在**集合**中，这相当于 Python 中的字典列表。在实践中，MongoDB 存储 [BSON 文档](https://docs.mongodb.com/manual/core/document/#bson-document-format)。

### Q5:使用 MongoDB 查询数据

让我们首先尝试复制`BoughtItem`表，就像您在 SQL 中所做的那样。为此，您必须向客户追加一个新字段。MongoDB 的[文档](https://docs.mongodb.com/manual/reference/method/db.collection.updateMany/#db.collection.updateMany)规定，可以使用关键字操作符**集合**来更新记录，而不必编写所有现有字段:

```py
# Just add "boughtitems" to the customer where the firstname is Bob
bob = customers.update_many(
        {"firstname": "Bob"},
        {
            "$set": {
                "boughtitems": [
                    {
                        "title": "USB",
                        "price": 10.2,
                        "currency": "EUR",
                        "notes": "Customer wants it delivered via FedEx",
                        "original_item_id": 1
                    }
                ]
            },
        }
    )
```

请注意，您是如何在没有事先明确定义模式的情况下向`customer`添加额外的字段的。俏皮！

事实上，您可以通过稍微修改模式来更新另一个客户:

```py
amy = customers.update_many(
        {"firstname": "Amy"},
        {
            "$set": {
                "boughtitems":[
                    {
                        "title": "Monitor",
                        "price": 199.99,
                        "original_item_id": 3,
                        "discounted": False
                    }
                ]
            } ,
        }
    )
print(type(amy))  # pymongo.results.UpdateResult
```

与 SQL 类似，基于文档的数据库也允许执行查询和聚合。但是，功能可能在语法和底层执行上有所不同。事实上，您可能已经注意到 MongoDB 保留了`$`字符来指定记录上的一些命令或聚合，比如`$group`。你可以在[官方文件](https://docs.mongodb.com/manual/reference/operator/query/)中了解更多关于这种行为的信息。

您可以像在 SQL 中一样执行查询。首先，您可以创建一个索引:

>>>

```py
>>> customers.create_index([("name", pymongo.DESCENDING)])
```

这是可选的，但是它加快了需要名称查找的查询。

然后，您可以检索按升序排序的客户名称:

>>>

```py
>>> items = customers.find().sort("name", pymongo.ASCENDING)
```

您还可以遍历并打印购买的商品:

>>>

```py
>>> for item in items:
...     print(item.get('boughtitems'))    
...
None
[{'title': 'Monitor', 'price': 199.99, 'original_item_id': 3, 'discounted': False}]
[{'title': 'USB', 'price': 10.2, 'currency': 'EUR', 'notes': 'Customer wants it delivered via FedEx', 'original_item_id': 1}]
```

您甚至可以检索数据库中唯一名称的列表:

>>>

```py
>>> customers.distinct("firstname")
['Bob', 'Amy', 'Rob']
```

现在您已经知道了数据库中客户的姓名，您可以创建一个查询来检索有关他们的信息:

>>>

```py
>>> for i in customers.find({"$or": [{'firstname':'Bob'}, {'firstname':'Amy'}]}, 
...                                  {'firstname':1, 'boughtitems':1, '_id':0}):
...     print(i)
...
{'firstname': 'Bob', 'boughtitems': [{'title': 'USB', 'price': 10.2, 'currency': 'EUR', 'notes': 'Customer wants it delivered via FedEx', 'original_item_id': 1}]}
{'firstname': 'Amy', 'boughtitems': [{'title': 'Monitor', 'price': 199.99, 'original_item_id': 3, 'discounted': False}]}
```

下面是等效的 SQL 查询:

```py
SELECT  firstname,  boughtitems  FROM  customers  WHERE  firstname  LIKE  ('Bob',  'Amy')
```

请注意，尽管语法可能略有不同，但在底层执行查询的方式上有很大的不同。这是意料之中的，因为 SQL 和 NoSQL 数据库之间的查询结构和用例不同。

[*Remove ads*](/account/join/)

### Q6: NoSQL vs SQL

如果您有一个不断变化的模式，如金融监管信息，那么 NoSQL 可以修改记录并嵌套相关信息。想象一下，如果您有八个嵌套顺序，您将不得不在 SQL 中进行多少次连接！然而，这种情况比你想象的更常见。

现在，如果您想要运行报告、提取财务数据的信息并推断结论，该怎么办呢？在这种情况下，您需要运行复杂的查询，而 SQL 在这方面往往更快。

**注意:** SQL 数据库，尤其是 [PostgreSQL](https://www.postgresql.org/) ，也发布了一个特性，允许将可查询的 [JSON](https://realpython.com/courses/working-json-data-python/) 数据作为记录的一部分插入。虽然这可以结合两个世界的优点，但速度可能是个问题。

从 NoSQL 数据库中查询非结构化数据比从 PostgreSQL 中的 JSON 类型列中查询 JSON 字段要快。你可以做一个[速度对比](https://www.arangodb.com/2018/02/nosql-performance-benchmark-2018-mongodb-postgresql-orientdb-neo4j-arangodb/)测试来得到一个明确的答案。

尽管如此，这个特性可能会减少对额外数据库的需求。有时，经过酸洗或序列化的对象以二进制类型的形式存储在记录中，然后在读取时反序列化。

然而，速度并不是唯一的衡量标准。您还需要考虑事务、原子性、持久性和可伸缩性。**交易**在金融应用中很重要，这样的特性优先。

由于数据库的范围很广，每个数据库都有自己的特性，所以数据工程师的工作就是做出明智的决定，在每个应用程序中使用哪个数据库。有关更多信息，您可以阅读与数据库事务相关的 [ACID](https://en.wikipedia.org/wiki/ACID) 属性。

在你的数据工程师面试中，你可能还会被问到你所知道的其他数据库。许多公司还使用其他几个相关的数据库:

*   [**弹性搜索**](https://marutitech.com/elasticsearch-big-data-analytics/) 在文本搜索中效率高。它利用其基于文档的数据库来创建一个强大的搜索工具。
*   [**Newt DB**](http://www.newtdb.org/en/latest/how-it-works.html) 结合了 [ZODB](http://www.zodb.org/en/latest/) 和 PostgreSQL JSONB 特性来创建一个 Python 友好的 NoSQL 数据库。
*   [**InfluxDB**](https://www.influxdata.com/) 用于时序应用中存储事件。

这个列表还可以继续下去，但这说明了各种各样的可用数据库是如何迎合他们的利基行业的。

## 关于缓存数据库的问题

**缓存数据库**保存频繁访问的数据。它们与主要的 SQL 和 NoSQL 数据库共存。他们的目标是减轻负载，更快地满足请求。

### a 再举一个例子

您已经讨论了 SQL 和 NoSQL 数据库的长期存储解决方案，但是更快、更即时的存储呢？数据工程师如何改变从数据库中检索数据的速度？

典型的 web 应用程序经常检索常用数据，如用户的个人资料或姓名。如果所有的数据都包含在一个数据库中，那么数据库服务器获得的**命中数**将会超过上限，并且是不必要的。因此，需要一种更快、更直接的存储解决方案。

虽然这降低了服务器负载，但也给数据工程师、后端团队和 DevOps 团队带来了两个难题。首先，您现在需要一些比您的主 SQL 或 NoSQL 数据库具有更快读取时间的数据库。但是，两个数据库的内容最终必须匹配。(欢迎来到数据库间**状态一致性**的问题！享受吧。)

第二个令人头疼的问题是，DevOps 现在需要为新的缓存数据库担心可伸缩性、冗余等问题。在下一节中，您将在 [Redis](https://realpython.com/python-redis/) 的帮助下深入研究这些问题。

### 问题 7:如何使用缓存数据库

你可能已经从简介中获取了足够的信息来回答这个问题！**缓存数据库**是一种快速存储解决方案，用于存储短期的结构化或非结构化数据。它可以根据您的需要进行分区和扩展，但是它的大小通常比您的主数据库小得多。因此，您的缓存数据库可以驻留在内存中，这样您就不必从磁盘中读取数据。

**注意:**如果你曾经在 Python 中使用过[字典](https://realpython.com/courses/dictionaries-python/)，那么 Redis 遵循相同的结构。这是一个键值存储，在这里你可以像一个 Python `dict`一样`SET`和`GET`数据。

当请求进来时，首先检查缓存数据库，然后检查主数据库。这样，您可以防止任何不必要的重复请求到达主数据库的服务器。由于高速缓存数据库的读取时间较短，因此您还可以从性能提升中获益！

您可以使用 [pip](https://realpython.com/what-is-pip/) 来安装所需的库:

```py
$ pip install redis
```

现在，考虑一个从用户 ID 获取用户名的请求:

```py
import redis
from datetime import timedelta

# In a real web application, configuration is obtained from settings or utils
r = redis.Redis()

# Assume this is a getter handling a request
def get_name(request, *args, **kwargs):
    id = request.get('id')
    if id in r:
        return r.get(id)  # Assume that we have an {id: name} store
    else:
        # Get data from the main DB here, assume we already did it
        name = 'Bob'
        # Set the value in the cache database, with an expiration time
        r.setex(id, timedelta(minutes=60), value=name)
        return name
```

这段代码使用`id`键检查这个名字是否在 Redis 中。如果没有，那么该名称会设置一个过期时间，因为缓存是短期的，所以使用该时间。

现在，如果你的面试官问你这个代码有什么问题呢？你的回答应该是没有[异常处理](https://realpython.com/python-exceptions/)！数据库可能会有很多问题，比如掉线，所以尝试捕捉这些异常总是一个好主意。

[*Remove ads*](/account/join/)

## 关于设计模式和 ETL 概念的问题

在大型应用程序中，您通常会使用多种类型的数据库。事实上，可以在一个应用程序中使用 PostgreSQL、MongoDB 和 Redis！一个具有挑战性的问题是处理数据库之间的状态变化，这使开发人员面临一致性问题。考虑以下场景:

1.  数据库#1 中的值被更新。
2.  数据库#2 中相同值保持不变(不更新)。
3.  在数据库#2 上运行查询。

现在，你得到了一个不一致和过时的结果！从第二个数据库返回的结果不会反映第一个数据库中的更新值。任何两个数据库都可能发生这种情况，但是当主数据库是 NoSQL 数据库，并且信息被转换为 SQL 以供查询时，这种情况尤其常见。

数据库可能有后台工作人员来处理这类问题。这些工人**从一个数据库中提取**数据，**以某种方式转换**，然后**将**数据加载到目标数据库中。当您从 NoSQL 数据库转换到 SQL 数据库时，提取、转换、加载(ETL)过程需要以下步骤:

1.  **Extract:** 每当一个记录被创建、更新等等时，都会有一个 MongoDB 触发器。一个回调函数在一个单独的线程上被异步调用。
2.  **Transform:** 部分记录被提取、规范化，并放入正确的数据结构(或行)中，以便插入 SQL。
3.  **Load:**SQL 数据库批量更新，或者作为大容量写入的单个记录更新。

这种工作流在金融、游戏和报告应用程序中很常见。在这些情况下，不断变化的模式需要 NoSQL 数据库，但是报告、分析和聚合需要 SQL 数据库。

### 问题 8: ETL 挑战

ETL 中有几个具有挑战性的概念，包括:

*   大数据
*   状态问题
*   异步工人
*   类型匹配

不胜枚举！然而，由于 ETL 过程中的步骤是定义良好且符合逻辑的，数据和后端工程师通常会更担心性能和可用性，而不是实现。

如果您的应用程序每秒向 MongoDB 写入数千条记录，那么您的 ETL 工作人员需要跟上数据的转换、加载和以请求的形式交付给用户。速度和延迟可能会成为一个问题，所以这些工作程序通常是用快速语言编写的。您可以在转换步骤中使用编译后的代码来加快速度，因为这部分通常是 CPU 受限的。

**注意:**多处理和工人分离是您可能要考虑的其他解决方案。

如果你正在处理大量 CPU 密集型函数，那么你可能想看看 [Numba](https://numba.pydata.org/) 。这个库编译函数以使它们执行起来更快。最重要的是，这很容易在 Python 中实现，尽管在这些编译后的函数中可以使用哪些函数上有一些限制。

### 问题 9:大数据中的设计模式

想象一下，亚马逊需要创建一个 [**推荐系统**](https://realpython.com/build-recommendation-engine-collaborative-filtering/) ，向用户推荐合适的商品。数据科学团队需要大量的数据！他们找到数据工程师，要求您创建一个单独的临时数据库仓库。他们将在那里清理和转换数据。

收到这样的请求，你可能会感到震惊。当您拥有万亿字节的数据时，您将需要多台机器来处理所有这些信息。数据库聚合函数可能是非常复杂的操作。如何高效地查询、聚集和利用相对较大的数据？

Apache 最初引入了 [MapReduce](https://en.wikipedia.org/wiki/MapReduce) ，它遵循了 **map，shuffle，reduce** 工作流。这个想法是将不同的数据映射到不同的机器上，也称为集群。然后，您可以对数据执行操作，按键进行分组，最后，在最后阶段聚合数据。

这种工作流程今天仍在使用，但最近它逐渐被 [Spark](https://spark.apache.org/) 所取代。然而，设计模式构成了大多数大数据工作流的基础，是一个非常有趣的概念。你可以在 [IBM Analytics](https://www.ibm.com/analytics/hadoop/mapreduce) 阅读更多关于 MapReduce 的内容。

### Q10:ETL 流程和大数据工作流的共同方面

你可能会认为这是一个相当奇怪的问题，但这只是对你的计算机科学知识，以及你的整体设计知识和经验的一个检验。

两个工作流都遵循**生产者-消费者**模式。一个工人(生产者)生产某种数据，并将其输出到管道。这种管道可以采取多种形式，包括网络消息和触发器。生产者输出数据后，消费者消费并利用数据。这些工作线程通常以异步方式工作，并在单独的进程中执行。

您可以将生产者比作 ETL 过程的提取和转换步骤。同样，在大数据中，**映射器**可以被视为生产者，而**缩减器**实际上是消费者。这种关注点的分离在应用程序的开发和架构设计中是极其重要和有效的。

[*Remove ads*](/account/join/)

## 结论

恭喜你！你已经覆盖了很多领域，回答了几个数据工程师面试问题。现在，您对数据工程师可能扮演的许多不同角色，以及您在数据库、设计和工作流方面的职责有了更多的了解。

**有了这些知识，你现在可以:**

*   将 Python 与 SQL、NoSQL 和缓存数据库结合使用
*   在 ETL 和查询应用程序中使用 Python
*   提前计划项目，牢记设计和工作流程

虽然面试问题可能是多种多样的，但你已经接触了多个主题，并学会了在计算机科学的许多不同领域跳出框框思考。现在你已经准备好进行一场精彩的面试了！******