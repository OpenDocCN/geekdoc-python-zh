# python MongoDB——完整概述

> 原文：<https://www.askpython.com/python-modules/python-mongodb>

MongoDB 是最流行的非关系数据库(也称为 NoSQL 数据库)之一。非关系或 NoSQL 数据库没有固定的表结构或模式可遵循，这使得数据库非常灵活和可伸缩。NoSQL 数据库中的数据以类似 JSON 的格式存储，称为 RSON。在处理大型和非结构化数据时，MongoDB 使用起来非常方便，因此它是数据分析中使用最广泛的数据库。它提供了高速度和高可用性。在本文中，让我们看看如何将 python 脚本连接到 MongoDB 并执行所需的操作。

## Python MongoDB 驱动程序

PyMongo 是连接 MongoDB 和 python 的原生驱动。PyMongo 拥有从 python 代码执行数据库操作的所有库。由于 pymongo 是一个低级驱动程序，所以它速度快、直观，并且提供了更多的控制。要安装 PyMongo，请打开命令行并键入以下命令

```py
C:\Users\Your Name\AppData\Local\Programs\Python\Python36-32\Scripts>python -m pip install pymongo

```

这个命令将安装 PyMongo。我们可以在脚本中安装 PyMongo，并开始访问 MongoDB 资源。

## MongoDB 数据库

现在让我们在 MongoDB 中创建一个数据库。我们将使用 PyMongo 的 MongoClient()类来创建数据库。我们将传递正确的本地主机 IP 地址和 post 来创建数据库。并使用客户端为数据库指定一个所需的名称。

```py
from pymongo import MongoClient

#Creating a pymongo client
client = MongoClient('localhost', 27017)

#Getting the database instance
db = client['mongodb1']
print("Database created.")

#Verify the database
print("List of existing databases")
print(client.list_database_names())

```

## 输出

```py
Database created.
List of existing databases:
['admin', 'config', 'local', 'mongodb1']
```

## 创建收藏

在数据库内部，我们可以创建多个集合，集合可以与传统数据库的表进行比较，我们可以在集合中存储多个记录。现在，让我们看看如何在数据库中创建集合。另外，请注意，当至少有一个文档插入到集合中时，就会创建我们的集合。

```py
#create a collection named "students"
mycol = mydb["students"]

```

## 插入收藏

在 MongoDB 中，记录被称为文档。要将文档插入到集合中，我们应该使用 insert_one()方法。我们可以在 insert_one 方法中将创建的文档作为参数传递。让我们用一个例子来了解如何插入一个文档。

```py
#create a document
test = { "name": "Ripun", "class": "Seventh" }

#insert a document to the collection
x = mycol.insert_one(test)

```

## 插入多条记录

要将多条记录插入到一个集合中，我们可以使用 insert_many()方法。为了实现这一点，我们将首先创建一个包含多个文档的列表，并将它们传递给 insert_many()方法。

mylist = [
{ "name": "Amy "，" class": "Seventh"}，
{ "name": "Hannah "，" class": "Sixth"}，
{ "name": "Viola "，" class ":" Sixth " }]x = mycol . insert _ many(my list)

我们也可以插入他们的 id。

```py
mylist = [ { "_id":1,"name": "Amy", "class": "Seventh"},
  { "_id":2,"name": "Hannah", "class": "Sixth"},
  { "_id":3,"name": "Viola", "class": "Sixth"}]   

x = mycol.insert_many(mylist)

print(x.inserted_ids)

```

## 从集合中访问文档

现在，一旦集合被结构化并加载了数据，我们就会希望根据我们的需求来访问它们。要访问数据，我们可以使用 find()方法。

find_one()方法返回集合中的第一个匹配项。

find()方法返回集合中的所有匹配项。当不带任何参数使用 find()方法时，其行为与 SQL 中的 Select all 相同。

## 输出

```py
x = mycol.find_one()

# This prints the first document
print(x)

for x in mycol.find():
  print(x)

```

有时，我们希望只检索文档的特定字段。要在结果中包含该字段，传递的参数值应为 1，如果该值为 0，则它将从结果中排除。

```py
for x in mycol.find({},{ "_id": 0, "name": 1, "class": 1 }):
  print(x)

```

上面的代码将只从我们的集合中返回 name 和 class 字段，而不包括 id 字段。

## 查询 MongoDB 数据库

通过使用查询对象，我们可以使用 find()以更精确的方式检索结果。

### 经营者

下面是 MongoDB 查询中使用的操作符列表。

| 操作 | 句法 | 例子 |
| 平等 | {"key" : "value"} | db.mycol.find({"by ":"教程点" }) |
| 不到 | {"key" :{$lt:"value"}} | db . mycol . find({ " likes ":{ $ lt:50 } }) |
| 小于等于 | {"key" :{$lte:"value"}} | db . mycol . find({ " likes ":{ $ LTE:50 } }) |
| 大于 | {"key" :{$gt:"value"}} | db . mycol . find({ " likes ":{ $ gt:50 } }) |
| 大于等于 | {"key" {$gte:"value"}} | db . mycol . find({ " likes ":{ $ GTE:50 } }) |
| 不等于 | {"key":{$ne: "value"}} | db . mycol . find({ " likes ":{ $ ne:50 } }) |

**示例代码:**

以下代码检索名称字段为 Sathish 的文档。

```py
from pymongo import MongoClient

#Creating a pymongo client
client = MongoClient('localhost', 27017)

#Getting the database instance
db = client['sdsegf']

#Creating a collection
coll = db['example']

#Inserting document into a collection
data = [
   {"_id": "1001", "name": "Ram", "age": "26", "city": "Hyderabad"},
   {"_id": "1002", "name": "Mukesh", "age": "27", "city": "Bangalore"},
   {"_id": "1003", "name": "Vel", "age": "28", "city": "Mumbai"},
   {"_id": "1004", "name": "Sathish", "age": "25", "city": "Pune"},
   {"_id": "1005", "name": "Rashiga", "age": "23", "city": "Delhi"},
   {"_id": "1006", "name": "Priya", "age": "26", "city": "Chennai"}
]
res = coll.insert_many(data)
print("Data inserted ......")

#Retrieving data
print("Documents in the collection: ")

for doc1 in coll.find({"name":"Sathish"}):
   print(doc1)

```

**输出**

```py
Data inserted ......
Documents in the collection:
{'_id': '1004', 'name': 'Sathish', 'age': '25', 'city': 'Pune'}

```

现在让我们检索年龄大于 25 岁的人的记录。我们将使用$gt 操作符来实现它。

```py
for doc in coll.find({"age":{"$gt":"25"}}):
   print(doc)
```

**输出**

{"_id": "1002 "，"姓名": "穆克什"，"年龄": " 27 "，"城市": "班加罗尔" }
{"_id": "1003 "，"姓名":" Vel "，"年龄":" 28 "，"城市":"孟买" }

以类似的方式，我们可以使用$lt 来过滤值小于我们指定值的记录。我们也可以在字符串上使用这些操作符。例如，当我们使用" name":{"$gt":"J"}来检索名称以' J '开头或其后有字母的所有记录时。

## Python MongoDB 中的删除操作

我们可以使用 delete_one()方法删除一个文档。

`delete_one()`方法的第一个参数是一个查询对象，表示要删除的文档。

```py
myquery = {"name" : "Mukesh"}

coll.delete_one(myquery)

```

要删除多个文档，我们可以使用 delete_many()方法。

```py
myquery = { "name": {"$regex": "^S"} }

x = coll.delete_many(myquery)

```

上面的代码将删除人名以“S”开头或以字母顺序放在 S 后面的所有记录。

要删除集合中的所有文档，我们可以向`delete_many()`方法传递一个空的查询对象。下面的代码将删除集合中的所有文档。

```py
x = coll.delete_many({})

```

如果我们想要删除整个集合本身，我们可以使用 drop()方法。

coll.drop()

## 结论

在本文中，我们已经了解了如何将 MongoDB 连接到 python，并在其上执行各种必需的基本操作。强烈建议读者获得一些使用 MongoDB 的实践经验，并熟悉语法和各种查询。

## 参考

[https://www.mongodb.com/languages/python](https://www.mongodb.com/languages/python)

[https://docs.mongodb.com/drivers/python/](https://docs.mongodb.com/drivers/python/)