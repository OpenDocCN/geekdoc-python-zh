# python 101:Pony ORM 简介

> 原文：<https://www.blog.pythonlibrary.org/2014/07/21/python-101-an-intro-to-pony-orm/>

Pony ORM 项目是 Python 的另一个对象关系映射器包。它们允许您使用生成器查询数据库。他们还有一个在线的 ER 图编辑器，可以帮助你创建一个模型。当我第一次开始使用 PonyORM 时，它们是我见过的唯一一个带有多许可方案的 Python 包，在这种方案中，您可以使用 GNU 许可证进行开发，或者购买非开源作品的许可证。但是，截至 2016 年 10 月，PonyORM 包处于 Apache 2.0 许可之下。

在本文中，我们将花一些时间学习这个包的基础知识。

* * *

### 入门指南

因为这个项目不包含在 Python 中，所以您需要下载并安装它。如果你有 pip，你可以这样做:

```py

pip install pony

```

否则你将不得不下载源代码并通过它的 **setup.py** 脚本安装它。

* * *

### 创建数据库

我们将首先创建一个数据库来保存一些音乐。我们将需要两个表:艺术家和专辑。我们开始吧！

```py

import datetime
import pony.orm as pny

database = pny.Database("sqlite",
                        "music.sqlite",
                        create_db=True)

########################################################################
class Artist(database.Entity):
    """
    Pony ORM model of the Artist table
    """
    name = pny.Required(unicode)
    albums = pny.Set("Album")

########################################################################
class Album(database.Entity):
    """
    Pony ORM model of album table
    """
    artist = pny.Required(Artist)
    title = pny.Required(unicode)
    release_date = pny.Required(datetime.date)
    publisher = pny.Required(unicode)
    media_type = pny.Required(unicode)

# turn on debug mode
pny.sql_debug(True)

# map the models to the database 
# and create the tables, if they don't exist
database.generate_mapping(create_tables=True)

```

如果我们不指定主键，Pony ORM 会自动为我们创建主键。要创建外键，您只需将 model 类传递到另一个表中，就像我们在 Album 类中所做的那样。每个必填字段都采用 Python 类型。我们的大多数字段都是 unicode 的，其中一个是 datatime 对象。接下来我们打开调试模式，它将输出 Pony 在最后一条语句中创建表时生成的 SQL。注意，如果多次运行这段代码，将不会重新创建表。Pony 将在创建表之前检查它们是否存在。

如果您运行上面的代码，您应该会看到这样的输出:

```py

GET CONNECTION FROM THE LOCAL POOL
PRAGMA foreign_keys = false
BEGIN IMMEDIATE TRANSACTION
CREATE TABLE "Artist" (
  "id" INTEGER PRIMARY KEY AUTOINCREMENT,
  "name" TEXT NOT NULL
)

CREATE TABLE "Album" (
  "id" INTEGER PRIMARY KEY AUTOINCREMENT,
  "artist" INTEGER NOT NULL REFERENCES "Artist" ("id"),
  "title" TEXT NOT NULL,
  "release_date" DATE NOT NULL,
  "publisher" TEXT NOT NULL,
  "media_type" TEXT NOT NULL
)

CREATE INDEX "idx_album__artist" ON "Album" ("artist")

SELECT "Album"."id", "Album"."artist", "Album"."title", "Album"."release_date", "Album"."publisher", "Album"."media_type"
FROM "Album" "Album"
WHERE 0 = 1

SELECT "Artist"."id", "Artist"."name"
FROM "Artist" "Artist"
WHERE 0 = 1

COMMIT
PRAGMA foreign_keys = true
CLOSE CONNECTION

```

是不是很棒？现在我们准备学习如何向数据库添加数据。

* * *

### 如何向表格中插入/添加数据

Pony 使得向表中添加数据变得相当容易。让我们来看看有多简单:

```py

import datetime
import pony.orm as pny

from models import Album, Artist

#----------------------------------------------------------------------
@pny.db_session
def add_data():
    """"""

    new_artist = Artist(name=u"Newsboys")
    bands = [u"MXPX", u"Kutless", u"Thousand Foot Krutch"]
    for band in bands:
        artist = Artist(name=band)

    album = Album(artist=new_artist,
                  title=u"Read All About It",
                  release_date=datetime.date(1988,12,01),
                  publisher=u"Refuge",
                  media_type=u"CD")

    albums = [{"artist": new_artist,
               "title": "Hell is for Wimps",
               "release_date": datetime.date(1990,07,31),
               "publisher": "Sparrow",
               "media_type": "CD"
               },
              {"artist": new_artist,
               "title": "Love Liberty Disco", 
               "release_date": datetime.date(1999,11,16),
               "publisher": "Sparrow",
               "media_type": "CD"
              },
              {"artist": new_artist,
               "title": "Thrive",
               "release_date": datetime.date(2002,03,26),
               "publisher": "Sparrow",
               "media_type": "CD"}
              ]

    for album in albums:
        a = Album(**album)

if __name__ == "__main__":
    add_data()

    # use db_session as a context manager
    with pny.db_session:
        a = Artist(name="Skillet")

```

您会注意到，我们需要使用一个名为 **db_session** 的装饰器来处理数据库。它负责打开连接、提交数据和关闭连接。您还可以将它用作上下文管理器，这段代码的最后演示了这一点。

* * *

### 使用基本查询修改记录

在本节中，我们将学习如何进行一些基本的查询并修改数据库中的一些条目。

```py

import pony.orm as pny

from models import Artist, Album

with pny.db_session:
    band = Artist.get(name="Newsboys")
    print band.name

    for record in band.albums:
        print record.title

    # update a record
    band_name = Artist.get(name="Kutless")
    band_name.name = "Beach Boys"

```

这里我们使用 **db_session** 作为上下文管理器。我们通过查询从数据库中获取一个 artist 对象并打印其名称。然后我们遍历返回对象中包含的艺术家的专辑。最后，我们更改一个艺术家的名字。

让我们尝试使用生成器查询数据库:

```py

result = pny.select(i.name for i in Artist)
result.show()

```

如果您运行这段代码，您应该会看到如下内容:

```py

i.name              
--------------------
Newsboys            
MXPX                
Beach Boys             
Thousand Foot Krutch

```

文档中有几个值得一试的例子。注意，Pony 还通过其 **select_by_sql** 和 **get_by_sql** 方法支持使用 SQL 本身。

* * *

### 如何在表格中删除记录

用 Pony 删除记录也很容易。让我们从数据库中删除一个波段:

```py

import pony.orm as pny

from models import Artist

with pny.db_session:
    band = Artist.get(name="MXPX")
    band.delete()

```

我们再次使用 **db_session** 来访问数据库并提交我们的更改。我们使用 band 对象的 **delete** 方法来删除记录。你将需要挖掘，找出是否 Pony 支持级联删除，如果你删除艺术家，它也将删除所有专辑连接到它。根据[文件](http://doc.ponyorm.com/collections.html?highlight=cascade#Set.cascade_delete)，如果该字段为必填，则级联被启用。

* * *

### 包扎

现在您知道了使用 Pony ORM 包的基本知识。我个人认为文档需要一点工作，因为你必须挖掘很多才能找到一些我认为应该在教程中的功能。尽管总的来说，文档仍然比大多数项目要好得多。试试看，看看你有什么想法！

* * *

### 额外资源

*   小马奥姆的[网站](http://ponyorm.com/)
*   小马[文档](http://doc.ponyorm.com/)
*   [SQLAlchemy 教程](https://www.blog.pythonlibrary.org/2012/07/01/a-simple-sqlalchemy-0-7-0-8-tutorial/)
*   peewee 的简介