# peewee 介绍——另一种 Python ORM

> 原文：<https://www.blog.pythonlibrary.org/2014/07/17/an-intro-to-peewee-another-python-orm/>

我认为除了 SQLAlchemy 之外，尝试一些不同的 Python 对象关系映射器(ORM)会很有趣。我最近偶然发现了一个名为 [peewee](https://github.com/coleifer/peewee) 的项目。对于这篇文章，我们将从我的 [SQLAlchemy 教程](https://www.blog.pythonlibrary.org/2012/07/01/a-simple-sqlalchemy-0-7-0-8-tutorial/)中提取例子，并将其移植到 peewee，看看它是如何站起来的。peewee 项目支持 sqlite、postgres 和 MySQL 开箱即用，虽然没有 SQLAlchemy 灵活，但也不错。你也可以通过一个方便的 [Flask-peewee 插件](https://github.com/coleifer/flask-peewee/)在 Flask web 框架中使用 peewee。

无论如何，让我们开始玩这个有趣的小图书馆吧！

* * *

### 入门指南

首先，你得去找皮威。幸运的是，如果您安装了 pip，这真的很容易:

```py

pip install peewee

```

一旦安装完毕，我们就可以开始了！

* * *

### 创建数据库

用 peewee 创建数据库非常容易。事实上，在 peewee 中创建数据库比在 SQLAlchemy 中更容易。你所需要做的就是调用 peewee 的 **SqliteDatabase** 方法，如果你想要一个内存中的数据库，把文件的路径或者“:memory:”传递给它。让我们创建一个数据库来保存我们音乐收藏的信息。我们将创建两个表:艺术家和专辑。

```py

# models.py
import peewee

database = peewee.SqliteDatabase("wee.db")

########################################################################
class Artist(peewee.Model):
    """
    ORM model of the Artist table
    """
    name = peewee.CharField()

    class Meta:
        database = database

########################################################################
class Album(peewee.Model):
    """
    ORM model of album table
    """
    artist = peewee.ForeignKeyField(Artist)
    title = peewee.CharField()
    release_date = peewee.DateTimeField()
    publisher = peewee.CharField()
    media_type = peewee.CharField()

    class Meta:
        database = database

if __name__ == "__main__":
    try:
        Artist.create_table()
    except peewee.OperationalError:
        print "Artist table already exists!"

    try:
        Album.create_table()
    except peewee.OperationalError:
        print "Album table already exists!"

```

这段代码非常简单。我们在这里所做的就是创建两个定义表的类。我们设置字段(或列)，并通过嵌套类 Meta 将数据库连接到模型。然后我们直接调用这个类来创建表。这有点奇怪，因为您通常不会像这样直接调用一个类，而是创建该类的一个实例。然而，这是根据 peewee 的[文档](http://nbviewer.ipython.org/gist/coleifer/d3faf30bbff67ce5f70c)推荐的程序，并且效果很好。现在我们准备学习如何向我们的数据库添加一些数据。

* * *

### 如何向表格中插入/添加数据

事实证明，将数据插入我们的数据库也非常容易。让我们来看看:

```py

# add_data.py

import datetime
import peewee

from models import Album, Artist

new_artist = Artist.create(name="Newsboys")
album_one = Album(artist=new_artist,
                  title="Read All About It",
                  release_date=datetime.date(1988,12,01),
                  publisher="Refuge",
                  media_type="CD")
album_one.save()

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
    a.save()

bands = ["MXPX", "Kutless", "Thousand Foot Krutch"]
for band in bands:
    artist = Artist.create(name=band)
    artist.save()

```

这里我们调用该类的 **create** 方法来添加乐队或唱片。该类也支持一个 **insert_many** 方法，但是每当我试图通过 **save()** 方法保存数据时，我都会收到一个 **OperationalError** 消息。如果你碰巧知道如何做到这一点，请在评论中给我留言，我会更新这篇文章。作为一种变通方法，我只是循环遍历一个字典列表，并以这种方式添加记录。

**更新**:《peewee》的作者在 [reddit](http://www.reddit.com/r/Python/comments/2bblm4/an_intro_to_peewee_another_python_orm/) 上回复了我，给了我这个一次添加多条记录的解决方案:

```py

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
Album.insert_many(albums).execute()

```

现在我们准备学习如何修改数据库中的记录！

* * *

### 使用基本查询通过 peewee 修改记录

在数据库世界中，修改记录是很常见的事情。peewee 项目使修改数据变得非常容易。下面是一些演示如何操作的代码:

```py

# edit_data.py

import peewee

from models import Album, Artist

band = Artist.select().where(Artist.name=="Kutless").get()
print band.name

# shortcut method
band = Artist.get(Artist.name=="Kutless")
print band.name

# change band name
band.name = "Beach Boys"
band.save()

album = Album.select().join(Artist).where(
    (Album.title=="Thrive") & (Artist.name == "Newsboys")
    ).get()
album.title = "Step Up to the Microphone"
album.save()

```

基本上，我们只需查询这些表，以获得我们想要修改的艺术家或专辑。前两个查询做同样的事情，但是一个比另一个短。这是因为 peewee 提供了一种快捷的查询方法。要实际更改记录，我们只需将返回对象的属性设置为其他值。在这种情况下，我们把乐队的名字从“无裤”改成了“沙滩男孩”。

最后一个查询演示了如何创建一个 SQL 连接，它允许我们在两个表之间进行匹配。如果您碰巧拥有两张同名的 CD，但您只想让查询返回与名为“Newsboys”的乐队相关联的专辑，那么这就很好了。

这些查询有点难以理解，所以您可以将它们分成更小的部分。这里有一个例子:

```py

query = Album.select().join(Artist)
qry_filter = (Album.title=="Step Up to the Microphone") & (Artist.name == "Newsboys")
album = query.where(qry_filter).get()

```

这更容易跟踪和调试。您也可以对 SQLAlchemy 的查询使用类似的技巧。

* * *

### 如何删除 peewee 中的记录

在 peewee 中从表中删除记录只需要很少的代码。看看这个:

```py

# del_data.py

from models import Artist

band = Artist.get(Artist.name=="MXPX")
band.delete_instance()

```

我们所要做的就是查询我们想要删除的记录。一旦我们有了实例，我们就调用它的 **delete_instance** 方法，这样就删除了记录。真的就这么简单！

* * *

### 包扎

peewee 项目很酷。它最大的缺点是它支持的数据库后端数量有限。然而，该项目比 SQLAlchemy 更容易使用，我认为这很神奇。peewee 项目的文档非常好，值得一读，以了解本教程中没有涉及的所有其他功能。试试看，看你怎么想！

* * *

### 额外资源

*   用于 [peewee 项目](https://github.com/coleifer/peewee)的 Github
*   peewee [文档](http://peewee.readthedocs.org/en/latest/index.html)