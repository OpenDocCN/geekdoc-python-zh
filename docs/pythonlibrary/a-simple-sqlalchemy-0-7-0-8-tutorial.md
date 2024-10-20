# 一个简单的 SqlAlchemy 0.7 / 0.8 教程

> 原文：<https://www.blog.pythonlibrary.org/2012/07/01/a-simple-sqlalchemy-0-7-0-8-tutorial/>

几年前，我写了一篇关于 SQLAlchemy 的相当有缺陷的教程。我决定是时候从头开始重新做这个教程了，希望这次能做得更好。因为我是一个音乐迷，我们将创建一个简单的数据库来存储专辑信息。没有一些关系的数据库就不是数据库，所以我们将创建两个表并将它们连接起来。以下是我们将要学习的其他一些东西:

*   向每个表中添加数据
*   修改数据
*   删除数据
*   基本查询

但是首先我们需要实际制作数据库，所以我们将从那里开始我们的旅程。注意 SQLAlchemy 是一个第三方包，所以如果你想继续的话，你需要[安装](http://www.sqlalchemy.org/download.html)它。

### 如何创建数据库

用 SQLAlchemy 创建数据库真的很容易。他们现在已经完全采用了他们的[声明式](http://docs.sqlalchemy.org/en/rel_0_7/orm/extensions/declarative.html)方法来创建数据库，所以我们不会讨论旧的学校方法。您可以在这里阅读代码，然后我们将在清单后面解释它。如果你想查看你的 SQLite 数据库，我推荐火狐的 SQLite 管理器[插件](https://addons.mozilla.org/en-US/firefox/addon/sqlite-manager/)。或者你可以使用我一个月前创建的[简单 wxPython 应用程序](https://www.blog.pythonlibrary.org/2012/06/04/wxpython-and-sqlalchemy-loading-random-sqlite-databases-for-viewing/)。

```py
# table_def.py
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

engine = create_engine('sqlite:///mymusic.db', echo=True)
Base = declarative_base()

########################################################################
class Artist(Base):
    """"""
    __tablename__ = "artists"

    id = Column(Integer, primary_key=True)
    name = Column(String)  

    #----------------------------------------------------------------------
    def __init__(self, name):
        """"""
        self.name = name    

########################################################################
class Album(Base):
    """"""
    __tablename__ = "albums"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    release_date = Column(Date)
    publisher = Column(String)
    media_type = Column(String)

    artist_id = Column(Integer, ForeignKey("artists.id"))
    artist = relationship("Artist", backref=backref("albums", order_by=id))

    #----------------------------------------------------------------------
    def __init__(self, title, release_date, publisher, media_type):
        """"""
        self.title = title
        self.release_date = release_date
        self.publisher = publisher
        self.media_type = media_type

# create tables
Base.metadata.create_all(engine)

```

如果您运行这段代码，那么您应该会看到发送到 stdout 的以下输出:

```py
2012-06-27 16:34:24,479 INFO sqlalchemy.engine.base.Engine PRAGMA table_info("artists")
2012-06-27 16:34:24,479 INFO sqlalchemy.engine.base.Engine ()
2012-06-27 16:34:24,480 INFO sqlalchemy.engine.base.Engine PRAGMA table_info("albums")
2012-06-27 16:34:24,480 INFO sqlalchemy.engine.base.Engine ()
2012-06-27 16:34:24,480 INFO sqlalchemy.engine.base.Engine 
CREATE TABLE artists (
    id INTEGER NOT NULL, 
    name VARCHAR, 
    PRIMARY KEY (id)
)

2012-06-27 16:34:24,483 INFO sqlalchemy.engine.base.Engine ()
2012-06-27 16:34:24,558 INFO sqlalchemy.engine.base.Engine COMMIT
2012-06-27 16:34:24,559 INFO sqlalchemy.engine.base.Engine 
CREATE TABLE albums (
    id INTEGER NOT NULL, 
    title VARCHAR, 
    release_date DATE, 
    publisher VARCHAR, 
    media_type VARCHAR, 
    artist_id INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(artist_id) REFERENCES artists (id)
)

2012-06-27 16:34:24,559 INFO sqlalchemy.engine.base.Engine ()
2012-06-27 16:34:24,615 INFO sqlalchemy.engine.base.Engine COMMIT

```

为什么会这样？因为当我们创建引擎对象时，我们将其 **echo** 参数设置为 True。[引擎](http://docs.sqlalchemy.org/en/rel_0_7/core/engines.html)是数据库连接信息所在的地方，它包含了所有的 DBAPI 内容，使得与数据库的通信成为可能。您会注意到我们正在创建一个 SQLite 数据库。从 Python 2.5 开始，该语言就支持 SQLite。如果您想连接到其他数据库，那么您需要编辑连接字符串。以防你对我们正在谈论的内容感到困惑，这里是有问题的代码:

```py
engine = create_engine('sqlite:///mymusic.db', echo=True)

```

字符串 **'sqlite:///mymusic.db'** ，是我们的连接字符串。接下来，我们创建声明性基类的一个实例，这是我们的表类所基于的。接下来我们有两个类，**艺术家**和**专辑**，它们定义了我们的数据库表的外观。您会注意到我们有列，但是没有列名。SQLAlchemy 实际上使用变量名作为列名，除非您在列定义中特别指定一个。您会注意到我们在两个类中都使用了一个“id”整数字段作为主键。该字段将自动递增。在使用外键之前，其他列都是不言自明的。在这里你会看到我们将 **artist_id** 绑定到 **Artist** 表中的 id。**关系**指令告诉 SQLAlchemy 将专辑类/表绑定到艺术家表。由于我们设置 ForeignKey 的方式，relationship 指令告诉 SQLAlchemy 这是一个**多对一**关系，这正是我们想要的。一位艺术家的多张专辑。你可以在这里阅读更多关于表关系[的内容。](http://docs.sqlalchemy.org/en/rel_0_7/orm/relationships.html#relationship-patterns)

脚本的最后一行将在数据库中创建表。如果您多次运行这个脚本，它在第一次之后不会做任何新的事情，因为表已经创建好了。您可以添加另一个表，然后它会创建一个新表。

### 如何向表格中插入/添加数据

除非数据库中有一些数据，否则它没有多大用处。在这一节中，我们将向您展示如何连接到您的数据库并将一些数据添加到这两个表中。看一看一些代码然后解释它要容易得多，所以我们就这么做吧！

```py
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from table_def import Album, Artist

engine = create_engine('sqlite:///mymusic.db', echo=True)

# create a Session
Session = sessionmaker(bind=engine)
session = Session()

# Create an artist
new_artist = Artist("Newsboys")
new_artist.albums = [Album("Read All About It", 
                           datetime.date(1988,12,1),
                           "Refuge", "CD")]

# add more albums
more_albums = [Album("Hell Is for Wimps",
                     datetime.date(1990,7,31),
                     "Star Song", "CD"),
               Album("Love Liberty Disco", 
                     datetime.date(1999,11,16),
                     "Sparrow", "CD"),
               Album("Thrive",
                     datetime.date(2002,3,26),
                     "Sparrow", "CD")]
new_artist.albums.extend(more_albums)

# Add the record to the session object
session.add(new_artist)
# commit the record the database
session.commit()

# Add several artists
session.add_all([
    Artist("MXPX"),
    Artist("Kutless"),
    Artist("Thousand Foot Krutch")
    ])
session.commit()

```

首先，我们需要从前面的脚本中导入我们的表定义。然后，我们用引擎连接到数据库，并创建新的东西，即会话对象。会话是数据库的句柄，让我们与它交互。我们用它来创建、修改和删除记录，我们还用会话来查询数据库。接下来，我们创建一个艺术家对象并添加一个相册。您会注意到，要添加一个相册，您只需创建一个相册对象列表，并将 artist 对象的“albums”属性设置为该列表，或者您可以扩展它，如示例的第二部分所示。在脚本的最后，我们使用 **add_all** 添加了三个额外的艺术家。您可能已经注意到，您需要使用会话对象的**提交**方法将数据写入数据库。现在是时候把注意力转向修改数据了。

**关于 __init__** 的一个注意事项:正如我的一些敏锐的读者指出的，对于表定义，您实际上不需要 **__init__** 构造函数。我把它们留在那里是因为官方文档仍然在使用它们，我没有意识到我可以把它们漏掉。无论如何，如果您在声明性表定义中不考虑 __init__ 的话，那么在创建记录时您将需要使用关键字参数。例如，您应该执行以下操作，而不是上一个示例中显示的操作:

```py
new_artist = Artist(name="Newsboys")
new_artist.albums = [Album(title="Read All About It", 
                           release_date=datetime.date(1988,12,01),
                           publisher="Refuge", media_type="CD")]

```

### 如何用 SQLAlchemy 修改记录

如果你保存了一些错误的数据会发生什么。例如，你打错了你最喜欢的专辑的名字，或者你弄错了你的粉丝版本的发行日期？你需要学习如何修改那个记录！这实际上是我们学习 SQLAlchemy 查询的起点，因为您需要找到需要更改的记录，这意味着您需要为它编写一个查询。下面的一些代码向我们展示了这种方法:

```py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from table_def import Album, Artist

engine = create_engine('sqlite:///mymusic.db', echo=True)

# create a Session
Session = sessionmaker(bind=engine)
session = Session()

# querying for a record in the Artist table
res = session.query(Artist).filter(Artist.name=="Kutless").first()
print(res.name)

# changing the name
res.name = "Beach Boys"
session.commit()

# editing Album data
artist, album = session.query(Artist, Album).filter(Artist.id==Album.artist_id).filter(Album.title=="Thrive").first()
album.title = "Step Up to the Microphone"
session.commit()

```

我们的第一个查询使用**过滤器**方法通过名字查找艺术家。的”。first()"告诉 SQLAlchemy 我们只想要第一个结果。我们本来可以用”。all()"如果我们认为会有多个结果，并且我们想要所有的结果。无论如何，这个查询返回一个我们可以操作的 Artist 对象。正如你所看到的，我们把**的名字**从“无裤袜”改成了“沙滩男孩”，然后进行了修改。

查询连接表稍微复杂一点。这一次，我们编写了一个查询来查询我们的两个表。它使用艺术家 id 和专辑标题进行过滤。它返回两个对象:艺术家和专辑。一旦我们有了这些，我们可以很容易地改变专辑的标题。那不是很容易吗？此时，我们可能应该注意到，如果我们错误地向会话添加了内容，我们可以通过使用 **session.rollback()** 回滚我们的更改/添加/删除。说到删除，让我们来解决这个问题吧！

### 如何删除 SQLAlchemy 中的记录

有时候你只需要删除一条记录。无论是因为你卷入了一场掩盖活动，还是因为你不想让人们知道你对布兰妮音乐的喜爱，你都必须清除证据。在本节中，我们将向您展示如何做到这一点！幸运的是，SQLAlchemy 让删除记录变得非常容易。看看下面的代码就知道了！

```py
# deleting_data.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from table_def import Album, Artist

engine = create_engine('sqlite:///mymusic.db', echo=True)

# create a Session
Session = sessionmaker(bind=engine)
session = Session()

res = session.query(Artist).filter(Artist.name=="MXPX").first()

session.delete(res)
session.commit()

```

如您所见，您所要做的就是创建另一个 SQL 查询来查找您想要删除的记录，然后调用 **session.delete(res)** 。在这种情况下，我们删除了我们的 MXPX 记录。有人觉得朋克永远不死，但一定不认识什么 DBA！我们已经看到了运行中的查询，但是让我们更仔细地看一看，看看我们是否能学到新的东西。

### SQLAlchemy 的基本 SQL 查询

SQLAlchemy 提供了您可能需要的所有查询。我们将花一点时间来看一些基本的，比如几个简单的选择，一个连接的选择和使用 LIKE 查询。您还将了解到哪里去获取其他类型查询的信息。现在，让我们看一些代码:

```py
# queries.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from table_def import Album, Artist

engine = create_engine('sqlite:///mymusic.db', echo=True)

# create a Session
Session = sessionmaker(bind=engine)
session = Session()

# how to do a SELECT * (i.e. all)
res = session.query(Artist).all()
for artist in res:
    print artist.name

# how to SELECT the first result
res = session.query(Artist).filter(Artist.name=="Newsboys").first()

# how to sort the results (ORDER_BY)
res = session.query(Album).order_by(Album.title).all()
for album in res:
    print album.title

# how to do a JOINed query
qry = session.query(Artist, Album)
qry = qry.filter(Artist.id==Album.artist_id)
artist, album = qry.filter(Album.title=="Thrive").first()
print

# how to use LIKE in a query
res = session.query(Album).filter(Album.publisher.like("S%a%")).all()
for item in res:
    print item.publisher

```

我们运行的第一个查询将获取数据库中的所有艺术家(SELECT *)并打印出他们的每个姓名字段。接下来，您将看到如何查询特定的艺术家并返回第一个结果。第三个查询显示了如何在相册表中选择*并按相册标题对结果进行排序。第四个查询与我们在编辑部分使用的查询相同(对一个连接的查询),只是我们对它进行了分解，以更好地适应关于行长度的标准。分解长查询的另一个原因是，如果你搞砸了，它们会变得更易读，更容易修复。最后一个查询使用 LIKE，它允许我们进行模式匹配或查找与指定字符串“相似”的内容。在这种情况下，我们希望找到出版商以大写字母“S”、某个字符“a”以及其他任何字母开头的任何记录。例如，这将匹配发行商 Sparrow 和 Star。

SQLAlchemy 还支持 IN、IS NULL、NOT、AND、OR 以及大多数 DBA 使用的所有其他过滤关键字。SQLAlchemy 还支持文字 SQL、标量等。

### 包扎

此时，您应该对 SQLAlchemy 有足够的了解，可以放心地开始使用它。该项目也有很好的文档，你应该能够用它来回答你需要知道的任何事情。如果您遇到困难，SQLAlchemy 用户组/邮件列表会对新用户做出积极响应，甚至主要开发人员也会在那里帮助您解决问题。

### 源代码

*   [sqlalchemy0708.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2012/07/sqlalchemy0708.zip)

### 进一步阅读

*   SQLAlchemy 官方[文档](http://docs.sqlalchemy.org/en/rel_0_7/index.html)
*   wxPython 和 SQLAlchemy: [加载随机 SQLite 数据库以供查看](https://www.blog.pythonlibrary.org/2012/06/04/wxpython-and-sqlalchemy-loading-random-sqlite-databases-for-viewing/)
*   SqlAlchemy: [连接到预先存在的数据库](https://www.blog.pythonlibrary.org/2010/09/10/sqlalchemy-connecting-to-pre-existing-databases/)
*   wxPython 和 SqlAlchemy:[MVC 和 CRUD 简介](https://www.blog.pythonlibrary.org/2011/11/10/wxpython-and-sqlalchemy-an-intro-to-mvc-and-crud/)
*   另一个循序渐进的 SqlAlchemy 教程(第 1 部分，共 2 部分)