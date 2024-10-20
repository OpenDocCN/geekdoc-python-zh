# SqlAlchemy:连接到预先存在的数据库

> 原文：<https://www.blog.pythonlibrary.org/2010/09/10/sqlalchemy-connecting-to-pre-existing-databases/>

用 Python 访问数据库是一个简单的过程。Python 甚至提供了一个内置在主发行版中的 sqlite 数据库库(从 2.5 开始)。我最喜欢用 Python 访问数据库的方法是使用第三方包 SqlAlchemy。SqlAlchemy 是一个对象关系映射器(ORM ),这意味着它采用 SQL 结构并使它们更像目标语言。在这种情况下，您最终使用 Python 语法执行 SQL，而不是直接执行 SQL，并且您可以使用相同的代码访问多个数据库后端(如果您小心的话)。

在本文中，我们将研究如何使用 SqlAlchemy 连接到预先存在的数据库。如果我的经验有任何启示的话，你可能会花更多的时间在你没有创建的数据库上，而不是在你创建的数据库上。本文将向您展示如何连接到它们。

## SqlAlchemy 的自动加载

SqlAlchemy 有两种方法来定义数据库列。第一种方法是长方法，在这种方法中，定义每个字段及其类型。简单(或简短)的方法是使用 SqlAlchemy 的 **autoload** 功能，它将以一种相当神奇的方式自省表格并提取字段名称。我们将从 autoload 方法开始，然后在下一节展示这个漫长的过程。

在我们开始之前，需要注意有两种配置 SqlAlchemy 的方法:一种是手写的方法，另一种是声明性的(或“速记的”)方法。我们将两种方式都过一遍。让我们从长版本开始，然后“声明性地”做。

```py

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import mapper, sessionmaker

class Bookmarks(object):
    pass

#----------------------------------------------------------------------
def loadSession():
    """"""    
    dbPath = 'places.sqlite'
    engine = create_engine('sqlite:///%s' % dbPath, echo=True)

    metadata = MetaData(engine)
    moz_bookmarks = Table('moz_bookmarks', metadata, autoload=True)
    mapper(Bookmarks, moz_bookmarks)

    Session = sessionmaker(bind=engine)
    session = Session()
    return session

if __name__ == "__main__":
    session = loadSession()
    res = session.query(Bookmarks).all()
    res[1].title

```

在这个代码片段中，我们从 sqlalchemy 导入了一些方便的类和实用程序，允许我们定义引擎(一种数据库连接/接口)、元数据(一个表目录)和会话(一个允许我们查询的数据库“句柄”)。注意，我们有一个“places.sqlite”文件。这个数据库来自 Mozilla Firefox。如果您已经安装了它，请去寻找它，因为它为这类事情提供了一个极好的测试平台。在我的 Windows XP 机器上，它位于以下位置:“C:\ Documents and Settings \ Mike \ Application Data \ Mozilla \ Firefox \ Profiles \ f 7 csnzvk . default”。只需将该文件复制到您将用于本文中的脚本的位置。如果你试着在适当的地方使用它，并且打开了 Firefox，你可能会遇到问题，因为你的代码可能会中断 Firefox，反之亦然。

在 **create_engine** 调用中，我们将 **echo** 设置为 **True** 。这将导致 SqlAlchemy 将其生成的所有 SQL 发送到 stdout，这对于调试来说非常方便。我建议在将代码投入生产时将其设置为 False。就我们的目的而言，最有趣的一行如下:

```py

moz_bookmarks = Table('moz_bookmarks', metadata, autoload=True)

```

这告诉 SqlAlchemy 尝试自动加载“moz_bookmarks”表。如果它是一个带有主键的合适的数据库，这将非常有用。如果该表没有主键，那么您可以这样做:

```py

from sqlalchemy import Column, Integer
moz_bookmarks = Table('moz_bookmarks', metadata, 
                      Column("id", Integer, primary_key=True),
                      autoload=True)

```

这增加了一个名为“id”的额外列。这基本上是对数据库进行猴子修补。当我不得不使用供应商设计不良的数据库时，我成功地使用了这种方法。如果数据库支持，SqlAlchemy 也会自动递增它。**映射器**将表对象映射到书签类。然后我们创建一个绑定到引擎的会话，这样我们就可以进行查询。res = **session.query(书签)。all()** 基本上意味着 **SELECT * FROM moz_bookmarks** 并将结果作为书签对象列表返回。

有时坏的数据库会有一个明显应该是主键的字段。如果是这样的话，你可以只用那个名字而不用“id”。其他时候，您可以通过将主键设置为两列(即字段)来创建唯一键。您可以通过创建两列并将它们的“primary_key”都设置为 True 来实现。如果一个表有几十个字段，这比定义整个表要好得多。

让我们转到声明性自动加载方法:

```py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///places.sqlite', echo=True)
Base = declarative_base(engine)
########################################################################
class Bookmarks(Base):
    """"""
    __tablename__ = 'moz_bookmarks'
    __table_args__ = {'autoload':True}

#----------------------------------------------------------------------
def loadSession():
    """"""
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

if __name__ == "__main__":
    session = loadSession()
    res = session.query(Bookmarks).all()
    print res[1].title

```

声明性方法看起来有点不同，是吧？让我们试着打开新东西。首先，我们有一个新导入:**来自 sqlalchemy . ext . declarative import declarative _ base**。我们使用“declarative_base”来创建一个使用我们的引擎的类，然后我们从这个类创建书签子类。为了指定表名，我们使用了神奇的方法， **__tablename__** ，为了让它自动加载，我们创建了 **__table_args__** dict。然后在 **loadSession** 函数中，我们得到这样的元数据对象: **metadata = Base.metadata** 。代码的其余部分是相同的。声明性语法通常会简化代码，因为它将所有内容都放在一个类中，而不是分别创建一个类和一个表对象。如果您喜欢声明式风格，您可能还想看看 Elixir，这是一个 SqlAlchemy 扩展，在他们将这种风格添加到主包之前，它就具有声明性。

## 明确定义您的数据库

有时您不能使用 autoload，或者您只想完全控制表定义。SqlAlchemy 几乎可以让您轻松地做到这一点。我们将首先看手写版本，然后看声明式风格。

```py

from sqlalchemy import create_engine, Column, MetaData, Table
from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import mapper, sessionmaker

class Bookmarks(object):
    pass

#----------------------------------------------------------------------
def loadSession():
    """"""
    dbPath = 'places.sqlite'

    engine = create_engine('sqlite:///%s' % dbPath, echo=True)

    metadata = MetaData(engine)    
    moz_bookmarks = Table('moz_bookmarks', metadata, 
                          Column('id', Integer, primary_key=True),
                          Column('type', Integer),
                          Column('fk', Integer),
                          Column('parent', Integer),
                          Column('position', Integer),
                          Column('title', String),
                          Column('keyword_id', Integer),
                          Column('folder_type', Text),
                          Column('dateAdded', Integer),
                          Column('lastModified', Integer)
                          )

    mapper(Bookmarks, moz_bookmarks)

    Session = sessionmaker(bind=engine)
    session = Session()

if __name__ == "__main__":
    session = loadSession()
    res = session.query(Bookmarks).all()
    print res[1].title

```

这段代码与我们看到的代码并没有什么不同。我们最关心的部分是表定义。这里我们使用关键字 Column 来定义每一列。Column 类接受列的名称、类型、是否将其设置为主键以及该列是否可为空。它可能会接受一些其他的论点，但这些是你会看到最多的。这个例子没有显示可空位，因为我不知道 Firefox 表是否有这些约束。

无论如何，一旦定义了列，剩下的代码就和以前一样了。那么让我们转到声明式风格:

```py

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///places.sqlite', echo=True)
Base = declarative_base(engine)
########################################################################
class Places(Base):
    """"""
    __tablename__ = 'moz_places'

    id = Column(Integer, primary_key=True)
    url = Column(String)
    title = Column(String)
    rev_host = Column(String)
    visit_count = Column(Integer)
    hidden = Column(Integer)
    typed = Column(Integer)
    favicon_id = Column(Integer)
    frecency = Column(Integer)
    last_visit_date = Column(Integer)

    #----------------------------------------------------------------------
    def __init__(self, id, url, title, rev_host, visit_count,
                 hidden, typed, favicon_id, frecency, last_visit_date):
        """"""
        self.id = id
        self.url = url
        self.title = title
        self.rev_host = rev_host
        self.visit_count = visit_count
        self.hidden = hidden
        self.typed = typed
        self.favicon_id = favicon_id
        self.frecency = frecency
        self.last_visit_date = last_visit_date

    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        return "" % (self.id, self.title,
                                                 self.url)

#----------------------------------------------------------------------
def loadSession():
    """"""
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

if __name__ == "__main__":
    session = loadSession()
    res = session.query(Places).all()
    print res[1].title 
```

我们对这个例子做了一点改动，将它设置为一个不同的表:“moz_places”。您会注意到，您通过创建类变量在这里设置了列。如果要访问实例的变量，需要在 __init__ 中重新定义它们。如果你不这样做，你将会面临一些非常混乱的问题。__repr__ 是一个“漂亮的打印”方法。当您打印其中一个“位置”对象时，您将得到 __repr__ 返回的任何内容。除此之外，代码与我们看到的其他部分非常相似。

## 包扎

如您所见，使用 SqlAlchemy 连接数据库轻而易举。大多数情况下，您会提前知道数据库的设计是什么，要么通过文档，因为您自己构建了它，要么因为您有一些实用程序可以告诉您。当涉及到在 SqlAlchemy 中定义表时，拥有这些信息是很有帮助的，但是现在即使您事先不知道表的配置，您也知道应该尝试什么了！学 Python 的招数不好玩吗？下次见！

## 进一步阅读

*   [SqlAlchemy 官方网站](http://www.sqlalchemy.org/)
*   [仙丹官方网站](http://elixir.ematia.de/trac/wiki)

## 下载

*   [SqlAlchemy _ existing _ db . tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/09/SqlAlchemy_existing_db.tar)
*   [SqlAlchemy _ existing _ db . zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/09/SqlAlchemy_existing_db.zip)