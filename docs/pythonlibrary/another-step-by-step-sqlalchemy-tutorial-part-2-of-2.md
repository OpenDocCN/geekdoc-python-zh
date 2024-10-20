# 另一个循序渐进的 SqlAlchemy 教程(第 2 部分，共 2 部分)

> 原文：<https://www.blog.pythonlibrary.org/2010/02/03/another-step-by-step-sqlalchemy-tutorial-part-2-of-2/>

在本系列的第一部分中，我们回顾了使用 SqlAlchemy 与数据库交互的“SQL 表达式”方法。这背后的理论是，我们应该在学习更高层次(和更抽象)的方法之前，学习不太抽象的做事方法。这在许多数学课中都是真实的，比如微积分，在你学习捷径之前，你需要学习很长的路去寻找一些计算的标准偏差。

对于后半部分，我们将做一些可能被认为是使用 SqlAlchemy 的简单方法。它被称为“对象关系”方法，[官方文档](http://www.sqlalchemy.org/docs/ormtutorial.html)实际上就是从它开始的。这种方法最初需要花费较长的时间来建立，但在许多方面，它也更容易遵循。

## 习惯数据映射

Robin Munn 的老学校 [SqlAlchemy 教程](http://www.rmunn.com/sqlalchemy-tutorial/tutorial.html)称这个部分为“数据映射”,因为我们将把数据库中的数据映射到 Python 类。我们开始吧！

```py

from sqlalchemy import create_engine
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import mapper, sessionmaker

####################################################
class User(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, name, fullname, password):
        """Constructor"""
        self.name = name
        self.fullname = fullname
        self.password = password

    def __repr__(self):
        return "" % (self.name, self.fullname, self.password)

# create a connection to a sqlite database
# turn echo on to see the auto-generated SQL
engine = create_engine("sqlite:///tutorial.db", echo=True)

# this is used to keep track of tables and their attributes
metadata = MetaData()
users_table = Table('users', metadata,
                    Column('user_id', Integer, primary_key=True),
                    Column('name', String),
                    Column('fullname', String),
                    Column('password', String)
                    )
email_table = Table('email', metadata,
                    Column('email_id', Integer, primary_key=True),
                    Column('email_address', String),
                    Column('user_id', Integer, ForeignKey('users.user_id'))
                    )

# create the table and tell it to create it in the 
# database engine that is passed
metadata.create_all(engine)

# create a mapping between the users_table and the User class
mapper(User, users_table) 
```

与前面的例子相比，要注意的第一个区别是*用户*类。我们对我们的原始示例(参见第一部分)做了一点修改，以匹配官方文档中的内容，即参数现在是 name、full name 和 password。剩下的看起来应该是一样的，直到我们看到*映射器*语句。这个方便的方法允许 SqlAlchemy 将 User 类映射到 users_table。这看起来没什么大不了的，但是这种方法使得向数据库添加用户变得更加简单。

然而，在此之前，我们需要讨论一下[声明性配置风格](http://www.sqlalchemy.org/docs/ormtutorial.html#creating-table-class-and-mapper-all-at-once-declaratively)。虽然上面的样式给了我们对表、映射器和类的细粒度控制，但在大多数情况下，我们不需要它那么复杂。这就是声明式风格的由来。这使得配置一切变得更加容易。我所知道的第一种声明式风格是 SqlAlchemy 的一个插件，名为[药剂](http://elixir.ematia.de/trac/wiki)。这种内置的声明式风格不像 Elixir 那样功能齐全，但是更方便，因为您没有额外的依赖性。让我们看看声明性有什么不同:

```py

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, mapper, relation, sessionmaker

Base = declarative_base()

########################################################################
class User(Base):
    """"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    fullname = Column(String)
    password = Column(String)

    #----------------------------------------------------------------------
    def __init__(self, name, fullname, password):
        """Constructor"""
        self.name = name
        self.fullname = fullname
        self.password = password

    def __repr__(self):
        return "" % (self.name, self.fullname, self.password)

########################################################################
class Address(Base):
    """
    Address Class

    Create some class properties before initilization
    """
    __tablename__ = "addresses"
    id = Column(Integer, primary_key=True)
    email_address = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))

    # creates a bidirectional relationship
    # from Address to User it's Many-to-One
    # from User to Address it's One-to-Many
    user = relation(User, backref=backref('addresses', order_by=id))

    #----------------------------------------------------------------------
    def __init__(self, email_address):
        """Constructor"""
        self.email_address = email_address

    def __repr__(self):
        return "

<address>" % self.email_address

# create a connection to a sqlite database
# turn echo on to see the auto-generated SQL
engine = create_engine("sqlite:///tutorial.db", echo=True)

# get a handle on the table object
users_table = User.__table__
# get a handle on the metadata
metadata = Base.metadata
metadata.create_all(engine)

```

如你所见，现在几乎所有的东西都是在类中创建的。我们创建类属性(类似于类的全局变量)来标识表的列。然后，我们创建与上面的原始类示例中相同的 __init__。同样，我们子类化 *declarative_base* 而不是基本的*对象*。如果我们需要一个表对象，我们必须调用下面的魔法方法 *User。__ 表 _ _*；为了获取元数据，我们需要调用 *Base.metadata* 。涵盖了我们所关心的差异。现在我们可以看看如何向数据库添加数据。

## 课程现在开始

使用对象关系方法与我们的数据库进行交互的好处可以在几个简短的代码片段中展示出来。让我们看看如何创建一行:

```py

mike_user = User("mike", "Mike Driscoll", "password")
print "User name: %s, fullname: %s, password: %s" % (mike_user.name,
                                                     mike_user.fullname,
                                                     mike_user.password)

```

如您所见，我们可以用 User 类创建用户。我们可以使用点符号来访问属性，就像在任何其他 Python 类中一样。我们甚至可以用它们来更新行。例如，如果我们需要更改上面的用户对象，我们将执行以下操作:

```py

# this is how you would change the name field
mike_user.fullname = "Mike Dryskull"

```

请注意，所有这些都不会像我们在第一篇文章中看到的那些插入方法那样自动将行添加到数据库中。相反，我们需要一个会话对象来完成这项工作。让我们浏览一下使用会话的一些基础知识:

```py

from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()

mike_user = User("mike", "Mike Driscoll", "password")
session.add(mike_user)

```

我们在这里暂停一下，解释一下是怎么回事。首先，我们需要从 *sqlalchemy.orm* 中导入 *sessionmaker* ，并将其绑定到引擎(从技术上讲，您可以在不绑定的情况下创建会话，但要做任何有用的事情，您最终需要绑定它)。接下来，我们创建一个会话实例。然后我们实例化一个用户对象，并将其添加到会话中。此时，还没有运行任何 SQL 代码，事务只是挂起。为了持久化这一行，我们需要调用 *session.commit()* 或者运行一个查询。

如果需要添加多个用户，请执行以下操作:

```py

session.add_all([
     User('Mary', 'Mary Wonka', 'foobar'),
     User('Sue', 'Sue Lawhead', 'xxg527'),
     User('Fay', 'Fay Ray', 'blah')])

```

如果您在将用户的属性提交到数据库后碰巧更改了其中一个属性，您可以使用 *session.dirty* 来检查哪个属性被修改了。如果您只需要知道哪些行是未决的，就调用 *session.new* 。最后，我们可以使用 *session.rollback()* 回滚一个事务。

现在让我们来看一些示例查询:

```py

# do a Select all
all_users = session.query(User).all()

# Select just one user by the name of "mike"
our_user = session.query(User).filter_by(name='mike').first()
print our_user

# select users that match "Mary" or "Fay"
users = session.query(User).filter(User.name.in_(['Mary', 'Fay'])).all()
print users

# select all and print out all the results sorted by id
for instance in session.query(User).order_by(User.id): 
    print instance.name, instance.fullname

```

我们不需要逐一讨论，因为它们在评论中都有解释。相反，我们将继续讨论连接的主题。

## 来凑热闹

有些连接使用了 [SQL 表达式语法](http://www.sqlalchemy.org/docs/sqlexpression.html#using-joins)，我不会在这里介绍。相反，我们将使用对象关系方法。如果您回头看一下创建表的开始示例，您会注意到我们已经设置了与*外键*对象的连接。声明格式如下所示:

```py

user_id = Column(Integer, ForeignKey('users.id'))

# creates a bidirectional relationship
# from Address to User it's Many-to-One
# from User to Address it's One-to-Many
user = relation(User, backref=backref('addresses', order_by=id))

```

让我们通过创建一个新用户来看看这是如何工作的:

```py

prof = User("Prof", "Prof. Xavier", "fudge")
prof.addresses

```

由于*外键*和 *backref* 命令，*用户*对象具有*地址*属性。如果您运行该代码，您将看到它是空的。让我们添加一些地址！(注意:一定要将 prof 用户添加到 session: session.add(prof))

```py

prof.addresses = [Address(email_address='profx@dc.com'), 
                        Address(email_address='xavier@yahoo.com')]

```

看到这有多简单了吗？甚至很容易把信息弄回来。例如，如果您只想访问第一个地址，您只需调用 *prof.addresses[0]* 。现在，假设您需要更改其中一个地址(即进行更新)。这很容易:

```py

# change the first address
prof.addresses[0].email_address = "profx@marvel.com"

```

现在让我们继续对连接进行查询:

```py

for u, a in session.query(User, Address).filter(User.id==Address.user_id).filter(Address.email_address=='xavier@yahoo.com').all():
    print u, a

```

这是一个很长的查询！我发现自己很难做到这些，所以我通常会做以下事情来让我的大脑更容易理解:

```py

sql = session.query(User, Address)
sql = sql.filter(User.id==Address.user_id)
sql = sql.filter(Address.email_address=='xavier@yahoo.com')

for u, a in sql.all():
    print u, a

```

现在，对于那些喜欢一行程序的人来说，第一个例子没有任何问题。它会产生完全相同的结果。我只是碰巧发现更长的版本更容易调试。最后，我们还可以使用一个真正的连接:

```py

from sqlalchemy.orm import join
session.query(User).select_from(join(User, Address)).filter(Address.email_address=='xavier@yahoo.com').all()

```

这也与前两个例子一样，但是以更明确的方式。关于使用对象关系语法连接的更多信息，我推荐使用官方文档。

## 包扎

此时，您应该能够用表创建数据库，用数据填充表，以及使用 SqlAlchemy 选择、更新和提交事务到数据库。我希望这篇教程有助于你理解这项神奇的技术。

*注意:本教程是在使用 Python 2.5 和 SqlAlchemy 0.5.8 的 Windows 上测试的。*

**延伸阅读**

*   [对象关系教程](http://www.sqlalchemy.org/docs/ormtutorial.html)
*   [罗宾·穆恩的 SqlAlchemy 教程](http://www.rmunn.com/sqlalchemy-tutorial/tutorial.html)